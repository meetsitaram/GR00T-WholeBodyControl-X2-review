#!/usr/bin/env python3
"""Capture the robot's current joint pose and emit a drop-in
``GANTRY_HANG_OFFSETS_MJ`` constant for the MuJoCo bridge.

Use this when the operator has manually placed the X2 in a physical
pose (e.g. the gantry-hang bent-knee crouch) and we want sim's
``--init-pose=gantry-hang`` to start from the same joint configuration
the real robot is actually in. Subscribes to all four
``/aima/hal/joint/{leg,waist,arm,head}/state`` topics + the chest /
torso IMU, samples for ``--duration`` seconds, and prints:

  * a 31-row table with each joint's measured median position vs the
    trained ``DEFAULT_DOF`` baseline (``policy_parameters.hpp``);
  * mean joint effort per group (sanity check that "torque-free" really
    means torque-free -- if any joint is sustaining > 1 Nm we likely
    have MC fighting the pose);
  * IMU gravity-body + tilt summary;
  * a ready-to-paste Python ``GANTRY_HANG_OFFSETS_MJ`` dict literal that
    contains ONLY the joints whose offset from default exceeds
    ``--dead-deg`` (default 5 deg). Joints that didn't move stay implicit
    (they're zero in the dict) so we don't pollute the constant with
    encoder noise.

Pass ``--write-bridge`` to atomically rewrite the
``GANTRY_HANG_OFFSETS_MJ`` constant inside
``gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py`` in place. The script
keeps a ``.bak`` next to the file so a bad capture is one ``mv`` away from
recovery.

Designed to run inside the docker_x2/ container in real-mode:

  cd gear_sonic_deploy/docker_x2 && \\
      docker compose -f docker-compose.yml -f docker-compose.real.yml \\
          run --rm x2sim bash -c '
              source /opt/ros/humble/setup.bash &&
              source /ros2_ws/install/setup.bash &&
              python3 /workspace/sonic/gear_sonic_deploy/scripts/x2_capture_pose.py
          '

The container has FastDDS + aimdk_msgs ready, ``ROS_DOMAIN_ID=0``, and
``ROS_LOCALHOST_ONLY=0`` from the real-mode overlay -- no extra setup
needed once the host can see the robot on any interface. If the host
has multiple paths to the robot (wifi *and* SDK ethernet), pin a NIC
explicitly via ``RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`` +
``CYCLONEDDS_URI`` at the command line for that run.

Mirrors the patterns in:
  * agitbot-x2-record-and-replay/src/x2_recorder/ros_interface.py
  * agibot-x2-monitor/bridge/bridge/ros_subscriber.py
  * gear_sonic_deploy/scripts/x2_preflight.py
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import sys
import threading
import time
from typing import Optional

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

try:
    from aimdk_msgs.msg import JointStateArray
except ImportError as e:
    print(
        "ERROR: aimdk_msgs not on the Python path. Run this inside the\n"
        "       docker_x2/ container (docker-compose.yml + docker-compose.real.yml)\n"
        "       or source the colcon overlay that built aimdk_msgs.",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

from sensor_msgs.msg import Imu


# ─────────────────────────────────────────────────────────────────────────
# Canonical 31-DOF MJ joint order. Mirrors x2_preflight.py:MUJOCO_JOINT_NAMES
# and policy_parameters.hpp. We match by joint NAME from the published
# ``JointState.name`` field, so the order inside any one /state topic
# doesn't matter -- only the names need to be canonical.
# ─────────────────────────────────────────────────────────────────────────
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_pitch_joint",     "waist_roll_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "left_wrist_yaw_joint",       "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_yaw_joint",      "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint",             "head_pitch_joint",
)
NUM_DOFS = len(MUJOCO_JOINT_NAMES)
NAME_TO_MJ_IDX = {n: i for i, n in enumerate(MUJOCO_JOINT_NAMES)}

# Mirror of DEFAULT_ANGLES in x2_preflight.py / policy_parameters.hpp /
# eval_x2_mujoco.py. Source of truth is the codegen; this duplicate is
# fine because if it ever drifts the printed offsets will be wrong by a
# constant and the operator will notice immediately.
DEFAULT_ANGLES = np.array((
    -0.312, 0.0,   0.0,    0.669, -0.363, 0.0,
    -0.312, 0.0,   0.0,    0.669, -0.363, 0.0,
     0.0,   0.0,   0.0,
     0.2,   0.2,   0.0,   -0.6,   0.0,    0.0,   0.0,
     0.2,  -0.2,   0.0,   -0.6,   0.0,    0.0,   0.0,
     0.0,   0.0,
), dtype=np.float64)
assert DEFAULT_ANGLES.shape == (NUM_DOFS,)

JOINT_GROUPS = (
    ("leg",   "/aima/hal/joint/leg/state",   12),
    ("waist", "/aima/hal/joint/waist/state",  3),
    ("arm",   "/aima/hal/joint/arm/state",   14),
    ("head",  "/aima/hal/joint/head/state",   2),
)


_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


# ─────────────────────────────────────────────────────────────────────────
# Subscriber
# ─────────────────────────────────────────────────────────────────────────
class CaptureNode(Node):
    """Aggregates joint-state samples per (group, joint name) for a window."""

    def __init__(self, imu_topic: str):
        super().__init__("x2_capture_pose")
        self._imu_topic = imu_topic

        self._lock = threading.Lock()
        # samples[joint_name] = list[(position, velocity, effort)]
        self._samples: dict[str, list[tuple[float, float, float]]] = {}
        self._group_msg_counts: dict[str, int] = {g: 0 for g, *_ in JOINT_GROUPS}

        # IMU rolling samples for gravity / tilt summary.
        self._imu_quat: list[tuple[float, float, float, float]] = []
        self._imu_angvel: list[tuple[float, float, float]] = []

        for group_name, topic, _expected_n in JOINT_GROUPS:
            self.create_subscription(
                JointStateArray,
                topic,
                lambda msg, g=group_name: self._on_joint(g, msg),
                _QOS,
            )

        self.create_subscription(Imu, imu_topic, self._on_imu, _QOS)

    def _on_joint(self, group: str, msg: JointStateArray) -> None:
        with self._lock:
            self._group_msg_counts[group] += 1
            for js in msg.joints:
                name = str(js.name)
                # Filter unknown names so we don't accidentally pick up
                # gripper / hand joints that share the topic.
                if name not in NAME_TO_MJ_IDX:
                    continue
                self._samples.setdefault(name, []).append(
                    (float(js.position), float(js.velocity), float(js.effort))
                )

    def _on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        a = msg.angular_velocity
        with self._lock:
            self._imu_quat.append((float(q.w), float(q.x), float(q.y), float(q.z)))
            self._imu_angvel.append((float(a.x), float(a.y), float(a.z)))
            # Cap window so a long capture doesn't eat memory.
            if len(self._imu_quat) > 4096:
                self._imu_quat = self._imu_quat[-2048:]
                self._imu_angvel = self._imu_angvel[-2048:]

    # ── Snapshot accessors ───────────────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "samples": {n: list(s) for n, s in self._samples.items()},
                "group_msg_counts": dict(self._group_msg_counts),
                "imu_quat": list(self._imu_quat),
                "imu_angvel": list(self._imu_angvel),
            }


# ─────────────────────────────────────────────────────────────────────────
# Math helpers (pure functions for unit-testability)
# ─────────────────────────────────────────────────────────────────────────
def gravity_body_from_quat_wxyz(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Return body-frame gravity vector for IMU orientation (WORLD z = -1, in body coords).

    Matches the convention used by aimdk_io.cpp::on_imu and the IsaacLab obs.
    Result is unit-norm (we drop the 9.81 magnitude; only direction matters
    for the upright check).
    """
    # World gravity = [0, 0, -1] in world frame. Rotate it INTO the body
    # frame by R^T (inverse of body->world). For unit quaternion q, R^T
    # corresponds to q with conjugated vector part. Closed-form for
    # vec = [0,0,-1]:
    #   g_body = [-2 * (qw*qy + qx*qz),
    #             -2 * (qw*(-qx) + qy*qz)   = -2 * (qy*qz - qw*qx),
    #             -(qw*qw - qx*qx - qy*qy + qz*qz)]
    # Equivalent to body_frame_gravity_from_quat_wxyz in aimdk_io.cpp.
    gx = -2.0 * (qx * qz - qw * qy)
    gy = -2.0 * (qy * qz + qw * qx)
    gz = -(qw * qw - qx * qx - qy * qy + qz * qz)
    g = np.array([gx, gy, gz], dtype=np.float64)
    n = np.linalg.norm(g)
    return g / n if n > 1e-9 else g


def tilt_deg_from_grav_body_z(grav_z: float) -> float:
    """Angle (deg) between body's -z axis and world gravity. 0 = perfectly upright."""
    return math.degrees(math.acos(max(-1.0, min(1.0, -grav_z))))


# ─────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────
def _summarize_pose(snap: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Returns (median_pos, median_eff, sample_counts, missing_names)."""
    pos = np.full(NUM_DOFS, np.nan, dtype=np.float64)
    eff = np.full(NUM_DOFS, np.nan, dtype=np.float64)
    counts = np.zeros(NUM_DOFS, dtype=np.int64)
    missing: list[str] = []
    for name, mj_idx in NAME_TO_MJ_IDX.items():
        rows = snap["samples"].get(name, [])
        counts[mj_idx] = len(rows)
        if not rows:
            missing.append(name)
            continue
        arr = np.asarray(rows, dtype=np.float64)
        pos[mj_idx] = float(np.median(arr[:, 0]))
        eff[mj_idx] = float(np.median(arr[:, 2]))
    return pos, eff, counts, missing


def _print_pose_table(pos: np.ndarray, eff: np.ndarray, counts: np.ndarray, dead_deg: float) -> None:
    print()
    print("Captured pose (median over the sampling window):")
    print(
        f"  {'idx':>3}  {'joint':<28}  {'meas (rad)':>11}  "
        f"{'meas (deg)':>10}  {'default':>9}  {'offset (deg)':>13}  "
        f"{'effort':>8}  {'n':>5}"
    )
    print("  " + "─" * 99)
    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        if math.isnan(pos[i]):
            print(f"  {i:>3}  {name:<28}  {'-- no samples --':>59}  {0:>5}")
            continue
        offset_rad = pos[i] - DEFAULT_ANGLES[i]
        offset_deg = math.degrees(offset_rad)
        flag = "*" if abs(offset_deg) >= dead_deg else " "
        print(
            f"  {i:>3}  {name:<28}  {pos[i]:>+11.4f}  {math.degrees(pos[i]):>+10.2f}  "
            f"{DEFAULT_ANGLES[i]:>+9.4f}  {offset_deg:>+12.2f}{flag}  "
            f"{eff[i]:>+8.2f}  {counts[i]:>5}"
        )
    print()
    print(f"  '*' = |offset| >= {dead_deg:.1f} deg (will appear in GANTRY_HANG_OFFSETS_MJ)")


def _print_imu_summary(snap: dict) -> None:
    quats = snap["imu_quat"]
    angvel = snap["imu_angvel"]
    if not quats:
        print()
        print("IMU: no samples received.")
        return
    quats_arr = np.asarray(quats, dtype=np.float64)
    angvel_arr = np.asarray(angvel, dtype=np.float64)
    qw, qx, qy, qz = np.median(quats_arr, axis=0)
    g_body = gravity_body_from_quat_wxyz(qw, qx, qy, qz)
    tilt = tilt_deg_from_grav_body_z(g_body[2])
    angvel_max = float(np.max(np.abs(angvel_arr)))
    print()
    print("IMU summary:")
    print(f"  median quat (w,x,y,z) = "
          f"({qw:+.4f}, {qx:+.4f}, {qy:+.4f}, {qz:+.4f})")
    print(f"  gravity_body          = "
          f"({g_body[0]:+.3f}, {g_body[1]:+.3f}, {g_body[2]:+.3f})  "
          f"|tilt|={tilt:.1f} deg from upright")
    print(f"  max |angular vel|     = {angvel_max:.4f} rad/s")
    print(f"  samples (quat / w)    = {len(quats)} / {len(angvel)}")


def _print_per_group_rates(snap: dict, duration: float) -> None:
    print()
    print("Topic activity:")
    for group, topic, n_expected in JOINT_GROUPS:
        msgs = snap["group_msg_counts"].get(group, 0)
        rate = msgs / duration if duration > 0 else 0.0
        ok = "ok" if msgs > 0 else "MISSING"
        print(f"  {group:<5} {topic:<32} {msgs:>5} msgs / {duration:.1f}s "
              f"= {rate:5.1f} Hz   [{ok}]")


def _build_offsets_dict(pos: np.ndarray, dead_deg: float) -> dict[int, float]:
    """Return ``{mj_idx: offset_rad}`` for every joint with |offset| >= dead_deg."""
    out: dict[int, float] = {}
    dead_rad = math.radians(dead_deg)
    for i in range(NUM_DOFS):
        if math.isnan(pos[i]):
            continue
        off = pos[i] - DEFAULT_ANGLES[i]
        if abs(off) >= dead_rad:
            out[i] = float(off)
    return out


def _format_offsets_dict(offsets: dict[int, float], pelvis_z: Optional[float]) -> str:
    """Pretty-print the dict literal as it'd appear in the bridge source."""
    if not offsets:
        body = "GANTRY_HANG_OFFSETS_MJ = {}  # no joints exceeded the dead band"
    else:
        lines = ["GANTRY_HANG_OFFSETS_MJ = {"]
        for mj_idx in sorted(offsets):
            name = MUJOCO_JOINT_NAMES[mj_idx]
            off = offsets[mj_idx]
            comment = f"# {name}: {math.degrees(off):+.2f} deg"
            lines.append(f"    {mj_idx:>2}: {off:+.4f},  {comment}")
        lines.append("}")
        body = "\n".join(lines)
    if pelvis_z is not None:
        body = f"GANTRY_HANG_PELVIS_Z = {pelvis_z:.3f}\n" + body
    return body


# ─────────────────────────────────────────────────────────────────────────
# Optional in-place rewrite of the bridge constants
# ─────────────────────────────────────────────────────────────────────────
_BRIDGE_PATH_DEFAULT = pathlib.Path(__file__).resolve().with_name(
    "x2_mujoco_ros_bridge.py",
)

_OFFSETS_RE = re.compile(
    r"^GANTRY_HANG_OFFSETS_MJ\s*=\s*\{[^}]*\}\s*$",
    re.MULTILINE | re.DOTALL,
)
_PELVIS_RE = re.compile(
    r"^GANTRY_HANG_PELVIS_Z\s*=\s*[-+0-9.eE]+\s*(#[^\n]*)?$",
    re.MULTILINE,
)


def _rewrite_bridge(bridge_path: pathlib.Path,
                    offsets: dict[int, float],
                    pelvis_z: Optional[float]) -> None:
    text = bridge_path.read_text()

    # Build the new offsets block.
    if not offsets:
        new_offsets = "GANTRY_HANG_OFFSETS_MJ = {}"
    else:
        lines = ["GANTRY_HANG_OFFSETS_MJ = {"]
        for mj_idx in sorted(offsets):
            name = MUJOCO_JOINT_NAMES[mj_idx]
            off = offsets[mj_idx]
            lines.append(
                f"    {mj_idx:>2}: {off:+.4f},  # {name}: {math.degrees(off):+.2f} deg"
            )
        lines.append("}")
        new_offsets = "\n".join(lines)

    if not _OFFSETS_RE.search(text):
        raise SystemExit(
            f"ERROR: could not find GANTRY_HANG_OFFSETS_MJ block in {bridge_path}. "
            "Has the constant been renamed? Update the regex in this script."
        )
    new_text = _OFFSETS_RE.sub(new_offsets.replace("\\", "\\\\"), text)

    if pelvis_z is not None:
        if not _PELVIS_RE.search(new_text):
            raise SystemExit(
                f"ERROR: could not find GANTRY_HANG_PELVIS_Z assignment in "
                f"{bridge_path}. Update the regex in this script."
            )
        replacement = (
            f"GANTRY_HANG_PELVIS_Z = {pelvis_z:.3f}  # m, world frame; "
            "captured from real robot via x2_capture_pose.py"
        )
        new_text = _PELVIS_RE.sub(replacement, new_text)

    backup = bridge_path.with_suffix(bridge_path.suffix + ".bak")
    backup.write_text(text)
    bridge_path.write_text(new_text)
    print()
    print(f"Wrote new GANTRY_HANG_* constants to {bridge_path}")
    print(f"  backup of previous content: {backup}")


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(
        description="Capture the X2's current joint pose and emit a "
                    "GANTRY_HANG_OFFSETS_MJ dict for the MuJoCo bridge.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--duration", type=float, default=2.0,
                   help="Sampling window in seconds.")
    p.add_argument("--imu-topic", default="/aima/hal/imu/torso/state",
                   help="IMU topic. Override to /aima/hal/imu/torse/state on "
                        "firmware shipped with the SDK-example typo, or to "
                        "/aima/hal/imu/chest/state to use the chest IMU.")
    p.add_argument("--dead-deg", type=float, default=5.0,
                   help="Joints whose |measured - default| is below this "
                        "many degrees are dropped from the offsets dict (so "
                        "encoder noise on joints the operator did not pose "
                        "doesn't pollute the constant).")
    p.add_argument("--pelvis-z", type=float, default=None,
                   help="If you've measured the pelvis height (m) via the "
                        "gantry, pass it here to also emit / write a new "
                        "GANTRY_HANG_PELVIS_Z constant. We can't measure this "
                        "from joint encoders alone -- it requires the gantry "
                        "or external mocap.")
    p.add_argument("--write-bridge", action="store_true",
                   help="Atomically rewrite GANTRY_HANG_OFFSETS_MJ (and "
                        "optionally GANTRY_HANG_PELVIS_Z) in "
                        "x2_mujoco_ros_bridge.py in place. Saves a .bak.")
    p.add_argument("--bridge-path", type=pathlib.Path,
                   default=_BRIDGE_PATH_DEFAULT,
                   help="Path to x2_mujoco_ros_bridge.py for --write-bridge.")
    p.add_argument("--out-json", type=pathlib.Path, default=None,
                   help="Optional path to also dump the raw capture as JSON "
                        "(for offline analysis).")
    p.add_argument("--require-min-msgs", type=int, default=10,
                   help="Refuse to emit offsets if any joint group received "
                        "fewer than this many messages during the window. "
                        "Set to 0 to disable.")
    args = p.parse_args()

    # ── Spin ROS for the sampling window ─────────────────────────────────
    rclpy.init()
    node: Optional[CaptureNode] = None
    executor = SingleThreadedExecutor()
    try:
        node = CaptureNode(imu_topic=args.imu_topic)
        executor.add_node(node)

        print("Capturing X2 joint pose:")
        print(f"  duration:    {args.duration:.2f} s")
        print(f"  IMU topic:   {args.imu_topic}")
        print(f"  dead band:   {args.dead_deg:.2f} deg")
        if args.pelvis_z is not None:
            print(f"  pelvis_z:    {args.pelvis_z:.3f} m (operator-supplied)")
        print()
        print(f"  ROS_DOMAIN_ID =     "
              f"{__import__('os').environ.get('ROS_DOMAIN_ID', '<unset>')}")
        print(f"  ROS_LOCALHOST_ONLY ="
              f"{__import__('os').environ.get('ROS_LOCALHOST_ONLY', '<unset>')}")
        print(f"  RMW_IMPLEMENTATION ="
              f"{__import__('os').environ.get('RMW_IMPLEMENTATION', '<unset>')}")
        print()
        print("Sampling...", flush=True)

        deadline = time.monotonic() + args.duration
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.05)

        snap = node.snapshot()
    finally:
        if node is not None:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        rclpy.shutdown()

    # ── Report ───────────────────────────────────────────────────────────
    pos, eff, counts, missing = _summarize_pose(snap)
    _print_per_group_rates(snap, args.duration)
    _print_pose_table(pos, eff, counts, args.dead_deg)
    _print_imu_summary(snap)

    # ── Validate ─────────────────────────────────────────────────────────
    fail = False
    for group, _topic, _expected_n in JOINT_GROUPS:
        n = snap["group_msg_counts"].get(group, 0)
        if args.require_min_msgs > 0 and n < args.require_min_msgs:
            print(f"\nERROR: group '{group}' got only {n} messages "
                  f"(need >= {args.require_min_msgs}).")
            fail = True
    if missing:
        print()
        print("WARNING: no samples for these joints (left out of the dict):")
        for n in missing:
            print(f"  - {n}")

    # ── Emit dict ────────────────────────────────────────────────────────
    offsets = _build_offsets_dict(pos, args.dead_deg)
    print()
    print("─" * 72)
    print(_format_offsets_dict(offsets, args.pelvis_z))
    print("─" * 72)

    # ── Optional JSON dump ───────────────────────────────────────────────
    if args.out_json is not None:
        import json
        payload = {
            "duration_s": args.duration,
            "default_angles": DEFAULT_ANGLES.tolist(),
            "joint_names": list(MUJOCO_JOINT_NAMES),
            "median_position_rad": [None if math.isnan(v) else v for v in pos],
            "median_effort_nm":   [None if math.isnan(v) else v for v in eff],
            "sample_counts":      counts.tolist(),
            "group_msg_counts":   snap["group_msg_counts"],
            "imu_topic":          args.imu_topic,
            "imu_quat_median":
                np.median(np.asarray(snap["imu_quat"], dtype=np.float64),
                          axis=0).tolist() if snap["imu_quat"] else None,
            "offsets_mj":         {str(k): v for k, v in offsets.items()},
            "pelvis_z":           args.pelvis_z,
        }
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nWrote raw capture to {args.out_json}")

    # ── Optional in-place rewrite ───────────────────────────────────────
    if args.write_bridge:
        if fail:
            print("\nRefusing to --write-bridge: validation failed (see above).")
            return 1
        _rewrite_bridge(args.bridge_path, offsets, args.pelvis_z)

    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
