#!/usr/bin/env python3
"""MuJoCo <-> ROS 2 bridge for the AgiBot X2 Ultra.

This is the X2 sibling of ``gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py``
(the G1 bridge). It lets the C++ deploy node ``agi_x2_deploy_onnx_ref`` run
in **closed-loop** against MuJoCo without any real robot:

    [MuJoCo physics @ 1 kHz]
        |---- joint state ---->  /aima/hal/joint/{leg,waist,arm,head}/state  ----+
        |---- IMU ------------>  /aima/hal/imu/torso/state                       |
        |                                                                        |
        +<---- joint command --  /aima/hal/joint/{leg,waist,arm,head}/command <--+
                                                                                 |
                                                                       [C++ deploy]

Topology (matches `aimdk_io.{hpp,cpp}`):
  - 4 publishers, one per joint group (leg=12, waist=3, arm=14, head=2),
    `aimdk_msgs/JointStateArray`, joint **names** + position + velocity
    in **MuJoCo joint order** (per group). The deploy will FATAL if the
    names don't match `mujoco_joint_names[start..start+len)`.
  - 1 publisher for `/aima/hal/imu/torso/state` (`sensor_msgs/Imu`).
    Despite the topic name we publish PELVIS-frame quantities, because
    the MJCF only has live IMU sensors at the pelvis (`imu_0` site) and
    that's what training implicitly trained against. Override with
    `--imu-from torso` to publish from torso_link if you want to mimic the
    real sensor mounting (sim2real residual).
  - 4 subscribers for `/aima/hal/joint/{leg,waist,arm,head}/command`
    (`aimdk_msgs/JointCommandArray`). Each callback updates a per-joint
    PD setpoint table; the physics loop applies
        tau = stiffness * (target_pos - q) + damping * (target_vel - dq) + effort_ff
    every step, clipped to the actuator's `actuatorfrcrange`.

QoS: `rclpy.qos.qos_profile_sensor_data` (BEST_EFFORT + VOLATILE +
KEEP_LAST(10)) — exactly what `aimdk_io.cpp` uses for both subs and pubs,
so the handshake matches with no warnings.

Defaults to ``ROS_LOCALHOST_ONLY=1`` and a quirky ``ROS_DOMAIN_ID`` so a
sim run on the SDK ethernet can't fight a real robot. Override via env.

The constants (joint names, IL/MJ remap, KP/KD, default standing pose)
are imported from ``gear_sonic/scripts/eval_x2_mujoco.py`` -- the same
module that's the source of truth for ``policy_parameters.hpp``. This
makes the bridge auto-update whenever `eval_x2_mujoco` changes; if the
two ever drift, the deploy's name-mismatch FATAL is the loud failure.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import pathlib
import sys
import threading
import time
from typing import Optional

import numpy as np
import scipy.spatial.transform

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


class ElasticBand:
    """Virtual spring that hangs the robot's base body from world ``[0,0,1]``.

    Verbatim port of ``gear_sonic/utils/mujoco_sim/unitree_sdk2py_bridge.py:ElasticBand``
    (the G1 sim's anti-faceplant device). We do NOT import that module directly
    because it pulls in ``unitree_sdk2py`` at top level, which is not installed
    in the X2 deploy container. Keep the two copies in lockstep -- if you tune
    one, tune the other.

    Usage matches ``base_sim.py``:
      - viewer mode: register ``MujuocoKeyCallback`` so key 9 toggles, 7/8 raise
        / lower the suspension height
      - per sim step: write ``mj_data.xfrc_applied[band_attached_link] =
        band.Advance(pose)`` when ``band.enable`` is True (else zero it out)

    The pose layout passed to ``Advance`` is the one ``base_sim.py`` builds:
        [pos(3), quat_wxyz(4), lin_vel_world(3), ang_vel_world(3)]
    where the velocities come out of ``mj_objectVelocity(... flg_local=0)`` and
    the angular/linear halves are SWAPPED to put linear first (MuJoCo's
    objectVelocity returns angular-first).
    """

    def __init__(self):
        self.kp_pos = 10000.0
        self.kd_pos = 1000.0
        self.kp_ang = 1000.0
        self.kd_ang = 10.0
        self.point = np.array([0.0, 0.0, 1.0])
        self.length = 0.0
        self.enable = True

    def Advance(self, pose: np.ndarray) -> np.ndarray:
        pos     = pose[0:3]
        quat    = pose[3:7]   # wxyz
        lin_vel = pose[7:10]
        ang_vel = pose[10:13]

        dx = self.point - pos
        f = (self.kp_pos * (dx + np.array([0.0, 0.0, self.length]))
             + self.kd_pos * (0.0 - lin_vel))

        quat_xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])
        rot = scipy.spatial.transform.Rotation.from_quat(quat_xyzw)
        rotvec = rot.as_rotvec()
        torque = -self.kp_ang * rotvec - self.kd_ang * ang_vel

        return np.concatenate([f, torque])

    def MujuocoKeyCallback(self, key):
        # Only import glfw if the viewer actually drives a key event;
        # headless runs never trigger this path.
        import glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        elif key == glfw.KEY_8:
            self.length += 0.1
        elif key == glfw.KEY_9:
            self.enable = not self.enable
            print(f"[bridge] ElasticBand enable: {self.enable}", flush=True)


def _load_eval_x2():
    """Import ``eval_x2_mujoco`` for the canonical X2 constants (no side effects).

    Mirrors the trick used by ``codegen_x2_policy_parameters.py`` -- importing
    by file path so we don't need ``gear_sonic`` on PYTHONPATH.
    """
    p = REPO_ROOT / "gear_sonic" / "scripts" / "eval_x2_mujoco.py"
    if not p.is_file():
        raise FileNotFoundError(f"eval_x2_mujoco.py not found at {p}")
    spec = importlib.util.spec_from_file_location("eval_x2_mujoco", p)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build import spec for {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["eval_x2_mujoco"] = mod
    spec.loader.exec_module(mod)
    return mod


# Per-group ranges in MUJOCO_JOINT_NAMES order; match
# `kLegStart/kWaistStart/kArmStart/kHeadStart` in `aimdk_io.hpp`.
GROUPS = [
    ("leg",   "/aima/hal/joint/leg",    0, 12),
    ("waist", "/aima/hal/joint/waist", 12,  3),
    ("arm",   "/aima/hal/joint/arm",   15, 14),
    ("head",  "/aima/hal/joint/head",  29,  2),
]


# ----------------------------------------------------------------------------
# Init-pose YAML loader
# ----------------------------------------------------------------------------
# The bridge supports a set of named "init poses" -- the joint-space and
# floating-base configuration the simulated robot starts in before the
# deploy node connects. ``--init-pose default`` is hard-coded inside the
# bridge (DEFAULT_DOF + pelvis_z=0.85 with the band engaged); every other
# named pose is loaded from
#
#   gear_sonic_deploy/config/sim_init_poses.yaml
#
# That YAML maps a name (e.g. ``gantry_hang``, ``gantry_dangle``) to a
# pelvis_z, an ElasticBand suspension length, and a per-joint offset
# table from DEFAULT_DOF. Operators edit the YAML to add or tune poses
# without touching this file. See ``sim_init_poses.yaml`` itself for the
# schema.
#
# Numbers in the shipped YAML were captured live from the real robot via
# ``gear_sonic_deploy/scripts/x2_capture_pose.py`` and post-processed
# (symmetrize L/R, drop noise) so sim mirrors a defensible operating
# point. The original captures are kept under ``scripts/_capture_*.json``.
_INIT_POSES_YAML_PATH = pathlib.Path(__file__).resolve().parents[1] / \
    "config" / "sim_init_poses.yaml"


def _load_named_init_poses(path: pathlib.Path = _INIT_POSES_YAML_PATH) -> dict:
    """Read the init-poses YAML; return ``{}`` if it's missing.

    The bridge falls back to the hard-coded ``default`` pose when the
    YAML is absent or empty, so a bad / missing config never bricks
    sim startup. Bad entries (missing required keys, joint name typos)
    raise loudly at first use, since silently dropping them would mask
    real bugs.
    """
    if not path.is_file():
        print(f"[bridge] init-pose YAML not found at {path}; only "
              f"--init-pose=default will be available.", file=sys.stderr)
        return {}
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            f"PyYAML is required to load {path}. Install python3-yaml or "
            "use --init-pose=default."
        ) from e
    with path.open() as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping of pose-name -> entry."
        )
    return raw


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Run with viewer, robot held in default standing pose until deploy connects:
  python gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py --viewer

  # Headless, RSI from a motion frame:
  python gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py \\
      --motion gear_sonic/data/motions/x2_ultra_standing_only.pkl \\
      --init-frame 0
""",
    )
    p.add_argument("--mjcf", type=pathlib.Path, default=None,
                   help="MJCF path (default: gear_sonic/data/assets/.../x2_ultra.xml)")
    p.add_argument("--motion", type=pathlib.Path, default=None,
                   help="Optional motion source for Reference State Initialization (RSI). "
                        "Accepts a motion-lib .pkl or a warehouse playlist .yaml/.yml "
                        "(resolved via the same loaders eval_x2_mujoco_onnx.py uses). "
                        "Mutually exclusive with --init-pose=gantry-hang. "
                        "If omitted, the robot starts in --init-pose (default: 'default').")
    p.add_argument("--init-frame", type=int, default=0,
                   help="Motion frame to RSI from (default 0).")
    p.add_argument("--init-pose", default="default",
                   help="Initial pose when --motion is NOT set. 'default' is "
                        "the legacy hard-coded behaviour (DEFAULT_DOF at "
                        "pelvis_z=0.85m, band fully engaged). Any other name "
                        "is looked up in config/sim_init_poses.yaml -- ship a "
                        "new pose by editing that file, no code changes "
                        "needed. The shipped YAML defines 'gantry_hang' "
                        "(MC-stand handoff pose, captured from the real X2) "
                        "and 'gantry_dangle' (zero-torque hanging pose, "
                        "captured from the real X2). Hyphens and underscores "
                        "are equivalent in pose names.")
    p.add_argument("--band-length", type=float, default=0.0,
                   help="ElasticBand suspension length (m). The band's anchor "
                        "is at world [0,0,1.0]; --band-length offsets the pull "
                        "target downward to [0,0,1.0-LENGTH]. Default 0.0 "
                        "(target at z=1.0, lifts pelvis above standing). For "
                        "the gantry-hang profile use ~0.22 to put the pull "
                        "target ~3 cm above the bent pelvis (~88 %% body "
                        "weight supported, ~12 %% on feet). Viewer keys 7/8 "
                        "still tweak at runtime.")
    p.add_argument("--band-kp-mult", type=float, default=1.0,
                   help="Multiplier on the ElasticBand's kp_pos / kd_pos. "
                        "1.0 = current 10000 / 1000 (stiff, near critically "
                        "damped at default mass). Lower values (0.3-0.5) make "
                        "the band softer / more forgiving so the body bobs "
                        "naturally on the legs instead of being yanked rigidly "
                        "around its anchor. kd_pos is scaled by sqrt(mult) to "
                        "preserve the damping ratio.")
    p.add_argument("--viewer", action="store_true",
                   help="Open the MuJoCo passive viewer window.")
    p.add_argument("--sim-dt", type=float, default=0.001,
                   help="Physics step (s). Default 0.001 (1 kHz).")
    p.add_argument("--state-rate-hz", type=float, default=200.0,
                   help="Joint-state publish rate (default 200 Hz, matches firmware).")
    p.add_argument("--imu-rate-hz", type=float, default=500.0,
                   help="IMU publish rate (default 500 Hz, matches firmware).")
    p.add_argument("--imu-from", choices=("pelvis", "torso"), default="pelvis",
                   help="Which body to read IMU quantities from. 'pelvis' matches "
                        "what training reads off `imu_0`; 'torso' uses torso_link "
                        "(closer to the real sensor mounting, but introduces a "
                        "kinematic offset the policy never saw).")
    p.add_argument("--fixed-base", action="store_true",
                   help="Pin the floating base (skips gravity / falls). Useful for "
                        "verifying joint command paths without dynamics. Note: many "
                        "X2 MJCFs ship with a floating base joint and pinning at "
                        "runtime is not supported -- this flag will warn if so.")
    p.add_argument("--hold-stiffness-mult", type=float, default=1.0,
                   help="Multiplier on policy_parameters.kps used to hold the default "
                        "pose BEFORE the first deploy command arrives. 1.0 = same as "
                        "policy gains; >1 makes the standby hold stiffer.")
    p.add_argument("--no-elastic-band", action="store_true",
                   help="Disable the virtual ElasticBand (hangs the robot's pelvis "
                        "from world [0,0,1] with stiff PD). The band is on by default "
                        "so the robot stays upright while the deploy warms up; press "
                        "key 9 in the viewer to drop, or use --band-release-after-s.")
    p.add_argument("--band-release-after-s", type=float, default=1.0,
                   help="When the band is on AND we're headless (no viewer to press "
                        "key 9), auto-disable the band this many seconds AFTER the "
                        "first deploy command arrives. Default 1.0s -- gives the "
                        "deploy's PD/control loop one tick to settle before gravity "
                        "kicks in. Use a negative value to keep the band on forever "
                        "(unsafe headless; the robot will never touch the ground).")
    p.add_argument("--no-localhost-only", action="store_true",
                   help="Don't force ROS_LOCALHOST_ONLY=1. Use only if you intentionally "
                        "want the bridge to be reachable across the SDK ethernet.")
    p.add_argument("--ros-domain-id", type=int, default=73,
                   help="ROS_DOMAIN_ID to use unless already set in env (default 73).")
    p.add_argument("--print-scene", action="store_true",
                   help="Dump the MJCF body/joint/actuator/sensor table on startup.")
    p.add_argument("--verbose", action="store_true", help="Per-tick debug logging.")
    return p


class X2MujocoRosBridge:
    """Owns the MuJoCo model + a single rclpy node + a sim thread."""

    def __init__(self, args: argparse.Namespace):
        # Lazy / late-bound imports: keep --help cheap, and let the user see
        # a clean error if rclpy or the MuJoCo wheels aren't installed yet.
        global mujoco, mujoco_viewer, rclpy, JointStateArray, JointCommandArray, ImuMsg
        import mujoco  # noqa: F811
        import mujoco.viewer as mujoco_viewer  # noqa: F811
        import rclpy  # noqa: F811
        from rclpy.qos import qos_profile_sensor_data  # noqa: F401  (used in build)
        from aimdk_msgs.msg import JointCommandArray, JointStateArray  # noqa: F811
        from sensor_msgs.msg import Imu as ImuMsg  # noqa: F811

        self.args = args
        self.eval_x2 = _load_eval_x2()

        # --- DDS env isolation (do this BEFORE rclpy.init in main) ---
        # main() handles the env vars; we just stash them for the banner.
        self._ros_localhost_only = os.environ.get("ROS_LOCALHOST_ONLY", "0")
        self._ros_domain_id = os.environ.get("ROS_DOMAIN_ID", "0")

        # --- MuJoCo ---
        mjcf_path = args.mjcf or pathlib.Path(self.eval_x2.MJCF_PATH)
        if not mjcf_path.is_file():
            raise FileNotFoundError(f"MJCF not found: {mjcf_path}")
        self.mjcf_path = mjcf_path

        self.mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = float(args.sim_dt)

        # Resolve per-joint qpos / qvel / actuator indices from joint NAMES,
        # so we don't have to trust the MJCF actuator order. Falls back to
        # eval_x2's JOINT_TO_ACTUATOR if name lookup fails (older MJCFs).
        self.qpos_adr = np.zeros(self.eval_x2.NUM_DOFS, dtype=np.int64)
        self.qvel_adr = np.zeros(self.eval_x2.NUM_DOFS, dtype=np.int64)
        self.act_idx  = np.zeros(self.eval_x2.NUM_DOFS, dtype=np.int64)
        for i, jname in enumerate(self.eval_x2.MUJOCO_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"MJCF missing joint '{jname}'")
            self.qpos_adr[i] = self.mj_model.jnt_qposadr[jid]
            self.qvel_adr[i] = self.mj_model.jnt_dofadr[jid]
            aname = f"motor_{jname}"
            aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid >= 0:
                self.act_idx[i] = aid
            else:
                # Fall back to eval_x2's hand-rolled map (warn once).
                self.act_idx[i] = self.eval_x2.JOINT_TO_ACTUATOR[i]

        # Effort limits from the MJCF actuator force range (per actuator).
        # Use the wider of |min|/|max| in case it's asymmetric.
        force_range = self.mj_model.actuator_forcerange[self.act_idx]
        self.tau_limit = np.maximum(np.abs(force_range[:, 0]), np.abs(force_range[:, 1]))
        # Some MJCFs publish 0/0 if the actuator has no force range; guard.
        self.tau_limit = np.where(self.tau_limit > 0.0, self.tau_limit, 1e6)

        # --- Floating base detection (matches base_sim.py logic) ---
        joint_names = [self.mj_model.joint(i).name for i in range(self.mj_model.njnt)]
        self.has_floating_base = any(
            self.mj_model.joint(i).type == mujoco.mjtJoint.mjJNT_FREE
            for i in range(self.mj_model.njnt)
        )
        if args.fixed_base and self.has_floating_base:
            print("[bridge] WARNING: --fixed-base requested but MJCF has a free joint; "
                  "ignoring (would need MJCF surgery to honour).", file=sys.stderr)

        # Resolve the body that owns the free joint (i.e. the floating base).
        # We need this to decide whether ``_publish_imu`` can short-circuit
        # to ``qvel[3:6]`` for bit-exact ang-vel parity with Python eval --
        # see the comment in ``_publish_imu``.
        self._free_base_body_id = -1
        if self.has_floating_base:
            for jid in range(self.mj_model.njnt):
                if self.mj_model.joint(jid).type == mujoco.mjtJoint.mjJNT_FREE:
                    self._free_base_body_id = int(self.mj_model.jnt_bodyid[jid])
                    break

        # --- IMU body ---
        if args.imu_from == "pelvis":
            self.imu_body = "pelvis"
        else:
            self.imu_body = "torso_link"
        try:
            self.imu_body_id = self.mj_model.body(self.imu_body).id
        except KeyError as e:
            raise RuntimeError(f"MJCF missing body '{self.imu_body}' for IMU readout") from e

        # --- Standby PD setpoint table (per joint, MJ order) ---
        # Before any deploy command arrives, hold the default standing pose
        # with policy_parameters' kps/kds * hold_stiffness_mult so the robot
        # doesn't immediately crumple under gravity.
        n = self.eval_x2.NUM_DOFS
        self.cmd_lock = threading.Lock()
        self._target_pos = self.eval_x2.DEFAULT_DOF.astype(np.float64).copy()
        self._target_vel = np.zeros(n, dtype=np.float64)
        self._effort_ff  = np.zeros(n, dtype=np.float64)
        self._kp = self.eval_x2.KP.astype(np.float64).copy() * float(args.hold_stiffness_mult)
        self._kd = self.eval_x2.KD.astype(np.float64).copy() * float(args.hold_stiffness_mult)
        self._first_command_received = False
        self._first_command_mono = 0.0  # set on first deploy command arrival

        # --- ElasticBand (G1-style "hang the robot" suspension) ---
        # Without this the robot tips over while the deploy spins up its
        # control loop and the C++ tilt watchdog instantly trips.
        # Disabled when (a) user passed --no-elastic-band or (b) the MJCF has
        # no floating base to push against (forces would be silently lost).
        self.elastic_band: Optional[ElasticBand] = None
        self.band_attached_link: int = -1
        if not args.no_elastic_band and self.has_floating_base:
            self.elastic_band = ElasticBand()
            # Initial value from the CLI; ``_reset_to_named_pose`` may
            # override this from the YAML entry's ``band_length`` if the
            # operator left ``--band-length`` at its default (0). With
            # length=0 and a low pelvis (e.g. z=0.6), the band would yank
            # the body upward with hundreds of N -- exactly the surprise
            # the YAML override engineers around.
            self.elastic_band.length = float(args.band_length)
            if args.band_kp_mult != 1.0:
                self.elastic_band.kp_pos *= float(args.band_kp_mult)
                # Scale damping by sqrt(mult) to preserve the damping ratio
                # ``zeta = kd / (2 * sqrt(kp * m))`` as kp scales.
                self.elastic_band.kd_pos *= float(np.sqrt(args.band_kp_mult))
            # X2's pelvis is the natural attach point (matches base_sim.py's
            # ``elif "g1" ... waist`` branch -- in our MJCF the waist body IS
            # the pelvis-mounted free body).
            try:
                self.band_attached_link = int(self.mj_model.body("pelvis").id)
            except KeyError as e:
                raise RuntimeError(
                    "ElasticBand: MJCF has no body named 'pelvis' -- pass "
                    "--no-elastic-band or extend this code to pick another."
                ) from e
        elif not args.no_elastic_band and not self.has_floating_base:
            print("[bridge] ElasticBand requested but MJCF has no floating "
                  "base; band disabled.", file=sys.stderr)

        # --- ROS 2 ---
        self.node = rclpy.create_node("x2_mujoco_ros_bridge")
        from rclpy.qos import qos_profile_sensor_data  # local; module-level alias may be missing
        self.qos = qos_profile_sensor_data

        # Pre-build per-group joint name lists (avoid per-publish python work).
        self._group_names = []
        for grp_name, topic_prefix, start, length in GROUPS:
            names = list(self.eval_x2.MUJOCO_JOINT_NAMES[start : start + length])
            self._group_names.append((grp_name, topic_prefix, start, length, names))

        # Joint-state publishers
        self._state_pubs = []
        for grp_name, topic_prefix, _start, _length, _names in self._group_names:
            self._state_pubs.append(
                self.node.create_publisher(JointStateArray, f"{topic_prefix}/state", self.qos)
            )

        # IMU publisher
        self._imu_pub = self.node.create_publisher(
            ImuMsg, "/aima/hal/imu/torso/state", self.qos
        )

        # Joint-command subscribers (one per group)
        self._cmd_subs = []
        for grp_name, topic_prefix, start, length, names in self._group_names:
            cb = self._make_cmd_callback(grp_name, start, length, names)
            self._cmd_subs.append(
                self.node.create_subscription(
                    JointCommandArray, f"{topic_prefix}/command", cb, self.qos
                )
            )

        # --- Initial pose ---
        # Hyphens and underscores are equivalent (so --init-pose=gantry-hang
        # and --init-pose=gantry_hang both resolve the same way).
        normalized_init_pose = args.init_pose.replace("-", "_")
        if args.motion and normalized_init_pose != "default":
            raise ValueError(
                f"Cannot combine --motion (RSI from motion frame) with "
                f"--init-pose={args.init_pose!r}. RSI puts the robot at the "
                "motion's frame 0 pose; named init poses put it in a fixed "
                "configuration. Pick one. (For sim profiles built around a "
                "named pose, no --motion is needed; the deploy node still "
                "loads its --motion arg as the *reference* the policy "
                "tracks; the bridge's init pose is independent.)"
            )
        if args.motion:
            self._rsi_from_motion(args.motion, args.init_frame)
        elif normalized_init_pose == "default":
            self._reset_to_default_pose()
        else:
            self._reset_to_named_pose(normalized_init_pose)

        if args.print_scene:
            self._print_scene()

        # --- Counters ---
        self._sim_step_count = 0
        self._cmd_count = 0
        self._state_pub_count = 0
        self._imu_pub_count = 0

        self._stop = threading.Event()
        self.viewer = None

    # ----------------------------------------------------------------
    # Command ingestion
    # ----------------------------------------------------------------
    def _make_cmd_callback(self, grp_name, start, length, expected_names):
        """Return a closure that ingests one group's JointCommandArray."""
        validated = [False]
        def cb(msg):
            if not validated[0]:
                if len(msg.joints) != length:
                    self.node.get_logger().fatal(
                        f"[bridge:{grp_name}] command has {len(msg.joints)} "
                        f"joints, expected {length}; refusing to ingest."
                    )
                    return
                for i, j in enumerate(msg.joints):
                    if j.name and j.name != expected_names[i]:
                        self.node.get_logger().fatal(
                            f"[bridge:{grp_name}] command slot {i} is "
                            f"'{j.name}', expected '{expected_names[i]}'."
                        )
                        return
                validated[0] = True
                self.node.get_logger().info(
                    f"[bridge:{grp_name}] joint name validation OK "
                    f"({length} joints)."
                )
            with self.cmd_lock:
                for i, j in enumerate(msg.joints):
                    mj = start + i
                    self._target_pos[mj] = float(j.position)
                    self._target_vel[mj] = float(j.velocity)
                    self._effort_ff[mj]  = float(j.effort)
                    self._kp[mj]         = float(j.stiffness)
                    self._kd[mj]         = float(j.damping)
                if not self._first_command_received:
                    self._first_command_received = True
                    self._first_command_mono = time.monotonic()
                    self.node.get_logger().info(
                        f"[bridge] first deploy command received on group '{grp_name}' "
                        "-- handing PD over to the deploy."
                    )
                self._cmd_count += 1
        return cb

    # ----------------------------------------------------------------
    # State publication
    # ----------------------------------------------------------------
    def _publish_joint_states(self):
        from aimdk_msgs.msg import JointState, JointStateArray
        now = self.node.get_clock().now().to_msg()
        for (grp_name, topic_prefix, start, length, names), pub in zip(
            self._group_names, self._state_pubs
        ):
            msg = JointStateArray()
            msg.header.stamp = now
            msg.header.frame_id = grp_name
            msg.header.sequence = self._state_pub_count & 0xFFFFFFFF
            msg.header.meas_stamp = now
            for i in range(length):
                mj = start + i
                js = JointState()
                js.name = names[i]
                js.position = float(self.mj_data.qpos[self.qpos_adr[mj]])
                js.velocity = float(self.mj_data.qvel[self.qvel_adr[mj]])
                js.effort = float(self.mj_data.actuator_force[self.act_idx[mj]])
                js.coil_temp = 0
                js.motor_temp = 0
                js.motor_vol = 0
                msg.joints.append(js)
            pub.publish(msg)
        self._state_pub_count += 1

    def _publish_imu(self):
        body = self.mj_data.body(self.imu_body_id)
        # MuJoCo body xquat is wxyz; sensor_msgs/Imu wants xyzw
        wxyz = body.xquat
        msg = ImuMsg()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self.imu_body
        msg.orientation.w = float(wxyz[0])
        msg.orientation.x = float(wxyz[1])
        msg.orientation.y = float(wxyz[2])
        msg.orientation.z = float(wxyz[3])
        # ---- Angular velocity (body-local frame) ----------------------------
        # Parity gotcha: ``mj_objectVelocity(..., flg_local=1)`` for a body
        # returns the angular component of ``mj_data.cvel``, which is the
        # spatial velocity computed at the body's CoM expressed in body
        # frame. After a manual ``qvel`` write + ``mj_forward`` it does NOT
        # round-trip exactly to ``qvel[3:6]`` -- we observed a 0.02 rad/s
        # drift on the parity gate vs ``eval_x2_mujoco{,_onnx}.py``, and the
        # policy reacts visibly to a 0.02 rad/s OOD ang-vel input. Both the
        # eval scripts AND IsaacLab's training observation read
        # ``qvel[3:6]`` directly (it is the BODY-LOCAL angular velocity of
        # the free-joint root body by MuJoCo convention -- see Gotcha G1 in
        # docs/source/user_guide/sim2sim_mujoco.md). To stay bit-exact with
        # Python eval and the trained policy, publish ``qvel[3:6]`` instead
        # of ``mj_objectVelocity`` when the IMU is on the floating-base
        # body (the "pelvis" default). For ``--imu-from torso`` we don't
        # have a clean qvel slice -- the policy isn't trained on a torso
        # IMU anyway, so the legacy mj_objectVelocity path is fine there.
        if self.imu_body_id == self._free_base_body_id:
            qvel_ang = self.mj_data.qvel[3:6]
            msg.angular_velocity.x = float(qvel_ang[0])
            msg.angular_velocity.y = float(qvel_ang[1])
            msg.angular_velocity.z = float(qvel_ang[2])
        else:
            ang = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY,
                self.imu_body_id, ang, 1,  # 1 = body local frame
            )
            msg.angular_velocity.x = float(ang[0])
            msg.angular_velocity.y = float(ang[1])
            msg.angular_velocity.z = float(ang[2])
        # Linear acceleration: derive from qacc projected to body frame would be
        # ideal; for now we send root linear acceleration in world frame which
        # is what an idealised IMU after gravity removal would not match. The
        # deploy doesn't read linear_acceleration today, so leave at zero (the
        # field is required to be present but the deploy doesn't consume it).
        msg.linear_acceleration.x = 0.0
        msg.linear_acceleration.y = 0.0
        msg.linear_acceleration.z = 0.0
        # Covariances unknown -> set first element to -1.0 per REP-145 to mark
        # the orientation/ang vel/lin accel covariance as unknown.
        msg.orientation_covariance[0] = -1.0
        msg.angular_velocity_covariance[0] = 0.0
        msg.linear_acceleration_covariance[0] = -1.0
        self._imu_pub.publish(msg)
        self._imu_pub_count += 1

    # ----------------------------------------------------------------
    # PD step
    # ----------------------------------------------------------------
    def _apply_pd(self):
        # Snapshot setpoints under the lock; release before the np math.
        with self.cmd_lock:
            target_pos = self._target_pos.copy()
            target_vel = self._target_vel.copy()
            effort_ff  = self._effort_ff.copy()
            kp         = self._kp.copy()
            kd         = self._kd.copy()

        q  = self.mj_data.qpos[self.qpos_adr]
        dq = self.mj_data.qvel[self.qvel_adr]
        tau = effort_ff + kp * (target_pos - q) + kd * (target_vel - dq)
        tau = np.clip(tau, -self.tau_limit, self.tau_limit)

        # Write into ctrl[] using the per-joint actuator index.
        self.mj_data.ctrl[self.act_idx] = tau

    # ----------------------------------------------------------------
    # ElasticBand
    # ----------------------------------------------------------------
    def _apply_elastic_band(self):
        """Apply the virtual suspension force to the pelvis (if armed).

        Mirrors the 6-line block in ``gear_sonic/utils/mujoco_sim/base_sim.py``
        verbatim: build the [pos, quat, lin_vel, ang_vel] pose, run it through
        ``ElasticBand.Advance``, and write to ``mj_data.xfrc_applied``. Auto-
        release: once the deploy has been driving for
        ``--band-release-after-s`` seconds we flip ``band.enable`` off so the
        robot drops onto its feet and the policy runs against gravity for real.
        Operators who want to hold the body in mid-air for camera work pass
        ``--band-release-after-s -1`` (never release); key 9 in the viewer
        also toggles the band on/off interactively.
        """
        if self.elastic_band is None:
            return
        # Auto-release applies in both headless and viewer modes -- when an
        # automated profile (e.g. --sim-profile handoff) schedules a release,
        # the viewer should faithfully show the body dropping when it would
        # in real life. Holding the band on forever is opt-in via -1.
        if (
            self.elastic_band.enable
            and self._first_command_received
            and self.args.band_release_after_s >= 0.0
            and (time.monotonic() - self._first_command_mono)
                > float(self.args.band_release_after_s)
        ):
            self.elastic_band.enable = False
            self.node.get_logger().info(
                f"[bridge] ElasticBand auto-released "
                f"({self.args.band_release_after_s:.2f}s after first deploy "
                "command) -- robot is now under gravity."
            )

        link = self.band_attached_link
        if self.elastic_band.enable:
            pose = np.concatenate([
                self.mj_data.xpos[link],
                self.mj_data.xquat[link],
                np.zeros(6),
            ])
            # Pull world-frame [angular(3), linear(3)] velocities of the body.
            mujoco.mj_objectVelocity(
                self.mj_model, self.mj_data, mujoco.mjtObj.mjOBJ_BODY,
                link, pose[7:13], 0,
            )
            # objectVelocity returns angular-first; ElasticBand wants linear-first.
            pose[7:10], pose[10:13] = pose[10:13].copy(), pose[7:10].copy()
            self.mj_data.xfrc_applied[link] = self.elastic_band.Advance(pose)
        else:
            self.mj_data.xfrc_applied[link] = np.zeros(6)

    # ----------------------------------------------------------------
    # Sim loop
    # ----------------------------------------------------------------
    def _sim_step_once(self, state_period_steps: int, imu_period_steps: int,
                       viewer=None):
        """One physics tick + maybe-publish. Caller owns the cadence/loop.

        When ``viewer`` is non-None we hold ``viewer.lock()`` across the
        ``mj_step`` write so the viewer's render thread can't read a
        half-mutated ``mj_data`` (segfaults out of GLFW otherwise -- this is
        the documented MuJoCo passive-viewer pattern).

        Pre-handoff freeze
        ------------------
        Before the first deploy command arrives we DO NOT call ``mj_step``.
        State and IMU publishers still fire on schedule (the deploy needs
        fresh state to advance INIT -> WAIT_FOR_CONTROL -> CONTROL), but
        the body is held bit-exact at its RSI / init-pose configuration.

        Why: the deploy's INIT and (default 5 s) WAIT phases together can
        easily last hundreds of ms before the policy sends its first
        command. If the sim integrated the entire time, gravity, ground
        contact, and any non-zero joint vel from the RSI'd motion frame
        would silently evolve the body off its initial state. The deploy's
        first-tick observation would then NOT match Python eval's
        observation of the same frame -- making sim-to-sim parity a moving
        target. Freezing makes the bridge match ``eval_x2_mujoco_onnx.py``
        exactly: the policy's first inference sees the bit-exact RSI
        state.

        On hardware this is also what actually happens: HAL streams joint
        states from MC, but joints don't physically move until the deploy
        starts driving them. Mirroring that in sim is more honest than
        letting the bridge run free.

        The freeze breaks at the moment the bridge's command callback
        flips ``self._first_command_received`` (see ``_make_cmd_callback``).
        From that tick onward, mj_step runs at full rate, the elastic band
        auto-release schedule starts ticking against ``_first_command_mono``,
        and the sim is identical to its prior behaviour.
        """
        if not self._first_command_received:
            # Frozen-body publish: state stays at RSI values, the deploy
            # gets fresh timestamps so its freshness check still clears.
            self._sim_step_count += 1
            if self._sim_step_count % state_period_steps == 0:
                self._publish_joint_states()
            if self._sim_step_count % imu_period_steps == 0:
                self._publish_imu()
            return
        self._apply_pd()
        self._apply_elastic_band()
        if viewer is not None:
            with viewer.lock():
                mujoco.mj_step(self.mj_model, self.mj_data)
        else:
            mujoco.mj_step(self.mj_model, self.mj_data)
        self._sim_step_count += 1
        if self._sim_step_count % state_period_steps == 0:
            self._publish_joint_states()
        if self._sim_step_count % imu_period_steps == 0:
            self._publish_imu()

    def _sim_loop_headless(self):
        """Background-thread sim loop used in headless (no viewer) mode."""
        sim_dt = float(self.args.sim_dt)
        state_period_steps = max(1, int(round(1.0 / float(self.args.state_rate_hz) / sim_dt)))
        imu_period_steps   = max(1, int(round(1.0 / float(self.args.imu_rate_hz)   / sim_dt)))

        next_t = time.monotonic()
        while not self._stop.is_set():
            self._sim_step_once(state_period_steps, imu_period_steps, viewer=None)

            # Real-time pacing.
            next_t += sim_dt
            now = time.monotonic()
            sleep_for = next_t - now
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # Falling behind; reset cadence rather than spiral.
                next_t = now

    # ----------------------------------------------------------------
    # Initial state
    # ----------------------------------------------------------------
    def _reset_to_default_pose(self):
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for i in range(self.eval_x2.NUM_DOFS):
            self.mj_data.qpos[self.qpos_adr[i]] = float(self.eval_x2.DEFAULT_DOF[i])
            self.mj_data.qvel[self.qvel_adr[i]] = 0.0
        # Floating base at a sane height so the robot isn't immediately in the
        # ground plane. Pelvis-up height: eyeballed to ~0.85 m for X2 standing.
        if self.has_floating_base:
            self.mj_data.qpos[2] = 0.85
            self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _reset_to_named_pose(self, pose_name: str):
        """Initialize from a named entry in ``sim_init_poses.yaml``.

        Looks up ``pose_name`` (hyphens already normalized to underscores
        upstream), validates schema, applies per-joint offsets atop
        DEFAULT_DOF, sets the floating base at the YAML's ``pelvis_z``,
        and seeds the standby PD setpoint to the resulting joint config
        so the bridge holds the captured pose during autostart instead of
        fighting it by pulling toward DEFAULT_DOF.

        ``ElasticBand.length`` is also overridden from the YAML's
        ``band_length`` *only if the operator left ``--band-length`` at
        its default (0.0)*; an explicit ``--band-length`` always wins.
        """
        poses = _load_named_init_poses()
        if pose_name not in poses:
            available = sorted(poses.keys()) or ["<none -- YAML missing>"]
            raise ValueError(
                f"Unknown --init-pose {pose_name!r}. "
                f"Available poses in {_INIT_POSES_YAML_PATH}: {available}. "
                "Use --init-pose=default for the built-in legacy behaviour."
            )
        entry = poses[pose_name]
        if not isinstance(entry, dict):
            raise ValueError(
                f"{_INIT_POSES_YAML_PATH}: pose {pose_name!r} must be a mapping."
            )

        # Required fields.
        try:
            pelvis_z = float(entry["pelvis_z"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(
                f"{_INIT_POSES_YAML_PATH}: pose {pose_name!r} missing or "
                "invalid required field 'pelvis_z' (float, m)."
            ) from e
        offsets_rad = entry.get("offsets_rad") or {}
        if not isinstance(offsets_rad, dict):
            raise ValueError(
                f"{_INIT_POSES_YAML_PATH}: pose {pose_name!r}.offsets_rad "
                "must be a mapping of joint_name -> float."
            )

        # Build the qpos joint vector starting from DEFAULT_DOF.
        target_dof = self.eval_x2.DEFAULT_DOF.astype(np.float64).copy()
        name_to_idx = {n: i for i, n in
                        enumerate(self.eval_x2.MUJOCO_JOINT_NAMES)}
        unknown: list[str] = []
        for joint_name, offset in offsets_rad.items():
            try:
                mj_idx = name_to_idx[str(joint_name)]
            except KeyError:
                unknown.append(str(joint_name))
                continue
            target_dof[mj_idx] += float(offset)
        if unknown:
            raise ValueError(
                f"{_INIT_POSES_YAML_PATH}: pose {pose_name!r} references "
                f"unknown joint name(s) {unknown}. Valid names are in "
                "x2_mujoco_ros_bridge.MUJOCO_JOINT_NAMES."
            )

        # Apply to MuJoCo state.
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for i in range(self.eval_x2.NUM_DOFS):
            self.mj_data.qpos[self.qpos_adr[i]] = float(target_dof[i])
            self.mj_data.qvel[self.qvel_adr[i]] = 0.0
        if self.has_floating_base:
            self.mj_data.qpos[2] = pelvis_z
            self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz identity
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Override ElasticBand length from YAML iff operator left it at the
        # CLI default. Explicit --band-length always wins.
        band_length = entry.get("band_length")
        if band_length is not None and self.elastic_band is not None \
                and float(self.args.band_length) == 0.0:
            self.elastic_band.length = float(band_length)

        with self.cmd_lock:
            self._target_pos = target_dof.copy()
        self._init_pose_name = pose_name  # for the startup banner

    def _rsi_from_motion(self, motion_path: pathlib.Path, frame: int):
        if not motion_path.is_file():
            raise FileNotFoundError(f"Motion not found: {motion_path}")
        suffix = motion_path.suffix.lower()
        if suffix == ".pkl":
            data = self.eval_x2.load_motion_data(str(motion_path))
        elif suffix in (".yaml", ".yml"):
            # Same builder ``eval_x2_mujoco_onnx.py --playlist`` uses, so the
            # bridge's RSI init is bit-identical to what the Python sim-to-sim
            # eval starts from.
            data = self.eval_x2.load_playlist_motion_data(str(motion_path))
        else:
            raise ValueError(
                f"Unsupported motion source extension {suffix!r} for {motion_path}; "
                f"expected .pkl, .yaml, or .yml. (X2M2 is the deploy-binary "
                f"runtime format, not an RSI source -- pass the original PKL/YAML "
                f"the X2M2 was baked from.)"
            )
        fps = self.eval_x2.get_motion_fps(data)
        state = self.eval_x2.compute_motion_state(data, frame, fps)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        if self.has_floating_base:
            self.mj_data.qpos[0:3] = state["root_pos_w"]
            self.mj_data.qpos[3:7] = state["root_quat_w_wxyz"]
            self.mj_data.qvel[0:3] = state["root_lin_vel_w"]
            # MuJoCo free-joint qvel convention: qvel[3:6] is BODY-LOCAL angular
            # velocity (NOT world frame). compute_motion_state returns
            # root_ang_vel_w in world frame, so we must rotate it into the body
            # frame here -- mirrors gear_sonic/scripts/eval_x2_mujoco.py:799-802
            # (_apply_init_state). Without this, the IMU's published
            # base_ang_vel disagrees with what the Python eval observes at the
            # same RSI state, which the policy reads as a fictitious tilt rate
            # and corrects against -- causing the parity-profile sim to fall
            # within ~2 s on iter-4000 even after the tokenizer layout fix.
            self.mj_data.qvel[3:6] = self.eval_x2.quat_rotate_inverse(
                state["root_quat_w_wxyz"], state["root_ang_vel_w"]
            )
        for i in range(self.eval_x2.NUM_DOFS):
            self.mj_data.qpos[self.qpos_adr[i]] = float(state["joint_pos_mj"][i])
            self.mj_data.qvel[self.qvel_adr[i]] = float(state["joint_vel_mj"][i])
        mujoco.mj_forward(self.mj_model, self.mj_data)
        # Also seed the standby setpoint to the current joint pose (so the
        # robot doesn't snap back to the default pose between RSI and the
        # first deploy command).
        with self.cmd_lock:
            self._target_pos = state["joint_pos_mj"].astype(np.float64).copy()

    def _print_scene(self):
        print("=== bodies ===")
        for i in range(self.mj_model.nbody):
            print(f"  {i:3d}  {self.mj_model.body(i).name}")
        print("=== joints ===")
        for i in range(self.mj_model.njnt):
            j = self.mj_model.joint(i)
            print(f"  {i:3d}  {j.name}  type={j.type}  qposadr={j.qposadr[0]}  dofadr={j.dofadr[0]}")
        print("=== actuators ===")
        for i in range(self.mj_model.nu):
            print(f"  {i:3d}  {self.mj_model.actuator(i).name}")

    # ----------------------------------------------------------------
    # Lifecycle
    # ----------------------------------------------------------------
    def run(self):
        """Run the bridge.

        Threading model differs by viewer mode:

        - **Headless** (``--viewer`` off): sim loop runs on a background
          thread, ``rclpy.spin_once`` runs on the main thread. This keeps
          the original lean structure for CI / smoke tests.
        - **Viewer on**: sim loop runs on the **main thread** (because
          ``mujoco.viewer.launch_passive`` requires its ``sync()`` to be
          called from the thread that opened it, and we want sim + render
          serialised through ``viewer.lock()`` to avoid the well-known
          GLFW segfault when ``mj_step`` and ``viewer.sync`` race on
          ``mj_data``). ROS spinning is pushed to a background thread.
        """
        self._print_banner()

        if not self.args.viewer:
            sim_thread = threading.Thread(
                target=self._sim_loop_headless,
                name="x2-mujoco-sim", daemon=True,
            )
            sim_thread.start()
            try:
                while not self._stop.is_set():
                    rclpy.spin_once(self.node, timeout_sec=0.05)
            except KeyboardInterrupt:
                pass
            self._stop.set()
            sim_thread.join(timeout=2.0)
            self.node.destroy_node()
            return

        # ---- viewer on: sim+render on main thread, ROS spin in bg ----
        spin_thread = threading.Thread(
            target=self._spin_until_stopped, name="x2-ros-spin", daemon=True,
        )
        spin_thread.start()

        sim_dt = float(self.args.sim_dt)
        state_period_steps = max(1, int(round(1.0 / float(self.args.state_rate_hz) / sim_dt)))
        imu_period_steps   = max(1, int(round(1.0 / float(self.args.imu_rate_hz)   / sim_dt)))

        # Forward keypresses to the band so 7/8/9 work just like in the
        # G1 sim (key 9 toggles the suspension on/off).
        kw = {}
        if self.elastic_band is not None:
            kw["key_callback"] = self.elastic_band.MujuocoKeyCallback
        try:
            with mujoco_viewer.launch_passive(
                self.mj_model, self.mj_data, **kw
            ) as viewer:
                self.viewer = viewer
                next_t = time.monotonic()
                while viewer.is_running() and not self._stop.is_set():
                    self._sim_step_once(state_period_steps, imu_period_steps,
                                        viewer=viewer)
                    # Render only at ~60 Hz to keep sim CPU-bound: sync()
                    # is cheap but does an X11 round-trip every call, so
                    # sub-sampling here helps headroom on the 1 kHz tick.
                    if self._sim_step_count % 16 == 0:
                        viewer.sync()
                    next_t += sim_dt
                    now = time.monotonic()
                    sleep_for = next_t - now
                    if sleep_for > 0:
                        time.sleep(sleep_for)
                    else:
                        next_t = now
        except KeyboardInterrupt:
            pass

        self._stop.set()
        spin_thread.join(timeout=2.0)
        self.node.destroy_node()

    def _spin_until_stopped(self):
        try:
            while not self._stop.is_set():
                rclpy.spin_once(self.node, timeout_sec=0.05)
        except Exception:
            # Don't drop a traceback on Ctrl-C / shutdown -- main thread
            # owns the lifecycle log.
            pass

    def _print_banner(self):
        print("=" * 72)
        print("  X2 Ultra MuJoCo <-> ROS 2 bridge")
        print("=" * 72)
        print(f"  MJCF                {self.mjcf_path}")
        print(f"  Floating base       {self.has_floating_base}")
        print(f"  IMU body            {self.imu_body}")
        print(f"  ROS_LOCALHOST_ONLY  {self._ros_localhost_only}")
        print(f"  ROS_DOMAIN_ID       {self._ros_domain_id}")
        print(f"  sim_dt              {self.args.sim_dt:.4f}s ({1.0/self.args.sim_dt:.0f} Hz)")
        print(f"  state pub rate      {self.args.state_rate_hz:.1f} Hz")
        print(f"  imu pub rate        {self.args.imu_rate_hz:.1f} Hz")
        if self.args.motion:
            print(f"  RSI motion          {self.args.motion} (frame {self.args.init_frame})")
        else:
            pose_label = getattr(self, "_init_pose_name", None)
            if pose_label is None:
                print(f"  Initial pose        DEFAULT_DOF (standing, pelvis_z=0.85m)")
            else:
                pelvis = float(self.mj_data.qpos[2]) if self.has_floating_base else float("nan")
                print(f"  Initial pose        {pose_label} (from "
                      f"{_INIT_POSES_YAML_PATH.name}, pelvis_z={pelvis:.3f}m)")
        print(f"  hold-stiffness-mult {self.args.hold_stiffness_mult}")
        if self.elastic_band is not None:
            band_extra = (
                f"length={self.elastic_band.length:.3f}m, "
                f"kp={self.elastic_band.kp_pos:.0f}/kd={self.elastic_band.kd_pos:.0f}"
            )
            if self.args.viewer:
                print(f"  ElasticBand         ENABLED  ({band_extra}; "
                      f"viewer keys: 9 toggle, 7/8 lower/raise)")
            else:
                if self.args.band_release_after_s >= 0.0:
                    print(f"  ElasticBand         ENABLED  ({band_extra}; "
                          f"auto-release {self.args.band_release_after_s:.2f}s "
                          f"after first deploy cmd)")
                else:
                    print(f"  ElasticBand         ENABLED  ({band_extra}; "
                          f"headless, NEVER auto-release)")
        else:
            print(f"  ElasticBand         disabled")
        print(f"  viewer              {self.args.viewer}")
        print("=" * 72)


def main():
    args = _build_arg_parser().parse_args()

    # DDS isolation -- set BEFORE rclpy imports happen inside the bridge ctor.
    if not args.no_localhost_only and "ROS_LOCALHOST_ONLY" not in os.environ:
        os.environ["ROS_LOCALHOST_ONLY"] = "1"
    if "ROS_DOMAIN_ID" not in os.environ:
        os.environ["ROS_DOMAIN_ID"] = str(int(args.ros_domain_id))

    import rclpy
    rclpy.init()
    try:
        bridge = X2MujocoRosBridge(args)
        bridge.run()
    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
