#!/usr/bin/env python3
"""Record a full X2 deploy run on the real robot for offline debug.

Sibling to ``x2_capture_pose.py``: where capture is a 2-second median
snapshot used to bake a sim init pose, this script is a continuous
high-rate recorder that captures EVERY message on the joint state /
joint command / IMU buses for the duration of a powered run, dumps
them to a single ``.npz``, and prints a quick movement-detection
summary so we can answer questions like "did the policy actually
command hand movement?" without waiting for matplotlib.

Captures:

* ``/aima/hal/joint/{leg,waist,arm,head}/state``    -- JointStateArray
* ``/aima/hal/joint/{leg,waist,arm,head}/command``  -- JointCommandArray
* ``/aima/hal/imu/torso/state``                     -- sensor_msgs/Imu
                                                       (overridable)

QoS matches the deploy's ``rclcpp::SensorDataQoS()``:
``BEST_EFFORT + KEEP_LAST(10) + VOLATILE``. Declaring RELIABLE here
would silently fail to match the deploy's BEST_EFFORT publishers and
we'd record zero samples (same gotcha documented in
``x2_action_monitor.py``).

Output ``.npz`` layout (per-group ragged because the four groups
publish at independent rates -- forced alignment at write time would
lose temporal fidelity):

  meta_json                                # JSON string (run metadata)
  joint_names_<group>                      # object array (k strings)

  t_state_<group>      [N]                 # float64, monotonic seconds
  state_pos_<group>    [N, k]              # float64, rad
  state_vel_<group>    [N, k]              # float64, rad/s
  state_eff_<group>    [N, k]              # float64, Nm

  t_cmd_<group>        [N]
  cmd_pos_<group>      [N, k]
  cmd_vel_<group>      [N, k]
  cmd_kp_<group>       [N, k]
  cmd_kd_<group>       [N, k]

  t_imu                [N]
  imu_quat_wxyz        [N, 4]
  imu_angvel           [N, 3]
  imu_linacc           [N, 3]

  # MC mode timeline (only when --track-mc-mode and GetMcAction is reachable):
  t_mc_mode            [M]                 # float64, monotonic seconds
  mc_mode_str          [M]                 # object array of mode-name strings,
                                           # e.g. "STAND_DEFAULT" / "JOINT_DEFAULT"
                                           # / "PASSIVE_DEFAULT" / "DAMPING_DEFAULT"
                                           # / "LOCOMOTION_DEFAULT" or "" if poll
                                           # failed during that sample.

Two run modes:

* default: record live until ``--duration`` or Ctrl-C, then dump+summarize.
* ``--summarize PATH.npz``: skip ROS, re-run the summary on a prior
  recording. Lets us iterate on analysis without burning robot time.

Usage from the host shell (recommended -- runs alongside an active
``deploy_x2.sh local`` in another terminal):

  cd gear_sonic_deploy/docker_x2 && \\
      docker compose -f docker-compose.yml -f docker-compose.real.yml \\
          run --rm x2sim bash -lc '
              source /ros2_ws/install/setup.bash &&
              python3 /workspace/sonic/gear_sonic_deploy/scripts/x2_record_real_run.py \\
                  --out /workspace/sonic/scratch/run_$(date +%Y%m%d_%H%M%S).npz \\
                  --duration 12
          '

The container has FastDDS + aimdk_msgs from the colcon overlay, plus
the ``--ipc=host`` + ``network_mode: host`` overrides from the real
compose file, so it sees the same DDS traffic as the deploy.

Mirrors patterns in:
  * gear_sonic_deploy/scripts/x2_capture_pose.py
  * gear_sonic_deploy/scripts/x2_action_monitor.py
  * gear_sonic_deploy/scripts/x2_preflight.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# rclpy / aimdk_msgs are only needed for live recording, NOT for
# --summarize on a previously saved .npz. Import lazily so the host
# (which has numpy but no ROS) can still re-print summaries.


# ─────────────────────────────────────────────────────────────────────────
# Joint metadata. MUST stay in lockstep with policy_parameters.hpp /
# x2_action_monitor.py. Hardcoded so the recorder has zero build dep on
# the deploy colcon workspace.
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GroupSpec:
    name: str
    state_topic: str
    cmd_topic: str
    joint_names: tuple[str, ...]


GROUPS: tuple[GroupSpec, ...] = (
    GroupSpec(
        name="leg",
        state_topic="/aima/hal/joint/leg/state",
        cmd_topic="/aima/hal/joint/leg/command",
        joint_names=(
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ),
    ),
    GroupSpec(
        name="waist",
        state_topic="/aima/hal/joint/waist/state",
        cmd_topic="/aima/hal/joint/waist/command",
        joint_names=("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"),
    ),
    GroupSpec(
        name="arm",
        state_topic="/aima/hal/joint/arm/state",
        cmd_topic="/aima/hal/joint/arm/command",
        joint_names=(
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_yaw_joint", "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_yaw_joint", "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ),
    ),
    GroupSpec(
        name="head",
        state_topic="/aima/hal/joint/head/state",
        cmd_topic="/aima/hal/joint/head/command",
        joint_names=("head_yaw_joint", "head_pitch_joint"),
    ),
)
GROUP_BY_NAME = {g.name: g for g in GROUPS}

ALL_JOINT_NAMES: tuple[str, ...] = tuple(jn for g in GROUPS for jn in g.joint_names)


# ─────────────────────────────────────────────────────────────────────────
# Per-stream rolling buffer
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class JointStreamBuf:
    """One lock-protected ring of samples for a single joint group's
    state OR command stream. We append per-callback and convert to dense
    np.ndarray at dump time."""

    n_joints: int
    t: list[float] = field(default_factory=list)
    pos: list[list[float]] = field(default_factory=list)
    vel: list[list[float]] = field(default_factory=list)
    # third-channel = effort for state, kp for command. fourth-channel
    # only used for command (kd). State leaves ch3=ch4=None.
    ch3: list[list[float]] = field(default_factory=list)
    ch4: list[list[float]] = field(default_factory=list)
    name_warned: bool = False


@dataclass
class ImuStreamBuf:
    t: list[float] = field(default_factory=list)
    quat_wxyz: list[list[float]] = field(default_factory=list)
    angvel: list[list[float]] = field(default_factory=list)
    linacc: list[list[float]] = field(default_factory=list)


@dataclass
class McModeBuf:
    """Timestamped MC mode samples from periodic GetMcAction polling.
    ``mode`` carries strings like ``"STAND_DEFAULT"`` / ``"JOINT_DEFAULT"``
    or empty string when a poll failed (cross-host service flakiness)."""

    t: list[float] = field(default_factory=list)
    mode: list[str] = field(default_factory=list)
    last_mode: str = ""           # for status-line printing
    fail_warned: bool = False     # gate single warn on first failure


# ─────────────────────────────────────────────────────────────────────────
# Recorder node. The rclpy / aimdk_msgs imports are guarded so this
# module remains importable on hosts that have only numpy (so
# ``--summarize`` can re-print analyses without a docker shell). When
# the imports fail, ``RecorderNode`` is set to None and ``cmd_record``
# bails with a helpful error.
# ─────────────────────────────────────────────────────────────────────────
try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node as _RclpyNode
    from rclpy.qos import (
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile,
        ReliabilityPolicy,
    )
    from aimdk_msgs.msg import JointCommandArray, JointStateArray
    from sensor_msgs.msg import Imu
    _ROS_IMPORT_ERROR: Optional[BaseException] = None
    # GetMcAction is optional -- on some firmwares the service may be absent
    # or namespaced differently. Failure to import just disables MC tracking,
    # rest of the recorder still works.
    try:
        from aimdk_msgs.srv import GetMcAction  # type: ignore
        _GETMCACTION_IMPORT_ERROR: Optional[BaseException] = None
    except ImportError as _e_get:
        GetMcAction = None  # type: ignore
        _GETMCACTION_IMPORT_ERROR = _e_get
except ImportError as _e:
    rclpy = None  # type: ignore
    SingleThreadedExecutor = None  # type: ignore
    _RclpyNode = object  # type: ignore
    JointCommandArray = None  # type: ignore
    JointStateArray = None  # type: ignore
    Imu = None  # type: ignore
    GetMcAction = None  # type: ignore
    _GETMCACTION_IMPORT_ERROR = _e
    _ROS_IMPORT_ERROR = _e

    class _DummyQoS:  # placeholders so module-level def below stays valid
        BEST_EFFORT = None
        KEEP_LAST = None
        VOLATILE = None

    DurabilityPolicy = _DummyQoS  # type: ignore
    HistoryPolicy = _DummyQoS  # type: ignore
    ReliabilityPolicy = _DummyQoS  # type: ignore

    def QoSProfile(*_a, **_kw):  # type: ignore
        return None


_QOS = (
    QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        history=HistoryPolicy.KEEP_LAST,
        depth=10,
        durability=DurabilityPolicy.VOLATILE,
    )
    if _ROS_IMPORT_ERROR is None
    else None
)


class RecorderNode(_RclpyNode):
    # Service name MC speaks at runtime. Mirrors the documented surface in
    # docs/source/user_guide/x2_first_real_robot.md (lines 196-204) and the
    # X2 SDK example_pkg::set_mc_action.cpp pattern. Note the literal "_5F"
    # in the path -- that's the ROS-mangled form of the leading underscore
    # in "aimdk_msgs", not a typo.
    GET_MC_ACTION_SERVICE = "/aimdk_5Fmsgs/srv/GetMcAction"

    def __init__(self, imu_topic: str, status_period_s: float = 1.0,
                 quiet: bool = False, track_mc_mode: bool = True,
                 mc_poll_hz: float = 5.0):
        if _ROS_IMPORT_ERROR is not None:
            raise SystemExit(
                "ERROR: rclpy / aimdk_msgs not importable. Live recording\n"
                "       requires the docker_x2/ container.\n"
                f"       Original error: {_ROS_IMPORT_ERROR}"
            )
        super().__init__("x2_record_real_run")
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._quiet = quiet

        self._state_buf: dict[str, JointStreamBuf] = {
            g.name: JointStreamBuf(n_joints=len(g.joint_names)) for g in GROUPS
        }
        self._cmd_buf: dict[str, JointStreamBuf] = {
            g.name: JointStreamBuf(n_joints=len(g.joint_names)) for g in GROUPS
        }
        self._imu_buf = ImuStreamBuf()
        self._mc_mode_buf = McModeBuf()
        self._imu_topic = imu_topic

        # Per-group MJ-index maps (joint_name -> col in our matrix).
        self._idx: dict[str, dict[str, int]] = {
            g.name: {jn: i for i, jn in enumerate(g.joint_names)} for g in GROUPS
        }

        for g in GROUPS:
            self.create_subscription(
                JointStateArray, g.state_topic,
                lambda msg, gn=g.name: self._on_state(gn, msg),
                _QOS,
            )
            self.create_subscription(
                JointCommandArray, g.cmd_topic,
                lambda msg, gn=g.name: self._on_cmd(gn, msg),
                _QOS,
            )

        self.create_subscription(Imu, imu_topic, self._on_imu, _QOS)

        # MC-mode polling. Optional: only if the operator wants it AND the
        # GetMcAction service type imported cleanly. Soft-failure on every
        # axis (no service type, service not advertised, single-call timeout)
        # so the rest of the recording is unaffected.
        self._mc_client = None
        self._mc_track = bool(track_mc_mode)
        self._mc_poll_hz = float(mc_poll_hz)
        if self._mc_track:
            if _GETMCACTION_IMPORT_ERROR is not None:
                self.get_logger().warn(
                    "MC-mode tracking disabled: aimdk_msgs.srv.GetMcAction "
                    f"unavailable ({_GETMCACTION_IMPORT_ERROR})"
                )
                self._mc_track = False
            else:
                self._mc_client = self.create_client(
                    GetMcAction, self.GET_MC_ACTION_SERVICE,
                )
        if self._mc_track and self._mc_poll_hz > 0:
            period = 1.0 / self._mc_poll_hz
            self.create_timer(period, self._on_mc_poll)

        if status_period_s > 0 and not quiet:
            self.create_timer(status_period_s, self._on_status)

    # ── Subscription callbacks ───────────────────────────────────────────

    def _now(self) -> float:
        return time.monotonic() - self._t0

    def _fill_one(self, buf: JointStreamBuf, group: str, joints,
                  *, kind: str) -> None:
        """Project a per-message joint list onto the canonical group order
        by NAME. Missing names get NaN; unknown names are silently dropped
        (same policy as x2_capture_pose.py). ``kind`` is "state" or "cmd"
        and selects which fields to read off each entry."""
        idx = self._idx[group]
        n = buf.n_joints
        row_pos = [math.nan] * n
        row_vel = [math.nan] * n
        row_ch3 = [math.nan] * n
        row_ch4 = [math.nan] * n if kind == "cmd" else None

        for j in joints:
            name = str(j.name) if j.name else ""
            col = idx.get(name)
            if col is None:
                if not buf.name_warned and name:
                    buf.name_warned = True
                    self.get_logger().warn(
                        f"{group}/{kind}: ignoring unknown joint '{name}'"
                    )
                continue
            row_pos[col] = float(j.position)
            row_vel[col] = float(j.velocity)
            if kind == "state":
                row_ch3[col] = float(j.effort)
            else:
                row_ch3[col] = float(j.stiffness)
                row_ch4[col] = float(j.damping)

        with self._lock:
            buf.t.append(self._now())
            buf.pos.append(row_pos)
            buf.vel.append(row_vel)
            buf.ch3.append(row_ch3)
            if row_ch4 is not None:
                buf.ch4.append(row_ch4)

    def _on_state(self, group: str, msg: JointStateArray) -> None:
        self._fill_one(self._state_buf[group], group, msg.joints, kind="state")

    def _on_cmd(self, group: str, msg: JointCommandArray) -> None:
        self._fill_one(self._cmd_buf[group], group, msg.joints, kind="cmd")

    def _on_imu(self, msg: Imu) -> None:
        with self._lock:
            self._imu_buf.t.append(self._now())
            q = msg.orientation
            self._imu_buf.quat_wxyz.append(
                [float(q.w), float(q.x), float(q.y), float(q.z)]
            )
            a = msg.angular_velocity
            self._imu_buf.angvel.append([float(a.x), float(a.y), float(a.z)])
            la = msg.linear_acceleration
            self._imu_buf.linacc.append([float(la.x), float(la.y), float(la.z)])

    # ── MC mode polling ──────────────────────────────────────────────────

    def _on_mc_poll(self) -> None:
        """Fire one async GetMcAction request. The result lands in
        ``_on_mc_response`` whenever the MC service answers. We never
        block the executor: if the service is slow / unreachable, the
        next tick just queues another request. Empty samples (poll
        failure) are still timestamped so the .npz preserves cadence
        and segmentation logic can detect 'silent windows'."""
        if self._mc_client is None:
            return
        # service_is_ready is non-blocking; cheap.
        if not self._mc_client.service_is_ready():
            with self._lock:
                self._mc_mode_buf.t.append(self._now())
                self._mc_mode_buf.mode.append("")
                if not self._mc_mode_buf.fail_warned and not self._quiet:
                    self._mc_mode_buf.fail_warned = True
                    self.get_logger().warn(
                        f"MC service '{self.GET_MC_ACTION_SERVICE}' not "
                        f"advertised (yet?). Future poll failures suppressed."
                    )
            return
        req = GetMcAction.Request()
        future = self._mc_client.call_async(req)
        future.add_done_callback(self._on_mc_response)

    def _on_mc_response(self, future) -> None:
        t = self._now()
        try:
            resp = future.result()
        except Exception:
            resp = None
        mode = ""
        if resp is not None:
            # The X2 SDK's reply nests the action inside `info`; older
            # firmwares put it at the top level. Try both before giving up.
            info = getattr(resp, "info", None)
            if info is not None:
                mode = str(getattr(info, "action_desc", "") or "")
            if not mode:
                mode = str(getattr(resp, "action_desc", "") or "")
        with self._lock:
            self._mc_mode_buf.t.append(t)
            self._mc_mode_buf.mode.append(mode)
            if mode:
                self._mc_mode_buf.last_mode = mode

    # ── 1 Hz status line ─────────────────────────────────────────────────

    def _on_status(self) -> None:
        with self._lock:
            elapsed = self._now()
            parts = [f"[{elapsed:6.1f}s]"]
            for g in GROUPS:
                ns = len(self._state_buf[g.name].t)
                nc = len(self._cmd_buf[g.name].t)
                parts.append(f"{g.name}:s{ns} c{nc}")
            parts.append(f"imu:{len(self._imu_buf.t)}")

            # Worst tracking error so far on arm group (most relevant for
            # the "no hand movement" debug). Cheap: compare last cmd vs
            # last state by name, take max abs.
            arm_cmd = self._cmd_buf["arm"]
            arm_st = self._state_buf["arm"]
            if arm_cmd.pos and arm_st.pos:
                last_c = np.asarray(arm_cmd.pos[-1], dtype=np.float64)
                last_s = np.asarray(arm_st.pos[-1], dtype=np.float64)
                diff = np.abs(last_c - last_s)
                diff = diff[np.isfinite(diff)]
                if diff.size > 0:
                    parts.append(f"arm_track_max={float(np.max(diff)):.3f}rad")

            if self._mc_track:
                mc_now = self._mc_mode_buf.last_mode or "?"
                parts.append(f"mc:{mc_now}")

            print(" ".join(parts), flush=True)

    # ── Snapshot (for atomic dump) ───────────────────────────────────────

    def snapshot(self) -> dict:
        with self._lock:
            out: dict = {}
            for g in GROUPS:
                sb = self._state_buf[g.name]
                cb = self._cmd_buf[g.name]
                out[f"t_state_{g.name}"] = np.asarray(sb.t, dtype=np.float64)
                out[f"state_pos_{g.name}"] = np.asarray(sb.pos, dtype=np.float64) \
                    if sb.pos else np.empty((0, len(g.joint_names)), dtype=np.float64)
                out[f"state_vel_{g.name}"] = np.asarray(sb.vel, dtype=np.float64) \
                    if sb.vel else np.empty((0, len(g.joint_names)), dtype=np.float64)
                out[f"state_eff_{g.name}"] = np.asarray(sb.ch3, dtype=np.float64) \
                    if sb.ch3 else np.empty((0, len(g.joint_names)), dtype=np.float64)

                out[f"t_cmd_{g.name}"] = np.asarray(cb.t, dtype=np.float64)
                out[f"cmd_pos_{g.name}"] = np.asarray(cb.pos, dtype=np.float64) \
                    if cb.pos else np.empty((0, len(g.joint_names)), dtype=np.float64)
                out[f"cmd_vel_{g.name}"] = np.asarray(cb.vel, dtype=np.float64) \
                    if cb.vel else np.empty((0, len(g.joint_names)), dtype=np.float64)
                out[f"cmd_kp_{g.name}"] = np.asarray(cb.ch3, dtype=np.float64) \
                    if cb.ch3 else np.empty((0, len(g.joint_names)), dtype=np.float64)
                out[f"cmd_kd_{g.name}"] = np.asarray(cb.ch4, dtype=np.float64) \
                    if cb.ch4 else np.empty((0, len(g.joint_names)), dtype=np.float64)

                out[f"joint_names_{g.name}"] = np.asarray(g.joint_names, dtype=object)

            ib = self._imu_buf
            out["t_imu"] = np.asarray(ib.t, dtype=np.float64)
            out["imu_quat_wxyz"] = np.asarray(ib.quat_wxyz, dtype=np.float64) \
                if ib.quat_wxyz else np.empty((0, 4), dtype=np.float64)
            out["imu_angvel"] = np.asarray(ib.angvel, dtype=np.float64) \
                if ib.angvel else np.empty((0, 3), dtype=np.float64)
            out["imu_linacc"] = np.asarray(ib.linacc, dtype=np.float64) \
                if ib.linacc else np.empty((0, 3), dtype=np.float64)

            mb = self._mc_mode_buf
            out["t_mc_mode"] = np.asarray(mb.t, dtype=np.float64)
            out["mc_mode_str"] = np.asarray(mb.mode, dtype=object) \
                if mb.mode else np.empty((0,), dtype=object)

            return out


# ─────────────────────────────────────────────────────────────────────────
# Summary helpers (also used by --summarize PATH.npz)
# ─────────────────────────────────────────────────────────────────────────
def _range_per_joint(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (min, max, range) per column, ignoring NaNs. arr is [N, k]."""
    if arr.size == 0:
        k = arr.shape[1] if arr.ndim == 2 else 0
        nan_arr = np.full(k, np.nan, dtype=np.float64)
        return nan_arr, nan_arr.copy(), nan_arr.copy()
    mn = np.nanmin(arr, axis=0)
    mx = np.nanmax(arr, axis=0)
    return mn, mx, mx - mn


def _resample_to(t_src: np.ndarray, x_src: np.ndarray,
                 t_dst: np.ndarray) -> np.ndarray:
    """Linearly resample x_src(t_src) onto t_dst. Per-column (assumes
    x_src is [N, k]). Used for tracking-error computation when cmd and
    state run at different rates."""
    if t_src.size == 0 or x_src.size == 0:
        return np.full((len(t_dst), x_src.shape[1] if x_src.ndim == 2 else 0),
                       np.nan, dtype=np.float64)
    out = np.empty((len(t_dst), x_src.shape[1]), dtype=np.float64)
    for j in range(x_src.shape[1]):
        out[:, j] = np.interp(t_dst, t_src, x_src[:, j],
                              left=np.nan, right=np.nan)
    return out


def _print_header(meta: dict) -> None:
    print()
    print("=" * 78)
    print("X2 real-run recording")
    print("=" * 78)
    print(f"  out:        {meta.get('out_path')}")
    print(f"  duration:   {meta.get('duration_s_actual', '?'):.2f} s "
          f"(requested {meta.get('duration_s_requested', '?')})")
    print(f"  hostname:   {meta.get('hostname')}")
    print(f"  ros_domain: {meta.get('ros_domain_id')}")
    print(f"  imu topic:  {meta.get('imu_topic')}")
    if meta.get('git_sha'):
        print(f"  git sha:    {meta['git_sha']}")


def _print_rates(snap: dict, duration_s: float) -> None:
    print()
    print("Topic rates:")
    for g in GROUPS:
        ns = snap[f"t_state_{g.name}"].size
        nc = snap[f"t_cmd_{g.name}"].size
        rs = ns / duration_s if duration_s > 0 else 0.0
        rc = nc / duration_s if duration_s > 0 else 0.0
        s_tag = "ok" if ns > 0 else "MISSING"
        c_tag = "ok" if nc > 0 else "MISSING"
        print(f"  {g.name:<5} state {ns:>6} msgs ({rs:6.1f} Hz) [{s_tag}]   "
              f"cmd {nc:>6} msgs ({rc:6.1f} Hz) [{c_tag}]")
    n_imu = snap["t_imu"].size
    rate_imu = n_imu / duration_s if duration_s > 0 else 0.0
    tag = "ok" if n_imu > 0 else "MISSING"
    print(f"  imu                          imu {n_imu:>6} msgs "
          f"({rate_imu:6.1f} Hz) [{tag}]")


def _print_movement_table(snap: dict, moving_threshold: float) -> None:
    """Per-joint cmd_range and state_range, sorted by max-of-the-two.
    This is THE table for 'did anything actually move?' debugging."""
    rows = []
    for g in GROUPS:
        cmd_pos = snap[f"cmd_pos_{g.name}"]
        st_pos = snap[f"state_pos_{g.name}"]
        c_mn, c_mx, c_rng = _range_per_joint(cmd_pos)
        s_mn, s_mx, s_rng = _range_per_joint(st_pos)
        for i, jn in enumerate(g.joint_names):
            rows.append((g.name, jn, c_rng[i], s_rng[i],
                         c_mn[i], c_mx[i], s_mn[i], s_mx[i]))

    def keyfn(r):
        c = r[2] if not math.isnan(r[2]) else -1.0
        s = r[3] if not math.isnan(r[3]) else -1.0
        return -max(c, s)

    rows.sort(key=keyfn)

    print()
    print("Movement detection (sorted by max(cmd_range, state_range)):")
    print(f"  {'group':<5}  {'joint':<28}  "
          f"{'cmd range':>10}  {'state range':>11}  "
          f"{'cmd [min, max] (deg)':<24}  {'state [min, max] (deg)':<24}")
    print("  " + "-" * 110)
    moved_count = 0
    for grp, jn, crng, srng, cmn, cmx, smn, smx in rows:
        cflag = "*" if (not math.isnan(crng) and crng > moving_threshold) else " "
        sflag = "*" if (not math.isnan(srng) and srng > moving_threshold) else " "
        if cflag == "*" or sflag == "*":
            moved_count += 1
        crng_s = f"{crng:>9.4f}{cflag}" if not math.isnan(crng) else "      n/a "
        srng_s = f"{srng:>10.4f}{sflag}" if not math.isnan(srng) else "      n/a  "
        c_box = (f"[{math.degrees(cmn):+6.1f}, {math.degrees(cmx):+6.1f}]"
                 if not math.isnan(cmn) else "  n/a")
        s_box = (f"[{math.degrees(smn):+6.1f}, {math.degrees(smx):+6.1f}]"
                 if not math.isnan(smn) else "  n/a")
        print(f"  {grp:<5}  {jn:<28}  {crng_s}  {srng_s}  "
              f"{c_box:<24}  {s_box:<24}")
    print()
    print(f"  '*' = range > {moving_threshold:.3f} rad "
          f"({math.degrees(moving_threshold):.1f} deg). "
          f"{moved_count} joint(s) flagged as moving.")


def _print_arm_spotlight(snap: dict) -> None:
    """Always-show table for the arm group. The user's debug question
    ('did the hands move?') lives here regardless of threshold."""
    g = GROUP_BY_NAME["arm"]
    cmd_pos = snap[f"cmd_pos_{g.name}"]
    st_pos = snap[f"state_pos_{g.name}"]
    t_cmd = snap[f"t_cmd_{g.name}"]
    t_st = snap[f"t_state_{g.name}"]

    if cmd_pos.size == 0 and st_pos.size == 0:
        print()
        print("Arm spotlight: no samples received.")
        return

    # Tracking error: resample state onto cmd timestamps and take diff.
    if cmd_pos.size > 0 and st_pos.size > 0 and t_cmd.size > 0 and t_st.size > 0:
        st_resamp = _resample_to(t_st, st_pos, t_cmd)
        err = cmd_pos - st_resamp
        rms = np.sqrt(np.nanmean(err ** 2, axis=0))
        peak = np.nanmax(np.abs(err), axis=0)
    else:
        rms = np.full(len(g.joint_names), np.nan, dtype=np.float64)
        peak = rms.copy()

    c_mn, c_mx, c_rng = _range_per_joint(cmd_pos)
    s_mn, s_mx, s_rng = _range_per_joint(st_pos)

    print()
    print("Arm spotlight (always shown, regardless of threshold):")
    print(f"  {'joint':<28}  {'cmd_range':>10}  {'state_range':>11}  "
          f"{'rms_err':>9}  {'peak_err':>9}")
    print("  " + "-" * 80)
    for i, jn in enumerate(g.joint_names):
        crng = f"{c_rng[i]:>9.4f}" if not math.isnan(c_rng[i]) else "      n/a"
        srng = f"{s_rng[i]:>10.4f}" if not math.isnan(s_rng[i]) else "       n/a"
        rms_s = f"{rms[i]:>8.4f}" if not math.isnan(rms[i]) else "     n/a"
        peak_s = f"{peak[i]:>8.4f}" if not math.isnan(peak[i]) else "     n/a"
        print(f"  {jn:<28}  {crng}  {srng}  {rms_s}  {peak_s}")
    print()
    print("  rms_err / peak_err units: rad. Computed as (cmd - resampled_state)")
    print("  over the recording window. Large rms with small state_range = "
          "policy commanded motion the robot did NOT execute "
          "(MC/HAL fighting back, or kp too low).")


def _segment_by_mode(t_mode: np.ndarray, mode_str: np.ndarray,
                     t_end: float) -> list[tuple[float, float, str]]:
    """Compress consecutive (t, mode_str) samples into [t_start, t_end, mode]
    segments. Empty-string ('' -- failed poll) samples extend whichever
    mode they're sandwiched in: they don't open a new segment but they
    don't close one either. The recording's wall end-time t_end caps
    the last segment.

    Returns a list of (t_start, t_end, mode_label) tuples, mode_label
    = "" if every sample in the window failed (rare; service down)."""
    if t_mode.size == 0 or mode_str.size == 0:
        return []
    segs: list[list] = []
    cur_label = ""
    seg_start = float(t_mode[0])
    for i in range(t_mode.size):
        m = str(mode_str[i]) if mode_str[i] is not None else ""
        if not m:
            continue  # skip failed polls; they extend whatever segment is open
        if not segs:
            segs.append([seg_start, float(t_mode[i]), m])
            cur_label = m
            continue
        if m != cur_label:
            segs[-1][1] = float(t_mode[i])
            segs.append([float(t_mode[i]), float(t_mode[i]), m])
            cur_label = m
        else:
            segs[-1][1] = float(t_mode[i])
    if segs:
        segs[-1][1] = max(segs[-1][1], t_end)
    return [(s, e, m) for s, e, m in segs]


def _slice_window(t: np.ndarray, x: np.ndarray, t_start: float, t_end: float):
    """Return rows of ``x`` whose timestamp lies in [t_start, t_end]. ``x``
    can be 1-D or 2-D. Cheap; segments are usually short."""
    if t.size == 0:
        return x[:0]
    mask = (t >= t_start) & (t <= t_end)
    return x[mask]


def _per_segment_stats(snap: dict, segs: list[tuple[float, float, str]]) -> list[dict]:
    """For each (t_start, t_end, mode) segment, compute per-group joint
    stats and IMU summary. Returns a list of dicts ready to render."""
    rows: list[dict] = []
    t_imu = snap.get("t_imu", np.empty(0))
    quats = snap.get("imu_quat_wxyz", np.empty((0, 4)))
    angvel = snap.get("imu_angvel", np.empty((0, 3)))

    for s, e, mode in segs:
        row: dict = {"t_start": s, "t_end": e, "mode": mode,
                     "duration_s": max(e - s, 0.0)}
        for g in GROUPS:
            t_st = snap[f"t_state_{g.name}"]
            eff = snap[f"state_eff_{g.name}"]
            vel = snap[f"state_vel_{g.name}"]
            t_cmd = snap[f"t_cmd_{g.name}"]
            kp = snap[f"cmd_kp_{g.name}"]
            kd = snap[f"cmd_kd_{g.name}"]

            eff_w = _slice_window(t_st, eff, s, e)
            vel_w = _slice_window(t_st, vel, s, e)
            kp_w = _slice_window(t_cmd, kp, s, e)
            kd_w = _slice_window(t_cmd, kd, s, e)

            row[f"eff_rms_{g.name}"] = (
                float(np.sqrt(np.nanmean(eff_w ** 2))) if eff_w.size else float("nan")
            )
            row[f"vel_rms_{g.name}"] = (
                float(np.sqrt(np.nanmean(vel_w ** 2))) if vel_w.size else float("nan")
            )
            row[f"kp_mean_{g.name}"] = (
                float(np.nanmean(np.abs(kp_w))) if kp_w.size else float("nan")
            )
            row[f"kd_mean_{g.name}"] = (
                float(np.nanmean(np.abs(kd_w))) if kd_w.size else float("nan")
            )

        if t_imu.size > 0 and quats.size > 0:
            q_w = _slice_window(t_imu, quats, s, e)
            av_w = _slice_window(t_imu, angvel, s, e)
            tilt_max = float("nan")
            if q_w.shape[0] > 0:
                tilts = []
                for j in range(q_w.shape[0]):
                    qw, qx, qy, qz = q_w[j]
                    gz = -(qw * qw - qx * qx - qy * qy + qz * qz)
                    tilts.append(math.degrees(math.acos(max(-1.0, min(1.0, -gz)))))
                tilt_max = float(np.max(tilts)) if tilts else float("nan")
            row["tilt_max_deg"] = tilt_max
            if av_w.size > 0:
                row["angvel_max"] = float(np.max(np.linalg.norm(av_w, axis=1)))
                row["angvel_mean"] = float(np.mean(np.linalg.norm(av_w, axis=1)))
            else:
                row["angvel_max"] = float("nan")
                row["angvel_mean"] = float("nan")
        else:
            row["tilt_max_deg"] = float("nan")
            row["angvel_max"] = float("nan")
            row["angvel_mean"] = float("nan")

        rows.append(row)
    return rows


def _print_mc_mode_summary(snap: dict, duration_s: float) -> None:
    t_mode = snap.get("t_mc_mode")
    mode_str = snap.get("mc_mode_str")
    if (t_mode is None or mode_str is None
            or t_mode.size == 0 or mode_str.size == 0):
        return
    segs = _segment_by_mode(t_mode, mode_str, duration_s)
    if not segs:
        print()
        print("MC mode timeline: no successful GetMcAction polls "
              "(service unavailable on the bus).")
        return
    rows = _per_segment_stats(snap, segs)
    print()
    print(f"MC mode timeline ({len(segs)} segment(s), "
          f"{sum(r['duration_s'] for r in rows):.1f} s of labelled coverage):")
    hdr = (
        f"  {'t_start':>7}  {'t_end':>7}  {'dur':>5}  {'mode':<22}"
        f"  {'kp_arm':>7}  {'eff_arm':>8}  {'eff_leg':>8}"
        f"  {'tilt_max':>9}  {'avel_max':>9}"
    )
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        print(
            f"  {r['t_start']:7.2f}  {r['t_end']:7.2f}  {r['duration_s']:5.2f}  "
            f"{r['mode'][:22]:<22}  "
            f"{r['kp_mean_arm']:7.2f}  {r['eff_rms_arm']:8.2f}  "
            f"{r['eff_rms_leg']:8.2f}  "
            f"{r['tilt_max_deg']:8.1f}d  {r['angvel_max']:8.2f}"
        )

    # Label inference -- aggregate over all visits to each mode.
    by_mode: dict[str, list[dict]] = {}
    for r in rows:
        by_mode.setdefault(r["mode"], []).append(r)

    def _avg(rs: list[dict], key: str) -> float:
        vals = [r[key] for r in rs if not math.isnan(r.get(key, float("nan")))]
        return float(np.mean(vals)) if vals else float("nan")

    def _max_of(rs: list[dict], key: str) -> float:
        vals = [r[key] for r in rs if not math.isnan(r.get(key, float("nan")))]
        return float(np.max(vals)) if vals else float("nan")

    print()
    print("Per-mode aggregate (averaged across segment visits):")
    print(f"  {'mode':<22}  {'visits':>6}  {'eff_rms_leg':>11}  "
          f"{'eff_rms_arm':>11}  {'vel_rms_leg':>11}  "
          f"{'tilt_max':>9}  {'avel_mean':>9}")
    print("  " + "-" * 90)
    for mode in sorted(by_mode):
        rs = by_mode[mode]
        print(
            f"  {mode[:22]:<22}  {len(rs):>6}  "
            f"{_avg(rs, 'eff_rms_leg'):11.2f}  "
            f"{_avg(rs, 'eff_rms_arm'):11.2f}  "
            f"{_avg(rs, 'vel_rms_leg'):11.4f}  "
            f"{_max_of(rs, 'tilt_max_deg'):8.1f}d  "
            f"{_avg(rs, 'angvel_mean'):9.3f}"
        )

    # Heuristic labels.
    print()
    print("Inferred labels (heuristic; cross-check against MC docs):")
    for mode in sorted(by_mode):
        rs = by_mode[mode]
        eff_leg = _avg(rs, "eff_rms_leg")
        eff_arm = _avg(rs, "eff_rms_arm")
        vel_leg = _avg(rs, "vel_rms_leg")
        tilt_max = _max_of(rs, "tilt_max_deg")
        if math.isnan(eff_leg) or math.isnan(eff_arm):
            label = "(insufficient data)"
        elif eff_leg < 0.5 and eff_arm < 0.5:
            label = "zero-torque (motors idle / passive)"
        elif eff_leg < 2.0 and eff_arm < 2.0 and (
                math.isnan(vel_leg) or vel_leg < 0.05):
            label = "damping only (low effort, near-zero velocity)"
        elif tilt_max < 5.0:
            label = "active balance (sustained effort, tilt bounded)"
        elif eff_leg > 2.0:
            label = "active controller (significant effort but tilt > 5deg "
            label += "-- pushed or commanded motion)"
        else:
            label = "(ambiguous)"
        print(f"  {mode[:22]:<22}  -> {label}")
    print()
    print("  Note: cmd_kp / cmd_kd reflect what *we* (the recorder host) "
          "publish on the bus; in this probe we publish nothing, so all "
          "kp/kd here are 0. The discriminator across MC modes is the "
          "actually-applied joint torque (state_eff) and the IMU response, "
          "both observed via HAL.")


def _print_imu_summary(snap: dict) -> None:
    quats = snap.get("imu_quat_wxyz")
    angvel = snap.get("imu_angvel")
    if quats is None or quats.size == 0:
        print()
        print("IMU: no samples received.")
        return
    qmed = np.median(quats, axis=0)
    qw, qx, qy, qz = qmed
    gx = -2.0 * (qx * qz - qw * qy)
    gy = -2.0 * (qy * qz + qw * qx)
    gz = -(qw * qw - qx * qx - qy * qy + qz * qz)
    n = math.sqrt(gx * gx + gy * gy + gz * gz)
    if n > 1e-9:
        gx, gy, gz = gx / n, gy / n, gz / n
    tilt_deg = math.degrees(math.acos(max(-1.0, min(1.0, -gz))))
    angvel_max = float(np.max(np.abs(angvel))) if angvel.size else float("nan")
    print()
    print("IMU summary:")
    print(f"  median quat (w,x,y,z) = ({qw:+.4f}, {qx:+.4f}, "
          f"{qy:+.4f}, {qz:+.4f})")
    print(f"  gravity_body          = ({gx:+.3f}, {gy:+.3f}, {gz:+.3f})  "
          f"|tilt|={tilt_deg:.1f} deg from upright")
    print(f"  max |angular vel|     = {angvel_max:.4f} rad/s")


def _summarize(snap: dict, meta: dict, moving_threshold: float) -> None:
    duration = float(meta.get("duration_s_actual", 0.0))
    _print_header(meta)
    _print_rates(snap, duration)
    _print_movement_table(snap, moving_threshold)
    _print_arm_spotlight(snap)
    _print_imu_summary(snap)
    _print_mc_mode_summary(snap, duration)
    print()


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def _git_sha() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(pathlib.Path(__file__).resolve().parent),
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip() or None
    except Exception:
        return None


def cmd_record(args: argparse.Namespace) -> int:
    if _ROS_IMPORT_ERROR is not None:
        print(
            "ERROR: rclpy / aimdk_msgs not importable on this interpreter.\n"
            "       Live recording requires running inside the docker_x2/\n"
            "       container. For offline analysis of an existing\n"
            "       recording, use --summarize PATH.npz.\n"
            f"       Original error: {_ROS_IMPORT_ERROR}",
            file=sys.stderr,
        )
        return 1
    out = pathlib.Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node: Optional[RecorderNode] = None
    executor = SingleThreadedExecutor()
    stop_flag = threading.Event()

    def _on_sig(_signum, _frame):
        stop_flag.set()

    # Install AFTER rclpy.init so we override its handlers (Ctrl-C should
    # close out the recording, not nuke it).
    signal.signal(signal.SIGINT, _on_sig)
    signal.signal(signal.SIGTERM, _on_sig)

    t_start_wall = time.time()
    t_start_mono = time.monotonic()
    try:
        node = RecorderNode(
            imu_topic=args.imu_topic,
            status_period_s=args.status_period,
            quiet=args.quiet,
            track_mc_mode=args.track_mc_mode,
            mc_poll_hz=args.mc_poll_hz,
        )
        executor.add_node(node)

        print(f"Recording -> {out}")
        print(f"  duration:    {args.duration:.1f} s "
              + ("(0 = until Ctrl-C)" if args.duration <= 0 else ""))
        print(f"  imu topic:   {args.imu_topic}")
        print(f"  status:      "
              + (f"every {args.status_period:.1f} s" if not args.quiet
                 else "quiet"))
        print()

        deadline = (t_start_mono + args.duration) if args.duration > 0 else math.inf
        while not stop_flag.is_set() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        if stop_flag.is_set():
            print("\nReceived stop signal; finalizing...", flush=True)

        snap = node.snapshot()
    finally:
        if node is not None:
            executor.remove_node(node)
            node.destroy_node()
        executor.shutdown()
        try:
            rclpy.shutdown()
        except Exception:
            pass

    duration_actual = time.monotonic() - t_start_mono

    meta = {
        "out_path": str(out),
        "duration_s_requested": args.duration,
        "duration_s_actual": duration_actual,
        "started_at_wall": t_start_wall,
        "started_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S",
                                         time.localtime(t_start_wall)),
        "hostname": socket.gethostname(),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "ros_localhost_only": os.environ.get("ROS_LOCALHOST_ONLY"),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
        "imu_topic": args.imu_topic,
        "git_sha": _git_sha(),
        "note": args.note,
    }

    snap_to_save = dict(snap)
    snap_to_save["meta_json"] = np.asarray(json.dumps(meta, indent=2))

    np.savez_compressed(out, **snap_to_save)
    print(f"\nSaved {out} ({out.stat().st_size / 1024:.1f} KiB)")

    _summarize(snap, meta, args.moving_threshold)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    path = pathlib.Path(args.summarize).expanduser().resolve()
    if not path.exists():
        print(f"ERROR: {path} does not exist", file=sys.stderr)
        return 1
    z = np.load(path, allow_pickle=True)
    snap = {k: z[k] for k in z.files if k != "meta_json"}
    if "meta_json" in z.files:
        try:
            meta = json.loads(str(z["meta_json"]))
        except Exception:
            meta = {}
    else:
        meta = {}
    meta.setdefault("out_path", str(path))
    if "duration_s_actual" not in meta:
        # Best-effort: longest channel timestamp.
        max_t = 0.0
        for k in z.files:
            if k.startswith("t_") and z[k].size > 0:
                max_t = max(max_t, float(np.max(z[k])))
        meta["duration_s_actual"] = max_t
    _summarize(snap, meta, args.moving_threshold)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description="Record an X2 deploy run on the real robot for offline debug.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out", type=str,
                   default=f"/tmp/x2_run_{time.strftime('%Y%m%d_%H%M%S')}.npz",
                   help="Output .npz path (record mode).")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Recording duration in seconds. 0 = run until Ctrl-C.")
    p.add_argument("--imu-topic", default="/aima/hal/imu/torso/state",
                   help="IMU topic. Use /aima/hal/imu/torse/state for "
                        "firmware that ships with the SDK-example typo.")
    p.add_argument("--status-period", type=float, default=1.0,
                   help="Live status print period in seconds (0 = disable).")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress the periodic status line.")
    p.add_argument("--moving-threshold", type=float, default=0.05,
                   help="Joint range (rad) above which we flag a joint as "
                        "having moved during the recording. ~3 deg.")
    p.add_argument("--note", type=str, default="",
                   help="Free-form note attached to meta_json (e.g. "
                        "'iter-4000 + minimal_v1, trial 2').")
    p.add_argument("--summarize", type=str, default=None,
                   help="Skip recording and re-print the summary on an "
                        "existing .npz produced by a prior run.")
    p.add_argument("--track-mc-mode", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Periodically poll /aimdk_5Fmsgs/srv/GetMcAction "
                        "and store the MC mode timeline alongside the "
                        "joint/IMU streams. Soft-fails if the service is "
                        "unavailable.")
    p.add_argument("--mc-poll-hz", type=float, default=5.0,
                   help="Poll rate for GetMcAction. 5 Hz is a good "
                        "compromise between transition resolution (~200 ms) "
                        "and cross-host service load.")
    args = p.parse_args()

    if args.summarize:
        return cmd_summarize(args)
    return cmd_record(args)


if __name__ == "__main__":
    raise SystemExit(main())
