"""Capture + compare the simulated robot's motion against the input PKL clip.

Sister to :mod:`gear_sonic.scripts.x2_pkl_command_source`. Runs in parallel
with ``run_x2_pkl_planner_stack.sh``: subscribes to the bridge's
ground-truth ``robot_pose`` PUB (port 5570) plus the deploy's
``x2_debug`` PUB (port 5557) for the duration of the run, dumps NPZ, and
then computes a side-by-side trajectory comparison against the input PKL
clip the source was playing back.

The captured streams come from the same bridge / deploy that drive the
sim, so we're measuring **what the robot actually did** — not what the
planner was told to do, and not what the policy commanded. That makes
the resulting comparison the cleanest possible "did the PKL replay
walk?" gate:

* ``robot_pose`` (sim-only) — MuJoCo's free-joint qpos
  ``[x, y, z, qw, qx, qy, qz]`` for the pelvis. Source of truth for
  world XY translation + heading.
* ``x2_debug`` — per-tick joint targets (``last_action``) and measured
  positions (``body_q``). Lets us check joint tracking and confirm the
  policy was actually fed sensible references.

Wire ports + topics mirror the defaults the deploy publishes (see
``gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py`` and
``gear_sonic_deploy/src/.../PublishDebug()``).

Usage::

    # Side-car: start this BEFORE run_x2_pkl_planner_stack.sh, so we
    # latch onto the deploy's PUBs as soon as they come up.
    .venv/bin/python gear_sonic/scripts/capture_pkl_replay_motion.py \\
        --pkl gear_sonic/data/motions/x2_ultra_locowalk.pkl \\
        --clip-id Loop_Forward_Walk_001__A018 \\
        --duration 30 \\
        --output-dir /tmp/pkl_replay_capture

    # After-the-fact compare (skip the live SUBs; just analyse an NPZ
    # that was already captured):
    .venv/bin/python gear_sonic/scripts/capture_pkl_replay_motion.py \\
        --pkl gear_sonic/data/motions/x2_ultra_locowalk.pkl \\
        --clip-id Loop_Forward_Walk_001__A018 \\
        --compare-only /tmp/pkl_replay_capture/capture.npz
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MOTIONBRICKS_SCRIPTS = _REPO_ROOT / "motionbricks" / "scripts"
if str(_MOTIONBRICKS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_MOTIONBRICKS_SCRIPTS))

from gear_sonic.utils.teleop.zmq.robot_pose_zmq import (  # noqa: E402
    ROBOT_POSE_DEFAULT_PUB_PORT,
    ROBOT_POSE_TOPIC,
    unpack_robot_pose,
)
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)

# Reuse the offline-replay helpers so the velocity convention used for
# comparison matches the one the source pushed onto the wire.
from replay_pkl_through_kplanner import (  # noqa: E402
    _build_clip_qpos,
    _instant_intent_from_clip,
    _quat_to_yaw_rad,
    _wxyz_to_yaw_rad,
)


log = logging.getLogger("capture_pkl_replay_motion")


# ---------------------------------------------------------------------------
# Capture buffers
# ---------------------------------------------------------------------------


@dataclass
class CaptureBuffers:
    """Mutable in-memory buffers populated by the live SUB loop."""

    pose_t: list[float] = field(default_factory=list)
    pose_xyz: list[list[float]] = field(default_factory=list)
    pose_quat_wxyz: list[list[float]] = field(default_factory=list)
    pose_sim_time: list[float] = field(default_factory=list)

    debug_t: list[float] = field(default_factory=list)
    debug_body_q: list[list[float]] = field(default_factory=list)
    debug_last_action: list[list[float]] = field(default_factory=list)

    # kplanner ``pose`` topic (port 5556): the reference the planner
    # publishes to the deploy. Captures the planner's own intent before
    # the policy gets a chance to track it; lets us isolate
    # planner-output from deploy-tracking errors.
    planner_pose_t: list[float] = field(default_factory=list)
    planner_pose_joint_pos: list[list[float]] = field(default_factory=list)
    planner_pose_root_quat_xyzw: list[list[float]] = field(default_factory=list)
    planner_pose_root_xy_world: list[list[float]] = field(default_factory=list)

    def to_npz_dict(self) -> dict[str, np.ndarray]:
        def _stack(rows: list[list[float]], width: int) -> np.ndarray:
            if not rows:
                return np.zeros((0, width), dtype=np.float64)
            return np.asarray(rows, dtype=np.float64)

        return {
            "pose_t": np.asarray(self.pose_t, dtype=np.float64),
            "pose_sim_time": np.asarray(self.pose_sim_time, dtype=np.float64),
            "pose_xyz": _stack(self.pose_xyz, 3),
            "pose_quat_wxyz": _stack(self.pose_quat_wxyz, 4),
            "debug_t": np.asarray(self.debug_t, dtype=np.float64),
            "debug_body_q": _stack(self.debug_body_q, 31),
            "debug_last_action": _stack(self.debug_last_action, 31),
            "planner_pose_t": np.asarray(
                self.planner_pose_t, dtype=np.float64),
            "planner_pose_joint_pos": _stack(
                self.planner_pose_joint_pos, 31),
            "planner_pose_root_quat_xyzw": _stack(
                self.planner_pose_root_quat_xyzw, 4),
            "planner_pose_root_xy_world": _stack(
                self.planner_pose_root_xy_world, 2),
        }


# ---------------------------------------------------------------------------
# Live ZMQ capture loop
# ---------------------------------------------------------------------------


def _capture_loop(
    duration_s: float,
    pose_host: str,
    pose_port: int,
    pose_topic: str,
    debug_host: str,
    debug_port: int,
    debug_topic: str,
    capture_debug: bool,
    planner_pose_host: str,
    planner_pose_port: int,
    planner_pose_topic: str,
    capture_planner_pose: bool,
    stop_event: threading.Event,
) -> CaptureBuffers:
    """SUB to robot_pose + optionally x2_debug + kplanner pose."""
    import zmq

    ctx = zmq.Context.instance()

    pose_sock = ctx.socket(zmq.SUB)
    pose_sock.setsockopt_string(zmq.SUBSCRIBE, pose_topic)
    pose_sock.setsockopt(zmq.RCVHWM, 500)
    pose_sock.setsockopt(zmq.LINGER, 0)
    pose_sock.connect(f"tcp://{pose_host}:{pose_port}")
    log.info(
        "SUB connected: %s tcp://%s:%d topic=%r",
        "robot_pose", pose_host, pose_port, pose_topic,
    )

    debug_sock = None
    if capture_debug:
        debug_sock = ctx.socket(zmq.SUB)
        debug_sock.setsockopt_string(zmq.SUBSCRIBE, debug_topic)
        debug_sock.setsockopt(zmq.RCVHWM, 500)
        debug_sock.setsockopt(zmq.LINGER, 0)
        debug_sock.connect(f"tcp://{debug_host}:{debug_port}")
        log.info(
            "SUB connected: %s tcp://%s:%d topic=%r",
            "x2_debug", debug_host, debug_port, debug_topic,
        )

    planner_pose_sock = None
    if capture_planner_pose:
        planner_pose_sock = ctx.socket(zmq.SUB)
        planner_pose_sock.setsockopt_string(zmq.SUBSCRIBE, planner_pose_topic)
        planner_pose_sock.setsockopt(zmq.RCVHWM, 500)
        planner_pose_sock.setsockopt(zmq.LINGER, 0)
        planner_pose_sock.connect(
            f"tcp://{planner_pose_host}:{planner_pose_port}"
        )
        log.info(
            "SUB connected: %s tcp://%s:%d topic=%r",
            "kplanner pose", planner_pose_host, planner_pose_port,
            planner_pose_topic,
        )

    poller = zmq.Poller()
    poller.register(pose_sock, zmq.POLLIN)
    if debug_sock is not None:
        poller.register(debug_sock, zmq.POLLIN)
    if planner_pose_sock is not None:
        poller.register(planner_pose_sock, zmq.POLLIN)

    buf = CaptureBuffers()
    t0 = time.monotonic()
    deadline = float("inf") if duration_s <= 0 else t0 + duration_s
    pose_warn_once = False
    debug_warn_once = False
    planner_pose_warn_once = False
    last_pose_log = t0
    while time.monotonic() < deadline:
        if stop_event.is_set():
            log.info("stop_event set; ending capture early")
            break
        events = dict(poller.poll(50))
        now = time.monotonic() - t0

        if pose_sock in events:
            try:
                raw = pose_sock.recv(zmq.NOBLOCK)
                payload = unpack_robot_pose(raw)
                qpos = payload.get("pelvis_qpos_wxyz")
                sim_time = float(payload.get("sim_time", float("nan")))
                if qpos is not None and len(qpos) == 7:
                    buf.pose_t.append(now)
                    buf.pose_sim_time.append(sim_time)
                    buf.pose_xyz.append([float(q) for q in qpos[:3]])
                    buf.pose_quat_wxyz.append([float(q) for q in qpos[3:7]])
            except zmq.Again:
                pass
            except Exception as exc:
                if not pose_warn_once:
                    log.warning("robot_pose decode err: %s", exc)
                    pose_warn_once = True

        if planner_pose_sock is not None and planner_pose_sock in events:
            try:
                raw = planner_pose_sock.recv(zmq.NOBLOCK)
                msg = unpack_message(raw, expected_topic=planner_pose_topic)
                joint = msg.fields.get("joint_pos_mj")
                root_q = msg.fields.get("root_quat_xyzw")
                root_xy = msg.fields.get("root_xy_world")
                if joint is not None and root_q is not None:
                    j_arr = np.asarray(joint, dtype=np.float64).reshape(-1)
                    q_arr = np.asarray(root_q, dtype=np.float64).reshape(-1)
                    if j_arr.size >= 31 and q_arr.size == 4:
                        buf.planner_pose_t.append(now)
                        buf.planner_pose_joint_pos.append(j_arr[:31].tolist())
                        buf.planner_pose_root_quat_xyzw.append(q_arr.tolist())
                        if root_xy is not None:
                            xy_arr = np.asarray(root_xy, dtype=np.float64).reshape(-1)
                            if xy_arr.size >= 2:
                                buf.planner_pose_root_xy_world.append(
                                    xy_arr[:2].tolist()
                                )
                            else:
                                buf.planner_pose_root_xy_world.append([0.0, 0.0])
                        else:
                            buf.planner_pose_root_xy_world.append([0.0, 0.0])
            except zmq.Again:
                pass
            except Exception as exc:
                if not planner_pose_warn_once:
                    log.warning("kplanner pose decode err: %s", exc)
                    planner_pose_warn_once = True

        if debug_sock is not None and debug_sock in events:
            try:
                raw = debug_sock.recv(zmq.NOBLOCK)
                msg = unpack_message(raw, expected_topic=debug_topic)
                # ``msg.fields`` values are numpy arrays; ``a or b`` on
                # an ndarray raises "truth value ambiguous". Fall back
                # explicitly to the legacy field name.
                meas = msg.fields.get("body_q")
                if meas is None:
                    meas = msg.fields.get("body_q_measured")
                tgt = msg.fields.get("last_action")
                if meas is not None:
                    meas_arr = np.asarray(meas, dtype=np.float64).reshape(-1)
                    if meas_arr.size == 31:
                        buf.debug_t.append(now)
                        buf.debug_body_q.append(meas_arr.tolist())
                        if tgt is not None:
                            tgt_arr = np.asarray(tgt, dtype=np.float64).reshape(-1)
                            if tgt_arr.size == 31:
                                buf.debug_last_action.append(tgt_arr.tolist())
                            else:
                                buf.debug_last_action.append([float("nan")] * 31)
                        else:
                            buf.debug_last_action.append([float("nan")] * 31)
            except zmq.Again:
                pass
            except Exception as exc:
                if not debug_warn_once:
                    log.warning("x2_debug decode err: %s", exc)
                    debug_warn_once = True

        # Periodic heartbeat so the side-car log shows progress even when
        # the deploy is slow to come up.
        if time.monotonic() - last_pose_log > 5.0:
            log.info(
                "capture progress: pose=%d debug=%d planner_pose=%d "
                "(elapsed=%.1fs)",
                len(buf.pose_t), len(buf.debug_t),
                len(buf.planner_pose_t), now,
            )
            last_pose_log = time.monotonic()

    pose_sock.close(linger=0)
    if debug_sock is not None:
        debug_sock.close(linger=0)
    if planner_pose_sock is not None:
        planner_pose_sock.close(linger=0)
    log.info(
        "capture done: pose=%d frames debug=%d frames planner_pose=%d frames",
        len(buf.pose_t), len(buf.debug_t), len(buf.planner_pose_t),
    )
    return buf


# ---------------------------------------------------------------------------
# PKL ground-truth resampling
# ---------------------------------------------------------------------------


def _load_clip_qpos(pkl_path: Path, clip_id: Optional[str]) -> tuple[np.ndarray, float, str]:
    """Load (qpos[T, 38], fps, key) from a MotionBricks PKL.

    Mirrors :func:`gear_sonic.scripts.x2_pkl_command_source._load_clip` so
    the comparison sees the exact same input the source was streaming.
    """
    raw = joblib.load(pkl_path)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{pkl_path}: expected dict of clips, got {type(raw).__name__}"
        )
    if clip_id is None:
        for k in raw.keys():
            if "forward" in str(k).lower() and "_M" not in str(k):
                clip_id = str(k)
                break
        if clip_id is None:
            clip_id = str(next(iter(raw.keys())))
    if clip_id not in raw:
        raise KeyError(
            f"clip {clip_id!r} not in {pkl_path}; "
            f"available: {list(raw.keys())[:10]}"
        )
    qpos, fps = _build_clip_qpos(raw[clip_id])
    return qpos, fps, clip_id


def _normalize_trajectory_body_frame(
    xyz: np.ndarray,
    yaw_rad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Re-express a world-frame trajectory in the BODY frame of frame 0.

    Returns:
        (xy_body[T, 2], z[T], dyaw[T])

    ``xy_body[:, 0]`` is forward (MuJoCo body +X), ``xy_body[:, 1]`` is
    lateral (MuJoCo body +Y, +Y = robot-left). ``dyaw[t]`` is heading
    change relative to ``yaw_rad[0]``, unwrapped.
    """
    if xyz.shape[0] == 0:
        return (
            np.zeros((0, 2), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
        )
    yaw_unw = np.unwrap(yaw_rad)
    yaw0 = float(yaw_unw[0])
    x0 = float(xyz[0, 0])
    y0 = float(xyz[0, 1])
    dx_world = xyz[:, 0] - x0
    dy_world = xyz[:, 1] - y0
    c, s = float(np.cos(-yaw0)), float(np.sin(-yaw0))
    x_body = c * dx_world - s * dy_world
    y_body = s * dx_world + c * dy_world
    xy_body = np.stack([x_body, y_body], axis=-1)
    return xy_body.astype(np.float64), xyz[:, 2].astype(np.float64), (yaw_unw - yaw0).astype(np.float64)


@dataclass
class TrajectorySummary:
    """Body-frame trajectory metrics that match across PKL and sim."""

    duration_s: float
    fwd_disp_m: float
    side_disp_m: float
    yaw_disp_deg: float
    path_length_m: float
    fwd_speed_mean_mps: float
    side_speed_mean_mps: float
    yaw_speed_mean_dps: float
    pelvis_z_mean_m: float
    pelvis_z_min_m: float
    pelvis_z_max_m: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "duration_s": self.duration_s,
            "fwd_disp_m": self.fwd_disp_m,
            "side_disp_m": self.side_disp_m,
            "yaw_disp_deg": self.yaw_disp_deg,
            "path_length_m": self.path_length_m,
            "fwd_speed_mean_mps": self.fwd_speed_mean_mps,
            "side_speed_mean_mps": self.side_speed_mean_mps,
            "yaw_speed_mean_dps": self.yaw_speed_mean_dps,
            "pelvis_z_mean_m": self.pelvis_z_mean_m,
            "pelvis_z_min_m": self.pelvis_z_min_m,
            "pelvis_z_max_m": self.pelvis_z_max_m,
            "n_samples": self.n_samples,
        }


def _summarize(
    t: np.ndarray,
    xy_body: np.ndarray,
    z: np.ndarray,
    dyaw_rad: np.ndarray,
) -> TrajectorySummary:
    if t.size == 0:
        return TrajectorySummary(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    duration = float(t[-1] - t[0]) if t.size > 1 else 0.0
    fwd_disp = float(xy_body[-1, 0])
    side_disp = float(xy_body[-1, 1])
    yaw_disp_deg = float(np.degrees(dyaw_rad[-1]))
    if xy_body.shape[0] > 1:
        seg = np.diff(xy_body, axis=0)
        path_length = float(np.sqrt((seg ** 2).sum(axis=-1)).sum())
    else:
        path_length = 0.0
    fwd_speed = fwd_disp / duration if duration > 0 else 0.0
    side_speed = side_disp / duration if duration > 0 else 0.0
    yaw_speed = yaw_disp_deg / duration if duration > 0 else 0.0
    return TrajectorySummary(
        duration_s=duration,
        fwd_disp_m=fwd_disp,
        side_disp_m=side_disp,
        yaw_disp_deg=yaw_disp_deg,
        path_length_m=path_length,
        fwd_speed_mean_mps=fwd_speed,
        side_speed_mean_mps=side_speed,
        yaw_speed_mean_dps=yaw_speed,
        pelvis_z_mean_m=float(np.mean(z)),
        pelvis_z_min_m=float(np.min(z)),
        pelvis_z_max_m=float(np.max(z)),
        n_samples=int(t.size),
    )


def _resample_to_grid(
    t_src: np.ndarray,
    y_src: np.ndarray,
    t_grid: np.ndarray,
) -> np.ndarray:
    """Linearly resample ``y_src(t_src)`` onto ``t_grid``.

    ``y_src`` shape: (T,) or (T, D). Out-of-range samples extrapolate
    flat (constant boundary), matching numpy's default ``left=`` /
    ``right=`` behaviour.
    """
    if t_src.size == 0:
        return np.zeros((t_grid.size,) + y_src.shape[1:], dtype=np.float64)
    if y_src.ndim == 1:
        return np.interp(t_grid, t_src, y_src)
    out = np.zeros((t_grid.size, y_src.shape[1]), dtype=np.float64)
    for j in range(y_src.shape[1]):
        out[:, j] = np.interp(t_grid, t_src, y_src[:, j])
    return out


def _mean_velocity_intent(
    qpos: np.ndarray,
    fps: float,
    start_frame: int,
    end_frame: int,
    window: int,
) -> tuple[float, float, float, float]:
    """Average per-frame velocity intent across the playback window.

    Mirrors what ``x2_pkl_command_source._spin`` would compute and
    publish: one ``_instant_intent_from_clip`` call per emitted frame.
    The mean of these is what the kplanner sees on average, so
    multiplying it by the sim's wall-clock duration is the cleanest
    ground-truth prediction for "where should the robot be by now if it
    perfectly tracked the planner's velocity commands?".

    Returns ``(yaw_rate, vel_x_lateral, vel_z_forward, hip_h)`` in the
    same convention the source uses.
    """
    if end_frame <= start_frame:
        return 0.0, 0.0, 0.0, float(qpos[0, 2])
    vals = np.zeros((end_frame - start_frame, 4), dtype=np.float64)
    for i, src_idx in enumerate(range(start_frame, end_frame)):
        vals[i] = _instant_intent_from_clip(qpos, fps, src_idx, window=window)
    mean = vals.mean(axis=0)
    return float(mean[0]), float(mean[1]), float(mean[2]), float(mean[3])


def _integrate_velocity_in_body_frame(
    yaw_rate: float,
    vel_x_lateral: float,
    vel_z_forward: float,
    t_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate a CONSTANT body-frame velocity over ``t_grid``.

    Body-frame convention matches the kplanner / planner_core /
    ``_instant_intent_from_clip``:

    * ``vel_z_forward`` advances MuJoCo body +X
    * ``vel_x_lateral`` advances MuJoCo body +Y (robot-left)
    * ``yaw_rate`` rotates around world +Z

    Returns ``(xy_body[T, 2], dyaw[T])`` where xy is in the body frame
    of t=0 (so xy[0] = (0, 0) and dyaw[0] = 0).
    """
    if t_grid.size == 0:
        return np.zeros((0, 2)), np.zeros(0)
    dt = np.diff(t_grid, prepend=t_grid[0])
    # Heading at each tick (relative to t=0).
    dyaw = np.cumsum(yaw_rate * dt)
    dyaw = dyaw - dyaw[0]
    # Per-tick world-frame increment in body-frame-of-t0:
    #   Δfwd_world = vel_z * cos(dyaw) - vel_x * sin(dyaw)
    #   Δside_world = vel_z * sin(dyaw) + vel_x * cos(dyaw)
    # (forward in current body frame projects to body-frame-of-t0 via
    # the rotation of the heading change.)
    dx = (vel_z_forward * np.cos(dyaw) - vel_x_lateral * np.sin(dyaw)) * dt
    dy = (vel_z_forward * np.sin(dyaw) + vel_x_lateral * np.cos(dyaw)) * dt
    x = np.cumsum(dx)
    y = np.cumsum(dy)
    return np.stack([x - x[0], y - y[0]], axis=-1), dyaw


def _compute_compare(
    pkl_qpos: np.ndarray,
    pkl_fps: float,
    pkl_start_frame: int,
    pkl_num_frames: Optional[int],
    pkl_loop: bool,
    pkl_velocity_window: int,
    pose_t: np.ndarray,
    pose_xyz: np.ndarray,
    pose_quat_wxyz: np.ndarray,
    planner_pose_t: Optional[np.ndarray] = None,
    planner_pose_root_quat_xyzw: Optional[np.ndarray] = None,
    planner_pose_root_xy_world: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute pelvis-trajectory comparison metrics.

    Two ground-truth references are computed:

    1. **Velocity-integrated** (the primary verdict gate). The source
       publishes per-frame velocities; integrating their MEAN over the
       sim's wall-clock duration tells us where the robot would be if
       it perfectly tracked those velocity commands. This is what the
       kplanner-deploy chain actually *should* reproduce.
    2. **Literal PKL trajectory** (informational). The raw body-frame
       trajectory of the PKL clip itself. Useful for visualisation but
       can mislead when ``--loop`` is on (cycles wrap back to the
       origin) or when the source's emission rate ≠ ``fps``.

    All trajectories are projected into the BODY frame of their
    starting orientation so the comparison is invariant to absolute
    world placement / heading.
    """
    if pose_t.size == 0 or pkl_qpos.shape[0] < 2:
        empty = _summarize(
            np.zeros(0), np.zeros((0, 2)), np.zeros(0), np.zeros(0),
        ).to_dict()
        return {
            "pkl_velocity_integrated": empty,
            "pkl_literal": empty,
            "sim": empty,
            "delta": {},
            "trace": {},
            "mean_velocity_intent": (0.0, 0.0, 0.0, 0.0),
        }

    # ── sim trajectory in body frame ────────────────────────────────
    sim_t = pose_t.astype(np.float64) - float(pose_t[0])
    sim_yaw = _wxyz_to_yaw_rad(pose_quat_wxyz)
    sim_xy_body, sim_z, sim_dyaw = _normalize_trajectory_body_frame(
        pose_xyz, sim_yaw,
    )
    sim_summary = _summarize(sim_t, sim_xy_body, sim_z, sim_dyaw)

    # ── PKL playback window ─────────────────────────────────────────
    end_frame = pkl_qpos.shape[0]
    if pkl_num_frames is not None:
        end_frame = min(pkl_qpos.shape[0], pkl_start_frame + pkl_num_frames)
    n_clip = end_frame - pkl_start_frame

    # ── Ground truth #1: velocity-integrated expected trajectory ────
    mean_intent = _mean_velocity_intent(
        pkl_qpos, pkl_fps, pkl_start_frame, end_frame, pkl_velocity_window,
    )
    yaw_rate, vel_x_lat, vel_z_fwd, hip_h_ref = mean_intent
    pkl_vel_xy, pkl_vel_dyaw = _integrate_velocity_in_body_frame(
        yaw_rate, vel_x_lat, vel_z_fwd, sim_t,
    )
    pkl_vel_z = np.full_like(sim_t, hip_h_ref)
    pkl_vel_summary = _summarize(sim_t, pkl_vel_xy, pkl_vel_z, pkl_vel_dyaw)

    # ── Ground truth #2: literal PKL trajectory (informational) ─────
    if n_clip >= 2:
        clip_window = pkl_qpos[pkl_start_frame:end_frame]
        if pkl_loop:
            # Stitch cycles together in body frame so the trajectory
            # extends naturally instead of wrapping back to origin.
            n_ticks = int(np.ceil(sim_t[-1] * pkl_fps)) + 1
            n_cycles = int(np.ceil(n_ticks / n_clip)) + 1
            t_lit = np.arange(n_clip * n_cycles) / max(pkl_fps, 1e-6)
            xyz_world = np.tile(clip_window[:, :3], (n_cycles, 1)).astype(np.float64)
            rot_world = np.tile(clip_window[:, 3:7], (n_cycles, 1)).astype(np.float64)
            # Add per-cycle offset (in world frame).
            per_cycle = clip_window[-1, :3] - clip_window[0, :3]
            for c in range(1, n_cycles):
                xyz_world[c * n_clip : (c + 1) * n_clip, :] += c * per_cycle
            yaw_world = _wxyz_to_yaw_rad(rot_world)
        else:
            t_lit = np.arange(n_clip) / max(pkl_fps, 1e-6)
            xyz_world = clip_window[:, :3].astype(np.float64)
            yaw_world = _wxyz_to_yaw_rad(clip_window[:, 3:7].astype(np.float64))
        pkl_lit_xy, pkl_lit_z, pkl_lit_dyaw = _normalize_trajectory_body_frame(
            xyz_world, yaw_world,
        )
        # Resample the literal PKL trajectory onto the sim's time grid
        # (the sim and PKL play at independent rates; resampling makes
        # the printed summary directly comparable to the sim's window).
        if t_lit.size >= 2:
            pkl_lit_xy_on_sim = _resample_to_grid(t_lit, pkl_lit_xy, sim_t)
            pkl_lit_z_on_sim = _resample_to_grid(t_lit, pkl_lit_z, sim_t)
            pkl_lit_dyaw_on_sim = _resample_to_grid(t_lit, pkl_lit_dyaw, sim_t)
        else:
            pkl_lit_xy_on_sim = np.zeros_like(sim_xy_body)
            pkl_lit_z_on_sim = np.zeros_like(sim_z)
            pkl_lit_dyaw_on_sim = np.zeros_like(sim_dyaw)
        pkl_lit_summary = _summarize(
            sim_t, pkl_lit_xy_on_sim, pkl_lit_z_on_sim, pkl_lit_dyaw_on_sim,
        )
    else:
        pkl_lit_xy_on_sim = np.zeros_like(sim_xy_body)
        pkl_lit_z_on_sim = np.zeros_like(sim_z)
        pkl_lit_dyaw_on_sim = np.zeros_like(sim_dyaw)
        pkl_lit_summary = _summarize(
            sim_t, pkl_lit_xy_on_sim, pkl_lit_z_on_sim, pkl_lit_dyaw_on_sim,
        )

    delta = {
        "fwd_disp_m": sim_summary.fwd_disp_m - pkl_vel_summary.fwd_disp_m,
        "side_disp_m": sim_summary.side_disp_m - pkl_vel_summary.side_disp_m,
        "yaw_disp_deg": sim_summary.yaw_disp_deg - pkl_vel_summary.yaw_disp_deg,
        "fwd_speed_mean_mps": sim_summary.fwd_speed_mean_mps - pkl_vel_summary.fwd_speed_mean_mps,
        "side_speed_mean_mps": sim_summary.side_speed_mean_mps - pkl_vel_summary.side_speed_mean_mps,
        "yaw_speed_mean_dps": sim_summary.yaw_speed_mean_dps - pkl_vel_summary.yaw_speed_mean_dps,
        "pelvis_z_mean_m": sim_summary.pelvis_z_mean_m - pkl_vel_summary.pelvis_z_mean_m,
        "fwd_disp_ratio": (
            sim_summary.fwd_disp_m / pkl_vel_summary.fwd_disp_m
            if abs(pkl_vel_summary.fwd_disp_m) > 1e-3 else float("nan")
        ),
        "yaw_disp_ratio": (
            sim_summary.yaw_disp_deg / pkl_vel_summary.yaw_disp_deg
            if abs(pkl_vel_summary.yaw_disp_deg) > 5.0 else float("nan")
        ),
    }

    # ── Optional: planner's OWN published trajectory (planner output)
    planner_summary: Optional[dict[str, Any]] = None
    planner_trace: Optional[dict[str, np.ndarray]] = None
    if (
        planner_pose_t is not None
        and planner_pose_t.size > 1
        and planner_pose_root_quat_xyzw is not None
        and planner_pose_root_quat_xyzw.shape[0] == planner_pose_t.size
    ):
        # The kplanner publishes ``root_xy_world`` as an advisory field
        # but the wire-critical channel is ``root_quat_xyzw`` + joints
        # (the deploy integrates pos itself via the body model). For
        # the comparison we want the planner's *intended* world XY too,
        # which we get from ``root_xy_world`` if it was published with
        # non-zero values; otherwise we fall back to integrating the
        # planner's local-frame motion via quat.
        p_t = planner_pose_t.astype(np.float64) - float(planner_pose_t[0])
        p_quat_xyzw = planner_pose_root_quat_xyzw.astype(np.float64)
        # Convert xyzw -> yaw_rad.
        p_yaw = _quat_to_yaw_rad(p_quat_xyzw)
        if (
            planner_pose_root_xy_world is not None
            and planner_pose_root_xy_world.shape[0] == planner_pose_t.size
            and np.any(np.abs(planner_pose_root_xy_world) > 1e-6)
        ):
            p_xyz = np.zeros((p_t.size, 3), dtype=np.float64)
            p_xyz[:, :2] = planner_pose_root_xy_world.astype(np.float64)
            p_xyz[:, 2] = float(mean_intent[3])
        else:
            # No advisory XY; the planner only published quat. Set xy=0.
            p_xyz = np.zeros((p_t.size, 3), dtype=np.float64)
            p_xyz[:, 2] = float(mean_intent[3])
        p_xy_body, p_z, p_dyaw = _normalize_trajectory_body_frame(p_xyz, p_yaw)
        planner_trajectory = _summarize(p_t, p_xy_body, p_z, p_dyaw)
        planner_summary = planner_trajectory.to_dict()
        planner_trace = {
            "planner_t": p_t,
            "planner_xy_body": p_xy_body,
            "planner_z": p_z,
            "planner_dyaw": p_dyaw,
        }

    trace = {
        "sim_t": sim_t,
        "sim_xy_body": sim_xy_body,
        "sim_z": sim_z,
        "sim_dyaw": sim_dyaw,
        "pkl_vel_xy": pkl_vel_xy,
        "pkl_vel_dyaw": pkl_vel_dyaw,
        "pkl_lit_xy_on_sim": pkl_lit_xy_on_sim,
        "pkl_lit_z_on_sim": pkl_lit_z_on_sim,
        "pkl_lit_dyaw_on_sim": pkl_lit_dyaw_on_sim,
    }
    if planner_trace is not None:
        trace.update(planner_trace)

    return {
        "pkl_velocity_integrated": pkl_vel_summary.to_dict(),
        "pkl_literal": pkl_lit_summary.to_dict(),
        "sim": sim_summary.to_dict(),
        "planner_published": planner_summary,
        "delta": delta,
        "mean_velocity_intent": mean_intent,
        "trace": trace,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_table(report: dict[str, Any]) -> None:
    pkl_vel = report["pkl_velocity_integrated"]
    pkl_lit = report["pkl_literal"]
    sim = report["sim"]
    delta = report["delta"]
    intent = report.get("mean_velocity_intent", (0.0, 0.0, 0.0, 0.0))
    print("")
    print("=" * 92)
    print("  PKL replay comparison — pelvis trajectory in body frame")
    print(
        "  mean source intent  yaw_rate={:+.3f} rad/s  vel_x(lat)={:+.3f}  "
        "vel_z(fwd)={:+.3f}  hip_h={:.3f}".format(*intent)
    )
    print("=" * 92)
    header = (
        f"{'metric':<28}{'pkl_vel (expected)':>20}"
        f"{'sim (robot)':>16}{'delta':>14}{'ratio':>8}"
    )
    print(header)
    print("-" * 92)
    fmt_rows = [
        ("duration_s",          "duration_s",           ""),
        ("fwd_disp_m",          "fwd_disp_m",           "fwd_disp_ratio"),
        ("side_disp_m",         "side_disp_m",          ""),
        ("yaw_disp_deg",        "yaw_disp_deg",         "yaw_disp_ratio"),
        ("path_length_m",       "path_length_m",        ""),
        ("fwd_speed_mean_mps",  "fwd_speed_mean_mps",   ""),
        ("side_speed_mean_mps", "side_speed_mean_mps",  ""),
        ("yaw_speed_mean_dps",  "yaw_speed_mean_dps",   ""),
        ("pelvis_z_mean_m",     "pelvis_z_mean_m",      ""),
        ("pelvis_z_min_m",      "pelvis_z_min_m",       ""),
        ("pelvis_z_max_m",      "pelvis_z_max_m",       ""),
        ("n_samples",           "n_samples",            ""),
    ]
    for label, key, ratio_key in fmt_rows:
        pv = pkl_vel.get(key, float("nan"))
        sv = sim.get(key, float("nan"))
        if isinstance(pv, (int, float)) and isinstance(sv, (int, float)):
            dv = delta.get(key, sv - pv)
        else:
            dv = float("nan")
        rv = delta.get(ratio_key, float("nan")) if ratio_key else float("nan")
        if key == "n_samples":
            print(
                f"{label:<28}{int(pv):>20d}{int(sv):>16d}{int(sv - pv):>14d}"
                f"{'':>8}"
            )
        else:
            ratio_str = (
                f"{rv:>8.2f}" if (isinstance(rv, float) and np.isfinite(rv))
                else f"{'':>8}"
            )
            print(
                f"{label:<28}{pv:>20.4f}{sv:>16.4f}{dv:>+14.4f}{ratio_str}"
            )
    print("-" * 92)
    print("  [reference] literal PKL trajectory over the SAME sim window "
          "(informational):")
    print(
        f"    fwd_disp_m={pkl_lit['fwd_disp_m']:+.4f}  "
        f"side_disp_m={pkl_lit['side_disp_m']:+.4f}  "
        f"yaw_disp_deg={pkl_lit['yaw_disp_deg']:+.2f}  "
        f"path_length_m={pkl_lit['path_length_m']:.4f}"
    )
    planner = report.get("planner_published")
    if planner is not None:
        print("-" * 92)
        print(
            "  [planner output] kplanner's OWN published pose stream "
            "(isolates planner vs deploy):"
        )
        print(
            f"    pelvis yaw_disp_deg={planner['yaw_disp_deg']:+.2f}  "
            f"(sim={sim['yaw_disp_deg']:+.2f})   "
            f"if these match  -> deploy follows planner;  "
            f"if planner=0 and sim>>0 -> deploy is spinning the robot"
        )
        print(
            f"    pelvis fwd_disp_m={planner['fwd_disp_m']:+.4f}  "
            f"side_disp_m={planner['side_disp_m']:+.4f}  "
            f"(note: planner only publishes root_quat + joints; XY is "
            f"advisory and may be 0)"
        )
    print("-" * 92)

    # Verdict gates on the velocity-integrated expected trajectory.
    expect_fwd = pkl_vel["fwd_disp_m"]
    expect_yaw = pkl_vel["yaw_disp_deg"]
    if abs(expect_fwd) > 0.05:
        ratio = delta.get("fwd_disp_ratio", float("nan"))
        if not np.isfinite(ratio):
            verdict = "indeterminate (sim too short to compare)"
        elif ratio >= 0.7:
            verdict = "OK — robot tracked planner velocity command"
        elif ratio >= 0.3:
            verdict = "WEAK — robot moved but undershot planner velocity"
        elif ratio > -0.1:
            verdict = "FAIL — robot barely moved (planner velocity not realized)"
        else:
            verdict = "WRONG WAY — robot moved opposite the planner velocity"
        print(f"  verdict (forward): {verdict}")
    elif abs(expect_yaw) > 5.0:
        ratio = delta.get("yaw_disp_ratio", float("nan"))
        if not np.isfinite(ratio) or ratio < 0.3:
            verdict = "FAIL — robot did not realize planner yaw command"
        elif ratio >= 0.7:
            verdict = "OK — robot tracked planner yaw command"
        else:
            verdict = "WEAK — robot turned but undershot planner yaw command"
        print(f"  verdict (yaw): {verdict}")
    else:
        print("  verdict: input pkl is mostly stationary; nothing to gate on.")
    print("=" * 92)
    print("")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="capture_pkl_replay_motion",
        description=(
            "Capture the simulated robot's pelvis trajectory + joint "
            "command/measured pairs while ``run_x2_pkl_planner_stack.sh`` is "
            "running, then compare against the input PKL clip."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--pkl", type=Path, required=True,
        help="Input PKL the pkl_command_source is streaming. Used as the "
             "ground-truth reference for the comparison.",
    )
    p.add_argument(
        "--clip-id", type=str, default=None,
        help="Clip id inside --pkl. Must match the id passed to "
             "x2_pkl_command_source so we compare apples to apples.",
    )
    p.add_argument(
        "--start-frame", type=int, default=0,
        help="Mirror --start-frame from x2_pkl_command_source.",
    )
    p.add_argument(
        "--num-frames", type=int, default=None,
        help="Mirror --num-frames from x2_pkl_command_source.",
    )
    p.add_argument(
        "--loop", action="store_true",
        help="Mirror --loop from x2_pkl_command_source (extends the PKL "
             "reference timeline by cycling instead of clamping at the end).",
    )
    p.add_argument(
        "--velocity-window", type=int, default=8,
        help="Mirror --velocity-window from x2_pkl_command_source so "
             "the comparison's mean-intent matches what the source published.",
    )
    p.add_argument(
        "--duration", type=float, default=0.0,
        help="Seconds to capture (0 = until Ctrl-C or compare-only).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="If set, save capture.npz + compare.json into this dir.",
    )
    p.add_argument(
        "--pose-host", type=str, default="127.0.0.1",
        help="Host where the bridge's robot_pose PUB is reachable.",
    )
    p.add_argument(
        "--pose-port", type=int, default=ROBOT_POSE_DEFAULT_PUB_PORT,
        help="ZMQ port for the bridge's robot_pose PUB (default 5570).",
    )
    p.add_argument(
        "--pose-topic", type=str, default=ROBOT_POSE_TOPIC,
        help="Topic on the robot_pose PUB (default 'robot_pose').",
    )
    p.add_argument(
        "--debug-host", type=str, default="127.0.0.1",
        help="Host where the deploy's x2_debug PUB is reachable.",
    )
    p.add_argument(
        "--debug-port", type=int, default=5557,
        help="ZMQ port for the deploy's x2_debug PUB (default 5557).",
    )
    p.add_argument(
        "--debug-topic", type=str, default="x2_debug",
        help="Topic on the x2_debug PUB (default 'x2_debug').",
    )
    p.add_argument(
        "--no-debug", action="store_true",
        help="Skip the x2_debug capture (smaller NPZ, no joint-tracking metrics).",
    )
    p.add_argument(
        "--planner-pose-host", type=str, default="127.0.0.1",
        help="Host where the kplanner's pose PUB is reachable.",
    )
    p.add_argument(
        "--planner-pose-port", type=int, default=5556,
        help="ZMQ port for the kplanner's pose PUB (default 5556 — direct-"
             "to-deploy mode used by run_x2_pkl_planner_stack.sh).",
    )
    p.add_argument(
        "--planner-pose-topic", type=str, default="pose",
        help="Topic on the kplanner pose PUB (default 'pose').",
    )
    p.add_argument(
        "--no-planner-pose", action="store_true",
        help="Skip the kplanner pose capture (planner-vs-robot isolation "
             "won't be possible without it).",
    )
    p.add_argument(
        "--compare-only", type=Path, default=None,
        help="Skip live capture; load this NPZ and just print + write the "
             "comparison report.",
    )
    p.add_argument(
        "--no-plot", action="store_true",
        help="Skip the trajectory.png render even when --output-dir is set.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Debug-level logging.",
    )
    return p.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)

    if not args.pkl.is_file():
        log.error("--pkl not found: %s", args.pkl)
        return 1

    # Load the PKL up front so we fail fast on a bad --clip-id before
    # we burn time waiting for the deploy to come up.
    try:
        pkl_qpos, pkl_fps, clip_id = _load_clip_qpos(args.pkl, args.clip_id)
    except (KeyError, ValueError) as exc:
        log.error("failed to load clip: %s", exc)
        return 1
    log.info(
        "PKL ground truth: %s clip=%r frames=%d fps=%.1f duration=%.2fs",
        args.pkl.name, clip_id, pkl_qpos.shape[0], pkl_fps,
        pkl_qpos.shape[0] / max(pkl_fps, 1.0),
    )

    planner_pose_t: Optional[np.ndarray] = None
    planner_pose_root_quat_xyzw: Optional[np.ndarray] = None
    planner_pose_root_xy_world: Optional[np.ndarray] = None

    # ---- branch 1: pure offline comparison -----------------------------
    if args.compare_only is not None:
        if not args.compare_only.is_file():
            log.error("--compare-only NPZ not found: %s", args.compare_only)
            return 1
        log.info("loading captured NPZ: %s", args.compare_only)
        npz = np.load(args.compare_only)
        pose_t = npz["pose_t"]
        pose_xyz = npz["pose_xyz"]
        pose_quat_wxyz = npz["pose_quat_wxyz"]
        if "planner_pose_t" in npz.files:
            planner_pose_t = npz["planner_pose_t"]
            planner_pose_root_quat_xyzw = npz["planner_pose_root_quat_xyzw"]
            planner_pose_root_xy_world = npz["planner_pose_root_xy_world"]
    else:
        # ---- branch 2: live capture --------------------------------------
        stop_event = threading.Event()

        def _on_signal(signum: int, _frame: object) -> None:
            log.info("signal %d -> stopping capture", signum)
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, _on_signal)

        buf = _capture_loop(
            duration_s=args.duration,
            pose_host=args.pose_host,
            pose_port=args.pose_port,
            pose_topic=args.pose_topic,
            debug_host=args.debug_host,
            debug_port=args.debug_port,
            debug_topic=args.debug_topic,
            capture_debug=(not args.no_debug),
            planner_pose_host=args.planner_pose_host,
            planner_pose_port=args.planner_pose_port,
            planner_pose_topic=args.planner_pose_topic,
            capture_planner_pose=(not args.no_planner_pose),
            stop_event=stop_event,
        )

        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            npz_path = args.output_dir / "capture.npz"
            np.savez(npz_path, **buf.to_npz_dict())
            log.info("wrote capture NPZ: %s", npz_path)

        pose_t = np.asarray(buf.pose_t, dtype=np.float64)
        pose_xyz = np.asarray(buf.pose_xyz, dtype=np.float64) if buf.pose_xyz else np.zeros((0, 3))
        pose_quat_wxyz = np.asarray(buf.pose_quat_wxyz, dtype=np.float64) if buf.pose_quat_wxyz else np.zeros((0, 4))
        if buf.planner_pose_t:
            planner_pose_t = np.asarray(buf.planner_pose_t, dtype=np.float64)
            planner_pose_root_quat_xyzw = np.asarray(
                buf.planner_pose_root_quat_xyzw, dtype=np.float64,
            )
            planner_pose_root_xy_world = np.asarray(
                buf.planner_pose_root_xy_world, dtype=np.float64,
            )

    if pose_t.size < 2:
        log.error(
            "no robot_pose samples captured (got %d). Is the deploy "
            "actually running with the default --robot-pose-pub-port 5570?",
            int(pose_t.size),
        )
        return 1

    report = _compute_compare(
        pkl_qpos=pkl_qpos,
        pkl_fps=pkl_fps,
        pkl_start_frame=args.start_frame,
        pkl_num_frames=args.num_frames,
        pkl_loop=args.loop,
        pkl_velocity_window=args.velocity_window,
        pose_t=pose_t,
        pose_xyz=pose_xyz,
        pose_quat_wxyz=pose_quat_wxyz,
        planner_pose_t=planner_pose_t,
        planner_pose_root_quat_xyzw=planner_pose_root_quat_xyzw,
        planner_pose_root_xy_world=planner_pose_root_xy_world,
    )

    _print_table(report)

    if args.output_dir is not None:
        # Strip numpy arrays from the trace before JSON-dumping the
        # summary; the NPZ already has them.
        trace = report.pop("trace", None)
        json_path = args.output_dir / "compare.json"
        json_path.write_text(json.dumps(
            {
                "pkl_clip_id": clip_id,
                "pkl_fps": pkl_fps,
                "pkl_path": str(args.pkl),
                **report,
            },
            indent=2,
        ))
        log.info("wrote compare JSON: %s", json_path)
        if trace is not None:
            trace_path = args.output_dir / "compare_trace.npz"
            np.savez(trace_path, **{k: np.asarray(v) for k, v in trace.items()})
            log.info("wrote trace NPZ: %s", trace_path)
            if not args.no_plot:
                try:
                    _render_plot(
                        trace,
                        clip_id=clip_id,
                        verdict_ratio=report.get("delta", {}).get(
                            "fwd_disp_ratio", float("nan"),
                        ),
                        out_path=args.output_dir / "trajectory.png",
                    )
                except Exception as exc:
                    log.warning("plot render failed: %s", exc)

    return 0


def _render_plot(
    trace: dict[str, np.ndarray],
    clip_id: str,
    verdict_ratio: float,
    out_path: Path,
) -> None:
    """Render the 3-panel trajectory comparison PNG.

    Matplotlib is imported lazily so the live capture path doesn't pay
    its import cost when ``--no-plot`` is set (or when matplotlib is
    not installed at all).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sim_t = np.asarray(trace["sim_t"])
    sim_xy = np.asarray(trace["sim_xy_body"])
    pkl_vel_xy = np.asarray(trace["pkl_vel_xy"])
    pkl_lit_xy = np.asarray(trace["pkl_lit_xy_on_sim"])
    sim_dyaw = np.asarray(trace["sim_dyaw"])
    pkl_vel_dyaw = np.asarray(trace["pkl_vel_dyaw"])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(pkl_vel_xy[:, 0], pkl_vel_xy[:, 1], "g-",
            label="planner velocity (expected)", linewidth=2)
    ax.plot(pkl_lit_xy[:, 0], pkl_lit_xy[:, 1], "b--",
            label="literal PKL (reference)", alpha=0.6)
    ax.plot(sim_xy[:, 0], sim_xy[:, 1], "r-",
            label="sim (robot)", linewidth=2)
    ax.scatter([0], [0], c="k", marker="o", s=50, zorder=5, label="start")
    ax.set_xlabel("forward (m, body frame of t0)")
    ax.set_ylabel("lateral (m, body frame of t0; +Y=left)")
    ax.set_title("Pelvis trajectory (top-down, body frame)")
    ax.legend()
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(sim_t, pkl_vel_xy[:, 0], "g-",
            label="planner velocity (expected)", linewidth=2)
    ax.plot(sim_t, pkl_lit_xy[:, 0], "b--",
            label="literal PKL (reference)", alpha=0.6)
    ax.plot(sim_t, sim_xy[:, 0], "r-",
            label="sim (robot)", linewidth=2)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("forward displacement (m)")
    ax.set_title("Forward displacement over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(sim_t, np.degrees(pkl_vel_dyaw), "g-",
            label="planner velocity (expected)", linewidth=2)
    ax.plot(sim_t, np.degrees(sim_dyaw), "r-",
            label="sim (robot)", linewidth=2)
    ax.set_xlabel("wall-clock time (s)")
    ax.set_ylabel("yaw change (deg)")
    ax.set_title("Yaw change over time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    verdict_str = (
        f"ratio={verdict_ratio:.2f}"
        if isinstance(verdict_ratio, float) and np.isfinite(verdict_ratio)
        else "ratio=N/A"
    )
    plt.suptitle(
        f"PKL Replay — sim robot vs planner velocity   clip='{clip_id}'   "
        f"fwd_disp_{verdict_str}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote trajectory plot: %s", out_path)


if __name__ == "__main__":
    raise SystemExit(main())
