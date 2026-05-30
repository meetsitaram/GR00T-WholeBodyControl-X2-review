"""Neural kinematic planner daemon for AgiBot X2 Ultra.

Streams 31-DOF reference body poses + root quaternion at 50 Hz on the
``body_pose`` (or legacy ``pose``) ZMQ topic, the same wire format as
``x2_heuristic_planner.py`` -- so downstream subscribers (recorder, MuJoCo
viewer, C++ deploy) are protocol-compatible.

Source of motion: a trained MotionBricks stack
(``motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/``). Source
of velocity intent: hard-coded ``INTENT_VELOCITY_MAP`` keyed on ``(intent,
magnitude)`` from the Quest3 / scripted command sources.

Command surface (any combination, mirrors the heuristic daemon's UX):
  - ``--demo PATH.yaml`` : scripted command sequence
  - ``--keyboard``       : interactive keyboard mode (TTY only)
  - ``--zmq-cmd-port``   : ZMQ SUB on ``planner_cmd`` topic for external drive

Architecture (per
``docs/source/references/x2_quest3_planner_stack_architecture.md`` updated
for kplanner):

  - Main thread runs the 50 Hz publish loop (``PosePublisher.publish``),
    pulling one frame per tick out of the ``NeuralPlannerCore`` ring buffer.
  - Worker thread monitors the ring-buffer occupancy. When it drops below
    ``--replan-threshold-frames`` it calls
    ``NeuralPlannerCore.replan_with_velocity(target)`` to refill. ``predict()``
    can take 5--15 ms on GPU; running it in a worker keeps the publish thread
    on its 20 ms tick.
  - The current ``target_local_root_values`` is updated atomically from the
    latest received ``LocomotionCommand`` via ``INTENT_VELOCITY_MAP``.

Run from the repo root::

    .venv/bin/python -m gear_sonic.scripts.x2_kplanner \\
        --body-pose-port 5555 \\
        --zmq-cmd-host 127.0.0.1 --zmq-cmd-port 5557

See ``run_x2_quest3_planner_stack.sh --planner kplanner`` for the integrated
launcher that wires this up with the recorder + deploy + Quest3 stack.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.planner.state_machine import (  # noqa: E402
    LocomotionCommand,
    OUTPUT_FPS,
    PlannerState,
    StreamFrame,
    build_pose_payload,
    commands_from_yaml,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402


log = logging.getLogger("x2_kplanner")


# ---------------------------------------------------------------------------
# Intent -> velocity dispatcher. The neural model consumes a continuous
# 4-D vector ``(yaw_rate_rad_s, vel_x_m_s, vel_z_m_s, hip_height_m)``.
#
# CHANNEL CONVENTION (verified against the X2 + G1 mujoco converter and
# the LocalRootLocalBody motion-rep, see
# ``motionbricks/scripts/probe_root_constraint_modes.py``):
#
#   * ``vel_x`` = motion-rep X = MuJoCo Y = **lateral** (positive = +X
#     in the motion-rep frame = robot's LEFT after canonicalization).
#   * ``vel_z`` = motion-rep Z = MuJoCo X = **forward** (positive = +Z
#     in the motion-rep frame = robot's FORWARD after canonicalization).
#
# The earlier docstring claim that "vel_z is lateral" was WRONG and the
# original ``_BASE_VELOCITY`` table fed forward speed into vel_x, which
# the model interpreted as a lateral command. Combined with the
# velocity-only target masking in NeuralPlannerCore (now fixed), this
# explains the historical "robot doesn't walk forward" symptom.
#
# Vocabulary contract with ``quest3_manager_x2``: the manager's
# ``IntentDecoder`` was authored for the heuristic planner's curated
# motion bins (``deg_15/30/45/90`` turns, ``quarter_ft / half_ft /
# one_ft`` strides). The kplanner does NOT honour those labels as
# literal angles or distances -- for the neural model they are
# intensity buckets scaling a continuous velocity. To avoid locking
# the kplanner into the heuristic's labels we split the table:
#
#   ``_BASE_VELOCITY``: direction-explicit intent -> 1× velocity vector.
#                       Direction is encoded in the intent name
#                       (``fwd_step`` vs ``back_step``, ``turn_left``
#                       vs ``turn_right``, etc.).
#   ``_TRANSLATIONAL_SCALE`` / ``_TURN_SCALE``: magnitude string ->
#                       scalar multiplier (``default`` = 1.0). Unknown
#                       magnitudes fall back to 1.0 so manager-side
#                       vocabulary additions don't silently idle.
#
# ``INTENT_VELOCITY_MAP`` is derived at import time from the dispatcher
# for docs / debug dumps. The two surfaces are guaranteed to agree
# because they go through the same ``_resolve_velocity()`` path.
#
# Numbers are intentionally conservative on the kinematic side; we can
# raise them once the deploy is verified stable on each direction.
# ---------------------------------------------------------------------------

_WALK_SPEED_MPS: float = 0.5
_FAST_WALK_SPEED_MPS: float = 0.9
_SIDE_SPEED_MPS: float = 0.4
_BACK_SPEED_MPS: float = 0.35
# Per-step yaw rate baseline; magnitude scalars in ``_TURN_SCALE`` rescale.
_TURN_15_RAD_S: float = 0.5
_TURN_30_RAD_S: float = 1.0
_TURN_45_RAD_S: float = 1.5
_TURN_90_RAD_S: float = 3.0
# Default hip height for the X2 (~0.95 m matches idle_stand).
_HIP_HEIGHT_M: float = 0.95

# Returned for any (intent, magnitude) the kplanner has no velocity
# meaning for (``hold_torso``, ``lean_*``, ``torso_*``, ``crouch``,
# unrecognised intents). The publisher's idle-gate compares against this
# tuple to decide whether to freeze on the static anchor.
_IDLE_INTENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, _HIP_HEIGHT_M)

# Direction-explicit 1× velocity vector per intent. Magnitude is applied
# separately via ``_TRANSLATIONAL_SCALE`` / ``_TURN_SCALE`` below.
_BASE_VELOCITY: dict[str, tuple[float, float, float, float]] = {
    # (yaw_rate, vel_x=lateral, vel_z=forward, hip_h).
    "idle":       (0.0,             0.0,                  0.0,                _HIP_HEIGHT_M),
    "fwd_step":   (0.0,             0.0,                  _WALK_SPEED_MPS,    _HIP_HEIGHT_M),
    "back_step":  (0.0,             0.0,                 -_BACK_SPEED_MPS,    _HIP_HEIGHT_M),
    "side_left":  (0.0,             _SIDE_SPEED_MPS,      0.0,                _HIP_HEIGHT_M),
    "side_right": (0.0,            -_SIDE_SPEED_MPS,      0.0,                _HIP_HEIGHT_M),
    "turn_left":  ( _TURN_45_RAD_S, 0.0,                  0.0,                _HIP_HEIGHT_M),
    "turn_right": (-_TURN_45_RAD_S, 0.0,                  0.0,                _HIP_HEIGHT_M),
}

# Translational magnitude -> multiplier on (vx, vy). ``default`` is 1.0
# (what the L-stick single-press emits today); the ``*_ft`` rows are
# carry-over from the heuristic vocabulary and treated as intensity
# buckets, NOT literal foot distances. Unknown magnitudes default to 1.0.
_TRANSLATIONAL_SCALE: dict[str, float] = {
    "default":    1.0,
    "stand":      0.0,
    "quarter_ft": 0.5,
    "half_ft":    1.0,
    "one_ft":     1.5,
}

# Rotational magnitude -> multiplier on yaw_rate. Baseline = 1.0 = the
# ``deg_45`` bucket so the legacy ``_TURN_45_RAD_S`` constant remains
# the natural unit. ``deg_*`` labels are intensity buckets, NOT literal
# angles. Unknown magnitudes default to 1.0.
_TURN_SCALE: dict[str, float] = {
    "default": 1.0,
    "deg_15":  _TURN_15_RAD_S / _TURN_45_RAD_S,
    "deg_30":  _TURN_30_RAD_S / _TURN_45_RAD_S,
    "deg_45":  1.0,
    "deg_90":  _TURN_90_RAD_S / _TURN_45_RAD_S,
}

_ROTATIONAL_INTENTS: frozenset[str] = frozenset({"turn_left", "turn_right"})

# ``walk`` is a manager-side legacy where direction is in the magnitude
# (``forward`` / ``backward``) rather than the intent name. Resolve it
# explicitly so the rest of the dispatcher can assume direction-in-name
# semantics. Unknown ``walk`` magnitudes idle out (safe default for
# unrecognised direction).
_WALK_VELOCITY_BY_MAGNITUDE: dict[str, tuple[float, float, float, float]] = {
    # (yaw_rate, vel_x=lateral, vel_z=forward, hip_h).
    "forward":  (0.0, 0.0,  _WALK_SPEED_MPS,      _HIP_HEIGHT_M),
    "backward": (0.0, 0.0, -_BACK_SPEED_MPS,      _HIP_HEIGHT_M),
    "fast":     (0.0, 0.0,  _FAST_WALK_SPEED_MPS, _HIP_HEIGHT_M),
}

# Runtime tuning scalars applied on top of the resolved velocity. Defaults
# are 1.0 so the dispatcher behaves identically to the static table when
# no CLI overrides are passed. Mutated by ``run()`` when the user passes
# ``--turn-left-scale`` / ``--turn-right-scale`` / ``--walk-scale`` etc.
# to compensate for model-side L/R asymmetry or oversized magnitudes
# without rebuilding the static tables.
_RUNTIME_TURN_LEFT_SCALE: float = 1.0
_RUNTIME_TURN_RIGHT_SCALE: float = 1.0
_RUNTIME_FORWARD_SCALE: float = 1.0
_RUNTIME_BACKWARD_SCALE: float = 1.0
_RUNTIME_LATERAL_SCALE: float = 1.0


def _resolve_velocity(intent: str, magnitude: str) -> tuple[float, float, float, float]:
    """Pure ``(intent, magnitude)`` -> 4-D velocity resolver; idle on miss.

    Direction lives in the intent name for everything except ``walk``
    (legacy: ``walk/forward`` vs ``walk/backward``). Magnitude is a
    scalar multiplier (1.0 = ``default``) and unknown magnitudes fall
    back to 1.0 rather than idling so manager-side vocabulary
    additions don't silently freeze the planner.
    """
    if intent == "walk":
        return _WALK_VELOCITY_BY_MAGNITUDE.get(magnitude, _IDLE_INTENT)
    base = _BASE_VELOCITY.get(intent)
    if base is None:
        return _IDLE_INTENT
    yaw, vx, vy, hip_h = base
    if intent in _ROTATIONAL_INTENTS:
        scale = _TURN_SCALE.get(magnitude, 1.0)
        return (yaw * scale, vx, vy, hip_h)
    scale = _TRANSLATIONAL_SCALE.get(magnitude, 1.0)
    return (yaw, vx * scale, vy * scale, hip_h)


def _apply_runtime_scales(
    intent: str,
    velocity: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Apply runtime-tuning scalars to a resolved velocity.

    Per-direction scales let the operator compensate for model-side
    asymmetries (e.g. weaker left-turn coverage in the training data)
    or shave magnitudes globally without rebuilding the static tables.
    All scales default to 1.0 (no-op) so the existing unit-test
    invariants hold when nothing is overridden.
    """
    yaw, vx, vy, hip_h = velocity
    if intent == "turn_left":
        yaw *= _RUNTIME_TURN_LEFT_SCALE
    elif intent == "turn_right":
        yaw *= _RUNTIME_TURN_RIGHT_SCALE
    elif intent in ("fwd_step",) or (intent == "walk" and vx > 0):
        vx *= _RUNTIME_FORWARD_SCALE
    elif intent in ("back_step",) or (intent == "walk" and vx < 0):
        vx *= _RUNTIME_BACKWARD_SCALE
    elif intent in ("side_left", "side_right"):
        vy *= _RUNTIME_LATERAL_SCALE
    return (yaw, vx, vy, hip_h)


def intent_to_velocity(cmd: LocomotionCommand) -> tuple[float, float, float, float]:
    """Translate ``LocomotionCommand`` -> 4-D velocity ``(yaw_rate, vx, vy, hip_h)``.

    Returns ``_IDLE_INTENT`` for any intent the kplanner has no velocity
    meaning for (``hold_torso``, ``lean_*``, ``torso_*``, ``crouch``,
    unrecognised intents). Logs a single DEBUG line per miss so missing
    vocabulary additions are visible in the planner log.
    """
    result = _resolve_velocity(cmd.intent, cmd.magnitude)
    if result == _IDLE_INTENT and cmd.intent != "idle":
        log.debug("intent %s,%s has no velocity mapping; idling",
                  cmd.intent, cmd.magnitude)
        return result
    return _apply_runtime_scales(cmd.intent, result)


def _build_intent_velocity_map() -> dict[
    tuple[str, str], tuple[float, float, float, float]
]:
    """Precompute every (intent, magnitude) the dispatcher recognises.

    Used to populate ``INTENT_VELOCITY_MAP`` for docs / debug dumps.
    Unknown magnitudes still resolve correctly at runtime via
    ``intent_to_velocity`` (scale tables fall back to 1.0); this derived
    map is informative, not exhaustive.
    """
    out: dict[tuple[str, str], tuple[float, float, float, float]] = {
        ("idle", "default"): _resolve_velocity("idle", "default"),
        ("idle", "stand"):   _resolve_velocity("idle", "stand"),
    }
    for mag in _WALK_VELOCITY_BY_MAGNITUDE:
        out[("walk", mag)] = _resolve_velocity("walk", mag)
    for intent in _BASE_VELOCITY:
        if intent == "idle":
            continue
        scales = _TURN_SCALE if intent in _ROTATIONAL_INTENTS else _TRANSLATIONAL_SCALE
        for mag in scales:
            out[(intent, mag)] = _resolve_velocity(intent, mag)
    return out


INTENT_VELOCITY_MAP: dict[
    tuple[str, str], tuple[float, float, float, float]
] = _build_intent_velocity_map()


# ---------------------------------------------------------------------------
# Process hygiene (forked verbatim from x2_heuristic_planner)
# ---------------------------------------------------------------------------


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
    except OSError:
        return True
    finally:
        s.close()
    return False


class PidFile:
    """Context manager owning ``/tmp/...pid`` for the lifetime of the daemon."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> "PidFile":
        if self.path.exists():
            try:
                old = int(self.path.read_text().strip())
                if _pid_alive(old):
                    raise RuntimeError(
                        f"PID file {self.path} exists and PID {old} is alive — "
                        f"another planner is running. Run "
                        f"`kill {old}` or use the stack cleanup helper."
                    )
                log.warning(
                    "stale PID file %s for dead PID %d, cleaning up",
                    self.path, old,
                )
            except ValueError:
                log.warning("bad PID file %s, cleaning up", self.path)
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Pose publisher (same wire format as heuristic)
# ---------------------------------------------------------------------------


class PosePublisher:
    """Publishes pose payloads on a configurable ZMQ topic."""

    def __init__(
        self,
        host: str,
        port: int,
        topic: str = "body_pose",
        version: int = 4,
        hand_dof: int = 10,
        motion_token_dim: int = 64,
    ) -> None:
        import zmq
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._version = version
        self._hand_dof = hand_dof
        self._motion_token_dim = motion_token_dim
        time.sleep(0.1)

    def publish(
        self,
        frame: StreamFrame,
        future_frames: list[StreamFrame] | None = None,
        future_dt_s: float = 0.1,
    ) -> None:
        payload = build_pose_payload(
            frame,
            motion_token_dim=self._motion_token_dim,
            hand_dof=self._hand_dof,
            future_frames=future_frames,
            future_dt_s=future_dt_s,
        )
        msg = pack_pose_message(payload, topic=self._topic, version=self._version)
        self._sock.send(msg)

    def close(self) -> None:
        self._sock.close(linger=0)


# ---------------------------------------------------------------------------
# qpos[T, 38] -> StreamFrame helpers
# ---------------------------------------------------------------------------


def _qpos_to_stream_frame(
    qpos: np.ndarray,
    frame_index: int,
    bin_name: str,
    state: PlannerState = PlannerState.IDLE_LOOP,
) -> StreamFrame:
    """Convert a single 38-D qpos vector into a wire-compatible StreamFrame.

    qpos layout (MuJoCo wxyz convention, set by NeuralPlannerCore):
        [0:3]   root xyz (world frame)
        [3:7]   root quaternion wxyz
        [7:38]  31 joint positions in MuJoCo joint order

    StreamFrame.root_quat_xyzw is xyzw, so we permute wxyz -> xyzw here.
    """
    if qpos.shape[-1] != 38:
        raise ValueError(f"qpos must be 38-D, got {qpos.shape}")
    root_xyz = qpos[:3].astype(np.float32)
    root_wxyz = qpos[3:7].astype(np.float32)
    root_xyzw = np.array(
        [root_wxyz[1], root_wxyz[2], root_wxyz[3], root_wxyz[0]],
        dtype=np.float32,
    )
    # Use MuJoCo's xy world position for the diagnostic field (it's advisory;
    # the wire-critical fields are joint_pos_mj + root_quat_xyzw).
    return StreamFrame(
        joint_pos_mj=qpos[7:].astype(np.float32),
        root_quat_xyzw=root_xyzw,
        root_xy_world=root_xyz[:2].astype(np.float64),
        yaw_world_deg=0.0,  # advisory only; not used by deploy
        state=state,
        bin_name=bin_name,
        frame_index=frame_index,
        seam_blend=False,
    )


# ---------------------------------------------------------------------------
# Command sources (forked verbatim from x2_heuristic_planner with light edits)
# ---------------------------------------------------------------------------


def _scripted_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    yaml_path: Path,
    stop_event: threading.Event,
) -> None:
    cmds = commands_from_yaml(yaml_path)
    log.info("scripted source: queueing %d commands from %s", len(cmds), yaml_path)
    for cmd in cmds:
        if stop_event.is_set():
            return
        cmd_queue.put(cmd)
    log.info("scripted source: done")


def _zmq_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
) -> None:
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    log.info("zmq command source: SUB %s on tcp://%s:%d", topic, host, port)
    try:
        while not stop_event.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
                intent = str(payload["intent"])
                if intent == "shutdown":
                    log.info("zmq command source: shutdown received")
                    stop_event.set()
                    continue
                magnitude = str(payload.get("magnitude", "default"))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("zmq command source: bad payload %r: %s", parts[1], exc)
                continue
            cmd_queue.put(
                LocomotionCommand(
                    intent=intent,
                    magnitude=magnitude,
                    source="zmq",
                )
            )
    finally:
        sock.close(linger=0)


_KPLANNER_KEYBOARD_HELP = """
X2 kplanner — keyboard commands (mirrors heuristic):
   w        walk forward
   W        walk fast
   b        back step
   a / d    side left / right
   q / e    turn left / right 45 deg
   Q / E    turn left / right 90 deg
   1 / 3    turn left / right 15 deg
   2 / 4    turn left / right 30 deg
   space    idle
   x        quit
"""

_KPLANNER_KEYBOARD_MAP: dict[str, tuple[str, str]] = {
    "w": ("walk", "forward"),
    "W": ("walk", "fast"),
    "b": ("back_step", "half_ft"),
    "a": ("side_left", "default"),
    "d": ("side_right", "default"),
    "1": ("turn_left", "deg_15"),
    "2": ("turn_left", "deg_30"),
    "q": ("turn_left", "deg_45"),
    "Q": ("turn_left", "deg_90"),
    "3": ("turn_right", "deg_15"),
    "4": ("turn_right", "deg_30"),
    "e": ("turn_right", "deg_45"),
    "E": ("turn_right", "deg_90"),
    " ": ("idle", "default"),
}


def _keyboard_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    stop_event: threading.Event,
) -> None:
    if not sys.stdin.isatty():
        log.error("keyboard source: stdin is not a TTY; refusing to start")
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print(_KPLANNER_KEYBOARD_HELP, flush=True)
        while not stop_event.is_set():
            ch = sys.stdin.read(1)
            if ch in ("x", "\x03"):
                stop_event.set()
                return
            if ch in _KPLANNER_KEYBOARD_MAP:
                intent, mag = _KPLANNER_KEYBOARD_MAP[ch]
                cmd_queue.put(
                    LocomotionCommand(intent=intent, magnitude=mag, source="kbd")
                )
                print(f"  -> {intent} {mag}", flush=True)
            else:
                print(_KPLANNER_KEYBOARD_HELP, flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# T-pose / warmup anchor helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Training-default stand pose. Source of truth for these numbers is
# ``policy_parameters.hpp::default_angles`` in the C++ deploy
# (mirrored in x2_capture_pose.py / x2_pelvis_z_from_capture.py /
# x2_action_monitor.py / x2_scan_mc_motors.py). When the deploy's
# pose-ref starvation watchdog trips it PD-controls toward this exact
# 31-DOF stand pose; using it as the kplanner's warmup qpos means
# SAFE_IDLE produces ZERO joint-target delta from the RSI'd spawn
# pose -- the robot sits perfectly still through the multi-second
# cold-start window where the planner is loading models and the
# recorder isn't yet merging pose:5556 frames.
#
# Floor-contact hip_z for this pose is 0.636 m (X2 with knees bent
# 0.669 rad / 38 deg, lowest foot geom just touches z=0). Computed via
# mujoco FK on the canonical sphere-feet x2_ultra MJCF; matches the
# Z that ``x2_pelvis_z_from_capture.py`` reports for DEFAULT_ANGLES.
# ---------------------------------------------------------------------------
_TRAINING_DEFAULT_ANGLES: np.ndarray = np.array((
    -0.312, 0.0,  0.0,  0.669, -0.363, 0.0,    # left  leg (6)
    -0.312, 0.0,  0.0,  0.669, -0.363, 0.0,    # right leg (6)
     0.0,  0.0,  0.0,                          # waist     (3)
     0.2,  0.2,  0.0, -0.6,  0.0,  0.0,  0.0,  # left  arm (7)
     0.2, -0.2,  0.0, -0.6,  0.0,  0.0,  0.0,  # right arm (7)
     0.0,  0.0,                                # head      (2)
), dtype=np.float32)
_TRAINING_DEFAULT_HIP_Z: float = 0.636


# Canonical recorded stand pose (from real-robot IMU + state telemetry).
# Frame 0 is a stable bent-knee stand at hip_z=0.655m with the leg / arm
# joint angles a real X2 settles into. Used as the SECONDARY default
# (after training_default_angles) when the operator explicitly wants a
# real-recorded stand pose and is OK with SAFE_IDLE pulling the joints
# the ~0.15--0.57 rad toward training_default_angles during cold-start.
_BALANCED_STAND_PKL: Path = (
    Path(__file__).resolve().parent.parent
    / "data" / "motions" / "x2_recorded" / "balanced_stand.pkl"
)


def _build_training_default_qpos() -> np.ndarray:
    """qpos[38] (wxyz) at the deploy's training-default stand pose.

    This is the deploy's SAFE_IDLE PD target so RSI'ing the bridge to
    this pose means the watchdog watchdog-trip during cold-start
    produces ZERO joint-target delta -- the robot doesn't move even
    though no upstream pose stream has arrived yet.
    """
    qpos = np.zeros(38, dtype=np.float32)
    qpos[2] = _TRAINING_DEFAULT_HIP_Z
    qpos[3] = 1.0  # quaternion w = 1 (identity rotation, wxyz)
    qpos[7:38] = _TRAINING_DEFAULT_ANGLES
    return qpos


def _qpos_from_deploy_pkl_frame(obj_inner: dict) -> Optional[np.ndarray]:
    """Convert one frame of a deploy-PKL schema dict to qpos[38] (wxyz).

    The deploy-PKL schema (used by bake_planner_rsi_anchor.py,
    bake_kplanner_rsi_anchor.py, and the on-disk balanced_stand.pkl)
    stores the root rotation in **xyzw** convention; MuJoCo qpos uses
    **wxyz**. This helper reorders rather than re-normalizes so a
    "stationary" PKL stays stationary.
    """
    needed = {"root_trans_offset", "root_rot", "dof"}
    if not needed.issubset(obj_inner):
        return None
    root_trans = np.asarray(obj_inner["root_trans_offset"])
    root_rot = np.asarray(obj_inner["root_rot"])  # xyzw
    dof = np.asarray(obj_inner["dof"])
    f0_trans = root_trans[0] if root_trans.ndim == 2 else root_trans
    f0_rot_xyzw = root_rot[0] if root_rot.ndim == 2 else root_rot
    f0_dof = dof[0] if dof.ndim == 2 else dof
    if f0_trans.shape != (3,) or f0_rot_xyzw.shape != (4,) or f0_dof.shape != (31,):
        return None
    f0_rot_wxyz = np.array(
        [f0_rot_xyzw[3], f0_rot_xyzw[0], f0_rot_xyzw[1], f0_rot_xyzw[2]],
        dtype=np.float32,
    )
    qpos = np.zeros(38, dtype=np.float32)
    qpos[0:3] = f0_trans
    qpos[3:7] = f0_rot_wxyz
    qpos[7:38] = f0_dof
    return qpos


def _build_hardcoded_warmup_qpos() -> np.ndarray:
    """Hand-crafted T-pose-ish stand qpos[38] in MuJoCo wxyz convention.

    Fallback for when ``balanced_stand.pkl`` isn't on disk. The joint
    angles are ALL ZERO and the hip is at 0.95 m -- the feet are
    therefore NOT on the floor and the deploy's tracking policy will
    fight gravity from tick 0. Only use this if balanced_stand.pkl is
    absent; the wrapper bake step will warn loudly when this fires.

    qpos layout:
      - [0:3]   root xyz (origin + hip height)
      - [3:7]   wxyz quaternion (identity)
      - [7:38]  31 joint angles (rad), all zero
    """
    qpos = np.zeros(38, dtype=np.float32)
    qpos[2] = _HIP_HEIGHT_M  # hip height above floor (does NOT settle)
    qpos[3] = 1.0            # quaternion w = 1 (identity rotation)
    return qpos


def _build_default_warmup_qpos() -> np.ndarray:
    """Resolved-default warmup qpos[38] in MuJoCo wxyz convention.

    Defaults to the deploy's **training_default_angles** stand pose.
    This is the single safest choice for the parity RSI anchor because
    it's bit-identical to the pose the deploy's pose-ref starvation
    watchdog falls back to (SAFE_IDLE PD target). When the wrapper
    spawns the deploy ~5 s before the planner is ready and ~15 s
    before the recorder is merging pose:5556 frames, the watchdog
    will trip and SAFE_IDLE will hold the robot at training_default_
    angles. RSI'ing the bridge to the same pose means SAFE_IDLE
    produces ZERO joint-target delta and the robot doesn't drift
    through the entire cold-start window.

    Both the daemon's first publish tick AND the parity RSI anchor PKL
    (baked via ``bake_kplanner_rsi_anchor.py``) derive from this, so
    spawn-pose / wire-pose / SAFE_IDLE-target stay bit-identical.

    Operators who want a recorded real-robot stand instead can pass
    ``--warmup-qpos-path .../balanced_stand.pkl``; the bake step
    forwards it.
    """
    qpos = _build_training_default_qpos()
    log.info(
        "warmup anchor: using training_default_angles stand pose "
        "(hip_z=%.3fm) -- matches deploy SAFE_IDLE PD target",
        float(qpos[2]),
    )
    return qpos


def _load_balanced_stand_default() -> Optional[np.ndarray]:
    """Best-effort load of frame 0 from balanced_stand.pkl. Returns None
    if the schema doesn't match (caller falls back)."""
    try:
        import joblib  # local import: only needed for the recorded PKL
    except ImportError:  # pragma: no cover -- joblib ships with sklearn
        log.warning("joblib missing; cannot load balanced_stand.pkl")
        return None
    try:
        obj = joblib.load(_BALANCED_STAND_PKL)
    except Exception as exc:  # noqa: BLE001
        log.warning("balanced_stand.pkl unreadable: %r", exc)
        return None
    # Schema: {'balanced_stand': {'root_trans_offset', 'root_rot', 'dof', ...}}
    if isinstance(obj, dict) and len(obj) >= 1:
        inner = next(iter(obj.values()))
        if isinstance(inner, dict):
            return _qpos_from_deploy_pkl_frame(inner)
    return None


def _load_warmup_qpos(path: Optional[Path]) -> np.ndarray:
    """Load a [38]-D qpos vector from a pickled stand-pose, or fall back to
    the resolved default (balanced_stand.pkl -> hardcoded T-pose)."""
    if path is None:
        return _build_default_warmup_qpos()
    if not path.is_file():
        log.warning(
            "warmup-qpos-path %s not found; falling back to resolved default",
            path,
        )
        return _build_default_warmup_qpos()
    try:
        import joblib  # try joblib first since recorded PKLs are zlib-compressed
        obj = joblib.load(path)
    except Exception:  # noqa: BLE001 -- fall through to raw pickle
        with path.open("rb") as f:
            obj = pickle.load(f)
    # Accept several schemas:
    #   1. Raw [38] array or [T, 38] (take frame 0)
    #   2. {'mujoco_qpos': ...} or {'qpos': ...}
    #   3. Deploy-PKL nested: {'<name>': {'root_trans_offset', 'root_rot', 'dof', ...}}
    #      (this matches both x2_recorded/*.pkl and the anchor PKL schemas)
    if isinstance(obj, dict):
        # Case 2: explicit qpos field
        arr = obj.get("mujoco_qpos", obj.get("qpos", None))
        if arr is None:
            # Case 3: deploy-PKL nested
            if len(obj) >= 1:
                inner = next(iter(obj.values()))
                if isinstance(inner, dict):
                    qpos = _qpos_from_deploy_pkl_frame(inner)
                    if qpos is not None:
                        log.info(
                            "warmup anchor: loaded from %s "
                            "(deploy-PKL schema, frame 0, hip_z=%.3fm)",
                            path, float(qpos[2]),
                        )
                        return qpos
            raise ValueError(
                f"warmup PKL {path} has no recognisable qpos / "
                f"deploy-PKL schema (keys={list(obj.keys())})"
            )
    else:
        arr = obj
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[-1] != 38:
        raise ValueError(
            f"warmup PKL {path} qpos shape {arr.shape} != [38]"
        )
    log.info("warmup anchor: loaded from %s (shape=[38])", path)
    return arr


# ---------------------------------------------------------------------------
# Worker thread: replan when ring buffer dips below threshold
# ---------------------------------------------------------------------------


class IntentState:
    """Thread-safe holder for the current velocity-intent target."""

    def __init__(self, initial: tuple[float, float, float, float]) -> None:
        self._target = initial
        self._lock = threading.Lock()
        self._version = 0  # bumped on every update; worker reads for logging

    def set(self, target: tuple[float, float, float, float]) -> None:
        with self._lock:
            self._target = tuple(target)
            self._version += 1

    def get(self) -> tuple[tuple[float, float, float, float], int]:
        with self._lock:
            return self._target, self._version


def _planner_worker(
    planner_core,  # NeuralPlannerCore; not type-hinted to avoid eager import
    intent: IntentState,
    replan_lock: threading.Lock,
    stop_event: threading.Event,
    replan_event: threading.Event,
) -> None:
    """Replan refill loop. Runs predict() each time the buffer drops below
    threshold, holding ``replan_lock`` only for the cursor swap (the predict
    call itself is unlocked so the publish thread can still pop frames).

    Idle gate: when the intent matches ``_IDLE_INTENT`` the publisher is
    holding the frozen default_angles anchor and never reads from the ring
    buffer, so running predict() here would just burn GPU cycles on
    frames nobody consumes -- and would also drift the buffer further
    from default_angles every call. Skip the replan while idle so the
    buffer's last-good state stays fresh for the next non-idle command.
    """
    log.info("planner worker thread started")
    while not stop_event.is_set():
        # Wait until the publisher signals it's draining the buffer, OR a
        # 50 ms timeout so we still check on stale buffers periodically.
        if not replan_event.wait(timeout=0.05):
            with replan_lock:
                needs_replan = planner_core.should_replan()
        else:
            replan_event.clear()
            needs_replan = True
        if not needs_replan or stop_event.is_set():
            continue
        target, ver = intent.get()
        if tuple(target) == _IDLE_INTENT:
            # Idle -- publisher holds the static anchor; no neural frames
            # are being consumed. Don't replan.
            continue
        log.debug("worker: replan intent v=%d target=%s", ver, target)
        t0 = time.monotonic()
        try:
            with replan_lock:
                planner_core.replan_with_velocity(list(target))
        except Exception:
            log.exception("worker: replan failed; will retry next cycle")
            time.sleep(0.05)
            continue
        dt_ms = (time.monotonic() - t0) * 1000.0
        log.debug("worker: replan done in %.1fms (frames_remaining=%d)",
                  dt_ms, planner_core.frames_remaining)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def run(
    vqvae_ckpt: Path,
    pose_ckpt: Path,
    root_ckpt: Path,
    warmup_qpos_path: Optional[Path],
    pub_host: str,
    pub_port: int,
    pid_file: Path,
    demo_yaml: Optional[Path],
    enable_keyboard: bool,
    zmq_cmd_host: Optional[str],
    zmq_cmd_port: Optional[int],
    zmq_cmd_topic: str,
    duration_s: float,
    hand_dof: int,
    verbose: bool,
    warmup_quiet_stand_s: float,
    body_pose_port: Optional[int],
    device: str,
    replan_threshold_frames: int,
    yaw_lock_epsilon_rad_s: float = 0.0,
    turn_left_scale: float = 1.0,
    turn_right_scale: float = 1.0,
    forward_scale: float = 1.0,
    backward_scale: float = 1.0,
    lateral_scale: float = 1.0,
) -> int:
    _setup_logging(verbose)

    # Stash runtime tuning scales on the module-level mutable singletons
    # used by ``intent_to_velocity``. Defaults are 1.0 so the static
    # unit-test invariants still hold when the user passes no override.
    global _RUNTIME_TURN_LEFT_SCALE, _RUNTIME_TURN_RIGHT_SCALE
    global _RUNTIME_FORWARD_SCALE, _RUNTIME_BACKWARD_SCALE, _RUNTIME_LATERAL_SCALE
    _RUNTIME_TURN_LEFT_SCALE = float(turn_left_scale)
    _RUNTIME_TURN_RIGHT_SCALE = float(turn_right_scale)
    _RUNTIME_FORWARD_SCALE = float(forward_scale)
    _RUNTIME_BACKWARD_SCALE = float(backward_scale)
    _RUNTIME_LATERAL_SCALE = float(lateral_scale)
    if any(s != 1.0 for s in (
        turn_left_scale, turn_right_scale,
        forward_scale, backward_scale, lateral_scale,
    )):
        log.info(
            "velocity scales: turn_left=%.2f turn_right=%.2f "
            "forward=%.2f backward=%.2f lateral=%.2f",
            turn_left_scale, turn_right_scale,
            forward_scale, backward_scale, lateral_scale,
        )
    log.info("yaw-lock epsilon: %.3f rad/s (0=disabled)",
             yaw_lock_epsilon_rad_s)

    # ---- Topic selection (mirrors heuristic daemon's two modes)
    if body_pose_port is not None:
        effective_port = body_pose_port
        effective_topic = "body_pose"
        log.info(
            "kplanner publish mode: 'body_pose' on tcp://%s:%d (Phase 0 recorder merge mode)",
            pub_host, effective_port,
        )
    else:
        effective_port = pub_port
        effective_topic = "pose"
        log.info(
            "kplanner publish mode: 'pose' on tcp://%s:%d (direct-to-deploy fallback)",
            pub_host, effective_port,
        )

    # ---- Pre-flight checks
    if _port_in_use(effective_port, "127.0.0.1") or _port_in_use(effective_port, "0.0.0.0"):
        log.error(
            "publish port %d already in use. Run the stack cleanup helper first.",
            effective_port,
        )
        return 1
    for label, p in (
        ("vqvae-ckpt", vqvae_ckpt),
        ("pose-ckpt", pose_ckpt),
        ("root-ckpt", root_ckpt),
    ):
        if not Path(p).is_file():
            log.error("%s not found: %s", label, p)
            return 1

    # ---- Load model + planner core (this is the slow part: ~5--10s on cold start)
    log.info("loading X2 kplanner stack on device=%s ...", device)
    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths,
        load_x2_planner,
    )

    default_paths = X2PlannerPaths.default()
    paths = X2PlannerPaths(
        vqvae_ckpt=vqvae_ckpt,
        pose_ckpt=pose_ckpt,
        root_ckpt=root_ckpt,
        vqvae_version_dir=default_paths.vqvae_version_dir,
        pose_version_dir=default_paths.pose_version_dir,
        root_version_dir=default_paths.root_version_dir,
    )
    planner_core = load_x2_planner(
        paths,
        device=device,
        replan_threshold_frames=replan_threshold_frames,
    )
    log.info("kplanner stack loaded.")

    # ---- Warmup anchor + first replan
    import torch
    warmup_qpos = _load_warmup_qpos(warmup_qpos_path)
    planner_core.reset(torch.from_numpy(warmup_qpos))
    intent_state = IntentState(_IDLE_INTENT)
    # First replan synchronously so the publish thread has frames immediately.
    planner_core.replan_with_velocity(list(_IDLE_INTENT))
    log.info(
        "first replan complete; ring buffer has %d frames",
        planner_core.frames_remaining,
    )

    # ---- Publisher
    publisher = PosePublisher(
        host=pub_host,
        port=effective_port,
        topic=effective_topic,
        hand_dof=hand_dof,
    )
    log.info(
        "publishing %r on tcp://%s:%d at %.1f Hz",
        effective_topic, pub_host, effective_port, OUTPUT_FPS,
    )

    # ---- Command sources
    cmd_queue: "queue.Queue[LocomotionCommand]" = queue.Queue()
    stop_event = threading.Event()
    replan_event = threading.Event()
    replan_lock = threading.Lock()
    threads: list[threading.Thread] = []

    if demo_yaml is not None:
        thr = threading.Thread(
            target=_scripted_command_thread,
            args=(cmd_queue, demo_yaml, stop_event),
            name="cmd-scripted",
            daemon=True,
        )
        thr.start()
        threads.append(thr)

    if zmq_cmd_host is not None and zmq_cmd_port is not None:
        thr = threading.Thread(
            target=_zmq_command_thread,
            args=(cmd_queue, zmq_cmd_host, zmq_cmd_port, zmq_cmd_topic, stop_event),
            name="cmd-zmq",
            daemon=True,
        )
        thr.start()
        threads.append(thr)

    if enable_keyboard:
        thr = threading.Thread(
            target=_keyboard_command_thread,
            args=(cmd_queue, stop_event),
            name="cmd-kbd",
            daemon=True,
        )
        thr.start()
        threads.append(thr)

    worker_thread = threading.Thread(
        target=_planner_worker,
        args=(planner_core, intent_state, replan_lock, stop_event, replan_event),
        name="kplanner-worker",
        daemon=True,
    )
    worker_thread.start()
    threads.append(worker_thread)

    # ---- Signal handlers
    def _on_signal(signum: int, _frame: object) -> None:
        log.info("signal %d -> draining and shutting down", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    period_s = 1.0 / OUTPUT_FPS
    next_tick = time.monotonic()
    end_at = (
        time.monotonic() + duration_s if duration_s > 0 else float("inf")
    )

    # The future window emitted alongside each frame (mirrors heuristic
    # daemon's tokenizer-obs contract). With OUTPUT_FPS=50 and
    # step_ticks=5, future[k] is at +(k+1)*0.1s from the current frame.
    num_future = 9
    step_ticks = 5

    global_tick = 0
    last_intent_log: tuple[float, float, float, float] = _IDLE_INTENT

    # Always-available frozen-anchor StreamFrame, shared by the warmup
    # ticks below AND the IDLE_LOOP branch of the main publish loop.
    # Built once from warmup_qpos so all three paths emit a bit-identical
    # pose (== deploy default_angles when the operator hasn't overridden
    # --kplanner-warmup-qpos). Hoisted out of the warmup-only branch so
    # the IDLE_LOOP gate works even when warmup_quiet_stand_s == 0
    # (e.g. tests that skip the warmup).
    anchor_frame = _qpos_to_stream_frame(
        warmup_qpos,
        frame_index=0,
        bin_name="kplanner_warmup_anchor",
        state=PlannerState.IDLE_LOOP,
    )

    # Integrated world pose persisted across IDLE_LOOP <-> PLAYING
    # transitions so the kplanner reference doesn't snap the robot back
    # to origin+identity quat every time the operator releases the stick.
    #
    # Without this, each new IDLE_LOOP entry freezes on the anchor's
    # root_quat = identity (= facing world +X), and the SONIC tracking
    # policy actively yaws the robot back to face forward after any
    # turn -- visible as "robot keeps coming back to the same direction".
    # Similarly each IDLE_LOOP -> PLAYING transition would reset
    # planner_core to the anchor (origin + identity), wiping the prior
    # translation / yaw so velocity intents start fresh from world (0,0,0)
    # -- visible as "robot stays fixed in place when commanded forward"
    # because the reference position keeps snapping back to origin.
    #
    # ``current_root_xy`` is in MuJoCo qpos units (m), ``current_root_wxyz``
    # in wxyz convention to match _BASE_VELOCITY's qpos layout. Joint
    # angles in IDLE_LOOP always snap to default_angles regardless;
    # only the root frame is preserved.
    current_root_xy = warmup_qpos[:2].astype(np.float32).copy()
    current_root_wxyz = warmup_qpos[3:7].astype(np.float32).copy()
    current_root_z = float(warmup_qpos[2])

    def _idle_root_xyzw() -> np.ndarray:
        """Current wxyz -> xyzw for the wire."""
        return np.array(
            [current_root_wxyz[1], current_root_wxyz[2],
             current_root_wxyz[3], current_root_wxyz[0]],
            dtype=np.float32,
        )

    def _build_warm_qpos() -> np.ndarray:
        """Construct a 38-D qpos at the current root frame with
        default_angles joints. Used to seed planner_core on
        IDLE_LOOP -> PLAYING so the neural buffer starts from the
        robot's current world pose rather than from origin.
        """
        warm = np.empty_like(warmup_qpos)
        warm[0] = current_root_xy[0]
        warm[1] = current_root_xy[1]
        warm[2] = current_root_z
        warm[3:7] = current_root_wxyz
        warm[7:] = warmup_qpos[7:]
        return warm

    try:
        with PidFile(pid_file):
            # ---- Optional quiet-stand warmup (publishes the frozen anchor qpos)
            warmup_n = (
                int(round(max(0.0, warmup_quiet_stand_s) * OUTPUT_FPS))
                if warmup_quiet_stand_s > 0 else 0
            )
            if warmup_n > 0:
                log.info(
                    "quiet-stand warmup: %d ticks (%.2fs) of frozen anchor qpos "
                    "(joint_pos=%s..., hip_h=%.2fm)",
                    warmup_n, warmup_quiet_stand_s,
                    warmup_qpos[7:10].tolist(), warmup_qpos[2],
                )
                for warm_idx in range(warmup_n):
                    if stop_event.is_set() or time.monotonic() >= end_at:
                        break
                    anchor_frame_idx = StreamFrame(
                        joint_pos_mj=anchor_frame.joint_pos_mj,
                        root_quat_xyzw=_idle_root_xyzw(),
                        root_xy_world=current_root_xy.astype(np.float64).copy(),
                        yaw_world_deg=anchor_frame.yaw_world_deg,
                        state=anchor_frame.state,
                        bin_name=anchor_frame.bin_name,
                        frame_index=warm_idx,
                        seam_blend=False,
                    )
                    publisher.publish(anchor_frame_idx)
                    next_tick += period_s
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        next_tick = time.monotonic()
                global_tick = warmup_n
                log.info("quiet-stand warmup done; switching to neural planner.")

            # Track current high-level state for the idle-gate. When the
            # operator isn't pushing the stick (intent == _IDLE_INTENT) we
            # freeze on the warmup anchor (== deploy default_angles) so
            # the wire carries a perfectly stationary reference; the
            # neural planner only runs while a real locomotion command
            # is active. Without this gate the model freely predicts
            # walking-like motion in idle (visible in viewer as wide
            # arms / striding legs / slow yaw drift) since the planner
            # has no notion of "no motion intended" -- its training
            # distribution doesn't include a static-stand label.
            current_planner_state = PlannerState.IDLE_LOOP

            def _build_idle_future(start_idx: int) -> list[StreamFrame]:
                """num_future copies of the anchor pose at the CURRENT
                integrated world root, monotonically indexed. Mirrors
                the wire shape PLAYING emits so the C++ tokenizer's
                10x68 obs gather doesn't need a special idle-mode code
                path. Reading current_root_* fresh each call so a
                PLAYING -> IDLE_LOOP transition picks up the last
                published root pose for every future slot."""
                xyzw = _idle_root_xyzw()
                xy = current_root_xy.astype(np.float64).copy()
                return [
                    StreamFrame(
                        joint_pos_mj=anchor_frame.joint_pos_mj,
                        root_quat_xyzw=xyzw,
                        root_xy_world=xy.copy(),
                        yaw_world_deg=anchor_frame.yaw_world_deg,
                        state=PlannerState.IDLE_LOOP,
                        bin_name="kplanner_idle_future",
                        frame_index=start_idx + step_ticks * (k + 1),
                        seam_blend=False,
                    )
                    for k in range(num_future)
                ]

            while not stop_event.is_set() and time.monotonic() < end_at:
                # ---- Drain command queue and apply the latest intent.
                latest_cmd: Optional[LocomotionCommand] = None
                while True:
                    try:
                        cmd = cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    latest_cmd = cmd
                if latest_cmd is not None:
                    target = intent_to_velocity(latest_cmd)
                    intent_state.set(target)
                    if target != last_intent_log:
                        log.info(
                            "intent applied (%s, %s, %s) -> target=%s",
                            latest_cmd.intent, latest_cmd.magnitude, latest_cmd.source,
                            target,
                        )
                        last_intent_log = target

                # ---- Resolve high-level state from current intent.
                # IDLE_LOOP <-> PLAYING transitions are edge-triggered
                # so we can log them and seed the neural buffer cleanly
                # on every entry to PLAYING.
                current_target, _ = intent_state.get()
                is_idle = tuple(current_target) == _IDLE_INTENT
                desired_state = (
                    PlannerState.IDLE_LOOP if is_idle else PlannerState.PLAYING
                )
                if desired_state != current_planner_state:
                    if desired_state == PlannerState.PLAYING:
                        # Seed the ring buffer with default_angles at
                        # the ROBOT'S CURRENT integrated root frame
                        # (not at world origin / identity quat). This
                        # makes the neural planner's canonicalize step
                        # treat the operator's velocity intent as
                        # "starting from where the robot currently is",
                        # so a fwd_step after a 90-degree turn walks in
                        # the new heading direction, not in world +X.
                        # Without this seed the planner snapped the
                        # reference back to (0,0,0) + identity every
                        # PLAYING entry, fighting the robot's physical
                        # drift.
                        warm = _build_warm_qpos()
                        with replan_lock:
                            planner_core.reset(torch.from_numpy(warm))
                        replan_event.set()
                        log.info(
                            "state: IDLE_LOOP -> PLAYING (intent=%s); "
                            "buffer seeded at root_xy=%s yaw_wxyz=%s, replan queued",
                            current_target,
                            warm[:2].tolist(),
                            warm[3:7].tolist(),
                        )
                    else:
                        log.info(
                            "state: PLAYING -> IDLE_LOOP (intent back to idle); "
                            "freezing at root_xy=%s yaw_wxyz=%s",
                            current_root_xy.tolist(),
                            current_root_wxyz.tolist(),
                        )
                    current_planner_state = desired_state

                # ---- Build the published frame.
                if current_planner_state == PlannerState.IDLE_LOOP:
                    # Frozen-anchor branch: emit default_angles joints
                    # at the LAST integrated world root pose. Joint
                    # angles snap to anchor (default_angles); root XY
                    # and yaw carry over from the prior PLAYING session
                    # so a release-after-turn doesn't trigger a SONIC
                    # yaw-correction back toward identity.
                    cur_frame = StreamFrame(
                        joint_pos_mj=anchor_frame.joint_pos_mj,
                        root_quat_xyzw=_idle_root_xyzw(),
                        root_xy_world=current_root_xy.astype(np.float64).copy(),
                        yaw_world_deg=anchor_frame.yaw_world_deg,
                        state=PlannerState.IDLE_LOOP,
                        bin_name="kplanner_idle",
                        frame_index=global_tick,
                        seam_blend=False,
                    )
                    future_frames = _build_idle_future(global_tick)
                else:
                    # PLAYING branch: sample from the neural ring buffer
                    # exactly as before, with the lookahead window the
                    # tokenizer expects.
                    with replan_lock:
                        qpos_tensor = planner_core.get_next_frame()
                        if planner_core.should_replan():
                            replan_event.set()
                        future_qposes: list[np.ndarray] = []
                        for k in range(num_future):
                            peek_idx = planner_core.current_frame_idx + step_ticks * (k + 1) - 1
                            buf = planner_core.frames["mujoco_qpos"]
                            peek_idx_clamped = max(0, min(peek_idx, buf.shape[1] - 1))
                            future_qposes.append(
                                buf[0, peek_idx_clamped].detach().cpu().numpy()
                            )
                    qpos_np = qpos_tensor.detach().cpu().numpy()

                    # Yaw-lock mitigation. When the operator's commanded
                    # yaw_rate is below threshold (pure forward / back /
                    # sidestep / idle), the model still predicts small
                    # per-frame yaw deltas that compound across replans
                    # into a visible spin. The deploy's SONIC policy
                    # tracks the published root_quat, so this drift
                    # makes the robot rotate instead of translate.
                    # Override the published yaw with the persisted
                    # wxyz when commanded yaw is near zero -- joint
                    # angles still produce stepping, only the root
                    # orientation reference is pinned. ``--yaw-lock-
                    # epsilon`` raises the threshold (0 disables, large
                    # values lock more aggressively).
                    yaw_locked = (
                        yaw_lock_epsilon_rad_s > 0.0
                        and abs(float(current_target[0])) < yaw_lock_epsilon_rad_s
                    )
                    if yaw_locked:
                        qpos_np[3:7] = current_root_wxyz.astype(qpos_np.dtype)
                        for fq in future_qposes:
                            fq[3:7] = current_root_wxyz.astype(fq.dtype)

                    # Persist the world-frame root for the next
                    # IDLE_LOOP freeze / PLAYING seed.
                    current_root_xy = qpos_np[:2].astype(np.float32).copy()
                    current_root_z = float(qpos_np[2])
                    if not yaw_locked:
                        # NeuralPlannerCore.replan_with_velocity packs the
                        # qpos's quat as wxyz in slots [3:7]; mirror that
                        # ordering when persisting (the xyzw permutation
                        # only happens in _qpos_to_stream_frame at publish).
                        # When yaw is locked we already overrode the
                        # published wxyz with the persisted value, so
                        # there's nothing to update.
                        current_root_wxyz = qpos_np[3:7].astype(np.float32).copy()
                    cur_frame = _qpos_to_stream_frame(
                        qpos_np,
                        frame_index=global_tick,
                        bin_name="kplanner",
                        state=PlannerState.PLAYING,
                    )
                    future_frames = [
                        _qpos_to_stream_frame(
                            future_qposes[k],
                            frame_index=global_tick + step_ticks * (k + 1),
                            bin_name="kplanner_future",
                            state=PlannerState.PLAYING,
                        )
                        for k in range(num_future)
                    ]
                publisher.publish(cur_frame, future_frames=future_frames, future_dt_s=0.1)
                global_tick += 1

                next_tick += period_s
                slack = next_tick - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    if -slack > 5 * period_s:
                        log.warning(
                            "loop fell behind by %.0fms; resyncing", -slack * 1000
                        )
                        next_tick = time.monotonic()
            log.info("main loop exited at tick %d", global_tick)
    finally:
        stop_event.set()
        replan_event.set()
        publisher.close()
        for thr in threads:
            thr.join(timeout=2.0)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="x2_kplanner",
        description=(
            "Neural kinematic locomotion planner for AgiBot X2 — streams 31-DOF "
            "body refs + root quat over the 'body_pose' (or 'pose') ZMQ topic "
            "at 50 Hz, conditioned on a Quest3 / scripted velocity intent."
        ),
    )
    # Pinned step checkpoints (not last.ckpt) so a fresh training run
    # doesn't silently re-point inference at an unverified checkpoint.
    # Override via the CLI when promoting a newer checkpoint.
    p.add_argument(
        "--vqvae-ckpt",
        type=Path,
        default=(
            _REPO_ROOT
            / "motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0200000.ckpt"
        ),
    )
    p.add_argument(
        "--pose-ckpt",
        type=Path,
        default=(
            _REPO_ROOT
            / "motionbricks/out/motionbricks_pose_x2_v2/version_1/checkpoints/model-step=0250000.ckpt"
        ),
    )
    p.add_argument(
        "--root-ckpt",
        type=Path,
        default=(
            _REPO_ROOT
            / "motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0235000.ckpt"
        ),
    )
    p.add_argument(
        "--warmup-qpos-path",
        type=Path,
        default=None,
        help=(
            "Optional PKL with a stand qpos[38] (or [T, 38] -> frame 0). "
            "Used to seed the ring buffer and as the warmup-quiet-stand "
            "anchor. Falls back to a hand-crafted zero-joints stand with "
            f"hip_h={_HIP_HEIGHT_M:.2f}m + identity quat."
        ),
    )
    p.add_argument(
        "--pub-host", default="127.0.0.1",
        help="ZMQ publish bind host (default: 127.0.0.1).",
    )
    p.add_argument(
        "--pub-port", type=int, default=5556,
        help="ZMQ publish port (default: 5556).",
    )
    p.add_argument(
        "--body-pose-port", type=int, default=None,
        help=(
            "Publish under topic 'body_pose' on this port (Phase 0 recorder "
            "merge mode). Mutually exclusive with --pub-port's direct-to-deploy "
            "mode; when set, overrides --pub-port."
        ),
    )
    p.add_argument(
        "--pid-file",
        type=Path,
        default=Path("/tmp/x2_kplanner.pid"),
    )
    p.add_argument("--demo", type=Path, help="Scripted demo YAML.")
    p.add_argument("--keyboard", action="store_true",
                   help="Enable interactive keyboard input.")
    p.add_argument(
        "--zmq-cmd-host", default=None,
        help="If set, also subscribe for high-level commands on tcp://host:port",
    )
    p.add_argument("--zmq-cmd-port", type=int, default=None)
    p.add_argument(
        "--zmq-cmd-topic", default="planner_cmd",
        help="Topic prefix expected on the command SUB socket.",
    )
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--hand-dof", type=int, default=10)
    p.add_argument(
        "--warmup-quiet-stand-s", type=float, default=0.5,
        help=(
            "Publish the warmup-qpos anchor frame for this many seconds at "
            "startup before the neural planner takes over."
        ),
    )
    p.add_argument(
        "--device", default="cuda",
        help="torch device (cuda / cpu). Falls back to cpu if cuda unusable.",
    )
    p.add_argument(
        "--replan-threshold-frames", type=int, default=16,
        help=(
            "Worker thread refills the ring buffer when frames_remaining < this "
            "value. With OUTPUT_FPS=50, 16 frames = 0.32 s of headroom -- safe "
            "even with 5--15 ms predict() jitter."
        ),
    )
    tune_grp = p.add_argument_group(
        "velocity tuning",
        "Runtime knobs for compensating model-side asymmetries and quelling "
        "low-frequency yaw drift without rebuilding the static intent tables.",
    )
    tune_grp.add_argument(
        "--yaw-lock-epsilon", type=float, default=0.0,
        help=(
            "Opt-in diagnostic mitigation. When > 0 and the commanded "
            "yaw_rate is below this threshold (rad/s) the kplanner "
            "overrides the published root_quat with the persisted yaw, "
            "suppressing cumulative yaw drift during translation intents. "
            "DEFAULT 0.0 (disabled) -- enabling this also kills the gait's "
            "natural yaw oscillation, which the SONIC policy needs as a "
            "phase signal to initiate stepping; the robot stops walking "
            "entirely. Use only for diagnostics (e.g. confirm the spin is "
            "from yaw drift rather than a joint-tracking failure)."
        ),
    )
    tune_grp.add_argument(
        "--turn-left-scale", type=float, default=1.0,
        help=(
            "Multiplier on turn_left yaw_rate intents only. >1.0 boosts left "
            "turns to compensate for training-data L/R asymmetry; <1.0 "
            "attenuates them. Default 1.0 (no-op)."
        ),
    )
    tune_grp.add_argument(
        "--turn-right-scale", type=float, default=1.0,
        help="Symmetric multiplier on turn_right yaw_rate. Default 1.0.",
    )
    tune_grp.add_argument(
        "--forward-scale", type=float, default=1.0,
        help="Multiplier on fwd_step / walk-forward vel_x. Default 1.0.",
    )
    tune_grp.add_argument(
        "--backward-scale", type=float, default=1.0,
        help="Multiplier on back_step / walk-backward vel_x. Default 1.0.",
    )
    tune_grp.add_argument(
        "--lateral-scale", type=float, default=1.0,
        help="Multiplier on side_left / side_right vel_y. Default 1.0.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(
        vqvae_ckpt=args.vqvae_ckpt,
        pose_ckpt=args.pose_ckpt,
        root_ckpt=args.root_ckpt,
        warmup_qpos_path=args.warmup_qpos_path,
        pub_host=args.pub_host,
        pub_port=args.pub_port,
        pid_file=args.pid_file,
        demo_yaml=args.demo,
        enable_keyboard=args.keyboard,
        zmq_cmd_host=args.zmq_cmd_host,
        zmq_cmd_port=args.zmq_cmd_port,
        zmq_cmd_topic=args.zmq_cmd_topic,
        duration_s=args.duration_s,
        hand_dof=args.hand_dof,
        verbose=args.verbose,
        warmup_quiet_stand_s=args.warmup_quiet_stand_s,
        body_pose_port=args.body_pose_port,
        device=args.device,
        replan_threshold_frames=args.replan_threshold_frames,
        yaw_lock_epsilon_rad_s=args.yaw_lock_epsilon,
        turn_left_scale=args.turn_left_scale,
        turn_right_scale=args.turn_right_scale,
        forward_scale=args.forward_scale,
        backward_scale=args.backward_scale,
        lateral_scale=args.lateral_scale,
    )


if __name__ == "__main__":
    raise SystemExit(main())
