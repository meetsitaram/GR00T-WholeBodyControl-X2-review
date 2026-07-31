#!/usr/bin/env python3
"""Torch-free X2 kinematic planner runtime for PC2 (Jetson).

A slim port of ``gear_sonic/scripts/x2_kplanner.py``'s publish/replan loop
that runs the fused planner graph through **onnxruntime** (CPU EP) instead
of torch, so the whole planner stack can live on the robot's Jetson with
no laptop in the loop.

Architecture (all on PC2):

  - SUB ``planner_cmd``  on tcp://127.0.0.1:5563 (published by
    ``pad_locomotion_bridge --bind``; JSON payloads
    ``{"intent":"locomotion","magnitude":"continuous","stick_fwd",
    "stick_side","stick_yaw"[,"speed_delta"]}``).
  - Ring buffer of planner qpos frames ([T, 38] MuJoCo qpos, world frame,
    wxyz root quat) refilled by a worker thread whenever occupancy drops
    below ``--replan-threshold-frames`` -- scheduling / threshold /
    cadence logic ported from ``x2_kplanner._planner_worker`` +
    ``NeuralPlannerCore.get_next_frame``.
  - Replan backend: the ONNX fused graph exported by
    ``motionbricks/scripts/export_x2_planner_onnx.py``:

        inputs : context_mujoco_qpos f32 [1, 4, 38]
                 velocity_intent     f32 [1, 4]
                 (template graph)    mode i64 [1], random_seed i64 [1]
        outputs: mujoco_qpos         f32 [1, 64, 38] (padded)
                 num_pred_frames     i64 [1]

    Tensor names / graph flavour are data-driven from an optional JSON
    sidecar next to the .onnx (see ``_load_onnx_contract``); the graph
    bakes the FILTER_QPOS first-4-frame context blend in-graph
    (``x2_kplanner`` never applies it Python-side -- it lives inside
    ``NeuralPlannerCore._predict_*`` which the export traces), so the
    runtime must NOT reapply it unless the sidecar says
    ``"filter_qpos_baked": false``.
  - ``--backend torch`` (laptop A/B only): drives the *real*
    ``motionbricks`` ``load_x2_planner`` + ``NeuralPlannerCore`` exactly
    like ``x2_kplanner.run()`` does, proving this file's glue independent
    of the ONNX graph. torch is imported lazily inside that branch only.
  - 50 Hz publisher: PUB bind tcp://*:5556 topic ``pose``, VLA wire
    format (v4 packed message). The frame dict is constructed with the
    exact field set + insertion order of
    ``gear_sonic.utils.planner.state_machine.build_pose_payload`` (the
    consumer decodes by order); byte-encoded via
    ``gear_sonic.utils.pose_pipeline.wire.pack_pose_message`` which is
    byte-identical to the ``zmq_planner_sender`` encoder x2_kplanner uses
    when called with ``version=4``.
  - Frame pacing: identical to x2_kplanner -- the publish loop pops ONE
    ring-buffer frame per 50 Hz tick (``get_next_frame`` clamps at the
    buffer tail), i.e. the model's 30 fps frames are consumed at the
    50 Hz wire rate with no interpolation, and the 9-slot future window
    peeks at ``cursor + 5*(k+1) - 1`` (0.1 s spacing on the wire clock).
  - DANCE PLAYBACK: SUB bind tcp://*:5568 topic ``motion_clip_cmd``
    accepting the ``play_locomotion`` payload
    ``{"action":"play","pkl":...,"motion_key":k,"kind":"locomotion"}``
    and ``{"action":"stop"}``. On play the x2m2 bake
    ``<dances-dir>/<motion_key>.x2m2`` is streamed through the same
    50 Hz publisher (planner output preempted, ring paused); root quats
    are yaw-rebased onto the CURRENT streamed heading (delta-rebase via
    ``rebase_quats_xyzw_by_yaw``, mirroring the fallback ladder's
    ``build_idle_frame_msg`` rebase) so the dance never snaps the robot's
    heading. On clip end / stop, the idle anchor pose streams for
    ``--post-dance-idle-s`` (~2 s) before normal planner idle resumes.

Dependency budget (PC2 Jetson venv): stdlib + numpy + pyzmq + joblib +
onnxruntime + ``gear_sonic.utils.pose_pipeline.*`` (numpy-only). NO torch
import outside the ``--backend torch`` branch.

Known deviations from x2_kplanner (all intentional, documented):
  - No closed-loop robot_pose reseed / yaw refresh (x2_kplanner's default
    config also runs open-loop; real-robot deploys used reseed scope
    ``none`` anyway).
  - No waist-overlay path (``hold_torso`` waist_*_deg are ignored; the
    pad bridge never emits them). ``hold_torso`` still resolves to idle
    with the optional ``hip_height_m`` override, as upstream.
  - No scripted-YAML / keyboard command sources (ZMQ only).

PC2 launch (defaults are the real PC2 ports/paths)::

    PYTHONPATH=/home/run/getsolo/planner_stack/gear_sonic \
    python pc2_kplanner_onnx.py \
        --onnx /home/run/getsolo/planner_stack/models/planner_onnx/x2_planner_velocity.onnx

Laptop A/B (live stack owns 5556/5563/5568 -- offset everything)::

    .venv/bin/python gear_sonic/scripts/pc2_kplanner_onnx.py \
        --backend torch --port-offset 100 --device cpu \
        --vqvae-ckpt ... --pose-ckpt ... --root-ckpt ... \
        --warmup-qpos gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import queue
import random
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# Make ``gear_sonic`` importable when launched as a plain file with the
# repo (or the PC2 planner_stack copy) as the parent tree.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.pose_pipeline.wire import (  # noqa: E402
    DEFAULT_HAND_DOF,
    NUM_BODY_DOFS,
    SONIC_MOTION_TOKEN_DIM,
    decode_x2_debug_base_quat,
    load_x2m2,
    pack_pose_message,
    rebase_quats_xyzw_by_yaw,
    yaw_from_quat_wxyz,
)

log = logging.getLogger("pc2_kplanner_onnx")

# ---------------------------------------------------------------------------
# Wire / timing constants (mirror x2_kplanner + state_machine.OUTPUT_FPS).
# ---------------------------------------------------------------------------
OUTPUT_FPS: float = 50.0
MODEL_FPS: float = 30.0     # kplanner model output rate (resample source rate)

QPOS_DIM: int = 38
NUM_FUTURE: int = 9         # future-window slots on the wire
STEP_TICKS: int = 5         # 50 Hz ticks between future slots (= 0.1 s)
FUTURE_DT_S: float = 0.1

# PC2 defaults (document the real deployment surface).
DEFAULT_PUB_PORT: int = 5556           # PUB bind, topic "pose"
DEFAULT_CMD_PORT: int = 5563           # SUB connect, topic "planner_cmd"
DEFAULT_CLIP_CMD_PORT: int = 5568      # SUB bind, topic "motion_clip_cmd"
DEFAULT_WARMUP_PKL = Path(
    "/home/run/getsolo/planner_stack/models/kplanner_idle_anchor_g1teleop_v3.pkl"
)
DEFAULT_DANCES_DIR = Path(
    "/home/run/getsolo/planner_stack/models/dances_x2m2"
)


def _slerp_wxyz_np(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    """Shortest-arc SLERP between two wxyz quaternions (numpy port).

    Mirrors ``NeuralPlannerCore._slerp_wxyz`` so the ONNX runtime resamples
    the root orientation identically to the torch stack.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    q0 = q0 / (np.linalg.norm(q0) + 1e-8)
    q1 = q1 / (np.linalg.norm(q1) + 1e-8)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + w * (q1 - q0)
        return (out / (np.linalg.norm(out) + 1e-8)).astype(np.float32)
    theta0 = math.acos(max(-1.0, min(1.0, dot)))
    theta = theta0 * w
    sin0 = math.sin(theta0)
    s0 = math.sin(theta0 - theta) / sin0
    s1 = math.sin(theta) / sin0
    out = s0 * q0 + s1 * q1
    return (out / (np.linalg.norm(out) + 1e-8)).astype(np.float32)

# ---------------------------------------------------------------------------
# Intent -> velocity dispatcher. Ported verbatim (numbers + shaping) from
# x2_kplanner.py; see that file for the full channel-convention rationale.
# Velocity tuple layout: (yaw_rate_rad_s, vel_x=lateral, vel_z=forward, hip_h).
# ---------------------------------------------------------------------------
_WALK_SPEED_MPS: float = 0.6
_FAST_WALK_SPEED_MPS: float = 0.9
_SIDE_SPEED_MPS: float = 0.4
_BACK_SPEED_MPS: float = 0.45

_TEST_FIXED_BACK_MPS: float = 0.30
_TEST_FIXED_SIDE_MPS: float = 0.30
# Turn-rate setpoint. 0.30 was the original conservative test value; the
# 2026-07-21 sweep (kplanner_turnrate_sweep.py) showed the root head scales
# to 1.3 rad/s with NO still chunks at >=0.8 (strong conditioning escapes
# the idle attractor and the pose head generates real turn stepping), and
# the robot tracks the reference heading ~1:1. Override like the forward
# setpoint: env KPLANNER_FIXED_TURN_RAD_S.
_TEST_FIXED_TURN_RAD_S: float = float(
    os.environ.get("KPLANNER_FIXED_TURN_RAD_S") or 0.30)
# Arc-turn rate: yaw applied WHILE walking. At the standing-turn rate the
# arc is too tight ("turns much more than walking" -- operator, sim,
# 2026-07-21); walking + full yaw makes the heading win over travel. A
# separate, lower setpoint gives a good turning walk without giving up the
# brisk standing 360s.
_TEST_FIXED_ARC_TURN_RAD_S: float = float(
    os.environ.get("KPLANNER_FIXED_ARC_TURN_RAD_S") or 0.55)
# Optional arc forward boost (default 1.0 = NO change). SONIC's obs carries
# no reference root translation -- forward progress is implied by gait
# joints while heading error is explicit -- so tracked arcs under-translate
# vs the reference (sim, 2026-07-21). A boost >1 over-commands forward
# during arcs to compensate AT THE INTENT LAYER. Deliberately opt-in: it is
# a tracker-era workaround that a better future model should not inherit.
_ARC_FWD_BOOST: float = float(
    os.environ.get("KPLANNER_ARC_FWD_BOOST") or 1.0)
_FWD_LATERAL_DEADBAND: float = 0.35

# Strafe gating (2026-07-18). The resolver below is a SIGN function: any
# non-zero lateral component becomes a FULL _TEST_FIXED_SIDE_MPS strafe. With
# forward and lateral sharing one stick, a mild diagonal push therefore
# injected a full 0.3 m/s side-step into what the operator intended as a
# walking turn -- observed on hardware as intents (yaw=+0.3, vel_x=-0.3,
# vel_z=+0.3) and felt as "unclean steps". There is also no good side-step
# clip in the corpus, so an unintended strafe is worse than no strafe.
#
# The old magnitude-only rule (|side| < 0.35 while moving forward) was too
# permissive: a 45 deg push has |side| ~= 0.7 and sailed through. Replaced by
# an ANGLE rule -- lateral must dominate, i.e. the push must be close to pure
# sideways. Strafe engages only when BOTH hold:
#   * |side| >= _LATERAL_MIN_MAG        (a decisive push, not a lean)
#   * |fwd|  <= |side| * tan(theta_max) (within theta_max of the lateral axis)
# tan(25 deg) ~= 0.466, so a 45 deg diagonal (ratio 1.0) is rejected and only a
# near-90-degree push strafes. Raise _LATERAL_MAX_TAN to loosen.
_LATERAL_MIN_MAG: float = 0.60
_LATERAL_MAX_TAN: float = 0.466   # tan(25 deg)

_TURN_15_RAD_S: float = 0.5
_TURN_30_RAD_S: float = 1.0
_TURN_45_RAD_S: float = 1.5
_TURN_90_RAD_S: float = 3.0

_HIP_HEIGHT_M: float = 0.687
_IDLE_INTENT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, _HIP_HEIGHT_M)

HOLD_TORSO_INTENT: str = "hold_torso"

# Runtime forward-speed SETPOINT (m/s): any forward stick deflection past
# the deadzone commands exactly this speed. Nudged live via the payload's
# one-shot "speed_delta" field, clamped to [_SETPOINT_MIN, _SETPOINT_MAX].
_SETPOINT_MIN: float = 0.2
_SETPOINT_MAX: float = 1.0
_SPEED_SETPOINT: float = float(os.environ.get("KPLANNER_FIXED_FWD_MPS") or 0.50)

_DEFAULT_CONTINUOUS_FORWARD_MIN_MPS: float = 0.30
_RUNTIME_CONTINUOUS_FORWARD_MIN_MPS: float = _DEFAULT_CONTINUOUS_FORWARD_MIN_MPS
_DEFAULT_STICK_SHAPING_EXPONENT: float = 1.0
_RUNTIME_STICK_SHAPING_EXPONENT: float = _DEFAULT_STICK_SHAPING_EXPONENT
_RUNTIME_TURN_LEFT_SCALE: float = 1.0
_RUNTIME_TURN_RIGHT_SCALE: float = 1.0
_RUNTIME_FORWARD_SCALE: float = 1.0
_RUNTIME_BACKWARD_SCALE: float = 1.0
_RUNTIME_LATERAL_SCALE: float = 1.0


def _adjust_speed_setpoint(delta: float) -> float:
    """Nudge the runtime forward-speed setpoint; returns the new value."""
    global _SPEED_SETPOINT
    _SPEED_SETPOINT = max(_SETPOINT_MIN, min(_SETPOINT_MAX, _SPEED_SETPOINT + delta))
    log.info("speed setpoint %+0.1f -> %.1f m/s", delta, _SPEED_SETPOINT)
    return _SPEED_SETPOINT


def _shape_stick(value: float) -> float:
    """sign(v) * |v|**exp -- port of x2_kplanner._shape_stick."""
    sign = 1.0 if value >= 0 else -1.0
    mag = abs(float(value))
    if mag == 0.0:
        return 0.0
    return sign * mag ** _RUNTIME_STICK_SHAPING_EXPONENT


def _resolve_locomotion_continuous(
    stick_fwd: float, stick_side: float, stick_yaw: float
) -> tuple[float, float, float, float]:
    """Continuous VR/pad stick resolver (port of x2_kplanner's)."""
    shaped_fwd = _shape_stick(stick_fwd)
    shaped_side = _shape_stick(stick_side)
    shaped_yaw = _shape_stick(stick_yaw)

    # Strafe requires a near-pure sideways push (see _LATERAL_MIN_MAG /
    # _LATERAL_MAX_TAN). Applies in BOTH travel directions -- the old rule only
    # gated lateral while moving forward, so a diagonal pull-back still strafed.
    if shaped_side != 0.0:
        if (abs(shaped_side) < _LATERAL_MIN_MAG
                or abs(shaped_fwd) > abs(shaped_side) * _LATERAL_MAX_TAN):
            shaped_side = 0.0

    if shaped_fwd > 0.0:
        vel_z = _SPEED_SETPOINT           # deterministic setpoint mode
    elif shaped_fwd < 0.0:
        vel_z = -_TEST_FIXED_BACK_MPS
    else:
        vel_z = 0.0
    if shaped_side > 0.0:
        vel_x = -_TEST_FIXED_SIDE_MPS     # stick right -> side_right -> -vel_x
    elif shaped_side < 0.0:
        vel_x = _TEST_FIXED_SIDE_MPS
    else:
        vel_x = 0.0
    turn_mag = (_TEST_FIXED_ARC_TURN_RAD_S if vel_z != 0.0
                else _TEST_FIXED_TURN_RAD_S)
    if shaped_yaw > 0.0:
        yaw_rate = -turn_mag              # stick right -> turn-right -> -yaw
    elif shaped_yaw < 0.0:
        yaw_rate = turn_mag
    else:
        yaw_rate = 0.0
    if yaw_rate != 0.0 and vel_z > 0.0 and _ARC_FWD_BOOST != 1.0:
        vel_z = vel_z * _ARC_FWD_BOOST
    return (yaw_rate, vel_x, vel_z, _HIP_HEIGHT_M)


# Bucketed (legacy) table -- kept for scripted / manager vocabulary parity.
_BASE_VELOCITY: dict[str, tuple[float, float, float, float]] = {
    "idle":       (0.0,             0.0,              0.0,             _HIP_HEIGHT_M),
    "fwd_step":   (0.0,             0.0,              _WALK_SPEED_MPS, _HIP_HEIGHT_M),
    "back_step":  (0.0,             0.0,             -_BACK_SPEED_MPS, _HIP_HEIGHT_M),
    "side_left":  (0.0,             _SIDE_SPEED_MPS,  0.0,             _HIP_HEIGHT_M),
    "side_right": (0.0,            -_SIDE_SPEED_MPS,  0.0,             _HIP_HEIGHT_M),
    "turn_left":  ( _TURN_45_RAD_S, 0.0,              0.0,             _HIP_HEIGHT_M),
    "turn_right": (-_TURN_45_RAD_S, 0.0,              0.0,             _HIP_HEIGHT_M),
}
_TRANSLATIONAL_SCALE: dict[str, float] = {
    "default": 1.0, "stand": 0.0, "quarter_ft": 0.5, "half_ft": 1.0, "one_ft": 1.5,
}
_TURN_SCALE: dict[str, float] = {
    "default": 1.0,
    "deg_15": _TURN_15_RAD_S / _TURN_45_RAD_S,
    "deg_30": _TURN_30_RAD_S / _TURN_45_RAD_S,
    "deg_45": 1.0,
    "deg_90": _TURN_90_RAD_S / _TURN_45_RAD_S,
}
_ROTATIONAL_INTENTS: frozenset[str] = frozenset({"turn_left", "turn_right"})
_WALK_VELOCITY_BY_MAGNITUDE: dict[str, tuple[float, float, float, float]] = {
    "forward":  (0.0, 0.0,  _WALK_SPEED_MPS,      _HIP_HEIGHT_M),
    "backward": (0.0, 0.0, -_BACK_SPEED_MPS,      _HIP_HEIGHT_M),
    "fast":     (0.0, 0.0,  _FAST_WALK_SPEED_MPS, _HIP_HEIGHT_M),
}


@dataclass(frozen=True)
class LocomotionCommand:
    """Slim local mirror of state_machine.LocomotionCommand (fields we use)."""

    intent: str
    magnitude: str = "default"
    source: str = "zmq"
    stick_fwd: float = 0.0
    stick_side: float = 0.0
    stick_yaw: float = 0.0
    direct_velocity: Optional[tuple[float, float, float, float]] = None
    hip_height_m: Optional[float] = None


def _resolve_velocity(intent: str, magnitude: str) -> tuple[float, float, float, float]:
    if intent == "walk":
        return _WALK_VELOCITY_BY_MAGNITUDE.get(magnitude, _IDLE_INTENT)
    base = _BASE_VELOCITY.get(intent)
    if base is None:
        return _IDLE_INTENT
    yaw, vx, vy, hip_h = base
    if intent in _ROTATIONAL_INTENTS:
        return (yaw * _TURN_SCALE.get(magnitude, 1.0), vx, vy, hip_h)
    scale = _TRANSLATIONAL_SCALE.get(magnitude, 1.0)
    return (yaw, vx * scale, vy * scale, hip_h)


def _apply_runtime_scales(
    intent: str, velocity: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    yaw, vel_x, vel_z, hip_h = velocity
    if intent == "turn_left":
        yaw *= _RUNTIME_TURN_LEFT_SCALE
    elif intent == "turn_right":
        yaw *= _RUNTIME_TURN_RIGHT_SCALE
    elif intent in ("fwd_step",) or (intent == "walk" and vel_z > 0):
        vel_z *= _RUNTIME_FORWARD_SCALE
    elif intent in ("back_step",) or (intent == "walk" and vel_z < 0):
        vel_z *= _RUNTIME_BACKWARD_SCALE
    elif intent in ("side_left", "side_right"):
        vel_x *= _RUNTIME_LATERAL_SCALE
    return (yaw, vel_x, vel_z, hip_h)


def _apply_continuous_runtime_scales(
    velocity: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    yaw, vx, vz, hip_h = velocity
    if yaw > 0:
        yaw *= _RUNTIME_TURN_LEFT_SCALE
    elif yaw < 0:
        yaw *= _RUNTIME_TURN_RIGHT_SCALE
    if vz > 0:
        vz *= _RUNTIME_FORWARD_SCALE
        if _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS > 0.0:
            vz = max(vz, _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS)
    elif vz < 0:
        vz *= _RUNTIME_BACKWARD_SCALE
    vx *= _RUNTIME_LATERAL_SCALE
    return (yaw, vx, vz, hip_h)


# ---- forward-obstacle guard -------------------------------------------------
# Subscribes to scan_guard_pub.py over ZMQ rather than importing rclpy: this
# process runs in the gear_sonic venv, which has no ROS bindings, and the gait
# loop is the last place to add an import that can fail.
#
# Clamped HERE (planner_cmd ingest) rather than in the pad bridge so every
# input source is covered -- pad, Quest/VR, anything else that publishes to
# this socket.
#
# FAIL-OPEN: stale or absent guard data means no clamping. Freezing a walking
# biped because a sensor process died is worse than not clamping; the operator
# deadman is the real stop.
_GUARD_PORT = 5571
_GUARD_STALE_S = 0.5
_guard_state = {"blocked": False, "dist": float("inf"), "ts": 0.0}
# Latched stop: once an obstacle trips the guard, forward stays dead until
# the operator releases the deadman (all sticks zero). Auto-release on a
# clear path would let the robot resume walking without a human deciding to.
_guard_latched = {"on": False}

# ---- command-source ownership (pad vs VR mutual exclusion) ------------------
# Enforced in _zmq_command_thread. VR supersedes pad while engaged; release
# is explicit (VR idle on disengage) with a crash-safety silence timeout.
# The VR manager keepalives every 0.5 s while engaged, so 2.0 s can only
# expire when the manager is actually gone (crash / network loss).
_VR_OWNER_TIMEOUT_S = 2.0
_cmd_owner = {"src": None, "vr_ts": 0.0}
# Set by main() when --arm-port is active; ownership release clears it.
_ARM_INGEST_REF: list = [None]


def _scan_guard_thread(stop_event) -> None:
    import zmq as _zmq
    ctx = _zmq.Context.instance()
    sock = ctx.socket(_zmq.SUB)
    sock.setsockopt(_zmq.SUBSCRIBE, b"scan_guard")
    sock.setsockopt(_zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://127.0.0.1:{_GUARD_PORT}")
    log.info("scan guard: SUB tcp://127.0.0.1:%d", _GUARD_PORT)
    while not stop_event.is_set():
        try:
            _, payload = sock.recv_multipart()
        except Exception:  # noqa: BLE001 -- timeout is normal
            continue
        try:
            d = json.loads(payload)
            _guard_state["blocked"] = bool(d["blocked"])
            _guard_state["dist"] = float(d["dist"])
            _guard_state["ts"] = time.monotonic()
        except Exception:  # noqa: BLE001
            continue
    sock.close(linger=0)


def _guard_blocked() -> bool:
    if time.monotonic() - _guard_state["ts"] > _GUARD_STALE_S:
        return False
    return _guard_state["blocked"]


def intent_to_velocity(cmd: LocomotionCommand) -> tuple[float, float, float, float]:
    """LocomotionCommand -> 4-D velocity; port of x2_kplanner.intent_to_velocity."""
    if cmd.direct_velocity is not None:
        yaw, vx, vz, hip_h = cmd.direct_velocity
        return (float(yaw), float(vx), float(vz), float(hip_h))
    if cmd.intent == "locomotion" and cmd.magnitude == "continuous":
        result = _resolve_locomotion_continuous(
            cmd.stick_fwd, cmd.stick_side, cmd.stick_yaw
        )
        return _apply_continuous_runtime_scales(result)
    if cmd.intent == HOLD_TORSO_INTENT:
        yaw_idle, vx_idle, vz_idle, hip_idle = _IDLE_INTENT
        hip_h = (
            float(cmd.hip_height_m) if cmd.hip_height_m is not None
            else float(hip_idle)
        )
        return (float(yaw_idle), float(vx_idle), float(vz_idle), hip_h)
    result = _resolve_velocity(cmd.intent, cmd.magnitude)
    if result == _IDLE_INTENT and cmd.intent != "idle":
        log.debug("intent %s,%s has no velocity mapping; idling",
                  cmd.intent, cmd.magnitude)
        return result
    return _apply_runtime_scales(cmd.intent, result)


# ---------------------------------------------------------------------------
# Process hygiene (port of x2_kplanner's).
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


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class PidFile:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> "PidFile":
        if self.path.exists():
            try:
                old = int(self.path.read_text().strip())
                if _pid_alive(old):
                    raise RuntimeError(
                        f"PID file {self.path} exists and PID {old} is alive -- "
                        f"another planner is running. `kill {old}` first."
                    )
                log.warning("stale PID file %s for dead PID %d, cleaning up",
                            self.path, old)
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


# ---------------------------------------------------------------------------
# Warmup / idle anchor qpos loading (port of x2_kplanner._load_warmup_qpos).
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


def _build_training_default_qpos() -> np.ndarray:
    qpos = np.zeros(QPOS_DIM, dtype=np.float32)
    qpos[2] = _TRAINING_DEFAULT_HIP_Z
    qpos[3] = 1.0  # wxyz identity
    qpos[7:38] = _TRAINING_DEFAULT_ANGLES
    return qpos


def _qpos_from_deploy_pkl_frame(obj_inner: dict) -> Optional[np.ndarray]:
    """Deploy-PKL schema frame 0 -> qpos[38] wxyz (root_rot stored xyzw)."""
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
    qpos = np.zeros(QPOS_DIM, dtype=np.float32)
    qpos[0:3] = f0_trans
    qpos[3:7] = [f0_rot_xyzw[3], f0_rot_xyzw[0], f0_rot_xyzw[1], f0_rot_xyzw[2]]
    qpos[7:38] = f0_dof
    return qpos


def _load_warmup_qpos(path: Optional[Path]) -> np.ndarray:
    """Load a [38]-D anchor qpos from a (joblib) PKL, with the same schema
    tolerance as x2_kplanner; falls back to training_default_angles."""
    if path is None:
        qpos = _build_training_default_qpos()
        log.info("warmup anchor: training_default_angles stand (hip_z=%.3fm)",
                 float(qpos[2]))
        return qpos
    if not path.is_file():
        log.warning("warmup-qpos %s not found; falling back to training default",
                    path)
        return _build_training_default_qpos()
    try:
        import joblib
        obj = joblib.load(path)
    except Exception:  # noqa: BLE001 -- fall through to raw pickle
        with path.open("rb") as f:
            obj = pickle.load(f)
    if isinstance(obj, dict):
        arr = obj.get("mujoco_qpos", obj.get("qpos", None))
        if arr is None:
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
                f"warmup PKL {path} has no recognisable qpos / deploy-PKL "
                f"schema (keys={list(obj.keys())})"
            )
    else:
        arr = obj
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[0]
    if arr.shape[-1] != QPOS_DIM:
        raise ValueError(f"warmup PKL {path} qpos shape {arr.shape} != [38]")
    log.info("warmup anchor: loaded from %s (shape=[38])", path)
    return arr


# ---------------------------------------------------------------------------
# Replan backends. Both implement the exact NeuralPlannerCore ring-buffer
# semantics x2_kplanner relies on:
#   * reset(qpos)      -> buffer := 64 tiled copies, cursor := 0
#   * get_next_frame() -> pop buf[clamp(cursor)], cursor := min(cursor+1, T-1)
#   * context          -> buf[[clamp(cursor + i) for i in 0..3]]
#                         (idx - NUM_FT + i + PRED_OFFSETS with the default
#                          PRED_OFFSETS == NUM_FT == 4)
#   * replan(target)   -> buffer := prediction[:num_pred_frames], cursor := 0
#                         (NUM_FT - PRED_OFFSETS)
#   * should_replan()  -> frames_remaining <= threshold
# ---------------------------------------------------------------------------

NUM_FRAMES_PER_TOKEN: int = 4
PRED_OFFSETS: int = 4
NUM_MIN_FRAMES_IN_BUFFER: int = 64

_PLANNER_MODE_NAMES: tuple[str, ...] = ("idle", "slow_walk", "walk", "run_proxy")

# Default ONNX runtime contract -- mirrors the interface documented in
# motionbricks/scripts/export_x2_planner_onnx.py. Every key can be
# overridden by a JSON sidecar next to the graph so a re-export with
# different tensor names / extra conditioning tensors needs no code change.
_DEFAULT_ONNX_CONTRACT: dict = {
    # role -> graph input tensor name
    "inputs": {
        "context": "context_mujoco_qpos",     # f32 [1, 4, 38]
        "velocity": "velocity_intent",        # f32 [1, 4]
        "mode": "mode",                       # i64 [1]   (template graph)
        "random_seed": "random_seed",         # i64 [1]   (template graph)
    },
    # role -> graph output tensor name
    "outputs": {
        "qpos": "mujoco_qpos",                # f32 [1, 64, 38]
        "num_pred_frames": "num_pred_frames", # i64 [1]
    },
    # The export bakes the FILTER_QPOS first-4-frame context blend
    # in-graph (it lives inside NeuralPlannerCore._predict_* which the
    # trace covers); x2_kplanner never applies it Python-side. Set false
    # in the sidecar only for a graph exported without the blend.
    "filter_qpos_baked": True,
    # mode-name -> index binding (build_x2_planner_clips DEFAULT_MODES).
    "modes": list(_PLANNER_MODE_NAMES),
    # Optional: dense constant extra inputs {tensor_name: {"dtype": "i64"
    # or "f32", "value": [...], "shape": [...]}} fed verbatim per replan.
    "extra_inputs": {},
    # Optional: fixed random seed (int) instead of per-replan random.
    "fixed_random_seed": None,
    # Optional: "template" / "velocity"; default auto-detect from the
    # session's input names (template iff the mode input exists).
    "graph_kind": None,
    # Optional default mode name for template graphs when --planner-mode
    # is not passed.
    "default_mode": "walk",
}


def _load_onnx_contract(onnx_path: Path, sidecar: Optional[Path]) -> dict:
    """Merge the JSON sidecar (if any) over the default contract.

    Search order when --onnx-sidecar is not given: ``<onnx>.json`` then
    ``<dir>/runtime_contract.json``. Missing sidecar -> pure defaults
    (matches the current export script's tensor names).
    """
    contract = json.loads(json.dumps(_DEFAULT_ONNX_CONTRACT))  # deep copy
    candidates = (
        [sidecar] if sidecar is not None
        else [onnx_path.with_suffix(onnx_path.suffix + ".json"),
              onnx_path.parent / "runtime_contract.json"]
    )
    for cand in candidates:
        if cand is not None and Path(cand).is_file():
            raw = json.loads(Path(cand).read_text())
            for key, val in raw.items():
                if key in ("inputs", "outputs") and isinstance(val, dict):
                    contract[key].update(val)
                else:
                    contract[key] = val
            log.info("onnx contract: merged sidecar %s", cand)
            return contract
    if sidecar is not None:
        raise FileNotFoundError(f"--onnx-sidecar {sidecar} not found")
    log.info("onnx contract: no sidecar found; using export-script defaults")
    return contract


class OnnxPlannerBackend:
    """Ring buffer + onnxruntime fused-graph replan (torch-free)."""

    # Flipped by --ort-gpu in main(). Class-level so the flag set once at
    # startup reaches every backend instance without threading it through.
    USE_GPU: bool = False

    def __init__(
        self,
        onnx_path: Path,
        contract: dict,
        replan_threshold_frames: int,
        planner_mode: Optional[str],
    ) -> None:
        import onnxruntime as ort

        # Provider order: GPU (CUDA) first with CPU fallback when --ort-gpu is
        # set, else CPU-only (the safe default that has always shipped). ORT
        # silently drops any provider not present in the build, so on a CPU-only
        # onnxruntime this still runs on CPU -- the flag is a no-op until a
        # Jetson GPU build of onnxruntime is installed in the venv.
        # NOTE: CUDA kernels may sample differently from CPU for the same
        # random_seed. Fine for deploy; but an intent-tape replay must use the
        # SAME provider it was recorded under to stay bit-exact.
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if OnnxPlannerBackend.USE_GPU
                     else ["CPUExecutionProvider"])
        if OnnxPlannerBackend.USE_GPU and hasattr(ort, "preload_dlls"):
            # Desktop pip installs ship CUDA/cuDNN as nvidia-* wheels that
            # are not on the loader path; preload_dlls() (ORT >= 1.21) loads
            # them from site-packages. No-op where libs resolve system-wide
            # (Jetson) -- guarded so a CPU-only or old ORT is unaffected.
            try:
                ort.preload_dlls()
            except Exception as exc:  # never let GPU setup kill the daemon
                log.warning("ort.preload_dlls() failed (%s); continuing", exc)
        t0 = time.monotonic()
        self._sess = ort.InferenceSession(str(onnx_path), providers=providers)
        _active = self._sess.get_providers()
        log.info("onnxruntime providers requested=%s active=%s", providers, _active)
        if OnnxPlannerBackend.USE_GPU and "CUDAExecutionProvider" not in _active:
            log.warning("--ort-gpu set but CUDAExecutionProvider NOT active -- "
                        "onnxruntime build lacks CUDA; running on CPU.")
        self._contract = contract
        sess_inputs = {i.name for i in self._sess.get_inputs()}
        sess_outputs = [o.name for o in self._sess.get_outputs()]
        roles = contract["inputs"]

        kind = contract.get("graph_kind")
        if kind is None:
            kind = "template" if roles["mode"] in sess_inputs else "velocity"
        self.graph_kind = kind

        # Validate the input roles we intend to feed actually exist.
        needed = [roles["context"], roles["velocity"]]
        if kind == "template":
            needed += [roles["mode"], roles["random_seed"]]
        missing = [n for n in needed if n not in sess_inputs]
        if missing:
            raise ValueError(
                f"ONNX graph {onnx_path} missing expected inputs {missing}; "
                f"graph has {sorted(sess_inputs)}. Fix the sidecar 'inputs' "
                f"mapping."
            )
        for out_name in contract["outputs"].values():
            if out_name not in sess_outputs:
                raise ValueError(
                    f"ONNX graph {onnx_path} missing output {out_name!r}; "
                    f"graph has {sess_outputs}. Fix the sidecar 'outputs'."
                )
        # Any session input not covered by roles/extra_inputs is an error
        # (a silent zero-feed would corrupt predictions).
        covered = set(needed) | set(contract.get("extra_inputs", {}))
        uncovered = sess_inputs - covered
        if uncovered:
            raise ValueError(
                f"ONNX graph has inputs {sorted(uncovered)} not covered by "
                f"the runtime contract; add them to the sidecar's "
                f"'extra_inputs'."
            )

        # Mode index resolution (template graphs only).
        self.mode_idx: Optional[int] = None
        if kind == "template":
            modes = list(contract.get("modes") or _PLANNER_MODE_NAMES)
            mode_name = planner_mode or contract.get("default_mode") or "walk"
            if mode_name not in modes:
                raise ValueError(
                    f"--planner-mode {mode_name!r} not in contract modes {modes}"
                )
            self.mode_idx = modes.index(mode_name)
            if planner_mode is None:
                log.warning(
                    "template graph with no --planner-mode; defaulting to "
                    "%r (idx=%d)", mode_name, self.mode_idx,
                )
            else:
                log.info("pose-template inference: mode=%s (idx=%d)",
                         mode_name, self.mode_idx)
        elif planner_mode is not None:
            log.warning(
                "--planner-mode=%s ignored: %s is a velocity-only graph",
                planner_mode, onnx_path.name,
            )

        self._fixed_seed = contract.get("fixed_random_seed")
        self._filter_baked = bool(contract.get("filter_qpos_baked", True))
        self.REPLAN_THRESHOLD_FRAMES = int(replan_threshold_frames)
        self._buf: Optional[np.ndarray] = None  # [T, 38] f32
        self._cursor = 0
        # Output resampling (30 Hz model -> OUTPUT_FPS control loop) + an
        # 8-tick cross-fade blend across replan seams. numpy port of
        # NeuralPlannerCore.get_next_frame_resampled; inert until
        # get_next_frame_resampled() is first called.
        self.BLEND_FRAMES = 8
        self._model_fps = float(MODEL_FPS)
        self._resample_active = False
        self._resample_output_fps = float(OUTPUT_FPS)
        self._read_pos = 0.0
        self._blend_prev_buf: Optional[np.ndarray] = None
        self._blend_prev_pos = 0.0
        self._blend_remaining = 0
        log.info(
            "onnx backend ready in %.2fs: %s (kind=%s, filter_qpos_baked=%s)",
            time.monotonic() - t0, onnx_path, kind, self._filter_baked,
        )

    # --- ring-buffer surface (NeuralPlannerCore-equivalent) ---------------

    @property
    def current_frame_idx(self) -> int:
        return self._cursor

    @property
    def frames_remaining(self) -> int:
        if self._buf is None:
            return 0
        if self._resample_active:
            return int(np.floor(self._buf.shape[0] - self._read_pos))
        return int(self._buf.shape[0]) - self._cursor

    def should_replan(self) -> bool:
        if self._buf is None:
            return False
        if self._resample_active:
            return (self._read_pos + self.REPLAN_THRESHOLD_FRAMES) >= float(
                self._buf.shape[0]
            )
        return self.frames_remaining <= self.REPLAN_THRESHOLD_FRAMES

    def reset(self, init_qpos: np.ndarray) -> None:
        init_qpos = np.asarray(init_qpos, dtype=np.float32).reshape(-1)
        if init_qpos.shape[0] != QPOS_DIM:
            raise ValueError(f"init qpos must be [38], got {init_qpos.shape}")
        self._buf = np.tile(init_qpos[None, :], (NUM_MIN_FRAMES_IN_BUFFER, 1))
        self._cursor = 0
        self._read_pos = 0.0
        self._blend_prev_buf = None
        self._blend_prev_pos = 0.0
        self._blend_remaining = 0

    def get_next_frame(self) -> np.ndarray:
        if self._buf is None:
            raise RuntimeError("get_next_frame() called before reset()")
        idx = max(0, min(self._cursor, self._buf.shape[0] - 1))
        self._cursor = min(self._cursor + 1, self._buf.shape[0] - 1)
        return self._buf[idx].copy()

    def peek_frame(self, idx: int) -> np.ndarray:
        assert self._buf is not None
        return self._buf[max(0, min(int(idx), self._buf.shape[0] - 1))].copy()

    # --- resampled read + 8-tick cross-fade (numpy port) ------------------

    def _frame_at(self, buf: np.ndarray, pos: float) -> np.ndarray:
        """Interpolate a single [38] qpos row at fractional index ``pos``.

        Layout ``[trans(3), root_quat_wxyz(4), dof(31)]``: lerp 0:3 and 7:,
        slerp the root quat at 3:7.
        """
        T = int(buf.shape[0])
        if T <= 1:
            return buf[0].copy()
        pos = min(max(float(pos), 0.0), float(T - 1))
        i0 = int(np.floor(pos))
        i1 = min(i0 + 1, T - 1)
        frac = pos - i0
        f0 = buf[i0]
        if i1 == i0 or frac <= 0.0:
            return f0.copy()
        f1 = buf[i1]
        out = f0.copy()
        out[:3] = f0[:3] * (1.0 - frac) + f1[:3] * frac
        out[7:] = f0[7:] * (1.0 - frac) + f1[7:] * frac
        out[3:7] = _slerp_wxyz_np(f0[3:7], f1[3:7], frac)
        return out

    def _blend_frames(
        self, old: np.ndarray, new: np.ndarray, w_new: float
    ) -> np.ndarray:
        w_old = 1.0 - w_new
        out = new.copy()
        out[:3] = old[:3] * w_old + new[:3] * w_new
        out[7:] = old[7:] * w_old + new[7:] * w_new
        out[3:7] = _slerp_wxyz_np(old[3:7], new[3:7], w_new)
        return out

    def _resampled_output_frame(self, output_offset_ticks: float = 0.0) -> np.ndarray:
        assert self._buf is not None
        step = self._model_fps / self._resample_output_fps
        new_pos = self._read_pos + step * output_offset_ticks
        frame = self._frame_at(self._buf, new_pos)
        if self._blend_prev_buf is not None:
            blend_left = self._blend_remaining - output_offset_ticks
            if blend_left > 0.0:
                old_pos = self._blend_prev_pos + step * output_offset_ticks
                old_frame = self._frame_at(self._blend_prev_buf, old_pos)
                w_new = 1.0 - (blend_left - 1.0) / float(self.BLEND_FRAMES)
                w_new = min(max(w_new, 0.0), 1.0)
                frame = self._blend_frames(old_frame, frame, w_new)
        return frame

    def get_next_frame_resampled(
        self, output_fps: Optional[float] = None
    ) -> np.ndarray:
        if self._buf is None:
            raise RuntimeError("get_next_frame_resampled() called before reset()")
        if output_fps is not None:
            self._resample_output_fps = float(output_fps)
        self._resample_active = True
        frame = self._resampled_output_frame(0.0)
        step = self._model_fps / self._resample_output_fps
        # Starvation telemetry: serving at/past the last buffered frame means
        # SONIC receives a frozen reference at full 50 Hz -- invisible to the
        # silence-based pose watchdog (tape 20260719: stumbles). Should be
        # unreachable with replan threshold 32; scream if it ever recurs.
        if self._read_pos >= self._buf.shape[0] - 1:
            self._starved_ticks = getattr(self, "_starved_ticks", 0) + 1
            if self._starved_ticks in (1, 25) or self._starved_ticks % 250 == 0:
                log.warning("ring STARVED: serving frozen end-of-buffer frame "
                            "(tick %d of this episode)", self._starved_ticks)
            _TAPE.ev("starved", n=self._starved_ticks)
        else:
            self._starved_ticks = 0
        self._read_pos += step
        self._cursor = int(np.floor(self._read_pos))
        if self._blend_prev_buf is not None:
            self._blend_prev_pos += step
            self._blend_remaining -= 1
            if self._blend_remaining <= 0:
                self._blend_prev_buf = None
                self._blend_remaining = 0
        return frame

    def peek_output_frame(self, output_offset_ticks: float) -> np.ndarray:
        if self._buf is None:
            raise RuntimeError("peek_output_frame() called before reset()")
        self._resample_active = True
        return self._resampled_output_frame(float(output_offset_ticks))

    def _get_context(self) -> np.ndarray:
        """buf[[clamp(cursor - 4 + i + PRED_OFFSETS)]] -> [1, 4, 38] f32."""
        assert self._buf is not None
        last = self._buf.shape[0] - 1
        indices = [
            max(0, min(self._cursor - NUM_FRAMES_PER_TOKEN + i + PRED_OFFSETS, last))
            for i in range(NUM_FRAMES_PER_TOKEN)
        ]
        return self._buf[indices][None, :, :].astype(np.float32)

    # --- replan ------------------------------------------------------------

    # -----------------------------------------------------------------
    # Replan is SPLIT into prepare / infer / commit so the ONNX inference
    # can run WITHOUT holding the publisher's lock.
    #
    # Why this matters: the publish loop takes replan_lock to read every
    # 50 Hz frame. When replan() ran wholly inside that lock, a 300-500 ms
    # inference blocked the pose stream for 15-25 frames. SONIC cannot hold
    # a frame -- a gap that long is a collapse. Observed on hardware as
    # repeated "loop fell behind by 300-550ms; resyncing" with the robot
    # nearly going down.
    #
    # Only prepare() and commit() touch shared state (_buf / _read_pos);
    # infer() is pure compute over a snapshot, so it is safe to run unlocked.
    #   with lock: prep = replan_prepare(t)
    #   (no lock): pred, npf = replan_infer(prep)
    #   with lock: replan_commit(pred, npf)
    # -----------------------------------------------------------------
    def replan_prepare(self, target: tuple[float, float, float, float]) -> dict:
        """Snapshot context + build ONNX feeds. CALLER MUST HOLD THE LOCK."""
        if self._buf is None:
            raise RuntimeError("replan() called before reset()")
        roles = self._contract["inputs"]
        context = self._get_context()
        feeds: dict = {
            roles["context"]: context,
            roles["velocity"]: np.asarray([list(target)], dtype=np.float32),
        }
        if self.graph_kind == "template":
            seed = (
                int(self._fixed_seed) if self._fixed_seed is not None
                else random.randint(0, 999_999)
            )
            feeds[roles["mode"]] = np.asarray([self.mode_idx], dtype=np.int64)
            feeds[roles["random_seed"]] = np.asarray([seed], dtype=np.int64)
            _TAPE.ev("replan_prep", seed=seed, mode=int(self.mode_idx),
                     target=list(target))
        # Snapshot the serve position so commit can fast-forward the new
        # buffer by whatever played during inference (see replan_commit).
        self._prep_read_pos = float(self._read_pos)
        self._prep_cursor = int(self._cursor)
        for name, spec in (self._contract.get("extra_inputs") or {}).items():
            dtype = np.int64 if spec.get("dtype", "f32") == "i64" else np.float32
            arr = np.asarray(spec["value"], dtype=dtype)
            if "shape" in spec:
                arr = arr.reshape(spec["shape"])
            feeds[name] = arr
        return {"feeds": feeds, "context": context}

    def replan_infer(self, prep: dict) -> tuple[np.ndarray, int]:
        """Run the planner graph. MUST be called WITHOUT the lock.

        This is the 300-500 ms step. It reads only the snapshot in ``prep``
        and mutates no shared state, so the publisher keeps streaming while
        it runs.
        """
        outs = self._contract["outputs"]
        context = prep["context"]

        qpos_out, npf_out = self._sess.run(
            [outs["qpos"], outs["num_pred_frames"]], prep["feeds"]
        )
        npf = int(np.asarray(npf_out).reshape(-1)[0])
        npf = max(1, min(npf, int(qpos_out.shape[1])))
        pred = np.asarray(qpos_out[0, :npf], dtype=np.float32).copy()

        if not self._filter_baked:
            # FILTER_QPOS context blend, numpy port of NeuralPlannerCore's
            # (linspace 0.3..0.7 over the 4 context frames; root quat rows
            # [3:7] untouched). Only used for graphs exported WITHOUT the
            # in-graph blend (sidecar filter_qpos_baked=false).
            ctx = context[0]  # [4, 38] raw context (pre-canonicalize copy)
            num_ctx = ctx.shape[0]
            n = min(num_ctx, pred.shape[0])
            blend = np.linspace(0.3, 0.7, num_ctx, dtype=np.float32)[:n, None]
            pred[:n, :3] = ctx[:n, :3] * (1 - blend) + pred[:n, :3] * blend
            pred[:n, 7:] = ctx[:n, 7:] * (1 - blend) + pred[:n, 7:] * blend
        return pred, npf

    def replan_commit(self, pred: np.ndarray, npf: int) -> int:
        """Arm the seam blend and swap in the new buffer.

        CALLER MUST HOLD THE LOCK. Pure bookkeeping -- microseconds, not the
        hundreds of milliseconds that inference costs.
        """
        # Arm the 8-tick cross-fade before swapping the buffer (snapshot the
        # still-playing old buffer + read cursor). No-op unless resampling
        # is active.
        if self._resample_active and self._buf is not None:
            self._blend_prev_buf = self._buf
            self._blend_prev_pos = float(self._read_pos)
            self._blend_remaining = self.BLEND_FRAMES
        # REWIND FIX (run 20260719_214150): the new chunk continues from the
        # PREP-time context, but the publisher kept serving the old buffer
        # during the 0.3-0.6 s inference. Restarting the new buffer at 0
        # therefore rewound the served reference by the frames consumed
        # during inference (~18 = half a gait cycle at walk cadence); the
        # 8-tick seam blend then averaged antiphase leg poses into a
        # near-still reference. Fast-forward the new buffer by exactly the
        # frames consumed since prep so served content stays continuous.
        base = NUM_FRAMES_PER_TOKEN - PRED_OFFSETS  # == 0
        consumed = 0.0
        if getattr(self, "commit_fastforward", True):
            if self._resample_active:
                consumed = max(0.0, float(self._read_pos)
                               - getattr(self, "_prep_read_pos", self._read_pos))
            else:
                consumed = float(max(0, self._cursor
                                     - getattr(self, "_prep_cursor", self._cursor)))
            consumed = min(consumed, max(0.0, float(npf - 2)))
        self._buf = pred
        self._cursor = int(base + consumed)
        if self._resample_active:
            self._read_pos = float(base) + consumed
        if consumed > 0:
            _TAPE.ev("commit_ff", consumed=round(consumed, 2), npf=int(npf))
        _TAPE.chunk(pred, npf)
        return npf

    def replan(self, target: tuple[float, float, float, float]) -> int:
        """Single-threaded convenience wrapper (offline clip generation, A/B).

        The live publisher must NOT use this -- it would hold the lock across
        inference again. Use prepare/infer/commit with the lock released around
        infer(). Safe here because these callers have no concurrent reader.
        """
        prep = self.replan_prepare(target)
        pred, npf = self.replan_infer(prep)
        return self.replan_commit(pred, npf)

    def describe(self) -> str:
        return f"onnx/{self.graph_kind}" + (
            "" if self.mode_idx is None else f"(mode_idx={self.mode_idx})"
        )


class TorchPlannerBackend:
    """A/B backend wrapping the real NeuralPlannerCore, called exactly the
    way ``x2_kplanner.run()`` calls it. torch/motionbricks imported lazily
    here so the ONNX path stays torch-free."""

    def __init__(
        self,
        vqvae_ckpt: Path,
        pose_ckpt: Path,
        root_ckpt: Path,
        device: str,
        replan_threshold_frames: int,
        planner_mode: Optional[str],
    ) -> None:
        import torch  # noqa: PLC0415 -- torch backend only
        from motionbricks.motion_backbone.inference.load_x2_planner import (
            X2PlannerPaths,
            load_x2_planner,
        )

        self._torch = torch
        default_paths = X2PlannerPaths.default()
        paths = X2PlannerPaths(
            vqvae_ckpt=vqvae_ckpt,
            pose_ckpt=pose_ckpt,
            root_ckpt=root_ckpt,
            vqvae_version_dir=default_paths.vqvae_version_dir,
            pose_version_dir=default_paths.pose_version_dir,
            root_version_dir=default_paths.root_version_dir,
        )
        log.info("loading X2 kplanner stack on device=%s ...", device)
        self._core = load_x2_planner(
            paths, device=device,
            replan_threshold_frames=replan_threshold_frames,
        )
        log.info("kplanner stack loaded.")

        self.graph_kind = "velocity" if planner_mode is None else "template"
        self.mode_idx: Optional[int] = None
        if planner_mode is not None:
            if self._core._clip_library is None:
                raise RuntimeError(
                    f"--planner-mode={planner_mode} requested but no clip "
                    f"library is loaded (bake out/X2-clip.ckpt first)."
                )
            self.mode_idx = _PLANNER_MODE_NAMES.index(planner_mode)
            log.info("pose-template inference: mode=%s (idx=%d)",
                     planner_mode, self.mode_idx)
        self.REPLAN_THRESHOLD_FRAMES = int(replan_threshold_frames)

    @property
    def current_frame_idx(self) -> int:
        return int(self._core.current_frame_idx)

    @property
    def frames_remaining(self) -> int:
        return int(self._core.frames_remaining)

    def should_replan(self) -> bool:
        return bool(self._core.should_replan())

    def reset(self, init_qpos: np.ndarray) -> None:
        self._core.reset(
            self._torch.from_numpy(np.asarray(init_qpos, dtype=np.float32))
        )

    def get_next_frame(self) -> np.ndarray:
        return self._core.get_next_frame().detach().cpu().numpy()

    def get_next_frame_resampled(
        self, output_fps: Optional[float] = None
    ) -> np.ndarray:
        return (
            self._core.get_next_frame_resampled(output_fps).detach().cpu().numpy()
        )

    def peek_output_frame(self, output_offset_ticks: float) -> np.ndarray:
        return (
            self._core.peek_output_frame(output_offset_ticks).detach().cpu().numpy()
        )

    def peek_frame(self, idx: int) -> np.ndarray:
        buf = self._core.frames["mujoco_qpos"]
        i = max(0, min(int(idx), buf.shape[1] - 1))
        return buf[0, i].detach().cpu().numpy()

    def replan(self, target: tuple[float, float, float, float]) -> int:
        if self.mode_idx is None:
            _, _, npf = self._core.replan_with_velocity(list(target))
        else:
            _, _, npf = self._core.replan_with_pose_template(
                list(target), mode_idx=self.mode_idx
            )
        return int(npf)

    # NOTE: deliberately does NOT implement the prepare/infer/commit split.
    # The torch core fuses inference and buffer swap, so there is no safe way
    # to run part of it unlocked. The worker feature-detects the split API and
    # falls back to the locked replan() here. That costs a stall, which is
    # acceptable because this backend is the offline A/B path and never runs
    # on the robot -- and a stall is far better than a half-applied replan.

    def describe(self) -> str:
        return f"torch/{self.graph_kind}" + (
            "" if self.mode_idx is None else f"(mode_idx={self.mode_idx})"
        )


# ---------------------------------------------------------------------------
# IntentState (port of x2_kplanner.IntentState).
# ---------------------------------------------------------------------------


class IntentState:
    def __init__(self, initial: tuple[float, float, float, float]) -> None:
        self._target = initial
        self._lock = threading.Lock()
        self._version = 0
        self._last_set_t = time.monotonic()

    def set(self, target: tuple[float, float, float, float]) -> None:
        with self._lock:
            self._target = tuple(target)
            self._version += 1
            self._last_set_t = time.monotonic()

    def get(self) -> tuple[tuple[float, float, float, float], int]:
        with self._lock:
            return self._target, self._version

    def seconds_since_last_set(self) -> float:
        with self._lock:
            return time.monotonic() - self._last_set_t

    def force_idle_if_stale(
        self, max_age_s: float, idle_target: tuple[float, float, float, float]
    ) -> bool:
        with self._lock:
            age = time.monotonic() - self._last_set_t
            if age < max_age_s:
                return False
            if tuple(self._target) == tuple(idle_target):
                return False
            self._target = tuple(idle_target)
            self._version += 1
            return True


# ---------------------------------------------------------------------------
# Cold-start velocity ramp (port; default OFF, matching x2_kplanner
# 2026-07-16 default).
# ---------------------------------------------------------------------------


class ColdStartVelocityRamp:
    def __init__(self, tau_s: float = 0.0) -> None:
        self.tau_s = float(tau_s)
        self._smoothed = np.zeros(3, dtype=np.float64)
        self._last_was_idle = True

    @property
    def enabled(self) -> bool:
        return self.tau_s > 0.0

    def step(
        self, target: tuple[float, float, float, float], dt_s: float
    ) -> tuple[float, float, float, float]:
        yaw, vx, vz, hip = target
        if self._last_was_idle:
            self._smoothed.fill(0.0)
        target_vec = np.array([yaw, vx, vz], dtype=np.float64)
        if self.tau_s <= 0.0 or dt_s <= 0.0:
            self._smoothed = target_vec
        else:
            alpha = float(dt_s) / (self.tau_s + float(dt_s))
            self._smoothed += alpha * (target_vec - self._smoothed)
        self._last_was_idle = False
        return (float(self._smoothed[0]), float(self._smoothed[1]),
                float(self._smoothed[2]), float(hip))

    def reset_idle(self) -> None:
        self._smoothed.fill(0.0)
        self._last_was_idle = True


# ---------------------------------------------------------------------------
# Reference-step smoother (compact port of x2_kplanner._ReferenceStepSmoother;
# same defaults: 300 ms halfcos ramp, 0.05 rad trigger, lower-body joints).
# ---------------------------------------------------------------------------

_REF_SMOOTHER_JOINTS_PRESETS: dict[str, np.ndarray] = {
    "lower_body": np.arange(0, 15, dtype=np.int64),
    "legs_only":  np.arange(0, 12, dtype=np.int64),
    "all":        np.arange(0, 31, dtype=np.int64),
}
_REF_SMOOTHER_SHAPES: tuple[str, ...] = ("halfcos", "linear", "off")


class ReferenceStepSmoother:
    def __init__(
        self,
        ramp_duration_s: float = 0.300,
        trigger_rad: float = 0.05,
        shape: str = "halfcos",
        blend_indices: Optional[np.ndarray] = None,
    ) -> None:
        if shape not in _REF_SMOOTHER_SHAPES:
            raise ValueError(f"shape={shape!r} not in {_REF_SMOOTHER_SHAPES}")
        self.ramp_duration_s = float(ramp_duration_s)
        self.trigger_rad = float(trigger_rad)
        self.shape = shape
        self.blend_indices = (
            blend_indices if blend_indices is not None
            else _REF_SMOOTHER_JOINTS_PRESETS["lower_body"]
        )
        self._last_q: Optional[np.ndarray] = None
        self._ramp_active = False
        self._ramp_start_t = 0.0
        self._source_q: Optional[np.ndarray] = None

    @property
    def enabled(self) -> bool:
        return self.shape != "off" and self.ramp_duration_s > 0.0

    def _alpha(self, t_in_ramp: float) -> float:
        x = max(0.0, min(1.0, t_in_ramp / self.ramp_duration_s))
        if self.shape == "halfcos":
            return 0.5 * (1.0 - math.cos(math.pi * x))
        return x  # linear

    def update(
        self, target_q: np.ndarray, t_now: float, allow_arm: bool = True
    ) -> np.ndarray:
        """``allow_arm=False`` suppresses NEW ramp arming (an active ramp
        still completes). Used by dance playback: the clip's own fast leg
        motion legitimately exceeds the step trigger every few ticks, and
        continuous re-arming would lag the choreography by up to one ramp
        duration -- only the entry step (anchor -> clip frame 0) should
        ramp."""
        target_q = np.asarray(target_q)
        if not self.enabled or self._last_q is None:
            self._last_q = target_q.astype(target_q.dtype, copy=True)
            return self._last_q.copy()
        bi = self.blend_indices
        delta_max = float(np.max(np.abs(target_q[bi] - self._last_q[bi])))
        if allow_arm and (not self._ramp_active) and delta_max > self.trigger_rad:
            self._ramp_active = True
            self._ramp_start_t = float(t_now)
            self._source_q = self._last_q.astype(target_q.dtype, copy=True)
            log.info(
                "ref-smoother: armed (delta=%.3f rad, trigger=%.3f rad, "
                "T=%.0f ms, shape=%s)",
                delta_max, self.trigger_rad,
                self.ramp_duration_s * 1000.0, self.shape,
            )
        if self._ramp_active and self._source_q is not None:
            t_in_ramp = float(t_now) - self._ramp_start_t
            if t_in_ramp >= self.ramp_duration_s:
                self._ramp_active = False
                self._source_q = None
                out = target_q.astype(target_q.dtype, copy=True)
            else:
                alpha = self._alpha(t_in_ramp)
                out = target_q.astype(target_q.dtype, copy=True)
                out[bi] = (
                    (1.0 - alpha) * self._source_q[bi] + alpha * target_q[bi]
                ).astype(target_q.dtype, copy=False)
        else:
            out = target_q.astype(target_q.dtype, copy=True)
        self._last_q = out
        return out.copy()


# ---------------------------------------------------------------------------
# Wire payload builder. EXACT field set + insertion order of
# state_machine.build_pose_payload (the consumer decodes by order):
#   joint_pos_mj, root_quat_xyzw, motion_token, left_hand_joints,
#   right_hand_joints, frame_index, root_xy_world, root_z_world,
#   joint_pos_mj_future, root_quat_xyzw_future, joint_vel_mj_future,
#   frame_index_future, future_dt_s
# ---------------------------------------------------------------------------


def build_pose_payload_np(
    jpos: np.ndarray,
    quat_xyzw: np.ndarray,
    root_xy: np.ndarray,
    root_z: float,
    frame_index: int,
    future_jpos: list[np.ndarray],
    future_quat: list[np.ndarray],
    motion_token_dim: int = SONIC_MOTION_TOKEN_DIM,
    hand_dof: int = DEFAULT_HAND_DOF,
    future_dt_s: float = FUTURE_DT_S,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {
        "joint_pos_mj": np.asarray(jpos, dtype=np.float32),
        "root_quat_xyzw": np.asarray(quat_xyzw, dtype=np.float32),
        "motion_token": np.zeros(motion_token_dim, dtype=np.float32),
        "left_hand_joints": np.zeros(hand_dof, dtype=np.float32),
        "right_hand_joints": np.zeros(hand_dof, dtype=np.float32),
        "frame_index": np.array([frame_index], dtype=np.int64),
        "root_xy_world": np.asarray(root_xy, dtype=np.float32),
        "root_z_world": np.array([float(root_z)], dtype=np.float32),
    }
    if future_jpos:
        n_future = len(future_jpos)
        jpos_future = np.stack(
            [np.asarray(f, dtype=np.float32) for f in future_jpos]
        )
        rot_future = np.stack(
            [np.asarray(f, dtype=np.float32) for f in future_quat]
        )
        prev_jpos = np.asarray(jpos, dtype=np.float32)[None, :]
        all_jpos = np.concatenate([prev_jpos, jpos_future], axis=0)
        jvel_future = (
            (all_jpos[1:] - all_jpos[:-1]) / max(float(future_dt_s), 1e-6)
        ).astype(np.float32)
        step_ticks = int(round(future_dt_s * OUTPUT_FPS))
        frame_idx_future = np.array(
            [frame_index + (k + 1) * step_ticks for k in range(n_future)],
            dtype=np.int64,
        )
        payload["joint_pos_mj_future"] = jpos_future
        payload["root_quat_xyzw_future"] = rot_future
        payload["joint_vel_mj_future"] = jvel_future
        payload["frame_index_future"] = frame_idx_future
        payload["future_dt_s"] = np.array([float(future_dt_s)], dtype=np.float32)
    return payload


# ---------------------------------------------------------------------------
# VR arm/hand target ingest (hop-in manipulation, 2026-07-30).
#
# The tethered laptop stack merges operator arm IK into the pose wire via the
# dataset recorder. The onboard (ritual) stack has no recorder, so this class
# is the robot-side replacement: a SUB (bind tcp://*:ARM_TARGET_PORT) that the
# laptop manager PUB-connects to (--arm-connect), caching the operator's arm
# and hand targets, plus an overlay applied at the PosePublisher choke point:
#   * joint_pos_mj[15:22]/[22:29]      <- left/right arm q (7 DOF each)
#   * joint_pos_mj_future[:, slices]   <- same pose pinned across the window
#     (mirrors the recorder: arm_targets is a "current command", there is no
#     arm trajectory to look ahead with)
#   * left/right_hand_joints           <- hand q (the deploy hand bridge reads
#     fingers straight off these pose-wire fields)
# Semantics mirror the recorder exactly: the cache holds the last commanded
# pose until a passthrough message clears it (link loss => arms HOLD, never
# snap); a dance clip (dance_active) suspends the overlay so clips own the
# whole body. Fail-open: no manager connected -> planner behaves as before.
# ---------------------------------------------------------------------------

_LEFT_ARM_MJ = slice(15, 22)
_RIGHT_ARM_MJ = slice(22, 29)


class ArmTargetIngest:
    def __init__(self, port: int) -> None:
        import zmq
        self._lock = threading.Lock()
        self._left: Optional[np.ndarray] = None
        self._right: Optional[np.ndarray] = None
        self._vel_left: Optional[np.ndarray] = None
        self._vel_right: Optional[np.ndarray] = None
        self._arm_msg_t: float = 0.0
        self._left_hand: Optional[np.ndarray] = None
        self._right_hand: Optional[np.ndarray] = None
        self._last_msg_t = 0.0
        self._stop = threading.Event()
        self._port = int(port)
        self._thr = threading.Thread(
            target=self._loop, name="arm-ingest", daemon=True)
        self._thr.start()

    def _loop(self) -> None:
        import zmq
        try:
            import msgpack
        except ImportError:
            log.error("arm-ingest: msgpack unavailable; VR arm targets OFF")
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        for t in ("arm_targets", "hand_finger_cmd"):
            sock.setsockopt_string(zmq.SUBSCRIBE, t)
        sock.setsockopt(zmq.RCVTIMEO, 200)
        sock.bind(f"tcp://*:{self._port}")
        log.info("arm-ingest: SUB bind tcp://*:%d (arm_targets, "
                 "hand_finger_cmd); laptop manager PUB-connects here "
                 "(--arm-connect)", self._port)
        while not self._stop.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                continue
            if len(parts) < 2:
                continue
            try:
                topic = parts[0].decode("ascii", errors="replace")
                data = msgpack.unpackb(parts[1], raw=False)
                if topic == "arm_targets":
                    with self._lock:
                        if data.get("passthrough_arm_targets"):
                            self._left = None
                            self._right = None
                            self._vel_left = None
                            self._vel_right = None
                        else:
                            lq = np.asarray(
                                data.get("left_q_rad", ()), dtype=np.float32)
                            rq = np.asarray(
                                data.get("right_q_rad", ()), dtype=np.float32)
                            if lq.shape == (7,) and rq.shape == (7,):
                                now_m = time.monotonic()
                                # Per-joint target velocity from the last
                                # two arrivals — used to extrapolate through
                                # stream gaps (first-order hold).
                                prev = self._left
                                prev_t = self._arm_msg_t
                                if prev is not None and prev_t and                                         0.005 < now_m - prev_t < 0.5:
                                    dt = now_m - prev_t
                                    self._vel_left = np.clip(
                                        (lq - prev) / dt, -3.0, 3.0)
                                    self._vel_right = np.clip(
                                        (rq - self._right) / dt, -3.0, 3.0)
                                else:
                                    self._vel_left = None
                                    self._vel_right = None
                                self._left = lq
                                self._right = rq
                                self._arm_msg_t = now_m
                        self._last_msg_t = time.monotonic()
                elif topic == "hand_finger_cmd":
                    lh = np.asarray(
                        data.get("left_hand_q", ()), dtype=np.float32)
                    rh = np.asarray(
                        data.get("right_hand_q", ()), dtype=np.float32)
                    if lh.shape == (10,) and rh.shape == (10,):
                        with self._lock:
                            self._left_hand = lh
                            self._right_hand = rh
                            self._last_msg_t = time.monotonic()
            except Exception as exc:  # malformed frame must never kill pose
                log.warning("arm-ingest: bad frame dropped (%r)", exc)
        sock.close(linger=0)

    # Max per-published-frame target step (rad @ 50 Hz => 3.0 rad/s arm,
    # 7.5 rad/s fingers). Smooths the sample-and-hold chop from wifi /
    # headset micro-stalls (measured: 29% frozen frames + 0.23 rad jumps
    # during continuous motion) and doubles as blend-in on first engage
    # (slew seeds from the planner's current arm pose, so arms ramp from
    # stand to the operator pose instead of snapping).
    ARM_SLEW_RAD_PER_FRAME = 0.06
    HAND_SLEW_RAD_PER_FRAME = 0.15

    @staticmethod
    def _slew(cur: np.ndarray, target: np.ndarray, step: float) -> np.ndarray:
        d = target - cur
        return cur + np.clip(d, -step, step)

    def overlay(self, payload: dict) -> bool:
        """Apply cached targets to a pose payload. Returns True if the
        arm slices were overlaid (used for state-transition logging)."""
        with self._lock:
            left, right = self._left, self._right
            lh, rh = self._left_hand, self._right_hand
            releasing = getattr(self, "_releasing", False)
            vel_l, vel_r = self._vel_left, self._vel_right
            arm_t = self._arm_msg_t
        if left is not None and vel_l is not None and vel_r is not None:
            # First-order hold: extrapolate along the last observed target
            # velocity during stream gaps (wifi / headset micro-stalls), so
            # motion continues instead of freezing. Capped at 250 ms — a
            # genuinely dead stream degrades to a plain hold, and the
            # ownership silence-release clears it at 2 s.
            gap = time.monotonic() - arm_t
            if 0.0 < gap:
                h = min(gap, 0.25)
                left = left + vel_l * h
                right = right + vel_r * h
        if releasing and left is None:
            # Ownership released: slew back toward the planner's own arm
            # pose; deactivate once converged (or state was never seeded).
            jpos = payload.get("joint_pos_mj")
            if (getattr(self, "_slew_left", None) is None or jpos is None
                    or jpos.shape != (31,)):
                self._releasing = False
                self._slew_left = None
                self._slew_right = None
                self._slew_lh = None
                self._slew_rh = None
                return False
            tgt_l = np.asarray(jpos[_LEFT_ARM_MJ], dtype=np.float32)
            tgt_r = np.asarray(jpos[_RIGHT_ARM_MJ], dtype=np.float32)
            self._slew_left = self._slew(
                self._slew_left, tgt_l, self.ARM_SLEW_RAD_PER_FRAME)
            self._slew_right = self._slew(
                self._slew_right, tgt_r, self.ARM_SLEW_RAD_PER_FRAME)
            done = (np.abs(self._slew_left - tgt_l).max() < 0.02
                    and np.abs(self._slew_right - tgt_r).max() < 0.02)
            jpos = jpos.copy()
            jpos[_LEFT_ARM_MJ] = self._slew_left
            jpos[_RIGHT_ARM_MJ] = self._slew_right
            payload["joint_pos_mj"] = jpos
            fut = payload.get("joint_pos_mj_future")
            if fut is not None and fut.ndim == 2 and fut.shape[1] == 31:
                fut = fut.copy()
                fut[:, _LEFT_ARM_MJ] = self._slew_left
                fut[:, _RIGHT_ARM_MJ] = self._slew_right
                payload["joint_pos_mj_future"] = fut
            if done:
                self._releasing = False
                self._slew_left = None
                self._slew_right = None
                self._slew_lh = None
                self._slew_rh = None
            return not done
        if left is not None and right is not None:
            # COPY-ON-WRITE, never mutate in place: build_pose_payload_np's
            # np.asarray typically aliases the planner's own anchor/output
            # arrays — in-place writes would pollute planner state and the
            # arm pose would survive a passthrough clear.
            jpos = payload.get("joint_pos_mj")
            if jpos is not None and jpos.shape == (31,):
                # Slew toward the operator target from the last frame we
                # PUBLISHED (seeded from the planner's own arms on first
                # engage), never jumping more than ARM_SLEW_RAD_PER_FRAME.
                if getattr(self, "_slew_left", None) is None:
                    self._slew_left = np.asarray(
                        jpos[_LEFT_ARM_MJ], dtype=np.float32).copy()
                    self._slew_right = np.asarray(
                        jpos[_RIGHT_ARM_MJ], dtype=np.float32).copy()
                self._slew_left = self._slew(
                    self._slew_left, left, self.ARM_SLEW_RAD_PER_FRAME)
                self._slew_right = self._slew(
                    self._slew_right, right, self.ARM_SLEW_RAD_PER_FRAME)
                left = self._slew_left
                right = self._slew_right
                jpos = jpos.copy()
                jpos[_LEFT_ARM_MJ] = left
                jpos[_RIGHT_ARM_MJ] = right
                payload["joint_pos_mj"] = jpos
            fut = payload.get("joint_pos_mj_future")
            if fut is not None and fut.ndim == 2 and fut.shape[1] == 31:
                fut = fut.copy()
                fut[:, _LEFT_ARM_MJ] = left
                fut[:, _RIGHT_ARM_MJ] = right
                payload["joint_pos_mj_future"] = fut
            # Future joint velocities were finite-differenced BEFORE the
            # overlay pinned the arm slices; a pinned pose has zero arm
            # velocity, so zero those slices for consistency.
            jvel = payload.get("joint_vel_mj_future")
            if jvel is not None and jvel.ndim == 2 and jvel.shape[1] == 31:
                jvel = jvel.copy()
                jvel[:, _LEFT_ARM_MJ] = 0.0
                jvel[:, _RIGHT_ARM_MJ] = 0.0
                payload["joint_vel_mj_future"] = jvel
        if lh is not None and payload.get("left_hand_joints") is not None:
            if getattr(self, "_slew_lh", None) is None:
                self._slew_lh = np.asarray(
                    payload["left_hand_joints"], dtype=np.float32).copy()
            self._slew_lh = self._slew(
                self._slew_lh, lh, self.HAND_SLEW_RAD_PER_FRAME)
            payload["left_hand_joints"] = self._slew_lh
        if rh is not None and payload.get("right_hand_joints") is not None:
            if getattr(self, "_slew_rh", None) is None:
                self._slew_rh = np.asarray(
                    payload["right_hand_joints"], dtype=np.float32).copy()
            self._slew_rh = self._slew(
                self._slew_rh, rh, self.HAND_SLEW_RAD_PER_FRAME)
            payload["right_hand_joints"] = self._slew_rh
        return left is not None and right is not None

    def clear(self) -> None:
        """Drop cached arm + hand targets (planner arms take over).

        Called when VR command ownership releases — explicit disengage or
        the crash-silence timeout — so a dead/disengaged manager can never
        leave the robot stuck holding a stale manipulation pose (incident
        2026-07-31: headset stream stalled mid-ARM_MAN, operator killed
        the manager, arms held the frozen pose with no reset path).
        """
        with self._lock:
            self._left = None
            self._right = None
            self._left_hand = None
            self._right_hand = None
            self._vel_left = None
            self._vel_right = None
            # Blend-out: if we were overlaying, keep slewing toward the
            # PLANNER's arms until converged instead of snapping back in
            # one frame (release can happen with arms fully extended).
            self._releasing = getattr(self, "_slew_left", None) is not None

    def stop(self) -> None:
        self._stop.set()


class PosePublisher:
    """PUB bind + v4 packed encoder (port of x2_kplanner.PosePublisher,
    defaulting to bind-on-all-interfaces for the PC2 deployment)."""

    def __init__(self, host: str, port: int, topic: str = "pose") -> None:
        import zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{port}")
        self._topic = topic
        # Optional VR arm/hand overlay (hop-in manipulation). Wired by main
        # when --arm-port > 0; dance_active suspends it so clips own the
        # whole body (arms return to the held operator pose afterwards).
        self.arm_ingest: Optional["ArmTargetIngest"] = None
        self.dance_active: Optional[threading.Event] = None
        time.sleep(0.1)

    def publish(self, payload: dict[str, np.ndarray]) -> None:
        if self.arm_ingest is not None and not (
            self.dance_active is not None and self.dance_active.is_set()
        ):
            applied = self.arm_ingest.overlay(payload)
            if applied != getattr(self, "_overlay_state", None):
                log.info("arm overlay %s",
                         "ACTIVE (operator arm/hand targets on the wire)"
                         if applied else "cleared (planner arms restored)")
                self._overlay_state = applied
        self._sock.send(pack_pose_message(payload, topic=self._topic, version=4))
        self._tape_seq = getattr(self, "_tape_seq", -1) + 1
        _TAPE.ev("tick", seq=self._tape_seq)

    def close(self) -> None:
        self._sock.close(linger=0)


# ---------------------------------------------------------------------------
# Quaternion helpers (numpy; conventions match wire.py / blending.py).
# ---------------------------------------------------------------------------


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def _yaw_of_quat_xyzw(q: np.ndarray) -> float:
    return yaw_from_quat_wxyz(np.array([q[3], q[0], q[1], q[2]], dtype=np.float64))


def _wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def _quat_wxyz_from_yaw(yaw_rad: float) -> np.ndarray:
    """``R_z(yaw)`` packed as (qw, qx, qy, qz).

    Yaw-only by construction: pitch/roll are dropped on purpose, mirroring
    ``x2_kplanner._yaw_only_wxyz_from_pelvis`` -- a transient leg lean (fall
    recovery, slip) must not bleed into the published reference and pull SONIC
    outside its upright-reference training distribution.
    """
    half = 0.5 * float(yaw_rad)
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float32)


# Closed-loop idle yaw resync (port of x2_kplanner.py:3094).
# Max age of a measured-yaw sample we will trust; matches x2_kplanner's
# ``pose_feedback_max_age_s`` default. Stale -> hold last good, never revert to
# identity (identity == world +X is a KNOWN-WRONG heading the deploy actively
# twists the body toward).
POSE_FEEDBACK_MAX_AGE_S: float = 0.5
# Rate cap on how fast the reference yaw may be re-anchored. Defence in depth:
# resync is an ASSIGNMENT (no feedback term, so it cannot wind up), but a cap
# means even a wrong sign or frame error can only leak slowly instead of
# spinning the robot. See the 2026-07-18 runaway-spin incident.
MAX_YAW_RESYNC_RAD_S: float = 1.5

# PLAYING -> IDLE stop blend length. 16 ticks @ 50 Hz = 320 ms -- long enough to
# take a 1.2 rad snap down to ~0.075 rad/tick, short enough that the robot still
# settles promptly when the operator releases.
STOP_BLEND_FRAMES: int = 16


# ---------------------------------------------------------------------------
# Dance playback
# ---------------------------------------------------------------------------


class DancePlayback:
    """Streams a loaded x2m2 clip through the 50 Hz publisher clock.

    x2m2 bakes carry (dof [T,31], quat_xyzw [T,4], fps); no root
    translation, so root xy/z hold at the values streamed when the dance
    started (same convention as the fallback ladder's idle-clip replay).
    The clip's own fps may differ from 50 (some bakes are 120 fps); a
    float phase accumulator advances ``fps / OUTPUT_FPS`` clip frames per
    wire tick so playback stays real-time with nearest-frame sampling.

    Yaw rebase: all root quats are pre-multiplied by
    ``R_z(current_yaw - clip_frame0_yaw)`` at start (delta form of the
    fallback ladder's ``build_idle_frame_msg`` rebase, which assumes
    yaw-0-aligned clips; dance bakes may start at arbitrary yaw) so the
    dance's heading track starts exactly at the robot's current streamed
    heading and evolves with the clip's authored yaw motion.
    """

    def __init__(
        self,
        dof: np.ndarray,
        quat_xyzw: np.ndarray,
        fps: float,
        current_yaw_rad: float,
        name: str,
    ) -> None:
        self.name = name
        self._dof = np.ascontiguousarray(dof, dtype=np.float32)
        clip_yaw0 = _yaw_of_quat_xyzw(np.asarray(quat_xyzw[0], dtype=np.float64))
        delta = float(current_yaw_rad) - float(clip_yaw0)
        self._quat = rebase_quats_xyzw_by_yaw(
            np.ascontiguousarray(quat_xyzw, dtype=np.float32), delta
        )
        self._fps = float(fps) if fps and fps > 0 else OUTPUT_FPS
        self._phase = 0.0
        self._n = int(dof.shape[0])
        log.info(
            "dance %s: %d frames @ %.1f fps (%.1fs), yaw rebase %+.1f deg",
            name, self._n, self._fps, self._n / self._fps, math.degrees(delta),
        )

    @property
    def finished(self) -> bool:
        return self._phase >= self._n - 1

    def _frame_at(self, clip_idx: float) -> tuple[np.ndarray, np.ndarray]:
        i = max(0, min(int(round(clip_idx)), self._n - 1))
        return self._dof[i], self._quat[i]

    def tick(self) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Emit (jpos, quat, future_jpos[9], future_quat[9]) and advance one
        50 Hz tick. Future slots sample the clip at +0.1 s spacing, clamped
        at the final frame (clip ends -> hold last pose in the horizon)."""
        jpos, quat = self._frame_at(self._phase)
        step_clip = self._fps * FUTURE_DT_S
        fut_j: list[np.ndarray] = []
        fut_q: list[np.ndarray] = []
        for k in range(NUM_FUTURE):
            fj, fq = self._frame_at(self._phase + (k + 1) * step_clip)
            fut_j.append(fj)
            fut_q.append(fq)
        self._phase += self._fps / OUTPUT_FPS
        return jpos, quat, fut_j, fut_q


def _resolve_dance_x2m2(
    dances_dir: Path, pkl: Optional[str], motion_key: Optional[str]
) -> Optional[Path]:
    """<dances-dir>/<motion_key>.x2m2, falling back to the pkl stem."""
    candidates = []
    if motion_key:
        candidates.append(dances_dir / f"{motion_key}.x2m2")
    if pkl:
        candidates.append(dances_dir / f"{Path(pkl).stem}.x2m2")
    for cand in candidates:
        if cand.is_file():
            return cand
    log.error(
        "motion_clip_cmd play: no x2m2 bake found (tried %s)",
        [str(c) for c in candidates],
    )
    return None


# ---------------------------------------------------------------------------
# Command source threads
# ---------------------------------------------------------------------------


def _zmq_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
    bind: bool = False,
) -> None:
    """SUB planner_cmd (port of x2_kplanner._zmq_command_thread minus the
    waist fields the pad bridge never sends -- hip_height/direct_velocity
    passthroughs kept for x2_pkl_command_source compatibility).

    ``bind=True`` flips the SUB to bind so MULTIPLE command sources
    (pad bridge + quest3 manager, each PUB-connect) can coexist —
    two PUBs cannot share one bound port, but one bound SUB accepts
    any number of connected PUBs."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    if bind:
        sock.bind(f"tcp://*:{port}")
        log.info("planner_cmd source: SUB bind %r on tcp://*:%d", topic, port)
    else:
        sock.connect(f"tcp://{host}:{port}")
        log.info("planner_cmd source: SUB %r on tcp://%s:%d", topic, host, port)
    try:
        while not stop_event.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                # No message this cycle (200 ms tick). VR-silence release
                # must NOT depend on pad traffic: a killed/disengaged
                # manager with an idle pad would otherwise hold ownership
                # (and the arm overlay) forever.
                if (_cmd_owner["src"] == "vr"
                        and time.monotonic() - _cmd_owner["vr_ts"]
                        > _VR_OWNER_TIMEOUT_S):
                    log.warning(
                        "cmd owner: VR silent %.1fs > %.1fs — releasing to "
                        "idle (arms restored); pad may re-acquire",
                        time.monotonic() - _cmd_owner["vr_ts"],
                        _VR_OWNER_TIMEOUT_S)
                    _cmd_owner["src"] = None
                    if _ARM_INGEST_REF[0] is not None:
                        _ARM_INGEST_REF[0].clear()
                    cmd_queue.put(LocomotionCommand(
                        intent="idle", magnitude="default"))
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
                intent = str(payload["intent"])
                if intent == "shutdown":
                    log.info("planner_cmd source: shutdown received")
                    stop_event.set()
                    continue
                # ---- source ownership (mutual exclusion) --------------------
                # Multiple PUBs may be connected (pad bridge + VR manager),
                # but exactly ONE source owns the robot at a time. VR
                # SUPERSEDES pad: any VR message takes ownership and pad
                # messages are dropped until VR explicitly releases (idle on
                # disengage) or goes silent past the crash timeout (the
                # manager keepalives every 0.5 s while engaged, so a quiet
                # held stick cannot false-expire). On timeout release the
                # planner idles; the pad re-acquires with its next message.
                src = str(payload.get("source", "pad"))
                now_own = time.monotonic()
                if src == "vr":
                    _cmd_owner["vr_ts"] = now_own
                    if payload.get("vr_release"):
                        # Explicit disengage from the manager: the ONLY
                        # VR message that hands control back. A plain VR
                        # idle is a stand-still command and keeps
                        # ownership (sticks centered != disengage).
                        if _cmd_owner["src"] == "vr":
                            log.warning("cmd owner: VR released — "
                                        "pad may re-acquire")
                            _cmd_owner["src"] = None
                            if _ARM_INGEST_REF[0] is not None:
                                _ARM_INGEST_REF[0].clear()
                        elif _cmd_owner["src"] == "pad":
                            continue  # stray release while pad drives
                    else:
                        if _cmd_owner["src"] != "vr":
                            log.warning("cmd owner: VR ENGAGED — pad input ignored")
                            _cmd_owner["src"] = "vr"
                else:  # pad (or untagged legacy = pad)
                    if _cmd_owner["src"] == "vr":
                        if now_own - _cmd_owner["vr_ts"] > _VR_OWNER_TIMEOUT_S:
                            log.warning(
                                "cmd owner: VR silent %.1fs > %.1fs — releasing "
                                "to idle; pad re-acquires",
                                now_own - _cmd_owner["vr_ts"], _VR_OWNER_TIMEOUT_S)
                            _cmd_owner["src"] = None
                            if _ARM_INGEST_REF[0] is not None:
                                _ARM_INGEST_REF[0].clear()
                            cmd_queue.put(LocomotionCommand(
                                intent="idle", magnitude="default"))
                            continue  # this pad msg is dropped; next one owns
                        continue  # VR owns: drop pad message
                    if _cmd_owner["src"] is None:
                        _cmd_owner["src"] = "pad"
                magnitude = str(payload.get("magnitude", "default"))
                stick_fwd = float(payload.get("stick_fwd", 0.0))
                stick_side = float(payload.get("stick_side", 0.0))
                stick_yaw = float(payload.get("stick_yaw", 0.0))
                # Forward-obstacle clamp. Yaw is deliberately untouched so the
                # operator can turn away instead of being stuck facing a wall.
                if _guard_blocked():
                    if not _guard_latched["on"]:
                        log.warning("OBSTACLE %.2fm -> LATCHED; release the "
                                    "deadman to reset", _guard_state["dist"])
                    _guard_latched["on"] = True
                elif (stick_fwd == 0.0 and stick_side == 0.0
                      and stick_yaw == 0.0 and _guard_latched["on"]):
                    # deadman released (bridge sends an all-zero frame) -> reset
                    log.info("guard latch reset")
                    _guard_latched["on"] = False
                if _guard_latched["on"]:
                    stick_fwd = 0.0
                    stick_side = 0.0
                sd = payload.get("speed_delta")
                if sd:
                    _adjust_speed_setpoint(float(sd))
                hip_height_raw = payload.get("hip_height_m", None)
                hip_height_m: Optional[float] = (
                    float(hip_height_raw) if hip_height_raw is not None else None
                )
                target_velocity = payload.get("target_velocity")
                direct_velocity: Optional[tuple[float, float, float, float]] = None
                if target_velocity is not None:
                    if (
                        not isinstance(target_velocity, (list, tuple))
                        or len(target_velocity) != 4
                    ):
                        log.warning(
                            "planner_cmd: target_velocity must be a 4-list; "
                            "got %r (ignoring)", target_velocity,
                        )
                    else:
                        direct_velocity = tuple(float(v) for v in target_velocity)
                        # VR/Quest sends velocity directly, bypassing sticks.
                        if _guard_latched["on"] and direct_velocity[0] > 0.0:
                            log.warning("OBSTACLE %.2fm -> direct vx held",
                                        _guard_state["dist"])
                            direct_velocity = (0.0, 0.0,
                                               direct_velocity[2],
                                               direct_velocity[3])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("planner_cmd: bad payload %r: %s", parts[1], exc)
                continue
            _TAPE.ev("intent_recv", intent=intent, magnitude=magnitude,
                     stick_fwd=stick_fwd, stick_side=stick_side,
                     stick_yaw=stick_yaw, direct_velocity=direct_velocity)
            cmd_queue.put(
                LocomotionCommand(
                    intent=intent,
                    magnitude=magnitude,
                    source="zmq",
                    stick_fwd=stick_fwd,
                    stick_side=stick_side,
                    stick_yaw=stick_yaw,
                    direct_velocity=direct_velocity,
                    hip_height_m=hip_height_m,
                )
            )
    finally:
        sock.close(linger=0)


class MeasuredYaw:
    """Thread-safe latch for the robot's live IMU yaw (from x2_debug).

    Mirrors the watchdog's yaw-rebase source: the C++ deploy PUBs
    ``x2_debug`` with the pelvis ``base_quat``; we decode it to a yaw so
    the planner can rebase its published root quats to the robot's
    actual heading (else SONIC twists the body to world +X -- the
    orientation snap). ``value`` is None until the first frame lands.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._yaw: Optional[float] = None
        self._ts: float = 0.0

    def set(self, yaw: float) -> None:
        with self._lock:
            self._yaw = float(yaw)
            self._ts = time.monotonic()

    def get(self, max_age_s: float = 1.0) -> Optional[float]:
        with self._lock:
            if self._yaw is None:
                return None
            if time.monotonic() - self._ts > max_age_s:
                return None
            return self._yaw


def _x2_debug_thread(
    measured: "MeasuredYaw",
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
) -> None:
    """SUB the deploy's x2_debug PUB; latch the measured pelvis yaw.

    Best-effort: transient decode failures are ignored (a misshapen
    frame must never wedge this thread). Absent on the laptop/sim
    (no C++ deploy) -- the latch simply stays None and rebase is a
    no-op there."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    log.info("x2_debug source: SUB %r on tcp://%s:%d (measured-yaw rebase)",
             topic, host, port)
    try:
        while not stop_event.is_set():
            try:
                raw = sock.recv()
            except zmq.error.Again:
                continue
            base_quat_wxyz = decode_x2_debug_base_quat(raw, topic)
            if base_quat_wxyz is None:
                continue
            try:
                measured.set(yaw_from_quat_wxyz(base_quat_wxyz))
            except Exception:  # noqa: BLE001 -- never wedge on a bad frame
                continue
    finally:
        sock.close(linger=0)


def _motion_clip_cmd_thread(
    dance_queue: "queue.Queue[tuple]",
    port: int,
    topic: str,
    dances_dir: Path,
    stop_event: threading.Event,
) -> None:
    """SUB bind motion_clip_cmd; loads the x2m2 in-thread and enqueues the
    arrays so the publish loop never blocks on disk I/O."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.bind(f"tcp://*:{port}")
    log.info("motion_clip_cmd source: SUB bind %r on tcp://*:%d (dances=%s)",
             topic, port, dances_dir)
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
                action = str(payload.get("action", ""))
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                log.warning("motion_clip_cmd: bad payload %r: %s", parts[1], exc)
                continue
            if action == "stop":
                log.info("motion_clip_cmd: STOP")
                dance_queue.put(("stop",))
                continue
            if action != "play":
                log.warning("motion_clip_cmd: unknown action %r", action)
                continue
            kind = str(payload.get("kind", "locomotion"))
            if kind != "locomotion":
                log.warning(
                    "motion_clip_cmd: kind=%r unsupported here (locomotion "
                    "only); ignoring", kind,
                )
                continue
            motion_key = payload.get("motion_key")
            x2m2_path = _resolve_dance_x2m2(
                dances_dir, payload.get("pkl"), motion_key
            )
            if x2m2_path is None:
                continue
            try:
                dof, quat, fps = load_x2m2(x2m2_path)
            except (ValueError, OSError) as exc:
                log.error("motion_clip_cmd: cannot load %s: %s", x2m2_path, exc)
                continue
            log.info("motion_clip_cmd: PLAY %s (%d frames @ %.1f fps)",
                     x2m2_path.name, dof.shape[0], fps)
            dance_queue.put(("play", x2m2_path.stem, dof, quat, fps))
    finally:
        sock.close(linger=0)


# ---------------------------------------------------------------------------
# Worker thread: replan when the ring buffer dips below threshold
# (port of x2_kplanner._planner_worker; pose reseed omitted -- open loop,
# same as x2_kplanner's default config; dance_active pauses replans).
# ---------------------------------------------------------------------------


def _planner_worker(
    backend,
    intent: IntentState,
    replan_lock: threading.Lock,
    stop_event: threading.Event,
    replan_event: threading.Event,
    dance_active: threading.Event,
    cold_start_ramp_tau_s: float = 0.0,
) -> None:
    log.info("planner worker thread started (backend=%s)", backend.describe())
    cold_start_ramp = ColdStartVelocityRamp(tau_s=cold_start_ramp_tau_s)
    if cold_start_ramp.enabled:
        log.info("planner worker: cold-start velocity ramp ENABLED (tau=%.3fs)",
                 cold_start_ramp.tau_s)
    last_replan_mono: Optional[float] = None

    while not stop_event.is_set():
        if not replan_event.wait(timeout=0.05):
            with replan_lock:
                needs_replan = backend.should_replan()
        else:
            replan_event.clear()
            # Stale-event guard (tape 20260719_205421: every mid-walk replan
            # double-fired). The 50 Hz loop re-arms this event on every tick
            # the buffer is below threshold -- including the whole inference
            # window of the replan already refilling it -- so a fresh commit
            # was immediately followed by a redundant replan and a second
            # overlapping seam blend. Re-check the actual buffer state after
            # clearing; explicit forces (IDLE->PLAYING reseed) still pass via
            # the _force_replan flag because reset() leaves the ring full and
            # should_replan() alone would skip them.
            with replan_lock:
                forced = getattr(backend, "_force_replan", False)
                if forced:
                    backend._force_replan = False
                needs_replan = forced or backend.should_replan()
        if not needs_replan or stop_event.is_set():
            continue
        if dance_active.is_set():
            # Dance playback preempts the planner; ring is paused. Reset
            # the ramp so post-dance locomotion ramps from zero.
            cold_start_ramp.reset_idle()
            last_replan_mono = None
            continue
        target, ver = intent.get()
        if tuple(target) == _IDLE_INTENT:
            # Idle gate: publisher holds the frozen anchor; don't replan.
            cold_start_ramp.reset_idle()
            last_replan_mono = None
            continue

        now_mono = time.monotonic()
        if last_replan_mono is None:
            dt_s = 1.0 / OUTPUT_FPS
        else:
            dt_s = max(1e-3, now_mono - last_replan_mono)
        target = cold_start_ramp.step(tuple(target), dt_s)
        last_replan_mono = now_mono

        log.info(
            "Replanning with mode: %s, target_vel(fwd): %+.3f, lateral: %+.3f, "
            "yaw_rate: %+.3f, hip_h: %.3f  [v=%d]",
            ("velocity-only" if backend.mode_idx is None
             else f"template_idx={backend.mode_idx}"),
            float(target[2]), float(target[1]), float(target[0]),
            float(target[3]), ver,
        )
        t0 = time.monotonic()
        try:
            # Lock held only for the two cheap phases. Inference -- the
            # 300-500 ms step -- runs UNLOCKED so the 50 Hz publisher keeps
            # streaming. Holding the lock across inference starved SONIC for
            # 15-25 frames and nearly dropped the robot.
            #
            # Feature-detected: the ONNX backend splits, the torch A/B backend
            # cannot (it fuses inference with the buffer swap) and falls back
            # to the fully-locked path.
            if hasattr(backend, "replan_prepare"):
                # Chunk LIVENESS GATE (run 20260719_214150: the model emitted
                # a standing chunk mid-walk -> 2.0 s dead reference -> violent
                # catch-up). A committed standing chunk contaminates the next
                # replan's context, making the collapse self-sustaining. So:
                # while a walk is commanded, reject statistically-still chunks
                # BEFORE commit (old, still-walking buffer keeps streaming),
                # re-roll with a fresh seed; after N failures commit anyway
                # and scream -- a sliding reference beats a starved one.
                # Threshold from measured data: walking chunks show hip-pitch
                # std ~0.12 rad, the dead-window reference ~0.01.
                walk_cmded = (abs(target[0]) + abs(target[1])
                              + abs(target[2])) > 0.10
                for attempt in range(3):
                    with replan_lock:
                        prep = backend.replan_prepare(target)
                    pred, npf = backend.replan_infer(prep)  # <-- no lock
                    if not walk_cmded:
                        break
                    hp_std = float(max(np.std(pred[:npf, 7]),
                                       np.std(pred[:npf, 13])))
                    if hp_std > 0.045:
                        break
                    _TAPE.ev("chunk_rejected", attempt=attempt + 1,
                             hip_pitch_std=round(hp_std, 4),
                             target=list(target))
                    log.warning(
                        "liveness gate: standing chunk while walk commanded "
                        "(hip_pitch_std=%.4f, attempt %d/3) -- re-rolling",
                        hp_std, attempt + 1)
                else:
                    log.error("liveness gate: 3 standing chunks in a row; "
                              "committing anyway (reference may slide)")
                with replan_lock:
                    backend.replan_commit(pred, npf)
            else:
                with replan_lock:
                    backend.replan(target)
        except Exception:
            log.exception("worker: replan failed; will retry next cycle")
            time.sleep(0.05)
            continue
        log.debug("worker: replan done in %.1fms (frames_remaining=%d)",
                  (time.monotonic() - t0) * 1000.0, backend.frames_remaining)
        _TAPE.ev("replan_done", ms=round((time.monotonic() - t0) * 1000.0, 1),
                 frames_remaining=int(backend.frames_remaining))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


class _IntentTape:
    """Machine-readable replay tape: jsonl of every intent (received and
    applied), every replan (with its RNG seed -> bit-exact offline replay),
    and every published tick, each stamped with monotonic + wall time.

    Exists because the human-readable daemon log has no reliable per-event
    timing (kplanner_gen_from_log.py had to invent a timing template). One
    tape per daemon start; capture_robot_run.py harvests it next to the
    deploy telemetry. Never raises: a broken tape must not touch the robot.

    Env: KPLANNER_TAPE=0 disables; KPLANNER_TAPE_DIR overrides the default
    <PC2_PREFIX or .>/log/kplanner_tape/ location.
    """

    def __init__(self) -> None:
        self._fh = None
        self._t0 = time.monotonic()
        if os.environ.get("KPLANNER_TAPE", "1") == "0":
            return
        try:
            root = os.environ.get("KPLANNER_TAPE_DIR")
            if not root:
                # rituals launch us with cwd=/ and no env: derive from the
                # script's own home (on PC2 that is /home/run/getsolo, which
                # has log/); fall back to /tmp rather than dying.
                prefix = os.environ.get("PC2_PREFIX", "")
                script_home = os.path.dirname(os.path.abspath(__file__))
                for base in (prefix, script_home, "."):
                    if base and os.path.isdir(os.path.join(base, "log")):
                        root = os.path.join(base, "log", "kplanner_tape")
                        break
                else:
                    import tempfile
                    root = os.path.join(tempfile.gettempdir(), "kplanner_tape")
            os.makedirs(root, exist_ok=True)
            path = os.path.join(
                root, time.strftime("tape_%Y%m%d_%H%M%S.jsonl"))
            self._fh = open(path, "a", buffering=1)
            # FRAME TAPE (full-content observability, run 20260719_214150:
            # source of a dead reference could not be attributed because no
            # artifact records what the planner actually puts on the wire).
            # Binary f32 records, 40 per tick:
            #   [tm, branch, root_xy(2), root_z, quat_xyzw(4), jpos(31)]
            # branch: 0=ring/planner 1=idle-anchor 2=stop-blend 3=dance.
            # Committed chunks are dumped whole as <session>_chunks/*.npy.
            self._ffh = open(path.replace(".jsonl", ".frames.f32"), "ab")
            self._chunk_dir = path.replace(".jsonl", "_chunks")
            os.makedirs(self._chunk_dir, exist_ok=True)
            self.ev("start", wall=time.time(), argv=sys.argv[1:])
            log.info("intent tape: %s (+frames.f32, +chunks/)", path)
        except Exception as exc:  # noqa: BLE001 - tape must never kill the daemon
            log.warning("intent tape disabled: %s", exc)
            self._fh = None
            self._ffh = None

    def frame(self, branch: float, xy, z: float, quat_xyzw, jpos) -> None:
        if getattr(self, "_ffh", None) is None:
            return
        try:
            rec = np.empty(40, dtype=np.float32)
            rec[0] = time.monotonic() - self._t0
            rec[1] = branch
            rec[2:4] = np.asarray(xy, dtype=np.float32)[:2]
            rec[4] = z
            rec[5:9] = np.asarray(quat_xyzw, dtype=np.float32)[:4]
            rec[9:40] = np.asarray(jpos, dtype=np.float32)[:31]
            rec.tofile(self._ffh)
            self._ffh.flush()
        except Exception:  # noqa: BLE001
            pass

    def chunk(self, pred: "np.ndarray", npf: int) -> None:
        if getattr(self, "_chunk_dir", None) is None:
            return
        try:
            tm = time.monotonic() - self._t0
            np.save(os.path.join(self._chunk_dir, f"chunk_{tm:09.3f}.npy"),
                    np.asarray(pred[:npf], dtype=np.float32))
        except Exception:  # noqa: BLE001
            pass

    def ev(self, kind: str, **kw) -> None:
        if self._fh is None:
            return
        try:
            kw["ev"] = kind
            kw["tm"] = round(time.monotonic() - self._t0, 6)
            kw["tw"] = round(time.time(), 3)
            self._fh.write(json.dumps(kw, default=str) + "\n")
        except Exception:  # noqa: BLE001
            pass


_TAPE = _IntentTape.__new__(_IntentTape)
_TAPE._fh = None   # inert until run() replaces it
_TAPE._ffh = None
_TAPE._chunk_dir = None


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def run(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    OnnxPlannerBackend.USE_GPU = bool(getattr(args, "ort_gpu", False))
    global _TAPE
    _TAPE = _IntentTape()

    global _RUNTIME_TURN_LEFT_SCALE, _RUNTIME_TURN_RIGHT_SCALE
    global _RUNTIME_FORWARD_SCALE, _RUNTIME_BACKWARD_SCALE, _RUNTIME_LATERAL_SCALE
    global _RUNTIME_STICK_SHAPING_EXPONENT, _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS
    global _SPEED_SETPOINT
    _RUNTIME_TURN_LEFT_SCALE = float(args.turn_left_scale)
    _RUNTIME_TURN_RIGHT_SCALE = float(args.turn_right_scale)
    _RUNTIME_FORWARD_SCALE = float(args.forward_scale)
    _RUNTIME_BACKWARD_SCALE = float(args.backward_scale)
    _RUNTIME_LATERAL_SCALE = float(args.lateral_scale)
    if args.stick_shape_exp > 0:
        _RUNTIME_STICK_SHAPING_EXPONENT = float(args.stick_shape_exp)
    _RUNTIME_CONTINUOUS_FORWARD_MIN_MPS = max(0.0, float(args.continuous_forward_min_mps))
    if args.speed_setpoint is not None:
        _SPEED_SETPOINT = max(_SETPOINT_MIN, min(_SETPOINT_MAX, float(args.speed_setpoint)))
    log.info("forward speed setpoint: %.2f m/s (X/Y speed_delta nudges, clamp "
             "[%.1f, %.1f])", _SPEED_SETPOINT, _SETPOINT_MIN, _SETPOINT_MAX)

    # ---- Resolve ports (offset support for laptop A/B testing).
    off = int(args.port_offset)
    pub_port = int(args.pub_port) + off
    cmd_port = int(args.cmd_port) + off
    clip_port = int(args.clip_cmd_port) + off
    if off:
        log.info("port offset %+d: pub=%d cmd=%d clip=%d",
                 off, pub_port, cmd_port, clip_port)

    if _port_in_use(pub_port, "127.0.0.1") or _port_in_use(pub_port, "0.0.0.0"):
        log.error("publish port %d already in use.", pub_port)
        return 1
    if _port_in_use(clip_port, "0.0.0.0"):
        log.error("motion_clip_cmd port %d already in use.", clip_port)
        return 1

    # ---- Backend (the slow part).
    if args.backend == "torch":
        for label, p in (("vqvae-ckpt", args.vqvae_ckpt),
                         ("pose-ckpt", args.pose_ckpt),
                         ("root-ckpt", args.root_ckpt)):
            if p is None or not Path(p).is_file():
                log.error("--backend torch requires --%s (got %s)", label, p)
                return 1
        backend = TorchPlannerBackend(
            vqvae_ckpt=Path(args.vqvae_ckpt),
            pose_ckpt=Path(args.pose_ckpt),
            root_ckpt=Path(args.root_ckpt),
            device=args.device,
            replan_threshold_frames=args.replan_threshold_frames,
            planner_mode=args.planner_mode,
        )
    else:
        if args.onnx is None or not Path(args.onnx).is_file():
            log.error("--backend onnx requires --onnx <graph.onnx> (got %s)",
                      args.onnx)
            return 1
        contract = _load_onnx_contract(
            Path(args.onnx),
            Path(args.onnx_sidecar) if args.onnx_sidecar else None,
        )
        backend = OnnxPlannerBackend(
            onnx_path=Path(args.onnx),
            contract=contract,
            replan_threshold_frames=args.replan_threshold_frames,
            planner_mode=args.planner_mode,
        )

    # ---- Warmup anchor + first (warm-the-model) replan.
    warmup_qpos = _load_warmup_qpos(
        Path(args.warmup_qpos) if args.warmup_qpos else None
    )
    backend.reset(warmup_qpos)
    intent_state = IntentState(_IDLE_INTENT)
    t0 = time.monotonic()
    backend.replan(_IDLE_INTENT)
    log.info("first replan complete in %.2fs; ring buffer has %d frames",
             time.monotonic() - t0, backend.frames_remaining)
    # Re-seed so the publish loop starts from a clean anchor buffer (the
    # warm-up replan output is discarded; x2_kplanner keeps it, but its
    # idle gate never reads it either -- the anchor freeze wins while idle).
    backend.reset(warmup_qpos)

    ref_smoother = ReferenceStepSmoother(
        ramp_duration_s=float(args.ref_smoother_ms) / 1000.0,
        trigger_rad=float(args.ref_smoother_trigger_rad),
        shape=args.ref_smoother_shape,
        blend_indices=_REF_SMOOTHER_JOINTS_PRESETS[args.ref_smoother_joints],
    )
    log.info("ref-smoother: shape=%s T=%.0fms trigger=%.3frad joints=%s enabled=%s",
             ref_smoother.shape, ref_smoother.ramp_duration_s * 1000.0,
             ref_smoother.trigger_rad, args.ref_smoother_joints,
             ref_smoother.enabled)

    arm_ingest: Optional[ArmTargetIngest] = None
    if int(getattr(args, "arm_port", 0)) > 0:
        arm_ingest = ArmTargetIngest(int(args.arm_port))
        _ARM_INGEST_REF[0] = arm_ingest

    publisher = PosePublisher(host=args.pub_host, port=pub_port,
                              topic=args.pub_topic)
    publisher.arm_ingest = arm_ingest
    log.info("publishing %r on tcp://%s:%d at %.1f Hz (backend=%s)",
             args.pub_topic, args.pub_host, pub_port, OUTPUT_FPS,
             backend.describe())

    cmd_queue: "queue.Queue[LocomotionCommand]" = queue.Queue()
    dance_queue: "queue.Queue[tuple]" = queue.Queue()
    stop_event = threading.Event()
    replan_event = threading.Event()
    replan_lock = threading.Lock()
    dance_active = threading.Event()
    publisher.dance_active = dance_active
    threads: list[threading.Thread] = []

    thr = threading.Thread(
        target=_zmq_command_thread,
        args=(cmd_queue, args.cmd_host, cmd_port, args.cmd_topic, stop_event,
              bool(args.cmd_bind)),
        name="cmd-zmq", daemon=True,
    )
    thr.start()
    threads.append(thr)

    # Forward-obstacle guard feed (scan_guard_pub.py over ZMQ). Fails open:
    # if that process is not running, _guard_blocked() stays False and the
    # planner behaves exactly as before.
    thr = threading.Thread(
        target=_scan_guard_thread, args=(stop_event,),
        name="scan-guard", daemon=True,
    )
    thr.start()
    threads.append(thr)

    thr = threading.Thread(
        target=_motion_clip_cmd_thread,
        args=(dance_queue, clip_port, args.clip_cmd_topic,
              Path(args.dances_dir), stop_event),
        name="cmd-motion-clip", daemon=True,
    )
    thr.start()
    threads.append(thr)

    thr = threading.Thread(
        target=_planner_worker,
        args=(backend, intent_state, replan_lock, stop_event, replan_event,
              dance_active),
        kwargs={"cold_start_ramp_tau_s": float(args.cold_start_ramp_tau_s)},
        name="kplanner-worker", daemon=True,
    )
    thr.start()
    threads.append(thr)

    # Measured-yaw rebase (port of the watchdog's x2_debug yaw fix): the
    # planner is the LIVE producer, and the watchdog only rebases its OWN
    # fallback states -- so the planner must rebase its published root
    # quats to the robot's heading itself, else SONIC twists to world +X.
    measured_yaw = MeasuredYaw()
    yaw_rebase_enabled = (not args.no_yaw_rebase) and int(args.x2_debug_port) > 0
    if yaw_rebase_enabled:
        thr = threading.Thread(
            target=_x2_debug_thread,
            args=(measured_yaw, args.x2_debug_host, int(args.x2_debug_port),
                  args.x2_debug_topic, stop_event),
            name="x2-debug-yaw", daemon=True,
        )
        thr.start()
        threads.append(thr)

    def _on_signal(signum: int, _frame: object) -> None:
        log.info("signal %d -> shutting down", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    period_s = 1.0 / OUTPUT_FPS
    next_tick = time.monotonic()
    end_at = time.monotonic() + args.duration_s if args.duration_s > 0 else float("inf")

    anchor_jpos = warmup_qpos[7:].astype(np.float32).copy()
    # Integrated world root persisted across IDLE <-> PLAYING <-> DANCE.
    current_root_xy = warmup_qpos[:2].astype(np.float64).copy()
    current_root_wxyz = warmup_qpos[3:7].astype(np.float32).copy()
    current_root_z = float(warmup_qpos[2])

    # One-shot measured-yaw offset (captured at ignition, below). The
    # planner runs in a world +X frame; rotating every published root
    # quat by R_z(yaw_offset) aligns that frame to the robot's actual
    # heading so idle holds heading and drive/turns are relative to it.
    # Kept constant per session (not continuous) so it never double-
    # counts the planner's own yaw integration during PLAYING. None ==
    # no x2_debug (sim/laptop) -> rebase is a no-op.
    yaw_offset: list[Optional[float]] = [None]
    # PLAYING-scope heading trim (2026-07-20). The mid-walk complement of the
    # IDLE yaw resync: a slew-limited, deadbanded wire-frame rotation that
    # bleeds reference-vs-measured heading error DURING walks, so nudges and
    # the root model's open-loop yaw wander (measured 6-33 deg per walk;
    # reference-led whip at the worst stumble) never accumulate into a
    # violent SONIC correction. ASSIGNMENT-form servo on the published error
    # (converges; step -> 0 as published -> measured), never the multiply
    # form of the 2026-07-18 runaway. Gated off while a turn is commanded so
    # deliberate turns still lead the robot. Default OFF (--playing-yaw-
    # resync-dps 0); evaluate in the sim stack first.
    yaw_trim: list[float] = [0.0]

    def _eff_off() -> Optional[float]:
        if yaw_offset[0] is None:
            return None
        return yaw_offset[0] + yaw_trim[0]

    def _reb1(q_xyzw: np.ndarray) -> np.ndarray:
        off = _eff_off()
        if off is None:
            return q_xyzw
        return rebase_quats_xyzw_by_yaw(
            np.asarray(q_xyzw, dtype=np.float32).reshape(1, 4), off
        )[0]

    def _rebL(qs: list[np.ndarray]) -> list[np.ndarray]:
        if yaw_offset[0] is None:
            return qs
        return [_reb1(q) for q in qs]

    def _reb_xy(xy: np.ndarray) -> np.ndarray:
        """Rotate a planner-frame root XY into the wire frame.

        The quat has ALWAYS been rebased by R_z(yaw_offset) (_reb1) but the
        XY went out raw -- so published position and orientation were in
        frames that disagree by the ignition heading. Nothing on the robot
        consumed XY (SONIC's tokenizer obs is joints + relative orientation
        only), but every tape consumer (frame tape, overlay, gait metrics)
        saw forward walks as world-frame crab-walks (2026-07-20, operator-
        caught). Rotate XY by the same offset so the wire is self-consistent.
        """
        off = _eff_off()
        if off is None:
            return xy
        c = math.cos(off)
        s = math.sin(off)
        return np.array([c * xy[0] - s * xy[1],
                         s * xy[0] + c * xy[1]], dtype=np.float64)

    def _resync_idle_yaw_from_measured() -> None:
        """IDLE-only: re-anchor the reference heading to the MEASURED yaw.

        Port of ``x2_kplanner.py:3094`` ("Yaw-only resync from robot_pose
        feedback"), whose omission is listed as a known deviation at the top of
        this file. Without it ``current_root_wxyz`` is only ever written by the
        model's own predictions, so anything that moves the real robot off-yaw
        while the stick is centred -- a push, a slip, fall recovery -- leaves us
        publishing a stale ABSOLUTE yaw target. The C++ tokenizer feeds SONIC
        ``rel = inv(measured) * reference``, so a stale reference makes the policy
        twist the body back to the old heading: the "robot always tries to recover
        to the same world orientation" symptom.

        WHY THIS IS SAFE, where the 2026-07-18 runaway spin was not:
        that incident PRE-MULTIPLIED the published quat by a live-updating offset,
        so the planner's frozen -35 deg residual survived as a constant lead and
        the robot chased it forever. This REPLACES the planner's internal heading
        belief with the measurement, discarding the residual. At rest the
        commanded heading equals the measured heading, so the error is exactly
        zero and there is no constant to chase. Assignment, not feedback.

        FRAME: ``current_root_wxyz`` is in the PLANNER frame, but what reaches the
        wire is ``_reb1()`` = ``R_z(yaw_offset) (x) current_root_wxyz``, and
        ``measured_yaw`` is world-frame. So the planner-frame target is
        ``measured - yaw_offset``; the published yaw then comes out as exactly
        ``measured``. Writing ``R_z(measured)`` here instead would double-count
        the ignition offset.

        Also fixes turn-start drift: ``_build_warm_qpos()`` seeds the ring from
        ``current_root_wxyz`` on IDLE -> PLAYING, so keeping it truthful means each
        turn begins from where the robot ACTUALLY points rather than from
        accumulated open-loop error. (Within a single sustained turn PLAYING still
        publishes model-predicted yaw verbatim -- deliberately, so commanded turns
        execute as intended.)
        """
        nonlocal current_root_wxyz
        off = _eff_off()
        if off is None:
            return                      # rebase not armed (sim / no x2_debug)
        m = measured_yaw.get(max_age_s=POSE_FEEDBACK_MAX_AGE_S)
        if m is None:
            return                      # stale -> HOLD LAST GOOD, never identity
        target = _wrap_pi(float(m) - float(off))   # -> planner frame
        cur = _yaw_of_quat_xyzw(_idle_root_xyzw())
        step = _wrap_pi(target - cur)
        cap = MAX_YAW_RESYNC_RAD_S / OUTPUT_FPS
        if step > cap:
            step = cap
        elif step < -cap:
            step = -cap
        current_root_wxyz = _quat_wxyz_from_yaw(_wrap_pi(cur + step))

    global_tick = 0
    last_intent_log: tuple[float, float, float, float] = _IDLE_INTENT
    is_playing = False           # False == IDLE_LOOP (frozen anchor)
    dance: Optional[DancePlayback] = None
    dance_started_t = 0.0
    post_dance_hold_until: Optional[float] = None
    _watchdog_last_log_t = 0.0

    # ---- PLAYING -> IDLE stop blend (2026-07-19) --------------------------
    # Releasing the stick used to snap the reference from the mid-stride gait
    # frame straight to the frozen idle anchor in ONE 20 ms tick. Measured on
    # the real robot (docs/experiments/incident_20260719_yaw_oscillation_fall):
    # single-tick reference jumps up to 1.213 rad (69.5 deg) on ankle_pitch --
    # ~3475 deg/s of commanded joint velocity, physically untrackable. Two of
    # those landed in the seconds before the robot lost balance.
    #
    # G1's stock stack never does this: its idle is a MODE, not a pose. The
    # controller keeps feeding the model the current velocity + heading and the
    # model generates a natural stop, which is why the G1 takes another step or
    # two after you release the key. We can't generate a stop without the model,
    # but we can stop TELEPORTING: cross-fade the last gait frame into the
    # anchor over STOP_BLEND_FRAMES with a half-cosine ease.
    stop_blend_from: Optional[np.ndarray] = None
    stop_blend_left: int = 0
    last_gait_jpos: Optional[np.ndarray] = None

    def _idle_root_xyzw() -> np.ndarray:
        return _wxyz_to_xyzw(current_root_wxyz)

    def _build_warm_qpos() -> np.ndarray:
        warm = np.empty_like(warmup_qpos)
        warm[0] = current_root_xy[0]
        warm[1] = current_root_xy[1]
        warm[2] = current_root_z
        warm[3:7] = current_root_wxyz
        warm[7:] = warmup_qpos[7:]
        return warm

    def _publish_anchor_tick() -> None:
        """One idle-anchor frame (current + 9 identical futures). The
        smoother shapes only the CURRENT frame; the future window stays
        raw -- same as x2_kplanner (futures are built pre-smoother)."""
        nonlocal global_tick
        # Close the yaw loop for IDLE only. This is the ASSIGNMENT form (replace
        # the planner's internal belief with the measurement), NOT the multiply
        # form that caused the 2026-07-18 runaway spin. Reached only from the
        # ``not is_playing`` branch, so PLAYING keeps publishing model-predicted
        # yaw verbatim and commanded turns still execute as intended.
        nonlocal stop_blend_left
        _resync_idle_yaw_from_measured()
        xyzw = _reb1(_idle_root_xyzw())

        # Stop blend: ease the last gait frame into the anchor instead of
        # snapping. w goes 0 -> 1 over STOP_BLEND_FRAMES (half-cosine, so the
        # derivative is zero at BOTH ends -- no velocity step at entry or exit).
        target_jpos = anchor_jpos
        if stop_blend_left > 0 and stop_blend_from is not None:
            done = 1.0 - (stop_blend_left / float(STOP_BLEND_FRAMES))
            w = 0.5 * (1.0 - math.cos(math.pi * done))
            target_jpos = (stop_blend_from * (1.0 - w)
                           + anchor_jpos * w).astype(anchor_jpos.dtype)
            stop_blend_left -= 1
        jpos = ref_smoother.update(target_jpos, time.monotonic())
        wire_xy = _reb_xy(current_root_xy)
        payload = build_pose_payload_np(
            jpos, xyzw, wire_xy, current_root_z, global_tick,
            future_jpos=[anchor_jpos] * NUM_FUTURE,
            future_quat=[xyzw] * NUM_FUTURE,
            hand_dof=args.hand_dof,
        )
        publisher.publish(payload)
        _TAPE.frame(2.0 if stop_blend_left > 0 else 1.0,
                    wire_xy, current_root_z, xyzw, jpos)
        global_tick += 1

    try:
        with PidFile(Path(args.pid_file)):
            # ---- Yaw-capture gate (ordering fix). x2_debug comes from the
            # C++ deploy, which the ritual starts AFTER this planner (the
            # gate before deploy is satisfied by the watchdog's COLD_IDLE,
            # not by us). So stay SILENT until the first x2_debug frame
            # lands -- the watchdog holds the robot at its measured heading
            # (its own rebased idle clip) during the wait -- then latch the
            # ignition heading ONCE. Publishing LIVE identity frames before
            # capture would make SONIC twist to world +X (the snap). A
            # generous fail-safe timeout proceeds unrebased if x2_debug
            # never appears (deploy never started / regression escape).
            if yaw_rebase_enabled:
                # FAIL-STOP, NOT FAIL-OPEN (2026-07-18). This wait used to give
                # up after --yaw-capture-timeout-s and publish unrebased. That
                # is the WORST possible fallback: an unrebased publish is
                # identity == world +X, a KNOWN-WRONG heading SONIC actively
                # twists the body toward. Measured on hardware: commanded yaw
                # pinned at exactly 0.0 deg while a 40 deg hand-nudge was driven
                # back to -0.5 deg in ~1.5s.
                #
                # It also could not succeed on a clean start: x2_debug is
                # published by the DEPLOY, which the ritual starts AFTER this
                # planner, so a bounded wait always expired. It only ever armed
                # when a PREVIOUS deploy happened to still be alive.
                #
                # Waiting indefinitely IS the lazy-arm: we latch on the first
                # x2_debug frame whenever it arrives (seconds after the deploy
                # comes up). Staying silent meanwhile is safe and intended --
                # the watchdog holds the robot on its own measured-yaw-rebased
                # idle clip, and the ritual's pre-deploy pose gate is satisfied
                # by the watchdog's COLD_IDLE, not by us. No stream, no snap.
                warn_every_s = max(5.0, float(args.yaw_capture_timeout_s))
                next_warn = time.monotonic() + warn_every_s
                waited_s = 0.0
                log.info("waiting for x2_debug to capture ignition heading "
                         "(silent; watchdog holds; will NOT proceed unrebased)...")
                while not stop_event.is_set():
                    cap = measured_yaw.get(max_age_s=1.0)
                    if cap is not None:
                        yaw_offset[0] = cap
                        log.info("measured-yaw rebase ARMED after %.1fs: "
                                 "ignition heading %.1f deg "
                                 "(root quats -> robot frame)",
                                 waited_s, math.degrees(cap))
                        break
                    now_w = time.monotonic()
                    if now_w >= next_warn:
                        next_warn = now_w + warn_every_s
                        log.warning("still no x2_debug after %.0fs -- staying "
                                    "SILENT (watchdog holds the robot). This is "
                                    "expected until the deploy starts; it "
                                    "publishes x2_debug. Planner will arm and "
                                    "begin publishing automatically.",
                                    waited_s)
                    time.sleep(period_s)   # SILENT: do not publish pre-capture
                    waited_s += period_s

            # ---- Quiet-stand warmup (frozen anchor).
            warmup_n = int(round(max(0.0, args.warmup_quiet_stand_s) * OUTPUT_FPS))
            if warmup_n > 0:
                log.info("quiet-stand warmup: %d ticks (%.2fs) of frozen anchor",
                         warmup_n, args.warmup_quiet_stand_s)
                for _ in range(warmup_n):
                    if stop_event.is_set() or time.monotonic() >= end_at:
                        break
                    _publish_anchor_tick()
                    next_tick += period_s
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        next_tick = time.monotonic()
                log.info("quiet-stand warmup done; planner active.")

            while not stop_event.is_set() and time.monotonic() < end_at:
                # ---- Dance command queue (drain everything; last wins).
                while True:
                    try:
                        item = dance_queue.get_nowait()
                    except queue.Empty:
                        break
                    if item[0] == "stop":
                        if dance is not None:
                            log.info("dance %s: STOP; idle hold %.1fs then "
                                     "planner resumes", dance.name,
                                     args.post_dance_idle_s)
                            dance = None
                            dance_active.clear()
                            post_dance_hold_until = (
                                time.monotonic() + args.post_dance_idle_s
                            )
                    else:
                        _, name, dof, quat, fps = item
                        cur_yaw = _yaw_of_quat_xyzw(_idle_root_xyzw())
                        dance = DancePlayback(dof, quat, fps, cur_yaw, name)
                        dance_started_t = time.monotonic()
                        dance_active.set()
                        post_dance_hold_until = None
                        # Planner ring paused + cleared: freeze the FSM at
                        # IDLE so the post-dance resume path re-seeds the
                        # neural buffer at the (new) current root.
                        is_playing = False
                        with replan_lock:
                            backend.reset(_build_warm_qpos())
                        log.info("dance %s: START (planner preempted)", name)

                # ---- Drain planner_cmd queue; apply the latest intent.
                latest_cmd: Optional[LocomotionCommand] = None
                while True:
                    try:
                        latest_cmd = cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                if latest_cmd is not None:
                    target = intent_to_velocity(latest_cmd)
                    intent_state.set(target)
                    if target != last_intent_log:
                        log.info("intent applied (%s, %s, %s) -> target=%s",
                                 latest_cmd.intent, latest_cmd.magnitude,
                                 latest_cmd.source, target)
                        _TAPE.ev("intent_applied", target=list(target),
                                 intent=latest_cmd.intent,
                                 magnitude=latest_cmd.magnitude)
                        last_intent_log = target

                # ---- Stale-command watchdog (opt-in, default OFF).
                if args.command_watchdog_s > 0.0 and intent_state.force_idle_if_stale(
                    max_age_s=args.command_watchdog_s, idle_target=_IDLE_INTENT
                ):
                    last_intent_log = _IDLE_INTENT
                    now_t = time.monotonic()
                    if now_t - _watchdog_last_log_t > 1.0:
                        log.warning(
                            "command watchdog: no upstream intent for %.2fs "
                            "(threshold %.2fs); forcing IDLE",
                            intent_state.seconds_since_last_set(),
                            args.command_watchdog_s,
                        )
                        _watchdog_last_log_t = now_t

                # =========== DANCE branch (preempts planner output) ========
                if dance is not None:
                    jpos, quat, fut_j, fut_q = dance.tick()
                    # Persist the streamed heading so post-dance idle (and
                    # the next PLAYING seed) holds the dance-final yaw.
                    current_root_wxyz = np.array(
                        [quat[3], quat[0], quat[1], quat[2]], dtype=np.float32
                    )
                    now_mono = time.monotonic()
                    jpos_s = ref_smoother.update(
                        jpos.astype(np.float32), now_mono,
                        # Only the entry step (anchor -> clip frame 0) may
                        # arm a ramp; the clip's own fast motion must not.
                        allow_arm=(now_mono - dance_started_t)
                        <= ref_smoother.ramp_duration_s,
                    )
                    payload = build_pose_payload_np(
                        jpos_s, _reb1(quat), _reb_xy(current_root_xy),
                        current_root_z,
                        global_tick, future_jpos=fut_j,
                        future_quat=_rebL(fut_q),
                        hand_dof=args.hand_dof,
                    )
                    publisher.publish(payload)
                    global_tick += 1
                    if dance.finished:
                        log.info("dance %s: clip complete; idle hold %.1fs "
                                 "then planner resumes", dance.name,
                                 args.post_dance_idle_s)
                        dance = None
                        dance_active.clear()
                        post_dance_hold_until = (
                            time.monotonic() + args.post_dance_idle_s
                        )
                    next_tick += period_s
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    elif -slack > 5 * period_s:
                        next_tick = time.monotonic()
                    continue

                # =========== POST-DANCE idle hold ==========================
                if post_dance_hold_until is not None:
                    if time.monotonic() < post_dance_hold_until:
                        _publish_anchor_tick()
                        next_tick += period_s
                        slack = next_tick - time.monotonic()
                        if slack > 0:
                            time.sleep(slack)
                        elif -slack > 5 * period_s:
                            next_tick = time.monotonic()
                        continue
                    post_dance_hold_until = None
                    log.info("post-dance idle hold done; planner resumes")

                # =========== Normal IDLE_LOOP / PLAYING FSM ================
                current_target, _ = intent_state.get()
                is_idle = tuple(current_target) == _IDLE_INTENT
                if (not is_idle) != is_playing:
                    if not is_idle:
                        # IDLE -> PLAYING: seed the ring at the CURRENT
                        # integrated root so velocity intents start from
                        # where the robot is (port of x2_kplanner's warm
                        # seed; no pose-feedback yaw refresh -- open loop).
                        warm = _build_warm_qpos()
                        with replan_lock:
                            backend.reset(warm)
                            backend._force_replan = True
                        replan_event.set()
                        log.info(
                            "state: IDLE_LOOP -> PLAYING (intent=%s); buffer "
                            "seeded at root_xy=%s yaw_wxyz=%s, replan queued",
                            current_target, warm[:2].tolist(),
                            warm[3:7].tolist(),
                        )
                    else:
                        # Arm the stop blend from the LAST gait frame so the
                        # reference eases into the anchor instead of snapping.
                        if last_gait_jpos is not None:
                            stop_blend_from = last_gait_jpos.copy()
                            stop_blend_left = STOP_BLEND_FRAMES
                        log.info(
                            "state: PLAYING -> IDLE_LOOP (intent back to "
                            "idle); blending to anchor over %d ticks; "
                            "freezing at root_xy=%s yaw_wxyz=%s",
                            STOP_BLEND_FRAMES,
                            current_root_xy.tolist(),
                            current_root_wxyz.tolist(),
                        )
                    is_playing = not is_idle

                if not is_playing:
                    _publish_anchor_tick()
                else:
                    with replan_lock:
                        # Resampled read (30 Hz model -> OUTPUT_FPS) with an
                        # 8-tick cross-fade blend at each replan seam. Future
                        # slot k is +0.1*(k+1) s in REAL time (STEP_TICKS=5
                        # output ticks * 0.6 native/tick = 3 native = 0.1 s).
                        qpos_np = backend.get_next_frame_resampled(OUTPUT_FPS)
                        if backend.should_replan():
                            replan_event.set()
                        future_qposes: list[np.ndarray] = []
                        for k in range(NUM_FUTURE):
                            future_qposes.append(
                                backend.peek_output_frame(STEP_TICKS * (k + 1))
                            )

                    # Yaw-lock mitigation (opt-in diagnostic; port).
                    yaw_locked = (
                        args.yaw_lock_epsilon > 0.0
                        and abs(float(current_target[0])) < args.yaw_lock_epsilon
                    )
                    if yaw_locked:
                        qpos_np[3:7] = current_root_wxyz.astype(qpos_np.dtype)
                        for fq in future_qposes:
                            fq[3:7] = current_root_wxyz.astype(fq.dtype)

                    current_root_xy = qpos_np[:2].astype(np.float64).copy()
                    current_root_z = float(qpos_np[2])
                    if not yaw_locked:
                        current_root_wxyz = qpos_np[3:7].astype(np.float32).copy()

                    jpos = ref_smoother.update(
                        qpos_np[7:].astype(np.float32), time.monotonic()
                    )
                    # Remember what SONIC actually last received, so the stop
                    # blend starts from the published frame (exact continuity)
                    # rather than the raw planner frame.
                    last_gait_jpos = jpos.copy()
                    # PLAYING yaw resync (see yaw_trim above): bleed the
                    # published-vs-measured heading error, slew-limited and
                    # deadbanded, only while no turn is commanded.
                    if (args.playing_yaw_resync_dps > 0.0
                            and yaw_offset[0] is not None
                            and abs(float(current_target[0])) < 0.05):
                        m_yaw = measured_yaw.get(
                            max_age_s=POSE_FEEDBACK_MAX_AGE_S)
                        if m_yaw is not None:
                            pub_yaw = _wrap_pi(
                                _yaw_of_quat_xyzw(_wxyz_to_xyzw(qpos_np[3:7]))
                                + _eff_off())
                            err = _wrap_pi(float(m_yaw) - pub_yaw)
                            dead = math.radians(
                                args.playing_yaw_resync_deadband_deg)
                            if abs(err) > dead:
                                cap = (math.radians(
                                    args.playing_yaw_resync_dps) / OUTPUT_FPS)
                                yaw_trim[0] = float(np.clip(
                                    yaw_trim[0] + max(-cap, min(cap, err)),
                                    -0.6, 0.6))
                    wire_xyzw = _reb1(_wxyz_to_xyzw(qpos_np[3:7]))
                    wire_xy = _reb_xy(current_root_xy)
                    payload = build_pose_payload_np(
                        jpos,
                        wire_xyzw,
                        wire_xy,
                        current_root_z,
                        global_tick,
                        future_jpos=[fq[7:].astype(np.float32)
                                     for fq in future_qposes],
                        future_quat=_rebL([_wxyz_to_xyzw(fq[3:7])
                                           for fq in future_qposes]),
                        hand_dof=args.hand_dof,
                    )
                    publisher.publish(payload)
                    _TAPE.frame(0.0, wire_xy, current_root_z,
                                wire_xyzw, jpos)
                    global_tick += 1

                next_tick += period_s
                slack = next_tick - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    if -slack > 5 * period_s:
                        log.warning("loop fell behind by %.0fms; resyncing",
                                    -slack * 1000)
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


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pc2_kplanner_onnx",
        description=(
            "Torch-free X2 kinematic planner runtime for PC2 (onnxruntime "
            "fused graph); slim port of x2_kplanner's publish/replan loop "
            "with built-in x2m2 dance playback."
        ),
    )
    p.add_argument("--backend", choices=("onnx", "torch"), default="onnx",
                   help="onnx (PC2 default, torch-free) or torch (laptop A/B).")
    p.add_argument("--onnx", type=Path, default=None,
                   help="Fused planner graph (.onnx) for --backend onnx.")
    p.add_argument("--onnx-sidecar", type=Path, default=None,
                   help="JSON runtime contract (tensor names etc.); default "
                        "<onnx>.json or <dir>/runtime_contract.json, else "
                        "the export-script defaults.")
    p.add_argument("--planner-mode", choices=_PLANNER_MODE_NAMES, default=None,
                   help="Pose-template mode name (template graph / "
                        "replan_with_pose_template). Default None = "
                        "velocity-only path.")
    # torch backend checkpoints (laptop A/B only).
    p.add_argument("--vqvae-ckpt", type=Path, default=None)
    p.add_argument("--pose-ckpt", type=Path, default=None)
    p.add_argument("--root-ckpt", type=Path, default=None)
    p.add_argument("--device", default="cpu",
                   help="torch device for --backend torch (default cpu).")

    p.add_argument("--warmup-qpos", type=Path, default=DEFAULT_WARMUP_PKL,
                   help=f"Idle-anchor PKL (joblib; deploy-PKL or raw qpos "
                        f"schema). Default {DEFAULT_WARMUP_PKL}")
    p.add_argument("--warmup-quiet-stand-s", type=float, default=0.5)

    net = p.add_argument_group("network (real PC2 defaults; use "
                               "--port-offset for laptop testing)")
    net.add_argument("--pub-host", default="0.0.0.0",
                     help="PUB bind host (default 0.0.0.0 = tcp://*).")
    net.add_argument("--pub-port", type=int, default=DEFAULT_PUB_PORT)
    net.add_argument("--pub-topic", default="pose")
    net.add_argument("--cmd-host", default="127.0.0.1",
                     help="planner_cmd SUB connect host (pad bridge --bind).")
    net.add_argument("--cmd-port", type=int, default=DEFAULT_CMD_PORT)
    net.add_argument("--cmd-topic", default="planner_cmd")
    net.add_argument("--cmd-bind", action="store_true",
                     help="bind the planner_cmd SUB instead of connecting, so "
                          "multiple sources (pad bridge + quest3 manager) can "
                          "PUB-connect into it; --cmd-host is ignored.")
    net.add_argument("--arm-port", type=int, default=5572,
                     help="VR arm/hand target ingest: SUB bind port the "
                          "laptop quest3 manager PUB-connects to with "
                          "--arm-connect (arm_targets + hand_finger_cmd; "
                          "overlaid onto the pose wire). 0 disables.")
    net.add_argument("--clip-cmd-port", type=int, default=DEFAULT_CLIP_CMD_PORT,
                     help="motion_clip_cmd SUB bind port.")
    net.add_argument("--clip-cmd-topic", default="motion_clip_cmd")
    net.add_argument("--x2-debug-host", default="127.0.0.1",
                     help="deploy x2_debug PUB host (measured-yaw rebase).")
    net.add_argument("--x2-debug-port", type=int, default=5557,
                     help="deploy x2_debug PUB port; <=0 disables the SUB.")
    net.add_argument("--x2-debug-topic", default="x2_debug")
    net.add_argument("--no-yaw-rebase", action="store_true",
                     help="disable measured-yaw rebase of published root "
                          "quats (published frames stay in world +X frame; "
                          "SONIC will twist to spawn heading -- regression "
                          "escape only).")
    net.add_argument("--yaw-capture-timeout-s", type=float, default=30.0,
                     help="how long to stay silent waiting for the first "
                          "x2_debug frame (deploy starts AFTER this planner) "
                          "before proceeding without rebase. The watchdog "
                          "holds the robot during the wait, so a generous "
                          "value is safe on the robot; sim has no x2_debug so "
                          "pass --no-yaw-rebase there.")
    net.add_argument("--port-offset", type=int, default=0,
                     help="Added to pub/cmd/clip ports (laptop testing; the "
                          "live stack owns 5556/5563/5568).")

    p.add_argument("--dances-dir", type=Path, default=DEFAULT_DANCES_DIR,
                   help=f"Directory of <motion_key>.x2m2 bakes. Default "
                        f"{DEFAULT_DANCES_DIR}")
    p.add_argument("--post-dance-idle-s", type=float, default=2.0,
                   help="Idle-anchor stream duration after a clip ends or "
                        "is stopped, before the planner resumes.")

    p.add_argument(
        "--replan-threshold-frames", type=int, default=32,
        help="Replan when this many model frames (30fps) remain. Was 16 = "
             "0.53s, which PC2's 0.3-0.6s CPU inference consumed entirely, "
             "starving the ring at every mid-walk seam (tape 20260719). 32 "
             "gives ~0.4s commit margin at worst-case latency.")
    p.add_argument("--duration-s", type=float, default=0.0)
    p.add_argument("--hand-dof", type=int, default=DEFAULT_HAND_DOF)
    p.add_argument("--ort-gpu", action="store_true",
                   help="request CUDAExecutionProvider (CPU fallback). No-op "
                        "unless the venv has a Jetson GPU build of onnxruntime. "
                        "GPU inference (~tens of ms vs 0.3-0.6s CPU) would also "
                        "let --replan-threshold-frames drop back toward 16.")
    p.add_argument("--pid-file", type=Path,
                   default=Path("/tmp/pc2_kplanner_onnx.pid"))

    tune = p.add_argument_group("velocity tuning (ported from x2_kplanner)")
    tune.add_argument("--speed-setpoint", type=float, default=None,
                      help="Initial forward speed setpoint m/s (default 0.3 "
                           "or KPLANNER_FIXED_FWD_MPS env).")
    tune.add_argument("--turn-left-scale", type=float, default=1.0)
    tune.add_argument("--turn-right-scale", type=float, default=1.0)
    tune.add_argument("--forward-scale", type=float, default=1.0)
    tune.add_argument("--backward-scale", type=float, default=1.0)
    tune.add_argument("--lateral-scale", type=float, default=1.0)
    tune.add_argument("--stick-shape-exp", type=float,
                      default=_DEFAULT_STICK_SHAPING_EXPONENT)
    tune.add_argument("--continuous-forward-min-mps", type=float,
                      default=_DEFAULT_CONTINUOUS_FORWARD_MIN_MPS)
    tune.add_argument("--yaw-lock-epsilon", type=float, default=0.0)
    tune.add_argument("--playing-yaw-resync-dps", type=float, default=0.0,
                      help="If >0, bleed published-vs-measured heading error "
                           "DURING walks at this slew rate (deg/s), deadbanded, "
                           "gated off while a turn is commanded. Assignment-"
                           "form wire trim; 0 = off (legacy). Try ~10.")
    tune.add_argument("--playing-yaw-resync-deadband-deg", type=float,
                      default=2.0,
                      help="No resync while |error| is under this (noise).")
    tune.add_argument("--cold-start-ramp-tau-s", type=float, default=0.0)
    tune.add_argument("--command-watchdog-s", type=float, default=0.0)

    sm = p.add_argument_group("reference-step smoother")
    sm.add_argument("--ref-smoother-ms", type=float, default=300.0)
    sm.add_argument("--ref-smoother-trigger-rad", type=float, default=0.05)
    # Default OFF: the 30->50 Hz output resampling + 8-tick cross-fade blend
    # now handles replan-seam discontinuities at the source; the ref-smoother
    # is opt-in (pass --ref-smoother-shape halfcos to re-enable).
    sm.add_argument("--ref-smoother-shape", choices=_REF_SMOOTHER_SHAPES,
                    default="off")
    sm.add_argument("--ref-smoother-joints",
                    choices=list(_REF_SMOOTHER_JOINTS_PRESETS),
                    default="lower_body")

    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
