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
import collections
import json
import logging
import math
import os
import pickle
import queue
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field
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
# Yaw-rate ceiling for the **continuous-locomotion** path (Quest 3
# R-stick X axis after deadzone rescale + stick shaping). Decoupled
# from the bucketed ``_TURN_45_RAD_S`` because:
#
#   * The current X2 root model trained on
#     ``Loop_Forward_Walk_001__A018``, which contains essentially zero
#     yaw motion. Any non-zero yaw_rate is extrapolation; the larger
#     the yaw the more OOD the prediction. Empirically full-stick at
#     1.5 rad/s (~86 deg/s = a 90 deg turn in ~1 s) overdrives the
#     model and the operator reports "turns are too aggressive". A
#     90 deg turn in ~2 s (0.75 rad/s, ~43 deg/s) sits comfortably
#     inside what the policy can track.
#   * Bucketed callers (``turn_left / deg_45`` etc.) intentionally
#     want sharp pivots from a single button press -- those continue
#     to use the legacy ``_TURN_*_RAD_S`` table.
#
# Tunable at runtime via ``--continuous-turn-max-rad-s`` on the
# kplanner CLI, or ``KPLANNER_CONTINUOUS_TURN_MAX_RAD_S`` env var on
# the Quest 3 / PKL wrappers. The per-side ``_RUNTIME_TURN_LEFT_SCALE
# / _RIGHT_SCALE`` runtime scales still apply on top (default 1.0)
# so the operator can compensate for any L/R asymmetry independently.
_DEFAULT_CONTINUOUS_TURN_MAX_RAD_S: float = 0.75

# Mutable runtime knob, mutated by ``run()`` per CLI flag. Reads in
# ``_resolve_locomotion_continuous`` pick up the override at every
# dispatch call.
_CONTINUOUS_TURN_MAX_RAD_S: float = _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S
# Default hip-height TARGET (channel 3 of the velocity intent the model
# consumes). This is NOT a metadata field -- ``NeuralPlannerCore``
# wires it into ``implied_target_y`` (see
# ``motion_backbone/inference/neural_planner.py:414``), so the model
# treats it as the world-frame pelvis Y the predicted gait must drive
# toward.
#
# Must therefore match the **training distribution** of pelvis height.
# The X2 PKL corpus (``x2_ultra_locowalk.pkl``) has pelvis_z spanning
# roughly 0.595 - 0.726 m with a mean around 0.661 m; the working
# PKL-replay sweep (``x2_pkl_command_source --use-mean-intent``)
# computed 0.687 m and produced the only configuration where the
# deploy actually walked (see "Deploy-integration diagnostics" in
# motionbricks/docs/x2_kplanner_evaluation.md).
#
# The historical value (0.95 m, "matches idle_stand") was a stale
# carry-over from an earlier checkpoint whose stand pose sat ~25 cm
# higher than the current ``_TRAINING_DEFAULT_HIP_Z = 0.636 m``.
# Feeding 0.95 to the current model puts the target pelvis ~25 cm
# above any pose in its training distribution; the model produces
# OOD predictions and the policy can't track them, presenting to the
# operator as "robot won't walk forward even on full stick". This
# was the dominant failure mode on the Quest 3 stack before
# 2026-05-30. PKL replay was insulated because ``direct_velocity``
# carries hip_h verbatim from the clip (0.687 m).
_HIP_HEIGHT_M: float = 0.687

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

# ---------------------------------------------------------------------------
# Continuous locomotion (analog Quest3 sticks)
# ---------------------------------------------------------------------------
#
# When the IntentDecoder is run with ``enable_continuous_locomotion=True``
# (the kplanner-mode wrapper flips this on), the manager publishes
# ``locomotion / continuous`` commands carrying three deadzone-rescaled
# stick deflections in ``[-1, 1]``:
#
#   stick_fwd  > 0 -> forward, < 0 -> backward
#   stick_side > 0 -> right,   < 0 -> left
#   stick_yaw  > 0 -> turn-right (negative yaw_rate), < 0 -> turn-left
#
# We shape each axis with a power curve (sign preserved) so the
# operator gets fine resolution near zero with full speed at full
# deflection -- this kills the "binary slam" failure mode that was
# making the SONIC controller fall when the bucketed path stepped the
# velocity 0 -> 0.5 m/s in a single tick. The peak velocities are
# pinned to the same constants the bucketed path uses
# (``_WALK_SPEED_MPS`` etc.) so scripted demos and the kplanner sweep
# harness stay calibrated.
#
# Default exponent = 1.0 (linear). Bigger -> harder to reach full
# speed (more fine control near zero, but operators report "robot
# struggles to move forward" because 50% stick deflection produces
# only 25% velocity at exp=2.0). Smaller -> easier to reach full
# speed (more bucketed-like). Override at runtime via ``--stick-shape-
# exp`` (CLI) or ``KPLANNER_STICK_SHAPE_EXP`` (env var on the
# wrapper). At exp=0.5 a 50% stick gives 71% of max velocity; at
# exp=1.0 a 50% stick gives 50%; at exp=2.0 a 50% stick gives 25%.

_DEFAULT_STICK_SHAPING_EXPONENT: float = 1.0

# Mutable runtime-tuned exponent, set from ``run()`` per CLI flag.
_RUNTIME_STICK_SHAPING_EXPONENT: float = _DEFAULT_STICK_SHAPING_EXPONENT


def _shape_stick(value: float) -> float:
    """Map ``[-1, 1]`` post-deadzone stick deflection to ``[-1, 1]`` velocity.

    Power curve with sign preserved: ``sign(v) * |v|**exp`` where ``exp``
    is the runtime-tuned ``_RUNTIME_STICK_SHAPING_EXPONENT``. Pure helper
    -- the IntentDecoder owns deadzone clamping, so values arrive here
    already in ``[-1, 1]`` (or 0 if inside the deadzone).
    """
    sign = 1.0 if value >= 0 else -1.0
    mag = abs(float(value))
    if mag == 0.0:
        return 0.0
    return sign * mag ** _RUNTIME_STICK_SHAPING_EXPONENT


def _resolve_locomotion_continuous(
    stick_fwd: float,
    stick_side: float,
    stick_yaw: float,
) -> tuple[float, float, float, float]:
    """Map continuous stick deflections to a 4-D velocity vector.

    Forward / backward use different peak speeds (``_WALK_SPEED_MPS``
    vs ``_BACK_SPEED_MPS``) because the model's training data has
    asymmetric forward / backward stride coverage. Lateral and turn
    axes are symmetric. Returns the ``(yaw_rate, vel_x, vel_z, hip_h)``
    tuple in the same convention as ``_BASE_VELOCITY``.

    The runtime-tuning scalars (``_RUNTIME_FORWARD_SCALE`` etc.) are
    intentionally NOT applied here -- ``_apply_runtime_scales`` handles
    them downstream, mirroring the bucketed path so a single
    ``--kplanner-forward-scale 0.6`` override caps both modes.
    """
    shaped_fwd  = _shape_stick(stick_fwd)
    shaped_side = _shape_stick(stick_side)
    shaped_yaw  = _shape_stick(stick_yaw)
    vel_z = (
        shaped_fwd * _WALK_SPEED_MPS
        if shaped_fwd >= 0
        else shaped_fwd * _BACK_SPEED_MPS
    )
    # ``stick_side > 0`` (L-stick right, lx > 0) -> side_right ->
    # negative vel_x, matching ``_BASE_VELOCITY['side_right']``.
    vel_x = -shaped_side * _SIDE_SPEED_MPS
    # ``stick_yaw > 0`` (R-stick right) -> turn-right -> negative
    # yaw_rate, matching ``_BASE_VELOCITY['turn_right']``.
    #
    # Continuous mode uses its own yaw ceiling
    # (``_CONTINUOUS_TURN_MAX_RAD_S``) rather than the bucketed
    # ``_TURN_45_RAD_S`` constant -- the bucketed callers want a
    # sharp pivot from a single button press, but the analog R-stick
    # wants gentler resolution. See the constant's comment block for
    # the full rationale (model is trained on a no-yaw clip; high
    # yaw_rate is OOD). The mutable global is set from ``run()`` per
    # CLI flag / env var; this read picks up any override applied
    # before the dispatcher fires.
    yaw_rate = -shaped_yaw * _CONTINUOUS_TURN_MAX_RAD_S
    return (yaw_rate, vel_x, vel_z, _HIP_HEIGHT_M)


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

    Velocity tuple layout: ``(yaw_rate, vel_x=lateral, vel_z=forward,
    hip_h)`` -- matches ``_BASE_VELOCITY`` after the 2026-05-29
    channel-swap bugfix. Prior to that fix this helper scaled the
    wrong axis for ``--kplanner-{forward,backward,lateral}-scale``
    (the pre-fix layout had forward in ``vel_x`` and lateral in
    ``vel_z``); the swap is now applied here too so the CLI knobs
    target the right channel.
    """
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


def intent_to_velocity(cmd: LocomotionCommand) -> tuple[float, float, float, float]:
    """Translate ``LocomotionCommand`` -> 4-D velocity ``(yaw_rate, vx, vy, hip_h)``.

    Returns ``_IDLE_INTENT`` for any intent the kplanner has no velocity
    meaning for (``hold_torso``, ``lean_*``, ``torso_*``, ``crouch``,
    unrecognised intents). Logs a single DEBUG line per miss so missing
    vocabulary additions are visible in the planner log.

    Dispatch order (first match wins):

    1. ``cmd.direct_velocity is not None`` -> the recorded-velocity
       passthrough used by ``x2_pkl_command_source``. Returns the
       4-tuple verbatim with NO shaping and NO runtime scales applied,
       so the PKL replay path lands the exact (yaw, vel_x, vel_z,
       hip_h) extracted from the motion clip into the model. This
       isolates the kplanner -> deploy link from the analog-stick
       shaping curve and the operator's per-direction runtime scales.

    2. ``intent == "locomotion"`` paired with ``magnitude ==
       "continuous"`` -> analog Quest3 sticks. Reads ``cmd.stick_fwd /
       cmd.stick_side / cmd.stick_yaw`` and shapes them via
       ``_resolve_locomotion_continuous``. Runtime tuning scalars still
       apply post-shaping so ``--kplanner-forward-scale`` caps both
       bucketed and continuous modes consistently.

    3. Bucketed ``(intent, magnitude)`` lookup via ``_resolve_velocity``,
       with runtime tuning scalars applied via ``_apply_runtime_scales``.
    """
    if cmd.direct_velocity is not None:
        # PKL replay path: the source publishes a raw 4-D velocity
        # extracted from the recorded motion clip. We honour it
        # verbatim so the wire content matches what
        # _instant_intent_from_clip computed.
        yaw, vx, vz, hip_h = cmd.direct_velocity
        return (float(yaw), float(vx), float(vz), float(hip_h))
    if cmd.intent == "locomotion" and cmd.magnitude == "continuous":
        result = _resolve_locomotion_continuous(
            cmd.stick_fwd, cmd.stick_side, cmd.stick_yaw,
        )
        return _apply_continuous_runtime_scales(result)
    result = _resolve_velocity(cmd.intent, cmd.magnitude)
    if result == _IDLE_INTENT and cmd.intent != "idle":
        log.debug("intent %s,%s has no velocity mapping; idling",
                  cmd.intent, cmd.magnitude)
        return result
    return _apply_runtime_scales(cmd.intent, result)


def _apply_continuous_runtime_scales(
    velocity: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Per-axis runtime scaling for the continuous-locomotion path.

    Mirrors ``_apply_runtime_scales`` but routes on the sign of each
    velocity axis (forward vs backward, left turn vs right turn, etc.)
    instead of the (now-absent) intent name. Same scalars, same defaults
    -- a single ``--kplanner-forward-scale 0.6`` flag caps continuous
    and bucketed modes identically.
    """
    yaw, vx, vz, hip_h = velocity
    if yaw > 0:
        yaw *= _RUNTIME_TURN_LEFT_SCALE
    elif yaw < 0:
        yaw *= _RUNTIME_TURN_RIGHT_SCALE
    if vz > 0:
        vz *= _RUNTIME_FORWARD_SCALE
    elif vz < 0:
        vz *= _RUNTIME_BACKWARD_SCALE
    vx *= _RUNTIME_LATERAL_SCALE
    return (yaw, vx, vz, hip_h)


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
                # Continuous locomotion (analog Quest3 sticks) carries
                # three deadzone-rescaled deflections in [-1, 1]. Missing
                # fields default to 0.0 so legacy bucketed payloads still
                # work for the same wire format.
                stick_fwd  = float(payload.get("stick_fwd",  0.0))
                stick_side = float(payload.get("stick_side", 0.0))
                stick_yaw  = float(payload.get("stick_yaw",  0.0))
                # ``hold_torso`` (continuous waist target) passes through
                # but the kplanner's dispatcher idles on it (no upper-body
                # bins). We still parse the waist fields so future kplanner
                # extensions can consume them without changing the wire
                # format.
                waist_pitch_deg = float(payload.get("waist_pitch_deg", 0.0))
                waist_roll_deg  = float(payload.get("waist_roll_deg",  0.0))
                waist_yaw_deg   = float(payload.get("waist_yaw_deg",   0.0))
                # Optional raw 4-D velocity passthrough used by
                # ``x2_pkl_command_source`` for replaying recorded motion
                # clips through the planner -> deploy chain without the
                # bucketed table / continuous shaping / runtime scales
                # distorting the recorded velocity. When present this is
                # ``[yaw_rate, vel_x, vel_z, hip_h]``; the dispatcher
                # short-circuits to it (see ``intent_to_velocity``).
                # Missing field -> ``None`` -> dispatcher follows its
                # normal bucketed / continuous path.
                target_velocity = payload.get("target_velocity")
                direct_velocity: Optional[tuple[float, float, float, float]] = None
                if target_velocity is not None:
                    if (
                        not isinstance(target_velocity, (list, tuple))
                        or len(target_velocity) != 4
                    ):
                        log.warning(
                            "zmq command source: target_velocity must be a "
                            "4-element list [yaw_rate, vel_x, vel_z, hip_h]; "
                            "got %r (ignoring)",
                            target_velocity,
                        )
                    else:
                        direct_velocity = (
                            float(target_velocity[0]),
                            float(target_velocity[1]),
                            float(target_velocity[2]),
                            float(target_velocity[3]),
                        )
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("zmq command source: bad payload %r: %s", parts[1], exc)
                continue
            cmd_queue.put(
                LocomotionCommand(
                    intent=intent,
                    magnitude=magnitude,
                    source="zmq",
                    waist_pitch_deg=waist_pitch_deg,
                    waist_roll_deg=waist_roll_deg,
                    waist_yaw_deg=waist_yaw_deg,
                    stick_fwd=stick_fwd,
                    stick_side=stick_side,
                    stick_yaw=stick_yaw,
                    direct_velocity=direct_velocity,
                )
            )
    finally:
        sock.close(linger=0)


# ---------------------------------------------------------------------------
# Closed-loop pose feedback. The kplanner's chain of self-conditioned
# replans accumulates yaw drift (see "Deploy-integration diagnostics" in
# motionbricks/docs/x2_kplanner_evaluation.md). To break the loop we
# subscribe to the sim bridge's ``robot_pose`` topic and, just before
# each replan, overwrite the root rows the planner is about to read as
# context with the robot's *actually observed* pelvis pose.
#
# Joint slots [7:] stay model-predicted on purpose: the policy's
# joint-level tracking error would inject high-frequency noise into the
# context if those were reseeded too. We only reseed the root because:
#   1. It is the only channel where compounding drift was measured.
#   2. Pelvis qpos is what robot_pose actually exposes.
#   3. Joint reseed would require a separate observability path.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PoseObservation:
    """One ``robot_pose`` sample.

    ``pelvis_qpos_wxyz`` matches the bridge's wire format:
    ``[x, y, z, qw, qx, qy, qz]`` (length 7), the same wxyz quat
    convention NeuralPlannerCore stores in ``frames['mujoco_qpos']``.

    ``t_mono`` is ``time.monotonic()`` at receive time. Used for staleness
    checks so a frozen bridge doesn't keep feeding ancient observations
    into the planner forever.
    """

    t_mono: float
    pelvis_qpos_wxyz: np.ndarray  # shape (7,)


def _pose_feedback_thread(
    pose_deque: "collections.deque[PoseObservation]",
    pose_lock: threading.Lock,
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
) -> None:
    """SUB on ``robot_pose:port`` and append observations to ``pose_deque``.

    Drains as fast as the bridge publishes (~50 Hz). Deque has a maxlen
    upstream so we don't grow unbounded if the worker stops draining.
    """
    import zmq
    from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    log.info("pose feedback source: SUB %r on tcp://%s:%d", topic, host, port)
    warn_once = False
    msg_count = 0
    try:
        while not stop_event.is_set():
            try:
                raw = sock.recv()
            except zmq.error.Again:
                continue
            try:
                payload = unpack_robot_pose(raw)
                qpos = payload.get("pelvis_qpos_wxyz")
                if not isinstance(qpos, list) or len(qpos) != 7:
                    continue
                obs = PoseObservation(
                    t_mono=time.monotonic(),
                    pelvis_qpos_wxyz=np.asarray(qpos, dtype=np.float32),
                )
                with pose_lock:
                    pose_deque.append(obs)
                msg_count += 1
                if msg_count == 1:
                    log.info(
                        "pose feedback: first observation received "
                        "(xyz=%.3f,%.3f,%.3f qw=%.3f)",
                        float(qpos[0]), float(qpos[1]), float(qpos[2]),
                        float(qpos[3]),
                    )
            except Exception as exc:
                if not warn_once:
                    log.warning("pose feedback: decode error %s (suppressed)", exc)
                    warn_once = True
    finally:
        sock.close(linger=0)


_RESEED_SCOPE_FULL_ROOT = "full_root"
_RESEED_SCOPE_QUAT_ONLY = "quat_only"
_VALID_RESEED_SCOPES = (_RESEED_SCOPE_FULL_ROOT, _RESEED_SCOPE_QUAT_ONLY)


def _reseed_root_from_observations(
    planner_core,
    pose_deque: "collections.deque[PoseObservation]",
    pose_lock: threading.Lock,
    max_age_s: float,
    scope: str = _RESEED_SCOPE_FULL_ROOT,
) -> Optional[str]:
    """Overwrite the root rows the planner is about to read as context.

    Mirrors the index math in
    ``NeuralPlannerCore.get_context_mujoco_qpos`` so the context indices
    computed here match the indices that ``replan_with_velocity`` will
    read on its next call. Writes to ``frames['mujoco_qpos'][0, ctx_i,
    0:7]`` only -- joints (slots 7:) stay model-predicted.

    Caller MUST hold ``replan_lock`` because we mutate the in-place
    tensor that the next ``predict()`` reads.

    **Sampling strategy.** The model's 4-frame context window is
    *trained* at 30 fps spacing (1/30 = 33 ms between frames). The
    bridge's ``robot_pose`` stream runs at 50 Hz (20 ms spacing). Sampling
    "the last 4 observations" would feed the model a 60 ms window
    that it would interpret as a 100 ms window (with ~1.66x inflated
    implied velocity) -- and worse, near the buffer tail the planner's
    context indices collapse to ``[idx, last, last, last]`` so the
    second-through-fourth writes simply overwrite each other, giving
    the model a "robot teleported then froze" context that nothing in
    training resembles.

    Instead we walk backwards from "now" in 1/fps increments and pick
    the deque entry closest to each target timestamp. For duplicate
    context indices (common at the buffer tail) we let the per-slot
    selection naturally collapse to the same observation -- the model
    sees coherent "the robot is here, walking at the trained fps"
    context.

    Returns ``None`` on success, or a short skip-reason string.
    """
    import torch

    buf = planner_core.frames.get("mujoco_qpos")
    if buf is None:
        return "buffer_uninitialized"

    n_ft = int(planner_core.NUM_FRAMES_PER_TOKEN)

    with pose_lock:
        obs_list = list(pose_deque)

    if len(obs_list) < n_ft:
        return f"insufficient_obs ({len(obs_list)}/{n_ft})"

    newest = obs_list[-1]
    newest_age = time.monotonic() - newest.t_mono
    if newest_age > max_age_s:
        return f"stale_obs (age={newest_age:.3f}s > {max_age_s:.3f}s)"

    model_fps = float(getattr(planner_core, "fps", 30.0) or 30.0)
    target_spacing_s = 1.0 / model_fps

    target_times = [
        newest.t_mono - (n_ft - 1 - k) * target_spacing_s
        for k in range(n_ft)
    ]
    selected: list[PoseObservation] = []
    for tt in target_times:
        best = min(obs_list, key=lambda o, _tt=tt: abs(o.t_mono - _tt))
        selected.append(best)

    idx = int(planner_core._current_frame_idx)
    pred_off = int(planner_core.PRED_OFFSETS)
    last_idx = int(buf.shape[1]) - 1
    context_indices = [
        max(0, min(idx - n_ft + i + pred_off, last_idx))
        for i in range(n_ft)
    ]

    if scope not in _VALID_RESEED_SCOPES:
        raise ValueError(
            f"unknown reseed scope {scope!r}; want one of {_VALID_RESEED_SCOPES}"
        )

    device = buf.device
    dtype = buf.dtype
    # ``full_root`` rewrites xyz + quat; the open-loop diagnostic that
    # motivated this work showed the model's INTERNAL xy prediction
    # actually overshoots in a way that helps the deploy track forward
    # (the policy chases a slightly-ahead reference). Pinning xy to
    # the deploy's observed position removes that lure and forward
    # tracking regresses. ``quat_only`` preserves the helpful xy
    # overshoot while still anchoring the planner's heading to
    # observed reality.
    for ctx_i, obs in zip(context_indices, selected):
        obs_t = torch.from_numpy(np.asarray(obs.pelvis_qpos_wxyz, dtype=np.float64)).to(
            device=device, dtype=dtype,
        )
        if scope == _RESEED_SCOPE_FULL_ROOT:
            buf[0, ctx_i, 0:7] = obs_t
        else:
            buf[0, ctx_i, 3:7] = obs_t[3:7]

    return None


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


# Default time constant for the cold-start velocity ramp (s). See
# ``_ColdStartVelocityRamp`` for the full rationale; 0.0 disables the
# ramp (the worker passes intent through verbatim, matching the
# pre-2026-05-30 behavior). Operators tune this via the kplanner CLI
# (``--cold-start-ramp-tau-s``) or the Quest 3 / PKL wrappers'
# ``KPLANNER_COLD_START_RAMP_TAU_S`` env var.
_DEFAULT_COLD_START_RAMP_TAU_S: float = 0.20


class _ColdStartVelocityRamp:
    """Per-channel EWMA velocity ramp to smooth idle -> playing transitions.

    The kplanner's neural root model trains exclusively on steady-state
    walking loops (e.g. ``Loop_Forward_Walk_001__A018``); it never sees
    a static-stand -> walking transition. When the operator's L-stick
    first crosses the deadzone, the state machine fires IDLE_LOOP ->
    PLAYING which:

    1. calls ``planner_core.reset(warm)``, wiping the 4-frame context
       ring buffer and refilling it with 4 identical static-stand
       poses;
    2. forwards the raw operator target (e.g. ``vel_z = +0.5 m/s``)
       to ``replan_with_velocity``.

    ``NeuralPlannerCore`` then builds an implied target position at
    ``current_xy + vel * TARGET_HORIZON_S`` (= ``current_xy + 0.5 *
    2.13 = current_xy + 1.07 m`` in this example), meaning the model is
    asked to project from "4 frames of static stand" to "1 m ahead in
    2.13 s" in a single replan. Because the model has no training
    coverage for that regime, the highest-likelihood prediction
    channel is **pelvis x-translation rather than leg swing**; the
    deploy policy faithfully tracks that and the operator sees
    "torso bends forward, no step". After ~2-4 replans the buffer is
    filled with the model's own (now non-static) predictions and the
    gait stabilises.

    The ramp keeps (yaw_rate, vel_x, vel_z) close to zero for the
    first few replans after PLAYING entry, so the implied target stays
    close to the robot's actual position and the model emits gentler
    forward motion the policy can step into. ``hip_h`` (channel 3) is
    NOT ramped -- it's a posture target, not a velocity, and the
    model needs the correct walking pelvis height from frame 1.

    Time constant defaults to 0.2 s. At a 200 ms replan period
    (``--replan-threshold-frames 2`` + 30 FPS output) that yields
    alpha ~= 0.5, reaching ~95% of the operator's target in ~3
    replans (600 ms). Operators wanting snappier acceleration can
    drop this; ``tau = 0`` disables the ramp entirely (pre-fix
    behaviour).

    Deceleration is handled by the IDLE gate, not by this ramp:
    releasing the stick makes ``intent_to_velocity`` resolve to
    ``_IDLE_INTENT``, the state machine fires PLAYING -> IDLE_LOOP,
    and the publisher freezes the anchor pose immediately. The
    ramper's idle-detection (``last_was_idle``) just makes sure the
    NEXT idle->playing transition starts the ramp from zero again.
    """

    def __init__(self, tau_s: float = _DEFAULT_COLD_START_RAMP_TAU_S) -> None:
        self.tau_s = float(tau_s)
        # (yaw_rate, vel_x, vel_z) -- hip_h passed through verbatim.
        self._smoothed = np.zeros(3, dtype=np.float64)
        self._last_was_idle = True

    @property
    def enabled(self) -> bool:
        return self.tau_s > 0.0

    def step(
        self,
        target: tuple[float, float, float, float],
        dt_s: float,
    ) -> tuple[float, float, float, float]:
        """Advance the ramp by ``dt_s`` and return the smoothed target.

        When the previous tick was idle (== the operator just started
        pushing the stick or we're in the very first PLAYING entry)
        the smoothed state is reset to zero so the new push gets the
        full ramp. ``hip_h`` is forwarded verbatim from ``target``.
        """
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
        return (
            float(self._smoothed[0]),
            float(self._smoothed[1]),
            float(self._smoothed[2]),
            float(hip),
        )

    def reset_idle(self) -> None:
        """Mark the next ``step()`` as a fresh idle->playing entry."""
        self._smoothed.fill(0.0)
        self._last_was_idle = True


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
    pose_deque: "Optional[collections.deque[PoseObservation]]" = None,
    pose_lock: Optional[threading.Lock] = None,
    pose_max_age_s: float = 0.5,
    pose_reseed_scope: str = _RESEED_SCOPE_FULL_ROOT,
    cold_start_ramp_tau_s: float = _DEFAULT_COLD_START_RAMP_TAU_S,
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

    Closed-loop pose reseed: if ``pose_deque`` is provided, just before
    each replan we overwrite the 4 root rows the model will read as
    context with the robot's actually-observed pelvis qpos. Breaks the
    chain of self-conditioned predictions that compounds yaw drift.
    """
    log.info("planner worker thread started")
    feedback_enabled = pose_deque is not None and pose_lock is not None
    if feedback_enabled:
        log.info(
            "planner worker: closed-loop pose reseed ENABLED "
            "(scope=%s, max_age=%.3fs)",
            pose_reseed_scope, pose_max_age_s,
        )
    else:
        log.info("planner worker: closed-loop pose reseed DISABLED (open-loop)")

    cold_start_ramp = _ColdStartVelocityRamp(tau_s=cold_start_ramp_tau_s)
    if cold_start_ramp.enabled:
        log.info(
            "planner worker: cold-start velocity ramp ENABLED (tau=%.3fs); "
            "applies EWMA to (yaw_rate, vel_x, vel_z) channels on every "
            "idle -> playing entry so the model's implied target stays "
            "close to current pose for the first 2-3 replans",
            cold_start_ramp.tau_s,
        )
    else:
        log.info(
            "planner worker: cold-start velocity ramp DISABLED (tau=0); "
            "raw operator intent is forwarded verbatim (pre-fix behaviour)"
        )
    last_replan_mono: Optional[float] = None

    reseed_stats = {
        "applied": 0,
        "skipped_insufficient": 0,
        "skipped_stale": 0,
        "skipped_buffer_uninit": 0,
        "skipped_other": 0,
    }
    stats_log_every = 50

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
            #
            # Mark the cold-start ramp idle so the NEXT non-idle replan
            # starts ramping from zero again. Without this the ramp's
            # smoothed state would persist across idle gaps and the
            # operator's "release then re-push" pattern would skip the
            # ramp on the second push -- which IS what we want for
            # brief blips through idle, but breaks for sustained idle
            # where the buffer drifts away from walking context. The
            # idle gate in the worker doesn't track time-in-idle, so
            # we conservatively reset on every idle tick: a 1-tick
            # idle blip costs <50 ms of ramp time on resumption.
            cold_start_ramp.reset_idle()
            last_replan_mono = None
            continue

        # Cold-start velocity ramp: smooth (yaw_rate, vel_x, vel_z) on
        # idle -> playing transitions so the model's implied target
        # (= current_xy + vel * 2.13 s) doesn't jump 1 m+ ahead while
        # the context buffer still holds 4 frames of static stand
        # pose. hip_h (channel 3) passes through verbatim. See
        # ``_ColdStartVelocityRamp`` docstring for the full mechanism.
        now_mono = time.monotonic()
        if last_replan_mono is None:
            dt_s = 1.0 / OUTPUT_FPS  # first replan in this PLAYING segment
        else:
            dt_s = max(1e-3, now_mono - last_replan_mono)
        smoothed_target = cold_start_ramp.step(tuple(target), dt_s)
        last_replan_mono = now_mono
        if smoothed_target != tuple(target) and log.isEnabledFor(logging.DEBUG):
            log.debug(
                "worker: cold-start ramp dt=%.3fs raw=%s -> smoothed=%s",
                dt_s, tuple(target), smoothed_target,
            )
        # Use the smoothed target from here on -- the model and the
        # closed-loop reseed both operate on the same vector.
        target = smoothed_target

        if feedback_enabled:
            with replan_lock:
                reason = _reseed_root_from_observations(
                    planner_core, pose_deque, pose_lock, pose_max_age_s,
                    scope=pose_reseed_scope,
                )
            if reason is None:
                reseed_stats["applied"] += 1
            elif reason.startswith("insufficient"):
                reseed_stats["skipped_insufficient"] += 1
            elif reason.startswith("stale"):
                reseed_stats["skipped_stale"] += 1
            elif reason.startswith("buffer_uninit"):
                reseed_stats["skipped_buffer_uninit"] += 1
            else:
                reseed_stats["skipped_other"] += 1
            total = sum(reseed_stats.values())
            if total > 0 and total % stats_log_every == 0:
                log.info(
                    "reseed stats: total=%d applied=%d "
                    "insufficient=%d stale=%d buf_uninit=%d other=%d",
                    total,
                    reseed_stats["applied"],
                    reseed_stats["skipped_insufficient"],
                    reseed_stats["skipped_stale"],
                    reseed_stats["skipped_buffer_uninit"],
                    reseed_stats["skipped_other"],
                )

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
    stick_shape_exp: float = _DEFAULT_STICK_SHAPING_EXPONENT,
    pose_feedback_host: Optional[str] = None,
    pose_feedback_port: Optional[int] = None,
    pose_feedback_topic: str = "robot_pose",
    pose_feedback_max_age_s: float = 0.5,
    pose_feedback_deque_maxlen: int = 32,
    pose_reseed_scope: str = _RESEED_SCOPE_FULL_ROOT,
    cold_start_ramp_tau_s: float = _DEFAULT_COLD_START_RAMP_TAU_S,
    continuous_turn_max_rad_s: float = _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S,
) -> int:
    _setup_logging(verbose)

    # Stash runtime tuning scales on the module-level mutable singletons
    # used by ``intent_to_velocity``. Defaults are 1.0 so the static
    # unit-test invariants still hold when the user passes no override.
    global _RUNTIME_TURN_LEFT_SCALE, _RUNTIME_TURN_RIGHT_SCALE
    global _RUNTIME_FORWARD_SCALE, _RUNTIME_BACKWARD_SCALE, _RUNTIME_LATERAL_SCALE
    global _RUNTIME_STICK_SHAPING_EXPONENT
    global _CONTINUOUS_TURN_MAX_RAD_S
    _RUNTIME_TURN_LEFT_SCALE = float(turn_left_scale)
    _RUNTIME_TURN_RIGHT_SCALE = float(turn_right_scale)
    _RUNTIME_FORWARD_SCALE = float(forward_scale)
    _RUNTIME_BACKWARD_SCALE = float(backward_scale)
    _RUNTIME_LATERAL_SCALE = float(lateral_scale)
    if stick_shape_exp <= 0:
        log.error("--stick-shape-exp must be > 0 (got %s); using default %.2f",
                  stick_shape_exp, _DEFAULT_STICK_SHAPING_EXPONENT)
        stick_shape_exp = _DEFAULT_STICK_SHAPING_EXPONENT
    _RUNTIME_STICK_SHAPING_EXPONENT = float(stick_shape_exp)
    if continuous_turn_max_rad_s <= 0:
        log.error(
            "--continuous-turn-max-rad-s must be > 0 (got %s); using "
            "default %.3f rad/s",
            continuous_turn_max_rad_s, _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S,
        )
        continuous_turn_max_rad_s = _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S
    _CONTINUOUS_TURN_MAX_RAD_S = float(continuous_turn_max_rad_s)
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
    log.info("continuous-locomotion stick shape exponent: %.3f "
             "(1.0=linear, <1 closer-to-bucketed, >1 more deadzone-feel)",
             _RUNTIME_STICK_SHAPING_EXPONENT)
    log.info(
        "continuous-locomotion yaw ceiling: %.3f rad/s (~%.1f deg/s, "
        "90-deg turn in %.2f s); bucketed turn_*/deg_45 unchanged at "
        "%.3f rad/s",
        _CONTINUOUS_TURN_MAX_RAD_S,
        math.degrees(_CONTINUOUS_TURN_MAX_RAD_S),
        (math.pi / 2.0) / max(_CONTINUOUS_TURN_MAX_RAD_S, 1e-9),
        _TURN_45_RAD_S,
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

    pose_deque: "Optional[collections.deque[PoseObservation]]" = None
    pose_lock: Optional[threading.Lock] = None
    if pose_feedback_host is not None and pose_feedback_port is not None:
        pose_deque = collections.deque(maxlen=int(pose_feedback_deque_maxlen))
        pose_lock = threading.Lock()
        log.info(
            "closed-loop pose reseed: subscribing to %r on tcp://%s:%d "
            "(maxlen=%d, max_age=%.3fs)",
            pose_feedback_topic, pose_feedback_host, pose_feedback_port,
            int(pose_feedback_deque_maxlen), float(pose_feedback_max_age_s),
        )
        thr = threading.Thread(
            target=_pose_feedback_thread,
            args=(
                pose_deque, pose_lock,
                pose_feedback_host, pose_feedback_port, pose_feedback_topic,
                stop_event,
            ),
            name="pose-feedback",
            daemon=True,
        )
        thr.start()
        threads.append(thr)
    else:
        log.info(
            "closed-loop pose reseed: DISABLED "
            "(no --pose-feedback-host/port) -- running open-loop"
        )

    worker_thread = threading.Thread(
        target=_planner_worker,
        args=(planner_core, intent_state, replan_lock, stop_event, replan_event),
        kwargs={
            "pose_deque": pose_deque,
            "pose_lock": pose_lock,
            "pose_max_age_s": float(pose_feedback_max_age_s),
            "pose_reseed_scope": pose_reseed_scope,
            "cold_start_ramp_tau_s": float(cold_start_ramp_tau_s),
        },
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
    tune_grp.add_argument(
        "--stick-shape-exp", type=float,
        default=_DEFAULT_STICK_SHAPING_EXPONENT,
        help=(
            "Power-curve exponent applied to ``locomotion / continuous`` "
            "stick deflections before scaling by the base velocity. "
            "1.0 (default) = linear; >1.0 (e.g. 2.0) = more dead near zero, "
            "harder to reach full speed; <1.0 (e.g. 0.5) = closer to the "
            "bucketed feel (50%% stick already at 71%% velocity)."
        ),
    )
    fb_grp = p.add_argument_group(
        "closed-loop pose feedback",
        "Break the kplanner's chain of self-conditioned predictions by "
        "refreshing the context window from the robot's actually-observed "
        "pelvis pose before each replan. See 'Deploy-integration diagnostics' "
        "in motionbricks/docs/x2_kplanner_evaluation.md for the why.",
    )
    fb_grp.add_argument(
        "--pose-feedback-host", default=None,
        help=(
            "Sim bridge host that publishes the 'robot_pose' topic. "
            "When set together with --pose-feedback-port enables closed-loop "
            "pose reseed; default (None) keeps the open-loop behaviour."
        ),
    )
    fb_grp.add_argument(
        "--pose-feedback-port", type=int, default=None,
        help="ZMQ PUB port for 'robot_pose' (sim bridge default: 5570).",
    )
    fb_grp.add_argument(
        "--pose-feedback-topic", default="robot_pose",
        help="Topic name on the pose-feedback SUB socket. Default 'robot_pose'.",
    )
    fb_grp.add_argument(
        "--pose-feedback-max-age-s", type=float, default=0.5,
        help=(
            "If the newest pose observation is older than this many seconds "
            "we skip the reseed for this replan and let the planner fall "
            "back to its own predictions. Default 0.5 s (15 frames @ 30 Hz)."
        ),
    )
    fb_grp.add_argument(
        "--pose-feedback-deque-maxlen", type=int, default=32,
        help=(
            "Capacity of the rolling buffer of observations the feedback "
            "thread fills. Each replan reads the latest 4 entries. Default "
            "32 = ~640 ms of headroom at the bridge's 50 Hz publish rate."
        ),
    )
    fb_grp.add_argument(
        "--pose-reseed-scope",
        choices=list(_VALID_RESEED_SCOPES),
        default=_RESEED_SCOPE_FULL_ROOT,
        help=(
            "Which root channels the reseed rewrites. 'full_root' (default) "
            "overwrites xyz + quat (4 root rows, 7 floats each). 'quat_only' "
            "overwrites just the quaternion -- preserves the planner's "
            "internal-model xy overshoot (which empirically helps the policy "
            "track forward motion) while still anchoring heading to observed "
            "reality. Use 'quat_only' when forward tracking regresses under "
            "'full_root'."
        ),
    )
    p.add_argument(
        "--continuous-turn-max-rad-s",
        type=float,
        default=_DEFAULT_CONTINUOUS_TURN_MAX_RAD_S,
        help=(
            "Yaw-rate ceiling (rad/s) at full R-stick deflection in the "
            "continuous-locomotion path. Default %.3f rad/s (~43 deg/s, a "
            "90-deg turn in ~2.1 s). The bucketed path keeps its own "
            "_TURN_45_RAD_S = 1.5 rad/s ceiling -- this knob only affects "
            "Quest 3 R-stick X analog turns. The per-side runtime turn "
            "scales (--turn-left-scale / --turn-right-scale) still apply "
            "on top of this ceiling for L/R asymmetry compensation."
        ) % _DEFAULT_CONTINUOUS_TURN_MAX_RAD_S,
    )
    p.add_argument(
        "--cold-start-ramp-tau-s",
        type=float,
        default=_DEFAULT_COLD_START_RAMP_TAU_S,
        help=(
            "Time constant (s) for the cold-start velocity ramp applied on "
            "every idle -> playing transition. Smooths (yaw_rate, vel_x, "
            "vel_z) via per-channel EWMA so the model's implied 2.13 s target "
            "doesn't jump 1 m+ ahead while the context buffer still holds 4 "
            "frames of static stand pose. Default 0.20 s ~= 95%% of operator "
            "target after ~3 replans at threshold=2. Set 0.0 to disable the "
            "ramp (raw intent verbatim; pre-fix behaviour). hip_h is never "
            "ramped -- it's a posture target, not a velocity."
        ),
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
        stick_shape_exp=args.stick_shape_exp,
        pose_feedback_host=args.pose_feedback_host,
        pose_feedback_port=args.pose_feedback_port,
        pose_feedback_topic=args.pose_feedback_topic,
        pose_feedback_max_age_s=args.pose_feedback_max_age_s,
        pose_feedback_deque_maxlen=args.pose_feedback_deque_maxlen,
        pose_reseed_scope=args.pose_reseed_scope,
        cold_start_ramp_tau_s=args.cold_start_ramp_tau_s,
        continuous_turn_max_rad_s=args.continuous_turn_max_rad_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
