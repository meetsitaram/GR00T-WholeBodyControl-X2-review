"""Live VLA → SONIC bridge for the X2 MuJoCo deploy (Stage C4 / M7).

This is the production-ish drop-in replacement for
:mod:`mock_vla_publish_stand_token`. It owns the full
"camera + state → VLA → motion-token chunk" loop expected by
``gear_sonic_deploy/deploy_x2.sh sim --vla``.

Architecture
------------

::

    ┌────────────────────┐                 ┌──────────────────────────┐
    │ x2_deploy_onnx_ref │                 │ live_vla_publish_motion_ │
    │ (C++, in container)│  x2_debug PUB   │ token.py (this script)   │
    │                    ├────tcp:5557────►│                          │
    │  MuJoCo + SONIC    │                 │  ┌────────────────────┐  │
    │   tracking policy  │                 │  │ ghost MuJoCo       │  │
    │                    │  pose SUB       │  │ renderer           │  │
    │                    ◄────tcp:5556─────┤  └────────────────────┘  │
    │                    │                 │  ┌────────────────────┐  │
    │                    │                 │  │ Isaac-GR00T N1.7   │  │
    │                    │                 │  │ Gr00tPolicy        │  │
    └────────────────────┘                 │  └────────────────────┘  │
                                           └──────────────────────────┘

Design notes
------------

* The C++ deploy runs MuJoCo in-process. We don't have direct access to its
  rendered ego camera, but the deploy publishes ``body_q`` (31 DOFs in
  MuJoCo joint order) + ``base_quat`` (wxyz) + ``left/right_hand_q`` over
  ``x2_debug``. We mirror that into a *passive* (kinematic) MuJoCo via
  :class:`gear_sonic.scripts.render_smoketest_episode_video.MujocoFrameRenderer`
  and re-render the ``ego_view`` camera. This is the same frame source
  that produced the M5/M6 training data so the VLA sees in-distribution
  pixels.

* Inference latency on a 32 GB RTX 5090 is ~150–300 ms / chunk (much
  slower than the 50 Hz publish cadence the C++ deploy expects). We
  therefore decouple the two: a worker thread runs the VLA as fast as
  it can, posting fresh 40-step action chunks; the main thread is a
  steady 50 Hz publisher that walks the *current* chunk timestep by
  timestep, rolling over to the freshest available chunk whenever the
  worker finishes another inference.

* **Planner-shaped safety on the wire:** the publisher always emits the
  trained ``DEFAULT_STAND_POSE`` as ``joint_pos_mj`` and identity root
  quat — that is the SONIC tracker setpoint, not a feedback channel.
  Mirroring the live ``body_q_mj`` from ``x2_debug`` back as the
  setpoint zeroes the tracking error and the robot falls under gravity
  (verified empirically 2026-05-14). The motion-token / hand slices
  come from the latest VLA chunk (zeros until the first inference, or
  always zero under ``--no-policy``). After a render or policy failure
  the inference thread posts a **zero-token** chunk so downstream never
  keeps consuming a bad latent while telemetry recovers.

* The state vector handed to ``Gr00tPolicy`` is in **Pinocchio URDF order**
  (matches ``meta/modality.json``: legs / waist / head / arms / hands),
  while ``body_q`` arriving over ``x2_debug`` is in **MuJoCo joint order**
  (legs / waist / arms / head). We pre-compute a ``MJ_TO_PIN`` permutation
  once at startup.

Acceptance gate (Stage C5)
--------------------------

Run the deploy under ``deploy_x2.sh sim --vla``, point this script at
the just-trained checkpoint, and dump telemetry with
``dump_x2_debug.py``. The token norms produced here will be non-zero
(unlike the mock publisher's stand-still latent) and the deploy should
remain upright + tilt-free for at least 10s of policy time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
from pathlib import Path
import signal
import sys
import threading
import time
import warnings
from enum import Enum
from typing import Any, Dict, Optional

# Silence the three cosmetic UserWarnings the vendored Isaac-GR00T
# ``image_augmentations`` module emits the first time each augmentation
# class is constructed: it passes ``always_apply=...`` to
# ``albumentations.BasicTransform.__init__``, which the installed
# albumentations release has deprecated. The warnings fire once per
# class on first instantiation -- three lines per bridge launch --
# and otherwise have no functional impact. Suppressing them keeps
# the bridge.log signal-to-noise sane (it's the first thing the
# operator sees scrolling past before the policy reports ready).
warnings.filterwarnings(
    "ignore",
    message=r".*'always_apply' are not valid for transform BasicTransform.*",
    category=UserWarning,
)

import joblib
import json
import numpy as np
import zmq

# Allow running this script without `pip install -e gear_sonic` --
# also pulls Isaac-GR00T onto sys.path so ``import gr00t`` resolves
# without the caller having to set PYTHONPATH manually.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ISAAC_GROOT_ROOT = REPO_ROOT / "external_dependencies" / "Isaac-GR00T"
if ISAAC_GROOT_ROOT.is_dir() and str(ISAAC_GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message  # noqa: E402


SONIC_MOTION_TOKEN_DIM: int = 64
DEFAULT_HAND_DOF: int = 10
DEFAULT_PUB_RATE_HZ: float = 50.0
NUM_BODY_DOFS: int = 31  # MuJoCo joint count
# Single-tick zero slices for planner-like idle wire (avoid per-tick alloc).
_ZERO_MOTION_TOKEN_STEP = np.zeros(SONIC_MOTION_TOKEN_DIM, dtype=np.float32)
_ZERO_HAND_STEP = np.zeros(DEFAULT_HAND_DOF, dtype=np.float32)
# v0 proprio placeholder for the bridge-side SONIC token decoder. The
# decoder is OOD with this input but still emits non-trivial actions
# (~0.30 rad RMSE per validate_encode_decode_loop.py); good enough to
# put dynamic body intent on the wire. A v1 follow-up would assemble a
# real 990-D proprio from x2_debug history (see
# gear_sonic/scripts/eval_x2_mujoco.py:493 ProprioceptionBuffer for the
# canonical layout).
_PROPRIO_ZERO_990 = np.zeros(990, dtype=np.float32)

# v5 future-window contract -- must mirror DT_FUTURE_REF / NUM_FUTURE_FRAMES
# in gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/zmq/zmq_pose_input_source.hpp
# (NUM_FUTURE_FRAMES = 10, k=0 == current frame, so we publish the 9
# strictly-future slots on the wire). Without this window the deploy's
# ZmqPoseInputSource falls back to the legacy single-frame Sample() path,
# which pins all 10 future tokens at the current pose -- the policy was
# trained with motion-bearing windows and produces saturated raw actions
# (act_clip ~100 % of ticks, max_pre_clip > action_clip) when fed the
# legacy fallback, which manifests as a slow gravity tilt after the
# elastic band releases. The heuristic planner emits the same window
# shape via :func:`gear_sonic.utils.planner.state_machine.build_pose_payload`,
# so promoting the bridge to v5 brings the wire to byte-parity with
# planner-only mode and unblocks --vla-no-policy / closed-loop runs.
_FUTURE_DT_S: float = 0.1
_NUM_FUTURE_SLOTS: int = 9
# step_ticks at 50 Hz so that adjacent future slots are DT_FUTURE_REF
# (0.1 s) apart -- matches ``planner.step_with_lookahead(step_ticks=5)``.
_FUTURE_STEP_TICKS: int = int(round(DEFAULT_PUB_RATE_HZ * _FUTURE_DT_S))
_FUTURE_DT_FIELD = np.array([_FUTURE_DT_S], dtype=np.float32)


# Stand-pose fallback (radians, MuJoCo body order). Mirrors the trained
# default in ``policy_parameters.hpp`` and the mock publisher.
DEFAULT_STAND_POSE_MUJOCO_RAD: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # left leg (6)
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # right leg (6)
    0.0, 0.0, 0.0,                                # waist (3)
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,           # left arm (7)
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,          # right arm (7)
    0.0, 0.0,                                     # head (2)
)
assert len(DEFAULT_STAND_POSE_MUJOCO_RAD) == NUM_BODY_DOFS

# MJ-order slot for ``waist_yaw_joint`` -- the dominant heading-correction
# effector. Matches ``gear_sonic.utils.planner.constants.WAIST_YAW_IDX``;
# duplicated here to avoid a planner-package import in the publisher hot
# path. Used by the partial-body freeze to surgically pin waist_yaw to
# the measured value while keeping the rest of the legs+waist DOFs on
# the idle_stand clip's training-distribution jitter (see section F).
WAIST_YAW_IDX: int = 12


# Permutation MJ_TO_PIN[i] = j means: the i-th joint in Pinocchio order
# is the j-th joint in MuJoCo order. The dataset's ``observation.state``
# is written in Pinocchio order, but the C++ deploy publishes ``body_q``
# in MuJoCo order. Both are fixed at build-time so we can hard-code the
# index list (verified against ``rm.joint_names`` and X2_BODY_JOINT_NAMES
# at module import time below).
MJ_TO_PIN: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5,           # left_leg  (pin 0..5  ← mj 0..5)
    6, 7, 8, 9, 10, 11,         # right_leg (pin 6..11 ← mj 6..11)
    12, 13, 14,                 # waist     (pin 12..14 ← mj 12..14)
    29, 30,                     # head      (pin 15..16 ← mj 29..30)
    15, 16, 17, 18, 19, 20, 21, # left arm  (pin 17..23 ← mj 15..21)
    22, 23, 24, 25, 26, 27, 28, # right arm (pin 24..30 ← mj 22..28)
)
assert len(MJ_TO_PIN) == NUM_BODY_DOFS

# SONIC / deploy expect a fixed 40-step horizon at 50 Hz unless the
# policy checkpoint changes modality horizons (we match the default chunk).
DEFAULT_ACTION_HORIZON: int = 40


def _identity_quat_xyzw() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def _root_quat_xyzw_from_base_quat_wxyz(base_quat_wxyz: np.ndarray) -> np.ndarray:
    """Yaw-only root quat for the wire, derived from live IMU base_quat.

    Mirrors :meth:`X2DatasetRecorder._compute_idle_root_quat_xyzw` so the
    deploy tokenizer's orientation reference matches the robot's current
    heading instead of the idle_stand clip's yaw=0 frame (which otherwise
    drives a ~45 deg waist-yaw correction when VLA connects mid-session).
    """
    from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw

    wxyz = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(-1)
    if wxyz.shape[0] < 4:
        return _identity_quat_xyzw()
    quat_xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64)
    yaw = float(yaw_of_quat_xyzw(quat_xyzw))
    half = 0.5 * yaw
    return np.array(
        [0.0, 0.0, math.sin(half), math.cos(half)],
        dtype=np.float32,
    )


def _tile_root_quat_future(quat_xyzw: np.ndarray) -> np.ndarray:
    """Broadcast a single xyzw quat across the 9-slot future window."""
    q = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
    return np.broadcast_to(q, (_NUM_FUTURE_SLOTS, 4)).copy()


def _default_stand_body_pose_f32() -> np.ndarray:
    return np.asarray(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32)


# Pre-allocated zero-motion future window. ``_idle_jpos_future`` is 9
# strictly-future copies of DEFAULT_STAND_POSE; ``_idle_quat_future`` is
# 9 identity quats; ``_idle_jvel_future`` is the matching zero-velocity
# slab (finite-diff over identical frames is zero, so we ship it
# explicitly to skip the deploy's backward-finite-diff path). Mutating
# the returned arrays is forbidden -- callers should copy if they need
# to write. We freeze them with WRITEABLE=False so accidental writes
# raise rather than silently corrupting future ticks.
def _make_idle_future_window() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    jpos = np.broadcast_to(
        np.asarray(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32),
        (_NUM_FUTURE_SLOTS, NUM_BODY_DOFS),
    ).copy()
    quat = np.broadcast_to(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        (_NUM_FUTURE_SLOTS, 4),
    ).copy()
    jvel = np.zeros((_NUM_FUTURE_SLOTS, NUM_BODY_DOFS), dtype=np.float32)
    for arr in (jpos, quat, jvel):
        arr.setflags(write=False)
    return jpos, quat, jvel


_IDLE_JPOS_FUTURE, _IDLE_QUAT_FUTURE, _IDLE_JVEL_FUTURE = _make_idle_future_window()


def _idle_future_payload_fields(*, base_frame_index: int) -> dict[str, np.ndarray]:
    """Return the v5 future-window subset of the wire payload.

    The deploy promotes a frame to v5 mode iff it carries BOTH
    ``joint_pos_mj_future`` and ``root_quat_xyzw_future`` (joint_vel is
    optional but we ship it to skip the deploy's backward-finite-diff
    path). ``frame_index_future`` is informational only -- the deploy
    does not gate v5 promotion on it -- but we keep it consistent so
    the parity tooling and dataset replays line up.
    """
    fidx_future = np.array(
        [base_frame_index + (k + 1) for k in range(_NUM_FUTURE_SLOTS)],
        dtype=np.int64,
    )
    return {
        "joint_pos_mj_future": _IDLE_JPOS_FUTURE,
        "root_quat_xyzw_future": _IDLE_QUAT_FUTURE,
        "joint_vel_mj_future": _IDLE_JVEL_FUTURE,
        "frame_index_future": fidx_future,
        "future_dt_s": _FUTURE_DT_FIELD,
    }


def _clamp_vector_deviation(
    target: np.ndarray,
    anchor: np.ndarray,
    max_dev: float,
) -> np.ndarray:
    """Pull ``target`` toward ``anchor`` if any joint exceeds ``max_dev`` (rad)."""
    if max_dev <= 0.0:
        return np.asarray(target, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    anc = np.asarray(anchor, dtype=np.float32)
    delta = tgt - anc
    peak = float(np.abs(delta).max())
    if peak <= max_dev:
        return tgt
    return (anc + delta * (max_dev / peak)).astype(np.float32)


def _clamp_vector_step(
    target: np.ndarray,
    prev: np.ndarray | None,
    max_step: float,
) -> np.ndarray:
    """Limit per-tick joint delta on the wire (rad)."""
    if max_step <= 0.0 or prev is None:
        return np.asarray(target, dtype=np.float32)
    tgt = np.asarray(target, dtype=np.float32)
    prv = np.asarray(prev, dtype=np.float32)
    delta = tgt - prv
    peak = float(np.abs(delta).max())
    if peak <= max_step:
        return tgt
    return (prv + delta * (max_step / peak)).astype(np.float32)


def _clamp_vector_step_per_joint(
    target: np.ndarray,
    prev: np.ndarray | None,
    max_step_per_joint: np.ndarray,
) -> np.ndarray:
    """Per-joint variant of :func:`_clamp_vector_step`.

    Unlike the scalar variant -- which scales the ENTIRE delta vector
    by ``max_step / peak`` (preserving direction) -- this variant
    clamps each joint INDEPENDENTLY: joint ``i`` can move at most
    ``max_step_per_joint[i]`` rad this tick, regardless of what the
    other joints are doing.

    Why per-joint independent: the tracking-feedback loop wants to
    slow down ONLY the joints that are actually lagging (e.g., the
    operator's wrist roll while the elbow tracks fine). Throttling
    the whole vector to the slowest joint would couple unrelated
    actuators and over-damp the wire when only one DOF is sluggish.
    The cost is that the wire's per-tick *direction* may shift
    slightly from the policy's intent when one joint is throttled --
    but that shift is exactly what a per-joint actuator with PID
    feedback would produce anyway, so it matches the physical reality
    the policy will encounter.

    Cap semantics (different from the scalar variant on purpose):

      * ``cap[i] > 0``: joint ``i``'s |delta| clamped to that value.
      * ``cap[i] == 0``: joint ``i`` is FROZEN this tick (delta = 0).
        This is what the tracking-feedback law emits when a joint's
        tracking error exceeds the hard threshold (actuator clearly
        saturating; back off until it catches up).
      * ``cap[i] < 0``: NO cap on joint ``i`` (delta passes through).
        Used as the "joint not subject to feedback" sentinel; the
        helper internally treats negatives as +inf.

    Note: the scalar variant uses ``max_step <= 0`` to mean "step cap
    DISABLED" (target passes through). The per-joint variant splits
    that into two cases (freeze vs no-cap) because tracking feedback
    needs to express both. If you want the scalar variant's
    "disabled" semantics here, pass an all-negative cap array.

    Args:
        target: desired wire pose this tick (e.g., post-LPF).
        prev:   previously published wire pose (anchor for the delta).
                ``None`` means "no anchor available yet" -- pass
                target through unchanged (same semantics as the
                scalar variant).
        max_step_per_joint: shape ``(N,)`` array of per-joint caps.
                            See cap semantics above.
    """
    tgt = np.asarray(target, dtype=np.float32)
    if prev is None:
        return tgt
    prv = np.asarray(prev, dtype=np.float32)
    cap = np.asarray(max_step_per_joint, dtype=np.float32)
    if cap.shape != tgt.shape:
        raise ValueError(
            f"max_step_per_joint shape {cap.shape} != target shape "
            f"{tgt.shape} (per-joint clamp needs one cap per joint)"
        )
    delta = tgt - prv
    # Negative cap = "no cap on this joint" (sentinel) -> +inf so
    # np.minimum is a no-op. Zero cap = "freeze" -> np.minimum with 0
    # forces |delta_i| = 0 (correct semantics for tracking-feedback
    # hard-threshold freeze).
    cap_eff = np.where(cap < 0.0, np.float32(np.inf), cap)
    # Element-wise clamp: |delta_i| <= cap_eff[i]. Preserves sign.
    sign = np.sign(delta).astype(np.float32)
    clamped = sign * np.minimum(np.abs(delta), cap_eff)
    return (prv + clamped).astype(np.float32)


# Joint-mask constants for the tracking-feedback loop. The bridge
# does NOT apply per-joint feedback to legs / waist / head: those
# DOFs have their own deploy-side dynamics (SONIC balance loop,
# waist_yaw freeze, head lock) that tracking feedback would fight
# with. Arms + hands are the only joints the VLA bridge directly
# authors during a chunk, so they're the only joints we gate.
_ARM_JOINT_INDICES: tuple[int, ...] = tuple(range(15, 29))  # MJ 15..28 = L_arm(7) + R_arm(7)


def _apply_tracking_feedback(
    target: np.ndarray,
    prev_wire: np.ndarray | None,
    measured_q: np.ndarray | None,
    measured_dq: np.ndarray | None,
    *,
    base_max_step: float,
    soft_rad: float,
    hard_rad: float,
    vel_margin: float,
    vel_floor_rad_tick: float,
    dt_s: float,
    joint_indices: tuple[int, ...] = _ARM_JOINT_INDICES,
) -> tuple[np.ndarray, int]:
    """Compute per-joint per-tick step caps from real proprio feedback.

    Returns ``(cap_per_joint, throttle_count)`` where:

      * ``cap_per_joint`` is a length-``len(target)`` array of step
        caps (rad/tick). Joints NOT in ``joint_indices`` are filled
        with ``base_max_step`` (i.e., no per-joint feedback applied;
        the existing static behaviour). Joints IN ``joint_indices``
        get the minimum of:

          - Position-error backoff: ``base_max_step * f(|tgt - meas|)``
            where ``f`` is 1.0 below ``soft_rad``, 0.0 above
            ``hard_rad``, linear in between. Above hard the wire
            freezes for that joint until the actuator catches up.
          - Velocity cap: ``max(vel_floor, vel_margin * |dq| * dt)``.
            Prevents commanding faster than the actuator is currently
            moving (with a small overhead via ``vel_margin``) but
            allows starts from rest via ``vel_floor``.

      * ``throttle_count`` is the number of ``joint_indices`` joints
        whose effective cap dropped below ``0.5 * base_max_step``
        (used for telemetry; "how many joints did feedback throttle
        this tick").

    Fallbacks (all preserve byte-identical behaviour to the static
    scalar clamp):

      * ``prev_wire is None``: returns ``cap = base_max_step``
        everywhere, ``throttle_count = 0``. Caller's scalar clamp
        will also no-op when ``prev is None`` (per
        ``_clamp_vector_step`` contract), so the wire is unaffected
        anyway -- this just keeps the return value consistent.
      * ``measured_q is None`` OR ``measured_dq is None``: same as
        above, signals "no proprio available, fall back to scalar".
      * ``base_max_step <= 0``: returns ``cap = 0`` everywhere
        (matches scalar's "step cap disabled = freeze" semantics).
    """
    n = int(np.asarray(target).shape[0])
    if base_max_step <= 0.0:
        return np.zeros(n, dtype=np.float32), 0
    base = float(base_max_step)
    if (
        prev_wire is None
        or measured_q is None
        or measured_dq is None
    ):
        return np.full(n, base, dtype=np.float32), 0

    tgt = np.asarray(target, dtype=np.float32)
    meas_q = np.asarray(measured_q, dtype=np.float32)
    meas_dq = np.asarray(measured_dq, dtype=np.float32)
    if meas_q.shape[0] != n or meas_dq.shape[0] != n:
        # Shape mismatch -> fall back to scalar (do not silently misalign).
        return np.full(n, base, dtype=np.float32), 0

    cap = np.full(n, base, dtype=np.float32)
    throttle_count = 0

    soft = float(max(soft_rad, 0.0))
    hard = float(max(hard_rad, soft + 1e-6))  # avoid divide-by-zero
    vmarg = float(max(vel_margin, 0.0))
    vfloor = float(max(vel_floor_rad_tick, 0.0))
    dt = float(max(dt_s, 1e-6))

    for j in joint_indices:
        if j < 0 or j >= n:
            continue
        err = float(abs(tgt[j] - meas_q[j]))
        # Position backoff: 1.0 at err<=soft, 0.0 at err>=hard.
        if err <= soft:
            pos_scale = 1.0
        elif err >= hard:
            pos_scale = 0.0
        else:
            pos_scale = (hard - err) / (hard - soft)
        pos_cap = base * pos_scale
        # Velocity cap: never command faster than vel_margin * |dq|
        # per tick. Floored so the wire can start from rest.
        vel_cap = max(vfloor, vmarg * abs(float(meas_dq[j])) * dt)
        eff = min(pos_cap, vel_cap)
        if eff < 0.0:
            eff = 0.0
        cap[j] = np.float32(eff)
        if eff < 0.5 * base:
            throttle_count += 1

    return cap, throttle_count


def _lpf_alpha_from_hz(cutoff_hz: float, dt_s: float) -> float:
    if cutoff_hz <= 0.0:
        return 0.0
    return float(1.0 - math.exp(-2.0 * math.pi * cutoff_hz * dt_s))


def _lpf_vector(
    target: np.ndarray,
    state: np.ndarray | None,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """One-pole low-pass; returns (filtered, new_state)."""
    tgt = np.asarray(target, dtype=np.float32)
    if alpha <= 0.0:
        return tgt, tgt.copy()
    if state is None:
        return tgt, tgt.copy()
    out = ((1.0 - alpha) * state + alpha * tgt).astype(np.float32)
    return out, out.copy()


class VlaBodyMode(str, Enum):
    """VLA body authority modes — mirrors Quest 3 ``StreamMode`` semantics.

    * ``manipulation`` — like VR ``ARM_MANIPULATION``: decode motion tokens
      for arms/head/hands but pin ``legs`` + ``waist`` to ``idle_stand``
      (reach + grasp without locomotion collapse risk).
    * ``locomotion`` — like VR ``LOCOMOTION``: full-body decode from motion
      tokens (requires ``--motion-token-decoder``).
    """

    MANIPULATION = "manipulation"
    LOCOMOTION = "locomotion"


def _parse_vla_body_mode(name: str) -> VlaBodyMode:
    key = name.strip().lower().replace("-", "_")
    if key == "upper_body":
        print(
            "[live-VLA] WARN: body mode 'upper_body' is deprecated; "
            "use 'manipulation' (arms+hands decode, legs/waist frozen).",
            flush=True,
        )
        key = VlaBodyMode.MANIPULATION.value
    for mode in VlaBodyMode:
        if mode.value == key:
            return mode
    known = ", ".join(m.value for m in VlaBodyMode)
    raise ValueError(f"unknown VLA body mode {name!r}; choose one of: {known}")


def _body_mode_wire_settings(
    mode: VlaBodyMode,
    *,
    decoder_loaded: bool,
    freeze_groups_override: str = "",
) -> tuple[np.ndarray, bool]:
    """Return ``(freeze_indices, decode_body)`` for a mode."""
    if mode is VlaBodyMode.MANIPULATION:
        groups = freeze_groups_override.strip() or "legs,waist"
        freeze = _resolve_freeze_body_indices(groups)
        return freeze, decoder_loaded
    # LOCOMOTION
    return np.array([], dtype=np.int64), decoder_loaded


def _read_body_mode_control_file(path: str) -> Optional[VlaBodyMode]:
    """Read optional runtime mode switch file (one word per line)."""
    try:
        raw = Path(path).read_text(encoding="utf-8").strip().split()
        if not raw:
            return None
        return _parse_vla_body_mode(raw[0])
    except (OSError, ValueError):
        return None


def _resolve_freeze_body_indices(groups: str) -> np.ndarray:
    """Parse comma-separated planner joint-group names into MJ DOF indices."""
    from gear_sonic.utils.planner.x2_recipes import _GROUP_INDICES

    if not groups.strip():
        return np.array([], dtype=np.int64)
    indices: set[int] = set()
    for raw in groups.split(","):
        name = raw.strip()
        if not name:
            continue
        if name not in _GROUP_INDICES:
            known = ", ".join(sorted(_GROUP_INDICES))
            raise ValueError(
                f"unknown --vla-freeze-body-groups entry {name!r}; "
                f"known groups: {known}"
            )
        indices.update(_GROUP_INDICES[name])
    return np.asarray(sorted(indices), dtype=np.int64)


def _apply_frozen_body_groups(
    *,
    jpos: np.ndarray,
    future_fields: dict[str, np.ndarray],
    idle_jpos: np.ndarray,
    freeze_indices: np.ndarray,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Overwrite selected body DOFs with the idle_stand reference."""
    if freeze_indices.size == 0:
        return jpos, future_fields
    out = np.asarray(jpos, dtype=np.float32).copy()
    idle = np.asarray(idle_jpos, dtype=np.float32)
    out[freeze_indices] = idle[freeze_indices]
    ff = dict(future_fields)
    if "joint_pos_mj_future" in ff:
        jf = np.asarray(ff["joint_pos_mj_future"], dtype=np.float32).copy()
        jf[:, freeze_indices] = idle[freeze_indices]
        ff["joint_pos_mj_future"] = jf
        ff["joint_vel_mj_future"] = _jvel_future_from_poses(out, jf)
    return out, ff


def _jvel_future_from_poses(
    jpos_now: np.ndarray, jpos_future: np.ndarray
) -> np.ndarray:
    """Finite-difference joint velocities for the v5 future window."""
    jvel_future = np.zeros_like(jpos_future)
    prev = np.asarray(jpos_now, dtype=np.float32)
    for k in range(jpos_future.shape[0]):
        jvel_future[k] = (jpos_future[k] - prev) / max(_FUTURE_DT_S, 1e-6)
        prev = jpos_future[k]
    return jvel_future


def _build_vla_decoded_pose_payload(
    *,
    decoder: Any,
    proprio_990: np.ndarray,
    token_chunk: np.ndarray,
    chunk_step: int,
    horizon: int,
    base_frame_index: int,
) -> Optional[tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]]:
    """Decode VLA tokens to body trajectory + build wire payload.

    Pulls the current step's motion_token plus 9 future steps spaced
    :data:`_FUTURE_STEP_TICKS` apart from ``token_chunk``, runs them
    through :class:`SonicTokenToPoseDecoder` (which mirrors the C++
    deploy's ``target_mj = default + action_il * action_scale``
    formula), and returns the 3-tuple

    * ``joint_pos_mj_now``: ``(31,) float32`` MuJoCo-order pose for
      the current tick;
    * ``future_fields``: dict matching :func:`_idle_future_payload_fields`
      shape (``joint_pos_mj_future`` etc.) with the next 9 decoded
      poses;
    * ``action_il_now``: ``(31,) float32`` raw IsaacLab-order
      residual that the decoder produced for the current step. The
      publisher feeds this back into its
      :class:`~gear_sonic.utils.teleop.sonic_decoder_proprio.ProprioceptionBuffer`
      as ``last_action_il`` for the *next* tick's decode, mirroring the
      C++ deploy's own ``prop_buf_.Append(..., last_action_il_, ...)``
      bookkeeping.

    Returns ``None`` (caller falls back to idle wire) on any decode
    failure -- we never want a render glitch / NaN to brick the
    publisher loop. The publisher is the only thing keeping the deploy
    upright.

    The future window samples ``token_chunk[step+(k+1)*5]`` for
    ``k=0..8``, clamped at ``horizon-1``. That mirrors the encoder's
    ``DT_FUTURE_REF=0.1 s`` sampling at the 50 Hz publisher cadence
    (``5 ticks = 0.1 s``).
    """
    try:
        from gear_sonic.utils.teleop.sonic_token_to_pose_decoder import (
            action_il_to_target_pose_mj,
        )
        slot_indices = [
            min(chunk_step + (k + 1) * _FUTURE_STEP_TICKS, horizon - 1)
            for k in range(_NUM_FUTURE_SLOTS)
        ]
        all_indices = [min(chunk_step, horizon - 1)] + slot_indices
        sampled_tokens = np.stack(
            [token_chunk[i] for i in all_indices], axis=0
        )
        # decode_chunk handles batched torch inference in one shot so
        # the per-tick cost stays around 1-2 ms even with the 9 future
        # slots in the same call. We split the wrapper here so the
        # caller can capture the raw ``action_il`` residual (needed
        # for next-tick proprio bookkeeping) before it gets folded
        # into a MuJoCo-order target via the deploy parity formula.
        actions_il = decoder.decode_chunk(
            sampled_tokens.astype(np.float32), proprio_990
        )
        poses = np.stack(
            [action_il_to_target_pose_mj(actions_il[i]) for i in range(actions_il.shape[0])],
            axis=0,
        )
        action_il_now = actions_il[0].astype(np.float32, copy=False)
    except Exception as exc:  # noqa: BLE001
        # Caller logs (sticky one-shot, see _publisher); we just bail.
        print(
            f"[live-VLA] decoder error: {exc} -- falling back to idle wire",
            flush=True,
        )
        return None

    joint_pos_mj_now = poses[0].astype(np.float32, copy=False)
    joint_pos_mj_future = poses[1:].astype(np.float32, copy=False)
    # Hold root quat at identity / repeat -- the SONIC decoder does not
    # predict a root pose (the policy is body-only), so we mirror the
    # idle wire's identity quat for both current + future. This is the
    # same convention the heuristic planner uses for IDLE_LOOP frames
    # (see _IDLE_QUAT_FUTURE construction).
    quat_future = _IDLE_QUAT_FUTURE
    # Finite-difference jvel from the decoded pose trajectory so the
    # deploy's tokenizer_obs gets a non-zero velocity component (rather
    # than the constant 0 the idle window ships). dt = _FUTURE_DT_S.
    jvel_future = np.zeros(
        (_NUM_FUTURE_SLOTS, NUM_BODY_DOFS), dtype=np.float32
    )
    prev = joint_pos_mj_now
    for k in range(_NUM_FUTURE_SLOTS):
        jvel_future[k] = (
            (joint_pos_mj_future[k] - prev) / max(_FUTURE_DT_S, 1e-6)
        )
        prev = joint_pos_mj_future[k]
    fidx_future = np.array(
        [base_frame_index + (k + 1) for k in range(_NUM_FUTURE_SLOTS)],
        dtype=np.int64,
    )
    future_fields = {
        "joint_pos_mj_future": joint_pos_mj_future,
        "root_quat_xyzw_future": quat_future,
        "joint_vel_mj_future": jvel_future,
        "frame_index_future": fidx_future,
        "future_dt_s": _FUTURE_DT_FIELD,
    }
    return joint_pos_mj_now, future_fields, action_il_now


class _IdleStandLoop:
    """Replay the planner's ``idle_stand`` primitive at 50 Hz.

    The bridge -- when run in idle / ``--no-policy`` mode -- needs to
    publish a wire-content-byte-equivalent stream to what the heuristic
    planner emits during IDLE_LOOP. Empirically (2026-05-14):

      * Static stand (DEFAULT_STAND_POSE_NP every tick + identity quat,
        no future window) makes the policy output saturated raw actions
        and the robot leans ~25 deg under gravity even with a v5 window.
      * Replaying ``idle_stand`` (the 30 fps captured stand clip
        resampled to 50 Hz, ~75 frames) keeps the robot perfectly
        upright (grav_z = -1.00 for 20k+ ticks under
        ``--planner-only``).

    The two policies-relevant differences are (a) ``idle_stand[0]``'s
    waist-yaw is ~33 deg off DEFAULT_STAND_POSE_NP, and (b) the clip
    has small per-frame DOF jitter the policy was trained against. We
    reproduce both by indexing into the raw clip arrays and wrapping
    modulo the clip length. Yaw alignment is a no-op when
    (xy_world, yaw_world) = (0, 0), which is the bridge's invariant
    (we do not move the base), so we skip the planner's
    ``yaw_align_segment`` machinery and read the dof / quat arrays
    verbatim.

    Future-window slots are pulled from the same clip at
    ``_FUTURE_STEP_TICKS`` (= 5) tick spacing, mirroring
    ``planner.step_with_lookahead(num_future=9, step_ticks=5)``. The
    clip is loopable, so wrap-around at the seam is fine for an idle
    stand (no DOF discontinuity bigger than a few mrad).
    """

    def __init__(
        self,
        *,
        dof: np.ndarray,
        quat_xyzw: np.ndarray,
        bin_name: str = "idle_stand",
    ) -> None:
        if dof.ndim != 2 or dof.shape[1] != NUM_BODY_DOFS:
            raise ValueError(
                f"idle_stand dof must be (T,{NUM_BODY_DOFS}), "
                f"got {dof.shape}"
            )
        if quat_xyzw.ndim != 2 or quat_xyzw.shape[1] != 4:
            raise ValueError(
                f"idle_stand root_quat_xyzw must be (T,4), got {quat_xyzw.shape}"
            )
        if dof.shape[0] != quat_xyzw.shape[0]:
            raise ValueError(
                f"idle_stand dof / quat length mismatch: "
                f"dof={dof.shape[0]} quat={quat_xyzw.shape[0]}"
            )
        if dof.shape[0] < 1:
            raise ValueError("idle_stand clip is empty")
        self._dof = np.ascontiguousarray(dof, dtype=np.float32)
        self._quat = np.ascontiguousarray(quat_xyzw, dtype=np.float32)
        self._n_frames = int(dof.shape[0])
        self._bin_name = str(bin_name)
        # Pre-broadcast a zero joint_vel slab (we ship it explicitly so
        # the deploy skips its finite-diff path; the clip's per-frame
        # delta is small enough that "zero velocity" is a close enough
        # IL-time-scale approximation -- the policy treats joint_vel
        # primarily as a sanity-of-state signal, not a feedforward).
        self._zero_jvel = np.zeros(
            (_NUM_FUTURE_SLOTS, NUM_BODY_DOFS), dtype=np.float32
        )
        self._zero_jvel.setflags(write=False)

    @property
    def n_frames(self) -> int:
        return self._n_frames

    @property
    def bin_name(self) -> str:
        return self._bin_name

    def current(self, tick: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(joint_pos_mj, root_quat_xyzw)`` for the current tick."""
        i = int(tick) % self._n_frames
        return self._dof[i].copy(), self._quat[i].copy()

    def future_window(self, tick: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(jpos_future, quat_future, jvel_future)`` for the next 9 slots."""
        idx = np.array(
            [
                (int(tick) + (k + 1) * _FUTURE_STEP_TICKS) % self._n_frames
                for k in range(_NUM_FUTURE_SLOTS)
            ],
            dtype=np.int64,
        )
        return self._dof[idx].copy(), self._quat[idx].copy(), self._zero_jvel

    def future_payload_fields(
        self, *, tick: int, base_frame_index: int,
    ) -> dict[str, np.ndarray]:
        jpos_future, quat_future, jvel_future = self.future_window(tick)
        fidx_future = np.array(
            [base_frame_index + (k + 1) for k in range(_NUM_FUTURE_SLOTS)],
            dtype=np.int64,
        )
        return {
            "joint_pos_mj_future": jpos_future,
            "root_quat_xyzw_future": quat_future,
            "joint_vel_mj_future": jvel_future,
            "frame_index_future": fidx_future,
            "future_dt_s": _FUTURE_DT_FIELD,
        }


def _load_idle_stand_loop(
    primitives_pkl: str | Path,
) -> _IdleStandLoop:
    """Load ``idle_stand`` from the planner's primitives PKL.

    The PKL is a ``{bin_name: {dof, root_rot_xyzw, root_trans, fps,
    motion_key, ...}}`` mapping written by ``curate_x2_primitives.py``.
    We use ``joblib.load`` here so we don't have to import the planner's
    state machine (which depends on scipy via ``blending``); the bridge
    must remain importable in ``--no-policy`` mode without the heavy
    planner dependency tree.

    Resampling to 50 Hz: at the time of writing, idle_stand ships at
    30 fps (captured from teleop). We mirror the same
    ``resample_motion_30_to_50hz`` upcast the planner's
    ``load_primitives_pkl`` performs so per-frame timing matches the
    deploy's 50 Hz control loop. The resampler lives in
    ``gear_sonic.utils.planner.blending`` -- importing it costs scipy
    on the bridge process, but only when the operator opts into
    ``--idle-stand-pkl`` (or its default). Without resampling the wire
    would tick through the clip at 60 % of the policy's expected rate
    and the per-frame DOF velocity would be 5/3 of training-time --
    same OOD risk we just got out of.
    """
    pkl_path = Path(primitives_pkl)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"primitives PKL not found: {pkl_path}")

    raw = joblib.load(pkl_path)
    if not isinstance(raw, dict):
        raise ValueError(f"{pkl_path}: expected dict at top level, got {type(raw)}")
    if "idle_stand" not in raw:
        raise KeyError(
            f"{pkl_path}: 'idle_stand' bin missing -- the planner needs this "
            f"too; re-curate primitives with curate_x2_primitives.py."
        )
    payload = raw["idle_stand"]
    dof_src = np.asarray(payload["dof"], dtype=np.float32)
    rot_src = np.asarray(payload["root_rot_xyzw"], dtype=np.float32)
    trans_src = np.asarray(payload["root_trans"], dtype=np.float64)
    src_fps = float(payload["fps"])
    target_fps = float(DEFAULT_PUB_RATE_HZ)
    if abs(src_fps - target_fps) < 0.5:
        dof_out, rot_out, trans_out = dof_src, rot_src, trans_src
    else:
        from gear_sonic.utils.planner.blending import resample_motion_30_to_50hz
        dof_out, rot_out, trans_out = resample_motion_30_to_50hz(
            dof_src, rot_src, trans_src, src_fps, target_fps,
        )
    # Yaw-align the clip so frame 0's heading sits at (xy=0, yaw=0).
    # Without this the wire would carry idle_stand's raw 90 deg yaw
    # (the operator was facing +y when the clip was captured), which
    # mismatches the deploy's robot-yaw=0 spawn and forces a same-tick
    # heading correction. The planner's
    # ``LocalMotionPlanner._start_idle_loop`` does this exact alignment
    # against ``(self._cur_xy, self._cur_yaw)`` = (0, 0); reproducing
    # it here keeps wire-content byte-equivalent to the planner.
    from gear_sonic.utils.planner.blending import yaw_align_segment
    aligned_dof, aligned_rot, _ = yaw_align_segment(
        dof_out, rot_out, trans_out,
        xy_world=np.zeros(2, dtype=np.float64),
        yaw_world=0.0,
    )
    return _IdleStandLoop(dof=aligned_dof, quat_xyzw=aligned_rot)


def _zeros_action_horizon(horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Safe SONIC latents + hands (matches bootstrap chunk semantics)."""
    h = max(int(horizon), 1)
    zt = np.zeros((h, SONIC_MOTION_TOKEN_DIM), dtype=np.float32)
    zh = np.zeros((h, DEFAULT_HAND_DOF), dtype=np.float32)
    return zt, zh, zh


def _post_safe_idle_chunk(
    chunk: _LatestChunk,
    *,
    state: _LatestState,
    last_inference_ms: float,
    log_line: str | None = None,
) -> None:
    """Reset the walking chunk to zero latents; body column mirrors live sim if fresh else stand.

    Called after render / ``get_action`` failures so the 50 Hz publisher
    does not keep advancing through a broken VLA horizon.
    """
    token_cur, _, _, _, _ = chunk.read()
    h = int(token_cur.shape[0])
    token, left, right = _zeros_action_horizon(h)
    bq, _, _, _, _, alive = state.snapshot()
    body_pose = bq.astype(np.float32, copy=True) if alive else _default_stand_body_pose_f32()
    chunk.post(
        token=token,
        left_hand=left,
        right_hand=right,
        body_pose=body_pose,
        last_inference_ms=float(last_inference_ms),
    )
    if log_line:
        print(log_line, flush=True)


def _quat_wxyz_to_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world ``[0, 0, -1]`` into the body frame using a wxyz quaternion.

    Mirrors the projection used by the IsaacLab observation pipeline:
    ``projected_gravity = R(quat).T @ [0, 0, -1]``.
    """
    w, x, y, z = (float(v) for v in quat_wxyz.reshape(-1))
    # Standard quaternion -> 3x3 rotation matrix (active, world<-body).
    # We need its transpose to bring the world gravity into the body frame.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    # R.T @ [0, 0, -1] picks out -1 * column 2.
    return np.array([-r02, -r12, -r22], dtype=np.float64)


def _quat_wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    """MuJoCo / x2_debug use ``wxyz``; ZMQ ``pose`` wire uses SciPy ``xyzw``."""
    w, x, y, z = (float(v) for v in quat_wxyz.reshape(-1)[:4])
    return np.array([x, y, z, w], dtype=np.float32)


# If we haven't seen a fresh ``x2_debug`` packet within this many seconds
# we declare the deploy *not* alive and stop both inference + video
# recording. The deploy publishes at 50 Hz, so anything past ~4 ticks
# (80 ms) is already abnormal; we use a generous 1.0 s to absorb
# transient stalls (long colcon build sleeping the bridge thread, GPU
# rendering hiccups, etc.) without falsely declaring death.
DEPLOY_ALIVE_STALE_THRESHOLD_S: float = 1.0


@dataclass
class _LatestState:
    """Thread-safe snapshot of the freshest ``x2_debug`` frame.

    We don't try to maintain a full queue: the inference loop always wants
    the *latest* frame, and a queue would just buffer staleness. A monotonic
    ``revision`` counter lets the inference thread block until something new
    arrives without polling.

    Liveness is tracked via :attr:`last_update_monotonic` (set every time
    ``update`` is called) and queried via :meth:`is_alive` against
    :data:`DEPLOY_ALIVE_STALE_THRESHOLD_S`. ``received_any`` is *only* a
    "have we ever seen a packet" flag and is deliberately one-way; the
    is_alive check is what callers should actually gate on, otherwise the
    bridge keeps recording forever after the deploy exits (the original
    sticky-True behaviour).
    """

    body_q_mj: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
    )
    base_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    # body_dq_mj + base_ang_vel are required to assemble the live
    # 990-D proprio that the SONIC pose decoder was trained on (see
    # gear_sonic/utils/teleop/sonic_decoder_proprio.py). The
    # ``x2_debug`` stream already publishes them per tick as ``body_dq``
    # and ``base_ang_vel`` respectively; sniffed live 2026-06-07.
    body_dq_mj: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_BODY_DOFS, dtype=np.float64)
    )
    base_ang_vel: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    left_hand_q: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_HAND_DOF))
    right_hand_q: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_HAND_DOF))
    revision: int = 0
    received_any: bool = False
    last_update_monotonic: float = 0.0
    cv: threading.Condition = field(default_factory=threading.Condition)

    def update(
        self,
        *,
        body_q_mj: np.ndarray,
        base_quat_wxyz: np.ndarray,
        left_hand_q: np.ndarray,
        right_hand_q: np.ndarray,
        # body_dq_mj + base_ang_vel were added 2026-06-07 for the
        # bridge-side ProprioceptionBuffer. They default to None so
        # legacy callers (the recorder yaw-rebase unit tests, future
        # mock-state helpers, etc.) keep working without having to
        # synthesise velocity terms they don't care about. When
        # omitted, the previously stored values are preserved -- so
        # an old caller that already filled in velocities via a prior
        # update() won't see them silently cleared.
        body_dq_mj: Optional[np.ndarray] = None,
        base_ang_vel: Optional[np.ndarray] = None,
    ) -> None:
        with self.cv:
            self.body_q_mj = body_q_mj.astype(np.float64, copy=False)
            self.base_quat_wxyz = base_quat_wxyz.astype(np.float64, copy=False)
            if body_dq_mj is not None:
                self.body_dq_mj = body_dq_mj.astype(np.float64, copy=False)
            if base_ang_vel is not None:
                self.base_ang_vel = base_ang_vel.astype(np.float64, copy=False)
            self.left_hand_q = left_hand_q.astype(np.float64, copy=False)
            self.right_hand_q = right_hand_q.astype(np.float64, copy=False)
            self.revision += 1
            self.received_any = True
            self.last_update_monotonic = time.monotonic()
            self.cv.notify_all()

    def is_alive(
        self,
        *,
        stale_threshold_s: float = DEPLOY_ALIVE_STALE_THRESHOLD_S,
        now_monotonic: float | None = None,
    ) -> bool:
        """Return True iff we have ever received an ``x2_debug`` packet
        AND the most recent one is younger than ``stale_threshold_s``.

        Note: ``last_update_monotonic`` defaults to 0.0 in the dataclass,
        which means the very first ``is_alive`` call before any packet
        arrives returns False (since ``now - 0.0 >> threshold``). After
        the deploy goes quiet (process exits / Ctrl-C), this flips back
        to False ``stale_threshold_s`` later, so the video recorder can
        gracefully close the writer instead of recording forever.
        """
        if not self.received_any:
            return False
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        return (now_monotonic - self.last_update_monotonic) <= stale_threshold_s

    def snapshot(
        self, *, stale_threshold_s: float = DEPLOY_ALIVE_STALE_THRESHOLD_S
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]:
        """Atomic snapshot. The bool return is ``is_alive``, NOT the
        sticky ``received_any`` (the latter caused the bridge to keep
        recording forever after the deploy exited)."""
        with self.cv:
            now = time.monotonic()
            alive = self.received_any and (
                (now - self.last_update_monotonic) <= stale_threshold_s
            )
            return (
                self.body_q_mj.copy(),
                self.base_quat_wxyz.copy(),
                self.left_hand_q.copy(),
                self.right_hand_q.copy(),
                self.revision,
                alive,
            )

    def snapshot_velocities(self) -> tuple[np.ndarray, np.ndarray]:
        """Atomic snapshot of (``body_dq_mj``, ``base_ang_vel``).

        Split from the main :meth:`snapshot` so existing call sites can
        keep their 6-tuple destructuring. The publisher uses this to
        feed the bridge-side ``ProprioceptionBuffer`` -- consuming
        ``snapshot()`` for positions+aliveness then this method for
        velocities is exactly equivalent to one combined call (both
        acquire ``self.cv`` and the underlying arrays are immutable
        copies); the worst case is one tick of staleness between the
        two reads, which is below the publisher's 50 Hz cadence.
        """
        with self.cv:
            return (self.body_dq_mj.copy(), self.base_ang_vel.copy())

    def wait_for_new(self, last_revision: int, timeout: float) -> int:
        """Block until ``revision`` advances past ``last_revision``."""
        with self.cv:
            self.cv.wait_for(lambda: self.revision > last_revision, timeout=timeout)
            return self.revision


@dataclass
class _LatestChunk:
    """Thread-safe latest-action chunk buffer.

    The publisher walks ``token[step]`` / ``left_hand[step]`` / ``right_hand[step]``
    one tick at a time. When the inference worker posts a new chunk
    (``chunk_id`` increments), the publisher resets ``step = 0``.
    """

    token: np.ndarray = field(
        default_factory=lambda: np.zeros((DEFAULT_ACTION_HORIZON, SONIC_MOTION_TOKEN_DIM), dtype=np.float32)
    )
    left_hand: np.ndarray = field(
        default_factory=lambda: np.zeros((DEFAULT_ACTION_HORIZON, DEFAULT_HAND_DOF), dtype=np.float32)
    )
    right_hand: np.ndarray = field(
        default_factory=lambda: np.zeros((DEFAULT_ACTION_HORIZON, DEFAULT_HAND_DOF), dtype=np.float32)
    )
    body_pose: np.ndarray = field(
        default_factory=lambda: np.asarray(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32).copy()
    )
    chunk_id: int = 0
    inference_count: int = 0  # how many inferences have completed
    last_inference_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def post(
        self,
        *,
        token: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        body_pose: np.ndarray,
        last_inference_ms: float,
    ) -> int:
        with self.lock:
            self.token = token.astype(np.float32, copy=False)
            self.left_hand = left_hand.astype(np.float32, copy=False)
            self.right_hand = right_hand.astype(np.float32, copy=False)
            self.body_pose = body_pose.astype(np.float32, copy=False)
            self.chunk_id += 1
            self.inference_count += 1
            self.last_inference_ms = float(last_inference_ms)
            return self.chunk_id

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        with self.lock:
            return (
                self.token,
                self.left_hand,
                self.right_hand,
                self.body_pose,
                self.chunk_id,
            )


@dataclass
class _LatestWireDebug:
    """Last wire pose + diagnostics for chunk dumps / postmortem."""

    joint_pos_mj: np.ndarray = field(
        default_factory=lambda: np.asarray(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32).copy()
    )
    left_hand_joints: np.ndarray = field(
        default_factory=lambda: np.zeros(DEFAULT_HAND_DOF, dtype=np.float32)
    )
    right_hand_joints: np.ndarray = field(
        default_factory=lambda: np.zeros(DEFAULT_HAND_DOF, dtype=np.float32)
    )
    # ``raw_joint_pos_mj`` is the policy's decoded body intent for the
    # CURRENT tick (post-SONIC-decode, pre-clamp / pre-LPF / pre-ramp /
    # pre-chunk-blend). Recorded so chunk dumps can FK both the raw
    # intent and the delivered wire and tell apart "policy never
    # targeted descend" from "bridge clamped the descend away". Mirrors
    # ``joint_pos_mj`` (the wire) when no clamp/LPF is active.
    raw_joint_pos_mj: np.ndarray = field(
        default_factory=lambda: np.asarray(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32).copy()
    )
    raw_delta_idle: float = 0.0
    wire_delta_idle: float = 0.0
    wire_delta_body: float = 0.0
    wire_delta_hand: float = 0.0
    chunk_id: int = 0
    chunk_step: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def update(
        self,
        *,
        joint_pos_mj: np.ndarray,
        left_hand_joints: np.ndarray,
        right_hand_joints: np.ndarray,
        raw_joint_pos_mj: np.ndarray,
        raw_delta_idle: float,
        wire_delta_idle: float,
        wire_delta_body: float,
        wire_delta_hand: float,
        chunk_id: int,
        chunk_step: int,
    ) -> None:
        with self.lock:
            self.joint_pos_mj = np.asarray(joint_pos_mj, dtype=np.float32).copy()
            self.left_hand_joints = np.asarray(
                left_hand_joints, dtype=np.float32
            ).copy()
            self.right_hand_joints = np.asarray(
                right_hand_joints, dtype=np.float32
            ).copy()
            self.raw_joint_pos_mj = np.asarray(
                raw_joint_pos_mj, dtype=np.float32
            ).copy()
            self.raw_delta_idle = float(raw_delta_idle)
            self.wire_delta_idle = float(wire_delta_idle)
            self.wire_delta_body = float(wire_delta_body)
            self.wire_delta_hand = float(wire_delta_hand)
            self.chunk_id = int(chunk_id)
            self.chunk_step = int(chunk_step)

    def snapshot(
        self,
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        float, float, float, float, int, int,
    ]:
        with self.lock:
            return (
                self.joint_pos_mj.copy(),
                self.left_hand_joints.copy(),
                self.right_hand_joints.copy(),
                self.raw_joint_pos_mj.copy(),
                self.raw_delta_idle,
                self.wire_delta_idle,
                self.wire_delta_body,
                self.wire_delta_hand,
                self.chunk_id,
                self.chunk_step,
            )


class _VlaControlSignal:
    """Thread-safe shared state for the optional ``vla_control`` SUB.

    Drives the bridge's manual-takeover cold-restart flow without
    racing the publisher thread:

    * ``override_engaged`` -> ``engage()`` sets ``override_active=True``;
      the publisher stops emitting decoded VLA chunks and instead
      ships the operator's current measured pose so the wire stays
      alive (the proxy ignores these frames -- override wins -- but
      this prevents a deploy starvation watchdog trip if the proxy
      were to drop both inputs simultaneously).
    * ``override_released`` -> ``release()`` clears
      ``override_active`` AND sets ``cold_restart_pending=True``. The
      publisher consumes the pending flag at the top of its next
      tick, clears all smoothing state (ramp / LPF / chunk blend),
      sets a chunk-id baseline so any stale chunk decoded against
      pre-override observations is ignored, and arms a brief
      "hold at measured pose" window so the proxy's HOLD -> LIVE
      handoff doesn't see a step change from the operator's pose to
      the bridge's idle-clip pose.

    The publisher reads ``snapshot()`` (atomic) at the top of each
    tick and ``consume_cold_restart()`` (atomic read-and-clear) once
    per pending-edge. ``stats()`` is used only for the periodic
    status print and never modifies state.

    See 2026-06-10 manual-takeover milestone for the full design.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._override_active = False
        self._cold_restart_pending = False
        self._engage_count = 0
        self._release_count = 0
        self._last_event_ts: float = -1.0
        # 2026-06-10: operator-pose handoff. ``release_pose`` snaps
        # the body + hand joints the operator was commanding the
        # instant override_released fired, so the bridge can hold
        # the wire at THAT pose during its cold-restart bridging
        # window (avoids the visible "pose reset" from snapping to
        # x2_debug's lagged measured pose). Cleared atomically by
        # ``consume_cold_restart()`` so a second release that has
        # no payload (older proxy, smoke test, ...) doesn't replay
        # the previous handoff. Empty dict / None both indicate
        # "fall back to legacy measured-pose hold".
        self._release_pose: Optional[Dict[str, np.ndarray]] = None

    def engage(self, ts: float = 0.0) -> None:
        with self._lock:
            self._override_active = True
            self._engage_count += 1
            self._last_event_ts = float(ts)

    def release(
        self,
        ts: float = 0.0,
        release_pose: Optional[Dict[str, np.ndarray]] = None,
    ) -> None:
        with self._lock:
            self._override_active = False
            self._cold_restart_pending = True
            self._release_count += 1
            self._last_event_ts = float(ts)
            self._release_pose = release_pose

    def snapshot(self) -> tuple[bool, bool]:
        """Return (override_active, cold_restart_pending) atomically."""
        with self._lock:
            return self._override_active, self._cold_restart_pending

    def consume_cold_restart(
        self,
    ) -> tuple[bool, Optional[Dict[str, np.ndarray]]]:
        """Atomically test-and-clear the cold-restart pending flag.

        Returns ``(pending, release_pose)``. ``release_pose`` is the
        body + hand joints the operator was commanding the instant
        the proxy fired override_released, or ``None`` when no pose
        snapshot was provided (legacy proxy, smoke tests). Callers
        should treat ``None`` as "use measured pose for the
        cold-restart bridging window" -- matches the pre-2026-06-10
        behaviour exactly.
        """
        with self._lock:
            pending = self._cold_restart_pending
            release_pose = self._release_pose
            self._cold_restart_pending = False
            self._release_pose = None
            return pending, release_pose

    def stats(self) -> tuple[int, int, float]:
        """Return (engage_count, release_count, last_event_ts)."""
        with self._lock:
            return (
                self._engage_count,
                self._release_count,
                self._last_event_ts,
            )


def _run_vla_control_sub(
    *,
    host: str,
    port: int,
    topic: str,
    signal: _VlaControlSignal,
    stop_event: threading.Event,
    poll_ms: int = 100,
) -> None:
    """Background SUB worker that translates proxy-emitted control
    events into ``_VlaControlSignal`` state flips.

    Lives in its own thread so the publisher's 50 Hz loop never has to
    poll the SUB itself (the SUB attaches to a remote PUB and could
    block on connect; we'd rather absorb that in a dedicated thread
    than risk skipping publisher ticks). The thread exits cleanly on
    ``stop_event.set()``; tearing down the SUB socket via
    ``close(linger=0)`` is safe because PUB/SUB has no in-flight
    acknowledgements.

    Unknown / malformed events are logged once and otherwise ignored
    so a future control-plane extension can't crash the bridge.
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVHWM, 16)
    url = f"tcp://{host}:{port}"
    sock.connect(url)
    sock.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(
        f"[live-VLA] vla_control SUB connected: {url} topic={topic!r}",
        flush=True,
    )
    warned_unknown_events: set[str] = set()
    try:
        while not stop_event.is_set():
            evs = dict(poller.poll(int(poll_ms)))
            if sock not in evs:
                continue
            try:
                parts = sock.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                continue
            if len(parts) < 2:
                continue
            try:
                evt = json.loads(parts[1].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            kind = evt.get("event")
            ts = float(evt.get("ts", 0.0))
            if kind == "override_engaged":
                signal.engage(ts=ts)
                print(
                    f"[live-VLA] vla_control: override_engaged "
                    f"(proxy_ts={ts:.3f}) -- pausing decoded chunks, "
                    f"holding measured pose on wire",
                    flush=True,
                )
            elif kind == "override_released":
                # 2026-06-10: optional release_pose payload (proxy
                # snapshot of operator's last commanded body + hand
                # joints) so the cold-restart bridging window can
                # hold the wire at the operator's pose instead of
                # x2_debug's lagged measured pose. Missing / bad
                # payload falls back to legacy behaviour with no
                # warning -- older proxies and smoke tests never
                # set it.
                release_pose: Optional[Dict[str, np.ndarray]] = None
                raw_release = evt.get("release_pose")
                if isinstance(raw_release, dict):
                    parsed: Dict[str, np.ndarray] = {}
                    for fname, expected_dim in (
                        ("joint_pos_mj", NUM_BODY_DOFS),
                        ("left_hand_joints", DEFAULT_HAND_DOF),
                        ("right_hand_joints", DEFAULT_HAND_DOF),
                    ):
                        raw = raw_release.get(fname)
                        if raw is None:
                            continue
                        try:
                            arr = np.asarray(raw, dtype=np.float32)
                        except (TypeError, ValueError):
                            continue
                        if arr.shape != (expected_dim,):
                            continue
                        parsed[fname] = arr
                    if parsed:
                        release_pose = parsed
                signal.release(ts=ts, release_pose=release_pose)
                rp_summary = (
                    "no release_pose; bridge will hold at "
                    "x2_debug measured pose (legacy)"
                    if release_pose is None
                    else (
                        "release_pose has " +
                        "+".join(sorted(release_pose.keys())) +
                        "; bridge will hold at operator pose"
                    )
                )
                print(
                    f"[live-VLA] vla_control: override_released "
                    f"(proxy_ts={ts:.3f}) -- cold restart armed; "
                    f"clearing ramp / LPF / chunk state on next "
                    f"tick; {rp_summary}",
                    flush=True,
                )
            else:
                if kind not in warned_unknown_events:
                    warned_unknown_events.add(str(kind))
                    print(
                        f"[live-VLA] vla_control: unknown event "
                        f"{kind!r}; ignoring (will not warn again).",
                        flush=True,
                    )
    finally:
        try:
            sock.close(linger=0)
        except Exception:
            pass


def _build_observation(
    *,
    body_q_mj: np.ndarray,
    base_quat_wxyz: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    camera_frames: dict[str, np.ndarray],
    prompt: str,
) -> dict[str, Any]:
    """Build the (B=1, T=1)-batched observation dict expected by ``Gr00tPolicy``.

    The slot layout matches the M5/M6 dataset's ``meta/modality.json``:

    * ``state.left_leg``           = pin[0:6]
    * ``state.right_leg``          = pin[6:12]
    * ``state.waist``              = pin[12:15]
    * ``state.left_arm``           = pin[17:24]
    * ``state.right_arm``          = pin[24:31]
    * ``state.left_hand``          = left_hand_q[:10]
    * ``state.right_hand``         = right_hand_q[:10]
    * ``state.projected_gravity``  = R(base_quat).T @ [0, 0, -1]
    * ``video.<key>``              = (1, 1, H, W, 3) uint8, one slot per
                                     ``modality_keys`` entry in the loaded
                                     ``video`` ModalityConfig (e.g.
                                     ``ego_view`` for sim, or
                                     ``stereo_left`` + ``stereo_right``
                                     for the real-robot omnihand stereo
                                     config).
    * ``language.annotation.human.task_description`` = [[prompt]]

    Note: head joints (pin[15:17]) are *not* exposed to the policy; they
    are intentionally omitted from ``modality.json``.

    The ``camera_frames`` keys must exactly match the registered video
    ``modality_keys`` (no remapping is performed here); the caller is
    expected to use :func:`_get_required_video_keys` to derive them.
    """
    body_q_pin = np.asarray(body_q_mj, dtype=np.float32)[list(MJ_TO_PIN)]
    proj_grav = _quat_wxyz_to_projected_gravity(base_quat_wxyz).astype(np.float32)
    left_h = np.asarray(left_hand_q, dtype=np.float32).reshape(-1)[:DEFAULT_HAND_DOF]
    right_h = np.asarray(right_hand_q, dtype=np.float32).reshape(-1)[:DEFAULT_HAND_DOF]

    def _b1(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(1, 1, -1)

    state = {
        "left_leg": _b1(body_q_pin[0:6]),
        "right_leg": _b1(body_q_pin[6:12]),
        "waist": _b1(body_q_pin[12:15]),
        "left_arm": _b1(body_q_pin[17:24]),
        "right_arm": _b1(body_q_pin[24:31]),
        "left_hand": _b1(left_h),
        "right_hand": _b1(right_h),
        "projected_gravity": _b1(proj_grav),
    }
    video: dict[str, np.ndarray] = {}
    for key, frame in camera_frames.items():
        frame_u8 = np.ascontiguousarray(frame).astype(np.uint8, copy=False)
        video[key] = frame_u8.reshape(1, 1, *frame_u8.shape)
    return {
        "video": video,
        "state": state,
        "language": {"annotation.human.task_description": [[prompt]]},
    }


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def _video_recorder(
    *,
    renderer_factory: Any,
    state: _LatestState,
    output_path: str,
    fps: float,
    stop_event: threading.Event,
    chunk: _LatestChunk,
    verbose: bool = False,
) -> None:
    """Thread D: render the live deploy state into an MP4 at ``fps`` Hz.

    Runs *separately* from the inference worker so the recorded video is
    smooth (50 Hz / 25 Hz, decoupled from the ~3 Hz VLA cadence) and so
    its EGL context doesn't share with the inference worker (mujoco's EGL
    backend uses thread-local GL contexts).

    Each frame is the same ``ego_view`` the VLA actually sees during
    inference, so the recorded clip is the policy's first-person camera.
    """
    print(f"[live-VLA] video thread: building MujocoFrameRenderer for {output_path}", flush=True)
    try:
        renderer = renderer_factory()
    except Exception as exc:
        print(f"[live-VLA] WARN: video renderer init failed: {exc}", flush=True)
        return

    # Lazy import keeps the bridge's hard-dep surface small for test
    # environments that don't ship pyav.
    from gear_sonic.data.video_writer import VideoWriter

    writer: Any = None
    period = 1.0 / max(fps, 1e-6)
    next_tick = time.monotonic()
    n_written = 0
    last_revision = -1
    last_tilt_warned = False
    waiting_logged = False
    try:
        while not stop_event.is_set():
            (
                body_q_mj, base_quat_wxyz, left_hq, right_hq, revision, alive
            ) = state.snapshot()

            # Three liveness states for the recorder:
            #   1. Pre-deploy: never received x2_debug. Hold writer closed,
            #      keep polling so the renderer's EGL context stays warm.
            #   2. Deploy alive: ``alive=True`` (recent packet within
            #      DEPLOY_ALIVE_STALE_THRESHOLD_S). Open writer (lazy) and
            #      record.
            #   3. Deploy died: writer is open but ``alive=False`` again
            #      (deploy stopped publishing). Close writer to flush the
            #      MP4, then exit. Without this branch the bridge kept
            #      recording forever after the deploy exited because
            #      ``received_any`` was sticky-True; the file stayed open
            #      and unflushable.
            if not alive:
                if writer is not None:
                    if verbose:
                        print(
                            f"[live-VLA] video: deploy went quiet "
                            f"(no x2_debug for >{DEPLOY_ALIVE_STALE_THRESHOLD_S}s); "
                            f"flushing {n_written} frames to {output_path}",
                            flush=True,
                        )
                    try:
                        # ``VideoWriter.stop()`` finalises the muxer (writes
                        # the moov atom on MP4) and closes the file. The
                        # outer ``finally`` will skip its own stop() call
                        # because we set ``writer = None`` here.
                        writer.stop()
                    except Exception as exc:
                        print(f"[live-VLA] video: writer.stop() warn: {exc}", flush=True)
                    writer = None
                    return  # Exit the recorder thread; nothing more to do.
                if verbose and not waiting_logged:
                    print(
                        f"[live-VLA] video: waiting for first x2_debug frame "
                        f"before opening {output_path} …",
                        flush=True,
                    )
                    waiting_logged = True
                time.sleep(period)
                next_tick = time.monotonic() + period
                continue

            try:
                frame = renderer.render_frame(
                    body_q=body_q_mj,
                    left_active=left_hq.astype(np.float64),
                    right_active=right_hq.astype(np.float64),
                    root_quat_wxyz=base_quat_wxyz,
                )
            except Exception as exc:
                if not last_tilt_warned:
                    print(f"[live-VLA] video render warn (will keep trying): {exc}", flush=True)
                    last_tilt_warned = True
                time.sleep(period)
                continue
            last_tilt_warned = False

            if writer is None:
                # PyAV's add_stream(rate=...) expects an int / Fraction
                # (a plain float trips
                #   AttributeError: 'float' object has no attribute 'numerator'
                # at the first encode call).
                writer = VideoWriter(
                    output_path=output_path,
                    width=renderer.width,
                    height=renderer.height,
                    fps=int(round(fps)),
                    codec="h264",
                    buffer_size=200,
                )
                if verbose:
                    print(
                        f"[live-VLA] video: deploy alive, recording -> "
                        f"{output_path} ({renderer.width}x{renderer.height} "
                        f"@ {int(round(fps))} fps)",
                        flush=True,
                    )

            writer.add_frame(np.ascontiguousarray(frame))
            n_written += 1
            last_revision = revision

            if verbose and n_written % int(max(fps, 1)) == 0:
                _, _, _, _, chunk_id_now = chunk.read()
                print(
                    f"[live-VLA] video: {n_written:5d} frames written  "
                    f"(rev={revision} chunk_id={chunk_id_now} alive={alive})",
                    flush=True,
                )

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    finally:
        if writer is not None:
            try:
                writer.stop()
            except Exception as exc:
                print(f"[live-VLA] video writer.stop error: {exc}", flush=True)
        try:
            renderer.close()
        except Exception:
            pass
        print(f"[live-VLA] video thread done; wrote {n_written} frames -> {output_path}", flush=True)


def _x2_debug_subscriber(
    *,
    sub_url: str,
    topic: str,
    state: _LatestState,
    stop_event: threading.Event,
    verbose: bool = True,
) -> None:
    """Thread A: SUB to ``x2_debug`` and update :class:`_LatestState`.

    ``verbose`` is honoured for parity with
    :class:`gear_sonic.utils.teleop.x2_dataset_recorder.X2DatasetRecorder`
    which reuses this helper and forwards its own ``cfg.verbose`` flag.
    Setting it to False mutes the one-time SUB-connected print and the
    per-frame decode-error messages -- useful when this thread is only
    feeding the recorder's deploy-silent watchdog and the operator does
    not need a second copy of the bind line in the recorder log.
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVHWM, 5)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(sub_url)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    if verbose:
        print(
            f"[live-VLA] x2_debug SUB connected to {sub_url} (topic={topic!r})",
            flush=True,
        )

    try:
        while not stop_event.is_set():
            events = dict(poller.poll(200))
            if sock not in events:
                continue
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            try:
                msg = unpack_message(raw, expected_topic=topic)
            except ValueError as exc:
                if verbose:
                    print(f"[live-VLA] x2_debug decode error: {exc}", flush=True)
                continue

            body_q = np.asarray(msg.fields.get("body_q", DEFAULT_STAND_POSE_MUJOCO_RAD), dtype=np.float64).reshape(-1)
            if body_q.shape[0] != NUM_BODY_DOFS:
                continue
            base_quat = np.asarray(msg.fields.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if base_quat.shape[0] != 4:
                base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            # ``body_dq`` + ``base_ang_vel`` feed the bridge-side
            # ProprioceptionBuffer (sonic_decoder_proprio.py). Fall
            # back to zeros if the deploy publishes an older schema
            # without them -- the SONIC decoder is OOD with zero
            # velocities but won't crash, and the operator will see a
            # WARN on first tick (logged from the publisher loop).
            body_dq = np.asarray(msg.fields.get("body_dq", np.zeros(NUM_BODY_DOFS)), dtype=np.float64).reshape(-1)
            if body_dq.shape[0] != NUM_BODY_DOFS:
                body_dq = np.zeros(NUM_BODY_DOFS, dtype=np.float64)
            base_ang_vel = np.asarray(msg.fields.get("base_ang_vel", np.zeros(3)), dtype=np.float64).reshape(-1)
            if base_ang_vel.shape[0] != 3:
                base_ang_vel = np.zeros(3, dtype=np.float64)
            left_hq = np.asarray(msg.fields.get("left_hand_q", np.zeros(DEFAULT_HAND_DOF)), dtype=np.float64).reshape(-1)[:DEFAULT_HAND_DOF]
            right_hq = np.asarray(msg.fields.get("right_hand_q", np.zeros(DEFAULT_HAND_DOF)), dtype=np.float64).reshape(-1)[:DEFAULT_HAND_DOF]
            if left_hq.shape[0] < DEFAULT_HAND_DOF:
                left_hq = np.pad(left_hq, (0, DEFAULT_HAND_DOF - left_hq.shape[0]))
            if right_hq.shape[0] < DEFAULT_HAND_DOF:
                right_hq = np.pad(right_hq, (0, DEFAULT_HAND_DOF - right_hq.shape[0]))

            state.update(
                body_q_mj=body_q,
                base_quat_wxyz=base_quat,
                body_dq_mj=body_dq,
                base_ang_vel=base_ang_vel,
                left_hand_q=left_hq,
                right_hand_q=right_hq,
            )
    finally:
        try:
            sock.close(linger=0)
        except Exception:
            pass


def _inference_worker(
    *,
    policy: Any,
    ghost_provider_factory: Any | None,
    camera_provider: _RealCameraProvider | None,
    camera_max_age_s: float,
    state: _LatestState,
    chunk: _LatestChunk,
    wire_debug: _LatestWireDebug,
    prompt: str,
    stop_event: threading.Event,
    min_period_s: float,
    verbose: bool = False,
    dump_chunks_dir: str | None = None,
    dump_chunks_every: int = 5,
    wait_for_ready_file: str | None = None,
    wait_for_ready_file_timeout_s: float = 120.0,
) -> None:
    """Thread B: render or grab frames + run VLA continuously, post fresh chunks.

    We don't run faster than ``1/min_period_s`` Hz to avoid burning GPU
    cycles when the deploy hasn't even consumed the previous chunk yet.

    Two camera sources are supported:

    * ``ghost_provider_factory`` (sim / ghost): a zero-arg callable that
      returns a *built* :class:`_GhostCameraProvider`. Called from inside
      this thread so the underlying MuJoCo EGL contexts (one per unique
      MJCF camera) are created on the thread that will actually render.
      An identity ``root_quat`` is used on the inference path so the
      VLA never sees a tilted horizon (the video-recorder thread uses
      the live ``base_quat`` instead — they are intentionally asymmetric).
    * ``camera_provider`` (real-robot / ZMQ): reuses the on-PC2
      :class:`ComposedCameraClientSensor` stream and returns whatever
      keys the registered video modality config requests (typically
      ``stereo_left`` + ``stereo_right``). The provider runs its own
      background polling thread, so we just call ``read_frames`` here
      with a staleness threshold.

    Exactly one of the two must be supplied.
    """
    if (ghost_provider_factory is None) == (camera_provider is None):
        raise ValueError(
            "_inference_worker requires exactly one of ghost_provider_factory or "
            "camera_provider (got ghost_provider_factory="
            f"{'set' if ghost_provider_factory is not None else 'None'}, "
            f"camera_provider={'set' if camera_provider is not None else 'None'})."
        )
    ghost_provider: _GhostCameraProvider | None = None
    if ghost_provider_factory is not None:
        print("[live-VLA] inference thread: building MuJoCo ghost cameras …", flush=True)
        try:
            ghost_provider = ghost_provider_factory()
        except Exception as exc:
            print(f"[live-VLA] FATAL: ghost camera init failed in worker: {exc}", flush=True)
            stop_event.set()
            return
        # ``ghost_provider`` is guaranteed non-None here.
        ghost_provider_obj = ghost_provider  # for type-narrowing below
        print(
            f"[live-VLA] inference thread: ghost cameras ready — "
            f"keys={ghost_provider_obj.required_keys} "
            f"(MJ cams={ghost_provider_obj.unique_cameras}, "
            f"{ghost_provider_obj.width}x{ghost_provider_obj.height}, "
            f"omnihand={ghost_provider_obj.with_omnihand})",
            flush=True,
        )
    else:
        print(
            f"[live-VLA] inference thread: using real-camera provider "
            f"(staleness threshold {camera_max_age_s:.2f}s)",
            flush=True,
        )

    # Optional ready-file gate: wait until a co-spawned recorder signals
    # it is subscribed before producing the first VLA chunk. The publisher
    # thread continues streaming idle stand at 50 Hz throughout the wait,
    # so the deploy never sees a wire gap. See ``--wait-for-ready-file``
    # for the rationale (capturing the arm rise from idle in the
    # recording, not the post-warm-up freeze).
    if wait_for_ready_file:
        ready_path = Path(wait_for_ready_file)
        print(
            f"[live-VLA] inference thread: holding at idle stand until "
            f"recorder ready-file appears at {ready_path} "
            f"(timeout={wait_for_ready_file_timeout_s:.0f}s; "
            f"publisher keeps streaming idle stand meanwhile)",
            flush=True,
        )
        wait_t0 = time.monotonic()
        warned_long_wait = False
        while not stop_event.is_set():
            if ready_path.exists():
                elapsed = time.monotonic() - wait_t0
                print(
                    f"[live-VLA] inference thread: ready-file detected "
                    f"after {elapsed:.1f}s -- starting VLA inference",
                    flush=True,
                )
                break
            elapsed = time.monotonic() - wait_t0
            if (
                wait_for_ready_file_timeout_s > 0
                and elapsed >= wait_for_ready_file_timeout_s
            ):
                print(
                    f"[live-VLA] inference thread: WARNING ready-file "
                    f"{ready_path} did not appear within "
                    f"{wait_for_ready_file_timeout_s:.0f}s -- proceeding "
                    "without recorder handshake (rise may not be "
                    "captured if a recorder is still warming up)",
                    flush=True,
                )
                break
            if elapsed >= 30.0 and not warned_long_wait:
                print(
                    f"[live-VLA] inference thread: still waiting for "
                    f"ready-file {ready_path} after {elapsed:.0f}s "
                    "(recorder slow to warm up?)",
                    flush=True,
                )
                warned_long_wait = True
            time.sleep(0.1)
        if stop_event.is_set():
            return

    last_revision = -1
    n_inferences = 0
    n_dropped_stale_cam = 0
    last_stale_log_t = 0.0
    # Counter for "policy returned malformed output" rejections. We use a
    # dedicated counter (rather than `n_inferences`, which only ticks on
    # SUCCESSFUL chunks) so we can bound the per-chunk BAD_chunk_*.npz dump
    # to the first few failures -- enough for offline NaN postmortem
    # without blowing up disk if every chunk is bad.
    n_bad_chunks = 0
    _MAX_BAD_CHUNK_DUMPS = 5
    dump_dir_path: Path | None = None
    if dump_chunks_dir:
        dump_dir_path = Path(dump_chunks_dir)
        dump_dir_path.mkdir(parents=True, exist_ok=True)
        print(
            f"[live-VLA] inference thread: chunk dump enabled "
            f"(dir={dump_dir_path}, every={dump_chunks_every})",
            flush=True,
        )
    while not stop_event.is_set():
        # Block until a new x2_debug frame arrives (don't run inference on
        # stale state).
        rev = state.wait_for_new(last_revision, timeout=0.5)
        if rev <= last_revision:
            continue  # timeout, retry

        body_q_mj, base_quat_wxyz, left_hq, right_hq, revision, deploy_fresh = state.snapshot()
        if not deploy_fresh:
            continue
        last_revision = revision

        t0 = time.monotonic()
        camera_frames: dict[str, np.ndarray] = {}
        if ghost_provider is not None:
            try:
                # The inference cameras deliberately use the renderer's
                # default identity root quat -- the VLA training set
                # (record_synthetic_smoketest_dataset.py) renders frames the
                # same way, so feeding a live ``base_quat`` here would
                # inject a tilted horizon the policy has never seen and
                # cause OOD behaviour mid-rollout. The *video recorder*
                # thread does pass the live ``base_quat`` so the recording
                # reflects physical reality (tip / fall) -- those two
                # renderers are intentionally asymmetric.
                camera_frames = ghost_provider.render_frames(
                    body_q=body_q_mj,
                    left_active=left_hq.astype(np.float64),
                    right_active=right_hq.astype(np.float64),
                )
            except Exception as exc:
                print(f"[live-VLA] render error: {exc}", flush=True)
                _post_safe_idle_chunk(
                    chunk,
                    state=state,
                    last_inference_ms=0.0,
                    log_line="[live-VLA] render error → zero-token safe chunk posted",
                )
                time.sleep(0.05)
                continue
        else:
            assert camera_provider is not None  # for type-checkers
            frames = camera_provider.read_frames(max_age_s=camera_max_age_s)
            if frames is None:
                n_dropped_stale_cam += 1
                now = time.monotonic()
                if now - last_stale_log_t >= 2.0:
                    print(
                        f"[live-VLA] cameras stale (>{camera_max_age_s:.2f}s "
                        f"old; latest age={camera_provider.latest_age_s:.2f}s, "
                        f"total dropped inferences={n_dropped_stale_cam}) → "
                        f"posting safe idle chunk",
                        flush=True,
                    )
                    last_stale_log_t = now
                _post_safe_idle_chunk(
                    chunk,
                    state=state,
                    last_inference_ms=0.0,
                    log_line=None,
                )
                time.sleep(0.05)
                continue
            camera_frames = frames

        observation = _build_observation(
            body_q_mj=body_q_mj,
            base_quat_wxyz=base_quat_wxyz,
            left_hand_q=left_hq,
            right_hand_q=right_hq,
            camera_frames=camera_frames,
            prompt=prompt,
        )

        try:
            action, _info = policy.get_action(observation)
        except Exception as exc:
            print(f"[live-VLA] inference error: {exc}", flush=True)
            _post_safe_idle_chunk(
                chunk,
                state=state,
                last_inference_ms=0.0,
                log_line="[live-VLA] inference error → zero-token safe chunk posted",
            )
            time.sleep(0.05)
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # Action shape: (B=1, T_horizon, D). Drop the batch axis.
        token = np.asarray(action["motion_token"], dtype=np.float32)[0]   # (T, 64)
        left = np.asarray(action["left_hand_joints"], dtype=np.float32)[0]  # (T, 10)
        right = np.asarray(action["right_hand_joints"], dtype=np.float32)[0]  # (T, 10)
        horizon_ok = (
            token.ndim == 2
            and token.shape[1] == SONIC_MOTION_TOKEN_DIM
            and left.ndim == 2
            and right.ndim == 2
            and left.shape[1] == DEFAULT_HAND_DOF
            and right.shape[1] == DEFAULT_HAND_DOF
            and left.shape[0] == token.shape[0]
            and right.shape[0] == token.shape[0]
            and np.isfinite(token).all()
            and np.isfinite(left).all()
            and np.isfinite(right).all()
        )
        if not horizon_ok:
            # Diagnostic detail: surface WHICH of the 9 horizon_ok checks
            # tripped so the operator doesn't have to instrument the bridge
            # to debug "100% bad chunks" runs. Cheap (~few hundred ns of
            # numpy reductions per failure) and only fires on the safety
            # path. Print shapes + finite counts + value ranges of the
            # finite slice (printing min/max of an all-NaN array is
            # useless).
            def _safe_range(a: np.ndarray) -> str:
                finite_mask = np.isfinite(a)
                if not finite_mask.any():
                    return "all non-finite"
                sub = a[finite_mask]
                return f"min={float(sub.min()):.3g} max={float(sub.max()):.3g}"

            diag = (
                f"token=shape{tuple(token.shape)}/finite"
                f"{int(np.isfinite(token).sum())}of{token.size} "
                f"left=shape{tuple(left.shape)}/finite"
                f"{int(np.isfinite(left).sum())}of{left.size} "
                f"right=shape{tuple(right.shape)}/finite"
                f"{int(np.isfinite(right).sum())}of{right.size}"
            )
            ranges = (
                f"token[{_safe_range(token)}] "
                f"left[{_safe_range(left)}] "
                f"right[{_safe_range(right)}]"
            )
            print(
                f"[live-VLA] invalid policy output ({diag}) "
                f"→ zero-token safe chunk posted",
                flush=True,
            )
            print(f"[live-VLA]   ranges: {ranges}", flush=True)

            # Bounded npz dump of the first few bad chunks so we can
            # re-inspect offline without re-running the whole live stack.
            # The successful-chunk dumper at line ~1602 keys off
            # `chunk_NNNNN.npz`; we use a separate `BAD_chunk_NNNN.npz`
            # prefix so the postmortem tools (e.g. inspect_vla_chunks)
            # don't try to parse these as valid chunks.
            if dump_dir_path is not None and n_bad_chunks < _MAX_BAD_CHUNK_DUMPS:
                try:
                    np.savez(
                        dump_dir_path / f"BAD_chunk_{n_bad_chunks:04d}.npz",
                        token=token,
                        left=left,
                        right=right,
                        elapsed_ms=np.array([elapsed_ms], dtype=np.float32),
                    )
                except Exception as exc:
                    print(
                        f"[live-VLA]   (warn) BAD_chunk dump failed: {exc}",
                        flush=True,
                    )
            n_bad_chunks += 1
            _post_safe_idle_chunk(
                chunk,
                state=state,
                last_inference_ms=float(elapsed_ms),
            )
            time.sleep(0.05)
            continue

        # Chunk ``body_pose`` is informational only — kept on the chunk so
        # ``--dump-chunks-dir`` snapshots include the live observation
        # alongside the predicted token. The publisher does NOT forward
        # this back over the wire; ``joint_pos_mj`` on ``pose`` is always
        # the trained ``DEFAULT_STAND_POSE`` so the C++ deploy's SONIC
        # tracker has a stable setpoint (mirroring the live observation
        # back as the setpoint zeroes the tracking error and the robot
        # falls under gravity — see _publisher docstring).
        body_pose = body_q_mj.astype(np.float32, copy=True)

        chunk.post(
            token=token,
            left_hand=left,
            right_hand=right,
            body_pose=body_pose,
            last_inference_ms=elapsed_ms,
        )
        n_inferences += 1
        if dump_dir_path is not None and (n_inferences - 1) % max(dump_chunks_every, 1) == 0:
            try:
                out_path = dump_dir_path / f"chunk_{n_inferences - 1:05d}.npz"
                # Save one ``view_<key>`` slot per camera frame the
                # observation carried. The ghost path keeps the legacy
                # ``view_ego_view`` key, the real-camera path saves one
                # ``view_stereo_left`` / ``view_stereo_right`` slot each.
                camera_arrays = {
                    f"view_{k}": v.astype(np.uint8)
                    for k, v in camera_frames.items()
                }
                (
                    wire_jpos,
                    wire_left,
                    wire_right,
                    raw_jpos,
                    raw_delta_idle,
                    wire_delta_idle,
                    wire_delta_body,
                    wire_delta_hand,
                    wire_chunk_id,
                    wire_chunk_step,
                ) = wire_debug.snapshot()
                np.savez_compressed(
                    out_path,
                    token=token,
                    left_hand=left,
                    right_hand=right,
                    body_q_mj=body_q_mj.astype(np.float32),
                    base_quat_wxyz=base_quat_wxyz.astype(np.float32),
                    left_hand_q_obs=left_hq.astype(np.float32),
                    right_hand_q_obs=right_hq.astype(np.float32),
                    wire_joint_pos_mj=wire_jpos.astype(np.float32),
                    wire_left_hand=wire_left.astype(np.float32),
                    wire_right_hand=wire_right.astype(np.float32),
                    raw_joint_pos_mj=raw_jpos.astype(np.float32),
                    raw_delta_idle_rad=np.array([raw_delta_idle], dtype=np.float32),
                    wire_delta_idle_rad=np.array([wire_delta_idle], dtype=np.float32),
                    wire_delta_body_rad=np.array([wire_delta_body], dtype=np.float32),
                    wire_delta_hand_rad=np.array([wire_delta_hand], dtype=np.float32),
                    wire_chunk_id=np.array([wire_chunk_id], dtype=np.int64),
                    wire_chunk_step=np.array([wire_chunk_step], dtype=np.int64),
                    elapsed_ms=np.array([elapsed_ms], dtype=np.float32),
                    revision=np.array([revision], dtype=np.int64),
                    n_inference=np.array([n_inferences - 1], dtype=np.int64),
                    wall_t_s=np.array([time.time()], dtype=np.float64),
                    **camera_arrays,
                )
            except Exception as exc:
                print(f"[live-VLA] chunk dump failed: {exc}", flush=True)
        if verbose:
            print(
                f"[live-VLA] inference #{n_inferences:04d}  "
                f"elapsed={elapsed_ms:6.1f} ms  "
                f"|token[0]|={float(np.linalg.norm(token[0])):.3f}  "
                f"|left[0]|={float(np.linalg.norm(left[0])):.3f}  "
                f"horizon={token.shape[0]}",
                flush=True,
            )

        # Pace ourselves to honour --inference-min-period-s. The publisher
        # walks the chunk at 50 Hz so a horizon=40 chunk represents 0.8 s
        # of motion; running inference faster than that means each new
        # chunk yanks the publisher back to step 0 mid-chunk and the
        # remaining steps are discarded (visible as a 1/min_period_s-Hz
        # spike pattern in the video).
        #
        # Loop the sleep in <=100 ms slices so we stay responsive to
        # stop_event during long pauses (the previous single-shot sleep
        # had a hard-coded 0.5 s cap which silently broke any min_period_s
        # > 0.5 s -- the L3 fix was a no-op as a result).
        if min_period_s > 0:
            while not stop_event.is_set():
                slack = min_period_s - (time.monotonic() - t0)
                if slack <= 0:
                    break
                time.sleep(min(slack, 0.1))

    if ghost_provider is not None:
        try:
            ghost_provider.close()
        except Exception:
            pass


def _publisher(
    *,
    pub_sock: zmq.Socket,
    topic: str,
    rate_hz: float,
    chunk: _LatestChunk,
    state: _LatestState,
    duration_s: float,
    stop_event: threading.Event,
    protocol_version: int = 4,
    print_every: int = 50,
    idle_loop: Optional[_IdleStandLoop] = None,
    silent_wire: bool = False,
    pose_decoder: Optional[Any] = None,
    ramp_in_ticks: int = 75,
    target_lpf_hz: float = 2.0,
    future_lpf_hz: float = 2.0,
    hand_lpf_hz: float = 1.0,
    hand_chunk_blend_ticks: int = 30,
    max_hand_step: float = 0.08,
    max_wire_dev_from_body: float = 0.18,
    max_wire_step: float = 0.035,
    chunk_blend_ticks: int = 40,
    max_action_il: float = 8.0,
    decode_delay_ticks: int = 150,
    body_mode: VlaBodyMode = VlaBodyMode.MANIPULATION,
    body_mode_control_file: str = "",
    decoder_loaded: bool = False,
    freeze_groups_override: str = "",
    wire_debug: Optional[_LatestWireDebug] = None,
    vla_control_signal: Optional[_VlaControlSignal] = None,
    cold_restart_hold_ticks: int = 25,
    handoff_max_hold_ticks: int = 200,
    handoff_max_wire_step: float = 0.012,
    handoff_step_ramp_ticks: int = 250,
    # 2026-06-10 follow-up 11: closed-loop tracking feedback on the
    # wire step cap. When enabled, the bridge reads x2_debug's
    # measured arm-joint positions + velocities each tick and
    # per-joint-throttles the wire step so it never outpaces the
    # actuator's actual response. Eliminates the open-loop
    # sensitivity to inference jitter / battery sag / motor temp
    # that drove the 2026-06-10 PM oscillation incident. All gated
    # on ``tracking_feedback_enabled`` (default False = byte-
    # identical to legacy scalar clamp); when enabled but proprio
    # is stale (>tracking_stale_ms since last x2_debug update) the
    # bridge falls back to the scalar clamp automatically. See
    # ``_apply_tracking_feedback`` docstring + the 2026-06-10
    # closed-loop wire milestone for the full feedback law.
    tracking_feedback_enabled: bool = False,
    tracking_soft_rad: float = 0.15,
    tracking_hard_rad: float = 0.40,
    tracking_vel_margin: float = 1.5,
    tracking_vel_floor_rad_tick: float = 0.01,
    tracking_stale_ms: int = 100,
) -> int:
    """Thread C (= main thread): publish action[step] at ``rate_hz``.

    Always publishes *something*, even before the first inference completes.

    Wire-content contract (matches ``mock_vla_publish_stand_token.py``
    plus the v5 future-window the heuristic planner emits via
    :func:`gear_sonic.utils.planner.state_machine.build_pose_payload`):

    * ``joint_pos_mj`` = stand setpoint. When ``idle_loop`` is provided
      we replay the planner's ``idle_stand`` primitive frame-by-frame
      so the wire is byte-equivalent to ``--planner-only`` (verified
      empirically 2026-05-14: ``idle_stand[0]``'s waist-yaw differs
      from ``DEFAULT_STAND_POSE`` by ~33 deg, and the policy was
      trained against the clip's per-frame DOF jitter -- without the
      clip the robot stabilises at a ~25 deg lean even with a v5
      window). When ``idle_loop`` is None we fall back to
      ``DEFAULT_STAND_POSE``, which keeps the deploy from outright
      falling but is NOT recommended.
      The C++ deploy uses ``joint_pos_mj`` as the **SONIC tracker
      reference**, NOT as a feedback channel. Mirroring the live
      ``body_q_mj`` from ``x2_debug`` here makes
      ``reference == observation`` -> tracking error 0 -> no corrective
      torque -> the robot falls under gravity (verified empirically
      2026-05-14).
    * ``root_quat_xyzw`` = the matching clip frame's quat (or identity
      when no clip is loaded).
    * ``motion_token`` / ``left_hand_joints`` / ``right_hand_joints`` =
      latest chunk step (zeros until the first inference, or always zero
      under ``--no-policy``).
    * ``joint_pos_mj_future`` / ``root_quat_xyzw_future`` /
      ``joint_vel_mj_future`` / ``frame_index_future`` /
      ``future_dt_s`` = 9 strictly-future slots at ``DT_FUTURE_REF`` =
      0.1 s spacing. With ``idle_loop`` we sample the clip at
      ``_FUTURE_STEP_TICKS`` (= 5 ticks) past the current frame; without
      it we ship 9 copies of the static stand pose. Both shapes promote
      the deploy to v5 mode. Without ANY future window the deploy's
      ``ZmqPoseInputSource::Sample()`` falls back to the legacy
      single-frame path, which pins all 10 future tokens at the current
      pose; the X2 policy was trained with motion-bearing windows and
      produces saturated raw actions (act_clip ~100 % of ticks,
      max_pre_clip > action_clip = 20.0) when fed the legacy fallback,
      manifesting as a slow gravity tilt after the elastic band
      releases. (Verified empirically 2026-05-14 -- before the window:
      grav_z drifted -1.00 -> -0.95 in 9 s under ``--vla-no-policy``.)

    When ``x2_debug`` itself is stale we keep this same idle wire (the
    fallback branch is essentially a no-op now since the fresh branch is
    already a stable setpoint).
    """
    period = 1.0 / max(rate_hz, 1e-6)
    next_tick = time.monotonic()
    deadline = float("inf") if duration_s <= 0 else time.monotonic() + duration_s

    last_chunk_id = -1
    chunk_step = 0
    horizon = 40
    tick = 0

    # --- Tracking feedback (2026-06-10 follow-up 11) -----------------
    # Per-tick throttle count for telemetry: how many of the 14 arm
    # joints did the closed-loop feedback throttle below 50% of the
    # base step cap this tick. Stays 0 when the feature is disabled
    # OR when proprio is stale -- both cases also fall back to the
    # scalar clamp downstream so the wire is byte-identical to the
    # legacy path. See ``_apply_tracking_feedback`` for the law.
    tracking_stale_s = float(max(tracking_stale_ms, 0)) / 1000.0
    tracking_arm_count = len(_ARM_JOINT_INDICES)
    last_tracking_throttle: int = 0

    # --- Bridge-side proprio assembly state ---------------------------
    # The SONIC pose decoder was trained against the IsaacLab
    # ProprioceptionBuffer (10-frame history of base_ang_vel /
    # jpos_rel / jvel / last_action / gravity = 990 floats). The
    # previous v0 placeholder (zeros) made the decoder OOD and pushed
    # the wire to ~0.46 rad from idle on EVERY chunk (the on-robot
    # vibration seen 2026-06-07). We assemble the real vector each
    # tick from the x2_debug stream and the bridge's own
    # ``last_action_il`` echo. See sonic_decoder_proprio.py for the
    # layout.
    decoder_buf: Optional[Any] = None
    last_action_il = np.zeros(NUM_BODY_DOFS, dtype=np.float32)
    decoder_default_il: Optional[np.ndarray] = None
    proprio_decode_log_done = False
    if pose_decoder is not None:
        from gear_sonic.utils.teleop.sonic_decoder_proprio import (
            ProprioceptionBuffer,
            build_proprio_990,
            default_angles_il,
        )
        decoder_buf = ProprioceptionBuffer()
        decoder_default_il = default_angles_il()

    # --- Ramp-in + LPF state ------------------------------------------
    # ``ramp_remaining`` is decremented per tick from
    # ``ramp_in_ticks`` after the first successful VLA decode lands;
    # while it's >0 the wire ships ``lerp(ramp_from, decoded, alpha)``
    # for both the current slot and the future window so the C++
    # tokeniser sees a coherent slowly-rising trajectory rather than
    # a step jump on every joint.
    # ``lpf_state`` is the one-pole low-pass output state for the
    # current ``joint_pos_mj`` slot; reset to the freshly transitioned
    # pose when leaving idle, then continuously updated.
    ramp_from: Optional[np.ndarray] = None
    ramp_remaining: int = 0
    lpf_state: Optional[np.ndarray] = None
    lpf_future_state: Optional[np.ndarray] = None
    lpf_alpha = _lpf_alpha_from_hz(target_lpf_hz, period)
    future_lpf_alpha = _lpf_alpha_from_hz(future_lpf_hz, period)
    hand_lpf_alpha = _lpf_alpha_from_hz(hand_lpf_hz, period)
    hand_lpf_left: Optional[np.ndarray] = None
    hand_lpf_right: Optional[np.ndarray] = None
    prev_wire_left: Optional[np.ndarray] = None
    prev_wire_right: Optional[np.ndarray] = None
    last_wire_hand_chunk_id = -1
    hand_blend_from_left: Optional[np.ndarray] = None
    hand_blend_from_right: Optional[np.ndarray] = None
    hand_chunk_blend_remaining = 0
    decoded_was_active = False  # tracks the previous tick's mode
    prev_wire_jpos: Optional[np.ndarray] = None
    last_wire_chunk_id = -1
    chunk_blend_from: Optional[np.ndarray] = None
    chunk_blend_remaining = 0
    yaw_rebase_logged = False
    bootstrap_publish_logged = False
    bootstrap_first_publish_logged = False
    # Manual-takeover (vla_control) cold-restart bookkeeping. When the
    # proxy emits ``override_engaged`` the bridge suppresses decoded
    # chunks and ships the operator's current measured pose so the
    # wire stays alive (the proxy ignores it but a SAFE_IDLE trip is
    # avoided if the proxy itself were to fall over mid-takeover).
    # ``override_released`` arms a cold restart which on the next tick
    # clears all smoothing state, bumps the chunk-id baseline so any
    # stale chunk decoded against pre-override observations is
    # ignored, and starts a brief "hold at operator pose" window to
    # bridge the proxy's HOLD -> LIVE handoff without a step change.
    #
    # 2026-06-10 follow-up: ``operator_hold_pose`` caches the body +
    # hand joints the proxy snapshotted at the moment of release. The
    # hold window uses these instead of x2_debug's measured pose to
    # avoid the visible "pose reset" on ARM_MANIPULATION -> LOCOMOTION
    # handoff -- measured lags actuated by motor / contact / gravity
    # sag, so the wire stepped from operator-commanded to
    # measured-pose for the 25-tick hold then ramped to VLA, which
    # produced two visible discontinuities. Falls back to measured
    # when the release event carries no payload (legacy proxy /
    # smoke tests with --override-engage-motion-ticks 0).
    override_active_now = False
    cold_restart_chunk_baseline = -1
    hold_at_measured_remaining = 0
    cold_restart_log_done_at_tick = -1
    operator_hold_pose: Optional[Dict[str, np.ndarray]] = None
    # 2026-06-10 (PM follow-up 3): smooth-handoff guard. The 25-tick
    # cold-restart hold is the MINIMUM dwell at the operator pose;
    # ``cold_restart_awaiting_first_chunk`` keeps the wire pinned at
    # the operator pose AFTER the minimum hold expires until the
    # first eligible decoded chunk (``chunk_id > cold_restart_chunk_
    # baseline``) arrives. Without this, the post-hold tick sees no
    # eligible chunk yet (inference cadence ~15 Hz vs wire 50 Hz means
    # up to 3 wire ticks between chunks; first-chunk decode time is
    # ~480 ms), and the publisher falls through to
    # ``cur_jpos = idle_loop_pose`` which is a hard step from the
    # operator's hand-off pose to the idle_stand clip.
    # ``cold_restart_max_hold_remaining`` is the SAFETY CAP that
    # forces a release after ``handoff_max_hold_ticks`` even if the
    # decoder never produces an eligible chunk (e.g. zero-token /
    # decoder crash / proprio starvation). When the safety cap fires
    # we fall through to the idle wire with a clear warning so
    # operators don't silently sit at the operator pose forever.
    cold_restart_awaiting_first_chunk = False
    cold_restart_max_hold_remaining = 0
    # 2026-06-10 follow-up 6: post-handoff per-tick wire-step ramp.
    # ``max_wire_step`` is a per-element rate clamp on the wire. The
    # default 0.035 rad/tick at 50 Hz = 1.75 rad/s per joint; with
    # ~31 joints all moving toward a VLA chunk that's ~3.7 rad away
    # from the operator pose, the L_inf bound becomes a coordinated
    # whole-body swing that the operator reports as a slam (terminal
    # 2 of session 73f3d2a2 at tick 16000: raw_Δ=3.720, body_Δ=0.247
    # = 14deg of tracking error sustained over the ramp). We instead
    # start the wire at ``handoff_max_wire_step`` (default 0.012
    # rad/tick = ~36deg/s per joint, ~3x slower) right after the
    # handoff and LINEARLY ramp back to ``max_wire_step`` over
    # ``handoff_step_ramp_ticks`` ticks (default 250 = 5 s @ 50 Hz).
    # The countdown arms when ``cold_restart_awaiting_first_chunk``
    # transitions to False on the success path; the safety-cap path
    # also arms it so the post-cap ramp from operator pose toward
    # idle_stand uses the same slow step.
    handoff_step_remaining = 0
    active_body_mode = body_mode
    _freeze_idx, decode_body = _body_mode_wire_settings(
        active_body_mode,
        decoder_loaded=decoder_loaded,
        freeze_groups_override=freeze_groups_override,
    )

    while not stop_event.is_set() and time.monotonic() < deadline:
        if body_mode_control_file:
            polled = _read_body_mode_control_file(body_mode_control_file)
            if polled is not None and polled is not active_body_mode:
                active_body_mode = polled
                _freeze_idx, decode_body = _body_mode_wire_settings(
                    active_body_mode,
                    decoder_loaded=decoder_loaded,
                    freeze_groups_override=freeze_groups_override,
                )
                ramp_from = None
                ramp_remaining = 0
                lpf_state = None
                lpf_future_state = None
                chunk_blend_remaining = 0
                hand_lpf_left = None
                hand_lpf_right = None
                hand_chunk_blend_remaining = 0
                print(
                    f"[live-VLA] body mode -> {active_body_mode.value} "
                    f"(decode_body={decode_body}, freeze={_freeze_idx.tolist()})",
                    flush=True,
                )
        body_q_mj_now, base_quat_now, _, _, _, deploy_fresh = state.snapshot()
        # 2026-06-10 follow-up 11: read measured velocities for the
        # closed-loop tracking feedback. Cheap second snapshot --
        # both reads acquire ``state.cv`` so the worst case is one
        # tick of staleness between q and dq, well below the 50 Hz
        # publish cadence. ``staleness_s`` is computed from the same
        # monotonic clock the state thread uses so the feedback can
        # decide whether to trust the snapshot or fall back to the
        # scalar clamp.
        body_dq_mj_now, _base_ang_vel_now = state.snapshot_velocities()
        with state.cv:
            tracking_staleness_s = (
                time.monotonic() - state.last_update_monotonic
                if state.received_any
                else float("inf")
            )
        token, left, right, _body_pose_chunk, chunk_id = chunk.read()
        horizon = int(token.shape[0])
        if chunk_id != last_chunk_id:
            chunk_step = 0
            last_chunk_id = chunk_id

        # ---- vla_control: manual-takeover signal processing ---------
        # Drain the signal once per tick. ``override_active_now``
        # gates the decode path (we never want to ship VLA chunks
        # while the operator is in control). A cold-restart pending
        # flag (set on the released edge) clears all per-tick
        # smoothing state and arms the "hold at measured pose"
        # bridging window. Both no-ops when vla_control is disabled.
        if vla_control_signal is not None:
            override_active_now, _ = vla_control_signal.snapshot()
            pending, pending_release_pose = (
                vla_control_signal.consume_cold_restart()
            )
            if pending:
                ramp_from = None
                ramp_remaining = 0
                lpf_state = None
                lpf_future_state = None
                hand_lpf_left = None
                hand_lpf_right = None
                chunk_blend_from = None
                chunk_blend_remaining = 0
                hand_blend_from_left = None
                hand_blend_from_right = None
                hand_chunk_blend_remaining = 0
                last_wire_chunk_id = -1
                last_wire_hand_chunk_id = -1
                decoded_was_active = False
                # Cache the operator-pose snapshot so the next 25
                # ticks hold the wire at exactly THAT pose (not
                # x2_debug's lagged measured pose). Seed prev_wire_*
                # from the same snapshot so the post-hold chunk
                # blend interpolates FROM operator pose TO the first
                # decoded chunk -- otherwise prev_wire_* would be
                # None and the chunk blend would blend from zeros
                # (visible step on the body / fingers).
                operator_hold_pose = pending_release_pose
                if (
                    operator_hold_pose is not None
                    and "joint_pos_mj" in operator_hold_pose
                ):
                    prev_wire_jpos = operator_hold_pose[
                        "joint_pos_mj"
                    ].astype(np.float32, copy=True)
                else:
                    prev_wire_jpos = None
                if (
                    operator_hold_pose is not None
                    and "left_hand_joints" in operator_hold_pose
                ):
                    prev_wire_left = operator_hold_pose[
                        "left_hand_joints"
                    ].astype(np.float32, copy=True)
                else:
                    prev_wire_left = None
                if (
                    operator_hold_pose is not None
                    and "right_hand_joints" in operator_hold_pose
                ):
                    prev_wire_right = operator_hold_pose[
                        "right_hand_joints"
                    ].astype(np.float32, copy=True)
                else:
                    prev_wire_right = None
                # Pin the chunk-id baseline so any in-flight chunk
                # decoded against pre-override observations is
                # ignored; only chunks produced AFTER the operator
                # released the wire are eligible to be decoded onto
                # the wire post-restart.
                cold_restart_chunk_baseline = int(chunk_id)
                hold_at_measured_remaining = max(
                    int(cold_restart_hold_ticks), 0
                )
                # Arm the smooth-handoff guard. The wire stays at the
                # operator pose for at LEAST ``cold_restart_hold_ticks``
                # (the minimum dwell, set above) and at MOST
                # ``handoff_max_hold_ticks`` (the safety cap, set here);
                # in between, the gate releases as soon as the first
                # eligible chunk arrives (``chunk_id > baseline``). Set
                # ``handoff_max_hold_ticks = 0`` to disable the
                # await-first-chunk behaviour entirely (legacy 2026-06-10
                # behaviour: snap to idle on hold expiry).
                cold_restart_awaiting_first_chunk = (
                    int(handoff_max_hold_ticks) > 0
                )
                cold_restart_max_hold_remaining = max(
                    int(handoff_max_hold_ticks), 0
                )
                cold_restart_log_done_at_tick = tick
                hold_target = (
                    "operator's last commanded pose (body"
                    + ("+left_hand" if (
                        operator_hold_pose is not None
                        and "left_hand_joints" in operator_hold_pose
                    ) else "")
                    + ("+right_hand" if (
                        operator_hold_pose is not None
                        and "right_hand_joints" in operator_hold_pose
                    ) else "")
                    + ")"
                ) if (
                    operator_hold_pose is not None
                    and "joint_pos_mj" in operator_hold_pose
                ) else "x2_debug measured pose (legacy fallback)"
                print(
                    f"[live-VLA] cold-restart fired tick={tick} "
                    f"baseline_chunk={cold_restart_chunk_baseline} "
                    f"min_hold_ticks={hold_at_measured_remaining} "
                    f"max_hold_ticks={cold_restart_max_hold_remaining}; "
                    f"will hold wire at {hold_target} "
                    f"until first eligible chunk (chunk_id > "
                    f"{cold_restart_chunk_baseline}) decodes, "
                    f"capped by max-hold safety",
                    flush=True,
                )
        else:
            override_active_now = False

        step = min(chunk_step, horizon - 1)
        if idle_loop is not None:
            cur_jpos, cur_quat = idle_loop.current(tick)
            future_fields = idle_loop.future_payload_fields(
                tick=tick, base_frame_index=tick,
            )
        else:
            cur_jpos = _default_stand_body_pose_f32()
            cur_quat = _identity_quat_xyzw()
            future_fields = _idle_future_payload_fields(base_frame_index=tick)
        # Capture the unmodified idle pose for the ramp-in baseline
        # BEFORE the decoder potentially overwrites cur_jpos.
        idle_baseline = np.asarray(cur_jpos, dtype=np.float32).copy()

        # ---- vla_control: hold at operator / measured pose ---------
        # When override is ACTIVE or we're in the cold-restart bridge
        # window, replace the idle-clip pose with a stationary hold
        # pose AND repeat it across the future window. Two sources
        # in priority order:
        #   1. ``operator_hold_pose["joint_pos_mj"]`` -- snapshotted
        #      by the proxy at the instant override_released fired.
        #      Matches what the deploy was being COMMANDED, so the
        #      handoff is bit-identical at the proxy / deploy seam.
        #   2. ``body_q_mj_now`` (x2_debug measured pose) -- legacy
        #      fallback when the release event carried no payload
        #      (older proxy / smoke tests with engage_motion_ticks 0).
        #      Visibly lags the commanded pose under motor / contact
        #      / gravity sag, which is the "pose reset" the operator
        #      reports when the bridge takes back from teleop.
        # The proxy ignores these frames during OVERRIDE; during the
        # post-release HOLD bridging window the proxy is in
        # STATE_HOLD replaying the operator frame anyway -- this
        # just makes sure our frames don't compete with them.
        # Decoded chunks are gated off via ``in_hold_window`` below.
        # ``cold_restart_awaiting_first_chunk`` extends the hold past
        # the minimum 25-tick dwell until the first eligible chunk
        # arrives (``chunk_id > cold_restart_chunk_baseline``); the
        # safety cap (``cold_restart_max_hold_remaining``) bounds
        # that wait. Resolve the await-state BEFORE computing
        # ``in_hold_window`` so a chunk that arrives exactly at the
        # transition tick releases the hold this same tick (else
        # we'd publish one extra "stuck at operator pose" frame
        # after the chunk became available).
        if (
            cold_restart_awaiting_first_chunk
            and not override_active_now
            and hold_at_measured_remaining == 0
        ):
            # 2026-06-10 follow-up 5: original gate was ``chunk_id >
            # baseline and chunk_id > 0``, which released the wire on
            # ANY new chunk -- even chunks whose token vector was zero.
            # Empirical: terminal 2 of session 73f3d2a2 showed
            # every chunk in the run with ``|token|=0.000 |left|=0.000
            # idle-pose``, the gate released as soon as
            # ``chunk_id=503 > baseline=499``, the decoder below
            # immediately failed its OWN ``np.linalg.norm(token[step])
            # > 1e-3`` guard, ``cur_jpos`` fell through to
            # ``idle_loop.current(tick)`` (= idle_stand pose), and the
            # wire snapped from operator pose to idle_stand pose. The
            # operator reported "the hand slammed into the table".
            # Mirror the decoder's token-magnitude guard here so the
            # gate stays armed until VLA is actually producing usable
            # output -- if VLA never produces useful tokens, the
            # safety cap below trips after ``handoff_max_hold_ticks``
            # and the always-on per-tick wire rate clamp (fix 2 below)
            # ramps the wire from operator pose to idle gracefully
            # instead of snapping.
            current_token_norm = float(np.linalg.norm(token[step]))
            first_eligible_chunk_ready = (
                chunk_id > cold_restart_chunk_baseline
                and chunk_id > 0
                and current_token_norm > 1e-3
            )
            if first_eligible_chunk_ready:
                cold_restart_awaiting_first_chunk = False
                # 2026-06-10 follow-up 6: arm the post-handoff
                # per-tick wire-step ramp so the wire walks from
                # operator pose toward the VLA target slowly for
                # the first few seconds, then accelerates back to
                # the steady-state ``max_wire_step``. Set unconditionally
                # because the success path's L_inf delta to VLA can
                # be ~3.7 rad (terminal 2 / 13:05 run) -- a single-
                # tick 75-step LPF ramp at full max_wire_step still
                # produced visible coordinated whole-body swings.
                handoff_step_remaining = max(int(handoff_step_ramp_ticks), 0)
                print(
                    f"[live-VLA] cold-restart handoff: first eligible "
                    f"chunk decoded (chunk_id={chunk_id} > baseline="
                    f"{cold_restart_chunk_baseline}, "
                    f"|token|={current_token_norm:.3f} > 1e-3); "
                    f"releasing wire hold at tick={tick}, ramping into "
                    f"VLA from operator pose (slow-step window: "
                    f"{handoff_step_remaining} ticks from "
                    f"{handoff_max_wire_step:.3f} -> "
                    f"{max_wire_step:.3f} rad/tick)",
                    flush=True,
                )
            elif cold_restart_max_hold_remaining > 0:
                cold_restart_max_hold_remaining -= 1
                if cold_restart_max_hold_remaining == 0:
                    cold_restart_awaiting_first_chunk = False
                    # Same slow-step window on the safety-cap path:
                    # the wire still has to walk from operator pose
                    # toward idle_stand_pose, and that delta can be
                    # comparable in magnitude. Avoids snap-to-idle
                    # being slower than snap-to-VLA.
                    handoff_step_remaining = max(int(handoff_step_ramp_ticks), 0)
                    print(
                        f"[live-VLA] WARNING: cold-restart handoff "
                        f"safety cap reached at tick={tick} "
                        f"(handoff_max_hold_ticks elapsed; latest "
                        f"chunk_id={chunk_id} baseline="
                        f"{cold_restart_chunk_baseline} "
                        f"|token|={current_token_norm:.3f}); releasing "
                        f"wire to idle with slow-step ramp ("
                        f"{handoff_step_remaining} ticks from "
                        f"{handoff_max_wire_step:.3f} -> "
                        f"{max_wire_step:.3f} rad/tick). Check decoder "
                        f"(stuck inference? proprio starvation? "
                        f"zero-token chunks?).",
                        flush=True,
                    )
        in_hold_window = (
            override_active_now
            or hold_at_measured_remaining > 0
            or cold_restart_awaiting_first_chunk
        )
        if in_hold_window and deploy_fresh:
            if (
                operator_hold_pose is not None
                and "joint_pos_mj" in operator_hold_pose
            ):
                hold_jpos = operator_hold_pose["joint_pos_mj"].astype(
                    np.float32, copy=True
                )
            else:
                hold_jpos = np.asarray(
                    body_q_mj_now, dtype=np.float32
                ).copy()
            cur_jpos = hold_jpos
            if "joint_pos_mj_future" in future_fields:
                future_fields = {
                    **future_fields,
                    "joint_pos_mj_future": np.broadcast_to(
                        hold_jpos[None, :],
                        (_NUM_FUTURE_SLOTS, NUM_BODY_DOFS),
                    ).copy(),
                    "joint_vel_mj_future": np.zeros(
                        (_NUM_FUTURE_SLOTS, NUM_BODY_DOFS),
                        dtype=np.float32,
                    ),
                }
            idle_baseline = hold_jpos.copy()
            if (
                hold_at_measured_remaining > 0
                and not override_active_now
            ):
                hold_at_measured_remaining -= 1
                # Keep operator_hold_pose alive while the smooth-handoff
                # guard is still waiting for the first eligible chunk;
                # the ramp-init block below seeds ``ramp_from`` and
                # ``lpf_state`` from this snapshot so the first
                # decoded tick interpolates operator->VLA instead of
                # snapping to ``idle_baseline``. Only clear it once
                # BOTH the minimum-dwell countdown AND the await flag
                # have released the wire -- after that any stale
                # snapshot would belong to a previous handoff cycle.
                if (
                    hold_at_measured_remaining == 0
                    and not cold_restart_awaiting_first_chunk
                ):
                    operator_hold_pose = None

        # --- Live 990-D proprio assembly --------------------------------
        # Even when we won't actually decode this tick (cold start,
        # deploy stale, no decoder, etc.) we still want to APPEND to
        # the history buffer so by the time decoding kicks in the 10
        # frames are real observations rather than the
        # broadcast-primed zeros.
        proprio_990 = None
        if decoder_buf is not None:
            body_dq_mj_now, base_ang_vel_now = state.snapshot_velocities()
            try:
                proprio_990 = build_proprio_990(
                    decoder_buf,
                    body_q_mj=body_q_mj_now,
                    body_dq_mj=body_dq_mj_now,
                    base_quat_wxyz=base_quat_now,
                    base_ang_vel=base_ang_vel_now,
                    last_action_il=last_action_il,
                    default_angles_il_cached=decoder_default_il,
                )
            except Exception as exc:  # noqa: BLE001
                if not proprio_decode_log_done:
                    print(
                        f"[live-VLA] WARN: live proprio assembly failed "
                        f"({exc!r}); falling back to zero proprio for "
                        f"the rest of this run.",
                        flush=True,
                    )
                    proprio_decode_log_done = True
                proprio_990 = _PROPRIO_ZERO_990

        # SONIC token decoder: decode chunk[step] + 9 future steps to a
        # body trajectory and publish that as joint_pos_mj instead of
        # idle_stand. Only kicks in when (a) a decoder was provided, (b)
        # we have a fresh deploy (so a real chunk is on hand), and (c)
        # the current chunk's token magnitude is non-trivial -- the
        # cold-start chunk_id=0 is all zeros and decoding zeros yields
        # the decoder's "stand" intent which is fine but slightly OOD;
        # falling back to idle_stand_loop in that window matches the
        # --vla-no-policy stable wire content. The chunk_id check
        # avoids any ambiguity.
        decoded_now = None
        proprio_ready = (
            decoder_buf is None or getattr(decoder_buf, "primed", False)
        )
        if (
            pose_decoder is not None
            and decode_body
            and deploy_fresh
            and tick >= decode_delay_ticks
            and proprio_ready
            and chunk_id > 0
            and chunk_id > cold_restart_chunk_baseline
            and not in_hold_window
            and np.linalg.norm(token[step]) > 1e-3
        ):
            decoded = _build_vla_decoded_pose_payload(
                decoder=pose_decoder,
                proprio_990=proprio_990 if proprio_990 is not None else _PROPRIO_ZERO_990,
                token_chunk=token,
                chunk_step=step,
                horizon=horizon,
                base_frame_index=tick,
            )
            if decoded is not None:
                decoded_now, future_fields, action_il_now = decoded
                # Feed the just-emitted residual back as next tick's
                # ``last_action_il`` -- mirrors the C++ deploy's
                # ``last_action_il_ = action_il`` line and is what the
                # ``last_action`` proprioception term was trained on.
                if max_action_il > 0.0:
                    action_il_now = np.clip(
                        action_il_now,
                        -max_action_il,
                        max_action_il,
                    ).astype(np.float32, copy=False)
                last_action_il = action_il_now
                cur_jpos = decoded_now
                # cur_quat stays as the idle_loop / identity choice --
                # the SONIC body-only decoder doesn't predict root.

                # ---- B: ramp-in on idle -> VLA transition ----------
                # When the cold-restart handoff guard is still holding
                # the operator pose snapshot, seed the ramp + LPF + hand
                # LPF from THAT pose so the wire interpolates
                # operator -> decoded (smooth) instead of
                # idle_baseline -> decoded (visible idle-clip-shaped
                # bump at the seam). Once we use the snapshot here, we
                # also clear ``operator_hold_pose`` and the await flag
                # so subsequent ticks fall through to the regular
                # decoded path. The snapshot has already been
                # broadcast onto the wire for the full hold + await
                # window, so ramp progress 0 == prev_wire_jpos by
                # construction.
                handoff_seed_from = None
                if (
                    operator_hold_pose is not None
                    and "joint_pos_mj" in operator_hold_pose
                ):
                    handoff_seed_from = operator_hold_pose[
                        "joint_pos_mj"
                    ].astype(np.float32, copy=True)
                if not decoded_was_active and ramp_in_ticks > 0:
                    ramp_from = (
                        handoff_seed_from.copy()
                        if handoff_seed_from is not None
                        else idle_baseline.copy()
                    )
                    ramp_remaining = int(ramp_in_ticks)
                    # Reset LPF state so the filter doesn't lag the
                    # ramp; we re-prime it at the same seed pose as
                    # the ramp (operator pose on cold-restart handoff,
                    # idle_baseline on the normal idle->VLA transition).
                    if lpf_alpha > 0.0:
                        lpf_state = (
                            handoff_seed_from.copy()
                            if handoff_seed_from is not None
                            else idle_baseline.copy()
                        )
                    else:
                        lpf_state = None
                    lpf_future_state = None
                    # Hand LPF: when the handoff carries operator hands
                    # in the snapshot, seed both LPFs there so the first
                    # filtered finger frame is the operator's, not the
                    # in-flight VLA chunk's. Falls back to "let the LPF
                    # re-prime from the first hand sample" when no
                    # handoff snapshot is available (legacy idle->VLA
                    # transition).
                    if (
                        operator_hold_pose is not None
                        and "left_hand_joints" in operator_hold_pose
                    ):
                        hand_lpf_left = operator_hold_pose[
                            "left_hand_joints"
                        ].astype(np.float32, copy=True)
                    else:
                        hand_lpf_left = None
                    if (
                        operator_hold_pose is not None
                        and "right_hand_joints" in operator_hold_pose
                    ):
                        hand_lpf_right = operator_hold_pose[
                            "right_hand_joints"
                        ].astype(np.float32, copy=True)
                    else:
                        hand_lpf_right = None
                    # Snapshot was used to seed the ramp + LPFs; drop
                    # it now so subsequent cold restarts don't see a
                    # stale snapshot if the proxy ever omits the
                    # ``release_pose`` payload on the next release.
                    if handoff_seed_from is not None:
                        operator_hold_pose = None
                        cold_restart_awaiting_first_chunk = False
                        print(
                            f"[live-VLA] cold-restart handoff: ramp + "
                            f"LPF seeded from operator pose "
                            f"(ramp_ticks={int(ramp_in_ticks)}); "
                            f"VLA wire re-engaging from hand-off pose "
                            f"without idle-clip detour",
                            flush=True,
                        )

                if ramp_remaining > 0 and ramp_from is not None:
                    progress = float(ramp_in_ticks - ramp_remaining + 1) / float(ramp_in_ticks)
                    progress = min(max(progress, 0.0), 1.0)
                    ramped_now = (
                        ramp_from * (1.0 - progress) + decoded_now * progress
                    ).astype(np.float32)
                    # Lerp the future window with the same progress.
                    # Future slots represent t + 0.1..0.9 s; keeping
                    # the ramp coherent across them avoids the tokeniser
                    # seeing a sharp slope between the current slot
                    # and the first future slot.
                    jpos_future = future_fields["joint_pos_mj_future"]
                    ramped_future = (
                        ramp_from[None, :] * (1.0 - progress)
                        + jpos_future * progress
                    ).astype(np.float32)
                    # Recompute jvel_future as the finite-diff of the
                    # ramped trajectory so the deploy doesn't see a
                    # velocity profile that disagrees with the ramped
                    # positions.
                    jvel_future = _jvel_future_from_poses(ramped_now, ramped_future)
                    future_fields = {
                        **future_fields,
                        "joint_pos_mj_future": ramped_future,
                        "joint_vel_mj_future": jvel_future,
                    }
                    cur_jpos = ramped_now
                    ramp_remaining -= 1

                # ---- C: inter-chunk blend (before LPF) --------------
                if chunk_id != last_wire_chunk_id:
                    chunk_blend_from = (
                        prev_wire_jpos.copy()
                        if prev_wire_jpos is not None
                        else idle_baseline.copy()
                    )
                    chunk_blend_remaining = int(chunk_blend_ticks)
                    last_wire_chunk_id = chunk_id
                if chunk_blend_remaining > 0 and chunk_blend_from is not None:
                    progress = float(
                        chunk_blend_ticks - chunk_blend_remaining + 1
                    ) / float(max(chunk_blend_ticks, 1))
                    progress = min(max(progress, 0.0), 1.0)
                    cur_jpos = (
                        chunk_blend_from * (1.0 - progress)
                        + np.asarray(cur_jpos, dtype=np.float32) * progress
                    ).astype(np.float32)
                    chunk_blend_remaining -= 1

                # ---- D: one-pole LPF on body (current + future) -----
                cur_jpos, lpf_state = _lpf_vector(cur_jpos, lpf_state, lpf_alpha)
                if future_lpf_alpha > 0.0 and "joint_pos_mj_future" in future_fields:
                    jf = future_fields["joint_pos_mj_future"]
                    jf_smooth, lpf_future_state = _lpf_vector(
                        jf, lpf_future_state, future_lpf_alpha
                    )
                    future_fields = {
                        **future_fields,
                        "joint_pos_mj_future": jf_smooth,
                        "joint_vel_mj_future": _jvel_future_from_poses(
                            cur_jpos, jf_smooth
                        ),
                    }

                # ---- E: rate + anchor clamps (anti-jitter) ------------
                # 2026-06-10 follow-up 6: use the slow-step value
                # during the post-handoff ramp window. Linearly
                # interpolate handoff_max_wire_step (slow) -> max_wire_step
                # (normal) over handoff_step_ramp_ticks so the first
                # ticks after handoff move at ~handoff_max_wire_step
                # (0.012 rad/tick = ~36 deg/s/joint by default) and
                # the wire accelerates back to normal as the body
                # catches up. The countdown is decremented at the
                # end of this tick path so the FIRST tick after
                # release gets the slowest step (which is when
                # raw_Δ is largest and the body has the most to
                # cover).
                if handoff_step_remaining > 0:
                    ramp_progress = 1.0 - float(handoff_step_remaining) / float(
                        max(handoff_step_ramp_ticks, 1)
                    )
                    ramp_progress = min(max(ramp_progress, 0.0), 1.0)
                    effective_max_step = (
                        (1.0 - ramp_progress) * float(handoff_max_wire_step)
                        + ramp_progress * float(max_wire_step)
                    )
                    handoff_step_remaining -= 1
                else:
                    effective_max_step = float(max_wire_step)
                # 2026-06-10 follow-up 11: closed-loop tracking
                # feedback. When enabled AND proprio is fresh, the
                # scalar ``effective_max_step`` becomes the per-arm-
                # joint UPPER bound; each arm joint's individual cap
                # is then throttled by position error and velocity
                # margin via ``_apply_tracking_feedback``. Falls back
                # to the scalar clamp transparently when disabled or
                # when proprio is stale (>tracking_stale_ms).
                tracking_active = (
                    tracking_feedback_enabled
                    and deploy_fresh
                    and tracking_staleness_s <= tracking_stale_s
                )
                if tracking_active:
                    cap_per_joint, throttle_count = _apply_tracking_feedback(
                        cur_jpos,
                        prev_wire_jpos,
                        body_q_mj_now,
                        body_dq_mj_now,
                        base_max_step=effective_max_step,
                        soft_rad=tracking_soft_rad,
                        hard_rad=tracking_hard_rad,
                        vel_margin=tracking_vel_margin,
                        vel_floor_rad_tick=tracking_vel_floor_rad_tick,
                        dt_s=period,
                    )
                    cur_jpos = _clamp_vector_step_per_joint(
                        cur_jpos, prev_wire_jpos, cap_per_joint
                    )
                    last_tracking_throttle = int(throttle_count)
                else:
                    cur_jpos = _clamp_vector_step(
                        cur_jpos, prev_wire_jpos, effective_max_step
                    )
                    last_tracking_throttle = 0
                cur_jpos = _clamp_vector_deviation(
                    cur_jpos,
                    np.asarray(body_q_mj_now, dtype=np.float32),
                    max_wire_dev_from_body,
                )

                decoded_was_active = True
        else:
            # Idle wire (deploy stale, no decoder, or zero-token chunk).
            # Clear the ramp / LPF state so the next idle->VLA
            # transition primes them again from a fresh idle baseline.
            if decoded_was_active:
                ramp_from = None
                ramp_remaining = 0
                lpf_state = None
                lpf_future_state = None
            decoded_was_active = False
            # 2026-06-10 follow-up 5: apply the per-tick wire rate
            # clamp HERE too. Previously this clamp only ran inside
            # the decoder-succeeded branch above, so when (a) the
            # cold-restart hold released due to chunk_id advancing
            # but (b) the chunk's token was zero (decoder gate
            # ``np.linalg.norm(token[step]) > 1e-3`` failed), the
            # decoder branch was skipped, we fell here, and
            # ``cur_jpos = idle_loop.current(tick)`` set the wire
            # straight from operator-pose (prev_wire_jpos) to
            # idle_stand_pose. That's the "slam" the operator
            # reported when the handoff guard from follow-up 3 lost
            # to a zero-token VLA. Fix 1 above tightened the gate
            # so the hold no longer releases on zero-token chunks,
            # but this clamp is the defense-in-depth: if the
            # ``handoff_max_hold_ticks`` safety cap DOES expire
            # (VLA still zero-token after 4s) the wire ramps from
            # operator-pose toward idle_stand at the slow handoff
            # step (until ``handoff_step_remaining`` elapses) then
            # at ``max_wire_step`` rad/tick instead of snapping.
            # Cheap (numpy abs+max) so we accept the cost on normal
            # idle frames where peak < max_step is a no-op (returns
            # tgt unchanged per ``_clamp_vector_step``). 2026-06-10
            # follow-up 6: the slow-step uses the same handoff
            # countdown as the decoder branch above so the body
            # transitions from operator-pose at a bounded rate
            # regardless of whether VLA succeeded or the safety
            # cap fired.
            if handoff_step_remaining > 0:
                ramp_progress = 1.0 - float(handoff_step_remaining) / float(
                    max(handoff_step_ramp_ticks, 1)
                )
                ramp_progress = min(max(ramp_progress, 0.0), 1.0)
                effective_max_step = (
                    (1.0 - ramp_progress) * float(handoff_max_wire_step)
                    + ramp_progress * float(max_wire_step)
                )
                handoff_step_remaining -= 1
            else:
                effective_max_step = float(max_wire_step)
            # 2026-06-10 follow-up 11: same tracking-feedback gate as
            # the decoder-succeeded branch above. The idle wire still
            # benefits from per-joint feedback (especially during the
            # post-handoff hold-at-operator-pose window where the wire
            # is ramping back from operator pose to idle and any
            # actuator lag could create a sustained tracking error).
            tracking_active = (
                tracking_feedback_enabled
                and deploy_fresh
                and tracking_staleness_s <= tracking_stale_s
            )
            if tracking_active:
                cap_per_joint, throttle_count = _apply_tracking_feedback(
                    cur_jpos,
                    prev_wire_jpos,
                    body_q_mj_now,
                    body_dq_mj_now,
                    base_max_step=effective_max_step,
                    soft_rad=tracking_soft_rad,
                    hard_rad=tracking_hard_rad,
                    vel_margin=tracking_vel_margin,
                    vel_floor_rad_tick=tracking_vel_floor_rad_tick,
                    dt_s=period,
                )
                cur_jpos = _clamp_vector_step_per_joint(
                    cur_jpos, prev_wire_jpos, cap_per_joint
                )
                last_tracking_throttle = int(throttle_count)
            else:
                cur_jpos = _clamp_vector_step(
                    cur_jpos, prev_wire_jpos, effective_max_step,
                )
                last_tracking_throttle = 0

        # ---- F: optional partial-body freeze (legs/waist hold idle_stand) ---
        # Freeze the selected DOFs to the ``idle_stand`` clip (with its
        # small per-frame jitter). The clip's jitter is what the policy
        # was TRAINED against during ManagerEnvWrapper episodes -- without
        # it, the tracker output saturates and the robot leans ~25 deg
        # under gravity within a few seconds (see ``_IdleStandLoop``
        # docstring for the empirical comparison).
        #
        # We then SURGICALLY pin only ``waist_yaw_joint`` (slot 12) to
        # the live measured value -- waist_yaw is the dominant
        # heading-correction effector and ``idle_stand[0]``'s waist_yaw
        # is ~33 deg off DEFAULT_STAND_POSE_NP, which would otherwise
        # drive a steady-state ~33 deg heading drift on every
        # manipulation run. Every other frozen DOF (waist_pitch,
        # waist_roll, hip/knee/ankle) keeps the clip jitter for
        # balance. Falls back to identity (no waist_yaw override) when
        # x2_debug has never arrived.
        if _freeze_idx.size > 0:
            cur_jpos, future_fields = _apply_frozen_body_groups(
                jpos=cur_jpos,
                future_fields=future_fields,
                idle_jpos=idle_baseline,
                freeze_indices=_freeze_idx,
            )
            if deploy_fresh and bool(np.isin(WAIST_YAW_IDX, _freeze_idx)):
                measured_waist_yaw = float(body_q_mj_now[WAIST_YAW_IDX])
                cur_jpos = np.asarray(cur_jpos, dtype=np.float32).copy()
                cur_jpos[WAIST_YAW_IDX] = measured_waist_yaw
                if "joint_pos_mj_future" in future_fields:
                    jf = np.asarray(
                        future_fields["joint_pos_mj_future"], dtype=np.float32
                    ).copy()
                    jf[:, WAIST_YAW_IDX] = measured_waist_yaw
                    future_fields = {
                        **future_fields,
                        "joint_pos_mj_future": jf,
                        "joint_vel_mj_future": _jvel_future_from_poses(
                            cur_jpos, jf
                        ),
                    }

        # ---- G: hand joint shaping (chunk blend + LPF + step cap) ---
        if deploy_fresh:
            left_tgt = np.asarray(left[step], dtype=np.float32)
            right_tgt = np.asarray(right[step], dtype=np.float32)
            if chunk_id != last_wire_hand_chunk_id:
                hand_blend_from_left = (
                    prev_wire_left.copy()
                    if prev_wire_left is not None
                    else np.zeros(DEFAULT_HAND_DOF, dtype=np.float32)
                )
                hand_blend_from_right = (
                    prev_wire_right.copy()
                    if prev_wire_right is not None
                    else np.zeros(DEFAULT_HAND_DOF, dtype=np.float32)
                )
                hand_chunk_blend_remaining = int(hand_chunk_blend_ticks)
                last_wire_hand_chunk_id = chunk_id
            if (
                hand_chunk_blend_remaining > 0
                and hand_blend_from_left is not None
                and hand_blend_from_right is not None
            ):
                progress = float(
                    hand_chunk_blend_ticks - hand_chunk_blend_remaining + 1
                ) / float(max(hand_chunk_blend_ticks, 1))
                progress = min(max(progress, 0.0), 1.0)
                left_tgt = (
                    hand_blend_from_left * (1.0 - progress)
                    + left_tgt * progress
                ).astype(np.float32)
                right_tgt = (
                    hand_blend_from_right * (1.0 - progress)
                    + right_tgt * progress
                ).astype(np.float32)
                hand_chunk_blend_remaining -= 1
            left_step = left_tgt
            right_step = right_tgt
            if hand_lpf_alpha > 0.0:
                left_step, hand_lpf_left = _lpf_vector(
                    left_step, hand_lpf_left, hand_lpf_alpha
                )
                right_step, hand_lpf_right = _lpf_vector(
                    right_step, hand_lpf_right, hand_lpf_alpha
                )
            left_step = _clamp_vector_step(
                left_step, prev_wire_left, max_hand_step
            )
            right_step = _clamp_vector_step(
                right_step, prev_wire_right, max_hand_step
            )
        else:
            left_step = _ZERO_HAND_STEP
            right_step = _ZERO_HAND_STEP
        # 2026-06-10: hand override during cold-restart hold window.
        # The body path above swaps cur_jpos for operator pose, but
        # hands are driven by the last decoded chunk's
        # ``left[step]`` / ``right[step]``. Without this override
        # the fingers visibly snap to the in-flight VLA chunk during
        # the 25-tick hold (which is what produced the "only VLA
        # controls fingers" symptom). We DON'T skip the LPF / clamp
        # above: those are seeded from the operator-pose snapshot at
        # cold-restart so re-running them on the operator pose is a
        # no-op, and keeping them in the path means the post-hold
        # transition to decoded hands inherits warm LPF state.
        if in_hold_window and operator_hold_pose is not None:
            if "left_hand_joints" in operator_hold_pose:
                left_step = operator_hold_pose[
                    "left_hand_joints"
                ].astype(np.float32, copy=True)
            if "right_hand_joints" in operator_hold_pose:
                right_step = operator_hold_pose[
                    "right_hand_joints"
                ].astype(np.float32, copy=True)
        hand_delta_tick = 0.0
        if deploy_fresh:
            if prev_wire_left is not None and prev_wire_right is not None:
                hand_delta_tick = float(
                    max(
                        np.abs(left_step - prev_wire_left).max(),
                        np.abs(right_step - prev_wire_right).max(),
                    )
                )

        # ---- H: live yaw rebase on root_quat (tokenizer orientation) ---
        # The idle_stand clip is yaw-aligned to 0; if we ship that quat
        # while the robot is physically facing another heading, SONIC's
        # tokenizer obs sees a large orientation error and keeps twisting
        # waist_yaw (even with legs/waist joint targets frozen).
        if deploy_fresh:
            live_root_quat = _root_quat_xyzw_from_base_quat_wxyz(base_quat_now)
            cur_quat = live_root_quat
            if "root_quat_xyzw_future" in future_fields:
                future_fields = {
                    **future_fields,
                    "root_quat_xyzw_future": _tile_root_quat_future(
                        live_root_quat
                    ),
                }
            if not yaw_rebase_logged:
                from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw

                wxyz = np.asarray(base_quat_now, dtype=np.float64).reshape(-1)
                q = np.array(
                    [wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float64
                )
                yaw_deg = math.degrees(float(yaw_of_quat_xyzw(q)))
                print(
                    f"[live-VLA] root_quat yaw-rebase ACTIVE: wire "
                    f"root_quat_xyzw now tracks live x2_debug heading "
                    f"(yaw={yaw_deg:+.1f}deg)",
                    flush=True,
                )
                yaw_rebase_logged = True

        if not deploy_fresh:
            payload = {
                "joint_pos_mj": cur_jpos,
                "root_quat_xyzw": cur_quat,
                "motion_token": _ZERO_MOTION_TOKEN_STEP,
                "left_hand_joints": _ZERO_HAND_STEP,
                "right_hand_joints": _ZERO_HAND_STEP,
                "frame_index": np.array([tick], dtype=np.int64),
                **future_fields,
            }
        else:
            wire_token = (
                _ZERO_MOTION_TOKEN_STEP
                if not decode_body
                else token[step]
            )
            payload = {
                "joint_pos_mj": cur_jpos,
                "root_quat_xyzw": cur_quat,
                "motion_token": wire_token,
                "left_hand_joints": left_step,
                "right_hand_joints": right_step,
                "frame_index": np.array([tick], dtype=np.int64),
                **future_fields,
            }
        if wire_debug is not None:
            idle_for_metrics = idle_baseline
            if decoded_now is not None:
                raw_d_met = float(
                    np.abs(
                        decoded_now.astype(np.float32) - idle_for_metrics.astype(np.float32)
                    ).max()
                )
            else:
                raw_d_met = 0.0
            wire_d_idle = float(
                np.abs(
                    np.asarray(cur_jpos, dtype=np.float32)
                    - idle_for_metrics.astype(np.float32)
                ).max()
            )
            wire_d_body = float(
                np.abs(
                    np.asarray(cur_jpos, dtype=np.float32)
                    - np.asarray(body_q_mj_now, dtype=np.float32)
                ).max()
            )
            # ``raw_joint_pos_mj`` is the policy's decoded body intent
            # before any wire-shaping (clamp / LPF / ramp / chunk-blend).
            # When the policy hasn't produced a decoded chunk yet
            # (idle window) we fall back to the wire so chunk-dump
            # diagnostics still get a finite 31-D vector.
            raw_jpos_now = (
                decoded_now
                if decoded_now is not None
                else np.asarray(cur_jpos, dtype=np.float32)
            )
            wire_debug.update(
                joint_pos_mj=cur_jpos,
                left_hand_joints=left_step,
                right_hand_joints=right_step,
                raw_joint_pos_mj=raw_jpos_now,
                raw_delta_idle=raw_d_met,
                wire_delta_idle=wire_d_idle,
                wire_delta_body=wire_d_body,
                wire_delta_hand=hand_delta_tick,
                chunk_id=chunk_id,
                chunk_step=step,
            )

        # ---- Bootstrap-safe publish gate ---------------------------------
        # CRITICAL: never publish a pose frame before the bridge has seen
        # at least one ``x2_debug`` packet. Until then ``base_quat_now`` is
        # the dataclass default (identity quat = yaw=0 in world frame) and
        # the wire would ship ``root_quat_xyzw = yaw=0`` to a deploy that
        # is ALREADY in CONTROL (the PC2 SONIC daemon persists across
        # teleop/VLA sessions). The deploy then sees
        # ``rel = inv(R_z(measured_yaw)) * R_z(0)`` -- a real orientation
        # error -- and commands waist_yaw rotation toward yaw=0. On a
        # robot that started at yaw=-45 deg, this drags the body to
        # yaw=0 over the bootstrap window (~tick 0..T where T is when
        # x2_debug first arrives at the bridge). Subsequent VLA starts
        # find the robot already at yaw=0 so no further turn -- the
        # symptom captured 2026-06-07 as "first VLA start turns the
        # robot ~45 deg left, subsequent runs hold heading".
        #
        # The C++ deploy already has a bootstrap-safe path: when
        # ``ZmqPoseInputSource::LastReceivedMonotonicS() < 0``,
        # ``BuildTokenizerObs`` substitutes the measured quat as the
        # orientation reference and the policy holds whatever heading
        # the body is in. We just have to NOT BREAK that escape hatch by
        # publishing a phantom yaw=0 frame the moment the PUB socket
        # binds. Gating on ``state.received_any`` (one-way sticky: True
        # once a real packet has ever arrived) keeps the deploy on the
        # measured-quat bootstrap path until the bridge can ship a
        # correctly yaw-rebased ``root_quat_xyzw``.
        if not silent_wire and state.received_any:
            if not bootstrap_first_publish_logged:
                print(
                    f"[live-VLA] first pose publish (tick={tick}); "
                    f"x2_debug seen, root_quat now tracks live heading.",
                    flush=True,
                )
                bootstrap_first_publish_logged = True
            msg = pack_pose_message(payload, topic=topic, version=protocol_version)
            try:
                pub_sock.send(msg, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
        elif not silent_wire and not bootstrap_publish_logged:
            print(
                "[live-VLA] withholding pose publish until first x2_debug "
                "frame arrives (deploy stays on its measured-quat bootstrap "
                "override; prevents yaw=0 phantom reference that would drag "
                "the robot to world +X heading on VLA start).",
                flush=True,
            )
            bootstrap_publish_logged = True

        prev_wire_jpos = np.asarray(cur_jpos, dtype=np.float32).copy()
        if deploy_fresh:
            prev_wire_left = np.asarray(left_step, dtype=np.float32).copy()
            prev_wire_right = np.asarray(right_step, dtype=np.float32).copy()

        if tick % print_every == 0:
            _, _, _, _, _, alive = state.snapshot()
            wire_tag = "SILENT wire (no send)" if silent_wire else (
                "IDLE wire" if not deploy_fresh else None
            )
            if wire_tag is not None:
                print(
                    f"[live-VLA] pub tick={tick:6d}  {wire_tag}  "
                    f"deploy_alive={alive}",
                    flush=True,
                )
            else:
                if decoded_now is not None:
                    # Joint-pose deviation from idle so the operator
                    # can confirm at-a-glance whether the decoded body
                    # is doing anything meaningful (vs. just hovering
                    # at the idle setpoint). Reports BOTH the raw
                    # decoder Δ (policy intent) and the wire Δ
                    # (post-ramp post-LPF, what the deploy actually
                    # receives) so operator can distinguish a quiet
                    # policy from a clamped wire.
                    if idle_loop is not None:
                        idle_now, _ = idle_loop.current(tick)
                    else:
                        idle_now = _default_stand_body_pose_f32()
                    raw_delta = float(
                        np.abs(
                            decoded_now.astype(np.float32)
                            - idle_now.astype(np.float32)
                        ).max()
                    )
                    wire_delta = float(
                        np.abs(
                            np.asarray(cur_jpos, dtype=np.float32)
                            - idle_now.astype(np.float32)
                        ).max()
                    )
                    ramp_tag = (
                        f" ramp={ramp_remaining}/{ramp_in_ticks}"
                        if ramp_remaining > 0 else ""
                    )
                    wire_d_body = float(
                        np.abs(
                            np.asarray(cur_jpos, dtype=np.float32)
                            - np.asarray(body_q_mj_now, dtype=np.float32)
                        ).max()
                    )
                    decoded_tag = (
                        f"VLA-pose raw_Δ={raw_delta:.3f}rad "
                        f"wire_Δ={wire_delta:.3f}rad "
                        f"body_Δ={wire_d_body:.3f}rad{ramp_tag}"
                    )
                else:
                    decoded_tag = "idle-pose"
                hand_tag = (
                    f" hand_Δ={hand_delta_tick:.3f}rad"
                    if deploy_fresh
                    else ""
                )
                # 2026-06-10 follow-up 11: tracking-feedback telemetry.
                # Only printed when the feature is enabled (avoid log
                # spam for legacy runs). ``tf_throttle=N/14`` = how
                # many of the 14 arm joints had their per-tick step
                # cap dropped below 50% of base this tick by the
                # closed-loop feedback. ``N>0`` means feedback is
                # actively protecting the wire; sustained ``N=14``
                # means the actuator is saturating and the operator
                # may want to back off the task.
                tf_tag = (
                    f" tf_throttle={last_tracking_throttle}/{tracking_arm_count}"
                    if tracking_feedback_enabled
                    else ""
                )
                print(
                    f"[live-VLA] pub tick={tick:6d} "
                    f"chunk_id={chunk_id:4d} step={step:2d}/{horizon} "
                    f"|token|={float(np.linalg.norm(token[step])):.3f} "
                    f"|left|={float(np.linalg.norm(left[step])):.3f} "
                    f"{decoded_tag}{hand_tag}{tf_tag} "
                    f"deploy_alive={alive}",
                    flush=True,
                )
        chunk_step = min(chunk_step + 1, horizon - 1)
        tick += 1
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()
    return tick


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def _validate_pin_order_or_die() -> None:
    """Best-effort runtime check that ``MJ_TO_PIN`` matches the X2 schema.

    Falls back to a warning if ``gear_sonic.data.features_x2_vla`` can't be
    imported (e.g. minimal-deps test env): the mapping is hard-coded so a
    crash at runtime would be visibly miscalibrated joints anyway.
    """
    try:
        from gear_sonic.data.features_x2_vla import get_x2_robot_model
        from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
            X2_BODY_JOINT_NAMES,
        )
    except Exception as exc:
        print(f"[live-VLA] WARN: could not validate joint order: {exc}", flush=True)
        return

    rm = get_x2_robot_model()
    pin_names = list(rm.joint_names)
    mj_names = list(X2_BODY_JOINT_NAMES)
    if len(pin_names) != NUM_BODY_DOFS or len(mj_names) != NUM_BODY_DOFS:
        raise RuntimeError(
            f"unexpected joint count: pin={len(pin_names)} mj={len(mj_names)} (want {NUM_BODY_DOFS})"
        )
    for i, mj_idx in enumerate(MJ_TO_PIN):
        if pin_names[i] != mj_names[mj_idx]:
            raise RuntimeError(
                f"MJ_TO_PIN mismatch at i={i}: pin[{i}]={pin_names[i]!r} "
                f"!= mj[{mj_idx}]={mj_names[mj_idx]!r}"
            )
    print(f"[live-VLA] joint-order check OK ({NUM_BODY_DOFS} body DOFs)", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path", default="",
        help="Path to the fine-tuned Isaac-GR00T checkpoint directory "
             "(expects model.safetensors + processor/ + experiment_cfg/). "
             "Required UNLESS --no-policy is passed.",
    )
    parser.add_argument(
        "--no-policy", action="store_true",
        help="Skip the Gr00tPolicy load and inference worker entirely. "
             "The 50 Hz publisher + x2_debug SUB still run so the wire "
             "carries the planner-like idle stand reference (live "
             "joint_pos_mj when x2_debug is fresh, canonical stand "
             "otherwise; zero motion_token / zero hand joints). Use this "
             "to validate the recorder + deploy sequence in isolation "
             "without paying the model-load cost.",
    )
    parser.add_argument(
        "--prompt", default="play minecraft music on piano",
        help="Language instruction passed to the VLA on every inference.",
    )
    parser.add_argument(
        "--embodiment-tag", default="new_embodiment",
        help="Embodiment tag registered for the X2 modality config.",
    )
    parser.add_argument(
        "--modality-config", default="gear_sonic/data/x2_modality_config_10dof.py",
        help="Side-loadable modality config that registers NEW_EMBODIMENT.",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Torch device for the VLA.",
    )
    parser.add_argument("--pub-host", default="*", help="Bind iface for the pose PUB.")
    parser.add_argument("--pub-port", type=int, default=5556, help="Port for the pose PUB (matches deploy --vla-zmq-port).")
    parser.add_argument("--pub-topic", default="pose")
    parser.add_argument("--sub-host", default="localhost", help="Host of the deploy's x2_debug PUB.")
    parser.add_argument("--sub-port", type=int, default=5557, help="Port of the deploy's x2_debug PUB (matches --vla-debug-port).")
    parser.add_argument("--sub-topic", default="x2_debug")

    # ----- vla_control SUB (manual-takeover cold restart) ----------
    # Optional edge-event SUB driven by x2_pose_proxy.py when running
    # alongside a teleop wire publisher on the override port. When
    # ``--vla-control-port`` > 0 the bridge subscribes for
    # ``override_engaged`` / ``override_released`` events and uses
    # the released edge to clear all smoothing state + drop any
    # in-flight chunk decoded against pre-override observations.
    # Disabled by default so existing autonomous-only runs are
    # byte-for-byte unchanged. See 2026-06-10 milestone.
    parser.add_argument(
        "--vla-control-host", default="127.0.0.1",
        help="Host of the x2_pose_proxy vla_control PUB (default "
             "127.0.0.1; proxy runs on PC2, bridge on laptop -- "
             "override to the PC2 IP for split topology).",
    )
    parser.add_argument(
        "--vla-control-port", type=int, default=-1,
        help="Port of the x2_pose_proxy vla_control PUB. Set to a "
             "positive int (e.g. 5559) to enable manual-takeover "
             "cold restarts. Default -1 = DISABLED.",
    )
    parser.add_argument(
        "--vla-control-topic", default="vla_control",
        help="Topic prefix on the vla_control PUB (default "
             "'vla_control'; must match the proxy's "
             "--vla-control-topic).",
    )
    parser.add_argument(
        "--vla-cold-restart-hold-ticks", type=int, default=25,
        help="Ticks (at --rate) to ship the operator's MEASURED "
             "pose on the wire after a cold restart, before any "
             "newly-decoded chunk is allowed to engage. Default 25 "
             "@ 50Hz = 500 ms -- long enough for the proxy's HOLD "
             "ladder to hand the wire back without a step change "
             "from the operator's pose to the idle clip's pose, "
             "short enough that VLA re-engagement feels prompt.",
    )
    parser.add_argument(
        "--vla-handoff-max-wire-step", type=float, default=0.012,
        help="Per-element max joint-position step on the wire DURING "
             "the post-handoff slow window. Default 0.012 rad/tick = "
             "~36 deg/s/joint at 50 Hz (vs --vla-max-wire-step default "
             "0.035 rad/tick = ~100 deg/s/joint). Applies for "
             "--vla-handoff-step-ramp-ticks ticks after the cold-restart "
             "hold releases, then linearly ramps back to "
             "--vla-max-wire-step. Motivation: the operator hand-off "
             "pose can be ~3.7 rad (L_inf) away from VLA's first "
             "decoded chunk; the existing 75-tick LPF + 0.035 rad/tick "
             "rate clamp lets the wire move all 31 joints at once "
             "at 1.75 rad/s coordinated, which the operator visually "
             "reports as a slam even though no single joint exceeds "
             "the limit (see 2026-06-10 follow-up 6). Set equal to "
             "--vla-max-wire-step to disable the slow-step window.",
    )
    parser.add_argument(
        "--vla-handoff-step-ramp-ticks", type=int, default=250,
        help="Number of ticks over which to linearly ramp the wire "
             "step from --vla-handoff-max-wire-step (slow, applied "
             "right after handoff) back to --vla-max-wire-step "
             "(normal steady-state). Default 250 @ 50Hz = 5.0 s. "
             "Set to 0 to disable the slow-step window (the rate "
             "clamp jumps to --vla-max-wire-step immediately).",
    )
    parser.add_argument(
        "--vla-handoff-max-hold-ticks", type=int, default=200,
        help="Safety cap on how long the bridge will keep the wire "
             "pinned at the operator's hand-off pose WHILE WAITING "
             "for the first eligible decoded chunk after a cold "
             "restart. The hold window (set by --vla-cold-restart-"
             "hold-ticks) is the MINIMUM wait; this cap is the "
             "MAXIMUM. Between them the bridge stays at the operator "
             "pose until chunk_id > cold_restart_chunk_baseline "
             "(= the first chunk the model decoded AFTER the "
             "operator released the wire). Default 200 @ 50Hz = "
             "4.0 s. Without this, when the post-hold tick hits and "
             "no eligible chunk has arrived yet (inference cadence "
             "~15Hz vs wire 50Hz means up to 3 wire ticks between "
             "chunks; first-chunk decode time is ~480 ms), the wire "
             "snaps from the operator's pose to the idle_stand clip "
             "for the gap. Setting this below --vla-cold-restart-"
             "hold-ticks is a CONFIG ERROR (the launcher catches it). "
             "Set to 0 to disable the wait-for-first-chunk behaviour "
             "(legacy 2026-06-10 behaviour: snap to idle on hold "
             "expiry; not recommended).",
    )
    # ---- 2026-06-10 follow-up 11: closed-loop tracking feedback ----
    # Per-joint proprio feedback on the wire step cap. When enabled,
    # the bridge reads x2_debug's measured arm-joint positions and
    # velocities each tick and throttles the per-arm-joint per-tick
    # step so the wire never outpaces the actuator's actual response.
    # Eliminates the open-loop sensitivity to inference jitter /
    # battery sag / motor temperature drift that drove the
    # 2026-06-10 PM oscillation incident.
    #
    # Step 1 rollout: default DISABLED (--vla-tracking-feedback to
    # opt in). v3 static defaults (LPF/blend/step-cap) remain in
    # place; feedback is additive belt-and-suspenders so any
    # regression on real robot can be isolated by flipping the flag.
    # Step 2 (separate commit) flips the default to ON and relaxes
    # the static defaults once feedback is validated.
    parser.add_argument(
        "--vla-tracking-feedback",
        dest="vla_tracking_feedback",
        action="store_true",
        default=False,
        help="Enable closed-loop tracking feedback on the wire step "
             "cap. When ON, the bridge per-arm-joint-throttles the "
             "per-tick step based on (a) position error |target - "
             "measured| and (b) measured joint velocity. Pairs with "
             "--vla-tracking-{soft,hard}-rad and --vla-tracking-"
             "{velocity-margin,velocity-floor-rad-tick}. Default OFF "
             "(byte-identical to scalar clamp). When ON but x2_debug "
             "proprio is stale (>--vla-tracking-stale-ms), the "
             "bridge falls back to the scalar clamp automatically.",
    )
    parser.add_argument(
        "--no-vla-tracking-feedback",
        dest="vla_tracking_feedback",
        action="store_false",
        help="Force-disable tracking feedback (overrides "
             "--vla-tracking-feedback; useful when the launcher "
             "exports a default-on env var but the operator wants "
             "to A/B against the scalar-clamp path mid-session).",
    )
    parser.add_argument(
        "--vla-tracking-soft-rad", type=float, default=0.15,
        help="Tracking-feedback POSITION error threshold (rad) below "
             "which the per-joint step cap stays at its base value. "
             "Default 0.15 rad (~8.6 deg) -- typical actuator-lag "
             "from a v3-tuned wire is ~0.05 rad, so legitimate "
             "motion never triggers backoff. Pair with "
             "--vla-tracking-hard-rad.",
    )
    parser.add_argument(
        "--vla-tracking-hard-rad", type=float, default=0.40,
        help="Tracking-feedback POSITION error threshold (rad) above "
             "which the per-joint step cap drops to 0 (joint frozen "
             "until actuator catches up). Default 0.40 rad (~23 deg) "
             "-- chosen so the freeze only fires when the actuator "
             "is clearly saturating (e.g., hit a contact, hit a "
             "joint limit, SONIC fighting the wire). Between soft "
             "and hard the cap drops linearly.",
    )
    parser.add_argument(
        "--vla-tracking-velocity-margin", type=float, default=1.5,
        help="Tracking-feedback VELOCITY margin: per-tick cap is "
             "capped by ``margin * |measured_dq| * dt``. Default 1.5 "
             "lets the wire move at most 50%% faster than the "
             "actuator is currently moving (with the floor below as "
             "a backstop so the wire can start from rest). Lower = "
             "more conservative.",
    )
    parser.add_argument(
        "--vla-tracking-velocity-floor-rad-tick", type=float, default=0.01,
        help="Tracking-feedback VELOCITY FLOOR (rad/tick): minimum "
             "per-joint step the velocity cap can produce, even when "
             "the actuator is at rest (measured_dq = 0). Default "
             "0.01 rad/tick @ 50 Hz = 0.5 rad/s -- matches v3-tuned "
             "max_wire_step's effective rise rate so cold starts "
             "from idle aren't artificially frozen. Set to 0 to "
             "REQUIRE non-zero measured velocity before allowing any "
             "wire motion (motion-only; do NOT use for cold start).",
    )
    parser.add_argument(
        "--vla-tracking-stale-ms", type=int, default=100,
        help="Tracking-feedback STALENESS threshold (ms). If the "
             "x2_debug snapshot is older than this, the bridge "
             "falls back to the scalar clamp for that tick. Default "
             "100 ms = 5 publish ticks at 50 Hz; covers a single "
             "x2_debug packet drop without disabling feedback. Set "
             "very high (e.g. 10000) to NEVER fall back (only use "
             "in sim where staleness is impossible).",
    )

    parser.add_argument("--rate", type=float, default=DEFAULT_PUB_RATE_HZ, help="Publish rate (Hz). 50 matches the deploy control loop.")
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Total run time in seconds (0 = run until Ctrl-C).",
    )
    parser.add_argument(
        "--inference-min-period-s", type=float, default=0.4,
        help="Lower bound on time between successive inferences (s). The "
             "publisher always advances at --rate; this just throttles the GPU.",
    )
    parser.add_argument(
        "--wait-for-ready-file", type=str, default=None,
        help="Path to a sentinel file written by a co-spawned recorder "
             "(via --ready-file). When set, the inference thread holds at "
             "idle stand AFTER 'policy ready' until this file appears, so "
             "the recorder captures the arm rise from idle instead of "
             "missing the first ~8 s while the recorder warms up. The "
             "publisher thread keeps streaming idle stand the whole time, "
             "so the deploy never sees a gap. No effect when unset.",
    )
    parser.add_argument(
        "--wait-for-ready-file-timeout-s", type=float, default=120.0,
        help="Max seconds the inference thread will block on the "
             "--wait-for-ready-file gate before logging a warning and "
             "proceeding anyway. 0 disables the timeout (wait forever). "
             "Default 120 s is enough for cold cameras + MuJoCo renderer "
             "init on a 4090-class laptop.",
    )
    parser.add_argument(
        "--idle-stand-pkl", type=str,
        default=str(REPO_ROOT / "gear_sonic" / "data" / "motions"
                    / "x2_planner_primitives.pkl"),
        help="Primitives PKL containing the planner's 'idle_stand' bin. "
             "When found, the publisher replays this clip at --rate so the "
             "wire is byte-equivalent to --planner-only (verified upright "
             "for 20k+ ticks). Pass an empty string to fall back to the "
             "static DEFAULT_STAND_POSE reference (NOT recommended -- "
             "stabilises at ~25 deg lean even with v5 future window).",
    )
    parser.add_argument(
        "--silent-wire", action="store_true",
        help="Bind the body_pose PUB but skip every send() call -- the "
             "publisher loop ticks, the x2_debug SUB stays live, and the "
             "process can be cleanly shut down, but NOTHING reaches the "
             "deploy. Used to validate the deploy's built-in 'no upstream' "
             "fallback: ZmqPoseInputSource::Connect() pre-fills its cache "
             "with default_angles (the trained stand pose), and "
             "has_body_reference_ stays False, so Sample() always returns "
             "that prefill. Pair with --no-policy + the recorder's "
             "--no-idle-publish so neither hop forwards anything either; "
             "the deploy then holds itself upright on its own reference. "
             "Without --silent-wire the bridge keeps publishing the "
             "idle_stand replay, which the deploy commits to the moment "
             "it decodes the first frame -- and that's the run that leans "
             "~28 deg under --vla-no-policy.",
    )
    parser.add_argument(
        "--motion-token-decoder", type=str, default=None,
        dest="motion_token_decoder",
        help="Path to the SONIC decoder .pt checkpoint (e.g. "
             "model_step_025000.pt) used to decode the VLA's predicted "
             "motion_token chunks back into joint_pos_mj poses on the wire. "
             "Without this, the bridge emits idle_stand for joint_pos_mj on "
             "every tick and the C++ deploy's fused encoder+FSQ+decoder ONNX "
             "re-tokenises that idle reference and ignores the live "
             "motion_token field (header explicitly documents 'motion_token: "
             "currently logged but otherwise unused' -- see "
             "zmq_pose_input_source.hpp:22-25), so the body never moves "
             "under VLA authority. With --motion-token-decoder set, each "
             "publish tick decodes chunk[step] (and 9 future steps) via the "
             "g1_dyn decoder + the C++ deploy's "
             "target_mj=default+action_il*scale formula and ships the result "
             "as joint_pos_mj / joint_pos_mj_future. The deploy's encoder "
             "then re-tokenises this VLA-driven trajectory, which makes the "
             "body actually track the predicted motion. Recommended path: "
             "the .pt that pairs with the deploy ONNX you're running.",
    )
    # Deprecated alias kept for backwards compat with old runbooks and
    # CI scripts. Same dest as the canonical flag; the post-parse
    # bookkeeping below logs a one-shot warning when the alias is what
    # the operator actually typed.
    parser.add_argument(
        "--sonic-checkpoint", type=str, default=None,
        dest="motion_token_decoder",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sonic-decoder-device", type=str, default="cpu",
        help="Torch device for the bridge-side SONIC decoder. Default cpu "
             "(decoder is ~5 M params, sub-millisecond per chunk; CPU is "
             "well within the 20 ms 50 Hz budget). Use cuda:0 only if your "
             ".venv torch build supports your GPU (see SONIC_TOKENIZER_DEVICE "
             "comment in run_x2_quest3_planner_stack.sh).",
    )
    parser.add_argument(
        "--vla-ramp-in-ticks", type=int, default=75,
        help="Number of 50 Hz publish ticks (default 25 = 0.5 s) over "
             "which to linearly interpolate from the idle wire to the "
             "first decoded VLA pose. Eliminates the step-input shove "
             "that previously corresponded to up to 0.5 rad joint "
             "deviation in a single tick on the very first chunk -- "
             "see the runtime-architecture doc for the failure mode. "
             "Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-target-lpf-hz", type=float, default=2.0,
        help="One-pole low-pass cutoff (Hz) on the wire's current "
             "``joint_pos_mj`` slot. Default 2 Hz heavily attenuates "
             "inter-chunk saw-tooth / vibration. Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-future-lpf-hz", type=float, default=2.0,
        help="LPF cutoff (Hz) for ``joint_pos_mj_future`` (the 9-slot "
             "window the deploy tokeniser reads). Matching the body "
             "LPF prevents sharp future slopes that excite SONIC "
             "tracking vibration. Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-hand-lpf-hz", type=float, default=1.0,
        help="LPF cutoff (Hz) for left/right hand joint targets on the "
             "wire. Default 1 Hz attenuates finger jitter from noisy VLA "
             "hand chunks. Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-hand-chunk-blend-ticks", type=int, default=30,
        help="Linear blend length (50 Hz ticks) when a new VLA hand "
             "chunk lands, softening inter-chunk finger seams. Pass 0 "
             "to disable.",
    )
    parser.add_argument(
        "--vla-max-hand-step", type=float, default=0.08,
        help="Max per-tick hand joint delta (rad) on the wire at 50 Hz. "
             "Limits how fast finger targets can move tick-to-tick. "
             "Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-max-wire-dev-from-body", type=float, default=0.18,
        help="Max per-joint deviation (rad) of the wire ``joint_pos_mj`` "
             "from the live observed ``body_q`` (x2_debug). Prevents the "
             "decoder from commanding runaway poses far from the robot's "
             "actual configuration -- the dominant sim collapse mode when "
             "this is unset. Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-max-wire-step", type=float, default=0.035,
        help="Max per-tick joint delta (rad) on the wire at 50 Hz. Limits "
             "how fast the pose reference can move tick-to-tick. Pass 0 "
             "to disable.",
    )
    parser.add_argument(
        "--vla-chunk-blend-ticks", type=int, default=40,
        help="Linear blend length (50 Hz ticks) when a new VLA chunk "
             "lands, softening chunk-boundary seams. Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-max-action-il", type=float, default=8.0,
        help="Clip the decoder's raw IsaacLab-order action residual before "
             "it is echoed into the 990-D proprio ``last_action`` term. "
             "Matches training ``action_clip_value=20`` headroom but "
             "prevents proprio feedback blow-up. Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-decode-delay-ticks", type=int, default=150,
        help="Publish ticks (50 Hz) of idle_stand wire after deploy comes "
             "up before the SONIC decoder is allowed to overwrite "
             "``joint_pos_mj``. Lets parity RSI + proprio history "
             "stabilise before VLA authority lands. Pass 0 to disable.",
    )
    parser.add_argument(
        "--vla-body-mode",
        type=str,
        default=VlaBodyMode.MANIPULATION.value,
        choices=[m.value for m in VlaBodyMode],
        help="Which DOFs the VLA may command on the pose wire. "
             "'manipulation' = decode arms/head/hands; freeze legs+waist "
             "to idle_stand (VR ARM_MANIPULATION analog; needs decoder). "
             "'locomotion' = full-body decode (VR LOCOMOTION analog).",
    )
    parser.add_argument(
        "--vla-mode-control-file", type=str, default="",
        help="Optional path watched each publish tick; file contains one "
             "of manipulation | locomotion. Lets an operator flip modes "
             "at runtime without restarting (e.g. echo "
             "manipulation > /tmp/vla_body_mode).",
    )
    parser.add_argument(
        "--vla-hands-only", action="store_true",
        help="DEPRECATED and ignored. Former fingers-only mode was removed; "
             "use --vla-body-mode manipulation (arms+hands, frozen legs).",
    )
    parser.add_argument(
        "--vla-freeze-body-groups", type=str, default="",
        help="Optional override for manipulation-mode freeze groups "
             "(comma-separated names pinned to idle_stand after decode). "
             "Default manipulation freeze is legs,waist. Group names match "
             "x2_recipes: legs, waist, arms, left_arm, ...",
    )
    parser.add_argument(
        "--dump-chunks-dir", type=str, default=None,
        help="If set, dump each Nth full action chunk (token + hand joints + "
             "ego_view + observation snapshot) to this directory as .npz files. "
             "Diagnostic use only -- helps verify whether the model predicts a "
             "full gesture across the whole horizon or just spits out a single "
             "near-constant frame.",
    )
    parser.add_argument(
        "--dump-chunks-every", type=int, default=5,
        help="When --dump-chunks-dir is set, save every Nth chunk (1 = every "
             "chunk; 5 means save chunk_id 0,5,10,...). Keeps disk usage sane "
             "during a long probe run.",
    )
    parser.add_argument(
        "--render-width", type=int, default=640,
        help="Ego-view width (must match the dataset M5 used).",
    )
    parser.add_argument(
        "--render-height", type=int, default=480,
        help="Ego-view height (must match the dataset M5 used).",
    )
    parser.add_argument(
        "--no-omnihand", action="store_true",
        help="Disable the OmniHand mesh fragment (debug only; M5/M6 trained "
             "with OmniHand on).",
    )
    parser.add_argument(
        "--protocol-version", type=int, choices=(3, 4), default=4,
        help="Pose wire protocol version. v4 includes 'count'.",
    )
    parser.add_argument("--quiet", action="store_true", help="Cut down log spam.")
    parser.add_argument(
        "--print-every", type=int, default=50,
        help="Log every Nth pose publish.",
    )
    parser.add_argument(
        "--video-out", type=str, default=None,
        help=(
            "Optional MP4 output path for the ego-view video. When set, a "
            "separate thread mirrors the deploy state and records the live "
            "``ego_view`` (same camera the VLA sees) at --video-fps."
        ),
    )
    parser.add_argument(
        "--video-fps", type=float, default=25.0,
        help="Video recording frame rate. 25 Hz is a smooth wall-clock playback; "
             "set to 50 for 1:1 with the deploy control loop.",
    )
    parser.add_argument(
        "--video-front-out", type=str, default=None,
        help=(
            "Optional MP4 output path for a third-person front view. "
            "Independent of --video-out: each runs in its own thread with "
            "its own EGL context, mirroring the same _LatestState."
        ),
    )
    parser.add_argument(
        "--video-front-camera", type=str, default="third_person_front",
        help="MuJoCo camera spec for the front view "
             "(third_person_front | third_person_side | third_person_above).",
    )
    parser.add_argument(
        "--video-front-width", type=int, default=1280,
        help="Front-view width. Independent of the VLA's --render-width.",
    )
    parser.add_argument(
        "--video-front-height", type=int, default=720,
        help="Front-view height. Independent of the VLA's --render-height.",
    )
    parser.add_argument(
        "--cameras-source",
        type=str,
        choices=("ghost", "zmq"),
        default="ghost",
        help=(
            "Where the VLA's image observation comes from. "
            "'ghost' (default, sim path): build a MuJoCo EGL renderer "
            "inside the inference thread and render an ``ego_view`` from "
            "the live body_q_mj — matches "
            "``record_synthetic_smoketest_dataset.py``. "
            "'zmq' (real-robot path): subscribe to the PC2 "
            "``ComposedCameraClientSensor`` PUB and pass through whichever "
            "video.modality_keys the registered modality config requests "
            "(typically ``stereo_left`` + ``stereo_right`` for the omnihand "
            "stereo config). Requires the PC2 camera bridge to be running — "
            "see ``gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve``."
        ),
    )
    parser.add_argument(
        "--cameras-zmq-host",
        type=str,
        default="10.0.1.41",
        help=(
            "Host of the PC2 ComposedCameraClientSensor PUB. Only used "
            "with --cameras-source zmq. Default matches the X2 ultra "
            "wired-LAN address used by ``x2_pc2_cameras.sh``."
        ),
    )
    parser.add_argument(
        "--cameras-zmq-port",
        type=int,
        default=5555,
        help=(
            "Port of the PC2 ComposedCameraClientSensor PUB. Must match "
            "the publisher's --port (default 5555)."
        ),
    )
    parser.add_argument(
        "--cameras-staleness-s",
        type=float,
        default=2.0,
        help=(
            "Maximum age (s) of the latest camera frame before the "
            "inference thread refuses to run and posts a safe idle chunk. "
            "Only used with --cameras-source zmq. 2 s is generous given "
            "the publisher runs at 15 Hz."
        ),
    )
    parser.add_argument(
        "--cameras-warmup-s",
        type=float,
        default=10.0,
        help=(
            "Time (s) to wait at startup for the first complete camera "
            "payload (containing all required modality keys) before "
            "aborting. Only used with --cameras-source zmq."
        ),
    )
    args = parser.parse_args(argv)
    if not args.no_policy and not args.model_path:
        parser.error("--model-path is required unless --no-policy is set")
    if args.cameras_source == "zmq":
        if args.no_policy:
            parser.error(
                "--cameras-source zmq is redundant with --no-policy (the "
                "inference worker isn't started, so the camera stream isn't "
                "consumed). Drop one or the other."
            )
        if args.video_out or args.video_front_out:
            parser.error(
                "--video-out / --video-front-out require the MuJoCo "
                "renderer and are sim-only; they cannot be combined with "
                "--cameras-source zmq. Use --dump-chunks-dir instead for "
                "real-robot diagnostics — the dump includes every camera "
                "frame the policy saw."
            )
    return args


def _load_modality_config(path_or_module: str) -> None:
    """Side-load the X2 modality config so ``NEW_EMBODIMENT`` resolves.

    Supports either a path to a .py file or a dotted module name.
    """
    if path_or_module.endswith(".py") or "/" in path_or_module:
        spec_path = (REPO_ROOT / path_or_module).resolve() if not Path(path_or_module).is_absolute() else Path(path_or_module).resolve()
        if not spec_path.is_file():
            raise FileNotFoundError(f"--modality-config not found: {spec_path}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("_x2_modality_config_loader", spec_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load module from {spec_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        import importlib
        importlib.import_module(path_or_module)


def _get_required_video_keys(embodiment_tag: str) -> list[str]:
    """Return the list of required ``video.modality_keys`` for the loaded config.

    Must be called *after* :func:`_load_modality_config` has populated
    ``gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS``. The
    returned keys are the source of truth for which cameras the policy
    expects on every inference (e.g. ``["ego_view"]`` for the sim 10dof
    config, ``["stereo_left", "stereo_right"]`` for the real-robot
    omnihand stereo config).
    """
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS

    if embodiment_tag not in MODALITY_CONFIGS:
        raise RuntimeError(
            f"embodiment_tag={embodiment_tag!r} is not registered in "
            f"MODALITY_CONFIGS. Available tags: {sorted(MODALITY_CONFIGS.keys())}. "
            f"Make sure --modality-config side-loaded a module that calls "
            f"register_modality_config(..., embodiment_tag=...)."
        )
    cfg = MODALITY_CONFIGS[embodiment_tag]
    video_cfg = cfg.get("video")
    if video_cfg is None:
        raise RuntimeError(
            f"embodiment_tag={embodiment_tag!r} has no 'video' modality config."
        )
    keys = list(video_cfg.modality_keys)
    if not keys:
        raise RuntimeError(
            f"embodiment_tag={embodiment_tag!r} has empty video.modality_keys."
        )
    return keys


class _GhostCameraProvider:
    """Multi-view MuJoCo ghost-renderer for the sim path.

    The bridge originally supported a single ego-view ``MujocoFrameRenderer``
    behind a simple ``renderer_factory`` callable. That covers the
    single-camera (``video.modality_keys=["ego_view"]``) sim modality
    config, but real-robot checkpoints are trained with the omnihand-stereo
    config which declares two video keys (``stereo_left``, ``stereo_right``)
    and the older single-renderer path silently fed nothing for them.

    This provider replaces the single ``renderer_factory`` slot in
    :func:`_inference_worker` with a multi-renderer dict. Internally it
    keeps **one** :class:`MujocoFrameRenderer` per *unique* underlying
    MJCF camera (the X2 MJCF has separate ``rgbd_head_front`` and
    ``stereo_head_front`` mounts but only one stereo optical centre), then
    aliases each required modality key onto its mapped MJCF camera before
    returning the per-tick frame dict. So requesting
    ``["stereo_left", "stereo_right"]`` builds *one* renderer for
    ``stereo_head_front`` and ships the same frame under both keys --
    "degenerate stereo" -- which is enough to validate the data flow,
    ramp/LPF, and SONIC body motion in sim without breaking the real
    robot. True stereo (separate L/R camera mounts in the MJCF) is the
    natural follow-up if depth-disparity reasoning becomes the bottleneck.

    EGL contexts are thread-local, so :meth:`build` (which actually
    instantiates the renderers) must be called from inside the thread
    that will call :meth:`render_frames`. Use the :func:`build_factory`
    helper to capture the construction args; the inference worker will
    invoke it from its own thread.

    Keys are mapped via :data:`MODALITY_TO_MJ_CAMERA`. Unknown keys raise
    at construction time so misconfigured modality configs fail fast
    instead of mid-rollout.
    """

    # Map ``video.modality_keys`` -> MJCF camera name (one of the head
    # mounts defined in ``HEAD_CAMERAS`` in
    # ``render_smoketest_episode_video.py``). The aliases mirror the
    # ``HeadCameraSpec.aliases`` field on each spec; we keep the explicit
    # table here so a bad key fails at provider construction with a
    # readable error rather than deep inside ``resolve_camera_spec``.
    MODALITY_TO_MJ_CAMERA: dict[str, str] = {
        # Egocentric / RGB-D head module
        "ego_view": "rgbd_head_front",
        "rgbd": "rgbd_head_front",
        "rgbd_head_front": "rgbd_head_front",
        "head_front": "rgbd_head_front",
        # Stereo head block (single optical centre in MJCF, aliased
        # into both L and R modality slots for "degenerate stereo")
        "stereo_left": "stereo_head_front",
        "stereo_right": "stereo_head_front",
        "stereo": "stereo_head_front",
        "stereo_head_front": "stereo_head_front",
        # Pass-through for any direct MJCF camera names someone might
        # use in a custom modality config.
        "rgb_head_center": "rgb_head_center",
        "rgb_head_rear": "rgb_head_rear",
    }

    def __init__(
        self,
        *,
        required_keys: list[str],
        width: int,
        height: int,
        with_omnihand: bool,
    ) -> None:
        self._required_keys: list[str] = list(required_keys)
        self._width = int(width)
        self._height = int(height)
        self._with_omnihand = bool(with_omnihand)
        # key -> MJCF camera name. Validated eagerly so misconfigured
        # modality configs blow up before the inference thread starts.
        self._key_to_camera: dict[str, str] = {}
        for key in self._required_keys:
            cam = self.MODALITY_TO_MJ_CAMERA.get(key)
            if cam is None:
                raise ValueError(
                    f"ghost camera provider: video.modality_key {key!r} has "
                    f"no MJCF camera mapping. Known mappings: "
                    f"{sorted(self.MODALITY_TO_MJ_CAMERA.keys())}. Add a new "
                    f"entry to _GhostCameraProvider.MODALITY_TO_MJ_CAMERA "
                    f"and (if needed) extend HEAD_CAMERAS in "
                    f"render_smoketest_episode_video.py."
                )
            self._key_to_camera[key] = cam
        # Lazy: filled by build() inside the inference thread.
        self._renderers: dict[str, Any] = {}

    @property
    def required_keys(self) -> list[str]:
        return list(self._required_keys)

    @property
    def key_to_camera(self) -> dict[str, str]:
        return dict(self._key_to_camera)

    @property
    def unique_cameras(self) -> list[str]:
        # Stable order: by camera name.
        return sorted(set(self._key_to_camera.values()))

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def with_omnihand(self) -> bool:
        return self._with_omnihand

    def build(self) -> "_GhostCameraProvider":
        """Instantiate the underlying MuJoCo renderers.

        Must be called from the thread that will call :meth:`render_frames`
        (EGL contexts are thread-local). Returns ``self`` so the inference
        worker can chain ``provider = factory().build()``.
        """
        if self._renderers:
            return self
        from gear_sonic.scripts.render_smoketest_episode_video import MujocoFrameRenderer

        for cam in self.unique_cameras:
            self._renderers[cam] = MujocoFrameRenderer(
                camera=cam,
                width=self._width,
                height=self._height,
                with_omnihand=self._with_omnihand,
                egl=True,
            )
        return self

    def render_frames(
        self,
        *,
        body_q: np.ndarray,
        left_active: np.ndarray,
        right_active: np.ndarray,
        root_quat_wxyz: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Render every unique MJCF camera once, alias into the required keys.

        The same uint8 RGB array is shared across modality keys that map
        to the same MJCF camera (e.g. ``stereo_left`` and ``stereo_right``
        both map to ``stereo_head_front``). Callers must treat the
        returned frames as read-only or copy them before mutating.
        """
        if not self._renderers:
            raise RuntimeError(
                "_GhostCameraProvider.render_frames() called before build(); "
                "call build() from the rendering thread first."
            )
        # Render each unique MJCF camera once; this is the expensive
        # MuJoCo + EGL step. For the omnihand-stereo case that's one
        # render per tick (both L and R alias the same camera).
        cam_to_frame: dict[str, np.ndarray] = {}
        for cam, renderer in self._renderers.items():
            if root_quat_wxyz is not None:
                cam_to_frame[cam] = renderer.render_frame(
                    body_q=body_q,
                    left_active=left_active,
                    right_active=right_active,
                    root_quat_wxyz=root_quat_wxyz,
                )
            else:
                cam_to_frame[cam] = renderer.render_frame(
                    body_q=body_q,
                    left_active=left_active,
                    right_active=right_active,
                )
        # Alias the underlying frames into the modality dict.
        return {
            key: cam_to_frame[self._key_to_camera[key]]
            for key in self._required_keys
        }

    def close(self) -> None:
        for renderer in list(self._renderers.values()):
            try:
                renderer.close()
            except Exception:
                pass
        self._renderers.clear()


class _RealCameraProvider:
    """Background-polled wrapper around :class:`ComposedCameraClientSensor`.

    The underlying ZMQ SUB socket already uses ``CONFLATE=True`` so each
    ``read()`` returns the latest available frame (intermediate frames
    are dropped by the kernel). We add a short polling thread so the
    inference thread never blocks on socket I/O and so we can track the
    age of the freshest delivered frame independently of read calls
    (the client only updates its internal ``_last_new_message_time``
    when a non-blocking ``read()`` happens to arrive at the right
    moment).

    Required keys are validated against the publisher's first arriving
    payload at construction time (with a configurable warm-up window) so
    a misconfigured publisher fails fast instead of silently feeding
    zeros into the VLA.

    Frames are returned as HxWx3 uint8 **RGB** ndarrays — matching the
    format produced by :class:`gear_sonic.camera.sensor_server.ImageMessageSchema`
    (the JPEG bytes are decoded BGR via cv2 and converted to RGB on the
    client side; see ``sensor_server.py::ImageMessageSchema.deserialize``).
    The training data was captured with the same publisher pipeline so
    no further colour-space conversion is needed.
    """

    def __init__(
        self,
        *,
        server_ip: str,
        port: int,
        required_keys: list[str],
        warmup_timeout_s: float = 10.0,
        poll_period_s: float = 0.02,
    ) -> None:
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor

        self._required_keys = list(required_keys)
        self._client = ComposedCameraClientSensor(server_ip=server_ip, port=port)
        self._server_ip = server_ip
        self._port = port

        self._frames_lock = threading.Lock()
        self._latest_frames: dict[str, np.ndarray] = {}
        self._latest_t_mono: float = 0.0
        self._n_seen = 0
        self._stop_event = threading.Event()

        # Warm-up: block until we've received at least one payload that
        # contains *all* required keys, or fail loudly.
        print(
            f"[live-VLA] cameras: connecting to ZMQ PUB tcp://{server_ip}:{port} "
            f"(required keys: {self._required_keys}, warmup {warmup_timeout_s:.1f}s) …",
            flush=True,
        )
        deadline = time.monotonic() + warmup_timeout_s
        observed_keys: set[str] = set()
        while time.monotonic() < deadline:
            data = self._client.read(blocking=False)
            if data is not None and "images" in data:
                imgs = data["images"]
                observed_keys.update(imgs.keys())
                if all(k in imgs and imgs[k] is not None for k in self._required_keys):
                    with self._frames_lock:
                        self._latest_frames = {
                            k: np.ascontiguousarray(imgs[k])
                            for k in self._required_keys
                        }
                        self._latest_t_mono = time.monotonic()
                        self._n_seen += 1
                    h, w = self._latest_frames[self._required_keys[0]].shape[:2]
                    print(
                        f"[live-VLA] cameras: ready — "
                        f"keys={self._required_keys} "
                        f"shape=({h}x{w}x3 uint8 RGB)",
                        flush=True,
                    )
                    break
            time.sleep(0.05)
        else:
            self._client.close()
            raise RuntimeError(
                f"camera warm-up failed: no payload containing all required keys "
                f"{self._required_keys} within {warmup_timeout_s:.1f}s. "
                f"Keys actually observed on tcp://{server_ip}:{port}: "
                f"{sorted(observed_keys) if observed_keys else 'NONE (publisher silent?)'}. "
                f"Check: 'x2-pc2-cameras serve --status' on PC2, plus the PC2 publisher "
                f"args (--width / --height / --mount keys)."
            )

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            args=(poll_period_s,),
            name="vla-camera-poll",
            daemon=True,
        )
        self._poll_thread.start()

    def _poll_loop(self, period_s: float) -> None:
        period_s = max(period_s, 0.005)
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            try:
                data = self._client.read(blocking=False)
            except Exception as exc:
                print(f"[live-VLA] camera poll error: {exc}", flush=True)
                data = None
            if data is not None and "images" in data:
                imgs = data["images"]
                # Only accept payloads that have every required key
                # populated; partial payloads would force us to mix
                # frames from different timestamps in the VLA
                # observation, which is OOD vs the training data.
                if all(k in imgs and imgs[k] is not None for k in self._required_keys):
                    snapshot = {
                        k: np.ascontiguousarray(imgs[k])
                        for k in self._required_keys
                    }
                    with self._frames_lock:
                        self._latest_frames = snapshot
                        self._latest_t_mono = time.monotonic()
                        self._n_seen += 1
            next_tick += period_s
            slack = next_tick - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                next_tick = time.monotonic()

    def read_frames(self, *, max_age_s: float) -> dict[str, np.ndarray] | None:
        """Return the latest frames if fresher than ``max_age_s``, else None."""
        with self._frames_lock:
            if not self._latest_frames:
                return None
            age = time.monotonic() - self._latest_t_mono
            if age > max_age_s:
                return None
            return {k: v.copy() for k, v in self._latest_frames.items()}

    @property
    def frame_count(self) -> int:
        with self._frames_lock:
            return self._n_seen

    @property
    def latest_age_s(self) -> float:
        with self._frames_lock:
            if self._latest_t_mono == 0.0:
                return float("inf")
            return time.monotonic() - self._latest_t_mono

    def close(self) -> None:
        self._stop_event.set()
        try:
            self._poll_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._client.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # ZMQ PUB + 50 Hz bootstrap publisher start **before** modality / torch /
    # MuJoCo imports so ``run_x2_quest3_planner_stack.sh`` can gate on the
    # bind log line within milliseconds (conda env first-import can
    # otherwise stall 10+ s before any log appears). The stack runner also
    # defers sim deploy until after the recorder is publishing ``pose`` so
    # the C++ SUB never comes up on a silent port.
    state = _LatestState()
    chunk = _LatestChunk()
    wire_debug = _LatestWireDebug()
    stop_event = threading.Event()

    ctx = zmq.Context.instance()
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.setsockopt(zmq.SNDHWM, 10)
    pub_sock.setsockopt(zmq.LINGER, 0)
    pub_url = f"tcp://{args.pub_host}:{args.pub_port}"
    pub_sock.bind(pub_url)
    print(
        f"[live-VLA] pose PUB bound on {pub_url} (topic={args.pub_topic!r}) "
        f"— streaming bootstrap stand @ {args.rate:g} Hz while policy loads",
        flush=True,
    )
    if args.silent_wire:
        print(
            "[live-VLA] --silent-wire ENABLED: publisher loop runs at "
            f"{args.rate:g} Hz but every send() is skipped. The deploy "
            "should NEVER decode a body_pose frame in this mode and will "
            "hold the trained stand pose via its built-in prefill. Use "
            "this to validate the 'no upstream' fallback path.",
            flush=True,
        )

    # Load the planner's idle_stand primitive so the publisher can replay
    # it instead of repeating DEFAULT_STAND_POSE. We do this BEFORE
    # spawning the publisher thread so the very first wire frame already
    # carries idle_stand[0] -- the deploy's WAIT_FOR_CONTROL gate latches
    # against this frame, and any mismatch shows up as a tracker fight on
    # tick 0. If loading fails (PKL missing / corrupt / no idle_stand
    # bin), we keep the bridge alive on the static fallback and surface
    # a loud warning -- the operator can then re-curate primitives or
    # pass --idle-stand-pkl="" to silence the warning.
    idle_loop: Optional[_IdleStandLoop] = None
    if args.idle_stand_pkl:
        try:
            idle_loop = _load_idle_stand_loop(args.idle_stand_pkl)
            print(
                f"[live-VLA] idle_stand loop loaded from {args.idle_stand_pkl} "
                f"({idle_loop.n_frames} frames @ {args.rate:g} Hz, bin="
                f"{idle_loop.bin_name!r})",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[live-VLA] WARN: could not load idle_stand from "
                f"{args.idle_stand_pkl}: {exc}. Falling back to static "
                f"DEFAULT_STAND_POSE -- expect ~25 deg lean post band release.",
                flush=True,
            )
    else:
        print(
            "[live-VLA] --idle-stand-pkl is empty: using static "
            "DEFAULT_STAND_POSE wire reference (NOT recommended).",
            flush=True,
        )

    # One-shot deprecation warning for the legacy --sonic-checkpoint
    # alias. We scan sys.argv directly because argparse collapses the
    # alias onto the canonical dest before we ever see it.
    if "--sonic-checkpoint" in sys.argv:
        print(
            "[live-VLA] WARN: --sonic-checkpoint is DEPRECATED on this "
            "bridge. Use --motion-token-decoder instead (same .pt path; "
            "the new name says what the bridge actually does with it). "
            "See docs/source/references/x2_sonic_runtime_architecture.md.",
            flush=True,
        )

    # SONIC token-to-pose decoder. Loaded once on the main thread
    # before the publisher starts so the very first VLA chunk that
    # arrives can be decoded without a cold-start hitch. Any failure
    # here is non-fatal -- the publisher falls back to the idle wire
    # (same behaviour as before this feature was added) and the
    # operator can rerun without --motion-token-decoder to confirm.
    freeze_groups_override = args.vla_freeze_body_groups.strip()
    if args.vla_hands_only:
        print(
            "[live-VLA] WARN: --vla-hands-only is deprecated and ignored. "
            "Fingers-only mode was removed; manipulation now decodes "
            "arms+hands with legs/waist frozen.",
            flush=True,
        )
    body_mode = _parse_vla_body_mode(args.vla_body_mode)

    pose_decoder: Optional[Any] = None
    if body_mode is VlaBodyMode.MANIPULATION:
        print(
            "[live-VLA] body mode=manipulation (VR ARM_MANIPULATION analog): "
            "decode motion tokens for arms/head/hands; pin "
            f"{'custom ' if freeze_groups_override else ''}"
            f"freeze groups "
            f"{freeze_groups_override or 'legs,waist'} to idle_stand.",
            flush=True,
        )
    elif body_mode is VlaBodyMode.LOCOMOTION:
        print(
            "[live-VLA] body mode=locomotion (VR LOCOMOTION analog): "
            "full-body decode from motion tokens when decoder is loaded.",
            flush=True,
        )

    if args.motion_token_decoder and not args.no_policy:
        try:
            from gear_sonic.utils.teleop.sonic_token_to_pose_decoder import (
                SonicTokenToPoseDecoder,
            )
            pose_decoder = SonicTokenToPoseDecoder(
                args.motion_token_decoder,
                device=args.sonic_decoder_device,
            )
            print(
                f"[live-VLA] motion-token decoder loaded from "
                f"{args.motion_token_decoder} (device={args.sonic_decoder_device}). "
                "Wire joint_pos_mj will be VLA-decoded for chunks with "
                "|token|>1e-3; cold-start chunks (chunk_id=0, all-zero "
                "tokens) keep the idle_stand reference.",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[live-VLA] WARN: motion-token decoder load failed: {exc}. "
                "Wire joint_pos_mj will stay at idle_stand and the body "
                "will not move under VLA authority. Pass "
                "--motion-token-decoder to a known-good .pt to fix.",
                flush=True,
            )
    decoder_loaded = pose_decoder is not None
    if not decoder_loaded and not args.no_policy:
        if body_mode is VlaBodyMode.MANIPULATION:
            print(
                "[live-VLA] WARN: manipulation mode needs "
                "--motion-token-decoder; arms will stay on idle_stand until "
                "a decoder is loaded (hands still follow VLA).",
                flush=True,
            )
        elif body_mode is VlaBodyMode.LOCOMOTION:
            print(
                "[live-VLA] WARN: locomotion mode without decoder — body "
                "tracks idle_stand only. Pass --motion-token-decoder for "
                "full-body VLA.",
                flush=True,
            )
    elif args.no_policy:
        print(
            "[live-VLA] --no-policy mode: motion-token decoder skipped (no "
            "VLA tokens to decode).",
            flush=True,
        )
    if args.vla_mode_control_file:
        print(
            f"[live-VLA] runtime mode switch file: {args.vla_mode_control_file} "
            "(write manipulation | locomotion to flip modes).",
            flush=True,
        )

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"[live-VLA] caught signal {signum}, shutting down…", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ---- Optional vla_control SUB worker (manual-takeover) ---------
    # Only spawned when --vla-control-port > 0. The worker runs in a
    # daemon thread; it doesn't block the publisher even if the proxy
    # never publishes anything (poll-with-timeout loop). The shared
    # signal object is read at the top of every publisher tick.
    # CLI-level validation: a max-hold smaller than the minimum hold would
    # mean the bridge releases the wire BEFORE the proxy's HOLD ladder
    # finishes replaying the operator pose, defeating the whole point of
    # the handoff guard. Fail at startup, not at the first cold restart.
    if (
        int(args.vla_handoff_max_hold_ticks) > 0
        and int(args.vla_handoff_max_hold_ticks)
            < int(args.vla_cold_restart_hold_ticks)
    ):
        print(
            f"[live-VLA] FATAL: --vla-handoff-max-hold-ticks "
            f"({int(args.vla_handoff_max_hold_ticks)}) must be >= "
            f"--vla-cold-restart-hold-ticks "
            f"({int(args.vla_cold_restart_hold_ticks)}). The max-hold "
            f"is the safety cap on the await-first-chunk wait; "
            f"shorter than the minimum hold would release the wire "
            f"to idle_stand mid-proxy-HOLD.",
            flush=True,
        )
        sys.exit(2)

    vla_control_signal: Optional[_VlaControlSignal] = None
    vla_control_thread: Optional[threading.Thread] = None
    if int(args.vla_control_port) > 0:
        vla_control_signal = _VlaControlSignal()

        def _run_vla_control_sub_thread() -> None:
            _run_vla_control_sub(
                host=str(args.vla_control_host),
                port=int(args.vla_control_port),
                topic=str(args.vla_control_topic),
                signal=vla_control_signal,
                stop_event=stop_event,
            )

        vla_control_thread = threading.Thread(
            target=_run_vla_control_sub_thread,
            name="vla-control-sub",
            daemon=True,
        )
        vla_control_thread.start()
        print(
            f"[live-VLA] vla_control SUB enabled "
            f"(host={args.vla_control_host} "
            f"port={args.vla_control_port} "
            f"topic={args.vla_control_topic!r}; "
            f"cold_restart_hold_ticks="
            f"{int(args.vla_cold_restart_hold_ticks)}; "
            f"handoff_max_hold_ticks="
            f"{int(args.vla_handoff_max_hold_ticks)})",
            flush=True,
        )
    else:
        print(
            "[live-VLA] vla_control SUB DISABLED "
            "(--vla-control-port not set; manual-takeover cold "
            "restart inactive)",
            flush=True,
        )

    n_ticks_holder: list[int] = [0]

    def _run_publisher() -> None:
        n_ticks_holder[0] = _publisher(
            pub_sock=pub_sock, topic=args.pub_topic, rate_hz=args.rate,
            chunk=chunk, state=state, duration_s=args.duration,
            stop_event=stop_event,
            protocol_version=args.protocol_version,
            print_every=args.print_every,
            idle_loop=idle_loop,
            silent_wire=bool(args.silent_wire),
            pose_decoder=pose_decoder,
            ramp_in_ticks=int(args.vla_ramp_in_ticks),
            target_lpf_hz=float(args.vla_target_lpf_hz),
            future_lpf_hz=float(args.vla_future_lpf_hz),
            hand_lpf_hz=float(args.vla_hand_lpf_hz),
            hand_chunk_blend_ticks=int(args.vla_hand_chunk_blend_ticks),
            max_hand_step=float(args.vla_max_hand_step),
            max_wire_dev_from_body=float(args.vla_max_wire_dev_from_body),
            max_wire_step=float(args.vla_max_wire_step),
            chunk_blend_ticks=int(args.vla_chunk_blend_ticks),
            max_action_il=float(args.vla_max_action_il),
            decode_delay_ticks=int(args.vla_decode_delay_ticks),
            body_mode=body_mode,
            body_mode_control_file=args.vla_mode_control_file,
            decoder_loaded=decoder_loaded,
            freeze_groups_override=freeze_groups_override,
            wire_debug=wire_debug,
            vla_control_signal=vla_control_signal,
            cold_restart_hold_ticks=int(args.vla_cold_restart_hold_ticks),
            handoff_max_hold_ticks=int(args.vla_handoff_max_hold_ticks),
            handoff_max_wire_step=float(args.vla_handoff_max_wire_step),
            handoff_step_ramp_ticks=int(args.vla_handoff_step_ramp_ticks),
            tracking_feedback_enabled=bool(args.vla_tracking_feedback),
            tracking_soft_rad=float(args.vla_tracking_soft_rad),
            tracking_hard_rad=float(args.vla_tracking_hard_rad),
            tracking_vel_margin=float(args.vla_tracking_velocity_margin),
            tracking_vel_floor_rad_tick=float(
                args.vla_tracking_velocity_floor_rad_tick
            ),
            tracking_stale_ms=int(args.vla_tracking_stale_ms),
        )

    publisher_thread = threading.Thread(
        target=_run_publisher,
        name="pose-publisher",
        daemon=False,
    )
    publisher_thread.start()

    _validate_pin_order_or_die()
    _load_modality_config(args.modality_config)

    # ------------------------------------------------------------------
    # Camera source selection
    # ------------------------------------------------------------------
    # ``ghost``: the original sim path — build a MuJoCo EGL renderer
    # inside the inference thread (thread-local GL contexts require
    # that the renderer is constructed by the thread that will call
    # ``render_frame``).
    #
    # ``zmq``: real-robot path — subscribe to the PC2
    # ``ComposedCameraClientSensor`` PUB stream. The required video
    # keys are pulled from the registered modality config so a config
    # mismatch fails loudly here rather than mid-rollout.
    # ------------------------------------------------------------------
    ghost_provider_factory = None
    camera_provider: _RealCameraProvider | None = None

    if args.cameras_source == "ghost":
        # Read the registered modality config to discover which video
        # keys the policy expects. The omnihand-stereo config (used by
        # real-robot checkpoints) declares ``stereo_left + stereo_right``;
        # the legacy 10dof sim config declares just ``ego_view``. The
        # provider maps both into the right number of MJCF renderers
        # (one per unique mount) without changing the wire / observation
        # plumbing further downstream.
        required_keys = _get_required_video_keys(args.embodiment_tag)
        # Inference-side renderer must match the dataset camera +
        # resolution exactly (the VLA's vision encoder is
        # dimension-locked).
        render_w = int(args.render_width)
        render_h = int(args.render_height)
        with_omnihand = not args.no_omnihand

        def _ghost_factory() -> _GhostCameraProvider:
            provider = _GhostCameraProvider(
                required_keys=required_keys,
                width=render_w,
                height=render_h,
                with_omnihand=with_omnihand,
            )
            return provider.build()

        ghost_provider_factory = _ghost_factory
        print(
            f"[live-VLA] cameras: ghost mode — modality keys "
            f"{required_keys} -> MJCF cameras "
            f"{sorted(set(_GhostCameraProvider.MODALITY_TO_MJ_CAMERA.get(k, '?') for k in required_keys))} "
            f"({render_w}x{render_h}, omnihand={with_omnihand}). "
            "Stereo keys both alias the same ``stereo_head_front`` "
            "optical centre (degenerate stereo) -- enough to validate the "
            "loop in sim without breaking the real robot.",
            flush=True,
        )
    else:  # args.cameras_source == "zmq"
        required_keys = _get_required_video_keys(args.embodiment_tag)
        try:
            camera_provider = _RealCameraProvider(
                server_ip=args.cameras_zmq_host,
                port=args.cameras_zmq_port,
                required_keys=required_keys,
                warmup_timeout_s=args.cameras_warmup_s,
            )
        except BaseException:
            stop_event.set()
            publisher_thread.join(timeout=30.0)
            try:
                pub_sock.close(linger=0)
            except Exception:
                pass
            raise

    sub_url = f"tcp://{args.sub_host}:{args.sub_port}"

    video_threads: list[threading.Thread] = []
    if args.video_out:
        # Guard: ``--video-out`` requires a MuJoCo renderer, which only
        # exists in ghost mode. Arg parser already rejects this combo,
        # but defend in depth in case someone wires the function
        # directly.
        if args.cameras_source != "ghost":
            raise RuntimeError(
                "--video-out requires --cameras-source ghost (sim renderer)"
            )
        from gear_sonic.scripts.render_smoketest_episode_video import MujocoFrameRenderer  # noqa: F401

        def _make_video_renderer_factory(
            *, camera: str, width: int, height: int
        ):
            from gear_sonic.scripts.render_smoketest_episode_video import MujocoFrameRenderer as _MFR

            def _factory() -> Any:
                return _MFR(
                    camera=camera,
                    width=width,
                    height=height,
                    with_omnihand=not args.no_omnihand,
                    egl=True,
                )
            return _factory

        ego_video_factory = _make_video_renderer_factory(
            camera="ego_view",
            width=args.render_width,
            height=args.render_height,
        )
        video_threads.append(threading.Thread(
            target=_video_recorder,
            kwargs=dict(
                renderer_factory=ego_video_factory,
                state=state,
                output_path=args.video_out,
                fps=args.video_fps,
                stop_event=stop_event,
                chunk=chunk,
                verbose=not args.quiet,
            ),
            name="vla-video-ego",
            daemon=True,
        ))
    if args.video_front_out:
        if args.cameras_source != "ghost":
            raise RuntimeError(
                "--video-front-out requires --cameras-source ghost (sim renderer)"
            )
        from gear_sonic.scripts.render_smoketest_episode_video import MujocoFrameRenderer as _MFR

        def _front_factory() -> Any:
            return _MFR(
                camera=args.video_front_camera,
                width=args.video_front_width,
                height=args.video_front_height,
                with_omnihand=not args.no_omnihand,
                egl=True,
            )

        video_threads.append(threading.Thread(
            target=_video_recorder,
            kwargs=dict(
                renderer_factory=_front_factory,
                state=state,
                output_path=args.video_front_out,
                fps=args.video_fps,
                stop_event=stop_event,
                chunk=chunk,
                verbose=not args.quiet,
            ),
            name="vla-video-front",
            daemon=True,
        ))

    sub_thread = threading.Thread(
        target=_x2_debug_subscriber,
        kwargs=dict(
            sub_url=sub_url, topic=args.sub_topic,
            state=state, stop_event=stop_event,
        ),
        name="x2_debug-sub",
        daemon=True,
    )

    sub_thread.start()
    # PUB-SUB late-join: give the deploy a moment to wire up its SUB before
    # we start blasting messages it can't keep up with.
    time.sleep(0.2)

    # Lazy import — pulls in torch + transformers + Isaac-GR00T, which must
    # only happen after the modality-config side-load. Runs *after* PUB
    # bind + publisher thread so the deploy never idles without a pose
    # stream during this window.
    inf_thread: Optional[threading.Thread] = None
    if args.no_policy:
        print(
            "[live-VLA] --no-policy set: skipping Gr00tPolicy load + inference worker. "
            "Publisher + x2_debug SUB stay live; wire carries planner-like idle stand "
            "(live joint_pos_mj when x2_debug is fresh, canonical stand otherwise; "
            "motion_token + hand joints stay zero).",
            flush=True,
        )
    else:
        print("[live-VLA] loading Gr00tPolicy …", flush=True)
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        try:
            policy = Gr00tPolicy(
                embodiment_tag=args.embodiment_tag,
                model_path=args.model_path,
                device=args.device,
            )
        except BaseException:
            stop_event.set()
            publisher_thread.join(timeout=30.0)
            sub_thread.join(timeout=2.0)
            try:
                pub_sock.close(linger=0)
            except Exception:
                pass
            raise
        print(f"[live-VLA] policy ready (device={args.device})", flush=True)

        inf_thread = threading.Thread(
            target=_inference_worker,
            kwargs=dict(
                policy=policy,
                ghost_provider_factory=ghost_provider_factory,
                camera_provider=camera_provider,
                camera_max_age_s=args.cameras_staleness_s,
                state=state, chunk=chunk, wire_debug=wire_debug,
                prompt=args.prompt, stop_event=stop_event,
                min_period_s=args.inference_min_period_s, verbose=not args.quiet,
                dump_chunks_dir=args.dump_chunks_dir,
                dump_chunks_every=args.dump_chunks_every,
                wait_for_ready_file=args.wait_for_ready_file,
                wait_for_ready_file_timeout_s=(
                    args.wait_for_ready_file_timeout_s
                ),
            ),
            name="vla-inference",
            daemon=True,
        )
        inf_thread.start()
    for vt in video_threads:
        vt.start()

    try:
        publisher_thread.join()
    finally:
        stop_event.set()
        publisher_thread.join(timeout=5.0)
        sub_thread.join(timeout=1.0)
        if inf_thread is not None:
            inf_thread.join(timeout=2.0)
        # Video threads may need an extra beat to flush the encoder queue.
        for vt in video_threads:
            vt.join(timeout=15.0)
        if camera_provider is not None:
            try:
                camera_provider.close()
            except Exception:
                pass
        try:
            pub_sock.close(linger=0)
        except Exception:
            pass
        cam_summary = ""
        if camera_provider is not None:
            cam_summary = (
                f", camera_frames_seen={camera_provider.frame_count}"
            )
        print(
            f"[live-VLA] done after {n_ticks_holder[0]} pub ticks, "
            f"{chunk.inference_count} inferences, "
            f"last_inference_ms={chunk.last_inference_ms:.1f}"
            f"{cam_summary}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
