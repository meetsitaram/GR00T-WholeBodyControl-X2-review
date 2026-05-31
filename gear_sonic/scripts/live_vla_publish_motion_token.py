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
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Optional

import joblib
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


def _build_vla_decoded_pose_payload(
    *,
    decoder: Any,
    proprio_990: np.ndarray,
    token_chunk: np.ndarray,
    chunk_step: int,
    horizon: int,
    base_frame_index: int,
) -> Optional[tuple[np.ndarray, dict[str, np.ndarray]]]:
    """Decode VLA tokens to body trajectory + build wire payload.

    Pulls the current step's motion_token plus 9 future steps spaced
    :data:`_FUTURE_STEP_TICKS` apart from ``token_chunk``, runs them
    through :class:`SonicTokenToPoseDecoder` (which mirrors the C++
    deploy's ``target_mj = default + action_il * action_scale``
    formula), and returns

    * ``joint_pos_mj_now``: ``(31,) float32`` MuJoCo-order pose for
      the current tick;
    * ``future_fields``: dict matching :func:`_idle_future_payload_fields`
      shape (``joint_pos_mj_future`` etc.) with the next 9 decoded
      poses.

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
            decode_token_chunk_to_pose_chunk,
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
        # slots in the same call.
        poses = decode_token_chunk_to_pose_chunk(
            decoder, sampled_tokens.astype(np.float32), proprio_990
        )
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
    return joint_pos_mj_now, future_fields


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
    ) -> None:
        with self.cv:
            self.body_q_mj = body_q_mj.astype(np.float64, copy=False)
            self.base_quat_wxyz = base_quat_wxyz.astype(np.float64, copy=False)
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


def _build_observation(
    *,
    body_q_mj: np.ndarray,
    base_quat_wxyz: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    ego_view: np.ndarray,
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
    * ``video.ego_view``           = (1, 1, H, W, 3) uint8
    * ``language.annotation.human.task_description`` = [[prompt]]

    Note: head joints (pin[15:17]) are *not* exposed to the policy; they
    are intentionally omitted from ``modality.json``.
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
    return {
        "video": {"ego_view": ego_view.reshape(1, 1, *ego_view.shape).astype(np.uint8)},
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
            left_hq = np.asarray(msg.fields.get("left_hand_q", np.zeros(DEFAULT_HAND_DOF)), dtype=np.float64).reshape(-1)[:DEFAULT_HAND_DOF]
            right_hq = np.asarray(msg.fields.get("right_hand_q", np.zeros(DEFAULT_HAND_DOF)), dtype=np.float64).reshape(-1)[:DEFAULT_HAND_DOF]
            if left_hq.shape[0] < DEFAULT_HAND_DOF:
                left_hq = np.pad(left_hq, (0, DEFAULT_HAND_DOF - left_hq.shape[0]))
            if right_hq.shape[0] < DEFAULT_HAND_DOF:
                right_hq = np.pad(right_hq, (0, DEFAULT_HAND_DOF - right_hq.shape[0]))

            state.update(
                body_q_mj=body_q,
                base_quat_wxyz=base_quat,
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
    renderer_factory: Any,
    state: _LatestState,
    chunk: _LatestChunk,
    prompt: str,
    stop_event: threading.Event,
    min_period_s: float,
    verbose: bool = False,
    dump_chunks_dir: str | None = None,
    dump_chunks_every: int = 5,
) -> None:
    """Thread B: render + run VLA continuously, post fresh chunks.

    We don't run faster than ``1/min_period_s`` Hz to avoid burning GPU
    cycles when the deploy hasn't even consumed the previous chunk yet.

    The MuJoCo renderer is constructed *inside* this thread because EGL
    contexts are thread-local: a renderer instantiated in the main thread
    raises ``EGLError(EGL_BAD_DISPLAY)`` the first time another thread
    tries to call ``render_frame``. Building it here owns the EGL context
    on the same thread that will use it for the entire process lifetime.
    """
    print("[live-VLA] inference thread: building MujocoFrameRenderer …", flush=True)
    try:
        renderer = renderer_factory()
    except Exception as exc:
        print(f"[live-VLA] FATAL: renderer init failed in worker: {exc}", flush=True)
        stop_event.set()
        return
    print(
        f"[live-VLA] inference thread: renderer ready "
        f"({renderer.width}x{renderer.height}, omnihand={renderer.with_omnihand})",
        flush=True,
    )

    last_revision = -1
    n_inferences = 0
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
        try:
            # The inference ego-view deliberately uses the renderer's
            # default identity root quat -- the VLA training set
            # (record_synthetic_smoketest_dataset.py) renders frames the
            # same way, so feeding a live ``base_quat`` here would
            # inject a tilted horizon the policy has never seen and
            # cause OOD behaviour mid-rollout. The *video recorder*
            # thread does pass the live ``base_quat`` so the recording
            # reflects physical reality (tip / fall) -- those two
            # renderers are intentionally asymmetric.
            ego_view = renderer.render_frame(
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

        observation = _build_observation(
            body_q_mj=body_q_mj,
            base_quat_wxyz=base_quat_wxyz,
            left_hand_q=left_hq,
            right_hand_q=right_hq,
            ego_view=ego_view,
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
            print(
                "[live-VLA] invalid policy output (shape or non-finite) "
                "→ zero-token safe chunk posted",
                flush=True,
            )
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
                np.savez_compressed(
                    out_path,
                    token=token,
                    left_hand=left,
                    right_hand=right,
                    body_q_mj=body_q_mj.astype(np.float32),
                    base_quat_wxyz=base_quat_wxyz.astype(np.float32),
                    left_hand_q_obs=left_hq.astype(np.float32),
                    right_hand_q_obs=right_hq.astype(np.float32),
                    ego_view=ego_view.astype(np.uint8),
                    elapsed_ms=np.array([elapsed_ms], dtype=np.float32),
                    revision=np.array([revision], dtype=np.int64),
                    n_inference=np.array([n_inferences - 1], dtype=np.int64),
                    wall_t_s=np.array([time.time()], dtype=np.float64),
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

    try:
        renderer.close()
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

    while not stop_event.is_set() and time.monotonic() < deadline:
        _, _, _, _, _, deploy_fresh = state.snapshot()
        token, left, right, _body_pose_chunk, chunk_id = chunk.read()
        horizon = int(token.shape[0])
        if chunk_id != last_chunk_id:
            chunk_step = 0
            last_chunk_id = chunk_id

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
        if (
            pose_decoder is not None
            and deploy_fresh
            and chunk_id > 0
            and np.linalg.norm(token[step]) > 1e-3
        ):
            decoded = _build_vla_decoded_pose_payload(
                decoder=pose_decoder,
                proprio_990=_PROPRIO_ZERO_990,
                token_chunk=token,
                chunk_step=step,
                horizon=horizon,
                base_frame_index=tick,
            )
            if decoded is not None:
                decoded_now, future_fields = decoded
                cur_jpos = decoded_now
                # cur_quat stays as the idle_loop / identity choice --
                # the SONIC body-only decoder doesn't predict root.

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
            payload = {
                "joint_pos_mj": cur_jpos,
                "root_quat_xyzw": cur_quat,
                "motion_token": token[step],
                "left_hand_joints": left[step],
                "right_hand_joints": right[step],
                "frame_index": np.array([tick], dtype=np.int64),
                **future_fields,
            }
        if not silent_wire:
            msg = pack_pose_message(payload, topic=topic, version=protocol_version)
            try:
                pub_sock.send(msg, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

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
                    # at the idle setpoint).
                    if idle_loop is not None:
                        idle_now, _ = idle_loop.current(tick)
                    else:
                        idle_now = _default_stand_body_pose_f32()
                    pose_delta = float(
                        np.abs(
                            decoded_now.astype(np.float32)
                            - idle_now.astype(np.float32)
                        ).max()
                    )
                    decoded_tag = f"VLA-pose Δ={pose_delta:.3f}rad"
                else:
                    decoded_tag = "idle-pose"
                print(
                    f"[live-VLA] pub tick={tick:6d} "
                    f"chunk_id={chunk_id:4d} step={step:2d}/{horizon} "
                    f"|token|={float(np.linalg.norm(token[step])):.3f} "
                    f"|left|={float(np.linalg.norm(left[step])):.3f} "
                    f"{decoded_tag} "
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
        "--sonic-checkpoint", type=str, default=None,
        help="Path to a SONIC .pt checkpoint (e.g. model_step_025000.pt) used "
             "to decode the predicted motion_token chunks back into "
             "joint_pos_mj poses on the wire. Without this, the bridge emits "
             "idle_stand for joint_pos_mj on every tick and the C++ deploy's "
             "fused encoder+FSQ+decoder ONNX re-tokenises that idle reference "
             "and ignores the live motion_token field (header explicitly "
             "documents 'motion_token: currently logged but otherwise unused' "
             "-- see zmq_pose_input_source.hpp:22-25), so the body never "
             "moves under VLA authority. With --sonic-checkpoint set, each "
             "publish tick decodes chunk[step] (and 9 future steps) via the "
             "g1_dyn decoder + the C++ deploy's "
             "target_mj=default+action_il*scale formula and ships the result "
             "as joint_pos_mj / joint_pos_mj_future. The deploy's encoder "
             "then re-tokenises this VLA-driven trajectory, which makes the "
             "body actually track the predicted motion. Recommended path: "
             "the .pt that pairs with the deploy ONNX you're running.",
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
    args = parser.parse_args(argv)
    if not args.no_policy and not args.model_path:
        parser.error("--model-path is required unless --no-policy is set")
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

    # SONIC token-to-pose decoder. Loaded once on the main thread
    # before the publisher starts so the very first VLA chunk that
    # arrives can be decoded without a cold-start hitch. Any failure
    # here is non-fatal -- the publisher falls back to the idle wire
    # (same behaviour as before this feature was added) and the
    # operator can rerun without --sonic-checkpoint to confirm.
    pose_decoder: Optional[Any] = None
    if args.sonic_checkpoint and not args.no_policy:
        try:
            from gear_sonic.utils.teleop.sonic_token_to_pose_decoder import (
                SonicTokenToPoseDecoder,
            )
            pose_decoder = SonicTokenToPoseDecoder(
                args.sonic_checkpoint,
                device=args.sonic_decoder_device,
            )
            print(
                f"[live-VLA] SONIC pose decoder loaded from "
                f"{args.sonic_checkpoint} (device={args.sonic_decoder_device}). "
                "Wire joint_pos_mj will be VLA-decoded for chunks with "
                "|token|>1e-3; cold-start chunks (chunk_id=0, all-zero "
                "tokens) keep the idle_stand reference.",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[live-VLA] WARN: SONIC decoder load failed: {exc}. "
                "Wire joint_pos_mj will stay at idle_stand and the body "
                "will not move under VLA authority. Pass --sonic-checkpoint "
                "to a known-good .pt to fix.",
                flush=True,
            )
    elif args.no_policy:
        print(
            "[live-VLA] --no-policy mode: SONIC pose decoder skipped (no "
            "VLA tokens to decode).",
            flush=True,
        )
    else:
        print(
            "[live-VLA] --sonic-checkpoint not set: VLA motion_token will be "
            "published on the wire but the C++ deploy ignores that field "
            "(see zmq_pose_input_source.hpp:22-25). Body will track idle "
            "stand only; hands still get the VLA chunk's per-tick targets "
            "via AimDK passthrough. Pass --sonic-checkpoint to make the "
            "body move under VLA control.",
            flush=True,
        )

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"[live-VLA] caught signal {signum}, shutting down…", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

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
        )

    publisher_thread = threading.Thread(
        target=_run_publisher,
        name="pose-publisher",
        daemon=False,
    )
    publisher_thread.start()

    _validate_pin_order_or_die()
    _load_modality_config(args.modality_config)

    # The MuJoCo renderer is constructed inside the inference thread (see
    # ``_inference_worker``): MuJoCo's EGL backend uses thread-local GL
    # contexts, so a renderer built here would raise EGLError the first
    # time the worker tried to render. We hand workers a zero-arg factory
    # instead. Each video thread / inference thread gets its own EGL
    # context via its own factory invocation.
    from gear_sonic.scripts.render_smoketest_episode_video import MujocoFrameRenderer

    def _make_renderer_factory(
        *, camera: str, width: int, height: int
    ):
        def _factory() -> MujocoFrameRenderer:
            return MujocoFrameRenderer(
                camera=camera,
                width=width,
                height=height,
                with_omnihand=not args.no_omnihand,
                egl=True,
            )
        return _factory

    # Inference-side renderer must match the dataset camera + resolution
    # exactly (the VLA's vision encoder is dimension-locked).
    _renderer_factory = _make_renderer_factory(
        camera="ego_view",
        width=args.render_width,
        height=args.render_height,
    )

    sub_url = f"tcp://{args.sub_host}:{args.sub_port}"

    video_threads: list[threading.Thread] = []
    if args.video_out:
        video_threads.append(threading.Thread(
            target=_video_recorder,
            kwargs=dict(
                renderer_factory=_renderer_factory,
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
        front_factory = _make_renderer_factory(
            camera=args.video_front_camera,
            width=args.video_front_width,
            height=args.video_front_height,
        )
        video_threads.append(threading.Thread(
            target=_video_recorder,
            kwargs=dict(
                renderer_factory=front_factory,
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
                policy=policy, renderer_factory=_renderer_factory,
                state=state, chunk=chunk,
                prompt=args.prompt, stop_event=stop_event,
                min_period_s=args.inference_min_period_s, verbose=not args.quiet,
                dump_chunks_dir=args.dump_chunks_dir,
                dump_chunks_every=args.dump_chunks_every,
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
        try:
            pub_sock.close(linger=0)
        except Exception:
            pass
        print(
            f"[live-VLA] done after {n_ticks_holder[0]} pub ticks, "
            f"{chunk.inference_count} inferences, "
            f"last_inference_ms={chunk.last_inference_ms:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
