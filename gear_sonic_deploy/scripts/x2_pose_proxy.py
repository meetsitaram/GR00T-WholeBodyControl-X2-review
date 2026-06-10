#!/usr/bin/env python3
"""x2_pose_proxy.py -- PC2-side idle-fallback pose proxy.

Sits between the laptop pose publisher (planner stack / recorder /
live_vla_publish_motion_token bridge on port 5556) and the C++ deploy
on PC2. The C++ deploy SUBs to this proxy on localhost rather than to
the laptop directly.

When upstream is fresh:
    Forward laptop pose frames byte-for-byte to downstream. Zero
    protocol logic, zero re-encode -- the deploy sees the exact wire
    bytes the laptop sent.

When upstream is silent for > --idle-stale-ms (default 100):
    Run the staged fallback ladder (LIVE -> HOLD -> BLEND -> IDLE_CLIP)
    keyed by ``--idle-mode``. The 2026-06-08 default is ``blend``:

      * HOLD (default 10 s): re-publish the LAST forwarded upstream
        frame BYTE-FOR-BYTE. The deploy sees zero kinematic surprise
        (identical bytes -> identical joint_pos -> jvel = 0) and keeps
        commanding the operator's last pose. Soaks up WiFi blips /
        laptop GC pauses / Cursor reloads of up to ``--hold-last-secs``
        with no observable effect on the robot.
      * BLEND (default 3 s): lerp joint_pos_mj from the cached upstream
        frame toward the baked idle clip. Smooth glide rather than a
        step, so the arms drift to default over seconds rather than
        slamming through their full ROM in 200 ms.
      * IDLE_CLIP: after the hold + blend window expires, publish
        baked idle_stand frames indefinitely (the legacy destination
        behaviour, reached gradually).

    ``--idle-mode hold-last`` skips BLEND / IDLE_CLIP entirely and
    holds the cached upstream frame forever (operator owns recovery).

    ``--idle-mode idle-stand`` reproduces pre-2026-06-08 behaviour:
    jump to IDLE_CLIP on the first stale tick. Regression escape only
    -- known to slam tables when the operator has arms extended during
    a WiFi hiccup.

    The baked idle clip is the same one
    ``live_vla_publish_motion_token.py --no-policy`` publishes in idle
    mode (built by ``bake_idle_stand_x2m2.py``), so the policy sees one
    consistent reference distribution wherever the wire happens to
    come from.

    YAW REBASE (default ON): the baked idle_stand clip is yaw-aligned
    to ``R_z(0)`` for every frame. Publishing those frames verbatim
    while the robot is at a different heading hands the SONIC policy
    a stale absolute-yaw reference, and the tokenizer's
    ``rel = inv(measured) * reference`` computation makes the policy
    actively twist the body back to world +X -- the "robot snaps to
    spawn heading the moment I kill the planner stack" symptom. To
    fix this, the proxy SUBs to the C++ deploy's ``x2_debug`` PUB
    (default ``tcp://127.0.0.1:5557``, topic ``x2_debug``), extracts
    the live ``base_quat`` (IMU pelvis quat) on every tick, and
    pre-multiplies the baked clip's root quats by ``R_z(measured_yaw)``
    before publishing. Net effect: ``rel ~= identity`` -> policy
    holds whatever heading the body is currently in. Falls back to
    the last-known-good measured yaw (or to the baked ``R_z(0)`` if
    never received) whenever x2_debug is stale; pass
    ``--no-x2-debug-yaw-track`` to revert to the legacy
    "publish baked yaw verbatim" behaviour.

Why this exists:
    The C++ deploy in --input-type=zmq mode requires a continuous
    50 Hz pose-ref stream or its starvation watchdog trips into
    SAFE_IDLE (which commands default_angles with 4x kd -- a hard PD
    step that whirs the motors when current pose != default_angles).
    The laptop publisher already handles VR-driven vs idle internally
    (overlays operator pose on an idle_stand baseline), but if the
    LAPTOP PROCESS itself dies or wifi drops mid-run, the wire goes
    silent and SAFE_IDLE fires. The proxy guarantees the wire never
    goes silent from the deploy's perspective by sourcing its own
    fallback frames when upstream stops flowing.

    Critically: the proxy must NOT inject a step change in the
    commanded reference (which is what pre-2026-06-08 did by jumping
    straight to idle_stand on the first stale tick). The deploy's
    target LPF + max_target_dev clamps cannot absorb a multi-radian
    step in joint_pos_mj, so a WiFi hiccup with the operator's arms
    extended would swing them through their full ROM to default in
    ~200 ms -- known to slam tables. The staged HOLD -> BLEND -> IDLE
    ladder above keeps the wire alive while only making per-frame
    commanded-reference moves the deploy can actually track.

Single-thread design: one zmq.Context, one SUB, one PUB. The 50 Hz
tick loop polls the SUB non-blockingly, drains the queue (forwards
the LATEST frame if any), and fills in with an idle_stand frame when
upstream has been silent past the stale threshold. PUB-SUB is
inherently lossy on slow consumers; we set RCVHWM=SNDHWM=100 to keep
buffer pressure bounded.

Dependencies: numpy, pyzmq, stdlib. No gear_sonic / scipy / joblib.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import time
from pathlib import Path

import numpy as np
import zmq


# ---------------------------------------------------------------------------
# Wire-format constants (must mirror gear_sonic/scripts/live_vla_publish_motion_token.py
# and gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/zmq/zmq_pose_input_source.hpp).
# ---------------------------------------------------------------------------
NUM_BODY_DOFS: int = 31
SONIC_MOTION_TOKEN_DIM: int = 64
DEFAULT_HAND_DOF: int = 10
DEFAULT_PUB_RATE_HZ: float = 50.0
_FUTURE_DT_S: float = 0.1
_NUM_FUTURE_SLOTS: int = 9
# step_ticks at 50 Hz so adjacent future slots are 0.1 s apart -- matches
# planner.step_with_lookahead(step_ticks=5) and _IdleStandLoop.future_window.
_FUTURE_STEP_TICKS: int = int(round(DEFAULT_PUB_RATE_HZ * _FUTURE_DT_S))

# X2M2 binary format (see bake_idle_stand_x2m2.py + reference_motion.hpp).
X2M2_MAGIC: int = 0x58324D32

# ZMQ wire header (matches gear_sonic.utils.teleop.zmq.zmq_planner_sender).
HEADER_SIZE: int = 1280

# Pre-allocated zero slices for idle frames (avoid per-tick allocation).
_ZERO_MOTION_TOKEN = np.zeros(SONIC_MOTION_TOKEN_DIM, dtype=np.float32)
_ZERO_HAND = np.zeros(DEFAULT_HAND_DOF, dtype=np.float32)
_FUTURE_DT_FIELD = np.array([_FUTURE_DT_S], dtype=np.float32)
_ZERO_QVEL_FUTURE = np.zeros((_NUM_FUTURE_SLOTS, NUM_BODY_DOFS), dtype=np.float32)
_ZERO_QVEL_FUTURE.setflags(write=False)


# ---------------------------------------------------------------------------
# Inline x2_debug packed-binary decoder. The proxy historically had a
# numpy + pyzmq + stdlib only dependency budget (no gear_sonic / scipy),
# so we duplicate the minimum slice of
# ``gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder`` we actually
# need (pulling ``base_quat`` out of one frame) rather than importing
# the real decoder and dragging in scipy via blending.py.
#
# Wire format: see zmq_packed_message_decoder.py header comment.
#   [topic_bytes][1280-byte JSON header (NUL-padded)][concatenated binary fields]
# Header is JSON {"v":int, "endian":"le", "count":int, "fields":[{name,dtype,shape},...]}
# ---------------------------------------------------------------------------
X2_DEBUG_HEADER_SIZE: int = 1280

_DTYPE_BPE: dict[str, int] = {
    "f32": 4, "f64": 8, "i32": 4, "i64": 8, "u8": 1, "bool": 1,
}


def decode_x2_debug_base_quat(
    msg: bytes, topic: str = "x2_debug"
) -> np.ndarray | None:
    """Extract ``base_quat`` (wxyz, length 4, f64) from one x2_debug frame.

    Returns ``None`` on any decode failure (wrong topic, truncated
    payload, malformed header, missing/mistyped field). The proxy's
    publish thread MUST survive transient decoder failures -- a
    misshapen frame can't be allowed to wedge the wire and trip the
    deploy's starvation watchdog. Callers should treat ``None`` as
    "fall back to last known measured yaw".

    Walks fields in order and bails out the moment we read ``base_quat``;
    we never need any later field, so this stays O(prefix_of_header).
    """
    topic_bytes = topic.encode("utf-8")
    if not msg.startswith(topic_bytes):
        return None
    body = msg[len(topic_bytes):]
    if len(body) < X2_DEBUG_HEADER_SIZE:
        return None
    header_blob = body[:X2_DEBUG_HEADER_SIZE].rstrip(b"\x00")
    payload = body[X2_DEBUG_HEADER_SIZE:]
    try:
        header = json.loads(header_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    fields = header.get("fields")
    if not isinstance(fields, list):
        return None
    cursor = 0
    for f in fields:
        try:
            name = str(f["name"])
            dtype = str(f["dtype"])
            shape = tuple(int(s) for s in f["shape"])
        except (KeyError, TypeError, ValueError):
            return None
        bpe = _DTYPE_BPE.get(dtype)
        if bpe is None:
            return None
        nelem = 1
        for s in shape:
            nelem *= s
        nbytes = nelem * bpe
        if cursor + nbytes > len(payload):
            return None
        if name == "base_quat":
            if dtype != "f64" or shape != (4,):
                return None
            return np.frombuffer(
                payload[cursor:cursor + nbytes], dtype="<f8"
            ).copy()
        cursor += nbytes
    return None  # field absent from this frame


def decode_pose_joint_pos_mj(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``joint_pos_mj`` (f32, shape (NUM_BODY_DOFS,)) from a packed
    pose frame.

    Mirrors ``decode_x2_debug_base_quat``: tolerant of header noise,
    returns ``None`` on any decode failure rather than raising (the
    publish thread must survive a malformed cached frame -- the worst
    case is we lose the HOLD-vs-BLEND lerp anchor and fall back to the
    baked idle clip, which is no worse than the legacy behaviour).

    Called once per fresh upstream tick (~50 Hz) to snapshot the most
    recent operator-commanded joint targets. When upstream goes silent
    the snapshot is reused as (a) the byte payload to re-publish during
    HOLD and (b) the lerp anchor for BLEND.
    """
    topic_bytes = topic.encode("utf-8")
    if not msg.startswith(topic_bytes):
        return None
    body = msg[len(topic_bytes):]
    if len(body) < HEADER_SIZE:
        return None
    header_blob = body[:HEADER_SIZE].rstrip(b"\x00")
    payload = body[HEADER_SIZE:]
    try:
        header = json.loads(header_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    fields = header.get("fields")
    if not isinstance(fields, list):
        return None
    cursor = 0
    for f in fields:
        try:
            name = str(f["name"])
            dtype = str(f["dtype"])
            shape = tuple(int(s) for s in f["shape"])
        except (KeyError, TypeError, ValueError):
            return None
        bpe = _DTYPE_BPE.get(dtype)
        if bpe is None:
            return None
        nelem = 1
        for s in shape:
            nelem *= s
        nbytes = nelem * bpe
        if cursor + nbytes > len(payload):
            return None
        if name == "joint_pos_mj":
            if dtype != "f32" or len(shape) != 1 or shape[0] != NUM_BODY_DOFS:
                return None
            return np.frombuffer(
                payload[cursor:cursor + nbytes], dtype="<f4"
            ).copy()
        cursor += nbytes
    return None  # joint_pos_mj absent (e.g. token-only side-channel frame)


# ---------------------------------------------------------------------------
# Yaw extraction + yaw-rebase math. Closed-form so the proxy stays
# scipy-free (validated against scipy's Rotation.as_euler("zyx")[0] to
# 5.7e-14 deg over 2000 random rotations -- see commit message / unit
# tests). The convention matches every other yaw-touching site in the
# stack (gear_sonic.utils.planner.blending.yaw_of_quat_xyzw) so a yaw
# extracted here can be passed unchanged into kplanner / recorder /
# scipy code anywhere else.
# ---------------------------------------------------------------------------
def yaw_from_quat_wxyz(quat_wxyz: np.ndarray) -> float:
    """Extract world-z yaw (rad, in ``(-pi, pi]``) from a wxyz quat.

    Equivalent to ``scipy.spatial.transform.Rotation.from_quat(
    [qx,qy,qz,qw]).as_euler("zyx")[0]`` (extrinsic ZYX, lowercase).
    Derivation: for ``R = Rx(roll) @ Ry(pitch) @ Rz(yaw)``,
    ``yaw = atan2(-R[0][1], R[0][0])``; substituting the quat->matrix
    closed form gives the expression below.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(-1)
    if q.shape[0] != 4:
        raise ValueError(f"quat_wxyz must be length 4, got {q.shape[0]}")
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    return float(math.atan2(
        2.0 * (qw * qz - qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    ))


def rebase_quats_xyzw_by_yaw(
    quats_xyzw: np.ndarray, yaw_rad: float
) -> np.ndarray:
    """Pre-multiply a batch of xyzw quats by ``R_z(yaw)``.

    ``quats_xyzw`` is (N, 4); returns (N, 4) of the same dtype.
    Applied to the baked idle clip's (yaw=0-aligned) root quats so the
    published reference matches the robot's actual heading instead of
    world +X. Vectorized for batch efficiency since we rebase the
    current frame + 9 future-window slots on every idle tick.
    """
    q = np.asarray(quats_xyzw)
    if q.ndim != 2 or q.shape[1] != 4:
        raise ValueError(f"quats_xyzw must be (N, 4); got {q.shape}")
    half = 0.5 * float(yaw_rad)
    rzx, rzy, rzz, rzw = 0.0, 0.0, math.sin(half), math.cos(half)
    qx = q[:, 0].astype(np.float64, copy=False)
    qy = q[:, 1].astype(np.float64, copy=False)
    qz = q[:, 2].astype(np.float64, copy=False)
    qw = q[:, 3].astype(np.float64, copy=False)
    out = np.empty_like(q, dtype=q.dtype)
    out[:, 0] = (rzw * qx + rzx * qw + rzy * qz - rzz * qy).astype(q.dtype)
    out[:, 1] = (rzw * qy - rzx * qz + rzy * qw + rzz * qx).astype(q.dtype)
    out[:, 2] = (rzw * qz + rzx * qy - rzy * qx + rzz * qw).astype(q.dtype)
    out[:, 3] = (rzw * qw - rzx * qx - rzy * qy - rzz * qz).astype(q.dtype)
    return out


# ---------------------------------------------------------------------------
# X2M2 loader (matches the format bake_idle_stand_x2m2.py emits, which is
# itself byte-compatible with PklMotionReference::Load on the C++ side).
# ---------------------------------------------------------------------------
def load_x2m2(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    raw = path.read_bytes()
    if len(raw) < 20:
        raise ValueError(f"{path}: file too short ({len(raw)} bytes)")
    magic, num_frames, num_dofs = struct.unpack("<III", raw[:12])
    if magic != X2M2_MAGIC:
        raise ValueError(
            f"{path}: bad magic 0x{magic:08x}, expected 0x{X2M2_MAGIC:08x}"
        )
    if num_dofs != NUM_BODY_DOFS:
        raise ValueError(
            f"{path}: num_dofs={num_dofs}, expected {NUM_BODY_DOFS}"
        )
    fps = struct.unpack("<d", raw[12:20])[0]
    frame_bytes = (num_dofs + 4) * 8
    expected = 20 + num_frames * frame_bytes
    if len(raw) != expected:
        raise ValueError(
            f"{path}: payload len {len(raw)} != expected {expected}"
        )
    flat = np.frombuffer(raw[20:], dtype=np.float64).reshape(
        num_frames, num_dofs + 4
    )
    dof = np.ascontiguousarray(flat[:, :num_dofs], dtype=np.float32)
    quat = np.ascontiguousarray(flat[:, num_dofs:], dtype=np.float32)
    return dof, quat, fps


# ---------------------------------------------------------------------------
# Idle-stand replay (matches _IdleStandLoop in
# gear_sonic/scripts/live_vla_publish_motion_token.py, condensed).
# ---------------------------------------------------------------------------
class IdleStandReplay:
    def __init__(self, dof: np.ndarray, quat: np.ndarray) -> None:
        if dof.ndim != 2 or dof.shape[1] != NUM_BODY_DOFS:
            raise ValueError(
                f"dof must be (T, {NUM_BODY_DOFS}); got {dof.shape}"
            )
        if quat.ndim != 2 or quat.shape[1] != 4:
            raise ValueError(f"quat must be (T, 4) xyzw; got {quat.shape}")
        if dof.shape[0] != quat.shape[0]:
            raise ValueError(
                f"dof / quat length mismatch: {dof.shape[0]} vs {quat.shape[0]}"
            )
        if dof.shape[0] < 1:
            raise ValueError("idle clip is empty")
        self._dof = np.ascontiguousarray(dof, dtype=np.float32)
        self._quat = np.ascontiguousarray(quat, dtype=np.float32)
        self._n = int(dof.shape[0])

    @property
    def n_frames(self) -> int:
        return self._n

    def current(self, tick: int) -> tuple[np.ndarray, np.ndarray]:
        i = int(tick) % self._n
        return self._dof[i].copy(), self._quat[i].copy()

    def future_window(
        self, tick: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.array(
            [
                (int(tick) + (k + 1) * _FUTURE_STEP_TICKS) % self._n
                for k in range(_NUM_FUTURE_SLOTS)
            ],
            dtype=np.int64,
        )
        return self._dof[idx].copy(), self._quat[idx].copy(), _ZERO_QVEL_FUTURE


# ---------------------------------------------------------------------------
# Wire encoder (inlined copy of
# gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message so the
# proxy stands alone on PC2 without gear_sonic on PYTHONPATH).
# ---------------------------------------------------------------------------
def pack_pose_message(
    pose_data: dict, topic: str = "pose", version: int = 4
) -> bytes:
    fields: list[dict] = []
    binary_data: list[bytes] = []
    for key, value in pose_data.items():
        if not isinstance(value, np.ndarray):
            continue
        if value.dtype == np.float32:
            dtype_str = "f32"
        elif value.dtype == np.float64:
            dtype_str = "f64"
        elif value.dtype == np.int32:
            dtype_str = "i32"
        elif value.dtype == np.int64:
            dtype_str = "i64"
        elif value.dtype == bool:
            dtype_str = "bool"
        else:
            dtype_str = "f32"
            value = value.astype(np.float32)
        fields.append(
            {"name": key, "dtype": dtype_str, "shape": list(value.shape)}
        )
        if not value.flags["C_CONTIGUOUS"]:
            value = np.ascontiguousarray(value)
        if value.dtype.byteorder == ">":
            value = value.astype(value.dtype.newbyteorder("<"))
        binary_data.append(value.tobytes())
    header = {"v": version, "endian": "le", "count": 1, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > HEADER_SIZE:
        raise ValueError(f"Header too large: {len(header_json)} > {HEADER_SIZE}")
    header_bytes = header_json.ljust(HEADER_SIZE, b"\x00")
    return topic.encode("utf-8") + header_bytes + b"".join(binary_data)


def build_idle_frame_msg(
    replay: IdleStandReplay,
    tick: int,
    topic: str,
    *,
    yaw_rebase_rad: float | None = None,
    joint_pos_mj_override: np.ndarray | None = None,
) -> bytes:
    """Pack one idle-fallback frame, optionally yaw-rebased.

    ``yaw_rebase_rad`` (radians) is the current measured pelvis yaw the
    proxy snapshot from the latest fresh ``x2_debug`` frame. When
    provided, every published quat (current frame + 9 future slots) is
    pre-multiplied by ``R_z(yaw_rebase_rad)`` so the deploy's tokenizer
    sees ``rel = inv(measured) * R_z(measured) ~= identity`` and the
    policy holds whatever heading the body is currently in instead of
    twisting back to world +X. ``None`` falls back to the legacy
    "publish baked yaw verbatim" behaviour (kept for the
    ``--no-x2-debug-yaw-track`` regression escape and the unit tests).

    ``joint_pos_mj_override`` (optional, shape (NUM_BODY_DOFS,) f32) lets
    the BLEND state machine substitute a lerp between cached-upstream
    and the baked idle clip for the CURRENT frame's joint targets while
    reusing the rest of the frame (root_quat with yaw rebase, motion
    token zeros, hand zeros, future window). The future-window slots
    are intentionally left on the idle clip -- they are an advisory
    horizon for the policy, not the immediate command, and treating
    them as "blend toward idle in the future" matches what the lerped
    current frame is doing in the present.
    """
    cur_jpos, cur_quat = replay.current(tick)
    if joint_pos_mj_override is not None:
        override = np.asarray(joint_pos_mj_override, dtype=np.float32)
        if override.shape != (NUM_BODY_DOFS,):
            raise ValueError(
                "joint_pos_mj_override must have shape "
                f"({NUM_BODY_DOFS},); got {override.shape}"
            )
        cur_jpos = override
    jpos_future, quat_future, jvel_future = replay.future_window(tick)
    if yaw_rebase_rad is not None:
        cur_quat = rebase_quats_xyzw_by_yaw(
            cur_quat.reshape(1, 4), yaw_rebase_rad
        ).reshape(4)
        quat_future = rebase_quats_xyzw_by_yaw(quat_future, yaw_rebase_rad)
    fidx_future = np.array(
        [tick + (k + 1) for k in range(_NUM_FUTURE_SLOTS)],
        dtype=np.int64,
    )
    # Field order + dtypes mirror live_vla_publish_motion_token._publish_loop
    # so the deploy's ZmqPoseInputSource decodes our idle frames the same way
    # it decodes laptop bridge frames (no v5 fallback path).
    payload = {
        "joint_pos_mj": cur_jpos,
        "root_quat_xyzw": cur_quat,
        "motion_token": _ZERO_MOTION_TOKEN,
        "left_hand_joints": _ZERO_HAND,
        "right_hand_joints": _ZERO_HAND,
        "frame_index": np.array([tick], dtype=np.int64),
        "joint_pos_mj_future": jpos_future,
        "root_quat_xyzw_future": quat_future,
        "joint_vel_mj_future": jvel_future,
        "frame_index_future": fidx_future,
        "future_dt_s": _FUTURE_DT_FIELD,
    }
    return pack_pose_message(payload, topic=topic, version=4)


# ---------------------------------------------------------------------------
# Upstream-silent fallback state machine.
#
# When the laptop's pose stream goes quiet, the proxy must keep the wire
# alive (the C++ deploy in --input-type=zmq mode expects a continuous
# pose-ref stream). The naive answer is "republish the baked idle_stand
# clip the moment the wire stalls" -- which is what the proxy did before
# 2026-06-08. Failure mode: any WiFi blip during teleop steps the
# commanded reference from "operator pose, arms up" to "default stand,
# arms down" in one tick. The deploy's target LPF (8 Hz) plus
# max_target_dev_arm (1.5 rad) clamps cannot absorb a multi-radian step,
# so the arms swing through the full ROM in ~200 ms and slam into
# whatever is in front of the robot. Hence the staged fallback below:
#
#   LIVE       -> upstream fresh this tick; forward bytes verbatim.
#   COLD_IDLE  -> proxy has NEVER seen upstream; publish baked idle clip
#                 (startup; legacy behaviour preserved).
#   HOLD       -> upstream silent < hold_last_secs after stale threshold;
#                 re-publish the LAST forwarded upstream bytes. The
#                 deploy keeps tracking the operator's last pose with
#                 jvel=0 (identical bytes, no kinematic surprise).
#   BLEND      -> upstream silent past hold_last_secs but still within
#                 blend_secs window; lerp joint_pos_mj from cached
#                 upstream toward baked idle. The lerp is monotonic so
#                 the deploy sees a smooth (~3 s) glide rather than a
#                 step.
#   IDLE_CLIP  -> upstream silent past hold_last_secs + blend_secs;
#                 publish baked idle clip indefinitely (legacy
#                 destination behaviour, just reached gradually rather
#                 than in one tick).
#
# Mode selection via --idle-mode:
#   blend      -> the full ladder above (NEW DEFAULT; safe).
#   hold-last  -> HOLD forever; never transitions to BLEND/IDLE_CLIP.
#                 Operator-responsibility mode; use when you know
#                 upstream WILL come back (e.g. live VR teleop where a
#                 stale wire == cut the power).
#   idle-stand -> skip HOLD/BLEND entirely; jump to IDLE_CLIP on first
#                 stale tick. Reproduces pre-2026-06-08 behaviour
#                 exactly. Regression escape for the milestone doc.
# ---------------------------------------------------------------------------
STATE_LIVE: str = "LIVE"
STATE_COLD_IDLE: str = "COLD_IDLE"
STATE_HOLD: str = "HOLD"
STATE_BLEND: str = "BLEND"
STATE_IDLE_CLIP: str = "IDLE_CLIP"
STATE_GAP: str = "GAP"  # silent < stale_s; don't publish (deploy holds last)

IDLE_MODE_BLEND: str = "blend"
IDLE_MODE_HOLD_LAST: str = "hold-last"
IDLE_MODE_IDLE_STAND: str = "idle-stand"
_IDLE_MODES: tuple[str, ...] = (
    IDLE_MODE_BLEND,
    IDLE_MODE_HOLD_LAST,
    IDLE_MODE_IDLE_STAND,
)


def decide_fallback_state(
    *,
    have_upstream: bool,
    age_s: float,
    stale_s: float,
    hold_last_secs: float,
    blend_secs: float,
    idle_mode: str,
) -> tuple[str, float]:
    """Pure decision function for the no-fresh-upstream branch.

    Returns ``(target_state, blend_alpha)`` where ``blend_alpha`` is
    only meaningful for ``STATE_BLEND`` (0.0 = cached, 1.0 = idle).
    Split out as a pure function so the state transitions can be unit
    tested without spinning up ZMQ sockets.

    The "fallback clock" starts at ``stale_s``: ``HOLD`` runs from
    ``stale_s`` to ``stale_s + hold_last_secs``, ``BLEND`` from
    ``stale_s + hold_last_secs`` to ``stale_s + hold_last_secs +
    blend_secs``, and ``IDLE_CLIP`` afterwards. This way the on-the-wire
    behaviour matches what an operator would naively expect from the
    CLI knobs (e.g. ``--hold-last-secs=10`` means 10 s of HOLD, not
    "10 s minus stale threshold of HOLD").
    """
    if not have_upstream:
        return STATE_COLD_IDLE, 0.0
    if age_s <= stale_s:
        return STATE_GAP, 0.0
    fallback_age = age_s - stale_s
    if idle_mode == IDLE_MODE_IDLE_STAND:
        return STATE_IDLE_CLIP, 1.0
    if idle_mode == IDLE_MODE_HOLD_LAST:
        return STATE_HOLD, 0.0
    # blend mode (default).
    if fallback_age <= hold_last_secs:
        return STATE_HOLD, 0.0
    if fallback_age <= hold_last_secs + blend_secs:
        if blend_secs <= 0.0:
            return STATE_IDLE_CLIP, 1.0
        alpha = (fallback_age - hold_last_secs) / blend_secs
        if alpha < 0.0:
            alpha = 0.0
        elif alpha > 1.0:
            alpha = 1.0
        return STATE_BLEND, float(alpha)
    return STATE_IDLE_CLIP, 1.0


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--upstream-host",
        required=True,
        help="Laptop IP publishing the pose stream (e.g. 192.168.86.22).",
    )
    p.add_argument(
        "--upstream-port",
        type=int,
        default=5556,
        help="Laptop pose PUB port (default 5556).",
    )
    p.add_argument(
        "--upstream-topic",
        default="pose",
        help="ZMQ topic prefix on upstream (default 'pose').",
    )
    p.add_argument(
        "--downstream-host",
        default="*",
        help="Local bind iface for downstream PUB (default '*' = all).",
    )
    p.add_argument(
        "--downstream-port",
        type=int,
        default=5558,
        help="Local PUB port the C++ deploy SUBs to (default 5558). "
             "MUST match the deploy's --vla-zmq-port.",
    )
    p.add_argument(
        "--downstream-topic",
        default="pose",
        help="Topic prefix on downstream (must match deploy --vla-zmq-topic).",
    )
    p.add_argument(
        "--idle-x2m2",
        type=Path,
        required=True,
        help="Path to baked idle_stand.x2m2 binary on PC2.",
    )
    p.add_argument(
        "--idle-stale-ms",
        type=int,
        default=100,
        help="Switch to idle fallback after this many ms of upstream "
             "silence (default 100).",
    )
    p.add_argument(
        "--idle-mode",
        choices=list(_IDLE_MODES),
        default=IDLE_MODE_BLEND,
        help="Behaviour when upstream is silent past --idle-stale-ms. "
             "'blend' (NEW DEFAULT, safe): hold the last forwarded "
             "upstream frame for --hold-last-secs, then lerp toward the "
             "baked idle_stand clip over --blend-secs. 'hold-last': "
             "hold the last forwarded frame indefinitely (operator owns "
             "recovery -- robot stays in commanded pose forever). "
             "'idle-stand': pre-2026-06-08 behaviour; switch to the "
             "baked idle clip on the first stale tick (causes arms to "
             "slam to default on any wifi hiccup -- regression escape "
             "only).",
    )
    p.add_argument(
        "--hold-last-secs",
        type=float,
        default=10.0,
        help="How long (s) to hold the last forwarded upstream frame "
             "before transitioning toward idle (default 10.0). Only "
             "applies when --idle-mode=blend. Sized to absorb laptop "
             "GC pauses / Cursor reloads / WiFi outages of up to ~10 s "
             "without changing the commanded reference at all.",
    )
    p.add_argument(
        "--blend-secs",
        type=float,
        default=3.0,
        help="Duration (s) of the lerp from cached-upstream to baked "
             "idle_stand at the end of the hold window (default 3.0). "
             "Only applies when --idle-mode=blend. Bounded by the "
             "deploy's max_target_dev_arm clamp (1.5 rad / control "
             "period) -- 3 s is comfortably above the worst-case "
             "shoulder swing through 180 deg.",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=DEFAULT_PUB_RATE_HZ,
        help="Downstream publish rate when idle (default 50; matches "
             "deploy control loop). Upstream forwarding is event-driven "
             "and inherits whatever cadence the laptop publishes at.",
    )
    p.add_argument(
        "--status-every-s",
        type=float,
        default=5.0,
        help="Periodic status print interval (default 5s).",
    )
    p.add_argument(
        "--x2-debug-host",
        default="127.0.0.1",
        help="Host of the deploy's x2_debug PUB (default 127.0.0.1; "
             "the deploy is colocated on PC2 in onbot mode).",
    )
    p.add_argument(
        "--x2-debug-port",
        type=int,
        default=5557,
        help="Port of the deploy's x2_debug PUB (default 5557; the "
             "deploy spawns this only when --zmq-debug-port > 0).",
    )
    p.add_argument(
        "--x2-debug-topic",
        default="x2_debug",
        help="Topic prefix for the x2_debug PUB (default 'x2_debug').",
    )
    p.add_argument(
        "--x2-debug-max-age-s",
        type=float,
        default=0.5,
        help="Max age (s) of the latest x2_debug frame before we treat "
             "it as stale and fall back to the last-known-good measured "
             "yaw (default 0.5s; one watchdog window). Stale entries "
             "are silently ignored on each idle tick.",
    )
    p.add_argument(
        "--no-x2-debug-yaw-track",
        action="store_true",
        help="Disable x2_debug yaw tracking. Idle-fallback frames will "
             "publish the baked clip's R_z(0) root quat verbatim, which "
             "causes the deploy to twist the body back to world +X "
             "(spawn heading) on every IDLE entry. Regression-test "
             "escape for the diagnostic baseline; never use in prod.",
    )
    args = p.parse_args(argv)

    if args.hold_last_secs < 0.0:
        print(
            f"[pose_proxy] ERROR: --hold-last-secs must be >= 0, got "
            f"{args.hold_last_secs}",
            file=sys.stderr,
        )
        return 1
    if args.blend_secs < 0.0:
        print(
            f"[pose_proxy] ERROR: --blend-secs must be >= 0, got "
            f"{args.blend_secs}",
            file=sys.stderr,
        )
        return 1

    if not args.idle_x2m2.is_file():
        print(
            f"[pose_proxy] ERROR: idle X2M2 not found: {args.idle_x2m2}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[pose_proxy] loading idle X2M2 from {args.idle_x2m2}", flush=True
    )
    dof, quat, fps = load_x2m2(args.idle_x2m2)
    print(
        f"[pose_proxy] idle clip: {dof.shape[0]} frames @ {fps:g} Hz "
        f"({dof.shape[0] / fps:.2f} s loop)",
        flush=True,
    )
    replay = IdleStandReplay(dof, quat)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    upstream_url = f"tcp://{args.upstream_host}:{args.upstream_port}"
    sub.setsockopt(zmq.RCVHWM, 100)
    sub.connect(upstream_url)
    sub.setsockopt(zmq.SUBSCRIBE, args.upstream_topic.encode("utf-8"))

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 100)
    bind_url = f"tcp://{args.downstream_host}:{args.downstream_port}"
    pub.bind(bind_url)

    # Optional x2_debug SUB for measured-yaw tracking on idle frames.
    # Disabled in two cases:
    #   1) --no-x2-debug-yaw-track (operator opt-out)
    #   2) --x2-debug-port <= 0 (deploy isn't publishing x2_debug)
    # When disabled, idle frames publish the baked R_z(0) quat verbatim
    # (legacy / regression behaviour). When enabled but the wire is
    # silent, we fall back gracefully to the last-known-good yaw.
    yaw_sub: zmq.Socket | None = None
    yaw_track_enabled = (
        not args.no_x2_debug_yaw_track
    ) and int(args.x2_debug_port) > 0
    if yaw_track_enabled:
        yaw_sub = ctx.socket(zmq.SUB)
        yaw_sub.setsockopt(zmq.RCVHWM, 4)
        x2_debug_url = f"tcp://{args.x2_debug_host}:{args.x2_debug_port}"
        yaw_sub.connect(x2_debug_url)
        yaw_sub.setsockopt(
            zmq.SUBSCRIBE, args.x2_debug_topic.encode("utf-8")
        )

    print(
        f"[pose_proxy] upstream SUB:   {upstream_url} topic={args.upstream_topic!r}",
        flush=True,
    )
    print(
        f"[pose_proxy] downstream PUB: {bind_url} topic={args.downstream_topic!r}",
        flush=True,
    )
    if yaw_track_enabled:
        print(
            f"[pose_proxy] yaw-track SUB:  tcp://{args.x2_debug_host}:"
            f"{args.x2_debug_port} topic={args.x2_debug_topic!r} "
            f"(max_age={args.x2_debug_max_age_s:.3f}s)",
            flush=True,
        )
    else:
        reason = (
            "--no-x2-debug-yaw-track"
            if args.no_x2_debug_yaw_track
            else f"--x2-debug-port={args.x2_debug_port}"
        )
        print(
            f"[pose_proxy] yaw-track SUB:  DISABLED ({reason}); "
            f"idle frames will publish baked R_z(0) -- expect snap-back "
            f"to world +X on planner-stack termination",
            flush=True,
        )
    print(
        f"[pose_proxy] idle stale threshold: {args.idle_stale_ms} ms "
        f"(switch to idle fallback after {args.idle_stale_ms} ms of "
        f"upstream silence)",
        flush=True,
    )
    if args.idle_mode == IDLE_MODE_BLEND:
        print(
            f"[pose_proxy] idle mode: blend "
            f"(HOLD last frame for {args.hold_last_secs:.1f}s, then "
            f"BLEND to idle_stand over {args.blend_secs:.1f}s)",
            flush=True,
        )
    elif args.idle_mode == IDLE_MODE_HOLD_LAST:
        print(
            "[pose_proxy] idle mode: hold-last (republish last upstream "
            "frame indefinitely; operator owns recovery)",
            flush=True,
        )
    else:  # idle-stand
        print(
            "[pose_proxy] idle mode: idle-stand (LEGACY; arms snap to "
            "default on the first stale tick -- known to slam tables "
            "during WiFi hiccups)",
            flush=True,
        )

    period = 1.0 / max(args.rate_hz, 1e-6)
    stale_s = args.idle_stale_ms / 1000.0
    hold_last_secs = float(args.hold_last_secs)
    blend_secs = float(args.blend_secs)
    idle_mode = str(args.idle_mode)
    yaw_max_age_s = float(args.x2_debug_max_age_s)
    next_tick = time.monotonic()
    # last_upstream_s = -1.0 sentinel for "never received". The age check
    # below treats any negative value as "infinitely stale", so we begin
    # in COLD_IDLE until upstream proves it's alive (single first frame).
    last_upstream_s = -1.0
    # Last measured yaw + monotonic timestamp. -1.0 sentinel means
    # "no x2_debug frame ever decoded successfully". When we fall into
    # idle and have never seen the robot's measured yaw, the rebase is
    # skipped and we publish baked R_z(0) frames -- worst-case
    # equivalent to the legacy behaviour, never worse.
    last_measured_yaw_rad = 0.0
    last_measured_yaw_s = -1.0
    yaw_decode_failures = 0
    # Cache of the last forwarded upstream frame. ``last_upstream_msg``
    # is re-published verbatim in HOLD (so the deploy sees zero kinematic
    # surprise -- identical bytes -> identical joint_pos -> jvel = 0).
    # ``last_upstream_jpos`` is the decoded ``joint_pos_mj`` slice used
    # as the lerp anchor in BLEND. If decoding fails (malformed cached
    # frame, side-channel-only message), we still keep the raw bytes for
    # HOLD and just skip BLEND -- safer to glide a touch later than
    # crash the publish thread.
    last_upstream_msg: bytes | None = None
    last_upstream_jpos: np.ndarray | None = None
    cur_state = STATE_COLD_IDLE
    prev_state = STATE_COLD_IDLE
    tick = 0
    idle_tick = 0
    fwd_frames = 0
    idle_frames = 0
    idle_frames_with_rebase = 0
    hold_frames = 0
    blend_frames = 0
    gap_skips = 0  # ticks where stale window not yet crossed
    last_status_s = time.monotonic()

    print(
        "[pose_proxy] starting (initial state: COLD_IDLE; will switch to "
        "LIVE as soon as upstream publishes anything)",
        flush=True,
    )

    try:
        while True:
            now = time.monotonic()

            # Drain x2_debug queue first so the measured-yaw cache is
            # fresh by the time we decide what to publish on this tick.
            # Like the upstream drain, we only keep the latest frame --
            # measured yaw doesn't accumulate, only the most recent
            # sample matters. Decode failures are tolerated (count for
            # observability + keep cache stale rather than crash).
            if yaw_sub is not None:
                latest_debug = None
                while True:
                    try:
                        latest_debug = yaw_sub.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                if latest_debug is not None:
                    base_quat_wxyz = decode_x2_debug_base_quat(
                        latest_debug, args.x2_debug_topic
                    )
                    if base_quat_wxyz is not None:
                        try:
                            last_measured_yaw_rad = yaw_from_quat_wxyz(
                                base_quat_wxyz
                            )
                            last_measured_yaw_s = now
                        except (ValueError, TypeError):
                            yaw_decode_failures += 1
                    else:
                        yaw_decode_failures += 1

            # Drain upstream queue. We forward the latest frame each tick,
            # not every frame -- if the laptop publishes faster than we
            # tick, intermediate frames are intentionally dropped (the
            # deploy only ever sees the freshest reference anyway).
            latest = None
            while True:
                try:
                    latest = sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break

            if latest is not None:
                # Got fresh upstream this tick -- forward raw bytes.
                try:
                    pub.send(latest, zmq.NOBLOCK)
                    fwd_frames += 1
                except zmq.Again:
                    pass
                # Cache for the upstream-silent fallback. Raw bytes are
                # used verbatim in HOLD; the decoded joint_pos_mj slice
                # is the lerp anchor in BLEND. A decode miss isn't fatal
                # -- HOLD still works (we re-publish raw bytes), only
                # BLEND degrades to "snap to idle" once the hold window
                # expires.
                last_upstream_msg = latest
                jpos = decode_pose_joint_pos_mj(
                    latest, args.upstream_topic
                )
                if jpos is None:
                    jpos = decode_pose_joint_pos_mj(
                        latest, args.downstream_topic
                    )
                if jpos is not None:
                    last_upstream_jpos = jpos
                cur_state = STATE_LIVE
                last_upstream_s = now
            else:
                # No upstream this tick. Decide what to fill in with via
                # the staged fallback ladder (LIVE -> HOLD -> BLEND ->
                # IDLE_CLIP) so a WiFi blip doesn't step the commanded
                # reference and slam the arms to default in one tick.
                age = (
                    float("inf") if last_upstream_s < 0
                    else (now - last_upstream_s)
                )
                target_state, blend_alpha = decide_fallback_state(
                    have_upstream=(last_upstream_msg is not None),
                    age_s=age,
                    stale_s=stale_s,
                    hold_last_secs=hold_last_secs,
                    blend_secs=blend_secs,
                    idle_mode=idle_mode,
                )
                cur_state = target_state

                if target_state == STATE_GAP:
                    # Still within stale window after the last upstream;
                    # send nothing this tick. The deploy's input source
                    # caches the last forwarded frame and Sample()
                    # returns it at 500 Hz; one missing 20 ms slice is
                    # invisible to the policy.
                    gap_skips += 1
                elif target_state == STATE_HOLD:
                    if last_upstream_msg is None:
                        # Belt-and-braces: decide_fallback_state would
                        # not return HOLD without have_upstream=True.
                        # If it does, glide gracefully to COLD_IDLE.
                        cur_state = STATE_COLD_IDLE
                    else:
                        try:
                            pub.send(last_upstream_msg, zmq.NOBLOCK)
                            hold_frames += 1
                        except zmq.Again:
                            pass
                elif target_state == STATE_BLEND:
                    yaw_rebase: float | None = None
                    if (
                        yaw_track_enabled
                        and last_measured_yaw_s >= 0
                        and (now - last_measured_yaw_s) <= yaw_max_age_s
                    ):
                        yaw_rebase = last_measured_yaw_rad
                    if last_upstream_jpos is None:
                        # No decode anchor for the lerp; fall back to
                        # the baked idle clip rather than blending from
                        # garbage. Still smoother than the legacy
                        # behaviour because we only reach BLEND after
                        # the full hold window already elapsed.
                        msg = build_idle_frame_msg(
                            replay,
                            idle_tick,
                            args.downstream_topic,
                            yaw_rebase_rad=yaw_rebase,
                        )
                    else:
                        idle_jpos, _ = replay.current(idle_tick)
                        lerp = (
                            (1.0 - blend_alpha) * last_upstream_jpos
                            + blend_alpha * idle_jpos
                        ).astype(np.float32)
                        msg = build_idle_frame_msg(
                            replay,
                            idle_tick,
                            args.downstream_topic,
                            yaw_rebase_rad=yaw_rebase,
                            joint_pos_mj_override=lerp,
                        )
                    try:
                        pub.send(msg, zmq.NOBLOCK)
                        blend_frames += 1
                    except zmq.Again:
                        pass
                    idle_tick += 1
                elif target_state in (STATE_COLD_IDLE, STATE_IDLE_CLIP):
                    yaw_rebase = None
                    if (
                        yaw_track_enabled
                        and last_measured_yaw_s >= 0
                        and (now - last_measured_yaw_s) <= yaw_max_age_s
                    ):
                        yaw_rebase = last_measured_yaw_rad
                    msg = build_idle_frame_msg(
                        replay,
                        idle_tick,
                        args.downstream_topic,
                        yaw_rebase_rad=yaw_rebase,
                    )
                    try:
                        pub.send(msg, zmq.NOBLOCK)
                        idle_frames += 1
                        if yaw_rebase is not None:
                            idle_frames_with_rebase += 1
                    except zmq.Again:
                        pass
                    idle_tick += 1

            # Emit a one-line transition log every time the state name
            # changes. Operators rely on this to correlate "I saw the
            # robot arm freeze for a few seconds then drift to stand"
            # with the proxy's own view of upstream availability.
            if cur_state != prev_state:
                if cur_state == STATE_LIVE:
                    if last_upstream_s < 0 or prev_state == STATE_COLD_IDLE:
                        msg_txt = (
                            f"{prev_state} -> LIVE (first upstream "
                            f"frame received)"
                        )
                    else:
                        gap_ms = (now - last_upstream_s) * 1000.0
                        msg_txt = (
                            f"{prev_state} -> LIVE (upstream pose "
                            f"frames flowing again after {gap_ms:.0f} "
                            f"ms gap)"
                        )
                elif cur_state == STATE_GAP:
                    msg_txt = (
                        f"{prev_state} -> GAP (upstream silent < "
                        f"{args.idle_stale_ms} ms; holding deploy "
                        f"cache)"
                    )
                elif cur_state == STATE_HOLD:
                    msg_txt = (
                        f"{prev_state} -> HOLD (re-publishing last "
                        f"upstream frame; will hold for "
                        f"{hold_last_secs:.1f}s)"
                    )
                elif cur_state == STATE_BLEND:
                    msg_txt = (
                        f"{prev_state} -> BLEND (lerping cached -> "
                        f"idle_stand over {blend_secs:.1f}s)"
                    )
                elif cur_state == STATE_IDLE_CLIP:
                    msg_txt = (
                        f"{prev_state} -> IDLE_CLIP (upstream silent "
                        f"past hold + blend window; tracking baked "
                        f"idle clip)"
                    )
                elif cur_state == STATE_COLD_IDLE:
                    msg_txt = f"{prev_state} -> COLD_IDLE"
                else:
                    msg_txt = f"{prev_state} -> {cur_state}"
                print(f"[pose_proxy] state: {msg_txt}", flush=True)
                # Reset idle_tick when (re-)entering an idle-clip path
                # so the looped baked clip starts at frame 0. Mostly
                # cosmetic but keeps logs predictable across restarts
                # of the fallback ladder.
                if (
                    cur_state in (STATE_IDLE_CLIP, STATE_COLD_IDLE, STATE_BLEND)
                    and prev_state not in (
                        STATE_IDLE_CLIP, STATE_COLD_IDLE, STATE_BLEND
                    )
                ):
                    idle_tick = 0
                prev_state = cur_state

            tick += 1

            if now - last_status_s >= args.status_every_s:
                age = (
                    float("inf") if last_upstream_s < 0
                    else (now - last_upstream_s)
                )
                age_str = (
                    "never" if last_upstream_s < 0
                    else f"{age * 1000:.0f}ms"
                )
                # Decorate the state with the fallback-clock timer so
                # operators reading the status line can see "I'm 4.2 s
                # into a 10 s hold" or "blend alpha=0.41" at a glance.
                if cur_state == STATE_HOLD and last_upstream_s >= 0:
                    fb_age = max(0.0, age - stale_s)
                    state_str = (
                        f"HOLD t={fb_age:.1f}/{hold_last_secs:.1f}s"
                    )
                elif cur_state == STATE_BLEND and last_upstream_s >= 0:
                    fb_age = max(0.0, age - stale_s)
                    if blend_secs > 0.0:
                        alpha = (fb_age - hold_last_secs) / blend_secs
                        alpha = max(0.0, min(1.0, alpha))
                    else:
                        alpha = 1.0
                    state_str = f"BLEND alpha={alpha:.2f}"
                else:
                    state_str = cur_state
                if not yaw_track_enabled:
                    yaw_str = "off"
                elif last_measured_yaw_s < 0:
                    yaw_str = "never"
                else:
                    yaw_age_ms = (now - last_measured_yaw_s) * 1000.0
                    yaw_str = (
                        f"yaw={math.degrees(last_measured_yaw_rad):+.1f}deg "
                        f"age={yaw_age_ms:.0f}ms"
                    )
                print(
                    f"[pose_proxy] tick={tick} state={state_str} "
                    f"mode={idle_mode} upstream_age={age_str} "
                    f"fwd={fwd_frames} hold={hold_frames} "
                    f"blend={blend_frames} idle={idle_frames} "
                    f"idle_rebased={idle_frames_with_rebase} "
                    f"gap_skip={gap_skips} "
                    f"x2_debug=({yaw_str}) "
                    f"yaw_decode_fail={yaw_decode_failures}",
                    flush=True,
                )
                last_status_s = now

            # Sleep to next 50 Hz boundary.
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # We're behind schedule (heavy GC, scheduler hiccup).
                # Reset baseline rather than spinning to catch up.
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("[pose_proxy] SIGINT received; tearing down", flush=True)
    finally:
        sub.close(linger=0)
        pub.close(linger=0)
        ctx.term()

    print(
        f"[pose_proxy] done. total_ticks={tick} fwd={fwd_frames} "
        f"hold={hold_frames} blend={blend_frames} idle={idle_frames} "
        f"gap_skip={gap_skips}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
