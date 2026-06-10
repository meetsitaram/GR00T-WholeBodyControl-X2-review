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

# msgpack is the wire format the quest3_manager_x2 uses for its
# ``stream_mode`` topic. Importing it lazily so the proxy still runs
# on a deployment that doesn't ship msgpack -- in that case the
# mode-SUB feature stays off and the proxy falls back to the legacy
# motion-hysteresis engage path (which is exactly what operators
# without the manager get anyway).
try:
    import msgpack as _msgpack
    _HAS_MSGPACK = True
except ImportError:
    _msgpack = None  # type: ignore[assignment]
    _HAS_MSGPACK = False


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


def rebuild_msg_with_field_overrides(
    msg: bytes,
    topic: str,
    overrides: dict[str, np.ndarray],
) -> bytes | None:
    """Surgically overwrite one or more f32 fields in a packed pose frame.

    Preserves every other field (root_quat, hand joints, motion_token,
    frame_index, ...) because the v4 header layout encodes each field
    as a fixed-length blob at a known cursor position. Same shape +
    same dtype = same byte count, so we splice the new bytes into
    the existing payload in place without re-packing the header or
    rewriting field order.

    ``overrides`` maps field names to numpy arrays. Each array must
    match the field's declared dtype + shape in the header (we
    enforce f32 here -- all jpos/jvel/quat fields are f32 on the
    wire). Returns ``None`` if any override mismatches or the
    header is malformed; callers should fall back to forwarding
    the original ``msg`` verbatim in that case so a corrupt frame
    never silently steals the wire.

    Used by the LIVE -> OVERRIDE engagement ramp (2026-06-10
    follow-up 9): the proxy clamps the operator's first few
    override frames to a slow per-tick step relative to the LAST
    FORWARDED jpos so the deploy doesn't see a single-tick step
    from VLA-pose to operator-pose. **Follow-up 9b extends this
    to also flatten ``joint_pos_mj_future`` to the clamped current
    jpos and zero ``joint_vel_mj_future``** -- without that, the
    deploy's window-mode policy reads the operator's untouched
    future window (which still encodes "go all the way to
    operator-pose in 0.9 s") and slams the body to follow the
    future even though the current frame is properly rate-limited.
    """
    topic_bytes = topic.encode("utf-8")
    if not msg.startswith(topic_bytes):
        return None
    body_start = len(topic_bytes)
    body = msg[body_start:]
    if len(body) < HEADER_SIZE:
        return None
    header_blob = body[:HEADER_SIZE].rstrip(b"\x00")
    payload_start = body_start + HEADER_SIZE
    payload_len = len(msg) - payload_start
    try:
        header = json.loads(header_blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    fields = header.get("fields")
    if not isinstance(fields, list):
        return None
    # Collect (abs_start, abs_end, new_bytes) tuples in one pass so
    # we can splice them all in a single buffer build at the end.
    splices: list[tuple[int, int, bytes]] = []
    matched: set[str] = set()
    cursor = 0
    for f in fields:
        try:
            fname = str(f["name"])
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
        if cursor + nbytes > payload_len:
            return None
        if fname in overrides:
            if dtype != "f32":
                return None
            arr = np.asarray(overrides[fname], dtype=np.float32)
            if arr.shape != shape:
                return None
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            new_bytes = arr.tobytes()
            if len(new_bytes) != nbytes:
                return None
            abs_start = payload_start + cursor
            splices.append((abs_start, abs_start + nbytes, new_bytes))
            matched.add(fname)
        cursor += nbytes
    # Require all overrides matched at least one field; silently
    # ignoring an unmatched key would mask typos.
    if matched != set(overrides):
        return None
    if not splices:
        return msg
    splices.sort(key=lambda s: s[0])
    out = bytearray()
    cur = 0
    for start, end, blob in splices:
        out.extend(msg[cur:start])
        out.extend(blob)
        cur = end
    out.extend(msg[cur:])
    return bytes(out)


def rebuild_msg_with_jpos_override(
    msg: bytes, topic: str, new_jpos: np.ndarray
) -> bytes | None:
    """Back-compat single-field wrapper around the multi-field helper.

    Kept so the regression pin (``test_rebuild_msg_with_jpos_override_
    preserves_other_fields``) and any future callers that only need
    to clamp the current jpos keep working. Internally delegates to
    ``rebuild_msg_with_field_overrides`` so both code paths share the
    same byte-splice logic.
    """
    return rebuild_msg_with_field_overrides(
        msg, topic, {"joint_pos_mj": new_jpos}
    )


def _clamp_vector_step_f32(
    target: np.ndarray,
    prev: np.ndarray | None,
    max_step: float,
) -> np.ndarray:
    """Per-element ``|target - prev| <= max_step`` rate clamp.

    Mirrors ``live_vla_publish_motion_token._clamp_vector_step``
    semantics: caps each element's step from ``prev`` to
    ``max_step`` while preserving the direction vector (so a large
    multi-joint step shrinks proportionally rather than slicing
    individual joints). Returns ``target`` unchanged when
    ``max_step <= 0`` or ``prev is None`` (cold-start tick).
    """
    tgt = np.asarray(target, dtype=np.float32)
    if max_step <= 0.0 or prev is None:
        return tgt.copy()
    prv = np.asarray(prev, dtype=np.float32)
    delta = tgt - prv
    peak = float(np.abs(delta).max())
    if peak <= max_step:
        return tgt.copy()
    return (prv + delta * (max_step / peak)).astype(np.float32)


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
    return _decode_pose_field_f32(
        msg, topic, name="joint_pos_mj", expected_shape=(NUM_BODY_DOFS,)
    )


def decode_pose_left_hand(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``left_hand_joints`` (f32, shape (DEFAULT_HAND_DOF,))."""
    return _decode_pose_field_f32(
        msg, topic, name="left_hand_joints",
        expected_shape=(DEFAULT_HAND_DOF,),
    )


def decode_pose_right_hand(
    msg: bytes, topic: str = "pose"
) -> np.ndarray | None:
    """Extract ``right_hand_joints`` (f32, shape (DEFAULT_HAND_DOF,))."""
    return _decode_pose_field_f32(
        msg, topic, name="right_hand_joints",
        expected_shape=(DEFAULT_HAND_DOF,),
    )


def _decode_pose_field_f32(
    msg: bytes,
    topic: str,
    *,
    name: str,
    expected_shape: tuple[int, ...],
) -> np.ndarray | None:
    """Generic packed-pose field extractor for f32 vectors.

    Walks the v4 header field list cursor-by-cursor (same scheme as
    the deploy's ZmqPoseInputSource) and returns the named field as
    a copied numpy array when the dtype + shape match, else ``None``.
    Tolerates header decode errors so the publish thread can survive
    a malformed cached frame.
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
            fname = str(f["name"])
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
        if fname == name:
            if dtype != "f32" or shape != expected_shape:
                return None
            return np.frombuffer(
                payload[cursor:cursor + nbytes], dtype="<f4"
            ).copy()
        cursor += nbytes
    return None  # named field absent (e.g. token-only side-channel frame)


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
# Operator override (e.g. VR teleop nudging the arm out of a stuck VLA
# pose). When --override-port is enabled and the override SUB has a fresh
# frame, the proxy forwards override bytes verbatim instead of primary.
# On the OVERRIDE -> LIVE edge a one-shot "override_released" event is
# emitted on the optional control PUB so the VLA bridge can cold-restart
# its ramp-in from the current measured pose. See 2026-06-10 manual-
# takeover milestone for the design.
STATE_OVERRIDE: str = "OVERRIDE"

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

    # ----- Operator override SUB (Phase 1 of manual-takeover) ----------
    # When --override-port is set, the proxy subscribes to a SECOND
    # upstream (typically the teleop wire publisher on a different port)
    # and prefers it over --upstream-port whenever the override SUB has
    # a fresh frame within --override-stale-ms. Disabled by default so
    # existing single-source deployments are byte-for-byte unchanged.
    p.add_argument(
        "--override-host",
        default=None,
        help="Host for the operator-override pose PUB (e.g. the laptop "
             "running the teleop recorder). Defaults to --upstream-host.",
    )
    p.add_argument(
        "--override-port",
        type=int,
        default=-1,
        help="Operator-override pose PUB port. Set to a positive int "
             "(e.g. 5558) to enable manual-takeover arbitration. "
             "Default -1 = DISABLED (legacy single-source behaviour).",
    )
    p.add_argument(
        "--override-topic",
        default=None,
        help="Topic prefix on the override PUB. Defaults to "
             "--upstream-topic.",
    )
    p.add_argument(
        "--override-stale-ms",
        type=int,
        default=200,
        help="Treat the override source as gone after this many ms of "
             "silence (default 200 = 10 ticks @ 50Hz). Acts as a "
             "debounce so a single dropped teleop frame doesn't flip "
             "the proxy back to primary and trigger a spurious VLA "
             "cold-restart.",
    )
    # ---- Frozen-frame release (2026-06-10 follow-up) ------------------
    # The quest3_manager publishes the FROZEN last commanded pose every
    # tick when teleop mode is OFF or LOCOMOTION (see manager lines
    # 1221-1229). That means the recorder's PUB on the override port
    # NEVER goes silent across an A+B+X+Y "drop teleop" gesture, so the
    # silence-based release above only fires when the operator Ctrl-C's
    # the entire stack. Frame-equality detection catches this: when N
    # consecutive override frames have ‖Δjpos‖₂ <= tol, treat the
    # override as released even though bytes are still arriving on the
    # wire. Reset on any frame above tol so re-engagement is automatic
    # when the operator starts driving again. Set ticks to 0 to disable
    # and fall back to the silence-only behaviour.
    p.add_argument(
        "--override-frozen-ticks",
        type=int,
        default=10,
        help="Fire override_released after this many consecutive "
             "override frames within --override-frozen-l2-tol of the "
             "previous one (default 10 = 200ms @ 50Hz, matching "
             "--override-stale-ms semantics). The Quest3 manager "
             "publishes a frozen pose every tick in OFF/LOCOMOTION "
             "mode, so this is what actually fires release after the "
             "operator hits the A+B+X+Y disengage chord. Set to 0 to "
             "disable and rely on silence-based release only.",
    )
    p.add_argument(
        "--override-frozen-l2-tol",
        type=float,
        default=5e-3,
        help="L2 distance tolerance (rad) for two override frames to "
             "be considered 'frozen' (default 5e-3 ~ 0.3 deg of total "
             "joint-space motion). Small enough that any intentional "
             "teleop push trips above tol within one tick; large "
             "enough to absorb (a) controller-rest jitter when the "
             "operator is holding the Quest 3 still, (b) planner-side "
             "body_pose float noise that may leak through the "
             "recorder's merge, and (c) IK retargeting flicker when "
             "the operator brushes a thumbstick. Bumped from the "
             "original 1e-4 on 2026-06-10 after observing repeated "
             "single-frame engage/release cycles in sim from sub-deg "
             "controller drift while the manager was in OFF; the new "
             "default still catches the bytes-identical frozen pose "
             "the manager publishes in OFF/LOCOMOTION. Lower (e.g. "
             "1e-4) for strict bytes-match detection only.",
    )
    p.add_argument(
        "--override-engage-motion-ticks",
        type=int,
        default=10,
        help="Require this many consecutive override frames with "
             "L2 delta ABOVE --override-frozen-l2-tol before firing "
             "override_engaged (default 10 = 200ms @ 50Hz, symmetric "
             "with --override-frozen-ticks). Prevents brief jitter "
             "from spurious engage/release cycles (each cycle "
             "triggers a heavy VLA cold-restart). Set to 0 for the "
             "legacy single-frame-engage behaviour used by older "
             "smoke tests; the launcher defaults to 10 in real and "
             "sim runs.",
    )
    p.add_argument(
        "--engagement-max-wire-step",
        type=float,
        default=0.012,
        help="Per-element max joint-position step (rad) applied to "
             "the override frames forwarded right AFTER the LIVE -> "
             "OVERRIDE transition. Default 0.012 rad/tick (~36 deg/s "
             "per joint at 50 Hz). The proxy snapshots the last "
             "forwarded VLA pose at engagement, then clamps each "
             "subsequent operator frame's joint_pos_mj per-element "
             "relative to the previously forwarded pose, linearly "
             "relaxing the clamp back to --engagement-steady-wire-"
             "step over --engagement-step-ramp-ticks ticks. Without "
             "this, the operator's first OVERRIDE frame can step "
             "the wire ~3 rad away from VLA's last command in one "
             "tick and the deploy slams the body across the delta "
             "(2026-06-10 follow-up 9). Set to 0 (or equal to "
             "--engagement-steady-wire-step) to disable the slow "
             "engagement ramp.",
    )
    p.add_argument(
        "--engagement-steady-wire-step",
        type=float,
        default=0.035,
        help="Per-element steady-state max joint-position step (rad) "
             "the engagement ramp converges to. Default 0.035 rad/tick "
             "(~100 deg/s per joint at 50 Hz, matches the bridge's "
             "--vla-max-wire-step default). After the engagement ramp "
             "completes the proxy stops clamping override frames "
             "entirely -- the operator's controller motion is the "
             "rate limit. The clamp is ONLY active for the first "
             "--engagement-step-ramp-ticks ticks of an OVERRIDE "
             "window; subsequent ticks forward verbatim.",
    )
    p.add_argument(
        "--engagement-step-ramp-ticks",
        type=int,
        default=250,
        help="Number of ticks over which to linearly ramp the "
             "engagement rate clamp from --engagement-max-wire-step "
             "(slow, applied at engagement) to --engagement-steady-"
             "wire-step (normal). Default 250 @ 50Hz = 5.0 s -- "
             "matches the bridge's --vla-handoff-step-ramp-ticks "
             "default for symmetry. Set to 0 to disable engagement "
             "clamping (operator frames forwarded verbatim from the "
             "very first OVERRIDE tick -- pre-2026-06-10 behaviour, "
             "produces the slam this guard exists to prevent).",
    )

    # ----- vla_control PUB (Phase 1 of manual-takeover) ----------------
    # Edge-triggered control plane for the VLA bridge. Emits
    # ``override_engaged`` on PRIMARY/IDLE -> OVERRIDE and
    # ``override_released`` on OVERRIDE -> PRIMARY/IDLE. The VLA bridge
    # subscribes and uses the released edge to cold-restart its ramp-in
    # from the current measured pose (clearing stale action chunks).
    # Disabled by default; only meaningful when --override-port is set.
    p.add_argument(
        "--vla-control-bind-host",
        default="127.0.0.1",
        help="Bind interface for the vla_control PUB (default "
             "127.0.0.1; bridge is colocated on the laptop).",
    )
    p.add_argument(
        "--vla-control-port",
        type=int,
        default=-1,
        help="Bind port for the vla_control edge-event PUB. Set to a "
             "positive int (e.g. 5559) to enable. Default -1 = "
             "DISABLED.",
    )
    p.add_argument(
        "--vla-control-topic",
        default="vla_control",
        help="Topic prefix on the vla_control PUB (default "
             "'vla_control').",
    )

    # ----- Operator mode SUB (stream_mode, 2026-06-10 follow-up) -----
    # When --teleop-mode-port > 0, subscribe to the quest3_manager's
    # ``stream_mode`` PUB (port 5564 by default, msgpack payload with
    # ``mode``: "OFF" | "LOCOMOTION" | "ARM_MANIPULATION"). This is the
    # deterministic A+B+X+Y signal driven by the operator's actual
    # button presses, so the proxy can gate engagement DIRECTLY on
    # ``mode != "OFF"`` instead of guessing via per-tick pose deltas
    # on the override SUB. The latter is intrinsically fragile because
    # the manager keeps publishing FROZEN arm/hand vectors in OFF and
    # LOCOMOTION (see manager lines 1221-1229) so the recorder never
    # goes silent across an A+B+X+Y disengage, and the operator holding
    # the controller still in ARM_MANIPULATION looks identical to
    # "operator dropped to OFF" from a pose-delta perspective.
    #
    # When --teleop-mode-port <= 0 (default), the proxy falls back to
    # the legacy motion-hysteresis path -- which still works for older
    # deployments that don't run the manager (e.g. dataset replay) but
    # WILL flicker when the operator holds still. The legacy path is
    # the only one available pre-2026-06-10.
    p.add_argument(
        "--teleop-mode-host",
        default="127.0.0.1",
        help="Host where the manager's stream_mode PUB lives "
             "(default 127.0.0.1; on PC2 + laptop split set this to "
             "the laptop's address).",
    )
    p.add_argument(
        "--teleop-mode-port",
        type=int,
        default=-1,
        help="Port of the manager's stream_mode PUB. Default is the "
             "manager's --recorder-pub-port (typically 5564). Set to "
             "a positive int to enable mode-gated engagement; set to "
             "-1 (default) to fall back to motion-hysteresis. STRICT "
             "MODE: when enabled and the mode signal goes stale "
             "(--teleop-mode-stale-ms), engagement is BLOCKED -- the "
             "operator's poses are ignored until the manager is back "
             "or --teleop-mode-port is removed.",
    )
    p.add_argument(
        "--teleop-mode-topic",
        default="stream_mode",
        help="Topic prefix on the manager's PUB (default "
             "'stream_mode'; matches the manager's "
             "--stream-mode-topic).",
    )
    p.add_argument(
        "--teleop-mode-stale-ms",
        type=int,
        default=1000,
        help="Treat the mode signal as gone after this many ms of "
             "silence (default 1000 = 50 ticks @ 50Hz, comfortably "
             "longer than any manager scheduler hiccup but short "
             "enough that a dead manager fails closed within a "
             "second). When stale, engagement is BLOCKED in strict "
             "mode (see --teleop-mode-port).",
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

    # ----- Manual-takeover dual-source: optional override SUB ----------
    # If --override-port > 0, subscribe to a second pose stream and
    # prefer its frames over --upstream-port whenever fresh. The override
    # bytes are forwarded verbatim to downstream (same wire format as
    # primary), so no decoding overhead on the hot path.
    override_sub: zmq.Socket | None = None
    override_enabled = int(args.override_port) > 0
    override_host = args.override_host or args.upstream_host
    override_topic = args.override_topic or args.upstream_topic
    override_stale_s = max(args.override_stale_ms, 1) / 1000.0
    if override_enabled:
        override_sub = ctx.socket(zmq.SUB)
        override_sub.setsockopt(zmq.RCVHWM, 100)
        override_url = f"tcp://{override_host}:{args.override_port}"
        override_sub.connect(override_url)
        override_sub.setsockopt(
            zmq.SUBSCRIBE, override_topic.encode("utf-8")
        )

    # ----- Operator mode SUB (stream_mode, 2026-06-10 follow-up) ------
    # Connect to the manager's recorder PUB (port 5564 by default) and
    # subscribe to ``stream_mode``. The payload is a msgpack dict
    # ``{ "mode": "OFF" | "LOCOMOTION" | "ARM_MANIPULATION", "tick":
    # ..., "ts": ... }`` published every manager tick. When the
    # incoming mode is non-OFF, the proxy treats the operator as
    # actively driving and forwards override frames verbatim; when
    # mode flips to OFF the proxy releases override on the EDGE and
    # the bridge cold-restarts. The mode-gated path is STRICT: when
    # --teleop-mode-port is set and the signal goes stale, engagement
    # is BLOCKED -- a dead manager fails closed within
    # --teleop-mode-stale-ms (default 1 s).
    teleop_mode_sub: zmq.Socket | None = None
    teleop_mode_enabled = int(args.teleop_mode_port) > 0 and _HAS_MSGPACK
    teleop_mode_stale_s = max(args.teleop_mode_stale_ms, 1) / 1000.0
    if int(args.teleop_mode_port) > 0 and not _HAS_MSGPACK:
        print(
            "[pose_proxy] WARN: --teleop-mode-port is set but msgpack "
            "is not installed in this venv; falling back to legacy "
            "motion-hysteresis engagement (pip install msgpack to "
            "enable strict mode-gated engagement).",
            flush=True,
        )
    if teleop_mode_enabled:
        teleop_mode_sub = ctx.socket(zmq.SUB)
        teleop_mode_sub.setsockopt(zmq.RCVHWM, 32)
        teleop_mode_url = (
            f"tcp://{args.teleop_mode_host}:{args.teleop_mode_port}"
        )
        teleop_mode_sub.connect(teleop_mode_url)
        teleop_mode_sub.setsockopt(
            zmq.SUBSCRIBE, args.teleop_mode_topic.encode("utf-8")
        )

    # ----- Manual-takeover edge-event PUB (vla_control) ----------------
    # If --vla-control-port > 0, bind a PUB socket and emit one-shot
    # JSON events on the OVERRIDE engage/release edges. The bridge SUBs
    # this and cold-restarts on the released edge.
    vla_control_pub: zmq.Socket | None = None
    vla_control_enabled = int(args.vla_control_port) > 0
    if vla_control_enabled:
        vla_control_pub = ctx.socket(zmq.PUB)
        vla_control_pub.setsockopt(zmq.SNDHWM, 32)
        vla_control_url = (
            f"tcp://{args.vla_control_bind_host}:{args.vla_control_port}"
        )
        vla_control_pub.bind(vla_control_url)

    print(
        f"[pose_proxy] upstream SUB:   {upstream_url} topic={args.upstream_topic!r}",
        flush=True,
    )
    if override_enabled:
        print(
            f"[pose_proxy] override SUB:   "
            f"tcp://{override_host}:{args.override_port} "
            f"topic={override_topic!r} "
            f"(stale_ms={args.override_stale_ms})",
            flush=True,
        )
    else:
        print(
            "[pose_proxy] override SUB:   DISABLED "
            "(--override-port not set; legacy single-source mode)",
            flush=True,
        )
    if teleop_mode_enabled:
        print(
            f"[pose_proxy] teleop_mode SUB: "
            f"tcp://{args.teleop_mode_host}:{args.teleop_mode_port} "
            f"topic={args.teleop_mode_topic!r} "
            f"(stale_ms={args.teleop_mode_stale_ms}) "
            f"-- STRICT mode-gated engage (motion-hysteresis bypassed)",
            flush=True,
        )
    elif override_enabled:
        print(
            "[pose_proxy] teleop_mode SUB: DISABLED "
            "(--teleop-mode-port not set; falling back to motion-"
            "hysteresis engage path which will flicker if operator "
            "holds the controller still in ARM_MANIPULATION)",
            flush=True,
        )
    if vla_control_enabled:
        print(
            f"[pose_proxy] vla_control PUB: "
            f"tcp://{args.vla_control_bind_host}:{args.vla_control_port} "
            f"topic={args.vla_control_topic!r}",
            flush=True,
        )
    elif override_enabled:
        print(
            "[pose_proxy] vla_control PUB: DISABLED "
            "(--vla-control-port not set; bridge won't cold-restart "
            "automatically on override release)",
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
    # Manual-takeover override tracking. ``last_override_s = -1`` means
    # "never seen an override frame"; the arbitration logic treats this
    # as "override is silent / not engaged" so the proxy behaves like
    # the legacy single-source proxy until the first override frame
    # arrives. Edge tracking happens via ``override_active`` flips.
    last_override_s = -1.0
    override_frames = 0
    override_engage_events = 0
    override_release_events = 0
    override_active = False
    # ----- Frozen-frame release state -------------------------------
    # ``prev_override_jpos`` is the most recently decoded override
    # joint_pos_mj; ``override_frozen_count`` is the running streak of
    # consecutive override frames within --override-frozen-l2-tol of
    # ``prev_override_jpos``. When the streak hits
    # --override-frozen-ticks, ``override_frozen_detected`` latches
    # True and forces ``override_fresh = False`` on the next loop --
    # which routes through the SAME edge handler that emits
    # override_released, so downstream consumers (bridge cold-restart)
    # see no behavioural difference vs the silence-based release.
    # The latch clears the moment we see motion above tol so the
    # operator's next ARM_MANIPULATION push re-engages without any
    # extra UX. release_count tracks how many releases were fired by
    # the frozen detector vs the silence detector (status line only).
    prev_override_jpos: np.ndarray | None = None
    override_frozen_count = 0
    override_frozen_detected = False
    override_frozen_release_events = 0
    frozen_ticks_threshold = max(int(args.override_frozen_ticks), 0)
    frozen_l2_tol = max(float(args.override_frozen_l2_tol), 0.0)
    # ----- Engagement ramp state (2026-06-10 follow-up 9) ----------
    # Symmetric to the bridge's post-handoff slow-step ramp: when
    # the proxy fires the LIVE -> OVERRIDE edge, the operator's
    # commanded body pose can be ~3 rad (L_inf) from VLA's last
    # decoded pose. Without this clamp the proxy forwarded
    # ``latest_override`` VERBATIM on the first OVERRIDE tick and
    # the deploy saw a single-tick joint-space step that visibly
    # slammed the body across the delta (the user at 13:29
    # confirmed this on the real robot).
    #
    # ``engagement_clamp_remaining`` is the countdown of remaining
    # ticks in the ramp window. ``engagement_last_forwarded_jpos``
    # is the running anchor for the per-element step clamp; it
    # starts at the last VLA pose forwarded before the engagement
    # edge, then updates to the just-forwarded (clamped) operator
    # pose every tick. Linear interpolation of ``effective_max_step``
    # from ``engagement_max_wire_step`` (slow) -> ``engagement_
    # steady_wire_step`` (normal) across ``engagement_step_ramp_
    # ticks`` ticks; when the countdown hits 0 the clamp deactivates
    # and override frames pass through verbatim (operator's
    # controller motion is the rate limit at that point).
    engagement_max_wire_step = max(float(args.engagement_max_wire_step), 0.0)
    engagement_steady_wire_step = max(float(args.engagement_steady_wire_step), 0.0)
    engagement_step_ramp_ticks = max(int(args.engagement_step_ramp_ticks), 0)
    engagement_clamp_remaining = 0
    engagement_last_forwarded_jpos: np.ndarray | None = None
    # ----- Engage hysteresis (2026-06-10 follow-up) ------------------
    # Symmetric to the frozen-release logic: count consecutive override
    # frames whose joint-space delta is ABOVE frozen_l2_tol and require
    # the streak to cross engage_motion_threshold before tripping
    # override_engaged. When threshold == 0, engage on the first
    # non-frozen frame (legacy behaviour, kept for older smoke tests).
    # The motion + frozen counters are stateful mirrors: any single
    # frame increments exactly one of them and zeros the other, so a
    # mid-engage pause doesn't immediately release nor does a
    # mid-release flicker immediately re-engage.
    override_motion_count = 0
    engage_motion_threshold = max(
        int(getattr(args, "override_engage_motion_ticks", 0)), 0
    )
    # ----- Operator-pose handoff (2026-06-10 follow-up) --------------
    # Snapshot the operator's last commanded body + hand joints from
    # every override frame so the override_released event can carry
    # them downstream. The VLA bridge uses these to hold the wire at
    # the operator's exact pose during its cold-restart bridging
    # window instead of stepping back to x2_debug's measured pose
    # (which lags due to motor / contact / gravity sag and produces
    # the visible "pose reset" the operator sees when handing off
    # from ARM_MANIPULATION to LOCOMOTION). All three are reset only
    # when the engage_motion / frozen streaks are reset, so a stale
    # snapshot from a previous session is never sent.
    last_override_left_hand: np.ndarray | None = None
    last_override_right_hand: np.ndarray | None = None
    # ----- Operator mode (stream_mode, 2026-06-10 follow-up) ---------
    # ``current_teleop_mode`` mirrors the manager's last published
    # ``mode`` field ("OFF" | "LOCOMOTION" | "ARM_MANIPULATION");
    # ``last_teleop_mode_s`` is the monotonic clock at which we
    # received it. When teleop_mode_enabled is True, these drive the
    # engage gate directly -- motion-hysteresis and frozen-detection
    # are BYPASSED (the operator's button press is the truth, not
    # pose deltas). ``teleop_mode_msgs`` counts received messages,
    # ``teleop_mode_decode_failures`` counts msgpack / topic decode
    # failures for the status line.
    current_teleop_mode: str | None = None
    last_teleop_mode_s = -1.0
    teleop_mode_msgs = 0
    teleop_mode_decode_failures = 0
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

            # ----- Drain operator mode SUB (stream_mode) ---------------
            # Pull the latest mode message from the manager's recorder
            # PUB so the engage gate below has fresh truth. We only
            # need the LATEST mode -- intermediate frames are dropped
            # (mode doesn't accumulate). Decode failures bump a
            # counter but don't crash; the gate falls into the
            # "stale" branch and blocks engagement until the next
            # good frame.
            if teleop_mode_sub is not None and _msgpack is not None:
                latest_mode_msg: list[bytes] | None = None
                while True:
                    try:
                        latest_mode_msg = teleop_mode_sub.recv_multipart(
                            zmq.NOBLOCK
                        )
                    except zmq.Again:
                        break
                if latest_mode_msg is not None:
                    # Manager publishes [topic_bytes, msgpack_payload].
                    # Tolerate a single-part frame too in case some
                    # future publisher omits the topic prefix.
                    payload_bytes: bytes | None = None
                    if len(latest_mode_msg) >= 2:
                        payload_bytes = latest_mode_msg[1]
                    elif len(latest_mode_msg) == 1:
                        payload_bytes = latest_mode_msg[0]
                    decoded_mode: str | None = None
                    if payload_bytes is not None:
                        try:
                            payload = _msgpack.unpackb(
                                payload_bytes, raw=False
                            )
                            if isinstance(payload, dict):
                                m = payload.get("mode")
                                if isinstance(m, str):
                                    decoded_mode = m
                        except Exception:
                            decoded_mode = None
                    if decoded_mode is not None:
                        current_teleop_mode = decoded_mode
                        last_teleop_mode_s = now
                        teleop_mode_msgs += 1
                    else:
                        teleop_mode_decode_failures += 1

            # ----- Drain operator override SUB (manual takeover) ------
            # Drain override BEFORE primary so the freshest override
            # frame wins on ties. Forward override bytes verbatim --
            # the wire format is identical to primary, so the deploy
            # decodes the operator's pose with zero extra work.
            latest_override = None
            if override_sub is not None:
                while True:
                    try:
                        latest_override = override_sub.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break

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

            # Always cache the primary frame even when override is the
            # forwarded source -- the HOLD fallback ladder still needs
            # a fresh primary cache for the moment we hand back to VLA
            # and the bridge's cold-restart hold-frame hasn't landed yet.
            if latest is not None:
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
                last_upstream_s = now

            # Override-active iff we received an override frame this
            # tick OR within the debounce window since the last one. The
            # debounce prevents flipping back to primary on a single
            # dropped 20ms tick at 50Hz (would trigger spurious VLA
            # cold-restarts mid-takeover).
            if latest_override is not None:
                last_override_s = now

            # ----- Frozen-frame release detection ----------------------
            # The quest3_manager publishes the frozen last commanded
            # pose every tick in OFF/LOCOMOTION (see manager lines
            # 1221-1229), so the override SUB never goes silent across
            # an A+B+X+Y "drop teleop" gesture. Detect the freeze by
            # tracking consecutive override frames whose joint_pos_mj
            # delta from the previous frame is within tolerance. When
            # the streak crosses the threshold, latch
            # override_frozen_detected so the existing edge handler
            # fires override_released exactly once (NOT every tick),
            # giving downstream consumers the same one-shot signal
            # they'd see on silence-based release. Disabled when
            # frozen_ticks_threshold == 0.
            if (
                latest_override is not None
                and override_sub is not None
            ):
                ojpos_new = decode_pose_joint_pos_mj(
                    latest_override, args.upstream_topic
                )
                if ojpos_new is None:
                    ojpos_new = decode_pose_joint_pos_mj(
                        latest_override, args.downstream_topic
                    )
                # Cache operator-commanded hand joints alongside body.
                # Both fields are optional in the wire format -- legacy
                # token-only frames omit them -- so we keep whichever
                # we got and leave the rest at the last seen value
                # (initially None). The release-event packer falls
                # back to omission when None.
                olh = decode_pose_left_hand(
                    latest_override, args.upstream_topic
                )
                if olh is None:
                    olh = decode_pose_left_hand(
                        latest_override, args.downstream_topic
                    )
                if olh is not None:
                    last_override_left_hand = olh
                orh = decode_pose_right_hand(
                    latest_override, args.upstream_topic
                )
                if orh is None:
                    orh = decode_pose_right_hand(
                        latest_override, args.downstream_topic
                    )
                if orh is not None:
                    last_override_right_hand = orh
                if ojpos_new is not None:
                    delta = 0.0
                    if prev_override_jpos is None:
                        # No baseline yet -- neither streak applies.
                        override_frozen_count = 0
                        override_motion_count = 0
                    else:
                        # L2 in joint space. Cheap (31 DoF body
                        # vector; hands ship as separate fields) and
                        # matches the manager's frozen-pose semantics
                        # exactly: numpy .copy() of the last commanded
                        # body vector means bytes are identical when
                        # frozen. The tolerance (default 5e-3 rad)
                        # absorbs sub-degree controller-rest jitter
                        # AND IK retargeting flicker; intentional
                        # teleop pushes trip well above on the first
                        # tick.
                        delta = float(np.linalg.norm(
                            ojpos_new.astype(np.float64)
                            - prev_override_jpos.astype(np.float64)
                        ))
                        if delta <= frozen_l2_tol:
                            override_frozen_count += 1
                            override_motion_count = 0
                        else:
                            # Operator is moving again. Clear the
                            # frozen latch so future frozen streaks
                            # can re-fire release; bump the motion
                            # streak so the engage-hysteresis can
                            # eventually trip.
                            override_frozen_count = 0
                            override_frozen_detected = False
                            override_motion_count += 1
                    prev_override_jpos = ojpos_new

                    if (
                        frozen_ticks_threshold > 0
                        and not override_frozen_detected
                        and override_frozen_count >= frozen_ticks_threshold
                    ):
                        override_frozen_detected = True
                        override_frozen_release_events += 1
                        print(
                            f"[pose_proxy] override frozen detected "
                            f"(streak={override_frozen_count} ticks "
                            f">= threshold={frozen_ticks_threshold}, "
                            f"L2={delta:.6f} <= tol={frozen_l2_tol:g}); "
                            f"forcing release without waiting for SUB "
                            f"silence",
                            flush=True,
                        )

            # Engage gate. Two paths:
            #
            #   STRICT (teleop_mode_enabled): the manager's stream_mode
            #     topic is the source of truth -- engage iff the mode
            #     signal is FRESH and the reported mode is NOT "OFF"
            #     (i.e. operator has pressed A+B+X+Y into LOCOMOTION
            #     or ARM_MANIPULATION). When the signal goes stale
            #     (--teleop-mode-stale-ms) engagement BLOCKS; a dead
            #     manager fails closed within ~1 s. Motion-hysteresis
            #     and frozen-detection are BYPASSED here because the
            #     operator holding still in ARM_MANIPULATION looks
            #     identical to "operator dropped to OFF" through the
            #     pose-delta lens (the bug the user just hit).
            #
            #   LEGACY (--teleop-mode-port not set, or msgpack
            #     missing): silence-debounced AND not frozen-latched
            #     AND (legacy: immediate, OR new: sustained motion).
            #     This is the pre-2026-06-10 behaviour and WILL flicker
            #     when the operator holds the controller still; keep
            #     it so older deployments / replay tests still work.
            teleop_mode_fresh = (
                last_teleop_mode_s >= 0
                and (now - last_teleop_mode_s) <= teleop_mode_stale_s
            )
            if teleop_mode_enabled:
                teleop_engaged = (
                    teleop_mode_fresh
                    and current_teleop_mode is not None
                    and current_teleop_mode != "OFF"
                )
                override_fresh = (
                    override_sub is not None
                    and last_override_s >= 0
                    and (now - last_override_s) <= override_stale_s
                    and teleop_engaged
                )
                # Status-line helpers don't need motion/frozen booleans
                # here -- they'd be misleading in strict mode. Keep
                # the counters running anyway (they're updated by the
                # frozen/motion detector blocks above) so an operator
                # debugging a hung mode SUB can still see whether the
                # override stream has motion under the hood.
                override_motion_sustained = teleop_engaged
            else:
                override_motion_sustained = (
                    engage_motion_threshold == 0
                    or override_motion_count >= engage_motion_threshold
                )
                override_fresh = (
                    override_sub is not None
                    and last_override_s >= 0
                    and (now - last_override_s) <= override_stale_s
                    and not override_frozen_detected
                    and override_motion_sustained
                )

            # ----- Edge: PRIMARY/IDLE -> OVERRIDE ---------------------
            # Fire the "engaged" event ASAP so the VLA bridge can stop
            # publishing chunks (avoids stale-chunk wire fights with the
            # operator's pose during the takeover window).
            if override_fresh and not override_active:
                override_active = True
                override_engage_events += 1
                # 2026-06-10 follow-up 9: arm the engagement slow-step
                # ramp. Anchor the clamp at the last successfully
                # forwarded pose (which under LIVE is the most
                # recent VLA frame and under HOLD/BLEND/IDLE is the
                # last cached or idle-clip pose). When ramp_ticks
                # is 0 we skip the arm entirely (legacy verbatim-
                # forward behaviour for smoke tests).
                if (
                    engagement_step_ramp_ticks > 0
                    and engagement_max_wire_step > 0.0
                ):
                    engagement_clamp_remaining = engagement_step_ramp_ticks
                    if last_upstream_jpos is not None:
                        engagement_last_forwarded_jpos = (
                            np.asarray(last_upstream_jpos, dtype=np.float32)
                            .copy()
                        )
                    else:
                        engagement_last_forwarded_jpos = None
                    print(
                        f"[pose_proxy] engagement slow-step ramp armed "
                        f"(window={engagement_step_ramp_ticks} ticks; "
                        f"max_step {engagement_max_wire_step:.3f} -> "
                        f"{engagement_steady_wire_step:.3f} rad/tick; "
                        f"anchor="
                        f"{'last_VLA_pose' if engagement_last_forwarded_jpos is not None else 'NONE (will forward operator frame 0 verbatim)'}"
                        f")",
                        flush=True,
                    )
                if vla_control_pub is not None:
                    try:
                        evt = json.dumps({
                            "event": "override_engaged",
                            "ts": now,
                            "tick": tick,
                        }).encode("utf-8")
                        vla_control_pub.send_multipart(
                            [args.vla_control_topic.encode("utf-8"),
                             evt],
                            zmq.NOBLOCK,
                        )
                    except zmq.Again:
                        pass

            # ----- Edge: OVERRIDE -> PRIMARY/IDLE ---------------------
            # Fire the "released" event the moment the override SUB
            # falls past its stale window, regardless of whether
            # primary is fresh or both are silent. The VLA bridge uses
            # this to cold-restart its ramp-in from the current
            # measured pose (clearing stale action chunks).
            if (not override_fresh) and override_active:
                override_active = False
                override_release_events += 1
                # 2026-06-10 follow-up 9: tear down the engagement
                # ramp state on release so the next engage edge
                # re-arms cleanly from a fresh VLA anchor. Without
                # this, a rapid release+re-engage within the ramp
                # window would inherit the previous anchor (= stale
                # operator pose) and skip the slow-step bridge from
                # VLA's new pose.
                engagement_clamp_remaining = 0
                engagement_last_forwarded_jpos = None
                if vla_control_pub is not None:
                    try:
                        # Pack the operator's last commanded pose
                        # (body + hands) into the event so the bridge
                        # can hold the wire at THIS exact pose during
                        # its cold-restart bridging window instead of
                        # snapping to x2_debug's measured pose, which
                        # lags by motor / contact / gravity sag and
                        # produces the observable "pose reset" on
                        # ARM_MANIPULATION -> LOCOMOTION handoff.
                        # Each field is optional -- when the source
                        # frame omits it (or proxy decode failed),
                        # the key is left out and the bridge falls
                        # back to its legacy measured-pose hold.
                        release_pose: dict[str, list[float]] = {}
                        if prev_override_jpos is not None:
                            release_pose["joint_pos_mj"] = (
                                prev_override_jpos.astype(float).tolist()
                            )
                        if last_override_left_hand is not None:
                            release_pose["left_hand_joints"] = (
                                last_override_left_hand.astype(float)
                                .tolist()
                            )
                        if last_override_right_hand is not None:
                            release_pose["right_hand_joints"] = (
                                last_override_right_hand.astype(float)
                                .tolist()
                            )
                        evt_payload: dict[str, object] = {
                            "event": "override_released",
                            "ts": now,
                            "tick": tick,
                        }
                        if release_pose:
                            evt_payload["release_pose"] = release_pose
                        evt = json.dumps(evt_payload).encode("utf-8")
                        vla_control_pub.send_multipart(
                            [args.vla_control_topic.encode("utf-8"),
                             evt],
                            zmq.NOBLOCK,
                        )
                    except zmq.Again:
                        pass

            if override_fresh:
                # Override owns the wire this tick. Forward the freshest
                # override frame (if we got one); else republish the
                # last one cached on the override path so the deploy
                # keeps tracking the operator's hold pose between SUB
                # ticks. Also overwrite the primary cache so a
                # subsequent HOLD fallback replays the operator's pose
                # (not the stale pre-override VLA frame).
                if latest_override is not None:
                    # 2026-06-10 follow-up 9: clamp the operator's
                    # joint_pos_mj per-element relative to the last
                    # forwarded pose during the engagement ramp.
                    # Without the clamp the first OVERRIDE frame
                    # steps the wire across the full VLA -> operator
                    # delta in one tick (~3 rad L_inf in the 13:05
                    # run) and the deploy slams the body.
                    fwd_msg = latest_override
                    op_jpos = decode_pose_joint_pos_mj(
                        latest_override, args.upstream_topic
                    )
                    if op_jpos is None:
                        op_jpos = decode_pose_joint_pos_mj(
                            latest_override, args.downstream_topic
                        )
                    if (
                        engagement_clamp_remaining > 0
                        and op_jpos is not None
                        and engagement_last_forwarded_jpos is not None
                    ):
                        # Linear interpolation of the per-element
                        # step clamp from slow -> steady across the
                        # ramp window. Mirrors the bridge's
                        # follow-up 6 formula so the two sides of
                        # the takeover handshake have symmetric
                        # rate-limit behaviour.
                        ramp_progress = 1.0 - float(
                            engagement_clamp_remaining
                        ) / float(max(engagement_step_ramp_ticks, 1))
                        ramp_progress = min(max(ramp_progress, 0.0), 1.0)
                        effective_max_step = (
                            (1.0 - ramp_progress) * engagement_max_wire_step
                            + ramp_progress * engagement_steady_wire_step
                        )
                        # The original engagement edge log already
                        # documented anchor presence; for per-tick
                        # diagnostics rely on the operator's status
                        # line + the bridge's body_Δ telemetry.
                        clamped_jpos = _clamp_vector_step_f32(
                            op_jpos,
                            engagement_last_forwarded_jpos,
                            effective_max_step,
                        )
                        # Follow-up 9b: ALSO flatten the future
                        # window. The deploy's window-mode policy
                        # uses joint_pos_mj_future (9 slots,
                        # 0.1 s apart) to predict the next 0.9 s
                        # of motion. The operator's untouched
                        # future encodes "go all the way to
                        # operator-pose in 0.9 s" -- ~3 rad delta
                        # over 9 slots ~ 0.33 rad/slot, which
                        # the policy slams to follow even when
                        # the current jpos is properly rate-
                        # limited (the failure mode the user just
                        # reported). Broadcasting the clamped
                        # current jpos to all 9 slots tells the
                        # policy "operator wants to hold here";
                        # zeroing joint_vel_mj_future cancels
                        # the velocity prediction. After the
                        # engagement ramp completes the operator's
                        # actual future window flows through.
                        flat_future = np.broadcast_to(
                            clamped_jpos, (_NUM_FUTURE_SLOTS, NUM_BODY_DOFS)
                        ).astype(np.float32, copy=True)
                        zero_future_vel = _ZERO_QVEL_FUTURE.copy()
                        overrides_for_clamp = {
                            "joint_pos_mj": clamped_jpos,
                            "joint_pos_mj_future": flat_future,
                            "joint_vel_mj_future": zero_future_vel,
                        }
                        rebuilt = rebuild_msg_with_field_overrides(
                            latest_override,
                            args.upstream_topic,
                            overrides_for_clamp,
                        )
                        if rebuilt is None:
                            # Try downstream topic; some pose
                            # producers use it instead. If both
                            # fail, fall back to forwarding the
                            # original frame verbatim (slam-risk
                            # but preserves the operator's intent
                            # better than dropping the frame).
                            rebuilt = rebuild_msg_with_field_overrides(
                                latest_override,
                                args.downstream_topic,
                                overrides_for_clamp,
                            )
                        if rebuilt is None:
                            # The override frame likely doesn't
                            # carry the full v5 future window
                            # (e.g. a v4 token-only frame from a
                            # legacy publisher). Fall back to
                            # clamping just the current jpos --
                            # still better than verbatim forward,
                            # and most callers DO publish the
                            # full window.
                            rebuilt = rebuild_msg_with_jpos_override(
                                latest_override,
                                args.upstream_topic,
                                clamped_jpos,
                            )
                            if rebuilt is None:
                                rebuilt = rebuild_msg_with_jpos_override(
                                    latest_override,
                                    args.downstream_topic,
                                    clamped_jpos,
                                )
                        if rebuilt is not None:
                            fwd_msg = rebuilt
                            engagement_last_forwarded_jpos = (
                                clamped_jpos.copy()
                            )
                            op_jpos = clamped_jpos
                        # ELSE: no-op, fwd_msg stays as the
                        # original operator frame. The slam risk
                        # is back but only when the frame header
                        # is corrupt -- decode_pose_joint_pos_mj
                        # above already returned non-None so the
                        # rebuild SHOULD succeed; this is purely
                        # defensive.
                        engagement_clamp_remaining -= 1
                    elif (
                        engagement_clamp_remaining > 0
                        and op_jpos is not None
                        and engagement_last_forwarded_jpos is None
                    ):
                        # First override tick of the engagement
                        # ramp with NO anchor (LIVE never ran on
                        # the bridge, so we don't have a VLA pose
                        # to clamp toward). Seed the anchor from
                        # this operator pose and forward verbatim;
                        # subsequent ticks will clamp relative to
                        # this anchor. This is the cold-start
                        # smoke-test path; in production the anchor
                        # is always seeded at the engage edge from
                        # last_upstream_jpos.
                        engagement_last_forwarded_jpos = op_jpos.copy()
                        engagement_clamp_remaining -= 1
                    try:
                        pub.send(fwd_msg, zmq.NOBLOCK)
                        override_frames += 1
                    except zmq.Again:
                        pass
                    last_upstream_msg = fwd_msg
                    if op_jpos is not None:
                        last_upstream_jpos = op_jpos
                    last_upstream_s = now
                cur_state = STATE_OVERRIDE
            elif latest is not None:
                # Got fresh primary this tick -- forward raw bytes.
                try:
                    pub.send(latest, zmq.NOBLOCK)
                    fwd_frames += 1
                except zmq.Again:
                    pass
                cur_state = STATE_LIVE
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
                elif cur_state == STATE_OVERRIDE:
                    msg_txt = (
                        f"{prev_state} -> OVERRIDE (operator teleop "
                        f"override engaged; forwarding override port "
                        f"frames with engagement slow-step clamp "
                        f"active for the first "
                        f"{engagement_step_ramp_ticks} ticks)"
                    )
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
                if override_enabled:
                    if last_override_s < 0:
                        ovr_age_str = "never"
                    else:
                        ovr_age_str = (
                            f"{(now - last_override_s) * 1000:.0f}ms"
                        )
                    if teleop_mode_enabled:
                        # Strict mode: surface the manager's last
                        # reported mode + how long ago it landed,
                        # plus a one-shot "stale" hint when we'd
                        # block engagement. Motion/frozen counters
                        # are intentionally hidden -- they're noise
                        # in this path and operators kept reading
                        # them as the engage signal.
                        if last_teleop_mode_s < 0:
                            mode_age_str = "never"
                        else:
                            mode_age_str = (
                                f"{(now - last_teleop_mode_s) * 1000:.0f}ms"
                            )
                        if not teleop_mode_fresh:
                            stale_tag = " STALE"
                        elif current_teleop_mode == "OFF":
                            stale_tag = " OFF"
                        else:
                            stale_tag = ""
                        gate_str = (
                            f" gate(mode={current_teleop_mode} "
                            f"age={mode_age_str} "
                            f"msgs={teleop_mode_msgs} "
                            f"fail={teleop_mode_decode_failures}"
                            f"{stale_tag})"
                        )
                    else:
                        if frozen_ticks_threshold > 0:
                            frz_str = (
                                f" frozen(det={override_frozen_detected} "
                                f"streak={override_frozen_count}/"
                                f"{frozen_ticks_threshold} "
                                f"rel={override_frozen_release_events})"
                            )
                        else:
                            frz_str = " frozen(disabled)"
                        if engage_motion_threshold > 0:
                            mot_str = (
                                f" moving(streak={override_motion_count}/"
                                f"{engage_motion_threshold} "
                                f"sustained={override_motion_sustained})"
                            )
                        else:
                            mot_str = " moving(legacy:immediate)"
                        gate_str = f"{frz_str}{mot_str}"
                    ovr_str = (
                        f" override(active={override_active} "
                        f"age={ovr_age_str} fwd={override_frames} "
                        f"eng={override_engage_events} "
                        f"rel={override_release_events})"
                        f"{gate_str}"
                    )
                else:
                    ovr_str = ""
                print(
                    f"[pose_proxy] tick={tick} state={state_str} "
                    f"mode={idle_mode} upstream_age={age_str}{ovr_str} "
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
        if override_sub is not None:
            override_sub.close(linger=0)
        if teleop_mode_sub is not None:
            teleop_mode_sub.close(linger=0)
        if yaw_sub is not None:
            yaw_sub.close(linger=0)
        if vla_control_pub is not None:
            vla_control_pub.close(linger=0)
        ctx.term()

    ovr_done = ""
    if override_enabled:
        ovr_done = (
            f" override_fwd={override_frames} "
            f"override_engaged={override_engage_events} "
            f"override_released={override_release_events}"
        )
    print(
        f"[pose_proxy] done. total_ticks={tick} fwd={fwd_frames} "
        f"hold={hold_frames} blend={blend_frames} idle={idle_frames} "
        f"gap_skip={gap_skips}{ovr_done}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
