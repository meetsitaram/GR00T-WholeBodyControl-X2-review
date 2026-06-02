"""Unit tests for the x2_pose_proxy yaw-rebase + x2_debug decoder additions.

The proxy is shipped to PC2 by ``pc2_bringup.sh`` and runs there with a
minimal numpy + pyzmq + stdlib budget. These tests pin the three new
helpers (yaw_from_quat_wxyz, rebase_quats_xyzw_by_yaw,
decode_x2_debug_base_quat) so a regression in the proxy can't silently
re-introduce the "robot snaps back to spawn heading on planner-stack
termination" symptom in production.

Each helper is exercised in isolation:

1. ``yaw_from_quat_wxyz`` -- closed-form ZYX-extrinsic yaw extraction.
   Validated against scipy's ``Rotation.as_euler("zyx")[0]`` over 2000
   random rotations including pitch + roll components (which are
   intentionally dropped on extraction).

2. ``rebase_quats_xyzw_by_yaw`` -- vectorised left-multiply by R_z(yaw).
   Confirms a yaw=0 baked clip becomes R_z(measured) after rebase
   (the production hot path), and that the inverse round-trips.

3. ``decode_x2_debug_base_quat`` -- tolerant decoder for the C++
   deploy's packed-binary x2_debug PUB. We construct synthetic frames
   matching the on-wire layout (1280-byte JSON header + concatenated
   binary fields) and verify base_quat is found at any position in
   the field list, mistyped frames return None, and truncated frames
   don't crash.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

REPO_ROOT = Path(__file__).resolve().parent.parent
PROXY_DIR = REPO_ROOT / "gear_sonic_deploy" / "scripts"
if str(PROXY_DIR) not in sys.path:
    sys.path.insert(0, str(PROXY_DIR))

import x2_pose_proxy as proxy  # noqa: E402


# ===========================================================================
# yaw_from_quat_wxyz
# ===========================================================================
def test_yaw_identity_quat_returns_zero() -> None:
    assert proxy.yaw_from_quat_wxyz(
        np.array([1.0, 0.0, 0.0, 0.0])
    ) == pytest.approx(0.0, abs=1e-12)


def test_yaw_pure_z_rotation_roundtrips_full_range() -> None:
    """A pure R_z(yaw) wxyz quat must come back as ``yaw``."""
    for yaw_rad in (-3.0, -1.5, -0.7, 0.0, 0.3, 1.2, 2.4, 3.0):
        half = 0.5 * yaw_rad
        q_wxyz = np.array(
            [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64
        )
        assert proxy.yaw_from_quat_wxyz(q_wxyz) == pytest.approx(
            yaw_rad, abs=1e-9
        )


def test_yaw_matches_scipy_extrinsic_zyx_over_random_orientations() -> None:
    """Closed-form must match scipy's as_euler("zyx")[0] convention so
    yaw values extracted here are directly comparable to every other
    yaw-touching site in the gear_sonic stack."""
    rng = np.random.default_rng(0)
    max_err_deg = 0.0
    for _ in range(2000):
        yaw_deg = rng.uniform(-179, 179)
        pitch_deg = rng.uniform(-60, 60)
        roll_deg = rng.uniform(-60, 60)
        r = Rot.from_euler(
            "zyx", [yaw_deg, pitch_deg, roll_deg], degrees=True
        )
        q_xyzw = r.as_quat()
        q_wxyz = np.array(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
        )
        got_deg = math.degrees(proxy.yaw_from_quat_wxyz(q_wxyz))
        err = abs(((got_deg - yaw_deg) + 180) % 360 - 180)
        if err > max_err_deg:
            max_err_deg = err
    assert max_err_deg < 1e-6, (
        f"yaw extraction drifted from scipy: max_err={max_err_deg:.3e} deg"
    )


def test_yaw_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="length 4"):
        proxy.yaw_from_quat_wxyz(np.array([1.0, 0.0, 0.0]))


# ===========================================================================
# rebase_quats_xyzw_by_yaw
# ===========================================================================
def test_rebase_yaw_zero_is_identity() -> None:
    """yaw=0 must leave quats bit-identical (modulo dtype)."""
    rng = np.random.default_rng(1)
    q = rng.standard_normal((5, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    out = proxy.rebase_quats_xyzw_by_yaw(q, 0.0)
    np.testing.assert_allclose(out, q, atol=1e-7)


def test_rebase_baked_yaw_zero_clip_becomes_rz_measured() -> None:
    """The production hot path: baked idle_stand has R_z(0) for every
    frame (yaw-aligned at bake time). After rebase by measured_yaw,
    the output must be R_z(measured_yaw) exactly."""
    n = 10
    baked = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    for measured_yaw_deg in (-90, -35, 0, 12.5, 67, 175):
        measured_yaw = math.radians(measured_yaw_deg)
        out = proxy.rebase_quats_xyzw_by_yaw(baked, measured_yaw)
        # Every row should be R_z(measured_yaw) packed as xyzw.
        half = 0.5 * measured_yaw
        expected = np.tile(
            [0.0, 0.0, math.sin(half), math.cos(half)], (n, 1)
        ).astype(np.float32)
        np.testing.assert_allclose(out, expected, atol=1e-6)


def test_rebase_delta_is_a_pure_world_z_rotation() -> None:
    """Yaw rebase is a left-multiply by R_z(yaw). So the relative
    rotation out * inv(in) MUST be a pure world-z rotation by exactly
    ``measured_yaw`` -- pitch / roll components untouched. This is the
    only invariant we can pin without falling into the
    Euler-decomposition trap: scipy.as_euler("zyx") on a composed
    rotation does NOT return yaw as a simple sum of input yaw +
    rebase yaw when pitch / roll are non-zero (the ZYX decomp
    re-projects the composed rotation onto its own basis)."""
    r_baked = Rot.from_euler("zyx", [12.0, 10.0, -5.0], degrees=True)
    q_xyzw = r_baked.as_quat()
    baked = q_xyzw.reshape(1, 4).astype(np.float32)
    measured_yaw = math.radians(45.0)
    out = proxy.rebase_quats_xyzw_by_yaw(baked, measured_yaw)
    delta = Rot.from_quat(out[0].astype(np.float64)) * Rot.from_quat(
        q_xyzw
    ).inv()
    rotvec = delta.as_rotvec()
    # Axis must be world-z (no x / y components beyond float noise).
    assert abs(rotvec[0]) < 1e-5
    assert abs(rotvec[1]) < 1e-5
    # Magnitude must equal measured_yaw exactly.
    assert float(rotvec[2]) == pytest.approx(measured_yaw, abs=1e-5)


def test_rebase_yaw_then_reverse_yaw_is_identity() -> None:
    """Round-trip: rebase by +yaw then by -yaw must restore the input."""
    rng = np.random.default_rng(2)
    n = 7
    q = rng.standard_normal((n, 4)).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    yaw = 1.234
    out = proxy.rebase_quats_xyzw_by_yaw(
        proxy.rebase_quats_xyzw_by_yaw(q, yaw), -yaw
    )
    # Allow tiny norm drift but check rotational equivalence.
    for i in range(n):
        diff = (
            Rot.from_quat(q[i].astype(np.float64)).inv()
            * Rot.from_quat(out[i].astype(np.float64))
        ).as_rotvec()
        assert float(np.linalg.norm(diff)) < 1e-5


def test_rebase_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        proxy.rebase_quats_xyzw_by_yaw(np.zeros(4), 0.0)


# ===========================================================================
# decode_x2_debug_base_quat
# ===========================================================================
def _pack_x2_debug_frame(
    fields: list[dict],
    payload: bytes,
    topic: str = "x2_debug",
) -> bytes:
    """Build a minimal-but-correct x2_debug-shaped wire message."""
    header = {"v": 1, "endian": "le", "count": 1, "fields": fields}
    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    if len(header_json) > proxy.X2_DEBUG_HEADER_SIZE:
        raise ValueError("synthetic header too large for test")
    header_bytes = header_json.ljust(proxy.X2_DEBUG_HEADER_SIZE, b"\x00")
    return topic.encode("utf-8") + header_bytes + payload


def test_decode_x2_debug_extracts_base_quat_when_first_field() -> None:
    quat = np.array([0.5, 0.1, -0.2, 0.8], dtype=np.float64)
    msg = _pack_x2_debug_frame(
        [{"name": "base_quat", "dtype": "f64", "shape": [4]}],
        quat.tobytes(),
    )
    out = proxy.decode_x2_debug_base_quat(msg)
    assert out is not None
    np.testing.assert_allclose(out, quat, atol=0.0)


def test_decode_x2_debug_extracts_base_quat_after_other_fields() -> None:
    """The real x2_debug frame has base_quat several fields in; decoder
    must walk the header in order and accumulate the byte cursor."""
    now_f64 = struct.pack("<d", 12345.6789)
    policy_time = struct.pack("<d", 0.42)
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    msg = _pack_x2_debug_frame(
        [
            {"name": "ros_timestamp", "dtype": "f64", "shape": [1]},
            {"name": "policy_time", "dtype": "f64", "shape": [1]},
            {"name": "base_quat", "dtype": "f64", "shape": [4]},
            # Trailing field that the decoder should never reach.
            {"name": "trailing", "dtype": "f32", "shape": [99]},
        ],
        now_f64 + policy_time + quat.tobytes(),
    )
    out = proxy.decode_x2_debug_base_quat(msg)
    assert out is not None
    np.testing.assert_allclose(out, quat, atol=0.0)


def test_decode_x2_debug_returns_none_when_field_absent() -> None:
    """Frames without base_quat are valid (some debug variants may
    omit it); decoder must return None rather than crash."""
    msg = _pack_x2_debug_frame(
        [{"name": "ros_timestamp", "dtype": "f64", "shape": [1]}],
        struct.pack("<d", 0.0),
    )
    assert proxy.decode_x2_debug_base_quat(msg) is None


def test_decode_x2_debug_returns_none_on_wrong_topic() -> None:
    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    msg = _pack_x2_debug_frame(
        [{"name": "base_quat", "dtype": "f64", "shape": [4]}],
        quat.tobytes(),
        topic="some_other_topic",
    )
    assert proxy.decode_x2_debug_base_quat(msg, topic="x2_debug") is None


def test_decode_x2_debug_returns_none_on_truncated_payload() -> None:
    """Payload shorter than declared field bytes must not crash."""
    msg = _pack_x2_debug_frame(
        [{"name": "base_quat", "dtype": "f64", "shape": [4]}],
        b"\x00" * 8,  # only 8 bytes; base_quat declared 32
    )
    assert proxy.decode_x2_debug_base_quat(msg) is None


def test_decode_x2_debug_returns_none_on_wrong_dtype() -> None:
    """A base_quat field with the wrong dtype (e.g. f32) is suspect; we
    refuse to coerce silently because a half-decoded quat would steer
    the policy into bad references."""
    msg = _pack_x2_debug_frame(
        [{"name": "base_quat", "dtype": "f32", "shape": [4]}],
        np.zeros(4, dtype=np.float32).tobytes(),
    )
    assert proxy.decode_x2_debug_base_quat(msg) is None


def test_decode_x2_debug_returns_none_on_malformed_header() -> None:
    bad = b"x2_debug" + b"{not json"[:8].ljust(
        proxy.X2_DEBUG_HEADER_SIZE, b"\x00"
    ) + b""
    assert proxy.decode_x2_debug_base_quat(bad) is None


# ===========================================================================
# build_idle_frame_msg integration -- yaw rebase plumbs through
# ===========================================================================
def test_build_idle_frame_msg_with_yaw_rebase_emits_rotated_quat() -> None:
    """End-to-end: a baked yaw=0 clip + yaw rebase parameter produces
    a published frame whose root_quat_xyzw is R_z(yaw)."""
    n = 4
    dof = np.zeros((n, proxy.NUM_BODY_DOFS), dtype=np.float32)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    replay = proxy.IdleStandReplay(dof, quat)
    yaw_rad = math.radians(35.0)
    msg = proxy.build_idle_frame_msg(
        replay, 0, "pose", yaw_rebase_rad=yaw_rad
    )
    # Wire layout: topic_bytes + 1280-byte header + binary payload.
    assert msg.startswith(b"pose")
    body = msg[len(b"pose"):]
    header_blob = body[:proxy.HEADER_SIZE].rstrip(b"\x00")
    header = json.loads(header_blob.decode("utf-8"))
    fnames = [f["name"] for f in header["fields"]]
    assert "root_quat_xyzw" in fnames
    # Walk the field offsets in order to pull root_quat_xyzw bytes.
    cursor = 0
    bpe_lut = {"f32": 4, "f64": 8, "i32": 4, "i64": 8, "u8": 1, "bool": 1}
    payload = body[proxy.HEADER_SIZE:]
    root_quat_bytes = None
    for f in header["fields"]:
        nelem = 1
        for s in f["shape"]:
            nelem *= int(s)
        nbytes = nelem * bpe_lut[f["dtype"]]
        if f["name"] == "root_quat_xyzw":
            root_quat_bytes = payload[cursor:cursor + nbytes]
            break
        cursor += nbytes
    assert root_quat_bytes is not None
    out = np.frombuffer(root_quat_bytes, dtype=np.float32)
    half = 0.5 * yaw_rad
    np.testing.assert_allclose(
        out,
        [0.0, 0.0, math.sin(half), math.cos(half)],
        atol=1e-6,
    )


def test_build_idle_frame_msg_without_rebase_emits_baked_yaw() -> None:
    """yaw_rebase_rad=None preserves the legacy behaviour bit-for-bit."""
    n = 4
    dof = np.zeros((n, proxy.NUM_BODY_DOFS), dtype=np.float32)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32)
    replay = proxy.IdleStandReplay(dof, quat)
    msg_a = proxy.build_idle_frame_msg(replay, 0, "pose", yaw_rebase_rad=None)
    msg_b = proxy.build_idle_frame_msg(replay, 0, "pose")
    assert msg_a == msg_b
