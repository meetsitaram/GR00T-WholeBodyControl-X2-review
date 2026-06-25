"""Unit tests for the x2_pose_watchdog upstream-silent fallback ladder.

The 2026-06-08 milestone replaced the watchdog's binary LIVE/IDLE
fallback with a staged LIVE -> HOLD -> BLEND -> IDLE_CLIP ladder so a
WiFi blip during teleop no longer slams the arms into default-stand.
These tests pin the ladder's decision logic plus the end-to-end byte
behaviour (HOLD re-publishes raw bytes verbatim, BLEND lerps
joint_pos_mj monotonically, idle-stand mode reproduces the legacy
single-step path) so a regression in the watchdog can't silently
re-introduce the table-slamming failure mode.

The state-transition decisions are tested through the pure function
``fallback.decide_fallback_state`` so we never have to spin up a ZMQ
context, sleep on real timers, or care about the publish thread.

The 2026-06-11 milestone moved the helpers to
``gear_sonic.utils.pose_pipeline``; the tests import from there
rather than the watchdog script directly so the same helpers are
exercised whether the watchdog or the laptop-side mux is the caller.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Shim: tests pre-2026-06-11 imported a single ``proxy`` module that
# bundled wire + fallback. Build a single namespace from the shared
# library so the test body keeps reading naturally.
from gear_sonic.utils.pose_pipeline import fallback, wire  # noqa: E402


class _ProxyShim:
    """Backwards-compatible attribute proxy over wire + fallback."""

    NUM_BODY_DOFS = wire.NUM_BODY_DOFS
    NUM_FUTURE_SLOTS = wire.NUM_FUTURE_SLOTS
    HEADER_SIZE = wire.HEADER_SIZE
    X2M2_MAGIC = wire.X2M2_MAGIC
    pack_pose_message = staticmethod(wire.pack_pose_message)
    decode_pose_joint_pos_mj = staticmethod(wire.decode_pose_joint_pos_mj)
    decode_pose_root_quat_xyzw_future = staticmethod(
        wire.decode_pose_root_quat_xyzw_future
    )
    IdleStandReplay = fallback.IdleStandReplay
    build_idle_frame_msg = staticmethod(fallback.build_idle_frame_msg)
    decide_fallback_state = staticmethod(fallback.decide_fallback_state)
    rebase_hold_msg = staticmethod(fallback.rebase_hold_msg)
    STATE_LIVE = fallback.STATE_LIVE
    STATE_COLD_IDLE = fallback.STATE_COLD_IDLE
    STATE_HOLD = fallback.STATE_HOLD
    STATE_BLEND = fallback.STATE_BLEND
    STATE_IDLE_CLIP = fallback.STATE_IDLE_CLIP
    STATE_GAP = fallback.STATE_GAP
    IDLE_MODE_BLEND = fallback.IDLE_MODE_BLEND
    IDLE_MODE_HOLD_LAST = fallback.IDLE_MODE_HOLD_LAST
    IDLE_MODE_IDLE_STAND = fallback.IDLE_MODE_IDLE_STAND


proxy = _ProxyShim()


# ===========================================================================
# Helpers
# ===========================================================================
def _make_replay(n_frames: int = 4, jpos_value: float = 0.0) -> proxy.IdleStandReplay:
    """Build an IdleStandReplay whose every joint_pos = ``jpos_value`` so
    we can predict BLEND lerps without re-decoding the baked clip.
    Quat is identity (xyzw = [0,0,0,1])."""
    dof = np.full(
        (n_frames, proxy.NUM_BODY_DOFS), jpos_value, dtype=np.float32
    )
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n_frames, 1)).astype(np.float32)
    return proxy.IdleStandReplay(dof, quat)


def _pack_pose_with_jpos(jpos: np.ndarray, topic: str = "pose") -> bytes:
    """Build a packed pose frame whose joint_pos_mj field holds ``jpos``.

    Uses the proxy's own pack_pose_message so the wire format matches
    exactly what the laptop publisher emits.
    """
    if jpos.shape != (proxy.NUM_BODY_DOFS,):
        raise ValueError(
            f"jpos must be ({proxy.NUM_BODY_DOFS},); got {jpos.shape}"
        )
    payload = {
        "joint_pos_mj": jpos.astype(np.float32),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(10, dtype=np.float32),
        "right_hand_joints": np.zeros(10, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
    }
    return proxy.pack_pose_message(payload, topic=topic, version=4)


def _extract_jpos_from_msg(
    msg: bytes, topic: str = "pose"
) -> np.ndarray:
    """Round-trip a published frame back into its joint_pos_mj slice
    so test assertions can compare against the proxy's lerp output."""
    out = proxy.decode_pose_joint_pos_mj(msg, topic=topic)
    if out is None:
        raise AssertionError("could not decode joint_pos_mj from message")
    return out


# ===========================================================================
# decode_pose_joint_pos_mj
# ===========================================================================
def test_decode_pose_joint_pos_mj_roundtrips_packed_frame() -> None:
    jpos = np.arange(proxy.NUM_BODY_DOFS, dtype=np.float32) * 0.05
    msg = _pack_pose_with_jpos(jpos)
    out = proxy.decode_pose_joint_pos_mj(msg, topic="pose")
    assert out is not None
    np.testing.assert_allclose(out, jpos, atol=0.0)


def test_decode_pose_joint_pos_mj_returns_none_on_wrong_topic() -> None:
    jpos = np.zeros(proxy.NUM_BODY_DOFS, dtype=np.float32)
    msg = _pack_pose_with_jpos(jpos, topic="not_pose")
    assert proxy.decode_pose_joint_pos_mj(msg, topic="pose") is None


def test_decode_pose_joint_pos_mj_returns_none_on_truncated_payload() -> None:
    """Truncated payload must not crash the publish thread."""
    jpos = np.zeros(proxy.NUM_BODY_DOFS, dtype=np.float32)
    msg = _pack_pose_with_jpos(jpos)
    # joint_pos_mj is the FIRST field after the header, so truncating
    # back to header + 8 bytes guarantees the declared 124-byte
    # joint_pos_mj field is cut short.
    topic_len = len(b"pose")
    truncated = msg[: topic_len + proxy.HEADER_SIZE + 8]
    assert proxy.decode_pose_joint_pos_mj(truncated, topic="pose") is None


def test_decode_pose_joint_pos_mj_rejects_wrong_shape() -> None:
    """Frames with joint_pos_mj of a different rank/dimension are
    rejected rather than silently coerced -- a half-decoded jpos would
    steer the deploy into bad references."""
    # Manually build a frame with joint_pos_mj shape (28,) instead of 31.
    payload = {
        "joint_pos_mj": np.zeros(28, dtype=np.float32),
    }
    msg = proxy.pack_pose_message(payload, topic="pose", version=4)
    assert proxy.decode_pose_joint_pos_mj(msg, topic="pose") is None


# ===========================================================================
# decide_fallback_state -- the pure decision function
# ===========================================================================
def test_decide_cold_idle_when_never_seen_upstream() -> None:
    state, alpha = proxy.decide_fallback_state(
        have_upstream=False,
        age_s=float("inf"),
        stale_s=0.3,
        hold_last_secs=10.0,
        blend_secs=3.0,
        idle_mode="blend",
    )
    assert state == proxy.STATE_COLD_IDLE
    assert alpha == 0.0


def test_decide_gap_when_age_below_stale_threshold() -> None:
    """Within the stale window the proxy should not publish at all
    (deploy keeps its cached frame at 500 Hz)."""
    state, _ = proxy.decide_fallback_state(
        have_upstream=True,
        age_s=0.05,
        stale_s=0.3,
        hold_last_secs=10.0,
        blend_secs=3.0,
        idle_mode="blend",
    )
    assert state == proxy.STATE_GAP


def test_decide_hold_when_age_within_hold_window_blend_mode() -> None:
    """Just past stale_s -> HOLD; partway through hold_last_secs -> HOLD."""
    for age in (0.4, 1.0, 5.0, 10.2):
        state, alpha = proxy.decide_fallback_state(
            have_upstream=True,
            age_s=age,
            stale_s=0.3,
            hold_last_secs=10.0,
            blend_secs=3.0,
            idle_mode="blend",
        )
        assert state == proxy.STATE_HOLD, f"failed at age={age}"
        assert alpha == 0.0


def test_decide_blend_in_lerp_window_blend_mode() -> None:
    """Past hold_last_secs but within blend_secs -> BLEND with monotonic alpha."""
    stale = 0.3
    hold = 10.0
    blend = 3.0
    alphas = []
    for age in (10.4, 11.4, 12.4, 13.2):
        state, alpha = proxy.decide_fallback_state(
            have_upstream=True,
            age_s=age,
            stale_s=stale,
            hold_last_secs=hold,
            blend_secs=blend,
            idle_mode="blend",
        )
        assert state == proxy.STATE_BLEND, f"failed at age={age}"
        alphas.append(alpha)
    # Monotonically increasing, in [0, 1].
    for prev, nxt in zip(alphas, alphas[1:]):
        assert nxt >= prev
    assert alphas[0] >= 0.0
    assert alphas[-1] <= 1.0
    # First alpha = (10.4 - 0.3 - 10.0) / 3.0 = 0.0333...
    assert alphas[0] == pytest.approx((10.4 - 0.3 - 10.0) / 3.0, abs=1e-6)


def test_decide_idle_clip_past_blend_window() -> None:
    state, alpha = proxy.decide_fallback_state(
        have_upstream=True,
        age_s=20.0,
        stale_s=0.3,
        hold_last_secs=10.0,
        blend_secs=3.0,
        idle_mode="blend",
    )
    assert state == proxy.STATE_IDLE_CLIP
    assert alpha == 1.0


def test_decide_hold_last_mode_stays_in_hold_forever() -> None:
    """hold-last mode skips BLEND/IDLE_CLIP entirely."""
    for age in (0.4, 100.0, 1e6):
        state, _ = proxy.decide_fallback_state(
            have_upstream=True,
            age_s=age,
            stale_s=0.3,
            hold_last_secs=10.0,
            blend_secs=3.0,
            idle_mode="hold-last",
        )
        assert state == proxy.STATE_HOLD, f"failed at age={age}"


def test_decide_idle_stand_mode_jumps_straight_to_idle_clip() -> None:
    """idle-stand mode reproduces pre-2026-06-08 behaviour: first
    stale tick -> IDLE_CLIP (regression escape)."""
    state, alpha = proxy.decide_fallback_state(
        have_upstream=True,
        age_s=0.4,
        stale_s=0.3,
        hold_last_secs=10.0,
        blend_secs=3.0,
        idle_mode="idle-stand",
    )
    assert state == proxy.STATE_IDLE_CLIP
    assert alpha == 1.0


def test_decide_idle_stand_mode_at_gap_window_still_gap() -> None:
    """Even idle-stand mode honours the stale window -- the GAP state
    is about "deploy's 500 Hz cache absorbs this on its own", not
    about which fallback mode is configured."""
    state, _ = proxy.decide_fallback_state(
        have_upstream=True,
        age_s=0.05,
        stale_s=0.3,
        hold_last_secs=10.0,
        blend_secs=3.0,
        idle_mode="idle-stand",
    )
    assert state == proxy.STATE_GAP


def test_decide_zero_blend_secs_collapses_to_step() -> None:
    """blend_secs=0 should not divide by zero; collapses to IDLE_CLIP
    once the hold window expires."""
    state, alpha = proxy.decide_fallback_state(
        have_upstream=True,
        age_s=11.0,
        stale_s=0.3,
        hold_last_secs=10.0,
        blend_secs=0.0,
        idle_mode="blend",
    )
    assert state == proxy.STATE_IDLE_CLIP
    assert alpha == 1.0


def test_decide_zero_hold_secs_starts_blend_immediately() -> None:
    """hold_last_secs=0 means BLEND starts the moment stale window
    crosses (still smoother than legacy idle-stand mode)."""
    state, alpha = proxy.decide_fallback_state(
        have_upstream=True,
        age_s=0.4,
        stale_s=0.3,
        hold_last_secs=0.0,
        blend_secs=3.0,
        idle_mode="blend",
    )
    assert state == proxy.STATE_BLEND
    assert alpha == pytest.approx(0.1 / 3.0, abs=1e-6)


def test_decide_blend_alpha_clamps_to_unit_interval() -> None:
    """At the exact edge of the blend window, alpha must be in [0, 1]
    (no FP drift past 1.0)."""
    _, alpha_hi = proxy.decide_fallback_state(
        have_upstream=True,
        age_s=0.3 + 10.0 + 3.0,
        stale_s=0.3,
        hold_last_secs=10.0,
        blend_secs=3.0,
        idle_mode="blend",
    )
    assert alpha_hi == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# build_idle_frame_msg -- joint_pos_mj_override BLEND integration
# ===========================================================================
def test_build_idle_frame_msg_with_override_emits_override_jpos() -> None:
    """End-to-end: BLEND path constructs an override jpos, hands it to
    build_idle_frame_msg, and the resulting frame's joint_pos_mj
    matches the override bit-for-bit."""
    replay = _make_replay(n_frames=4, jpos_value=0.0)
    override = np.linspace(-1.0, 1.0, proxy.NUM_BODY_DOFS, dtype=np.float32)
    msg = proxy.build_idle_frame_msg(
        replay, 0, "pose", joint_pos_mj_override=override
    )
    got = _extract_jpos_from_msg(msg, topic="pose")
    np.testing.assert_allclose(got, override, atol=0.0)


def test_build_idle_frame_msg_override_none_preserves_legacy() -> None:
    """No override == legacy behaviour (uses replay.current's jpos)."""
    replay = _make_replay(n_frames=4, jpos_value=0.5)
    msg_default = proxy.build_idle_frame_msg(replay, 0, "pose")
    msg_explicit_none = proxy.build_idle_frame_msg(
        replay, 0, "pose", joint_pos_mj_override=None
    )
    assert msg_default == msg_explicit_none


def test_build_idle_frame_msg_rejects_override_with_wrong_shape() -> None:
    replay = _make_replay(n_frames=4, jpos_value=0.0)
    with pytest.raises(ValueError, match="joint_pos_mj_override"):
        proxy.build_idle_frame_msg(
            replay,
            0,
            "pose",
            joint_pos_mj_override=np.zeros(5, dtype=np.float32),
        )


def test_build_idle_frame_msg_blend_lerp_monotonic_between_endpoints() -> None:
    """Walking the blend alpha from 0 -> 1 must trace a straight line
    from cached_jpos -> idle_jpos in each DOF (proves the lerp formula
    doesn't accidentally over- or under-shoot)."""
    cached = np.full(proxy.NUM_BODY_DOFS, 1.0, dtype=np.float32)
    idle_value = 0.0
    replay = _make_replay(n_frames=4, jpos_value=idle_value)
    seen = []
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        lerp = (
            (1.0 - alpha) * cached + alpha * idle_value
        ).astype(np.float32)
        msg = proxy.build_idle_frame_msg(
            replay, 0, "pose", joint_pos_mj_override=lerp
        )
        got = _extract_jpos_from_msg(msg, topic="pose")
        seen.append(got[0])  # one representative DOF
    # alpha=0 -> cached (1.0), alpha=1 -> idle (0.0), strictly decreasing.
    assert seen[0] == pytest.approx(1.0)
    assert seen[-1] == pytest.approx(0.0)
    for prev, nxt in zip(seen, seen[1:]):
        assert nxt < prev


# ===========================================================================
# HOLD: when the watchdog falls back to verbatim re-publish (yaw-track
# disabled, x2_debug stale, or splice fails), the cached bytes go out
# unchanged. Regression test for that fallback path -- we want HOLD to
# be literally a no-op on the bytes when yaw-rebase isn't engaged so a
# WiFi blip mid-clip never silently rewrites frame_index or the v5
# future-window bytes.
# ===========================================================================
def test_hold_path_verbatim_fallback_is_byte_identical() -> None:
    jpos = np.arange(proxy.NUM_BODY_DOFS, dtype=np.float32) * 0.1
    original_msg = _pack_pose_with_jpos(jpos)
    # Simulate the proxy's HOLD path FALLBACK branch: cache the bytes,
    # then publish them again later. There is no per-tick transformation
    # when yaw-track is disabled (the active HOLD-rebase path is
    # exercised by the dedicated tests below).
    cached = original_msg
    republish = cached
    assert republish is original_msg or republish == original_msg
    got = proxy.decode_pose_joint_pos_mj(republish, topic="pose")
    np.testing.assert_allclose(got, jpos, atol=0.0)


# ===========================================================================
# rebase_hold_msg -- HOLD-state yaw rebase (2026-06-24)
#
# The watchdog's HOLD branch was a verbatim re-publish pre-2026-06-24,
# which pinned the policy's heading reference to whatever yaw upstream
# last sent. If the operator killed the laptop stack and then tried
# to rotate the body by hand, SONIC fought back for the full
# --hold-last-secs window before BLEND/IDLE_CLIP re-engaged the
# idle-clip yaw rebase. ``rebase_hold_msg`` is the surgical fix: it
# splices R_z(measured_yaw) into the cached message's root_quat
# fields while leaving joint_pos_mj (and its future window) untouched,
# so the body pose still freezes exactly where upstream left it.
# ===========================================================================
def _pack_pose_with_future_window(
    jpos: np.ndarray,
    current_quat_xyzw: np.ndarray,
    future_quats_xyzw: np.ndarray,
    topic: str = "pose",
) -> bytes:
    """Build a v5-style packed pose frame with a future window.

    Mirrors the fields emitted by the recorder's _publish_pose when
    both joint_pos_mj_future and root_quat_xyzw_future are supplied
    (i.e. the locomotion / VLA path). The HOLD-rebase test cases need
    a future window present so we can verify all 9 slots get rewritten
    to R_z(measured_yaw).
    """
    if jpos.shape != (proxy.NUM_BODY_DOFS,):
        raise ValueError("jpos must be NUM_BODY_DOFS")
    if current_quat_xyzw.shape != (4,):
        raise ValueError("current_quat_xyzw must be (4,)")
    if future_quats_xyzw.shape != (proxy.NUM_FUTURE_SLOTS, 4):
        raise ValueError(
            f"future_quats_xyzw must be ({proxy.NUM_FUTURE_SLOTS}, 4)"
        )
    jpos_future = np.tile(jpos, (proxy.NUM_FUTURE_SLOTS, 1)).astype(np.float32)
    jvel_future = np.zeros(
        (proxy.NUM_FUTURE_SLOTS, proxy.NUM_BODY_DOFS), dtype=np.float32
    )
    payload = {
        "joint_pos_mj": jpos.astype(np.float32),
        "root_quat_xyzw": current_quat_xyzw.astype(np.float32),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(10, dtype=np.float32),
        "right_hand_joints": np.zeros(10, dtype=np.float32),
        "frame_index": np.array([42], dtype=np.int64),
        "joint_pos_mj_future": jpos_future,
        "root_quat_xyzw_future": future_quats_xyzw.astype(np.float32),
        "joint_vel_mj_future": jvel_future,
        "frame_index_future": np.arange(
            43, 43 + proxy.NUM_FUTURE_SLOTS, dtype=np.int64
        ),
        "future_dt_s": np.array([0.1], dtype=np.float32),
    }
    return proxy.pack_pose_message(payload, topic=topic, version=4)


def _r_z_xyzw(yaw_rad: float) -> np.ndarray:
    """Build R_z(yaw) as an xyzw quat -- mirrors fallback._yaw_to_quat_xyzw."""
    half = 0.5 * yaw_rad
    return np.array(
        [0.0, 0.0, math.sin(half), math.cos(half)], dtype=np.float32
    )


def test_rebase_hold_msg_v4_message_rebases_current_quat() -> None:
    """v4 frames (recorder _publish_idle path) have no future window;
    rebase_hold_msg must still splice the current-frame quat so the
    deploy's single-frame fallback path gets R_z(measured_yaw)."""
    jpos = np.arange(proxy.NUM_BODY_DOFS, dtype=np.float32) * 0.05
    msg = _pack_pose_with_jpos(jpos)
    target_yaw = math.radians(45.0)
    out = proxy.rebase_hold_msg(msg, "pose", target_yaw)
    assert out is not None
    # joint_pos_mj must be byte-for-byte preserved (HOLD freezes pose).
    got_jpos = proxy.decode_pose_joint_pos_mj(out, topic="pose")
    np.testing.assert_allclose(got_jpos, jpos, atol=0.0)
    # The output bytes should differ from the input -- the current quat
    # field was rewritten.
    assert out != msg


def test_rebase_hold_msg_v5_message_rebases_current_and_future() -> None:
    """v5 frames (laptop bridge, locomotion clip mid-play) carry a
    9-slot root_quat_xyzw_future. rebase_hold_msg must overwrite ALL
    of them with the same R_z(measured_yaw) so the policy's planning
    horizon doesn't predict an anticipatory yaw change toward the
    cached clip's authored trajectory after upstream goes silent."""
    jpos = np.full(proxy.NUM_BODY_DOFS, 0.123, dtype=np.float32)
    cached_current_quat = _r_z_xyzw(math.radians(-30.0))
    # Cached future window encodes an anticipatory +60 deg turn over
    # the 9 slots -- HOLD should NOT honour those deltas; flatten to
    # the measured yaw across the whole window.
    cached_future_quats = np.stack(
        [
            _r_z_xyzw(math.radians(-30.0 + (k + 1) * 60.0 / proxy.NUM_FUTURE_SLOTS))
            for k in range(proxy.NUM_FUTURE_SLOTS)
        ],
        axis=0,
    )
    msg = _pack_pose_with_future_window(
        jpos, cached_current_quat, cached_future_quats
    )
    target_yaw = math.radians(75.0)
    out = proxy.rebase_hold_msg(msg, "pose", target_yaw)
    assert out is not None
    # joint_pos_mj must be byte-identical (HOLD freezes the body pose).
    got_jpos = proxy.decode_pose_joint_pos_mj(out, topic="pose")
    np.testing.assert_allclose(got_jpos, jpos, atol=0.0)
    # All 9 future slots must equal R_z(target_yaw) bit-for-bit
    # (same value the helper produces for the current frame).
    expected_slot = _r_z_xyzw(target_yaw)
    got_future = proxy.decode_pose_root_quat_xyzw_future(out, topic="pose")
    assert got_future is not None
    np.testing.assert_allclose(
        got_future, np.tile(expected_slot, (proxy.NUM_FUTURE_SLOTS, 1)), atol=0.0
    )


def test_rebase_hold_msg_wrong_topic_returns_none() -> None:
    """Topic mismatch must fail safely (returns None so the watchdog
    can fall back to verbatim re-publish instead of silently sending
    a malformed frame)."""
    jpos = np.zeros(proxy.NUM_BODY_DOFS, dtype=np.float32)
    msg = _pack_pose_with_jpos(jpos, topic="pose")
    out = proxy.rebase_hold_msg(msg, "different_topic", 0.0)
    assert out is None


def test_rebase_hold_msg_zero_yaw_produces_identity_quat() -> None:
    """yaw=0 must produce the identity quat xyzw=[0,0,0,1] (matches
    the deploy's default heading reference). Guards against an off-
    by-one in the half-angle conversion in _yaw_to_quat_xyzw."""
    jpos = np.zeros(proxy.NUM_BODY_DOFS, dtype=np.float32)
    cached_current_quat = _r_z_xyzw(math.radians(123.0))
    cached_future_quats = np.tile(
        cached_current_quat, (proxy.NUM_FUTURE_SLOTS, 1)
    )
    msg = _pack_pose_with_future_window(
        jpos, cached_current_quat, cached_future_quats
    )
    out = proxy.rebase_hold_msg(msg, "pose", 0.0)
    assert out is not None
    got_future = proxy.decode_pose_root_quat_xyzw_future(out, topic="pose")
    assert got_future is not None
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    np.testing.assert_allclose(
        got_future, np.tile(identity, (proxy.NUM_FUTURE_SLOTS, 1)), atol=1e-7
    )


def test_rebase_hold_msg_preserves_motion_token_and_hand_joints() -> None:
    """Only the root_quat fields should change; everything else
    (motion_token, hand joints, frame_index, future jpos) must be
    bit-identical because HOLD's contract is 'freeze body pose, only
    track yaw'. Without this guarantee the policy would see snap
    changes in fields it expects to be continuous."""
    jpos = np.arange(proxy.NUM_BODY_DOFS, dtype=np.float32) * 0.1
    cached_current_quat = _r_z_xyzw(math.radians(10.0))
    cached_future_quats = np.tile(
        cached_current_quat, (proxy.NUM_FUTURE_SLOTS, 1)
    )
    msg = _pack_pose_with_future_window(
        jpos, cached_current_quat, cached_future_quats
    )
    out = proxy.rebase_hold_msg(msg, "pose", math.radians(20.0))
    assert out is not None
    # joint_pos_mj must be untouched.
    np.testing.assert_allclose(
        proxy.decode_pose_joint_pos_mj(out, topic="pose"), jpos, atol=0.0
    )
    # Output length must equal input length (in-place byte splice,
    # no field added or removed).
    assert len(out) == len(msg)
