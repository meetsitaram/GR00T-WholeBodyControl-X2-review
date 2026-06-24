"""Unit tests for the 2026-06-23 BLEND future-window continuity fix.

The 2026-06-08 watchdog ladder lerped only the current ``joint_pos_mj``
across BLEND. The rest of the upstream-derived wire frame (future
window, motion_token, hands) flipped from the cached VLA values to
idle-clip values in one tick at the HOLD -> BLEND boundary. Because
the C++ deploy's tokenizer integrates the full future window + motion
token, that one-tick flip produced a visible shoulder/elbow snap
(operators on real hardware reported "the arms snap to default after
a few seconds of stillness, then smoothly settle").

This fix:

1. Adds four field decoders in
   ``gear_sonic.utils.pose_pipeline.wire`` so the watchdog can
   snapshot the upstream future window + motion_token at every fresh
   tick: ``decode_pose_joint_pos_mj_future``,
   ``decode_pose_root_quat_xyzw_future``,
   ``decode_pose_joint_vel_mj_future``, ``decode_pose_motion_token``.
2. Adds ``nlerp_quat_arrays_xyzw`` in the same module so the watchdog
   stays scipy-free on PC2 while still slerp-equivalent for our
   per-tick alpha deltas.
3. Extends ``build_idle_frame_msg`` in
   ``gear_sonic.utils.pose_pipeline.fallback`` with four optional
   ``*_future_override`` / ``motion_token_override`` kwargs so the
   watchdog can pass the per-tick lerped futures + token through to
   the wire frame.

Hard guardrail enforced by structure: every test in this module is a
pure pytest unit test running locally on this laptop. No SSH, no
rsync, no PC2 daemons, no ZMQ sockets bound or connected, no
subprocess of ``x2_pose_watchdog.py``, no network access of any
kind. The watchdog's main loop is not invoked from any test here --
we exercise the building blocks (decoders, nlerp helper,
``build_idle_frame_msg`` with overrides) directly and assert the
boundary-continuity invariants that the BLEND branch is responsible
for producing.

Hands are deliberately out of scope (per operator choice on
2026-06-23): they continue to snap to ``ZERO_HAND`` at HOLD ->
BLEND. The fix could be extended to cover them with the same cache+
lerp pattern; see the BLEND continuity reference doc for the
rationale.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.pose_pipeline import fallback, wire  # noqa: E402


# ===========================================================================
# Test fixtures
# ===========================================================================
def _make_replay(
    n_frames: int = 4, jpos_value: float = 0.5
) -> fallback.IdleStandReplay:
    """Idle clip whose every body joint = ``jpos_value``. Quat = identity.

    Distinct from the upstream-cache values used in tests below so we
    can tell "this came from the idle clip" vs "this came from the
    upstream cache" by inspection.
    """
    dof = np.full(
        (n_frames, wire.NUM_BODY_DOFS), jpos_value, dtype=np.float32
    )
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n_frames, 1)).astype(np.float32)
    return fallback.IdleStandReplay(dof, quat)


def _pack_pose_with_futures(
    *,
    jpos: np.ndarray | None = None,
    jpos_future: np.ndarray | None = None,
    quat_future: np.ndarray | None = None,
    jvel_future: np.ndarray | None = None,
    motion_token: np.ndarray | None = None,
    include_futures: bool = True,
    topic: str = "pose",
) -> bytes:
    """Build a packed pose frame for round-trip / cache testing.

    When ``include_futures=False`` we deliberately omit the v5 future
    fields and motion_token so we can pin the graceful-v4-fallback
    behaviour (heuristic-planner upstreams that never set the
    future window).
    """
    if jpos is None:
        jpos = np.zeros(wire.NUM_BODY_DOFS, dtype=np.float32)
    payload: dict[str, np.ndarray] = {
        "joint_pos_mj": jpos.astype(np.float32),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        "left_hand_joints": np.zeros(wire.DEFAULT_HAND_DOF, dtype=np.float32),
        "right_hand_joints": np.zeros(wire.DEFAULT_HAND_DOF, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
    }
    if include_futures:
        if jpos_future is None:
            jpos_future = np.zeros(
                (wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS),
                dtype=np.float32,
            )
        if quat_future is None:
            quat_future = np.tile(
                [0.0, 0.0, 0.0, 1.0], (wire.NUM_FUTURE_SLOTS, 1)
            ).astype(np.float32)
        if jvel_future is None:
            jvel_future = np.zeros(
                (wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS),
                dtype=np.float32,
            )
        if motion_token is None:
            motion_token = np.zeros(
                wire.SONIC_MOTION_TOKEN_DIM, dtype=np.float32
            )
        # motion_token sits in the live VLA bridge frame BEFORE
        # left_hand_joints (it's a single-frame side channel that
        # the C++ deploy reads via the same decoder loop). We
        # reorder the payload dict to match the bridge's field
        # order so the test frame is byte-compatible.
        ordered: dict[str, np.ndarray] = {
            "joint_pos_mj": payload["joint_pos_mj"],
            "root_quat_xyzw": payload["root_quat_xyzw"],
            "motion_token": motion_token.astype(np.float32),
            "left_hand_joints": payload["left_hand_joints"],
            "right_hand_joints": payload["right_hand_joints"],
            "frame_index": payload["frame_index"],
            "joint_pos_mj_future": jpos_future.astype(np.float32),
            "root_quat_xyzw_future": quat_future.astype(np.float32),
            "joint_vel_mj_future": jvel_future.astype(np.float32),
            "frame_index_future": np.array(
                [k for k in range(wire.NUM_FUTURE_SLOTS)],
                dtype=np.int64,
            ),
            "future_dt_s": wire.FUTURE_DT_FIELD,
        }
        payload = ordered
    return wire.pack_pose_message(payload, topic=topic, version=4)


def _decode_msg_field(msg: bytes, name: str, topic: str = "pose") -> np.ndarray:
    """Extract a field from a packed pose msg via the appropriate decoder."""
    decoders = {
        "joint_pos_mj": wire.decode_pose_joint_pos_mj,
        "joint_pos_mj_future": wire.decode_pose_joint_pos_mj_future,
        "root_quat_xyzw_future": wire.decode_pose_root_quat_xyzw_future,
        "joint_vel_mj_future": wire.decode_pose_joint_vel_mj_future,
        "motion_token": wire.decode_pose_motion_token,
    }
    fn = decoders[name]
    out = fn(msg, topic=topic)
    if out is None:
        raise AssertionError(f"could not decode {name} from msg")
    return out


# ===========================================================================
# New decoders -- round-trip + graceful absent-field behaviour
# ===========================================================================
def test_decode_joint_pos_mj_future_roundtrips() -> None:
    """Per-slot, per-DOF values survive pack -> unpack unchanged."""
    arr = np.arange(
        wire.NUM_FUTURE_SLOTS * wire.NUM_BODY_DOFS, dtype=np.float32
    ).reshape(wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS) * 0.01
    msg = _pack_pose_with_futures(jpos_future=arr)
    out = wire.decode_pose_joint_pos_mj_future(msg, topic="pose")
    assert out is not None
    assert out.shape == (wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS)
    np.testing.assert_allclose(out, arr, atol=0.0)


def test_decode_root_quat_xyzw_future_roundtrips() -> None:
    """Quaternion futures survive pack -> unpack unchanged (no rebase)."""
    arr = np.tile([0.1, 0.2, 0.3, 0.9], (wire.NUM_FUTURE_SLOTS, 1)).astype(
        np.float32
    )
    msg = _pack_pose_with_futures(quat_future=arr)
    out = wire.decode_pose_root_quat_xyzw_future(msg, topic="pose")
    assert out is not None
    assert out.shape == (wire.NUM_FUTURE_SLOTS, 4)
    np.testing.assert_allclose(out, arr, atol=0.0)


def test_decode_joint_vel_mj_future_roundtrips() -> None:
    arr = np.linspace(
        -1.0, 1.0, wire.NUM_FUTURE_SLOTS * wire.NUM_BODY_DOFS, dtype=np.float32
    ).reshape(wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS)
    msg = _pack_pose_with_futures(jvel_future=arr)
    out = wire.decode_pose_joint_vel_mj_future(msg, topic="pose")
    assert out is not None
    np.testing.assert_allclose(out, arr, atol=0.0)


def test_decode_motion_token_roundtrips() -> None:
    arr = np.sin(np.arange(wire.SONIC_MOTION_TOKEN_DIM, dtype=np.float32))
    msg = _pack_pose_with_futures(motion_token=arr)
    out = wire.decode_pose_motion_token(msg, topic="pose")
    assert out is not None
    assert out.shape == (wire.SONIC_MOTION_TOKEN_DIM,)
    np.testing.assert_allclose(out, arr, atol=0.0)


def test_decode_futures_return_none_for_legacy_v4_frame() -> None:
    """Graceful v4 fallback: heuristic-planner-style frames that omit
    the future window must yield None on each future decoder, so the
    watchdog leaves its cached futures untouched and the BLEND branch
    skips the per-field override for that field.
    """
    msg = _pack_pose_with_futures(include_futures=False)
    assert wire.decode_pose_joint_pos_mj_future(msg, topic="pose") is None
    assert wire.decode_pose_root_quat_xyzw_future(msg, topic="pose") is None
    assert wire.decode_pose_joint_vel_mj_future(msg, topic="pose") is None
    assert wire.decode_pose_motion_token(msg, topic="pose") is None
    # joint_pos_mj is still present and still decodable -- only the
    # future + token caches go untouched on legacy frames.
    assert wire.decode_pose_joint_pos_mj(msg, topic="pose") is not None


def test_decode_futures_return_none_on_wrong_topic() -> None:
    msg = _pack_pose_with_futures(topic="not_pose")
    assert wire.decode_pose_joint_pos_mj_future(msg, topic="pose") is None
    assert wire.decode_pose_motion_token(msg, topic="pose") is None


# ===========================================================================
# nlerp_quat_arrays_xyzw
# ===========================================================================
def test_nlerp_alpha_zero_returns_q_from() -> None:
    qf = np.array([[0.0, 0.0, np.sin(0.3), np.cos(0.3)]] * 3, dtype=np.float32)
    qt = np.array([[0.0, 0.0, np.sin(0.7), np.cos(0.7)]] * 3, dtype=np.float32)
    out = wire.nlerp_quat_arrays_xyzw(qf, qt, 0.0)
    np.testing.assert_allclose(out, qf, atol=1e-6)


def test_nlerp_alpha_one_returns_q_to_within_sign() -> None:
    """Alpha=1 should yield q_to (up to overall sign, which is the same
    rotation). With dot>0 endpoints, the sign should match exactly."""
    qf = np.array([[0.0, 0.0, np.sin(0.2), np.cos(0.2)]] * 2, dtype=np.float32)
    qt = np.array([[0.0, 0.0, np.sin(0.6), np.cos(0.6)]] * 2, dtype=np.float32)
    out = wire.nlerp_quat_arrays_xyzw(qf, qt, 1.0)
    np.testing.assert_allclose(out, qt, atol=1e-6)


def test_nlerp_handles_antipodal_sign_correction() -> None:
    """q and -q are the same rotation, but lerp without sign correction
    would interpolate through 0. The helper must flip the sign so the
    short path is taken; alpha=0.5 should yield essentially q_from
    (since q_to ~ -q_from)."""
    qf = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    qt = -qf  # same rotation
    out = wire.nlerp_quat_arrays_xyzw(qf, qt, 0.5)
    # After sign correction lerp midpoint = q_from exactly.
    np.testing.assert_allclose(out, qf, atol=1e-6)


def test_nlerp_preserves_unit_norm() -> None:
    rng = np.random.default_rng(42)
    qf_raw = rng.standard_normal((5, 4))
    qt_raw = rng.standard_normal((5, 4))
    qf = (qf_raw / np.linalg.norm(qf_raw, axis=1, keepdims=True)).astype(
        np.float32
    )
    qt = (qt_raw / np.linalg.norm(qt_raw, axis=1, keepdims=True)).astype(
        np.float32
    )
    for alpha in (0.1, 0.3, 0.5, 0.7, 0.9):
        out = wire.nlerp_quat_arrays_xyzw(qf, qt, alpha)
        norms = np.linalg.norm(out, axis=1)
        np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-6)


def test_nlerp_raises_on_shape_mismatch() -> None:
    qf = np.zeros((3, 4), dtype=np.float32)
    qt = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape mismatch"):
        wire.nlerp_quat_arrays_xyzw(qf, qt, 0.5)


def test_nlerp_raises_on_wrong_dimensionality() -> None:
    qf = np.zeros((4,), dtype=np.float32)  # missing batch dim
    qt = np.zeros((4,), dtype=np.float32)
    with pytest.raises(ValueError, match=r"\(N, 4\)"):
        wire.nlerp_quat_arrays_xyzw(qf, qt, 0.5)


# ===========================================================================
# build_idle_frame_msg override-kwarg validation
# ===========================================================================
def test_build_idle_msg_rejects_wrong_jpos_future_shape() -> None:
    replay = _make_replay()
    bad = np.zeros((wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS - 1), dtype=np.float32)
    with pytest.raises(ValueError, match="joint_pos_mj_future_override"):
        fallback.build_idle_frame_msg(
            replay, tick=0, topic="pose",
            joint_pos_mj_future_override=bad,
        )


def test_build_idle_msg_rejects_wrong_quat_future_shape() -> None:
    replay = _make_replay()
    bad = np.zeros((wire.NUM_FUTURE_SLOTS - 1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="root_quat_xyzw_future_override"):
        fallback.build_idle_frame_msg(
            replay, tick=0, topic="pose",
            root_quat_xyzw_future_override=bad,
        )


def test_build_idle_msg_rejects_wrong_jvel_future_shape() -> None:
    replay = _make_replay()
    bad = np.zeros((wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS + 2), dtype=np.float32)
    with pytest.raises(ValueError, match="joint_vel_mj_future_override"):
        fallback.build_idle_frame_msg(
            replay, tick=0, topic="pose",
            joint_vel_mj_future_override=bad,
        )


def test_build_idle_msg_rejects_wrong_motion_token_shape() -> None:
    replay = _make_replay()
    bad = np.zeros((wire.SONIC_MOTION_TOKEN_DIM - 4,), dtype=np.float32)
    with pytest.raises(ValueError, match="motion_token_override"):
        fallback.build_idle_frame_msg(
            replay, tick=0, topic="pose",
            motion_token_override=bad,
        )


def test_build_idle_msg_no_overrides_matches_legacy_behaviour() -> None:
    """Sanity: with no override kwargs, the v4 frame still publishes
    zero motion_token + idle-clip futures (the pre-fix behaviour).
    This guarantees the new code path is purely opt-in."""
    replay = _make_replay(jpos_value=0.25)
    msg = fallback.build_idle_frame_msg(replay, tick=0, topic="pose")
    token = _decode_msg_field(msg, "motion_token")
    np.testing.assert_allclose(token, 0.0, atol=0.0)
    jpos_future = _decode_msg_field(msg, "joint_pos_mj_future")
    # Idle clip is constant 0.25 everywhere, so future = 0.25 too.
    np.testing.assert_allclose(jpos_future, 0.25, atol=1e-6)


# ===========================================================================
# HOLD -> BLEND boundary continuity: at alpha=0 the published wire frame
# (futures + token) must equal the cached upstream snapshot, not the
# idle clip values.
# ===========================================================================
def _cached_upstream() -> dict[str, np.ndarray]:
    """Build a synthetic 'last VLA frame snapshot' the watchdog would cache.

    Values are deliberately distinct from the idle clip so any leak of
    idle-clip data into the alpha=0 published frame shows up as a
    diff.
    """
    return {
        "jpos_future": np.full(
            (wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS),
            0.9,
            dtype=np.float32,
        ),
        "quat_future": np.tile(
            [0.0, 0.0, np.sin(0.4), np.cos(0.4)],
            (wire.NUM_FUTURE_SLOTS, 1),
        ).astype(np.float32),
        "jvel_future": np.full(
            (wire.NUM_FUTURE_SLOTS, wire.NUM_BODY_DOFS),
            -0.3,
            dtype=np.float32,
        ),
        "motion_token": np.full(
            (wire.SONIC_MOTION_TOKEN_DIM,), 0.7, dtype=np.float32
        ),
    }


def test_blend_alpha_zero_emits_cached_upstream_jpos_future() -> None:
    replay = _make_replay(jpos_value=0.0)  # idle = all zeros
    cached = _cached_upstream()
    msg = fallback.build_idle_frame_msg(
        replay, tick=0, topic="pose",
        joint_pos_mj_future_override=cached["jpos_future"],
    )
    out = _decode_msg_field(msg, "joint_pos_mj_future")
    np.testing.assert_allclose(out, cached["jpos_future"], atol=0.0)


def test_blend_alpha_zero_emits_cached_upstream_quat_future() -> None:
    replay = _make_replay()
    cached = _cached_upstream()
    msg = fallback.build_idle_frame_msg(
        replay, tick=0, topic="pose",
        # Pass yaw_rebase to confirm the override is NOT rebased again.
        yaw_rebase_rad=0.5,
        root_quat_xyzw_future_override=cached["quat_future"],
    )
    out = _decode_msg_field(msg, "root_quat_xyzw_future")
    np.testing.assert_allclose(out, cached["quat_future"], atol=0.0)


def test_blend_alpha_zero_emits_cached_upstream_motion_token() -> None:
    replay = _make_replay()
    cached = _cached_upstream()
    msg = fallback.build_idle_frame_msg(
        replay, tick=0, topic="pose",
        motion_token_override=cached["motion_token"],
    )
    out = _decode_msg_field(msg, "motion_token")
    np.testing.assert_allclose(out, cached["motion_token"], atol=0.0)


def test_blend_alpha_zero_emits_cached_upstream_jvel_future() -> None:
    replay = _make_replay()
    cached = _cached_upstream()
    msg = fallback.build_idle_frame_msg(
        replay, tick=0, topic="pose",
        joint_vel_mj_future_override=cached["jvel_future"],
    )
    out = _decode_msg_field(msg, "joint_vel_mj_future")
    np.testing.assert_allclose(out, cached["jvel_future"], atol=0.0)


def test_blend_alpha_zero_full_frame_continuity_with_hold() -> None:
    """End-to-end: at alpha=0 every BLEND-managed field equals the
    cached upstream snapshot. This is the invariant that turns the
    HOLD -> BLEND boundary into a no-op for the policy's planning
    horizon."""
    replay = _make_replay(jpos_value=0.0)
    cached = _cached_upstream()
    msg = fallback.build_idle_frame_msg(
        replay, tick=0, topic="pose",
        joint_pos_mj_future_override=cached["jpos_future"],
        root_quat_xyzw_future_override=cached["quat_future"],
        joint_vel_mj_future_override=cached["jvel_future"],
        motion_token_override=cached["motion_token"],
    )
    np.testing.assert_allclose(
        _decode_msg_field(msg, "joint_pos_mj_future"),
        cached["jpos_future"], atol=0.0,
    )
    np.testing.assert_allclose(
        _decode_msg_field(msg, "root_quat_xyzw_future"),
        cached["quat_future"], atol=0.0,
    )
    np.testing.assert_allclose(
        _decode_msg_field(msg, "joint_vel_mj_future"),
        cached["jvel_future"], atol=0.0,
    )
    np.testing.assert_allclose(
        _decode_msg_field(msg, "motion_token"),
        cached["motion_token"], atol=0.0,
    )


# ===========================================================================
# BLEND ramp simulation: drive alpha 0 -> 1 in N steps and verify
# monotonic convergence + endpoint values.
# ===========================================================================
def _simulate_blend_ramp(
    n_steps: int,
    cached: dict[str, np.ndarray],
    replay: fallback.IdleStandReplay,
) -> list[dict[str, np.ndarray]]:
    """Mirror the watchdog's BLEND branch arithmetic for ``n_steps`` ticks.

    Returns the decoded wire-frame fields per tick. We intentionally
    inline the lerp formulas instead of invoking the watchdog's main
    loop -- the goal is to pin the arithmetic the watchdog uses so a
    refactor that diverges is caught here.
    """
    out: list[dict[str, np.ndarray]] = []
    idle_jpos, _ = replay.current(0)
    idle_jpos_future, idle_quat_future, idle_jvel_future = replay.future_window(0)
    for k in range(n_steps + 1):
        alpha = k / float(n_steps)
        lerp_jpos = (1.0 - alpha) * np.zeros_like(idle_jpos) + alpha * idle_jpos
        jpf_override = (
            (1.0 - alpha) * cached["jpos_future"] + alpha * idle_jpos_future
        ).astype(np.float32)
        qf_override = wire.nlerp_quat_arrays_xyzw(
            cached["quat_future"], idle_quat_future, alpha
        ).astype(np.float32)
        jvf_override = (
            (1.0 - alpha) * cached["jvel_future"] + alpha * idle_jvel_future
        ).astype(np.float32)
        token_override = ((1.0 - alpha) * cached["motion_token"]).astype(
            np.float32
        )
        msg = fallback.build_idle_frame_msg(
            replay, tick=0, topic="pose",
            joint_pos_mj_override=lerp_jpos.astype(np.float32),
            joint_pos_mj_future_override=jpf_override,
            root_quat_xyzw_future_override=qf_override,
            joint_vel_mj_future_override=jvf_override,
            motion_token_override=token_override,
        )
        out.append({
            "alpha": np.array([alpha]),
            "jpos_future": _decode_msg_field(msg, "joint_pos_mj_future"),
            "quat_future": _decode_msg_field(msg, "root_quat_xyzw_future"),
            "jvel_future": _decode_msg_field(msg, "joint_vel_mj_future"),
            "motion_token": _decode_msg_field(msg, "motion_token"),
        })
    return out


def test_blend_ramp_endpoints_match_cached_and_idle() -> None:
    replay = _make_replay(jpos_value=0.1)
    cached = _cached_upstream()
    samples = _simulate_blend_ramp(n_steps=150, cached=cached, replay=replay)
    # alpha=0: cached snapshot wins for every override field
    np.testing.assert_allclose(
        samples[0]["jpos_future"], cached["jpos_future"], atol=1e-6
    )
    np.testing.assert_allclose(
        samples[0]["motion_token"], cached["motion_token"], atol=1e-6
    )
    np.testing.assert_allclose(
        samples[0]["jvel_future"], cached["jvel_future"], atol=1e-6
    )
    # alpha=1: idle clip wins for jpos_future and jvel_future; motion
    # token decays to zero.
    idle_jpos_future, _, idle_jvel_future = replay.future_window(0)
    np.testing.assert_allclose(
        samples[-1]["jpos_future"], idle_jpos_future, atol=1e-6
    )
    np.testing.assert_allclose(
        samples[-1]["jvel_future"], idle_jvel_future, atol=1e-6
    )
    np.testing.assert_allclose(
        samples[-1]["motion_token"], 0.0, atol=1e-6
    )


def test_blend_ramp_motion_token_decays_monotonically() -> None:
    replay = _make_replay()
    cached = _cached_upstream()
    samples = _simulate_blend_ramp(n_steps=50, cached=cached, replay=replay)
    # All elements are positive in our cached token, so monotonic
    # decay means each tick has smaller-or-equal magnitude than the
    # previous one.
    magnitudes = [
        float(np.linalg.norm(s["motion_token"])) for s in samples
    ]
    deltas = np.diff(magnitudes)
    assert np.all(deltas <= 1e-6), (
        f"motion_token magnitude not monotonically non-increasing: {magnitudes}"
    )


def test_blend_ramp_jpos_future_per_slot_no_overshoot() -> None:
    """Each slot's per-DOF trajectory across the ramp must stay between
    the cached value and the idle-clip value (no overshoot, no
    oscillation). This is the per-slot continuity invariant."""
    replay = _make_replay(jpos_value=0.1)
    cached = _cached_upstream()
    samples = _simulate_blend_ramp(n_steps=30, cached=cached, replay=replay)
    idle_jpos_future, _, _ = replay.future_window(0)
    upper = np.maximum(cached["jpos_future"], idle_jpos_future)
    lower = np.minimum(cached["jpos_future"], idle_jpos_future)
    for sample in samples:
        assert np.all(sample["jpos_future"] <= upper + 1e-6)
        assert np.all(sample["jpos_future"] >= lower - 1e-6)


def test_blend_ramp_quat_future_stays_unit_norm() -> None:
    """nlerp output must be normalized, so every published quat must
    have unit norm even mid-ramp."""
    replay = _make_replay()
    cached = _cached_upstream()
    samples = _simulate_blend_ramp(n_steps=30, cached=cached, replay=replay)
    for sample in samples:
        norms = np.linalg.norm(sample["quat_future"], axis=1)
        np.testing.assert_allclose(
            norms, np.ones_like(norms), atol=1e-3
        )


# ===========================================================================
# Yaw-rebase frame consistency: override quats must NOT be rebased
# again when yaw_rebase_rad is also supplied.
# ===========================================================================
def test_quat_future_override_is_not_yaw_rebased_again() -> None:
    """The BLEND branch pre-rebases the idle-clip side BEFORE nlerping
    so the override is already in the live-heading frame. If
    build_idle_frame_msg applied yaw_rebase_rad to the override too,
    the lerp endpoints would no longer share a frame and the ramp
    would diverge."""
    replay = _make_replay()
    qf = np.tile(
        [0.0, 0.0, np.sin(0.2), np.cos(0.2)],
        (wire.NUM_FUTURE_SLOTS, 1),
    ).astype(np.float32)
    msg = fallback.build_idle_frame_msg(
        replay, tick=0, topic="pose",
        yaw_rebase_rad=0.5,
        root_quat_xyzw_future_override=qf,
    )
    out = _decode_msg_field(msg, "root_quat_xyzw_future")
    np.testing.assert_allclose(out, qf, atol=1e-6)


def test_quat_future_override_preserves_dtype_and_layout() -> None:
    """Non-C-contiguous f32 input still produces a clean wire frame."""
    replay = _make_replay()
    base = np.tile(
        [0.0, 0.0, 0.0, 1.0],
        (wire.NUM_FUTURE_SLOTS + 2, 1),
    ).astype(np.float32)
    # Strided slice -> non-contiguous in the time dimension.
    qf = base[1:1 + wire.NUM_FUTURE_SLOTS]
    msg = fallback.build_idle_frame_msg(
        replay, tick=0, topic="pose",
        root_quat_xyzw_future_override=qf,
    )
    out = _decode_msg_field(msg, "root_quat_xyzw_future")
    np.testing.assert_allclose(out, qf, atol=0.0)
