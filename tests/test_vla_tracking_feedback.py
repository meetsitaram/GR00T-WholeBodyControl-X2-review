"""Tests for the closed-loop tracking feedback on the VLA bridge wire.

2026-06-10 follow-up 11: ``_apply_tracking_feedback`` + the per-joint
clamp variant ``_clamp_vector_step_per_joint``. Together they replace
the bridge's static scalar wire step cap with a per-joint cap that's
modulated by real proprio feedback from ``x2_debug``, eliminating the
open-loop sensitivity to inference jitter / battery sag / motor temp
drift that drove the 2026-06-10 PM oscillation incident.

These tests pin the feedback law (position backoff + velocity cap),
the fallback paths (no proprio / stale proprio / disabled), and the
joint-mask behaviour (only arms throttled; legs/waist/head pass
through).
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.scripts.live_vla_publish_motion_token import (
    _apply_tracking_feedback,
    _ARM_JOINT_INDICES,
    _clamp_vector_step,
    _clamp_vector_step_per_joint,
    NUM_BODY_DOFS,
)


# --------------------------------------------------------------------
# _clamp_vector_step_per_joint: per-joint element-wise clamp
# --------------------------------------------------------------------


def _make_pose(value: float = 0.0) -> np.ndarray:
    return np.full(NUM_BODY_DOFS, value, dtype=np.float32)


class TestClampVectorStepPerJoint:
    """Pin the per-joint clamp semantics. Unlike the scalar variant
    (which preserves direction by scaling the WHOLE delta), this
    variant clamps each joint INDEPENDENTLY so tracking feedback
    can throttle only the joints that are actually lagging without
    dragging unrelated DOFs down with them.
    """

    def test_prev_none_passes_target_through_unchanged(self) -> None:
        """No anchor available -> first publish ever -> the clamp
        is a no-op (matches scalar variant's first-call contract)."""
        tgt = _make_pose(0.5)
        out = _clamp_vector_step_per_joint(
            tgt, None, np.zeros(NUM_BODY_DOFS, dtype=np.float32)
        )
        np.testing.assert_array_equal(out, tgt)

    def test_uniform_cap_matches_per_element_scalar_behaviour(self) -> None:
        """When the per-joint cap is a UNIFORM array, the per-joint
        clamp behaves as a strict element-wise clamp -- i.e., each
        joint's |delta| is clipped to the cap independently. (This
        is intentionally DIFFERENT from the scalar variant, which
        scales the entire delta vector; the per-joint variant trades
        direction preservation for per-DOF responsiveness.)
        """
        prev = _make_pose(0.0)
        # Target: joint 0 wants +0.10 (above cap), joint 1 wants
        # +0.02 (below cap), joint 2 wants -0.08 (above cap, negative).
        tgt = prev.copy()
        tgt[0] = 0.10
        tgt[1] = 0.02
        tgt[2] = -0.08
        cap = np.full(NUM_BODY_DOFS, 0.05, dtype=np.float32)
        out = _clamp_vector_step_per_joint(tgt, prev, cap)
        # Joint 0 capped at +0.05; joint 1 unchanged (below cap);
        # joint 2 capped at -0.05.
        np.testing.assert_allclose(out[0], 0.05, atol=1e-6)
        np.testing.assert_allclose(out[1], 0.02, atol=1e-6)
        np.testing.assert_allclose(out[2], -0.05, atol=1e-6)
        # All other joints unchanged.
        np.testing.assert_array_equal(out[3:], prev[3:])

    def test_zero_cap_freezes_joint(self) -> None:
        """A cap of 0 on a joint means "freeze that joint"; delta
        from prev is forced to zero regardless of target. Used by
        the feedback law when tracking error exceeds the hard
        threshold (actuator clearly saturating)."""
        prev = _make_pose(0.0)
        tgt = prev.copy()
        tgt[15] = 0.20  # Big push on L_sh_p
        tgt[16] = 0.20  # Big push on L_sh_r
        cap = np.full(NUM_BODY_DOFS, 0.05, dtype=np.float32)
        cap[15] = 0.0  # Freeze L_sh_p
        # L_sh_r still gets its normal cap.
        out = _clamp_vector_step_per_joint(tgt, prev, cap)
        assert out[15] == 0.0, "frozen joint must not move"
        np.testing.assert_allclose(out[16], 0.05, atol=1e-6)

    def test_negative_cap_treated_as_no_cap(self) -> None:
        """STRICTLY negative cap entries mean "no per-joint cap on
        this DOF -- pass through". Used as the sentinel for joints
        outside the tracking-feedback mask. (Exactly 0 means FREEZE,
        per ``test_zero_cap_freezes_joint`` above; the two cases are
        intentionally split because the feedback law emits 0 to
        freeze a saturating joint.)
        """
        prev = _make_pose(0.0)
        tgt = prev.copy()
        tgt[5] = 5.0  # Huge delta on a leg joint
        cap = np.full(NUM_BODY_DOFS, 0.05, dtype=np.float32)
        cap[5] = -1.0  # "No cap" sentinel
        out = _clamp_vector_step_per_joint(tgt, prev, cap)
        assert out[5] == pytest.approx(5.0)

    def test_shape_mismatch_raises(self) -> None:
        """Wrong-length cap array must raise loudly -- silently
        truncating or broadcasting could mis-align which joint is
        being throttled."""
        prev = _make_pose(0.0)
        tgt = prev.copy()
        with pytest.raises(ValueError, match="max_step_per_joint shape"):
            _clamp_vector_step_per_joint(
                tgt, prev, np.zeros(5, dtype=np.float32)
            )


# --------------------------------------------------------------------
# _apply_tracking_feedback: the feedback law
# --------------------------------------------------------------------


def _law_args(**overrides):
    """Convenient defaults for the helper kwargs. Matches CLI defaults
    so tests track the operator-facing behaviour."""
    base = dict(
        base_max_step=0.07,
        soft_rad=0.15,
        hard_rad=0.40,
        vel_margin=1.5,
        vel_floor_rad_tick=0.01,
        dt_s=0.02,  # 50 Hz
    )
    base.update(overrides)
    return base


class TestApplyTrackingFeedbackFallback:
    """Fallback paths: the helper must never throw and must always
    return a usable per-joint cap array. Wrong inputs degrade to
    "use the scalar cap" so the bridge's wire path is preserved."""

    def test_prev_wire_none_falls_back_to_scalar(self) -> None:
        """First call ever (no prev wire yet): every joint gets the
        scalar base step. Caller's clamp will also no-op because
        prev is None, so the wire is unaffected -- but we still
        return a consistent cap array so downstream code paths
        don't have to special-case ``None``."""
        tgt = _make_pose(0.5)
        cap, throttle = _apply_tracking_feedback(
            tgt, None,
            np.zeros(NUM_BODY_DOFS, dtype=np.float32),
            np.zeros(NUM_BODY_DOFS, dtype=np.float32),
            **_law_args(),
        )
        assert cap.shape == (NUM_BODY_DOFS,)
        np.testing.assert_array_equal(cap, np.full(NUM_BODY_DOFS, 0.07, dtype=np.float32))
        assert throttle == 0

    def test_measured_q_none_falls_back_to_scalar(self) -> None:
        """No proprio position snapshot: fall back to scalar cap;
        zero throttle reported."""
        tgt = _make_pose(0.5)
        prev = _make_pose(0.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev, None,
            np.zeros(NUM_BODY_DOFS, dtype=np.float32),
            **_law_args(),
        )
        np.testing.assert_array_equal(cap, np.full(NUM_BODY_DOFS, 0.07, dtype=np.float32))
        assert throttle == 0

    def test_measured_dq_none_falls_back_to_scalar(self) -> None:
        """No proprio velocity snapshot: fall back to scalar cap."""
        tgt = _make_pose(0.5)
        prev = _make_pose(0.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev,
            np.zeros(NUM_BODY_DOFS, dtype=np.float32),
            None,
            **_law_args(),
        )
        np.testing.assert_array_equal(cap, np.full(NUM_BODY_DOFS, 0.07, dtype=np.float32))
        assert throttle == 0

    def test_base_max_step_zero_freezes_everything(self) -> None:
        """Scalar cap of 0 means "wire step cap disabled = freeze";
        the helper respects that by returning all-zero caps. (This
        mirrors the scalar clamp's behaviour when ``max_step<=0``.)"""
        tgt = _make_pose(0.5)
        prev = _make_pose(0.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev,
            _make_pose(0.0), _make_pose(0.0),
            **_law_args(base_max_step=0.0),
        )
        np.testing.assert_array_equal(cap, np.zeros(NUM_BODY_DOFS, dtype=np.float32))
        # Throttle count is 0 by definition (no joints fell BELOW
        # 0.5 * base because base is zero).
        assert throttle == 0

    def test_shape_mismatch_falls_back_to_scalar(self) -> None:
        """If x2_debug ever publishes a body_q with a different
        DOF count (e.g., schema bug), we MUST NOT silently misalign
        -- fall back to the scalar cap and log the issue downstream."""
        tgt = _make_pose(0.5)
        prev = _make_pose(0.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev,
            np.zeros(5, dtype=np.float32),  # wrong shape
            np.zeros(NUM_BODY_DOFS, dtype=np.float32),
            **_law_args(),
        )
        np.testing.assert_array_equal(cap, np.full(NUM_BODY_DOFS, 0.07, dtype=np.float32))
        assert throttle == 0


class TestApplyTrackingFeedbackPositionBackoff:
    """The position backoff law: caps shrink linearly from base to 0
    as |tgt - meas| sweeps from soft_rad to hard_rad."""

    def test_zero_tracking_error_full_step(self) -> None:
        """If the actuator is exactly at the target, no backoff."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        # Measured == target on all arm joints (other joints
        # auto-default to base step from the mask filter).
        meas_q = tgt.copy()
        meas_dq = _make_pose(0.5)  # Some velocity to avoid floor binding
        cap, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        for j in _ARM_JOINT_INDICES:
            # Velocity cap with margin 1.5 * 0.5 * 0.02 = 0.015 > base 0.07? No, 0.015 < 0.07
            # So vel cap dominates here.
            assert cap[j] == pytest.approx(0.015, abs=1e-6), (
                f"joint {j}: expected vel-cap-dominant 0.015 (1.5 * 0.5 rad/s * 0.02 s), "
                f"got {cap[j]}"
            )
        # Throttle count: vel_cap (0.015) < 0.5 * base (0.035), so
        # every arm joint counts as throttled.
        assert throttle == len(_ARM_JOINT_INDICES)

    def test_error_below_soft_no_position_backoff(self) -> None:
        """Errors smaller than soft_rad don't trigger position
        backoff. With a high enough velocity to keep the velocity
        cap above base, we should see the full base step."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = _make_pose(0.05)  # 0.05 rad lag (well below 0.15 soft)
        # Velocity high enough that vel_cap = 1.5 * 5.0 * 0.02 = 0.15
        # >> base 0.07, so vel cap doesn't bind.
        meas_dq = _make_pose(5.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.07, abs=1e-6), (
                f"joint {j}: expected full base step 0.07 (no backoff "
                f"below soft, no vel binding), got {cap[j]}"
            )
        assert throttle == 0

    def test_error_at_soft_starts_backoff(self) -> None:
        """Exactly at the soft threshold the cap is still full
        (the law uses ``err <= soft -> scale=1``)."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = _make_pose(0.15)  # exactly at soft
        meas_dq = _make_pose(5.0)  # vel cap doesn't bind
        cap, _ = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.07, abs=1e-6)

    def test_error_between_soft_and_hard_linear_backoff(self) -> None:
        """At err = (soft + hard) / 2 the cap should be exactly
        base * 0.5 (halfway between full and frozen)."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        # soft=0.15, hard=0.40, midpoint=0.275
        meas_q = _make_pose(0.275)
        meas_dq = _make_pose(5.0)  # vel cap doesn't bind
        cap, _ = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.035, abs=1e-6), (
                f"joint {j}: expected base*0.5=0.035 at midpoint, got {cap[j]}"
            )

    def test_error_above_hard_freezes_joint(self) -> None:
        """At err >= hard the cap drops to 0 (joint frozen until
        actuator catches up)."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = _make_pose(0.5)  # well above 0.40 hard
        meas_dq = _make_pose(5.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.0, abs=1e-6), (
                f"joint {j}: expected frozen (cap=0) above hard, got {cap[j]}"
            )
        assert throttle == len(_ARM_JOINT_INDICES)


class TestApplyTrackingFeedbackVelocityCap:
    """The velocity cap law: ``min(pos_cap, max(vel_floor, vel_margin * |dq| * dt))``."""

    def test_velocity_floor_holds_when_actuator_at_rest(self) -> None:
        """When measured velocity is 0, vel_cap = max(vel_floor,
        1.5 * 0 * 0.02) = vel_floor. Lets the wire start from
        rest at a known minimum rate."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = tgt.copy()
        meas_dq = _make_pose(0.0)  # at rest
        cap, _ = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq,
            **_law_args(vel_floor_rad_tick=0.01),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.01, abs=1e-6), (
                f"joint {j}: expected vel_floor 0.01 at rest, got {cap[j]}"
            )

    def test_velocity_floor_zero_freezes_at_rest(self) -> None:
        """vel_floor=0 means "require non-zero measured velocity
        before allowing any wire motion" (motion-only mode)."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = tgt.copy()
        meas_dq = _make_pose(0.0)
        cap, _ = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq,
            **_law_args(vel_floor_rad_tick=0.0),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.0, abs=1e-6)

    def test_high_velocity_lifts_cap_above_floor(self) -> None:
        """At high measured velocity, vel_cap = margin * |dq| * dt
        > floor. With margin=1.5, |dq|=2 rad/s, dt=0.02s: cap = 0.06.
        Below base 0.07 so vel cap binds."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = tgt.copy()
        meas_dq = _make_pose(2.0)  # 2 rad/s
        cap, _ = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.06, abs=1e-6)

    def test_position_cap_binds_when_smaller_than_velocity_cap(self) -> None:
        """Both caps active: min() picks the smaller one. Here we
        engineer position to give pos_cap=0.035 and velocity to
        give vel_cap=0.06; pos_cap should bind."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = _make_pose(0.275)  # midpoint -> pos_cap = base * 0.5 = 0.035
        meas_dq = _make_pose(2.0)   # vel_cap = 0.06
        cap, _ = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.035, abs=1e-6)

    def test_velocity_uses_absolute_value(self) -> None:
        """Negative velocity still produces a positive cap (sign
        of motion doesn't matter for the throttle decision)."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = tgt.copy()
        meas_dq = _make_pose(-2.0)  # negative velocity
        cap, _ = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == pytest.approx(0.06, abs=1e-6)


class TestApplyTrackingFeedbackJointMask:
    """The joint mask: only arm joints (MJ 15..28) get per-joint
    feedback. Legs / waist / head pass through with the base step.

    Why: legs and waist have SONIC's balance loop; head has the
    lock-head-straight pin. Tracking feedback would fight those
    deploy-side controllers and produce stuck legs or a head that
    can't recover from a load disturbance.
    """

    def test_only_arm_joints_throttled(self) -> None:
        """Same lag on every joint; only arm slots should drop
        below base."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        # Lag every joint into the backoff zone.
        meas_q = _make_pose(0.5)
        meas_dq = _make_pose(0.0)  # vel_floor wins on arms
        cap, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        # Arms: vel_floor 0.01 vs pos_cap 0.0 (above hard); min wins -> 0.0
        for j in _ARM_JOINT_INDICES:
            assert cap[j] == 0.0, f"arm joint {j} should be frozen"
        # Non-arm joints get base step regardless of how big the lag is.
        non_arm = [
            i for i in range(NUM_BODY_DOFS) if i not in _ARM_JOINT_INDICES
        ]
        for j in non_arm:
            assert cap[j] == pytest.approx(0.07, abs=1e-6), (
                f"non-arm joint {j} should pass through with base step, "
                f"got {cap[j]}"
            )
        assert throttle == len(_ARM_JOINT_INDICES)

    def test_arm_indices_constant_is_14_dof(self) -> None:
        """Pin the constant -- if the X2 ever gets new DOFs the
        joint mask needs updating + this test will catch it."""
        assert len(_ARM_JOINT_INDICES) == 14
        # First left-arm slot is MJ 15 (after legs 0..11 and waist
        # 12..14), last right-arm slot is MJ 28 (before head 29..30).
        assert _ARM_JOINT_INDICES[0] == 15
        assert _ARM_JOINT_INDICES[-1] == 28

    def test_custom_joint_indices(self) -> None:
        """Allow passing a different mask (useful for tests + future
        whole-body feedback experiments)."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = _make_pose(0.5)
        meas_dq = _make_pose(0.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq,
            joint_indices=(0, 1),  # legs only
            **_law_args(),
        )
        # Only joints 0 and 1 should be throttled now.
        for j in range(NUM_BODY_DOFS):
            if j in (0, 1):
                assert cap[j] == 0.0
            else:
                assert cap[j] == pytest.approx(0.07, abs=1e-6)
        assert throttle == 2


class TestApplyTrackingFeedbackThrottleCounter:
    """The throttle counter is the operator-visible telemetry
    knob. ``tf_throttle=N/14`` in the pub-tick log tells the
    operator how many arm joints feedback is actively protecting."""

    def test_zero_when_no_joints_throttled(self) -> None:
        """All arm joints at full step -> throttle = 0."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = tgt.copy()  # zero error
        meas_dq = _make_pose(5.0)  # high vel, no vel binding
        _, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        assert throttle == 0

    def test_counts_only_joints_below_half_base(self) -> None:
        """Throttle counter increments when a joint's cap drops
        below 0.5 * base (= 0.035 with default base=0.07).
        Engineer a partial mix: half the arm at base, half throttled."""
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = tgt.copy()
        meas_dq = _make_pose(5.0)  # vel cap doesn't bind
        # Push half the arm joints into the backoff zone (pos_cap < 0.035).
        first_half = _ARM_JOINT_INDICES[:7]
        for j in first_half:
            meas_q[j] = 0.35  # close to hard 0.40 -> pos_cap = base * (0.40-0.35)/(0.40-0.15) = base * 0.2 = 0.014
        _, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        assert throttle == 7

    def test_full_throttle_when_all_arm_frozen(self) -> None:
        tgt = _make_pose(0.0)
        prev = _make_pose(0.0)
        meas_q = _make_pose(0.5)  # all above hard
        meas_dq = _make_pose(5.0)
        _, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        assert throttle == len(_ARM_JOINT_INDICES)


# --------------------------------------------------------------------
# Integration sketch: feedback law + per-joint clamp end-to-end
# --------------------------------------------------------------------


class TestFeedbackEndToEnd:
    """The feedback law produces per-joint caps; the clamp consumes
    them. Verify the composed pipeline matches what the bridge hot
    loop will see."""

    def test_clean_motion_passes_through(self) -> None:
        """When the actuator is tracking well, the wire moves at
        the requested per-tick delta (capped by base step)."""
        prev = _make_pose(0.0)
        tgt = prev.copy()
        # Modest motion on L_sh_p (MJ 15): 0.04 rad delta.
        tgt[15] = 0.04
        meas_q = prev.copy()  # exact tracking
        meas_dq = _make_pose(5.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        out = _clamp_vector_step_per_joint(tgt, prev, cap)
        # L_sh_p should hit the requested 0.04 (under base 0.07 cap,
        # under vel cap 1.5*5*0.02=0.15).
        assert out[15] == pytest.approx(0.04, abs=1e-6)
        # Other joints unchanged.
        for j in range(NUM_BODY_DOFS):
            if j == 15:
                continue
            assert out[j] == pytest.approx(prev[j], abs=1e-6)
        assert throttle == 0

    def test_lagging_joint_frozen_others_pass(self) -> None:
        """One arm joint lagging severely -> that joint frozen,
        others continue at requested rate."""
        prev = _make_pose(0.0)
        tgt = prev.copy()
        # L_sh_p wants huge delta, R_sh_p wants normal delta.
        tgt[15] = 0.50  # delta way past hard
        tgt[22] = 0.04
        # L_sh_p is far behind target; R_sh_p is exactly tracking.
        meas_q = prev.copy()
        meas_q[15] = -0.05  # current pose 0, target 0.50, err = 0.55 (>> 0.40 hard)
        meas_dq = _make_pose(5.0)
        cap, throttle = _apply_tracking_feedback(
            tgt, prev, meas_q, meas_dq, **_law_args(),
        )
        out = _clamp_vector_step_per_joint(tgt, prev, cap)
        # L_sh_p frozen.
        assert out[15] == pytest.approx(0.0, abs=1e-6)
        # R_sh_p moves to its requested 0.04 (under base cap).
        assert out[22] == pytest.approx(0.04, abs=1e-6)
        # Throttle: L_sh_p capped at 0 (below half-base) -> 1.
        assert throttle == 1

    def test_repeated_application_converges(self) -> None:
        """Simulate 50 ticks of feedback where the per-tick TARGET
        is always (prev_wire + small_delta) -- i.e., the post-LPF
        chunk output, which is what the bridge actually passes in.
        The "sluggish actuator" lags by 0.3 rad behind the wire.

        Realistic operating point: ``target - measured = small_delta
        + lag``, so the tracking error stays in the BACKOFF zone (not
        the FREEZE zone) and the wire slows but still progresses.
        Verifies the feedback both PREVENTS runaway and ALLOWS
        legitimate motion to continue at the throttled rate."""
        prev = _make_pose(0.0)
        meas_dq = _make_pose(0.5)  # modest measured velocity
        wire = prev.copy()
        for _ in range(50):
            # Per-tick target: prev_wire + 0.02 rad (typical LPF-
            # smoothed chunk step toward a distant goal).
            tgt = wire.copy()
            tgt[15] = wire[15] + 0.02
            # Measured actuator lags wire by 0.3 rad (sluggish motor).
            meas_q = wire.copy()
            meas_q[15] = wire[15] - 0.3
            # Feedback law sees err = |tgt - meas_q| = |0.32| =
            # 0.32, between soft (0.15) and hard (0.40), so
            # pos_cap = 0.07 * (0.40 - 0.32) / 0.25 = 0.07 * 0.32 =
            # 0.0224. Vel cap = 1.5 * 0.5 * 0.02 = 0.015. min wins,
            # so wire moves 0.015 rad/tick.
            cap, _ = _apply_tracking_feedback(
                tgt, wire, meas_q, meas_dq, **_law_args(),
            )
            wire = _clamp_vector_step_per_joint(tgt, wire, cap)
        # After 50 ticks at ~0.015 rad/tick: wire[15] ~= 0.75 rad.
        # Some sources of slack: per-tick target is +0.02, but the
        # clamp limits to 0.015, so it accumulates at that rate.
        assert wire[15] > 0.5, (
            f"feedback should allow legitimate motion to continue at "
            f"the throttled rate; wire[15] = {wire[15]}"
        )
        assert wire[15] < 1.0, (
            f"feedback should keep wire BELOW free-running rate "
            f"(50 ticks * 0.02 rad/tick = 1.0 rad); "
            f"wire[15] = {wire[15]}"
        )


# --------------------------------------------------------------------
# Regression pin: scalar clamp remains byte-identical when feedback
# disabled. (The bridge hot loop's gate ``if tracking_active`` falls
# through to ``_clamp_vector_step`` in the disabled case; this test
# pins that the helper FUNCTIONS produce different outputs when both
# are configured to clamp the same delta -- proving the gate change
# actually matters and isn't a no-op.)
# --------------------------------------------------------------------


class TestScalarVsPerJointDivergence:
    def test_scalar_scales_direction_per_joint_clamps_elementwise(self) -> None:
        """When all joints want to move but one wants to move FAST:
        - scalar variant scales the WHOLE delta by max_step/peak
          (preserving the wire's per-tick direction).
        - per-joint variant clamps each independently (the fast
          joint slows, others continue at requested rate).
        """
        prev = _make_pose(0.0)
        tgt = prev.copy()
        # Joint 15 wants huge delta, joint 16 wants small delta.
        tgt[15] = 0.30
        tgt[16] = 0.04
        scalar_out = _clamp_vector_step(tgt, prev, 0.07)
        # Scalar: delta has peak 0.30; scale = 0.07/0.30 ~= 0.233.
        # So delta[15] -> 0.07, delta[16] -> 0.04 * 0.233 ~= 0.0093.
        assert scalar_out[15] == pytest.approx(0.07, abs=1e-4)
        assert scalar_out[16] == pytest.approx(0.0093, abs=1e-3)
        # Per-joint: each clamped independently to 0.07.
        cap = np.full(NUM_BODY_DOFS, 0.07, dtype=np.float32)
        per_out = _clamp_vector_step_per_joint(tgt, prev, cap)
        assert per_out[15] == pytest.approx(0.07, abs=1e-4)
        assert per_out[16] == pytest.approx(0.04, abs=1e-6)  # under cap, passes
        # Confirm divergence on joint 16: per-joint preserves
        # responsiveness on smaller-delta joints; scalar drags them
        # down. (The whole motivation for the per-joint variant.)
        assert per_out[16] != pytest.approx(scalar_out[16], abs=1e-3)
