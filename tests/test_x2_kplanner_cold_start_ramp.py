"""Unit tests for the x2_kplanner cold-start velocity ramp.

Pins the contract of :class:`_ColdStartVelocityRamp` -- the EWMA
velocity smoother that fires on every ``idle -> playing`` transition
to keep the model's implied 2.13 s target close to the robot's actual
position while the context buffer still holds 4 frames of static
stand pose. See the class docstring in ``x2_kplanner.py`` for the
full mechanism; these tests cover:

1. **First step after idle starts at zero**: on a fresh ramper /
   after ``reset_idle()`` the smoothed velocity ignores prior state
   and begins ramping from zero, regardless of what target is asked
   for.
2. **EWMA semantics**: each step moves ``alpha = dt / (tau + dt)`` of
   the way from the current smoothed value to the target.
3. **hip_h (channel 3) is forwarded verbatim** -- it's a posture
   target, not a velocity, and the model needs the correct walking
   pelvis height from frame 1.
4. **Convergence**: holding a constant target long enough drives the
   smoothed value arbitrarily close to it.
5. **tau == 0 disables smoothing**: the ramper passes the raw target
   through (matches pre-fix verbatim behaviour).
6. **reset_idle restores cold-start state** between PLAYING segments
   so a release-then-repush of the stick gets a fresh ramp.
7. **Negative / mixed-sign targets ramp symmetrically** -- backward
   walking + lateral side-stepping get the same treatment as
   forward walking.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "gear_sonic" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import x2_kplanner  # noqa: E402  -- module under test, after sys.path fix
from x2_kplanner import (  # noqa: E402
    _ColdStartVelocityRamp,
    _DEFAULT_COLD_START_RAMP_TAU_S,
    _HIP_HEIGHT_M,
)


# ---------------------------------------------------------------------------
# Constructor / default invariants
# ---------------------------------------------------------------------------


def test_default_tau_matches_module_constant():
    ramp = _ColdStartVelocityRamp()
    assert ramp.tau_s == pytest.approx(_DEFAULT_COLD_START_RAMP_TAU_S)
    assert ramp.tau_s > 0.0
    assert ramp.enabled is True


def test_explicit_tau_is_stored():
    ramp = _ColdStartVelocityRamp(tau_s=0.10)
    assert ramp.tau_s == pytest.approx(0.10)
    assert ramp.enabled is True


def test_zero_tau_disables_ramp():
    ramp = _ColdStartVelocityRamp(tau_s=0.0)
    assert ramp.enabled is False


def test_negative_tau_is_treated_as_disabled():
    # Defensive: ``tau_s > 0`` is the only gate; negative tau would
    # produce a non-physical EWMA. The class converts ``tau_s <= 0``
    # to "no smoothing" in ``step``.
    ramp = _ColdStartVelocityRamp(tau_s=-0.05)
    assert ramp.enabled is False


# ---------------------------------------------------------------------------
# Cold-start: first step ignores prior state, starts from zero
# ---------------------------------------------------------------------------


def test_first_step_after_construction_starts_from_zero():
    """A fresh ramper has ``last_was_idle = True``; the first ``step``
    must reset smoothed state to zero before applying the EWMA so the
    operator's first push doesn't slam from a stale value."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    target = (1.5, -0.3, 0.5, _HIP_HEIGHT_M)
    out = ramp.step(target, dt_s=0.20)
    yaw, vx, vz, hip = out
    # alpha = 0.20 / (0.20 + 0.20) = 0.5; smoothed starts at 0 each
    # cold start, so output is 0.5 * target on the velocity channels.
    assert yaw == pytest.approx(0.5 * 1.5)
    assert vx == pytest.approx(0.5 * -0.3)
    assert vz == pytest.approx(0.5 * 0.5)
    # hip_h is forwarded verbatim.
    assert hip == pytest.approx(_HIP_HEIGHT_M)


def test_subsequent_step_continues_from_smoothed_state():
    """Second step (no idle marker between) advances the existing
    smoothed value -- it does NOT zero out again."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    target = (0.0, 0.0, 0.5, _HIP_HEIGHT_M)
    first = ramp.step(target, dt_s=0.20)
    # After tick 1: vz = 0.5 * 0.5 = 0.25
    assert first[2] == pytest.approx(0.25)
    second = ramp.step(target, dt_s=0.20)
    # After tick 2: vz = 0.25 + 0.5 * (0.5 - 0.25) = 0.375
    assert second[2] == pytest.approx(0.375)


def test_reset_idle_zeroes_state_and_marks_idle():
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    ramp.step((0.0, 0.0, 0.5, _HIP_HEIGHT_M), dt_s=0.20)
    # Halfway up the ramp; assert state is non-zero now.
    second = ramp.step((0.0, 0.0, 0.5, _HIP_HEIGHT_M), dt_s=0.20)
    assert second[2] > 0.0
    # Reset clears smoothed state + marks the next step as cold.
    ramp.reset_idle()
    after_reset = ramp.step((0.0, 0.0, 0.5, _HIP_HEIGHT_M), dt_s=0.20)
    # Back to the cold-start ramp -- vz starts at 0 again, single-step
    # output is 0.5 * 0.5 = 0.25.
    assert after_reset[2] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# EWMA semantics: alpha and convergence
# ---------------------------------------------------------------------------


def test_alpha_matches_formula_for_various_tau_dt():
    """``alpha = dt / (tau + dt)`` is the EWMA discrete update; verify
    the closed-form against several (tau, dt) pairs."""
    cases = [
        (0.20, 0.20, 0.5),       # dt == tau -> half
        (0.10, 0.20, 2.0 / 3.0),  # dt > tau  -> more aggressive
        (0.40, 0.10, 0.2),       # dt < tau  -> gentler
        (0.30, 0.10, 0.25),
    ]
    for tau, dt, expected_alpha in cases:
        ramp = _ColdStartVelocityRamp(tau_s=tau)
        # Single cold-start step from 0 -> target hits exactly
        # alpha * target on the velocity channels.
        out = ramp.step((0.0, 0.0, 1.0, _HIP_HEIGHT_M), dt_s=dt)
        assert out[2] == pytest.approx(expected_alpha), (
            f"tau={tau} dt={dt}: expected alpha={expected_alpha}, got {out[2]}"
        )


def test_holding_constant_target_converges_to_target():
    """After enough ticks at constant target the smoothed value
    converges to within machine epsilon of the target."""
    ramp = _ColdStartVelocityRamp(tau_s=0.10)
    target = (0.5, 0.3, 0.4, _HIP_HEIGHT_M)
    # 50 ticks at dt=0.20 with tau=0.10 -> alpha ~ 0.667 per tick;
    # convergence is geometric in (1-alpha) ~ 0.333.
    for _ in range(50):
        out = ramp.step(target, dt_s=0.20)
    yaw, vx, vz, hip = out
    assert yaw == pytest.approx(target[0], abs=1e-9)
    assert vx == pytest.approx(target[1], abs=1e-9)
    assert vz == pytest.approx(target[2], abs=1e-9)
    assert hip == pytest.approx(target[3])


def test_step_count_to_95_percent_at_default_settings():
    """Sanity-check the docstring's published claim: at the default
    tau=0.20 s and a 200 ms replan period the ramp hits ~95% of the
    operator's target in about 3 replans."""
    ramp = _ColdStartVelocityRamp(tau_s=_DEFAULT_COLD_START_RAMP_TAU_S)
    target = (0.0, 0.0, 1.0, _HIP_HEIGHT_M)
    values = []
    for _ in range(5):
        values.append(ramp.step(target, dt_s=0.20)[2])
    # alpha = 0.5; cumulative reach is 0.5, 0.75, 0.875, 0.9375, 0.96875.
    # Replan 3 (1-indexed) -> 0.875 = 87.5%, replan 4 -> 93.75%, replan
    # 5 -> ~96.9%. The docstring claim is "~95% in ~3 replans", which
    # is right for tau=0.20 + dt=0.20 modulo the off-by-one between
    # "after 3 replans" and "the 3rd value". Pin the exact sequence
    # so future tuning of the default tau gets a visible regression.
    assert values[0] == pytest.approx(0.5)
    assert values[1] == pytest.approx(0.75)
    assert values[2] == pytest.approx(0.875)
    assert values[3] == pytest.approx(0.9375)
    assert values[4] == pytest.approx(0.96875)


# ---------------------------------------------------------------------------
# hip_h (channel 3) is NOT smoothed
# ---------------------------------------------------------------------------


def test_hip_h_is_forwarded_verbatim():
    """Channel 3 is a posture target (world-frame pelvis Y) the model
    consumes as ``implied_target_y``. The walking gait needs the
    correct hip height from frame 1 -- smoothing it would put the
    pelvis at an intermediate height the policy can't reach. The
    ramper MUST pass it through unchanged."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    out1 = ramp.step((0.0, 0.0, 0.5, 0.687), dt_s=0.20)
    out2 = ramp.step((0.0, 0.0, 0.5, 0.700), dt_s=0.20)
    out3 = ramp.step((0.0, 0.0, 0.5, 0.625), dt_s=0.20)
    assert out1[3] == pytest.approx(0.687)
    assert out2[3] == pytest.approx(0.700)
    assert out3[3] == pytest.approx(0.625)


# ---------------------------------------------------------------------------
# tau == 0 disables smoothing entirely
# ---------------------------------------------------------------------------


def test_tau_zero_passes_target_verbatim_on_every_step():
    """With ``tau_s = 0`` the ramper short-circuits the EWMA and
    returns the raw target -- this mode reproduces the pre-fix
    behaviour for regression testing."""
    ramp = _ColdStartVelocityRamp(tau_s=0.0)
    target = (1.5, -0.4, 0.5, _HIP_HEIGHT_M)
    # Cold start: tau=0 still resets smoothed state to zero, but then
    # immediately overwrites it with the target (the "no smoothing"
    # branch). The single-step output equals the raw target.
    out = ramp.step(target, dt_s=0.20)
    assert out == target
    # Subsequent steps still pass through.
    out2 = ramp.step((0.1, 0.2, 0.3, _HIP_HEIGHT_M), dt_s=0.20)
    assert out2 == (0.1, 0.2, 0.3, _HIP_HEIGHT_M)


def test_zero_dt_also_short_circuits():
    """Defensive: a degenerate ``dt_s = 0`` would mean alpha=0 and
    no progress -- the ramper instead treats it the same as
    ``tau_s = 0`` and snaps to the target. This matches the
    documented contract that ``tau_s <= 0 or dt_s <= 0`` bypasses
    the EWMA."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    target = (0.5, 0.5, 0.5, _HIP_HEIGHT_M)
    out = ramp.step(target, dt_s=0.0)
    # Bypass triggers -> smoothed snaps to target exactly.
    assert out == target


# ---------------------------------------------------------------------------
# Symmetric ramping for backward / lateral / yaw
# ---------------------------------------------------------------------------


def test_backward_walking_ramps_symmetrically():
    """Backward step (vel_z < 0) gets the same treatment as forward."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    out = ramp.step((0.0, 0.0, -0.35, _HIP_HEIGHT_M), dt_s=0.20)
    assert out[2] == pytest.approx(-0.35 * 0.5)


def test_lateral_side_step_ramps_symmetrically():
    """Side-step (vel_x != 0) also ramps from zero."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    out = ramp.step((0.0, 0.4, 0.0, _HIP_HEIGHT_M), dt_s=0.20)
    assert out[1] == pytest.approx(0.4 * 0.5)
    assert out[2] == pytest.approx(0.0)


def test_pure_turn_ramps_yaw_channel():
    """Pure in-place turn (yaw_rate != 0, vel = 0) ramps yaw rate
    too, so the model doesn't see a 1.5 rad/s step input from a
    static-head context."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    out = ramp.step((1.5, 0.0, 0.0, _HIP_HEIGHT_M), dt_s=0.20)
    assert out[0] == pytest.approx(1.5 * 0.5)


def test_mixed_intent_ramps_all_three_velocity_channels_in_parallel():
    """All three velocity channels are ramped independently with the
    same alpha (they share the EWMA filter state)."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    out = ramp.step((1.0, -0.5, 0.5, _HIP_HEIGHT_M), dt_s=0.20)
    alpha = 0.5  # 0.20 / (0.20 + 0.20)
    assert out[0] == pytest.approx(1.0 * alpha)
    assert out[1] == pytest.approx(-0.5 * alpha)
    assert out[2] == pytest.approx(0.5 * alpha)
    assert out[3] == pytest.approx(_HIP_HEIGHT_M)


# ---------------------------------------------------------------------------
# Realistic scenario: release-and-repush of the L-stick
# ---------------------------------------------------------------------------


def test_release_and_repush_resets_cold_start():
    """Real-world Quest 3 pattern: operator pushes stick, ramps up,
    briefly releases (kplanner state goes PLAYING -> IDLE_LOOP), then
    pushes again. The ``reset_idle()`` call between push windows MUST
    restart the ramp from zero -- otherwise the second push would
    skip the ramp and re-introduce the cold-start failure mode."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    target = (0.0, 0.0, 0.5, _HIP_HEIGHT_M)

    # First push: ramps up over a few ticks.
    for _ in range(10):
        ramp.step(target, dt_s=0.20)
    # Operator releases -- worker sees idle intent, calls reset_idle.
    ramp.reset_idle()
    # Second push: must start from zero again.
    out = ramp.step(target, dt_s=0.20)
    assert out[2] == pytest.approx(0.5 * 0.5)  # alpha=0.5, target=0.5


def test_idle_state_persists_across_multiple_reset_idle_calls():
    """``reset_idle()`` is idempotent -- the worker calls it on every
    idle tick (since it doesn't track time-in-idle separately), so
    multiple calls in a row must not break the ramp."""
    ramp = _ColdStartVelocityRamp(tau_s=0.20)
    for _ in range(5):
        ramp.reset_idle()
    out = ramp.step((0.0, 0.0, 0.5, _HIP_HEIGHT_M), dt_s=0.20)
    assert out[2] == pytest.approx(0.25)
