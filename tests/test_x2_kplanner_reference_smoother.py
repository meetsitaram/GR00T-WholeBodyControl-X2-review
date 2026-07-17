"""Unit tests for ``_ReferenceStepSmoother`` in ``x2_kplanner``.

The smoother sits on the kplanner's publisher output and applies a
half-cosine ramp whenever the per-tick lower-body reference delta
exceeds a configurable trigger. It is intentionally **step-detection
driven**, not FSM-driven -- nothing in here imports ``PlannerState`` or
otherwise assumes any state machine.

Why the smoother exists is documented at length in the class docstring;
these tests pin down the observable invariants:

* ``shape='off'`` is a byte-equivalent passthrough (single-flag revert).
* The default joint mask (``lower_body``) NEVER modifies arm or head
  channels, even mid-ramp. This is the manipulation-safety invariant
  the operator explicitly requested before approving the rollout.
* The detection threshold cleanly separates the multi-degree jumps a
  stick push/release creates from the per-tick neural-buffer motion
  under steady walking.
* The halfcos shape has zero velocity at both ramp endpoints (the
  property that actually kills the audible click; ``linear`` is exposed
  for A/B comparison and is verified to have a non-zero endpoint dq/dt
  here so the choice of default has a quantitative justification).
* The ramp ends on the *live* target, not the source-snapshot target,
  so the hand-off at ``t=T`` doesn't introduce a second click.
* A ramp in flight does not restart on subsequent steps; the smoother
  is one-shot per detected edge.

Lightweight: no torch / no ZMQ / no checkpoints. Pure numpy + smoother
class import.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gear_sonic.scripts.x2_kplanner as kp  # noqa: E402
from gear_sonic.scripts.x2_kplanner import (  # noqa: E402
    _DEFAULT_REF_SMOOTHER_JOINTS,
    _DEFAULT_REF_SMOOTHER_MS,
    _DEFAULT_REF_SMOOTHER_SHAPE,
    _DEFAULT_REF_SMOOTHER_TRIGGER_RAD,
    _REF_SMOOTHER_JOINTS_PRESETS,
    _REF_SMOOTHER_SHAPES,
    _ReferenceStepSmoother,
)


NUM_DOFS = 31

# Canonical lower-body / upper-body splits, mirrored from the smoother's
# preset table so the tests fail loudly if the layout drifts.
LOWER_BODY_IDX = _REF_SMOOTHER_JOINTS_PRESETS["lower_body"]   # [0..14]
LEGS_ONLY_IDX = _REF_SMOOTHER_JOINTS_PRESETS["legs_only"]     # [0..11]
UPPER_BODY_IDX = np.array(
    [i for i in range(NUM_DOFS) if i not in set(LOWER_BODY_IDX.tolist())],
    dtype=np.int64,
)


def _seed_smoother(s: _ReferenceStepSmoother, q0: np.ndarray) -> None:
    """Drive one passthrough tick at ``t=0`` so the smoother has a
    last-published snapshot to compare against. Tests that want a
    detectable step on tick 2 onward call this first."""
    s.update(q0, t_now=0.0)


# ---------------------------------------------------------------------------
# Sanity: defaults + shape vocabulary
# ---------------------------------------------------------------------------


def test_default_constants_match_policy() -> None:
    """Pins the 2026-05-31 defaults so a drift here is loud, not silent."""
    assert _DEFAULT_REF_SMOOTHER_MS == 300.0
    assert _DEFAULT_REF_SMOOTHER_TRIGGER_RAD == 0.05
    # Default flipped to "off" when the 30->50 Hz output resampling +
    # 8-frame cross-fade blend landed: the seam blend now handles replan
    # discontinuities at the source, so the reactive smoother is opt-in.
    assert _DEFAULT_REF_SMOOTHER_SHAPE == "off"
    assert _DEFAULT_REF_SMOOTHER_JOINTS == "lower_body"
    assert _REF_SMOOTHER_SHAPES == ("halfcos", "linear", "off")
    # Layout invariants: lower_body = 15 DoFs (legs 0-11 + waist 12-14),
    # legs_only = 12 DoFs (legs only).
    assert LOWER_BODY_IDX.tolist() == list(range(0, 15))
    assert LEGS_ONLY_IDX.tolist() == list(range(0, 12))
    assert _REF_SMOOTHER_JOINTS_PRESETS["all"].tolist() == list(range(0, 31))


# ---------------------------------------------------------------------------
# (1) Steady walking: no step => passthrough, no ramp
# ---------------------------------------------------------------------------


def test_no_step_passthrough() -> None:
    """50 ticks of ~smooth gait deltas (~0.01 rad / tick on hip pitch)
    must not arm a single ramp; output must equal input every tick."""
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3,
        trigger_rad=0.05,
        shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    rng = np.random.default_rng(42)
    q = np.zeros(NUM_DOFS, dtype=np.float32)
    dt = 1.0 / 50.0
    for tick in range(50):
        # Smooth ~5 Hz oscillation on the lower body, sub-trigger amplitude.
        q[LOWER_BODY_IDX] = 0.05 * math.sin(2 * math.pi * 1.0 * tick * dt)
        out = s.update(q, t_now=tick * dt)
        np.testing.assert_array_equal(out, q)
        assert not s._ramp_active


# ---------------------------------------------------------------------------
# (2) Step above trigger arms a ramp; alpha=0 on the arming tick
# ---------------------------------------------------------------------------


def test_step_above_trigger_starts_ramp() -> None:
    """A 0.15 rad step on a leg joint must arm a ramp; the very tick
    the ramp arms must return the previous published value (alpha=0)
    so the PD law sees no step at the ramp's onset."""
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)

    q_step = q0.copy()
    q_step[3] = 0.15  # left_knee_joint (clearly above trigger)
    out = s.update(q_step, t_now=0.02)

    assert s._ramp_active
    # alpha(0) = 0 -> blended channels output equals source (q0).
    np.testing.assert_allclose(out[LOWER_BODY_IDX], q0[LOWER_BODY_IDX])
    # Upper body always passes through verbatim.
    np.testing.assert_array_equal(out[UPPER_BODY_IDX], q_step[UPPER_BODY_IDX])


def test_step_below_trigger_passthrough() -> None:
    """A 0.04 rad step (below the 0.05 default) must NOT arm a ramp."""
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)

    q_step = q0.copy()
    q_step[3] = 0.04
    out = s.update(q_step, t_now=0.02)

    assert not s._ramp_active
    np.testing.assert_array_equal(out, q_step)


# ---------------------------------------------------------------------------
# (3) Halfcos shape correctness
# ---------------------------------------------------------------------------


def test_halfcos_endpoint_values() -> None:
    """blend(0)=0, blend(T)=1, blend(T/2)=0.5 -- the math contract."""
    T = 0.3
    s = _ReferenceStepSmoother(
        ramp_duration_s=T, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    q1 = q0.copy(); q1[3] = 0.5  # large step on leg
    _seed_smoother(s, q0)

    # Tick the smoother at t=0.02 (arm) then walk through to t=T+1tick.
    out0 = s.update(q1, t_now=0.02)         # alpha = halfcos((t=0)/T) = 0
    np.testing.assert_allclose(out0[3], 0.0, atol=1e-7)

    out_mid = s.update(q1, t_now=0.02 + T / 2.0)  # alpha = 0.5
    np.testing.assert_allclose(out_mid[3], 0.5 * 0.5, atol=1e-6)

    out_end = s.update(q1, t_now=0.02 + T)        # alpha = 1
    np.testing.assert_allclose(out_end[3], 0.5, atol=1e-6)
    assert not s._ramp_active  # ramp completes at t >= T


def test_halfcos_endpoint_derivatives_are_zero() -> None:
    """C^1 invariant: numerical dq/dt at t=0 and t=T is small relative
    to the peak at t=T/2.

    Halfcos is ``0.5 * (1 - cos(pi*x))`` whose analytic derivative is
    ``(pi/2) * sin(pi*x)``, zero at x=0 and x=1. A forward finite
    difference over a finite dt picks up the local *quadratic*
    curvature -- for halfcos near x=0, ``alpha(dt) ~= 0.25 * (pi*dt/T)^2``
    so the forward slope ~= 0.25*pi^2*dt/T^2. The point of this test
    is not to assert "exactly zero" but to verify the endpoint slope
    is many orders of magnitude smaller than the mid-ramp peak (so
    the PD law sees essentially no torque step at the ramp boundaries).
    """
    T = 0.3
    expected_peak = math.pi / (2.0 * T)  # ~ 5.236 rad/s per unit step

    s = _ReferenceStepSmoother(
        ramp_duration_s=T, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    dt = 1e-4

    # Endpoint slopes (forward at t=0, backward at t=T).
    slope_start = (s._alpha(dt) - s._alpha(0.0)) / dt
    slope_end = (s._alpha(T) - s._alpha(T - dt)) / dt

    # Both endpoints must be < 1 % of the analytic peak slope. With
    # dt=1e-4 / T=0.3 the actual ratio is ~5e-4 ; the 1 % gate is a
    # comfortable margin.
    assert slope_start < 0.01 * expected_peak
    assert slope_end < 0.01 * expected_peak

    # Peak at t=T/2 must match analytic value within finite-difference
    # error (centred difference, 2nd-order accurate).
    a_mm = s._alpha(T / 2.0 - dt)
    a_mp = s._alpha(T / 2.0 + dt)
    measured = (a_mp - a_mm) / (2.0 * dt)
    assert abs(measured - expected_peak) < 1e-2


def test_linear_shape_has_velocity_step() -> None:
    """Sanity check that the linear shape *does* inject a velocity step
    at t=0 (which is why halfcos is the default). The slope at t=0+ for
    linear is 1/T per unit step -- non-zero, in contrast to halfcos's 0."""
    T = 0.3
    s = _ReferenceStepSmoother(
        ramp_duration_s=T, trigger_rad=0.05, shape="linear",
        blend_indices=LOWER_BODY_IDX,
    )
    dt = 1e-4
    a0 = s._alpha(0.0)
    a1 = s._alpha(dt)
    slope = (a1 - a0) / dt
    expected = 1.0 / T
    # Slope should be at least 90 % of the analytic value (numerical
    # differentiation has finite-difference error, but order of magnitude
    # should match -- anything near 0 means halfcos snuck in).
    assert slope > 0.9 * expected


# ---------------------------------------------------------------------------
# (4) Ramp follows live target (handoff doesn't snap)
# ---------------------------------------------------------------------------


def test_ramp_follows_live_target() -> None:
    """During the ramp, the target keeps moving (neural buffer evolving).
    The smoother must follow the LIVE target each tick so the t=T
    output is the current target, not a stale snapshot."""
    T = 0.3
    s = _ReferenceStepSmoother(
        ramp_duration_s=T, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)

    # Tick the smoother every 20 ms; target ramps from 0.20 -> 0.50 rad
    # linearly on joint 3 during the 300 ms window.
    dt = 0.02
    n_ticks = int(round(T / dt)) + 1
    last_target_value = 0.0
    for k in range(1, n_ticks + 1):
        target = q0.copy()
        target[3] = 0.20 + 0.30 * (k / n_ticks)
        last_target_value = float(target[3])
        out = s.update(target, t_now=k * dt)
    # On the final tick (t >= T) the smoother must have completed and
    # emit exactly the live target value.
    assert not s._ramp_active
    np.testing.assert_allclose(out[3], last_target_value, atol=1e-5)


# ---------------------------------------------------------------------------
# (5) shape='off' is byte-equivalent passthrough (single-flag revert)
# ---------------------------------------------------------------------------


def test_off_shape_passthrough() -> None:
    """shape='off' must never modify any channel, even when a step that
    would normally arm a ramp arrives. This is the safety-net revert."""
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="off",
        blend_indices=LOWER_BODY_IDX,
    )
    assert not s.enabled
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)

    q_step = q0.copy()
    q_step[3] = 0.50
    for k in range(20):
        out = s.update(q_step, t_now=k * 0.02)
        np.testing.assert_array_equal(out, q_step)
    assert not s._ramp_active


def test_zero_duration_passthrough() -> None:
    """ramp_duration_s=0 must also be passthrough (not divide by zero)."""
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.0, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    assert not s.enabled
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    s.update(q0, t_now=0.0)
    q_step = q0.copy(); q_step[3] = 0.5
    out = s.update(q_step, t_now=0.02)
    np.testing.assert_array_equal(out, q_step)


# ---------------------------------------------------------------------------
# (6) Upper-body passthrough invariant (manipulation safety)
# ---------------------------------------------------------------------------


def test_upper_body_passthrough_default() -> None:
    """The critical guarantee: with the default 'lower_body' mask, the
    smoother NEVER modifies indices [15..30] -- not in steady state, not
    during step detection, not mid-ramp. Manipulation tasks driven by
    the same body_pose stream rely on this.
    """
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    q0[UPPER_BODY_IDX] = np.linspace(0.1, 1.6, UPPER_BODY_IDX.size, dtype=np.float32)
    _seed_smoother(s, q0)

    # Big step on the lower body to arm a ramp.
    q1 = q0.copy()
    q1[3] = 0.25
    q1[10] = -0.30
    # Also wiggle the upper body to make sure it tracks the LIVE target
    # bit-exactly even while a leg ramp is in flight.
    upper_targets = np.linspace(0.2, 1.7, UPPER_BODY_IDX.size, dtype=np.float32)
    q1[UPPER_BODY_IDX] = upper_targets

    for k in range(1, 25):  # well past the 15-tick ramp window @ 50 Hz
        target = q1.copy()
        # Simulate a continuously-moving upper body during the ramp.
        target[UPPER_BODY_IDX] = (
            upper_targets
            + 0.01 * math.sin(2 * math.pi * 0.5 * k * 0.02)
        )
        out = s.update(target, t_now=k * 0.02)
        np.testing.assert_array_equal(out[UPPER_BODY_IDX], target[UPPER_BODY_IDX])


def test_upper_body_step_does_not_arm_ramp_with_default_mask() -> None:
    """A step on an arm joint must NOT arm a ramp under the default
    lower_body mask -- arms are excluded from step detection too."""
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)

    q_arm_step = q0.copy()
    q_arm_step[18] = 0.50  # left_elbow_joint (well above trigger)
    out = s.update(q_arm_step, t_now=0.02)
    assert not s._ramp_active
    np.testing.assert_array_equal(out, q_arm_step)


# ---------------------------------------------------------------------------
# (7) Joint mask variants
# ---------------------------------------------------------------------------


def test_joints_legs_only_skips_waist() -> None:
    """With legs_only, a step on the waist (indices 12-14) must NOT
    arm a ramp; legs do."""
    s_legs = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=LEGS_ONLY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s_legs, q0)
    q_waist = q0.copy(); q_waist[12] = 0.5
    s_legs.update(q_waist, t_now=0.02)
    assert not s_legs._ramp_active

    s_legs2 = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=LEGS_ONLY_IDX,
    )
    _seed_smoother(s_legs2, q0)
    q_leg = q0.copy(); q_leg[3] = 0.5
    s_legs2.update(q_leg, t_now=0.02)
    assert s_legs2._ramp_active


def test_joints_all_blends_every_dof() -> None:
    """With joints='all', a step on an arm joint DOES arm a ramp (debug
    mode for parity testing with the legacy formulation)."""
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=_REF_SMOOTHER_JOINTS_PRESETS["all"],
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)
    q_arm = q0.copy(); q_arm[18] = 0.5
    s.update(q_arm, t_now=0.02)
    assert s._ramp_active


# ---------------------------------------------------------------------------
# (8) One-shot semantics: an active ramp doesn't restart on further steps
# ---------------------------------------------------------------------------


def test_does_not_restart_during_active_ramp() -> None:
    """Once a ramp is in flight, further large target deltas must NOT
    reset ramp_start_t. The ramp completes its planned duration; the
    next ramp can only fire after this one ends."""
    T = 0.3
    s = _ReferenceStepSmoother(
        ramp_duration_s=T, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)

    q1 = q0.copy(); q1[3] = 0.30
    s.update(q1, t_now=0.02)  # arms ramp at t=0.02
    assert s._ramp_active
    original_start = s._ramp_start_t

    # Mid-ramp, the target lurches again. The smoother must NOT re-arm.
    q2 = q0.copy(); q2[3] = 0.80
    s.update(q2, t_now=0.02 + T / 4.0)
    assert s._ramp_active
    assert s._ramp_start_t == original_start

    # After the original ramp completes, a fresh step CAN arm again.
    s.update(q2, t_now=0.02 + T + 0.001)
    # First post-ramp tick deactivates the ramp; the immediate next
    # tick re-detects if there's still a big delta vs the just-emitted
    # frame.
    assert not s._ramp_active


# ---------------------------------------------------------------------------
# (9) Reset clears state
# ---------------------------------------------------------------------------


def test_reset_clears_state() -> None:
    """After reset(), an in-flight ramp is cancelled and the next tick
    is a fresh passthrough that re-seeds the cache."""
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)
    q1 = q0.copy(); q1[3] = 0.5
    s.update(q1, t_now=0.02)
    assert s._ramp_active

    s.reset()
    assert not s._ramp_active
    assert s._last_published_q is None
    assert s._source_q is None

    # First update post-reset: passthrough, no ramp armed.
    out = s.update(q1, t_now=10.0)
    np.testing.assert_array_equal(out, q1)
    assert not s._ramp_active


# ---------------------------------------------------------------------------
# (10) Torque-step reduction at high kp (the actual physics motivation)
# ---------------------------------------------------------------------------


def test_lower_body_torque_step_reduction() -> None:
    """With hip kp=99 Nm/rad and a 0.15 rad step on a leg joint, the
    peak commanded torque rate on the wire must drop by at least the
    analytic halfcos-vs-passthrough ratio.

    Analytic ratio: passthrough delivers ``step / dt`` rad/s in one
    tick; halfcos peaks at ``(pi / (2T)) * step`` rad/s at t=T/2. So
    the reduction factor is

        baseline / smoothed = (step / dt) / ((pi / (2T)) * step)
                            = 2T / (pi * dt)

    At T=300 ms, dt=20 ms (50 Hz publisher) this is
    ``2*0.3 / (pi*0.02) ~= 9.55x``. We assert ">= 8x" to leave
    floating-point + finite-difference margin while still pinning the
    qualitative win. The actual measured reduction prints into the
    assert message so a regression on the shape's math is loud.
    """
    kp_hip = 99.0
    step_rad = 0.15
    dt = 1.0 / 50.0
    T = 0.30

    baseline_peak_torque_rate = kp_hip * (step_rad / dt)

    s = _ReferenceStepSmoother(
        ramp_duration_s=T, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    q0 = np.zeros(NUM_DOFS, dtype=np.float32)
    _seed_smoother(s, q0)
    q1 = q0.copy(); q1[3] = step_rad

    prev = float(q0[3])
    peak_rate = 0.0
    for k in range(1, int(T / dt) + 5):
        out = s.update(q1, t_now=k * dt)
        rate = abs(float(out[3]) - prev) / dt
        peak_rate = max(peak_rate, rate)
        prev = float(out[3])

    smoothed_peak_torque_rate = kp_hip * peak_rate
    ratio = baseline_peak_torque_rate / smoothed_peak_torque_rate

    assert ratio >= 8.0, (
        f"baseline {baseline_peak_torque_rate:.1f} N*m/s vs smoothed "
        f"{smoothed_peak_torque_rate:.1f} N*m/s ({ratio:.2f}x reduction; "
        f"expected >= 8x at T=300 ms, analytic = 2T/(pi*dt) ~= 9.55x)"
    )


# ---------------------------------------------------------------------------
# (11) Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_unknown_shape() -> None:
    with pytest.raises(ValueError):
        _ReferenceStepSmoother(
            ramp_duration_s=0.3, trigger_rad=0.05, shape="ema",
        )


def test_constructor_rejects_empty_blend_indices() -> None:
    with pytest.raises(ValueError):
        _ReferenceStepSmoother(
            ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
            blend_indices=np.array([], dtype=np.int64),
        )


def test_constructor_rejects_out_of_range_indices() -> None:
    with pytest.raises(ValueError):
        _ReferenceStepSmoother(
            ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
            blend_indices=np.array([0, 31], dtype=np.int64),
        )


def test_update_rejects_wrong_shape() -> None:
    s = _ReferenceStepSmoother(
        ramp_duration_s=0.3, trigger_rad=0.05, shape="halfcos",
        blend_indices=LOWER_BODY_IDX,
    )
    with pytest.raises(ValueError):
        s.update(np.zeros(30, dtype=np.float32), t_now=0.0)


# ---------------------------------------------------------------------------
# (12) Run-level integration: CLI defaults are honoured
# ---------------------------------------------------------------------------


def test_parse_args_default_ref_smoother_knobs() -> None:
    """Pin the CLI default surface so an accidental rename / removal
    doesn't ship silently to the wrapper script."""
    args = kp._parse_args([])
    assert args.ref_smoother_ms == _DEFAULT_REF_SMOOTHER_MS
    assert args.ref_smoother_trigger_rad == _DEFAULT_REF_SMOOTHER_TRIGGER_RAD
    assert args.ref_smoother_shape == _DEFAULT_REF_SMOOTHER_SHAPE
    assert args.ref_smoother_joints == _DEFAULT_REF_SMOOTHER_JOINTS


def test_parse_args_accepts_all_shape_choices() -> None:
    for shape in _REF_SMOOTHER_SHAPES:
        args = kp._parse_args(["--ref-smoother-shape", shape])
        assert args.ref_smoother_shape == shape


def test_parse_args_accepts_all_joint_presets() -> None:
    for joints in _REF_SMOOTHER_JOINTS_PRESETS:
        args = kp._parse_args(["--ref-smoother-joints", joints])
        assert args.ref_smoother_joints == joints
