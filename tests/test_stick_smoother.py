"""Unit tests for :mod:`gear_sonic.utils.teleop.vr.stick_smoother`.

Covers:

* Default config is the no-op identity filter.
* First tick always passes through (no spurious slew clamp from a zero
  anchor).
* LPF step response matches the analytic ``1 - exp(-t/tau)`` envelope.
* Slew limit caps the per-tick output delta at ``slew * dt`` regardless
  of input magnitude.
* ``reset()`` clears state so the next tick is again a pass-through.
* Asymmetric release tau triggers on zero-crossing and on
  magnitude-collapse-toward-zero.
* ``max_dt_s`` caps the effective dt so a long pause doesn't burn the
  slew limit in one tick.
* Multi-channel independence: filtering ``fwd`` does not perturb
  ``side`` or ``yaw`` state.
"""

from __future__ import annotations

import math

import pytest

from gear_sonic.utils.teleop.vr.stick_smoother import (
    StickFilter,
    StickFilterConfig,
)


# ---------------------------------------------------------------------------
# Config & no-op behaviour
# ---------------------------------------------------------------------------


def test_default_config_is_noop() -> None:
    cfg = StickFilterConfig()
    assert cfg.is_noop()


def test_config_with_lpf_is_not_noop() -> None:
    cfg = StickFilterConfig(tau_lpf_fwd_s=0.2)
    assert not cfg.is_noop()


def test_config_with_slew_is_not_noop() -> None:
    cfg = StickFilterConfig(slew_max_fwd_per_s=5.0)
    assert not cfg.is_noop()


# ---------------------------------------------------------------------------
# First-tick pass-through
# ---------------------------------------------------------------------------


def test_first_tick_passes_through_when_filter_is_engaged() -> None:
    """Even with aggressive LPF + slew, the very first tick after init or
    reset must return the raw input verbatim. Otherwise the operator's
    first stick push after a mode change is dragged toward 0 by the
    zero anchor, which feels like a stuck stick."""
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=0.2, slew_max_fwd_per_s=1.0,
        tau_lpf_side_s=0.2, slew_max_side_per_s=1.0,
        tau_lpf_yaw_s=0.2, slew_max_yaw_per_s=1.0,
    )
    f = StickFilter(cfg)
    fwd, side, yaw = f.step(
        stick_fwd=0.8, stick_side=-0.4, stick_yaw=0.3, dt=0.02,
    )
    assert fwd == pytest.approx(0.8)
    assert side == pytest.approx(-0.4)
    assert yaw == pytest.approx(0.3)


def test_default_config_is_identity_for_any_input() -> None:
    """A no-op config must be the identity function on every tick, not
    just the first. Useful for ``QUEST3_STICK_LPF/SLEW`` unset case."""
    f = StickFilter(StickFilterConfig())
    for i, x in enumerate([0.0, 0.5, -0.7, 1.0, 0.0, 0.3]):
        fwd, side, yaw = f.step(
            stick_fwd=x, stick_side=x * 0.5, stick_yaw=-x, dt=0.02,
        )
        assert fwd == pytest.approx(x), f"tick {i}: fwd"
        assert side == pytest.approx(x * 0.5), f"tick {i}: side"
        assert yaw == pytest.approx(-x), f"tick {i}: yaw"


# ---------------------------------------------------------------------------
# LPF analytic envelope
# ---------------------------------------------------------------------------


def test_lpf_step_response_matches_analytic_envelope() -> None:
    """For a step input ``x = 1.0`` after the first-tick seed, the LPF
    output at time ``t`` (measured from the second tick) should follow
    ``y(t) = y0 + (1.0 - y0) * (1 - exp(-t/tau))``.

    We seed with ``x = 0`` so ``y0 = 0``, then drive ``x = 1.0`` for
    many ticks and check the envelope at three time points.
    """
    tau = 0.2
    dt = 0.02  # 50 Hz
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=tau,
        slew_max_fwd_per_s=math.inf,
    )
    f = StickFilter(cfg)
    # Seed at zero.
    f.step(stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0, dt=dt)
    # Drive step.
    actual = []
    for _ in range(40):  # 40 * 20 ms = 0.8 s
        fwd, _, _ = f.step(
            stick_fwd=1.0, stick_side=0.0, stick_yaw=0.0, dt=dt,
        )
        actual.append(fwd)
    # Check at three time points (in ticks-since-step).
    for k in (1, 5, 10, 20):
        t_since_step = k * dt
        expected = 1.0 - math.exp(-t_since_step / tau)
        # 1e-9 absolute tolerance; the math is exact for a first-order
        # discrete pole with this alpha definition.
        assert actual[k - 1] == pytest.approx(expected, abs=1e-9), (
            f"tick {k}: expected {expected:.6f}, got {actual[k - 1]:.6f}"
        )


# ---------------------------------------------------------------------------
# Slew clamp
# ---------------------------------------------------------------------------


def test_slew_caps_per_tick_delta() -> None:
    """With ``slew = 5.0`` and ``dt = 0.02`` the per-tick delta is capped
    at 0.1. A step from 0 to 1 must therefore take >= 10 ticks (0.2 s)
    regardless of LPF setting."""
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=0.0,  # LPF off so slew is the only shaper
        slew_max_fwd_per_s=5.0,
    )
    f = StickFilter(cfg)
    # Seed at zero.
    f.step(stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0, dt=0.02)
    last = 0.0
    for tick in range(15):
        fwd, _, _ = f.step(
            stick_fwd=1.0, stick_side=0.0, stick_yaw=0.0, dt=0.02,
        )
        # Every per-tick delta must be <= 0.1 + epsilon.
        assert abs(fwd - last) <= 0.1 + 1e-9, (
            f"tick {tick}: delta {fwd - last:.6f} exceeds slew cap 0.1"
        )
        last = fwd
    # By tick 10 we should have reached 1.0 exactly (linear ramp).
    assert last == pytest.approx(1.0, abs=1e-9)


def test_slew_symmetric_on_negative_step() -> None:
    """Slew limit applies symmetrically; a step from +1 to -1 must take
    20 ticks at slew=5, dt=0.02 (delta capped at 0.1 per tick)."""
    cfg = StickFilterConfig(tau_lpf_fwd_s=0.0, slew_max_fwd_per_s=5.0)
    f = StickFilter(cfg)
    f.step(stick_fwd=1.0, stick_side=0.0, stick_yaw=0.0, dt=0.02)
    last = 1.0
    for _ in range(25):
        fwd, _, _ = f.step(
            stick_fwd=-1.0, stick_side=0.0, stick_yaw=0.0, dt=0.02,
        )
        assert abs(fwd - last) <= 0.1 + 1e-9
        last = fwd
    assert last == pytest.approx(-1.0, abs=1e-9)


# ---------------------------------------------------------------------------
# reset() returns the filter to first-tick state
# ---------------------------------------------------------------------------


def test_reset_clears_state() -> None:
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=0.2, slew_max_fwd_per_s=5.0,
    )
    f = StickFilter(cfg)
    # Drive the filter so it has non-trivial state.
    for _ in range(10):
        f.step(stick_fwd=0.7, stick_side=0.0, stick_yaw=0.0, dt=0.02)
    f.reset()
    # Next tick must be a pass-through again, NOT a slow ramp from 0.
    fwd, _, _ = f.step(stick_fwd=0.4, stick_side=0.0, stick_yaw=0.0, dt=0.02)
    assert fwd == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Asymmetric release tau
# ---------------------------------------------------------------------------


def test_release_tau_triggers_on_zero_crossing() -> None:
    """When ``return_to_zero_tau_fwd_s = 0.5`` (slower than the engaged
    ``tau_lpf_fwd_s = 0.05``), a release should drain noticeably more
    slowly than an engage with the same magnitude step."""
    engaged_tau = 0.05
    release_tau = 0.5
    dt = 0.02
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=engaged_tau,
        slew_max_fwd_per_s=math.inf,
        return_to_zero_tau_fwd_s=release_tau,
    )

    # Engage: 0 -> 1.0 step. After 5 ticks the LPF should be at
    # 1 - exp(-5*0.02/0.05) ~= 0.865.
    f = StickFilter(cfg)
    f.step(stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0, dt=dt)
    for _ in range(5):
        engaged_y, _, _ = f.step(
            stick_fwd=1.0, stick_side=0.0, stick_yaw=0.0, dt=dt,
        )
    assert engaged_y == pytest.approx(1.0 - math.exp(-5 * dt / engaged_tau), abs=1e-9)

    # Now release: drive raw -> 0. The filter should switch to release
    # tau (0.5 s) so the output drains MUCH slower than an engage step.
    for _ in range(5):
        release_y, _, _ = f.step(
            stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0, dt=dt,
        )
    # If release tau == engaged tau, release_y would be engaged_y *
    # exp(-5*0.02/0.05) ~= 0.135 * engaged_y. With release tau 10x
    # slower, release_y should be much closer to engaged_y.
    naive_engaged_drain = engaged_y * math.exp(-5 * dt / engaged_tau)
    assert release_y > naive_engaged_drain * 3.0, (
        f"release_y={release_y:.4f} drained too fast; expected slower "
        f"than naive engaged-tau drain {naive_engaged_drain:.4f}"
    )


def test_release_tau_disabled_by_default() -> None:
    """When ``return_to_zero_tau_fwd_s`` is None, engage and release use
    the same tau (the engaged value). This is the legacy / safe default
    for operators who don't want asymmetric behaviour."""
    tau = 0.1
    dt = 0.02
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=tau,
        slew_max_fwd_per_s=math.inf,
        return_to_zero_tau_fwd_s=None,
    )
    f = StickFilter(cfg)
    # Engage from 0 to 1, then release back to 0; the trajectories
    # should be mirror images (same magnitude residual after equal
    # number of ticks).
    f.step(stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0, dt=dt)
    for _ in range(5):
        engaged_y, _, _ = f.step(
            stick_fwd=1.0, stick_side=0.0, stick_yaw=0.0, dt=dt,
        )
    for _ in range(5):
        release_y, _, _ = f.step(
            stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0, dt=dt,
        )
    # Residual on release should equal engaged_y * exp(-5*dt/tau) within
    # the numerical-LPF discretisation. (engaged_y itself is
    # 1 - exp(-5*dt/tau); residual is that times exp(-5*dt/tau).)
    expected = engaged_y * math.exp(-5 * dt / tau)
    assert release_y == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# max_dt clamp
# ---------------------------------------------------------------------------


def test_max_dt_caps_effective_dt_so_slew_holds() -> None:
    """A long pause (e.g., debugger break or scheduler hiccup) should NOT
    let the slew limiter swing the output across its full range in one
    tick. The filter clamps ``dt`` to ``max_dt_s`` (default 50 ms = 2.5
    ticks @ 50 Hz) so a 1 s gap still respects the slew cap."""
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=0.0,
        slew_max_fwd_per_s=5.0,
        max_dt_s=0.05,
    )
    f = StickFilter(cfg)
    f.step(stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0, dt=0.02)
    # Hand the filter a 1.0-second dt with a step input.
    fwd, _, _ = f.step(stick_fwd=1.0, stick_side=0.0, stick_yaw=0.0, dt=1.0)
    # Effective dt = 0.05; slew cap = 5.0 * 0.05 = 0.25. fwd MUST not
    # exceed 0.25 even though the raw input is 1.0.
    assert fwd == pytest.approx(0.25, abs=1e-9)


# ---------------------------------------------------------------------------
# Multi-channel independence
# ---------------------------------------------------------------------------


def test_channels_are_independent() -> None:
    """Filtering one channel must not perturb the others' state.

    Drive ``fwd`` aggressively while ``side`` and ``yaw`` stay at zero,
    then check the latter two are still bit-exact zero after the run.
    """
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=0.2, slew_max_fwd_per_s=2.0,
        tau_lpf_side_s=0.2, slew_max_side_per_s=2.0,
        tau_lpf_yaw_s=0.2, slew_max_yaw_per_s=2.0,
    )
    f = StickFilter(cfg)
    for _ in range(30):
        _, side, yaw = f.step(
            stick_fwd=1.0, stick_side=0.0, stick_yaw=0.0, dt=0.02,
        )
        assert side == 0.0
        assert yaw == 0.0


def test_per_channel_taus_picked_correctly() -> None:
    """Each channel reads its own tau, not the fwd-channel default. We
    set fwd LPF tight, side LPF loose, yaw LPF off, and verify the
    convergence rates differ accordingly under the same step input."""
    cfg = StickFilterConfig(
        tau_lpf_fwd_s=0.05,
        tau_lpf_side_s=0.50,
        tau_lpf_yaw_s=0.0,
    )
    f = StickFilter(cfg)
    f.step(stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0, dt=0.02)
    for _ in range(5):
        fwd, side, yaw = f.step(
            stick_fwd=1.0, stick_side=1.0, stick_yaw=1.0, dt=0.02,
        )
    # fwd converges fast (tau=0.05): after 5 ticks (0.1 s = 2*tau),
    # should be at 1 - exp(-2) ~= 0.865.
    assert fwd == pytest.approx(1.0 - math.exp(-0.1 / 0.05), abs=1e-9)
    # side converges slowly (tau=0.5): after 5 ticks (0.1 s = 0.2*tau),
    # should be at ~0.181.
    assert side == pytest.approx(1.0 - math.exp(-0.1 / 0.5), abs=1e-9)
    # yaw has no LPF: full step on tick 2 already.
    assert yaw == pytest.approx(1.0, abs=1e-9)
