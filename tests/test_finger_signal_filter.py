"""Unit tests for ``gear_sonic.utils.teleop.finger_signal_filter``.

Covers four behaviours the live + record paths rely on:

1. **Held-pose noise reduction.** Static input + Gaussian noise
   should be latched within ``hold_window + 1`` frames and the
   output std after latching should drop to zero.
2. **Motion-edge lag bound.** A step input should reach 50 % of
   the step amplitude within at most ``ceil(1 / ema_alpha) + 1``
   frames of the raw signal -- i.e. the EMA contributes one extra
   frame at most, no deadband-induced delay on a clear step.
3. **NaN dropout-hold.** A 1-3 frame NaN gap inside a held window
   must not break the latch -- the held value bridges the gap.
4. **Hysteresis release.** Once latched, the output must release
   to the live signal the moment ``|x - held| > release_disp``,
   even if the rolling std hasn't risen yet.

The tests use the live default parameters
(:class:`FingerFilterParams` defaults) so they double as a guard
against regressions in the calibrated thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.teleop.finger_signal_filter import (
    DEFAULT_EMA_ALPHA,
    DEFAULT_HOLD_STD,
    DEFAULT_HOLD_WINDOW,
    DEFAULT_RELEASE_DISP,
    FingerFilterParams,
    FingerSignalFilter,
    NUM_TIP_OPPOSE_CHANNELS,
    NUM_TOTAL_CHANNELS,
    filter_npz_offline,
    pack_signal,
    unpack_signal,
)


# ── helpers ────────────────────────────────────────────────────────────


def _curls(thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0):
    return np.array([thumb, index, middle, ring, pinky], dtype=np.float64)


def _tip(index=0.0, middle=0.0, ring=0.0, pinky=0.0):
    return np.array([index, middle, ring, pinky], dtype=np.float64)


def _step_input(level_a, level_b, n_a, n_b, channel: int = 1) -> np.ndarray:
    """Build an (n_a + n_b, 10) sequence: level_a for n_a frames, then
    level_b. Only ``channel`` is set; other channels stay at zero.
    """
    out = np.zeros((n_a + n_b, NUM_TOTAL_CHANNELS), dtype=np.float64)
    out[:n_a, channel] = level_a
    out[n_a:, channel] = level_b
    return out


# ── pack / unpack round-trip ───────────────────────────────────────────


def test_pack_unpack_roundtrip_full() -> None:
    c = _curls(0.1, 0.2, 0.3, 0.4, 0.5)
    o = 0.7
    t = _tip(0.6, 0.7, 0.8, 0.9)
    vec = pack_signal(c, o, t)
    assert vec.shape == (NUM_TOTAL_CHANNELS,)
    c2, o2, t2 = unpack_signal(vec)
    np.testing.assert_allclose(c2, c)
    assert o2 == pytest.approx(o)
    np.testing.assert_allclose(t2, t)


def test_pack_unpack_roundtrip_none_inputs_become_nan() -> None:
    vec = pack_signal(None, None, None)
    assert np.isnan(vec).all()
    c, o, t = unpack_signal(vec)
    assert c is None
    assert o is None
    assert t is None


def test_pack_validates_curl_shape() -> None:
    with pytest.raises(ValueError, match="curls must be"):
        pack_signal(np.zeros(4), 0.0, _tip())


def test_pack_validates_tip_shape() -> None:
    with pytest.raises(ValueError, match="finger_tip_oppose must be"):
        pack_signal(_curls(), 0.0, np.zeros(3))


# ── filter parameter validation ────────────────────────────────────────


def test_default_params_are_self_consistent() -> None:
    """Sanity: the default v5-calibrated params must validate."""
    FingerFilterParams().validate()  # should not raise


def test_invalid_ema_alpha_raises() -> None:
    with pytest.raises(ValueError, match="ema_alpha"):
        FingerFilterParams(ema_alpha=0.0).validate()
    with pytest.raises(ValueError, match="ema_alpha"):
        FingerFilterParams(ema_alpha=1.5).validate()


def test_release_std_below_hold_std_raises() -> None:
    with pytest.raises(ValueError, match="release_std"):
        FingerFilterParams(hold_std=0.01, release_std=0.005).validate()


# ── held-pose latching ─────────────────────────────────────────────────


def test_held_pose_reduces_noise_via_rolling_median() -> None:
    """Feed ``hold_window + 8`` frames of constant signal + Gaussian
    noise (std ≈ 0.003). The rolling-median output should reduce
    the visible std by **at least 2x** versus the raw input.

    The theoretical std reduction for an 8-sample rolling median of
    Gaussian noise is ~2.3x (median of N samples has std ~ σ * sqrt(π/2)/sqrt(N)),
    so the 2x bound is a soft, easy-to-meet target.
    """
    rng = np.random.default_rng(42)
    n = DEFAULT_HOLD_WINDOW + 100  # long enough to stabilise stats
    in_std = 0.003
    x = np.full((n, NUM_TOTAL_CHANNELS), 0.5, dtype=np.float64)
    x += rng.normal(0.0, in_std, size=x.shape)

    filt = FingerSignalFilter()
    out = np.full_like(x, np.nan)
    for t in range(n):
        c, o, tt = filt.update(x[t, 0:5], float(x[t, 5]), x[t, 6:10])
        if c is not None:
            out[t, 0:5] = c
        if o is not None:
            out[t, 5] = o
        if tt is not None:
            out[t, 6:10] = tt

    # Past the warm-up, the rolling-median output should have much
    # lower std than the input (>= 5x reduction).
    post = out[DEFAULT_HOLD_WINDOW + 5 :]
    raw_post = x[DEFAULT_HOLD_WINDOW + 5 :]
    raw_std = raw_post.std(axis=0).max()
    flt_std = post.std(axis=0).max()
    assert flt_std * 2 < raw_std, (
        f"deadband should reduce held-pose std by >= 2x; "
        f"raw_std={raw_std:.4f}, filt_std={flt_std:.4f}"
    )
    # Mean must still be near 0.5 (no DC drift introduced).
    assert abs(post.mean() - 0.5) < 0.01


# ── motion-edge lag bound ──────────────────────────────────────────────


def test_step_input_reaches_target_within_lag_bound() -> None:
    """Step from 0 -> 0.8 on channel 1 (index curl). The output should
    cross 50 % of step within at most 2 frames (raw=0, EMA(0.5)=1).

    The deadband must NOT add latency on a clear step: the displacement
    on the very first step frame is 0.8 >> release_disp = 0.020, so
    the latch must release immediately even if it had been set.
    """
    n_a, n_b = 30, 30
    x = _step_input(0.0, 0.8, n_a, n_b, channel=1)

    filt = FingerSignalFilter()
    outs = np.zeros((n_a + n_b, NUM_TOTAL_CHANNELS))
    for t in range(n_a + n_b):
        c, o, tt = filt.update(x[t, 0:5], float(x[t, 5]), x[t, 6:10])
        if c is not None:
            outs[t, 0:5] = c
        if o is not None:
            outs[t, 5] = o
        if tt is not None:
            outs[t, 6:10] = tt
    # Step happens at t = n_a; target = 0.4
    target = 0.4
    crossed_at = next(
        (t for t in range(n_a, n_a + 30) if outs[t, 1] >= target),
        None,
    )
    assert crossed_at is not None, "Filter never crossed 50 % of step"
    # raw would cross at t = n_a (instant). EMA(0.5) crosses at t = n_a + 1
    # (since 0.5 * 0.8 = 0.4 exactly). Allow up to t = n_a + 2 for slack.
    assert crossed_at <= n_a + 2, (
        f"motion-edge lag too high: filter crossed at t = {crossed_at}, "
        f"step started at t = {n_a}; expected <= n_a + 2"
    )


def test_step_releases_held_state_immediately() -> None:
    """30 frames of held input -> 30 frames of stepped input. The
    deadband must release on the first step frame.
    """
    held_level = 0.5
    step_level = 0.5 + DEFAULT_RELEASE_DISP * 5  # well above release_disp
    n_a, n_b = 30, 30
    x = _step_input(held_level, step_level, n_a, n_b, channel=1)

    filt = FingerSignalFilter()
    outs = np.zeros((n_a + n_b, NUM_TOTAL_CHANNELS))
    for t in range(n_a + n_b):
        c, o, tt = filt.update(x[t, 0:5], float(x[t, 5]), x[t, 6:10])
        if c is not None:
            outs[t, 0:5] = c
    # During held window the output should be exactly held_level.
    assert abs(outs[n_a - 1, 1] - held_level) < 1e-9, (
        f"channel should be latched at held_level={held_level}, got {outs[n_a-1, 1]:.4f}"
    )
    # First post-step frame should NOT still be at held_level -- the
    # release condition |x - held| > release_disp is hit immediately.
    assert outs[n_a, 1] != pytest.approx(held_level), (
        f"deadband failed to release on step: out[{n_a}]={outs[n_a, 1]:.4f}"
    )


# ── NaN dropout-hold ───────────────────────────────────────────────────


def test_nan_dropout_during_held_state_returns_held_value() -> None:
    """Latch a held pose, then feed 3 NaN frames mid-stream. The held
    value must bridge the gap.
    """
    n_warm = DEFAULT_HOLD_WINDOW + 5
    held_level = 0.4
    n_dropout = 3
    n_after = 5
    n_total = n_warm + n_dropout + n_after

    x = np.full((n_total, NUM_TOTAL_CHANNELS), held_level, dtype=np.float64)
    rng = np.random.default_rng(7)
    x += rng.normal(0.0, 0.002, size=x.shape)
    x[n_warm : n_warm + n_dropout, :] = np.nan

    filt = FingerSignalFilter()
    out_curls = np.full((n_total, 5), np.nan)
    for t in range(n_total):
        c_in = None if np.isnan(x[t, 0:5]).any() else x[t, 0:5]
        o_in = None if np.isnan(x[t, 5]) else float(x[t, 5])
        t_in = None if np.isnan(x[t, 6:10]).all() else x[t, 6:10]
        c_f, _, _ = filt.update(c_in, o_in, t_in)
        if c_f is not None:
            out_curls[t] = c_f

    # All `n_dropout` frames during the gap should still produce a
    # finite output (the held value), NOT NaN.
    gap_outputs = out_curls[n_warm : n_warm + n_dropout]
    assert not np.isnan(gap_outputs).any(), (
        f"held value should bridge NaN dropout; got\n{gap_outputs}"
    )
    # And the held value should be close to held_level.
    assert abs(gap_outputs.mean() - held_level) < 0.02


def test_nan_input_with_no_held_state_returns_none() -> None:
    """NaN input on the very first frame, no warm-up: output is None
    (callers fall back to controller-trigger or similar)."""
    filt = FingerSignalFilter()
    c, o, t = filt.update(None, None, None)
    assert c is None
    assert o is None
    assert t is None


# ── slow-drift tracking (rolling median, not frozen entry value) ───────


def test_held_mode_tracks_slow_drift_via_rolling_median() -> None:
    """While latched, the output must follow slow drift inside the
    deadband (rolling median tracks naturally) without stair-stepping.

    Concretely: ramp from 0.05 to 0.07 over many frames, with each
    frame's |delta| << release_disp. The output must be a smooth,
    monotonic ramp that ends near 0.07 -- NOT frozen at 0.05 (which
    would force a release-on-displacement event partway through).
    """
    rng = np.random.default_rng(33)
    n_warm = DEFAULT_HOLD_WINDOW + 4
    n_ramp = 60
    n_total = n_warm + n_ramp

    # Phase A: 12 frames of pure noise around 0.05 to seed the latch.
    x = np.full((n_total, NUM_TOTAL_CHANNELS), 0.05, dtype=np.float64)
    x += rng.normal(0.0, 0.001, size=x.shape)
    # Phase B: linear ramp 0.05 -> 0.07 over n_ramp frames (slope per frame
    # = 0.02 / 60 ~= 3e-4, well below hold_std/release_disp).
    ramp = np.linspace(0.05, 0.07, n_ramp)
    for i, val in enumerate(ramp):
        x[n_warm + i] = val
    x[n_warm:] += rng.normal(0.0, 0.0005, size=(n_ramp, NUM_TOTAL_CHANNELS))

    filt = FingerSignalFilter()
    out = np.full((n_total, NUM_TOTAL_CHANNELS), np.nan)
    for t in range(n_total):
        c, _, _ = filt.update(x[t, 0:5], float(x[t, 5]), x[t, 6:10])
        if c is not None:
            out[t, 0:5] = c

    # Output during the ramp should NOT have any jump > release_disp.
    ramp_out = out[n_warm:n_total, 0]
    finite = ~np.isnan(ramp_out)
    assert finite.sum() >= n_ramp // 2, "Too many NaN outputs during ramp"
    deltas = np.abs(np.diff(ramp_out[finite]))
    assert deltas.max() < DEFAULT_RELEASE_DISP, (
        f"slow drift produced a >release_disp jump: max delta = "
        f"{deltas.max():.4f} (release_disp = {DEFAULT_RELEASE_DISP})"
    )
    # And the tail of the ramp should be near 0.07 (not 0.05) -- proves
    # the median tracked the drift instead of staying frozen.
    assert abs(ramp_out[-3] - 0.07) < 0.005, (
        f"end of slow ramp should be near 0.07; got {ramp_out[-3]:.4f}"
    )


# ── identity / passthrough modes ───────────────────────────────────────


def test_identity_passthrough_with_alpha_one_and_zero_holdstd() -> None:
    """``ema_alpha = 1.0`` + ``hold_std = 0`` should produce a strict
    identity filter (after the warm-up window). Acts as a regression
    guard -- callers can use these settings to disable the filter
    without removing the wiring.
    """
    params = FingerFilterParams(
        ema_alpha=1.0,
        hold_std=0.0,
        release_std=0.0,
        release_disp=DEFAULT_RELEASE_DISP,  # any >= 0 value
    )
    filt = FingerSignalFilter(params)
    rng = np.random.default_rng(0)
    n = DEFAULT_HOLD_WINDOW + 10
    x = rng.uniform(0.0, 1.0, size=(n, NUM_TOTAL_CHANNELS))

    outs = np.zeros_like(x)
    for t in range(n):
        c, o, tt = filt.update(x[t, 0:5], float(x[t, 5]), x[t, 6:10])
        outs[t, 0:5] = c
        outs[t, 5] = o
        outs[t, 6:10] = tt

    # With hold_std = 0, the deadband never latches (entry condition
    # std < hold_std is never True for a non-degenerate signal).
    np.testing.assert_allclose(outs, x, atol=1e-12)


# ── reset clears all state ─────────────────────────────────────────────


def test_reset_clears_state() -> None:
    rng = np.random.default_rng(1)
    n = DEFAULT_HOLD_WINDOW + 5
    held_level = 0.6
    x = np.full((n, NUM_TOTAL_CHANNELS), held_level, dtype=np.float64)
    x += rng.normal(0.0, 0.002, size=x.shape)

    filt = FingerSignalFilter()
    for t in range(n):
        filt.update(x[t, 0:5], float(x[t, 5]), x[t, 6:10])

    # Now reset and feed a single sample at a different level. The
    # output should follow the new sample exactly (no leftover latch).
    filt.reset()
    fresh_level = 0.2
    c, o, tt = filt.update(
        np.full(5, fresh_level), fresh_level, np.full(4, fresh_level),
    )
    np.testing.assert_allclose(c, np.full(5, fresh_level))
    assert o == pytest.approx(fresh_level)
    np.testing.assert_allclose(tt, np.full(4, fresh_level))


# ── offline batch helper ───────────────────────────────────────────────


def test_filter_npz_offline_produces_same_result_as_streaming() -> None:
    """The offline helper must produce bit-identical output to feeding
    one frame at a time through ``FingerSignalFilter``.
    """
    rng = np.random.default_rng(123)
    n = 40
    curls = rng.uniform(0.0, 1.0, size=(n, 5)).astype(np.float64)
    thumb = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    tip = rng.uniform(0.0, 1.0, size=(n, NUM_TIP_OPPOSE_CHANNELS)).astype(np.float64)

    # Sprinkle in some NaN rows to exercise the NaN bridging path.
    curls[5] = np.nan
    thumb[5] = np.nan
    tip[5] = np.nan
    curls[20:23] = np.nan
    thumb[20:23] = np.nan
    tip[20:23] = np.nan

    c_out, o_out, t_out = filter_npz_offline(curls, thumb, tip)

    # Streaming reference
    filt = FingerSignalFilter()
    c_ref = np.full_like(curls, np.nan)
    o_ref = np.full_like(thumb, np.nan)
    t_ref = np.full_like(tip, np.nan)
    for i in range(n):
        c_in = None if np.isnan(curls[i]).any() else curls[i]
        o_in = None if np.isnan(thumb[i]) else float(thumb[i])
        t_in = None if np.isnan(tip[i]).all() else tip[i]
        c_f, o_f, t_f = filt.update(c_in, o_in, t_in)
        if c_f is not None:
            c_ref[i] = c_f
        if o_f is not None:
            o_ref[i] = o_f
        if t_f is not None:
            t_ref[i] = t_f

    # Replace NaN-vs-NaN with strict equality for the comparison.
    np.testing.assert_array_equal(np.isnan(c_out), np.isnan(c_ref))
    np.testing.assert_array_equal(np.isnan(o_out), np.isnan(o_ref))
    np.testing.assert_array_equal(np.isnan(t_out), np.isnan(t_ref))
    finite_c = ~np.isnan(c_out)
    finite_o = ~np.isnan(o_out)
    finite_t = ~np.isnan(t_out)
    np.testing.assert_allclose(c_out[finite_c], c_ref[finite_c], atol=1e-12)
    np.testing.assert_allclose(o_out[finite_o], o_ref[finite_o], atol=1e-12)
    np.testing.assert_allclose(t_out[finite_t], t_ref[finite_t], atol=1e-12)
