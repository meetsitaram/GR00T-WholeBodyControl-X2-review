"""Unit tests for ``gear_sonic.scripts.tune_finger_curl_compensation``.

The tuner is mostly a CLI orchestrator on top of small numpy
helpers; we test the helpers in isolation since they encode the
scoring objectives the runtime defaults are tuned against.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.scripts.tune_finger_curl_compensation import (
    _decode_raw_curls,
    _detect_hand_mode,
    _detect_static_frames,
    _natural_modes,
    _score,
    _stretch,
    _toggle_rate_per_s,
    _within_block_residuals,
)


def test_detect_hand_mode_picks_up_per_finger_variation() -> None:
    """Hand-mode frames have non-uniform per-finger curl values
    (XRHand reports each finger separately). Controller-mode
    frames have all 5 curls equal (one trigger drives all 10
    motors uniformly)."""
    # 10 controller-mode frames (all curls equal per row).
    controller = np.tile(np.linspace(0.1, 0.9, 10)[:, None], (1, 5))
    # 10 hand-mode frames (per-finger spread).
    rng = np.random.default_rng(0)
    hand = rng.uniform(0, 1, size=(10, 5))

    mask_ctrl = _detect_hand_mode(controller)
    mask_hand = _detect_hand_mode(hand)

    assert not mask_ctrl.any(), "controller-mode frames have zero std"
    assert mask_hand.all(), "hand-mode frames have spread >= 0.02"


def test_stretch_matches_per_frame_call() -> None:
    """``_stretch`` (used internally by the tuner) matches the
    canonical :func:`stretch_finger_curls` math for any shape."""
    from gear_sonic.utils.teleop.x2_hand_retarget import stretch_finger_curls

    rng = np.random.default_rng(1)
    raws = rng.uniform(0, 1, size=(20, 5))
    out_tuner = _stretch(raws, deadzone=0.30, full_threshold=0.40, gamma=4.0)
    for i in range(raws.shape[0]):
        ref = stretch_finger_curls(
            raws[i], deadzone=0.30, full_threshold=0.40, gamma=4.0,
        )
        np.testing.assert_allclose(out_tuner[i], ref, atol=1e-12)


def test_natural_modes_finds_obvious_break() -> None:
    """A bimodal distribution should yield a rest-mode top below
    the curl-mode bottom."""
    rng = np.random.default_rng(2)
    rest = rng.normal(0.20, 0.03, size=(500, 5)).clip(0, 1)
    curl = rng.normal(0.85, 0.05, size=(500, 5)).clip(0, 1)
    raw = np.concatenate([rest, curl], axis=0)
    rest_top, curl_bot = _natural_modes(raw)
    assert rest_top < curl_bot, f"{rest_top=} should be < {curl_bot=}"
    # Rest mode top should sit near rest distribution's right tail.
    assert 0.15 <= rest_top <= 0.40
    # Curl mode bottom should sit near curl distribution's left tail.
    assert 0.55 <= curl_bot <= 0.95


def test_toggle_rate_for_constant_signal_is_zero() -> None:
    """A signal that never crosses 0.5 should have toggle rate 0."""
    out = np.full((100, 5), 0.9)
    rate = _toggle_rate_per_s(out, fps=50.0)
    assert rate == 0.0


def test_toggle_rate_counts_alternating_signal() -> None:
    """An alternating 0/1 signal at fps=50 has 49 transitions
    per finger over 100 frames -> 49/(99/50) toggles per second."""
    pattern = np.tile([0.0, 1.0], 50)
    out = np.tile(pattern[:, None], (1, 5))
    rate = _toggle_rate_per_s(out, fps=50.0)
    expected = 99.0 / (99 / 50.0)
    np.testing.assert_allclose(rate, expected, atol=0.01)


def test_score_prefers_bimodal_output_over_floating() -> None:
    """A parameter triple that produces a clean bimodal distribution
    should score higher than one that leaves outputs floating."""
    rng = np.random.default_rng(3)
    rest = rng.normal(0.20, 0.03, size=(500, 5)).clip(0, 1)
    curl = rng.normal(0.85, 0.05, size=(500, 5)).clip(0, 1)
    raw = np.concatenate([rest, curl], axis=0)

    fps = 50.0
    weights = dict(w_bimodal=1.0, w_rest=2.0, w_fist=1.0, w_toggle=0.05)

    bad = _score(raw, fps, deadzone=0.0, full_threshold=1.0, gamma=1.0, **weights)
    good = _score(raw, fps, deadzone=0.30, full_threshold=0.50, gamma=3.0, **weights)
    assert good.composite > bad.composite
    assert good.bimodality > bad.bimodality


def test_score_rejects_invalid_parameter_ordering() -> None:
    """Out-of-range / mis-ordered parameters return a -inf score."""
    raw = np.zeros((10, 5))
    weights = dict(w_bimodal=1.0, w_rest=2.0, w_fist=1.0, w_toggle=0.05)
    s = _score(raw, 50.0, deadzone=0.5, full_threshold=0.4, gamma=1.0, **weights)
    assert s.composite == -1e9
    s2 = _score(raw, 50.0, deadzone=0.2, full_threshold=0.5, gamma=-1.0, **weights)
    assert s2.composite == -1e9


def test_decode_raw_curls_inverse_of_per_finger_command() -> None:
    """Tuner decode and replay decode share the same OPEN/CLOSED
    anchors and produce identical results."""
    from gear_sonic.utils.teleop.x2_hand_retarget import (
        per_finger_grasp_command_from_curls,
    )

    rng = np.random.default_rng(4)
    n = 50
    raw_left = rng.uniform(0, 1, size=(n, 5))
    raw_right = rng.uniform(0, 1, size=(n, 5))
    cmd_left = np.empty((n, 10))
    cmd_right = np.empty((n, 10))
    for i in range(n):
        cmd_left[i] = per_finger_grasp_command_from_curls(
            "left", raw_left[i], apply_curl_compensation=False,
        )
        cmd_right[i] = per_finger_grasp_command_from_curls(
            "right", raw_right[i], apply_curl_compensation=False,
        )
    decoded_left, decoded_right = _decode_raw_curls(cmd_left, cmd_right)
    np.testing.assert_allclose(decoded_left, raw_left, atol=1e-12)
    np.testing.assert_allclose(decoded_right, raw_right, atol=1e-12)


def test_detect_static_frames_finds_truly_static_segment() -> None:
    """A 50-frame sequence where the second half is constant should
    flag those as static."""
    rng = np.random.default_rng(5)
    moving = rng.uniform(0, 1, size=(25, 5))
    static = np.full((25, 5), 0.3) + rng.uniform(-0.001, 0.001, size=(25, 5))
    seq = np.concatenate([moving, static], axis=0)
    mask = _detect_static_frames(seq, win=10)
    # The first 25 are too short / too varied to flag.
    assert mask[:25].mean() < 0.2
    # The last 15 (after the 10-frame window has been entirely in
    # the static region) should be mostly flagged.
    assert mask[35:].mean() > 0.8


def test_within_block_residuals_returns_centered_data() -> None:
    """Residuals should have approximately zero mean per finger."""
    rng = np.random.default_rng(6)
    static = np.full((30, 5), 0.5) + rng.uniform(-0.002, 0.002, size=(30, 5))
    # Insert a 10-frame moving prefix and suffix to exercise block
    # detection.
    moving = rng.uniform(0, 1, size=(15, 5))
    seq = np.concatenate([moving, static, moving], axis=0)
    res = _within_block_residuals(seq, win=10)
    assert res is not None
    np.testing.assert_allclose(
        res.mean(axis=0), np.zeros(5), atol=0.001,
    )
