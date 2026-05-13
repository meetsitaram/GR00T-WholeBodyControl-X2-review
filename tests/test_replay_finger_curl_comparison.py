"""Unit tests for ``gear_sonic.scripts.replay_finger_curl_comparison``.

The script is mostly plotting glue, but the analytics pieces
(``_decode_raw_curls``, ``_apply_new_stretch``, the recording-
freshness heuristic) are testable without any matplotlib state.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.scripts.replay_finger_curl_comparison import (
    _apply_new_stretch,
    _check_recording_is_pre_stretch,
    _decode_raw_curls,
)
from gear_sonic.utils.teleop.x2_hand_retarget import (
    HAND_GRASP_CLOSED_RAD_LEFT,
    HAND_GRASP_CLOSED_RAD_RIGHT,
    HAND_GRASP_OPEN_RAD_LEFT,
    HAND_GRASP_OPEN_RAD_RIGHT,
    per_finger_grasp_command_from_curls,
    stretch_finger_curls,
)


def test_decode_raw_curls_round_trips_no_stretch() -> None:
    """If we synthesise commanded motor commands by linearly lerping
    OPEN -> CLOSED on a known per-finger curl, ``_decode_raw_curls``
    must recover those exact curls. This is the round-trip the
    replay tool depends on."""
    rng = np.random.default_rng(42)
    n = 100
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


def test_apply_new_stretch_matches_per_frame_call() -> None:
    """``_apply_new_stretch`` is a thin loop over
    :func:`stretch_finger_curls`. Verify per-row equality."""
    rng = np.random.default_rng(123)
    raw = rng.uniform(0, 1, size=(50, 5))
    out = _apply_new_stretch(raw)
    assert out.shape == raw.shape
    for i in range(raw.shape[0]):
        np.testing.assert_allclose(
            out[i], stretch_finger_curls(raw[i]), atol=1e-12,
        )


def test_check_recording_is_pre_stretch_true_for_pre_stretch_data() -> None:
    """A pre-stretch recording has many raw curls in the [0.30, 0.70]
    middle range -- the heuristic must return True."""
    rng = np.random.default_rng(7)
    n = 200
    raw_left = rng.uniform(0, 1, size=(n, 5))
    raw_right = rng.uniform(0, 1, size=(n, 5))
    assert _check_recording_is_pre_stretch(raw_left, raw_right)


def test_check_recording_is_pre_stretch_false_for_bimodal_data() -> None:
    """A post-stretch recording has raw curls heavily clustered
    near 0 or 1 -- the heuristic must return False."""
    rng = np.random.default_rng(9)
    n = 200
    # Bernoulli-like: 50% near 0, 50% near 1, only 0% in middle.
    flips = rng.integers(0, 2, size=(n, 5))
    raw_left = flips.astype(np.float64) + rng.uniform(-0.02, 0.02, size=(n, 5))
    raw_right = flips.astype(np.float64) + rng.uniform(-0.02, 0.02, size=(n, 5))
    raw_left = np.clip(raw_left, 0, 1)
    raw_right = np.clip(raw_right, 0, 1)
    assert not _check_recording_is_pre_stretch(raw_left, raw_right)


def test_decode_handles_ndarray_dtype_promotion() -> None:
    """The script's anchor arrays are float64; commanded q can be
    float32 in older recordings. Round-trip still works."""
    n = 10
    rng = np.random.default_rng(0)
    raw = rng.uniform(0, 1, size=(n, 5)).astype(np.float64)
    cmd_left = np.empty((n, 10), dtype=np.float32)
    for i in range(n):
        cmd_left[i] = per_finger_grasp_command_from_curls(
            "left", raw[i], apply_curl_compensation=False,
        ).astype(np.float32)
    cmd_right = cmd_left.copy()
    decoded_left, _ = _decode_raw_curls(cmd_left, cmd_right)
    np.testing.assert_allclose(decoded_left, raw, atol=1e-5)
