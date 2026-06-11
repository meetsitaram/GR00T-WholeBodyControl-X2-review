"""Unit tests for gear_sonic.utils.pose_pipeline.clamp.

The clamps are tiny but high-stakes: they cap the per-tick joint-
position step on the wire, so a regression that silently doubles the
cap or stops freezing capped-to-zero joints would slam the body
across the LIVE -> OVERRIDE takeover edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.pose_pipeline.clamp import (
    clamp_vector_step_f32,
    clamp_vector_step_per_joint_f32,
)


# ===========================================================================
# clamp_vector_step_f32
# ===========================================================================
def test_scalar_clamp_passthrough_when_step_within_cap() -> None:
    prev = np.zeros(5, dtype=np.float32)
    target = np.array([0.01, -0.005, 0.0, 0.008, -0.001], dtype=np.float32)
    out = clamp_vector_step_f32(target, prev, max_step=0.02)
    np.testing.assert_allclose(out, target, atol=0.0)


def test_scalar_clamp_shrinks_vector_proportionally() -> None:
    """When ANY joint exceeds the cap, the whole vector shrinks by
    the SAME factor (peak / max_step) so the direction is preserved."""
    prev = np.zeros(3, dtype=np.float32)
    target = np.array([0.0, 0.1, 0.05], dtype=np.float32)
    cap = 0.01
    out = clamp_vector_step_f32(target, prev, max_step=cap)
    # Peak is 0.1, so factor = 0.01/0.1 = 0.1; output = target * 0.1.
    np.testing.assert_allclose(
        out, target * 0.1, atol=1e-7
    )
    assert float(np.abs(out).max()) == pytest.approx(cap, abs=1e-7)


def test_scalar_clamp_returns_target_copy_when_max_step_zero() -> None:
    """max_step == 0 means "no clamp" by convention (callers signal
    disable by passing 0)."""
    prev = np.zeros(3, dtype=np.float32)
    target = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    out = clamp_vector_step_f32(target, prev, max_step=0.0)
    np.testing.assert_allclose(out, target, atol=0.0)
    assert out is not target  # must be a copy


def test_scalar_clamp_returns_target_copy_when_prev_is_none() -> None:
    """Cold start (no anchor) -> no clamp; pass-through with copy."""
    target = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    out = clamp_vector_step_f32(target, None, max_step=0.1)
    np.testing.assert_allclose(out, target, atol=0.0)


# ===========================================================================
# clamp_vector_step_per_joint_f32
# ===========================================================================
def test_per_joint_clamp_caps_each_joint_independently() -> None:
    prev = np.zeros(4, dtype=np.float32)
    target = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    caps = np.array([0.1, 0.2, 0.05, 1.0], dtype=np.float32)
    out = clamp_vector_step_per_joint_f32(target, prev, caps)
    expected = np.array([0.1, 0.2, 0.05, 0.5], dtype=np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-7)


def test_per_joint_clamp_freezes_joint_with_zero_cap() -> None:
    """cap == 0 means freeze the joint (used by tracking-feedback's
    hard-cap when a joint's measured error exceeds the hard threshold)."""
    prev = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    target = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    caps = np.array([0.5, 0.0, 0.5], dtype=np.float32)
    out = clamp_vector_step_per_joint_f32(target, prev, caps)
    # joint 0: |delta| = 0.9 > 0.5 -> prev + 0.5 = 0.6
    # joint 1: cap 0 -> stays at prev = 0.2
    # joint 2: |delta| = 0.7 > 0.5 -> prev + 0.5 = 0.8
    np.testing.assert_allclose(
        out, [0.6, 0.2, 0.8], atol=1e-6
    )


def test_per_joint_clamp_negative_cap_means_uncapped() -> None:
    """cap < 0 is the sentinel "no cap" (lets callers mask out non-
    arm joints from the per-joint clamp by passing a negative value)."""
    prev = np.zeros(3, dtype=np.float32)
    target = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    caps = np.array([-1.0, 0.5, -10.0], dtype=np.float32)
    out = clamp_vector_step_per_joint_f32(target, prev, caps)
    # joint 0: -1 -> uncapped -> 5.0
    # joint 1: cap 0.5 -> prev + 0.5 = 0.5
    # joint 2: -10 -> uncapped -> 5.0
    np.testing.assert_allclose(out, [5.0, 0.5, 5.0], atol=1e-6)


def test_per_joint_clamp_returns_copy_when_prev_is_none() -> None:
    target = np.array([1.0, 2.0], dtype=np.float32)
    caps = np.array([0.1, 0.1], dtype=np.float32)
    out = clamp_vector_step_per_joint_f32(target, None, caps)
    np.testing.assert_allclose(out, target, atol=0.0)


def test_per_joint_clamp_broadcasts_1d_cap() -> None:
    """A 1-D cap broadcasts against a 2-D target (matches the bridge's
    multi-batch policy output shape)."""
    prev = np.zeros((2, 3), dtype=np.float32)
    target = np.array(
        [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]], dtype=np.float32
    )
    caps = np.array([0.1, 0.2, 0.5], dtype=np.float32)
    out = clamp_vector_step_per_joint_f32(target, prev, caps)
    np.testing.assert_allclose(
        out,
        [[0.1, 0.2, 0.5], [0.1, 0.2, 0.5]],
        atol=1e-6,
    )


def test_per_joint_clamp_rejects_incompatible_shape() -> None:
    prev = np.zeros(4, dtype=np.float32)
    target = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    caps = np.array([0.1, 0.1, 0.1], dtype=np.float32)  # length 3, not 4
    with pytest.raises(ValueError, match="shape mismatch"):
        clamp_vector_step_per_joint_f32(target, prev, caps)
