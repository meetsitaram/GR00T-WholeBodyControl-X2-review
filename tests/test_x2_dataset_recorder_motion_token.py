"""Inline-tokenizer tests for the DEPRECATED direct-mode chokepoint.

These tests pin the contract for ``X2DatasetRecorder._encode_motion_token``
-- the *legacy* freeze-pose path used by the recorder's direct mode
(Quest-driven, no planner snapshot to source a real future window
from). Subscribe-mode (planner-driven) coverage lives in
:mod:`tests.test_x2_dataset_recorder_real_future_token`, which exercises
``_encode_motion_token_from_snapshot`` -- the multi-frame replacement.

This file kept intact so the v0 fall-back path (and the kinematic-only
"no checkpoint" smoke-test path) stays guarded against silent
regressions.

The unit tests don't need a live checkpoint -- they exercise the
helper's gating against a tiny stub. The integration test loads the
real :data:`gear_sonic.scripts.record_synthetic_smoketest_dataset.DEFAULT_SONIC_CHECKPOINT`
and is skipped automatically when that file is absent (keeps CI green
on machines without the cloud checkpoint mirror).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pytest


# ── Helper: minimal stand-in for X2DatasetRecorder._encode_motion_token ──
#
# The real recorder constructor pulls in MuJoCo, ZMQ, the robot model,
# the Quest 3 reader and a handful of file-system probes -- none of
# which we want to drag into a unit test of a 4-line dispatcher. The
# helper itself is small enough that we can import the *unbound method*
# and run it against a hand-rolled "self".


from gear_sonic.utils.teleop.x2_dataset_recorder import (  # noqa: E402
    SONIC_MOTION_TOKEN_DIM,
    X2DatasetRecorder,
)


class _StubRecorder:
    """Just enough surface to call ``_encode_motion_token`` on."""

    def __init__(self, tokenizer=None) -> None:
        self._tokenizer = tokenizer
        self._zero_motion_token = np.zeros(
            SONIC_MOTION_TOKEN_DIM, dtype=np.float64
        )


def _encode(stub, body_q, root_quat_xyzw=None):
    return X2DatasetRecorder._encode_motion_token.__get__(stub)(
        body_q, root_quat_xyzw=root_quat_xyzw
    )


# ── Unit tests (no checkpoint required) ──────────────────────────────────


def test_encode_motion_token_no_tokenizer_returns_zeros():
    """Without ``--sonic-checkpoint`` the helper returns zeros.

    This is the documented kinematic-only smoke-test path. The recorder
    fires a one-shot warning at startup (covered by the integration
    test below); per-tick we just hand back the cached zero vector.
    """
    stub = _StubRecorder(tokenizer=None)
    body_q = np.linspace(-0.5, 0.5, 31, dtype=np.float64)
    out = _encode(stub, body_q)
    assert out.shape == (SONIC_MOTION_TOKEN_DIM,)
    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, np.zeros(SONIC_MOTION_TOKEN_DIM))


def test_encode_motion_token_no_tokenizer_returns_cached_object():
    """The zero path returns the *cached* zero vector, not a new alloc.

    A defensive caller can mutate the returned buffer (legacy code does
    this in a couple of places); the test pins that we hand back the
    same object so a subsequent mutation surfaces immediately rather
    than getting silently swallowed by per-call allocation. (Today no
    caller mutates -- but pinning the contract documents the intent.)
    """
    stub = _StubRecorder(tokenizer=None)
    a = _encode(stub, np.zeros(31, dtype=np.float64))
    b = _encode(stub, np.ones(31, dtype=np.float64))
    assert a is b
    assert a is stub._zero_motion_token


def test_encode_motion_token_forwards_body_q_and_quat_to_tokenizer():
    """The helper passes through to ``tokenizer.encode`` verbatim.

    This is the chokepoint the plan introduces -- if it ever drifts
    (e.g. a future caller swaps body_q for a Pinocchio-ordered vector,
    or drops the root quat) the contract test above breaks loudly.
    """
    captured: dict = {}

    class _SpyTokenizer:
        def encode(self, body_q, *, root_rot_xyzw=None):
            captured["body_q"] = np.asarray(body_q).copy()
            captured["root_rot_xyzw"] = (
                np.asarray(root_rot_xyzw).copy()
                if root_rot_xyzw is not None
                else None
            )
            return np.full(SONIC_MOTION_TOKEN_DIM, 0.125, dtype=np.float64)

    stub = _StubRecorder(tokenizer=_SpyTokenizer())
    body_q = np.linspace(-0.3, 0.4, 31, dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    out = _encode(stub, body_q, root_quat_xyzw=quat)

    np.testing.assert_array_equal(captured["body_q"], body_q)
    np.testing.assert_array_equal(captured["root_rot_xyzw"], quat)
    np.testing.assert_array_equal(
        out, np.full(SONIC_MOTION_TOKEN_DIM, 0.125)
    )


def test_encode_motion_token_passes_none_quat_through():
    """When the caller omits root_quat_xyzw, ``None`` reaches the encoder.

    OnlineSonicTokenizer.encode treats ``None`` as identity. The
    recorder's VR-driven path (no merged body_pose subscription)
    relies on this fallback because the operator's pelvis isn't
    tracked -- only head + wrists are.
    """
    captured: dict = {}

    class _SpyTokenizer:
        def encode(self, body_q, *, root_rot_xyzw=None):
            captured["root_rot_xyzw"] = root_rot_xyzw
            return np.zeros(SONIC_MOTION_TOKEN_DIM, dtype=np.float64)

    stub = _StubRecorder(tokenizer=_SpyTokenizer())
    _encode(stub, np.zeros(31, dtype=np.float64))
    assert captured["root_rot_xyzw"] is None


# ── Integration test (gated on real SONIC checkpoint) ────────────────────


@pytest.fixture(scope="module")
def _sonic_tokenizer():
    """Real OnlineSonicTokenizer, skipped when the .pt is missing."""
    from gear_sonic.scripts.record_synthetic_smoketest_dataset import (
        DEFAULT_SONIC_CHECKPOINT,
    )
    if not DEFAULT_SONIC_CHECKPOINT.exists():
        pytest.skip(
            f"SONIC checkpoint not found at {DEFAULT_SONIC_CHECKPOINT}; "
            "this test is dev-box-only (requires the cloud checkpoint "
            "mirror)."
        )
    from gear_sonic.utils.teleop.online_sonic_tokenizer import (
        OnlineSonicTokenizer,
    )
    return OnlineSonicTokenizer.from_checkpoint(
        DEFAULT_SONIC_CHECKPOINT, device="cpu"
    )


@pytest.mark.slow
def test_encode_motion_token_real_checkpoint_returns_nonzero(_sonic_tokenizer):
    """End-to-end: a real body_q snapshot encodes to a non-zero token.

    The test uses CPU device on purpose -- it costs ~5 s and avoids
    GPU contention with the deploy / VLA when the dev box happens to
    be running them. The token shape + dtype contract here pins the
    parquet-writer expectation; per-frame variance / determinism are
    asserted in the next two tests.
    """
    stub = _StubRecorder(tokenizer=_sonic_tokenizer)
    body_q = np.linspace(-0.2, 0.2, 31, dtype=np.float64)
    out = _encode(stub, body_q)

    assert out.shape == (SONIC_MOTION_TOKEN_DIM,)
    assert out.dtype == np.float64
    assert np.any(out != 0.0), (
        "real checkpoint must produce a non-zero token; got all zeros "
        "which would silently re-introduce the v0 dataset gap"
    )
    assert np.all(np.isfinite(out)), "token must be finite"


@pytest.mark.slow
def test_encode_motion_token_real_checkpoint_is_deterministic(_sonic_tokenizer):
    """Encoding the same body_q twice gives bit-identical tokens.

    The FSQ-quantized output is by construction deterministic for a
    fixed encoder + input. Pinning this catches accidental dropout /
    eval-mode regressions in OnlineSonicTokenizer.from_checkpoint.
    """
    stub = _StubRecorder(tokenizer=_sonic_tokenizer)
    body_q = np.linspace(-0.1, 0.3, 31, dtype=np.float64)
    a = _encode(stub, body_q)
    b = _encode(stub, body_q)
    np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_encode_motion_token_real_checkpoint_per_frame_variance(_sonic_tokenizer):
    """Distinct body_q snapshots produce distinct tokens.

    If the encoder mode-collapsed (e.g. fed an OOD input that lands in
    a single FSQ codebook entry for every input) the dataset would
    degenerate to a single token across all frames -- which trains the
    VLA to predict a constant. We enforce that ~20 distinct synthetic
    body_q snapshots map to multiple distinct tokens.
    """
    stub = _StubRecorder(tokenizer=_sonic_tokenizer)
    rng = np.random.default_rng(seed=20260513)
    base = np.linspace(-0.2, 0.2, 31, dtype=np.float64)
    tokens = np.stack(
        [
            _encode(stub, base + rng.uniform(-0.1, 0.1, size=31))
            for _ in range(20)
        ]
    )
    unique_rows = {tuple(row.tolist()) for row in tokens}
    # 20 random offsets should produce many distinct FSQ tokens; we
    # only require >=3 to keep the test robust against the encoder
    # picking semantically similar quantization buckets for nearby
    # poses (which is correct behavior, just not what the test
    # constraint cares about).
    assert len(unique_rows) >= 3, (
        f"only {len(unique_rows)} distinct tokens across 20 random "
        "body_q snapshots -- encoder may have mode-collapsed"
    )
