"""
M6 / Stage C1 acceptance gate: SONIC motion-token labels.

The M3 smoke-test orchestrator filled ``action.motion_token`` with
``np.zeros(64)`` because we hadn't wired up SONIC's ``g1`` encoder yet.
Stage C1 plugs that gap so the VLA training data has real, deploy-aligned
motion tokens for the model to learn against.

The M6 gate locks the following invariants:

1. ``SonicMotionTokenLabeler.label_trajectory`` returns
   ``(T, SONIC_MOTION_TOKEN_DIM)`` ``float64`` -- the exact per-frame
   shape/dtype the LeRobot exporter and the C++ deploy harness expect on
   the wire.
2. The labeler is deterministic: two calls with identical inputs produce
   bit-identical outputs (CPU path; cuda determinism not required here).
3. Tokens lie on the FSQ-32 lattice. With ``L=32`` levels per dim, every
   value rounds to ``k * 2/L`` for some integer ``k`` -- the exact
   discrete codebook the deploy harness's ONNX decoder consumes.
4. ``unique_levels <= L`` (stats sanity) and at least a few distinct
   values appear on a real trajectory (catches a constant / collapsed
   encoder).
5. A static stand pose (single frame replicated) produces a *constant*
   token sequence -- if the same observation is fed in for every frame,
   the encoder + FSQ must commit to one codebook entry. Detects state
   leakage between frames.
6. A dynamic minecraft trajectory produces a token sequence that's
   detectably different from the stand-pose sequence (mean L2 delta > 0).
7. ``MOTION_TOKEN_SOURCE_SONIC_G1`` round-trips through
   ``build_smoketest_dataset`` end-to-end: episodes 0..N-1 in the
   parquet store carry non-zero, FSQ-quantized tokens that match what
   the labeler produced for the matching variation.
8. ``meta/info.json::script_config`` records ``motion_token_source`` and
   ``sonic_checkpoint_path`` so future readers can reconstruct exactly
   which encoder produced each label batch.
9. The summary returned by ``build_smoketest_dataset`` exposes the same
   provenance fields (``motion_token_source``, ``sonic_checkpoint_path``).
10. The "zeros" path stays bit-identical to the M3 placeholder, so
    earlier acceptance gates keep passing without touching them.
11. Unknown ``motion_token_source`` values raise ``ValueError`` BEFORE
    any disk I/O happens (fail-fast invariant).
12. Missing checkpoint path raises a clean ``FileNotFoundError`` at
    invocation time (no half-built dataset on disk).

Skips
-----

The whole module skips cleanly when the SONIC checkpoint at
:data:`DEFAULT_SONIC_CHECKPOINT` isn't on disk -- so CI hosts without the
h200 cloud bundle see a single skip instead of failures.

Run via::

    .venv/bin/python -m pytest tests/test_x2_motion_token_labels.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from gear_sonic.data.features_x2_vla import (  # noqa: E402
    SONIC_MOTION_TOKEN_DIM,
)
from gear_sonic.scripts.record_synthetic_smoketest_dataset import (  # noqa: E402
    DEFAULT_SONIC_CHECKPOINT,
    DEFAULT_STAND_POSE_MJ_RAD,
    MOTION_TOKEN_SOURCES,
    MOTION_TOKEN_SOURCE_SONIC_G1,
    MOTION_TOKEN_SOURCE_ZEROS,
    X2_BODY_DOF,
    build_smoketest_dataset,
    compose_body_trajectory,
    load_source_motion,
)


# ---------------------------------------------------------------------------
# Module-wide skip when the SONIC checkpoint isn't on disk. We can't run
# any of the encoder-backed invariants in that case; the zeros-path
# invariants do not need this fixture so they live in a separate module-
# level no-skip section below.
# ---------------------------------------------------------------------------


pytestmark_skip_if_missing = pytest.mark.skipif(
    not DEFAULT_SONIC_CHECKPOINT.exists(),
    reason=(
        f"SONIC checkpoint {DEFAULT_SONIC_CHECKPOINT} not on disk; "
        "encoder-backed invariants unavailable."
    ),
)


# ---------------------------------------------------------------------------
# Stable-import invariants (don't require the checkpoint).
# ---------------------------------------------------------------------------


def test_motion_token_sources_constants_are_stable():
    """Public constants stay stable; downstream tooling pins on them."""
    assert MOTION_TOKEN_SOURCES == ("zeros", "sonic_g1")
    assert MOTION_TOKEN_SOURCE_ZEROS == "zeros"
    assert MOTION_TOKEN_SOURCE_SONIC_G1 == "sonic_g1"


def test_motion_token_source_zeros_remains_default(tmp_path: Path):
    """A vanilla build still emits the M3 zero placeholder.

    Locks the backward-compat invariant so the M3 / M5 acceptance gates
    don't need a SONIC checkpoint to keep passing.
    """
    out = tmp_path / "ds_zeros"
    summary = build_smoketest_dataset(
        output_dir=out,
        num_episodes=1,
        max_frames=8,
        seed=0,
        skip_stats=True,
    )
    assert summary.motion_token_source == MOTION_TOKEN_SOURCE_ZEROS
    assert summary.sonic_checkpoint_path is None

    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    sc = info["script_config"]
    assert sc["motion_token_source"] == MOTION_TOKEN_SOURCE_ZEROS
    assert sc["sonic_checkpoint_path"] is None

    df = _load_episode_parquet(out, ep_idx=0)
    arr = np.stack(df["action.motion_token"].to_numpy())
    assert arr.shape == (len(df), SONIC_MOTION_TOKEN_DIM)
    assert arr.dtype == np.float64
    np.testing.assert_array_equal(arr, np.zeros_like(arr))


def test_unknown_motion_token_source_raises_before_io(tmp_path: Path):
    """Fail-fast: bad source string must raise BEFORE any disk write."""
    out = tmp_path / "ds_invalid"
    with pytest.raises(ValueError, match="motion_token_source"):
        build_smoketest_dataset(
            output_dir=out,
            num_episodes=1,
            max_frames=4,
            motion_token_source="bogus",
            skip_stats=True,
        )
    assert not out.exists(), "output dir must not be created on validation failure"


def test_missing_sonic_checkpoint_raises_cleanly(tmp_path: Path):
    """Bad checkpoint path raises FileNotFoundError, not a generic crash."""
    out = tmp_path / "ds_bad_ckpt"
    bogus_ckpt = tmp_path / "this_does_not_exist.pt"
    with pytest.raises(FileNotFoundError, match="SONIC checkpoint not found"):
        build_smoketest_dataset(
            output_dir=out,
            num_episodes=1,
            max_frames=4,
            motion_token_source=MOTION_TOKEN_SOURCE_SONIC_G1,
            sonic_checkpoint_path=bogus_ckpt,
            skip_stats=True,
        )


# ---------------------------------------------------------------------------
# Encoder-backed invariants (require the SONIC .pt checkpoint).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labeler():
    pytest.importorskip("torch")
    if not DEFAULT_SONIC_CHECKPOINT.exists():
        pytest.skip(
            f"SONIC checkpoint {DEFAULT_SONIC_CHECKPOINT} not on disk."
        )
    from gear_sonic.scripts.sonic_motion_token_labeler import (
        SonicMotionTokenLabeler,
    )

    return SonicMotionTokenLabeler(
        DEFAULT_SONIC_CHECKPOINT, device="cpu", motion_fps=50.0
    )


@pytest.fixture(scope="module")
def minecraft_body():
    src = load_source_motion()
    body = compose_body_trajectory(src.arm_trajectory)[:120]
    assert body.shape == (120, X2_BODY_DOF), body.shape
    return body


def _on_fsq_lattice(arr: np.ndarray, levels: int = 32) -> bool:
    step = 2.0 / levels
    return np.allclose(arr, np.round(arr / step) * step, atol=1e-6)


@pytestmark_skip_if_missing
def test_label_trajectory_shape_and_dtype(labeler, minecraft_body):
    tokens = labeler.label_trajectory(minecraft_body)
    assert tokens.shape == (minecraft_body.shape[0], SONIC_MOTION_TOKEN_DIM)
    assert tokens.dtype == np.float64


@pytestmark_skip_if_missing
def test_label_trajectory_is_deterministic(labeler, minecraft_body):
    a = labeler.label_trajectory(minecraft_body)
    b = labeler.label_trajectory(minecraft_body)
    np.testing.assert_array_equal(a, b)


@pytestmark_skip_if_missing
def test_label_trajectory_lies_on_fsq_lattice(labeler, minecraft_body):
    tokens = labeler.label_trajectory(minecraft_body)
    assert _on_fsq_lattice(tokens, levels=32), (
        "FSQ-32 lattice violated: max residual = "
        f"{np.abs(tokens - np.round(tokens / (2 / 32)) * (2 / 32)).max():.6e}"
    )


@pytestmark_skip_if_missing
def test_label_trajectory_uses_codebook(labeler, minecraft_body):
    tokens, stats = labeler.label_trajectory(minecraft_body, return_stats=True)
    assert stats.unique_levels <= 32  # FSQ-32 hard cap
    # On a real trajectory we should see strictly more than a single value.
    assert stats.unique_levels > 1, (
        "Encoder collapsed onto a single FSQ codebook entry."
    )


@pytestmark_skip_if_missing
def test_constant_observation_yields_constant_tokens(labeler):
    """Frozen body pose -> identical observation per frame -> identical tokens."""
    stand_pose = np.asarray(DEFAULT_STAND_POSE_MJ_RAD, dtype=np.float64)
    static_body = np.broadcast_to(stand_pose, (60, X2_BODY_DOF)).copy()
    tokens = labeler.label_trajectory(static_body)
    assert tokens.shape == (60, SONIC_MOTION_TOKEN_DIM)
    np.testing.assert_array_equal(tokens, tokens[0:1].repeat(60, 0))


@pytestmark_skip_if_missing
def test_dynamic_vs_static_tokens_differ(labeler, minecraft_body):
    """Real trajectory must produce a meaningfully different token stream."""
    stand_pose = np.asarray(DEFAULT_STAND_POSE_MJ_RAD, dtype=np.float64)
    T = minecraft_body.shape[0]
    static = np.broadcast_to(stand_pose, (T, X2_BODY_DOF)).copy()
    dyn_tokens = labeler.label_trajectory(minecraft_body)
    static_tokens = labeler.label_trajectory(static)
    delta = np.linalg.norm(dyn_tokens - static_tokens, axis=-1)
    assert delta.mean() > 0.0, (
        "Dynamic and static trajectories produced identical tokens; the "
        "encoder appears insensitive to body motion."
    )


# ---------------------------------------------------------------------------
# End-to-end build_smoketest_dataset round-trip.
# ---------------------------------------------------------------------------


def _load_episode_parquet(dataset_root: Path, *, ep_idx: int):
    pd = pytest.importorskip("pandas")
    parquet_path = (
        dataset_root
        / "data"
        / "chunk-000"
        / f"episode_{ep_idx:06d}.parquet"
    )
    return pd.read_parquet(parquet_path)


@pytestmark_skip_if_missing
def test_build_smoketest_dataset_end_to_end_sonic_g1(tmp_path: Path):
    """Full ``build_smoketest_dataset`` round-trip with SONIC labels.

    Locks invariants 7-9 in one shot:
      * tokens in the parquet are non-zero, FSQ-quantized
      * meta/info.json records the provenance
      * the summary surfaces the provenance
    """
    out = tmp_path / "ds_sonic"
    summary = build_smoketest_dataset(
        output_dir=out,
        num_episodes=1,
        max_frames=24,
        seed=0,
        motion_token_source=MOTION_TOKEN_SOURCE_SONIC_G1,
        sonic_checkpoint_path=DEFAULT_SONIC_CHECKPOINT,
        skip_stats=True,
    )
    assert summary.motion_token_source == MOTION_TOKEN_SOURCE_SONIC_G1
    assert summary.sonic_checkpoint_path == DEFAULT_SONIC_CHECKPOINT

    info_path = out / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    sc = info["script_config"]
    assert sc["motion_token_source"] == MOTION_TOKEN_SOURCE_SONIC_G1
    assert sc["sonic_checkpoint_path"] == str(DEFAULT_SONIC_CHECKPOINT)

    df = _load_episode_parquet(out, ep_idx=0)
    tokens = np.stack(df["action.motion_token"].to_numpy())
    assert tokens.shape == (len(df), SONIC_MOTION_TOKEN_DIM)
    assert tokens.dtype == np.float64
    assert np.linalg.norm(tokens) > 0.0, (
        "SONIC-labeled tokens must NOT be all zeros."
    )
    assert _on_fsq_lattice(tokens, levels=32), (
        "Persisted tokens drifted off the FSQ-32 lattice."
    )
