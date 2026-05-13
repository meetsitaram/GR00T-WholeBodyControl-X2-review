"""
M3 acceptance gate: autoencoder smoke-test pipeline (offline slice).

The full M3 acceptance gate is the closed-loop sim rollout described in
``docs/source/tutorials/vla_training.md`` §2 -- record -> fine-tune ->
deploy -> sim -> compare. That gate requires a GR00T fine-tune step
plus the C++ deploy on the X2 dev box, both of which are outside the
laptop this suite runs on.

This pytest gate covers the *pre-rollout* slice of the same loop, in
three layers:

1. **generate_motion_variations.py** -- pure numpy variation generator.
   Determinism (same seed -> byte-identical output), reachability of
   each transform branch, and shape contracts.

2. **record_synthetic_smoketest_dataset.py** -- LeRobot v2.1
   orchestrator. Builds a ~30-frame, 2-episode dataset using either
   the real Minecraft recording from ``agitbot-x2-record-and-replay``
   (when present at the canonical sibling path) or the deterministic
   synthetic fallback (when not). Either way, the gate proves the
   dataset round-trips through Isaac-GR00T's ``LeRobotEpisodeLoader``
   with the X2 modality config -- the same invariant M1 enforces, but
   now exercised on smoketest-shaped data.

3. **compare_motion_trajectories.py** -- L2 reconstruction metric. We
   feed it a pair of trajectories that are exactly equal (the recorded
   trajectory compared against itself) and assert the result is the
   identity (RMSE == 0, pass). Then we feed it a perturbed copy and
   assert the metric is non-trivial and the alignment modes work as
   advertised.

Together these three layers prove every Python-only link in the M3
chain is sound *before* a single GR00T fine-tune step has run -- which
is the entire point of the autoencoder smoke test methodology.

Run with::

    .venv/bin/python -m pytest tests/test_x2_smoketest_pipeline.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
GROOT_CLONE = REPO_ROOT / "external_dependencies" / "Isaac-GR00T"


def _ensure_groot_on_path() -> None:
    if not GROOT_CLONE.is_dir():
        raise RuntimeError(
            f"Isaac-GR00T clone not found at {GROOT_CLONE}. "
            "See docs/source/references/x2_isaac_groot_data_contract.md."
        )
    s = str(GROOT_CLONE)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Layer 1: generate_motion_variations
# ---------------------------------------------------------------------------


def test_generate_variations_is_deterministic_for_a_given_seed() -> None:
    from gear_sonic.scripts.generate_motion_variations import generate_variations

    base = np.tile(np.linspace(0, 1, 64)[:, None], (1, 14))
    a = generate_variations(base, num_variations=4, seed=123)
    b = generate_variations(base, num_variations=4, seed=123)
    assert len(a) == len(b) == 4
    for (pa, ta), (pb, tb) in zip(a, b):
        assert pa == pb, f"params diverged across runs: {pa} vs {pb}"
        np.testing.assert_array_equal(ta, tb)


def test_generate_variations_first_entry_is_identity() -> None:
    """The orchestrator + acceptance gate rely on the first variation
    being the unmodified base trajectory."""
    from gear_sonic.scripts.generate_motion_variations import generate_variations

    base = np.linspace(0, 1, 32)[:, None] * np.arange(14)
    out = generate_variations(base, num_variations=3, seed=7, include_identity=True)
    params, traj = out[0]
    assert params.stretch == 1.0
    assert params.noise_std == 0.0
    assert params.phase_shift_frames == 0
    assert params.lr_mirror is False
    np.testing.assert_array_equal(traj, base)


def test_time_stretch_changes_length_in_proportion_to_factor() -> None:
    from gear_sonic.scripts.generate_motion_variations import time_stretch

    base = np.zeros((100, 4))
    out_long = time_stretch(base, 1.5)
    out_short = time_stretch(base, 0.5)
    assert out_long.shape == (150, 4)
    assert out_short.shape == (50, 4)


def test_lr_mirror_swaps_columns_with_optional_signs() -> None:
    from gear_sonic.scripts.generate_motion_variations import apply_lr_mirror

    arr = np.array([[1.0, 2.0, 3.0, 4.0]])
    # Pair (0, 2) and (1, 3); flip sign on the first pair.
    out = apply_lr_mirror(arr, [0, 1], [2, 3], flip_signs=[-1, 1])
    np.testing.assert_array_equal(out, [[-3.0, 4.0, -1.0, 2.0]])


def test_phase_shift_is_cyclic() -> None:
    from gear_sonic.scripts.generate_motion_variations import phase_shift

    arr = np.arange(8).reshape(8, 1).astype(np.float64)
    out = phase_shift(arr, 3)
    np.testing.assert_array_equal(out.flatten(), [5, 6, 7, 0, 1, 2, 3, 4])


# ---------------------------------------------------------------------------
# joint_bias_noise: per-episode constant offset, smooth velocity profile
# ---------------------------------------------------------------------------


def test_joint_bias_noise_is_constant_offset_across_frames() -> None:
    """All frames must receive the SAME offset vector -- so the
    velocity / acceleration profile of the input is preserved up to
    float64 round-off. This is the property that distinguishes
    joint_bias_noise from gaussian_noise; if it ever regresses to
    per-frame draws the gentle preset breaks SONIC's encoder."""
    from gear_sonic.scripts.generate_motion_variations import joint_bias_noise

    rng = np.random.default_rng(0)
    base = np.cumsum(np.linspace(-0.05, 0.05, 50))[:, None] * np.arange(1, 8)
    out = joint_bias_noise(base, std=0.01, rng=rng)
    delta = out - base  # (T, D)
    # Every row must equal row 0 within float64 round-off (~1 ULP at
    # |x|≈0.4 is ~5e-17). Critically, the row-to-row deviation must be
    # *much* smaller than the bias scale (std=0.01 rad), so even a
    # tight 1e-12 atol is several orders of magnitude below the
    # signal we'd see if the function regressed to per-frame draws
    # (which would be O(std) ~= 1e-2).
    np.testing.assert_allclose(
        delta, np.broadcast_to(delta[0], delta.shape), atol=1e-12
    )
    # Velocity profile preserved up to round-off (1 ULP per
    # subtraction, much smaller than the std=0.01 bias scale).
    np.testing.assert_allclose(
        np.diff(out, axis=0), np.diff(base, axis=0), atol=1e-12
    )


def test_joint_bias_noise_zero_std_is_passthrough() -> None:
    from gear_sonic.scripts.generate_motion_variations import joint_bias_noise

    rng = np.random.default_rng(0)
    base = np.random.RandomState(7).randn(20, 14)
    out = joint_bias_noise(base, std=0.0, rng=rng)
    np.testing.assert_array_equal(out, base)


def test_joint_bias_noise_clip_caps_long_tail() -> None:
    """With clip_sigmas=2.0 every component of the per-episode offset
    must lie in [-2*std, +2*std]. Run many seeds and verify no draw
    escapes the clip."""
    from gear_sonic.scripts.generate_motion_variations import joint_bias_noise

    std = 0.01
    base = np.zeros((4, 14))
    max_abs_delta = 0.0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        out = joint_bias_noise(base, std=std, rng=rng, clip_sigmas=2.0)
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(out))))
    assert max_abs_delta <= 2.0 * std + 1e-12, (
        f"clip_sigmas=2 violated: saw |delta|={max_abs_delta}, expected <= {2*std}"
    )


def test_joint_bias_noise_unclipped_can_exceed_2sigma() -> None:
    """Sanity check: passing clip_sigmas=inf actually disables clipping
    so we can verify the clip is doing real work in the test above."""
    from gear_sonic.scripts.generate_motion_variations import joint_bias_noise

    std = 0.01
    base = np.zeros((4, 14))
    saw_over_2sigma = False
    for seed in range(500):
        rng = np.random.default_rng(seed)
        out = joint_bias_noise(base, std=std, rng=rng, clip_sigmas=np.inf)
        if float(np.max(np.abs(out))) > 2.0 * std:
            saw_over_2sigma = True
            break
    assert saw_over_2sigma, (
        "Without clipping we should see ≥1 draw above 2σ over 500 trials. "
        "If this regresses, the unclipped path is broken."
    )


def test_generate_variations_gentle_preset_yields_smooth_offsets() -> None:
    """End-to-end sanity check that the gentle preset produces what the
    user actually wants: same trajectory shape, bounded home-pose drift,
    no per-frame jitter."""
    from gear_sonic.scripts.generate_motion_variations import generate_variations
    from gear_sonic.scripts.record_synthetic_smoketest_dataset import PRESETS

    # 14-DoF arm-shaped synthetic base trajectory.
    T = 200
    t = np.linspace(0.0, 2.0 * np.pi, T)
    base = np.stack([0.3 * np.sin(t + 0.1 * d) for d in range(14)], axis=1)

    out = generate_variations(
        base, num_variations=8, seed=2026, **PRESETS["gentle"]
    )
    assert len(out) == 8
    # Episode 0 = identity.
    p0, traj0 = out[0]
    assert p0.bias_std == 0.0
    np.testing.assert_array_equal(traj0, base)

    # Episodes 1..7: bias is constant per episode, velocity matches base
    # bit-exact, max abs offset stays under the 2σ clip.
    sigma_arm = PRESETS["gentle"]["bias_std_range"][0]
    base_diff = np.diff(base, axis=0)
    for params, traj in out[1:]:
        assert params.stretch == 1.0
        assert params.noise_std == 0.0
        assert params.phase_shift_frames == 0
        assert params.lr_mirror is False
        assert params.bias_std == sigma_arm
        assert traj.shape == base.shape
        # Velocity profile is invariant up to float64 round-off
        # (per-episode constant offset preserves diffs to ~1 ULP).
        np.testing.assert_allclose(np.diff(traj, axis=0), base_diff, atol=1e-12)
        # Home-pose offset is bounded by the ±2σ clip.
        offset = traj[0] - base[0]
        np.testing.assert_allclose(traj - offset[None, :], base, atol=1e-12)
        assert float(np.max(np.abs(offset))) <= 2.0 * sigma_arm + 1e-12


# ---------------------------------------------------------------------------
# Layer 2: record_synthetic_smoketest_dataset
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smoketest_dataset(tmp_path_factory: pytest.TempPathFactory):
    """Build a tiny smoketest dataset for the test session."""
    from gear_sonic.scripts.record_synthetic_smoketest_dataset import (
        build_smoketest_dataset,
    )

    out_dir = tmp_path_factory.mktemp("x2_smoketest") / "dataset"
    summary = build_smoketest_dataset(
        output_dir=out_dir,
        num_episodes=2,
        max_frames=60,
        seed=42,
        overwrite=True,
    )
    return summary


def test_smoketest_dataset_has_lerobot_v21_layout(smoketest_dataset) -> None:
    out = smoketest_dataset.output_dir
    assert (out / "meta" / "info.json").is_file()
    assert (out / "meta" / "modality.json").is_file()
    assert (out / "meta" / "tasks.jsonl").is_file()
    assert (out / "meta" / "episodes.jsonl").is_file()
    assert (out / "meta" / "stats.json").is_file()
    for ep in (0, 1):
        assert (
            out
            / "data"
            / "chunk-000"
            / f"episode_{ep:06d}.parquet"
        ).is_file()
        assert (
            out
            / "videos"
            / "chunk-000"
            / "observation.images.ego_view"
            / f"episode_{ep:06d}.mp4"
        ).is_file()


def test_smoketest_info_json_carries_provenance(smoketest_dataset) -> None:
    info = json.loads(
        (smoketest_dataset.output_dir / "meta" / "info.json").read_text()
    )
    sc = info["script_config"]
    assert sc["robot_type"] == "agibot_x2_ultra"
    assert sc["embodiment_tag"] == "new_embodiment"
    assert sc["hand_variant"] == "omnihand_10"
    assert sc["smoketest"] is True
    assert sc["seed"] == 42
    assert sc["variations_planned"] == 2
    # source_label points to the asset (real or synthetic_fallback)
    assert isinstance(sc["source_label"], str) and len(sc["source_label"]) > 0


def test_smoketest_per_episode_recordings_are_written(smoketest_dataset) -> None:
    rec_root = smoketest_dataset.base_recordings_path
    for ep in range(smoketest_dataset.num_episodes):
        npz = rec_root / f"episode_{ep:04d}_recorded.npz"
        assert npz.is_file(), f"missing reference recording: {npz}"
        with np.load(npz) as data:
            assert data["body_trajectory"].shape[1] == 31
            assert data["left_hand_trajectory"].shape[1] == 10
            assert data["right_hand_trajectory"].shape[1] == 10
            assert data["arm_trajectory"].shape[1] == 14
            # ``trajectory`` is the default key that
            # compare_motion_trajectories.py CLI looks for.
            assert "trajectory" in data
            assert data["trajectory"].shape[1] == 31


def test_smoketest_first_episode_preserves_stand_pose_at_legs(smoketest_dataset) -> None:
    """The identity variation (episode 0) must leave legs/waist/head at the stand pose."""
    from gear_sonic.scripts.record_synthetic_smoketest_dataset import (
        DEFAULT_STAND_POSE_MJ_RAD,
        LEFT_LEG_INDICES,
        RIGHT_LEG_INDICES,
        WAIST_INDICES,
        HEAD_INDICES,
    )

    npz = smoketest_dataset.base_recordings_path / "episode_0000_recorded.npz"
    with np.load(npz) as data:
        body = np.asarray(data["body_trajectory"])
    stand = np.asarray(DEFAULT_STAND_POSE_MJ_RAD, dtype=body.dtype)

    for indices in (LEFT_LEG_INDICES, RIGHT_LEG_INDICES, WAIST_INDICES, HEAD_INDICES):
        sub = body[:, list(indices)]
        ref = stand[list(indices)]
        np.testing.assert_allclose(sub, np.broadcast_to(ref, sub.shape), atol=1e-6)


def test_smoketest_dataset_round_trips_through_groot_loader(smoketest_dataset) -> None:
    _ensure_groot_on_path()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
    from gear_sonic.data.x2_modality_config import make_x2_modality_config

    modality_configs = make_x2_modality_config(hand_dof=10, action_horizon=4)
    loader = LeRobotEpisodeLoader(
        dataset_path=str(smoketest_dataset.output_dir),
        modality_configs=modality_configs,
        video_backend="torchcodec",
    )
    assert len(loader) == 2
    assert loader.fps == 50
    episode = loader[0]
    for col in (
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.projected_gravity",
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
        "video.ego_view",
        "language.annotation.human.task_description",
    ):
        assert col in episode.columns, f"loader missing {col}"
    assert episode["action.motion_token"].iloc[0].shape == (64,)
    assert episode["action.left_hand_joints"].iloc[0].shape == (10,)
    text = episode["language.annotation.human.task_description"].iloc[0]
    assert text == "play minecraft music on piano"


# ---------------------------------------------------------------------------
# Layer 3: compare_motion_trajectories
# ---------------------------------------------------------------------------


def test_compare_trajectories_returns_zero_for_identity_pair() -> None:
    from gear_sonic.scripts.compare_motion_trajectories import compare_trajectories

    rec = np.linspace(0, 1, 100)[:, None] * np.arange(31)
    result = compare_trajectories(rec, rec.copy(), threshold_rad=1e-9)
    assert result.pass_, "identity comparison must pass at any threshold"
    assert result.rmse_overall == 0.0
    assert result.max_abs_error_overall == 0.0
    assert result.failing_dof_indices == []
    assert result.aligned_length == 100


def test_compare_trajectories_flags_dofs_above_threshold() -> None:
    from gear_sonic.scripts.compare_motion_trajectories import compare_trajectories

    rng = np.random.default_rng(0)
    rec = rng.normal(size=(50, 4))
    roll = rec.copy()
    roll[:, 2] += 0.5  # large bias on DOF 2

    res = compare_trajectories(rec, roll, threshold_rad=0.05, alignment_mode="exact")
    assert not res.pass_
    assert 2 in res.failing_dof_indices
    assert all(i not in res.failing_dof_indices for i in (0, 1, 3))


def test_compare_trajectories_resample_aligns_different_lengths() -> None:
    from gear_sonic.scripts.compare_motion_trajectories import compare_trajectories

    rec = np.linspace(0, 1, 60)[:, None] * np.array([1.0, 2.0])
    # Same trajectory but at twice the rate -> resample to match.
    roll = np.linspace(0, 1, 120)[:, None] * np.array([1.0, 2.0])
    res = compare_trajectories(rec, roll, threshold_rad=1e-6, alignment_mode="resample")
    assert res.aligned_length == 60
    assert res.pass_, f"linear resample of identical signal should pass: {res}"
    assert any("resample" in n for n in res.notes)


def test_compare_trajectories_shortest_truncates_and_warns() -> None:
    from gear_sonic.scripts.compare_motion_trajectories import compare_trajectories

    rec = np.zeros((40, 3))
    roll = np.zeros((25, 3))
    res = compare_trajectories(rec, roll, threshold_rad=1e-6, alignment_mode="shortest")
    assert res.aligned_length == 25
    assert any("truncated" in n for n in res.notes)
    assert res.pass_


def test_compare_trajectories_to_json_dict_renames_pass_field() -> None:
    """Downstream JSON consumers expect ``pass`` (no underscore)."""
    from gear_sonic.scripts.compare_motion_trajectories import compare_trajectories

    res = compare_trajectories(np.zeros((10, 2)), np.zeros((10, 2)), threshold_rad=0.0)
    d = res.to_json_dict()
    assert "pass" in d
    assert "pass_" not in d
    assert d["pass"] is True
