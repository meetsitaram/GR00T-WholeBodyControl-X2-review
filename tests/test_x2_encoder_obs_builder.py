"""Layer 1 unit tests for the X2 encoder-obs builder + YAML config.

Covers the pure-Python pieces of the multi-frame inline-tokenization
path:

* :func:`gather_x2_command_multi_future_nonflat` produces a 680-D
  float32 vector with the correct (10, 68) reference layout when fed
  a synthetic planner snapshot.
* :class:`X2EncoderConfig.from_yaml` parses the canonical config that
  ships under ``gear_sonic/data/encoder/x2_observation_config.yaml``.
* :class:`X2EncoderObsBuilder` rejects unknown observations / missing
  required-observation references at construction time.

Layer 2 (parity vs. ``build_tokenizer_obs`` and
``label_trajectory``) lives in
``test_x2_dataset_recorder_real_future_token.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "gear_sonic" / "data" / "encoder" / "x2_observation_config.yaml"


from gear_sonic.utils.teleop.x2_encoder_obs_builder import (  # noqa: E402
    X2_ENCODER_OBS_DIM,
    X2_FEATURES_PER_FRAME,
    X2_NUM_BODY_DOFS,
    X2_NUM_FUTURE_FRAMES,
    X2_OBSERVATION_REGISTRY,
    X2EncoderConfig,
    X2EncoderObsBuilder,
    gather_x2_command_multi_future_nonflat,
)


# ── Fixtures ──────────────────────────────────────────────────────────────


def _synth_snapshot(seed: int = 20260513) -> dict:
    """Synthesize a planner snapshot with the layout the recorder ships.

    Matches the exact shapes :meth:`_SubscribeModeState.snapshot`
    produces: a current 31-D body_q + xyzw quat plus 9 future frames
    (the planner stride matches DT_FUTURE_REF, so total 10 frames once
    we stack current in front).
    """
    rng = np.random.default_rng(seed)
    F = X2_NUM_FUTURE_FRAMES - 1  # planner ships current + 9 future
    return {
        "body_pose_q_mj": rng.uniform(
            -0.3, 0.3, size=X2_NUM_BODY_DOFS
        ).astype(np.float64),
        "root_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "joint_pos_mj_future": rng.uniform(
            -0.3, 0.3, size=(F, X2_NUM_BODY_DOFS)
        ).astype(np.float64),
        "root_quat_xyzw_future": np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (F, 1)
        ),
    }


# ── Registry / gather ─────────────────────────────────────────────────────


def test_registry_contains_default_observation():
    """The single-modality release pins the default gather name."""
    assert "x2_command_multi_future_nonflat" in X2_OBSERVATION_REGISTRY


def test_gather_returns_correct_shape_and_dtype():
    """Pin the (680,) float32 contract the encoder ``.pt`` expects.

    The SONIC ``g1`` encoder's first layer has ``in_features=680``
    (10 frames x (31 jpos + 31 jvel + 6 ori) = 680). If this drifts
    the encoder forward pass blows up immediately, but it's worth
    pinning at the gather level so a refactor that breaks the layout
    surfaces here, not deep inside torch.
    """
    snap = _synth_snapshot()
    obs = gather_x2_command_multi_future_nonflat(snap, motion_fps=50.0)
    assert obs.shape == (X2_ENCODER_OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_gather_layout_matches_reference_10x68():
    """Reshape (10, 68) and check (jpos, jvel, ori) split holds.

    Mirrors :func:`gear_sonic.scripts.eval_x2_mujoco.build_tokenizer_obs`'s
    ``cat([command_nonflat (10, 62), ori_nonflat (10, 6)],
    axis=-1).reshape(-1)``:

    * columns 0..30 of each row -> joint positions (31 dims)
    * columns 31..61 of each row -> joint velocities (31 dims)
    * columns 62..67 of each row -> 6D rotation diff
    """
    snap = _synth_snapshot()
    obs = gather_x2_command_multi_future_nonflat(snap, motion_fps=50.0)
    grid = obs.reshape(X2_NUM_FUTURE_FRAMES, X2_FEATURES_PER_FRAME)
    assert grid.shape == (10, 68)

    jpos_block = grid[:, :X2_NUM_BODY_DOFS]
    jvel_block = grid[:, X2_NUM_BODY_DOFS : 2 * X2_NUM_BODY_DOFS]
    ori_block = grid[:, 2 * X2_NUM_BODY_DOFS :]
    assert jpos_block.shape == (10, 31)
    assert jvel_block.shape == (10, 31)
    assert ori_block.shape == (10, 6)
    # Row 0 (current frame): identity quaternion -> 6D rot diff is the
    # first two columns of the identity matrix flattened row-major:
    # [m00, m01, m10, m11, m20, m21] = [1, 0, 0, 1, 0, 0].
    np.testing.assert_allclose(
        ori_block[0], np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0]), atol=1e-6
    )


def test_gather_raises_on_missing_snapshot_field():
    """Defensive contract: missing fields should raise, not zero-fill.

    The recorder's caller is expected to gate on the planner being
    warm before calling the gather; this unit test pins the
    "obviously wrong inputs raise loudly" half of the contract so a
    silent degradation can't sneak in.
    """
    snap = _synth_snapshot()
    snap["joint_pos_mj_future"] = None
    with pytest.raises(ValueError, match="joint_pos_mj_future"):
        gather_x2_command_multi_future_nonflat(snap, motion_fps=50.0)


def test_gather_raises_on_wrong_shape():
    snap = _synth_snapshot()
    snap["body_pose_q_mj"] = np.zeros(29, dtype=np.float64)  # wrong dof count
    with pytest.raises(ValueError, match="body_pose_q_mj"):
        gather_x2_command_multi_future_nonflat(snap, motion_fps=50.0)


# ── YAML config ────────────────────────────────────────────────────────────


def test_default_yaml_loads_with_expected_fields():
    """The shipped default must parse cleanly into a config object."""
    cfg = X2EncoderConfig.from_yaml(DEFAULT_YAML)
    assert cfg.dimension == 64
    assert cfg.num_future_frames == X2_NUM_FUTURE_FRAMES
    assert pytest.approx(cfg.dt_future_ref) == 0.1
    assert pytest.approx(cfg.motion_fps) == 50.0
    assert cfg.encoder_observations == ["x2_command_multi_future_nonflat"]
    assert [m.name for m in cfg.encoder_modes] == ["retargeted_body_q"]
    assert cfg.encoder_modes[0].required_observations == [
        "x2_command_multi_future_nonflat"
    ]


def test_yaml_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        X2EncoderConfig.from_yaml(
            REPO_ROOT / "tests" / "_does_not_exist_x2_encoder.yaml"
        )


def test_yaml_disabled_observation_dropped(tmp_path):
    """An ``enabled: false`` observation should be filtered out."""
    yaml_text = """
encoder:
  dimension: 64
  motion_fps: 50.0
  dt_future_ref: 0.1
  num_future_frames: 10
  encoder_observations:
    - name: x2_command_multi_future_nonflat
      enabled: true
    - name: smpl_human_pose
      enabled: false
  encoder_modes:
    - name: retargeted_body_q
      mode_id: 0
      required_observations:
        - x2_command_multi_future_nonflat
"""
    p = tmp_path / "encoder_with_disabled.yaml"
    p.write_text(yaml_text)
    cfg = X2EncoderConfig.from_yaml(p)
    assert cfg.encoder_observations == ["x2_command_multi_future_nonflat"]


# ── Builder validation ───────────────────────────────────────────────────


def test_builder_rejects_unknown_observation(tmp_path):
    """A typo in the YAML must fail at startup, not silently pass."""
    yaml_text = """
encoder:
  dimension: 64
  motion_fps: 50.0
  dt_future_ref: 0.1
  num_future_frames: 10
  encoder_observations:
    - name: x2_command_BOGUS
      enabled: true
  encoder_modes:
    - name: retargeted_body_q
      mode_id: 0
      required_observations:
        - x2_command_BOGUS
"""
    p = tmp_path / "encoder_unknown.yaml"
    p.write_text(yaml_text)
    with pytest.raises(KeyError, match="x2_command_BOGUS"):
        X2EncoderObsBuilder.from_yaml(p)


def test_builder_rejects_required_obs_not_in_observations(tmp_path):
    """An encoder_mode cannot require an obs that isn't enabled."""
    yaml_text = """
encoder:
  dimension: 64
  motion_fps: 50.0
  dt_future_ref: 0.1
  num_future_frames: 10
  encoder_observations:
    - name: x2_command_multi_future_nonflat
      enabled: true
  encoder_modes:
    - name: retargeted_body_q
      mode_id: 0
      required_observations:
        - x2_command_multi_future_nonflat
        - x2_some_other_obs
"""
    p = tmp_path / "encoder_required_missing.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError, match="x2_some_other_obs"):
        X2EncoderObsBuilder.from_yaml(p)


def test_builder_build_obs_returns_correct_shape():
    """End-to-end: load default YAML + dispatch through registry."""
    builder = X2EncoderObsBuilder.from_yaml(DEFAULT_YAML)
    snap = _synth_snapshot()
    obs = builder.build_obs(snap)
    assert obs.shape == (X2_ENCODER_OBS_DIM,)
    assert obs.dtype == np.float32


def test_builder_unknown_mode_raises():
    builder = X2EncoderObsBuilder.from_yaml(DEFAULT_YAML)
    snap = _synth_snapshot()
    with pytest.raises(KeyError, match="not declared in YAML"):
        builder.build_obs(snap, mode="not_a_mode")
