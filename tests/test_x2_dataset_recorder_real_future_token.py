"""Real-future inline-tokenization tests (Layers 1 + 2 of the plan).

Layer 1 (no checkpoint required)
--------------------------------

* :meth:`OnlineSonicTokenizer.encode_with_snapshot` raises a clear
  error when constructed without an obs builder (so the recorder
  fall-back path doesn't silently swallow a misconfigured tokenizer).
* :meth:`OnlineSonicTokenizer.encode` (deprecated freeze-pose path)
  prints a one-shot warning on first use and stays silent afterwards.
* The recorder's :meth:`_encode_motion_token_from_snapshot` chokepoint
  delegates to the deprecated freeze-pose encoder when the tokenizer
  has no obs builder, and returns zeros when no checkpoint was loaded.

Layer 2 (gated on the real SONIC ``.pt`` checkpoint)
----------------------------------------------------

* The 680-D obs the recorder builds via ``X2EncoderObsBuilder``
  byte-equals what
  :func:`gear_sonic.scripts.eval_x2_mujoco.build_tokenizer_obs` emits
  from the same synthetic motion-lib clip (parity 2a).
* The 64-D token the recorder writes byte-equals what
  :meth:`SonicMotionTokenLabeler.label_trajectory` emits at frame 0
  for the same clip (parity 2b).
* Real-future tokens differ tick-to-tick when the planner future is
  non-trivial (vs. the freeze-pose path collapsing to a constant).

The Layer-2 tests are marked ``slow`` and skipped automatically when
the SONIC checkpoint mirror is absent (CI machines without the GPU
artifact bundle).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_YAML = REPO_ROOT / "gear_sonic" / "data" / "encoder" / "x2_observation_config.yaml"


from gear_sonic.utils.teleop.online_sonic_tokenizer import (  # noqa: E402
    SONIC_MOTION_TOKEN_DIM,
    OnlineSonicTokenizer,
)
from gear_sonic.utils.teleop.x2_dataset_recorder import (  # noqa: E402
    X2DatasetRecorder,
)
from gear_sonic.utils.teleop.x2_encoder_obs_builder import (  # noqa: E402
    X2_ENCODER_OBS_DIM,
    X2_NUM_BODY_DOFS,
    X2_NUM_FUTURE_FRAMES,
    X2EncoderObsBuilder,
)


# ── Shared fixtures ───────────────────────────────────────────────────────


def _walking_snapshot(
    seed: int = 20260513, future_offset: float = 0.05
) -> dict:
    """Synthesize a planner snapshot with non-trivial future motion.

    Adds a constant per-frame offset to the future body_q so
    ``build_tokenizer_obs`` sees nonzero joint velocities (the
    freeze-pose path would see exactly zero velocities, which is the
    distinguishing signature for the parity tests below).
    """
    rng = np.random.default_rng(seed)
    F = X2_NUM_FUTURE_FRAMES - 1  # planner ships 9 future frames
    cur = rng.uniform(-0.2, 0.2, size=X2_NUM_BODY_DOFS).astype(np.float64)
    fut = np.stack(
        [cur + (i + 1) * future_offset for i in range(F)], axis=0
    )
    return {
        "body_pose_q_mj": cur,
        "root_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        "joint_pos_mj_future": fut,
        "root_quat_xyzw_future": np.tile(
            np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), (F, 1)
        ),
    }


# ── Layer 1: no-checkpoint contract tests ────────────────────────────────


class _StubLabeler:
    """Just enough surface to instantiate :class:`OnlineSonicTokenizer`.

    The deprecated ``encode()`` path lands inside
    ``self._labeler.label_trajectory``; we never reach it in the tests
    that don't load a real checkpoint. The unbound-method tests that
    *do* call ``encode()`` patch ``label_trajectory`` to a deterministic
    stub.
    """

    def __init__(self, motion_fps: float = 50.0, device: str = "cpu") -> None:
        self.motion_fps = motion_fps
        self.device = device

    def label_trajectory(self, body_clip, *, root_rot_xyzw=None):
        T = body_clip.shape[0]
        return np.full(
            (T, SONIC_MOTION_TOKEN_DIM), 0.25, dtype=np.float64
        )


def test_encode_with_snapshot_raises_without_builder():
    """Subscribe-mode caller must use ``from_checkpoint_with_config``.

    A tokenizer constructed via plain ``from_checkpoint`` has no
    builder; calling ``encode_with_snapshot`` should fail loudly so
    the recorder's deferred fall-back path stays the *only* way to
    silently degrade to freeze-pose semantics.
    """
    tok = OnlineSonicTokenizer(_StubLabeler())
    with pytest.raises(RuntimeError, match="from_checkpoint_with_config"):
        tok.encode_with_snapshot(_walking_snapshot())


def test_encode_deprecated_warns_once(capsys):
    """The freeze-pose ``encode()`` path warns exactly once per process."""
    tok = OnlineSonicTokenizer(_StubLabeler())
    body_q = np.zeros(X2_NUM_BODY_DOFS, dtype=np.float64)
    tok.encode(body_q)
    out_first = capsys.readouterr().out
    assert "DEPRECATED encode() called" in out_first
    tok.encode(body_q)
    out_second = capsys.readouterr().out
    assert "DEPRECATED" not in out_second


# Stub recorder so we can call _encode_motion_token_from_snapshot
# without instantiating the full recorder (MuJoCo, ZMQ, robot model).
class _StubRecorder:
    def __init__(self, tokenizer=None) -> None:
        self._tokenizer = tokenizer
        self._zero_motion_token = np.zeros(
            SONIC_MOTION_TOKEN_DIM, dtype=np.float64
        )

    def _encode_motion_token(self, body_q, root_quat_xyzw=None):
        return X2DatasetRecorder._encode_motion_token.__get__(self)(
            body_q, root_quat_xyzw=root_quat_xyzw
        )

    def _encode_motion_token_from_snapshot(self, snap):
        return X2DatasetRecorder._encode_motion_token_from_snapshot.__get__(
            self
        )(snap)


def test_recorder_chokepoint_returns_zeros_without_tokenizer():
    stub = _StubRecorder(tokenizer=None)
    out = stub._encode_motion_token_from_snapshot(_walking_snapshot())
    np.testing.assert_array_equal(out, np.zeros(SONIC_MOTION_TOKEN_DIM))


def test_recorder_chokepoint_falls_back_when_no_obs_builder():
    """Without an obs_builder we must take the deprecated freeze path.

    The recorder's ``__init__`` only assigns ``self._tokenizer.obs_builder``
    when ``--encoder-config`` was provided. If the operator opts out of
    multi-frame tokenization, the chokepoint is supposed to fall back
    to ``encode()`` (which prints the one-shot deprecation warning).
    """
    stub = _StubRecorder(tokenizer=OnlineSonicTokenizer(_StubLabeler()))
    out = stub._encode_motion_token_from_snapshot(_walking_snapshot())
    # Stub labeler returns 0.25 everywhere; chokepoint should pass it
    # through verbatim.
    np.testing.assert_array_equal(
        out, np.full(SONIC_MOTION_TOKEN_DIM, 0.25)
    )


def test_recorder_chokepoint_falls_back_when_planner_not_warm():
    """Planner future may not be available before the planner is warm.

    The chokepoint should silently fall back to ``encode()`` rather
    than crashing -- the freeze-pose label is wrong, but a wrong label
    is preferable to dropping the frame entirely while the planner
    spins up.
    """
    builder = X2EncoderObsBuilder.from_yaml(DEFAULT_YAML)
    tok = OnlineSonicTokenizer(_StubLabeler(), obs_builder=builder)
    stub = _StubRecorder(tokenizer=tok)
    snap = _walking_snapshot()
    snap["joint_pos_mj_future"] = None
    out = stub._encode_motion_token_from_snapshot(snap)
    np.testing.assert_array_equal(
        out, np.full(SONIC_MOTION_TOKEN_DIM, 0.25)
    )


# ── Layer 2: parity tests gated on real SONIC checkpoint ─────────────────


@pytest.fixture(scope="module")
def _checkpoint_path() -> Path:
    from gear_sonic.scripts.record_synthetic_smoketest_dataset import (
        DEFAULT_SONIC_CHECKPOINT,
    )
    if not DEFAULT_SONIC_CHECKPOINT.exists():
        pytest.skip(
            f"SONIC checkpoint not found at {DEFAULT_SONIC_CHECKPOINT}; "
            "Layer 2 parity tests are dev-box-only."
        )
    return DEFAULT_SONIC_CHECKPOINT


@pytest.fixture(scope="module")
def _real_tokenizer(_checkpoint_path: Path) -> OnlineSonicTokenizer:
    return OnlineSonicTokenizer.from_checkpoint_with_config(
        _checkpoint_path,
        DEFAULT_YAML,
        device="cpu",
    )


@pytest.mark.slow
def test_layer2a_gather_byte_equals_build_tokenizer_obs(_real_tokenizer):
    """Recorder gather output == ``build_tokenizer_obs`` output (byte-exact).

    Plan Layer 2a: the recorder's Python gather must produce the same
    680-D observation that the offline labeler's
    :func:`build_tokenizer_obs` produces from the same motion clip.
    Without this, the inline token can diverge from the supervised
    label even when both use the same encoder weights.

    We construct the synthetic clip as ``stack([current, *future])``
    so the labeler at ``current_time=0`` reads exactly the frames the
    gather feeds it.
    """
    snap = _walking_snapshot()
    builder = _real_tokenizer.obs_builder
    assert builder is not None

    obs_recorder = builder.build_obs(snap)

    # Reference: rebuild the same clip and run build_tokenizer_obs at
    # current_time=0 (the offline path).
    import sys
    scripts_dir = REPO_ROOT / "gear_sonic" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from eval_x2_mujoco import build_tokenizer_obs  # noqa: E402

    body_clip = np.concatenate(
        [snap["body_pose_q_mj"][None, :], snap["joint_pos_mj_future"]],
        axis=0,
    )
    root_clip = np.concatenate(
        [snap["root_quat_xyzw"][None, :], snap["root_quat_xyzw_future"]],
        axis=0,
    )
    cur_quat_xyzw = root_clip[0]
    base_quat_wxyz = np.array(
        [cur_quat_xyzw[3], cur_quat_xyzw[0], cur_quat_xyzw[1], cur_quat_xyzw[2]],
        dtype=np.float64,
    )
    motion_data = {
        "x2_test_clip": {
            "dof": body_clip,
            "root_rot": root_clip,
            "fps": 50.0,
        }
    }
    obs_ref = build_tokenizer_obs(
        motion_data,
        current_time=0.0,
        base_quat_wxyz=base_quat_wxyz,
        motion_fps=50.0,
    )
    assert obs_recorder.shape == (X2_ENCODER_OBS_DIM,)
    assert obs_ref.shape == (X2_ENCODER_OBS_DIM,)
    np.testing.assert_array_equal(obs_recorder, obs_ref)


@pytest.mark.slow
def test_layer2b_token_byte_equals_label_trajectory(_real_tokenizer):
    """Recorder token == ``label_trajectory(clip)[0]`` (byte-exact).

    Plan Layer 2b: with byte-equal observations (Layer 2a) AND the
    same encoder weights, the FSQ-quantized token must also be byte-
    equal. This catches accidental dtype / device drift in the
    encoder forward pass.
    """
    snap = _walking_snapshot()
    body_clip = np.concatenate(
        [snap["body_pose_q_mj"][None, :], snap["joint_pos_mj_future"]],
        axis=0,
    )
    root_clip = np.concatenate(
        [snap["root_quat_xyzw"][None, :], snap["root_quat_xyzw_future"]],
        axis=0,
    )
    token_recorder = _real_tokenizer.encode_with_snapshot(snap)
    tokens_ref = _real_tokenizer.labeler.label_trajectory(
        body_clip, root_rot_xyzw=root_clip
    )
    np.testing.assert_array_equal(token_recorder, tokens_ref[0])


@pytest.mark.slow
def test_layer1_real_future_tokens_differ_row_to_row(_real_tokenizer):
    """Distinct planner snapshots must produce distinct tokens.

    The single signature of the freeze-pose path is that it collapses
    every snapshot of "the same body_q" to the same token regardless
    of the future. With the real-future path, varying the future
    window must move the token off that single FSQ bucket. We require
    >=3 distinct tokens across 12 snapshots to keep the test robust
    against the encoder picking semantically similar buckets for
    nearby trajectories.
    """
    rng = np.random.default_rng(seed=20260513)
    base = rng.uniform(-0.2, 0.2, size=X2_NUM_BODY_DOFS).astype(np.float64)
    F = X2_NUM_FUTURE_FRAMES - 1
    tokens = []
    for i in range(12):
        offset = (i + 1) * 0.04
        snap = {
            "body_pose_q_mj": base.copy(),
            "root_quat_xyzw": np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float64
            ),
            "joint_pos_mj_future": np.stack(
                [base + j * offset for j in range(1, F + 1)], axis=0
            ),
            "root_quat_xyzw_future": np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                (F, 1),
            ),
        }
        tokens.append(_real_tokenizer.encode_with_snapshot(snap))
    distinct = {tuple(t.tolist()) for t in tokens}
    assert len(distinct) >= 3, (
        f"only {len(distinct)} distinct tokens across 12 varied futures; "
        "encoder may have collapsed the multi-frame signal."
    )


@pytest.mark.slow
def test_layer1_real_future_differs_from_freeze_pose(_real_tokenizer):
    """Real-future token != freeze-pose token for the same body_q.

    Pins the *semantic* difference between the two paths: a non-trivial
    future window must move the token off whatever FSQ bucket the
    freeze-pose tile lands in. Otherwise the multi-frame plumbing is
    decorative.
    """
    snap = _walking_snapshot()
    real_tok = _real_tokenizer.encode_with_snapshot(snap)
    freeze_tok = _real_tokenizer.encode(snap["body_pose_q_mj"])
    assert not np.array_equal(real_tok, freeze_tok), (
        "real-future and freeze-pose tokens are identical -- the "
        "future window had no effect on the FSQ output, which means "
        "the gather is broken or the future is being silently dropped."
    )
