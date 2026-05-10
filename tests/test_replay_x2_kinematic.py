"""Tests for :mod:`gear_sonic.scripts.replay_x2_kinematic`.

These tests exercise the CLI's pure helpers (arg parsing, dataset
resolution, chunk-path math, parquet validation) without ever
launching the MuJoCo viewer. The viewer-launch path
(:func:`_run_viewer`) is exercised by the manual smoke test described
in the tutorial; it cannot run headlessly under CI without a display.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from gear_sonic.scripts.replay_x2_kinematic import (
    _DEFAULT_CHUNK_SIZE,
    _episode_parquet_path,
    _load_and_validate_parquet,
    _parse_args,
    _read_chunk_size,
    _resolve_dataset_path,
    _resolve_frame_window,
)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def test_parse_args_defaults() -> None:
    args = _parse_args(["--dataset", "x2_quest3_kinematic_v4", "--episode", "0"])
    assert args.dataset == "x2_quest3_kinematic_v4"
    assert args.episode == 0
    assert args.robot == "x2"
    assert args.rate == pytest.approx(50.0)
    assert args.start_frame == 0
    assert args.end_frame == -1
    assert args.loop is False
    assert args.with_omnihand is True
    assert args.quiet is False


def test_parse_args_overrides() -> None:
    args = _parse_args([
        "--dataset", "/abs/path",
        "--episode", "5",
        "--robot", "g1",
        "--rate", "25.0",
        "--start-frame", "100",
        "--end-frame", "500",
        "--loop",
        "--no-omnihand",
        "--quiet",
    ])
    assert args.dataset == "/abs/path"
    assert args.episode == 5
    assert args.robot == "g1"
    assert args.rate == pytest.approx(25.0)
    assert args.start_frame == 100
    assert args.end_frame == 500
    assert args.loop is True
    assert args.with_omnihand is False
    assert args.quiet is True


def test_parse_args_dataset_and_episode_now_optional() -> None:
    """Dataset / episode are no longer argparse-required; the
    either-or constraint with --parquet is enforced inside ``main``."""
    args = _parse_args([])
    assert args.dataset is None
    assert args.episode is None
    assert args.parquet is None


def test_parse_args_accepts_parquet_override(tmp_path: Path) -> None:
    fake_parquet = tmp_path / "fake.parquet"
    fake_parquet.write_text("")  # contents irrelevant for arg parsing
    args = _parse_args(["--parquet", str(fake_parquet)])
    assert args.parquet == fake_parquet
    assert args.dataset is None
    assert args.episode is None


# ---------------------------------------------------------------------------
# Dataset path resolution
# ---------------------------------------------------------------------------


def test_resolve_dataset_path_accepts_absolute_path(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    resolved = _resolve_dataset_path(str(tmp_path))
    assert resolved == tmp_path.resolve()


def test_resolve_dataset_path_resolves_short_name() -> None:
    """The recorded v4 dataset lives under data/lerobot/ and should resolve."""
    repo_root = Path(__file__).resolve().parent.parent
    target = repo_root / "data" / "lerobot" / "x2_quest3_kinematic_v4"
    if not target.is_dir():
        pytest.skip(f"sample dataset missing at {target}")
    resolved = _resolve_dataset_path("x2_quest3_kinematic_v4")
    assert resolved == target.resolve()


def test_resolve_dataset_path_unknown_raises() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        _resolve_dataset_path("__definitely_not_a_real_dataset__")


# ---------------------------------------------------------------------------
# Chunk path math
# ---------------------------------------------------------------------------


def test_episode_parquet_path_chunk_zero() -> None:
    root = Path("/datasets/x2_v0")
    p = _episode_parquet_path(root, 0)
    assert p == root / "data" / "chunk-000" / "episode_000000.parquet"


def test_episode_parquet_path_high_episode_default_chunk() -> None:
    root = Path("/datasets/x2_v0")
    p = _episode_parquet_path(root, 1500)
    # 1500 // 1000 = 1 -> chunk-001
    assert p == root / "data" / "chunk-001" / "episode_001500.parquet"


def test_episode_parquet_path_custom_chunk_size() -> None:
    root = Path("/datasets/x2_v0")
    p = _episode_parquet_path(root, 250, chunk_size=100)
    # 250 // 100 = 2 -> chunk-002
    assert p == root / "data" / "chunk-002" / "episode_000250.parquet"


def test_episode_parquet_path_negative_episode_rejected() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        _episode_parquet_path(Path("/foo"), -1)


def test_read_chunk_size_uses_meta_info_when_present(tmp_path: Path) -> None:
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir()
    (meta_dir / "info.json").write_text(json.dumps({"chunks_size": 250}))
    assert _read_chunk_size(tmp_path) == 250


def test_read_chunk_size_falls_back_when_meta_missing(tmp_path: Path) -> None:
    assert _read_chunk_size(tmp_path) == _DEFAULT_CHUNK_SIZE


def test_read_chunk_size_falls_back_on_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{{not json}}")
    assert _read_chunk_size(tmp_path) == _DEFAULT_CHUNK_SIZE


def test_read_chunk_size_falls_back_on_non_int(tmp_path: Path) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text(json.dumps({"chunks_size": "huge"}))
    assert _read_chunk_size(tmp_path) == _DEFAULT_CHUNK_SIZE


# ---------------------------------------------------------------------------
# Frame window
# ---------------------------------------------------------------------------


def test_resolve_frame_window_default_end() -> None:
    assert _resolve_frame_window(100, start_frame=0, end_frame=-1) == (0, 100)


def test_resolve_frame_window_clamps() -> None:
    assert _resolve_frame_window(100, start_frame=-5, end_frame=200) == (0, 100)


def test_resolve_frame_window_partial() -> None:
    assert _resolve_frame_window(100, start_frame=20, end_frame=80) == (20, 80)


def test_resolve_frame_window_empty_raises() -> None:
    with pytest.raises(ValueError, match="Empty frame window"):
        _resolve_frame_window(100, start_frame=50, end_frame=10)


def test_resolve_frame_window_zero_frames_raises() -> None:
    with pytest.raises(ValueError, match="zero frames"):
        _resolve_frame_window(0, start_frame=0, end_frame=-1)


# ---------------------------------------------------------------------------
# Parquet validation
# ---------------------------------------------------------------------------


def _write_fixture_parquet(
    path: Path,
    *,
    num_frames: int,
    body_dim: int,
    hand_dim: int,
    skip_columns: tuple[str, ...] = (),
    bad_body_width: int | None = None,
    bad_left_width: int | None = None,
) -> None:
    """Synthesize a minimal LeRobot-shaped parquet for unit tests."""
    body_w = bad_body_width if bad_body_width is not None else body_dim
    left_w = bad_left_width if bad_left_width is not None else hand_dim
    rng = np.random.default_rng(0)

    cols: dict[str, list] = {}
    if "action.commanded_body_q_mj" not in skip_columns:
        cols["action.commanded_body_q_mj"] = list(
            rng.standard_normal((num_frames, body_w)).astype(np.float64)
        )
    if "action.left_hand_joints" not in skip_columns:
        cols["action.left_hand_joints"] = list(
            rng.standard_normal((num_frames, left_w)).astype(np.float64)
        )
    if "action.right_hand_joints" not in skip_columns:
        cols["action.right_hand_joints"] = list(
            rng.standard_normal((num_frames, hand_dim)).astype(np.float64)
        )

    table = pa.table(cols)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)


def test_load_and_validate_parquet_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "ep.parquet"
    _write_fixture_parquet(p, num_frames=50, body_dim=31, hand_dim=10)
    body, left, right = _load_and_validate_parquet(
        p,
        num_body_dofs=31,
        num_hand_dof_per_side=10,
        require_omnihand=True,
    )
    assert body.shape == (50, 31)
    assert left is not None and left.shape == (50, 10)
    assert right is not None and right.shape == (50, 10)


def test_load_and_validate_parquet_no_omnihand(tmp_path: Path) -> None:
    p = tmp_path / "ep.parquet"
    _write_fixture_parquet(p, num_frames=10, body_dim=31, hand_dim=10)
    body, left, right = _load_and_validate_parquet(
        p,
        num_body_dofs=31,
        num_hand_dof_per_side=10,
        require_omnihand=False,
    )
    assert body.shape == (10, 31)
    assert left is None and right is None


def test_load_and_validate_parquet_missing_columns_raises(tmp_path: Path) -> None:
    p = tmp_path / "ep.parquet"
    _write_fixture_parquet(
        p, num_frames=5, body_dim=31, hand_dim=10,
        skip_columns=("action.left_hand_joints",),
    )
    with pytest.raises(ValueError, match="missing required columns"):
        _load_and_validate_parquet(
            p, num_body_dofs=31, num_hand_dof_per_side=10, require_omnihand=True,
        )


def test_load_and_validate_parquet_wrong_body_width_raises(tmp_path: Path) -> None:
    p = tmp_path / "ep.parquet"
    _write_fixture_parquet(
        p, num_frames=5, body_dim=31, hand_dim=10, bad_body_width=29,
    )
    with pytest.raises(ValueError, match="action.commanded_body_q_mj"):
        _load_and_validate_parquet(
            p, num_body_dofs=31, num_hand_dof_per_side=10, require_omnihand=True,
        )


def test_load_and_validate_parquet_wrong_hand_width_raises(tmp_path: Path) -> None:
    p = tmp_path / "ep.parquet"
    _write_fixture_parquet(
        p, num_frames=5, body_dim=31, hand_dim=10, bad_left_width=7,
    )
    with pytest.raises(ValueError, match="action.left_hand_joints"):
        _load_and_validate_parquet(
            p, num_body_dofs=31, num_hand_dof_per_side=10, require_omnihand=True,
        )


def test_load_and_validate_parquet_missing_file_raises(tmp_path: Path) -> None:
    p = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError, match="not found"):
        _load_and_validate_parquet(
            p, num_body_dofs=31, num_hand_dof_per_side=10, require_omnihand=True,
        )


# ---------------------------------------------------------------------------
# main() either-or contract
# ---------------------------------------------------------------------------


def _make_fixture_dataset_with_parquet(
    tmp_path: Path,
    *,
    num_frames: int = 5,
    body_dim: int = 31,
    hand_dim: int = 10,
) -> tuple[Path, Path]:
    """Build a tiny LeRobot-shaped dataset on disk for main() smoke tests."""
    dataset_root = tmp_path / "fixture_ds"
    parquet = dataset_root / "data" / "chunk-000" / "episode_000000.parquet"
    _write_fixture_parquet(parquet, num_frames=num_frames, body_dim=body_dim, hand_dim=hand_dim)
    return dataset_root, parquet


def test_main_rejects_when_neither_parquet_nor_dataset_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear_sonic.scripts import replay_x2_kinematic as mod

    monkeypatch.setattr(mod, "_run_viewer", lambda **kw: None)
    with pytest.raises(SystemExit, match="required"):
        mod.main(["--quiet"])


def test_main_uses_parquet_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear_sonic.scripts import replay_x2_kinematic as mod

    parquet = tmp_path / "override.parquet"
    _write_fixture_parquet(parquet, num_frames=4, body_dim=31, hand_dim=10)

    seen: dict[str, object] = {}

    def fake_viewer(**kw):
        seen["body_shape"] = kw["body_q"].shape
        seen["left_shape"] = kw["left_q"].shape
        seen["right_shape"] = kw["right_q"].shape

    monkeypatch.setattr(mod, "_run_viewer", fake_viewer)
    rc = mod.main(["--parquet", str(parquet), "--quiet"])
    assert rc == 0
    assert seen["body_shape"] == (4, 31)
    assert seen["left_shape"] == (4, 10)
    assert seen["right_shape"] == (4, 10)


def test_main_parquet_override_missing_file_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear_sonic.scripts import replay_x2_kinematic as mod

    monkeypatch.setattr(mod, "_run_viewer", lambda **kw: None)
    missing = tmp_path / "nope.parquet"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        mod.main(["--parquet", str(missing), "--quiet"])


def test_main_dataset_episode_path_still_works(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gear_sonic.scripts import replay_x2_kinematic as mod

    dataset_root, _ = _make_fixture_dataset_with_parquet(tmp_path, num_frames=6)
    monkeypatch.setattr(mod, "_run_viewer", lambda **kw: None)
    rc = mod.main([
        "--dataset", str(dataset_root),
        "--episode", "0",
        "--quiet",
    ])
    assert rc == 0


def test_load_and_validate_parquet_against_recorded_v4(
    tmp_path: Path,  # noqa: ARG001
) -> None:
    """Round-trip through a real recorded episode using the X2 EmbodimentConfig."""
    repo_root = Path(__file__).resolve().parent.parent
    parquet = (
        repo_root
        / "data" / "lerobot" / "x2_quest3_kinematic_v4"
        / "data" / "chunk-000" / "episode_000000.parquet"
    )
    if not parquet.is_file():
        pytest.skip(f"sample dataset missing at {parquet}")

    from gear_sonic.utils.embodiment import get_embodiment

    cfg = get_embodiment("x2")
    body, left, right = _load_and_validate_parquet(
        parquet,
        num_body_dofs=cfg.num_body_dofs,
        num_hand_dof_per_side=cfg.num_hand_dof_per_side,
        require_omnihand=True,
    )
    assert body.shape[1] == cfg.num_body_dofs
    assert left is not None and left.shape[1] == cfg.num_hand_dof_per_side
    assert right is not None and right.shape[1] == cfg.num_hand_dof_per_side
    assert body.shape[0] == left.shape[0] == right.shape[0]
