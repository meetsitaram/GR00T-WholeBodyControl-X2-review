"""Unit tests for the X2 heuristic-planner primitive curator.

Synthetic motion-lib fixtures exercise:

  - constants & joint indices match ``eval_x2_mujoco`` (drift detector)
  - per-window metrics (XY, yaw, end-at-square, feet-planted, waist apex)
  - per-bin scoring & pass/fail gates (locomotion + static_upper_body + idle)
  - pin handling (pinned rows are preserved + re-measured but not searched)
  - report generation has expected sections / bin rows
  - PKL output schema matches what the runtime planner consumes

Runs without MuJoCo / Isaac Lab / GPUs::

    .venv/bin/python -m pytest tests/test_x2_planner_curator.py -v
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts import curate_x2_primitives as cur  # noqa: E402
from gear_sonic.utils.planner import constants as const  # noqa: E402
from gear_sonic.utils.planner.metrics import compute_window_metrics  # noqa: E402
from gear_sonic.utils.planner.registry import (  # noqa: E402
    BinSpec,
    PrimitiveEntry,
    load_bin_specs,
    load_primitive_registry,
    write_primitive_registry,
)


# ---------------------------------------------------------------------------
# Constants drift detector
# ---------------------------------------------------------------------------


def test_constants_match_eval_x2_mujoco() -> None:
    """If eval_x2_mujoco's joint order changes, this catches it on next CI."""
    eval_path = REPO_ROOT / "gear_sonic" / "scripts" / "eval_x2_mujoco.py"
    text = eval_path.read_text()
    block_start = text.index("MUJOCO_JOINT_NAMES = [")
    block_end = text.index("]", block_start)
    block = text[block_start:block_end]
    parsed_names: list[str] = []
    for chunk in block.split('"'):
        if chunk.endswith("_joint"):
            parsed_names.append(chunk)
    assert tuple(parsed_names) == const.MUJOCO_JOINT_NAMES, (
        f"MUJOCO_JOINT_NAMES drift between planner constants and eval_x2_mujoco: "
        f"{parsed_names} vs {list(const.MUJOCO_JOINT_NAMES)}"
    )

    assert const.WAIST_YAW_IDX == const.MUJOCO_JOINT_NAMES.index("waist_yaw_joint")
    assert const.WAIST_PITCH_IDX == const.MUJOCO_JOINT_NAMES.index("waist_pitch_joint")
    assert const.WAIST_ROLL_IDX == const.MUJOCO_JOINT_NAMES.index("waist_roll_joint")

    # Counter-balance index lookups used by op_synthesize_waist_ramp's
    # hip_pitch_share / hip_yaw_share / ankle_roll_share knobs.
    assert const.LEFT_HIP_YAW_IDX == const.MUJOCO_JOINT_NAMES.index("left_hip_yaw_joint")
    assert const.RIGHT_HIP_YAW_IDX == const.MUJOCO_JOINT_NAMES.index("right_hip_yaw_joint")
    assert const.LEFT_HIP_ROLL_IDX == const.MUJOCO_JOINT_NAMES.index("left_hip_roll_joint")
    assert const.RIGHT_HIP_ROLL_IDX == const.MUJOCO_JOINT_NAMES.index("right_hip_roll_joint")
    assert const.LEFT_ANKLE_ROLL_IDX == const.MUJOCO_JOINT_NAMES.index("left_ankle_roll_joint")
    assert const.RIGHT_ANKLE_ROLL_IDX == const.MUJOCO_JOINT_NAMES.index("right_ankle_roll_joint")


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _identity_quat_xyzw(n: int) -> np.ndarray:
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 3] = 1.0
    return q


def _quat_from_yaw_xyzw(yaw_rad: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)],
        dtype=np.float32,
    )


def _make_clip_idle(n: int = 90, fps: float = 30.0) -> dict:
    """Stationary stand pose."""
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    root_rot = _identity_quat_xyzw(n)
    root_trans = np.zeros((n, 3), dtype=np.float32)
    root_trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return {
        "dof": dof,
        "root_rot": root_rot,
        "root_trans_offset": root_trans,
        "fps": fps,
    }


def _make_clip_fwd_step(
    distance_m: float, n: int = 60, fps: float = 30.0
) -> dict:
    """One step forward by ``distance_m``, ending at square stance.

    Modulates the leg DOFs in a sinusoidal swing (for stride-count = 1) and
    forces the last frame to a perfect mirror so end_at_square_score is high.
    """
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    t = np.linspace(0.0, math.pi, n)  # half-cycle of sin
    swing = 0.25 * np.sin(t)
    dof[:, const.LEFT_HIP_PITCH_IDX] += swing
    dof[:, const.RIGHT_HIP_PITCH_IDX] -= swing
    dof[:, const.LEFT_KNEE_IDX] += 0.4 * swing
    dof[:, const.RIGHT_KNEE_IDX] += 0.4 * swing
    # Snap last frame back to default for clean end-at-square.
    dof[-1, :] = const.DEFAULT_STAND_POSE_NP

    root_rot = _identity_quat_xyzw(n)
    root_trans = np.zeros((n, 3), dtype=np.float32)
    root_trans[:, 0] = np.linspace(0.0, distance_m, n)
    root_trans[:, 2] = const.DEFAULT_PELVIS_Z_M + 0.005 * np.sin(t)
    return {
        "dof": dof,
        "root_rot": root_rot,
        "root_trans_offset": root_trans,
        "fps": fps,
    }


def _make_clip_in_place_turn(
    yaw_deg: float, n: int = 60, fps: float = 30.0
) -> dict:
    """Pivot in place by ``yaw_deg`` degrees (signed), no XY drift."""
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    # Tiny leg motion to make stride detector happy without hurting end-at-square.
    t = np.linspace(0.0, math.pi, n)
    swing = 0.10 * np.sin(t)
    dof[:, const.LEFT_HIP_PITCH_IDX] += swing
    dof[:, const.RIGHT_HIP_PITCH_IDX] -= swing
    dof[-1, :] = const.DEFAULT_STAND_POSE_NP

    yaws_rad = np.linspace(0.0, math.radians(yaw_deg), n)
    root_rot = np.zeros((n, 4), dtype=np.float32)
    for i, y in enumerate(yaws_rad):
        root_rot[i] = _quat_from_yaw_xyzw(float(y))
    root_trans = np.zeros((n, 3), dtype=np.float32)
    root_trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return {
        "dof": dof,
        "root_rot": root_rot,
        "root_trans_offset": root_trans,
        "fps": fps,
    }


def _make_clip_lean_fwd(
    waist_pitch_deg: float, n: int = 45, fps: float = 30.0
) -> dict:
    """Static lean: feet planted, waist pitch ramps to apex and HOLDS."""
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    pitch_rad = math.radians(waist_pitch_deg)
    n_ramp = n // 2
    n_hold = n - n_ramp
    ramp = np.linspace(0.0, pitch_rad, n_ramp)
    dof[:n_ramp, const.WAIST_PITCH_IDX] += ramp
    dof[n_ramp:, const.WAIST_PITCH_IDX] += pitch_rad
    root_rot = _identity_quat_xyzw(n)
    root_trans = np.zeros((n, 3), dtype=np.float32)
    root_trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return {
        "dof": dof,
        "root_rot": root_rot,
        "root_trans_offset": root_trans,
        "fps": fps,
    }


def _make_clip_torso_left(
    waist_yaw_deg: float, n: int = 45, fps: float = 30.0
) -> dict:
    """Static twist: feet planted, waist yaw ramps to apex (signed)."""
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    yaw_rad = math.radians(waist_yaw_deg)
    n_ramp = n // 2
    ramp = np.linspace(0.0, yaw_rad, n_ramp)
    dof[:n_ramp, const.WAIST_YAW_IDX] += ramp
    dof[n_ramp:, const.WAIST_YAW_IDX] += yaw_rad
    root_rot = _identity_quat_xyzw(n)
    root_trans = np.zeros((n, 3), dtype=np.float32)
    root_trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return {
        "dof": dof,
        "root_rot": root_rot,
        "root_trans_offset": root_trans,
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# WindowMetrics
# ---------------------------------------------------------------------------


def test_metrics_idle_clip_is_loopable_and_planted() -> None:
    clip = _make_clip_idle(90)
    m = compute_window_metrics(
        clip["dof"], clip["root_rot"], clip["root_trans_offset"], clip["fps"]
    )
    assert m.n_frames == 90
    assert abs(m.net_yaw_deg) < 0.1
    assert float(np.linalg.norm(m.net_xy_body_m)) < 1e-4
    assert m.loop_dof_drift < 1e-4
    assert m.loop_quat_distance < 1e-4
    assert m.feet_planted_score > 0.95
    assert m.stride_count == 0
    assert m.waist_pitch_apex_deg < 0.1
    assert m.end_at_apex_score == 1.0  # no waist motion → trivially "at apex"


def test_metrics_fwd_step_distance_in_body_frame() -> None:
    clip = _make_clip_fwd_step(distance_m=0.30)
    m = compute_window_metrics(
        clip["dof"], clip["root_rot"], clip["root_trans_offset"], clip["fps"]
    )
    assert m.net_xy_body_m[0] == pytest.approx(0.30, abs=1e-3)
    assert abs(m.net_xy_body_m[1]) < 1e-3
    assert abs(m.net_yaw_deg) < 0.1
    assert m.end_at_square_score > 0.95  # snapped last frame
    assert m.stride_count >= 1


def test_metrics_in_place_turn_yaw_signed() -> None:
    clip_left = _make_clip_in_place_turn(yaw_deg=45.0)
    m = compute_window_metrics(
        clip_left["dof"], clip_left["root_rot"], clip_left["root_trans_offset"], clip_left["fps"]
    )
    assert m.net_yaw_deg == pytest.approx(45.0, abs=0.5)
    assert float(np.linalg.norm(m.net_xy_body_m)) < 1e-3

    clip_right = _make_clip_in_place_turn(yaw_deg=-30.0)
    m_r = compute_window_metrics(
        clip_right["dof"], clip_right["root_rot"], clip_right["root_trans_offset"], clip_right["fps"]
    )
    assert m_r.net_yaw_deg == pytest.approx(-30.0, abs=0.5)


def test_metrics_lean_fwd_apex_and_planted() -> None:
    clip = _make_clip_lean_fwd(waist_pitch_deg=20.0)
    m = compute_window_metrics(
        clip["dof"], clip["root_rot"], clip["root_trans_offset"], clip["fps"]
    )
    assert m.waist_pitch_apex_deg == pytest.approx(20.0, abs=0.2)
    # Last frame is at apex (held), so end_at_apex_score should be ~1.0.
    assert m.end_at_apex_score > 0.99
    assert m.feet_planted_score > 0.95


def test_metrics_torso_left_unsigned_apex_with_signed_dof() -> None:
    clip = _make_clip_torso_left(waist_yaw_deg=30.0)
    m = compute_window_metrics(
        clip["dof"], clip["root_rot"], clip["root_trans_offset"], clip["fps"]
    )
    assert m.waist_yaw_apex_deg == pytest.approx(30.0, abs=0.2)
    # Direction filter at the curator level uses the signed last-frame DOF.
    assert clip["dof"][-1, const.WAIST_YAW_IDX] > 0


# ---------------------------------------------------------------------------
# Bin scoring
# ---------------------------------------------------------------------------


def _spec_fwd_step_half() -> BinSpec:
    return BinSpec(
        name="fwd_step_half_ft",
        family="locomotion",
        target_intent="fwd_step",
        target_magnitude="half_ft",
        target_xy_m=(0.1524, 0.0),
        target_yaw_deg=0.0,
        tol_xy_m=0.05,
        tol_yaw_deg=4.0,
        cross_axis_max_m=0.05,
        pelvis_z_band_m=(0.62, 0.80),
        end_at_square_min=0.4,
        stride_count_target=1,
        window_frames_min=30,
        window_frames_max=90,
    )


def _spec_lean_fwd_medium() -> BinSpec:
    return BinSpec(
        name="lean_fwd_medium",
        family="static_upper_body",
        target_intent="lean_fwd",
        target_magnitude="medium",
        target_waist_pitch_deg=20.0,
        tol_waist_deg=6.0,
        pelvis_z_band_m=(0.68, 0.80),
        end_at_apex_min=0.7,
        feet_planted_min=0.4,
        stride_count_target=0,
        window_frames_min=30,
        window_frames_max=90,
    )


def _spec_idle() -> BinSpec:
    return BinSpec(
        name="idle_stand",
        family="idle",
        target_intent="idle",
        target_magnitude="default",
        tol_xy_m=0.04,
        tol_yaw_deg=3.0,
        pelvis_z_band_m=(0.70, 0.80),
        end_at_square_min=0.5,
        feet_planted_min=0.5,
        stride_count_target=0,
        window_frames_min=45,
        window_frames_max=150,
    )


def test_score_fwd_step_perfect_match_passes() -> None:
    spec = _spec_fwd_step_half()
    clip = _make_clip_fwd_step(distance_m=0.1524)
    m = compute_window_metrics(
        clip["dof"], clip["root_rot"], clip["root_trans_offset"], clip["fps"]
    )
    s = cur.score_window(spec, m)
    assert s.passes, f"expected pass; sub_scores={s.sub_scores}, gates={s.pass_fail}"
    assert s.score > 0.5


def test_score_fwd_step_off_target_fails_gate() -> None:
    spec = _spec_fwd_step_half()
    clip = _make_clip_fwd_step(distance_m=0.50)  # way too far
    m = compute_window_metrics(
        clip["dof"], clip["root_rot"], clip["root_trans_offset"], clip["fps"]
    )
    s = cur.score_window(spec, m)
    assert not s.passes
    assert not s.pass_fail["xy_along"]


def test_score_lean_fwd_perfect_match_passes() -> None:
    spec = _spec_lean_fwd_medium()
    clip = _make_clip_lean_fwd(waist_pitch_deg=20.0)
    m = compute_window_metrics(
        clip["dof"], clip["root_rot"], clip["root_trans_offset"], clip["fps"]
    )
    s = cur.score_window(spec, m)
    assert s.passes, f"sub={s.sub_scores}, gates={s.pass_fail}"


def test_score_idle_static_pose_passes() -> None:
    spec = _spec_idle()
    clip = _make_clip_idle(120)
    m = compute_window_metrics(
        clip["dof"], clip["root_rot"], clip["root_trans_offset"], clip["fps"]
    )
    s = cur.score_window(spec, m)
    assert s.passes, f"sub={s.sub_scores}, gates={s.pass_fail}"


# ---------------------------------------------------------------------------
# End-to-end curator + outputs
# ---------------------------------------------------------------------------


def _build_synthetic_corpus(out_pkl: Path) -> str:
    """Build a 6-clip synthetic motion library."""
    library = {
        "synth__idle_stand_loop_001": _make_clip_idle(120),
        "synth__fwd_step_half_ft_clean_001": _make_clip_fwd_step(0.1524, 60),
        "synth__fwd_step_one_ft_clean_001": _make_clip_fwd_step(0.3048, 75),
        "synth__turn_left_45deg_clean_001": _make_clip_in_place_turn(45.0, 75),
        "synth__lean_fwd_medium_clean_001": _make_clip_lean_fwd(20.0, 60),
        "synth__torso_left_30deg_clean_001": _make_clip_torso_left(30.0, 60),
    }
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(library, out_pkl)
    return str(out_pkl)


def _write_synth_bins_yaml(path: Path) -> None:
    yaml_body = """
bins:
  - name: idle_stand
    family: idle
    target_intent: idle
    target_magnitude: default
    target_xy_m: [0.0, 0.0]
    target_yaw_deg: 0.0
    tol_xy_m: 0.04
    tol_yaw_deg: 3.0
    cross_axis_max_m: 0.04
    pelvis_z_band_m: [0.70, 0.80]
    end_at_square_min: 0.5
    feet_planted_min: 0.5
    stride_count_target: 0
    window_frames_min: 60
    window_frames_max: 120
    name_regex: "(?i)idle"

  - name: fwd_step_half_ft
    family: locomotion
    target_intent: fwd_step
    target_magnitude: half_ft
    target_xy_m: [0.1524, 0.0]
    target_yaw_deg: 0.0
    tol_xy_m: 0.05
    tol_yaw_deg: 4.0
    cross_axis_max_m: 0.05
    pelvis_z_band_m: [0.62, 0.80]
    end_at_square_min: 0.4
    stride_count_target: 1
    window_frames_min: 30
    window_frames_max: 75

  - name: turn_left_45deg
    family: locomotion
    target_intent: turn_left
    target_magnitude: deg_45
    target_xy_m: [0.0, 0.0]
    target_yaw_deg: 45.0
    tol_xy_m: 0.06
    tol_yaw_deg: 5.0
    cross_axis_max_m: 0.06
    pelvis_z_band_m: [0.62, 0.80]
    end_at_square_min: 0.4
    stride_count_target: null
    window_frames_min: 45
    window_frames_max: 90

  - name: lean_fwd_medium
    family: static_upper_body
    target_intent: lean_fwd
    target_magnitude: medium
    target_waist_pitch_deg: 20.0
    tol_waist_deg: 6.0
    pelvis_z_band_m: [0.68, 0.80]
    end_at_apex_min: 0.7
    feet_planted_min: 0.4
    stride_count_target: 0
    window_frames_min: 30
    window_frames_max: 75

  - name: torso_left_30deg
    family: static_upper_body
    target_intent: torso_left
    target_magnitude: deg_30
    target_waist_yaw_deg: 30.0
    tol_waist_deg: 6.0
    pelvis_z_band_m: [0.70, 0.80]
    end_at_apex_min: 0.7
    feet_planted_min: 0.4
    stride_count_target: 0
    window_frames_min: 30
    window_frames_max: 75
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_body)


def test_end_to_end_curator_run(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    source = tmp_path / "src" / "synth.pkl"
    bins = tmp_path / "spec" / "x2_planner_bins.yaml"
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)

    _build_synthetic_corpus(source)
    _write_synth_bins_yaml(bins)

    rc = cur.main(
        [
            "--source", str(source),
            "--bins", str(bins),
            "--out", str(out_dir),
            "--top-k", "5",
        ]
    )
    assert rc == 0

    pkl_path = out_dir / "x2_planner_primitives.pkl"
    yaml_path = out_dir / "x2_planner_primitives.yaml"
    md_path = out_dir / "x2_planner_primitives_report.md"
    assert pkl_path.exists()
    assert yaml_path.exists()
    assert md_path.exists()

    # PKL schema
    primitives = joblib.load(pkl_path)
    expected_bins = {
        "idle_stand", "fwd_step_half_ft", "turn_left_45deg",
        "lean_fwd_medium", "torso_left_30deg",
    }
    assert set(primitives.keys()) == expected_bins
    for name, payload in primitives.items():
        for k in ("dof", "root_rot_xyzw", "root_trans", "fps", "source_pkl",
                  "motion_key", "start_frame", "n_frames"):
            assert k in payload, f"{name} payload missing key {k}"
        assert payload["dof"].shape[1] == 31
        assert payload["root_rot_xyzw"].shape[1] == 4
        assert payload["root_trans"].shape[1] == 3
        assert payload["dof"].shape[0] == payload["n_frames"]
        assert payload["dof"].shape[0] == payload["root_rot_xyzw"].shape[0]
        assert payload["dof"].shape[0] == payload["root_trans"].shape[0]

    # Registry rows present
    registry = load_primitive_registry(yaml_path)
    assert set(registry.keys()) == expected_bins

    # Report has expected sections
    md = md_path.read_text()
    for bin_name in expected_bins:
        assert f"`{bin_name}`" in md, f"report missing bin section for {bin_name}"
    assert "Bin status summary" in md
    assert "Per-bin candidates" in md

    # CLI summary line
    captured = capsys.readouterr()
    assert "DONE" in captured.out


def test_pinned_row_is_preserved_and_remeasured(tmp_path: Path) -> None:
    """Pinned registry rows survive re-runs even when a 'better' candidate exists."""
    source = tmp_path / "src.pkl"
    bins = tmp_path / "bins.yaml"
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    _build_synthetic_corpus(source)
    _write_synth_bins_yaml(bins)

    cur.main(["--source", str(source), "--bins", str(bins), "--out", str(out_dir)])

    yaml_path = out_dir / "x2_planner_primitives.yaml"
    registry = load_primitive_registry(yaml_path)
    # Hand-pin the lean clip to a different motion_key (the half-step one) and
    # restrict frames so the pin is unambiguously not what the curator would
    # have picked. This still must round-trip on re-run.
    registry["lean_fwd_medium"] = PrimitiveEntry(
        bin_name="lean_fwd_medium",
        source_pkl=str(source),
        motion_key="synth__lean_fwd_medium_clean_001",
        start_frame=10,
        n_frames=30,
        fps=30.0,
        partial=True,
        pinned=True,
        notes="hand-pinned for test",
    )
    write_primitive_registry(yaml_path, registry)

    # Re-run; pin must be preserved (same key+window) and re-measured.
    cur.main(["--source", str(source), "--bins", str(bins), "--out", str(out_dir)])

    registry2 = load_primitive_registry(yaml_path)
    pinned_row = registry2["lean_fwd_medium"]
    assert pinned_row.pinned is True
    assert pinned_row.motion_key == "synth__lean_fwd_medium_clean_001"
    assert pinned_row.start_frame == 10
    assert pinned_row.n_frames == 30
    assert pinned_row.notes == "hand-pinned for test"
    # Measured fields are now populated (curator re-measured the pin).
    assert pinned_row.measured_waist_pitch_deg > 0.0


def test_bins_yaml_in_repo_loads_cleanly() -> None:
    """The hand-edited bins YAML in the repo parses into BinSpec objects."""
    path = REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_bins.yaml"
    specs = load_bin_specs(path)
    # Spot-check a few we know we authored.
    for required in (
        "idle_stand",
        "fwd_walk_standard",
        "fwd_step_1ft",
        "fwd_step_half_ft",
        "fwd_step_quarter_ft",
        "back_step_half_ft",
        "side_left_step",
        "side_right_step",
        "turn_left_15deg",
        "turn_left_30deg",
        "turn_left_45deg",
        "turn_left_90deg",
        "turn_right_90deg",
        "lean_fwd_small",
        "lean_fwd_medium",
        "lean_fwd_large",
        # Lateral lean family (NEW in v6).
        "lean_left_small",
        "lean_left_medium",
        "lean_left_large",
        "lean_right_small",
        "lean_right_medium",
        "lean_right_large",
        "torso_left_15deg",
        # torso_*_45deg renamed to torso_*_40deg in v6 (yaw cap = 40 deg).
        "torso_right_40deg",
    ):
        assert required in specs, f"bins.yaml missing required bin {required!r}"
        s = specs[required]
        assert s.window_frames_min < s.window_frames_max
        assert s.pelvis_z_band_m[0] < s.pelvis_z_band_m[1]
