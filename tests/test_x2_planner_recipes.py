"""Unit tests for the recipe-driven primitive builder.

Pure-numpy + tiny synthetic source-clip fixtures. Exercises:

  - Each of the 7 ops in isolation (plus error paths).
  - Bilateral identity invariant: ``mirror_lr(idle_stand)`` == idle_stand.
  - Quaternion mirror identity: pure-yaw clip mirrors to negated yaw.
  - Recipe loader: schema validation, derive_from chains, cycle detection.
  - End-to-end pipeline: synthesize -> freeze -> mirror -> scale.

Runs without joblib / MuJoCo / Isaac::

    .venv/bin/python -m pytest tests/test_x2_planner_recipes.py -v
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.planner import constants as const  # noqa: E402
from gear_sonic.utils.planner.x2_recipes import (  # noqa: E402
    Buffer,
    Recipe,
    SourceClip,
    load_recipes,
    op_clip_window,
    op_freeze,
    op_mirror_lr,
    op_pad_idle,
    op_recenter_root,
    op_scale_magnitude,
    op_synthesize_crouch_ramp,
    op_synthesize_waist_ramp,
    run_recipe,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _idle_buffer(n: int = 30, fps: float = 30.0) -> Buffer:
    """A buffer of n frames at default stand pose, identity rot, origin trans."""
    dof = np.broadcast_to(
        const.DEFAULT_STAND_POSE_NP.astype(np.float64), (n, const.NUM_BODY_DOFS)
    ).copy()
    rot = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (n, 4)).copy()
    trans = np.zeros((n, 3), dtype=np.float64)
    trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return Buffer(dof=dof, root_rot_xyzw=rot, root_trans=trans, fps=fps)


def _make_source_clip(motion_key: str, n: int = 60) -> SourceClip:
    """Synthetic clip that ramps several joints over n frames so we can detect
    L<->R swaps and scaling later."""
    dof = np.tile(
        const.DEFAULT_STAND_POSE_NP.astype(np.float64), (n, 1)
    )
    # Walk one joint per side to make swaps visible.
    t = np.linspace(0.0, 1.0, n)
    dof[:, 0] += 0.3 * t                  # left_hip_pitch
    dof[:, 6] += 0.6 * t                  # right_hip_pitch (different scale)
    dof[:, 15] += 0.5 * t                 # left_shoulder_pitch
    dof[:, 22] += 0.7 * t                 # right_shoulder_pitch (different scale)
    dof[:, const.WAIST_YAW_IDX] = 0.4 * t  # waist yaw rises
    rot = np.broadcast_to(np.array([0.0, 0.0, 0.0, 1.0]), (n, 4)).copy()
    trans = np.zeros((n, 3), dtype=np.float64)
    trans[:, 0] = 0.5 * t  # walks forward
    trans[:, 1] = 0.1 * t  # slight lateral drift
    trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return SourceClip(
        motion_key=motion_key,
        dof=dof.astype(np.float32),
        root_rot_xyzw=rot.astype(np.float32),
        root_trans=trans.astype(np.float32),
        fps=30.0,
    )


# ---------------------------------------------------------------------------
# clip_window
# ---------------------------------------------------------------------------


def test_clip_window_basic() -> None:
    clips = {"foo": _make_source_clip("foo", n=60)}
    out = op_clip_window({"motion_key": "foo", "start_frame": 10, "n_frames": 20},
                        None, clips)
    assert out.dof.shape == (20, const.NUM_BODY_DOFS)
    assert out.root_rot_xyzw.shape == (20, 4)
    assert out.root_trans.shape == (20, 3)
    assert out.fps == pytest.approx(30.0)
    np.testing.assert_allclose(
        out.dof[0], clips["foo"].dof[10].astype(np.float64)
    )


def test_clip_window_unknown_motion_key_raises() -> None:
    with pytest.raises(KeyError, match="motion_key"):
        op_clip_window({"motion_key": "missing", "start_frame": 0, "n_frames": 5},
                       None, {})


def test_clip_window_out_of_bounds_raises() -> None:
    clips = {"foo": _make_source_clip("foo", n=10)}
    with pytest.raises(ValueError, match="out of bounds"):
        op_clip_window({"motion_key": "foo", "start_frame": 5, "n_frames": 20},
                       None, clips)


# ---------------------------------------------------------------------------
# synthesize_waist_ramp
# ---------------------------------------------------------------------------


def test_synthesize_waist_ramp_pitch_peak_and_length() -> None:
    out = op_synthesize_waist_ramp(
        {"axis": "pitch", "peak_deg": 20.0,
         "ramp_in_frames": 10, "hold_frames": 5, "ramp_out_frames": 10},
        None, {},
    )
    assert out.n_frames() == 25
    assert out.fps == pytest.approx(50.0)
    # Frames 10..14 inclusive should be at the peak.
    peak_rad = np.deg2rad(20.0)
    expected = const.DEFAULT_STAND_POSE_NP[const.WAIST_PITCH_IDX] + peak_rad
    np.testing.assert_allclose(
        out.dof[10:15, const.WAIST_PITCH_IDX], expected, atol=1e-6
    )
    # First and last frames return to default.
    np.testing.assert_allclose(
        out.dof[0, const.WAIST_PITCH_IDX],
        const.DEFAULT_STAND_POSE_NP[const.WAIST_PITCH_IDX],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        out.dof[-1, const.WAIST_PITCH_IDX],
        const.DEFAULT_STAND_POSE_NP[const.WAIST_PITCH_IDX],
        atol=1e-6,
    )
    # Arms + head remain at default for the whole ramp.
    arms_idx = list(const.LEFT_ARM_INDICES) + list(const.RIGHT_ARM_INDICES)
    for idx in arms_idx + list(const.HEAD_INDICES):
        np.testing.assert_allclose(
            out.dof[:, idx],
            const.DEFAULT_STAND_POSE_NP[idx],
            atol=1e-6,
        )
    # Root rot identity, root trans at default pelvis Z.
    np.testing.assert_allclose(out.root_rot_xyzw[:, :3], 0.0)
    np.testing.assert_allclose(out.root_rot_xyzw[:, 3], 1.0)
    np.testing.assert_allclose(out.root_trans[:, 2], const.DEFAULT_PELVIS_Z_M)


def test_synthesize_waist_ramp_negative_yaw() -> None:
    out = op_synthesize_waist_ramp(
        {"axis": "yaw", "peak_deg": -15.0,
         "ramp_in_frames": 5, "hold_frames": 5, "ramp_out_frames": 5},
        None, {},
    )
    peak_rad = np.deg2rad(-15.0)
    expected = const.DEFAULT_STAND_POSE_NP[const.WAIST_YAW_IDX] + peak_rad
    np.testing.assert_allclose(
        out.dof[5:10, const.WAIST_YAW_IDX], expected, atol=1e-6
    )


def test_synthesize_waist_ramp_unknown_axis_raises() -> None:
    with pytest.raises(ValueError, match="axis"):
        op_synthesize_waist_ramp(
            {"axis": "twist", "peak_deg": 10.0,
             "ramp_in_frames": 5, "hold_frames": 5, "ramp_out_frames": 5},
            None, {},
        )


# ---------------------------------------------------------------------------
# synthesize_crouch_ramp
# ---------------------------------------------------------------------------


def test_synthesize_crouch_ramp_geometry_and_envelope() -> None:
    out = op_synthesize_crouch_ramp(
        {"peak_drop_m": 0.06,
         "ramp_in_frames": 10, "hold_frames": 5, "ramp_out_frames": 10},
        None, {},
    )
    assert out.n_frames() == 25
    assert out.fps == pytest.approx(50.0)

    # Apex Z drop reached during the hold (frames 10..14 inclusive).
    apex_z = const.DEFAULT_PELVIS_Z_M - 0.06
    np.testing.assert_allclose(out.root_trans[10:15, 2], apex_z, atol=1e-9)
    # Returns to baseline at start and end.
    np.testing.assert_allclose(
        out.root_trans[0, 2], const.DEFAULT_PELVIS_Z_M, atol=1e-9
    )
    np.testing.assert_allclose(
        out.root_trans[-1, 2], const.DEFAULT_PELVIS_Z_M, atol=1e-9
    )

    # Geometry invariant: knee = 2*hip, ankle = -hip (peak frame).
    template = const.DEFAULT_STAND_POSE_NP.astype(np.float64)
    hip_l = out.dof[12, const.LEFT_HIP_PITCH_IDX] - template[const.LEFT_HIP_PITCH_IDX]
    knee_l = out.dof[12, const.LEFT_KNEE_IDX] - template[const.LEFT_KNEE_IDX]
    ankle_l = out.dof[12, const.LEFT_ANKLE_PITCH_IDX] - template[const.LEFT_ANKLE_PITCH_IDX]
    np.testing.assert_allclose(knee_l, -2.0 * hip_l, atol=1e-9)
    np.testing.assert_allclose(ankle_l, hip_l, atol=1e-9)
    # Hip delta is negative (more flex) for a forward-leg-fold squat.
    assert hip_l < 0.0
    # Bilateral symmetry.
    np.testing.assert_allclose(
        out.dof[:, const.LEFT_HIP_PITCH_IDX],
        out.dof[:, const.RIGHT_HIP_PITCH_IDX],
        atol=1e-12,
    )

    # Identity root quat throughout, root XY pinned at origin.
    np.testing.assert_allclose(out.root_rot_xyzw[:, :3], 0.0, atol=1e-12)
    np.testing.assert_allclose(out.root_rot_xyzw[:, 3], 1.0, atol=1e-12)
    np.testing.assert_allclose(out.root_trans[:, :2], 0.0, atol=1e-12)


def test_synthesize_crouch_ramp_rejects_invalid_drop() -> None:
    with pytest.raises(ValueError, match="peak_drop_m"):
        op_synthesize_crouch_ramp(
            {"peak_drop_m": 0.0,
             "ramp_in_frames": 5, "hold_frames": 5, "ramp_out_frames": 5},
            None, {},
        )
    with pytest.raises(ValueError, match="peak_drop_m"):
        op_synthesize_crouch_ramp(
            {"peak_drop_m": 0.25,
             "ramp_in_frames": 5, "hold_frames": 5, "ramp_out_frames": 5},
            None, {},
        )


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def test_freeze_arms_and_head_resets_only_those_dims() -> None:
    buf = _idle_buffer(n=12)
    # Pollute every joint so we can detect which ones survive freeze.
    buf.dof[:] += 0.5
    out = op_freeze({"groups": ["arms", "head"]}, buf, {})
    arms_idx = list(const.LEFT_ARM_INDICES) + list(const.RIGHT_ARM_INDICES)
    for idx in arms_idx + list(const.HEAD_INDICES):
        np.testing.assert_allclose(
            out.dof[:, idx],
            const.DEFAULT_STAND_POSE_NP[idx],
            atol=1e-6,
        )
    # Legs and waist must keep the +0.5 pollution.
    for idx in list(const.LEG_INDICES) + list(const.WAIST_INDICES):
        np.testing.assert_allclose(
            out.dof[:, idx],
            const.DEFAULT_STAND_POSE_NP[idx] + 0.5,
            atol=1e-6,
        )


def test_freeze_unknown_group_raises() -> None:
    buf = _idle_buffer(n=4)
    with pytest.raises(ValueError, match="unknown group"):
        op_freeze({"groups": ["fingers"]}, buf, {})


def test_freeze_requires_groups() -> None:
    buf = _idle_buffer(n=4)
    with pytest.raises(ValueError, match="non-empty"):
        op_freeze({"groups": []}, buf, {})


def test_freeze_without_producer_raises() -> None:
    with pytest.raises(ValueError, match="producer"):
        op_freeze({"groups": ["arms"]}, None, {})


# ---------------------------------------------------------------------------
# mirror_lr
# ---------------------------------------------------------------------------


def test_mirror_lr_idle_is_identity() -> None:
    """Critical invariant: bilaterally-symmetric stand pose mirrors to itself."""
    buf = _idle_buffer(n=8)
    out = op_mirror_lr({}, buf, {})
    np.testing.assert_allclose(
        out.dof, buf.dof, atol=1e-6,
        err_msg="mirror_lr must leave the bilaterally-symmetric idle pose unchanged",
    )
    np.testing.assert_allclose(
        out.root_rot_xyzw, buf.root_rot_xyzw, atol=1e-6
    )
    np.testing.assert_allclose(out.root_trans, buf.root_trans, atol=1e-6)


def test_mirror_lr_swaps_left_right_pitch_joints() -> None:
    """L<->R hip_pitch / shoulder_pitch should swap (no negation)."""
    buf = _idle_buffer(n=4)
    buf.dof[:, 0] = 0.1   # left_hip_pitch
    buf.dof[:, 6] = 0.4   # right_hip_pitch
    buf.dof[:, 15] = 0.2  # left_shoulder_pitch
    buf.dof[:, 22] = 0.8  # right_shoulder_pitch
    out = op_mirror_lr({}, buf, {})
    np.testing.assert_allclose(out.dof[:, 0], 0.4, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, 6], 0.1, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, 15], 0.8, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, 22], 0.2, atol=1e-6)


def test_mirror_lr_negates_anti_symmetric_joints() -> None:
    """hip_roll / shoulder_roll values should swap AND flip sign."""
    buf = _idle_buffer(n=4)
    # Wipe stand-pose roll values so we measure pure swap-and-negate.
    buf.dof[:, 1] = 0.3   # left_hip_roll
    buf.dof[:, 7] = -0.1  # right_hip_roll
    buf.dof[:, 16] = 0.2  # left_shoulder_roll
    buf.dof[:, 23] = -0.5 # right_shoulder_roll
    out = op_mirror_lr({}, buf, {})
    # left_hip_roll <- -right_hip_roll = -(-0.1) = 0.1
    np.testing.assert_allclose(out.dof[:, 1], 0.1, atol=1e-6)
    # right_hip_roll <- -left_hip_roll = -0.3
    np.testing.assert_allclose(out.dof[:, 7], -0.3, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, 16], 0.5, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, 23], -0.2, atol=1e-6)


def test_mirror_lr_negates_waist_yaw_roll_and_head_yaw() -> None:
    buf = _idle_buffer(n=3)
    buf.dof[:, const.WAIST_YAW_IDX] = 0.4
    buf.dof[:, const.WAIST_PITCH_IDX] = 0.2  # pitch unchanged
    buf.dof[:, const.WAIST_ROLL_IDX] = 0.1
    buf.dof[:, 29] = 0.3  # head_yaw
    buf.dof[:, 30] = 0.2  # head_pitch unchanged
    out = op_mirror_lr({}, buf, {})
    np.testing.assert_allclose(out.dof[:, const.WAIST_YAW_IDX], -0.4, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, const.WAIST_PITCH_IDX], 0.2, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, const.WAIST_ROLL_IDX], -0.1, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, 29], -0.3, atol=1e-6)
    np.testing.assert_allclose(out.dof[:, 30], 0.2, atol=1e-6)


def test_mirror_lr_double_is_identity() -> None:
    clips = {"foo": _make_source_clip("foo", n=40)}
    base = op_clip_window({"motion_key": "foo", "start_frame": 0, "n_frames": 40},
                          None, clips)
    once = op_mirror_lr({}, base, {})
    twice = op_mirror_lr({}, once, {})
    np.testing.assert_allclose(twice.dof, base.dof, atol=1e-6)
    np.testing.assert_allclose(
        twice.root_rot_xyzw, base.root_rot_xyzw, atol=1e-6
    )
    np.testing.assert_allclose(twice.root_trans, base.root_trans, atol=1e-6)


def test_mirror_lr_root_quat_yaw_negation() -> None:
    """A pure +yaw rotation should mirror to a -yaw rotation."""
    n = 5
    buf = _idle_buffer(n=n)
    yaw = np.deg2rad(45.0)
    buf.root_rot_xyzw[:] = Rot.from_euler("z", yaw).as_quat()
    out = op_mirror_lr({}, buf, {})
    out_yaws = Rot.from_quat(out.root_rot_xyzw).as_euler("zyx")[:, 0]
    np.testing.assert_allclose(out_yaws, -yaw, atol=1e-6)


def test_mirror_lr_negates_root_trans_y() -> None:
    buf = _idle_buffer(n=4)
    buf.root_trans[:, 0] = 0.3
    buf.root_trans[:, 1] = 0.5
    buf.root_trans[:, 2] = 0.78
    out = op_mirror_lr({}, buf, {})
    np.testing.assert_allclose(out.root_trans[:, 0], 0.3)
    np.testing.assert_allclose(out.root_trans[:, 1], -0.5)
    np.testing.assert_allclose(out.root_trans[:, 2], 0.78)


def test_mirror_lr_negate_flags_can_be_disabled() -> None:
    buf = _idle_buffer(n=2)
    buf.root_trans[:, 1] = 0.2
    out = op_mirror_lr({"also_negate_root_y": False}, buf, {})
    np.testing.assert_allclose(out.root_trans[:, 1], 0.2)


# ---------------------------------------------------------------------------
# scale_magnitude
# ---------------------------------------------------------------------------


def test_scale_magnitude_factor_one_is_identity() -> None:
    clips = {"foo": _make_source_clip("foo", n=20)}
    base = op_clip_window({"motion_key": "foo", "start_frame": 0, "n_frames": 20},
                          None, clips)
    out = op_scale_magnitude({"factor": 1.0}, base, {})
    np.testing.assert_allclose(out.dof, base.dof, atol=1e-6)
    np.testing.assert_allclose(out.root_trans, base.root_trans, atol=1e-6)
    np.testing.assert_allclose(out.root_rot_xyzw, base.root_rot_xyzw, atol=1e-6)


def test_scale_magnitude_factor_zero_collapses_to_stand_pose() -> None:
    clips = {"foo": _make_source_clip("foo", n=10)}
    base = op_clip_window({"motion_key": "foo", "start_frame": 0, "n_frames": 10},
                          None, clips)
    out = op_scale_magnitude({"factor": 0.0}, base, {})
    expected = np.tile(
        const.DEFAULT_STAND_POSE_NP.astype(np.float64), (10, 1)
    )
    np.testing.assert_allclose(out.dof, expected, atol=1e-6)
    # XY collapses to start.
    np.testing.assert_allclose(out.root_trans[:, 0], base.root_trans[0, 0])
    np.testing.assert_allclose(out.root_trans[:, 1], base.root_trans[0, 1])


def test_scale_magnitude_half_halves_xy_and_dof_delta() -> None:
    clips = {"foo": _make_source_clip("foo", n=20)}
    base = op_clip_window({"motion_key": "foo", "start_frame": 0, "n_frames": 20},
                          None, clips)
    out = op_scale_magnitude({"factor": 0.5}, base, {})
    expected_xy = base.root_trans[0, :2] + 0.5 * (
        base.root_trans[:, :2] - base.root_trans[0, :2]
    )
    np.testing.assert_allclose(out.root_trans[:, :2], expected_xy, atol=1e-6)
    template = const.DEFAULT_STAND_POSE_NP.astype(np.float64)
    expected_dof = template[None, :] + 0.5 * (base.dof - template[None, :])
    np.testing.assert_allclose(out.dof, expected_dof, atol=1e-6)


def test_scale_magnitude_yaw_scaled_when_root_yaws() -> None:
    n = 10
    buf = _idle_buffer(n=n)
    yaws = np.linspace(0.0, np.deg2rad(40.0), n)
    buf.root_rot_xyzw = Rot.from_euler("z", yaws).as_quat()
    out = op_scale_magnitude({"factor": 0.25}, buf, {})
    out_yaws = Rot.from_quat(out.root_rot_xyzw).as_euler("zyx")[:, 0]
    expected = 0.25 * yaws
    np.testing.assert_allclose(out_yaws, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# recenter_root
# ---------------------------------------------------------------------------


def test_recenter_root_yaw_zeroes_net_drift() -> None:
    n = 10
    buf = _idle_buffer(n=n)
    yaws = np.linspace(0.0, np.deg2rad(15.0), n)
    buf.root_rot_xyzw = Rot.from_euler("z", yaws).as_quat()
    out = op_recenter_root({"yaw": True}, buf, {})
    out_yaws = Rot.from_quat(out.root_rot_xyzw).as_euler("zyx")[:, 0]
    assert abs(out_yaws[-1] - out_yaws[0]) < 1e-6


def test_recenter_root_xy_zeroes_net_drift() -> None:
    n = 8
    buf = _idle_buffer(n=n)
    buf.root_trans[:, 0] = np.linspace(0.0, 0.4, n)
    buf.root_trans[:, 1] = np.linspace(0.0, 0.2, n)
    out = op_recenter_root({"xy": True}, buf, {})
    np.testing.assert_allclose(out.root_trans[-1, :2], buf.root_trans[0, :2],
                               atol=1e-6)


def test_recenter_root_requires_at_least_one_axis() -> None:
    buf = _idle_buffer(n=3)
    with pytest.raises(ValueError, match="at least one"):
        op_recenter_root({}, buf, {})


# ---------------------------------------------------------------------------
# pad_idle
# ---------------------------------------------------------------------------


def test_pad_idle_lead_and_trail_lengths() -> None:
    clips = {"foo": _make_source_clip("foo", n=20)}
    base = op_clip_window({"motion_key": "foo", "start_frame": 0, "n_frames": 20},
                          None, clips)
    out = op_pad_idle({"leading_frames": 5, "trailing_frames": 7}, base, {})
    assert out.n_frames() == 5 + 20 + 7
    np.testing.assert_allclose(
        out.dof[0], const.DEFAULT_STAND_POSE_NP.astype(np.float64), atol=1e-6
    )
    np.testing.assert_allclose(
        out.dof[-1], const.DEFAULT_STAND_POSE_NP.astype(np.float64), atol=1e-6
    )
    # Middle preserves the base.
    np.testing.assert_allclose(out.dof[5:25], base.dof, atol=1e-6)


def test_pad_idle_zero_is_noop() -> None:
    buf = _idle_buffer(n=10)
    out = op_pad_idle({}, buf, {})
    assert out.n_frames() == 10
    np.testing.assert_allclose(out.dof, buf.dof)


def test_pad_idle_anchors_to_start_and_end_xy() -> None:
    """Padding must stay at the existing start/end XY (no teleport)."""
    n = 6
    buf = _idle_buffer(n=n)
    buf.root_trans[:, 0] = np.linspace(1.0, 2.0, n)
    out = op_pad_idle({"leading_frames": 3, "trailing_frames": 3}, buf, {})
    np.testing.assert_allclose(out.root_trans[:3, 0], 1.0)
    np.testing.assert_allclose(out.root_trans[-3:, 0], 2.0)


# ---------------------------------------------------------------------------
# Recipe loader / runner
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "recipes.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_load_recipes_basic(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        primitives:
          - bin_name: lean
            family: static_upper_body
            ops:
              - synthesize_waist_ramp:
                  axis: pitch
                  peak_deg: 10
                  ramp_in_frames: 4
                  hold_frames: 4
                  ramp_out_frames: 4
              - freeze: {groups: [arms, head]}
    """)
    recipes = load_recipes(p)
    assert set(recipes) == {"lean"}
    assert recipes["lean"].family == "static_upper_body"
    assert len(recipes["lean"].ops) == 2


def test_load_recipes_first_op_must_be_producer(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        primitives:
          - bin_name: bad
            family: locomotion
            ops:
              - freeze: {groups: [arms]}
    """)
    with pytest.raises(ValueError, match="producer"):
        load_recipes(p)


def test_load_recipes_unknown_op_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        primitives:
          - bin_name: bad
            family: locomotion
            ops:
              - rotate_180: {}
    """)
    with pytest.raises(ValueError, match="unknown op"):
        load_recipes(p)


def test_load_recipes_op_must_be_single_key_dict(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        primitives:
          - bin_name: bad
            family: locomotion
            ops:
              - {clip_window: {motion_key: x, start_frame: 0, n_frames: 5}, freeze: {}}
    """)
    with pytest.raises(ValueError, match="single-key dict"):
        load_recipes(p)


def test_load_recipes_derive_from_unknown_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        primitives:
          - bin_name: child
            family: locomotion
            derive_from: ghost
            ops:
              - mirror_lr: {}
    """)
    recipes = load_recipes(p)
    with pytest.raises(ValueError, match="derive_from"):
        run_recipe(recipes["child"], recipes, {})


def test_load_recipes_derive_from_cycle_raises(tmp_path: Path) -> None:
    p = _write(tmp_path, """
        primitives:
          - bin_name: a
            family: locomotion
            derive_from: b
            ops: [mirror_lr: {}]
          - bin_name: b
            family: locomotion
            derive_from: a
            ops: [mirror_lr: {}]
    """)
    recipes = load_recipes(p)
    with pytest.raises(ValueError, match="cycle"):
        run_recipe(recipes["a"], recipes, {})


def test_run_recipe_chain_synthesize_freeze_mirror_scale(tmp_path: Path) -> None:
    """End-to-end: synthesized lean -> freeze -> mirror -> half scale."""
    p = _write(tmp_path, """
        primitives:
          - bin_name: torso_left
            family: static_upper_body
            ops:
              - synthesize_waist_ramp:
                  axis: yaw
                  peak_deg: 30
                  ramp_in_frames: 5
                  hold_frames: 5
                  ramp_out_frames: 5
              - freeze: {groups: [arms, head]}
          - bin_name: torso_right
            family: static_upper_body
            derive_from: torso_left
            ops: [mirror_lr: {}]
          - bin_name: torso_right_small
            family: static_upper_body
            derive_from: torso_right
            ops:
              - scale_magnitude: {factor: 0.5}
    """)
    recipes = load_recipes(p)
    out = run_recipe(recipes["torso_right_small"], recipes, {})
    assert out.n_frames() == 15
    # Peak at frame 7 (middle of hold) should be -15deg of waist_yaw,
    # because mirror negates 30 -> -30, and scale halves it to -15.
    expected = const.DEFAULT_STAND_POSE_NP[const.WAIST_YAW_IDX] + np.deg2rad(-15.0)
    np.testing.assert_allclose(
        out.dof[7, const.WAIST_YAW_IDX], expected, atol=1e-6
    )
    # Arms + head still at default after the chain.
    arms_idx = list(const.LEFT_ARM_INDICES) + list(const.RIGHT_ARM_INDICES)
    for idx in arms_idx + list(const.HEAD_INDICES):
        np.testing.assert_allclose(
            out.dof[:, idx],
            const.DEFAULT_STAND_POSE_NP[idx],
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Repo-shipped recipes file: must parse, reference real source clips, and
# build without error if the source library is present. The actual physics
# / SONIC validation lives in the sim deploy smoke test.
# ---------------------------------------------------------------------------


def test_repo_recipes_yaml_loads() -> None:
    yaml_path = (
        REPO_ROOT
        / "gear_sonic"
        / "data"
        / "motions"
        / "x2_planner_primitives_recipes.yaml"
    )
    if not yaml_path.exists():
        pytest.skip(f"{yaml_path} not present in this checkout")
    recipes = load_recipes(yaml_path)
    # All 28 planner bins should be present.
    assert "idle_stand" in recipes
    assert len([r for r in recipes.values() if r.family == "static_upper_body"]) >= 9
    assert len([r for r in recipes.values() if r.family == "locomotion"]) >= 14
    # All derive_from references resolve.
    for r in recipes.values():
        if r.derive_from is not None:
            assert r.derive_from in recipes, (
                f"recipe {r.bin_name!r} derives from missing {r.derive_from!r}"
            )
