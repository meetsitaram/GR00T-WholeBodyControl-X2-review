"""Regression tests for the v7.4-followup VR wrist quat offset.

The operator-side wrist orientation offset is the "controller mount
calibration" stop-gap: when one of the controllers is mounted at a
fixed rotation on the operator's wrist (e.g. left controller cuff
twisted ~30 deg outward), pre-rotating the reported op-wrist quat by
the inverse of the mount alignment makes
:class:`OperatorCalibration.apply_to_wrist_quat` see a corrected
operator quat. The full lift / fix is to re-run
``vr_operator_calibrate.py``; this file pins the stop-gap behaviour.

These tests verify:

1. Helper math: ``_rpy_deg_to_quat_wxyz`` matches scipy's intrinsic
   XYZ Tait-Bryan convention and short-circuits on all-zero input.
2. Default behaviour is bit-exact identical to today (no offset).
3. A non-zero offset rotates the IK target quat by the expected
   amount in the operator-wrist's local frame.
4. The offset is silently ignored when ``rotation_weight=0``
   (position-only IK never sees a target quat).
5. Per-side independence: a left-only offset must not change the
   right-side IK target.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.operator_calibration import (
    CALIBRATION_POSE_IDS,
    PoseMeasurement,
    _quat_multiply_wxyz,
    fit_calibration,
    robot_reference_wrist_positions,
    robot_reference_wrist_quats,
)
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import (
    VRArmTeleopCalibrated,
    _is_identity_quat,
    _rpy_deg_to_quat_wxyz,
)


# ── helpers (mirror tests/test_teleop_v2_dropout_and_orientation.py) ──────


def _make_non_identity_calibration():
    """Calibration whose ``wrist_alignment_quat`` is non-identity on
    both sides. We keep operator wrist *positions* equal to the robot
    reference (so position-side fit comes out as scale=1, translation=0)
    but rotate the operator's measured wrist quats by a fixed 20-deg
    yaw to force a non-identity alignment. The teleop's auto-disable
    detector only fires when both alignments look like identity, so
    this lets the rotation IK actually run.
    """
    ref = robot_reference_wrist_positions()
    qref = robot_reference_wrist_quats()
    op_twist_xyzw = sRot.from_euler("z", 20.0, degrees=True).as_quat()
    op_twist_wxyz = np.array(
        [op_twist_xyzw[3], *op_twist_xyzw[:3]], dtype=np.float64,
    )
    ms = {}
    for pose in CALIBRATION_POSE_IDS:
        l_q_op = _quat_multiply_wxyz(op_twist_wxyz, qref[pose]["left"])
        r_q_op = _quat_multiply_wxyz(op_twist_wxyz, qref[pose]["right"])
        ms[pose] = PoseMeasurement(
            pose_id=pose,
            left_wrist_mean=ref[pose]["left"].copy(),
            right_wrist_mean=ref[pose]["right"].copy(),
            sample_count=50,
            left_wrist_vel_rms_mps=0.01,
            right_wrist_vel_rms_mps=0.01,
            left_wrist_quat_head_yaw=l_q_op,
            right_wrist_quat_head_yaw=r_q_op,
        )
    return fit_calibration(ms, operator_id="synthetic")


def _vr_pose(
    *,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    head_pos: np.ndarray,
    head_yaw_rad: float = 0.0,
    left_wrist_quat: np.ndarray | None = None,
    right_wrist_quat: np.ndarray | None = None,
) -> np.ndarray:
    if left_wrist_quat is None:
        left_wrist_quat = sRot.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat(
            scalar_first=True
        )
    if right_wrist_quat is None:
        right_wrist_quat = sRot.from_euler("xyz", [-0.1, 0.2, 0.3]).as_quat(
            scalar_first=True
        )
    qhead = sRot.from_euler("z", head_yaw_rad).as_quat(scalar_first=True)
    out = np.zeros((3, 7), dtype=np.float64)
    out[0, :3] = left_wrist
    out[0, 3:] = left_wrist_quat
    out[1, :3] = right_wrist
    out[1, 3:] = right_wrist_quat
    out[2, :3] = head_pos
    out[2, 3:] = qhead
    return out


# ── helper math ──────────────────────────────────────────────────────────


def test_rpy_zero_short_circuits_to_identity() -> None:
    q = _rpy_deg_to_quat_wxyz((0.0, 0.0, 0.0))
    np.testing.assert_array_equal(q, np.array([1.0, 0.0, 0.0, 0.0]))
    assert _is_identity_quat(q)


def test_rpy_matches_scipy_intrinsic_xyz() -> None:
    """``_rpy_deg_to_quat_wxyz`` must equal scipy's
    ``Rotation.from_euler('XYZ', ..., degrees=True)`` (uppercase = intrinsic).
    """
    rpys = [
        (10.0, 0.0, 0.0),
        (0.0, 25.0, 0.0),
        (0.0, 0.0, -30.0),
        (12.5, -7.5, 18.0),
    ]
    for rpy in rpys:
        ours = _rpy_deg_to_quat_wxyz(rpy)
        ref_xyzw = sRot.from_euler("XYZ", rpy, degrees=True).as_quat()
        ref_wxyz = np.array([ref_xyzw[3], *ref_xyzw[:3]], dtype=np.float64)
        # quaternions are double-cover; compare up to sign.
        diff_pos = float(np.linalg.norm(ours - ref_wxyz))
        diff_neg = float(np.linalg.norm(ours + ref_wxyz))
        assert min(diff_pos, diff_neg) < 1e-12, (
            f"rpy={rpy}: ours={ours} vs scipy={ref_wxyz}"
        )


def test_rpy_30deg_yaw_quat_matches_axis_angle_about_z() -> None:
    """A pure 30-deg yaw should produce ``[cos(15), 0, 0, sin(15)]``."""
    q = _rpy_deg_to_quat_wxyz((0.0, 0.0, 30.0))
    half = np.deg2rad(15.0)
    np.testing.assert_allclose(
        q, [np.cos(half), 0.0, 0.0, np.sin(half)], atol=1e-12
    )


# ── default (zero offset) preserves today's behaviour ─────────────────────


def test_zero_offset_is_byte_equal_to_today() -> None:
    """With both offsets at the default ``(0, 0, 0)``, the IK target
    quat per side must match what the legacy ``VRArmTeleopCalibrated``
    produced. We verify by running the teleop with and without the
    explicit zero offsets and asserting bit-equal arm joint outputs.
    """
    cal = _make_non_identity_calibration()
    ref = robot_reference_wrist_positions()
    head = np.array([0.0, 0.0, 1.7])
    pose = _vr_pose(
        left_wrist=head + ref["t_pose"]["left"],
        right_wrist=head + ref["t_pose"]["right"],
        head_pos=head,
    )

    teleop_legacy = VRArmTeleopCalibrated(calibration=cal, rotation_weight=0.5)
    teleop_legacy.set_engaged(True)
    res_legacy = teleop_legacy.step(pose)

    teleop_explicit = VRArmTeleopCalibrated(
        calibration=cal,
        rotation_weight=0.5,
        left_wrist_op_quat_offset_rpy_deg=(0.0, 0.0, 0.0),
        right_wrist_op_quat_offset_rpy_deg=(0.0, 0.0, 0.0),
    )
    teleop_explicit.set_engaged(True)
    res_explicit = teleop_explicit.step(pose)

    np.testing.assert_array_equal(res_legacy.left_q, res_explicit.left_q)
    np.testing.assert_array_equal(res_legacy.right_q, res_explicit.right_q)
    np.testing.assert_array_equal(
        res_legacy.left_target_pos, res_explicit.left_target_pos
    )
    np.testing.assert_array_equal(
        res_legacy.right_target_pos, res_explicit.right_target_pos
    )


# ── non-zero offset rotates the IK target quat correctly ──────────────────


def _capture_ik_target_quat_via_solver(side: str, monkeypatch_target):
    """The ``CalibratedTeleopTickResult`` doesn't expose the rotation
    target, but the solver receives it as ``target_quat_wxyz=...``.
    We intercept the solver to record it.
    """
    captured: dict[str, np.ndarray | None] = {"target": None}

    def _spy_solve(self, *, q_seed, target_pos, target_quat_wxyz, max_iters):
        if target_quat_wxyz is not None:
            captured["target"] = np.asarray(target_quat_wxyz, dtype=np.float64).copy()
        return self.__class__.__bases__[0].solve(
            self,
            q_seed=q_seed,
            target_pos=target_pos,
            target_quat_wxyz=target_quat_wxyz,
            max_iters=max_iters,
        )

    return captured, _spy_solve


def _run_one_tick_capturing_target_quats(
    *,
    cal,
    left_offset_rpy_deg,
    right_offset_rpy_deg,
    pose,
):
    """Run one engaged tick and capture the IK target quats via a
    monkey-patched ``ArmIKSolver.solve`` proxy on the teleop's
    private solver instances. Returns ``(l_target_quat, r_target_quat)``
    in wxyz.
    """
    teleop = VRArmTeleopCalibrated(
        calibration=cal,
        rotation_weight=0.5,
        left_wrist_op_quat_offset_rpy_deg=left_offset_rpy_deg,
        right_wrist_op_quat_offset_rpy_deg=right_offset_rpy_deg,
    )
    teleop.set_engaged(True)

    captured = {"left": None, "right": None}
    real_left_solve = teleop._left_solver.solve
    real_right_solve = teleop._right_solver.solve

    def _patched_left(*, q_seed, target_pos, target_quat_wxyz, max_iters):
        captured["left"] = (
            None if target_quat_wxyz is None
            else np.asarray(target_quat_wxyz, dtype=np.float64).copy()
        )
        return real_left_solve(
            q_seed=q_seed,
            target_pos=target_pos,
            target_quat_wxyz=target_quat_wxyz,
            max_iters=max_iters,
        )

    def _patched_right(*, q_seed, target_pos, target_quat_wxyz, max_iters):
        captured["right"] = (
            None if target_quat_wxyz is None
            else np.asarray(target_quat_wxyz, dtype=np.float64).copy()
        )
        return real_right_solve(
            q_seed=q_seed,
            target_pos=target_pos,
            target_quat_wxyz=target_quat_wxyz,
            max_iters=max_iters,
        )

    teleop._left_solver.solve = _patched_left  # type: ignore[method-assign]
    teleop._right_solver.solve = _patched_right  # type: ignore[method-assign]
    teleop.step(pose)
    return captured["left"], captured["right"]


def test_left_offset_rotates_only_left_target_quat() -> None:
    """A 30-deg yaw offset on the left side must produce a left IK
    target rotated by the offset (in op-wrist local frame, post-mul
    on the head-yaw-frame op quat then alignment), with the right side
    untouched.
    """
    cal = _make_non_identity_calibration()
    ref = robot_reference_wrist_positions()
    head = np.array([0.0, 0.0, 1.7])
    pose = _vr_pose(
        left_wrist=head + ref["t_pose"]["left"],
        right_wrist=head + ref["t_pose"]["right"],
        head_pos=head,
    )

    l_quat_no_offset, r_quat_no_offset = _run_one_tick_capturing_target_quats(
        cal=cal,
        left_offset_rpy_deg=(0.0, 0.0, 0.0),
        right_offset_rpy_deg=(0.0, 0.0, 0.0),
        pose=pose,
    )
    assert l_quat_no_offset is not None and r_quat_no_offset is not None

    l_quat_with_offset, r_quat_with_offset = _run_one_tick_capturing_target_quats(
        cal=cal,
        left_offset_rpy_deg=(0.0, 0.0, -30.0),  # left wrist yaw -30 deg
        right_offset_rpy_deg=(0.0, 0.0, 0.0),
        pose=pose,
    )
    assert l_quat_with_offset is not None and r_quat_with_offset is not None

    # Right side must be byte-equal (no offset on right).
    np.testing.assert_array_equal(r_quat_with_offset, r_quat_no_offset)

    # Left side must differ. We sanity-check that the two quats are
    # NOT the same (up to sign) -- a 30-deg rotation produces a clear
    # angular change in the IK target quat.
    diff_pos = float(np.linalg.norm(l_quat_with_offset - l_quat_no_offset))
    diff_neg = float(np.linalg.norm(l_quat_with_offset + l_quat_no_offset))
    assert min(diff_pos, diff_neg) > 1e-3, (
        f"left target quat unchanged despite -30deg yaw offset: "
        f"with_offset={l_quat_with_offset}, no_offset={l_quat_no_offset}"
    )


def test_left_offset_equals_calibration_post_multiply_of_head_yaw_frame() -> None:
    """The full contract: with a left offset of ``q_offset``,
    the left IK target quat must satisfy

        target_with_offset == q_align_left * (q_op_in_head_yaw * q_offset)
                           == (q_align_left * q_op_in_head_yaw) * q_offset
                           == target_no_offset * q_offset

    i.e. post-multiplying the op-quat by ``q_offset`` is equivalent to
    post-multiplying the IK target quat by the same ``q_offset``. This
    is the operator's mental model: "rotate in the wrist's local frame
    after the calibration places it" gives the same answer as "rotate
    the operator's wrist before the calibration sees it".
    """
    cal = _make_non_identity_calibration()
    ref = robot_reference_wrist_positions()
    head = np.array([0.0, 0.0, 1.7])
    pose = _vr_pose(
        left_wrist=head + ref["t_pose"]["left"],
        right_wrist=head + ref["t_pose"]["right"],
        head_pos=head,
    )

    rpy = (5.0, -8.0, -30.0)
    q_offset = _rpy_deg_to_quat_wxyz(rpy)

    l_quat_no_offset, _ = _run_one_tick_capturing_target_quats(
        cal=cal,
        left_offset_rpy_deg=(0.0, 0.0, 0.0),
        right_offset_rpy_deg=(0.0, 0.0, 0.0),
        pose=pose,
    )
    l_quat_with_offset, _ = _run_one_tick_capturing_target_quats(
        cal=cal,
        left_offset_rpy_deg=rpy,
        right_offset_rpy_deg=(0.0, 0.0, 0.0),
        pose=pose,
    )

    expected = _quat_multiply_wxyz(l_quat_no_offset, q_offset)
    diff_pos = float(np.linalg.norm(l_quat_with_offset - expected))
    diff_neg = float(np.linalg.norm(l_quat_with_offset + expected))
    assert min(diff_pos, diff_neg) < 1e-12, (
        f"with_offset={l_quat_with_offset} != expected={expected}"
    )


def test_right_offset_does_not_leak_into_left_side() -> None:
    """Symmetric to the left-only test: a right-only offset must not
    change the left IK target quat. Pins per-side independence.
    """
    cal = _make_non_identity_calibration()
    ref = robot_reference_wrist_positions()
    head = np.array([0.0, 0.0, 1.7])
    pose = _vr_pose(
        left_wrist=head + ref["t_pose"]["left"],
        right_wrist=head + ref["t_pose"]["right"],
        head_pos=head,
    )

    l_quat_no_offset, _ = _run_one_tick_capturing_target_quats(
        cal=cal,
        left_offset_rpy_deg=(0.0, 0.0, 0.0),
        right_offset_rpy_deg=(0.0, 0.0, 0.0),
        pose=pose,
    )
    l_quat_with_right_offset, _ = _run_one_tick_capturing_target_quats(
        cal=cal,
        left_offset_rpy_deg=(0.0, 0.0, 0.0),
        right_offset_rpy_deg=(15.0, -10.0, 22.0),
        pose=pose,
    )

    np.testing.assert_array_equal(l_quat_with_right_offset, l_quat_no_offset)


# ── rotation_weight=0 silently disables the offset ────────────────────────


def test_offset_is_noop_when_rotation_weight_zero(capsys) -> None:
    """When the IK runs position-only, the offset has no effect on the
    arm joints (the IK never sees a rotation target). The teleop
    still emits a one-line warning so the operator notices.
    """
    cal = _make_non_identity_calibration()
    ref = robot_reference_wrist_positions()
    head = np.array([0.0, 0.0, 1.7])
    pose = _vr_pose(
        left_wrist=head + ref["t_pose"]["left"],
        right_wrist=head + ref["t_pose"]["right"],
        head_pos=head,
    )

    teleop_no_offset = VRArmTeleopCalibrated(calibration=cal, rotation_weight=0.0)
    teleop_no_offset.set_engaged(True)
    res_baseline = teleop_no_offset.step(pose)

    teleop_with_offset = VRArmTeleopCalibrated(
        calibration=cal,
        rotation_weight=0.0,
        left_wrist_op_quat_offset_rpy_deg=(0.0, 0.0, -30.0),
    )
    teleop_with_offset.set_engaged(True)
    res_with_offset = teleop_with_offset.step(pose)

    # Warning should have been printed at __init__.
    out = capsys.readouterr().out
    assert "wrist op-quat offsets requested" in out, out
    assert "rotation_weight is 0" in out, out

    # Arm joints must be byte-identical (position-only IK ignores quat target).
    np.testing.assert_array_equal(res_baseline.left_q, res_with_offset.left_q)
    np.testing.assert_array_equal(res_baseline.right_q, res_with_offset.right_q)
