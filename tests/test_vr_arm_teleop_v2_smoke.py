"""Smoke tests for ``gear_sonic.utils.teleop.vr_arm_teleop_v2``.

Cover the stateless head-relative mapping invariants: idle returns the
neutral pose, engaging while the operator is in their calibrated
arms-down pose puts the robot wrists near the calibrated reference
position, and rotating the operator's head + body 90 deg in place does
NOT move the wrist target (the bug the engage-anchor solver had).
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.operator_calibration import (
    CALIBRATION_POSE_IDS,
    PoseMeasurement,
    fit_calibration,
    robot_reference_wrist_positions,
)
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import (
    CalibratedTeleopTickResult,
    VRArmTeleopCalibrated,
)


def _make_identity_calibration():
    """Calibration where operator wrist == robot wrist (no anatomy mismatch).

    Lets us check IK convergence directly: if the operator's wrist in
    their head-yaw frame matches the FK wrist of the robot's reference
    pose, the IK should converge close to that reference q.
    """
    ref = robot_reference_wrist_positions()
    ms = {}
    for pose in CALIBRATION_POSE_IDS:
        ms[pose] = PoseMeasurement(
            pose_id=pose,
            left_wrist_mean=ref[pose]["left"].copy(),
            right_wrist_mean=ref[pose]["right"].copy(),
            sample_count=50,
            left_wrist_vel_rms_mps=0.01,
            right_wrist_vel_rms_mps=0.01,
        )
    return fit_calibration(ms, operator_id="synthetic")


def _vr_pose(
    *,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    head_pos: np.ndarray,
    head_yaw_rad: float = 0.0,
) -> np.ndarray:
    """Build a ``(3, 7)`` VR pose array compatible with ``VRArmTeleopCalibrated.step``.

    Wrist quaternions are slightly off-identity so the dropout
    detector (which treats identity-quaternion samples as "controller
    lost tracking") doesn't reject them.
    """
    qhead = sRot.from_euler("z", head_yaw_rad).as_quat(scalar_first=True)
    # 5-degree rotation about z = clearly non-identity. Real Quest 3
    # controllers never emit pure identity quats during normal use.
    qwrist = sRot.from_euler("z", np.deg2rad(5.0)).as_quat(scalar_first=True)
    out = np.zeros((3, 7), dtype=np.float64)
    out[0, :3] = left_wrist
    out[0, 3:] = qwrist
    out[1, :3] = right_wrist
    out[1, 3:] = qwrist
    out[2, :3] = head_pos
    out[2, 3:] = qhead
    return out


def test_constructor_requires_calibration() -> None:
    with pytest.raises(ValueError):
        VRArmTeleopCalibrated(calibration=None)


def test_idle_returns_neutral_q_with_zero_residuals() -> None:
    cal = _make_identity_calibration()
    teleop = VRArmTeleopCalibrated(calibration=cal)
    pose = _vr_pose(
        left_wrist=np.array([0.0, 0.3, 1.4]),
        right_wrist=np.array([0.0, -0.3, 1.4]),
        head_pos=np.array([0.0, 0.0, 1.7]),
    )
    res = teleop.step(pose)
    assert isinstance(res, CalibratedTeleopTickResult)
    assert not res.engaged
    np.testing.assert_allclose(res.left_q, teleop.left_neutral_q)
    np.testing.assert_allclose(res.right_q, teleop.right_neutral_q)
    assert res.left_ik.pos_err_m == 0.0
    assert res.right_ik.pos_err_m == 0.0


def test_engaged_step_drives_wrist_target_through_calibration() -> None:
    """When operator wrist is at arms-down calibration pose, the
    composed robot wrist target must equal the FK reference."""
    cal = _make_identity_calibration()
    ref = robot_reference_wrist_positions()
    teleop = VRArmTeleopCalibrated(calibration=cal)
    teleop.set_engaged(True)

    # In an identity calibration, op_wrist_in_head_yaw == ref wrist.
    # Build a vr pose where (wrist_world - head_world) == ref position
    # in the head-yaw frame (head at origin, head yaw = 0).
    head = np.array([0.0, 0.0, 1.7])
    pose = _vr_pose(
        left_wrist=head + ref["arms_down"]["left"],
        right_wrist=head + ref["arms_down"]["right"],
        head_pos=head,
        head_yaw_rad=0.0,
    )
    res = teleop.step(pose)
    assert res.engaged
    np.testing.assert_allclose(res.left_target_pos, ref["arms_down"]["left"])
    np.testing.assert_allclose(res.right_target_pos, ref["arms_down"]["right"])


def test_head_yaw_invariance_when_operator_rotates_with_arms_static() -> None:
    """The big test: operator turns in place 90 deg while keeping arms
    static relative to their head. Robot wrist target MUST stay put.

    This is the regression test for the engage-anchor "hands behind the
    body" bug: the old solver computed deltas in world frame and would
    sweep the wrist target around the side of the robot when the
    operator rotated. The calibrated head-yaw mapping must be
    invariant to body rotation.
    """
    cal = _make_identity_calibration()
    ref = robot_reference_wrist_positions()
    teleop = VRArmTeleopCalibrated(calibration=cal)
    teleop.set_engaged(True)

    head = np.array([0.0, 0.0, 1.7])

    # Initial pose: facing +X (yaw=0), wrists at arms-down reference.
    pose0 = _vr_pose(
        left_wrist=head + ref["arms_down"]["left"],
        right_wrist=head + ref["arms_down"]["right"],
        head_pos=head,
        head_yaw_rad=0.0,
    )
    res0 = teleop.step(pose0)
    target0_left = res0.left_target_pos.copy()
    target0_right = res0.right_target_pos.copy()

    # Operator rotates 90 deg (yaw = +pi/2). In world frame their wrists
    # have to rotate with them: if the wrist was at body +Y in initial
    # orientation, it's now at world -X.
    yaw = np.pi / 2
    R = sRot.from_euler("z", yaw).as_matrix()
    rotated_left = R @ ref["arms_down"]["left"]
    rotated_right = R @ ref["arms_down"]["right"]

    pose1 = _vr_pose(
        left_wrist=head + rotated_left,
        right_wrist=head + rotated_right,
        head_pos=head,
        head_yaw_rad=yaw,
    )
    res1 = teleop.step(pose1)

    # The robot wrist target should be UNCHANGED (operator's wrist
    # didn't move relative to their body).
    np.testing.assert_allclose(res1.left_target_pos, target0_left, atol=1e-9)
    np.testing.assert_allclose(res1.right_target_pos, target0_right, atol=1e-9)


def test_head_translation_invariance() -> None:
    """Operator walks 1 m forward without moving their arms relative to
    their head. Robot wrist target must stay put.
    """
    cal = _make_identity_calibration()
    ref = robot_reference_wrist_positions()
    teleop = VRArmTeleopCalibrated(calibration=cal)
    teleop.set_engaged(True)

    head0 = np.array([0.0, 0.0, 1.7])
    pose0 = _vr_pose(
        left_wrist=head0 + ref["t_pose"]["left"],
        right_wrist=head0 + ref["t_pose"]["right"],
        head_pos=head0,
    )
    res0 = teleop.step(pose0)
    target0_left = res0.left_target_pos.copy()

    head1 = head0 + np.array([1.0, 0.0, 0.0])
    pose1 = _vr_pose(
        left_wrist=head1 + ref["t_pose"]["left"],
        right_wrist=head1 + ref["t_pose"]["right"],
        head_pos=head1,
    )
    res1 = teleop.step(pose1)
    np.testing.assert_allclose(res1.left_target_pos, target0_left, atol=1e-9)


def test_set_engaged_toggles_state() -> None:
    cal = _make_identity_calibration()
    teleop = VRArmTeleopCalibrated(calibration=cal)
    assert teleop.is_engaged is False
    teleop.set_engaged(True)
    assert teleop.is_engaged is True
    teleop.set_engaged(False)
    assert teleop.is_engaged is False


def test_step_rejects_wrong_shape() -> None:
    cal = _make_identity_calibration()
    teleop = VRArmTeleopCalibrated(calibration=cal)
    with pytest.raises(ValueError):
        teleop.step(np.zeros((2, 7)))


def test_ik_converges_within_a_few_steps_on_static_target() -> None:
    """With identity calibration + a static arms-down target, the IK
    should reach the reference pose within a few ticks (< 5 cm pos err).
    """
    cal = _make_identity_calibration()
    ref = robot_reference_wrist_positions()
    teleop = VRArmTeleopCalibrated(calibration=cal)
    teleop.set_engaged(True)

    head = np.array([0.0, 0.0, 1.7])
    pose = _vr_pose(
        left_wrist=head + ref["arms_down"]["left"],
        right_wrist=head + ref["arms_down"]["right"],
        head_pos=head,
    )
    last = None
    for _ in range(50):
        last = teleop.step(pose)
    assert last is not None
    assert last.left_ik.pos_err_m < 0.05, f"left pos err did not converge: {last.left_ik.pos_err_m}"
    assert last.right_ik.pos_err_m < 0.05, f"right pos err did not converge: {last.right_ik.pos_err_m}"
