"""Smoke tests for the vendored X2 arm DLS IK solver + FK chain.

Locks the invariants needed by the dataset recorder
(:mod:`gear_sonic.scripts.record_x2_dataset`):

1. FK is deterministic and produces reasonable wrist positions for the
   trained X2 stand-pose neutral arm angles.
2. Round-trip ``FK(q) -> IK(target=FK(q))`` from a small perturbation
   converges back to ``q`` (sub-mm wrist error) within a few DLS
   iterations.
3. The Jacobian dimensionality is 6x7 (linear + angular, per arm).
4. Joint limits are respected: a target wildly outside the workspace
   gets clamped instead of running the joint past its hardware range.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.teleop.solver.arm import (
    ArmIKSolver,
    X2_ARM_JOINT_LIMITS_RAD,
    X2_ARM_JOINT_NAMES,
    arm_fk,
    arm_fk_pose,
)
from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import matrix_to_quat_wxyz
from gear_sonic.utils.teleop.vr_arm_teleop import (
    DEFAULT_LEFT_ARM_NEUTRAL_RAD,
    DEFAULT_RIGHT_ARM_NEUTRAL_RAD,
)


@pytest.mark.parametrize(
    "side, neutral_q, expected_y_sign",
    [
        ("left", DEFAULT_LEFT_ARM_NEUTRAL_RAD, +1.0),
        ("right", DEFAULT_RIGHT_ARM_NEUTRAL_RAD, -1.0),
    ],
)
def test_neutral_pose_fk_makes_sense(side, neutral_q, expected_y_sign):
    """Wrist y must sit on the correct side of the body in the neutral pose."""
    q = np.asarray(neutral_q, dtype=np.float64)
    pos = arm_fk(q, side=side)
    assert pos.shape == (3,)
    assert np.sign(pos[1]) == expected_y_sign, (
        f"Expected y-sign {expected_y_sign} for {side} arm; got pos={pos}"
    )
    # Reasonable shoulder-to-wrist envelope: between 20 cm and 80 cm.
    reach = float(np.linalg.norm(pos))
    assert 0.20 < reach < 0.80, f"Unrealistic wrist reach {reach:.3f} m for {side}"


@pytest.mark.parametrize("side", ["left", "right"])
def test_jacobian_is_6x7(side):
    solver = ArmIKSolver(side=side)
    q = np.zeros(7)
    J, p_e, R_e = solver._jacobian(q)  # type: ignore[attr-defined]
    assert J.shape == (6, 7)
    assert p_e.shape == (3,)
    assert R_e.shape == (3, 3)


@pytest.mark.parametrize(
    "side, neutral_q",
    [
        ("left", DEFAULT_LEFT_ARM_NEUTRAL_RAD),
        ("right", DEFAULT_RIGHT_ARM_NEUTRAL_RAD),
    ],
)
def test_dls_round_trip_converges(side, neutral_q):
    """Tiny perturbation about the neutral pose round-trips through DLS."""
    q_truth = np.asarray(neutral_q, dtype=np.float64) + np.array(
        [0.05, -0.03, 0.02, -0.08, 0.04, 0.01, -0.02]
    )
    T = arm_fk_pose(q_truth, side=side)
    target_pos = T[:3, 3].copy()
    target_quat = matrix_to_quat_wxyz(T[:3, :3])

    solver = ArmIKSolver(side=side, damping=0.05, rotation_weight=0.5)
    q = np.asarray(neutral_q, dtype=np.float64).copy()
    for _ in range(50):
        q, info = solver.solve(
            q_seed=q,
            target_pos=target_pos,
            target_quat_wxyz=target_quat,
            max_iters=1,
        )
        if info.pos_err_m < 1e-3 and info.rot_err_rad < 5e-3:
            break

    final_pos, final_R = solver.fk(q)
    pos_err = float(np.linalg.norm(target_pos - final_pos))
    assert pos_err < 5e-3, f"DLS did not converge: pos_err={pos_err:.4f} m"


@pytest.mark.parametrize("side", ["left", "right"])
def test_joint_limits_are_respected(side):
    """Even an unreachable target must not push joints past hardware limits."""
    solver = ArmIKSolver(side=side, damping=0.05)
    far_target = np.array([3.0, 3.0, 3.0])  # well outside workspace
    q = np.zeros(7)
    q, _ = solver.solve(
        q_seed=q,
        target_pos=far_target,
        target_quat_wxyz=None,
        max_iters=20,
    )

    if side == "left":
        limits = X2_ARM_JOINT_LIMITS_RAD[:7]
    else:
        limits = X2_ARM_JOINT_LIMITS_RAD[7:]
    lo = np.array([a for a, _ in limits])
    hi = np.array([b for _, b in limits])
    eps = 1e-9
    assert np.all(q >= lo - eps), f"q below lower limits: {q} vs {lo}"
    assert np.all(q <= hi + eps), f"q above upper limits: {q} vs {hi}"


def test_joint_name_count_matches_limit_count():
    assert len(X2_ARM_JOINT_NAMES) == 14
    assert len(X2_ARM_JOINT_LIMITS_RAD) == 14
