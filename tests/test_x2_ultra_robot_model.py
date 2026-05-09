"""Acceptance gate for the X2 Ultra RobotModel scaffolding.

Verifies that:
- ``instantiate_x2_ultra_robot_model`` builds Pinocchio chains for both
  ``omnihand_10`` and ``g1_compat_7`` variants without errors.
- All 31 X2 body joints resolve through Pinocchio's ``joint_to_dof_index``.
- ``joint_groups`` covers every URDF body joint via the ``body`` aggregate.
- Hand joint *names* and *limits* are exposed even though they are not
  in the URDF kinematic chain.
- The G1-compat variant exposes exactly 7 finger names per side; the
  full OmniHand variant exposes 10.

Run with::

    .venv/bin/python -m pytest tests/test_x2_ultra_robot_model.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.data.robot_model.instantiation import instantiate_x2_ultra_robot_model
from gear_sonic.data.robot_model.supplemental_info.x2_ultra import (
    HandVariant,
    X2UltraSupplementalInfo,
)
from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
    HAND_JOINT_LIMITS,
    OMNIHAND_FINGER_NAMES_PER_SIDE,
    OMNIHAND_LEFT_FINGER_NAMES,
    OMNIHAND_RIGHT_FINGER_NAMES,
    X2_BODY_JOINT_LIMITS,
    X2_BODY_JOINT_NAMES,
)


def test_body_joint_names_match_limits() -> None:
    """Every body joint must have a limit entry."""
    assert len(X2_BODY_JOINT_NAMES) == 31
    assert set(X2_BODY_JOINT_LIMITS.keys()) == set(X2_BODY_JOINT_NAMES)


def test_omnihand_finger_constants() -> None:
    """OmniHand finger constants align with the agitbot record-and-replay layout."""
    assert len(OMNIHAND_FINGER_NAMES_PER_SIDE) == 10
    assert len(OMNIHAND_LEFT_FINGER_NAMES) == 10
    assert len(OMNIHAND_RIGHT_FINGER_NAMES) == 10
    for n in OMNIHAND_LEFT_FINGER_NAMES + OMNIHAND_RIGHT_FINGER_NAMES:
        assert n in HAND_JOINT_LIMITS, f"missing hand limit for {n}"


def test_supplemental_info_omnihand_10() -> None:
    info = X2UltraSupplementalInfo(hand_variant=HandVariant.OMNIHAND_10)
    assert info.name == "X2Ultra_Omnihand10"
    assert len(info.body_actuated_joints) == 31
    # Hands are out-of-band; URDF-side lists must be empty.
    assert info.left_hand_actuated_joints == []
    assert info.right_hand_actuated_joints == []
    # But finger names should be discoverable for downstream consumers.
    assert len(info.left_finger_names) == 10
    assert len(info.right_finger_names) == 10
    assert all(n.startswith("left_") for n in info.left_finger_names)
    assert all(n.startswith("right_") for n in info.right_finger_names)
    # Joint limits cover both body and hand DOFs.
    for n in info.body_actuated_joints:
        assert n in info.joint_limits, f"missing limit for body joint {n}"
    for n in info.left_finger_names + info.right_finger_names:
        assert n in info.joint_limits, f"missing limit for finger {n}"


def test_supplemental_info_g1_compat_7() -> None:
    info = X2UltraSupplementalInfo(hand_variant=HandVariant.G1_COMPAT_7)
    assert info.name == "X2Ultra_G1Compat7"
    assert len(info.left_finger_names) == 7
    assert len(info.right_finger_names) == 7
    # Body DOFs are unchanged across variants.
    assert len(info.body_actuated_joints) == 31


@pytest.mark.parametrize("variant", ["omnihand_10", "g1_compat_7"])
def test_robot_model_instantiates(variant: str) -> None:
    """Pinocchio loads the URDF and resolves every body joint."""
    rm = instantiate_x2_ultra_robot_model(variant)
    assert rm.supplemental_info is not None
    # All 31 body joints should resolve cleanly through Pinocchio.
    for jn in rm.supplemental_info.body_actuated_joints:
        assert jn in rm.joint_to_dof_index, f"Pinocchio missing {jn}"
    # Joint limits array sized correctly.
    assert rm.upper_joint_limits.shape == (31,)
    assert rm.lower_joint_limits.shape == (31,)


def test_joint_groups_aggregate_body() -> None:
    """The ``body`` aggregate group must enumerate every body joint."""
    rm = instantiate_x2_ultra_robot_model("omnihand_10")
    body_indices = rm._joint_group_indices["body"]
    body_joint_count = len(rm.supplemental_info.body_actuated_joints)
    # body == lower_body + upper_body_no_hands == legs + waist + arms + head
    assert len(body_indices) == body_joint_count, (
        f"body group has {len(body_indices)} indices, expected {body_joint_count}"
    )
    # Sanity check: indices are a permutation of [0, 31).
    assert sorted(body_indices) == list(range(body_joint_count))


def test_arm_groups_have_seven_dof_per_side() -> None:
    rm = instantiate_x2_ultra_robot_model("omnihand_10")
    assert len(rm._joint_group_indices["left_arm"]) == 7
    assert len(rm._joint_group_indices["right_arm"]) == 7


def test_leg_groups_have_six_dof_per_side() -> None:
    rm = instantiate_x2_ultra_robot_model("omnihand_10")
    assert len(rm._joint_group_indices["left_leg"]) == 6
    assert len(rm._joint_group_indices["right_leg"]) == 6


def test_waist_and_head_groups() -> None:
    rm = instantiate_x2_ultra_robot_model("omnihand_10")
    assert len(rm._joint_group_indices["waist"]) == 3
    assert len(rm._joint_group_indices["head"]) == 2


def test_hand_groups_are_empty_in_urdf_view() -> None:
    """Hand groups stay empty for Pinocchio consumers (URDF has no hand joints)."""
    rm = instantiate_x2_ultra_robot_model("omnihand_10")
    assert rm._joint_group_indices["left_hand"] == []
    assert rm._joint_group_indices["right_hand"] == []
    assert rm._joint_group_indices["hands"] == []


def test_hand_rotation_correction_is_identity_on_y_axis() -> None:
    info = X2UltraSupplementalInfo(hand_variant=HandVariant.OMNIHAND_10)
    expected = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    assert np.array_equal(info.hand_rotation_correction, expected)


def test_invalid_hand_variant_raises() -> None:
    with pytest.raises(ValueError):
        instantiate_x2_ultra_robot_model("not_a_real_variant")  # type: ignore[arg-type]
