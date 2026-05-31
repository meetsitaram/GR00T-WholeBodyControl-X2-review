"""Regression: finger filter must not freeze XRHand curls across mode switches."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.utils.teleop.finger_signal_filter import FingerFilterParams
from gear_sonic.utils.teleop.operator_calibration import OperatorCalibration
from gear_sonic.utils.teleop.x2_hand_retarget import grasp_command_from_ratio
from gear_sonic.utils.teleop.x2_retarget_pipeline import (
    Retargeter,
    RetargetTickInput,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_CAL = REPO_ROOT / "data" / "operator_calibrations" / "default.yaml"


def _identity_vr_pose() -> np.ndarray:
    q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    row = np.concatenate([np.zeros(3, dtype=np.float64), q])
    return np.stack([row, row, row], axis=0)


@pytest.mark.skipif(not _CAL.is_file(), reason=f"missing calibration {_CAL}")
def test_retargeter_clears_finger_filter_when_hand_source_changes() -> None:
    """After ``hand`` -> ``controller`` + raw curls ``None``, triggers drive hand_q.

    Without a filter reset on ``left_hand_source`` transition, the EMA
    would keep outputting the last finite XRHand curls and the
    retargeter would never take the trigger/grip fallback.
    """
    cal = OperatorCalibration.load_yaml(_CAL)
    r = Retargeter(
        calibration=cal,
        finger_filter_params=FingerFilterParams(),
        hand_input_mode="trigger",
    )
    r.set_engaged(False)
    pose = _identity_vr_pose()
    tr_open = (0.0, 0.0, 0.0, 0.0)
    tr_squeeze = (0.95, 0.0, 0.0, 0.0)

    curls_hand = np.array([0.9, 0.9, 0.9, 0.9, 0.9], dtype=np.float64)
    out_hand = r.step(
        RetargetTickInput(
            vr_pose=pose,
            triggers=tr_open,
            left_curls=curls_hand,
            right_curls=curls_hand,
            left_thumb_oppose=0.5,
            right_thumb_oppose=0.5,
            left_finger_tip_oppose=np.zeros(4, dtype=np.float64),
            right_finger_tip_oppose=np.zeros(4, dtype=np.float64),
            left_hand_source="hand",
            right_hand_source="hand",
        )
    )
    assert out_hand.left_hand_q.shape == (10,)

    out_ctrl = r.step(
        RetargetTickInput(
            vr_pose=pose,
            triggers=tr_squeeze,
            left_curls=None,
            right_curls=None,
            left_thumb_oppose=None,
            right_thumb_oppose=None,
            left_finger_tip_oppose=None,
            right_finger_tip_oppose=None,
            left_hand_source="controller",
            right_hand_source="controller",
        )
    )
    want_l = grasp_command_from_ratio("left", float(tr_squeeze[0]))
    want_r = grasp_command_from_ratio("right", float(tr_squeeze[1]))
    np.testing.assert_allclose(out_ctrl.left_hand_q, want_l, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(out_ctrl.right_hand_q, want_r, rtol=0.0, atol=1e-9)
