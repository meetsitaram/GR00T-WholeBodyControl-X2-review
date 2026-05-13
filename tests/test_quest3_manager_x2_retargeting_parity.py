"""BLOCKING parity test: new Retargeter must reproduce the X2 v6 dataset.

This test pipes the recorded ``debug/teleop_episode_000000.npz`` from
``data/lerobot/x2_quest3_kinematic_v6/`` through the new
:class:`gear_sonic.utils.teleop.x2_retarget_pipeline.Retargeter` and
diffs the per-tick outputs against the recorded ground truth:

- ``ik_left_q_rad`` / ``ik_right_q_rad``           -- arm IK (degrees)
- ``commanded_body_q_mj``                          -- 31-DOF body command
- ``commanded_left_hand_q`` / ``commanded_right_hand_q``  -- OmniHand cmd
- ``quest_*_hand_curls_filtered``                  -- finger filter output

This is the GATE for Phase 0 Step 2: the new manager script
``quest3_manager_x2.py`` shares its retargeting code path with this
test. If the test passes, the lift-and-shift cannot have silently
regressed the X2 arm-and-hand quality bar that v6 baked in.

Acceptance thresholds
---------------------

- Hand commands (10 DOF per side): float32 ULP tolerance (< 1e-6).
  The retargeter uses float64 internally; the recorded ground truth is
  stored as float32, so the comparison eats one cast's worth of
  rounding. Anything bigger than ~1e-7 would indicate a real
  semantic divergence in the per_finger / curl path.
- Filtered curls: float32 ULP tolerance (< 1e-6) for the same reason.
- IK arm joints, post-engagement, no dropouts:
    per-motor L1 < 0.5 deg, per-motor Linf < 2.0 deg
- Body q (31 DOF), same mask: per-joint Linf < 2.0 deg

Methodology notes
-----------------

The Retargeter is initialised with the NPZ's frame-0 IK output (not
the default neutral pose) so the IK starts from the same DLS state as
the recorder did at v6 recording time. Without this, the first
~30 frames look like a >50 deg "transient" while the IK converges from
neutral; that transient is meaningless for parity (it just measures
DLS warmup, not retargeting fidelity).

Engagement state is replayed from the NPZ's ``engaged`` field per
frame. Dropout frames are masked OUT of the metric because the
recorder's dropout-hold logic is intentionally non-deterministic w.r.t.
the input-only replay (the "last good" target depends on history that
the NPZ doesn't fully preserve).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
NPZ_PATH = (
    REPO_ROOT / "data" / "lerobot" / "x2_quest3_kinematic_v6"
    / "debug" / "teleop_episode_000000.npz"
)
CALIBRATION_PATH = REPO_ROOT / "data" / "operator_calibrations" / "default.yaml"


# Acceptance thresholds (degrees for arm joints; native units for hands).
ARM_PER_MOTOR_L1_DEG = 0.5
ARM_PER_MOTOR_LINF_DEG = 2.0
BODY_LINF_DEG = 2.0

# Float32 ULP tolerance: 1 ULP at float32 around values in [0, 1] is
# ~1.2e-7. The recorded NPZ stores hand commands and filter outputs as
# float32 while our retargeter computes in float64, so a single
# downcast eats ~one ULP of rounding noise. We allow up to 1e-6 (8x
# margin) for both kinds; anything larger would mean a real semantic
# divergence in the per-finger path.
HAND_AND_FILTER_LINF = 1e-6


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _data_files_present() -> bool:
    return NPZ_PATH.is_file() and CALIBRATION_PATH.is_file()


pytestmark = pytest.mark.skipif(
    not _data_files_present(),
    reason=(
        f"Parity test requires {NPZ_PATH.relative_to(REPO_ROOT)} and "
        f"{CALIBRATION_PATH.relative_to(REPO_ROOT)}. Skipping in CI / "
        "fresh-clone environments without the v6 dataset."
    ),
)


def _maybe_arr(arr: np.ndarray, i: int) -> np.ndarray | None:
    """Return ``arr[i]`` as float64, or ``None`` if the slice is all-NaN."""
    x = arr[i]
    if np.ndim(x) == 0:
        return None if np.isnan(x) else float(x)
    if np.isnan(x).all():
        return None
    return x.astype(np.float64)


@pytest.fixture(scope="module")
def npz_episode():
    return np.load(NPZ_PATH, allow_pickle=True)


@pytest.fixture(scope="module")
def replay_results(npz_episode):
    """Replay the entire NPZ through the Retargeter and capture diffs.

    Module-scoped because replaying the full 4656-frame v6 episode
    takes ~30 s of IK; we don't want to pay that for every assert.
    """
    from gear_sonic.utils.teleop.finger_signal_filter import FingerFilterParams
    from gear_sonic.utils.teleop.operator_calibration import OperatorCalibration
    from gear_sonic.utils.teleop.x2_retarget_pipeline import (
        Retargeter,
        RetargetTickInput,
    )

    npz = npz_episode
    n = int(npz["num_frames"])
    engaged = npz["engaged"]

    cal = OperatorCalibration.load_yaml(CALIBRATION_PATH)
    retargeter = Retargeter(
        calibration=cal,
        finger_filter_params=FingerFilterParams(),
        # Match the recorder's IK state at the first frame so we don't
        # measure DLS convergence transients (see module docstring).
        left_neutral_q=npz["ik_left_q_rad"][0].astype(np.float64),
        right_neutral_q=npz["ik_right_q_rad"][0].astype(np.float64),
    )

    left_diff_per_joint = np.zeros((n, 7))
    right_diff_per_joint = np.zeros((n, 7))
    body_diff_per_joint = np.zeros((n, 31))
    hand_diff_l = np.zeros((n, 10))
    hand_diff_r = np.zeros((n, 10))
    left_drop = np.zeros(n, dtype=bool)
    right_drop = np.zeros(n, dtype=bool)
    filtered_curls_l = np.full((n, 5), np.nan, dtype=np.float32)
    filtered_curls_r = np.full((n, 5), np.nan, dtype=np.float32)

    for i in range(n):
        inp = RetargetTickInput(
            vr_pose=np.array([
                [*npz["vr_left_wrist_pos"][i],  *npz["vr_left_wrist_quat"][i]],
                [*npz["vr_right_wrist_pos"][i], *npz["vr_right_wrist_quat"][i]],
                [*npz["vr_head_pos"][i],        *npz["vr_head_quat"][i]],
            ], dtype=np.float64),
            triggers=tuple(float(x) for x in npz["controller_triggers"][i]),
            left_curls=_maybe_arr(npz["quest_left_hand_curls"], i),
            right_curls=_maybe_arr(npz["quest_right_hand_curls"], i),
            left_thumb_oppose=_maybe_arr(npz["quest_left_thumb_oppose"], i),
            right_thumb_oppose=_maybe_arr(npz["quest_right_thumb_oppose"], i),
            left_finger_tip_oppose=_maybe_arr(
                npz["quest_left_finger_tip_oppose"], i,
            ),
            right_finger_tip_oppose=_maybe_arr(
                npz["quest_right_finger_tip_oppose"], i,
            ),
        )
        retargeter.set_engaged(bool(engaged[i]))
        out = retargeter.step(inp)

        left_diff_per_joint[i] = np.rad2deg(
            np.abs(out.left_arm_q - npz["ik_left_q_rad"][i])
        )
        right_diff_per_joint[i] = np.rad2deg(
            np.abs(out.right_arm_q - npz["ik_right_q_rad"][i])
        )
        body_diff_per_joint[i] = np.rad2deg(
            np.abs(out.body_q_mj - npz["commanded_body_q_mj"][i])
        )
        hand_diff_l[i] = np.abs(out.left_hand_q - npz["commanded_left_hand_q"][i])
        hand_diff_r[i] = np.abs(out.right_hand_q - npz["commanded_right_hand_q"][i])
        left_drop[i] = out.left_dropout
        right_drop[i] = out.right_dropout
        if out.left_curls_filtered is not None:
            filtered_curls_l[i] = out.left_curls_filtered
        if out.right_curls_filtered is not None:
            filtered_curls_r[i] = out.right_curls_filtered

    # Mask: engaged + no dropout. We do NOT need a warmup skip because
    # we initialised the IK to the recorder's frame-0 state.
    mask = engaged & ~left_drop & ~right_drop

    return {
        "n_frames": n,
        "n_clean": int(mask.sum()),
        "mask": mask,
        "left_diff_per_joint": left_diff_per_joint,
        "right_diff_per_joint": right_diff_per_joint,
        "body_diff_per_joint": body_diff_per_joint,
        "hand_diff_l": hand_diff_l,
        "hand_diff_r": hand_diff_r,
        "filtered_curls_l": filtered_curls_l,
        "filtered_curls_r": filtered_curls_r,
        "ref_filtered_curls_l": npz["quest_left_hand_curls_filtered"].astype(np.float32),
        "ref_filtered_curls_r": npz["quest_right_hand_curls_filtered"].astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Sanity check: enough frames in the recorded episode for a meaningful diff
# ---------------------------------------------------------------------------


def test_dataset_has_enough_engaged_frames(replay_results):
    assert replay_results["n_clean"] > 1000, (
        f"v6 episode has only {replay_results['n_clean']} engaged "
        f"non-dropout frames (out of {replay_results['n_frames']}); not "
        "enough to make a meaningful parity assertion. Re-record."
    )


# ---------------------------------------------------------------------------
# Hand commands MUST match exactly (no IK; deterministic mapping)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["l", "r"])
def test_hand_q_match_within_float32_ulp(replay_results, side):
    diff = replay_results[f"hand_diff_{side}"][replay_results["mask"]]
    assert diff.max() < HAND_AND_FILTER_LINF, (
        f"{side}_hand_q diverged from v6: Linf={diff.max():.3e} > "
        f"{HAND_AND_FILTER_LINF:.0e}. The XRHand fast path / per-finger "
        "map / finger filter has a real (not float32 round-off) "
        "regression."
    )


def test_filtered_curls_match_within_float32_ulp(replay_results):
    """The FingerSignalFilter is deterministic; its output should match
    the recorded ``*_filtered`` channel up to float32 round-off."""
    for side in ("l", "r"):
        ours = replay_results[f"filtered_curls_{side}"]
        ref = replay_results[f"ref_filtered_curls_{side}"]
        valid = ~(np.isnan(ours).any(axis=1) | np.isnan(ref).any(axis=1))
        assert valid.sum() > 100, (
            f"filtered_curls_{side}: only {int(valid.sum())} valid frames"
        )
        d = np.abs(ours[valid] - ref[valid])
        assert d.max() < HAND_AND_FILTER_LINF, (
            f"filtered_curls_{side} diverged: Linf={d.max():.3e} > "
            f"{HAND_AND_FILTER_LINF:.0e}"
        )


# ---------------------------------------------------------------------------
# IK arm joints: small numerical drift allowed (DLS is iterative)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["left", "right"])
def test_arm_ik_per_joint_l1(replay_results, side):
    diffs = replay_results[f"{side}_diff_per_joint"]
    d = diffs[replay_results["mask"]]
    per_joint_l1 = d.mean(axis=0)
    assert per_joint_l1.max() < ARM_PER_MOTOR_L1_DEG, (
        f"{side} arm IK per-joint L1 = "
        + " ".join(f"{x:.4f}" for x in per_joint_l1)
        + f" (max {per_joint_l1.max():.4f} deg) exceeds "
        f"{ARM_PER_MOTOR_L1_DEG} deg threshold"
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_arm_ik_per_joint_linf(replay_results, side):
    diffs = replay_results[f"{side}_diff_per_joint"]
    d = diffs[replay_results["mask"]]
    per_joint_linf = d.max(axis=0)
    assert per_joint_linf.max() < ARM_PER_MOTOR_LINF_DEG, (
        f"{side} arm IK per-joint Linf = "
        + " ".join(f"{x:.4f}" for x in per_joint_linf)
        + f" (max {per_joint_linf.max():.4f} deg) exceeds "
        f"{ARM_PER_MOTOR_LINF_DEG} deg threshold"
    )


# ---------------------------------------------------------------------------
# Composed body_q (31-DOF) sanity
# ---------------------------------------------------------------------------


def test_body_q_per_joint_linf(replay_results):
    d = replay_results["body_diff_per_joint"][replay_results["mask"]]
    per_joint_linf = d.max(axis=0)
    assert per_joint_linf.max() < BODY_LINF_DEG, (
        f"body_q_mj per-joint Linf max = {per_joint_linf.max():.4f} "
        f"deg exceeds {BODY_LINF_DEG} deg threshold. "
        f"Worst joint index: {int(per_joint_linf.argmax())}"
    )
