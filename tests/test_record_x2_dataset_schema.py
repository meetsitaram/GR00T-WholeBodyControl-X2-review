"""Schema-level smoke tests for the Quest 3 → X2 LeRobot recorder.

These tests exercise the *non*-runtime pieces (no MuJoCo, no GPU, no
live VR connection): hand retargeting, VR arm teleop step, the
recorder's body-q composer, and the LeRobot feature schema as it
will be written for the new recorder.

The tokenizer + full :class:`X2DatasetRecorder` lifecycle are gated
on the SONIC checkpoint being present locally (see
:data:`DEFAULT_SONIC_CHECKPOINT` in
:mod:`gear_sonic.scripts.record_synthetic_smoketest_dataset`); when
that file is missing we skip cleanly rather than failing CI.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gear_sonic.data.features_x2_vla import (
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    FPS as DATASET_FPS,
    FRONT_CAM_HEIGHT,
    FRONT_CAM_WIDTH,
    HAND_DOF_OMNI,
    SONIC_MOTION_TOKEN_DIM,
    get_features_x2_vla,
    get_modality_config_x2_vla,
    get_x2_robot_model,
)
from gear_sonic.scripts.record_synthetic_smoketest_dataset import (
    DEFAULT_SONIC_CHECKPOINT,
)
from gear_sonic.utils.teleop.x2_hand_retarget import (
    HAND_FINGER_NAMES_PER_SIDE,
    NUM_HAND_DOF_PER_SIDE,
    controller_grasp_ratio,
    grasp_command_from_ratio,
)
from gear_sonic.utils.teleop.vr_arm_teleop import (
    DEFAULT_LEFT_ARM_NEUTRAL_RAD,
    DEFAULT_RIGHT_ARM_NEUTRAL_RAD,
    VRArmTeleop,
)


# ── Hand retargeting ──────────────────────────────────────────────────────


def test_hand_finger_count_constant():
    assert len(HAND_FINGER_NAMES_PER_SIDE) == NUM_HAND_DOF_PER_SIDE == 10


@pytest.mark.parametrize("side", ["left", "right"])
def test_grasp_open_close_endpoints(side):
    open_q = grasp_command_from_ratio(side, 0.0)
    close_q = grasp_command_from_ratio(side, 1.0)
    assert open_q.shape == (NUM_HAND_DOF_PER_SIDE,)
    assert close_q.shape == (NUM_HAND_DOF_PER_SIDE,)
    # Closed state should differ meaningfully from open: sum of
    # absolute deltas > 1 rad over 10 motors.
    assert np.sum(np.abs(close_q - open_q)) > 1.0


@pytest.mark.parametrize("side", ["left", "right"])
def test_grasp_ratio_clamps_input(side):
    out_high = grasp_command_from_ratio(side, 5.0)
    out_low = grasp_command_from_ratio(side, -5.0)
    open_q = grasp_command_from_ratio(side, 0.0)
    close_q = grasp_command_from_ratio(side, 1.0)
    np.testing.assert_allclose(out_high, close_q)
    np.testing.assert_allclose(out_low, open_q)


def test_hand_input_modes():
    assert controller_grasp_ratio(0.7, 0.3, 0.1, 0.9, mode="trigger") == (0.7, 0.3)
    assert controller_grasp_ratio(0.7, 0.3, 0.1, 0.9, mode="grip") == (0.1, 0.9)
    assert controller_grasp_ratio(0.7, 0.3, 0.1, 0.9, mode="max") == (0.7, 0.9)
    with pytest.raises(ValueError):
        controller_grasp_ratio(0.0, 0.0, 0.0, 0.0, mode="bogus")


# ── VR arm teleop step ────────────────────────────────────────────────────


def test_teleop_step_idempotent_when_anchored():
    """Engaging at the FK pose then stepping with the same VR pose
    should keep the joint angles within tight numerical tolerance."""
    teleop = VRArmTeleop(rotation_weight=0.0)

    from scipy.spatial.transform import Rotation as sRot
    from gear_sonic.utils.teleop.solver.arm import ArmIKSolver

    l = ArmIKSolver(side="left")
    r = ArmIKSolver(side="right")
    l_pos, _ = l.fk(np.asarray(DEFAULT_LEFT_ARM_NEUTRAL_RAD))
    r_pos, _ = r.fk(np.asarray(DEFAULT_RIGHT_ARM_NEUTRAL_RAD))
    q_id = sRot.identity().as_quat(scalar_first=True)

    vr = np.zeros((3, 7), dtype=np.float64)
    vr[0, :3] = l_pos
    vr[0, 3:] = q_id
    vr[1, :3] = r_pos
    vr[1, 3:] = q_id
    vr[2, :3] = (0.0, 0.0, 1.6)
    vr[2, 3:] = q_id

    teleop.engage(vr)
    assert teleop.is_engaged

    last_l = teleop.left_q.copy()
    last_r = teleop.right_q.copy()
    for _ in range(20):
        res = teleop.step(vr)
        assert res.engaged
    np.testing.assert_allclose(res.left_q, last_l, atol=2e-3)
    np.testing.assert_allclose(res.right_q, last_r, atol=2e-3)


def test_teleop_step_unengaged_returns_neutral():
    teleop = VRArmTeleop()
    vr = np.zeros((3, 7), dtype=np.float64)
    vr[:, 3] = 1.0  # quat w=1
    res = teleop.step(vr)
    assert not res.engaged
    np.testing.assert_allclose(res.left_q, np.asarray(DEFAULT_LEFT_ARM_NEUTRAL_RAD))
    np.testing.assert_allclose(res.right_q, np.asarray(DEFAULT_RIGHT_ARM_NEUTRAL_RAD))


# ── Dataset feature schema ────────────────────────────────────────────────


def test_dataset_features_match_recorder_outputs():
    """v1 SONIC schema (post_sonic_canonical=True, the default)."""
    rm = get_x2_robot_model(hand_variant="omnihand_10")
    features = get_features_x2_vla(rm, hand_dof_per_side=HAND_DOF_OMNI)

    state_shape = tuple(features["observation.state"]["shape"])
    assert state_shape == (rm.num_joints + 2 * HAND_DOF_OMNI,)
    assert tuple(features["observation.images.ego_view"]["shape"]) == (
        EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3,
    )
    assert tuple(features["action.motion_token"]["shape"]) == (
        SONIC_MOTION_TOKEN_DIM,
    )
    # Bare-canonical body action (post-SONIC executed q in SONIC mode,
    # commanded q in kinematic mode).
    assert tuple(features["action.body_q_mj"]["shape"]) == (rm.num_joints,)
    # Names must be in MuJoCo order, not Pinocchio order. The MJ layout
    # exposes head as the LAST two scalars (slots 29..31) whereas
    # Pinocchio puts head between waist and left_arm.
    body_names = features["action.body_q_mj"]["names"]
    assert body_names[-2:] == ["head_yaw_joint", "head_pitch_joint"], (
        f"action.body_q_mj must be MuJoCo-ordered; got tail={body_names[-2:]}"
    )
    assert tuple(features["action.left_hand_joints"]["shape"]) == (HAND_DOF_OMNI,)
    assert tuple(features["action.right_hand_joints"]["shape"]) == (HAND_DOF_OMNI,)
    assert tuple(features["observation.projected_gravity"]["shape"]) == (3,)
    # v1 SONIC schema also adds debug-only sibling columns and a
    # corrective-delta scalar. These are NOT in get_modality_config_x2_vla.
    assert tuple(features["action.body_q_mj_pre_sonic"]["shape"]) == (rm.num_joints,)
    pre_names = features["action.body_q_mj_pre_sonic"]["names"]
    assert pre_names == body_names, "pre_sonic must mirror canonical names"
    assert tuple(features["action.left_hand_joints_pre_sonic"]["shape"]) == (HAND_DOF_OMNI,)
    assert tuple(features["action.right_hand_joints_pre_sonic"]["shape"]) == (HAND_DOF_OMNI,)
    assert tuple(features["action.sonic_correction_max_rad"]["shape"]) == (1,)
    assert features["action.sonic_correction_max_rad"]["dtype"] == "float32"


def test_kinematic_features_omit_pre_sonic_columns():
    """post_sonic_canonical=False schema is what teleop_x2_kinematic writes."""
    rm = get_x2_robot_model(hand_variant="omnihand_10")
    features = get_features_x2_vla(
        rm, hand_dof_per_side=HAND_DOF_OMNI, post_sonic_canonical=False
    )
    # Bare canonical columns are present.
    assert "action.body_q_mj" in features
    assert "action.left_hand_joints" in features
    assert "action.right_hand_joints" in features
    # SONIC-only debug siblings are absent.
    for k in (
        "action.body_q_mj_pre_sonic",
        "action.left_hand_joints_pre_sonic",
        "action.right_hand_joints_pre_sonic",
        "action.sonic_correction_max_rad",
    ):
        assert k not in features, (
            f"{k} should not be in the kinematic feature schema"
        )


def test_modality_config_consistency():
    rm = get_x2_robot_model(hand_variant="omnihand_10")
    cfg = get_modality_config_x2_vla(rm, hand_dof_per_side=HAND_DOF_OMNI)
    assert "state" in cfg and "action" in cfg and "video" in cfg
    assert "ego_view" in cfg["video"]
    # Hand slices land *inside* observation.state.
    state_dim = rm.num_joints + 2 * HAND_DOF_OMNI
    for side in ("left_hand", "right_hand"):
        sl = cfg["state"][side]
        assert sl["original_key"] == "observation.state"
        assert 0 <= sl["start"] < sl["end"] <= state_dim


# ── front_cam (witness camera) opt-in schema ──────────────────────────────


def test_front_cam_default_off_keeps_legacy_schema():
    """Without ``include_front_cam=True`` the schema must be byte-identical
    to the pre-front_cam recorder output -- otherwise old datasets in the
    wild can't be re-validated by the exporter."""
    rm = get_x2_robot_model(hand_variant="omnihand_10")
    features = get_features_x2_vla(rm, hand_dof_per_side=HAND_DOF_OMNI)
    image_keys = sorted(k for k in features if k.startswith("observation.images"))
    assert image_keys == ["observation.images.ego_view"], (
        f"front_cam must NOT appear in default features; got {image_keys}"
    )
    cfg = get_modality_config_x2_vla(rm, hand_dof_per_side=HAND_DOF_OMNI)
    assert sorted(cfg["video"]) == ["ego_view"], (
        f"front_cam must NOT appear in default modality.video; got "
        f"{sorted(cfg['video'])}"
    )


def test_front_cam_include_adds_video_feature():
    """``include_front_cam=True`` must add the second video feature with
    the documented shape AND register it in ``modality.video``. Both
    sides MUST agree because the LeRobot exporter cross-checks them at
    first-frame validation."""
    rm = get_x2_robot_model(hand_variant="omnihand_10")
    features = get_features_x2_vla(
        rm, hand_dof_per_side=HAND_DOF_OMNI, include_front_cam=True,
    )
    assert "observation.images.front_cam" in features
    fc = features["observation.images.front_cam"]
    assert fc["dtype"] == "video"
    assert tuple(fc["shape"]) == (FRONT_CAM_HEIGHT, FRONT_CAM_WIDTH, 3)
    assert fc["names"] == ["height", "width", "channel"]
    # ego_view must still be present (we add front_cam, not replace).
    assert "observation.images.ego_view" in features

    cfg = get_modality_config_x2_vla(
        rm, hand_dof_per_side=HAND_DOF_OMNI, include_front_cam=True,
    )
    assert sorted(cfg["video"]) == ["ego_view", "front_cam"]
    assert cfg["video"]["front_cam"]["original_key"] == (
        "observation.images.front_cam"
    )


def test_front_cam_resolver_default_in_record_cli():
    """The CLI helper `_resolve_front_cam_default` must:
      * default to True iff a scene XML is loaded (robocasa mode), AND
      * always honour an explicit operator flag (True or False).

    The wrapper script relies on this so passing only ``--robocasa-env``
    automatically lights up the second video track.
    """
    from gear_sonic.scripts.record_x2_dataset import (
        _resolve_front_cam_default,
    )
    fake_scene = Path("/tmp/never_actually_read.xml")
    assert _resolve_front_cam_default(None, None) is False
    assert _resolve_front_cam_default(None, fake_scene) is True
    assert _resolve_front_cam_default(False, fake_scene) is False
    assert _resolve_front_cam_default(True, None) is True


def test_front_cam_baked_into_robocasa_scene_xmls():
    """Each shipped robocasa scene XML must declare a ``front_cam``
    camera with the documented pose + 120° FoV. Catches accidental
    rebuilds that drop the second camera."""
    import mujoco

    scenes_dir = (
        Path(__file__).resolve().parent.parent
        / "gear_sonic" / "data" / "assets" / "robocasa_scenes"
    )
    for env in ("X2PickPlaceCube", "X2PickPlaceBowl", "X2PickPlaceApple"):
        xml = scenes_dir / f"{env}.xml"
        if not xml.is_file():
            pytest.skip(f"scene XML missing: {xml}")
        m = mujoco.MjModel.from_xml_path(str(xml))
        cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "front_cam")
        assert cam_id >= 0, f"{env}: front_cam not in compiled MJCF"
        # Pose: 3 ft (~0.9144 m) in front of the robot launch, chest height.
        np.testing.assert_allclose(
            m.cam_pos[cam_id], [0.9144, 0.0, 1.10], atol=1e-3,
            err_msg=f"{env}: front_cam pos drifted",
        )
        # Wide-angle 120° vertical FoV (per user spec).
        assert float(m.cam_fovy[cam_id]) == pytest.approx(120.0, abs=1e-3), (
            f"{env}: front_cam fovy != 120 (got {float(m.cam_fovy[cam_id])})"
        )
        # mode=0 -> mjCAMLIGHT_FIXED (camera does NOT track the robot).
        assert int(m.cam_mode[cam_id]) == 0, (
            f"{env}: front_cam must be mode=fixed (got {int(m.cam_mode[cam_id])})"
        )


# ── Online tokenizer (gated on the SONIC checkpoint) ──────────────────────


def _has_sonic_checkpoint() -> bool:
    return DEFAULT_SONIC_CHECKPOINT.exists()


@pytest.mark.skipif(
    not _has_sonic_checkpoint(),
    reason=f"SONIC checkpoint not found at {DEFAULT_SONIC_CHECKPOINT}",
)
def test_online_tokenizer_returns_64d_per_frame():
    from gear_sonic.scripts.live_vla_publish_motion_token import (
        DEFAULT_STAND_POSE_MUJOCO_RAD,
    )
    from gear_sonic.utils.teleop.online_sonic_tokenizer import (
        OnlineSonicTokenizer,
    )

    tok = OnlineSonicTokenizer.from_checkpoint(
        DEFAULT_SONIC_CHECKPOINT,
        device="cpu",
        motion_fps=float(DATASET_FPS),
    )
    body = np.asarray(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
    out = tok.encode(body)
    assert out.shape == (SONIC_MOTION_TOKEN_DIM,)
    assert out.dtype == np.float64
    # FSQ lattice values must round to k * 2/32 exactly.
    step = 2.0 / 32.0
    near = out / step
    assert np.allclose(near, np.round(near), atol=1e-5)
