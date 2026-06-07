"""
LeRobot v2.1 feature schema and modality config for the AgiBot X2 Ultra
SONIC VLA pipeline.

Schema design summary
---------------------

The X2 dataset writes a single concatenated proprioception vector
``observation.state`` of shape ``(num_body_joints + 2 * hand_dof,)`` =
``(31 + 20,)`` = ``(51,)`` for the ``omnihand_10`` variant. The body
slice is laid out in Pinocchio's URDF parse order (driven by the X2
``RobotModel``); the hand slice is appended after the body and is
out-of-band (the X2 URDF does not contain finger joints, so hand
positions come from the AimDK HAL via the ``hand_joints`` field of
the C++ deploy's ``x2_debug`` ZMQ topic).

Action space mirrors ``unitree_g1_sonic`` for cross-embodiment compat:
``action.motion_token`` (64-D SONIC latent) plus
``action.left_hand_joints`` / ``action.right_hand_joints`` (10-D each
in the OmniHand variant, or 7-D when used with the G1-compatible
modality config). The 10-DOF dataset can be down-projected to 7-DOF
on the fly during training via the modality registry alias loaded
through ``--modality-config-path``.

The schema targets GR00T / ``unitree_g1_sonic``-compatible training.
Raw Quest 3 inputs (per-finger curls, thumb opposition, full 3-point VR
pose, IK diagnostics) are **not** duplicated here: they live in the
side-channel **debug NPZ** written by :file:`teleop_x2_kinematic.py` /
the recorder (``…/debug/teleop_episode_NNNNNN.npz``) for offline
analysis and sim-to-real debugging.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from gear_sonic.data.robot_model import RobotModel


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EGO_VIEW_HEIGHT: int = 480
"""Ego-view image height (pixels). Matches the AimDK head RGB-D camera."""

EGO_VIEW_WIDTH: int = 640
"""Ego-view image width (pixels)."""

FRONT_CAM_HEIGHT: int = 480
"""``front_cam`` image height (pixels). Matched to ``EGO_VIEW_HEIGHT``
so both video tracks land in the same per-episode mp4 codec config and
the trainer can assemble matched RGB pairs without per-camera resize
gymnastics. The render path itself is independent of the head camera
(see :class:`gear_sonic.scripts.render_smoketest_episode_video.MujocoFrameRenderer`)."""

FRONT_CAM_WIDTH: int = 640
"""``front_cam`` image width (pixels). See :data:`FRONT_CAM_HEIGHT`."""

HEAD_CAM_HEIGHT: int = 480
"""Physical head-camera image height (pixels) for the three real cameras
``head_front`` (Orbbec RGB), ``stereo_left`` and ``stereo_right`` (IMX900s).
Matches the resize target in
``gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py`` so the
bridge can drop frames straight into ``observation.images.*`` without
the recorder having to resize."""

HEAD_CAM_WIDTH: int = 640
"""Physical head-camera image width (pixels). See :data:`HEAD_CAM_HEIGHT`."""

HEAD_CAM_KEYS: tuple[str, ...] = ("head_front", "stereo_left", "stereo_right")
"""Mount-key names used by the PC2 camera bridge AND by the
``observation.images.<key>`` features below when
``include_head_cameras=True``. Single source of truth — the recorder
trusts that the bridge publishes exactly these keys, and the LeRobot
exporter rejects any mismatch at first frame."""

FPS: int = 50
"""Dataset frame rate. Matches the SONIC tracking-policy control rate
and the upstream ``unitree_g1_sonic`` reference."""

SONIC_MOTION_TOKEN_DIM: int = 64
"""SONIC latent motion-token dimensionality (frozen 22k checkpoint)."""

HAND_DOF_OMNI: int = 10
"""Per-side OmniHand DOF count (matches agitbot-x2-record-and-replay)."""

HAND_DOF_G1_COMPAT: int = 7
"""Per-side hand DOF count when down-projected to the G1 ThreeFinger surface."""


# Canonical 31-joint name list in MuJoCo (MJCF) order. Mirrors
# ``mujoco_joint_names[]`` in
# ``gear_sonic_deploy/.../include/policy_parameters.hpp``. Used to
# label the per-scalar names of ``action.body_q_mj`` (and its
# ``_pre_sonic`` sibling) in the LeRobot v2.1 features schema. NOTE:
# this is *not* the same as ``RobotModel.joint_names`` (Pinocchio URDF
# order); the two orderings differ for the head / arm blocks.
# ``observation.state`` keeps the Pinocchio order; ``action.body_q_mj``
# keeps the MuJoCo order to match what the C++ deploy actually consumes.
MUJOCO_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_pitch_joint",     "waist_roll_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "left_wrist_yaw_joint",       "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_yaw_joint",      "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint",             "head_pitch_joint",
)
assert len(MUJOCO_JOINT_NAMES) == 31


# Body-joint groups exposed to the modality config. Order matters: the
# groups are projected into ``observation.state`` slices via the
# RobotModel's joint-group indices, and the trainer attends to *these*
# group names (not the head). Head DOFs remain present in
# ``observation.state`` for completeness but are not surfaced as a
# state group, mirroring ``unitree_g1_sonic`` (which has no head).
_BODY_JOINT_GROUPS: tuple[str, ...] = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_x2_robot_model(
    hand_variant: Literal["omnihand_10", "g1_compat_7"] = "omnihand_10",
) -> RobotModel:
    """Instantiate the X2 Ultra ``RobotModel`` for VLA dataset assembly.

    Args:
        hand_variant: ``"omnihand_10"`` (default, v0 target) or
            ``"g1_compat_7"`` for cross-embodiment evaluation.

    Returns:
        Pinocchio-backed ``RobotModel`` exposing the 31 body DOFs through
        ``joint_to_dof_index`` plus the side-loaded finger names via
        ``supplemental_info.left_finger_names`` / ``right_finger_names``.
    """
    from gear_sonic.data.robot_model.instantiation.x2_ultra import (
        instantiate_x2_ultra_robot_model,
    )

    return instantiate_x2_ultra_robot_model(hand_variant=hand_variant)


def _get_body_group_slices(robot_model: RobotModel) -> dict[str, dict[str, int]]:
    """Project each body joint group into a contiguous ``observation.state`` slice.

    Group indices are pulled directly from Pinocchio (URDF parse order). Each
    group must be contiguous in DOF index space -- this is the case for X2's
    body URDF (legs/waist/arms/head are laid out in blocks). The returned
    dict has the exact shape that ``meta/modality.json`` expects.
    """
    slices: dict[str, dict[str, int]] = {}
    for group in _BODY_JOINT_GROUPS:
        indices = sorted(robot_model.get_joint_group_indices(group))
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError(
                f"Joint group '{group}' is not contiguous in DOF index space: "
                f"{indices}. Update the URDF or split the group."
            )
        slices[group] = {
            "start": indices[0],
            "end": indices[-1] + 1,
            "original_key": "observation.state",
        }
    return slices


def assemble_observation_state(
    robot_model: RobotModel,
    body_q: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
) -> np.ndarray:
    """Concatenate body + hand joint positions into a single proprio vector.

    The layout matches the slices generated by :func:`get_modality_config_x2_vla`:
    ``[body_31][left_hand_10][right_hand_10]`` for the ``omnihand_10`` variant.

    Args:
        robot_model: X2 ``RobotModel`` instance (used for body sizing only).
        body_q: 1-D body configuration of shape ``(num_body_joints,)`` in
            Pinocchio URDF order. Equivalent to
            ``robot_model.get_configuration_from_actuated_joints(body_q,
            left_hand_actuated_joint_values=[], right_hand_actuated_joint_values=[])``
            but skips the empty-hand round-trip.
        left_hand_q: 1-D vector of left-hand joint angles (typically 10).
        right_hand_q: 1-D vector of right-hand joint angles (matches ``left_hand_q``).

    Returns:
        1-D ``float64`` array of shape ``(num_body_joints + len(left_hand_q) + len(right_hand_q),)``.
    """
    body_q = np.asarray(body_q, dtype=np.float64).reshape(-1)
    if body_q.shape[0] != robot_model.num_joints:
        raise ValueError(
            f"body_q has {body_q.shape[0]} dims; expected "
            f"robot_model.num_joints={robot_model.num_joints}"
        )
    left = np.asarray(left_hand_q, dtype=np.float64).reshape(-1)
    right = np.asarray(right_hand_q, dtype=np.float64).reshape(-1)
    return np.concatenate([body_q, left, right])


# ---------------------------------------------------------------------------
# Feature schema (meta/info.json -> "features")
# ---------------------------------------------------------------------------


def get_features_x2_vla(
    robot_model: RobotModel,
    hand_dof_per_side: int = HAND_DOF_OMNI,
    *,
    post_sonic_canonical: bool = True,
    include_front_cam: bool = False,
    include_head_cameras: bool = False,
) -> dict:
    """Return the LeRobot v2.1 ``features`` dict for the X2 SONIC dataset.

    Action surface (v1 schema, ``post_sonic_canonical=True``):

    * ``action.motion_token`` (64-D) -- the SONIC FSQ encoding of the
      *commanded* ``body_q_mj`` (operator intent), filled inline by the
      recorder via
      :class:`~gear_sonic.utils.teleop.online_sonic_tokenizer.OnlineSonicTokenizer`
      when ``--sonic-checkpoint`` is set. **This is the supervision
      target for VLA training on top of SONIC.** When the checkpoint is
      not provided (kinematic-only smoke tests) this column is all
      zeros and the recorder warns once at startup.
    * ``action.body_q_mj`` (``num_body``-D, MuJoCo joint ordering) --
      **CANONICAL training target**. The post-SONIC executed q (what
      the trained tracking policy actually achieved, i.e. what's
      visible in the MuJoCo viewer). For pure-kinematic recordings
      (``post_sonic_canonical=False``) it is just the commanded q
      since no policy is in the loop.
    * ``action.left_hand_joints`` / ``action.right_hand_joints``
      (``hand_dof_per_side``-D each) -- canonical hand action;
      post-deploy URDF-clipped q in v1, raw retargeted q in kinematic
      mode.

    When ``post_sonic_canonical=True`` (the default for
    SONIC-stabilised recordings), four debug-only sibling columns are
    added. They live on disk for retargeter / SONIC-correction
    analysis and are explicitly *not* exposed as training targets via
    :func:`get_modality_config_x2_vla`:

    * ``action.body_q_mj_pre_sonic`` -- the operator's X2 joint
      command sent on the wire to the deploy, before SONIC and
      MuJoCo physics. Same vector space as ``action.body_q_mj``.
    * ``action.left_hand_joints_pre_sonic`` /
      ``action.right_hand_joints_pre_sonic`` -- pre-deploy hand q
      (operator retarget output, before URDF clipping).
    * ``action.sonic_correction_max_rad`` -- per-frame
      ``max_arms |body_q_mj - body_q_mj_pre_sonic|`` summary scalar.

    For pure-kinematic recordings (``post_sonic_canonical=False``)
    these debug siblings are omitted because there's no SONIC in the
    loop to correct anything.

    When ``include_front_cam=True``, a second video feature
    ``observation.images.front_cam`` is added with shape
    ``(FRONT_CAM_HEIGHT, FRONT_CAM_WIDTH, 3)``. This corresponds to
    the world-fixed wide-angle witness camera defined in the robocasa
    scene XMLs (see ``front_cam`` in
    ``gear_sonic/scripts/build_x2_robocasa_scene_xml.py``
    :data:`_WORKSPACE_CAMERAS`). The recorder enables this flag
    automatically in robocasa scene mode and writes both ``ego_view``
    and ``front_cam`` per frame; non-robocasa recordings keep the
    original single-camera schema for backwards compat with existing
    parquet files.

    When ``include_head_cameras=True``, three additional video
    features are added — one per real physical head camera on the
    AgiBot X2's PC2 (Jetson Orin NX):

    * ``observation.images.head_front`` — Orbbec Gemini 335 RGB
      (native 2688×1944, downscaled to 640×480 at the bridge).
    * ``observation.images.stereo_left`` / ``observation.images.stereo_right``
      — Sony IMX900 GMSL cameras mounted on the X2's stereo head
      rig (native 2064×1552, downscaled to 640×480 at the bridge).

    Frames arrive over ZMQ from
    ``gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py`` and
    are consumed by the recorder via
    :class:`gear_sonic.camera.composed_camera.ComposedCameraClientSensor`.
    The synthetic ``ego_view`` (MuJoCo render) coexists with these
    real-sensor streams so a single episode carries both viewpoints
    and downstream training can pick which one to use as the canonical
    GR00T ``ego_view`` input.
    """
    body_joint_names = robot_model.joint_names
    num_body = robot_model.num_joints

    if hand_dof_per_side not in (HAND_DOF_OMNI, HAND_DOF_G1_COMPAT):
        raise ValueError(
            f"hand_dof_per_side must be {HAND_DOF_OMNI} or {HAND_DOF_G1_COMPAT}; "
            f"got {hand_dof_per_side}."
        )

    finger_names_left = (
        robot_model.supplemental_info.left_finger_names
        if hasattr(robot_model.supplemental_info, "left_finger_names")
        else [f"left_finger_{i}" for i in range(hand_dof_per_side)]
    )
    finger_names_right = (
        robot_model.supplemental_info.right_finger_names
        if hasattr(robot_model.supplemental_info, "right_finger_names")
        else [f"right_finger_{i}" for i in range(hand_dof_per_side)]
    )

    state_dim = num_body + 2 * hand_dof_per_side
    state_names = list(body_joint_names) + list(finger_names_left) + list(finger_names_right)

    features: dict = {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
    }
    if include_front_cam:
        features["observation.images.front_cam"] = {
            "dtype": "video",
            "shape": [FRONT_CAM_HEIGHT, FRONT_CAM_WIDTH, 3],
            "names": ["height", "width", "channel"],
        }
    if include_head_cameras:
        for cam_key in HEAD_CAM_KEYS:
            features[f"observation.images.{cam_key}"] = {
                "dtype": "video",
                "shape": [HEAD_CAM_HEIGHT, HEAD_CAM_WIDTH, 3],
                "names": ["height", "width", "channel"],
            }
    features.update({
        "observation.state": {
            "dtype": "float64",
            "shape": (state_dim,),
            "names": state_names,
        },
        "observation.projected_gravity": {
            "dtype": "float64",
            "shape": (3,),
            "names": ["gravity_x", "gravity_y", "gravity_z"],
        },
        "action.motion_token": {
            "dtype": "float64",
            "shape": (SONIC_MOTION_TOKEN_DIM,),
            "names": [f"motion_token_{i}" for i in range(SONIC_MOTION_TOKEN_DIM)],
        },
        "action.body_q_mj": {
            "dtype": "float64",
            "shape": (num_body,),
            "names": list(MUJOCO_JOINT_NAMES),
        },
        "action.left_hand_joints": {
            "dtype": "float64",
            "shape": (hand_dof_per_side,),
            "names": list(finger_names_left),
        },
        "action.right_hand_joints": {
            "dtype": "float64",
            "shape": (hand_dof_per_side,),
            "names": list(finger_names_right),
        },
    })

    if post_sonic_canonical:
        features["action.body_q_mj_pre_sonic"] = {
            "dtype": "float64",
            "shape": (num_body,),
            "names": list(MUJOCO_JOINT_NAMES),
        }
        features["action.left_hand_joints_pre_sonic"] = {
            "dtype": "float64",
            "shape": (hand_dof_per_side,),
            "names": list(finger_names_left),
        }
        features["action.right_hand_joints_pre_sonic"] = {
            "dtype": "float64",
            "shape": (hand_dof_per_side,),
            "names": list(finger_names_right),
        }
        features["action.sonic_correction_max_rad"] = {
            "dtype": "float32",
            "shape": (1,),
            "names": ["arm_delta_max_rad"],
        }

    return features


# ---------------------------------------------------------------------------
# Modality config (meta/modality.json)
# ---------------------------------------------------------------------------


def _video_modality(
    *,
    include_front_cam: bool,
    include_head_cameras: bool = False,
) -> dict:
    """Build the ``video`` block of ``meta/modality.json``.

    Single-source-of-truth helper so :func:`get_modality_config_x2_vla`
    and :func:`get_features_x2_vla` can't drift on which video keys
    are in the schema. Both should be called with the same
    ``include_front_cam`` / ``include_head_cameras`` values (the
    recorder enforces this; mismatched callers will get a LeRobot
    exporter error at first frame).
    """
    video = {
        "ego_view": {"original_key": "observation.images.ego_view"},
    }
    if include_front_cam:
        video["front_cam"] = {
            "original_key": "observation.images.front_cam",
        }
    if include_head_cameras:
        for cam_key in HEAD_CAM_KEYS:
            video[cam_key] = {
                "original_key": f"observation.images.{cam_key}",
            }
    return video


def get_modality_config_x2_vla(
    robot_model: RobotModel,
    hand_dof_per_side: int = HAND_DOF_OMNI,
    *,
    include_front_cam: bool = False,
    include_head_cameras: bool = False,
) -> dict:
    """Return the ``meta/modality.json`` content for the X2 SONIC dataset.

    Trainer-visible state groups mirror ``unitree_g1_sonic``:
    ``left_leg``, ``right_leg``, ``waist``, ``left_arm``, ``right_arm``,
    ``left_hand``, ``right_hand``, ``projected_gravity``. Head DOFs live
    in ``observation.state`` for completeness but are not exposed as a
    state group (matches G1).

    Action surface uses three keys to match ``unitree_g1_sonic``:
    ``motion_token`` (64), ``left_hand_joints`` (10 or 7), ``right_hand_joints`` (10 or 7).

    When ``include_front_cam=True`` the returned dict's ``video`` block
    additionally maps ``front_cam -> observation.images.front_cam`` so
    the trainer learns a second world-fixed witness view (see
    :func:`get_features_x2_vla`). Must match the recorder's
    ``record_front_cam`` setting; mismatches will be rejected by the
    LeRobot exporter's ``validate_frame`` at runtime.
    """
    if hand_dof_per_side not in (HAND_DOF_OMNI, HAND_DOF_G1_COMPAT):
        raise ValueError(
            f"hand_dof_per_side must be {HAND_DOF_OMNI} or {HAND_DOF_G1_COMPAT}; "
            f"got {hand_dof_per_side}."
        )

    num_body = robot_model.num_joints
    body_slices = _get_body_group_slices(robot_model)

    hand_slices = {
        "left_hand": {
            "start": num_body,
            "end": num_body + hand_dof_per_side,
            "original_key": "observation.state",
        },
        "right_hand": {
            "start": num_body + hand_dof_per_side,
            "end": num_body + 2 * hand_dof_per_side,
            "original_key": "observation.state",
        },
    }

    return {
        "state": {
            **body_slices,
            **hand_slices,
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
        },
        "action": {
            "motion_token": {
                "start": 0,
                "end": SONIC_MOTION_TOKEN_DIM,
                "original_key": "action.motion_token",
            },
            "left_hand_joints": {
                "start": 0,
                "end": hand_dof_per_side,
                "original_key": "action.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": hand_dof_per_side,
                "original_key": "action.right_hand_joints",
            },
        },
        "video": _video_modality(
            include_front_cam=include_front_cam,
            include_head_cameras=include_head_cameras,
        ),
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


__all__ = [
    "EGO_VIEW_HEIGHT",
    "EGO_VIEW_WIDTH",
    "FPS",
    "FRONT_CAM_HEIGHT",
    "FRONT_CAM_WIDTH",
    "HAND_DOF_G1_COMPAT",
    "HAND_DOF_OMNI",
    "HEAD_CAM_HEIGHT",
    "HEAD_CAM_KEYS",
    "HEAD_CAM_WIDTH",
    "MUJOCO_JOINT_NAMES",
    "SONIC_MOTION_TOKEN_DIM",
    "assemble_observation_state",
    "get_features_x2_vla",
    "get_modality_config_x2_vla",
    "get_x2_robot_model",
]
