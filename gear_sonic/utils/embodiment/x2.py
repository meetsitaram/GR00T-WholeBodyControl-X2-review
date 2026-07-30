"""AgiBot X2 Ultra :class:`EmbodimentConfig`.

Wraps the existing X2 helpers (MJCF builder, OmniHand qpos applier,
stand-pose constants) into the embodiment-registry surface so the
kinematic-replay CLI can dispatch off ``--robot``. Everything here is a
thin adapter over modules that have already been the X2 source of
truth for a long time:

* MJCF + camera attachment + OmniHand augmentation:
  :func:`gear_sonic.scripts.render_smoketest_episode_video.build_model_with_camera`.
* OmniHand qpos applier:
  :func:`gear_sonic.scripts.compose_x2_with_omnihand.apply_active_hand_qpos`.
* Default stand pose:
  :data:`gear_sonic.scripts.vla.live_vla_publish_motion_token.DEFAULT_STAND_POSE_MUJOCO_RAD`.
* Pelvis pose: matches the gantry_hang firmware-stand entry in
  ``gear_sonic_deploy/config/sim_init_poses.yaml`` (also reused in
  :file:`gear_sonic/scripts/teleop_x2_kinematic.py`).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from gear_sonic.utils.embodiment.config import EmbodimentConfig
from gear_sonic.utils.embodiment.registry import register_embodiment


__all__ = ["build_x2_embodiment_config"]


# Pelvis pose for kinematic playback. Matches the on-feet stand entry
# the C++ deploy uses at boot. Keep these in lockstep with
# gear_sonic/scripts/teleop_x2_kinematic.py if either side moves.
_X2_PELVIS_POS_XYZ: tuple[float, float, float] = (0.0, 0.0, 0.665)
_X2_PELVIS_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def _x2_build_kinematic_model(*, with_omnihand: bool) -> tuple[Any, Any, np.ndarray]:
    """Lazy-import the X2 MJCF builder and return ``(model, layout, body_qposadr)``."""
    from gear_sonic.scripts.render_smoketest_episode_video import (
        build_model_with_camera,
        resolve_camera_spec,
    )

    cam = resolve_camera_spec("ego_view")
    return build_model_with_camera(cam, with_omnihand=with_omnihand)


def _x2_apply_omnihand_fn(
    data: Any,
    layout: Any,
    *,
    left_active: np.ndarray,
    right_active: np.ndarray,
) -> None:
    """Lazy-import wrapper around ``apply_active_hand_qpos`` for X2's OmniHand."""
    from gear_sonic.scripts.compose_x2_with_omnihand import apply_active_hand_qpos

    apply_active_hand_qpos(
        data,
        layout,
        left_active=left_active,
        right_active=right_active,
    )


def build_x2_embodiment_config() -> EmbodimentConfig:
    """Construct the X2 :class:`EmbodimentConfig`.

    Pulls the canonical 31-D stand pose from
    :mod:`gear_sonic.scripts.vla.live_vla_publish_motion_token` (the same
    constant the live VLA bridge and the teleop scripts already use)
    so there is exactly one stand pose definition for X2 in the repo.
    """
    from gear_sonic.scripts.vla.live_vla_publish_motion_token import (
        DEFAULT_HAND_DOF,
        DEFAULT_STAND_POSE_MUJOCO_RAD,
        NUM_BODY_DOFS,
    )

    return EmbodimentConfig(
        name="x2",
        num_body_dofs=int(NUM_BODY_DOFS),
        num_hand_dof_per_side=int(DEFAULT_HAND_DOF),
        pelvis_pos_xyz=_X2_PELVIS_POS_XYZ,
        pelvis_quat_wxyz=_X2_PELVIS_QUAT_WXYZ,
        default_stand_pose_mj=np.array(
            DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64
        ),
        build_kinematic_model=_x2_build_kinematic_model,
        apply_omnihand_fn=_x2_apply_omnihand_fn,
    )


register_embodiment(build_x2_embodiment_config())
