"""Shared kinematic-viewer helpers for the X2 in MuJoCo.

These helpers are used both by the live kinematic teleop entry point
(:file:`gear_sonic/scripts/teleop_x2_kinematic.py`) and by the offline
kinematic replay CLI
(:file:`gear_sonic/scripts/replay_x2_kinematic.py`). Keeping them in
one module avoids the two paths drifting apart on the on-feet stand
pose, the OmniHand qpos write, or the floating-base placement.

Everything in this module is *purely kinematic*:

* No physics step (``mj_forward`` only).
* No ZMQ, no SONIC, no policy.
* The floating base is pinned at the on-feet stand pose; only body /
  hand DOFs are mutated by the per-frame writer.

The functions are robot-agnostic in shape (they take ``model``,
``data``, ``body_qposadr`` and a hand applier as arguments) but the
default constants exported here are X2-specific. Other embodiments
should pass their own constants via
:class:`gear_sonic.utils.embodiment.EmbodimentConfig`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


__all__ = [
    "DEFAULT_PELVIS_POS_XYZ",
    "DEFAULT_PELVIS_QUAT_WXYZ",
    "build_kinematic_model",
    "set_kinematic_pose",
]


# Pinned floating-base pose. Matches the ``gantry_hang`` firmware-stand
# entry in ``gear_sonic_deploy/config/sim_init_poses.yaml`` -- robot on
# its feet, pelvis ~0.665 m above the floor, identity orientation.
DEFAULT_PELVIS_POS_XYZ: tuple[float, float, float] = (0.0, 0.0, 0.665)
DEFAULT_PELVIS_QUAT_WXYZ: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def build_kinematic_model(*, with_omnihand: bool) -> tuple[Any, Any, np.ndarray]:
    """Build the X2 (+ optional OmniHand) MuJoCo model purely kinematically.

    Thin wrapper around
    :func:`gear_sonic.scripts.render_smoketest_episode_video.build_model_with_camera`
    pre-bound to the canonical ``ego_view`` head camera. Returns the
    same triple the smoketest renderer does:

    * ``model``: the compiled :class:`mujoco.MjModel`.
    * ``layout``: a ``compose_x2_with_omnihand.HandQposLayout`` when
      ``with_omnihand`` is True, otherwise ``None``.
    * ``body_qposadr``: a length-31 ``np.ndarray[int64]`` mapping each
      slot of the canonical body trajectory to its ``qposadr`` in the
      compiled model (the OmniHand augmented model fragments the
      contiguous body block, so per-name addressing is mandatory).
    """
    from gear_sonic.scripts.render_smoketest_episode_video import (
        build_model_with_camera,
        resolve_camera_spec,
    )

    cam = resolve_camera_spec("ego_view")
    return build_model_with_camera(cam, with_omnihand=with_omnihand)


def set_kinematic_pose(
    *,
    mujoco_mod: Any,
    model: Any,
    data: Any,
    body_q_mj: np.ndarray,
    body_qposadr: np.ndarray,
    layout: Any,
    apply_hand_fn: Optional[Any],
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    pelvis_pos_xyz: tuple[float, float, float] = DEFAULT_PELVIS_POS_XYZ,
    pelvis_quat_wxyz: tuple[float, float, float, float] = DEFAULT_PELVIS_QUAT_WXYZ,
) -> None:
    """Write the floating base, body and hand DOFs into ``data.qpos``.

    Args:
        mujoco_mod: the imported ``mujoco`` module (passed in to keep
            this helper import-free at module scope).
        model: compiled :class:`mujoco.MjModel`.
        data: companion :class:`mujoco.MjData`.
        body_q_mj: length-31 (or whatever ``body_qposadr`` indexes)
            array of body joint positions in MuJoCo joint order.
        body_qposadr: per-name ``qposadr`` table from
            :func:`build_kinematic_model`.
        layout: OmniHand qpos layout (``None`` when ``with_omnihand`` was
            False at build time).
        apply_hand_fn: callable matching
            ``fn(data, layout, *, left_active, right_active)``. Skipped
            when either ``apply_hand_fn`` or ``layout`` is ``None``.
        left_hand_q / right_hand_q: per-side hand DOFs (10 for OmniHand).
        pelvis_pos_xyz / pelvis_quat_wxyz: floating-base pose. Defaults
            match :data:`DEFAULT_PELVIS_POS_XYZ` /
            :data:`DEFAULT_PELVIS_QUAT_WXYZ` (X2 on-feet stand).
    """
    data.qpos[0:3] = pelvis_pos_xyz
    data.qpos[3:7] = pelvis_quat_wxyz
    data.qpos[body_qposadr] = body_q_mj.astype(np.float64, copy=False)
    if apply_hand_fn is not None and layout is not None:
        apply_hand_fn(
            data,
            layout,
            left_active=left_hand_q.astype(np.float64, copy=False),
            right_active=right_hand_q.astype(np.float64, copy=False),
        )
    data.qvel[:] = 0.0
    mujoco_mod.mj_forward(model, data)
