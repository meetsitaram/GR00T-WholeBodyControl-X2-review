"""Tests for :mod:`gear_sonic.utils.embodiment`.

These tests are pure-Python: no MuJoCo model load, no parquet read.
They verify that the registry contract holds:

* ``get_embodiment("x2")`` returns a usable config with the canonical
  X2 dimensions.
* ``get_embodiment("g1")`` returns a stub config whose factories raise
  :class:`NotImplementedError` (so callers fail at the point they try
  to *use* G1, not at registry lookup time).
* ``get_embodiment("unknown")`` raises :class:`KeyError`.
* ``register_embodiment`` adds new entries and overwrites existing ones.
* ``EmbodimentConfig`` validates pose / stand-pose shapes at
  construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.embodiment import (
    EmbodimentConfig,
    get_embodiment,
    register_embodiment,
    registered_embodiments,
)


def test_x2_embodiment_registered_with_canonical_dims() -> None:
    cfg = get_embodiment("x2")
    assert cfg.name == "x2"
    assert cfg.num_body_dofs == 31
    assert cfg.num_hand_dof_per_side == 10
    assert cfg.default_stand_pose_mj.shape == (31,)
    assert callable(cfg.build_kinematic_model)
    assert callable(cfg.apply_omnihand_fn)
    # Pelvis pose should be the on-feet stand pose (z ~ 0.665 m).
    assert cfg.pelvis_pos_xyz[2] == pytest.approx(0.665, abs=1e-3)
    assert cfg.pelvis_quat_wxyz == (1.0, 0.0, 0.0, 0.0)


def test_g1_embodiment_registered_but_factories_raise_not_implemented() -> None:
    cfg = get_embodiment("g1")
    assert cfg.name == "g1"
    # Stub still satisfies the dataclass shape invariants.
    assert cfg.default_stand_pose_mj.shape == (cfg.num_body_dofs,)
    with pytest.raises(NotImplementedError, match="G1"):
        cfg.build_kinematic_model(with_omnihand=False)
    with pytest.raises(NotImplementedError, match="G1"):
        cfg.apply_omnihand_fn(
            data=None,
            layout=None,
            left_active=np.zeros(cfg.num_hand_dof_per_side),
            right_active=np.zeros(cfg.num_hand_dof_per_side),
        )


def test_get_embodiment_unknown_name_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="Unknown embodiment"):
        get_embodiment("does-not-exist")


def test_registered_embodiments_includes_x2_and_g1() -> None:
    names = registered_embodiments()
    assert "x2" in names
    assert "g1" in names


def test_register_embodiment_adds_and_overwrites() -> None:
    """Registering a fresh name adds it; re-registering overwrites it."""

    def _fake_build(*, with_omnihand: bool):  # noqa: ARG001 -- match signature
        return None, None, np.array([0])

    cfg_a = EmbodimentConfig(
        name="__test_robot",
        num_body_dofs=2,
        num_hand_dof_per_side=0,
        pelvis_pos_xyz=(0.0, 0.0, 0.5),
        pelvis_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        default_stand_pose_mj=np.zeros(2),
        build_kinematic_model=_fake_build,
        apply_omnihand_fn=None,
    )
    register_embodiment(cfg_a)
    assert get_embodiment("__test_robot") is cfg_a

    cfg_b = EmbodimentConfig(
        name="__test_robot",
        num_body_dofs=3,
        num_hand_dof_per_side=0,
        pelvis_pos_xyz=(0.0, 0.0, 0.6),
        pelvis_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        default_stand_pose_mj=np.zeros(3),
        build_kinematic_model=_fake_build,
        apply_omnihand_fn=None,
    )
    register_embodiment(cfg_b)
    looked_up = get_embodiment("__test_robot")
    assert looked_up is cfg_b
    assert looked_up.num_body_dofs == 3


def test_register_embodiment_rejects_non_config() -> None:
    with pytest.raises(TypeError, match="EmbodimentConfig"):
        register_embodiment(object())  # type: ignore[arg-type]


def test_embodiment_config_rejects_wrong_pose_shape() -> None:
    def _fake_build(*, with_omnihand: bool):  # noqa: ARG001
        return None, None, np.array([0])

    with pytest.raises(ValueError, match="default_stand_pose_mj"):
        EmbodimentConfig(
            name="__bad_pose",
            num_body_dofs=4,
            num_hand_dof_per_side=0,
            pelvis_pos_xyz=(0.0, 0.0, 0.0),
            pelvis_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            default_stand_pose_mj=np.zeros(3),  # wrong width
            build_kinematic_model=_fake_build,
            apply_omnihand_fn=None,
        )


def test_embodiment_config_rejects_bad_pelvis_pos_len() -> None:
    def _fake_build(*, with_omnihand: bool):  # noqa: ARG001
        return None, None, np.array([0])

    with pytest.raises(ValueError, match="pelvis_pos_xyz"):
        EmbodimentConfig(
            name="__bad_pelvis",
            num_body_dofs=2,
            num_hand_dof_per_side=0,
            pelvis_pos_xyz=(0.0, 0.0),  # type: ignore[arg-type]
            pelvis_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            default_stand_pose_mj=np.zeros(2),
            build_kinematic_model=_fake_build,
            apply_omnihand_fn=None,
        )


def test_embodiment_config_rejects_bad_pelvis_quat_len() -> None:
    def _fake_build(*, with_omnihand: bool):  # noqa: ARG001
        return None, None, np.array([0])

    with pytest.raises(ValueError, match="pelvis_quat_wxyz"):
        EmbodimentConfig(
            name="__bad_quat",
            num_body_dofs=2,
            num_hand_dof_per_side=0,
            pelvis_pos_xyz=(0.0, 0.0, 0.0),
            pelvis_quat_wxyz=(1.0, 0.0, 0.0),  # type: ignore[arg-type]
            default_stand_pose_mj=np.zeros(2),
            build_kinematic_model=_fake_build,
            apply_omnihand_fn=None,
        )
