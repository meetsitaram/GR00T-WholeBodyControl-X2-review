"""Tests for :mod:`gear_sonic.utils.teleop.x2_kinematic_view`.

These tests build a real X2 MuJoCo model (no OmniHand variant for
speed; the augmented variant is exercised by the existing
``test_x2_omnihand_renderer`` suite) and verify the lifted helpers
write the floating base + body slots correctly.

No viewer launch -- :func:`mujoco.viewer.launch_passive` is never
called.
"""

from __future__ import annotations

import numpy as np
import pytest


def _import_mujoco_or_skip():
    try:
        import mujoco  # noqa: F401
    except Exception as exc:
        pytest.skip(f"mujoco not importable: {exc}")
    return mujoco


def test_build_kinematic_model_no_omnihand_returns_valid_triple() -> None:
    mujoco = _import_mujoco_or_skip()
    from gear_sonic.utils.teleop.x2_kinematic_view import build_kinematic_model

    model, layout, body_qposadr = build_kinematic_model(with_omnihand=False)
    assert layout is None
    assert isinstance(body_qposadr, np.ndarray)
    assert body_qposadr.dtype == np.int64
    # X2 has 31 body joints in MuJoCo joint order; the address table
    # must be one entry per body joint.
    assert body_qposadr.shape == (31,)
    # All addresses are within bounds and fall after the 7-slot
    # floating-base block (qpos[0:3] = pos, qpos[3:7] = quat).
    assert int(model.nq) > 7
    assert (body_qposadr >= 7).all()
    assert (body_qposadr < model.nq).all()


def test_set_kinematic_pose_writes_floating_base_and_body() -> None:
    mujoco = _import_mujoco_or_skip()
    from gear_sonic.utils.teleop.x2_kinematic_view import (
        DEFAULT_PELVIS_POS_XYZ,
        DEFAULT_PELVIS_QUAT_WXYZ,
        build_kinematic_model,
        set_kinematic_pose,
    )

    model, layout, body_qposadr = build_kinematic_model(with_omnihand=False)
    data = mujoco.MjData(model)
    body_q = np.linspace(-0.1, 0.1, num=body_qposadr.shape[0], dtype=np.float64)
    set_kinematic_pose(
        mujoco_mod=mujoco,
        model=model,
        data=data,
        body_q_mj=body_q,
        body_qposadr=body_qposadr,
        layout=layout,
        apply_hand_fn=None,
        left_hand_q=np.zeros(10),
        right_hand_q=np.zeros(10),
    )
    np.testing.assert_allclose(data.qpos[0:3], DEFAULT_PELVIS_POS_XYZ)
    np.testing.assert_allclose(data.qpos[3:7], DEFAULT_PELVIS_QUAT_WXYZ)
    np.testing.assert_allclose(data.qpos[body_qposadr], body_q)
    # mj_forward should have been called -- xpos for the pelvis body
    # should now reflect DEFAULT_PELVIS_POS_XYZ rather than zeros.
    np.testing.assert_allclose(data.qvel, 0.0)


def test_set_kinematic_pose_calls_apply_hand_fn_when_layout_present() -> None:
    """``apply_hand_fn`` is invoked iff both ``layout`` and the fn are non-None."""
    mujoco = _import_mujoco_or_skip()
    from gear_sonic.utils.teleop.x2_kinematic_view import (
        build_kinematic_model,
        set_kinematic_pose,
    )

    model, layout, body_qposadr = build_kinematic_model(with_omnihand=False)
    data = mujoco.MjData(model)
    calls: list[tuple] = []

    def _record_call(d, lay, *, left_active, right_active):  # noqa: ARG001
        calls.append((left_active.copy(), right_active.copy()))

    # layout is None -> apply_hand_fn should NOT be called.
    set_kinematic_pose(
        mujoco_mod=mujoco,
        model=model,
        data=data,
        body_q_mj=np.zeros(body_qposadr.shape[0]),
        body_qposadr=body_qposadr,
        layout=layout,
        apply_hand_fn=_record_call,
        left_hand_q=np.full(10, 0.5),
        right_hand_q=np.full(10, 0.7),
    )
    assert calls == []  # layout was None

    # Provide a non-None layout sentinel; now the fn must run.
    set_kinematic_pose(
        mujoco_mod=mujoco,
        model=model,
        data=data,
        body_q_mj=np.zeros(body_qposadr.shape[0]),
        body_qposadr=body_qposadr,
        layout="sentinel",
        apply_hand_fn=_record_call,
        left_hand_q=np.full(10, 0.5),
        right_hand_q=np.full(10, 0.7),
    )
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0][0], 0.5)
    np.testing.assert_allclose(calls[0][1], 0.7)


def test_set_kinematic_pose_accepts_custom_pelvis_pose() -> None:
    """Non-default pelvis pose is written when explicitly provided."""
    mujoco = _import_mujoco_or_skip()
    from gear_sonic.utils.teleop.x2_kinematic_view import (
        build_kinematic_model,
        set_kinematic_pose,
    )

    model, layout, body_qposadr = build_kinematic_model(with_omnihand=False)
    data = mujoco.MjData(model)
    custom_pos = (0.1, 0.2, 0.7)
    custom_quat = (1.0, 0.0, 0.0, 0.0)
    set_kinematic_pose(
        mujoco_mod=mujoco,
        model=model,
        data=data,
        body_q_mj=np.zeros(body_qposadr.shape[0]),
        body_qposadr=body_qposadr,
        layout=layout,
        apply_hand_fn=None,
        left_hand_q=np.zeros(10),
        right_hand_q=np.zeros(10),
        pelvis_pos_xyz=custom_pos,
        pelvis_quat_wxyz=custom_quat,
    )
    np.testing.assert_allclose(data.qpos[0:3], custom_pos)
    np.testing.assert_allclose(data.qpos[3:7], custom_quat)


def test_x2_embodiment_config_uses_lifted_helpers() -> None:
    """The X2 embodiment config builds the same model the helpers do."""
    _import_mujoco_or_skip()
    from gear_sonic.utils.embodiment import get_embodiment

    cfg = get_embodiment("x2")
    model, layout, body_qposadr = cfg.build_kinematic_model(with_omnihand=False)
    assert layout is None
    assert body_qposadr.shape == (cfg.num_body_dofs,)
