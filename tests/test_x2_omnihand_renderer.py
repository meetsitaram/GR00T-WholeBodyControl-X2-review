"""
M3.5 acceptance gate: X2 + OmniHand-2025 renderer composer + qpos helper.

The X2 Ultra training MJCF deliberately ends each arm at ``*_wrist_roll_link``
(31 body DOFs total) -- the deploy ONNX, SONIC tracking decoder, and AimDK
ROS 2 HAL all expect that. Hand commands flow out-of-band via
``/aima/hal/joint/hand/command`` on the real robot.

For *visual* renderering (M3 inspection videos and the M5 camera plumbing
that bakes camera frames into the LeRobot dataset), we compose an augmented
MJCF in memory by attaching two articulated 10-active-DOF OmniHand-2025
chains to the X2 wrist bodies, swapping the wrist-roll *visual* mesh for a
clipped variant that drops the baked-in dummy fist, and recreating the six
URDF mimic relationships per side as MJCF ``<equality joint>`` constraints.

This test gate verifies:

1. The augmented model compiles, has the expected shape (33 X2 bodies + 32
   hand bodies, 31 X2 hinges + 32 hand hinges, 12 mimic equality
   constraints, 70 qpos slots).
2. The 10 active joints per side match
   ``OMNIHAND_FINGER_NAMES_PER_SIDE`` from
   ``x2_ultra_supplemental_info.py`` exactly -- so a 10-D vector recorded
   in M1's LeRobot dataset writes to the right qpos slot.
3. ``apply_active_hand_qpos`` projects mimic relationships exactly
   (passive = multiplier × active for all 6 rules per side).
4. The clipped wrist mesh actually replaces the original visual geom on
   each ``*_wrist_roll_link`` body (so the dummy fist no longer renders).
5. Hand collision geoms have ``contype=0, conaffinity=0`` (purely
   kinematic; the dynamics never sees them).
6. The renderer ``--with-omnihand`` flag accepts split per-side hand
   trajectory keys (``left_hand_trajectory`` / ``right_hand_trajectory``)
   from a ``record_synthetic_smoketest_dataset`` recording.
7. The training MJCF (``x2_ultra.xml``) is unaffected -- its mesh table
   still resolves to the original 38-mesh asset list, so M1, M2, and the
   trainer do not need to be touched.

Run via::

    .venv/bin/python -m pytest tests/test_x2_omnihand_renderer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Skip the entire suite if MuJoCo is not importable (CI environments without
# OpenGL deps); M3.5 is a renderer-side milestone and is not required for
# trainer-side gates to pass.
mujoco = pytest.importorskip("mujoco")

from gear_sonic.scripts.compose_x2_with_omnihand import (  # noqa: E402
    ACTIVE_FINGER_JOINTS,
    LOCKED_PASSIVE_JOINTS,
    PASSIVE_MIMIC_RULES,
    _DEFAULT_MOUNT_QUAT_WXYZ,
    _FINGER_ARMATURE,
    _FINGER_DAMPING,
    _LEFT_MOUNT_QUAT_WXYZ,
    _RIGHT_MOUNT_QUAT_WXYZ,
    apply_active_hand_ctrl,
    apply_active_hand_qpos,
    apply_active_hand_rest_pose,
    build_x2_with_omnihand_spec,
    get_active_hand_rest_pose,
)
from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (  # noqa: E402
    OMNIHAND_FINGER_NAMES_PER_SIDE,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def augmented():
    """Compose the X2 + OmniHand augmented model once for all tests."""
    spec, model, layout = build_x2_with_omnihand_spec()
    return spec, model, layout


# ────────────────────────────────────────────────────────────────────────────
# 1. Augmented model shape
# ────────────────────────────────────────────────────────────────────────────


def test_augmented_model_shape(augmented):
    """Augmented model has the expected count of bodies, joints, equality constraints.

    Per-side hand hinge count = 17:
      * 5 thumb joints (roll, abad, mcp, pip, dip)
      * 3 index joints (abad, pip, dip)
      * 3 middle joints (abad-placeholder, pip, dip) -- the abad joint is a
        free hinge that the upstream OmniHand URDF declares (probably
        copy-pasted from index/ring/pinky which DO have an active abad)
        but the middle finger is single-DOF by hardware design. We pin
        it to qpos=0 with a constant-form ``mjEQ_JOINT`` equality at
        compose time (see ``LOCKED_PASSIVE_JOINTS``); without that
        pin any equality-solver noise spins it open-loop and the
        finger visually punches through the palm collision shell.
      * 3 ring joints (abad, pip, dip)
      * 3 pinky joints (abad, pip, dip)
      = 17 per side, 34 across both hands.

    Total joints: 1 free + 31 body + 34 hand = 66.
    nq = 7 + 31 + 34 = 72.
    """
    _, model, _ = augmented

    assert model.njnt == 66, (
        f"expected 66 joints (1 free + 31 body + 34 hand), got {model.njnt}"
    )
    assert model.nq == 72, f"expected 72 qpos slots, got {model.nq}"

    # Equality constraints:
    #   * 6 mimic rules per side × 2 sides = 12 mimic
    #   * len(LOCKED_PASSIVE_JOINTS) per side × 2 sides = 2 lock pins
    # = 14 total. Each lock pin is a constant-form ``mjEQ_JOINT``
    # (``q1 = polycoef[0] = 0``), validated separately in
    # ``test_locked_passive_joints_have_zero_pin_equality``.
    expected_neq = 12 + 2 * len(LOCKED_PASSIVE_JOINTS)
    assert model.neq == expected_neq, (
        f"expected {expected_neq} equality constraints "
        f"(12 mimic + {2 * len(LOCKED_PASSIVE_JOINTS)} lock pins), got {model.neq}"
    )


# ────────────────────────────────────────────────────────────────────────────
# 2. Joint name parity vs supplemental info
# ────────────────────────────────────────────────────────────────────────────


def test_active_joint_order_matches_supplemental_info():
    """Composer's canonical 10 active joints == X2 supplemental_info contract.

    Both sources MUST agree -- otherwise our M1 LeRobot dataset's
    ``action.{left,right}_hand_joints[i]`` would write to the wrong joint
    in the augmented MJCF, and the renderer would lie about what the policy
    is commanding.
    """
    assert tuple(ACTIVE_FINGER_JOINTS) == tuple(OMNIHAND_FINGER_NAMES_PER_SIDE), (
        "ACTIVE_FINGER_JOINTS in compose_x2_with_omnihand.py drifted from "
        "OMNIHAND_FINGER_NAMES_PER_SIDE in x2_ultra_supplemental_info.py"
    )


def test_layout_resolves_all_active_and_passive_joints(augmented):
    """Every active + passive joint per side resolves to a valid qpos index."""
    _, model, layout = augmented

    for side in ("left", "right"):
        # 10 active joints, all distinct qpos addresses.
        active = layout.active_qposadr[side]
        assert len(active) == 10
        assert len(set(active)) == 10
        for qadr in active:
            assert 0 <= qadr < model.nq

        # 6 passive joints, all distinct, all in valid range.
        passive = layout.passive_qposadr[side]
        assert len(passive) == 6
        assert set(passive.keys()) == {r.passive for r in PASSIVE_MIMIC_RULES}
        for qadr in passive.values():
            assert 0 <= qadr < model.nq


# ────────────────────────────────────────────────────────────────────────────
# 3. Mimic projection numerics
# ────────────────────────────────────────────────────────────────────────────


def test_apply_active_hand_qpos_projects_mimic_exactly(augmented):
    """``apply_active_hand_qpos`` writes passive = multiplier × active."""
    _, model, layout = augmented
    data = mujoco.MjData(model)

    rng = np.random.default_rng(0)
    left_active = rng.uniform(-1.0, 1.0, size=10)
    right_active = rng.uniform(-1.0, 1.0, size=10)
    apply_active_hand_qpos(
        data, layout, left_active=left_active, right_active=right_active,
    )
    mujoco.mj_forward(model, data)

    for side, vec in (("left", left_active), ("right", right_active)):
        active_qadrs = layout.active_qposadr[side]
        passive_qadrs = layout.passive_qposadr[side]
        # Active values round-trip exactly.
        for k, qadr in enumerate(active_qadrs):
            assert data.qpos[qadr] == pytest.approx(vec[k], abs=1e-12), (
                f"{side} active joint #{k} ({ACTIVE_FINGER_JOINTS[k]}) round-trip mismatch"
            )
        # Passive values match multiplier × active.
        for rule in PASSIVE_MIMIC_RULES:
            active_idx = ACTIVE_FINGER_JOINTS.index(rule.active)
            expected = rule.multiplier * vec[active_idx]
            actual = data.qpos[passive_qadrs[rule.passive]]
            assert actual == pytest.approx(expected, abs=1e-12), (
                f"{side} passive {rule.passive} = {actual:.6f} but expected "
                f"{rule.multiplier} × {rule.active}({vec[active_idx]:.6f}) = {expected:.6f}"
            )


def test_apply_active_hand_qpos_rejects_wrong_shape(augmented):
    """Helper raises a clear error for shape-mismatched active vectors."""
    _, model, layout = augmented
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match=r"shape \(10,\)"):
        apply_active_hand_qpos(data, layout, left_active=np.zeros(7))
    with pytest.raises(ValueError, match=r"shape \(10,\)"):
        apply_active_hand_qpos(data, layout, right_active=np.zeros(11))


# ────────────────────────────────────────────────────────────────────────────
# 3b. Position actuators on active fingers (dynamics path)
#
# These guards lock in the fix that lets the augmented MJCF be stepped with
# ``mj_step`` (live VLA bridge / SONIC closed-loop sim). The earlier
# qpos-stamp approach -- still used by the offline renderer via
# ``apply_active_hand_qpos`` + ``mj_forward`` -- caused QACC NaN within
# 0.5 s of sim time when paired with ``mj_step`` because the equality
# solver couldn't reconcile the per-tick qpos discontinuities. Position
# actuators close that loop properly: the bridge writes setpoints to
# ``mj_data.ctrl`` and MuJoCo's integrator handles finger motion as
# continuous physics.
# ────────────────────────────────────────────────────────────────────────────


def test_augmented_model_has_position_actuators_on_active_fingers(augmented):
    """20 position actuators (10 per side) named ``pos_<jname>`` exist."""
    _, model, layout = augmented
    # 31 body actuators (motor_*_joint) + 20 finger actuators (pos_*_joint).
    assert model.nu == 51, (
        f"expected 51 actuators (31 body motor_* + 20 finger pos_*), got {model.nu}"
    )

    for side in ("left", "right"):
        sdk_prefix = "L_" if side == "left" else "R_"
        assert side in layout.active_actadr, (
            f"layout.active_actadr missing side={side!r}"
        )
        actadr = layout.active_actadr[side]
        assert len(actadr) == 10
        for k, aid in enumerate(actadr):
            short = ACTIVE_FINGER_JOINTS[k]
            jname = f"{side}_{sdk_prefix}{short}_joint"
            expected_name = f"pos_{jname}"
            actual_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid)
            assert actual_name == expected_name, (
                f"side={side} k={k} ({short}): actuator name {actual_name!r} != {expected_name!r}"
            )
            # Each actuator must be a position actuator: gain=FIXED, bias=AFFINE.
            assert int(model.actuator_gaintype[aid]) == int(mujoco.mjtGain.mjGAIN_FIXED), (
                f"{expected_name} gaintype != mjGAIN_FIXED"
            )
            assert int(model.actuator_biastype[aid]) == int(mujoco.mjtBias.mjBIAS_AFFINE), (
                f"{expected_name} biastype != mjBIAS_AFFINE"
            )


def test_apply_active_hand_ctrl_writes_ctrl(augmented):
    """``apply_active_hand_ctrl`` writes the 10-D vector into ``mj_data.ctrl``."""
    _, model, layout = augmented
    data = mujoco.MjData(model)

    rng = np.random.default_rng(42)
    left_active = rng.uniform(-0.1, 0.1, size=10)
    right_active = rng.uniform(-0.1, 0.1, size=10)
    apply_active_hand_ctrl(
        data, layout, left_active=left_active, right_active=right_active,
    )

    for side, vec in (("left", left_active), ("right", right_active)):
        for k, aid in enumerate(layout.active_actadr[side]):
            assert data.ctrl[aid] == pytest.approx(vec[k], abs=1e-12), (
                f"{side} actuator #{k} ({ACTIVE_FINGER_JOINTS[k]}) ctrl write round-trip mismatch"
            )


def test_augmented_mjcf_steps_dynamics_without_nan(augmented):
    """``mj_step`` on the augmented MJCF stays finite for 250 ms with zero ctrl.

    This is the regression guard for the bug that took down the live
    VLA bridge with ``--sim-with-omnihand``: stamping finger qpos every
    sim tick drove the equality solver into QACC NaN at sim t≈0.5 s,
    on a finger DOF (DOF 54), even with no operator input. With proper
    position actuators in place, the same scenario must run cleanly.
    """
    _, model, layout = augmented
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    # Lift the floating base off the floor so we don't trigger ground
    # contact NaNs unrelated to the fingers.
    if model.nq >= 7:
        data.qpos[2] = 1.5  # pelvis_z
    apply_active_hand_ctrl(
        data, layout,
        left_active=np.zeros(10), right_active=np.zeros(10),
    )

    # Step ~250 ms at 1 kHz (model.opt.timestep is typically 0.001 in
    # the bridge; whatever the spec defaults to is fine for this
    # finiteness check).
    for i in range(250):
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all(), (
            f"qpos went non-finite at step {i}; finger actuator path regressed"
        )
        assert np.isfinite(data.qvel).all(), (
            f"qvel went non-finite at step {i}; finger actuator path regressed"
        )


def test_apply_active_hand_ctrl_rejects_wrong_shape(augmented):
    """Helper raises a clear error for shape-mismatched ctrl vectors."""
    _, model, layout = augmented
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match=r"shape \(10,\)"):
        apply_active_hand_ctrl(data, layout, left_active=np.zeros(7))
    with pytest.raises(ValueError, match=r"shape \(10,\)"):
        apply_active_hand_ctrl(data, layout, right_active=np.zeros(11))


# ────────────────────────────────────────────────────────────────────────────
# Rest-pose initialisation (Patch B for the silent-stream slamming bug)
# ────────────────────────────────────────────────────────────────────────────


def test_rest_pose_constants_match_retargeter_open_endpoint():
    """``get_active_hand_rest_pose`` returns the retargeter's
    ``ratio=0`` open-hand pose verbatim, in canonical motor order."""
    from gear_sonic.utils.teleop.x2_hand_retarget import (
        HAND_GRASP_OPEN_RAD_LEFT,
        HAND_GRASP_OPEN_RAD_RIGHT,
        grasp_command_from_ratio,
    )
    left_rest = get_active_hand_rest_pose("left")
    right_rest = get_active_hand_rest_pose("right")
    assert left_rest.shape == (10,)
    assert right_rest.shape == (10,)
    np.testing.assert_allclose(left_rest, np.asarray(HAND_GRASP_OPEN_RAD_LEFT))
    np.testing.assert_allclose(right_rest, np.asarray(HAND_GRASP_OPEN_RAD_RIGHT))
    np.testing.assert_allclose(left_rest, grasp_command_from_ratio("left", 0.0))
    np.testing.assert_allclose(right_rest, grasp_command_from_ratio("right", 0.0))
    with pytest.raises(ValueError, match="must be 'left' or 'right'"):
        get_active_hand_rest_pose("middle")


def test_apply_active_hand_rest_pose_writes_qpos_and_ctrl(augmented):
    """``apply_active_hand_rest_pose`` parks both qpos and ctrl on the
    canonical open-hand rest pose, in canonical motor order, for both
    sides. Mimic equalities are propagated by the internal
    ``mj_forward`` so passive joints land on ``multiplier × active``.
    """
    _, model, layout = augmented
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    apply_active_hand_rest_pose(model, data, layout)

    left_rest = get_active_hand_rest_pose("left")
    right_rest = get_active_hand_rest_pose("right")
    for k, qadr in enumerate(layout.active_qposadr["left"]):
        assert data.qpos[qadr] == pytest.approx(left_rest[k]), (
            f"left active joint #{k} qpos != rest pose at qposadr {qadr}"
        )
    for k, qadr in enumerate(layout.active_qposadr["right"]):
        assert data.qpos[qadr] == pytest.approx(right_rest[k]), (
            f"right active joint #{k} qpos != rest pose at qposadr {qadr}"
        )
    for k, aid in enumerate(layout.active_actadr["left"]):
        assert data.ctrl[aid] == pytest.approx(left_rest[k]), (
            f"left actuator #{k} ctrl != rest pose at actuator id {aid}"
        )
    for k, aid in enumerate(layout.active_actadr["right"]):
        assert data.ctrl[aid] == pytest.approx(right_rest[k]), (
            f"right actuator #{k} ctrl != rest pose at actuator id {aid}"
        )


def test_locked_passive_joints_have_zero_pin_equality(augmented):
    """Each ``LOCKED_PASSIVE_JOINTS`` entry must have an
    ``mjEQ_JOINT`` equality with no second joint and ``polycoef[0]=0``,
    on both sides. This pins ``q1 = 0`` permanently.

    Without this, joints like ``middle_abad`` -- which exist in the
    upstream OmniHand URDF as free hinges but are NOT actuated by
    hardware design (the middle finger has only 1 active DOF) and NOT
    mimic-coupled -- accumulate runaway angular velocity from any
    numerical noise and visually rotate out of the palm collision
    mesh within a couple of seconds of sim time.
    """
    _, model, _ = augmented
    assert len(LOCKED_PASSIVE_JOINTS) >= 1, (
        "LOCKED_PASSIVE_JOINTS must include at least middle_abad; the test "
        "is paranoia against an empty tuple silently passing"
    )
    expected_lock_count = 2 * len(LOCKED_PASSIVE_JOINTS)
    lock_eqs = []
    for eid in range(model.neq):
        ename = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, eid)
        if ename and ename.endswith(tuple(f"_lock_{j}" for j in LOCKED_PASSIVE_JOINTS)):
            lock_eqs.append(eid)
    assert len(lock_eqs) == expected_lock_count, (
        f"expected {expected_lock_count} lock equalities (per side x "
        f"{len(LOCKED_PASSIVE_JOINTS)} joints), got {len(lock_eqs)}"
    )
    for eid in lock_eqs:
        assert int(model.eq_type[eid]) == int(mujoco.mjtEq.mjEQ_JOINT), (
            "lock equality must be mjEQ_JOINT type"
        )
        assert int(model.eq_obj2id[eid]) < 0, (
            "lock equality must NOT have a second joint (constant pin form: q1 = polycoef[0])"
        )
        assert float(model.eq_data[eid][0]) == pytest.approx(0.0), (
            "lock equality polycoef[0] must be 0 so q1 is pinned to 0"
        )


def test_locked_joints_stay_near_zero_through_two_seconds_of_stepping(augmented):
    """End-to-end regression for the runaway middle_abad bug. With the
    lock equality in place, ``middle_abad`` (and any other entry in
    ``LOCKED_PASSIVE_JOINTS``) must stay within ~5° of zero through
    2 s of dynamic stepping with no actuation pressure on it -- the
    soft-constraint solver's residual envelope on a properly pinned
    DOF is on the order of milliradians, not the open-loop runaway we
    saw at integration time before this fix.
    """
    _, model, layout = augmented
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nq >= 7:
        data.qpos[2] = 1.5
    apply_active_hand_rest_pose(model, data, layout)

    locked_qadrs: dict[str, int] = {}
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname is None:
            continue
        for short in LOCKED_PASSIVE_JOINTS:
            if jname.endswith(f"_{short}_joint"):
                locked_qadrs[jname] = int(model.jnt_qposadr[jid])
    assert locked_qadrs, (
        "no locked joints resolved by name; LOCKED_PASSIVE_JOINTS / "
        "URDF naming must have drifted apart"
    )

    n_steps = int(round(2.0 / float(model.opt.timestep)))
    max_abs_per_joint = {n: 0.0 for n in locked_qadrs}
    for _ in range(n_steps):
        mujoco.mj_step(model, data)
        for n, qadr in locked_qadrs.items():
            v = float(data.qpos[qadr])
            if abs(v) > max_abs_per_joint[n]:
                max_abs_per_joint[n] = abs(v)

    threshold_rad = np.deg2rad(5.0)
    for jname, max_abs in max_abs_per_joint.items():
        assert max_abs < threshold_rad, (
            f"locked joint {jname} drifted to {np.rad2deg(max_abs):.2f}° "
            f"during 2 s of dynamics (threshold 5°); the equality lock "
            f"is not effective"
        )
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()


def test_finger_joints_have_inertial_stabilization(augmented):
    """Every OmniHand finger joint (active + mimic-passive + locked) must
    have ``armature == _FINGER_ARMATURE`` and ``damping == _FINGER_DAMPING``
    after compose. This is the cheap structural check; the dynamic
    regression below verifies the chosen values actually stop the runaway.

    Body / wrist joints must NOT be touched -- those are SONIC-policy
    inputs and were trained on the upstream mass-matrix conditioning.
    """
    _, model, _ = augmented

    # Build the set of finger joint short names we expect to be patched.
    finger_short_names: set[str] = set()
    for short in ACTIVE_FINGER_JOINTS:
        finger_short_names.add(short)
    for rule in PASSIVE_MIMIC_RULES:
        finger_short_names.add(rule.passive)
    for short in LOCKED_PASSIVE_JOINTS:
        finger_short_names.add(short)

    expected_full_names: set[str] = set()
    for side, sdk in (("left", "L_"), ("right", "R_")):
        for short in finger_short_names:
            expected_full_names.add(f"{side}_{sdk}{short}_joint")

    seen_finger_joints: set[str] = set()
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname is None:
            continue
        dofadr = int(model.jnt_dofadr[jid])
        arm = float(model.dof_armature[dofadr])
        dmp = float(model.dof_damping[dofadr])
        if jname in expected_full_names:
            seen_finger_joints.add(jname)
            assert arm == pytest.approx(_FINGER_ARMATURE, abs=1e-9), (
                f"finger joint {jname} has armature={arm}; "
                f"expected {_FINGER_ARMATURE} from "
                f"_apply_finger_inertial_stabilization"
            )
            assert dmp == pytest.approx(_FINGER_DAMPING, abs=1e-9), (
                f"finger joint {jname} has damping={dmp}; "
                f"expected {_FINGER_DAMPING}"
            )
        elif "wrist" in jname or "shoulder" in jname or "elbow" in jname:
            # Body / arm joints must keep their original (zero by default
            # for the X2 MJCF) armature / damping. Allow either zero or
            # whatever the upstream MJCF specifies, but reject our
            # finger value bleeding into them.
            assert arm != _FINGER_ARMATURE or arm == 0.0, (
                f"body joint {jname} got finger-side armature={arm}; "
                f"the inertial-stabilization patch leaked outside the "
                f"OmniHand chain"
            )

    missing = expected_full_names - seen_finger_joints
    assert not missing, (
        f"expected finger joints not present in augmented model: {missing}"
    )


def test_finger_joints_stay_still_with_frozen_ctrl(augmented):
    """End-to-end regression for the constant-jitter / wide-range-wiggle
    bug. With ctrl held at the rest pose for 5 s, no finger joint may
    drift more than 5° from rest, oscillate with > 50 deg/s of frame-
    to-frame jitter, or wander more than 10° peak-to-peak.

    Pre-fix numbers (no armature, no damping) on the same model:
        worst drift   = 75.5°
        worst jitter  = 3393 deg/s
        worst max-dev = 100.5°
    Post-fix (armature=1e-3, damping=0.05):
        worst drift   = 0.05°
        worst jitter  = 3.4 deg/s
        worst max-dev = 0.91°

    Thresholds below are set ~10x the post-fix worst case, leaving
    plenty of headroom for solver / integrator changes between MuJoCo
    versions while still catching any future regression on the order
    of 1°+ of finger drift.
    """
    _, model, layout = augmented
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nq >= 7:
        data.qpos[2] = 1.5  # lift pelvis off the floor
    apply_active_hand_rest_pose(model, data, layout)
    mujoco.mj_forward(model, data)

    q_rest = data.qpos.copy()
    ctrl_rest = data.ctrl.copy()

    n_steps = int(round(5.0 / float(model.opt.timestep)))
    qpos_log = np.zeros((n_steps, model.njnt), dtype=np.float64)
    for s in range(n_steps):
        data.ctrl[:] = ctrl_rest
        mujoco.mj_step(model, data)
        qpos_log[s] = data.qpos[model.jnt_qposadr]
    dt = float(model.opt.timestep)

    # Aggregate over every OmniHand finger joint (active + mimic + locked).
    finger_short_names: set[str] = set()
    for short in ACTIVE_FINGER_JOINTS:
        finger_short_names.add(short)
    for rule in PASSIVE_MIMIC_RULES:
        finger_short_names.add(rule.passive)
    for short in LOCKED_PASSIVE_JOINTS:
        finger_short_names.add(short)

    DRIFT_DEG = 5.0
    JITTER_DPS = 50.0
    MAX_DEV_DEG = 10.0
    failures: list[str] = []
    n_finger_checked = 0
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if jname is None:
            continue
        if not any(jname.endswith(f"_{s}_joint") for s in finger_short_names):
            continue
        n_finger_checked += 1
        qa = int(model.jnt_qposadr[jid])
        rest_val = float(q_rest[qa])
        final_val = float(qpos_log[-1, jid])
        drift = float(np.rad2deg(abs(final_val - rest_val)))
        jitter = float(np.rad2deg(np.std(np.diff(qpos_log[:, jid])) / dt))
        max_dev = float(np.rad2deg(qpos_log[:, jid].ptp()))
        if drift > DRIFT_DEG:
            failures.append(
                f"  {jname}: drifted {drift:.2f}° "
                f"(threshold {DRIFT_DEG}°)"
            )
        if jitter > JITTER_DPS:
            failures.append(
                f"  {jname}: frame-to-frame jitter {jitter:.1f} deg/s "
                f"(threshold {JITTER_DPS} deg/s)"
            )
        if max_dev > MAX_DEV_DEG:
            failures.append(
                f"  {jname}: peak-to-peak deviation {max_dev:.2f}° "
                f"(threshold {MAX_DEV_DEG}°)"
            )

    assert n_finger_checked >= 30, (
        f"expected to check >= 30 finger joints (17 per side × 2), "
        f"only checked {n_finger_checked}; the joint-name pattern may "
        f"have drifted"
    )
    assert np.isfinite(qpos_log).all(), (
        "non-finite qpos during the frozen-ctrl run -- the augmented "
        "MJCF went unstable; check armature / damping / actuator gains"
    )
    if failures:
        msg = "Finger joints drifted under frozen ctrl (this is the "
        msg += "wide-range-wiggle regression):\n" + "\n".join(failures)
        pytest.fail(msg)


def test_rest_pose_init_holds_silent_stream_finite_for_one_second(augmented):
    """End-to-end Patch B regression. With qpos and ctrl both seeded
    to the rest pose, ``mj_step`` on the augmented MJCF must stay
    finite for >=1 s of sim time *without* any further ctrl writes.

    This is the test that fails on the broken (zero-init) path: ctrl=0
    drives every active joint toward qpos=0, several joints land on
    a joint-limit edge, the equality solver builds up integration
    error, and QACC blows up at a finger DOF within a few hundred ms.
    With Patch B, ctrl and qpos start in agreement at the rest pose
    so the actuators idle at zero force and the sim is stable
    indefinitely.
    """
    _, model, layout = augmented
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if model.nq >= 7:
        data.qpos[2] = 1.5  # lift pelvis off the floor

    apply_active_hand_rest_pose(model, data, layout)

    n_steps = int(round(1.0 / float(model.opt.timestep)))
    for i in range(n_steps):
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            pytest.fail(
                f"non-finite state at step {i}/{n_steps} with rest-pose init; "
                f"Patch B regressed (silent-stream slamming returned)"
            )


# ────────────────────────────────────────────────────────────────────────────
# 4. Clipped wrist-roll mesh actually swaps in
# ────────────────────────────────────────────────────────────────────────────


def test_wrist_roll_visual_swapped_to_clipped_mesh(augmented):
    """Each wrist_roll body's *visual* geom uses the clipped (no-fist) mesh.

    The collision geom must keep the original mesh so contact behaviour is
    unchanged from the un-augmented X2 model.
    """
    _, model, _ = augmented

    for side in ("left", "right"):
        bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_roll_link",
        )
        assert bid >= 0, f"{side}_wrist_roll_link missing from augmented model"

        meshes_seen: list[str] = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid:
                continue
            mid = model.geom_dataid[g]
            if mid < 0:
                continue
            mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
            meshes_seen.append(mname)

        assert f"{side}_wrist_roll_clipped_link" in meshes_seen, (
            f"clipped mesh missing on {side}_wrist_roll_link; "
            f"saw {meshes_seen}"
        )
        # Original mesh must still be present (collision geom).
        assert f"{side}_wrist_roll_link" in meshes_seen, (
            f"original wrist_roll mesh dropped on {side}; "
            f"contact-solver behaviour would diverge. saw {meshes_seen}"
        )


def test_clipped_mesh_files_are_present():
    """The vendor step ``clip_x2_wrist_for_omnihand`` was run."""
    base = (
        REPO_ROOT
        / "gear_sonic" / "data" / "assets" / "robot_description"
        / "omnihand" / "meshes"
    )
    for side in ("left", "right"):
        p = base / f"{side}_wrist_roll_clipped_link.STL"
        assert p.is_file(), (
            f"missing vendored clipped wrist mesh {p}. "
            "Re-run gear_sonic/scripts/clip_x2_wrist_for_omnihand.py."
        )
        # STL is at least a few hundred KB; if it's tiny something failed.
        assert p.stat().st_size > 100_000, (
            f"{p} is suspiciously small ({p.stat().st_size} bytes); "
            "the clip step may have produced an empty mesh."
        )


# ────────────────────────────────────────────────────────────────────────────
# 5. Hand geoms are kinematic-only (no collisions)
# ────────────────────────────────────────────────────────────────────────────


def test_hand_geoms_have_no_collision(augmented):
    """All geoms attached to the OmniHand chain have ``contype=conaffinity=0``."""
    _, model, _ = augmented

    # Find every body that's part of the hand chain (name has the side prefix
    # from MjSpec.attach AND the SDK 'L_' / 'R_' prefix).
    for b in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        is_hand = (
            bname.startswith("left_L_") or bname.startswith("right_R_")
        )
        if not is_hand:
            continue
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b:
                continue
            assert model.geom_contype[g] == 0, (
                f"hand geom on {bname} has contype={model.geom_contype[g]} "
                "(expected 0; renderer must be kinematic-only)"
            )
            assert model.geom_conaffinity[g] == 0, (
                f"hand geom on {bname} has conaffinity={model.geom_conaffinity[g]} "
                "(expected 0; renderer must be kinematic-only)"
            )


# ────────────────────────────────────────────────────────────────────────────
# 6. Renderer ``--with-omnihand`` accepts smoketest-recording layout
# ────────────────────────────────────────────────────────────────────────────


def test_renderer_accepts_split_hand_trajectory_keys(tmp_path):
    """Renderer accepts split ``left_hand_trajectory``+``right_hand_trajectory``.

    ``record_synthetic_smoketest_dataset.py`` writes per-side 10-D arrays;
    the renderer must accept this layout (the canonical form) without
    requiring callers to concatenate first.
    """
    pytest.importorskip("imageio")  # MP4 writer
    pytest.importorskip("scipy")

    # Synthesise a tiny smoketest-style recording (4 frames, deterministic).
    T = 4
    body = np.zeros((T, 31), dtype=np.float32)
    body[:, 0] = -0.312  # left_hip_pitch (any non-zero so the renderer doesn't
                         # silently zero-out qpos in the test harness)
    left_hand = np.linspace(0.0, 0.5, T * 10).reshape(T, 10).astype(np.float32)
    right_hand = np.linspace(-0.3, 0.3, T * 10).reshape(T, 10).astype(np.float32)

    rec_path = tmp_path / "fake_episode_recorded.npz"
    np.savez(
        rec_path,
        body_trajectory=body,
        left_hand_trajectory=left_hand,
        right_hand_trajectory=right_hand,
    )

    # Import here so the EGL backend is set up only when this test actually
    # runs (other tests in this file don't need offscreen rendering).
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from gear_sonic.scripts.render_smoketest_episode_video import render_episode

    out_path = tmp_path / "ego.mp4"
    summary = render_episode(
        recording_path=rec_path,
        output_path=out_path,
        camera="ego_view",
        width=128,
        height=96,
        fps=10.0,
        max_frames=T,
        with_omnihand=True,
    )
    assert summary["with_omnihand"] is True
    assert summary["num_frames"] == T
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


# ────────────────────────────────────────────────────────────────────────────
# 6b. Mount orientation is locked to the documented convention
# ────────────────────────────────────────────────────────────────────────────


def test_mount_quat_is_documented_orientation():
    """Per-side mount quaternions follow the documented convention.

    Right wrist: ``(180° about Y) ∘ (+90° about Z)`` -> wxyz
        ``(0, √½, +√½, 0)``.
    Left  wrist: ``(180° about Y) ∘ (-90° about Z)`` -> wxyz
        ``(0, √½, -√½, 0)``.

    The X2 left and right wrist_roll bodies are mirrored about the body
    centerline, so the SAME world-frame "palm down" pose corresponds to
    OPPOSITE local-Z rolls. Without the sign flip the right hand renders
    palm-down (correct) but the left hand renders palm-up.

    The convention is validated by the ``v7_ego_t*.png`` head-camera
    audit frames in the Minecraft-piano smoketest: both backs of hands
    visible from the head camera with thumbs medial.

    If you intentionally change this convention (for example to align
    with a different OmniHand SDK version) update both
    ``_LEFT_MOUNT_QUAT_WXYZ`` and ``_RIGHT_MOUNT_QUAT_WXYZ`` in
    ``compose_x2_with_omnihand.py`` and regenerate the audit frames in
    ``docs/source/tutorials/vla_training.md``.
    """
    expected_right = (0.0, 0.7071067811865476, +0.7071067811865476, 0.0)
    expected_left = (0.0, 0.7071067811865476, -0.7071067811865476, 0.0)
    assert _RIGHT_MOUNT_QUAT_WXYZ == pytest.approx(expected_right, abs=1e-12), (
        "Right OmniHand mount quaternion drifted from the documented "
        "convention. Re-render the m35_visual_audit ego-view frames "
        "before changing this."
    )
    assert _LEFT_MOUNT_QUAT_WXYZ == pytest.approx(expected_left, abs=1e-12), (
        "Left OmniHand mount quaternion drifted from the documented "
        "convention. The left side mirrors the right side's wrist long-"
        "axis roll (-90° instead of +90°). Re-render the audit frames "
        "before changing this."
    )
    # The legacy ``_DEFAULT_MOUNT_QUAT_WXYZ`` alias must stay bound to the
    # right-side value for backwards compatibility with downstream code
    # that imported the single "default" name.
    assert _DEFAULT_MOUNT_QUAT_WXYZ == _RIGHT_MOUNT_QUAT_WXYZ


# ────────────────────────────────────────────────────────────────────────────
# 7. Training MJCF unaffected
# ────────────────────────────────────────────────────────────────────────────


def test_training_mjcf_still_loads_independently():
    """The X2 training MJCF compiles with its original 31-DOF body in isolation.

    M3.5 must not have side-effected ``x2_ultra.xml`` -- the training
    pipeline, the C++ deploy reference, and the SONIC tracking decoder all
    still expect 31 DOFs / 38 meshes there.
    """
    spec = mujoco.MjSpec.from_file(
        str(REPO_ROOT / "gear_sonic" / "data" / "assets"
            / "robot_description" / "mjcf" / "x2_ultra.xml")
    )
    model = spec.compile()
    assert model.njnt == 32, f"expected 1 free + 31 body = 32 joints, got {model.njnt}"
    assert model.nq == 38, f"expected 7 + 31 = 38 qpos slots, got {model.nq}"
    assert model.neq == 0, "training MJCF must have zero equality constraints"
    # 37 X2 meshes (per the spec compile probe). M3.5 vendoring must not
    # have leaked clipped meshes back into x2_ultra.xml.
    mesh_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m)
        for m in range(model.nmesh)
    }
    assert "left_wrist_roll_clipped_link" not in mesh_names
    assert "right_wrist_roll_clipped_link" not in mesh_names


# ────────────────────────────────────────────────────────────────────────────
# 8. Trainer / deploy / SONIC side stays renderer-agnostic
#
# These guard against the most dangerous regression mode of M3.5: a future
# refactor accidentally pulling the OmniHand augmentation into the trainer
# or deploy code, expanding the body action surface from 31 DOF to 31+20
# DOF and silently breaking the SONIC checkpoint contract.
# ────────────────────────────────────────────────────────────────────────────


def test_x2_robot_model_body_is_31_dof_no_hands():
    """The X2 RobotModel exposes exactly 31 actuated body joints, zero hand DOFs.

    The trainer, ONNX export, C++ deploy harness, and SONIC tracking
    decoder all consume ``RobotModel.body_actuated_joints``. If a future
    change accidentally sneaks the OmniHand active joints into this list,
    the body-action head explodes from 31 to 31+20 = 51 and the deployed
    22k checkpoint becomes unloadable.
    """
    from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
        X2UltraSupplementalInfo,
    )

    info = X2UltraSupplementalInfo()
    assert len(info.body_actuated_joints) == 31, (
        f"X2 body must remain 31 actuated DOFs; got "
        f"{len(info.body_actuated_joints)} ({info.body_actuated_joints!r})."
    )
    assert info.left_hand_actuated_joints == [], (
        "left hand must stay out-of-band (HAL stream); never an actuated "
        "URDF joint on the trainer side."
    )
    assert info.right_hand_actuated_joints == [], (
        "right hand must stay out-of-band (HAL stream); never an actuated "
        "URDF joint on the trainer side."
    )
    # No hand joint name should appear in body_actuated_joints.
    for jn in info.body_actuated_joints:
        assert "thumb" not in jn, f"hand joint {jn} leaked into body actuated list"
        assert "index" not in jn, f"hand joint {jn} leaked into body actuated list"
        assert "middle" not in jn, f"hand joint {jn} leaked into body actuated list"
        assert "ring" not in jn, f"hand joint {jn} leaked into body actuated list"
        assert "pinky" not in jn, f"hand joint {jn} leaked into body actuated list"


def test_x2_modality_config_keeps_hand_streams_separate():
    """``x2_modality_config`` keeps body (motion_token) and hands as separate
    action keys -- the OmniHand integration does not collapse them into one
    blob. SONIC tracking decoder consumes only ``motion_token``; AimDK HAL
    consumes only ``{left,right}_hand_joints``.
    """
    pytest.importorskip("gr00t")
    from gear_sonic.data.x2_modality_config import (
        DEFAULT_STATE_GROUPS,
        make_x2_modality_config,
    )

    cfg = make_x2_modality_config(hand_dof=10)
    action_keys = list(cfg["action"].modality_keys)
    # Three separate streams; no merged "body+hand" concatenation.
    assert action_keys == ["motion_token", "left_hand_joints", "right_hand_joints"], (
        f"action key layout drifted; got {action_keys}"
    )

    # State groups stay the seven SONIC-canonical chunks + projected_gravity.
    # In particular, "left_hand" and "right_hand" must remain as DISTINCT
    # entries from "left_arm" / "right_arm" -- the trainer relies on this
    # split when slicing the parquet.
    assert "left_arm" in DEFAULT_STATE_GROUPS
    assert "right_arm" in DEFAULT_STATE_GROUPS
    assert "left_hand" in DEFAULT_STATE_GROUPS
    assert "right_hand" in DEFAULT_STATE_GROUPS


def test_omnihand_composer_is_not_imported_by_trainer_or_deploy():
    """Nothing on the trainer / deploy / ZMQ / motion-replay path may
    import the OmniHand composer.

    Allowed importers (renderer-only path):

    * ``gear_sonic/scripts/render_smoketest_episode_video.py`` (M3 inspection
      video; M3.5 augmented MJCF lives only here).
    * ``gear_sonic/scripts/clip_x2_wrist_for_omnihand.py`` (vendoring step
      that produces the clipped wrist meshes).
    * ``tests/test_x2_omnihand_renderer.py`` (this file).

    Anything else importing ``compose_x2_with_omnihand`` would leak the
    65-body / 70-qpos augmented model into the training or deploy
    pipelines and invalidate the 31-DOF body action surface.
    """
    forbidden_substrings = (
        "compose_x2_with_omnihand",
        "build_x2_with_omnihand_spec",
        "apply_active_hand_qpos",
    )
    allowed_path_endings = (
        # Renderer + vendoring scripts that legitimately use the composer.
        "gear_sonic/scripts/render_smoketest_episode_video.py",
        "gear_sonic/scripts/clip_x2_wrist_for_omnihand.py",
        "gear_sonic/scripts/compose_x2_with_omnihand.py",
        # Kinematic-viewer paths (live teleop + offline replay). These
        # only consume the augmented MJCF inside an interactive
        # ``mujoco.viewer.launch_passive`` loop, never as input to the
        # 31-DOF training / deploy surfaces. Keeping them allowlisted
        # means the augmented model never reaches a trainer or the C++
        # deploy.
        "gear_sonic/scripts/teleop_x2_kinematic.py",
        "gear_sonic/scripts/replay_x2_kinematic.py",
        "gear_sonic/utils/teleop/x2_kinematic_view.py",
        "gear_sonic/utils/embodiment/x2.py",
        # MuJoCo<->ROS bridge for the C++ deploy's kinematic-stand
        # bootstrap (offline visualization only; never feeds the
        # training pipeline).
        "gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py",
        # Tests.
        "tests/test_x2_omnihand_renderer.py",
    )
    # Scan only Python source files inside the repo (skip vendor SDK,
    # build artifacts, .venv, .git, ...).
    code_roots = [
        REPO_ROOT / "gear_sonic",
        REPO_ROOT / "gear_sonic_deploy",
        REPO_ROOT / "decoupled_wbc",
        REPO_ROOT / "tests",
    ]
    bad: list[str] = []
    for root in code_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            # Skip vendored upstream; we don't control imports there.
            if "agibot-x2-references" in path.parts:
                continue
            rel = str(path.relative_to(REPO_ROOT))
            if any(rel.endswith(allowed) for allowed in allowed_path_endings):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            for sym in forbidden_substrings:
                if sym in text:
                    bad.append(f"{rel} :: contains forbidden symbol {sym!r}")
    assert not bad, (
        "OmniHand composer leaked outside the renderer-only allowlist:\n  "
        + "\n  ".join(bad)
        + "\n\nIf you intentionally added a new caller, extend "
        "``allowed_path_endings`` in this test, but FIRST consider whether "
        "the new caller really needs the augmented MJCF -- M3.5's whole "
        "point is that the 31-DOF training MJCF is the canonical pipeline."
    )


def test_x2_zmq_pose_protocol_is_31_dof():
    """The mock VLA publisher sends 31-DOF body joint commands, not 31+20.

    This locks the wire-format invariant: the C++ deploy ``ZmqPoseInputSource``
    consumes a 31-element ``joint_pos_mj`` payload. If a future change
    expanded that to 51 (body + hands), every existing deploy harness and
    the SONIC tracking decoder would silently mis-slice.
    """
    from gear_sonic.scripts.mock_vla_publish_stand_token import (
        DEFAULT_STAND_POSE_MUJOCO_RAD,
        NUM_BODY_DOFS,
    )

    assert NUM_BODY_DOFS == 31
    assert len(DEFAULT_STAND_POSE_MUJOCO_RAD) == 31
