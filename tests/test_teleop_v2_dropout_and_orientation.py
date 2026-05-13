"""Regression tests for the v2 teleop dropout, null-space, and wrist-quat fixes.

Covers:

* P0a — controller-dropout frames (raw pos at origin or quat identity)
  hold the last good IK target instead of being applied raw, which was
  the proximate cause of the "elbow flips backward" failure mode in
  ``data/lerobot/x2_quest3_kinematic_v2/debug/teleop_episode_000000.npz``.
* P0b — null-space elbow-down bias makes the IK solver bias toward the
  preferred (arms-down) posture in the redundant DOF instead of picking
  arbitrary branches that minimise ``||dq||``.
* P1  — head-yaw correction + per-arm wrist alignment lets us track the
  operator's wrist orientation through body rotation.
* P2  — XRHand 5-finger curl maps to the 10-DOF OmniHand command in a
  monotone way: increasing curl on a single finger only moves that
  finger's motors.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

from gear_sonic.utils.teleop.operator_calibration import (
    CALIBRATION_POSE_IDS,
    PoseMeasurement,
    fit_calibration,
    head_yaw_from_quat,
    robot_reference_wrist_positions,
    robot_reference_wrist_quats,
    wrist_quat_to_head_yaw_frame,
)
from gear_sonic.utils.teleop.solver.arm.x2_arm_ik import ArmIKSolver
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import (
    VRArmTeleopCalibrated,
    _is_controller_dropout,
    _is_twin_dropout,
)
from gear_sonic.utils.teleop.x2_hand_retarget import (
    DEFAULT_CURL_DEADZONE,
    DEFAULT_CURL_FULL_THRESHOLD,
    DEFAULT_CURL_GAMMA,
    DEFAULT_OPPOSE_DEADZONE,
    DEFAULT_OPPOSE_FULL_THRESHOLD,
    DEFAULT_OPPOSE_GAMMA,
    HAND_FINGER_NAMES_PER_SIDE,
    HAND_GRASP_CLOSED_LEFT_DEG,
    HAND_GRASP_CLOSED_RIGHT_DEG,
    HAND_GRASP_CLOSED_RAD_LEFT,
    HAND_GRASP_CLOSED_RAD_RIGHT,
    HAND_GRASP_OPEN_RAD_LEFT,
    HAND_GRASP_OPEN_RAD_RIGHT,
    NUM_HAND_DOF_PER_SIDE,
    grasp_command_from_ratio,
    normalize_finger_curls,
    normalize_thumb_oppose,
    per_finger_grasp_command_from_curls,
    per_finger_grasp_command_from_curls_and_oppose,
    stretch_finger_curls,
    stretch_thumb_oppose,
)


# ── helpers (lifted from the existing v2 smoke suite) ────────────────────


def _make_identity_calibration():
    """Calibration where operator wrist == robot wrist for all 3 poses."""
    ref = robot_reference_wrist_positions()
    qref = robot_reference_wrist_quats()
    ms = {}
    for pose in CALIBRATION_POSE_IDS:
        ms[pose] = PoseMeasurement(
            pose_id=pose,
            left_wrist_mean=ref[pose]["left"].copy(),
            right_wrist_mean=ref[pose]["right"].copy(),
            sample_count=50,
            left_wrist_vel_rms_mps=0.01,
            right_wrist_vel_rms_mps=0.01,
            # Arms-down operator quat == robot quat -> alignment = identity.
            left_wrist_quat_head_yaw=qref[pose]["left"].copy(),
            right_wrist_quat_head_yaw=qref[pose]["right"].copy(),
        )
    return fit_calibration(ms, operator_id="synthetic")


def _vr_pose(
    *,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    head_pos: np.ndarray,
    head_yaw_rad: float = 0.0,
    left_wrist_quat: np.ndarray | None = None,
    right_wrist_quat: np.ndarray | None = None,
) -> np.ndarray:
    # Default wrist quats: a small non-identity rotation so they DON'T
    # trigger the controller-dropout heuristic (which fires on exact
    # identity). The exact value doesn't matter for the position-only
    # tests below; we just need to be off-identity.
    if left_wrist_quat is None:
        left_wrist_quat = sRot.from_euler("xyz", [0.1, 0.2, 0.3]).as_quat(
            scalar_first=True
        )
    if right_wrist_quat is None:
        right_wrist_quat = sRot.from_euler("xyz", [-0.1, 0.2, 0.3]).as_quat(
            scalar_first=True
        )
    qhead = sRot.from_euler("z", head_yaw_rad).as_quat(scalar_first=True)
    out = np.zeros((3, 7), dtype=np.float64)
    out[0, :3] = left_wrist
    out[0, 3:] = left_wrist_quat
    out[1, :3] = right_wrist
    out[1, 3:] = right_wrist_quat
    out[2, :3] = head_pos
    out[2, 3:] = qhead
    return out


# ── P0a: controller dropout detection + last-good hold ────────────────────


def test_dropout_pos_at_origin_is_detected() -> None:
    quat_ok = np.array([0.5, 0.5, 0.5, 0.5])
    assert _is_controller_dropout(np.zeros(3), quat_ok)


def test_dropout_quat_identity_is_detected() -> None:
    pos_ok = np.array([0.5, 0.0, 1.0])
    assert _is_controller_dropout(pos_ok, np.array([1.0, 0.0, 0.0, 0.0]))


def test_clean_sample_is_not_a_dropout() -> None:
    pos_ok = np.array([0.5, 0.0, 1.0])
    quat_ok = np.array([0.5, 0.5, 0.5, 0.5])
    assert not _is_controller_dropout(pos_ok, quat_ok)


def test_twin_dropout_detected_when_both_wrists_collapse() -> None:
    p = np.array([0.5, 0.0, 1.0])
    assert _is_twin_dropout(p, p + 0.01)
    assert not _is_twin_dropout(p, p + np.array([0.0, 0.5, 0.0]))


def test_dropout_holds_last_good_target_for_left_wrist() -> None:
    """Once we've seen one clean tick, a subsequent left-wrist dropout
    must reuse the previous left target (not collapse to origin).
    """
    cal = _make_identity_calibration()
    ref = robot_reference_wrist_positions()
    teleop = VRArmTeleopCalibrated(calibration=cal, rotation_weight=0.0)
    teleop.set_engaged(True)

    head = np.array([0.0, 0.0, 1.7])
    good_pose = _vr_pose(
        left_wrist=head + ref["t_pose"]["left"],
        right_wrist=head + ref["t_pose"]["right"],
        head_pos=head,
    )
    res0 = teleop.step(good_pose)
    assert not res0.left_dropout
    last_left_target = res0.left_target_pos.copy()

    # Synthesise a dropout for the LEFT wrist only: pos = origin + small
    # noise, quat = identity. Right side stays clean.
    drop_pose = good_pose.copy()
    drop_pose[0, :3] = np.array([0.001, 0.0, 0.0])
    drop_pose[0, 3:] = np.array([1.0, 0.0, 0.0, 0.0])
    res1 = teleop.step(drop_pose)
    assert res1.left_dropout
    assert res1.left_target_held
    np.testing.assert_allclose(res1.left_target_pos, last_left_target, atol=1e-9)
    # Right side should still track normally (not held).
    assert not res1.right_dropout
    assert not res1.right_target_held


def test_dropout_holds_q_not_just_target_no_nullspace_drift() -> None:
    """The IK target being held during a dropout is not enough on its
    own: if the solver still runs each tick, the null-space bias term
    keeps nudging q toward ``q_preferred`` even when the position
    target is already satisfied. Over a 200 ms dropout (10 ticks at
    50 Hz) the wrist roll DOF then drifts by ~50 deg, and the first
    "real" tick after the dropout shows a recovery jolt.

    Verify that the solver does NOT iterate during a dropout: ``q``
    must be byte-equal to its pre-dropout value across many dropout
    ticks.
    """
    cal = _make_identity_calibration()
    ref = robot_reference_wrist_positions()
    teleop = VRArmTeleopCalibrated(calibration=cal, rotation_weight=0.0)
    teleop.set_engaged(True)

    head = np.array([0.0, 0.0, 1.7])
    good_pose = _vr_pose(
        left_wrist=head + ref["t_pose"]["left"],
        right_wrist=head + ref["t_pose"]["right"],
        head_pos=head,
    )

    # Run a few clean ticks so the IK converges on a stable q.
    for _ in range(10):
        teleop.step(good_pose)
    q_before = teleop._left_q.copy()
    rq_before = teleop._right_q.copy()

    # Now feed 10 consecutive dropout frames (BOTH wrists corrupted).
    drop_pose = good_pose.copy()
    drop_pose[0, :3] = np.array([0.001, 0.0, 0.0])
    drop_pose[0, 3:] = np.array([1.0, 0.0, 0.0, 0.0])
    drop_pose[1, :3] = np.array([-0.001, 0.0, 0.0])
    drop_pose[1, 3:] = np.array([1.0, 0.0, 0.0, 0.0])
    for tick_i in range(10):
        res = teleop.step(drop_pose)
        assert res.left_dropout and res.right_dropout, f"tick {tick_i}"
        # q must be IDENTICAL to the pre-dropout value -- not just
        # close. If the solver is iterating at all the null-space
        # bias term will move q a few mrad per tick.
        np.testing.assert_array_equal(
            teleop._left_q, q_before,
            err_msg=f"left q drifted on dropout tick {tick_i}",
        )
        np.testing.assert_array_equal(
            teleop._right_q, rq_before,
            err_msg=f"right q drifted on dropout tick {tick_i}",
        )


def test_dropout_first_tick_falls_back_to_current_fk() -> None:
    """If the very first tick is a dropout we have no last-good target;
    falling back to the FK of the current q keeps the IK from blowing
    up at the origin.
    """
    cal = _make_identity_calibration()
    teleop = VRArmTeleopCalibrated(calibration=cal, rotation_weight=0.0)
    teleop.set_engaged(True)

    head = np.array([0.0, 0.0, 1.7])
    drop_pose = _vr_pose(
        left_wrist=np.array([0.0, 0.0, 0.0]),
        right_wrist=np.array([0.0, 0.0, 0.0]),
        head_pos=head,
    )
    res = teleop.step(drop_pose)
    assert res.left_dropout and res.right_dropout
    # Without a last-good target, the fallback is the FK of the
    # current q (the neutral pose). Verify the target is non-trivial
    # (not the origin) -- that's the failure mode we're guarding
    # against.
    assert float(np.linalg.norm(res.left_target_pos)) > 0.2
    assert float(np.linalg.norm(res.right_target_pos)) > 0.2


# ── P0b: null-space elbow-down bias ───────────────────────────────────────


def test_null_space_bias_pulls_toward_preferred_in_redundant_dof() -> None:
    """With a tiny null-space gain, repeated solves at the same target
    must drift the joint angles toward q_preferred along the redundant
    DOF without changing the end-effector position.
    """
    q_pref = np.array([0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0])
    solver_with = ArmIKSolver(
        side="left",
        rotation_weight=0.0,
        null_space_gain=0.30,
        q_preferred=q_pref,
        max_per_tick_step_rad=0.30,
    )
    solver_without = ArmIKSolver(
        side="left",
        rotation_weight=0.0,
        null_space_gain=0.0,
        max_per_tick_step_rad=0.30,
    )

    target_pos, _ = solver_with.fk(q_pref)

    # Start far from q_pref but at a q whose FK is exactly target_pos.
    # We construct that by perturbing only along the elbow-swivel
    # direction: shoulder_yaw moves, elbow / shoulder_pitch fixed -- a
    # crude but functional null-space sample.
    q_start = q_pref.copy()
    q_start[2] += 0.5

    q_with, _ = solver_with.solve(q_start, target_pos, max_iters=30)
    q_without, _ = solver_without.solve(q_start, target_pos, max_iters=30)

    # End-effector still on target for both solvers.
    pos_with, _ = solver_with.fk(q_with)
    pos_without, _ = solver_without.fk(q_without)
    np.testing.assert_allclose(pos_with, target_pos, atol=5e-3)
    np.testing.assert_allclose(pos_without, target_pos, atol=5e-3)

    # ...but the WITH-bias solver got closer to q_preferred.
    err_with = float(np.linalg.norm(q_with - q_pref))
    err_without = float(np.linalg.norm(q_without - q_pref))
    assert err_with < err_without, (
        f"null-space bias should pull toward q_pref; "
        f"err_with={err_with:.3f}, err_without={err_without:.3f}"
    )


def test_null_space_zero_gain_matches_legacy_behaviour() -> None:
    """With ``null_space_gain=0`` the solver must behave bit-identically
    to the legacy implementation (no surprise drift)."""
    q0 = np.array([0.0, 0.4, 0.0, -0.3, 0.0, 0.0, 0.0])
    solver = ArmIKSolver(side="left", rotation_weight=0.0, null_space_gain=0.0)
    target, _ = solver.fk(q0)
    q_next, info = solver.solve(q0, target, max_iters=1)
    np.testing.assert_allclose(q_next, q0, atol=1e-6)
    assert info.pos_err_m < 1e-3


# ── P1: head-yaw + wrist-orientation alignment ────────────────────────────


def test_wrist_quat_to_head_yaw_frame_round_trip() -> None:
    """Identity head + identity wrist must round-trip to identity, and a
    pure yaw rotation of the head must factor out of the wrist quat."""
    qident = np.array([1.0, 0.0, 0.0, 0.0])
    out = wrist_quat_to_head_yaw_frame(qident, qident)
    np.testing.assert_allclose(out, qident, atol=1e-9)

    # Operator turned 30 deg in yaw; wrist follows the head -> in the
    # head-yaw frame the wrist still reads as identity.
    yaw = np.deg2rad(30.0)
    qhead = sRot.from_euler("z", yaw).as_quat(scalar_first=True)
    qwrist_world = qhead.copy()  # wrist tracks head
    out = wrist_quat_to_head_yaw_frame(qwrist_world, qhead)
    np.testing.assert_allclose(out, qident, atol=1e-9)


def test_wrist_alignment_quat_recovers_robot_arms_down() -> None:
    """Calibration alignment must satisfy
    ``q_align * q_op_arms_down = q_robot_arms_down``.

    This is the contract that lets the IK solver receive *robot-frame*
    target orientations from operator-frame wrist quats.
    """
    cal = _make_identity_calibration()
    qref = robot_reference_wrist_quats()
    for side in ("left", "right"):
        target = cal.apply_to_wrist_quat(
            cal.measurements["arms_down"].left_wrist_quat_head_yaw
            if side == "left"
            else cal.measurements["arms_down"].right_wrist_quat_head_yaw,
            side,
        )
        # Quaternions are double-cover; check up to sign.
        ref = qref["arms_down"][side]
        diff_pos = float(np.linalg.norm(target - ref))
        diff_neg = float(np.linalg.norm(target + ref))
        assert min(diff_pos, diff_neg) < 1e-6, (
            f"{side}: alignment-applied quat {target} does not match "
            f"robot reference {ref}"
        )


def test_legacy_calibration_without_wrist_quat_disables_orientation_ik(tmp_path) -> None:
    """A calibration loaded from a v0 YAML carries identity alignment
    quats (default for missing fields). The teleop must auto-detect
    that and force ``rotation_weight=0`` to avoid driving wrists
    randomly.

    We simulate this by writing a v0-style YAML with the ``fit`` block
    missing ``wrist_alignment_quat_wxyz`` and loading it back through
    :meth:`OperatorCalibration.load_yaml`.
    """
    import yaml as _yaml

    from gear_sonic.utils.teleop.operator_calibration import (
        OperatorCalibration,
        ROBOT_REFERENCE_Q_RAD,
        SCHEMA_VERSION,
    )

    ref = robot_reference_wrist_positions()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "operator_id": "legacy-v0",
        "created_utc": "2025-01-01T00:00:00+00:00",
        "units": "meters",
        "notes": "",
        "poses": {
            pose_id: {
                "left_wrist": ref[pose_id]["left"].tolist(),
                "right_wrist": ref[pose_id]["right"].tolist(),
                "samples": 50,
                "left_vel_rms_mps": 0.01,
                "right_vel_rms_mps": 0.01,
                # Note: NO wrist quat fields -- legacy YAML.
            }
            for pose_id in CALIBRATION_POSE_IDS
        },
        "robot_reference_q_rad": {
            pose_id: {
                side: [float(v) for v in q]
                for side, q in by_side.items()
            }
            for pose_id, by_side in ROBOT_REFERENCE_Q_RAD.items()
        },
        "fit": {
            "left":  {"scale": [1, 1, 1], "translation": [0, 0, 0], "residual_m": 0.0},
            "right": {"scale": [1, 1, 1], "translation": [0, 0, 0], "residual_m": 0.0},
            # Note: NO wrist_alignment_quat_wxyz -- legacy YAML.
        },
    }
    yaml_path = tmp_path / "legacy_v0.yaml"
    yaml_path.write_text(_yaml.safe_dump(payload, sort_keys=False))

    cal = OperatorCalibration.load_yaml(yaml_path)
    np.testing.assert_allclose(cal.fit["left"].wrist_alignment_quat,
                               [1.0, 0.0, 0.0, 0.0])

    teleop = VRArmTeleopCalibrated(calibration=cal, rotation_weight=0.5)
    # Auto-detect should force rotation_weight=0 since alignment is identity.
    assert teleop._rotation_weight == 0.0


# ── P2: per-finger curl -> OmniHand command ──────────────────────────────


def _curl_array(*, thumb=0.0, index=0.0, middle=0.0, ring=0.0, pinky=0.0) -> np.ndarray:
    return np.array([thumb, index, middle, ring, pinky], dtype=np.float64)


def test_per_finger_zero_curl_matches_uniform_open_anchor() -> None:
    cmd = per_finger_grasp_command_from_curls("left", _curl_array())
    np.testing.assert_allclose(cmd, grasp_command_from_ratio("left", 0.0), atol=1e-12)


def test_per_finger_full_curl_matches_uniform_closed_anchor() -> None:
    cmd = per_finger_grasp_command_from_curls(
        "right", _curl_array(thumb=1, index=1, middle=1, ring=1, pinky=1)
    )
    np.testing.assert_allclose(cmd, grasp_command_from_ratio("right", 1.0), atol=1e-12)


def test_per_finger_index_only_moves_index_motors() -> None:
    """Curling only the index finger should move ``index_abad`` and
    ``index_pip`` and leave the other 8 motors at their open value.
    """
    open_cmd = per_finger_grasp_command_from_curls("left", _curl_array())
    index_cmd = per_finger_grasp_command_from_curls("left", _curl_array(index=1.0))
    moved = np.abs(index_cmd - open_cmd) > 1e-6
    moved_names = {HAND_FINGER_NAMES_PER_SIDE[i] for i, m in enumerate(moved) if m}
    assert moved_names == {"index_abad", "index_pip"}


def test_per_finger_middle_only_moves_middle_pip() -> None:
    open_cmd = per_finger_grasp_command_from_curls("right", _curl_array())
    mid_cmd = per_finger_grasp_command_from_curls("right", _curl_array(middle=1.0))
    moved = np.abs(mid_cmd - open_cmd) > 1e-6
    moved_names = {HAND_FINGER_NAMES_PER_SIDE[i] for i, m in enumerate(moved) if m}
    assert moved_names == {"middle_pip"}


def test_per_finger_curl_is_monotone_per_finger() -> None:
    """For every finger and every motor that finger drives, the motor
    output is monotone (non-decreasing OR non-increasing) in the
    finger's curl."""
    finger_to_motors = {
        "thumb": (0, 1, 2),
        "index": (3, 4),
        "middle": (5,),
        "ring": (6, 7),
        "pinky": (8, 9),
    }
    samples = np.linspace(0.0, 1.0, 5)
    for side in ("left", "right"):
        for finger_idx, (finger_name, motor_idxs) in enumerate(
            finger_to_motors.items()
        ):
            for m in motor_idxs:
                trace = []
                for c in samples:
                    curls = np.zeros(5, dtype=np.float64)
                    curls[finger_idx] = c
                    trace.append(per_finger_grasp_command_from_curls(side, curls)[m])
                arr = np.asarray(trace)
                non_decreasing = bool(np.all(np.diff(arr) >= -1e-9))
                non_increasing = bool(np.all(np.diff(arr) <= 1e-9))
                assert non_decreasing or non_increasing, (
                    f"{side}/{finger_name}/"
                    f"{HAND_FINGER_NAMES_PER_SIDE[m]} not monotone: {arr}"
                )


def test_per_finger_command_shape_and_clamp() -> None:
    cmd = per_finger_grasp_command_from_curls("left", _curl_array(index=10.0, ring=-3.0))
    assert cmd.shape == (NUM_HAND_DOF_PER_SIDE,)
    # No NaNs / Infs and all within left-arm limits.
    from gear_sonic.utils.teleop.x2_hand_retarget import HAND_JOINT_LIMITS_LEFT_RAD
    for i, (lo, hi) in enumerate(HAND_JOINT_LIMITS_LEFT_RAD):
        assert lo - 1e-9 <= cmd[i] <= hi + 1e-9


def test_per_finger_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        per_finger_grasp_command_from_curls("left", np.zeros(4))


def test_per_finger_invalid_side_raises() -> None:
    with pytest.raises(ValueError):
        per_finger_grasp_command_from_curls("middle", np.zeros(5))


# ── P3: opposition-aware retargeting ─────────────────────────────────────


def test_oppose_none_falls_back_to_curl_only_path() -> None:
    """``thumb_oppose=None`` must reproduce the curl-only path bit-for-bit."""
    for side in ("left", "right"):
        for finger_kw in (
            dict(),
            dict(thumb=0.6, index=0.3),
            dict(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
        ):
            curls = _curl_array(**finger_kw)
            np.testing.assert_allclose(
                per_finger_grasp_command_from_curls_and_oppose(side, curls, None),
                per_finger_grasp_command_from_curls(side, curls),
                atol=1e-12,
                err_msg=f"side={side} curls={finger_kw}",
            )


def test_oppose_drives_all_three_thumb_motors_independent_of_curl() -> None:
    """Holding the thumb-flex curl at 0 but sweeping opposition from 0->1
    must move all three thumb motors (roll, abad, mcp) to their
    CLOSED anchor.

    Rationale: the canonical "thumb pad on fingertip" gesture has
    high opposition but low ``thumb_flex`` (Quest 3 reports
    ``thumb_flex`` mostly when the IP joint folds, not when the
    thumb just swings across the palm). For the X2 thumb to
    actually reach the fingertips visually, the MCP knuckle must
    bend during opposition too -- otherwise the thumb sweeps across
    the palm but stays straight and ends up alongside the fingers
    rather than touching them. See
    ``data/lerobot/x2_quest3_kinematic_v4/debug/teleop_episode_000000.npz``
    for the screenshot/analysis (May 10).
    """
    for side in ("left", "right"):
        zero_curl = _curl_array()
        cmd_open = per_finger_grasp_command_from_curls_and_oppose(side, zero_curl, 0.0)
        cmd_oppose = per_finger_grasp_command_from_curls_and_oppose(side, zero_curl, 1.0)
        diff = cmd_oppose - cmd_open
        moved = np.abs(diff) > 1e-6
        moved_names = {HAND_FINGER_NAMES_PER_SIDE[i] for i, m in enumerate(moved) if m}
        # All three thumb motors must move on a pure opposition
        # gesture: roll + abad swing the thumb across the palm and
        # mcp bends the knuckle so the tip actually reaches the
        # finger pads.
        assert moved_names == {"thumb_roll", "thumb_abad", "thumb_mcp"}, (
            f"{side}: unexpected motors moved by oppose-only signal: {moved_names}"
        )


def test_oppose_full_with_thumb_curl_full_matches_uniform_closed_anchor() -> None:
    """When BOTH the thumb curl and the opposition signal are 1, the
    full thumb (roll/abad/mcp) reaches its CLOSED anchor -- same as
    a uniform thumb=1 from the curl-only path.
    """
    curls = _curl_array(thumb=1.0)
    cmd = per_finger_grasp_command_from_curls_and_oppose("right", curls, 1.0)
    legacy = per_finger_grasp_command_from_curls("right", curls)
    # Same on the 3 thumb motors, identical on the others (where
    # opposition isn't used at all).
    np.testing.assert_allclose(cmd, legacy, atol=1e-12)


def test_high_thumb_flex_drives_opposition_motors_to_closed_even_with_zero_oppose() -> None:
    """A closed-fist gesture in hand-tracking mode (thumb_flex=1)
    must drive ``thumb_roll`` / ``thumb_abad`` all the way to CLOSED
    even when the lateral opposition signal is low or zero. The
    closed-fist gesture wraps the thumb over the curled fingers so
    the thumb-tip lands near palm centre, giving only a moderate
    opposition signal -- but anatomically the thumb's CMC abduction
    motors swing through their full travel during a fist. This is
    parity with controller mode (where one uniform trigger drives
    all three thumb motors) and matches the empirical
    "controller mode = thumb closes nicely; hand mode =
    thumb_abad barely moves" complaint from
    teleop_episode_000000.npz in v3.
    """
    curls = _curl_array(thumb=1.0)
    cmd = per_finger_grasp_command_from_curls_and_oppose("left", curls, 0.0)
    cmd_full_closed = per_finger_grasp_command_from_curls(
        "left", _curl_array(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0)
    )
    # thumb_roll (idx 0) and thumb_abad (idx 1) should be at the
    # CLOSED anchor since max(oppose=0, thumb_flex=1) = 1.
    np.testing.assert_allclose(cmd[0], cmd_full_closed[0], atol=1e-12)
    np.testing.assert_allclose(cmd[1], cmd_full_closed[1], atol=1e-12)
    # thumb_mcp (idx 2) is at CLOSED anchor since thumb-flex curl=1.
    np.testing.assert_allclose(cmd[2], cmd_full_closed[2], atol=1e-12)


def test_oppose_high_with_low_thumb_curl_drives_all_thumb_motors_via_oppose() -> None:
    """A thumb-finger touch with the fingers extended (thumb_flex low,
    oppose high) must drive **all three** thumb motors (roll, abad,
    mcp) on the opposition signal. The MCP knuckle bend is
    essential for the thumb pad to actually reach the finger pads
    -- without it the thumb sweeps across the palm but stays
    straight and lands alongside the fingers rather than touching
    them.

    Rationale: ``per_finger_grasp_command_from_curls_and_oppose``
    drives all three thumb motors from
    ``max(thumb_flex_curl, oppose)``. With low ``thumb_flex`` and
    high ``oppose``, ``max`` returns ``oppose``, so all three
    thumb motors lerp on ``oppose``.
    """
    curls = _curl_array(thumb=0.1)
    # Disable hand-prior curl compensation so the exact lerp ratio
    # is assertable (otherwise stretch_finger_curls maps thumb=0.1
    # -> 0.0 since 0.1 sits in the deadzone).
    cmd = per_finger_grasp_command_from_curls_and_oppose(
        "left", curls, 1.0, apply_curl_compensation=False
    )
    full_closed = per_finger_grasp_command_from_curls(
        "left",
        _curl_array(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
        apply_curl_compensation=False,
    )
    # All three thumb motors ride the opposition signal (= 1) -> CLOSED.
    np.testing.assert_allclose(cmd[0], full_closed[0], atol=1e-12)
    np.testing.assert_allclose(cmd[1], full_closed[1], atol=1e-12)
    np.testing.assert_allclose(cmd[2], full_closed[2], atol=1e-12)


def test_zero_oppose_with_thumb_flex_drives_thumb_mcp_via_flex_curl() -> None:
    """Symmetric coverage: when ``oppose == 0`` and ``thumb_flex`` is
    moderate, the combined ``max(oppose, thumb_flex)`` signal
    reduces to the raw ``thumb_flex`` curl, so all three thumb
    motors lerp on that single value.

    This protects the back-compat path -- closed-fist-with-no-
    opposition (e.g. when the operator points the thumb sideways
    rather than across the palm) must still partially close the
    thumb on the flex signal alone.
    """
    curls = _curl_array(thumb=0.1)
    cmd = per_finger_grasp_command_from_curls_and_oppose(
        "left", curls, 0.0, apply_curl_compensation=False
    )
    full_closed = per_finger_grasp_command_from_curls(
        "left",
        _curl_array(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
        apply_curl_compensation=False,
    )
    full_open = per_finger_grasp_command_from_curls(
        "left", _curl_array(), apply_curl_compensation=False
    )
    # All three thumb motors lerp 90 % toward OPEN on the flex curl
    # (= 0.1) since max(0.0, 0.1) = 0.1.
    for idx in (0, 1, 2):
        expected = 0.9 * full_open[idx] + 0.1 * full_closed[idx]
        np.testing.assert_allclose(cmd[idx], expected, atol=1e-12)


def test_finger_tip_oppose_drives_pinky_motors_independent_of_curl() -> None:
    """A deliberate thumb-to-pinky-tip touch with the rest of the
    hand otherwise relaxed must drive ``pinky_pip`` (and the matching
    ``pinky_abad``) all the way to the CLOSED anchor, even when the
    raw pinky curl is well below 1.

    Rationale (the v0.5 fix for the "open: non-thumb fingertip-to-
    thumb touch" issue): Quest 3's "fingers move together" prior
    caps isolated single-finger curls at ~0.30-0.40 raw, so even
    after per-operator p05/p95 normalisation the receiving finger
    only travels ~36 % to CLOSED on a deliberate thumb-finger
    touch. The WebXR ``computeFingerTipOppose`` saturates at
    literal contact (d_norm < 0.06 ~ 0.5 cm) and the Python
    retargeter folds it in via ``max(curls[i], finger_tip_oppose[i])``
    so the touched fingertip closes regardless of curl
    under-reporting. None of the OTHER finger motors should move
    (their tip_oppose is zero on this gesture).
    """
    for side in ("left", "right"):
        # Low-but-not-zero pinky curl (matches the v4 NPZ
        # touch-frame distribution where raw pinky_curl ~= 0.37 on
        # an actual operator-pinky-touches-operator-thumb frame).
        curls = _curl_array(thumb=0.5, pinky=0.37)
        thumb_oppose = 0.95
        finger_tip_oppose = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        cmd_with_tip = per_finger_grasp_command_from_curls_and_oppose(
            side, curls, thumb_oppose,
            finger_tip_oppose=finger_tip_oppose,
        )
        cmd_without = per_finger_grasp_command_from_curls_and_oppose(
            side, curls, thumb_oppose,
            finger_tip_oppose=None,
        )
        full_closed = per_finger_grasp_command_from_curls(
            side, _curl_array(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
        )

        # pinky_abad (idx 8) and pinky_pip (idx 9): ride
        # max(0.37, 1.0) = 1.0 -> CLOSED anchor.
        np.testing.assert_allclose(cmd_with_tip[8], full_closed[8], atol=1e-12)
        np.testing.assert_allclose(cmd_with_tip[9], full_closed[9], atol=1e-12)

        # All non-pinky / non-thumb motors must match the no-tip-oppose
        # baseline EXACTLY (their tip_oppose is zero in this gesture).
        for motor_idx in (3, 4, 5, 6, 7):  # index_abad..ring_pip
            np.testing.assert_allclose(
                cmd_with_tip[motor_idx],
                cmd_without[motor_idx],
                atol=1e-12,
                err_msg=f"{side}: motor {motor_idx} unexpectedly moved by pinky-only tip_oppose",
            )

        # The pinky output should also be strictly more closed than
        # the no-tip-oppose baseline (since 1.0 > 0.37).
        assert abs(cmd_with_tip[9] - full_closed[9]) < abs(cmd_without[9] - full_closed[9]), (
            f"{side}: pinky_pip with tip_oppose=1 should be closer to CLOSED than without."
        )


def test_finger_tip_oppose_zero_matches_curl_only_baseline() -> None:
    """When all four ``finger_tip_oppose`` entries are zero, the
    motor command must equal the legacy curl-only path EXACTLY for
    the four non-thumb fingers. This is the "smooth proportional
    control on non-touch frames" guarantee.
    """
    for side in ("left", "right"):
        curls = _curl_array(thumb=0.4, index=0.6, middle=0.5, ring=0.45, pinky=0.55)
        thumb_oppose = 0.3
        finger_tip_oppose_zero = np.zeros(4, dtype=np.float64)

        cmd_zero = per_finger_grasp_command_from_curls_and_oppose(
            side, curls, thumb_oppose,
            finger_tip_oppose=finger_tip_oppose_zero,
        )
        cmd_none = per_finger_grasp_command_from_curls_and_oppose(
            side, curls, thumb_oppose,
            finger_tip_oppose=None,
        )
        np.testing.assert_allclose(cmd_zero, cmd_none, atol=1e-12)


def test_finger_tip_oppose_nan_falls_back_to_curl_per_finger() -> None:
    """Per-finger NaN entries in ``finger_tip_oppose`` must fall back
    to the curl signal for that finger only, leaving the other
    fingers driven by their finite tip_oppose values. This handles
    the "some XRHand fingertip joints dropped out this frame"
    case where the WebXR client emits a partial 4-vector.
    """
    side = "right"
    curls = _curl_array(thumb=0.5, index=0.3, middle=0.6, ring=0.3, pinky=0.3)
    thumb_oppose = 0.5
    finger_tip_oppose = np.array(
        [np.nan, 0.9, np.nan, 0.0],
        dtype=np.float64,
    )

    cmd = per_finger_grasp_command_from_curls_and_oppose(
        side, curls, thumb_oppose,
        finger_tip_oppose=finger_tip_oppose,
    )
    cmd_baseline = per_finger_grasp_command_from_curls_and_oppose(
        side, curls, thumb_oppose,
        finger_tip_oppose=None,
    )
    # Index (NaN tip) and ring (NaN tip) must match baseline EXACTLY.
    np.testing.assert_allclose(cmd[3], cmd_baseline[3], atol=1e-12)  # index_abad
    np.testing.assert_allclose(cmd[4], cmd_baseline[4], atol=1e-12)  # index_pip
    np.testing.assert_allclose(cmd[6], cmd_baseline[6], atol=1e-12)  # ring_abad
    np.testing.assert_allclose(cmd[7], cmd_baseline[7], atol=1e-12)  # ring_pip
    # Pinky (tip=0.0 < pinky_curl=0.3) must match baseline EXACTLY (max
    # returns the curl).
    np.testing.assert_allclose(cmd[8], cmd_baseline[8], atol=1e-12)
    np.testing.assert_allclose(cmd[9], cmd_baseline[9], atol=1e-12)
    # Middle (tip=0.9 > middle_curl=0.6) must close FURTHER than baseline.
    full_closed = per_finger_grasp_command_from_curls(
        side, _curl_array(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
    )
    # cmd[5] should lerp at 0.9 toward the new (88°) CLOSED anchor.
    assert abs(cmd[5] - full_closed[5]) < abs(cmd_baseline[5] - full_closed[5])


def test_finger_tip_oppose_invalid_shape_raises() -> None:
    """``finger_tip_oppose`` must be a (4,) array; (3,) or (5,) must
    raise -- the function asserts the shape rather than silently
    truncating because a wrong shape almost always means a wiring
    bug upstream.
    """
    side = "right"
    curls = _curl_array(index=0.3)
    thumb_oppose = 0.5
    for bad in (
        np.zeros(3, dtype=np.float64),
        np.zeros(5, dtype=np.float64),
        np.zeros((4, 1), dtype=np.float64),
    ):
        try:
            per_finger_grasp_command_from_curls_and_oppose(
                side, curls, thumb_oppose, finger_tip_oppose=bad,
            )
        except ValueError as e:
            assert "finger_tip_oppose" in str(e)
        else:
            raise AssertionError(
                f"finger_tip_oppose with shape {bad.shape} should have raised ValueError"
            )


def test_non_thumb_pip_closed_anchors_at_88_degrees() -> None:
    """The non-thumb pip CLOSED anchors must be 88° (~98 % of the
    0..90° hardware range), not the historical 80°. Complements
    the JS-side ``computeFingerTipOppose`` plumbing: even when
    the ``*_pip`` motor lerps all the way to its CLOSED anchor on
    a deliberate thumb-finger touch, the OmniHand fingertip arc is
    still narrower than the operator's because the human hand has
    3 cascaded knuckles per finger vs OmniHand's 1 active PIP +
    1 mimic-coupled DIP. Pushing the anchor to hardware max is
    the kinematic ceiling for this design.
    """
    expected_pip_indices = (4, 5, 7, 9)  # index_pip, middle_pip, ring_pip, pinky_pip
    for closed_deg in (HAND_GRASP_CLOSED_LEFT_DEG, HAND_GRASP_CLOSED_RIGHT_DEG):
        for idx in expected_pip_indices:
            assert abs(abs(closed_deg[idx]) - 88.0) < 1e-9, (
                f"non-thumb pip CLOSED anchor at idx={idx} is {closed_deg[idx]} (expected ±88°)"
            )


def test_stretch_finger_curls_deadzone_and_saturation() -> None:
    """``stretch_finger_curls`` is the Quest-3 hand-prior coupling
    compensation. Verify the boundary behaviour with explicit scalar
    parameters (per-finger defaults are exercised by the
    ``test_stretch_finger_curls_per_finger_*`` tests below).
    """
    deadzone = DEFAULT_CURL_DEADZONE
    full_thresh = DEFAULT_CURL_FULL_THRESHOLD
    gamma = DEFAULT_CURL_GAMMA
    span = full_thresh - deadzone

    # Below deadzone -> 0 across all 5 finger slots.
    cz = stretch_finger_curls(
        np.full(5, deadzone - 1e-6),
        deadzone=deadzone, full_threshold=full_thresh, gamma=gamma,
    )
    np.testing.assert_allclose(cz, np.zeros(5), atol=1e-12)

    # Exactly at deadzone -> 0 (boundary inclusive on the low side).
    cz_eq = stretch_finger_curls(
        np.full(5, deadzone),
        deadzone=deadzone, full_threshold=full_thresh, gamma=gamma,
    )
    np.testing.assert_allclose(cz_eq, np.zeros(5), atol=1e-12)

    # At full_threshold -> 1.
    cs_full = stretch_finger_curls(
        np.full(5, full_thresh),
        deadzone=deadzone, full_threshold=full_thresh, gamma=gamma,
    )
    np.testing.assert_allclose(cs_full, np.ones(5), atol=1e-12)

    # Above full_threshold (e.g. group-fist value 0.92) -> still 1.
    cs_over = stretch_finger_curls(
        np.full(5, 0.92),
        deadzone=deadzone, full_threshold=full_thresh, gamma=gamma,
    )
    np.testing.assert_allclose(cs_over, np.ones(5), atol=1e-12)

    # Power-curve interior.
    midpoint = deadzone + 0.5 * span
    cs_mid = stretch_finger_curls(
        np.full(5, midpoint),
        deadzone=deadzone, full_threshold=full_thresh, gamma=gamma,
    )
    np.testing.assert_allclose(cs_mid, np.full(5, 0.5 ** gamma), atol=1e-12)


def test_stretch_finger_curls_low_sensitivity_in_active_range() -> None:
    """The per-finger defaults must suppress all "open hand + slight
    movement" raw curls completely and saturate any "intentional
    curl" raw value to the full motor-closed command. Tested with
    the live :data:`DEFAULT_CURL_*_PER_FINGER` arrays.

    Empirical distribution of raw curls on the recorded teleop
    data (gear_sonic/scripts/tune_finger_curl_compensation.py,
    pooled across 4 v3 episodes, 7626 hand-mode frames):

      finger    p10    p50    p90       per-finger deadzone
      thumb     0.26   0.43   0.89      0.25
      index     0.11   0.26   0.84      0.35
      middle    0.11   0.28   0.87      0.35
      ring      0.09   0.28   0.85      0.35
      pinky     0.12   0.25   0.84      0.35

    A "casual finger movement" that the user explicitly does NOT
    want to register on the robot sits at raw <= 0.20 for ALL
    fingers (including thumb). An "intentional curl" raw value
    always reaches >= 0.45 across all fingers.
    """
    # raw <= 0.20 -> exactly 0 across all five fingers under
    # per-finger defaults (thumb dz=0.25, fingers dz=0.35).
    for raw_val in (0.05, 0.10, 0.15, 0.20):
        out = stretch_finger_curls(np.full(5, raw_val))
        np.testing.assert_array_equal(
            out, np.zeros(5),
            err_msg=f"raw={raw_val} must produce 0 closure; got {out}",
        )

    # raw >= 0.45 -> exactly 1 across all five fingers under
    # per-finger defaults (thumb full=0.27, fingers full=0.40).
    for raw_val in (0.45, 0.50, 0.92, 1.00):
        out = stretch_finger_curls(np.full(5, raw_val))
        np.testing.assert_allclose(
            out, np.ones(5), atol=1e-12,
            err_msg=f"raw={raw_val} must saturate to 1; got {out}",
        )

    # Mid-deadzone values: thumb saturates at raw=0.30 (above its
    # full=0.27) but the other four fingers stay at 0 (below their
    # dz=0.35).
    out = stretch_finger_curls(np.full(5, 0.30))
    assert out[0] == 1.0, f"thumb at raw=0.30 must saturate; got {out[0]}"
    np.testing.assert_array_equal(
        out[1:], np.zeros(4),
        err_msg=f"non-thumb fingers at raw=0.30 must stay 0; got {out[1:]}",
    )


def test_stretch_finger_curls_isolated_quest3_max_saturates() -> None:
    """Empirical: when the operator curls a single non-thumb finger
    in isolation, Quest 3's reported per-finger curl tops out at
    ~0.50 in the wild teleop data (pooled p75 across the four
    fingers is 0.70-0.76). The post-stretch curl for that finger
    must therefore reach 1.0 by raw=0.50 -- otherwise the X2's
    matching omnihand motors won't fully close on isolated-finger
    gestures.
    """
    raw = np.array([0.10, 0.50, 0.10, 0.10, 0.10])  # only index curled
    stretched = stretch_finger_curls(raw)
    assert stretched[1] == 1.0, f"index at raw=0.50 must saturate; got {stretched[1]}"
    np.testing.assert_allclose(stretched[[0, 2, 3, 4]], np.zeros(4), atol=1e-12)


def test_stretch_finger_curls_propagates_to_motor_command() -> None:
    """End-to-end: with ``apply_curl_compensation=True`` an "isolated
    index curl" raw signal (above the per-finger full_threshold) must
    produce a motor command where index_pip / index_abad are at the
    CLOSED anchor while every other motor stays at OPEN.

    This explicitly opts into the stretch since the live default is
    linear (``apply_curl_compensation=False``).
    """
    raw = np.array([0.10, 0.50, 0.10, 0.10, 0.10])
    cmd = per_finger_grasp_command_from_curls(
        "left", raw, apply_curl_compensation=True
    )
    full_closed = per_finger_grasp_command_from_curls(
        "left",
        _curl_array(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
        apply_curl_compensation=False,
    )
    full_open = per_finger_grasp_command_from_curls(
        "left", _curl_array(), apply_curl_compensation=False
    )
    np.testing.assert_allclose(cmd[3], full_closed[3], atol=1e-12)
    np.testing.assert_allclose(cmd[4], full_closed[4], atol=1e-12)
    for motor_idx in (0, 1, 2, 5, 6, 7, 8, 9):
        np.testing.assert_allclose(cmd[motor_idx], full_open[motor_idx], atol=1e-12)


def test_stretch_finger_curls_per_finger_arrays() -> None:
    """Per-finger ``deadzone`` / ``full_threshold`` / ``gamma``
    arrays apply different parameters to different fingers in the
    same call. The default behaviour exercises this implicitly.
    """
    # Construct a raw input where each finger sits at its own
    # deadzone boundary (per-finger defaults). All five must map
    # to exactly 0.
    dz = np.array([0.25, 0.35, 0.35, 0.35, 0.35])
    raw_at_dz = dz.copy()
    np.testing.assert_array_equal(stretch_finger_curls(raw_at_dz), np.zeros(5))

    # Raw input where each finger sits exactly at its full_threshold
    # -- all five must saturate to 1.
    full = np.array([0.27, 0.40, 0.40, 0.40, 0.40])
    np.testing.assert_allclose(
        stretch_finger_curls(full), np.ones(5), atol=1e-12,
    )

    # Mix: thumb above its full_threshold, index below its deadzone,
    # middle exactly at its full_threshold, ring inside its active
    # range, pinky above. Expect saturate, 0, saturate, partial, saturate.
    raw = np.array([0.30, 0.20, 0.40, 0.375, 0.50])
    out = stretch_finger_curls(raw)
    assert out[0] == 1.0  # thumb 0.30 > 0.27
    assert out[1] == 0.0  # index 0.20 < 0.35
    assert out[2] == 1.0  # middle 0.40 = full
    # ring 0.375 -> t = 0.5 -> 0.5**5 = 0.03125
    np.testing.assert_allclose(out[3], 0.5 ** 5.0, atol=1e-12)
    assert out[4] == 1.0  # pinky 0.50 > 0.40

    # Custom per-finger gamma: pass a non-uniform array.
    out2 = stretch_finger_curls(
        np.full(5, 0.375),
        deadzone=np.full(5, 0.35),
        full_threshold=np.full(5, 0.40),
        gamma=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    )
    expected = np.array([0.5 ** g for g in (1.0, 2.0, 3.0, 4.0, 5.0)])
    np.testing.assert_allclose(out2, expected, atol=1e-12)


def test_stretch_finger_curls_validation() -> None:
    """Reject malformed inputs and out-of-range parameters."""
    import pytest

    # Wrong shape.
    with pytest.raises(ValueError, match=r"finger_curls must be \(5,\)"):
        stretch_finger_curls(np.zeros(4))

    # Invalid deadzone / full_threshold ordering.
    with pytest.raises(ValueError, match="deadzone < full_threshold"):
        stretch_finger_curls(np.zeros(5), deadzone=0.5, full_threshold=0.5)
    with pytest.raises(ValueError, match="deadzone < full_threshold"):
        stretch_finger_curls(np.zeros(5), deadzone=0.6, full_threshold=0.4)
    with pytest.raises(ValueError, match="deadzone < full_threshold"):
        stretch_finger_curls(np.zeros(5), deadzone=-0.1, full_threshold=0.5)

    # Invalid gamma (must be > 0).
    with pytest.raises(ValueError, match="gamma must be > 0"):
        stretch_finger_curls(np.zeros(5), gamma=0.0)
    with pytest.raises(ValueError, match="gamma must be > 0"):
        stretch_finger_curls(np.zeros(5), gamma=-1.0)

    # Per-finger array shape mismatch.
    with pytest.raises(ValueError, match=r"deadzone must be a scalar or \(5,\)"):
        stretch_finger_curls(np.zeros(5), deadzone=np.array([0.1, 0.2, 0.3]))
    with pytest.raises(ValueError, match=r"full_threshold must be a scalar or \(5,\)"):
        stretch_finger_curls(np.zeros(5), full_threshold=np.array([0.5, 0.6]))


def test_stretch_finger_curls_gamma_one_is_linear() -> None:
    """``gamma = 1`` reduces to the original linear stretch -- a
    useful escape hatch for callers that want the un-eased shape."""
    raws = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    out = stretch_finger_curls(raws, deadzone=0.10, full_threshold=0.30, gamma=1.0)
    np.testing.assert_allclose(
        out, np.array([0.0, 0.25, 0.50, 0.75, 1.00]), atol=1e-12,
    )


def test_stretch_thumb_oppose_suppresses_rest_bleed() -> None:
    """The JS-side ``computeThumbOpposition`` reports ~0.05-0.25
    even at relaxed-hand rest because the thumb tip naturally sits
    a few cm from the index fingertip. ``stretch_thumb_oppose``
    must suppress this bleed (output 0 below deadzone) and
    saturate clear thumb-finger touches."""
    # Below deadzone -> 0
    for v in (0.0, 0.05, 0.10, 0.15, 0.20, 0.24):
        assert stretch_thumb_oppose(v) == 0.0, (
            f"oppose={v} must produce 0 closure (rest-bleed); "
            f"got {stretch_thumb_oppose(v)}"
        )
    # At/above full_threshold -> 1
    for v in (0.40, 0.50, 0.85, 0.95, 1.00):
        np.testing.assert_allclose(
            stretch_thumb_oppose(v), 1.0, atol=1e-12,
            err_msg=f"oppose={v} must saturate to 1; got {stretch_thumb_oppose(v)}",
        )
    # Out-of-range clamps gracefully
    assert stretch_thumb_oppose(-0.5) == 0.0
    assert stretch_thumb_oppose(1.5) == 1.0


def test_stretch_thumb_oppose_validates_params() -> None:
    """Reject malformed parameters."""
    import pytest
    with pytest.raises(ValueError, match="deadzone < full_threshold"):
        stretch_thumb_oppose(0.5, deadzone=0.6, full_threshold=0.5)
    with pytest.raises(ValueError, match="gamma must be > 0"):
        stretch_thumb_oppose(0.5, gamma=0.0)


def test_oppose_compensation_when_enabled_suppresses_rest_oppose() -> None:
    """At a synthetic "relaxed bend" gesture (raw thumb=0.18,
    oppose=0.10) the thumb_roll/abad motors must stay at OPEN when the
    caller explicitly opts into ``apply_oppose_compensation=True``.

    The live default is linear (``apply_oppose_compensation=False``),
    in which case oppose=0.10 leaks 10 % closure into both motors --
    that is covered by :func:`test_oppose_compensation_can_be_disabled`.
    This test only exercises the opt-in compensation path.
    """
    curls = _curl_array(thumb=0.18)
    full_open = per_finger_grasp_command_from_curls(
        "left", _curl_array(), apply_curl_compensation=False,
    )
    cmd_with = per_finger_grasp_command_from_curls_and_oppose(
        "left",
        curls,
        0.10,
        apply_curl_compensation=True,
        apply_oppose_compensation=True,
    )
    np.testing.assert_allclose(cmd_with[0], full_open[0], atol=1e-12)
    np.testing.assert_allclose(cmd_with[1], full_open[1], atol=1e-12)
    np.testing.assert_allclose(cmd_with[2], full_open[2], atol=1e-12)


def test_oppose_compensation_can_be_disabled() -> None:
    """Setting ``apply_oppose_compensation=False`` reverts to the
    legacy direct-lerp behaviour (oppose=0.10 -> 10 % closure)."""
    curls = _curl_array()  # all 0
    full_closed = per_finger_grasp_command_from_curls(
        "left",
        _curl_array(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
        apply_curl_compensation=False,
    )
    full_open = per_finger_grasp_command_from_curls(
        "left", _curl_array(), apply_curl_compensation=False,
    )
    cmd = per_finger_grasp_command_from_curls_and_oppose(
        "left", curls, 0.10,
        apply_curl_compensation=False, apply_oppose_compensation=False,
    )
    expected_roll = 0.9 * full_open[0] + 0.1 * full_closed[0]
    expected_abad = 0.9 * full_open[1] + 0.1 * full_closed[1]
    np.testing.assert_allclose(cmd[0], expected_roll, atol=1e-12)
    np.testing.assert_allclose(cmd[1], expected_abad, atol=1e-12)


def test_oppose_clamps_to_unit_interval() -> None:
    """Out-of-range opposition values get clamped to ``[0, 1]``."""
    curls = _curl_array()
    cmd_neg = per_finger_grasp_command_from_curls_and_oppose("right", curls, -2.0)
    cmd_zero = per_finger_grasp_command_from_curls_and_oppose("right", curls, 0.0)
    cmd_huge = per_finger_grasp_command_from_curls_and_oppose("right", curls, 17.0)
    cmd_one = per_finger_grasp_command_from_curls_and_oppose("right", curls, 1.0)
    np.testing.assert_allclose(cmd_neg, cmd_zero, atol=1e-12)
    np.testing.assert_allclose(cmd_huge, cmd_one, atol=1e-12)


# ── Per-finger range normalization ─────────────────────────────────────


def test_normalize_finger_curls_floor_maps_to_zero() -> None:
    """Curls at the per-finger floor map exactly to 0 (operator's
    "fully open hand")."""
    floor = np.array([0.20, 0.10, 0.10, 0.10, 0.10])
    ceiling = np.array([0.95, 0.85, 0.85, 0.85, 0.85])
    out = normalize_finger_curls(floor.copy(), floor=floor, ceiling=ceiling)
    np.testing.assert_allclose(out, np.zeros(5), atol=1e-12)


def test_normalize_finger_curls_ceiling_maps_to_one() -> None:
    """Curls at the per-finger ceiling map exactly to 1 (operator's
    "fullest fist")."""
    floor = np.array([0.20, 0.10, 0.10, 0.10, 0.10])
    ceiling = np.array([0.95, 0.85, 0.85, 0.85, 0.85])
    out = normalize_finger_curls(ceiling.copy(), floor=floor, ceiling=ceiling)
    np.testing.assert_allclose(out, np.ones(5), atol=1e-12)


def test_normalize_finger_curls_below_floor_clips_to_zero() -> None:
    raw = np.array([0.05, 0.0, 0.0, 0.0, 0.0])
    floor = np.array([0.20, 0.10, 0.10, 0.10, 0.10])
    ceiling = np.array([0.95, 0.85, 0.85, 0.85, 0.85])
    out = normalize_finger_curls(raw, floor=floor, ceiling=ceiling)
    np.testing.assert_allclose(out, np.zeros(5), atol=1e-12)


def test_normalize_finger_curls_above_ceiling_clips_to_one() -> None:
    raw = np.array([1.0, 0.95, 0.92, 0.99, 0.90])
    floor = np.array([0.20, 0.10, 0.10, 0.10, 0.10])
    ceiling = np.array([0.95, 0.85, 0.85, 0.85, 0.85])
    out = normalize_finger_curls(raw, floor=floor, ceiling=ceiling)
    np.testing.assert_allclose(out, np.ones(5), atol=1e-12)


def test_normalize_finger_curls_midpoint_is_linear() -> None:
    """The midpoint between floor and ceiling maps to 0.5 (proves the
    rescale is linear, not power / deadzone)."""
    floor = np.array([0.20, 0.10, 0.10, 0.10, 0.10])
    ceiling = np.array([0.95, 0.85, 0.85, 0.85, 0.85])
    midpoint = 0.5 * (floor + ceiling)
    out = normalize_finger_curls(midpoint, floor=floor, ceiling=ceiling)
    np.testing.assert_allclose(out, 0.5 * np.ones(5), atol=1e-12)


def test_normalize_finger_curls_validation() -> None:
    raw = np.zeros(5)
    with pytest.raises(ValueError, match="ceiling > floor"):
        normalize_finger_curls(raw, floor=np.full(5, 0.5), ceiling=np.full(5, 0.4))
    with pytest.raises(ValueError, match="ceiling > floor"):
        normalize_finger_curls(raw, floor=np.full(5, 0.5), ceiling=np.full(5, 0.5))
    with pytest.raises(ValueError, match=r"\(5,\)"):
        normalize_finger_curls(np.zeros(4), floor=np.zeros(5), ceiling=np.ones(5))


def test_normalize_finger_curls_scalar_broadcasts() -> None:
    """Scalar floor/ceiling should broadcast to all 5 fingers."""
    out = normalize_finger_curls(np.full(5, 0.5), floor=0.2, ceiling=0.8)
    np.testing.assert_allclose(out, np.full(5, 0.5), atol=1e-12)


def test_normalize_thumb_oppose_floor_and_ceiling() -> None:
    assert normalize_thumb_oppose(0.0, floor=0.0, ceiling=0.5) == 0.0
    assert normalize_thumb_oppose(0.5, floor=0.0, ceiling=0.5) == 1.0
    assert normalize_thumb_oppose(1.0, floor=0.0, ceiling=0.5) == 1.0
    assert abs(normalize_thumb_oppose(0.25, floor=0.0, ceiling=0.5) - 0.5) < 1e-12


def test_per_finger_cmd_with_normalization_reaches_anchors() -> None:
    """End-to-end: with floor=p05/ceiling=p95 (typical Quest 3 stats),
    raw curl at the floor produces motor commands at the OPEN anchor,
    and raw curl at the ceiling produces motor commands at the CLOSED
    anchor. The recorded "linear baseline" cannot do this because
    Quest 3 never emits raw=0 (resting bias) or raw=1 (incomplete fist)."""
    floor = np.array([0.20, 0.10, 0.10, 0.10, 0.10])
    ceiling = np.array([0.95, 0.85, 0.85, 0.85, 0.85])
    cmd_open = per_finger_grasp_command_from_curls(
        "left", floor.copy(),
        curl_floor=floor, curl_ceiling=ceiling,
    )
    cmd_closed = per_finger_grasp_command_from_curls(
        "left", ceiling.copy(),
        curl_floor=floor, curl_ceiling=ceiling,
    )
    np.testing.assert_allclose(
        cmd_open, np.asarray(HAND_GRASP_OPEN_RAD_LEFT), atol=1e-12
    )
    np.testing.assert_allclose(
        cmd_closed, np.asarray(HAND_GRASP_CLOSED_RAD_LEFT), atol=1e-12
    )


def test_per_finger_cmd_with_normalization_and_oppose_reaches_anchors() -> None:
    """Same as above but through the oppose-aware retargeting path,
    with thumb-opposition normalization also enabled."""
    floor = np.array([0.20, 0.10, 0.10, 0.10, 0.10])
    ceiling = np.array([0.95, 0.85, 0.85, 0.85, 0.85])
    cmd_open = per_finger_grasp_command_from_curls_and_oppose(
        "right", floor.copy(), 0.0,
        curl_floor=floor, curl_ceiling=ceiling,
        oppose_floor=0.0, oppose_ceiling=0.5,
    )
    cmd_closed = per_finger_grasp_command_from_curls_and_oppose(
        "right", ceiling.copy(), 0.5,
        curl_floor=floor, curl_ceiling=ceiling,
        oppose_floor=0.0, oppose_ceiling=0.5,
    )
    np.testing.assert_allclose(
        cmd_open, np.asarray(HAND_GRASP_OPEN_RAD_RIGHT), atol=1e-12
    )
    np.testing.assert_allclose(
        cmd_closed, np.asarray(HAND_GRASP_CLOSED_RAD_RIGHT), atol=1e-12
    )


def test_per_finger_cmd_normalization_kwargs_paired() -> None:
    """``curl_floor`` / ``curl_ceiling`` must be passed together."""
    raw = np.full(5, 0.5)
    with pytest.raises(ValueError, match="together"):
        per_finger_grasp_command_from_curls("left", raw, curl_floor=np.zeros(5))
    with pytest.raises(ValueError, match="together"):
        per_finger_grasp_command_from_curls("left", raw, curl_ceiling=np.ones(5))
    with pytest.raises(ValueError, match="together"):
        per_finger_grasp_command_from_curls_and_oppose(
            "left", raw, 0.5, oppose_floor=0.1
        )
    with pytest.raises(ValueError, match="together"):
        per_finger_grasp_command_from_curls_and_oppose(
            "left", raw, 0.5, oppose_ceiling=0.5
        )


def test_per_finger_cmd_no_normalization_matches_linear_baseline() -> None:
    """When ``curl_floor`` / ``curl_ceiling`` are None (default), the
    output must be byte-identical to the un-normalised linear lerp.
    This is the regression guard that ensures the normalization is
    purely opt-in."""
    rng = np.random.default_rng(42)
    for _ in range(20):
        raw = rng.uniform(0.0, 1.0, size=5)
        oppose = float(rng.uniform(0.0, 1.0))
        for side in ("left", "right"):
            without = per_finger_grasp_command_from_curls(side, raw.copy())
            without_paired_None = per_finger_grasp_command_from_curls(
                side, raw.copy(), curl_floor=None, curl_ceiling=None
            )
            np.testing.assert_array_equal(without, without_paired_None)
            without_op = per_finger_grasp_command_from_curls_and_oppose(
                side, raw.copy(), oppose,
            )
            without_op_paired_None = per_finger_grasp_command_from_curls_and_oppose(
                side, raw.copy(), oppose,
                curl_floor=None, curl_ceiling=None,
                oppose_floor=None, oppose_ceiling=None,
            )
            np.testing.assert_array_equal(without_op, without_op_paired_None)
