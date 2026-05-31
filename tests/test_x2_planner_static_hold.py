"""Behaviour tests for the planner's STATIC_HOLD state and ``_HoldTracker``.

STATIC_HOLD is the runtime continuous-waist-hold path: the operator's
right stick (or B-press latch) tells the planner to synthesize a single
31-DOF frame each tick from a (pitch, roll, yaw) target. This test file
pins the state-machine entry / update / exit contracts so the VR stack
(``IntentDecoder`` -> ``quest3_manager_x2`` -> ``x2_heuristic_planner``)
can rely on them.

The covered invariants:

  - Entry from IDLE_LOOP via ``hold_torso`` blends through BLENDING and
    settles in STATIC_HOLD with the synthesized target pose.
  - In-state target updates do NOT trigger another blend; they update
    the tracker and let the slew limit smooth the transition.
  - The slew limit caps per-axis change at HOLD_SLEW_DPS / OUTPUT_FPS
    deg per tick.
  - Exit on a non-hold command blends out of the held pose into the
    requested primitive.
  - ``frame_index`` is monotonic across the entire enter/update/exit
    cycle (must keep test_frame_index_is_monotonic_no_drops happy).
  - ``step_with_lookahead`` snapshot/restore preserves STATIC_HOLD
    state -- the live tracker is not mutated by the peek pass.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.planner import constants as const  # noqa: E402
from gear_sonic.utils.planner.state_machine import (  # noqa: E402
    HOLD_HIP_PITCH_SHARE,
    HOLD_HIP_YAW_SHARE,
    HOLD_SLEW_DPS,
    HOLD_TORSO_INTENT,
    HeuristicPlanner,
    LocomotionCommand,
    OUTPUT_FPS,
    PlannerState,
    Primitive,
    _HoldTracker,
)
from gear_sonic.utils.planner.x2_recipes import make_waist_pose_frame  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _identity_quat_xyzw(n: int) -> np.ndarray:
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 3] = 1.0
    return q


def _idle_prim(n: int = 100) -> Primitive:
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    rot = _identity_quat_xyzw(n)
    trans = np.zeros((n, 3), dtype=np.float64)
    trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return Primitive(
        bin_name="idle_stand",
        family="idle",
        dof=dof,
        root_rot_xyzw=rot,
        root_trans=trans,
        fps=OUTPUT_FPS,
        loopable=True,
        partial=False,
        motion_key="synth-idle",
    )


def _fwd_walk_prim(n: int = 80) -> Primitive:
    """Continuous-walk primitive used to verify exit path."""
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    t = np.linspace(0.0, 2.0 * math.pi, n)
    dof[:, const.LEFT_HIP_PITCH_IDX] += 0.10 * np.sin(t)
    dof[:, const.RIGHT_HIP_PITCH_IDX] += 0.10 * np.sin(t + math.pi)
    rot = _identity_quat_xyzw(n)
    trans = np.zeros((n, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(0.0, 0.30, n)
    trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return Primitive(
        bin_name="fwd_walk_standard",
        family="continuous_walk",
        dof=dof,
        root_rot_xyzw=rot,
        root_trans=trans,
        fps=OUTPUT_FPS,
        loopable=True,
        partial=False,
        motion_key="synth-walk",
    )


def _make_planner() -> HeuristicPlanner:
    prims = {
        "idle_stand": _idle_prim(),
        "fwd_walk_standard": _fwd_walk_prim(),
    }
    return HeuristicPlanner(primitives=prims)


def _hold_cmd(
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
) -> LocomotionCommand:
    return LocomotionCommand(
        intent=HOLD_TORSO_INTENT,
        magnitude="continuous",
        source="test",
        waist_pitch_deg=pitch,
        waist_roll_deg=roll,
        waist_yaw_deg=yaw,
    )


def _step_until_state(
    planner: HeuristicPlanner,
    target_state: PlannerState,
    *,
    max_ticks: int = 200,
) -> int:
    """Step until ``planner.state == target_state``. Returns ticks consumed."""
    ticks = 0
    while planner.state != target_state and ticks < max_ticks:
        planner.step()
        ticks += 1
    if planner.state != target_state:
        raise AssertionError(
            f"expected to reach {target_state} within {max_ticks} ticks; "
            f"still in {planner.state} after {ticks}"
        )
    return ticks


# ---------------------------------------------------------------------------
# _HoldTracker unit tests
# ---------------------------------------------------------------------------


def test_hold_tracker_starts_at_zero() -> None:
    t = _HoldTracker()
    assert t.current_pitch_deg == 0.0
    assert t.target_pitch_deg == 0.0
    assert t.at_target()


def test_hold_tracker_set_target_does_not_snap_current() -> None:
    t = _HoldTracker()
    t.set_target(15.0, -5.0, 30.0)
    assert t.target_pitch_deg == 15.0
    assert t.target_yaw_deg == 30.0
    # Current is still at zero -- requires step() or snap_to_target().
    assert t.current_pitch_deg == 0.0


def test_hold_tracker_snap_to_target() -> None:
    t = _HoldTracker()
    t.set_target(8.0, 2.0, -10.0)
    t.snap_to_target()
    assert t.current_pitch_deg == 8.0
    assert t.current_roll_deg == 2.0
    assert t.current_yaw_deg == -10.0


def test_hold_tracker_step_respects_slew_limit() -> None:
    t = _HoldTracker()
    t.set_target(20.0, 0.0, 0.0)
    dt = 1.0 / OUTPUT_FPS
    max_step = HOLD_SLEW_DPS * dt  # ~1.2 deg
    t.step(dt_s=dt)
    assert t.current_pitch_deg == pytest.approx(max_step)
    # Walks toward the target, never overshoots in one step.
    for _ in range(100):
        t.step(dt_s=dt)
    assert t.at_target()
    assert t.current_pitch_deg == pytest.approx(20.0)


def test_hold_tracker_step_handles_negative_targets() -> None:
    t = _HoldTracker()
    t.current_pitch_deg = 10.0
    t.set_target(-5.0, 0.0, 0.0)
    dt = 1.0 / OUTPUT_FPS
    max_step = HOLD_SLEW_DPS * dt
    t.step(dt_s=dt)
    assert t.current_pitch_deg == pytest.approx(10.0 - max_step)


def test_hold_tracker_step_clamps_to_target_when_close() -> None:
    """When within one slew step of the target, snap exactly to avoid overshoot."""
    t = _HoldTracker()
    t.current_yaw_deg = 0.5
    t.set_target(0.0, 0.0, 0.0)
    dt = 1.0 / OUTPUT_FPS  # max_step = 1.2 deg, > 0.5
    t.step(dt_s=dt)
    assert t.current_yaw_deg == 0.0
    assert t.at_target()


def test_hold_tracker_step_rejects_negative_slew() -> None:
    t = _HoldTracker()
    with pytest.raises(ValueError):
        t.step(dt_s=1.0 / OUTPUT_FPS, slew_dps=-1.0)


# ---------------------------------------------------------------------------
# State transition tests
# ---------------------------------------------------------------------------


def test_idle_to_static_hold_via_blend() -> None:
    p = _make_planner()
    # Settle into idle.
    for _ in range(5):
        p.step()
    assert p.state == PlannerState.IDLE_LOOP

    p.enqueue(_hold_cmd(pitch=10.0, yaw=20.0))
    # Should pass through BLENDING then land in STATIC_HOLD.
    saw_blending = False
    last = None
    for _ in range(50):
        last = p.step()
        if p.state == PlannerState.BLENDING:
            saw_blending = True
        if p.state == PlannerState.STATIC_HOLD:
            break
    assert saw_blending, "expected a blending segment between idle and hold"
    assert p.state == PlannerState.STATIC_HOLD
    # First STATIC_HOLD frame matches the target pose (snap_to_target on entry).
    expected = make_waist_pose_frame(
        pitch_deg=10.0, yaw_deg=20.0,
        hip_pitch_share=HOLD_HIP_PITCH_SHARE,
        hip_yaw_share=HOLD_HIP_YAW_SHARE,
    )
    np.testing.assert_allclose(
        last.joint_pos_mj.astype(np.float64),
        expected,
        atol=1e-5,
    )


def test_static_hold_in_place_target_update_no_blend() -> None:
    """Subsequent hold_torso commands update the tracker; no extra blend."""
    p = _make_planner()
    p.enqueue(_hold_cmd(pitch=10.0))
    _step_until_state(p, PlannerState.STATIC_HOLD)
    # Now update the target. Should NOT enter BLENDING.
    p.enqueue(_hold_cmd(pitch=10.0, yaw=15.0))
    states_seen = set()
    for _ in range(40):
        f = p.step()
        states_seen.add(f.state)
    assert PlannerState.BLENDING not in states_seen, (
        f"hold->hold should stay in STATIC_HOLD; saw {states_seen}"
    )
    assert p.state == PlannerState.STATIC_HOLD


def test_static_hold_target_update_obeys_slew_limit() -> None:
    """The per-tick yaw delta is bounded by HOLD_SLEW_DPS / OUTPUT_FPS deg."""
    p = _make_planner()
    p.enqueue(_hold_cmd(yaw=0.0))
    _step_until_state(p, PlannerState.STATIC_HOLD)
    # Snap the target way out; observe that the emitted waist_yaw walks
    # in capped steps.
    p.enqueue(_hold_cmd(yaw=40.0))
    max_step_rad = math.radians(HOLD_SLEW_DPS) / OUTPUT_FPS
    prev_yaw = float(p.step().joint_pos_mj[const.WAIST_YAW_IDX])
    for _ in range(50):
        f = p.step()
        cur_yaw = float(f.joint_pos_mj[const.WAIST_YAW_IDX])
        assert abs(cur_yaw - prev_yaw) <= max_step_rad + 1e-6, (
            f"per-tick waist_yaw delta exceeded slew cap: "
            f"{abs(cur_yaw - prev_yaw):.4f} > {max_step_rad:.4f}"
        )
        prev_yaw = cur_yaw


def test_static_hold_exits_via_blend_to_walk() -> None:
    """Non-hold cmd in queue triggers blend out of STATIC_HOLD."""
    p = _make_planner()
    p.enqueue(_hold_cmd(pitch=8.0, yaw=20.0))
    _step_until_state(p, PlannerState.STATIC_HOLD)
    # Now request a walk.
    p.enqueue(LocomotionCommand("walk", "forward"))
    saw_blending = False
    saw_playing = False
    for _ in range(100):
        p.step()
        if p.state == PlannerState.BLENDING:
            saw_blending = True
        if p.state == PlannerState.PLAYING:
            saw_playing = True
            break
    assert saw_blending, "expected a blend window between hold and walk"
    assert saw_playing, "expected to reach PLAYING (fwd_walk_standard)"


def test_static_hold_exits_via_blend_to_idle() -> None:
    p = _make_planner()
    p.enqueue(_hold_cmd(pitch=10.0))
    _step_until_state(p, PlannerState.STATIC_HOLD)
    p.enqueue(LocomotionCommand("idle", "default"))
    saw_blending = False
    for _ in range(60):
        p.step()
        if p.state == PlannerState.BLENDING:
            saw_blending = True
        if p.state == PlannerState.IDLE_LOOP:
            break
    assert saw_blending, "expected a blend window between hold and idle"
    assert p.state == PlannerState.IDLE_LOOP


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------


def test_frame_index_monotonic_through_hold_cycle() -> None:
    """Critical: STATIC_HOLD must not break the global tick counter."""
    p = _make_planner()
    indices: list[int] = []
    for _ in range(20):
        indices.append(p.step().frame_index)
    p.enqueue(_hold_cmd(yaw=20.0))
    for _ in range(60):
        indices.append(p.step().frame_index)
    p.enqueue(_hold_cmd(yaw=-30.0))
    for _ in range(60):
        indices.append(p.step().frame_index)
    p.enqueue(LocomotionCommand("walk", "forward"))
    for _ in range(60):
        indices.append(p.step().frame_index)
    diffs = np.diff(indices)
    assert (diffs == 1).all(), (
        f"frame_index has gaps or rewinds: "
        f"min={diffs.min()} max={diffs.max()}"
    )


def test_static_hold_with_lookahead_preserves_live_state() -> None:
    """``step_with_lookahead`` must not mutate tracker state for live ticks."""
    p = _make_planner()
    p.enqueue(_hold_cmd(yaw=15.0))
    _step_until_state(p, PlannerState.STATIC_HOLD)
    # Run a few ticks so the tracker is firmly at target.
    for _ in range(10):
        p.step()
    snap_state = p.state
    snap_yaw = p._hold_tracker.current_yaw_deg

    cur, future = p.step_with_lookahead(num_future=9, step_ticks=5)
    # Live ticks advanced by exactly one (per the contract).
    assert p.state == snap_state
    assert p._hold_tracker.current_yaw_deg == pytest.approx(snap_yaw)
    # All future frames should match the held pose (no slew because we're at target).
    expected_yaw_rad = math.radians(15.0) + const.DEFAULT_STAND_POSE_NP[const.WAIST_YAW_IDX]
    for f in future:
        assert float(f.joint_pos_mj[const.WAIST_YAW_IDX]) == pytest.approx(
            expected_yaw_rad, abs=1e-5,
        )


def test_static_hold_rapid_target_updates_collapse_to_latest() -> None:
    """Multiple hold cmds queued between ticks fold to the last target."""
    p = _make_planner()
    p.enqueue(_hold_cmd(yaw=10.0))
    _step_until_state(p, PlannerState.STATIC_HOLD)
    # Queue several hold cmds in a row; only the last target should
    # influence the tracker.
    p.enqueue(_hold_cmd(yaw=5.0))
    p.enqueue(_hold_cmd(yaw=-10.0))
    p.enqueue(_hold_cmd(yaw=25.0))
    p.step()
    assert p._hold_tracker.target_yaw_deg == 25.0


def test_static_hold_blend_back_seam_continuity() -> None:
    """Per-tick joint jump on blend-out from hold must stay <= 0.06 rad."""
    p = _make_planner()
    p.enqueue(_hold_cmd(pitch=15.0, yaw=30.0))
    _step_until_state(p, PlannerState.STATIC_HOLD)
    # Settle in hold for a while.
    for _ in range(20):
        p.step()
    # Trigger exit.
    p.enqueue(LocomotionCommand("walk", "forward"))
    prev_dof = p.step().joint_pos_mj.astype(np.float64)
    for _ in range(80):
        f = p.step()
        cur = f.joint_pos_mj.astype(np.float64)
        delta = float(np.abs(cur - prev_dof).max())
        assert delta <= 0.06 + 1e-6, (
            f"joint jump on blend out of STATIC_HOLD exceeded gate: "
            f"{delta:.4f} > 0.06 rad in state={f.state}"
        )
        prev_dof = cur


def test_static_hold_arms_remain_at_default_stand_pose() -> None:
    """STATIC_HOLD synthesizes only legs+waist; arms stay frozen at neutral.

    This is the contract that lets ``record_x2_dataset.py`` overlay VR-IK
    arm targets on top of the planner's body_pose without colliding with
    a planner-driven arm signal.
    """
    p = _make_planner()
    p.enqueue(_hold_cmd(pitch=15.0, yaw=20.0))
    _step_until_state(p, PlannerState.STATIC_HOLD)
    f = p.step()
    arm_indices = list(const.LEFT_ARM_INDICES) + list(const.RIGHT_ARM_INDICES)
    np.testing.assert_array_equal(
        f.joint_pos_mj[arm_indices],
        const.DEFAULT_STAND_POSE_NP[arm_indices].astype(np.float32),
    )
