"""Seam continuity tests for the X2 heuristic planner state machine.

Drives ``HeuristicPlanner.step()`` synthetically (no ZMQ, no clock) and
asserts:

  - per-tick joint-angle jump <= ``MAX_DOF_JUMP_RAD`` at every seam
  - per-tick yaw jump <= ``MAX_YAW_JUMP_DEG`` at every seam (incl. wrap-around)
  - state-machine transitions follow the documented FSM
  - command-source latency: an idle-loop interrupt happens within
    ``MAX_INTERRUPT_TICKS`` of enqueueing
  - the planner falls back to ``idle_stand`` (with a warning, not a crash)
    when a command resolves to an unknown bin

These thresholds are chosen so a 50 Hz loop never exceeds the X2 deploy's
per-tick velocity gate (joint vel limit of ~3 rad/s -> 0.06 rad / 20ms).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.planner import constants as const  # noqa: E402
from gear_sonic.utils.planner.state_machine import (  # noqa: E402
    BLEND_FRAMES_END_AT_SQUARE,
    BLEND_FRAMES_STATIC_UPPER_BODY,
    HeuristicPlanner,
    LocomotionCommand,
    OUTPUT_FPS,
    PlannerState,
    Primitive,
    StreamFrame,
    build_pose_payload,
)


MAX_DOF_JUMP_RAD: float = 0.06  # ~3 rad/s at 50 Hz — synthetic-clip + blend bound
MAX_YAW_JUMP_DEG: float = 6.0  # ~5.2 rad/s root-yaw at 50 Hz
MAX_INTERRUPT_TICKS: int = 5  # idle loop must yield within ~100 ms

# Real BONES-SEED mocap clips have natural per-frame dynamics. The v2
# locomotion sources (``walk_forward_loop_004__A042_M`` etc.) are direct
# walk-loop mocap with peak knee-pitch swings of ~0.18 rad per 50 Hz tick at
# heel-strike (joint vel limits on the X2 are 12-30 rad/s depending on group,
# so well within physical capability — knees are ~9 rad/s peak here). Blends
# should still stay within ``MAX_DOF_JUMP_RAD``; this looser bound only
# applies to intra-clip play frames on real curated data.
MAX_DOF_JUMP_RAD_INTRA_MOCAP: float = 0.20


# ---------------------------------------------------------------------------
# Fake primitives
# ---------------------------------------------------------------------------


def _identity_quat_xyzw(n: int) -> np.ndarray:
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 3] = 1.0
    return q


def _quat_from_yaw_xyzw(yaw_rad: float) -> np.ndarray:
    return np.array(
        [0.0, 0.0, math.sin(yaw_rad / 2.0), math.cos(yaw_rad / 2.0)],
        dtype=np.float32,
    )


def _idle_prim(n: int = 100) -> Primitive:
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    rot = _identity_quat_xyzw(n)
    trans = np.zeros((n, 3), dtype=np.float64)
    trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    # Tiny pelvis bob so loop_dof_drift > 0 isn't a problem (curator only).
    trans[:, 2] += 0.002 * np.sin(np.linspace(0, 2 * math.pi, n))
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


def _fwd_step_prim(distance_m: float = 0.1524, n: int = 60) -> Primitive:
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    t = np.linspace(0.0, math.pi, n)
    swing = 0.25 * np.sin(t)
    dof[:, const.LEFT_HIP_PITCH_IDX] += swing
    dof[:, const.RIGHT_HIP_PITCH_IDX] -= swing
    dof[:, const.LEFT_KNEE_IDX] += 0.3 * swing
    dof[:, const.RIGHT_KNEE_IDX] += 0.3 * swing
    dof[-1, :] = const.DEFAULT_STAND_POSE_NP  # snap to square stance at end
    rot = _identity_quat_xyzw(n)
    trans = np.zeros((n, 3), dtype=np.float64)
    trans[:, 0] = np.linspace(0.0, distance_m, n)
    trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return Primitive(
        bin_name="fwd_step_1ft",
        family="locomotion",
        dof=dof,
        root_rot_xyzw=rot,
        root_trans=trans,
        fps=OUTPUT_FPS,
        loopable=False,
        partial=False,
        motion_key="synth-fwd-step",
    )


def _turn_left_prim(yaw_deg: float = 45.0, n: int = 60) -> Primitive:
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    yaws_rad = np.linspace(0.0, math.radians(yaw_deg), n)
    rot = np.zeros((n, 4), dtype=np.float32)
    for i, y in enumerate(yaws_rad):
        rot[i] = _quat_from_yaw_xyzw(float(y))
    trans = np.zeros((n, 3), dtype=np.float64)
    trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    dof[-1, :] = const.DEFAULT_STAND_POSE_NP
    return Primitive(
        bin_name="turn_left_45deg",
        family="locomotion",
        dof=dof,
        root_rot_xyzw=rot,
        root_trans=trans,
        fps=OUTPUT_FPS,
        loopable=False,
        partial=False,
        motion_key="synth-turn-left",
    )


def _lean_fwd_prim(pitch_deg: float = 20.0, n: int = 60) -> Primitive:
    dof = np.tile(const.DEFAULT_STAND_POSE_NP[None, :], (n, 1)).astype(np.float32)
    pitch = math.radians(pitch_deg)
    n_ramp = n // 2
    dof[:n_ramp, const.WAIST_PITCH_IDX] += np.linspace(0.0, pitch, n_ramp)
    dof[n_ramp:, const.WAIST_PITCH_IDX] += pitch
    rot = _identity_quat_xyzw(n)
    trans = np.zeros((n, 3), dtype=np.float64)
    trans[:, 2] = const.DEFAULT_PELVIS_Z_M
    return Primitive(
        bin_name="lean_fwd_medium",
        family="static_upper_body",
        dof=dof,
        root_rot_xyzw=rot,
        root_trans=trans,
        fps=OUTPUT_FPS,
        loopable=False,
        partial=False,
        motion_key="synth-lean",
    )


def _all_synthetic_primitives() -> dict[str, Primitive]:
    # NOTE: registered as ``fwd_step_1ft`` because every fwd_step
    # magnitude is now aliased to the 1ft bin in
    # ``LocomotionCommand.as_bin_name``. Tests below still enqueue
    # ``fwd_step + half_ft`` (so they also exercise the alias).
    return {
        "idle_stand": _idle_prim(),
        "fwd_step_1ft": _fwd_step_prim(),
        "turn_left_45deg": _turn_left_prim(),
        "lean_fwd_medium": _lean_fwd_prim(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _drive(planner: HeuristicPlanner, n_ticks: int) -> list[StreamFrame]:
    return [planner.step() for _ in range(n_ticks)]


def _max_dof_jump(frames: Iterable[StreamFrame]) -> tuple[float, int]:
    peak = 0.0
    peak_at = -1
    prev = None
    for i, f in enumerate(frames):
        if prev is not None:
            jump = float(np.max(np.abs(f.joint_pos_mj - prev)))
            if jump > peak:
                peak = jump
                peak_at = i
        prev = f.joint_pos_mj
    return peak, peak_at


def _max_dof_jump_blend_only(frames: Iterable[StreamFrame]) -> tuple[float, int]:
    """Peak per-tick dof jump within blend windows AND on the blend->play seam."""
    peak = 0.0
    peak_at = -1
    prev = None
    prev_was_blend = False
    for i, f in enumerate(frames):
        if prev is not None and (f.seam_blend or prev_was_blend):
            jump = float(np.max(np.abs(f.joint_pos_mj - prev)))
            if jump > peak:
                peak = jump
                peak_at = i
        prev = f.joint_pos_mj
        prev_was_blend = f.seam_blend
    return peak, peak_at


def _max_yaw_jump_deg(frames: Iterable[StreamFrame]) -> tuple[float, int]:
    peak = 0.0
    peak_at = -1
    prev = None
    for i, f in enumerate(frames):
        if prev is not None:
            d = ((f.yaw_world_deg - prev) + 180.0) % 360.0 - 180.0
            jump = abs(d)
            if jump > peak:
                peak = jump
                peak_at = i
        prev = f.yaw_world_deg
    return peak, peak_at


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_state_machine_starts_in_idle_loop() -> None:
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    f = p.step()
    assert f.state == PlannerState.IDLE_LOOP
    np.testing.assert_allclose(f.joint_pos_mj, const.DEFAULT_STAND_POSE_NP, atol=1e-4)


def test_idle_loop_per_tick_jumps_within_bound() -> None:
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    frames = _drive(p, 250)
    peak, where = _max_dof_jump(frames)
    assert peak <= MAX_DOF_JUMP_RAD, (
        f"idle-loop dof jump {peak:.4f} rad at tick {where} exceeds "
        f"limit {MAX_DOF_JUMP_RAD}"
    )


def test_idle_loop_interrupted_within_max_interrupt_ticks() -> None:
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    _drive(p, 30)  # warm up idle
    p.enqueue(LocomotionCommand(intent="fwd_step", magnitude="half_ft"))
    states = [p.step().state for _ in range(MAX_INTERRUPT_TICKS + 2)]
    assert PlannerState.BLENDING in states or PlannerState.PLAYING in states, (
        f"command never picked up within {MAX_INTERRUPT_TICKS} ticks; "
        f"states seen: {states}"
    )


def test_seam_continuity_idle_to_fwd_step_to_idle() -> None:
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    _drive(p, 30)
    p.enqueue(LocomotionCommand(intent="fwd_step", magnitude="half_ft"))
    frames = _drive(p, 200)
    peak, where = _max_dof_jump(frames)
    assert peak <= MAX_DOF_JUMP_RAD, (
        f"dof jump {peak:.4f} rad at tick {where} exceeds {MAX_DOF_JUMP_RAD}"
    )
    yaw_peak, yaw_where = _max_yaw_jump_deg(frames)
    assert yaw_peak <= MAX_YAW_JUMP_DEG, (
        f"yaw jump {yaw_peak:.2f} deg at tick {yaw_where} exceeds "
        f"{MAX_YAW_JUMP_DEG}"
    )


def test_seam_continuity_through_turn_then_lean() -> None:
    """Stress test: idle -> turn_left_45 -> lean_fwd_medium -> idle."""
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    _drive(p, 25)
    p.enqueue(LocomotionCommand(intent="turn_left", magnitude="deg_45"))
    p.enqueue(LocomotionCommand(intent="lean_fwd", magnitude="medium"))
    p.enqueue(LocomotionCommand(intent="idle", magnitude="default"))
    frames = _drive(p, 400)
    peak, where = _max_dof_jump(frames)
    assert peak <= MAX_DOF_JUMP_RAD, (
        f"dof jump {peak:.4f} rad at tick {where} exceeds {MAX_DOF_JUMP_RAD}"
    )
    yaw_peak, yaw_where = _max_yaw_jump_deg(frames)
    assert yaw_peak <= MAX_YAW_JUMP_DEG, (
        f"yaw jump {yaw_peak:.2f} deg at tick {yaw_where} exceeds "
        f"{MAX_YAW_JUMP_DEG}"
    )


def test_yaw_aligns_to_post_turn_heading_for_next_segment() -> None:
    """After a 45-deg turn, the next segment must start at +45 deg yaw."""
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    _drive(p, 10)
    p.enqueue(LocomotionCommand(intent="turn_left", magnitude="deg_45"))
    # Run long enough for turn + return-to-idle blend to complete.
    frames = _drive(p, 220)
    yaws = [f.yaw_world_deg for f in frames]
    final_yaw = yaws[-1]
    assert 40.0 <= final_yaw <= 50.0, (
        f"after a 45-deg turn the world yaw should be ~45 deg; got {final_yaw:.1f}"
    )


def test_blend_window_lengths_match_family_pairs() -> None:
    """Blends use the documented frame counts based on segment families."""
    from gear_sonic.utils.planner.state_machine import _pick_blend_frames

    assert _pick_blend_frames("idle", "locomotion") == BLEND_FRAMES_END_AT_SQUARE
    assert _pick_blend_frames("locomotion", "locomotion") == BLEND_FRAMES_END_AT_SQUARE
    assert _pick_blend_frames("idle", "continuous_walk") != BLEND_FRAMES_END_AT_SQUARE
    assert _pick_blend_frames("static_upper_body", "idle") == (
        BLEND_FRAMES_STATIC_UPPER_BODY
    )


def test_unknown_command_falls_back_to_idle() -> None:
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    _drive(p, 25)
    p.enqueue(LocomotionCommand(intent="moonwalk", magnitude="forward"))
    # Should not raise and should keep producing valid frames.
    frames = _drive(p, 60)
    assert all(f.joint_pos_mj.shape == (31,) for f in frames)
    peak, _ = _max_dof_jump(frames)
    assert peak <= MAX_DOF_JUMP_RAD


def test_payload_keys_match_zmq_pose_wire_format() -> None:
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    f = p.step()
    payload = build_pose_payload(f, motion_token_dim=64, hand_dof=10)
    assert payload["joint_pos_mj"].shape == (31,)
    assert payload["joint_pos_mj"].dtype == np.float32
    assert payload["root_quat_xyzw"].shape == (4,)
    assert payload["motion_token"].shape == (64,)
    assert payload["left_hand_joints"].shape == (10,)
    assert payload["right_hand_joints"].shape == (10,)
    assert payload["frame_index"].shape == (1,)
    assert payload["frame_index"].dtype == np.int64
    # v4-compat path: no future_* fields when future_frames omitted.
    for k in (
        "joint_pos_mj_future", "root_quat_xyzw_future",
        "joint_vel_mj_future", "frame_index_future", "future_dt_s",
    ):
        assert k not in payload, f"v4-compat payload should not carry {k!r}"


def test_payload_v5_includes_future_window() -> None:
    """v5 payload (future_frames provided) carries strictly-future arrays.

    Pins the exact wire shape the C++ ``ZmqPoseInputSource`` decodes
    in ``HandleDecoded`` for the v5 future-window path. If any of these
    shapes change, the C++ side must change in lock-step or the
    decode-and-promote check at ``zmq_pose_input_source.cpp`` will
    fall back to the v4 single-frame path silently.
    """
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    cur, fut = p.step_with_lookahead(num_future=9, step_ticks=5)
    payload = build_pose_payload(
        cur, motion_token_dim=64, hand_dof=10,
        future_frames=fut, future_dt_s=0.1,
    )
    # v4 fields still present (window[0] for the C++ side).
    assert payload["joint_pos_mj"].shape == (31,)
    assert payload["root_quat_xyzw"].shape == (4,)
    assert payload["frame_index"].shape == (1,)
    # v5 future arrays must match the C++ ``kFutureSlots = NUM_FUTURE_FRAMES - 1``.
    assert payload["joint_pos_mj_future"].shape == (9, 31)
    assert payload["joint_pos_mj_future"].dtype == np.float32
    assert payload["root_quat_xyzw_future"].shape == (9, 4)
    assert payload["root_quat_xyzw_future"].dtype == np.float32
    assert payload["joint_vel_mj_future"].shape == (9, 31)
    assert payload["joint_vel_mj_future"].dtype == np.float32
    assert payload["frame_index_future"].shape == (9,)
    assert payload["frame_index_future"].dtype == np.int64
    assert payload["future_dt_s"].shape == (1,)
    assert payload["future_dt_s"].dtype == np.float32
    assert abs(float(payload["future_dt_s"][0]) - 0.1) < 1e-5
    # frame indices are spaced by step_ticks=5 ticks (= 0.1s @ 50Hz)
    cur_idx = int(payload["frame_index"][0])
    expected = np.array([cur_idx + (k + 1) * 5 for k in range(9)], dtype=np.int64)
    assert np.array_equal(payload["frame_index_future"], expected)


def test_step_with_lookahead_does_not_mutate_live_state() -> None:
    """Snapshot/restore must leave live state IDENTICAL to a plain step().

    Run two parallel planners with identical commands. One uses
    ``step()``, the other uses ``step_with_lookahead()``. After N ticks
    they must agree on every emitted frame. If snapshot/restore leaks
    ANY mutation (cursor offset, queue pop, segment swap), this
    diverges immediately.
    """
    p_plain = HeuristicPlanner(primitives=_all_synthetic_primitives())
    p_lookahead = HeuristicPlanner(primitives=_all_synthetic_primitives())
    for plnr in (p_plain, p_lookahead):
        plnr.enqueue(LocomotionCommand("fwd_step", "half_ft"))
        plnr.enqueue(LocomotionCommand("idle", "default"))
        plnr.enqueue(LocomotionCommand("turn_left", "deg_45"))
    n_ticks = 80
    plain_frames = [p_plain.step() for _ in range(n_ticks)]
    look_frames = []
    for _ in range(n_ticks):
        cur, _ = p_lookahead.step_with_lookahead(num_future=9, step_ticks=5)
        look_frames.append(cur)
    for t, (a, b) in enumerate(zip(plain_frames, look_frames, strict=True)):
        assert a.frame_index == b.frame_index, f"tick {t}: frame_index mismatch"
        assert a.state == b.state, f"tick {t}: state {a.state} != {b.state}"
        assert a.bin_name == b.bin_name, f"tick {t}: bin {a.bin_name} != {b.bin_name}"
        assert np.allclose(a.joint_pos_mj, b.joint_pos_mj, atol=0, rtol=0), (
            f"tick {t}: joint_pos_mj diverged "
            f"(max abs diff {np.max(np.abs(a.joint_pos_mj - b.joint_pos_mj))})"
        )
        assert np.allclose(a.root_quat_xyzw, b.root_quat_xyzw, atol=0, rtol=0)


def test_step_with_lookahead_future_matches_fresh_planner_trajectory() -> None:
    """``future[k]`` must equal what a fresh planner would emit at +(k+1)*step_ticks.

    The C++ ``ZmqPoseInputSource::Sample(time)`` indexes the future
    window assuming ``window[k]`` for k=1..9 is the policy-time-aligned
    pose at +k*0.1s. If the planner's lookahead returns frames at the
    wrong tick offsets, the policy gets a mistimed reference.
    """
    p_main = HeuristicPlanner(primitives=_all_synthetic_primitives())
    p_main.enqueue(LocomotionCommand("fwd_step", "half_ft"))
    p_main.enqueue(LocomotionCommand("idle", "default"))
    cur, fut = p_main.step_with_lookahead(num_future=9, step_ticks=5)

    # Fresh planner with same command queue, plain step() N times.
    p_truth = HeuristicPlanner(primitives=_all_synthetic_primitives())
    p_truth.enqueue(LocomotionCommand("fwd_step", "half_ft"))
    p_truth.enqueue(LocomotionCommand("idle", "default"))
    truth_frames = [p_truth.step() for _ in range(46)]  # 1 current + 9*5 future
    assert truth_frames[0].joint_pos_mj.tolist() == cur.joint_pos_mj.tolist()
    for k in range(9):
        truth_idx = (k + 1) * 5
        truth_f = truth_frames[truth_idx]
        assert fut[k].frame_index == truth_f.frame_index, (
            f"future[{k}].frame_index={fut[k].frame_index} != "
            f"truth[{truth_idx}].frame_index={truth_f.frame_index}"
        )
        assert np.allclose(fut[k].joint_pos_mj, truth_f.joint_pos_mj, atol=0, rtol=0)
        assert np.allclose(fut[k].root_quat_xyzw, truth_f.root_quat_xyzw, atol=0, rtol=0)


def test_step_with_lookahead_invalid_args_rejected() -> None:
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    with pytest.raises(ValueError):
        p.step_with_lookahead(num_future=-1, step_ticks=5)
    with pytest.raises(ValueError):
        p.step_with_lookahead(num_future=9, step_ticks=0)


def test_command_resolves_to_correct_bin_name() -> None:
    cases = [
        (("idle", "default"), "idle_stand"),
        (("walk", "forward"), "fwd_walk_standard"),
        # Every fwd_step magnitude collapses to the 1ft bin; see
        # LocomotionCommand.as_bin_name docstring for rationale.
        (("fwd_step", "half_ft"), "fwd_step_1ft"),
        (("fwd_step", "one_ft"), "fwd_step_1ft"),
        (("fwd_step", "quarter_ft"), "fwd_step_1ft"),
        (("fwd_step", "default"), "fwd_step_1ft"),
        # back_step has no 1ft variant in the current library; every
        # magnitude collapses to back_step_half_ft so the manager's
        # default-magnitude emission resolves to a real primitive
        # (was silently falling back to idle_stand before 2026-05-13).
        (("back_step", "half_ft"), "back_step_half_ft"),
        (("back_step", "quarter_ft"), "back_step_half_ft"),
        (("back_step", "default"), "back_step_half_ft"),
        (("side_left", "default"), "side_left_step"),
        (("side_left", "half_ft"), "side_left_step"),
        (("side_left", "quarter_ft"), "side_left_step"),
        (("side_right", "default"), "side_right_step"),
        (("turn_left", "deg_15"), "turn_left_15deg"),
        (("turn_left", "deg_30"), "turn_left_30deg"),
        (("turn_left", "deg_45"), "turn_left_45deg"),
        (("turn_left", "deg_90"), "turn_left_90deg"),
        (("turn_right", "deg_90"), "turn_right_90deg"),
        (("lean_fwd", "small"), "lean_fwd_small"),
        (("lean_fwd", "medium"), "lean_fwd_medium"),
        (("lean_fwd", "large"), "lean_fwd_large"),
        # Lateral lean family (NEW in v6) -- direct {intent}_{magnitude}
        # mapping, no alias collapse.
        (("lean_left", "small"), "lean_left_small"),
        (("lean_left", "medium"), "lean_left_medium"),
        (("lean_left", "large"), "lean_left_large"),
        (("lean_right", "small"), "lean_right_small"),
        (("lean_right", "medium"), "lean_right_medium"),
        (("lean_right", "large"), "lean_right_large"),
        (("torso_left", "deg_30"), "torso_left_30deg"),
        # torso_*_45deg renamed to torso_*_40deg in v6 (yaw cap = 40 deg).
        (("torso_right", "deg_40"), "torso_right_40deg"),
        (("torso_left", "deg_40"), "torso_left_40deg"),
        # Every crouch magnitude collapses to crouch_medium; see
        # LocomotionCommand.as_bin_name docstring for rationale.
        (("crouch", "small"), "crouch_medium"),
        (("crouch", "medium"), "crouch_medium"),
        (("crouch", "large"), "crouch_medium"),
        (("crouch", "default"), "crouch_medium"),
    ]
    for (intent, magnitude), expected_bin in cases:
        cmd = LocomotionCommand(intent=intent, magnitude=magnitude)
        assert cmd.as_bin_name() == expected_bin, (
            f"({intent},{magnitude}) -> {cmd.as_bin_name()!r}, expected {expected_bin!r}"
        )


def test_replace_pending_clears_backlog_and_returns_dropped_count() -> None:
    """``replace_pending`` should drop the existing pending queue."""
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    p.enqueue(LocomotionCommand(intent="fwd_step", magnitude="half_ft"))
    p.enqueue(LocomotionCommand(intent="turn_left", magnitude="deg_45"))
    p.enqueue(LocomotionCommand(intent="lean_fwd", magnitude="medium"))
    assert p.queue_depth == 3

    dropped = p.replace_pending(
        LocomotionCommand(intent="side_right", magnitude="default", source="kbd")
    )
    assert dropped == 3, f"expected 3 commands dropped, got {dropped}"
    assert p.queue_depth == 1
    head = p._cmd_queue[0]
    assert (head.intent, head.magnitude, head.source) == (
        "side_right", "default", "kbd",
    ), f"queue head should be the new kbd command; got {head!r}"


def test_replace_pending_on_empty_queue_returns_zero() -> None:
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    assert p.queue_depth == 0
    dropped = p.replace_pending(
        LocomotionCommand(intent="side_left", magnitude="default", source="kbd")
    )
    assert dropped == 0
    assert p.queue_depth == 1


def test_replace_pending_does_not_preempt_active_segment() -> None:
    """A blend or play in progress must finish; only the *queue* is reset.

    Preempting mid-stride risks falls because the policy assumes the next
    reference is a small delta from the previous one. The whole point of
    leaving _active alone is to preserve that safety property.
    """
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    _drive(p, 25)  # warm up idle
    p.enqueue(LocomotionCommand(intent="turn_left", magnitude="deg_45"))
    # Drive a few ticks so the planner enters BLENDING / PLAYING.
    _drive(p, 10)
    state_before = p.state
    assert state_before in (PlannerState.BLENDING, PlannerState.PLAYING), (
        f"expected planner to have entered blend/play; state={state_before}"
    )
    pre_active = p._active

    p.replace_pending(
        LocomotionCommand(intent="side_right", magnitude="default", source="kbd")
    )

    assert p._active is pre_active, (
        "replace_pending must NOT touch the active segment (preemption "
        "mid-stride risks falls)"
    )
    # State machine should keep finishing the current segment, not jump.
    f_after = p.step()
    assert f_after.state in (PlannerState.BLENDING, PlannerState.PLAYING), (
        f"replace_pending should not jump the state; state after step = {f_after.state}"
    )


def test_replace_pending_does_not_clear_next_after_active_during_blend() -> None:
    """While a blend is mid-flight, ``_next_after_active`` is the blend's
    target. Clearing it would leave the blend with no destination -- so
    ``replace_pending`` must leave it intact.
    """
    p = HeuristicPlanner(primitives=_all_synthetic_primitives())
    _drive(p, 25)
    p.enqueue(LocomotionCommand(intent="turn_left", magnitude="deg_45"))
    # Step exactly enough to enter the blend (which sets _next_after_active).
    for _ in range(MAX_INTERRUPT_TICKS + 1):
        p.step()
        if p.state == PlannerState.BLENDING and p._next_after_active is not None:
            break
    assert p._next_after_active is not None, (
        "fixture failed to enter a blend with a queued after-command"
    )
    pre_after = p._next_after_active

    p.replace_pending(
        LocomotionCommand(intent="side_right", magnitude="default", source="kbd")
    )

    assert p._next_after_active is pre_after, (
        "replace_pending must NOT clear _next_after_active during a blend "
        "(the blend has nowhere to go without it)"
    )


def _make_planner() -> HeuristicPlanner:
    """Build a planner backed by the synthetic test primitives."""
    return HeuristicPlanner(
        primitives={
            "idle_stand": _idle_prim(),
            "fwd_step_1ft": _fwd_step_prim(),
        }
    )


def test_current_anchor_frame_matches_first_step_and_does_not_advance() -> None:
    """``current_anchor_frame()`` must (1) be bit-identical to ``step()``'s
    first emit and (2) not advance the cursor.

    This is the contract the warmup loop in ``x2_heuristic_planner.py`` and
    ``bake_planner_rsi_anchor.py`` rely on: bridge spawn pose == warmup wire
    content == state-machine first frame, all from the same source.
    """
    p1 = _make_planner()
    anchor = p1.current_anchor_frame()
    # Calling it a second time must return the same values (no internal
    # mutation).
    anchor2 = p1.current_anchor_frame()
    assert np.array_equal(anchor.joint_pos_mj, anchor2.joint_pos_mj)
    assert np.array_equal(anchor.root_quat_xyzw, anchor2.root_quat_xyzw)

    # Now run step() on a fresh planner and compare to the anchor.
    p2 = _make_planner()
    first_step = p2.step()
    assert np.allclose(anchor.joint_pos_mj, first_step.joint_pos_mj, atol=0.0), (
        "anchor joints differ from step()[0] joints"
    )
    assert np.allclose(anchor.root_quat_xyzw, first_step.root_quat_xyzw, atol=0.0), (
        "anchor quat differs from step()[0] quat"
    )
    # And the anchor must publish identity-yaw (planner's initial_yaw_world=0
    # default), independent of whatever yaw the underlying primitive's clip
    # was captured at.
    assert anchor.yaw_world_deg == pytest.approx(0.0, abs=1e-9), (
        f"anchor yaw_world_deg should be 0.0 (initial_yaw_world default), "
        f"got {anchor.yaw_world_deg}"
    )


def test_hold_seconds_yaml_expands_to_idle_commands(tmp_path: Path) -> None:
    """``hold_seconds`` on any intent appends ``round(hold_s)`` idle
    commands after that intent. This is the ``yaml-hold`` sugar path
    that lets demos request "settle on the trained stand pose for ~N
    seconds" between gait segments without freezing the previous
    segment's last frame (which the dropped ``hold_last_pose`` did and
    proved error-prone -- side-step clips end mid-stride).
    """
    from gear_sonic.utils.planner.state_machine import commands_from_yaml

    yaml_path = tmp_path / "hold_test.yaml"
    yaml_path.write_text(
        "commands:\n"
        "  - intent: fwd_step\n"
        "    magnitude: half_ft\n"
        "    hold_seconds: 1.5\n"
        "  - intent: side_left\n"
        "    magnitude: default\n"
    )
    cmds = commands_from_yaml(yaml_path)
    assert cmds[0].intent == "fwd_step"
    n_idle = sum(1 for c in cmds if c.intent == "idle" and c.source == "yaml-hold")
    assert n_idle == 2, (
        f"expected 2 idle expansions for hold_seconds=1.5 (round(1.5)=2); "
        f"got {n_idle}, cmds={cmds}"
    )
    # The expansion must sit BETWEEN the two intents.
    side_idx = next(i for i, c in enumerate(cmds) if c.intent == "side_left")
    fwd_idx = next(i for i, c in enumerate(cmds) if c.intent == "fwd_step")
    assert side_idx > fwd_idx + n_idle


def test_real_curated_primitives_load_and_run() -> None:
    """Smoke test: load the actual curator output and run the planner for 5s.

    Two thresholds: blend windows must stay within MAX_DOF_JUMP_RAD (the math
    is ours), while intra-clip frames are allowed up to
    MAX_DOF_JUMP_RAD_INTRA_MOCAP because real BONES-SEED clips have natural
    transient dynamics (heel-strike, wrist throws).
    """
    from gear_sonic.utils.planner.registry import load_bin_specs
    from gear_sonic.utils.planner.state_machine import load_primitives_pkl

    pkl = REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_primitives.pkl"
    bins_yaml = REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_bins.yaml"
    if not pkl.exists() or not bins_yaml.exists():
        pytest.skip("primitives PKL not curated yet")

    bin_family = {n: s.family for n, s in load_bin_specs(bins_yaml).items()}
    prims = load_primitives_pkl(pkl, bin_family)
    p = HeuristicPlanner(primitives=prims)
    _drive(p, 25)
    p.enqueue(LocomotionCommand(intent="walk", magnitude="forward"))
    p.enqueue(LocomotionCommand(intent="turn_left", magnitude="deg_45"))
    p.enqueue(LocomotionCommand(intent="lean_fwd", magnitude="medium"))
    frames = _drive(p, int(5 * OUTPUT_FPS))

    blend_peak, blend_where = _max_dof_jump_blend_only(frames)
    assert blend_peak <= MAX_DOF_JUMP_RAD, (
        f"real-data BLEND dof jump {blend_peak:.4f} rad at tick {blend_where} "
        f"exceeds {MAX_DOF_JUMP_RAD} — blend math is wrong"
    )
    overall_peak, overall_where = _max_dof_jump(frames)
    assert overall_peak <= MAX_DOF_JUMP_RAD_INTRA_MOCAP, (
        f"real-data overall dof jump {overall_peak:.4f} rad at tick "
        f"{overall_where} exceeds {MAX_DOF_JUMP_RAD_INTRA_MOCAP}"
    )
    yaw_peak, yaw_where = _max_yaw_jump_deg(frames)
    assert yaw_peak <= MAX_YAW_JUMP_DEG, (
        f"real-data yaw jump {yaw_peak:.2f} deg at tick {yaw_where} "
        f"exceeds {MAX_YAW_JUMP_DEG}"
    )


# ---------------------------------------------------------------------------
# STATIC_HOLD seam continuity
# ---------------------------------------------------------------------------
# These tests pin the seam quality of the v7 continuous-waist-hold path.
# We exercise three seams: idle->hold (entry blend), hold->hold (slew-only,
# no blend), and hold->walk (exit blend). Per-tick joint jumps must stay
# within the same MAX_DOF_JUMP_RAD bound the rest of the FSM honors.


def _hold_planner() -> HeuristicPlanner:
    """Planner with the synthetic primitives plus a continuous-walk bin."""
    return HeuristicPlanner(
        primitives={
            "idle_stand": _idle_prim(),
            "fwd_step_1ft": _fwd_step_prim(),
            "turn_left_45deg": _turn_left_prim(),
            "fwd_walk_standard": _fwd_step_prim(distance_m=0.30, n=80),
        }
    )


def test_static_hold_entry_blend_seam_within_dof_gate() -> None:
    """idle -> blend -> STATIC_HOLD must respect MAX_DOF_JUMP_RAD per tick."""
    p = _hold_planner()
    _drive(p, 5)  # settle in idle
    p.enqueue(
        LocomotionCommand(
            intent="hold_torso",
            magnitude="continuous",
            source="test",
            waist_pitch_deg=18.0,
            waist_yaw_deg=35.0,
        )
    )
    frames = _drive(p, 80)
    peak, where = _max_dof_jump(frames)
    assert peak <= MAX_DOF_JUMP_RAD, (
        f"idle->STATIC_HOLD entry seam jump {peak:.4f} rad at tick {where} "
        f"exceeds {MAX_DOF_JUMP_RAD}"
    )
    assert any(f.state == PlannerState.STATIC_HOLD for f in frames)


def test_static_hold_in_state_target_updates_within_slew_cap() -> None:
    """Target hops while in STATIC_HOLD must walk the slew cap, not jump."""
    p = _hold_planner()
    p.enqueue(
        LocomotionCommand(
            intent="hold_torso",
            magnitude="continuous",
            source="test",
            waist_yaw_deg=10.0,
        )
    )
    # Settle in hold.
    while p.state != PlannerState.STATIC_HOLD:
        p.step()
    # Whip the target across the cap repeatedly.
    big_targets = [40.0, -40.0, 25.0, -10.0, 0.0]
    frames: list[StreamFrame] = []
    for tgt in big_targets:
        p.enqueue(
            LocomotionCommand(
                intent="hold_torso",
                magnitude="continuous",
                source="test",
                waist_yaw_deg=tgt,
            )
        )
        frames.extend(_drive(p, 60))
    peak, where = _max_dof_jump(frames)
    assert peak <= MAX_DOF_JUMP_RAD, (
        f"in-state STATIC_HOLD slew jump {peak:.4f} rad at tick {where} "
        f"exceeds {MAX_DOF_JUMP_RAD} (slew limit failed)"
    )


def test_static_hold_exit_blend_seam_within_dof_gate() -> None:
    """STATIC_HOLD -> blend -> walking must respect the per-tick gate."""
    p = _hold_planner()
    p.enqueue(
        LocomotionCommand(
            intent="hold_torso",
            magnitude="continuous",
            source="test",
            waist_pitch_deg=15.0,
            waist_yaw_deg=20.0,
        )
    )
    while p.state != PlannerState.STATIC_HOLD:
        p.step()
    _drive(p, 20)  # dwell at target
    p.enqueue(LocomotionCommand("walk", "forward"))
    frames = _drive(p, 120)
    peak, where = _max_dof_jump(frames)
    assert peak <= MAX_DOF_JUMP_RAD, (
        f"STATIC_HOLD->walk exit seam jump {peak:.4f} rad at tick {where} "
        f"exceeds {MAX_DOF_JUMP_RAD}"
    )
    assert any(f.state == PlannerState.PLAYING for f in frames)
