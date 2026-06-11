"""Unit tests for gear_sonic.utils.pose_pipeline.arbitrate.

These exercise the TakeoverArbiter state machine in isolation -- no
ZMQ, no subprocesses, no sleeps. The end-to-end behaviour through
real sockets is covered by tests/test_x2_pose_mux_dual_source.py
(gated on X2_POSE_PROXY_SMOKE=1); this file pins the pure decision
logic so any regression that breaks the engage / release ladder is
caught in the fast unit-test pass.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from gear_sonic.utils.pose_pipeline.arbitrate import (
    ArbiterConfig,
    ArbiterDecision,
    EDGE_ENGAGED,
    EDGE_NONE,
    EDGE_RELEASED,
    SOURCE_NEITHER,
    SOURCE_OVERRIDE,
    SOURCE_PRIMARY,
    TakeoverArbiter,
)
from gear_sonic.utils.pose_pipeline.wire import (
    NUM_BODY_DOFS,
    pack_pose_message,
)


def _pack(jpos_value: float, topic: str = "pose") -> bytes:
    payload = {
        "joint_pos_mj": np.full(
            NUM_BODY_DOFS, jpos_value, dtype=np.float32
        ),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
    }
    return pack_pose_message(payload, topic=topic, version=4)


def _legacy_cfg(**overrides) -> ArbiterConfig:
    """Build a config that disables the new hysteresis defaults so we
    can test the basic engage/release ladder without having to feed
    10 consecutive non-frozen frames before every engage."""
    base = dict(
        override_stale_s=0.100,
        frozen_ticks_threshold=0,
        frozen_l2_tol=1e-4,
        engage_motion_threshold=0,
        teleop_mode_enabled=False,
        engagement_max_wire_step=0.0,
        engagement_steady_wire_step=0.0,
        engagement_step_ramp_ticks=0,
    )
    base.update(overrides)
    return ArbiterConfig(**base)


# ===========================================================================
# Engage / release edges (basic ladder, no hysteresis, no ramp)
# ===========================================================================
def test_engage_edge_fires_once_on_first_override_frame() -> None:
    arb = TakeoverArbiter(_legacy_cfg())
    now = 1000.0
    # Primary alone: no engage.
    arb.observe_primary(_pack(0.0), now=now)
    d = arb.decide(
        now=now, tick=0, primary_fresh=True, override_recvd_this_tick=False,
    )
    assert d.source == SOURCE_PRIMARY
    assert d.edge == EDGE_NONE

    # Override arrives: engage edge fires this tick.
    arb.observe_override(_pack(0.1), now=now)
    d = arb.decide(
        now=now, tick=1, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_OVERRIDE
    assert d.edge == EDGE_ENGAGED
    assert d.engage_event_payload is not None
    parsed = json.loads(d.engage_event_payload.decode("utf-8"))
    assert parsed["event"] == "override_engaged"
    assert parsed["tick"] == 1

    # Next tick with override still fresh: no edge.
    arb.observe_override(_pack(0.2), now=now)
    d = arb.decide(
        now=now, tick=2, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_OVERRIDE
    assert d.edge == EDGE_NONE


def test_release_edge_fires_once_when_override_goes_silent() -> None:
    arb = TakeoverArbiter(_legacy_cfg(override_stale_s=0.1))
    t = 1000.0
    # Engage.
    arb.observe_primary(_pack(0.0), now=t)
    arb.observe_override(_pack(0.1), now=t)
    arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    # Advance time past override-stale window; primary still fresh.
    t_late = t + 0.5
    arb.observe_primary(_pack(0.0), now=t_late)
    d = arb.decide(
        now=t_late, tick=1,
        primary_fresh=True, override_recvd_this_tick=False,
    )
    assert d.source == SOURCE_PRIMARY
    assert d.edge == EDGE_RELEASED
    assert d.release_event_payload is not None
    parsed = json.loads(d.release_event_payload.decode("utf-8"))
    assert parsed["event"] == "override_released"
    assert "release_pose" in parsed
    assert "joint_pos_mj" in parsed["release_pose"]
    # The release_pose joint_pos_mj must match the LAST override jpos.
    assert pytest.approx(parsed["release_pose"]["joint_pos_mj"][0]) == 0.1


def test_neither_source_when_both_silent() -> None:
    arb = TakeoverArbiter(_legacy_cfg())
    d = arb.decide(
        now=1000.0, tick=0,
        primary_fresh=False, override_recvd_this_tick=False,
    )
    assert d.source == SOURCE_NEITHER
    assert d.edge == EDGE_NONE


# ===========================================================================
# Frozen-frame release detection
# ===========================================================================
def test_frozen_detection_fires_release_without_silence() -> None:
    """Send N consecutive identical override frames; release should
    fire even though the SUB is still receiving bytes."""
    arb = TakeoverArbiter(_legacy_cfg(
        frozen_ticks_threshold=3, frozen_l2_tol=1e-4,
    ))
    t = 1000.0
    arb.observe_primary(_pack(0.0), now=t)
    # Engage with motion.
    arb.observe_override(_pack(0.1), now=t)
    arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    # 3 frozen frames identical to (0.2, 0.2, 0.2). frozen_count
    # increments on the second + third (delta vs prev_override_jpos).
    arb.observe_override(_pack(0.2), now=t)  # baseline (replaces 0.1)
    arb.observe_override(_pack(0.2), now=t)  # frozen_count = 1
    arb.observe_override(_pack(0.2), now=t)  # frozen_count = 2
    arb.observe_override(_pack(0.2), now=t)  # frozen_count = 3 -> latch
    assert arb.override_frozen_detected
    # Now decide: override is "still fresh" by silence but
    # frozen_detected=True forces override_fresh=False in the gate.
    d = arb.decide(
        now=t, tick=1, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_PRIMARY
    assert d.edge == EDGE_RELEASED


def test_frozen_latch_clears_on_motion() -> None:
    arb = TakeoverArbiter(_legacy_cfg(
        frozen_ticks_threshold=2, frozen_l2_tol=1e-4,
    ))
    t = 1000.0
    arb.observe_primary(_pack(0.0), now=t)
    # Three identical override frames -> frozen latch.
    arb.observe_override(_pack(0.1), now=t)
    arb.observe_override(_pack(0.1), now=t)
    arb.observe_override(_pack(0.1), now=t)
    assert arb.override_frozen_detected
    # A frame with non-trivial motion clears the latch.
    arb.observe_override(_pack(0.5), now=t)
    assert not arb.override_frozen_detected


# ===========================================================================
# Motion hysteresis on engage
# ===========================================================================
def test_motion_hysteresis_blocks_single_frame_engage() -> None:
    """With engage_motion_threshold=3, a single non-frozen frame must
    NOT engage; only after 3 consecutive moving frames does the gate
    open."""
    arb = TakeoverArbiter(_legacy_cfg(
        engage_motion_threshold=3,
        frozen_ticks_threshold=0,
        frozen_l2_tol=1e-4,
    ))
    t = 1000.0
    arb.observe_primary(_pack(0.0), now=t)
    # First override frame: baseline; motion count = 0.
    arb.observe_override(_pack(0.1), now=t)
    d = arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_PRIMARY  # hysteresis blocks engage
    assert d.edge == EDGE_NONE
    # 2nd, 3rd: motion_count goes to 1, 2.
    arb.observe_override(_pack(0.2), now=t)
    arb.observe_override(_pack(0.3), now=t)
    d = arb.decide(
        now=t, tick=1, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_PRIMARY
    # 4th: motion_count = 3 -> threshold met -> engage.
    arb.observe_override(_pack(0.4), now=t)
    d = arb.decide(
        now=t, tick=2, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_OVERRIDE
    assert d.edge == EDGE_ENGAGED


# ===========================================================================
# Strict stream_mode gate
# ===========================================================================
def test_strict_mode_blocks_engage_when_mode_off() -> None:
    arb = TakeoverArbiter(_legacy_cfg(
        teleop_mode_enabled=True, teleop_mode_stale_s=1.0,
    ))
    t = 1000.0
    arb.observe_teleop_mode("OFF", now=t)
    arb.observe_override(_pack(0.1), now=t)
    d = arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_PRIMARY  # OFF blocks engage
    assert d.edge == EDGE_NONE


def test_strict_mode_engages_when_arm_manipulation() -> None:
    arb = TakeoverArbiter(_legacy_cfg(
        teleop_mode_enabled=True, teleop_mode_stale_s=1.0,
    ))
    t = 1000.0
    arb.observe_teleop_mode("ARM_MANIPULATION", now=t)
    arb.observe_override(_pack(0.1), now=t)
    d = arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_OVERRIDE
    assert d.edge == EDGE_ENGAGED


def test_strict_mode_fails_closed_on_stale_signal() -> None:
    """Strict mode: stale stream_mode -> engagement BLOCKED. A dead
    manager fails closed within teleop_mode_stale_s."""
    arb = TakeoverArbiter(_legacy_cfg(
        teleop_mode_enabled=True, teleop_mode_stale_s=0.1,
    ))
    t = 1000.0
    arb.observe_teleop_mode("ARM_MANIPULATION", now=t)
    arb.observe_override(_pack(0.1), now=t)
    # Decide at t (fresh mode): engages.
    d = arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    assert d.source == SOURCE_OVERRIDE
    assert d.edge == EDGE_ENGAGED
    # Advance time past stale window with no fresh mode messages.
    t_late = t + 0.5
    arb.observe_override(_pack(0.2), now=t_late)
    d = arb.decide(
        now=t_late, tick=1,
        primary_fresh=True, override_recvd_this_tick=True,
    )
    # Mode signal stale -> blocked -> release edge.
    assert d.source == SOURCE_PRIMARY
    assert d.edge == EDGE_RELEASED


# ===========================================================================
# Engagement slow-step ramp
# ===========================================================================
def test_engagement_ramp_clamps_first_override_jpos_step() -> None:
    """At the engage edge, with a primary anchor at 0.0 and an
    override at 0.5 + max_step 0.1, the first OVERRIDE frame should
    be clamped to 0.1 not 0.5."""
    arb = TakeoverArbiter(_legacy_cfg(
        engagement_max_wire_step=0.1,
        engagement_steady_wire_step=0.1,
        engagement_step_ramp_ticks=10,
    ))
    t = 1000.0
    arb.observe_primary(_pack(0.0), now=t)
    arb.observe_override(_pack(0.5), now=t)
    arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    msg = _pack(0.5)
    clamped_msg, clamped_jpos = arb.maybe_clamp_override(msg)
    assert clamped_jpos is not None
    # peak delta = 0.5, max_step = 0.1 -> factor 0.2 -> 0.5 * 0.2 = 0.1
    np.testing.assert_allclose(
        clamped_jpos, np.full(NUM_BODY_DOFS, 0.1), atol=1e-6
    )
    # The returned msg must NOT be the same bytes as the input msg
    # (the clamp rebuilds joint_pos_mj + future).
    assert clamped_msg != msg


def test_engagement_ramp_passthrough_after_window() -> None:
    arb = TakeoverArbiter(_legacy_cfg(
        engagement_max_wire_step=0.1,
        engagement_steady_wire_step=0.1,
        engagement_step_ramp_ticks=2,  # tiny window
    ))
    t = 1000.0
    arb.observe_primary(_pack(0.0), now=t)
    arb.observe_override(_pack(0.5), now=t)
    arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    # Drain the ramp window.
    for _ in range(3):
        arb.maybe_clamp_override(_pack(0.5))
    # After the window: forwarded verbatim (jpos = 0.5).
    msg = _pack(0.5)
    clamped_msg, clamped_jpos = arb.maybe_clamp_override(msg)
    assert clamped_jpos is not None
    np.testing.assert_allclose(
        clamped_jpos, np.full(NUM_BODY_DOFS, 0.5), atol=1e-6
    )
    assert clamped_msg == msg


def test_release_payload_includes_hand_joints_when_present() -> None:
    arb = TakeoverArbiter(_legacy_cfg(override_stale_s=0.05))
    t = 1000.0
    arb.observe_primary(_pack(0.0), now=t)
    payload = {
        "joint_pos_mj": np.full(NUM_BODY_DOFS, 0.3, dtype=np.float32),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        "left_hand_joints": np.full(10, 0.7, dtype=np.float32),
        "right_hand_joints": np.full(10, 0.8, dtype=np.float32),
    }
    op_msg = pack_pose_message(payload, topic="pose", version=4)
    arb.observe_override(op_msg, now=t)
    arb.decide(
        now=t, tick=0, primary_fresh=True, override_recvd_this_tick=True,
    )
    # Advance past stale window.
    t_late = t + 0.5
    arb.observe_primary(_pack(0.0), now=t_late)
    d = arb.decide(
        now=t_late, tick=1,
        primary_fresh=True, override_recvd_this_tick=False,
    )
    assert d.edge == EDGE_RELEASED
    parsed = json.loads(d.release_event_payload.decode("utf-8"))
    rp = parsed["release_pose"]
    assert "joint_pos_mj" in rp
    assert "left_hand_joints" in rp
    assert "right_hand_joints" in rp
    assert pytest.approx(rp["left_hand_joints"][0]) == 0.7
    assert pytest.approx(rp["right_hand_joints"][0]) == 0.8
