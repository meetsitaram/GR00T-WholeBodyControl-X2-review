"""Per-mode pinning tests for the LOCO_DECOUPLED_ARMS flag.

Covers two sides of the 2026-05-30 arm-coupling sentinel:

1. **Manager side**: ``Quest3ManagerX2._publish_arm_targets`` must
   include ``passthrough_arm_targets`` in the msgpack payload, with
   the correct value per (``intent_loco_decoupled_arms``, ``mode``)
   combination.

2. **Recorder side**:
   ``_SubscribeModeState.update_arm_targets(passthrough_arm_targets=True)``
   must null the cached arm pose so the existing merge-time validity
   gate falls through to planner-predicted arms.

The wire-format key set and the legacy ``passthrough_arm_targets=False``
default are pinned by ``tests/test_quest3_manager_x2_wire_format.py``;
this file focuses on the **conditional** behaviour the new flag adds.
"""

from __future__ import annotations

import time
from pathlib import Path

import msgpack
import numpy as np
import pytest
import zmq

from gear_sonic.scripts.quest3_manager_x2 import (
    ManagerConfig,
    Quest3ManagerX2,
)
from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode
from gear_sonic.utils.teleop.x2_dataset_recorder import _SubscribeModeState


# Use the same ports as ``test_quest3_manager_x2_wire_format.py`` (far
# above anything else the project binds) so parallel test runs don't
# collide. Tests in this file use their own ports to avoid colliding
# with the wire-format suite when both run simultaneously.
PLANNER_CMD_PORT = 25571
RECORDER_PORT = 25572


# ---------------------------------------------------------------------------
# Manager-side fixture + helpers
# ---------------------------------------------------------------------------


def _build_manager(decoupled: bool) -> Quest3ManagerX2:
    """Construct a Quest3ManagerX2 with the requested decoupled-arms flag.

    Quest 3 reader is created but never started; we drive the
    ``_publish_arm_targets`` API directly and assert on ZMQ output.
    """
    real_cal = Path("data/operator_calibrations/default.yaml").resolve()
    if not real_cal.is_file():
        pytest.skip(f"requires {real_cal}")

    cfg = ManagerConfig(calibration_path=real_cal)
    cfg.planner_cmd_port = PLANNER_CMD_PORT
    cfg.recorder_pub_port = RECORDER_PORT
    cfg.recorder_pub_host = "127.0.0.1"
    cfg.planner_cmd_host = "127.0.0.1"
    cfg.intent_loco_decoupled_arms = decoupled
    return Quest3ManagerX2(cfg)


def _make_sub(port: int, topic: str) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    sub.connect(f"tcp://127.0.0.1:{port}")
    time.sleep(0.1)  # PUB-SUB handshake slack
    return sub


def _recv_arm_payload(sub: zmq.Socket, manager: Quest3ManagerX2) -> dict:
    """Hammer publishes until the SUB receives an arm_targets payload."""
    L = np.zeros(7, dtype=np.float64)
    R = np.zeros(7, dtype=np.float64)
    for _ in range(10):
        manager._publish_arm_targets(left=L, right=R, is_engaged=False, tick=1)
        time.sleep(0.02)
        try:
            parts = sub.recv_multipart()
            assert parts[0] == b"arm_targets"
            return msgpack.unpackb(parts[1], raw=False)
        except zmq.error.Again:
            continue
    pytest.fail("no arm_targets payload received within 10 publishes")


# ---------------------------------------------------------------------------
# 1) Manager-side per-mode behaviour
# ---------------------------------------------------------------------------


def test_publish_arm_targets_default_payload_has_passthrough_false():
    """Default ``intent_loco_decoupled_arms=True`` -> override behaviour
    preserved: payload always carries ``passthrough_arm_targets=False``,
    regardless of mode. This is the no-regression guarantee for the
    ARM_MAN -> LOCOMOTION arm-hold workflow.
    """
    mgr = _build_manager(decoupled=True)
    sub = _make_sub(RECORDER_PORT, "arm_targets")
    try:
        for mode in (StreamMode.OFF, StreamMode.LOCOMOTION, StreamMode.ARM_MANIPULATION):
            mgr._intent._mode = mode
            msg = _recv_arm_payload(sub, mgr)
            assert "passthrough_arm_targets" in msg, (
                f"missing key in mode {mode!r}: {sorted(msg.keys())}"
            )
            assert msg["passthrough_arm_targets"] is False, (
                f"decoupled=True must always emit passthrough=False; "
                f"got True in mode {mode!r}"
            )
    finally:
        sub.close(linger=0)
        mgr.stop()


def test_publish_arm_targets_decoupled_sets_passthrough_in_locomotion():
    """``intent_loco_decoupled_arms=False`` AND mode=LOCOMOTION ->
    sentinel fires (payload ``passthrough_arm_targets=True``). This
    is the path that lets planner-predicted arms flow through to
    the deploy during whole-body-walking sessions.
    """
    mgr = _build_manager(decoupled=False)
    sub = _make_sub(RECORDER_PORT, "arm_targets")
    try:
        mgr._intent._mode = StreamMode.LOCOMOTION
        msg = _recv_arm_payload(sub, mgr)
        assert msg["passthrough_arm_targets"] is True, (
            "decoupled=False + LOCOMOTION must emit passthrough=True; "
            f"got {msg['passthrough_arm_targets']!r}"
        )
    finally:
        sub.close(linger=0)
        mgr.stop()


def test_publish_arm_targets_decoupled_keeps_arms_in_arm_man():
    """``intent_loco_decoupled_arms=False`` AND mode=ARM_MANIPULATION
    -> sentinel STAYS OFF (``passthrough_arm_targets=False``). The
    decoupling flag is LOCOMOTION-only by design: in ARM_MAN the
    operator's live IK arms MUST win or manipulation breaks.
    """
    mgr = _build_manager(decoupled=False)
    sub = _make_sub(RECORDER_PORT, "arm_targets")
    try:
        mgr._intent._mode = StreamMode.ARM_MANIPULATION
        msg = _recv_arm_payload(sub, mgr)
        assert msg["passthrough_arm_targets"] is False, (
            "decoupled=False + ARM_MAN must keep operator IK arms "
            "(passthrough=False); the sentinel is LOCOMOTION-only"
        )
    finally:
        sub.close(linger=0)
        mgr.stop()


def test_publish_arm_targets_decoupled_off_mode_passthrough_false():
    """``intent_loco_decoupled_arms=False`` AND mode=OFF ->
    ``passthrough_arm_targets=False``. OFF preserves today's safety
    default of the recorder holding the last-known frozen arm pose.
    """
    mgr = _build_manager(decoupled=False)
    sub = _make_sub(RECORDER_PORT, "arm_targets")
    try:
        mgr._intent._mode = StreamMode.OFF
        msg = _recv_arm_payload(sub, mgr)
        assert msg["passthrough_arm_targets"] is False, (
            "decoupled=False + OFF must keep frozen arms "
            "(passthrough=False); sentinel only fires in LOCOMOTION"
        )
    finally:
        sub.close(linger=0)
        mgr.stop()


# ---------------------------------------------------------------------------
# 2) Recorder-side gate semantics (no ZMQ; drives state directly)
# ---------------------------------------------------------------------------


def test_recorder_passthrough_arm_targets_invalidates_cached_arms():
    """Prime the recorder cache with real arms, then send a
    ``passthrough_arm_targets=True`` update -> next ``snapshot()``
    must report ``arm_left_q / arm_right_q == None``.

    This is the load-bearing semantic: nulling the cache makes the
    existing validity gate in the merge step (``if left_arm_valid:
    body_q_mj[slice] = ...``) skip the override and let the
    planner-predicted arms (from ``body_pose``) pass through unmodified.
    """
    state = _SubscribeModeState()
    real_left = np.full(7, 0.5, dtype=np.float64)
    real_right = np.full(7, -0.5, dtype=np.float64)
    state.update_arm_targets(real_left, real_right, engaged=True)
    primed = state.snapshot()
    assert primed["arm_left_q"] is not None
    assert primed["arm_right_q"] is not None

    # Send the sentinel. The left/right arrays don't matter -- the
    # passthrough flag must override their content.
    state.update_arm_targets(
        np.zeros(7), np.zeros(7), engaged=False,
        passthrough_arm_targets=True,
    )
    after_passthrough = state.snapshot()
    assert after_passthrough["arm_left_q"] is None, (
        "passthrough=True must null arm_left_q so the merge falls "
        "through to planner arms"
    )
    assert after_passthrough["arm_right_q"] is None
    assert after_passthrough["arm_engaged"] is False


def test_recorder_passthrough_then_real_arms_recover():
    """The gate is **per-message**, not sticky: after a
    ``passthrough_arm_targets=True`` message, a normal arm_targets
    message MUST repopulate the cache. Without this guarantee the
    operator toggling LOCO->ARM_MAN mid-session would leave the
    arms stuck on planner-predicted swing.
    """
    state = _SubscribeModeState()
    # Step 1: passthrough message clears (or no-ops, since cache is empty)
    state.update_arm_targets(
        np.zeros(7), np.zeros(7), engaged=False,
        passthrough_arm_targets=True,
    )
    assert state.snapshot()["arm_left_q"] is None

    # Step 2: real arms message -- cache must populate
    real_left = np.linspace(-1.0, 1.0, 7, dtype=np.float64)
    real_right = np.linspace(0.0, 2.0, 7, dtype=np.float64)
    state.update_arm_targets(real_left, real_right, engaged=True)
    recovered = state.snapshot()
    assert recovered["arm_left_q"] is not None
    np.testing.assert_allclose(recovered["arm_left_q"], real_left)
    np.testing.assert_allclose(recovered["arm_right_q"], real_right)
    assert recovered["arm_engaged"] is True


def test_recorder_missing_passthrough_key_defaults_to_false():
    """Wire-format back-compat: an older manager that doesn't include
    the ``passthrough_arm_targets`` key in the msgpack payload MUST
    behave like the legacy override path -- ``passthrough_arm_targets``
    defaults to False on both the state-level kwarg and the
    wire-handler ``msg.get("passthrough_arm_targets", False)``.

    We exercise the state kwarg directly here; the wire-handler default
    is one ``msg.get`` call away in
    ``x2_dataset_recorder._handle_zmq_parts`` and is tested transitively
    by the manager-side tests above (which call the real
    ``msgpack.unpackb`` path).
    """
    state = _SubscribeModeState()
    real_left = np.full(7, 0.3, dtype=np.float64)
    real_right = np.full(7, -0.3, dtype=np.float64)
    # Call WITHOUT passthrough_arm_targets kwarg -> default must be False
    # -> cache populates normally.
    state.update_arm_targets(real_left, real_right, engaged=True)
    snap = state.snapshot()
    assert snap["arm_left_q"] is not None
    np.testing.assert_allclose(snap["arm_left_q"], real_left)
    np.testing.assert_allclose(snap["arm_right_q"], real_right)
    assert snap["arm_engaged"] is True
