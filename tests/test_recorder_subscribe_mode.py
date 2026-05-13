"""Integration tests for the recorder's Phase 0 subscribe-only mode.

Covers two layers:

1. **Helper round-trip**: ``_handle_body_pose_msg`` and
   ``_handle_arm_and_hands_msg`` correctly decode the
   wire formats published by the planner
   (:func:`pack_pose_message`) and the manager
   (msgpack + JSON multipart) into ``_SubscribeModeState``.

2. **End-to-end ZMQ**: spinning up the actual
   ``_subscribe_mode_thread`` against fake planner +
   manager publishers, asserting that body_pose, arm_targets,
   hand_finger_cmd, stream_mode and recorder_cmd all flow
   through and update state.

These tests are the consumer-side complement to
``test_quest3_manager_x2_wire_format.py`` (the producer side):
together they pin down the recorder<->manager<->planner ZMQ
contract for the planner-driven Quest 3 stack.

Note on the recorder class itself: full
:class:`X2DatasetRecorder` construction requires the LeRobot
writer chain (``datasets``, ``av``, ``lerobot``) which the
``record_x2_dataset`` CLI installs lazily via
``ensure_runtime_deps``. To keep this test light, we test only
the subscribe-mode helpers directly and let the CLI exercise
the full pipeline in the smoke test.
"""

from __future__ import annotations

import json
import threading
import time

import msgpack
import numpy as np
import pytest
import zmq

from gear_sonic.utils.teleop.x2_dataset_recorder import (
    NUM_BODY_DOFS,
    NUM_HAND_DOF_PER_SIDE,
    RecorderConfig,
    _handle_arm_and_hands_msg,
    _handle_body_pose_msg,
    _SubscribeModeState,
    _subscribe_mode_thread,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


# Use ports far above defaults to avoid colliding with parallel runs.
_PLANNER_PORT = 25665
_MGR_PORT = 25666


# ---------------------------------------------------------------------------
# _SubscribeModeState basics
# ---------------------------------------------------------------------------


def test_subscribe_state_initial_snapshot_is_empty() -> None:
    state = _SubscribeModeState()
    snap = state.snapshot()
    assert snap["body_pose_q_mj"] is None
    assert snap["arm_left_q"] is None
    assert snap["arm_right_q"] is None
    assert snap["arm_engaged"] is False
    assert snap["left_hand_q"] is None
    assert snap["right_hand_q"] is None
    assert snap["stream_mode"] == "OFF"
    assert state.drain_recorder_cmds() == []


def test_subscribe_state_updates_round_trip() -> None:
    state = _SubscribeModeState()
    body = np.linspace(-1.0, 1.0, NUM_BODY_DOFS)
    state.update_body_pose(body)
    state.update_arm_targets(
        np.zeros(7), np.ones(7), engaged=True,
    )
    state.update_hand_finger_cmd(
        np.full(NUM_HAND_DOF_PER_SIDE, 0.5),
        np.full(NUM_HAND_DOF_PER_SIDE, -0.5),
    )
    state.update_stream_mode("LOCOMOTION")
    state.push_recorder_cmd("start", 7)
    state.push_recorder_cmd("save", 42)

    snap = state.snapshot()
    np.testing.assert_array_equal(snap["body_pose_q_mj"], body)
    np.testing.assert_array_equal(snap["arm_left_q"], np.zeros(7))
    np.testing.assert_array_equal(snap["arm_right_q"], np.ones(7))
    assert snap["arm_engaged"] is True
    np.testing.assert_array_equal(
        snap["left_hand_q"], np.full(NUM_HAND_DOF_PER_SIDE, 0.5),
    )
    np.testing.assert_array_equal(
        snap["right_hand_q"], np.full(NUM_HAND_DOF_PER_SIDE, -0.5),
    )
    assert snap["stream_mode"] == "LOCOMOTION"

    cmds = state.drain_recorder_cmds()
    assert cmds == [("start", 7), ("save", 42)]
    # Drain must consume.
    assert state.drain_recorder_cmds() == []


# ---------------------------------------------------------------------------
# _handle_body_pose_msg (planner-style packed message)
# ---------------------------------------------------------------------------


def test_handle_body_pose_msg_decodes_packed_planner_payload() -> None:
    state = _SubscribeModeState()
    payload = {
        "joint_pos_mj": np.linspace(-0.3, 0.3, NUM_BODY_DOFS, dtype=np.float32),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32,
        ),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "right_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "frame_index": np.array([123], dtype=np.int64),
    }
    msg = pack_pose_message(payload, topic="body_pose", version=4)

    _handle_body_pose_msg([msg], state, expected_topic="body_pose")

    snap = state.snapshot()
    assert snap["body_pose_q_mj"] is not None
    np.testing.assert_allclose(
        snap["body_pose_q_mj"], payload["joint_pos_mj"], rtol=0, atol=0,
    )


def test_handle_body_pose_msg_rejects_wrong_topic_silently() -> None:
    state = _SubscribeModeState()
    payload = {
        "joint_pos_mj": np.zeros(NUM_BODY_DOFS, dtype=np.float32),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32,
        ),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "right_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
    }
    msg = pack_pose_message(payload, topic="other_topic", version=4)
    _handle_body_pose_msg([msg], state, expected_topic="body_pose")
    # Must NOT crash; state stays empty.
    assert state.snapshot()["body_pose_q_mj"] is None


def test_handle_body_pose_msg_rejects_wrong_dof_silently() -> None:
    state = _SubscribeModeState()
    bad_payload = {
        "joint_pos_mj": np.zeros(NUM_BODY_DOFS - 1, dtype=np.float32),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32,
        ),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "right_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
    }
    msg = pack_pose_message(bad_payload, topic="body_pose", version=4)
    _handle_body_pose_msg([msg], state, expected_topic="body_pose")
    # Wrong DOF count -> silent drop.
    assert state.snapshot()["body_pose_q_mj"] is None


# ---------------------------------------------------------------------------
# v5 future-window passthrough: the planner publishes a strictly-future
# trajectory window that the C++ deploy's tokenizer needs to anticipate
# the next 0.9 s of motion. Without it the deploy falls back to its
# legacy single-frame Sample() path and the policy's future tokens are
# pinned at the current pose -- which is exactly what made the robot
# "step in place" when commanded to walk in Phase 0 smoke testing.
# ---------------------------------------------------------------------------


def _v5_planner_payload(
    *,
    n_future: int = 9,
    body_seed: float = 0.1,
    root_quat_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    future_dt_s: float = 0.1,
) -> dict[str, np.ndarray]:
    """Mirror :func:`build_pose_payload`'s output for tests."""
    rng = np.random.default_rng(int(body_seed * 1_000))
    jpos = rng.uniform(-body_seed, body_seed, size=NUM_BODY_DOFS).astype(
        np.float32
    )
    jpos_future = rng.uniform(
        -body_seed, body_seed, size=(n_future, NUM_BODY_DOFS)
    ).astype(np.float32)
    rot = np.array(root_quat_xyzw, dtype=np.float32)
    rot_future = np.tile(rot, (n_future, 1))
    prev = jpos[None, :]
    all_jpos = np.concatenate([prev, jpos_future], axis=0)
    jvel_future = (
        (all_jpos[1:] - all_jpos[:-1]) / max(future_dt_s, 1e-6)
    ).astype(np.float32)
    fidx_future = np.arange(1, n_future + 1, dtype=np.int64)
    return {
        "joint_pos_mj": jpos,
        "root_quat_xyzw": rot,
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "right_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
        "joint_pos_mj_future": jpos_future,
        "root_quat_xyzw_future": rot_future,
        "joint_vel_mj_future": jvel_future,
        "frame_index_future": fidx_future,
        "future_dt_s": np.array([future_dt_s], dtype=np.float32),
    }


def test_handle_body_pose_msg_captures_root_quat() -> None:
    state = _SubscribeModeState()
    payload = _v5_planner_payload(
        # 30-degree rotation about Z -> non-identity quat the recorder
        # must NOT silently squash to identity.
        root_quat_xyzw=(0.0, 0.0, 0.2588, 0.9659),
    )
    msg = pack_pose_message(payload, topic="body_pose", version=4)
    _handle_body_pose_msg([msg], state, expected_topic="body_pose")

    snap = state.snapshot()
    assert snap["root_quat_xyzw"] is not None
    np.testing.assert_allclose(
        snap["root_quat_xyzw"], payload["root_quat_xyzw"], rtol=0, atol=1e-6,
    )


def test_handle_body_pose_msg_captures_full_v5_future_window() -> None:
    state = _SubscribeModeState()
    payload = _v5_planner_payload(n_future=9)
    msg = pack_pose_message(payload, topic="body_pose", version=4)
    _handle_body_pose_msg([msg], state, expected_topic="body_pose")

    snap = state.snapshot()
    assert snap["joint_pos_mj_future"] is not None
    assert snap["root_quat_xyzw_future"] is not None
    assert snap["joint_vel_mj_future"] is not None
    assert snap["frame_index_future"] is not None
    assert snap["future_dt_s"] == pytest.approx(0.1)
    np.testing.assert_allclose(
        snap["joint_pos_mj_future"], payload["joint_pos_mj_future"],
        rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        snap["root_quat_xyzw_future"], payload["root_quat_xyzw_future"],
        rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        snap["joint_vel_mj_future"], payload["joint_vel_mj_future"],
        rtol=0, atol=1e-6,
    )
    np.testing.assert_array_equal(
        snap["frame_index_future"], payload["frame_index_future"],
    )


def test_handle_body_pose_msg_v4_payload_leaves_future_none() -> None:
    """A v4 publisher (no future fields) must still be accepted; the
    snapshot's future slots stay None so the recorder's publish path
    falls back to the v4 wire and the deploy uses single-frame Sample().
    """
    state = _SubscribeModeState()
    payload = {
        "joint_pos_mj": np.zeros(NUM_BODY_DOFS, dtype=np.float32),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32,
        ),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "right_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
    }
    msg = pack_pose_message(payload, topic="body_pose", version=4)
    _handle_body_pose_msg([msg], state, expected_topic="body_pose")

    snap = state.snapshot()
    assert snap["body_pose_q_mj"] is not None
    assert snap["root_quat_xyzw"] is not None
    assert snap["joint_pos_mj_future"] is None
    assert snap["root_quat_xyzw_future"] is None
    assert snap["joint_vel_mj_future"] is None
    assert snap["frame_index_future"] is None
    assert snap["future_dt_s"] is None


def test_handle_body_pose_msg_partial_window_drops_future() -> None:
    """A planner that emits ``joint_pos_mj_future`` but forgets
    ``root_quat_xyzw_future`` must NOT poison the snapshot with a
    half-formed window: the C++ deploy requires both arrays to
    promote into has_future_window_, so we discard partial windows
    here too instead of silently shipping inconsistent data."""
    state = _SubscribeModeState()
    payload = {
        "joint_pos_mj": np.zeros(NUM_BODY_DOFS, dtype=np.float32),
        "root_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "right_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
        # joint_pos future without rotation future -> partial.
        "joint_pos_mj_future": np.zeros((9, NUM_BODY_DOFS), dtype=np.float32),
    }
    msg = pack_pose_message(payload, topic="body_pose", version=4)
    _handle_body_pose_msg([msg], state, expected_topic="body_pose")

    snap = state.snapshot()
    assert snap["body_pose_q_mj"] is not None
    assert snap["joint_pos_mj_future"] is None
    assert snap["root_quat_xyzw_future"] is None


# ---------------------------------------------------------------------------
# _handle_arm_and_hands_msg (manager multipart messages)
# ---------------------------------------------------------------------------


def _msgpack(d: dict) -> bytes:
    return msgpack.packb(d, use_bin_type=True)


def test_handle_arm_and_hands_msg_arm_targets() -> None:
    state = _SubscribeModeState()
    left = np.linspace(-1, 1, 7, dtype=np.float32)
    right = np.linspace(1, -1, 7, dtype=np.float32)
    parts = [
        b"arm_targets",
        _msgpack({
            "left_q_rad": left.tolist(),
            "right_q_rad": right.tolist(),
            "is_engaged": True,
            "tick": 99,
            "ts": 1.234,
        }),
    ]
    _handle_arm_and_hands_msg(
        parts, state,
        arm_targets_topic="arm_targets",
        hand_finger_cmd_topic="hand_finger_cmd",
        stream_mode_topic="stream_mode",
        recorder_cmd_topic="recorder_cmd",
    )
    snap = state.snapshot()
    np.testing.assert_allclose(
        snap["arm_left_q"], left, rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        snap["arm_right_q"], right, rtol=0, atol=1e-6,
    )
    assert snap["arm_engaged"] is True


def test_handle_arm_and_hands_msg_hand_finger_cmd() -> None:
    state = _SubscribeModeState()
    left = np.linspace(0, 1, NUM_HAND_DOF_PER_SIDE, dtype=np.float32)
    right = np.linspace(1, 0, NUM_HAND_DOF_PER_SIDE, dtype=np.float32)
    parts = [
        b"hand_finger_cmd",
        _msgpack({
            "left_hand_q": left.tolist(),
            "right_hand_q": right.tolist(),
            "tick": 1,
            "ts": 0.0,
        }),
    ]
    _handle_arm_and_hands_msg(
        parts, state,
        arm_targets_topic="arm_targets",
        hand_finger_cmd_topic="hand_finger_cmd",
        stream_mode_topic="stream_mode",
        recorder_cmd_topic="recorder_cmd",
    )
    snap = state.snapshot()
    np.testing.assert_allclose(
        snap["left_hand_q"], left, rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        snap["right_hand_q"], right, rtol=0, atol=1e-6,
    )


def test_handle_arm_and_hands_msg_stream_mode() -> None:
    state = _SubscribeModeState()
    parts = [
        b"stream_mode",
        _msgpack({"mode": "ARM_MANIPULATION", "tick": 0, "ts": 0.0}),
    ]
    _handle_arm_and_hands_msg(
        parts, state,
        arm_targets_topic="arm_targets",
        hand_finger_cmd_topic="hand_finger_cmd",
        stream_mode_topic="stream_mode",
        recorder_cmd_topic="recorder_cmd",
    )
    assert state.snapshot()["stream_mode"] == "ARM_MANIPULATION"


def test_handle_arm_and_hands_msg_recorder_cmd_json() -> None:
    state = _SubscribeModeState()
    parts = [
        b"recorder_cmd",
        json.dumps({"action": "start", "tick": 42}).encode("utf-8"),
    ]
    _handle_arm_and_hands_msg(
        parts, state,
        arm_targets_topic="arm_targets",
        hand_finger_cmd_topic="hand_finger_cmd",
        stream_mode_topic="stream_mode",
        recorder_cmd_topic="recorder_cmd",
    )
    assert state.drain_recorder_cmds() == [("start", 42)]


def test_handle_arm_and_hands_msg_recorder_cmd_malformed_silent() -> None:
    state = _SubscribeModeState()
    # Missing "action" key.
    parts = [
        b"recorder_cmd",
        json.dumps({"tick": 0}).encode("utf-8"),
    ]
    _handle_arm_and_hands_msg(
        parts, state,
        arm_targets_topic="arm_targets",
        hand_finger_cmd_topic="hand_finger_cmd",
        stream_mode_topic="stream_mode",
        recorder_cmd_topic="recorder_cmd",
    )
    assert state.drain_recorder_cmds() == []

    # Not JSON at all.
    parts = [b"recorder_cmd", b"\x80\x81not json"]
    _handle_arm_and_hands_msg(
        parts, state,
        arm_targets_topic="arm_targets",
        hand_finger_cmd_topic="hand_finger_cmd",
        stream_mode_topic="stream_mode",
        recorder_cmd_topic="recorder_cmd",
    )
    assert state.drain_recorder_cmds() == []


# ---------------------------------------------------------------------------
# Full _subscribe_mode_thread end-to-end
# ---------------------------------------------------------------------------


@pytest.fixture
def planner_pub():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{_PLANNER_PORT}")
    yield sock
    sock.close(linger=0)


@pytest.fixture
def manager_pub():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://127.0.0.1:{_MGR_PORT}")
    yield sock
    sock.close(linger=0)


def test_subscribe_mode_thread_end_to_end(planner_pub, manager_pub) -> None:
    """Boot the real subscribe thread, publish from both sides, expect
    the state to reflect every wire we publish."""
    state = _SubscribeModeState()
    stop = threading.Event()
    thread = threading.Thread(
        target=_subscribe_mode_thread,
        kwargs=dict(
            body_pose_url=f"tcp://127.0.0.1:{_PLANNER_PORT}",
            body_pose_topic="body_pose",
            arm_and_hands_url=f"tcp://127.0.0.1:{_MGR_PORT}",
            arm_targets_topic="arm_targets",
            hand_finger_cmd_topic="hand_finger_cmd",
            stream_mode_topic="stream_mode",
            recorder_cmd_topic="recorder_cmd",
            state=state,
            stop_event=stop,
            verbose=False,
        ),
        name="test-recorder-sub",
        daemon=True,
    )
    thread.start()
    # Allow PUB-SUB handshake.
    time.sleep(0.2)

    body = np.linspace(-0.1, 0.1, NUM_BODY_DOFS, dtype=np.float32)
    body_msg = pack_pose_message(
        {
            "joint_pos_mj": body,
            "root_quat_xyzw": np.array(
                [0.0, 0.0, 0.0, 1.0], dtype=np.float32,
            ),
            "motion_token": np.zeros(64, dtype=np.float32),
            "left_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
            "right_hand_joints": np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float32),
            "frame_index": np.array([0], dtype=np.int64),
        },
        topic="body_pose", version=4,
    )

    left_arm = np.linspace(-1, 1, 7, dtype=np.float32)
    right_arm = np.linspace(1, -1, 7, dtype=np.float32)
    left_hand = np.full(NUM_HAND_DOF_PER_SIDE, 0.3, dtype=np.float32)
    right_hand = np.full(NUM_HAND_DOF_PER_SIDE, 0.7, dtype=np.float32)

    # Publish each topic a few times to defeat slow-joiner drops.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        planner_pub.send(body_msg)
        manager_pub.send_multipart([
            b"arm_targets",
            _msgpack({
                "left_q_rad": left_arm.tolist(),
                "right_q_rad": right_arm.tolist(),
                "is_engaged": True,
                "tick": 0,
                "ts": 0.0,
            }),
        ])
        manager_pub.send_multipart([
            b"hand_finger_cmd",
            _msgpack({
                "left_hand_q": left_hand.tolist(),
                "right_hand_q": right_hand.tolist(),
                "tick": 0,
                "ts": 0.0,
            }),
        ])
        manager_pub.send_multipart([
            b"stream_mode",
            _msgpack({"mode": "LOCOMOTION", "tick": 0, "ts": 0.0}),
        ])
        manager_pub.send_multipart([
            b"recorder_cmd",
            json.dumps({"action": "start", "tick": 5}).encode("utf-8"),
        ])
        time.sleep(0.05)
        snap = state.snapshot()
        if (
            snap["body_pose_q_mj"] is not None
            and snap["arm_left_q"] is not None
            and snap["left_hand_q"] is not None
            and snap["stream_mode"] == "LOCOMOTION"
        ):
            break

    snap = state.snapshot()
    try:
        assert snap["body_pose_q_mj"] is not None, "body_pose never received"
        np.testing.assert_allclose(
            snap["body_pose_q_mj"], body, rtol=0, atol=1e-6,
        )

        assert snap["arm_left_q"] is not None, "arm_targets never received"
        np.testing.assert_allclose(
            snap["arm_left_q"], left_arm, rtol=0, atol=1e-6,
        )
        np.testing.assert_allclose(
            snap["arm_right_q"], right_arm, rtol=0, atol=1e-6,
        )
        assert snap["arm_engaged"] is True

        assert snap["left_hand_q"] is not None, "hand_finger_cmd never received"
        np.testing.assert_allclose(
            snap["left_hand_q"], left_hand, rtol=0, atol=1e-6,
        )
        np.testing.assert_allclose(
            snap["right_hand_q"], right_hand, rtol=0, atol=1e-6,
        )

        assert snap["stream_mode"] == "LOCOMOTION"

        cmds = state.drain_recorder_cmds()
        assert len(cmds) >= 1
        # We may have sent the same start command many times during the
        # warm-up loop above; the state machine de-duplicates nothing,
        # so just assert at least one was queued and they all match.
        assert all(c == ("start", 5) for c in cmds)
    finally:
        stop.set()
        thread.join(timeout=1.0)


# ---------------------------------------------------------------------------
# RecorderConfig validation surfaced in __init__
# ---------------------------------------------------------------------------


def test_recorder_rejects_mixed_sources() -> None:
    from gear_sonic.utils.teleop.x2_dataset_recorder import X2DatasetRecorder
    cfg = RecorderConfig(
        output_dir=None, task="", teleop_only=True,
        body_pose_source="internal", arm_targets_source="zmq",
    )
    with pytest.raises(ValueError, match="Mixing body_pose_source"):
        X2DatasetRecorder(cfg)


def test_recorder_rejects_unknown_body_pose_source() -> None:
    from gear_sonic.utils.teleop.x2_dataset_recorder import X2DatasetRecorder
    cfg = RecorderConfig(
        output_dir=None, task="", teleop_only=True,
        body_pose_source="bogus", arm_targets_source="bogus",
    )
    with pytest.raises(ValueError, match="body_pose_source must be"):
        X2DatasetRecorder(cfg)


def test_recorder_rejects_unknown_arm_targets_source() -> None:
    from gear_sonic.utils.teleop.x2_dataset_recorder import X2DatasetRecorder
    cfg = RecorderConfig(
        output_dir=None, task="", teleop_only=True,
        body_pose_source="internal", arm_targets_source="bogus",
    )
    with pytest.raises(ValueError, match="arm_targets_source must be"):
        X2DatasetRecorder(cfg)


# ---------------------------------------------------------------------------
# _publish_pose v5 wire passthrough: prove the recorder's republish path
# (planner body_pose -> recorder merger -> deploy) carries the future
# window the C++ ZmqPoseInputSource needs to flip has_future_window_.
# ---------------------------------------------------------------------------


from gear_sonic.utils.teleop.x2_dataset_recorder import (  # noqa: E402
    _LEFT_ARM_MJ_SLICE,
    _RIGHT_ARM_MJ_SLICE,
)
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)


# Capture-only stub: stand in for X2DatasetRecorder._publish_pose's only
# external side effect (sending on a ZMQ PUB socket) so we can exercise
# the publish path without booting the full LeRobot writer chain.
class _FakeSock:
    def __init__(self) -> None:
        self.sent: list[bytes] = []

    def send(self, msg: bytes, flags: int = 0) -> None:
        self.sent.append(msg)


class _PubProbe:
    """Minimal shim around X2DatasetRecorder._publish_pose.

    We bind the unbound method to a tiny duck-typed object so we can
    test the publish path without instantiating the full recorder
    (which pulls in datasets/av/lerobot just to run a unit test).
    """

    def __init__(self, *, pub_topic: str = "pose", protocol_version: int = 4) -> None:
        from types import SimpleNamespace
        self._cfg = SimpleNamespace(
            pub_topic=pub_topic, protocol_version=protocol_version,
        )
        self._pub_sock = _FakeSock()

    @property
    def sent(self) -> list[bytes]:
        return self._pub_sock.sent


def _call_publish_pose(probe: _PubProbe, **kwargs) -> None:
    """Invoke X2DatasetRecorder._publish_pose against the probe."""
    from gear_sonic.utils.teleop.x2_dataset_recorder import X2DatasetRecorder
    X2DatasetRecorder._publish_pose(probe, **kwargs)  # type: ignore[arg-type]


def _decode_sent(probe: _PubProbe, *, topic: str = "pose"):
    assert len(probe.sent) == 1, "expected exactly one published frame"
    return unpack_message(probe.sent[0], expected_topic=topic)


def test_publish_pose_v4_compat_omits_future_fields() -> None:
    """Without v5 kwargs, _publish_pose must keep emitting the legacy
    payload (no future fields). This pins the back-compat contract for
    the standalone Quest path that doesn't have a planner subscriber.
    """
    probe = _PubProbe()
    body = np.linspace(-0.2, 0.2, NUM_BODY_DOFS, dtype=np.float64)
    _call_publish_pose(
        probe,
        body_q_mj=body,
        motion_token=np.zeros(64, dtype=np.float32),
        left_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        right_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        tick=7,
    )
    decoded = _decode_sent(probe)
    assert "joint_pos_mj" in decoded.fields
    assert "root_quat_xyzw" in decoded.fields
    np.testing.assert_allclose(
        decoded.fields["root_quat_xyzw"],
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    )
    # No v5 fields when caller didn't supply them: the deploy stays on
    # its single-frame Sample() fallback (which is what we want for
    # the legacy code path).
    assert "joint_pos_mj_future" not in decoded.fields
    assert "root_quat_xyzw_future" not in decoded.fields
    assert "joint_vel_mj_future" not in decoded.fields
    assert "future_dt_s" not in decoded.fields


def test_publish_pose_forwards_root_quat_passthrough() -> None:
    """Heading commanded by the planner (e.g. mid-turn yaw) must flow
    through to the deploy verbatim instead of being clobbered by the
    legacy hardcoded identity quat.
    """
    probe = _PubProbe()
    rq = np.array([0.0, 0.0, 0.2588, 0.9659], dtype=np.float32)  # 30 deg
    _call_publish_pose(
        probe,
        body_q_mj=np.zeros(NUM_BODY_DOFS, dtype=np.float64),
        motion_token=np.zeros(64, dtype=np.float32),
        left_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        right_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        tick=0,
        root_quat_xyzw=rq,
    )
    decoded = _decode_sent(probe)
    np.testing.assert_allclose(
        decoded.fields["root_quat_xyzw"], rq, rtol=0, atol=1e-6,
    )


def test_publish_pose_emits_full_v5_window_when_provided() -> None:
    """When the caller passes both jpos+rot futures, _publish_pose
    must emit the full v5 set (including a recomputed jvel) so the
    deploy's ZmqPoseInputSource flips has_future_window_=true and
    the policy gets a real anticipatory future window.
    """
    probe = _PubProbe()
    n_future = 9
    body = np.linspace(-0.2, 0.2, NUM_BODY_DOFS, dtype=np.float64)
    jpos_future = np.tile(
        np.linspace(-0.1, 0.1, NUM_BODY_DOFS, dtype=np.float32), (n_future, 1),
    ) + np.linspace(0.0, 0.45, n_future, dtype=np.float32)[:, None]
    rot_future = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_future, 1),
    )
    fidx_future = np.arange(1, n_future + 1, dtype=np.int64)
    _call_publish_pose(
        probe,
        body_q_mj=body,
        motion_token=np.zeros(64, dtype=np.float32),
        left_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        right_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        tick=42,
        joint_pos_mj_future=jpos_future,
        root_quat_xyzw_future=rot_future,
        frame_index_future=fidx_future,
        future_dt_s=0.1,
    )
    decoded = _decode_sent(probe)
    np.testing.assert_allclose(
        decoded.fields["joint_pos_mj_future"], jpos_future, rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        decoded.fields["root_quat_xyzw_future"], rot_future, rtol=0, atol=1e-6,
    )
    np.testing.assert_array_equal(
        decoded.fields["frame_index_future"], fidx_future,
    )
    np.testing.assert_allclose(
        decoded.fields["future_dt_s"], np.array([0.1], dtype=np.float32),
    )

    # jvel is recomputed from finite-diff over [current, *future].
    expected_prev = np.concatenate(
        [body[None, :].astype(np.float32), jpos_future], axis=0,
    )
    expected_jvel = (
        (expected_prev[1:] - expected_prev[:-1]) / 0.1
    ).astype(np.float32)
    np.testing.assert_allclose(
        decoded.fields["joint_vel_mj_future"], expected_jvel,
        rtol=1e-5, atol=1e-5,
    )


def test_publish_pose_skips_v5_when_window_shape_invalid() -> None:
    """A jpos_future with the wrong DOF count (e.g. 27 instead of 31)
    must NOT be forwarded -- partial/malformed windows would be
    silently rejected by the C++ deploy too, and shipping them gives
    no benefit over the v4 fallback path.
    """
    probe = _PubProbe()
    body = np.zeros(NUM_BODY_DOFS, dtype=np.float64)
    bad_jpos_future = np.zeros((9, NUM_BODY_DOFS - 4), dtype=np.float32)
    rot_future = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (9, 1),
    )
    _call_publish_pose(
        probe,
        body_q_mj=body,
        motion_token=np.zeros(64, dtype=np.float32),
        left_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        right_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        tick=0,
        joint_pos_mj_future=bad_jpos_future,
        root_quat_xyzw_future=rot_future,
        future_dt_s=0.1,
    )
    decoded = _decode_sent(probe)
    assert "joint_pos_mj_future" not in decoded.fields
    assert "root_quat_xyzw_future" not in decoded.fields
    assert "joint_vel_mj_future" not in decoded.fields


def test_publish_pose_arm_overlay_in_future_window_zeros_arm_jvel() -> None:
    """When the caller pins the same commanded arm pose into all
    future joint_pos_mj_future slots (the recorder's merge semantic),
    the recomputed joint_vel_mj_future must report ZERO velocity for
    the arm DOFs (operator's arm pose is held still through the look-
    ahead) while leg / waist DOFs continue to follow the planner's
    finite-diff. This is the exact wire shape the SONIC tracking
    policy was trained against; without it the future tokens look
    like the arms are about to teleport from the planner's stand
    pose to the operator's commanded pose at every k.
    """
    probe = _PubProbe()
    n_future = 9
    arm_dof = _LEFT_ARM_MJ_SLICE.stop - _LEFT_ARM_MJ_SLICE.start

    # Body: arms held at operator's commanded pose; legs / waist at
    # the planner's current frame. We mimic the recorder's overlay by
    # constructing future arrays where arm slices are PINNED to the
    # operator's commanded value across all 9 future frames, while
    # leg slices ramp forward (e.g. the planner's stride).
    arm_left_cmd = np.full(arm_dof, 0.5, dtype=np.float32)
    arm_right_cmd = np.full(arm_dof, -0.3, dtype=np.float32)

    body = np.zeros(NUM_BODY_DOFS, dtype=np.float64)
    body[_LEFT_ARM_MJ_SLICE] = arm_left_cmd
    body[_RIGHT_ARM_MJ_SLICE] = arm_right_cmd

    jpos_future = np.zeros((n_future, NUM_BODY_DOFS), dtype=np.float32)
    # Ramp the leg DOFs (slice 0..15) so we get non-zero leg jvel.
    jpos_future[:, :15] = np.linspace(0.0, 0.45, n_future, dtype=np.float32)[:, None]
    # Pin both arm slices to the operator's commanded pose -- this is
    # what the recorder's merge loop does after lifting the planner's
    # body_pose future window into the snapshot.
    jpos_future[:, _LEFT_ARM_MJ_SLICE] = arm_left_cmd
    jpos_future[:, _RIGHT_ARM_MJ_SLICE] = arm_right_cmd
    rot_future = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (n_future, 1),
    )

    _call_publish_pose(
        probe,
        body_q_mj=body,
        motion_token=np.zeros(64, dtype=np.float32),
        left_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        right_hand_q=np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64),
        tick=0,
        joint_pos_mj_future=jpos_future,
        root_quat_xyzw_future=rot_future,
        future_dt_s=0.1,
    )
    decoded = _decode_sent(probe)
    jvel = decoded.fields["joint_vel_mj_future"]
    # Arm slices: zero velocity throughout (operator pinned).
    np.testing.assert_allclose(
        jvel[:, _LEFT_ARM_MJ_SLICE],
        np.zeros((n_future, arm_dof), dtype=np.float32),
        rtol=0, atol=1e-6,
    )
    np.testing.assert_allclose(
        jvel[:, _RIGHT_ARM_MJ_SLICE],
        np.zeros((n_future, arm_dof), dtype=np.float32),
        rtol=0, atol=1e-6,
    )
    # Leg DOFs: non-zero velocity (planner stride).
    assert np.any(np.abs(jvel[:, :15]) > 1e-3), (
        "expected non-zero leg jvel from the planner's stride; got "
        f"max={float(np.max(np.abs(jvel[:, :15]))):.3e}"
    )
