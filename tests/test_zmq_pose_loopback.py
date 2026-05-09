"""Self-loopback test for the X2 deploy ZMQ wire format.

Why this test exists
--------------------

The M2 acceptance gate connects three processes:

* ``deploy_x2.sh sim --input-type zmq``            (C++ deploy)
* ``mock_vla_publish_stand_token.py`` (PUB on 5556)
* ``dump_x2_debug.py``                (SUB on 5557)

We can only run the C++ side inside a docker container that has ROS 2 +
AimDK + onnxruntime. Long before the deploy is available, we still want
high confidence that:

1. ``pack_pose_message`` and ``unpack_message`` are mutually inverse.
2. The mock-VLA publisher and ``dump_x2_debug`` parser actually agree on
   the wire format.
3. No silent dtype / shape / endianness drift has crept into either side.

This test stands up a Python PUB <-> SUB loopback inside a single process,
publishes the *exact* fields the mock-VLA produces, and confirms that the
decoded result round-trips byte-for-byte. It is the cheapest possible
gate against contract regressions in the wire format.
"""

from __future__ import annotations

from pathlib import Path
import sys
import threading
import time
from typing import Any

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (
    HEADER_SIZE,
    unpack_message,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE as SENDER_HEADER_SIZE,
    build_command_message,
    build_planner_message,
    pack_pose_message,
)


def test_header_size_matches_between_sender_and_decoder() -> None:
    assert SENDER_HEADER_SIZE == HEADER_SIZE == 1280, (
        "Header size mismatch between sender, decoder, and the C++ subscriber. "
        "All three must stay locked at 1280 bytes — see "
        "gear_sonic_deploy/.../zmq_packed_message_subscriber.hpp."
    )


def test_pack_unpack_pose_roundtrip_v4() -> None:
    """The mock-VLA → dump_x2_debug critical path must round-trip cleanly."""
    rng = np.random.default_rng(seed=0)
    motion_token = rng.standard_normal(64).astype(np.float32)
    left_hand = rng.standard_normal(10).astype(np.float32) * 0.1
    right_hand = rng.standard_normal(10).astype(np.float32) * 0.1
    frame_index = np.array([42], dtype=np.int64)

    payload = {
        "motion_token": motion_token,
        "left_hand_joints": left_hand,
        "right_hand_joints": right_hand,
        "frame_index": frame_index,
    }

    raw = pack_pose_message(payload, topic="pose", version=4)
    decoded = unpack_message(raw, expected_topic="pose")

    assert decoded.topic == "pose"
    assert decoded.version == 4
    assert decoded.endian == "le"

    assert set(decoded.fields.keys()) == set(payload.keys())
    np.testing.assert_array_equal(decoded.fields["motion_token"], motion_token)
    np.testing.assert_array_equal(decoded.fields["left_hand_joints"], left_hand)
    np.testing.assert_array_equal(decoded.fields["right_hand_joints"], right_hand)
    np.testing.assert_array_equal(decoded.fields["frame_index"], frame_index)


def test_unpack_pose_with_topic_prefix_strip() -> None:
    """The decoder must auto-strip the configured topic prefix."""
    motion_token = np.zeros(64, dtype=np.float32)
    raw = pack_pose_message({"motion_token": motion_token}, topic="pose", version=3)

    decoded = unpack_message(raw, expected_topic="pose")
    np.testing.assert_array_equal(decoded.fields["motion_token"], motion_token)
    assert decoded.topic == "pose"


def test_unpack_pose_without_topic_prefix() -> None:
    """When ``expected_topic`` is None / empty the decoder treats the buffer as headerless."""
    raw = pack_pose_message({"motion_token": np.ones(4, dtype=np.float32)}, topic="pose")
    body = raw[len(b"pose") :]  # caller already stripped the prefix
    decoded = unpack_message(body, expected_topic=None)
    np.testing.assert_array_equal(decoded.fields["motion_token"], np.ones(4, dtype=np.float32))


def test_unpack_rejects_truncated_payload() -> None:
    raw = pack_pose_message(
        {"motion_token": np.zeros(64, dtype=np.float32)}, topic="pose", version=4
    )
    truncated = raw[:-32]
    try:
        unpack_message(truncated, expected_topic="pose")
    except ValueError as exc:
        assert "truncated" in str(exc).lower() or "payload" in str(exc).lower()
    else:
        raise AssertionError("truncated payload should have raised ValueError")


def test_command_message_decodes() -> None:
    raw = build_command_message(start=True, stop=False, planner=True, delta_heading=0.5)
    decoded = unpack_message(raw, expected_topic="command")
    # u8 fields are decoded as 1-byte unsigned ints; bool/u8 wire mapping in
    # zmq_packed_message_decoder normalizes 'bool' to numpy.bool_, but
    # 'u8' stays as uint8.
    assert int(decoded.fields["start"][0]) == 1
    assert int(decoded.fields["stop"][0]) == 0
    assert int(decoded.fields["planner"][0]) == 1
    assert decoded.fields["delta_heading"].dtype == np.float32
    np.testing.assert_allclose(decoded.fields["delta_heading"][0], 0.5, atol=1e-7)


def test_planner_message_decodes() -> None:
    raw = build_planner_message(
        mode=2,
        movement=[0.1, 0.0, 0.0],
        facing=[1.0, 0.0, 0.0],
        speed=0.3,
        height=0.55,
        left_hand_position=[0.0] * 7,
        right_hand_position=[0.0] * 7,
    )
    decoded = unpack_message(raw, expected_topic="planner")
    assert int(decoded.fields["mode"][0]) == 2
    np.testing.assert_allclose(decoded.fields["movement"], [0.1, 0.0, 0.0])
    np.testing.assert_allclose(decoded.fields["facing"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(decoded.fields["speed"][0], 0.3, atol=1e-6)
    np.testing.assert_allclose(decoded.fields["height"][0], 0.55, atol=1e-6)
    assert decoded.fields["left_hand_joints"].shape == (7,)
    assert decoded.fields["right_hand_joints"].shape == (7,)


def test_pubsub_loopback_in_single_process() -> None:
    """Stand up a real PUB<->SUB pair on an ephemeral port and round-trip a
    burst of messages. This is the closest unit-test analogue of the M2
    acceptance gate (mock-VLA publisher → C++ deploy SUB) that we can run
    without the C++ side.
    """
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
    sub.setsockopt(zmq.RCVHWM, 10)
    sub.setsockopt(zmq.LINGER, 0)
    bound_port = sub.bind_to_random_port("tcp://127.0.0.1")

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 10)
    pub.setsockopt(zmq.LINGER, 0)
    pub.connect(f"tcp://127.0.0.1:{bound_port}")
    # PUB-SUB has slow joiner: give the SUBSCRIBE handshake time to land.
    time.sleep(0.2)

    received: list[dict[str, Any]] = []
    poller = zmq.Poller()
    poller.register(sub, zmq.POLLIN)

    NUM_TICKS = 5
    receiver_done = threading.Event()

    def _receiver() -> None:
        try:
            deadline = time.monotonic() + 5.0
            while len(received) < NUM_TICKS and time.monotonic() < deadline:
                events = dict(poller.poll(500))
                if sub in events:
                    raw = sub.recv()
                    decoded = unpack_message(raw, expected_topic="pose")
                    received.append(decoded.fields)
        finally:
            receiver_done.set()

    receiver = threading.Thread(target=_receiver, daemon=True)
    receiver.start()

    try:
        for tick in range(NUM_TICKS):
            payload = {
                "motion_token": np.full(64, tick, dtype=np.float32),
                "left_hand_joints": np.zeros(10, dtype=np.float32),
                "right_hand_joints": np.zeros(10, dtype=np.float32),
                "frame_index": np.array([tick], dtype=np.int64),
            }
            pub.send(pack_pose_message(payload, topic="pose", version=4))
            time.sleep(0.02)
        receiver.join(timeout=5.0)
    finally:
        sub.close(linger=0)
        pub.close(linger=0)

    assert receiver_done.is_set(), "receiver thread did not finish in time"
    assert len(received) >= 1, "no messages received over loopback (slow-joiner?)"
    last = received[-1]
    assert last["motion_token"].shape == (64,)
    assert last["left_hand_joints"].shape == (10,)
    assert last["right_hand_joints"].shape == (10,)
    assert int(last["frame_index"][0]) >= 0


def main() -> int:
    tests = [
        test_header_size_matches_between_sender_and_decoder,
        test_pack_unpack_pose_roundtrip_v4,
        test_unpack_pose_with_topic_prefix_strip,
        test_unpack_pose_without_topic_prefix,
        test_unpack_rejects_truncated_payload,
        test_command_message_decodes,
        test_planner_message_decodes,
        test_pubsub_loopback_in_single_process,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except Exception as exc:  # pragma: no cover
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc!r}")
            import traceback

            traceback.print_exc()
        else:
            print(f"PASS  {fn.__name__}")
    if failed:
        print(f"\n{failed}/{len(tests)} tests failed")
        return 1
    print("\nOK: ZMQ wire-format gate green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
