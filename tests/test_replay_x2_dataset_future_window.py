"""Tests for the v5 future-window plumbing in ``replay_x2_dataset``.

These tests pin down the lean fix that makes the dataset replayer move the
robot's body (not just the OmniHand fingers) by promoting each wire frame
to the deploy's v5 mode. We exercise the pure helpers here -- no ZMQ
socket, no parquet on disk, no deploy -- so the tests stay fast and run
in CI without the SONIC checkpoint.

The deploy's v5 promotion contract is documented at the top of
:mod:`gear_sonic.scripts.replay_x2_dataset` and mirrored from
:mod:`gear_sonic.scripts.live_vla_publish_motion_token`. The short
version: presence of ``joint_pos_mj_future`` + ``root_quat_xyzw_future``
is what flips the deploy from "tokenize the trained idle stand pose" to
"tokenize the wire trajectory" each tick. Without that, the C++ deploy
ignores the wire's ``motion_token`` and the body holds idle_stand --
the exact bug this script previously hit.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.scripts.replay_x2_dataset import (
    DEFAULT_PROTOCOL_VERSION,
    FUTURE_DT_S,
    IDENTITY_QUAT_XYZW,
    NUM_BODY_DOFS,
    NUM_FUTURE_SLOTS,
    NUM_HAND_DOF_PER_SIDE,
    NUM_MOTION_TOKEN,
    _build_future_window,
    _build_payload,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _ramp_body_q(n_frames: int) -> np.ndarray:
    """``body_q[i, j] = i`` -- trivially-verifiable future-window indices."""
    return np.broadcast_to(
        np.arange(n_frames, dtype=np.float32)[:, None],
        (n_frames, NUM_BODY_DOFS),
    ).astype(np.float32, copy=True)


def _episode_arrays(n_frames: int) -> tuple[np.ndarray, ...]:
    """Synthetic (body_q, token, left_q, right_q) bundle."""
    body_q = _ramp_body_q(n_frames)
    token = np.zeros((n_frames, NUM_MOTION_TOKEN), dtype=np.float32)
    left_q = np.zeros((n_frames, NUM_HAND_DOF_PER_SIDE), dtype=np.float32)
    right_q = np.zeros((n_frames, NUM_HAND_DOF_PER_SIDE), dtype=np.float32)
    return body_q, token, left_q, right_q


# ---------------------------------------------------------------------------
# _build_future_window
# ---------------------------------------------------------------------------


def test_future_window_basic_indexing() -> None:
    """At f=0, step=5 the 9 slots are body_q[5, 10, ..., 45]."""
    body_q = _ramp_body_q(100)
    jpos, jvel = _build_future_window(body_q, f=0, step=5)

    assert jpos.shape == (NUM_FUTURE_SLOTS, NUM_BODY_DOFS)
    assert jvel.shape == (NUM_FUTURE_SLOTS, NUM_BODY_DOFS)
    assert jpos.dtype == np.float32
    assert jvel.dtype == np.float32

    expected_first_col = np.arange(5, 5 * (NUM_FUTURE_SLOTS + 1), 5,
                                   dtype=np.float32)
    np.testing.assert_array_equal(jpos[:, 0], expected_first_col)
    # All 31 DOFs share the ramp, so every column matches the first one.
    for col in range(1, NUM_BODY_DOFS):
        np.testing.assert_array_equal(jpos[:, col], expected_first_col)


def test_future_window_basic_indexing_mid_episode() -> None:
    """At f=20, step=5 the 9 slots are body_q[25, 30, ..., 65]."""
    body_q = _ramp_body_q(100)
    jpos, _ = _build_future_window(body_q, f=20, step=5)
    expected = np.arange(25, 25 + 5 * NUM_FUTURE_SLOTS, 5, dtype=np.float32)
    np.testing.assert_array_equal(jpos[:, 0], expected)


def test_future_window_tail_tiles_past_episode_end() -> None:
    """Indices past ``n_frames`` clamp to the last frame (n_frames-1)."""
    n_frames = 12
    body_q = _ramp_body_q(n_frames)
    last = float(n_frames - 1)

    # f=10, step=5 -> raw indices [15, 20, 25, ..., 55]; all past end -> all 11.
    jpos, _ = _build_future_window(body_q, f=10, step=5)
    np.testing.assert_array_equal(
        jpos[:, 0], np.full(NUM_FUTURE_SLOTS, last, dtype=np.float32)
    )

    # f=5, step=2 -> raw indices [7, 9, 11, 13, 15, 17, 19, 21, 23];
    # frames >= 12 should clamp to 11.
    jpos, _ = _build_future_window(body_q, f=5, step=2)
    expected = np.array([7, 9, 11, 11, 11, 11, 11, 11, 11], dtype=np.float32)
    np.testing.assert_array_equal(jpos[:, 0], expected)


def test_future_window_velocity_finite_diff_against_current_frame() -> None:
    """Slot-0 velocity diffs against ``body_q[f]``, not the slot itself."""
    body_q = _ramp_body_q(100)
    f = 7
    step = 5
    jpos, jvel = _build_future_window(body_q, f=f, step=step)

    # body_q values are integers; slot k = f + (k+1)*step.
    # slot-0 vel = (body_q[f+step] - body_q[f]) / dt = step / dt.
    # slot-k vel for k>=1 = (jpos[k] - jpos[k-1]) / dt = step / dt.
    expected_per_slot = step / FUTURE_DT_S
    np.testing.assert_allclose(
        jvel, np.full_like(jvel, expected_per_slot), rtol=0, atol=1e-5,
    )


def test_future_window_velocity_zero_when_tail_tiled() -> None:
    """Tail-tiled slots all hold the last frame, so their inter-slot vel is 0."""
    body_q = _ramp_body_q(12)
    jpos, jvel = _build_future_window(body_q, f=10, step=5)
    last = float(11)
    # All slots are last frame, so all jpos rows are equal.
    np.testing.assert_array_equal(jpos, np.full_like(jpos, last))
    # slot-0 vel = (last - body_q[10]) / dt = (11 - 10) / 0.1 = 10.
    np.testing.assert_allclose(jvel[0], np.full(NUM_BODY_DOFS, 10.0), atol=1e-5)
    # slot-1..8 vel = 0 (no change between tail-tiled slots).
    np.testing.assert_allclose(
        jvel[1:], np.zeros((NUM_FUTURE_SLOTS - 1, NUM_BODY_DOFS)), atol=1e-5,
    )


def test_future_window_step_zero_raises() -> None:
    body_q = _ramp_body_q(20)
    with pytest.raises(ValueError, match="step must be >=1"):
        _build_future_window(body_q, f=0, step=0)


# ---------------------------------------------------------------------------
# _build_payload
# ---------------------------------------------------------------------------


_EXPECTED_PAYLOAD_FIELDS = {
    "joint_pos_mj": (np.float32, (NUM_BODY_DOFS,)),
    "root_quat_xyzw": (np.float32, (4,)),
    "motion_token": (np.float32, (NUM_MOTION_TOKEN,)),
    "left_hand_joints": (np.float32, (NUM_HAND_DOF_PER_SIDE,)),
    "right_hand_joints": (np.float32, (NUM_HAND_DOF_PER_SIDE,)),
    "frame_index": (np.int64, (1,)),
    "joint_pos_mj_future": (np.float32, (NUM_FUTURE_SLOTS, NUM_BODY_DOFS)),
    "root_quat_xyzw_future": (np.float32, (NUM_FUTURE_SLOTS, 4)),
    "joint_vel_mj_future": (np.float32, (NUM_FUTURE_SLOTS, NUM_BODY_DOFS)),
    "frame_index_future": (np.int64, (NUM_FUTURE_SLOTS,)),
    "future_dt_s": (np.float32, (1,)),
}


def test_build_payload_schema_matches_v5_contract() -> None:
    """Every v5 promotion field is present with the deploy-expected dtype+shape."""
    body_q, token, left_q, right_q = _episode_arrays(100)
    payload = _build_payload(
        body_q, token, left_q, right_q,
        f=10, wire_frame=42, future_step=5,
    )

    assert set(payload.keys()) == set(_EXPECTED_PAYLOAD_FIELDS.keys()), (
        f"Unexpected payload schema. Diff: "
        f"missing={set(_EXPECTED_PAYLOAD_FIELDS) - set(payload)} "
        f"extra={set(payload) - set(_EXPECTED_PAYLOAD_FIELDS)}"
    )
    for name, (expected_dtype, expected_shape) in _EXPECTED_PAYLOAD_FIELDS.items():
        arr = payload[name]
        assert isinstance(arr, np.ndarray), f"{name} is not an ndarray"
        assert arr.dtype == expected_dtype, (
            f"{name} dtype mismatch: got {arr.dtype}, want {expected_dtype}"
        )
        assert arr.shape == expected_shape, (
            f"{name} shape mismatch: got {arr.shape}, want {expected_shape}"
        )


def test_build_payload_current_frame_sourced_from_f() -> None:
    body_q, token, left_q, right_q = _episode_arrays(100)
    payload = _build_payload(
        body_q, token, left_q, right_q,
        f=17, wire_frame=0, future_step=5,
    )
    np.testing.assert_array_equal(payload["joint_pos_mj"], body_q[17])


def test_build_payload_clamps_f_past_episode_end() -> None:
    body_q, token, left_q, right_q = _episode_arrays(20)
    payload = _build_payload(
        body_q, token, left_q, right_q,
        f=999, wire_frame=0, future_step=5,
    )
    np.testing.assert_array_equal(payload["joint_pos_mj"], body_q[-1])
    # Future window stays valid -- tail-tiled to last frame.
    np.testing.assert_array_equal(
        payload["joint_pos_mj_future"],
        np.broadcast_to(body_q[-1], (NUM_FUTURE_SLOTS, NUM_BODY_DOFS)),
    )


def test_build_payload_wire_frame_indexing() -> None:
    """``frame_index_future = wire_frame + [1..9]`` (monotonic across run)."""
    body_q, token, left_q, right_q = _episode_arrays(100)
    wire_frame = 1234
    payload = _build_payload(
        body_q, token, left_q, right_q,
        f=0, wire_frame=wire_frame, future_step=5,
    )
    np.testing.assert_array_equal(payload["frame_index"],
                                  np.array([wire_frame], dtype=np.int64))
    np.testing.assert_array_equal(
        payload["frame_index_future"],
        wire_frame + np.arange(1, NUM_FUTURE_SLOTS + 1, dtype=np.int64),
    )


def test_build_payload_root_quat_identity_for_current_and_future() -> None:
    """Replay has no recorded base orientation; both wire fields are identity."""
    body_q, token, left_q, right_q = _episode_arrays(50)
    payload = _build_payload(
        body_q, token, left_q, right_q,
        f=3, wire_frame=0, future_step=5,
    )
    np.testing.assert_array_equal(payload["root_quat_xyzw"], IDENTITY_QUAT_XYZW)
    np.testing.assert_array_equal(
        payload["root_quat_xyzw_future"],
        np.broadcast_to(IDENTITY_QUAT_XYZW, (NUM_FUTURE_SLOTS, 4)),
    )


def test_build_payload_future_dt_matches_module_constant() -> None:
    body_q, token, left_q, right_q = _episode_arrays(50)
    payload = _build_payload(
        body_q, token, left_q, right_q,
        f=0, wire_frame=0, future_step=5,
    )
    np.testing.assert_array_equal(
        payload["future_dt_s"], np.array([FUTURE_DT_S], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# Wire roundtrip -- proves the payload is parseable by the deploy decoder
# ---------------------------------------------------------------------------


def test_payload_packs_and_unpacks_byte_for_byte() -> None:
    """``pack_pose_message`` + ``unpack_message`` roundtrips all v5 fields.

    This is the single most important test in the file: if the deploy's
    decoder can read what we pack, the deploy will promote the frame to
    v5 mode and the body will track the trajectory.
    """
    from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (
        unpack_message,
    )
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
        pack_pose_message,
    )

    body_q, token, left_q, right_q = _episode_arrays(100)
    # Fill arrays with non-trivial values so any swap/transpose bug surfaces.
    rng = np.random.default_rng(seed=42)
    body_q = rng.standard_normal(body_q.shape).astype(np.float32)
    token = rng.standard_normal(token.shape).astype(np.float32)
    left_q = rng.standard_normal(left_q.shape).astype(np.float32)
    right_q = rng.standard_normal(right_q.shape).astype(np.float32)

    payload = _build_payload(
        body_q, token, left_q, right_q,
        f=42, wire_frame=99, future_step=5,
    )
    msg = pack_pose_message(payload, topic="pose",
                            version=DEFAULT_PROTOCOL_VERSION)
    decoded = unpack_message(msg, expected_topic="pose")

    assert decoded.version == DEFAULT_PROTOCOL_VERSION
    assert decoded.topic == "pose"
    assert set(decoded.fields.keys()) == set(_EXPECTED_PAYLOAD_FIELDS.keys())
    for name, expected in payload.items():
        got = decoded.fields[name]
        assert got.dtype == expected.dtype, (
            f"{name} dtype lost in roundtrip: {got.dtype} vs {expected.dtype}"
        )
        assert got.shape == expected.shape, (
            f"{name} shape lost in roundtrip: {got.shape} vs {expected.shape}"
        )
        np.testing.assert_array_equal(
            got, expected, err_msg=f"{name} bytes drifted across pack/unpack",
        )
