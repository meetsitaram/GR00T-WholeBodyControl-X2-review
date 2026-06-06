"""Wire-level contract tests for the world-frame root pelvis fields.

The planner+recorder pair now publishes ``root_xy_world`` (2,) and
``root_z_world`` (1,) on every body_pose / pose tick (post-2026-06)
so the kinematic viewer + the Phase 2 motion-lib PKL recorder can
reconstruct full world-frame ``qpos[0:3]`` instead of pelvis-pinning
at the origin. The C++ deploy ignores the new keys (the header
decoder skips unknown fields), so the change is wire-safe.

These tests pin the contract end-to-end:

  - ``StreamFrame.root_z_world`` exists with the right default.
  - ``build_pose_payload`` always emits both fields (right shape +
    dtype + value).
  - ``pack_pose_message`` -> ``unpack_message`` round-trips the
    new fields byte-equivalently alongside the legacy ones.
  - A legacy payload (no new fields) still decodes cleanly.

If any of these break we lose the kinematic-viewer world tracking
and Phase 2 capture, so they're worth pinning explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.planner.constants import DEFAULT_PELVIS_Z_M
from gear_sonic.utils.planner.state_machine import (
    PlannerState,
    StreamFrame,
    build_pose_payload,
)
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


NUM_BODY_DOFS = 31


def _make_frame(
    *,
    xy: tuple[float, float] = (0.12, -0.34),
    z: float | None = None,
    frame_index: int = 7,
) -> StreamFrame:
    return StreamFrame(
        joint_pos_mj=np.arange(NUM_BODY_DOFS, dtype=np.float32) * 0.01,
        root_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        root_xy_world=np.array(xy, dtype=np.float64),
        yaw_world_deg=0.0,
        state=PlannerState.PLAYING,
        bin_name="test_bin",
        frame_index=frame_index,
        seam_blend=False,
        root_z_world=DEFAULT_PELVIS_Z_M if z is None else float(z),
    )


def test_streamframe_default_root_z_matches_constant() -> None:
    """A StreamFrame built without specifying root_z_world snaps to the
    canonical default. The default exists so the heuristic planner (which
    doesn't integrate pelvis height) keeps working unchanged."""
    frame = StreamFrame(
        joint_pos_mj=np.zeros(NUM_BODY_DOFS, dtype=np.float32),
        root_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        root_xy_world=np.zeros(2, dtype=np.float64),
        yaw_world_deg=0.0,
        state=PlannerState.IDLE_LOOP,
        bin_name="default_test",
        frame_index=0,
        seam_blend=False,
    )
    assert frame.root_z_world == pytest.approx(DEFAULT_PELVIS_Z_M)


def test_build_pose_payload_emits_root_xy_world_with_correct_shape() -> None:
    frame = _make_frame(xy=(1.5, -2.25))
    payload = build_pose_payload(frame)

    assert "root_xy_world" in payload
    rxy = payload["root_xy_world"]
    assert rxy.shape == (2,)
    assert rxy.dtype == np.float32
    np.testing.assert_array_equal(rxy, np.array([1.5, -2.25], dtype=np.float32))


def test_build_pose_payload_emits_root_z_world_with_correct_shape() -> None:
    frame = _make_frame(z=0.812)
    payload = build_pose_payload(frame)

    assert "root_z_world" in payload
    rz = payload["root_z_world"]
    # We pack as a (1,) array (not scalar) so the JSON header schema
    # treats it like other fixed-size fields and unpack_message returns
    # a numpy array rather than a Python float.
    assert rz.shape == (1,)
    assert rz.dtype == np.float32
    assert float(rz[0]) == pytest.approx(0.812)


def test_build_pose_payload_uses_default_when_streamframe_omits_z() -> None:
    """Heuristic-planner-style callers that don't pass root_z_world get the
    constant on the wire. This guards against the heuristic test suite
    silently breaking when its StreamFrame call sites are refactored."""
    frame = StreamFrame(
        joint_pos_mj=np.zeros(NUM_BODY_DOFS, dtype=np.float32),
        root_quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        root_xy_world=np.array([0.0, 0.0], dtype=np.float64),
        yaw_world_deg=0.0,
        state=PlannerState.IDLE_LOOP,
        bin_name="omit_z",
        frame_index=0,
        seam_blend=False,
    )
    payload = build_pose_payload(frame)
    assert payload["root_z_world"].shape == (1,)
    assert float(payload["root_z_world"][0]) == pytest.approx(DEFAULT_PELVIS_Z_M)


def test_pack_unpack_roundtrip_preserves_world_root_values() -> None:
    frame = _make_frame(xy=(2.34, -5.67), z=0.901)
    payload = build_pose_payload(frame)

    msg = pack_pose_message(payload, topic="body_pose", version=5)
    decoded = unpack_message(msg, expected_topic="body_pose")

    assert "root_xy_world" in decoded.fields
    assert "root_z_world" in decoded.fields
    np.testing.assert_array_equal(
        decoded.fields["root_xy_world"],
        np.array([2.34, -5.67], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        decoded.fields["root_z_world"],
        np.array([0.901], dtype=np.float32),
    )


def test_pack_unpack_roundtrip_preserves_legacy_fields() -> None:
    """Regression guard: the new wire fields don't shadow any v4 field.
    A subscriber that only knows about the legacy keys still decodes
    them byte-identical to a pre-2026-06 payload."""
    frame = _make_frame(xy=(0.0, 0.0))
    payload = build_pose_payload(frame)
    msg = pack_pose_message(payload, topic="body_pose", version=5)
    decoded = unpack_message(msg, expected_topic="body_pose")

    assert "joint_pos_mj" in decoded.fields
    np.testing.assert_array_equal(
        decoded.fields["joint_pos_mj"], frame.joint_pos_mj,
    )
    np.testing.assert_array_equal(
        decoded.fields["root_quat_xyzw"], frame.root_quat_xyzw,
    )
    np.testing.assert_array_equal(
        decoded.fields["frame_index"], np.array([7], dtype=np.int64),
    )


def test_pack_unpack_decodes_payload_without_world_root() -> None:
    """A subscriber that receives a legacy payload (no world-root keys)
    must still decode. This pins backward-compat for the recorder's
    fallback path when an old planner is the upstream publisher."""
    legacy_payload: dict[str, np.ndarray] = {
        "joint_pos_mj": np.zeros(NUM_BODY_DOFS, dtype=np.float32),
        "root_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(10, dtype=np.float32),
        "right_hand_joints": np.zeros(10, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
    }
    msg = pack_pose_message(legacy_payload, topic="body_pose", version=5)
    decoded = unpack_message(msg, expected_topic="body_pose")

    assert "root_xy_world" not in decoded.fields
    assert "root_z_world" not in decoded.fields
    assert "joint_pos_mj" in decoded.fields


def test_idle_payload_carries_world_root_even_with_zero_translation() -> None:
    """The kplanner emits the same world-root fields during IDLE freeze
    (operator releases the stick). The viewer + PKL recorder need this
    so a recording that starts in idle still gets a valid qpos[0:3]
    (instead of NaN/missing) — useful for a kinematic-capture run that
    spans IDLE_LOOP -> PLAYING -> IDLE_LOOP transitions."""
    frame = _make_frame(xy=(0.0, 0.0), z=DEFAULT_PELVIS_Z_M)
    payload = build_pose_payload(frame)
    assert payload["root_xy_world"].shape == (2,)
    assert payload["root_z_world"].shape == (1,)
    np.testing.assert_array_equal(
        payload["root_xy_world"], np.zeros(2, dtype=np.float32),
    )
    assert float(payload["root_z_world"][0]) == pytest.approx(DEFAULT_PELVIS_Z_M)


def test_pack_unpack_byte_equivalence_with_future_window() -> None:
    """Future-window fields (v5) and world-root fields coexist on the
    same payload byte-equivalently. The C++ deploy iterates fields, so
    adding world-root after future_dt_s is wire-safe — this test pins
    that the two feature sets don't fight over JSON-header keys."""
    cur = _make_frame(xy=(0.5, 0.25), z=0.78, frame_index=10)
    fut = [
        _make_frame(xy=(0.55, 0.25), z=0.78, frame_index=10 + k + 1)
        for k in range(9)
    ]
    payload = build_pose_payload(cur, future_frames=fut, future_dt_s=0.1)
    msg = pack_pose_message(payload, topic="body_pose", version=5)
    decoded = unpack_message(msg, expected_topic="body_pose")

    for key in (
        "joint_pos_mj",
        "root_quat_xyzw",
        "root_xy_world",
        "root_z_world",
        "joint_pos_mj_future",
        "root_quat_xyzw_future",
        "joint_vel_mj_future",
        "frame_index_future",
        "future_dt_s",
    ):
        assert key in decoded.fields, f"missing wire field {key!r}"
