"""
M2 acceptance gate (offline part): mock-VLA -> ZMQ -> Python decoder smoke test.

The full M2 acceptance gate is a sim-mode run of the C++ deploy, driven by
``gear_sonic/scripts/mock_vla_publish_stand_token.py`` and tailed by
``gear_sonic/scripts/dump_x2_debug.py``. That gate requires ROS 2 + AimDK
SDK + ONNX runtime + the 22k SONIC checkpoint, none of which are present
on the laptop running these tests.

This pytest gate covers the *Python-only* slice of the same loop:

* The mock-VLA helper (``mock_vla_publish_stand_token.py``) publishes a
  packed-binary message on the ``pose`` topic.
* The Python decoder (``zmq_packed_message_decoder.unpack_message``) -- the
  reference implementation that ``dump_x2_debug.py`` and the C++ deploy's
  ``ZmqPoseInputSource`` agree on -- reads it back.
* We assert the wire format carries every field the v0 deploy expects:

    - ``joint_pos_mj``    float32[31]   (body refs)
    - ``root_quat_xyzw``  float32[4]    (root orientation)
    - ``motion_token``    float32[64]
    - ``left_hand_joints``  float32[10]
    - ``right_hand_joints`` float32[10]
    - ``frame_index``     int64[1]

The C++ side is exercised separately by the offline syntax-check build
(``cmake -S ... -B ... -DAGI_X2_OFFLINE_SYNTAX_CHECK=ON``); together these
two gates cover the wire-format invariant from both languages without
needing the X2 dev box.

Run with::

    .venv/bin/python -m pytest tests/test_x2_zmq_vla_smoke.py -v
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import zmq


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_mock_vla_module():
    """Load the mock-VLA module without depending on package install state."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec = importlib.util.spec_from_file_location(
        "mock_vla_publish_stand_token",
        REPO_ROOT / "gear_sonic" / "scripts" / "mock_vla_publish_stand_token.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load mock_vla_publish_stand_token.py spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_one_payload(mock_module, tick: int, hand_dof: int = 10):
    """Construct the same dict the running mock-VLA loop hands to pack_pose_message."""
    body_pose_mj = np.array(mock_module.DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32)
    root_quat_xyzw = np.array(mock_module.IDENTITY_ROOT_QUAT_XYZW, dtype=np.float32)
    token = np.zeros(mock_module.SONIC_MOTION_TOKEN_DIM, dtype=np.float32)
    left_hand = np.zeros(hand_dof, dtype=np.float32)
    right_hand = np.zeros(hand_dof, dtype=np.float32)
    return {
        "joint_pos_mj": body_pose_mj,
        "root_quat_xyzw": root_quat_xyzw,
        "motion_token": token,
        "left_hand_joints": left_hand,
        "right_hand_joints": right_hand,
        "frame_index": np.array([tick], dtype=np.int64),
    }


def test_default_stand_pose_constants_match_cpp_header() -> None:
    """The 31-D hard-coded pose mirrors policy_parameters.hpp::default_angles."""
    mock = _load_mock_vla_module()
    cpp_header = (
        REPO_ROOT
        / "gear_sonic_deploy"
        / "src"
        / "x2"
        / "agi_x2_deploy_onnx_ref"
        / "include"
        / "policy_parameters.hpp"
    )
    text = cpp_header.read_text()
    block_start = text.index("default_angles")
    block_end = text.index("};", block_start)
    block = text[block_start:block_end]
    raw_floats: list[float] = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("/*"):
            continue
        if "{" in line or "default_angles" in line or "[" in line:
            continue
        token = line.split(",")[0].split("//")[0].strip()
        if not token:
            continue
        try:
            raw_floats.append(float(token))
        except ValueError:
            continue
    assert len(raw_floats) == mock.NUM_BODY_DOFS, (
        f"parsed {len(raw_floats)} angles from policy_parameters.hpp, "
        f"expected {mock.NUM_BODY_DOFS}"
    )
    np.testing.assert_allclose(
        mock.DEFAULT_STAND_POSE_MUJOCO_RAD, raw_floats, rtol=1e-6, atol=1e-9,
        err_msg=(
            "DEFAULT_STAND_POSE_MUJOCO_RAD diverged from "
            "policy_parameters.hpp::default_angles. Re-sync the Python "
            "constant when the C++ default pose changes."
        ),
    )


def test_mock_vla_payload_round_trip_through_zmq() -> None:
    """Mock-VLA bytes -> ZMQ inproc loopback -> Python decoder dict."""
    sys.path.insert(0, str(REPO_ROOT))
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
    from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message

    mock = _load_mock_vla_module()
    payload = _build_one_payload(mock, tick=42, hand_dof=10)
    raw = pack_pose_message(payload, topic="pose", version=4)

    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, "pose")
    sub.setsockopt(zmq.RCVTIMEO, 1000)
    endpoint = "inproc://x2_vla_smoke"
    pub.bind(endpoint)
    sub.connect(endpoint)

    # PUB-SUB needs a brief settling delay even on inproc transport.
    time.sleep(0.1)

    pub.send(raw)
    received = sub.recv()
    pub.close(linger=0)
    sub.close(linger=0)
    ctx.term()

    assert received == raw, "ZMQ delivered a mutated frame"

    decoded = unpack_message(received, expected_topic="pose")
    fields = decoded.fields

    expected_keys = {
        "joint_pos_mj",
        "root_quat_xyzw",
        "motion_token",
        "left_hand_joints",
        "right_hand_joints",
        "frame_index",
    }
    missing = expected_keys - set(fields.keys())
    assert not missing, f"decoder dropped fields: {missing}"

    np.testing.assert_array_equal(
        fields["joint_pos_mj"].astype(np.float32),
        np.asarray(mock.DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        fields["root_quat_xyzw"].astype(np.float32),
        np.asarray(mock.IDENTITY_ROOT_QUAT_XYZW, dtype=np.float32),
    )
    assert fields["motion_token"].shape == (mock.SONIC_MOTION_TOKEN_DIM,)
    assert fields["left_hand_joints"].shape == (10,)
    assert fields["right_hand_joints"].shape == (10,)
    assert int(fields["frame_index"][0]) == 42
    assert decoded.topic == "pose"
    assert decoded.version == 4


def test_mock_vla_payload_supports_g1_compat_7dof() -> None:
    """The hand-dof override flag must yield a smaller hand-joints field."""
    sys.path.insert(0, str(REPO_ROOT))
    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message
    from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message

    mock = _load_mock_vla_module()
    payload = _build_one_payload(mock, tick=7, hand_dof=7)
    raw = pack_pose_message(payload, topic="pose", version=4)
    decoded = unpack_message(raw, expected_topic="pose")
    fields = decoded.fields
    assert fields["left_hand_joints"].shape == (7,)
    assert fields["right_hand_joints"].shape == (7,)
    # Body remains 31-D regardless of hand variant.
    assert fields["joint_pos_mj"].shape == (mock.NUM_BODY_DOFS,)


@pytest.mark.skipif(
    not (REPO_ROOT / "gear_sonic_deploy" / "deploy_x2.sh").exists(),
    reason="deploy_x2.sh is not in the repo (out-of-tree build)",
)
def test_deploy_x2_sh_exposes_vla_flags() -> None:
    """The bash wrapper must surface the VLA flags so operators can flip the loop."""
    text = (REPO_ROOT / "gear_sonic_deploy" / "deploy_x2.sh").read_text()
    for flag in ("--vla", "--vla-zmq-host", "--vla-zmq-port", "--vla-debug-port"):
        assert flag in text, f"deploy_x2.sh is missing {flag}"
    # The actual ros2 invocation must propagate --input-type zmq.
    assert "--input-type" in text
    assert "zmq" in text
