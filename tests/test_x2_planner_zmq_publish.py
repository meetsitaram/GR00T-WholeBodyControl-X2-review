"""ZMQ wire-format + cadence smoke for the X2 heuristic planner.

Spawns the planner CLI as a subprocess against an EPHEMERAL port (so parallel
test runs don't collide), subscribes with the same Python decoder the C++
deploy uses (``zmq_packed_message_decoder.unpack_message``), and asserts:

  - the wire format carries the keys the deploy expects:
      ``joint_pos_mj`` (f32[31]),
      ``root_quat_xyzw`` (f32[4]),
      ``motion_token`` (f32[64]),
      ``left_hand_joints`` / ``right_hand_joints`` (f32[10]),
      ``frame_index`` (i64[1])
  - the publish cadence is ~50 Hz (over a 1.5s window),
  - frame_index increments monotonically (no skipped frames),
  - the planner exits cleanly on SIGTERM and removes its PID file.

An autouse fixture drains any lingering ZMQ contexts at teardown (no
zombie sockets between tests).
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import zmq


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)


# ---------------------------------------------------------------------------
# Ephemeral port + autouse cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ephemeral_tcp_port() -> int:
    """Reserve a free TCP port on 127.0.0.1 and immediately release it.

    Subject to a brief race against other process binders, but acceptable for
    test orchestration. We pick from the high ephemeral range to minimize
    collisions with system services.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(autouse=True)
def _no_lingering_zmq_contexts():
    """Tear down any ZMQ context this test created.

    Test creates its own context (``zmq.Context()``) rather than using the
    process-wide ``Context.instance()``, so this fixture only verifies that
    the public ``Context.instance()`` isn't holding open sockets between tests.
    """
    yield
    # If a test left ``Context.instance()`` polluted, kill it so the next
    # test starts cleanly.
    try:
        ctx = zmq.Context.instance()
        if ctx is not None and not ctx.closed:
            ctx.term()
    except zmq.error.ZMQError:
        pass


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def planner_subprocess(ephemeral_tcp_port: int, tmp_path: Path):
    """Spawn the planner CLI; tear it down with SIGTERM at teardown."""
    pkl = REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_primitives.pkl"
    bins = REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_bins.yaml"
    if not pkl.exists() or not bins.exists():
        pytest.skip("primitives PKL not curated yet — run curate_x2_primitives first")

    pid_file = tmp_path / "planner.pid"
    cmd = [
        sys.executable,
        "-m", "gear_sonic.scripts.x2_heuristic_planner",
        "--primitives", str(pkl),
        "--bins", str(bins),
        "--pub-host", "127.0.0.1",
        "--pub-port", str(ephemeral_tcp_port),
        "--pid-file", str(pid_file),
        "--duration-s", "10.0",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for the publisher to bind. We can't easily race-detect this without
    # querying ZMQ's state, so just sleep; the planner does ``time.sleep(0.1)``
    # on startup for PUB-SUB warmup which is in the critical path.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if pid_file.exists():
            break
        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"planner subprocess exited early (rc={proc.returncode}):\n{stdout}"
            )
        time.sleep(0.05)
    else:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5.0)
        raise RuntimeError("planner did not write PID file within 5s")

    try:
        yield proc, ephemeral_tcp_port, pid_file
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _subscribe_and_collect(
    port: int, n_seconds: float, topic: str = "pose"
) -> tuple[list, list[float]]:
    """Subscribe to the planner's pose topic and collect frames + receive timestamps."""
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.connect(f"tcp://127.0.0.1:{port}")
    # PUB-SUB warmup; the planner already slept 100ms on bind, so this is
    # a small additional cushion.
    time.sleep(0.2)
    received: list = []
    t_recv: list[float] = []
    deadline = time.monotonic() + n_seconds
    while time.monotonic() < deadline:
        try:
            raw = sub.recv()
        except zmq.error.Again:
            continue
        decoded = unpack_message(raw, expected_topic=topic)
        received.append(decoded)
        t_recv.append(time.monotonic())
    sub.close(linger=0)
    ctx.term()
    return received, t_recv


def test_pose_wire_format_matches_deploy_expectations(planner_subprocess) -> None:
    proc, port, _pid_file = planner_subprocess
    received, _ = _subscribe_and_collect(port, n_seconds=0.6)
    assert len(received) > 5, (
        f"got {len(received)} frames in 0.6s; expected ~30 — is publisher alive?"
    )
    msg = received[0]
    fields = msg.fields
    expected = {
        "joint_pos_mj": (31,),
        "root_quat_xyzw": (4,),
        "motion_token": (64,),
        "left_hand_joints": (10,),
        "right_hand_joints": (10,),
        "frame_index": (1,),
    }
    missing = set(expected) - set(fields)
    assert not missing, f"wire format missing keys: {missing}"
    for name, shape in expected.items():
        assert fields[name].shape == shape, (
            f"field {name}: shape {fields[name].shape} != expected {shape}"
        )
    # Body & root must be float32; frame_index must be int64.
    assert fields["joint_pos_mj"].dtype == np.float32
    assert fields["root_quat_xyzw"].dtype == np.float32
    assert fields["motion_token"].dtype == np.float32
    assert fields["frame_index"].dtype == np.int64
    # Quaternion is unit-length.
    quat = fields["root_quat_xyzw"]
    assert abs(float(np.linalg.norm(quat)) - 1.0) < 1e-3


def test_publish_cadence_is_close_to_50hz(planner_subprocess) -> None:
    _proc, port, _pid_file = planner_subprocess
    n_seconds = 1.5
    received, t_recv = _subscribe_and_collect(port, n_seconds=n_seconds)
    assert len(received) >= 50, (
        f"got {len(received)} frames in {n_seconds}s; expected >=50 (well below 50Hz)"
    )
    # End-to-end rate including warmup.
    duration = t_recv[-1] - t_recv[0]
    rate = (len(received) - 1) / duration
    assert 40.0 <= rate <= 60.0, (
        f"observed rate {rate:.1f} Hz outside [40, 60] band; planner missed the 50Hz target"
    )


def test_frame_index_is_monotonic_no_drops(planner_subprocess) -> None:
    _proc, port, _pid_file = planner_subprocess
    received, _ = _subscribe_and_collect(port, n_seconds=1.0)
    indices = [int(m.fields["frame_index"][0]) for m in received]
    # Allow PUB-SUB warmup to lose the first few; once we're locked, frame_index
    # must increment by exactly 1 each step.
    deltas = np.diff(indices)
    assert (deltas == 1).all(), (
        f"frame_index has gaps or rewinds; indices={indices[:30]}..."
    )


def test_planner_exits_cleanly_on_sigterm(planner_subprocess) -> None:
    proc, _port, pid_file = planner_subprocess
    assert pid_file.exists(), "planner did not create PID file"
    proc.send_signal(signal.SIGTERM)
    try:
        rc = proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("planner did not exit within 5s of SIGTERM")
    assert rc == 0, f"planner exited with rc={rc} (expected 0 on SIGTERM)"
    # PID file must be removed by the cleanup path.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline and pid_file.exists():
        time.sleep(0.05)
    assert not pid_file.exists(), "planner left a stale PID file behind"


def test_command_zmq_topic_drives_planner(
    ephemeral_tcp_port: int, tmp_path: Path
) -> None:
    """Spawn the planner with --zmq-cmd-port and push a few commands at it."""
    pkl = REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_primitives.pkl"
    bins = REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_bins.yaml"
    if not pkl.exists() or not bins.exists():
        pytest.skip("primitives PKL not curated yet")

    # Reserve a second ephemeral port for the command socket.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    cmd_port = s.getsockname()[1]
    s.close()

    pid_file = tmp_path / "planner.pid"
    cmd = [
        sys.executable, "-m", "gear_sonic.scripts.x2_heuristic_planner",
        "--primitives", str(pkl),
        "--bins", str(bins),
        "--pub-host", "127.0.0.1",
        "--pub-port", str(ephemeral_tcp_port),
        "--zmq-cmd-host", "127.0.0.1",
        "--zmq-cmd-port", str(cmd_port),
        "--zmq-cmd-topic", "planner_cmd",
        "--pid-file", str(pid_file),
        "--duration-s", "5.0",
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.Popen(
        cmd, cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not pid_file.exists():
            if proc.poll() is not None:
                stdout = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"planner exited early:\n{stdout}")
            time.sleep(0.05)
        assert pid_file.exists()

        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.setsockopt(zmq.LINGER, 0)
        pub.bind(f"tcp://127.0.0.1:{cmd_port}")
        time.sleep(0.3)  # PUB-SUB warmup
        for intent, magnitude in [
            ("turn_left", "deg_45"),
            ("walk", "forward"),
            ("idle", "default"),
        ]:
            payload = json.dumps({"intent": intent, "magnitude": magnitude})
            pub.send_multipart([b"planner_cmd", payload.encode("utf-8")])
            time.sleep(0.05)
        time.sleep(0.5)
        pub.close(linger=0)
        ctx.term()

        # Send shutdown via the command channel and verify clean exit.
        ctx2 = zmq.Context()
        pub2 = ctx2.socket(zmq.PUB)
        pub2.setsockopt(zmq.LINGER, 0)
        pub2.bind(f"tcp://127.0.0.1:{cmd_port}")
        time.sleep(0.3)
        pub2.send_multipart([
            b"planner_cmd",
            json.dumps({"intent": "shutdown"}).encode("utf-8"),
        ])
        pub2.close(linger=0)
        ctx2.term()

        try:
            rc = proc.wait(timeout=4.0)
            assert rc == 0
        except subprocess.TimeoutExpired:
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=3.0)
            pytest.fail("planner did not honor zmq shutdown")
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)
