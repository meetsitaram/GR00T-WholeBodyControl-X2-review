"""ZMQ wire-format + cadence smoke for the X2 neural kplanner.

Mirrors ``test_x2_planner_zmq_publish.py`` but exercises
``gear_sonic.scripts.x2_kplanner`` instead. The wire-format contract is
identical (``build_pose_payload`` is shared), so the only differences are:

  - The daemon publishes on the ``body_pose`` topic by default
    (``--body-pose-port``), not ``pose``.
  - Boot is slower: it must load three MotionBricks Lightning checkpoints
    (~2 GB) and run one synchronous ``replan_with_velocity()`` before
    the publish loop starts. We wait up to 120 s for the
    ``"first replan complete"`` marker in stdout.
  - Device selection auto-detects a CUDA-capable torch. The test
    fixture probes the active Python's torch for ``cuda.is_available()``
    AND a compute capability the installed wheel was built for, then
    passes either ``--device cuda`` (preferred, ~14 ms predict latency
    on a 5090 + cu128 wheel) or falls back to ``--device cpu``
    (~500--800 ms predict latency). Force CPU with
    ``KPLANNER_TEST_FORCE_CPU=1`` if you need to reproduce a CPU-only
    deploy.

If any of the kplanner checkpoints (or env deps like
``pytorch-lightning`` / ``vector-quantize-pytorch``) are missing, the
tests skip rather than fail -- this matches the heuristic test's
"skip when primitives not curated" pattern.

Recommended invocation::

    ~/miniconda3/envs/env_isaaclab/bin/python -m pytest \\
        tests/test_x2_kplanner_zmq_publish.py -v

(env_isaaclab ships torch 2.7.0+cu128 with sm_120 support; the base
miniconda env ships torch 2.6+cu124 which is fine on CPU but crashes
on Blackwell GPUs.)
"""

from __future__ import annotations

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
# motionbricks/ is a sibling package next to gear_sonic/; the daemon
# subprocess sets PYTHONPATH explicitly, but this test file probes the
# loader here at import-skip time, so we also need it on sys.path.
_MB_ROOT = REPO_ROOT / "motionbricks"
if str(_MB_ROOT) not in sys.path:
    sys.path.insert(0, str(_MB_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)


# ---------------------------------------------------------------------------
# Skip when the kplanner stack isn't installable.
# ---------------------------------------------------------------------------


# Must mirror ``X2PlannerPaths.default()`` + the argparse defaults in
# ``gear_sonic/scripts/x2_kplanner.py``. Pinned step checkpoints so a
# fresh training run doesn't silently re-point inference (and tests)
# at an unverified checkpoint.
_REQUIRED_CKPTS = [
    REPO_ROOT / "motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0200000.ckpt",
    REPO_ROOT / "motionbricks/out/motionbricks_pose_x2_v2/version_1/checkpoints/model-step=0250000.ckpt",
    REPO_ROOT / "motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0235000.ckpt",
]


def _skip_if_kplanner_unavailable() -> None:
    """Skip the test gracefully if the kplanner can't load on this machine."""
    for p in _REQUIRED_CKPTS:
        if not p.is_file():
            pytest.skip(f"kplanner checkpoint missing: {p}")
    # Quick import-only probe; avoids paying the full model-load cost
    # just to gate the test.
    try:
        import pytorch_lightning  # noqa: F401
        import vector_quantize_pytorch  # noqa: F401
        from motionbricks.motion_backbone.inference.load_x2_planner import (  # noqa: F401
            X2PlannerPaths,
        )
    except Exception as exc:  # pragma: no cover -- env-dependent
        pytest.skip(f"kplanner deps not importable: {exc!r}")


def _pick_kplanner_device() -> str:
    """Probe torch for a CUDA-capable wheel; fall back to cpu otherwise.

    The base miniconda env ships torch 2.6+cu124 which crashes on
    Blackwell (sm_120 RTX 5090) the moment any kernel launches. Probing
    ``torch.cuda.is_available()`` alone isn't enough -- it returns True
    on cu124 + 5090 too, but the first kernel raises ``CUDA error: no
    kernel image is available for execution on the device``. We also
    check the device's compute capability is in the wheel's arch list.
    """
    if os.environ.get("KPLANNER_TEST_FORCE_CPU"):
        return "cpu"
    try:
        import torch
    except Exception:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        # ``get_arch_list()`` returns e.g. ['sm_75', 'sm_80', ..., 'sm_120'].
        arches = {a.replace("sm_", "") for a in torch.cuda.get_arch_list()}
        if f"{major}{minor}" not in arches:
            return "cpu"
    except Exception:
        return "cpu"
    return "cuda"


# ---------------------------------------------------------------------------
# Fixtures (mostly verbatim from test_x2_planner_zmq_publish.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def ephemeral_tcp_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(autouse=True)
def _no_lingering_zmq_contexts():
    yield
    try:
        ctx = zmq.Context.instance()
        if ctx is not None and not ctx.closed:
            ctx.term()
    except zmq.error.ZMQError:
        pass


@pytest.fixture
def kplanner_subprocess(ephemeral_tcp_port: int, tmp_path: Path):
    """Spawn the kplanner CLI on ``--body-pose-port``; SIGTERM at teardown.

    Boot is slow: pose model is ~1.5 GB, predict() does one round-trip
    synchronously before the publish loop starts. We bound the readiness
    wait at 120 s -- generous because on CPU the first replan is the
    slowest the daemon will ever do (compiled torch graphs cached after).
    """
    _skip_if_kplanner_unavailable()
    pid_file = tmp_path / "kplanner.pid"
    device = _pick_kplanner_device()
    # GPU replans take ~14 ms; CPU replans take ~500-800 ms. The buffer
    # cap is ~48 frames = ~960 ms at 50 Hz, so on GPU we can let the
    # buffer drain to ~16 frames before refilling, but on CPU we need
    # the worker to fire while there's still meaningful headroom.
    replan_threshold = "16" if device == "cuda" else "40"
    cmd = [
        sys.executable,
        "-m", "gear_sonic.scripts.x2_kplanner",
        "--device", device,
        "--body-pose-port", str(ephemeral_tcp_port),
        "--pid-file", str(pid_file),
        "--duration-s", "20.0",
        # Skip warmup so the very first published frame already has the
        # neural-planner's future window (the warmup frames omit it).
        "--warmup-quiet-stand-s", "0.0",
        "--replan-threshold-frames", replan_threshold,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = (
        str(REPO_ROOT / "motionbricks") + os.pathsep + str(REPO_ROOT)
    )
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for the kplanner's readiness marker (post-first-replan) OR the
    # PID file (which is created inside the ``with PidFile(...):`` block
    # the daemon enters AFTER the first replan completes).
    deadline = time.monotonic() + 120.0
    while time.monotonic() < deadline:
        if pid_file.exists():
            break
        if proc.poll() is not None:
            stdout = proc.stdout.read() if proc.stdout else ""
            raise RuntimeError(
                f"kplanner subprocess exited early (rc={proc.returncode}):\n{stdout}"
            )
        time.sleep(0.1)
    else:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=10.0)
        raise RuntimeError("kplanner did not write PID file within 120s")

    try:
        yield proc, ephemeral_tcp_port, pid_file
    finally:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subscribe_and_collect(
    port: int, n_seconds: float, topic: str = "body_pose"
) -> tuple[list, list[float]]:
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.connect(f"tcp://127.0.0.1:{port}")
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_body_pose_wire_format_matches_deploy_expectations(kplanner_subprocess) -> None:
    _proc, port, _pid_file = kplanner_subprocess
    received, _ = _subscribe_and_collect(port, n_seconds=0.8)
    assert len(received) > 5, (
        f"got {len(received)} frames in 0.8s; expected ~40 — is publisher alive?"
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
    assert fields["joint_pos_mj"].dtype == np.float32
    assert fields["root_quat_xyzw"].dtype == np.float32
    assert fields["motion_token"].dtype == np.float32
    assert fields["frame_index"].dtype == np.int64
    quat = fields["root_quat_xyzw"]
    assert abs(float(np.linalg.norm(quat)) - 1.0) < 1e-2

    # Future-window contract (v4 wire from build_pose_payload).
    assert "joint_pos_mj_future" in fields
    assert "root_quat_xyzw_future" in fields
    assert "joint_vel_mj_future" in fields
    assert "frame_index_future" in fields
    assert fields["joint_pos_mj_future"].shape == (9, 31)
    assert fields["root_quat_xyzw_future"].shape == (9, 4)


def test_publish_cadence_is_close_to_50hz(kplanner_subprocess) -> None:
    """The publisher must tick at ~50 Hz when running on the deploy
    target (GPU); on CPU the buffer occasionally starves mid-replan and
    cadence dips, but the publisher must stay above ~15 Hz.

    On GPU + sm_120-capable torch: predict() runs in ~14 ms, the ring
    buffer never starves, and the rate sits in the 40--60 Hz band.
    On CPU with the production-sized pose checkpoint: predict() runs in
    ~500--800 ms, so we accept the lower bound of ~15 Hz to catch a
    dead publisher without flagging the known CPU latency.

    The upper bound stays at 60 Hz: any value above that means the
    publish loop's tick timer is broken.
    """
    _proc, port, _pid_file = kplanner_subprocess
    n_seconds = 2.0
    received, t_recv = _subscribe_and_collect(port, n_seconds=n_seconds)
    device = _pick_kplanner_device()
    if device == "cuda":
        lower_count, lower_rate = 60, 40.0
    else:
        lower_count, lower_rate = 20, 15.0
    assert len(received) >= lower_count, (
        f"got {len(received)} frames in {n_seconds}s; publisher appears "
        f"dead (expected >={lower_count} on device={device})"
    )
    duration = t_recv[-1] - t_recv[0]
    rate = (len(received) - 1) / duration
    assert lower_rate <= rate <= 60.0, (
        f"observed rate {rate:.1f} Hz outside [{lower_rate}, 60] band "
        f"on device={device}"
    )


def test_frame_index_is_monotonic(kplanner_subprocess) -> None:
    """frame_index must strictly increase. The kplanner doesn't guarantee
    +1-per-tick across ring-buffer rollovers in the warmup interval, so
    we assert strict-monotonic rather than gap-free."""
    _proc, port, _pid_file = kplanner_subprocess
    received, _ = _subscribe_and_collect(port, n_seconds=1.0)
    indices = [int(m.fields["frame_index"][0]) for m in received]
    diffs = np.diff(indices)
    assert (diffs >= 1).all(), (
        f"frame_index has rewinds or duplicates; indices={indices[:30]}..."
    )


def test_kplanner_exits_cleanly_on_sigterm(kplanner_subprocess) -> None:
    proc, _port, pid_file = kplanner_subprocess
    assert pid_file.exists(), "kplanner did not create PID file"
    proc.send_signal(signal.SIGTERM)
    try:
        rc = proc.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        pytest.fail("kplanner did not exit within 10s of SIGTERM")
    assert rc == 0, f"kplanner exited with rc={rc} (expected 0 on SIGTERM)"
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline and pid_file.exists():
        time.sleep(0.05)
    assert not pid_file.exists(), "kplanner left a stale PID file behind"
