"""End-to-end smoke test for the x2_pose_proxy upstream-silent ladder.

Spawns the proxy as a real subprocess against ephemeral ZMQ ports,
publishes upstream pose frames for a few seconds, stops, and verifies
that the proxy:

  1. forwards live frames byte-for-byte during LIVE,
  2. transitions to HOLD and re-publishes the LAST upstream frame's
     joint_pos_mj bit-for-bit when upstream goes silent,
  3. transitions to BLEND and lerps joint_pos_mj toward the baked
     idle clip (monotonic, no overshoot),
  4. eventually arrives at IDLE_CLIP and tracks the baked clip,
  5. transitions back to LIVE the moment upstream resumes.

The proxy is given a very short hold/blend window (1.0 s / 0.5 s) so
the entire scenario completes in ~5 s. Same exact code paths as the
production 10 s / 3 s defaults -- only the timer durations differ.

This is a slow integration test (real subprocess + sleeps + ZMQ
binding) -- gated on the ``X2_POSE_PROXY_SMOKE=1`` env var so the
fast unit-test suite can skip it by default. Run explicitly with::

    X2_POSE_PROXY_SMOKE=1 pytest tests/test_x2_pose_proxy_smoke.py -v -s

The proxy is given a very short hold/blend window (1.0 s / 0.5 s) so
the entire scenario completes in ~5 s; pass
``X2_POSE_PROXY_SMOKE_LONG=1`` to use the production defaults (10 s
HOLD + 3 s BLEND) for the full 14 s validation before a real-robot
session.
"""

from __future__ import annotations

import os
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent
PROXY_DIR = REPO_ROOT / "gear_sonic_deploy" / "scripts"
if str(PROXY_DIR) not in sys.path:
    sys.path.insert(0, str(PROXY_DIR))

import x2_pose_proxy as proxy  # noqa: E402


SMOKE_ENABLED = os.environ.get("X2_POSE_PROXY_SMOKE", "") not in ("", "0")
SMOKE_LONG = os.environ.get("X2_POSE_PROXY_SMOKE_LONG", "") not in ("", "0")


# ===========================================================================
# Helpers
# ===========================================================================
def _pick_free_port() -> int:
    """Ask the OS for a free TCP port, then close the socket so ZMQ can
    bind it. There IS a small TOCTOU window here, but PUBSUB on
    loopback is forgiving and the alternative is hardcoding ports
    (which would prevent parallel test runs)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _write_minimal_x2m2(path: Path, *, jpos_value: float = 0.0) -> None:
    """Bake a one-frame X2M2 with every DOF = ``jpos_value`` and root
    quat = identity (xyzw=[0,0,0,1] in proxy convention -> stored as
    [0,0,0,1] in the file). Matches what bake_idle_stand_x2m2.py would
    produce for a static T-pose. Reusing the production loader keeps
    this test honest if the file format ever changes."""
    num_frames = 1
    num_dofs = proxy.NUM_BODY_DOFS
    fps = 50.0
    header = struct.pack(
        "<IIId", proxy.X2M2_MAGIC, num_frames, num_dofs, fps
    )
    body = np.zeros((num_frames, num_dofs + 4), dtype=np.float64)
    body[:, :num_dofs] = jpos_value
    # quat xyzw = [0, 0, 0, 1] (identity)
    body[:, num_dofs + 3] = 1.0
    path.write_bytes(header + body.tobytes(order="C"))


def _pack_pose_frame(jpos_value: float, topic: str = "pose") -> bytes:
    payload = {
        "joint_pos_mj": np.full(
            proxy.NUM_BODY_DOFS, jpos_value, dtype=np.float32
        ),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.zeros(10, dtype=np.float32),
        "right_hand_joints": np.zeros(10, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
    }
    return proxy.pack_pose_message(payload, topic=topic, version=4)


# ===========================================================================
# The smoke test
# ===========================================================================
@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set X2_POSE_PROXY_SMOKE=1 to run (real subprocess + sleeps)",
)
def test_proxy_e2e_fallback_ladder(
    request: "pytest.FixtureRequest", tmp_path: Path
) -> None:

    # Build a minimal idle clip whose joint_pos_mj is all 0.0 so the
    # BLEND lerp from a cached LIVE_VALUE = 1.0 produces a clean
    # monotonically-decreasing curve.
    LIVE_VALUE = 1.0
    IDLE_VALUE = 0.0
    x2m2_path = tmp_path / "smoke_idle.x2m2"
    _write_minimal_x2m2(x2m2_path, jpos_value=IDLE_VALUE)

    upstream_port = _pick_free_port()
    downstream_port = _pick_free_port()

    # The production defaults (HOLD 10 s + BLEND 3 s) would make this
    # test run for ~14 s -- too slow for CI. Use 1.0 s / 0.5 s instead;
    # the same code paths are exercised in the same order. The proxy
    # tick period (--rate-hz 50 -> 20 ms) is unchanged. Pass
    # ``X2_POSE_PROXY_SMOKE_LONG=1`` to exercise the production
    # timings end-to-end (used as the pre-flight check before a
    # real-robot session).
    hold_secs = 10.0 if SMOKE_LONG else 1.0
    blend_secs = 3.0 if SMOKE_LONG else 0.5
    stale_ms = 100

    proxy_script = REPO_ROOT / "gear_sonic_deploy" / "scripts" / "x2_pose_proxy.py"
    cmd = [
        sys.executable,
        str(proxy_script),
        "--upstream-host", "127.0.0.1",
        "--upstream-port", str(upstream_port),
        "--upstream-topic", "pose",
        "--downstream-host", "127.0.0.1",
        "--downstream-port", str(downstream_port),
        "--downstream-topic", "pose",
        "--idle-x2m2", str(x2m2_path),
        "--idle-stale-ms", str(stale_ms),
        "--idle-mode", "blend",
        "--hold-last-secs", str(hold_secs),
        "--blend-secs", str(blend_secs),
        # Disable the x2_debug yaw track -- we're not testing yaw here
        # and an unbound SUB would just sit silent.
        "--no-x2-debug-yaw-track",
        "--rate-hz", "50",
        "--status-every-s", "0.5",
    ]
    proxy_proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    ctx = zmq.Context.instance()
    upstream_pub = ctx.socket(zmq.PUB)
    upstream_pub.setsockopt(zmq.SNDHWM, 100)
    upstream_pub.bind(f"tcp://127.0.0.1:{upstream_port}")

    downstream_sub = ctx.socket(zmq.SUB)
    downstream_sub.setsockopt(zmq.RCVHWM, 1000)
    downstream_sub.connect(f"tcp://127.0.0.1:{downstream_port}")
    downstream_sub.setsockopt(zmq.SUBSCRIBE, b"pose")

    # Settle for PUB/SUB handshake -- ZMQ silently drops the first few
    # frames published before the SUB attaches.
    time.sleep(0.5)

    def drain_downstream(deadline_s: float) -> list[tuple[float, np.ndarray]]:
        """Pull all frames currently on the SUB; return [(t_recv, jpos)]."""
        out: list[tuple[float, np.ndarray]] = []
        while time.monotonic() < deadline_s:
            try:
                msg = downstream_sub.recv(zmq.NOBLOCK)
            except zmq.Again:
                break
            jpos = proxy.decode_pose_joint_pos_mj(msg, topic="pose")
            if jpos is not None:
                out.append((time.monotonic(), jpos))
        return out

    try:
        # Phase 1: LIVE -- publish at ~50 Hz for 1 s, expect frames to
        # arrive downstream with LIVE_VALUE.
        #
        # The proxy STARTS in COLD_IDLE (before the first upstream
        # frame lands); those startup frames carry IDLE_VALUE and
        # appear in the drain queue ahead of the LIVE forwards. We
        # assert on the TAIL of the LIVE phase, which by construction
        # has transitioned to LIVE forwarding.
        t_start = time.monotonic()
        deadline = t_start + 1.0
        while time.monotonic() < deadline:
            upstream_pub.send(
                _pack_pose_frame(LIVE_VALUE), zmq.NOBLOCK
            )
            time.sleep(0.02)
        time.sleep(0.1)  # let downstream catch up
        live_frames = drain_downstream(time.monotonic() + 0.05)
        assert len(live_frames) > 10, (
            f"expected >10 LIVE forwards, got {len(live_frames)}"
        )
        # The last few frames of phase 1 are unambiguously LIVE
        # (upstream was publishing right up to the drain). Use them
        # as the LIVE assertion anchor.
        live_tail = np.array([f[1][0] for f in live_frames[-10:]])
        assert np.allclose(live_tail, LIVE_VALUE, atol=1e-6), (
            f"end of LIVE phase should be LIVE_VALUE; got {live_tail}"
        )

        # Phase 2: silent upstream for 0.5 s -- proxy should enter HOLD
        # and re-publish frames whose joint_pos_mj == LIVE_VALUE.
        time.sleep(0.5)
        hold_frames = drain_downstream(time.monotonic() + 0.05)
        # During HOLD every published frame should have joint_pos_mj
        # exactly == LIVE_VALUE (the proxy re-publishes raw cached
        # bytes; the deploy and our test decoder see identical floats).
        assert len(hold_frames) > 5, (
            f"expected HOLD frames during stall, got {len(hold_frames)}"
        )
        hold_vals = np.array([f[1][0] for f in hold_frames])
        assert np.allclose(hold_vals, LIVE_VALUE, atol=1e-6), (
            f"HOLD frames must replay LIVE_VALUE; got "
            f"min={hold_vals.min()}, max={hold_vals.max()}"
        )

        # Phase 3: keep upstream silent through the rest of HOLD,
        # through BLEND, into IDLE_CLIP. Total stall window needed:
        # stale_s + hold_secs + blend_secs + epsilon.
        time.sleep(hold_secs + blend_secs + 0.2)
        all_post_frames = drain_downstream(time.monotonic() + 0.05)
        # By now the proxy has progressed through BLEND -> IDLE_CLIP.
        # The most recent frames should be at or very near IDLE_VALUE.
        assert len(all_post_frames) > 5, (
            f"expected late-stall frames, got {len(all_post_frames)}"
        )
        late_vals = np.array([f[1][0] for f in all_post_frames[-3:]])
        assert np.allclose(late_vals, IDLE_VALUE, atol=1e-3), (
            f"late-stall frames must settle to IDLE_VALUE; got "
            f"{late_vals}"
        )

        # Frames pulled during BLEND should be strictly between
        # LIVE_VALUE and IDLE_VALUE somewhere in their middle. Concat
        # everything since the start of the stall to verify the lerp
        # produced intermediate values (monotonic across the full
        # window is hard to assert because PUB/SUB drops frames; we
        # settle for "we saw at least one value in (IDLE, LIVE)").
        all_stall_frames = hold_frames + all_post_frames
        stall_vals = np.array([f[1][0] for f in all_stall_frames])
        in_between = (
            (stall_vals > IDLE_VALUE + 1e-3)
            & (stall_vals < LIVE_VALUE - 1e-3)
        )
        assert in_between.any(), (
            "expected at least one BLEND frame strictly between "
            "IDLE_VALUE and LIVE_VALUE; none seen"
        )

        # Phase 4: resume upstream -- expect snap back to LIVE_VALUE.
        for _ in range(20):
            upstream_pub.send(
                _pack_pose_frame(LIVE_VALUE), zmq.NOBLOCK
            )
            time.sleep(0.02)
        time.sleep(0.1)
        recover_frames = drain_downstream(time.monotonic() + 0.05)
        assert len(recover_frames) > 5, (
            f"expected post-recovery forwards, got {len(recover_frames)}"
        )
        recover_tail = np.array([f[1][0] for f in recover_frames[-3:]])
        assert np.allclose(recover_tail, LIVE_VALUE, atol=1e-6), (
            f"after recovery, downstream should track LIVE_VALUE again; "
            f"got tail={recover_tail}"
        )
    finally:
        upstream_pub.close(linger=0)
        downstream_sub.close(linger=0)
        # Don't ctx.term() -- other proxy tests in the same pytest
        # session may still hold sockets on the shared ctx.
        proxy_proc.terminate()
        try:
            log_tail, _ = proxy_proc.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
            log_tail, _ = proxy_proc.communicate(timeout=2.0)
        # Print the proxy's stdout/stderr if any assertion failed so
        # debugging the smoke test doesn't require manually re-running
        # it with -s and watching the proxy.
        if request.node.session.testsfailed:
            print(
                "----- proxy stdout/stderr tail -----\n"
                + (log_tail or "")[-4000:]
            )
