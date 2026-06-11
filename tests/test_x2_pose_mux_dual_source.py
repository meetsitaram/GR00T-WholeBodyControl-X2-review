"""End-to-end smoke test for x2_pose_mux dual-source manual-takeover.

The 2026-06-10 milestone added dual-source arbitration + a
``vla_control`` edge-event PUB so an operator can teleop-nudge the arm
out of a stuck VLA pose without restarting any process. The 2026-06-11
milestone moved this logic from PC2 to the laptop-side
``x2_pose_mux``; this test spawns the mux as a real subprocess with
all three sockets wired (primary SUB, override SUB, vla_control PUB)
against ephemeral loopback ports and validates the arbitration
semantics end-to-end:

  1. Primary fresh / override silent -> primary frames forwarded
     verbatim downstream; no vla_control events on the wire.
  2. Override starts publishing -> override frames take priority and
     are forwarded verbatim; an ``override_engaged`` event lands on
     vla_control on the first override tick.
  3. Override goes silent (primary still publishing) -> mux waits
     out the ``--override-stale-ms`` debounce window, emits
     ``override_released`` exactly once, then forwards primary again.
  4. The mux is always dual-source -- if the operator doesn't want
     takeover they simply don't run it (the recorder publishes
     straight to the PC2 watchdog). The pre-2026-06-10 "single source"
     test below is therefore retained as a primary-only assertion
     against the mux (override SUB attached but no operator activity).

Like the fallback-ladder smoke, this is a slow integration test
(subprocess + sleeps + ZMQ binding) and is gated on the
``X2_POSE_PROXY_SMOKE=1`` env var (kept under the old prefix so
existing operator runbooks keep working) so the fast unit-test pass
can skip it. Run explicitly with::

    X2_POSE_PROXY_SMOKE=1 pytest \\
        tests/test_x2_pose_mux_dual_source.py -v -s
"""

from __future__ import annotations

import json
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

try:
    import msgpack as _msgpack
except ImportError:
    _msgpack = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.pose_pipeline import wire  # noqa: E402


class _ProxyShim:
    NUM_BODY_DOFS = wire.NUM_BODY_DOFS
    X2M2_MAGIC = wire.X2M2_MAGIC
    pack_pose_message = staticmethod(wire.pack_pose_message)
    decode_pose_joint_pos_mj = staticmethod(wire.decode_pose_joint_pos_mj)
    decode_pose_left_hand = staticmethod(wire.decode_pose_left_hand)
    decode_pose_right_hand = staticmethod(wire.decode_pose_right_hand)


proxy = _ProxyShim()

MUX_SCRIPT = REPO_ROOT / "gear_sonic" / "scripts" / "x2_pose_mux.py"


SMOKE_ENABLED = os.environ.get("X2_POSE_PROXY_SMOKE", "") not in ("", "0")


# ===========================================================================
# Helpers (small enough to inline; we don't share with the fallback smoke
# because the two tests intentionally exercise disjoint surfaces and the
# helpers diverge in subtle ways -- e.g. this one needs override-port
# scaffolding the ladder smoke doesn't care about).
# ===========================================================================
def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _pack_pose_frame(
    jpos_value: float,
    topic: str = "pose",
    *,
    left_hand_value: float = 0.0,
    right_hand_value: float = 0.0,
) -> bytes:
    payload = {
        "joint_pos_mj": np.full(
            proxy.NUM_BODY_DOFS, jpos_value, dtype=np.float32
        ),
        "root_quat_xyzw": np.array(
            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
        ),
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": np.full(10, left_hand_value, dtype=np.float32),
        "right_hand_joints": np.full(10, right_hand_value, dtype=np.float32),
        "frame_index": np.array([0], dtype=np.int64),
    }
    return proxy.pack_pose_message(payload, topic=topic, version=4)


def _drain_pose(sub: zmq.Socket, deadline_s: float) -> list[float]:
    """Pull all pending pose frames; return [joint_pos_mj[0]]."""
    out: list[float] = []
    while time.monotonic() < deadline_s:
        try:
            msg = sub.recv(zmq.NOBLOCK)
        except zmq.Again:
            break
        jpos = proxy.decode_pose_joint_pos_mj(msg, topic="pose")
        if jpos is not None:
            out.append(float(jpos[0]))
    return out


def _drain_control(sub: zmq.Socket, deadline_s: float) -> list[dict]:
    """Pull all pending vla_control events; return parsed JSON dicts."""
    out: list[dict] = []
    while time.monotonic() < deadline_s:
        try:
            parts = sub.recv_multipart(zmq.NOBLOCK)
        except zmq.Again:
            break
        if len(parts) >= 2:
            try:
                out.append(json.loads(parts[1].decode("utf-8")))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    return out


def _spawn_proxy(
    cmd: list[str],
) -> subprocess.Popen:
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )


# ===========================================================================
# Dual-source arbitration smoke
# ===========================================================================
@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set X2_POSE_PROXY_SMOKE=1 to run (real subprocess + sleeps)",
)
def test_proxy_dual_source_override_takes_priority_and_emits_events(
    tmp_path: Path,
) -> None:
    PRIMARY_VALUE = 0.10
    OVERRIDE_VALUE = 0.42
    primary_port = _pick_free_port()
    override_port = _pick_free_port()
    downstream_port = _pick_free_port()
    control_port = _pick_free_port()

    # Short debounce so the test runs in a few seconds. 100 ms = 5
    # ticks @ 50Hz, plenty to absorb one dropped frame without flipping
    # state. Production default is 200 ms.
    override_stale_ms = 100

    proxy_script = MUX_SCRIPT
    cmd = [
        sys.executable,
        str(proxy_script),
        "--primary-host", "127.0.0.1",
        "--primary-port", str(primary_port),
        "--primary-topic", "pose",
        "--out-host", "127.0.0.1",
        "--out-port", str(downstream_port),
        "--out-topic", "pose",
        "--override-host", "127.0.0.1",
        "--override-port", str(override_port),
        "--override-topic", "pose",
        "--override-stale-ms", str(override_stale_ms),
        # Disable frozen-frame release for THIS test: we send the same
        # constant OVERRIDE_VALUE for every frame, which the frozen
        # detector would (correctly) treat as a stationary operator
        # and fire release after ~200 ms -- masking the silence-based
        # release this test is actually exercising. The frozen
        # detector has its own dedicated test below.
        "--override-frozen-ticks", "0",
        # Disable engage-motion hysteresis too: with constant frames
        # the motion streak never increments past 0 and the engage
        # would never fire under the runtime default (10 ticks). 0
        # restores the legacy single-frame engage semantics this
        # silence-based release test was originally written against.
        "--override-engage-motion-ticks", "0",
        "--vla-control-bind-host", "127.0.0.1",
        "--vla-control-port", str(control_port),
        "--vla-control-topic", "vla_control",
        # Large hold/blend so the fallback ladder doesn't accidentally
        # mask the dual-source behaviour we're testing.
        "--rate-hz", "50",
        "--status-every-s", "0.5",
    ]
    proxy_proc = _spawn_proxy(cmd)

    ctx = zmq.Context.instance()
    primary_pub = ctx.socket(zmq.PUB)
    primary_pub.setsockopt(zmq.SNDHWM, 100)
    primary_pub.bind(f"tcp://127.0.0.1:{primary_port}")

    override_pub = ctx.socket(zmq.PUB)
    override_pub.setsockopt(zmq.SNDHWM, 100)
    override_pub.bind(f"tcp://127.0.0.1:{override_port}")

    downstream_sub = ctx.socket(zmq.SUB)
    downstream_sub.setsockopt(zmq.RCVHWM, 1000)
    downstream_sub.connect(f"tcp://127.0.0.1:{downstream_port}")
    downstream_sub.setsockopt(zmq.SUBSCRIBE, b"pose")

    control_sub = ctx.socket(zmq.SUB)
    control_sub.setsockopt(zmq.RCVHWM, 100)
    control_sub.connect(f"tcp://127.0.0.1:{control_port}")
    control_sub.setsockopt(zmq.SUBSCRIBE, b"vla_control")

    # Settle PUB/SUB handshake. Both pose sockets and the control
    # socket need to attach before the proxy starts publishing.
    time.sleep(0.6)

    # Collect events incrementally so we can attribute them to a phase
    # by ordering, but never assert that a specific event MUST land in
    # a specific drain (the debounce window can straddle a drain
    # boundary -- a release event may show up at the end of phase 2's
    # post-loop sleep or at the start of phase 3 depending on exact
    # scheduler timing). We only assert on the total event sequence
    # at the end.
    all_events: list[dict] = []

    try:
        # ---- Phase 1: primary only --------------------------------
        # 0.8 s of primary publishing. Expect forwarded primary frames
        # and zero control events (override SUB is alive but the
        # operator never engaged).
        t_end = time.monotonic() + 0.8
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            time.sleep(0.02)
        time.sleep(0.1)
        primary_only_vals = _drain_pose(
            downstream_sub, time.monotonic() + 0.05
        )
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.05)
        )
        assert len(primary_only_vals) > 10, (
            f"expected >10 primary forwards, got {len(primary_only_vals)}"
        )
        tail = np.array(primary_only_vals[-10:])
        assert np.allclose(tail, PRIMARY_VALUE, atol=1e-6), (
            f"primary phase tail should be PRIMARY_VALUE; got {tail}"
        )
        assert not all_events, (
            f"primary-only phase must emit zero vla_control events; "
            f"got {all_events}"
        )

        # ---- Phase 2: override engaged ---------------------------
        # Publish both primary and override for 0.8 s. Override wins.
        t_end = time.monotonic() + 0.8
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(OVERRIDE_VALUE), zmq.NOBLOCK
            )
            time.sleep(0.02)
        # NOTE: do NOT sleep here -- the post-loop 100 ms sleep in the
        # previous test version straddled the override-stale debounce
        # window and caused the release event to land before our phase
        # 3 drain started. Drain immediately so phase 2's events are
        # captured before the debounce expires.
        override_vals = _drain_pose(
            downstream_sub, time.monotonic() + 0.05
        )
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.05)
        )
        assert len(override_vals) > 10, (
            f"expected >10 override forwards, got {len(override_vals)}"
        )
        tail = np.array(override_vals[-10:])
        assert np.allclose(tail, OVERRIDE_VALUE, atol=1e-6), (
            f"override phase tail should be OVERRIDE_VALUE; got {tail}"
        )
        # Phase 2 should have at minimum recorded the engage event.
        engaged_so_far = [
            e for e in all_events if e.get("event") == "override_engaged"
        ]
        assert len(engaged_so_far) == 1, (
            f"expected exactly 1 override_engaged by end of phase 2; "
            f"got {all_events}"
        )

        # ---- Phase 3: override silent, primary still alive --------
        # Stop publishing override; keep primary alive for 0.8 s. The
        # proxy waits out override_stale_ms (100 ms debounce), emits
        # override_released, and resumes forwarding primary.
        t_end = time.monotonic() + 0.8
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            time.sleep(0.02)
        time.sleep(0.1)
        primary_resume_vals = _drain_pose(
            downstream_sub, time.monotonic() + 0.05
        )
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.05)
        )
        assert len(primary_resume_vals) > 10, (
            f"expected >10 frames after override release, got "
            f"{len(primary_resume_vals)}"
        )
        tail = np.array(primary_resume_vals[-10:])
        assert np.allclose(tail, PRIMARY_VALUE, atol=1e-6), (
            f"resumed-primary tail should be PRIMARY_VALUE; got {tail}"
        )
        # Total event sequence over the whole run: exactly one
        # engage + exactly one release, in that order.
        kinds = [e.get("event") for e in all_events]
        assert kinds == ["override_engaged", "override_released"], (
            f"expected event sequence [engaged, released]; got "
            f"{all_events}"
        )

    finally:
        try:
            proxy_proc.terminate()
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
            proxy_proc.wait(timeout=5)
        primary_pub.close(linger=0)
        override_pub.close(linger=0)
        downstream_sub.close(linger=0)
        control_sub.close(linger=0)
        # Don't ctx.term() -- the fallback-ladder smoke uses the
        # global instance from the same process and would deadlock if
        # we tear it down here.


@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set X2_POSE_PROXY_SMOKE=1 to run (real subprocess + sleeps)",
)
def test_mux_requires_override_port() -> None:
    """The 2026-06-11 mux always merges two sources; ``--override-port``
    is now a required argparse argument. The previous "single-source
    proxy" test (which exercised the proxy with override SUB disabled)
    is no longer meaningful: a deployment that doesn't want manual
    takeover simply doesn't run the mux at all -- the recorder
    publishes straight to the PC2 watchdog.

    This regression pin confirms argparse rejects a mux invocation
    that omits ``--override-port`` so an operator who copies a stale
    launcher recipe gets a fast clear failure instead of a half-up
    process that silently never engages."""
    cmd = [
        sys.executable,
        str(MUX_SCRIPT),
        "--primary-port", "0",
        "--out-port", "0",
        # NB: NO --override-port.
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode != 0, (
        f"mux should refuse to start without --override-port; got "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "override-port" in result.stderr, (
        f"argparse error should mention --override-port; got "
        f"{result.stderr!r}"
    )


@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set X2_POSE_PROXY_SMOKE=1 to run (real subprocess + sleeps)",
)
def test_proxy_override_frozen_release_after_manager_freeze(
    tmp_path: Path,
) -> None:
    """Frozen-frame release: simulates the quest3_manager freeze gesture.

    The 2026-06-10 follow-up adds ``--override-frozen-ticks`` because
    the manager publishes the FROZEN last commanded pose every tick
    when teleop mode is OFF or LOCOMOTION (manager lines 1221-1229).
    The override SUB therefore never goes silent across A+B+X+Y, so
    silence-based release alone (the prior smoke) only fires on full
    Ctrl-C teardown. This test exercises the new path:

      1. Operator drives ARM_MANIPULATION: override frames vary
         per-tick (simulating VR-IK + tracking jitter). Proxy stays
         in OVERRIDE. No release event.
      2. Operator hits A+B+X+Y: manager freezes. We start sending
         IDENTICAL override frames (the manager's frozen pose). After
         N consecutive identical frames (default 10 ticks @ 50Hz =
         200 ms), the proxy fires override_released exactly once,
         WITHOUT the operator having to Ctrl-C anything.
      3. Operator engages again: override frames vary again. Proxy
         re-enters OVERRIDE on the first non-frozen frame, no extra
         CLI dance.

    Pins the engage / frozen-release / re-engage edge sequence with
    a single subprocess proxy. Gated on X2_POSE_PROXY_SMOKE=1 like
    the rest of this file (real ZMQ + sleeps).
    """
    PRIMARY_VALUE = 0.10
    OVERRIDE_BASE = 0.42
    OVERRIDE_FROZEN = 0.55
    primary_port = _pick_free_port()
    override_port = _pick_free_port()
    downstream_port = _pick_free_port()
    control_port = _pick_free_port()

    # Tight knobs to keep the test fast. 5 frozen ticks @ 50Hz = 100 ms
    # of frozen frames triggers release. Tol of 1e-6 means we treat
    # any sub-microradian delta as identical -- comfortably above
    # float32 quantisation but well below VR IK jitter.
    override_frozen_ticks = 5
    override_frozen_tol = 1e-6

    proxy_script = MUX_SCRIPT
    cmd = [
        sys.executable,
        str(proxy_script),
        "--primary-host", "127.0.0.1",
        "--primary-port", str(primary_port),
        "--primary-topic", "pose",
        "--out-host", "127.0.0.1",
        "--out-port", str(downstream_port),
        "--out-topic", "pose",
        "--override-host", "127.0.0.1",
        "--override-port", str(override_port),
        "--override-topic", "pose",
        # Silence-based release: long debounce so we KNOW any release
        # we observe came from the frozen detector, not from a
        # publish-loop scheduling gap. 2000 ms is 100 ticks @ 50Hz --
        # vastly longer than any realistic test pause.
        "--override-stale-ms", "2000",
        "--override-frozen-ticks", str(override_frozen_ticks),
        "--override-frozen-l2-tol", str(override_frozen_tol),
        "--vla-control-bind-host", "127.0.0.1",
        "--vla-control-port", str(control_port),
        "--vla-control-topic", "vla_control",
        "--rate-hz", "50",
        "--status-every-s", "0.5",
    ]
    proxy_proc = _spawn_proxy(cmd)

    ctx = zmq.Context.instance()
    primary_pub = ctx.socket(zmq.PUB)
    primary_pub.setsockopt(zmq.SNDHWM, 100)
    primary_pub.bind(f"tcp://127.0.0.1:{primary_port}")

    override_pub = ctx.socket(zmq.PUB)
    override_pub.setsockopt(zmq.SNDHWM, 100)
    override_pub.bind(f"tcp://127.0.0.1:{override_port}")

    downstream_sub = ctx.socket(zmq.SUB)
    downstream_sub.setsockopt(zmq.RCVHWM, 1000)
    downstream_sub.connect(f"tcp://127.0.0.1:{downstream_port}")
    downstream_sub.setsockopt(zmq.SUBSCRIBE, b"pose")

    control_sub = ctx.socket(zmq.SUB)
    control_sub.setsockopt(zmq.RCVHWM, 100)
    control_sub.connect(f"tcp://127.0.0.1:{control_port}")
    control_sub.setsockopt(zmq.SUBSCRIBE, b"vla_control")

    # PUB/SUB handshake settle (matches the dual-source smoke).
    time.sleep(0.6)

    all_events: list[dict] = []

    try:
        # ---- Phase 1: operator drives ARM_MANIPULATION ------------
        # Send varying override frames for 0.6 s. Even small per-frame
        # deltas (>tol) keep the frozen detector at streak=0 and the
        # proxy locked in OVERRIDE. Engage event fires once at start.
        # We intentionally pulse OVERRIDE_BASE + i*0.01 (much larger
        # than tol) so the detector sees motion every tick.
        t_end = time.monotonic() + 0.6
        i = 0
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(OVERRIDE_BASE + i * 0.01),
                zmq.NOBLOCK,
            )
            i += 1
            time.sleep(0.02)
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.05)
        )
        # Should have exactly one engage event and ZERO releases so far.
        engage_phase1 = [
            e for e in all_events if e.get("event") == "override_engaged"
        ]
        release_phase1 = [
            e for e in all_events
            if e.get("event") == "override_released"
        ]
        assert len(engage_phase1) == 1, (
            f"expected 1 engage event in phase 1; got events={all_events}"
        )
        assert not release_phase1, (
            f"phase 1 (operator actively driving) must NOT release; "
            f"got events={all_events}"
        )

        # ---- Phase 2: A+B+X+Y → manager freezes -------------------
        # Send the SAME OVERRIDE_FROZEN value for 0.6 s. The proxy
        # detects the freeze after override_frozen_ticks (5 ticks =
        # 100 ms) and fires release exactly once.
        t_end = time.monotonic() + 0.6
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(OVERRIDE_FROZEN), zmq.NOBLOCK
            )
            time.sleep(0.02)
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.05)
        )
        release_phase2 = [
            e for e in all_events
            if e.get("event") == "override_released"
        ]
        assert len(release_phase2) == 1, (
            f"expected exactly 1 frozen-release event after manager "
            f"freeze; got events={all_events}"
        )

        # ---- Phase 3: operator re-engages ARM_MANIPULATION --------
        # Resume varying override frames. The frozen latch clears on
        # the first frame above tol, the proxy re-enters OVERRIDE,
        # and a SECOND engage event lands on the control PUB.
        t_end = time.monotonic() + 0.6
        i = 100
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(OVERRIDE_BASE + i * 0.01),
                zmq.NOBLOCK,
            )
            i += 1
            time.sleep(0.02)
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.05)
        )
        engages = [
            e for e in all_events if e.get("event") == "override_engaged"
        ]
        releases = [
            e for e in all_events
            if e.get("event") == "override_released"
        ]
        assert len(engages) == 2, (
            f"expected 2 engage events (phase 1 + phase 3 re-engage); "
            f"got events={all_events}"
        )
        assert len(releases) == 1, (
            f"expected 1 release event (frozen-release at phase 2); "
            f"got events={all_events}"
        )
        # Sanity: the ordering must be engage -> release -> engage.
        events_chrono = [
            e["event"] for e in all_events
            if e.get("event") in ("override_engaged", "override_released")
        ]
        assert events_chrono == [
            "override_engaged",
            "override_released",
            "override_engaged",
        ], (
            f"expected engage -> release -> engage ordering; "
            f"got {events_chrono}"
        )

    finally:
        try:
            proxy_proc.terminate()
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
            proxy_proc.wait(timeout=5)
        primary_pub.close(linger=0)
        override_pub.close(linger=0)
        downstream_sub.close(linger=0)
        control_sub.close(linger=0)


# ===========================================================================
# Engage hysteresis (2026-06-10 follow-up)
# ===========================================================================
@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set X2_POSE_PROXY_SMOKE=1 to run (real subprocess + sleeps)",
)
def test_proxy_override_engage_hysteresis_blocks_single_frame_flicker(
    tmp_path: Path,
) -> None:
    """Engage hysteresis: brief motion must NOT trigger engage.

    Repro of the 2026-06-10 sim observation where the operator's
    controller-rest jitter produced a single non-frozen override
    frame per second, which the legacy proxy treated as "engage on
    the first non-frozen frame", causing:

      tick T:    motion frame  -> engage (cold-restart fires)
      tick T+1:  frozen frame  -> count=1
      ...
      tick T+11: frozen frame  -> count=11 >= threshold -> release
                                  (second cold-restart fires)

    With ``--override-engage-motion-ticks 10`` the proxy now requires
    10 consecutive motion frames before engage; a single brief
    flicker is ignored and the wire stays on primary throughout.
    This test sends 1 motion frame followed by 30 frozen frames and
    asserts no engage / release events were emitted.
    """
    PRIMARY_VALUE = 0.10
    OVERRIDE_FROZEN = 0.42
    OVERRIDE_FLICKER = 0.44
    primary_port = _pick_free_port()
    override_port = _pick_free_port()
    downstream_port = _pick_free_port()
    control_port = _pick_free_port()

    # Tight threshold so the test runs fast. 10 ticks @ 50Hz = 200ms;
    # matches the runtime default.
    engage_motion_threshold = 10

    proxy_script = MUX_SCRIPT
    cmd = [
        sys.executable,
        str(proxy_script),
        "--primary-host", "127.0.0.1",
        "--primary-port", str(primary_port),
        "--primary-topic", "pose",
        "--out-host", "127.0.0.1",
        "--out-port", str(downstream_port),
        "--out-topic", "pose",
        "--override-host", "127.0.0.1",
        "--override-port", str(override_port),
        "--override-topic", "pose",
        "--override-stale-ms", "2000",
        "--override-frozen-ticks", "5",
        "--override-frozen-l2-tol", "1e-6",
        "--override-engage-motion-ticks", str(engage_motion_threshold),
        "--vla-control-bind-host", "127.0.0.1",
        "--vla-control-port", str(control_port),
        "--vla-control-topic", "vla_control",
        "--rate-hz", "50",
        "--status-every-s", "0.5",
    ]
    proxy_proc = _spawn_proxy(cmd)

    ctx = zmq.Context.instance()
    primary_pub = ctx.socket(zmq.PUB)
    primary_pub.setsockopt(zmq.SNDHWM, 100)
    primary_pub.bind(f"tcp://127.0.0.1:{primary_port}")

    override_pub = ctx.socket(zmq.PUB)
    override_pub.setsockopt(zmq.SNDHWM, 100)
    override_pub.bind(f"tcp://127.0.0.1:{override_port}")

    downstream_sub = ctx.socket(zmq.SUB)
    downstream_sub.setsockopt(zmq.RCVHWM, 1000)
    downstream_sub.connect(f"tcp://127.0.0.1:{downstream_port}")
    downstream_sub.setsockopt(zmq.SUBSCRIBE, b"pose")

    control_sub = ctx.socket(zmq.SUB)
    control_sub.setsockopt(zmq.RCVHWM, 100)
    control_sub.connect(f"tcp://127.0.0.1:{control_port}")
    control_sub.setsockopt(zmq.SUBSCRIBE, b"vla_control")

    time.sleep(0.6)

    all_events: list[dict] = []
    try:
        # Phase A: 0.4 s of frozen override frames. Establishes the
        # baseline (no engage; motion_count stays 0).
        t_end = time.monotonic() + 0.4
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(OVERRIDE_FROZEN), zmq.NOBLOCK
            )
            time.sleep(0.02)

        # Phase B: ONE motion frame. With hysteresis motion_count ticks
        # to 1 (< threshold 10) so engage MUST NOT fire. Without
        # hysteresis (legacy), this would engage within one tick and
        # release within 200 ms.
        primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
        override_pub.send(_pack_pose_frame(OVERRIDE_FLICKER), zmq.NOBLOCK)
        time.sleep(0.02)

        # Phase C: 0.5 s frozen again. motion_count resets to 0 on
        # the first frozen tick; frozen detector may trip but since
        # override never engaged, no release event fires either.
        t_end = time.monotonic() + 0.5
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(OVERRIDE_FROZEN), zmq.NOBLOCK
            )
            time.sleep(0.02)

        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.1)
        )
        engages = [
            e for e in all_events
            if e.get("event") == "override_engaged"
        ]
        releases = [
            e for e in all_events
            if e.get("event") == "override_released"
        ]
        assert not engages, (
            f"engage hysteresis broken: single-tick motion flicker "
            f"produced {len(engages)} engage events; events={all_events}"
        )
        assert not releases, (
            f"engage hysteresis broken: spurious release fired "
            f"without preceding engage; events={all_events}"
        )

        # Forward-progress sanity: while engage was blocked, the
        # proxy must keep forwarding PRIMARY to downstream.
        primary_vals = _drain_pose(
            downstream_sub, time.monotonic() + 0.05
        )
        assert primary_vals, "expected proxy to keep forwarding primary"
        tail = np.array(primary_vals[-10:])
        assert np.allclose(tail, PRIMARY_VALUE, atol=1e-6), (
            f"wire should still carry PRIMARY_VALUE during blocked "
            f"engage; got {tail}"
        )

    finally:
        try:
            proxy_proc.terminate()
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
            proxy_proc.wait(timeout=5)
        primary_pub.close(linger=0)
        override_pub.close(linger=0)
        downstream_sub.close(linger=0)
        control_sub.close(linger=0)


# ===========================================================================
# Operator-pose handoff (2026-06-10 follow-up)
# ===========================================================================
@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set X2_POSE_PROXY_SMOKE=1 to run (real subprocess + sleeps)",
)
def test_proxy_override_released_payload_carries_operator_pose(
    tmp_path: Path,
) -> None:
    """override_released must include operator body + hand joints.

    The 2026-06-10 follow-up snapshots the body (``joint_pos_mj``)
    and both hand vectors (``left_hand_joints`` / ``right_hand_joints``)
    from the most recent override frame and embeds them in the event
    payload under ``release_pose``. The bridge uses these to hold the
    wire at the operator's commanded pose during cold-restart
    bridging (avoids snapping to x2_debug's lagged measured pose).
    """
    OVERRIDE_BODY_FROZEN = 0.55
    OVERRIDE_LEFT_FROZEN = 0.61
    OVERRIDE_RIGHT_FROZEN = 0.73
    PRIMARY_VALUE = 0.10
    primary_port = _pick_free_port()
    override_port = _pick_free_port()
    downstream_port = _pick_free_port()
    control_port = _pick_free_port()

    proxy_script = MUX_SCRIPT
    cmd = [
        sys.executable,
        str(proxy_script),
        "--primary-host", "127.0.0.1",
        "--primary-port", str(primary_port),
        "--primary-topic", "pose",
        "--out-host", "127.0.0.1",
        "--out-port", str(downstream_port),
        "--out-topic", "pose",
        "--override-host", "127.0.0.1",
        "--override-port", str(override_port),
        "--override-topic", "pose",
        "--override-stale-ms", "2000",
        "--override-frozen-ticks", "5",
        "--override-frozen-l2-tol", "1e-6",
        # Small engage threshold so phase 1 reliably engages within
        # the 0.6 s motion window. Runtime default is 10.
        "--override-engage-motion-ticks", "3",
        "--vla-control-bind-host", "127.0.0.1",
        "--vla-control-port", str(control_port),
        "--vla-control-topic", "vla_control",
        "--rate-hz", "50",
        "--status-every-s", "0.5",
    ]
    proxy_proc = _spawn_proxy(cmd)

    ctx = zmq.Context.instance()
    primary_pub = ctx.socket(zmq.PUB)
    primary_pub.setsockopt(zmq.SNDHWM, 100)
    primary_pub.bind(f"tcp://127.0.0.1:{primary_port}")

    override_pub = ctx.socket(zmq.PUB)
    override_pub.setsockopt(zmq.SNDHWM, 100)
    override_pub.bind(f"tcp://127.0.0.1:{override_port}")

    downstream_sub = ctx.socket(zmq.SUB)
    downstream_sub.setsockopt(zmq.RCVHWM, 1000)
    downstream_sub.connect(f"tcp://127.0.0.1:{downstream_port}")
    downstream_sub.setsockopt(zmq.SUBSCRIBE, b"pose")

    control_sub = ctx.socket(zmq.SUB)
    control_sub.setsockopt(zmq.RCVHWM, 100)
    control_sub.connect(f"tcp://127.0.0.1:{control_port}")
    control_sub.setsockopt(zmq.SUBSCRIBE, b"vla_control")

    time.sleep(0.6)

    all_events: list[dict] = []
    try:
        # Phase 1: varying body + hand commands ramps engage_motion
        # past threshold and the proxy enters OVERRIDE.
        t_end = time.monotonic() + 0.6
        i = 0
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(
                    0.30 + i * 0.005,
                    left_hand_value=0.20 + i * 0.005,
                    right_hand_value=0.40 + i * 0.005,
                ),
                zmq.NOBLOCK,
            )
            i += 1
            time.sleep(0.02)

        # Phase 2: freeze body + hands. After 5 frozen ticks the
        # release fires with the frozen values as the snapshot.
        t_end = time.monotonic() + 0.6
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(
                    OVERRIDE_BODY_FROZEN,
                    left_hand_value=OVERRIDE_LEFT_FROZEN,
                    right_hand_value=OVERRIDE_RIGHT_FROZEN,
                ),
                zmq.NOBLOCK,
            )
            time.sleep(0.02)

        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.1)
        )

        release_events = [
            e for e in all_events
            if e.get("event") == "override_released"
        ]
        assert len(release_events) == 1, (
            f"expected exactly 1 release event with payload; got "
            f"events={all_events}"
        )
        release_pose = release_events[0].get("release_pose")
        assert isinstance(release_pose, dict), (
            f"release_pose must be a dict; got {release_pose!r}"
        )
        for fname, expected_dim, expected_val in (
            ("joint_pos_mj", proxy.NUM_BODY_DOFS, OVERRIDE_BODY_FROZEN),
            ("left_hand_joints", 10, OVERRIDE_LEFT_FROZEN),
            ("right_hand_joints", 10, OVERRIDE_RIGHT_FROZEN),
        ):
            arr_raw = release_pose.get(fname)
            assert isinstance(arr_raw, list), (
                f"release_pose[{fname!r}] must be a list; got "
                f"{arr_raw!r}"
            )
            arr = np.asarray(arr_raw, dtype=np.float64)
            assert arr.shape == (expected_dim,), (
                f"release_pose[{fname!r}] expected shape "
                f"({expected_dim},); got {arr.shape}"
            )
            assert np.allclose(arr, expected_val, atol=1e-5), (
                f"release_pose[{fname!r}] expected ~{expected_val}; "
                f"got first 5 elems {arr[:5]}"
            )

    finally:
        try:
            proxy_proc.terminate()
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
            proxy_proc.wait(timeout=5)
        primary_pub.close(linger=0)
        override_pub.close(linger=0)
        downstream_sub.close(linger=0)
        control_sub.close(linger=0)


# ===========================================================================
# Strict mode-gated engagement (2026-06-10 follow-up)
#
# The motion-hysteresis / frozen-detection path used to be the only engage
# signal, and it FLICKERS when the operator holds the controller still in
# ARM_MANIPULATION: the manager keeps publishing identical frozen frames
# every tick so the frozen detector trips and releases override back to
# VLA, and any micro-twitch above tol re-engages. The user hit exactly
# this on 2026-06-10 ("the robot keeps flipping between the two states .
# i was just holding the robot in certain pose").
#
# The fix is to wire the proxy directly to the manager's ``stream_mode``
# topic and gate engagement on ``mode != "OFF"`` -- the operator's button
# presses are the source of truth, not joint-space deltas. These tests
# pin the contract:
#
#   1. mode=OFF blocks engagement even with fresh override frames
#      streaming. (today: motion-hysteresis would engage; we don't.)
#   2. mode flipping OFF -> ARM_MANIPULATION engages on the next tick.
#   3. Holding the override frames FROZEN while mode stays
#      ARM_MANIPULATION does NOT release. (the user's actual bug.)
#   4. mode flipping back to OFF releases on the next tick with the
#      operator's last commanded pose in the payload.
#   5. Stale mode signal (manager died) BLOCKS engagement -- fail closed.
# ===========================================================================
def _pack_stream_mode(mode: str, tick: int = 0) -> list[bytes]:
    """Return a multipart frame matching quest3_manager_x2._publish_stream_mode."""
    assert _msgpack is not None, "msgpack required for stream_mode test"
    payload = {"mode": mode, "tick": int(tick), "ts": time.time()}
    return [b"stream_mode", _msgpack.packb(payload, use_bin_type=True)]


@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set X2_POSE_PROXY_SMOKE=1 to run (real subprocess + sleeps)",
)
@pytest.mark.skipif(
    _msgpack is None,
    reason="msgpack not installed; stream_mode SUB requires it",
)
def test_proxy_strict_mode_gate_blocks_off_engages_arm_and_holds_through_freeze(
    tmp_path: Path,
) -> None:
    """STRICT mode: holding a frozen pose in ARM_MANIPULATION does NOT release.

    Reproduces the 2026-06-10 user bug: with motion-hysteresis as the
    engage gate, holding the controller still flickered between OVERRIDE
    and PRIMARY because the frozen detector kept firing release. With
    --teleop-mode-port set, the proxy MUST trust the manager's
    stream_mode broadcast (mode != "OFF") and ignore pose deltas.
    """
    PRIMARY_VALUE = 0.10
    OVERRIDE_VALUE = 0.42
    OVERRIDE_LEFT = 0.21
    OVERRIDE_RIGHT = 0.31
    primary_port = _pick_free_port()
    override_port = _pick_free_port()
    downstream_port = _pick_free_port()
    control_port = _pick_free_port()
    mode_port = _pick_free_port()

    proxy_script = MUX_SCRIPT
    cmd = [
        sys.executable,
        str(proxy_script),
        "--primary-host", "127.0.0.1",
        "--primary-port", str(primary_port),
        "--primary-topic", "pose",
        "--out-host", "127.0.0.1",
        "--out-port", str(downstream_port),
        "--out-topic", "pose",
        "--override-host", "127.0.0.1",
        "--override-port", str(override_port),
        "--override-topic", "pose",
        "--override-stale-ms", "2000",
        # Frozen / motion hysteresis MUST be effectively disabled in
        # the strict-mode path (the implementation bypasses them, but
        # we pass aggressive values that would absolutely flicker
        # under the legacy path so the test reliably distinguishes a
        # broken mode-gate from a working one).
        "--override-frozen-ticks", "3",
        "--override-frozen-l2-tol", "1e-6",
        "--override-engage-motion-ticks", "0",
        # The new strict gate.
        "--teleop-mode-host", "127.0.0.1",
        "--teleop-mode-port", str(mode_port),
        "--teleop-mode-topic", "stream_mode",
        "--teleop-mode-stale-ms", "1000",
        "--vla-control-bind-host", "127.0.0.1",
        "--vla-control-port", str(control_port),
        "--vla-control-topic", "vla_control",
        "--rate-hz", "50",
        "--status-every-s", "0.5",
    ]
    proxy_proc = _spawn_proxy(cmd)

    ctx = zmq.Context.instance()
    primary_pub = ctx.socket(zmq.PUB)
    primary_pub.setsockopt(zmq.SNDHWM, 100)
    primary_pub.bind(f"tcp://127.0.0.1:{primary_port}")

    override_pub = ctx.socket(zmq.PUB)
    override_pub.setsockopt(zmq.SNDHWM, 100)
    override_pub.bind(f"tcp://127.0.0.1:{override_port}")

    mode_pub = ctx.socket(zmq.PUB)
    mode_pub.setsockopt(zmq.SNDHWM, 32)
    mode_pub.bind(f"tcp://127.0.0.1:{mode_port}")

    downstream_sub = ctx.socket(zmq.SUB)
    downstream_sub.setsockopt(zmq.RCVHWM, 1000)
    downstream_sub.connect(f"tcp://127.0.0.1:{downstream_port}")
    downstream_sub.setsockopt(zmq.SUBSCRIBE, b"pose")

    control_sub = ctx.socket(zmq.SUB)
    control_sub.setsockopt(zmq.RCVHWM, 100)
    control_sub.connect(f"tcp://127.0.0.1:{control_port}")
    control_sub.setsockopt(zmq.SUBSCRIBE, b"vla_control")

    time.sleep(0.6)

    all_events: list[dict] = []
    try:
        # ---- Phase 1: mode=OFF, override streaming ----------------
        # 0.8 s of override + primary publishing, with mode=OFF
        # heart-beating every 20 ms. Under STRICT mode, override must
        # be IGNORED -- downstream sees PRIMARY frames only, and no
        # control events fire. Under the legacy motion-hysteresis
        # path this would engage on every changing frame.
        t_end = time.monotonic() + 0.8
        i = 0
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            # Send VARYING override poses so motion-hysteresis WOULD
            # engage under the legacy path. Strict mode must reject.
            override_pub.send(
                _pack_pose_frame(
                    OVERRIDE_VALUE + i * 0.005,
                    left_hand_value=OVERRIDE_LEFT,
                    right_hand_value=OVERRIDE_RIGHT,
                ),
                zmq.NOBLOCK,
            )
            mode_pub.send_multipart(_pack_stream_mode("OFF", tick=i))
            i += 1
            time.sleep(0.02)

        time.sleep(0.1)
        off_vals = _drain_pose(
            downstream_sub, time.monotonic() + 0.05
        )
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.05)
        )
        assert len(off_vals) > 10, (
            f"expected >10 forwards in OFF phase; got {len(off_vals)}"
        )
        tail = np.array(off_vals[-10:])
        assert np.allclose(tail, PRIMARY_VALUE, atol=1e-6), (
            f"OFF phase tail should be PRIMARY={PRIMARY_VALUE}; got {tail}"
        )
        assert not all_events, (
            f"OFF phase must emit zero vla_control events; "
            f"got {all_events}"
        )

        # ---- Phase 2: mode flips to ARM_MANIPULATION --------------
        # Now strict gate should ENGAGE on the next tick that arrives
        # after mode is observed. Send override frames with the
        # CONSTANT FROZEN value -- under motion-hysteresis this would
        # never even engage. Under strict mode, ONE override_engaged
        # event fires and downstream tail becomes OVERRIDE.
        t_end = time.monotonic() + 0.8
        i = 0
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(
                    OVERRIDE_VALUE,
                    left_hand_value=OVERRIDE_LEFT,
                    right_hand_value=OVERRIDE_RIGHT,
                ),
                zmq.NOBLOCK,
            )
            mode_pub.send_multipart(
                _pack_stream_mode("ARM_MANIPULATION", tick=i)
            )
            i += 1
            time.sleep(0.02)

        time.sleep(0.1)
        arm_vals = _drain_pose(
            downstream_sub, time.monotonic() + 0.05
        )
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.05)
        )
        assert len(arm_vals) > 10, (
            f"expected >10 forwards in ARM phase; got {len(arm_vals)}"
        )
        tail = np.array(arm_vals[-10:])
        assert np.allclose(tail, OVERRIDE_VALUE, atol=1e-6), (
            f"ARM phase tail should be OVERRIDE={OVERRIDE_VALUE}; "
            f"got {tail}"
        )
        # Exactly one engage event by now, no releases. CRITICAL:
        # this is the bug regression check -- under the legacy
        # frozen detector, ARM phase with constant frames would
        # release after ~3 frozen ticks and we'd see release events
        # piling up.
        engage_events = [
            e for e in all_events if e.get("event") == "override_engaged"
        ]
        release_events = [
            e for e in all_events if e.get("event") == "override_released"
        ]
        assert len(engage_events) == 1, (
            f"expected exactly 1 engage event in ARM phase; got "
            f"events={all_events}"
        )
        assert not release_events, (
            f"holding override frozen in ARM_MANIPULATION must NOT "
            f"release (the 2026-06-10 user bug regression); got "
            f"events={all_events}"
        )

        # ---- Phase 3: mode flips back to OFF ----------------------
        # Strict gate releases on the next tick with the operator's
        # last commanded pose in the payload. After release the
        # downstream tail returns to PRIMARY.
        t_end = time.monotonic() + 0.5
        i = 0
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(PRIMARY_VALUE), zmq.NOBLOCK)
            # Keep publishing override frames -- the manager doesn't
            # stop publishing them when it flips to OFF (see the
            # manager design notes). The proxy must rely on the mode
            # signal alone, NOT on override silence.
            override_pub.send(
                _pack_pose_frame(
                    OVERRIDE_VALUE,
                    left_hand_value=OVERRIDE_LEFT,
                    right_hand_value=OVERRIDE_RIGHT,
                ),
                zmq.NOBLOCK,
            )
            mode_pub.send_multipart(_pack_stream_mode("OFF", tick=i))
            i += 1
            time.sleep(0.02)

        time.sleep(0.1)
        post_vals = _drain_pose(
            downstream_sub, time.monotonic() + 0.05
        )
        all_events.extend(
            _drain_control(control_sub, time.monotonic() + 0.1)
        )
        tail = np.array(post_vals[-10:])
        assert np.allclose(tail, PRIMARY_VALUE, atol=1e-6), (
            f"post-OFF tail should be PRIMARY={PRIMARY_VALUE}; "
            f"got {tail}"
        )
        release_events = [
            e for e in all_events if e.get("event") == "override_released"
        ]
        assert len(release_events) == 1, (
            f"expected exactly 1 release event after OFF; got "
            f"events={all_events}"
        )
        release_pose = release_events[0].get("release_pose")
        assert isinstance(release_pose, dict), (
            f"release_pose must be a dict; got {release_pose!r}"
        )
        for fname, expected_dim, expected_val in (
            ("joint_pos_mj", proxy.NUM_BODY_DOFS, OVERRIDE_VALUE),
            ("left_hand_joints", 10, OVERRIDE_LEFT),
            ("right_hand_joints", 10, OVERRIDE_RIGHT),
        ):
            arr_raw = release_pose.get(fname)
            assert isinstance(arr_raw, list), (
                f"release_pose[{fname!r}] must be a list; got "
                f"{arr_raw!r}"
            )
            arr = np.asarray(arr_raw, dtype=np.float64)
            assert arr.shape == (expected_dim,), (
                f"release_pose[{fname!r}] shape mismatch; got "
                f"{arr.shape}"
            )
            assert np.allclose(arr, expected_val, atol=1e-5), (
                f"release_pose[{fname!r}] expected ~{expected_val}; "
                f"got first 5 elems {arr[:5]}"
            )

        assert [e.get("event") for e in all_events] == [
            "override_engaged",
            "override_released",
        ], f"unexpected event sequence: {all_events}"

    finally:
        try:
            proxy_proc.terminate()
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
            proxy_proc.wait(timeout=5)
        primary_pub.close(linger=0)
        override_pub.close(linger=0)
        mode_pub.close(linger=0)
        downstream_sub.close(linger=0)
        control_sub.close(linger=0)


@pytest.mark.skipif(
    not SMOKE_ENABLED,
    reason="set X2_POSE_PROXY_SMOKE=1 to run (real subprocess + sleeps)",
)
@pytest.mark.skipif(
    _msgpack is None,
    reason="msgpack not installed; stream_mode SUB requires it",
)
def test_proxy_strict_mode_gate_fails_closed_on_stale_signal(
    tmp_path: Path,
) -> None:
    """STRICT mode: a dead manager (no mode messages) BLOCKS engagement.

    The user explicitly wanted "fail closed" semantics -- if we can't
    confirm the operator is in teleop, we shouldn't engage. Without
    this, a transient mode-SUB disconnect mid-run would silently fall
    back to motion-hysteresis flicker, defeating the whole point.
    """
    primary_port = _pick_free_port()
    override_port = _pick_free_port()
    downstream_port = _pick_free_port()
    control_port = _pick_free_port()
    mode_port = _pick_free_port()

    proxy_script = MUX_SCRIPT
    cmd = [
        sys.executable,
        str(proxy_script),
        "--primary-host", "127.0.0.1",
        "--primary-port", str(primary_port),
        "--primary-topic", "pose",
        "--out-host", "127.0.0.1",
        "--out-port", str(downstream_port),
        "--out-topic", "pose",
        "--override-host", "127.0.0.1",
        "--override-port", str(override_port),
        "--override-topic", "pose",
        "--override-stale-ms", "2000",
        "--override-frozen-ticks", "0",
        "--override-engage-motion-ticks", "0",
        "--teleop-mode-host", "127.0.0.1",
        "--teleop-mode-port", str(mode_port),
        "--teleop-mode-topic", "stream_mode",
        "--teleop-mode-stale-ms", "200",
        "--vla-control-bind-host", "127.0.0.1",
        "--vla-control-port", str(control_port),
        "--vla-control-topic", "vla_control",
        "--rate-hz", "50",
        "--status-every-s", "0.5",
    ]
    proxy_proc = _spawn_proxy(cmd)

    ctx = zmq.Context.instance()
    primary_pub = ctx.socket(zmq.PUB)
    primary_pub.setsockopt(zmq.SNDHWM, 100)
    primary_pub.bind(f"tcp://127.0.0.1:{primary_port}")

    override_pub = ctx.socket(zmq.PUB)
    override_pub.setsockopt(zmq.SNDHWM, 100)
    override_pub.bind(f"tcp://127.0.0.1:{override_port}")

    # Mode PUB binds but NEVER publishes -- simulates a dead manager.
    mode_pub = ctx.socket(zmq.PUB)
    mode_pub.setsockopt(zmq.SNDHWM, 32)
    mode_pub.bind(f"tcp://127.0.0.1:{mode_port}")

    downstream_sub = ctx.socket(zmq.SUB)
    downstream_sub.setsockopt(zmq.RCVHWM, 1000)
    downstream_sub.connect(f"tcp://127.0.0.1:{downstream_port}")
    downstream_sub.setsockopt(zmq.SUBSCRIBE, b"pose")

    control_sub = ctx.socket(zmq.SUB)
    control_sub.setsockopt(zmq.RCVHWM, 100)
    control_sub.connect(f"tcp://127.0.0.1:{control_port}")
    control_sub.setsockopt(zmq.SUBSCRIBE, b"vla_control")

    time.sleep(0.6)

    try:
        # Stream both primary AND override for 1.0 s with NO mode
        # messages. Strict gate must block engagement -- downstream
        # sees PRIMARY only, no control events.
        t_end = time.monotonic() + 1.0
        i = 0
        while time.monotonic() < t_end:
            primary_pub.send(_pack_pose_frame(0.10), zmq.NOBLOCK)
            override_pub.send(
                _pack_pose_frame(0.42 + i * 0.005),
                zmq.NOBLOCK,
            )
            i += 1
            time.sleep(0.02)

        time.sleep(0.1)
        vals = _drain_pose(downstream_sub, time.monotonic() + 0.05)
        events = _drain_control(control_sub, time.monotonic() + 0.05)

        assert len(vals) > 10, f"expected >10 forwards; got {len(vals)}"
        tail = np.array(vals[-10:])
        assert np.allclose(tail, 0.10, atol=1e-6), (
            f"stale-mode tail must be PRIMARY=0.10 (engagement blocked); "
            f"got {tail}"
        )
        assert not events, (
            f"stale mode signal must block all engagement events "
            f"(fail closed); got {events}"
        )

    finally:
        try:
            proxy_proc.terminate()
            proxy_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proxy_proc.kill()
            proxy_proc.wait(timeout=5)
        primary_pub.close(linger=0)
        override_pub.close(linger=0)
        mode_pub.close(linger=0)
        downstream_sub.close(linger=0)
        control_sub.close(linger=0)
