"""Live wifi-gap + deploy-state monitor for the laptop -> PC2 pose link.

Subscribes to PC2's ``x2_debug`` stream and watches the four fields
that tell us, at 50 Hz, exactly how the deploy sees the laptop:

  * ``pose_ref_age_s``  -- age of the last ``pose`` frame the deploy
                           received from the recorder. ``-1.0`` is the
                           sentinel for "never received any frame". A
                           healthy 50 Hz wire sits around 20 ms.
  * ``ramp_alpha``      -- soft-start ramp coefficient (0=motors slack,
                           1=full policy authority). Any sub-second dip
                           below ~0.99 is what the operator hears as
                           "motors unlock then lock".
  * ``in_safe_idle``    -- 1 if the pose-ref watchdog tripped into
                           SAFE_IDLE (motors held at default_angles +
                           4x kd; policy NOT running).
  * ``control_tick``    -- monotonic 500 Hz tick counter; gaps in this
                           are dropped CONTROL ticks (very rare).

For each tick we ALSO compute the inter-frame interval on the wire
(``recv_dt_ms``) so a wifi packet drop shows up as a single 40 ms or
80 ms hiccup (vs. the steady 20 ms expected).

Spike thresholds are tight by default so the operator sees every
event:

  --pose-age-warn-ms       150  (default 0.150s -- ~7 dropped pose ticks)
  --recv-gap-warn-ms        50  (default 50 ms wifi inter-frame gap)
  --alpha-dip-warn         0.95 (any alpha dip below this for one tick)

Output legend:
  [TICK]   normal frame (rate-limited; 1 line/s)
  [POSE  ] pose_ref_age crossed warn threshold
  [WIRE  ] inter-frame wifi gap > warn threshold (the wifi missed packet(s))
  [ALPHA ] ramp_alpha dipped below threshold (motor unlock symptom)
  [SAFE  ] in_safe_idle flipped (motors held, policy paused)
  [SUMMRY] 1-s rate + max-pose-age + min-alpha + wire-gap stats per second
  [HOLD  ] x2_debug stream went silent (PC2 unreachable / deploy dead)

Run alongside the planner stack:

    .venv/bin/python -m gear_sonic_deploy.scripts.x2_wifi_gap_monitor \
        --pc2-host 192.168.86.32 | tee /tmp/wifi_gap.log
"""

from __future__ import annotations

import argparse
import math
import signal
import statistics
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--pc2-host", default="192.168.86.32",
        help="PC2 host (deploy publishes x2_debug here on tcp://HOST:5557).",
    )
    p.add_argument(
        "--pc2-port", type=int, default=5557,
        help="PC2 x2_debug PUB port.",
    )
    p.add_argument("--topic", default="x2_debug")
    p.add_argument(
        "--pose-age-warn-ms", type=float, default=150.0,
        help="Print [POSE  ] event when pose_ref_age crosses above this "
             "(default 150 ms ~= 7 dropped 20 ms pose ticks).",
    )
    p.add_argument(
        "--recv-gap-warn-ms", type=float, default=50.0,
        help="Print [WIRE  ] event when consecutive x2_debug frames are "
             "more than this many ms apart (= wifi packet drop on the "
             "telemetry side; correlates with pose-side drops).",
    )
    p.add_argument(
        "--alpha-dip-warn", type=float, default=0.95,
        help="Print [ALPHA ] event when ramp_alpha drops below this "
             "(default 0.95 = motors briefly unlocked).",
    )
    p.add_argument(
        "--summary-every-s", type=float, default=1.0,
        help="Print [SUMMRY] line every N seconds.",
    )
    return p.parse_args()


@dataclass
class State:
    sock: zmq.Socket
    topic_bytes: bytes
    prev_t_mono: Optional[float] = None
    prev_pose_age: Optional[float] = None
    prev_alpha: Optional[float] = None
    prev_safe_idle: Optional[int] = None
    prev_tick: Optional[int] = None

    # Per-second rolling stats (reset on each SUMMRY).
    last_summary_t: float = field(default_factory=time.monotonic)
    last_message_t: float = field(default_factory=time.monotonic)
    holding: bool = False
    frame_count: int = 0
    recv_dt_ms_list: list[float] = field(default_factory=list)
    pose_age_max_ms: float = 0.0
    pose_age_avg_accum: float = 0.0
    pose_age_avg_n: int = 0
    alpha_min: float = 1.0
    safe_idle_ticks: int = 0
    tick_gaps: int = 0  # number of control_tick non-monotone or skip events


def _now_str() -> str:
    return time.strftime("%H:%M:%S") + f".{int((time.time()%1)*1000):03d}"


def main() -> int:
    args = parse_args()
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.SUB)
    s.setsockopt(zmq.RCVHWM, 200)
    s.setsockopt(zmq.LINGER, 0)
    url = f"tcp://{args.pc2_host}:{args.pc2_port}"
    s.connect(url)
    s.setsockopt(zmq.SUBSCRIBE, args.topic.encode())

    print(f"[wire] x2_debug SUB {url} topic={args.topic!r}", flush=True)
    print(
        f"[cfg]  pose_age_warn={args.pose_age_warn_ms:.0f}ms  "
        f"recv_gap_warn={args.recv_gap_warn_ms:.0f}ms  "
        f"alpha_dip_warn={args.alpha_dip_warn:.2f}",
        flush=True,
    )
    print("[ready] monitor live; teleop now. Ctrl-C to stop.", flush=True)

    st = State(sock=s, topic_bytes=args.topic.encode())

    stop = {"flag": False}
    def _sig(*_a):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    poller = zmq.Poller()
    poller.register(s, zmq.POLLIN)

    while not stop["flag"]:
        try:
            events = dict(poller.poll(timeout=200))
        except (KeyboardInterrupt, zmq.error.ZMQError):
            break

        now = time.monotonic()

        if events:
            try:
                msg = s.recv(zmq.NOBLOCK)
            except zmq.error.Again:
                msg = None
            if msg is not None:
                payload = msg
                if payload.startswith(st.topic_bytes):
                    payload = payload[len(st.topic_bytes):]
                    if payload.startswith(b" "):
                        payload = payload[1:]
                try:
                    decoded = unpack_message(payload, expected_topic=None)
                except ValueError as exc:
                    print(f"[warn] decode failed: {exc}", flush=True)
                    decoded = None

                if decoded is not None:
                    f = decoded.fields
                    pose_age_s = float(f.get("pose_ref_age_s", np.array([-1.0]))[0])
                    ramp_alpha = float(f.get("ramp_alpha", np.array([1.0]))[0])
                    in_safe_idle = int(f.get("in_safe_idle", np.array([0], dtype=np.uint8))[0])
                    tick = int(f.get("control_tick", np.array([-1], dtype=np.int64))[0])

                    # WIRE-side gap.
                    if st.prev_t_mono is not None:
                        dt_ms = (now - st.prev_t_mono) * 1e3
                        st.recv_dt_ms_list.append(dt_ms)
                        if dt_ms > args.recv_gap_warn_ms:
                            print(
                                f"[WIRE  ] {_now_str()} x2_debug inter-frame gap {dt_ms:5.1f}ms  "
                                f"(pose_age={pose_age_s*1e3:7.1f}ms alpha={ramp_alpha:.3f} "
                                f"safe_idle={in_safe_idle} tick={tick})",
                                flush=True,
                            )
                    st.prev_t_mono = now

                    # POSE-side age.
                    age_ms_for_stats = (
                        pose_age_s * 1e3 if pose_age_s >= 0.0 else float("nan")
                    )
                    if pose_age_s >= 0.0:
                        st.pose_age_max_ms = max(st.pose_age_max_ms, age_ms_for_stats)
                        st.pose_age_avg_accum += age_ms_for_stats
                        st.pose_age_avg_n += 1
                        prev_age_ms = (
                            st.prev_pose_age * 1e3 if st.prev_pose_age is not None and st.prev_pose_age >= 0.0
                            else None
                        )
                        crossed_up = (
                            age_ms_for_stats > args.pose_age_warn_ms
                            and (prev_age_ms is None or prev_age_ms <= args.pose_age_warn_ms)
                        )
                        if crossed_up:
                            print(
                                f"[POSE  ] {_now_str()} pose_ref_age={age_ms_for_stats:7.1f}ms "
                                f"crossed above warn {args.pose_age_warn_ms:.0f}ms  "
                                f"(alpha={ramp_alpha:.3f} safe_idle={in_safe_idle} tick={tick})",
                                flush=True,
                            )
                    else:
                        # pose_ref_age is the -1.0 sentinel (never received).
                        if (
                            st.prev_pose_age is not None
                            and st.prev_pose_age >= 0.0
                        ):
                            print(
                                f"[POSE  ] {_now_str()} pose_ref_age REVERTED to -1 sentinel "
                                f"(=deploy lost the source entirely)",
                                flush=True,
                            )
                    st.prev_pose_age = pose_age_s

                    # ALPHA dip.
                    st.alpha_min = min(st.alpha_min, ramp_alpha)
                    if (
                        ramp_alpha < args.alpha_dip_warn
                        and (st.prev_alpha is None or st.prev_alpha >= args.alpha_dip_warn)
                    ):
                        print(
                            f"[ALPHA ] {_now_str()} ramp_alpha DIPPED to {ramp_alpha:.3f} "
                            f"(below {args.alpha_dip_warn:.2f}; motors briefly unlocked)  "
                            f"(pose_age={pose_age_s*1e3:7.1f}ms safe_idle={in_safe_idle} tick={tick})",
                            flush=True,
                        )
                    elif (
                        ramp_alpha >= args.alpha_dip_warn
                        and st.prev_alpha is not None and st.prev_alpha < args.alpha_dip_warn
                    ):
                        print(
                            f"[ALPHA ] {_now_str()} ramp_alpha RECOVERED to {ramp_alpha:.3f} "
                            f"(was below {args.alpha_dip_warn:.2f}; motors re-locked)",
                            flush=True,
                        )
                    st.prev_alpha = ramp_alpha

                    # SAFE_IDLE transitions.
                    if st.prev_safe_idle is None or in_safe_idle != st.prev_safe_idle:
                        if st.prev_safe_idle is not None:
                            print(
                                f"[SAFE  ] {_now_str()} in_safe_idle "
                                f"{st.prev_safe_idle} -> {in_safe_idle}  "
                                f"(pose_age={pose_age_s*1e3:7.1f}ms alpha={ramp_alpha:.3f})",
                                flush=True,
                            )
                        st.prev_safe_idle = in_safe_idle
                    if in_safe_idle:
                        st.safe_idle_ticks += 1

                    # control_tick monotonicity.
                    if st.prev_tick is not None and tick >= 0:
                        if tick <= st.prev_tick:
                            st.tick_gaps += 1
                            print(
                                f"[TICK!] {_now_str()} control_tick reversed: "
                                f"{st.prev_tick} -> {tick}",
                                flush=True,
                            )
                    st.prev_tick = tick
                    st.last_message_t = now
                    if st.holding:
                        print(f"[RESUME] {_now_str()} x2_debug stream resumed", flush=True)
                        st.holding = False
                    st.frame_count += 1

        # Silence detection.
        if now - st.last_message_t > 1.0 and not st.holding:
            print(
                f"[HOLD  ] {_now_str()} x2_debug silent {now - st.last_message_t:.1f}s",
                flush=True,
            )
            st.holding = True

        # Periodic summary.
        since = now - st.last_summary_t
        if since >= args.summary_every_s:
            hz = st.frame_count / since if since > 0 else 0.0
            if st.recv_dt_ms_list:
                arr = st.recv_dt_ms_list
                dt_max = max(arr)
                dt_p99 = statistics.quantiles(arr, n=100)[98] if len(arr) >= 100 else dt_max
                dt_avg = sum(arr) / len(arr)
            else:
                dt_max = dt_p99 = dt_avg = float("nan")
            pose_avg = (
                st.pose_age_avg_accum / st.pose_age_avg_n
                if st.pose_age_avg_n else float("nan")
            )
            print(
                f"[SUMMRY] {_now_str()} "
                f"x2_debug rate={hz:5.1f}Hz  "
                f"wire_dt(ms) avg={dt_avg:5.1f} p99={dt_p99:5.1f} max={dt_max:5.1f}  "
                f"pose_age(ms) avg={pose_avg:7.1f} max={st.pose_age_max_ms:7.1f}  "
                f"alpha_min={st.alpha_min:.3f}  safe_idle_ticks={st.safe_idle_ticks}  "
                f"tick_reversals={st.tick_gaps}",
                flush=True,
            )
            st.frame_count = 0
            st.recv_dt_ms_list = []
            st.pose_age_max_ms = 0.0
            st.pose_age_avg_accum = 0.0
            st.pose_age_avg_n = 0
            st.alpha_min = 1.0
            st.safe_idle_ticks = 0
            st.tick_gaps = 0
            st.last_summary_t = now

    print("[stop] shutting down monitor", flush=True)
    try:
        s.close(linger=0)
        ctx.term()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
