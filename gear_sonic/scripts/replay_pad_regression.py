#!/usr/bin/env python3
"""Replay a scripted gamepad session against a running stack -- the deploy regression.

Design
------
This script does **only one thing**: put the same messages on the wire that
``pad_locomotion_bridge.py`` puts there when a human drives the pad. Everything
else -- idle fallback, blend windows, clip->idle composition, the 30->50 Hz
handoff -- is the *stack's* job and is deliberately NOT reimplemented here.

That is the whole point. If the regression stitched its own reference track, it
would be testing the harness's blending instead of the robot's, and a bug in the
deployed state machine would be invisible to the test meant to catch it. Sonic
also never resets mid-demo and cannot hold a frozen frame, so the stream must
stay continuous and gaps must be filled by the stack's own IDLE_LOOP fallback --
exactly as on the robot.

So: launch the stack (sim or robot), then run this. What you watch in the viewer
is the real deployed path end to end.

Wire (mirrors pad_locomotion_bridge.py exactly):
  planner_cmd     :5563  {"intent":"locomotion","magnitude":"continuous",
                          "stick_fwd":f,"stick_side":s,"stick_yaw":y}
  motion_clip_cmd :5568  {"action":"play","pkl":...,"kind":"locomotion","motion_key":k}
                         {"action":"stop"}

Usage:
    python gear_sonic/scripts/replay_pad_regression.py \
        --clip-pkl gear_sonic/data/motions/x2_dances_easy.pkl \
        --dances dance_party_hips_003__A467,dance_freedom_wheels_001__A465
"""
from __future__ import annotations

import argparse
import json
import time

import zmq

STICK_HZ = 50.0  # rate we re-publish a held stick, mimicking a human holding it


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5563, help="planner_cmd")
    ap.add_argument("--topic", default="planner_cmd")
    ap.add_argument("--clip-host", default="127.0.0.1")
    ap.add_argument("--clip-port", type=int, default=5568)
    ap.add_argument("--clip-topic", default="motion_clip_cmd")
    ap.add_argument("--clip-pkl", default="gear_sonic/data/motions/x2_dances_easy.pkl")
    ap.add_argument("--dances", default="", help="comma-separated motion keys")
    ap.add_argument("--walk-speed", type=float, default=0.5, help="stick_fwd for walks")
    ap.add_argument("--walk-seconds", type=float, default=8.0)
    ap.add_argument("--dance-seconds", type=float, default=12.0, help="per dance")
    ap.add_argument("--settle-seconds", type=float, default=3.0,
                    help="idle between items; the stack holds IDLE_LOOP here")
    ap.add_argument("--lead-in", type=float, default=5.0,
                    help="idle before the first item (let sonic settle)")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, send nothing")
    ap.add_argument("--bind", action="store_true", default=True,
                    help="BIND planner_cmd instead of connecting. Required when no pad "
                         "bridge is running: in --pad-only the bridge is the binder and "
                         "the stack SUBs, so with no pad nothing binds and sends vanish. "
                         "Use --no-bind to sit alongside a running pad bridge.")
    ap.add_argument("--no-bind", dest="bind", action="store_false")
    args = ap.parse_args()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    if args.bind:
        sock.bind(f"tcp://*:{args.port}")
    else:
        sock.connect(f"tcp://{args.host}:{args.port}")
    clip = ctx.socket(zmq.PUB)
    clip.setsockopt(zmq.SNDHWM, 10)
    clip.connect(f"tcp://{args.clip_host}:{args.clip_port}")
    time.sleep(0.3)  # PUB/SUB handshake before the first send, else it is dropped

    def send_stick(f: float, s: float, y: float) -> None:
        if args.dry_run:
            return
        sock.send_multipart([args.topic.encode("ascii"), json.dumps({
            "intent": "locomotion", "magnitude": "continuous",
            "stick_fwd": round(f, 3), "stick_side": round(s, 3),
            "stick_yaw": round(y, 3)}).encode("utf-8")])

    def send_clip(payload: dict) -> None:
        if args.dry_run:
            return
        clip.send_multipart([args.clip_topic.encode("ascii"),
                             json.dumps(payload).encode("utf-8")])

    def hold_stick(f: float, s: float, y: float, seconds: float) -> None:
        """Hold a stick position, republishing at STICK_HZ like a human would."""
        end = time.time() + seconds
        while time.time() < end:
            send_stick(f, s, y)
            time.sleep(1.0 / STICK_HZ)

    def idle(seconds: float, label: str) -> None:
        """Release the stick. The stack falls back to its IDLE_LOOP -- we do
        NOT synthesize an idle here; watching that fallback IS the test."""
        banner(f"idle / settle ({label})", seconds)
        send_stick(0.0, 0.0, 0.0)
        time.sleep(seconds)

    t0 = time.time()
    step = {"n": 0}

    def banner(what: str, seconds: float) -> None:
        step["n"] += 1
        print(f"\n[{time.time() - t0:7.1f}s] === STEP {step['n']}: {what} "
              f"({seconds:.0f}s) ===", flush=True)

    dances = [d for d in args.dances.split(",") if d.strip()]

    print("=" * 68)
    print("  PAD REGRESSION REPLAY -- mimics the gamepad; the stack composes.")
    print(f"  walks @ stick_fwd={args.walk_speed}   dances: {len(dances)}")
    print("  Watch the viewer; this stdout is your timeline.")
    print("=" * 68)

    idle(args.lead_in, "lead-in, sonic settling")

    banner(f"WALK straight, stick_fwd={args.walk_speed}", args.walk_seconds)
    hold_stick(args.walk_speed, 0.0, 0.0, args.walk_seconds)
    idle(args.settle_seconds, "after straight walk")

    banner(f"WALK + TURN left, fwd={args.walk_speed} yaw=+0.4", args.walk_seconds)
    hold_stick(args.walk_speed, 0.0, 0.4, args.walk_seconds)
    idle(args.settle_seconds, "after left turn")

    banner(f"WALK + TURN right, fwd={args.walk_speed} yaw=-0.4", args.walk_seconds)
    hold_stick(args.walk_speed, 0.0, -0.4, args.walk_seconds)
    idle(args.settle_seconds, "after right turn")

    for i, key in enumerate(dances, 1):
        banner(f"DANCE {i}/{len(dances)}: {key}", args.dance_seconds)
        send_clip({"action": "play", "pkl": args.clip_pkl,
                   "kind": "locomotion", "motion_key": key})
        time.sleep(args.dance_seconds)
        send_clip({"action": "stop"})
        idle(args.settle_seconds, f"after dance {i}")

    print(f"\n[{time.time() - t0:7.1f}s] === REPLAY COMPLETE -- "
          f"{step['n']} steps. Stack left in IDLE_LOOP. ===")
    print("  Verdict: GO only if every walk, turn, dance AND every seam was clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
