#!/usr/bin/env python3
"""Validate that the PC2 head-camera ZMQ bridge is publishing real frames.

Preflight gate for ``run_x2_quest3_planner_stack.sh --with-record
--head-cameras`` (added 2026-07-28 after a session where the bridge
process started cleanly while the Orbbec em-app was down -- ``serve``
succeeding proves the *bridge* is up, not that any camera behind it is
delivering frames, and a dataset recorded in that state silently lacks
the head_front stream the VLA features schema expects).

Subscribes to the bridge's PUB socket (single-part msgpack payloads,
``{"timestamps": {key: t}, "images": {key: jpeg_bytes}}`` -- see
``gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py``) and
requires ``--min-frames`` VALID frames per required mount key within
``--timeout`` seconds. A frame counts only if its bytes carry the JPEG
SOI magic -- a key that appears with empty/garbage payloads still
fails the gate.

head_front (the Orbbec Gemini 335) is the stream that matters most for
VLA datasets; it is always required. The IMX900 stereo pair defaults to
required as well because the recorder's feature schema declares all
three streams -- pass ``--require head_front`` to gate on the Orbbec
alone (e.g. while the stereo HAL is known-degraded and you accept the
schema consequences).

Exit codes: 0 = every required key flowing; 1 = gate failed (per-key
detail + the matching recovery command printed to stderr); 2 = bad args.
"""

from __future__ import annotations

import argparse
import sys
import time

import msgpack
import zmq

DEFAULT_REQUIRED = "head_front,stereo_left,stereo_right"

# Per-key recovery hints, printed on failure so the operator gets the
# exact next command instead of a generic "check the cameras".
_RECOVERY = {
    "head_front": (
        "Orbbec stream down. Check `x2_pc2_cameras.sh status`; if the "
        "orbbec_camera em-app is stopped: ssh PC2 `aima em start-app "
        "orbbec_camera` (it must be RUNNING for the bridge's ROS topic)."
    ),
    "stereo_left": (
        "IMX900 stereo stream down -- the boot-time Argus race. Run "
        "`x2_pc2_cameras.sh restart-hal` and re-probe."
    ),
    "stereo_right": (
        "IMX900 stereo stream down -- the boot-time Argus race. Run "
        "`x2_pc2_cameras.sh restart-hal` and re-probe."
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
    )
    ap.add_argument("--host", required=True, help="PC2 camera bridge host")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument(
        "--timeout", type=float, default=15.0,
        help="seconds to wait for the required frame counts (default 15; "
             "covers bridge ROS-discovery warm-up after a fresh serve)",
    )
    ap.add_argument(
        "--min-frames", type=int, default=3,
        help="valid frames required per key (default 3 -- more than one "
             "so a cached late-joiner frame can't pass the gate)",
    )
    ap.add_argument(
        "--require", default=DEFAULT_REQUIRED,
        help=f"comma-separated mount keys to gate on "
             f"(default {DEFAULT_REQUIRED})",
    )
    args = ap.parse_args()

    required = [k.strip() for k in args.require.split(",") if k.strip()]
    if not required:
        print("[camera-probe] --require resolved to an empty list", file=sys.stderr)
        return 2

    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.RCVTIMEO = 500
    sub.connect(f"tcp://{args.host}:{args.port}")

    counts: dict[str, int] = {}
    first_seen: dict[str, float] = {}
    last_seen: dict[str, float] = {}
    t0 = time.monotonic()
    deadline = t0 + args.timeout
    try:
        while time.monotonic() < deadline:
            try:
                raw = sub.recv()
            except zmq.Again:
                continue
            try:
                payload = msgpack.unpackb(raw, raw=False)
                images = payload.get("images") or {}
            except Exception:
                continue  # partial/foreign message; keep listening
            now = time.monotonic()
            for key, jpeg in images.items():
                # JPEG SOI magic: an empty or garbage buffer must not count.
                if not isinstance(jpeg, (bytes, bytearray)) or len(jpeg) < 4:
                    continue
                if jpeg[0] != 0xFF or jpeg[1] != 0xD8:
                    continue
                counts[key] = counts.get(key, 0) + 1
                first_seen.setdefault(key, now)
                last_seen[key] = now
            if all(counts.get(k, 0) >= args.min_frames for k in required):
                break
    finally:
        sub.close(0)
        ctx.term()

    ok = True
    for key in required:
        n = counts.get(key, 0)
        if n >= args.min_frames:
            span = max(last_seen[key] - first_seen[key], 1e-6)
            hz = (n - 1) / span if n > 1 else 0.0
            print(f"[camera-probe] OK   {key}: {n} frames (~{hz:.1f} Hz)")
        else:
            ok = False
            print(f"[camera-probe] FAIL {key}: {n}/{args.min_frames} valid "
                  f"frames in {args.timeout:.0f}s", file=sys.stderr)
            hint = _RECOVERY.get(key)
            if hint:
                print(f"[camera-probe]      -> {hint}", file=sys.stderr)
    extra = sorted(set(counts) - set(required))
    if extra:
        print(f"[camera-probe] also flowing (not gated): {', '.join(extra)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
