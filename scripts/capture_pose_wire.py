"""Capture N frames from a pose ZMQ topic and dump them to .npz for diffing.

Subscribe to a packed-message PUB (default tcp://localhost:5556 topic 'pose')
and capture the next N frames. Save each frame's joint_pos_mj /
root_quat_xyzw / future-window fields to a .npz so we can byte-diff
two runs offline.

Usage::

    .venv/bin/python scripts/capture_pose_wire.py \\
        --port 5556 --topic pose --n 100 --out /tmp/pose_planner.npz

    .venv/bin/python scripts/capture_pose_wire.py \\
        --port 5556 --topic pose --n 100 --out /tmp/pose_bridge.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--topic", default="pose")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (
        unpack_message,
    )

    url = f"tcp://{args.host}:{args.port}"
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    sock.setsockopt(zmq.RCVHWM, 100)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(url)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    import time as _time
    _time.sleep(0.3)
    print(f"connected to {url} topic={args.topic!r}, capturing {args.n} frames...")

    frames: list[dict] = []
    captured = 0
    while captured < args.n:
        events = dict(poller.poll(5000))
        if sock not in events:
            print(f"timeout after {captured} frames")
            break
        raw = sock.recv()
        try:
            decoded = unpack_message(raw, expected_topic=args.topic)
        except ValueError as exc:
            print(f"decode error: {exc}")
            continue
        f = {k: np.asarray(v) for k, v in decoded.fields.items()}
        frames.append(f)
        captured += 1

    if not frames:
        print("no frames captured")
        return 1

    print(f"captured {len(frames)} frames; first frame fields:")
    for k, v in frames[0].items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    out_dict: dict = {}
    for k in frames[0].keys():
        try:
            stacked = np.stack([f[k] for f in frames])
            out_dict[k] = stacked
        except (KeyError, ValueError) as exc:
            print(f"  skipping {k}: {exc}")
    np.savez(args.out, **out_dict)
    print(f"saved {len(out_dict)} fields to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
