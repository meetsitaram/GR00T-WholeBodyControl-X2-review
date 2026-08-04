#!/usr/bin/env python3
"""Capture raw joint states through a STOCK MC damping activation.

Purpose (2026-08-04): our operator e-stop's stage-2 damping (Kp=0,
Kd=8, G1 values) works but the X2 motors audibly complain — a loud
whir/growl during the slump. The vendor MC has its own damping mode
that presumably sounds/behaves better on this hardware. This tool
records high-rate joint pos/vel/effort while the operator activates
the STOCK damping, so we can fit the vendor's effective per-joint
damping profile:

    tau_j ~= -Kd_j * vel_j   (+ any Kp toward a floor pose, ramps,
                              or per-group differences)

and retune our SafetyStack damping to match.

Runs ON PC2 (needs system rclpy + aimdk_msgs):

    /home/run/gear-sonic/venv/bin/python /home/run/gear-sonic/x2_damp_capture.py \
        --seconds 30 --out /home/run/gear-sonic/log/damp_capture

Protocol:
  1. Robot standing under MC (SONIC ritual NOT running, or after RAMP_OUT).
  2. Start this script; it prints ARMED once states flow.
  3. Trigger the stock MC damping / protect mode (vendor remote or app).
  4. Let the robot finish slumping; script auto-stops after --seconds.

Output: one JSONL per run, fsync'd twice a second, AND live-streamed
over ZMQ PUB (--pub-port, default 5599) so a laptop-side mirror holds
every row up to the last wifi packet — the 2026-08-04 battery-pull
postmortem lost the final ~60 s of every on-robot log; disk syncing
alone still loses the last half-second, the wire does not.

Laptop mirror (run BEFORE triggering the damping):

    python3 gear_sonic_deploy/scripts/x2_damp_capture.py \
        --mirror tcp://${X2_PC2_HOST}:5599 --out ~/x2_damp_captures

Analyze either side's file with:

    python3 x2_damp_capture.py --analyze <capture.jsonl>

which reports, per joint: vel/effort peaks and a least-squares Kd fit
over the high-velocity window (|vel| > 0.3 rad/s).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


def analyze(path: str) -> int:
    import numpy as np

    rows = []
    for line in open(path, errors="replace"):
        line = line.strip().strip("\x00")
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("kind") == "sample":
            rows.append(r)
    if not rows:
        print("no samples in", path)
        return 1
    names = rows[0]["names"]
    t = np.array([r["t"] for r in rows])
    vel = np.array([r["vel"] for r in rows])
    eff = np.array([r["eff"] for r in rows])
    print(f"{len(rows)} samples over {t[-1] - t[0]:.1f}s, {len(names)} joints")
    print(f"{'joint':<28} {'|vel|max':>8} {'|eff|max':>8} "
          f"{'Kd_fit':>7} {'pts':>5}")
    for j, name in enumerate(names):
        v = vel[:, j]
        e = eff[:, j]
        mask = np.abs(v) > 0.3
        if mask.sum() >= 10:
            # tau = -Kd * vel  ->  Kd = -<tau, vel> / <vel, vel>
            kd = -float(np.dot(e[mask], v[mask]) / np.dot(v[mask], v[mask]))
            kd_s = f"{kd:7.2f}"
        else:
            kd_s = "      -"
        print(f"{name:<28} {np.abs(v).max():8.3f} {np.abs(e).max():8.2f} "
              f"{kd_s} {int(mask.sum()):5d}")
    return 0


def mirror(endpoint: str, out_dir: str) -> int:
    """Laptop-side: SUB the capture PUB and persist every row locally.

    Survives robot battery pulls by construction — rows live here the
    moment they cross the wifi. Ctrl-C to stop; auto-stops after 60 s
    of silence once rows have flowed (capture ended)."""
    import zmq

    os.makedirs(os.path.expanduser(out_dir), exist_ok=True)
    path = os.path.join(
        os.path.expanduser(out_dir),
        f"damp_mirror_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    f = open(path, "w", buffering=1)
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.setsockopt(zmq.RCVTIMEO, 1000)
    sub.connect(endpoint)
    print(f"mirror {endpoint} -> {path}", flush=True)
    n = 0
    last_rx = None
    try:
        while True:
            try:
                f.write(sub.recv().decode() + "\n")
                n += 1
                last_rx = time.monotonic()
                if n % 250 == 0:
                    print(f"  {n} rows", flush=True)
            except zmq.Again:
                if last_rx is not None and time.monotonic() - last_rx > 60:
                    print("60s silence after data — capture ended.",
                          flush=True)
                    break
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
    print(f"mirrored {n} rows -> {path}", flush=True)
    print(f"analyze: python3 {os.path.basename(sys.argv[0])} "
          f"--analyze {path}", flush=True)
    return 0


def capture(seconds: float, out_dir: str, pub_port: int = 5599) -> int:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy

    from aimdk_msgs.msg import JointStateArray  # type: ignore

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(
        out_dir, f"damp_capture_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")
    f = open(path, "w", buffering=1)
    fd = f.fileno()

    import zmq
    zctx = zmq.Context()
    zpub = zctx.socket(zmq.PUB)
    zpub.setsockopt(zmq.SNDHWM, 2000)
    zpub.bind(f"tcp://*:{pub_port}")
    print(f"live PUB tcp://*:{pub_port} — start the laptop mirror now "
          f"(--mirror tcp://<pc2-ip>:{pub_port})", flush=True)

    groups = {
        "leg":   "/aima/hal/joint/leg/state",
        "waist": "/aima/hal/joint/waist/state",
        "arm":   "/aima/hal/joint/arm/state",
        "head":  "/aima/hal/joint/head/state",
    }
    # Fall back to the command-echo topic names if state topics differ
    # on this software revision; probe with `ros2 topic list | grep joint`.

    state: dict[str, tuple[list, list, list, list]] = {}
    n_rows = [0]
    t_last_fsync = [0.0]

    class Cap(Node):
        def __init__(self) -> None:
            super().__init__("x2_damp_capture")
            qos = QoSProfile(depth=10)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            for g, topic in groups.items():
                self.create_subscription(
                    JointStateArray, topic,
                    (lambda msg, gg=g: self.on_state(gg, msg)), qos)
            self.timer = self.create_timer(0.004, self.tick)  # 250 Hz
            self.armed_logged = False

        def on_state(self, g: str, msg) -> None:
            names, pos, vel, eff = [], [], [], []
            for js in msg.joints:
                names.append(str(js.name))
                pos.append(float(js.position))
                vel.append(float(js.velocity))
                eff.append(float(js.effort))
            state[g] = (names, pos, vel, eff)

        def tick(self) -> None:
            if len(state) < len(groups):
                return
            if not self.armed_logged:
                self.armed_logged = True
                print("ARMED — all state groups flowing. Trigger the "
                      "STOCK MC damping now.", flush=True)
            now = time.monotonic()
            names: list = []
            pos: list = []
            vel: list = []
            eff: list = []
            for g in ("leg", "waist", "arm", "head"):
                n, p, v, e = state[g]
                names += n
                pos += p
                vel += v
                eff += e
            row = {"kind": "sample", "t": now, "wall": time.time(),
                   "names": names, "pos": pos, "vel": vel, "eff": eff}
            encoded = json.dumps(row)
            f.write(encoded + "\n")
            try:
                zpub.send_string(encoded, zmq.NOBLOCK)
            except Exception:
                pass
            n_rows[0] += 1
            # fsync twice a second: evidence must survive a battery pull.
            if now - t_last_fsync[0] > 0.5:
                t_last_fsync[0] = now
                f.flush()
                os.fsync(fd)

    print(f"capture -> {path} ({seconds:.0f}s)", flush=True)
    rclpy.init()
    node = Cap()
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        f.flush()
        os.fsync(fd)
        f.close()
        node.destroy_node()
        rclpy.shutdown()
    print(f"done: {n_rows[0]} rows -> {path}", flush=True)
    print(f"analyze: python3 {os.path.basename(sys.argv[0])} "
          f"--analyze {path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--out", default="/home/run/gear-sonic/log/damp_capture")
    ap.add_argument("--analyze", metavar="JSONL",
                    help="analyze an existing capture instead of recording")
    ap.add_argument("--pub-port", type=int, default=5599,
                    help="live ZMQ PUB port for the laptop mirror")
    ap.add_argument("--mirror", metavar="ENDPOINT",
                    help="laptop mode: SUB this endpoint (tcp://ip:port) "
                         "and persist rows locally instead of capturing")
    args = ap.parse_args()
    if args.analyze:
        return analyze(args.analyze)
    if args.mirror:
        return mirror(args.mirror, args.out)
    return capture(args.seconds, args.out, args.pub_port)


if __name__ == "__main__":
    raise SystemExit(main())
