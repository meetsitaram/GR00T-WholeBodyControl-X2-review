#!/usr/bin/env python3
"""Motor thermal notifier: robot speaks + pad rumbles while motors run hot.

Runs ON PC2, independent of the SONIC deploy (subscribes the aimdk HAL
state topics directly, so it works during vendor-MC operation too).
Two joint classes with separate thresholds, voice lines, and cadence
(operator design 2026-08-04):

  * CRITICAL  = legs + waist (locomotion joints): louder voice line
    ("Warning. Leg motor temperature high...") + 5-pulse pad rumble,
    repeated every --critical-interval (default 120 s) while hot.
  * UPPER     = arms + head: informational line ("Notice. Arm motor
    temperature elevated."), no rumble by default, repeated every
    --upper-interval (default 300 s).

Hysteresis: an alert class re-arms only after its max temp drops
--hysteresis (default 3 C) below its threshold.

BRING-UP PROTOCOL: thresholds default LOW (--critical-c 40, --upper-c 40)
so alerts fire during normal operation and the whole pipeline (speaker,
rumble, cadence) is confirmed on hardware. Once verified, raise to real
values (motor spec ~90 C enter / 85 C exit — see the deploy's C++
ThermalMonitor) via CLI flags or the RAISED defaults marked below.

Alert paths (both best-effort, never crash the loop):
  * PC3 speaker: sshpass ssh agi@${X2_PC3_HOST} aplay (WAVs pre-staged at
    /opt/x2_interact/audio/ by gen_pc3_audio_prompts.py --stage).
  * Pad rumble: ZMQ PUSH to the pc2_pad_daemon rumble PULL (:5570),
    schema {"strength", "ms", "count", "gap_ms"} — the daemon schedules
    the pulse train, we never sleep.

Usage (PC2):
    /home/run/gear-sonic/venv/bin/python3 /home/run/gear-sonic/x2_thermal_notifier.py
    # ... verify alerts fire, then:
    ... --critical-c 90 --upper-c 90 --hysteresis 5

Status: prints per-group max temps every --status-interval (30 s) so
the operator always sees live values; logs every alert with the peak
joint name + temperature.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

PC3 = "agi@${X2_PC3_HOST}"
PC3_PASS = os.environ.get("X2_PC3_PASS", "")
AUDIO_DIR = "/opt/x2_interact/audio"

GROUP_TOPICS = {
    "leg":   "/aima/hal/joint/leg/state",
    "waist": "/aima/hal/joint/waist/state",
    "arm":   "/aima/hal/joint/arm/state",
    "head":  "/aima/hal/joint/head/state",
}
CRITICAL_GROUPS = ("leg", "waist")
UPPER_GROUPS = ("arm", "head")


def play_pc3(wav: str) -> None:
    try:
        subprocess.Popen(
            ["sshpass", "-p", PC3_PASS, "ssh",
             "-o", "ConnectTimeout=2", "-o", "StrictHostKeyChecking=no",
             PC3, f"aplay -D playback_def {AUDIO_DIR}/{wav}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, start_new_session=True)
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # PRODUCTION defaults since 2026-08-04 (bring-up at 40C verified the
    # full pipeline on hardware: operator heard both voice lines and felt
    # the 5-pulse rumble). Early-warning margins under the deploy's hard
    # 90/85C ThermalMonitor.
    ap.add_argument("--critical-c", type=float, default=75.0,
                    help="legs+waist alert threshold C")
    ap.add_argument("--upper-c", type=float, default=80.0,
                    help="arms+head alert threshold C")
    ap.add_argument("--hysteresis", type=float, default=3.0)
    ap.add_argument("--critical-interval", type=float, default=120.0,
                    help="seconds between repeated critical alerts while hot")
    ap.add_argument("--upper-interval", type=float, default=300.0)
    ap.add_argument("--status-interval", type=float, default=30.0)
    ap.add_argument("--rumble-port", type=int, default=5570)
    ap.add_argument("--no-rumble", action="store_true")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    import zmq

    from aimdk_msgs.msg import JointStateArray  # type: ignore

    zctx = zmq.Context()
    rum = zctx.socket(zmq.PUSH)
    rum.setsockopt(zmq.SNDHWM, 4)
    rum.setsockopt(zmq.LINGER, 0)
    rum.connect(f"tcp://127.0.0.1:{args.rumble_port}")

    def rumble_pattern() -> None:
        if args.no_rumble:
            return
        try:  # 5 long pulses; daemon schedules the train (never sleeps here)
            rum.send_json({"strength": 1.0, "ms": 600,
                           "count": 5, "gap_ms": 350}, flags=zmq.NOBLOCK)
        except Exception:
            pass

    temps: dict[str, tuple[list, list]] = {}   # group -> (names, coil C)
    _invalid_logged: set = set()

    class Notifier(Node):
        def __init__(self) -> None:
            super().__init__("x2_thermal_notifier")
            qos = QoSProfile(depth=5)
            qos.reliability = ReliabilityPolicy.BEST_EFFORT
            for g, topic in GROUP_TOPICS.items():
                self.create_subscription(
                    JointStateArray, topic,
                    (lambda msg, gg=g: self.on_state(gg, msg)), qos)

        def on_state(self, g: str, msg) -> None:
            names, cs = [], []
            for js in msg.joints:
                n = str(js.name)
                # coil (winding) heats fastest; use max(coil, motor)
                c = max(float(js.coil_temp), float(js.motor_temp))
                # Plausibility window (2026-08-04 bring-up finding:
                # head_yaw reports a constant 121C — unsensed motors
                # publish sentinel garbage). Readings outside (5, 110)C
                # are sensor-invalid: excluded from alerting, logged
                # once so real thermal runaway is never silently
                # filtered with them.
                if not (5.0 < c < 110.0):
                    if n not in _invalid_logged:
                        _invalid_logged.add(n)
                        print(f"[thermal] sensor-invalid reading "
                              f"EXCLUDED from alerts: {n}={c:.1f}C",
                              flush=True)
                    continue
                names.append(n)
                cs.append(c)
            temps[g] = (names, cs)

    def group_peak(groups) -> tuple[str, float]:
        peak_n, peak_c = "?", float("-inf")
        for g in groups:
            if g not in temps:
                continue
            names, cs = temps[g]
            for n, c in zip(names, cs):
                if c > peak_c:
                    peak_n, peak_c = n, c
        return peak_n, peak_c

    classes = {
        "CRITICAL": dict(groups=CRITICAL_GROUPS, thr=args.critical_c,
                         interval=args.critical_interval,
                         wav="thermal_legs.wav", rumble=True,
                         last_alert=0.0, hot=False),
        "UPPER":    dict(groups=UPPER_GROUPS, thr=args.upper_c,
                         interval=args.upper_interval,
                         wav="thermal_upper.wav", rumble=False,
                         last_alert=0.0, hot=False),
    }

    print(json.dumps({"kind": "boot", "critical_c": args.critical_c,
                      "upper_c": args.upper_c,
                      "hysteresis": args.hysteresis}), flush=True)
    rclpy.init()
    node = Notifier()
    last_status = 0.0
    try:
        while True:
            rclpy.spin_once(node, timeout_sec=0.5)
            now = time.monotonic()
            for cname, c in classes.items():
                if not any(g in temps for g in c["groups"]):
                    continue
                peak_n, peak_c = group_peak(c["groups"])
                if c["hot"] and peak_c < c["thr"] - args.hysteresis:
                    c["hot"] = False
                    print(f"[thermal {time.strftime('%H:%M:%S')}] {cname} "
                          f"RECOVERED: {peak_n}={peak_c:.1f}C", flush=True)
                if peak_c >= c["thr"] and (
                        now - c["last_alert"] >= c["interval"]):
                    c["hot"] = True
                    c["last_alert"] = now
                    print(f"[thermal {time.strftime('%H:%M:%S')}] {cname} "
                          f"ALERT: {peak_n}={peak_c:.1f}C "
                          f"(thr {c['thr']:.0f}C) -> speaker"
                          f"{' + rumble' if c['rumble'] else ''}",
                          flush=True)
                    play_pc3(c["wav"])
                    if c["rumble"]:
                        rumble_pattern()
            if now - last_status >= args.status_interval:
                last_status = now
                stat = {}
                for cname, c in classes.items():
                    n, t = group_peak(c["groups"])
                    if t > float("-inf"):
                        stat[cname] = f"{n}={t:.1f}C"
                print(f"[thermal {time.strftime('%H:%M:%S')}] status "
                      f"{stat}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        # rclpy handles SIGINT/SIGTERM itself; a second shutdown here
        # raises RCLError and pollutes the exit log. Best-effort only.
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
