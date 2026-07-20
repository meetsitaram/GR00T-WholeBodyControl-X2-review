#!/usr/bin/env python3
"""Snapshot a real-robot run: config fingerprint + telemetry + summary metrics.

WHY
---
Tuning experiments are only worth running if the results are comparable later.
Two things were missing:

  1. Run separation. Every deploy since 2026-07-16 appended into ONE directory
     (152 MB of interleaved runs) because the log paths were frozen. Fixed by
     stamping them per-run; this tool assumes that fix is in place.
  2. Provenance. Telemetry without "which sonic / which planner / which tuning
     YAML / which gains" cannot be compared against anything.

This pulls both: it fingerprints what the robot is ACTUALLY running (md5s, not
filenames -- a filename says what someone intended, an md5 says what is there),
copies the run's telemetry, and computes the metrics we keep re-deriving by
hand.

    # after a run, with the robot reachable
    python gear_sonic/scripts/capture_robot_run.py --pc2 192.168.86.32 \
        --label soft_kp_run1 --note "clicks on release, 3 turns"

Writes to docs/experiments/robot_runs/<timestamp>_<label>/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "docs" / "experiments" / "robot_runs"
GETSOLO = "/home/run/getsolo"


def ssh(host: str, cmd: str, timeout: int = 30) -> str:
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=8", f"run@{host}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return r.stdout.strip()


def fingerprint(host: str) -> dict:
    """What the robot is ACTUALLY running, by content not by name."""
    script = rf"""
cd {GETSOLO}
echo "sonic_md5=$(md5sum policies/agibot_x2_sonic.onnx 2>/dev/null | cut -d' ' -f1)"
echo "planner_tmpl_md5=$(md5sum planner_stack/models/planner_onnx/x2_planner_template.onnx 2>/dev/null | cut -d' ' -f1)"
echo "runtime_md5=$(md5sum pc2_kplanner_onnx.py 2>/dev/null | cut -d' ' -f1)"
echo "ritual_md5=$(md5sum ritual_start_demo.sh 2>/dev/null | cut -d' ' -f1)"
echo "tuning_file=$(grep -oE 'walking_[a-z_]+\.yaml|expressive\.yaml|conservative\.yaml' log/start_x2_deploy.sh | head -1)"
echo "anchor=$(grep -o 'kplanner_idle_anchor_g1teleop_v[0-9]' ritual_start_demo.sh | head -1)"
echo "smoother=$(grep -oE '\-\-ref-smoother-shape [a-z]+' ritual_start_demo.sh | awk '{{print $2}}')"
echo "stop_blend=$(grep -c 'blending to anchor' pc2_kplanner_onnx.py)"
"""
    fp = {}
    for line in ssh(host, script).splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            fp[k.strip()] = v.strip()
    # the gains actually in force
    tf = fp.get("tuning_file", "")
    if tf:
        gains = ssh(host, f"grep -E '^(kp_scale|kd_scale)' "
                          f"{GETSOLO}/gear_sonic_deploy/configs/real_deploy_tuning/{tf}")
        fp["gains"] = {}
        for line in gains.splitlines():
            line = line.split("#")[0].strip()
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    fp["gains"][k.strip()] = float(v.strip())
                except ValueError:
                    pass
    return fp


def newest_run_dir(host: str) -> str:
    return ssh(host, f"ls -dt {GETSOLO}/log/deploy_* 2>/dev/null | head -1")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pc2", required=True)
    ap.add_argument("--label", required=True, help="short name, e.g. soft_kp_run1")
    ap.add_argument("--note", default="", help="what you observed, in your words")
    ap.add_argument("--run-dir", default=None, help="override; default = newest")
    ap.add_argument("--no-csv", action="store_true",
                    help="skip the big CSVs (fingerprint + logs only)")
    args = ap.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = OUT_ROOT / f"{stamp}_{args.label}"
    out.mkdir(parents=True, exist_ok=True)

    print(f"  capturing -> {out.relative_to(REPO)}")
    fp = fingerprint(args.pc2)
    rd = args.run_dir or newest_run_dir(args.pc2)
    fp["run_dir"] = rd
    fp["captured_at"] = stamp
    fp["label"] = args.label
    fp["note"] = args.note

    print("  --- what the robot is running ---")
    for k in ("sonic_md5", "planner_tmpl_md5", "runtime_md5", "tuning_file",
              "anchor", "smoother", "stop_blend"):
        v = fp.get(k, "?")
        v = v[:12] if k.endswith("md5") else v
        print(f"    {k:<18} {v}")

    if rd:
        files = ["imu.csv", "tick.csv"] + ([] if args.no_csv else
                 ["target_pos.csv", "joint_pos.csv", "joint_vel.csv", "action_il.csv"])
        for f in files:
            subprocess.run(["scp", "-q", f"run@{args.pc2}:{rd}/{f}", str(out / f)],
                           capture_output=True)
            p = out / f
            if p.exists():
                print(f"    pulled {f:<16} {p.stat().st_size/1e6:6.1f} MB")
    # event logs (small, always worth having)
    for remote, local in [("log/pc2_kplanner.log", "pc2_kplanner.log"),
                          ("log/pad_bridge.log", "pad_bridge.log"),
                          ("log/ritual_fired.log", "ritual_fired.log")]:
        subprocess.run(["scp", "-q", f"run@{args.pc2}:{GETSOLO}/{remote}",
                        str(out / local)], capture_output=True)
    wd = ssh(args.pc2, f"ls -t {GETSOLO}/log/pose_watchdog_*.log | head -1")
    if wd:
        subprocess.run(["scp", "-q", f"run@{args.pc2}:{wd}",
                        str(out / "pose_watchdog.log")], capture_output=True)
    # intent tape (jsonl; the replay key: gamepad intents + replan seeds +
    # publish ticks with precise timing). Newest tape = this daemon session.
    tape = ssh(args.pc2, f"ls -t {GETSOLO}/log/kplanner_tape/tape_*.jsonl "
                         f"2>/dev/null | head -1")
    if tape:
        subprocess.run(["scp", "-q", f"run@{args.pc2}:{tape}",
                        str(out / "intent_tape.jsonl")], capture_output=True)
        p = out / "intent_tape.jsonl"
        if p.exists():
            n = sum(1 for _ in open(p))
            print(f"    pulled intent_tape.jsonl ({n} events)")
        # full-content frame tape + committed chunks (same session stem)
        stem = tape[:-len(".jsonl")]
        subprocess.run(["scp", "-q", f"run@{args.pc2}:{stem}.frames.f32",
                        str(out / "frame_tape.f32")], capture_output=True)
        fp2 = out / "frame_tape.f32"
        if fp2.exists():
            print(f"    pulled frame_tape.f32 ({fp2.stat().st_size // 160} ticks)")
        subprocess.run(["scp", "-q", "-r", f"run@{args.pc2}:{stem}_chunks",
                        str(out / "chunks")], capture_output=True)
        cd = out / "chunks"
        if cd.exists():
            print(f"    pulled chunks/ ({len(list(cd.glob('*.npy')))} committed chunks)")
    else:
        print("    WARNING: no intent tape found -- daemon predates the tape "
              "patch or KPLANNER_TAPE=0")

    (out / "fingerprint.json").write_text(json.dumps(fp, indent=2))
    print(f"\n  wrote fingerprint.json  ({len(fp.get('gains', {}))} gain values)")
    print(f"  run: python gear_sonic/scripts/summarize_robot_run.py {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
