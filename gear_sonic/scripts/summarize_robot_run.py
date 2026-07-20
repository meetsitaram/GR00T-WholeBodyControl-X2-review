#!/usr/bin/env python3
"""Compute comparable metrics from a captured robot run.

Every metric here exists because we derived it by hand during an incident and
then wanted it again for the next run. Making them automatic is the difference
between "the clicks felt better" and a number you can put next to a config.

METRICS
  reference/command jumps  -- max per-tick change in target_pos (the POLICY's
      commanded joint target = default + action*scale). Big jumps are what kp
      converts into an actuator slam, i.e. the audible click. Clean planner
      output tops out ~0.15 rad; on-robot has hit 1.21 rad.
  yaw oscillation          -- reversals/s and peak rate from imu.csv. A limit
      cycle here preceded the 2026-07-19 fall (+-20 deg at 40-70 deg/s).
  tilt                     -- max deviation from vertical; fall precursor.
  tracking error           -- |target_pos - joint_pos|, i.e. what SONIC failed
      to follow. Distinguishes "bad reference" from "bad tracking".
  planner events           -- stalls, state transitions, intents.

    python gear_sonic/scripts/summarize_robot_run.py docs/experiments/robot_runs/<dir>
    python gear_sonic/scripts/summarize_robot_run.py --compare A B   # two runs
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_csv(p: Path):
    if not p.exists():
        return None, None
    rows = []
    with p.open() as f:
        hdr = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != len(hdr):
                continue           # truncated tail when a run is killed
            try:
                rows.append([float(x) for x in parts])
            except ValueError:
                continue
    return (np.array(rows) if rows else None), hdr


def summarize(d: Path) -> dict:
    m: dict = {"dir": d.name}
    fp_p = d / "fingerprint.json"
    if fp_p.exists():
        fp = json.loads(fp_p.read_text())
        m["label"] = fp.get("label", "")
        m["note"] = fp.get("note", "")
        m["tuning"] = fp.get("tuning_file", "")
        m["sonic"] = (fp.get("sonic_md5") or "")[:12]
        g = fp.get("gains", {})
        m["kp_knee"] = g.get("kp_scale_knee")
        m["kp_ankle_p"] = g.get("kp_scale_ankle_pitch")
        m["kd_ankle_p"] = g.get("kd_scale_ankle_pitch")
        m["kd_waist_yaw"] = g.get("kd_scale_waist_yaw")

    tp, _ = load_csv(d / "target_pos.csv")
    if tp is not None and tp.shape[0] > 2:
        jump = np.abs(np.diff(tp[:, 1:], axis=0)).max(axis=1)
        m["ticks"] = int(tp.shape[0])
        m["jump_med"] = float(np.median(jump))
        m["jump_p95"] = float(np.percentile(jump, 95))
        m["jump_max"] = float(jump.max())
        m["jump_over_0.3_pct"] = float(100.0 * (jump > 0.3).mean())

        jp, _ = load_csv(d / "joint_pos.csv")
        if jp is not None and jp.shape[0] >= tp.shape[0]:
            n = min(tp.shape[0], jp.shape[0])
            err = np.abs(tp[:n, 1:] - jp[:n, 1:])
            m["track_err_med"] = float(np.median(err.max(axis=1)))
            m["track_err_p95"] = float(np.percentile(err.max(axis=1), 95))

    imu, _ = load_csv(d / "imu.csv")
    if imu is not None and imu.shape[0] > 10:
        t = imu[:, 0]
        qw, qx, qy, qz = imu[:, 1], imu[:, 2], imu[:, 3], imu[:, 4]
        yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy),
                                    1 - 2 * (qy ** 2 + qz ** 2)))
        yu = np.degrees(np.unwrap(np.radians(yaw)))
        dt = np.diff(t)
        rate = np.diff(yu) / np.where(dt > 0, dt, 1e-6)
        sign = np.sign(rate)
        reversals = int((np.diff(sign) != 0).sum())
        dur = float(t[-1] - t[0]) or 1.0
        m["yaw_reversals_per_s"] = reversals / dur
        m["yaw_rate_p99"] = float(np.percentile(np.abs(rate), 99))
        gz = 1 - 2 * (qx ** 2 + qy ** 2)
        m["tilt_max_deg"] = float(np.degrees(np.arccos(np.clip(gz, -1, 1))).max())
        m["duration_s"] = dur

    kl = d / "pc2_kplanner.log"
    if kl.exists():
        txt = kl.read_text(errors="ignore")
        m["stalls"] = txt.count("fell behind")
        m["to_playing"] = txt.count("IDLE_LOOP -> PLAYING")
        m["to_idle"] = txt.count("PLAYING -> IDLE_LOOP")
        m["stop_blend_fired"] = txt.count("blending to anchor")
    return m


def show(m: dict) -> None:
    print(f"\n=== {m.get('label') or m['dir']} ===")
    if m.get("note"):
        print(f"  note        : {m['note']}")
    print(f"  tuning      : {m.get('tuning','?')}   sonic {m.get('sonic','?')}")
    g = [f"{k}={m[k]}" for k in ("kp_knee", "kp_ankle_p", "kd_ankle_p", "kd_waist_yaw")
         if m.get(k) is not None]
    if g:
        print(f"  gains       : {'  '.join(g)}")
    if "duration_s" in m:
        print(f"  duration    : {m['duration_s']:.1f}s")
    if "jump_max" in m:
        print(f"  cmd jumps   : med {m['jump_med']:.4f}  p95 {m['jump_p95']:.4f}  "
              f"MAX {m['jump_max']:.3f} rad   ({m['jump_over_0.3_pct']:.1f}% of ticks >0.3)")
    if "track_err_p95" in m:
        print(f"  track error : med {m['track_err_med']:.4f}  p95 {m['track_err_p95']:.4f} rad")
    if "yaw_reversals_per_s" in m:
        print(f"  yaw         : {m['yaw_reversals_per_s']:.2f} reversals/s   "
              f"p99 rate {m['yaw_rate_p99']:.1f} deg/s   tilt max {m['tilt_max_deg']:.1f} deg")
    if "stalls" in m:
        print(f"  planner     : stalls {m['stalls']}   ->PLAYING {m['to_playing']}   "
              f"->IDLE {m['to_idle']}   stop-blends {m['stop_blend_fired']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()
    ms = [summarize(d) for d in args.dirs]
    for m in ms:
        show(m)
    if len(ms) > 1:
        print("\n=== side by side ===")
        keys = [("jump_max", "max cmd jump (rad)", "lower better"),
                ("jump_over_0.3_pct", "% ticks >0.3 rad", "lower better"),
                ("track_err_p95", "p95 track err (rad)", "lower better"),
                ("yaw_reversals_per_s", "yaw reversals/s", "lower better"),
                ("stalls", "planner stalls", "lower better")]
        w = max(len(m.get("label") or m["dir"]) for m in ms)
        print(f"  {'metric':<24}" + "".join(f"{(m.get('label') or m['dir'])[:w]:>{w+3}}" for m in ms))
        for k, lbl, hint in keys:
            if all(m.get(k) is not None for m in ms):
                print(f"  {lbl:<24}" + "".join(f"{m[k]:>{w+3}.3f}" for m in ms) + f"   ({hint})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
