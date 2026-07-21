#!/usr/bin/env python3
"""Sweep planner graphs x commands x speeds; score reference quality.

Built for the 2026-07-21 template A/B/C ("forward+right mostly turns in
place; right step bumpy; fwd speed seemed inert" -- the last one was a
launcher env clobber, fixed separately). Runs each (graph, command, speed)
cell for N seconds through the exact deploy backend + serve path at zero
latency and reports, per cell:

  fwd m/s     realized forward speed (body-frame, from wire root)
  yaw d/s     realized heading rate
  arc ratio   fwd travel vs heading (walking-turn quality; in-place -> ~0)
  skate       stance-foot slide mean (m/s; good clip ~0.115)
  swings L/R  swing events per 10s per foot (stepping quality + symmetry)
  still%      still chunks (hip-std < 0.045)

    python gear_sonic/scripts/kplanner_template_sweep.py \
        --graphs p500k=~/x2_cloud_checkpoints/planner_onnx_fixedscratch_p500k,...
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gear_sonic/scripts"))

import pc2_kplanner_onnx as k  # noqa: E402
from kplanner_frame_eval import frame_series, swing_events, _mjcf_path  # noqa: E402


def yaw_of_wxyz(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_cell(backend, warm, target, seconds):
    backend.reset(warm)
    backend._force_replan = True
    frames, stills, chunks = [], 0, 0
    pending = None
    for _ in range(int(seconds * 50)):
        if pending is not None:
            pred, npf = backend.replan_infer(pending)
            chunks += 1
            hp = max(float(np.std(pred[:npf, 7])), float(np.std(pred[:npf, 13])))
            if hp <= 0.045:
                stills += 1
            backend.replan_commit(pred, npf)
            pending = None
        forced = getattr(backend, "_force_replan", False)
        if pending is None and (forced or backend.should_replan()):
            backend._force_replan = False
            pending = backend.replan_prepare(target)
        frames.append(backend.get_next_frame_resampled(50.0))
    q = np.asarray(frames, dtype=np.float64)  # [T,38] xyz + wxyz + dof31
    T = len(q)
    yaws = np.unwrap([yaw_of_wxyz(f[3:7]) for f in q])
    yaw_dps = float(np.degrees(yaws[-1] - yaws[0]) / seconds)
    # body-frame forward speed: rotate each step into current heading
    d = np.diff(q[:, :2], axis=0)
    c, s = np.cos(yaws[:-1]), np.sin(yaws[:-1])
    fwd_mps = float(np.mean(d[:, 0] * c + d[:, 1] * s) * 50)
    fs = frame_series(q, _mjcf_path(None), 50.0)
    sl = fs["slide"]; act = sl[sl > 0]
    skate = float(act.mean()) if len(act) else 0.0
    xyzw = np.column_stack([q[:, 4], q[:, 5], q[:, 6], q[:, 3]])
    sw = swing_events(q[:, 7:], xyzw, q[:, :3])
    return {
        "fwd": fwd_mps, "yaw": yaw_dps,
        "skate": skate, "swL": sw[0][0], "swR": sw[1][0],
        "still": 100.0 * stills / max(1, chunks),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graphs", required=True,
                    help="name=dir[,name=dir...] planner graph dirs")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    warm = k._load_warmup_qpos(
        REPO / "gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl")
    hip = float(warm[2])

    # (label, yaw_rate, vx_lat, vz_fwd) -- daemon target order [yaw, lat, fwd, hip]
    def cells():
        for v in (0.3, 0.5, 0.8, 1.0):
            yield (f"FWD {v}", (0.0, 0.0, v, hip))
        for arc in (0.4, 0.55):
            yield (f"FWD0.5+R arc{arc}", (-arc, 0.0, 0.5, hip))
            yield (f"FWD0.5+L arc{arc}", (+arc, 0.0, 0.5, hip))
        yield ("TURN L 1.0", (+1.0, 0.0, 0.0, hip))
        yield ("TURN R 1.0", (-1.0, 0.0, 0.0, hip))

    rows = []
    for spec in args.graphs.split(","):
        name, d = spec.split("=", 1)
        d = Path(d).expanduser()
        onnx = d / "x2_planner_template.onnx"
        sidecars = sorted(d.glob("*.json"))
        contract = k._load_onnx_contract(
            onnx, sidecars[0] if sidecars else None)
        backend = k.OnnxPlannerBackend(
            onnx_path=onnx, contract=contract,
            replan_threshold_frames=32, planner_mode="slow_walk")
        for label, tgt in cells():
            r = run_cell(backend, warm, tgt, args.seconds)
            r.update(graph=name, cmd=label)
            rows.append(r)
            print(f"{name:8s} {label:18s} fwd {r['fwd']:+5.2f}  "
                  f"yaw {r['yaw']:+6.1f}  skate {r['skate']:.3f}  "
                  f"sw {r['swL']}/{r['swR']}  still {r['still']:3.0f}%",
                  flush=True)
    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
