#!/usr/bin/env python3
"""Sweep the in-place turn-rate command through the deployed planner graph.

WHY
---
The deploy turn command is a fixed bang-bang setpoint
(``_TEST_FIXED_TURN_RAD_S = 0.30``) and the 2026-07-20 GPU session showed the
robot tracks the reference heading ~1:1 (SONIC does its own footwork) — so
turn speed is limited by the REFERENCE ramp rate, i.e. by this constant.
This sweeps candidate rates offline (same backend + serve path as deploy,
zero latency) and reports, per rate and direction:

  yaw rate achieved   -- deg/s of the served heading stream (vs commanded)
  response ratio      -- achieved / commanded
  still chunks        -- pose-head standing chunks (expected; gate handles)
  heading smoothness  -- p95 per-tick heading step (SONIC tracks this)

Use it to pick the setpoint the ritual should run, then validate in the
sim stack before hardware.

    python gear_sonic/scripts/kplanner_turnrate_sweep.py \
        --onnx-dir ~/x2_cloud_checkpoints/planner_onnx_deployed \
        --rates 0.3,0.45,0.6,0.8,1.0
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


def _yaw_of_wxyz(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def run_one(backend, warm, yaw_rate: float, seconds: float, hip: float):
    backend.reset(warm)
    backend._force_replan = True
    target = (yaw_rate, 0.0, 0.0, hip)
    frames = []
    stills = 0
    chunks = 0
    pending = None
    n_ticks = int(seconds * 50)
    for i in range(n_ticks):
        if pending is not None:
            pred, npf = backend.replan_infer(pending)
            hp_std = max(float(np.std(pred[:npf, 7])),
                         float(np.std(pred[:npf, 13])))
            chunks += 1
            if hp_std <= 0.045:
                stills += 1
            backend.replan_commit(pred, npf)
            pending = None
        forced = getattr(backend, "_force_replan", False)
        if pending is None and (forced or backend.should_replan()):
            backend._force_replan = False
            pending = backend.replan_prepare(target)
        frames.append(backend.get_next_frame_resampled(50.0))
    q = np.asarray(frames, dtype=np.float64)  # [T,38] trans3+wxyz4+dof31
    yaws = np.unwrap([_yaw_of_wxyz(f[3:7]) for f in q])
    total = np.degrees(yaws[-1] - yaws[0])
    steps = np.degrees(np.abs(np.diff(yaws)))
    return {
        "total_deg": total,
        "rate_dps": total / seconds,
        "chunks": chunks,
        "stills": stills,
        "step_p95_deg": float(np.percentile(steps, 95)),
        "step_max_deg": float(steps.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx-dir", required=True, type=Path)
    ap.add_argument("--rates", default="0.3,0.45,0.6,0.8,1.0")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--replan-threshold", type=int, default=32)
    args = ap.parse_args()

    onnx = args.onnx_dir / "x2_planner_template.onnx"
    sidecars = sorted(args.onnx_dir.glob("*.json"))
    contract = k._load_onnx_contract(onnx, sidecars[0] if sidecars else None)
    backend = k.OnnxPlannerBackend(
        onnx_path=onnx, contract=contract,
        replan_threshold_frames=args.replan_threshold,
        planner_mode="slow_walk",
    )
    warm_path = REPO / ("gear_sonic/data/motions/"
                        "kplanner_idle_anchor_g1teleop_v3.pkl")
    warm = k._load_warmup_qpos(warm_path if warm_path.exists() else None)
    hip = float(warm[2])

    rates = [float(r) for r in args.rates.split(",")]
    cmd_hdr = f'{"cmd rad/s":>10s} {"dir":>4s} {"achieved":>9s} ' \
              f'{"ratio":>6s} {"chunks":>7s} {"still":>6s} {"stp p95":>8s}'
    print(cmd_hdr)
    for r in rates:
        for sgn, lab in [(+1.0, "L"), (-1.0, "R")]:
            res = run_one(backend, warm, sgn * r, args.seconds, hip)
            cmd_dps = np.degrees(r)
            print(f"{sgn*r:10.2f} {lab:>4s} {res['rate_dps']:+8.1f}° "
                  f"{abs(res['rate_dps'])/cmd_dps:6.2f} "
                  f"{res['chunks']:7d} {res['stills']:6d} "
                  f"{res['step_p95_deg']:7.2f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
