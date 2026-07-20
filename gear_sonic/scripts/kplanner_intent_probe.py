#!/usr/bin/env python3
"""Replay a REAL clip's intent through the kplanner at several settings.

WHY
---
We can measure planner output, but until now we could only drive it with
hand-written schedules. This takes the intent (forward/lateral velocity +
yaw rate) out of a clip that is KNOWN to walk the robot, and feeds that same
intent back through the planner. Then planner output and the clip that
produced the intent are directly comparable: same commands, one is real
motion, the other is generated.

CHUNKED, NOT PER-FRAME -- this matters. Re-issuing intent every frame would
let the planner track a velocity profile no operator can reproduce: a gamepad
holds a roughly constant command for a while, then changes. So intent is
averaged over ``--chunk-s`` windows and issued piecewise-constant, which is
what the controller actually produces and what the deployed replan cadence
can actually follow.

Each setting is a (pose-template mode, speed scale) pair. Speed scale
multiplies linear velocity only -- facing is left alone, so the same path is
requested at different speeds. That isolates the speed axis, which is where
the measured ~0.18 m/s constant foot-slide floor stops being negligible.

    python gear_sonic/scripts/kplanner_intent_probe.py \
        --clip gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl \
        --settings slow_walk@1.0,walk@1.0,walk@1.5,slow_walk@0.5 \
        --out-dir out/frame_eval/intent_probe
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "motionbricks"))

_RSD = REPO / "motionbricks/scripts/kplanner_validation/run_scripted_demo.py"
MODE_IDX = {"idle": 0, "slow_walk": 1, "walk": 2, "run_proxy": 3, "velocity": None}


def _load_harness():
    spec = importlib.util.spec_from_file_location("run_scripted_demo", _RSD)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_scripted_demo"] = mod
    spec.loader.exec_module(mod)
    return mod


def clip_intent(pkl: Path, key: str | None, chunk_s: float):
    """Body-frame (yaw_rate, vel_lateral, vel_forward) averaged per chunk."""
    lib = joblib.load(pkl)
    k = key or next(iter(lib))
    if k not in lib:
        raise KeyError(f"{k!r} not in {pkl}; keys={list(lib)[:5]}")
    c = lib[k]
    tr = np.asarray(c["root_trans_offset"], dtype=np.float64)
    rot = np.asarray(c["root_rot"], dtype=np.float64)          # xyzw
    fps = float(c.get("fps", 30.0))
    x, y, z, w = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    yaw_u = np.unwrap(yaw)

    v_world = np.diff(tr[:, :2], axis=0) * fps                 # (T-1, 2)
    yaw_mid = yaw_u[:-1]
    fwd = v_world[:, 0] * np.cos(yaw_mid) + v_world[:, 1] * np.sin(yaw_mid)
    lat = -v_world[:, 0] * np.sin(yaw_mid) + v_world[:, 1] * np.cos(yaw_mid)
    yaw_rate = np.diff(yaw_u) * fps

    n = max(1, int(round(chunk_s * fps)))
    chunks = []
    for a in range(0, len(fwd) - n + 1, n):
        b = a + n
        chunks.append((float(np.mean(yaw_rate[a:b])),
                       float(np.mean(lat[a:b])),
                       float(np.mean(fwd[a:b]))))
    return chunks, fps, k, float(np.median(tr[:, 2]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--key", default=None)
    ap.add_argument("--chunk-s", type=float, default=0.5,
                    help="intent hold time; mimics a gamepad hold (default 0.5)")
    ap.add_argument("--settings", default="slow_walk@1.0,walk@1.0,walk@1.5",
                    help="comma list of mode@speed_scale")
    ap.add_argument("--out-dir", type=Path,
                    default=REPO / "out/frame_eval/intent_probe")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    H = _load_harness()
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunks, fps, key, hip_h = clip_intent(args.clip, args.key, args.chunk_s)
    dur = len(chunks) * args.chunk_s
    fwd = [c[2] for c in chunks]
    print(f"  clip {key}: {len(chunks)} chunks x {args.chunk_s}s = {dur:.1f}s")
    print(f"  intent fwd vel: min {min(fwd):+.3f}  median "
          f"{np.median(fwd):+.3f}  max {max(fwd):+.3f} m/s   hip_h {hip_h:.3f}")

    # The planner is seeded from the SAME clip, so any divergence is the
    # planner's, not a mismatched starting pose.
    seed_qpos, seed_fps = H._load_x2_fixture_qpos(args.clip, key)
    planner = H._load_planner("x2", device)

    results = []
    for spec in args.settings.split(","):
        spec = spec.strip()
        mode_name, _, scale_s = spec.partition("@")
        scale = float(scale_s) if scale_s else 1.0
        if mode_name not in MODE_IDX:
            raise SystemExit(f"unknown mode {mode_name!r}; "
                             f"choose from {list(MODE_IDX)}")
        midx = MODE_IDX[mode_name]
        sched = [H.Step(f"c{i}", args.chunk_s, yr, lat * scale, fw * scale,
                        mode_idx=midx)
                 for i, (yr, lat, fw) in enumerate(chunks)]
        qpos, seglog = H._run_schedule(planner, seed_qpos, 0, sched, hip_h,
                                       fps, device)
        out = args.out_dir / f"{key}__{mode_name}_x{scale:g}.npz"
        np.savez_compressed(out, qpos_traj=qpos[None], fps=fps,
                            setting=spec, clip=key)
        results.append((spec, out, qpos))
        print(f"  ran {spec:<18} -> {out.name}  ({qpos.shape[0]} frames)")

    # --- score every run with the shared per-frame evaluator --------------
    from gear_sonic.scripts.kplanner_frame_eval import (frame_series, summarize,
                                                        _mjcf_path)
    mjcf = _mjcf_path(None)
    print(f"\n  {'setting':<20}{'root':>8}{'slide':>8}{'ratio':>8}"
          f"{'yaw p95':>9}{'air%':>7}{'jump':>7}")
    print("  " + "-" * 67)
    ref = frame_series(H._load_x2_fixture_qpos(args.clip, key)[0], mjcf, fps)
    rm = summarize(ref, "REFERENCE CLIP")
    rr = rm["root_speed_mean"]
    print(f"  {'REFERENCE (real)':<20}{rr:>8.3f}{rm['slide_mean']:>8.3f}"
          f"{(rm['slide_mean']/rr if rr > 0.05 else float('nan')):>8.2f}"
          f"{np.degrees(rm['yaw_rate_p95']):>9.1f}"
          f"{rm['both_feet_air_pct']:>7.1f}{rm['jump_max']:>7.3f}")
    for spec, _out, qpos in results:
        s = frame_series(qpos, mjcf, fps)
        m = summarize(s, spec)
        r = m["root_speed_mean"]
        print(f"  {spec:<20}{r:>8.3f}{m['slide_mean']:>8.3f}"
              f"{(m['slide_mean']/r if r > 0.05 else float('nan')):>8.2f}"
              f"{np.degrees(m['yaw_rate_p95']):>9.1f}"
              f"{m['both_feet_air_pct']:>7.1f}{m['jump_max']:>7.3f}")
    print(f"\n  npz written to {args.out_dir} -- view with:")
    print(f"    python gear_sonic/scripts/view_multi_qpos.py "
          f"--npz {args.out_dir}/*.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
