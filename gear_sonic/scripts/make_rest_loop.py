#!/usr/bin/env python3
"""Create a looped "rest" motion PKL by extracting a few frames from a
source motion-lib PKL.

Use case:
    The C++ deploy needs a `--motion` reference clip when running with the
    motion-tracking SONIC policy. A pure constant pose (`StandStillReference`)
    is out-of-distribution; the policy was trained on naturally noisy
    motion-lib clips. This script lets us bake a short "rest" clip:

      - take the first N frames of an existing source motion (typically an
        idle-loop or the start of a relaxed walk)
      - optionally zero the world XY drift (so the looped reference doesn't
        wander when sampled past its end)
      - write the result back as a new motion-lib PKL with the same schema

The new PKL can then be fed straight to ``gear_sonic_deploy/scripts/
export_motion_for_deploy.py`` to produce a ``.x2m2`` for the deploy
pipeline, *or* visualised with ``play_motion_mujoco.py``.

Example:
    python gear_sonic/scripts/make_rest_loop.py \\
        --source gear_sonic/data/motions/x2_ultra_stand_idle_smoke.pkl \\
        --motion-key loco__neutral_idle_loop_002__A074 \\
        --frames 60 --start 0 \\
        --out gear_sonic/data/motions/x2_ultra_rest_loop_idle.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True, type=Path,
                   help="Source motion-lib .pkl path.")
    p.add_argument("--motion-key", default=None,
                   help="Which motion within the source PKL to slice. "
                        "Defaults to the first key.")
    p.add_argument("--start", type=int, default=0,
                   help="First frame to take (inclusive). Default 0.")
    p.add_argument("--frames", type=int, default=60,
                   help="Number of frames to keep. Default 60 (~2s @30fps).")
    p.add_argument("--zero-xy", action="store_true",
                   help="Subtract first-frame world XY so the looped clip "
                        "doesn't drift in space. Pelvis Z is left intact.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output PKL path (motion-lib schema).")
    p.add_argument("--out-key", default=None,
                   help="Motion key inside the new PKL. Defaults to "
                        "<source motion key>_rest_loop.")
    args = p.parse_args(argv)

    print(f"[make_rest_loop] loading {args.source} ...", flush=True)
    src = joblib.load(args.source)
    if not isinstance(src, dict) or not src:
        print("ERROR: source PKL is not a non-empty dict-of-motions",
              file=sys.stderr)
        return 1

    key = args.motion_key or next(iter(src))
    if key not in src:
        print(f"ERROR: motion key {key!r} not in source. "
              f"Available: {list(src.keys())[:10]}{' ...' if len(src) > 10 else ''}",
              file=sys.stderr)
        return 1

    motion = src[key]
    for required in ("dof", "root_rot", "root_trans_offset", "fps"):
        if required not in motion:
            print(f"ERROR: source motion missing key {required!r}", file=sys.stderr)
            return 1

    dof = np.asarray(motion["dof"])
    rot = np.asarray(motion["root_rot"])
    pos = np.asarray(motion["root_trans_offset"])
    fps = float(motion["fps"])

    T = dof.shape[0]
    s = max(0, args.start)
    e = min(T, s + args.frames)
    if e - s < 2:
        print(f"ERROR: requested slice [{s}:{e}) is too short", file=sys.stderr)
        return 1

    out_dof = dof[s:e].copy()
    out_rot = rot[s:e].copy()
    out_pos = pos[s:e].copy()

    if args.zero_xy:
        xy0 = out_pos[0, :2].copy()
        out_pos[:, :2] -= xy0
        print(f"[make_rest_loop] zeroed world XY (was [{xy0[0]:+.3f},{xy0[1]:+.3f}])")

    new_motion = dict(motion)
    new_motion["dof"] = out_dof
    new_motion["root_rot"] = out_rot
    new_motion["root_trans_offset"] = out_pos
    new_motion["fps"] = fps

    out_key = args.out_key or f"{key}_rest_loop"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({out_key: new_motion}, args.out)

    arm_cols = [15, 16, 17, 18, 22, 23, 24, 25]  # L/R shoulder p/r/y, elbow
    arm_means = out_dof[:, arm_cols].mean(axis=0)
    arm_std = out_dof[:, arm_cols].std(axis=0).mean()
    pelvis_z = float(out_pos[:, 2].mean())
    print(f"[make_rest_loop] wrote {args.out}")
    print(f"                  out motion key  = {out_key!r}")
    print(f"                  source key      = {key!r}")
    print(f"                  frames kept     = {e - s} ({(e - s)/fps:.2f} s @ {fps:g} fps)")
    print(f"                  pelvis_z mean   = {pelvis_z:+.3f} m")
    print(f"                  arm std (mean)  = {arm_std:.4f} rad")
    print(f"                  arm means (rad) [Lsh_p Lsh_r Lsh_y Lel Rsh_p Rsh_r Rsh_y Rel]")
    print(f"                                  = {np.array2string(arm_means, precision=3, suppress_small=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
