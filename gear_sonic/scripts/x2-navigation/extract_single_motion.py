#!/usr/bin/env python3
"""Extract ONE motion from a multi-motion corpus pkl into its own file.

Why this exists (session 2026-07-21): the IsaacLab tracking env RSIs every
reset from the loaded motion file. Pointing it at a whole corpus (e.g. the
4.3 GB x2_sonic_executed_feasible.pkl) makes resets resample ARBITRARY
motions — including crawl/faint floor clips → upside-down respawns — and
costs minutes of load time. A single-clip scaffold makes spawns
deterministic and boot fast.

Usage:
    python extract_single_motion.py --src <corpus.pkl> --key <motion_key> \
        --out <single.pkl> [--max-travel 0.2]

    # key omitted -> first motion in the corpus; --list to see keys
"""
import argparse

import joblib
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out")
    ap.add_argument("--key", default=None, help="default: first motion")
    ap.add_argument("--list", action="store_true", help="list keys and exit")
    ap.add_argument("--max-travel", type=float, default=None,
                    help="abort if clip travels farther (guard for stand clips)")
    args = ap.parse_args()

    d = joblib.load(args.src)
    if args.list:
        for k in d:
            print(k)
        return 0
    key = args.key or next(iter(d))
    if key not in d:
        raise SystemExit(f"{key!r} not in corpus; use --list")
    m = d[key]
    tr = np.asarray(m["root_trans_offset"])
    travel = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
    print(f"{key}: frames={len(tr)} fps={m.get('fps')} travel={travel:.3f}m "
          f"keys={sorted(m.keys())}")
    if args.max_travel is not None and travel > args.max_travel:
        raise SystemExit(f"travel {travel:.3f} > --max-travel {args.max_travel}")
    if "pose_aa" not in m:
        print("WARNING: no pose_aa — env loader will reject this clip "
              "(add zeros (N,32,3) if fix_height is no_fix)")
    if not args.out:
        raise SystemExit("--out required to write")
    joblib.dump({key: m}, args.out, compress=3)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
