#!/usr/bin/env python3
"""Merge a base BONES-SEED motion_lib PKL with a halfspeed variant PKL.

Round-2 kplanner training uses two PKLs:

* ``base``       -- 1.0x speed corpus (e.g. ``x2_ultra_bones_seed_chain_matched.pkl``),
                    built with ``--fps-source 120``. Used standalone for the VQVAE
                    stage (no halfspeed redundancy in codebook training).
* ``halfspeed``  -- locowalk-only 0.5x variant (e.g.
                    ``x2_ultra_bones_seed_chain_matched_halfspeed.pkl``), built
                    with ``--fps-source 60 --min-mean-speed 0.20
                    --min-mean-yaw-rate 0.15`` so non-locomotion clips are dropped.

This script writes a third PKL that contains all entries from base PLUS the
halfspeed entries with their keys suffixed by ``__speed_0.5`` so the two
trajectories remain distinct in the trainer's motion_lib. Pose + Root stages
load this combined PKL to expand intent-space coverage on the slow-velocity end
that round-1 VR teleop hit out-of-distribution.

The merged PKL is roughly base + halfspeed entries in count, with no value
overwrites (the speed suffix collision-proofs the namespace).

Run from anywhere (no repo-relative deps):

  python motionbricks/scripts/cloud/merge_halfspeed_pkl.py \\
      --base gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched.pkl \\
      --halfspeed gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_halfspeed.pkl \\
      --out gear_sonic/data/motions/x2_ultra_bones_seed_chain_matched_v2.pkl

Halfspeed key suffix is configurable via ``--halfspeed-key-suffix`` so the same
script can produce other-fraction variants in the future (e.g. ``__speed_0.75``).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True,
                        help="Base 1.0x motion_lib PKL (e.g. x2_ultra_bones_seed_chain_matched.pkl)")
    parser.add_argument("--halfspeed", type=Path, required=True,
                        help="Halfspeed motion_lib PKL to merge in (e.g. x2_ultra_bones_seed_chain_matched_halfspeed.pkl)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output merged PKL path")
    parser.add_argument("--halfspeed-key-suffix", default="__speed_0.5",
                        help="Suffix appended to every halfspeed key (default '__speed_0.5')")
    parser.add_argument("--compress", type=int, default=3,
                        help="joblib.dump compress level (default 3)")
    args = parser.parse_args()

    if not args.base.is_file():
        print(f"ERROR: base PKL not found: {args.base}", file=sys.stderr)
        return 1
    if not args.halfspeed.is_file():
        print(f"ERROR: halfspeed PKL not found: {args.halfspeed}", file=sys.stderr)
        return 1

    print(f"Loading base PKL:      {args.base}")
    base = joblib.load(args.base)
    print(f"  base entries:        {len(base):,}")

    print(f"Loading halfspeed PKL: {args.halfspeed}")
    half = joblib.load(args.halfspeed)
    print(f"  halfspeed entries:   {len(half):,}")

    print(f"Halfspeed key suffix:  {args.halfspeed_key_suffix!r}")
    merged: dict[str, dict] = dict(base)
    collisions = 0
    for k, v in half.items():
        new_k = f"{k}{args.halfspeed_key_suffix}"
        if new_k in merged:
            collisions += 1
        merged[new_k] = v
    if collisions:
        print(f"  WARNING: {collisions} key collisions with base namespace (overwritten)")

    print(f"\nMerged entries:        {len(merged):,}")
    print(f"  expected:            {len(base) + len(half):,} (base + halfspeed)")
    if len(merged) != len(base) + len(half):
        print(f"  delta:               {len(merged) - (len(base) + len(half))} "
              "(non-zero implies key collisions before suffix; investigate)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting merged PKL:    {args.out}")
    joblib.dump(merged, args.out, compress=args.compress)
    size_mb = args.out.stat().st_size / 1e6
    print(f"  size:                {size_mb:.1f} MB")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
