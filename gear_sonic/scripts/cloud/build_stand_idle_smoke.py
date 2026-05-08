#!/usr/bin/env python3
"""Carve a tiny stand-still smoke PKL out of the BONES-SEED motion library.

The full ``x2_ultra_bones_seed.pkl`` (~210 MB, 2,550 motions) is overkill for
verifying the training pipeline end-to-end. This script picks the
``neutral_idle_loop`` clips (pure standing, no props, no walking) and writes
them to a small smoke PKL that's safe to use with ``num_envs >= 4096``
without the policy needing to learn anything aggressive.

Run on the cloud node from the repo root, after the BONES-SEED bundle has
been unpacked:

    conda activate env_isaaclab
    python gear_sonic/scripts/cloud/build_stand_idle_smoke.py

Output: ``gear_sonic/data/motions/x2_ultra_stand_idle_smoke.pkl`` (~1.6 MB).

Then reference it from your launch with::

    ++manager_env.commands.motion.motion_lib_cfg.motion_file=\\
        gear_sonic/data/motions/x2_ultra_stand_idle_smoke.pkl
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO / "gear_sonic" / "data" / "motions" / "x2_ultra_bones_seed.pkl"
DEFAULT_OUT = REPO / "gear_sonic" / "data" / "motions" / "x2_ultra_stand_idle_smoke.pkl"
DEFAULT_FILTER = "neutral_idle_loop"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=str(DEFAULT_SRC),
                        help=f"Source motion-lib PKL (default: {DEFAULT_SRC})")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help=f"Output PKL (default: {DEFAULT_OUT})")
    parser.add_argument("--match", default=DEFAULT_FILTER,
                        help=("Substring that must appear in the motion key. "
                              f"Default: {DEFAULT_FILTER!r}."))
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_file():
        sys.exit(f"ERROR: source PKL not found: {src}")

    print(f"Loading {src} ...")
    full = joblib.load(src)
    print(f"  {len(full):,} motions in source")

    keep = {k: v for k, v in full.items() if args.match in k}
    if not keep:
        sys.exit(f"ERROR: no motions matched substring {args.match!r}")

    print(f"  {len(keep)} motions match {args.match!r}:")
    for k, v in keep.items():
        n = len(v["dof"])
        fps = v.get("fps")
        print(f"    {k}  ({n} frames @ {fps} fps)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(keep, out, compress=3)
    size_kb = os.path.getsize(out) / 1024
    print(f"\nWrote {out} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
