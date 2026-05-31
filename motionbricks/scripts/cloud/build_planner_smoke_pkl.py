#!/usr/bin/env python3
"""Carve a tiny multi-clip smoke PKL out of the BONES-SEED motion library.

The full ``x2_ultra_bones_seed.pkl`` (~210 MB, 2,550 motions) trains a real
planner in many hours. For a ~3 min smoke that exercises VQVAE + pose + root
on every GPU, we want ~30 clips covering walk / turn / idle so the perplexity
moves and the dataloaders, FK extractor, and DDP all-reduce are all stressed.

Run on the cloud node from the repo root, after the BONES-SEED bundle has
been unpacked:

    conda activate motionbricks
    python motionbricks/scripts/cloud/build_planner_smoke_pkl.py

Output: ``gear_sonic/data/motions/x2_ultra_planner_smoke.pkl`` (~3 MB).

Then reference it from your launch with::

    PKL=gear_sonic/data/motions/x2_ultra_planner_smoke.pkl \\
        bash motionbricks/scripts/cloud/run_planner_smoke_8gpu.sh
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[3]
DEFAULT_SRC = REPO / "gear_sonic" / "data" / "motions" / "x2_ultra_bones_seed.pkl"
DEFAULT_OUT = REPO / "gear_sonic" / "data" / "motions" / "x2_ultra_planner_smoke.pkl"

# Default include patterns — broad locomotion coverage (walk + turn + idle)
# without the noisy ``neutral_*`` standalone clips.
DEFAULT_INCLUDE = [
  r"loco__.*walk.*",
  r"loco__.*turn.*",
  r"loco__.*run.*",
  r"loco__.*idle.*",
  r"stable-loco__.*walk.*",
]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--src", default=str(DEFAULT_SRC), help=f"Source motion-lib PKL (default: {DEFAULT_SRC})"
  )
  parser.add_argument(
    "--out", default=str(DEFAULT_OUT), help=f"Output PKL (default: {DEFAULT_OUT})"
  )
  parser.add_argument(
    "--max-clips",
    type=int,
    default=30,
    help="Maximum number of clips to keep (default: 30 — enough for ~3 min smoke).",
  )
  parser.add_argument(
    "--min-frames",
    type=int,
    default=80,
    help="Skip clips shorter than this (default: 80, matches train_*_x2.py).",
  )
  parser.add_argument(
    "--max-frames",
    type=int,
    default=200,
    help="Skip clips longer than this (default: 200, matches train_*_x2.py).",
  )
  parser.add_argument(
    "--include",
    nargs="+",
    default=DEFAULT_INCLUDE,
    help="Regex patterns; a clip is kept if any pattern matches its key.",
  )
  args = parser.parse_args()

  src = Path(args.src)
  if not src.is_file():
    sys.exit(f"ERROR: source PKL not found: {src}")

  print(f"Loading {src} ...")
  full = joblib.load(src)
  print(f"  {len(full):,} motions in source")

  patterns = [re.compile(p) for p in args.include]
  candidates = [k for k in full.keys() if any(p.search(k) for p in patterns)]
  candidates.sort()
  print(f"  {len(candidates)} keys matched include patterns")

  kept = {}
  for key in candidates:
    if len(kept) >= args.max_clips:
      break
    n = int(full[key]["dof"].shape[0])
    if n < args.min_frames or n > args.max_frames:
      continue
    kept[key] = full[key]

  if not kept:
    sys.exit("ERROR: no motions kept after frame-count filtering")

  print(f"  {len(kept)} motions kept after frame filtering "
        f"(min={args.min_frames} max={args.max_frames}):")
  for k, v in list(kept.items())[:10]:
    n = len(v["dof"])
    fps = v.get("fps")
    print(f"    {k}  ({n} frames @ {fps} fps)")
  if len(kept) > 10:
    print(f"    ... and {len(kept) - 10} more")

  out = Path(args.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  joblib.dump(kept, out, compress=3)
  size_kb = os.path.getsize(out) / 1024
  print(f"\nWrote {out} ({size_kb:.1f} KB / {len(kept)} clips)")


if __name__ == "__main__":
  main()
