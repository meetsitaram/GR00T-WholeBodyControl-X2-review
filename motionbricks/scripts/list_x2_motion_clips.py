#!/usr/bin/env python3
"""Inspect X2 motion-lib PKL(s): clip counts (filter on/off) + direction breakdown."""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List

import joblib

MB_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MB_ROOT.parent
sys.path.insert(0, str(MB_ROOT))

from motionbricks.data.x2_loco_filters import (  # noqa: E402
  DEFAULT_EXCLUDE_PATTERNS,
  DEFAULT_INCLUDE_PATTERNS,
  filter_motion_keys,
)


_DIRECTION_BUCKETS = [
  ("forward", re.compile(r"(?i)(?<![A-Za-z])(forward|fwd|front)(?![A-Za-z])")),
  ("backward", re.compile(r"(?i)(?<![A-Za-z])(backward|back|reverse)(?![A-Za-z])")),
  ("left", re.compile(r"(?i)(?<![A-Za-z])(left)(?![A-Za-z])")),
  ("right", re.compile(r"(?i)(?<![A-Za-z])(right)(?![A-Za-z])")),
  ("turn", re.compile(r"(?i)(?<![A-Za-z])(turn|pivot|rotate|spin)(?![A-Za-z])")),
  ("standing", re.compile(r"(?i)(?<![A-Za-z])(stand|standing|idle|rest)(?![A-Za-z])")),
]


def _bucket_keys(keys: Iterable[str]) -> Counter:
  counts: Counter = Counter()
  keys_list = list(keys)
  for k in keys_list:
    for name, pat in _DIRECTION_BUCKETS:
      if pat.search(k):
        counts[name] += 1
  counts["total"] = len(keys_list)
  return counts


def _summarize(label: str, keys: List[str], total: int) -> None:
  print(f"{label}: {len(keys)} / {total} clips ({100.0 * len(keys) / max(total, 1):.1f}%)")
  buckets = _bucket_keys(keys)
  for name, _ in _DIRECTION_BUCKETS:
    n = buckets.get(name, 0)
    pct = 100.0 * n / max(len(keys), 1)
    flag = "  <5% bucket" if (len(keys) > 0 and pct < 5.0) else ""
    print(f"    {name:>9}: {n:>5} ({pct:5.1f}%){flag}")


def main() -> None:
  p = argparse.ArgumentParser(description="List/summarize X2 motion-lib PKL clips")
  p.add_argument(
    "--pkl",
    type=Path,
    nargs="+",
    default=[REPO_ROOT / "gear_sonic/data/motions/x2_ultra_bones_seed.pkl"],
    help="One or more PKL paths (clips concatenated)",
  )
  p.add_argument("--filter", choices=["none", "loco"], default="none",
                 help="Apply walk/turn filter (off by default to match SONIC)")
  p.add_argument("--show", type=int, default=20, help="Print first N kept keys")
  args = p.parse_args()

  all_keys: List[str] = []
  for pkl in args.pkl:
    lib = joblib.load(pkl)
    keys = sorted(lib.keys())
    print(f"PKL: {pkl}    raw clips: {len(keys)}")
    all_keys.extend(keys)
    del lib

  total = len(all_keys)
  print(f"\nCombined raw clips: {total}\n")

  if args.filter == "none":
    kept = list(all_keys)
    print("--- filter: NONE (default, matches SONIC corpus) ---")
  else:
    kept = filter_motion_keys(
      all_keys,
      include_patterns=DEFAULT_INCLUDE_PATTERNS,
      exclude_patterns=DEFAULT_EXCLUDE_PATTERNS,
    )
    print("--- filter: LOCO (walk + turn + idle, manipulation excluded) ---")

  _summarize("kept", kept, total)

  print("\nFirst kept keys:")
  for k in kept[: args.show]:
    print(f"  {k}")
  if len(kept) > args.show:
    print(f"  ... +{len(kept) - args.show} more")


if __name__ == "__main__":
  main()
