#!/usr/bin/env python3
"""Build the curated deploy visual-regression clip suite.

The suite PKL lives under ``gear_sonic/data/motions/`` which is gitignored, so
this script regenerates it deterministically from the source motion libraries.
``deploy_regression_check.sh`` calls it automatically when the suite is missing.

Coverage (deliberately spans the motion classes a deploy can regress):
  2x locomotion (straight walk + walk-with-turns)
  2x easy dance
  2x medium dance
  1x combat / boxing

Usage:
    python gear_sonic/scripts/build_deploy_regression_suite.py [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[2]
MOTIONS = REPO / "gear_sonic" / "data" / "motions"
DEFAULT_OUT = MOTIONS / "deploy_regression_suite.pkl"

# (source pkl stem, clip key, label prefix used in the suite)
SPEC: list[tuple[str, str, str]] = [
    ("x2_g1teleop_30fps", "slow_walk_0.5_001", "walk"),
    ("x2_g1teleop_30fps", "slow_walk_turns_0.5_001", "walk"),
    ("x2_dances_easy", "dance_party_hips_003__A467", "easy"),
    ("x2_dances_easy", "dance_freedom_wheels_001__A465", "easy"),
    ("x2_dances_medium", "dance_disco_fever_001__A465", "medium"),
    ("x2_dances_medium", "dance_hiphop_funky_guitar_R_fast_001__A319", "medium"),
    ("x2_combat_dance_finetune", "ROM_Box_01_Box_02_Box_03_Box_04_002__A520", "combat"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    out: dict = {}
    missing: list[str] = []
    cache: dict[str, dict] = {}
    for stem, key, label in SPEC:
        src = MOTIONS / f"{stem}.pkl"
        if not src.is_file():
            missing.append(f"{stem}.pkl (source missing)")
            continue
        if stem not in cache:
            cache[stem] = joblib.load(src)
        lib = cache[stem]
        # exact key, else unique substring match (clip names carry long suffixes)
        k = key if key in lib else next((x for x in lib if key in x), None)
        if k is None:
            missing.append(f"{stem}:{key}")
            continue
        out[f"{label}__{k[:40]}"] = lib[k]

    if missing:
        print("WARNING: could not resolve:", ", ".join(missing), file=sys.stderr)
    if not out:
        print("ERROR: no clips resolved; suite not written", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, args.out)
    print(f"wrote {args.out}  ({len(out)} clips)")
    for k in out:
        print("  -", k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
