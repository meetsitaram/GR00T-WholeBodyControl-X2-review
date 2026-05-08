#!/usr/bin/env python3
"""Build combined X2 motion_lib PKL from BONES-SEED retargeted CSV subsets.

Reads CSVs from each subset directory under
  agibot-x2-references/bones-seed/retargeted/x2/<subset>/*.csv
converts every CSV to a motion_lib entry (using the existing
``convert_soma_csv_to_motion_lib`` helpers), prefixes the motion name with the
subset to keep all retarget variants distinct, and writes:

  gear_sonic/data/motions/x2_ultra_bones_seed.pkl   # combined
  gear_sonic/data/motions/x2_ultra_loco_manipulation.pkl
  gear_sonic/data/motions/x2_ultra_standing_manipulation.pkl

Run from repo root:
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/data_process/build_x2_bones_seed_motion_lib.py
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    convert_sequence,
    downsample_sequence,
    load_bones_csv,
    set_robot,
)


def _convert_one(args):
    csv_path, fps_source, fps_target, robot = args
    set_robot(robot)
    seq = load_bones_csv(csv_path)
    entry = convert_sequence(seq, fps_source)
    if fps_source != fps_target:
        entry = downsample_sequence(entry, fps_source, fps_target)
    return Path(csv_path).stem, entry


def convert_subset(subset_dir: Path, fps_source: int, fps_target: int, workers: int) -> dict:
    csvs = sorted(subset_dir.glob("*.csv"))
    print(f"  {subset_dir.name}: {len(csvs)} CSVs")
    out: dict[str, dict] = {}
    args = [(str(p), fps_source, fps_target, "x2_ultra") for p in csvs]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_convert_one, a) for a in args]
        done = 0
        for fut in as_completed(futures):
            name, entry = fut.result()
            out[name] = entry
            done += 1
            if done % 100 == 0 or done == len(args):
                print(f"    [{subset_dir.name}] {done}/{len(args)}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retargeted-root",
        default=str(REPO.parent / "GR00T-WholeBodyControl"
                    / "agibot-x2-references" / "bones-seed" / "retargeted" / "x2"),
        help="Parent directory containing <subset>/*.csv folders",
    )
    parser.add_argument("--out-dir", default=str(REPO / "gear_sonic" / "data" / "motions"))
    parser.add_argument("--fps-source", type=int, default=120,
                        help="BONES-SEED native frame rate (default 120)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target motion_lib frame rate (default 30)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--subsets", nargs="+",
                        default=["loco-manipulation", "standing-manipulation"])
    args = parser.parse_args()

    rt = Path(args.retargeted_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source FPS: {args.fps_source}  ->  Target FPS: {args.fps}")
    print(f"Workers: {args.workers}")
    print(f"Retargeted root: {rt}")
    print(f"Output dir: {out_dir}")

    combined: dict[str, dict] = {}
    for subset in args.subsets:
        subset_dir = rt / subset
        if not subset_dir.is_dir():
            print(f"WARNING: missing {subset_dir}, skipping")
            continue
        entries = convert_subset(subset_dir, args.fps_source, args.fps, args.workers)
        per_subset_path = out_dir / f"x2_ultra_{subset.replace('-', '_')}.pkl"
        print(f"  Saving {per_subset_path} ({len(entries)} entries)")
        joblib.dump(entries, per_subset_path, compress=3)
        # merge with subset prefix to keep retarget variants distinct
        prefix = subset.split("-")[0]  # 'loco' or 'standing'
        for k, v in entries.items():
            combined[f"{prefix}__{k}"] = v

    combined_path = out_dir / "x2_ultra_bones_seed.pkl"
    print(f"\nSaving combined: {combined_path} ({len(combined)} entries)")
    joblib.dump(combined, combined_path, compress=3)
    print("Done.")


if __name__ == "__main__":
    main()
