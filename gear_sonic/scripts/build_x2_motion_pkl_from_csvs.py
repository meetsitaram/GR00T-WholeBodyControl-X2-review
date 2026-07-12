"""Combine a folder of already-retargeted X2 soma CSVs into one motion-lib PKL.

Unlike ``g1_captures_to_x2_motion_pkl.py`` (which trims + retargets), this
assumes the X2 CSVs are the FINAL retargeted output (e.g. from the batched
G1->X2 driver over the G1-SONIC executed-feasible corpus) and only does the
CSV -> motion-lib entry conversion + combine step. Reuses the same
``_x2_csv_to_entry`` so the PKL matches the bones_seed format exactly.

    python gear_sonic/scripts/build_x2_motion_pkl_from_csvs.py \
        --x2-dir gear_sonic/data/motions/x2_sonic_executed_feasible/csv \
        --out-pkl gear_sonic/data/motions/x2_sonic_executed_feasible.pkl --fps 50

Large corpora (tens of thousands of clips) are held in RAM before the single
joblib.dump; pass --shard N to write N-clip shards instead (out-pkl becomes a
stem, files are <stem>_000.pkl ...).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
from gear_sonic.scripts.g1_captures_to_x2_motion_pkl import _x2_csv_to_entry  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--x2-dir", required=True, type=Path, help="folder of retargeted X2 soma CSVs")
    ap.add_argument("--out-pkl", required=True, type=Path)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--shard", type=int, default=0, help="clips per shard (0 = single PKL)")
    ap.add_argument("--compress", type=int, default=3)
    ap.add_argument("--merge-pkl", nargs="*", default=[], type=Path,
                    help="extra motion-lib PKLs whose entries are merged in "
                         "(e.g. already-converted X2 teleop walks); fps must match --fps")
    args = ap.parse_args()

    csvs = sorted(args.x2_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs in {args.x2_dir}")
    print(f"[build] {len(csvs)} X2 CSVs in {args.x2_dir}", flush=True)
    args.out_pkl.parent.mkdir(parents=True, exist_ok=True)

    def flush(motions: dict, path: Path) -> None:
        joblib.dump(motions, path, compress=args.compress)
        secs = sum(m["dof"].shape[0] for m in motions.values()) / args.fps
        print(f"[build] wrote {path}: {len(motions)} clips, {secs:.0f}s", flush=True)

    motions: dict = {}
    shard_idx, n_total = 0, 0
    for i, xc in enumerate(csvs):
        try:
            motions[xc.stem] = _x2_csv_to_entry(xc, args.fps)
        except Exception as e:  # skip a malformed CSV rather than abort the whole build
            print(f"[build] SKIP {xc.name}: {e}", flush=True)
            continue
        n_total += 1
        if i % 2000 == 0:
            print(f"[build] {i + 1}/{len(csvs)}", flush=True)
        if args.shard and len(motions) >= args.shard:
            flush(motions, args.out_pkl.with_name(f"{args.out_pkl.stem}_{shard_idx:03d}.pkl"))
            motions = {}
            shard_idx += 1
    n_merged = 0
    if args.merge_pkl:
        if args.shard:
            raise SystemExit("--merge-pkl is only supported in single-PKL mode (drop --shard)")
        for mp in args.merge_pkl:
            extra = joblib.load(mp)
            added = 0
            for name, entry in extra.items():
                efps = int(entry.get("fps", args.fps))
                if efps != args.fps:
                    raise SystemExit(f"{mp}:{name} fps={efps} != --fps {args.fps}; resample first")
                key = name if name not in motions else f"{name}__merged"
                motions[key] = entry
                added += 1
            n_merged += added
            print(f"[build] merged {added} clips from {mp}", flush=True)
    if motions:
        out = (args.out_pkl if not args.shard
               else args.out_pkl.with_name(f"{args.out_pkl.stem}_{shard_idx:03d}.pkl"))
        flush(motions, out)
    print(f"[build] DONE: {n_total} converted + {n_merged} merged = {n_total + n_merged} clips", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
