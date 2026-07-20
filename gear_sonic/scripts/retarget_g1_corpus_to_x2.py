"""Retarget a whole G1-teleop corpus PKL to an X2 motion-lib PKL.

Explodes each clip in the G1 corpus to a soma-retargeter G1 CSV, then runs the
existing folder retargeter (g1_captures_to_x2_motion_pkl.py) over them.

    python gear_sonic/scripts/retarget_g1_corpus_to_x2.py \
        --g1-pkl gear_sonic/data/motions/g1_teleop_corpus_30fps.pkl \
        --out-pkl gear_sonic/data/motions/x2_g1teleop_30fps.pkl \
        --fps 30 --no-trim

Keeps idle start/stop by default via --no-trim (pass without it to auto-trim).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
from gear_sonic.scripts.record_motion_to_pkl import write_g1_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g1-pkl", required=True, type=Path)
    ap.add_argument("--out-pkl", required=True, type=Path)
    ap.add_argument("--fps", type=int, required=True)
    ap.add_argument("--no-trim", action="store_true")
    ap.add_argument("--csv-dir", type=Path, default=None,
                    help="keep exploded G1 CSVs here (default: temp, deleted)")
    args = ap.parse_args()

    data = joblib.load(args.g1_pkl)
    print(f"loaded {len(data)} G1 clips from {args.g1_pkl}")

    tmp = None
    if args.csv_dir:
        csv_dir = args.csv_dir
        csv_dir.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="g1csv_")
        csv_dir = Path(tmp.name)

    n = 0
    for key, entry in data.items():
        write_g1_csv(entry, csv_dir / f"{key}.csv")
        n += 1
    print(f"wrote {n} G1 CSVs -> {csv_dir}")

    cmd = [
        sys.executable, str(REPO / "scripts" / "g1_captures_to_x2_motion_pkl.py"),
        "--g1-dir", str(csv_dir),
        "--out-pkl", str(args.out_pkl),
        "--fps", str(args.fps),
    ]
    if args.no_trim:
        cmd.append("--no-trim")
    print("+", " ".join(cmd))
    rc = subprocess.call(cmd)
    if tmp:
        tmp.cleanup()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
