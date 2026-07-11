"""Turn a folder of captured G1 planner motions into one X2 motion-lib PKL
ready to mix into the X2 SONIC fine-tune corpus.

Pipeline per clip:  trim idle -> G1->X2 retarget -> motion-lib entry, then
combine all clips into a single {name: entry} PKL.

Each input is a retargeter-ready G1 CSV (as written by
``record_motion_to_pkl.py --g1-csv``). Run after capturing several G1 planner
motions (keyboard / Quest3) into one folder:

    python gear_sonic/scripts/g1_captures_to_x2_motion_pkl.py \\
        --g1-dir /tmp/g1_ft --out-pkl gear_sonic/data/motions/x2_g1planner_finetune_v1.pkl \\
        --fps 50

The motion keys are the CSV stems. Idle startup/stop is auto-trimmed to the
window where the root actually moves (>= --min-speed). Pass --no-trim to keep
full clips.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO.parent))
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    X2_DOF_AXIS, X2_NUM_BODIES, X2_NUM_DOF,
)

_RETARGETER = REPO.parent / "agibot-x2-references" / "soma-retargeter"
_RT_PY = _RETARGETER / ".venv" / "bin" / "python"


def _trim(csv_in: Path, csv_out: Path, fps: int, min_speed: float) -> tuple[int, float]:
    d = pd.read_csv(csv_in)
    rt = d[["root_translateX", "root_translateY", "root_translateZ"]].values / 100.0
    sp = np.linalg.norm(np.diff(rt[:, :2], axis=0), axis=1) * fps
    k = min(15, max(1, len(sp) // 4))
    spf = np.convolve(sp, np.ones(k) / k, mode="same")
    moving = spf > min_speed
    if moving.any():
        i0 = int(np.argmax(moving))
        i1 = int(len(moving) - np.argmax(moving[::-1]))
    else:
        i0, i1 = 0, len(d)
    d = d.iloc[i0:i1].reset_index(drop=True)
    d["Frame"] = range(len(d))
    d.to_csv(csv_out, index=False)
    win_speed = float(sp[i0:max(i0 + 1, i1 - 1)].mean()) if i1 > i0 + 1 else 0.0
    return len(d), win_speed


def _x2_csv_to_entry(csv: Path, fps: int) -> dict:
    d = pd.read_csv(csv)
    rt = np.stack([d["root_translateX"], d["root_translateY"], d["root_translateZ"]], 1) / 100.0
    rq = R.from_euler("xyz",
                      np.stack([d["root_rotateX"], d["root_rotateY"], d["root_rotateZ"]], 1),
                      degrees=True).as_quat()
    jc = [c for c in d.columns if c.endswith("_dof")]
    dof = np.deg2rad(d[jc].values)
    assert dof.shape[1] == X2_NUM_DOF, (dof.shape, X2_NUM_DOF)
    T = len(dof)
    pa = np.zeros((T, X2_NUM_BODIES, 3), np.float32)
    pa[:, 1:X2_NUM_DOF + 1, :] = np.asarray(X2_DOF_AXIS)[None] * dof[:, :, None]
    pa[:, 0, :] = R.from_quat(rq).as_rotvec()
    return {
        "root_trans_offset": rt.astype(np.float32),
        "root_rot": rq.astype(np.float32),
        "dof": dof.astype(np.float32),
        "pose_aa": pa,
        "smpl_joints": np.zeros((T, 24, 3), np.float32),
        "fps": int(fps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--g1-dir", required=True, type=Path, help="folder of captured G1 CSVs")
    ap.add_argument("--out-pkl", required=True, type=Path, help="combined X2 motion PKL")
    ap.add_argument("--fps", type=int, default=50, help="capture fps (default 50)")
    ap.add_argument("--min-speed", type=float, default=0.15, help="active-walk speed threshold m/s")
    ap.add_argument("--no-trim", action="store_true")
    args = ap.parse_args()

    csvs = sorted(args.g1_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no CSVs in {args.g1_dir}")
    print(f"[ft] {len(csvs)} G1 captures in {args.g1_dir}")

    with tempfile.TemporaryDirectory() as td:
        trim_dir, x2_dir = Path(td) / "trim", Path(td) / "x2"
        trim_dir.mkdir(); x2_dir.mkdir()
        # 1. trim
        for c in csvs:
            if args.no_trim:
                (trim_dir / c.name).write_text(c.read_text())
                continue
            n, spd = _trim(c, trim_dir / c.name, args.fps, args.min_speed)
            print(f"[ft]   trim {c.stem:32s} -> {n} frames ({n/args.fps:.1f}s) @ {spd:.2f} m/s")
        # 2. retarget (batched, in the soma-retargeter venv)
        print("[ft] retargeting G1 -> X2 ...")
        subprocess.run([str(_RT_PY), "app/g1_csv_to_x2_csv.py",
                        "--g1-dir", str(trim_dir), "--out-dir", str(x2_dir)],
                       cwd=str(_RETARGETER), check=True)
        # 3. combine into one motion PKL
        motions = {}
        for xc in sorted(x2_dir.glob("*.csv")):
            motions[xc.stem] = _x2_csv_to_entry(xc, args.fps)
        args.out_pkl.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(motions, args.out_pkl, compress=3)

    total_s = sum(m["dof"].shape[0] for m in motions.values()) / args.fps
    print(f"[ft] wrote {args.out_pkl}: {len(motions)} clips, {total_s:.0f}s total")
    print(f"[ft] keys: {', '.join(list(motions)[:6])}{' ...' if len(motions) > 6 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
