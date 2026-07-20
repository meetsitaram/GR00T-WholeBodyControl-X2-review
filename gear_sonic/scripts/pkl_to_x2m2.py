#!/usr/bin/env python3
"""Window a motion-lib clip and write it as X2M2 for the dance/clip player.

WHY: to isolate SONIC from the kplanner. The dance path streams a FIXED
reference clip straight to SONIC, bypassing planner generation entirely. If
SONIC tracks a known-good walk cleanly but stumbles on planner output, the
problem is upstream; if it stumbles on both, it is the tracker.

Windowing matters on a gantry: full corpus clips run 10 s and travel metres.
A few seconds of a low-speed turning clip gives a couple of steps and a turn
without crossing the room.

The written file is round-trip verified with the repo's OWN loader
(``gear_sonic.utils.pose_pipeline.wire.load_x2m2``) before it is accepted --
a hand-written binary format that only this script can read would be worse
than useless.

    python gear_sonic/scripts/pkl_to_x2m2.py \
        --pkl gear_sonic/data/motions/x2_g1teleop_30fps.pkl \
        --key slow_walk_turns_0.2_001 --start-s 1.0 --dur-s 5.0 \
        --out /tmp/probe_walk_turn.x2m2
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gear_sonic.utils.pose_pipeline.wire import (  # noqa: E402
    NUM_BODY_DOFS, X2M2_MAGIC, load_x2m2)


def write_x2m2(path: Path, dof: np.ndarray, quat_xyzw: np.ndarray,
               fps: float) -> None:
    n, d = dof.shape
    if d != NUM_BODY_DOFS:
        raise ValueError(f"dof has {d} cols, need {NUM_BODY_DOFS}")
    if quat_xyzw.shape != (n, 4):
        raise ValueError(f"quat shape {quat_xyzw.shape}, need ({n}, 4)")
    flat = np.concatenate([dof, quat_xyzw], axis=1).astype(np.float64)
    with path.open("wb") as f:
        f.write(struct.pack("<III", X2M2_MAGIC, n, d))
        f.write(struct.pack("<d", float(fps)))
        f.write(flat.tobytes(order="C"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--key", required=True)
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--dur-s", type=float, default=5.0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    lib = joblib.load(args.pkl)
    if args.key not in lib:
        cand = [k for k in lib if args.key in k]
        print(f"key {args.key!r} not found; close matches: {cand[:5]}",
              file=sys.stderr)
        return 1
    clip = lib[args.key]
    dof = np.asarray(clip["dof"], dtype=np.float64)
    rot = np.asarray(clip["root_rot"], dtype=np.float64)      # xyzw
    tr = np.asarray(clip["root_trans_offset"], dtype=np.float64)
    fps = float(clip.get("fps", 30.0))

    a = int(round(args.start_s * fps))
    b = min(dof.shape[0], a + int(round(args.dur_s * fps)))
    dof, rot, tr = dof[a:b], rot[a:b], tr[a:b]
    n = dof.shape[0]
    if n < 2:
        print("window too short", file=sys.stderr)
        return 1

    write_x2m2(args.out, dof, rot, fps)

    # Round-trip through the REPO loader -- the only check that matters.
    rd, rq, rf = load_x2m2(args.out)
    ok = (rd.shape == dof.shape and rq.shape == rot.shape
          and abs(rf - fps) < 1e-9
          and np.allclose(rd, dof.astype(np.float32), atol=1e-6)
          and np.allclose(rq, rot.astype(np.float32), atol=1e-6))
    disp = float(np.linalg.norm(tr[-1, :2] - tr[0, :2]))
    path_len = float(np.sum(np.linalg.norm(np.diff(tr[:, :2], axis=0), axis=1)))
    jump = float(np.abs(np.diff(dof, axis=0)).max())
    print(f"  {args.key}[{a}:{b}]  {n} frames @ {fps:g}fps = {n/fps:.1f}s")
    print(f"  travel: net {disp:.2f} m, path {path_len:.2f} m")
    print(f"  max per-frame joint jump: {jump:.3f} rad")
    print(f"  round-trip via load_x2m2: {'OK' if ok else '*** MISMATCH ***'}")
    print(f"  wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
