"""Byte-diff two pose-wire captures (.npz from capture_pose_wire.py).

Compares per-frame content between two captures of the deploy's pose
topic. Shows per-field max abs delta and flags real divergences.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--n", type=int, default=100, help="Compare first N frames")
    args = p.parse_args()

    da = np.load(args.a)
    db = np.load(args.b)
    n = min(args.n, len(da[next(iter(da.files))]), len(db[next(iter(db.files))]))
    print(f"comparing first {n} frames of:")
    print(f"  A: {args.a}")
    print(f"  B: {args.b}")

    for k in sorted(da.files):
        if k not in db.files:
            print(f"  {k}: missing in B")
            continue
        a = np.asarray(da[k])[:n]
        b = np.asarray(db[k])[:n]
        if a.shape != b.shape:
            print(f"  {k}: shape A={a.shape} B={b.shape}")
            continue
        if np.issubdtype(a.dtype, np.number):
            diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
            mx = float(diff.max())
            mn = float(diff.mean())
            tag = "  " if mx < 1e-4 else "!!"
            print(
                f"{tag} {k:30s} shape={str(a.shape):20s} "
                f"max|A-B|={mx:.6e} mean|A-B|={mn:.6e}"
            )
            if mx >= 1e-4 and a.ndim <= 2:
                worst_idx = np.unravel_index(diff.argmax(), diff.shape)
                print(
                    f"     worst at {worst_idx}: A={a[worst_idx]:.4f} "
                    f"B={b[worst_idx]:.4f}"
                )
        else:
            print(f"  {k}: non-numeric, dtype={a.dtype}")

    print()
    print("--- frame_index sequences ---")
    for label, d in [("A", da), ("B", db)]:
        fi = d["frame_index"][:5].flatten()
        fi_end = d["frame_index"][-5:].flatten()
        print(f"  {label}: first={fi.tolist()} last={fi_end.tolist()}")

    print()
    print("--- per-frame motion (joint_pos_mj variation) ---")
    for label, d in [("A", da), ("B", db)]:
        jp = d["joint_pos_mj"][:n]
        var = jp.max(axis=0) - jp.min(axis=0)
        print(
            f"  {label}: per-DOF variation max={var.max():.6e} "
            f"mean={var.mean():.6e}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
