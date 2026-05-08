#!/usr/bin/env python3
"""Stitch a warehouse-task playlist YAML into a single motion-lib PKL.

Thin CLI wrapper over :mod:`_warehouse_playlist`. The interesting code is
there; this script only:

    1. Loads a playlist YAML.
    2. Runs ``build_concat`` to chain segments + interleave rest layers.
    3. Writes the resulting ``{output_key: motion}`` dict to ``--out``.
    4. Prints per-segment / per-seam / end-to-end diagnostics.
    5. (Optional) Re-loads the written PKL and asserts ``np.allclose`` with
       the in-memory ``build_concat`` output, so ``eval_x2_mujoco
       --playlist`` is byte-equivalent to ``eval_x2_mujoco --motion`` on
       the freshly written file.

Example:
    python gear_sonic/scripts/make_warehouse_motion.py \\
        --playlist gear_sonic/data/motions/playlists/warehouse_v1.yaml \\
        --out      gear_sonic/data/motions/x2_ultra_warehouse_v1.pkl \\
        --check-runtime-parity

Then preview kinematically:
    python gear_sonic/scripts/play_motion_mujoco.py \\
        --motion gear_sonic/data/motions/x2_ultra_warehouse_v1.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _warehouse_playlist import (  # noqa: E402
    build_concat,
    diagnostics_lines,
    load_playlist,
    write_pkl,
)


def _parity_check(motion_in_memory: dict, out_path: Path) -> bool:
    """Re-load `out_path` and verify it matches `motion_in_memory` exactly."""
    reread = joblib.load(out_path)
    if len(reread) != 1:
        print(f"[parity] FAIL: written PKL has {len(reread)} keys, expected 1")
        return False
    written = next(iter(reread.values()))
    ok = True
    for k in ("dof", "root_rot", "root_trans_offset"):
        a = np.asarray(motion_in_memory[k])
        b = np.asarray(written[k])
        if a.shape != b.shape:
            print(f"[parity] FAIL: {k} shape {a.shape} vs {b.shape}")
            ok = False
            continue
        if not np.allclose(a, b, rtol=0, atol=0):
            diff = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            print(f"[parity] FAIL: {k} max|delta|={diff:.3e}")
            ok = False
        else:
            print(f"[parity] OK   {k} {a.shape} dtype={a.dtype}")
    if float(motion_in_memory["fps"]) != float(written["fps"]):
        print(f"[parity] FAIL: fps {motion_in_memory['fps']} vs {written['fps']}")
        ok = False
    return ok


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--playlist", required=True, type=Path,
                   help="Path to a warehouse playlist YAML.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output motion-lib PKL path.")
    p.add_argument("--check-runtime-parity", action="store_true",
                   help="After writing, re-load the PKL and assert it matches "
                        "the in-memory build_concat output bit-for-bit. Cheap "
                        "regression that the runtime --playlist path in "
                        "eval_x2_mujoco is equivalent to --motion on the .pkl.")
    args = p.parse_args(argv)

    print(f"[stitch] loading playlist {args.playlist}", flush=True)
    playlist = load_playlist(args.playlist)
    print(
        f"[stitch] playlist '{playlist.name}' "
        f"({len(playlist.segments)} segments, fps={playlist.fps:g})",
        flush=True,
    )

    motion = build_concat(playlist)
    print(f"[stitch] writing -> {args.out}", flush=True)
    write_pkl(playlist, motion, args.out)

    print("")
    for line in diagnostics_lines(playlist, motion):
        print(line)

    if args.check_runtime_parity:
        print("")
        ok = _parity_check(motion, args.out)
        if not ok:
            print("[parity] FAILED", file=sys.stderr)
            return 2
        print("[parity] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
