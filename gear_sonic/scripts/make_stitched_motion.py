#!/usr/bin/env python3
"""Stitch together two motion-lib PKL clips along the X2 Ultra body partition:

    - LOWER (legs + waist)         <- DOF indices 0..14   (15 joints)
    - UPPER (arms + wrists + head) <- DOF indices 15..30  (16 joints)

Use case:
    The relaxed-walk PKL has nice arms-down posture but the legs are
    mid-stride. The idle_hands_on_back PKL has rock-solid lower-body and
    waist but the arms are tucked behind. Combine the two -> "standing
    still on idle legs, arms hanging from the relaxed walk."

The root pose (qpos[0:7]) is taken from the LOWER source so the body
visibly stands where the legs say it stands.

Joint partition (matches MUJOCO_JOINT_NAMES in eval_x2_mujoco.py):
       0..5  left  leg     (hip p/r/y, knee, ankle p/r)
       6..11 right leg     (hip p/r/y, knee, ankle p/r)
      12..14 waist          (yaw, pitch, roll)
      15..21 left  arm      (shoulder p/r/y, elbow, wrist y/p/r)
      22..28 right arm      (shoulder p/r/y, elbow, wrist y/p/r)
      29..30 head           (pitch, yaw)

Example:
    python gear_sonic/scripts/make_stitched_motion.py \\
        --lower-source gear_sonic/data/motions/x2_ultra_idle_stand.pkl \\
        --upper-source gear_sonic/data/motions/x2_ultra_relaxed_walk_postfix.pkl \\
        --frames 60 --zero-xy \\
        --out gear_sonic/data/motions/x2_ultra_stitched_idle_relaxed_arms.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

LOWER_END = 15  # exclusive: indices [0, 15) = legs+waist
TOTAL_DOFS = 31


def _load_motion(path: Path, key: str | None) -> tuple[str, dict]:
    data = joblib.load(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: not a non-empty dict-of-motions")
    name = key or next(iter(data))
    if name not in data:
        raise KeyError(
            f"{path}: motion key {name!r} not found "
            f"(available: {list(data)[:5]}...)"
        )
    return name, data[name]


def _arm_summary(dof_slice: np.ndarray) -> str:
    means = dof_slice.mean(axis=0)
    return (
        f"L_sh[p={means[15]:+.2f} r={means[16]:+.2f}] L_el={means[18]:+.2f}  "
        f"R_sh[p={means[22]:+.2f} r={means[23]:+.2f}] R_el={means[25]:+.2f}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lower-source", required=True, type=Path,
                   help="PKL providing legs+waist DOFs (and root pose).")
    p.add_argument("--lower-key", default=None,
                   help="Motion key in --lower-source (default: first).")
    p.add_argument("--lower-start", type=int, default=0,
                   help="First frame to take from --lower-source.")
    p.add_argument("--upper-source", required=True, type=Path,
                   help="PKL providing arms+head DOFs.")
    p.add_argument("--upper-key", default=None,
                   help="Motion key in --upper-source (default: first).")
    p.add_argument("--upper-start", type=int, default=0,
                   help="First frame to take from --upper-source.")
    p.add_argument("--frames", type=int, default=60,
                   help="Number of stitched output frames (default 60).")
    p.add_argument("--zero-xy", action="store_true",
                   help="Subtract first-frame world XY from root_trans_offset "
                        "so the looped clip doesn't drift.")
    p.add_argument("--out", required=True, type=Path,
                   help="Output PKL path.")
    p.add_argument("--out-key", default=None,
                   help="Output motion key (default <lower>_+_<upper>).")
    args = p.parse_args(argv)

    try:
        lkey, lower = _load_motion(args.lower_source, args.lower_key)
        ukey, upper = _load_motion(args.upper_source, args.upper_key)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not np.isclose(float(lower["fps"]), float(upper["fps"])):
        print(
            f"WARNING: fps mismatch ({lower['fps']} vs {upper['fps']}); "
            f"sampling at integer frame indices anyway.",
            file=sys.stderr,
        )
    fps = float(lower["fps"])

    l_dof = np.asarray(lower["dof"])
    u_dof = np.asarray(upper["dof"])
    l_rot = np.asarray(lower["root_rot"])
    l_pos = np.asarray(lower["root_trans_offset"])

    if l_dof.shape[1] != TOTAL_DOFS or u_dof.shape[1] != TOTAL_DOFS:
        print(f"ERROR: expected dof[*, {TOTAL_DOFS}], got "
              f"lower={l_dof.shape}, upper={u_dof.shape}", file=sys.stderr)
        return 1

    ls, us = max(0, args.lower_start), max(0, args.upper_start)
    n = args.frames
    le, ue = ls + n, us + n
    if le > l_dof.shape[0]:
        print(f"ERROR: lower source has {l_dof.shape[0]} frames, "
              f"can't take [{ls}:{le}]", file=sys.stderr)
        return 1
    if ue > u_dof.shape[0]:
        print(f"ERROR: upper source has {u_dof.shape[0]} frames, "
              f"can't take [{us}:{ue}]", file=sys.stderr)
        return 1

    new_dof = np.zeros((n, TOTAL_DOFS), dtype=np.float64)
    new_dof[:, :LOWER_END] = l_dof[ls:le, :LOWER_END]
    new_dof[:, LOWER_END:] = u_dof[us:ue, LOWER_END:]

    new_pos = l_pos[ls:le].copy()
    new_rot = l_rot[ls:le].copy()
    if args.zero_xy:
        xy0 = new_pos[0, :2].copy()
        new_pos[:, :2] -= xy0
        print(f"[stitch] zeroed world XY (was [{xy0[0]:+.3f},{xy0[1]:+.3f}])")

    new_motion = dict(lower)
    new_motion["dof"] = new_dof
    new_motion["root_rot"] = new_rot
    new_motion["root_trans_offset"] = new_pos
    new_motion["fps"] = fps

    out_key = args.out_key or f"stitched__lower={lkey[:30]}__upper={ukey[:30]}"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({out_key: new_motion}, args.out)

    print(f"[stitch] wrote {args.out}")
    print(f"         out key            = {out_key!r}")
    print(f"         lower from         = {lkey!r}  frames [{ls}:{le}]")
    print(f"         upper from         = {ukey!r}  frames [{us}:{ue}]")
    print(f"         frames             = {n}  ({n/fps:.2f}s @ {fps:g} fps)")
    print(f"         pelvis_z mean      = {float(new_pos[:,2].mean()):+.3f} m")
    print(f"         lower DOF std mean = {float(l_dof[ls:le,:LOWER_END].std(axis=0).mean()):.4f} rad")
    print(f"         upper DOF std mean = {float(u_dof[us:ue,LOWER_END:].std(axis=0).mean()):.4f} rad")
    print(f"         lower arm pose     = {_arm_summary(l_dof[ls:le])}")
    print(f"         upper arm pose     = {_arm_summary(u_dof[us:ue])}")
    print(f"         stitched arm pose  = {_arm_summary(new_dof)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
