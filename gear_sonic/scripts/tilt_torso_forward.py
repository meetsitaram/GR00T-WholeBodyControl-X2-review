#!/usr/bin/env python3
"""Lean a clip's torso forward by biasing waist_pitch, with ramped edges.

WHY: the recorded gestures track in sim but the real robot drifts BACKWARD --
the arms come up, the CoM moves aft, and there is nothing in an upper-body-only
clip to compensate. A small forward torso bias shifts the CoM ahead of the
ankles and buys margin, without touching the gesture itself.

waist_pitch (dof 13, limit +-0.314 rad) is the right joint: POSITIVE is
forward, measured -- at +0.30 rad the head moves +0.117 m along the facing
direction and 16 bodies downstream follow. Legs and arms are untouched, so the
gesture reads identically; only the trunk carriage changes.

The bias is ramped in and out with a half-cosine over --ramp-s. A step change
in a reference is exactly what destabilises SONIC, so the edges matter as much
as the magnitude: the clip must start and end at the pose the idle stream is
already holding.

    python gear_sonic/scripts/tilt_torso_forward.py \
        --in  gear_sonic/data/motions/x2_recorded/mc_gestures/right_wave_001.pkl \
        --out out/frame_eval/tilted/right_wave_001.pkl --deg 6
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

WAIST_PITCH_DOF = 13          # verified against the MJCF, +ve = forward
WAIST_PITCH_LIMIT = 0.314     # rad, from jnt_range


def half_cos_ramp(n: int, ramp: int) -> np.ndarray:
    """1.0 in the middle, 0 at both ends, half-cosine edges."""
    w = np.ones(n)
    ramp = max(1, min(ramp, n // 2))
    e = 0.5 * (1 - np.cos(np.linspace(0, np.pi, ramp)))
    w[:ramp] = e
    w[-ramp:] = e[::-1]
    return w


def tilt(clip: dict, deg: float, ramp_s: float) -> tuple[dict, dict]:
    dof = np.asarray(clip["dof"], dtype=np.float64).copy()
    fps = float(clip.get("fps", 30.0))
    n = dof.shape[0]
    bias = np.deg2rad(deg)
    w = half_cos_ramp(n, int(round(ramp_s * fps)))

    before = dof[:, WAIST_PITCH_DOF].copy()
    after = before + bias * w
    # Respect the joint limit rather than clipping silently at play time.
    clipped = np.clip(after, -WAIST_PITCH_LIMIT, WAIST_PITCH_LIMIT)
    n_clipped = int((np.abs(after - clipped) > 1e-9).sum())
    dof[:, WAIST_PITCH_DOF] = clipped

    out = dict(clip)
    out["dof"] = dof.astype(np.float32)
    stats = {
        "frames": n, "fps": fps,
        "waist_before_mean": float(before.mean()),
        "waist_after_mean": float(clipped.mean()),
        "waist_after_max": float(clipped.max()),
        "n_clipped": n_clipped,
        # the number that matters: a step here is what breaks SONIC
        "jump_before": float(np.abs(np.diff(before)).max()),
        "jump_after": float(np.abs(np.diff(clipped)).max()),
        "clip_jump_all_dofs": float(np.abs(np.diff(dof, axis=0)).max()),
    }
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--key", default=None)
    ap.add_argument("--deg", type=float, default=6.0,
                    help="forward lean in degrees (+ve = forward)")
    ap.add_argument("--ramp-s", type=float, default=0.7,
                    help="half-cosine ramp in/out, seconds")
    args = ap.parse_args()

    lib = joblib.load(args.src)
    key = args.key or next(iter(lib))
    new, st = tilt(lib[key], args.deg, args.ramp_s)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({key: new}, args.out)

    print(f"  {key}")
    print(f"    {st['frames']} frames @ {st['fps']:g} fps   lean {args.deg:+.1f} deg "
          f"({np.deg2rad(args.deg):+.3f} rad), ramp {args.ramp_s:g}s")
    print(f"    waist_pitch mean {st['waist_before_mean']:+.3f} -> "
          f"{st['waist_after_mean']:+.3f} rad   peak {st['waist_after_max']:+.3f}"
          f"   (limit {WAIST_PITCH_LIMIT:+.3f})")
    if st["n_clipped"]:
        print(f"    WARNING: {st['n_clipped']} frames hit the joint limit")
    print(f"    per-frame waist step {st['jump_before']:.4f} -> {st['jump_after']:.4f} rad")
    print(f"    max per-frame step, ALL dofs: {st['clip_jump_all_dofs']:.3f} rad")
    print(f"    wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
