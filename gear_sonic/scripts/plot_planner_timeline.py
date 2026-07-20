#!/usr/bin/env python3
"""Plot what the planner does AS TIME PASSES, with replan-seam statistics.

WHY
---
Every aggregate we computed tonight hid the thing that mattered. Foot slide
averaged over a clip looked like a clean constant until the contact count
showed it rested on 5 frames. Per-frame series fixed the measurement; this
makes the series visible, which is the only way to see structure that has a
TIME shape -- drift that accumulates, seams that recur on a period, contact
that degrades as a run goes on.

PANELS (shared time axis)
  1 pelvis z          -- vertical drift. Planner output climbed 3.5 cm over
                         13 s where the reference clip held flat.
  2 foot height       -- both feet, with contact shaded. The reference plants
                         641 frames; planner output planted 5.
  3 per-frame joint step -- replan seams appear here as spikes. Detected seams
                         are marked, so you can see whether drift steps line
                         up with seams (which would implicate the 8-frame
                         cross-fade) or accumulate smoothly between them.
  4 root speed + foot slide -- commanded travel vs skating, over time.

SEAM STATS
Seams are found as outliers in the per-frame joint step (> median + k*MAD,
robust to the gait's own rhythm). Reported: count, mean interval vs the
expected replan period, magnitude, and how much of each seam falls inside the
8-frame blend window.

    python gear_sonic/scripts/plot_planner_timeline.py \
        --npz out/frame_eval/intent_probe/*.npz \
        --ref gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl \
        --out out/frame_eval/timeline.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from gear_sonic.scripts.kplanner_frame_eval import (  # noqa: E402
    _mjcf_path, frame_series, load_qpos)

BLEND_FRAMES = 8          # pc2_kplanner_onnx.py:729
MODEL_FPS = 30.0


def seam_stats(jump: np.ndarray, fps: float, k: float = 6.0) -> dict:
    """Replan seams as robust outliers in the per-frame joint step.

    Median+MAD rather than a fixed threshold: a walking gait has its own
    rhythmic step magnitude, and a fixed cut would either drown in it or miss
    seams entirely on a slow clip.
    """
    med = float(np.median(jump))
    mad = float(np.median(np.abs(jump - med))) or 1e-9
    thr = med + k * 1.4826 * mad
    idx = np.where(jump > thr)[0]
    # collapse runs of adjacent frames into one seam
    seams = [int(i) for j, i in enumerate(idx) if j == 0 or i - idx[j - 1] > 3]
    iv = np.diff(seams) / fps if len(seams) > 1 else np.array([])
    return {
        "threshold": thr, "median_step": med, "n_seams": len(seams),
        "seams": seams,
        "interval_mean_s": float(iv.mean()) if iv.size else float("nan"),
        "interval_sd_s": float(iv.std()) if iv.size else float("nan"),
        "mag_mean": float(jump[seams].mean()) if seams else 0.0,
        "mag_max": float(jump[seams].max()) if seams else 0.0,
    }


def panel(ax, t, s, q, label, seams):
    ax.set_title(label, fontsize=9, loc="left")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", nargs="+", type=Path, required=True)
    ap.add_argument("--ref", type=Path, default=None,
                    help="reference clip (PKL) plotted first for comparison")
    ap.add_argument("--ref-key", default=None)
    ap.add_argument("--out", type=Path,
                    default=REPO / "out/frame_eval/timeline.png")
    ap.add_argument("--mjcf", default=None)
    args = ap.parse_args()

    mjcf = _mjcf_path(args.mjcf)
    sources = []
    if args.ref:
        q, fps = load_qpos(args.ref, args.ref_key)
        sources.append(("REFERENCE (real clip)", q, fps))
    for p in args.npz:
        q, fps = load_qpos(p, None)
        sources.append((p.stem.split("__")[-1], q, fps))

    n = len(sources)
    fig, axes = plt.subplots(4, n, figsize=(5.2 * n, 11), squeeze=False,
                             sharex="col")
    print(f"\n{'source':<22}{'seams':>7}{'interval':>10}{'expected':>10}"
          f"{'mag':>8}{'contact%':>10}{'z drift':>9}")
    print("-" * 76)

    for c, (label, q, fps) in enumerate(sources):
        s = frame_series(q, mjcf, fps)
        T = s["T"]
        t = np.arange(T) / fps
        st = seam_stats(s["jump"], fps)

        # 1 pelvis z
        ax = axes[0][c]
        ax.plot(t, q[:, 2], lw=1.0, color="tab:blue")
        drift = q[-1, 2] - q[0, 2]
        ax.set_title(f"{label}\npelvis z  (drift {drift:+.3f} m)",
                     fontsize=9, loc="left")
        ax.set_ylabel("m")
        ax.grid(alpha=.3)

        # 2 foot height + contact shading
        ax = axes[1][c]
        for f, col in ((0, "tab:green"), (1, "tab:orange")):
            ax.plot(t, s["foot_z"][:, f], lw=.8, color=col,
                    label=f"foot {'L' if f == 0 else 'R'}")
            ax.fill_between(t, 0, s["foot_z"][:, f].max(),
                            where=s["contact"][:, f], color=col, alpha=.10,
                            step="mid", lw=0)
        cpct = 100.0 * s["contact"].any(axis=1).mean()
        ax.set_title(f"foot height + contact  ({cpct:.1f}% frames w/ a "
                     f"planted foot)", fontsize=9, loc="left")
        ax.set_ylabel("m (registered)")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.3)

        # 3 per-frame joint step + seams
        ax = axes[2][c]
        ax.plot(t, s["jump"], lw=.7, color="tab:red")
        ax.axhline(st["threshold"], ls="--", lw=.7, color="k",
                   label=f"seam thr {st['threshold']:.3f}")
        for i in st["seams"]:
            ax.axvspan(i / fps, (i + BLEND_FRAMES) / fps, color="tab:red",
                       alpha=.12, lw=0)
        ax.set_title(f"per-frame joint step; {st['n_seams']} seams, "
                     f"{BLEND_FRAMES}-frame blend shaded", fontsize=9,
                     loc="left")
        ax.set_ylabel("rad")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.3)

        # 4 root speed + foot slide
        ax = axes[3][c]
        ax.plot(t, s["root_speed"], lw=.9, color="tab:purple",
                label="root speed")
        ax.plot(t, s["slide"].max(axis=1), lw=.7, color="tab:brown",
                alpha=.8, label="foot slide (in contact)")
        ax.set_title("root speed vs foot slide", fontsize=9, loc="left")
        ax.set_ylabel("m/s"); ax.set_xlabel("time (s)")
        ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.3)

        expected = 1.0 / (MODEL_FPS / 64) if True else float("nan")
        print(f"{label:<22}{st['n_seams']:>7}"
              f"{st['interval_mean_s']:>10.2f}{expected:>10.2f}"
              f"{st['mag_mean']:>8.3f}{cpct:>10.1f}{drift:>+9.3f}")

    fig.suptitle("kplanner timeline -- drift, contact, replan seams",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=115)
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
