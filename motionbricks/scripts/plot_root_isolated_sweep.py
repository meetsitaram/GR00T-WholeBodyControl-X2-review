"""Matplotlib viewer for one or more root-isolation sweep reports.

Reads ``--report-json`` files produced by ``test_root_isolated.py`` and
plots forward displacement vs commanded forward velocity, with the
ideal slope-1 line overlaid for parity.

Usage::

    python motionbricks/scripts/plot_root_isolated_sweep.py \\
        --reports out/per_model_report/root_x2_walking_cold.json \\
                  out/per_model_report/root_x2_walking_warm.json \\
                  out/per_model_report/root_x2_stationary_cold.json \\
        --output  out/per_model_report/root_x2_sweep.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNG, no display
import matplotlib.pyplot as plt
import numpy as np


def _load_report(p: Path) -> dict:
    with open(p) as f:
        return json.load(f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_root, ax_e2e = axes

    for p in args.reports:
        r = _load_report(p)
        label = f"{r['ckpt_set']}/{r['fixture']}/{r['mode']}"
        rows = r["sweep"]
        vx = np.array([row["intent"]["vel_x"] for row in rows])
        fwd_root = np.array([row["root_only"]["pred_forward_m"] for row in rows])
        fwd_e2e = np.array(
            [row["e2e"].get("achieved_forward_m", np.nan) for row in rows]
        )
        # horizon_s for the slope reference line: pull from the largest
        # vx>0 row so we have a stable horizon estimate.
        horizons = [row["root_only"]["horizon_s"] for row in rows]
        horizon_s = float(np.mean(horizons))

        ax_root.plot(vx, fwd_root, marker="o", label=label)
        ax_e2e.plot(vx, fwd_e2e, marker="o", label=label)

    # Overlay slope-1 ideal (forward = vx * horizon_s, using mean horizon).
    vx_grid = np.linspace(0.0, 0.8, 100)
    horizon_s_ref = 2.13  # 64 frames @ 30 fps; cosmetic reference
    ax_root.plot(
        vx_grid, vx_grid * horizon_s_ref,
        linestyle="--", color="grey", label=f"ideal (slope=1 at {horizon_s_ref:.2f}s)"
    )
    ax_e2e.plot(
        vx_grid, vx_grid * horizon_s_ref,
        linestyle="--", color="grey", label=f"ideal (slope=1 at {horizon_s_ref:.2f}s)"
    )

    ax_root.set_xlabel("commanded vel_x (m/s)")
    ax_root.set_ylabel("pred_forward_m  (body frame)")
    ax_root.set_title("Root model isolation  (pred_global_root only)")
    ax_root.grid(True, alpha=0.3)
    ax_root.legend(fontsize=8)

    ax_e2e.set_xlabel("commanded vel_x (m/s)")
    ax_e2e.set_ylabel("achieved_forward_m  (body frame)")
    ax_e2e.set_title("End-to-end planner output  (integrated qpos)")
    ax_e2e.grid(True, alpha=0.3)
    ax_e2e.legend(fontsize=8)

    fig.suptitle(
        "Forward displacement vs commanded velocity  —  ideal slope = 1",
        fontsize=12,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=120)
    print(f"[plot] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
