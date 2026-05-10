"""Per-episode commanded-vs-executed (pre-SONIC vs post-SONIC) diagnostic.

Loads a recorded LeRobot v2.1 episode parquet and computes the
arm-only ``|executed - commanded|`` joint delta over time, then prints
a per-joint summary table and writes a 4-panel matplotlib PNG of the
delta time-series under ``<dataset>/debug/sonic_correction_ep<N>.png``.

This is the offline counterpart to the recorder's live
``--sonic-correction-warn-rad`` print: same signal, full-episode
aggregation, plus a per-arm time-series so you can tell whether SONIC
is correcting one persistent joint (calibration / IK issue) or
transient operator overreach.

Schema dispatch
---------------

* **v1 datasets** (``meta/dataset_format_version.json`` reports
  ``post_sonic_canonical=True``): canonical body action lives in
  ``action.body_q_mj`` (post-SONIC executed q), pre-SONIC operator
  command lives in ``action.body_q_mj_pre_sonic``. Both columns are
  in MuJoCo joint order; we compare them directly.
* **v1 kinematic datasets** (``post_sonic_canonical=False``): no
  SONIC in the loop, so there is nothing to compare. The script
  prints a notice and exits 0.
* **v0 datasets** (no version file): only the operator-commanded
  body q is available (``action.commanded_body_q_mj``); the script
  prints summary statistics on the commanded trajectory itself
  (no delta) and writes a single-panel time-series plot.

Example::

    python -m gear_sonic.scripts.inspect_sonic_correction \\
        --dataset x2_quest3_sonic_v1 --episode 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# Index ranges into the 31-DOF MuJoCo body_q vector (legs/waist/arms/head).
# Mirrors gear_sonic/utils/teleop/x2_dataset_recorder.py.
_LEFT_ARM_MJ_SLICE = slice(15, 22)
_RIGHT_ARM_MJ_SLICE = slice(22, 29)
_ARM_JOINT_NAMES = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_yaw", "left_wrist_pitch", "left_wrist_roll",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_yaw", "right_wrist_pitch", "right_wrist_roll",
)


def _resolve_dataset_path(name_or_path: str) -> Path:
    direct = Path(name_or_path).expanduser()
    if direct.is_dir():
        return direct.resolve()
    candidate = (REPO_ROOT / "data" / "lerobot" / name_or_path).resolve()
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Dataset {name_or_path!r} not found. Tried:\n  - {direct}\n  - {candidate}"
    )


def _read_format_version(dataset_root: Path) -> dict:
    f = dataset_root / "meta" / "dataset_format_version.json"
    if not f.is_file():
        return {"version": 0, "post_sonic_canonical": None}
    return json.loads(f.read_text())


def _episode_parquet_path(dataset_root: Path, episode: int) -> Path:
    chunk = episode // 1000
    return (
        dataset_root
        / "data"
        / f"chunk-{chunk:03d}"
        / f"episode_{episode:06d}.parquet"
    )


def _stack_col(table, col: str) -> np.ndarray:
    return np.stack(table[col].to_numpy()).astype(np.float64)


def _print_arm_summary(label: str, delta_arms: np.ndarray) -> None:
    """Print mean / p99 / max per arm joint over the episode."""
    print(f"\n{label} (per-joint delta in radians)")
    print(f"  {'joint':<25} {'mean':>8} {'p99':>8} {'max':>8}")
    for j, name in enumerate(_ARM_JOINT_NAMES):
        col = np.abs(delta_arms[:, j])
        print(
            f"  {name:<25} "
            f"{col.mean():>8.4f} "
            f"{np.percentile(col, 99):>8.4f} "
            f"{col.max():>8.4f}"
        )


def _save_correction_plot(
    out_path: Path,
    *,
    delta_arms: np.ndarray,
    rate_hz: float,
    title_prefix: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(delta_arms.shape[0]) / float(rate_hz)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    axes = axes.flatten()
    panels = (
        ("Left shoulder + elbow", slice(0, 4), _ARM_JOINT_NAMES[0:4]),
        ("Left wrist", slice(4, 7), _ARM_JOINT_NAMES[4:7]),
        ("Right shoulder + elbow", slice(7, 11), _ARM_JOINT_NAMES[7:11]),
        ("Right wrist", slice(11, 14), _ARM_JOINT_NAMES[11:14]),
    )
    for ax, (label, sl, names) in zip(axes, panels):
        for j, name in zip(range(sl.start, sl.stop), names):
            ax.plot(t, np.abs(delta_arms[:, j]), lw=0.9, label=name)
        ax.set_title(label)
        ax.set_ylabel("|delta_q| (rad)")
        ax.set_xlabel("time (s)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(title_prefix)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"\nSaved time-series plot: {out_path}")


def _flag_infeasible_frames(
    delta_arms: np.ndarray, *, threshold_rad: float
) -> None:
    per_frame = np.abs(delta_arms).max(axis=1)
    mask = per_frame > threshold_rad
    if not mask.any():
        print(f"\nNo frames exceeded |delta_q| > {threshold_rad:.3f} rad.")
        return
    runs: list[tuple[int, int, float]] = []
    start = None
    peak = 0.0
    for i, hot in enumerate(mask):
        if hot and start is None:
            start = i
            peak = per_frame[i]
        elif hot and start is not None:
            peak = max(peak, per_frame[i])
        elif not hot and start is not None:
            runs.append((start, i - 1, peak))
            start = None
            peak = 0.0
    if start is not None:
        runs.append((start, len(mask) - 1, peak))
    print(
        f"\nFrames with |delta_q| > {threshold_rad:.3f} rad "
        f"({mask.sum()} / {len(mask)} = {100 * mask.mean():.1f}%):"
    )
    for first, last, run_peak in runs[:20]:
        print(
            f"  frames {first:5d}..{last:<5d}  "
            f"peak={run_peak:.3f} rad ({np.rad2deg(run_peak):.1f}°)"
        )
    if len(runs) > 20:
        print(f"  ...and {len(runs) - 20} more runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, help="Name or path under data/lerobot/")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--rate", type=float, default=50.0,
        help="Recording rate (Hz) used for the time axis. Default 50.",
    )
    parser.add_argument(
        "--infeasible-threshold-rad", type=float, default=0.15,
        help="Flag frames where |delta_q|max exceeds this. Default 0.15 rad (~8.6°).",
    )
    parser.add_argument(
        "--output-png", type=Path, default=None,
        help="Override the output PNG path. Defaults to "
             "<dataset>/debug/sonic_correction_ep<N>.png.",
    )
    args = parser.parse_args(argv)

    dataset_root = _resolve_dataset_path(args.dataset)
    parquet_path = _episode_parquet_path(dataset_root, args.episode)
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    version_meta = _read_format_version(dataset_root)
    print(f"[inspect-sonic-correction] dataset = {dataset_root}")
    print(f"[inspect-sonic-correction] episode = {args.episode}")
    print(
        "[inspect-sonic-correction] format = "
        f"v{version_meta.get('version', 0)}, "
        f"post_sonic_canonical={version_meta.get('post_sonic_canonical')}"
    )

    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    cols = set(table.column_names)

    out_png = args.output_png or (
        dataset_root / "debug" / f"sonic_correction_ep{args.episode:06d}.png"
    )

    has_v1 = (
        "action.body_q_mj" in cols and "action.body_q_mj_pre_sonic" in cols
    )
    if has_v1:
        executed = _stack_col(table, "action.body_q_mj")
        commanded = _stack_col(table, "action.body_q_mj_pre_sonic")
        if executed.shape != commanded.shape:
            raise ValueError(
                f"action.body_q_mj shape {executed.shape} != "
                f"action.body_q_mj_pre_sonic shape {commanded.shape}"
            )
        delta = executed - commanded
        delta_arms = np.concatenate(
            [delta[:, _LEFT_ARM_MJ_SLICE], delta[:, _RIGHT_ARM_MJ_SLICE]],
            axis=1,
        )
        print(
            f"[inspect-sonic-correction] {delta.shape[0]} frames; "
            f"computing post-SONIC vs pre-SONIC arm delta…"
        )
        _print_arm_summary(
            "SONIC corrective delta (executed - commanded)", delta_arms
        )
        _flag_infeasible_frames(
            delta_arms, threshold_rad=args.infeasible_threshold_rad
        )
        _save_correction_plot(
            out_png,
            delta_arms=delta_arms,
            rate_hz=args.rate,
            title_prefix=(
                f"SONIC corrective |delta_q| -- {dataset_root.name} "
                f"ep {args.episode}"
            ),
        )
        return 0

    # No SONIC delta available. Decide between v0 (commanded only) and
    # v1 kinematic (no SONIC in the loop, no delta to compute).
    if not version_meta.get("post_sonic_canonical", True):
        print(
            "[inspect-sonic-correction] dataset is kinematic-only "
            "(no SONIC in the loop). Nothing to compare."
        )
        return 0

    # v0 fallback: just stat the commanded trajectory itself.
    legacy_col = "action.commanded_body_q_mj"
    if legacy_col not in cols:
        raise ValueError(
            f"Parquet has neither v1 columns (action.body_q_mj + "
            f"action.body_q_mj_pre_sonic) nor v0 column ({legacy_col}). "
            f"Available: {sorted(cols)}"
        )
    commanded = _stack_col(table, legacy_col)
    print(
        f"[inspect-sonic-correction] v0 dataset ({legacy_col} only); "
        f"reporting per-joint trajectory range (no delta available)."
    )
    arms = np.concatenate(
        [commanded[:, _LEFT_ARM_MJ_SLICE], commanded[:, _RIGHT_ARM_MJ_SLICE]],
        axis=1,
    )
    arms_centered = arms - arms.mean(axis=0, keepdims=True)
    _print_arm_summary("Commanded arm trajectory (centered)", arms_centered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
