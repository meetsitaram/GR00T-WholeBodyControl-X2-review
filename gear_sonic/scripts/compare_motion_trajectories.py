"""
Reconstruction-error metric for the M3 autoencoder smoke test.

The "autoencoder smoke test" feeds a known motion (the Minecraft piano
recording overlaid on a bones-seed standing motion) through the entire
collect → fine-tune → deploy → sim chain and measures how well the
closed-loop sim rollout reproduces the original. If the reconstruction
error stays under threshold, every link in the pipeline is byte-clean
*before* a single full-scale training step has run.

This module is the **metric helper** used by the M3 acceptance gate.
It is intentionally simulator-agnostic: callers hand it two ``(T, D)``
trajectories (recorded vs. rolled-out, in the same DOF order) plus an
optional per-DOF threshold, and it returns a structured
:class:`ComparisonResult`. The orchestrator
(``record_synthetic_smoketest_dataset.py``) and any downstream rollout
script can drive the same comparison without round-tripping data
through CSVs or pickles.

Time alignment
--------------

The helper supports three alignment modes:

* ``"exact"`` -- both trajectories must already share the same shape.
  Fails fast if they don't. Use this when the comparison is between
  the recorded trajectory and the dataset round-trip (the M3 gate uses
  this mode -- the round-trip should be byte-identical modulo float64
  precision).
* ``"resample"`` -- linearly resample the rollout onto the recorded
  trajectory's time grid. Use when the rollout runs at a different
  control rate than the recording (e.g. 50 Hz dataset vs. 200 Hz
  underlying sim).
* ``"shortest"`` -- truncate both to the shorter length. Use as a
  fallback when the rollout exits early (safety stack tripped, etc.)
  -- the helper still returns a metric over the surviving frames.

CLI usage
---------

::

    .venv/bin/python gear_sonic/scripts/compare_motion_trajectories.py \\
        --recorded /tmp/x2_smoketest/episode_0_recorded.npz \\
        --rollout /tmp/x2_smoketest/episode_0_rollout.npz \\
        --threshold 0.05 \\
        --json-out /tmp/x2_smoketest/compare_summary.json

The CLI loads ``.npz`` files containing a single array each (key=
``trajectory`` by default; override via ``--recorded-key``/``--rollout-key``)
and prints the summary plus exits non-zero if the metric exceeds the
threshold. The pytest gate (``tests/test_x2_smoketest_pipeline.py``)
calls :func:`compare_trajectories` directly so it does not need to
shell out.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


AlignmentMode = Literal["exact", "resample", "shortest"]


@dataclass
class ComparisonResult:
    """Structured output of :func:`compare_trajectories`."""

    aligned_length: int
    """Frame count after alignment."""

    rmse_overall: float
    """Root-mean-squared error pooled over all frames and DOFs (rad)."""

    max_abs_error_overall: float
    """Worst single-frame, single-DOF |delta| (rad)."""

    rmse_per_dof: list[float]
    """Per-DOF RMSE (length = D)."""

    max_abs_error_per_dof: list[float]
    """Per-DOF worst |delta| (length = D)."""

    threshold_rad: float
    """Per-DOF threshold the comparison was checked against."""

    pass_: bool = field(metadata={"json_key": "pass"})
    """True iff every per-DOF RMSE is below ``threshold_rad``."""

    failing_dof_indices: list[int]
    """DOF indices whose RMSE exceeded the threshold (empty when ``pass_``)."""

    alignment_mode: AlignmentMode
    """Which alignment branch was taken."""

    notes: list[str]
    """Human-readable notes (length-mismatch warnings, etc.)."""

    def to_json_dict(self) -> dict:
        """Serializer-friendly dict (renames ``pass_`` -> ``pass``)."""
        d = asdict(self)
        d["pass"] = d.pop("pass_")
        return d


def _align(
    recorded: np.ndarray, rollout: np.ndarray, mode: AlignmentMode
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Bring both trajectories onto a common time grid per ``mode``."""
    notes: list[str] = []
    if mode == "exact":
        if recorded.shape != rollout.shape:
            raise ValueError(
                f"alignment_mode='exact' requires identical shapes; "
                f"recorded={recorded.shape}, rollout={rollout.shape}"
            )
        return recorded, rollout, notes
    if mode == "shortest":
        T = min(recorded.shape[0], rollout.shape[0])
        if recorded.shape[0] != rollout.shape[0]:
            notes.append(
                f"truncated to shortest length: recorded={recorded.shape[0]}, "
                f"rollout={rollout.shape[0]} -> {T}"
            )
        return recorded[:T], rollout[:T], notes
    if mode == "resample":
        if recorded.shape[1] != rollout.shape[1]:
            raise ValueError(
                f"alignment_mode='resample' requires the same DOF count; "
                f"recorded D={recorded.shape[1]}, rollout D={rollout.shape[1]}"
            )
        T_ref = recorded.shape[0]
        T_in = rollout.shape[0]
        if T_in == T_ref:
            return recorded, rollout, notes
        notes.append(f"linearly resampled rollout: {T_in} -> {T_ref} frames")
        src_x = np.linspace(0.0, 1.0, T_in)
        dst_x = np.linspace(0.0, 1.0, T_ref)
        out = np.empty_like(recorded)
        for d in range(recorded.shape[1]):
            out[:, d] = np.interp(dst_x, src_x, rollout[:, d])
        return recorded, out, notes
    raise ValueError(f"unknown alignment_mode {mode!r}")


def compare_trajectories(
    recorded: np.ndarray,
    rollout: np.ndarray,
    *,
    threshold_rad: float = 0.05,
    alignment_mode: AlignmentMode = "exact",
) -> ComparisonResult:
    """Compute the L2 reconstruction error between two motion streams.

    Args:
        recorded: ``(T, D)`` float array of the *reference* trajectory
            (the recording, or the trajectory written to the LeRobot
            dataset before training).
        rollout: ``(T, D)`` float array of the *candidate* trajectory
            (the closed-loop sim rollout from the trained model). Will
            be aligned to ``recorded`` per ``alignment_mode``.
        threshold_rad: Per-DOF RMSE threshold (radians). Each DOF's
            RMSE must stay below this value for the result to pass.
            ``0.05`` (~2.9 deg) is the M3 gate's default; tighten for
            production deployment.
        alignment_mode: How to reconcile differing trajectory lengths.
            ``"exact"`` is the strictest; ``"resample"`` is the most
            forgiving; ``"shortest"`` is a CI-safe fallback.

    Returns:
        A populated :class:`ComparisonResult`.

    Raises:
        ValueError: shape / dtype / mode disagreements.
    """
    rec = np.asarray(recorded, dtype=np.float64)
    roll = np.asarray(rollout, dtype=np.float64)
    if rec.ndim != 2 or roll.ndim != 2:
        raise ValueError(
            f"compare_trajectories expects 2-D arrays; got rec.ndim={rec.ndim}, "
            f"roll.ndim={roll.ndim}"
        )
    if rec.shape[1] != roll.shape[1]:
        raise ValueError(
            f"DOF counts disagree: recorded D={rec.shape[1]}, rollout D={roll.shape[1]}"
        )
    if threshold_rad < 0:
        raise ValueError(f"threshold_rad must be >= 0; got {threshold_rad}")

    rec_aligned, roll_aligned, notes = _align(rec, roll, alignment_mode)
    delta = roll_aligned - rec_aligned

    rmse_per_dof = np.sqrt(np.mean(delta * delta, axis=0))
    max_abs_per_dof = np.max(np.abs(delta), axis=0)
    rmse_overall = float(np.sqrt(np.mean(delta * delta)))
    max_abs_overall = float(np.max(np.abs(delta)))

    failing = np.where(rmse_per_dof > threshold_rad)[0].tolist()
    return ComparisonResult(
        aligned_length=int(rec_aligned.shape[0]),
        rmse_overall=rmse_overall,
        max_abs_error_overall=max_abs_overall,
        rmse_per_dof=[float(x) for x in rmse_per_dof],
        max_abs_error_per_dof=[float(x) for x in max_abs_per_dof],
        threshold_rad=float(threshold_rad),
        pass_=len(failing) == 0,
        failing_dof_indices=failing,
        alignment_mode=alignment_mode,
        notes=notes,
    )


def format_summary(
    result: ComparisonResult,
    *,
    dof_names: list[str] | None = None,
    top_k_offenders: int = 5,
) -> str:
    """Pretty-print a :class:`ComparisonResult` for the operator.

    Returns a multi-line string. Uses ``dof_names`` to label per-DOF
    rows when provided; otherwise falls back to ``dof_<i>``.
    """
    lines = []
    verdict = "PASS" if result.pass_ else "FAIL"
    lines.append(f"M3 reconstruction comparison: {verdict}")
    lines.append(f"  alignment       : {result.alignment_mode}")
    lines.append(f"  aligned frames  : {result.aligned_length}")
    lines.append(f"  threshold       : {result.threshold_rad:.5f} rad/DOF (RMSE)")
    lines.append(f"  RMSE overall    : {result.rmse_overall:.5f} rad")
    lines.append(f"  max |delta|     : {result.max_abs_error_overall:.5f} rad")
    if result.notes:
        lines.append("  notes:")
        for n in result.notes:
            lines.append(f"    - {n}")
    # Top-k worst DOFs, ranked by RMSE.
    rmse = np.asarray(result.rmse_per_dof)
    k = min(top_k_offenders, len(rmse))
    if k > 0:
        order = np.argsort(rmse)[::-1][:k]
        lines.append(f"  top {k} DOFs by RMSE:")
        for idx in order:
            label = dof_names[idx] if dof_names else f"dof_{idx}"
            lines.append(
                f"    {label:>32s}: rmse={rmse[idx]:.5f} max={result.max_abs_error_per_dof[idx]:.5f}"
            )
    if not result.pass_:
        lines.append(
            f"  failing DOFs ({len(result.failing_dof_indices)}): "
            f"{result.failing_dof_indices}"
        )
    return "\n".join(lines)


def _load_npz_trajectory(path: Path, key: str) -> np.ndarray:
    """Load a single ``(T, D)`` array from an .npz file."""
    with np.load(path) as data:
        if key not in data:
            raise KeyError(
                f"key {key!r} not in {path}; available: {list(data.keys())}"
            )
        arr = np.asarray(data[key])
    if arr.ndim != 2:
        raise ValueError(f"{path}[{key!r}] must be 2-D; got shape {arr.shape}")
    return arr


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorded", type=Path, required=True,
        help="Path to the reference trajectory (.npz).",
    )
    parser.add_argument(
        "--rollout", type=Path, required=True,
        help="Path to the candidate trajectory (.npz).",
    )
    parser.add_argument("--recorded-key", default="trajectory")
    parser.add_argument("--rollout-key", default="trajectory")
    parser.add_argument(
        "--threshold", type=float, default=0.05,
        help="Per-DOF RMSE threshold (radians). Default 0.05 (~2.9 deg).",
    )
    parser.add_argument(
        "--alignment", choices=["exact", "resample", "shortest"], default="resample",
        help="Time-alignment mode. Default 'resample'.",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="If set, write the structured result to this JSON path.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Show top-k DOFs by RMSE in the printout.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rec = _load_npz_trajectory(args.recorded, args.recorded_key)
    roll = _load_npz_trajectory(args.rollout, args.rollout_key)
    result = compare_trajectories(
        rec,
        roll,
        threshold_rad=args.threshold,
        alignment_mode=args.alignment,
    )
    print(format_summary(result, top_k_offenders=args.top_k))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result.to_json_dict(), indent=2))
        print(f"  wrote summary -> {args.json_out}")
    return 0 if result.pass_ else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AlignmentMode",
    "ComparisonResult",
    "compare_trajectories",
    "format_summary",
]
