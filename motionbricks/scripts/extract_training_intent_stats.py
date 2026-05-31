"""Extract Flavor A + B intent statistics from the curated PKL primitive subset.

Two reference distributions land in the same output JSON; the analyzer
overlays both onto the live VR distribution to pick smoothing knobs.

Flavor A -- training-time per-frame intent
-----------------------------------------

The neural planner consumes a 4-D velocity intent ``(yaw_rate, vel_x,
vel_z, hip_h)`` computed at training time by
``NeuralPlannerCore._predict_with_velocity`` as a one-frame finite
difference of root pos / yaw, multiplied by fps. We reproduce that
formula with ``_instant_intent_from_clip(window=2)`` (the smallest
window) so each frame's intent is the rawest "this is what the
training data carries on this tick" value.

Flavor B -- live PKL-replay intent (rolling window)
---------------------------------------------------

``x2_pkl_command_source.py`` is the live PKL replay path. It uses
``_instant_intent_from_clip(window=8)`` (~0.27 s @ 30 fps) to produce
a slightly-smoothed intent that the kplanner ingests during PKL replay.
This is the "smooth source" the analyzer compares the (raw) VR stream
against, so we include both A (raw) and B (smoothed) -- they bracket
the in-distribution band.

Per-clip + aggregate output
---------------------------

The output JSON contains, for each clip:
  * ``intent_A``: per-frame raw intent ``[[yaw_rate, vel_x, vel_z, hip_h], ...]``
  * ``intent_B``: per-frame rolling-window intent (same shape)
  * ``dA_dt``  : per-frame finite difference of A (the step-input metric)
  * ``dB_dt``  : per-frame finite difference of B

And aggregate (across all clips concatenated):
  * ``A_stats`` / ``B_stats``: per-channel mean, std, p1, p5, p25, p50,
    p75, p95, p99, min, max
  * ``dA_dt_stats`` / ``dB_dt_stats``: same percentiles on |d/dt|, in
    units of (channel-unit / s). The ``p99`` of ``|d(vel_z)/dt|`` from
    Flavor A is the canonical "training-distribution acceleration
    ceiling" that the StickFilter targets.

Usage
-----

  .venv/bin/python motionbricks/scripts/extract_training_intent_stats.py \\
      --subset gear_sonic/data/motions/x2_intent_reference_subset.yaml \\
      --out    out/intent_reference/training_intent_stats.json

Reads ``--subset`` directly (one entry per clip with self-contained
``pkl`` + ``motion_key`` + ``start_frame`` + ``n_frames``), loads each
PKL, slices the window, and runs the two extractors. One file per run
is the deliverable; the analyzer reads it without re-loading PKLs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MOTIONBRICKS_SCRIPTS = _REPO_ROOT / "motionbricks" / "scripts"
if str(_MOTIONBRICKS_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_MOTIONBRICKS_SCRIPTS))

# Share the velocity-extraction helpers with the live PKL source so
# Flavors A / B stay in lockstep with what the kplanner actually
# consumes during PKL replay.
from replay_pkl_through_kplanner import (  # noqa: E402
    _build_clip_qpos,
    _instant_intent_from_clip,
)

log = logging.getLogger("extract_training_intent_stats")


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"yaml not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def _slice_clip_qpos_from_loaded(
    raw: dict,
    motion_key: str,
    start_frame: int,
    n_frames: int,
) -> tuple[np.ndarray, float]:
    """Slice ``motion_key`` out of an already-loaded PKL dict.

    Separated from disk I/O so the extractor caches one PKL load across
    all clips that share a source file (the locowalk PKL is ~1.5 GB
    and ~10 s to load -- reloading per clip would be silly).
    """
    if motion_key not in raw:
        raise KeyError(
            f"motion_key {motion_key!r} not present; "
            f"available[0:5]={list(raw.keys())[:5]}"
        )
    qpos, fps = _build_clip_qpos(raw[motion_key])
    T = qpos.shape[0]
    lo = max(0, int(start_frame))
    hi = min(T, lo + int(n_frames))
    if hi <= lo:
        raise ValueError(
            f"{motion_key}: empty slice (T={T}, start={start_frame}, n={n_frames})"
        )
    return qpos[lo:hi].copy(), float(fps)


# ---------------------------------------------------------------------------
# Per-clip extraction
# ---------------------------------------------------------------------------


def _per_frame_intent(
    qpos: np.ndarray, fps: float, window: int,
) -> np.ndarray:
    """Return ``[T, 4]`` per-frame intent computed with the given window.

    ``window=2`` reproduces the training-time finite-difference intent
    (Flavor A); ``window=8`` reproduces the live PKL-replay smoothed
    intent (Flavor B).
    """
    T = qpos.shape[0]
    out = np.zeros((T, 4), dtype=np.float64)
    for i in range(T):
        out[i, :] = _instant_intent_from_clip(qpos, fps, i, window=window)
    return out


def _per_frame_derivative(intent: np.ndarray, fps: float) -> np.ndarray:
    """Return ``[T, 4]`` per-frame finite difference of ``intent``, in
    channel-unit / s. The first row's derivative is set to 0 by
    convention (the step-input metric we care about is the BULK of the
    distribution, not the boundary).
    """
    if intent.shape[0] < 2:
        return np.zeros_like(intent)
    d = np.zeros_like(intent)
    d[1:, :] = (intent[1:, :] - intent[:-1, :]) * fps
    return d


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


_PCT_LIST = [1, 5, 25, 50, 75, 95, 99]
_CHANNEL_NAMES = ["yaw_rate", "vel_x", "vel_z", "hip_h"]


def _channel_stats(arr: np.ndarray, channel_idx: int) -> dict:
    """Return mean/std/min/max + percentile dict for one channel."""
    col = arr[:, channel_idx]
    out: dict[str, float] = {
        "count": int(col.shape[0]),
        "mean": float(np.mean(col)),
        "std": float(np.std(col)),
        "min": float(np.min(col)),
        "max": float(np.max(col)),
    }
    for p in _PCT_LIST:
        out[f"p{p:02d}"] = float(np.percentile(col, p))
    # Also report |.| percentiles -- for derivatives we care about
    # magnitude, sign is direction.
    abs_col = np.abs(col)
    for p in _PCT_LIST:
        out[f"abs_p{p:02d}"] = float(np.percentile(abs_col, p))
    out["abs_max"] = float(np.max(abs_col))
    return out


def _per_channel_stats(arr: np.ndarray) -> dict[str, dict]:
    return {
        _CHANNEL_NAMES[i]: _channel_stats(arr, i)
        for i in range(arr.shape[1])
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--subset", type=Path,
        default=_REPO_ROOT / "gear_sonic/data/motions/x2_intent_reference_subset.yaml",
        help="Path to the curated clip subset YAML. Each entry must be "
             "self-contained: (name, pkl, motion_key, start_frame, "
             "n_frames). Defaults to the 7-clip subset matching the VR "
             "maneuver script.",
    )
    p.add_argument(
        "--window-A", type=int, default=2,
        help="Frame window for the Flavor-A (raw training-intent) "
             "finite difference. Default 2 = adjacent-frame diff, "
             "matching NeuralPlannerCore._predict_with_velocity.",
    )
    p.add_argument(
        "--window-B", type=int, default=8,
        help="Frame window for the Flavor-B (live PKL-replay) rolling "
             "intent. Default 8 = what x2_pkl_command_source ships at 30 fps.",
    )
    p.add_argument(
        "--out", "-o", type=Path, required=True,
        help="Output JSON path. Parent created on demand.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if args.verbose else logging.INFO,
    )

    subset_doc = _load_yaml(args.subset)
    subset_entries = subset_doc.get("clip_subset", [])
    if not subset_entries:
        log.error("subset YAML has no clip_subset entries: %s", args.subset)
        return 2
    names = [c["name"] for c in subset_entries]
    log.info("subset: %d clips -- %s", len(names), ", ".join(names))

    # Cache one PKL load per unique source path (locowalk.pkl is ~1.5 GB).
    pkl_cache: dict[Path, dict] = {}

    clips_out: dict[str, dict[str, Any]] = {}
    all_A: list[np.ndarray] = []
    all_B: list[np.ndarray] = []
    all_dA: list[np.ndarray] = []
    all_dB: list[np.ndarray] = []

    for entry in subset_entries:
        name = entry["name"]
        pkl_path = _REPO_ROOT / entry["pkl"]
        try:
            if pkl_path not in pkl_cache:
                if not pkl_path.is_file():
                    raise FileNotFoundError(f"PKL not found: {pkl_path}")
                log.info("loading PKL %s ...", pkl_path)
                pkl_cache[pkl_path] = joblib.load(pkl_path)
                log.info("  loaded (%d keys).", len(pkl_cache[pkl_path]))
            raw = pkl_cache[pkl_path]
            qpos, fps = _slice_clip_qpos_from_loaded(
                raw,
                motion_key=entry["motion_key"],
                start_frame=int(entry["start_frame"]),
                n_frames=int(entry["n_frames"]),
            )
        except Exception as exc:  # noqa: BLE001
            log.error("failed to load %s: %s", name, exc)
            continue

        A = _per_frame_intent(qpos, fps, window=args.window_A)
        B = _per_frame_intent(qpos, fps, window=args.window_B)
        dA = _per_frame_derivative(A, fps)
        dB = _per_frame_derivative(B, fps)
        log.info(
            "%-22s T=%3d fps=%.1f  A: vel_z [%+.3f, %+.3f] m/s   "
            "|dA/dt|_p99 m/s^2: vel_z=%.2f",
            name, qpos.shape[0], fps,
            float(A[:, 2].min()), float(A[:, 2].max()),
            float(np.percentile(np.abs(dA[:, 2]), 99)),
        )

        clips_out[name] = {
            "motion_key": entry["motion_key"],
            "pkl": entry["pkl"],
            "start_frame": int(entry["start_frame"]),
            "n_frames": int(entry["n_frames"]),
            "fps": float(fps),
            "intent_A": A.tolist(),
            "intent_B": B.tolist(),
            "dA_dt":   dA.tolist(),
            "dB_dt":   dB.tolist(),
        }
        all_A.append(A)
        all_B.append(B)
        all_dA.append(dA)
        all_dB.append(dB)

    if not clips_out:
        log.error("no clips loaded; aborting")
        return 1

    A_concat = np.concatenate(all_A, axis=0)
    B_concat = np.concatenate(all_B, axis=0)
    dA_concat = np.concatenate(all_dA, axis=0)
    dB_concat = np.concatenate(all_dB, axis=0)

    aggregate = {
        "n_clips": len(clips_out),
        "n_frames_total": int(A_concat.shape[0]),
        "A_stats": _per_channel_stats(A_concat),
        "B_stats": _per_channel_stats(B_concat),
        "dA_dt_stats": _per_channel_stats(dA_concat),
        "dB_dt_stats": _per_channel_stats(dB_concat),
    }

    payload = {
        "schema_version": 2,
        "subset_yaml": str(args.subset),
        "window_A": int(args.window_A),
        "window_B": int(args.window_B),
        "channel_names": _CHANNEL_NAMES,
        "clips": clips_out,
        "aggregate": aggregate,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(payload, f, indent=2)
    log.info(
        "wrote %s (%d clips, %d frames; A.vel_z |d/dt|_p99=%.2f m/s^2)",
        args.out, len(clips_out), A_concat.shape[0],
        aggregate["dA_dt_stats"]["vel_z"]["abs_p99"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
