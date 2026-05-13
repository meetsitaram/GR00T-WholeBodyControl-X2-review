"""Data-driven tuner for the Quest-3 finger-curl compensation.

Loads one or more debug NPZ files recorded by
``teleop_x2_kinematic.py`` / ``record_x2_dataset.py`` (the files at
``data/lerobot/.../debug/teleop_episode_NNNN.npz``) and finds the
parameter values for :func:`stretch_finger_curls` that best satisfy
the operator's stated preference of "fingers should be visibly
open or visibly closed, not floating at half-bent".

The tuner can run in three modes:

* ``--mode global``: find a single ``(deadzone, full_threshold,
  gamma)`` triple that works across all 5 fingers (current default
  in :data:`x2_hand_retarget.DEFAULT_CURL_*`).

* ``--mode per-finger``: independently sweep each of the 5 fingers
  and report a per-finger ``(dz, full, gamma)``. This is useful
  for diagnostic purposes; it is not yet wired into the runtime
  retarget code (which uses a single global stretch).

* ``--mode noise-floor``: skip the parameter sweep and only
  characterise the per-finger Quest3 raw-curl noise floor during
  static-hand frames. Useful for choosing a deadzone above the
  noise floor.

Decoding raw curls from the dataset
-----------------------------------

The v3 episodes were recorded *before* the Python-side
:func:`stretch_finger_curls` was wired in, so the recorded
``commanded_*_hand_q`` is a direct linear lerp
``cmd = (1 - c) * OPEN + c * CLOSED`` from the raw Quest curl
``c`` and the inverse is exact. Episodes recorded *after* the
stretch was wired in cannot have raw curls recovered (the inverse
is non-injective near the deadzone / saturation boundaries); the
script prints a warning if it detects clipping consistent with a
stretch having been applied and exits.

Usage
-----

::

    # Global tune across all 4 v3 episodes:
    python -m gear_sonic.scripts.tune_finger_curl_compensation \\
        --mode global \\
        data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_*.npz

    # Per-finger diagnostic:
    python -m gear_sonic.scripts.tune_finger_curl_compensation \\
        --mode per-finger \\
        data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_000000.npz

    # Noise-floor only:
    python -m gear_sonic.scripts.tune_finger_curl_compensation \\
        --mode noise-floor \\
        data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_000000.npz

The script is read-only on the dataset; it does not modify the NPZ
or any other state. To apply the recommended global parameters,
edit :data:`gear_sonic.utils.teleop.x2_hand_retarget.DEFAULT_CURL_*`
(or pass the values explicitly to :func:`stretch_finger_curls`).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# Re-export the OPEN/CLOSED anchors from x2_hand_retarget so this
# tuner stays in sync whenever the anchors are tweaked. (We
# previously mirrored the deg lists locally and they drifted: the
# May 10 thumb-anchor expansion (50% -> 80% hardware travel) and
# the subsequent non-thumb pip expansion (80° -> 88°) didn't
# propagate here, which silently broke the tuner's
# decode-vs-replay parity check.)
from gear_sonic.utils.teleop.x2_hand_retarget import (  # noqa: E402
    HAND_GRASP_CLOSED_RAD_LEFT as _CLOSED_LEFT_TUPLE,
    HAND_GRASP_CLOSED_RAD_RIGHT as _CLOSED_RIGHT_TUPLE,
    HAND_GRASP_OPEN_RAD_LEFT as _OPEN_LEFT_TUPLE,
    HAND_GRASP_OPEN_RAD_RIGHT as _OPEN_RIGHT_TUPLE,
)

_OPEN_LEFT = np.asarray(_OPEN_LEFT_TUPLE, dtype=np.float64)
_CLOSED_LEFT = np.asarray(_CLOSED_LEFT_TUPLE, dtype=np.float64)
_OPEN_RIGHT = np.asarray(_OPEN_RIGHT_TUPLE, dtype=np.float64)
_CLOSED_RIGHT = np.asarray(_CLOSED_RIGHT_TUPLE, dtype=np.float64)

# Motor index per finger that is most reliably driven by that
# finger's flex curl alone (the *_pip motors and thumb_mcp).
_FINGER_TO_MOTOR_IDX = {
    "thumb": 2,    # thumb_mcp
    "index": 4,    # index_pip
    "middle": 5,   # middle_pip
    "ring": 7,     # ring_pip
    "pinky": 9,    # pinky_pip
}
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


# ── Loaders / decoders ──────────────────────────────────────────────────
def _decode_raw_curls(
    cmd_left: np.ndarray, cmd_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(raw_curl_left[N,5], raw_curl_right[N,5])`` from
    recorded motor commands."""
    n = cmd_left.shape[0]
    raw_left = np.empty((n, 5), dtype=np.float64)
    raw_right = np.empty((n, 5), dtype=np.float64)
    for fi, fname in enumerate(_FINGER_NAMES):
        m = _FINGER_TO_MOTOR_IDX[fname]
        span_l = _CLOSED_LEFT[m] - _OPEN_LEFT[m]
        span_r = _CLOSED_RIGHT[m] - _OPEN_RIGHT[m]
        raw_left[:, fi] = (cmd_left[:, m] - _OPEN_LEFT[m]) / span_l
        raw_right[:, fi] = (cmd_right[:, m] - _OPEN_RIGHT[m]) / span_r
    return np.clip(raw_left, 0.0, 1.0), np.clip(raw_right, 0.0, 1.0)


def _detect_hand_mode(raw_curl: np.ndarray) -> np.ndarray:
    """Return a boolean mask of frames where the per-finger curl
    spread is large enough to indicate hand-tracking mode (vs
    controller-uniform-trigger mode where all 5 curls are equal)."""
    return raw_curl.std(axis=1) >= 0.02


def _detect_static_frames(raw: np.ndarray, *, win: int = 10) -> np.ndarray:
    """Return a boolean mask of "static hand" frames -- frames where
    every finger's curl has been changing by < 0.005 over a sliding
    window of ``win`` frames. Useful for noise-floor characterisation.
    """
    n = raw.shape[0]
    mask = np.zeros(n, dtype=bool)
    if n < win:
        return mask
    for i in range(win, n):
        window = raw[i - win : i + 1]
        if (window.max(axis=0) - window.min(axis=0)).max() < 0.005:
            mask[i] = True
    return mask


def _within_block_residuals(raw: np.ndarray, win: int) -> np.ndarray | None:
    """Detect runs of contiguous static frames (each at most ``win``
    long) and return the residual ``(curl - block_mean)`` per finger
    pooled across all such blocks. Returns ``None`` if no blocks
    were found."""
    n = raw.shape[0]
    if n < win:
        return None
    static = _detect_static_frames(raw, win=win)
    blocks = []
    in_block = False
    start = 0
    for i in range(n):
        if static[i] and not in_block:
            start = i
            in_block = True
        elif not static[i] and in_block:
            if i - start >= win:
                blocks.append(raw[start:i])
            in_block = False
    if in_block and n - start >= win:
        blocks.append(raw[start:n])
    if not blocks:
        return None
    residuals = []
    for blk in blocks:
        residuals.append(blk - blk.mean(axis=0, keepdims=True))
    return np.concatenate(residuals, axis=0)


def _stretch(
    raw: np.ndarray, *, deadzone: float, full_threshold: float, gamma: float,
) -> np.ndarray:
    """Same math as :func:`stretch_finger_curls` but works on any
    array shape."""
    span = full_threshold - deadzone
    t = np.clip((raw - deadzone) / span, 0.0, 1.0)
    return t ** gamma


# ── Scoring ─────────────────────────────────────────────────────────────
@dataclass
class CandidateScore:
    deadzone: float
    full_threshold: float
    gamma: float
    bimodality: float
    rest_floor_leakage: float
    fist_saturation: float
    toggle_rate_per_s: float
    composite: float

    def fmt(self) -> str:
        return (
            f"dz={self.deadzone:.2f} full={self.full_threshold:.2f} "
            f"γ={self.gamma:.1f}  | "
            f"bimodal={self.bimodality:.3f} "
            f"rest_leak={self.rest_floor_leakage:.3f} "
            f"fist_sat={self.fist_saturation:.3f} "
            f"toggles={self.toggle_rate_per_s:>5.2f}/s "
            f"score={self.composite:+.3f}"
        )


def _score(
    raw_curls: np.ndarray, fps: float,
    deadzone: float, full_threshold: float, gamma: float, *,
    w_bimodal: float, w_rest: float, w_fist: float, w_toggle: float,
) -> CandidateScore:
    """Score a parameter triple against ``raw_curls`` (shape ``(N, 5)``).

    All four metrics are averaged across fingers. The composite is

        w_bimodal * bimodality
        + w_fist  * fist_saturation
        - w_rest  * rest_floor_leakage
        - w_toggle * toggle_rate_per_s

    Bigger composite is better.
    """
    if not (0.0 <= deadzone < full_threshold <= 1.0) or gamma <= 0:
        return CandidateScore(deadzone, full_threshold, gamma, 0, 1, 0, 1e9, -1e9)

    out = _stretch(
        raw_curls, deadzone=deadzone, full_threshold=full_threshold, gamma=gamma,
    )
    flat = out.reshape(-1)

    # 1. Bimodality
    bimodal = float(((flat < 0.05) | (flat > 0.95)).mean())

    # 2. Rest-floor leakage: post-stretch output at frames where
    # the raw curl is in the bottom 10 % of the per-finger
    # distribution.
    raw_p10 = np.percentile(raw_curls, 10, axis=0)
    rest_mask = raw_curls <= raw_p10[None, :]
    rest_leak = float(out[rest_mask].mean()) if rest_mask.any() else 0.0

    # 3. Fist saturation: post-stretch output at frames where the
    # raw curl is in the top 10 % of the per-finger distribution.
    raw_p90 = np.percentile(raw_curls, 90, axis=0)
    fist_mask = raw_curls >= raw_p90[None, :]
    fist_sat = float(out[fist_mask].mean()) if fist_mask.any() else 0.0

    # 4. Toggle rate: how often does the post-stretch output cross
    # back-and-forth between the two modes (< 0.05 and > 0.95) per
    # second per finger, averaged across fingers? A high toggle rate
    # indicates jitter at the threshold boundary.
    toggle_rate = _toggle_rate_per_s(out, fps)

    composite = (
        w_bimodal * bimodal
        + w_fist * fist_sat
        - w_rest * rest_leak
        - w_toggle * toggle_rate
    )
    return CandidateScore(
        deadzone, full_threshold, gamma, bimodal, rest_leak, fist_sat,
        toggle_rate, composite,
    )


def _toggle_rate_per_s(out: np.ndarray, fps: float) -> float:
    """Return the average per-finger rate (per second) of
    transitions in the binarised output (open <0.5 vs closed >=0.5).
    """
    binarised = (out >= 0.5).astype(np.int8)
    diffs = np.abs(np.diff(binarised, axis=0))
    n = out.shape[0]
    transitions_per_finger = diffs.sum(axis=0)
    duration_s = max(n - 1, 1) / fps
    return float(transitions_per_finger.mean() / duration_s)


def _score_per_finger(
    raw_curls: np.ndarray, fps: float, finger_idx: int,
    deadzone: float, full_threshold: float, gamma: float, *,
    w_bimodal: float, w_rest: float, w_fist: float, w_toggle: float,
) -> CandidateScore:
    """Same as :func:`_score` but only for one finger column."""
    return _score(
        raw_curls[:, finger_idx : finger_idx + 1], fps,
        deadzone, full_threshold, gamma,
        w_bimodal=w_bimodal, w_rest=w_rest, w_fist=w_fist, w_toggle=w_toggle,
    )


def _natural_modes(raw_curls: np.ndarray) -> tuple[float, float]:
    """1D Otsu split on the pooled raw curl distribution; returns
    ``(rest_mode_top_p95, curl_mode_bottom_p5)``."""
    flat = raw_curls.reshape(-1)
    flat = flat[(flat > 0.0) & (flat < 1.0)]
    if flat.size == 0:
        return 0.10, 0.30
    candidates = np.linspace(0.10, 0.50, 41)
    best_t = 0.20
    best_v = -1.0
    for t in candidates:
        below = flat[flat < t]
        above = flat[flat >= t]
        if below.size < 5 or above.size < 5:
            continue
        w0 = below.size / flat.size
        w1 = above.size / flat.size
        v = w0 * w1 * (below.mean() - above.mean()) ** 2
        if v > best_v:
            best_v = v
            best_t = float(t)
    rest_top = float(np.percentile(flat[flat < best_t], 95)) if (flat < best_t).any() else best_t
    curl_bot = float(np.percentile(flat[flat >= best_t], 5)) if (flat >= best_t).any() else best_t
    return rest_top, curl_bot


# ── Main ────────────────────────────────────────────────────────────────
def _load_episodes(npz_paths: list[Path]) -> tuple[np.ndarray, float]:
    """Pool hand-mode raw curls from all NPZ files. Returns
    ``(pooled_raw, fps)`` where ``pooled_raw`` has shape ``(N, 5)``.
    """
    pooled = []
    fps_seen: set[float] = set()
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        raw_l, raw_r = _decode_raw_curls(d["commanded_left_hand_q"], d["commanded_right_hand_q"])
        hand_l = _detect_hand_mode(raw_l)
        hand_r = _detect_hand_mode(raw_r)
        if hand_l.any():
            pooled.append(raw_l[hand_l])
        if hand_r.any():
            pooled.append(raw_r[hand_r])
        fps_seen.add(float(d["fps"]))
        n = int(d["num_frames"])
        print(f"# {p.name}: {n} frames @ {float(d['fps']):.1f} fps; "
              f"hand-mode L={int(hand_l.sum())} ({100*hand_l.mean():.1f}%) "
              f"R={int(hand_r.sum())} ({100*hand_r.mean():.1f}%)")
    if not pooled:
        raise SystemExit("error: no hand-mode frames found in any episode")
    if len(fps_seen) != 1:
        print(f"# warning: episodes have differing fps: {fps_seen}; using max")
    fps = max(fps_seen) if fps_seen else 50.0
    return np.concatenate(pooled, axis=0), fps


def _print_distribution(raw: np.ndarray) -> None:
    print()
    print("# Raw curl distribution per finger (pooled hand-mode frames)")
    print(f"  {'finger':<8} {'min':>6} {'p05':>6} {'p10':>6} {'p25':>6} "
          f"{'p50':>6} {'p75':>6} {'p90':>6} {'p99':>6} {'max':>6}")
    for fi, fname in enumerate(_FINGER_NAMES):
        col = raw[:, fi]
        print(f"  {fname:<8} {col.min():>6.3f} "
              f"{np.percentile(col, 5):>6.3f} "
              f"{np.percentile(col, 10):>6.3f} "
              f"{np.percentile(col, 25):>6.3f} "
              f"{np.percentile(col, 50):>6.3f} "
              f"{np.percentile(col, 75):>6.3f} "
              f"{np.percentile(col, 90):>6.3f} "
              f"{np.percentile(col, 99):>6.3f} "
              f"{col.max():>6.3f}")


def _print_response_curve(dz: float, full: float, gamma: float) -> None:
    canon = np.array([0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.27, 0.28,
                      0.30, 0.40, 0.50, 0.92])
    out = _stretch(canon, deadzone=dz, full_threshold=full, gamma=gamma)
    print(f"# Response curve (dz={dz:.2f}, full={full:.2f}, γ={gamma:.1f}):")
    print(f"  {'raw':>6}  {'output':>8}")
    for r, o in zip(canon, out):
        print(f"  {r:>6.2f}  {o:>8.3f}")


def _mode_global(args: argparse.Namespace, raw: np.ndarray, fps: float) -> int:
    rest_top, curl_bot = _natural_modes(raw)
    print(f"# Natural bimodal break (1D Otsu): rest mode tops at "
          f"{rest_top:.3f}, curl mode begins at {curl_bot:.3f}")

    dz_grid = [float(x) for x in args.dz_grid.split(",")]
    full_grid = [float(x) for x in args.full_grid.split(",")]
    gamma_grid = [float(x) for x in args.gamma_grid.split(",")]
    print(f"# Sweeping {len(dz_grid) * len(full_grid) * len(gamma_grid)} candidates "
          f"({len(dz_grid)} dz x {len(full_grid)} full x {len(gamma_grid)} gamma)")

    cands: list[CandidateScore] = []
    for dz in dz_grid:
        for fl in full_grid:
            if fl <= dz:
                continue
            for g in gamma_grid:
                cands.append(_score(
                    raw, fps, dz, fl, g,
                    w_bimodal=args.w_bimodal, w_rest=args.w_rest,
                    w_fist=args.w_fist, w_toggle=args.w_toggle,
                ))
    cands.sort(key=lambda c: c.composite, reverse=True)

    print()
    print(f"# Top-{args.top_k} candidates by composite "
          f"({args.w_bimodal:.1f}*bimodal + {args.w_fist:.1f}*fist "
          f"- {args.w_rest:.1f}*rest - {args.w_toggle:.1f}*toggles)")
    print(f"# {'rank':>4}  {'cand':<70}")
    for i, c in enumerate(cands[: args.top_k]):
        print(f"  {i+1:>4}  {c.fmt()}")

    print()
    print("# Best candidate -> recommended parameter values:")
    best = cands[0]
    print(f"  DEFAULT_CURL_DEADZONE       = {best.deadzone:.2f}")
    print(f"  DEFAULT_CURL_FULL_THRESHOLD = {best.full_threshold:.2f}")
    print(f"  DEFAULT_CURL_GAMMA          = {best.gamma:.1f}")
    print()
    _print_response_curve(best.deadzone, best.full_threshold, best.gamma)
    return 0


def _mode_per_finger(args: argparse.Namespace, raw: np.ndarray, fps: float) -> int:
    print()
    print("# Per-finger best parameters (independent sweeps):")
    print(f"  {'finger':<8} {'dz':>5} {'full':>5} {'γ':>4} "
          f"{'bimod':>6} {'rest':>6} {'sat':>5} {'tog/s':>6}")

    dz_grid = [float(x) for x in args.dz_grid.split(",")]
    full_grid = [float(x) for x in args.full_grid.split(",")]
    gamma_grid = [float(x) for x in args.gamma_grid.split(",")]

    for fi, fname in enumerate(_FINGER_NAMES):
        cands = []
        for dz in dz_grid:
            for fl in full_grid:
                if fl <= dz:
                    continue
                for g in gamma_grid:
                    cands.append(_score_per_finger(
                        raw, fps, fi, dz, fl, g,
                        w_bimodal=args.w_bimodal, w_rest=args.w_rest,
                        w_fist=args.w_fist, w_toggle=args.w_toggle,
                    ))
        cands.sort(key=lambda c: c.composite, reverse=True)
        b = cands[0]
        print(f"  {fname:<8} {b.deadzone:>5.2f} {b.full_threshold:>5.2f} "
              f"{b.gamma:>4.1f} {b.bimodality:>6.3f} {b.rest_floor_leakage:>6.3f} "
              f"{b.fist_saturation:>5.3f} {b.toggle_rate_per_s:>6.2f}")
    return 0


def _mode_noise_floor(args: argparse.Namespace, raw: np.ndarray, fps: float) -> int:
    static_mask = _detect_static_frames(raw, win=args.static_win)
    static_count = int(static_mask.sum())
    print()
    print(f"# Static-hand frames (window={args.static_win}, max ptp <0.005): "
          f"{static_count} / {raw.shape[0]} ({100 * static_count / raw.shape[0]:.1f}%)")
    if static_count < 50:
        print("# warning: too few static frames to characterise noise floor")
        return 0

    residuals = _within_block_residuals(raw, win=args.static_win)
    if residuals is None or residuals.size == 0:
        print("# warning: no contiguous static blocks of length >= "
              f"{args.static_win} found")
        return 0
    abs_resid = np.abs(residuals)
    print()
    print(f"# Per-finger noise floor (within-block RMS deviation across "
          f"{residuals.shape[0]} pooled static frames):")
    print(f"  {'finger':<8} {'rms':>7} {'p95':>7} {'p99':>7} {'max':>7}")
    for fi, fname in enumerate(_FINGER_NAMES):
        col = abs_resid[:, fi]
        print(f"  {fname:<8} {np.sqrt(np.mean(col ** 2)):>7.4f} "
              f"{np.percentile(col, 95):>7.4f} "
              f"{np.percentile(col, 99):>7.4f} "
              f"{col.max():>7.4f}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("npz", type=Path, nargs="+",
                   help="One or more teleop_episode_NNNN.npz paths.")
    p.add_argument(
        "--mode", choices=["global", "per-finger", "noise-floor"],
        default="global", help="Tuning strategy (see module docstring).",
    )
    p.add_argument("--top-k", type=int, default=10,
                   help="Print the K best parameter combos in global mode.")
    p.add_argument("--w-bimodal", type=float, default=1.0)
    p.add_argument("--w-fist", type=float, default=1.0)
    p.add_argument("--w-rest", type=float, default=2.0)
    p.add_argument("--w-toggle", type=float, default=0.05)
    p.add_argument(
        "--dz-grid", type=str,
        default="0.10,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.32,0.35",
    )
    p.add_argument(
        "--full-grid", type=str,
        default="0.27,0.28,0.30,0.32,0.35,0.40,0.45,0.50",
    )
    p.add_argument(
        "--gamma-grid", type=str, default="1.0,1.5,2.0,2.5,3.0,3.5,4.0,5.0",
    )
    p.add_argument("--static-win", type=int, default=10,
                   help="Sliding window length (frames) for static-hand "
                        "detection in noise-floor mode.")
    args = p.parse_args()

    for path in args.npz:
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    raw, fps = _load_episodes(list(args.npz))
    print(f"# Pooled hand-mode frames: {raw.shape[0]}")

    _print_distribution(raw)

    if args.mode == "global":
        return _mode_global(args, raw, fps)
    elif args.mode == "per-finger":
        return _mode_per_finger(args, raw, fps)
    elif args.mode == "noise-floor":
        return _mode_noise_floor(args, raw, fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
