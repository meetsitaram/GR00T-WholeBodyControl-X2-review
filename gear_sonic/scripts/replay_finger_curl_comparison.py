"""Visual replay-comparison tool for the Quest3 finger-curl mapping.

Loads a debug NPZ recorded by ``teleop_x2_kinematic.py`` /
``record_x2_dataset.py``, decodes the raw per-finger curls from
the recorded ``commanded_*_hand_q``, then re-applies the live
:func:`stretch_finger_curls` (per-finger defaults) to produce the
*new* motor commands that would have been issued under the current
runtime. Renders a side-by-side time-series PNG so you can
visually verify that:

* "Open hand" frames where the operator wasn't intentionally
  curling no longer drive the motor anywhere off zero, AND
* "Intentional curl" frames produce a fully closed motor command
  even if Quest3's per-finger raw value tops out below 1.0.

The script *only* works on debug NPZs recorded **before** the
Python-side stretch was wired in (i.e., the v3 episodes from
May 10 03:02 and earlier), where the recorded
``commanded_*_hand_q`` is a direct linear lerp from raw curl and
the inverse is exact. For newer recordings (post-stretch) the
recorded command can't be inverted to recover raw curls
unambiguously and the script will emit a warning and exit.

Usage
-----

::

    python -m gear_sonic.scripts.replay_finger_curl_comparison \\
        data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_000000.npz \\
        --out outputs/finger_curl_comparison_ep0.png

    # Replay all 4 episodes in a batch:
    for npz in data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_*.npz; do
        python -m gear_sonic.scripts.replay_finger_curl_comparison "$npz" \\
            --out "outputs/finger_curl_comparison_$(basename "$npz" .npz).png"
    done

Each output PNG has 5 rows (one per finger) and 2 columns (left
/ right hand). Each panel plots three lines on a 0..1 y-axis:

* gray dotted: raw Quest3 curl (already in [0, 1])
* orange dashed: OLD motor command -- direct linear lerp,
  no stretch applied
* green solid: NEW motor command -- per-finger stretch applied,
  using the live :data:`DEFAULT_CURL_*_PER_FINGER`

A vertical light-blue band at each finger's deadzone .. full
threshold is drawn for reference. Red dots mark "false trigger"
frames where the OLD path would have moved the motor > 0.10
while the operator was at rest (raw < deadzone) -- these are the
frames the new mapping cleans up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless-friendly
import matplotlib.pyplot as plt

from gear_sonic.utils.teleop.x2_hand_retarget import (
    DEFAULT_CURL_DEADZONE_PER_FINGER,
    DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER,
    DEFAULT_CURL_GAMMA_PER_FINGER,
    HAND_GRASP_CLOSED_RAD_LEFT,
    HAND_GRASP_CLOSED_RAD_RIGHT,
    HAND_GRASP_OPEN_RAD_LEFT,
    HAND_GRASP_OPEN_RAD_RIGHT,
    stretch_finger_curls,
)


# Same finger-to-canonical-motor mapping the tuner uses to
# decode raw curls from the recorded motor commands.
_FINGER_TO_MOTOR_IDX = {
    "thumb": 2,    # thumb_mcp
    "index": 4,    # index_pip
    "middle": 5,   # middle_pip
    "ring": 7,     # ring_pip
    "pinky": 9,    # pinky_pip
}
_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

# The motors driven directly by per-finger curl (used to score
# "false-trigger" rate). thumb_roll/abad ride opposition in newer
# code paths and aren't captured by this comparison.
_FINGER_TO_CMD_NORM_MOTORS = {
    "thumb": (2,),                  # thumb_mcp only
    "index": (3, 4),                 # index_abad, index_pip
    "middle": (5,),
    "ring": (6, 7),
    "pinky": (8, 9),
}


def _decode_raw_curls(
    cmd_left: np.ndarray, cmd_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    open_l = np.asarray(HAND_GRASP_OPEN_RAD_LEFT, dtype=np.float64)
    closed_l = np.asarray(HAND_GRASP_CLOSED_RAD_LEFT, dtype=np.float64)
    open_r = np.asarray(HAND_GRASP_OPEN_RAD_RIGHT, dtype=np.float64)
    closed_r = np.asarray(HAND_GRASP_CLOSED_RAD_RIGHT, dtype=np.float64)
    n = cmd_left.shape[0]
    raw_left = np.empty((n, 5), dtype=np.float64)
    raw_right = np.empty((n, 5), dtype=np.float64)
    for fi, fname in enumerate(_FINGER_NAMES):
        m = _FINGER_TO_MOTOR_IDX[fname]
        span_l = closed_l[m] - open_l[m]
        span_r = closed_r[m] - open_r[m]
        raw_left[:, fi] = (cmd_left[:, m] - open_l[m]) / span_l
        raw_right[:, fi] = (cmd_right[:, m] - open_r[m]) / span_r
    return np.clip(raw_left, 0.0, 1.0), np.clip(raw_right, 0.0, 1.0)


def _normalize_cmd_to_curl(cmd: np.ndarray, side: str) -> np.ndarray:
    """Decode the *post-stretch* effective curl per finger from a
    motor command, by inverting the linear lerp on the canonical
    motor index per finger."""
    if side == "left":
        open_a = np.asarray(HAND_GRASP_OPEN_RAD_LEFT, dtype=np.float64)
        closed_a = np.asarray(HAND_GRASP_CLOSED_RAD_LEFT, dtype=np.float64)
    else:
        open_a = np.asarray(HAND_GRASP_OPEN_RAD_RIGHT, dtype=np.float64)
        closed_a = np.asarray(HAND_GRASP_CLOSED_RAD_RIGHT, dtype=np.float64)
    n = cmd.shape[0]
    out = np.empty((n, 5), dtype=np.float64)
    for fi, fname in enumerate(_FINGER_NAMES):
        m = _FINGER_TO_MOTOR_IDX[fname]
        span = closed_a[m] - open_a[m]
        out[:, fi] = (cmd[:, m] - open_a[m]) / span
    return np.clip(out, 0.0, 1.0)


def _check_recording_is_pre_stretch(
    raw_left: np.ndarray, raw_right: np.ndarray, *, eps: float = 0.001,
) -> bool:
    """Heuristic: post-stretch recordings have raw curls heavily
    pulled to 0 or 1 (because the stretch is near-binary). Pre-
    stretch recordings have a smooth distribution. Specifically,
    a pre-stretch recording will have at least 2% of frames in
    the [0.30, 0.70] middle range; a post-stretch recording is
    much closer to 0%.
    """
    pooled = np.concatenate([raw_left, raw_right], axis=0)
    middle = ((pooled >= 0.30) & (pooled <= 0.70)).any(axis=1)
    middle_frac = middle.mean()
    return middle_frac > 0.02


def _apply_new_stretch(raw_curls: np.ndarray) -> np.ndarray:
    out = np.empty_like(raw_curls)
    for i in range(raw_curls.shape[0]):
        out[i] = stretch_finger_curls(raw_curls[i])
    return out


def _make_plot(
    raw_left: np.ndarray, raw_right: np.ndarray,
    new_left: np.ndarray, new_right: np.ndarray,
    fps: float, title: str, out_path: Path,
) -> None:
    n = raw_left.shape[0]
    t = np.arange(n) / fps

    dz = np.asarray(DEFAULT_CURL_DEADZONE_PER_FINGER)
    full = np.asarray(DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER)

    fig, axes = plt.subplots(
        5, 2, figsize=(14, 12), sharex=True, sharey=True, constrained_layout=True,
    )
    fig.suptitle(title, fontsize=14)

    for fi, fname in enumerate(_FINGER_NAMES):
        for col, (raw, new, side) in enumerate([
            (raw_left[:, fi], new_left[:, fi], "left"),
            (raw_right[:, fi], new_right[:, fi], "right"),
        ]):
            ax = axes[fi, col]
            ax.axhspan(dz[fi], full[fi], color="lightblue", alpha=0.3,
                       label=f"active zone [{dz[fi]:.2f}, {full[fi]:.2f}]")
            ax.axhline(dz[fi], color="steelblue", linestyle=":", linewidth=0.7)
            ax.axhline(full[fi], color="steelblue", linestyle=":", linewidth=0.7)
            ax.plot(t, raw, color="dimgray", linestyle=":", linewidth=0.8,
                    label="raw Quest3 curl")
            ax.plot(t, raw, color="orange", linestyle="--", linewidth=1.0,
                    alpha=0.7, label="OLD: raw lerp (no stretch)")
            ax.plot(t, new, color="green", linewidth=1.4, alpha=0.85,
                    label="NEW: per-finger stretch")
            # Red dots = false-trigger frames the new mapping cleans up.
            false_trigger = (raw <= dz[fi]) & (raw > 0.10)
            if false_trigger.any():
                ax.scatter(
                    t[false_trigger], raw[false_trigger],
                    color="red", s=4, alpha=0.4, zorder=5,
                    label=f"frames at rest with raw>0.10 ({false_trigger.sum()})",
                )

            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{fname} ({side})", fontsize=10)
            ax.grid(alpha=0.3)
            if fi == 4:
                ax.set_xlabel("time (s)")
            if col == 0:
                ax.set_ylabel("[0, 1]")
            if fi == 0 and col == 1:
                ax.legend(loc="upper right", fontsize=7, framealpha=0.7)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"# wrote {out_path}")


def _print_summary(
    raw_left: np.ndarray, raw_right: np.ndarray,
    new_left: np.ndarray, new_right: np.ndarray,
) -> None:
    """Print quantitative before/after stats per finger pooled
    across both sides."""
    pooled_raw = np.concatenate([raw_left, raw_right], axis=0)
    pooled_new = np.concatenate([new_left, new_right], axis=0)
    dz = np.asarray(DEFAULT_CURL_DEADZONE_PER_FINGER)
    full = np.asarray(DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER)

    print()
    print("# Per-finger summary (pooled left+right)")
    print(f"  {'finger':<8} {'frames':>6} "
          f"{'old<0.05':>9} {'old>0.95':>9} {'old_bimod':>10} "
          f"{'new<0.05':>9} {'new>0.95':>9} {'new_bimod':>10} "
          f"{'cleaned':>8}")
    for fi, fname in enumerate(_FINGER_NAMES):
        n = pooled_raw.shape[0]
        old = pooled_raw[:, fi]   # OLD = raw lerp = same value as raw
        new = pooled_new[:, fi]
        old_low = float((old < 0.05).mean())
        old_hi = float((old > 0.95).mean())
        new_low = float((new < 0.05).mean())
        new_hi = float((new > 0.95).mean())
        # "cleaned" = frames where raw was in active zone but new
        # snapped to 0 or 1 (i.e., resolved an ambiguous frame).
        ambig = (old >= 0.05) & (old <= 0.95)
        resolved = ambig & ((new < 0.05) | (new > 0.95))
        cleaned = float(resolved.sum() / max(ambig.sum(), 1))
        print(f"  {fname:<8} {n:>6} "
              f"{old_low:>9.3f} {old_hi:>9.3f} {old_low + old_hi:>10.3f} "
              f"{new_low:>9.3f} {new_hi:>9.3f} {new_low + new_hi:>10.3f} "
              f"{cleaned:>8.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("npz", type=Path, help="A teleop_episode_NNNN.npz path.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output PNG path. Defaults to "
                        "outputs/finger_curl_comparison_<stem>.png.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the plot, only print the summary.")
    args = p.parse_args()

    if not args.npz.is_file():
        print(f"error: {args.npz} not found", file=sys.stderr)
        return 1

    d = np.load(args.npz, allow_pickle=True)
    n = int(d["num_frames"])
    fps = float(d["fps"])
    print(f"# {args.npz.name}: {n} frames @ {fps:.1f} fps "
          f"({n / fps:.1f}s)")

    raw_left, raw_right = _decode_raw_curls(
        d["commanded_left_hand_q"], d["commanded_right_hand_q"],
    )
    if not _check_recording_is_pre_stretch(raw_left, raw_right):
        print("# WARNING: recording appears to have been made AFTER the "
              "stretch was wired in (raw curls are bimodal already). "
              "Comparison may be misleading.", file=sys.stderr)

    new_left = _apply_new_stretch(raw_left)
    new_right = _apply_new_stretch(raw_right)

    _print_summary(raw_left, raw_right, new_left, new_right)

    if not args.no_plot:
        out_path = args.out or Path(
            f"outputs/finger_curl_comparison_{args.npz.stem}.png"
        )
        title = (
            f"Finger curl mapping comparison -- {args.npz.name}\n"
            f"OLD: raw lerp (no stretch) vs NEW: per-finger stretch "
            f"(dz={DEFAULT_CURL_DEADZONE_PER_FINGER}, "
            f"full={DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER}, "
            f"γ={DEFAULT_CURL_GAMMA_PER_FINGER})"
        )
        _make_plot(raw_left, raw_right, new_left, new_right, fps, title, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
