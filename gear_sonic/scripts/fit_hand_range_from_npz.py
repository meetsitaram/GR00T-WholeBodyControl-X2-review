"""Fit a per-finger ``(floor, ceiling)`` hand-range from a recorded
debug NPZ and patch it into an operator-calibration YAML.

The recorded NPZ must contain ``quest_left_hand_curls`` /
``quest_right_hand_curls`` (shape ``(N, 5)``) and ``quest_left_thumb_oppose``
/ ``quest_right_thumb_oppose`` (shape ``(N,)``). The NPZ written by
``teleop_x2_kinematic.py`` after May 2026 carries these per-frame
arrays automatically (see the ``_save_debug_npz`` block in that
script).

The fit is per-finger, per-side:

* floor   = ``p_low``-th percentile of valid (non-NaN) frames
* ceiling = ``p_high``-th percentile, clamped to ``floor + min_spread``
* oppose floor / ceiling = same statistics on the scalar opposition signal

Pass ``--combine min,max`` to use literal min/max instead of percentiles
(more aggressive; recommended only when the operator deliberately held
both extremes during recording).

Usage::

    python -m gear_sonic.scripts.fit_hand_range_from_npz \\
        --npz data/lerobot/x2_quest3_kinematic_v4/debug/teleop_episode_000000.npz \\
        --calibration data/operator_calibrations/default.yaml \\
        --p-low 5 --p-high 95

The calibration YAML is rewritten in-place with a new ``hand_range``
section. Pass ``--output-calibration`` to write to a different path.

If you supply ``--dry-run`` the new ranges are printed but the YAML is
NOT rewritten -- handy for inspecting the fit before committing it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--npz", type=Path, required=True)
    p.add_argument(
        "--calibration", type=Path, required=True,
        help="Operator-calibration YAML to patch (read + rewritten).",
    )
    p.add_argument(
        "--output-calibration", type=Path, default=None,
        help="Write the patched calibration to this path instead of "
             "overwriting --calibration.",
    )
    p.add_argument(
        "--combine", choices=("p", "minmax"), default="p",
        help="`p` (default): use ``--p-low`` / ``--p-high`` percentiles. "
             "`minmax`: use literal min/max of valid frames (aggressive).",
    )
    p.add_argument("--p-low", type=float, default=5.0)
    p.add_argument("--p-high", type=float, default=95.0)
    p.add_argument(
        "--min-spread", type=float, default=0.05,
        help="Minimum (ceiling - floor) per finger after clamping. "
             "Prevents fingers that didn't see a closed pose during the "
             "session from collapsing to a single point.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the fit but do NOT rewrite the calibration YAML.",
    )
    return p.parse_args(argv)


def _fit_side(
    *,
    curls_arr: np.ndarray,
    oppose_arr: np.ndarray,
    label: str,
    combine: str,
    p_low: float,
    p_high: float,
    min_spread: float,
) -> tuple[np.ndarray, np.ndarray, float, float, int]:
    valid = ~np.isnan(curls_arr).any(axis=1)
    n_valid = int(valid.sum())
    if n_valid < 50:
        raise SystemExit(
            f"{label}: need >=50 hand-mode frames; got {n_valid}. "
            f"Make sure the operator engaged hand tracking and made some "
            f"finger gestures during the recording."
        )
    if combine == "minmax":
        fl = curls_arr[valid].min(axis=0)
        ce = curls_arr[valid].max(axis=0)
    else:
        fl = np.percentile(curls_arr[valid], p_low, axis=0)
        ce = np.percentile(curls_arr[valid], p_high, axis=0)
    ce = np.maximum(ce, fl + min_spread)
    fl = np.clip(fl, 0.0, 0.999)
    ce = np.clip(ce, fl + min_spread, 1.0)

    valid_o = ~np.isnan(oppose_arr)
    if int(valid_o.sum()) < 10:
        o_fl, o_ce = 0.0, 1.0
    else:
        if combine == "minmax":
            o_fl = float(oppose_arr[valid_o].min())
            o_ce = float(oppose_arr[valid_o].max())
        else:
            o_fl = float(np.percentile(oppose_arr[valid_o], p_low))
            o_ce = float(np.percentile(oppose_arr[valid_o], p_high))
        o_ce = max(o_ce, o_fl + min_spread)
        o_fl = float(np.clip(o_fl, 0.0, 0.999))
        o_ce = float(np.clip(o_ce, o_fl + min_spread, 1.0))
    return fl, ce, o_fl, o_ce, n_valid


def _print_side(side_label: str, fl: np.ndarray, ce: np.ndarray, o_fl: float, o_ce: float) -> None:
    print(f"  {side_label}:")
    print(f"    finger    floor  ceiling  span")
    for i, n in enumerate(_FINGER_NAMES):
        print(f"    {n:6s}    {fl[i]:.3f}  {ce[i]:.3f}    {ce[i]-fl[i]:.3f}")
    print(f"    oppose    {o_fl:.3f}  {o_ce:.3f}    {o_ce-o_fl:.3f}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.npz.is_file():
        raise SystemExit(f"NPZ not found: {args.npz}")
    if not args.calibration.is_file():
        raise SystemExit(f"calibration YAML not found: {args.calibration}")

    npz = np.load(args.npz, allow_pickle=True)
    for key in (
        "quest_left_hand_curls", "quest_right_hand_curls",
        "quest_left_thumb_oppose", "quest_right_thumb_oppose",
    ):
        if key not in npz.files:
            raise SystemExit(
                f"NPZ {args.npz} is missing key {key!r}. Re-record with a "
                f"recent teleop_x2_kinematic.py (>= May 2026) -- raw Quest "
                f"hand inputs are stored per-frame in the debug NPZ."
            )
    l_curls = np.asarray(npz["quest_left_hand_curls"], dtype=np.float64)
    r_curls = np.asarray(npz["quest_right_hand_curls"], dtype=np.float64)
    l_opp = np.asarray(npz["quest_left_thumb_oppose"], dtype=np.float64)
    r_opp = np.asarray(npz["quest_right_thumb_oppose"], dtype=np.float64)

    print(f"[fit_hand_range] NPZ: {args.npz}")
    print(f"[fit_hand_range] frames: {l_curls.shape[0]}")
    print(f"[fit_hand_range] combine: {args.combine}  "
          f"p_low={args.p_low}  p_high={args.p_high}  "
          f"min_spread={args.min_spread}")
    l_fl, l_ce, l_o_fl, l_o_ce, n_l = _fit_side(
        curls_arr=l_curls, oppose_arr=l_opp, label="left",
        combine=args.combine,
        p_low=args.p_low, p_high=args.p_high, min_spread=args.min_spread,
    )
    r_fl, r_ce, r_o_fl, r_o_ce, n_r = _fit_side(
        curls_arr=r_curls, oppose_arr=r_opp, label="right",
        combine=args.combine,
        p_low=args.p_low, p_high=args.p_high, min_spread=args.min_spread,
    )

    print(f"\n[fit_hand_range] hand-mode frames: L={n_l} R={n_r}\n")
    _print_side("LEFT ", l_fl, l_ce, l_o_fl, l_o_ce)
    print()
    _print_side("RIGHT", r_fl, r_ce, r_o_fl, r_o_ce)

    from gear_sonic.utils.teleop.operator_calibration import (
        HandRangeCalibration,
        HandRangeFit,
        OperatorCalibration,
    )

    cal = OperatorCalibration.load_yaml(args.calibration)
    npz_resolved = args.npz.resolve()
    try:
        npz_rel = npz_resolved.relative_to(REPO_ROOT.resolve())
        source = f"npz:{npz_rel}"
    except ValueError:
        source = f"npz:{npz_resolved}"
    cal.hand_range = HandRangeCalibration(
        left=HandRangeFit(
            floor=l_fl, ceiling=l_ce,
            oppose_floor=l_o_fl, oppose_ceiling=l_o_ce,
        ),
        right=HandRangeFit(
            floor=r_fl, ceiling=r_ce,
            oppose_floor=r_o_fl, oppose_ceiling=r_o_ce,
        ),
        source=source,
        samples=min(n_l, n_r),
    )

    if args.dry_run:
        print("\n[fit_hand_range] --dry-run: NOT writing the YAML")
        return 0

    out_path = args.output_calibration if args.output_calibration is not None else args.calibration
    cal.save_yaml(out_path)
    print(f"\n[fit_hand_range] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
