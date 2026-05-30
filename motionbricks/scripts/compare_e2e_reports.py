"""Side-by-side parity table for two ``test_e2e_velocity_tracking`` reports.

Loads two JSON reports (e.g. one X2 and one G1, same sweep and horizon),
aligns trials by ``(axis, intent)`` and prints a dimensionless tracking
comparison plus a verdict on structural parity. The dimensionless slope
is the headline "apples-to-apples" number: skeleton-independent, target
~1.0 for a healthy stack.

Usage::

    python motionbricks/scripts/compare_e2e_reports.py \\
        --left  out/per_model_report/e2e_x2_all.json --left-label X2 \\
        --right out/per_model_report/e2e_g1_all.json --right-label G1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _fmt(v: float | None, fmt: str = "{:+.3f}") -> str:
    return fmt.format(v) if v is not None else "  --"


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _intent_key(intent: dict[str, float]) -> tuple:
    return (
        round(float(intent["yaw_rate"]), 4),
        round(float(intent["vel_x"]), 4),
        round(float(intent["vel_z"]), 4),
    )


def _index_by_intent(rows: list[dict]) -> dict[tuple, dict]:
    return {_intent_key(r["intent"]): r for r in rows}


def _verdict(left_slope: float | None, right_slope: float | None) -> str:
    if left_slope is None or right_slope is None:
        return "n/a"
    # Both within 25% of 1.0 + within 25% of each other = parity.
    l_dev = abs(left_slope - 1.0)
    r_dev = abs(right_slope - 1.0)
    pair_dev = abs(left_slope - right_slope)
    if l_dev < 0.25 and r_dev < 0.25 and pair_dev < 0.25:
        return "OK (both within 0.25 of ideal)"
    if pair_dev < 0.25:
        return f"PARITY (left={left_slope:.2f} ~ right={right_slope:.2f}, both miss ideal)"
    if l_dev < r_dev:
        return "LEFT closer to ideal"
    return "RIGHT closer to ideal"


def _print_axis_compare(
    axis: str,
    left_rows: list[dict],
    right_rows: list[dict],
    left_label: str,
    right_label: str,
) -> None:
    print(f"\n  axis: {axis}")
    if axis == "forward":
        cmd_attr = "vel_z"
        ach_key = "achieved_forward_m"
        tr_key = "tracking_forward"
    elif axis == "lateral":
        cmd_attr = "vel_x"
        ach_key = "achieved_lateral_m"
        tr_key = "tracking_lateral"
    else:
        cmd_attr = "yaw_rate"
        ach_key = "achieved_dyaw_deg"
        tr_key = "tracking_yaw"

    left_idx = _index_by_intent(left_rows)
    right_idx = _index_by_intent(right_rows)
    keys = sorted(set(left_idx) | set(right_idx))

    cmd_label = "yaw_cmd" if axis == "yaw" else f"{cmd_attr}_cmd"
    print(
        f"  {cmd_label:>10} | {left_label + '_ach':>10} {right_label + '_ach':>10} | "
        f"{left_label + '_track':>9} {right_label + '_track':>9}"
    )
    print(f"  {'-'*10} | {'-'*10} {'-'*10} | {'-'*9} {'-'*9}")
    for k in keys:
        l = left_idx.get(k)
        r = right_idx.get(k)
        intent = (l or r)["intent"]
        cmd = float(intent[cmd_attr])
        l_ach = l["metrics"][ach_key] if l else None
        r_ach = r["metrics"][ach_key] if r else None
        l_tr = l["metrics"][tr_key] if l else None
        r_tr = r["metrics"][tr_key] if r else None
        print(
            f"  {cmd:>10.3f} | {_fmt(l_ach):>10} {_fmt(r_ach):>10} | "
            f"{_fmt(l_tr, '{:.2f}'):>9} {_fmt(r_tr, '{:.2f}'):>9}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--left", type=Path, required=True)
    p.add_argument("--right", type=Path, required=True)
    p.add_argument("--left-label", default="LEFT")
    p.add_argument("--right-label", default="RIGHT")
    args = p.parse_args(argv)

    left = _load(args.left)
    right = _load(args.right)

    print("=" * 92)
    print(
        f"  E2E parity:  {args.left_label} vs {args.right_label}"
        f"   (horizon={left['horizon_s']:.1f}s/{right['horizon_s']:.1f}s)"
    )
    print("=" * 92)
    print(
        f"  {args.left_label}:  {left['ckpt_set']}/{left['fixture']}  "
        f"hip_h={left['hip_h']:.3f}  fps={left['fps']}"
    )
    print(
        f"  {args.right_label}: {right['ckpt_set']}/{right['fixture']}  "
        f"hip_h={right['hip_h']:.3f}  fps={right['fps']}"
    )

    common_axes = sorted(
        set(left["sweep_axes"]) & set(right["sweep_axes"]),
        key=["forward", "lateral", "yaw"].index,
    )
    for axis in common_axes:
        l_rows = [r for r in left["trials"] if r["axis"] == axis]
        r_rows = [r for r in right["trials"] if r["axis"] == axis]
        _print_axis_compare(
            axis, l_rows, r_rows, args.left_label, args.right_label
        )

    # Dimensionless slope parity summary.
    print()
    print("=" * 92)
    print("  Dimensionless slope parity  (ideal ~1.0)")
    print("=" * 92)
    print(
        f"  {'axis':<10} | {args.left_label:>10} | {args.right_label:>10} | "
        f"{'verdict':<40}"
    )
    print(f"  {'-'*10} | {'-'*10} | {'-'*10} | {'-'*40}")
    for axis in common_axes:
        l_slope = left["slopes"].get(axis)
        r_slope = right["slopes"].get(axis)
        verdict = _verdict(l_slope, r_slope)
        print(
            f"  {axis:<10} | {_fmt(l_slope, '{:.3f}'):>10} | "
            f"{_fmt(r_slope, '{:.3f}'):>10} | {verdict:<40}"
        )
    print("=" * 92)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
