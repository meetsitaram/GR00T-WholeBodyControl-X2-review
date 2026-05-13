"""Offline replay of a recorded Quest 3 → X2 LeRobot dataset.

Reads a recorded debug NPZ (raw Quest 3 inputs per frame) and the
matching LeRobot v2.1 parquet (recorded robot commands per frame),
and re-runs the **current** retargeting code path
(:func:`per_finger_grasp_command_from_curls_and_oppose` /
:func:`grasp_command_from_ratio`) over the recorded raw inputs to
produce a regenerated commanded-hand-q time series.

The regenerated commands are written into a NEW parquet alongside
the original (the original is never overwritten). Every other column
in the parquet is copied bit-for-bit. The script then diffs the
regenerated ``action.left_hand_joints`` / ``action.right_hand_joints``
columns against the recorded values and reports per-motor
``L1`` / ``Linf`` errors plus exact-match counts.

Use this as a sanity check **before** changing any retargeting code:
running this with no code changes must produce a near-zero diff
(float-precision noise only). Once that passes, modify the
retargeting and re-run -- the diff then quantifies the effect of
the change.

Example::

    python -m gear_sonic.scripts.replay_recorded_dataset \\
        --npz data/lerobot/x2_quest3_kinematic_v4/debug/teleop_episode_000000.npz \\
        --parquet data/lerobot/x2_quest3_kinematic_v4/data/chunk-000/episode_000000.parquet \\
        --output-suffix _replay_baseline \\
        --hand-input max
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--npz", type=Path, required=True,
        help="Debug NPZ for the episode (must contain quest_*_hand_curls, "
             "quest_*_thumb_oppose, controller_triggers).",
    )
    p.add_argument(
        "--parquet", type=Path, required=True,
        help="Recorded LeRobot parquet for the same episode "
             "(action.left_hand_joints / action.right_hand_joints are "
             "regenerated; every other column is copied as-is).",
    )
    p.add_argument(
        "--output-suffix", type=str, default="_replay_baseline",
        help="Suffix appended to the parquet stem for the replay output. "
             "Default `_replay_baseline` -> `episode_000000_replay_baseline.parquet`.",
    )
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write the replay parquet into. Defaults to the "
             "input parquet's parent directory.",
    )
    p.add_argument(
        "--hand-input", choices=("trigger", "grip", "max"), default="max",
        help="Controller-fallback hand-input mode (only used on frames where "
             "XRHand curls are NaN). Must match the live recording's "
             "--hand-input value to reproduce the recorded commands. "
             "Default `max` matches the v4 recording.",
    )
    p.add_argument(
        "--apply-curl-compensation", action="store_true",
        help="If set, opt into stretch_finger_curls (default off; matches the "
             "live linear-mapping default).",
    )
    p.add_argument(
        "--apply-oppose-compensation", action="store_true",
        help="If set, opt into stretch_thumb_oppose (default off).",
    )
    p.add_argument(
        "--curl-floor", type=str, default=None,
        help="Per-finger raw-curl floor for normalization. Either a single "
             "scalar or 5 comma-separated floats in order "
             "[thumb,index,middle,ring,pinky]. Must be paired with "
             "--curl-ceiling (or --auto-range to derive both from the NPZ).",
    )
    p.add_argument(
        "--curl-ceiling", type=str, default=None,
        help="Per-finger raw-curl ceiling for normalization. Same format as "
             "--curl-floor.",
    )
    p.add_argument(
        "--oppose-floor", type=float, default=None,
        help="Scalar floor for thumb-opposition normalization. Pair with "
             "--oppose-ceiling.",
    )
    p.add_argument(
        "--oppose-ceiling", type=float, default=None,
        help="Scalar ceiling for thumb-opposition normalization.",
    )
    p.add_argument(
        "--auto-range", action="store_true",
        help="Estimate per-finger floor/ceiling from the NPZ raw curls "
             "(p05 / p95 over hand-mode frames, per side) instead of taking "
             "explicit --curl-floor / --curl-ceiling. Each side gets its own "
             "range. The opposition floor/ceiling come from p05 / p95 of "
             "valid oppose frames per side (clamped to >0 spread).",
    )
    p.add_argument(
        "--calibration", type=Path, default=None,
        help="Operator-calibration YAML to source the hand_range from. "
             "When set, the replay uses ``cal.hand_range.{left,right}`` "
             "(per-finger floor/ceiling and oppose floor/ceiling) -- this "
             "is the EXACT path the live retargeting takes, so the replay "
             "is bit-equivalent to a live recording. Mutually exclusive "
             "with --auto-range and --curl-floor / --curl-ceiling.",
    )

    # Per-finger noise / jitter / occlusion filter (v0.6)
    p.add_argument(
        "--apply-finger-filter",
        choices=("auto", "always", "never"),
        default="auto",
        help="Whether to apply the per-side EMA + rolling-median deadband "
             "during replay. 'auto' (default) uses pre-computed filtered "
             "channels from the NPZ if present (live runs since v0.6 "
             "persist them); falls back to applying the filter offline "
             "for older NPZs that only have raw signals. 'always' forces "
             "an offline pass even when filtered keys are present (useful "
             "for tuning a different filter cfg). 'never' disables the "
             "filter and replays directly from the raw curls/oppose "
             "channels (matches pre-v0.6 behaviour).",
    )
    return p.parse_args(argv)


def _parse_5(name: str, arg: str | None) -> np.ndarray | None:
    if arg is None:
        return None
    parts = [float(x) for x in arg.split(",")]
    if len(parts) == 1:
        return np.full(5, parts[0], dtype=np.float64)
    if len(parts) != 5:
        raise SystemExit(
            f"--{name} must be a scalar or 5 comma-separated floats; got {arg!r}"
        )
    return np.asarray(parts, dtype=np.float64)


def _estimate_range_from_npz(
    *,
    curls_arr: np.ndarray,
    oppose_arr: np.ndarray,
    label: str,
    p_low: float = 5.0,
    p_high: float = 95.0,
    min_spread: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Estimate per-finger (floor, ceiling) and oppose (floor, ceiling)
    from a single side's recorded NPZ arrays.

    Floor = ``p_low``-th percentile, ceiling = ``p_high``-th percentile,
    measured over only the frames where the input is valid (non-NaN).
    A minimum spread of ``min_spread`` is enforced so that fingers
    that didn't see a closed pose during the session don't collapse
    to a single point.
    """
    valid = ~np.isnan(curls_arr).any(axis=1)
    if valid.sum() < 50:
        raise SystemExit(
            f"--auto-range needs >=50 hand-mode frames for {label}; got "
            f"{int(valid.sum())}."
        )
    fl = np.percentile(curls_arr[valid], p_low, axis=0)
    ce = np.percentile(curls_arr[valid], p_high, axis=0)
    ce = np.maximum(ce, fl + min_spread)
    fl = np.clip(fl, 0.0, 0.999)
    ce = np.clip(ce, fl + min_spread, 1.0)
    valid_o = ~np.isnan(oppose_arr)
    if valid_o.sum() < 10:
        return fl, ce, 0.0, 1.0
    o_fl = float(np.percentile(oppose_arr[valid_o], p_low))
    o_ce = float(np.percentile(oppose_arr[valid_o], p_high))
    o_ce = max(o_ce, o_fl + min_spread)
    o_fl = float(np.clip(o_fl, 0.0, 0.999))
    o_ce = float(np.clip(o_ce, o_fl + min_spread, 1.0))
    return fl, ce, o_fl, o_ce


def _replay_hand_q(
    *,
    side: str,
    curls: np.ndarray,
    oppose: float,
    finger_tip_oppose: np.ndarray | None,
    triggers_4: np.ndarray,
    hand_input: str,
    apply_curl_compensation: bool,
    apply_oppose_compensation: bool,
    curl_floor: np.ndarray | None,
    curl_ceiling: np.ndarray | None,
    oppose_floor: float | None,
    oppose_ceiling: float | None,
) -> np.ndarray:
    """Reproduce one tick of the live retargeting dispatch.

    Mirrors ``teleop_x2_kinematic.py`` lines 715--745 (revision at the
    time of recording): use XRHand curls when available, else fall
    back to the controller analog scalar.
    """
    from gear_sonic.utils.teleop.x2_hand_retarget import (
        controller_grasp_ratio,
        grasp_command_from_ratio,
        per_finger_grasp_command_from_curls_and_oppose,
    )

    if curls is not None and not np.isnan(curls).any():
        # finger_tip_oppose is forwarded as-is when present in the NPZ
        # and finite. Old-format NPZs (no per-finger field) pass None
        # here -- the retargeter then exactly reproduces the legacy
        # curls+oppose drive (no kinematic regression).
        return per_finger_grasp_command_from_curls_and_oppose(
            side, curls,
            None if (oppose is None or np.isnan(oppose)) else float(oppose),
            finger_tip_oppose=finger_tip_oppose,
            apply_curl_compensation=apply_curl_compensation,
            apply_oppose_compensation=apply_oppose_compensation,
            curl_floor=curl_floor,
            curl_ceiling=curl_ceiling,
            oppose_floor=oppose_floor,
            oppose_ceiling=oppose_ceiling,
        )

    l_ratio, r_ratio = controller_grasp_ratio(
        left_trigger=float(triggers_4[0]),
        right_trigger=float(triggers_4[1]),
        left_grip=float(triggers_4[2]),
        right_grip=float(triggers_4[3]),
        mode=hand_input,
    )
    return grasp_command_from_ratio(side, l_ratio if side == "left" else r_ratio)


def _summarise_cmd_coverage(
    label: str,
    cmd: np.ndarray,
    raw_curls: np.ndarray,
    open_anchor: tuple[float, ...],
    closed_anchor: tuple[float, ...],
) -> None:
    """Report how close the per-motor cmd extrema land to the OPEN
    and CLOSED anchors. This metric is NOT tautological with the
    auto-range normalizer because the anchors are hardware constants
    independent of the raw-curl distribution.

    "open-extreme%" = how close the smallest commanded q got to the
    OPEN anchor, expressed as a fraction of the motor's
    OPEN→CLOSED span. 0 % = touched OPEN exactly; 100 % = stuck at
    CLOSED. Lower is better.

    "close-extreme%" = symmetric for the largest commanded q.
    Lower is better here too -- 0 % = touched CLOSED exactly.
    """
    valid = ~np.isnan(raw_curls).any(axis=1)
    if valid.sum() < 50:
        print(f"  {label}: skipped (only {int(valid.sum())} hand-mode frames)")
        return
    cm = cmd[valid]
    open_arr = np.asarray(open_anchor, dtype=np.float64)
    closed_arr = np.asarray(closed_anchor, dtype=np.float64)
    span = np.abs(closed_arr - open_arr) + 1e-9
    motor_names = (
        "thumb_roll", "thumb_abad", "thumb_mcp",
        "index_abad", "index_pip",
        "middle_pip",
        "ring_abad", "ring_pip",
        "pinky_abad", "pinky_pip",
    )
    print(f"\n  {label} (n={int(valid.sum())} hand-mode frames):")
    print(f"    motor          open-extreme%   close-extreme%")
    for m in range(10):
        sign = 1.0 if closed_arr[m] >= open_arr[m] else -1.0
        if sign > 0:
            min_q = cm[:, m].min()
            max_q = cm[:, m].max()
            open_extreme = abs(min_q - open_arr[m]) / span[m]
            close_extreme = abs(closed_arr[m] - max_q) / span[m]
        else:
            min_q = cm[:, m].max()
            max_q = cm[:, m].min()
            open_extreme = abs(min_q - open_arr[m]) / span[m]
            close_extreme = abs(closed_arr[m] - max_q) / span[m]
        print(f"    {motor_names[m]:11s}    "
              f"{open_extreme*100:6.1f}          {close_extreme*100:6.1f}")


def _summarise_anchor_saturation(
    label: str,
    cmd: np.ndarray,
    raw_curls: np.ndarray,
    open_anchor: tuple[float, ...],
    closed_anchor: tuple[float, ...],
) -> None:
    """Quantify how close commanded q lands to the OPEN anchor at
    "operator-open" frames and to the CLOSED anchor at
    "operator-closed" frames. ``raw_curls`` is the *raw* per-finger
    curl, used only to tag frames as open vs closed via per-finger
    quantiles -- the cmd vs anchor distance is the actual metric.
    """
    valid = ~np.isnan(raw_curls).any(axis=1)
    if valid.sum() < 50:
        print(f"  {label}: skipped (only {int(valid.sum())} hand-mode frames)")
        return
    raw = raw_curls[valid]
    cm = cmd[valid]
    open_arr = np.asarray(open_anchor, dtype=np.float64)
    closed_arr = np.asarray(closed_anchor, dtype=np.float64)
    span = np.abs(closed_arr - open_arr) + 1e-9
    motor_names = (
        "thumb_roll", "thumb_abad", "thumb_mcp",
        "index_abad", "index_pip",
        "middle_pip",
        "ring_abad", "ring_pip",
        "pinky_abad", "pinky_pip",
    )
    motor_finger = [0, 0, 0, 1, 1, 2, 3, 3, 4, 4]
    print(f"\n  {label} (n={int(valid.sum())} hand-mode frames):")
    print(f"    motor          open-gap%        close-gap%        span(rad)")
    for m in range(10):
        f = motor_finger[m]
        col = raw[:, f]
        lo_q = np.percentile(col, 5.0)
        hi_q = np.percentile(col, 95.0)
        lo_mask = col <= lo_q
        hi_mask = col >= hi_q
        if lo_mask.sum() < 5 or hi_mask.sum() < 5:
            continue
        # Gap to OPEN anchor at operator-open frames, normalised by motor span.
        open_gap = np.abs(cm[lo_mask, m] - open_arr[m]) / span[m]
        # Gap from CLOSED anchor at operator-closed frames, normalised similarly.
        close_gap = np.abs(cm[hi_mask, m] - closed_arr[m]) / span[m]
        print(f"    {motor_names[m]:11s}  "
              f"med={np.median(open_gap)*100:5.1f}  p95={np.percentile(open_gap,95)*100:5.1f}    "
              f"med={np.median(close_gap)*100:5.1f}  p95={np.percentile(close_gap,95)*100:5.1f}    "
              f"{span[m]:.3f}")


def _summarise_diff(label: str, recorded: np.ndarray, regen: np.ndarray) -> None:
    """Print per-column L1/Linf diff stats. Both arrays are (N, 10)."""
    diff = regen - recorded
    abs_diff = np.abs(diff)
    print(f"\n  {label} (n={len(recorded)} frames, 10 motors):")
    print(f"    overall  Linf={abs_diff.max():.3e}  "
          f"mean(|d|)={abs_diff.mean():.3e}  "
          f"exact-match-rows={int(np.all(abs_diff < 1e-9, axis=1).sum())}/{len(recorded)}")
    print(f"    motor    Linf       mean|d|    p99|d|     bias")
    motor_names = (
        "thumb_roll", "thumb_abad", "thumb_mcp",
        "index_abad", "index_pip",
        "middle_pip",
        "ring_abad", "ring_pip",
        "pinky_abad", "pinky_pip",
    )
    for m in range(10):
        col = abs_diff[:, m]
        print(f"    {motor_names[m]:11s} "
              f"{col.max():.3e}  {col.mean():.3e}  "
              f"{np.percentile(col, 99):.3e}  "
              f"{diff[:, m].mean():+.3e}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if not args.npz.is_file():
        raise SystemExit(f"NPZ not found: {args.npz}")
    if not args.parquet.is_file():
        raise SystemExit(f"Parquet not found: {args.parquet}")

    out_dir = args.output_dir if args.output_dir is not None else args.parquet.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.parquet.stem}{args.output_suffix}.parquet"
    if out_path.resolve() == args.parquet.resolve():
        raise SystemExit(
            f"Refusing to overwrite the input parquet "
            f"({out_path}). Pick a different --output-suffix or --output-dir."
        )

    print(f"[replay] NPZ:     {args.npz}")
    print(f"[replay] parquet: {args.parquet}")
    print(f"[replay] output:  {out_path}")
    print(f"[replay] mode:    apply_curl_comp={args.apply_curl_compensation} "
          f"apply_oppose_comp={args.apply_oppose_compensation} "
          f"hand_input={args.hand_input}")

    npz = np.load(args.npz, allow_pickle=True)
    n_frames = int(npz["num_frames"])
    l_curls_arr = np.asarray(npz["quest_left_hand_curls"], dtype=np.float64)
    r_curls_arr = np.asarray(npz["quest_right_hand_curls"], dtype=np.float64)
    l_oppose_arr = np.asarray(npz["quest_left_thumb_oppose"], dtype=np.float64)
    r_oppose_arr = np.asarray(npz["quest_right_thumb_oppose"], dtype=np.float64)
    triggers_arr = np.asarray(npz["controller_triggers"], dtype=np.float64)

    # Per-finger thumb-tip-to-fingertip proximity. Added in the post
    # May-2026 schema; older NPZs lack the field and we fall back to
    # the curls + thumb_oppose drive (kinematically identical to the
    # legacy live behaviour for those recordings).
    if "quest_left_finger_tip_oppose" in npz.files:
        l_tip_oppose_arr = np.asarray(
            npz["quest_left_finger_tip_oppose"], dtype=np.float64
        )
        r_tip_oppose_arr = np.asarray(
            npz["quest_right_finger_tip_oppose"], dtype=np.float64
        )
        if l_tip_oppose_arr.shape != (n_frames, 4) or r_tip_oppose_arr.shape != (
            n_frames,
            4,
        ):
            raise SystemExit(
                f"Unexpected finger_tip_oppose shapes: "
                f"left={l_tip_oppose_arr.shape} right={r_tip_oppose_arr.shape} "
                f"(expected ({n_frames}, 4))"
            )
        print(
            f"[replay] finger_tip_oppose present in NPZ "
            f"(L={int((~np.isnan(l_tip_oppose_arr).any(axis=1)).sum())}/{n_frames} valid, "
            f"R={int((~np.isnan(r_tip_oppose_arr).any(axis=1)).sum())}/{n_frames} valid)"
        )
    else:
        l_tip_oppose_arr = None
        r_tip_oppose_arr = None
        print(
            "[replay] finger_tip_oppose NOT in NPZ (pre-May-2026 schema); "
            "non-thumb pip motors will be driven on curls + thumb_oppose only."
        )

    # ── Per-finger smoothing filter ──
    # 'auto'   (default): use the v0.6 pre-computed filtered channels
    #                     from the NPZ if present, else apply offline.
    # 'always': always re-apply the filter offline (override pre-computed).
    # 'never':  use the raw signals, bypassing the filter entirely.
    has_filtered = (
        "quest_left_hand_curls_filtered" in npz.files
        and "quest_right_hand_curls_filtered" in npz.files
    )
    if args.apply_finger_filter == "never":
        filter_mode = "off"
    elif args.apply_finger_filter == "always":
        filter_mode = "offline"
    else:  # auto
        filter_mode = "live" if has_filtered else "offline"

    if filter_mode == "live":
        # The recording was made with the v0.6+ live filter; replay the
        # pre-computed filtered channels for bit-equivalent behaviour.
        l_curls_arr = np.asarray(
            npz["quest_left_hand_curls_filtered"], dtype=np.float64,
        )
        r_curls_arr = np.asarray(
            npz["quest_right_hand_curls_filtered"], dtype=np.float64,
        )
        l_oppose_arr = np.asarray(
            npz["quest_left_thumb_oppose_filtered"], dtype=np.float64,
        )
        r_oppose_arr = np.asarray(
            npz["quest_right_thumb_oppose_filtered"], dtype=np.float64,
        )
        if (
            "quest_left_finger_tip_oppose_filtered" in npz.files
            and l_tip_oppose_arr is not None
        ):
            l_tip_oppose_arr = np.asarray(
                npz["quest_left_finger_tip_oppose_filtered"], dtype=np.float64,
            )
            r_tip_oppose_arr = np.asarray(
                npz["quest_right_finger_tip_oppose_filtered"], dtype=np.float64,
            )
        print("[replay] using PRE-COMPUTED filtered signals from NPZ "
              "(auto mode; use --apply-finger-filter=always to re-filter offline)")
    elif filter_mode == "offline":
        from gear_sonic.utils.teleop.finger_signal_filter import (
            FingerFilterParams, filter_npz_offline,
        )
        params = FingerFilterParams()  # v5-calibrated defaults
        if l_tip_oppose_arr is None:
            l_tip_oppose_arr = np.full((n_frames, 4), np.nan, dtype=np.float64)
            r_tip_oppose_arr = np.full((n_frames, 4), np.nan, dtype=np.float64)
            tip_was_synthetic = True
        else:
            tip_was_synthetic = False
        l_curls_arr, l_oppose_arr, l_tip_filt = filter_npz_offline(
            l_curls_arr, l_oppose_arr, l_tip_oppose_arr, params=params,
        )
        r_curls_arr, r_oppose_arr, r_tip_filt = filter_npz_offline(
            r_curls_arr, r_oppose_arr, r_tip_oppose_arr, params=params,
        )
        if not tip_was_synthetic:
            l_tip_oppose_arr = l_tip_filt
            r_tip_oppose_arr = r_tip_filt
        else:
            # NPZ didn't carry tip_oppose; restore None so downstream
            # code falls back to the curls + thumb_oppose drive.
            l_tip_oppose_arr = None
            r_tip_oppose_arr = None
        print(
            f"[replay] applying finger filter OFFLINE: "
            f"alpha={params.ema_alpha} hold_window={params.hold_window} "
            f"hold_std={params.hold_std}"
        )
    else:  # off
        print("[replay] finger filter OFF (--apply-finger-filter=never): "
              "replaying raw Quest 3 signals as recorded.")

    cli_curl_floor = _parse_5("curl-floor", args.curl_floor)
    cli_curl_ceiling = _parse_5("curl-ceiling", args.curl_ceiling)
    if (cli_curl_floor is None) != (cli_curl_ceiling is None):
        raise SystemExit(
            "--curl-floor and --curl-ceiling must be provided together."
        )
    if (args.oppose_floor is None) != (args.oppose_ceiling is None):
        raise SystemExit(
            "--oppose-floor and --oppose-ceiling must be provided together."
        )

    if args.calibration is not None:
        if (
            args.auto_range
            or cli_curl_floor is not None
            or args.oppose_floor is not None
        ):
            raise SystemExit(
                "--calibration is mutually exclusive with --auto-range and "
                "explicit --curl-floor / --curl-ceiling / --oppose-floor / "
                "--oppose-ceiling."
            )
        from gear_sonic.utils.teleop.operator_calibration import OperatorCalibration
        cal = OperatorCalibration.load_yaml(args.calibration)
        if cal.hand_range is None:
            raise SystemExit(
                f"calibration {args.calibration} has no hand_range. Run "
                f"`gear_sonic.scripts.fit_hand_range_from_npz` first."
            )
        l_floor = cal.hand_range.left.floor
        l_ceil = cal.hand_range.left.ceiling
        l_o_fl = cal.hand_range.left.oppose_floor
        l_o_ce = cal.hand_range.left.oppose_ceiling
        r_floor = cal.hand_range.right.floor
        r_ceil = cal.hand_range.right.ceiling
        r_o_fl = cal.hand_range.right.oppose_floor
        r_o_ce = cal.hand_range.right.oppose_ceiling
        names = ("thumb", "index", "middle", "ring", "pinky")
        print(f"[replay] hand_range from {args.calibration}:")
        print(f"  source: {cal.hand_range.source}  samples: {cal.hand_range.samples}")
        for s, fl, ce in (("L", l_floor, l_ceil), ("R", r_floor, r_ceil)):
            print(f"  {s}: " + " ".join(
                f"{n}=[{fl[i]:.2f},{ce[i]:.2f}]" for i, n in enumerate(names)
            ))
        print(f"  L oppose=[{l_o_fl:.2f},{l_o_ce:.2f}]  "
              f"R oppose=[{r_o_fl:.2f},{r_o_ce:.2f}]")
    elif args.auto_range:
        if cli_curl_floor is not None or args.oppose_floor is not None:
            raise SystemExit(
                "--auto-range is mutually exclusive with explicit "
                "--curl-floor / --curl-ceiling / --oppose-floor / --oppose-ceiling."
            )
        l_floor, l_ceil, l_o_fl, l_o_ce = _estimate_range_from_npz(
            curls_arr=l_curls_arr, oppose_arr=l_oppose_arr, label="left",
        )
        r_floor, r_ceil, r_o_fl, r_o_ce = _estimate_range_from_npz(
            curls_arr=r_curls_arr, oppose_arr=r_oppose_arr, label="right",
        )
        print("[replay] auto-range (p05/p95 per side):")
        names = ("thumb", "index", "middle", "ring", "pinky")
        for s, fl, ce in (("L", l_floor, l_ceil), ("R", r_floor, r_ceil)):
            print(f"  {s}: " + " ".join(
                f"{n}=[{fl[i]:.2f},{ce[i]:.2f}]" for i, n in enumerate(names)
            ))
        print(f"  L oppose=[{l_o_fl:.2f},{l_o_ce:.2f}]  "
              f"R oppose=[{r_o_fl:.2f},{r_o_ce:.2f}]")
    elif cli_curl_floor is not None:
        l_floor = r_floor = cli_curl_floor
        l_ceil = r_ceil = cli_curl_ceiling
        l_o_fl = r_o_fl = args.oppose_floor
        l_o_ce = r_o_ce = args.oppose_ceiling
        print(f"[replay] explicit range  floor={cli_curl_floor}  ceiling={cli_curl_ceiling}")
        if args.oppose_floor is not None:
            print(f"[replay] explicit oppose range  floor={args.oppose_floor}  "
                  f"ceiling={args.oppose_ceiling}")
    else:
        l_floor = r_floor = l_ceil = r_ceil = None
        l_o_fl = r_o_fl = l_o_ce = r_o_ce = None
        print("[replay] no normalization (linear pass-through)")

    if l_curls_arr.shape != (n_frames, 5) or r_curls_arr.shape != (n_frames, 5):
        raise SystemExit(
            f"Unexpected curl array shapes: left={l_curls_arr.shape} "
            f"right={r_curls_arr.shape} (expected ({n_frames}, 5))"
        )
    if triggers_arr.shape != (n_frames, 4):
        raise SystemExit(f"Unexpected triggers shape: {triggers_arr.shape}")

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(args.parquet)
    if table.num_rows != n_frames:
        raise SystemExit(
            f"parquet rows ({table.num_rows}) != NPZ num_frames ({n_frames}); "
            f"are these from the same episode?"
        )

    print(f"[replay] regenerating {n_frames} frames ...")
    left_regen = np.zeros((n_frames, 10), dtype=np.float64)
    right_regen = np.zeros((n_frames, 10), dtype=np.float64)
    for i in range(n_frames):
        l_tip = l_tip_oppose_arr[i] if l_tip_oppose_arr is not None else None
        r_tip = r_tip_oppose_arr[i] if r_tip_oppose_arr is not None else None
        left_regen[i] = _replay_hand_q(
            side="left",
            curls=l_curls_arr[i],
            oppose=l_oppose_arr[i],
            finger_tip_oppose=l_tip,
            triggers_4=triggers_arr[i],
            hand_input=args.hand_input,
            apply_curl_compensation=args.apply_curl_compensation,
            apply_oppose_compensation=args.apply_oppose_compensation,
            curl_floor=l_floor,
            curl_ceiling=l_ceil,
            oppose_floor=l_o_fl,
            oppose_ceiling=l_o_ce,
        )
        right_regen[i] = _replay_hand_q(
            side="right",
            curls=r_curls_arr[i],
            oppose=r_oppose_arr[i],
            finger_tip_oppose=r_tip,
            triggers_4=triggers_arr[i],
            hand_input=args.hand_input,
            apply_curl_compensation=args.apply_curl_compensation,
            apply_oppose_compensation=args.apply_oppose_compensation,
            curl_floor=r_floor,
            curl_ceiling=r_ceil,
            oppose_floor=r_o_fl,
            oppose_ceiling=r_o_ce,
        )

    recorded_left = np.stack(
        [np.asarray(x, dtype=np.float64) for x in table["action.left_hand_joints"].to_pylist()]
    )
    recorded_right = np.stack(
        [np.asarray(x, dtype=np.float64) for x in table["action.right_hand_joints"].to_pylist()]
    )

    _summarise_diff("LEFT  hand q (recorded vs regenerated)", recorded_left, left_regen)
    _summarise_diff("RIGHT hand q (recorded vs regenerated)", recorded_right, right_regen)

    n_hand_mode = int((~np.isnan(l_curls_arr).any(axis=1)).sum())
    n_ctrl_mode = n_frames - n_hand_mode
    print(f"\n  frames: {n_hand_mode} hand-mode (XRHand curls), "
          f"{n_ctrl_mode} controller-only fallback")

    from gear_sonic.utils.teleop.x2_hand_retarget import (
        HAND_GRASP_CLOSED_RAD_LEFT, HAND_GRASP_CLOSED_RAD_RIGHT,
        HAND_GRASP_OPEN_RAD_LEFT, HAND_GRASP_OPEN_RAD_RIGHT,
    )
    print("\n  Cmd extreme coverage -- distance from per-motor MIN cmd to")
    print("  OPEN anchor (%span) and from MAX cmd to CLOSED anchor (%span).")
    print("  Lower = better. 0 = anchor was actually reached at least once.")
    print("\n  RECORDED (live recording):")
    _summarise_cmd_coverage(
        "LEFT  recorded", recorded_left, l_curls_arr,
        HAND_GRASP_OPEN_RAD_LEFT, HAND_GRASP_CLOSED_RAD_LEFT,
    )
    _summarise_cmd_coverage(
        "RIGHT recorded", recorded_right, r_curls_arr,
        HAND_GRASP_OPEN_RAD_RIGHT, HAND_GRASP_CLOSED_RAD_RIGHT,
    )
    print("\n  REGENERATED (this replay):")
    _summarise_cmd_coverage(
        "LEFT  regen", left_regen, l_curls_arr,
        HAND_GRASP_OPEN_RAD_LEFT, HAND_GRASP_CLOSED_RAD_LEFT,
    )
    _summarise_cmd_coverage(
        "RIGHT regen", right_regen, r_curls_arr,
        HAND_GRASP_OPEN_RAD_RIGHT, HAND_GRASP_CLOSED_RAD_RIGHT,
    )

    schema = table.schema
    list_type = pa.list_(pa.field("element", pa.float64()), 10)
    new_left_col = pa.array(left_regen.tolist(), type=list_type)
    new_right_col = pa.array(right_regen.tolist(), type=list_type)
    cols: list[Any] = []
    for name in table.column_names:
        if name == "action.left_hand_joints":
            cols.append(new_left_col)
        elif name == "action.right_hand_joints":
            cols.append(new_right_col)
        else:
            cols.append(table[name])
    out_table = pa.Table.from_arrays(cols, schema=schema)

    pq.write_table(out_table, out_path)
    print(f"\n[replay] wrote {out_path} ({out_path.stat().st_size/1024:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
