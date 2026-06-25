#!/usr/bin/env python3
"""Build combined X2 motion_lib PKL from BONES-SEED retargeted CSV subsets.

Reads CSVs from each subset directory under
  agibot-x2-references/bones-seed/retargeted/x2/<subset>/*.csv
converts every CSV to a motion_lib entry (using the existing
``convert_soma_csv_to_motion_lib`` helpers), prefixes the motion name with the
subset to keep all retarget variants distinct, and writes:

  gear_sonic/data/motions/x2_ultra_bones_seed.pkl   # combined
  gear_sonic/data/motions/x2_ultra_loco_manipulation.pkl
  gear_sonic/data/motions/x2_ultra_standing_manipulation.pkl

Run from repo root:
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/data_process/build_x2_bones_seed_motion_lib.py
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    convert_sequence,
    downsample_sequence,
    load_bones_csv,
    set_robot,
)


def _convert_one(args):
    csv_path, fps_source, fps_target, robot = args
    set_robot(robot)
    seq = load_bones_csv(csv_path)
    entry = convert_sequence(seq, fps_source)
    if fps_source != fps_target:
        entry = downsample_sequence(entry, fps_source, fps_target)
    return Path(csv_path).stem, entry


def _compute_motion_stats(entry) -> tuple[float, float]:
    """Return (mean_xy_speed_m_per_s, net_yaw_rate_rad_per_s) for a motion entry.

    Both values are computed at the entry's stamped fps, so for a halfspeed-stamped
    entry (built with --fps-source 60 --fps 30) the returned speed is HALF of the
    1.0x clip's true real-world velocity. Threshold accordingly when filtering.

    The yaw rate is the *net* yaw change (|yaw_T - yaw_0| / duration) on unwrapped
    yaw from the xyzw quaternion, not mean(|d_yaw|). This intentionally rejects
    pelvis sway during walking gait as turning intent -- a straight forward walk
    has near-zero net yaw rate even though its per-frame |d_yaw| is sizeable.
    """
    pos = entry["root_trans_offset"]  # (T, 3) float32, meters
    rot = entry["root_rot"]  # (T, 4) xyzw quaternion
    fps = float(entry.get("fps", 30))
    T = pos.shape[0]
    if T < 2:
        return 0.0, 0.0
    dxy = np.linalg.norm(pos[1:, :2] - pos[:-1, :2], axis=1)
    mean_speed = float(np.mean(dxy) * fps)
    x, y, z, w = rot[:, 0], rot[:, 1], rot[:, 2], rot[:, 3]
    yaws = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaws_unwrapped = np.unwrap(yaws)
    duration = (T - 1) / fps
    net_yaw_rate = float(abs(yaws_unwrapped[-1] - yaws_unwrapped[0]) / max(duration, 1e-6))
    return mean_speed, net_yaw_rate


def convert_subset(
    subset_dir: Path,
    fps_source: int,
    fps_target: int,
    workers: int,
    min_mean_speed: float = 0.0,
    min_mean_yaw_rate: float = 0.0,
) -> dict:
    csvs = sorted(subset_dir.glob("*.csv"))
    print(f"  {subset_dir.name}: {len(csvs)} CSVs")
    filter_active = (min_mean_speed > 0.0) or (min_mean_yaw_rate > 0.0)
    if filter_active:
        print(
            f"    filter active: keep iff mean_xy_speed > {min_mean_speed:.3f} m/s"
            f" OR net_yaw_rate > {min_mean_yaw_rate:.3f} rad/s"
        )
    out: dict[str, dict] = {}
    args = [(str(p), fps_source, fps_target, "x2_ultra") for p in csvs]
    dropped: list[tuple[str, float, float]] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_convert_one, a) for a in args]
        done = 0
        for fut in as_completed(futures):
            name, entry = fut.result()
            done += 1
            if filter_active:
                speed, yaw_rate = _compute_motion_stats(entry)
                if speed < min_mean_speed and yaw_rate < min_mean_yaw_rate:
                    dropped.append((name, speed, yaw_rate))
                    if done % 100 == 0 or done == len(args):
                        print(f"    [{subset_dir.name}] {done}/{len(args)}"
                              f" (kept {len(out)}, dropped {len(dropped)})")
                    continue
            out[name] = entry
            if done % 100 == 0 or done == len(args):
                tag = f" (kept {len(out)}, dropped {len(dropped)})" if filter_active else ""
                print(f"    [{subset_dir.name}] {done}/{len(args)}{tag}")
    if filter_active:
        print(
            f"    [{subset_dir.name}] velocity filter summary:"
            f" kept {len(out)}, dropped {len(dropped)}"
            f" ({100.0 * len(dropped) / max(len(csvs), 1):.1f}%)"
        )
        for name, s, y in dropped[:3]:
            print(f"      dropped example: {name}  speed={s:.3f} m/s  yaw_rate={y:.3f} rad/s")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--retargeted-root",
        default=str(REPO.parent / "GR00T-WholeBodyControl"
                    / "agibot-x2-references" / "bones-seed" / "retargeted" / "x2"),
        help="Parent directory containing <subset>/*.csv folders",
    )
    parser.add_argument("--out-dir", default=str(REPO / "gear_sonic" / "data" / "motions"))
    parser.add_argument("--fps-source", type=int, default=120,
                        help="BONES-SEED native frame rate (default 120)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target motion_lib frame rate (default 30)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--subsets", nargs="+",
                        default=["loco-manipulation", "standing-manipulation"])
    parser.add_argument(
        "--out-suffix",
        default="",
        help=(
            "Optional suffix appended to all output PKL filenames so a non-legacy"
            " corpus (e.g. chain_matched retargeted CSVs) does not overwrite the"
            " pinned legacy PKLs in --out-dir. Example: '--out-suffix _chain_matched'"
            " writes x2_ultra_bones_seed_chain_matched.pkl and"
            " x2_ultra_<subset>_chain_matched.pkl."
        ),
    )
    parser.add_argument(
        "--min-mean-speed",
        type=float,
        default=0.0,
        help=(
            "Drop motion entries whose mean root XY speed (m/s, computed from the"
            " *built* entry at its stamped fps) is below this threshold AND whose"
            " net yaw rate is also below --min-mean-yaw-rate. Default 0.0 = no"
            " filtering. Recommended for halfspeed builds: 0.20."
            " Threshold semantics: at --fps-source 60 (halfspeed), the entry's"
            " stamped velocity is HALF of the real 1.0x clip's velocity, so"
            " 0.20 m/s here keeps 1.0x clips with real velocity > 0.40 m/s."
        ),
    )
    parser.add_argument(
        "--min-mean-yaw-rate",
        type=float,
        default=0.0,
        help=(
            "Drop motion entries whose net yaw rate (|yaw_T - yaw_0| / duration,"
            " rad/s, from unwrapped quaternion yaw at the stamped fps) is below"
            " this AND linear speed is also below --min-mean-speed."
            " OR-filter on motion: a rotate-in-place clip with zero linear speed"
            " still passes if it has meaningful net yaw. Recommended for halfspeed"
            " builds: 0.15. Default 0.0 = no filtering."
        ),
    )
    args = parser.parse_args()

    rt = Path(args.retargeted_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Source FPS: {args.fps_source}  ->  Target FPS: {args.fps}")
    print(f"Workers: {args.workers}")
    print(f"Retargeted root: {rt}")
    print(f"Output dir: {out_dir}")

    suffix = args.out_suffix
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix
    if suffix:
        print(f"Output suffix: {suffix} (non-destructive write; legacy PKLs untouched)")

    combined: dict[str, dict] = {}
    for subset in args.subsets:
        subset_dir = rt / subset
        if not subset_dir.is_dir():
            print(f"WARNING: missing {subset_dir}, skipping")
            continue
        entries = convert_subset(
            subset_dir,
            args.fps_source,
            args.fps,
            args.workers,
            min_mean_speed=args.min_mean_speed,
            min_mean_yaw_rate=args.min_mean_yaw_rate,
        )
        per_subset_path = out_dir / f"x2_ultra_{subset.replace('-', '_')}{suffix}.pkl"
        print(f"  Saving {per_subset_path} ({len(entries)} entries)")
        joblib.dump(entries, per_subset_path, compress=3)
        # merge with subset prefix to keep retarget variants distinct
        prefix = subset.split("-")[0]  # 'loco' or 'standing'
        for k, v in entries.items():
            combined[f"{prefix}__{k}"] = v

    combined_path = out_dir / f"x2_ultra_bones_seed{suffix}.pkl"
    print(f"\nSaving combined: {combined_path} ({len(combined)} entries)")
    joblib.dump(combined, combined_path, compress=3)
    print("Done.")


if __name__ == "__main__":
    main()
