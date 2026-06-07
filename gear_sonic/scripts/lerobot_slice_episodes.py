"""Slice a LeRobot v2.1 dataset into a new dataset by episode selection.

Common uses:

* Cull a "warm-up" prefix (drop the first N episodes that are not
  representative of the final teleop quality):

    .venv/bin/python -m gear_sonic.scripts.lerobot_slice_episodes \\
        --src data/lerobot/x2_grab_a_drink \\
        --dst data/lerobot/x2_grab_a_drink_train_v1 \\
        --episodes 11-

* Drop dud episodes (one-frame X-then-Y misfires) while keeping the
  rest contiguous:

    .venv/bin/python -m gear_sonic.scripts.lerobot_slice_episodes \\
        --src data/lerobot/x2_grab_a_drink \\
        --dst data/lerobot/x2_grab_a_drink_clean \\
        --exclude 5

* Cherry-pick a hand-picked subset:

    .venv/bin/python -m gear_sonic.scripts.lerobot_slice_episodes \\
        --src data/lerobot/x2_pick_place_apple_v1 \\
        --dst data/lerobot/x2_pick_place_apple_top5 \\
        --episodes 0,2,3,5,7

The output is a self-contained LeRobot v2.1 dataset:

* Renumbers ``episode_index`` so the kept episodes become 0..K-1.
* Recomputes the cumulative ``index`` column inside each parquet so
  it starts at 0 in the first kept frame and runs contiguously.
* Copies every camera's MP4 with the new episode index.
* Rewrites ``meta/info.json``, ``meta/episodes.jsonl`` and
  ``meta/episodes_stats.jsonl`` to match the slice (and recomputes
  ``splits`` accordingly).
* Copies ``meta/tasks.jsonl``, ``meta/modality.json`` and
  ``meta/dataset_format_version.json`` verbatim -- they're not
  episode-indexed.

The script never mutates the source dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _parse_episode_spec(spec: str | None, total_episodes: int) -> list[int]:
    """Resolve ``--episodes`` to a sorted list of source episode indices.

    Accepted forms (mixable with commas):

    * ``11-30``  -> ``[11, 12, ..., 30]``
    * ``11-``    -> ``[11, 12, ..., total-1]``
    * ``-5``     -> ``[0, 1, ..., 5]``
    * ``11,15,22`` -> ``[11, 15, 22]``
    * Mix of the above: ``0,3-5,10-`` -> ``[0, 3, 4, 5, 10, ..., total-1]``

    None / empty means "all source episodes".
    """
    if not spec:
        return list(range(total_episodes))
    out: set[int] = set()
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            lo, hi = piece.split("-", 1)
            lo_i = int(lo) if lo else 0
            hi_i = int(hi) if hi else total_episodes - 1
            if lo_i < 0 or hi_i >= total_episodes or lo_i > hi_i:
                raise SystemExit(
                    f"Error: episode range {piece!r} out of bounds "
                    f"(source has {total_episodes} episodes, indices "
                    f"0..{total_episodes - 1})."
                )
            out.update(range(lo_i, hi_i + 1))
        else:
            idx = int(piece)
            if idx < 0 or idx >= total_episodes:
                raise SystemExit(
                    f"Error: episode {idx} out of bounds "
                    f"(source has {total_episodes} episodes)."
                )
            out.add(idx)
    return sorted(out)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--src", type=Path, required=True,
        help="Source LeRobot v2.1 dataset root.",
    )
    p.add_argument(
        "--dst", type=Path, required=True,
        help="Destination dataset root (must not exist unless --force).",
    )
    p.add_argument(
        "--episodes", type=str, default=None,
        help="Source episode selector (e.g. ``11-30``, ``0,3-5,10-``). "
        "Defaults to all episodes.",
    )
    p.add_argument(
        "--exclude", type=str, default=None,
        help="Source episodes to drop after applying --episodes "
        "(e.g. ``5`` to drop the dud, ``5,17`` for several).",
    )
    p.add_argument(
        "--cameras", type=str, default=None,
        help="Comma-separated list of camera stems to KEEP "
        "(e.g. ``stereo_left,stereo_right``). Defaults to all "
        "source cameras. The stem is the last dotted segment of "
        "the feature name, so ``observation.images.stereo_left`` "
        "is selected as ``stereo_left``. Cameras not listed are "
        "fully dropped from the destination: no MP4s copied, no "
        "entries in ``meta/info.json#features`` or "
        "``meta/modality.json#video``. The source MP4s are NEVER "
        "touched -- this only changes what's copied to --dst.",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite --dst if it already exists.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan (kept episodes, frame counts, output paths) "
        "and exit without copying anything.",
    )
    return p.parse_args(argv)


def _camera_keys(info: dict) -> list[str]:
    return [
        k for k, v in info.get("features", {}).items()
        if v.get("dtype") == "video"
    ]


def _load_episodes_jsonl(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            out[int(ep["episode_index"])] = ep
    return out


def _load_episodes_stats_jsonl(path: Path) -> dict[int, dict]:
    if not path.is_file():
        return {}
    out: dict[int, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[int(d["episode_index"])] = d
    return out


def _rewrite_parquet(
    src_pq: Path,
    dst_pq: Path,
    new_episode_index: int,
    new_global_index_start: int,
) -> int:
    """Copy a parquet with rewritten episode_index + index columns.

    Returns the number of rows written (== episode length).
    """
    table = pq.read_table(src_pq)
    n = table.num_rows

    # Rewrite ``episode_index`` to the new dataset's index.
    ep_arr = pa.array(np.full(n, new_episode_index, dtype=np.int64))
    # Rewrite ``index`` to the new dataset's cumulative global frame
    # index. ``frame_index`` (per-episode) is preserved verbatim.
    idx_arr = pa.array(
        np.arange(new_global_index_start, new_global_index_start + n,
                  dtype=np.int64)
    )

    schema = table.schema
    cols = []
    for name in table.column_names:
        if name == "episode_index":
            cols.append(ep_arr)
        elif name == "index":
            cols.append(idx_arr)
        else:
            cols.append(table.column(name))
    new_table = pa.Table.from_arrays(cols, schema=schema)
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_pq)
    return n


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()

    if not src.is_dir():
        print(f"Error: --src {src} is not a directory.", file=sys.stderr)
        return 2

    src_info_path = src / "meta" / "info.json"
    if not src_info_path.is_file():
        print(f"Error: {src_info_path} not found -- source is not a "
              f"LeRobot v2.1 dataset.", file=sys.stderr)
        return 2

    src_info = json.loads(src_info_path.read_text())
    total_src_eps = int(src_info["total_episodes"])
    fps = float(src_info.get("fps", 50))
    chunks_size = int(src_info.get("chunks_size", 1000))
    data_path_pattern = src_info["data_path"]
    video_path_pattern = src_info["video_path"]
    all_src_cams = _camera_keys(src_info)

    # ── Resolve camera selection ─────────────────────────────────────
    # ``cams`` is the set of camera *feature keys* (e.g.
    # ``observation.images.stereo_left``) we'll carry across.
    # ``dropped_cams`` is the complement -- used to scrub
    # ``info.json#features`` and ``modality.json#video`` so the
    # destination's metadata never references cameras whose MP4s
    # don't exist.
    if args.cameras:
        wanted_stems = {s.strip() for s in args.cameras.split(",") if s.strip()}
        cams = [c for c in all_src_cams if c.split(".")[-1] in wanted_stems]
        unknown = wanted_stems - {c.split(".")[-1] for c in all_src_cams}
        if unknown:
            print(
                f"Error: --cameras references unknown stem(s) "
                f"{sorted(unknown)}. Source has: "
                f"{[c.split('.')[-1] for c in all_src_cams]}",
                file=sys.stderr,
            )
            return 2
        if not cams:
            print("Error: --cameras filtered everything out.",
                  file=sys.stderr)
            return 2
    else:
        cams = list(all_src_cams)
    dropped_cams = [c for c in all_src_cams if c not in cams]

    # ── Resolve selection ────────────────────────────────────────────
    selected = _parse_episode_spec(args.episodes, total_src_eps)
    excluded = set(_parse_episode_spec(args.exclude, total_src_eps)) \
        if args.exclude else set()
    keep = [i for i in selected if i not in excluded]
    if not keep:
        print("Error: selection is empty (nothing to slice).",
              file=sys.stderr)
        return 2

    # ── Resolve metadata ─────────────────────────────────────────────
    src_episodes = _load_episodes_jsonl(src / "meta" / "episodes.jsonl")
    src_stats = _load_episodes_stats_jsonl(
        src / "meta" / "episodes_stats.jsonl"
    )

    plan: list[tuple[int, int, int, int]] = []  # (old_idx, new_idx, n_frames, new_global_start)
    running_global = 0
    for new_idx, old_idx in enumerate(keep):
        if old_idx not in src_episodes:
            print(f"Error: source episodes.jsonl missing index "
                  f"{old_idx}", file=sys.stderr)
            return 2
        n = int(src_episodes[old_idx]["length"])
        plan.append((old_idx, new_idx, n, running_global))
        running_global += n
    new_total_frames = running_global
    new_total_episodes = len(plan)

    # ── Dry-run preview ──────────────────────────────────────────────
    print(f"[slice] SOURCE IS READ-ONLY: no original parquets or MP4s "
          f"will be modified or deleted.")
    print(f"[slice] src={src.name}  ({total_src_eps} episodes, "
          f"{int(src_info.get('total_frames', 0))} frames)")
    print(f"[slice] dst={dst.name}  ({new_total_episodes} episodes, "
          f"{new_total_frames} frames, {new_total_frames / fps / 60:.2f} min)")
    print(f"[slice] cameras kept: {[c.split('.')[-1] for c in cams]}")
    if dropped_cams:
        print(f"[slice] cameras dropped: "
              f"{[c.split('.')[-1] for c in dropped_cams]} "
              f"(MP4s and feature entries excluded from --dst; source intact)")
    print(f"[slice] selected (after --episodes/--exclude):")
    for old_idx, new_idx, n, _start in plan[:5]:
        print(f"    src ep {old_idx:>3} -> new ep {new_idx:>3}  "
              f"({n:>6} frames)")
    if len(plan) > 6:
        print(f"    ... {len(plan) - 6} more ...")
    if len(plan) > 5:
        old_idx, new_idx, n, _start = plan[-1]
        print(f"    src ep {old_idx:>3} -> new ep {new_idx:>3}  "
              f"({n:>6} frames)")

    if args.dry_run:
        print("[slice] --dry-run set; not copying anything.")
        return 0

    # ── Output dir handling ─────────────────────────────────────────
    if dst.exists():
        if not args.force:
            print(f"Error: --dst {dst} already exists. Re-run with "
                  f"--force to overwrite (will rm -rf and rebuild).",
                  file=sys.stderr)
            return 2
        print(f"[slice] --force: removing existing {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # ── Copy parquet + video files with renumbering ─────────────────
    for old_idx, new_idx, n, new_global_start in plan:
        old_chunk = old_idx // chunks_size
        new_chunk = new_idx // chunks_size

        src_pq = src / data_path_pattern.format(
            episode_chunk=old_chunk, episode_index=old_idx,
        )
        dst_pq = dst / data_path_pattern.format(
            episode_chunk=new_chunk, episode_index=new_idx,
        )
        written = _rewrite_parquet(
            src_pq, dst_pq,
            new_episode_index=new_idx,
            new_global_index_start=new_global_start,
        )
        if written != n:
            print(f"[slice] WARN: ep {old_idx} parquet has {written} "
                  f"rows but episodes.jsonl claims {n}. Keeping "
                  f"{written}.")

        for cam in cams:
            src_mp4 = src / video_path_pattern.format(
                episode_chunk=old_chunk, episode_index=old_idx,
                video_key=cam,
            )
            dst_mp4 = dst / video_path_pattern.format(
                episode_chunk=new_chunk, episode_index=new_idx,
                video_key=cam,
            )
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            if src_mp4.is_file():
                shutil.copy2(src_mp4, dst_mp4)
            else:
                print(f"[slice] WARN: src video missing: "
                      f"{src_mp4.relative_to(src)}")

        print(f"[slice]   wrote new ep {new_idx:>3} "
              f"(was src ep {old_idx:>3}, {n} frames, "
              f"global index {new_global_start}..{new_global_start + n - 1})")

    # ── Write new meta files ────────────────────────────────────────
    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # info.json: same as source but with new totals + splits.
    # If --cameras filtered any cameras out, also scrub the matching
    # entries from ``features`` so consumers (LeRobot loader, GR00T
    # data exporter, etc.) don't try to open MP4s we never copied.
    new_info = dict(src_info)
    new_info["total_episodes"] = new_total_episodes
    new_info["total_frames"] = new_total_frames
    new_info["total_chunks"] = (
        (new_total_episodes + chunks_size - 1) // chunks_size
    )
    new_info["splits"] = {"train": f"0:{new_total_episodes}"}
    if dropped_cams:
        new_features = {
            k: v for k, v in new_info.get("features", {}).items()
            if k not in set(dropped_cams)
        }
        new_info["features"] = new_features
    if "total_videos" in new_info:
        new_info["total_videos"] = new_total_episodes * len(cams)
    (dst_meta / "info.json").write_text(json.dumps(new_info, indent=2))

    # episodes.jsonl: renumber + drop skipped.
    old_to_new = {old: new for old, new, _, _ in plan}
    with open(dst_meta / "episodes.jsonl", "w") as f:
        for old_idx, new_idx, n, _start in plan:
            ep = dict(src_episodes[old_idx])
            ep["episode_index"] = new_idx
            f.write(json.dumps(ep) + "\n")

    # episodes_stats.jsonl: same renumber; per-episode payload is
    # otherwise unchanged.
    if src_stats:
        with open(dst_meta / "episodes_stats.jsonl", "w") as f:
            missing = 0
            for old_idx, new_idx, _n, _start in plan:
                if old_idx not in src_stats:
                    missing += 1
                    continue
                d = dict(src_stats[old_idx])
                d["episode_index"] = new_idx
                f.write(json.dumps(d) + "\n")
            if missing:
                print(f"[slice] WARN: {missing} kept episodes had no "
                      f"entry in episodes_stats.jsonl.")

    # Verbatim copies for episode-agnostic + camera-agnostic metadata.
    for fname in ("tasks.jsonl", "dataset_format_version.json"):
        src_f = src / "meta" / fname
        if src_f.is_file():
            shutil.copy2(src_f, dst_meta / fname)

    # modality.json: GR00T-specific schema. The ``video`` block
    # references camera stems (e.g. ``stereo_left``) so we have to
    # filter it if --cameras dropped anything. State / action /
    # annotation blocks are camera-agnostic and copied verbatim.
    src_mod_path = src / "meta" / "modality.json"
    if src_mod_path.is_file():
        modality = json.loads(src_mod_path.read_text())
        if dropped_cams and isinstance(modality.get("video"), dict):
            dropped_stems = {c.split(".")[-1] for c in dropped_cams}
            modality["video"] = {
                k: v for k, v in modality["video"].items()
                if k not in dropped_stems
            }
        (dst_meta / "modality.json").write_text(
            json.dumps(modality, indent=4)
        )

    # ── Final summary + sanity check ────────────────────────────────
    written_frames = sum(n for _, _, n, _ in plan)
    print()
    print(f"[slice] done.")
    print(f"[slice]   dst:          {dst}")
    print(f"[slice]   episodes:     {new_total_episodes} "
          f"(src had {total_src_eps})")
    print(f"[slice]   frames:       {written_frames}")
    print(f"[slice]   wall time:    {written_frames / fps / 60:.2f} min @ {fps:.0f} fps")
    print(f"[slice]   cameras kept: {[c.split('.')[-1] for c in cams]}")
    if dropped_cams:
        print(f"[slice]   cameras dropped from --dst: "
              f"{[c.split('.')[-1] for c in dropped_cams]}")
    print(f"[slice]   source dataset {src.name}: UNTOUCHED "
          f"(read-only, no MP4s removed)")
    print()
    print(f"[slice] view a sample:")
    print(f"    ./gear_sonic/scripts/view_x2_recorded_dataset.sh "
          f"--root {dst} --episode 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
