"""Fast Rerun viewer for a recorded X2 LeRobot v2.1 dataset.

Streams the per-episode MP4 files directly to the Rerun viewer via
``rr.AssetVideo`` + ``rr.VideoFrameReference`` so the viewer's GPU
decoder plays the H.264 tracks natively -- no Python-side
``av.open`` -> numpy -> ``rr.Image`` loop, no ~3 s/batch decode
penalty.

Why a dedicated venv?
=====================

This script targets ``rerun-sdk >= 0.30`` (matches the conda-shipped
``rerun-cli`` 0.31.4 viewer and adds native H.264 decode in the
desktop viewer). ``rerun-sdk >= 0.30`` hard-requires ``numpy >= 2``,
which is incompatible with the ``pin`` (pinocchio) 2.7.x cmeel wheel
used by the kinematic planner in the main ``.venv/``. To keep the
planner stack untouched, the viewer lives in its own venv:

    .venv-viewer/              ← rerun-sdk 0.31.4 + numpy>=2 + pyarrow + pillow
    .venv/                     ← planner: pinocchio 2.7.0 + numpy 1.26.4 + rerun-sdk 0.21 (unused here)

Use the wrapper ``gear_sonic/scripts/view_x2_recorded_dataset.sh``
which already points at the correct interpreter, or invoke directly:

    .venv-viewer/bin/python -m gear_sonic.scripts.view_x2_recorded_dataset \\
        --dataset x2_grab_a_drink --episode 6

Running this script from the main ``.venv/`` will fail fast with a
clear error pointing at the wrapper -- see ``_check_rerun_version``.

Why not just call ``lerobot.scripts.visualize_dataset``?
========================================================

The upstream viewer was written before ``rr.AssetVideo`` existed and
uses ``rr.Image`` per-frame. On a 1300-frame episode with 4 camera
tracks that's ~2.5 minutes of cold-load before any pixel appears
in the viewer. It also rebuilds the video frames via
``torch.utils.data.DataLoader``, which on this laptop crashes
because of an unrelated torchcodec ABI mismatch
(libavutil.so.{56,57,59} not present; system has ``.so.58``).

What this script does instead
=============================

1. Read the LeRobot ``meta/info.json`` + ``meta/episodes.jsonl``
   directly to enumerate the episode's camera tracks and length.
2. Read the episode parquet via ``pyarrow`` -- pure I/O, no
   torch / video decode.
3. ``rr.log(camera_key, rr.AssetVideo(path=...), static=True)``
   for each camera. The viewer reads the MP4 itself.
4. Columnar ``rr.send_columns`` to drive every camera's
   ``VideoFrameReference`` playhead AND every scalar series in a
   handful of batched sends (one per camera + one per scalar
   series) instead of millions of per-frame ``rr.log`` calls --
   that's the fix for the broken-pipe / "Dropping messages"
   cascade we used to hit on 8K-frame episodes.
5. Per-dim scalar expansion for vector-valued columns
   (``action.body_q_mj``, etc.) so each joint is independently
   plottable. Capped by ``--max-scalar-dims`` (default 64).

Caveats
-------

* Rerun bumps the wire format between releases. The 0.31.4 conda
  viewer cannot read 0.21 SDK output and vice-versa. If you see
  "Invalid encoding options" or "buffered_client: Broken pipe"
  warnings, you're probably running this script from the wrong
  venv; use the wrapper or ``.venv-viewer/bin/python`` directly.
* ``rr.AssetVideo`` for H.264 requires desktop viewer ``>= 0.30``;
  older viewers will render blank camera panes silently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rerun as rr


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _check_rerun_version() -> None:
    """Bail with a helpful error if rerun-sdk is too old.

    This script uses ``rr.TimeColumn(name, sequence=...)``,
    ``rr.Scalars.columns(...)``, and the ``indexes=/columns=``
    keyword form of ``rr.send_columns`` -- all introduced in
    rerun-sdk 0.30 (and renamed/cleaned up in 0.31). The 0.21
    pin in the main ``.venv/`` uses ``TimeSequenceColumn`` +
    ``ScalarBatch`` + ``times=/components=`` which is wire-
    incompatible with the conda 0.31.4 viewer.

    Rather than maintain two API paths in this file, we hard-
    fail with a clear pointer at the wrapper script. The wrapper
    invokes ``.venv-viewer/bin/python``, where rerun-sdk 0.31.4
    and numpy 2 are installed without touching the planner venv.
    """
    try:
        parts = rr.__version__.split(".")
        major = int(parts[0])
        minor = int(parts[1])
    except (ValueError, IndexError):
        # Unrecognised version string -- assume it's recent enough
        # rather than block on a parsing edge case.
        return
    if (major, minor) < (0, 30):
        print(
            f"Error: rerun-sdk {rr.__version__} is too old for this "
            f"viewer (need >= 0.30 for the columnar API + H.264 decode "
            f"in the desktop viewer).",
            file=sys.stderr,
        )
        print(
            "\nThis script intentionally lives in a separate venv "
            "(``.venv-viewer/``) so the upgrade doesn't disturb the "
            "planner's pinned pinocchio + numpy<2 stack.",
            file=sys.stderr,
        )
        print(
            "\nRun it via the wrapper:",
            file=sys.stderr,
        )
        print(
            "    ./gear_sonic/scripts/view_x2_recorded_dataset.sh "
            "--dataset <name> --episode <n>",
            file=sys.stderr,
        )
        print(
            "\nor directly:",
            file=sys.stderr,
        )
        print(
            "    .venv-viewer/bin/python -m "
            "gear_sonic.scripts.view_x2_recorded_dataset "
            "--dataset <name> --episode <n>",
            file=sys.stderr,
        )
        print(
            "\nIf ``.venv-viewer/`` is missing, recreate with the "
            "idempotent installer:",
            file=sys.stderr,
        )
        print(
            "    bash install_scripts/install_viewer.sh",
            file=sys.stderr,
        )
        print(
            "\n(Pinned dependencies live in ``requirements-viewer.txt`` "
            "at the repo root.)",
            file=sys.stderr,
        )
        sys.exit(2)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset name under ``data/lerobot/`` (e.g. "
        "``x2_grab_a_drink``). Mutually exclusive with ``--root``.",
    )
    grp.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Absolute path to a LeRobot v2.1 dataset root.",
    )
    p.add_argument(
        "--episode",
        "--episode-index",
        dest="episode_index",
        type=int,
        required=True,
        help="Episode index to visualize.",
    )
    p.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Label for the Rerun recording. Defaults to "
        "``local/<dataset-stem>``.",
    )
    p.add_argument(
        "--save",
        type=Path,
        default=None,
        help="When set, write a ``.rrd`` file at this path instead of "
        "spawning a viewer. Replay later with ``rerun <path>.rrd``.",
    )
    p.add_argument(
        "--max-scalar-dims",
        type=int,
        default=64,
        help="Per-column cap on how many scalar dims to expand for a "
        "vector column. Higher = more plots, slower viewer. 64 covers "
        "the full body_q_mj (30) + both hands (10+10) + projected "
        "gravity (3) comfortably. Set to 0 to skip scalars entirely.",
    )
    p.add_argument(
        "--skip-scalars",
        action="store_true",
        help="Don't log any scalar timeseries; videos only. Fastest "
        "load when you just want a visual eyeball pass.",
    )
    p.add_argument(
        "--scalar-decimate",
        type=int,
        default=1,
        metavar="N",
        help="Subsample scalar series to every Nth frame "
        "(videos always log every frame). Defaults to 1 = no "
        "subsampling. Bump to 5 or 10 for very long episodes "
        "(>30 k frames) where the viewer chart panes get sluggish.",
    )
    p.add_argument(
        "--drain-sec",
        type=float,
        default=1.0,
        help="Seconds to sleep after the last batched send before "
        "the script exits, so the rerun TCP buffer has time to "
        "flush into the spawned viewer. Ignored when --save is "
        "set (the .rrd sink writes synchronously).",
    )
    return p.parse_args(argv)


def _load_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(
            f"Error: {info_path} not found -- {root} is not a LeRobot "
            f"v2.1 dataset (or recording crashed before meta/info.json "
            f"was written)."
        )
    with open(info_path, "r") as f:
        return json.load(f)


def _episode_length(root: Path, episode_index: int) -> int:
    episodes_path = root / "meta" / "episodes.jsonl"
    with open(episodes_path, "r") as f:
        for line in f:
            ep = json.loads(line)
            if ep["episode_index"] == episode_index:
                return int(ep["length"])
    raise SystemExit(
        f"Error: --episode {episode_index} not found in "
        f"{episodes_path}."
    )


def _camera_keys(info: dict) -> list[str]:
    cams: list[str] = []
    for key, ft in info.get("features", {}).items():
        if ft.get("dtype") == "video":
            cams.append(key)
    return cams


def _resolve_episode_paths(
    root: Path,
    info: dict,
    episode_index: int,
    camera_keys: list[str],
) -> tuple[Path, dict[str, Path]]:
    chunks_size = info["chunks_size"]
    chunk_idx = episode_index // chunks_size
    data_path_pattern = info["data_path"]
    video_path_pattern = info["video_path"]

    parquet_path = root / data_path_pattern.format(
        episode_chunk=chunk_idx, episode_index=episode_index
    )
    video_paths: dict[str, Path] = {}
    for key in camera_keys:
        video_paths[key] = root / video_path_pattern.format(
            episode_chunk=chunk_idx,
            episode_index=episode_index,
            video_key=key,
        )
    return parquet_path, video_paths


def _expand_scalar_columns(
    table: "pq.Table",
    columns: list[str],
    max_dims: int,
) -> dict[str, np.ndarray]:
    """Flatten vector parquet columns into (col, dim) -> 1-D series.

    LeRobot v2.1 stores high-dim joint vectors as ``list<float>``
    columns; one row per frame, each cell is the full vector for
    that frame. We emit one (frame_count,) numpy array per scalar
    series so we can log them as rerun ``Scalars`` over time.
    """
    out: dict[str, np.ndarray] = {}
    for col in columns:
        if col not in table.column_names:
            continue
        arr = table.column(col).to_pylist()
        if not arr:
            continue
        first = arr[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            stacked = np.asarray(arr, dtype=np.float64)
            if stacked.ndim == 1:
                out[col] = stacked
                continue
            n_dims = stacked.shape[1]
            keep = min(n_dims, max_dims)
            for d in range(keep):
                out[f"{col}/{d:02d}"] = stacked[:, d]
            if n_dims > max_dims:
                print(
                    f"[view] WARN: column {col!r} has {n_dims} dims; "
                    f"capping at --max-scalar-dims={max_dims}. Use "
                    f"--skip-scalars or raise the cap to see more.",
                    flush=True,
                )
        elif isinstance(first, (int, float, np.floating)):
            out[col] = np.asarray(arr, dtype=np.float64)
        else:
            try:
                out[col] = np.asarray(arr, dtype=np.float64)
            except Exception:
                pass
    return out


def main(argv: list[str] | None = None) -> int:
    _check_rerun_version()
    args = _parse_args(argv)

    # ── Resolve dataset root ──────────────────────────────────────────
    if args.root is not None:
        root = args.root.resolve()
        if not root.is_dir():
            print(f"Error: --root {root} is not a directory", file=sys.stderr)
            return 2
        dataset_stem = root.name
    else:
        root = (REPO_ROOT / "data" / "lerobot" / args.dataset).resolve()
        if not root.is_dir():
            print(
                f"Error: --dataset {args.dataset!r} resolved to {root} "
                f"which does not exist.",
                file=sys.stderr,
            )
            return 2
        dataset_stem = args.dataset

    repo_id = args.repo_id or f"local/{dataset_stem}"

    info = _load_info(root)
    cams = _camera_keys(info)
    ep_len = _episode_length(root, args.episode_index)
    parquet_path, video_paths = _resolve_episode_paths(
        root, info, args.episode_index, cams
    )

    print(
        f"[view] dataset={root.name} episode={args.episode_index} "
        f"length={ep_len} cameras={len(cams)} "
        f"(rerun-sdk {rr.__version__})",
        flush=True,
    )
    for k, p in video_paths.items():
        if not p.is_file():
            print(
                f"[view] WARN: video for {k} not found at {p}; skipping.",
                flush=True,
            )
            continue
        size_mb = p.stat().st_size / 1024 / 1024
        print(f"[view]   {k}: {p.name} ({size_mb:.1f} MB)", flush=True)

    # ── Spawn viewer / open .rrd sink ────────────────────────────────
    recording_id = f"{repo_id}/episode_{args.episode_index}"
    if args.save is not None:
        rr.init(recording_id, spawn=False)
        save_path = args.save.resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        rr.save(str(save_path))
        print(f"[view] writing .rrd to {save_path}", flush=True)
    else:
        # We deliberately let ``rr.init(spawn=True)`` pick whichever
        # ``rerun`` binary is first on PATH. The conda-shipped 0.31.4
        # binary is the expected target (matches rerun-sdk 0.31 wire
        # format and decodes H.264 natively). The ``rerun-sdk`` wheel
        # also installs a matching CLI at ``<venv>/bin/rerun`` as a
        # belt-and-suspenders fallback if conda's binary is ever
        # uninstalled or out of date.
        rr.init(recording_id, spawn=True)
        print("[view] rerun viewer spawned", flush=True)

    # ── 1) Log each video as a static asset ──────────────────────────
    fps = float(info.get("fps", 50))
    nanos_per_frame = int(round(1e9 / fps))
    for key, vpath in video_paths.items():
        if not vpath.is_file():
            continue
        rr.log(key, rr.AssetVideo(path=str(vpath)), static=True)

    # ── 2) Build columnar batches (NOT per-row rr.log) ────────────────
    #
    # Why columnar?
    # -------------
    # The naive ``for i in range(n_frames): rr.log(scalar, ...)`` path
    # makes ``n_frames * len(scalar_series)`` python->TCP send calls.
    # On a 7,897-frame episode with ~221 scalar series (full body_q_mj
    # + omnihand joints + projected gravity + action.* mirror), that's
    # ~1.7M individual sends. The TCP buffer overflows somewhere
    # around the first second, the connection breaks
    # (``Broken pipe (os error 32)``), and the loop keeps hammering
    # the dead socket.
    #
    # ``rr.send_columns`` with the ``.columns(...)`` archetype helper
    # collapses that to one batched send per camera + one per scalar
    # series (~225 total for a 4-cam recording with the full state
    # vector). Arrow-native end-to-end.
    print(f"[view] loading parquet {parquet_path.name} …", flush=True)
    table = pq.read_table(parquet_path)
    print(
        f"[view] parquet loaded: {table.num_rows} rows, "
        f"{len(table.column_names)} columns",
        flush=True,
    )

    scalar_series: dict[str, np.ndarray] = {}
    if not args.skip_scalars and args.max_scalar_dims > 0:
        scalar_columns = [
            c
            for c in table.column_names
            if c.startswith("observation.") or c.startswith("action.")
        ]
        scalar_series = _expand_scalar_columns(
            table, scalar_columns, args.max_scalar_dims
        )
        print(
            f"[view] expanded {len(scalar_series)} scalar series from "
            f"{len(scalar_columns)} columns",
            flush=True,
        )

    timestamps = table.column("timestamp").to_numpy().astype(np.float64)
    frame_indices = table.column("frame_index").to_numpy().astype(np.int64)
    n_frames = int(len(frame_indices))

    if n_frames == 0:
        print(
            "[view] WARN: parquet has 0 rows; nothing to stream. "
            "This is usually a recording that was opened and "
            "immediately discarded (X-then-Y in the same tick).",
            flush=True,
        )
        if args.save is None:
            print(
                "[view] viewer is still open; close it manually when done.",
                flush=True,
            )
        return 0

    if n_frames == 1:
        # Not an error -- but warn loudly. The Jun-6 10:41 dud episode
        # 5 in x2_grab_a_drink is one of these: 1 row in parquet,
        # ~6 KB mp4 (effectively just the H.264 SPS/PPS + one I-frame
        # with no real signal). The viewer "loads" instantly but
        # there's nothing to scrub through.
        print(
            "[view] WARN: episode has length=1 -- nothing meaningful "
            "to scrub. Check ``meta/episodes.jsonl`` for episode "
            f"{args.episode_index!r}. Likely a misfired X/Y press; "
            "consider deleting it before training.",
            flush=True,
        )

    # ── 2a) Drive video playhead per camera (one send per camera) ─────
    print(
        f"[view] streaming {n_frames} frames via columnar batches "
        f"({len(video_paths)} cams + {len(scalar_series)} scalar series)…",
        flush=True,
    )
    time_cols = [
        rr.TimeColumn("frame_index", sequence=frame_indices),
        rr.TimeColumn("timestamp", duration=timestamps),
    ]

    # Per-frame nanos for VideoFrameReference; use the parquet's
    # frame_index (not loop index) so that any future support for
    # decimated parquets still resolves to the right MP4 PTS.
    ts_ns_array = (frame_indices.astype(np.int64) * nanos_per_frame).tolist()
    for key, vpath in video_paths.items():
        if not vpath.is_file():
            continue
        rr.send_columns(
            key,
            indexes=time_cols,
            columns=rr.VideoFrameReference.columns(timestamp=ts_ns_array),
        )

    # ── 2b) Emit scalar series (one send per series) ──────────────────
    decim = max(1, int(args.scalar_decimate))
    if decim > 1:
        idx_subset = np.arange(0, n_frames, decim, dtype=np.int64)
        time_cols_scalars = [
            rr.TimeColumn("frame_index", sequence=frame_indices[idx_subset]),
            rr.TimeColumn("timestamp", duration=timestamps[idx_subset]),
        ]
        print(
            f"[view] scalar decimation: every {decim}th frame "
            f"({len(idx_subset)}/{n_frames} samples per series)",
            flush=True,
        )
    else:
        idx_subset = None
        time_cols_scalars = time_cols

    for series_key, series_values in scalar_series.items():
        values = np.asarray(series_values, dtype=np.float64)
        if values.shape != (n_frames,):
            print(
                f"[view] WARN: series {series_key!r} has shape "
                f"{values.shape}, expected ({n_frames},); skipping.",
                flush=True,
            )
            continue
        if idx_subset is not None:
            values = values[idx_subset]
        rr.send_columns(
            series_key,
            indexes=time_cols_scalars,
            columns=rr.Scalars.columns(scalars=values),
        )

    # Drain pending TCP sends before the script exits.
    if args.save is None and args.drain_sec > 0:
        time.sleep(args.drain_sec)

    print(
        f"[view] done. {n_frames} frames logged "
        f"(scalar series={len(scalar_series)}, "
        f"cams={sum(1 for p in video_paths.values() if p.is_file())}).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
