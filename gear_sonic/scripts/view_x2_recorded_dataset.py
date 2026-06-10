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
import rerun.blueprint as rrb


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
    p.add_argument(
        "--ee-trace",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run forward kinematics on the canonical x2_ultra MJCF for "
        "each frame and log the LEFT + RIGHT wrist 3-D positions "
        "(pelvis-relative). Adds:\n"
        "  * 3 scalar series per side (``tcp/<side>_wrist/x|y|z``)\n"
        "  * 1 animated 3-D point per side (``tcp/<side>_wrist``)\n"
        "  * 1 static line trail per side (``tcp/<side>_wrist_path``)\n"
        "This is the real metric for grasping-task progress. The 31-dim "
        "wire (``action.body_q_mj``) is the source; root is held at "
        "identity so the path is purely arm-pose driven (invariant to "
        "walking). Pass --no-ee-trace to skip.",
    )
    p.add_argument(
        "--ee-mjcf",
        type=Path,
        default=None,
        help="Override the MJCF used for end-effector FK. Defaults to "
        "``gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml``. "
        "The MJCF must expose the 31 X2 body joints by their canonical "
        "names plus the ``left_wrist_roll_link`` and "
        "``right_wrist_roll_link`` bodies.",
    )
    p.add_argument(
        "--blueprint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Send a default Rerun blueprint that arranges the cameras "
        "in a 2×2 grid (top-left), the 3-D wrist trace in its own pane "
        "(top-right), and the joint / TCP scalar series in a chart pane "
        "underneath. Pass --no-blueprint to keep whatever layout the "
        "viewer already had cached for this recording id (handy if "
        "you've hand-customised the panes and saved them).",
    )
    p.add_argument(
        "--ee-source",
        choices=("action", "observation"),
        default="action",
        help="Which 31-dim column drives FK. 'action' uses "
        "``action.body_q_mj`` (the wire delivered by the VLA bridge "
        "= what we COMMANDED the robot to do). 'observation' uses the "
        "first 31 dims of ``observation.state`` (what we MEASURED). "
        "Defaults to 'action' so the trace lines up with policy intent. "
        "Note: in VLA subscribe-mode the recorder mirrors the wire into "
        "``observation.state``, so both sources are identical there.",
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


# Wrist FK helpers (joint-name table + per-frame FK loop) live in a
# shared module so the segmentation pipeline and diagnostic CLIs can
# reuse them. The thin wrapper below preserves the viewer's existing
# call sites (positional ``mjcf_path``, ``body_q_mj_series``) and
# routes the prints through the ``[view]`` prefix.
from gear_sonic.utils.teleop.x2_fk_wrist import (  # noqa: E402
    X2_BODY_JOINT_NAMES_31 as _X2_BODY_JOINT_NAMES_31,
    compute_wrist_trajectories as _compute_wrist_trajectories_shared,
)


def _compute_wrist_trajectories(
    mjcf_path: Path,
    body_q_mj_series: np.ndarray,
) -> dict[str, np.ndarray] | None:
    """Viewer-flavoured wrapper around the shared wrist-FK helper.

    Keeps the existing call signature (``mjcf_path`` positional first,
    then the joint-pose series) and the ``[view]`` log prefix so the
    viewer logs are unchanged. See
    :func:`gear_sonic.utils.teleop.x2_fk_wrist.compute_wrist_trajectories`
    for the actual implementation.
    """
    return _compute_wrist_trajectories_shared(
        body_q_mj_series,
        mjcf_path=mjcf_path,
        log_prefix="[view]",
    )


def _build_default_blueprint(
    camera_keys: list[str],
    has_ee_trace: bool,
) -> "rrb.Blueprint":
    """Construct a sensible default layout for the viewer.

    Layout:

    .. code-block::

        +-------------------+--------------------+
        | head_front | ego  |  wrist trajectory  |
        +------------+------+   (Spatial3D)      |
        | stereo_L   | st_R |                    |
        +-------------------+--------------------+
        |          signals (TimeSeries)          |
        +----------------------------------------+

    The camera grid auto-collapses to whatever cameras actually exist
    in the recording (so 1-cam smoketests, 3-cam stereo-only sessions,
    and the full 4-cam VLA dataset all render cleanly).
    """
    # ── Cameras: 2×2 (or smaller) grid of Spatial2DView panes ─────────
    camera_views: list[rrb.View] = [
        rrb.Spatial2DView(
            origin=key,
            name=key.split(".")[-1],
        )
        for key in camera_keys
    ]
    if camera_views:
        n_cols = 2 if len(camera_views) > 1 else 1
        cameras_container: rrb.Container | rrb.View = rrb.Grid(
            *camera_views,
            grid_columns=n_cols,
            name="cameras",
        )
    else:
        # Degenerate case: parquet has no video tracks. Use an empty
        # text-doc-style placeholder so the layout still composes.
        cameras_container = rrb.TextDocumentView(
            origin="/", name="cameras (none)"
        )

    # ── Right column: 3-D wrist trace + scalar plots ──────────────────
    right_top_views: list[rrb.View] = []
    if has_ee_trace:
        right_top_views.append(
            rrb.Spatial3DView(
                origin="tcp",
                name="wrist trajectories",
                line_grid=rrb.LineGrid3D(visible=True),
            )
        )

    signals_view = rrb.TimeSeriesView(
        origin="/",
        contents=[
            "observation.state/**",
            "observation.projected_gravity/**",
            "action.body_q_mj/**",
            "action.left_hand_joints/**",
            "action.right_hand_joints/**",
            "action.body_q_mj_pre_sonic/**",
            "action.left_hand_joints_pre_sonic/**",
            "action.right_hand_joints_pre_sonic/**",
            "tcp/**/x",
            "tcp/**/y",
            "tcp/**/z",
        ],
        name="signals",
    )

    if right_top_views:
        right_column: rrb.Container = rrb.Vertical(
            *right_top_views,
            signals_view,
            row_shares=[1.0, 1.0],
            name="right_pane",
        )
    else:
        right_column = signals_view  # type: ignore[assignment]

    layout = rrb.Horizontal(
        cameras_container,
        right_column,
        column_shares=[1.0, 1.0],
        name="root",
    )
    return rrb.Blueprint(layout, collapse_panels=False)


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

    # ── 0) Send a default blueprint so the layout is stable across runs.
    #
    # Rerun normally derives the default blueprint from the entity tree.
    # When we add new entities between runs (e.g. when --ee-trace flips
    # on the ``tcp/`` 3-D ribbons), the auto-generated blueprint reshuffles
    # the panes, blanks half the camera grid, and pushes the 3-D view
    # somewhere unhelpful. Sending an explicit blueprint pins the layout:
    # cameras on the left (2×2 grid), wrist 3-D + signals on the right.
    # Pass --no-blueprint to opt out (e.g. when iterating on a manual
    # layout you've saved in the viewer).
    if args.blueprint:
        live_cameras = [k for k, p in video_paths.items() if p.is_file()]
        bp = _build_default_blueprint(
            live_cameras, has_ee_trace=args.ee_trace
        )
        rr.send_blueprint(bp)

    # ── 1) Log each video as a static asset ──────────────────────────
    fps = float(info.get("fps", 50))
    nanos_per_frame = int(round(1e9 / fps))
    for key, vpath in video_paths.items():
        if not vpath.is_file():
            continue
        rr.log(key, rr.AssetVideo(path=str(vpath)), static=True)

    # Robot frame convention: +x forward, +y left, +z up. Pinning this on
    # the ``tcp/`` namespace makes the auto-camera in the Spatial3DView
    # land at a useful eye / target / up triple instead of a top-down
    # near-orthographic projection.
    if args.ee_trace:
        rr.log("tcp", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

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

    # ── 3) End-effector FK trace (left/right wrist 3-D path) ─────────
    #
    # The 31-dim ``action.body_q_mj`` (or first 31 of ``observation.state``)
    # encodes commanded joint angles; what the operator actually cares
    # about for grasping is *where the wrist ended up in 3-D*. We run
    # forward kinematics on the canonical x2_ultra MJCF for every frame
    # and surface the wrist positions as:
    #
    #   (a) 3 scalar series per side -> easy to plot alongside the joint
    #       traces in the existing chart panes.
    #   (b) An animated ``Points3D`` per side -> shows the current wrist
    #       position when the playhead moves through a 3-D view.
    #   (c) A static ``LineStrips3D`` per side -> shows the full
    #       trajectory as a 3-D ribbon (toggleable independent of the
    #       timeline).
    #
    # Root is held at identity so the path is purely arm + waist pose
    # driven; walking / base drift do NOT contaminate the trace.
    if args.ee_trace:
        src_col = (
            "action.body_q_mj"
            if args.ee_source == "action"
            else "observation.state"
        )
        if src_col not in table.column_names:
            print(
                f"[view] WARN: --ee-source={args.ee_source} requested "
                f"but column {src_col!r} is not in the parquet; "
                "skipping end-effector trace.",
                flush=True,
            )
        else:
            raw = np.asarray(
                table.column(src_col).to_pylist(),
                dtype=np.float64,
            )
            if raw.ndim != 2 or raw.shape[1] < 31:
                print(
                    f"[view] WARN: {src_col} has shape {raw.shape}, "
                    "expected (N, >=31); skipping FK trace.",
                    flush=True,
                )
            else:
                mjcf_path = (
                    args.ee_mjcf
                    if args.ee_mjcf is not None
                    else REPO_ROOT
                    / "gear_sonic" / "data" / "assets"
                    / "robot_description" / "mjcf" / "x2_ultra.xml"
                )
                if not mjcf_path.is_file():
                    print(
                        f"[view] WARN: --ee-mjcf {mjcf_path} not found; "
                        "skipping end-effector trace.",
                        flush=True,
                    )
                    wrist_paths = None
                else:
                    print(
                        f"[view] running FK on {mjcf_path.name} "
                        f"(source={src_col}, {raw.shape[0]} frames)…",
                        flush=True,
                    )
                    wrist_paths = _compute_wrist_trajectories(
                        mjcf_path, raw
                    )

                if wrist_paths is not None:
                    for body_name, xyz in wrist_paths.items():
                        side = (
                            "left" if "left" in body_name else "right"
                        )
                        ee_key = f"tcp/{side}_wrist"
                        path_key = f"tcp/{side}_wrist_path"

                        # (a) Per-axis scalar series for chart panes
                        for d, axis in enumerate(("x", "y", "z")):
                            axis_values = xyz[:, d].astype(np.float64)
                            if idx_subset is not None:
                                axis_values = axis_values[idx_subset]
                            rr.send_columns(
                                f"{ee_key}/{axis}",
                                indexes=time_cols_scalars,
                                columns=rr.Scalars.columns(
                                    scalars=axis_values
                                ),
                            )

                        # (b) Animated Points3D (one point per frame
                        # so the 3-D view shows the wrist "now"). We
                        # send one point per frame via the columnar
                        # API: positions=(N, 1, 3) is interpreted as
                        # one point per index (N indices, batch of 1).
                        rr.send_columns(
                            ee_key,
                            indexes=time_cols,
                            columns=rr.Points3D.columns(
                                positions=xyz.reshape(-1, 3),
                            ),
                        )

                        # (c) Full static trajectory ribbon
                        rr.log(
                            path_key,
                            rr.LineStrips3D(
                                [xyz.tolist()],
                                colors=[
                                    [255, 80, 80]
                                    if side == "left"
                                    else [80, 200, 255]
                                ],
                                radii=[0.004],
                            ),
                            static=True,
                        )

                    print(
                        f"[view] end-effector trace logged for "
                        f"{', '.join(wrist_paths.keys())} "
                        "(see ``tcp/<side>_wrist/{x,y,z}`` chart "
                        "panes + 3-D ``tcp/`` entities).",
                        flush=True,
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
