"""Convert each LeRobot v2.1 episode into a motion-lib PKL that
``play_gesture --pkl`` (and the rest of the bones-seed motion_clip
family) accepts as a drop-in clip.

Output schema per PKL file (single motion key per file, matching the
filename stem) mirrors the existing MC-captured gesture PKLs under
``gear_sonic/data/motions/x2_recorded/mc_gestures/`` so they are
binary-indistinguishable from a "real" capture as far as
:func:`gear_sonic.utils.teleop.motion_clip_session._load_pkl_arrays`
is concerned::

    dof               (T, 31) float32  -- body joint angles
                                         (copied from action.body_q_mj)
    root_rot          (T, 4)  float32  -- xyzw quaternion per frame
                                         (identity by default; in-place
                                         gestures don't reorient the
                                         pelvis)
    root_trans_offset (T, 3)  float32  -- pelvis translation per frame
                                         (zeros; the motion_clip
                                         publish path does not consume
                                         translation for the gesture
                                         kind, but ``_load_pkl_arrays``
                                         requires the field)
    pose_aa           (T, 32, 3) float32 zeros  -- SMPL-style axis-angle,
                                                   unused by gesture path
    smpl_joints       (T, 24, 3) float32 zeros  -- SMPL joints, unused
    fps               int                       -- source recording fps
                                                   (defaults to the
                                                   dataset's meta/info.json,
                                                   typically 50)

    # Provenance (x2_record_*) carried over so ``play_x2_motion_mujoco.py``
    # and downstream training tools can trace each PKL back to its
    # source dataset / episode.
    x2_record_source_dataset:     str (--dataset path)
    x2_record_source_episode:     int (0-based episode index in --dataset)
    x2_record_task_description:   str (from meta/episodes.jsonl, used to
                                       derive the slug)
    x2_record_source_n_frames:    int
    x2_record_fps:                int

Hands and motion_token are intentionally dropped:
:class:`gear_sonic.utils.teleop.x2_dataset_recorder` zeroes both
``left_hand_q`` and ``right_hand_q`` for every motion_clip frame (see
``_publish_clip_frame``), so the motion_clip wire cannot carry recorded
hand motion. This matches the existing MC-gesture PKLs which are also
body-only. For full-fidelity replay (body + hands + motion_token)
use :file:`gear_sonic/scripts/replay_x2_dataset.sh` instead.

Naming convention
-----------------

For each kept episode, the script:

1. Slugifies the episode's task description (lowercase, non-alphanumeric
   collapses to ``_``, leading / trailing ``_`` stripped).
2. Groups episodes by slug; within each group assigns an incrementing
   ``001``, ``002``, ... suffix in source-episode order.
3. Writes ``<out_dir>/<slug>_<NNN>.pkl`` with a single motion-key entry
   matching the filename stem (e.g. ``wave_hello_with_right_hand_001``).

So a curated dataset with three episodes labelled ``"wave hello with
right hand"`` and two labelled ``"energetic right-hand wave"`` produces::

    wave_hello_with_right_hand_001.pkl  motion-key: wave_hello_with_right_hand_001
    wave_hello_with_right_hand_002.pkl  motion-key: wave_hello_with_right_hand_002
    wave_hello_with_right_hand_003.pkl  motion-key: wave_hello_with_right_hand_003
    energetic_right_hand_wave_001.pkl   motion-key: energetic_right_hand_wave_001
    energetic_right_hand_wave_002.pkl   motion-key: energetic_right_hand_wave_002

Examples
--------

Convert the curated x2_demo_gestures into per-episode PKLs::

    .venv/bin/python -m gear_sonic.scripts.lerobot_episode_to_motion_pkl \\
        --dataset data/lerobot/x2_demo_gestures_curated_v1 \\
        --out-dir gear_sonic/data/motions/x2_recorded/demo_gestures

Then play one from the running stack::

    python -m gear_sonic.scripts.play_gesture \\
        --pkl gear_sonic/data/motions/x2_recorded/demo_gestures/wave_hello_with_right_hand_001.pkl

Dry-run to see the plan without writing PKLs::

    .venv/bin/python -m gear_sonic.scripts.lerobot_episode_to_motion_pkl \\
        --dataset data/lerobot/x2_demo_gestures_curated_v1 \\
        --out-dir /tmp/preview --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Dataset layout helpers (kept local so the script has no hard
#    dependency on lerobot / the kinematic-replay module, which depends
#    on env_isaaclab for mujoco import paths). The few helpers we need
#    are tiny and stable; replicating them here keeps this script
#    importable from the lean ``.venv/`` planner stack. ──

_DEFAULT_CHUNK_SIZE: int = 1000
_BODY_ACTION_CANDIDATES: tuple[str, ...] = (
    "action.body_q_mj",            # LeRobot v1 schema (canonical body action)
    "action.commanded_body_q_mj",  # v0 legacy
)
NUM_BODY_DOFS = 31


def _resolve_dataset_path(name_or_path: str) -> Path:
    direct = Path(name_or_path).expanduser()
    if direct.is_dir():
        return direct.resolve()
    candidate = (REPO_ROOT / "data" / "lerobot" / name_or_path).resolve()
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(
        f"Dataset {name_or_path!r} not found. Tried:\n"
        f"  - {direct}\n"
        f"  - {candidate}\n"
        "Pass either an absolute path or a short name under data/lerobot/."
    )


def _episode_parquet_path(
    dataset_root: Path, episode: int, chunk_size: int = _DEFAULT_CHUNK_SIZE,
) -> Path:
    if episode < 0:
        raise ValueError(f"episode index must be >= 0; got {episode}")
    chunk_idx = episode // chunk_size
    return (
        dataset_root / "data" / f"chunk-{chunk_idx:03d}"
        / f"episode_{episode:06d}.parquet"
    )


def _read_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(
            f"Error: {info_path} not found -- {dataset_root} is not a "
            f"LeRobot v2.1 dataset."
        )
    return json.loads(info_path.read_text())


def _load_episodes_jsonl(path: Path) -> dict[int, dict]:
    if not path.is_file():
        raise SystemExit(f"Error: {path} not found.")
    out: dict[int, dict] = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ep = json.loads(line)
            out[int(ep["episode_index"])] = ep
    return out


def _parse_episode_spec(spec: str | None, total_episodes: int) -> list[int]:
    """Mirror the selector in lerobot_slice_episodes.py."""
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


_SLUG_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _slugify(text: str, *, fallback: str) -> str:
    """Lowercase, replace non-alphanumeric runs with ``_``, strip ends.

    Empty / all-separator inputs fall back to ``fallback``. The result
    is suitable as a filesystem stem AND as a motion-key inside the
    PKL dict.
    """
    lowered = text.strip().lower()
    slug = _SLUG_NON_ALNUM.sub("_", lowered).strip("_")
    return slug or fallback


def _load_body_q(parquet_path: Path) -> np.ndarray:
    """Load the per-frame body_q_mj column as ``(T, 31) float32``."""
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)
    table = pq.read_table(parquet_path)
    body_col = next(
        (c for c in _BODY_ACTION_CANDIDATES if c in table.column_names),
        None,
    )
    if body_col is None:
        raise ValueError(
            f"{parquet_path}: missing body action column. Tried "
            f"{_BODY_ACTION_CANDIDATES}. Available: {table.column_names}"
        )
    body = np.stack(table[body_col].to_numpy()).astype(np.float32)
    if body.ndim != 2 or body.shape[1] != NUM_BODY_DOFS:
        raise ValueError(
            f"{parquet_path}[{body_col}]: shape {body.shape} != "
            f"expected (T, {NUM_BODY_DOFS})"
        )
    if body.shape[0] < 2:
        raise ValueError(
            f"{parquet_path}: clip has only {body.shape[0]} frame(s); "
            "motion_clip_session requires >= 2."
        )
    return body


def _build_pkl_entry(
    *,
    body_q: np.ndarray,
    fps: int,
    src_dataset: Path,
    src_episode: int,
    task_description: str,
) -> dict:
    """Assemble one PKL motion entry from a body_q trajectory.

    The schema matches the MC gesture PKLs (e.g. ``salute_001.pkl``)
    field-for-field. Unused-by-gesture fields (``pose_aa``,
    ``smpl_joints``) are zero-filled with the expected dtypes / shapes
    so the PKL also loads cleanly in MuJoCo previewers that may
    consume them.
    """
    t = int(body_q.shape[0])
    # Identity quaternion in scipy / motion_clip_session's xyzw layout:
    # (qx, qy, qz, qw) = (0, 0, 0, 1).
    root_rot = np.tile(
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (t, 1),
    )
    root_trans_offset = np.zeros((t, 3), dtype=np.float32)
    pose_aa = np.zeros((t, 32, 3), dtype=np.float32)
    smpl_joints = np.zeros((t, 24, 3), dtype=np.float32)

    return {
        "root_trans_offset": root_trans_offset,
        "pose_aa": pose_aa,
        "dof": body_q.astype(np.float32, copy=False),
        "root_rot": root_rot,
        "smpl_joints": smpl_joints,
        "fps": int(fps),
        # Provenance: trace each PKL back to its LeRobot origin.
        "x2_record_source_dataset": str(src_dataset),
        "x2_record_source_episode": int(src_episode),
        "x2_record_task_description": str(task_description),
        "x2_record_source_n_frames": t,
        "x2_record_fps": int(fps),
        # Match the MC-gesture provenance shape: free-form note.
        "x2_record_meta": {
            "note": (
                f"converted from {src_dataset.name} episode "
                f"{src_episode} via lerobot_episode_to_motion_pkl"
            ),
            "task_description": str(task_description),
        },
    }


def _episode_task_string(ep_meta: dict, src_episode: int) -> str:
    """Extract a single task description string from an episodes.jsonl row.

    LeRobot stores ``tasks`` as a list (one episode can carry multiple
    annotations). The curator emits exactly one entry per episode; we
    join with ' / ' for safety if anything else writes multi-entry
    rows so the slug captures both labels rather than silently dropping
    one.
    """
    tasks = ep_meta.get("tasks")
    if isinstance(tasks, list) and tasks:
        return " / ".join(str(t).strip() for t in tasks if str(t).strip())
    if isinstance(tasks, str) and tasks.strip():
        return tasks.strip()
    raise SystemExit(
        f"Error: episode {src_episode} has no task description in "
        f"meta/episodes.jsonl (got {tasks!r})."
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True, type=str,
                   help="LeRobot v2.1 dataset short name (resolved under "
                        "data/lerobot/<name>) or absolute path.")
    p.add_argument("--out-dir", required=True, type=Path,
                   help="Directory to write the per-episode PKLs into. "
                        "Created if missing.")
    p.add_argument("--episodes", type=str, default=None,
                   help="Subset selector (e.g. '0,2-5,11-'). Defaults to "
                        "all source episodes.")
    p.add_argument("--prefix", type=str, default="",
                   help="Optional filename prefix prepended to every PKL "
                        "stem (e.g. 'demo_' produces "
                        "'demo_wave_hello_with_right_hand_001.pkl'). The "
                        "prefix is also embedded into the motion-key.")
    p.add_argument("--fps", type=int, default=None,
                   help="Override the fps stamped into the PKL (default: "
                        "from meta/info.json). Motion_clip_session will "
                        "resample to the recorder's publish rate at "
                        "playback time, so changing this just relabels "
                        "the source rate; it does NOT change the recorded "
                        "trajectory's effective speed.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing PKLs at the same output path. "
                        "Default skips any episode whose target PKL "
                        "already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan (output paths + frame counts) "
                        "and exit without writing anything.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    dataset_root = _resolve_dataset_path(args.dataset)
    info = _read_info(dataset_root)
    chunk_size = int(info.get("chunks_size", _DEFAULT_CHUNK_SIZE))
    src_fps = int(info.get("fps", 50))
    fps = int(args.fps) if args.fps is not None else src_fps

    total_eps = int(info.get("total_episodes", 0))
    if total_eps <= 0:
        print(
            f"Error: dataset {dataset_root} reports total_episodes="
            f"{total_eps}.", file=sys.stderr,
        )
        return 2
    episodes_meta = _load_episodes_jsonl(dataset_root / "meta" / "episodes.jsonl")
    selected = _parse_episode_spec(args.episodes, total_eps)
    if not selected:
        print("Error: episode selection is empty.", file=sys.stderr)
        return 2

    out_dir: Path = args.out_dir.resolve()

    # ── Plan: assign filenames by slug-and-count BEFORE doing any IO ──
    # We sort each slug group by source episode index so the suffix
    # numbering is deterministic regardless of selection order.
    plan: list[tuple[int, str, str, Path]] = []
    seen: dict[str, int] = {}
    by_slug: dict[str, list[int]] = {}
    for src_idx in selected:
        if src_idx not in episodes_meta:
            print(f"[lerobot2pkl] WARN: episode {src_idx} missing from "
                  f"meta/episodes.jsonl; skipping.", file=sys.stderr)
            continue
        task = _episode_task_string(episodes_meta[src_idx], src_idx)
        slug = _slugify(task, fallback=f"episode_{src_idx:06d}")
        by_slug.setdefault(slug, []).append(src_idx)
    for slug in by_slug:
        by_slug[slug].sort()
    for slug, idxs in by_slug.items():
        for i, src_idx in enumerate(idxs, start=1):
            stem = f"{args.prefix}{slug}_{i:03d}"
            out_pkl = out_dir / f"{stem}.pkl"
            plan.append((src_idx, slug, stem, out_pkl))
    plan.sort(key=lambda row: row[0])  # walk in src-episode order for nice logs

    # ── Plan summary ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("LEROBOT → MOTION-LIB PKL CONVERTER")
    print("=" * 72)
    print(f"  src dataset:    {dataset_root}")
    print(f"  src episodes:   {total_eps}  (selected: {len(selected)})")
    print(f"  out dir:        {out_dir}")
    print(f"  fps stamped:    {fps}  (source recording fps = {src_fps})")
    print(f"  prefix:         {args.prefix!r}")
    print(f"  unique tasks:   {len(by_slug)}")
    print()
    print(f"  {'src ep':>6}  {'frames':>6}  {'slug':<40}  output")
    for src_idx, slug, stem, out_pkl in plan:
        n = int(episodes_meta[src_idx]["length"])
        print(
            f"  {src_idx:>6}  {n:>6}  {slug[:40]:<40}  "
            f"{out_pkl.relative_to(REPO_ROOT) if out_pkl.is_relative_to(REPO_ROOT) else out_pkl}"
        )
    print("=" * 72)
    print()

    if args.dry_run:
        print("[lerobot2pkl] --dry-run set; not writing anything.")
        return 0

    # ── Write PKLs ──────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    n_failed = 0
    for src_idx, slug, stem, out_pkl in plan:
        if out_pkl.exists() and not args.force:
            print(
                f"[lerobot2pkl]   ep {src_idx:>3} -> {out_pkl.name}: "
                f"exists (pass --force to overwrite); SKIP"
            )
            n_skipped += 1
            continue

        parquet_path = _episode_parquet_path(
            dataset_root, src_idx, chunk_size=chunk_size,
        )
        try:
            body_q = _load_body_q(parquet_path)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[lerobot2pkl] ERR ep {src_idx}: {exc}",
                  file=sys.stderr)
            n_failed += 1
            continue
        task = _episode_task_string(episodes_meta[src_idx], src_idx)
        entry = _build_pkl_entry(
            body_q=body_q,
            fps=fps,
            src_dataset=dataset_root,
            src_episode=src_idx,
            task_description=task,
        )
        pkl_data = {stem: entry}
        out_pkl.parent.mkdir(parents=True, exist_ok=True)
        # ``compress=3`` matches what the warehouse-stitch / bones-seed
        # PKLs in the repo use (joblib default-ish, modest level).
        joblib.dump(pkl_data, out_pkl, compress=3)
        n_written += 1
        print(
            f"[lerobot2pkl]   ep {src_idx:>3} -> {out_pkl.name}  "
            f"(motion-key={stem!r}, {body_q.shape[0]} frames, "
            f"{body_q.shape[0] / fps:.2f} s)"
        )

    print()
    print(f"[lerobot2pkl] done. wrote={n_written} skipped={n_skipped} failed={n_failed}")
    if n_written > 0:
        print()
        print("[lerobot2pkl] play one on the running stack:")
        first = next(p for _, _, _, p in plan if p.exists())
        try:
            rel = first.relative_to(REPO_ROOT)
            shown = str(rel)
        except ValueError:
            shown = str(first)
        print(f"    python -m gear_sonic.scripts.play_gesture --pkl {shown}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
