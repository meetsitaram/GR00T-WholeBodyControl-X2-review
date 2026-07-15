"""Interactive curator: review LeRobot v2.1 episodes one by one, keep / drop / relabel,
then write a new curated dataset.

For each source episode the curator pops the MuJoCo kinematic viewer
(``gear_sonic.scripts.replay_x2_kinematic``) so you can eyeball the
trajectory, then asks at a shell prompt whether to keep it and what new
task description to give it. Decisions persist to a state file next to
``--dst`` so you can Ctrl-C any time and resume on the next run.

When every (selected) episode has a decision, the curator prints a
summary and asks for confirmation before materialising the new dataset.
Materialisation reuses the same slice + parquet-rewrite logic as
``lerobot_slice_episodes.py`` (renumber episodes 0..K-1, copy MP4s,
rebuild ``meta/info.json``, ``meta/episodes.jsonl``,
``meta/episodes_stats.jsonl``) and additionally:

* Rewrites ``meta/tasks.jsonl`` with one entry per unique task string
  the operator typed (case-sensitive de-duplication).
* Rewrites the per-frame ``task_index`` column inside every copied
  parquet so each new episode actually carries its new task index.
* Rewrites the per-episode ``task_index`` stats inside
  ``meta/episodes_stats.jsonl`` so the metadata isn't lying.
* Sets each ``episodes.jsonl`` entry's ``tasks`` field to the new
  task description.

The source dataset is **never** modified -- the curator only reads
``--src`` and writes to ``--dst`` and the resumable state file.

Example
-------

Review the just-recorded ``right_wave`` gesture set, drop the duds,
relabel each kept episode with its own task description, write the
curated dataset to ``data/lerobot/x2_demo_gestures_curated_v1``::

    .venv/bin/python -m gear_sonic.scripts.curate_x2_demo_episodes \\
        --src data/lerobot/x2_demo_gestures \\
        --dst data/lerobot/x2_demo_gestures_curated_v1

If you Ctrl-C mid-review the next run skips episodes you've already
decided. Pass ``--restart`` to clear state and start over.

Keyboard / prompt shortcuts
---------------------------

After each viewer closes the curator prompts ``[ep N]> `` where you can
type:

* ``k <task description>`` -- keep this episode with the given task.
* ``k`` (no text)            -- keep with last-typed task description.
* ``s`` / ``skip``           -- drop this episode.
* ``r`` / ``replay``         -- re-pop the viewer for this episode.
* ``b`` / ``back``           -- undo the previous decision and re-review it.
* ``q`` / ``quit``           -- save state and stop (resume later).
* ``l`` / ``list``           -- print all decisions so far.
* ``h`` / ``help``           -- show this help.

Implementation notes
--------------------

* The viewer is spawned as a subprocess (``sys.executable -m
  gear_sonic.scripts.replay_x2_kinematic ... --loop``) and the curator
  blocks until the user closes the viewer window. This keeps the
  prompt single-shot per episode and avoids any in-process viewer
  state leaking between episodes.
* The slice / relabel half of this script lifts helpers (parquet
  rewrite, camera enumeration) from ``lerobot_slice_episodes`` so the
  byte layout of the output dataset matches what that script would
  produce -- the only delta is the per-episode task plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_episode_spec(spec: str | None, total_episodes: int) -> list[int]:
    """Mirror :func:`lerobot_slice_episodes._parse_episode_spec`."""
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


@dataclass
class Decision:
    """One per-episode review outcome."""

    src_index: int
    keep: bool
    task: str | None = None  # None when keep=False
    notes: str = ""


@dataclass
class CurateState:
    """Persisted on disk after every prompt so we can resume."""

    src_root: str
    decisions: dict[int, Decision] = field(default_factory=dict)
    last_task: str = ""

    def to_dict(self) -> dict:
        return {
            "src_root": self.src_root,
            "last_task": self.last_task,
            "decisions": [
                {
                    "src_index": d.src_index,
                    "keep": d.keep,
                    "task": d.task,
                    "notes": d.notes,
                }
                for d in sorted(self.decisions.values(), key=lambda x: x.src_index)
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CurateState":
        st = cls(src_root=d.get("src_root", ""), last_task=d.get("last_task", ""))
        for entry in d.get("decisions", []):
            dec = Decision(
                src_index=int(entry["src_index"]),
                keep=bool(entry["keep"]),
                task=entry.get("task"),
                notes=entry.get("notes", ""),
            )
            st.decisions[dec.src_index] = dec
        return st


def _state_path_for_dst(dst: Path) -> Path:
    """Sidecar JSON next to the eventual dataset dir."""
    return dst.with_suffix(dst.suffix + ".curate_state.json")


def _load_state(state_path: Path, src_root: Path) -> CurateState:
    if not state_path.is_file():
        return CurateState(src_root=str(src_root))
    try:
        raw = json.loads(state_path.read_text())
    except json.JSONDecodeError:
        print(
            f"[curate] WARN: state file {state_path} is corrupt; "
            "starting fresh.", file=sys.stderr,
        )
        return CurateState(src_root=str(src_root))
    st = CurateState.from_dict(raw)
    saved_src = Path(st.src_root).resolve() if st.src_root else None
    if saved_src and saved_src != src_root.resolve():
        print(
            f"[curate] WARN: state file {state_path} was created for "
            f"src={saved_src}, but this run uses src={src_root}. "
            "Existing decisions still apply by src_index -- pass "
            "--restart if that's wrong.", file=sys.stderr,
        )
    st.src_root = str(src_root)
    return st


def _save_state(state_path: Path, state: CurateState) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp.write_text(json.dumps(state.to_dict(), indent=2))
    os.replace(tmp, state_path)


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


def _spawn_viewer(
    src: Path,
    src_index: int,
    rate: float,
    loop: bool,
    extra: list[str] | None,
) -> int:
    """Block on the kinematic viewer subprocess for one episode."""
    cmd = [
        sys.executable,
        "-m",
        "gear_sonic.scripts.replay_x2_kinematic",
        "--dataset",
        str(src),
        "--episode",
        str(src_index),
        "--rate",
        f"{rate:.3f}",
        "--quiet",
    ]
    if loop:
        cmd.append("--loop")
    if extra:
        cmd.extend(extra)
    print(
        f"[curate] launching viewer: {' '.join(shlex.quote(c) for c in cmd)}",
        flush=True,
    )
    print("[curate] close the MuJoCo window when you're done watching.",
          flush=True)
    try:
        return subprocess.call(cmd, cwd=str(REPO_ROOT))
    except KeyboardInterrupt:
        # The user hit Ctrl-C in the curator while the viewer was up.
        # Let the prompt loop decide what to do next.
        return 130


def _print_help() -> None:
    print(
        "\n"
        "  k <task>   keep this episode with the given task description\n"
        "  k          keep with the LAST task description you typed\n"
        "  s / skip   drop this episode\n"
        "  r / replay re-pop the viewer for this episode\n"
        "  b / back   undo previous decision and re-review it\n"
        "  q / quit   save state and stop (resume by re-running)\n"
        "  l / list   print every decision so far\n"
        "  h / help   show this menu\n",
        flush=True,
    )


def _print_decisions(state: CurateState) -> None:
    if not state.decisions:
        print("[curate] no decisions yet.", flush=True)
        return
    print("[curate] decisions so far:", flush=True)
    for src_idx in sorted(state.decisions):
        d = state.decisions[src_idx]
        if d.keep:
            print(f"  src ep {src_idx:>3}: KEEP  task={d.task!r}",
                  flush=True)
        else:
            print(f"  src ep {src_idx:>3}: drop", flush=True)


def _prompt_for_decision(
    src_index: int,
    n_frames: int,
    fps: float,
    last_task: str,
) -> tuple[str, str]:
    """Return ``(verb, payload)`` parsed from the prompt.

    ``verb`` is one of ``keep`` / ``skip`` / ``replay`` / ``back`` / ``quit`` /
    ``list`` / ``help``. ``payload`` is the task description when ``verb ==
    'keep'`` and empty otherwise.
    """
    duration = n_frames / fps if fps > 0 else 0.0
    last_hint = f"  (last task: {last_task!r})" if last_task else ""
    while True:
        try:
            raw = input(
                f"[ep {src_index}] {n_frames} frames ({duration:.1f} s){last_hint}\n"
                f"  k <task> / s / r / b / q / l / h > "
            )
        except EOFError:
            return "quit", ""
        cmd = raw.strip()
        if not cmd:
            continue
        lower = cmd.lower()
        if lower in ("h", "help", "?"):
            _print_help()
            continue
        if lower in ("l", "list", "ls"):
            return "list", ""
        if lower in ("r", "replay", "rr"):
            return "replay", ""
        if lower in ("b", "back", "undo"):
            return "back", ""
        if lower in ("q", "quit", "exit"):
            return "quit", ""
        if lower in ("s", "skip", "drop", "d"):
            return "skip", ""
        if lower == "k" or lower == "keep":
            if not last_task:
                print(
                    "[curate] no previous task to reuse -- type 'k <task>' "
                    "with a description.", flush=True,
                )
                continue
            return "keep", last_task
        if lower.startswith("k ") or lower.startswith("keep "):
            payload = cmd.split(None, 1)[1].strip()
            if not payload:
                print("[curate] task description is empty.", flush=True)
                continue
            return "keep", payload
        # Plain text counts as a keep decision (lets the operator just
        # type the task instead of prefixing it with `k `).
        return "keep", cmd


def _rewrite_parquet(
    src_pq: Path,
    dst_pq: Path,
    new_episode_index: int,
    new_global_index_start: int,
    new_task_index: int,
) -> int:
    """Copy a parquet rewriting ``episode_index``, ``index`` and ``task_index``."""
    table = pq.read_table(src_pq)
    n = table.num_rows

    ep_arr = pa.array(np.full(n, new_episode_index, dtype=np.int64))
    idx_arr = pa.array(
        np.arange(
            new_global_index_start, new_global_index_start + n, dtype=np.int64,
        )
    )
    task_arr = pa.array(np.full(n, new_task_index, dtype=np.int64))

    schema = table.schema
    cols = []
    for name in table.column_names:
        if name == "episode_index":
            cols.append(ep_arr)
        elif name == "index":
            cols.append(idx_arr)
        elif name == "task_index":
            cols.append(task_arr)
        else:
            cols.append(table.column(name))
    new_table = pa.Table.from_arrays(cols, schema=schema)
    dst_pq.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_pq)
    return n


def _materialize(
    *,
    src: Path,
    dst: Path,
    kept: list[Decision],
    src_info: dict,
    src_episodes: dict[int, dict],
    src_stats: dict[int, dict],
    force: bool,
) -> int:
    """Write the curated dataset to ``dst`` and return its episode count."""
    fps = float(src_info.get("fps", 50))
    chunks_size = int(src_info.get("chunks_size", 1000))
    data_path_pattern = src_info["data_path"]
    video_path_pattern = src_info["video_path"]
    cams = _camera_keys(src_info)

    # Build a stable task table from the unique strings the user typed.
    # We preserve first-seen order so the operator gets a predictable
    # task_index layout (matches the order episodes appear in --dst).
    task_to_idx: dict[str, int] = {}
    for d in kept:
        assert d.task is not None
        if d.task not in task_to_idx:
            task_to_idx[d.task] = len(task_to_idx)

    # Plan (old_idx, new_idx, n_frames, new_global_start, new_task_index)
    plan: list[tuple[int, int, int, int, int]] = []
    running_global = 0
    for new_idx, d in enumerate(kept):
        if d.src_index not in src_episodes:
            print(
                f"Error: src episodes.jsonl missing index {d.src_index}",
                file=sys.stderr,
            )
            return 2
        n = int(src_episodes[d.src_index]["length"])
        plan.append(
            (d.src_index, new_idx, n, running_global, task_to_idx[d.task])
        )
        running_global += n
    new_total_frames = running_global
    new_total_episodes = len(plan)

    print()
    print(f"[curate] materializing curated dataset")
    print(f"[curate]   src:     {src}  (READ-ONLY)")
    print(f"[curate]   dst:     {dst}")
    print(
        f"[curate]   episodes: {new_total_episodes} "
        f"(from {int(src_info.get('total_episodes', 0))} src)"
    )
    print(
        f"[curate]   frames:   {new_total_frames} "
        f"({new_total_frames / fps / 60:.2f} min @ {fps:.0f} fps)"
    )
    print(f"[curate]   unique tasks ({len(task_to_idx)}):")
    for task, idx in task_to_idx.items():
        n_eps_with_task = sum(1 for d in kept if d.task == task)
        print(f"     [{idx:>2}] {task!r}  ({n_eps_with_task} episode(s))")

    if dst.exists():
        if not force:
            print(
                f"Error: --dst {dst} already exists. Re-run with --force "
                f"to overwrite (rm -rf and rebuild).", file=sys.stderr,
            )
            return 2
        print(f"[curate] --force: removing existing {dst}")
        shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Copy parquet + video files with renumbering / task-index rewrite.
    for old_idx, new_idx, n, new_global_start, new_task_index in plan:
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
            new_task_index=new_task_index,
        )
        if written != n:
            print(
                f"[curate] WARN: ep {old_idx} parquet has {written} rows "
                f"but episodes.jsonl claims {n}. Keeping {written}.",
                file=sys.stderr,
            )

        for cam in cams:
            src_mp4 = src / video_path_pattern.format(
                episode_chunk=old_chunk, episode_index=old_idx, video_key=cam,
            )
            dst_mp4 = dst / video_path_pattern.format(
                episode_chunk=new_chunk, episode_index=new_idx, video_key=cam,
            )
            dst_mp4.parent.mkdir(parents=True, exist_ok=True)
            if src_mp4.is_file():
                shutil.copy2(src_mp4, dst_mp4)
            else:
                print(
                    f"[curate] WARN: src video missing: "
                    f"{src_mp4.relative_to(src)}",
                    file=sys.stderr,
                )

        print(
            f"[curate]   wrote new ep {new_idx:>3} "
            f"(was src ep {old_idx:>3}, {n} frames, "
            f"task_index={new_task_index})",
            flush=True,
        )

    # ── meta/info.json ──────────────────────────────────────────────
    dst_meta = dst / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)
    new_info = dict(src_info)
    new_info["total_episodes"] = new_total_episodes
    new_info["total_frames"] = new_total_frames
    new_info["total_tasks"] = len(task_to_idx)
    new_info["total_chunks"] = (
        (new_total_episodes + chunks_size - 1) // chunks_size
    )
    new_info["splits"] = {"train": f"0:{new_total_episodes}"}
    if "total_videos" in new_info:
        new_info["total_videos"] = new_total_episodes * len(cams)
    (dst_meta / "info.json").write_text(json.dumps(new_info, indent=2))

    # ── meta/tasks.jsonl (rebuilt from unique typed descriptions) ──
    with open(dst_meta / "tasks.jsonl", "w") as f:
        for task, idx in task_to_idx.items():
            f.write(json.dumps({"task_index": idx, "task": task}) + "\n")

    # ── meta/episodes.jsonl ────────────────────────────────────────
    with open(dst_meta / "episodes.jsonl", "w") as f:
        for old_idx, new_idx, n, _start, new_task_index in plan:
            ep_meta = dict(src_episodes[old_idx])
            ep_meta["episode_index"] = new_idx
            # Replace the source task list with the new single-task label.
            ep_meta["tasks"] = [next(
                t for t, i in task_to_idx.items() if i == new_task_index
            )]
            f.write(json.dumps(ep_meta) + "\n")

    # ── meta/episodes_stats.jsonl ──────────────────────────────────
    # Renumber the top-level ``episode_index`` and rewrite the
    # ``task_index`` per-column stats so the metadata matches the new
    # labels. Leave the other per-column stats alone -- they're
    # frame-derived and don't depend on which task this episode
    # belongs to (matches the policy of ``lerobot_slice_episodes`` for
    # the other index columns; we explicitly fix task_index because
    # we changed the values).
    if src_stats:
        with open(dst_meta / "episodes_stats.jsonl", "w") as f:
            missing = 0
            for old_idx, new_idx, n, _start, new_task_index in plan:
                if old_idx not in src_stats:
                    missing += 1
                    continue
                d = json.loads(json.dumps(src_stats[old_idx]))  # deep copy
                d["episode_index"] = new_idx
                if "stats" in d and "task_index" in d["stats"]:
                    d["stats"]["task_index"] = {
                        "min": [int(new_task_index)],
                        "max": [int(new_task_index)],
                        "mean": [float(new_task_index)],
                        "std": [0.0],
                        "count": [int(n)],
                    }
                f.write(json.dumps(d) + "\n")
            if missing:
                print(
                    f"[curate] WARN: {missing} kept episodes had no entry "
                    f"in episodes_stats.jsonl.", file=sys.stderr,
                )

    # ── Verbatim copies ────────────────────────────────────────────
    for fname in ("dataset_format_version.json", "modality.json"):
        src_f = src / "meta" / fname
        if src_f.is_file():
            shutil.copy2(src_f, dst_meta / fname)

    print()
    print(f"[curate] done.")
    print(f"[curate]   dst:          {dst}")
    print(
        f"[curate]   episodes:     {new_total_episodes} "
        f"(src had {int(src_info.get('total_episodes', 0))})"
    )
    print(f"[curate]   frames:       {new_total_frames}")
    print(
        f"[curate]   wall time:    {new_total_frames / fps / 60:.2f} min "
        f"@ {fps:.0f} fps"
    )
    print(f"[curate]   tasks.jsonl:  {len(task_to_idx)} entries")
    print()
    print(f"[curate] review one episode in the rerun viewer:")
    print(
        f"    ./gear_sonic/scripts/view_x2_recorded_dataset.sh "
        f"--root {dst} --episode 0"
    )
    print()
    print(f"[curate] replay on the deploy / robot (SONIC loop):")
    print(
        f"    ./gear_sonic/scripts/run_x2_replay_stack.sh "
        f"--dataset {dst.name} --episode 0 --pc2-host <PC2_IP>"
    )
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--src", type=Path, required=True,
                   help="Source LeRobot v2.1 dataset root (READ-ONLY).")
    p.add_argument("--dst", type=Path, required=True,
                   help="Destination dataset root (must not exist unless --force).")
    p.add_argument("--episodes", type=str, default=None,
                   help="Subset selector (e.g. '0,2-5,11-'). Defaults to all "
                        "source episodes.")
    p.add_argument("--rate", type=float, default=50.0,
                   help="MuJoCo viewer playback rate Hz (default 50.0 "
                        "= source recording rate).")
    p.add_argument("--no-viewer-loop", action="store_true",
                   help="Pass --loop=False to the kinematic viewer "
                        "(viewer plays once then waits for window close).")
    p.add_argument("--no-viewer", action="store_true",
                   help="Skip launching the kinematic viewer; prompt only "
                        "(useful for re-curating from notes / state file).")
    p.add_argument("--restart", action="store_true",
                   help="Delete the resumable state file before starting.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite --dst if it already exists.")
    p.add_argument("--auto-materialize", action="store_true",
                   help="Skip the final 'materialize? (y/N)' confirmation "
                        "and write the dataset immediately when every "
                        "selected episode has a decision.")
    p.add_argument("--state-file", type=Path, default=None,
                   help="Override the resumable state file location "
                        "(default: <dst>.curate_state.json).")
    p.add_argument("--viewer-extra", type=str, default=None,
                   help="Extra args to forward to replay_x2_kinematic "
                        "(quoted string, shell-style).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    src: Path = args.src.resolve()
    dst: Path = args.dst.resolve()

    if not src.is_dir():
        print(f"Error: --src {src} is not a directory.", file=sys.stderr)
        return 2
    info_path = src / "meta" / "info.json"
    if not info_path.is_file():
        print(
            f"Error: {info_path} not found -- --src is not a LeRobot "
            f"v2.1 dataset.", file=sys.stderr,
        )
        return 2

    src_info = json.loads(info_path.read_text())
    total_src_eps = int(src_info["total_episodes"])
    fps = float(src_info.get("fps", 50))
    src_episodes = _load_episodes_jsonl(src / "meta" / "episodes.jsonl")
    src_stats = _load_episodes_stats_jsonl(
        src / "meta" / "episodes_stats.jsonl"
    )

    selected = _parse_episode_spec(args.episodes, total_src_eps)
    if not selected:
        print("Error: episode selection is empty.", file=sys.stderr)
        return 2

    state_path = args.state_file.resolve() if args.state_file else _state_path_for_dst(dst)
    if args.restart and state_path.is_file():
        print(f"[curate] --restart: removing {state_path}")
        state_path.unlink()
    state = _load_state(state_path, src)

    viewer_extra = shlex.split(args.viewer_extra) if args.viewer_extra else None

    print()
    print("=" * 72)
    print("X2 LEROBOT EPISODE CURATOR")
    print("=" * 72)
    print(f"  src:           {src}")
    print(f"  dst:           {dst}")
    print(f"  state file:    {state_path}")
    print(f"  src episodes:  {total_src_eps}")
    print(f"  selected:      {len(selected)} indices")
    print(f"  decisions:     {len(state.decisions)} loaded from state")
    print(f"  viewer:        {'OFF' if args.no_viewer else 'kinematic MuJoCo'}")
    print("=" * 72)
    print()

    # ── Interactive review loop ────────────────────────────────────
    review_queue = list(selected)
    quit_early = False
    while review_queue:
        src_idx = review_queue[0]
        if src_idx in state.decisions:
            # Already decided -- skip silently. We still pop it so the
            # queue advances.
            review_queue.pop(0)
            continue

        if src_idx not in src_episodes:
            print(
                f"[curate] WARN: src ep {src_idx} not in episodes.jsonl; "
                "skipping.", file=sys.stderr,
            )
            review_queue.pop(0)
            continue
        n_frames = int(src_episodes[src_idx]["length"])

        # Cheap dud guard: 1-frame episodes are recorder misfires; the
        # viewer would just flash one pose. Offer to auto-skip but let
        # the operator override with a real `k` if they want.
        if n_frames <= 1:
            print(
                f"[curate] ep {src_idx} has only {n_frames} frame(s) "
                f"(likely recorder misfire). Auto-skipping; type 'b' "
                "at any later prompt to come back.", flush=True,
            )
            state.decisions[src_idx] = Decision(
                src_index=src_idx, keep=False, notes="auto-skip: 1-frame dud",
            )
            _save_state(state_path, state)
            review_queue.pop(0)
            continue

        while True:
            if not args.no_viewer:
                _spawn_viewer(
                    src=src,
                    src_index=src_idx,
                    rate=args.rate,
                    loop=not args.no_viewer_loop,
                    extra=viewer_extra,
                )
            try:
                verb, payload = _prompt_for_decision(
                    src_index=src_idx,
                    n_frames=n_frames,
                    fps=fps,
                    last_task=state.last_task,
                )
            except KeyboardInterrupt:
                print("\n[curate] Ctrl-C -- saving state and exiting.",
                      flush=True)
                _save_state(state_path, state)
                return 130

            if verb == "help":
                _print_help()
                continue
            if verb == "list":
                _print_decisions(state)
                continue
            if verb == "replay":
                if args.no_viewer:
                    print("[curate] --no-viewer set; nothing to replay.",
                          flush=True)
                continue
            if verb == "back":
                # Undo the most recent decision (highest src_index <= src_idx).
                prior = [i for i in sorted(state.decisions) if i < src_idx]
                if not prior:
                    print("[curate] no previous decision to undo.",
                          flush=True)
                    continue
                last_idx = prior[-1]
                undone = state.decisions.pop(last_idx)
                _save_state(state_path, state)
                print(
                    f"[curate] undone src ep {last_idx} "
                    f"(was keep={undone.keep}, task={undone.task!r}); "
                    "re-queuing.", flush=True,
                )
                review_queue.insert(0, last_idx)
                break
            if verb == "quit":
                quit_early = True
                break
            if verb == "skip":
                state.decisions[src_idx] = Decision(
                    src_index=src_idx, keep=False,
                )
                _save_state(state_path, state)
                print(f"[curate]   -> dropped ep {src_idx}", flush=True)
                review_queue.pop(0)
                break
            if verb == "keep":
                state.decisions[src_idx] = Decision(
                    src_index=src_idx, keep=True, task=payload,
                )
                state.last_task = payload
                _save_state(state_path, state)
                print(
                    f"[curate]   -> kept   ep {src_idx} as task={payload!r}",
                    flush=True,
                )
                review_queue.pop(0)
                break
            # Defensive: unknown verb just re-prompts.
            print(f"[curate] unknown command {verb!r}", file=sys.stderr)

        if quit_early:
            break

    # ── Summary ─────────────────────────────────────────────────────
    decided = [state.decisions[i] for i in selected if i in state.decisions]
    kept = [d for d in decided if d.keep]
    skipped = [d for d in decided if not d.keep]
    pending = [i for i in selected if i not in state.decisions]

    print()
    print("=" * 72)
    print(f"  decided:    {len(decided)} / {len(selected)} selected episodes")
    print(f"  kept:       {len(kept)}")
    print(f"  skipped:    {len(skipped)}")
    print(f"  pending:    {len(pending)} (run again to finish)")
    if kept:
        total_n = sum(int(src_episodes[d.src_index]["length"]) for d in kept)
        print(
            f"  duration:   {total_n / fps:.1f} s "
            f"({total_n} frames @ {fps:.0f} fps)"
        )
        unique = sorted({d.task for d in kept if d.task})
        print(f"  unique task descriptions ({len(unique)}):")
        for t in unique:
            n = sum(1 for d in kept if d.task == t)
            print(f"     - {t!r}  ({n} episode(s))")
    print("=" * 72)
    print()

    if pending:
        print(
            f"[curate] {len(pending)} episode(s) still un-reviewed: "
            f"{pending}. Re-run to finish; not materialising yet.",
            file=sys.stderr,
        )
        return 0
    if not kept:
        print(
            "[curate] no episodes kept -- nothing to materialise.",
            file=sys.stderr,
        )
        return 0

    if not args.auto_materialize:
        try:
            ans = input(
                f"[curate] materialise {len(kept)} kept episode(s) at "
                f"{dst} ? (y/N) "
            )
        except (EOFError, KeyboardInterrupt):
            print("\n[curate] aborted before materialisation.", flush=True)
            return 0
        if ans.strip().lower() not in ("y", "yes"):
            print("[curate] aborted by operator; state preserved.",
                  flush=True)
            return 0

    rc = _materialize(
        src=src,
        dst=dst,
        kept=kept,
        src_info=src_info,
        src_episodes=src_episodes,
        src_stats=src_stats,
        force=args.force,
    )
    return rc


if __name__ == "__main__":
    sys.exit(main())
