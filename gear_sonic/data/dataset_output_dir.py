"""Shared preflight for dataset output directories.

Both the pure-kinematic teleop script and the full SONIC-stabilised
recorder write LeRobot v2.1 datasets via ``Gr00tDataExporter.create()``,
which has a sharp edge: it distinguishes "fresh dataset" from "resume
existing dataset" purely by whether ``save_root`` exists on disk. If a
previous run crashed *after* writing ``meta/info.json`` but *before*
finalising the first episode, the next launch lands in the resume code
path with no real data, and the exporter calls out to Hugging Face Hub
looking for the placeholder repo ``tmp/tmp_dataset`` -- 404 ->
"Failed to resume from corrupted dataset".

This module's job is to fail fast in that case (and in a few other
ill-formed states) BEFORE the operator boots Quest 3 / MuJoCo. Cheap
CI-able preflight is the right place to fail.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def preflight_dataset_output_dir(path: Path, *, log_prefix: str = "preflight") -> None:
    """Validate or auto-clean an existing dataset ``--output-dir``.

    Cases:

    * Path does not exist  -- nothing to do; exporter will create fresh.
    * Path is a valid LeRobot dataset (``meta/info.json`` present AND at
      least one finalized episode parquet under ``data/``) -- nothing
      to do; exporter will resume in append-mode.
    * Path is a "half-init stub": ``meta/info.json`` was written by a
      previous run that crashed before any episode finalized, so
      ``data/`` has zero parquet files. Auto-clean and continue.
    * Path exists but is empty, OR contains ONLY a stale ``debug/`` from
      a previous crashed run -- auto-clean and continue.
    * Path exists with real content but no ``meta/info.json`` -- raise
      ``SystemExit`` with an actionable message (operator can ``rm -rf``
      or pick a new ``--output-dir``).

    Args:
        path: directory the exporter will be pointed at.
        log_prefix: prefix to use in WARN log lines so the operator can
            tell which script's preflight is talking.
    """
    if not path.exists():
        return

    info_json = path / "meta" / "info.json"
    if info_json.is_file():
        data_dir = path / "data"
        has_episodes = data_dir.is_dir() and any(data_dir.rglob("*.parquet"))
        if has_episodes:
            return
        print(
            f"[{log_prefix}] WARN: output dir {path} has meta/info.json but "
            f"zero finalized episodes (data/ is empty); cleaning up half-init "
            f"stub before fresh dataset write.",
            flush=True,
        )
        shutil.rmtree(path)
        return

    children = sorted(p.name for p in path.iterdir())
    safe_to_clean = (
        len(children) == 0
        or set(children).issubset({"debug"})
    )
    if safe_to_clean:
        print(
            f"[{log_prefix}] WARN: output dir {path} exists but has no "
            f"meta/info.json (children={children}); cleaning up before "
            f"fresh dataset write.",
            flush=True,
        )
        shutil.rmtree(path)
        return

    raise SystemExit(
        f"Error: --output-dir {path} already exists but is not a valid "
        f"LeRobot dataset (missing meta/info.json) AND has unexpected "
        f"contents: {children}. Either delete the directory yourself "
        f"(`rm -rf {path}`) to start a fresh dataset, or pick a different "
        f"--output-dir."
    )


__all__ = ["preflight_dataset_output_dir"]
