#!/usr/bin/env python3
"""Export everything logged to a W&B run -- metadata, full scan history, system
metrics, and all run files (output.log, code snapshot, etc.) -- into a local
directory you can keep alongside the checkpoint mirror.

Why scan_history instead of history?
  ``run.history()`` returns a *sampled* time series capped at ~500 rows,
  which loses fidelity for a 25k-iter run. ``run.scan_history()`` is the
  unsampled iterator -- every step, every metric. Slow over the wire but
  faithful, and the correct thing for paper plots.

Outputs (under --out):
  metadata.json          run config, summary, state, tags, runtime, sweep info
  history.parquet        wide table: one row per logged step, all metrics
  history_long.parquet   long table: (step, key, value) -- robust to schema drift
  system.parquet         GPU/CPU/mem time series (10s cadence by default)
  files/                 every file in run.files() (output.log, requirements, etc.)
  history_keys.txt       sorted list of every metric key seen across history

Usage:

    python gear_sonic/scripts/cloud/export_wandb_run.py \
        --run meetsitaram/TRL_X2Ultra_BonesSeed_SphereFeet/z7docj57 \
        --out ~/x2_cloud_checkpoints/run-mirror-h200-sphere-feet-20260501_150437/wandb_export

The --run can be either of:
  - "<entity>/<project>/<run_id>"            (canonical)
  - "<entity>/<project>/runs/<run_id>"       (the URL path style)

Requires only ``wandb``, ``pandas``, and (optionally) ``pyarrow`` for parquet.
Falls back to csv.gz if pyarrow is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

try:
    import wandb
except ImportError as exc:  # pragma: no cover
    sys.exit(f"wandb is required: pip install wandb  ({exc})")

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover
    sys.exit(f"pandas is required: pip install pandas  ({exc})")


def parse_run_path(spec: str) -> str:
    """Normalize the run spec to the canonical 'entity/project/run_id' form."""
    spec = spec.strip().strip("/")
    spec = re.sub(r"/runs/", "/", spec)
    parts = spec.split("/")
    if len(parts) != 3:
        raise SystemExit(
            f"--run must be entity/project/run_id (or .../runs/run_id); got {spec!r}"
        )
    return "/".join(parts)


def write_table(rows: list[dict[str, Any]], path_no_ext: Path, label: str) -> Path:
    """Write a list-of-dict to parquet (preferred) or csv.gz (fallback)."""
    if not rows:
        path = path_no_ext.with_suffix(".empty")
        path.touch()
        print(f"  [{label}] no rows; touched {path}")
        return path
    df = pd.DataFrame(rows)
    try:
        path = path_no_ext.with_suffix(".parquet")
        df.to_parquet(path, index=False)
    except Exception as exc:  # pyarrow missing or schema issue
        path = path_no_ext.with_suffix(".csv.gz")
        df.to_csv(path, index=False, compression="gzip")
        print(f"  [{label}] parquet failed ({exc.__class__.__name__}); wrote {path}")
        return path
    print(f"  [{label}] wrote {path}  ({len(df):,} rows x {len(df.columns)} cols)")
    return path


def export_metadata(run, out: Path) -> None:
    """Run-level config + summary + provenance, single JSON."""
    meta = {
        "run_path": f"{run.entity}/{run.project}/{run.id}",
        "name": run.name,
        "id": run.id,
        "entity": run.entity,
        "project": run.project,
        "url": run.url,
        "state": run.state,
        "tags": list(run.tags or []),
        "notes": run.notes,
        "sweep_id": run.sweep.id if run.sweep else None,
        "created_at": str(run.created_at),
        "updated_at": str(getattr(run, "updated_at", "")),
        "runtime_s": run.summary.get("_runtime"),
        "step_max": run.summary.get("_step"),
        "config": dict(run.config),
        "summary": {k: v for k, v in dict(run.summary).items() if not k.startswith("_wandb")},
        "_wandb_summary": {k: v for k, v in dict(run.summary).items() if k.startswith("_wandb")},
        "system_metadata": run.metadata or {},  # GPU model, host, env, git sha, ...
    }
    path = out / "metadata.json"
    path.write_text(json.dumps(meta, indent=2, default=str))
    print(f"  [metadata] wrote {path}  state={run.state}  steps={meta['step_max']}")


def export_history(run, out: Path, page_size: int) -> None:
    """Full scan history (every logged step). Two flavors: wide and long."""
    print(f"  [history] streaming scan_history (page_size={page_size:,}; this may take minutes)")
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    last_print = 0
    for i, row in enumerate(run.scan_history(page_size=page_size)):
        rows.append(row)
        keys.update(row.keys())
        if i and i - last_print >= 5000:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(f"    ... {i + 1:,} rows pulled in {elapsed:0.1f}s ({rate:0.0f} rows/s)")
            last_print = i
    print(f"  [history] pulled {len(rows):,} rows total in {time.time() - t0:0.1f}s")

    # Wide table -- one row per step, lots of NaNs because not every metric is logged every step.
    write_table(rows, out / "history", "history")

    # Long table -- robust to schema drift; (step, key, value).
    long_rows = []
    for r in rows:
        step = r.get("_step")
        for k, v in r.items():
            if k.startswith("_") and k not in {"_step", "_runtime", "_timestamp"}:
                continue
            long_rows.append({"_step": step, "key": k, "value": v})
    write_table(long_rows, out / "history_long", "history_long")

    keys_path = out / "history_keys.txt"
    keys_path.write_text("\n".join(sorted(keys)))
    print(f"  [history] {len(keys)} unique keys -> {keys_path}")


def export_system(run, out: Path) -> None:
    """System metrics stream (GPU util, mem, cpu, disk) at ~10s cadence."""
    try:
        sys_rows = list(run.history(stream="system", samples=int(1e9), pandas=False))
    except Exception as exc:
        print(f"  [system] could not fetch system stream: {exc}")
        sys_rows = []
    write_table(sys_rows, out / "system", "system")


def export_files(run, out: Path, skip_globs: list[str]) -> None:
    """Download every file the run logged (output.log, code snapshot, etc.)."""
    files_dir = out / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    pulled = 0
    skipped = 0
    total_bytes = 0
    for f in run.files():
        # Skip patterns the user explicitly excluded.
        if any(re.search(g, f.name) for g in skip_globs):
            print(f"    [files] skip (matches --skip-files): {f.name}")
            skipped += 1
            continue
        try:
            f.download(root=str(files_dir), replace=False, exist_ok=True)
            pulled += 1
            total_bytes += getattr(f, "size", 0) or 0
        except Exception as exc:
            print(f"    [files] FAILED {f.name}: {exc}")
    print(
        f"  [files] downloaded {pulled} files ({total_bytes / 1e6:0.1f} MB)"
        f" into {files_dir}; skipped {skipped}"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="See module docstring for full output layout.",
    )
    p.add_argument(
        "--run",
        required=True,
        help='W&B run path: "entity/project/run_id" or ".../runs/run_id"',
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Local directory to write the export into (will be created)",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=10_000,
        help="scan_history page size (default: 10000); larger = fewer round trips",
    )
    p.add_argument(
        "--skip-files",
        nargs="*",
        default=[],
        help="Regex(es) of file paths to NOT download (e.g. 'output\\.log' to skip the 170MB stdout)",
    )
    p.add_argument(
        "--no-history",
        action="store_true",
        help="Skip the full scan_history dump (useful for a quick metadata-only pull)",
    )
    p.add_argument(
        "--no-files",
        action="store_true",
        help="Skip downloading run.files()",
    )
    p.add_argument(
        "--no-system",
        action="store_true",
        help="Skip the system metrics stream",
    )
    args = p.parse_args()

    run_path = parse_run_path(args.run)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[wandb] connecting via Public API for run={run_path}")
    api = wandb.Api(timeout=120)
    run = api.run(run_path)
    print(f"[wandb] resolved -> {run.url}")
    print(f"[out  ] {args.out.resolve()}")

    print("\n=== metadata ===")
    export_metadata(run, args.out)

    if not args.no_history:
        print("\n=== history ===")
        export_history(run, args.out, page_size=args.page_size)
    else:
        print("\n=== history === SKIPPED (--no-history)")

    if not args.no_system:
        print("\n=== system metrics ===")
        export_system(run, args.out)
    else:
        print("\n=== system metrics === SKIPPED (--no-system)")

    if not args.no_files:
        print("\n=== files ===")
        export_files(run, args.out, skip_globs=args.skip_files)
    else:
        print("\n=== files === SKIPPED (--no-files)")

    print("\n[done] all exports complete")
    print(f"[done] inspect with: ls -lh {args.out.resolve()}")


if __name__ == "__main__":
    main()
