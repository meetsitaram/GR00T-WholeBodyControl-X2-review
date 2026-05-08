#!/usr/bin/env python3
"""Run the IsaacLab <-> MuJoCo mirror ablation sweep for X2 Ultra.

For each (checkpoint, ablation_row) combination, this driver:

  1. Builds Hydra overrides that enable the relevant MuJoCo-mirroring knob.
  2. Subprocesses ``gear_sonic/eval_agent_trl.py`` headless with the
     ``im_eval`` callback so it writes ``metrics_eval.json`` for the run.
  3. Parses the resulting metrics, extracts mean / per-motion progress and
     success rates, and appends a row to a CSV at
     ``$OUT_DIR/sweep_results.csv``.

The output CSV is what feeds the ablation table in
``SUMMARY_isaac_mujoco_mirror.md`` and ``docs/source/user_guide/sim2sim_mujoco.md``
(section G18).

Usage (from repo root):

    python gear_sonic/scripts/sweep_isaac_mujoco_mirror.py \
        --checkpoints 002000 006000 016000 \
        --motion gear_sonic/data/motions/x2_ultra_top15_standing.pkl \
        --out-dir /home/stickbot/sim2sim_armature_eval/isaaclab_mujoco_mirror

Useful flags:
    --rows A0 A1 A2 A3 A4 A5    # subset of rows to run
    --num-envs 15               # number of parallel envs (= number of motions)
    --dry-run                   # print commands, do not execute
    --skip-existing             # skip cells whose metrics_eval.json already exists
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_CHECKPOINT_ROOT = Path("/home/stickbot/x2_cloud_checkpoints/run-20260420_083925")
DEFAULT_MOTION = Path("gear_sonic/data/motions/x2_ultra_top15_standing.pkl")
DEFAULT_OUT_DIR = Path("/home/stickbot/sim2sim_armature_eval/isaaclab_mujoco_mirror")
DEFAULT_CONDA_ENV = "env_isaaclab"

# Events that the level0_4 composition introduces. We strip these for the
# "DR off" ablations so the policy gets a deterministic Isaac. Names must
# match the keys defined in
# gear_sonic/config/manager_env/events/tracking/level0_4.yaml.
DR_EVENTS = [
    "push_robot",
    "compliance_force_push",
    "physics_material",
    "randomize_rigid_body_mass",
    "base_com",
    "add_joint_default_pos",
]


@dataclasses.dataclass
class AblationRow:
    """A single row in the IsaacLab <-> MuJoCo mirror ablation table."""

    name: str
    description: str
    # Extra Hydra overrides this row introduces, beyond the base eval cmdline.
    overrides: list[str]


# Base overrides applied to EVERY cell so the diagnostic is reproducible:
#   - im_eval callback writes metrics_eval.json
#   - run_eval_loop=False so the eval is driven entirely by im_eval
#   - terrain_type=plane (matches MuJoCo plane floor; X2 normally trains on
#     trimesh, which adds height-noise that masks the physics axis we care
#     about). Use a Hydra override to keep the change local to this sweep.
#   - motion_lib_cfg.multi_thread=False (matches eval_exp.py production runs)
BASE_OVERRIDES = [
    "+headless=True",
    "++eval_callbacks=im_eval",
    "++run_eval_loop=False",
    "++manager_env.commands.motion.motion_lib_cfg.multi_thread=False",
    "++manager_env.config.terrain_type=plane",
    "+manager_env/terminations=tracking/eval",
]


def _drop_dr_events_override() -> str:
    # train_only_events pop happens in eval_agent_trl.py before instantiate.
    # The list must be valid YAML inside a Hydra ++ override → bracket-list.
    items = ",".join(DR_EVENTS)
    return f"++manager_env.config.train_only_events=[{items}]"


# Six rows go from "Isaac as trained" toward "Isaac with all MuJoCo physics
# knobs in place." See sim2sim_mujoco.md (section G18) and the cited
# MUJOCO_REFERENCE.md for the exact MuJoCo numbers each knob mirrors.
ROWS: list[AblationRow] = [
    AblationRow(
        name="A0_isaac_stock",
        description="Isaac stock eval (DR + obs noise ON) — upper bound, model as trained",
        overrides=[],
    ),
    AblationRow(
        name="A1_no_dr_no_noise",
        description="A0 minus DR events and observation noise — Isaac stripped down",
        overrides=[
            _drop_dr_events_override(),
            "++manager_env.observations.policy.enable_corruption=False",
        ],
    ),
    AblationRow(
        name="A2_frictionloss",
        description="A1 + joint frictionloss=0.3 N.m (mirrors MJCF)",
        overrides=[
            _drop_dr_events_override(),
            "++manager_env.observations.policy.enable_corruption=False",
            "++manager_env.config.robot.frictionloss=0.3",
        ],
    ),
    AblationRow(
        name="A3_sphere_feet",
        description="A1 + 12-sphere foot URDF (mirrors MJCF foot geometry exactly)",
        overrides=[
            _drop_dr_events_override(),
            "++manager_env.observations.policy.enable_corruption=False",
            "++manager_env.config.robot.foot=sphere",
        ],
    ),
    AblationRow(
        name="A4_explicit_pd",
        description="A1 + IdealPDActuatorCfg + ankle KP x1.5 (G5 + G16b)",
        overrides=[
            _drop_dr_events_override(),
            "++manager_env.observations.policy.enable_corruption=False",
            "++manager_env.config.robot.actuator_regime=explicit",
            "++manager_env.config.robot.ankle_kp_scale=1.5",
        ],
    ),
    AblationRow(
        name="A5_full_mirror",
        description="A1 + frictionloss + sphere feet + explicit PD + ankle x1.5",
        overrides=[
            _drop_dr_events_override(),
            "++manager_env.observations.policy.enable_corruption=False",
            "++manager_env.config.robot.frictionloss=0.3",
            "++manager_env.config.robot.foot=sphere",
            "++manager_env.config.robot.actuator_regime=explicit",
            "++manager_env.config.robot.ankle_kp_scale=1.5",
        ],
    ),
]


def _build_command(
    checkpoint_path: Path,
    motion_path: Path,
    work_dir: Path,
    num_envs: int,
    extra_overrides: list[str],
    conda_env: str | None,
) -> list[str]:
    base = []
    if conda_env:
        base = ["conda", "run", "-n", conda_env, "--no-capture-output"]
    base.append("accelerate")
    base.append("launch")
    base.append("gear_sonic/eval_agent_trl.py")
    base.append(f"+checkpoint={checkpoint_path}")
    base.append(f"++num_envs={num_envs}")
    base.append(f"++eval_output_dir={work_dir}")
    base.append(
        f"+manager_env.commands.motion.motion_lib_cfg.motion_file={motion_path}"
    )
    base.extend(BASE_OVERRIDES)
    base.extend(extra_overrides)
    return base


def _parse_metrics(metrics_path: Path) -> dict[str, float]:
    with metrics_path.open() as f:
        data = json.load(f)
    out: dict[str, float] = {}

    # Top-line aggregates.
    for src_key, csv_key in (
        ("eval/success/success_rate", "success_rate"),
        ("eval/success/progress_rate", "progress_rate"),
        ("eval/all/mpjpe_l", "mpjpe_l_all"),
        ("eval/all/mpjpe_g", "mpjpe_g_all"),
        ("eval/success/mpjpe_l", "mpjpe_l_succ"),
        ("eval/success/mpjpe_g", "mpjpe_g_succ"),
    ):
        v = data.get(src_key)
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        out[csv_key] = float(v) if v is not None else float("nan")

    # Per-motion: mean progress over all motions in this run.
    md = data.get("eval/all_metrics_dict") or {}
    progress = md.get("progress")
    terminated = md.get("terminated")
    if progress is not None:
        progress = [float(p) for p in progress]
        out["progress_mean"] = sum(progress) / max(len(progress), 1)
        out["progress_min"] = min(progress) if progress else float("nan")
        out["progress_max"] = max(progress) if progress else float("nan")
        out["num_motions"] = len(progress)
    if terminated is not None:
        terminated_b = [bool(t) for t in terminated]
        out["terminated_frac"] = sum(terminated_b) / max(len(terminated_b), 1)
    return out


def _run_cell(
    row: AblationRow,
    checkpoint_path: Path,
    motion_path: Path,
    out_dir: Path,
    num_envs: int,
    conda_env: str | None,
    timeout_seconds: int,
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, str | float]:
    step = checkpoint_path.stem.split("_")[-1]
    work_dir = out_dir / row.name / f"step_{step}"
    work_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = work_dir / "metrics_eval.json"
    log_path = work_dir / "run.log"
    cell_csv_path = work_dir / "summary.json"

    cmd = _build_command(
        checkpoint_path=checkpoint_path,
        motion_path=motion_path,
        work_dir=work_dir,
        num_envs=num_envs,
        extra_overrides=row.overrides,
        conda_env=conda_env,
    )

    cmd_str = " ".join(cmd)
    print(f"\n[{row.name} step={step}] working_dir={work_dir}")
    print(f"  cmd: {cmd_str}")

    base_row: dict[str, str | float] = {
        "row": row.name,
        "step": step,
        "checkpoint": str(checkpoint_path),
        "motion": str(motion_path),
        "work_dir": str(work_dir),
    }

    if dry_run:
        base_row["status"] = "dry-run"
        return base_row

    if skip_existing and metrics_path.exists():
        print("  [skip] metrics_eval.json already exists")
        try:
            metrics = _parse_metrics(metrics_path)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip-error] failed to parse existing metrics: {e}")
            metrics = {}
        base_row.update({"status": "skipped"})
        base_row.update(metrics)
        return base_row

    started = time.time()
    with log_path.open("w") as log_f:
        log_f.write(f"# cmd: {cmd_str}\n")
        log_f.flush()
        try:
            proc = subprocess.run(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_seconds,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            log_f.write(f"\n# TIMEOUT after {timeout_seconds}s\n")
            returncode = -1
    elapsed = time.time() - started

    base_row["elapsed_s"] = round(elapsed, 1)
    base_row["returncode"] = returncode

    if returncode != 0:
        print(f"  [FAIL] returncode={returncode} elapsed={elapsed:.0f}s log={log_path}")
        base_row["status"] = "failed"
        return base_row

    if not metrics_path.exists():
        print(f"  [FAIL] no metrics_eval.json at {metrics_path}")
        base_row["status"] = "no-metrics"
        return base_row

    try:
        metrics = _parse_metrics(metrics_path)
    except Exception as e:  # noqa: BLE001
        print(f"  [FAIL] metrics parse error: {e}")
        base_row["status"] = "metrics-parse-error"
        return base_row

    base_row["status"] = "ok"
    base_row.update(metrics)

    with cell_csv_path.open("w") as f:
        json.dump(base_row, f, indent=2)
    print(
        f"  [OK] elapsed={elapsed:.0f}s "
        f"progress_rate={metrics.get('progress_rate', 'nan'):.3f} "
        f"success_rate={metrics.get('success_rate', 'nan'):.3f} "
        f"progress_mean={metrics.get('progress_mean', float('nan')):.3f}"
    )
    return base_row


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint-root",
        type=Path,
        default=DEFAULT_CHECKPOINT_ROOT,
        help="Directory with model_step_NNNNNN.pt files",
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        default=["002000", "006000", "016000"],
        help="Checkpoint step suffixes (e.g. 002000 006000 016000)",
    )
    p.add_argument(
        "--motion",
        type=Path,
        default=DEFAULT_MOTION,
        help="Motion .pkl file to evaluate against",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Top-level output directory",
    )
    p.add_argument("--num-envs", type=int, default=15)
    p.add_argument(
        "--rows",
        nargs="+",
        default=None,
        help="Subset of row names to run (default = all)",
    )
    p.add_argument("--conda-env", default=DEFAULT_CONDA_ENV)
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=2400,
        help="Timeout per eval cell in seconds (default 40 min)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a cell if its metrics_eval.json already exists",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = ROWS
    if args.rows:
        keep = set(args.rows)
        rows = [r for r in rows if r.name in keep]
        if not rows:
            print(f"No rows match {args.rows}; available: {[r.name for r in ROWS]}")
            return 1

    checkpoints: list[Path] = []
    for step in args.checkpoints:
        cp = args.checkpoint_root / f"model_step_{step}.pt"
        if not cp.exists():
            print(f"Missing checkpoint: {cp}")
            return 1
        checkpoints.append(cp)

    # Each invocation gets its own timestamped CSV so reruns never clobber
    # earlier results. The `latest.csv` symlink always points at the most
    # recent invocation for quick "show me the last sweep" access. Per-cell
    # directories (<row>/step_<step>/) still get reused across runs so
    # --skip-existing can resume an interrupted sweep without re-launching
    # IsaacSim for already-finished cells.
    rows_tag = "_".join(r.name.split("_")[0] for r in rows) if args.rows else "all"
    steps_tag = "+".join(args.checkpoints)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    sweep_csv_path = args.out_dir / f"sweep_{timestamp}_{rows_tag}_{steps_tag}.csv"
    latest_link = args.out_dir / "latest.csv"

    if not args.motion.exists():
        print(f"Missing motion file: {args.motion}")
        return 1

    print(f"Output dir:    {args.out_dir}")
    print(f"Sweep CSV:     {sweep_csv_path}")
    print(f"Checkpoints:   {[cp.name for cp in checkpoints]}")
    print(f"Rows:          {[r.name for r in rows]}")
    print(f"Motion:        {args.motion}")
    print(f"num_envs:      {args.num_envs}")
    print(f"timeout/cell:  {args.timeout_seconds}s")
    print(f"dry_run:       {args.dry_run}")

    # Stream into CSV as we go so a crash mid-sweep still leaves usable data.
    fieldnames = [
        "row",
        "step",
        "status",
        "elapsed_s",
        "returncode",
        "progress_rate",
        "success_rate",
        "progress_mean",
        "progress_min",
        "progress_max",
        "terminated_frac",
        "num_motions",
        "mpjpe_l_all",
        "mpjpe_g_all",
        "mpjpe_l_succ",
        "mpjpe_g_succ",
        "checkpoint",
        "motion",
        "work_dir",
    ]
    with sweep_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        f.flush()
        for row in rows:
            for cp in checkpoints:
                result = _run_cell(
                    row=row,
                    checkpoint_path=cp,
                    motion_path=args.motion,
                    out_dir=args.out_dir,
                    num_envs=args.num_envs,
                    conda_env=args.conda_env,
                    timeout_seconds=args.timeout_seconds,
                    dry_run=args.dry_run,
                    skip_existing=args.skip_existing,
                )
                writer.writerow(result)
                f.flush()

    if not args.dry_run:
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(sweep_csv_path.name)
        except OSError as e:
            print(f"Could not update latest.csv symlink: {e}")

    print(f"\nDone. Sweep CSV: {sweep_csv_path}")
    print(f"Latest symlink: {latest_link} -> {sweep_csv_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
