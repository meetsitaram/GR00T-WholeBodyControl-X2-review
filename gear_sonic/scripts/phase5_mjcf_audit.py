#!/usr/bin/env python3
"""Phase 5 MJCF variant generator + audit driver.

Generates targeted MJCF variants for the sim2sim ablation study (Phase 5,
documented in docs/source/user_guide/sim2sim_ablation_study.md §5) by
patching the canonical x2_ultra.xml on a per-knob basis, then drives a
multi-init MuJoCo benchmark using ``record_x2_eval_mujoco.py --no-render``.

A "variant" is a one-axis edit of the canonical MJCF that probes a
deploy-side knob without an IsaacLab counterpart (e.g. solref, condim,
floor friction). The audit walks every (variant × motion × init_frame)
cell and tabulates time-to-fall.

Usage:
    # Generate all variants under /tmp/p5_mjcf/
    python gear_sonic/scripts/phase5_mjcf_audit.py --gen-variants

    # Run the audit (3 motions * 5 init frames * N variants in parallel)
    python gear_sonic/scripts/phase5_mjcf_audit.py --run \\
        --checkpoint /path/to/4k_sphere_ft.pt \\
        --variants baseline nofric floor_mu08 condim4 solref15 \\
        --parallel 4 \\
        --out-dir /tmp/p5_audit

The driver prints per-cell fall times and a final summary table.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MJCF_DIR = REPO / "gear_sonic" / "data" / "assets" / "robot_description" / "mjcf"
CANONICAL_MJCF = MJCF_DIR / "x2_ultra.xml"
ABS_MESHDIR = str((REPO / "gear_sonic" / "data" / "assets"
                   / "robot_description" / "urdf" / "x2_ultra" / "meshes").resolve())

VARIANT_DIR = Path("/tmp/p5_mjcf")

# Each variant is (name, list-of-(pattern, replacement) edits applied to canonical).
# The pattern is a literal string (str.replace), not regex, for auditability.
VARIANTS = {
    "baseline": [],  # canonical, used as the reference cell
    "nofric": [
        ('frictionloss="0.3"', 'frictionloss="0.0"'),
    ],
    "floor_mu08": [
        ('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
         '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane" '
         'friction="0.8 0.005 0.0001"/>'),
    ],
    "floor_mu12": [
        ('<geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>',
         '<geom name="floor" size="0 0 0.05" type="plane" material="groundplane" '
         'friction="1.2 0.005 0.0001"/>'),
    ],
    "condim4": [
        # Add condim=4 + torsional friction onto every foot sphere via the
        # class="foot" default block.
        ('<default class="foot">\n                  <geom type="sphere" size="0.005"/>',
         '<default class="foot">\n                  <geom type="sphere" size="0.005" '
         'condim="4" friction="1.0 0.05 0.0001"/>'),
    ],
    "solref15": [
        # Stiffer foot contact (15ms time constant vs default 20ms) — biases
        # toward kinematic contact, in the direction of PhysX mesh contact.
        ('<default class="foot">\n                  <geom type="sphere" size="0.005"/>',
         '<default class="foot">\n                  <geom type="sphere" size="0.005" '
         'solref="0.015 1"/>'),
    ],
    "solref10": [
        # Even stiffer foot contact (10ms time constant).
        ('<default class="foot">\n                  <geom type="sphere" size="0.005"/>',
         '<default class="foot">\n                  <geom type="sphere" size="0.005" '
         'solref="0.010 1"/>'),
    ],
    "nofric_solref15": [
        # Stack the two most promising single-axis edits.
        ('frictionloss="0.3"', 'frictionloss="0.0"'),
        ('<default class="foot">\n                  <geom type="sphere" size="0.005"/>',
         '<default class="foot">\n                  <geom type="sphere" size="0.005" '
         'solref="0.015 1"/>'),
    ],
}


def gen_variants():
    """Materialize every variant MJCF under VARIANT_DIR."""
    VARIANT_DIR.mkdir(parents=True, exist_ok=True)
    src = CANONICAL_MJCF.read_text()

    # Always rewrite meshdir to absolute so variants can live anywhere on disk.
    src = src.replace(
        'meshdir="../urdf/x2_ultra/meshes"',
        f'meshdir="{ABS_MESHDIR}"',
    )

    for name, edits in VARIANTS.items():
        out = src
        for pat, repl in edits:
            if pat not in out:
                raise RuntimeError(
                    f"Variant {name!r}: pattern not found in MJCF: {pat!r}")
            out = out.replace(pat, repl)
        out_path = VARIANT_DIR / f"x2_ultra_p5_{name}.xml"
        out_path.write_text(out)
        n_edits = len(edits)
        print(f"  wrote {out_path}  ({n_edits} edit{'s' if n_edits != 1 else ''})",
              flush=True)
    print(f"\n{len(VARIANTS)} variants generated in {VARIANT_DIR}\n", flush=True)


# Phase 3 motion suite — keep stable across audits for cross-comparison.
MOTIONS = {
    "icecream":   "/tmp/x2_ultra_icecream_only.pkl",
    "relaxed":    str(REPO / "gear_sonic/data/motions/x2_ultra_relaxed_walk_postfix.pkl"),
    "walkforward": str(REPO / "gear_sonic/data/motions/x2_ultra_walk_forward.pkl"),
}
INIT_FRAMES = [0, 10, 20, 30, 40]
DURATION = 15.0  # seconds per rollout

FALL_RE = re.compile(r"\[fall\] at step (\d+), t=([\d.]+)s")


def run_one(variant: str, motion_name: str, motion_pkl: str, init: int,
            ckpt: str, log_path: Path) -> tuple[str, str, int, float | None]:
    """Run one rollout; return (variant, motion_name, init, fall_time_s_or_None)."""
    mjcf = VARIANT_DIR / f"x2_ultra_p5_{variant}.xml"
    cmd = [
        "conda", "run", "-n", "env_isaaclab", "--no-capture-output",
        "python", str(REPO / "gear_sonic/scripts/record_x2_eval_mujoco.py"),
        "--mjcf", str(mjcf),
        "--checkpoint", ckpt,
        "--motion", motion_pkl,
        "--init-frame", str(init),
        "--duration", str(DURATION),
        "--out", "/dev/null",
        "--no-render",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    log_path.write_text(res.stdout + "\n--- STDERR ---\n" + res.stderr)
    wall = time.time() - t0
    fall_t: float | None = None
    for line in res.stdout.splitlines():
        m = FALL_RE.search(line)
        if m:
            fall_t = float(m.group(2))
            break
    if fall_t is None and res.returncode == 0:
        fall_t = DURATION  # survived
    print(f"  [{wall:5.1f}s wall]  {variant:20s} {motion_name:13s} "
          f"init={init:2d}  fall={fall_t}", flush=True)
    return variant, motion_name, init, fall_t


def run_audit(variants: list[str], ckpt: str, parallel: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Materialize the job list.
    jobs = []
    for v in variants:
        for mname, mpkl in MOTIONS.items():
            for init in INIT_FRAMES:
                log = out_dir / f"{v}_{mname}_init{init}.log"
                jobs.append((v, mname, mpkl, init, ckpt, log))

    print(f"\nRunning {len(jobs)} rollouts "
          f"({len(variants)} variants * {len(MOTIONS)} motions * "
          f"{len(INIT_FRAMES)} init-frames), parallel={parallel}\n", flush=True)

    results: list[tuple[str, str, int, float | None]] = []
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futs = [pool.submit(run_one, *j) for j in jobs]
        for f in as_completed(futs):
            results.append(f.result())

    # Tabulate: rows = (variant, motion), cols = init frames -> fall_time
    print("\n\n=== Phase 5 audit summary ===\n", flush=True)
    by_cell: dict[tuple[str, str], dict[int, float | None]] = {}
    for v, m, i, t in results:
        by_cell.setdefault((v, m), {})[i] = t

    header = (f"{'Variant':22s} {'Motion':13s}  " +
              "  ".join(f"i={i:02d}" for i in INIT_FRAMES) +
              f"   {'mean':>6s}  {'std':>5s}")
    print(header)
    print("-" * len(header))

    summary_rows = []
    for v in variants:
        for m in MOTIONS:
            cell = by_cell.get((v, m), {})
            vals = [cell.get(i) for i in INIT_FRAMES]
            num = [x for x in vals if x is not None]
            mean = sum(num) / len(num) if num else float("nan")
            var = (sum((x - mean) ** 2 for x in num) / len(num)) if num else 0.0
            std = var ** 0.5
            cell_str = "  ".join(f"{(x if x is not None else float('nan')):4.2f}" for x in vals)
            print(f"{v:22s} {m:13s}  {cell_str}   {mean:6.2f}  {std:5.2f}")
            summary_rows.append((v, m, vals, mean, std))

    print()
    # Write CSV for downstream paper plotting.
    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    csv = out_dir / f"summary_{ts}.csv"
    with csv.open("w") as f:
        f.write("variant,motion," + ",".join(f"init{i}" for i in INIT_FRAMES) +
                ",mean,std\n")
        for v, m, vals, mean, std in summary_rows:
            row = [v, m] + [f"{x:.3f}" if x is not None else "" for x in vals]
            row += [f"{mean:.3f}", f"{std:.3f}"]
            f.write(",".join(row) + "\n")
    print(f"Wrote {csv}\n", flush=True)
    latest = out_dir / "latest.csv"
    if latest.is_symlink() or latest.exists():
        latest.unlink()
    latest.symlink_to(csv.name)
    print(f"Updated symlink {latest} -> {csv.name}\n", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gen-variants", action="store_true",
                   help="(Re)generate all variant MJCFs under /tmp/p5_mjcf/.")
    p.add_argument("--list-variants", action="store_true",
                   help="Print the registered variants and exit.")
    p.add_argument("--run", action="store_true",
                   help="Execute the audit. Requires --checkpoint.")
    p.add_argument("--checkpoint", default=None,
                   help="Path to a SONIC .pt checkpoint to evaluate.")
    p.add_argument("--variants", nargs="+", default=["baseline"],
                   help="Subset of variants to audit. Default: just baseline.")
    p.add_argument("--parallel", type=int, default=4,
                   help="Concurrent rollouts. CPU MuJoCo handles ~4 well; "
                        "more starts oversaturating context-switching.")
    p.add_argument("--out-dir", default="/tmp/p5_audit",
                   help="Directory for per-cell logs and summary CSVs.")
    args = p.parse_args()

    if args.list_variants:
        for v, edits in VARIANTS.items():
            print(f"  {v}: {len(edits)} edit(s)")
            for pat, repl in edits:
                print(f"      - {pat!r}\n        => {repl!r}")
        return

    if args.gen_variants:
        gen_variants()

    if args.run:
        if not args.checkpoint:
            print("ERROR: --run requires --checkpoint", file=sys.stderr)
            sys.exit(2)
        unknown = [v for v in args.variants if v not in VARIANTS]
        if unknown:
            print(f"ERROR: unknown variant(s): {unknown}. "
                  f"Registered: {list(VARIANTS)}", file=sys.stderr)
            sys.exit(2)
        # Ensure variants exist.
        missing = [v for v in args.variants
                   if not (VARIANT_DIR / f"x2_ultra_p5_{v}.xml").exists()]
        if missing:
            print(f"Variants {missing} not yet generated; auto-generating all...",
                  flush=True)
            gen_variants()
        run_audit(args.variants, args.checkpoint, args.parallel,
                  Path(args.out_dir))


if __name__ == "__main__":
    main()
