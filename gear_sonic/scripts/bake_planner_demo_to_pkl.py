"""Capture a planner demo run as a single deploy-format PKL.

Why this exists
---------------

The single-bin diagnostic
(``gear_sonic.scripts.diagnose_planner_vs_pkl``) proved the planner's
emitted joint trajectory for ``side_left_step`` is bit-identical to
what ``deploy_x2.sh sim --motion x2_browser_side_left_step.pkl``
replays from PKL -- and that PKL path makes the body actually side-step.

What we have NOT yet tested is the **composed** sequence: idle ->
blend -> side_left -> idle (settle) -> blend -> side_right -> idle,
which the planner stitches together in real time over a demo YAML.
The auto-blend-to-idle settles, the per-bin blend windows, and the
bin-to-bin transitions are all unique to the planner state machine
and don't appear in any individual single-bin PKL.

This script runs a demo YAML through the planner offline (no ZMQ, no
deploy, no MuJoCo), captures every 50 Hz emitted frame, and saves
the full ``(T, 31)`` joint trajectory + ``(T, 4)`` root quaternion +
``(T, 3)`` root translation as a single PKL in the format
``deploy_x2.sh sim --motion <pkl>`` already knows how to play.

You can then feed that PKL straight to the deploy:

    bash gear_sonic_deploy/deploy_x2.sh sim --no-confirm \\
        --motion <baked.pkl> --model <onnx> \\
        --sim-profile parity --sim-viewer

If the body **side-steps under the baked PKL** but does NOT side-step
under the live planner ZMQ stream, the bug is in the wire path
(ZmqPoseInputSource ingest, no ``Anchor()`` call, velocity finite-diff
jitter, etc.). If the body **also fails under the baked PKL**, the
bug is in how the demo is composed (blend windows too short, hold
poses physically unreachable, etc.).

Usage::

    .venv/bin/python -m gear_sonic.scripts.bake_planner_demo_to_pkl \\
        --demo gear_sonic/data/scripted_demos/side_steps_only_smoke.yaml \\
        --out data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_demo_side_steps.pkl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.planner.registry import load_bin_specs  # noqa: E402
from gear_sonic.utils.planner.state_machine import (  # noqa: E402
    HeuristicPlanner,
    LocomotionCommand,
    OUTPUT_FPS,
    PlannerState,
    commands_from_yaml,
    load_primitives_pkl,
)

_DEFAULT_PRIMS_PKL = (
    _REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_primitives.pkl"
)
_DEFAULT_BINS_YAML = (
    _REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_bins.yaml"
)
_DEFAULT_BAKED_DIR = (
    _REPO_ROOT / "data" / "sim_to_real_anchors" / "browse_sonic" / "baked_pkls"
)

# Per-command upper bound on ticks the planner is allowed to consume.
# The state machine's per-bin frame counts are bounded (no bin exceeds
# 12 s), and end-of-queue idle is finite by construction. A 60 s cap
# gives generous headroom and prevents pathological infinite blends
# from running away.
_MAX_TICKS_PER_COMMAND = 60 * OUTPUT_FPS


def _drain_planner_through_demo(
    planner: HeuristicPlanner,
    commands: list[LocomotionCommand],
    pre_idle_s: float,
    trail_idle_s: float,
) -> list[dict]:
    """Run the planner offline, queue all commands, capture every frame.

    We mirror what the live ``x2_heuristic_planner`` script does, but
    without ZMQ, threading, or wall-clock timing -- pure state-machine
    iteration. The result is the sequence of frames the wire would carry
    in real time.
    """
    pre_n = int(round(max(0.0, pre_idle_s) * OUTPUT_FPS))
    trail_n = int(round(max(0.0, trail_idle_s) * OUTPUT_FPS))

    log: list[dict] = []
    tick = 0

    # --- Pre-idle: establish a stable baseline so the first few wire
    # frames look like the warmup the live planner would publish.
    for _ in range(pre_n):
        f = planner.step()
        log.append(_record(f, tick))
        tick += 1

    # --- Enqueue every command up front; the state machine drains them
    # in order and inserts blends between bins itself.
    for cmd in commands:
        planner.enqueue(cmd)

    # --- Step until the queue is fully drained AND the planner has
    # returned to IDLE_LOOP. Cap each command's ticks so a stuck bin
    # can't run forever.
    quiescence_target = len(commands) * _MAX_TICKS_PER_COMMAND
    quiescent_idle = 0
    while quiescent_idle < trail_n and tick < quiescence_target:
        f = planner.step()
        log.append(_record(f, tick))
        tick += 1
        # Count consecutive idle ticks at end -- once we've sat in
        # IDLE_LOOP for `trail_n` ticks AFTER all commands consumed,
        # we're done.
        if (
            planner.queue_depth == 0
            and f.state == PlannerState.IDLE_LOOP
        ):
            quiescent_idle += 1
        else:
            quiescent_idle = 0

    return log


def _record(frame, tick: int) -> dict:
    return {
        "tick": tick,
        "state": frame.state.value,
        "bin_name": frame.bin_name,
        "seam_blend": bool(frame.seam_blend),
        "joint_pos_mj": frame.joint_pos_mj.copy(),
        "root_quat_xyzw": frame.root_quat_xyzw.copy(),
        "root_xy_world": frame.root_xy_world.copy(),
        "yaw_world_deg": float(frame.yaw_world_deg),
    }


def _stack_log_to_pkl(
    log: list[dict],
    pelvis_z_m: float,
    motion_name: str,
) -> dict:
    """Stack frames_log into the ``deploy_x2.sh --motion`` PKL schema.

    The deploy reads:
      - ``dof``               : (T, 31) float64 joint positions
      - ``root_rot``          : (T, 4) xyzw quaternions
      - ``root_trans_offset`` : (T, 3) world XYZ
      - ``fps``               : float
    The C++ ``PklMotionReference`` only loads ``dof`` + ``root_rot`` (root
    translation isn't part of the policy obs), but the Python bridge's
    RSI uses ``root_trans_offset`` to set the spawn pose, so we must
    write a sensible Z value.
    """
    dof = np.stack([f["joint_pos_mj"] for f in log]).astype(np.float64)
    rot = np.stack([f["root_quat_xyzw"] for f in log]).astype(np.float64)
    xy = np.stack([f["root_xy_world"] for f in log]).astype(np.float64)
    z_col = np.full((dof.shape[0], 1), pelvis_z_m, dtype=np.float64)
    trans = np.concatenate([xy, z_col], axis=1)

    return {
        motion_name: {
            "dof": dof,
            "root_rot": rot,
            "root_trans_offset": trans,
            "fps": float(OUTPUT_FPS),
        }
    }


def _summarize(log: list[dict]) -> str:
    """Compact transition timeline (one row per state/bin change)."""
    rows = [
        f"  {'tick':>6}  {'t (s)':>7}  {'state':<10}  {'bin':<28}"
    ]
    rows.append("  " + "-" * 60)
    prev_state = None
    prev_bin = None
    for f in log:
        if f["state"] != prev_state or f["bin_name"] != prev_bin:
            rows.append(
                f"  {f['tick']:>6}  {f['tick']/OUTPUT_FPS:>7.2f}  "
                f"{f['state']:<10}  {f['bin_name']:<28}"
            )
            prev_state = f["state"]
            prev_bin = f["bin_name"]
    return "\n".join(rows)


def bake_demo(
    demo_yaml: Path,
    out_pkl: Path,
    primitives_pkl: Path = _DEFAULT_PRIMS_PKL,
    bins_yaml: Path = _DEFAULT_BINS_YAML,
    pre_idle_s: float = 2.0,
    trail_idle_s: float = 1.0,
    motion_name: str | None = None,
) -> dict:
    if not demo_yaml.is_file():
        raise FileNotFoundError(f"demo YAML not found: {demo_yaml}")
    bin_specs = load_bin_specs(bins_yaml)
    bin_family = {name: spec.family for name, spec in bin_specs.items()}
    primitives = load_primitives_pkl(primitives_pkl, bin_family)

    planner = HeuristicPlanner(primitives=primitives)
    anchor = planner.current_anchor_frame()
    pelvis_z = float(planner._active.aligned_trans[0, 2])  # noqa: SLF001

    commands = commands_from_yaml(demo_yaml)
    if not commands:
        raise ValueError(f"demo YAML had no commands: {demo_yaml}")

    log = _drain_planner_through_demo(
        planner, commands, pre_idle_s=pre_idle_s, trail_idle_s=trail_idle_s
    )
    if not log:
        raise RuntimeError("planner emitted zero frames; check the demo YAML.")

    name = motion_name or f"planner_demo__{demo_yaml.stem}"
    payload = _stack_log_to_pkl(log, pelvis_z, name)
    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_pkl)

    n = len(log)
    return {
        "out_pkl": str(out_pkl),
        "motion_name": name,
        "n_frames": n,
        "fps": OUTPUT_FPS,
        "duration_s": n / OUTPUT_FPS,
        "pelvis_z_anchor": pelvis_z,
        "anchor_quat_xyzw": anchor.root_quat_xyzw.tolist(),
        "summary": _summarize(log),
        "n_commands_planned": len(commands),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--demo", type=Path, required=True,
        help="Scripted demo YAML to play through the planner."
    )
    p.add_argument(
        "--out", type=Path, default=None,
        help="Output PKL path (default: "
             "data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_demo_<stem>.pkl)"
    )
    p.add_argument("--primitives-pkl", type=Path, default=_DEFAULT_PRIMS_PKL)
    p.add_argument("--bins-yaml", type=Path, default=_DEFAULT_BINS_YAML)
    p.add_argument(
        "--pre-idle-s", type=float, default=2.0,
        help="Seconds of idle to play BEFORE enqueuing the demo's commands. "
             "Mirrors the live planner's quiet-stand warmup."
    )
    p.add_argument(
        "--trail-idle-s", type=float, default=1.0,
        help="Seconds of idle to capture AFTER the queue is drained. Lets "
             "the policy ride out the final blend back to idle."
    )
    p.add_argument(
        "--motion-name", type=str, default=None,
        help="Override the dict key inside the PKL (default: "
             "planner_demo__<demo-stem>)."
    )
    args = p.parse_args()

    out = args.out or (_DEFAULT_BAKED_DIR / f"x2_planner_demo_{args.demo.stem}.pkl")

    info = bake_demo(
        demo_yaml=args.demo,
        out_pkl=out,
        primitives_pkl=args.primitives_pkl,
        bins_yaml=args.bins_yaml,
        pre_idle_s=args.pre_idle_s,
        trail_idle_s=args.trail_idle_s,
        motion_name=args.motion_name,
    )
    print(f"[bake-demo] wrote {info['out_pkl']}")
    print(f"           motion key      = {info['motion_name']}")
    print(f"           n_frames        = {info['n_frames']}  "
          f"({info['duration_s']:.2f}s @ {info['fps']:.0f}Hz)")
    print(f"           pelvis_z anchor = {info['pelvis_z_anchor']:.3f} m")
    print(f"           commands queued = {info['n_commands_planned']}")
    print()
    print("--- timeline (one row per state/bin change) ---")
    print(info["summary"])
    print()
    print("# Replay via the proven --motion path:")
    print(
        f"bash {_REPO_ROOT}/gear_sonic_deploy/deploy_x2.sh sim --no-confirm \\\n"
        f"    --motion {info['out_pkl']} \\\n"
        "    --model <PATH_TO_DEPLOY.onnx> \\\n"
        "    --sim-profile parity \\\n"
        f"    --sim-viewer --max-duration {int(info['duration_s']) + 5}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
