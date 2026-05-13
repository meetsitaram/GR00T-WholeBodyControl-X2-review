"""A/B telemetry capture: planner ZMQ stream vs deploy ``--motion <pkl>``.

What this script does
---------------------

For each "path" you pass on the command line, it:

  1. Spawns the appropriate sim:
       PATH A (``--motion``) : ``deploy_x2.sh sim --motion <baked.pkl>``
       PATH B (``--planner``): ``run_planner_smoke.sh --with-deploy --demo <yaml>``
  2. Concurrently subscribes to two ZMQ topics the C++ deploy + the
     bridge already publish:
       * ``x2_debug``   (port 5557): per-tick joint commands + measurements
       * ``robot_pose`` (port 5570): ground-truth pelvis qpos from MuJoCo
  3. Saves both streams to NPZ files keyed by path name.
  4. After both runs finish, prints a side-by-side table showing for each
     path:
       * pelvis displacement vs starting xy
       * pelvis path length (cumulative travel)
       * pelvis Z range (min/max)
       * per-joint range (max - min, deg) for the 13 leg+waist drive joints
       * mean joint tracking error |target - measured| (deg)

This makes the visible difference between PklMotionReference (the
proven path) and ZmqPoseInputSource (the planner path) numerically
obvious without needing to record video.

Run::

    .venv/bin/python -m gear_sonic.scripts.compare_planner_vs_motion \\
        --motion-pkl data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_demo_side_steps_only_smoke.pkl \\
        --planner-demo gear_sonic/data/scripted_demos/side_steps_only_smoke.yaml \\
        --duration 14

The two paths are run sequentially (NOT in parallel) so they don't
fight over the deploy docker, ports, or the X11 viewer.
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zmq

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)
from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose  # noqa: E402

_DEPLOY_SH = _REPO_ROOT / "gear_sonic_deploy" / "deploy_x2.sh"
_SMOKE_SH = _REPO_ROOT / "gear_sonic" / "scripts" / "run_planner_smoke.sh"
_DEFAULT_MODEL = Path(
    "/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/"
    "exported/model_step_025000_g1.onnx"
)
_DEFAULT_OUT_DIR = (
    _REPO_ROOT / "data" / "sim_to_real_anchors" / "planner_smoke" / "ab_compare"
)

# Joints we care about for side-step / locomotion analysis.
DRIVE_JOINTS: dict[str, int] = {
    "L_hip_pitch": 0,
    "L_hip_roll": 1,
    "L_hip_yaw": 2,
    "L_knee": 3,
    "L_ankle_pitch": 4,
    "L_ankle_roll": 5,
    "R_hip_pitch": 6,
    "R_hip_roll": 7,
    "R_hip_yaw": 8,
    "R_knee": 9,
    "R_ankle_pitch": 10,
    "R_ankle_roll": 11,
    "waist_yaw": 12,
    "waist_roll": 14,
}


# ---------------------------------------------------------------------------
# Telemetry capture (background subscribers)
# ---------------------------------------------------------------------------

@dataclass
class CapturedRun:
    label: str
    debug_recv_t: list[float]
    debug_body_q_target: list[np.ndarray]   # per-tick (31,) double, may be empty
    debug_body_q_measured: list[np.ndarray]
    pose_recv_t: list[float]
    pelvis_xyz: list[np.ndarray]            # per-tick (3,) world pos
    pelvis_quat_wxyz: list[np.ndarray]      # per-tick (4,)


def _subscriber_loop(
    label: str,
    duration_s: float,
    debug_port: int,
    debug_topic: str,
    pose_port: int,
    pose_topic: str,
    out: dict,
) -> None:
    """Run inside a subprocess; saves a JSON+NPZ to ``out['npz_path']``."""
    ctx = zmq.Context()
    debug_sock = ctx.socket(zmq.SUB)
    debug_sock.setsockopt_string(zmq.SUBSCRIBE, debug_topic)
    debug_sock.setsockopt(zmq.RCVHWM, 200)
    debug_sock.setsockopt(zmq.LINGER, 0)
    debug_sock.connect(f"tcp://127.0.0.1:{debug_port}")

    pose_sock = ctx.socket(zmq.SUB)
    pose_sock.setsockopt_string(zmq.SUBSCRIBE, pose_topic)
    pose_sock.setsockopt(zmq.RCVHWM, 200)
    pose_sock.setsockopt(zmq.LINGER, 0)
    pose_sock.connect(f"tcp://127.0.0.1:{pose_port}")

    poller = zmq.Poller()
    poller.register(debug_sock, zmq.POLLIN)
    poller.register(pose_sock, zmq.POLLIN)

    t0 = time.monotonic()
    deadline = t0 + duration_s
    debug_recv_t: list[float] = []
    debug_body_q_target: list[list[float]] = []
    debug_body_q_measured: list[list[float]] = []
    pose_recv_t: list[float] = []
    pelvis_xyz: list[list[float]] = []
    pelvis_quat_wxyz: list[list[float]] = []

    while time.monotonic() < deadline:
        events = dict(poller.poll(50))
        now = time.monotonic() - t0
        if debug_sock in events:
            try:
                raw = debug_sock.recv(zmq.NOBLOCK)
                msg = unpack_message(raw, expected_topic=debug_topic)
                # x2_debug field names per
                # gear_sonic_deploy/src/.../x2_deploy_onnx_ref.cpp PublishDebug():
                #   "body_q"      = measured joint positions (this tick)
                #   "last_action" = commanded joint targets (this tick, post-ramp)
                meas = msg.fields.get("body_q")
                tgt  = msg.fields.get("last_action")
                if tgt is not None and meas is not None:
                    debug_recv_t.append(now)
                    debug_body_q_target.append(np.asarray(tgt, dtype=np.float64).tolist())
                    debug_body_q_measured.append(np.asarray(meas, dtype=np.float64).tolist())
            except zmq.Again:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"[{label}] debug decode err: {exc}", flush=True)
        if pose_sock in events:
            try:
                raw = pose_sock.recv(zmq.NOBLOCK)
                payload = unpack_robot_pose(raw)
                qpos = payload.get("pelvis_qpos_wxyz")
                if qpos is not None and len(qpos) == 7:
                    pose_recv_t.append(now)
                    pelvis_xyz.append([float(qpos[0]), float(qpos[1]), float(qpos[2])])
                    pelvis_quat_wxyz.append(
                        [float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6])]
                    )
            except zmq.Again:
                pass
            except Exception as exc:  # noqa: BLE001
                print(f"[{label}] pose decode err: {exc}", flush=True)

    debug_sock.close(linger=0)
    pose_sock.close(linger=0)
    ctx.term()

    np.savez(
        out["npz_path"],
        label=label,
        debug_recv_t=np.asarray(debug_recv_t, dtype=np.float64),
        debug_body_q_target=np.asarray(debug_body_q_target, dtype=np.float64)
            if debug_body_q_target else np.zeros((0, 31), dtype=np.float64),
        debug_body_q_measured=np.asarray(debug_body_q_measured, dtype=np.float64)
            if debug_body_q_measured else np.zeros((0, 31), dtype=np.float64),
        pose_recv_t=np.asarray(pose_recv_t, dtype=np.float64),
        pelvis_xyz=np.asarray(pelvis_xyz, dtype=np.float64)
            if pelvis_xyz else np.zeros((0, 3), dtype=np.float64),
        pelvis_quat_wxyz=np.asarray(pelvis_quat_wxyz, dtype=np.float64)
            if pelvis_quat_wxyz else np.zeros((0, 4), dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Sim spawners (sequential)
# ---------------------------------------------------------------------------

def _wait_for_marker(log_path: Path, marker: str, pid: int, timeout_s: int) -> bool:
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        if not _pid_alive(pid):
            return False
        if log_path.is_file():
            try:
                txt = log_path.read_text(errors="ignore")
            except OSError:
                txt = ""
            if marker in txt:
                return True
        time.sleep(0.5)
    return False


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill(pid: int) -> None:
    if not _pid_alive(pid):
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    for _ in range(20):
        if not _pid_alive(pid):
            return
        time.sleep(0.25)
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _run_path(
    label: str,
    spawn_cmd: list[str],
    log_path: Path,
    duration_s: float,
    capture_npz: Path,
    debug_port: int = 5557,
    pose_port: int = 5570,
    ready_marker: str = "Launching ...",
    settle_s: float = 3.0,
) -> CapturedRun:
    print(f"\n{'='*78}\n  RUN {label}\n{'='*78}")
    print(f"  cmd: {' '.join(spawn_cmd)}")
    print(f"  log: {log_path}")
    print(f"  capture: {capture_npz}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    capture_npz.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as logf:
        proc = subprocess.Popen(
            spawn_cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
    print(f"  pid={proc.pid}; waiting up to 180s for '{ready_marker}'...")
    if not _wait_for_marker(log_path, ready_marker, proc.pid, 180):
        print(f"  [{label}] sim never became ready; killing.")
        _kill(proc.pid)
        proc.wait(timeout=10)
        raise RuntimeError(
            f"{label}: sim did not reach '{ready_marker}' within 180s. "
            f"Check {log_path}"
        )
    print(f"  [{label}] sim ready; sleeping {settle_s:.1f}s settle, "
          f"then capturing for {duration_s:.1f}s")
    time.sleep(settle_s)

    # Run subscriber in this process (single-threaded, no extra subprocess
    # to manage). Saves NPZ on completion.
    out = {"npz_path": str(capture_npz)}
    _subscriber_loop(
        label=label,
        duration_s=duration_s,
        debug_port=debug_port,
        debug_topic="x2_debug",
        pose_port=pose_port,
        pose_topic="robot_pose",
        out=out,
    )

    print(f"  [{label}] capture done; tearing down sim.")
    _kill(proc.pid)
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print(f"  [{label}] WARN: sim did not exit cleanly after SIGKILL.")

    # Tear-down stragglers (docker x2sim, planner pid file, port locks).
    if shutil.which("docker") is not None:
        subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=x2sim"],
            capture_output=True, text=True,
        )
        subprocess.run(
            "docker ps -q --filter name=x2sim | xargs -r docker stop -t 5 >/dev/null 2>&1",
            shell=True, check=False,
        )
    pid_file = Path("/tmp/x2_heuristic_planner.pid")
    if pid_file.exists():
        pid_file.unlink(missing_ok=True)

    # Wait for sockets to free (5556 + 5570 + 5557).
    time.sleep(2.0)

    return _load_capture(capture_npz, label)


def _load_capture(npz_path: Path, label: str) -> CapturedRun:
    z = np.load(npz_path, allow_pickle=False)
    return CapturedRun(
        label=label,
        debug_recv_t=list(z["debug_recv_t"]),
        debug_body_q_target=[r for r in z["debug_body_q_target"]],
        debug_body_q_measured=[r for r in z["debug_body_q_measured"]],
        pose_recv_t=list(z["pose_recv_t"]),
        pelvis_xyz=[r for r in z["pelvis_xyz"]],
        pelvis_quat_wxyz=[r for r in z["pelvis_quat_wxyz"]],
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _summary_row(run: CapturedRun) -> dict[str, float]:
    pose = np.asarray(run.pelvis_xyz)
    target = np.asarray(run.debug_body_q_target)
    measured = np.asarray(run.debug_body_q_measured)
    summary: dict[str, float] = {
        "pose_n_frames": float(pose.shape[0]),
        "debug_n_frames": float(target.shape[0]),
    }
    if pose.size:
        xy = pose[:, :2]
        z = pose[:, 2]
        rel = xy - xy[0:1]
        summary["pelvis_dx_max_m"] = float(np.max(rel[:, 0]) - np.min(rel[:, 0]))
        summary["pelvis_dy_max_m"] = float(np.max(rel[:, 1]) - np.min(rel[:, 1]))
        # Total path length (sum of per-tick xy step magnitudes).
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        summary["pelvis_path_len_m"] = float(np.sum(steps))
        summary["pelvis_max_dist_from_start_m"] = float(
            np.max(np.linalg.norm(rel, axis=1))
        )
        summary["pelvis_z_min_m"] = float(np.min(z))
        summary["pelvis_z_max_m"] = float(np.max(z))
        summary["pelvis_z_range_m"] = float(np.max(z) - np.min(z))
    if measured.size:
        for name, idx in DRIVE_JOINTS.items():
            summary[f"meas_range_{name}_deg"] = float(
                np.degrees(np.max(measured[:, idx]) - np.min(measured[:, idx]))
            )
    if target.size and measured.size:
        n = min(target.shape[0], measured.shape[0])
        for name, idx in DRIVE_JOINTS.items():
            err = target[:n, idx] - measured[:n, idx]
            summary[f"track_err_mean_{name}_deg"] = float(
                np.degrees(np.mean(np.abs(err)))
            )
    return summary


def _print_compare(runs: list[CapturedRun]) -> None:
    summaries = [(r.label, _summary_row(r)) for r in runs]
    if not summaries:
        return
    keys = sorted({k for _, s in summaries for k in s.keys()})
    label_w = max(len(k) for k in keys) + 2

    print("\n" + "=" * 78)
    print("  PELVIS / BODY MOTION SUMMARY")
    print("=" * 78)
    header = f"  {'metric':<{label_w}}" + "".join(
        f"{lab:>16}" for lab, _ in summaries
    )
    print(header)
    print("  " + "-" * (label_w + 16 * len(summaries)))
    for k in keys:
        if k.startswith("track_err_") or k.startswith("meas_range_"):
            continue  # printed in their own section
        row = f"  {k:<{label_w}}"
        for _, s in summaries:
            v = s.get(k, float("nan"))
            row += f"{v:>16.3f}" if not np.isnan(v) else f"{'-':>16}"
        print(row)

    print("\n" + "=" * 78)
    print("  PER-JOINT MEASURED RANGE (max - min, deg)  --  did the joint actually move?")
    print("=" * 78)
    print(header)
    print("  " + "-" * (label_w + 16 * len(summaries)))
    for name in DRIVE_JOINTS.keys():
        k = f"meas_range_{name}_deg"
        row = f"  {k:<{label_w}}"
        for _, s in summaries:
            v = s.get(k, float("nan"))
            row += f"{v:>16.3f}" if not np.isnan(v) else f"{'-':>16}"
        print(row)

    print("\n" + "=" * 78)
    print("  PER-JOINT TRACKING ERROR  mean(|target - measured|), deg")
    print("=" * 78)
    print(header)
    print("  " + "-" * (label_w + 16 * len(summaries)))
    for name in DRIVE_JOINTS.keys():
        k = f"track_err_mean_{name}_deg"
        row = f"  {k:<{label_w}}"
        for _, s in summaries:
            v = s.get(k, float("nan"))
            row += f"{v:>16.3f}" if not np.isnan(v) else f"{'-':>16}"
        print(row)


# ---------------------------------------------------------------------------
# Spawn-cmd builders
# ---------------------------------------------------------------------------

def _build_motion_cmd(motion_pkl: Path, model: Path, max_duration: int,
                      sim_viewer: bool) -> list[str]:
    args = [
        str(_DEPLOY_SH), "sim", "--no-confirm",
        "--motion", str(motion_pkl),
        "--model", str(model),
        "--sim-profile", "parity",
        "--autostart-after", "0",
        "--max-duration", str(max_duration),
    ]
    if sim_viewer:
        args.append("--sim-viewer")
    return args


def _build_planner_cmd(demo_yaml: Path, duration: int, sim_viewer: bool) -> list[str]:
    args = [
        "bash", str(_SMOKE_SH),
        "--demo", str(demo_yaml),
        "--duration", str(duration),
        "--with-deploy",
    ]
    if not sim_viewer:
        args.append("--no-sim-viewer")
    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--motion-pkl", type=Path, required=True,
        help="Baked planner-emit PKL for PATH A (deploy_x2.sh sim --motion <pkl>). "
             "Generate with bake_planner_demo_to_pkl.py."
    )
    p.add_argument(
        "--planner-demo", type=Path, required=True,
        help="Scripted-demo YAML for PATH B (run_planner_smoke.sh --demo <yaml>)."
    )
    p.add_argument("--model", type=Path, default=_DEFAULT_MODEL)
    p.add_argument(
        "--duration", type=int, default=14,
        help="Capture duration per path (seconds). Should cover the demo's "
             "interesting region. Default 14s suits side_steps_only_smoke."
    )
    p.add_argument(
        "--out-dir", type=Path, default=_DEFAULT_OUT_DIR,
        help="Where to save NPZ captures + sim logs."
    )
    p.add_argument(
        "--no-sim-viewer", action="store_true",
        help="Disable the MuJoCo viewer (faster, lower flicker, headless OK)."
    )
    p.add_argument(
        "--only", choices=["motion", "planner"], default=None,
        help="Run only one path (skip the other). Useful for re-running."
    )
    args = p.parse_args()

    if not args.motion_pkl.is_file():
        print(f"ERROR: --motion-pkl not found: {args.motion_pkl}", file=sys.stderr)
        return 1
    if not args.planner_demo.is_file():
        print(f"ERROR: --planner-demo not found: {args.planner_demo}", file=sys.stderr)
        return 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    sim_viewer = not args.no_sim_viewer

    runs: list[CapturedRun] = []

    if args.only != "planner":
        runs.append(_run_path(
            label="A_motion_pkl",
            spawn_cmd=_build_motion_cmd(
                args.motion_pkl, args.model, args.duration + 5, sim_viewer
            ),
            log_path=args.out_dir / f"{ts}_A_motion.log",
            capture_npz=args.out_dir / f"{ts}_A_motion.npz",
            duration_s=float(args.duration),
        ))

    if args.only != "motion":
        runs.append(_run_path(
            label="B_planner_zmq",
            spawn_cmd=_build_planner_cmd(args.planner_demo, args.duration, sim_viewer),
            log_path=args.out_dir / f"{ts}_B_planner.log",
            capture_npz=args.out_dir / f"{ts}_B_planner.npz",
            duration_s=float(args.duration),
        ))

    _print_compare(runs)
    print(f"\n[saved] captures + logs under: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
