"""End-to-end velocity-tracking test for the full kinematic planner.

Drives ``NeuralPlannerCore`` for a multi-second horizon under a constant
commanded velocity, then measures how far the robot ACTUALLY moved in
its body frame -- i.e. the same metric a teleop operator sees.

This is the production-flow companion to
``motionbricks/scripts/test_root_isolated.py``, which only inspects the
root model's one-shot output. ``test_e2e_velocity_tracking.py`` runs the
**full stack** (VQVAE + Pose + Root + qpos integration), invoking
``replan_with_velocity`` repeatedly as ``NeuralPlannerCore`` requests
replans, and accumulates the integrated MuJoCo qpos for the entire
horizon. That captures any compounding drift the single-replan root
test would miss.

Sweeps three axes (default grids; override via CLI flags):

  * **Forward** ``vel_z`` ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6} m/s --
    primary axis of interest, with everything else zeroed.
  * **Lateral** ``vel_x`` ∈ {-0.3, -0.15, 0, +0.15, +0.3} m/s -- checks
    the channel didn't get re-swapped and that side-stepping works.
  * **Yaw** ``yaw_rate`` ∈ {-0.4, -0.2, 0, +0.2, +0.4} rad/s -- checks
    spinning in place.

For each intent we emit:

  *Physical (body frame, in m / deg):*
    - achieved_forward_m, achieved_lateral_m, achieved_dyaw_deg
    - hip_z_mean_m, hip_z_std_m (fall alarm)
    - lateral_leakage = |lat| / max(|fwd|, eps) (sideways drift fraction)

  *Dimensionless:*
    - tracking_forward = achieved_fwd / (vel_z_cmd * horizon_s)
    - tracking_lateral = achieved_lat / (vel_x_cmd * horizon_s)
    - tracking_yaw     = achieved_dyaw_rad / (yaw_rate_cmd * horizon_s)
    - slope (linear fit per axis over the sweep)

The dimensionless ratios are the "apples-to-apples" X2-vs-G1 numbers:
both skeletons should track ~1.0 if their stacks are healthy, regardless
of leg length.

Channel convention (see ``NeuralPlannerCore.replan_with_velocity`` and
``probe_root_constraint_modes.py``):

  * ``vel_x`` = motion-rep X = MuJoCo body-Y = **LATERAL**
  * ``vel_z`` = motion-rep Z = MuJoCo body-X = **FORWARD**

Usage::

    PYTHONPATH="${PWD}/motionbricks:${PWD}" python motionbricks/scripts/test_e2e_velocity_tracking.py \\
        --ckpt-set x2 \\
        --fixture walking \\
        --sweep forward \\
        --horizon-s 3.0 \\
        --save-npz out/per_model_report/e2e_x2_forward.npz \\
        --report-json out/per_model_report/e2e_x2_forward.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "motionbricks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "motionbricks"))


# ---------------------------------------------------------------------------
# Fixtures (shared with test_root_isolated.py).
# ---------------------------------------------------------------------------


FIXTURES = {
    "x2": {
        "walking": {
            "kind": "x2_pkl",
            "pkl": "gear_sonic/data/motions/x2_ultra_locowalk.pkl",
            "clip_key": "Loop_Forward_Walk_001__A018",
        },
        "stationary": {
            "kind": "x2_pkl",
            "pkl": "gear_sonic/data/motions/x2_ultra_locowalk.pkl",
            "clip_key": "Idle_Right_001__A019",
        },
    },
    "g1": {
        "walking": {
            "kind": "g1_clip",
            "g1_clip_path": "motionbricks/out/G1-clip.ckpt",
            "clip_idx": 11,
        },
        "stationary": {
            "kind": "g1_clip",
            "g1_clip_path": "motionbricks/out/G1-clip.ckpt",
            "clip_idx": 0,
        },
    },
}


# Default sweep grids (m/s for translational, rad/s for yaw).
DEFAULT_FORWARD_GRID = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
DEFAULT_LATERAL_GRID = (-0.30, -0.15, 0.0, 0.15, 0.30)
DEFAULT_YAW_GRID = (-0.40, -0.20, 0.0, 0.20, 0.40)


# ---------------------------------------------------------------------------
# Quaternion + yaw helpers.
# ---------------------------------------------------------------------------


def _quat_xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        axis=-1,
    )


def _wxyz_to_yaw_rad(quat_wxyz: np.ndarray) -> np.ndarray:
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _project_world_to_body(
    dx_w: float, dy_w: float, yaw0_rad: float
) -> tuple[float, float]:
    """Rotate world-frame XY delta into the body frame at yaw0.

    Returns (forward_m, lateral_m). ``forward`` is body +X (the direction
    the robot was facing at t=0 in MuJoCo world coords).
    """
    c, s = np.cos(-yaw0_rad), np.sin(-yaw0_rad)
    return float(c * dx_w - s * dy_w), float(s * dx_w + c * dy_w)


# ---------------------------------------------------------------------------
# Fixture loaders -> seed qpos.
# ---------------------------------------------------------------------------


def _load_x2_fixture_qpos(pkl_path: Path, clip_key: str) -> tuple[np.ndarray, float]:
    payload = joblib.load(pkl_path)[clip_key]
    trans = np.asarray(payload["root_trans_offset"], dtype=np.float32)
    rot_xyzw = np.asarray(payload["root_rot"], dtype=np.float32)
    dof = np.asarray(payload["dof"], dtype=np.float32)
    rot_wxyz = _quat_xyzw_to_wxyz(rot_xyzw)
    qpos = np.concatenate([trans, rot_wxyz, dof], axis=-1).astype(np.float32)
    fps = float(payload.get("fps", 30))
    return qpos, fps


def _load_g1_fixture_qpos(
    g1_clip_path: Path, clip_idx: int
) -> tuple[np.ndarray, float]:
    sd = torch.load(g1_clip_path, map_location="cpu", weights_only=False)
    qpos_all = sd["mujoco_qpos"].numpy()  # [N, 150, 36]
    nfpc = sd["num_frames_per_clip"].numpy()
    if clip_idx < 0 or clip_idx >= qpos_all.shape[0]:
        raise IndexError(
            f"G1 clip_idx={clip_idx} out of range [0, {qpos_all.shape[0]})"
        )
    n = int(nfpc[clip_idx])
    return qpos_all[clip_idx, :n].astype(np.float32), 30.0


# ---------------------------------------------------------------------------
# Planner loader.
# ---------------------------------------------------------------------------


def _load_planner(ckpt_set: str, device: str, clip_ckpt: "Path | None" = None):
    """Load the planner. ``clip_ckpt`` overrides the pose-template clip library.

    For x2, ``None`` -> ``"auto"`` (out/X2-clip.ckpt if present). For g1,
    ``None`` -> no clip library (velocity-only). Pass an explicit path (e.g.
    out/X2-clip-g1style.ckpt or out/G1-clip.ckpt) to exercise the
    pose-template path.
    """
    if ckpt_set == "x2":
        from motionbricks.motion_backbone.inference.load_x2_planner import (
            X2PlannerPaths,
            load_x2_planner,
        )
        paths = X2PlannerPaths.default()
        paths.validate()
        return load_x2_planner(
            paths,
            device=device,
            clip_library_ckpt=(clip_ckpt if clip_ckpt is not None else "auto"),
        )
    if ckpt_set == "g1":
        from motionbricks.motion_backbone.inference.load_g1_planner import (
            G1PlannerPaths,
            load_g1_planner,
        )
        paths = G1PlannerPaths.default()
        paths.validate()
        # load_g1_planner forwards **kwargs to NeuralPlannerCore. G1's loader
        # has no "auto" clip resolution, so pass an explicit path or nothing.
        kw = {} if clip_ckpt is None else {"clip_library_ckpt": clip_ckpt}
        return load_g1_planner(paths, device=device, **kw)
    raise ValueError(f"Unknown ckpt-set: {ckpt_set!r}")


# ---------------------------------------------------------------------------
# Intent / trial dataclasses.
# ---------------------------------------------------------------------------


@dataclass
class Intent:
    yaw_rate: float
    vel_x: float  # lateral (motion-rep X = MuJoCo body Y)
    vel_z: float  # forward (motion-rep Z = MuJoCo body X)
    hip_h: float

    def as_tensor(self, device: str) -> torch.Tensor:
        return torch.tensor(
            [self.yaw_rate, self.vel_x, self.vel_z, self.hip_h],
            device=device,
            dtype=torch.float32,
        )

    def as_list(self) -> list[float]:
        return [self.yaw_rate, self.vel_x, self.vel_z, self.hip_h]


# ---------------------------------------------------------------------------
# Single-trial driver.
# ---------------------------------------------------------------------------


def _run_one_trial(
    planner,
    seed_qpos_np: np.ndarray,
    seed_offset: int,
    intent: Intent,
    horizon_frames: int,
    device: str,
    mode_idx: "int | None" = None,
) -> np.ndarray:
    """Drive the full planner for ``horizon_frames`` under a constant intent.

    When ``mode_idx`` is None (default) the velocity-only path
    (``replan_with_velocity``) is used. When set, the pose-template path
    (``replan_with_pose_template``) is used with that clip-library mode
    index -- the command velocity still drives the root target; the clip
    supplies the pose keyframe.

    Returns:
        Integrated qpos trajectory of shape [horizon_frames, qpos_dim].
    """
    n_avail = seed_qpos_np.shape[0] - seed_offset
    seed_window = seed_qpos_np[seed_offset : seed_offset + min(n_avail, 64)]
    seed_t = torch.from_numpy(seed_window).to(device)
    planner.reset(seed_t)

    intent_t = intent.as_tensor(device)

    def _replan() -> None:
        if mode_idx is None:
            planner.replan_with_velocity(intent_t)
        else:
            planner.replan_with_pose_template(intent_t, mode_idx=mode_idx)

    _replan()

    qpos_dim = int(planner.frames["mujoco_qpos"].shape[-1])
    traj = np.zeros((horizon_frames, qpos_dim), dtype=np.float32)
    for i in range(horizon_frames):
        if planner.should_replan():
            _replan()
        traj[i] = planner.get_next_frame().detach().cpu().numpy()
    return traj


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------


def _metrics_from_trial(
    traj: np.ndarray, intent: Intent, fps: float
) -> dict[str, Any]:
    """Body-frame physical + dimensionless metrics for one trial."""
    n = traj.shape[0]
    horizon_s = n / float(fps) if n > 0 else 0.0
    trans = traj[:, :3]
    quat_wxyz = traj[:, 3:7]
    yaw = np.unwrap(_wxyz_to_yaw_rad(quat_wxyz))
    yaw0 = float(yaw[0])
    dx_w = float(trans[-1, 0] - trans[0, 0])
    dy_w = float(trans[-1, 1] - trans[0, 1])
    fwd_m, lat_m = _project_world_to_body(dx_w, dy_w, yaw0)
    dyaw_rad = float(yaw[-1] - yaw[0])
    dyaw_deg = float(np.degrees(dyaw_rad))
    hip_z = trans[:, 2]
    hip_z_mean = float(hip_z.mean())
    hip_z_std = float(hip_z.std())

    cmd_fwd_m = intent.vel_z * horizon_s
    cmd_lat_m = intent.vel_x * horizon_s
    cmd_dyaw_rad = intent.yaw_rate * horizon_s

    def _ratio(achieved: float, commanded: float) -> float | None:
        return achieved / commanded if abs(commanded) > 1e-6 else None

    leakage = abs(lat_m) / max(abs(fwd_m), 1e-3) if abs(intent.vel_z) > 1e-6 else None

    return {
        "horizon_s": horizon_s,
        "horizon_frames": int(n),
        "achieved_forward_m": fwd_m,
        "achieved_lateral_m": lat_m,
        "achieved_dyaw_deg": dyaw_deg,
        "hip_z_mean_m": hip_z_mean,
        "hip_z_std_m": hip_z_std,
        "lateral_leakage": leakage,
        "tracking_forward": _ratio(fwd_m, cmd_fwd_m),
        "tracking_lateral": _ratio(lat_m, cmd_lat_m),
        "tracking_yaw": _ratio(dyaw_rad, cmd_dyaw_rad),
    }


def _fit_slope(
    commands: list[float], achieved: list[float], horizon_s: float | None = None
) -> float | None:
    """Dimensionless slope of achieved-vs-commanded across the sweep.

    ``np.polyfit`` returns ``d(achieved_distance) / d(commanded_velocity)``
    which has units of seconds (meters per m/s). Dividing by the
    integration horizon gives the dimensionless ``tracking_ratio`` slope,
    which is comparable across skeletons / horizons. Slope ~1.0 = perfect
    tracking; slope < 1 = under-tracking.

    If ``horizon_s`` is None the raw polyfit slope is returned (units:
    seconds), which only makes sense within a single run.
    """
    if len(commands) < 2:
        return None
    c = np.asarray(commands, dtype=np.float64)
    a = np.asarray(achieved, dtype=np.float64)
    if np.allclose(c, 0.0):
        return None
    # Regular polyfit so the intercept absorbs any static bias.
    raw_slope, _intercept = np.polyfit(c, a, deg=1)
    raw_slope = float(raw_slope)
    if horizon_s is None or horizon_s <= 0.0:
        return raw_slope
    return raw_slope / float(horizon_s)


# ---------------------------------------------------------------------------
# Sweep grid builders.
# ---------------------------------------------------------------------------


def _build_grid(
    sweep_axes: list[str],
    forward_grid: tuple[float, ...],
    lateral_grid: tuple[float, ...],
    yaw_grid: tuple[float, ...],
    hip_h: float,
) -> list[tuple[str, Intent]]:
    """Build (axis_label, Intent) pairs for the requested sweeps."""
    out: list[tuple[str, Intent]] = []
    if "forward" in sweep_axes:
        for v in forward_grid:
            out.append(("forward", Intent(0.0, 0.0, float(v), hip_h)))
    if "lateral" in sweep_axes:
        for v in lateral_grid:
            out.append(("lateral", Intent(0.0, float(v), 0.0, hip_h)))
    if "yaw" in sweep_axes:
        for v in yaw_grid:
            out.append(("yaw", Intent(float(v), 0.0, 0.0, hip_h)))
    if not out:
        raise ValueError(
            f"No trials produced. sweep={sweep_axes}, grids "
            f"forward={forward_grid}, lateral={lateral_grid}, yaw={yaw_grid}"
        )
    return out


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_axes(s: str) -> list[str]:
    valid = {"forward", "lateral", "yaw"}
    if s == "all":
        return ["forward", "lateral", "yaw"]
    axes = [a.strip() for a in s.split(",") if a.strip()]
    bad = [a for a in axes if a not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f"Unknown sweep axis: {bad!r}. valid={valid}")
    return axes


def _parse_floats(s: str | None) -> tuple[float, ...] | None:
    if s is None:
        return None
    return tuple(float(x) for x in s.split(",") if x.strip())


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ckpt-set",
        choices=("x2", "g1"),
        default="x2",
    )
    p.add_argument(
        "--fixture",
        choices=("walking", "stationary"),
        default="walking",
    )
    p.add_argument(
        "--sweep",
        type=_parse_axes,
        default=["forward"],
        help="Comma-separated axes from {forward, lateral, yaw} (or 'all'). "
             "Default: forward only.",
    )
    p.add_argument(
        "--horizon-s",
        type=float,
        default=3.0,
        help="Seconds of integrated trajectory per trial (default 3.0).",
    )
    p.add_argument(
        "--forward-grid",
        type=_parse_floats,
        default=None,
        help="Comma-separated forward (vel_z) speeds in m/s. "
             f"Default: {DEFAULT_FORWARD_GRID}",
    )
    p.add_argument(
        "--lateral-grid",
        type=_parse_floats,
        default=None,
        help="Comma-separated lateral (vel_x) speeds in m/s. "
             f"Default: {DEFAULT_LATERAL_GRID}",
    )
    p.add_argument(
        "--yaw-grid",
        type=_parse_floats,
        default=None,
        help="Comma-separated yaw rates in rad/s. "
             f"Default: {DEFAULT_YAW_GRID}",
    )
    p.add_argument(
        "--seed-frame",
        type=int,
        default=0,
        help="Frame index of fixture to use as the seed window start.",
    )
    p.add_argument(
        "--hip-h",
        type=float,
        default=None,
        help="Hip-height intent (m). Default: mean hip-Z over seed window.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--x2-seed-clip-key",
        type=str,
        default=None,
        help="Override the x2_pkl fixture's seed clip key (e.g. "
        "neutral_idle_loop_001__A076 to match G1's idle seed for a fair "
        "same-start comparison). Ignored for g1_clip fixtures.",
    )
    p.add_argument(
        "--seed-clip-idx",
        type=int,
        default=None,
        help="Override the g1_clip fixture's seed clip index (G1-clip.ckpt "
        "order: 0=idle 1=slow_walk 2=walk 11=walk_gun ...). Use 2 for a "
        "neutral-arm walk seed instead of the fixture default. Ignored for "
        "x2_pkl fixtures.",
    )
    p.add_argument(
        "--planner-mode-idx",
        type=int,
        default=None,
        help="If set, use POSE-TEMPLATE inference with this clip-library mode "
        "index (0=idle, 1=slow_walk, 2=walk, 3=run_proxy) instead of the "
        "velocity-only path. The command velocity still drives the root; "
        "the clip supplies the target pose keyframe.",
    )
    p.add_argument(
        "--clip-ckpt",
        type=Path,
        default=None,
        help="Override the pose-template clip-library ckpt. x2 default: auto "
        "(out/X2-clip.ckpt). Pass out/X2-clip-g1style.ckpt for the G1-style "
        "A/B, or out/G1-clip.ckpt for g1 pose-template runs.",
    )
    p.add_argument("--save-npz", type=Path, default=None)
    p.add_argument("--report-json", type=Path, default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Pretty-printing.
# ---------------------------------------------------------------------------


def _fmt_opt(v: float | None, fmt: str = "{:.2f}") -> str:
    return fmt.format(v) if v is not None else "  --"


def _print_axis_table(axis: str, rows: list[dict], horizon_s: float) -> None:
    if not rows:
        return
    print(f"\n  axis: {axis}")
    if axis == "forward":
        cmd_label = "vel_z_cmd"
        cmd_key = lambda r: r["intent"]["vel_z"]
        achieved_label = "fwd_m"
        achieved_key = lambda m: m["achieved_forward_m"]
        slope_y_key = lambda m: m["achieved_forward_m"]
        track_key = "tracking_forward"
    elif axis == "lateral":
        cmd_label = "vel_x_cmd"
        cmd_key = lambda r: r["intent"]["vel_x"]
        achieved_label = "lat_m"
        achieved_key = lambda m: m["achieved_lateral_m"]
        slope_y_key = lambda m: m["achieved_lateral_m"]
        track_key = "tracking_lateral"
    else:  # yaw -- table prints deg for readability but slope must use rad
        cmd_label = "yaw_cmd_rad_s"
        cmd_key = lambda r: r["intent"]["yaw_rate"]
        achieved_label = "dyaw_deg"
        achieved_key = lambda m: m["achieved_dyaw_deg"]
        slope_y_key = lambda m: float(np.radians(m["achieved_dyaw_deg"]))
        track_key = "tracking_yaw"

    print(
        f"  {cmd_label:>10} | {achieved_label:>9} {'lat_m':>7} {'fwd_m':>7} "
        f"{'dyaw':>7} | {'track':>6} {'leak':>6} {'hipZ_std_cm':>12}"
    )
    print(f"  {'-'*10} | {'-'*9} {'-'*7} {'-'*7} {'-'*7} | "
          f"{'-'*6} {'-'*6} {'-'*12}")
    cmds: list[float] = []
    achieved_for_slope: list[float] = []
    for r in rows:
        m = r["metrics"]
        cmd = cmd_key(r)
        cmds.append(cmd)
        achieved_for_slope.append(slope_y_key(m))
        print(
            f"  {cmd:>10.3f} | {achieved_key(m):>9.3f} "
            f"{m['achieved_lateral_m']:>7.3f} {m['achieved_forward_m']:>7.3f} "
            f"{m['achieved_dyaw_deg']:>7.1f} | "
            f"{_fmt_opt(m[track_key]):>6} "
            f"{_fmt_opt(m['lateral_leakage']):>6} "
            f"{m['hip_z_std_m'] * 100:>12.2f}"
        )
    slope = _fit_slope(cmds, achieved_for_slope, horizon_s=horizon_s)
    print(
        f"  -> dimensionless slope(track_{axis}) = {_fmt_opt(slope, fmt='{:.3f}')}  "
        f"(ideal ~1.0)"
    )


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    fixture_spec = FIXTURES[args.ckpt_set][args.fixture]
    if fixture_spec["kind"] == "x2_pkl":
        pkl = REPO_ROOT / fixture_spec["pkl"]
        clip_key = args.x2_seed_clip_key or fixture_spec["clip_key"]
        print(f"[fixture] {args.ckpt_set} {args.fixture}: {pkl.name}::{clip_key}")
        seed_qpos_np, fps = _load_x2_fixture_qpos(pkl, clip_key)
    else:
        g1_path = REPO_ROOT / fixture_spec["g1_clip_path"]
        idx = (
            args.seed_clip_idx
            if args.seed_clip_idx is not None
            else fixture_spec["clip_idx"]
        )
        print(f"[fixture] {args.ckpt_set} {args.fixture}: {g1_path.name}[{idx}]")
        seed_qpos_np, fps = _load_g1_fixture_qpos(g1_path, idx)
    print(
        f"[fixture] T={seed_qpos_np.shape[0]} frames @ {fps} fps "
        f"(dur={seed_qpos_np.shape[0] / fps:.2f} s, qpos dim={seed_qpos_np.shape[-1]})"
    )

    hip_h = (
        args.hip_h
        if args.hip_h is not None
        else float(seed_qpos_np[args.seed_frame : args.seed_frame + 4, 2].mean())
    )

    forward_grid = args.forward_grid or DEFAULT_FORWARD_GRID
    lateral_grid = args.lateral_grid or DEFAULT_LATERAL_GRID
    yaw_grid = args.yaw_grid or DEFAULT_YAW_GRID
    trials = _build_grid(args.sweep, forward_grid, lateral_grid, yaw_grid, hip_h)
    horizon_frames = max(2, int(round(args.horizon_s * fps)))
    print(
        f"[plan] sweep={args.sweep}  hip_h={hip_h:.3f}  "
        f"horizon={args.horizon_s:.2f}s ({horizon_frames} frames)  "
        f"n_trials={len(trials)}"
    )

    print(f"[load] ckpt-set={args.ckpt_set} on device={args.device}")
    if args.planner_mode_idx is not None:
        print(
            f"[plan] pose-template path: mode_idx={args.planner_mode_idx}  "
            f"clip_ckpt={args.clip_ckpt if args.clip_ckpt else '(auto/default)'}"
        )
    planner = _load_planner(args.ckpt_set, args.device, clip_ckpt=args.clip_ckpt)

    rows: list[dict] = []
    trajs: list[np.ndarray] = []
    for axis, intent in trials:
        traj = _run_one_trial(
            planner, seed_qpos_np, args.seed_frame, intent, horizon_frames,
            args.device, mode_idx=args.planner_mode_idx,
        )
        metrics = _metrics_from_trial(traj, intent, fps)
        rows.append({"axis": axis, "intent": asdict(intent), "metrics": metrics})
        trajs.append(traj)
        print(
            f"  {axis:>7}: cmd=(yaw={intent.yaw_rate:+.2f}, vx={intent.vel_x:+.2f}, "
            f"vz={intent.vel_z:+.2f}) -> "
            f"fwd={metrics['achieved_forward_m']:+.3f}m  "
            f"lat={metrics['achieved_lateral_m']:+.3f}m  "
            f"dyaw={metrics['achieved_dyaw_deg']:+.1f}deg  "
            f"hipZ_std={metrics['hip_z_std_m'] * 100:.2f}cm"
        )

    # Aggregate slope per requested axis.
    slopes: dict[str, float | None] = {}
    for axis in args.sweep:
        axis_rows = [r for r in rows if r["axis"] == axis]
        if axis == "forward":
            cmds = [r["intent"]["vel_z"] for r in axis_rows]
            ach = [r["metrics"]["achieved_forward_m"] for r in axis_rows]
        elif axis == "lateral":
            cmds = [r["intent"]["vel_x"] for r in axis_rows]
            ach = [r["metrics"]["achieved_lateral_m"] for r in axis_rows]
        else:
            cmds = [r["intent"]["yaw_rate"] for r in axis_rows]
            ach = [np.radians(r["metrics"]["achieved_dyaw_deg"]) for r in axis_rows]
        slopes[axis] = _fit_slope(cmds, ach, horizon_s=args.horizon_s)

    # Pretty stdout: per-axis tables + summary.
    print()
    print("=" * 100)
    print(
        f"  E2E velocity tracking  (ckpt={args.ckpt_set}, fixture={args.fixture}, "
        f"horizon={args.horizon_s:.2f}s)"
    )
    print("=" * 100)
    for axis in args.sweep:
        axis_rows = [r for r in rows if r["axis"] == axis]
        _print_axis_table(axis, axis_rows, horizon_s=args.horizon_s)
    print()
    print("=" * 100)
    print("  Summary slopes (dimensionless; ideal ~1.0):")
    for axis, slope in slopes.items():
        print(f"    slope_{axis:<8} = {_fmt_opt(slope, fmt='{:.3f}')}")
    print("=" * 100)
    print(
        "  track  : achieved / commanded over the full horizon; ideal ~1.0\n"
        "  leak   : |lateral| / |forward| -- sideways drift fraction (forward only)\n"
        "  hipZ_std_cm : hip-Z drift across the horizon (fall alarm if huge)"
    )

    # Save report JSON.
    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "ckpt_set": args.ckpt_set,
            "fixture": args.fixture,
            "fixture_spec": fixture_spec,
            "fps": fps,
            "horizon_s": args.horizon_s,
            "horizon_frames": horizon_frames,
            "seed_frame": args.seed_frame,
            "hip_h": hip_h,
            "sweep_axes": args.sweep,
            "forward_grid": list(forward_grid),
            "lateral_grid": list(lateral_grid),
            "yaw_grid": list(yaw_grid),
            "slopes": slopes,
            "trials": rows,
        }
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[report] wrote {args.report_json}")

    # Save NPZ for viewer / offline analysis.
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        intents_np = np.array(
            [
                [
                    rows[i]["intent"]["yaw_rate"],
                    rows[i]["intent"]["vel_x"],
                    rows[i]["intent"]["vel_z"],
                    rows[i]["intent"]["hip_h"],
                ]
                for i in range(len(rows))
            ],
            dtype=np.float32,
        )
        axes_np = np.array([r["axis"] for r in rows])
        traj_stack = np.stack(trajs, axis=0)
        np.savez(
            args.save_npz,
            ckpt_set=args.ckpt_set,
            fixture=args.fixture,
            fps=fps,
            horizon_s=args.horizon_s,
            seed_window=seed_qpos_np[args.seed_frame : args.seed_frame + 8],
            intents=intents_np,
            axes=axes_np,
            qpos_traj=traj_stack,
            planner_mode_idx=(-1 if args.planner_mode_idx is None else args.planner_mode_idx),
            clip_ckpt=("" if args.clip_ckpt is None else str(args.clip_ckpt)),
        )
        print(f"[npz]    wrote {args.save_npz}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
