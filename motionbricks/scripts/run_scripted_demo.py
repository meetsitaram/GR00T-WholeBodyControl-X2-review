"""Drive the kinematic planner through a scripted velocity-intent sequence.

Strings together a sequence of constant-intent segments (walk fwd, stop,
turn left, walk, stop, turn back, walk, stop, side-step), running the
full planner stack continuously across the boundaries. Saves the
integrated qpos trajectory plus the schedule of intent transitions in
the same NPZ layout as ``test_e2e_velocity_tracking.py`` so
``view_e2e_x2_vs_g1.py`` can render it side-by-side against the same
demo driven by a different checkpoint set.

NB on the yaw discrepancy across stacks: X2 over-rotates yaw at ~2.35×
the commanded rate and G1 at ~1.27× (see
``out/per_model_report/e2e_*.json``). A fixed-duration "turn 90°" step
will therefore look different on each robot -- that's the point of the
side-by-side, not a bug in this script.

Usage::

    PYTHONPATH="${PWD}/motionbricks:${PWD}" python motionbricks/scripts/run_scripted_demo.py \\
        --ckpt-set x2 --save-npz out/per_model_report/demo_x2.npz

    # Then for the side-by-side, repeat with --ckpt-set g1 and view both:
    python motionbricks/scripts/view_e2e_x2_vs_g1.py \\
        --x2-npz out/per_model_report/demo_x2.npz \\
        --g1-npz out/per_model_report/demo_g1.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "motionbricks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "motionbricks"))


# ---------------------------------------------------------------------------
# Fixtures and planner loader (mirrors test_e2e_velocity_tracking.py).
# ---------------------------------------------------------------------------


FIXTURES = {
    "x2": {
        "walking": {
            "kind": "x2_pkl",
            "pkl": "gear_sonic/data/motions/x2_ultra_locowalk.pkl",
            "clip_key": "Loop_Forward_Walk_001__A018",
        },
    },
    "g1": {
        "walking": {
            "kind": "g1_clip",
            "g1_clip_path": "motionbricks/out/G1-clip.ckpt",
            "clip_idx": 11,
        },
    },
}


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.stack(
        [q[..., 3], q[..., 0], q[..., 1], q[..., 2]], axis=-1
    )


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
    qpos_all = sd["mujoco_qpos"].numpy()
    nfpc = sd["num_frames_per_clip"].numpy()
    n = int(nfpc[clip_idx])
    return qpos_all[clip_idx, :n].astype(np.float32), 30.0


def _load_planner(ckpt_set: str, device: str):
    if ckpt_set == "x2":
        from motionbricks.motion_backbone.inference.load_x2_planner import (
            X2PlannerPaths,
            load_x2_planner,
        )
        paths = X2PlannerPaths.default()
        paths.validate()
        return load_x2_planner(paths, device=device)
    if ckpt_set == "g1":
        from motionbricks.motion_backbone.inference.load_g1_planner import (
            G1PlannerPaths,
            load_g1_planner,
        )
        paths = G1PlannerPaths.default()
        paths.validate()
        return load_g1_planner(paths, device=device)
    raise ValueError(f"Unknown ckpt-set: {ckpt_set!r}")


# ---------------------------------------------------------------------------
# Demo schedule.
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One scheduled segment of constant or linearly-ramped intent.

    Set the ``*_end`` fields to a value different from the base channel
    to linearly interpolate that channel from base -> end across the
    segment's duration. Channels left as ``None`` stay constant at the
    base value.
    """

    label: str
    duration_s: float
    yaw_rate: float
    vel_x: float  # lateral (motion-rep X = MuJoCo body Y)
    vel_z: float  # forward (motion-rep Z = MuJoCo body X)
    yaw_rate_end: float | None = None
    vel_x_end: float | None = None
    vel_z_end: float | None = None


def _default_schedule(hip_h: float) -> list[Step]:
    """Locomotion sequence with in-place + walking turns (~19.5 s).

    Mixes pure-translation, pure-yaw (turn in place), and combined
    (walk-while-turning, i.e. arcs). X2's yaw over-tracks ~2.35x and
    G1's ~1.27x (see ``out/per_model_report/e2e_*.json``) so the in-
    place turns and the arc curvature will visibly diverge between the
    two stacks -- that's the cross-stack signal.
    """
    return [
        Step("slow_walk",          2.0, 0.0,  0.0,  0.20),
        Step("run",                2.5, 0.0,  0.0,  1.00),
        Step("stop_1",             1.0, 0.0,  0.0,  0.0),
        Step("turn_left_inplace",  1.5, +0.4, 0.0,  0.0),
        Step("turn_right_inplace", 1.5, -0.4, 0.0,  0.0),
        Step("stop_2",             0.7, 0.0,  0.0,  0.0),
        Step("arc_left_while_walking",  2.5, +0.3, 0.0,  0.40),
        Step("arc_right_while_walking", 2.5, -0.3, 0.0,  0.40),
        Step("stop_3",             0.7, 0.0,  0.0,  0.0),
        Step("walk_back",          2.0, 0.0,  0.0, -0.30),
        Step("stop_4",             0.7, 0.0,  0.0,  0.0),
        Step("sidestep_left",      1.5, 0.0,  +0.30, 0.0),
        Step("sidestep_right",     1.5, 0.0,  -0.30, 0.0),
        Step("stop_5",             0.5, 0.0,  0.0,  0.0),
    ]


def _intent_at_fraction(step: Step, frac: float) -> tuple[float, float, float]:
    """Return (yaw_rate, vel_x, vel_z) at progress ``frac`` in [0, 1]."""
    def _lerp(a: float, b: float | None) -> float:
        if b is None:
            return a
        return a + (b - a) * frac

    return (
        _lerp(step.yaw_rate, step.yaw_rate_end),
        _lerp(step.vel_x, step.vel_x_end),
        _lerp(step.vel_z, step.vel_z_end),
    )


def _step_to_intent_tensor(step: Step, hip_h: float, device: str) -> torch.Tensor:
    return torch.tensor(
        [step.yaw_rate, step.vel_x, step.vel_z, hip_h],
        device=device,
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Driver: roll the planner across the whole schedule.
# ---------------------------------------------------------------------------


def _run_schedule(
    planner,
    seed_qpos_np: np.ndarray,
    seed_offset: int,
    schedule: list[Step],
    hip_h: float,
    fps: float,
    device: str,
) -> tuple[np.ndarray, list[dict]]:
    """Execute the schedule continuously, return [T_total, D] qpos + segment log."""
    n_avail = seed_qpos_np.shape[0] - seed_offset
    seed_window = seed_qpos_np[seed_offset : seed_offset + min(n_avail, 64)]
    seed_t = torch.from_numpy(seed_window).to(device)
    planner.reset(seed_t)

    # Prime the buffer with the first step's intent.
    initial_intent = _step_to_intent_tensor(schedule[0], hip_h, device)
    planner.replan_with_velocity(initial_intent)

    qpos_dim = int(planner.frames["mujoco_qpos"].shape[-1])
    def _intent_tensor(yaw_rate: float, vel_x: float, vel_z: float) -> torch.Tensor:
        return torch.tensor(
            [yaw_rate, vel_x, vel_z, hip_h], device=device, dtype=torch.float32
        )

    chunks: list[np.ndarray] = []
    segments: list[dict] = []
    frame_idx = 0
    for step in schedule:
        n_frames = max(1, int(round(step.duration_s * fps)))
        is_ramped = (
            step.yaw_rate_end is not None
            or step.vel_x_end is not None
            or step.vel_z_end is not None
        )
        # Force a replan at the segment boundary so the new intent takes
        # effect immediately (otherwise we'd burn through the previously
        # buffered frames first).
        yaw0, vx0, vz0 = _intent_at_fraction(step, 0.0)
        planner.replan_with_velocity(_intent_tensor(yaw0, vx0, vz0))

        chunk = np.zeros((n_frames, qpos_dim), dtype=np.float32)
        for i in range(n_frames):
            # Sub-frame fraction: 0 at start, 1 at last frame.
            frac = i / max(1, n_frames - 1) if n_frames > 1 else 0.0
            if is_ramped:
                yaw_r, vx_r, vz_r = _intent_at_fraction(step, frac)
                if planner.should_replan():
                    planner.replan_with_velocity(_intent_tensor(yaw_r, vx_r, vz_r))
            else:
                if planner.should_replan():
                    planner.replan_with_velocity(
                        _intent_tensor(step.yaw_rate, step.vel_x, step.vel_z)
                    )
            chunk[i] = planner.get_next_frame().detach().cpu().numpy()
        chunks.append(chunk)
        yaw1, vx1, vz1 = _intent_at_fraction(step, 1.0)
        segments.append(
            {
                "label": step.label,
                "start_frame": frame_idx,
                "n_frames": n_frames,
                "duration_s": step.duration_s,
                "yaw_rate": step.yaw_rate,
                "vel_x": step.vel_x,
                "vel_z": step.vel_z,
                "yaw_rate_end": yaw1,
                "vel_x_end": vx1,
                "vel_z_end": vz1,
                "ramped": is_ramped,
                "hip_h": hip_h,
            }
        )
        frame_idx += n_frames
    traj = np.concatenate(chunks, axis=0)
    return traj, segments


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-set", choices=("x2", "g1"), default="x2")
    p.add_argument(
        "--seed-frame", type=int, default=0,
        help="Frame index of the walking fixture to seed the planner from.",
    )
    p.add_argument(
        "--hip-h", type=float, default=None,
        help="Hip-height intent (m). Default: mean hip-Z of seed window.",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument(
        "--save-npz", type=Path, default=None,
        help="Output NPZ for the viewer (qpos_traj shape [1, T, D]).",
    )
    p.add_argument(
        "--save-schedule-json", type=Path, default=None,
        help="Optional JSON dump of the per-segment schedule.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    fixture_spec = FIXTURES[args.ckpt_set]["walking"]
    if fixture_spec["kind"] == "x2_pkl":
        pkl = REPO_ROOT / fixture_spec["pkl"]
        clip_key = fixture_spec["clip_key"]
        print(f"[fixture] {args.ckpt_set} walking: {pkl.name}::{clip_key}")
        seed_qpos_np, fps = _load_x2_fixture_qpos(pkl, clip_key)
    else:
        g1_path = REPO_ROOT / fixture_spec["g1_clip_path"]
        idx = fixture_spec["clip_idx"]
        print(f"[fixture] {args.ckpt_set} walking: {g1_path.name}[{idx}]")
        seed_qpos_np, fps = _load_g1_fixture_qpos(g1_path, idx)

    hip_h = (
        args.hip_h
        if args.hip_h is not None
        else float(seed_qpos_np[args.seed_frame : args.seed_frame + 4, 2].mean())
    )
    schedule = _default_schedule(hip_h)
    total_s = sum(s.duration_s for s in schedule)
    print(
        f"[plan] hip_h={hip_h:.3f}  fps={fps:.0f}  total={total_s:.2f}s "
        f"(~{int(round(total_s * fps))} frames)"
    )
    for s in schedule:
        def _fmt(a: float, b: float | None) -> str:
            if b is None or abs(a - b) < 1e-6:
                return f"{a:+.2f}"
            return f"{a:+.2f}->{b:+.2f}"

        print(
            f"  - {s.label:<16} dur={s.duration_s:>4.1f}s  "
            f"yaw={_fmt(s.yaw_rate, s.yaw_rate_end)}  "
            f"vx(lat)={_fmt(s.vel_x, s.vel_x_end)}  "
            f"vz(fwd)={_fmt(s.vel_z, s.vel_z_end)}"
        )

    print(f"[load] ckpt-set={args.ckpt_set} on device={args.device}")
    planner = _load_planner(args.ckpt_set, args.device)

    traj, segments = _run_schedule(
        planner, seed_qpos_np, args.seed_frame, schedule, hip_h, fps, args.device
    )
    print(
        f"[run]  qpos shape = {traj.shape}  "
        f"(dur ~{traj.shape[0] / fps:.2f}s @ {fps:.0f}fps)"
    )

    # Saving: match view_e2e_x2_vs_g1.py's expected NPZ layout
    # (qpos_traj=[N_trials, T, D] with N_trials=1).
    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        # intents: dump the FIRST step's intent so the viewer's
        # trial-summary header has something meaningful to print.
        first = schedule[0]
        intents = np.array(
            [[first.yaw_rate, first.vel_x, first.vel_z, hip_h]],
            dtype=np.float32,
        )
        np.savez(
            args.save_npz,
            ckpt_set=args.ckpt_set,
            fixture="scripted_demo",
            fps=fps,
            horizon_s=traj.shape[0] / fps,
            seed_window=seed_qpos_np[args.seed_frame : args.seed_frame + 8],
            intents=intents,
            axes=np.array(["demo"]),
            qpos_traj=traj[None, ...],
            segments_json=json.dumps(segments),
        )
        print(f"[npz]  wrote {args.save_npz}")

    if args.save_schedule_json is not None:
        args.save_schedule_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.save_schedule_json, "w") as f:
            json.dump(
                {
                    "ckpt_set": args.ckpt_set,
                    "fps": fps,
                    "hip_h": hip_h,
                    "segments": segments,
                },
                f,
                indent=2,
            )
        print(f"[json] wrote {args.save_schedule_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
