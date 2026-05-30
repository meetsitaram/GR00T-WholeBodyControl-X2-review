"""Side-by-side MuJoCo viewer: X2 vs G1 driven by the same velocity intent.

Loads two E2E-tracking NPZs produced by
``motionbricks/scripts/test_e2e_velocity_tracking.py`` (one per checkpoint
set, sharing the same sweep grid + horizon) and plays them in a single
MuJoCo scene -- X2 on +Y, G1 on -Y -- with synchronized playback. You
can flip between sweep trials with hotkeys to compare how each stack
responds to the same forward / lateral / yaw command.

Controls (focus the viewer window):

    SPACE   pause / resume
    R       reset playback to frame 0 of the current trial
    [ / ]   step one frame back / forward when paused
    , / .   slow down / speed up playback (0.25x .. 4x)
    n / p   next / previous trial in the sweep
    1..9    jump to trial index (1-based)

Usage::

    python motionbricks/scripts/view_e2e_x2_vs_g1.py \\
        --x2-npz out/per_model_report/e2e_x2_all.npz \\
        --g1-npz out/per_model_report/e2e_g1_all.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_DEFAULT_X2_MJCF = (
    REPO_ROOT / "gear_sonic" / "data" / "assets" / "robot_description"
    / "mjcf" / "x2_ultra.xml"
)
_DEFAULT_G1_MJCF = REPO_ROOT / "gear_sonic_deploy" / "g1" / "g1_29dof.xml"


# ---------------------------------------------------------------------------
# NPZ loading + trial alignment.
# ---------------------------------------------------------------------------


def _load_npz(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"NPZ not found: {path}")
    blob = np.load(path, allow_pickle=True)
    segments = None
    if "segments_json" in blob.files:
        import json
        try:
            segments = json.loads(str(blob["segments_json"]))
        except (TypeError, ValueError):
            segments = None
    return {
        "ckpt_set": str(blob["ckpt_set"]),
        "fixture": str(blob["fixture"]),
        "fps": float(blob["fps"]),
        "horizon_s": float(blob["horizon_s"]),
        "intents": np.asarray(blob["intents"], dtype=np.float32),  # [N, 4]
        "axes": np.asarray(blob["axes"]),  # [N] object/str
        "qpos_traj": np.asarray(blob["qpos_traj"], dtype=np.float32),  # [N, T, D]
        "segments": segments,  # optional: list of {label, start_frame, n_frames, ...}
    }


# ---------------------------------------------------------------------------
# Canonicalization helpers.
# ---------------------------------------------------------------------------


def _wxyz_to_yaw_rad(quat_wxyz: np.ndarray) -> np.ndarray:
    w = quat_wxyz[..., 0]
    x = quat_wxyz[..., 1]
    y = quat_wxyz[..., 2]
    z = quat_wxyz[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _rotate_quat_wxyz_about_z(quat: np.ndarray, dyaw: float) -> np.ndarray:
    """Pre-multiply each wxyz quat by a z-axis rotation of ``dyaw`` radians."""
    c, s = np.cos(dyaw / 2.0), np.sin(dyaw / 2.0)
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    out = np.empty_like(quat)
    out[..., 0] = c * w - s * z
    out[..., 1] = c * x - s * y
    out[..., 2] = c * y + s * x
    out[..., 3] = c * z + s * w
    return out


def _canonicalize_trial_qpos(qpos_traj: np.ndarray) -> np.ndarray:
    """Translate and yaw-rotate each [T, D] trial so frame 0 sits at (0,0)
    facing world +X. Z and joint DoFs are untouched.

    Returns a new array; the input is not mutated.
    """
    if qpos_traj.ndim != 3:
        raise ValueError(f"Expected [N, T, D] trajectory, got {qpos_traj.shape}")
    out = qpos_traj.copy()
    for i in range(out.shape[0]):
        traj = out[i]
        x0 = float(traj[0, 0])
        y0 = float(traj[0, 1])
        yaw0 = float(_wxyz_to_yaw_rad(traj[0, 3:7]))
        c, s = np.cos(-yaw0), np.sin(-yaw0)
        dx = traj[:, 0] - x0
        dy = traj[:, 1] - y0
        traj[:, 0] = c * dx - s * dy
        traj[:, 1] = s * dx + c * dy
        traj[:, 3:7] = _rotate_quat_wxyz_about_z(traj[:, 3:7], -yaw0)
    return out


def _intent_key(intent_row: np.ndarray) -> tuple:
    """Hashable key for matching trials across runs."""
    return tuple(round(float(v), 4) for v in intent_row[:3])  # yaw, vx, vz


def _align_trials(x2: dict, g1: dict) -> list[tuple[int, int, str, np.ndarray]]:
    """Return list of (x2_idx, g1_idx, axis_label, intent[4])."""
    g1_index = {
        _intent_key(g1["intents"][i]): i
        for i in range(g1["intents"].shape[0])
    }
    aligned = []
    for i in range(x2["intents"].shape[0]):
        k = _intent_key(x2["intents"][i])
        if k not in g1_index:
            continue
        j = g1_index[k]
        if str(x2["axes"][i]) != str(g1["axes"][j]):
            continue
        aligned.append((i, j, str(x2["axes"][i]), x2["intents"][i]))
    if not aligned:
        raise RuntimeError(
            "No matching trials between X2 and G1 NPZs. Ensure both were run "
            "with the same sweep grid and horizon."
        )
    return aligned


# ---------------------------------------------------------------------------
# Scene assembly.
# ---------------------------------------------------------------------------


def _build_scene(
    x2_mjcf: Path, g1_mjcf: Path, separation_m: float
) -> mujoco.MjModel:
    parent = mujoco.MjSpec()
    parent.modelname = "x2_vs_g1_e2e"
    parent.option.timestep = 1.0 / 30.0
    parent.option.gravity = [0.0, 0.0, 0.0]

    parent.worldbody.add_light(
        pos=[0, 0, 4], dir=[0, 0, -1], castshadow=False
    )
    parent.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[5.0, 5.0, 0.1],
        rgba=[0.85, 0.85, 0.85, 1.0],
    )

    x2_frame = parent.worldbody.add_frame(pos=[0.0, +separation_m / 2.0, 0.0])
    g1_frame = parent.worldbody.add_frame(pos=[0.0, -separation_m / 2.0, 0.0])

    child_x2 = mujoco.MjSpec.from_file(str(x2_mjcf))
    child_g1 = mujoco.MjSpec.from_file(str(g1_mjcf))

    parent.attach(child_x2, prefix="x2_", frame=x2_frame)
    parent.attach(child_g1, prefix="g1_", frame=g1_frame)

    return parent.compile()


def _root_qpos_addr(model: mujoco.MjModel, prefix: str) -> int:
    name = f"{prefix}floating_base_joint"
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if jid < 0:
        raise KeyError(f"Joint {name!r} not found in compiled model.")
    return int(model.jnt_qposadr[jid])


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--x2-npz", type=Path, required=True)
    p.add_argument("--g1-npz", type=Path, required=True)
    p.add_argument("--x2-mjcf", type=Path, default=None)
    p.add_argument("--g1-mjcf", type=Path, default=None)
    p.add_argument(
        "--separation", type=float, default=1.0,
        help="meters between the two robots in Y (default 1.0)",
    )
    p.add_argument(
        "--fps", type=float, default=0.0,
        help="playback fps; 0 = use the NPZ's saved fps",
    )
    p.add_argument(
        "--start-trial", type=int, default=0,
        help="0-based trial index to start on (default 0)",
    )
    p.add_argument(
        "--no-canonicalize", action="store_true",
        help="Replay raw qpos in each clip's world frame (default: canonicalize "
             "so both robots start at (0,0) facing +X for clean side-by-side).",
    )
    args = p.parse_args(argv)

    x2 = _load_npz(args.x2_npz)
    g1 = _load_npz(args.g1_npz)
    if not args.no_canonicalize:
        x2["qpos_traj"] = _canonicalize_trial_qpos(x2["qpos_traj"])
        g1["qpos_traj"] = _canonicalize_trial_qpos(g1["qpos_traj"])
        print("[viewer] canonicalized both trajectories: each starts at "
              "(0,0) facing +X (use --no-canonicalize to disable).")
    aligned = _align_trials(x2, g1)
    fps = args.fps if args.fps > 0 else x2["fps"]
    horizon_frames = int(x2["qpos_traj"].shape[1])

    print(f"[viewer] X2 npz = {args.x2_npz.name}  ({x2['ckpt_set']}/{x2['fixture']})")
    print(f"[viewer] G1 npz = {args.g1_npz.name}  ({g1['ckpt_set']}/{g1['fixture']})")
    print(f"[viewer] aligned trials = {len(aligned)}")
    print(
        f"[viewer] horizon = {horizon_frames} frames @ {fps:.1f} fps "
        f"({horizon_frames / fps:.2f} s)"
    )

    x2_mjcf = args.x2_mjcf or _DEFAULT_X2_MJCF
    g1_mjcf = args.g1_mjcf or _DEFAULT_G1_MJCF
    for path, label in [(x2_mjcf, "X2"), (g1_mjcf, "G1")]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} MJCF not found: {path}")
    print(f"[viewer] X2 MJCF = {x2_mjcf}")
    print(f"[viewer] G1 MJCF = {g1_mjcf}")
    print(
        f"[viewer] separation = {args.separation:.2f} m "
        "(X2 = +Y / left, G1 = -Y / right)"
    )

    model = _build_scene(x2_mjcf, g1_mjcf, args.separation)
    data = mujoco.MjData(model)
    x2_addr = _root_qpos_addr(model, "x2_")
    g1_addr = _root_qpos_addr(model, "g1_")
    x2_qpos_dim = int(x2["qpos_traj"].shape[-1])
    g1_qpos_dim = int(g1["qpos_traj"].shape[-1])
    print(
        f"[viewer] qpos addr: X2={x2_addr} (dim={x2_qpos_dim}), "
        f"G1={g1_addr} (dim={g1_qpos_dim}), model.nq={model.nq}"
    )

    n_trials = len(aligned)
    trial_idx = max(0, min(args.start_trial, n_trials - 1))
    frame_idx = 0
    paused = False
    speed_mult = 1.0
    speeds = [0.25, 0.5, 1.0, 2.0, 4.0]

    # Segment schedule (only present for scripted-demo NPZs).
    segments = x2.get("segments") or g1.get("segments")
    last_segment_idx = -1

    def _print_trial_header() -> None:
        x2_i, g1_i, axis, intent = aligned[trial_idx]
        yaw_rate, vx, vz, hip_h = (float(intent[k]) for k in range(4))
        print(
            f"[trial {trial_idx + 1}/{n_trials}] axis={axis:<7}  "
            f"yaw_rate={yaw_rate:+.2f}  vx(lat)={vx:+.2f}  vz(fwd)={vz:+.2f}  "
            f"hip_h={hip_h:.2f}"
        )

    half_sep = args.separation / 2.0

    def _set_qpos_for_frame(frame: int) -> None:
        x2_i, g1_i, _axis, _intent = aligned[trial_idx]
        f = max(0, min(frame, horizon_frames - 1))
        x2_qpos = x2["qpos_traj"][x2_i, f].copy()
        g1_qpos = g1["qpos_traj"][g1_i, f].copy()
        # MuJoCo free joints encode the body pose in WORLD coordinates, so
        # the MjSpec ``frame`` attach-offset doesn't apply to free-joint
        # qpos. We bake the side-by-side Y offset directly into qpos[1]
        # here: X2 at +sep/2, G1 at -sep/2.
        x2_qpos[1] += half_sep
        g1_qpos[1] -= half_sep
        data.qpos[x2_addr : x2_addr + x2_qpos_dim] = x2_qpos
        data.qpos[g1_addr : g1_addr + g1_qpos_dim] = g1_qpos
        mujoco.mj_forward(model, data)

    def key_cb(keycode: int) -> None:
        nonlocal frame_idx, paused, speed_mult, trial_idx
        try:
            c = chr(keycode)
        except ValueError:
            return
        if c == " ":
            paused = not paused
            print(f"[viewer] paused={paused}")
        elif c == "R":
            frame_idx = 0
            print("[viewer] reset to frame 0")
        elif c == "[":
            frame_idx = max(0, frame_idx - 1)
            print(f"[viewer] frame {frame_idx}")
        elif c == "]":
            frame_idx = min(horizon_frames - 1, frame_idx + 1)
            print(f"[viewer] frame {frame_idx}")
        elif c == ",":
            i = speeds.index(speed_mult) if speed_mult in speeds else 2
            speed_mult = speeds[max(0, i - 1)]
            print(f"[viewer] speed = {speed_mult}x")
        elif c == ".":
            i = speeds.index(speed_mult) if speed_mult in speeds else 2
            speed_mult = speeds[min(len(speeds) - 1, i + 1)]
            print(f"[viewer] speed = {speed_mult}x")
        elif c in ("n", "N"):
            trial_idx = (trial_idx + 1) % n_trials
            frame_idx = 0
            _print_trial_header()
        elif c in ("p", "P"):
            trial_idx = (trial_idx - 1) % n_trials
            frame_idx = 0
            _print_trial_header()
        elif c.isdigit() and c != "0":
            target = int(c) - 1
            if 0 <= target < n_trials:
                trial_idx = target
                frame_idx = 0
                _print_trial_header()

    print(
        "[viewer] keys: SPACE=pause R=reset [/]=step ,/.=speed n/p=trial 1-9=jump"
    )
    _print_trial_header()
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        # Wider view for scripted-demo NPZs (robots wander further).
        is_demo = segments is not None
        viewer.cam.distance = 8.0 if is_demo else 6.0
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -25.0 if is_demo else -20.0
        viewer.cam.lookat[:] = [1.5, 0.0, 0.8] if is_demo else [1.0, 0.0, 0.8]

        prev = time.time()
        while viewer.is_running():
            _set_qpos_for_frame(frame_idx)
            viewer.sync()

            # Announce segment transitions for scripted-demo NPZs.
            if segments is not None:
                seg_idx = -1
                for i, seg in enumerate(segments):
                    start = int(seg["start_frame"])
                    end = start + int(seg["n_frames"])
                    if start <= frame_idx < end:
                        seg_idx = i
                        break
                if seg_idx != last_segment_idx and seg_idx >= 0:
                    seg = segments[seg_idx]

                    def _fmt(a: float, b: float | None) -> str:
                        if b is None or abs(a - float(b)) < 1e-6:
                            return f"{a:+.2f}"
                        return f"{a:+.2f}->{float(b):+.2f}"

                    print(
                        f"[demo  frame {frame_idx:>3d}] step {seg_idx + 1}/"
                        f"{len(segments)}  {seg['label']:<16}  "
                        f"yaw={_fmt(seg['yaw_rate'], seg.get('yaw_rate_end'))}  "
                        f"vx(lat)={_fmt(seg['vel_x'], seg.get('vel_x_end'))}  "
                        f"vz(fwd)={_fmt(seg['vel_z'], seg.get('vel_z_end'))}"
                    )
                    last_segment_idx = seg_idx

            now = time.time()
            dt = now - prev
            prev = now
            if not paused:
                advance = max(1, int(round(dt * fps * speed_mult)))
                new_idx = frame_idx + advance
                if new_idx >= horizon_frames:
                    new_idx = 0
                    last_segment_idx = -1  # re-announce on loop
                frame_idx = new_idx

            time.sleep(max(0.0, 1.0 / (fps * speed_mult) - (time.time() - now)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
