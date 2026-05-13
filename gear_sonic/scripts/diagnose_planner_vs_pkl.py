"""Diagnose: does the planner's joint stream match the deploy-PKL's frame data?

Background
----------

Two paths feed the X2 deploy with motion references:

1. Direct PKL: ``deploy_x2.sh sim --motion x2_browser_side_left_step.pkl``
   -- the C++ ``PklMotionReference`` reads the PKL frame-by-frame and
   samples ``frame[(t * fps) mod N]`` per policy tick. This path is
   confirmed to make the side-step actually move under SONIC.

2. Planner ZMQ: ``run_planner_smoke.sh --with-deploy --demo
   side_steps_only_smoke.yaml`` -- ``HeuristicPlanner.step()`` emits 50 Hz
   ``StreamFrame`` records, ``PosePublisher`` packs them into the
   ``pose`` ZMQ topic, the C++ ``ZmqPoseInputSource`` decodes and feeds
   them as the reference. This path produces no visible side-step.

This script verifies whether the **planner side** of path (2) is
faithful to the deploy-PKL of path (1) at the joint-trajectory level.
Concretely we compare three trajectories for the side-step bin:

  A. SOURCE_PKL: the same PKL the working ``--motion`` path replays
     (default: ``x2_browser_side_left_step.pkl``). 30 Hz, recipe-baked
     joints.

  B. RUNTIME_PRIM: the same bin loaded by ``load_primitives_pkl`` (the
     planner's runtime), which resamples 30->50 Hz. Same joint arc as A
     but denser sampling.

  C. PLANNER_EMIT: the actual ``StreamFrame.joint_pos_mj`` the planner
     would publish on the wire when given the side-step command. Drained
     from a synthetic in-process run (no ZMQ, no deploy, no MuJoCo) so
     it isolates the state-machine + alignment + blend logic.

Reports per-joint max / mean abs diffs for the leg + waist subset that
actually move during a side-step (hip pitch/roll, knee, ankle, waist
roll). Optional ``--csv`` writes the three trajectories to disk for
plotting.

Usage::

    .venv/bin/python -m gear_sonic.scripts.diagnose_planner_vs_pkl \\
        --bin side_left_step

    # Or for the right side:
    .venv/bin/python -m gear_sonic.scripts.diagnose_planner_vs_pkl \\
        --bin side_right_step \\
        --source-pkl data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_browser_side_right_step.pkl

If A vs B differs > tolerance, the 30->50 Hz resampler is mangling the
clip. If B vs C differs > tolerance, the state machine / blend logic
is dropping or distorting frames. If both check out, the planner side
is faithful and the discrepancy must lie downstream (ZmqPoseInputSource
ingest, Anchor() not called, looping-vs-one-shot semantics, etc).
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

from gear_sonic.utils.planner.constants import (  # noqa: E402
    LEFT_ANKLE_PITCH_IDX,
    LEFT_HIP_PITCH_IDX,
    LEFT_KNEE_IDX,
    MUJOCO_JOINT_NAMES,
    RIGHT_ANKLE_PITCH_IDX,
    RIGHT_HIP_PITCH_IDX,
    RIGHT_KNEE_IDX,
    WAIST_ROLL_IDX,
)
from gear_sonic.utils.planner.registry import load_bin_specs  # noqa: E402
from gear_sonic.utils.planner.state_machine import (  # noqa: E402
    HeuristicPlanner,
    LocomotionCommand,
    OUTPUT_FPS,
    PlannerState,
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

# Joints that actually move during a side-step. We focus the diff
# report on these because the arms+head are frozen by the recipe and
# would just dilute the signal with constant zeros.
SIDE_STEP_DRIVE_JOINTS: dict[str, int] = {
    "L_hip_pitch": LEFT_HIP_PITCH_IDX,
    "L_hip_roll": 1,
    "L_hip_yaw": 2,
    "L_knee": LEFT_KNEE_IDX,
    "L_ankle_pitch": LEFT_ANKLE_PITCH_IDX,
    "L_ankle_roll": 5,
    "R_hip_pitch": RIGHT_HIP_PITCH_IDX,
    "R_hip_roll": 7,
    "R_hip_yaw": 8,
    "R_knee": RIGHT_KNEE_IDX,
    "R_ankle_pitch": RIGHT_ANKLE_PITCH_IDX,
    "R_ankle_roll": 11,
    "waist_roll": WAIST_ROLL_IDX,
}


def _load_source_pkl(pkl_path: Path) -> tuple[np.ndarray, float, str]:
    """Return (dof[T,31], fps, motion_key) from a deploy-format PKL."""
    if not pkl_path.is_file():
        raise FileNotFoundError(f"source PKL not found: {pkl_path}")
    payload = joblib.load(pkl_path)
    if not isinstance(payload, dict) or len(payload) == 0:
        raise ValueError(f"unexpected PKL layout: {pkl_path}")
    motion_key = next(iter(payload))
    entry = payload[motion_key]
    dof = np.asarray(entry["dof"], dtype=np.float64)
    fps = float(entry.get("fps", 30.0))
    if dof.ndim != 2 or dof.shape[1] != 31:
        raise ValueError(
            f"bad dof shape in {pkl_path}: {dof.shape} (expected (T, 31))"
        )
    return dof, fps, motion_key


def _planner_emit_for_bin(
    bin_name: str,
    primitives_pkl: Path,
    bins_yaml: Path,
    pre_idle_s: float = 1.0,
    post_idle_s: float = 0.5,
) -> tuple[list[dict], dict]:
    """Drive the planner state machine offline; capture every emitted frame.

    Returns (frames_log, runtime_primitive_view) where ``frames_log`` is a
    list of per-tick dicts and ``runtime_primitive_view`` exposes the
    50 Hz resampled primitive data the planner internally holds for the
    requested bin.
    """
    bin_specs = load_bin_specs(bins_yaml)
    bin_family = {name: spec.family for name, spec in bin_specs.items()}
    primitives = load_primitives_pkl(primitives_pkl, bin_family)

    if bin_name not in primitives:
        raise KeyError(
            f"bin {bin_name!r} not found in {primitives_pkl}; "
            f"have: {sorted(primitives.keys())}"
        )

    intent = bin_name.replace("_step", "").replace("side_", "side_")
    intent_map = {
        "side_left_step": "side_left",
        "side_right_step": "side_right",
        "fwd_step": "fwd_step",
        "back_step": "back_step",
    }
    intent = intent_map.get(bin_name, bin_name)

    planner = HeuristicPlanner(primitives=primitives)

    pre_idle_ticks = int(round(pre_idle_s * OUTPUT_FPS))
    for _ in range(pre_idle_ticks):
        planner.step()

    planner.enqueue(LocomotionCommand(intent=intent, magnitude="default"))

    # Step until we've drained the side-step bin and a bit of trailing
    # idle. Side-step is 90 frames @ 30Hz = 150 frames @ 50Hz; allow
    # generous buffer for blend-in (6 frames) + blend-out + post-idle.
    bin_n_50hz = primitives[bin_name].dof.shape[0]
    play_budget = 50 + bin_n_50hz + 50 + int(post_idle_s * OUTPUT_FPS)

    frames_log: list[dict] = []
    for tick in range(pre_idle_ticks + play_budget):
        f = planner.step()
        frames_log.append({
            "tick": tick,
            "state": f.state.value,
            "bin_name": f.bin_name,
            "seam_blend": f.seam_blend,
            "joint_pos_mj": f.joint_pos_mj.copy(),
            "root_quat_xyzw": f.root_quat_xyzw.copy(),
            "frame_index": f.frame_index,
        })

    rp = primitives[bin_name]
    runtime_view = {
        "dof_50hz": rp.dof.copy(),
        "rot_50hz": rp.root_rot_xyzw.copy(),
        "trans_50hz": rp.root_trans.copy(),
        "fps": rp.fps,
    }
    return frames_log, runtime_view


def _format_diff(name_to_idx: dict[str, int], a: np.ndarray, b: np.ndarray,
                 tol_rad: float = 1e-3) -> str:
    """Side-by-side per-joint max abs diff.  ``a`` and ``b`` are (T,31)."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    rows = []
    rows.append(
        f"  {'joint':<16} {'max|diff|deg':>14} {'mean|diff|deg':>16} "
        f"{'verdict':>10}"
    )
    rows.append("  " + "-" * 60)
    for name, idx in name_to_idx.items():
        diff = a[:, idx] - b[:, idx]
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        verdict = "OK" if max_abs < tol_rad else "MISMATCH"
        rows.append(
            f"  {name:<16} {np.degrees(max_abs):>14.4f} "
            f"{np.degrees(mean_abs):>16.4f} {verdict:>10}"
        )
    return "\n".join(rows)


def _resample_pkl_to_50hz(dof_30hz: np.ndarray, src_fps: float) -> np.ndarray:
    """Time-align the source PKL onto the 50 Hz grid, mirroring the planner."""
    from gear_sonic.utils.planner.blending import resample_motion_30_to_50hz

    rot_dummy = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (dof_30hz.shape[0], 1))
    trans_dummy = np.zeros((dof_30hz.shape[0], 3))
    out_dof, _, _ = resample_motion_30_to_50hz(
        dof_30hz, rot_dummy, trans_dummy, src_fps, OUTPUT_FPS
    )
    return out_dof


def _find_play_segment(
    frames_log: list[dict], bin_name: str
) -> tuple[int, int]:
    """Return (start_tick_inclusive, end_tick_exclusive) of the PLAYING run."""
    play_ticks = [
        i for i, f in enumerate(frames_log)
        if f["state"] == PlannerState.PLAYING.value and f["bin_name"] == bin_name
    ]
    if not play_ticks:
        raise RuntimeError(
            f"planner never entered PLAYING state for bin {bin_name!r}; "
            f"states seen: {sorted({f['state'] for f in frames_log})}"
        )
    return play_ticks[0], play_ticks[-1] + 1


def diagnose(
    bin_name: str,
    source_pkl: Path,
    primitives_pkl: Path,
    bins_yaml: Path,
    csv_path: Path | None = None,
) -> int:
    print("=" * 78)
    print(f"  DIAGNOSE  bin = {bin_name}")
    print("=" * 78)
    src_dof, src_fps, motion_key = _load_source_pkl(source_pkl)
    print()
    print(f"A. SOURCE_PKL : {source_pkl.relative_to(_REPO_ROOT)}")
    print(f"               motion_key = {motion_key}")
    print(f"               shape      = {src_dof.shape}  fps={src_fps}  "
          f"duration={src_dof.shape[0]/src_fps:.2f}s")

    # Drive the planner offline to capture its actual emit.
    frames_log, runtime_view = _planner_emit_for_bin(
        bin_name, primitives_pkl, bins_yaml,
    )
    rt_dof_50 = runtime_view["dof_50hz"]
    print()
    print(f"B. RUNTIME_PRIM (50Hz resampled view): shape = {rt_dof_50.shape}  "
          f"duration={rt_dof_50.shape[0]/OUTPUT_FPS:.2f}s")

    # Slice the captured trajectory to just the PLAYING segment.
    seg_start, seg_end = _find_play_segment(frames_log, bin_name)
    emit_play = np.stack(
        [frames_log[i]["joint_pos_mj"] for i in range(seg_start, seg_end)]
    )
    n_blend_in = sum(
        1 for f in frames_log[:seg_start]
        if f["state"] == PlannerState.BLENDING.value
    )
    print()
    print(f"C. PLANNER_EMIT (drained from state machine):")
    print(f"               total ticks captured = {len(frames_log)}")
    print(f"               PLAYING segment      = ticks "
          f"[{seg_start}, {seg_end})  ({emit_play.shape[0]} frames "
          f"= {emit_play.shape[0]/OUTPUT_FPS:.2f}s)")
    print(f"               blend-in ticks       = {n_blend_in}")

    # ------------------------------------------------------------------
    # Diff A vs B (resampling fidelity)
    # ------------------------------------------------------------------
    pkl_50 = _resample_pkl_to_50hz(src_dof, src_fps)
    ab_n = min(pkl_50.shape[0], rt_dof_50.shape[0])
    print()
    print(f"--- A vs B  (source PKL resampled 30->50Hz vs runtime primitive) ---")
    print(f"   common length = {ab_n} frames")
    print(_format_diff(SIDE_STEP_DRIVE_JOINTS, pkl_50[:ab_n], rt_dof_50[:ab_n]))

    # ------------------------------------------------------------------
    # Diff B vs C (state machine fidelity to its own primitive)
    # ------------------------------------------------------------------
    bc_n = min(rt_dof_50.shape[0], emit_play.shape[0])
    print()
    print(f"--- B vs C  (runtime primitive vs PLAYING-segment emit) ---")
    print(f"   common length = {bc_n} frames")
    print(_format_diff(SIDE_STEP_DRIVE_JOINTS, rt_dof_50[:bc_n], emit_play[:bc_n]))

    # ------------------------------------------------------------------
    # Diff A vs C (end-to-end: source PKL vs what the planner emits)
    # ------------------------------------------------------------------
    ac_n = min(pkl_50.shape[0], emit_play.shape[0])
    print()
    print(f"--- A vs C  (source PKL resampled 30->50Hz vs planner emit) ---")
    print(f"   common length = {ac_n} frames")
    print(_format_diff(SIDE_STEP_DRIVE_JOINTS, pkl_50[:ac_n], emit_play[:ac_n]))

    # ------------------------------------------------------------------
    # Root-quat / yaw trajectory diagnostic
    # ------------------------------------------------------------------
    print()
    print(f"--- ROOT-QUAT TRAJECTORY (planner emit during PLAYING) ---")
    emit_quat = np.stack(
        [frames_log[i]["root_quat_xyzw"] for i in range(seg_start, seg_end)]
    )
    rt_quat = runtime_view["rot_50hz"]
    rt_trans = runtime_view["trans_50hz"]
    yaw_emit = 2.0 * np.arctan2(emit_quat[:, 2], emit_quat[:, 3])
    yaw_rt = 2.0 * np.arctan2(rt_quat[:, 2], rt_quat[:, 3])
    print(f"   primitive's raw yaw  : start={np.degrees(yaw_rt[0]):.2f} deg, "
          f"end={np.degrees(yaw_rt[-1]):.2f} deg, "
          f"span={np.degrees(yaw_rt.max() - yaw_rt.min()):.2f} deg")
    print(f"   planner emit yaw     : start={np.degrees(yaw_emit[0]):.2f} deg, "
          f"end={np.degrees(yaw_emit[-1]):.2f} deg, "
          f"span={np.degrees(yaw_emit.max() - yaw_emit.min()):.2f} deg")
    print(f"   primitive root_trans : start={rt_trans[0]}, end={rt_trans[-1]}")
    print(f"                          dx={rt_trans[-1, 0]-rt_trans[0, 0]:+.3f}m  "
          f"dy={rt_trans[-1, 1]-rt_trans[0, 1]:+.3f}m  "
          f"dz={rt_trans[-1, 2]-rt_trans[0, 2]:+.3f}m")
    print(f"   NOTE: root_trans is NOT carried on the ZMQ pose wire (the "
          f"wire only carries joint_pos_mj + root_quat_xyzw + motion_token + "
          f"hand_joints + frame_index). PklMotionReference also doesn't "
          f"send root_trans to the policy -- it's not part of the obs.")

    # ------------------------------------------------------------------
    # Blend / state trajectory: show every state transition
    # ------------------------------------------------------------------
    print()
    print(f"--- STATE TIMELINE (every transition) ---")
    prev_state = None
    prev_bin = None
    transitions = []
    for f in frames_log:
        if f["state"] != prev_state or f["bin_name"] != prev_bin:
            transitions.append(f)
            prev_state = f["state"]
            prev_bin = f["bin_name"]
    rows = [f"  {'tick':>5}  {'state':<10} {'bin':<28} {'first joint dof[0]':>20}"]
    rows.append("  " + "-" * 70)
    for f in transitions:
        rows.append(
            f"  {f['tick']:>5}  {f['state']:<10} {f['bin_name']:<28} "
            f"{np.degrees(f['joint_pos_mj'][LEFT_HIP_PITCH_IDX]):>20.3f}"
        )
    print("\n".join(rows))

    # Now zoom in on the IDLE -> BLEND -> PLAY transition: show per-tick
    # joint values for L_hip_pitch and L_hip_roll (the two joints that
    # most reveal a side-step swing).
    blend_ticks = [
        i for i, f in enumerate(frames_log)
        if f["state"] == PlannerState.BLENDING.value
    ]
    if blend_ticks:
        print()
        print(f"--- BLEND-WINDOW DETAIL (idle -> {bin_name}) ---")
        zoom = list(range(max(0, blend_ticks[0] - 2), min(len(frames_log), blend_ticks[-1] + 3)))
        rows = [f"  {'tick':>5}  {'state':<10} "
                f"{'L_hip_p deg':>12}  {'L_hip_r deg':>12}  {'R_hip_p deg':>12}  "
                f"{'R_hip_r deg':>12}  {'L_knee deg':>12}"]
        rows.append("  " + "-" * 90)
        for i in zoom:
            f = frames_log[i]
            j = f["joint_pos_mj"]
            rows.append(
                f"  {i:>5}  {f['state']:<10} "
                f"{np.degrees(j[LEFT_HIP_PITCH_IDX]):>12.3f}  "
                f"{np.degrees(j[1]):>12.3f}  "
                f"{np.degrees(j[RIGHT_HIP_PITCH_IDX]):>12.3f}  "
                f"{np.degrees(j[7]):>12.3f}  "
                f"{np.degrees(j[LEFT_KNEE_IDX]):>12.3f}"
            )
        print("\n".join(rows))

    # Quick energy check: does the planner emit have ANY motion at all?
    print()
    print("--- ENERGY (rad std over time, per drive-joint) ---")
    rows = [f"  {'joint':<16} {'A_pkl':>10} {'B_runtime':>12} {'C_emit':>10}"]
    rows.append("  " + "-" * 50)
    for name, idx in SIDE_STEP_DRIVE_JOINTS.items():
        a_std = float(np.std(pkl_50[:ab_n, idx]))
        b_std = float(np.std(rt_dof_50[:bc_n, idx]))
        c_std = float(np.std(emit_play[:, idx]))
        rows.append(
            f"  {name:<16} {np.degrees(a_std):>10.3f} "
            f"{np.degrees(b_std):>12.3f} {np.degrees(c_std):>10.3f}"
        )
    print("\n".join(rows))

    if csv_path is not None:
        # Stack the three trajectories trimmed to common length for plotting.
        n = min(pkl_50.shape[0], rt_dof_50.shape[0], emit_play.shape[0])
        out = {
            "fps": OUTPUT_FPS,
            "joints": list(SIDE_STEP_DRIVE_JOINTS.keys()),
            "indices": list(SIDE_STEP_DRIVE_JOINTS.values()),
            "joint_names_full": list(MUJOCO_JOINT_NAMES),
            "A_source_pkl_50hz": pkl_50[:n],
            "B_runtime_primitive_50hz": rt_dof_50[:n],
            "C_planner_emit_play_segment": emit_play[:n],
            "frames_log": frames_log,
        }
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(csv_path, **{
            k: v for k, v in out.items()
            if isinstance(v, (np.ndarray, int, float))
        })
        print(f"\n[saved] full trajectories to {csv_path}")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--bin", default="side_left_step",
                   help="Primitive bin to diagnose (default: side_left_step).")
    p.add_argument(
        "--source-pkl", type=Path, default=None,
        help="Deploy-format PKL to compare against (the file you'd pass to "
             "deploy_x2.sh sim --motion). Defaults to "
             "data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_browser_<bin>.pkl"
    )
    p.add_argument("--primitives-pkl", type=Path, default=_DEFAULT_PRIMS_PKL)
    p.add_argument("--bins-yaml", type=Path, default=_DEFAULT_BINS_YAML)
    p.add_argument(
        "--csv", type=Path, default=None,
        help="Optional .npz path to dump A/B/C trajectories for offline plotting."
    )
    args = p.parse_args()

    src_pkl = args.source_pkl or (_DEFAULT_BAKED_DIR / f"x2_browser_{args.bin}.pkl")

    return diagnose(
        bin_name=args.bin,
        source_pkl=src_pkl,
        primitives_pkl=args.primitives_pkl,
        bins_yaml=args.bins_yaml,
        csv_path=args.csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
