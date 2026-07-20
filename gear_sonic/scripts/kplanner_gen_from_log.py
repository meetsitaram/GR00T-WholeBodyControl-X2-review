"""Drive the fine-tuned X2 kplanner OFFLINE (no SONIC, no physics) from the
intent schedules reconstructed by kplanner_log_parse_align.py, and dump each
clip to the motion-lib pkl schema at 30 fps.

Adapts motionbricks/scripts/kplanner_validation/run_scripted_demo.py:
  - _apply_ckpt_overrides  (point the 3 ckpts at the staged g1ret dirs)
  - the _run_schedule driving loop (reset -> replan -> get_next_frame)

Channel convention (verified vs neural_planner docstring):
  intent = [yaw_rate_rad_s, vel_x(lateral), vel_z(forward), hip_h]

Timing template (log has NO timestamps): ~1s idle lead-in + ~7s active @30fps.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
import sys
for p in (REPO, REPO / "motionbricks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from scipy.spatial.transform import Rotation as R  # noqa: E402
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    X2_DOF_AXIS, X2_NUM_BODIES, X2_NUM_DOF,
)

STAGED = REPO / "out/kplanner_g1ret_eval/staged"
VQVAE = STAGED / "motionbricks_vqvae_x2/version_1/checkpoints/vqvae_g1ret_250k.ckpt"
POSE = STAGED / "motionbricks_pose_x2/version_1/checkpoints/pose_g1ret_250k.ckpt"
ROOT = STAGED / "motionbricks_root_x2/version_1/checkpoints/root_g1ret_300k.ckpt"

SEED_PKL = REPO / "gear_sonic/data/motions/x2_ultra_locowalk.pkl"
SEED_KEY = "Loop_Forward_Walk_001__A018"
CLIP_LIB = REPO / "motionbricks/out/X2-clip.ckpt"

MODE_IDX = {"IDLE": 0, "SLOW_WALK": 1, "WALK": 2, "RUN": 3}
# forward-speed defaults when target_vel is the -1 sentinel (WALK/RUN)
DEFAULT_SPEED = {"SLOW_WALK": 0.4, "WALK": 0.8, "RUN": 1.5}
YAW_MAG = 0.4     # rad/s for turns (matches run_scripted_demo validation)
FPS = 30
IDLE_LEAD_S = 1.0
ACTIVE_S = 7.0


@dataclass
class Step:
    label: str
    duration_s: float
    yaw_rate: float
    vel_x: float
    vel_z: float
    mode_idx: int


def schedule_for_segment(seg_sig: dict) -> tuple[list, float]:
    """Reconstruct a [yaw, vx, vz, mode] timeline from a segment signature."""
    mode = seg_sig["mode"]
    m_idx = MODE_IDX.get(mode, 1)
    tvel = seg_sig["target_vel"]
    speed = tvel if tvel and tvel > 0 else DEFAULT_SPEED.get(mode, 0.4)
    is_turn = seg_sig["turn"]
    is_back = seg_sig["back"]
    is_circle = seg_sig["circle"]
    sign = seg_sig.get("turn_sign", 0) or 1

    steps = [Step("idle_lead", IDLE_LEAD_S, 0.0, 0.0, 0.0, MODE_IDX["IDLE"])]
    if is_back:
        steps.append(Step("back", ACTIVE_S, 0.0, 0.0, -speed, m_idx))
    elif is_circle:
        # sustained arc in the net-sweep direction (a big circle)
        steps.append(Step("circle", ACTIVE_S, sign * YAW_MAG, 0.0, speed, m_idx))
    elif is_turn:
        # "turns" plural -> S-curve: turn one way then the other while walking
        half = ACTIVE_S / 2.0
        steps.append(Step("turn_a", half, sign * YAW_MAG, 0.0, speed, m_idx))
        steps.append(Step("turn_b", half, -sign * YAW_MAG, 0.0, speed, m_idx))
    else:
        steps.append(Step("straight", ACTIVE_S, 0.0, 0.0, speed, m_idx))
    return steps, speed


def _load_seed():
    e = joblib.load(SEED_PKL)[SEED_KEY]
    trans = np.asarray(e["root_trans_offset"], np.float32)
    rot_xyzw = np.asarray(e["root_rot"], np.float32)
    dof = np.asarray(e["dof"], np.float32)
    rot_wxyz = rot_xyzw[:, [3, 0, 1, 2]]
    qpos = np.concatenate([trans, rot_wxyz, dof], axis=-1).astype(np.float32)
    return qpos


def _load_planner(device):
    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths, load_x2_planner,
    )
    paths = X2PlannerPaths.default()
    paths.vqvae_ckpt = VQVAE; paths.vqvae_version_dir = VQVAE.resolve().parents[1]
    paths.pose_ckpt = POSE; paths.pose_version_dir = POSE.resolve().parents[1]
    paths.root_ckpt = ROOT; paths.root_version_dir = ROOT.resolve().parents[1]
    paths.validate()
    return load_x2_planner(paths, device=device, clip_library_ckpt=CLIP_LIB)


def _drive(planner, seed_qpos, steps, hip_h, device, seed_frame=0):
    seed_window = seed_qpos[seed_frame:seed_frame + 64]
    planner.reset(torch.from_numpy(seed_window).to(device))

    def intent(s: Step):
        return torch.tensor([s.yaw_rate, s.vel_x, s.vel_z, hip_h],
                            device=device, dtype=torch.float32)

    planner.replan_with_pose_template(intent(steps[0]), mode_idx=steps[0].mode_idx,
                                      random_seed=0)
    qdim = int(planner.frames["mujoco_qpos"].shape[-1])
    chunks = []
    for s in steps:
        n = max(1, int(round(s.duration_s * FPS)))
        planner.replan_with_pose_template(intent(s), mode_idx=s.mode_idx, random_seed=0)
        chunk = np.zeros((n, qdim), dtype=np.float32)
        for i in range(n):
            if planner.should_replan():
                planner.replan_with_pose_template(intent(s), mode_idx=s.mode_idx,
                                                  random_seed=0)
            chunk[i] = planner.get_next_frame().detach().cpu().numpy()
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def qpos_to_entry(qpos: np.ndarray, fps: int = FPS) -> dict:
    """MuJoCo qpos [T,38] (trans3 + quat_wxyz4 + dof31) -> motion-lib entry,
    reproducing record_motion_to_pkl.build_entry field-for-field."""
    trans = qpos[:, :3].astype(np.float32)
    quat_wxyz = qpos[:, 3:7]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]].astype(np.float32)
    dof = qpos[:, 7:7 + X2_NUM_DOF].astype(np.float32)
    ax = np.asarray(X2_DOF_AXIS, np.float32)
    pose_aa = np.zeros((len(qpos), X2_NUM_BODIES, 3), dtype=np.float32)
    pose_aa[:, 1:X2_NUM_DOF + 1, :] = ax[None, :, :] * dof[:, :, None]
    pose_aa[:, 0, :] = R.from_quat(quat_xyzw).as_rotvec().astype(np.float32)
    return {
        "root_trans_offset": trans,
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": quat_xyzw,
        "smpl_joints": np.zeros((len(qpos), 24, 3), dtype=np.float32),
        "fps": int(fps),
        "x2_record_source": "kplanner_offline:g1ret_250k/300k+X2-clip",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", type=Path,
                    default=REPO / "out/kplanner_gen_proof/alignment.json")
    ap.add_argument("--keys", nargs="*", default=[
        "slow_walk_0.3_001", "slow_walk_0.6_001", "slow_walk_turns_0.2_002",
        "slow_walk_back_0.3_001", "walk_circle_001", "run_001"])
    ap.add_argument("--out-dir", type=Path, default=REPO / "out/kplanner_gen_proof")
    ap.add_argument("--merged", type=Path,
                    default=REPO / "out/kplanner_gen_proof/proof_batch_30fps.pkl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    data = json.loads(args.alignment.read_text())
    rows = {r["motion_key"]: r for r in data["rows"] if r["seg_idx"] is not None}

    print(f"[load] planner (g1ret ckpts) on {args.device}")
    planner = _load_planner(args.device)
    seed = _load_seed()
    hip_h = float(seed[0:4, 2].mean())
    print(f"[seed] hip_h={hip_h:.3f}  planner.fps={planner.fps}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged = {}
    report = []
    for key in args.keys:
        if key not in rows:
            print(f"[skip] {key}: not matched in alignment"); continue
        sig = rows[key]["seg_sig"]
        steps, speed = schedule_for_segment(sig)
        qpos = _drive(planner, seed, steps, hip_h, args.device)
        entry = qpos_to_entry(qpos)
        merged[key] = entry
        joblib.dump({key: entry}, args.out_dir / f"{key}.pkl", compress=3)

        xy = entry["root_trans_offset"][:, :2]
        # measure over the active window (skip 1s lead-in)
        lead = int(IDLE_LEAD_S * FPS)
        disp = float(np.linalg.norm(xy[-1] - xy[lead]))
        path_len = float(np.sum(np.linalg.norm(np.diff(xy[lead:], axis=0), axis=1)))
        active_s = (len(qpos) - lead) / FPS
        meas_speed = path_len / active_s
        report.append(dict(key=key, mode=sig["mode"], label_speed=speed,
                           frames=len(qpos), net_disp_m=round(disp, 2),
                           path_len_m=round(path_len, 2),
                           meas_speed_mps=round(meas_speed, 3),
                           expect_disp_m=round(speed * ACTIVE_S, 2)))
        print(f"[gen] {key:<26} T={len(qpos):>4} disp={disp:5.2f}m path={path_len:5.2f}m "
              f"v={meas_speed:.2f} (label {speed:.2f})")

    joblib.dump(merged, args.merged, compress=3)
    print(f"\n[out] merged proof pkl: {args.merged}  ({len(merged)} clips @ {FPS}fps)")
    (args.out_dir / "proof_report.json").write_text(json.dumps(report, indent=2))
    print(f"[out] {args.out_dir / 'proof_report.json'}")


if __name__ == "__main__":
    main()
