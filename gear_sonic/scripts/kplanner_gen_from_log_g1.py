"""Drive the HIGH-QUALITY G1 STOCK kplanner OFFLINE from the reconstructed
intent schedules, then kinematically retarget G1->X2 via the existing soma
pipeline. This is the accurate path (the recordings were driven by the G1
stock planner); the X2-planner variant (kplanner_gen_from_log.py) is kept
only for comparison.

Per key:
  1. reconstruct intent timeline: idle lead-in (~1s) -> active (~7s)
     -> idle/zero-vel tail (~1.5s)  [KEEP head+tail: start/stop transitions]
  2. drive load_g1_planner (pose-template, mode_map {3:2} since G1 has no run)
  3. dump G1 mujoco-qpos -> retargeter-ready G1 CSV (write_g1_csv)
Then g1_captures_to_x2_motion_pkl.py --no-trim retargets all clips to one X2 pkl.

Channels: intent = [yaw_rate_rad_s, vel_x(lateral), vel_z(forward), hip_h].
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "motionbricks"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from gear_sonic.scripts.record_motion_to_pkl import write_g1_csv  # noqa: E402
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    X2_DOF_AXIS, X2_NUM_BODIES,
)

G1_CLIP = REPO / "motionbricks/out/G1-clip.ckpt"
SEED_IDX = 2  # neutral walk seed (G1-clip order: 0=idle 2=walk 11=walk_gun)

MODE_IDX = {"IDLE": 0, "SLOW_WALK": 1, "WALK": 2, "RUN": 3}
G1_MODE_MAP = {3: 2}  # G1 has no run clip -> realize run as walk mode
DEFAULT_SPEED = {"SLOW_WALK": 0.4, "WALK": 0.8, "RUN": 1.5}
YAW_MAG = 0.4
FPS = 30
IDLE_LEAD_S = 1.0
ACTIVE_S = 7.0
IDLE_TAIL_S = 1.5


@dataclass
class Step:
    label: str
    duration_s: float
    yaw_rate: float
    vel_x: float
    vel_z: float
    mode_idx: int


def schedule_for_segment(sig: dict):
    mode = sig["mode"]
    m_idx = MODE_IDX.get(mode, 1)
    tvel = sig["target_vel"]
    speed = tvel if tvel and tvel > 0 else DEFAULT_SPEED.get(mode, 0.4)
    sign = sig.get("turn_sign", 0) or 1

    steps = [Step("idle_lead", IDLE_LEAD_S, 0.0, 0.0, 0.0, MODE_IDX["IDLE"])]
    if sig["back"]:
        steps.append(Step("back", ACTIVE_S, 0.0, 0.0, -speed, m_idx))
    elif sig["circle"]:
        steps.append(Step("circle", ACTIVE_S, sign * YAW_MAG, 0.0, speed, m_idx))
    elif sig["turn"]:
        half = ACTIVE_S / 2.0
        steps.append(Step("turn_a", half, sign * YAW_MAG, 0.0, speed, m_idx))
        steps.append(Step("turn_b", half, -sign * YAW_MAG, 0.0, speed, m_idx))
    else:
        steps.append(Step("straight", ACTIVE_S, 0.0, 0.0, speed, m_idx))
    # idle/zero-velocity tail so the planner decelerates & settles (stop transition)
    steps.append(Step("idle_tail", IDLE_TAIL_S, 0.0, 0.0, 0.0, MODE_IDX["IDLE"]))
    return steps, speed


def _load_g1_seed(idx: int):
    sd = torch.load(str(G1_CLIP), map_location="cpu", weights_only=False)
    n = int(sd["num_frames_per_clip"][idx])
    return sd["mujoco_qpos"][idx, :n].numpy().astype(np.float32)


def _load_planner(device):
    from motionbricks.motion_backbone.inference.load_g1_planner import (
        G1PlannerPaths, load_g1_planner,
    )
    paths = G1PlannerPaths.default()
    paths.validate()
    return load_g1_planner(paths, device=device, clip_library_ckpt=G1_CLIP)


def _drive(planner, seed_qpos, steps, hip_h, device, base_seed=0):
    """base_seed varies the pose-template segment sampling per clip so keys that
    share the same intent signature (distinct human takes) become distinct gait
    realizations rather than byte-identical duplicates. Deterministic per clip."""
    seed_window = seed_qpos[:64]
    planner.reset(torch.from_numpy(seed_window).to(device))
    ctr = [0]

    def intent(s):
        return torch.tensor([s.yaw_rate, s.vel_x, s.vel_z, hip_h],
                            device=device, dtype=torch.float32)

    def replan(s):
        planner.replan_with_pose_template(
            intent(s), mode_idx=G1_MODE_MAP.get(s.mode_idx, s.mode_idx),
            random_seed=base_seed * 1000 + ctr[0])
        ctr[0] += 1

    replan(steps[0])
    qdim = int(planner.frames["mujoco_qpos"].shape[-1])
    chunks = []
    for s in steps:
        n = max(1, int(round(s.duration_s * FPS)))
        replan(s)
        chunk = np.zeros((n, qdim), dtype=np.float32)
        for i in range(n):
            if planner.should_replan():
                replan(s)
            chunk[i] = planner.get_next_frame().detach().cpu().numpy()
        chunks.append(chunk)
    return np.concatenate(chunks, axis=0)


def qpos_to_g1_entry(qpos: np.ndarray) -> dict:
    """G1 mujoco qpos [T,36] (trans3 + quat_wxyz4 + dof29) -> entry for write_g1_csv."""
    return {
        "root_trans_offset": qpos[:, :3].astype(np.float32),
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]].astype(np.float32),  # wxyz->xyzw
        "dof": qpos[:, 7:7 + 29].astype(np.float32),
    }


def resample_entry(e: dict, dst_fps: int, src_fps: int = FPS) -> dict:
    """Resample a motion entry to dst_fps: linear on trans+dof, slerp on root_rot,
    then rebuild pose_aa exactly like build_entry. Faithful up/down-sample of a
    kinematic clip (src_fps is the planner-native rate)."""
    from scipy.spatial.transform import Slerp
    T = e["dof"].shape[0]
    dur = (T - 1) / src_fps
    Tn = max(2, int(round(dur * dst_fps)) + 1)
    ts = np.arange(T) / src_fps
    tn = np.linspace(0.0, ts[-1], Tn)
    trans = np.stack([np.interp(tn, ts, e["root_trans_offset"][:, i]) for i in range(3)], 1)
    dof = np.stack([np.interp(tn, ts, e["dof"][:, j]) for j in range(e["dof"].shape[1])], 1)
    quat = Slerp(ts, R.from_quat(e["root_rot"]))(tn).as_quat()
    ndof = dof.shape[1]
    ax = np.asarray(X2_DOF_AXIS, np.float32)
    pa = np.zeros((Tn, X2_NUM_BODIES, 3), np.float32)
    pa[:, 1:ndof + 1, :] = ax[None] * dof[:, :, None]
    pa[:, 0, :] = R.from_quat(quat).as_rotvec()
    return {
        "root_trans_offset": trans.astype(np.float32),
        "root_rot": quat.astype(np.float32),
        "dof": dof.astype(np.float32),
        "pose_aa": pa,
        "smpl_joints": np.zeros((Tn, 24, 3), np.float32),
        "fps": int(dst_fps),
    }


def select_keys(rows_list, want_all: bool, explicit):
    """Return ordered list of (unique_key, motion_key, sig). Duplicate motion-key
    names (genuine re-records) get a _r2/_r3 suffix so all invocations survive."""
    out, seen = [], {}
    src = rows_list if want_all else [r for r in rows_list if r["motion_key"] in set(explicit)]
    for r in src:
        if r["seg_idx"] is None:
            continue
        name = r["motion_key"]
        seen[name] = seen.get(name, 0) + 1
        uk = name if seen[name] == 1 else f"{name}_r{seen[name]}"
        out.append((uk, name, r["seg_sig"], r.get("flag", "")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alignment", type=Path,
                    default=REPO / "out/kplanner_gen_proof/alignment.json")
    ap.add_argument("--keys", nargs="*", default=[
        "slow_walk_0.3_001", "slow_walk_0.6_001", "slow_walk_turns_0.2_002",
        "slow_walk_back_0.3_001", "walk_circle_001", "run_001"])
    ap.add_argument("--all", action="store_true", help="generate every matched key (~64)")
    ap.add_argument("--out-dir", type=Path, default=REPO / "out/kplanner_gen_proof_g1")
    ap.add_argument("--out-pkl", type=Path, default=None,
                    help="30fps output pkl (default <out-dir>/proof_batch_g1retarget.pkl)")
    ap.add_argument("--also-fps", default="", help="comma list of extra fps grids, e.g. 50")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-retarget", action="store_true")
    args = ap.parse_args()

    data = json.loads(args.alignment.read_text())
    rows_list = data["rows"]  # ordered
    selected = select_keys(rows_list, args.all, args.keys)
    print(f"[select] {len(selected)} clips ({'ALL matched' if args.all else 'explicit keys'})")

    print(f"[load] G1 stock planner on {args.device}")
    planner = _load_planner(args.device)
    seed = _load_g1_seed(SEED_IDX)
    hip_h = float(seed[0:4, 2].mean())
    print(f"[seed] G1-clip idx {SEED_IDX}, hip_h={hip_h:.3f}, planner.fps={planner.fps}")

    csv_dir = args.out_dir / "g1_csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    for f in csv_dir.glob("*.csv"):
        f.unlink()  # clean stale CSVs so retarget only sees this run
    meta = {}
    for ci, (uk, name, sig, flag) in enumerate(selected):
        steps, speed = schedule_for_segment(sig)
        qpos = _drive(planner, seed, steps, hip_h, args.device, base_seed=ci + 1)
        write_g1_csv(qpos_to_g1_entry(qpos), csv_dir / f"{uk}.csv")
        meta[uk] = dict(motion_key=name, label_speed=speed, flag=flag,
                        g1_frames=len(qpos))
        print(f"[gen-g1] {uk:<28} T={len(qpos)} ({len(qpos)/FPS:.1f}s) flag={flag or '-'}")

    if args.skip_retarget:
        print("[done] skipped retarget"); return

    import joblib
    out_pkl = args.out_pkl or (args.out_dir / "proof_batch_g1retarget.pkl")
    print("[retarget] G1 -> X2 (soma venv, --no-trim to keep idle head/tail) ...")
    subprocess.run([sys.executable, str(REPO / "gear_sonic/scripts/g1_captures_to_x2_motion_pkl.py"),
                    "--g1-dir", str(csv_dir), "--out-pkl", str(out_pkl),
                    "--fps", str(FPS), "--no-trim"], check=True)
    m30 = joblib.load(out_pkl)

    # extra fps grids by resampling the 30fps retarget output
    extra = [int(x) for x in args.also_fps.split(",") if x.strip()]
    for f in extra:
        mf = {k: resample_entry(e, f) for k, e in m30.items()}
        p = out_pkl.with_name(out_pkl.stem.replace("_30fps", "") + f"_{f}fps.pkl")
        joblib.dump(mf, p, compress=3)
        print(f"[fps] wrote {p}  ({len(mf)} clips @ {f}fps)")

    # ---- report / degeneracy scan ----
    report = []
    lead, tail, h = int(IDLE_LEAD_S * FPS), int(IDLE_TAIL_S * FPS), int(0.5 * FPS)
    print(f"\n=== FULL G1-RETARGET SET ({len(m30)} clips @ {FPS}fps) ===")
    print(f"{'key':<30}{'T':>4} {'v(meas)':>8}{'v(lbl)':>7} {'disp':>6}{'path':>6} {'flag':>6} {'status'}")
    for uk, name, sig, flag in selected:
        if uk not in m30:
            report.append(dict(key=uk, status="RETARGET_FAILED"));
            print(f"{uk:<30}  --   RETARGET_FAILED"); continue
        e = m30[uk]
        xy = e["root_trans_offset"][:, :2]
        T = len(xy)
        nan = bool(np.isnan(e["dof"]).any() or np.isnan(xy).any())
        act = xy[lead:T - tail]
        path_len = float(np.sum(np.linalg.norm(np.diff(act, axis=0), axis=1))) if len(act) > 1 else 0.0
        disp = float(np.linalg.norm(act[-1] - act[0])) if len(act) > 1 else 0.0
        act_s = max(1e-6, len(act) / FPS)
        v = path_len / act_s
        head_v = float(np.linalg.norm(np.diff(xy[:h], axis=0), axis=1).mean() * FPS) if T > h else 0
        tail_v = float(np.linalg.norm(np.diff(xy[-h:], axis=0), axis=1).mean() * FPS) if T > h else 0
        speed = meta[uk]["label_speed"]
        is_back = sig["back"]
        degenerate = nan or (not is_back and path_len < 0.15)
        status = "NAN" if nan else ("DEGENERATE" if degenerate else "ok")
        report.append(dict(key=uk, motion_key=name, T=T, label_speed=speed,
                           meas_speed_mps=round(v, 3), net_disp_m=round(disp, 2),
                           path_len_m=round(path_len, 2), head_speed_mps=round(head_v, 3),
                           tail_speed_mps=round(tail_v, 3), flag=flag, status=status))
        print(f"{uk:<30}{T:>4} {v:>8.2f}{speed:>7.2f} {disp:>6.2f}{path_len:>6.2f} "
              f"{flag or '-':>6} {status}")
    (args.out_dir / "full_report.json").write_text(json.dumps(report, indent=2))
    n_bad = sum(1 for r in report if r.get("status") not in ("ok", None))
    print(f"\n[summary] {len(m30)} clips generated, {n_bad} flagged (degenerate/nan/failed)")
    print(f"[out] {out_pkl}")
    for f in extra:
        print(f"[out] {out_pkl.with_name(out_pkl.stem.replace('_30fps','') + f'_{f}fps.pkl')}")
    print(f"[out] {args.out_dir/'full_report.json'}")


if __name__ == "__main__":
    main()
