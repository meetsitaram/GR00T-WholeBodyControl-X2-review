"""Gait-level metrics for kplanner replay outputs (pred vs actual).

Consumes the ``--save-npz`` file written by ``replay_pkl_through_kplanner.py``
(``pred``/``actual``: (T, 7+ndof) MuJoCo qpos, root quat wxyz, 30 fps) and
reports, for BOTH sequences:

  footsteps     swing count (contact->air transitions, both feet, via FK foot
                height with a per-foot adaptive contact threshold)
  cadence       steps / second
  path_len      integrated planar root path length (m) -- wandering shows here
  net_disp      straight-line start->end displacement (m)
  heading       direction of the net displacement vector (deg, world frame)
  final_dyaw    net root yaw change start->end (deg)
  foot_slide    mean planar foot speed while in contact (m/s) -- skating metric

Usage:
    python motionbricks/scripts/kplanner_gait_metrics.py --npz out.npz [--json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from motionbricks.helper.mujoco_helper_x2 import default_x2_mjcf_path_str  # noqa: E402

FPS = 30.0
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def _fk_feet(qpos: np.ndarray, mjcf: str) -> np.ndarray:
    """FK each frame -> (T, 2, 3) world positions of the two foot bodies."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in FOOT_BODIES]
    if any(b < 0 for b in bids):
        raise RuntimeError(f"foot bodies {FOOT_BODIES} not found in {mjcf}")
    T = qpos.shape[0]
    out = np.zeros((T, 2, 3))
    nq = min(model.nq, qpos.shape[1])
    for t in range(T):
        data.qpos[:nq] = qpos[t, :nq]
        mujoco.mj_kinematics(model, data)
        for i, b in enumerate(bids):
            out[t, i] = data.xpos[b]
    return out


def _wxyz_yaw(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def gait_metrics(qpos: np.ndarray, mjcf: str) -> dict:
    feet = _fk_feet(qpos, mjcf)                      # (T, 2, 3)
    root_xy = qpos[:, :2]
    yaw = _wxyz_yaw(qpos[:, 3:7])
    T = qpos.shape[0]
    dur = T / FPS

    steps = 0
    slide_speeds: list[float] = []
    for f in range(2):
        z = feet[:, f, 2]
        # adaptive contact threshold: lowest 20% of foot height + 2 cm
        thr = np.percentile(z, 20) + 0.02
        contact = z < thr
        # swing count = contact -> air transitions (debounced 3 frames)
        c = contact.astype(int)
        trans = np.where((c[:-1] == 1) & (c[1:] == 0))[0]
        # debounce: require >=3 air frames after lift-off
        good = [i for i in trans if (~contact[i + 1:i + 4]).all()]
        steps += len(good)
        # slide: planar foot speed while in contact
        v = np.linalg.norm(np.diff(feet[:, f, :2], axis=0), axis=1) * FPS
        if contact[:-1].any():
            slide_speeds.append(float(v[contact[:-1]].mean()))

    d_xy = np.diff(root_xy, axis=0)
    path_len = float(np.linalg.norm(d_xy, axis=1).sum())
    net_vec = root_xy[-1] - root_xy[0]
    net_disp = float(np.linalg.norm(net_vec))
    heading = float(math.degrees(math.atan2(net_vec[1], net_vec[0]))) if net_disp > 0.05 else 0.0
    final_dyaw = float(math.degrees(yaw[-1] - yaw[0]))

    return {
        "footsteps": steps,
        "cadence_hz": round(steps / dur, 2),
        "path_len_m": round(path_len, 3),
        "net_disp_m": round(net_disp, 3),
        "heading_deg": round(heading, 1),
        "final_dyaw_deg": round(final_dyaw, 1),
        "foot_slide_mps": round(float(np.mean(slide_speeds)) if slide_speeds else 0.0, 3),
        "straightness": round(net_disp / path_len, 2) if path_len > 0.1 else 1.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--mjcf", default=None)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    mjcf = args.mjcf or default_x2_mjcf_path_str()
    d = np.load(args.npz, allow_pickle=True)
    res = {
        "clip": str(d.get("clip_key", args.npz.stem)),
        "pred": gait_metrics(d["pred"], mjcf),
        "actual": gait_metrics(d["actual"], mjcf),
    }
    if args.json:
        print(json.dumps(res))
        return
    print(f"clip: {res['clip']}")
    keys = list(res["actual"].keys())
    print(f"  {'metric':<16}{'actual':>10}{'pred':>10}{'delta':>10}")
    for k in keys:
        a, p = res["actual"][k], res["pred"][k]
        print(f"  {k:<16}{a:>10}{p:>10}{round(p - a, 3):>10}")


if __name__ == "__main__":
    main()
