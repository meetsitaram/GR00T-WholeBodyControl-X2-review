#!/usr/bin/env python3
"""Measure X2 stance width (lateral L/R ankle_roll separation in the pelvis frame).

FK the X2 MJCF on each frame of a motion-lib entry (dof in MuJoCo joint order,
root free-joint = root_trans_offset + root_rot xyzw). Report the median lateral
(pelvis-Y) separation between left_ankle_roll_link and right_ankle_roll_link over
the frames where the robot is actually walking (both feet roughly level / moving),
which is the stance metric the corpus cares about.

Usage:
    .venv/bin/python gear_sonic/scripts/measure_x2_stance_width.py --pkl X.pkl [--key K]
"""
import argparse
import joblib
import mujoco
import numpy as np

MJCF = "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
NDOF = 31


def measure(entry, model, data):
    rt = np.asarray(entry["root_trans_offset"], np.float64)
    rq = np.asarray(entry["root_rot"], np.float64)          # xyzw
    dof = np.asarray(entry["dof"], np.float64)[:, :NDOF]
    T = dof.shape[0]
    rq_wxyz = rq[:, [3, 0, 1, 2]]
    lb = model.body("left_ankle_roll_link").id
    rb = model.body("right_ankle_roll_link").id
    pb = model.body("pelvis").id
    sep = np.zeros(T)
    for t in range(T):
        data.qpos[0:3] = rt[t]
        data.qpos[3:7] = rq_wxyz[t]
        data.qpos[7:7 + NDOF] = dof[t]
        mujoco.mj_forward(model, data)
        p = data.xpos[pb]
        Rp = data.xmat[pb].reshape(3, 3)
        l_local = Rp.T @ (data.xpos[lb] - p)
        r_local = Rp.T @ (data.xpos[rb] - p)
        sep[t] = abs(l_local[1] - r_local[1])   # lateral (pelvis-Y)
    return sep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--key", default=None)
    args = ap.parse_args()
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    assert model.nq == 7 + NDOF, (model.nq, 7 + NDOF)
    motions = joblib.load(args.pkl)
    keys = [args.key] if args.key else list(motions)
    for k in keys:
        sep = measure(motions[k], model, data)
        print(f"{k}: frames={len(sep)}  stance median={np.median(sep)*100:.1f}cm  "
              f"mean={sep.mean()*100:.1f}cm  p25={np.percentile(sep,25)*100:.1f}cm  "
              f"p75={np.percentile(sep,75)*100:.1f}cm")


if __name__ == "__main__":
    main()
