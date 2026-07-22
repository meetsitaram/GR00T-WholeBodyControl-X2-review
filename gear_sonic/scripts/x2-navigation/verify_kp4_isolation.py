#!/usr/bin/env python3
"""Generate + verify the 4-command planner isolation test (RL prerequisite).

Runs the deployed planner ONNX four times (separate processes = OS-level
isolation) with forward / backward / turn-left / turn-right intents, then:
  1. numerically verifies each trajectory obeys its command,
  2. proves the four dof streams are unique (md5),
  3. merges them into a 4-motion pkl playable in the eval env (adds the
     pose_aa zeros the loader requires; use with
     ++manager_env.commands.motion.motion_lib_cfg.fine_tune_dataset.enable=false
     +num_envs=4 so envs map 1:1 to motions),
  4. renders a top-down trajectory figure.

First run (2026-07-22) verdict: all four obey + unique; right turns ran
~2.7x the commanded rate vs left ~1.07x — OPEN ISSUE before RL treats
+/-yaw as symmetric.

Usage:
    python verify_kp4_isolation.py \
        --planner-onnx ~/x2_cloud_checkpoints/planner_onnx_fixedscratch_p500k/x2_planner_template.onnx \
        --workdir /tmp/kp4 --out gear_sonic/data/motions/kp4_isolation_test.pkl
"""
import argparse
import hashlib
import math
import os
import subprocess
import sys

import joblib
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

SPECS = [  # name, vel_z (fwd m/s), yaw_rate (rad/s)
    ("straight", 0.3, 0.0),
    ("backward", -0.3, 0.0),
    ("turnleft", 0.0, 1.0),
    ("turnright", 0.0, -1.0),
]


def yaw_of(q):  # xyzw
    x, y, z, w = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner-onnx", required=True)
    ap.add_argument("--workdir", default="/tmp/kp4")
    ap.add_argument("--out", default="gear_sonic/data/motions/kp4_isolation_test.pkl")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--skip-gen", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    if not args.skip_gen:
        for name, vz, yr in SPECS:
            subprocess.run(
                [sys.executable,
                 os.path.join(REPO, "gear_sonic/scripts/gen_kplanner_clip.py"),
                 "--planner-onnx", os.path.expanduser(args.planner_onnx),
                 "--out", f"{args.workdir}/kp4_{name}.pkl",
                 "--seconds", str(args.seconds),
                 "--vel-z", str(vz), "--yaw-rate", str(yr),
                 "--name", f"kp4_{name}"],
                check=True)

    merged, sigs, rows = {}, {}, []
    for name, vz, yr in SPECS:
        m = next(iter(joblib.load(f"{args.workdir}/kp4_{name}.pkl").values()))
        tr = np.asarray(m["root_trans_offset"])
        q = np.asarray(m["root_rot"])
        y0 = yaw_of(q[0])
        dyaw, prev = 0.0, y0
        for i in range(1, len(q)):
            y = yaw_of(q[i])
            dyaw += math.atan2(math.sin(y - prev), math.cos(y - prev))
            prev = y
        fwd = np.array([math.cos(y0), math.sin(y0)])
        disp = tr[-1, :2] - tr[0, :2]
        along = float(disp @ fwd)
        sigs[name] = hashlib.md5(np.asarray(m["dof"]).tobytes()).hexdigest()[:10]
        rows.append((name, along, dyaw))
        print(f"{name:9s} along={along:+.2f}m net_yaw={dyaw:+.2f}rad "
              f"({dyaw/args.seconds:+.2f}rad/s) md5={sigs[name]}")
        n = np.asarray(m["dof"]).shape[0]
        m["pose_aa"] = np.zeros((n, 32, 3), dtype=np.float32)  # loader requires
        merged[f"kp4_{name}"] = m

    assert len(set(sigs.values())) == 4, "CONFLICT: identical dof streams!"
    assert rows[0][1] > 1.0 and rows[1][1] < -1.0, "fwd/back displacement wrong"
    assert rows[2][2] > 2.0 and rows[3][2] < -2.0, "turn directions wrong"
    joblib.dump(merged, args.out, compress=3)
    print(f"\nOK: 4 unique command-obeying streams -> {args.out}")

    ratio = abs(rows[3][2]) / max(abs(rows[2][2]), 1e-6)
    if not 0.7 < ratio < 1.4:
        print(f"WARNING: turn-rate asymmetry right/left = {ratio:.2f}x "
              f"(open issue, first seen 2026-07-22 at 2.55x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
