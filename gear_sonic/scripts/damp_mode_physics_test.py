#!/usr/bin/env python3
"""Physics test for a proposed DAMP state: kp=0, kd>0, held by a soft spring.

WHY
---
On stop, nothing in our stack damps the motors. EM's MC app starts in ZERO
TORQUE, so the robot drops and the operator has to catch it. The proposed fix
is a DAMP state in the deploy: zero position stiffness (so the robot is free to
be moved by hand) with velocity damping retained (so it SINKS slowly instead of
collapsing), letting the operator place it on the floor before MC starts.

This script answers the physics question BEFORE any C++ is written:
  * does kp=0 / kd>0 actually produce a slow, controllable descent?
  * what kd gives a few-seconds sink rather than a collapse or a rigid lock?

The overhead spring mimics a hand (or gantry strap) taking some weight: a soft
upward pull toward a rest height, so the robot cannot trip or slam the floor
and we can watch the descent characteristics in isolation. It is applied as an
external force at the pelvis (``xfrc_applied``) rather than an MJCF edit, so no
model file is touched.

NOTE: no policy, no ONNX, no deploy -- this is pure actuator physics. It tells
us whether the CONCEPT holds and roughly what kd to pick. It does not model the
real robot's contacts or friction, so treat the kd as a starting point to tune
on hardware, not a final value.

    python gear_sonic/scripts/damp_mode_physics_test.py                  # sweep
    python gear_sonic/scripts/damp_mode_physics_test.py --viewer --kd-scale 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer  # module-level: a function-local import would shadow `mujoco`
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_x2_mujoco import (  # noqa: E402
    DEFAULT_DOF, JOINT_TO_ACTUATOR, KD, MJCF_PATH, NUM_DOFS, SIM_DT)


def run(kd_scale: float, spring_k: float, spring_damp: float, seconds: float,
        support_frac: float, viewer: bool = False, kd_abs: float | None = None) -> dict:
    model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = SIM_DT
    pelvis = model.body("pelvis").id

    # Start standing at the trained default pose.
    data.qpos[:] = 0.0
    data.qpos[2] = 0.72
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7 : 7 + NUM_DOFS] = DEFAULT_DOF
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    rest_z = float(data.qpos[2])
    weight = float(mujoco.mj_getTotalmass(model)) * 9.81
    # kd_abs: one flat damping value on every joint (N*m*s/rad), rather than
    # scaling the per-joint trained KD. Simpler to reason about and to state
    # in the deploy config.
    kd = np.full(NUM_DOFS, float(kd_abs)) if kd_abs is not None else KD * kd_scale

    n = int(seconds / SIM_DT)
    zs, ts = [], []
    v = None
    if viewer:

        v = mujoco.viewer.launch_passive(model, data)

    for i in range(n):
        # --- DAMP command: kp = 0, so torque is pure velocity damping -------
        # Joint order != actuator order. ctrl MUST be written through
        # JOINT_TO_ACTUATOR, exactly as eval_x2_mujoco.py does -- writing
        # ctrl[:] directly scrambles torques across joints and the sim
        # explodes (robot "rises" metres, |v| in the thousands).
        qvel_j = data.qvel[6 : 6 + NUM_DOFS]
        torque = -kd * qvel_j
        for j in range(NUM_DOFS):
            a = JOINT_TO_ACTUATOR[j]
            lo, hi = model.actuator_ctrlrange[a]
            data.ctrl[a] = float(np.clip(torque[j], lo, hi))

        # --- overhead spring: a hand/strap taking `support_frac` of weight --
        # Pulls UP only (a strap cannot push), toward rest_z, plus damping so
        # it does not oscillate. Deliberately soft -- it prevents a trip/slam,
        # it does not hold the robot up.
        z = float(data.qpos[2])
        vz = float(data.qvel[2])
        f = spring_k * max(0.0, rest_z - z) - spring_damp * vz
        f += support_frac * weight          # static partial unweighting
        data.xfrc_applied[pelvis, 2] = max(0.0, f)

        mujoco.mj_step(model, data)
        if i % 10 == 0:
            ts.append(i * SIM_DT); zs.append(float(data.qpos[2]))
        if v is not None:
            v.sync()
    if v is not None:
        v.close()

    zs = np.array(zs); ts = np.array(ts)
    drop = rest_z - zs[-1]
    # time to sink 10 cm -- a proxy for "can a human react and place it down"
    idx = np.where(rest_z - zs >= 0.10)[0]
    t10 = float(ts[idx[0]]) if len(idx) else float("inf")
    return {"kd_scale": kd_scale, "rest_z": rest_z, "final_z": float(zs[-1]),
            "drop": float(drop), "t_10cm": t10,
            "max_speed": float(np.abs(np.diff(zs) / np.diff(ts)).max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kd-scale", type=float, default=None,
                    help="single kd multiplier (default: sweep)")
    ap.add_argument("--spring-k", type=float, default=800.0,
                    help="overhead spring stiffness N/m (soft by design)")
    ap.add_argument("--spring-damp", type=float, default=60.0)
    ap.add_argument("--support-frac", type=float, default=0.35,
                    help="fraction of body weight the 'hand' carries")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--kd-abs", type=float, default=None,
                    help="flat damping on every joint, N*m*s/rad")
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    if args.kd_abs is not None:
        scales = [args.kd_abs]
    else:
        scales = ([args.kd_scale] if args.kd_scale is not None
                  else [0.0, 0.5, 1.0, 2.0, 4.0, 8.0])
    print(f"  spring: k={args.spring_k} N/m  damp={args.spring_damp}  "
          f"support={args.support_frac:.0%} of body weight")
    print(f"  kp = 0 for ALL joints (this is the DAMP proposal)\n")
    print(f"  {'kd x':>6}{'drop (m)':>11}{'final z':>10}{'t to -10cm':>13}"
          f"{'peak sink':>12}")
    print("  " + "-" * 52)
    for s in scales:
        r = run(s, args.spring_k, args.spring_damp, args.seconds,
                args.support_frac, args.viewer,
                kd_abs=(s if args.kd_abs is not None else None))
        t10 = "never" if r["t_10cm"] == float("inf") else f"{r['t_10cm']:.2f}s"
        print(f"  {s:>6.1f}{r['drop']:>11.3f}{r['final_z']:>10.3f}"
              f"{t10:>13}{r['max_speed']:>11.2f} m/s")
    print(f"\n  (rest z = {r['rest_z']:.3f} m; kd x0 = pure zero-torque, "
          f"the collapse we get today)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
