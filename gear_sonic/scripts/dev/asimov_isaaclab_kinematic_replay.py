#!/usr/bin/env python3
"""Parity gate B: kinematic replay of an Asimov motion pkl through IsaacLab.

Drives the actual IsaacLab articulation (same URDF/USD, same IL<->MJ maps as
training) with reference poses written directly to sim — no policy, no
physics forces (gravity off, kinematic writes). If the walk looks correct
here, the joint mappings and motion-lib conventions are visually PROVEN in
the training simulator; any weirdness in policy evals is then the policy,
not the plumbing.

    python gear_sonic/scripts/dev/asimov_isaaclab_kinematic_replay.py \
        --pkl gear_sonic/data/motions/asimov_walk1.pkl [--clip <key>] [--headless]
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pkl", required=True)
parser.add_argument("--clip", default=None)
parser.add_argument("--speed", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from gear_sonic.envs.manager_env.robots.asimov import (  # noqa: E402
    ASIMOV_CFG,
    ASIMOV_ISAACLAB_TO_MUJOCO_MAPPING as M,
)


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 60.0, gravity=(0.0, 0.0, 0.0)))
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/ground", cfg)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/light", light)
    robot = Articulation(ASIMOV_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    d = joblib.load(args.pkl)
    key = args.clip if args.clip in d else list(d)[0]
    e = d[key]
    dof_mj = torch.tensor(np.asarray(e["dof"]), dtype=torch.float32)      # (T,23) MJ order
    root_p = torch.tensor(np.asarray(e["root_trans_offset"]), dtype=torch.float32)
    rq = np.asarray(e["root_rot"])                                        # xyzw
    root_q = torch.tensor(rq[:, [3, 0, 1, 2]], dtype=torch.float32)      # wxyz
    fps = float(e["fps"])
    T = len(dof_mj)
    # MJ -> IL gather: dof_il = dof_mj[src], src = mapping["mujoco_to_isaaclab_dof"]
    src = torch.tensor(M["mujoco_to_isaaclab_dof"], dtype=torch.long)
    dof_il = dof_mj[:, src]
    print(f"[replay] {key}: {T} frames @{fps}fps — kinematic, gravity off")

    device = robot.device
    zeros6 = torch.zeros((1, 6), device=device)
    zeros_dof = torch.zeros((1, 23), device=device)
    t0 = time.time()
    while simulation_app.is_running():
        f = int(((time.time() - t0) * args.speed) * fps) % T
        root_state = torch.cat(
            [root_p[f][None].to(device), root_q[f][None].to(device), zeros6], dim=-1)
        robot.write_root_state_to_sim(root_state)
        robot.write_joint_state_to_sim(dof_il[f][None].to(device), zeros_dof)
        sim.render()

    simulation_app.close()


if __name__ == "__main__":
    main()
