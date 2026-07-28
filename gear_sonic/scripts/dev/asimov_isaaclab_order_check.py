#!/usr/bin/env python3
"""Parity gate C1: verify Asimov's PREDICTED IsaacLab joint/body order at runtime.

conventions.md: IsaacLab = BFS + alphabetical sibling sort, and hand-derived
orders are the #1 silent porting bug ("the robot simply moves wrong"). The
lists in robots/asimov.py were predicted from the URDF tree; this script
spawns the actual articulation headless and compares. Run in env_isaaclab:

    python gear_sonic/scripts/dev/asimov_isaaclab_order_check.py

Exit 0 = orders match (mapping arrays are trustworthy). On mismatch it prints
the runtime order to paste into robots/asimov.py.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from gear_sonic.envs.manager_env.robots.asimov import (  # noqa: E402
    ASIMOV_CFG,
    ASIMOV_ISAACLAB_BODIES,
    ASIMOV_ISAACLAB_JOINTS_ORDER,
)


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=0.005))
    cfg = ASIMOV_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(cfg)
    sim.reset()

    jn = list(robot.joint_names)
    bn = list(robot.body_names)
    ok = True
    if jn != ASIMOV_ISAACLAB_JOINTS_ORDER:
        ok = False
        print("JOINT ORDER MISMATCH — runtime order is:")
        print("ASIMOV_ISAACLAB_JOINTS_ORDER = [")
        for n in jn:
            print(f'    "{n}",')
        print("]")
    else:
        print(f"joint order OK ({len(jn)} joints)")
    if bn != ASIMOV_ISAACLAB_BODIES:
        ok = False
        print("BODY ORDER MISMATCH — runtime order is:")
        print("ASIMOV_ISAACLAB_BODIES = [")
        for n in bn:
            print(f'    "{n}",')
        print("]")
    else:
        print(f"body order OK ({len(bn)} bodies)")
    print("GATE C1:", "PASS" if ok else "FAIL")
    simulation_app.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
