#!/usr/bin/env python3
"""Drive the IsaacLab kitchen rig with the trained nav policy.

Replaces pad_locomotion_bridge: subscribes to the rig's robot_pose feedback
(:5570, kitchen frame), builds the stage-0 observation (goal in body frame +
finite-difference velocity + ESDF clearance rays from nav_grid.npz), runs the
overnight checkpoint, and publishes stick commands on planner_cmd (:5563)
exactly like the pad does.

Usage (rig must be running; kill the pad bridge first — dueling commands):
    python nav_policy_bridge.py --route pantry entrance
    python nav_policy_bridge.py --goal sink            # single goal
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import zmq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))   # repo root
from train_nav_teacher import KITCHEN, NavKitchenEnv  # noqa: E402

CMD_HZ = 2.0          # decisions/s sent to the planner (rate-limited)
ARRIVE_SLACK = 0.10
RESYNC_EVERY = 8.0    # s of continuous motion before a micro-resync stop
RESYNC_PAUSE = 1.2    # s of commanded stop: planner freezes -> next command
                      # re-seeds its frame at the ROBOT's true pose, zeroing
                      # the open-loop (scope=none) reference drift that made
                      # markers walk ahead of the robot into walls


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default=f"{KITCHEN}/runs/nav_teacher_hardened_0722c")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--route", nargs="+", default=None,
                    help="waypoint names, visited in order")
    ap.add_argument("--goal", default=None)
    ap.add_argument("--pose-port", type=int, default=6570)
    ap.add_argument("--cmd-port", type=int, default=5563)
    args = ap.parse_args()
    route = args.route or ([args.goal] if args.goal else ["entrance"])
    device = "cuda"

    # policy (env instance only supplies shapes/grid/waypoints)
    env = NavKitchenEnv(1, device, seed=0)
    from rsl_rl.runners import OnPolicyRunner
    cfg = {
        "num_steps_per_env": 24, "save_interval": 500,
        "empirical_normalization": True, "logger": "tensorboard",
        "obs_groups": {"policy": ["critic"], "critic": ["critic"]},
        "policy": {"class_name": "ActorCritic", "activation": "elu",
                   "actor_hidden_dims": [256, 128, 64],
                   "critic_hidden_dims": [256, 128, 64],
                   "init_noise_std": 1.0},
        "algorithm": {"class_name": "PPO", "value_loss_coef": 1.0,
                      "use_clipped_value_loss": True, "clip_param": 0.2,
                      "entropy_coef": 0.005, "num_learning_epochs": 5,
                      "num_mini_batches": 4, "learning_rate": 1e-3,
                      "schedule": "adaptive", "gamma": 0.99, "lam": 0.95,
                      "desired_kl": 0.01, "max_grad_norm": 1.0},
    }
    runner = OnPolicyRunner(env, cfg, log_dir=None, device=device)
    ckpt = args.ckpt or sorted(
        glob.glob(f"{args.run_dir}/model_*.pt"),
        key=lambda p: int(p.split("_")[-1][:-3]))[-1]
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=device)
    print(f"[nav-bridge] policy {ckpt}")
    print(f"[nav-bridge] route: {' -> '.join(route)}")
    for r in route:
        assert r in env.wp_names, f"unknown waypoint {r!r}"

    # zmq wiring (pad-bridge parity)
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(f"tcp://127.0.0.1:{args.pose_port}")
    pub = ctx.socket(zmq.PUB)
    pub.connect(f"tcp://127.0.0.1:{args.cmd_port}")
    from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose

    def latest_pose():
        p = None
        while True:
            try:
                parts = sub.recv_multipart(flags=zmq.NOBLOCK)
                p = parts[-1]
            except zmq.Again:
                return p

    def send(fwd, side, yawr):
        pub.send_multipart([b"planner_cmd", json.dumps({
            "intent": "locomotion", "magnitude": "continuous",
            "stick_fwd": float(fwd), "stick_side": float(side),
            "stick_yaw": float(yawr)}).encode()])

    print("[nav-bridge] waiting for robot_pose ...", flush=True)
    raw = None
    while raw is None:
        raw = latest_pose()
        time.sleep(0.1)
    print("[nav-bridge] pose stream up", flush=True)

    leg = 0
    best_dist = None     # stuck detection: (best, t_of_best)
    t_moving = None      # continuous-motion timer for micro-resyncs
    resync_until = 0.0
    escape = None        # (phase_end_t, phase) when un-wedging
    escapes_this_leg = 0
    prev = None          # (t, x, y, yaw)
    vel_ema = np.zeros(3)
    prev_act = torch.zeros(1, 3, device=device)
    t_next = time.monotonic()
    t_leg = time.monotonic()
    while leg < len(route):
        time.sleep(max(0.0, t_next - time.monotonic()))
        t_next = time.monotonic() + 1.0 / CMD_HZ
        raw = latest_pose() or raw
        f = unpack_robot_pose(raw)
        px, py, pz, qw, qx_, qy_, qz_ = f["pelvis_qpos_wxyz"]
        yaw = math.atan2(2 * (qw * qz_ + qx_ * qy_),
                         1 - 2 * (qy_ ** 2 + qz_ ** 2))
        now = time.monotonic()
        if prev is not None and now > prev[0]:
            dt = now - prev[0]
            vw = np.array([(px - prev[1]) / dt, (py - prev[2]) / dt])
            cy, sy = math.cos(yaw), math.sin(yaw)
            vb = np.array([cy * vw[0] + sy * vw[1],
                           -sy * vw[0] + cy * vw[1],
                           math.atan2(math.sin(yaw - prev[3]),
                                      math.cos(yaw - prev[3])) / dt])
            vel_ema = 0.6 * vel_ema + 0.4 * vb
        prev = (now, px, py, yaw)

        wi = env.wp_names.index(route[leg])
        gxy = env.wp_xy[wi].cpu().numpy()
        gyaw = float(env.wp_yaw[wi])
        grad = float(env.wp_rad[wi])
        d = gxy - np.array([px, py])
        dist = float(np.linalg.norm(d))
        dyaw = math.atan2(math.sin(gyaw - yaw), math.cos(gyaw - yaw))
        if dist < grad + ARRIVE_SLACK and abs(dyaw) < 0.35:
            print(f"[nav-bridge] ARRIVED {route[leg]} "
                  f"({time.monotonic()-t_leg:.1f}s)", flush=True)
            send(0, 0, 0)
            leg += 1
            t_leg = time.monotonic()
            best_dist = None
            continue
        # stall handling: at 8 s of no progress run an ESCAPE primitive
        # (reverse ~2.5 s, then rotate toward open space ~1.5 s — the
        # un-wedge reflex the policy never learned because training
        # contacts were terminal). After 2 failed escapes, skip the leg.
        now2 = time.monotonic()
        if best_dist is None or dist < best_dist[0] - 0.10:
            best_dist = (dist, now2)
        elif escape is None and now2 - best_dist[1] > 8.0:
            if escapes_this_leg >= 2:
                print(f"[nav-bridge] STUCK at {route[leg]} leg "
                      f"(dist {dist:.2f}m, 2 escapes failed) — skipping",
                      flush=True)
                send(0, 0, 0)
                leg += 1
                t_leg = now2
                best_dist = None
                escapes_this_leg = 0
                continue
            escapes_this_leg += 1
            escape = (now2 + 2.5, "reverse")
            print(f"[nav-bridge] ESCAPE #{escapes_this_leg}: reversing "
                  f"(wedged at dist {dist:.2f}m)", flush=True)
        if escape is not None:
            phase_end, phase = escape
            if phase == "reverse":
                send(-0.8, 0, 0)
                if now2 >= phase_end:
                    escape = (now2 + 1.8, "rotate")
            else:
                # rotate toward the more open side (body-frame rays:
                # left half vs right half). stick_yaw>0 = turn RIGHT
                # per daemon chart, so open-left -> negative stick.
                rleft = float(rays[1:env.N_RAYS // 2].sum())
                rright = float(rays[env.N_RAYS // 2 + 1:].sum())
                send(0, 0, 0.7 if rright > rleft else -0.7)
                if now2 >= phase_end:
                    escape = None
                    best_dist = (dist, now2)   # fresh stall window
                    print("[nav-bridge] escape done, policy resumes",
                          flush=True)
            continue

        # micro-resync: pause briefly so the planner re-seeds at the
        # robot's true pose (kills accumulated reference drift)
        now3 = time.monotonic()
        if now3 < resync_until:
            send(0, 0, 0)
            continue
        if t_moving is None:
            t_moving = now3
        elif now3 - t_moving > RESYNC_EVERY:
            print(f"[nav-bridge] micro-resync ({RESYNC_PAUSE}s pause, "
                  f"planner re-seeds at robot pose)", flush=True)
            resync_until = now3 + RESYNC_PAUSE
            t_moving = None
            send(0, 0, 0)
            continue

        cy, sy = math.cos(yaw), math.sin(yaw)
        gb = np.array([cy * d[0] + sy * d[1], -sy * d[0] + cy * d[1]])
        pos_t = torch.tensor([[px, py]], dtype=torch.float32, device=device)
        wa = torch.tensor(yaw, device=device) + env.ray_ang
        rays = []
        for r in (0.4, 0.8):
            p = pos_t + r * torch.stack(
                (torch.cos(wa), torch.sin(wa)), dim=1)
            rays.append(env._esdf_at(p))
        rays = torch.minimum(rays[0], rays[1]).clamp(max=2.0) / 2.0
        obs = torch.cat([
            torch.tensor([[gb[0] / 5, gb[1] / 5, dist / 5,
                           math.cos(dyaw), math.sin(dyaw), 1.0]],
                         dtype=torch.float32, device=device),
            torch.tensor(vel_ema, dtype=torch.float32,
                         device=device).unsqueeze(0),
            prev_act,
            rays.unsqueeze(0)], dim=1)
        with torch.no_grad():
            act = policy(obs).clamp(-1, 1)
        cmd = env._envelope(act)[0].cpu().numpy()
        prev_act = act
        # sticks (daemon chart x2_kplanner.py:398): fwd>0 forward (same);
        # side>0 RIGHT and yaw>0 TURN-RIGHT — both INVERTED vs the policy's
        # body frame (+lat = left, +yaw = CCW). Negate on the wire.
        send(cmd[0] / env.V_MAX, -cmd[1] / env.V_MAX, -cmd[2] / env.V_MAX)
        print(f"[nav-bridge] {route[leg]}: dist={dist:.2f}m dyaw={dyaw:+.2f} "
              f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})", flush=True)

    print("[nav-bridge] route complete.", flush=True)
    send(0, 0, 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
