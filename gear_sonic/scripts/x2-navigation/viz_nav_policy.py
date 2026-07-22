#!/usr/bin/env python3
"""Visualize the trained nav teacher: roll out all 56 waypoint routes and
paint the driven trajectories over the kitchen walkable map.

Outputs (default /tmp/claude-1000/):
  nav_routes_all.png        all 56 routes, color = time along path
  nav_routes_showcase.png   6 showcase routes with start/goal/heading detail
  nav_smoothness.png        action-delta + hold-length histograms (the
                            action-rate penalty's measurable effect)

Usage:
    python viz_nav_policy.py [--ckpt <model_*.pt>] [--out-dir DIR]
"""
import argparse
import glob
import itertools
import os

import numpy as np
import torch

from train_nav_teacher import KITCHEN, NavKitchenEnv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default=f"{KITCHEN}/runs/nav_teacher_hardened_0722c")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out-dir", default="/tmp/claude-1000")
    args = ap.parse_args()
    device = "cuda"
    ckpt = args.ckpt or sorted(
        glob.glob(f"{args.run_dir}/model_*.pt"),
        key=lambda p: int(p.split("_")[-1][:-3]))[-1]
    print("checkpoint:", ckpt)

    env = NavKitchenEnv(64, device, seed=7)
    from rsl_rl.runners import OnPolicyRunner
    from train_nav_teacher import main as _  # noqa: F401  (cfg parity)
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
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=device)

    # -- roll out all ordered waypoint routes ------------------------------
    pairs = list(itertools.permutations(range(len(env.wp_names)), 2))
    n = len(pairs)
    ev = NavKitchenEnv(n, device, seed=123)
    ev.waypoint_eval_mode = True
    a = torch.tensor([p[0] for p in pairs], device=device)
    b = torch.tensor([p[1] for p in pairs], device=device)
    ev.pos[:] = ev.wp_xy_snap[a]
    ev.yaw[:] = ev.wp_yaw[a]
    ev.goal[:] = ev.wp_xy[b]
    ev.goal_yaw[:] = ev.wp_yaw[b]
    ev.goal_has_yaw[:] = True
    ev.goal_rad[:] = ev.wp_rad[b]
    ev.vel_cmd[:] = 0; ev.vel_act[:] = 0; ev.prev_act[:] = 0
    ev.prev_dist = (ev.goal - ev.pos).norm(dim=1)
    ev.lag[:] = 0.25
    ev.episode_length_buf[:] = 0
    obs = ev._obs()
    traj = [ev.pos.cpu().numpy().copy()]
    acts, holds = [], []
    done_at = torch.full((n,), -1, device=device)
    with torch.no_grad():
        for t in range(ev.max_episode_length):
            act = policy(obs)
            obs, _, done, ex = ev.step(act)
            traj.append(ev.pos.cpu().numpy().copy())
            # record the ENVELOPED command (what the planner would receive),
            # not the raw unbounded network output
            acts.append(ev._envelope(act.clamp(-1, 1)).cpu().numpy().copy())
            newly = (done_at < 0) & done & ~ex["time_outs"]
            done_at[newly] = t
            if (done_at >= 0).all():
                break
    traj = np.asarray(traj)                        # (T+1, n, 2)
    acts = np.asarray(acts)                        # (T, n, 3)
    succ = (done_at >= 0).cpu().numpy()
    print(f"routes: {n} | succeeded: {succ.sum()} "
          f"| mean time: {float(done_at[done_at>=0].float().mean())*ev.DT:.1f}s")

    # -- figure 1: all routes ---------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    g = np.load(f"{KITCHEN}/assets/nav_grid.npz")
    walk, esdf, origin, res = g["walkable"], g["esdf"], g["origin"], float(g["res"])
    nx, ny = walk.shape
    ext = [origin[0], origin[0] + nx * res, origin[1], origin[1] + ny * res]
    bg = np.where(esdf.T < 0.03, 0.15, np.where(walk.T, 1.0, 0.72))

    def draw_map(ax):
        ax.imshow(bg, origin="lower", cmap="gray", extent=ext, zorder=0)
        for i, name in enumerate(env.wp_names):
            x, y = env.wp_xy[i].cpu().numpy()
            ax.plot(x, y, "r*", ms=13, zorder=5)
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(7, 6), color="darkred", fontsize=9, zorder=6)
        ax.set_aspect("equal")
        ax.set_xlim(-4.2, 2.6); ax.set_ylim(-5.5, 3.2)

    fig, ax = plt.subplots(figsize=(11, 11))
    draw_map(ax)
    cmap = plt.get_cmap("viridis")
    for i in range(n):
        end = int(done_at[i]) + 1 if succ[i] else traj.shape[0] - 1
        pts = traj[:end + 1, i]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc = LineCollection(segs, colors=cmap(np.linspace(0, 1, len(segs))),
                            lw=1.4, alpha=0.55, zorder=2)
        ax.add_collection(lc)
    ax.set_title(f"nav teacher {os.path.basename(ckpt)} — all {n} waypoint "
                 f"routes ({succ.sum()}/{n} reached; color = time along path)")
    fig.savefig(f"{args.out_dir}/nav_routes_all.png", dpi=110,
                bbox_inches="tight")

    # -- figure 2: showcase routes ----------------------------------------
    SHOW = [("entrance", "dining_table"), ("hallway", "fridge"),
            ("dining_table", "sink"), ("pantry", "cooking_range"),
            ("sink", "hallway"), ("cooking_range", "entrance")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    for ax, (sa, sb) in zip(axes.flat, SHOW):
        ia, ib = env.wp_names.index(sa), env.wp_names.index(sb)
        k = pairs.index((ia, ib))
        end = int(done_at[k]) + 1 if succ[k] else traj.shape[0] - 1
        pts = traj[:end + 1, k]
        draw_map(ax)
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(
            segs, colors=cmap(np.linspace(0, 1, len(segs))), lw=3, zorder=3))
        ax.plot(*pts[0], "go", ms=11, zorder=4)
        ax.plot(*pts[-1], "r^", ms=11, zorder=4)
        t = end * ev.DT
        ax.set_title(f"{sa} → {sb}   ({t:.1f}s)", fontsize=12)
    fig.suptitle("showcase routes (green = start, red = arrival)", fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/nav_routes_showcase.png", dpi=100,
                bbox_inches="tight")

    # -- figure 3: smoothness ---------------------------------------------
    d = np.linalg.norm(np.diff(acts, axis=0), axis=2).ravel()
    ch = (np.abs(np.diff(acts, axis=0)).max(axis=2) > 0.05)
    hold_lens = []
    for i in range(n):
        run = 0
        for t in range(ch.shape[0]):
            if ch[t, i]:
                if run:
                    hold_lens.append(run)
                run = 0
            else:
                run += 1
    hold_lens = np.asarray(hold_lens, dtype=np.float64) * ev.DT
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.4))
    a1.hist(d, bins=60, color="#4477aa")
    a1.set_title(f"per-tick COMMAND change  (mean {d.mean():.3f} m/s-equiv)")
    a1.set_xlabel("|Δaction| per 200 ms tick")
    if len(hold_lens):
        a2.hist(hold_lens, bins=40, color="#66aa77")
        a2.set_title(f"intent hold lengths  (mean {hold_lens.mean():.2f}s, "
                     f"p90 {np.percentile(hold_lens, 90):.2f}s)")
    else:
        a2.text(0.5, 0.5, "zero intent changes >0.05 across all routes\n"
                "(each route ridden on one held command)",
                ha="center", va="center", fontsize=12)
        a2.set_title("intent hold lengths")
    a2.set_xlabel("seconds held")
    fig.suptitle("what the action-rate penalty bought (56-route rollout)")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/nav_smoothness.png", dpi=110,
                bbox_inches="tight")
    print("wrote nav_routes_all.png / nav_routes_showcase.png / "
          "nav_smoothness.png ->", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
