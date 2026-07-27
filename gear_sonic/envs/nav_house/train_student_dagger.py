#!/usr/bin/env python3
"""Overnight DAgger: camera student distilled from the stage-0 nav teacher.

Success criterion (user, 2026-07-22): commanded-waypoint arrival accuracy
UNDER ODOMETRY DRIFT (the real-rig failure mode, worst for bipeds) — the
entrance is the acid test. Three metric lines in metrics.jsonl:
  teacher_clean  : teacher + true state        -> the supervision ceiling
  teacher_drift  : teacher + drifted odometry  -> today's real-world behavior
  student_drift  : camera student + drift      -> must climb toward the ceiling

Design: rollouts in the fast surrogate (train_nav_teacher.NavKitchenEnv),
student sees gallery-rendered sensor RGB (pose-indexed; baked via the G1
path) + goal/proprio with drift-DR injected; frozen teacher labels every
visited state with TRUE-state actions; visual DR per VR-Robo; rare push DR.
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
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
NAV = os.path.join(os.path.dirname(os.path.dirname(HERE)),
                   "scripts", "x2-navigation")
sys.path.insert(0, NAV)
from train_nav_teacher import KITCHEN, NavKitchenEnv  # noqa: E402

# NAV_V2=1: run every surrogate env in v2 realism (shoulder-rect
# footprint, deploy-quantized sticks) — REQUIRED when distilling from a
# v2 teacher (TEACHER_RUN pointing at nav_teacher_v2_*).
NAV_V2 = os.environ.get("NAV_V2", "0") == "1"

sys.path.insert(0, HERE)
from poc_student_bc import StudentNet  # noqa: E402

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class Gallery:
    """Pose-indexed sensor renders on GPU. lookup(xy, yaw) -> uint8 BCHW."""

    def __init__(self, npz_path, device):
        d = np.load(npz_path)
        self.imgs = torch.tensor(d["imgs"]).to(device).permute(0, 3, 1, 2)
        poses = torch.tensor(d["poses"]).to(device)
        self.step = float(d["step"])
        self.nyaw = int(d["nyaw"])
        gx = torch.round(poses[:, 0] / self.step).long()
        gy = torch.round(poses[:, 1] / self.step).long()
        yi = torch.round(poses[:, 2] / (2 * math.pi / self.nyaw)).long() \
            % self.nyaw
        self.x0, self.y0 = int(gx.min()), int(gy.min())
        W = int(gx.max()) - self.x0 + 1
        H = int(gy.max()) - self.y0 + 1
        self.index = torch.full((W, H, self.nyaw), -1,
                                dtype=torch.long, device=device)
        self.index[gx - self.x0, gy - self.y0, yi] = torch.arange(
            len(poses), device=device)
        print(f"[gallery] {len(poses)} views, grid {W}x{H}x{self.nyaw}, "
              f"{self.imgs.element_size()*self.imgs.nelement()/1e6:.0f} MB",
              flush=True)

    def lookup(self, xy, yaw):
        gx = (torch.round(xy[:, 0] / self.step).long() - self.x0).clamp(
            0, self.index.shape[0] - 1)
        gy = (torch.round(xy[:, 1] / self.step).long() - self.y0).clamp(
            0, self.index.shape[1] - 1)
        yi = torch.round(
            (yaw % (2 * math.pi)) / (2 * math.pi / self.nyaw)).long() \
            % self.nyaw
        rows = self.index[gx, gy, yi]
        rows = torch.where(rows >= 0, rows, torch.zeros_like(rows))
        return self.imgs[rows]


def augment(rgb_u8, train=True):
    x = rgb_u8.float() / 255.0
    if train:
        b = x.shape[0]
        gain = 1.0 + 0.25 * (torch.rand(b, 3, 1, 1, device=x.device) - 0.5)
        bias = 0.1 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5)
        x = (x * gain + bias).clamp(0, 1)
        x = x + 0.02 * torch.randn_like(x)
        sh = torch.randint(-4, 5, (2,))
        x = torch.roll(x, shifts=(int(sh[0]), int(sh[1])), dims=(-2, -1))
        x = x.clamp(0, 1)
    return (x - IMNET_MEAN.to(x.device)) / IMNET_STD.to(x.device)


def build_teacher(device):
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
    tenv = NavKitchenEnv(4, device, seed=2, v2=NAV_V2)
    runner = OnPolicyRunner(tenv, cfg, log_dir=None, device=device)
    run_dir = os.environ.get(
        "TEACHER_RUN", f"{KITCHEN}/runs/nav_teacher_hardened_0722c")
    ckpt = sorted(glob.glob(f"{run_dir}/model_*.pt"),
                  key=lambda p: int(p.split("_")[-1][:-3]))[-1]
    runner.load(ckpt)
    print(f"[dagger] teacher: {os.path.basename(ckpt)}", flush=True)
    return runner.get_inference_policy(device=device)


@torch.no_grad()
def eval_entrance_3starts(policy_fn, device, drift_std, seed=5, K=8):
    """mid/pantry/dining_table -> entrance, K drift directions each —
    the long-route drift case (dining_table was the observed failure)."""
    n = 3 * K
    ev = NavKitchenEnv(n, device, seed=seed, v2=NAV_V2)
    ev.waypoint_eval_mode = True
    ei = ev.wp_names.index("entrance")
    torch.manual_seed(seed)
    # starts from SNAPPED waypoint cells — raw waypoint coords can sit
    # inside the wall margin (raw pantry start collides at t=0!)
    starts = {"mid": torch.tensor([-0.9, -0.3], device=device),
              "pantry": ev.wp_xy_snap[ev.wp_names.index("pantry")],
              "dining_table": ev.wp_xy_snap[
                  ev.wp_names.index("dining_table")]}
    sxy = torch.stack(list(starts.values())).float().repeat_interleave(
        K, dim=0)
    ev.pos[:] = sxy
    ev.yaw[:] = 0.0
    ev.goal[:] = ev.wp_xy[ei]
    ev.goal_yaw[:] = ev.wp_yaw[ei]
    ev.goal_has_yaw[:] = True
    ev.goal_rad[:] = ev.wp_rad[ei]
    ev.vel_cmd[:] = 0
    ev.vel_act[:] = 0
    ev.prev_act[:] = 0
    ev.prev_dist = (ev.goal - ev.pos).norm(dim=1)
    ev.lag[:] = 0.25
    ev.episode_length_buf[:] = 0
    success = torch.zeros(n, dtype=torch.bool, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    ang = (2 * math.pi / K) * (torch.arange(n, device=device) % K)
    u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    acc = torch.zeros(n, 1, device=device)
    prev_pos = ev.pos.clone()
    prev_yaw = ev.yaw.clone()
    obs = ev._obs()
    for _ in range(int(ev.max_episode_length)):
        disp = ev.pos - prev_pos
        dyaw_s = torch.atan2(torch.sin(ev.yaw - prev_yaw),
                             torch.cos(ev.yaw - prev_yaw)).abs()[:, None]
        prev_pos = ev.pos.clone()
        prev_yaw = ev.yaw.clone()
        dmag = disp.norm(dim=1, keepdim=True)
        # EMPIRICAL turn-gated drift (see drift_distribution.png):
        # curved-motion accrual + in-place rotational slip (0.04 m/rad)
        gate = (dyaw_s / (0.3 * dmag + 1e-6)).clamp(max=1.0)
        acc = (acc + drift_std * gate * dmag + 0.04 * dyaw_s).clamp(max=1.6)
        drift = acc * u
        act = policy_fn(obs, ev.pos, ev.yaw, drift)
        obs, _, done, ex = ev.step(act)
        success |= active & done & ~ex["time_outs"] & (
            (ev.goal - ev.pos).norm(dim=1) < ev.goal_rad + 0.05)
        active &= ~done
        if not active.any():
            break
    out = {}
    for i, nm in enumerate(starts):
        m = (torch.arange(n, device=device) // K) == i
        out[nm] = round(float(success[m].float().mean()), 3)
    return out


@torch.no_grad()
def eval_pivot_turns_student(policy_fn, device, seed=9):
    """USER CRITERION: same position, goal heading rotated {L90, R90, B180}
    — must pass with in-place turns only (net displacement < 0.30 m).
    Belief is seeded with a 0.5 m bias (fan directions): the corrupted goal
    vector claims displacement, so a drift-gullible policy walks toward the
    phantom; pixels must hold it in place. 3 spawns x 3 rotations."""
    import math as m
    ev = NavKitchenEnv(9, device, seed=seed, v2=NAV_V2)
    ev.waypoint_eval_mode = True
    spawns = [torch.tensor([-0.9, -0.3], device=device),
              ev.wp_xy_snap[ev.wp_names.index("pantry")],
              ev.wp_xy_snap[ev.wp_names.index("hallway")]]
    sxy = torch.stack(spawns).float().repeat_interleave(3, dim=0)
    rots = torch.tensor([m.pi / 2, -m.pi / 2, m.pi] * 3, device=device)
    ev.pos[:] = sxy
    ev.yaw[:] = 0.0
    ev.goal[:] = sxy
    ev.goal_yaw[:] = rots
    ev.goal_has_yaw[:] = True
    ev.goal_rad[:] = 0.4
    ev.vel_cmd[:] = 0; ev.vel_act[:] = 0; ev.prev_act[:] = 0
    ev.prev_dist = (ev.goal - ev.pos).norm(dim=1)
    ev.lag[:] = 0.25
    ev.episode_length_buf[:] = 0
    if NAV_V2 and hasattr(ev, "_prev_he"):
        ev._prev_he = torch.atan2(torch.sin(ev.goal_yaw - ev.yaw),
                                  torch.cos(ev.goal_yaw - ev.yaw)).abs()
    n = 9
    ang = 2.399963 * torch.arange(n, device=device)
    u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    acc = torch.full((n, 1), 0.5, device=device)     # seeded belief bias
    prev_yaw = ev.yaw.clone()
    success = torch.zeros(n, dtype=torch.bool, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    max_disp = torch.zeros(n, device=device)
    obs = ev._obs()
    for _ in range(int(20.0 / ev.DT)):
        dyaw_s = torch.atan2(torch.sin(ev.yaw - prev_yaw),
                             torch.cos(ev.yaw - prev_yaw)).abs()[:, None]
        prev_yaw = ev.yaw.clone()
        acc = (acc + 0.04 * dyaw_s).clamp(max=1.6)   # rotational slip
        drift = acc * u
        act = policy_fn(obs, ev.pos, ev.yaw, drift)
        obs, _, done, ex = ev.step(act)
        max_disp = torch.maximum(max_disp, (ev.pos - sxy).norm(dim=1))
        success |= (active & done & ~ex["time_outs"]
                    & ((ev.goal - ev.pos).norm(dim=1) < ev.goal_rad + 0.05)
                    & (max_disp < 0.30))
        active &= ~done
        if not active.any():
            break
    out = {}
    for j, lbl in enumerate(["L90", "R90", "B180"]):
        msk = (torch.arange(n, device=device) % 3) == j
        out[lbl] = round(float(success[msk].float().mean()), 3)
    out["max_disp"] = round(float(max_disp.max()), 2)
    return out


@torch.no_grad()
def eval_routes(policy_fn, device, drift_std, seed=3, K=8):
    """Each waypoint x K deterministic drift directions (compass fan) from
    the mid-kitchen spawn, on the env's OWN eval template
    (waypoint_eval_mode=True: no auto-reset, no noise DR). Reports per-
    waypoint success RATE over directions. Drift = drift_std [m/m] x
    distance walked along the replica's fixed direction; corrupts only the
    policy's goal-vector input (odometry bias) — sim stays true."""
    ev = NavKitchenEnv(8 * K, device, seed=seed, v2=NAV_V2)
    ev.waypoint_eval_mode = True
    names = ev.wp_names
    n = 8 * K
    torch.manual_seed(seed)
    wp_idx = torch.arange(8, device=device).repeat_interleave(K)
    ev.pos[:] = torch.tensor([[-0.9, -0.3]], device=device).repeat(n, 1)
    ev.yaw[:] = 0.0
    ev.goal[:] = ev.wp_xy[wp_idx]
    ev.goal_yaw[:] = ev.wp_yaw[wp_idx]
    ev.goal_has_yaw[:] = True
    ev.goal_rad[:] = ev.wp_rad[wp_idx]
    ev.vel_cmd[:] = 0
    ev.vel_act[:] = 0
    ev.prev_act[:] = 0
    ev.prev_dist = (ev.goal - ev.pos).norm(dim=1)
    ev.lag[:] = 0.25
    ev.episode_length_buf[:] = 0
    success = torch.zeros(n, dtype=torch.bool, device=device)
    active = torch.ones(n, dtype=torch.bool, device=device)
    ang = (2 * math.pi / K) * (torch.arange(n, device=device) % K)
    u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    acc = torch.zeros(n, 1, device=device)
    prev_pos = ev.pos.clone()
    prev_yaw = ev.yaw.clone()
    obs = ev._obs()
    for _ in range(int(ev.max_episode_length)):
        disp = ev.pos - prev_pos
        dyaw_s = torch.atan2(torch.sin(ev.yaw - prev_yaw),
                             torch.cos(ev.yaw - prev_yaw)).abs()[:, None]
        prev_pos = ev.pos.clone()
        prev_yaw = ev.yaw.clone()
        dmag = disp.norm(dim=1, keepdim=True)
        # EMPIRICAL turn-gated drift (see drift_distribution.png):
        # curved-motion accrual + in-place rotational slip (0.04 m/rad)
        gate = (dyaw_s / (0.3 * dmag + 1e-6)).clamp(max=1.0)
        acc = (acc + drift_std * gate * dmag + 0.04 * dyaw_s).clamp(max=1.6)
        drift = acc * u
        act = policy_fn(obs, ev.pos, ev.yaw, drift)
        obs, _, done, ex = ev.step(act)
        success |= active & done & ~ex["time_outs"] & (
            (ev.goal - ev.pos).norm(dim=1) < ev.goal_rad + 0.05)
        active &= ~done
        if not active.any():
            break
    d_end = (ev.goal - ev.pos).norm(dim=1)
    out = {}
    for i, nm in enumerate(names):
        m = wp_idx == i
        out[nm] = {"rate": round(float(success[m].float().mean()), 3),
                   "mean_final_dist": round(float(d_end[m].mean()), 2)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery",
                    default="/tmp/claude-1000/kitchen_gallery.npz")
    ap.add_argument("--run-dir",
                    default=f"{KITCHEN}/runs/nav_student_cam_0722")
    ap.add_argument("--num-envs", type=int, default=512)
    ap.add_argument("--iters", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--buffer", type=int, default=40000)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--teacher-mix-iters", type=int, default=3000)
    ap.add_argument("--drift-std", type=float, default=0.184,
                    help="odometry drift RATE (m bias per m walked); bipedal-pessimistic 15%/m + 3deg/m heading")
    ap.add_argument("--resume", default=None,
                    help="init student from this checkpoint (drift-fix "
                         "continuation training)")
    ap.add_argument("--drift-cap", type=float, default=0.6,
                    help="drift bias saturation (m) at eval; training "
                         "randomizes around it")
    ap.add_argument("--smoke", action="store_true",
                    help="300 iters + 1 eval, then exit")
    args = ap.parse_args()
    if args.smoke:
        args.iters = 300
        args.eval_every = 250
        args.teacher_mix_iters = 200
    device = "cuda"
    os.makedirs(args.run_dir, exist_ok=True)
    metrics_path = os.path.join(args.run_dir, "metrics.jsonl")

    gallery = Gallery(args.gallery, device)
    teacher = build_teacher(device)
    student = StudentNet(state_dim=12).to(device)
    if args.resume:
        student.load_state_dict(torch.load(args.resume,
                                           map_location=device))
        print(f"[dagger] resumed from {args.resume}", flush=True)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr,
                            weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.iters, eta_min=args.lr * 0.1)

    def teacher_clean_fn(obs, xy, yaw, drift):
        with torch.no_grad():
            return teacher(obs).clamp(-1, 1)

    def teacher_drift_fn(obs, xy, yaw, drift):
        o = obs.clone()
        o[:, 0:2] = o[:, 0:2] + drift / 5.0
        with torch.no_grad():
            return teacher(o).clamp(-1, 1)

    def student_drift_fn(obs, xy, yaw, drift):
        s = obs[:, :12].clone()
        s[:, 0:2] = s[:, 0:2] + drift / 5.0
        rgb = gallery.lookup(xy, yaw)
        student.eval()
        with torch.no_grad():
            a = student(augment(rgb, train=False), s)
        student.train()
        return a

    def log_eval(tag, it, res):
        rate = sum(r["rate"] for r in res.values()) / len(res)
        rec = {"iter": it, "tag": tag, "wall": time.time(),
               "mean_rate": round(rate, 3),
               "entrance": res.get("entrance"), "per_wp": res}
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"[eval] it={it} {tag}: mean={rate:.2f} "
              f"entrance={res.get('entrance')}", flush=True)
        return rate

    # baselines (the user's question in numbers)
    log_eval("teacher_clean", -1,
             eval_routes(teacher_clean_fn, device, drift_std=0.0))
    log_eval("teacher_drift", -1,
             eval_routes(teacher_drift_fn, device, args.drift_std))

    env = NavKitchenEnv(args.num_envs, device, seed=1, v2=NAV_V2)
    obs = env._obs()
    ang = 2 * math.pi * torch.rand(args.num_envs, device=device)
    u = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1)
    walked = torch.zeros(args.num_envs, 1, device=device)
    # EMPIRICAL (12 ref/exec pairs, 2026-07-23): rate median
    # 0.184/m p90 0.29 max 0.294; err max 1.56m, no saturation
    # within kitchen routes -> DR covers [0, 0.35] rate, soft caps
    rate = 0.50 * torch.rand(args.num_envs, 1, device=device)
    cap = 1.0 + 1.5 * torch.rand(args.num_envs, 1, device=device)
    prev_pos = env.pos.clone()

    N = args.buffer
    hw = gallery.imgs.shape[-1]
    buf_img = torch.zeros(N, 3, hw, hw, dtype=torch.uint8, device=device)
    buf_state = torch.zeros(N, 12, device=device)
    buf_act = torch.zeros(N, 3, device=device)
    buf_n, buf_p = 0, 0
    loss = torch.tensor(0.0)
    best = -1
    t0 = time.time()

    prev_yaw = env.yaw.clone()
    acc = torch.zeros(args.num_envs, 1, device=device)
    rot_rate = 0.12 * torch.rand(args.num_envs, 1, device=device)
    for it in range(args.iters + 1):
        disp = env.pos - prev_pos
        dyaw_s = torch.atan2(torch.sin(env.yaw - prev_yaw),
                             torch.cos(env.yaw - prev_yaw)).abs()[:, None]
        prev_pos = env.pos.clone()
        prev_yaw = env.yaw.clone()
        dmag = disp.norm(dim=1, keepdim=True)
        walked += dmag
        # fresh episodes: reset drift state + resample DR params
        fresh = env.episode_length_buf <= 1
        if bool(fresh.any()):
            k = int(fresh.sum())
            walked[fresh] = 0
            acc[fresh] = 0
            na = 2 * math.pi * torch.rand(k, device=device)
            u[fresh] = torch.stack([torch.cos(na), torch.sin(na)], dim=1)
            rate[fresh] = 0.50 * torch.rand(k, 1, device=device)
            rot_rate[fresh] = 0.12 * torch.rand(k, 1, device=device)
            cap[fresh] = 1.0 + 1.5 * torch.rand(k, 1, device=device)
            prev_pos[fresh] = env.pos[fresh]
        # EMPIRICAL turn-gated drift DR (2026-07-23 measurement): drift
        # accrues during CURVED motion (~rate m/m while turning, ~0 on
        # straights) + in-place rotational slip (rot_rate m/rad)
        gate = (dyaw_s / (0.3 * dmag + 1e-6)).clamp(max=1.0)
        acc = torch.minimum(acc + rate * gate * dmag + rot_rate * dyaw_s,
                            cap)
        drift = acc * u
        rgb = gallery.lookup(env.pos, env.yaw)
        s = obs[:, :12].clone()
        s[:, 0:2] = s[:, 0:2] + drift / 5.0
        with torch.no_grad():
            teacher_a = teacher(obs).clamp(-1, 1)
            mix = max(0.0, 1.0 - it / max(1, args.teacher_mix_iters))
            use_t = torch.rand(args.num_envs, device=device) < mix
            student.eval()
            student_a = student(augment(rgb, train=False), s)
            student.train()
            act = torch.where(use_t[:, None], teacher_a, student_a)
        kick = torch.rand(args.num_envs, device=device) < 0.002
        if bool(kick.any()):
            k = int(kick.sum())
            env.pos[kick] += 0.4 * torch.randn(k, 2, device=device)
            env.yaw[kick] += 0.8 * torch.randn(k, device=device)
        bidx = (buf_p + torch.arange(args.num_envs, device=device)) % N
        buf_img[bidx] = rgb
        buf_state[bidx] = s
        buf_act[bidx] = teacher_a
        buf_p = int((buf_p + args.num_envs) % N)
        buf_n = min(N, buf_n + args.num_envs)
        ret = env.step(act)
        obs = ret[0] if isinstance(ret, tuple) else env._obs()
        if isinstance(obs, dict):
            obs = obs.get("policy", obs.get("critic"))
        if buf_n >= args.batch:
            idx = torch.randint(0, buf_n, (args.batch,), device=device)
            pred = student(augment(buf_img[idx]), buf_state[idx])
            loss = F.mse_loss(pred, buf_act[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            sched.step()
        if it % 100 == 0:
            sps = (it + 1) * args.num_envs / (time.time() - t0)
            print(f"[dagger] it={it} loss={float(loss):.4f} mix={mix:.2f} "
                  f"buf={buf_n} {sps:.0f} samp/s", flush=True)
        if it and it % args.eval_every == 0:
            succ = log_eval("student_drift", it,
                            eval_routes(student_drift_fn, device,
                                        args.drift_std))
            e3 = eval_entrance_3starts(student_drift_fn, device,
                                       0.29)   # empirical p90
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps({"iter": it, "tag": "entrance_3starts",
                                     "wall": time.time(), **e3}) + "\n")
            print(f"[eval] it={it} entrance_3starts: {e3}", flush=True)
            pv = eval_pivot_turns_student(student_drift_fn, device)
            with open(metrics_path, "a") as fh:
                fh.write(json.dumps({"iter": it, "tag": "pivot_turns",
                                     "wall": time.time(), **pv}) + "\n")
            print(f"[eval] it={it} pivot_turns: {pv}", flush=True)
            torch.save(student.state_dict(),
                       f"{args.run_dir}/student_{it}.pt")
            if succ >= best:
                best = succ
                torch.save(student.state_dict(),
                           f"{args.run_dir}/student_best.pt")
    print(f"[dagger] done. best={best}/8", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
