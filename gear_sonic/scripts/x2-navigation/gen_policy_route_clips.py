#!/usr/bin/env python3
"""Generate per-goal route clips: the trained nav policy driving the REAL
planner ONNX closed-loop (policy reads the planner's integrated pose each
tick — a stage-1 preview), one clip per destination, all starting from the
IDENTICAL pose. For the 6-robot showcase video.

Usage:
    python gen_policy_route_clips.py \
        --goals pantry entrance fridge sink dishwasher cooking_range \
        --out gear_sonic/data/motions/kp6_showcase.pkl
"""
import argparse
import glob
import importlib.util
import math
import os
import sys

import joblib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
from train_nav_teacher import KITCHEN, NavKitchenEnv  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
OUTPUT_FPS = 50.0
TICK = 0.2                      # policy decision period (s)


def yaw_of_wxyz(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goals", nargs="+", default=None)
    ap.add_argument("--routes", nargs="+", default=None,
                    help="start:goal waypoint pairs (spawn at start's "
                         "snapped cell); overrides --goals")
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir",
                    default=f"{KITCHEN}/runs/nav_teacher_hardened_0722c")
    ap.add_argument("--planner-onnx", type=__import__("pathlib").Path, default=__import__("pathlib").Path(os.path.expanduser(
        "~/x2_cloud_checkpoints/planner_onnx_fixedscratch_p500k/x2_planner_template.onnx")))
    ap.add_argument("--mode", default="slow_walk")
    ap.add_argument("--max-s", type=float, default=30.0)
    ap.add_argument("--spawn", default="-0.9,-0.3",
                    help="kitchen-frame spawn xy (mid-kitchen open floor)")
    ap.add_argument("--goal-fan", type=float, default=0.0,
                    help="fan arrival points around the shared goal: route i "
                         "offset i-alternating multiples of this many meters "
                         "perpendicular to the goal heading (robots gathering "
                         "at a door instead of a terminal pileup)")
    ap.add_argument("--stagger-s", type=float, default=3.0,
                    help="dispatch stagger between robots (s)")
    ap.add_argument("--lead-in-s", type=float, default=0.0,
                    help="global stand-still pad before anyone departs (s)")
    ap.add_argument("--depart-stagger", action="store_true",
                    help="stagger DEPARTURES i*stagger_s apart (peel-off "
                         "look for shared-spawn different-goal scenes) "
                         "instead of arrival-time choreography")
    ap.add_argument("--student-ckpt", default=None,
                    help="drive with the CAMERA STUDENT (StudentNet .pt) "
                         "instead of the teacher: gallery-rendered RGB + "
                         "drift-corrupted belief (its training condition)")
    ap.add_argument("--gallery",
                    default="/tmp/claude-1000/kitchen_gallery.npz")
    ap.add_argument("--drift-rate", type=float, default=0.15,
                    help="odometry drift m per m walked (student mode)")
    ap.add_argument("--approach-dist", type=float, default=0.0,
                    help="two-stage arrival: first reach a staging point "
                         "this many meters IN FRONT of the goal along its "
                         "heading (e.g. in the hallway before the entrance "
                         "door), then drive the final straight segment — "
                         "produces the natural approach-corridor path")
    ap.add_argument("--precision-goals", default="entrance",
                    help="comma list of goals that get recenter+staging "
                         "(doorway-type). Appliance goals keep their stored "
                         "coords — low clearance there is intentional.")
    ap.add_argument("--recenter-goals", action="store_true",
                    help="snap each goal to the local clearance maximum "
                         "within +-0.8m perpendicular to its heading — "
                         "corrects waypoint coords captured from the "
                         "planner's drifted pose (e.g. entrance sits right "
                         "of the actual doorway)")
    args = ap.parse_args()
    device = "cuda"

    # policy
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
    ckpt = sorted(glob.glob(f"{args.run_dir}/model_*.pt"),
                  key=lambda p: int(p.split("_")[-1][:-3]))[-1]
    runner.load(ckpt)
    policy = runner.get_inference_policy(device=device)
    print(f"[gen] policy {os.path.basename(ckpt)}")

    student = gallery = None
    if args.student_ckpt:
        nav_house = os.path.join(REPO, "gear_sonic", "envs", "nav_house")
        sys.path.insert(0, nav_house)
        from poc_student_bc import StudentNet
        from train_student_dagger import Gallery, augment
        student = StudentNet(state_dim=12).to(device)
        student.load_state_dict(torch.load(args.student_ckpt,
                                           map_location=device))
        student.eval()
        gallery = Gallery(args.gallery, device)
        globals()["_augment"] = augment
        print(f"[gen] STUDENT mode: {os.path.basename(args.student_ckpt)} "
              f"drift={args.drift_rate}/m", flush=True)

    # planner backend (the deployed ONNX runtime, as gen_kplanner_clip does)
    spec = importlib.util.spec_from_file_location(
        "pc2_kplanner_onnx",
        os.path.join(REPO, "gear_sonic", "scripts", "pc2_kplanner_onnx.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pc2_kplanner_onnx"] = mod
    spec.loader.exec_module(mod)
    contract = mod._load_onnx_contract(args.planner_onnx, None)

    base_spawn = np.array([float(v) for v in args.spawn.split(",")])
    per_tick = int(round(OUTPUT_FPS * TICK))
    merged = {}
    origins = {}
    used_spawns = []
    routes = ([tuple(r.split(":")) for r in args.routes] if args.routes
              else [(None, g) for g in args.goals])
    for ri, (start_name, goal_name) in enumerate(routes):
        if start_name and "," in start_name:
            # literal spawn coords "x,y" — for open-floor spots the 2D
            # grid snap can't be trusted to find (it is blind to 3D mesh
            # fuzz that ejects robots at spawn)
            spawn = np.array([float(v) for v in start_name.split(",")])
            used_spawns.append(spawn.copy())
            start_name = f"p{ri}"
        elif start_name:
            si = env.wp_names.index(start_name)
            # spawn-safe snap: >=0.6m clearance so the robot never spawns
            # intersecting the fuzzy splat collision mesh (spawn-in-mesh
            # ejection piled the counter-station robots onto the counter)
            import torch as _t
            wp = env.wp_xy[si]
            cost = ((env.free_xy - wp).norm(dim=1)
                    + 10.0 * (env._esdf_at(env.free_xy) < 0.6).float())
            # mutual separation: spawns >=0.9m apart (counter stations'
            # safe cells otherwise cluster in the aisle -> spawn tangles)
            for us in used_spawns:
                us_t = _t.tensor(us, dtype=_t.float32,
                                 device=env.free_xy.device)
                cost = cost + 10.0 * ((env.free_xy - us_t).norm(dim=1)
                                      < 0.9).float()
            spawn = env.free_xy[cost.argmin()].cpu().numpy()
            used_spawns.append(spawn.copy())
        else:
            spawn = base_spawn
        wi = env.wp_names.index(goal_name)
        gxy = env.wp_xy[wi].cpu().numpy().copy()
        gyaw = float(env.wp_yaw[wi])
        grad = float(env.wp_rad[wi])
        precision = goal_name in args.precision_goals.split(",")
        if args.recenter_goals and precision:
            perp = np.array([-math.sin(gyaw), math.cos(gyaw)])
            best = (None, 0.0)
            for k in range(-16, 17):
                cand = gxy + (0.05 * k) * perp
                ct = torch.tensor(cand[None], dtype=torch.float32,
                                  device=device)
                e = float(env._esdf_at(ct)[0])
                if best[0] is None or e > best[0]:
                    best = (e, 0.05 * k, cand)
            if abs(best[1]) > 0.01:
                print(f"[gen] recenter {goal_name}: {best[1]:+.2f}m perp "
                      f"(esdf {best[0]:.2f})", flush=True)
                gxy = best[2]
        if args.goal_fan > 0.0 and ri > 0:
            # arrival queue: the goal is a doorway (esdf ~0.18 at the
            # waypoint — no lateral room), so later robots stop at
            # increasing depths BEFORE it along the approach direction,
            # like a queue forming at a door. Robot 0 takes the door.
            fwd = np.array([math.cos(gyaw), math.sin(gyaw)])
            depth = ri * args.goal_fan
            best, best_e = None, -1.0
            for mag in (depth, depth + 0.3, depth - 0.3):
                cand = gxy - mag * fwd
                ct = torch.tensor(cand[None], dtype=torch.float32,
                                  device=device)
                e = float(env._esdf_at(ct)[0])
                if e > best_e:
                    best, best_e = cand, e
                if e >= 0.4:
                    break
            print(f"[gen] queue ri={ri} depth={depth:.1f} "
                  f"goal=({best[0]:+.2f},{best[1]:+.2f}) esdf={best_e:.2f}",
                  flush=True)
            gxy = best

        backend = mod.OnnxPlannerBackend(args.planner_onnx, contract, 16,
                                         args.mode)
        warm = mod._build_training_default_qpos()
        backend.reset(warm)
        # planner frame starts at origin, identity-ish yaw; our goal is
        # expressed relative to the spawn in the kitchen frame
        goal_p = gxy - spawn
        gyaw_p = gyaw            # spawn yaw ~0 in planner frame

        frames = []
        prev_root = None
        vel_ema = np.zeros(3)
        prev_act = torch.zeros(1, 3, device=device)
        walked = 0.0
        last_xy = None
        # two-stage arrival: staging point in front of the goal heading
        stage_p = None
        if args.approach_dist > 0.0 and precision:
            fwd_g = np.array([math.cos(gyaw), math.sin(gyaw)])
            # staging point is BEFORE the goal along the approach: the
            # robot passes it travelling in +gyaw direction into the goal
            cand = gxy - args.approach_dist * fwd_g
            ct = torch.tensor(cand[None], dtype=torch.float32,
                              device=device)
            if float(env._esdf_at(ct)[0]) >= 0.25:
                stage_p = cand - spawn      # planner frame
                print(f"[gen] staging point for {goal_name}: "
                      f"({cand[0]:+.2f},{cand[1]:+.2f})", flush=True)
        drift_ang = 2.399963 * ri          # golden-angle spread per route
        drift_u = np.array([math.cos(drift_ang), math.sin(drift_ang)])
        intent = (0.0, 0.0, 0.0, 0.687)
        backend.replan(intent)
        reached = False
        n_ticks = int(args.max_s / TICK)
        for t in range(n_ticks):
            # advance planner one decision period
            for _ in range(per_tick):
                frames.append(backend.get_next_frame_resampled(OUTPUT_FPS))
                if backend.should_replan():
                    backend.replan(intent)
            f = frames[-1]
            px, py = float(f[0]), float(f[1])
            yaw = yaw_of_wxyz(f[3:7])
            if prev_root is not None:
                vw = (np.array([px, py]) - prev_root[:2]) / TICK
                cy, sy = math.cos(yaw), math.sin(yaw)
                vb = np.array([cy * vw[0] + sy * vw[1],
                               -sy * vw[0] + cy * vw[1],
                               math.atan2(math.sin(yaw - prev_root[2]),
                                          math.cos(yaw - prev_root[2])) / TICK])
                vel_ema = 0.6 * vel_ema + 0.4 * vb
            prev_root = np.array([px, py, yaw])

            staging = stage_p is not None
            tgt_p = stage_p if staging else goal_p
            d = tgt_p - np.array([px, py])
            dist = float(np.linalg.norm(d))
            dyaw = math.atan2(math.sin(gyaw_p - yaw), math.cos(gyaw_p - yaw))
            if staging:
                if dist < 0.30:
                    stage_p = None          # staging reached -> final leg
                    continue
            # precision arrival: tight position gate, then an in-place
            # ALIGN phase to the waypoint heading (planner turns natively)
            pos_tol = min(grad, 0.30)
            if not staging and dist < pos_tol:
                if abs(dyaw) < 0.15:
                    reached = True
                    break
                intent = (float(np.clip(2.0 * dyaw, -0.8, 0.8)), 0.0, 0.0,
                          0.687)
                backend.replan(intent)
                continue
            cy, sy = math.cos(yaw), math.sin(yaw)
            gb = np.array([cy * d[0] + sy * d[1], -sy * d[0] + cy * d[1]])
            # rays from the kitchen ESDF at the KITCHEN-frame position
            kpos = torch.tensor([[px + spawn[0], py + spawn[1]]],
                                dtype=torch.float32, device=device)
            wa = torch.tensor(yaw, device=device) + env.ray_ang
            rr = []
            for r in (0.4, 0.8):
                p = kpos + r * torch.stack(
                    (torch.cos(wa), torch.sin(wa)), dim=1)
                rr.append(env._esdf_at(p))
            rays = torch.minimum(rr[0], rr[1]).clamp(max=2.0) / 2.0
            obs = torch.cat([
                torch.tensor([[gb[0] / 5, gb[1] / 5, dist / 5,
                               math.cos(dyaw), math.sin(dyaw),
                               0.0 if staging else 1.0]],
                             dtype=torch.float32, device=device),
                torch.tensor(vel_ema, dtype=torch.float32,
                             device=device).unsqueeze(0),
                prev_act, rays.unsqueeze(0)], dim=1)
            with torch.no_grad():
                if student is not None:
                    # student drives: gallery eyes at TRUE pose, belief
                    # corrupted by EMPIRICAL turn-gated drift (accrues in
                    # curved motion + in-place slip; straights ~free)
                    if last_xy is not None:
                        step_d = float(np.hypot(px - last_xy[0],
                                                py - last_xy[1]))
                        step_y = abs(math.atan2(
                            math.sin(yaw - last_xy[2]),
                            math.cos(yaw - last_xy[2])))
                        g_ = min(1.0, step_y / (0.3 * step_d + 1e-6))
                        walked = min(walked + args.drift_rate * g_ * step_d
                                     + 0.04 * step_y, 1.6)
                    last_xy = (px, py, yaw)
                    bias = walked * drift_u
                    s12 = obs[:, :12].clone()
                    s12[0, 0] += bias[0] / 5.0
                    s12[0, 1] += bias[1] / 5.0
                    rgb = gallery.lookup(kpos, torch.tensor([yaw],
                                                            device=device))
                    act = student(_augment(rgb, train=False),
                                  s12).clamp(-1, 1)
                else:
                    act = policy(obs).clamp(-1, 1)
            cmd = env._envelope(act)[0].cpu().numpy()
            prev_act = act
            # DEPLOY-HONEST CLAMP: the live rig's daemon quantizes to the
            # certified 0.3 m/s; uncapped clips (up to 1.0) made SONIC
            # collapse. Preserve direction, cap magnitude.
            cmd = np.clip(cmd, -0.3, 0.3)
            # planner intent = (yaw_rate, lateral, fwd, hip)
            intent = (float(cmd[2]), float(cmd[1]), float(cmd[0]), 0.687)
            backend.replan(intent)

        # settle: freeze intent to zero for a beat so the robot stands
        intent = (0.0, 0.0, 0.0, 0.687)
        backend.replan(intent)
        for _ in range(int(1.5 * OUTPUT_FPS)):
            frames.append(backend.get_next_frame_resampled(OUTPUT_FPS))
            if backend.should_replan():
                backend.replan(intent)

        fr = np.asarray(frames, dtype=np.float32)
        # shift the whole clip so it starts at the kitchen-frame spawn
        fr[:, 0] += spawn[0]
        fr[:, 1] += spawn[1]
        n = fr.shape[0]
        key = (f"route_{start_name}_to_{goal_name}" if start_name
               else f"route_{ri}_{goal_name}")
        origins[key] = [float(spawn[0]), float(spawn[1])]
        merged[key] = {
            "root_trans_offset": fr[:, :3].copy(),
            "root_rot": fr[:, [4, 5, 6, 3]].copy(),   # wxyz -> xyzw
            "dof": fr[:, 7:].copy(),
            "fps": OUTPUT_FPS,
            "pose_aa": np.zeros((n, 32, 3), dtype=np.float32),
        }
        end = fr[-1, :2] - spawn
        print(f"[gen] {(start_name or 'mid')}->{goal_name:14s} {'REACHED' if reached else 'timeout'} "
              f"{n/OUTPUT_FPS:5.1f}s  end=({end[0]:+.2f},{end[1]:+.2f}) "
              f"goal=({goal_p[0]:+.2f},{goal_p[1]:+.2f})", flush=True)

    # ARRIVAL CHOREOGRAPHY: queue slots lie ON the corridor the door-robot
    # walks through, so uniform departure staggers still collide (a slot is
    # occupied exactly when a deeper-goal robot passes it). Schedule ARRIVAL
    # times instead: robot i arrives args.stagger_s seconds after robot i-1,
    # so every robot clears a slot before its occupant shows up.
    lens = [m["dof"].shape[0] for m in merged.values()]
    gap = int(args.stagger_s * OUTPUT_FPS)
    lead_in = int(args.lead_in_s * OUTPUT_FPS)
    for i, m in enumerate(merged.values()):
        if args.depart_stagger:
            lead = lead_in + i * gap
        else:
            lead = lead_in + (0 if i == 0
                              else max(0, lens[0] + i * gap - lens[i]))
        if lead > 0:
            for k in ("root_trans_offset", "root_rot", "dof", "pose_aa"):
                m[k] = np.concatenate(
                    [np.repeat(m[k][:1], lead, axis=0), m[k]], axis=0)
    # pad all clips to equal length (repeat last frame) so nobody respawns
    nmax = max(m["dof"].shape[0] for m in merged.values())
    for m in merged.values():
        pad = nmax - m["dof"].shape[0]
        if pad > 0:
            for k in ("root_trans_offset", "root_rot", "dof", "pose_aa"):
                m[k] = np.concatenate(
                    [m[k], np.repeat(m[k][-1:], pad, axis=0)], axis=0)
    joblib.dump(merged, args.out, compress=3)
    import json as _json
    with open(str(args.out) + ".origins.json", "w") as fh:
        _json.dump(origins, fh, indent=1)
    print(f"[gen] wrote {args.out}: {len(merged)} routes, "
          f"{nmax / OUTPUT_FPS:.1f}s each")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
