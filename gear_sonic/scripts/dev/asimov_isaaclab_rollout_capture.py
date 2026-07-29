#!/usr/bin/env python3
"""Fast single-pass IsaacLab rollout capturer (obs/action/state ground truth).

The im_eval harness (``gear_sonic/eval_agent_trl.py``) wraps a policy rollout
in a multi-round metric protocol (MPJPE subsets, gather, success bookkeeping)
that takes 30-120 min to "evaluate" under a minute of motion. This tool boots
the SAME env + policy the eval uses (identical config resolution via the
checkpoint-dir config.yaml, identical obs pipeline — the env's observation
manager builds the obs, nothing is reimplemented) and then runs ONE plain
pass: reset -> for each clip run exactly clip_length steps -> dump per-step
ground truth streams. No retries, no rounds, no metrics.

Per step, per clip it records:
  * motion frame index (motion_start_time_steps + time_steps)
  * every obs group fed to the policy (actor_obs, critic_obs, tokenizer, ...)
  * policy action (deterministic act_inference output, same call im_eval uses)
  * applied env action (post action-transform/clip, from extras["env_actions"])
  * root_state_w (pos3 + quat4 wxyz + linvel3 + angvel3), joint_pos, joint_vel
    (IsaacLab joint order; joint_names saved in meta)
  * done / time_out flags (a done before the final frame = fall + auto-reset;
    flagged as terminated_early in meta, data after it is a restarted attempt)

Usage:
    ~/miniconda3/envs/env_isaaclab/bin/python \
        gear_sonic/scripts/dev/asimov_isaaclab_rollout_capture.py \
        --checkpoint out/asimov_bigrun_evals/locoft2_eval/locoft2_9600.pt \
        --motion /path/to/motions.pkl \
        [--num-envs 1] [--steps-per-clip N] [--out DIR] [--no-headless]

Output: <out>/<clip_key>.npz per clip + manifest.json (sanity table incl.
steps recorded, mean |action|, root height range).

--video is accepted but a no-op in v1: the state dump is the priority and the
IM_EVAL rendering path (render_results/enable_cameras) is not trivially
reusable without dragging in the recorder manager. Replay the npz offline.
"""

import argparse
import json
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_T_START = time.time()


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint (config.yaml beside it)")
    p.add_argument("--motion", required=True, help="Motion .pkl file (motion_lib_cfg.motion_file override)")
    p.add_argument("--num-envs", type=int, default=None,
                   help="Default: auto = number of clips in --motion (capped 32), so all clips "
                        "run in ONE batch — the same all-motions-loaded path the eval uses. "
                        "num_envs < num_clips exercises the multi-loop forward_motion_samples "
                        "path, which showed instant-termination artifacts (stale refs).")
    p.add_argument("--steps-per-clip", type=int, default=None,
                   help="Cap steps per clip (default: auto = full clip length)")
    p.add_argument("--out", default=None,
                   help="Output dir (default: out/rollout_capture/<ckpt>_<motion>)")
    p.add_argument("--video", action="store_true",
                   help="No-op in v1 (dump-only); kept for CLI compat")
    p.add_argument("--no-headless", dest="headless", action="store_false", default=True)
    return p.parse_args()


ARGS = _parse_args()
if ARGS.video:
    print("[rollout_capture] --video is a no-op in v1; dumping state streams only", flush=True)

_CKPT = os.path.abspath(ARGS.checkpoint)
_MOTION = os.path.abspath(ARGS.motion)
if ARGS.num_envs is None:
    import joblib

    _n_clips = len(joblib.load(_MOTION))
    ARGS.num_envs = min(_n_clips, 32)
    print(f"[rollout_capture] --num-envs auto: {ARGS.num_envs} ({_n_clips} clips)", flush=True)
elif ARGS.num_envs > 0:
    import joblib

    _n_clips = len(joblib.load(_MOTION))
    if ARGS.num_envs < _n_clips:
        print(
            f"[rollout_capture] WARNING: num_envs={ARGS.num_envs} < {_n_clips} clips -> "
            "multi-loop eval path; observed instant-termination artifacts on reloaded "
            "motions (stale refs). Prefer num_envs >= num_clips.",
            flush=True,
        )
if ARGS.out is None:
    ARGS.out = os.path.join(
        "out", "rollout_capture",
        os.path.splitext(os.path.basename(_CKPT))[0] + "_" + os.path.splitext(os.path.basename(_MOTION))[0],
    )
_OUT_DIR = os.path.abspath(ARGS.out)

# Hand hydra the exact override set the eval CLI uses; everything else
# (config.yaml discovery, eval_overrides merge, module-path rewrites) is
# replicated from eval_agent_trl below.
sys.argv = [sys.argv[0]] + [
    f"++checkpoint={_CKPT}",
    f"++headless={ARGS.headless}",
    f"++num_envs={ARGS.num_envs}",
    f"++manager_env.commands.motion.motion_lib_cfg.motion_file={_MOTION}",
]

import filelock  # noqa: E402
import hydra  # noqa: E402
import numpy as np  # noqa: E402
import omegaconf  # noqa: E402

from gear_sonic.utils import config_utils  # noqa: E402

config_utils.register_rl_resolvers()


@hydra.main(config_path="../../config", config_name="base_eval", version_base="1.1")
def main(override_config: omegaconf.OmegaConf):
    os.chdir(hydra.utils.get_original_cwd())

    from pathlib import Path

    from loguru import logger

    # ---- Config resolution: verbatim from eval_agent_trl.py ----
    checkpoint = Path(override_config.checkpoint)
    config_path = checkpoint.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint.parent.parent / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"No config.yaml next to (or one above) {checkpoint}")

    logger.info(f"Loading training config file from {config_path}")
    with open(config_path) as file:
        raw = file.read()
    raw = raw.replace("groot.rl.trl.", "gear_sonic.trl.")
    raw = raw.replace("groot.rl.envs.", "gear_sonic.envs.")
    raw = raw.replace("groot.rl.utils.", "gear_sonic.utils.")
    raw = raw.replace("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module.")
    raw = raw.replace("groot.rl.agents.", "gear_sonic.trl.")
    raw = raw.replace("groot/rl/data/", "gear_sonic/data/")
    raw = raw.replace("assets/bm/unitree_description/", "assets/robot_description/")
    raw = raw.replace("1215_bones_seed_filtered", "bones_seed_smpl")
    import io

    train_config = omegaconf.OmegaConf.load(io.StringIO(raw))
    if train_config.eval_overrides is not None:
        train_config = omegaconf.OmegaConf.merge(train_config, train_config.eval_overrides)
    config = omegaconf.OmegaConf.merge(train_config, override_config)
    config.experiment_dir = checkpoint.parent

    with omegaconf.open_dict(config):
        for event in config.manager_env.config.get("train_only_events", []):
            if event in config.manager_env.events:
                config.manager_env.events.pop(event)
            remove_schedule_keys = []
            for key in config.trainer.get("schedule_dict", {}):
                if event in key:
                    remove_schedule_keys.append(key)
            for key in remove_schedule_keys:
                config.trainer.schedule_dict.pop(key)
        for termination in config.manager_env.config.get("train_only_terminations", []):
            if termination in config.manager_env.terminations:
                config.manager_env.terminations.pop(termination)

    env_config = config.manager_env

    import torch

    from gear_sonic.utils import common as rl_utils_common

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.multi_gpu = False
    rl_utils_common.seeding(config.seed)

    # ---- App launch: same as eval (headless, no cameras) ----
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="rollout capture")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args([])
    args_cli.num_envs = config.num_envs
    args_cli.seed = config.seed
    args_cli.env_spacing = env_config.config.env_spacing
    args_cli.output_dir = config.output_dir
    args_cli.enable_cameras = False
    args_cli.headless = config.headless
    args_cli.device = device
    base_kit_args = "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
    args_cli.kit_args = base_kit_args + (" --no-window" if args_cli.headless else "")

    with filelock.FileLock("/tmp/isaaclab_app_launcher.lock"):  # noqa: S108
        app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app  # noqa: F841

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if env_config.config.get("save_rendering_dir", None) is None:
        ckpt_num = str(checkpoint.name).split("_")[-1].split(".")[0]
        env_config.config.save_rendering_dir = str(checkpoint.parent / "renderings" / f"ckpt_{ckpt_num}")

    # ---- Env + policy: same construction path as eval_agent_trl ----
    from gear_sonic import train_agent_trl
    from gear_sonic.trl.utils import common as trl_utils_common
    from gear_sonic.trl.utils import scheduler
    from gear_sonic.utils import obs_utils

    env = train_agent_trl.create_manager_env(config, device, args_cli)

    module_dim_dict = getattr(config.algo.config, "module_dim", {})
    env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
    env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
    example_obs = env.reset(flatten_dict_obs=False)
    for key in env.env.observation_space:
        if key not in ["policy", "critic"]:
            group_obs_dims, group_obs_names, group_obs_total_dim = obs_utils.get_group_term_obs_shape(
                example_obs, key
            )
            env.config["obs"]["group_obs_dims"][key] = group_obs_dims
            env.config["obs"]["group_obs_names"][key] = group_obs_names
            env.config["obs"]["obs_dims"][key] = group_obs_total_dim
            env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim

    meta_action_dim = env.config.get("meta_action_dim", None)
    if meta_action_dim is not None and meta_action_dim > 0:
        env.config["robot"]["actions_dim"] = meta_action_dim
    else:
        env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

    policy = trl_utils_common.custom_instantiate(
        config.algo.config.actor,
        env_config=env.config,
        algo_config=config.algo.config,
        module_dim_dict=module_dim_dict,
        backbone_kwargs={},
        _resolve=False,
    ).to(device)

    logger.info(f"Loading checkpoint from {checkpoint}")
    ckpt = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if "actor_model_state_dict" in ckpt:
        state_dict = ckpt["actor_model_state_dict"]
    elif "policy_state_dict" in ckpt:
        state_dict = ckpt["policy_state_dict"]
    else:
        raise KeyError("Checkpoint has neither actor_model_state_dict nor policy_state_dict")

    # std/log_std backward compat (verbatim from eval_agent_trl)
    model_uses_std = "std" in policy.state_dict()
    if model_uses_std and "log_std" in state_dict and "std" not in state_dict:
        state_dict["std"] = torch.exp(state_dict.pop("log_std"))
    elif not model_uses_std and "std" in state_dict and "log_std" not in state_dict:
        state_dict["log_std"] = torch.log(state_dict.pop("std"))
    policy.load_state_dict(state_dict)
    logger.info("Loaded policy state dict")

    global_step = ckpt["state"].global_step
    if "schedule_dict" in config.trainer:
        import easydict

        schedule_wrapper = easydict.EasyDict(env=env, model=easydict.EasyDict(policy=policy))
        scheduler.update_scheduled_params(schedule_wrapper, config.trainer.schedule_dict, global_step)
    env.reinit_dr()

    t_boot = time.time() - _T_START
    logger.info(f"[rollout_capture] boot done in {t_boot:.1f}s (global_step={global_step})")

    # ---- Plain single-pass rollout ----
    policy.eval()
    if hasattr(policy, "eval_mode"):
        policy.eval_mode()
    env.set_is_evaluating(True)  # sequential eval motion loading + reset

    motion_lib = env._motion_lib  # noqa: SLF001
    num_unique = motion_lib._num_unique_motions  # noqa: SLF001
    num_envs = env.num_envs
    num_loops = int(math.ceil(num_unique / num_envs))
    robot = env.env.scene["robot"]
    joint_names = list(robot.data.joint_names)
    fps = float(getattr(motion_lib, "target_fps", 50))

    os.makedirs(_OUT_DIR, exist_ok=True)
    manifest = {
        "checkpoint": str(checkpoint),
        "global_step": int(global_step),
        "motion_file": _MOTION,
        "num_envs": num_envs,
        "fps": fps,
        "step_dt": float(env.env.step_dt),
        "joint_names_isaaclab": joint_names,
        "clips": [],
    }

    t_roll0 = time.time()
    clips_done = 0
    with torch.no_grad():
        for loop_idx in range(num_loops):
            if loop_idx > 0:
                env.forward_motion_samples(0, 1)  # advances start_idx + resets
            obs_dict = env.reset_all()
            for k in obs_dict:
                obs_dict[k] = obs_dict[k].to(device)
            policy.init_rollout()

            motion_ids = env.motion_ids.clone()
            clip_steps = motion_lib.get_motion_num_steps(motion_ids).cpu().numpy()
            global_ids = (env.start_idx + torch.arange(num_envs, device=motion_ids.device)).cpu().numpy()
            valid_envs = [e for e in range(num_envs) if global_ids[e] < num_unique]
            try:
                keys = [str(motion_lib.curr_motion_keys[motion_ids[e].item()]) for e in range(num_envs)]
            except Exception:  # noqa: BLE001
                keys = [str(motion_lib._motion_data_keys[global_ids[e] % num_unique]) for e in range(num_envs)]  # noqa: SLF001

            per_env_T = np.minimum(clip_steps, ARGS.steps_per_clip) if ARGS.steps_per_clip else clip_steps
            batch_T = int(per_env_T[valid_envs].max())
            logger.info(
                f"[rollout_capture] batch {loop_idx + 1}/{num_loops}: "
                + ", ".join(f"{keys[e]}({per_env_T[e]})" for e in valid_envs)
            )

            rec = {
                "motion_frame": [], "policy_action": [], "env_action": [],
                "root_state_w": [], "joint_pos": [], "joint_vel": [],
                "done": [], "time_out": [],
            }
            obs_rec = {k: [] for k in obs_dict}
            dones = torch.zeros(num_envs, dtype=torch.long, device=device)
            mc = env.motion_command

            for _step in range(batch_T):
                # State/obs snapshot at time t (pre-step, aligned with obs fed to policy)
                rec["motion_frame"].append(
                    (mc.motion_start_time_steps + mc.time_steps).cpu().numpy().copy()
                )
                rec["root_state_w"].append(robot.data.root_state_w.cpu().numpy().copy())
                rec["joint_pos"].append(robot.data.joint_pos.cpu().numpy().copy())
                rec["joint_vel"].append(robot.data.joint_vel.cpu().numpy().copy())
                for k in obs_rec:
                    obs_rec[k].append(obs_dict[k].cpu().numpy().astype(np.float32, copy=True))

                # Exact im_eval inference call (deterministic mean)
                actions = policy.act_inference(
                    obs_dict=obs_dict, cur_dones=dones, skip_episode_attnmask=True
                )
                rec["policy_action"].append(actions.cpu().numpy().copy())

                obs_dict, _rew, dones, extras = env.step({"actions": actions, "obs_dict": obs_dict})
                rec["env_action"].append(np.asarray(extras["env_actions"]).copy())
                rec["done"].append(dones.cpu().numpy().copy())
                rec["time_out"].append(extras["time_outs"].cpu().numpy().copy())
                for k in obs_dict:
                    obs_dict[k] = obs_dict[k].to(device)

            # ---- Per-clip slicing + dump ----
            for e in valid_envs:
                T = int(per_env_T[e])
                key = keys[e]
                safe_key = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in key)
                arrays = {name: np.stack([fr[e] for fr in frames[:T]]) for name, frames in rec.items()}
                for k, frames in obs_rec.items():
                    arrays[f"obs__{k}"] = np.stack([fr[e] for fr in frames[:T]])

                done_arr = arrays["done"].astype(bool)
                timeout_arr = arrays["time_out"].astype(bool)
                early = done_arr & ~timeout_arr
                terminated_early = bool(early[:-1].any()) if T > 1 else False
                z = arrays["root_state_w"][:, 2]
                mean_abs_act = float(np.abs(arrays["policy_action"]).mean())

                out_path = os.path.join(_OUT_DIR, f"{safe_key}.npz")
                np.savez_compressed(
                    out_path,
                    motion_key=np.str_(key),
                    fps=np.float32(fps),
                    **arrays,
                )
                sz = os.path.getsize(out_path)
                clip_info = {
                    "motion_key": key,
                    "file": out_path,
                    "steps_recorded": T,
                    "clip_steps": int(clip_steps[e]),
                    "mean_abs_policy_action": mean_abs_act,
                    "root_z_min": float(z.min()),
                    "root_z_max": float(z.max()),
                    "terminated_early": terminated_early,
                    "first_early_done_step": int(np.argmax(early)) if early.any() else None,
                    "size_bytes": sz,
                }
                manifest["clips"].append(clip_info)
                clips_done += 1
                logger.info(
                    f"[rollout_capture] {key}: {T} steps, mean|a|={mean_abs_act:.3f}, "
                    f"root_z=[{z.min():.3f},{z.max():.3f}], "
                    f"{'TERMINATED EARLY' if terminated_early else 'upright/complete'}, "
                    f"{sz / 1e6:.1f} MB -> {out_path}"
                )

    t_roll = time.time() - t_roll0
    manifest["boot_seconds"] = round(t_boot, 1)
    manifest["rollout_seconds"] = round(t_roll, 1)
    manifest["total_seconds"] = round(time.time() - _T_START, 1)
    with open(os.path.join(_OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        f"[rollout_capture] DONE: {clips_done} clips -> {_OUT_DIR} "
        f"(boot {t_boot:.1f}s, rollout {t_roll:.1f}s, total {time.time() - _T_START:.1f}s)"
    )

    os._exit(0)  # IsaacSim teardown hangs otherwise; same exit as eval


if __name__ == "__main__":
    main()
