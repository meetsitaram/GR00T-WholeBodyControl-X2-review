#!/usr/bin/env python3
"""Dump an IsaacLab evaluation TRAJECTORY for policy stability diagnosis.

Sister script to ``dump_isaaclab_step0.py``. Where the step-0 dump captures
one frame of obs/intermediates for parity-vs-MuJoCo comparison, this script
runs the policy for ``max_render_steps`` ticks and records, per step:

    * decoder action_mean (the raw policy output the deploy clamps)
    * env GT joint_pos, root_pos_w (height), root_quat_w_wxyz (upright)
    * per-step done flag

This is the diagnostic for "does the trained policy hold standing in the
training simulator at all?" If the action norm explodes or root_pos_w[z]
drops within a few hundred ticks, the policy itself is unstable -- no point
debugging the deploy further until the checkpoint is retrained.

Usage:
    python gear_sonic/scripts/dump_isaaclab_trajectory.py \\
        +checkpoint=$HOME/x2_cloud_checkpoints/run-.../model_step_NNNNNN.pt \\
        +headless=True \\
        ++num_envs=1 \\
        ++run_eval_loop=true \\
        ++max_render_steps=500 \\
        ++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/motions/x2_ultra_idle_stand.pkl \\
        ++traj_path=/tmp/x2_traj_isaaclab_idle.pt
"""

import os
import sys
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


_TRAJ_PATH = None
for _arg in sys.argv:
    if _arg.startswith("++traj_path=") or _arg.startswith("traj_path="):
        _TRAJ_PATH = _arg.split("=", 1)[1]
        break
if _TRAJ_PATH is None:
    _TRAJ_PATH = "/tmp/x2_traj_isaaclab.pt"
sys.argv = [a for a in sys.argv if not (a.startswith("++traj_path=") or a.startswith("traj_path="))]

print(f"[dump_isaaclab_trajectory] Will dump trajectory to: {_TRAJ_PATH}", flush=True)


_TRAJ = {
    "action_mean": [],
    "joint_pos": [],
    "root_pos_w": [],
    "root_quat_w_wxyz": [],
    "root_lin_vel_b": [],
    "root_ang_vel_b": [],
    "step_idx": [],
}
_INNER_ENV_HOLDER = {"env": None}
_STEP_COUNT = {"n": 0}


def _capture_env_snapshot():
    """Pull joint/root state from the live IsaacLab env (1-env)."""
    inner = _INNER_ENV_HOLDER["env"]
    if inner is None:
        return None
    try:
        robot = inner.scene["robot"]
        return {
            "joint_pos": robot.data.joint_pos.detach().cpu().clone(),
            "root_pos_w": robot.data.root_pos_w.detach().cpu().clone(),
            "root_quat_w_wxyz": robot.data.root_quat_w.detach().cpu().clone(),
            "root_lin_vel_b": robot.data.root_lin_vel_b.detach().cpu().clone(),
            "root_ang_vel_b": robot.data.root_ang_vel_b.detach().cpu().clone(),
        }
    except Exception as exc:  # noqa: BLE001
        print(f"[dump_isaaclab_trajectory] env snapshot failed: {exc}", flush=True)
        return None


def _install_forward_capture_patch():
    """Patch UniversalTokenModule.forward to capture action_mean every step."""
    from gear_sonic.trl.modules import universal_token_modules as utm

    original_forward = utm.UniversalTokenModule.forward

    def patched_forward(self, input_data, compute_aux_loss=False, return_dict=False, **kwargs):
        out = original_forward(
            self, input_data, compute_aux_loss=compute_aux_loss,
            return_dict=return_dict, **kwargs
        )
        # ``out`` for the g1_dyn decoder path is a dict containing the action.
        # Grab whatever the decoder produced (action mean tensor of shape
        # (B, S, A)). We tolerate either dict-with-"action" or a raw tensor.
        action = None
        if isinstance(out, dict):
            for k in ("action", "actions", "action_mean", "decoder_action_mean"):
                if k in out:
                    action = out[k]
                    break
            if action is None:
                # Sometimes the action is nested one level deeper.
                for v in out.values():
                    if isinstance(v, dict):
                        for k in ("action", "actions", "action_mean"):
                            if k in v:
                                action = v[k]
                                break
                    if action is not None:
                        break
        elif torch.is_tensor(out):
            action = out

        if action is not None:
            _TRAJ["action_mean"].append(action.detach().cpu().clone())
            snap = _capture_env_snapshot()
            if snap is not None:
                _TRAJ["joint_pos"].append(snap["joint_pos"])
                _TRAJ["root_pos_w"].append(snap["root_pos_w"])
                _TRAJ["root_quat_w_wxyz"].append(snap["root_quat_w_wxyz"])
                _TRAJ["root_lin_vel_b"].append(snap["root_lin_vel_b"])
                _TRAJ["root_ang_vel_b"].append(snap["root_ang_vel_b"])
            _TRAJ["step_idx"].append(_STEP_COUNT["n"])
            _STEP_COUNT["n"] += 1
            if _STEP_COUNT["n"] % 50 == 0:
                a = action.detach().abs()
                z_str = ""
                if _TRAJ["root_pos_w"]:
                    z_str = f"  z={_TRAJ['root_pos_w'][-1][0,2].item():.3f}"
                print(
                    f"[traj] step={_STEP_COUNT['n']:4d}  "
                    f"max|a|={a.max().item():.3f}  mean|a|={a.mean().item():.3f}{z_str}",
                    flush=True,
                )
                # eval_agent_trl.py calls os._exit(0) at the end which skips
                # atexit, so checkpoint the trajectory periodically to disk.
                _save_traj()
        return out

    utm.UniversalTokenModule.forward = patched_forward
    print("[dump_isaaclab_trajectory] Installed forward capture patch.", flush=True)


def _install_env_capture_patch():
    """Hook env creation so we can grab the inner IsaacLab env handle."""
    from gear_sonic import train_agent_trl

    original_create = train_agent_trl.create_manager_env

    def patched_create(*args, **kwargs):
        env = original_create(*args, **kwargs)
        inner = env.env if hasattr(env, "env") else env
        _INNER_ENV_HOLDER["env"] = inner
        print(f"[dump_isaaclab_trajectory] Captured inner env handle: {type(inner).__name__}",
              flush=True)
        return env

    train_agent_trl.create_manager_env = patched_create
    print("[dump_isaaclab_trajectory] Installed env capture patch.", flush=True)


def _save_traj():
    """Persist whatever frames we have so far. Safe to call repeatedly."""
    if not _TRAJ["action_mean"]:
        return
    out = {
        "action_mean": torch.stack(_TRAJ["action_mean"], dim=0),
        "step_idx": torch.tensor(_TRAJ["step_idx"], dtype=torch.long),
    }
    for k in ("joint_pos", "root_pos_w", "root_quat_w_wxyz",
              "root_lin_vel_b", "root_ang_vel_b"):
        if _TRAJ[k]:
            out[k] = torch.stack(_TRAJ[k], dim=0)
    torch.save(out, _TRAJ_PATH)


def _install_atexit_dump():
    """Save trajectory on normal exit, Ctrl-C, or eval_agent_trl's os._exit."""
    import atexit, signal

    def _save_with_summary():
        _save_traj()
        if not _TRAJ["action_mean"]:
            print("[dump_isaaclab_trajectory] No frames captured; nothing to save.",
                  flush=True)
            return
        out = torch.load(_TRAJ_PATH, map_location="cpu", weights_only=False)
        print(f"[dump_isaaclab_trajectory] Wrote {len(_TRAJ['action_mean'])} frames -> {_TRAJ_PATH}",
              flush=True)
        a = out["action_mean"].abs()
        print(f"[dump_isaaclab_trajectory] action_mean |.| stats: "
              f"max={a.max().item():.3f}  mean={a.mean().item():.3f}", flush=True)
        if "root_pos_w" in out and out["root_pos_w"].shape[1] >= 1:
            z = out["root_pos_w"][:, 0, 2]
            print(f"[dump_isaaclab_trajectory] root_pos_w[z]: "
                  f"start={z[0].item():.3f}  end={z[-1].item():.3f}  "
                  f"min={z.min().item():.3f}", flush=True)

    atexit.register(_save_with_summary)

    def _sig_handler(signum, frame):  # noqa: ARG001
        _save_with_summary()
        os._exit(0)
    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # eval_agent_trl.py calls ``os._exit(0)`` after the loop which skips
    # atexit. Wrap it so we save first.
    _orig_os_exit = os._exit

    def _wrapped_os_exit(code):
        try:
            _save_with_summary()
        finally:
            _orig_os_exit(code)
    os._exit = _wrapped_os_exit


def main():
    _install_env_capture_patch()
    _install_forward_capture_patch()
    _install_atexit_dump()

    import runpy
    eval_script_path = os.path.join(_REPO_ROOT, "gear_sonic", "eval_agent_trl.py")
    print(f"[dump_isaaclab_trajectory] Handing off to {eval_script_path}", flush=True)
    sys.argv[0] = eval_script_path
    runpy.run_path(eval_script_path, run_name="__main__")


if __name__ == "__main__":
    main()
