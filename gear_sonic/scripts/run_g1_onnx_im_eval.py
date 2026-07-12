#!/usr/bin/env python3
"""Run the STOCK G1 SONIC ONNX through the IsaacLab ``im_eval`` feasibility sweep.

There is no gear_sonic-native G1 ``.pt``; the released policy is an encoder+decoder
ONNX pair. This launcher injects ``G1OnnxPolicyShim`` as ``model.policy`` into
``gear_sonic/eval_agent_trl.py`` (which normally ``torch.load``s a ``.pt``), so the
stock ONNX can be swept over a G1 motion-lib PKL. It follows the monkeypatch +
``runpy`` handoff pattern of ``dump_isaaclab_step0.py``.

It reuses the released **sonic_release** training config (which selects the g1
robot, the 3-encoder UniversalToken actor, ``num_future_frames=10`` == deploy
``10frame_step5``, ``target_fps=50``) from an existing run dir's ``config.yaml``,
and satisfies the checkpoint loader with a throwaway ``.pt`` (the shim ignores it).

Usage (env_isaaclab):
    python gear_sonic/scripts/run_g1_onnx_im_eval.py \
        --motion-file /path/to/g1_parity_walk.pkl \
        --num-envs 4 \
        --eval-output-dir /tmp/g1_onnx_eval

Everything after ``--`` (or any ``++key=val``) is forwarded to eval_agent_trl.
"""

import argparse
import os
import runpy
import shutil
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_DEFAULT_CONFIG_SRC = (
    "logs_rl/TRL_G1_Track/manager/universal_token/all_modes/"
    "sonic_release_test-20260416_230440/config.yaml"
)
_DEFAULT_ENC = "gear_sonic_deploy/policy/release/model_encoder.onnx"
_DEFAULT_DEC = "gear_sonic_deploy/policy/release/model_decoder.onnx"

_ENV_HOLDER = {"env": None}


def _install_env_capture():
    from gear_sonic import train_agent_trl

    orig = train_agent_trl.create_manager_env

    def patched(*a, **k):
        env = orig(*a, **k)
        _ENV_HOLDER["env"] = env
        print("[run_g1_onnx_im_eval] captured manager env", flush=True)
        return env

    train_agent_trl.create_manager_env = patched


def _install_shim_injection(enc_path, dec_path):
    from gear_sonic.scripts.g1_onnx_policy_shim import G1OnnxPolicyShim
    from gear_sonic.trl.trainer import ppo_trainer

    orig_init = ppo_trainer.PolicyAndValueWrapper.__init__

    def patched_init(self, policy, value_model, **kwargs):
        orig_init(self, policy, value_model, **kwargs)
        env = _ENV_HOLDER["env"]
        assert env is not None, "env not captured before model build"
        inner = env.env if hasattr(env, "env") else env
        device = getattr(env, "device", None) or getattr(inner, "device")
        shim = G1OnnxPolicyShim(
            os.path.join(_REPO_ROOT, enc_path),
            os.path.join(_REPO_ROOT, dec_path),
            device=device,
            env=inner,
        )
        self.policy = shim  # valid: shim is an nn.Module
        print(
            f"[run_g1_onnx_im_eval] injected G1OnnxPolicyShim as model.policy "
            f"(device={device}, num_envs={getattr(inner, 'num_envs', '?')})",
            flush=True,
        )

    ppo_trainer.PolicyAndValueWrapper.__init__ = patched_init


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion-file", required=True, help="G1 motion-lib PKL to sweep")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--eval-output-dir", default="/tmp/g1_onnx_eval")
    p.add_argument("--config-src", default=_DEFAULT_CONFIG_SRC)
    p.add_argument("--encoder-onnx", default=_DEFAULT_ENC)
    p.add_argument("--decoder-onnx", default=_DEFAULT_DEC)
    p.add_argument("--run-dir", default="/tmp/g1_onnx_eval_run",
                   help="scratch dir holding the throwaway checkpoint + config.yaml")
    p.add_argument("extra", nargs="*", help="extra ++key=val overrides for eval_agent_trl")
    args = p.parse_args()

    os.chdir(_REPO_ROOT)

    # --- set up a throwaway checkpoint whose parent dir carries the config ----
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy(os.path.join(_REPO_ROOT, args.config_src), os.path.join(run_dir, "config.yaml"))
    ckpt_path = os.path.join(run_dir, "model_step_000000.pt")
    import torch
    torch.save(
        {"state": types.SimpleNamespace(global_step=0), "actor_model_state_dict": {}},
        ckpt_path,
    )
    os.makedirs(args.eval_output_dir, exist_ok=True)

    os.environ.setdefault("WANDB_MODE", "disabled")

    # --- install patches BEFORE handing off to the hydra entry point ---------
    _install_env_capture()
    _install_shim_injection(args.encoder_onnx, args.decoder_onnx)

    # --- build eval_agent_trl argv -------------------------------------------
    motion_file = os.path.abspath(args.motion_file)
    overrides = [
        f"+checkpoint={ckpt_path}",
        "+headless=True",
        f"++num_envs={args.num_envs}",
        "++eval_callbacks=im_eval",
        "++run_eval_loop=true",
        f"++eval_output_dir={args.eval_output_dir}",
        f"++manager_env.commands.motion.motion_lib_cfg.motion_file={motion_file}",
    ]
    overrides += list(args.extra)

    eval_script = os.path.join(_REPO_ROOT, "gear_sonic", "eval_agent_trl.py")
    sys.argv = [eval_script] + overrides
    print(f"[run_g1_onnx_im_eval] handing off to eval_agent_trl with:\n  "
          + "\n  ".join(overrides), flush=True)
    runpy.run_path(eval_script, run_name="__main__")


if __name__ == "__main__":
    main()
