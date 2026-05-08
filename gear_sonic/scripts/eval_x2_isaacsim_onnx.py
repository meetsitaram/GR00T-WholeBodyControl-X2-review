#!/usr/bin/env python3
"""Run the deploy ONNX policy inside the IsaacLab eval pipeline.

This is a thin wrapper around ``gear_sonic.eval_agent_trl.main`` that:

1. Pre-parses our own CLI flags (``--onnx``, ``--run-dir``, ``--compare``,
   ``--diff-csv``) and removes them from ``sys.argv`` before Hydra sees it.
2. If ``--run-dir`` is given (and no explicit ``checkpoint=`` override),
   appends ``checkpoint=<run-dir>/last.pt`` so Hydra finds the run's
   ``config.yaml``. We still load the .pt to instantiate the
   ``actor_module`` skeleton (which the shim borrows
   ``parse_tokenizer_obs`` / ``encoder_input_features`` from), but its
   weights are never used at inference unless ``--compare`` is set.
3. Monkey-patches ``gear_sonic.trl.modules.actor_critic_modules.Actor.forward``
   to route all action queries through an :class:`OnnxPolicyShim`, so the
   IsaacLab eval loop drives the env using the ONNX policy.

Why both .pt and run-dir?
-------------------------
Only one is strictly needed. ``--run-dir`` is the cleanest entry point:
the script will set ``checkpoint=<run-dir>/last.pt`` for you. We need
the .pt only for its ``config.yaml`` neighbor (env/motion/asset wiring)
and for the constructed actor_module's ``parse_tokenizer_obs`` helper
that knows how to slice the tokenizer obs. The .pt's WEIGHTS are unused
in normal mode -- only the ONNX is queried for actions.

Examples
--------
Bare-minimum drive ONNX in IsaacSim::

    python -m gear_sonic.scripts.eval_x2_isaacsim_onnx \\
        --run-dir $HOME/x2_cloud_checkpoints/run-20260420_083925 \\
        --onnx /path/to/x2_sonic_16k.onnx \\
        num_envs=1 headless=False

Compare ONNX vs in-sim .pt action-by-action (writes a CSV)::

    python -m gear_sonic.scripts.eval_x2_isaacsim_onnx \\
        --run-dir $HOME/x2_cloud_checkpoints/run-20260420_083925 \\
        --onnx /path/to/x2_sonic_16k.onnx \\
        --compare \\
        --diff-csv logs/x2/onnx_vs_pt.csv \\
        num_envs=1 headless=False

In ``--compare`` mode the .pt drives the env (so the rollout stays
on-distribution), and ONNX is queried alongside on every step purely
for diagnostic logging.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# We must keep our argparse out of Hydra's argv. Pre-parse and strip.

_OWN_FLAGS = {"--onnx", "--run-dir", "--compare", "--diff-csv", "--encoder-name"}


def _split_argv(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    """Pull our flags out of argv before Hydra sees the rest."""
    parser = argparse.ArgumentParser(
        prog="eval_x2_isaacsim_onnx",
        description=(
            "Run a deploy ONNX policy inside IsaacLab. All non-recognized "
            "args are forwarded to gear_sonic.eval_agent_trl (Hydra)."
        ),
        add_help=False,
    )
    parser.add_argument("--onnx", required=True, type=str,
                        help="Path to the deploy ONNX (e.g. x2_sonic_16k.onnx).")
    parser.add_argument("--run-dir", default=None, type=str,
                        help="Run directory containing config.yaml + last.pt. "
                             "If set and no checkpoint=... override is present, "
                             "we add checkpoint=<run-dir>/last.pt automatically.")
    parser.add_argument("--compare", action="store_true",
                        help="Drive env with .pt actions and run ONNX alongside, "
                             "logging per-step action diff to --diff-csv.")
    parser.add_argument("--diff-csv", default="logs/x2/onnx_vs_pt.csv", type=str,
                        help="CSV path for --compare mode (default: logs/x2/onnx_vs_pt.csv).")
    parser.add_argument("--encoder-name", default="g1", type=str,
                        help="Which encoder bank to assemble for the ONNX input "
                             "(default: g1; matches x2_sonic_16k.onnx).")
    parser.add_argument("-h", "--help", action="store_true",
                        help="Print this help and exit (does not show Hydra help).")

    own = []
    rest = []
    i = 1
    while i < len(argv):
        tok = argv[i]
        # Match --flag=val or --flag val for our own flags.
        head = tok.split("=", 1)[0]
        if head in _OWN_FLAGS or head in ("-h", "--help"):
            if "=" in tok or head in ("--compare", "-h", "--help"):
                own.append(tok)
                i += 1
            else:
                # Take the next token as the value.
                own.append(tok)
                if i + 1 < len(argv):
                    own.append(argv[i + 1])
                    i += 2
                else:
                    i += 1
        else:
            rest.append(tok)
            i += 1

    args = parser.parse_args(own)
    if getattr(args, "help", False):
        parser.print_help()
        sys.exit(0)

    return args, rest


def _ensure_checkpoint_override(rest_argv: list[str], run_dir: str | None) -> list[str]:
    """If user passed --run-dir without checkpoint=, inject the default."""
    if run_dir is None:
        return rest_argv
    if any(tok.startswith("checkpoint=") for tok in rest_argv):
        return rest_argv
    ckpt = Path(run_dir) / "last.pt"
    if not ckpt.exists():
        # Fall back to the most recent model_step_*.pt.
        candidates = sorted(Path(run_dir).glob("model_step_*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"--run-dir {run_dir} has no last.pt or model_step_*.pt"
            )
        ckpt = candidates[-1]
    print(f"[eval_x2_isaacsim_onnx] Using checkpoint={ckpt} (loaded for actor_module skeleton + config.yaml)",
          flush=True)
    return rest_argv + [f"checkpoint={ckpt}"]


def _patch_actor_with_onnx(onnx_path: str, encoder_name: str, compare: bool, diff_csv: str):
    """Monkey-patch ``Actor.forward`` to route through the ONNX shim.

    Must be called BEFORE eval_agent_trl.main() instantiates the model.
    """
    from gear_sonic.scripts.onnx_policy_shim import CompareLogger, OnnxPolicyShim
    from gear_sonic.trl.modules import actor_critic_modules as ac

    original_forward = ac.Actor.forward

    state = {"shim": None, "logger": None, "compare": compare, "diff_csv": diff_csv}

    def _patched_forward(self, obs_dict, is_training=False, **kwargs):
        # Lazy-build the shim on first call so we have access to actor_module.
        if state["shim"] is None:
            state["shim"] = OnnxPolicyShim(
                actor_module=self.actor_module,
                onnx_path=onnx_path,
                encoder_name=encoder_name,
            )
            print(f"[eval_x2_isaacsim_onnx] OnnxPolicyShim attached "
                  f"(onnx={onnx_path}, encoder={encoder_name}, compare={state['compare']})",
                  flush=True)

        # Reproduce the running_mean_std normalization Actor.forward applies.
        normalized_obs = obs_dict.copy()
        if self.running_mean_std is not None:
            with __import__("torch").no_grad():
                normalized_obs[self.input_key] = self.running_mean_std(
                    obs_dict[self.input_key]
                )

        action_onnx = state["shim"](normalized_obs)

        if not state["compare"]:
            return action_onnx

        # --compare: also run the original PyTorch path, log diff, but
        # let the .pt drive the env so we stay on-distribution.
        action_pt = original_forward(self, obs_dict, is_training=is_training, **kwargs)
        if state["logger"] is None:
            state["logger"] = CompareLogger(
                state["diff_csv"], action_dim=int(action_pt.shape[-1])
            )
        # Take the last timestep for logging (matches what rollout uses).
        a_pt_last = action_pt[:, -1] if action_pt.dim() == 3 else action_pt
        a_on_last = action_onnx[:, -1] if action_onnx.dim() == 3 else action_onnx
        state["logger"].log(a_pt_last, a_on_last)
        return action_pt

    ac.Actor.forward = _patched_forward
    print(f"[eval_x2_isaacsim_onnx] Patched Actor.forward to use ONNX "
          f"({'COMPARE mode: pt drives env, onnx logged' if compare else 'ONNX drives env'}).",
          flush=True)


def main():
    args, rest_argv = _split_argv(sys.argv)

    onnx_path = os.path.abspath(args.onnx)
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"--onnx {onnx_path} not found")

    rest_argv = _ensure_checkpoint_override(rest_argv, args.run_dir)
    sys.argv = [sys.argv[0]] + rest_argv

    # Patch BEFORE importing/calling main() so even early model construction
    # sees the patched class. The patch lives on the imported class; we
    # import the module once here to make the patch resolve, then run
    # eval_agent_trl as if invoked via ``python -m`` so Hydra's config
    # resolution sees the same module/CWD it does in the normal flow.
    _patch_actor_with_onnx(
        onnx_path=onnx_path,
        encoder_name=args.encoder_name,
        compare=args.compare,
        diff_csv=args.diff_csv,
    )

    # ``runpy`` is the canonical way to programmatically execute a module
    # as ``__main__``, which is what @hydra.main requires for config_path
    # resolution to find ``gear_sonic/config/*.yaml`` (the @hydra.main
    # decorator on ``eval_agent_trl.main`` uses the calling module's
    # location to locate config; importing-and-calling main() instead
    # leaves Hydra confused about where ``config/`` lives and you get
    # "Primary config module 'gear_sonic.config' not found").
    import runpy

    runpy.run_module("gear_sonic.eval_agent_trl", run_name="__main__")


if __name__ == "__main__":
    main()
