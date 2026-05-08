#!/usr/bin/env python3
"""Slot-by-slot diff: IsaacLab step-0 obs vs MuJoCo eval rebuild.

The user's checkpoint trains and evaluates cleanly in IsaacSim but fails on
the deploy stack (which inherits MuJoCo's obs construction). This script is
the only ground-truth check that doesn't require the real robot:

  1. Read /tmp/x2_step0_isaaclab.pt (produced by dump_isaaclab_step0.py).
     That dump contains:
       - actor_obs[0] (1670)            -- the exact tensor IsaacLab fed
       - tokenizer_obs dict              -- per-feature parsed slices
       - proprioception_input            -- 990 from IsaacLab's term order
       - decoder_action_mean             -- the action IsaacLab produced
       - env_state.{joint_pos, joint_vel, root_quat_w_wxyz, root_ang_vel_b,
                    motion_ids, motion_times, default_joint_pos, ...}

  2. Use env_state to drive the MuJoCo `build_tokenizer_obs` /
     `ProprioceptionBuffer` exactly as eval_x2_mujoco.py does, producing
     a rebuilt 1670-D actor_obs.

  3. Print per-block max-abs diff and first few mismatching indices. Run
     the ONNX session on BOTH vectors and print both action norms.

Run:

    python gear_sonic_deploy/scripts/compare_isaaclab_vs_mujoco_obs.py \
        --dump /tmp/x2_step0_isaaclab.pt \
        --model gear_sonic_deploy/models/x2_sonic_16k.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.eval_x2_mujoco import (  # noqa: E402
    DEFAULT_DOF,
    HISTORY_LEN,
    IL_TO_MJ_DOF,
    MJ_TO_IL_DOF,
    NUM_DOFS,
    NUM_FUTURE_FRAMES,
    DT_FUTURE_REF,
    ProprioceptionBuffer,
    build_tokenizer_obs,
    quat_rotate_inverse,
)
from gear_sonic.scripts.eval_x2_mujoco_onnx import (  # noqa: E402
    OnnxActor,
    PROP_DIM,
    TOK_DIM,
)


def _block_diff(
    name: str,
    a: np.ndarray,
    b: np.ndarray,
    *,
    head: int = 8,
    noise_envelope: float | None = None,
):
    """Print per-block max-abs / L2 diff between IsaacLab and MuJoCo views.

    ``noise_envelope`` is the half-width of the
    ``isaaclab.utils.noise.AdditiveUniformNoiseCfg`` configured for the
    matching observation term (see
    ``gear_sonic/config/manager_env/observations/policy/global.yaml`` and
    ``observations/tokenizer/...``). When set, we annotate the line with
    "OK (within X)" if inf <= envelope, else "GAP (>X)" -- so you don't
    confuse the configured corruption noise (G6 in sim2sim_mujoco.md)
    with a real eval_x2_mujoco vs IsaacLab obs-pipeline bug.

    Pass --no-noise (or run dump with
    ``++manager_env.observations.policy.enable_corruption=False``) to
    drive the envelope to zero on the IL side and reveal the underlying
    parity directly.
    """
    diff = np.abs(a - b)
    inf = float(diff.max()) if diff.size else 0.0
    l2 = float(np.linalg.norm(diff))
    n_bad = int((diff > 1e-4).sum())
    if noise_envelope is not None:
        verdict = (
            f"OK (<= ±{noise_envelope:.3g} noise)"
            if inf <= noise_envelope
            else f"GAP (> ±{noise_envelope:.3g} noise)"
        )
        line_tail = f"  [{verdict}]"
    else:
        line_tail = ""
    print(f"  {name:<28s}  size={diff.size:5d}  inf={inf:.4e}  L2={l2:.4e}  "
          f"  >1e-4: {n_bad}/{diff.size}{line_tail}")
    if (noise_envelope is None or inf > noise_envelope) and inf > 1e-4 and head > 0:
        bad_idx = np.argsort(diff)[::-1][:head]
        for i in bad_idx:
            print(f"      idx {int(i):4d}  IL={a[i]:+.5f}  MJ={b[i]:+.5f}  "
                  f"diff={diff[i]:+.5f}")


# Per-term noise envelopes from
# gear_sonic/config/manager_env/observations/policy/global.yaml.
# These are the half-widths of AdditiveUniformNoiseCfg(n_min=-X, n_max=+X)
# applied to each proprioception term when ``enable_corruption=True``
# (the default in eval). The MuJoCo rebuild does NOT add noise, so the
# IL-vs-MJ inf-diff is bounded above by these values per element. If we
# see anything OUTSIDE these envelopes that's a real obs-pipeline gap.
PROP_NOISE_ENVELOPE = {
    "base_ang_vel":  0.2,
    "joint_pos_rel": 0.01,
    "joint_vel":     0.5,
    "last_action":   0.0,   # no noise on actions
    "gravity_dir":   0.05,  # not in global.yaml; conservative (often 0 too)
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, type=Path,
                    help="path to /tmp/x2_step0_isaaclab.pt")
    ap.add_argument("--model", required=False, type=Path, default=None,
                    help="OPTIONAL: path to the same ONNX file IsaacLab eval "
                         "used. If omitted, the slot-by-slot obs diff still "
                         "runs (that's the actual sim-sim parity signal); we "
                         "just skip the final 'ONNX action on each vector' "
                         "sanity block. Useful when you don't trust the "
                         "current ONNX export or just want the obs diff.")
    ap.add_argument("--motion", type=Path, default=None,
                    help="motion .pkl matching env_state.motion_ids[0]; "
                         "if omitted, we attempt to find one in "
                         "gear_sonic/data/motions/")
    ap.add_argument("--env-idx", type=int, default=0,
                    help="which env in the dump batch to use (default 0)")
    args = ap.parse_args()

    dump = torch.load(args.dump, map_location="cpu", weights_only=False)

    # IsaacLab's full 1670-D actor_obs is NOT in dump["actor_obs"] (which is
    # the 990-D proprioception alone). It's split across two top-level
    # tensors that are also what the C++ deploy comparator
    # (compare_deploy_vs_isaaclab_obs.py) consumes:
    #
    #   tok_il  = encoder_input_for_mlp_view  (680,)  -- (10, 68) flatten of
    #             per-frame [command(62) | ori(6)], the ONNX-grouped layout
    #             the g1 encoder consumes.
    #   prop_il = proprioception_input        (990,)  -- 10-frame history of
    #             [base_ang_vel(3) | joint_pos_rel(31) | joint_vel(31) |
    #              last_action(31) | gravity_dir(3)], term-wise stacked.
    #
    # Concat in that order to reproduce the 1670-D actor_obs the policy sees.
    tok_il = dump["encoder_input_for_mlp_view"].squeeze().cpu().numpy().astype(np.float32)
    prop_il = dump["proprioception_input"].squeeze().cpu().numpy().astype(np.float32)
    if tok_il.shape != (680,):
        sys.exit(f"unexpected encoder_input_for_mlp_view shape {tok_il.shape}; "
                 f"expected (680,)")
    if prop_il.shape != (990,):
        sys.exit(f"unexpected proprioception_input shape {prop_il.shape}; "
                 f"expected (990,)")
    actor_obs_il = np.concatenate([tok_il, prop_il])
    print(f"IsaacLab actor_obs:  shape={actor_obs_il.shape}  "
          f"min={actor_obs_il.min():+.3f} max={actor_obs_il.max():+.3f}  "
          f"(rebuilt from encoder_input_for_mlp_view + proprioception_input)")
    print(f"IsaacLab decoder_action_mean[{args.env_idx}]: "
          f"{dump['decoder_action_mean'][args.env_idx].cpu().numpy()}")

    env_state = dump.get("env_state")
    if env_state is None:
        sys.exit("dump has no env_state; rerun dump_isaaclab_step0.py with "
                 "the env state patch enabled (it should be by default).")

    joint_pos_il = env_state["joint_pos"][args.env_idx].cpu().numpy().astype(np.float64)
    joint_vel_il = env_state["joint_vel"][args.env_idx].cpu().numpy().astype(np.float64)
    root_quat_wxyz = env_state["root_quat_w_wxyz"][args.env_idx].cpu().numpy().astype(np.float64)
    root_ang_vel_b = env_state["root_ang_vel_b"][args.env_idx].cpu().numpy().astype(np.float64)

    motion_id = int(env_state["motion_ids"][args.env_idx].item())
    # motion_times is optional in the dump: TrackingCommand doesn't expose it
    # directly, so dump_isaaclab_step0.py reconstructs it from
    # motion_start_time_steps + time_steps when those exist. If neither is
    # available, fall back to t=0 (the start of the clip), which is
    # accurate when RSI is disabled in eval. See dump_isaaclab_step0.py for
    # the upstream capture logic.
    if "motion_times" in env_state:
        motion_t = float(env_state["motion_times"][args.env_idx].item())
        motion_t_src = "env_state[motion_times]"
    else:
        motion_t = 0.0
        motion_t_src = "fallback (no motion_times in dump; assuming t=0)"
    print(f"\nIL env_state: motion_id={motion_id}  motion_t={motion_t:.4f}s  "
          f"({motion_t_src})  root_quat(wxyz)={root_quat_wxyz}")

    motion_path = args.motion
    if motion_path is None:
        # eval_agent_trl loads motions from a directory; we don't replicate
        # the resolution, just nudge the user.
        sys.exit("--motion not provided. Re-run with the same motion .pkl "
                 "the IsaacLab eval was using (look for "
                 "`motion_command.motion_path` in dump's hydra config or "
                 "the eval_agent_trl logs).")

    import joblib
    motion_data = joblib.load(motion_path)
    first = motion_data[list(motion_data.keys())[0]]
    motion_fps = float(first.get("fps", 30.0))

    # Rebuild tokenizer obs using MuJoCo eval pipeline.
    tok_mj = build_tokenizer_obs(motion_data, motion_t, root_quat_wxyz, motion_fps)

    # Rebuild proprioception. IsaacLab's CircularBuffer is broadcast-primed
    # on the first sample, so for step 0 the 10-frame history is just the
    # current obs replicated.
    grav_b = quat_rotate_inverse(root_quat_wxyz, np.array([0.0, 0.0, -1.0]))
    jpos_rel_il = joint_pos_il - DEFAULT_DOF[IL_TO_MJ_DOF]
    last_action_il = np.zeros(NUM_DOFS, dtype=np.float32)
    pb = ProprioceptionBuffer()
    pb.append(grav_b, root_ang_vel_b, jpos_rel_il, joint_vel_il, last_action_il)
    prop_mj = pb.get_flat()

    # Convert the MuJoCo tokenizer (PT-interleaved) to ONNX-grouped to
    # match what IsaacLab feeds the fused graph. eval_x2_mujoco_onnx does
    # the same conversion inside OnnxActor.__call__.
    from gear_sonic.scripts.eval_x2_mujoco_onnx import _interleaved_to_grouped
    tok_mj_onnx = _interleaved_to_grouped(tok_mj.astype(np.float32))
    actor_obs_mj = np.concatenate([tok_mj_onnx, prop_mj.astype(np.float32)])

    print("\n=== Slot-by-slot diff (IsaacLab vs MuJoCo-eval rebuild) ===")
    print("    OK = within configured AdditiveUniformNoiseCfg envelope; GAP = real obs gap")
    # Tokenizer: 620 cmd + 60 ori. Tokenizer noise lives in
    # observations/tokenizer/*.yaml; we don't annotate it here because (a)
    # the tokenizer block diffs huge unless --motion matches the IsaacLab
    # eval's loaded motion exactly, which dwarfs any obs-noise envelope,
    # and (b) the per-term layout there isn't 1:1 with global.yaml.
    _block_diff("tok cmd_flat", actor_obs_il[:620], actor_obs_mj[:620])
    _block_diff("tok ori_flat", actor_obs_il[620:680], actor_obs_mj[620:680])
    # Proprioception term-by-term (matches PolicyCfg attribute order):
    #   base_ang_vel(30), joint_pos_rel(310), joint_vel(310),
    #   last_action(310), gravity_dir(30)
    p0, p1 = 680, 680 + 30
    _block_diff("prop base_ang_vel(30)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1],
                noise_envelope=PROP_NOISE_ENVELOPE["base_ang_vel"])
    p0, p1 = p1, p1 + 310
    _block_diff("prop joint_pos_rel(310)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1],
                noise_envelope=PROP_NOISE_ENVELOPE["joint_pos_rel"])
    p0, p1 = p1, p1 + 310
    _block_diff("prop joint_vel(310)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1],
                noise_envelope=PROP_NOISE_ENVELOPE["joint_vel"])
    p0, p1 = p1, p1 + 310
    _block_diff("prop last_action(310)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1],
                noise_envelope=PROP_NOISE_ENVELOPE["last_action"])
    p0, p1 = p1, p1 + 30
    _block_diff("prop gravity_dir(30)", actor_obs_il[p0:p1], actor_obs_mj[p0:p1],
                noise_envelope=PROP_NOISE_ENVELOPE["gravity_dir"])

    # Run the ONNX on BOTH vectors and print the action norms. The obs-diff
    # block above is the actual sim-sim parity signal; this block is only a
    # sanity check that says "given the same obs, does the ONNX produce the
    # same action as IsaacLab's .pt did?" Skip it cleanly when the operator
    # didn't pass --model -- the obs diff is what they actually came for.
    if args.model is None:
        print("\n=== ONNX action check skipped (no --model passed) ===")
        print("  Pass --model PATH.onnx to also run the ONNX on both obs "
              "vectors and compare against dumped IL decoder_action_mean.")
    else:
        actor = OnnxActor(str(args.model))
        out_il = actor.session.run(
            [actor.output_name], {actor.input_name: actor_obs_il.reshape(1, -1)}
        )[0][0]
        out_mj = actor.session.run(
            [actor.output_name], {actor.input_name: actor_obs_mj.reshape(1, -1)}
        )[0][0]
        print("\n=== ONNX action on each vector ===")
        print(f"  on IsaacLab obs:  inf={np.max(np.abs(out_il)):.4f}  L2={np.linalg.norm(out_il):.4f}")
        print(f"  on MuJoCo  obs:  inf={np.max(np.abs(out_mj)):.4f}  L2={np.linalg.norm(out_mj):.4f}")
        print(f"  diff(IL action vs MJ action): inf={np.max(np.abs(out_il-out_mj)):.4e}")
        print(f"  vs dumped IL decoder_action_mean: "
              f"inf={np.max(np.abs(out_il - dump['decoder_action_mean'][args.env_idx].cpu().numpy())):.4e}")


if __name__ == "__main__":
    main()
