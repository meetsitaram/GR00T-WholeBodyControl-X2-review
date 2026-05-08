#!/usr/bin/env python3
"""Standalone ONNX parity check for the X2 sonic policy.

Builds a 1670-D actor_obs vector for a perfectly-standing robot using the
SAME Python machinery `eval_x2_mujoco_onnx.py` uses against MuJoCo, and runs
the ONNX session on it. Then prints the action vector.

Why this exists
---------------
A real-robot dry-run with the deploy node showed that the policy commands
30 of 31 joints to deviate by exactly the safety clamp (0.05 rad) from
default, every single tick. That is an obs-construction red flag, not a
"the policy is twitchy" signal. The two possible causes are:

  (a) the trained policy itself emits saturated actions even for ideal
      stand-still inputs (broken / overfit / undertrained model)
  (b) the C++ deploy node feeds the policy obs that don't match what
      eval_x2_mujoco_onnx.py would feed (layout, sign, scale)

This script tests (a): if Python's known-good pipeline ALSO produces a
saturated action for the same conceptual stand-still input, then the model
file is the problem and no amount of C++ fiddling will help. If Python
produces a small action, then (b) is the cause and we should byte-compare
the two pipelines slot by slot.

Run from the repo root:

    python gear_sonic_deploy/scripts/parity_check_onnx_standstill.py \\
        --model gear_sonic_deploy/models/x2_sonic_16k.onnx
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Re-use the Python ground-truth obs builders.
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
)
from gear_sonic.scripts.eval_x2_mujoco_onnx import (  # noqa: E402
    OnnxActor,
    PROP_DIM,
    TOK_DIM,
)


def make_standstill_motion_data():
    """Fake motion_data dict with a single repeated frame at the default pose.

    `build_tokenizer_obs` indexes into m["dof"] and m["root_rot"] at integer
    frame indices; we just need enough frames for the 10-future-frame window
    at fps=10 -> 1.0 s lookahead = 10 frames is the minimum that won't get
    clipped to the last frame.
    """
    fps = 10.0
    n_frames = NUM_FUTURE_FRAMES + 2
    # default_dof is in MuJoCo order; motion_data uses MuJoCo order too.
    dof = np.tile(DEFAULT_DOF, (n_frames, 1)).astype(np.float64)
    # root_rot is xyzw (scipy convention) -- identity quaternion.
    root_rot = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (n_frames, 1))
    # _m(data) does data[list(data.keys())[0]] so wrap one level.
    return {"motion_0": {"dof": dof, "root_rot": root_rot}}, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--clamp", type=float, default=0.05,
                    help="reference safety clamp in rad (for the verdict only)")
    ap.add_argument("--motion", type=Path, default=None,
                    help="optional real motion .pkl to test against instead "
                         "of synthetic stand-still")
    ap.add_argument("--init-frame", type=int, default=0,
                    help="frame index for RSI when --motion is set")
    args = ap.parse_args()

    if not args.model.exists():
        ap.error(f"Model not found: {args.model}")

    actor = OnnxActor(str(args.model))
    print(actor.describe())

    if args.motion is not None:
        import joblib
        motion_data = joblib.load(args.motion)
        # match eval_x2_mujoco.get_motion_fps default
        first = motion_data[list(motion_data.keys())[0]]
        fps = float(first.get("fps", 30.0)) if isinstance(first, dict) else 30.0
        # RSI: snap base_quat to the motion's frame[init_frame] so future quats
        # are relative to the actual init pose (not identity).
        rr = first["root_rot"][args.init_frame]  # xyzw
        base_quat_wxyz = np.array([rr[3], rr[0], rr[1], rr[2]])
        current_time = args.init_frame / fps
        print(f"Loaded motion: {args.motion.name}  fps={fps:.1f}  "
              f"frames={first['root_rot'].shape[0]}  init_frame={args.init_frame}")
    else:
        motion_data, fps = make_standstill_motion_data()
        base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        current_time = 0.0

    # Build tokenizer obs for stand-still at t=0 with identity orientation.
    # Both `cur` and every future frame have identity orientation -> ori
    # block should be exactly 10x [1, 0, 0, 0, 1, 0]. jpos block should be
    # 10x default-IL (since DEFAULT_DOF is MuJoCo-ordered, IL_TO_MJ_DOF
    # picks the right ones). jvel block should be 10x zeros.
    tok_interleaved = build_tokenizer_obs(motion_data, current_time, base_quat_wxyz, fps)
    assert tok_interleaved.shape == (TOK_DIM,), tok_interleaved.shape

    # Build proprioception for stand-still: at the default pose, all jpos_rel
    # = 0, jvel = 0, last_action = 0, ang_vel = 0, gravity_body = [0, 0, -1]
    # (because cur quat = identity).
    prop_buf = ProprioceptionBuffer()
    zeros_dof = np.zeros(NUM_DOFS, dtype=np.float32)
    zeros_3 = np.zeros(3, dtype=np.float32)
    grav_body = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    # ProprioceptionBuffer.append signature: (gravity, angvel, jpos_rel, jvel, action)
    prop_buf.append(grav_body, zeros_3, zeros_dof, zeros_dof, zeros_dof)
    prop = prop_buf.get_flat()
    assert prop.shape == (PROP_DIM,), prop.shape

    # Run ONNX through the SAME wrapper the eval script uses (it owns the
    # interleaved->grouped conversion + the [tok | prop] concat).
    action_il = actor(prop, tok_interleaved)
    assert action_il.shape == (NUM_DOFS,), action_il.shape

    # Convert to MuJoCo-ordered target_pos like the deploy does:
    #   target_mj[i] = default[i] + action_il[mj_to_il[i]] * action_scale[i]
    # (we don't need it for the verdict, but it lets us print "would the
    # safety clamp have fired?" the same way the C++ deploy does.)
    from gear_sonic.scripts.eval_x2_mujoco import ACTION_SCALE
    target_mj = DEFAULT_DOF + np.array(
        [action_il[MJ_TO_IL_DOF[i]] * ACTION_SCALE[i] for i in range(NUM_DOFS)]
    )
    dev_mj = target_mj - DEFAULT_DOF

    print("\n=== Python-pipeline ONNX action for perfect stand-still ===\n")
    print(f"action_il (raw network output, IL order):")
    print(f"  inf-norm     = {np.max(np.abs(action_il)):.4f}")
    print(f"  L2-norm      = {np.linalg.norm(action_il):.4f}")
    print(f"  mean abs     = {np.mean(np.abs(action_il)):.4f}")
    print(f"  first 8      = {action_il[:8]}")

    print(f"\ntarget_pos_mj - default_angles (after action_scale, MJ order):")
    print(f"  inf-norm     = {np.max(np.abs(dev_mj)):.4f} rad "
          f"({np.max(np.abs(dev_mj)) * 180 / np.pi:.2f} deg)")
    print(f"  L2-norm      = {np.linalg.norm(dev_mj):.4f}")

    n_over = int(np.sum(np.abs(dev_mj) > args.clamp))
    print(f"\n{n_over}/{NUM_DOFS} joints would exceed the {args.clamp} rad clamp.")
    if n_over >= 25:
        print("\n>>> Verdict: Python's known-good pipeline ALSO saturates the clamp.")
        print(">>> The ONNX file itself produces wild actions for ideal stand-still")
        print(">>> input. This is a model/training problem, not a C++ deploy bug.")
    elif n_over <= 2:
        print("\n>>> Verdict: Python pipeline produces a SMALL action (within clamp).")
        print(">>> The ONNX is healthy on this input. The C++ deploy is feeding it")
        print(">>> something different. Next step: byte-by-byte slot compare.")
    else:
        print("\n>>> Verdict: Mixed -- some joints clamp-saturated, some not.")
        print(">>> Could be either a partial obs bug or a borderline policy.")


if __name__ == "__main__":
    main()
