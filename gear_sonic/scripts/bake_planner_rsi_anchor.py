"""Bake a 1-pose PKL the X2 deploy bridge can RSI from in --vla mode.

Why this exists
---------------

``deploy_x2.sh sim --motion <pkl>`` works perfectly because the parity
profile RSIs the bridge from the motion file's frame 0 (band off,
ramp 0, autostart 0). The robot spawns on the floor in the exact pose
the C++ deploy is about to track.

``deploy_x2.sh sim --vla`` (used by the planner smoke test) does NOT
get the same treatment by default: the manual profile spawns the
pelvis at z=0.85 m on an elastic band, which lets the robot launch
from the air for ~2 s before the band releases.

The fix is to keep using parity profile in the planner case too --
parity supports --vla (the C++ deploy ignores --motion in VLA mode but
the bridge still uses MOTION_SOURCE for RSI). What's missing is a
"stand pose" PKL whose frame 0 is bit-identical to what the planner's
state machine emits on its first tick. Without a match, the policy's
tracker error term sees a yaw / joint mismatch on tick 0 and yanks the
body to correct it.

This script writes that PKL by:
  1. Constructing a ``HeuristicPlanner`` from the runtime primitives.
  2. Pulling its canonical anchor frame
     (``planner.current_anchor_frame()``), which is the aligned
     ``idle_stand[0]`` pose at (xy=(0,0), yaw=0, z=clip's frame-0 Z).
  3. Writing it as a 5-frame PKL (5 copies of the same pose, so the
     bridge's velocity-by-finite-differences math stays at zero) in
     the deploy-PKL schema (``dof``, ``root_rot``,
     ``root_trans_offset``, ``fps``).

Usage::

    .venv/bin/python -m gear_sonic.scripts.bake_planner_rsi_anchor

prints the absolute path of the baked PKL. Pass it to deploy as::

    bash deploy_x2.sh sim --vla --sim-profile parity \\
        --motion <baked_pkl> --model <onnx>

(``run_planner_smoke.sh --with-deploy`` does this automatically when
the anchor PKL is absent.)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from gear_sonic.utils.planner.registry import load_bin_specs
from gear_sonic.utils.planner.state_machine import (
    HeuristicPlanner,
    load_primitives_pkl,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PRIMS_PKL = (
    _REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_primitives.pkl"
)
_DEFAULT_BINS_YAML = (
    _REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_bins.yaml"
)
_DEFAULT_OUT = (
    _REPO_ROOT
    / "data"
    / "sim_to_real_anchors"
    / "browse_sonic"
    / "baked_pkls"
    / "x2_planner_rsi_anchor.pkl"
)
# Bridge's compute_motion_state computes velocity by finite differences,
# so we need >= 2 identical frames; 5 gives a comfortable margin and
# guarantees ang_vel / lin_vel both come out as exact zero.
_N_HOLD_FRAMES = 5
_FPS = 50.0


def bake(
    prims_pkl: Path = _DEFAULT_PRIMS_PKL,
    bins_yaml: Path = _DEFAULT_BINS_YAML,
    out_path: Path = _DEFAULT_OUT,
) -> dict:
    bin_specs = load_bin_specs(bins_yaml)
    bin_family = {name: spec.family for name, spec in bin_specs.items()}
    primitives = load_primitives_pkl(prims_pkl, bin_family)

    planner = HeuristicPlanner(primitives=primitives)
    anchor = planner.current_anchor_frame()

    dof_anchor = anchor.joint_pos_mj.astype(np.float64)
    rot_anchor = anchor.root_quat_xyzw.astype(np.float64)
    # current_anchor_frame() doesn't carry root_z (it stores xy_world only),
    # but the active segment's aligned_trans does -- read it directly so the
    # bridge spawns at the clip's natural pelvis Z, not a guess.
    z_anchor = float(planner._active.aligned_trans[0, 2])  # noqa: SLF001
    trans_anchor = np.array([0.0, 0.0, z_anchor], dtype=np.float64)

    dof = np.broadcast_to(dof_anchor, (_N_HOLD_FRAMES, dof_anchor.shape[0])).copy()
    rot = np.broadcast_to(rot_anchor, (_N_HOLD_FRAMES, 4)).copy()
    trans = np.broadcast_to(trans_anchor, (_N_HOLD_FRAMES, 3)).copy()

    name = "planner_rsi_anchor"
    payload = {
        name: {
            "dof": dof,
            "root_rot": rot,
            "root_trans_offset": trans,
            "fps": _FPS,
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, out_path)

    return {
        "out_path": str(out_path),
        "n_frames": _N_HOLD_FRAMES,
        "fps": _FPS,
        "anchor_dof_first3": dof_anchor[:3].tolist(),
        "anchor_quat_xyzw": rot_anchor.tolist(),
        "anchor_trans": trans_anchor.tolist(),
        "source_bin": anchor.bin_name,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--primitives-pkl", type=Path, default=_DEFAULT_PRIMS_PKL)
    p.add_argument("--bins-yaml", type=Path, default=_DEFAULT_BINS_YAML)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args()

    info = bake(args.primitives_pkl, args.bins_yaml, args.out)
    print(f"[anchor] wrote {info['out_path']}")
    print(f"         source bin   = {info['source_bin']}")
    print(f"         n_frames     = {info['n_frames']}")
    print(f"         fps          = {info['fps']}")
    print(
        f"         anchor quat  = {info['anchor_quat_xyzw']}  "
        f"(identity yaw, pitch/roll preserved from clip)"
    )
    print(
        f"         anchor trans = {info['anchor_trans']}  "
        f"(pelvis_z from clip frame 0)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
