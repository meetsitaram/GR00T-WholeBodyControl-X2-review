"""Bake a 1-pose PKL the X2 deploy bridge can RSI from in ``--vla`` mode
when the kplanner is the locomotion source.

Why this exists
---------------

The deploy supports two MuJoCo spawn profiles:

  - ``--sim-profile manual``: the pelvis spawns at z=0.85 m on an
    elastic band that auto-releases after ~2 s. If no pose-ref stream
    arrives in that window the robot literally launches from the air
    and collapses.
  - ``--sim-profile parity --motion <pkl>``: the bridge RSIs from
    frame 0 of the PKL. The robot spawns on the floor in the exact
    pose the C++ deploy is about to track. No band, no drop.

The heuristic planner ships
``gear_sonic/scripts/bake_planner_rsi_anchor.py`` which derives the
anchor from ``planner.current_anchor_frame()`` (i.e. ``idle_stand[0]``
from the curated primitives PKL). The kplanner has no primitives PKL,
but it does have a well-defined initial pose: the **warmup quiet-stand
qpos** it publishes during the first ``--warmup-quiet-stand-s`` seconds
before switching to neural inference.

This script writes that PKL by:
  1. Loading the kplanner's 38-D warmup qpos -- either the
     hand-crafted ``_build_default_warmup_qpos()`` baked into the
     daemon, or a user-supplied PKL via ``--warmup-qpos-path``.
  2. Converting the MuJoCo wxyz quaternion convention to the
     deploy-PKL's xyzw convention and splitting into the schema the
     deploy bridge expects (``dof`` / ``root_rot`` / ``root_trans_offset``
     / ``fps``).
  3. Holding for 5 identical frames (bridge computes velocity by
     finite differences and needs >=2 frames to produce zero velocity).

The resulting PKL is byte-identical to what the kplanner's first
publish tick emits when ``warmup_quiet_stand_s > 0``, so the deploy's
tracker sees zero error on tick 0 and stays upright through the
~5 s neural-model cold-start window.

Usage::

    .venv/bin/python -m gear_sonic.scripts.bake_kplanner_rsi_anchor

(``run_x2_quest3_planner_stack.sh --planner kplanner --sim-profile parity``
does this automatically when the anchor PKL is absent.)

Pass it to the deploy as::

    bash deploy_x2.sh sim --vla --sim-profile parity \\
        --motion <baked_pkl> --model <onnx>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Re-use the daemon's loader so the default warmup qpos stays a single
# source of truth across the daemon and the bake step. This is an
# import-only call -- no torch / motionbricks heavy weight loaded.
from gear_sonic.scripts.x2_kplanner import (  # noqa: E402
    _build_default_warmup_qpos,
    _load_warmup_qpos,
)

_DEFAULT_OUT = (
    _REPO_ROOT
    / "data"
    / "sim_to_real_anchors"
    / "browse_sonic"
    / "baked_pkls"
    / "x2_kplanner_rsi_anchor.pkl"
)
# Bridge's compute_motion_state computes velocity by finite differences,
# so we need >= 2 identical frames; 5 gives a comfortable margin and
# guarantees ang_vel / lin_vel both come out as exact zero.
_N_HOLD_FRAMES = 5
_FPS = 50.0


def bake(
    out_path: Path = _DEFAULT_OUT,
    warmup_qpos_path: Path | None = None,
) -> dict:
    """Bake a 5-frame parity PKL from the kplanner's warmup qpos.

    Args:
        out_path: Destination PKL.
        warmup_qpos_path: Optional path to a stand-pose PKL. If ``None``
            (or the file doesn't exist), the hand-crafted default in
            ``x2_kplanner._build_default_warmup_qpos()`` is used.
    """
    if warmup_qpos_path is None:
        # The daemon's default is training_default_angles -- matches
        # the deploy's SAFE_IDLE PD target so the bridge RSI to this
        # pose produces zero joint-target delta during the deploy's
        # cold-start pose-ref starvation window.
        qpos = _build_default_warmup_qpos().astype(np.float64)
        source = (
            "training_default_angles (policy_parameters.hpp; "
            "matches deploy SAFE_IDLE PD target)"
        )
    else:
        # ``_load_warmup_qpos`` handles array / qpos-dict / deploy-PKL
        # schemas and falls back to the resolved default on missing-file.
        qpos = _load_warmup_qpos(warmup_qpos_path).astype(np.float64)
        source = (
            str(warmup_qpos_path)
            if warmup_qpos_path.is_file()
            else f"resolved default (warmup PKL {warmup_qpos_path} missing)"
        )
    if qpos.shape != (38,):
        raise ValueError(f"expected qpos[38], got shape {qpos.shape}")

    trans_anchor = qpos[0:3].copy()
    # MuJoCo qpos quat is ``[w, x, y, z]``; the deploy-PKL schema (and
    # the heuristic anchor PKL) uses ``[x, y, z, w]``. Reorder rather
    # than re-normalize so we don't smuggle drift into a "stationary"
    # PKL.
    w, x, y, z = qpos[3], qpos[4], qpos[5], qpos[6]
    rot_anchor = np.array([x, y, z, w], dtype=np.float64)
    dof_anchor = qpos[7:38].copy()

    dof = np.broadcast_to(dof_anchor, (_N_HOLD_FRAMES, dof_anchor.shape[0])).copy()
    rot = np.broadcast_to(rot_anchor, (_N_HOLD_FRAMES, 4)).copy()
    trans = np.broadcast_to(trans_anchor, (_N_HOLD_FRAMES, 3)).copy()

    name = "kplanner_rsi_anchor"
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
        "source": source,
        "anchor_quat_xyzw": rot_anchor.tolist(),
        "anchor_trans": trans_anchor.tolist(),
        "anchor_dof_first3": dof_anchor[:3].tolist(),
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output PKL path (default: %(default)s).",
    )
    p.add_argument(
        "--warmup-qpos-path",
        type=Path,
        default=None,
        help=(
            "Optional PKL with a stand qpos[38] (or [T, 38] -> frame 0). "
            "Must match the value passed to ``x2_kplanner --warmup-qpos-path`` "
            "so the daemon's first tick is byte-identical to the RSI pose. "
            "If omitted, the hand-crafted default is used "
            "(zero joints, hip_h=0.95m, identity quaternion)."
        ),
    )
    args = p.parse_args()

    info = bake(args.out, args.warmup_qpos_path)
    print(f"[kplanner-anchor] wrote {info['out_path']}")
    print(f"                  source       = {info['source']}")
    print(f"                  n_frames     = {info['n_frames']}")
    print(f"                  fps          = {info['fps']}")
    print(
        f"                  anchor quat  = {info['anchor_quat_xyzw']}  "
        f"(xyzw; mujoco wxyz reordered)"
    )
    print(
        f"                  anchor trans = {info['anchor_trans']}  "
        f"(root xyz)"
    )
    print(
        f"                  dof[0:3]     = {info['anchor_dof_first3']}  "
        f"(first 3 of 31 joint angles)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
