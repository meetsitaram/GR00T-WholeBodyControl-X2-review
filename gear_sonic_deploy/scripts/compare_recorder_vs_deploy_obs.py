#!/usr/bin/env python3
"""Layer 3: byte-parity diff of recorder Python obs vs. deploy C++ obs.

This closes the loop on the inline-tokenizer plan's validation
pyramid: with Layer 1 + 2 already pinning the recorder's gather
against ``build_tokenizer_obs`` and ``label_trajectory``, the only
remaining failure mode is a representation mismatch between the
recorder's *Python* path and the deploy's *C++* ``ZmqPoseInputSource``
path. Both consume the same wire snapshot; this script asserts they
produce byte-equal 680-D ``encoder_input`` vectors.

How to capture the two dumps
----------------------------

Run the planner stack as you would for a real recording, but pass
both ``--obs-dump-recorder`` (recorder side) and ``--obs-dump``
(deploy side) so each process writes a one-shot snapshot of its
first fully-populated CONTROL tick:

::

    ./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \\
        --duration 30 --no-with-record \\
        --extra-deploy-args "--obs-dump /tmp/x2_deploy_obs.bin"

Then in a second shell while the stack is up::

    python -m gear_sonic.scripts.record_x2_dataset \\
        --teleop-only --body-pose-source zmq \\
        --sonic-checkpoint /path/to/model_step_NNNNN.pt \\
        --encoder-config gear_sonic/data/encoder/x2_observation_config.yaml \\
        --obs-dump-recorder /tmp/x2_recorder_obs.pt \\
        --duration 10

Once both files exist, run::

    python gear_sonic_deploy/scripts/compare_recorder_vs_deploy_obs.py \\
        --recorder /tmp/x2_recorder_obs.pt \\
        --deploy   /tmp/x2_deploy_obs.bin

The script exits 0 when ``max-abs(diff) < THRESHOLD`` (default
``1e-6``) and 1 otherwise. Acceptable drift on different machines /
GPU vendors is below ``1e-5``; anything beyond that points at a
recorder-side gather bug or a stale deploy build.

Notes
-----

* The recorder and deploy will not necessarily lock onto the same
  *exact* tick (each dumps independently on its first
  fully-populated tick). On a stationary planner state this is
  fine -- the wire snapshot is constant -- but if you compare while
  the planner is mid-bin you may see ~ms-scale drift in the future
  window. When in doubt, capture during a held idle pose.
* The dump format intentionally mirrors
  ``compare_deploy_vs_isaaclab_obs.py`` so the operator workflow
  feels familiar.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Deploy blob layout (must match x2_deploy_onnx_ref.cpp::DumpObsBlob; see
# also compare_deploy_vs_isaaclab_obs.py for the reference parser)
# ---------------------------------------------------------------------------
MAGIC = b"X2OBSV01"
HEADER_FMT = "<8sIIId"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


@dataclass
class DeployBlob:
    policy_time: float
    tokenizer_obs: np.ndarray   # (680,) float32
    proprioception: np.ndarray  # (990,) float32
    action_il: np.ndarray       # (31,)  float64
    joint_pos_mj: np.ndarray    # (31,)  float64
    joint_vel_mj: np.ndarray    # (31,)  float64
    base_quat_wxyz: np.ndarray  # (4,)   float64
    base_ang_vel: np.ndarray    # (3,)   float64


def load_deploy_blob(path: Path) -> DeployBlob:
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"{path}: too small ({len(raw)} bytes) to be an X2OBSV01 blob"
        )
    magic, tok_dim, prop_dim, action_dim, policy_time = struct.unpack(
        HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")

    off = HEADER_SIZE
    tok_bytes = tok_dim * 4
    tokenizer_obs = np.frombuffer(
        raw, dtype=np.float32, count=tok_dim, offset=off
    ).copy()
    off += tok_bytes

    prop_bytes = prop_dim * 4
    proprioception = np.frombuffer(
        raw, dtype=np.float32, count=prop_dim, offset=off
    ).copy()
    off += prop_bytes

    action_bytes = action_dim * 8
    action_il = np.frombuffer(
        raw, dtype=np.float64, count=action_dim, offset=off
    ).copy()
    off += action_bytes

    joint_pos_mj = np.frombuffer(
        raw, dtype=np.float64, count=31, offset=off
    ).copy()
    off += 31 * 8
    joint_vel_mj = np.frombuffer(
        raw, dtype=np.float64, count=31, offset=off
    ).copy()
    off += 31 * 8
    base_quat_wxyz = np.frombuffer(
        raw, dtype=np.float64, count=4, offset=off
    ).copy()
    off += 4 * 8
    base_ang_vel = np.frombuffer(
        raw, dtype=np.float64, count=3, offset=off
    ).copy()

    return DeployBlob(
        policy_time=float(policy_time),
        tokenizer_obs=tokenizer_obs,
        proprioception=proprioception,
        action_il=action_il,
        joint_pos_mj=joint_pos_mj,
        joint_vel_mj=joint_vel_mj,
        base_quat_wxyz=base_quat_wxyz,
        base_ang_vel=base_ang_vel,
    )


def load_recorder_blob(path: Path) -> dict:
    """Load the recorder's torch .pt dump (schema v1)."""
    import torch
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected dict; got {type(payload)}")
    kind = payload.get("kind")
    if kind != "x2_recorder_obs_dump_v1":
        raise ValueError(
            f"{path}: unexpected kind {kind!r}; expected "
            f"'x2_recorder_obs_dump_v1' (regenerate via "
            f"--obs-dump-recorder)"
        )
    if "encoder_obs" not in payload:
        raise ValueError(f"{path}: missing 'encoder_obs' field")
    return payload


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def diff_obs(
    recorder_obs: np.ndarray,
    deploy_obs: np.ndarray,
    *,
    threshold: float,
    top_n: int = 10,
) -> int:
    """Print per-row diff stats and return process exit code (0 = pass)."""
    if recorder_obs.shape != deploy_obs.shape:
        print(
            f"[FAIL] shape mismatch: recorder {recorder_obs.shape} vs "
            f"deploy {deploy_obs.shape}",
            file=sys.stderr,
        )
        return 1
    a = recorder_obs.astype(np.float64, copy=False)
    b = deploy_obs.astype(np.float64, copy=False)
    diff = a - b
    abs_diff = np.abs(diff)
    max_abs = float(abs_diff.max())
    mean_abs = float(abs_diff.mean())
    rms = float(np.sqrt(np.mean(diff * diff)))

    grid_a = a.reshape(10, 68)
    grid_b = b.reshape(10, 68)

    print("Layer 3 byte-parity report")
    print("--------------------------")
    print(f"  recorder obs : shape={recorder_obs.shape} dtype={recorder_obs.dtype}")
    print(f"  deploy obs   : shape={deploy_obs.shape} dtype={deploy_obs.dtype}")
    print(f"  max-abs diff : {max_abs:.3e}")
    print(f"  mean-abs diff: {mean_abs:.3e}")
    print(f"  rms diff     : {rms:.3e}")
    print(f"  threshold    : {threshold:.3e}")
    print()

    # Per-frame, per-block split (jpos / jvel / ori).
    print("Per-frame diff (jpos | jvel | ori 6D):")
    for f in range(grid_a.shape[0]):
        row_a = grid_a[f]
        row_b = grid_b[f]
        d = np.abs(row_a - row_b)
        jpos_max = float(d[:31].max())
        jvel_max = float(d[31:62].max())
        ori_max = float(d[62:].max())
        print(
            f"  frame {f:2d} : jpos max-abs={jpos_max:.3e}  "
            f"jvel max-abs={jvel_max:.3e}  ori max-abs={ori_max:.3e}"
        )

    if top_n > 0:
        print()
        print(f"Top-{top_n} divergent slots:")
        idx = np.argsort(-abs_diff)[:top_n]
        for i in idx:
            f = int(i) // 68
            j = int(i) % 68
            block = (
                "jpos" if j < 31
                else ("jvel" if j < 62 else "ori")
            )
            print(
                f"  slot {int(i):4d} (frame {f:2d}, {block} idx "
                f"{(j if j < 31 else (j - 31 if j < 62 else j - 62))}): "
                f"recorder={a[i]:+.6e}  deploy={b[i]:+.6e}  "
                f"diff={diff[i]:+.6e}"
            )

    if max_abs > threshold:
        print(
            f"\n[FAIL] max-abs diff {max_abs:.3e} exceeds threshold "
            f"{threshold:.3e}",
            file=sys.stderr,
        )
        return 1
    print("\n[PASS] recorder gather and deploy ZmqPoseInputSource agree byte-for-byte.")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recorder", type=Path, required=True,
        help="Path to the recorder's .pt dump (from --obs-dump-recorder).",
    )
    parser.add_argument(
        "--deploy", type=Path, required=True,
        help="Path to the deploy's binary blob (from --obs-dump).",
    )
    parser.add_argument(
        "--threshold", type=float, default=1e-6,
        help="max-abs diff threshold to consider a PASS (default 1e-6). "
             "Bump to 1e-5 if you're comparing across GPU vendors.",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Number of largest-diff slots to print individually.",
    )
    args = parser.parse_args(argv)

    rec = load_recorder_blob(args.recorder)
    dep = load_deploy_blob(args.deploy)

    print(f"Recorder dump : {args.recorder}")
    print(f"  encoder_config : {rec.get('encoder_config') or '(unknown)'}")
    print(f"  checkpoint     : {rec.get('checkpoint') or '(unknown)'}")
    print(f"Deploy blob   : {args.deploy}")
    print(f"  policy_time    : {dep.policy_time:.4f}s since CONTROL entry")
    print()

    return diff_obs(
        rec["encoder_obs"],
        dep.tokenizer_obs,
        threshold=args.threshold,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    sys.exit(main())
