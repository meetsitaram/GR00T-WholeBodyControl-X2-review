#!/usr/bin/env python3
"""Diff a C++ deploy obs blob against IsaacLab GT, slot by slot.

The C++ deploy node, when run with ``--obs-dump PATH``, captures one full
inference payload from the first CONTROL tick:

    char[8]  magic         = "X2OBSV01"
    uint32   tok_dim       = 680
    uint32   prop_dim      = 990
    uint32   action_dim    = 31
    float64  policy_time   = (seconds since CONTROL entry)
    float32  tokenizer_obs[tok_dim]
    float32  proprioception[prop_dim]
    float64  action_il[action_dim]
    float64  joint_pos_mj[31]
    float64  joint_vel_mj[31]
    float64  base_quat_wxyz[4]
    float64  base_ang_vel[3]

Total: 8 + 12 + 8 + (680+990)*4 + (31+31+31+4+3)*8 = 7508 bytes.

This script reads that blob and an IsaacLab dump produced by
``gear_sonic.scripts.dump_isaaclab_step0`` (default
/tmp/x2_step0_isaaclab_lastpt.pt) and compares them slot by slot.

The IsaacLab dump is captured from a *different* robot pose than the C++
deploy will see on the gantry, so several slots are EXPECTED to differ:

    * proprioception.base_ang_vel  (depends on robot)
    * proprioception.joint_pos_rel (depends on robot)
    * proprioception.joint_vel     (depends on robot)
    * proprioception.gravity_dir   (depends on IMU orientation)

Slots that should be IDENTICAL regardless of robot pose, given that both
runs use the same StandStill reference motion:

    * tokenizer.command_multi_future_nonflat   (all 10 frames are the
      default standing pose, in MOTION ANCHOR FRAME -- depends on the
      anchor quat, so technically pose-dependent for non-StandStill but
      *should* be near-zero on StandStill regardless)
    * tokenizer.motion_anchor_ori_b_mf_nonflat (anchor-frame ori --
      should be identity rotation flat-6 for StandStill)
    * proprioception.last_action               (zeros on the very first
      tick after CONTROL entry, by construction)

Any divergence in those state-invariant slots is a smoking gun for an
obs-construction bug in the C++ deploy.

The state-dependent slots are still useful as a sanity check: their
magnitudes should match the gantry pose (joint_pos_rel small if joints
are near default, gravity_dir close to IMU's actual world-down vector,
etc).

Usage:
    python gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py \\
        --deploy /workspace/sonic/logs/x2/obs_dump.bin \\
        [--isaaclab /tmp/x2_step0_isaaclab_lastpt.pt] \\
        [--top-n 8]
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Blob layout (must match x2_deploy_onnx_ref.cpp::DumpObsBlob)
# ---------------------------------------------------------------------------
MAGIC = b"X2OBSV01"
HEADER_FMT = "<8sIIId"        # magic, tok_dim, prop_dim, action_dim, policy_time
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
        raise ValueError(f"{path}: too small ({len(raw)} bytes) to be an X2OBSV01 blob")
    magic, tok_dim, prop_dim, action_dim, policy_time = struct.unpack(
        HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"{path}: bad magic {magic!r}, expected {MAGIC!r}")

    expected = (
        HEADER_SIZE
        + tok_dim * 4
        + prop_dim * 4
        + action_dim * 8
        + (31 + 31 + 4 + 3) * 8
    )
    if len(raw) != expected:
        raise ValueError(
            f"{path}: size {len(raw)} != expected {expected} for "
            f"tok={tok_dim} prop={prop_dim} act={action_dim}"
        )

    cursor = HEADER_SIZE
    def take(count: int, dtype) -> np.ndarray:
        nonlocal cursor
        nbytes = count * np.dtype(dtype).itemsize
        arr = np.frombuffer(raw[cursor : cursor + nbytes], dtype=dtype).copy()
        cursor += nbytes
        return arr

    return DeployBlob(
        policy_time    = float(policy_time),
        tokenizer_obs  = take(tok_dim, np.float32),
        proprioception = take(prop_dim, np.float32),
        action_il      = take(action_dim, np.float64),
        joint_pos_mj   = take(31, np.float64),
        joint_vel_mj   = take(31, np.float64),
        base_quat_wxyz = take(4, np.float64),
        base_ang_vel   = take(3, np.float64),
    )


# ---------------------------------------------------------------------------
# IsaacLab dump views
# ---------------------------------------------------------------------------
def load_isaaclab_dump(path: Path) -> dict:
    import torch
    return torch.load(str(path), map_location="cpu", weights_only=False)


def il_proprio_views(dump: dict) -> dict[str, np.ndarray]:
    """Slice the 990-D actor_obs into the 5 per-frame terms × 10 frames.

    Layout (matches ProprioceptionBuffer::GetFlat() in C++):
        base_ang_vel  (3 ) x 10 = 30
        joint_pos_rel (31) x 10 = 310
        joint_vel     (31) x 10 = 310
        last_action   (31) x 10 = 310
        gravity_dir   (3 ) x 10 = 30
    """
    a = dump["proprioception_input"].squeeze().cpu().numpy().astype(np.float32)
    if a.shape != (990,):
        raise ValueError(f"unexpected proprioception_input shape {a.shape}")
    o = 0
    out = {}
    for name, dim in [
        ("base_ang_vel", 3),
        ("joint_pos_rel", 31),
        ("joint_vel", 31),
        ("last_action", 31),
        ("gravity_dir", 3),
    ]:
        block = a[o : o + dim * 10].reshape(10, dim)  # (frames, dim)
        out[name] = block
        o += dim * 10
    assert o == 990
    return out


def deploy_proprio_views(blob: DeployBlob) -> dict[str, np.ndarray]:
    """Same slicing as il_proprio_views but on the deploy blob."""
    a = blob.proprioception
    if a.shape != (990,):
        raise ValueError(f"unexpected deploy proprioception shape {a.shape}")
    o = 0
    out = {}
    for name, dim in [
        ("base_ang_vel", 3),
        ("joint_pos_rel", 31),
        ("joint_vel", 31),
        ("last_action", 31),
        ("gravity_dir", 3),
    ]:
        block = a[o : o + dim * 10].reshape(10, dim)
        out[name] = block
        o += dim * 10
    assert o == 990
    return out


def il_tokenizer_views(dump: dict) -> dict[str, np.ndarray]:
    """Slice the 680-D encoder_input_for_mlp_view into per-frame command/ori.

    Layout (matches BuildTokenizerObs in C++ and dump_isaaclab_step0's
    ``encoder_input_for_mlp_view`` flatten of (10, 68) per-frame interleaved
    cat([command_multi_future_nonflat (62), motion_anchor_ori_b_mf_nonflat (6)],
    dim=-1)):
        per frame f in 0..9:
            command(62) | ori(6)
    """
    a = dump["encoder_input_for_mlp_view"].squeeze().cpu().numpy().astype(np.float32)
    if a.shape != (680,):
        raise ValueError(f"unexpected encoder_input_for_mlp_view shape {a.shape}")
    framed = a.reshape(10, 68)
    return {
        "command_multi_future_nonflat":   framed[:, :62],   # (10, 62)
        "motion_anchor_ori_b_mf_nonflat": framed[:, 62:],   # (10, 6)
    }


def deploy_tokenizer_views(blob: DeployBlob) -> dict[str, np.ndarray]:
    a = blob.tokenizer_obs
    if a.shape != (680,):
        raise ValueError(f"unexpected deploy tokenizer shape {a.shape}")
    framed = a.reshape(10, 68)
    return {
        "command_multi_future_nonflat":   framed[:, :62],
        "motion_anchor_ori_b_mf_nonflat": framed[:, 62:],
    }


# ---------------------------------------------------------------------------
# Diff reporting
# ---------------------------------------------------------------------------
@dataclass
class SlotDiff:
    name: str
    state_invariant: bool   # True = should match exactly regardless of robot pose
    deploy_shape: tuple
    il_shape: tuple
    max_abs: float
    mean_abs: float
    deploy_first: np.ndarray
    il_first: np.ndarray


def report_slot(name: str, state_invariant: bool, deploy: np.ndarray,
                il: np.ndarray, top_n: int = 6) -> SlotDiff:
    if deploy.shape != il.shape:
        return SlotDiff(name, state_invariant, deploy.shape, il.shape,
                        float("inf"), float("inf"),
                        deploy.flatten()[:top_n], il.flatten()[:top_n])
    diff = np.abs(deploy - il)
    return SlotDiff(
        name=name,
        state_invariant=state_invariant,
        deploy_shape=deploy.shape,
        il_shape=il.shape,
        max_abs=float(diff.max()),
        mean_abs=float(diff.mean()),
        deploy_first=deploy.flatten()[:top_n],
        il_first=il.flatten()[:top_n],
    )


def print_slot(d: SlotDiff, threshold: float, top_n: int) -> bool:
    """Print one slot's diff. Returns True if the slot is suspicious."""
    suspicious = (d.state_invariant and d.max_abs > threshold) or (
        d.deploy_shape != d.il_shape
    )
    tag = "INV" if d.state_invariant else "DEP"
    flag = " <-- DIVERGENT" if suspicious else ""
    print(f"  [{tag}] {d.name:<40} dep={d.deploy_shape}  il={d.il_shape}  "
          f"max|Δ|={d.max_abs:.6f}  mean|Δ|={d.mean_abs:.6f}{flag}")
    if suspicious or d.max_abs > 1e-2:
        print(f"        deploy[:{top_n}] = {np.array2string(d.deploy_first, precision=4, suppress_small=True)}")
        print(f"        il    [:{top_n}] = {np.array2string(d.il_first,    precision=4, suppress_small=True)}")
    return suspicious


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diff C++ deploy obs against IsaacLab GT slot-by-slot.")
    ap.add_argument("--deploy", required=True, type=Path,
                    help="Path to the X2OBSV01 binary blob from --obs-dump")
    ap.add_argument("--isaaclab", default=Path("/tmp/x2_step0_isaaclab_lastpt.pt"),
                    type=Path,
                    help="Path to dump_isaaclab_step0 output "
                         "(default: /tmp/x2_step0_isaaclab_lastpt.pt)")
    ap.add_argument("--threshold", default=1e-3, type=float,
                    help="Flag a state-invariant slot as DIVERGENT if its "
                         "max|Δ| exceeds this many radians (default 1e-3)")
    ap.add_argument("--top-n", default=6, type=int,
                    help="Show first N elements of each tensor (default 6)")
    ap.add_argument("--rerun-onnx", type=Path, default=None,
                    help="If given, also run this ONNX on the deploy obs and "
                         "compare its action against the deploy's recorded "
                         "action_il (sanity-checks the C++ ONNX wiring).")
    args = ap.parse_args()

    if not args.deploy.exists():
        print(f"ERROR: deploy blob not found: {args.deploy}", file=sys.stderr)
        return 1
    if not args.isaaclab.exists():
        print(f"ERROR: IsaacLab dump not found: {args.isaaclab}", file=sys.stderr)
        return 1

    blob = load_deploy_blob(args.deploy)
    dump = load_isaaclab_dump(args.isaaclab)

    print(f"=== Deploy blob ===")
    print(f"  path:        {args.deploy}")
    print(f"  policy_time: {blob.policy_time:.4f} s")
    print(f"  tokenizer:   {blob.tokenizer_obs.shape}  "
          f"({blob.tokenizer_obs.dtype})")
    print(f"  proprio:     {blob.proprioception.shape}  "
          f"({blob.proprioception.dtype})")
    print(f"  action_il:   max|a|={np.abs(blob.action_il).max():.4f}  "
          f"mean|a|={np.abs(blob.action_il).mean():.4f}")
    print(f"  joint_pos_mj[:6]: {blob.joint_pos_mj[:6]}")
    print(f"  base_quat_wxyz:   {blob.base_quat_wxyz}")
    print()
    print(f"=== IsaacLab dump ===")
    print(f"  path:                  {args.isaaclab}")
    print(f"  decoder_action_mean[0:6]: "
          f"{dump['decoder_action_mean'].flatten()[:6].numpy()}")
    if "env_state" in dump:
        es = dump["env_state"]
        if "joint_pos" in es:
            print(f"  env_state.joint_pos[0,:6]: "
                  f"{es['joint_pos'][0, :6].numpy()}")
        if "root_quat_w_wxyz" in es:
            print(f"  env_state.root_quat_w_wxyz[0]: "
                  f"{es['root_quat_w_wxyz'][0].numpy()}")
    print()

    # ------------------------------------------------------------------
    # Tokenizer slots
    # ------------------------------------------------------------------
    print("=== Tokenizer slots (DEP=state-dependent, INV=state-invariant) ===")
    deploy_tok = deploy_tokenizer_views(blob)
    il_tok = il_tokenizer_views(dump)
    suspicious_count = 0
    # For StandStill, both command and motion_anchor_ori_b_mf_nonflat are
    # constant across robot poses (the future window just repeats the default
    # standing pose in anchor-frame). On a non-StandStill motion, command
    # would be motion-pose-dependent so flip these to DEP.
    state_invariant_motion = True  # we run StandStill in --obs-dump mode
    for name in ["command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"]:
        d = report_slot(name, state_invariant_motion,
                        deploy_tok[name], il_tok[name], args.top_n)
        if print_slot(d, args.threshold, args.top_n):
            suspicious_count += 1
    print()

    # ------------------------------------------------------------------
    # Proprioception slots
    # ------------------------------------------------------------------
    print("=== Proprioception slots ===")
    deploy_prop = deploy_proprio_views(blob)
    il_prop = il_proprio_views(dump)
    PROPRIO_INVARIANTS = {
        # last_action is 0 on the very first tick after CONTROL entry on both
        # sides, since the buffer broadcast-fills with last_action_il_=0.
        "last_action": True,
        # The other terms depend on the robot pose, so we expect a mismatch.
        "base_ang_vel":  False,
        "joint_pos_rel": False,
        "joint_vel":     False,
        "gravity_dir":   False,
    }
    for name, invariant in PROPRIO_INVARIANTS.items():
        d = report_slot(name, invariant,
                        deploy_prop[name], il_prop[name], args.top_n)
        if print_slot(d, args.threshold, args.top_n):
            suspicious_count += 1
    print()

    # ------------------------------------------------------------------
    # Optional: re-run the ONNX on the deploy obs and compare to recorded
    # ------------------------------------------------------------------
    if args.rerun_onnx is not None:
        print(f"=== Re-running ONNX on deploy obs ({args.rerun_onnx}) ===")
        try:
            import onnxruntime as ort
        except ImportError:
            print("  onnxruntime not available, skipping rerun.")
        else:
            sess = ort.InferenceSession(str(args.rerun_onnx),
                                        providers=["CPUExecutionProvider"])
            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name
            x = np.concatenate([blob.tokenizer_obs, blob.proprioception])[None].astype(np.float32)
            y = sess.run([out_name], {in_name: x})[0]
            if y.ndim == 3:
                y = y.squeeze(1)
            y = y[0].astype(np.float64)
            diff = np.abs(y - blob.action_il)
            print(f"  Python ONNX action[:6]: {y[:6]}")
            print(f"  C++   ONNX action[:6]:  {blob.action_il[:6]}")
            print(f"  max|py - cpp| = {diff.max():.6f}  mean|py - cpp| = {diff.mean():.6f}")
            if diff.max() > 1e-3:
                print("  WARNING: Python and C++ ONNX disagree -- check input "
                      "dtype/layout in the C++ wiring.")
                suspicious_count += 1
        print()

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    if suspicious_count == 0:
        print(f"OK -- no DIVERGENT state-invariant slots above {args.threshold} rad.")
        print(f"     The C++ obs pipeline appears consistent with IsaacLab GT")
        print(f"     for the slots that don't depend on robot pose.")
        return 0

    print(f"FAILED -- {suspicious_count} state-invariant slot(s) DIVERGENT "
          f"above {args.threshold} rad.")
    print(f"          That's a smoking gun for an obs-construction bug in the C++ "
          f"deploy.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
