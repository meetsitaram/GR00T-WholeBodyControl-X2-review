#!/usr/bin/env python3
"""Diff two ``X2OBSV01`` first-tick obs blobs slot-by-slot.

Both inputs use the same binary layout that ``x2_deploy_onnx_ref.cpp::
DumpObsBlob`` writes (and that ``eval_x2_mujoco_onnx.py --obs-dump`` mirrors
on the Python side). This script directly compares a C++ deploy dump
against a Python eval dump for sim-to-sim parity, with no IsaacLab GT in
the loop.

What we expect, given the same checkpoint, the same motion source, and the
bridge's RSI putting MuJoCo at the motion's frame 0:

    * tokenizer_obs            slot-for-slot identical (both eval against
                                the same motion frame 0 from the same PKL)
    * proprioception           identical EXCEPT for tiny float-precision
                                noise (both compute the same gravity rotate
                                and joint_pos_rel from the same qpos)
    * action_il                identical to within ONNX numerical noise
                                (~1e-5 typical) when the obs match

Any non-trivial divergence (>1e-3 in float32 slots, >1e-5 in action) is
either an obs-construction bug in the C++ deploy or a state-injection
mismatch between the two sides (e.g. base_quat sign convention, joint
ordering). The output ranks slots by ``max|Δ|`` so the worst offender
floats to the top.

Usage::

    python gear_sonic_deploy/scripts/compare_deploy_vs_python_obs.py \\
        --cpp /tmp/cpp_obs.bin \\
        --py  /tmp/py_obs.bin \\
        [--top-n 8]
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = b"X2OBSV01"
HEADER_FMT = "<8sIIId"
HEADER_SIZE = struct.calcsize(HEADER_FMT)


@dataclass
class ObsBlob:
    path: Path
    policy_time: float
    tokenizer_obs: np.ndarray   # (tok_dim,)  float32
    proprioception: np.ndarray  # (prop_dim,) float32
    action_il: np.ndarray       # (action_dim,) float64
    joint_pos_mj: np.ndarray    # (31,) float64
    joint_vel_mj: np.ndarray    # (31,) float64
    base_quat_wxyz: np.ndarray  # (4,)  float64
    base_ang_vel: np.ndarray    # (3,)  float64


def load_blob(path: Path) -> ObsBlob:
    raw = path.read_bytes()
    if len(raw) < HEADER_SIZE:
        sys.exit(f"{path}: too short ({len(raw)} < {HEADER_SIZE} header bytes)")
    magic, tok_dim, prop_dim, action_dim, policy_time = struct.unpack(
        HEADER_FMT, raw[:HEADER_SIZE]
    )
    if magic != MAGIC:
        sys.exit(f"{path}: magic {magic!r} != {MAGIC!r}")

    off = HEADER_SIZE
    tok = np.frombuffer(raw[off : off + tok_dim * 4], dtype="<f4").copy()
    off += tok_dim * 4
    prop = np.frombuffer(raw[off : off + prop_dim * 4], dtype="<f4").copy()
    off += prop_dim * 4
    act = np.frombuffer(raw[off : off + action_dim * 8], dtype="<f8").copy()
    off += action_dim * 8
    jp = np.frombuffer(raw[off : off + 31 * 8], dtype="<f8").copy()
    off += 31 * 8
    jv = np.frombuffer(raw[off : off + 31 * 8], dtype="<f8").copy()
    off += 31 * 8
    bq = np.frombuffer(raw[off : off + 4 * 8], dtype="<f8").copy()
    off += 4 * 8
    bv = np.frombuffer(raw[off : off + 3 * 8], dtype="<f8").copy()
    off += 3 * 8

    if off != len(raw):
        sys.exit(f"{path}: trailing {len(raw) - off} bytes after parse")

    return ObsBlob(
        path=path,
        policy_time=float(policy_time),
        tokenizer_obs=tok,
        proprioception=prop,
        action_il=act,
        joint_pos_mj=jp,
        joint_vel_mj=jv,
        base_quat_wxyz=bq,
        base_ang_vel=bv,
    )


def diff_summary(name: str, a: np.ndarray, b: np.ndarray, top_n: int = 8) -> str:
    if a.shape != b.shape:
        return f"  {name:30s}  SHAPE MISMATCH: {a.shape} vs {b.shape}"
    d = np.abs(a - b)
    max_abs = float(d.max()) if d.size else 0.0
    mean_abs = float(d.mean()) if d.size else 0.0
    max_rel = (
        float((d / (np.abs(a) + np.abs(b) + 1e-12)).max()) if d.size else 0.0
    )
    flag = "" if max_abs < 1e-5 else (" <-- DIVERGENT" if max_abs > 1e-3 else " <-- minor")
    line = (
        f"  {name:30s}  shape={tuple(a.shape)}  "
        f"max|Δ|={max_abs:.6f}  mean|Δ|={mean_abs:.6f}  "
        f"max|Δ/scale|={max_rel:.4f}{flag}"
    )
    if max_abs > 1e-3 and a.size > 1:
        # Show the worst offenders in flat order.
        idx = np.argsort(d.ravel())[::-1][:top_n]
        rows = []
        for i in idx:
            rows.append(f"      [{int(i):4d}]  cpp={float(a.ravel()[i]):+.6f}  "
                        f"py={float(b.ravel()[i]):+.6f}  Δ={float(d.ravel()[i]):+.6f}")
        line = line + "\n" + "\n".join(rows)
    return line


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cpp", required=True, type=Path,
                    help="C++ deploy obs dump (--obs-dump from x2_deploy_onnx_ref).")
    ap.add_argument("--py", required=True, type=Path,
                    help="Python eval obs dump (--obs-dump from eval_x2_mujoco_onnx.py).")
    ap.add_argument("--top-n", type=int, default=8,
                    help="For divergent slots, show the top-N worst-offender entries.")
    ap.add_argument("--action-threshold", type=float, default=1e-4,
                    help="PASS/FAIL threshold for action_il max|Δ| (default 1e-4).")
    args = ap.parse_args()

    cpp = load_blob(args.cpp)
    py = load_blob(args.py)

    print("=== blob headers ===")
    print(f"  cpp  policy_time={cpp.policy_time:.4f}  tok={cpp.tokenizer_obs.shape}  "
          f"prop={cpp.proprioception.shape}  act={cpp.action_il.shape}")
    print(f"  py   policy_time={py.policy_time:.4f}  tok={py.tokenizer_obs.shape}  "
          f"prop={py.proprioception.shape}  act={py.action_il.shape}")

    print()
    print("=== state injected by the bridge / Python eval (should match if RSI is consistent) ===")
    print(diff_summary("joint_pos_mj",   cpp.joint_pos_mj,   py.joint_pos_mj,   args.top_n))
    print(diff_summary("joint_vel_mj",   cpp.joint_vel_mj,   py.joint_vel_mj,   args.top_n))
    print(diff_summary("base_quat_wxyz", cpp.base_quat_wxyz, py.base_quat_wxyz, args.top_n))
    print(diff_summary("base_ang_vel",   cpp.base_ang_vel,   py.base_ang_vel,   args.top_n))

    print()
    print("=== obs vectors fed to ONNX (should match modulo float32 noise) ===")
    print(diff_summary("tokenizer_obs (680)",  cpp.tokenizer_obs,  py.tokenizer_obs,  args.top_n))
    print(diff_summary("proprioception (990)", cpp.proprioception, py.proprioception, args.top_n))

    print()
    print("=== action (ONNX output; should match to ~1e-5) ===")
    print(diff_summary("action_il (31)", cpp.action_il, py.action_il, args.top_n))

    action_diff = float(np.abs(cpp.action_il - py.action_il).max())
    passed = action_diff < args.action_threshold
    print()
    print(f"VERDICT: action max|Δ|={action_diff:.3e} vs threshold={args.action_threshold:.3e} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
