#!/usr/bin/env python3
"""Bake the planner's ``idle_stand`` primitive into an X2M2 binary.

Output is a flat little-endian binary file consumed by:

  * ``gear_sonic_deploy/scripts/x2_pose_proxy.py`` (Python; PC2 side)
    -- the split-topology idle fallback proxy that publishes idle_stand
    frames to the deploy on PC2 loopback whenever the upstream pose
    wire from the laptop goes silent (e.g. wifi drop, laptop crash).

  * ``PklMotionReference::Load`` in
    ``gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/reference_motion.hpp``
    (C++; same loader the deploy already uses for ``--motion <x2m2>``
    offline replay). Kept format-compatible so a future native C++
    idle fallback can read the exact same file.

Why a dedicated baker rather than ``export_motion_for_deploy.py``:
  * the planner primitives PKL keys each bin by name and stores
    ``root_rot_xyzw`` (not ``root_rot``), so the generic exporter would
    pick up the wrong dict entry / wrong field name.
  * we apply the SAME 30->50 Hz resample + yaw-align-to-(0, 0) that
    ``live_vla_publish_motion_token._load_idle_stand_loop`` applies
    before publishing to the wire in ``--no-policy`` mode. Re-using
    that loader here means the X2M2 file is byte-equivalent to what
    the laptop bridge emits, so the policy sees one consistent idle
    reference distribution regardless of whether the operator is
    teleoping (laptop bridge -> deploy) or the wire went silent and
    the proxy is filling in (X2M2 -> deploy).

Format spec (mirrors ``export_motion_for_deploy.bake_x2m2`` and
``reference_motion.hpp``)::

    uint32   magic        == 0x58324D32  ("X2M2", little-endian)
    uint32   num_frames
    uint32   num_dofs     (must equal 31)
    double   fps
    for each frame f in [0, num_frames):
        double   joint_pos_mj[31]
        double   root_quat_xyzw[4]

Joint velocity is reconstructed at runtime via finite-diff in the
proxy (matches what the C++ deploy does for ``--motion`` replay), so
we intentionally do NOT serialise qvel here.

Example::

    .venv/bin/python -m gear_sonic_deploy.scripts.bake_idle_stand_x2m2 \\
        --primitives-pkl gear_sonic/data/motions/x2_planner_primitives.pkl \\
        --out gear_sonic_deploy/data/idle_stand.x2m2

``pc2_bringup.sh`` calls this script during PC2 setup and rsyncs the
output to ``${PC2_PREFIX}/data/idle_stand.x2m2`` so the proxy has it
on disk before the deploy ever starts.
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Re-use the laptop bridge's loader so we share the exact same
# resample-30-to-50-Hz + yaw-align-to-(0, 0) machinery.
# live_vla_publish_motion_token.py has lightweight module-level imports
# (joblib, numpy, zmq, gear_sonic.utils.teleop.zmq.*); no torch /
# isaaclab pulls happen here.
sys.path.insert(0, str(REPO_ROOT / "gear_sonic" / "scripts"))
from live_vla_publish_motion_token import _load_idle_stand_loop  # noqa: E402

X2M2_MAGIC = 0x58324D32  # "X2M2" little-endian
NUM_DOFS = 31


def bake(primitives_pkl: Path, out_path: Path, fps: float = 50.0) -> None:
    if not primitives_pkl.is_file():
        raise FileNotFoundError(f"primitives PKL not found: {primitives_pkl}")

    print(f"[bake_idle] loading idle_stand from {primitives_pkl} ...", flush=True)
    loop = _load_idle_stand_loop(primitives_pkl)
    # _IdleStandLoop stores dof (T, 31) f32 and quat (T, 4) f32, both
    # already resampled to DEFAULT_PUB_RATE_HZ (50 Hz) and yaw-aligned
    # to (xy=0, yaw=0). Promote to f64 for X2M2 (matches the format
    # the C++ PklMotionReference loader and export_motion_for_deploy
    # emit -- consistent across the X2M2 ecosystem).
    dof_f64 = loop._dof.astype(np.float64, copy=False)
    quat_f64 = loop._quat.astype(np.float64, copy=False)
    n_frames = int(dof_f64.shape[0])

    # Sanity: quaternion |q| ~ 1 on first / last frame catches obvious
    # wxyz-vs-xyzw swaps before the deploy parses them and tilts the
    # robot 180 deg.
    for label, q in (("first", quat_f64[0]), ("last", quat_f64[-1])):
        norm = float(np.linalg.norm(q))
        if not (0.95 <= norm <= 1.05):
            raise ValueError(
                f"{label} root_quat_xyzw has |q|={norm:.4f}, expected ~1.0; "
                f"is the primitives PKL really xyzw scipy convention?"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<III", X2M2_MAGIC, n_frames, NUM_DOFS))
        f.write(struct.pack("<d", float(fps)))
        for i in range(n_frames):
            f.write(dof_f64[i].tobytes(order="C"))
            f.write(quat_f64[i].tobytes(order="C"))

    size = out_path.stat().st_size
    expected = 4 * 3 + 8 + n_frames * (8 * NUM_DOFS + 8 * 4)
    print(
        f"[bake_idle] wrote {out_path}\n"
        f"            num_frames = {n_frames}\n"
        f"            fps        = {fps:g}\n"
        f"            duration   = {n_frames / fps:.2f} s\n"
        f"            bytes      = {size} (expected {expected})",
        flush=True,
    )
    if size != expected:
        print(
            "[bake_idle] WARNING: byte count mismatch -- check struct alignment",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--primitives-pkl",
        type=Path,
        default=REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_primitives.pkl",
        help="Source PKL containing the planner's 'idle_stand' bin "
             "(default: gear_sonic/data/motions/x2_planner_primitives.pkl).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "gear_sonic_deploy" / "data" / "idle_stand.x2m2",
        help="Output .x2m2 path (default: gear_sonic_deploy/data/idle_stand.x2m2).",
    )
    p.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="FPS stamped into the X2M2 header (default 50; matches "
             "the deploy control loop and bridge publish rate).",
    )
    args = p.parse_args(argv)
    bake(args.primitives_pkl, args.out, fps=args.fps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
