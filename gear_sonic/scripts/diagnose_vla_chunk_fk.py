"""Compare policy intent (raw decoded body) vs delivered wire by FK.

Reads chunk dumps written by the VLA bridge (``vla_chunks/chunk_*.npz``)
and FKs three trajectories for each dump:

    * raw_joint_pos_mj  -- the SONIC decoder's body intent for the
                           current tick, BEFORE wire-shaping
                           (clamp + LPF + ramp + chunk-blend).
    * wire_joint_pos_mj -- the actual wire that left the bridge for
                           that tick (post-clamp/LPF/ramp/blend).
    * body_q_mj         -- the measured body at chunk-dump time.

For each side (left/right) we compute the wrist xyz in pelvis frame
(root held at identity, so the trajectory reflects pure waist+arm
motion) and print a per-chunk table plus aggregate statistics.

The diagnostic question this answers: when the policy commands a
3.0 rad joint-space delta but the wire only delivers 1.1 rad, is
the bridge clamping AWAY a real "descend to the can" motion (raw
wrist z drops, wire wrist z stays high) or is the 3.0 rad just
shoulder-yaw + wrist-roll that doesn't translate to a descend
(raw wrist z also stays high)? Without this script we were
inferring policy intent from a scalar max-abs deviation, which is
ambiguous; with full FK on the raw decode we get the unambiguous
3D answer.

Usage:

    .venv-viewer/bin/python -m gear_sonic.scripts.diagnose_vla_chunk_fk \\
        --chunk-dir /tmp/x2_vla_runtime-20260610_055442/vla_chunks
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np


_DEFAULT_MJCF = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "assets"
    / "robot_description"
    / "mjcf"
    / "x2_ultra.xml"
)


_X2_BODY_JOINT_NAMES_31: list[str] = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint",
    "head_pitch_joint",
]


def _fk_wrists(
    model,
    data,
    qpos0: np.ndarray,
    qpos_idx_arr: np.ndarray,
    body_ids: dict[str, int],
    body_q_mj_series: np.ndarray,
) -> dict[str, np.ndarray]:
    import mujoco

    n_frames = int(body_q_mj_series.shape[0])
    out: dict[str, np.ndarray] = {
        body_name: np.empty((n_frames, 3), dtype=np.float64)
        for body_name in body_ids
    }
    for f in range(n_frames):
        np.copyto(data.qpos, qpos0)
        data.qpos[qpos_idx_arr] = body_q_mj_series[f, :31].astype(np.float64)
        mujoco.mj_kinematics(model, data)
        for body_name, bid in body_ids.items():
            out[body_name][f] = data.xpos[bid]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--chunk-dir",
        type=Path,
        required=True,
        help="Directory containing chunk_*.npz dumps from the VLA bridge.",
    )
    ap.add_argument(
        "--mjcf",
        type=Path,
        default=_DEFAULT_MJCF,
        help=f"Path to X2 MJCF (default: {_DEFAULT_MJCF}).",
    )
    ap.add_argument(
        "--side",
        choices=("left", "right", "both"),
        default="left",
        help="Which wrist to print per-chunk (default: left).",
    )
    args = ap.parse_args()

    chunk_files = sorted(args.chunk_dir.glob("chunk_*.npz"))
    if not chunk_files:
        print(f"[diag] no chunk_*.npz under {args.chunk_dir}")
        return 1

    import mujoco

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)

    qpos_indices = [
        int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)])
        for jn in _X2_BODY_JOINT_NAMES_31
    ]
    qpos_idx_arr = np.asarray(qpos_indices, dtype=np.int64)
    qpos0 = np.zeros(model.nq, dtype=np.float64)
    qpos0[3:7] = [1.0, 0.0, 0.0, 0.0]

    body_ids: dict[str, int] = {}
    for body_name in ("left_wrist_roll_link", "right_wrist_roll_link"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            print(f"[diag] WARN: body {body_name!r} not in {args.mjcf}")
            continue
        body_ids[body_name] = bid
    if not body_ids:
        return 2

    # Collect (raw, wire, body) per chunk and FK in one batch.
    raws, wires, bodies, chunk_meta = [], [], [], []
    skipped_no_raw = 0
    for cp in chunk_files:
        d = np.load(cp)
        if "raw_joint_pos_mj" not in d.files:
            skipped_no_raw += 1
            continue
        raws.append(d["raw_joint_pos_mj"])
        wires.append(d["wire_joint_pos_mj"])
        bodies.append(d["body_q_mj"])
        chunk_meta.append({
            "name": cp.name,
            "chunk_id": int(d["wire_chunk_id"][0]),
            "chunk_step": int(d["wire_chunk_step"][0]),
            "raw_dΔ": float(d["raw_delta_idle_rad"][0]),
            "wire_dΔ": float(d["wire_delta_idle_rad"][0]),
            "wall_t": float(d["wall_t_s"][0]),
        })

    if skipped_no_raw:
        print(
            f"[diag] {skipped_no_raw} dump(s) lacked raw_joint_pos_mj "
            "(written before bridge patch). Re-record with the new bridge."
        )
    if not raws:
        print(
            "[diag] none of the dumps had raw_joint_pos_mj; "
            "re-run the bridge after rebuilding."
        )
        return 3

    raw_arr = np.asarray(raws, dtype=np.float32)
    wire_arr = np.asarray(wires, dtype=np.float32)
    body_arr = np.asarray(bodies, dtype=np.float32)

    raw_fk = _fk_wrists(model, data, qpos0, qpos_idx_arr, body_ids, raw_arr)
    wire_fk = _fk_wrists(model, data, qpos0, qpos_idx_arr, body_ids, wire_arr)
    body_fk = _fk_wrists(model, data, qpos0, qpos_idx_arr, body_ids, body_arr)

    sides = (
        [("left_wrist_roll_link", "left")]
        if args.side == "left"
        else [("right_wrist_roll_link", "right")]
        if args.side == "right"
        else [
            ("left_wrist_roll_link", "left"),
            ("right_wrist_roll_link", "right"),
        ]
    )

    t0 = chunk_meta[0]["wall_t"]
    for body_name, side_label in sides:
        if body_name not in raw_fk:
            continue
        print(
            f"\n========== {side_label.upper()} WRIST (pelvis frame, meters) =========="
        )
        print(
            f"{'idx':>3} {'t_s':>6} {'cid':>4} {'rawΔ':>5} {'wireΔ':>5}  "
            f"{'rawX':>6} {'rawY':>6} {'rawZ':>6}  "
            f"{'wirX':>6} {'wirY':>6} {'wirZ':>6}  "
            f"{'bodX':>6} {'bodY':>6} {'bodZ':>6}  "
            f"{'(raw-wir)Z':>10}"
        )
        for i, m in enumerate(chunk_meta):
            r = raw_fk[body_name][i]
            w = wire_fk[body_name][i]
            b = body_fk[body_name][i]
            dz = r[2] - w[2]
            print(
                f"{i:3d} {m['wall_t']-t0:6.1f} {m['chunk_id']:4d} "
                f"{m['raw_dΔ']:5.2f} {m['wire_dΔ']:5.2f}  "
                f"{r[0]:6.3f} {r[1]:6.3f} {r[2]:6.3f}  "
                f"{w[0]:6.3f} {w[1]:6.3f} {w[2]:6.3f}  "
                f"{b[0]:6.3f} {b[1]:6.3f} {b[2]:6.3f}  "
                f"{dz:+10.3f}"
            )

        # Aggregate stats over post-idle chunks.
        active_mask = np.array(
            [m["chunk_id"] > 0 for m in chunk_meta], dtype=bool
        )
        if not active_mask.any():
            continue
        r_act = raw_fk[body_name][active_mask]
        w_act = wire_fk[body_name][active_mask]
        b_act = body_fk[body_name][active_mask]
        print(
            f"\n=== POST-IDLE {side_label.upper()} wrist stats "
            f"({int(active_mask.sum())} chunks) ==="
        )
        for label, arr in (("raw  intent", r_act), ("wire delivered", w_act), ("body measured", b_act)):
            print(
                f"  {label:>15}:  "
                f"x[min/med/max]=[{arr[:,0].min():+.3f}/"
                f"{np.median(arr[:,0]):+.3f}/{arr[:,0].max():+.3f}]  "
                f"y[min/med/max]=[{arr[:,1].min():+.3f}/"
                f"{np.median(arr[:,1]):+.3f}/{arr[:,1].max():+.3f}]  "
                f"z[min/med/max]=[{arr[:,2].min():+.3f}/"
                f"{np.median(arr[:,2]):+.3f}/{arr[:,2].max():+.3f}]"
            )

        # Headline number: does raw wrist descend below wire wrist?
        dz_med = float(np.median(r_act[:, 2] - w_act[:, 2]))
        dz_min = float((r_act[:, 2] - w_act[:, 2]).min())
        print(
            f"\n  median (raw - wire) Z = {dz_med:+.3f} m   "
            f"min (raw - wire) Z = {dz_min:+.3f} m"
        )
        if dz_med < -0.05:
            print(
                f"  -> RAW WRIST IS LOWER than wire by >5 cm: bridge IS "
                "clamping away policy-commanded descent.\n"
                "     Try relaxing --vla-max-wire-step / --vla-target-lpf-hz "
                "/ --vla-max-wire-dev-from-body and re-run."
            )
        elif dz_med > 0.05:
            print(
                f"  -> RAW WRIST IS HIGHER than wire by >5 cm: bridge is "
                "NOT the bottleneck.\n"
                "     Policy targets a higher pose than what the wire reaches; "
                "bridge clamps are letting wire LAG the policy intent "
                "but the policy itself isn't asking for descend. "
                "Look upstream (training data / mode collapse)."
            )
        else:
            print(
                f"  -> raw ≈ wire (|ΔZ| ≤ 5 cm): bridge is faithfully "
                "tracking policy intent; whatever the wrist does is "
                "what the policy commanded. Look upstream "
                "(training data / mode collapse)."
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
