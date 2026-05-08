#!/usr/bin/env python3
"""Compute pelvis Z (assuming feet on ground) for one or more
``x2_capture_pose.py`` JSON dumps.

Loads ``x2_ultra.xml`` via MuJoCo, sets every joint to the captured
median position (or ``DEFAULT_ANGLES`` for the baseline), runs
``mj_forward``, and reports the vertical distance from the pelvis body
origin to the lowest foot contact point. The kinematic relationship is

    pelvis_z_above_ground = pelvis_z_in_world - min(foot_z_in_world)

Pass several capture JSONs to compare them side-by-side; the script also
prints the difference between each capture and the trained ``DEFAULT_DOF``
baseline, which is the "pelvis Z drop" value to bake into
``GANTRY_HANG_PELVIS_Z`` in ``x2_mujoco_ros_bridge.py``.

Run inside the docker_x2 container (``gr00t-x2sim`` image already has
mujoco):

    docker compose -f docker-compose.yml -f docker-compose.real.yml \\
        run --rm x2sim bash -c '
            python3 /workspace/sonic/gear_sonic_deploy/scripts/x2_pelvis_z_from_capture.py \\
                /workspace/sonic/gear_sonic_deploy/scripts/_capture_A_standing.json \\
                /workspace/sonic/gear_sonic_deploy/scripts/_capture_B_gantry_loosened.json
        '
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import mujoco

# Joint name list mirrors x2_capture_pose.py / policy_parameters.hpp.
MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
    "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
    "right_knee_joint",       "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",        "waist_pitch_joint",     "waist_roll_joint",
    "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",    "left_elbow_joint",
    "left_wrist_yaw_joint",       "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",   "right_elbow_joint",
    "right_wrist_yaw_joint",      "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
    "head_yaw_joint",             "head_pitch_joint",
)
NUM_DOFS = len(MUJOCO_JOINT_NAMES)

# Mirror of DEFAULT_ANGLES in policy_parameters.hpp / eval_x2_mujoco.py.
DEFAULT_ANGLES = np.array((
    -0.312, 0.0,   0.0,    0.669, -0.363, 0.0,
    -0.312, 0.0,   0.0,    0.669, -0.363, 0.0,
     0.0,   0.0,   0.0,
     0.2,   0.2,   0.0,   -0.6,   0.0,    0.0,   0.0,
     0.2,  -0.2,   0.0,   -0.6,   0.0,    0.0,   0.0,
     0.0,   0.0,
), dtype=np.float64)
assert DEFAULT_ANGLES.shape == (NUM_DOFS,)


_DEFAULT_MJCF = pathlib.Path(__file__).resolve().parents[2] / \
    "gear_sonic" / "data" / "assets" / "robot_description" / "mjcf" / "x2_ultra.xml"


def _qpos_indices(model: mujoco.MjModel) -> tuple[np.ndarray, bool]:
    """Map MUJOCO_JOINT_NAMES -> qpos addresses; flag whether the model
    has a free floating base (qpos length 7 prefix)."""
    qpos_adr = np.zeros(NUM_DOFS, dtype=np.int64)
    for i, name in enumerate(MUJOCO_JOINT_NAMES):
        try:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise KeyError(name)
        except Exception as e:
            raise SystemExit(
                f"ERROR: joint '{name}' not in MJCF. Update either the MJCF or "
                "MUJOCO_JOINT_NAMES in this script."
            ) from e
        qpos_adr[i] = int(model.jnt_qposadr[jid])
    has_floating = bool(model.nq >= NUM_DOFS + 7)
    return qpos_adr, has_floating


def _foot_bottom_zs(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    """Return (left_foot_min_z, right_foot_min_z) in world frame.

    We use the geom AABB of every geom whose body is the ankle-roll link
    of each leg. ``geom_xpos`` is the world centre, ``geom_size`` the
    half-extents (sensible for box / capsule); the lowest point is
    ``z - half_extent_z``. For mesh geoms ``geom_size`` is zero, so we
    fall back to ``geom_aabb`` (mujoco fills it on mj_forward).
    """
    out = []
    for body_name in ("left_ankle_roll_link", "right_ankle_roll_link"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            raise SystemExit(f"ERROR: body '{body_name}' not in MJCF.")
        # Iterate every geom whose root body is `bid`.
        min_z = math.inf
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != bid:
                continue
            gpos = data.geom_xpos[gid]   # (3,) world
            gmat = data.geom_xmat[gid].reshape(3, 3)
            # AABB extent: model.geom_aabb is (ngeom, 6) = (cx,cy,cz,ex,ey,ez)
            # in geom-local frame. Project the local AABB corners through
            # the geom's world transform and take the min Z.
            aabb = model.geom_aabb[gid]
            cx, cy, cz, ex, ey, ez = aabb
            corners_local = np.array([
                [cx + sx*ex, cy + sy*ey, cz + sz*ez]
                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)
            ])
            corners_world = (gmat @ corners_local.T).T + gpos
            min_z = min(min_z, float(np.min(corners_world[:, 2])))
        if not math.isfinite(min_z):
            raise SystemExit(f"ERROR: no geoms found on '{body_name}'.")
        out.append(min_z)
    return out[0], out[1]


def _pelvis_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if bid < 0:
        # Some MJCF variants use 'base_link' / 'root'.
        for alt in ("base_link", "torso", "root"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, alt)
            if bid >= 0:
                break
        if bid < 0:
            raise SystemExit("ERROR: no 'pelvis' (or base_link/torso) body in MJCF.")
    return float(data.xpos[bid][2])


def _set_pose(model: mujoco.MjModel, data: mujoco.MjData,
               qpos_adr: np.ndarray, dof: np.ndarray,
               has_floating: bool, base_z: float = 1.0) -> None:
    mujoco.mj_resetData(model, data)
    if has_floating:
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = float(base_z)
        data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)  # wxyz identity
    for i in range(NUM_DOFS):
        data.qpos[qpos_adr[i]] = float(dof[i])
    mujoco.mj_forward(model, data)


def _dof_from_capture_json(payload: dict) -> np.ndarray:
    """Extract the 31-DOF pose vector (median positions) from a capture JSON.
    Joints with no samples fall back to DEFAULT_ANGLES at that slot.
    """
    dof = DEFAULT_ANGLES.copy()
    names = payload["joint_names"]
    pos = payload["median_position_rad"]
    for name, val in zip(names, pos):
        if val is None:
            continue
        try:
            i = MUJOCO_JOINT_NAMES.index(name)
        except ValueError:
            continue
        dof[i] = float(val)
    return dof


def _label(p: pathlib.Path) -> str:
    return p.name


def main() -> int:
    p = argparse.ArgumentParser(
        description="FK pelvis Z from x2_capture_pose.py JSON dumps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("captures", nargs="+", type=pathlib.Path,
                   help="One or more capture JSON files.")
    p.add_argument("--mjcf", type=pathlib.Path, default=_DEFAULT_MJCF,
                   help="Path to x2_ultra.xml MJCF.")
    args = p.parse_args()

    if not args.mjcf.is_file():
        print(f"ERROR: MJCF not found: {args.mjcf}", file=sys.stderr)
        return 1

    print(f"Loading MJCF: {args.mjcf}")
    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    qpos_adr, has_floating = _qpos_indices(model)
    if not has_floating:
        print("WARNING: MJCF has no floating base; pelvis Z will be relative "
              "to the world origin in the kinematic tree, not to the feet.")

    poses: list[tuple[str, np.ndarray]] = [
        ("DEFAULT_ANGLES (trained)", DEFAULT_ANGLES.copy()),
    ]
    for path in args.captures:
        try:
            payload = json.loads(path.read_text())
        except Exception as e:
            print(f"  WARNING: could not parse {path}: {e}")
            continue
        dof = _dof_from_capture_json(payload)
        poses.append((_label(path), dof))

    print()
    print(f"  {'pose':<48} {'pelvis_z':>11}  {'L_foot_z':>9}  "
          f"{'R_foot_z':>9}  {'pelvis_above':>13}")
    print("  " + "─" * 95)
    rows = []
    for name, dof in poses:
        _set_pose(model, data, qpos_adr, dof, has_floating, base_z=1.0)
        lz, rz = _foot_bottom_zs(model, data)
        pz = _pelvis_z(model, data)
        ground_z = min(lz, rz)
        pelvis_above = pz - ground_z
        rows.append((name, dof, pz, lz, rz, pelvis_above))
        print(f"  {name[:48]:<48} {pz:>+11.4f}  {lz:>+9.4f}  "
              f"{rz:>+9.4f}  {pelvis_above:>+13.4f}")

    if len(rows) > 1:
        baseline_above = rows[0][5]
        print()
        print("Pelvis-Z relative to DEFAULT_ANGLES baseline (negative = lower):")
        for name, _dof, _pz, _lz, _rz, above in rows[1:]:
            delta = above - baseline_above
            print(f"  {name[:48]:<48} delta = {delta:+.4f} m  "
                  f"({delta*100:+.2f} cm)")
        print()
        print("Suggested GANTRY_HANG_PELVIS_Z:")
        # The bridge currently spawns the floating base at z = pelvis_z (so
        # the pelvis body origin sits at that z). With feet on ground, the
        # right setting is pelvis_above_ground for the chosen pose.
        # We pick the LAST capture as the gantry-hang reference.
        chosen_name, _dof, _pz, _lz, _rz, chosen_above = rows[-1]
        print(f"  Using pose: {chosen_name}")
        print(f"  GANTRY_HANG_PELVIS_Z = {chosen_above:.3f}")
        # Reference: bridge's _reset_to_default_pose uses 0.85 for the
        # standing pose. Print the corresponding default-pose height for
        # cross-check.
        print(f"  (DEFAULT_ANGLES baseline: pelvis_above = "
              f"{baseline_above:.3f} m -- bridge's hard-coded 0.85 differs "
              f"by {0.85 - baseline_above:+.3f} m)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
