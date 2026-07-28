#!/usr/bin/env python3
"""Generate the Asimov v1 URDF from the mjlab MJCF, with FK parity verification.

Why this exists: IsaacLab needs a URDF; Asimov ships only an MJCF (the public
asimov-1 repo's URDF-less MJCF is deprecated). Hand-converting MJCF->URDF is
exactly where the port plan's silent failures live -- above all the elbow
`ref="+/-0.785398"` (MJCF shifts joint zero; URDF has no equivalent, and the
error "fails silently" as a 45 deg arm offset).

Approach that makes `ref` a non-issue: every URDF frame is derived from MuJoCo
FORWARD KINEMATICS EVALUATED AT TRUE qpos=0 (explicitly zeroed -- MuJoCo seeds
qpos from qpos0=ref on load!). URDF link frames are placed AT the joint anchor
(URDF rotates children about the child-frame origin; MuJoCo about jnt_pos):

    L_b = T(xpos_b, xquat_b) * Trans(jnt_pos_b)        # world, at qpos=0
    joint origin = L_parent^-1 * L_b
    axis          = jnt_axis (child body frame; unchanged by pure translation)
    inertial/geom origins = body-frame values shifted by -jnt_pos

Conventions copied from x2_ultra_sphere_feet.urdf: every link gets mesh visual
+ mesh collision, EXCEPT the ankle_roll links which get the MJCF's 4 foot
collision spheres verbatim (contact parity with MuJoCo). Welded neck bodies
become fixed joints. Effort/velocity limits are the hardware-characterized
values from mjlab's asimov_1_constant.py (effort = hard clamp, NOT
saturation_effort).

Usage:
    python gear_sonic/scripts/dev/mjcf_to_urdf_asimov.py            # write + verify
    python gear_sonic/scripts/dev/mjcf_to_urdf_asimov.py --verify   # verify only

Verification = parity gate A of docs/experiments/asimov_sonic_bringup_plan.md:
FK of 200 random in-range qpos through MuJoCo vs an independent numpy FK over
the WRITTEN URDF file; all link-frame positions must agree to <1e-6 m (exact
math, so tolerance is numerical only). This catches ref-baking, anchor, axis
and ordering mistakes in one shot.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MJCF = os.path.join(_REPO, "gear_sonic/data/assets/robot_description/mjcf/asimov.xml")
URDF_DIR = os.path.join(_REPO, "gear_sonic/data/assets/robot_description/urdf/asimov")
URDF = os.path.join(URDF_DIR, "asimov.urdf")
MESH_SRC = os.path.join(_REPO, "gear_sonic/data/assets/robot_description/mjcf/asimov_assets")

# Hardware-characterized limits from mjlab asimov_1_constant.py (per side).
# effort = the HARD clamp; velocity in rad/s. Keyed by joint basename.
LIMITS = {
    "hip_pitch": (40.0, 12.57), "hip_roll": (30.0, 3.98), "hip_yaw": (20.0, 5.45),
    "knee": (25.0, 12.25), "ankle_pitch": (40.0, 9.32), "ankle_roll": (17.0, 9.32),
    "waist_yaw": (40.0, 12.57),
    "shoulder_pitch": (30.0, 3.98), "shoulder_roll": (25.0, 12.25),
    "shoulder_yaw": (20.0, 5.45), "elbow": (12.0, 9.32), "wrist_yaw": (12.0, 9.32),
}


def _limits_for(joint_name: str):
    base = re.sub(r"^(left|right)_", "", joint_name).replace("_joint", "")
    return LIMITS[base]


def _fmt(v, nd=8):
    return " ".join(f"{x:.{nd}g}" for x in np.asarray(v).ravel())


def _quat_wxyz_to_rpy(q):
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")


def _bake_mesh_stl(model, mesh_id, dst, pos, Rm):
    """Write a binary STL of a mesh in its LINK frame, from the COMPILED
    model's processed vertices (model.mesh_vert). Never transform raw STL
    files: MuJoCo re-centers/re-aligns mesh assets at compile and the
    compiled geom pos/quat apply to the PROCESSED vertices -- applying them
    to raw vertices offsets every mesh by its own centroid correction
    (operator-caught twice: detached head, then jumbled hips)."""
    import struct as _st
    va, vn = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
    fa, fn = model.mesh_faceadr[mesh_id], model.mesh_facenum[mesh_id]
    verts = model.mesh_vert[va:va + vn]
    faces = model.mesh_face[fa:fa + fn]
    v = (verts @ Rm.T) + pos
    out = bytearray(b"\0" * 80)
    out += _st.pack("<I", fn)
    for f in faces:
        a, b, c = v[f[0]], v[f[1]], v[f[2]]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        n = n / ln if ln > 1e-12 else n
        out += _st.pack("<12fH", *n, *a, *b, *c, 0)
    open(dst, "wb").write(bytes(out))


def _mesh_file_map(mjcf_path):
    """MJCF mesh asset name -> STL filename (name defaults to file stem)."""
    tree = ET.parse(mjcf_path)
    out = {}
    for m in tree.iter("mesh"):
        f = m.get("file")
        if f:
            out[m.get("name") or os.path.splitext(os.path.basename(f))[0]] = os.path.basename(f)
    return out


def build(model: mujoco.MjModel, mesh_map: dict) -> str:
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    data.qpos[3] = 1.0                       # freejoint quat wxyz identity
    mujoco.mj_forward(model, data)

    def name(objtype, i):
        return mujoco.mj_id2name(model, objtype, i)

    B = mujoco.mjtObj.mjOBJ_BODY
    # world link frame L_b for every body: body frame shifted to joint anchor
    L_pos, L_rot, anchor = {}, {}, {}
    for b in range(1, model.nbody):
        Rb = data.xmat[b].reshape(3, 3).copy()
        pb = data.xpos[b].copy()
        a = np.zeros(3)
        if model.body_jntnum[b] == 1:
            j = model.body_jntadr[b]
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                a = model.jnt_pos[j].copy()
        anchor[b] = a
        L_rot[b] = Rb
        L_pos[b] = pb + Rb @ a

    links, joints = [], []
    for b in range(1, model.nbody):
        bn = name(B, b)
        a = anchor[b]
        # ---- inertial (shifted by -anchor) ----------------------------------
        ip = model.body_ipos[b] - a
        irpy = _quat_wxyz_to_rpy(model.body_iquat[b])
        inert = model.body_inertia[b]
        body_xml = [f'  <link name="{bn}">',
                    "    <inertial>",
                    f'      <origin xyz="{_fmt(ip)}" rpy="{_fmt(irpy)}" />',
                    f'      <mass value="{model.body_mass[b]:.8g}" />',
                    f'      <inertia ixx="{inert[0]:.8g}" ixy="0" ixz="0" '
                    f'iyy="{inert[1]:.8g}" iyz="0" izz="{inert[2]:.8g}" />',
                    "    </inertial>"]
        # ---- geoms ----------------------------------------------------------
        # VISUAL = mesh geoms. COLLISION = ONLY the MJCF's collision-enabled
        # geoms (contype|conaffinity != 0): per-limb capsules + foot spheres
        # (22 total on Asimov). Meshes are NEVER colliders -- exporting them
        # as such gave every link a convex-hull collider and poisoned
        # training with phantom self-collisions (operator-caught visually).
        for g in range(model.body_geomadr[b], model.body_geomadr[b] + model.body_geomnum[b]):
            gp = model.geom_pos[g] - a
            grpy = _quat_wxyz_to_rpy(model.geom_quat[g])
            t = model.geom_type[g]
            collidable = bool(model.geom_contype[g] or model.geom_conaffinity[g])
            if t == mujoco.mjtGeom.mjGEOM_MESH and not collidable:
                mesh = mesh_map[name(mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])]
                gR = R.from_quat([model.geom_quat[g][1], model.geom_quat[g][2],
                                  model.geom_quat[g][3], model.geom_quat[g][0]]).as_matrix()
                baked = f"{bn}__{mesh}"
                _bake_mesh_stl(model, model.geom_dataid[g],
                               os.path.join(URDF_DIR, "meshes", baked), gp, gR)
                body_xml += ["    <visual>",
                             '      <origin xyz="0 0 0" rpy="0 0 0" />',
                             f'      <geometry><mesh filename="meshes/{baked}" /></geometry>',
                             "    </visual>"]
            elif collidable and t == mujoco.mjtGeom.mjGEOM_SPHERE:
                body_xml += [f'    <collision><origin xyz="{_fmt(gp)}" rpy="0 0 0" />'
                             f'<geometry><sphere radius="{model.geom_size[g][0]:.8g}" />'
                             "</geometry></collision>"]
            elif collidable and t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                r_, hl = model.geom_size[g][0], model.geom_size[g][1]
                body_xml += ["    <collision>",
                             f'      <origin xyz="{_fmt(gp)}" rpy="{_fmt(grpy)}" />',
                             f'      <geometry><cylinder radius="{r_:.8g}" length="{2*hl:.8g}" /></geometry>',
                             "    </collision>"]
            elif collidable and t == mujoco.mjtGeom.mjGEOM_BOX:
                sx, sy, sz = model.geom_size[g]
                body_xml += ["    <collision>",
                             f'      <origin xyz="{_fmt(gp)}" rpy="{_fmt(grpy)}" />',
                             f'      <geometry><box size="{2*sx:.8g} {2*sy:.8g} {2*sz:.8g}" /></geometry>',
                             "    </collision>"]
            elif collidable and t == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = mesh_map[name(mujoco.mjtObj.mjOBJ_MESH, model.geom_dataid[g])]
                body_xml += ["    <collision>",
                             f'      <origin xyz="{_fmt(gp)}" rpy="{_fmt(grpy)}" />',
                             f'      <geometry><mesh filename="meshes/{mesh}" /></geometry>',
                             "    </collision>"]
        body_xml.append("  </link>")
        links.append("\n".join(body_xml))

        # ---- joint ----------------------------------------------------------
        parent = model.body_parentid[b]
        if parent == 0:
            continue                                        # pelvis = root link
        pn = name(B, parent)
        rel_R = L_rot[parent].T @ L_rot[b]
        rel_p = L_rot[parent].T @ (L_pos[b] - L_pos[parent])
        rpy = R.from_matrix(rel_R).as_euler("xyz")
        if model.body_jntnum[b] == 1 and \
                model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_HINGE:
            j = model.body_jntadr[b]
            jn = name(mujoco.mjtObj.mjOBJ_JOINT, j)
            lo, hi = model.jnt_range[j]
            eff, vel = _limits_for(jn)
            joints.append("\n".join([
                f'  <joint name="{jn}" type="revolute">',
                f'    <origin xyz="{_fmt(rel_p)}" rpy="{_fmt(rpy)}" />',
                f'    <parent link="{pn}" />',
                f'    <child link="{bn}" />',
                f'    <axis xyz="{_fmt(model.jnt_axis[j])}" />',
                f'    <limit lower="{lo:.8g}" upper="{hi:.8g}" '
                f'effort="{eff}" velocity="{vel}" />',
                "  </joint>"]))
        else:                                               # welded (neck)
            joints.append("\n".join([
                f'  <joint name="{bn}_fixed" type="fixed">',
                f'    <origin xyz="{_fmt(rel_p)}" rpy="{_fmt(rpy)}" />',
                f'    <parent link="{pn}" />',
                f'    <child link="{bn}" />',
                "  </joint>"]))

    hdr = ('<?xml version="1.0" encoding="utf-8"?>\n'
           "<!-- GENERATED by gear_sonic/scripts/dev/mjcf_to_urdf_asimov.py from\n"
           "     mjcf/asimov.xml (mjlab feature/asimov-velocity-training-cfg f5c4caa).\n"
           "     URDF zero == MJCF qpos=0 (elbow ref +/-45deg baked via FK-at-zero;\n"
           "     verified to 1e-6 m against MuJoCo FK; rerun the script's verify\n"
           "     mode after ANY model update). DO NOT hand-edit. -->\n"
           '<robot name="asimov">\n')
    return hdr + "\n".join(links) + "\n" + "\n".join(joints) + "\n</robot>\n"


# ---------------- independent URDF FK for verification -----------------------
def _urdf_fk(urdf_path, joint_q: dict) -> dict:
    """World position+rotation of every link, from the WRITTEN file only."""
    tree = ET.parse(urdf_path)
    robot = tree.getroot()
    jinfo = []
    children = set()
    for j in robot.iter("joint"):
        o = j.find("origin")
        xyz = np.fromstring(o.get("xyz"), sep=" ")
        rpy = np.fromstring(o.get("rpy", "0 0 0"), sep=" ")
        ax = j.find("axis")
        jinfo.append(dict(
            name=j.get("name"), type=j.get("type"),
            parent=j.find("parent").get("link"), child=j.find("child").get("link"),
            xyz=xyz, R=R.from_euler("xyz", rpy).as_matrix(),
            axis=np.fromstring(ax.get("xyz"), sep=" ") if ax is not None else None))
        children.add(j.find("child").get("link"))
    root = [l.get("name") for l in robot.iter("link") if l.get("name") not in children][0]
    frames = {root: (np.zeros(3), np.eye(3))}
    pending = list(jinfo)
    while pending:
        for j in list(pending):
            if j["parent"] in frames:
                pp, pR = frames[j["parent"]]
                p = pp + pR @ j["xyz"]
                Rm = pR @ j["R"]
                if j["type"] == "revolute":
                    Rm = Rm @ R.from_rotvec(j["axis"] * joint_q.get(j["name"], 0.0)).as_matrix()
                frames[j["child"]] = (p, Rm)
                pending.remove(j)
    return frames


def verify(model: mujoco.MjModel, urdf_path: str, n: int = 200, seed: int = 0) -> float:
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    B = mujoco.mjtObj.mjOBJ_BODY
    hinges = [j for j in range(model.njnt)
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    worst = 0.0
    for _ in range(n):
        data.qpos[:] = 0
        data.qpos[3] = 1.0
        q = {}
        for j in hinges:
            lo, hi = model.jnt_range[j]
            v = rng.uniform(lo, hi)
            data.qpos[model.jnt_qposadr[j]] = v
            q[mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)] = v
        mujoco.mj_forward(model, data)
        frames = _urdf_fk(urdf_path, q)
        for b in range(1, model.nbody):
            bn = mujoco.mj_id2name(model, B, b)
            a = np.zeros(3)
            if model.body_jntnum[b] == 1 and \
                    model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_HINGE:
                a = model.jnt_pos[model.body_jntadr[b]]
            mj_p = data.xpos[b] + data.xmat[b].reshape(3, 3) @ a
            worst = max(worst, float(np.linalg.norm(frames[bn][0] - mj_p)))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="verify existing URDF only")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(MJCF)
    if not args.verify:
        os.makedirs(os.path.join(URDF_DIR, "meshes"), exist_ok=True)
        mesh_map = _mesh_file_map(MJCF)
        for f in set(mesh_map.values()):
            shutil.copy2(os.path.join(MESH_SRC, f), os.path.join(URDF_DIR, "meshes", f))
        with open(URDF, "w") as fh:
            fh.write(build(model, mesh_map))
        print(f"wrote {URDF} (+{len(set(_mesh_file_map(MJCF).values()))} meshes)")

    err = verify(model, URDF)
    print(f"FK parity (200 random in-range poses, all link frames): "
          f"worst {err*1000:.6f} mm -> {'PASS' if err < 1e-6 else 'FAIL'}")
    raise SystemExit(0 if err < 1e-6 else 1)


if __name__ == "__main__":
    main()
