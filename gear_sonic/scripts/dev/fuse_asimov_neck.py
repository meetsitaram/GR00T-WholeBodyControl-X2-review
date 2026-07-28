#!/usr/bin/env python3
"""Fuse Asimov's welded neck bodies into waist_yaw_link (training MJCF only).

Why: the motion-lib pose_aa convention is bodies == 1 + num_dof (true for G1
30=1+29 and X2 32=1+31). Asimov's mjlab MJCF carries two WELDED neck bodies
(26 = 1+23+2) which would silently break every consumer of that convention.
MuJoCo's `fusestatic` refuses to fuse them (they are referenced by contact
excludes/sensors), so this script does it explicitly:

  1. compose mass/COM/inertia of {waist_yaw, neck_yaw, neck_pitch} into an
     explicit <inertial> on waist_yaw_link (parallel-axis, exact);
  2. reparent the neck visual mesh geoms onto waist_yaw_link with their
     world-equivalent local transforms;
  3. drop the neck <body> subtree and any exclude/sensor lines naming it.

Verified in-script: body count 24, total mass conserved, waist subtree COM
identical to the original model to 1e-9. Rerun after any upstream MJCF sync,
then regenerate the URDF (mjcf_to_urdf_asimov.py).
"""

import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MJCF = os.path.join(_REPO, "gear_sonic/data/assets/robot_description/mjcf/asimov.xml")
FUSE = ["neck_yaw_link", "neck_pitch_link"]
INTO = "waist_yaw_link"


def _body_T_in(model, data, b, ref):
    """(pos, R) of body b expressed in body ref's frame (any fixed config)."""
    Rr = data.xmat[ref].reshape(3, 3)
    pr = data.xpos[ref]
    Rb = data.xmat[b].reshape(3, 3)
    pb = data.xpos[b]
    return Rr.T @ (pb - pr), Rr.T @ Rb


def main():
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    data.qpos[:] = 0
    data.qpos[3] = 1.0
    mujoco.mj_forward(model, data)
    B = mujoco.mjtObj.mjOBJ_BODY
    ids = {mujoco.mj_id2name(model, B, i): i for i in range(model.nbody)}
    group = [INTO] + FUSE

    # ---- composed inertial in waist frame -----------------------------------
    M, first = 0.0, ids[INTO]
    com = np.zeros(3)
    parts = []
    for n in group:
        b = ids[n]
        p, Rb = _body_T_in(model, data, b, first)
        Ri = Rb @ R.from_quat(np.roll(model.body_iquat[b], -1)).as_matrix()
        c = p + Rb @ model.body_ipos[b]
        I = Ri @ np.diag(model.body_inertia[b]) @ Ri.T
        m = model.body_mass[b]
        parts.append((m, c, I))
        M += m
        com += m * c
    com /= M
    Itot = np.zeros((3, 3))
    for m, c, I in parts:
        d = c - com
        Itot += I + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))

    # ---- collect neck geoms via XML-LEVEL transform composition -------------
    # DO NOT use compiled geom_pos/geom_quat here: for mesh geoms MuJoCo bakes
    # a mesh-alignment correction into the compiled values; writing them back
    # into XML re-applies the correction (visually: the head detaches/rotates
    # ~90deg, caught by the operator from a render). Compose the raw XML
    # attributes instead: waist->neck_yaw->neck_pitch body pos/quat and each
    # geom's own pos/quat are all mesh-correction-free.
    tree0 = ET.parse(MJCF)

    def _attr_vec(el, name, default):
        v = el.get(name)
        return np.fromstring(v, sep=" ") if v else np.array(default, dtype=float)

    def _attr_rot(el):
        q = el.get("quat")
        if q is not None:
            w, x, y, z = np.fromstring(q, sep=" ")
            return R.from_quat([x, y, z, w]).as_matrix()
        e = el.get("euler")
        if e is not None:
            return R.from_euler("XYZ", np.fromstring(e, sep=" ")).as_matrix()
        return np.eye(3)

    def _find_body0(name):
        for el in tree0.iter("body"):
            if el.get("name") == name:
                return el
        raise KeyError(name)

    geom_moves = []          # (attrs_dict) new geom elements for waist
    T_chain = {}             # body name -> (pos, R) in waist frame
    T_chain[INTO] = (np.zeros(3), np.eye(3))
    for n in FUSE:
        el = _find_body0(n)
        parent_name = None
        for pb in tree0.iter("body"):
            if el in list(pb):
                parent_name = pb.get("name")
        pp, pR = T_chain[parent_name]
        bp = pp + pR @ _attr_vec(el, "pos", [0, 0, 0])
        bR = pR @ _attr_rot(el)
        T_chain[n] = (bp, bR)
        for g in el.findall("geom"):
            gp = bp + bR @ _attr_vec(g, "pos", [0, 0, 0])
            gR = bR @ _attr_rot(g)
            attrs = dict(g.attrib)
            attrs.pop("euler", None)
            if "fromto" in attrs:
                ft = np.fromstring(attrs["fromto"], sep=" ")
                p1 = bp + bR @ ft[:3]
                p2 = bp + bR @ ft[3:]
                attrs["fromto"] = " ".join(f"{v:.9g}" for v in np.concatenate([p1, p2]))
                attrs.pop("pos", None)
                attrs.pop("quat", None)
            else:
                gq = R.from_matrix(gR).as_quat()         # xyzw
                attrs["pos"] = " ".join(f"{v:.9g}" for v in gp)
                attrs["quat"] = " ".join(f"{v:.9g}" for v in [gq[3], gq[0], gq[1], gq[2]])
            geom_moves.append(attrs)

    # ---- rewrite the XML -----------------------------------------------------
    tree = ET.parse(MJCF)
    root = tree.getroot()
    parent_map = {c: p for p in tree.iter() for c in p}

    def find_body(name):
        for el in root.iter("body"):
            if el.get("name") == name:
                return el
        raise KeyError(name)

    waist = find_body(INTO)
    neck = find_body(FUSE[0])
    parent_map[neck].remove(neck)
    # explicit composed inertial (replaces any existing)
    for el in list(waist.findall("inertial")):
        waist.remove(el)
    fi = Itot
    inert = ET.Element("inertial", {
        "pos": " ".join(f"{v:.9g}" for v in com),
        "mass": f"{M:.9g}",
        "fullinertia": " ".join(f"{v:.9g}" for v in
                                [fi[0, 0], fi[1, 1], fi[2, 2], fi[0, 1], fi[0, 2], fi[1, 2]]),
    })
    waist.insert(0, inert)
    for attrs in geom_moves:
        waist.append(ET.Element("geom", attrs))
    # drop dangling references (contact excludes, sensors, sites on neck)
    for tag in ("exclude", "sensor", "site", "framepos", "framequat"):
        for el in list(root.iter(tag)):
            attrs = " ".join(el.attrib.values())
            if any(n.replace("_link", "") in attrs or n in attrs for n in FUSE):
                parent_map[el].remove(el)
    # fusestatic no longer needed
    comp = root.find("compiler")
    comp.attrib.pop("fusestatic", None)
    tree.write(MJCF)
    # framework body-name conventions (motion.yaml defaults, base_com DR):
    s = open(MJCF).read()
    s = s.replace('"pelvis_link"', '"pelvis"').replace('"waist_yaw_link"', '"torso_link"')
    open(MJCF, "w").write(s)
    # motion_lib Humanoid_Batch requires an <actuator> block (mjlab defines
    # actuators in Python, so upstream MJCF has none) -- add 1 motor per joint
    import mujoco as _mj
    _m = _mj.MjModel.from_xml_path(MJCF)
    _joints = [_mj.mj_id2name(_m, _mj.mjtObj.mjOBJ_JOINT, j) for j in range(1, _m.njnt)]
    s = open(MJCF).read()
    if "<actuator>" not in s:
        _blk = "  <actuator>\n" + "\n".join(
            f'    <motor name="{j}" joint="{j}" gear="1"/>' for j in _joints) + "\n  </actuator>\n"
        open(MJCF, "w").write(s.replace("</mujoco>", _blk + "</mujoco>"))

    # ---- verify --------------------------------------------------------------
    m2 = mujoco.MjModel.from_xml_path(MJCF)
    d2 = mujoco.MjData(m2)
    d2.qpos[:] = 0
    d2.qpos[3] = 1.0
    mujoco.mj_forward(m2, d2)
    n2 = m2.nbody - 1
    mass2 = float(sum(m2.body_mass[1:]))
    Mo = float(sum(model.body_mass[1:]))
    com_o = sum(model.body_mass[b] * data.xipos[b] for b in range(1, model.nbody)) / Mo
    com_n = sum(m2.body_mass[b] * d2.xipos[b] for b in range(1, m2.nbody)) / mass2
    err = np.linalg.norm(com_o - com_n)
    print(f"bodies {model.nbody-1} -> {n2}; mass {sum(model.body_mass[1:]):.4f} -> {mass2:.4f}; "
          f"whole-robot COM shift {err*1000:.6f} mm")
    ok = n2 == 24 and abs(mass2 - sum(model.body_mass[1:])) < 1e-4 and err < 1e-6
    print("PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
