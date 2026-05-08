#!/usr/bin/env python3
"""Side-by-side kinematic playback of two recorder npz files.

Builds a temporary MJCF that contains two copies of the X2 Ultra body
(prefixed ``real_`` and ``sim_``), places them ``--separation`` metres
apart in the y-axis, disables gravity, removes contacts, and replays
the recorded ``state_pos`` joint trajectories from each npz onto its
robot. The IMU quaternion drives the floating-base orientation so torso
tilt and full-body yaw are visible side-by-side.

Use this to visually compare a real-robot deploy against its MuJoCo
parity counterpart for the same checkpoint + motion playlist (Anchor C
or Anchor D in ``data/sim_to_real_anchors/``).

Controls:
    SPACE         pause / resume
    R             restart from frame 0
    LEFT / RIGHT  scrub +/- 10 frames

Usage:
    conda run -n env_isaaclab --no-capture-output python \\
        gear_sonic_deploy/scripts/play_npz_dual_kinematic.py \\
        --real data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/real.npz \\
        --sim  data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/sim.npz \\
        --separation 1.5 \\
        --speed 1.0
"""
from __future__ import annotations

import argparse
import copy
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import mujoco.viewer
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MJCF = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
)

# Recorder writes per-group state in fixed name order (see
# x2_record_real_run.py). Hard-coding these avoids loading the
# `joint_names_*` pickled object arrays, which break across numpy
# major versions (numpy._core symbol).
GROUP_JOINTS = {
    "leg": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    ],
    "waist": ["waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"],
    "arm": [
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint", "left_elbow_joint",
        "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    ],
    "head": ["head_yaw_joint", "head_pitch_joint"],
}


# Names that exist in each robot subtree and need prefixing. Anything in
# <asset>, <default>, <option>, etc. is shared and not prefixed.
PREFIX_REF_TYPES = {
    "joint", "body", "geom", "site", "sensor", "frame", "camera", "light"
}
# Attributes that hold name references that must also get prefixed.
NAME_REF_ATTRS = {
    "joint", "joint1", "joint2", "body1", "body2", "site",
    "site1", "site2", "geom", "geom1", "geom2", "frame", "ref", "objname",
    "name", "actuator", "tendon",
}


def _prefix_subtree(elem: ET.Element, prefix: str) -> None:
    """Walk an MJCF subtree and prefix every name + name-reference."""
    for node in elem.iter():
        for k in list(node.attrib.keys()):
            if k in NAME_REF_ATTRS:
                v = node.attrib[k]
                # Skip empty, numeric, or already-prefixed values.
                if not v or v.startswith(prefix):
                    continue
                # Skip class/childclass/material references (shared via <default>).
                node.attrib[k] = f"{prefix}{v}"
        # Strip any 'class' / 'childclass' refs that point at robot-local
        # default classes? They're shared (single <default class="x2"> tree),
        # so we leave them alone.


def _build_dual_mjcf(separation: float, work_dir: Path) -> Path:
    """Build /tmp/x2_dual_kinematic.xml with two prefixed robot trees."""
    tree = ET.parse(SOURCE_MJCF)
    root = tree.getroot()

    # Resolve meshdir to absolute so the temp file works from anywhere.
    compiler = root.find("compiler")
    if compiler is not None:
        meshdir = compiler.get("meshdir", "")
        if meshdir and not Path(meshdir).is_absolute():
            abs_meshdir = (SOURCE_MJCF.parent / meshdir).resolve()
            compiler.set("meshdir", str(abs_meshdir))

    # Disable gravity for kinematic playback.
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 0")
    # Remove contact computations — we never integrate forward, but this
    # keeps the visualiser cheap.
    option.set("iterations", "1")

    # Drop actuators and sensors (we don't need them for kinematic playback,
    # and they reference joint names that we'd otherwise need to duplicate).
    for tag in ("actuator", "sensor", "tendon", "equality", "contact"):
        for child in list(root.findall(tag)):
            root.remove(child)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("No <worldbody> in x2_ultra.xml")

    # Locate the pelvis subtree (the robot root).
    pelvis = None
    for body in list(worldbody.findall("body")):
        if body.get("name") == "pelvis":
            pelvis = body
            break
    if pelvis is None:
        raise RuntimeError("No <body name='pelvis'> in worldbody")

    # We'll keep the original ground plane (light/geom) and remove just
    # the pelvis body from worldbody, then append two prefixed copies.
    worldbody.remove(pelvis)

    # Build real_ and sim_ copies, each placed at y = +/- separation/2.
    # Sim robot is colour-coded near-white so it's distinguishable from
    # real (which keeps the X2 stock colours: dark grey body + white
    # limbs + orange feet). Visual geoms (group=1) get the override;
    # collision geoms keep their colours so they don't appear in the
    # visualiser by default.
    # Sim robot is rendered as a translucent ghost regardless of whether
    # the two robots are overlaid (--separation 0) or side-by-side. This
    # makes the sim instantly identifiable: solid X2 = real, ghostly
    # white-and-blue = sim.
    SIM_BODY_RGBA = "0.95 0.95 0.95 0.30"   # near-white, 30% opaque
    SIM_FOOT_RGBA = "0.20 0.55 0.95 0.45"   # blue feet, 45% opaque
    half = float(separation) / 2.0
    for prefix, y_offset in [("real_", -half), ("sim_", +half)]:
        clone = copy.deepcopy(pelvis)
        clone.set("pos", f"0 {y_offset:+.4f} 0.95")
        _prefix_subtree(clone, prefix)
        if prefix == "sim_":
            for geom in clone.iter("geom"):
                gclass = geom.get("class", "")
                group = geom.get("group", "")
                if gclass == "visual" or group == "1":
                    geom.set("rgba", SIM_BODY_RGBA)
                elif gclass == "foot":
                    geom.set("rgba", SIM_FOOT_RGBA)
        worldbody.append(clone)

    # Add visual labels (small text-like geoms above each robot's head)
    # via simple coloured spheres. The viewer doesn't support text labels
    # without sites + custom rendering, so colour is the cue.
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path = work_dir / "x2_dual_kinematic.xml"
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path


def _load_run(path: Path) -> Dict[str, np.ndarray]:
    """Load only the numeric arrays from a recorder npz.

    Skips object arrays (joint_names_*, mc_mode_str, meta_json) which
    are pickled and not portable across numpy major versions; we
    hard-code the joint name order in GROUP_JOINTS instead.
    """
    out: Dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as f:
        for k in f.files:
            if k.startswith(("joint_names_", "mc_mode_str")) or k == "meta_json":
                continue
            try:
                out[k] = f[k]
            except ValueError:
                # object array slipped through; skip
                continue
    return out


def _detect_control_window(raw: dict) -> Tuple[float, float]:
    """Use leg-knee kp to find the longest contiguous CONTROL block."""
    t = raw["t_cmd_leg"]
    kp = raw["cmd_kp_leg"][:, 3]
    n = t.size
    pol = float(np.median(kp[n // 4 : 3 * n // 4]))
    mask = np.abs(kp - pol) < 5.0
    if not mask.any():
        return float(t[0]), float(t[-1])
    diff = np.diff(mask.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if mask[0]:
        starts = np.concatenate(([0], starts))
    if mask[-1]:
        ends = np.concatenate((ends, [n]))
    L = ends - starts
    b = int(np.argmax(L))
    return float(t[starts[b]]), float(t[ends[b] - 1])


def _gather_joint_state(raw: dict, t_grid: np.ndarray, jname_to_qpos_idx: Dict[str, int]) -> np.ndarray:
    """Resample state_pos for all four limb groups onto t_grid.

    Returns an array of shape (T, qpos_dim) with non-joint qpos slots
    untouched (left zero -- caller is expected to set freejoint slots).
    """
    qdim = max(jname_to_qpos_idx.values()) + 1
    out = np.zeros((t_grid.size, qdim), dtype=np.float64)
    for grp, names in GROUP_JOINTS.items():
        t_g = raw[f"t_state_{grp}"].astype(np.float64)
        v_g = raw[f"state_pos_{grp}"].astype(np.float64)
        for j, jname in enumerate(names):
            if jname not in jname_to_qpos_idx:
                continue
            out[:, jname_to_qpos_idx[jname]] = np.interp(t_grid, t_g, v_g[:, j])
    return out


def _gather_imu_quat(raw: dict, t_grid: np.ndarray) -> np.ndarray:
    """Resample IMU quaternion onto t_grid. Returns (T, 4) wxyz."""
    t_imu = raw["t_imu"].astype(np.float64)
    q = raw["imu_quat_wxyz"].astype(np.float64)
    out = np.zeros((t_grid.size, 4), dtype=np.float64)
    for c in range(4):
        out[:, c] = np.interp(t_grid, t_imu, q[:, c])
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    norm[norm < 1e-9] = 1.0
    return out / norm


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--real", required=True, type=Path,
                    help="Path to real-robot run.npz")
    ap.add_argument("--sim", required=True, type=Path,
                    help="Path to MuJoCo run.npz")
    ap.add_argument("--separation", type=float, default=1.5,
                    help="Lateral separation between the two robots, metres "
                         "(default 1.5; positive y for sim, negative y for real)")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="Playback speed multiplier (default 1x)")
    ap.add_argument("--fps", type=float, default=60.0,
                    help="Playback grid Hz (default 60)")
    ap.add_argument("--start-skip", type=float, default=0.0,
                    help="Skip first N seconds of CONTROL (default 0; useful "
                         "to drop the initial-pose-mismatch transient)")
    ap.add_argument("--end-trim", type=float, default=0.0,
                    help="Trim last N seconds of CONTROL (default 0)")
    ap.add_argument("--keep-base-translation", action="store_true",
                    help="Allow base x/y to drift (default: lock at the "
                         "starting offset; only orientation drifts, so the "
                         "two robots stay side-by-side rather than walking "
                         "off in different directions in MuJoCo's model — "
                         "remember the recorder does NOT capture base "
                         "translation, so this flag is informational only).")
    ap.add_argument("--lock-yaw", action="store_true",
                    help="Zero out ALL IMU yaw on both robots so they keep "
                         "facing forward throughout the playback (still "
                         "applies roll/pitch). Removes turning motion -- "
                         "use only when you want to compare in-place gait "
                         "without any base rotation.")
    ap.add_argument("--no-imu-quat", action="store_true",
                    help="Ignore IMU quaternion entirely; both robots stay "
                         "perfectly upright. Cleanest for showing pure "
                         "joint kinematic differences without any base "
                         "orientation.")
    ap.add_argument("--no-anchor-feet", action="store_true",
                    help="Disable per-frame ground-anchoring of the lower "
                         "foot. Default: each frame the base z is adjusted "
                         "so the lowest foot sole sits at z=0, giving a "
                         "realistic 'walking on the ground' look. Disable "
                         "this to lock the base at z=0.95 (the legs will "
                         "swing freely above/below the ground plane).")
    ap.add_argument("--raw-imu-yaw", action="store_true",
                    help="Use raw IMU yaw as recorded (default is to "
                         "subtract each robot's starting yaw so both begin "
                         "facing the same direction, preserving the natural "
                         "turning motion). Set this flag to see the actual "
                         "world-frame heading divergence between sim and "
                         "real (e.g. Anchor B's 166-deg base-trajectory gap).")
    args = ap.parse_args()

    if not args.real.exists():
        raise SystemExit(f"--real does not exist: {args.real}")
    if not args.sim.exists():
        raise SystemExit(f"--sim does not exist: {args.sim}")

    work_dir = Path("/tmp/x2_dual_kinematic")
    mjcf_path = _build_dual_mjcf(args.separation, work_dir)
    print(f"[mjcf] wrote dual model: {mjcf_path}")

    mj_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    mj_data = mujoco.MjData(mj_model)

    # Map joint name -> qpos index (for hinges, qpos has 1 entry per joint;
    # for the freejoint at root, 7 entries starting at the base index).
    jname_to_qpos_idx: Dict[str, int] = {}
    for j in range(mj_model.njnt):
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = int(mj_model.jnt_qposadr[j])
        jtype = int(mj_model.jnt_type[j])
        if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            jname_to_qpos_idx[jname] = adr
        # freejoint -> handled separately below

    # Locate the freejoint qpos slots for each pelvis (real_ and sim_).
    free_slots: Dict[str, int] = {}
    for j in range(mj_model.njnt):
        jname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if int(mj_model.jnt_type[j]) == mujoco.mjtJoint.mjJNT_FREE:
            adr = int(mj_model.jnt_qposadr[j])
            for prefix in ("real_", "sim_"):
                if jname.startswith(prefix):
                    free_slots[prefix.rstrip("_")] = adr
                    break
    if "real" not in free_slots or "sim" not in free_slots:
        raise SystemExit(
            f"Could not find freejoint qpos slots; got {free_slots}. "
            f"Likely the prefix scheme failed."
        )

    # Per-robot joint name -> qpos idx (with prefix stripped for lookup).
    real_jmap: Dict[str, int] = {}
    sim_jmap: Dict[str, int] = {}
    for jname, adr in jname_to_qpos_idx.items():
        if jname.startswith("real_"):
            real_jmap[jname[len("real_"):]] = adr
        elif jname.startswith("sim_"):
            sim_jmap[jname[len("sim_"):]] = adr

    print(f"[joints] real robot: {len(real_jmap)} hinges; sim robot: {len(sim_jmap)} hinges")

    # Load both runs and detect CONTROL windows.
    real_raw = _load_run(args.real)
    sim_raw  = _load_run(args.sim)
    cs_r, ce_r = _detect_control_window(real_raw)
    cs_s, ce_s = _detect_control_window(sim_raw)
    dur = min(ce_r - cs_r, ce_s - cs_s) - args.start_skip - args.end_trim
    if dur <= 0:
        raise SystemExit("Compared duration is non-positive; check --start-skip/--end-trim")

    fps = float(args.fps)
    nT = int(np.floor(dur * fps))
    t_rel = np.arange(nT) / fps  # seconds since CONTROL start (post-skip)
    t_real = cs_r + args.start_skip + t_rel
    t_sim  = cs_s + args.start_skip + t_rel

    print(f"[align] real CONTROL {cs_r:.2f}..{ce_r:.2f}s, sim CONTROL {cs_s:.2f}..{ce_s:.2f}s")
    print(f"[align] playback duration: {dur:.2f}s ({nT} frames @ {fps} Hz)")

    real_state = _gather_joint_state(real_raw, t_real, real_jmap)
    sim_state  = _gather_joint_state(sim_raw,  t_sim,  sim_jmap)
    real_quat  = _gather_imu_quat(real_raw, t_real)
    sim_quat   = _gather_imu_quat(sim_raw,  t_sim)

    def _apply_yaw_offset(qs: np.ndarray, yaw_offset: np.ndarray) -> np.ndarray:
        """Pre-multiply qs by Rz(yaw_offset). yaw_offset is per-frame radians."""
        w, x, y, z = qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3]
        half = yaw_offset / 2.0
        qz_w = np.cos(half)
        qz_z = np.sin(half)
        nw = qz_w * w - qz_z * z
        nx = qz_w * x - qz_z * y
        ny = qz_w * y + qz_z * x
        nz = qz_w * z + qz_z * w
        out = np.stack([nw, nx, ny, nz], axis=1)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        n[n < 1e-9] = 1.0
        return out / n

    def _quat_yaw(qs: np.ndarray) -> np.ndarray:
        w, x, y, z = qs[:, 0], qs[:, 1], qs[:, 2], qs[:, 3]
        return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    if args.no_imu_quat:
        real_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (real_quat.shape[0], 1))
        sim_quat  = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (sim_quat.shape[0], 1))
        print("[viz] IMU quat disabled; both robots fixed upright")
    elif args.lock_yaw:
        real_yaw = _quat_yaw(real_quat)
        sim_yaw  = _quat_yaw(sim_quat)
        real_quat = _apply_yaw_offset(real_quat, -real_yaw)
        sim_quat  = _apply_yaw_offset(sim_quat,  -sim_yaw)
        print("[viz] base yaw fully locked on both robots (no turning visible)")
    elif not args.raw_imu_yaw:
        # Default: subtract each robot's starting yaw so both begin facing
        # forward. Preserves natural turning motion through the run.
        real_yaw0 = float(_quat_yaw(real_quat[:1])[0])
        sim_yaw0  = float(_quat_yaw(sim_quat[:1])[0])
        real_quat = _apply_yaw_offset(real_quat, np.full(real_quat.shape[0], -real_yaw0))
        sim_quat  = _apply_yaw_offset(sim_quat,  np.full(sim_quat.shape[0],  -sim_yaw0))
        print(f"[viz] start-yaw aligned: real -={np.degrees(real_yaw0):+.1f}deg, "
              f"sim -={np.degrees(sim_yaw0):+.1f}deg; turning motion preserved")
    else:
        print("[viz] using raw IMU yaw as recorded (sim/real heading may diverge)")

    initial_base_z = 0.95
    half = float(args.separation) / 2.0

    # Pre-resolve foot-link body ids for each robot so we can ground them
    # frame-by-frame. The feet are spheres of radius ~0.005 mounted at
    # local z=-0.068 inside the ankle_roll_link, so the lowest point of
    # the foot is roughly (ankle_roll_link world z) - 0.073.
    FOOT_SOLE_OFFSET = 0.085
    foot_bids = {}
    for prefix in ("real_", "sim_"):
        lid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{prefix}left_ankle_roll_link")
        rid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{prefix}right_ankle_roll_link")
        if lid < 0 or rid < 0:
            print(f"[warn] could not find ankle_roll_link bodies for {prefix}; "
                  f"feet anchoring disabled for this robot.")
        foot_bids[prefix.rstrip("_")] = (lid, rid)

    def _set_robot(prefix: str, idx: int, y_offset: float,
                   quat_arr: np.ndarray, jmap: Dict[str, int],
                   state_arr: np.ndarray) -> None:
        a = free_slots[prefix]
        mj_data.qpos[a + 0] = 0.0
        mj_data.qpos[a + 1] = y_offset
        mj_data.qpos[a + 2] = initial_base_z
        mj_data.qpos[a + 3] = quat_arr[idx, 0]
        mj_data.qpos[a + 4] = quat_arr[idx, 1]
        mj_data.qpos[a + 5] = quat_arr[idx, 2]
        mj_data.qpos[a + 6] = quat_arr[idx, 3]
        for _, qidx in jmap.items():
            mj_data.qpos[qidx] = state_arr[idx, qidx]

    def _anchor_feet(prefix: str) -> None:
        lid, rid = foot_bids[prefix]
        if lid < 0 or rid < 0:
            return
        lz = float(mj_data.xpos[lid, 2])
        rz = float(mj_data.xpos[rid, 2])
        sole_z = min(lz, rz) - FOOT_SOLE_OFFSET
        # Shift base z so the lower sole sits at z=0.
        a = free_slots[prefix]
        mj_data.qpos[a + 2] -= sole_z

    def apply_frame(idx: int) -> None:
        _set_robot("real", idx, -half, real_quat, real_jmap, real_state)
        _set_robot("sim",  idx, +half, sim_quat,  sim_jmap,  sim_state)
        mujoco.mj_forward(mj_model, mj_data)

        if not args.no_anchor_feet:
            _anchor_feet("real")
            _anchor_feet("sim")
            mujoco.mj_forward(mj_model, mj_data)

    paused = [False]
    cur_frame = [0]

    def key_callback(keycode: int) -> None:
        if keycode == 32:  # SPACE
            paused[0] = not paused[0]
        elif keycode == ord("R"):
            cur_frame[0] = 0
        elif keycode == 262:  # RIGHT
            cur_frame[0] = min(nT - 1, cur_frame[0] + 10)
        elif keycode == 263:  # LEFT
            cur_frame[0] = max(0, cur_frame[0] - 10)

    apply_frame(0)
    print(
        "\n=== Dual kinematic playback ===\n"
        f"  real robot: y = {-half:+.2f} m  ({args.real.name})\n"
        f"  sim  robot: y = {+half:+.2f} m  ({args.sim.name})\n"
        "  SPACE pause | R restart | LEFT/RIGHT scrub +/-10 frames\n",
        flush=True,
    )

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 90
        viewer.cam.elevation = -15
        viewer.cam.distance = max(3.5, args.separation * 2.5)
        viewer.cam.lookat[:] = [0.0, 0.0, initial_base_z]

        frame_dt = 1.0 / (fps * max(args.speed, 1e-6))
        wall_start = time.time()
        wall_origin = cur_frame[0]

        while viewer.is_running():
            if paused[0]:
                viewer.sync()
                time.sleep(0.02)
                continue

            elapsed = time.time() - wall_start
            target = wall_origin + int(elapsed / frame_dt)
            target = target % nT
            if target != cur_frame[0]:
                cur_frame[0] = target
                apply_frame(target)
            viewer.sync()
            time.sleep(min(frame_dt, 0.02))


if __name__ == "__main__":
    main()
