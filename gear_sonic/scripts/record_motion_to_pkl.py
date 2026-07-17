"""Capture a live sim motion (world pose + joints) into a motion-lib PKL.

Unlike ``lerobot_episode_to_motion_pkl.py`` (which only has joint commands and
therefore hardcodes root translation to zero), this records the robot's REAL
world trajectory by subscribing to the sim bridge's ground-truth ``robot_pose``
PUB (pelvis free-joint qpos) alongside the deploy's ``<robot>_debug`` PUB
(measured joint positions). It resamples both to a common ``--fps`` grid and
writes the standard motion-lib entry so locomotion (planner slow_walk / walk /
run, teleop, etc.) round-trips with correct world XY + heading.

    root_trans_offset (T,3)   <- robot_pose pelvis_qpos_wxyz[0:3]   (world xyz)
    root_rot          (T,4)   <- robot_pose pelvis_qpos_wxyz[3:7]   (wxyz -> xyzw)
    dof               (T,ND)  <- <robot>_debug body_q_measured      (MuJoCo order)
    pose_aa           (T,NB,3)   body0 = root rotvec; 1..ND = DOF_AXIS * dof
    smpl_joints       (T,24,3)   zeros placeholder
    fps               int

World pose comes from ``robot_pose`` (port 5570), published by the sim bridge
(x2_mujoco_ros_bridge.py for X2; add the equivalent to the G1 sim). The
``<robot>_debug`` stream (port 5557) carries IMU orientation + joints but NOT
world XY, which is why the separate ``robot_pose`` stream is required.

Usage (run the planner/teleop stack in another terminal first, then):

    .venv/bin/python gear_sonic/scripts/record_motion_to_pkl.py --robot g1 \\
        --duration 20 --out gear_sonic/data/motions/g1_recorded/slow_walk_001.pkl \\
        --motion-key slow_walk_001

Ctrl-C stops early and still writes whatever was captured.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial import transform
from scipy.spatial.transform import Rotation as R, Slerp

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (  # noqa: E402
    DOF_AXIS as G1_DOF_AXIS, NUM_BODIES as G1_NUM_BODIES, NUM_DOF as G1_NUM_DOF,
    X2_DOF_AXIS, X2_NUM_BODIES, X2_NUM_DOF,
)

# robot -> (debug topic, num_dof, num_bodies, dof_axis)
_ROBOTS = {
    "g1": ("g1_debug", G1_NUM_DOF, G1_NUM_BODIES, np.asarray(G1_DOF_AXIS, np.float32)),
    "x2": ("x2_debug", X2_NUM_DOF, X2_NUM_BODIES, np.asarray(X2_DOF_AXIS, np.float32)),
}
_BODY_Q_FIELDS = ("body_q_measured", "body_q")  # prefer measured; fall back


def build_entry(t_pose, pose7, t_dof, body_q, *, fps, num_dof, num_bodies, dof_axis):
    """Resample captured (pose, joints) onto a common fps grid -> motion entry.

    t_pose/pose7: (Np,) times and (Np,7) pelvis qpos [x,y,z, qw,qx,qy,qz].
    t_dof/body_q: (Nd,) times and (Nd, num_dof) MuJoCo-order joint positions.
    Times are a common monotonic clock (seconds). Returns a motion-lib entry.
    """
    t_pose, pose7 = np.asarray(t_pose, float), np.asarray(pose7, float)
    t_dof, body_q = np.asarray(t_dof, float), np.asarray(body_q, float)
    if len(t_pose) < 2 or len(t_dof) < 2:
        raise ValueError(f"need >=2 samples on each stream; got pose={len(t_pose)} dof={len(t_dof)}")
    if body_q.shape[1] != num_dof:
        raise ValueError(f"body_q has {body_q.shape[1]} dofs, expected {num_dof}")

    t0 = max(t_pose[0], t_dof[0])
    t1 = min(t_pose[-1], t_dof[-1])
    if t1 <= t0:
        raise ValueError("pose and dof streams do not overlap in time")
    T = max(2, int(round((t1 - t0) * fps)))
    grid = t0 + np.arange(T) / float(fps)

    # translation: linear interp
    root_trans = np.stack([np.interp(grid, t_pose, pose7[:, i]) for i in range(3)], axis=1)
    # rotation: slerp (robot_pose is wxyz -> scipy xyzw)
    quat_xyzw = pose7[:, [4, 5, 6, 3]]
    rot = Slerp(t_pose, R.from_quat(quat_xyzw))(grid)
    root_quat_xyzw = rot.as_quat().astype(np.float32)
    # joints: linear interp per dof
    dof = np.stack([np.interp(grid, t_dof, body_q[:, j]) for j in range(num_dof)], axis=1).astype(np.float32)

    pose_aa = np.zeros((T, num_bodies, 3), dtype=np.float32)
    pose_aa[:, 1:num_dof + 1, :] = dof_axis[None, :, :] * dof[:, :, None]
    pose_aa[:, 0, :] = rot.as_rotvec()

    return {
        "root_trans_offset": root_trans.astype(np.float32),
        "pose_aa": pose_aa,
        "dof": dof,
        "root_rot": root_quat_xyzw,
        "smpl_joints": np.zeros((T, 24, 3), dtype=np.float32),
        "fps": int(fps),
    }


def _capture(args, num_dof):
    import zmq
    import msgpack
    import msgpack_numpy as mnp
    from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose

    ctx = zmq.Context.instance()
    pose_sock = ctx.socket(zmq.SUB)
    pose_sock.connect(f"tcp://{args.host}:{args.pose_port}")
    pose_sock.setsockopt_string(zmq.SUBSCRIBE, "robot_pose")
    dbg_sock = ctx.socket(zmq.SUB)
    dbg_sock.connect(f"tcp://{args.host}:{args.debug_port}")
    dbg_sock.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    poller = zmq.Poller()
    poller.register(pose_sock, zmq.POLLIN)
    poller.register(dbg_sock, zmq.POLLIN)

    t_pose, pose7, t_dof, body_q = [], [], [], []
    print(f"[record] SUB robot_pose@{args.pose_port}, {args.topic}@{args.debug_port}; "
          f"capturing {args.duration}s (Ctrl-C to stop early)...")
    start = time.monotonic()
    try:
        while time.monotonic() - start < args.duration:
            for sock, _ in poller.poll(timeout=100):
                now = time.monotonic()
                if sock is pose_sock:
                    p = unpack_robot_pose(pose_sock.recv())
                    t_pose.append(now); pose7.append(p["pelvis_qpos_wxyz"])
                else:
                    raw = dbg_sock.recv()
                    d = msgpack.unpackb(raw[len(args.topic):], object_hook=mnp.decode, raw=False)
                    q = next((d[f] for f in _BODY_Q_FIELDS if f in d), None)
                    if q is not None:
                        t_dof.append(now); body_q.append(np.asarray(q, float)[:num_dof])
    except KeyboardInterrupt:
        print("\n[record] Ctrl-C -- stopping capture.")
    print(f"[record] captured pose={len(t_pose)} dof={len(t_dof)} samples "
          f"over {time.monotonic()-start:.1f}s")
    return t_pose, pose7, t_dof, body_q


_G1_CSV_HEADER = (
    "Frame,root_translateX,root_translateY,root_translateZ,"
    "root_rotateX,root_rotateY,root_rotateZ,"
    "left_hip_pitch_joint_dof,left_hip_roll_joint_dof,left_hip_yaw_joint_dof,"
    "left_knee_joint_dof,left_ankle_pitch_joint_dof,left_ankle_roll_joint_dof,"
    "right_hip_pitch_joint_dof,right_hip_roll_joint_dof,right_hip_yaw_joint_dof,"
    "right_knee_joint_dof,right_ankle_pitch_joint_dof,right_ankle_roll_joint_dof,"
    "waist_yaw_joint_dof,waist_roll_joint_dof,waist_pitch_joint_dof,"
    "left_shoulder_pitch_joint_dof,left_shoulder_roll_joint_dof,left_shoulder_yaw_joint_dof,"
    "left_elbow_joint_dof,left_wrist_roll_joint_dof,left_wrist_pitch_joint_dof,left_wrist_yaw_joint_dof,"
    "right_shoulder_pitch_joint_dof,right_shoulder_roll_joint_dof,right_shoulder_yaw_joint_dof,"
    "right_elbow_joint_dof,right_wrist_roll_joint_dof,right_wrist_pitch_joint_dof,right_wrist_yaw_joint_dof")


def write_g1_csv(entry: dict, path) -> None:
    """Write a 29-DOF G1 motion entry as a soma-retargeter G1 CSV (the input
    format for app/g1_csv_to_x2_csv.py): Frame, root_translate[cm],
    root_rotate[euler xyz deg], 29 named *_dof joints[deg]."""
    rt = entry["root_trans_offset"]        # (T,3) m
    rq = entry["root_rot"]                  # (T,4) xyzw
    dof = entry["dof"]                      # (T,29) rad
    if dof.shape[1] != 29:
        raise ValueError(f"--g1-csv expects 29 G1 DOF, got {dof.shape[1]}")
    T = len(dof)
    out = np.zeros((T, 36))
    out[:, 0] = np.arange(T)
    out[:, 1:4] = rt * 100.0               # m -> cm
    out[:, 4:7] = R.from_quat(rq).as_euler("xyz", degrees=True)
    out[:, 7:36] = np.rad2deg(dof)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(_G1_CSV_HEADER + "\n")
        np.savetxt(f, out, delimiter=",", fmt="%.6f")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot", required=True, choices=list(_ROBOTS))
    ap.add_argument("--out", required=True, type=Path, help="output .pkl path")
    ap.add_argument("--g1-csv", default=None, type=Path,
                    help="also write a retargeter-ready G1 CSV (--robot g1 only); "
                         "feed to soma-retargeter app/g1_csv_to_x2_csv.py")
    ap.add_argument("--motion-key", default=None, help="motion key (default: out stem)")
    ap.add_argument("--duration", type=float, default=20.0, help="capture seconds")
    ap.add_argument("--fps", default="50",
                    help="output fps grid(s); single (e.g. 50) or comma-list (e.g. 30,50) to "
                         "write BOTH grids from ONE capture -- 30fps = kplanner corpus, "
                         "50fps = SONIC. Both are resampled from the SAME raw samples (linear + "
                         "Slerp), so neither is a lossy downsample of the other. Multi-fps writes "
                         "siblings <out_stem>_<fps>fps<ext>.")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--debug-port", type=int, default=5557)
    ap.add_argument("--topic", default=None, help="debug topic (default <robot>_debug)")
    ap.add_argument("--pose-port", type=int, default=5570)
    args = ap.parse_args(argv)

    topic_default, num_dof, num_bodies, dof_axis = _ROBOTS[args.robot]
    if args.topic is None:
        args.topic = topic_default
    key = args.motion_key or args.out.stem

    t_pose, pose7, t_dof, body_q = _capture(args, num_dof)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fps_list = [int(x) for x in str(args.fps).split(",") if x.strip()]

    def _merge_save(out_path, entry):
        """Merge {key: entry} into out_path -- atomic write, .prev one-level undo,
        .corrupt guard. NEVER clobbers other clips."""
        existing: dict = {}
        if out_path.exists():
            try:
                loaded = joblib.load(out_path)
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception as exc:  # noqa: BLE001
                print(f"[record] WARNING: could not read existing {out_path} ({exc}); "
                      f"backing it up to {out_path}.corrupt before writing")
                out_path.replace(out_path.with_suffix(out_path.suffix + ".corrupt"))
        if key in existing:
            print(f"[record] NOTE: key {key!r} already in {out_path.name} -- replacing that clip")
        existing[key] = entry
        if out_path.exists():
            import shutil
            shutil.copy2(out_path, out_path.with_suffix(out_path.suffix + ".prev"))
        tmp = out_path.with_suffix(out_path.suffix + ".tmp")
        joblib.dump(existing, tmp, compress=3)
        tmp.replace(out_path)
        return len(existing)

    # One raw capture -> one grid per requested fps. Multi-fps writes siblings so a
    # single drive yields both the 30fps (kplanner corpus) and 50fps (SONIC) pkls.
    entry = None
    for fps in fps_list:
        entry = build_entry(t_pose, pose7, t_dof, body_q, fps=fps,
                            num_dof=num_dof, num_bodies=num_bodies, dof_axis=dof_axis)
        entry["x2_record_source"] = f"live:{args.robot}:{args.topic}+robot_pose"
        out_path = (args.out if len(fps_list) == 1
                    else args.out.with_name(f"{args.out.stem}_{fps}fps{args.out.suffix}"))
        n = _merge_save(out_path, entry)
        T = entry["dof"].shape[0]
        trav = np.linalg.norm(entry["root_trans_offset"][-1, :2] - entry["root_trans_offset"][0, :2])
        print(f"[record] wrote {out_path}  key={key!r}  {T} frames @ {fps}fps "
              f"({T/fps:.1f}s), traveled {trav:.2f} m  [{n} clip(s) in file]")
    if args.g1_csv:
        if args.robot != "g1":
            raise SystemExit("--g1-csv is only valid with --robot g1")
        write_g1_csv(entry, args.g1_csv)
        print(f"[record] wrote retargeter-ready G1 CSV {args.g1_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
