#!/usr/bin/env python3
"""
DDS <-> ZMQ adapter: lets the C++ deploy node drive an IsaacLab world.

Runs INSIDE the deploy docker (ROS Humble + aimdk_msgs sourced), replacing
x2_mujoco_ros_bridge.py in the `deploy_x2.sh sim --world isaaclab` topology.
The physics lives on the host in `x2_isaaclab_bridge.py --dds`; this process
only translates:

    ZMQ SUB tcp://<host>:5581 topic "il_state"   (from Isaac bridge)
        msgpack {t, qpos[31], qvel[31], effort[31], quat_wxyz[4], angvel[3]}
        -> 4x /aima/hal/joint/{leg,waist,arm,head}/state  (JointStateArray)
        -> /aima/hal/imu/torso/state                      (sensor_msgs/Imu)

    4x /aima/hal/joint/{...}/command (JointCommandArray, from deploy node)
        -> ZMQ PUB tcp://*:5582 topic "il_cmd"
        msgpack {t, pos[31], vel[31], ff[31], kp[31], kd[31]}

Message field semantics are copied 1:1 from x2_mujoco_ros_bridge.py (joint
names + validation, IMU wxyz orientation, body-local angular velocity,
covariance markers). Joint order on the wire is MuJoCo order, same as the
deploy expects.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

# Same group layout as x2_mujoco_ros_bridge.GROUPS (MuJoCo joint order).
GROUPS = [
    ("leg",   "/aima/hal/joint/leg",   0, 12),
    ("waist", "/aima/hal/joint/waist", 12, 3),
    ("arm",   "/aima/hal/joint/arm",   15, 14),
    ("head",  "/aima/hal/joint/head",  29, 2),
]
NUM_DOFS = 31


def load_mujoco_joint_names():
    """MUJOCO_JOINT_NAMES from eval_x2_mujoco.py without importing mujoco."""
    import importlib.util
    p = os.path.join(REPO_ROOT, "gear_sonic", "scripts", "eval_x2_mujoco.py")
    spec = importlib.util.spec_from_file_location("eval_x2_mujoco_names", p)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return list(mod.MUJOCO_JOINT_NAMES)
    except Exception:
        # Fallback: parse the source for the name list (keeps the adapter
        # runnable even if eval_x2_mujoco grows heavy imports).
        import re
        src = open(p).read()
        m = re.search(r"MUJOCO_JOINT_NAMES\s*=\s*\[(.*?)\]", src, re.S)
        names = re.findall(r'"([a-z_]+_joint)"', m.group(1))
        assert len(names) == NUM_DOFS, f"parsed {len(names)} joint names"
        return names


def main():
    ap = argparse.ArgumentParser(description="DDS<->ZMQ adapter for IsaacLab world")
    ap.add_argument("--isaac-host", default="127.0.0.1",
                    help="host where x2_isaaclab_bridge.py --dds runs")
    ap.add_argument("--state-port", type=int, default=5581)
    ap.add_argument("--cmd-port", type=int, default=5582)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    import msgpack
    import zmq
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from aimdk_msgs.msg import JointCommandArray, JointState, JointStateArray
    from sensor_msgs.msg import Imu as ImuMsg

    names = load_mujoco_joint_names()
    group_names = [(g, t, s, l, names[s:s + l]) for g, t, s, l in GROUPS]

    rclpy.init()
    node = rclpy.create_node("x2_isaaclab_dds_adapter")
    qos = qos_profile_sensor_data

    state_pubs = [node.create_publisher(JointStateArray, f"{t}/state", qos)
                  for _, t, _, _, _ in group_names]
    imu_pub = node.create_publisher(ImuMsg, "/aima/hal/imu/torso/state", qos)

    ctx = zmq.Context.instance()
    cmd_pub = ctx.socket(zmq.PUB)
    cmd_pub.bind(f"tcp://*:{args.cmd_port}")
    state_sub = ctx.socket(zmq.SUB)
    state_sub.connect(f"tcp://{args.isaac_host}:{args.state_port}")
    state_sub.setsockopt(zmq.SUBSCRIBE, b"il_state")
    state_sub.setsockopt(zmq.RCVTIMEO, 200)

    # --- deploy command ingestion -> ZMQ ------------------------------------
    lock = threading.Lock()
    cmd = {
        "pos": np.zeros(NUM_DOFS), "vel": np.zeros(NUM_DOFS),
        "ff": np.zeros(NUM_DOFS), "kp": np.zeros(NUM_DOFS), "kd": np.zeros(NUM_DOFS),
    }
    counters = {"cmd": 0, "state": 0, "first_cmd_logged": False}

    def make_cb(grp, start, length, expected):
        validated = [False]

        def cb(msg):
            if not validated[0]:
                if len(msg.joints) != length:
                    node.get_logger().fatal(
                        f"[{grp}] {len(msg.joints)} joints, expected {length}")
                    return
                for i, j in enumerate(msg.joints):
                    if j.name and j.name != expected[i]:
                        node.get_logger().fatal(
                            f"[{grp}] slot {i} is '{j.name}', expected '{expected[i]}'")
                        return
                validated[0] = True
                node.get_logger().info(f"[{grp}] joint name validation OK ({length})")
            with lock:
                for i, j in enumerate(msg.joints):
                    mj = start + i
                    cmd["pos"][mj] = j.position
                    cmd["vel"][mj] = j.velocity
                    cmd["ff"][mj] = j.effort
                    cmd["kp"][mj] = j.stiffness
                    cmd["kd"][mj] = j.damping
                counters["cmd"] += 1
                if not counters["first_cmd_logged"]:
                    counters["first_cmd_logged"] = True
                    node.get_logger().info(
                        f"first deploy command (group {grp}) -> forwarding to Isaac")
                payload = msgpack.packb({
                    "t": time.monotonic(),
                    "pos": cmd["pos"].tolist(), "vel": cmd["vel"].tolist(),
                    "ff": cmd["ff"].tolist(), "kp": cmd["kp"].tolist(),
                    "kd": cmd["kd"].tolist(),
                    "n_cmds": counters["cmd"],
                })
            cmd_pub.send_multipart([b"il_cmd", payload])
        return cb

    subs = [node.create_subscription(JointCommandArray, f"{t}/command",
                                     make_cb(g, s, l, n), qos)
            for g, t, s, l, n in group_names]
    _ = subs

    # --- ZMQ state -> DDS publishers ----------------------------------------
    def state_thread():
        seq = 0
        while rclpy.ok():
            try:
                _, payload = state_sub.recv_multipart()
            except zmq.Again:
                continue
            st = msgpack.unpackb(payload)
            qpos = st["qpos"]; qvel = st["qvel"]; eff = st["effort"]
            now = node.get_clock().now().to_msg()
            for (grp, _t, start, length, gnames), pub in zip(group_names, state_pubs):
                msg = JointStateArray()
                msg.header.stamp = now
                msg.header.frame_id = grp
                msg.header.sequence = seq & 0xFFFFFFFF
                msg.header.meas_stamp = now
                for i in range(length):
                    mj = start + i
                    js = JointState()
                    js.name = gnames[i]
                    js.position = float(qpos[mj])
                    js.velocity = float(qvel[mj])
                    js.effort = float(eff[mj])
                    js.coil_temp = 0
                    js.motor_temp = 0
                    js.motor_vol = 0
                    msg.joints.append(js)
                pub.publish(msg)
            imu = ImuMsg()
            imu.header.stamp = now
            imu.header.frame_id = "pelvis"
            w, x, y, z = st["quat_wxyz"]
            imu.orientation.w = float(w); imu.orientation.x = float(x)
            imu.orientation.y = float(y); imu.orientation.z = float(z)
            av = st["angvel"]
            imu.angular_velocity.x = float(av[0])
            imu.angular_velocity.y = float(av[1])
            imu.angular_velocity.z = float(av[2])
            imu.orientation_covariance[0] = -1.0
            imu.angular_velocity_covariance[0] = 0.0
            imu.linear_acceleration_covariance[0] = -1.0
            imu_pub.publish(imu)
            seq += 1
            counters["state"] += 1
            if args.verbose and seq % 1000 == 0:
                node.get_logger().info(
                    f"state msgs {counters['state']}, cmds {counters['cmd']}")

    t = threading.Thread(target=state_thread, daemon=True)
    t.start()
    print(f"[adapter] up: state SUB {args.isaac_host}:{args.state_port} -> /aima/hal/*, "
          f"commands -> PUB :{args.cmd_port}", flush=True)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == "__main__":
    main()
