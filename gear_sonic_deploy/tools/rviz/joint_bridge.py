#!/usr/bin/env python3
"""aimdk JointStateArray (leg/arm/waist/head) -> sensor_msgs/JointState.

``robot_state_publisher`` cannot consume ``aimdk_msgs``, so without this RViz
shows the URDF in its zero pose no matter what the robot is doing.

HAL joint names match the URDF exactly -- verified against
``/aima/hal/joint/leg/state``, which reports ``left_hip_pitch_joint``,
``left_knee_joint`` and so on. Names therefore pass straight through: no
ordering table, and no chance of a silently wrong pose from a mis-indexed
array. 31 joints across the four groups.

Only ever run ONE publisher on /joint_states. Two of them (this plus a
leftover zero-pose stand-in) makes the model flicker between upright and the
real pose, and ``ros2 topic hz`` shows a negative ``min`` from the interleaved
timestamps.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from aimdk_msgs.msg import JointStateArray

GROUPS = ["leg", "arm", "waist", "head"]

_qos = QoSProfile(depth=10)
_qos.reliability = ReliabilityPolicy.BEST_EFFORT
_qos.durability = DurabilityPolicy.VOLATILE


class Bridge(Node):
    def __init__(self) -> None:
        super().__init__("joint_bridge")
        self.latest: dict[str, list] = {}
        self.pub = self.create_publisher(JointState, "/joint_states", 10)
        for g in GROUPS:
            self.create_subscription(
                JointStateArray,
                f"/aima/hal/joint/{g}/state",
                lambda m, g=g: self.latest.__setitem__(g, m.joints),
                _qos,
            )
        # 20 Hz is plenty for visualisation; 50 Hz measured ~84% of a core on
        # a CPU that is also running the gait controller.
        self.create_timer(0.05, self.tick)
        self.get_logger().info("bridging " + ", ".join(GROUPS))

    def tick(self) -> None:
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        for joints in self.latest.values():
            for j in joints:
                m.name.append(j.name)
                m.position.append(j.position)
                m.velocity.append(j.velocity)
                m.effort.append(j.effort)
        if m.name:
            self.pub.publish(m)


if __name__ == "__main__":
    rclpy.init()
    rclpy.spin(Bridge())
