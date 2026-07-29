#!/usr/bin/env python3
"""Forward-obstacle guard for the X2.

``blocked()`` is True while something sits inside ``STOP_M`` in the forward
wedge. Consumers zero forward/lateral velocity while blocked; yaw is left alone
on purpose so the operator can turn away instead of being trapped facing a wall.

FAIL-OPEN: stale or absent scan data means no clamping. A guard that freezes a
walking biped because its own sensor died is worse than none -- the operator
deadman is the real stop.

Run directly for a live readout::

    python3 scan_guard.py
"""
import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

STOP_M = 1.6
CLEAR_M = 1.6         # == STOP_M is a strict threshold; raise for hysteresis
ARC_DEG = 25.0        # forward wedge half-angle
# MEASURED, not derived: the scan is produced in base_link and "forward" does
# NOT land at 0 deg. A person 0.6 m directly ahead read -58 deg; the usable
# scan spans roughly -90..-20 deg and everything else is inf. Re-measure after
# any change to the converter's target_frame -- see OBSTACLE_GUARD.md.
ARC_CENTER_DEG = -58.0
STALE_S = 0.5
MIN_HITS = 4          # beams are ~1 deg apart; 4 rejects a single stray return
SELF_M = 0.25         # closer than this is the robot's own body

_qos = QoSProfile(depth=5)
_qos.reliability = ReliabilityPolicy.BEST_EFFORT
_qos.durability = DurabilityPolicy.VOLATILE


class ScanGuard(Node):
    def __init__(self) -> None:
        super().__init__("scan_guard")
        self._blocked = False
        self._dist = float("inf")
        self._stamp = 0.0
        self.create_subscription(LaserScan, "/scan", self._cb, _qos)

    def _cb(self, m: LaserScan) -> None:
        half = math.radians(ARC_DEG)
        center = math.radians(ARC_CENTER_DEG)
        near = []
        for i, r in enumerate(m.ranges):
            if not math.isfinite(r) or r < m.range_min or r > m.range_max:
                continue
            if r < SELF_M:
                continue
            a = m.angle_min + i * m.angle_increment
            if abs(a - center) <= half:
                near.append(r)
        self._dist = min(near) if near else float("inf")
        # Hysteresis when CLEAR_M > STOP_M: without it an object sitting right
        # at the threshold flips the state ~10x/sec, re-tripping the latch and
        # re-firing the spoken warning on every transition.
        thr = CLEAR_M if self._blocked else STOP_M
        self._blocked = sum(1 for r in near if r < thr) >= MIN_HITS
        self._stamp = self.get_clock().now().nanoseconds / 1e9

    def blocked(self) -> bool:
        now = self.get_clock().now().nanoseconds / 1e9
        if now - self._stamp > STALE_S:
            return False          # fail open
        return self._blocked

    def distance(self) -> float:
        return self._dist


def start() -> ScanGuard:
    """Spin a ScanGuard on a daemon thread; returns the node."""
    if not rclpy.ok():
        rclpy.init()
    g = ScanGuard()
    threading.Thread(target=rclpy.spin, args=(g,), daemon=True).start()
    return g


if __name__ == "__main__":
    import time

    g = start()
    while True:
        d = g.distance()
        shown = "inf" if math.isinf(d) else f"{d:.2f}m"
        print(f"\rblocked={str(g.blocked()):5} nearest={shown}      ",
              end="", flush=True)
        time.sleep(0.2)
