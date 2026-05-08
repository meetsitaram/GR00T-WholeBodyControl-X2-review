#!/usr/bin/env python3
"""Live monitor for the X2 deploy node's published joint commands.

Sibling to ``x2_preflight.py``. Where preflight is a one-shot "is the world
ready" gate, this is a continuous "is the policy doing reasonable things"
watchdog meant to run alongside an active deploy (especially in --dry-run,
when you want to manually nudge the robot on the gantry and observe how
the policy reacts without risking torque commands taking effect).

KNOWN BUG (2026-04-22, observed during X2 sonic bring-up): the deploy's
500 Hz writer republishes each 50 Hz policy tick 10x via zero-order hold.
That makes most consecutive samples on /aima/hal/joint/*/command identical,
so this monitor's per-sample step calculation only sees the small jumps at
policy-tick boundaries (typically <= --max-target-dev clamp, i.e. a few
hundredths of a rad). Combined with the summary's ``{:.3f}`` formatting,
that prints ``max|step|=0.000`` for joints that did move several rad over
the run. Two fixes needed:
  1. Detect the writer ZOH and only step-compare across distinct values
     (or sample at 50 Hz to match policy-tick boundaries).
  2. Bump summary precision to ``{:.4f}`` so sub-millirad steps aren't
     hidden by rounding.
Until then, prefer the deploy's own ``tick.csv`` / ``target_pos.csv`` for
post-mortem analysis -- the live monitor's deviation/NaN/dry-run-leak
checks are still sound, but ``max|step|`` is unreliable.

Subscribes to all four ``/aima/hal/joint/{leg,waist,arm,head}/command``
topics that the deploy publishes on, and on every received message runs
these checks per-joint:

  1. NaN / Inf in target_pos, target_vel, stiffness, or damping.
  2. ``|target_pos - default_angle|`` exceeds ``--max-deviation`` (default
     0.5 rad ~ 29 deg). Catches the policy commanding extreme poses.
  3. ``|target_pos - prev_target_pos|`` exceeds ``--max-step`` (default
     0.3 rad in one 20ms tick ~ 15 rad/s commanded angular velocity).
     Catches discontinuous / jerky actions.
  4. Non-zero ``stiffness`` or ``damping`` when ``--expect-dry-run`` is
     set. (Real dry-run published commands have both fields zeroed; if
     you see them non-zero something is wrong upstream.)

WARN lines are throttled to one per (joint, check) per second so a stuck
condition prints only ~1 line/s instead of 50 lines/s. A 1 Hz status line
prints the per-family command rate + worst-case deviation + worst-case
step. On Ctrl-C, a final summary table dumps per-joint maxima.

Usage (run inside the docker_x2 container, in a SECOND terminal while
``deploy_x2.sh --dry-run`` is running in the FIRST terminal):

  ./gear_sonic_deploy/docker_x2/get_x2_sonic_ready.sh -- \
      python3 gear_sonic_deploy/scripts/x2_action_monitor.py --expect-dry-run

Flags:
  --max-deviation RAD   warn if |target-default| > RAD (default 0.5)
  --max-step RAD        warn if |target_t - target_{t-1}| > RAD (default 0.3)
  --expect-dry-run      additionally warn if stiffness or damping != 0
  --quiet               suppress the 1 Hz status line; only print WARNs
"""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from aimdk_msgs.msg import JointCommandArray  # type: ignore


# ─────────────────────────────────────────────────────────────────────────
# Canonical joint metadata. MUST stay in lockstep with
# gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp
# (hardcoded here so the monitor has zero build dependency on the deploy
# colcon workspace -- it just needs ``aimdk_msgs`` from the container).
# TODO(rename): once the mujoco_joint_names rename happens (see TODO in
# codegen_x2_policy_parameters.py), update GROUPS keys / labels too.
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GroupSpec:
    topic: str
    joint_names: Tuple[str, ...]
    defaults: Tuple[float, ...]


GROUPS: Dict[str, GroupSpec] = {
    "leg": GroupSpec(
        topic="/aima/hal/joint/leg/command",
        joint_names=(
            "left_hip_pitch_joint",   "left_hip_roll_joint",   "left_hip_yaw_joint",
            "left_knee_joint",        "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint",  "right_hip_roll_joint",  "right_hip_yaw_joint",
            "right_knee_joint",       "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ),
        defaults=(
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
            -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        ),
    ),
    "waist": GroupSpec(
        topic="/aima/hal/joint/waist/command",
        joint_names=("waist_yaw_joint", "waist_pitch_joint", "waist_roll_joint"),
        defaults=(0.0, 0.0, 0.0),
    ),
    "arm": GroupSpec(
        topic="/aima/hal/joint/arm/command",
        joint_names=(
            "left_shoulder_pitch_joint",  "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",    "left_elbow_joint",
            "left_wrist_yaw_joint",       "left_wrist_pitch_joint",
            "left_wrist_roll_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",   "right_elbow_joint",
            "right_wrist_yaw_joint",      "right_wrist_pitch_joint",
            "right_wrist_roll_joint",
        ),
        defaults=(
            0.2,  0.2,  0.0, -0.6, 0.0, 0.0, 0.0,
            0.2, -0.2,  0.0, -0.6, 0.0, 0.0, 0.0,
        ),
    ),
    "head": GroupSpec(
        topic="/aima/hal/joint/head/command",
        joint_names=("head_yaw_joint", "head_pitch_joint"),
        defaults=(0.0, 0.0),
    ),
}

# Per-(group, joint_name, check_kind) WARN throttle: emit at most once per
# THROTTLE_WINDOW_S to keep the console legible if a condition stays true.
THROTTLE_WINDOW_S = 1.0


# ─────────────────────────────────────────────────────────────────────────
# Per-joint running state for a single command stream.
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class JointStats:
    name: str
    default: float
    last_target: float = math.nan
    max_abs_dev: float = 0.0  # worst |target - default| seen so far
    max_abs_step: float = 0.0  # worst |target_t - target_{t-1}| seen so far


@dataclass
class GroupStats:
    spec: GroupSpec
    joints: List[JointStats]
    msg_count: int = 0
    last_msg_t: float = 0.0
    cmd_hz_estimate: float = 0.0
    # Throttle: maps (joint_name, check_kind) -> last emission monotonic time.
    last_warn_t: Dict[Tuple[str, str], float] = field(default_factory=dict)


def _color(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if sys.stdout.isatty() else s


def _yellow(s: str) -> str:
    return _color("33", s)


def _red(s: str) -> str:
    return _color("31;1", s)


def _green(s: str) -> str:
    return _color("32", s)


def _grey(s: str) -> str:
    return _color("90", s)


# ─────────────────────────────────────────────────────────────────────────
# Monitor node
# ─────────────────────────────────────────────────────────────────────────
class ActionMonitor(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("x2_action_monitor")
        self._args = args
        self._t0 = time.monotonic()

        self._stats: Dict[str, GroupStats] = {
            name: GroupStats(
                spec=spec,
                joints=[
                    JointStats(name=jn, default=df)
                    for jn, df in zip(spec.joint_names, spec.defaults)
                ],
            )
            for name, spec in GROUPS.items()
        }

        # Match the QoS the C++ deploy publisher uses for HAL command streams.
        # aimdk_io.cpp creates the publishers with rclcpp::SensorDataQoS(), i.e.
        # BEST_EFFORT + KEEP_LAST(10) + VOLATILE. If we declared RELIABLE here
        # DDS would refuse to match (a reliable subscriber cannot accept a
        # best-effort publisher) and we would silently never receive a single
        # message -- which is exactly the failure mode that gives a status line
        # full of 0.0 Hz / 0.00 rad on a fully-functional dry run, accompanied
        # by "incompatible QoS ... Last incompatible policy: RELIABILITY"
        # warnings on stderr.
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
        )

        for group_name, gs in self._stats.items():
            self.create_subscription(
                JointCommandArray,
                gs.spec.topic,
                lambda msg, gn=group_name: self._on_cmd(gn, msg),
                qos,
            )

        # 1 Hz status timer.
        if not args.quiet:
            self.create_timer(1.0, self._on_status)

        self.get_logger().info(
            f"x2_action_monitor up. "
            f"max_deviation={args.max_deviation:.3f} rad, "
            f"max_step={args.max_step:.3f} rad, "
            f"expect_dry_run={args.expect_dry_run}"
        )
        self.get_logger().info(
            "Subscribed to: "
            + ", ".join(g.spec.topic for g in self._stats.values())
        )

    # ─────────────────────────────────────────────────────────────────
    # Per-message handler
    # ─────────────────────────────────────────────────────────────────
    def _on_cmd(self, group: str, msg: JointCommandArray) -> None:
        gs = self._stats[group]
        now = time.monotonic()

        # cmd_hz EWMA: instantaneous = 1 / dt_since_last_msg, smoothed alpha=0.1.
        if gs.last_msg_t > 0.0:
            inst_hz = 1.0 / max(now - gs.last_msg_t, 1e-6)
            gs.cmd_hz_estimate = (
                0.9 * gs.cmd_hz_estimate + 0.1 * inst_hz
                if gs.cmd_hz_estimate > 0.0 else inst_hz
            )
        gs.last_msg_t = now
        gs.msg_count += 1

        if len(msg.joints) != len(gs.joints):
            self._warn_throttled(
                gs, "_group_", "joint_count",
                f"{group}: command msg has {len(msg.joints)} joints, "
                f"expected {len(gs.joints)}",
            )
            return

        for i, (jstat, jmsg) in enumerate(zip(gs.joints, msg.joints)):
            # Sanity: name mismatch (the C++ deploy validates this once at
            # startup, but we double-check here in case the firmware shuffles
            # ordering between the two control streams).
            if jmsg.name and jmsg.name != jstat.name:
                self._warn_throttled(
                    gs, jstat.name, "name_drift",
                    f"{group}[{i}] name '{jmsg.name}' != expected "
                    f"'{jstat.name}' (firmware joint shuffle?)",
                )

            self._check_joint(gs, jstat, jmsg)

    # ─────────────────────────────────────────────────────────────────
    # Per-joint checks
    # ─────────────────────────────────────────────────────────────────
    def _check_joint(self, gs: GroupStats, j: JointStats, jmsg) -> None:  # noqa: ANN001
        target = float(jmsg.position)
        target_vel = float(jmsg.velocity)
        kp = float(jmsg.stiffness)
        kd = float(jmsg.damping)

        # 1. NaN / Inf.
        for label, value in (
            ("target_pos", target),
            ("target_vel", target_vel),
            ("stiffness", kp),
            ("damping", kd),
        ):
            if not math.isfinite(value):
                self._warn_throttled(
                    gs, j.name, f"nan_{label}",
                    f"{j.name}: {label} is {value!r} (NaN/Inf)",
                    severity="HARD",
                )

        if not math.isfinite(target):
            j.last_target = math.nan
            return

        # 2. Deviation from default.
        dev = target - j.default
        if abs(dev) > j.max_abs_dev:
            j.max_abs_dev = abs(dev)
        if abs(dev) > self._args.max_deviation:
            self._warn_throttled(
                gs, j.name, "max_dev",
                f"{j.name}: target={target:+.3f} rad, default={j.default:+.3f}, "
                f"dev={dev:+.3f} rad ({math.degrees(dev):+.1f} deg) > "
                f"{self._args.max_deviation:.3f}",
            )

        # 3. Step from previous target.
        if math.isfinite(j.last_target):
            step = target - j.last_target
            if abs(step) > j.max_abs_step:
                j.max_abs_step = abs(step)
            if abs(step) > self._args.max_step:
                # 50 Hz -> implied angular velocity if this step were a
                # constant velocity over one tick.
                implied_w = step / 0.02
                self._warn_throttled(
                    gs, j.name, "max_step",
                    f"{j.name}: target jumped {step:+.3f} rad in 20ms "
                    f"(~{implied_w:+.1f} rad/s, {math.degrees(implied_w):+.0f} deg/s) "
                    f"> {self._args.max_step:.3f} rad",
                )
        j.last_target = target

        # 4. Dry-run gain leak.
        if self._args.expect_dry_run:
            if abs(kp) > 1e-9:
                self._warn_throttled(
                    gs, j.name, "kp_leak",
                    f"{j.name}: stiffness={kp:.3g} != 0 in --expect-dry-run "
                    f"(deploy_x2.sh --dry-run should zero this)",
                    severity="HARD",
                )
            if abs(kd) > 1e-9:
                self._warn_throttled(
                    gs, j.name, "kd_leak",
                    f"{j.name}: damping={kd:.3g} != 0 in --expect-dry-run",
                    severity="HARD",
                )

    # ─────────────────────────────────────────────────────────────────
    # Throttled WARN print
    # ─────────────────────────────────────────────────────────────────
    def _warn_throttled(
        self,
        gs: GroupStats,
        joint_name: str,
        check_kind: str,
        message: str,
        severity: str = "SOFT",
    ) -> None:
        now = time.monotonic()
        key = (joint_name, check_kind)
        last = gs.last_warn_t.get(key, 0.0)
        if now - last < THROTTLE_WINDOW_S:
            return
        gs.last_warn_t[key] = now
        tag = _red("FAIL") if severity == "HARD" else _yellow("WARN")
        elapsed = now - self._t0
        print(f"[{elapsed:7.2f}s] {tag} {message}", flush=True)

    # ─────────────────────────────────────────────────────────────────
    # 1 Hz status line
    # ─────────────────────────────────────────────────────────────────
    def _on_status(self) -> None:
        elapsed = time.monotonic() - self._t0
        parts = []
        for group_name, gs in self._stats.items():
            hz = gs.cmd_hz_estimate
            worst_dev = max((j.max_abs_dev for j in gs.joints), default=0.0)
            worst_step = max((j.max_abs_step for j in gs.joints), default=0.0)
            color = _grey if gs.msg_count == 0 else (_green if hz > 5 else _yellow)
            parts.append(
                color(
                    f"{group_name:>5s}: "
                    f"{hz:5.1f}Hz "
                    f"max|dev|={worst_dev:.2f}rad "
                    f"max|step|={worst_step:.2f}rad"
                )
            )
        print(f"[{elapsed:7.2f}s] " + "  ".join(parts), flush=True)

    # ─────────────────────────────────────────────────────────────────
    # Final summary on shutdown
    # ─────────────────────────────────────────────────────────────────
    def print_final_summary(self) -> None:
        elapsed = time.monotonic() - self._t0
        print()
        print(_grey("=" * 78))
        print(f"x2_action_monitor summary (runtime {elapsed:.2f}s)")
        print(_grey("=" * 78))
        for group_name, gs in self._stats.items():
            print(
                f"\n{group_name:>5s}  msgs={gs.msg_count}  "
                f"cmd_hz~{gs.cmd_hz_estimate:.1f}"
            )
            print(f"        {'joint':32s}  {'max|dev|':>10s}  {'max|step|':>10s}")
            for j in gs.joints:
                tag = ""
                if j.max_abs_dev > self._args.max_deviation:
                    tag += " " + _yellow("dev>thr")
                if j.max_abs_step > self._args.max_step:
                    tag += " " + _yellow("step>thr")
                print(
                    f"        {j.name:32s}  "
                    f"{j.max_abs_dev:10.3f}  "
                    f"{j.max_abs_step:10.3f}{tag}"
                )
        print(_grey("=" * 78))


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────
def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live monitor for X2 deploy joint commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--max-deviation", type=float, default=0.5,
        help="warn if |target_pos - default| exceeds this (rad)",
    )
    p.add_argument(
        "--max-step", type=float, default=0.3,
        help="warn if |target_pos_t - target_pos_{t-1}| exceeds this (rad)",
    )
    p.add_argument(
        "--expect-dry-run", action="store_true",
        help="additionally warn if stiffness or damping != 0 (deploy_x2.sh "
             "--dry-run should zero both)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="suppress 1Hz status line; only print WARN/FAIL lines",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    rclpy.init()
    node = ActionMonitor(args)

    def _on_sigint(_sig, _frame):  # noqa: ANN001
        node.print_final_summary()
        rclpy.shutdown()

    signal.signal(signal.SIGINT, _on_sigint)
    try:
        rclpy.spin(node)
    finally:
        if rclpy.ok():
            node.print_final_summary()
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
