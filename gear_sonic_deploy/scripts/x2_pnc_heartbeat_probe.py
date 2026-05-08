#!/usr/bin/env python3
"""x2_pnc_heartbeat_probe.py -- end-to-end validation of MC's pnc-slot
arbitration.

What it does:

  1. Subscribes briefly to /aima/hal/joint/{leg,waist,arm,head}/state,
     captures the current joint positions + per-group joint name lists.
  2. Reads the current MC mode (`GetMcAction`).
  3. Reads baseline current_input_source (`GetCurrentInputSource`).
  4. SetMcInputSource(MODIFY, pnc, prio, timeout_ms) -- update params.
  5. SetMcInputSource(ENABLE, pnc, ...)              -- claim the bus.
  6. Starts a 50 Hz publisher that sends JointCommandArray on each of
     the 4 command topics, with `position` = the captured-state position,
     `velocity = effort = stiffness = damping = 0`. Zero torque, zero
     motion -- the robot's joints stay exactly where they were sampled
     in step 1, no matter what mode MC is in.
  7. Starts a 10 Hz poll loop that calls GetCurrentInputSource and
     records (t_rel_s, owner_name).
  8. After --publish-duration-s, STOPS publishing but keeps polling for
     --post-stop-s. Goal: watch current_input_source flip back to ''
     after the 200 ms watchdog elapses.
  9. SetMcInputSource(DISABLE, pnc) -- explicit cleanup.
 10. Prints the timeline as a human-readable table; optionally dumps
     the full log to --log-json for offline analysis.

Safety:

  * Refuses to run unless MC mode is one of {PASSIVE_DEFAULT,
    DAMPING_DEFAULT}. Override with --allow-balancing-mode (you'll be
    prompted to re-confirm).
  * Publishes the *captured* joint positions with kp=kd=effort=0 -- the
    motors apply zero torque, the joints don't move.
  * Trap-cleanup always re-issues DISABLE on exit/SIGINT so we never
    leave pnc claimed.

Output:

  Timeline table on stdout, e.g.

    t_rel_s   event                        owner
    -------   --------------------------   ------------
     -0.012   baseline GetCurrentInput     ''
      0.000   SetMcInputSource(ENABLE)     <pre-publish>
      0.020   first publish on /...leg
      0.052   poll                         ''
      0.105   poll                         pnc      ← activated!
      0.207   poll                         pnc
      ...
      1.000   stop publishing
      1.105   poll                         pnc      ← still owns within timeout
      1.205   poll                         ''       ← watchdog reclaimed
      1.305   SetMcInputSource(DISABLE)
      1.405   poll                         ''
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from aimdk_msgs.msg import (  # type: ignore[import-not-found]
    JointCommand,
    JointCommandArray,
    JointStateArray,
)
from aimdk_msgs.srv import (  # type: ignore[import-not-found]
    GetCurrentInputSource,
    GetMcAction,
    SetMcInputSource,
)


GROUPS = ("leg", "waist", "arm", "head")
SAFE_MODES = ("PASSIVE_DEFAULT", "DAMPING_DEFAULT")

# McInputAction.value codes confirmed empirically by
# x2_mc_input_source_probe.sh (scratch/probes/mc_input_source_*/):
#   1001 ADD | 1002 MODIFY | 1003 DELETE | 2001 ENABLE | 2002 DISABLE
ACTION_MODIFY = 1002
ACTION_ENABLE = 2001
ACTION_DISABLE = 2002


@dataclass
class Observation:
    """A single (t_rel_s, owner_name) sample."""
    t_rel_s: float
    event: str
    owner: Optional[str] = None
    extra: dict = field(default_factory=dict)


class PncHeartbeatProbe(Node):
    def __init__(self, args: argparse.Namespace):
        super().__init__("x2_pnc_heartbeat_probe")
        self.args = args

        # ------------------------------------------------------------
        # Service clients
        # ------------------------------------------------------------
        self._get_mc_action = self.create_client(
            GetMcAction, "/aimdk_5Fmsgs/srv/GetMcAction"
        )
        self._get_input_source = self.create_client(
            GetCurrentInputSource, "/aimdk_5Fmsgs/srv/GetCurrentInputSource"
        )
        self._set_input_source = self.create_client(
            SetMcInputSource, "/aimdk_5Fmsgs/srv/SetMcInputSource"
        )

        # ------------------------------------------------------------
        # Joint state capture: one subscriber per group, latched first
        # sample wins. We wait on _captured_event below.
        # ------------------------------------------------------------
        self._captured_state: dict[str, JointStateArray] = {}
        self._captured_event = threading.Event()
        self._state_subs = []
        self._state_lock = threading.Lock()
        for grp in GROUPS:
            topic = f"/aima/hal/joint/{grp}/state"
            self._state_subs.append(
                self.create_subscription(
                    JointStateArray,
                    topic,
                    self._make_state_cb(grp),
                    qos_profile_sensor_data,
                )
            )

        # ------------------------------------------------------------
        # Joint command publishers (one per group)
        # ------------------------------------------------------------
        self._cmd_pubs = {}
        for grp in GROUPS:
            topic = f"/aima/hal/joint/{grp}/command"
            self._cmd_pubs[grp] = self.create_publisher(
                JointCommandArray, topic, qos_profile_sensor_data
            )

        # ------------------------------------------------------------
        # Pre-built command messages -- updated to reflect captured
        # state in capture_initial_state().
        # ------------------------------------------------------------
        self._cmd_templates: dict[str, JointCommandArray] = {}

        # Observation log, with timestamps relative to t_enable_mono.
        self._observations: list[Observation] = []
        self._obs_lock = threading.Lock()
        self._t_enable_mono: Optional[float] = None
        self._publish_seq = 0

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------
    def _make_state_cb(self, grp: str):
        def cb(msg: JointStateArray):
            with self._state_lock:
                if grp not in self._captured_state:
                    self._captured_state[grp] = msg
                    if len(self._captured_state) == len(GROUPS):
                        self._captured_event.set()
        return cb

    # ------------------------------------------------------------------
    # Initial state capture
    # ------------------------------------------------------------------
    def capture_initial_state(self, timeout_s: float = 5.0) -> None:
        self.get_logger().info(
            f"waiting for first JointStateArray on each of "
            f"{', '.join(GROUPS)} (timeout {timeout_s}s) ..."
        )
        if not self._captured_event.wait(timeout=timeout_s):
            missing = [g for g in GROUPS if g not in self._captured_state]
            raise RuntimeError(
                f"timed out before all groups reported state; missing={missing}. "
                f"Is the robot powered and the SDK ethernet up?"
            )
        for grp in GROUPS:
            msg = self._captured_state[grp]
            n = len(msg.joints)
            sample = ", ".join(
                f"{j.name}={j.position:.3f}" for j in msg.joints[:3]
            )
            self.get_logger().info(
                f"  {grp:5s}: {n} joints, sample [{sample}, ...]"
            )

        # Build command templates with kp=kd=effort=ff=0 and the
        # captured positions. These templates are reused every cycle;
        # only the header.stamp + sequence are bumped.
        for grp in GROUPS:
            state = self._captured_state[grp]
            tmpl = JointCommandArray()
            tmpl.header.frame_id = grp
            for js in state.joints:
                jc = JointCommand()
                jc.name = js.name
                jc.position = float(js.position)
                jc.velocity = 0.0
                jc.effort = 0.0
                jc.stiffness = 0.0
                jc.damping = 0.0
                tmpl.joints.append(jc)
            self._cmd_templates[grp] = tmpl

    # ------------------------------------------------------------------
    # Sync wrappers around service calls
    # ------------------------------------------------------------------
    def _wait_service(self, client, name: str, timeout_s: float = 3.0):
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"{name} service not available after {timeout_s}s")

    def _wait_future(self, future, timeout_s: float):
        """Block calling thread until `future` completes (relies on
        the executor spinning in a background thread)."""
        deadline = time.monotonic() + timeout_s
        while not future.done():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            time.sleep(min(0.005, remaining))
        return future.result()

    def get_mc_mode(self) -> str:
        self._wait_service(self._get_mc_action, "GetMcAction")
        req = GetMcAction.Request()
        future = self._get_mc_action.call_async(req)
        result = self._wait_future(future, timeout_s=3.0)
        if result is None:
            raise RuntimeError("GetMcAction returned no result within 3 s")
        return result.info.action_desc or "<empty>"

    def get_current_input_source(self) -> tuple[str, int, int]:
        self._wait_service(self._get_input_source, "GetCurrentInputSource")
        req = GetCurrentInputSource.Request()
        future = self._get_input_source.call_async(req)
        result = self._wait_future(future, timeout_s=2.0)
        if result is None:
            return ("<rpc-failed>", -1, -1)
        src = result.input_source
        return (src.name, src.priority, src.timeout)

    def set_input_source(self, action_value: int, name: str, priority: int,
                         timeout_ms: int) -> int:
        """Returns response.header.code (0 on success)."""
        self._wait_service(self._set_input_source, "SetMcInputSource")
        req = SetMcInputSource.Request()
        req.action.value = action_value
        req.input_source.name = name
        req.input_source.priority = priority
        req.input_source.timeout = timeout_ms
        future = self._set_input_source.call_async(req)
        result = self._wait_future(future, timeout_s=3.0)
        if result is None:
            return -1
        return int(result.response.header.code)

    # ------------------------------------------------------------------
    # The publish loop -- runs in its own thread, ticks at 50 Hz
    # ------------------------------------------------------------------
    def publish_loop(self, duration_s: float, stop_event: threading.Event,
                    rate_hz: float = 50.0) -> None:
        period = 1.0 / rate_hz
        t0 = time.monotonic()
        next_tick = t0
        seq = 0
        while not stop_event.is_set() and (time.monotonic() - t0) < duration_s:
            now_msg = self.get_clock().now().to_msg()
            for grp in GROUPS:
                tmpl = self._cmd_templates[grp]
                tmpl.header.stamp = now_msg
                tmpl.header.sequence = seq & 0xFFFFFFFF
                tmpl.header.meas_stamp = now_msg
                self._cmd_pubs[grp].publish(tmpl)
            if seq == 0:
                self._record(
                    t_now=time.monotonic(),
                    event="first publish",
                )
            seq += 1
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()  # we slipped; resync
        self._publish_seq = seq
        self._record(
            t_now=time.monotonic(),
            event="stop publishing",
            extra={"frames_published": seq},
        )

    # ------------------------------------------------------------------
    # The polling loop -- runs in its own thread, ticks at args.poll_hz
    # ------------------------------------------------------------------
    def poll_loop(self, total_duration_s: float, stop_event: threading.Event) -> None:
        period = 1.0 / self.args.poll_hz
        t0 = time.monotonic()
        next_tick = t0
        while not stop_event.is_set() and (time.monotonic() - t0) < total_duration_s:
            owner, prio, timeout_ms = self.get_current_input_source()
            self._record(
                t_now=time.monotonic(),
                event="poll",
                owner=owner,
                extra={"priority": prio, "timeout_ms": timeout_ms},
            )
            next_tick += period
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()

    def _record(self, *, t_now: float, event: str, owner: Optional[str] = None,
                extra: Optional[dict] = None) -> None:
        if self._t_enable_mono is None:
            t_rel = -float("inf")
        else:
            t_rel = t_now - self._t_enable_mono
        with self._obs_lock:
            self._observations.append(
                Observation(
                    t_rel_s=t_rel,
                    event=event,
                    owner=owner,
                    extra=extra or {},
                )
            )

    # ------------------------------------------------------------------
    # Result printing
    # ------------------------------------------------------------------
    def print_timeline(self) -> None:
        print()
        print("=== timeline (t_rel_s relative to ENABLE call) ===")
        print(f"{'t_rel_s':>9}  {'event':<30}  {'owner':<14}  notes")
        print(f"{'-' * 9}  {'-' * 30}  {'-' * 14}  {'-' * 24}")
        with self._obs_lock:
            obs = list(self._observations)
        for o in obs:
            t = f"{o.t_rel_s:9.3f}" if o.t_rel_s > -1e9 else "  <pre>  "
            owner = repr(o.owner) if o.owner is not None else ""
            extras = " ".join(f"{k}={v}" for k, v in o.extra.items())
            print(f"{t}  {o.event:<30}  {owner:<14}  {extras}")
        print()

    def evaluate_verdict(self) -> dict:
        with self._obs_lock:
            obs = list(self._observations)

        # When did owner first flip to 'pnc'?
        t_first_pnc: Optional[float] = None
        for o in obs:
            if o.event == "poll" and o.owner == "pnc" and o.t_rel_s > 0:
                t_first_pnc = o.t_rel_s
                break

        # When did owner go back to '' AFTER stop publishing?
        t_stop_pub: Optional[float] = None
        for o in obs:
            if o.event == "stop publishing":
                t_stop_pub = o.t_rel_s
                break

        t_reclaim: Optional[float] = None
        if t_stop_pub is not None:
            for o in obs:
                if (o.event == "poll" and o.t_rel_s > t_stop_pub
                        and (o.owner == "" or o.owner is None or o.owner == "<rpc-failed>")):
                    if o.owner == "<rpc-failed>":
                        continue
                    t_reclaim = o.t_rel_s
                    break

        verdict = {
            "t_first_pnc_s": t_first_pnc,
            "t_stop_publishing_s": t_stop_pub,
            "t_watchdog_reclaim_s": t_reclaim,
            "watchdog_latency_s": (
                t_reclaim - t_stop_pub if (t_reclaim is not None and t_stop_pub is not None) else None
            ),
        }
        return verdict


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end pnc-arbitration validation: publish 1 s of "
        "captured-state-as-command while pnc is enabled, watch "
        "GetCurrentInputSource flip to 'pnc' and back."
    )
    p.add_argument(
        "--gantry-confirmed", action="store_true",
        help="Operator confirms the robot is gantry-supported. Required for "
        "the publish/enable test path; not required for --watch-only."
    )
    p.add_argument(
        "--watch-only", type=float, default=0.0, metavar="SECONDS",
        help="Skip the publish/enable test entirely. Just poll "
        "GetCurrentInputSource + GetMcAction for SECONDS and log only when "
        "either field changes. Use this while you toggle the mobile app or "
        "joystick to discover what makes 'current_input_source' light up. "
        "Pass --watch-only 0 (or omit) to run the normal pnc test."
    )
    p.add_argument(
        "--allow-balancing-mode", action="store_true", default=False,
        help="Permit running while MC is in STAND_DEFAULT / LOCOMOTION_DEFAULT. "
        "Off by default; the test publishes zero-kp commands which would not "
        "actively balance, so the gantry must really be supporting the robot."
    )
    p.add_argument(
        "--publish-duration-s", type=float, default=1.0,
        help="How long the 50 Hz publisher streams. Default: 1.0"
    )
    p.add_argument(
        "--post-stop-s", type=float, default=0.6,
        help="Extra polling time AFTER stop-publishing to observe watchdog "
        "reclaim. Should be > expiration_time (default 200 ms). Default: 0.6"
    )
    p.add_argument(
        "--poll-hz", type=float, default=10.0,
        help="GetCurrentInputSource poll rate. Default: 10.0 (100 ms granularity)."
    )
    p.add_argument(
        "--pnc-priority", type=int, default=40,
        help="Priority used for SetMcInputSource. Default: 40 (mc.yaml canonical)."
    )
    p.add_argument(
        "--pnc-timeout-ms", type=int, default=200,
        help="Timeout used for SetMcInputSource. Default: 200 (matches expiration_time)."
    )
    p.add_argument(
        "--log-json", type=str, default="",
        help="Optional path to dump the full observation log as JSON."
    )
    return p.parse_args(argv)


# Topics whose publisher count we want to watch in --watch-only mode. These
# are the ones we suspect MC may vacate / claim across mode changes, plus a
# few candidate "input-stream" buses for InputManager arbitration.
WATCH_TOPICS = (
    "/aima/hal/joint/leg/command",
    "/aima/hal/joint/waist/command",
    "/aima/hal/joint/arm/command",
    "/aima/hal/joint/head/command",
    "/aima/mc/locomotion/velocity",
    "/aima/mc/body_pose",
    "/aima/teleop_bridge/vr_data",
    "/aima/hal/joint/hand/command",
)


def _topic_pub_counts(node, topics) -> dict:
    """Return {topic: publisher_count}. Quiet on rclpy errors."""
    out: dict = {}
    for t in topics:
        try:
            out[t] = node.count_publishers(t)
        except Exception:
            out[t] = -1
    return out


def _format_pubs(pubs: dict) -> str:
    """Compact one-line representation, only the topics with count > 0."""
    short = {
        "/aima/hal/joint/leg/command":      "leg.cmd",
        "/aima/hal/joint/waist/command":    "waist.cmd",
        "/aima/hal/joint/arm/command":      "arm.cmd",
        "/aima/hal/joint/head/command":     "head.cmd",
        "/aima/hal/joint/hand/command":     "hand.cmd",
        "/aima/mc/locomotion/velocity":     "mc.loc_vel",
        "/aima/mc/body_pose":               "mc.body_pose",
        "/aima/teleop_bridge/vr_data":      "tele.vr",
    }
    parts = []
    for t, n in pubs.items():
        label = short.get(t, t)
        parts.append(f"{label}={n}")
    return " ".join(parts)


def run_watch_only(node: "PncHeartbeatProbe", duration_s: float) -> int:
    """Read-only loop: poll GetCurrentInputSource + GetMcAction + per-topic
    publisher counts and print only when something changes. Returns when
    duration elapses or SIGINT."""
    print(f"[watch-only] polling for {duration_s:.1f} s "
          f"(Ctrl-C to stop early). Press buttons on the mobile app, etc.")
    print(f"[watch-only] columns: t_rel_s | mc_mode | owner | prio | to_ms | publisher counts")
    print()

    stop_event = threading.Event()

    def on_sigint(signum, frame):
        print("\n[watch-only] SIGINT received; stopping.", flush=True)
        stop_event.set()
    signal.signal(signal.SIGINT, on_sigint)

    t0 = time.monotonic()
    last_owner = "<unset>"
    last_prio = -999
    last_to = -999
    last_mode = "<unset>"
    last_pubs: dict = {}
    rows: list[dict] = []
    poll_period = 1.0 / max(node.args.poll_hz, 1.0)
    mode_poll_every = max(int(node.args.poll_hz / 2), 1)  # ~2 Hz mc-mode poll
    poll_count = 0
    deadline = t0 + duration_s

    print(f"{'t_rel_s':>9}  {'mc_mode':<22}  {'owner':<14}  {'prio':>5}  {'to_ms':>6}  pub_counts")
    print(f"{'-' * 9}  {'-' * 22}  {'-' * 14}  {'-' * 5}  {'-' * 6}  {'-' * 20}")

    while not stop_event.is_set() and time.monotonic() < deadline:
        t_rel = time.monotonic() - t0
        owner, prio, to_ms = node.get_current_input_source()
        pubs = _topic_pub_counts(node, WATCH_TOPICS)
        if poll_count % mode_poll_every == 0:
            try:
                mode = node.get_mc_mode()
            except Exception as e:  # noqa: BLE001
                mode = f"<rpc-failed:{e!s:.30s}>"
        else:
            mode = last_mode

        changed = (
            owner != last_owner
            or prio != last_prio
            or to_ms != last_to
            or mode != last_mode
            or pubs != last_pubs
        )
        first_row = poll_count == 0

        if changed or first_row:
            pub_str = _format_pubs(pubs)
            line = (f"{t_rel:9.3f}  {mode:<22}  {owner!r:<14}  "
                    f"{prio:>5}  {to_ms:>6}  {pub_str}")
            if changed and not first_row:
                deltas = []
                if owner != last_owner:
                    deltas.append(f"owner: {last_owner!r}->{owner!r}")
                if mode != last_mode:
                    deltas.append(f"mode: {last_mode!r}->{mode!r}")
                if prio != last_prio:
                    deltas.append(f"prio: {last_prio}->{prio}")
                if to_ms != last_to:
                    deltas.append(f"to: {last_to}->{to_ms}")
                if pubs != last_pubs:
                    pub_changes = []
                    for t, n in pubs.items():
                        old = last_pubs.get(t, -1)
                        if old != n:
                            pub_changes.append(f"{t.split('/')[-2]}.{t.split('/')[-1]}: {old}->{n}")
                    if pub_changes:
                        deltas.append("pubs: " + ", ".join(pub_changes))
                line += "    [" + " | ".join(deltas) + "]"
            print(line, flush=True)
            rows.append({
                "t_rel_s": t_rel,
                "mc_mode": mode,
                "owner": owner,
                "priority": prio,
                "timeout_ms": to_ms,
                "publishers": dict(pubs),
            })
            last_owner, last_prio, last_to, last_mode = owner, prio, to_ms, mode
            last_pubs = dict(pubs)

        poll_count += 1
        next_t = t0 + (poll_count + 1) * poll_period
        sleep = next_t - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)

    print()
    print(f"[watch-only] {poll_count} polls over {time.monotonic()-t0:.2f} s; "
          f"{len(rows)} delta row(s) recorded.")

    if node.args.log_json:
        os.makedirs(os.path.dirname(os.path.abspath(node.args.log_json)),
                    exist_ok=True)
        with open(node.args.log_json, "w") as fh:
            json.dump({
                "args": vars(node.args),
                "mode": "watch-only",
                "duration_s": time.monotonic() - t0,
                "poll_count": poll_count,
                "watched_topics": list(WATCH_TOPICS),
                "rows": rows,
            }, fh, indent=2)
        print(f"[watch-only] log: {node.args.log_json}")

    return 0


def main(argv: list[str]) -> int:
    args = parse_args(argv)

    if args.watch_only <= 0 and not args.gantry_confirmed:
        print("ERROR: --gantry-confirmed is required for the publish/enable "
              "test path. Use --watch-only N to run a no-publish observer "
              "instead (no gantry gate needed; entirely passive).",
              file=sys.stderr)
        return 64

    rclpy.init()
    executor = MultiThreadedExecutor(num_threads=4)
    spin_thread = None
    try:
        node = PncHeartbeatProbe(args)
        executor.add_node(node)
        spin_thread = threading.Thread(target=executor.spin, daemon=True)
        spin_thread.start()

        # ── Watch-only short-circuit ─────────────────────────────────
        if args.watch_only > 0:
            return run_watch_only(node, args.watch_only)

        # ── A. Capture current joint state on every group ────────────
        node.capture_initial_state(timeout_s=5.0)

        # ── B. MC mode safety gate ───────────────────────────────────
        mode = node.get_mc_mode()
        node.get_logger().info(f"current MC mode: {mode}")
        if mode not in SAFE_MODES and not args.allow_balancing_mode:
            node.get_logger().error(
                f"MC mode is {mode!r}, expected one of {SAFE_MODES}. "
                "The probe publishes zero-kp commands which would not actively "
                "balance the robot. Either put MC into PASSIVE_DEFAULT / "
                "DAMPING_DEFAULT first, or pass --allow-balancing-mode if you "
                "REALLY mean it (and the robot is fully gantry-supported)."
            )
            return 2

        # ── C. Baseline GetCurrentInputSource ─────────────────────────
        node._t_enable_mono = time.monotonic()  # provisional; reset at ENABLE
        owner_pre, prio_pre, to_pre = node.get_current_input_source()
        node._record(
            t_now=time.monotonic(),
            event="baseline GetCurrentInput",
            owner=owner_pre,
            extra={"priority": prio_pre, "timeout_ms": to_pre},
        )

        # ── D. MODIFY pnc (param-only; no claim yet) ──────────────────
        code_modify = node.set_input_source(
            ACTION_MODIFY, "pnc",
            args.pnc_priority, args.pnc_timeout_ms,
        )
        node._record(
            t_now=time.monotonic(),
            event="SetMcInputSource(MODIFY,pnc)",
            extra={"code": code_modify},
        )
        if code_modify != 0:
            node.get_logger().error(
                f"MODIFY pnc returned code={code_modify} (expected 0). Aborting."
            )
            return 3

        # ── E. ENABLE pnc -- the moment we start the experiment ───────
        node._t_enable_mono = time.monotonic()
        code_enable = node.set_input_source(
            ACTION_ENABLE, "pnc",
            args.pnc_priority, args.pnc_timeout_ms,
        )
        node._record(
            t_now=time.monotonic(),
            event="SetMcInputSource(ENABLE,pnc)",
            extra={"code": code_enable},
        )
        if code_enable != 0:
            node.get_logger().error(
                f"ENABLE pnc returned code={code_enable} (expected 0). Aborting."
            )
            # cleanup
            node.set_input_source(
                ACTION_DISABLE, "pnc",
                args.pnc_priority, args.pnc_timeout_ms,
            )
            return 3

        # ── F. Run publish + poll concurrently ────────────────────────
        stop_event = threading.Event()
        # SIGINT handler: make sure we always DISABLE on Ctrl-C
        def on_sigint(signum, frame):
            node.get_logger().warning("SIGINT received; stopping early.")
            stop_event.set()
        signal.signal(signal.SIGINT, on_sigint)

        total_poll_duration = args.publish_duration_s + args.post_stop_s + 0.1
        pub_thread = threading.Thread(
            target=node.publish_loop,
            args=(args.publish_duration_s, stop_event),
            daemon=True,
        )
        poll_thread = threading.Thread(
            target=node.poll_loop,
            args=(total_poll_duration, stop_event),
            daemon=True,
        )
        pub_thread.start()
        poll_thread.start()

        pub_thread.join(timeout=args.publish_duration_s + 5.0)
        # publish thread is done; wait for poll thread to finish post-stop window
        poll_thread.join(timeout=args.post_stop_s + 5.0)

        # ── G. DISABLE pnc -- explicit cleanup ────────────────────────
        code_disable = node.set_input_source(
            ACTION_DISABLE, "pnc",
            args.pnc_priority, args.pnc_timeout_ms,
        )
        node._record(
            t_now=time.monotonic(),
            event="SetMcInputSource(DISABLE,pnc)",
            extra={"code": code_disable},
        )

        # ── H. Final poll to confirm release ──────────────────────────
        owner_post, prio_post, to_post = node.get_current_input_source()
        node._record(
            t_now=time.monotonic(),
            event="final GetCurrentInput",
            owner=owner_post,
            extra={"priority": prio_post, "timeout_ms": to_post},
        )

        # ── I. Print + dump ───────────────────────────────────────────
        node.print_timeline()
        verdict = node.evaluate_verdict()

        print("=== verdict ===")
        if verdict["t_first_pnc_s"] is None:
            print("  ⚠ owner NEVER flipped to 'pnc' during the publish window.")
            print("    -> publishing did NOT activate the source.")
            print("    -> heartbeat model is NOT publish-based; investigate.")
        else:
            print(f"  ✓ owner flipped to 'pnc' at t_rel_s ≈ {verdict['t_first_pnc_s']:.3f}")
            print(f"    (latency from ENABLE = {verdict['t_first_pnc_s']*1000:.0f} ms; "
                  f"first_publish + service_RTT)")

        if verdict["watchdog_latency_s"] is None:
            print("  ⚠ owner did NOT return to '' after publishing stopped.")
            if verdict["t_first_pnc_s"] is None:
                print("    (because owner never became 'pnc' to begin with).")
            else:
                print("    -> watchdog might NOT auto-reclaim, OR post-stop window was too short.")
                print(f"    -> retry with --post-stop-s {args.post_stop_s + 1.0:.1f}.")
        else:
            print(f"  ✓ watchdog reclaimed at t_rel_s ≈ {verdict['t_watchdog_reclaim_s']:.3f}")
            print(f"    (latency from stop-publishing = "
                  f"{verdict['watchdog_latency_s']*1000:.0f} ms; expected ~ "
                  f"expiration_time={args.pnc_timeout_ms} ms + poll-period 100 ms).")

        if args.log_json:
            os.makedirs(os.path.dirname(os.path.abspath(args.log_json)), exist_ok=True)
            with open(args.log_json, "w") as fh:
                json.dump({
                    "args": vars(args),
                    "mc_mode": mode,
                    "code_modify": code_modify,
                    "code_enable": code_enable,
                    "code_disable": code_disable,
                    "owner_pre": owner_pre,
                    "owner_post": owner_post,
                    "verdict": verdict,
                    "observations": [
                        {
                            "t_rel_s": o.t_rel_s if o.t_rel_s > -1e9 else None,
                            "event": o.event,
                            "owner": o.owner,
                            "extra": o.extra,
                        } for o in node._observations
                    ],
                }, fh, indent=2)
            print(f"\n  log: {args.log_json}")

        return 0
    finally:
        try:
            executor.shutdown(timeout_sec=1.0)
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
