#!/usr/bin/env python3
"""Scan MC-controlled joints to capture the stock controller's PD baseline
and the robot's measured response to operator nudges.

Use case: BEFORE switching to the C++ deploy with our policy, we want to
know what KP/KD/target the stock motion controller (MC, in STAND_DEFAULT)
is publishing on every joint, AND how the measured joint state moves when
the operator nudges the robot. That gives us:

  * Ground-truth stiffness / damping per joint -- the "if we had the same
    PD MC uses, we'd be at least as stable as MC" baseline.
  * Target-vs-measured tracking error during the nudge -- shows the
    intrinsic disturbance response of the trained PD on real hardware,
    which we can then compare against our policy's response with the
    same nudge applied during a deploy run.

The script subscribes to all 8 topics simultaneously:

  /aima/hal/joint/{leg,waist,arm,head}/state   -- HAL measured pos/vel/effort
  /aima/hal/joint/{leg,waist,arm,head}/command -- MC commanded target / KP / KD

samples for ``--duration`` seconds (default 30 s = plenty for one nudge),
then prints a per-joint summary table and writes the raw 8-topic samples
to a JSONL for offline analysis. Mark the nudge window in your test log
(the script timestamps every sample so you can crop the window after).

Designed to run inside the docker_x2/ container in real-mode (same path
as x2_capture_pose.py). Will NOT publish anything on the bus -- this is
a pure subscriber. Safe to run while MC is in STAND_DEFAULT actively
holding the robot up.

Run pattern:

  cd /home/stickbot/Projects/GR00T-WholeBodyControl
  ./gear_sonic_deploy/scripts/x2_scan_mc_motors.sh --duration 30

(The .sh shim handles the docker re-exec + ROS env sourcing.)

Output (stdout):

  - per-group sample counts + average update rate, so you can confirm
    MC is actually publishing before counting on the numbers;
  - per-joint table with columns:
      name | mc_kp | mc_kd | mc_target | pos_med | pos_max_abs_err |
      vel_max_abs | effort_max_abs
    Compared against ``DEFAULT_ANGLES`` (codegen baseline from
    policy_parameters.hpp) so you can spot any joint that's holding off
    from default.

Output (file): ``mc_motor_scan_<timestamp>.jsonl`` -- one line per
JointStateArray / JointCommandArray sample, with monotonic timestamps so
you can crop the nudge window post-hoc.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import threading
import time
from typing import Any

import numpy as np

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

try:
    from aimdk_msgs.msg import JointCommandArray, JointStateArray
except ImportError as e:
    print(
        "ERROR: aimdk_msgs not on the Python path. Run this inside the\n"
        "       docker_x2/ container (docker-compose.yml + docker-compose.real.yml)\n"
        "       or source the colcon overlay that built aimdk_msgs.",
        file=sys.stderr,
    )
    raise SystemExit(1) from e


# Canonical 31-DOF MJ joint order. Same source-of-truth as
# x2_capture_pose.py / x2_preflight.py / policy_parameters.hpp. Match by
# name so the per-topic ordering doesn't matter.
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
NAME_SET = set(MUJOCO_JOINT_NAMES)

DEFAULT_ANGLES = {
    "left_hip_pitch_joint": -0.312, "left_hip_roll_joint": 0.0, "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.669, "left_ankle_pitch_joint": -0.363, "left_ankle_roll_joint": 0.0,
    "right_hip_pitch_joint": -0.312, "right_hip_roll_joint": 0.0, "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.669, "right_ankle_pitch_joint": -0.363, "right_ankle_roll_joint": 0.0,
    "waist_yaw_joint": 0.0, "waist_pitch_joint": 0.0, "waist_roll_joint": 0.0,
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2,
    "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": -0.6,
    "left_wrist_yaw_joint": 0.0, "left_wrist_pitch_joint": 0.0, "left_wrist_roll_joint": 0.0,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2,
    "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": -0.6,
    "right_wrist_yaw_joint": 0.0, "right_wrist_pitch_joint": 0.0, "right_wrist_roll_joint": 0.0,
    "head_yaw_joint": 0.0, "head_pitch_joint": 0.0,
}

JOINT_GROUPS = ("leg", "waist", "arm", "head")
STATE_TOPIC = "/aima/hal/joint/{group}/state"
COMMAND_TOPIC = "/aima/hal/joint/{group}/command"

# Best-effort QoS: MC and the HAL both publish best-effort, depth=1, so
# we have to match or we won't see anything. (Mirrors x2_preflight.py and
# x2_capture_pose.py.)
_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    durability=DurabilityPolicy.VOLATILE,
)


class _ScanNode(Node):
    """Subscribes to MC's command + the HAL state for all 4 joint groups."""

    def __init__(self) -> None:
        super().__init__("x2_scan_mc_motors")
        self._lock = threading.Lock()
        # samples[name] = list of dicts -- one per HAL state callback.
        self._state_samples: dict[str, list[dict[str, float]]] = {}
        # cmds[name] = list of dicts -- one per MC command callback.
        self._cmd_samples: dict[str, list[dict[str, float]]] = {}
        self._group_state_counts: dict[str, int] = {g: 0 for g in JOINT_GROUPS}
        self._group_cmd_counts: dict[str, int] = {g: 0 for g in JOINT_GROUPS}
        # Raw JSONL log: one record per callback so we keep arrival
        # ordering + monotonic timestamps for post-hoc nudge cropping.
        self._raw_log: list[dict[str, Any]] = []
        self._t0 = time.monotonic()

        for group in JOINT_GROUPS:
            self.create_subscription(
                JointStateArray,
                STATE_TOPIC.format(group=group),
                lambda msg, g=group: self._on_state(g, msg),
                _QOS,
            )
            self.create_subscription(
                JointCommandArray,
                COMMAND_TOPIC.format(group=group),
                lambda msg, g=group: self._on_command(g, msg),
                _QOS,
            )

    def probe_publishers(self) -> dict[str, dict[str, Any]]:
        """Snapshot what publishers DDS sees on each command topic.

        Returns one entry per topic with the publisher count + the
        reliability / durability of each offered QoS. Lets us tell apart
        the two failure modes when a scan returns zero command messages:

          (a) ``count == 0``        -> nothing is publishing (MC isn't in
              control AND our deploy isn't in CONTROL state; or topic
              prefix mismatch);
          (b) ``count > 0, hz==0``  -> publishers exist but the messages
              never reach us. This is almost always a QoS-incompat issue
              (we subscribe BEST_EFFORT, which accepts both BEST_EFFORT
              and RELIABLE publishers, so the only way this happens is
              if the publisher offers something exotic).

        Discovery takes ~1 s to settle; callers should wait a beat after
        node construction before invoking.
        """
        out: dict[str, dict[str, Any]] = {}
        for group in JOINT_GROUPS:
            topic = COMMAND_TOPIC.format(group=group)
            infos = self.get_publishers_info_by_topic(topic)
            qos_list: list[str] = []
            for info in infos:
                q = info.qos_profile
                qos_list.append(
                    f"node={info.node_namespace}/{info.node_name} "
                    f"reliability={q.reliability.name} "
                    f"durability={q.durability.name} "
                    f"depth={q.depth}"
                )
            out[topic] = {
                "count": len(infos),
                "publishers": qos_list,
            }
        return out

    def _on_state(self, group: str, msg: JointStateArray) -> None:
        ts = time.monotonic() - self._t0
        with self._lock:
            self._group_state_counts[group] += 1
            joints_dump = []
            for js in msg.joints:
                name = str(js.name)
                if name not in NAME_SET:
                    continue
                rec = {
                    "t":   ts,
                    "pos": float(js.position),
                    "vel": float(js.velocity),
                    "eff": float(js.effort),
                }
                self._state_samples.setdefault(name, []).append(rec)
                joints_dump.append({"name": name, **rec})
            self._raw_log.append({
                "kind":  "state",
                "group": group,
                "t":     ts,
                "joints": joints_dump,
            })

    def _on_command(self, group: str, msg: JointCommandArray) -> None:
        ts = time.monotonic() - self._t0
        with self._lock:
            self._group_cmd_counts[group] += 1
            joints_dump = []
            for jc in msg.joints:
                name = str(jc.name)
                if name not in NAME_SET:
                    continue
                rec = {
                    "t":   ts,
                    "tgt": float(jc.position),
                    "vel": float(jc.velocity),
                    "eff": float(jc.effort),
                    "kp":  float(jc.stiffness),
                    "kd":  float(jc.damping),
                }
                self._cmd_samples.setdefault(name, []).append(rec)
                joints_dump.append({"name": name, **rec})
            self._raw_log.append({
                "kind":  "command",
                "group": group,
                "t":     ts,
                "joints": joints_dump,
            })

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state":            {n: list(s) for n, s in self._state_samples.items()},
                "cmd":              {n: list(s) for n, s in self._cmd_samples.items()},
                "group_state_cnt":  dict(self._group_state_counts),
                "group_cmd_cnt":    dict(self._group_cmd_counts),
                "raw":              list(self._raw_log),
            }


def _summarize(snap: dict[str, Any], duration: float) -> str:
    """Build a human-readable per-joint summary table from a snapshot."""
    lines: list[str] = []
    lines.append("")
    lines.append("=== Topic update rates over scan window ===")
    lines.append(f"  duration: {duration:.2f} s")
    for group in JOINT_GROUPS:
        s_cnt = snap["group_state_cnt"][group]
        c_cnt = snap["group_cmd_cnt"][group]
        s_hz  = s_cnt / max(duration, 1e-6)
        c_hz  = c_cnt / max(duration, 1e-6)
        verdict = ""
        if c_cnt == 0:
            verdict = "  ⚠ NO MC COMMANDS SEEN -- is MC actually in control?"
        elif c_hz < 50:
            verdict = "  ⚠ < 50 Hz -- MC may be in PASSIVE_DEFAULT"
        lines.append(f"  {group:5s}: state {s_cnt:5d} ({s_hz:6.1f} Hz)   "
                     f"command {c_cnt:5d} ({c_hz:6.1f} Hz){verdict}")

    lines.append("")
    lines.append("=== Per-joint summary (MC command + measured response) ===")
    lines.append("")
    header = (
        f"  {'joint':28s}  "
        f"{'mc_kp':>7s} {'mc_kd':>6s}  "
        f"{'mc_tgt':>7s}  "
        f"{'def':>7s}  "
        f"{'pos_med':>8s}  "
        f"{'tgt-def':>8s}  "
        f"{'pos-tgt_p95':>11s}  "
        f"{'|vel|_p95':>10s}  "
        f"{'|eff|_p95':>10s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for name in MUJOCO_JOINT_NAMES:
        state = snap["state"].get(name, [])
        cmd   = snap["cmd"].get(name, [])
        if not state and not cmd:
            lines.append(f"  {name:28s}  -- no samples --")
            continue

        if cmd:
            kps = np.array([c["kp"]  for c in cmd])
            kds = np.array([c["kd"]  for c in cmd])
            tgs = np.array([c["tgt"] for c in cmd])
            mc_kp  = float(np.median(kps))
            mc_kd  = float(np.median(kds))
            mc_tgt = float(np.median(tgs))
        else:
            mc_kp  = mc_kd  = mc_tgt = float("nan")

        if state:
            pos = np.array([s["pos"] for s in state])
            vel = np.array([s["vel"] for s in state])
            eff = np.array([s["eff"] for s in state])
            pos_med = float(np.median(pos))
            vel_p95 = float(np.percentile(np.abs(vel), 95))
            eff_p95 = float(np.percentile(np.abs(eff), 95))
        else:
            pos_med = vel_p95 = eff_p95 = float("nan")

        default = DEFAULT_ANGLES.get(name, float("nan"))
        tgt_off = mc_tgt - default if cmd else float("nan")
        if state and cmd:
            err = np.abs(np.array([s["pos"] for s in state]) - mc_tgt)
            err_p95 = float(np.percentile(err, 95))
        else:
            err_p95 = float("nan")

        lines.append(
            f"  {name:28s}  "
            f"{mc_kp:7.2f} {mc_kd:6.2f}  "
            f"{mc_tgt:+7.3f}  "
            f"{default:+7.3f}  "
            f"{pos_med:+8.3f}  "
            f"{tgt_off:+8.3f}  "
            f"{err_p95:11.4f}  "
            f"{vel_p95:10.4f}  "
            f"{eff_p95:10.3f}"
        )

    lines.append("")
    lines.append("Notes:")
    lines.append("  - mc_kp / mc_kd: MEDIAN MC-published stiffness / damping over the")
    lines.append("    scan. If MC modulates these (e.g. variable-stiffness modes), the")
    lines.append("    median hides that; check the JSONL for the full timeseries.")
    lines.append("  - tgt-def: how far MC's commanded target sits from the codegen")
    lines.append("    default_angles. Should be ~0 in STAND_DEFAULT; nonzero = MC is")
    lines.append("    actively servoing somewhere else.")
    lines.append("  - pos-tgt_p95: 95th-percentile |measured pos - mc_target|.")
    lines.append("    During a NUDGE this is the disturbance amplitude MC is rejecting.")
    lines.append("  - |vel|_p95 and |eff|_p95: 95th-percentile absolute speed / effort.")
    lines.append("    A nudge that's just barely noticeable to you typically shows up")
    lines.append("    as 0.05-0.20 rad/s velocity and a few Nm of effort on legs/waist.")
    return "\n".join(lines)


def _filter_velocity(t: np.ndarray, v: np.ndarray,
                     lpf_hz: float, vel_dead_zone: float
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Low-pass + dead-zone filter the raw velocity signal.

    The HAL state stream comes through at ~1067 Hz on legs/waist/arm. At
    that rate, velocity quantization noise (encoder-tick differencing,
    motor commutation, structural micro-vibration) creates dozens of
    spurious sign flips per second on a joint that is, physically,
    rotating smoothly. Counting those as oscillations gives nonsense
    flip_hz values like 25-38 Hz on top of an underlying 0.5 rad
    smooth motion (which we observed in the first cut of the metric).

    Two-stage clean-up:

    1. **LPF (boxcar moving average)** at ``lpf_hz``. Cuts everything
       above the cutoff. A 10 Hz cutoff still preserves real underdamped
       ankle / knee / waist ring-downs (1-5 Hz) with margin, and
       collapses the high-freq sensor floor to ~zero.
    2. **Dead-zone** at ``vel_dead_zone`` rad/s. Below this magnitude
       the joint is "effectively still"; we set it to 0 so we don't
       count micro-reversals around the quiescent point as oscillation.

    Returns ``(v_filtered, v_after_dead_zone)``. The first is for RMS
    reporting (preserves smooth motion magnitude); the second feeds the
    sign-flip counter.
    """
    if len(t) < 4:
        return v.copy(), v.copy()
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return v.copy(), v.copy()
    fs = 1.0 / dt
    # Boxcar moving-average window. The 3 dB cutoff of an N-tap boxcar
    # filter sits near fs / (2N), so window N = fs / (2 * lpf_hz). At
    # 1067 Hz / (2 * 10) ≈ 53 samples that means a ~9 Hz cutoff with the
    # ~6 dB roll-off you'd expect from a sinc^1 rolloff -- plenty for
    # our purposes. Convolve with 'same' so length is preserved.
    w = max(3, int(round(fs / max(2.0 * lpf_hz, 1e-6))))
    w = min(w, max(1, len(v) // 4))  # never use more than 1/4 of samples
    if w < 3:
        v_filt = v.copy()
    else:
        kernel = np.ones(w, dtype=np.float64) / w
        v_filt = np.convolve(v, kernel, mode="same")
    v_dz = np.where(np.abs(v_filt) >= vel_dead_zone, v_filt, 0.0)
    return v_filt, v_dz


def _flip_count_with_dead_zone(v_dz: np.ndarray) -> int:
    """Count physically-meaningful velocity sign flips.

    State-machine: track the last *non-zero* velocity sign. A flip is
    counted only when we transition from one sign to the opposite via
    any number of dead-zone (zero) samples in between. This makes a
    "settle into stillness" event NOT count as an oscillation.
    """
    if len(v_dz) == 0:
        return 0
    last_sign = 0.0
    flips = 0
    for vv in v_dz:
        if vv == 0.0:
            continue
        s = 1.0 if vv > 0.0 else -1.0
        if last_sign != 0.0 and s != last_sign:
            flips += 1
        last_sign = s
    return flips


def _oscillation_summary(snap: dict[str, Any], duration: float,
                         window_s: float = 1.0,
                         step_s: float = 0.25,
                         top_k: int = 12,
                         lpf_hz: float = 10.0,
                         vel_dead_zone: float = 0.05) -> str:
    """Per-joint oscillation analysis on a sliding 1 s window.

    For each joint we compute, across overlapping windows of ``window_s``
    seconds:

      v_filt         = velocity LPF'd at lpf_hz to suppress the
                       sub-cycle quantization noise that lives well
                       above any physical joint resonance.
      v_dz           = v_filt zeroed out wherever |v_filt| < vel_dead_zone
                       so micro-reversals around the quiescent point
                       don't get counted as oscillation flips.
      flip_rate (Hz) = (#sign-flips of v_dz, ignoring zero spans)
                       / (2 * window_s). One full oscillation cycle =
                       2 sign flips, so dividing by 2 gives the true
                       physical resonance frequency in Hz.
      vel_rms        = RMS of v_filt in the window (preserves real
                       motion magnitude; the dead-zone is only used
                       for flip counting).
      osc_power      = flip_rate * vel_rms

    osc_power separates oscillation (repeated zero-crossings) from a
    single nudge that then settles (high vel, ~0 flips). Joints holding
    stably score ~0; joints ringing at a few Hz with non-trivial speed
    score the highest. We then report the joints with the largest peak
    osc_power across all windows -- those are the ones still ringing
    after a perturbation.

    Also reports the "baseline" osc_power (median across all windows
    EXCLUDING the peak ±2 step neighbourhood) so you can see whether
    the peak is a true transient ring-down or chronic background jitter.

    No external dependencies beyond numpy (already imported above).
    """
    lines: list[str] = []
    lines.append("")
    lines.append(f"=== Oscillation analysis (sliding {window_s:.1f} s window, step {step_s:.2f} s) ===")
    lines.append(f"    Velocity LPF cutoff: {lpf_hz:.1f} Hz   |   dead-zone: {vel_dead_zone:.3f} rad/s")
    lines.append("")
    lines.append("  Top-K joints by PEAK oscillation power across the scan window.")
    lines.append("  Use this AFTER nudging the robot during the scan to see which")
    lines.append("  joints are ringing during the recovery transient.")
    lines.append("")

    rows: list[tuple[str, float, float, float, float, float, float, float]] = []
    # Each row: (name, peak_osc, peak_t, peak_flip_hz, peak_vel_rms,
    #            peak_pos_p2p, baseline_osc, ratio)

    for name in MUJOCO_JOINT_NAMES:
        state = snap["state"].get(name, [])
        if len(state) < 10:
            continue
        ts  = np.array([s["t"]   for s in state], dtype=np.float64)
        pos = np.array([s["pos"] for s in state], dtype=np.float64)
        vel = np.array([s["vel"] for s in state], dtype=np.float64)
        if ts[-1] - ts[0] < window_s:
            continue

        # Pre-filter velocity ONCE for the whole joint timeseries (not
        # per-window) so the LPF doesn't have edge artefacts at every
        # window boundary. v_filt -> RMS reporting; v_dz -> flip counter.
        v_filt, v_dz_full = _filter_velocity(ts, vel, lpf_hz, vel_dead_zone)

        # Sliding-window scan over the filtered series.
        win_starts = np.arange(ts[0], ts[-1] - window_s + 1e-9, step_s)
        peak_osc      = 0.0
        peak_t        = float("nan")
        peak_flip_hz  = 0.0
        peak_vel_rms  = 0.0
        peak_pos_p2p  = 0.0
        per_window_osc: list[float] = []
        peak_idx      = -1
        for idx, t0 in enumerate(win_starts):
            t1 = t0 + window_s
            mask = (ts >= t0) & (ts < t1)
            if mask.sum() < 4:
                per_window_osc.append(0.0)
                continue
            vf  = v_filt[mask]
            vdz = v_dz_full[mask]
            p   = pos[mask]
            flips      = _flip_count_with_dead_zone(vdz)
            flip_hz    = flips / (2.0 * window_s)
            vel_rms    = float(np.sqrt(np.mean(vf * vf)))
            osc_power  = flip_hz * vel_rms
            per_window_osc.append(osc_power)
            if osc_power > peak_osc:
                peak_osc     = osc_power
                peak_t       = float(t0)
                peak_flip_hz = flip_hz
                peak_vel_rms = vel_rms
                peak_pos_p2p = float(np.max(p) - np.min(p))
                peak_idx     = idx

        if peak_idx < 0 or not per_window_osc:
            continue
        # Baseline = median of all OTHER windows (excluding the peak ±2 step
        # neighbourhood, so we don't bias the baseline by the same transient).
        excl = set(range(max(0, peak_idx - 2), min(len(per_window_osc), peak_idx + 3)))
        baseline_pool = [v for i, v in enumerate(per_window_osc) if i not in excl]
        baseline_osc = float(np.median(baseline_pool)) if baseline_pool else 0.0
        ratio = peak_osc / max(baseline_osc, 1e-6)
        rows.append((name, peak_osc, peak_t, peak_flip_hz, peak_vel_rms,
                     peak_pos_p2p, baseline_osc, ratio))

    if not rows:
        lines.append("  No joints had enough samples for oscillation analysis.")
        lines.append(f"  (Scan window was {duration:.1f} s; need >= {window_s:.1f} s of state samples per joint.)")
        return "\n".join(lines)

    rows.sort(key=lambda r: r[1], reverse=True)

    header = (
        f"  {'joint':28s}  "
        f"{'peak_osc':>9s}  "
        f"{'peak@t':>7s}  "
        f"{'flip_hz':>8s}  "
        f"{'vel_rms':>8s}  "
        f"{'pos_p2p':>8s}  "
        f"{'baseline':>9s}  "
        f"{'ratio':>6s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for (name, peak_osc, peak_t, flip_hz, vel_rms, pos_p2p,
         baseline_osc, ratio) in rows[:top_k]:
        # After LPF + dead-zone, real ring-downs sit well above
        # baseline (ratio >= 5x is a reliable signal). The "background"
        # cutoff at < 2x catches joints that never really oscillated
        # during the scan window.
        marker = ""
        if ratio < 2.0:
            marker = " <- background"
        elif ratio >= 5.0:
            marker = " <- RINGING"
        lines.append(
            f"  {name:28s}  "
            f"{peak_osc:9.4f}  "
            f"{peak_t:7.2f}  "
            f"{flip_hz:8.2f}  "
            f"{vel_rms:8.4f}  "
            f"{pos_p2p:8.4f}  "
            f"{baseline_osc:9.4f}  "
            f"{ratio:6.1f}x{marker}"
        )

    lines.append("")
    lines.append("Reading the table (LPF + dead-zone applied to velocity):")
    lines.append("  - peak_osc:  flip_hz * vel_rms in the worst-offender 1 s window.")
    lines.append("               > 0.05 with ratio > 5x = clear ring-down.")
    lines.append("  - peak@t:    elapsed seconds when the peak window started.")
    lines.append("               Lines up with when you nudged + a brief settle.")
    lines.append("  - flip_hz:   velocity sign-flip rate (= oscillation freq) of the")
    lines.append("               LPF'd velocity in the peak window. Physical ankle")
    lines.append("               ring-down typically 2-5 Hz; knees / waist 1-3 Hz.")
    lines.append("               > 8 Hz post-LPF means stiff joint / very fast resonance.")
    lines.append("  - vel_rms:   RMS LPF'd velocity in the peak window.")
    lines.append("  - pos_p2p:   peak-to-peak position swing during the peak window.")
    lines.append("               Compare across joints to find the largest physical")
    lines.append("               excursions (often the most informative signal).")
    lines.append("  - baseline:  median osc_power excluding the peak ±2-step window.")
    lines.append("               Should be near 0 for stable joints.")
    lines.append("  - ratio:     peak_osc / baseline_osc. > 5x = real ring-down.")
    lines.append("")
    lines.append("Tuning note:")
    lines.append("  - To boost SNR, scan during a single isolated nudge (--duration 8-10).")
    lines.append("  - To re-process an existing JSONL with different filter settings,")
    lines.append("    use: x2_scan_mc_motors.py --replay PATH --osc-lpf-hz N --osc-vel-dead-zone X")
    return "\n".join(lines)


def _load_snapshot_from_jsonl(path: str) -> dict[str, Any]:
    """Reconstruct the same ``snapshot`` dict that ``_ScanNode.snapshot``
    would have returned, by replaying the ``raw`` records persisted by
    a previous live scan.

    The JSONL has one record per callback (``kind`` in {"state","command"})
    with a flat list of joints. We undo that grouping back into the
    per-joint time series ``state[name]`` and ``cmd[name]`` that the
    summary functions expect, plus per-group sample counts.
    """
    state_samples: dict[str, list[dict[str, float]]] = {}
    cmd_samples: dict[str, list[dict[str, float]]] = {}
    group_state_counts: dict[str, int] = {g: 0 for g in JOINT_GROUPS}
    group_cmd_counts: dict[str, int] = {g: 0 for g in JOINT_GROUPS}
    raw: list[dict[str, Any]] = []

    p = pathlib.Path(path)
    if not p.is_file():
        raise SystemExit(f"replay JSONL not found: {path}")
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            raw.append(rec)
            kind  = rec.get("kind")
            group = rec.get("group")
            if group not in JOINT_GROUPS:
                continue
            if kind == "state":
                group_state_counts[group] += 1
                for j in rec.get("joints", []):
                    name = j.get("name")
                    if name not in NAME_SET:
                        continue
                    state_samples.setdefault(name, []).append({
                        "t":   float(j.get("t", rec.get("t", 0.0))),
                        "pos": float(j.get("pos", 0.0)),
                        "vel": float(j.get("vel", 0.0)),
                        "eff": float(j.get("eff", 0.0)),
                    })
            elif kind == "command":
                group_cmd_counts[group] += 1
                for j in rec.get("joints", []):
                    name = j.get("name")
                    if name not in NAME_SET:
                        continue
                    cmd_samples.setdefault(name, []).append({
                        "t":   float(j.get("t", rec.get("t", 0.0))),
                        "tgt": float(j.get("tgt", 0.0)),
                        "vel": float(j.get("vel", 0.0)),
                        "eff": float(j.get("eff", 0.0)),
                        "kp":  float(j.get("kp",  0.0)),
                        "kd":  float(j.get("kd",  0.0)),
                    })
    return {
        "state":            state_samples,
        "cmd":              cmd_samples,
        "group_state_cnt":  group_state_counts,
        "group_cmd_cnt":    group_cmd_counts,
        "raw":              raw,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--duration", type=float, default=30.0,
                    help="Scan window in seconds (default 30). Nudge during this window.")
    ap.add_argument("--out", default=None,
                    help="Output JSONL path. Default: mc_motor_scan_<unix_ts>.jsonl in CWD.")
    ap.add_argument("--countdown", type=float, default=3.0,
                    help="Seconds to wait before starting (gives you time to position your hand).")
    ap.add_argument("--no-oscillation", action="store_true",
                    help="Skip the per-joint oscillation analysis at the end of the summary.")
    ap.add_argument("--osc-window", type=float, default=1.0,
                    help="Sliding-window size (seconds) for oscillation analysis (default 1.0).")
    ap.add_argument("--osc-step", type=float, default=0.25,
                    help="Sliding-window step (seconds) for oscillation analysis (default 0.25).")
    ap.add_argument("--osc-top-k", type=int, default=12,
                    help="How many top-oscillating joints to print (default 12).")
    ap.add_argument("--osc-lpf-hz", type=float, default=10.0,
                    help="Velocity low-pass filter cutoff in Hz before counting "
                         "sign-flips. Suppresses sub-cycle quantization noise. "
                         "Default 10.0 (covers all physical joint resonances).")
    ap.add_argument("--osc-vel-dead-zone", type=float, default=0.05,
                    help="Velocity dead-zone in rad/s. Below this magnitude the "
                         "joint is considered still and micro-reversals do not "
                         "count as oscillation flips. Default 0.05.")
    ap.add_argument("--replay", default=None,
                    help="Skip the live ROS scan and instead re-analyze an "
                         "existing JSONL written by a previous run. Useful for "
                         "trying different LPF / dead-zone settings on the same "
                         "captured nudge without re-running the scan.")
    ap.add_argument("--stop-sentinel", default=None,
                    help="Optional sentinel file path. When the file is "
                         "touch'd while the scan is running, the capture loop "
                         "exits cleanly at the next tick (within ~250 ms) and "
                         "the JSONL + summary are written normally. Lets you "
                         "run open-ended captures: pass a generous --duration "
                         "(e.g. 600 s) and signal stop by 'touch <sentinel>' "
                         "when the operator session is done. Defaults to None "
                         "(no sentinel; --duration is the only timer).")
    args = ap.parse_args()

    # ──────────────────────────────────────────────────────────────────────
    # Replay mode: load JSONL, rebuild the snapshot dict, run summaries.
    # ──────────────────────────────────────────────────────────────────────
    if args.replay:
        snap = _load_snapshot_from_jsonl(args.replay)
        # Approximate scan duration from the first/last sample timestamps.
        all_ts: list[float] = []
        for samples in snap["state"].values():
            if samples:
                all_ts.append(samples[-1]["t"])
        for samples in snap["cmd"].values():
            if samples:
                all_ts.append(samples[-1]["t"])
        replay_duration = max(all_ts) if all_ts else 0.0
        print(f"\n[x2_scan_mc_motors] REPLAY: {args.replay}")
        print(f"[x2_scan_mc_motors] Reconstructed scan duration: {replay_duration:.2f} s")
        print(_summarize(snap, duration=replay_duration))
        if not args.no_oscillation:
            print(_oscillation_summary(snap, duration=replay_duration,
                                       window_s=args.osc_window,
                                       step_s=args.osc_step,
                                       top_k=args.osc_top_k,
                                       lpf_hz=args.osc_lpf_hz,
                                       vel_dead_zone=args.osc_vel_dead_zone))
        return 0

    rclpy.init()
    node = _ScanNode()
    exec_ = SingleThreadedExecutor()
    exec_.add_node(node)

    spin_thread = threading.Thread(target=exec_.spin, daemon=True)
    spin_thread.start()

    out_path = pathlib.Path(args.out) if args.out else \
        pathlib.Path.cwd() / f"mc_motor_scan_{int(time.time())}.jsonl"

    print(f"\n[x2_scan_mc_motors] Subscribed to /aima/hal/joint/{{leg,waist,arm,head}}/{{state,command}}.")
    print(f"[x2_scan_mc_motors] Output JSONL: {out_path}")
    print(f"[x2_scan_mc_motors] Scan duration: {args.duration:.1f} s")

    # Probe publishers BEFORE the nudge window so a missing /command
    # publisher surfaces at startup, not 30s later when the summary
    # prints NaNs. Gives DDS ~1s to discover everyone first.
    time.sleep(1.0)
    probe = node.probe_publishers()
    print(f"[x2_scan_mc_motors] DDS publisher discovery on /command topics:")
    any_zero = False
    for topic, info in probe.items():
        cnt = info["count"]
        if cnt == 0:
            any_zero = True
            print(f"  {topic}: NO publishers  (MC isn't in control AND deploy isn't in CONTROL)")
        else:
            print(f"  {topic}: {cnt} publisher(s)")
            for pub in info["publishers"]:
                print(f"    - {pub}")
    if any_zero:
        print(
            "[x2_scan_mc_motors] WARNING: at least one /command topic has zero\n"
            "                   publishers. The summary at the end will show NaN\n"
            "                   for that subgroup. To see MC's PD, stop the deploy\n"
            "                   and let MC retake STAND_DEFAULT before scanning."
        )

    if args.countdown > 0:
        for s in range(int(args.countdown), 0, -1):
            print(f"[x2_scan_mc_motors] Starting in {s} ...", end="\r", flush=True)
            time.sleep(1.0)
        print(" " * 40, end="\r", flush=True)
    print(f"[x2_scan_mc_motors] >>> NUDGE NOW <<<  (recording for {args.duration:.0f} s)")

    # Optional early-stop sentinel: when the file at ``args.stop_sentinel``
    # appears, the capture loop exits cleanly at the next tick, snapshot
    # is written, and the summary prints normally. Useful for open-ended
    # operator-driven captures where you don't know up front how long the
    # robot's session will take -- launch the scan with --duration 600 (10
    # min), let the operator do their thing, then touch the sentinel to
    # stop. SIGTERM/SIGINT are NOT used for this because killing the
    # process would skip the JSONL write at the bottom of main(); the
    # sentinel keeps the normal exit path intact.
    stop_sentinel = pathlib.Path(args.stop_sentinel) if args.stop_sentinel else None
    if stop_sentinel is not None and stop_sentinel.exists():
        try:
            stop_sentinel.unlink()
        except OSError:
            pass
    t_start = time.monotonic()
    early_stop = False
    while time.monotonic() - t_start < args.duration:
        elapsed = time.monotonic() - t_start
        remaining = args.duration - elapsed
        if stop_sentinel is not None and stop_sentinel.exists():
            print(" " * 70, end="\r", flush=True)
            print(f"[x2_scan_mc_motors] stop sentinel detected at {stop_sentinel} "
                  f"after {elapsed:.1f}s; exiting capture early")
            early_stop = True
            break
        print(f"[x2_scan_mc_motors] elapsed {elapsed:5.1f} s / remaining {remaining:5.1f} s",
              end="\r", flush=True)
        time.sleep(0.25)
    print(" " * 70, end="\r", flush=True)
    actual_duration = time.monotonic() - t_start
    print(f"[x2_scan_mc_motors] Scan complete "
          f"({'early-stopped' if early_stop else 'duration-elapsed'}, "
          f"actual {actual_duration:.1f}s). Stopping subscribers ...")

    snap = node.snapshot()
    exec_.shutdown()
    node.destroy_node()
    rclpy.shutdown()

    with out_path.open("w") as f:
        for rec in snap["raw"]:
            f.write(json.dumps(rec) + "\n")
    print(f"[x2_scan_mc_motors] Raw JSONL written: {out_path} "
          f"({len(snap['raw'])} records)")

    print(_summarize(snap, duration=actual_duration))
    if not args.no_oscillation:
        print(_oscillation_summary(snap, duration=actual_duration,
                                   window_s=args.osc_window,
                                   step_s=args.osc_step,
                                   top_k=args.osc_top_k,
                                   lpf_hz=args.osc_lpf_hz,
                                   vel_dead_zone=args.osc_vel_dead_zone))
    return 0


if __name__ == "__main__":
    sys.exit(main())
