"""Live yaw / waist-yaw discontinuity sniffer for the X2 stack.

Subscribes to the three orientation-carrying streams the SONIC policy
sees and flags any single-tick discontinuity large enough to cause an
audible waist-yaw click. Pinpoints which side of the pipeline is at
fault for each click event:

  body_pose  (tcp://127.0.0.1:5565)   -- kplanner publishes here
  pose       (tcp://127.0.0.1:5556)   -- recorder publishes here (after
                                         merging arms / hands / gestures
                                         / idle fallback into body_pose)
  robot_pose (tcp://127.0.0.1:5570)   -- x2_debug_to_robot_pose bridge
                                         (live IMU measurement on real
                                         robot, ground truth in sim)

For each stream the sniffer tracks two scalars per frame:

  * **root_yaw_deg**     -- world-Z rotation of ``root_quat_xyzw``
                            (canonical "where is the body facing").
  * **waist_yaw_rad**    -- ``joint_pos_mj[12]`` ( waist_yaw_joint, the
                            dominant heading-correction effector inside
                            the SONIC tracking-policy reference).

Both are differenced against the previous frame on the same stream;
any |delta| above the per-channel spike threshold is printed in real
time with the values from ALL three streams at that instant so you can
read off who agreed with whom and who jumped.

Run alongside the planner stack:

    .venv/bin/python -m gear_sonic_deploy.scripts.x2_yaw_click_sniffer

Default thresholds (5 deg root yaw / 3 deg waist yaw per tick) are
chosen to flag everything the SONIC policy would track as a step input
at 50 Hz, but you can lower them with --yaw-spike-deg / --waist-spike-deg
to catch finer drifts.

Output legend:
  [TICK]   normal frame (rate-limited; once per second per source)
  [SPIKE]  delta above threshold -- prints prev/curr + cross-stream context
  [GAP]    gap > 100 ms between two frames on the same stream (stream may
           have dropped)
  [HOLD]   stream went silent for > 1 s
  [SUMMARY] once per second: rate + max-|delta| per channel per stream

Output is line-buffered so you can ``| tee yaw_sniff.log`` cleanly.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (  # noqa: E402
    unpack_message,
)


# MJ slot indices in the 31-DOF X2 body vector (see X2_BODY_JOINT_NAMES
# in gear_sonic/data/robot_model/.../x2_ultra_supplemental_info.py).
WAIST_YAW_MJ_SLOT = 12
WAIST_PITCH_MJ_SLOT = 13
WAIST_ROLL_MJ_SLOT = 14
WAIST_SLOTS: tuple[tuple[str, int], ...] = (
    ("waist_yaw",   WAIST_YAW_MJ_SLOT),
    ("waist_pitch", WAIST_PITCH_MJ_SLOT),
    ("waist_roll",  WAIST_ROLL_MJ_SLOT),
)


@dataclass
class StreamState:
    name: str
    sock: zmq.Socket
    topic: str
    decoder: str  # 'packed' or 'json'
    prev_yaw_deg: Optional[float] = None
    # Per-waist-joint previous values, keyed by joint name ("waist_yaw" / "waist_pitch" / "waist_roll").
    prev_waist_rad: dict[str, float] = field(default_factory=dict)
    prev_t_mono: Optional[float] = None
    # Rolling per-second telemetry (used for the once-per-second SUMMARY).
    frame_count_since_summary: int = 0
    max_dyaw_since_summary: float = 0.0
    # Per-joint max-|delta| in the current summary window.
    max_dwaist_since_summary: dict[str, float] = field(default_factory=dict)
    last_summary_t: float = field(default_factory=time.monotonic)
    last_message_t: float = field(default_factory=time.monotonic)
    holding: bool = False  # last loop already printed [HOLD]


def _wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def _xyzw_yaw_deg(q_xyzw: np.ndarray) -> float:
    return math.degrees(float(yaw_of_quat_xyzw(q_xyzw.astype(np.float64))))


def _decode_packed_frame(
    payload: bytes,
) -> tuple[Optional[float], dict[str, float]]:
    """Return ``(root_yaw_deg, {waist_joint_name: rad})`` for a packed frame.

    Root yaw may be ``None`` if the wire shape omits ``root_quat_xyzw``.
    The waist dict carries any of the three waist joints that ``joint_pos_mj``
    has room for; an empty dict means no joint info on this frame.
    """
    try:
        msg = unpack_message(payload, expected_topic=None)
    except ValueError:
        return None, {}

    root_yaw_deg: Optional[float] = None
    waist: dict[str, float] = {}

    q = msg.fields.get("root_quat_xyzw")
    if q is not None and q.shape == (4,):
        try:
            root_yaw_deg = math.degrees(
                float(yaw_of_quat_xyzw(q.astype(np.float64)))
            )
        except (ValueError, TypeError):
            pass

    j = msg.fields.get("joint_pos_mj")
    if j is not None:
        flat = j.reshape(-1)
        for name, slot in WAIST_SLOTS:
            if flat.size > slot:
                waist[name] = float(flat[slot])

    return root_yaw_deg, waist


def _decode_json_frame(
    payload: bytes,
) -> tuple[Optional[float], dict[str, float]]:
    """robot_pose is JSON ``{quat_wxyz, xy, z, sim_time}``.

    No joint_pos_mj on this stream (the bridge only forwards IMU
    orientation), so we return an empty waist dict and only flag
    root-yaw discontinuities here.
    """
    try:
        if payload.startswith(b"robot_pose "):
            payload = payload[len(b"robot_pose "):]
        elif payload.startswith(b"robot_pose"):
            payload = payload[len(b"robot_pose"):].lstrip()
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None, {}

    q = obj.get("quat_wxyz") or obj.get("base_quat") or obj.get("base_quat_wxyz")
    if not isinstance(q, list) or len(q) != 4:
        return None, {}
    try:
        q_arr = np.array(q, dtype=np.float64)
    except (TypeError, ValueError):
        return None, {}
    q_xyzw = _wxyz_to_xyzw(q_arr)
    try:
        yaw_deg = math.degrees(float(yaw_of_quat_xyzw(q_xyzw)))
    except (ValueError, TypeError):
        return None, {}
    return yaw_deg, {}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--body-pose-url", default="tcp://127.0.0.1:5565")
    p.add_argument("--body-pose-topic", default="body_pose")
    p.add_argument("--pose-url", default="tcp://127.0.0.1:5556")
    p.add_argument("--pose-topic", default="pose")
    p.add_argument("--robot-pose-url", default="tcp://127.0.0.1:5570")
    p.add_argument("--robot-pose-topic", default="robot_pose")

    p.add_argument(
        "--yaw-spike-deg", type=float, default=5.0,
        help="Flag any single-tick |delta root_yaw| above this (degrees). "
             "Default 5.0 deg ~= 250 deg/s at 50 Hz -- ~2x the policy's "
             "trained turn rate, so anything above this is a step input.",
    )
    p.add_argument(
        "--waist-spike-deg", type=float, default=3.0,
        help="Flag any single-tick |delta waist_yaw_joint| above this "
             "(degrees, converted to rad internally). Tighter than root "
             "yaw because the waist effector is what physically clicks.",
    )
    p.add_argument(
        "--gap-warn-ms", type=float, default=100.0,
        help="Flag any inter-frame gap > this on a stream "
             "(default 100 ms = 2 ticks at 50 Hz).",
    )
    p.add_argument(
        "--summary-every-s", type=float, default=1.0,
        help="Print [SUMMARY] line for each stream every N seconds.",
    )
    p.add_argument(
        "--no-summary", action="store_true",
        help="Suppress periodic [SUMMARY] lines; only print spikes / gaps.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    waist_spike_rad = math.radians(args.waist_spike_deg)
    gap_warn_s = args.gap_warn_ms * 1e-3

    ctx = zmq.Context.instance()
    poller = zmq.Poller()

    def _make_sub(url: str, topic: str, decoder: str, name: str) -> StreamState:
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 100)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(url)
        s.setsockopt(zmq.SUBSCRIBE, topic.encode())
        poller.register(s, zmq.POLLIN)
        print(f"[wire] {name:<11} SUB {url} topic={topic!r}", flush=True)
        return StreamState(name=name, sock=s, topic=topic, decoder=decoder)

    streams: dict[int, StreamState] = {}
    for st in (
        _make_sub(args.body_pose_url, args.body_pose_topic, "packed", "body_pose"),
        _make_sub(args.pose_url, args.pose_topic, "packed", "pose"),
        _make_sub(args.robot_pose_url, args.robot_pose_topic, "json", "robot_pose"),
    ):
        streams[st.sock.fileno()] = st  # for cleanup
        # also key by sock object identity for poll lookups:
        streams[id(st.sock)] = st

    print(
        f"[cfg] yaw_spike={args.yaw_spike_deg:.2f}deg  "
        f"waist_spike={args.waist_spike_deg:.2f}deg  "
        f"gap_warn={args.gap_warn_ms:.0f}ms",
        flush=True,
    )
    print("[ready] sniffer live; teleop now. Ctrl-C to stop.", flush=True)

    # Last-seen scalars across ALL streams for spike-time context.
    # ``waist`` is a dict keyed by joint name so the context line can
    # cleanly omit channels a particular stream doesn't carry.
    latest: dict[str, dict] = {
        "body_pose":  {"yaw": None, "waist": {}, "age_ms": None},
        "pose":       {"yaw": None, "waist": {}, "age_ms": None},
        "robot_pose": {"yaw": None, "waist": {}, "age_ms": None},
    }

    stop = {"flag": False}
    def _sig(*_a):
        stop["flag"] = True
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    def _now_str() -> str:
        return time.strftime("%H:%M:%S") + f".{int((time.time()%1)*1000):03d}"

    def _context_line(now_mono: float) -> str:
        parts = []
        for name in ("body_pose", "pose", "robot_pose"):
            d = latest[name]
            if d["yaw"] is None and not d["waist"]:
                parts.append(f"{name}=--")
                continue
            age_ms = (now_mono - d["age_ms"]) * 1e3 if d["age_ms"] is not None else 0.0
            yaw_s = f"yaw{d['yaw']:+7.2f}" if d["yaw"] is not None else "yaw=--"
            w_short = "/".join(
                f"{d['waist'][k]:+.3f}" for k, _ in WAIST_SLOTS if k in d["waist"]
            )
            w_s = f" wYPR=[{w_short}]" if w_short else ""
            parts.append(f"{name}={yaw_s}{w_s} age={age_ms:.0f}ms")
        return "  |  ".join(parts)

    while not stop["flag"]:
        try:
            events = dict(poller.poll(timeout=200))
        except KeyboardInterrupt:
            break
        except zmq.error.ZMQError:
            break

        now_mono = time.monotonic()

        for sock_obj, _ in events.items():
            st = streams.get(id(sock_obj))
            if st is None:
                continue
            try:
                msg = sock_obj.recv(flags=zmq.NOBLOCK)
            except zmq.error.Again:
                continue
            # Strip topic prefix (multi-part is one frame here; topic is
            # the first space-or-binary-header bytes).
            payload = msg
            tb = st.topic.encode()
            if payload.startswith(tb):
                payload = payload[len(tb):]
                if payload.startswith(b" "):
                    payload = payload[1:]

            if st.decoder == "packed":
                yaw_deg, waist_rad = _decode_packed_frame(payload)
            else:
                yaw_deg, waist_rad = _decode_json_frame(payload)

            if yaw_deg is None and not waist_rad:
                continue

            latest[st.name]["yaw"] = yaw_deg
            if waist_rad:
                latest[st.name]["waist"] = waist_rad
            latest[st.name]["age_ms"] = now_mono
            st.last_message_t = now_mono
            if st.holding:
                print(
                    f"[RESUME] {_now_str()} {st.name:<11} -- stream resumed after silence",
                    flush=True,
                )
                st.holding = False

            # Gap detection.
            if st.prev_t_mono is not None:
                gap_s = now_mono - st.prev_t_mono
                if gap_s > gap_warn_s:
                    print(
                        f"[GAP   ] {_now_str()} {st.name:<11} "
                        f"gap={gap_s*1e3:.0f}ms (>{args.gap_warn_ms:.0f}ms)",
                        flush=True,
                    )

            # Root-yaw delta.
            dyaw: Optional[float] = None
            if yaw_deg is not None and st.prev_yaw_deg is not None:
                dy = yaw_deg - st.prev_yaw_deg
                while dy > 180.0:
                    dy -= 360.0
                while dy < -180.0:
                    dy += 360.0
                dyaw = dy
                st.max_dyaw_since_summary = max(
                    st.max_dyaw_since_summary, abs(dy)
                )
            is_yaw_spike = dyaw is not None and abs(dyaw) > args.yaw_spike_deg

            # Per-waist-joint deltas (yaw / pitch / roll independently).
            joint_deltas: list[tuple[str, float, float, float, bool]] = []
            # tuples of (joint_name, prev_rad, curr_rad, delta_rad, is_spike)
            for joint_name, curr_rad in waist_rad.items():
                prev_rad = st.prev_waist_rad.get(joint_name)
                if prev_rad is None:
                    continue
                d = curr_rad - prev_rad
                spike = abs(d) > waist_spike_rad
                joint_deltas.append((joint_name, prev_rad, curr_rad, d, spike))
                st.max_dwaist_since_summary[joint_name] = max(
                    st.max_dwaist_since_summary.get(joint_name, 0.0), abs(d)
                )

            waist_spikes = [t for t in joint_deltas if t[4]]
            if is_yaw_spike or waist_spikes:
                gap_ms = (
                    (now_mono - st.prev_t_mono) * 1e3
                    if st.prev_t_mono is not None else float("nan")
                )
                bits = []
                if dyaw is not None and (is_yaw_spike or waist_spikes):
                    # Always show root yaw whenever ANY spike on this stream
                    # fires, so the operator can see whether the click came
                    # WITH a heading change (gesture / replan) or WITHOUT one
                    # (pure joint-level discontinuity).
                    flag = "!" if is_yaw_spike else " "
                    bits.append(
                        f"root_yaw {st.prev_yaw_deg:+7.2f}->{yaw_deg:+7.2f} "
                        f"d={dyaw:+6.2f}deg{flag}"
                    )
                for joint_name, prev_rad, curr_rad, d, spike in joint_deltas:
                    if not (spike or is_yaw_spike):
                        # When the spike was driven by a different channel,
                        # still surface the other waist joints so cross-axis
                        # leakage is visible.
                        if abs(d) < waist_spike_rad * 0.25:
                            continue
                    flag = "!" if spike else " "
                    bits.append(
                        f"{joint_name} {prev_rad:+.3f}->{curr_rad:+.3f} "
                        f"d={math.degrees(d):+6.2f}deg{flag}"
                    )
                print(
                    f"[SPIKE ] {_now_str()} {st.name:<11} gap={gap_ms:5.1f}ms  "
                    + "  ".join(bits),
                    flush=True,
                )
                print(
                    f"           context: {_context_line(now_mono)}",
                    flush=True,
                )

            if yaw_deg is not None:
                st.prev_yaw_deg = yaw_deg
            for joint_name, curr_rad in waist_rad.items():
                st.prev_waist_rad[joint_name] = curr_rad
            st.prev_t_mono = now_mono
            st.frame_count_since_summary += 1

        # Periodic per-stream summary + silence detection.
        for st in {id(s.sock): s for s in
                   (s for s in streams.values()
                    if isinstance(s, StreamState))}.values():
            silent_s = now_mono - st.last_message_t
            if silent_s > 1.0 and not st.holding:
                print(
                    f"[HOLD  ] {_now_str()} {st.name:<11} silent {silent_s:.1f}s",
                    flush=True,
                )
                st.holding = True

            if args.no_summary:
                continue
            since = now_mono - st.last_summary_t
            if since >= args.summary_every_s:
                hz = st.frame_count_since_summary / since if since > 0 else 0.0
                wbits = []
                for jn, _ in WAIST_SLOTS:
                    if jn in st.max_dwaist_since_summary:
                        wbits.append(
                            f"{jn}={math.degrees(st.max_dwaist_since_summary[jn]):5.2f}"
                        )
                wstr = " ".join(wbits) if wbits else "(no joint data)"
                print(
                    f"[SUMMRY] {_now_str()} {st.name:<11} "
                    f"rate={hz:5.1f}Hz  max|dyaw|={st.max_dyaw_since_summary:5.2f}deg  "
                    f"max|dwaist_deg|: {wstr}",
                    flush=True,
                )
                st.frame_count_since_summary = 0
                st.max_dyaw_since_summary = 0.0
                st.max_dwaist_since_summary = {}
                st.last_summary_t = now_mono

    print("[stop] shutting down sniffer", flush=True)
    for st in {id(s.sock): s for s in
               (s for s in streams.values() if isinstance(s, StreamState))}.values():
        try:
            st.sock.close(linger=0)
        except Exception:
            pass
    try:
        ctx.term()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
