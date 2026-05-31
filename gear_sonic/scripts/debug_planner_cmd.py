"""Subscribe to ``planner_cmd`` and print every publication in real time.

Standalone diagnostic for the Quest 3 -> kplanner wire. Subscribes to the
same ZMQ topic ``quest3_manager_x2`` publishes on (default ``planner_cmd``
at ``tcp://localhost:5563``), parses each JSON payload, and prints the
intent / magnitude / continuous fields plus the kplanner's resolved
4-D velocity for ``locomotion / continuous`` and bucketed intents alike.

Use case: operator pushes the Quest 3 stick and we want to know whether
(a) the manager is actually emitting what we expect, (b) the deadzone
rescale produces the values we expect, and (c) the kplanner's intent
dispatcher resolves them to a non-zero velocity. Diagnoses the
"struggling to move forward" failure mode where the controller's
analog stick output and the planner's velocity intent get out of sync.

Run on the same machine as the manager (typical setup):

    .venv/bin/python -m gear_sonic.scripts.debug_planner_cmd

Or against a remote manager / non-default port:

    .venv/bin/python -m gear_sonic.scripts.debug_planner_cmd \
        --planner-cmd-host 192.168.1.42 --planner-cmd-port 5563

The kplanner velocity resolution mirrors ``intent_to_velocity`` so the
numbers printed here equal what the daemon would feed into
``replan_with_velocity``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gear_sonic.scripts.x2_kplanner as kp  # noqa: E402
from gear_sonic.scripts.x2_kplanner import intent_to_velocity  # noqa: E402
from gear_sonic.utils.planner.state_machine import LocomotionCommand  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planner-cmd-host", default="localhost",
                   help="Host the manager publishes planner_cmd on (default: localhost).")
    p.add_argument("--planner-cmd-port", type=int, default=5563,
                   help="Port the manager publishes planner_cmd on (default: 5563).")
    p.add_argument("--planner-cmd-topic", default="planner_cmd",
                   help="ZMQ topic prefix (default: planner_cmd).")
    p.add_argument("--duration-s", type=float, default=0.0,
                   help="Exit after this many seconds (0 = run until Ctrl-C).")
    p.add_argument("--show-idle", action="store_true",
                   help="Also print idle / no-velocity emissions (otherwise filtered).")
    # Mirror x2_kplanner.py daemon flags so the diagnostic reports the
    # velocity the deploy actually sees, not the unscaled raw value.
    # Pass the same scales you launched run_x2_quest3_planner_stack.sh
    # with (e.g. ``--forward-scale 0.6``).
    p.add_argument("--forward-scale", type=float, default=1.0,
                   help="Multiplier on positive vel_z (mirror of "
                        "kplanner --forward-scale). Default 1.0.")
    p.add_argument("--backward-scale", type=float, default=1.0,
                   help="Multiplier on negative vel_z. Default 1.0.")
    p.add_argument("--lateral-scale", type=float, default=1.0,
                   help="Multiplier on vel_x. Default 1.0.")
    p.add_argument("--turn-left-scale", type=float, default=1.0,
                   help="Multiplier on positive yaw_rate (left turn). Default 1.0.")
    p.add_argument("--turn-right-scale", type=float, default=1.0,
                   help="Multiplier on negative yaw_rate (right turn). Default 1.0.")
    p.add_argument("--stick-shape-exp", type=float,
                   default=kp._DEFAULT_STICK_SHAPING_EXPONENT,
                   help="Stick shape exponent (mirror of kplanner "
                        "--stick-shape-exp). Default linear (1.0).")
    return p.parse_args(argv)


def _apply_daemon_scales(args: argparse.Namespace) -> None:
    """Mutate the kplanner module-level globals used by
    ``intent_to_velocity`` so the displayed velocity matches what
    the running daemon would emit for the same payload.
    """
    kp._RUNTIME_FORWARD_SCALE = float(args.forward_scale)
    kp._RUNTIME_BACKWARD_SCALE = float(args.backward_scale)
    kp._RUNTIME_LATERAL_SCALE = float(args.lateral_scale)
    kp._RUNTIME_TURN_LEFT_SCALE = float(args.turn_left_scale)
    kp._RUNTIME_TURN_RIGHT_SCALE = float(args.turn_right_scale)
    if args.stick_shape_exp > 0:
        kp._RUNTIME_STICK_SHAPING_EXPONENT = float(args.stick_shape_exp)


def _format_payload(payload: dict) -> str:
    intent = payload.get("intent", "?")
    magnitude = payload.get("magnitude", "?")
    parts = [f"intent={intent:<10}", f"magnitude={magnitude:<11}"]
    if intent == "locomotion":
        sf = float(payload.get("stick_fwd", 0.0))
        ss = float(payload.get("stick_side", 0.0))
        sy = float(payload.get("stick_yaw", 0.0))
        parts.append(
            f"stick=(fwd={sf:+5.2f}, side={ss:+5.2f}, yaw={sy:+5.2f})"
        )
    elif intent == "hold_torso":
        pitch = float(payload.get("waist_pitch_deg", 0.0))
        roll = float(payload.get("waist_roll_deg", 0.0))
        yaw = float(payload.get("waist_yaw_deg", 0.0))
        parts.append(
            f"waist=(pitch={pitch:+5.1f}, roll={roll:+5.1f}, yaw={yaw:+5.1f})deg"
        )
    return "  ".join(parts)


def _format_velocity(payload: dict) -> str:
    cmd = LocomotionCommand(
        intent=str(payload.get("intent", "idle")),
        magnitude=str(payload.get("magnitude", "default")),
        waist_pitch_deg=float(payload.get("waist_pitch_deg", 0.0)),
        waist_roll_deg=float(payload.get("waist_roll_deg", 0.0)),
        waist_yaw_deg=float(payload.get("waist_yaw_deg", 0.0)),
        stick_fwd=float(payload.get("stick_fwd", 0.0)),
        stick_side=float(payload.get("stick_side", 0.0)),
        stick_yaw=float(payload.get("stick_yaw", 0.0)),
    )
    yaw_rate, vel_x, vel_z, hip_h = intent_to_velocity(cmd)
    return (
        f"vel=(yaw_rate={yaw_rate:+6.3f}rad/s, "
        f"vel_x={vel_x:+5.3f}m/s, "
        f"vel_z={vel_z:+5.3f}m/s, "
        f"hip_h={hip_h:.2f}m)"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _apply_daemon_scales(args)
    endpoint = f"tcp://{args.planner_cmd_host}:{args.planner_cmd_port}"
    print(f"[debug_planner_cmd] subscribing to {endpoint} "
          f"topic={args.planner_cmd_topic!r}")
    print(
        f"[debug_planner_cmd] daemon-mirror scales: "
        f"forward={args.forward_scale:.2f} backward={args.backward_scale:.2f} "
        f"lateral={args.lateral_scale:.2f} "
        f"turn_left={args.turn_left_scale:.2f} turn_right={args.turn_right_scale:.2f} "
        f"stick_shape_exp={kp._RUNTIME_STICK_SHAPING_EXPONENT:.2f}"
    )
    print("[debug_planner_cmd] columns: time intent magnitude (stick/waist) -> "
          "resolved velocity (post-scale, i.e. what the deploy receives)")
    print("-" * 100)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.RCVTIMEO, 500)
    sock.connect(endpoint)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.planner_cmd_topic)

    deadline: float | None = (
        time.monotonic() + args.duration_s if args.duration_s > 0 else None
    )
    t0 = time.monotonic()
    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                print(f"[bad payload] {parts[1]!r}: {exc}")
                continue

            intent = payload.get("intent", "?")
            if not args.show_idle and intent == "idle":
                continue
            t = time.monotonic() - t0
            print(f"t={t:7.2f}s  {_format_payload(payload)}  ->  "
                  f"{_format_velocity(payload)}")
            sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n[debug_planner_cmd] Ctrl-C, exiting.")
    finally:
        sock.close(linger=0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
