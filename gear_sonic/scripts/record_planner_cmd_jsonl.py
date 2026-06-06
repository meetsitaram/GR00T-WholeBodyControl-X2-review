"""Subscribe to ``planner_cmd`` and append one JSONL row per payload.

Companion to :mod:`gear_sonic.scripts.debug_planner_cmd`. That helper
prints to stdout in a human-readable shape; this one dumps a structured
JSONL fixture you can replay through the analyzer offline. The wire is
the convergence point of the Quest 3 manager and ``x2_pkl_command_source``
PKL replay paths, so a single recorder file can capture either source.

Each row carries:

    * ``t_mono``      : ``time.monotonic()`` at receipt
    * ``wall_clock``  : ``time.time()`` for human correlation
    * ``intent``      : raw payload intent string
    * ``magnitude``   : raw payload magnitude string
    * ``stick_fwd``   : VR continuous-locomotion stick deflection [-1, 1]
    * ``stick_side``
    * ``stick_yaw``
    * ``waist_pitch_deg`` / ``waist_roll_deg`` / ``waist_yaw_deg``
    * ``target_velocity``: ``[yaw_rate, vel_x, vel_z, hip_h]`` when the
      sender shipped a raw velocity (e.g. ``x2_pkl_command_source``);
      ``null`` otherwise.
    * ``resolved_velocity``: ``[yaw_rate, vel_x, vel_z, hip_h]`` computed
      by ``intent_to_velocity(LocomotionCommand(...))`` -- the exact
      4-tuple the kplanner feeds into ``NeuralPlannerCore``. This is the
      column the analyzer keys off when comparing VR vs PKL vs the
      training distribution. Runtime scales (``--forward-scale``, etc.)
      mirror the daemon flags so the captured velocities match what the
      live deploy sees.
    * ``source_tag``  : ``"pkl"`` when ``magnitude == "raw_velocity"`` (the
      magnitude the PKL source uses by convention) else ``"vr"``.

Run on the same machine as the manager / PKL source:

    .venv/bin/python -m gear_sonic.scripts.record_planner_cmd_jsonl \\
        --out out/intent_reference/live/planner_cmd_vr.jsonl

Against a remote source / non-default port:

    .venv/bin/python -m gear_sonic.scripts.record_planner_cmd_jsonl \\
        --planner-cmd-host 192.168.1.42 --planner-cmd-port 5563 \\
        --out planner_cmd.jsonl --duration-s 75

Stops on Ctrl-C, on ``--duration-s`` expiry, or when an ``intent ==
"shutdown"`` payload arrives. Flushes + fsyncs the file on close so
partial captures are still structurally complete.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gear_sonic.scripts.x2_kplanner as kp  # noqa: E402
from gear_sonic.scripts.x2_kplanner import intent_to_velocity  # noqa: E402
from gear_sonic.utils.planner.state_machine import LocomotionCommand  # noqa: E402


log = logging.getLogger("record_planner_cmd_jsonl")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--planner-cmd-host", default="localhost",
        help="Host the manager / PKL source publishes planner_cmd on.",
    )
    p.add_argument(
        "--planner-cmd-port", type=int, default=5563,
        help="Port the manager / PKL source publishes planner_cmd on.",
    )
    p.add_argument(
        "--planner-cmd-topic", default="planner_cmd",
        help="ZMQ topic prefix (default: planner_cmd).",
    )
    p.add_argument(
        "--out", "-o", type=Path, required=True,
        help="Path to write the JSONL fixture. Parent directory is "
             "created on demand. Append mode -- pre-existing rows are "
             "preserved (run with a fresh path if that's not what you want).",
    )
    p.add_argument(
        "--duration-s", type=float, default=0.0,
        help="Exit after this many seconds of wall time (0 = run until "
             "Ctrl-C). The recording is closed cleanly either way.",
    )
    p.add_argument(
        "--rate-hz", type=float, default=0.0,
        help="If > 0, emit a WARN log if the rolling rate of received "
             "payloads drops below this for >5 s. Use to catch a silent "
             "publisher in the live session (operator confidence check).",
    )
    p.add_argument(
        "--print-every", type=int, default=0,
        help="If > 0, print one diagnostic line every N received "
             "payloads (for live tailing). Default 0 -- silent unless "
             "the rate watchdog fires.",
    )

    # Mirror the kplanner daemon's runtime scale flags so the resolved
    # velocity column matches what the live deploy receives. Same defaults
    # as ``debug_planner_cmd.py``.
    p.add_argument("--forward-scale", type=float, default=1.0)
    p.add_argument("--backward-scale", type=float, default=1.0)
    p.add_argument("--lateral-scale", type=float, default=1.0)
    p.add_argument("--turn-left-scale", type=float, default=1.0)
    p.add_argument("--turn-right-scale", type=float, default=1.0)
    p.add_argument(
        "--stick-shape-exp", type=float,
        default=kp._DEFAULT_STICK_SHAPING_EXPONENT,
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _apply_daemon_scales(args: argparse.Namespace) -> None:
    """Mutate the kplanner module-level scale globals so the recorded
    ``resolved_velocity`` column matches what the running daemon emits
    for the same payload. Mirrors the helper of the same name in
    :mod:`gear_sonic.scripts.debug_planner_cmd`.
    """
    kp._RUNTIME_FORWARD_SCALE = float(args.forward_scale)
    kp._RUNTIME_BACKWARD_SCALE = float(args.backward_scale)
    kp._RUNTIME_LATERAL_SCALE = float(args.lateral_scale)
    kp._RUNTIME_TURN_LEFT_SCALE = float(args.turn_left_scale)
    kp._RUNTIME_TURN_RIGHT_SCALE = float(args.turn_right_scale)
    if args.stick_shape_exp > 0:
        kp._RUNTIME_STICK_SHAPING_EXPONENT = float(args.stick_shape_exp)


def _resolve_velocity(payload: dict) -> tuple[float, float, float, float]:
    """Run the payload through the same dispatcher the kplanner uses.

    Honours the ``target_velocity`` short-circuit so PKL-replay rows
    carry the raw 4-tuple their source published, not the bucketed
    fallback that the kplanner would also discard.
    """
    target_velocity = payload.get("target_velocity")
    direct_velocity: tuple[float, float, float, float] | None = None
    if isinstance(target_velocity, (list, tuple)) and len(target_velocity) == 4:
        direct_velocity = (
            float(target_velocity[0]),
            float(target_velocity[1]),
            float(target_velocity[2]),
            float(target_velocity[3]),
        )

    cmd = LocomotionCommand(
        intent=str(payload.get("intent", "idle")),
        magnitude=str(payload.get("magnitude", "default")),
        waist_pitch_deg=float(payload.get("waist_pitch_deg", 0.0)),
        waist_roll_deg=float(payload.get("waist_roll_deg", 0.0)),
        waist_yaw_deg=float(payload.get("waist_yaw_deg", 0.0)),
        stick_fwd=float(payload.get("stick_fwd", 0.0)),
        stick_side=float(payload.get("stick_side", 0.0)),
        stick_yaw=float(payload.get("stick_yaw", 0.0)),
        direct_velocity=direct_velocity,
    )
    yaw_rate, vel_x, vel_z, hip_h = intent_to_velocity(cmd)
    return float(yaw_rate), float(vel_x), float(vel_z), float(hip_h)


def _source_tag(payload: dict) -> str:
    """Classify the payload sender. The PKL source publishes
    ``magnitude == "raw_velocity"`` by convention (see
    ``x2_pkl_command_source._build_payload``); everything else is the
    Quest 3 manager or a hand-launched debug source.
    """
    if str(payload.get("magnitude", "")).lower() == "raw_velocity":
        return "pkl"
    return "vr"


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def _open_out(path: Path):
    """Open the output JSONL in append mode, creating parents.

    We use line buffering so a Ctrl-C between writes still leaves
    structurally-valid JSONL on disk (each line is fully flushed). The
    final close() in the recorder flushes + fsyncs explicitly so even a
    SIGKILL-from-tmux doesn't lose the last few rows.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("a", buffering=1)


def _close_out(f, count: int, path: Path) -> None:
    try:
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass
        f.close()
        log.info("closed %s after %d rows", path, count)
    except Exception as exc:  # noqa: BLE001
        log.warning("close failed: %s", exc)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    _apply_daemon_scales(args)

    endpoint = f"tcp://{args.planner_cmd_host}:{args.planner_cmd_port}"
    log.info(
        "subscribing to %s topic=%r -> %s",
        endpoint, args.planner_cmd_topic, args.out,
    )
    log.info(
        "daemon-mirror scales: forward=%.2f backward=%.2f lateral=%.2f "
        "turn_left=%.2f turn_right=%.2f stick_shape_exp=%.2f",
        args.forward_scale, args.backward_scale, args.lateral_scale,
        args.turn_left_scale, args.turn_right_scale,
        kp._RUNTIME_STICK_SHAPING_EXPONENT,
    )

    f = _open_out(args.out)
    stop_event = threading.Event()

    def _on_signal(signum, _frame):
        log.info("signal %d received; closing recorder", signum)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 200)  # 200 ms so we can poll stop_event
    sock.connect(endpoint)
    sock.setsockopt_string(zmq.SUBSCRIBE, args.planner_cmd_topic)

    deadline = (
        time.monotonic() + args.duration_s if args.duration_s > 0 else None
    )
    count = 0
    last_beacon = time.monotonic()
    last_rate_check = time.monotonic()
    rate_window_count = 0
    rate_warned = False
    try:
        while not stop_event.is_set():
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                log.info("duration_s elapsed; closing recorder")
                break
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                # Watchdog: every 5 s, warn if --rate-hz was set and we
                # received fewer payloads than expected (i.e. the source
                # has gone quiet).
                if args.rate_hz > 0 and (now - last_rate_check) >= 5.0:
                    elapsed = now - last_rate_check
                    observed = rate_window_count / max(elapsed, 1e-6)
                    if observed < args.rate_hz:
                        log.warning(
                            "[rate-watchdog] observed %.1f Hz < target "
                            "%.1f Hz over last %.1fs (rows so far: %d)",
                            observed, args.rate_hz, elapsed, count,
                        )
                        rate_warned = True
                    last_rate_check = now
                    rate_window_count = 0
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                log.warning("bad payload %r: %s", parts[1], exc)
                continue

            intent = str(payload.get("intent", ""))
            if intent == "shutdown":
                log.info("shutdown payload received; closing recorder")
                break

            try:
                yaw_rate, vel_x, vel_z, hip_h = _resolve_velocity(payload)
            except Exception as exc:  # noqa: BLE001
                log.warning("resolve velocity failed: %s; payload=%r", exc, payload)
                continue

            target_velocity = payload.get("target_velocity")
            if isinstance(target_velocity, (list, tuple)):
                target_velocity_out = [float(x) for x in target_velocity]
            else:
                target_velocity_out = None

            rec = {
                "t_mono": float(now),
                "wall_clock": time.time(),
                "intent": intent,
                "magnitude": str(payload.get("magnitude", "default")),
                "stick_fwd": float(payload.get("stick_fwd", 0.0)),
                "stick_side": float(payload.get("stick_side", 0.0)),
                "stick_yaw": float(payload.get("stick_yaw", 0.0)),
                "waist_pitch_deg": float(payload.get("waist_pitch_deg", 0.0)),
                "waist_roll_deg": float(payload.get("waist_roll_deg", 0.0)),
                "waist_yaw_deg": float(payload.get("waist_yaw_deg", 0.0)),
                "target_velocity": target_velocity_out,
                "resolved_velocity": [yaw_rate, vel_x, vel_z, hip_h],
                "source_tag": _source_tag(payload),
            }
            # Preserve any advisory fields the source attaches without
            # exploding the schema (e.g. PKL source's frame_index / clip_id).
            for k in ("frame_index", "clip_id"):
                if k in payload:
                    rec[k] = payload[k]

            f.write(json.dumps(rec) + "\n")
            count += 1
            rate_window_count += 1

            if args.print_every > 0 and count % args.print_every == 0:
                log.info(
                    "rows=%d intent=%s magnitude=%s vel=(%.3f, %.3f, %.3f, %.3f)",
                    count, rec["intent"], rec["magnitude"],
                    yaw_rate, vel_x, vel_z, hip_h,
                )

            # Periodic row-count beacon every ~5 s (independent of
            # --print-every) so operators tailing the recorder log see
            # liveness without enabling per-row prints.
            if (now - last_beacon) >= 5.0:
                log.info(
                    "rows=%d (last 5s: %d) source_tag=%s",
                    count, rate_window_count, rec["source_tag"],
                )
                last_beacon = now
                # Reset rate window only if the watchdog is disabled --
                # otherwise the watchdog's own timer drives the reset
                # to keep the 5 s windows aligned.
                if args.rate_hz <= 0:
                    rate_window_count = 0
                    last_rate_check = now
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt; closing recorder")
    finally:
        try:
            sock.close(linger=0)
        except Exception:
            pass
        _close_out(f, count, args.out)

    if rate_warned:
        log.warning(
            "exited with rate-watchdog warnings -- the source went "
            "quiet at least once; spot-check the fixture before "
            "treating it as a reference recording."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
