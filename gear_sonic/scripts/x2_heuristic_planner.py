"""Heuristic locomotion planner daemon for AgiBot X2 Ultra.

Streams 31-DOF reference body poses + root quaternion at 50 Hz on the ``pose``
ZMQ topic, the same wire format as ``mock_vla_publish_stand_token.py``. The
X2 deploy's ``ZmqPoseInputSource`` consumes it, finite-differences velocities,
and its internal encoder tokenizes the stream for the Sonic policy.

Source of motion: the curator-built primitive PKL
(``gear_sonic/data/motions/x2_planner_primitives.pkl``). Source of bin
metadata: the bins YAML (``gear_sonic/data/motions/x2_planner_bins.yaml``).

Command surface (any combination, behind one queue):
  - ``--demo PATH.yaml`` : scripted command sequence
  - ``--keyboard``       : interactive keyboard mode (TTY only)
  - ``--zmq-cmd-port``   : ZMQ SUB on ``planner_cmd`` topic for external drive

Process hygiene (per planner plan section 5):
  - PID file written to ``/tmp/x2_heuristic_planner.pid`` by default, removed
    on clean exit.
  - SIGINT and SIGTERM cleanly drain the publisher and tear down ZMQ.
  - Pre-flight checks the publish port is free, fail-fast if not.

Run from the repo root::

    .venv/bin/python -m gear_sonic.scripts.x2_heuristic_planner \\
        --demo gear_sonic/data/scripted_demos/forward_back_turn.yaml

For sim integration, use ``gear_sonic/scripts/run_planner_smoke.sh`` rather
than launching this directly — it manages the deploy / planner / debug
processes together with trap-based cleanup.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.planner.registry import load_bin_specs  # noqa: E402
from gear_sonic.utils.planner.state_machine import (  # noqa: E402
    HeuristicPlanner,
    LocomotionCommand,
    OUTPUT_FPS,
    PlannerState,
    StreamFrame,
    build_pose_payload,
    commands_from_yaml,
    load_primitives_pkl,
    required_bins_summary,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402


log = logging.getLogger("x2_planner")


# ---------------------------------------------------------------------------
# Process hygiene helpers
# ---------------------------------------------------------------------------


def _port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Return True if ``host:port`` already has a TCP listener.

    Uses bind() probe rather than ``lsof`` so the check runs without root and
    on minimal containers.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind((host, port))
    except OSError:
        return True
    finally:
        s.close()
    return False


class PidFile:
    """Context manager that owns ``/tmp/...pid`` for the lifetime of the daemon."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __enter__(self) -> "PidFile":
        if self.path.exists():
            try:
                old = int(self.path.read_text().strip())
                if _pid_alive(old):
                    raise RuntimeError(
                        f"PID file {self.path} exists and PID {old} is alive — "
                        f"another planner is running. Run "
                        f"`kill {old}` or use `run_planner_smoke.sh --cleanup-only`."
                    )
                log.warning(
                    "stale PID file %s for dead PID %d, cleaning up", self.path, old
                )
            except ValueError:
                log.warning("bad PID file %s, cleaning up", self.path)
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(str(os.getpid()))
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class PosePublisher:
    """Publishes pose payloads on the ``pose`` ZMQ topic."""

    def __init__(
        self,
        host: str,
        port: int,
        topic: str = "pose",
        version: int = 4,
        hand_dof: int = 10,
        motion_token_dim: int = 64,
    ) -> None:
        import zmq
        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{port}")
        self._topic = topic
        self._version = version
        self._hand_dof = hand_dof
        self._motion_token_dim = motion_token_dim
        # PUB-SUB warmup: many subscribers miss the first few frames.
        time.sleep(0.1)

    def publish(
        self,
        frame: StreamFrame,
        future_frames: list[StreamFrame] | None = None,
        future_dt_s: float = 0.1,
    ) -> None:
        """Publish a pose message; ``future_frames`` carries the policy's lookahead."""
        payload = build_pose_payload(
            frame,
            motion_token_dim=self._motion_token_dim,
            hand_dof=self._hand_dof,
            future_frames=future_frames,
            future_dt_s=future_dt_s,
        )
        msg = pack_pose_message(payload, topic=self._topic, version=self._version)
        self._sock.send(msg)

    def close(self) -> None:
        self._sock.close(linger=0)


# ---------------------------------------------------------------------------
# Command sources
# ---------------------------------------------------------------------------


KEYBOARD_HELP = """
X2 heuristic planner — keyboard commands:
   w        fwd_walk_standard
   W / b    back_step_half_ft
   f        fwd_step_half_ft
   F        fwd_step_1ft
   r        fwd_step_quarter_ft
   a / d    side_left_step / side_right_step  (single canonical side step)
   q / e    turn_left_45deg / turn_right_45deg
   Q / E    turn_left_90deg / turn_right_90deg
   1 / 3    turn_left_15deg / turn_right_15deg
   2 / 4    turn_left_30deg / turn_right_30deg
   k / l / L    lean_fwd small / medium / large
   j / ;        lean_left_medium / lean_right_medium
   J / :        lean_left_large  / lean_right_large
   ,        torso_left_30deg   (< = torso_left_40deg)
   .        torso_right_30deg  (> = torso_right_40deg)

Continuous waist hold (STATIC_HOLD) -- nudges a single target pose
that the planner slews to and holds indefinitely. Use this to verify
the VR right-stick path from a TTY:
   T        enter hold at neutral (resets continuous target to 0,0,0)
   i / o    pitch -2 / +2 deg  (forward lean)
   u / p    yaw   -5 / +5 deg  (twist left / right)
   y / h    roll  -2 / +2 deg  (lean left / right)
   space    idle_stand          (also exits STATIC_HOLD)
   x        quit

Each keypress REPLACES any pending commands -- the latest press wins.
The currently-playing primitive finishes naturally (no preemption
mid-stride for safety); your new command runs after that.
"""

KEYBOARD_MAP: dict[str, tuple[str, str]] = {
    "w": ("walk", "forward"),
    "W": ("back_step", "half_ft"),
    "b": ("back_step", "half_ft"),
    "B": ("back_step", "quarter_ft"),
    "f": ("fwd_step", "half_ft"),
    "F": ("fwd_step", "one_ft"),
    "r": ("fwd_step", "quarter_ft"),
    "a": ("side_left", "default"),
    "d": ("side_right", "default"),
    "1": ("turn_left", "deg_15"),
    "2": ("turn_left", "deg_30"),
    "q": ("turn_left", "deg_45"),
    "Q": ("turn_left", "deg_90"),
    "3": ("turn_right", "deg_15"),
    "4": ("turn_right", "deg_30"),
    "e": ("turn_right", "deg_45"),
    "E": ("turn_right", "deg_90"),
    "k": ("lean_fwd", "small"),
    "l": ("lean_fwd", "medium"),
    "L": ("lean_fwd", "large"),
    "j": ("lean_left", "medium"),
    "J": ("lean_left", "large"),
    ";": ("lean_right", "medium"),
    ":": ("lean_right", "large"),
    ",": ("torso_left", "deg_30"),
    "<": ("torso_left", "deg_40"),
    ".": ("torso_right", "deg_30"),
    ">": ("torso_right", "deg_40"),
    " ": ("idle", "default"),
}

# Per-keypress nudge step (degrees) for the continuous-hold debug keys.
# Smaller for pitch/roll because the safety caps there are tight (20/10).
_HOLD_NUDGE_PITCH_DEG: float = 2.0
_HOLD_NUDGE_ROLL_DEG: float = 2.0
_HOLD_NUDGE_YAW_DEG: float = 5.0
# Maps a keypress to (axis, signed_step_deg). The keyboard thread keeps
# a local (pitch, roll, yaw) state and re-emits a hold_torso command
# with the updated target after each nudge.
_HOLD_NUDGE_KEYS: dict[str, tuple[str, float]] = {
    "i": ("pitch", -_HOLD_NUDGE_PITCH_DEG),
    "o": ("pitch", +_HOLD_NUDGE_PITCH_DEG),
    "u": ("yaw", -_HOLD_NUDGE_YAW_DEG),
    "p": ("yaw", +_HOLD_NUDGE_YAW_DEG),
    "y": ("roll", -_HOLD_NUDGE_ROLL_DEG),
    "h": ("roll", +_HOLD_NUDGE_ROLL_DEG),
}


def _scripted_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    yaml_path: Path,
    stop_event: threading.Event,
) -> None:
    cmds = commands_from_yaml(yaml_path)
    log.info("scripted source: queueing %d commands from %s", len(cmds), yaml_path)
    for cmd in cmds:
        if stop_event.is_set():
            return
        cmd_queue.put(cmd)
    log.info("scripted source: done")


def _zmq_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    host: str,
    port: int,
    topic: str,
    stop_event: threading.Event,
) -> None:
    """Subscribe to a JSON-payload command topic.

    Wire format: ZMQ multipart
      - frame 0: ``topic.encode()``
      - frame 1: utf-8 JSON ``{"intent": "...", "magnitude": "..."}``

    Subscriber sends ``{"intent": "shutdown"}`` to stop the daemon.
    """
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    log.info("zmq command source: SUB %s on tcp://%s:%d", topic, host, port)
    try:
        while not stop_event.is_set():
            try:
                parts = sock.recv_multipart()
            except zmq.error.Again:
                continue
            if len(parts) < 2:
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
                intent = str(payload["intent"])
                if intent == "shutdown":
                    log.info("zmq command source: shutdown received")
                    stop_event.set()
                    continue
                magnitude = str(payload.get("magnitude", "default"))
                # Optional continuous waist targets (degrees). Only the
                # ``hold_torso`` intent reads them; for every other
                # intent the LocomotionCommand defaults (0.0) are
                # ignored downstream by the state machine.
                waist_pitch_deg = float(payload.get("waist_pitch_deg", 0.0))
                waist_roll_deg = float(payload.get("waist_roll_deg", 0.0))
                waist_yaw_deg = float(payload.get("waist_yaw_deg", 0.0))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                log.warning("zmq command source: bad payload %r: %s", parts[1], exc)
                continue
            cmd_queue.put(
                LocomotionCommand(
                    intent=intent,
                    magnitude=magnitude,
                    source="zmq",
                    waist_pitch_deg=waist_pitch_deg,
                    waist_roll_deg=waist_roll_deg,
                    waist_yaw_deg=waist_yaw_deg,
                )
            )
    finally:
        sock.close(linger=0)


def _keyboard_command_thread(
    cmd_queue: "queue.Queue[LocomotionCommand]",
    stop_event: threading.Event,
) -> None:
    """Single-character TTY input. Requires stdin to be a real TTY.

    Most keys map 1:1 through ``KEYBOARD_MAP`` to a discrete
    ``(intent, magnitude)`` pair. The continuous-hold debug keys
    (``T``, ``i/o``, ``u/p``, ``y/h``) maintain a local
    ``(pitch, roll, yaw)`` target and re-emit ``hold_torso`` commands
    after each nudge so the operator can drive STATIC_HOLD without
    needing the VR stack.
    """
    if not sys.stdin.isatty():
        log.error("keyboard source: stdin is not a TTY; refusing to start")
        return
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    # Local hold target (degrees) -- updated by 'T' (reset to 0) and the
    # nudge keys.
    hold_target: dict[str, float] = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}

    def _emit_hold() -> None:
        cmd_queue.put(
            LocomotionCommand(
                intent="hold_torso",
                magnitude="continuous",
                source="kbd",
                waist_pitch_deg=hold_target["pitch"],
                waist_roll_deg=hold_target["roll"],
                waist_yaw_deg=hold_target["yaw"],
            )
        )
        print(
            "  -> hold_torso "
            f"pitch={hold_target['pitch']:+.1f} "
            f"roll={hold_target['roll']:+.1f} "
            f"yaw={hold_target['yaw']:+.1f}",
            flush=True,
        )

    try:
        tty.setcbreak(fd)
        print(KEYBOARD_HELP, flush=True)
        while not stop_event.is_set():
            ch = sys.stdin.read(1)
            if ch in ("x", "\x03"):  # x or Ctrl-C
                stop_event.set()
                return
            if ch == "T":
                hold_target.update(pitch=0.0, roll=0.0, yaw=0.0)
                _emit_hold()
                continue
            if ch in _HOLD_NUDGE_KEYS:
                axis, step = _HOLD_NUDGE_KEYS[ch]
                hold_target[axis] += step
                _emit_hold()
                continue
            if ch in KEYBOARD_MAP:
                intent, mag = KEYBOARD_MAP[ch]
                cmd_queue.put(
                    LocomotionCommand(intent=intent, magnitude=mag, source="kbd")
                )
                print(f"  -> {intent} {mag}", flush=True)
            else:
                print(KEYBOARD_HELP, flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


def run(
    primitives_pkl: Path,
    bins_yaml: Path,
    pub_host: str,
    pub_port: int,
    pid_file: Path,
    demo_yaml: Optional[Path],
    enable_keyboard: bool,
    zmq_cmd_host: Optional[str],
    zmq_cmd_port: Optional[int],
    zmq_cmd_topic: str,
    duration_s: float,
    hand_dof: int,
    verbose: bool,
    warmup_quiet_stand_s: float = 2.0,
    body_pose_port: Optional[int] = None,
) -> int:
    _setup_logging(verbose)

    # ---- Phase 0 architecture: when ``body_pose_port`` is set, the
    # planner publishes its 31-DOF reference on a NON-deploy port
    # under topic ``body_pose``. The recorder subscribes here, merges
    # in operator-driven ``arm_targets`` from the manager, and forwards
    # ``final_pose`` to the deploy on port 5556. Today's
    # "publish straight to deploy" workflow continues to work with
    # ``--pub-port 5556`` (topic ``pose``); the two modes are mutually
    # exclusive at the CLI layer (``body_pose_port`` overrides
    # ``pub_port`` + topic when both are passed).
    if body_pose_port is not None:
        effective_port = body_pose_port
        effective_topic = "body_pose"
        log.info(
            "Phase 0 mode: publishing 'body_pose' to recorder for merging "
            "(port %d). Pass --pub-port without --body-pose-port to fall "
            "back to direct-to-deploy 'pose' on 5556.",
            effective_port,
        )
    else:
        effective_port = pub_port
        effective_topic = "pose"

    # ---- Pre-flight checks
    if _port_in_use(effective_port, "127.0.0.1") or _port_in_use(effective_port, "0.0.0.0"):
        log.error(
            "publish port %d already in use. Run "
            "`gear_sonic/scripts/run_planner_smoke.sh --cleanup-only` first.",
            effective_port,
        )
        return 1
    if not primitives_pkl.exists():
        log.error("primitives PKL not found: %s — run curate_x2_primitives first", primitives_pkl)
        return 1
    if not bins_yaml.exists():
        log.error("bins YAML not found: %s", bins_yaml)
        return 1

    bin_specs = load_bin_specs(bins_yaml)
    bin_family = {name: spec.family for name, spec in bin_specs.items()}
    primitives = load_primitives_pkl(primitives_pkl, bin_family)
    summary = required_bins_summary(primitives)
    log.info(
        "loaded %d primitives (%d partial). Families: %s",
        summary["n_total"],
        summary["n_partial"],
        {fam: len(b) for fam, b in summary["by_family"].items()},
    )
    if summary["missing_idle_stand"]:
        log.error("primitives PKL missing 'idle_stand' — cannot run.")
        return 1

    planner = HeuristicPlanner(primitives=primitives)

    publisher = PosePublisher(
        host=pub_host,
        port=effective_port,
        topic=effective_topic,
        hand_dof=hand_dof,
    )
    log.info(
        "publishing %r on tcp://%s:%d at %.1f Hz",
        effective_topic, pub_host, effective_port, OUTPUT_FPS,
    )

    # ---- Command sources
    cmd_queue: "queue.Queue[LocomotionCommand]" = queue.Queue()
    stop_event = threading.Event()
    threads: list[threading.Thread] = []

    if demo_yaml is not None:
        t = threading.Thread(
            target=_scripted_command_thread,
            args=(cmd_queue, demo_yaml, stop_event),
            name="cmd-scripted",
            daemon=True,
        )
        t.start()
        threads.append(t)

    if zmq_cmd_host is not None and zmq_cmd_port is not None:
        t = threading.Thread(
            target=_zmq_command_thread,
            args=(cmd_queue, zmq_cmd_host, zmq_cmd_port, zmq_cmd_topic, stop_event),
            name="cmd-zmq",
            daemon=True,
        )
        t.start()
        threads.append(t)

    if enable_keyboard:
        t = threading.Thread(
            target=_keyboard_command_thread,
            args=(cmd_queue, stop_event),
            name="cmd-kbd",
            daemon=True,
        )
        t.start()
        threads.append(t)

    # ---- Signal handlers
    def _on_signal(signum: int, _frame: object) -> None:
        log.info("signal %d -> draining and shutting down", signum)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _on_signal)

    # ---- Main 50 Hz loop
    period_s = 1.0 / OUTPUT_FPS
    next_tick = time.monotonic()
    end_at = (
        time.monotonic() + duration_s if duration_s > 0 else float("inf")
    )

    pid_path = pid_file
    try:
        with PidFile(pid_path):
            # ---- Optional quiet-stand warmup
            #
            # Publish the planner's *canonical anchor frame* -- frozen --
            # for the first ``warmup_quiet_stand_s`` seconds. This is the
            # pose the state machine will emit on its first real tick:
            # idle_stand[0] aligned to (xy=0, yaw=0). Holding it constant
            # for ~2s gives the policy a stable reference to settle against
            # while the C++ deploy finishes its INIT->WAIT->CONTROL ramp.
            #
            # CRITICAL: this anchor is also what
            # ``bake_planner_rsi_anchor.py`` writes into the RSI PKL the
            # bridge spawns from under ``--sim-profile parity``. Bridge
            # spawn pose == warmup wire content == state-machine first
            # frame -> bit-identical, no tracker-error fight on tick 0.
            #
            # Why NOT DEFAULT_STAND_POSE_NP + identity quat (the previous
            # behaviour): idle_stand[0]'s joints differ from
            # DEFAULT_STAND_POSE_NP by up to ~33 deg on waist_yaw (the
            # idle clip was captured with a small twist), so the warmup
            # would lie about the body pose and force a discontinuity at
            # the warmup -> state-machine handover.
            #
            # Disable with ``--warmup-quiet-stand-s 0`` (e.g. for kinematic
            # viewer runs that don't involve a deploy).
            warmup_n = (
                int(round(max(0.0, warmup_quiet_stand_s) * OUTPUT_FPS))
                if warmup_quiet_stand_s > 0 else 0
            )
            if warmup_n > 0:
                anchor = planner.current_anchor_frame()
                log.info(
                    "quiet-stand warmup: %d ticks (%.2fs) of frozen "
                    "anchor frame from bin '%s' (bit-identical to RSI)",
                    warmup_n, warmup_quiet_stand_s, anchor.bin_name,
                )
                anchor_dof = anchor.joint_pos_mj.copy()
                anchor_quat = anchor.root_quat_xyzw.copy()
                anchor_xy = anchor.root_xy_world.copy()
                anchor_yaw_deg = float(anchor.yaw_world_deg)
                for warm_idx in range(warmup_n):
                    if stop_event.is_set() or time.monotonic() >= end_at:
                        break
                    frame = StreamFrame(
                        joint_pos_mj=anchor_dof,
                        root_quat_xyzw=anchor_quat,
                        root_xy_world=anchor_xy.copy(),
                        yaw_world_deg=anchor_yaw_deg,
                        state=PlannerState.IDLE_LOOP,
                        bin_name="warmup_anchor",
                        frame_index=warm_idx,
                        seam_blend=False,
                    )
                    publisher.publish(frame)
                    next_tick += period_s
                    slack = next_tick - time.monotonic()
                    if slack > 0:
                        time.sleep(slack)
                    else:
                        next_tick = time.monotonic()
                log.info("quiet-stand warmup done; entering state machine.")

            while not stop_event.is_set() and time.monotonic() < end_at:
                # Drain command queue first so commands are picked up promptly.
                #
                # Dispatch by source: keyboard and ZMQ teleop are
                # *interactive* sources, so the latest command should win
                # over any backlog (user pressed 'a' then 'd' in 100ms ->
                # they want 'd', not both). Scripted YAML commands are
                # batch-loaded and must accumulate to play the demo.
                while True:
                    try:
                        cmd = cmd_queue.get_nowait()
                    except queue.Empty:
                        break
                    if cmd.source in ("kbd", "zmq"):
                        dropped = planner.replace_pending(cmd)
                        if dropped > 0:
                            log.info(
                                "interactive command (%s, %s) from %s -- "
                                "dropped %d pending command(s)",
                                cmd.intent, cmd.magnitude, cmd.source, dropped,
                            )
                        else:
                            log.info(
                                "interactive command (%s, %s) from %s",
                                cmd.intent, cmd.magnitude, cmd.source,
                            )
                    else:
                        log.info(
                            "queued command (%s, %s) from %s",
                            cmd.intent, cmd.magnitude, cmd.source,
                        )
                        planner.enqueue(cmd)
                # Emit the current tick AND a 9-frame future window so the
                # C++ tokenizer obs has its full 10-frame look-ahead. Without
                # this, ZmqPoseInputSource::Sample(t) can only return the
                # latest received frame regardless of the requested time --
                # the policy then sees ``future == current`` 10 times and
                # under-commits to dynamic gait (proven empirically; see
                # docs/source/dev_notes 2026-05-11). Future window matches
                # NUM_FUTURE_FRAMES=10 / DT_FUTURE_REF=0.1 in the deploy.
                frame, future = planner.step_with_lookahead(
                    num_future=9, step_ticks=5
                )
                publisher.publish(frame, future_frames=future, future_dt_s=0.1)

                # Sleep to next tick boundary; never spin.
                next_tick += period_s
                slack = next_tick - time.monotonic()
                if slack > 0:
                    time.sleep(slack)
                else:
                    # Fell behind by more than one tick — log and resync.
                    if -slack > 5 * period_s:
                        log.warning("loop fell behind by %.0fms; resyncing", -slack * 1000)
                        next_tick = time.monotonic()
            log.info("main loop exited at tick %d", planner.tick)
    finally:
        stop_event.set()
        publisher.close()
        for t in threads:
            t.join(timeout=2.0)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="x2_heuristic_planner",
        description=(
            "Heuristic locomotion planner for AgiBot X2 — streams 31-DOF body "
            "refs + root quat over the 'pose' ZMQ topic at 50 Hz."
        ),
    )
    p.add_argument(
        "--primitives",
        type=Path,
        default=Path("gear_sonic/data/motions/x2_planner_primitives.pkl"),
        help="Curator-built primitives PKL.",
    )
    p.add_argument(
        "--bins",
        type=Path,
        default=Path("gear_sonic/data/motions/x2_planner_bins.yaml"),
        help="Bins YAML (used for family lookup).",
    )
    p.add_argument(
        "--pub-host",
        default="127.0.0.1",
        help="ZMQ publish bind host (default: 127.0.0.1).",
    )
    p.add_argument(
        "--pub-port", type=int, default=5556,
        help="ZMQ publish port (default: 5556 — matches mock_vla_publish_stand_token).",
    )
    p.add_argument(
        "--body-pose-port", type=int, default=None,
        help=(
            "Phase 0: publish the 31-DOF reference under topic "
            "'body_pose' on this port (instead of topic 'pose' on "
            "--pub-port). Use when the recorder is the merger that "
            "combines body_pose + arm_targets and forwards "
            "final_pose to the deploy. Mutually exclusive with the "
            "default direct-to-deploy mode; when set, overrides "
            "--pub-port."
        ),
    )
    p.add_argument(
        "--pid-file",
        type=Path,
        default=Path("/tmp/x2_heuristic_planner.pid"),
        help="PID file path; cleaned up on exit.",
    )
    p.add_argument("--demo", type=Path, help="Scripted demo YAML.")
    p.add_argument("--keyboard", action="store_true", help="Enable interactive keyboard input.")
    p.add_argument(
        "--zmq-cmd-host", default=None,
        help="If set, also subscribe for high-level commands on tcp://host:port",
    )
    p.add_argument("--zmq-cmd-port", type=int, default=None)
    p.add_argument(
        "--zmq-cmd-topic", default="planner_cmd",
        help="Topic prefix expected on the command SUB socket (default: planner_cmd).",
    )
    p.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Stop after N seconds. 0 means run forever (until SIGINT).",
    )
    p.add_argument(
        "--hand-dof", type=int, default=10,
        help="Hand-joint length to publish (10 for X2 native, 7 for G1 compat).",
    )
    p.add_argument(
        "--warmup-quiet-stand-s", type=float, default=0.5,
        help=(
            "Publish the planner's anchor frame for this many seconds at "
            "startup before the state machine takes over. With "
            "``--sim-profile parity`` and the bake_planner_rsi_anchor.py "
            "RSI PKL, the bridge already spawns at frame 0 of idle_stand, "
            "so this is just defensive pre-roll: it makes sure ZMQ "
            "publishers have flushed their first message before deploy "
            "enters CONTROL. 0.5 s is the rolled-down default after the "
            "v5 future-window fix (was 2.0 s during the air-launch debug "
            "era)."
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run(
        primitives_pkl=args.primitives,
        bins_yaml=args.bins,
        pub_host=args.pub_host,
        pub_port=args.pub_port,
        pid_file=args.pid_file,
        demo_yaml=args.demo,
        enable_keyboard=args.keyboard,
        zmq_cmd_host=args.zmq_cmd_host,
        zmq_cmd_port=args.zmq_cmd_port,
        zmq_cmd_topic=args.zmq_cmd_topic,
        duration_s=args.duration_s,
        hand_dof=args.hand_dof,
        verbose=args.verbose,
        warmup_quiet_stand_s=args.warmup_quiet_stand_s,
        body_pose_port=args.body_pose_port,
    )


# Re-export wire/payload helpers so the smoke runner & tests don't need
# to know about the state_machine module path.
NUM_BODY_DOFS = 31
SONIC_MOTION_TOKEN_DIM = 64


if __name__ == "__main__":
    raise SystemExit(main())
