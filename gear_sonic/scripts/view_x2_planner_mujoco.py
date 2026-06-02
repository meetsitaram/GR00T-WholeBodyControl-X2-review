#!/usr/bin/env python3
"""Live MuJoCo visualization for the X2 heuristic planner.

Two modes:

  in-process (default):
      Runs the planner state machine inside this process, opens the X2 MuJoCo
      viewer, and renders each StreamFrame straight into ``mj_data.qpos``.
      No ZMQ, no policy, no physics — purely kinematic, purely the same data
      the publisher would put on the wire. Best for demos & primitive QA.

  --from-zmq HOST:PORT:
      Subscribes to a running ``x2_heuristic_planner`` daemon on
      ``tcp://HOST:PORT`` (default port 5556) and renders received frames as
      they arrive. Use this to verify the wire format end-to-end while the
      planner publishes for the C++ deploy in another terminal.

Examples (from the repo root):

    # Standalone demo (the simplest "see the robot move" path):
    .venv/bin/python -m gear_sonic.scripts.view_x2_planner_mujoco \\
        --demo gear_sonic/data/scripted_demos/forward_back_turn.yaml

    # Interactive teleop in the viewer window:
    .venv/bin/python -m gear_sonic.scripts.view_x2_planner_mujoco --keyboard

    # Watch a separately-running daemon's pose stream:
    .venv/bin/python -m gear_sonic.scripts.view_x2_planner_mujoco \\
        --from-zmq 127.0.0.1:5556

In ZMQ-subscribe mode the viewer auto-detects ``root_xy_world`` (2,) and
``root_z_world`` (1,) on the wire (planner + recorder emit them on every
tick post-2026-06) and renders full world-frame pelvis translation. If
the publisher is older and omits those keys, the viewer falls back to
the legacy pelvis-pinned-at-origin behaviour and only shows body
articulation + heading. In-process mode always shows full world
translation.

Keyboard (in-process mode, viewer must have focus):
    w / b           walk / back-step
    f / F / r       fwd_step half / one / quarter ft
    a / d           side_left/right_half_ft
    A / D           side_left/right_quarter_ft
    1 / 2 / q / Q   turn_left  15 / 30 / 45 / 90 deg
    3 / 4 / e / E   turn_right 15 / 30 / 45 / 90 deg
    l / k           lean_fwd medium / small (use the headless CLI for "large")
    j / ;           lean_left_medium / lean_right_medium  (lateral, v6)
    , / .           torso_left / torso_right 30 deg
    T               STATIC_HOLD at neutral (continuous waist debug)
    i / o           hold pitch -2 / +2 deg
    u / n           hold yaw   -5 / +5 deg
    y / h           hold roll  -2 / +2 deg
    SPACE           idle (and pause / resume)
    R               reset to origin
    X / ESC         quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

GEAR_SONIC_ROOT = Path(__file__).resolve().parents[2]
if str(GEAR_SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GEAR_SONIC_ROOT))

from gear_sonic.utils.planner.constants import (  # noqa: E402
    DEFAULT_PELVIS_Z_M,
    DEFAULT_STAND_POSE_NP,
    NUM_BODY_DOFS,
)
from gear_sonic.utils.planner.registry import load_bin_specs  # noqa: E402
from gear_sonic.utils.planner.state_machine import (  # noqa: E402
    HeuristicPlanner,
    LocomotionCommand,
    OUTPUT_FPS,
    commands_from_yaml,
    load_primitives_pkl,
)


MJCF_PATH = str(
    GEAR_SONIC_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
)


# Keyboard map mirrors gear_sonic/scripts/x2_heuristic_planner.py::KEYBOARD_MAP
# but is keyed by GLFW key codes so the MuJoCo viewer's key_callback can use it.
def _build_glfw_keymap() -> dict[int, tuple[str, str]]:
    import glfw  # noqa: PLC0415 — viewer-only optional dep
    base: dict[int, tuple[str, str]] = {
        glfw.KEY_W: ("walk", "forward"),
        glfw.KEY_B: ("back_step", "half_ft"),
        glfw.KEY_F: ("fwd_step", "half_ft"),
        glfw.KEY_R: ("fwd_step", "quarter_ft"),  # 'r' lower-case = quarter
        glfw.KEY_A: ("side_left", "half_ft"),
        glfw.KEY_D: ("side_right", "half_ft"),
        glfw.KEY_1: ("turn_left", "deg_15"),
        glfw.KEY_2: ("turn_left", "deg_30"),
        glfw.KEY_Q: ("turn_left", "deg_45"),
        glfw.KEY_3: ("turn_right", "deg_15"),
        glfw.KEY_4: ("turn_right", "deg_30"),
        glfw.KEY_E: ("turn_right", "deg_45"),
        glfw.KEY_L: ("lean_fwd", "medium"),
        glfw.KEY_K: ("lean_fwd", "small"),
        # Lateral lean (v6) -- GLFW key_callback can't see Shift here, so
        # only the "medium" magnitude is bound. Use the headless planner
        # CLI (gear_sonic/scripts/x2_heuristic_planner.py) for "large".
        glfw.KEY_J: ("lean_left", "medium"),
        glfw.KEY_SEMICOLON: ("lean_right", "medium"),
        glfw.KEY_COMMA: ("torso_left", "deg_30"),
        glfw.KEY_PERIOD: ("torso_right", "deg_30"),
        glfw.KEY_SPACE: ("idle", "default"),
    }
    return base


# Per-keypress nudge step (degrees) for the continuous-hold debug keys.
# Match the planner CLI defaults so the two surfaces feel identical.
_HOLD_NUDGE_PITCH_DEG: float = 2.0
_HOLD_NUDGE_ROLL_DEG: float = 2.0
_HOLD_NUDGE_YAW_DEG: float = 5.0


def _build_glfw_hold_nudge_map() -> dict[int, tuple[str, float]]:
    """GLFW keycode -> (axis, signed_step_deg) for STATIC_HOLD nudges.

    Mirrors the planner CLI's ``_HOLD_NUDGE_KEYS`` but substitutes
    ``KEY_N`` for ``KEY_P`` because the viewer reserves ``P`` for the
    pause toggle.
    """
    import glfw  # noqa: PLC0415 — viewer-only optional dep
    return {
        glfw.KEY_I: ("pitch", -_HOLD_NUDGE_PITCH_DEG),
        glfw.KEY_O: ("pitch", +_HOLD_NUDGE_PITCH_DEG),
        glfw.KEY_U: ("yaw", -_HOLD_NUDGE_YAW_DEG),
        glfw.KEY_N: ("yaw", +_HOLD_NUDGE_YAW_DEG),
        glfw.KEY_Y: ("roll", -_HOLD_NUDGE_ROLL_DEG),
        glfw.KEY_H: ("roll", +_HOLD_NUDGE_ROLL_DEG),
    }


def _qpos_from_pose(
    qpos: np.ndarray,
    joint_pos_mj: np.ndarray,
    root_quat_xyzw: np.ndarray,
    root_xyz: np.ndarray | None,
) -> None:
    """Write a planner StreamFrame's pose into a MuJoCo qpos buffer in place.

    qpos layout: [root_xyz (3), root_wxyz (4), joint_pos (NUM_BODY_DOFS)].
    """
    if root_xyz is None:
        qpos[0] = 0.0
        qpos[1] = 0.0
        qpos[2] = DEFAULT_PELVIS_Z_M
    else:
        qpos[0] = float(root_xyz[0])
        qpos[1] = float(root_xyz[1])
        qpos[2] = float(root_xyz[2])
    # MuJoCo wxyz <- planner xyzw
    qpos[3] = float(root_quat_xyzw[3])
    qpos[4] = float(root_quat_xyzw[0])
    qpos[5] = float(root_quat_xyzw[1])
    qpos[6] = float(root_quat_xyzw[2])
    qpos[7 : 7 + NUM_BODY_DOFS] = joint_pos_mj


# ---------------------------------------------------------------------------
# In-process driver
# ---------------------------------------------------------------------------


def run_in_process(
    primitives_pkl: Path,
    bins_yaml: Path,
    demo_yaml: Path | None,
    enable_keyboard: bool,
    duration_s: float,
    initial_commands: list[LocomotionCommand],
) -> int:
    import mujoco
    import mujoco.viewer

    bin_specs = load_bin_specs(bins_yaml)
    bin_family = {n: s.family for n, s in bin_specs.items()}
    primitives = load_primitives_pkl(primitives_pkl, bin_family)
    print(
        f"[viewer] loaded {len(primitives)} primitives "
        f"({sum(1 for p in primitives.values() if p.partial)} PARTIAL)"
    )
    planner = HeuristicPlanner(primitives=primitives)

    if demo_yaml is not None:
        cmds = commands_from_yaml(demo_yaml)
        for c in cmds:
            planner.enqueue(c)
        print(f"[viewer] queued {len(cmds)} commands from {demo_yaml}")
    for c in initial_commands:
        planner.enqueue(c)

    print(f"[viewer] loading MuJoCo model: {MJCF_PATH}")
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    pelvis_id = mj_model.body("pelvis").id

    # Seed at default stand so the very first frame isn't garbage.
    _qpos_from_pose(
        mj_data.qpos,
        DEFAULT_STAND_POSE_NP,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, DEFAULT_PELVIS_Z_M], dtype=np.float64),
    )
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    paused = [False]
    keymap = _build_glfw_keymap() if enable_keyboard else {}
    hold_nudge_map = _build_glfw_hold_nudge_map() if enable_keyboard else {}
    # Local STATIC_HOLD target tracked by the viewer's nudge keys. Updated
    # by KEY_T (reset to 0) and the nudge keys; re-emitted as a hold_torso
    # command after every change so the planner's _HoldTracker slews to it.
    hold_target: dict[str, float] = {"pitch": 0.0, "roll": 0.0, "yaw": 0.0}
    last_cmd_label = ["—"]

    def _emit_hold() -> None:
        planner.enqueue(
            LocomotionCommand(
                intent="hold_torso",
                magnitude="continuous",
                source="kbd",
                waist_pitch_deg=hold_target["pitch"],
                waist_roll_deg=hold_target["roll"],
                waist_yaw_deg=hold_target["yaw"],
            )
        )
        last_cmd_label[0] = (
            f"hold_torso pitch={hold_target['pitch']:+.1f} "
            f"roll={hold_target['roll']:+.1f} "
            f"yaw={hold_target['yaw']:+.1f}"
        )
        print(f"[viewer] -> {last_cmd_label[0]}")

    def key_callback(keycode: int) -> None:
        import glfw

        if keycode in (glfw.KEY_X, glfw.KEY_ESCAPE):
            print("[viewer] quit requested")
            paused[0] = True  # main loop will see is_running()==False soon
        elif keycode == glfw.KEY_SPACE:
            # Space doubles as "queue idle" and "toggle pause" — pause only on
            # double-tap-style: enqueue idle once, but if we're already idle,
            # toggle pause. Keep it simple: enqueue idle, never pause.
            planner.enqueue(LocomotionCommand("idle", "default", source="kbd"))
            last_cmd_label[0] = "idle default"
            # Reset hold target so the next 'T' starts fresh.
            hold_target.update(pitch=0.0, roll=0.0, yaw=0.0)
        elif keycode == glfw.KEY_P:
            paused[0] = not paused[0]
            print("[viewer] paused" if paused[0] else "[viewer] resumed")
        elif keycode == glfw.KEY_T:
            hold_target.update(pitch=0.0, roll=0.0, yaw=0.0)
            _emit_hold()
        elif keycode in hold_nudge_map:
            axis, step = hold_nudge_map[keycode]
            hold_target[axis] += step
            _emit_hold()
        elif keycode in keymap:
            intent, mag = keymap[keycode]
            planner.enqueue(LocomotionCommand(intent, mag, source="kbd"))
            last_cmd_label[0] = f"{intent} {mag}"
            print(f"[viewer] -> {intent} {mag}")

    print(
        "\n=== X2 Heuristic Planner — Live View ===\n"
        f"  rate: {OUTPUT_FPS:.0f} Hz, duration: "
        f"{'forever' if duration_s <= 0 else f'{duration_s:.1f}s'}\n"
        "  controls: P=pause, X/ESC=quit"
        + (", + see file header for the rest" if enable_keyboard else "")
        + "\n",
        flush=True,
    )

    period_s = 1.0 / OUTPUT_FPS
    next_tick = time.monotonic()
    end_at = float("inf") if duration_s <= 0 else time.monotonic() + duration_s

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0.0, 0.0, DEFAULT_PELVIS_Z_M]
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id

        while viewer.is_running() and time.monotonic() < end_at:
            if paused[0]:
                viewer.sync()
                time.sleep(0.05)
                next_tick = time.monotonic()
                continue

            frame = planner.step()
            _qpos_from_pose(
                mj_data.qpos,
                frame.joint_pos_mj,
                frame.root_quat_xyzw,
                np.array(
                    [frame.root_xy_world[0], frame.root_xy_world[1], DEFAULT_PELVIS_Z_M],
                    dtype=np.float64,
                ),
            )
            mj_data.qvel[:] = 0.0
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            next_tick += period_s
            slack = next_tick - time.monotonic()
            if slack > 0:
                time.sleep(slack)
            else:
                if -slack > 5 * period_s:
                    next_tick = time.monotonic()
        print(f"[viewer] loop done at tick {planner.tick}")
    return 0


# ---------------------------------------------------------------------------
# ZMQ-subscribe driver
# ---------------------------------------------------------------------------


def run_from_zmq(host: str, port: int, duration_s: float, topic: str = "pose") -> int:
    import mujoco
    import mujoco.viewer
    import zmq

    from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message

    print(f"[viewer] subscribing to tcp://{host}:{port} (topic={topic!r})")
    ctx = zmq.Context()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVTIMEO, 200)
    sock.connect(f"tcp://{host}:{port}")
    time.sleep(0.2)

    print(f"[viewer] loading MuJoCo model: {MJCF_PATH}")
    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    pelvis_id = mj_model.body("pelvis").id

    _qpos_from_pose(
        mj_data.qpos,
        DEFAULT_STAND_POSE_NP,
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 0.0, DEFAULT_PELVIS_Z_M], dtype=np.float64),
    )
    mj_data.qvel[:] = 0.0
    mujoco.mj_forward(mj_model, mj_data)

    end_at = float("inf") if duration_s <= 0 else time.monotonic() + duration_s
    n_received = 0
    print("[viewer] waiting for first pose message...")
    try:
        with mujoco.viewer.launch_passive(
            mj_model,
            mj_data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            viewer.cam.azimuth = 120
            viewer.cam.elevation = -20
            viewer.cam.distance = 3.0
            viewer.cam.lookat[:] = [0.0, 0.0, DEFAULT_PELVIS_Z_M]
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = pelvis_id

            while viewer.is_running() and time.monotonic() < end_at:
                try:
                    raw = sock.recv()
                except zmq.error.Again:
                    viewer.sync()
                    continue
                try:
                    decoded = unpack_message(raw, expected_topic=topic)
                except ValueError as exc:
                    print(f"[viewer] decode error: {exc}")
                    continue
                fields = decoded.fields
                if "joint_pos_mj" not in fields or "root_quat_xyzw" not in fields:
                    continue
                # Auto-detect world-frame root fields. Post-2026-06 the
                # planner + recorder include ``root_xy_world`` (2,) and
                # ``root_z_world`` (1,) on every body_pose / pose tick,
                # which lets the kinematic viewer track actual pelvis
                # translation instead of pinning at the origin. Older
                # publishers omit the keys -> we fall back to None
                # (legacy pelvis-pinned behaviour).
                if "root_xy_world" in fields and "root_z_world" in fields:
                    rxy = np.asarray(fields["root_xy_world"]).reshape(-1)
                    rz = np.asarray(fields["root_z_world"]).reshape(-1)
                    if rxy.shape == (2,) and rz.shape == (1,):
                        root_xyz = np.array(
                            [float(rxy[0]), float(rxy[1]), float(rz[0])],
                            dtype=np.float64,
                        )
                    else:
                        root_xyz = None
                else:
                    root_xyz = None
                if n_received == 0:
                    print(
                        f"[viewer] first frame received "
                        f"(version={decoded.version}, "
                        f"world-root={'yes' if root_xyz is not None else 'no (pelvis pinned)'}, "
                        f"fields={list(fields)})"
                    )
                _qpos_from_pose(
                    mj_data.qpos,
                    fields["joint_pos_mj"],
                    fields["root_quat_xyzw"],
                    root_xyz,
                )
                mj_data.qvel[:] = 0.0
                mujoco.mj_forward(mj_model, mj_data)
                viewer.sync()
                n_received += 1
            print(f"[viewer] received {n_received} frames")
    finally:
        sock.close(linger=0)
        ctx.term()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="view_x2_planner_mujoco",
        description=(
            "Live MuJoCo viewer for the X2 heuristic planner. Default mode "
            "runs the state machine in-process and renders frames straight "
            "from StreamFrame. With --from-zmq, subscribes to a running daemon."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--primitives", type=Path,
        default=Path("gear_sonic/data/motions/x2_planner_primitives.pkl"),
    )
    p.add_argument(
        "--bins", type=Path,
        default=Path("gear_sonic/data/motions/x2_planner_bins.yaml"),
    )
    p.add_argument(
        "--demo", type=Path,
        help="Scripted-demo YAML (in-process mode only). "
             "If omitted and --keyboard not set, the planner sits on idle.",
    )
    p.add_argument(
        "--keyboard", action="store_true",
        help="Enable keyboard input via the MuJoCo viewer (in-process mode only).",
    )
    p.add_argument(
        "--initial-cmd", action="append", default=[],
        metavar="INTENT:MAGNITUDE",
        help="Queue an initial command (repeatable). E.g. --initial-cmd walk:forward",
    )
    p.add_argument(
        "--duration-s", type=float, default=0.0,
        help="Auto-quit after N seconds. 0 = run until window closed.",
    )
    p.add_argument(
        "--from-zmq", default=None,
        metavar="HOST:PORT",
        help="Subscribe to a running planner instead of running in-process.",
    )
    p.add_argument(
        "--topic", default="pose",
        choices=("pose", "body_pose"),
        help=(
            "ZMQ topic to subscribe to in --from-zmq mode. Use 'pose' (default) "
            "for the heuristic planner's direct-to-deploy mode and the legacy "
            "kplanner direct mode; use 'body_pose' for the new Phase 0 "
            "recorder-merge mode and for the x2_kplanner.py daemon's "
            "default publish topic."
        ),
    )
    return p.parse_args(argv)


def _parse_initial_cmds(specs: list[str]) -> list[LocomotionCommand]:
    out: list[LocomotionCommand] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--initial-cmd must be intent:magnitude, got {spec!r}")
        intent, magnitude = spec.split(":", 1)
        out.append(
            LocomotionCommand(intent=intent, magnitude=magnitude, source="cli-init")
        )
    return out


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.from_zmq is not None:
        if ":" in args.from_zmq:
            host, port_str = args.from_zmq.rsplit(":", 1)
            port = int(port_str)
        else:
            host = args.from_zmq
            port = 5556
        return run_from_zmq(host, port, args.duration_s, topic=args.topic)

    if not args.primitives.exists():
        print(
            f"[viewer] ERROR: primitives PKL not found: {args.primitives}\n"
            f"Run: .venv/bin/python -m gear_sonic.scripts.curate_x2_primitives",
            file=sys.stderr,
        )
        return 1
    if not args.bins.exists():
        print(f"[viewer] ERROR: bins YAML not found: {args.bins}", file=sys.stderr)
        return 1

    return run_in_process(
        primitives_pkl=args.primitives,
        bins_yaml=args.bins,
        demo_yaml=args.demo,
        enable_keyboard=args.keyboard,
        duration_s=args.duration_s,
        initial_commands=_parse_initial_cmds(args.initial_cmd),
    )


if __name__ == "__main__":
    raise SystemExit(main())
