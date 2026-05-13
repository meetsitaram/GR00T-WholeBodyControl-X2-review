"""Quest 3 VR manager for GEAR-SONIC whole-body control.

Replaces ``pico_manager_thread_server.py`` for users with a Meta Quest 3
headset.  Uses WebXR + WebSocket instead of XRoboToolkit to receive
head + controller tracking data and feeds them into the GEAR-SONIC
pipeline via ZMQ, reusing the same planner and 3-point VR control modes.

Supported modes
---------------
* **PLANNER**          – Locomotion only (joystick control), no upper-body VR.
* **PLANNER_VR_3PT**   – Locomotion + upper-body VR 3-point tracking.
* **OFF**              – Policy stopped.

Typical usage (sim2sim)::

    # Terminal 1 – MuJoCo sim
    source .venv_sim/bin/activate
    python gear_sonic/scripts/run_sim_loop.py

    # Terminal 2 – C++ deployment
    cd gear_sonic_deploy && bash deploy.sh sim --input-type zmq_manager

    # Terminal 3 – Quest 3 manager
    source .venv_teleop/bin/activate
    python gear_sonic/scripts/quest3_manager_thread_server.py \\
        --ws-port 8765 --http-port 8443

Then open ``https://<workstation-ip>:8443`` in the Quest 3 browser,
connect the WebSocket, and start the VR session.
"""

import argparse
import time

import numpy as np
import zmq

from gear_sonic.utils.teleop.common import (
    FeedbackReader,
    LocomotionMode,
    StreamMode,
    ThreePointPose,
    compute_hand_joints_from_inputs,
    init_hand_ik_solvers,
)
from gear_sonic.utils.teleop.vr.button_state_machine import ButtonStateMachine
from gear_sonic.utils.teleop.vr.joystick_mapping import (
    JOYSTICK_DEADZONE,
    YawAccumulator,
    apply_radial_deadzone,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)
from gear_sonic.utils.teleop.vr.quest3_reader import Quest3Reader


# ---------------------------------------------------------------------------
# Quest3 Planner Streamer
# ---------------------------------------------------------------------------


class Quest3PlannerStreamer:
    """Reads Quest 3 controller axes / buttons and sends planner ZMQ messages.

    Mirrors ``PlannerStreamer`` from pico_manager_thread_server but reads
    input from :class:`Quest3Reader` instead of XRoboToolkit.
    """

    def __init__(
        self,
        socket,
        reader: Quest3Reader,
        three_point: ThreePointPose,
        poll_hz: int = 20,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5557,
    ):
        self.socket = socket
        self.reader = reader
        self.three_point = three_point
        self.feedback_reader = FeedbackReader(
            zmq_feedback_host=zmq_feedback_host,
            zmq_feedback_port=zmq_feedback_port,
        )

        self.dt = 1.0 / max(1, poll_hz)
        self.mode = LocomotionMode.IDLE
        self.button_sm = ButtonStateMachine(log_prefix="Input")
        self.yaw_accumulator = YawAccumulator()
        self.last_send = time.time()
        self._last_axes_log = 0.0

        self.left_hand_ik_solver, self.right_hand_ik_solver = init_hand_ik_solvers()

    def reset_yaw(self):
        self.yaw_accumulator.reset()

    def save_upper_body_position_target(self):
        self.feedback_reader.poll_feedback()

    def recalibrate_for_vr3pt(self):
        self.feedback_reader.poll_feedback()
        if self.feedback_reader.full_body_q_measured is not None:
            self.three_point.reset_with_measured_q(self.feedback_reader.full_body_q_measured)
            print("[Quest3Planner] VR 3PT recalibration scheduled with measured robot pose")
        else:
            print("[Quest3Planner] WARNING: No feedback data, using zero body_q as fallback")
            self.three_point.reset_with_measured_q(np.zeros(29, dtype=np.float64))

    def run_once(self, stream_mode: StreamMode):
        try:
            # --- button state (edge-triggered mode switching) -----------------
            a, b, x, y = self.reader.get_buttons()
            ev = self.button_sm.tick(a, b, x, y)

            if ev.ab_pressed:
                self.mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self.mode + 1))
                print(f"[Quest3Planner] Mode -> {self.mode.value}: {self.mode.name}")
            if ev.xy_pressed:
                self.mode = LocomotionMode(max(LocomotionMode.IDLE, self.mode - 1))
                print(f"[Quest3Planner] Mode -> {self.mode.value}: {self.mode.name}")

            # --- joystick movement / facing -----------------------------------
            lx, ly, rx, ry = self.reader.get_controller_axes()

            # Log joystick + triggers periodically (every 0.5s, only when active)
            now = time.time()
            if now - self._last_axes_log > 0.5:
                lt, rt, lg, rg = self.reader.get_controller_inputs()
                has_stick = abs(lx) > JOYSTICK_DEADZONE or abs(ly) > JOYSTICK_DEADZONE or abs(rx) > JOYSTICK_DEADZONE or abs(ry) > JOYSTICK_DEADZONE
                has_trigger = lt > 0.1 or rt > 0.1 or lg > 0.1 or rg > 0.1
                if has_stick or has_trigger:
                    parts = []
                    if has_stick:
                        parts.append(f"sticks L({lx:+.2f},{ly:+.2f}) R({rx:+.2f},{ry:+.2f})")
                    if has_trigger:
                        parts.append(f"LT={lt:.2f} RT={rt:.2f} LG={lg:.2f} RG={rg:.2f}")
                    print(f"[Input] {' | '.join(parts)}")
                    self._last_axes_log = now
            facing = self.yaw_accumulator.update(rx, self.dt)

            raw_mag = float(np.clip(np.hypot(lx, ly), 0.0, 1.0))
            mag = apply_radial_deadzone(raw_mag)
            if mag == 0.0:
                speed = -1.0
                mode_to_send = LocomotionMode.IDLE
            else:
                mode_to_send = self.mode

                if self.mode == LocomotionMode.SLOW_WALK:
                    speed = 0.1 + 0.5 * mag
                elif self.mode == LocomotionMode.WALK:
                    speed = -1.0
                elif self.mode == LocomotionMode.RUN:
                    speed = 1.5 + 3 * mag
                else:
                    speed = mag

            denom = raw_mag if raw_mag > 0.0 else 1.0
            scale = mag / denom
            movement_local = np.array([-lx, ly]) * scale
            perp_x, perp_y = -facing[1], facing[0]
            rotation_facing = np.array([[perp_x, perp_y], [facing[0], facing[1]]])
            movement_global = rotation_facing @ movement_local
            movement = [movement_global[0], movement_global[1], 0.0]

            # --- optional upper-body / VR 3PT data ----------------------------
            upper_body_position = None
            left_hand_position = None
            right_hand_position = None
            vr_3pt_position = None
            vr_3pt_orientation = None
            vr_3pt_compliance = None

            if stream_mode == StreamMode.PLANNER_FROZEN_UPPER_BODY:
                upper_body_position = self.feedback_reader.upper_body_position_target
                left_hand_position = self.feedback_reader.left_hand_position_target
                right_hand_position = self.feedback_reader.right_hand_position_target

            if stream_mode == StreamMode.PLANNER_VR_3PT:
                raw_3pt = self.reader.get_3pt_pose()
                if raw_3pt is not None:
                    vr_3pt_pose = self.three_point.process(raw_3pt)
                    vr_3pt_position = vr_3pt_pose[:, :3].flatten().tolist()
                    vr_3pt_orientation = vr_3pt_pose[:, 3:].flatten().tolist()

                lt, rt, lg, rg = self.reader.get_controller_inputs()
                lh, rh = compute_hand_joints_from_inputs(
                    self.left_hand_ik_solver,
                    self.right_hand_ik_solver,
                    lt, lg, rt, rg,
                )
                left_hand_position = lh.reshape(-1).astype(np.float32).tolist()
                right_hand_position = rh.reshape(-1).astype(np.float32).tolist()

            msg = build_planner_message(
                mode_to_send.value,
                movement,
                facing,
                speed=speed,
                height=-1.0,
                upper_body_position=upper_body_position,
                left_hand_position=left_hand_position,
                right_hand_position=right_hand_position,
                vr_3pt_position=vr_3pt_position,
                vr_3pt_orientation=vr_3pt_orientation,
                vr_3pt_compliance=vr_3pt_compliance,
            )
            self.socket.send(msg)

        except Exception as e:
            import traceback

            print(f"[Quest3Planner] error: {e}")
            traceback.print_exc()
            raise

        now = time.time()
        sleep_t = self.dt - (now - self.last_send)
        if sleep_t > 0:
            time.sleep(sleep_t)
        self.last_send = time.time()


# ---------------------------------------------------------------------------
# Manager state machine
# ---------------------------------------------------------------------------


def run_quest3_manager(
    port: int = 5556,
    ws_port: int = 8765,
    http_port: int = 8443,
    use_ssl: bool = True,
    zmq_feedback_host: str = "localhost",
    zmq_feedback_port: int = 5557,
    enable_vis_vr3pt: bool = False,
    with_g1_robot: bool = True,
    enable_waist_tracking: bool = False,
):
    """Main manager loop for Quest 3 VR teleop.

    State machine::

        OFF --(A+B+X+Y)--> PLANNER --(left_axis_click)--> PLANNER_VR_3PT
                                                                |
         ^---(A+B+X+Y)---<----(A+B+X+Y)---<---(left_axis_click)--+

    The ``POSE`` and ``PLANNER_FROZEN_UPPER_BODY`` modes from the PICO
    manager are omitted because Quest 3 does not provide SMPL body
    tracking; only 3-point (head + hands) tracking is available.
    """

    # -- Quest 3 reader -------------------------------------------------------
    reader = Quest3Reader(
        ws_port=ws_port,
        http_port=http_port,
        use_ssl=use_ssl,
    )
    reader.start()

    print("[Manager] Waiting for Quest 3 client to connect ...")
    print(f"[Manager] Open the WebXR page in the Quest 3 browser.")
    proto = "https" if use_ssl else "http"
    from gear_sonic.utils.teleop.vr.quest3_reader import _get_lan_ip
    lan_ip = _get_lan_ip()
    print(f"[Manager]   URL: {proto}://{lan_ip}:{http_port}")
    while not reader.is_connected:
        time.sleep(0.5)
    print("[Manager] Quest 3 connected!")

    # -- ZMQ publisher --------------------------------------------------------
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{port}")
    time.sleep(0.1)
    print(f"[Manager] ZMQ socket bound to port {port}")

    # -- Shared ThreePointPose and planner ------------------------------------
    three_point = ThreePointPose(
        enable_vis_vr3pt=enable_vis_vr3pt,
        with_g1_robot=with_g1_robot,
        enable_waist_tracking=enable_waist_tracking,
        enable_smpl_vis=False,
        log_prefix="Quest3",
    )

    planner = Quest3PlannerStreamer(
        socket=socket,
        reader=reader,
        three_point=three_point,
        poll_hz=20,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
    )

    print("[Manager] Available locomotion modes:")
    for mode in LocomotionMode:
        print(f"  {mode.value}: {mode.name}")

    # -- State machine --------------------------------------------------------
    #
    #  Controls (Quest 3 controller buttons):
    #    A+B+X+Y : start / emergency stop
    #    A+X     : toggle PLANNER <-> PLANNER_VR_3PT
    #
    current_mode = StreamMode.OFF
    mgr_button_sm = ButtonStateMachine(log_prefix="Input")

    print("[Manager] Controls: A+B+X+Y = start/stop, A+X = toggle VR 3PT")

    try:
        while True:
            if not reader.is_connected:
                time.sleep(0.1)
                continue

            a, b, x, y = reader.get_buttons()
            ev = mgr_button_sm.tick(a, b, x, y)

            new_mode = current_mode

            if current_mode == StreamMode.OFF:
                if ev.abxy_pressed:
                    new_mode = StreamMode.PLANNER
                    raw_3pt = reader.get_3pt_pose()
                    if raw_3pt is not None:
                        three_point.calibrate_now(raw_3pt)
                    else:
                        print("[Manager] WARNING: No tracking data for calibration")

            elif current_mode == StreamMode.PLANNER:
                if ev.abxy_pressed:
                    new_mode = StreamMode.OFF
                elif ev.ax_pressed:
                    new_mode = StreamMode.PLANNER_VR_3PT

            elif current_mode == StreamMode.PLANNER_VR_3PT:
                if ev.abxy_pressed:
                    new_mode = StreamMode.OFF
                elif ev.ax_pressed:
                    new_mode = StreamMode.PLANNER

            # -- handle transitions -------------------------------------------
            if new_mode != current_mode:
                if new_mode == StreamMode.PLANNER and current_mode != StreamMode.PLANNER_VR_3PT:
                    planner.reset_yaw()
                elif new_mode == StreamMode.PLANNER_VR_3PT:
                    planner.recalibrate_for_vr3pt()

            # -- run one iteration of the active mode -------------------------
            if new_mode in (StreamMode.PLANNER, StreamMode.PLANNER_VR_3PT):
                planner.run_once(new_mode)

            # -- send ZMQ command messages on transition ----------------------
            if new_mode != current_mode:
                if new_mode == StreamMode.OFF:
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    print(f"[Manager] {current_mode.name} -> OFF (stopped)")
                    break
                else:
                    socket.send(build_command_message(start=True, stop=False, planner=True))

                print(f"[Manager] {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

    except KeyboardInterrupt:
        print("\n[Manager] Interrupted")
    finally:
        reader.stop()
        three_point.close()
        socket.close()
        context.term()
        print("[Manager] Shutdown complete")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quest 3 VR manager for GEAR-SONIC whole-body control"
    )
    parser.add_argument(
        "--port", type=int, default=5556, help="ZMQ publisher port (default: 5556)"
    )
    parser.add_argument(
        "--ws-port", type=int, default=8765, help="WebSocket server port (default: 8765)"
    )
    parser.add_argument(
        "--http-port", type=int, default=8443, help="HTTPS server port for WebXR app (default: 8443)"
    )
    parser.add_argument(
        "--no-ssl", action="store_true", help="Disable TLS (use plain WS/HTTP)"
    )
    parser.add_argument(
        "--zmq-feedback-host", type=str, default="localhost",
        help="ZMQ feedback host (default: localhost)"
    )
    parser.add_argument(
        "--zmq-feedback-port", type=int, default=5557,
        help="ZMQ feedback port (default: 5557)"
    )
    parser.add_argument(
        "--vis-vr3pt", action="store_true",
        help="Enable VR 3-point pose visualization (requires display + pyvista)"
    )
    parser.add_argument(
        "--no-g1", action="store_true",
        help="Disable G1 robot in VR 3pt pose visualization"
    )
    parser.add_argument(
        "--waist-tracking", action="store_true",
        help="Enable waist tracking in VR 3pt visualization"
    )
    args = parser.parse_args()

    run_quest3_manager(
        port=args.port,
        ws_port=args.ws_port,
        http_port=args.http_port,
        use_ssl=not args.no_ssl,
        zmq_feedback_host=args.zmq_feedback_host,
        zmq_feedback_port=args.zmq_feedback_port,
        enable_vis_vr3pt=args.vis_vr3pt,
        with_g1_robot=not args.no_g1,
        enable_waist_tracking=args.waist_tracking,
    )
