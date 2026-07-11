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
import json
import logging
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
from gear_sonic.utils.teleop.vr.stick_smoother import StickFilter, StickFilterConfig
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    build_command_message,
    build_planner_message,
)
from gear_sonic.utils.teleop.vr.quest3_reader import Quest3Reader

log = logging.getLogger("quest3_manager")


# Mild default stick smoothing ported from the X2 stack: a light LPF plus a
# slew cap removes WebXR jitter without feeling sluggish. Channels map
# fwd<-ly, side<-lx, yaw<-rx. Tune live; pass a no-op StickFilterConfig() (or
# --no-stick-filter) to disable.
_DEFAULT_STICK_FILTER = StickFilterConfig(
    tau_lpf_fwd_s=0.06, slew_max_fwd_per_s=8.0, return_to_zero_tau_fwd_s=0.10,
    tau_lpf_side_s=0.06, slew_max_side_per_s=8.0, return_to_zero_tau_side_s=0.10,
    tau_lpf_yaw_s=0.05, slew_max_yaw_per_s=10.0, return_to_zero_tau_yaw_s=0.08,
)


# Angular deadband on the left-stick MOVEMENT direction. Humans can't hold a
# precise heading on an analog stick, and near a cardinal the *normalized*
# direction wobbles by 5-10 deg -> every wobble trips the G1 deploy's exact-`!=`
# replan trigger -> replan storm -> overshoot strides. Fix: keep analog SPEED
# but snap the DIRECTION to the nearest 8-way axis (fwd/back/strafe/diagonal)
# when within +/- the band. Snapping is in the FACING frame, so a curved walk
# (left stick forward + right stick steering) still works: the local dir stays
# "forward" while facing rotates smoothly.
_STICK_SNAP_BAND_RAD = float(np.radians(15.0))


def _snap_dir_8way(vx: float, vy: float, band_rad: float) -> tuple[float, float]:
    """Snap a 2-D vector's angle to the nearest 8-way axis when within
    ``band_rad``; magnitude preserved. Outside the band, returned unchanged."""
    mag = float(np.hypot(vx, vy))
    if mag < 1e-9 or band_rad <= 0.0:
        return vx, vy
    ang = float(np.arctan2(vy, vx))
    sector = np.pi / 4.0  # 45 deg between 8-way axes
    nearest = round(ang / sector) * sector
    diff = (ang - nearest + np.pi) % (2.0 * np.pi) - np.pi  # signed dist to axis
    if abs(diff) <= band_rad:
        return mag * float(np.cos(nearest)), mag * float(np.sin(nearest))
    return vx, vy


# Discretized SLOW_WALK speed vs left-stick magnitude. Instead of a continuous
# analog speed (which jitters and storms the deploy's exact-`!=` replan trigger),
# quantize to 3 levels so movement_speed only changes on a deliberate band
# crossing:
#   mag in (0, 0.5) -> 0.25  (lower ~50%; 0.2 is below the kplanner's stepping
#                             threshold and doesn't translate, so floor at 0.25)
#   mag in [0.5,0.8) -> 0.4  (next ~30%)
#   mag in [0.8, 1.0] -> 0.6 (top ~20%, "maxing it out")
# EDGES are the up-boundaries between adjacent levels. HYST widens each edge into
# a dead band so noise sitting on a boundary can't flicker the level.
_SLOW_WALK_LEVELS = (0.25, 0.4, 0.6)
_SLOW_WALK_EDGES = (0.5, 0.8)   # len == len(_SLOW_WALK_LEVELS) - 1
_SLOW_WALK_HYST = 0.04


def _discretize_slow_walk_speed(mag: float, prev_speed: float) -> float:
    """3-level stepped speed with hysteresis. ``prev_speed`` is the last emitted
    level (for the dead band); ``mag`` is the deadzoned+rescaled stick magnitude
    in (0, 1]. Stepping up needs mag >= edge + HYST; stepping down needs
    mag < edge - HYST."""
    lvl = min(range(len(_SLOW_WALK_LEVELS)),
              key=lambda i: abs(_SLOW_WALK_LEVELS[i] - prev_speed))
    while lvl < len(_SLOW_WALK_EDGES) and mag >= _SLOW_WALK_EDGES[lvl] + _SLOW_WALK_HYST:
        lvl += 1
    while lvl > 0 and mag < _SLOW_WALK_EDGES[lvl - 1] - _SLOW_WALK_HYST:
        lvl -= 1
    return _SLOW_WALK_LEVELS[lvl]


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
        poll_hz: int = 50,
        zmq_feedback_host: str = "localhost",
        zmq_feedback_port: int = 5557,
        vr_input_max_age_s: float = 0.5,
        invert_lx: bool = False,
        invert_ly: bool = False,
        invert_rx: bool = False,
        invert_ry: bool = False,
        stick_filter_cfg: StickFilterConfig | None = None,
        sidecar_path: str | None = None,
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

        # --- ported from X2 stack (Phase A teleop upgrades) ---
        # Input freshness gate: when the WebSocket goes silent (headset asleep,
        # tab backgrounded, WS drop) Quest3Reader keeps returning its last
        # cached snapshot, so without this the manager would publish stale
        # sticks forever and the robot would keep walking. Snap locomotion
        # inputs to neutral when the last packet is older than this.
        self.vr_input_max_age_s = float(vr_input_max_age_s)
        self._vr_stale_active = False
        self._vr_last_warn_t = 0.0
        # Per-axis stick inversion (was hardcoded before).
        self.invert_lx, self.invert_ly = invert_lx, invert_ly
        self.invert_rx, self.invert_ry = invert_rx, invert_ry
        # Stick smoothing (LPF + slew). Default mild; None -> disabled.
        self._stick_filter = (
            StickFilter(stick_filter_cfg) if stick_filter_cfg is not None else None
        )
        self._last_tick_t = None
        # Optional per-tick sidecar JSONL of what we published (headset-free
        # replay / tuning). None -> disabled.
        self._sidecar = open(sidecar_path, "w") if sidecar_path else None

    def reset_yaw(self):
        self.yaw_accumulator.reset()
        if self._stick_filter is not None:
            self._stick_filter.reset()

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
            tick_now = time.monotonic()
            dt_tick = self.dt if self._last_tick_t is None \
                else max(1e-4, min(0.1, tick_now - self._last_tick_t))
            self._last_tick_t = tick_now

            # --- read all inputs once -----------------------------------------
            a, b, x, y = self.reader.get_buttons()
            lx, ly, rx, ry = self.reader.get_controller_axes()
            lt, rt, lg, rg = self.reader.get_controller_inputs()

            # per-axis stick inversion (was hardcoded before)
            if self.invert_lx:
                lx = -lx
            if self.invert_ly:
                ly = -ly
            if self.invert_rx:
                rx = -rx
            if self.invert_ry:
                ry = -ry

            # --- Quest 3 input freshness gate (safety) ------------------------
            # If the WS goes silent the reader keeps returning its last cached
            # snapshot; without this the robot keeps walking on stale sticks.
            vr_age_s = self.reader.get_last_message_age_s()
            if self.vr_input_max_age_s > 0.0 and vr_age_s > self.vr_input_max_age_s:
                lx = ly = rx = ry = 0.0
                a = b = x = y = False
                lt = rt = lg = rg = 0.0
                if not self._vr_stale_active:
                    log.warning(
                        "[Quest3Planner] input stale: last WS packet %.2fs ago "
                        "(threshold %.2fs); forcing sticks/buttons to neutral. "
                        "Headset asleep, tab backgrounded, or WS dropped?",
                        vr_age_s, self.vr_input_max_age_s,
                    )
                    self._vr_last_warn_t = tick_now
                elif tick_now - self._vr_last_warn_t > 30.0:
                    log.info("[Quest3Planner] input still stale (age %.1fs)", vr_age_s)
                    self._vr_last_warn_t = tick_now
                self._vr_stale_active = True
            elif self._vr_stale_active:
                log.info("[Quest3Planner] input recovered (age %.2fs); resuming", vr_age_s)
                self._vr_stale_active = False

            # --- stick smoothing (fwd<-ly, side<-lx, yaw<-rx) -----------------
            if self._stick_filter is not None:
                ly, lx, rx = self._stick_filter.step(
                    stick_fwd=ly, stick_side=lx, stick_yaw=rx, dt=dt_tick)

            # --- locomotion mode stepping (edge-triggered) --------------------
            ev = self.button_sm.tick(a, b, x, y)
            if ev.ab_pressed:
                self.mode = LocomotionMode(min(LocomotionMode.INJURED_WALK, self.mode + 1))
                print(f"[Quest3Planner] Mode -> {self.mode.value}: {self.mode.name}")
            if ev.xy_pressed:
                self.mode = LocomotionMode(max(LocomotionMode.IDLE, self.mode - 1))
                print(f"[Quest3Planner] Mode -> {self.mode.value}: {self.mode.name}")

            # periodic input log (only when active)
            now = time.time()
            if now - self._last_axes_log > 0.5:
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
                # reset the SLOW_WALK band so a resume from idle starts low
                self._slow_walk_speed = _SLOW_WALK_LEVELS[0]
            else:
                mode_to_send = self.mode

                if self.mode == LocomotionMode.SLOW_WALK:
                    # discretize to 0.2 / 0.4 / 0.6 by stick magnitude (with
                    # hysteresis) so movement_speed is stable and doesn't storm
                    # the deploy's exact-`!=` replan trigger. See
                    # _discretize_slow_walk_speed for the band edges.
                    self._slow_walk_speed = _discretize_slow_walk_speed(
                        mag, getattr(self, "_slow_walk_speed", _SLOW_WALK_LEVELS[0]))
                    speed = self._slow_walk_speed
                elif self.mode == LocomotionMode.WALK:
                    speed = -1.0
                elif self.mode == LocomotionMode.RUN:
                    speed = 1.5 + 3 * mag
                else:
                    speed = mag

            denom = raw_mag if raw_mag > 0.0 else 1.0
            scale = mag / denom
            # snap the movement direction to the nearest 8-way axis (facing
            # frame) to kill near-cardinal angular wobble that storms the
            # deploy's replan trigger; analog speed is unaffected.
            snap_x, snap_y = _snap_dir_8way(-lx, ly, _STICK_SNAP_BAND_RAD)
            movement_local = np.array([snap_x, snap_y]) * scale
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

                # reuse the gated triggers read at the top of run_once
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

            # optional per-tick sidecar (headset-free replay / tuning)
            if self._sidecar is not None:
                self._sidecar.write(json.dumps({
                    "t": tick_now, "stream_mode": stream_mode.name,
                    "mode": int(mode_to_send.value), "speed": float(speed),
                    "movement": [float(v) for v in movement],
                    "facing": [float(v) for v in facing],
                    "lx": float(lx), "ly": float(ly), "rx": float(rx),
                    "vr_stale": bool(self._vr_stale_active), "vr_age_s": float(vr_age_s),
                }) + "\n")
                self._sidecar.flush()

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
    poll_hz: int = 50,
    vr_input_max_age_s: float = 0.5,
    invert_lx: bool = False,
    invert_ly: bool = False,
    invert_rx: bool = False,
    invert_ry: bool = False,
    stick_filter: bool = True,
    sidecar_path: str | None = None,
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
        poll_hz=poll_hz,
        zmq_feedback_host=zmq_feedback_host,
        zmq_feedback_port=zmq_feedback_port,
        vr_input_max_age_s=vr_input_max_age_s,
        invert_lx=invert_lx, invert_ly=invert_ly,
        invert_rx=invert_rx, invert_ry=invert_ry,
        stick_filter_cfg=(_DEFAULT_STICK_FILTER if stick_filter else None),
        sidecar_path=sidecar_path,
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
                    # Idle the robot but STAY RUNNING: the deploy treats this
                    # stop command as a hold (operator_state.paused), not a
                    # shutdown, so chord A+B+X+Y again to restart PLANNER and
                    # keep capturing without relaunching. Ctrl-C fully quits.
                    socket.send(build_command_message(start=False, stop=True, planner=True))
                    print(f"[Manager] {current_mode.name} -> OFF (idled; A+B+X+Y to restart, Ctrl-C to quit)")
                else:
                    socket.send(build_command_message(start=True, stop=False, planner=True))
                    print(f"[Manager] {current_mode.name} -> {new_mode.name}")
                current_mode = new_mode

    except KeyboardInterrupt:
        print("\n[Manager] Interrupted")
    finally:
        reader.stop()
        three_point.close()
        if planner._sidecar is not None:
            planner._sidecar.close()
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
    # -- Phase A teleop upgrades (ported from the X2 stack) --
    parser.add_argument("--poll-hz", type=int, default=50,
                        help="Manager control rate (default 50; was 20)")
    parser.add_argument("--vr-input-max-age-s", type=float, default=0.5,
                        help="Snap sticks/buttons to neutral if the last WS packet "
                             "is older than this (safety gate). 0 disables (default 0.5)")
    parser.add_argument("--invert-lx", action="store_true", help="Invert left-stick X")
    parser.add_argument("--invert-ly", action="store_true", help="Invert left-stick Y")
    parser.add_argument("--invert-rx", action="store_true", help="Invert right-stick X")
    parser.add_argument("--invert-ry", action="store_true", help="Invert right-stick Y")
    parser.add_argument("--no-stick-filter", action="store_true",
                        help="Disable stick smoothing (LPF+slew); default on")
    parser.add_argument("--sidecar", type=str, default=None,
                        help="Write a per-tick JSONL of published commands to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

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
        poll_hz=args.poll_hz,
        vr_input_max_age_s=args.vr_input_max_age_s,
        invert_lx=args.invert_lx, invert_ly=args.invert_ly,
        invert_rx=args.invert_rx, invert_ry=args.invert_ry,
        stick_filter=not args.no_stick_filter,
        sidecar_path=args.sidecar,
    )
