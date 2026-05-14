"""X2-specific Quest 3 manager — sibling of ``quest3_manager_thread_server.py``.

This is the manager process for the Phase 0 planner-driven recorder
architecture. It owns the Quest 3 connection, runs the X2 retargeting
stack (calibrated arm IK + per-finger curl + finger filter + thumb
opposition), and publishes three ZMQ streams:

1. ``planner_cmd``  (JSON)    -> ``x2_heuristic_planner.py``
2. ``arm_targets`` + ``hand_finger_cmd`` + ``stream_mode``  (msgpack)
                              -> ``record_x2_dataset.py`` (subscribe-only mode)
3. ``recorder_cmd`` (JSON)    -> ``record_x2_dataset.py`` (episode control)

In Phase 0 the recorder also subscribes to the planner's ``body_pose``
topic and merges ``body_pose`` + ``arm_targets`` -> ``final_pose`` to
the deploy. This manager is the single source of truth for Quest 3
inputs across that pipeline; the recorder no longer touches the
headset.

See ``planner_driven_quest3_recorder_mvp_*.plan.md`` for the full
data-flow diagram.

Wire format
-----------

planner_cmd (ZMQ PUB, multipart):
    frame 0: ``b"planner_cmd"``
    frame 1: utf-8 JSON ``{"intent": str, "magnitude": str}``

arm_targets (ZMQ PUB, msgpack on topic ``arm_targets``):
    ``{"left_q_rad": [7], "right_q_rad": [7], "is_engaged": bool, "tick": int, "ts": float}``

hand_finger_cmd (ZMQ PUB, msgpack on topic ``hand_finger_cmd``):
    ``{"left_hand_q": [10], "right_hand_q": [10], "tick": int, "ts": float}``

stream_mode (ZMQ PUB, msgpack on topic ``stream_mode``):
    ``{"mode": str, "tick": int, "ts": float}``

recorder_cmd (ZMQ PUB, multipart):
    frame 0: ``b"recorder_cmd"``
    frame 1: utf-8 JSON ``{"action": str, "tick": int, "ts": float}``

Buttons (X2 manager-mediated mode)
----------------------------------

- ``A+B+X+Y`` chord: toggle OFF <-> LOCOMOTION (engage / E-stop)
- ``B`` single (only when not OFF): toggle LOCOMOTION <-> ARM_MANIPULATION
- ``A`` single (in ARM_MANIPULATION): toggle arm IK engaged
- ``X`` single (in ARM_MANIPULATION): start episode
  (forwarded to recorder; no-op when not in ARM_MANIPULATION)
- ``Y`` single (in ARM_MANIPULATION): stop & save episode
  (forwarded to recorder; no-op when not in ARM_MANIPULATION)
- Left stick (LOCOMOTION mode): ``fwd_step`` / ``back_step`` / ``side_*``
- Right stick (LOCOMOTION mode): ``turn_left`` / ``turn_right`` (deg_45)
- ``A`` held (LOCOMOTION mode): converts L-stick fwd/back into
  continuous ``walk / forward`` / ``walk / backward``
- ``X`` held (LOCOMOTION mode): upgrades hard R-stick rx into a
  90deg turn (``turn_left / deg_90`` / ``turn_right / deg_90``)
- Y held: was crouch; currently disabled in IntentDecoder

Note: the OmniHand fingers are driven from XRHand curls + thumb
opposition; per-side state freezes on a mode flip into LOCOMOTION so
the operator's relaxed grip during walking doesn't open the hand
mid-grasp.

Sidecar logging
---------------

Per-tick ``planner_cmd`` events are appended to a JSONL file (one line
per emit) so post-hoc analysis can correlate operator intent with the
recorded ``observation.*`` / ``action.*`` columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import msgpack
import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.finger_signal_filter import FingerFilterParams  # noqa: E402
from gear_sonic.utils.teleop.operator_calibration import OperatorCalibration  # noqa: E402
from gear_sonic.utils.teleop.vr.button_state_machine import ButtonStateMachine  # noqa: E402
from gear_sonic.utils.teleop.vr.intent_decoder import (  # noqa: E402
    IntentDecoder,
    LocomotionCmd,
    ModeTransition,
    StreamMode,
)
from gear_sonic.utils.teleop.vr.quest3_reader import Quest3Reader  # noqa: E402
from gear_sonic.utils.teleop.x2_retarget_pipeline import (  # noqa: E402
    Retargeter,
    RetargetTickInput,
)


log = logging.getLogger("quest3_manager_x2")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ManagerConfig:
    # Quest 3
    quest3_ws_port: int = 8765
    quest3_http_port: int = 8443
    quest3_use_ssl: bool = True

    # Calibration (default path resolves at construction-time so unit
    # tests can override REPO_ROOT without re-importing this module).
    calibration_path: Path = field(default_factory=lambda: _default_calibration_path())

    # Tick rate (Hz). The planner runs at 50 Hz; we match.
    publish_rate_hz: float = 50.0

    # PUB sockets
    planner_cmd_host: str = "*"
    planner_cmd_port: int = 5563
    planner_cmd_topic: str = "planner_cmd"

    recorder_pub_host: str = "*"
    recorder_pub_port: int = 5564

    # Topic names on the recorder PUB socket
    arm_targets_topic: str = "arm_targets"
    hand_finger_cmd_topic: str = "hand_finger_cmd"
    stream_mode_topic: str = "stream_mode"
    recorder_cmd_topic: str = "recorder_cmd"

    # IntentDecoder
    intent_stick_deadzone: float = 0.30
    intent_repeat_interval_s: float = 0.0
    # Both default to False because the curated bins are *replay*
    # primitives that snap the body back to standing instead of
    # holding the static pose -- pushing the right stick fwd or
    # softly L/R looks like a flicker to the operator and isn't
    # useful for live teleop yet. The right-stick X-axis hard
    # deflection (turn_*) is unaffected. Re-enable here (or via
    # --enable-lean-fwd / --enable-torso) once the planner can
    # hold the static lean / torso pose without snapping back.
    intent_enable_lean_fwd: bool = False
    intent_enable_torso: bool = False
    # Continuous waist hold via the right stick: pitch (ry > 0), yaw
    # (rx, no A), roll (rx, A held). Default True because this is the
    # primary VR teleop surface for static reach now that the planner
    # has STATIC_HOLD wired up. Set False to fall back to the legacy
    # discrete soft-band torso bins.
    intent_enable_continuous_torso: bool = True
    # Per-axis sign flips applied BEFORE the decoder sees the sticks.
    #
    # Operator UX contract: pushing the left stick AWAY from your body
    # must walk the robot forward in world; pulling it toward you must
    # walk the robot backward in world.
    #
    # Two layers shape the polarity:
    #   1. Hardware: Quest 3 controllers report ly < 0 when the stick
    #      is pushed forward (away from operator). The IntentDecoder
    #      is trained on the WebXR convention "+ly = forward push" and
    #      emits ``fwd_step`` on +ly, ``back_step`` on -ly.
    #   2. Bin world frame: the curated planner bins were authored in
    #      a body frame that is rotated 180 deg from the bridge's
    #      RSI init orientation, so the bin labelled ``fwd_step``
    #      actually translates the body BACKWARD in world (legs step
    #      forward in body-local frame, but the body's facing is
    #      reversed). Conversely the ``back_step_half_ft`` bin
    #      translates the body FORWARD in world.
    #
    # Net: leaving ``invert_ly = False`` lets the raw -ly from a
    # forward stick push fall through the decoder as ``back_step``,
    # which (after the world inversion above) actually moves the body
    # forward in world. That gives the operator the right end-to-end
    # behaviour without rebaking the bins or rotating the RSI anchor.
    # Operators who later fix the bin world frame (or use a different
    # RSI source) can restore the literal "+ly emits fwd_step" mapping
    # by passing ``--invert-ly``.
    invert_lx: bool = False
    invert_ly: bool = False
    invert_rx: bool = False
    invert_ry: bool = False

    # Retargeter
    ik_damping: float = 0.08
    ik_rotation_weight: float = 0.3
    ik_per_tick_step_rad: float = 0.30
    hand_input_mode: str = "trigger"
    apply_curl_compensation: bool = False
    apply_oppose_compensation: bool = False
    enable_finger_filter: bool = True

    # Sidecar
    sidecar_log_path: Optional[Path] = None
    """If set, append one JSONL line per emitted ``planner_cmd`` to this
    file. Useful for post-hoc analysis (which intent fired when)."""

    # Camera cycler (right-stick-click -> Tab to deploy MuJoCo viewer)
    enable_viewer_camera_cycler: bool = True
    """When True (default), pressing the right thumbstick click cycles
    the deploy MuJoCo viewer's fixed cameras via xdotool (Tab). Set
    to False on headless / CI runs where no GLFW window exists.

    TODO(unified-vr-input-topic): replace this entire xdotool path with
    a proper ZMQ ``vr_input`` topic that the manager publishes and any
    interested process (deploy viewer, recorder, future tools) can
    subscribe to. See ViewerCameraCycler.__doc__ for rationale."""
    viewer_window_pattern: str = "MuJoCo"
    """``xdotool search --name`` pattern used as a *fallback* to find
    the deploy viewer window. The cycler tries the more precise
    ``--classname`` search first (see ``viewer_window_classname``);
    this title-substring path only fires when the classname search
    returns nothing. Override if you have multiple MuJoCo windows
    open or if your deploy build sets a custom title (rare)."""
    viewer_window_classname: str = "MuJoCo"
    """``xdotool search --classname`` pattern -- the primary, more
    precise way to find the deploy viewer window. WM_CLASS is set
    by GLFW on the application window itself and is NOT inherited
    by the GNOME / mutter compositor's frame wrapper, so this filter
    avoids the "Tab vanishes into mutter-x11-frames" failure mode
    we hit before 2026-05-13. Only override if you've rebuilt
    MuJoCo with a custom WM_CLASS string (very rare)."""

    # Misc
    verbose: bool = False


def _default_calibration_path() -> Path:
    """Default to the repo's standard operator_calibrations path."""
    return REPO_ROOT / "data" / "operator_calibrations" / "default.yaml"


# ---------------------------------------------------------------------------
# Helpers: ZMQ wire format
# ---------------------------------------------------------------------------


def _planner_cmd_payload(cmd: LocomotionCmd) -> bytes:
    """Build the JSON payload the planner's _zmq_command_thread expects.

    For ``hold_torso`` commands we also serialize the continuous waist
    targets; the planner's ``_zmq_command_thread`` reads them as
    optional fields and feeds them into ``LocomotionCommand.waist_*_deg``.
    For every other intent we omit them (defaulting to 0.0 on the
    receiving end), which keeps wire payloads minimal and matches the
    pre-v7 wire format.
    """
    payload: dict[str, object] = {
        "intent": cmd.intent,
        "magnitude": cmd.magnitude,
    }
    if cmd.intent == "hold_torso":
        payload["waist_pitch_deg"] = float(cmd.waist_pitch_deg)
        payload["waist_roll_deg"] = float(cmd.waist_roll_deg)
        payload["waist_yaw_deg"] = float(cmd.waist_yaw_deg)
    return json.dumps(payload).encode("utf-8")


def _recorder_cmd_payload(action: str, tick: int) -> bytes:
    return json.dumps({
        "action": action,
        "tick": tick,
        "ts": time.time(),
    }).encode("utf-8")


def _msgpack_payload(data: dict) -> bytes:
    return msgpack.packb(data, use_bin_type=True)


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------


class Quest3ManagerX2:
    """X2-specific Quest 3 manager process.

    Runs a single 50 Hz loop. Owns:
    - :class:`Quest3Reader` (Quest 3 WebXR ingest)
    - :class:`IntentDecoder` (stick + button -> ``LocomotionCmd``)
    - :class:`Retargeter`    (VR + hand inputs -> arm + hand commands)
    - Two PUB sockets        (planner + recorder fan-out)
    """

    def __init__(self, cfg: ManagerConfig) -> None:
        self._cfg = cfg
        self._stop = threading.Event()

        self._calibration = self._resolve_calibration()
        self._retargeter = Retargeter(
            calibration=self._calibration,
            finger_filter_params=(
                FingerFilterParams() if cfg.enable_finger_filter else None
            ),
            ik_damping=cfg.ik_damping,
            ik_rotation_weight=cfg.ik_rotation_weight,
            ik_per_tick_step_rad=cfg.ik_per_tick_step_rad,
            hand_input_mode=cfg.hand_input_mode,
            apply_curl_compensation=cfg.apply_curl_compensation,
            apply_oppose_compensation=cfg.apply_oppose_compensation,
        )
        self._intent = IntentDecoder(
            stick_deadzone=cfg.intent_stick_deadzone,
            repeat_interval_s=cfg.intent_repeat_interval_s,
            enable_lean_fwd=cfg.intent_enable_lean_fwd,
            enable_torso=cfg.intent_enable_torso,
            enable_continuous_torso=cfg.intent_enable_continuous_torso,
        )
        # Latched continuous waist target. Set in two situations:
        #   1) The operator presses B to flip LOCOMOTION ->
        #      ARM_MANIPULATION (existing behavior; ARM_MAN is implicitly
        #      a hold because the right stick is a no-op for waist there).
        #   2) The operator presses the right thumbstick CLICK while in
        #      LOCOMOTION (or ARM_MANIPULATION) to toggle ``_waist_frozen``
        #      ON; the live waist target is captured here so the planner
        #      can be re-pinned later if needed.
        # ``None`` means "no latch active" (live stick drives the waist).
        self._latched_waist: tuple[float, float, float] | None = None
        # R-thumbstick-click freeze toggle. Independent of mode: a press
        # in LOCOMOTION freezes the body so the operator can keep
        # leaning/twisting while walking with the L stick; the freeze
        # persists across B-press mode flips so the body stays leaned
        # through ARM_MANIPULATION and back. Toggled off by another
        # R-click. Reset on any transition to OFF.
        self._waist_frozen: bool = False
        self._button_sm = ButtonStateMachine(log_prefix="Input")

        # Stick-click rising-edge trackers. The WebXR client polls the
        # gamepad ~50 Hz so a click typically holds True for several
        # ticks. We only fire on the press transition (False -> True);
        # this mirrors the ``ButtonStateMachine`` debounce for the four
        # face buttons. Left click cycles deploy MuJoCo viewer cameras
        # (was on right click pre-v7); right click toggles waist freeze.
        self._prev_left_stick_click = False
        self._prev_right_stick_click = False

        # Camera cycler. Always constructed (even when xdotool isn't
        # installed) so the cycle() call path is exercised on every
        # press; the helper logs a one-shot warning and no-ops if its
        # prerequisites are missing. See ViewerCameraCycler docstring
        # for the planned vr_input-topic replacement.
        if cfg.enable_viewer_camera_cycler:
            from gear_sonic.utils.teleop.vr.viewer_camera_cycler import (
                ViewerCameraCycler,
            )
            self._viewer_cycler: Optional[ViewerCameraCycler] = (
                ViewerCameraCycler(
                    window_search_pattern=cfg.viewer_window_pattern,
                    window_class_name=cfg.viewer_window_classname,
                )
            )
        else:
            self._viewer_cycler = None

        self._quest = Quest3Reader(
            ws_port=cfg.quest3_ws_port,
            http_port=cfg.quest3_http_port,
            use_ssl=cfg.quest3_use_ssl,
        )

        self._ctx = zmq.Context.instance()
        self._planner_sock = self._ctx.socket(zmq.PUB)
        self._planner_sock.setsockopt(zmq.LINGER, 0)
        self._planner_sock.bind(
            f"tcp://{cfg.planner_cmd_host}:{cfg.planner_cmd_port}"
        )
        log.info(
            "planner_cmd PUB bound at tcp://%s:%d (topic=%s)",
            cfg.planner_cmd_host, cfg.planner_cmd_port, cfg.planner_cmd_topic,
        )
        self._recorder_sock = self._ctx.socket(zmq.PUB)
        self._recorder_sock.setsockopt(zmq.LINGER, 0)
        self._recorder_sock.bind(
            f"tcp://{cfg.recorder_pub_host}:{cfg.recorder_pub_port}"
        )
        log.info(
            "recorder PUB bound at tcp://%s:%d (topics=%s, %s, %s, %s)",
            cfg.recorder_pub_host, cfg.recorder_pub_port,
            cfg.arm_targets_topic, cfg.hand_finger_cmd_topic,
            cfg.stream_mode_topic, cfg.recorder_cmd_topic,
        )

        # Last published arm + hand targets (used for freezing in
        # LOCOMOTION mode so the recorder gets a steady stream).
        self._frozen_left_arm_q = self._retargeter._teleop._left_q.copy()
        self._frozen_right_arm_q = self._retargeter._teleop._right_q.copy()
        self._frozen_left_hand_q = np.zeros(10, dtype=np.float64)
        self._frozen_right_hand_q = np.zeros(10, dtype=np.float64)

        # One-shot OFF-mode operator hint (re-armed on every successful
        # mode transition). Stops noisy spam every time the operator
        # presses a single button while still in OFF.
        self._off_mode_hint_logged = False

        if cfg.sidecar_log_path is not None:
            cfg.sidecar_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._sidecar = cfg.sidecar_log_path.open("a", buffering=1)
            log.info("sidecar log -> %s", cfg.sidecar_log_path)
        else:
            self._sidecar = None

    # -- lifecycle ------------------------------------------------------------

    def _resolve_calibration(self) -> OperatorCalibration:
        cal_path = self._cfg.calibration_path
        if not cal_path.is_file():
            raise SystemExit(
                f"calibration not found at {cal_path}. Run "
                "`python -m gear_sonic.scripts.vr_operator_calibrate "
                "--operator-id default` first."
            )
        cal = OperatorCalibration.load_yaml(cal_path)
        log.info(
            "loaded calibration %s (operator=%s, L_residual=%.1fcm, "
            "R_residual=%.1fcm)",
            cal_path,
            cal.operator_id,
            cal.fit["left"].residual_m * 100,
            cal.fit["right"].residual_m * 100,
        )
        return cal

    def start(self) -> None:
        self._quest.start()
        # Allow PUB-SUB to settle before flooding the wire.
        time.sleep(0.2)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._quest.stop()
        except Exception:
            pass
        try:
            self._planner_sock.close(linger=0)
        except Exception:
            pass
        try:
            self._recorder_sock.close(linger=0)
        except Exception:
            pass
        if self._sidecar is not None:
            try:
                self._sidecar.close()
            except Exception:
                pass

    # -- main loop ------------------------------------------------------------

    def run(self) -> int:
        period = 1.0 / max(self._cfg.publish_rate_hz, 1e-6)
        next_tick = time.monotonic()
        tick = 0
        wait_logged = False

        log.info(
            "[manager-x2] running. mode=OFF; press A+B+X+Y to engage. "
            "Publishing planner_cmd to tcp://%s:%d, "
            "arm_targets/hand_finger_cmd/stream_mode/recorder_cmd to "
            "tcp://%s:%d",
            self._cfg.planner_cmd_host, self._cfg.planner_cmd_port,
            self._cfg.recorder_pub_host, self._cfg.recorder_pub_port,
        )
        log.info(
            "[manager-x2] stick polarity: invert_lx=%s invert_ly=%s "
            "invert_rx=%s invert_ry=%s (push --no-invert-ly etc. to flip)",
            self._cfg.invert_lx, self._cfg.invert_ly,
            self._cfg.invert_rx, self._cfg.invert_ry,
        )

        try:
            while not self._stop.is_set():
                tick_now = time.monotonic()
                vr_pose = self._quest.get_3pt_pose()
                buttons = self._quest.get_buttons()
                triggers = self._quest.get_controller_inputs()
                lx, ly, rx, ry = self._quest.get_controller_axes()
                if self._cfg.invert_lx:
                    lx = -lx
                if self._cfg.invert_ly:
                    ly = -ly
                if self._cfg.invert_rx:
                    rx = -rx
                if self._cfg.invert_ry:
                    ry = -ry

                if vr_pose is None:
                    if not wait_logged:
                        log.info(
                            "waiting for first Quest 3 packet "
                            "(open WebXR app at "
                            "https://<HOST>:%d) ...",
                            self._cfg.quest3_http_port,
                        )
                        wait_logged = True
                    self._publish_stream_mode(tick)
                    self._sleep_until(next_tick + period)
                    next_tick += period
                    continue
                wait_logged = False

                ev = self._button_sm.tick(*buttons)
                a_held, _b_held, x_held, y_held = buttons

                # Live continuous waist target derived from the right
                # stick + A modifier. We compute this BEFORE the mode
                # transition handler so a B-press into ARM_MANIPULATION
                # can latch exactly the pose the operator was holding
                # at the moment of the press, rather than the pose from
                # the previous tick (which would drift by up to one
                # 50 Hz interval). The continuous target is well-defined
                # in any mode (it's a pure function of stick state) but
                # only consumed on the LOCOMOTION -> ARM_MANIPULATION
                # transition.
                live_waist_target = self._intent.continuous_waist_target(
                    rx=rx, ry=ry,
                    a_held=(a_held and self._intent.mode == StreamMode.LOCOMOTION),
                )

                # 1) Mode transitions ---------------------------------------
                transition = self._intent.update_mode(ev, now=tick_now)
                if transition is not None:
                    self._on_mode_transition(
                        transition,
                        vr_pose=vr_pose,
                        tick=tick,
                        live_waist_target=live_waist_target,
                    )

                # 2) Operator-facing UX hint: in OFF mode, A/B/X/Y by
                #    themselves do nothing useful. Print a one-shot hint
                #    so newcomers know they need the A+B+X+Y chord to
                #    leave OFF, then B once to flip to ARM_MANIPULATION.
                #    Re-armed after every successful mode transition.
                if self._intent.mode == StreamMode.OFF and (
                    ev.a_pressed or ev.b_pressed or ev.x_pressed or ev.y_pressed
                ) and not ev.abxy_pressed and not self._off_mode_hint_logged:
                    log.info(
                        "Mode is OFF; single button press is a no-op. "
                        "Press A+B+X+Y SIMULTANEOUSLY to enter LOCOMOTION, "
                        "then press B alone to flip to ARM_MANIPULATION "
                        "(arms IK), then A to engage arm tracking."
                    )
                    self._off_mode_hint_logged = True

                # 3) Arm IK engage toggle (manager-internal; recorder
                #    does not need to know).
                if ev.a_pressed and self._intent.mode == StreamMode.ARM_MANIPULATION:
                    self._retargeter.set_engaged(not self._retargeter.is_engaged)
                    log.info(
                        "[A] arm tracking -> %s",
                        "ACTIVE" if self._retargeter.is_engaged else "IDLE",
                    )

                # 4) Episode lifecycle (ARM_MANIPULATION ONLY).
                #    The recorder consumes 'start' / 'save' / 'discard'
                #    actions; mismatched names silently no-op. We gate
                #    every recording trigger on ARM_MAN mode so:
                #      - A and X behave purely as locomotion modifiers
                #        in LOCOMOTION (A held = walk, X held + rx =
                #        90° turn) without ever poking the recorder.
                #      - The operator cannot accidentally start / save
                #        an episode while still positioning the robot.
                #
                # Mapping (deliberately single-button, no chord):
                #   X press -> 'start'  (only when not currently recording)
                #   Y press -> 'save'   (only when currently recording)
                #
                # The previous A+B chord for 'start' collided with the
                # B-single mode toggle: pressing them in the same tick
                # both started a recording AND flipped to LOCOMOTION,
                # immediately yanking the planner reference out from
                # under the freshly-opened episode. Splitting start /
                # save onto two distinct one-shot buttons removes the
                # chord entirely so B is always unambiguously a mode
                # toggle. Discard mid-episode is intentionally not
                # bound to a button right now (kill the manager and
                # delete the latest parquet/mp4 if you need to drop a
                # bad episode -- the recorder still understands the
                # 'discard' wire action if we re-bind it later).
                in_arm_man = self._intent.mode == StreamMode.ARM_MANIPULATION
                if in_arm_man and ev.x_pressed:
                    log.info("[X] start episode forwarded to recorder")
                    self._publish_recorder_cmd("start", tick)
                    self._play_audio_prompt("record_start", fallback="Recording.")
                if in_arm_man and ev.y_pressed:
                    log.info("[Y] save episode forwarded to recorder")
                    self._publish_recorder_cmd("save", tick)
                    self._play_audio_prompt("record_save", fallback="Saved.")

                # 4b) Stick clicks ------------------------------------------
                #     LEFT thumbstick click  -> cycle deploy MuJoCo viewer
                #         cameras (xdotool Tab keypress). Pre-v7 this was
                #         the right click; moved here so the operator can
                #         keep their right thumb on the lean / twist stick
                #         while clicking the LEFT stick to re-frame.
                #     RIGHT thumbstick click -> toggle ``_waist_frozen``.
                #         While frozen, the right stick is suppressed for
                #         waist control: the planner stays in STATIC_HOLD
                #         at the pose the operator was holding when the
                #         click landed. Another R-click releases it.
                #
                #     Both are active in LOCOMOTION + ARM_MAN, idle in
                #     OFF (consistent with the rest of the manager:
                #     OFF means "ignore controller events"). Rising-edge
                #     tracked manually because ButtonStateMachine only
                #     handles the four face buttons today.
                #
                # TODO(unified-vr-input-topic): once the manager
                # publishes a unified ``vr_input`` ZMQ topic carrying
                # the full controller state, the deploy viewer should
                # subscribe directly and update mjvCamera in-process,
                # making the xdotool hack unnecessary. See
                # ViewerCameraCycler.__doc__ for the rationale.
                l_click, r_click = self._quest.get_stick_clicks()
                l_click_edge = l_click and not self._prev_left_stick_click
                r_click_edge = r_click and not self._prev_right_stick_click
                self._prev_left_stick_click = l_click
                self._prev_right_stick_click = r_click
                if (
                    l_click_edge
                    and self._intent.mode != StreamMode.OFF
                    and self._viewer_cycler is not None
                ):
                    if self._viewer_cycler.cycle():
                        log.info("[L-click] cycled deploy viewer camera (Tab)")
                if r_click_edge and self._intent.mode != StreamMode.OFF:
                    self._toggle_waist_freeze(live_waist_target)

                # 3) Locomotion command ---------------------------------------
                # Held-button modifiers (LOCOMOTION-only via decoder
                # short-circuit). A held + ly = continuous walk;
                # X held + rx = 90° turn. Y held was crouch; currently
                # gated off in IntentDecoder. Buttons already destructured
                # above so the live_waist_target sample agrees with the
                # decoder's view of the held modifiers.
                cmd = self._intent.decode_locomotion(
                    lx=lx, ly=ly, rx=rx, ry=ry,
                    y_held=(y_held and self._intent.mode == StreamMode.LOCOMOTION),
                    a_held=(a_held and self._intent.mode == StreamMode.LOCOMOTION),
                    x_held=(x_held and self._intent.mode == StreamMode.LOCOMOTION),
                    now=tick_now,
                )
                if cmd is not None:
                    # Drop live ``hold_torso`` updates while the operator
                    # has the waist frozen (R-click toggle). The planner
                    # stays in STATIC_HOLD at its current target because
                    # no new hold_torso commands arrive; non-hold commands
                    # (walk / turn / idle) still flow through so the
                    # operator can keep walking with the body leaned.
                    if self._waist_frozen and cmd.intent == "hold_torso":
                        pass
                    else:
                        self._publish_planner_cmd(cmd)
                        self._sidecar_emit(cmd, tick)

                # In ARM_MANIPULATION we keep the planner held at idle by
                # emitting an idle command on entry (handled in the mode
                # transition); we do NOT emit one per tick here because
                # the planner's queue dedupes anyway and we want to leave
                # bandwidth for the operator to flip back to LOCOMOTION
                # without a stale walk command lurking in the queue.

                # 4) Retargeting ---------------------------------------------
                # We RUN the retargeter every tick (even in LOCOMOTION
                # mode) so the IK doesn't snap when the operator
                # switches back to ARM. But the published values follow
                # the freeze rule below.
                inp = self._build_retarget_input(vr_pose=vr_pose, triggers=triggers)
                out = self._retargeter.step(inp)

                if self._intent.mode == StreamMode.ARM_MANIPULATION:
                    publish_left_arm = out.left_arm_q
                    publish_right_arm = out.right_arm_q
                    publish_left_hand = out.left_hand_q
                    publish_right_hand = out.right_hand_q
                    # Refresh the freeze cache so a flip back to
                    # LOCOMOTION continues from the latest pose.
                    self._frozen_left_arm_q = out.left_arm_q.copy()
                    self._frozen_right_arm_q = out.right_arm_q.copy()
                    self._frozen_left_hand_q = out.left_hand_q.copy()
                    self._frozen_right_hand_q = out.right_hand_q.copy()
                else:
                    # OFF or LOCOMOTION: hold last commanded arm + hand
                    # targets. Walking with VR-driven arms is unsafe
                    # because the operator's hands aren't visible in
                    # their HMD view while they're looking at the floor.
                    publish_left_arm = self._frozen_left_arm_q
                    publish_right_arm = self._frozen_right_arm_q
                    publish_left_hand = self._frozen_left_hand_q
                    publish_right_hand = self._frozen_right_hand_q

                self._publish_arm_targets(
                    left=publish_left_arm,
                    right=publish_right_arm,
                    is_engaged=self._retargeter.is_engaged,
                    tick=tick,
                )
                self._publish_hand_finger_cmd(
                    left=publish_left_hand,
                    right=publish_right_hand,
                    tick=tick,
                )
                self._publish_stream_mode(tick)

                tick += 1
                next_tick += period
                self._sleep_until(next_tick)
        except KeyboardInterrupt:
            log.info("\n[manager-x2] interrupted")
        finally:
            self.stop()

        return tick

    # -- per-tick helpers -----------------------------------------------------

    def _build_retarget_input(
        self,
        *,
        vr_pose: np.ndarray,
        triggers: tuple[float, float, float, float],
    ) -> RetargetTickInput:
        l_curls, r_curls, l_src, r_src = self._quest.get_hand_curls()
        l_oppose, r_oppose = self._quest.get_thumb_opposition()
        l_tip, r_tip = self._quest.get_finger_tip_oppose()
        return RetargetTickInput(
            vr_pose=vr_pose,
            triggers=tuple(float(x) for x in triggers),
            left_curls=l_curls,
            right_curls=r_curls,
            left_thumb_oppose=None if l_oppose is None else float(l_oppose),
            right_thumb_oppose=None if r_oppose is None else float(r_oppose),
            left_finger_tip_oppose=l_tip,
            right_finger_tip_oppose=r_tip,
            left_hand_source=l_src,
            right_hand_source=r_src,
        )

    def _toggle_waist_freeze(
        self,
        live_waist_target: tuple[float, float, float],
    ) -> None:
        """R-thumbstick-click handler: toggle waist freeze on/off.

        On freeze ON: the live waist target at the moment of click is
        captured into ``_latched_waist`` so subsequent code paths (e.g.
        a B-press into ARM_MANIPULATION while frozen) source the
        latched pose instead of resampling. The decoder keeps emitting
        ``hold_torso`` updates internally, but the manager suppresses
        them at publish time (see the ``_waist_frozen`` check in the
        main loop), so the planner stays at its current STATIC_HOLD
        target even as the operator's right stick drifts.

        On freeze OFF: ``_latched_waist`` is cleared and the suppression
        lifts. The next decoder tick re-emits the live target, so the
        planner blends to whatever the operator is now holding (or to
        neutral, if the stick is centered).
        """
        if self._waist_frozen:
            self._waist_frozen = False
            self._latched_waist = None
            log.info("[R-click] waist freeze -> RELEASED")
            self._play_audio_prompt(
                "torso_released", fallback="Torso released.",
            )
        else:
            pitch, roll, yaw = live_waist_target
            self._waist_frozen = True
            self._latched_waist = (pitch, roll, yaw)
            log.info(
                "[R-click] waist freeze -> FROZEN at "
                "pitch=%+.1f roll=%+.1f yaw=%+.1f",
                pitch, roll, yaw,
            )
            self._play_audio_prompt(
                "torso_frozen", fallback="Torso frozen.",
            )

    def _on_mode_transition(
        self,
        transition: ModeTransition,
        *,
        vr_pose: np.ndarray,
        tick: int,
        live_waist_target: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        log.info("[manager-x2] mode %s -> %s", transition.previous.name, transition.current.name)
        # Re-arm the OFF-mode hint so it fires once again on next entry
        # to OFF (helps if the operator forgets the chord mid-session).
        self._off_mode_hint_logged = False
        # Whenever we leave OFF we want a fresh idle planner_cmd so the
        # planner clears any stale queue entries and goes to idle_stand.
        if transition.previous == StreamMode.OFF:
            self._publish_planner_cmd(LocomotionCmd("idle", "default"))

        # ----- LOCOMOTION <-> ARM_MANIPULATION transitions ---------------
        # Going INTO ARM_MANIPULATION: latch whatever pitch / roll / yaw
        # the operator was holding via the right stick at the moment of
        # B-press, and pin the planner's STATIC_HOLD to that pose. This
        # mirrors the existing arm-latch on the reverse transition: the
        # operator picks an upper-body pose in LOCOMOTION, B-clicks into
        # ARM_MANIPULATION, then drives arms via VR IK while the lower
        # body stays at the chosen lean / twist for extra reach.
        #
        # Going OUT of ARM_MANIPULATION (back to LOCOMOTION): clear the
        # latch and emit a single idle cmd so the planner cleanly
        # blends out of STATIC_HOLD; subsequent ticks resume normal
        # continuous emission from the right stick.
        if (
            transition.previous == StreamMode.LOCOMOTION
            and transition.current == StreamMode.ARM_MANIPULATION
        ):
            # Source the latch from whichever target is currently
            # authoritative: if the operator already R-clicked to freeze
            # in LOCOMOTION, use the frozen pose (so the B-press doesn't
            # snap to a slightly different live sample). Otherwise, take
            # the live continuous target as today.
            if self._waist_frozen and self._latched_waist is not None:
                pitch, roll, yaw = self._latched_waist
            else:
                pitch, roll, yaw = live_waist_target
                self._latched_waist = (pitch, roll, yaw)
            self._publish_planner_cmd(
                LocomotionCmd(
                    intent="hold_torso",
                    magnitude="continuous",
                    waist_pitch_deg=pitch,
                    waist_roll_deg=roll,
                    waist_yaw_deg=yaw,
                )
            )
            self._retargeter.reset_finger_filter()
            log.info(
                "[manager-x2] latched waist hold pitch=%+.1f roll=%+.1f yaw=%+.1f"
                "%s",
                pitch, roll, yaw,
                " (R-click freeze active)" if self._waist_frozen else "",
            )
            # Audio cue: separate "torso_locked" prompt only when the
            # latched pose is meaningfully non-neutral (>= 1 deg on
            # any axis). For neutral poses the standard
            # "mode_arm_manipulation" cue below covers it.
            if max(abs(pitch), abs(roll), abs(yaw)) >= 1.0:
                self._play_audio_prompt(
                    "mode_torso_locked", fallback="Torso locked.",
                )
        elif (
            transition.previous == StreamMode.ARM_MANIPULATION
            and transition.current == StreamMode.LOCOMOTION
        ):
            # If the operator R-click-froze the waist before / during
            # ARM_MANIPULATION, KEEP the freeze across the transition
            # back to LOCOMOTION: don't clear ``_latched_waist`` and
            # don't blow the planner's STATIC_HOLD away with an idle
            # cmd. The operator can now walk / turn with the L stick
            # while the body stays at the locked pose. R-click again
            # to release. If the freeze flag is OFF, fall through to
            # the legacy release-into-idle path so the body smoothly
            # blends back to standing.
            if self._waist_frozen and self._latched_waist is not None:
                log.info(
                    "[manager-x2] ARM->LOCO with R-click freeze active; "
                    "keeping STATIC_HOLD at latched pose"
                )
            else:
                self._latched_waist = None
                self._publish_planner_cmd(LocomotionCmd("idle", "default"))
        elif transition.current == StreamMode.ARM_MANIPULATION:
            # Reached ARM_MANIPULATION not via LOCOMOTION (e.g. would
            # only happen if a future chord adds a direct OFF -> ARM
            # path). Keep the legacy "planner idles" semantics.
            self._publish_planner_cmd(LocomotionCmd("idle", "default"))
            self._retargeter.reset_finger_filter()

        # On leaving an active mode -> OFF: tell the recorder to drop
        # any in-progress episode (this is a hard E-stop semantic;
        # operator can re-arm and start fresh). Also drop the R-click
        # freeze so the next engagement starts with a clean slate.
        if transition.current == StreamMode.OFF and transition.previous != StreamMode.OFF:
            self._latched_waist = None
            self._waist_frozen = False
            self._publish_planner_cmd(LocomotionCmd("idle", "default"))
            self._publish_recorder_cmd("estop", tick)

        # Headset audio cue for the transition. Mapping is the same
        # vocabulary as the IntentDecoder.StreamMode enum so adding a
        # new mode automatically requires a matching prompt key (the
        # ``test_every_stream_mode_has_a_prompt`` contract test in
        # tests/test_quest3_audio_prompts.py guards against drift).
        _MODE_AUDIO_KEYS = {
            StreamMode.OFF:               ("mode_off",              "Off."),
            StreamMode.LOCOMOTION:        ("mode_locomotion",       "Locomotion."),
            StreamMode.ARM_MANIPULATION:  ("mode_arm_manipulation", "Arm manipulation."),
        }
        cue = _MODE_AUDIO_KEYS.get(transition.current)
        if cue is not None:
            key, fallback = cue
            self._play_audio_prompt(key, fallback=fallback)

    # -- publishers -----------------------------------------------------------

    def _publish_planner_cmd(self, cmd: LocomotionCmd) -> None:
        try:
            self._planner_sock.send_multipart(
                [
                    self._cfg.planner_cmd_topic.encode("ascii"),
                    _planner_cmd_payload(cmd),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass
        if self._cfg.verbose:
            log.debug("planner_cmd <- intent=%s magnitude=%s", cmd.intent, cmd.magnitude)

    def _publish_arm_targets(
        self,
        *,
        left: np.ndarray,
        right: np.ndarray,
        is_engaged: bool,
        tick: int,
    ) -> None:
        payload = {
            "left_q_rad": np.asarray(left, dtype=np.float32).tolist(),
            "right_q_rad": np.asarray(right, dtype=np.float32).tolist(),
            "is_engaged": bool(is_engaged),
            "tick": int(tick),
            "ts": time.time(),
        }
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.arm_targets_topic.encode("ascii"),
                    _msgpack_payload(payload),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _publish_hand_finger_cmd(
        self,
        *,
        left: np.ndarray,
        right: np.ndarray,
        tick: int,
    ) -> None:
        payload = {
            "left_hand_q": np.asarray(left, dtype=np.float32).tolist(),
            "right_hand_q": np.asarray(right, dtype=np.float32).tolist(),
            "tick": int(tick),
            "ts": time.time(),
        }
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.hand_finger_cmd_topic.encode("ascii"),
                    _msgpack_payload(payload),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _publish_stream_mode(self, tick: int) -> None:
        payload = {
            "mode": self._intent.mode.name,
            "tick": int(tick),
            "ts": time.time(),
        }
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.stream_mode_topic.encode("ascii"),
                    _msgpack_payload(payload),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _publish_recorder_cmd(self, action: str, tick: int) -> None:
        try:
            self._recorder_sock.send_multipart(
                [
                    self._cfg.recorder_cmd_topic.encode("ascii"),
                    _recorder_cmd_payload(action, tick),
                ],
                flags=zmq.NOBLOCK,
            )
        except zmq.Again:
            pass

    def _play_audio_prompt(
        self, key: str, *, fallback: Optional[str] = None,
    ) -> None:
        """Push a ``play_audio`` message to the WebXR client.

        The client maps ``key`` to ``/audio/<key>.mp3`` (cached on disk
        by ``ensure_prompt_audio_files``) and falls back to
        ``speechSynthesis`` with ``fallback`` if the MP3 is missing.
        Silently no-ops if no headset is connected (e.g. during
        startup before the operator clicks "Enter VR") -- audio cues
        are best-effort feedback, never a hard dependency.
        """
        try:
            payload = {"_type": "play_audio", "key": key}
            if fallback:
                payload["fallback"] = fallback
            self._quest.send_message(payload)
        except Exception as exc:
            # Audio is decorative; never let a send failure crash the
            # control loop. Log at debug because the most common cause
            # (no client connected yet) is expected on startup.
            log.debug("[manager-x2] audio prompt %r send failed: %s", key, exc)

    # -- sidecar --------------------------------------------------------------

    def _sidecar_emit(self, cmd: LocomotionCmd, tick: int) -> None:
        if self._sidecar is None:
            return
        rec = {
            "tick": int(tick),
            "ts": time.time(),
            "intent": cmd.intent,
            "magnitude": cmd.magnitude,
            "stream_mode": self._intent.mode.name,
        }
        self._sidecar.write(json.dumps(rec) + "\n")

    @staticmethod
    def _sleep_until(deadline_mono: float) -> None:
        rem = deadline_mono - time.monotonic()
        if rem > 0:
            time.sleep(rem)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="X2 Quest 3 manager — planner_cmd + arm_targets + hand_finger_cmd",
    )
    # Quest 3
    p.add_argument("--ws-port", type=int, default=8765)
    p.add_argument("--http-port", type=int, default=8443)
    p.add_argument("--no-ssl", action="store_true", help="disable TLS for WebXR")

    # Calibration
    p.add_argument(
        "--calibration",
        type=Path,
        default=_default_calibration_path(),
        help="Operator calibration YAML (default: data/operator_calibrations/default.yaml)",
    )

    # Planner output
    p.add_argument("--planner-cmd-host", default="*")
    p.add_argument("--planner-cmd-port", type=int, default=5563)
    p.add_argument("--planner-cmd-topic", default="planner_cmd")

    # Recorder output
    p.add_argument("--recorder-pub-host", default="*")
    p.add_argument("--recorder-pub-port", type=int, default=5564)

    # Cadence
    p.add_argument("--rate", type=float, default=50.0, help="Publish rate Hz")

    # IntentDecoder
    p.add_argument("--stick-deadzone", type=float, default=0.30)
    p.add_argument(
        "--repeat-interval", type=float, default=0.0,
        help="If > 0, re-emit the most recent planner_cmd at this cadence (s)",
    )
    # Right-stick primitives currently authored as replay-and-snap-back
    # bins, off by default. Flip on once the planner can hold a static
    # lean / torso pose without snapping.
    p.add_argument(
        "--enable-lean-fwd", dest="enable_lean_fwd", action="store_true",
        default=False,
        help="Re-enable graded lean_fwd_{small,medium,large} on R-stick "
             "fwd push (off by default; bin currently snaps back)",
    )
    p.add_argument(
        "--enable-torso", dest="enable_torso", action="store_true",
        default=False,
        help="Re-enable torso_left_30deg / torso_right_30deg on soft "
             "R-stick L/R (off by default; bin currently snaps back)",
    )

    # Stick polarity (axis sign flips applied before the decoder).
    # All default to False: the operator-facing UX of "push the stick
    # the way you want the robot to move" is achieved by the natural
    # cancellation between the Quest 3's hardware sign convention
    # (-ly = stick fwd) and the curated bin world frame (the bin
    # labelled 'back_step' actually translates the body fwd in
    # world). See the ManagerConfig.invert_ly docstring for the full
    # picture. Flip per-axis here if you re-author the bins or use a
    # different RSI source.
    inv_grp = p.add_argument_group("stick polarity")
    inv_grp.add_argument("--invert-lx", dest="invert_lx", action="store_true",
                         default=False)
    inv_grp.add_argument("--no-invert-lx", dest="invert_lx",
                         action="store_false")
    inv_grp.add_argument("--invert-ly", dest="invert_ly", action="store_true",
                         default=False)
    inv_grp.add_argument("--no-invert-ly", dest="invert_ly",
                         action="store_false")
    inv_grp.add_argument("--invert-rx", dest="invert_rx", action="store_true",
                         default=False)
    inv_grp.add_argument("--no-invert-rx", dest="invert_rx",
                         action="store_false")
    inv_grp.add_argument("--invert-ry", dest="invert_ry", action="store_true",
                         default=False)
    inv_grp.add_argument("--no-invert-ry", dest="invert_ry",
                         action="store_false")

    # Retargeter knobs (mirror x2_dataset_recorder defaults exactly)
    p.add_argument("--ik-damping", type=float, default=0.08)
    p.add_argument("--ik-rotation-weight", type=float, default=0.3)
    p.add_argument("--ik-per-tick-step-rad", type=float, default=0.30)
    p.add_argument("--hand-input-mode", choices=("trigger", "grip", "max"),
                   default="trigger")
    p.add_argument("--apply-curl-compensation", action="store_true")
    p.add_argument("--apply-oppose-compensation", action="store_true")
    p.add_argument("--no-finger-filter", action="store_true")

    # Sidecar
    p.add_argument(
        "--sidecar-log", type=Path, default=None,
        help="Path to write a JSONL sidecar of emitted planner_cmds",
    )

    # Camera cycler (xdotool path; TODO replace with vr_input topic)
    cam_grp = p.add_argument_group("viewer camera cycler")
    cam_grp.add_argument(
        "--no-viewer-camera-cycler", dest="enable_viewer_camera_cycler",
        action="store_false", default=True,
        help=(
            "Disable the right-stick-click -> Tab camera cycler. "
            "Useful for headless / CI runs where no GLFW window exists."
        ),
    )
    cam_grp.add_argument(
        "--viewer-window-pattern", default="MuJoCo",
        help=(
            "xdotool search --name pattern -- FALLBACK only, used when "
            "the --classname search returns nothing. Override if your "
            "deploy build sets a non-default window title."
        ),
    )
    cam_grp.add_argument(
        "--viewer-window-classname", default="MuJoCo",
        help=(
            "xdotool search --classname pattern -- the PRIMARY way to "
            "find the deploy viewer window (avoids GNOME mutter's "
            "frame-wrapper that swallows synthetic Tab events). "
            "Only override if you've rebuilt MuJoCo with a custom "
            "WM_CLASS via glfwWindowHintString."
        ),
    )

    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )

    cfg = ManagerConfig(
        quest3_ws_port=args.ws_port,
        quest3_http_port=args.http_port,
        quest3_use_ssl=not args.no_ssl,
        calibration_path=args.calibration,
        publish_rate_hz=args.rate,
        planner_cmd_host=args.planner_cmd_host,
        planner_cmd_port=args.planner_cmd_port,
        planner_cmd_topic=args.planner_cmd_topic,
        recorder_pub_host=args.recorder_pub_host,
        recorder_pub_port=args.recorder_pub_port,
        intent_stick_deadzone=args.stick_deadzone,
        intent_repeat_interval_s=args.repeat_interval,
        intent_enable_lean_fwd=args.enable_lean_fwd,
        intent_enable_torso=args.enable_torso,
        invert_lx=args.invert_lx,
        invert_ly=args.invert_ly,
        invert_rx=args.invert_rx,
        invert_ry=args.invert_ry,
        enable_viewer_camera_cycler=args.enable_viewer_camera_cycler,
        viewer_window_pattern=args.viewer_window_pattern,
        viewer_window_classname=args.viewer_window_classname,
        ik_damping=args.ik_damping,
        ik_rotation_weight=args.ik_rotation_weight,
        ik_per_tick_step_rad=args.ik_per_tick_step_rad,
        hand_input_mode=args.hand_input_mode,
        apply_curl_compensation=args.apply_curl_compensation,
        apply_oppose_compensation=args.apply_oppose_compensation,
        enable_finger_filter=not args.no_finger_filter,
        sidecar_log_path=args.sidecar_log,
        verbose=args.verbose,
    )

    mgr = Quest3ManagerX2(cfg)

    # Graceful shutdown on SIGINT / SIGTERM.
    def _sig_handler(signum, _frame):
        log.info("[manager-x2] signal %d received; stopping", signum)
        mgr._stop.set()

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    mgr.start()
    n_ticks = mgr.run()
    log.info("[manager-x2] shutdown complete (%d ticks)", n_ticks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
