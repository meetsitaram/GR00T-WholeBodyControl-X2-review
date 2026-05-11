"""Closed-loop X2 dataset recorder for VR-driven LeRobot data collection.

Architecture overview
---------------------

::

    ┌─────────────┐  raw 3-pt + buttons   ┌─────────────────────────┐
    │ Quest 3 VR  ├──────WebSocket───────►│ Quest3Reader            │
    │ headset     │                       └────────────┬────────────┘
    └─────────────┘                                    │
                                                       ▼
                                       ┌──────────────────────────────┐
                                       │ VRArmTeleop (DLS IK x2)      │
                                       │ + ControllerHandRetargeter   │
                                       └────────────┬─────────────────┘
                                                    │ 31-DOF body_q (mj order)
                                                    │ + 2 * 10 hand_q
                                                    ▼
                                       ┌──────────────────────────────┐
                  pose ZMQ PUB :5556───┤ X2DatasetRecorder            │
                                       │  • publishes joint_pos_mj +  │
                                       │    zero motion_token @ 50 Hz │
                                       │  • subscribes x2_debug :5557 │
                                       │  • renders ego_view via      │
                                       │    MujocoFrameRenderer       │
                                       │  • writes LeRobot v2.1 ds    │
                                       └────────────┬─────────────────┘
                                                    │
                                                    ▼
                                          ``Gr00tDataExporter``

There is **no SONIC FSQ encoder in the live recording loop**. The dataset
is built from the operator's *intent* (pre-SONIC ``body_q`` + hand joints)
plus the deploy's observed proprio + ego_view. If you want VLA training
labels (FSQ ``motion_token``) attached, run the offline labeler in a
post-processing pass over the recorded ``action.body_q_mj`` (post-SONIC
canonical) trajectories. Putting an FSQ encoder *inside* a "data to
train a VLA" recording loop would just bake that VLA's biases into its
own training data.

The script is meant to be co-launched with the C++ deploy in VLA-input
mode (``deploy_x2.sh sim --vla --sim-profile gantry --sim-with-omnihand``).
The deploy consumes ``joint_pos_mj`` from the wire as the SONIC tracking
policy's reference motion (the actor *ignores* ``motion_token`` today --
the field is documented as a "v1 hook for VLA-direct token streaming"
in ``zmq_pose_input_source.hpp``). The deploy's MuJoCo replica then
publishes the resulting body / hand / pelvis state back over
``x2_debug`` for ground-truth proprioception.

Recording lifecycle (Quest 3 controller buttons)
-------------------------------------------------

* ``A``  — toggle active arm tracking on / off. Stateless: idle holds
  the arms at neutral; active drives them through the calibrated
  head-relative wrist mapping (see ``operator_calibration.py``).
* ``B``  — start a fresh episode (ignored if an episode is already
  recording).
* ``X``  — stop and *save* the current episode.
* ``Y``  — stop and *discard* the current episode (deletes the
  partially-written buffers; the on-disk dataset is unchanged
  because :class:`Gr00tDataExporter` only writes a parquet shard
  on ``save_episode``).

V0 scope
--------

* **Stationary** robot (gantry profile). Lower body uses the trained
  X2 stand pose for every frame; only arms + hands are operator-driven.
* **Controllers-only** hands: trigger maps to a uniform per-side
  finger curl (open->closed motor anchors).
* **Single task string** per session, passed via ``--task``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Optional

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.data.exporter import Gr00tDataExporter  # noqa: E402
from gear_sonic.data.features_x2_vla import (  # noqa: E402
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    FPS as DATASET_FPS,
    HAND_DOF_OMNI,
    SONIC_MOTION_TOKEN_DIM,
    assemble_observation_state,
    get_features_x2_vla,
    get_modality_config_x2_vla,
    get_x2_robot_model,
)
from gear_sonic.scripts.live_vla_publish_motion_token import (  # noqa: E402
    DEFAULT_HAND_DOF,
    DEFAULT_STAND_POSE_MUJOCO_RAD,
    DEPLOY_ALIVE_STALE_THRESHOLD_S,
    MJ_TO_PIN,
    NUM_BODY_DOFS,
    _LatestState,
    _quat_wxyz_to_projected_gravity,
    _x2_debug_subscriber,
)
from gear_sonic.utils.teleop.finger_signal_filter import (
    FingerFilterParams,
    FingerSignalFilter,
)
from gear_sonic.utils.teleop.operator_calibration import OperatorCalibration
from gear_sonic.utils.teleop.vr.quest3_reader import Quest3Reader
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import VRArmTeleopCalibrated
from gear_sonic.utils.teleop.x2_hand_retarget import (
    NUM_HAND_DOF_PER_SIDE,
    controller_grasp_ratio,
    grasp_command_from_ratio,
    per_finger_grasp_command_from_curls_and_oppose,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


# ── Constants ─────────────────────────────────────────────────────────────


# Index ranges into the 31-DOF MuJoCo body_q vector (legs/waist/arms/head).
_LEFT_ARM_MJ_SLICE = slice(15, 22)   # left shoulder/elbow/wrist (7 joints)
_RIGHT_ARM_MJ_SLICE = slice(22, 29)  # right arm (7 joints)


@dataclass
class RecorderConfig:
    """Configuration for :class:`X2DatasetRecorder`."""

    output_dir: Optional[Path]
    task: str
    sonic_checkpoint: Optional[Path] = None
    teleop_only: bool = False

    # Networking
    pub_host: str = "*"
    pub_port: int = 5556
    pub_topic: str = "pose"
    sub_host: str = "localhost"
    sub_port: int = 5557
    sub_topic: str = "x2_debug"

    # Quest 3 server
    quest3_ws_port: int = 8765
    quest3_http_port: int = 8443
    quest3_use_ssl: bool = True

    # Cadence
    publish_rate_hz: float = float(DATASET_FPS)  # 50 Hz
    record_rate_hz: float = float(DATASET_FPS)
    protocol_version: int = 4

    # Render
    render_width: int = EGO_VIEW_WIDTH
    render_height: int = EGO_VIEW_HEIGHT
    with_omnihand: bool = True

    # Hand retargeting
    hand_input_mode: str = "trigger"  # trigger | grip | max

    # IK / VR teleop
    ik_damping: float = 0.08
    # v0 default 0.0 = position-only IK. Wrist orientation is not
    # calibrated yet, so feeding the raw VR quaternion in confuses the
    # solver more than it helps.
    ik_rotation_weight: float = 0.3
    ik_per_tick_step_rad: float = 0.30

    # Operator calibration (per-arm head-relative wrist mapping). One
    # of the two MUST be supplied: either a pre-captured YAML or
    # --recalibrate to capture inline before recording starts.
    calibration_path: Optional[Path] = None
    recalibrate: bool = False
    operator_id: str = "default"

    # Dataset
    embodiment_tag: str = "new_embodiment"

    # Per-finger noise / jitter / occlusion filter (v0.6).
    # ``finger_filter_params = None`` disables the filter; setting it
    # to ``FingerFilterParams()`` enables the v5-calibrated defaults.
    finger_filter_params: Optional[FingerFilterParams] = field(
        default_factory=FingerFilterParams
    )

    # SONIC corrective-delta observability (v1 schema).
    # ``sonic_correction_warn_rad`` is the threshold over which the
    # operator log fires once per second; ``log_sonic_correction``
    # toggles the print itself. The ``action.sonic_correction_max_rad``
    # column is always populated regardless.
    sonic_correction_warn_rad: float = 0.05
    log_sonic_correction: bool = True

    # Misc
    verbose: bool = True


@dataclass
class _EpisodeBuffer:
    """Per-episode in-memory storage, flushed to the exporter on save."""

    frames: list[dict[str, Any]] = field(default_factory=list)
    started_at: float = 0.0
    task: str = ""

    def push(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def __len__(self) -> int:
        return len(self.frames)

    def reset(self) -> None:
        self.frames.clear()


class X2DatasetRecorder:
    """Top-level orchestrator for the VR-driven X2 dataset recorder."""

    def __init__(self, cfg: RecorderConfig) -> None:
        self._cfg = cfg
        if cfg.teleop_only:
            self._cfg.output_dir = None
        else:
            if cfg.output_dir is None:
                raise ValueError(
                    "output_dir is required unless teleop_only=True"
                )
            self._cfg.output_dir = Path(cfg.output_dir)
            # Preflight FIRST so a half-init stub from a previous
            # crashed run gets auto-cleaned (otherwise exporter.create
            # falls into the HF-Hub resume path and 404s on
            # tmp/tmp_dataset). Don't pre-create the leaf dir; the
            # exporter does that on first write, and pre-creating would
            # itself trip the same trap.
            from gear_sonic.data.dataset_output_dir import preflight_dataset_output_dir
            preflight_dataset_output_dir(
                self._cfg.output_dir, log_prefix="recorder"
            )
            self._cfg.output_dir.parent.mkdir(parents=True, exist_ok=True)
        # The recorder drives the SONIC *tracking policy* via ``joint_pos_mj``
        # on the wire; the C++ deploy's actor consumes the reference motion,
        # not the FSQ-encoded ``motion_token``. So the tokenizer is only
        # needed if/when we add offline post-recording motion-token
        # labelling for VLA training. For live teleop it stays out of
        # the loop.
        self._cfg.sonic_checkpoint = (
            Path(cfg.sonic_checkpoint) if cfg.sonic_checkpoint is not None else None
        )

        self._stop_event = threading.Event()

        # Robot model + dataset features
        print("[recorder] loading X2 robot model + features …", flush=True)
        self._robot_model = get_x2_robot_model(hand_variant="omnihand_10")
        self._features = get_features_x2_vla(
            self._robot_model, hand_dof_per_side=HAND_DOF_OMNI
        )
        self._modality_cfg = get_modality_config_x2_vla(
            self._robot_model, hand_dof_per_side=HAND_DOF_OMNI
        )
        if self._robot_model.num_joints != NUM_BODY_DOFS:
            raise RuntimeError(
                f"unexpected body DOF count {self._robot_model.num_joints} "
                f"!= {NUM_BODY_DOFS}"
            )

        # The recorder drives the deploy via ``joint_pos_mj`` on the wire
        # (which the SONIC tracking policy consumes as reference motion).
        # ``motion_token`` is published as zeros for forward-compat with the
        # protocol-v4 schema; the C++ actor ignores it today (the field is
        # documented as a "v1 hook for VLA-direct token streaming").
        self._zero_motion_token = np.zeros(
            SONIC_MOTION_TOKEN_DIM, dtype=np.float64
        )
        print(
            "[recorder] tokenizer skipped (live loop drives SONIC via "
            "joint_pos_mj; motion_token=zeros, ignored by deploy actor)",
            flush=True,
        )
        # Calibration is loaded lazily after Quest 3 boot in case
        # --recalibrate is set (the inline capture flow needs a live
        # WebXR client). The teleop is created in start() once we have
        # the calibration in hand.
        self._calibration: Optional[OperatorCalibration] = None
        self._teleop: Optional[VRArmTeleopCalibrated] = None

        # Quest 3 reader
        self._quest = Quest3Reader(
            ws_port=cfg.quest3_ws_port,
            http_port=cfg.quest3_http_port,
            use_ssl=cfg.quest3_use_ssl,
        )

        # ZMQ
        ctx = zmq.Context.instance()
        self._pub_sock = ctx.socket(zmq.PUB)
        self._pub_sock.setsockopt(zmq.SNDHWM, 10)
        self._pub_sock.setsockopt(zmq.LINGER, 0)
        self._pub_sock.bind(f"tcp://{cfg.pub_host}:{cfg.pub_port}")

        # ``x2_debug`` SUB lives in its own thread, mutating shared state.
        self._latest_state = _LatestState()
        self._sub_thread = threading.Thread(
            target=_x2_debug_subscriber,
            kwargs=dict(
                sub_url=f"tcp://{cfg.sub_host}:{cfg.sub_port}",
                topic=cfg.sub_topic,
                state=self._latest_state,
                stop_event=self._stop_event,
                verbose=cfg.verbose,
            ),
            name="x2_debug-sub",
            daemon=True,
        )

        # MuJoCo renderer is constructed lazily on the recording thread:
        # MuJoCo's EGL backend pins to the thread that created the
        # renderer. The recording loop needs it, so we delay until that
        # loop spins up.
        self._renderer: Any | None = None

        # Exporter (created on first episode start to avoid empty
        # dataset directories when the operator never records anything).
        self._exporter: Optional[Gr00tDataExporter] = None
        self._episode_buffer = _EpisodeBuffer()
        self._is_recording = False
        self._episode_count = 0

        # Button-edge tracking for the Quest 3 controller buttons.
        self._prev_buttons = (False, False, False, False)

        # SONIC corrective-delta logging state. ``_last_correction_log_t``
        # throttles the operator log to once per second.
        self._last_correction_log_t: float = 0.0
        self._frame_correction_max_seen: float = 0.0
        self._frame_correction_max_idx: int = -1

        if cfg.teleop_only:
            print(
                f"[recorder] init done. mode=TELEOP_ONLY (no dataset writes) "
                f"pub=tcp://{cfg.pub_host}:{cfg.pub_port}",
                flush=True,
            )
        else:
            print(
                f"[recorder] init done. output_dir={self._cfg.output_dir} "
                f"task={cfg.task!r} pub=tcp://{cfg.pub_host}:{cfg.pub_port}",
                flush=True,
            )

    # -- lifecycle ------------------------------------------------------------

    def start(self) -> None:
        self._quest.start()
        self._sub_thread.start()
        # Give PUB-SUB sockets a beat to wire up before we start
        # blasting messages.
        time.sleep(0.2)
        # Calibration is loaded *after* Quest 3 boot in case the operator
        # passed --recalibrate (which needs a connected WebXR client).
        self._calibration = self._resolve_calibration()
        self._teleop = VRArmTeleopCalibrated(
            calibration=self._calibration,
            damping=self._cfg.ik_damping,
            rotation_weight=self._cfg.ik_rotation_weight,
            per_tick_step_rad=self._cfg.ik_per_tick_step_rad,
        )

        # Per-side stateful smoothers for the Quest 3 hand signals.
        # See ``finger_signal_filter`` for design + tuning rationale.
        if self._cfg.finger_filter_params is not None:
            self._cfg.finger_filter_params.validate()
            self._finger_filter_left: Optional[FingerSignalFilter] = (
                FingerSignalFilter(self._cfg.finger_filter_params)
            )
            self._finger_filter_right: Optional[FingerSignalFilter] = (
                FingerSignalFilter(self._cfg.finger_filter_params)
            )
            if self._cfg.verbose:
                p = self._cfg.finger_filter_params
                print(
                    f"[recorder] finger filter ENABLED: alpha={p.ema_alpha} "
                    f"hold_window={p.hold_window} hold_std={p.hold_std} "
                    f"release_std={p.release_std} release_disp={p.release_disp}",
                    flush=True,
                )
        else:
            self._finger_filter_left = None
            self._finger_filter_right = None
            if self._cfg.verbose:
                print(
                    "[recorder] finger filter DISABLED",
                    flush=True,
                )

    def _resolve_calibration(self) -> OperatorCalibration:
        """Load the operator calibration YAML, or capture inline."""
        cfg = self._cfg
        cal_path = cfg.calibration_path
        if cal_path is not None:
            cal_path = Path(cal_path)

        if cfg.recalibrate:
            from gear_sonic.scripts.vr_operator_calibrate import (
                _wait_for_first_packet,
                run_inline_calibration,
            )
            if cal_path is None:
                cal_path = (
                    Path(__file__).resolve().parent.parent.parent.parent
                    / "data" / "operator_calibrations"
                    / f"{cfg.operator_id}.yaml"
                )
            print(
                f"[recorder] --recalibrate set; running guided calibration "
                f"before recording. Output: {cal_path}",
                flush=True,
            )
            _wait_for_first_packet(self._quest)
            return run_inline_calibration(
                self._quest,
                output_path=cal_path,
                operator_id=cfg.operator_id,
            )

        if cal_path is None or not cal_path.is_file():
            raise SystemExit(
                f"Error: calibration file not found at {cal_path}. Either:\n"
                f"  - Run `python -m gear_sonic.scripts.vr_operator_calibrate "
                f"--operator-id {cfg.operator_id}` to capture one, OR\n"
                f"  - Pass `recalibrate=True` (or --recalibrate) to capture "
                f"it inline before recording starts."
            )
        cal = OperatorCalibration.load_yaml(cal_path)
        print(
            f"[recorder] loaded calibration {cal_path} "
            f"(operator='{cal.operator_id}', "
            f"L_residual={cal.fit['left'].residual_m*100:.1f} cm, "
            f"R_residual={cal.fit['right'].residual_m*100:.1f} cm)",
            flush=True,
        )
        return cal

    def stop(self) -> None:
        self._stop_event.set()
        # Persist any buffered episode? No -- the recorder requires an
        # explicit save command. Buffered-but-not-saved frames are
        # discarded on shutdown so an accidental Ctrl-C doesn't
        # contaminate the dataset.
        if self._is_recording:
            print(
                f"[recorder] stop: dropping {len(self._episode_buffer)} "
                "buffered frames (use button X before exiting to save).",
                flush=True,
            )
            self._episode_buffer.reset()
        try:
            self._sub_thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._pub_sock.close(linger=0)
        except Exception:
            pass
        try:
            self._quest.stop()
        except Exception:
            pass
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass

    # -- main loop ------------------------------------------------------------

    def run(self) -> int:
        """Blocking 50 Hz publish + record loop. Returns total ticks."""
        if not self._cfg.teleop_only:
            self._build_renderer()
        period = 1.0 / max(self._cfg.publish_rate_hz, 1e-6)
        next_tick = time.monotonic()
        tick = 0
        wait_msg = False

        try:
            while not self._stop_event.is_set():
                tick_start = time.monotonic()

                # Read VR + buttons.
                vr_pose = self._quest.get_3pt_pose()
                buttons = self._quest.get_buttons()
                triggers = self._quest.get_controller_inputs()

                if vr_pose is None:
                    if not wait_msg:
                        print(
                            f"[recorder] waiting for first Quest 3 packet "
                            f"(make sure the WebXR app is open at "
                            f"https://<HOST>:{self._cfg.quest3_http_port}) …",
                            flush=True,
                        )
                        wait_msg = True
                    self._publish_idle()
                    self._sleep_until(next_tick + period)
                    next_tick += period
                    continue

                self._handle_buttons(buttons, vr_pose)
                # Hand command: prefer XRHand 5-finger curls + thumb
                # opposition WHENEVER they are reported (which is on
                # every frame the headset has hand-tracking enabled,
                # including multimodal mode where the operator is also
                # holding controllers). Fall back to the controller
                # trigger/grip scalar only when XRHand is not available.
                #
                # IMPORTANT: do NOT gate on ``l_src``. In multimodal the
                # WebXR client tags ``source = "controller"`` because
                # gripSpace wins for IK pose, but the same frame still
                # carries XRHand curls + the thumb opposition signal.
                # Gating on source kind would spuriously route us to the
                # uniform-trigger path in multimodal and throw away the
                # per-finger detail AND the thumb opposition correction.
                l_curls, r_curls, l_src, r_src = self._quest.get_hand_curls()
                l_oppose, r_oppose = self._quest.get_thumb_opposition()
                l_finger_tip_oppose, r_finger_tip_oppose = (
                    self._quest.get_finger_tip_oppose()
                )

                # Apply the per-side smoothing filter on top of the raw
                # Quest 3 inputs. The retargeter sees only the filtered
                # values; the raw values are no longer kept (the SONIC-
                # record path doesn't write a debug NPZ).
                if self._finger_filter_left is not None and self._finger_filter_right is not None:
                    l_curls, l_oppose, l_finger_tip_oppose = (
                        self._finger_filter_left.update(
                            l_curls, l_oppose, l_finger_tip_oppose,
                        )
                    )
                    r_curls, r_oppose, r_finger_tip_oppose = (
                        self._finger_filter_right.update(
                            r_curls, r_oppose, r_finger_tip_oppose,
                        )
                    )

                hr = self._calibration.hand_range if self._calibration is not None else None
                l_hr = hr.left if hr is not None else None
                r_hr = hr.right if hr is not None else None
                if l_curls is not None:
                    left_hand_q = per_finger_grasp_command_from_curls_and_oppose(
                        "left", l_curls, l_oppose,
                        finger_tip_oppose=l_finger_tip_oppose,
                        curl_floor=l_hr.floor if l_hr is not None else None,
                        curl_ceiling=l_hr.ceiling if l_hr is not None else None,
                        oppose_floor=l_hr.oppose_floor if l_hr is not None else None,
                        oppose_ceiling=l_hr.oppose_ceiling if l_hr is not None else None,
                    )
                    left_ratio = float(np.mean(l_curls))
                else:
                    left_ratio, _ = controller_grasp_ratio(
                        left_trigger=triggers[0],
                        right_trigger=triggers[1],
                        left_grip=triggers[2],
                        right_grip=triggers[3],
                        mode=self._cfg.hand_input_mode,
                    )
                    left_hand_q = grasp_command_from_ratio("left", left_ratio)

                if r_curls is not None:
                    right_hand_q = per_finger_grasp_command_from_curls_and_oppose(
                        "right", r_curls, r_oppose,
                        finger_tip_oppose=r_finger_tip_oppose,
                        curl_floor=r_hr.floor if r_hr is not None else None,
                        curl_ceiling=r_hr.ceiling if r_hr is not None else None,
                        oppose_floor=r_hr.oppose_floor if r_hr is not None else None,
                        oppose_ceiling=r_hr.oppose_ceiling if r_hr is not None else None,
                    )
                    right_ratio = float(np.mean(r_curls))
                else:
                    _, right_ratio = controller_grasp_ratio(
                        left_trigger=triggers[0],
                        right_trigger=triggers[1],
                        left_grip=triggers[2],
                        right_grip=triggers[3],
                        mode=self._cfg.hand_input_mode,
                    )
                    right_hand_q = grasp_command_from_ratio("right", right_ratio)

                # Run IK; if not engaged, this returns the operator's
                # calibrated neutral q (a posture that is *not* the
                # SONIC-trained stand-pose arm slice).
                tick_result = self._teleop.step(vr_pose)
                if self._teleop.is_engaged:
                    body_q_mj = self._compose_body_q(
                        left_arm_q=tick_result.left_q,
                        right_arm_q=tick_result.right_q,
                    )
                else:
                    # IDLE -> publish the trained stand pose verbatim.
                    # Overlaying the operator's "neutral" arm pose onto
                    # DEFAULT_STAND_POSE_MUJOCO_RAD pushes the SONIC
                    # tracking policy OOD (the trained attractor and the
                    # operator's idle pose disagree by tens of degrees
                    # at every arm joint), which generates large action
                    # corrections that ripple into the lower body and
                    # eventually flip the robot. Surrender the body
                    # channel to the trained stand pose until A is
                    # pressed; finger commands above still flow so the
                    # bridge's hand position actuators don't slam to
                    # qpos=0. This is a stopgap until the X2 Heuristic
                    # Motion Planner takes over reference generation.
                    body_q_mj = np.array(
                        DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64
                    )

                # ``motion_token`` is a forward-compat wire field; the
                # current C++ deploy actor consumes only ``joint_pos_mj``
                # as reference motion. Send zeros so we don't pin a CPU
                # core spinning the SONIC encoder at 50 Hz for output
                # nobody reads.
                token = self._zero_motion_token

                # Publish to deploy.
                self._publish_pose(
                    body_q_mj=body_q_mj,
                    motion_token=token,
                    left_hand_q=left_hand_q,
                    right_hand_q=right_hand_q,
                    tick=tick,
                )

                # Record an observation/action frame from the latest
                # deploy state. The video frame is rendered now to keep
                # observation + action time-aligned.
                if self._is_recording:
                    self._record_frame(
                        commanded_body_q_mj=body_q_mj,
                        commanded_left_hand_q=left_hand_q,
                        commanded_right_hand_q=right_hand_q,
                        commanded_token=token,
                    )

                if (
                    self._cfg.verbose
                    and tick % int(max(self._cfg.publish_rate_hz, 1)) == 0
                ):
                    self._print_status(tick=tick, tick_result=tick_result)

                tick += 1
                next_tick += period
                self._sleep_until(next_tick)
        finally:
            # Auto-flush any open episode at the end if the user
            # actually pressed B but forgot X. Keeps recordings
            # recoverable.
            if self._is_recording and len(self._episode_buffer) > 0:
                print(
                    f"[recorder] auto-saving open episode with "
                    f"{len(self._episode_buffer)} frames on shutdown",
                    flush=True,
                )
                self._stop_episode(save=True)
        return tick

    # -- buttons --------------------------------------------------------------

    def _handle_buttons(
        self,
        buttons: tuple[bool, bool, bool, bool],
        vr_pose: np.ndarray,
    ) -> None:
        a, b, x, y = buttons
        prev_a, prev_b, prev_x, prev_y = self._prev_buttons
        # Edge-trigger so we only act once per press.
        if a and not prev_a:
            # Stateless: A only toggles whether IK runs. Calibration is
            # loaded once at startup; there's no per-press anchor.
            self._teleop.set_engaged(not self._teleop.is_engaged)
            state = "ACTIVE" if self._teleop.is_engaged else "IDLE"
            print(f"[recorder] [A] arm tracking -> {state}", flush=True)
        if b and not prev_b:
            self._start_episode()
        if x and not prev_x:
            self._stop_episode(save=True)
        if y and not prev_y:
            self._stop_episode(save=False)
        self._prev_buttons = buttons

    def _start_episode(self) -> None:
        if self._cfg.teleop_only:
            print(
                "[recorder] [B] ignored: --teleop-only mode (no dataset writes)",
                flush=True,
            )
            return
        if self._is_recording:
            print("[recorder] [B] ignored: already recording", flush=True)
            return
        self._ensure_exporter()
        self._episode_buffer.reset()
        self._episode_buffer.started_at = time.time()
        self._episode_buffer.task = self._cfg.task
        # Each episode starts with a clean filter buffer so the warm-up
        # window doesn't leak state from the previous episode.
        if self._finger_filter_left is not None:
            self._finger_filter_left.reset()
        if self._finger_filter_right is not None:
            self._finger_filter_right.reset()
        self._is_recording = True
        print(
            f"[recorder] [B] episode start (task={self._cfg.task!r}, "
            f"# {self._episode_count + 1})",
            flush=True,
        )

    def _stop_episode(self, *, save: bool) -> None:
        if not self._is_recording:
            kind = "X" if save else "Y"
            print(f"[recorder] [{kind}] ignored: no active episode", flush=True)
            return
        n = len(self._episode_buffer)
        if save and n > 0:
            assert self._exporter is not None
            for frame in self._episode_buffer.frames:
                self._exporter.add_frame(frame)
            self._exporter.save_episode()
            self._episode_count += 1
            # Resolve and print the on-disk paths so the operator
            # doesn't have to ``find data/lerobot/<dataset> -newer ...``
            # to locate what they just recorded. Layout matches
            # :class:`Gr00tDataExporter` v2.1: parquet under
            # ``data/chunk-000/`` and the ego-view mp4 under
            # ``videos/chunk-000/observation.images.ego_view/``.
            saved_idx = self._episode_count - 1
            out_root = self._cfg.output_dir
            parquet_path = (
                out_root / "data" / "chunk-000"
                / f"episode_{saved_idx:06d}.parquet"
            )
            mp4_path = (
                out_root / "videos" / "chunk-000"
                / "observation.images.ego_view"
                / f"episode_{saved_idx:06d}.mp4"
            )
            print(
                f"[recorder] [X] episode saved: {n} frames "
                f"(total saved={self._episode_count})",
                flush=True,
            )
            print(f"[recorder]     parquet -> {parquet_path}", flush=True)
            print(f"[recorder]     mp4     -> {mp4_path}", flush=True)
        else:
            kind = "X" if save else "Y"
            reason = "no frames" if save else "discarded by operator"
            print(f"[recorder] [{kind}] dropping {n} frames ({reason})", flush=True)
        self._episode_buffer.reset()
        self._is_recording = False

    # -- pipeline pieces ------------------------------------------------------

    def _compose_body_q(
        self,
        *,
        left_arm_q: np.ndarray,
        right_arm_q: np.ndarray,
    ) -> np.ndarray:
        """Pin the lower body / waist / head to the standing pose;
        overlay the operator-driven 7+7 arm joints."""
        body = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
        body[_LEFT_ARM_MJ_SLICE] = left_arm_q
        body[_RIGHT_ARM_MJ_SLICE] = right_arm_q
        return body

    def _publish_pose(
        self,
        *,
        body_q_mj: np.ndarray,
        motion_token: np.ndarray,
        left_hand_q: np.ndarray,
        right_hand_q: np.ndarray,
        tick: int,
    ) -> None:
        payload = {
            "joint_pos_mj": body_q_mj.astype(np.float32),
            "root_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "motion_token": motion_token.astype(np.float32),
            "left_hand_joints": left_hand_q.astype(np.float32),
            "right_hand_joints": right_hand_q.astype(np.float32),
            "frame_index": np.array([tick], dtype=np.int64),
        }
        msg = pack_pose_message(
            payload, topic=self._cfg.pub_topic, version=self._cfg.protocol_version
        )
        try:
            self._pub_sock.send(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def _publish_idle(self) -> None:
        """Emit the trained stand pose while we wait for VR to wake up.

        ``joint_pos_mj`` is the SONIC tracking policy's reference
        motion. Sending the trained stand pose (with identity root quat
        and a zero ``motion_token``) is exactly the wire format
        :mod:`gear_sonic.scripts.mock_vla_publish_stand_token` uses to
        keep the X2 standing on the gantry indefinitely.
        """
        body = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
        zero_hand = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
        self._publish_pose(
            body_q_mj=body,
            motion_token=self._zero_motion_token,
            left_hand_q=zero_hand,
            right_hand_q=zero_hand,
            tick=-1,
        )

    def _record_frame(
        self,
        *,
        commanded_body_q_mj: np.ndarray,
        commanded_left_hand_q: np.ndarray,
        commanded_right_hand_q: np.ndarray,
        commanded_token: np.ndarray,
    ) -> None:
        """Capture an aligned (observation, action) tuple for the buffer.

        observation.* fields read from the live deploy (``x2_debug``)
        when available; if the deploy hasn't published yet we fall
        back to the commanded body_q so the dataset never has gaps.
        """
        (
            obs_body_q_mj,
            obs_base_quat_wxyz,
            obs_left_hand_q,
            obs_right_hand_q,
            _revision,
            alive,
        ) = self._latest_state.snapshot()
        if not alive:
            obs_body_q_mj = commanded_body_q_mj.copy()
            obs_base_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            obs_left_hand_q = commanded_left_hand_q.copy()
            obs_right_hand_q = commanded_right_hand_q.copy()

        # The observation.state vector is built in Pinocchio joint order
        # (matches the trained Gr00t modality). MJ_TO_PIN[i] = j means
        # pinocchio's i-th joint is mujoco's j-th, so we permute the
        # MuJoCo body_q the same way the live VLA bridge does.
        body_q_pin = obs_body_q_mj[list(MJ_TO_PIN)]
        observation_state = assemble_observation_state(
            self._robot_model,
            body_q_pin,
            obs_left_hand_q,
            obs_right_hand_q,
        )
        proj_grav = _quat_wxyz_to_projected_gravity(obs_base_quat_wxyz)

        # Render the same ego_view the policy will see at deploy time.
        # We render against the *observed* body_q so the recorded
        # dataset's image and proprio tracks the actual robot, not the
        # commanded target. (Otherwise the frames would diverge from
        # ground truth whenever the tracking policy lags.)
        try:
            ego_view = self._renderer.render_frame(  # type: ignore[union-attr]
                body_q=obs_body_q_mj,
                left_active=obs_left_hand_q.astype(np.float64),
                right_active=obs_right_hand_q.astype(np.float64),
                root_quat_wxyz=obs_base_quat_wxyz,
            )
        except Exception as exc:
            print(f"[recorder] render warn (frame skipped): {exc}", flush=True)
            return
        ego_view = np.ascontiguousarray(ego_view, dtype=np.uint8)
        if ego_view.shape != (
            self._cfg.render_height, self._cfg.render_width, 3,
        ):
            print(
                f"[recorder] WARN: ego_view shape {ego_view.shape} != "
                f"({self._cfg.render_height}, {self._cfg.render_width}, 3); "
                "skipping frame",
                flush=True,
            )
            return

        # v1 schema: bare-canonical action columns carry the post-SONIC
        # executed q (what the trained tracking policy achieved, i.e.
        # what the MuJoCo viewer shows). The pre-SONIC operator command
        # is preserved as ``_pre_sonic`` siblings for retargeter /
        # SONIC-correction analysis -- training-invisible.
        commanded_body_q_arr = np.asarray(commanded_body_q_mj, dtype=np.float64)
        commanded_left_hand_arr = np.asarray(commanded_left_hand_q, dtype=np.float64)
        commanded_right_hand_arr = np.asarray(commanded_right_hand_q, dtype=np.float64)
        executed_body_q_arr = np.asarray(obs_body_q_mj, dtype=np.float64)
        executed_left_hand_arr = np.asarray(obs_left_hand_q, dtype=np.float64)
        executed_right_hand_arr = np.asarray(obs_right_hand_q, dtype=np.float64)

        # Scalar summary of how much SONIC pushed back this frame:
        # max |executed - commanded| over the 14 arm joints. The lower
        # body is pinned to the stand pose so its delta is uninteresting
        # noise; head / waist similarly. Live deploy already sets
        # alive=False during the warm-up window, in which case
        # executed == commanded so the delta is zero.
        arm_delta_max = max(
            float(
                np.abs(
                    executed_body_q_arr[_LEFT_ARM_MJ_SLICE]
                    - commanded_body_q_arr[_LEFT_ARM_MJ_SLICE]
                ).max()
            ),
            float(
                np.abs(
                    executed_body_q_arr[_RIGHT_ARM_MJ_SLICE]
                    - commanded_body_q_arr[_RIGHT_ARM_MJ_SLICE]
                ).max()
            ),
        )
        self._maybe_log_sonic_correction(
            executed_body_q=executed_body_q_arr,
            commanded_body_q=commanded_body_q_arr,
            arm_delta_max=arm_delta_max,
        )

        frame_data = {
            "observation.state": observation_state,
            "observation.projected_gravity": proj_grav,
            "action.motion_token": commanded_token.astype(np.float64),
            "action.body_q_mj": executed_body_q_arr,
            "action.left_hand_joints": executed_left_hand_arr,
            "action.right_hand_joints": executed_right_hand_arr,
            "action.body_q_mj_pre_sonic": commanded_body_q_arr,
            "action.left_hand_joints_pre_sonic": commanded_left_hand_arr,
            "action.right_hand_joints_pre_sonic": commanded_right_hand_arr,
            "action.sonic_correction_max_rad": np.array(
                [arm_delta_max], dtype=np.float32
            ),
            "observation.images.ego_view": ego_view,
            "task": self._episode_buffer.task,
        }
        self._episode_buffer.push(frame_data)

    # -- helpers --------------------------------------------------------------

    def _build_renderer(self) -> None:
        if self._renderer is not None:
            return
        from gear_sonic.scripts.render_smoketest_episode_video import (
            MujocoFrameRenderer,
        )
        print("[recorder] building MuJoCo ego renderer …", flush=True)
        self._renderer = MujocoFrameRenderer(
            camera="ego_view",
            width=self._cfg.render_width,
            height=self._cfg.render_height,
            with_omnihand=self._cfg.with_omnihand,
            egl=True,
        )
        print(
            f"[recorder] renderer ready ({self._renderer.width}x"
            f"{self._renderer.height}, omnihand={self._renderer.with_omnihand})",
            flush=True,
        )

    def _ensure_exporter(self) -> None:
        if self._exporter is not None:
            return
        self._exporter = Gr00tDataExporter.create(
            save_root=self._cfg.output_dir,
            fps=int(self._cfg.publish_rate_hz),
            features=self._features,
            modality_config=self._modality_cfg,
            task=self._cfg.task,
            script_config={
                "robot_type": "agibot_x2_ultra",
                "embodiment_tag": self._cfg.embodiment_tag,
                "hand_variant": "omnihand_10",
                "num_body_joints": self._robot_model.num_joints,
                "hand_dof_per_side": HAND_DOF_OMNI,
                "fps": int(self._cfg.publish_rate_hz),
                "source": "quest3_record_x2_dataset",
                "sonic_checkpoint_path": str(self._cfg.sonic_checkpoint),
                "sim_profile": "gantry",
            },
            robot_type="agibot_x2_ultra",
        )
        # v1 format marker: distinguishes post-SONIC canonical action
        # columns (this recorder) from kinematic-only datasets and from
        # legacy v0 datasets (which only had ``action.commanded_body_q_mj``).
        # Tools downstream of the parquet should dispatch on this file.
        import json
        version_path = self._exporter.root / "meta" / "dataset_format_version.json"
        version_path.write_text(json.dumps({
            "version": 1,
            "post_sonic_canonical": True,
            "writer": "X2DatasetRecorder",
            "sim_profile": "gantry",
        }, indent=2) + "\n")
        print(f"[recorder] exporter ready -> {self._cfg.output_dir}", flush=True)

    def _print_status(self, *, tick: int, tick_result: Any) -> None:
        (
            _, _, _, _, _, alive
        ) = self._latest_state.snapshot()
        l_err = tick_result.left_ik.pos_err_m
        r_err = tick_result.right_ik.pos_err_m
        rec = "REC" if self._is_recording else "idle"
        eng = "ON" if self._teleop.is_engaged else "off"
        n = len(self._episode_buffer)
        print(
            f"[recorder] tick={tick:6d} engaged={eng} state={rec} "
            f"buffered={n} deploy_alive={alive} "
            f"|ik_err|=({l_err:.3f}, {r_err:.3f}) m",
            flush=True,
        )

    @staticmethod
    def _sleep_until(target_monotonic: float) -> None:
        slack = target_monotonic - time.monotonic()
        if slack > 0:
            time.sleep(slack)

    def _maybe_log_sonic_correction(
        self,
        *,
        executed_body_q: np.ndarray,
        commanded_body_q: np.ndarray,
        arm_delta_max: float,
    ) -> None:
        """Once-per-second print when SONIC is overriding operator commands.

        Tracks the worst arm-joint delta seen in the last second and
        emits one line if it exceeds ``cfg.sonic_correction_warn_rad``.
        Suppressed entirely when ``cfg.log_sonic_correction`` is False.
        """
        if not self._cfg.log_sonic_correction or not self._cfg.verbose:
            return
        if arm_delta_max > self._frame_correction_max_seen:
            # Re-derive the offending joint index across the full body
            # vector so the operator gets a useful joint name.
            full_delta = np.abs(executed_body_q - commanded_body_q)
            full_delta[: _LEFT_ARM_MJ_SLICE.start] = 0.0
            full_delta[_RIGHT_ARM_MJ_SLICE.stop :] = 0.0
            self._frame_correction_max_seen = arm_delta_max
            self._frame_correction_max_idx = int(np.argmax(full_delta))
        now = time.monotonic()
        if now - self._last_correction_log_t < 1.0:
            return
        self._last_correction_log_t = now
        if self._frame_correction_max_seen >= self._cfg.sonic_correction_warn_rad:
            from gear_sonic.data.features_x2_vla import MUJOCO_JOINT_NAMES
            idx = self._frame_correction_max_idx
            joint_name = (
                MUJOCO_JOINT_NAMES[idx]
                if 0 <= idx < len(MUJOCO_JOINT_NAMES)
                else f"joint_{idx}"
            )
            print(
                f"[recorder] SONIC override |Δq|max={self._frame_correction_max_seen:.3f}"
                f" rad ({np.rad2deg(self._frame_correction_max_seen):.1f}°) at "
                f"{joint_name}",
                flush=True,
            )
        # Reset the rolling-second peak.
        self._frame_correction_max_seen = 0.0
        self._frame_correction_max_idx = -1


__all__ = [
    "RecorderConfig",
    "X2DatasetRecorder",
]
