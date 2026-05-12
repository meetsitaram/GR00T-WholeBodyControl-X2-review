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

# How often to re-print the dead-deploy warning when proprio stays
# silent. The first warning fires immediately on the alive->stale
# transition; subsequent re-warnings are gated to once per
# ``_DEPLOY_SILENT_REWARN_S`` seconds so a long crash doesn't spam.
_DEPLOY_SILENT_REWARN_S: float = 30.0


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

    # ── Curl / opposition stretch (opt-in binarisation) ───────────────
    # Default live path is affine normalisation only -- the operator's
    # raw Quest 3 curls get linearly mapped through their per-operator
    # ``(floor[i], ceiling[i])`` calibration window so deliberate full-
    # squeeze hits the OmniHand CLOSED anchor and rest hits OPEN. That
    # preserves smooth intermediate gestures (half-grasp, soft pinch)
    # but on a tight power-grasp pick-and-place the operator usually
    # squeezes only ~70-85 % of their calibrated max which lands
    # commanded q at ~70-85 % of the way to CLOSED -- the OmniHand
    # fingers don't quite wrap the cube. Setting these knobs to True
    # threads ``apply_curl_compensation`` / ``apply_oppose_compensation``
    # into ``per_finger_grasp_command_from_curls_and_oppose`` so the
    # per-finger / per-thumb-oppose stretch curve (defined in
    # :mod:`gear_sonic.utils.teleop.x2_hand_retarget`) pushes mid-range
    # curls toward CLOSED and saturates touch-onset oppose more
    # aggressively. The ``finger_tip_oppose`` proximity drive (May 11
    # milestone) still fires on actual fingertip touches independently
    # of these two flags.
    #
    # When you flip these on you trade smooth-intermediate fidelity for
    # reliable full closure. The docs reference (with calibration
    # rationale): docs/source/tutorials/x2_dataset_record_and_replay.md
    # § "Why we abandoned the global power-curve compensation" and the
    # follow-up "opt-in tool" paragraph immediately below it.
    apply_curl_compensation: bool = False
    apply_oppose_compensation: bool = False

    # SONIC corrective-delta observability (v1 schema).
    # ``sonic_correction_warn_rad`` is the threshold over which the
    # operator log fires once per second; ``log_sonic_correction``
    # toggles the print itself. The ``action.sonic_correction_max_rad``
    # column is always populated regardless.
    sonic_correction_warn_rad: float = 0.05
    log_sonic_correction: bool = True

    # ── Phase-1 robocasa scene plumbing (G1 architecture) ──────────────
    # When ``scene_xml_path`` is set, the recorder:
    #   (a) loads the static MJCF for ego-view rendering so frames show
    #       the table + cube + bowl the deploy bridge is simulating;
    #   (b) instantiates a :class:`RobocasaTaskMirror` so per-tick
    #       success / reward / subtask signals can be appended to each
    #       LeRobot frame as ``task.success`` / ``task.reward`` /
    #       ``task.subtask_<name>`` columns;
    #   (c) opens a SUB on the bridge's ``scene_state`` topic to receive
    #       cube / bowl freejoint qpos at state-rate;
    #   (d) opens a PUB on the bridge's ``scene_reset`` topic so episode
    #       starts can push fresh per-episode object poses sampled by
    #       the matching robocasa env's placement_initializer.
    #
    # All four behaviours are silent no-ops on a "no scene" recording so
    # existing flat-floor data collection keeps working unchanged.
    scene_xml_path: Optional[Path] = None
    """Path to the static scene MJCF (built by
    ``gear_sonic/scripts/build_x2_robocasa_scene_xml.py``). When None
    the recorder runs in legacy flat-floor mode."""

    robocasa_env: Optional[str] = None
    """Robocasa env name (e.g. ``"X2PickPlaceCube"``). Falls back to the
    metadata sidecar's ``env_name`` field when not set explicitly."""

    scene_state_sub_host: str = "localhost"
    scene_state_sub_port: int = 5559
    scene_reset_pub_host: str = "*"
    scene_reset_pub_port: int = 5560
    """Defaults match
    :data:`gear_sonic.utils.teleop.zmq.scene_state_zmq.SCENE_STATE_DEFAULT_PUB_PORT`
    and the bridge's ``--scene-state-pub-port`` / ``--scene-reset-sub-port``
    defaults. Override on both sides if you're running multiple bridges
    on the same host."""

    episode_seed: Optional[int] = None
    """Optional RNG seed for the mirror's per-episode reset. ``None``
    delegates to numpy's global RNG (default behaviour). Set this for
    reproducible test recordings."""

    # Misc
    verbose: bool = True

    # Operator-log cadence. Status print and SONIC-override print share
    # this period (wall-clock seconds). Lower = chattier; raise to keep
    # the terminal quiet during long recording sessions. The deploy-
    # silence warning fires *independently* of this throttle (within
    # ~``DEPLOY_ALIVE_STALE_THRESHOLD_S`` of the bridge going quiet),
    # so cranking this up does not delay failure visibility.
    status_log_period_s: float = 5.0


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

        # Quest 3 reader. ``quiet_periodic=True`` suppresses the per-100-msg
        # ``[Quest3Reader] msgs=N fps=X idle`` heartbeat (otherwise once-per-
        # second flood that drowns out the throttled recorder/deploy logs).
        # First-packet snapshot and one-shot XR / source-changed events
        # still fire.
        self._quest = Quest3Reader(
            ws_port=cfg.quest3_ws_port,
            http_port=cfg.quest3_http_port,
            use_ssl=cfg.quest3_use_ssl,
            quiet_periodic=True,
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

        # ── Robocasa scene plumbing (G1) ───────────────────────────────
        # Mirror is constructed eagerly here (its underlying robosuite
        # env is built lazily on the first reset(), so this is cheap).
        # The scene_state SUB thread + scene_reset PUB are armed only
        # when a scene XML was supplied.
        self._task_mirror = None
        self._scene_state_thread: Optional[threading.Thread] = None
        self._scene_reset_pub_sock = None
        self._latest_scene_state = None
        self._scene_state_lock = threading.Lock()
        self._scene_metadata: Optional[dict[str, Any]] = None
        if cfg.scene_xml_path is not None:
            scene_xml = Path(cfg.scene_xml_path)
            if not scene_xml.is_file():
                raise FileNotFoundError(
                    f"scene_xml_path does not exist: {scene_xml}"
                )
            meta_path = scene_xml.with_suffix(".json")
            if not meta_path.is_file():
                raise FileNotFoundError(
                    f"scene metadata sidecar missing: {meta_path}. "
                    f"Build it via gear_sonic/scripts/build_x2_robocasa_scene_xml.py."
                )
            import json as _json
            self._scene_metadata = _json.loads(meta_path.read_text())
            from gear_sonic.utils.teleop.robocasa_task_mirror import (
                RobocasaTaskMirror,
            )
            print(
                f"[recorder] robocasa scene mode: env="
                f"{cfg.robocasa_env or self._scene_metadata.get('env_name')!r} "
                f"xml={scene_xml}",
                flush=True,
            )
            self._task_mirror = RobocasaTaskMirror(
                scene_xml_path=scene_xml,
                scene_metadata=self._scene_metadata,
                env_name=cfg.robocasa_env,
            )
            # Override the operator-supplied task string with the env's
            # canonical instruction so the LeRobot ``task`` column
            # matches what the deploy is actually presenting.
            if not cfg.task:
                self._cfg.task = self._task_mirror.task_string
            elif cfg.task != self._task_mirror.task_string:
                print(
                    f"[recorder] NOTE: cfg.task={cfg.task!r} differs from "
                    f"scene metadata task_string="
                    f"{self._task_mirror.task_string!r}; using cfg.task.",
                    flush=True,
                )

            # scene_reset PUB (recorder -> bridge).
            self._scene_reset_pub_sock = ctx.socket(zmq.PUB)
            self._scene_reset_pub_sock.setsockopt(zmq.LINGER, 0)
            self._scene_reset_pub_sock.bind(
                f"tcp://{cfg.scene_reset_pub_host}:{cfg.scene_reset_pub_port}"
            )
            print(
                f"[recorder] scene_reset PUB bound at "
                f"tcp://{cfg.scene_reset_pub_host}:{cfg.scene_reset_pub_port}",
                flush=True,
            )

            # scene_state SUB (bridge -> recorder, runs in its own thread
            # so the main loop never blocks on ZMQ recv).
            self._scene_state_thread = threading.Thread(
                target=self._scene_state_subscriber,
                name="scene-state-sub",
                daemon=True,
            )

            # Extend the LeRobot features schema with the per-frame
            # task.* columns the recorder will produce. Must be done
            # before :meth:`_ensure_exporter` runs (i.e. before the
            # first ``_start_episode``), otherwise the exporter's
            # ``add_frame`` validator rejects the unknown keys.
            self._features["task.success"] = {
                "dtype": "int32",
                "shape": (1,),
                "names": ["success"],
            }
            self._features["task.reward"] = {
                "dtype": "float32",
                "shape": (1,),
                "names": ["reward"],
            }
            # Pre-register every subtask signal advertised by the mirror's
            # oracle so the per-episode schema is stable. Prefer the
            # mirror's static name list (introspectable without a synced
            # state) to keep startup deterministic; fall back to a live
            # ``subtask_signals()`` call when the env doesn't expose the
            # static list (older mirrors).
            sig_names: tuple[str, ...]
            try:
                sig_names = tuple(self._task_mirror.static_subtask_names)
            except AttributeError:
                sig_names = ()
            if not sig_names:
                try:
                    sig_names = tuple(self._task_mirror.subtask_signals().keys())
                except Exception as exc:
                    sig_names = ()
                    print(
                        f"[recorder] WARN: could not introspect mirror "
                        f"subtask signals ({exc}); subtask columns will be "
                        f"dropped from frames.",
                        flush=True,
                    )
            for sig_name in sig_names:
                self._features[f"task.subtask_{sig_name}"] = {
                    "dtype": "int32",
                    "shape": (1,),
                    "names": [sig_name],
                }
            print(
                f"[recorder] task.subtask_* columns registered: "
                f"{[f'task.subtask_{n}' for n in sig_names]}",
                flush=True,
            )

        # Exporter (created on first episode start to avoid empty
        # dataset directories when the operator never records anything).
        self._exporter: Optional[Gr00tDataExporter] = None
        self._episode_buffer = _EpisodeBuffer()
        self._is_recording = False
        self._episode_count = 0

        # Button-edge tracking for the Quest 3 controller buttons.
        self._prev_buttons = (False, False, False, False)

        # SONIC corrective-delta logging state. ``_last_correction_log_t``
        # throttles the operator log; cadence is ``cfg.status_log_period_s``.
        self._last_correction_log_t: float = 0.0
        self._frame_correction_max_seen: float = 0.0
        self._frame_correction_max_idx: int = -1

        # Periodic status print throttle (wall clock; same cadence as
        # the SONIC log). Initialised to 0.0 so the first status fires
        # immediately on the first tick rather than after one period.
        self._last_status_log_t: float = 0.0

        # Dead-deploy detector. Once we have *ever* seen the deploy
        # alive (``_deploy_was_alive=True``) and the latest snapshot
        # reports stale, we print a one-shot red warning so operators
        # don't sit and stare at a dead viewer thinking the recording
        # is healthy. Re-warns are rate-limited (see
        # ``_DEPLOY_SILENT_REWARN_S``) so a long crash doesn't spam
        # the console either.
        self._deploy_was_alive: bool = False
        self._last_deploy_silent_warn_t: float = 0.0

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
        # Robocasa scene_state subscriber (no-op when not in scene mode).
        if self._scene_state_thread is not None:
            self._scene_state_thread.start()
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
        if self._scene_state_thread is not None:
            try:
                self._scene_state_thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            self._pub_sock.close(linger=0)
        except Exception:
            pass
        if self._scene_reset_pub_sock is not None:
            try:
                self._scene_reset_pub_sock.close(linger=0)
            except Exception:
                pass
        if self._task_mirror is not None:
            try:
                self._task_mirror.close()
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
        self._print_startup_banner()
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
                        apply_curl_compensation=self._cfg.apply_curl_compensation,
                        apply_oppose_compensation=self._cfg.apply_oppose_compensation,
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
                        apply_curl_compensation=self._cfg.apply_curl_compensation,
                        apply_oppose_compensation=self._cfg.apply_oppose_compensation,
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

                # Two independent operator-visibility hooks:
                #   1. dead-deploy warning (one-shot, fires within
                #      ``DEPLOY_ALIVE_STALE_THRESHOLD_S`` of the bridge
                #      going silent regardless of status throttle),
                #   2. periodic status print (wall-clock cadence so it
                #      doesn't drift with control rate).
                now_log = time.monotonic()
                self._maybe_warn_deploy_silent(now=now_log)
                if (
                    self._cfg.verbose
                    and (now_log - self._last_status_log_t)
                        >= self._cfg.status_log_period_s
                ):
                    self._last_status_log_t = now_log
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

        # Robocasa: re-randomise scene objects via the task mirror's
        # placement_initializer, then push the new poses to the deploy.
        if self._task_mirror is not None and self._scene_reset_pub_sock is not None:
            from gear_sonic.utils.teleop.zmq.scene_state_zmq import (
                serialize_reset_objects,
            )
            try:
                payload = self._task_mirror.reset(
                    seed=self._cfg.episode_seed
                    if self._cfg.episode_seed is None
                    else int(self._cfg.episode_seed) + self._episode_count
                )
                self._scene_reset_pub_sock.send(serialize_reset_objects(payload))
                print(
                    f"[recorder] [B] scene_reset sent: "
                    f"freejoints={list(payload.object_freejoint_qpos)} "
                    f"welded={list(payload.mutable_body_pos)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[recorder] [B] WARNING: scene_reset failed: {exc}",
                    flush=True,
                )

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
        # Robocasa mode: append per-frame success / reward / subtask
        # signals from the task mirror. The mirror's mj_data has been
        # synced from the deploy bridge's most recent ``scene_state``
        # publish (see :meth:`_drain_scene_state_into_mirror`), so these
        # match the deploy's view of the scene at this tick. We ALWAYS
        # populate the task.* columns when the mirror exists -- even
        # on mirror eval failure -- because the LeRobot exporter's
        # ``validate_frame`` is strict about schema completeness and
        # would reject frames missing keys we registered upfront.
        if self._task_mirror is not None:
            success_val = 0
            reward_val = 0.0
            subtask_vals: dict[str, int] = {}
            try:
                self._drain_scene_state_into_mirror()
                success_val = int(bool(self._task_mirror.check_success()))
                reward_val = float(self._task_mirror.compute_reward())
                subtask_vals = {
                    name: int(v)
                    for name, v in self._task_mirror.subtask_signals().items()
                }
            except Exception as exc:
                if self._cfg.verbose:
                    print(
                        f"[recorder] task mirror eval failed: {exc}; "
                        "writing zero labels for this frame.",
                        flush=True,
                    )
            frame_data["task.success"] = np.array(
                [success_val], dtype=np.int32
            )
            frame_data["task.reward"] = np.array(
                [reward_val], dtype=np.float32
            )
            for name in [
                k.removeprefix("task.subtask_")
                for k in self._features
                if k.startswith("task.subtask_")
            ]:
                frame_data[f"task.subtask_{name}"] = np.array(
                    [subtask_vals.get(name, 0)], dtype=np.int32
                )
        self._episode_buffer.push(frame_data)

    # -- helpers --------------------------------------------------------------

    def _scene_state_subscriber(self) -> None:
        """Background SUB on the bridge's scene_state topic.

        Stores the most recent :class:`SceneState` under
        :attr:`_scene_state_lock` so the main loop can read it without
        racing the ZMQ recv. The bridge sends at state-rate (default
        200 Hz) so we use ``CONFLATE=1`` to avoid building up a queue.
        """
        try:
            from gear_sonic.utils.teleop.zmq.scene_state_zmq import (
                parse_scene_state, SCENE_STATE_TOPIC,
            )
        except ImportError as exc:
            print(f"[recorder] scene_state SUB import failed: {exc}",
                  flush=True)
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.SUBSCRIBE, SCENE_STATE_TOPIC.encode())
        sock.setsockopt(zmq.RCVTIMEO, 200)
        endpoint = (
            f"tcp://{self._cfg.scene_state_sub_host}:"
            f"{self._cfg.scene_state_sub_port}"
        )
        try:
            sock.connect(endpoint)
        except Exception as exc:
            print(f"[recorder] scene_state SUB connect to {endpoint} "
                  f"failed: {exc}", flush=True)
            sock.close(linger=0)
            return
        print(f"[recorder] scene_state SUB connected at {endpoint}",
              flush=True)
        first = True
        while not self._stop_event.is_set():
            try:
                raw = sock.recv()
            except zmq.error.Again:
                continue
            except zmq.error.ContextTerminated:
                break
            except Exception as exc:
                print(f"[recorder] scene_state SUB recv error: {exc}",
                      flush=True)
                continue
            try:
                state = parse_scene_state(raw)
            except ValueError as exc:
                print(f"[recorder] scene_state decode error: {exc}",
                      flush=True)
                continue
            with self._scene_state_lock:
                self._latest_scene_state = state
            if first:
                first = False
                print(
                    f"[recorder] scene_state SUB: first message "
                    f"(t={state.sim_time:.3f} freejoints="
                    f"{list(state.object_freejoint_qpos)})",
                    flush=True,
                )
        try:
            sock.close(linger=0)
        except Exception:
            pass

    def _drain_scene_state_into_mirror(self) -> None:
        """Pull the latest scene_state snapshot and apply it to the mirror.

        Idempotent: calling repeatedly with the same latest state writes
        the same qpos/body_pos values into the mirror, which is fine
        (mj_forward is cheap and the geometry doesn't change). Returns
        immediately if no scene_state has been received yet.
        """
        if self._task_mirror is None:
            return
        with self._scene_state_lock:
            state = self._latest_scene_state
        if state is None:
            return
        self._task_mirror.sync_from_state(state)

    def _build_renderer(self) -> None:
        if self._renderer is not None:
            return
        from gear_sonic.scripts.render_smoketest_episode_video import (
            MujocoFrameRenderer,
        )
        # In robocasa mode we want the renderer to load the SAME static
        # scene XML the deploy bridge sees so the recorded ego_view
        # frames show the table + cube + bowl. Otherwise fall back to
        # the legacy compose-only flat-floor model.
        scene_xml_path = self._cfg.scene_xml_path
        print("[recorder] building MuJoCo ego renderer …", flush=True)
        self._renderer = MujocoFrameRenderer(
            camera="ego_view",
            width=self._cfg.render_width,
            height=self._cfg.render_height,
            with_omnihand=self._cfg.with_omnihand,
            egl=True,
            scene_xml_path=scene_xml_path,
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

    def _print_startup_banner(self) -> None:
        """One-shot operator cheat-sheet printed before the main loop.

        Surfaces the things you need to remember mid-session that aren't
        visible anywhere else: the Quest 3 button map, the MuJoCo viewer
        shortcuts (especially the new ``obj_left`` / ``obj_right``
        workspace cameras), and the current scene + output paths. Kept
        terse so the rest of the throttled per-5s status lines remain
        readable.
        """
        scene = self._cfg.robocasa_env or "(no robocasa env -- pinned base only)"
        out = self._cfg.output_dir if not self._cfg.teleop_only else "(teleop-only, no dataset)"
        finger_filter = "ON" if self._cfg.finger_filter_params is not None else "OFF"
        curl_comp = "ON" if self._cfg.apply_curl_compensation else "OFF (linear)"
        oppose_comp = "ON" if self._cfg.apply_oppose_compensation else "OFF (linear)"
        bar = "─" * 72
        print(
            "\n".join([
                "",
                bar,
                "  X2 dataset recorder ready",
                bar,
                f"  scene       : {scene}",
                f"  output      : {out}",
                f"  task        : {self._cfg.task!r}",
                "",
                "  Finger control:",
                f"    smoothing filter (v0.6) : {finger_filter}",
                f"    curl compensation       : {curl_comp}  "
                "(--apply-curl-compensation)",
                f"    oppose compensation     : {oppose_comp}  "
                "(--apply-oppose-compensation)",
                "    > Turn ON both compensations if fingers don't fully",
                "      close on a power-grasp pick-and-place. Default OFF",
                "      preserves smooth intermediate gestures.",
                "",
                "  Quest 3 buttons:",
                "    A   toggle arm tracking (engage / disengage IK)",
                "    B   start a new episode (re-randomises scene objects)",
                "    X   save current episode to disk",
                "    Y   discard current episode",
                "",
                "  MuJoCo viewer (the GLFW window):",
                "    Tab        cycle fixed cameras (obj_left, obj_right,",
                "               rgbd_head_front, then free orbit camera)",
                "    [ / ]      previous / next fixed camera",
                "    H or F1    show / hide the help overlay (full keymap)",
                "    Ctrl-L     toggle contact-point visualisation",
                "    Ctrl-F     toggle contact-force vectors",
                "    Mouse      orbit / pan / zoom (only when on free camera)",
                "",
                "  Status lines below are throttled to one every "
                f"{self._cfg.status_log_period_s:.0f} s.",
                bar,
                "",
            ]),
            flush=True,
        )

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

    def _maybe_warn_deploy_silent(self, *, now: float) -> None:
        """Loud one-shot warning when the deploy bridge stops publishing.

        Called every tick from the main loop. The check is cheap (a
        single ``snapshot``-style read of ``last_update_monotonic``)
        and *independent* of the ``status_log_period_s`` throttle so
        operators see the failure within ~1.5 s of the bridge dying
        regardless of how quiet the status log is.

        The most common cause is an external process killing the
        deploy's docker container (``docker kill`` from
        ``run_live_vla_demo.sh stop`` or a parallel test cleanup hook
        that filters by ``ancestor=x2sim`` -- see the comment in
        ``run_live_vla_demo.sh`` ``stop_all`` for details). When that
        happens the MuJoCo viewer vanishes and the deploy log stops
        mid-stream with no goodbye message; without this hook the
        recorder happily keeps buffering frames against a stale stand
        pose for as long as the operator stays put.

        Re-warnings are gated to once per
        :data:`_DEPLOY_SILENT_REWARN_S` so a sustained outage doesn't
        spam the terminal.
        """
        if not self._cfg.verbose:
            return
        with self._latest_state.cv:
            received_any = self._latest_state.received_any
            last_rx = self._latest_state.last_update_monotonic
        if not received_any:
            return
        # Once we've seen at least one packet, latch ``_deploy_was_alive``
        # so the warning only fires on a *transition* from alive -> stale,
        # not during the legitimate startup window.
        self._deploy_was_alive = True
        silent_for = now - last_rx
        if silent_for <= DEPLOY_ALIVE_STALE_THRESHOLD_S:
            # Healthy: clear any prior warning state so a recovery + new
            # outage will re-warn immediately.
            self._last_deploy_silent_warn_t = 0.0
            return
        # Stale. Print once on the transition, then re-warn at most
        # every ``_DEPLOY_SILENT_REWARN_S`` while it stays silent.
        if (
            self._last_deploy_silent_warn_t == 0.0
            or (now - self._last_deploy_silent_warn_t) >= _DEPLOY_SILENT_REWARN_S
        ):
            self._last_deploy_silent_warn_t = now
            # ANSI red to the terminal; harmless if redirected to a file.
            red = "\033[31m"
            reset = "\033[0m"
            print(
                f"{red}[recorder] !! deploy went silent (no x2_debug "
                f"proprio for {silent_for:.2f}s) -- the MuJoCo "
                f"viewer/container most likely died. Common cause: an "
                f"external script ran ``docker kill`` on every container "
                f"matching ``ancestor=x2sim`` (see "
                f"``run_live_vla_demo.sh`` stop_all). Stop the recorder "
                f"with X (save) or Ctrl-C (drop), then relaunch.{reset}",
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
        """Periodic print when SONIC is overriding operator commands.

        Tracks the worst arm-joint delta seen in the throttle window and
        emits one line if it exceeds ``cfg.sonic_correction_warn_rad``.
        Cadence is shared with the status print
        (``cfg.status_log_period_s``, default 5 s) so the operator
        terminal stays quiet during long sessions. Suppressed entirely
        when ``cfg.log_sonic_correction`` is False.
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
        if (now - self._last_correction_log_t) < self._cfg.status_log_period_s:
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
