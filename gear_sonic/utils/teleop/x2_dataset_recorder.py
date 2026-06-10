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
                                       │    motion_token @ 50 Hz      │
                                       │    (zeros for planner path;  │
                                       │     passthrough when present │
                                       │     on body_pose, e.g. VLA)  │
                                       │  • subscribes x2_debug :5557 │
                                       │  • renders ego_view via      │
                                       │    MujocoFrameRenderer       │
                                       │  • writes LeRobot v2.1 ds    │
                                       └────────────┬─────────────────┘
                                                    │
                                                    ▼
                                          ``Gr00tDataExporter``

**Inline SONIC FSQ encoder for VLA-training-ready labels.** When the
operator passes ``--sonic-checkpoint`` (the wrapper auto-resolves the
``.pt`` sibling of the deploy ONNX), the recorder loads
:class:`~gear_sonic.utils.teleop.online_sonic_tokenizer.OnlineSonicTokenizer`
once at startup and encodes every commanded ``body_q_mj`` into a 64-D
FSQ token in-process. The token lands as ``action.motion_token`` in the
parquet so the dataset is VLA-trainable directly off disk -- no
post-recording label step. Without ``--sonic-checkpoint`` the recorder
emits zero tokens and a single one-shot warning so the operator knows
the dataset is kinematic-only (intentional for smoke tests, never
correct for production data collection).

This is the right shape because the SONIC tracking encoder is **not**
the VLA -- it's the canonical decoder-side reference that every VLA on
top of SONIC must target. Labeling with the SONIC encoder gives the VLA
a consistent ground-truth signal; using anything else (raw body_q,
identity tokens) means the VLA has no consistent target to optimize
against.

The wire's ``motion_token`` field is normally zeros in the planner-driven
stack (the heuristic planner does not populate it). When a
``body_pose`` publisher includes a 64-D ``motion_token`` (the live VLA
bridge in recorder-first layout), the subscribe-mode loop forwards that
tensor verbatim on the deploy ``pose`` topic so closed-loop SONIC+VLA
matches the direct-bridge contract. The inline tokenizer remains
dataset-only for ``action.motion_token`` labels.

The script is meant to be co-launched with the C++ deploy in VLA-input
mode (``deploy_x2.sh sim --vla --sim-profile gantry --sim-with-omnihand``).
The deploy consumes ``joint_pos_mj`` from the wire as the SONIC tracking
policy's reference motion; ``motion_token`` is the VLA / SONIC latent
hook documented in ``zmq_pose_input_source.hpp`` (non-zero on VLA runs).
The deploy's MuJoCo replica then publishes the resulting body / hand /
pelvis state back over ``x2_debug`` for ground-truth proprioception.

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
import json
import math
import queue
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
from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw
from gear_sonic.utils.teleop.finger_signal_filter import (
    FingerFilterParams,
    FingerSignalFilter,
)
from gear_sonic.utils.teleop.gesture_session import (
    GESTURE_CMD_DEFAULT_PORT,
    GESTURE_CMD_DEFAULT_TOPIC,
    GestureCatalogEntry,
    GesturePlayRequest,
    GestureSession,
    GestureStopRequest,
    load_catalog as load_gesture_catalog,
    parse_gesture_command,
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

# Prior ``Quest3Reader.get_hand_curls`` ``*_source`` for filter reset
# (distinct from ``None`` meaning unknown on the wire).
_PREV_Q3_HAND_SRC_UNSET: Any = object()


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
    """SONIC tracker ``.pt`` checkpoint. When provided, the recorder runs
    :class:`~gear_sonic.utils.teleop.online_sonic_tokenizer.OnlineSonicTokenizer`
    per tick to fill ``action.motion_token`` in the LeRobot dataset
    directly. Required for VLA training; if ``None`` the recorder emits
    zero tokens and warns once at startup."""

    sonic_tokenizer_device: str = "cuda:0"
    """Torch device for the inline tokenizer. ``cuda:0`` keeps per-tick
    cost under 100 us; ``cpu`` adds ~1 ms per tick (still well within
    the 20 ms budget at 50 Hz). Use ``cpu`` if cuda:0 is contended by
    the deploy / VLA on the same GPU."""

    sonic_encoder_config: Optional[Path] = None
    """Path to the YAML encoder-observation config (parses to
    :class:`~gear_sonic.utils.teleop.x2_encoder_obs_builder.X2EncoderConfig`).
    When set *and* the recorder is in subscribe mode (planner-driven),
    the inline tokenizer builds the same 680-D 10-frame future
    observation the deploy actor consumes from the wire and encodes
    it via the YAML-selected encoder mode. When ``None``, or when the
    recorder is in direct mode (Quest-driven, no planner), the
    tokenizer falls back to the legacy freeze-pose path (one body_q
    tiled 11 times) and prints a one-shot deprecation warning. The
    canonical config lives at
    ``gear_sonic/data/encoder/x2_observation_config.yaml`` and the
    wrapper :file:`run_x2_quest3_planner_stack.sh` forwards it
    automatically."""

    obs_dump_recorder_path: Optional[Path] = None
    """Layer 3 byte-parity probe. When set, the subscribe-mode loop
    writes a ``.pt`` snapshot of the first tick whose planner future
    window is fully populated (snap dict + 680-D builder obs) to this
    path and continues running normally. Pair with the deploy's
    ``--obs-dump`` to assert byte-equal observations between the
    recorder's Python gather path and the deploy's C++
    ``ZmqPoseInputSource`` -- diffed by
    :file:`gear_sonic_deploy/scripts/compare_recorder_vs_deploy_obs.py`.
    Has no effect outside subscribe mode."""

    teleop_only: bool = False

    ready_file: Optional[Path] = None
    """When set, the subscribe-mode loop touches this file the moment
    the first ``body_pose`` arrives (i.e. the recorder is fully
    initialised AND actively ingesting). The VLA bridge's
    ``--wait-for-ready-file`` gate watches this path and only starts
    inference once it appears, so the recording captures the arm
    rise from idle without missing the first ~8 s of warm-up. Has no
    effect outside VLA subscribe mode."""

    idle_publish_enabled: bool = True
    """When True (default), the subscribe-mode loop calls
    :meth:`X2DatasetRecorder._publish_idle` every tick that no
    ``body_pose`` has arrived yet, so the C++ deploy SUB at
    :attr:`pub_port` always sees a stand-pose reference on the wire.
    Set to False (via ``--no-idle-publish``) to keep the recorder
    completely silent on the ``pose`` topic until a real ``body_pose``
    arrives -- the deploy will then never decode a frame and falls back
    to ``ZmqPoseInputSource``'s built-in ``default_angles`` prefill.
    Use that combination together with the bridge's ``--silent-wire``
    to validate the deploy's true 'no upstream' behaviour under
    ``--vla-no-policy``: the goal is ``has_body_reference_=False`` for
    the entire run, ``wrist_bypass_ticks=0``, and the robot held
    upright by the deploy's own reference cache."""

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

    # ── Phase 0 subscribe-only mode (planner-driven recorder) ──────────
    # When ``body_pose_source`` AND ``arm_targets_source`` are both
    # ``"zmq"``, the recorder is Quest-unaware: it skips the
    # :class:`Quest3Reader` + :class:`VRArmTeleopCalibrated` stack
    # entirely and instead subscribes to:
    #   - the planner's ``body_pose`` topic (31-DOF reference)
    #   - the manager's ``arm_targets`` topic (14-DOF arm IK)
    #   - the manager's ``hand_finger_cmd`` topic (10-DOF/side OmniHand)
    #   - the manager's ``stream_mode`` + ``recorder_cmd`` topics
    # It then merges body_pose + arm_targets into a 31-DOF ``final_pose``
    # and publishes that to the deploy on the existing
    # ``--pub-port`` (5556) under topic ``pose``. Episode control
    # buttons (start / save / discard) are forwarded from the manager
    # over ``recorder_cmd``.
    body_pose_source: str = "internal"
    """One of:

    * ``"internal"`` -- legacy: run Quest 3 + IK in this process.
    * ``"zmq"`` -- Phase 0: subscribe to the planner's ``body_pose``
      stream (default port 5565) and the manager's ``arm_targets`` /
      ``hand_finger_cmd`` streams (port 5564). Episode boundaries
      come from the manager's ``recorder_cmd`` topic.
    * ``"vla"`` -- subscribe to the VLA bridge's unified ``pose``
      stream on port 5556. The bridge emits a superset of the
      planner payload (body_q + hands + token + future window in
      one message), so the recorder skips the manager SUB entirely
      and auto-starts a single episode on first body_pose. Pair
      with ``--with-record / --output-dir / --task`` so the
      one-run = one-episode auto-save lands the dataset. The
      recorder also skips its pose-PUB bind in this mode (the
      bridge already owns :5556)."""

    arm_targets_source: str = "internal"
    """Mirror of :attr:`body_pose_source`. Allowed values are
    ``"internal"`` / ``"zmq"`` / ``"vla"``. Must equal
    :attr:`body_pose_source` -- mixing internal IK with external
    body_pose, or splitting body and hands across different sources,
    is intentionally rejected."""

    body_pose_sub_host: str = "localhost"
    body_pose_sub_port: int = 5565
    body_pose_sub_topic: str = "body_pose"

    arm_and_hands_sub_host: str = "localhost"
    arm_and_hands_sub_port: int = 5564
    arm_targets_topic: str = "arm_targets"
    hand_finger_cmd_topic: str = "hand_finger_cmd"
    stream_mode_topic: str = "stream_mode"
    recorder_cmd_topic: str = "recorder_cmd"

    # ── Live gesture playback (PKL takeover during subscribe mode) ─────
    # When ``gesture_catalog_path`` is set (or even when it isn't --
    # ad-hoc ``--pkl`` payloads still work), the recorder opens a SUB
    # on ``gesture_cmd_*`` for play / stop commands. While a gesture
    # is active inside :meth:`_run_subscribe_mode` the kplanner
    # body_pose + manager arm/hand merge are bypassed and the PKL
    # frames are emitted on the ``pose`` topic verbatim. On natural
    # completion or an explicit ``stop`` the recorder snaps back to
    # forwarding kplanner frames (no blend; see ``GestureSession``
    # docstring for the design rationale).
    #
    # Wire-level details + JSON payload shapes live in
    # :mod:`gear_sonic.utils.teleop.gesture_session`. Setting
    # ``gesture_catalog_path = None`` disables gesture support entirely
    # (no SUB bound). Only meaningful in ``body_pose_source=='zmq'``;
    # ignored in legacy internal-Quest mode.
    gesture_cmd_host: str = "*"
    """Interface for the gesture_cmd SUB ``bind``. Defaults to ``*``
    (all interfaces) because trigger scripts are transient: the
    recorder is the stable side, so it binds and ``play_gesture``
    connects. Mirrors the asymmetry already used for ``scene_reset``
    (recorder PUB ``bind`` vs bridge SUB ``connect``)."""
    gesture_cmd_port: int = GESTURE_CMD_DEFAULT_PORT
    gesture_cmd_topic: str = GESTURE_CMD_DEFAULT_TOPIC
    gesture_catalog_path: Optional[Path] = None
    gesture_future_dt_s: float = 0.1
    """Spacing of the strictly-future window the gesture player passes
    to the C++ deploy tokenizer. Matches the kplanner default (see
    :func:`gear_sonic.utils.planner.state_machine.build_pose_payload`)."""
    gesture_future_window_frames: int = 9
    """Number of strictly-future frames packed into the deploy wire's
    ``joint_pos_mj_future`` / ``root_quat_xyzw_future`` arrays during
    gesture playback. Matches the kplanner's
    ``NUM_FUTURE_FRAMES - 1 == 9`` convention."""

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

    record_front_cam: bool = False
    """When True, the recorder builds a second :class:`MujocoFrameRenderer`
    bound to the scene XML's ``front_cam`` (a wide-angle world-fixed
    witness camera baked in by ``_WORKSPACE_CAMERAS`` -- see
    ``gear_sonic/scripts/build_x2_robocasa_scene_xml.py``) and writes
    its frames to ``observation.images.front_cam`` alongside
    ``observation.images.ego_view``. Also flips ``include_front_cam=True``
    on :func:`get_features_x2_vla` / :func:`get_modality_config_x2_vla`
    so the LeRobot v2.1 schema declares the second video track up
    front; mismatches trip the exporter's per-frame validator. The
    ``record_x2_dataset.py`` CLI defaults this to True iff
    ``--scene-xml-path`` resolves (i.e. ``--robocasa-env != none``),
    since the camera only exists in robocasa-built scenes; pass
    ``--no-front-cam`` to opt out (e.g. for legacy single-camera
    parquets). Has no effect in ``--teleop-only`` mode (no recorder ->
    no schema)."""

    # ── PC2 physical head-cameras (Orbbec + IMX900 stereo pair) ────────
    # When ``record_head_cameras`` is True the recorder opens a ZMQ
    # ``SUB`` (via :class:`gear_sonic.camera.composed_camera.ComposedCameraClientSensor`)
    # to the PC2 camera bridge launched by
    # ``gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py`` and
    # writes the three real head streams into the LeRobot dataset
    # alongside the synthetic MuJoCo ``ego_view``. Feature keys:
    # ``observation.images.head_front`` (Orbbec RGB),
    # ``observation.images.stereo_left`` / ``stereo_right`` (IMX900s).
    # The flag also flips ``include_head_cameras=True`` on
    # :func:`get_features_x2_vla` / :func:`get_modality_config_x2_vla`
    # so the schema declares the three new video tracks up front --
    # mismatches trip the exporter's strict per-frame validator.
    # No-op in ``--teleop-only`` mode.
    record_head_cameras: bool = False
    """Whether to ingest the three PC2 head-camera ZMQ streams into the
    LeRobot dataset (``head_front`` Orbbec + ``stereo_left/right``
    IMX900s). The bridge MUST be running before the recorder is
    started; the recorder fails fast (within
    :attr:`camera_warmup_timeout_s`) if the first complete frame
    bundle does not arrive."""

    camera_host: str = "10.0.1.41"
    """PC2 hostname/IP where ``x2_pc2_camera_zmq_publisher.py`` is bound.
    Default matches the LAN-wired Jetson Orin NX address used by
    every other deploy script (PC1 is .40, PC2 is .41, PC3 is .42)."""

    camera_port: int = 5555
    """ZMQ PUB port for the camera bridge. Default 5555 matches the
    ``composed_camera`` convention; override on both sides if you're
    running multiple bridges on the same robot."""

    camera_warmup_timeout_s: float = 8.0
    """Seconds to wait at startup for the first fully-populated camera
    frame bundle (all of ``head_front`` / ``stereo_left`` /
    ``stereo_right`` present). Exceeding this is a fail-fast at
    recorder boot so we never write partial-schema parquet shards.
    Set to 0 to skip the wait (recorder will still error out on the
    first frame if the bundle is incomplete)."""

    camera_max_staleness_s: float = 0.5
    """If the most recent camera frame is older than this many seconds
    when the recorder tries to write a tick, the frame is skipped (no
    parquet row, no warning spam). Larger values mean the dataset
    will reuse stale frames during transient bridge hiccups; smaller
    values mean tighter freshness at the cost of episode length.
    Default 500 ms tolerates short HAL restarts without dropping
    whole episodes."""

    robot_pose_sub_host: str = "localhost"
    robot_pose_sub_port: int = 5570
    robot_pose_sub_topic: str = "robot_pose"
    """Bridge -> recorder pelvis-pose telemetry (see
    :mod:`gear_sonic.utils.teleop.zmq.robot_pose_zmq`). Defaults match
    ``x2_mujoco_ros_bridge.py --robot-pose-pub-port`` (5570). The
    recorder uses the freshest pelvis ``(x, y, z)`` from this topic to
    drive ``MujocoFrameRenderer.render_frame(root_pos_xyz=…)`` so a
    fixed witness camera (``front_cam``) actually sees the robot
    translate when it walks; the head-mounted ``ego_view`` is
    insensitive to root translation but using the live value here keeps
    both renderers consistent. Falls back to the renderer's hardcoded
    ``(0, 0, 0.793)`` when no ``robot_pose`` packet has arrived yet
    (e.g. before the bridge has written its first tick)."""

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


class _SubscribeModeState:
    """Latest snapshot of the manager + planner ZMQ topics.

    Threadsafe: a single background thread writes via :meth:`update_*`,
    the recorder loop reads via :meth:`snapshot`. All state behind
    one lock to keep the contract obvious; payloads are small (<1 KB)
    so contention is negligible.

    Used only in Phase 0 subscribe mode.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._body_pose_q_mj: Optional[np.ndarray] = None
        # v5 wire-format fields from the planner's body_pose payload.
        # ``root_quat_xyzw`` is the current-frame root orientation
        # reference; the ``*_future`` arrays carry the strictly-future
        # window the C++ deploy's tokenizer needs to anticipate the
        # next 0.9 s of motion (without it the deploy returns the
        # latest single frame for all 10 future slots, which is what
        # makes the policy "step in place" -- legs animate but the
        # body never gets a forward-thrust ref).
        self._root_quat_xyzw: Optional[np.ndarray] = None
        # Wire-format world-frame pelvis position (post-2026-06 publisher).
        # Optional; older planners omit these fields. When present the
        # recorder forwards them on the merged ``pose`` stream so the
        # kinematic viewer (and downstream PKL recorder in Phase 2) can
        # reconstruct full ``qpos[0:3]`` instead of pelvis-pinning at the
        # origin. The C++ deploy ignores these keys (consumes joints +
        # root_quat only), so plumbing them through is wire-safe.
        self._root_xy_world: Optional[np.ndarray] = None
        self._root_z_world: Optional[float] = None
        self._joint_pos_mj_future: Optional[np.ndarray] = None
        self._root_quat_xyzw_future: Optional[np.ndarray] = None
        self._joint_vel_mj_future: Optional[np.ndarray] = None
        self._frame_index_future: Optional[np.ndarray] = None
        self._future_dt_s: Optional[float] = None
        self._arm_left_q: Optional[np.ndarray] = None
        self._arm_right_q: Optional[np.ndarray] = None
        self._arm_engaged: bool = False
        self._left_hand_q: Optional[np.ndarray] = None
        self._right_hand_q: Optional[np.ndarray] = None
        self._stream_mode: str = "OFF"
        self._pending_recorder_cmd: list[tuple[str, int]] = []
        self._last_body_pose_t: float = 0.0
        self._last_arm_targets_t: float = 0.0
        # Optional 64-D wire token from body_pose (VLA bridge). None when
        # absent or invalid — subscribe loop then publishes zeros on pose.
        self._wire_motion_token: Optional[np.ndarray] = None

    def update_body_pose(
        self,
        q_mj: np.ndarray,
        *,
        root_quat_xyzw: Optional[np.ndarray] = None,
        root_xy_world: Optional[np.ndarray] = None,
        root_z_world: Optional[float] = None,
        joint_pos_mj_future: Optional[np.ndarray] = None,
        root_quat_xyzw_future: Optional[np.ndarray] = None,
        joint_vel_mj_future: Optional[np.ndarray] = None,
        frame_index_future: Optional[np.ndarray] = None,
        future_dt_s: Optional[float] = None,
        wire_motion_token: Optional[np.ndarray] = None,
    ) -> None:
        """Atomically refresh the planner's body_pose snapshot.

        ``q_mj`` is required (single-frame fallback always available).
        The v5 fields are optional: when present they enable the C++
        deploy's future-window tokenizer; when absent the deploy falls
        back to the legacy single-frame Sample() path. The whole set
        is updated under the lock so a reader never sees a current
        frame from tick N with a future window from tick N-1.
        """
        with self._lock:
            self._body_pose_q_mj = q_mj.copy()
            self._root_quat_xyzw = (
                None if root_quat_xyzw is None else root_quat_xyzw.copy()
            )
            self._root_xy_world = (
                None if root_xy_world is None else root_xy_world.copy()
            )
            self._root_z_world = (
                None if root_z_world is None else float(root_z_world)
            )
            self._joint_pos_mj_future = (
                None if joint_pos_mj_future is None
                else joint_pos_mj_future.copy()
            )
            self._root_quat_xyzw_future = (
                None if root_quat_xyzw_future is None
                else root_quat_xyzw_future.copy()
            )
            self._joint_vel_mj_future = (
                None if joint_vel_mj_future is None
                else joint_vel_mj_future.copy()
            )
            self._frame_index_future = (
                None if frame_index_future is None
                else frame_index_future.copy()
            )
            self._future_dt_s = (
                None if future_dt_s is None else float(future_dt_s)
            )
            self._last_body_pose_t = time.time()
            if wire_motion_token is None:
                self._wire_motion_token = None
            else:
                mt = np.asarray(wire_motion_token, dtype=np.float32).reshape(-1)
                if mt.shape == (SONIC_MOTION_TOKEN_DIM,):
                    self._wire_motion_token = mt.copy()
                else:
                    self._wire_motion_token = None

    def update_arm_targets(
        self,
        left: np.ndarray,
        right: np.ndarray,
        engaged: bool,
        passthrough_arm_targets: bool = False,
    ) -> None:
        """Update cached arm IK targets from the manager.

        ``passthrough_arm_targets=True`` is the
        ``LOCO_DECOUPLED_ARMS=0`` sentinel: the manager is signalling
        that for this message it has no arm override and the recorder
        should let the planner-predicted arms (from ``body_pose``)
        flow through the merge step unmodified. We implement that by
        nulling the cached arm pose so the existing validity gate in
        the merge loop (``if left_arm_valid: body_q_mj[slice] = ...``)
        skips the override. The next non-passthrough message
        repopulates the cache normally, so this gate is per-message
        and not sticky -- if the operator toggles modes mid-walk the
        recorder immediately recovers.

        ``passthrough_arm_targets`` is a kwarg with a False default so
        legacy callers (and tests that mock this method) keep working
        without rewrites; older managers that don't include the wire
        field also fall through to the legacy real-arms-override path.
        """
        with self._lock:
            if passthrough_arm_targets:
                self._arm_left_q = None
                self._arm_right_q = None
                self._arm_engaged = False
            else:
                self._arm_left_q = left.copy()
                self._arm_right_q = right.copy()
                self._arm_engaged = bool(engaged)
            self._last_arm_targets_t = time.time()

    def update_hand_finger_cmd(
        self, left: np.ndarray, right: np.ndarray,
    ) -> None:
        with self._lock:
            self._left_hand_q = left.copy()
            self._right_hand_q = right.copy()

    def update_stream_mode(self, mode: str) -> None:
        with self._lock:
            self._stream_mode = mode

    def push_recorder_cmd(self, action: str, tick: int) -> None:
        with self._lock:
            self._pending_recorder_cmd.append((action, tick))

    def drain_recorder_cmds(self) -> list[tuple[str, int]]:
        with self._lock:
            out, self._pending_recorder_cmd = self._pending_recorder_cmd, []
        return out

    def snapshot(self) -> dict:
        """Return a frozen view of the latest state for one tick."""
        with self._lock:
            return {
                "body_pose_q_mj": (
                    None if self._body_pose_q_mj is None
                    else self._body_pose_q_mj.copy()
                ),
                "root_quat_xyzw": (
                    None if self._root_quat_xyzw is None
                    else self._root_quat_xyzw.copy()
                ),
                "root_xy_world": (
                    None if self._root_xy_world is None
                    else self._root_xy_world.copy()
                ),
                "root_z_world": self._root_z_world,
                "joint_pos_mj_future": (
                    None if self._joint_pos_mj_future is None
                    else self._joint_pos_mj_future.copy()
                ),
                "root_quat_xyzw_future": (
                    None if self._root_quat_xyzw_future is None
                    else self._root_quat_xyzw_future.copy()
                ),
                "joint_vel_mj_future": (
                    None if self._joint_vel_mj_future is None
                    else self._joint_vel_mj_future.copy()
                ),
                "frame_index_future": (
                    None if self._frame_index_future is None
                    else self._frame_index_future.copy()
                ),
                "future_dt_s": self._future_dt_s,
                "arm_left_q": (
                    None if self._arm_left_q is None else self._arm_left_q.copy()
                ),
                "arm_right_q": (
                    None if self._arm_right_q is None else self._arm_right_q.copy()
                ),
                "arm_engaged": self._arm_engaged,
                "left_hand_q": (
                    None if self._left_hand_q is None else self._left_hand_q.copy()
                ),
                "right_hand_q": (
                    None if self._right_hand_q is None else self._right_hand_q.copy()
                ),
                "stream_mode": self._stream_mode,
                "last_body_pose_t": self._last_body_pose_t,
                "last_arm_targets_t": self._last_arm_targets_t,
                "wire_motion_token": (
                    None if self._wire_motion_token is None
                    else self._wire_motion_token.copy()
                ),
            }


def _subscribe_mode_thread(
    *,
    body_pose_url: str,
    body_pose_topic: str,
    arm_and_hands_url: str,
    arm_targets_topic: str,
    hand_finger_cmd_topic: str,
    stream_mode_topic: str,
    recorder_cmd_topic: str,
    state: _SubscribeModeState,
    stop_event: threading.Event,
    verbose: bool = False,
    vla_mode: bool = False,
) -> None:
    """Single SUB thread that fans the upstream pose streams into ``state``.

    Two flavours:

    * Planner-driven (``vla_mode=False``): subscribes to all five topics
      on the two PUB sockets the planner + manager bind
      (``body_pose`` on the planner @ :5565, plus ``arm_targets`` /
      ``hand_finger_cmd`` / ``stream_mode`` / ``recorder_cmd`` on the
      manager @ :5564).
    * VLA-driven (``vla_mode=True``): subscribes only to the bridge's
      unified ``pose`` topic on :5556. Arms and hands are extracted
      from the same payload by :func:`_handle_body_pose_msg`; the
      manager URL / topic args are accepted for signature
      compatibility but never connected. Episode boundaries are NOT
      driven by ``recorder_cmd`` in this mode -- the bridge has no
      operator console -- so the recorder loop auto-starts a single
      episode on first body_pose and auto-saves on shutdown.

    Polls with a 50 ms RCVTIMEO so shutdown is responsive.
    """
    try:
        import msgpack
    except ImportError:
        msgpack = None  # type: ignore[assignment]

    ctx = zmq.Context.instance()
    sub_planner = ctx.socket(zmq.SUB)
    sub_planner.setsockopt(zmq.LINGER, 0)
    sub_planner.setsockopt(zmq.RCVTIMEO, 50)
    sub_planner.setsockopt_string(zmq.SUBSCRIBE, body_pose_topic)
    sub_planner.connect(body_pose_url)

    sub_mgr: Optional[zmq.Socket] = None
    if not vla_mode:
        sub_mgr = ctx.socket(zmq.SUB)
        sub_mgr.setsockopt(zmq.LINGER, 0)
        sub_mgr.setsockopt(zmq.RCVTIMEO, 50)
        for topic in (
            arm_targets_topic, hand_finger_cmd_topic,
            stream_mode_topic, recorder_cmd_topic,
        ):
            sub_mgr.setsockopt_string(zmq.SUBSCRIBE, topic)
        sub_mgr.connect(arm_and_hands_url)

    if verbose:
        if vla_mode:
            print(
                f"[recorder] VLA subscribe-mode SUBs:\n"
                f"  bridge    {body_pose_url} topic={body_pose_topic!r} "
                f"(arms + hands embedded in payload; manager SUB skipped)",
                flush=True,
            )
        else:
            print(
                f"[recorder] subscribe-mode SUBs:\n"
                f"  planner   {body_pose_url} topic={body_pose_topic!r}\n"
                f"  manager   {arm_and_hands_url} topics="
                f"{[arm_targets_topic, hand_finger_cmd_topic, stream_mode_topic, recorder_cmd_topic]}",
                flush=True,
            )

    poller = zmq.Poller()
    poller.register(sub_planner, zmq.POLLIN)
    if sub_mgr is not None:
        poller.register(sub_mgr, zmq.POLLIN)

    try:
        while not stop_event.is_set():
            events = dict(poller.poll(timeout=50))

            if sub_planner in events:
                try:
                    parts = sub_planner.recv_multipart(flags=zmq.NOBLOCK)
                    _handle_body_pose_msg(
                        parts, state, expected_topic=body_pose_topic,
                        vla_mode=vla_mode,
                    )
                except zmq.error.Again:
                    pass

            if sub_mgr is not None and sub_mgr in events:
                try:
                    parts = sub_mgr.recv_multipart(flags=zmq.NOBLOCK)
                    _handle_arm_and_hands_msg(
                        parts, state,
                        arm_targets_topic=arm_targets_topic,
                        hand_finger_cmd_topic=hand_finger_cmd_topic,
                        stream_mode_topic=stream_mode_topic,
                        recorder_cmd_topic=recorder_cmd_topic,
                    )
                except zmq.error.Again:
                    pass
    finally:
        try:
            sub_planner.close(linger=0)
        except Exception:
            pass
        if sub_mgr is not None:
            try:
                sub_mgr.close(linger=0)
            except Exception:
                pass


def _handle_body_pose_msg(
    parts: list[bytes], state: _SubscribeModeState,
    *, expected_topic: str, vla_mode: bool = False,
) -> None:
    """Decode the planner's body_pose payload and update state.

    The planner uses :func:`pack_pose_message` (single-frame topic +
    1280-byte JSON header + binary payload) so we decode with the
    matching :func:`unpack_message` from the packed-message decoder.

    The decoder exposes every named array in the wire payload; we
    forward the v5 future-window fields verbatim so the recorder's
    publish path can pass them on to the C++ deploy. Without this,
    the deploy's tokenizer falls back to its single-frame path and
    the policy gets a frozen future window (10 copies of the current
    pose), which is exactly the "legs animate but body doesn't
    translate" symptom we hit in Phase 0 smoke testing.

    ``vla_mode`` gates the embedded-hand-joints update (see lines
    988-1008 below). In VLA mode the bridge is the sole producer of
    hand state and stamps :func:`left_hand_joints` / :func:`right_
    hand_joints` into the unified ``pose`` payload, so we MUST
    forward them onto the state slot the manager would normally
    write. In teleop mode the planner ALSO publishes the same
    fields but always-zero (legacy wire-format compat); without
    the flag the always-zero update would race the manager's
    ``hand_finger_cmd`` writes and silently win at 50 Hz, which
    is exactly the 2026-06-10 follow-up 5b "fingers not
    responding" symptom -- recorder log shows ``hand|L|=0.000
    (manager)`` even though the manager log shows ``published
    hand_q|L|=3.519`` for the same tick window.
    """
    from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (
        unpack_message,
    )
    if not parts:
        return
    # Planner sends a SINGLE-PART message ([topic][header][binary]),
    # not multipart. ZMQ subscribers still wrap it in a multipart list
    # when called via recv_multipart, so parts[0] is the whole frame.
    try:
        decoded = unpack_message(parts[0], expected_topic=expected_topic)
    except ValueError:
        return
    fields = decoded.fields
    if "joint_pos_mj" not in fields:
        return
    q = np.asarray(fields["joint_pos_mj"], dtype=np.float64).reshape(-1)
    if q.shape != (NUM_BODY_DOFS,):
        return

    # Optional fields. We accept any frame that has the required
    # ``joint_pos_mj``; the v5 future window is opt-in (older planners
    # only emit single-frame fields).
    root_quat = None
    if "root_quat_xyzw" in fields:
        rq = np.asarray(fields["root_quat_xyzw"], dtype=np.float32).reshape(-1)
        if rq.shape == (4,):
            root_quat = rq

    # World-frame pelvis XY/Z (post-2026-06 publisher; see
    # build_pose_payload in state_machine.py). The kinematic viewer and
    # the Phase 2 PKL recorder both need these to reconstruct full
    # qpos[0:3]; older publishers omit the keys and the merged stream
    # then falls back to pelvis-pinned at origin (the previous behaviour).
    root_xy_world_arr: Optional[np.ndarray] = None
    if "root_xy_world" in fields:
        rxy = np.asarray(fields["root_xy_world"], dtype=np.float32).reshape(-1)
        if rxy.shape == (2,):
            root_xy_world_arr = rxy
    root_z_world_val: Optional[float] = None
    if "root_z_world" in fields:
        rz = np.asarray(fields["root_z_world"], dtype=np.float32).reshape(-1)
        if rz.shape == (1,):
            root_z_world_val = float(rz[0])

    wire_mt: Optional[np.ndarray] = None
    if "motion_token" in fields:
        mt = np.asarray(fields["motion_token"], dtype=np.float32).reshape(-1)
        if mt.shape == (SONIC_MOTION_TOKEN_DIM,):
            wire_mt = mt

    # VLA bridge: left/right hand joints embedded in the unified pose
    # payload (built by ``_build_vla_decoded_pose_payload`` in
    # live_vla_publish_motion_token). The teleop planner-driven
    # pipeline keeps these slots zero on ``body_pose`` and publishes
    # hand commands on a separate ``hand_finger_cmd`` topic via the
    # manager; for VLA recording the bridge is the sole producer so
    # we forward them onto the same state slot the manager would
    # normally write. We only update when BOTH are present + correctly
    # shaped so a partial / mid-rollover frame can't corrupt a side.
    #
    # 2026-06-10 follow-up 5b: GATED on ``vla_mode`` because in
    # teleop mode the planner also stamps both fields (as zeros, for
    # legacy wire-format compatibility) at 50 Hz on every body_pose
    # tick. The race-condition winner against the manager's 50 Hz
    # hand_finger_cmd writes was the planner's always-zero update,
    # so the recorder silently dropped the operator's finger
    # commands and the OmniHand never saw a non-zero target. The
    # ``vla_mode`` flag short-circuits the embedded-hand-joints
    # update in teleop mode so the manager's hand_finger_cmd writes
    # are the sole source of truth for finger state then.
    if (
        vla_mode
        and "left_hand_joints" in fields
        and "right_hand_joints" in fields
    ):
        lh = np.asarray(
            fields["left_hand_joints"], dtype=np.float64,
        ).reshape(-1)
        rh = np.asarray(
            fields["right_hand_joints"], dtype=np.float64,
        ).reshape(-1)
        if (
            lh.shape == (NUM_HAND_DOF_PER_SIDE,)
            and rh.shape == (NUM_HAND_DOF_PER_SIDE,)
        ):
            state.update_hand_finger_cmd(lh, rh)

    # Future-window fields are only forwarded when ALL the required
    # parts are present and self-consistent. A partial window would
    # confuse the C++ deploy (it requires both jpos+rotation futures
    # to promote into ``has_future_window_``).
    jpos_future = fields.get("joint_pos_mj_future")
    rot_future = fields.get("root_quat_xyzw_future")
    jvel_future = fields.get("joint_vel_mj_future")
    fidx_future = fields.get("frame_index_future")
    fdt_field = fields.get("future_dt_s")

    have_full_window = (
        jpos_future is not None
        and rot_future is not None
        and jpos_future.ndim == 2
        and jpos_future.shape[1] == NUM_BODY_DOFS
        and rot_future.ndim == 2
        and rot_future.shape == (jpos_future.shape[0], 4)
    )

    if have_full_window:
        jpos_future_arr = np.asarray(jpos_future, dtype=np.float32)
        rot_future_arr = np.asarray(rot_future, dtype=np.float32)
        jvel_future_arr = (
            None if jvel_future is None
            else np.asarray(jvel_future, dtype=np.float32)
        )
        if (
            jvel_future_arr is not None
            and jvel_future_arr.shape != jpos_future_arr.shape
        ):
            jvel_future_arr = None
        fidx_future_arr = (
            None if fidx_future is None
            else np.asarray(fidx_future, dtype=np.int64).reshape(-1)
        )
        if (
            fidx_future_arr is not None
            and fidx_future_arr.shape != (jpos_future_arr.shape[0],)
        ):
            fidx_future_arr = None
        future_dt_val: Optional[float] = None
        if fdt_field is not None:
            try:
                future_dt_val = float(np.asarray(fdt_field).reshape(-1)[0])
            except (IndexError, ValueError):
                future_dt_val = None
        state.update_body_pose(
            q,
            root_quat_xyzw=root_quat,
            root_xy_world=root_xy_world_arr,
            root_z_world=root_z_world_val,
            joint_pos_mj_future=jpos_future_arr,
            root_quat_xyzw_future=rot_future_arr,
            joint_vel_mj_future=jvel_future_arr,
            frame_index_future=fidx_future_arr,
            future_dt_s=future_dt_val,
            wire_motion_token=wire_mt,
        )
    else:
        state.update_body_pose(
            q,
            root_quat_xyzw=root_quat,
            root_xy_world=root_xy_world_arr,
            root_z_world=root_z_world_val,
            wire_motion_token=wire_mt,
        )


def _handle_arm_and_hands_msg(
    parts: list[bytes],
    state: _SubscribeModeState,
    *,
    arm_targets_topic: str,
    hand_finger_cmd_topic: str,
    stream_mode_topic: str,
    recorder_cmd_topic: str,
) -> None:
    if len(parts) < 2:
        return
    topic = parts[0].decode("ascii", errors="replace")
    payload_bytes = parts[1]

    import msgpack
    if topic == arm_targets_topic:
        msg = msgpack.unpackb(payload_bytes, raw=False)
        l = np.asarray(msg["left_q_rad"], dtype=np.float64)
        r = np.asarray(msg["right_q_rad"], dtype=np.float64)
        # ``passthrough_arm_targets`` (added 2026-05-30) is the
        # LOCO_DECOUPLED_ARMS=0 sentinel from the manager: when True
        # the recorder treats this message as "no arm override" and
        # nulls its cache so the merge falls through to planner arms.
        # Missing key defaults to False for wire-format back-compat
        # with older manager builds.
        state.update_arm_targets(
            l,
            r,
            bool(msg.get("is_engaged", False)),
            passthrough_arm_targets=bool(
                msg.get("passthrough_arm_targets", False)
            ),
        )
    elif topic == hand_finger_cmd_topic:
        msg = msgpack.unpackb(payload_bytes, raw=False)
        l = np.asarray(msg["left_hand_q"], dtype=np.float64)
        r = np.asarray(msg["right_hand_q"], dtype=np.float64)
        state.update_hand_finger_cmd(l, r)
    elif topic == stream_mode_topic:
        msg = msgpack.unpackb(payload_bytes, raw=False)
        state.update_stream_mode(str(msg.get("mode", "OFF")))
    elif topic == recorder_cmd_topic:
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
            state.push_recorder_cmd(
                str(payload["action"]), int(payload.get("tick", -1))
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            return


def _subscribe_gesture_cmd_thread(
    *,
    url: str,
    topic: str,
    request_queue: "queue.Queue[Any]",
    stop_event: threading.Event,
    verbose: bool = False,
) -> None:
    """Dedicated SUB on the ``gesture_cmd`` topic.

    Decodes each JSON play / stop payload into a
    :class:`GesturePlayRequest` or :class:`GestureStopRequest` and
    pushes it onto ``request_queue`` for the recorder publish loop to
    drain on its next tick. Malformed payloads are logged + dropped
    so an external trigger script can never tear down the recorder.

    Wire shape is multipart ``[topic_bytes, json_payload_bytes]`` to
    match the manager's existing ``recorder_cmd`` convention.
    """
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVTIMEO, 100)
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    # SUB binds (not connects) because the trigger script is the
    # transient side. See ``RecorderConfig.gesture_cmd_host`` doc.
    sub.bind(url)
    if verbose:
        print(
            f"[recorder] gesture_cmd SUB bind: {url} topic={topic!r}",
            flush=True,
        )

    try:
        while not stop_event.is_set():
            try:
                parts = sub.recv_multipart()
            except zmq.error.Again:
                continue
            if len(parts) < 2:
                if verbose:
                    print(
                        f"[recorder] gesture_cmd: dropping single-part "
                        f"message (expected [topic, json])",
                        flush=True,
                    )
                continue
            try:
                payload = json.loads(parts[1].decode("utf-8"))
                req = parse_gesture_command(payload)
            except (
                json.JSONDecodeError,
                UnicodeDecodeError,
                ValueError,
            ) as exc:
                print(
                    f"[recorder] gesture_cmd: ignoring malformed "
                    f"payload: {exc}",
                    flush=True,
                )
                continue
            request_queue.put(req)
    finally:
        try:
            sub.close(linger=0)
        except Exception:
            pass


class X2DatasetRecorder:
    """Top-level orchestrator for the VR-driven X2 dataset recorder."""

    def __init__(self, cfg: RecorderConfig) -> None:
        # Validate subscribe-mode coherence early.
        _allowed_sources = ("internal", "zmq", "vla")
        if cfg.body_pose_source not in _allowed_sources:
            raise ValueError(
                f"body_pose_source must be one of {_allowed_sources}; "
                f"got {cfg.body_pose_source!r}"
            )
        if cfg.arm_targets_source not in _allowed_sources:
            raise ValueError(
                f"arm_targets_source must be one of {_allowed_sources}; "
                f"got {cfg.arm_targets_source!r}"
            )
        if cfg.body_pose_source != cfg.arm_targets_source:
            raise ValueError(
                "Mixing body_pose_source != arm_targets_source is not "
                "supported. Both must take the same value "
                "('internal', 'zmq', or 'vla')."
            )
        # ``_subscribe_mode`` is the umbrella "drive everything from
        # an upstream ZMQ stream" flag; ``_vla_subscribe_mode`` further
        # narrows to "bridge owns the wire and embeds hands in the
        # pose payload", which controls whether we bind :5556 ourselves
        # and whether we wait for the manager's recorder_cmd to start
        # an episode.
        self._subscribe_mode = (cfg.body_pose_source in ("zmq", "vla"))
        self._vla_subscribe_mode = (cfg.body_pose_source == "vla")

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
        # not the FSQ-encoded ``motion_token`` over the wire. The tokenizer
        # below is *dataset-only*: it produces the ground-truth
        # ``action.motion_token`` column the VLA trains against. The wire
        # motion_token field stays as zeros regardless.
        self._cfg.sonic_checkpoint = (
            Path(cfg.sonic_checkpoint) if cfg.sonic_checkpoint is not None else None
        )

        self._stop_event = threading.Event()
        # Re-entrancy guard for ``stop()``. Set the first time stop()
        # is called; subsequent calls short-circuit so the lerobot
        # save path can't run twice on the same buffered episode.
        # See the comment in ``stop()`` for the full failure mode.
        self._stop_called = False

        # Robot model + dataset features. Each step prints with
        # ``flush=True`` so a slow / failing init shows the exact stage
        # that stalled (was a silent hang before the launcher started
        # passing ``-u`` to python -- belt + suspenders here in case
        # someone invokes the recorder directly without the launcher).
        _init_t0 = time.monotonic()
        print("[recorder] loading X2 robot model + features …", flush=True)
        try:
            self._robot_model = get_x2_robot_model(hand_variant="omnihand_10")
        except Exception as exc:
            import traceback
            print(
                f"[recorder] FATAL: get_x2_robot_model failed: {exc}\n"
                + traceback.format_exc(),
                flush=True,
            )
            raise
        print(
            f"[recorder]   robot model ready "
            f"({self._robot_model.num_joints} joints, "
            f"{time.monotonic() - _init_t0:.1f}s)",
            flush=True,
        )
        _features_t0 = time.monotonic()
        try:
            self._features = get_features_x2_vla(
                self._robot_model,
                hand_dof_per_side=HAND_DOF_OMNI,
                include_front_cam=cfg.record_front_cam,
                include_head_cameras=cfg.record_head_cameras,
            )
            self._modality_cfg = get_modality_config_x2_vla(
                self._robot_model,
                hand_dof_per_side=HAND_DOF_OMNI,
                include_front_cam=cfg.record_front_cam,
                include_head_cameras=cfg.record_head_cameras,
            )
        except Exception as exc:
            import traceback
            print(
                f"[recorder] FATAL: features / modality config build "
                f"failed: {exc}\n" + traceback.format_exc(),
                flush=True,
            )
            raise
        print(
            f"[recorder]   features + modality ready "
            f"(front_cam={cfg.record_front_cam}, "
            f"head_cameras={cfg.record_head_cameras}, "
            f"{time.monotonic() - _features_t0:.1f}s)",
            flush=True,
        )
        if self._robot_model.num_joints != NUM_BODY_DOFS:
            raise RuntimeError(
                f"unexpected body DOF count {self._robot_model.num_joints} "
                f"!= {NUM_BODY_DOFS}"
            )

        # The recorder drives the deploy via ``joint_pos_mj`` on the wire
        # (SONIC tracking reference). ``motion_token`` on the wire is
        # zeros unless ``body_pose`` carried a 64-D tensor (VLA bridge);
        # the planner path leaves the field unset so we publish zeros.
        self._zero_motion_token = np.zeros(
            SONIC_MOTION_TOKEN_DIM, dtype=np.float64
        )

        # Layer 3 byte-parity probe state: dumps once per process the
        # first time the subscribe loop sees a fully-populated snap.
        self._obs_dump_recorder_done: bool = False

        # Inline SONIC FSQ tokenizer for ``action.motion_token`` in the
        # LeRobot dataset. Loaded once at startup (~50 MB of encoder
        # weights from the ~398 MB ``.pt``), then called per tick in
        # :meth:`_encode_motion_token` (direct mode) or
        # :meth:`_encode_motion_token_from_snapshot` (subscribe mode).
        # When ``cfg.sonic_checkpoint`` is ``None`` we leave
        # ``self._tokenizer`` unset and the helper falls back to zeros
        # + a one-shot warning so the operator knows the dataset is
        # kinematic-only (intentional for smoke tests, never correct
        # for production data collection).
        #
        # Two construction paths:
        #
        # * ``from_checkpoint_with_config`` (recommended) -- subscribe
        #   mode + an encoder YAML. The recorder builds the same
        #   680-D 10-frame future observation the deploy actor's
        #   internal encoder consumes (via X2EncoderObsBuilder) and
        #   runs the encoder on that exact obs. Token labels are
        #   semantically aligned with what the policy sees.
        #
        # * ``from_checkpoint`` (legacy) -- direct mode (Quest-driven,
        #   no planner snapshot to source a real future window from)
        #   or subscribe mode without --encoder-config. Tiles the
        #   current body_q 11 times into a freeze-pose virtual clip
        #   and encodes that. Tokens encode static intent and the VLA
        #   will only learn to predict "stay where you are". A one-
        #   shot deprecation warning fires on first encode().
        self._tokenizer = None
        if self._cfg.sonic_checkpoint is not None:
            # NB: the checkpoint is ~400 MB on disk and the load runs
            # synchronously on CPU (when ``--sonic-tokenizer-device cpu``)
            # which can take 10-30 s on a cold filesystem cache. Without
            # a "loading…" + flush=True print the operator sees the
            # earlier "features ready" line as the last log entry for
            # tens of seconds and assumes the recorder has hung.
            from gear_sonic.utils.teleop.online_sonic_tokenizer import (
                OnlineSonicTokenizer,
            )
            use_subscribe_mode_path = (
                self._subscribe_mode
                and self._cfg.sonic_encoder_config is not None
            )
            _tok_t0 = time.monotonic()
            print(
                f"[recorder] loading SONIC tokenizer "
                f"(checkpoint={self._cfg.sonic_checkpoint.name}, "
                f"device={self._cfg.sonic_tokenizer_device}, "
                f"path={'+encoder-config' if use_subscribe_mode_path else 'legacy freeze-pose'}"
                f") -- this can take 10-30s on cold cache …",
                flush=True,
            )
            try:
                if use_subscribe_mode_path:
                    self._tokenizer = (
                        OnlineSonicTokenizer.from_checkpoint_with_config(
                            self._cfg.sonic_checkpoint,
                            self._cfg.sonic_encoder_config,
                            device=self._cfg.sonic_tokenizer_device,
                        )
                    )
                else:
                    self._tokenizer = OnlineSonicTokenizer.from_checkpoint(
                        self._cfg.sonic_checkpoint,
                        device=self._cfg.sonic_tokenizer_device,
                    )
            except Exception as exc:
                import traceback
                print(
                    f"[recorder] FATAL: SONIC tokenizer load failed "
                    f"after {time.monotonic() - _tok_t0:.1f}s: {exc}\n"
                    + traceback.format_exc(),
                    flush=True,
                )
                raise
            _tok_dt = time.monotonic() - _tok_t0
            if use_subscribe_mode_path:
                builder = self._tokenizer.obs_builder
                assert builder is not None
                modes = ", ".join(m.name for m in builder.encoder_modes)
                print(
                    f"[recorder] motion_token tokenizer ready in {_tok_dt:.1f}s "
                    f"(checkpoint={self._cfg.sonic_checkpoint.name}, "
                    f"device={self._cfg.sonic_tokenizer_device}, "
                    f"encoder_config="
                    f"{Path(self._cfg.sonic_encoder_config).name}, "
                    f"modes=[{modes}], multi-frame=10x68 -> 680-D)",
                    flush=True,
                )
            else:
                if self._subscribe_mode:
                    # Subscribe mode without --encoder-config. We still
                    # have a real planner snapshot but no YAML to drive
                    # the gather. Operator has explicitly opted out of
                    # the multi-frame path -- warn loudly.
                    print(
                        "[recorder] WARNING: subscribe mode is ON but "
                        "no --encoder-config was passed; falling back "
                        "to the DEPRECATED freeze-pose path (current "
                        "body_q tiled 11 times). Resulting "
                        "action.motion_token labels encode static "
                        "intent and the VLA will only learn 'stand "
                        "still'. Pass --encoder-config "
                        "gear_sonic/data/encoder/x2_observation_config"
                        ".yaml for training-ready labels.",
                        flush=True,
                    )
                else:
                    print(
                        f"[recorder] motion_token tokenizer ready "
                        f"(checkpoint={self._cfg.sonic_checkpoint.name}, "
                        f"device={self._cfg.sonic_tokenizer_device}, "
                        "DEPRECATED freeze-pose mode -- direct-mode "
                        "loop has no planner future to encode)",
                        flush=True,
                    )
        else:
            print(
                "[recorder] WARNING: no --sonic-checkpoint provided; "
                "action.motion_token will be ZEROS for every frame. "
                "Dataset will NOT be VLA-trainable. Re-record with "
                "--sonic-checkpoint <path/to/model_step_*.pt> for "
                "training-ready data.",
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
        #
        # Phase 0 subscribe mode: the manager owns the Quest 3 connection
        # and runs all VR retargeting (IK, hand mapping, calibration); the
        # recorder is Quest-unaware and only consumes ZMQ topics. We set
        # ``self._quest = None`` so any stray reference from the legacy
        # internal-mode path crashes loudly instead of silently mixing
        # two competing input sources.
        if self._subscribe_mode:
            self._quest = None
            print(
                "[recorder] subscribe-mode: Quest3Reader skipped "
                "(manager owns the VR loop)",
                flush=True,
            )
        else:
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
        # In VLA subscribe-mode the bridge owns ``tcp://*:5556``; if the
        # recorder bound the same endpoint zmq would fail with
        # ``Address already in use`` (or silently fight for subscribers
        # depending on platform). Skip the bind and leave the socket
        # connect-less: every ``_publish_pose`` / ``_publish_idle`` call
        # site is also gated below so we never try to send on it.
        if not self._vla_subscribe_mode:
            self._pub_sock.bind(f"tcp://{cfg.pub_host}:{cfg.pub_port}")
        else:
            print(
                f"[recorder] VLA subscribe-mode: skipping pose PUB bind on "
                f"tcp://{cfg.pub_host}:{cfg.pub_port} (VLA bridge owns "
                f"the wire); recorder is ingest-only",
                flush=True,
            )

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

        # Phase 0 subscribe-mode SUB: a single thread that fans the
        # planner's body_pose stream and the manager's
        # arm_targets/hand_finger_cmd/stream_mode/recorder_cmd streams
        # into ``self._sub_state``. Built unconditionally so tests can
        # introspect it; the thread is only ``start()``-ed when
        # ``_subscribe_mode`` is True.
        self._sub_state: Optional[_SubscribeModeState] = None
        self._sub_mode_thread: Optional[threading.Thread] = None
        if self._subscribe_mode:
            self._sub_state = _SubscribeModeState()
            self._sub_mode_thread = threading.Thread(
                target=_subscribe_mode_thread,
                kwargs=dict(
                    body_pose_url=(
                        f"tcp://{cfg.body_pose_sub_host}:"
                        f"{cfg.body_pose_sub_port}"
                    ),
                    body_pose_topic=cfg.body_pose_sub_topic,
                    arm_and_hands_url=(
                        f"tcp://{cfg.arm_and_hands_sub_host}:"
                        f"{cfg.arm_and_hands_sub_port}"
                    ),
                    arm_targets_topic=cfg.arm_targets_topic,
                    hand_finger_cmd_topic=cfg.hand_finger_cmd_topic,
                    stream_mode_topic=cfg.stream_mode_topic,
                    recorder_cmd_topic=cfg.recorder_cmd_topic,
                    state=self._sub_state,
                    stop_event=self._stop_event,
                    verbose=cfg.verbose,
                    vla_mode=self._vla_subscribe_mode,
                ),
                name="recorder-sub-mode",
                daemon=True,
            )
            if self._vla_subscribe_mode:
                print(
                    "[recorder] VLA subscribe-mode wired:\n"
                    f"  bridge     SUB tcp://{cfg.body_pose_sub_host}:"
                    f"{cfg.body_pose_sub_port} "
                    f"topic={cfg.body_pose_sub_topic!r} "
                    f"(arms + hands embedded; manager SUB skipped)\n"
                    f"  episodes   auto-start on first body_pose, "
                    f"auto-save on shutdown (one run = one episode)",
                    flush=True,
                )
            else:
                print(
                    "[recorder] subscribe-mode wired:\n"
                    f"  body_pose  SUB tcp://{cfg.body_pose_sub_host}:"
                    f"{cfg.body_pose_sub_port} topic={cfg.body_pose_sub_topic!r}\n"
                    f"  manager    SUB tcp://{cfg.arm_and_hands_sub_host}:"
                    f"{cfg.arm_and_hands_sub_port} topics="
                    f"[{cfg.arm_targets_topic!r}, "
                    f"{cfg.hand_finger_cmd_topic!r}, "
                    f"{cfg.stream_mode_topic!r}, "
                    f"{cfg.recorder_cmd_topic!r}]",
                    flush=True,
                )

        # ── Gesture playback wiring ────────────────────────────────────
        # Catalog is best-effort: if the file is missing / malformed we
        # log a warning and proceed with an empty catalog so the SUB
        # is still bound (ad-hoc ``--pkl`` payloads continue to work).
        # Setting ``cfg.gesture_catalog_path is None`` disables gesture
        # support entirely (no SUB, no queue). Only meaningful in
        # subscribe mode; legacy internal-Quest path ignores all of
        # this regardless.
        self._gesture_catalog: dict[str, GestureCatalogEntry] = {}
        self._gesture_request_queue: Optional["queue.Queue[Any]"] = None
        self._gesture_thread: Optional[threading.Thread] = None
        self._active_gesture: Optional[GestureSession] = None
        # When the active gesture finishes with ``hold_after=True`` we
        # latch its final body_q + root_quat here so the publish loop
        # can republish them tick-after-tick until an explicit stop or
        # a new play arrives. ``None`` means "not holding". See
        # :meth:`_run_subscribe_mode` for the publish gate and the
        # GestureCatalogEntry.hold_after docstring for the wire semantics.
        self._active_gesture_hold_after: bool = False
        self._gesture_held_frame: Optional[dict[str, np.ndarray]] = None

        # Idle-yaw rebase logging gates. We re-derive the idle frame's
        # ``root_quat_xyzw`` from the live ``x2_debug`` ``base_quat``
        # every tick that we publish an idle stand pose (see
        # :meth:`_publish_idle`). The log messages are gated by these
        # one-way flags so the operator sees one info line on first
        # activation and one when the source goes stale; without the
        # gates a 50 Hz idle publish would spam either message.
        self._idle_yaw_rebase_logged_active: bool = False
        self._idle_yaw_rebase_logged_fallback: bool = False
        if self._subscribe_mode and cfg.gesture_catalog_path is not None:
            cat_path = Path(cfg.gesture_catalog_path)
            try:
                self._gesture_catalog = load_gesture_catalog(cat_path)
                if cfg.verbose:
                    print(
                        f"[recorder] gesture catalog: {cat_path} "
                        f"({len(self._gesture_catalog)} entries)",
                        flush=True,
                    )
            except (FileNotFoundError, ValueError) as exc:
                print(
                    f"[recorder] gesture catalog disabled: {exc} "
                    f"(ad-hoc --pkl play still works)",
                    flush=True,
                )
            self._gesture_request_queue = queue.Queue()
            self._gesture_thread = threading.Thread(
                target=_subscribe_gesture_cmd_thread,
                kwargs=dict(
                    url=(
                        f"tcp://{cfg.gesture_cmd_host}:"
                        f"{cfg.gesture_cmd_port}"
                    ),
                    topic=cfg.gesture_cmd_topic,
                    request_queue=self._gesture_request_queue,
                    stop_event=self._stop_event,
                    verbose=cfg.verbose,
                ),
                name="recorder-gesture-cmd-sub",
                daemon=True,
            )
            if cfg.verbose:
                print(
                    f"[recorder] gesture_cmd wired: "
                    f"SUB tcp://{cfg.gesture_cmd_host}:"
                    f"{cfg.gesture_cmd_port} "
                    f"topic={cfg.gesture_cmd_topic!r}",
                    flush=True,
                )

        # MuJoCo renderer is constructed lazily on the recording thread:
        # MuJoCo's EGL backend pins to the thread that created the
        # renderer. The recording loop needs it, so we delay until that
        # loop spins up.
        self._renderer: Any | None = None
        # Optional second renderer for ``front_cam`` (gated on
        # ``cfg.record_front_cam`` and on ``cfg.scene_xml_path`` actually
        # containing a camera with that name). Same EGL-thread-pinning
        # constraint, same lazy build-on-recording-thread pattern. See
        # :meth:`_build_renderer` and :meth:`_record_frame`.
        self._front_cam_renderer: Any | None = None

        # ── PC2 head-camera ZMQ client (head_front + stereo L/R) ───────
        # ``cfg.record_head_cameras`` flips on a background thread that
        # pulls merged ``ImageMessageSchema`` frames from the bridge
        # (see ``gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py``)
        # at the polling rate below. Each tick of ``_record_frame``
        # consults :meth:`_snapshot_head_camera_images` for the freshest
        # bundle; bundles older than ``cfg.camera_max_staleness_s`` cause
        # the tick to be dropped (no parquet row written) so the dataset
        # never contains stale physical-camera frames silently.
        self._head_camera_client: Any | None = None
        self._head_camera_thread: Optional[threading.Thread] = None
        self._head_camera_lock = threading.Lock()
        self._head_camera_latest: Optional[dict[str, np.ndarray]] = None
        self._head_camera_latest_ts: float = 0.0
        self._head_camera_frames_received: int = 0
        self._head_camera_stale_warns: int = 0
        if not cfg.teleop_only and cfg.record_head_cameras:
            self._init_head_cameras()

        # ── Live pelvis-pose cache ─────────────────────────────────────
        # Most-recent ``(x, y, z)`` from the bridge's ``robot_pose`` PUB
        # (see :data:`gear_sonic.utils.teleop.zmq.robot_pose_zmq`). The
        # background SUB thread (:meth:`_robot_pose_subscriber`) keeps
        # this current at state-rate; the recording loop reads it every
        # frame so the rendered scene -- particularly the world-fixed
        # ``front_cam`` -- shows the robot wherever the deploy actually
        # placed it, instead of the renderer's hardcoded
        # ``(0, 0, 0.793)`` default. The latch starts at the renderer's
        # default so the very first render before any ``robot_pose``
        # packet arrives still composes a sensible frame.
        self._pelvis_pose_lock = threading.Lock()
        self._latest_pelvis_xyz: np.ndarray = np.array(
            [0.0, 0.0, 0.793], dtype=np.float64
        )
        self._pelvis_pose_seen_any: bool = False
        self._robot_pose_thread: Optional[threading.Thread] = None

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

            # robot_pose SUB (bridge -> recorder, sim-only ground-truth
            # pelvis qpos). Always-on companion to ``scene_state``: the
            # bridge publishes this unconditionally on the default port
            # (5570) regardless of recorder config, and we want the
            # cache primed so the very first ``front_cam`` render lands
            # at the actual robot location instead of the hardcoded
            # ``(0, 0, 0.793)`` default. Only spun up in scene mode
            # because that's the only mode where the bridge runs at
            # all (the legacy flat-floor recorder path doesn't talk to
            # the bridge).
            self._robot_pose_thread = threading.Thread(
                target=self._robot_pose_subscriber,
                name="robot-pose-sub",
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

        # Hand diagnostic snapshot, populated each tick from the
        # dispatch site (raw Quest 3 inputs + filtered values + final
        # hand_q published to deploy). Read by ``_log_hand_diag`` on
        # the same throttle as ``_print_status`` so we don't spam the
        # terminal. Set to ``None`` here so the diag is a no-op until
        # at least one tick has run. The diag is most useful for
        # debugging "thumb won't close" / "controller trigger
        # ignored" issues -- it makes the per-tick dispatch explicit.
        self._last_hand_diag: Optional[dict] = None

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
        if not self._subscribe_mode:
            self._quest.start()
        self._sub_thread.start()
        if self._sub_mode_thread is not None:
            self._sub_mode_thread.start()
        if self._gesture_thread is not None:
            self._gesture_thread.start()
        # Robocasa scene_state subscriber (no-op when not in scene mode).
        if self._scene_state_thread is not None:
            self._scene_state_thread.start()
        if self._robot_pose_thread is not None:
            self._robot_pose_thread.start()
        if self._head_camera_thread is not None:
            self._head_camera_thread.start()
        # Give PUB-SUB sockets a beat to wire up before we start
        # blasting messages.
        time.sleep(0.2)
        # Calibration + IK + finger filter only matter in internal mode;
        # in subscribe mode the manager owns all of that and the recorder
        # just routes its outputs.
        if self._subscribe_mode:
            self._calibration = None
            self._teleop = None
            self._finger_filter_left = None
            self._finger_filter_right = None
            print(
                "[recorder] subscribe-mode: skipping calibration / IK / "
                "finger-filter init (manager owns them)",
                flush=True,
            )
            return

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
            self._prev_q3_left_hand_src: Any = _PREV_Q3_HAND_SRC_UNSET
            self._prev_q3_right_hand_src: Any = _PREV_Q3_HAND_SRC_UNSET
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
            self._prev_q3_left_hand_src = _PREV_Q3_HAND_SRC_UNSET
            self._prev_q3_right_hand_src = _PREV_Q3_HAND_SRC_UNSET
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
        # Idempotency guard. ``stop()`` is wired into the launcher's
        # SIGINT + SIGTERM handlers (record_x2_dataset.py registers
        # ``_on_signal`` for both), so the launcher sending SIGTERM
        # right after the operator's Ctrl-C produces TWO stop() calls.
        # Worse, Python delivers signal handlers between bytecode
        # instructions on the main thread, so the second call can land
        # mid-flight inside the first call's ``_stop_episode`` (the
        # lerobot writer's ``save_episode`` releases the GIL while
        # encoding the mp4). Without this guard the second call
        # observes ``_is_recording=True`` (the first hasn't reset it
        # yet) and saves the *same* 380 buffered frames a second time
        # under episode_000001 -- producing duplicate parquets that
        # corrupt the dataset's episode counter on the next run.
        # A simple boolean is safe because (a) all stop() invocations
        # land on the main thread (signal handlers run there) and
        # (b) the read-then-set is atomic at the Python bytecode
        # level so re-entrant calls observe the True state set by the
        # outer call.
        if getattr(self, "_stop_called", False):
            return
        self._stop_called = True
        self._stop_event.set()
        # Persist any buffered episode?
        #
        # Internal / planner-zmq modes: NO -- the recorder requires an
        # explicit save command (X / Y button) from the operator.
        # Buffered-but-not-saved frames are discarded on shutdown so an
        # accidental Ctrl-C doesn't contaminate the dataset with junk.
        #
        # VLA subscribe-mode: YES -- there is no operator console, and
        # the runtime-script wrapper is the only thing that can trigger
        # ``stop()`` (via SIGTERM on Ctrl-C). The recorder auto-started
        # the episode on first body_pose, so the buffer at this point
        # IS the rollout. Dropping it would silently lose every
        # ``--with-record`` capture. We MUST save here -- the
        # ``finally`` block in :meth:`_run_subscribe_mode` would also
        # try, but the signal-handler path calls ``stop()`` BEFORE
        # ``run()`` returns, so the run-loop's finally never sees
        # ``_is_recording=True``.
        if self._is_recording:
            n_frames = len(self._episode_buffer)
            if (
                getattr(self, "_vla_subscribe_mode", False)
                and not self._cfg.teleop_only
            ):
                print(
                    f"[recorder] stop: VLA subscribe-mode auto-save: "
                    f"flushing {n_frames} buffered frames to dataset",
                    flush=True,
                )
                try:
                    self._stop_episode(save=True)
                except Exception as exc:  # noqa: BLE001
                    # If the writer chain explodes mid-save, log loudly
                    # but never let the recorder hang the launcher's
                    # 30 s SIGTERM budget. Drop the buffer as a fallback
                    # so the next stop() call short-circuits.
                    print(
                        f"[recorder] stop: VLA auto-save FAILED ({exc}); "
                        f"dropping {n_frames} frames so shutdown can "
                        f"complete. Re-run; bridge.log + recorder.log "
                        f"hold the chunk dumps for postmortem.",
                        flush=True,
                    )
                    self._episode_buffer.reset()
                    self._is_recording = False
            else:
                print(
                    f"[recorder] stop: dropping {n_frames} buffered "
                    f"frames (use button X before exiting to save).",
                    flush=True,
                )
                self._episode_buffer.reset()
        try:
            self._sub_thread.join(timeout=1.0)
        except Exception:
            pass
        if self._sub_mode_thread is not None:
            try:
                self._sub_mode_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._gesture_thread is not None:
            try:
                self._gesture_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._scene_state_thread is not None:
            try:
                self._scene_state_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._robot_pose_thread is not None:
            try:
                self._robot_pose_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._head_camera_thread is not None:
            try:
                self._head_camera_thread.join(timeout=1.0)
            except Exception:
                pass
        if self._head_camera_client is not None:
            try:
                self._head_camera_client.close()
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
        if self._quest is not None:
            try:
                self._quest.stop()
            except Exception:
                pass
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
        if self._front_cam_renderer is not None:
            try:
                self._front_cam_renderer.close()
            except Exception:
                pass

    # -- main loop ------------------------------------------------------------

    def run(self) -> int:
        """Blocking 50 Hz publish + record loop. Returns total ticks."""
        if not self._cfg.teleop_only:
            self._build_renderer()
        self._print_startup_banner()
        if self._subscribe_mode:
            return self._run_subscribe_mode()
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
                # Snapshot the raw + filtered finger inputs so the
                # throttled hand-diag log (every status_log_period_s)
                # can render exactly what the dispatch saw this tick.
                # We populate this dict at the *end* of the dispatch
                # block once we know the final ``hand_q`` values, but
                # initialise it here so an early `continue` (e.g. no
                # vr_pose this frame) doesn't leak last tick's diag.
                hand_diag_capture: dict = {
                    "tick": tick,
                    "triggers": triggers,
                }

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
                # IMPORTANT: do NOT gate XRHand vs controller *dispatch*
                # on ``l_src`` alone. In multimodal mode the WebXR client
                # tags ``source = "controller"`` because gripSpace wins
                # for IK pose, but the same frame can still carry XRHand
                # curls + thumb opposition.
                #
                # DO reset :class:`FingerSignalFilter` when ``l_src`` /
                # ``r_src`` *change* (``hand`` <-> ``controller`` /
                # ``None``). Otherwise the filter's NaN-holding EMA
                # freezes the last XRHand curls while raw ``curls`` are
                # ``None`` after returning to controllers-only, and
                # trigger/grip never drives ``hand_q`` again.
                l_curls_raw, r_curls_raw, l_src, r_src = self._quest.get_hand_curls()
                l_oppose_raw, r_oppose_raw = self._quest.get_thumb_opposition()
                l_finger_tip_oppose, r_finger_tip_oppose = (
                    self._quest.get_finger_tip_oppose()
                )
                # Capture the unfiltered values so the diag can show
                # both raw and filtered (so we can tell whether it's
                # the headset or the FingerSignalFilter zeroing out
                # the thumb).
                hand_diag_capture["l_src"] = l_src
                hand_diag_capture["r_src"] = r_src
                hand_diag_capture["l_curls_raw"] = (
                    None if l_curls_raw is None else np.asarray(l_curls_raw).copy()
                )
                hand_diag_capture["r_curls_raw"] = (
                    None if r_curls_raw is None else np.asarray(r_curls_raw).copy()
                )
                hand_diag_capture["l_oppose_raw"] = l_oppose_raw
                hand_diag_capture["r_oppose_raw"] = r_oppose_raw

                # Apply the per-side smoothing filter on top of the raw
                # Quest 3 inputs. The retargeter sees only the filtered
                # values; the raw values are no longer kept (the SONIC-
                # record path doesn't write a debug NPZ).
                l_curls, r_curls = l_curls_raw, r_curls_raw
                l_oppose, r_oppose = l_oppose_raw, r_oppose_raw
                if self._finger_filter_left is not None and self._finger_filter_right is not None:
                    if (
                        self._prev_q3_left_hand_src is not _PREV_Q3_HAND_SRC_UNSET
                        and l_src != self._prev_q3_left_hand_src
                    ):
                        self._finger_filter_left.reset()
                    if (
                        self._prev_q3_right_hand_src is not _PREV_Q3_HAND_SRC_UNSET
                        and r_src != self._prev_q3_right_hand_src
                    ):
                        self._finger_filter_right.reset()
                    self._prev_q3_left_hand_src = l_src
                    self._prev_q3_right_hand_src = r_src

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
                hand_diag_capture["l_curls_filt"] = (
                    None if l_curls is None else np.asarray(l_curls).copy()
                )
                hand_diag_capture["r_curls_filt"] = (
                    None if r_curls is None else np.asarray(r_curls).copy()
                )
                hand_diag_capture["l_oppose_filt"] = l_oppose
                hand_diag_capture["r_oppose_filt"] = r_oppose

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
                    hand_diag_capture["l_dispatch"] = "xrhand"
                else:
                    left_ratio, _ = controller_grasp_ratio(
                        left_trigger=triggers[0],
                        right_trigger=triggers[1],
                        left_grip=triggers[2],
                        right_grip=triggers[3],
                        mode=self._cfg.hand_input_mode,
                    )
                    left_hand_q = grasp_command_from_ratio("left", left_ratio)
                    hand_diag_capture["l_dispatch"] = "controller"

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
                    hand_diag_capture["r_dispatch"] = "xrhand"
                else:
                    _, right_ratio = controller_grasp_ratio(
                        left_trigger=triggers[0],
                        right_trigger=triggers[1],
                        left_grip=triggers[2],
                        right_grip=triggers[3],
                        mode=self._cfg.hand_input_mode,
                    )
                    right_hand_q = grasp_command_from_ratio("right", right_ratio)
                    hand_diag_capture["r_dispatch"] = "controller"

                # Final per-tick capture: the actual 10-D commands the
                # bridge will see for each hand. Indices 0/1/2 are
                # thumb_roll / thumb_abad / thumb_mcp -- the diag log
                # focuses on those because that's where the "thumb
                # won't close" symptom shows up.
                hand_diag_capture["left_hand_q"] = left_hand_q.copy()
                hand_diag_capture["right_hand_q"] = right_hand_q.copy()
                hand_diag_capture["engaged"] = bool(self._teleop.is_engaged)
                self._last_hand_diag = hand_diag_capture

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

                # The wire's ``motion_token`` field is always zeros (the
                # C++ deploy actor consumes only ``joint_pos_mj`` as
                # reference motion -- spending CPU/GPU on the wire copy
                # would burn power for no consumer). The dataset's
                # ``action.motion_token`` column, however, is the SONIC
                # FSQ encoding of the *commanded* body_q (operator
                # intent) and is the supervision target for VLA
                # training. ``_encode_motion_token`` returns zeros if
                # no ``--sonic-checkpoint`` was provided (warning
                # already printed at startup).
                wire_token = self._zero_motion_token
                dataset_token = self._encode_motion_token(body_q_mj)

                # Publish to deploy.
                self._publish_pose(
                    body_q_mj=body_q_mj,
                    motion_token=wire_token,
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
                        commanded_token=dataset_token,
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
                    self._log_hand_diag()

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

    # -- subscribe-mode loop --------------------------------------------------

    def _run_subscribe_mode(self) -> int:
        """Phase 0 subscribe-only loop: planner + manager drive everything.

        The recorder is Quest-unaware in this mode. Each tick we:
            1. Snapshot the latest body_pose (planner) +
               arm_targets / hand_finger_cmd (manager).
            2. Drain pending recorder_cmd episode events from the manager.
            3. Merge body_pose with arm/hand targets into a 31-DOF
               joint_pos_mj reference.
            4. Publish to the deploy on ``cfg.pub_port`` under topic
               ``cfg.pub_topic`` (back-compat with the legacy direct
               wiring -- the deploy doesn't know the planner moved).
            5. Optionally record an aligned (observation, action) frame.
        """
        assert self._sub_state is not None
        period = 1.0 / max(self._cfg.publish_rate_hz, 1e-6)
        next_tick = time.monotonic()
        tick = 0
        wait_msg = False
        first_body_pose_logged = False
        zero_hand = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)

        try:
            while not self._stop_event.is_set():
                snap = self._sub_state.snapshot()

                # Drain manager-issued episode commands first so a
                # ``start`` and an immediate ``save`` both get serviced
                # in their original order.
                for action, sub_tick in self._sub_state.drain_recorder_cmds():
                    if action == "start":
                        self._start_episode()
                    elif action == "save":
                        self._stop_episode(save=True)
                    elif action == "discard":
                        self._stop_episode(save=False)
                    elif action == "estop":
                        if self._is_recording:
                            print(
                                f"[recorder] [estop@{sub_tick}] manager "
                                f"signalled OFF; dropping open episode",
                                flush=True,
                            )
                            self._stop_episode(save=False)
                    else:
                        if self._cfg.verbose:
                            print(
                                f"[recorder] unknown recorder_cmd "
                                f"action={action!r} (tick={sub_tick}); "
                                f"ignoring",
                                flush=True,
                            )

                # Gesture override path. Drained AFTER recorder_cmd so
                # the manager can still start/stop dataset episodes
                # mid-gesture; runs BEFORE the body_pose-None check so
                # an operator can fire a gesture even before kplanner
                # publishes (e.g. on a cold-started stack). Skipped in
                # VLA mode: the bridge owns the wire and there is no
                # external gesture trigger surface.
                if not self._vla_subscribe_mode:
                    self._drain_gesture_commands(snap)
                if (
                    not self._vla_subscribe_mode
                    and self._active_gesture is not None
                    and not self._active_gesture.is_done()
                ):
                    self._publish_gesture_frame(tick=tick)
                    if self._active_gesture.is_done():
                        # Clip just finished on this tick: either snap
                        # back to kplanner (hold_after=False) or latch
                        # the last frame for indefinite republish
                        # (hold_after=True).
                        if self._active_gesture_hold_after:
                            self._gesture_held_frame = {
                                "body_q_mj": self._active_gesture.body_q_mj[-1].astype(
                                    np.float64, copy=True
                                ),
                                "root_quat_xyzw": self._active_gesture.root_quat_xyzw[-1].astype(
                                    np.float32, copy=True
                                ),
                            }
                            print(
                                f"[recorder] gesture "
                                f"{self._active_gesture.entry.name!r} "
                                f"completed; HOLDING last frame "
                                f"(send 'stop' or another 'play' to release)",
                                flush=True,
                            )
                        else:
                            if self._cfg.verbose:
                                print(
                                    f"[recorder] gesture "
                                    f"{self._active_gesture.entry.name!r} "
                                    f"completed; resuming kplanner forwarding",
                                    flush=True,
                                )
                        self._active_gesture = None
                        self._active_gesture_hold_after = False
                    tick += 1
                    next_tick += period
                    self._sleep_until(next_tick)
                    continue
                if (
                    not self._vla_subscribe_mode
                    and self._gesture_held_frame is not None
                ):
                    # Hold mode: keep the robot parked at the last gesture
                    # frame. We bypass the body_pose-None check on purpose
                    # -- the operator chose to leave the robot here.
                    self._publish_held_gesture_frame(tick=tick)
                    tick += 1
                    next_tick += period
                    self._sleep_until(next_tick)
                    continue

                body_pose = snap["body_pose_q_mj"]
                if body_pose is None:
                    if not wait_msg:
                        if self._vla_subscribe_mode:
                            print(
                                f"[recorder] VLA subscribe-mode: waiting "
                                f"for first body_pose on tcp://"
                                f"{self._cfg.body_pose_sub_host}:"
                                f"{self._cfg.body_pose_sub_port} topic="
                                f"{self._cfg.body_pose_sub_topic!r} … "
                                f"(bridge owns wire; no idle publish)",
                                flush=True,
                            )
                        else:
                            suffix = (
                                ""
                                if self._cfg.idle_publish_enabled
                                else "  [no-idle-publish: pose wire stays SILENT]"
                            )
                            print(
                                f"[recorder] subscribe-mode: waiting for first "
                                f"body_pose on tcp://"
                                f"{self._cfg.body_pose_sub_host}:"
                                f"{self._cfg.body_pose_sub_port} topic="
                                f"{self._cfg.body_pose_sub_topic!r} …"
                                f"{suffix}",
                                flush=True,
                            )
                        wait_msg = True
                    # In VLA mode the bridge owns :5556; do NOT publish
                    # idle here or we'd race the bridge's wire. Without
                    # idle the recorder simply ingests once the bridge
                    # starts flowing.
                    if (
                        not self._vla_subscribe_mode
                        and self._cfg.idle_publish_enabled
                    ):
                        self._publish_idle()
                    next_tick += period
                    self._sleep_until(next_tick)
                    continue

                if not first_body_pose_logged:
                    print(
                        "[recorder] subscribe-mode: first body_pose "
                        "received; entering merge+publish loop",
                        flush=True,
                    )
                    first_body_pose_logged = True
                    # VLA mode: there is no manager publishing
                    # ``recorder_cmd start``; the runtime is "one
                    # invocation = one episode". Auto-start now so the
                    # rollout actually lands in the dataset. The
                    # finally block already auto-saves on shutdown.
                    if (
                        self._vla_subscribe_mode
                        and not self._cfg.teleop_only
                        and not self._is_recording
                    ):
                        print(
                            "[recorder] VLA subscribe-mode: auto-starting "
                            "the first (and only) episode for this run",
                            flush=True,
                        )
                        self._start_episode()
                    # Signal the VLA bridge that we are subscribed + ready
                    # to record. The bridge's --wait-for-ready-file gate
                    # is holding its inference thread at idle stand until
                    # this file appears, so creating it here ensures the
                    # arm rise lands in the recording instead of in the
                    # missed warm-up window. Best-effort: failures only
                    # log so a botched signal can't crash the loop.
                    if self._cfg.ready_file is not None:
                        try:
                            ready_path = Path(self._cfg.ready_file)
                            ready_path.parent.mkdir(parents=True, exist_ok=True)
                            ready_path.touch()
                            print(
                                f"[recorder] ready-file touched: {ready_path} "
                                "(bridge may now start VLA inference)",
                                flush=True,
                            )
                        except Exception as exc:
                            print(
                                f"[recorder] WARNING: failed to touch "
                                f"ready-file {self._cfg.ready_file}: {exc}",
                                flush=True,
                            )

                # Merge: planner-driven legs + waist + head come from
                # body_pose; arm slices come from the manager IF it is
                # publishing arm_targets, otherwise we fall through to
                # the planner's stand-pose arm slice (== legacy idle).
                body_q_mj = np.asarray(body_pose, dtype=np.float64).copy()
                left_arm = snap["arm_left_q"]
                right_arm = snap["arm_right_q"]
                arm_dof = _LEFT_ARM_MJ_SLICE.stop - _LEFT_ARM_MJ_SLICE.start
                left_arm_valid = (
                    left_arm is not None and left_arm.shape == (arm_dof,)
                )
                right_arm_valid = (
                    right_arm is not None and right_arm.shape == (arm_dof,)
                )
                if left_arm_valid:
                    body_q_mj[_LEFT_ARM_MJ_SLICE] = left_arm
                if right_arm_valid:
                    body_q_mj[_RIGHT_ARM_MJ_SLICE] = right_arm

                # Forward the planner's full v5 future-window so the
                # deploy's tokenizer can anticipate the next 0.9 s of
                # locomotion (without it the C++ ZmqPoseInputSource
                # falls back to the legacy single-frame Sample() path
                # and the policy's future tokens are pinned at the
                # current pose -- which is what made the robot "step
                # in place" when commanded to walk).
                jpos_future_planner = snap["joint_pos_mj_future"]
                rot_future_planner = snap["root_quat_xyzw_future"]
                fidx_future_planner = snap["frame_index_future"]
                future_dt_s = snap["future_dt_s"]
                jpos_future_overlaid: Optional[np.ndarray] = None
                if (
                    jpos_future_planner is not None
                    and rot_future_planner is not None
                    and jpos_future_planner.ndim == 2
                    and jpos_future_planner.shape[1] == NUM_BODY_DOFS
                    and rot_future_planner.shape == (
                        jpos_future_planner.shape[0], 4,
                    )
                ):
                    # The manager's arm_targets is a "current command"
                    # from the operator -- we don't have an arm
                    # trajectory to look ahead with. Pin the same
                    # commanded arm pose across every future slot;
                    # leg / waist / head trajectories continue to
                    # follow the planner's curated bin verbatim.
                    jpos_future_overlaid = jpos_future_planner.astype(
                        np.float32, copy=True,
                    )
                    if left_arm_valid:
                        jpos_future_overlaid[
                            :, _LEFT_ARM_MJ_SLICE
                        ] = left_arm.astype(np.float32, copy=False)
                    if right_arm_valid:
                        jpos_future_overlaid[
                            :, _RIGHT_ARM_MJ_SLICE
                        ] = right_arm.astype(np.float32, copy=False)

                left_hand = snap["left_hand_q"]
                right_hand = snap["right_hand_q"]
                left_hand_q = (
                    left_hand if left_hand is not None else zero_hand
                )
                right_hand_q = (
                    right_hand if right_hand is not None else zero_hand
                )

                # Dataset token: SONIC FSQ encoding of the planner's (or
                # VLA body's) future window for ``action.motion_token``.
                # Wire token: passthrough 64-D ``motion_token`` from
                # ``body_pose`` when present (live VLA); else zeros
                # (heuristic planner omits the field).
                merged_snap = dict(snap)
                merged_snap["body_pose_q_mj"] = body_q_mj
                if jpos_future_overlaid is not None:
                    merged_snap["joint_pos_mj_future"] = (
                        jpos_future_overlaid
                    )
                wt = snap["wire_motion_token"]
                if wt is not None and wt.shape == (SONIC_MOTION_TOKEN_DIM,):
                    wire_token = wt.astype(np.float64, copy=False)
                else:
                    wire_token = self._zero_motion_token
                dataset_token = self._encode_motion_token_from_snapshot(
                    merged_snap
                )

                # Layer 3 byte-parity probe: dump the first
                # fully-populated snapshot + builder obs to disk and
                # continue. Triggered once per process; safe to leave
                # enabled across long sessions.
                self._maybe_dump_recorder_obs(merged_snap)

                # Publish merged pose back onto the deploy wire. Skipped
                # in VLA mode: the bridge already owns :5556 and the
                # recorder is ingest-only there; publishing would
                # spawn a competing PUB on the same endpoint.
                if not self._vla_subscribe_mode:
                    self._publish_pose(
                        body_q_mj=body_q_mj,
                        motion_token=wire_token,
                        left_hand_q=left_hand_q,
                        right_hand_q=right_hand_q,
                        tick=tick,
                        root_quat_xyzw=snap["root_quat_xyzw"],
                        root_xy_world=snap.get("root_xy_world"),
                        root_z_world=snap.get("root_z_world"),
                        joint_pos_mj_future=jpos_future_overlaid,
                        root_quat_xyzw_future=rot_future_planner,
                        frame_index_future=fidx_future_planner,
                        future_dt_s=future_dt_s,
                    )

                if self._is_recording:
                    self._record_frame(
                        commanded_body_q_mj=body_q_mj,
                        commanded_left_hand_q=left_hand_q,
                        commanded_right_hand_q=right_hand_q,
                        commanded_token=dataset_token,
                    )

                now_log = time.monotonic()
                self._maybe_warn_deploy_silent(now=now_log)
                if (
                    self._cfg.verbose
                    and (now_log - self._last_status_log_t)
                        >= self._cfg.status_log_period_s
                ):
                    self._last_status_log_t = now_log
                    # Operator-hand diag: surfaces what's actually flowing
                    # downstream when the operator is in ARM_MANIPULATION.
                    # Diagnoses the 2026-06-10 "fingers not responding"
                    # symptom: if |L_hand|/|R_hand| are zero throughout an
                    # override window the manager isn't producing
                    # non-zero hand_q (operator hasn't pulled triggers),
                    # NOT a wiring bug. Conversely, non-zero norms here
                    # paired with VLA-looking fingers in MuJoCo means the
                    # OmniHand SUB isn't reading the proxy downstream
                    # (regression on the spawn_sim_deploy wiring).
                    l_norm = (
                        0.0 if left_hand is None
                        else float(np.linalg.norm(left_hand))
                    )
                    r_norm = (
                        0.0 if right_hand is None
                        else float(np.linalg.norm(right_hand))
                    )
                    l_src = "manager" if left_hand is not None else "zero-fallback"
                    r_src = "manager" if right_hand is not None else "zero-fallback"
                    print(
                        f"[recorder] subscribe-mode status: tick={tick} "
                        f"mode={snap['stream_mode']} "
                        f"recording={self._is_recording} "
                        f"buffered={len(self._episode_buffer)} frames "
                        f"hand|L|={l_norm:.3f}({l_src}) "
                        f"hand|R|={r_norm:.3f}({r_src})",
                        flush=True,
                    )

                tick += 1
                next_tick += period
                self._sleep_until(next_tick)
        finally:
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
        # Operator-facing label: start is triggered by the X button
        # in ARM_MANIPULATION mode (post-2026-05-13 button rebind;
        # the manager publishes "start" on recorder_cmd from the X
        # press). Pre-rebind this branch said [B] because the chord
        # was A+B; the label was never updated.
        if self._cfg.teleop_only:
            print(
                "[recorder] [X] ignored: --teleop-only mode (no dataset writes)",
                flush=True,
            )
            return
        if self._is_recording:
            print("[recorder] [X] ignored: already recording", flush=True)
            return
        self._ensure_exporter()
        self._episode_buffer.reset()
        self._episode_buffer.started_at = time.time()
        self._episode_buffer.task = self._cfg.task
        # Each episode starts with a clean filter buffer so the warm-up
        # window doesn't leak state from the previous episode.
        if self._finger_filter_left is not None:
            self._finger_filter_left.reset()
            self._prev_q3_left_hand_src = _PREV_Q3_HAND_SRC_UNSET
        if self._finger_filter_right is not None:
            self._finger_filter_right.reset()
            self._prev_q3_right_hand_src = _PREV_Q3_HAND_SRC_UNSET

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
                    f"[recorder] [X] scene_reset sent: "
                    f"freejoints={list(payload.object_freejoint_qpos)} "
                    f"welded={list(payload.mutable_body_pos)}",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[recorder] [X] WARNING: scene_reset failed: {exc}",
                    flush=True,
                )

        self._is_recording = True
        # Surface BOTH the session-local 1-based counter (matches
        # operator intuition "I just opened the Nth episode of this
        # session") AND the exporter's actual on-disk episode_index
        # (the slot the parquet will land at). They differ by the
        # number of episodes already in the dataset when this process
        # spawned -- so on a resumed dataset the on-disk index is
        # ``last_finalized + 1 + self._episode_count``, not just
        # ``self._episode_count + 1``.
        next_disk_idx_str = "?"
        if self._exporter is not None:
            try:
                next_disk_idx_str = str(
                    int(self._exporter.episode_buffer["episode_index"])
                )
            except (KeyError, TypeError, ValueError):
                pass
        print(
            f"[recorder] [X] episode start (task={self._cfg.task!r}, "
            f"# {self._episode_count + 1} this session, "
            f"on-disk episode_index={next_disk_idx_str})",
            flush=True,
        )

    def _stop_episode(self, *, save: bool) -> None:
        # Operator-facing label rules (post-2026-05-13 button rebind):
        #   - save=True  is triggered by the Y button (publish "save"
        #     on recorder_cmd from the manager). Log as [Y] so the
        #     operator can match each recorder log line back to the
        #     button they pressed.
        #   - save=False is no longer reachable from any button press
        #     -- the manager does not publish "discard" today -- so
        #     it can only fire from an internal auto-discard path
        #     (timeout, error abort, planned but unimplemented). Log
        #     as [auto-discard] to make that provenance unambiguous.
        kind = "Y" if save else "auto-discard"
        if not self._is_recording:
            print(f"[recorder] [{kind}] ignored: no active episode", flush=True)
            return
        n = len(self._episode_buffer)
        if save and n > 0:
            assert self._exporter is not None
            for frame in self._episode_buffer.frames:
                self._exporter.add_frame(frame)
            # Capture the exporter's on-disk episode_index BEFORE
            # ``save_episode`` rolls the buffer over to the next slot
            # (it reassigns ``self._exporter.episode_buffer`` to a
            # fresh buffer for episode_index+1 as part of finalize).
            # When the operator resumes an existing dataset (preflight
            # found ``meta/info.json`` + finalized episodes), this is
            # ``last_finalized + 1``, NOT ``self._episode_count`` --
            # the latter is process-local and starts at 0 on every
            # spawn, so it would mis-label resumed runs as
            # ``episode_000000`` even though the file lands at slot N.
            saved_idx = int(self._exporter.episode_buffer["episode_index"])
            self._exporter.save_episode()
            self._episode_count += 1
            # Resolve and print the on-disk paths so the operator
            # doesn't have to ``find data/lerobot/<dataset> -newer ...``
            # to locate what they just recorded. Layout matches
            # :class:`Gr00tDataExporter` v2.1: parquet under
            # ``data/chunk-000/`` and the ego-view mp4 under
            # ``videos/chunk-000/observation.images.ego_view/``.
            #
            # IMPORTANT: keep the ``parquet ->`` / ``mp4 ->`` prefix
            # exactly as-is. The wrapper's recorder-log mirror (in
            # run_x2_quest3_planner_stack.sh) greps on the
            # ``[recorder]`` line plus a 4-space indent to surface
            # these to the foreground; changing the indent or the
            # arrow style breaks that mirror silently.
            out_root = self._cfg.output_dir
            chunk_idx = saved_idx // 1000
            parquet_path = (
                out_root / "data" / f"chunk-{chunk_idx:03d}"
                / f"episode_{saved_idx:06d}.parquet"
            )
            mp4_path = (
                out_root / "videos" / f"chunk-{chunk_idx:03d}"
                / "observation.images.ego_view"
                / f"episode_{saved_idx:06d}.mp4"
            )
            print(
                f"[recorder] [Y] episode saved: {n} frames "
                f"(on-disk episode_index={saved_idx}, "
                f"total saved this session={self._episode_count})",
                flush=True,
            )
            print(f"[recorder]     parquet -> {parquet_path}", flush=True)
            print(f"[recorder]     mp4     -> {mp4_path}", flush=True)
        else:
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

    def _encode_motion_token(
        self,
        body_q_mj: np.ndarray,
        root_quat_xyzw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """DEPRECATED freeze-pose chokepoint for the direct-mode loop.

        Tiles the current ``body_q_mj`` 11 times and runs the SONIC
        encoder at frame 0. Used by the recorder's direct mode (Quest-
        driven, no planner snapshot to source a real future window
        from). The first call prints a one-shot deprecation warning
        from :meth:`OnlineSonicTokenizer.encode`. Prefer
        :meth:`_encode_motion_token_from_snapshot` whenever a planner
        snapshot is available (subscribe mode).

        Args:
            body_q_mj: ``(31,)`` MuJoCo-order commanded body pose
                (operator intent, post-IK + post-planner merge).
            root_quat_xyzw: optional ``(4,)`` xyzw root quaternion. The
                manager / planner publish wxyz; callers should convert
                before passing here. Defaults to identity (operator
                intent for the gantry profile keeps the pelvis upright).

        Returns:
            ``(64,)`` float64 token. When the recorder was started
            without ``--sonic-checkpoint`` this is the cached zero
            vector (the warning was printed once at startup).
        """
        if self._tokenizer is None:
            return self._zero_motion_token
        return self._tokenizer.encode(
            body_q_mj, root_rot_xyzw=root_quat_xyzw
        )

    def _encode_motion_token_from_snapshot(
        self,
        snap: dict,
    ) -> np.ndarray:
        """Encode a planner snapshot into a 64-D SONIC motion token.

        Subscribe-mode chokepoint. Builds the 680-D 10-frame future
        observation via :class:`X2EncoderObsBuilder` (real planner
        future, not freeze pose) and runs the SONIC encoder + FSQ on
        it. Token labels are byte-identical to what the deploy actor's
        internal encoder would emit from the same wire snapshot
        (Layer 3 byte-parity test asserts this).

        Falls back to :meth:`_encode_motion_token` when the planner
        future window is missing (planner not yet warm) or when the
        tokenizer was constructed without an obs builder.

        Args:
            snap: Planner snapshot from
                :meth:`_SubscribeModeState.snapshot`. Must contain
                ``body_pose_q_mj``, ``root_quat_xyzw``,
                ``joint_pos_mj_future``, ``root_quat_xyzw_future``.

        Returns:
            ``(64,)`` float64 token.
        """
        if self._tokenizer is None:
            return self._zero_motion_token
        if self._tokenizer.obs_builder is None:
            return self._encode_motion_token(
                snap["body_pose_q_mj"],
                root_quat_xyzw=snap.get("root_quat_xyzw"),
            )
        if (
            snap.get("body_pose_q_mj") is None
            or snap.get("root_quat_xyzw") is None
            or snap.get("joint_pos_mj_future") is None
            or snap.get("root_quat_xyzw_future") is None
        ):
            return self._encode_motion_token(
                snap.get("body_pose_q_mj"),
                root_quat_xyzw=snap.get("root_quat_xyzw"),
            )
        return self._tokenizer.encode_with_snapshot(snap)

    def _maybe_dump_recorder_obs(self, snap: dict) -> None:
        """Layer 3 probe: write one snap + builder obs to disk.

        Triggered exactly once, on the first subscribe-mode tick where
        ``snap`` is fully populated. The dump is consumed by
        :file:`gear_sonic_deploy/scripts/compare_recorder_vs_deploy_obs.py`
        which diffs it against the deploy's matching ``--obs-dump``
        blob (Layer 3 of the validation pyramid).

        Schema: a torch ``.pt`` file containing a dict with::

            {
              "kind": "x2_recorder_obs_dump_v1",
              "snap": {
                "body_pose_q_mj":         (31,) float64,
                "root_quat_xyzw":         (4,) float64,
                "joint_pos_mj_future":    (F, 31) float64,
                "root_quat_xyzw_future":  (F, 4) float64,
              },
              "encoder_obs":           (680,) float32,  # gather output
              "encoder_config":        path str,        # YAML used
              "checkpoint":            path str,        # .pt used
            }

        Silently no-ops when ``--obs-dump-recorder`` was not set,
        when the dump has already fired, when the planner future is
        not yet warm, or when the tokenizer has no obs_builder.
        """
        if self._obs_dump_recorder_done:
            return
        if self._cfg.obs_dump_recorder_path is None:
            return
        if self._tokenizer is None or self._tokenizer.obs_builder is None:
            return
        if (
            snap.get("body_pose_q_mj") is None
            or snap.get("root_quat_xyzw") is None
            or snap.get("joint_pos_mj_future") is None
            or snap.get("root_quat_xyzw_future") is None
        ):
            return

        try:
            import torch
            obs_680 = self._tokenizer.obs_builder.build_obs(snap)
            payload = {
                "kind": "x2_recorder_obs_dump_v1",
                "snap": {
                    "body_pose_q_mj": np.asarray(
                        snap["body_pose_q_mj"], dtype=np.float64
                    ),
                    "root_quat_xyzw": np.asarray(
                        snap["root_quat_xyzw"], dtype=np.float64
                    ),
                    "joint_pos_mj_future": np.asarray(
                        snap["joint_pos_mj_future"], dtype=np.float64
                    ),
                    "root_quat_xyzw_future": np.asarray(
                        snap["root_quat_xyzw_future"], dtype=np.float64
                    ),
                },
                "encoder_obs": obs_680.astype(np.float32, copy=False),
                "encoder_config": (
                    str(self._cfg.sonic_encoder_config)
                    if self._cfg.sonic_encoder_config is not None
                    else ""
                ),
                "checkpoint": (
                    str(self._cfg.sonic_checkpoint)
                    if self._cfg.sonic_checkpoint is not None
                    else ""
                ),
            }
            out_path = Path(self._cfg.obs_dump_recorder_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, str(out_path))
            print(
                f"[recorder] Layer 3 obs dump written to {out_path} "
                f"(680-D obs + snap; checkpoint+config recorded)",
                flush=True,
            )
        except Exception as exc:
            print(
                f"[recorder] WARNING: --obs-dump-recorder failed: "
                f"{exc!r}; continuing without dump",
                flush=True,
            )
        finally:
            self._obs_dump_recorder_done = True

    def _publish_pose(
        self,
        *,
        body_q_mj: np.ndarray,
        motion_token: np.ndarray,
        left_hand_q: np.ndarray,
        right_hand_q: np.ndarray,
        tick: int,
        root_quat_xyzw: Optional[np.ndarray] = None,
        root_xy_world: Optional[np.ndarray] = None,
        root_z_world: Optional[float] = None,
        joint_pos_mj_future: Optional[np.ndarray] = None,
        root_quat_xyzw_future: Optional[np.ndarray] = None,
        frame_index_future: Optional[np.ndarray] = None,
        future_dt_s: Optional[float] = None,
    ) -> None:
        """Publish a single reference frame to the deploy.

        When ``joint_pos_mj_future`` and ``root_quat_xyzw_future`` are
        both provided, this also emits the v5 future-window fields
        (``joint_vel_mj_future`` is recomputed by backward finite-diff
        over the supplied future window so it remains consistent with
        any arm-pose overlay the caller has applied). Without those
        future fields the deploy's ``ZmqPoseInputSource::Sample()``
        falls back to its single-frame v4 path -- which leaves the
        policy's 10-slot future window pinned at the current pose
        and prevents anticipatory locomotion thrust.
        """
        payload: dict[str, np.ndarray] = {
            "joint_pos_mj": body_q_mj.astype(np.float32),
            "root_quat_xyzw": (
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                if root_quat_xyzw is None
                else np.asarray(root_quat_xyzw, dtype=np.float32).reshape(4)
            ),
            "motion_token": motion_token.astype(np.float32),
            "left_hand_joints": left_hand_q.astype(np.float32),
            "right_hand_joints": right_hand_q.astype(np.float32),
            "frame_index": np.array([tick], dtype=np.int64),
        }

        # Pass through world-frame root pose when the upstream planner
        # publisher provides it (post-2026-06). The C++ deploy ignores
        # these keys; the kinematic viewer + Phase 2 PKL recorder use
        # them to render and store actual locomotion translation. Older
        # planners omit them and the merged stream simply falls back to
        # the pelvis-pinned legacy behaviour, which is wire-safe.
        if root_xy_world is not None:
            rxy = np.asarray(root_xy_world, dtype=np.float32).reshape(-1)
            if rxy.shape == (2,):
                payload["root_xy_world"] = rxy
        if root_z_world is not None:
            payload["root_z_world"] = np.array(
                [float(root_z_world)], dtype=np.float32
            )

        if (
            joint_pos_mj_future is not None
            and root_quat_xyzw_future is not None
        ):
            jpos_future = np.asarray(joint_pos_mj_future, dtype=np.float32)
            rot_future = np.asarray(root_quat_xyzw_future, dtype=np.float32)
            if (
                jpos_future.ndim == 2
                and jpos_future.shape[1] == NUM_BODY_DOFS
                and rot_future.shape == (jpos_future.shape[0], 4)
            ):
                dt = (
                    float(future_dt_s)
                    if future_dt_s is not None and future_dt_s > 1e-6
                    else 0.1
                )
                # Recompute joint_vel from finite-diff over the supplied
                # future window. This matches the planner's own scheme
                # (build_pose_payload in state_machine.py) and keeps the
                # arm-overlay slices internally consistent: when the
                # caller pinned arms across all future frames, the
                # corresponding arm dq is zero (correct), while leg dq
                # follows the planner's stride exactly.
                prev_jpos = payload["joint_pos_mj"][None, :]
                all_jpos = np.concatenate([prev_jpos, jpos_future], axis=0)
                jvel_future = (
                    (all_jpos[1:] - all_jpos[:-1]) / dt
                ).astype(np.float32)

                payload["joint_pos_mj_future"] = jpos_future
                payload["root_quat_xyzw_future"] = rot_future
                payload["joint_vel_mj_future"] = jvel_future
                if frame_index_future is not None:
                    fidx = np.asarray(frame_index_future, dtype=np.int64).reshape(-1)
                    if fidx.shape == (jpos_future.shape[0],):
                        payload["frame_index_future"] = fidx
                payload["future_dt_s"] = np.array([dt], dtype=np.float32)

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
        motion. Sending the trained stand pose with a zero
        ``motion_token`` is exactly the wire format
        :mod:`gear_sonic.scripts.mock_vla_publish_stand_token` uses to
        keep the X2 standing on the gantry indefinitely.

        **Root-quat hygiene.** Historically this method left
        ``root_quat_xyzw`` unset, which made :meth:`_publish_pose`
        default to identity (``R_z(0)`` = facing world +X). On real
        robot that's a startup waist-yaw click: the recorder publishes
        identity-quat idle frames in the gap between the pose proxy's
        yaw-rebased idle (correct) and the kplanner's first
        measured-yaw-seeded warmup (also correct), so the SONIC policy
        briefly tries to twist the body back to world +X via
        ``waist_yaw_joint`` (slot 12 of the 31-DOF vector and the
        dominant heading-correction effector). The proxy was patched
        2026-06-01 and the kplanner warmup was patched the same day;
        this method was the last remaining identity-quat publisher in
        the chain.

        Fix: yaw-project the live ``x2_debug`` ``base_quat`` (already
        cached in ``self._latest_state``) to a pure ``R_z(yaw)`` quat
        and pass that as the idle frame's reference. Pitch / roll are
        deliberately dropped so a momentary lean / fall-pose doesn't
        bleed into the upright training distribution. Falls back to
        identity (the original behaviour) when ``x2_debug`` has gone
        silent / never arrived, never regressing relative to the prior
        wire shape.
        """
        body = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
        zero_hand = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
        root_quat_xyzw = self._compute_idle_root_quat_xyzw()
        self._publish_pose(
            body_q_mj=body,
            motion_token=self._zero_motion_token,
            left_hand_q=zero_hand,
            right_hand_q=zero_hand,
            tick=-1,
            root_quat_xyzw=root_quat_xyzw,
        )

    def _compute_idle_root_quat_xyzw(self) -> Optional[np.ndarray]:
        """Build the yaw-rebased idle root_quat (or None for identity).

        Returns a length-4 xyzw quat representing ``R_z(measured_yaw)``
        when ``x2_debug`` is alive, else ``None`` so :meth:`_publish_pose`
        falls back to identity (the pre-2026-06-01 wire behaviour).
        Logging is one-way-gated by ``_idle_yaw_rebase_logged_active``
        / ``_idle_yaw_rebase_logged_fallback`` so a 50 Hz idle publish
        loop emits at most two lines, not 100/s.
        """
        try:
            (
                _body_q,
                base_quat_wxyz,
                _lh,
                _rh,
                _rev,
                alive,
            ) = self._latest_state.snapshot()
        except Exception as exc:  # noqa: BLE001 - defensive at the wire
            if self._cfg.verbose:
                print(
                    f"[recorder] idle yaw-rebase: snapshot failed ({exc!r}); "
                    f"falling back to identity root_quat",
                    flush=True,
                )
            return None

        if not alive:
            # x2_debug has never arrived or has gone stale (deploy not
            # up yet on first boot, or deploy quit). Identity is the
            # safest fallback -- matches the pre-2026-06-01 behaviour.
            if (
                self._idle_yaw_rebase_logged_active
                and not self._idle_yaw_rebase_logged_fallback
            ):
                print(
                    "[recorder] idle yaw-rebase: x2_debug went stale; "
                    "falling back to identity root_quat (waist-yaw click "
                    "may resume until deploy comes back)",
                    flush=True,
                )
                self._idle_yaw_rebase_logged_fallback = True
                # Drop the active sticky so the next live tick re-logs
                # an ACTIVE line -- gives the operator a visible
                # recovery marker without spamming the boot gap.
                self._idle_yaw_rebase_logged_active = False
            return None

        try:
            yaw = float(yaw_of_quat_xyzw(np.array(
                [base_quat_wxyz[1], base_quat_wxyz[2],
                 base_quat_wxyz[3], base_quat_wxyz[0]],
                dtype=np.float64,
            )))
        except (ValueError, TypeError) as exc:
            if self._cfg.verbose:
                print(
                    f"[recorder] idle yaw-rebase: yaw extraction failed "
                    f"({exc!r}); falling back to identity root_quat",
                    flush=True,
                )
            return None

        half = 0.5 * yaw
        quat_xyzw = np.array(
            [0.0, 0.0, math.sin(half), math.cos(half)],
            dtype=np.float32,
        )

        if not self._idle_yaw_rebase_logged_active:
            print(
                f"[recorder] idle yaw-rebase: ACTIVE -- root_quat_xyzw "
                f"now derived from live x2_debug base_quat "
                f"(yaw={math.degrees(yaw):+.2f}deg); waist-yaw click "
                f"protection on",
                flush=True,
            )
            self._idle_yaw_rebase_logged_active = True
            # If we previously logged a fallback, allow the next stale
            # transition to log again so the operator can correlate
            # recoveries.
            self._idle_yaw_rebase_logged_fallback = False

        return quat_xyzw

    # -- gesture playback (subscribe-mode override path) ---------------------

    def _snapshot_robot_yaw(self, snap: dict) -> float:
        """Best-effort estimate of the robot's current world-frame yaw.

        Used as the rebase target for the PKL's frame-0 yaw so the
        gesture starts at the operator's current heading. Falls back
        to 0 rad when the body_pose snapshot has no root quat yet
        (e.g. gesture triggered before kplanner publishes anything).
        """
        rq = snap.get("root_quat_xyzw")
        if rq is None:
            return 0.0
        try:
            return yaw_of_quat_xyzw(np.asarray(rq, dtype=np.float64))
        except (ValueError, TypeError):
            return 0.0

    def _resolve_gesture_entry(
        self, req: GesturePlayRequest
    ) -> Optional[GestureCatalogEntry]:
        """Resolve a play request to a catalog entry (or ad-hoc one).

        Returns ``None`` on unknown name with an operator log. Per-
        request ``motion_key`` / ``start_frame`` / ``n_frames`` fields
        override the catalog entry's defaults when provided.
        """
        if req.pkl_path is not None:
            return GestureCatalogEntry(
                name=f"adhoc:{req.pkl_path.name}",
                source=req.pkl_path,
                motion_key=req.motion_key,
                start_frame=req.start_frame,
                n_frames=req.n_frames,
            )
        name = req.name
        if not self._gesture_catalog:
            print(
                f"[recorder] gesture PLAY: catalog empty / unavailable; "
                f"cannot resolve name {name!r} (use --pkl for ad-hoc)",
                flush=True,
            )
            return None
        if name not in self._gesture_catalog:
            avail = ", ".join(list(self._gesture_catalog.keys())[:5])
            print(
                f"[recorder] gesture PLAY: unknown name {name!r}; "
                f"have {len(self._gesture_catalog)} entries (first few: {avail})",
                flush=True,
            )
            return None
        base = self._gesture_catalog[name]
        if (
            req.motion_key is None
            and req.start_frame == 0
            and req.n_frames is None
        ):
            return base
        # Per-request overrides win over catalog defaults.
        return GestureCatalogEntry(
            name=base.name,
            source=base.source,
            motion_key=(
                base.motion_key if req.motion_key is None else req.motion_key
            ),
            start_frame=(
                req.start_frame if req.start_frame else base.start_frame
            ),
            n_frames=(
                base.n_frames if req.n_frames is None else req.n_frames
            ),
        )

    def _drain_gesture_commands(self, snap: dict) -> None:
        """Apply queued gesture_cmd requests at the top of a publish tick.

        Multiple commands in one tick are applied in order; a `play`
        after another `play` supersedes the in-flight session at
        frame 0 of the new clip. `stop` always wins regardless of
        what arrived after it (the queue is drained FIFO so any post-
        stop play takes effect on subsequent ticks, which matches
        operator intuition: 'stop, then play X' acts like 'play X').

        ``stop`` also releases a held-final-frame state
        (:attr:`_gesture_held_frame`), so the same wire command covers
        both "abort mid-clip" and "release from indefinite hold". When
        a new ``play`` arrives during a held state, the new session's
        yaw rebase seeds off the held root_quat (instead of the
        kplanner body_pose snap) so the takeover between two halves
        of the same source PKL stays continuous.
        """
        if self._gesture_request_queue is None:
            return
        while True:
            try:
                req = self._gesture_request_queue.get_nowait()
            except queue.Empty:
                return
            if isinstance(req, GestureStopRequest):
                if self._active_gesture is not None:
                    print(
                        f"[recorder] gesture STOP (was playing "
                        f"{self._active_gesture.entry.name!r} at frame "
                        f"{self._active_gesture.current_index}/"
                        f"{self._active_gesture.n_frames})",
                        flush=True,
                    )
                elif self._gesture_held_frame is not None:
                    print(
                        "[recorder] gesture STOP (releasing held pose; "
                        "resuming kplanner forwarding)",
                        flush=True,
                    )
                self._active_gesture = None
                self._active_gesture_hold_after = False
                self._gesture_held_frame = None
                continue
            entry = self._resolve_gesture_entry(req)
            if entry is None:
                continue
            # Yaw seed: when superseding a held pose, take the held
            # root_quat's yaw rather than the kplanner body_pose snap
            # (kplanner has no idea we're holding; using its idle-stand
            # snap would re-rotate the world at takeover and create a
            # visible body twist between two halves of the same source
            # PKL).
            if self._gesture_held_frame is not None:
                yaw = float(
                    yaw_of_quat_xyzw(
                        np.asarray(
                            self._gesture_held_frame["root_quat_xyzw"],
                            dtype=np.float64,
                        )
                    )
                )
                yaw_source = "held-frame"
            else:
                yaw = self._snapshot_robot_yaw(snap)
                yaw_source = "kplanner-snap"
            try:
                self._active_gesture = GestureSession(
                    entry=entry,
                    target_rate_hz=float(self._cfg.publish_rate_hz),
                    robot_root_yaw_rad=yaw,
                    future_dt_s=float(self._cfg.gesture_future_dt_s),
                )
            except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
                print(
                    f"[recorder] gesture PLAY failed for {entry.name!r}: "
                    f"{exc}",
                    flush=True,
                )
                self._active_gesture = None
                self._active_gesture_hold_after = False
                self._gesture_held_frame = None
                continue
            # Resolve the effective hold_after: per-request wire
            # override wins, else the catalog default. Ad-hoc --pkl
            # entries default to False because they're constructed
            # without a catalog row.
            if req.hold_after is not None:
                effective_hold_after = bool(req.hold_after)
            else:
                effective_hold_after = bool(entry.hold_after)
            self._active_gesture_hold_after = effective_hold_after
            # Starting a new play always clears the previous held
            # frame: the new session owns the publish path.
            self._gesture_held_frame = None
            hold_tag = " (will HOLD on completion)" if effective_hold_after else ""
            print(
                f"[recorder] gesture PLAY {entry.name!r}: "
                f"{self._active_gesture.n_frames} frames @ "
                f"{self._cfg.publish_rate_hz} Hz "
                f"(~{self._active_gesture.duration_s:.1f}s) "
                f"rebased_yaw={np.degrees(yaw):.1f}deg [{yaw_source}]"
                f"{hold_tag}",
                flush=True,
            )

    def _publish_gesture_frame(self, *, tick: int) -> None:
        """Emit one gesture frame on the deploy ``pose`` topic.

        Bypasses the kplanner body_pose + manager arm/hand merge path:
        the PKL frames carry the full 31-DOF body, hands go to zero,
        motion_token to zero. v5 future window is synthesized from
        the session's resampled buffer with the operator-configured
        spacing.
        """
        assert self._active_gesture is not None  # noqa: S101 -- gated by caller
        body, root_quat = self._active_gesture.next_frame()
        zero_hand = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
        n_future = int(self._cfg.gesture_future_window_frames)
        jpos_future, rot_future = self._active_gesture.future_window(n_future)
        if n_future > 0:
            step_ticks = max(
                1,
                int(round(
                    self._cfg.gesture_future_dt_s
                    * self._cfg.publish_rate_hz
                )),
            )
            frame_idx_future = np.array(
                [tick + (k + 1) * step_ticks for k in range(n_future)],
                dtype=np.int64,
            )
        else:
            frame_idx_future = None
        self._publish_pose(
            body_q_mj=body.astype(np.float64),
            motion_token=self._zero_motion_token,
            left_hand_q=zero_hand,
            right_hand_q=zero_hand,
            tick=tick,
            root_quat_xyzw=root_quat,
            joint_pos_mj_future=jpos_future if n_future > 0 else None,
            root_quat_xyzw_future=rot_future if n_future > 0 else None,
            frame_index_future=frame_idx_future,
            future_dt_s=float(self._cfg.gesture_future_dt_s),
        )

    def _publish_held_gesture_frame(self, *, tick: int) -> None:
        """Re-emit the latched final gesture frame for one publish tick.

        Engaged after a ``hold_after=True`` gesture finishes its clip;
        the recorder parks the robot at the gesture's last body_q +
        root_quat (zero hands, zero motion_token) every tick at the
        configured publish rate until either an explicit ``stop``
        clears the latch or a new ``play`` takes over the publish path.

        The strictly-future window is filled with the same last frame
        on every slot -- the deploy's tokenizer expects ``(n, 31)`` /
        ``(n, 4)`` arrays even when there's no upcoming motion, and
        repeating the held pose is the natural "no future motion"
        signal.
        """
        assert self._gesture_held_frame is not None  # noqa: S101 -- gated by caller
        body = self._gesture_held_frame["body_q_mj"]
        root_quat = self._gesture_held_frame["root_quat_xyzw"]
        zero_hand = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
        n_future = int(self._cfg.gesture_future_window_frames)
        if n_future > 0:
            jpos_future = np.broadcast_to(
                body.astype(np.float32, copy=False),
                (n_future, body.shape[0]),
            ).copy()
            rot_future = np.broadcast_to(
                root_quat.astype(np.float32, copy=False), (n_future, 4),
            ).copy()
            step_ticks = max(
                1,
                int(round(
                    self._cfg.gesture_future_dt_s
                    * self._cfg.publish_rate_hz
                )),
            )
            frame_idx_future = np.array(
                [tick + (k + 1) * step_ticks for k in range(n_future)],
                dtype=np.int64,
            )
        else:
            jpos_future = None
            rot_future = None
            frame_idx_future = None
        self._publish_pose(
            body_q_mj=body.astype(np.float64),
            motion_token=self._zero_motion_token,
            left_hand_q=zero_hand,
            right_hand_q=zero_hand,
            tick=tick,
            root_quat_xyzw=root_quat,
            joint_pos_mj_future=jpos_future,
            root_quat_xyzw_future=rot_future,
            frame_index_future=frame_idx_future,
            future_dt_s=float(self._cfg.gesture_future_dt_s),
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
        #
        # ``root_pos_xyz`` comes from the bridge's ``robot_pose`` PUB
        # (cached by :meth:`_robot_pose_subscriber`). The head-mounted
        # ego_view is rigidly attached to the robot so its visual
        # output is invariant under root translation -- but the
        # world-fixed ``front_cam`` is not: it needs the live root pos
        # to actually see the robot move. We pass the same value to
        # both renderers so any future debug overlays stay consistent.
        pelvis_xyz = self._snapshot_pelvis_xyz()
        try:
            ego_view = self._renderer.render_frame(  # type: ignore[union-attr]
                body_q=obs_body_q_mj,
                left_active=obs_left_hand_q.astype(np.float64),
                right_active=obs_right_hand_q.astype(np.float64),
                root_pos_xyz=pelvis_xyz,
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

        # Optional second view: the world-fixed wide-angle ``front_cam``
        # baked into the robocasa scene XMLs. Renders the SAME body_q +
        # base_quat + pelvis_xyz from a different MJCF camera, so the
        # two video tracks are bit-for-bit synchronized at the proprio
        # level. Render failures here are non-fatal: we drop the whole
        # frame (rather than write a partial one with a missing
        # ``observation.images.front_cam`` key) so the LeRobot
        # exporter's ``validate_frame`` strict-schema check stays
        # happy. ``front_view`` lives in local scope here and is
        # spliced into ``frame_data`` just below.
        front_view: np.ndarray | None = None
        if self._front_cam_renderer is not None:
            try:
                front_view = self._front_cam_renderer.render_frame(
                    body_q=obs_body_q_mj,
                    left_active=obs_left_hand_q.astype(np.float64),
                    right_active=obs_right_hand_q.astype(np.float64),
                    root_pos_xyz=pelvis_xyz,
                    root_quat_wxyz=obs_base_quat_wxyz,
                )
            except Exception as exc:
                print(
                    f"[recorder] front_cam render warn (frame skipped): "
                    f"{exc}",
                    flush=True,
                )
                return
            front_view = np.ascontiguousarray(front_view, dtype=np.uint8)
            if front_view.shape != (
                self._cfg.render_height, self._cfg.render_width, 3,
            ):
                print(
                    f"[recorder] WARN: front_cam shape {front_view.shape} "
                    f"!= ({self._cfg.render_height}, "
                    f"{self._cfg.render_width}, 3); skipping frame",
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

        # Pull the real head-camera bundle (Orbbec + IMX900 stereo pair)
        # BEFORE we assemble the frame_data dict so we can fail-fast and
        # skip the tick if the bridge is stale -- writing a partial
        # frame would trip ``Gr00tDataExporter.validate_frame`` and
        # crash the recording session. The helper returns None when
        # head cameras are disabled (no-op for non-cam recordings) OR
        # when the bridge is silent / stale.
        head_cam_frame_data: dict[str, np.ndarray] = {}
        if self._cfg.record_head_cameras:
            head_bundle = self._snapshot_head_camera_images()
            if head_bundle is None:
                # Stale / missing bundle. We've already logged a
                # rate-limited warning inside the snapshot helper;
                # just drop the tick here so the parquet stays
                # schema-complete. Resumes automatically on the next
                # tick once the bridge catches up.
                return
            head_cam_frame_data = self._format_head_camera_frame_data(head_bundle)

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
        if front_view is not None:
            # Schema parity with :func:`get_features_x2_vla`: the key
            # exists in ``self._features`` iff
            # ``cfg.record_front_cam=True``. Conditionally emitting it
            # here mirrors the conditional render above; the exporter
            # rejects either side mismatching at validate time.
            frame_data["observation.images.front_cam"] = front_view
        # Splice in the three real head-camera streams. The keys map
        # 1:1 with ``observation.images.{head_front,stereo_left,stereo_right}``
        # so a missing key here means the bridge missed that mount
        # this tick -- the staleness check above already guarantees
        # the bundle was complete + fresh, so this is a noop on the
        # happy path.
        if head_cam_frame_data:
            frame_data.update(head_cam_frame_data)
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

    def _robot_pose_subscriber(self) -> None:
        """Background SUB on the bridge's ``robot_pose`` topic.

        Updates :attr:`_latest_pelvis_xyz` whenever a fresh packet
        arrives so :meth:`_record_frame` can pass the live pelvis
        ``(x, y, z)`` into :meth:`MujocoFrameRenderer.render_frame` as
        ``root_pos_xyz``. Bridge publishes at state-rate (default
        200 Hz); we use ``CONFLATE=1`` to avoid backlog. The orientation
        component (``[3:7]``) is currently ignored here -- the recorder
        already gets that via the ``x2_debug`` ``base_quat`` field
        (cached in :class:`_LatestState`) and the two paths must agree
        on the same fresh ``base_quat`` for the rendered ``ego_view`` /
        ``front_cam`` to look right.

        No-op when the bridge isn't running (recv just times out
        forever); the recorder stays alive on the renderer's hardcoded
        ``(0, 0, 0.793)`` fallback.
        """
        try:
            from gear_sonic.utils.teleop.zmq.robot_pose_zmq import (
                unpack_robot_pose,
            )
        except ImportError as exc:
            print(f"[recorder] robot_pose SUB import failed: {exc}",
                  flush=True)
            return
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(
            zmq.SUBSCRIBE, self._cfg.robot_pose_sub_topic.encode()
        )
        sock.setsockopt(zmq.RCVTIMEO, 200)
        endpoint = (
            f"tcp://{self._cfg.robot_pose_sub_host}:"
            f"{self._cfg.robot_pose_sub_port}"
        )
        try:
            sock.connect(endpoint)
        except Exception as exc:
            print(f"[recorder] robot_pose SUB connect to {endpoint} "
                  f"failed: {exc}", flush=True)
            sock.close(linger=0)
            return
        print(
            f"[recorder] robot_pose SUB connected at {endpoint} "
            f"(topic={self._cfg.robot_pose_sub_topic!r})",
            flush=True,
        )
        first = True
        while not self._stop_event.is_set():
            try:
                raw = sock.recv()
            except zmq.error.Again:
                continue
            except zmq.error.ContextTerminated:
                break
            except Exception as exc:
                print(f"[recorder] robot_pose SUB recv error: {exc}",
                      flush=True)
                continue
            try:
                payload = unpack_robot_pose(raw)
            except ValueError as exc:
                print(f"[recorder] robot_pose decode error: {exc}",
                      flush=True)
                continue
            qpos = payload.get("pelvis_qpos_wxyz")
            if not isinstance(qpos, list) or len(qpos) < 3:
                continue
            xyz = np.array(qpos[0:3], dtype=np.float64)
            with self._pelvis_pose_lock:
                self._latest_pelvis_xyz = xyz
                self._pelvis_pose_seen_any = True
            if first:
                first = False
                print(
                    f"[recorder] robot_pose SUB: first packet "
                    f"(pelvis_xyz={xyz.tolist()})",
                    flush=True,
                )
        try:
            sock.close(linger=0)
        except Exception:
            pass

    def _snapshot_pelvis_xyz(self) -> np.ndarray:
        """Return the latest cached pelvis ``(x, y, z)``.

        Returns a copy so callers can pass it straight into the
        renderer without worrying about the SUB thread overwriting it
        mid-render. Falls back to the renderer's default
        ``(0, 0, 0.793)`` when no ``robot_pose`` packet has arrived.
        """
        with self._pelvis_pose_lock:
            return self._latest_pelvis_xyz.copy()

    # -- PC2 head-camera ingestion ------------------------------------------

    def _init_head_cameras(self) -> None:
        """Bootstrap the head-camera ZMQ SUB and verify the bridge is live.

        Called once from ``__init__`` when ``cfg.record_head_cameras`` is
        set. Constructs a :class:`ComposedCameraClientSensor`, drains it
        for up to :attr:`RecorderConfig.camera_warmup_timeout_s` seconds
        waiting for the first frame bundle that contains all three
        expected mount keys (``head_front`` + ``stereo_left`` +
        ``stereo_right``), and then kicks off a daemon poller. Failing
        the warmup check is a fail-fast: better to abort here than to
        write a half-populated parquet shard the trainer rejects.
        """
        from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
        from gear_sonic.data.features_x2_vla import (
            HEAD_CAM_HEIGHT,
            HEAD_CAM_KEYS,
            HEAD_CAM_WIDTH,
        )

        cfg = self._cfg
        print(
            f"[recorder] head cameras ENABLED -> connecting to "
            f"tcp://{cfg.camera_host}:{cfg.camera_port}",
            flush=True,
        )
        self._head_camera_client = ComposedCameraClientSensor(
            server_ip=cfg.camera_host, port=cfg.camera_port
        )

        # Warmup: pull until we either see a complete bundle or hit the
        # operator-configured timeout. Discarded warmup bundles also
        # double as the publisher-rate sanity check (we report what we
        # saw).
        deadline = time.monotonic() + max(cfg.camera_warmup_timeout_s, 0.0)
        last_seen_keys: set[str] = set()
        last_print = 0.0
        required = set(HEAD_CAM_KEYS)
        ready = False
        while time.monotonic() < deadline:
            msg = self._head_camera_client.read(blocking=False)
            if msg and msg.get("images"):
                got_keys = {k for k, v in msg["images"].items() if v is not None}
                if got_keys >= required:
                    sample = {
                        k: msg["images"][k].shape for k in HEAD_CAM_KEYS
                    }
                    print(
                        f"[recorder] head-camera bridge ready: "
                        f"keys={sorted(got_keys)} shapes={sample}",
                        flush=True,
                    )
                    ready = True
                    break
                if got_keys != last_seen_keys and time.monotonic() - last_print > 1.0:
                    print(
                        f"[recorder] head cameras: waiting for full bundle "
                        f"(got {sorted(got_keys)}, need {sorted(required)})",
                        flush=True,
                    )
                    last_seen_keys = got_keys
                    last_print = time.monotonic()
            time.sleep(0.05)

        if not ready:
            try:
                self._head_camera_client.close()
            except Exception:
                pass
            self._head_camera_client = None
            raise RuntimeError(
                f"[recorder] head-camera bridge at "
                f"tcp://{cfg.camera_host}:{cfg.camera_port} did NOT publish a "
                f"complete frame bundle within {cfg.camera_warmup_timeout_s:.1f}s "
                f"(required mount keys: {sorted(required)}). Start the bridge "
                f"with `./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve` "
                f"and confirm `status` shows publishers for "
                f"rgb_head_front_center + stereo_head_front_{{left,right}}."
            )

        # Sanity-check the bridge's output dimensions match the schema.
        # The bridge is expected to resize to (HEAD_CAM_WIDTH, HEAD_CAM_HEIGHT)
        # so the recorder can skip a resize per tick.
        for k in HEAD_CAM_KEYS:
            shp = msg["images"][k].shape  # type: ignore[index]
            if shp[:2] != (HEAD_CAM_HEIGHT, HEAD_CAM_WIDTH):
                print(
                    f"[recorder] WARN: head camera {k!r} produces "
                    f"{shp[1]}x{shp[0]} (HxW={shp[:2]}); expected "
                    f"{HEAD_CAM_WIDTH}x{HEAD_CAM_HEIGHT}. Re-encoding per tick "
                    f"to the schema size (extra CPU cost). Run the bridge with "
                    f"--width {HEAD_CAM_WIDTH} --height {HEAD_CAM_HEIGHT} to "
                    f"eliminate this.",
                    flush=True,
                )

        # Seed cache with the warmup bundle so the very first
        # ``_record_frame`` call has something to write even before the
        # poller wakes up.
        with self._head_camera_lock:
            self._head_camera_latest = {
                k: msg["images"][k] for k in HEAD_CAM_KEYS  # type: ignore[index]
            }
            self._head_camera_latest_ts = time.time()
            self._head_camera_frames_received = 1

        self._head_camera_thread = threading.Thread(
            target=self._head_camera_subscriber,
            name="head-camera-sub",
            daemon=True,
        )

    def _head_camera_subscriber(self) -> None:
        """Background poller for the PC2 camera bridge.

        Pulls merged frames as fast as the wire delivers them and keeps
        the latest complete bundle in ``self._head_camera_latest`` for
        :meth:`_snapshot_head_camera_images`. The poll period is set to
        a fraction of the record-tick interval so the cache is always
        fresher than the recorder's next read.
        """
        from gear_sonic.data.features_x2_vla import HEAD_CAM_KEYS

        client = self._head_camera_client
        if client is None:
            return
        required = set(HEAD_CAM_KEYS)
        # Poll at 4x the record rate so the cache is fresher than the
        # next tick read by definition; cap at 200 Hz.
        record_period = 1.0 / max(self._cfg.record_rate_hz, 1.0)
        poll_period = min(record_period / 4.0, 1.0 / 200.0)
        while not self._stop_event.is_set():
            try:
                msg = client.read(blocking=False)
            except Exception as exc:
                print(
                    f"[recorder] head-camera poll error: {exc}",
                    flush=True,
                )
                time.sleep(0.1)
                continue
            if msg and msg.get("images"):
                got = {
                    k: v
                    for k, v in msg["images"].items()
                    if v is not None and k in required
                }
                if set(got.keys()) >= required:
                    with self._head_camera_lock:
                        self._head_camera_latest = got
                        self._head_camera_latest_ts = time.time()
                        self._head_camera_frames_received += 1
            time.sleep(poll_period)

    def _snapshot_head_camera_images(self) -> Optional[dict[str, np.ndarray]]:
        """Return the latest complete head-camera bundle if fresh enough.

        Returns ``None`` when:
        * head cameras are disabled, OR
        * no bundle has arrived yet, OR
        * the latest bundle is older than
          :attr:`RecorderConfig.camera_max_staleness_s`.

        The caller is expected to skip the frame (no parquet write) in
        all three cases. A throttled warning fires the first time we
        drop a frame for staleness so silent freezes are obvious.
        """
        if not self._cfg.record_head_cameras or self._head_camera_client is None:
            return None
        with self._head_camera_lock:
            latest = self._head_camera_latest
            ts = self._head_camera_latest_ts
        now = time.time()
        if latest is None:
            return None
        if now - ts > self._cfg.camera_max_staleness_s:
            # Rate-limit the warning to once per second; the dropped
            # frames are otherwise silent so the recorder doesn't spam
            # the terminal during a long bridge hiccup.
            if (
                self._head_camera_stale_warns == 0
                or now - getattr(self, "_last_cam_warn_t", 0.0) > 1.0
            ):
                print(
                    f"[recorder] WARN: head-camera bundle is "
                    f"{(now - ts)*1000:.0f}ms stale (>"
                    f"{self._cfg.camera_max_staleness_s*1000:.0f}ms); "
                    f"skipping frame. Check the PC2 bridge is alive.",
                    flush=True,
                )
                self._head_camera_stale_warns += 1
                self._last_cam_warn_t = now
            return None
        # Return a shallow copy so the recorder can mutate dtype/shape
        # without affecting the cache.
        return dict(latest)

    def _format_head_camera_frame_data(
        self,
        bundle: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Normalize a head-camera bundle into the LeRobot-frame dict shape.

        The bridge already publishes uint8 RGB at (HEAD_CAM_HEIGHT,
        HEAD_CAM_WIDTH, 3) so this is usually just a key-rename
        (``head_front`` -> ``observation.images.head_front``) + a
        contiguity guarantee. Frames that arrive at the wrong size are
        resized here (with a one-shot warning logged at warmup); frames
        already at the right size pay only the ``ascontiguousarray``
        cost.
        """
        from gear_sonic.data.features_x2_vla import (
            HEAD_CAM_HEIGHT,
            HEAD_CAM_KEYS,
            HEAD_CAM_WIDTH,
        )
        import cv2 as _cv2

        out: dict[str, np.ndarray] = {}
        for cam_key in HEAD_CAM_KEYS:
            img = bundle.get(cam_key)
            if img is None:
                continue
            if img.shape[:2] != (HEAD_CAM_HEIGHT, HEAD_CAM_WIDTH):
                img = _cv2.resize(
                    img,
                    (HEAD_CAM_WIDTH, HEAD_CAM_HEIGHT),
                    interpolation=_cv2.INTER_AREA,
                )
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
            out[f"observation.images.{cam_key}"] = np.ascontiguousarray(img)
        return out

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
        # Optional second renderer for the wide-angle world-fixed
        # ``front_cam`` baked into the robocasa scene XMLs. Spinning up
        # a second :class:`MujocoFrameRenderer` keeps the per-camera
        # render path completely independent (separate ``mjData``,
        # separate EGL framebuffer) -- ~20-40 MB extra GPU/CPU memory
        # and one extra ``render()`` call per tick (sub-millisecond at
        # 640x480 on a modern GPU). Only built when (a) the operator
        # asked for it AND (b) we're in scene mode (the camera doesn't
        # exist in the legacy flat-floor MJCF). See
        # ``RecorderConfig.record_front_cam``.
        if self._cfg.record_front_cam and scene_xml_path is not None:
            print("[recorder] building MuJoCo front_cam renderer …", flush=True)
            self._front_cam_renderer = MujocoFrameRenderer(
                camera="front_cam",
                width=self._cfg.render_width,
                height=self._cfg.render_height,
                with_omnihand=self._cfg.with_omnihand,
                egl=True,
                scene_xml_path=scene_xml_path,
            )
            print(
                f"[recorder] front_cam renderer ready "
                f"({self._front_cam_renderer.width}x"
                f"{self._front_cam_renderer.height})",
                flush=True,
            )
        elif self._cfg.record_front_cam:
            # Operator opted in but we're not in scene mode -> the
            # ``front_cam`` camera does not exist in the legacy MJCF.
            # Don't blow up; just warn loudly so the missing video
            # track in the dataset is obvious during triage.
            print(
                "[recorder] WARN: record_front_cam=True but "
                "scene_xml_path is None -- the legacy flat-floor MJCF "
                "has no 'front_cam' camera. Skipping the second "
                "renderer; dataset will only contain ego_view.",
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

    def _log_hand_diag(self) -> None:
        """Per-side hand diagnostic: raw vs filtered curls + final hand_q.

        Throttled by the same ``status_log_period_s`` cadence as
        :meth:`_print_status` (default 5 s) so it doesn't spam the
        terminal at 50 Hz. Useful for "thumb won't close" /
        "controller trigger ignored" investigations because it makes
        the per-tick dispatch path explicit:

          * ``src``      -- which Quest 3 source is feeding the curls
                            (xrhand, controller, multimodal). When the
                            headset reports multimodal we **always**
                            take the xrhand path -- the controller
                            triggers are silently ignored. That's by
                            design but trips up operators who think
                            the trigger should still close fingers.
          * ``raw``      -- the unfiltered Quest 3 thumb_flex + oppose
                            values. If raw thumb_flex maxes out around
                            0.10-0.20 then the calibration floor (0.198
                            in the bundled default) is clamping it to
                            zero before per-finger retargeting ever
                            sees it.
          * ``filt``     -- post FingerSignalFilter. If raw is healthy
                            but filt is zero, the filter's smoothing
                            constants are eating the signal.
          * ``hand_q``   -- the final 10-D command published to the
                            bridge. Indices 0/1/2 are
                            thumb_roll / thumb_abad / thumb_mcp -- the
                            three actuators that drive the thumb's
                            "close into palm" motion. If these are
                            non-zero in the recorder log but the robot
                            doesn't move, the bug is in the bridge
                            (actuator routing, contact gate, etc) not
                            the recorder.

        We only print when at least one tick has populated the diag
        snapshot, and we keep the line under 200 cols so it survives
        operator log scraping.
        """
        diag = self._last_hand_diag
        if diag is None:
            return

        def _fmt_arr(a: Any, n: int = 5) -> str:
            if a is None:
                return "None"
            arr = np.asarray(a)
            if arr.size < n:
                n = int(arr.size)
            parts = ",".join(f"{float(arr[i]):+.2f}" for i in range(n))
            return f"[{parts}]"

        def _fmt_scalar(v: Any) -> str:
            if v is None:
                return " None"
            return f"{float(v):+.2f}"

        triggers = diag.get("triggers")
        if triggers is None:
            tr_str = "trig=None"
        else:
            t0, t1, g0, g1 = triggers
            tr_str = (
                f"trig=L_t{float(t0):+.2f}/g{float(g0):+.2f} "
                f"R_t{float(t1):+.2f}/g{float(g1):+.2f}"
            )

        def _safe_str(v: Any, default: str = "None") -> str:
            # Quest3Reader can return ``None`` for the per-side source
            # tag when no XR hand input is bound yet (controller-only,
            # or before the first hand-tracking frame), and any other
            # diag field can race during shutdown. Coerce so the
            # f-string format spec ``{:<11s}`` doesn't blow up.
            return default if v is None else str(v)

        for side, hand_key in (("l", "left_hand_q"), ("r", "right_hand_q")):
            label = "LEFT " if side == "l" else "RIGHT"
            src = _safe_str(diag.get(f"{side}_src"))
            dispatch = _safe_str(diag.get(f"{side}_dispatch"), default="?")
            curls_raw = diag.get(f"{side}_curls_raw")
            curls_filt = diag.get(f"{side}_curls_filt")
            opp_raw = diag.get(f"{side}_oppose_raw")
            opp_filt = diag.get(f"{side}_oppose_filt")
            hand_q = diag.get(hand_key)
            print(
                f"[hand-diag] {label} src={src:<11s} dispatch={dispatch:<10s} "
                f"curls_raw={_fmt_arr(curls_raw)} curls_filt={_fmt_arr(curls_filt)} "
                f"oppose_raw={_fmt_scalar(opp_raw)} oppose_filt={_fmt_scalar(opp_filt)} "
                f"hand_q[0:3]={_fmt_arr(hand_q, 3)} "
                f"hand_q[3:5]={_fmt_arr(hand_q[3:5] if hand_q is not None else None, 2)} "
                f"hand_q[5:10]={_fmt_arr(hand_q[5:10] if hand_q is not None else None, 5)}",
                flush=True,
            )
        engaged = diag.get("engaged", False)
        print(
            f"[hand-diag] {tr_str} engaged={engaged} "
            f"compensation curl={self._cfg.apply_curl_compensation} "
            f"oppose={self._cfg.apply_oppose_compensation}",
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
