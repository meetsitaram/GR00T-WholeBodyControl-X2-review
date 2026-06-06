"""CLI entry-point for the Quest 3 → X2 LeRobot dataset recorder.

Reads Quest 3 controller motion + face buttons, runs DLS arm IK,
tokenizes the live body pose with the SONIC encoder + FSQ, publishes
motion tokens to the C++ deploy over ZMQ at 50 Hz, and writes a
LeRobot v2.1 dataset to ``--output-dir``.

Run this **after** starting the C++ deploy in VLA mode::

    deploy_x2.sh sim --vla --sim-profile gantry --sim-with-omnihand

…or co-launch both via :file:`record_x2_dataset.sh`.

Controls (Quest 3 controller buttons)
-------------------------------------

* **A** — engage / re-calibrate wrist anchors
* **B** — start a new episode
* **X** — stop and save the current episode
* **Y** — stop and discard the current episode

The script blocks until Ctrl-C; on shutdown it auto-saves any open
episode that the operator forgot to close with X.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# NOTE: ``X2DatasetRecorder`` transitively imports ``datasets`` (the
# Hugging Face dataset library, used by the LeRobot writer). On a
# fresh teleop venv that package may not be installed yet, so we
# defer the import until AFTER ``ensure_runtime_deps`` has had a
# chance to pip-install the recorder dependencies. The actual import
# happens inside ``main()`` -- see the deferred-import comment below.


def _resolve_front_cam_default(
    explicit: bool | None,
    scene_xml_path: Path | None,
) -> bool:
    """Resolve ``--front-cam`` / ``--no-front-cam`` to a ``bool``.

    Precedence:
      1. Explicit operator flag (``True`` / ``False``) wins.
      2. Otherwise default to ``True`` iff a scene XML is loaded
         (``front_cam`` only exists in the robocasa-built MJCFs --
         see ``_WORKSPACE_CAMERAS`` in
         ``gear_sonic/scripts/build_x2_robocasa_scene_xml.py``).

    Centralized here so the wrapper script
    (``run_x2_quest3_planner_stack.sh``) doesn't need to mirror the
    decision: passing ``--robocasa-env <env>`` alone is enough to
    light up both the scene XML and the second camera.
    """
    if explicit is not None:
        return bool(explicit)
    return scene_xml_path is not None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write the LeRobot v2.1 dataset into. "
             "Required unless --teleop-only is set.",
    )
    parser.add_argument(
        "--sonic-checkpoint", type=Path, default=None,
        help="SONIC tracker .pt checkpoint. When provided, the recorder "
             "loads OnlineSonicTokenizer once at startup and encodes the "
             "commanded body_q into action.motion_token in the LeRobot "
             "dataset every tick. Required for VLA training; without it "
             "the column is all zeros and the dataset is kinematic-only "
             "(intentional for smoke tests, never correct for production "
             "data collection). The wrapper "
             "run_x2_quest3_planner_stack.sh auto-resolves this from the "
             "deploy ONNX path.",
    )
    parser.add_argument(
        "--sonic-tokenizer-device", type=str, default="cuda:0",
        help="Torch device for the inline SONIC tokenizer ('cpu', "
             "'cuda:0', 'cuda:N'). Default cuda:0 keeps per-tick cost "
             "<100 us; cpu adds ~1 ms per tick (still well under the "
             "20 ms 50 Hz budget). Use cpu if cuda:0 is contended by "
             "the deploy / VLA on the same GPU.",
    )
    parser.add_argument(
        "--encoder-config", type=Path,
        default=Path("gear_sonic/data/encoder/x2_observation_config.yaml"),
        help="YAML encoder-observation config (G1-style schema). When "
             "set AND --body-pose-source=zmq (subscribe mode), the "
             "inline tokenizer builds the same 680-D 10-frame future "
             "observation the deploy actor's internal encoder consumes "
             "and runs the encoder on that exact obs. The default "
             "ships at gear_sonic/data/encoder/x2_observation_config"
             ".yaml. Pass --encoder-config '' to disable and fall "
             "back to the deprecated freeze-pose path (one body_q "
             "tiled 11 times -- semantically incorrect for VLA "
             "training, kept for backward compat with the v0 direct-"
             "mode loop).",
    )
    parser.add_argument(
        "--obs-dump-recorder", type=Path, default=None,
        help="Layer 3 byte-parity probe. When set, the subscribe-mode "
             "loop writes a torch .pt snapshot (snap dict + 680-D "
             "builder obs) on the first fully-populated tick to this "
             "path and continues running. Pair with the deploy's "
             "--obs-dump and run "
             "gear_sonic_deploy/scripts/compare_recorder_vs_deploy"
             "_obs.py to assert byte-equal observations between the "
             "Python gather and the C++ ZmqPoseInputSource.",
    )
    parser.add_argument(
        "--task", type=str, default="",
        help="Language instruction for every episode in this session "
             "(e.g. 'pick up the red block from the table'). "
             "Required unless --teleop-only is set.",
    )
    parser.add_argument(
        "--teleop-only", action="store_true",
        help="VR-drives-the-policy mode: publish motion tokens at 50 Hz "
             "and watch the SONIC + deploy follow them in MuJoCo, but do "
             "NOT build an exporter / renderer / write any dataset files. "
             "B/X/Y buttons become no-ops; A still engages IK calibration.",
    )
    parser.add_argument(
        "--no-idle-publish", action="store_true",
        help="Subscribe-mode only: while waiting for the first body_pose "
             "from the upstream planner / VLA bridge, do NOT publish the "
             "static DEFAULT_STAND_POSE on the pose wire. The deploy's "
             "ZmqPoseInputSource then never sees a frame, has_body_reference_ "
             "stays False, and Sample() falls back to its prefilled "
             "default_angles (the trained stand pose). Pair with the "
             "bridge's --silent-wire under --vla-no-policy to validate "
             "the 'no upstream' fallback path: wrist_bypass_ticks should "
             "stay at 0 for the whole run and grav_z should pin at -1.00.",
    )

    # ZMQ
    parser.add_argument("--pub-host", default="*", help="Bind iface for the pose PUB.")
    parser.add_argument("--pub-port", type=int, default=5556)
    parser.add_argument("--pub-topic", default="pose")
    parser.add_argument("--sub-host", default="localhost")
    parser.add_argument("--sub-port", type=int, default=5557)
    parser.add_argument("--sub-topic", default="x2_debug")
    parser.add_argument("--protocol-version", type=int, choices=(3, 4), default=4)

    # Quest 3
    parser.add_argument("--quest3-ws-port", type=int, default=8765)
    parser.add_argument("--quest3-http-port", type=int, default=8443)
    parser.add_argument(
        "--quest3-no-ssl", action="store_true",
        help="Disable TLS for the Quest 3 WebSocket / HTTP servers. "
             "WebXR refuses non-secure contexts, so leave SSL on for "
             "production use.",
    )

    # Cadence
    parser.add_argument("--rate", type=float, default=50.0)

    # Render
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--no-omnihand", action="store_true")

    # Hand mapping
    parser.add_argument(
        "--hand-input", choices=("trigger", "grip", "max"), default="trigger",
        help="Which controller analog drives finger curl. 'trigger' is "
             "the default index-finger trigger; 'grip' is the middle "
             "grip squeeze; 'max' picks whichever is greater this frame.",
    )

    # Per-finger noise / jitter / occlusion filter (v0.6)
    parser.add_argument(
        "--no-finger-filter", action="store_true",
        help="Disable the per-side EMA + rolling-median deadband on "
             "the Quest 3 hand-curl / thumb-oppose / finger-tip-oppose "
             "streams. The filter reduces visual finger tremor by "
             "~20-40%% on held poses with ~20 ms motion lag.",
    )
    parser.add_argument(
        "--finger-filter-alpha", type=float, default=None,
        help="EMA alpha for the finger-signal filter. Default 0.5.",
    )
    parser.add_argument(
        "--finger-filter-hold-window", type=int, default=None,
        help="Rolling-window length for the deadband-hold. Default 8.",
    )
    parser.add_argument(
        "--finger-filter-hold-std", type=float, default=None,
        help="Per-channel rolling-std threshold for entering the "
             "held-pose latch. Default 0.005.",
    )

    # Per-finger / thumb-oppose stretch (opt-in additional shaping
    # on top of the per-operator affine normalisation). See
    # docs/source/tutorials/x2_dataset_record_and_replay.md
    # § "Why we abandoned the global power-curve compensation".
    #
    # As of 2026-05-13 the default ``stretch_finger_curls`` /
    # ``stretch_thumb_oppose`` parameters are SMOOTH PROPORTIONAL
    # (``dz=0.05, full=0.95, gamma=1`` -- linear in the active zone
    # with tiny rest-noise / saturation cushions), so enabling
    # these flags on top of an operator calibration leaves
    # mid-range curls intact and the operator gets continuous
    # control of the closure depth. The previous defaults were
    # bimodal (``dz=0.35, full=0.40, gamma=5``) and silently
    # destroyed the proportional response of a calibrated teleop
    # loop -- see the long block-comment above
    # DEFAULT_CURL_DEADZONE_PER_FINGER in
    # gear_sonic/utils/teleop/x2_hand_retarget.py for the history.
    parser.add_argument(
        "--apply-curl-compensation", action="store_true",
        help="Apply the per-finger curl stretch curve on top of the "
             "operator's affine normalisation. With the smooth-"
             "proportional defaults this only adds a tiny rest-noise "
             "cutoff and saturation cushion; pass explicit per-finger "
             "params to recover the legacy bimodal 'isolated-curl "
             "detector' behaviour for tight power-grasp tasks.",
    )
    parser.add_argument(
        "--apply-oppose-compensation", action="store_true",
        help="Apply the thumb-opposition stretch curve on top of the "
             "operator's affine normalisation. Smooth-proportional by "
             "default; pair with --apply-curl-compensation for "
             "consistent shaping across the curl and oppose channels.",
    )

    # SONIC corrective-delta observability (v1 schema)
    parser.add_argument(
        "--sonic-correction-warn-rad", type=float, default=0.05,
        help="Threshold in radians; the operator log prints when the "
             "max arm |executed - commanded| over the last second "
             "exceeds this value. Default 0.05 rad (~2.9 deg).",
    )
    parser.add_argument(
        "--no-sonic-correction-log", action="store_true",
        help="Suppress the once-per-second SONIC corrective-delta "
             "operator log. The action.sonic_correction_max_rad column "
             "is still populated regardless.",
    )

    # IK
    parser.add_argument("--ik-damping", type=float, default=0.08)
    parser.add_argument(
        "--ik-rotation-weight", type=float, default=0.3,
        help="0.0 = position-only IK; >0 enables wrist orientation "
             "tracking. Default 0.3 works once a v1+ calibration YAML "
             "is in place (recapture with vr_operator_calibrate.py to "
             "get wrist alignment quats). Legacy v0 calibrations are "
             "auto-detected and force position-only.",
    )
    parser.add_argument("--ik-per-tick-step-rad", type=float, default=0.30)

    # Operator calibration (replaces engage-anchor wrist anchoring)
    default_cal = (
        Path(__file__).resolve().parent.parent.parent
        / "data" / "operator_calibrations" / "default.yaml"
    )
    parser.add_argument(
        "--calibration", type=Path, default=default_cal,
        help="YAML produced by vr_operator_calibrate.py. Required "
             "unless --recalibrate is set.",
    )
    parser.add_argument(
        "--recalibrate", action="store_true",
        help="Run the 3-pose calibration inline before recording starts. "
             "Use for the first session with a new operator.",
    )
    parser.add_argument(
        "--operator-id", type=str, default="default",
        help="Free-form operator label stamped into the calibration YAML.",
    )

    # ── Robocasa scene mode (G1: deploy stays in the loop, recorder
    #    just learns about the static scene XML the deploy is loading).
    #    See gear_sonic/utils/teleop/x2_dataset_recorder.py
    #    RecorderConfig docs for the full plumbing diagram.
    _scenes_dir = (
        Path(__file__).resolve().parent.parent
        / "data" / "assets" / "robocasa_scenes"
    )
    parser.add_argument(
        "--robocasa-env",
        choices=("none", "X2PickPlaceCube", "X2PickPlaceBowl", "X2PickPlaceApple"),
        default="none",
        help="When != 'none', the recorder switches into robocasa scene "
             "mode: it loads "
             "data/assets/robocasa_scenes/<env>.xml + .json so the ego "
             "renderer shows the table + objects, instantiates a "
             "RobocasaTaskMirror for per-tick success/reward/subtask "
             "labels, and PUB/SUBs scene-state with the deploy bridge. "
             "The matching .xml must be built ahead of time via "
             "gear_sonic/scripts/build_x2_robocasa_scene_xml.py. "
             "The companion record_x2_dataset.sh forwards --sim-mjcf "
             "to deploy_x2.sh so the deploy loads the same scene.",
    )
    parser.add_argument(
        "--scene-xml-path", type=Path, default=None,
        help="Override the scene MJCF path resolved from --robocasa-env. "
             "Use this only when developing custom robocasa scenes that "
             "live outside data/assets/robocasa_scenes/.",
    )
    parser.add_argument(
        "--scene-state-sub-host", default="localhost",
        help="Host for the scene_state ZMQ SUB (bridge -> recorder). "
             "Match the deploy bridge's --scene-state-pub-host.",
    )
    parser.add_argument(
        "--scene-state-sub-port", type=int, default=5559,
        help="Port for the scene_state ZMQ SUB. Default matches the "
             "bridge's --scene-state-pub-port.",
    )
    parser.add_argument(
        "--scene-reset-pub-host", default="*",
        help="Bind iface for the scene_reset ZMQ PUB "
             "(recorder -> bridge).",
    )
    parser.add_argument(
        "--scene-reset-pub-port", type=int, default=5560,
        help="Port for the scene_reset ZMQ PUB. Default matches the "
             "bridge's --scene-reset-sub-port.",
    )
    parser.add_argument(
        "--episode-seed", type=int, default=None,
        help="Optional RNG seed for the RobocasaTaskMirror's per-episode "
             "reset (object placement randomization). Useful for "
             "reproducible smoke tests.",
    )
    # ── Wide-angle world-fixed witness camera (`front_cam`) ────────────
    # The robocasa scene XMLs (built by build_x2_robocasa_scene_xml.py)
    # bake in a 120° FoV camera 3 ft in front of the robot launch
    # position at chest height. When --front-cam is set, the recorder
    # builds a second MujocoFrameRenderer pinned to it and writes the
    # frames as `observation.images.front_cam` alongside the existing
    # `observation.images.ego_view`. Defaults to None so we can flip
    # to True iff a scene XML is actually loaded (the legacy flat-floor
    # MJCF doesn't contain the camera). Pass --no-front-cam to opt out
    # explicitly (e.g. to keep the legacy single-camera schema for an
    # existing dataset directory you're appending to).
    front_grp = parser.add_argument_group("front_cam (witness view)")
    front_grp.add_argument(
        "--front-cam", dest="front_cam",
        action="store_true", default=None,
        help=(
            "Enable the world-fixed wide-angle witness camera "
            "(observation.images.front_cam). Defaults to True iff "
            "--robocasa-env != none / --scene-xml-path resolves; pass "
            "--no-front-cam to opt out. The camera lives in the scene "
            "XML (see _WORKSPACE_CAMERAS in "
            "gear_sonic/scripts/build_x2_robocasa_scene_xml.py); "
            "asking for it without a scene XML is a no-op + warning "
            "(legacy flat-floor MJCF has no front_cam definition)."
        ),
    )
    front_grp.add_argument(
        "--no-front-cam", dest="front_cam",
        action="store_false",
        help=(
            "Suppress the front_cam video track. Use this when "
            "appending to a pre-existing single-camera LeRobot "
            "dataset directory whose meta/info.json was written "
            "without the front_cam feature (the exporter rejects "
            "post-hoc schema additions)."
        ),
    )
    # Stash the resolved scenes dir on the parser so the resolver in
    # main() can find scene XMLs without recomputing the path.
    parser.set_defaults(_robocasa_scenes_dir=_scenes_dir)

    # ── Phase 0 subscribe-only mode (planner-driven recorder) ──────────
    # Default = "internal": legacy direct-Quest pipeline. Set both
    # flags to "zmq" to subscribe to the planner's body_pose AND the
    # manager's arm_targets / hand_finger_cmd / stream_mode /
    # recorder_cmd topics. Mixing the two is rejected at startup.
    parser.add_argument(
        "--body-pose-source", choices=("internal", "zmq"), default="internal",
        help=(
            "Where the recorder gets its 31-DOF body_pose reference "
            "from. 'internal' (default): legacy in-process Quest 3 + "
            "VR IK. 'zmq': subscribe to the planner's body_pose topic "
            "(use with --arm-targets-source=zmq for the Phase 0 "
            "planner-driven stack)."
        ),
    )
    parser.add_argument(
        "--arm-targets-source", choices=("internal", "zmq"), default="internal",
        help=(
            "Where arm_targets + hand_finger_cmd come from. 'internal' "
            "runs IK + finger filter inline; 'zmq' subscribes to the "
            "manager's arm_targets / hand_finger_cmd / stream_mode / "
            "recorder_cmd topics. Must match --body-pose-source."
        ),
    )
    parser.add_argument(
        "--body-pose-sub-host", default="localhost",
        help="Host for the body_pose SUB (planner -> recorder).",
    )
    parser.add_argument(
        "--body-pose-sub-port", type=int, default=5565,
        help=(
            "Port for the body_pose SUB. Must match the planner's "
            "--body-pose-port."
        ),
    )
    parser.add_argument(
        "--body-pose-sub-topic", default="body_pose",
        help="Topic for the body_pose SUB. Must match the planner.",
    )
    parser.add_argument(
        "--arm-and-hands-sub-host", default="localhost",
        help=(
            "Host for the manager's multi-topic SUB "
            "(arm_targets / hand_finger_cmd / stream_mode / "
            "recorder_cmd)."
        ),
    )
    parser.add_argument(
        "--arm-and-hands-sub-port", type=int, default=5564,
        help=(
            "Port for the manager's multi-topic SUB. Must match the "
            "manager's --recorder-pub-port."
        ),
    )

    # ── Live gesture playback (PKL takeover during subscribe mode) ─────
    # See gear_sonic/utils/teleop/gesture_session.py for wire shape and
    # gear_sonic/data/motions/gestures/gestures_v1.yaml for catalog
    # format. Trigger CLI lives at gear_sonic/scripts/play_gesture.py.
    _default_gesture_catalog = (
        Path(__file__).resolve().parent.parent
        / "data" / "motions" / "gestures" / "gestures_v1.yaml"
    )
    parser.add_argument(
        "--gesture-cmd-host", default="*",
        help=(
            "Interface for the gesture_cmd SUB bind. Defaults to '*' "
            "(all interfaces) because the trigger script is the "
            "transient side: recorder binds, play_gesture connects."
        ),
    )
    parser.add_argument(
        "--gesture-cmd-port", type=int, default=5568,
        help=(
            "Port for the gesture_cmd SUB. Must match the trigger "
            "script's --port. Default matches "
            "gesture_session.GESTURE_CMD_DEFAULT_PORT."
        ),
    )
    parser.add_argument(
        "--gesture-cmd-topic", default="gesture_cmd",
        help="Topic for the gesture_cmd SUB.",
    )
    parser.add_argument(
        "--gesture-catalog", type=str,
        default=str(_default_gesture_catalog),
        help=(
            "YAML catalog mapping gesture names to PKL clips. Loaded "
            "once at startup; ad-hoc --pkl payloads always work "
            "regardless. Pass an empty string ('') to disable gesture "
            "support entirely (no SUB bound)."
        ),
    )
    parser.add_argument(
        "--gesture-future-dt-s", type=float, default=0.1,
        help=(
            "Spacing of the strictly-future window during gesture "
            "playback. Default 0.1 s matches the kplanner and the C++ "
            "deploy's DT_FUTURE_REF."
        ),
    )
    parser.add_argument(
        "--gesture-future-window-frames", type=int, default=9,
        help=(
            "Number of strictly-future frames packed into the deploy "
            "wire during gesture playback. Default 9 matches the "
            "kplanner's NUM_FUTURE_FRAMES-1 convention."
        ),
    )

    # Misc
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--embodiment-tag", default="new_embodiment")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Auto-install optional packages needed by the recorder workflow.
    # ``RECORDER_DEPS`` includes:
    #   * gTTS (Quest 3 calibration audio prompts)
    #   * datasets / av / lerobot (LeRobot v2.1 dataset writer)
    # These can be ~200 MB on a clean venv; we install them on first
    # launch instead of failing with a cryptic ImportError. The
    # heavy deps are deferred behind a lazy import below so this
    # ``ensure_runtime_deps`` call has a chance to materialise them
    # before the dataset writer tries to ``import datasets``.
    from gear_sonic.utils.install import (
        RECORDER_DEPS,
        ensure_runtime_deps,
    )

    ensure_runtime_deps(
        RECORDER_DEPS,
        purpose="X2 dataset recorder (LeRobot writer + Quest 3 audio)",
    )

    # Deferred import: pulling X2DatasetRecorder also pulls
    # ``datasets`` and the LeRobot writer chain, which the
    # ensure_runtime_deps() call above just guaranteed are present.
    from gear_sonic.utils.teleop.finger_signal_filter import FingerFilterParams
    from gear_sonic.utils.teleop.x2_dataset_recorder import (
        RecorderConfig,
        X2DatasetRecorder,
    )

    if args.no_finger_filter:
        finger_filter_params = None
    else:
        kwargs: dict[str, Any] = {}
        if args.finger_filter_alpha is not None:
            kwargs["ema_alpha"] = float(args.finger_filter_alpha)
        if args.finger_filter_hold_window is not None:
            kwargs["hold_window"] = int(args.finger_filter_hold_window)
        if args.finger_filter_hold_std is not None:
            kwargs["hold_std"] = float(args.finger_filter_hold_std)
        finger_filter_params = FingerFilterParams(**kwargs)

    if not args.teleop_only:
        if args.output_dir is None or not args.task:
            # Robocasa mode auto-fills the task string from the scene
            # metadata (the env's canonical instruction is what the
            # mirror's success oracle is grading against), so the
            # operator only needs to pass --output-dir.
            if args.robocasa_env != "none" and args.output_dir is not None:
                pass
            else:
                raise SystemExit(
                    "Error: --output-dir and --task are required unless "
                    "--teleop-only is set "
                    "(robocasa mode auto-fills --task from the env's "
                    "instruction; --output-dir is still required)."
                )

    # Resolve --robocasa-env -> scene XML path. The recorder + the deploy
    # bridge both need to load the SAME .xml so their MuJoCo replicas
    # stay in lock-step; we surface a clear error here when the asset
    # is missing instead of letting the recorder try to subscribe to a
    # bridge that's loading something different.
    scene_xml_path: Path | None = None
    robocasa_env_name: str | None = None
    if args.scene_xml_path is not None:
        scene_xml_path = args.scene_xml_path
        if args.robocasa_env != "none":
            robocasa_env_name = args.robocasa_env
    elif args.robocasa_env != "none":
        scene_xml_path = (
            args._robocasa_scenes_dir / f"{args.robocasa_env}.xml"
        )
        robocasa_env_name = args.robocasa_env
        if not scene_xml_path.is_file():
            raise SystemExit(
                f"Error: scene MJCF for --robocasa-env "
                f"{args.robocasa_env!r} not found at {scene_xml_path}.\n"
                "       Build it via:\n"
                "         python -m gear_sonic.scripts.build_x2_robocasa_scene_xml "
                f"--env {args.robocasa_env}"
            )

    cfg = RecorderConfig(
        output_dir=args.output_dir,
        task=args.task,
        sonic_checkpoint=args.sonic_checkpoint,
        sonic_tokenizer_device=args.sonic_tokenizer_device,
        sonic_encoder_config=(
            None
            if args.encoder_config is None
            or str(args.encoder_config).strip() == ""
            else args.encoder_config
        ),
        obs_dump_recorder_path=args.obs_dump_recorder,
        teleop_only=args.teleop_only,
        pub_host=args.pub_host,
        pub_port=args.pub_port,
        pub_topic=args.pub_topic,
        sub_host=args.sub_host,
        sub_port=args.sub_port,
        sub_topic=args.sub_topic,
        protocol_version=args.protocol_version,
        quest3_ws_port=args.quest3_ws_port,
        quest3_http_port=args.quest3_http_port,
        quest3_use_ssl=(not args.quest3_no_ssl),
        publish_rate_hz=args.rate,
        record_rate_hz=args.rate,
        render_width=args.render_width,
        render_height=args.render_height,
        with_omnihand=(not args.no_omnihand),
        hand_input_mode=args.hand_input,
        ik_damping=args.ik_damping,
        ik_rotation_weight=args.ik_rotation_weight,
        ik_per_tick_step_rad=args.ik_per_tick_step_rad,
        calibration_path=args.calibration,
        recalibrate=args.recalibrate,
        operator_id=args.operator_id,
        embodiment_tag=args.embodiment_tag,
        finger_filter_params=finger_filter_params,
        apply_curl_compensation=bool(args.apply_curl_compensation),
        apply_oppose_compensation=bool(args.apply_oppose_compensation),
        sonic_correction_warn_rad=args.sonic_correction_warn_rad,
        log_sonic_correction=(not args.no_sonic_correction_log),
        scene_xml_path=scene_xml_path,
        robocasa_env=robocasa_env_name,
        record_front_cam=_resolve_front_cam_default(
            args.front_cam, scene_xml_path
        ),
        scene_state_sub_host=args.scene_state_sub_host,
        scene_state_sub_port=args.scene_state_sub_port,
        scene_reset_pub_host=args.scene_reset_pub_host,
        scene_reset_pub_port=args.scene_reset_pub_port,
        episode_seed=args.episode_seed,
        body_pose_source=args.body_pose_source,
        arm_targets_source=args.arm_targets_source,
        body_pose_sub_host=args.body_pose_sub_host,
        body_pose_sub_port=args.body_pose_sub_port,
        body_pose_sub_topic=args.body_pose_sub_topic,
        arm_and_hands_sub_host=args.arm_and_hands_sub_host,
        arm_and_hands_sub_port=args.arm_and_hands_sub_port,
        gesture_cmd_host=args.gesture_cmd_host,
        gesture_cmd_port=args.gesture_cmd_port,
        gesture_cmd_topic=args.gesture_cmd_topic,
        # Empty string disables gesture support entirely. Strip
        # whitespace so accidental "  " doesn't accidentally enable
        # an unintended catalog path.
        gesture_catalog_path=(
            None
            if not str(args.gesture_catalog).strip()
            else Path(args.gesture_catalog)
        ),
        gesture_future_dt_s=args.gesture_future_dt_s,
        gesture_future_window_frames=args.gesture_future_window_frames,
        idle_publish_enabled=(not args.no_idle_publish),
        verbose=(not args.quiet),
    )

    recorder = X2DatasetRecorder(cfg)

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"[recorder] caught signal {signum}, shutting down …", flush=True)
        recorder.stop()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    recorder.start()
    try:
        recorder.run()
    finally:
        recorder.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
