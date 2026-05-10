"""Pure-kinematics VR teleop for the X2 in MuJoCo.

This is the **deploy-free** teleop path:

    Quest 3 (WebXR) -> DLS arm IK -> joint_pos_mj -> MuJoCo viewer

No SONIC, no C++ deploy, no docker, no ROS2, no ZMQ. Lower body, waist
and head are pinned to the trained stand pose for every frame; only
the 14 arm DOFs and (with ``--with-omnihand``) the 20 OmniHand DOFs
follow operator input. The MuJoCo passive viewer renders the live
X2 with omnihand fingers exactly the way the offline smoketest
renderer does (purely kinematic, ``mj_forward`` only -- no physics
step, no falling, no collision response).

Why a separate path?

* Iteration speed: no docker startup, no policy load, no ZMQ wiring.
  Click-to-arm-tracking in well under 5 s.
* Debugging: when the SONIC-stabilised path
  (:file:`record_x2_dataset.sh`) does something weird, this is the
  ground-truth answer to *what is the operator actually asking for?*
  -- there is nothing in the loop except IK + viewer.
* Optional dataset writes follow the same LeRobot v2.1 schema the
  SONIC-driven recorder uses. The recorded ``observation.state``
  is the kinematic body pose (no policy interaction) and
  ``observation.images.ego_view`` is rendered off-screen at 50 Hz.
* When recording is active a side-channel debug NPZ is written to
  ``<output-dir>/debug/teleop_episode_NNNNNN.npz`` containing the
  raw VR 3-pt pose (head + L/R wrist xyz+quat), trigger/grip values,
  face-button held state, IK targets, IK position/rotation residuals,
  the commanded full-body / per-hand joint commands in radians, and
  (when XRHand is active) per-frame ``quest_*_hand_curls`` (5-float
  vectors) plus ``quest_*_thumb_oppose`` scalars — the same Quest 3
  signals that drive retargeting. Rows use NaN for curls/oppose when
  that side falls back to uniform controller grasp. Use this to
  diagnose retargeting offline with numpy + matplotlib without
  re-running VR sessions. The 1 Hz stdout line also prints raw curls
  for quick checks; it samples less often than the NPZ.

Run from your interactive shell (the MuJoCo viewer needs ``DISPLAY``)::

    python -m gear_sonic.scripts.teleop_x2_kinematic
    python -m gear_sonic.scripts.teleop_x2_kinematic \\
        --output-dir data/lerobot/x2_quest3_kinematic_v0 \\
        --task "wave hello"

Quest 3 controller buttons:

* **A** -- toggle active arm tracking on / off. Stateless: idle holds
  the arms at neutral; active drives them through the calibrated
  head-relative wrist mapping. Pressing A repeatedly does NOT recapture
  any anchor (calibration is offline; see ``vr_operator_calibrate.py``).
* **B** -- start a new episode (only if ``--output-dir`` was passed).
* **X** -- stop and save.
* **Y** -- stop and discard.
* **Trigger / Grip** -- per-side analog finger curl.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import signal
import sys
import time
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.live_vla_publish_motion_token import (  # noqa: E402
    DEFAULT_HAND_DOF,
    DEFAULT_STAND_POSE_MUJOCO_RAD,
    NUM_BODY_DOFS,
)
from gear_sonic.utils.teleop.operator_calibration import (  # noqa: E402
    OperatorCalibration,
)
from gear_sonic.utils.teleop.vr.quest3_reader import Quest3Reader  # noqa: E402
from gear_sonic.utils.teleop.vr_arm_teleop_v2 import (  # noqa: E402
    VRArmTeleopCalibrated,
)
from gear_sonic.utils.teleop.x2_hand_retarget import (  # noqa: E402
    DEFAULT_APPLY_CURL_COMPENSATION,
    NUM_HAND_DOF_PER_SIDE,
    controller_grasp_ratio,
    grasp_command_from_ratio,
    per_finger_grasp_command_from_curls_and_oppose,
    stretch_finger_curls,
)


_DEFAULT_CALIBRATION_PATH = (
    REPO_ROOT / "data" / "operator_calibrations" / "default.yaml"
)


from gear_sonic.utils.teleop.x2_kinematic_view import (  # noqa: E402
    DEFAULT_PELVIS_POS_XYZ as _DEFAULT_PELVIS_POS_XYZ,
    DEFAULT_PELVIS_QUAT_WXYZ as _DEFAULT_PELVIS_QUAT_WXYZ,
    build_kinematic_model as _build_kinematic_model,
    set_kinematic_pose as _set_kinematic_pose,
)


_LEFT_ARM_MJ_SLICE = slice(15, 22)
_RIGHT_ARM_MJ_SLICE = slice(22, 29)

# Placeholder for action.motion_token written into recorded parquets.
# Live kinematic teleop never runs the FSQ encoder online -- the offline
# labeler pass is responsible for re-tokenising action.commanded_body_q_mj
# into the 64-D motion-token surface used by the SONIC tracker.
_ZERO_MOTION_TOKEN = np.zeros(64, dtype=np.float64)


@dataclass
class _EpisodeBuffer:
    frames: list[dict[str, Any]]
    debug: list[dict[str, Any]]
    started_at: float = 0.0
    task: str = ""

    def push(self, frame: dict[str, Any]) -> None:
        self.frames.append(frame)

    def push_debug(self, row: dict[str, Any]) -> None:
        self.debug.append(row)

    def __len__(self) -> int:
        return len(self.frames)

    def reset(self) -> None:
        self.frames.clear()
        self.debug.clear()


def _preflight_output_dir(path: Path) -> None:
    """Validate or auto-clean an existing ``--output-dir`` before exporter init.

    Thin wrapper around
    :func:`gear_sonic.data.dataset_output_dir.preflight_dataset_output_dir`
    so the existing test surface (``test_teleop_x2_kinematic_smoke``)
    keeps importing the same name.
    """
    from gear_sonic.data.dataset_output_dir import preflight_dataset_output_dir

    preflight_dataset_output_dir(path, log_prefix="teleop-kinematic")


def _save_debug_npz(
    debug_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    episode_index: int,
    fps: float,
    task: str,
) -> Path:
    """Stack a list of per-frame debug dicts into an NPZ for offline analysis."""
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    out_path = debug_dir / f"teleop_episode_{episode_index:06d}.npz"

    if not debug_rows:
        np.savez_compressed(out_path, num_frames=np.int64(0), task=task)
        return out_path

    keys = list(debug_rows[0].keys())
    stacked: dict[str, np.ndarray] = {}
    for k in keys:
        col = [row[k] for row in debug_rows]
        try:
            stacked[k] = np.asarray(col)
        except Exception:
            # Heterogeneous / non-uniform shapes -- store as object array
            # so np.savez_compressed accepts it. Loaders can dtype.kind == 'O'.
            stacked[k] = np.asarray(col, dtype=object)

    np.savez_compressed(
        out_path,
        fps=float(fps),
        task=str(task),
        num_frames=np.int64(len(debug_rows)),
        **stacked,
    )
    return out_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Output / dataset
    p.add_argument(
        "--output-dir", type=Path, default=None,
        help="Directory to write a LeRobot v2.1 dataset into. Omit for "
             "pure-viewer teleop with no disk writes.",
    )
    p.add_argument(
        "--task", type=str, default="",
        help="Language instruction stamped on every recorded episode. "
             "Required if --output-dir is set.",
    )

    # Quest 3
    p.add_argument("--quest3-ws-port", type=int, default=8765)
    p.add_argument("--quest3-http-port", type=int, default=8443)
    p.add_argument(
        "--quest3-no-ssl", action="store_true",
        help="Disable TLS for the Quest 3 server. WebXR refuses non-secure "
             "contexts so this is for trusted-LAN debugging only.",
    )

    # Cadence
    p.add_argument("--rate", type=float, default=50.0)

    # Render / model
    p.add_argument("--with-omnihand", dest="with_omnihand", action="store_true")
    p.add_argument("--no-omnihand", dest="with_omnihand", action="store_false")
    p.set_defaults(with_omnihand=True)
    p.add_argument("--render-width", type=int, default=640)
    p.add_argument("--render-height", type=int, default=480)

    # Hand mapping
    p.add_argument(
        "--hand-input", choices=("trigger", "grip", "max"), default="trigger",
        help="Which controller analog drives finger curl.",
    )

    # IK
    p.add_argument("--ik-damping", type=float, default=0.08)
    p.add_argument(
        "--ik-rotation-weight", type=float, default=0.3,
        help="0.0 = position-only IK; >0 enables wrist orientation "
             "tracking. Default 0.3 works once a v1+ calibration YAML "
             "is in place (recapture with vr_operator_calibrate.py to "
             "get wrist alignment quats). Legacy v0 calibrations are "
             "auto-detected and force position-only.",
    )
    p.add_argument("--ik-per-tick-step-rad", type=float, default=0.30)

    # Calibration (replaces the engage-anchor path)
    p.add_argument(
        "--calibration", type=Path, default=_DEFAULT_CALIBRATION_PATH,
        help="Path to a YAML produced by vr_operator_calibrate.py. "
             "Defines the per-arm head-relative -> torso-frame wrist "
             "mapping. Required unless --recalibrate is set.",
    )
    p.add_argument(
        "--recalibrate", action="store_true",
        help="Run the 3-pose calibration inline before teleop starts, "
             "then load the freshly-written YAML. Use this for the "
             "first session with a new operator.",
    )
    p.add_argument(
        "--operator-id", type=str, default="default",
        help="Free-form operator label stamped into the calibration "
             "YAML (only used with --recalibrate).",
    )

    # Misc
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--embodiment-tag", default="new_embodiment")
    return p.parse_args(argv)


def _load_or_recalibrate(
    quest: Quest3Reader,
    args: argparse.Namespace,
) -> OperatorCalibration:
    """Resolve the operator calibration: load YAML, or run inline capture.

    Centralised here so the recorder script can call it the same way.
    """
    cal_path = args.calibration

    if args.recalibrate:
        from gear_sonic.scripts.vr_operator_calibrate import (
            run_inline_calibration,
        )
        print(
            f"[teleop-kinematic] --recalibrate set; running guided "
            f"calibration before teleop. Output: {cal_path}",
            flush=True,
        )
        return run_inline_calibration(
            quest,
            output_path=cal_path,
            operator_id=args.operator_id,
        )

    if not cal_path.is_file():
        raise SystemExit(
            f"Error: calibration file not found at {cal_path}. Either:\n"
            f"  - Run `python -m gear_sonic.scripts.vr_operator_calibrate "
            f"--operator-id {args.operator_id}` to capture one, OR\n"
            f"  - Pass `--recalibrate` to capture it inline before "
            f"teleop starts."
        )
    cal = OperatorCalibration.load_yaml(cal_path)
    print(
        f"[teleop-kinematic] loaded calibration {cal_path} "
        f"(operator='{cal.operator_id}', "
        f"L_residual={cal.fit['left'].residual_m*100:.1f} cm, "
        f"R_residual={cal.fit['right'].residual_m*100:.1f} cm)",
        flush=True,
    )
    return cal


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    # Auto-install optional teleop deps (currently just gTTS for the
    # Quest 3 calibration audio cache used during --recalibrate).
    # The kinematic-teleop entry point intentionally does NOT pull
    # the heavy LeRobot / datasets stack -- that's reserved for
    # record_x2_dataset.py.
    from gear_sonic.utils.install import (
        CALIBRATION_DEPS,
        ensure_runtime_deps,
    )

    ensure_runtime_deps(
        CALIBRATION_DEPS,
        purpose="kinematic teleop (Quest 3 audio prompts during recalibrate)",
    )

    record_to_disk = args.output_dir is not None
    if record_to_disk and not args.task:
        raise SystemExit(
            "Error: --task is required when --output-dir is set."
        )
    if record_to_disk:
        # Preflight FIRST, before we boot Quest 3 / MuJoCo. Catches the
        # half-broken-output-dir case that traps Gr00tDataExporter.create()
        # into the resume code path with no meta/info.json.
        _preflight_output_dir(args.output_dir)
        # Note: do NOT mkdir(parents=True, exist_ok=True) here -- the
        # exporter's create() distinguishes "fresh" from "resume" purely
        # by Path.exists(), so pre-creating an empty dir would land us
        # in the resume code path with no metadata. Let the exporter
        # create the leaf directory on its first frame.
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)

    import mujoco  # noqa: E402  -- defer so --help doesn't need GL
    import mujoco.viewer  # noqa: E402

    model, hand_layout, body_qposadr = _build_kinematic_model(
        with_omnihand=args.with_omnihand
    )
    data = mujoco.MjData(model)
    apply_hand_fn = None
    if args.with_omnihand and hand_layout is not None:
        from gear_sonic.scripts.compose_x2_with_omnihand import (
            apply_active_hand_qpos as apply_hand_fn,
        )

    # Initial kinematic stand pose so the viewer opens on a calm robot.
    init_body = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
    zero_hand = np.zeros(NUM_HAND_DOF_PER_SIDE, dtype=np.float64)
    _set_kinematic_pose(
        mujoco_mod=mujoco,
        model=model,
        data=data,
        body_q_mj=init_body,
        body_qposadr=body_qposadr,
        layout=hand_layout,
        apply_hand_fn=apply_hand_fn,
        left_hand_q=zero_hand,
        right_hand_q=zero_hand,
    )

    # VR (Quest3 server starts first so --recalibrate has a live client).
    quest = Quest3Reader(
        ws_port=args.quest3_ws_port,
        http_port=args.quest3_http_port,
        use_ssl=(not args.quest3_no_ssl),
        quiet_periodic=True,
    )
    quest.start()

    # Resolve calibration BEFORE building the teleop. This may run an
    # inline 3-pose capture flow if --recalibrate was set, in which case
    # the operator should already have the headset on by the time we
    # reach the teleop loop.
    if args.recalibrate:
        # The inline calibrator needs a connected client; block here
        # until the WebXR session is up.
        from gear_sonic.scripts.vr_operator_calibrate import (
            _wait_for_first_packet,
        )
        _wait_for_first_packet(quest)
    calibration = _load_or_recalibrate(quest, args)

    teleop = VRArmTeleopCalibrated(
        calibration=calibration,
        damping=args.ik_damping,
        rotation_weight=args.ik_rotation_weight,
        per_tick_step_rad=args.ik_per_tick_step_rad,
    )

    if calibration.hand_range is not None:
        hr = calibration.hand_range
        print(
            f"[teleop-kinematic] hand range loaded "
            f"(source={hr.source!r} samples={hr.samples}); curl ranges:",
            flush=True,
        )
        for side_name, side_fit in (("L", hr.left), ("R", hr.right)):
            ranges = " ".join(
                f"{n}=[{side_fit.floor[i]:.2f},{side_fit.ceiling[i]:.2f}]"
                for i, n in enumerate(("thumb", "index", "middle", "ring", "pinky"))
            )
            print(
                f"[teleop-kinematic]   {side_name}: {ranges}  "
                f"oppose=[{side_fit.oppose_floor:.2f},{side_fit.oppose_ceiling:.2f}]",
                flush=True,
            )
    else:
        print(
            "[teleop-kinematic] no hand_range in calibration -- using "
            "linear pass-through (raw Quest curls drive the full motor "
            "lerp). Add a hand_range section via "
            "`gear_sonic.scripts.fit_hand_range_from_npz` to enable "
            "per-operator finger normalization.",
            flush=True,
        )

    # Optional dataset exporter + ego renderer
    exporter: Any = None
    renderer: Any = None
    if record_to_disk:
        from gear_sonic.data.exporter import Gr00tDataExporter
        from gear_sonic.data.features_x2_vla import (
            HAND_DOF_OMNI,
            assemble_observation_state,
            get_features_x2_vla,
            get_modality_config_x2_vla,
            get_x2_robot_model,
        )
        from gear_sonic.scripts.live_vla_publish_motion_token import MJ_TO_PIN
        from gear_sonic.scripts.render_smoketest_episode_video import (
            MujocoFrameRenderer,
        )

        robot_model = get_x2_robot_model(hand_variant="omnihand_10")
        features = get_features_x2_vla(
            robot_model, hand_dof_per_side=HAND_DOF_OMNI
        )
        modality_cfg = get_modality_config_x2_vla(
            robot_model, hand_dof_per_side=HAND_DOF_OMNI
        )
        exporter = Gr00tDataExporter.create(
            save_root=args.output_dir,
            fps=int(args.rate),
            features=features,
            modality_config=modality_cfg,
            task=args.task,
            script_config={
                "robot_type": "agibot_x2_ultra",
                "embodiment_tag": args.embodiment_tag,
                "hand_variant": "omnihand_10",
                "num_body_joints": robot_model.num_joints,
                "hand_dof_per_side": HAND_DOF_OMNI,
                "fps": int(args.rate),
                "teleop_mode": "kinematic",
            },
            robot_type="agibot_x2_ultra",
        )
        renderer = MujocoFrameRenderer(
            camera="ego_view",
            width=args.render_width,
            height=args.render_height,
            with_omnihand=args.with_omnihand,
            egl=True,
        )
        print(
            f"[teleop-kinematic] dataset exporter ready -> {args.output_dir}",
            flush=True,
        )

    # Banner
    try:
        import socket

        ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip = "<workstation-ip>"
    print("─" * 60, flush=True)
    print("  X2 VR teleop -- pure kinematics (no SONIC, no deploy)", flush=True)
    print("─" * 60, flush=True)
    print(f"  output_dir        : {args.output_dir or '(none -- not recording)'}", flush=True)
    print(f"  task              : {args.task!r}", flush=True)
    print(f"  with_omnihand     : {args.with_omnihand}", flush=True)
    print(f"  rate              : {args.rate} Hz", flush=True)
    print(f"  Quest 3 WebXR URL : https://{ip}:{args.quest3_http_port}", flush=True)
    print("─" * 60, flush=True)
    print(
        f"  calibration       : {args.calibration} "
        f"(L_resid={calibration.fit['left'].residual_m*100:.1f} cm, "
        f"R_resid={calibration.fit['right'].residual_m*100:.1f} cm)",
        flush=True,
    )
    print("─" * 60, flush=True)
    print("  Press A on either Quest 3 controller to toggle active arm tracking.", flush=True)
    if record_to_disk:
        print("  B = start episode  X = save  Y = discard", flush=True)
    print("  Ctrl-C to exit.", flush=True)
    print("─" * 60, flush=True)

    # Episode lifecycle
    episode_buffer = _EpisodeBuffer(frames=[], debug=[])
    is_recording = False
    episode_count = 0
    prev_buttons = (False, False, False, False)
    episode_t0 = 0.0
    last_tick_result: Any = None

    period = 1.0 / max(args.rate, 1e-6)
    next_tick = time.monotonic()

    stop = {"flag": False}

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"\n[teleop-kinematic] caught signal {signum}, stopping …", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    last_log_t = 0.0
    wait_msg = False
    engaged_now = False
    saw_first_vr_packet = False
    prev_left_grasp_closed = False
    prev_right_grasp_closed = False
    GRASP_THRESH = 0.5
    BTN_NAMES = ("A", "B", "X", "Y")

    def _fmt3(v: np.ndarray) -> str:
        return f"({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})"

    def _format_vr_snapshot(vr_pose: np.ndarray, triggers: tuple, buttons: tuple) -> str:
        # vr_pose is (3, 7): rows = [left_wrist, right_wrist, head], xyz+quat
        l_pos = vr_pose[0, :3]
        r_pos = vr_pose[1, :3]
        h_pos = vr_pose[2, :3]
        lt, rt, lg, rg = triggers
        held = "+".join(BTN_NAMES[i] for i, b in enumerate(buttons) if b) or "—"
        return (
            f"head={_fmt3(h_pos)} L={_fmt3(l_pos)} R={_fmt3(r_pos)} "
            f"trig=L{lt:.2f}/R{rt:.2f} grip=L{lg:.2f}/R{rg:.2f} btns={held}"
        )

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and not stop["flag"]:
                tick_start = time.monotonic()

                vr_pose = quest.get_3pt_pose()
                buttons = quest.get_buttons()
                triggers = quest.get_controller_inputs()

                if vr_pose is None:
                    if not wait_msg:
                        print(
                            f"[teleop-kinematic] waiting for first Quest 3 packet "
                            f"(open https://{ip}:{args.quest3_http_port} on the "
                            "Quest 3 browser, accept cert, hit Connect WS) …",
                            flush=True,
                        )
                        wait_msg = True
                    body_q_mj = init_body
                    left_hand = zero_hand
                    right_hand = zero_hand
                else:
                    if not saw_first_vr_packet:
                        saw_first_vr_packet = True
                        print(
                            f"[teleop-kinematic] first VR packet  "
                            + _format_vr_snapshot(vr_pose, triggers, buttons),
                            flush=True,
                        )
                        print(
                            "[teleop-kinematic]   3-pt pose frame: rows=[L_wrist, R_wrist, head]; "
                            "xyz in robot frame (root = floor below head).",
                            flush=True,
                        )
                        print(
                            "[teleop-kinematic]   press A on EITHER controller to "
                            "activate arm tracking. The calibration is already "
                            "loaded, so no engage pose is required.",
                            flush=True,
                        )

                    # Button edge logging (every press / release)
                    for i, name in enumerate(BTN_NAMES):
                        if buttons[i] and not prev_buttons[i]:
                            print(
                                f"[teleop-kinematic] [{name}] press  "
                                + _format_vr_snapshot(vr_pose, triggers, buttons),
                                flush=True,
                            )
                        elif not buttons[i] and prev_buttons[i]:
                            print(
                                f"[teleop-kinematic] [{name}] release",
                                flush=True,
                            )

                    # A: toggle active arm tracking. Stateless -- the
                    # calibration is loaded once at startup and applied
                    # every tick; A only gates whether IK runs or whether
                    # the arms hold their last commanded q.
                    if buttons[0] and not prev_buttons[0]:
                        teleop.set_engaged(not teleop.is_engaged)
                        state = "ACTIVE" if teleop.is_engaged else "IDLE"
                        print(
                            f"[teleop-kinematic] [A] arm tracking -> {state}",
                            flush=True,
                        )

                    # B/X/Y only meaningful with --output-dir
                    if record_to_disk:
                        if buttons[1] and not prev_buttons[1] and not is_recording:
                            episode_buffer.reset()
                            episode_buffer.started_at = time.time()
                            episode_buffer.task = args.task
                            is_recording = True
                            episode_t0 = time.monotonic()
                            print(
                                f"[teleop-kinematic] [B] episode start "
                                f"(task={args.task!r}, # {episode_count + 1})",
                                flush=True,
                            )
                        if buttons[2] and not prev_buttons[2] and is_recording:
                            n = len(episode_buffer)
                            if n > 0:
                                for fr in episode_buffer.frames:
                                    exporter.add_frame(fr)
                                exporter.save_episode()
                                debug_path = _save_debug_npz(
                                    episode_buffer.debug,
                                    output_dir=args.output_dir,
                                    episode_index=episode_count,
                                    fps=args.rate,
                                    task=args.task,
                                )
                                episode_count += 1
                                print(
                                    f"[teleop-kinematic] [X] episode saved: "
                                    f"{n} frames (total saved={episode_count})",
                                    flush=True,
                                )
                                print(
                                    f"[teleop-kinematic]   debug stream "
                                    f"({len(episode_buffer.debug)} rows) -> {debug_path}",
                                    flush=True,
                                )
                            else:
                                print(
                                    "[teleop-kinematic] [X] dropping 0 frames "
                                    "(no frames captured)",
                                    flush=True,
                                )
                            episode_buffer.reset()
                            is_recording = False
                        if buttons[3] and not prev_buttons[3] and is_recording:
                            n = len(episode_buffer)
                            print(
                                f"[teleop-kinematic] [Y] discarding {n} frames",
                                flush=True,
                            )
                            episode_buffer.reset()
                            is_recording = False

                    prev_buttons = buttons

                    # Hand command: prefer XRHand 5-finger curls + thumb
                    # opposition WHENEVER they are reported (which is on
                    # every frame the headset has hand-tracking enabled,
                    # including multimodal mode where the operator is
                    # also holding controllers). Fall back to the
                    # controller trigger/grip scalar only when XRHand is
                    # not available at all (e.g. hand-tracking disabled
                    # in Quest 3 settings, or the hand has briefly left
                    # the camera FOV).
                    #
                    # IMPORTANT: do NOT gate on ``l_src``. In multimodal
                    # the WebXR client tags ``source = "controller"``
                    # because gripSpace wins for IK pose, but the same
                    # frame still carries XRHand curls + the thumb
                    # opposition signal. Gating on source kind would
                    # spuriously route us to the uniform-trigger path
                    # in multimodal and throw away the per-finger detail
                    # AND the thumb opposition correction.
                    l_curls, r_curls, l_src, r_src = quest.get_hand_curls()
                    l_oppose, r_oppose = quest.get_thumb_opposition()
                    hr = calibration.hand_range
                    l_hr = hr.left if hr is not None else None
                    r_hr = hr.right if hr is not None else None
                    if l_curls is not None:
                        left_hand = per_finger_grasp_command_from_curls_and_oppose(
                            "left", l_curls, l_oppose,
                            curl_floor=l_hr.floor if l_hr is not None else None,
                            curl_ceiling=l_hr.ceiling if l_hr is not None else None,
                            oppose_floor=l_hr.oppose_floor if l_hr is not None else None,
                            oppose_ceiling=l_hr.oppose_ceiling if l_hr is not None else None,
                        )
                        l_ratio = float(np.mean(l_curls))
                    else:
                        l_ratio_only, _ = controller_grasp_ratio(
                            left_trigger=triggers[0],
                            right_trigger=triggers[1],
                            left_grip=triggers[2],
                            right_grip=triggers[3],
                            mode=args.hand_input,
                        )
                        l_ratio = l_ratio_only
                        left_hand = grasp_command_from_ratio("left", l_ratio)

                    if r_curls is not None:
                        right_hand = per_finger_grasp_command_from_curls_and_oppose(
                            "right", r_curls, r_oppose,
                            curl_floor=r_hr.floor if r_hr is not None else None,
                            curl_ceiling=r_hr.ceiling if r_hr is not None else None,
                            oppose_floor=r_hr.oppose_floor if r_hr is not None else None,
                            oppose_ceiling=r_hr.oppose_ceiling if r_hr is not None else None,
                        )
                        r_ratio = float(np.mean(r_curls))
                    else:
                        _, r_ratio_only = controller_grasp_ratio(
                            left_trigger=triggers[0],
                            right_trigger=triggers[1],
                            left_grip=triggers[2],
                            right_grip=triggers[3],
                            mode=args.hand_input,
                        )
                        r_ratio = r_ratio_only
                        right_hand = grasp_command_from_ratio("right", r_ratio)

                    # Grasp threshold edge logging
                    l_closed = l_ratio >= GRASP_THRESH
                    r_closed = r_ratio >= GRASP_THRESH

                    def _hand_path_tag(curls: Optional[np.ndarray], oppose: Optional[float]) -> str:
                        # Single-line tag describing which path is in use
                        # this frame. Helps debug "thumb not closing" /
                        # "abad not moving" issues by surfacing the raw
                        # XRHand signals next to the close/open events.
                        if curls is None:
                            return "ctrl-uniform"
                        op_str = "no-oppose" if oppose is None else f"oppose={oppose:.2f}"
                        return f"xrhand thumb={curls[0]:.2f} {op_str}"

                    if l_closed and not prev_left_grasp_closed:
                        print(
                            f"[teleop-kinematic] hand:L close  ratio={l_ratio:.2f} "
                            f"({_hand_path_tag(l_curls, l_oppose)} "
                            f"trig={triggers[0]:.2f} grip={triggers[2]:.2f})",
                            flush=True,
                        )
                    elif not l_closed and prev_left_grasp_closed:
                        print(
                            f"[teleop-kinematic] hand:L open   ratio={l_ratio:.2f} "
                            f"({_hand_path_tag(l_curls, l_oppose)})",
                            flush=True,
                        )
                    if r_closed and not prev_right_grasp_closed:
                        print(
                            f"[teleop-kinematic] hand:R close  ratio={r_ratio:.2f} "
                            f"({_hand_path_tag(r_curls, r_oppose)} "
                            f"trig={triggers[1]:.2f} grip={triggers[3]:.2f})",
                            flush=True,
                        )
                    elif not r_closed and prev_right_grasp_closed:
                        print(
                            f"[teleop-kinematic] hand:R open   ratio={r_ratio:.2f} "
                            f"({_hand_path_tag(r_curls, r_oppose)})",
                            flush=True,
                        )
                    prev_left_grasp_closed = l_closed
                    prev_right_grasp_closed = r_closed

                    # Arms from IK
                    tick_result = teleop.step(vr_pose)
                    engaged_now = bool(tick_result.engaged)
                    last_tick_result = tick_result
                    body_q_mj = init_body.copy()
                    body_q_mj[_LEFT_ARM_MJ_SLICE] = tick_result.left_q
                    body_q_mj[_RIGHT_ARM_MJ_SLICE] = tick_result.right_q

                # Apply to MuJoCo + render viewer
                _set_kinematic_pose(
                    mujoco_mod=mujoco,
                    model=model,
                    data=data,
                    body_q_mj=body_q_mj,
                    body_qposadr=body_qposadr,
                    layout=hand_layout,
                    apply_hand_fn=apply_hand_fn,
                    left_hand_q=left_hand,
                    right_hand_q=right_hand,
                )
                viewer.sync()

                # Optional record
                if is_recording and record_to_disk and renderer is not None:
                    try:
                        ego = renderer.render_frame(
                            body_q=body_q_mj,
                            left_active=left_hand.astype(np.float64),
                            right_active=right_hand.astype(np.float64),
                        )
                    except Exception as exc:
                        print(
                            f"[teleop-kinematic] render warn (frame skipped): {exc}",
                            flush=True,
                        )
                        ego = None
                    if ego is not None:
                        body_q_pin = body_q_mj[list(MJ_TO_PIN)]
                        obs_state = assemble_observation_state(
                            robot_model, body_q_pin, left_hand, right_hand
                        )
                        # Pure-kinematics: pelvis is upright by construction,
                        # so projected_gravity is the canonical body-frame
                        # gravity vector [0, 0, -1].
                        proj_grav = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                        # action.motion_token is a placeholder filled by an
                        # offline labeling pass (SonicMotionTokenLabeler over
                        # action.commanded_body_q_mj). Recording it as zeros
                        # keeps the parquet schema-compatible with
                        # unitree_g1_sonic / GR00T datasets without forcing
                        # the live loop to run an FSQ encoder.
                        episode_buffer.push(
                            {
                                "observation.state": obs_state,
                                "observation.projected_gravity": proj_grav,
                                "action.motion_token": _ZERO_MOTION_TOKEN.copy(),
                                "action.commanded_body_q_mj": body_q_mj.copy(),
                                "action.left_hand_joints": left_hand.copy(),
                                "action.right_hand_joints": right_hand.copy(),
                                "observation.images.ego_view": np.ascontiguousarray(
                                    ego, dtype=np.uint8
                                ),
                                "task": args.task,
                            }
                        )
                        # Side-channel debug row (raw VR + IK diagnostics).
                        # vr_pose row layout: [0]=L_wrist, [1]=R_wrist, [2]=head;
                        # cols [0:3]=xyz, [3:7]=quat (4-D, see compute_3pt_pose_from_robot).
                        debug_row: dict[str, Any] = {
                            "t_episode_s": float(time.monotonic() - episode_t0),
                            # Raw VR (3 points, robot frame, root = floor below head)
                            "vr_left_wrist_pos": np.asarray(vr_pose[0, :3], dtype=np.float32),
                            "vr_left_wrist_quat": np.asarray(vr_pose[0, 3:7], dtype=np.float32),
                            "vr_right_wrist_pos": np.asarray(vr_pose[1, :3], dtype=np.float32),
                            "vr_right_wrist_quat": np.asarray(vr_pose[1, 3:7], dtype=np.float32),
                            "vr_head_pos": np.asarray(vr_pose[2, :3], dtype=np.float32),
                            "vr_head_quat": np.asarray(vr_pose[2, 3:7], dtype=np.float32),
                            # Controller analog inputs (left_trig, right_trig, left_grip, right_grip)
                            "controller_triggers": np.asarray(triggers, dtype=np.float32),
                            # Face buttons (A, B, X, Y) -- this frame's held-state
                            "controller_buttons_held": np.asarray(buttons, dtype=np.bool_),
                            # IK target / output / residual
                            "engaged": np.bool_(engaged_now),
                            "ik_left_target_pos": (
                                np.asarray(last_tick_result.left_target_pos, dtype=np.float32)
                                if last_tick_result is not None
                                else np.zeros(3, dtype=np.float32)
                            ),
                            "ik_right_target_pos": (
                                np.asarray(last_tick_result.right_target_pos, dtype=np.float32)
                                if last_tick_result is not None
                                else np.zeros(3, dtype=np.float32)
                            ),
                            "ik_left_q_rad": np.asarray(body_q_mj[_LEFT_ARM_MJ_SLICE], dtype=np.float32),
                            "ik_right_q_rad": np.asarray(body_q_mj[_RIGHT_ARM_MJ_SLICE], dtype=np.float32),
                            "ik_left_pos_err_m": np.float32(
                                last_tick_result.left_ik.pos_err_m if last_tick_result else 0.0
                            ),
                            "ik_right_pos_err_m": np.float32(
                                last_tick_result.right_ik.pos_err_m if last_tick_result else 0.0
                            ),
                            "ik_left_rot_err_rad": np.float32(
                                last_tick_result.left_ik.rot_err_rad if last_tick_result else 0.0
                            ),
                            "ik_right_rot_err_rad": np.float32(
                                last_tick_result.right_ik.rot_err_rad if last_tick_result else 0.0
                            ),
                            # Calibration diagnostics: operator wrist in
                            # the head-yaw frame (input to the affine
                            # mapping) + head yaw used for the rotation.
                            "op_left_wrist_head_yaw": (
                                np.asarray(last_tick_result.left_op_wrist_head_yaw, dtype=np.float32)
                                if last_tick_result is not None
                                else np.zeros(3, dtype=np.float32)
                            ),
                            "op_right_wrist_head_yaw": (
                                np.asarray(last_tick_result.right_op_wrist_head_yaw, dtype=np.float32)
                                if last_tick_result is not None
                                else np.zeros(3, dtype=np.float32)
                            ),
                            "head_yaw_rad": np.float32(
                                last_tick_result.head_yaw_rad if last_tick_result is not None else 0.0
                            ),
                            # Commanded full-body output
                            "commanded_body_q_mj": np.asarray(body_q_mj, dtype=np.float32),
                            "commanded_left_hand_q": np.asarray(left_hand, dtype=np.float32),
                            "commanded_right_hand_q": np.asarray(right_hand, dtype=np.float32),
                            # Quest 3 XRHand inputs (same signals as live retargeting).
                            # NaN when that side uses uniform controller grasp (no per-finger curls).
                            "quest_left_hand_curls": (
                                np.asarray(l_curls, dtype=np.float32)
                                if l_curls is not None
                                else np.full(5, np.nan, dtype=np.float32)
                            ),
                            "quest_right_hand_curls": (
                                np.asarray(r_curls, dtype=np.float32)
                                if r_curls is not None
                                else np.full(5, np.nan, dtype=np.float32)
                            ),
                            "quest_left_thumb_oppose": (
                                np.float32(l_oppose)
                                if l_oppose is not None
                                else np.float32(np.nan)
                            ),
                            "quest_right_thumb_oppose": (
                                np.float32(r_oppose)
                                if r_oppose is not None
                                else np.float32(np.nan)
                            ),
                        }
                        episode_buffer.push_debug(debug_row)

                # Status log: 1 Hz, structured, only when VR is up
                now = time.monotonic()
                if not args.quiet and (now - last_log_t) >= 1.0:
                    eng = "engaged" if engaged_now else "neutral"
                    rec = (
                        f"rec({len(episode_buffer)})"
                        if is_recording
                        else "idle"
                    )

                    def _hand_diag(side: str, curls, oppose) -> str:
                        if curls is None:
                            return f"{side}:ctrl"
                        # RAW Quest 3 per-finger curls in
                        # [thumb, index, middle, ring, pinky]. Second value is
                        # the effective lerp ratio driving motors: identical to
                        # raw when linear mapping is enabled (default); when
                        # ``apply_curl_compensation`` is True it shows post-
                        # stretch values from ``stretch_finger_curls``.
                        op_str = "--" if oppose is None else f"{oppose:+.2f}"
                        curls_arr = np.asarray(curls, dtype=np.float64)
                        if DEFAULT_APPLY_CURL_COMPENSATION:
                            st = stretch_finger_curls(curls_arr)
                        else:
                            st = curls_arr
                        return (
                            f"{side}:T{curls_arr[0]:.2f}->{st[0]:.2f}/"
                            f"I{curls_arr[1]:.2f}->{st[1]:.2f}/"
                            f"M{curls_arr[2]:.2f}->{st[2]:.2f}/"
                            f"R{curls_arr[3]:.2f}->{st[3]:.2f}/"
                            f"P{curls_arr[4]:.2f}->{st[4]:.2f} "
                            f"oppose={op_str}"
                        )

                    if vr_pose is None:
                        print(
                            f"[teleop-kinematic] VR=no-pkt | {eng} | {rec}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[teleop-kinematic] VR=ok | {eng} | {rec} | "
                            + _format_vr_snapshot(vr_pose, triggers, buttons)
                            + " | "
                            + _hand_diag("L", l_curls, l_oppose)
                            + " | "
                            + _hand_diag("R", r_curls, r_oppose),
                            flush=True,
                        )
                    last_log_t = now

                # Cadence
                next_tick += period
                sleep_s = next_tick - time.monotonic()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_tick = time.monotonic()
    finally:
        if is_recording and len(episode_buffer) > 0 and exporter is not None:
            print(
                f"[teleop-kinematic] auto-saving open episode "
                f"({len(episode_buffer)} frames) on shutdown",
                flush=True,
            )
            for fr in episode_buffer.frames:
                exporter.add_frame(fr)
            exporter.save_episode()
            try:
                debug_path = _save_debug_npz(
                    episode_buffer.debug,
                    output_dir=args.output_dir,
                    episode_index=episode_count,
                    fps=args.rate,
                    task=args.task,
                )
                print(
                    f"[teleop-kinematic] auto-saved debug stream "
                    f"({len(episode_buffer.debug)} rows) -> {debug_path}",
                    flush=True,
                )
                episode_count += 1
            except Exception as exc:
                print(
                    f"[teleop-kinematic] WARN: failed to write debug NPZ on shutdown: {exc}",
                    flush=True,
                )
        try:
            quest.stop()
        except Exception:
            pass
        if renderer is not None:
            try:
                renderer.close()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
