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
        help="(Reserved for offline post-processing.) The live recorder "
             "does NOT load this; the deploy's tracking policy follows "
             "joint_pos_mj as reference. Provide this only if you plan "
             "to attach FSQ motion_token labels in a follow-up offline "
             "pass; the recorder itself never reads it.",
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
    from gear_sonic.utils.teleop.x2_dataset_recorder import (
        RecorderConfig,
        X2DatasetRecorder,
    )

    if not args.teleop_only:
        if args.output_dir is None or not args.task:
            raise SystemExit(
                "Error: --output-dir and --task are required unless --teleop-only is set."
            )

    cfg = RecorderConfig(
        output_dir=args.output_dir,
        task=args.task,
        sonic_checkpoint=args.sonic_checkpoint,
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
