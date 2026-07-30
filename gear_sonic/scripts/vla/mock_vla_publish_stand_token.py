"""Mock-VLA stand-still motion-token publisher (M2 acceptance gate helper).

Publishes a constant 64-D SONIC motion token plus left/right hand joints over
ZMQ in the same wire format the C++ deploy harness expects from the real
GR00T N1.7 VLA. With this in the loop the X2 deploy can be exercised in sim
mode WITHOUT a trained VLA model — proving that the post-VLA path
(``ZmqMotionTokenSource`` → SONIC decoder → ``x2_mujoco_ros_bridge``)
is correctly wired end-to-end.

Wire format
-----------

This script uses ``gear_sonic.utils.teleop.zmq.zmq_planner_sender.pack_pose_message``
which produces::

    [topic_bytes][1280-byte JSON header][concatenated f32 fields]

with ``topic = "pose"``. That's the exact byte layout the C++
``ZMQEndpointInterface`` / ``ZMQPackedMessageSubscriber`` consume.

Stand-still latent
------------------

By default we publish a zeroed 64-D token vector. This corresponds to the
SONIC encoder's "do nothing" embedding when ``LATENT_INITIAL_MOTION_TOKEN``
is set to zeros — the deploy harness's safety stack (soft-start ramp +
default-pose anchor) will keep the robot upright in MuJoCo while the policy
"holds the latent flat".

If a more meaningful stand-still token is captured later via
``capture_x2_initial_motion_token.py``, point ``--token-file`` at the saved
``.npy`` to publish that instead.

Hand joints default to all-zeros (open hands).

Usage
-----

::

    # Terminal 1 (sim):
    cd gear_sonic_deploy && bash deploy_x2.sh sim --input-type zmq

    # Terminal 2 (mock VLA):
    .venv/bin/python gear_sonic/scripts/vla/mock_vla_publish_stand_token.py \\
        --port 5556 --rate 50

    # Terminal 3 (state dump):
    .venv/bin/python gear_sonic/scripts/dump_x2_debug.py \\
        --port 5557 --duration 10

Acceptance: the C++ deploy must keep MuJoCo standing for the duration of
``mock_vla_publish_stand_token.py`` without tilt-watchdog tripping, and
``dump_x2_debug.py`` must show the standing token being received and the
joint positions remaining within 0.05 rad of the trained default pose.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import signal
import sys
import time
from typing import Optional

import numpy as np
import zmq

# Allow running this script before `pip install -e gear_sonic`.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402


SONIC_MOTION_TOKEN_DIM: int = 64
DEFAULT_HAND_DOF: int = 10  # X2 OmniHand. Use 7 for the G1-compatible variant.
DEFAULT_PUB_RATE_HZ: float = 50.0  # X2 deploy control loop runs at 50 Hz tokenizer.
NUM_BODY_DOFS: int = 31  # X2 Ultra body DOFs (MuJoCo URDF order).

# Trained "stand pose" (radians) -- the X2 deploy harness's
# StandStillReference / safety stack converges to exactly these values.
# Mirrors `default_angles` in
# `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp`.
# We hard-code it on the Python side rather than re-parsing the C++ header so
# the mock-VLA helper stays import-light (no codegen tooling at runtime).
DEFAULT_STAND_POSE_MUJOCO_RAD: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # right leg
    0.0, 0.0, 0.0,                                # waist
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,           # left arm
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,          # right arm
    0.0, 0.0,                                     # head
)
assert len(DEFAULT_STAND_POSE_MUJOCO_RAD) == NUM_BODY_DOFS

# Identity quaternion (xyzw): scipy convention, matches the X2 deploy's
# StandStillReference + the convention the .pkl motion files use.
IDENTITY_ROOT_QUAT_XYZW: tuple[float, ...] = (0.0, 0.0, 0.0, 1.0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host",
        default="*",
        help="ZMQ bind interface ('*' for all).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5556,
        help="ZMQ PUB port. Must match the deploy harness's --pose-port.",
    )
    parser.add_argument(
        "--topic",
        default="pose",
        help="ZMQ topic prefix the deploy subscribes to.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=DEFAULT_PUB_RATE_HZ,
        help="Publish rate in Hz.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Total run time in seconds (0 = run until Ctrl-C).",
    )
    parser.add_argument(
        "--hand-dof",
        type=int,
        choices=(7, 10),
        default=DEFAULT_HAND_DOF,
        help="DOF count per hand. 10 = full X2 OmniHand, 7 = G1-compatible view.",
    )
    parser.add_argument(
        "--token-file",
        type=str,
        default=None,
        help=(
            "Optional path to a .npy file containing a saved SONIC motion token "
            "(shape (64,) float32). When omitted, an all-zeros token is used."
        ),
    )
    parser.add_argument(
        "--protocol-version",
        type=int,
        choices=(3, 4),
        default=4,
        help="Wire protocol version. v4 includes the 'count' field.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-tick stdout (just print every 100 ticks).",
    )
    return parser.parse_args(argv)


def _load_token(path: Optional[str]) -> np.ndarray:
    if path is None:
        return np.zeros(SONIC_MOTION_TOKEN_DIM, dtype=np.float32)
    arr = np.load(path).astype(np.float32, copy=False)
    if arr.shape != (SONIC_MOTION_TOKEN_DIM,):
        raise ValueError(
            f"--token-file must be shape ({SONIC_MOTION_TOKEN_DIM},) float32, "
            f"got shape={arr.shape} dtype={arr.dtype}"
        )
    return arr


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    token = _load_token(args.token_file)
    left_hand = np.zeros(args.hand_dof, dtype=np.float32)
    right_hand = np.zeros(args.hand_dof, dtype=np.float32)
    # Body refs: trained stand pose, identity root quaternion. The C++
    # ZmqPoseInputSource consumes these as the ReferenceMotion drop-in
    # (joint_pos_mj + root_quat_xyzw). Without these fields the deploy
    # would still work (it falls back to default_angles internally) but we
    # keep them on the wire so dump_x2_debug.py / parity tooling sees the
    # exact pose the VLA is "asking for".
    body_pose_mj = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32)
    root_quat_xyzw = np.array(IDENTITY_ROOT_QUAT_XYZW, dtype=np.float32)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 10)
    sock.setsockopt(zmq.LINGER, 0)
    bind_url = f"tcp://{args.host}:{args.port}"
    sock.bind(bind_url)
    print(f"[mock-VLA] PUB bound on {bind_url} (topic={args.topic!r})", flush=True)
    print(
        f"[mock-VLA] motion_token={'zeros' if args.token_file is None else args.token_file} "
        f"({SONIC_MOTION_TOKEN_DIM}D)  hand_dof={args.hand_dof}  rate={args.rate} Hz",
        flush=True,
    )

    # PUB-SUB requires the subscriber to connect AND complete the SUBSCRIBE
    # handshake before the publisher's first message lands. ZMQ has no
    # late-join guarantee for PUB sockets — early publishes get dropped. Burn
    # a couple of ticks so the deploy's SUB has time to attach.
    time.sleep(0.2)

    period = 1.0 / max(args.rate, 1e-6)
    deadline = float("inf") if args.duration <= 0.0 else time.monotonic() + args.duration

    stop_requested = {"flag": False}

    def _on_signal(signum, _frame):  # type: ignore[unused-argument]
        print(f"[mock-VLA] caught signal {signum}, shutting down…", flush=True)
        stop_requested["flag"] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    tick = 0
    next_tick = time.monotonic()
    try:
        while not stop_requested["flag"] and time.monotonic() < deadline:
            payload = {
                # Body reference for the C++ deploy's ZmqPoseInputSource
                # (drop-in for ReferenceMotion).
                "joint_pos_mj": body_pose_mj,
                "root_quat_xyzw": root_quat_xyzw,
                # SONIC latent (forward-compat: not consumed by the v0
                # deploy, which still runs the encoder ONNX in-process).
                "motion_token": token,
                # AimDK HAL passthrough: hand joints flow out of band.
                "left_hand_joints": left_hand,
                "right_hand_joints": right_hand,
                # frame_index lets the C++ deploy log monotonic VLA ticks.
                # int64 to match the upstream LeRobot exporter's convention.
                "frame_index": np.array([tick], dtype=np.int64),
            }
            msg = pack_pose_message(payload, topic=args.topic, version=args.protocol_version)
            sock.send(msg, flags=zmq.NOBLOCK)

            if not args.quiet and tick % 50 == 0:
                print(
                    f"[mock-VLA] tick={tick:6d} "
                    f"|token|={float(np.linalg.norm(token)):.3f} "
                    f"|lh|={float(np.linalg.norm(left_hand)):.3f} "
                    f"|rh|={float(np.linalg.norm(right_hand)):.3f}",
                    flush=True,
                )

            tick += 1
            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                # If we fall behind, snap the schedule back to now to avoid
                # spiraling drift.
                next_tick = time.monotonic()
    finally:
        sock.close(linger=0)
        ctx.term()
        print(f"[mock-VLA] done after {tick} ticks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
