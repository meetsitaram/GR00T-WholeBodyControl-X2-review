"""Re-publish a recorded LeRobot v2.1 episode to a live X2 deploy.

Reads ``action.{body_q_mj, motion_token, left_hand_joints, right_hand_joints}``
straight out of the recorded parquet and PUBs it on the deploy's ``pose``
ZMQ topic (port 5556) at the dataset's native FPS. The C++ deploy on PC2
(``deploy_x2.sh sim --vla`` or the real-robot daemons) consumes the wire
exactly as it would from the Quest 3 manager or the live VLA bridge, so
the trajectory replays end-to-end through SONIC -> motors.

This is the SONIC counterpart to ``replay_x2_kinematic.py`` (MuJoCo-only).
The kinematic replay shows what the parquet says; this script shows what
the deploy + SONIC actually do with it.

Usage (sim deploy)
------------------

In one shell, bring up the sim deploy::

    gear_sonic_deploy/deploy_x2.sh sim --vla --sim-with-omnihand --sim-viewer \\
        --model $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501

In a second shell, replay an episode (rate-controlled, soft-start, Ctrl-C
to abort safely)::

    python -m gear_sonic.scripts.replay_x2_dataset \\
        --dataset x2_reach_and_retract_v1 --episode 0

Usage (real robot via PC2)
--------------------------

The script just binds a PUB on the local laptop. The PC2 SONIC daemon
must already be running and configured to SUB at the laptop's IP
(``--laptop-host`` passed to ``x2_pc2_daemons.sh start``). After that
this script is a drop-in:

    python -m gear_sonic.scripts.replay_x2_dataset \\
        --dataset x2_reach_and_retract_v1 --episode 0

Safety
------

* The script binds, prints a banner, then COUNTS DOWN before publishing.
  Cancel with Ctrl-C in the countdown window if the workspace isn't clear.
* On Ctrl-C during playback, it sends ~0.5 s of "hold the last commanded
  pose" frames then exits. SONIC's safety stack decays PD gains in
  ~200 ms so this is enough for a soft stop.
* The recorded ``action.body_q_mj`` is what the operator commanded, NOT
  what the robot actually achieved. Object positions on the table MUST
  match the recording, or the hand will swing into empty air / collide.

What the wire payload contains
------------------------------

Per-frame ``pose`` message (protocol version 4, with the v5 future-window
promotion fields appended) mirrors what the live VLA bridge / Quest 3
manager publish::

    joint_pos_mj          f32 (31,)   <- action.body_q_mj from parquet
    root_quat_xyzw        f32 (4,)    <- identity quat (recipe 6.3 convention;
                                         the deploy uses ``base_quat`` from
                                         its measured x2_debug for actual
                                         orientation -- root_quat on the
                                         wire is the heading REFERENCE only)
    motion_token          f32 (64,)   <- action.motion_token from parquet
                                         (re-tokenized by the deploy from
                                         the future window below; on the
                                         wire it's a debug echo, not used)
    left_hand_joints      f32 (10,)   <- action.left_hand_joints from parquet
    right_hand_joints     f32 (10,)   <- action.right_hand_joints from parquet
    frame_index           i64 (1,)    <- monotonic across the replay run

    joint_pos_mj_future   f32 (9,31)  <- body_q at f+5, f+10, ..., f+45
                                         (at 50 Hz native = 0.1 s spacing),
                                         tail-tiled past episode end
    root_quat_xyzw_future f32 (9,4)   <- identity quat per slot (no recorded
                                         root in the parquet; consistent with
                                         the current-frame root_quat above)
    joint_vel_mj_future   f32 (9,31)  <- finite-diff of joint_pos_mj_future
                                         / future_dt_s, first slot diff'd
                                         against body_q[f]
    frame_index_future    i64 (9,)    <- wire_frame + [1..9]
    future_dt_s           f32 (1,)    <- 0.1 (per-slot spacing in seconds)

The v5 fields are MANDATORY for body motion: the C++ deploy ignores
``motion_token`` from the wire and instead re-tokenizes the trajectory
from ``joint_pos_mj_future`` each tick (per ``agi_x2_deploy_onnx_ref``
+ the SONIC token-to-pose decoder). Without them it back-fills with the
trained ``default_angles`` stand pose and the body stays in idle_stand
while only the OmniHand pose-streamer fingers move -- the exact bug this
script previously hit.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from gear_sonic.scripts.replay_x2_kinematic import (  # noqa: E402
    _BODY_ACTION_CANDIDATES,
    _episode_parquet_path,
    _read_chunk_size,
    _resolve_dataset_path,
)


# Match the conventions in ``live_vla_publish_motion_token.py`` and
# ``mock_vla_publish_stand_token.py`` so the C++ deploy SUB sees an
# identical envelope to a live run.
DEFAULT_PUB_HOST = "*"
DEFAULT_PUB_PORT = 5556
DEFAULT_PUB_TOPIC = "pose"
DEFAULT_PROTOCOL_VERSION = 4

NUM_BODY_DOFS = 31
NUM_MOTION_TOKEN = 64
NUM_HAND_DOF_PER_SIDE = 10
IDENTITY_QUAT_XYZW = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

# v5 future-window contract -- mirrors live_vla_publish_motion_token.py.
# The deploy promotes a frame to v5 mode (i.e. actually tracks the wire
# trajectory) iff BOTH ``joint_pos_mj_future`` and ``root_quat_xyzw_future``
# are present. Without them the deploy back-fills the future window with the
# trained ``default_angles`` stand pose and the body holds idle_stand --
# which is exactly the "fingers move but body doesn't" symptom the replay
# tool had before this fix.
NUM_FUTURE_SLOTS = 9
FUTURE_DT_S = 0.1
_FUTURE_DT_FIELD = np.array([FUTURE_DT_S], dtype=np.float32)
_IDENTITY_QUAT_FUTURE = np.broadcast_to(
    IDENTITY_QUAT_XYZW, (NUM_FUTURE_SLOTS, 4)
).astype(np.float32, copy=True)
_FUTURE_OFFSET_BASE = np.arange(1, NUM_FUTURE_SLOTS + 1, dtype=np.int64)


def _build_future_window(
    body_q: np.ndarray, f: int, step: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build the 9-slot future ``joint_pos_mj`` + ``joint_vel_mj`` window.

    For source frame ``f``, the 9 future slots are taken at
    ``body_q[f + step, f + 2*step, ..., f + 9*step]``, tail-tiled with the
    final episode frame for indices past the end. ``step`` is the per-slot
    stride in source frames; at the dataset's native 50 Hz it is 5 frames
    (= 0.1 s, matching ``FUTURE_DT_S``).

    ``joint_vel_mj_future`` is the per-slot finite difference of
    ``joint_pos_mj_future`` divided by ``FUTURE_DT_S``, with the first slot
    diff'd against the current frame ``body_q[f]``. Shipping it lets the
    deploy skip its backward-finite-diff path (per the bridge's
    ``_idle_future_payload_fields`` docstring).
    """
    if step <= 0:
        raise ValueError(f"future-window step must be >=1, got {step}")
    n_frames = body_q.shape[0]
    idx = np.minimum(f + step * _FUTURE_OFFSET_BASE, n_frames - 1)
    jpos = body_q[idx].astype(np.float32, copy=False)
    prev = np.concatenate(
        [body_q[f : f + 1].astype(np.float32, copy=False), jpos[:-1]], axis=0
    )
    jvel = ((jpos - prev) / FUTURE_DT_S).astype(np.float32)
    return jpos, jvel


def _stack_column(table, col: str) -> np.ndarray:
    """Stack a list-of-arrays parquet column into a 2-D numpy array."""
    return np.stack(table[col].to_numpy())


def _load_episode_payload(
    parquet_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read body_q_mj, motion_token, left/right hand joints out of a parquet.

    Returns ``(body_q, token, left_q, right_q)`` as ``float32`` arrays
    with the wire dtypes the deploy expects.
    """
    import pyarrow.parquet as pq

    if not parquet_path.is_file():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")

    table = pq.read_table(parquet_path)
    body_col = next(
        (c for c in _BODY_ACTION_CANDIDATES if c in table.column_names),
        None,
    )
    if body_col is None:
        raise ValueError(
            f"Parquet {parquet_path} missing body action column. "
            f"Tried {_BODY_ACTION_CANDIDATES}. "
            f"Available: {table.column_names}"
        )

    required = (
        "action.motion_token",
        "action.left_hand_joints",
        "action.right_hand_joints",
    )
    missing = [c for c in required if c not in table.column_names]
    if missing:
        raise ValueError(
            f"Parquet {parquet_path} missing required columns: {missing}. "
            f"Available: {table.column_names}"
        )

    body_q = _stack_column(table, body_col).astype(np.float32)
    token = _stack_column(table, "action.motion_token").astype(np.float32)
    left_q = _stack_column(table, "action.left_hand_joints").astype(np.float32)
    right_q = _stack_column(table, "action.right_hand_joints").astype(np.float32)

    if body_q.shape[1] != NUM_BODY_DOFS:
        raise ValueError(
            f"{body_col} width {body_q.shape[1]} != expected {NUM_BODY_DOFS}"
        )
    if token.shape[1] != NUM_MOTION_TOKEN:
        raise ValueError(
            f"action.motion_token width {token.shape[1]} != expected {NUM_MOTION_TOKEN}"
        )
    if left_q.shape[1] != NUM_HAND_DOF_PER_SIDE:
        raise ValueError(
            f"action.left_hand_joints width {left_q.shape[1]} != "
            f"expected {NUM_HAND_DOF_PER_SIDE}"
        )
    if right_q.shape[1] != NUM_HAND_DOF_PER_SIDE:
        raise ValueError(
            f"action.right_hand_joints width {right_q.shape[1]} != "
            f"expected {NUM_HAND_DOF_PER_SIDE}"
        )

    return body_q, token, left_q, right_q


def _read_fps(dataset_root: Path, fallback: int = 50) -> int:
    import json

    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        return fallback
    try:
        info = json.loads(info_path.read_text())
        return int(info.get("fps", fallback))
    except Exception:
        return fallback


def _build_payload(
    body_q: np.ndarray,
    token: np.ndarray,
    left_q: np.ndarray,
    right_q: np.ndarray,
    f: int,
    wire_frame: int,
    future_step: int,
) -> dict:
    """Build the v5-promoted wire payload for source frame ``f``.

    ``body_q`` / ``token`` / ``left_q`` / ``right_q`` are the WHOLE-episode
    arrays; the current frame is sliced internally and clamped to the last
    available frame, and the 9-slot future window is built off ``body_q``
    via :func:`_build_future_window`. The deploy needs the future-window
    fields to track our trajectory -- without them it back-fills with the
    trained idle stand pose and the body never moves.
    """
    cur_f = min(f, body_q.shape[0] - 1)
    jpos_future, jvel_future = _build_future_window(body_q, cur_f, future_step)
    return {
        "joint_pos_mj": body_q[cur_f].astype(np.float32),
        "root_quat_xyzw": IDENTITY_QUAT_XYZW,
        "motion_token": token[cur_f].astype(np.float32),
        "left_hand_joints": left_q[cur_f].astype(np.float32),
        "right_hand_joints": right_q[cur_f].astype(np.float32),
        "frame_index": np.array([wire_frame], dtype=np.int64),
        # v5 promotion fields -- presence of joint_pos_mj_future +
        # root_quat_xyzw_future is what flips the deploy out of idle-stand.
        "joint_pos_mj_future": jpos_future,
        "root_quat_xyzw_future": _IDENTITY_QUAT_FUTURE,
        "joint_vel_mj_future": jvel_future,
        "frame_index_future": (
            np.int64(wire_frame) + _FUTURE_OFFSET_BASE
        ).astype(np.int64),
        "future_dt_s": _FUTURE_DT_FIELD,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", required=True,
                   help="Short name under data/lerobot/ or absolute path.")
    p.add_argument("--episode", type=int, required=True,
                   help="0-based episode index within the dataset.")
    p.add_argument("--pub-host", default=DEFAULT_PUB_HOST,
                   help=f"Bind interface for the pose PUB (default {DEFAULT_PUB_HOST}).")
    p.add_argument("--pub-port", type=int, default=DEFAULT_PUB_PORT,
                   help=f"Bind port for the pose PUB (default {DEFAULT_PUB_PORT}).")
    p.add_argument("--pub-topic", default=DEFAULT_PUB_TOPIC,
                   help=f"ZMQ topic (default {DEFAULT_PUB_TOPIC!r}).")
    p.add_argument("--rate", type=float, default=None,
                   help="Override publish rate Hz. Default = dataset fps "
                        "from meta/info.json.")
    p.add_argument("--rate-scale", type=float, default=1.0,
                   help="Multiplier on rate (0.5 = half speed; default 1.0).")
    p.add_argument("--countdown", type=float, default=3.0,
                   help="Seconds of warm-up holding frame 0 before stepping "
                        "through the trajectory (default 3.0). Set to 0 to "
                        "publish immediately.")
    p.add_argument("--hold-on-exit", type=float, default=0.5,
                   help="Seconds to keep publishing the LAST frame after the "
                        "trajectory ends or Ctrl-C arrives (default 0.5). "
                        "Gives SONIC time to ramp down on the same pose.")
    p.add_argument("--loop", action="store_true",
                   help="Loop the episode indefinitely.")
    p.add_argument("--pc2-host", default=None,
                   help="Optional: PC2 IP. Informational only -- the PUB "
                        "binds locally; PC2 connects out. Logged in the "
                        "banner so the operator can verify routing.")
    p.add_argument("--dry-run", action="store_true",
                   help="Load the parquet and print stats; do NOT bind ZMQ "
                        "or publish anything.")
    return p.parse_args(argv)


def main() -> int:
    args = _parse_args()

    dataset_root = _resolve_dataset_path(args.dataset)
    chunk_size = _read_chunk_size(dataset_root)
    parquet_path = _episode_parquet_path(
        dataset_root, args.episode, chunk_size=chunk_size,
    )

    body_q, token, left_q, right_q = _load_episode_payload(parquet_path)
    n_frames = body_q.shape[0]

    native_fps = _read_fps(dataset_root)
    fps = args.rate if args.rate else native_fps
    effective_rate = fps * args.rate_scale
    if effective_rate <= 0:
        raise SystemExit(f"effective rate {effective_rate} Hz must be > 0")
    period_s = 1.0 / effective_rate
    duration_s = n_frames / effective_rate

    # Future-window stride is in SOURCE frames per ``FUTURE_DT_S`` of the
    # native recording -- not the publish cadence. We want each future slot
    # to represent the same physical 0.1 s the bridge/policy uses, so the
    # deploy's tokenizer sees a window with matching dynamics regardless of
    # ``--rate`` or ``--rate-scale``.
    future_step = max(1, int(round(native_fps * FUTURE_DT_S)))

    print()
    print("=" * 72)
    print("X2 DATASET REPLAY -- live PUB to deploy")
    print("=" * 72)
    print(f"  dataset      : {dataset_root}")
    print(f"  episode      : {args.episode}  ({parquet_path.name})")
    print(f"  frames       : {n_frames}")
    print(f"  fps          : {fps:.2f} Hz (scaled to {effective_rate:.2f} Hz)")
    print(f"  duration     : {duration_s:.2f} s per loop")
    print(f"  pub          : tcp://{args.pub_host}:{args.pub_port} "
          f"topic={args.pub_topic!r} protocol=v{DEFAULT_PROTOCOL_VERSION} "
          f"(+ v5 future window: {NUM_FUTURE_SLOTS} slots @ "
          f"{FUTURE_DT_S * 1000:.0f} ms = {future_step} src-frames each)")
    if args.pc2_host:
        print(f"  pc2_host     : {args.pc2_host} (informational)")
    print(f"  loop         : {args.loop}")
    print(f"  hold_on_exit : {args.hold_on_exit:.2f} s")
    print()
    print("  !!  THE ROBOT WILL PHYSICALLY MOVE  !!")
    print("  - Clear the workspace and have e-stop in reach.")
    print("  - Object position must match the recording, or the hand")
    print("    will swing into empty air.")
    print("  - Ctrl-C aborts cleanly (last-frame hold + exit).")
    print("=" * 72)
    print(flush=True)

    if args.dry_run:
        print("[dry-run] no ZMQ socket bound; exiting.")
        return 0

    import zmq

    from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 10)
    pub.setsockopt(zmq.LINGER, 0)
    pub_url = f"tcp://{args.pub_host}:{args.pub_port}"
    pub.bind(pub_url)
    print(f"[replay] PUB bound on {pub_url}", flush=True)

    abort = {"stop": False}

    def _sigint_handler(signum, frame):  # noqa: ARG001
        abort["stop"] = True
        print("\n[replay] Ctrl-C received -- finishing current frame, "
              "holding last pose, and exiting.", flush=True)

    signal.signal(signal.SIGINT, _sigint_handler)
    signal.signal(signal.SIGTERM, _sigint_handler)

    def _send(payload: dict) -> None:
        msg = pack_pose_message(
            payload, topic=args.pub_topic, version=DEFAULT_PROTOCOL_VERSION,
        )
        try:
            pub.send(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    # Soft start: publish frame 0 for ``--countdown`` seconds so SONIC
    # has a stable target before the trajectory begins. Counts down
    # visibly so the operator has a chance to abort.
    wire_frame = 0
    if args.countdown > 0:
        n_warm = max(1, int(args.countdown * effective_rate))
        print(f"[replay] warm-up: holding frame 0 for {args.countdown:.1f} s "
              f"({n_warm} ticks) ...", flush=True)
        last_log = -1.0
        next_deadline = time.monotonic()
        for i in range(n_warm):
            if abort["stop"]:
                print("[replay] aborted during warm-up.", flush=True)
                return 130
            _send(_build_payload(
                body_q, token, left_q, right_q, 0, wire_frame, future_step,
            ))
            wire_frame += 1
            now = time.monotonic()
            secs_left = args.countdown - (i / effective_rate)
            if secs_left <= last_log - 0.5 or last_log < 0:
                if secs_left >= 0.05:
                    print(f"  ... starting in {secs_left:0.1f} s", flush=True)
                last_log = secs_left
            next_deadline += period_s
            sleep = next_deadline - now
            if sleep > 0:
                time.sleep(sleep)

    # Trajectory playback.
    loop_idx = 0
    try:
        while True:
            loop_idx += 1
            print(f"[replay] === loop {loop_idx} : publishing {n_frames} "
                  f"frames @ {effective_rate:.1f} Hz ===", flush=True)
            next_deadline = time.monotonic()
            t0 = next_deadline
            for f in range(n_frames):
                if abort["stop"]:
                    break
                _send(_build_payload(
                    body_q, token, left_q, right_q, f, wire_frame, future_step,
                ))
                wire_frame += 1
                if (f + 1) % int(max(1, effective_rate)) == 0:
                    elapsed = time.monotonic() - t0
                    print(f"  frame {f + 1:4d}/{n_frames}  "
                          f"t={elapsed:5.2f}s  wire_frame={wire_frame}",
                          flush=True)
                next_deadline += period_s
                sleep = next_deadline - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
            if abort["stop"] or not args.loop:
                break
            print(f"[replay] loop {loop_idx} done; restarting in 0.5 s "
                  "(Ctrl-C to stop) ...", flush=True)
            time.sleep(0.5)
    finally:
        # Soft stop: hold the most recent commanded frame for a few ticks
        # so SONIC ramps down on a stable target rather than a sudden
        # cutoff. Uses the LAST frame we actually published, not the
        # parquet's last frame.
        if args.hold_on_exit > 0:
            last_f = min(n_frames - 1, f if 'f' in locals() else 0)
            n_hold = max(1, int(args.hold_on_exit * effective_rate))
            print(f"[replay] holding last pose (src frame {last_f}) for "
                  f"{args.hold_on_exit:.2f} s ({n_hold} ticks) ...",
                  flush=True)
            next_deadline = time.monotonic()
            for _ in range(n_hold):
                _send(_build_payload(
                    body_q, token, left_q, right_q, last_f, wire_frame,
                    future_step,
                ))
                wire_frame += 1
                next_deadline += period_s
                sleep = next_deadline - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
        pub.close(linger=0)
        ctx.term()
        print("[replay] PUB closed. Bye.", flush=True)
    return 0 if not abort["stop"] else 130


if __name__ == "__main__":
    sys.exit(main())
