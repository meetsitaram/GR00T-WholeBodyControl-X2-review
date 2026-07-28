#!/usr/bin/env python3
"""Trigger a locomotion clip mid-session on a running X2 stack.

Sibling to :file:`play_gesture.py`. Both scripts publish on the same
``motion_clip_cmd`` wire (default port
:data:`gear_sonic.utils.teleop.motion_clip_session.MOTION_CLIP_CMD_DEFAULT_PORT`);
the difference is the ``kind`` field on the play payload and the
session-side branch it picks inside
:class:`gear_sonic.utils.teleop.motion_clip_session.MotionClipSession`:

* ``play_gesture``  -> ``kind="gesture"``  -> PKL frame-0 yaw rebased
  onto the robot's current heading (arm wave, sit-down, bow, ...).
* ``play_locomotion`` -> ``kind="locomotion"`` -> authored yaw kept
  verbatim so a walk / turn / sidestep clip steers the body the way
  the mocap intended.

There is no catalog for locomotion clips -- the operator passes
``--pkl <path>`` directly. The SOMA locomotion library has thousands
of single-cycle clips plus a handful of pre-stitched home loops
(``gear_sonic/data/motions/x2_ultra_walk_demo_v*.pkl``); a curated
YAML would be more maintenance than it's worth at this stage.

Usage::

    # Stitched home loop (walk forward, turn 180, walk back).
    play_locomotion --pkl gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl

    # Single-cycle forward walk, then idle stand on completion.
    play_locomotion --pkl ~/locomotion_smoke/Loop_Forward_Walk_001__A018.pkl

    # Specific motion-key inside a multi-clip PKL.
    play_locomotion \\
        --pkl gear_sonic/data/motions/x2_ultra_locowalk.pkl \\
        --motion-key Loop_Forward_Walk_001__A018

    play_locomotion --release   # send stop, return immediately

On natural completion the recorder snaps back to its idle-stand
fallback (locomotion clips don't get a hold-after pose -- the robot
should stand, not freeze mid-stride). On SIGINT during the block,
the script publishes ``{"action": "stop"}`` before exiting with
code 130 so the recorder snaps to idle immediately.

See :file:`pkl_direct_commands.md` for end-to-end sim + real recipe.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import zmq  # noqa: E402

from gear_sonic.utils.teleop.motion_clip_session import (  # noqa: E402
    MOTION_CLIP_CMD_DEFAULT_PORT,
    MOTION_CLIP_CMD_DEFAULT_TOPIC,
    MotionClipEntry,
    estimate_duration_s,
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--pkl", type=Path, default=None,
        help="Path to the motion_lib PKL. Required for play (omit "
             "when using --release).",
    )
    parser.add_argument(
        "--release", action="store_true",
        help="Publish a single 'stop' command and exit immediately. "
             "Use this to abort a long stitched clip from another "
             "terminal without Ctrl-C'ing the blocked play_locomotion.",
    )
    parser.add_argument(
        "--motion-key", type=str, default=None,
        help="Motion key inside a multi-clip PKL. Default: first key. "
             "Substring match allowed by the recorder side.",
    )
    parser.add_argument(
        "--start-frame", type=int, default=0,
        help="Optional clip start_frame override.",
    )
    parser.add_argument(
        "--n-frames", type=int, default=None,
        help="Optional clip n_frames override (default: to end).",
    )
    parser.add_argument(
        "--host", default="localhost",
        help="Recorder host. Default 'localhost' assumes the recorder "
             "and trigger script are on the same machine.",
    )
    parser.add_argument(
        "--port", type=int, default=MOTION_CLIP_CMD_DEFAULT_PORT,
        help="motion_clip_cmd port. Must match the recorder's "
             "--motion-clip-cmd-port.",
    )
    parser.add_argument(
        "--topic", default=MOTION_CLIP_CMD_DEFAULT_TOPIC,
        help="motion_clip_cmd topic. Must match the recorder's "
             "--motion-clip-cmd-topic.",
    )
    parser.add_argument(
        "--linger-ms", type=int, default=200,
        help="Milliseconds to sleep after PUB connect (before sending) "
             "to give the recorder's SUB time to wire up, and again "
             "after stop on SIGINT. ZMQ's slow-joiner can drop the "
             "very first message if this is too short.",
    )
    parser.add_argument(
        "--target-rate-hz", type=float, default=50.0,
        help="Publish rate the recorder is running at; used to "
             "estimate clip duration so the script knows how long to "
             "block.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to count down (client-side) BEFORE publishing "
             "the play command. Useful for letting the operator step "
             "clear of the robot before a walk starts. Ctrl-C during "
             "the countdown exits cleanly without sending anything; "
             "the recorder's current state is left untouched.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress operator log lines.",
    )
    return parser.parse_args(argv)


def _connect_pub(args: argparse.Namespace) -> "zmq.Socket[Any]":
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, max(0, args.linger_ms))
    url = f"tcp://{args.host}:{args.port}"
    pub.connect(url)
    if not args.quiet:
        print(
            f"[play_locomotion] PUB connect {url} topic={args.topic!r}",
            flush=True,
        )
    return pub


def _send(pub: "zmq.Socket[Any]", topic: str, payload: dict[str, Any]) -> None:
    pub.send_multipart([
        topic.encode("ascii"),
        json.dumps(payload).encode("utf-8"),
    ])


def _countdown(delay_s: float, *, quiet: bool) -> None:
    """Sleep ``delay_s`` seconds with a once-per-second countdown.

    Raises :class:`KeyboardInterrupt` on SIGINT (caller decides
    whether to translate that into a wire stop). No-op when
    ``delay_s <= 0``.
    """
    if delay_s <= 0:
        return
    if not quiet:
        print(
            f"[play_locomotion] countdown {delay_s:.0f}s before play "
            f"(Ctrl-C to abort, recorder state untouched)",
            flush=True,
        )
    full_secs = int(delay_s)
    for remaining in range(full_secs, 0, -1):
        if not quiet:
            print(f"[play_locomotion] T-{remaining}", flush=True)
        time.sleep(1.0)
    leftover = delay_s - float(full_secs)
    if leftover > 0:
        time.sleep(leftover)


def _build_entry(args: argparse.Namespace) -> tuple[MotionClipEntry, dict[str, Any]]:
    """Build the local entry (for duration estimation) and the wire
    payload, both tagged ``kind="locomotion"``.
    """
    pkl_path: Path = args.pkl
    motion_key: Optional[str] = args.motion_key
    start_frame = int(args.start_frame)
    n_frames = None if args.n_frames is None else int(args.n_frames)

    entry = MotionClipEntry(
        name=f"loco:{pkl_path.name}",
        source=pkl_path,
        motion_key=motion_key,
        start_frame=start_frame,
        n_frames=n_frames,
        hold_after=False,
        kind="locomotion",
    )
    payload: dict[str, Any] = {
        "action": "play",
        "pkl": str(pkl_path),
        "kind": "locomotion",
    }
    if motion_key is not None:
        payload["motion_key"] = motion_key
    if start_frame:
        payload["start_frame"] = start_frame
    if n_frames is not None:
        payload["n_frames"] = n_frames
    return entry, payload


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.release:
        if args.pkl is not None:
            raise SystemExit(
                "play_locomotion: --release is exclusive with --pkl "
                "(it sends a single 'stop' command and exits)."
            )
        pub = _connect_pub(args)
        time.sleep(max(0.0, args.linger_ms / 1000.0))
        if not args.quiet:
            print("[play_locomotion] RELEASE (sending stop)", flush=True)
        _send(pub, args.topic, {"action": "stop"})
        time.sleep(max(0.0, args.linger_ms / 1000.0))
        pub.close(linger=max(0, args.linger_ms))
        return 0

    if args.pkl is None:
        raise SystemExit(
            "play_locomotion: --pkl <path> is required for play "
            "(or use --release to drop the active clip)."
        )

    entry, payload = _build_entry(args)

    # Pre-flight: load PKL to estimate duration. This also catches a
    # missing PKL / bad motion_key / malformed clip BEFORE we send
    # anything on the wire -- much friendlier failure than the
    # recorder logging the error after the operator is already
    # blocked waiting.
    try:
        duration_s = estimate_duration_s(entry, float(args.target_rate_hz))
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"play_locomotion: cannot load clip: {exc}")

    pub = _connect_pub(args)
    # Slow-joiner mitigation: PUB.connect doesn't synchronise with
    # the SUB; the first message after connect is reliably dropped
    # without this sleep.
    time.sleep(max(0.0, args.linger_ms / 1000.0))

    if not args.quiet:
        target = args.pkl if args.motion_key is None else (
            f"{args.pkl}[{args.motion_key}]"
        )
        print(
            f"[play_locomotion] PLAY {target} "
            f"(~{duration_s:.1f}s; authored yaw preserved; "
            f"Ctrl-C to abort)",
            flush=True,
        )

    # Pre-play countdown gives the operator time to step clear before
    # motion starts. Ctrl-C here is "I changed my mind" -- we never
    # sent anything on the wire, so we leave whatever the recorder
    # was doing alone.
    if args.delay > 0:
        try:
            _countdown(float(args.delay), quiet=args.quiet)
        except KeyboardInterrupt:
            if not args.quiet:
                print(
                    "\n[play_locomotion] countdown aborted -- "
                    "nothing sent to recorder",
                    flush=True,
                )
            pub.close(linger=max(0, args.linger_ms))
            return 130

    _send(pub, args.topic, payload)

    # Block for clip duration, interruptible by SIGINT. We catch in
    # main() rather than installing a signal handler so the
    # KeyboardInterrupt path is the single shutdown branch.
    exit_code = 0
    try:
        # +0.1 s buffer: estimate_duration_s rounds down via the
        # resampler's floor() semantics, and we want the recorder to
        # naturally finalise the clip before we tear down.
        time.sleep(duration_s + 0.1)
        if not args.quiet:
            print("[play_locomotion] clip complete; exiting", flush=True)
    except KeyboardInterrupt:
        if not args.quiet:
            print(
                "\n[play_locomotion] SIGINT received -- sending STOP",
                flush=True,
            )
        _send(pub, args.topic, {"action": "stop"})
        # Give linger window for the stop message to reach the recorder.
        time.sleep(max(0.0, args.linger_ms / 1000.0))
        exit_code = 130

    pub.close(linger=max(0, args.linger_ms))
    return exit_code


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
