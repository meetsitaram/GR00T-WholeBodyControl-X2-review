#!/usr/bin/env python3
"""Trigger an in-place gesture mid-session on a running X2 stack.

The recorder process owns the ``motion_clip_cmd`` SUB (bind on
:data:`gear_sonic.utils.teleop.motion_clip_session.MOTION_CLIP_CMD_DEFAULT_PORT`).
This script PUB-connects, sends a JSON ``play`` payload tagged with
``kind="gesture"``, and BLOCKS for the estimated clip duration so
the operator gets a single foreground command they can ``Ctrl-C``
out of:

    play_gesture sit_stand_sit_A538       # block ~24 s, exit
    play_gesture --pkl /path/to/clip.pkl  # ad-hoc; bypass catalog
    play_gesture --list                   # print catalog, no traffic
    play_gesture --release                # send stop, return immediately

The gesture branch of :class:`MotionClipSession` yaw-rebases the
PKL's frame-0 yaw onto the robot's current heading so an arm-wave or
sit-down starts from wherever the robot is facing. For walks /
turns / sidesteps where the authored heading evolution matters, use
:file:`gear_sonic/scripts/play_locomotion.py` instead.

On natural completion the recorder ends the gesture on its own (its
:class:`MotionClipSession.is_done` flips after the last frame), so
the script just exits 0. On SIGINT during the block, the script
publishes ``{"action": "stop"}`` before exiting with code 130
(Ctrl-C convention) so the recorder snaps back to kplanner
forwarding immediately.

Hold-after semantics
--------------------

When the catalog entry's ``hold_after: true`` is set (or the operator
passes ``--hold``), the recorder *holds* the final clip frame
indefinitely after the clip ends. The script still exits as soon as
the clip duration elapses (it has no way to know the recorder is
holding); the operator releases the hold later with
``play_gesture --release`` or by triggering another gesture that
takes over. Pair this with the ``sit_down_A540`` / ``stand_up_A540``
catalog entries: the first one holds the seated pose, the second
returns control to kplanner.

See :file:`clip_motion_commands.md` for end-to-end MuJoCo recipe.
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
    GESTURE_DEFAULT_CATALOG_PATH,
    MOTION_CLIP_CMD_DEFAULT_PORT,
    MOTION_CLIP_CMD_DEFAULT_TOPIC,
    MotionClipEntry,
    estimate_duration_s,
    load_catalog,
)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "name",
        nargs="?",
        default=None,
        help="Catalog gesture name to play. Mutually exclusive with "
             "--list and --pkl.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="Print the loaded catalog and exit (no ZMQ traffic).",
    )
    parser.add_argument(
        "--release", action="store_true",
        help="Publish a single 'stop' command and exit immediately. "
             "Use this to release a held-final-frame pose (set by a "
             "catalog entry with hold_after: true) without playing "
             "another clip. Equivalent to Ctrl-C'ing a still-blocked "
             "play_gesture invocation.",
    )
    parser.add_argument(
        "--pkl", type=Path, default=None,
        help="Ad-hoc PKL path. Bypasses the catalog lookup; the "
             "recorder builds a one-off gesture MotionClipEntry from "
             "this path (motion-key defaults to first key in the PKL).",
    )
    hold_group = parser.add_mutually_exclusive_group()
    hold_group.add_argument(
        "--hold", dest="hold_after", action="store_const", const=True,
        default=None,
        help="Override the catalog's hold_after flag to TRUE for this "
             "invocation. The recorder will republish the gesture's "
             "last frame indefinitely after the clip ends; release "
             "with 'play_gesture --release' or another play.",
    )
    hold_group.add_argument(
        "--no-hold", dest="hold_after", action="store_const", const=False,
        help="Override the catalog's hold_after flag to FALSE for this "
             "invocation. Forces the auto-handback behaviour even on "
             "entries the catalog marks as hold_after: true.",
    )
    parser.add_argument(
        "--motion-key", type=str, default=None,
        help="Optional motion key inside a multi-clip PKL (used with "
             "--pkl, or to override the catalog entry's motion_key).",
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
        "--catalog", type=Path, default=GESTURE_DEFAULT_CATALOG_PATH,
        help="Path to the gesture catalog YAML (default ships with the "
             "repo).",
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
        help="Publish rate the recorder is running at; used to estimate "
             "clip duration so the script knows how long to block.",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="Seconds to count down (client-side) BEFORE publishing the "
             "play command -- gives the operator time to walk over to "
             "the robot and spot any instability before motion starts. "
             "Ctrl-C during the countdown exits cleanly without "
             "sending anything; the recorder's current state (held "
             "pose, kplanner forwarding, etc.) is left untouched.",
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
        print(f"[play_gesture] PUB connect {url} topic={args.topic!r}", flush=True)
    return pub


def _send(pub: "zmq.Socket[Any]", topic: str, payload: dict[str, Any]) -> None:
    pub.send_multipart([
        topic.encode("ascii"),
        json.dumps(payload).encode("utf-8"),
    ])


def _countdown(delay_s: float, *, quiet: bool) -> None:
    """Sleep ``delay_s`` seconds, printing a once-per-second countdown.

    Raises :class:`KeyboardInterrupt` on SIGINT (caller decides whether
    to translate that into a wire stop). No-op when ``delay_s <= 0``.
    """
    if delay_s <= 0:
        return
    if not quiet:
        print(
            f"[play_gesture] countdown {delay_s:.0f}s before play "
            f"(Ctrl-C to abort, recorder state untouched)",
            flush=True,
        )
    # Tick at 1 Hz; the final partial second (if any) is one sleep at
    # the end so the user sees the countdown actually hit zero.
    full_secs = int(delay_s)
    for remaining in range(full_secs, 0, -1):
        if not quiet:
            print(f"[play_gesture] T-{remaining}", flush=True)
        time.sleep(1.0)
    leftover = delay_s - float(full_secs)
    if leftover > 0:
        time.sleep(leftover)


def _print_catalog(catalog: dict[str, MotionClipEntry]) -> None:
    if not catalog:
        print("(catalog is empty)")
        return
    name_w = max(len(n) for n in catalog) + 2
    print(f"{'name'.ljust(name_w)}{'hold':<6}source")
    print("-" * (name_w + 66))
    for entry in catalog.values():
        src = entry.resolved_source()
        marker = "" if src.is_file() else "  [MISSING]"
        hold_tag = "yes" if entry.hold_after else "no"
        print(f"{entry.name.ljust(name_w)}{hold_tag:<6}{src}{marker}")


def _resolve_entry(args: argparse.Namespace) -> tuple[MotionClipEntry, dict[str, Any]]:
    """Resolve CLI args to (catalog_entry, wire_payload) for play.

    Returns the wire payload separately because the recorder side
    only sees ``{name, ...}`` or ``{pkl, ...}`` and resolves locally.
    The local ``entry`` retains the effective ``hold_after`` so the
    operator log can describe what will happen even though the
    recorder is the source of truth on the wire. Every payload is
    stamped with ``kind="gesture"`` -- locomotion clips go through
    :file:`gear_sonic/scripts/play_locomotion.py` which stamps
    ``kind="locomotion"`` instead.
    """
    motion_key = args.motion_key
    start_frame = int(args.start_frame)
    n_frames = None if args.n_frames is None else int(args.n_frames)
    hold_override: Optional[bool] = args.hold_after  # None / True / False
    if args.pkl is not None:
        # Ad-hoc PKL has no catalog row; default hold_after to False
        # unless the operator passed --hold explicitly.
        effective_hold = bool(hold_override) if hold_override is not None else False
        entry = MotionClipEntry(
            name=f"adhoc:{args.pkl.name}",
            source=args.pkl,
            motion_key=motion_key,
            start_frame=start_frame,
            n_frames=n_frames,
            hold_after=effective_hold,
            kind="gesture",
        )
        payload: dict[str, Any] = {
            "action": "play",
            "pkl": str(args.pkl),
            "kind": "gesture",
        }
    else:
        catalog = load_catalog(args.catalog)
        if args.name not in catalog:
            avail = ", ".join(list(catalog)[:5])
            raise SystemExit(
                f"play_gesture: unknown gesture name {args.name!r}. "
                f"Have {len(catalog)} entries (first few: {avail}). "
                f"Run with --list to see the full catalog."
            )
        base = catalog[args.name]
        effective_hold = (
            bool(hold_override) if hold_override is not None else bool(base.hold_after)
        )
        entry = MotionClipEntry(
            name=base.name,
            source=base.source,
            motion_key=motion_key if motion_key is not None else base.motion_key,
            start_frame=start_frame or base.start_frame,
            n_frames=n_frames if n_frames is not None else base.n_frames,
            hold_after=effective_hold,
            kind="gesture",
        )
        payload = {"action": "play", "name": args.name, "kind": "gesture"}
    if motion_key is not None:
        payload["motion_key"] = motion_key
    if start_frame:
        payload["start_frame"] = start_frame
    if n_frames is not None:
        payload["n_frames"] = n_frames
    # Only include hold_after on the wire when the operator explicitly
    # overrode it; that keeps named-gesture payloads minimal and lets
    # the recorder's catalog stay the source of truth in the common case.
    if hold_override is not None:
        payload["hold_after"] = bool(hold_override)
    return entry, payload


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    if args.list:
        catalog = load_catalog(args.catalog)
        _print_catalog(catalog)
        return 0

    if args.release:
        if args.name is not None or args.pkl is not None:
            raise SystemExit(
                "play_gesture: --release is exclusive with <name>/--pkl "
                "(it sends a single 'stop' command and exits)."
            )
        pub = _connect_pub(args)
        time.sleep(max(0.0, args.linger_ms / 1000.0))
        if not args.quiet:
            print("[play_gesture] RELEASE (sending stop)", flush=True)
        _send(pub, args.topic, {"action": "stop"})
        # Give the SUB time to actually receive the stop before close.
        time.sleep(max(0.0, args.linger_ms / 1000.0))
        pub.close(linger=max(0, args.linger_ms))
        return 0

    if (args.name is None) == (args.pkl is None):
        raise SystemExit(
            "play_gesture: pass exactly one of <name> or --pkl <path> "
            "(or use --list to see the catalog, --release to drop a held pose)."
        )

    entry, payload = _resolve_entry(args)

    # Pre-flight: load PKL to estimate duration. This also catches a
    # missing PKL / bad motion_key / malformed clip BEFORE we send
    # anything on the wire -- much friendlier failure than the
    # recorder logging the error after the operator is already
    # blocked waiting.
    try:
        duration_s = estimate_duration_s(entry, float(args.target_rate_hz))
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"play_gesture: cannot load gesture: {exc}")

    pub = _connect_pub(args)
    # Slow-joiner mitigation: PUB.connect doesn't synchronise with
    # the SUB; the first message after connect is reliably dropped
    # without this sleep.
    time.sleep(max(0.0, args.linger_ms / 1000.0))

    if not args.quiet:
        target = entry.name if args.pkl is None else str(args.pkl)
        hold_tag = (
            " [HOLD on completion -- release with --release]"
            if entry.hold_after else ""
        )
        print(
            f"[play_gesture] PLAY {target} "
            f"(~{duration_s:.1f}s; Ctrl-C to abort){hold_tag}",
            flush=True,
        )

    # Pre-play countdown gives the operator time to walk over to the
    # robot before motion starts. Ctrl-C here is "I changed my mind"
    # -- we never sent anything on the wire, so we leave whatever the
    # recorder was doing (idle / holding a prior gesture) alone.
    if args.delay > 0:
        try:
            _countdown(float(args.delay), quiet=args.quiet)
        except KeyboardInterrupt:
            if not args.quiet:
                print(
                    "\n[play_gesture] countdown aborted -- "
                    "nothing sent to recorder",
                    flush=True,
                )
            pub.close(linger=max(0, args.linger_ms))
            return 130

    _send(pub, args.topic, payload)

    # Block for clip duration, interruptible by SIGINT. We catch in
    # main() rather than installing a signal handler so the KeyboardInterrupt
    # path is the single shutdown branch.
    exit_code = 0
    try:
        # +0.1s buffer: estimate_duration_s rounds down via the
        # resampler's floor() semantics, and we want the recorder to
        # naturally finalise the gesture before we tear down.
        time.sleep(duration_s + 0.1)
        if not args.quiet:
            print("[play_gesture] clip complete; exiting", flush=True)
    except KeyboardInterrupt:
        if not args.quiet:
            print(
                "\n[play_gesture] SIGINT received -- sending STOP",
                flush=True,
            )
        _send(pub, args.topic, {"action": "stop"})
        # Give linger window for the stop message to reach the recorder.
        time.sleep(max(0.0, args.linger_ms / 1000.0))
        exit_code = 130

    pub.close(linger=max(0, args.linger_ms))
    return exit_code


if __name__ == "__main__":
    # Restore default SIGINT so the KeyboardInterrupt path actually
    # triggers (some environments install a handler that swallows it).
    signal.signal(signal.SIGINT, signal.default_int_handler)
    raise SystemExit(main())
