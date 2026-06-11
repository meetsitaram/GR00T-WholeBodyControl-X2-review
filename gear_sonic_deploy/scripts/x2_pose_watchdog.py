#!/usr/bin/env python3
"""x2_pose_watchdog.py -- PC2-side single-input fallback watchdog.

Single-input pose forwarder + staged fallback ladder. Sits between
ONE upstream pose source (the laptop ``x2_pose_mux`` over wifi, or
a recorder publishing directly when no VLA run is active) and the
C++ deploy on PC2. The deploy SUBs to this watchdog on localhost
rather than to the laptop directly.

When upstream is fresh:
    Forward laptop pose frames byte-for-byte to downstream. Zero
    protocol logic, zero re-encode -- the deploy sees the exact wire
    bytes the laptop sent.

When upstream is silent for > --idle-stale-ms (default 100):
    Run the staged fallback ladder (LIVE -> HOLD -> BLEND -> IDLE_CLIP)
    keyed by ``--idle-mode``. The default is ``blend``:

      * HOLD (default 10 s): re-publish the LAST forwarded upstream
        frame BYTE-FOR-BYTE. The deploy sees zero kinematic surprise
        (identical bytes -> identical joint_pos -> jvel = 0) and keeps
        commanding the operator's last pose. Soaks up WiFi blips /
        laptop GC pauses / Cursor reloads of up to ``--hold-last-secs``
        with no observable effect on the robot.
      * BLEND (default 3 s): lerp joint_pos_mj from the cached upstream
        frame toward the baked idle clip. Smooth glide rather than a
        step, so the arms drift to default over seconds rather than
        slamming through their full ROM in 200 ms.
      * IDLE_CLIP: after the hold + blend window expires, publish
        baked idle_stand frames indefinitely (the legacy destination
        behaviour, reached gradually).

    ``--idle-mode hold-last`` skips BLEND / IDLE_CLIP entirely and
    holds the cached upstream frame forever (operator owns recovery).

    ``--idle-mode idle-stand`` reproduces pre-2026-06-08 behaviour:
    jump to IDLE_CLIP on the first stale tick. Regression escape only.

    The baked idle clip is the same one
    ``live_vla_publish_motion_token.py --no-policy`` publishes in idle
    mode (built by ``bake_idle_stand_x2m2.py``).

    YAW REBASE (default ON): the baked idle_stand clip is yaw-aligned
    to ``R_z(0)`` for every frame. Publishing those frames verbatim
    while the robot is at a different heading hands the SONIC policy
    a stale absolute-yaw reference, and the tokenizer's
    ``rel = inv(measured) * reference`` computation makes the policy
    actively twist the body back to world +X -- the "robot snaps to
    spawn heading the moment I kill the planner stack" symptom. To
    fix this, the watchdog SUBs to the C++ deploy's ``x2_debug`` PUB
    (default ``tcp://127.0.0.1:5557``, topic ``x2_debug``), extracts
    the live ``base_quat`` (IMU pelvis quat) on every tick, and
    pre-multiplies the baked clip's root quats by ``R_z(measured_yaw)``
    before publishing.

What this watchdog DOES NOT DO:
    No dual-source arbitration. No engagement ramp. No teleop-mode
    gate. No ``vla_control`` PUB. Those concerns moved to the
    laptop-side ``x2_pose_mux`` on the 2026-06-11 split. The watchdog
    has one SUB, one PUB, and one optional x2_debug SUB for yaw
    rebase. If you need manual takeover, run ``run_x2_vla_runtime.sh
    --enable-takeover`` on the laptop; the mux merges sources there
    and ships ONE wire to this watchdog.

Why this exists (still):
    The C++ deploy in --input-type=zmq mode requires a continuous
    50 Hz pose-ref stream or its starvation watchdog trips into
    SAFE_IDLE (which commands default_angles with 4x kd -- a hard PD
    step that whirs the motors when current pose != default_angles).
    If the LAPTOP PROCESS itself dies or wifi drops mid-run, the wire
    goes silent and SAFE_IDLE fires. This watchdog guarantees the wire
    never goes silent from the deploy's perspective by sourcing its
    own fallback frames when upstream stops flowing.

    Critically: the watchdog must NOT inject a step change in the
    commanded reference. The deploy's target LPF + max_target_dev
    clamps cannot absorb a multi-radian step in joint_pos_mj, so a
    WiFi hiccup with the operator's arms extended would swing them
    through their full ROM to default in ~200 ms -- known to slam
    tables. The staged HOLD -> BLEND -> IDLE ladder keeps the wire
    alive while only making per-frame commanded-reference moves the
    deploy can actually track.

Single-thread design: one zmq.Context, one SUB, one PUB. The 50 Hz
tick loop polls the SUB non-blockingly, drains the queue (forwards
the LATEST frame if any), and fills in with an idle_stand frame when
upstream has been silent past the stale threshold. PUB-SUB is
inherently lossy on slow consumers; we set RCVHWM=SNDHWM=100 to keep
buffer pressure bounded.

Dependencies: numpy, pyzmq, stdlib, gear_sonic.utils.pose_pipeline
(numpy + stdlib only; no scipy). pc2_bringup.sh rsyncs the
pose_pipeline modules alongside this script.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import zmq

# pc2_bringup.sh rsyncs gear_sonic/utils/pose_pipeline/ to
# ${PC2_WS}/src/gear_sonic/utils/pose_pipeline/ and adds ${PC2_WS}/src
# to PYTHONPATH. On developer laptops where the repo root is already
# on PYTHONPATH (e.g. inside .venv), this import resolves to the same
# files. Belt-and-braces: also add this script's directory and the
# inferred repo root to sys.path so running the watchdog from a
# checked-out tree works without env-var setup.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
for _candidate in (_REPO_ROOT,):
    _p = str(_candidate)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gear_sonic.utils.pose_pipeline.fallback import (  # noqa: E402
    IDLE_MODE_BLEND,
    IDLE_MODE_HOLD_LAST,
    IDLE_MODE_IDLE_STAND,
    IDLE_MODES,
    IdleStandReplay,
    STATE_BLEND,
    STATE_COLD_IDLE,
    STATE_GAP,
    STATE_HOLD,
    STATE_IDLE_CLIP,
    STATE_LIVE,
    build_idle_frame_msg,
    decide_fallback_state,
)
from gear_sonic.utils.pose_pipeline.wire import (  # noqa: E402
    DEFAULT_PUB_RATE_HZ,
    decode_pose_joint_pos_mj,
    decode_x2_debug_base_quat,
    load_x2m2,
    yaw_from_quat_wxyz,
)


# ---------------------------------------------------------------------------
# Migration helper: detect legacy CLI flags and emit a clean error
# pointing at the 2026-06-11 pose_mux_split milestone. The old
# x2_pose_proxy.py accepted --override-port and --vla-control-port
# directly; both moved to the laptop mux. Catching them here means
# anyone running an old recipe gets a one-shot informative error
# instead of a confusing argparse "unrecognised argument" message.
# ---------------------------------------------------------------------------
_LEGACY_TAKEOVER_FLAGS: tuple[str, ...] = (
    "--override-host",
    "--override-port",
    "--override-topic",
    "--override-stale-ms",
    "--override-frozen-ticks",
    "--override-frozen-l2-tol",
    "--override-engage-motion-ticks",
    "--engagement-max-wire-step",
    "--engagement-steady-wire-step",
    "--engagement-step-ramp-ticks",
    "--vla-control-bind-host",
    "--vla-control-port",
    "--vla-control-topic",
    "--teleop-mode-host",
    "--teleop-mode-port",
    "--teleop-mode-topic",
    "--teleop-mode-stale-ms",
)


def _check_legacy_takeover_flags(argv: list[str]) -> None:
    hits = [a for a in argv if a.startswith("--") and a in _LEGACY_TAKEOVER_FLAGS]
    if not hits:
        return
    print(
        "[pose_watchdog] ERROR: the following flags are no longer "
        "accepted on PC2:",
        file=sys.stderr,
    )
    for flag in hits:
        print(f"  {flag}", file=sys.stderr)
    print(
        "\n[pose_watchdog] Manual takeover moved to the laptop-side "
        "x2_pose_mux as of 2026-06-11.\n"
        "[pose_watchdog] Run 'run_x2_vla_runtime.sh --enable-takeover ...' "
        "on the laptop;\n"
        "[pose_watchdog] the mux merges VLA + operator override and "
        "ships one wire to this watchdog.\n"
        "[pose_watchdog] See docs/source/user_guide/milestones/"
        "2026-06-11_pose_mux_split.md",
        file=sys.stderr,
    )
    sys.exit(2)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    _check_legacy_takeover_flags(raw_argv)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--upstream-host",
        required=True,
        help="Laptop IP publishing the pose stream "
             "(e.g. 192.168.86.22). With --enable-takeover this is "
             "the laptop running the x2_pose_mux; without it, the "
             "laptop running the recorder directly.",
    )
    p.add_argument(
        "--upstream-port",
        type=int,
        default=5556,
        help="Laptop pose PUB port (default 5556).",
    )
    p.add_argument(
        "--upstream-topic",
        default="pose",
        help="ZMQ topic prefix on upstream (default 'pose').",
    )
    p.add_argument(
        "--downstream-host",
        default="*",
        help="Local bind iface for downstream PUB (default '*' = all).",
    )
    p.add_argument(
        "--downstream-port",
        type=int,
        default=5558,
        help="Local PUB port the C++ deploy SUBs to (default 5558). "
             "MUST match the deploy's --vla-zmq-port.",
    )
    p.add_argument(
        "--downstream-topic",
        default="pose",
        help="Topic prefix on downstream (must match deploy "
             "--vla-zmq-topic).",
    )
    p.add_argument(
        "--idle-x2m2",
        type=Path,
        required=True,
        help="Path to baked idle_stand.x2m2 binary on PC2.",
    )
    p.add_argument(
        "--idle-stale-ms",
        type=int,
        default=100,
        help="Switch to idle fallback after this many ms of upstream "
             "silence (default 100).",
    )
    p.add_argument(
        "--idle-mode",
        choices=list(IDLE_MODES),
        default=IDLE_MODE_BLEND,
        help="Behaviour when upstream is silent past --idle-stale-ms. "
             "'blend' (DEFAULT, safe): hold the last forwarded "
             "upstream frame for --hold-last-secs, then lerp toward "
             "the baked idle_stand clip over --blend-secs. 'hold-"
             "last': hold the last forwarded frame indefinitely "
             "(operator owns recovery). 'idle-stand': pre-2026-06-08 "
             "behaviour; switch to the baked idle clip on the first "
             "stale tick (causes arms to slam to default on any wifi "
             "hiccup -- regression escape only).",
    )
    p.add_argument(
        "--hold-last-secs",
        type=float,
        default=10.0,
        help="How long (s) to hold the last forwarded upstream frame "
             "before transitioning toward idle (default 10.0). Only "
             "applies when --idle-mode=blend.",
    )
    p.add_argument(
        "--blend-secs",
        type=float,
        default=3.0,
        help="Duration (s) of the lerp from cached-upstream to baked "
             "idle_stand at the end of the hold window (default 3.0). "
             "Only applies when --idle-mode=blend.",
    )
    p.add_argument(
        "--rate-hz",
        type=float,
        default=DEFAULT_PUB_RATE_HZ,
        help="Downstream publish rate when idle (default 50; matches "
             "deploy control loop). Upstream forwarding is event-"
             "driven and inherits whatever cadence the laptop "
             "publishes at.",
    )
    p.add_argument(
        "--status-every-s",
        type=float,
        default=5.0,
        help="Periodic status print interval (default 5s).",
    )
    p.add_argument(
        "--x2-debug-host",
        default="127.0.0.1",
        help="Host of the deploy's x2_debug PUB (default 127.0.0.1; "
             "the deploy is colocated on PC2 in onbot mode).",
    )
    p.add_argument(
        "--x2-debug-port",
        type=int,
        default=5557,
        help="Port of the deploy's x2_debug PUB (default 5557).",
    )
    p.add_argument(
        "--x2-debug-topic",
        default="x2_debug",
        help="Topic prefix for the x2_debug PUB (default 'x2_debug').",
    )
    p.add_argument(
        "--x2-debug-max-age-s",
        type=float,
        default=0.5,
        help="Max age (s) of the latest x2_debug frame before we treat "
             "it as stale and fall back to the last-known-good "
             "measured yaw (default 0.5s).",
    )
    p.add_argument(
        "--no-x2-debug-yaw-track",
        action="store_true",
        help="Disable x2_debug yaw tracking. Idle-fallback frames will "
             "publish the baked clip's R_z(0) root quat verbatim, "
             "which causes the deploy to twist the body back to world "
             "+X on every IDLE entry. Regression-test escape; never "
             "use in prod.",
    )

    args = p.parse_args(raw_argv)

    if args.hold_last_secs < 0.0:
        print(
            f"[pose_watchdog] ERROR: --hold-last-secs must be >= 0, "
            f"got {args.hold_last_secs}",
            file=sys.stderr,
        )
        return 1
    if args.blend_secs < 0.0:
        print(
            f"[pose_watchdog] ERROR: --blend-secs must be >= 0, got "
            f"{args.blend_secs}",
            file=sys.stderr,
        )
        return 1

    if not args.idle_x2m2.is_file():
        print(
            f"[pose_watchdog] ERROR: idle X2M2 not found: "
            f"{args.idle_x2m2}",
            file=sys.stderr,
        )
        return 1

    print(
        f"[pose_watchdog] loading idle X2M2 from {args.idle_x2m2}",
        flush=True,
    )
    dof, quat, fps = load_x2m2(args.idle_x2m2)
    print(
        f"[pose_watchdog] idle clip: {dof.shape[0]} frames @ {fps:g} "
        f"Hz ({dof.shape[0] / fps:.2f} s loop)",
        flush=True,
    )
    replay = IdleStandReplay(dof, quat)

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    upstream_url = f"tcp://{args.upstream_host}:{args.upstream_port}"
    sub.setsockopt(zmq.RCVHWM, 100)
    sub.connect(upstream_url)
    sub.setsockopt(zmq.SUBSCRIBE, args.upstream_topic.encode("utf-8"))

    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.SNDHWM, 100)
    bind_url = f"tcp://{args.downstream_host}:{args.downstream_port}"
    pub.bind(bind_url)

    # Optional x2_debug SUB for measured-yaw tracking on idle frames.
    yaw_sub: zmq.Socket | None = None
    yaw_track_enabled = (
        not args.no_x2_debug_yaw_track
    ) and int(args.x2_debug_port) > 0
    if yaw_track_enabled:
        yaw_sub = ctx.socket(zmq.SUB)
        yaw_sub.setsockopt(zmq.RCVHWM, 4)
        x2_debug_url = f"tcp://{args.x2_debug_host}:{args.x2_debug_port}"
        yaw_sub.connect(x2_debug_url)
        yaw_sub.setsockopt(
            zmq.SUBSCRIBE, args.x2_debug_topic.encode("utf-8")
        )

    print(
        f"[pose_watchdog] upstream SUB:   {upstream_url} "
        f"topic={args.upstream_topic!r}",
        flush=True,
    )
    print(
        f"[pose_watchdog] downstream PUB: {bind_url} "
        f"topic={args.downstream_topic!r}",
        flush=True,
    )
    if yaw_track_enabled:
        print(
            f"[pose_watchdog] yaw-track SUB:  "
            f"tcp://{args.x2_debug_host}:{args.x2_debug_port} "
            f"topic={args.x2_debug_topic!r} "
            f"(max_age={args.x2_debug_max_age_s:.3f}s)",
            flush=True,
        )
    else:
        reason = (
            "--no-x2-debug-yaw-track"
            if args.no_x2_debug_yaw_track
            else f"--x2-debug-port={args.x2_debug_port}"
        )
        print(
            f"[pose_watchdog] yaw-track SUB:  DISABLED ({reason}); "
            f"idle frames will publish baked R_z(0) -- expect snap-"
            f"back to world +X on planner-stack termination",
            flush=True,
        )
    print(
        f"[pose_watchdog] idle stale threshold: {args.idle_stale_ms} "
        f"ms (switch to idle fallback after {args.idle_stale_ms} ms "
        f"of upstream silence)",
        flush=True,
    )
    if args.idle_mode == IDLE_MODE_BLEND:
        print(
            f"[pose_watchdog] idle mode: blend "
            f"(HOLD last frame for {args.hold_last_secs:.1f}s, then "
            f"BLEND to idle_stand over {args.blend_secs:.1f}s)",
            flush=True,
        )
    elif args.idle_mode == IDLE_MODE_HOLD_LAST:
        print(
            "[pose_watchdog] idle mode: hold-last (republish last "
            "upstream frame indefinitely; operator owns recovery)",
            flush=True,
        )
    else:  # idle-stand
        print(
            "[pose_watchdog] idle mode: idle-stand (LEGACY; arms snap "
            "to default on the first stale tick -- known to slam "
            "tables during WiFi hiccups)",
            flush=True,
        )

    period = 1.0 / max(args.rate_hz, 1e-6)
    stale_s = args.idle_stale_ms / 1000.0
    hold_last_secs = float(args.hold_last_secs)
    blend_secs = float(args.blend_secs)
    idle_mode = str(args.idle_mode)
    yaw_max_age_s = float(args.x2_debug_max_age_s)
    next_tick = time.monotonic()
    last_upstream_s = -1.0
    last_measured_yaw_rad = 0.0
    last_measured_yaw_s = -1.0
    yaw_decode_failures = 0
    last_upstream_msg: bytes | None = None
    last_upstream_jpos: np.ndarray | None = None
    cur_state = STATE_COLD_IDLE
    prev_state = STATE_COLD_IDLE
    tick = 0
    idle_tick = 0
    fwd_frames = 0
    idle_frames = 0
    idle_frames_with_rebase = 0
    hold_frames = 0
    blend_frames = 0
    gap_skips = 0
    last_status_s = time.monotonic()

    print(
        "[pose_watchdog] starting (initial state: COLD_IDLE; will "
        "switch to LIVE as soon as upstream publishes anything)",
        flush=True,
    )

    try:
        while True:
            now = time.monotonic()

            # Drain x2_debug queue first so the measured-yaw cache is
            # fresh by the time we decide what to publish on this tick.
            if yaw_sub is not None:
                latest_debug = None
                while True:
                    try:
                        latest_debug = yaw_sub.recv(zmq.NOBLOCK)
                    except zmq.Again:
                        break
                if latest_debug is not None:
                    base_quat_wxyz = decode_x2_debug_base_quat(
                        latest_debug, args.x2_debug_topic
                    )
                    if base_quat_wxyz is not None:
                        try:
                            last_measured_yaw_rad = yaw_from_quat_wxyz(
                                base_quat_wxyz
                            )
                            last_measured_yaw_s = now
                        except (ValueError, TypeError):
                            yaw_decode_failures += 1
                    else:
                        yaw_decode_failures += 1

            # Drain upstream queue. We forward the latest frame each
            # tick, not every frame -- if the laptop publishes faster
            # than we tick, intermediate frames are intentionally
            # dropped (the deploy only ever sees the freshest
            # reference anyway).
            latest = None
            while True:
                try:
                    latest = sub.recv(zmq.NOBLOCK)
                except zmq.Again:
                    break

            if latest is not None:
                last_upstream_msg = latest
                jpos = decode_pose_joint_pos_mj(
                    latest, args.upstream_topic
                )
                if jpos is None:
                    jpos = decode_pose_joint_pos_mj(
                        latest, args.downstream_topic
                    )
                if jpos is not None:
                    last_upstream_jpos = jpos
                last_upstream_s = now

            if latest is not None:
                try:
                    pub.send(latest, zmq.NOBLOCK)
                    fwd_frames += 1
                except zmq.Again:
                    pass
                cur_state = STATE_LIVE
            else:
                # No upstream this tick. Decide what to fill in with
                # via the staged fallback ladder.
                age = (
                    float("inf") if last_upstream_s < 0
                    else (now - last_upstream_s)
                )
                target_state, blend_alpha = decide_fallback_state(
                    have_upstream=(last_upstream_msg is not None),
                    age_s=age,
                    stale_s=stale_s,
                    hold_last_secs=hold_last_secs,
                    blend_secs=blend_secs,
                    idle_mode=idle_mode,
                )
                cur_state = target_state

                if target_state == STATE_GAP:
                    gap_skips += 1
                elif target_state == STATE_HOLD:
                    if last_upstream_msg is None:
                        cur_state = STATE_COLD_IDLE
                    else:
                        try:
                            pub.send(last_upstream_msg, zmq.NOBLOCK)
                            hold_frames += 1
                        except zmq.Again:
                            pass
                elif target_state == STATE_BLEND:
                    yaw_rebase: float | None = None
                    if (
                        yaw_track_enabled
                        and last_measured_yaw_s >= 0
                        and (now - last_measured_yaw_s) <= yaw_max_age_s
                    ):
                        yaw_rebase = last_measured_yaw_rad
                    if last_upstream_jpos is None:
                        msg = build_idle_frame_msg(
                            replay,
                            idle_tick,
                            args.downstream_topic,
                            yaw_rebase_rad=yaw_rebase,
                        )
                    else:
                        idle_jpos, _ = replay.current(idle_tick)
                        lerp = (
                            (1.0 - blend_alpha) * last_upstream_jpos
                            + blend_alpha * idle_jpos
                        ).astype(np.float32)
                        msg = build_idle_frame_msg(
                            replay,
                            idle_tick,
                            args.downstream_topic,
                            yaw_rebase_rad=yaw_rebase,
                            joint_pos_mj_override=lerp,
                        )
                    try:
                        pub.send(msg, zmq.NOBLOCK)
                        blend_frames += 1
                    except zmq.Again:
                        pass
                    idle_tick += 1
                elif target_state in (STATE_COLD_IDLE, STATE_IDLE_CLIP):
                    yaw_rebase = None
                    if (
                        yaw_track_enabled
                        and last_measured_yaw_s >= 0
                        and (now - last_measured_yaw_s) <= yaw_max_age_s
                    ):
                        yaw_rebase = last_measured_yaw_rad
                    msg = build_idle_frame_msg(
                        replay,
                        idle_tick,
                        args.downstream_topic,
                        yaw_rebase_rad=yaw_rebase,
                    )
                    try:
                        pub.send(msg, zmq.NOBLOCK)
                        idle_frames += 1
                        if yaw_rebase is not None:
                            idle_frames_with_rebase += 1
                    except zmq.Again:
                        pass
                    idle_tick += 1

            # Emit a one-line transition log every time the state name
            # changes. Operators rely on this to correlate "I saw the
            # robot arm freeze for a few seconds then drift to stand"
            # with the watchdog's view of upstream availability.
            if cur_state != prev_state:
                if cur_state == STATE_LIVE:
                    if last_upstream_s < 0 or prev_state == STATE_COLD_IDLE:
                        msg_txt = (
                            f"{prev_state} -> LIVE (first upstream "
                            f"frame received)"
                        )
                    else:
                        gap_ms = (now - last_upstream_s) * 1000.0
                        msg_txt = (
                            f"{prev_state} -> LIVE (upstream pose "
                            f"frames flowing again after {gap_ms:.0f} "
                            f"ms gap)"
                        )
                elif cur_state == STATE_GAP:
                    msg_txt = (
                        f"{prev_state} -> GAP (upstream silent < "
                        f"{args.idle_stale_ms} ms; holding deploy "
                        f"cache)"
                    )
                elif cur_state == STATE_HOLD:
                    msg_txt = (
                        f"{prev_state} -> HOLD (re-publishing last "
                        f"upstream frame; will hold for "
                        f"{hold_last_secs:.1f}s)"
                    )
                elif cur_state == STATE_BLEND:
                    msg_txt = (
                        f"{prev_state} -> BLEND (lerping cached -> "
                        f"idle_stand over {blend_secs:.1f}s)"
                    )
                elif cur_state == STATE_IDLE_CLIP:
                    msg_txt = (
                        f"{prev_state} -> IDLE_CLIP (upstream silent "
                        f"past hold + blend window; tracking baked "
                        f"idle clip)"
                    )
                elif cur_state == STATE_COLD_IDLE:
                    msg_txt = f"{prev_state} -> COLD_IDLE"
                else:
                    msg_txt = f"{prev_state} -> {cur_state}"
                print(f"[pose_watchdog] state: {msg_txt}", flush=True)
                if (
                    cur_state in (
                        STATE_IDLE_CLIP, STATE_COLD_IDLE, STATE_BLEND
                    )
                    and prev_state not in (
                        STATE_IDLE_CLIP, STATE_COLD_IDLE, STATE_BLEND
                    )
                ):
                    idle_tick = 0
                prev_state = cur_state

            tick += 1

            if now - last_status_s >= args.status_every_s:
                age = (
                    float("inf") if last_upstream_s < 0
                    else (now - last_upstream_s)
                )
                age_str = (
                    "never" if last_upstream_s < 0
                    else f"{age * 1000:.0f}ms"
                )
                if cur_state == STATE_HOLD and last_upstream_s >= 0:
                    fb_age = max(0.0, age - stale_s)
                    state_str = (
                        f"HOLD t={fb_age:.1f}/{hold_last_secs:.1f}s"
                    )
                elif cur_state == STATE_BLEND and last_upstream_s >= 0:
                    fb_age = max(0.0, age - stale_s)
                    if blend_secs > 0.0:
                        alpha = (fb_age - hold_last_secs) / blend_secs
                        alpha = max(0.0, min(1.0, alpha))
                    else:
                        alpha = 1.0
                    state_str = f"BLEND alpha={alpha:.2f}"
                else:
                    state_str = cur_state
                if not yaw_track_enabled:
                    yaw_str = "off"
                elif last_measured_yaw_s < 0:
                    yaw_str = "never"
                else:
                    yaw_age_ms = (now - last_measured_yaw_s) * 1000.0
                    yaw_str = (
                        f"yaw={math.degrees(last_measured_yaw_rad):+.1f}deg "
                        f"age={yaw_age_ms:.0f}ms"
                    )
                print(
                    f"[pose_watchdog] tick={tick} state={state_str} "
                    f"mode={idle_mode} upstream_age={age_str} "
                    f"fwd={fwd_frames} hold={hold_frames} "
                    f"blend={blend_frames} idle={idle_frames} "
                    f"idle_rebased={idle_frames_with_rebase} "
                    f"gap_skip={gap_skips} "
                    f"x2_debug=({yaw_str}) "
                    f"yaw_decode_fail={yaw_decode_failures}",
                    flush=True,
                )
                last_status_s = now

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print(
            "[pose_watchdog] SIGINT received; tearing down",
            flush=True,
        )
    finally:
        sub.close(linger=0)
        pub.close(linger=0)
        if yaw_sub is not None:
            yaw_sub.close(linger=0)
        ctx.term()

    print(
        f"[pose_watchdog] done. total_ticks={tick} fwd={fwd_frames} "
        f"hold={hold_frames} blend={blend_frames} idle={idle_frames} "
        f"gap_skip={gap_skips}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
