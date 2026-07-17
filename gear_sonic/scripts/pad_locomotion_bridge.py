#!/usr/bin/env python3
"""Gamepad -> kplanner continuous-locomotion bridge (PS/Xbox pad).

Runs ALONGSIDE ``play_xbox_controller.py`` (gestures/dances/e-stop): this
script owns ONLY the analog sticks -> ``planner_cmd`` continuous locomotion,
mirroring the Quest3 manager's payloads so the kplanner needs zero changes.

Controls (consistent with play_xbox_controller's conventions):
  L2+R2 BOTH held   = deadman: sticks are live
    left stick      = forward/back + lateral
    right stick X   = yaw
    L1 (tap)        = speed setpoint -0.1   (kplanner-side, 0.2..1.0)
    R1 (tap)        = speed setpoint +0.1
  triggers released = one zero-stick command (robot idles), sticks dead
  L1+R1+L2+R2      = e-stop chord (owned by play_xbox_controller; we also
                      emit a zero-cmd for good measure)

Wire: PUB-connect to the stack's planner_cmd SUB (default 127.0.0.1:5563),
multipart [b"planner_cmd", json] -- same frames as quest3_manager_x2.py.

    python gear_sonic/scripts/pad_locomotion_bridge.py            # defaults
    python gear_sonic/scripts/pad_locomotion_bridge.py --probe    # mapping check
"""
from __future__ import annotations

import argparse
import json
import time

import pygame
import zmq

# Pad mapping (Xbox-style SDL defaults; PS DualSense/DS4 map the same core
# indices under SDL2 for sticks/shoulders; verify with --probe if unsure).
AXIS_LX, AXIS_LY = 0, 1
AXIS_RX = 3          # right stick X (some pads: 2 -- use --probe / --axis-rx)
AXIS_LT, AXIS_RT = 2, 5
BTN_LB, BTN_RB = 4, 5   # L1 / R1

DEADZONE = 0.15
RATE_HZ = 50.0
HEARTBEAT_S = 0.2        # re-publish while driving even if sticks unchanged
CHANGE_EPS = 0.02


def _dz(v: float) -> float:
    return 0.0 if abs(v) < DEADZONE else max(-1.0, min(1.0, v))


class _ZmqPad:
    """Duck-typed pygame.Joystick reading pc2_pad_daemon's pad_state feed.

    SAFETY: if no pad_state arrives for ``stale_s`` the state zeroes out --
    triggers read released -> the bridge's deadman drops -> robot idles.
    """

    def __init__(self, host: str, port: int, stale_s: float = 0.5) -> None:
        ctx = zmq.Context.instance()
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"pad_state")
        self._sub.setsockopt(zmq.RCVTIMEO, 0)   # non-blocking drain
        self._sub.setsockopt(zmq.CONFLATE, 0)
        self._sub.connect(f"tcp://{host}:{port}")
        self._axes = [0.0] * 8
        # triggers rest at -1 in SDL convention
        self._axes[AXIS_LT] = -1.0
        self._axes[AXIS_RT] = -1.0
        self._buttons = [0] * 16
        self._last_rx = 0.0
        self._stale_s = stale_s

    def pump(self) -> None:
        while True:
            try:
                _, payload = self._sub.recv_multipart()
            except zmq.Again:
                break
            d = json.loads(payload)
            ax = d.get("axes", [])
            bt = d.get("buttons", [])
            self._axes[:len(ax)] = ax
            self._buttons[:len(bt)] = bt
            self._last_rx = time.monotonic()
        if self._last_rx and time.monotonic() - self._last_rx > self._stale_s:
            # feed lost -> fail safe to released
            self._axes = [0.0] * len(self._axes)
            self._axes[AXIS_LT] = -1.0
            self._axes[AXIS_RT] = -1.0
            self._buttons = [0] * len(self._buttons)

    def get_axis(self, i: int) -> float:
        return self._axes[i] if i < len(self._axes) else 0.0

    def get_button(self, i: int) -> int:
        return self._buttons[i] if i < len(self._buttons) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5563)
    ap.add_argument("--topic", default="planner_cmd")
    ap.add_argument("--axis-rx", type=int, default=AXIS_RX)
    ap.add_argument("--bind", action="store_true",
                    help="bind the PUB (pad-only mode, no quest3 manager); "
                         "default is connect (manager owns the bind)")
    ap.add_argument("--source", choices=("local", "zmq"), default="local",
                    help="'local' = pygame reads a directly-attached pad; "
                         "'zmq' = consume pad_state events from a "
                         "pc2_pad_daemon (pad lives on the robot)")
    ap.add_argument("--pad-host", default="192.168.86.32",
                    help="(--source zmq) host running pc2_pad_daemon")
    ap.add_argument("--pad-port", type=int, default=5569)
    ap.add_argument("--deadman", choices=("both", "left"), default="both",
                    help="'both' = L2+R2 held (two-handed); 'left' = L2 only "
                         "(one-handed drive, right hand free to spot the robot)")
    ap.add_argument("--lock-speed", action="store_true",
                    help="disable L1/R1 speed nudges: the kplanner setpoint "
                         "stays at its launch value (constant-speed demo mode)")
    ap.add_argument("--clip-pkl", default=None,
                    help="enable the dance chord: L1+Y (Triangle on PS) plays "
                         "this motion PKL via the recorder's motion_clip_cmd "
                         "wire; L1+B (Circle) stops it. Only fires while the "
                         "deadman is RELEASED (no dancing mid-drive).")
    ap.add_argument("--clip-key", default=None,
                    help="motion_key inside --clip-pkl (default: first key)")
    ap.add_argument("--clip-keys", default=None,
                    help="comma-separated motion_keys: L1+Y cycles FORWARD "
                         "through the list, L1+A cycles BACKWARD (each press "
                         "plays the next/prev clip). Overrides --clip-key.")
    ap.add_argument("--clip-host", default="127.0.0.1",
                    help="recorder host owning the motion_clip_cmd SUB bind")
    ap.add_argument("--clip-port", type=int, default=5568)
    ap.add_argument("--clip-topic", default="motion_clip_cmd")
    ap.add_argument("--log-buttons", action="store_true",
                    help="log every button-index rising edge (pad mapping aid)")
    ap.add_argument("--probe", action="store_true",
                    help="print axis/button events and exit on ctrl-c")
    ap.add_argument("--invert-ly", action="store_true", default=True,
                    help="stick up (-1 in SDL) = forward (default on)")
    args = ap.parse_args()

    if args.source == "zmq":
        js = _ZmqPad(args.pad_host, args.pad_port)
        print(f"[pad-bridge] pad source: pc2_pad_daemon at "
              f"tcp://{args.pad_host}:{args.pad_port}", flush=True)
    else:
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("[pad-bridge] no gamepad found"); return 1
        js = pygame.joystick.Joystick(0); js.init()
        print(f"[pad-bridge] pad: {js.get_name()} "
              f"(axes={js.get_numaxes()} buttons={js.get_numbuttons()})")

    if args.probe:
        prev = None
        while True:
            pygame.event.pump()
            state = ([round(js.get_axis(i), 1) for i in range(js.get_numaxes())],
                     [js.get_button(i) for i in range(js.get_numbuttons())])
            if state != prev:
                print(f"axes={state[0]} buttons={state[1]}", flush=True)
                prev = state
            time.sleep(0.05)

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 10)
    if args.bind:
        # Pad-only sessions: the quest3 manager (the usual :5563 binder) is
        # not running, so WE bind and the kplanner's connected SUB attaches.
        sock.bind(f"tcp://*:{args.port}")
        print(f"[pad-bridge] PUB bind tcp://*:{args.port} "
              f"topic={args.topic!r}; hold L2+R2 to drive", flush=True)
    else:
        sock.connect(f"tcp://{args.host}:{args.port}")
        print(f"[pad-bridge] PUB connect tcp://{args.host}:{args.port} "
              f"topic={args.topic!r}; hold L2+R2 to drive", flush=True)
    time.sleep(0.3)  # slow-joiner

    def send(payload: dict) -> None:
        sock.send_multipart([args.topic.encode("ascii"),
                             json.dumps(payload).encode("utf-8")])

    # Dance chord wire (optional): separate PUB to the recorder's
    # motion_clip_cmd SUB. Payload mirrors play_locomotion.py exactly.
    clip_sock = None
    btn_play, btn_stop, btn_prev = 3, 1, 0   # Xbox: Y / B / A
    clip_list: list[str] = []
    clip_cur = -1                            # first L1+Y plays clip_list[0]
    if args.clip_pkl:
        pad_name = js.get_name().lower() if hasattr(js, "get_name") else ""
        if any(k in pad_name for k in ("dualsense", "sony", "playstation")):
            btn_play, btn_stop, btn_prev = 2, 1, 0  # PS: Triangle/Circle/Cross
        if args.clip_keys:
            clip_list = [k for k in args.clip_keys.split(",") if k.strip()]
        elif args.clip_key:
            clip_list = [args.clip_key]
        clip_sock = ctx.socket(zmq.PUB)
        clip_sock.setsockopt(zmq.SNDHWM, 10)
        clip_sock.connect(f"tcp://{args.clip_host}:{args.clip_port}")
        print(f"[pad-bridge] dance chord armed: L1+Y next / L1+A prev of "
              f"{len(clip_list)} clip(s), L1+B stops; deadman must be "
              f"released. pkl={args.clip_pkl}", flush=True)

    def send_clip(payload: dict) -> None:
        clip_sock.send_multipart([args.clip_topic.encode("ascii"),
                                  json.dumps(payload).encode("utf-8")])

    def buzz(strength: float, ms: int) -> None:
        """Haptic cue on a locally-attached pad. No-op for the ZMQ-fed
        pad (pc2_pad_daemon owns rumble on the robot) and when rumble is
        unsupported by the transport."""
        rumble = getattr(js, "rumble", None)
        if rumble is None:
            return
        try:
            rumble(strength, strength, ms)
        except Exception:  # noqa: BLE001 -- haptics are best-effort
            pass

    tick = 1.0 / RATE_HZ
    last_sent: tuple | None = None
    last_pub_ts = 0.0
    was_live = False
    prev_lb = prev_rb = False
    prev_play = prev_stop = prev_prev = False
    clip_cooldown_until = 0.0
    pending_delta = 0.0

    _dbg_prev_btns: set[int] = set()
    while True:
        if isinstance(js, _ZmqPad):
            js.pump()
        else:
            pygame.event.pump()
        # Button-index debug: log rising edges so pad layouts can be
        # mapped from a live session (harmless, low-rate).
        if args.log_buttons and not isinstance(js, _ZmqPad):
            cur_btns = {b for b in range(js.get_numbuttons()) if js.get_button(b)}
            for b in sorted(cur_btns - _dbg_prev_btns):
                print(f"[pad-bridge] button {b} pressed", flush=True)
            _dbg_prev_btns = cur_btns
        lt = js.get_axis(AXIS_LT) > 0.0   # SDL triggers: -1 rest -> +1 pressed
        rt = js.get_axis(AXIS_RT) > 0.0
        lb, rb = bool(js.get_button(BTN_LB)), bool(js.get_button(BTN_RB))
        deadman_held = lt if args.deadman == "left" else (lt and rt)
        live = deadman_held and not (lb and rb)   # all-four = e-stop chord: go dead

        # Speed nudges: shoulder taps while deadman held (rising edges).
        # Suppressed entirely in --lock-speed mode (constant-speed demos:
        # stray shoulder presses while gripping the triggers are easy).
        if not args.lock_speed:
            if live and lb and not prev_lb:
                pending_delta -= 0.1
            if live and rb and not prev_rb:
                pending_delta += 0.1
        prev_lb, prev_rb = lb, rb

        # Dance chord: L1 + play-button, rising edge, deadman released,
        # 2s cooldown. Stop chord: L1 + stop-button. Both no-ops mid-drive
        # so a dance can never preempt active locomotion.
        if clip_sock is not None:
            play_b = bool(js.get_button(btn_play))
            stop_b = bool(js.get_button(btn_stop))
            prev_b = bool(js.get_button(btn_prev))
            now_c = time.monotonic()
            step = 0
            if (lb and play_b and not prev_play and not live
                    and now_c >= clip_cooldown_until):
                step = 1        # L1+Y: next dance
            elif (lb and prev_b and not prev_prev and not live
                    and now_c >= clip_cooldown_until):
                step = -1       # L1+A: previous dance
            if step and clip_list:
                clip_cur = (clip_cur + step) % len(clip_list)
                key = clip_list[clip_cur]
                send_clip({"action": "play", "pkl": args.clip_pkl,
                           "kind": "locomotion", "motion_key": key})
                clip_cooldown_until = now_c + 2.0
                buzz(0.7, 200)   # dance fired: single firm pulse
                print(f"[pad-bridge] DANCE {clip_cur + 1}/{len(clip_list)} "
                      f"-> {key}", flush=True)
            elif lb and stop_b and not prev_stop and not live:
                send_clip({"action": "stop"})
                clip_cooldown_until = now_c + 1.0
                buzz(0.3, 100)   # stop: weak blip
                print("[pad-bridge] dance STOP chord -> idle", flush=True)
            prev_play, prev_stop, prev_prev = play_b, stop_b, prev_b

        # Input-change logging: every deadman transition and (throttled)
        # stick change is printed so the operator can see what the pad
        # is sending without watching the planner log.
        if live and not was_live:
            print(f"[pad-bridge] DEADMAN ENGAGED (L2+R2) -> sticks live", flush=True)

        if live:
            fwd = _dz(-js.get_axis(AXIS_LY)) if args.invert_ly else _dz(js.get_axis(AXIS_LY))
            side = _dz(js.get_axis(AXIS_LX))
            yaw = _dz(js.get_axis(args.axis_rx))
            cur = (round(fwd, 3), round(side, 3), round(yaw, 3))
            now = time.monotonic()
            if (last_sent is None
                    or any(abs(a - b) > CHANGE_EPS for a, b in zip(cur, last_sent))
                    or pending_delta != 0.0
                    or now - last_pub_ts > HEARTBEAT_S):
                payload = {
                    "intent": "locomotion",
                    "magnitude": "continuous",
                    "stick_fwd": cur[0],
                    "stick_side": cur[1],
                    "stick_yaw": cur[2],
                }
                if pending_delta:
                    payload["speed_delta"] = round(pending_delta, 2)
                    print(f"[pad-bridge] speed_delta {pending_delta:+.1f}")
                    pending_delta = 0.0
                send(payload)
                if last_sent is None or any(
                    abs(a - b) > 0.1 for a, b in zip(cur, last_sent or (9, 9, 9))
                ):
                    print(f"[pad-bridge] sticks fwd={cur[0]:+.2f} "
                          f"side={cur[1]:+.2f} yaw={cur[2]:+.2f}", flush=True)
                last_sent, last_pub_ts = cur, now
            was_live = True
        elif was_live:
            # Deadman released (or e-stop chord): one zero-stick cmd -> idle.
            send({"intent": "locomotion", "magnitude": "continuous",
                  "stick_fwd": 0.0, "stick_side": 0.0, "stick_yaw": 0.0})
            print("[pad-bridge] deadman released -> idle")
            last_sent, was_live = None, False

        time.sleep(tick)


if __name__ == "__main__":
    raise SystemExit(main())
