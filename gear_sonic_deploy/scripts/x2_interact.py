#!/usr/bin/env python3
"""x2_interact.py -- speaker + face-display output for the X2, callable from PC2.

Import this from any task running on PC2 to put a line on the robot's face
display or play a pre-baked sound through its speaker. Both live on PC3 (the
RK3588S interaction unit); this module hides the two-hop plumbing.

    from x2_interact import X2Interact
    ui = X2Interact()

    ui.log("planner up")                  # append a row to the on-screen log
    ui.log("pose stream lost", "WARN")
    ui.say("step_back")                   # play a pre-baked wav on the speaker
    ui.banner("hello!", "how are you?")   # big centred message
    ui.music("bed.wav", loop=True)        # background music (mixes under speech)
    ui.stop_music()
    ui.restore_face()                     # hand the display back to the robot

DESIGN RULES (this sits next to a live demo, so it must never be the thing
that breaks it):

  * Nothing blocks. Every call returns immediately; work happens on a single
    background thread. A wedged network or a dead PC3 cannot stall your task.
  * Nothing raises. Failures are counted and logged, never propagated. Cosmetic
    output must not take down the caller.
  * Updates coalesce. Rapid log() calls collapse into one render, so a chatty
    loop can't build an unbounded backlog of ~1s renders.
  * Audio is detached. aplay runs under setsid on PC3 with its pid recorded, so
    playback survives the SSH session and stays independently stoppable.

WHY IT WORKS THIS WAY (verified on hardware 2026-07-17):
  * The vendor audiohal holds ALSA card 1 exclusively -- `default`/`hw:1,0` give
    "Device or resource busy". The dmix device `playback_def` is the shared path,
    and it MIXES with the vendor stack, so TTS/alerts cut through music.
  * flutter-pi binds 0.0.0.0:18080, so PC2 can drive the screen directly.
  * The face UI enforces priority: requests below the current priority are
    silently rejected. TTS raises it to 60, so panels post at 100.
  * Rendering runs on PC3 (see x2_pc3_render.py) so each update is ~200 bytes
    of JSON instead of a ~100KB mp4 over the wire.
"""
from __future__ import annotations

import base64
import collections
import json
import logging
import os
import queue
import shlex
import subprocess
import threading
import time

log = logging.getLogger("x2_interact")

PC3_HOST = os.environ.get("X2_PC3_HOST", "10.0.1.42")
PC3_USER = os.environ.get("X2_PC3_USER", "agi")
PC3_PASSWORD = os.environ.get("X2_PC3_PASSWORD", "1")
# These paths are on PC3, not PC2. PC3's /home/run is drwxr-x--- (owner-only),
# and we connect as `agi`, so the getsolo convention can't apply here -- /opt is
# the world-readable equivalent on the interaction unit.
REMOTE_ROOT = os.environ.get("X2_PC3_ROOT", "/opt/x2_interact")
AUDIO_DIR = f"{REMOTE_ROOT}/audio"
RENDERER = f"{REMOTE_ROOT}/x2_pc3_render.py"

SSH_OPTS = ["-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "LogLevel=ERROR", "-o", "BatchMode=no"]
ALSA_DEV = "playback_def"          # dmix -- shared with the vendor audiohal
SLOT_SPEECH, SLOT_MUSIC = "speech", "music"


class X2Interact:
    """Fire-and-forget speaker + display output. Safe to construct once and
    share; all public methods are non-blocking and never raise."""

    def __init__(self, host: str = PC3_HOST, user: str = PC3_USER,
                 password: str = PC3_PASSWORD, title: str = "X2  ::  STATUS",
                 enabled: bool = True, log_rows: int = 9):
        self.host, self.user, self.password = host, user, password
        self.title = title
        self.enabled = enabled
        self._rows: collections.deque = collections.deque(maxlen=log_rows)
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._lock = threading.Lock()
        self.failures = 0
        self._last_error: str | None = None
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="x2-interact")
        self._worker.start()

    # ---------------------------------------------------------------- public
    def log(self, message: str, level: str = "INFO") -> None:
        """Append a row to the on-screen log and refresh the display."""
        with self._lock:
            self._rows.append((level, message))
            rows = list(self._rows)
        self._submit(("panel", {"title": self.title, "lines": rows}))

    def banner(self, text: str, sub: str = "", accent: str = "#12245a") -> None:
        """Replace the screen with one large centred message."""
        self._submit(("panel", {"title": self.title, "banner": text, "sub": sub,
                                "accent": accent}))

    def panel(self, lines, title: str | None = None) -> None:
        """Show an explicit list of (level, message) rows, ignoring log history."""
        self._submit(("panel", {"title": title or self.title,
                                "lines": [list(x) for x in lines]}))

    def say(self, name: str) -> None:
        """Play a pre-baked wav from the PC3 audio dir. `name` may be a bare
        stem ('step_back'), a filename, or an absolute path on PC3."""
        self._submit(("audio", {"slot": SLOT_SPEECH, "path": self._resolve(name),
                                "loop": False}))

    def music(self, name: str, loop: bool = True) -> None:
        """Play background music. Mixes underneath say()/TTS via dmix."""
        self._submit(("audio", {"slot": SLOT_MUSIC, "path": self._resolve(name),
                                "loop": loop}))

    def stop_music(self) -> None:
        self._submit(("stop", {"slot": SLOT_MUSIC}))

    def stop_speech(self) -> None:
        self._submit(("stop", {"slot": SLOT_SPEECH}))

    def restore_face(self) -> None:
        """Hand the display back to the robot's own idle animation."""
        self._submit(("restore", {}))

    def clear(self) -> None:
        with self._lock:
            self._rows.clear()

    def flush(self, timeout: float = 10.0) -> bool:
        """Block until queued work drains. For shutdown paths and tests only --
        normal callers should never need this."""
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self._q.unfinished_tasks == 0:
                return True
            time.sleep(0.05)
        return False

    @property
    def healthy(self) -> bool:
        return self.failures == 0

    # --------------------------------------------------------------- internal
    def _resolve(self, name: str) -> str:
        if name.startswith("/"):
            return name
        if not name.endswith((".wav", ".mp3", ".ogg")):
            name += ".wav"
        return f"{AUDIO_DIR}/{name}"

    def _submit(self, item) -> None:
        if not self.enabled:
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            # Display/audio is cosmetic: drop rather than block the caller.
            log.debug("x2_interact queue full, dropping %s", item[0])

    def _run(self) -> None:
        while True:
            kind, spec = self._q.get()
            try:
                # Coalesce: if newer panels are already queued, only the last
                # one matters -- skip this render entirely.
                if kind == "panel" and self._panel_superseded():
                    continue
                self._dispatch(kind, spec)
            except Exception as e:                      # never kill the worker
                self.failures += 1
                self._last_error = str(e)
                log.warning("x2_interact %s failed: %s", kind, e)
            finally:
                self._q.task_done()

    def _panel_superseded(self) -> bool:
        with self._q.mutex:
            return any(k == "panel" for k, _ in self._q.queue)

    def _dispatch(self, kind: str, spec: dict) -> None:
        if kind == "panel":
            # The spec rides in as base64 inside the command: the outer SSH
            # wrapper already consumes stdin to feed itself, so stdin is not
            # available to carry the payload.
            payload = base64.b64encode(json.dumps(spec).encode()).decode()
            self._ssh(f"echo {payload} | base64 -d | python3 {RENDERER}", timeout=20)
        elif kind == "audio":
            self._play(spec)
        elif kind == "stop":
            self._stop(spec["slot"])
        elif kind == "restore":
            self._ssh("curl -s -m 5 http://127.0.0.1:18080/PlayDefaultEmoji "
                      ">/dev/null", timeout=15)

    def _play(self, spec: dict) -> None:
        path, slot = spec["path"], spec["slot"]
        pidfile = f"/tmp/x2_audio_{slot}.pid"
        # setsid + nohup so playback outlives this SSH session; pid recorded so
        # stop_*() can kill exactly this stream and nothing else.
        inner = (f"aplay -q -D {ALSA_DEV} {shlex.quote(path)}"
                 if not spec.get("loop") else
                 f"while :; do aplay -q -D {ALSA_DEV} {shlex.quote(path)} || break; done")
        cmd = (f"test -f {shlex.quote(path)} || {{ echo 'missing: {path}' >&2; exit 4; }}; "
               f"[ -f {pidfile} ] && kill -- -$(cat {pidfile}) 2>/dev/null; "
               f"setsid nohup bash -c {shlex.quote(inner)} >/dev/null 2>&1 & "
               f"echo $! > {pidfile}")
        self._ssh(cmd, timeout=15)

    def _stop(self, slot: str) -> None:
        pidfile = f"/tmp/x2_audio_{slot}.pid"
        # Negative pid kills the whole setsid process group (the loop AND aplay).
        self._ssh(f"[ -f {pidfile} ] && kill -- -$(cat {pidfile}) 2>/dev/null; "
                  f"rm -f {pidfile}; true", timeout=10)

    def _ssh(self, remote_cmd: str, timeout: int = 20) -> str:
        # The command is base64'd so arbitrary quoting survives both shells.
        # Note this consumes the remote stdin, so payloads must be embedded in
        # the command itself rather than piped in (see _dispatch).
        b64 = base64.b64encode(remote_cmd.encode()).decode()
        wrapped = f"echo {b64} | base64 -d | bash -s"
        argv = ["sshpass", "-p", self.password, "ssh", *SSH_OPTS,
                f"{self.user}@{self.host}", wrapped]
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            raise RuntimeError(f"rc={p.returncode} {p.stderr.strip()[:200]}")
        return p.stdout


# Module-level singleton for callers that just want one line of output.
_default: X2Interact | None = None


def _get() -> X2Interact:
    global _default
    if _default is None:
        _default = X2Interact()
    return _default


def log_msg(message: str, level: str = "INFO") -> None:
    _get().log(message, level)


def say(name: str) -> None:
    _get().say(name)


def banner(text: str, sub: str = "") -> None:
    _get().banner(text, sub)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="X2 speaker/display CLI")
    ap.add_argument("--log"), ap.add_argument("--level", default="INFO")
    ap.add_argument("--banner"), ap.add_argument("--sub", default="")
    ap.add_argument("--say"), ap.add_argument("--music")
    ap.add_argument("--stop-music", action="store_true")
    ap.add_argument("--restore", action="store_true")
    ap.add_argument("--demo", action="store_true", help="exercise every path")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ui = X2Interact()
    if a.demo:
        for lvl, msg in [("OK", "interact module online"),
                         ("INFO", "face_ui 0.0.0.0:18080 reachable"),
                         ("INFO", "audio dmix playback_def ready"),
                         ("WARN", "pose stream degraded (demo text)"),
                         ("OK", "all systems nominal")]:
            ui.log(msg, lvl)
            time.sleep(1.2)
    if a.log:
        ui.log(a.log, a.level)
    if a.banner:
        ui.banner(a.banner, a.sub)
    if a.say:
        ui.say(a.say)
    if a.music:
        ui.music(a.music)
    if a.stop_music:
        ui.stop_music()
    if a.restore:
        ui.restore_face()
    ui.flush()
    print(f"done (failures={ui.failures})")
