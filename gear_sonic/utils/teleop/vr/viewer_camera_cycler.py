"""Cycle the deploy MuJoCo viewer's fixed cameras via xdotool.

Why this exists
---------------

The deploy's MuJoCo viewer (a Python ``mujoco.viewer.launch_passive``
window) cycles through its fixed cameras on the ``Tab`` key. The
operator wears the Quest 3 headset and can't reach the workstation
keyboard, so we let them tap the right thumbstick (in
:mod:`quest3_manager_x2`) and synthesise a ``Tab`` keypress targeted
at the viewer window via ``xdotool``.

This is intentionally an MVP. The "right" architecture is documented
in the ``TODO`` comment on :class:`ViewerCameraCycler` -- a single
``vr_input`` ZMQ topic publishes the full controller state, the
deploy subscribes, and updates ``mjvCamera`` in-process without any
X11 round-trip. That removes the X11 / Wayland / multi-monitor /
window-name-collision footguns of this hack.

Until that lands, this module:

* Locates the viewer window once on first use (cached for the
  lifetime of the process; window IDs are stable as long as the
  viewer isn't restarted).
* Sends a single ``Tab`` keystroke per :meth:`cycle` call, with a
  short cooldown so a noisy thumbstick doesn't fire ten Tabs and
  blow past the camera the operator wanted.
* Logs cleanly on every failure mode (missing ``xdotool``, no
  ``DISPLAY``, no MuJoCo window found) so the operator can diagnose
  without a debugger.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from typing import Optional

log = logging.getLogger(__name__)


class ViewerCameraCycler:
    """Send Tab to the deploy MuJoCo viewer to cycle its fixed cameras.

    Thread-safe (all public methods take a lock); cheap to instantiate
    even when xdotool / DISPLAY are unavailable -- the failure surfaces
    on the first :meth:`cycle` call as a one-shot WARN log line. The
    manager constructs one of these unconditionally and binds it to a
    Quest 3 button; runtime conditions decide whether the Tab actually
    goes anywhere.

    TODO(unified-vr-input-topic): Replace this entire xdotool path with
    a proper ZMQ ``vr_input`` topic published by the manager. Any
    process (deploy viewer, recorder, future logging tools) can then
    subscribe and react to controller events in-process, without the
    X11-forwarding hack. Tracked as a follow-up after the Phase 0 MVP
    settles. See run_x2_quest3_planner_stack.sh banner / cheatsheet
    for the user-facing rationale.
    """

    # How long to wait after a successful cycle before honouring the
    # next press. Stick clicks tend to come in pairs because the
    # gamepad polling loop sees the press for a couple of frames; a
    # 250 ms cooldown collapses those into one Tab without making the
    # operator wait a noticeable beat.
    DEFAULT_COOLDOWN_S: float = 0.25

    # Window classes we always exclude from the candidate list. GNOME's
    # mutter compositor wraps every X11 client in a ``mutter-x11-frames``
    # window that ALSO carries the client's title (e.g. "MuJoCo : x2t2.5")
    # and ALSO matches ``xdotool search --name MuJoCo``. Tab synthesised
    # at that wrapper is silently dropped on the floor (the wrapper is
    # decorative -- it has no GLFW event loop). Excluding it removes
    # ~100% of the "the manager logs cycled but the camera doesn't move"
    # symptom on GNOME-Shell desktops.
    EXCLUDED_WINDOW_CLASSES: tuple[str, ...] = ("mutter-x11-frames",)

    def __init__(
        self,
        *,
        window_search_pattern: str = "MuJoCo",
        window_class_name: str = "MuJoCo",
        cooldown_s: float = DEFAULT_COOLDOWN_S,
    ) -> None:
        self._pattern = window_search_pattern
        self._classname = window_class_name
        self._cooldown_s = float(cooldown_s)
        self._lock = threading.Lock()
        self._cached_wid: Optional[str] = None
        self._last_cycle_t: float = 0.0
        # One-shot log gates so we don't spam the manager's foreground
        # log on every press when something is misconfigured.
        self._warned_no_xdotool = False
        self._warned_no_display = False
        self._warned_no_window = False

    def cycle(self) -> bool:
        """Send a single Tab keystroke to the deploy viewer.

        Returns True if the keystroke was dispatched (xdotool exited 0),
        False otherwise. Never raises -- camera cycling is decorative
        and must not crash the manager control loop.
        """
        with self._lock:
            now = time.monotonic()
            if (now - self._last_cycle_t) < self._cooldown_s:
                return False

            if shutil.which("xdotool") is None:
                if not self._warned_no_xdotool:
                    log.warning(
                        "[viewer-cycler] xdotool not installed; install "
                        "with `sudo apt install xdotool` to enable "
                        "right-stick-click camera cycling. Press is a "
                        "no-op until then."
                    )
                    self._warned_no_xdotool = True
                return False

            if not os.environ.get("DISPLAY"):
                if not self._warned_no_display:
                    log.warning(
                        "[viewer-cycler] no DISPLAY env var; this is "
                        "expected on headless runs (--no-sim-viewer). "
                        "Camera cycling will be a no-op."
                    )
                    self._warned_no_display = True
                return False

            wid = self._cached_wid or self._locate_window()
            if wid is None:
                if not self._warned_no_window:
                    log.warning(
                        "[viewer-cycler] no window matching %r found "
                        "via xdotool. Is the deploy MuJoCo viewer up? "
                        "Re-run the planner stack with viewer enabled "
                        "(default) and click the deploy window once.",
                        self._pattern,
                    )
                    self._warned_no_window = True
                return False
            self._cached_wid = wid

            ok = self._send_tab(wid)
            if ok:
                self._last_cycle_t = now
            return ok

    def reset(self) -> None:
        """Clear the cached window ID so the next :meth:`cycle` re-searches.

        Call this if the deploy is restarted mid-session (the new
        viewer process gets a new X11 window ID; the cached one is
        stale and any Tab sent to it lands in the void).
        """
        with self._lock:
            self._cached_wid = None
            self._warned_no_window = False

    # -- internals ---------------------------------------------------

    def _locate_window(self) -> Optional[str]:
        """Run ``xdotool search`` and return the most-likely window ID.

        Search strategy (most-precise first):

        1. ``xdotool search --classname <classname>``. WM_CLASS is set
           by GLFW directly on the application window and is *not*
           inherited by the GNOME / mutter compositor's frame wrapper,
           so this filter zeroes in on the real GLFW window without
           any post-filtering. This is the path that fixes the
           common "manager logs cycled but the camera doesn't move on
           GNOME-Shell" footgun (the wrapper window matches the title
           filter and silently swallows the synthetic Tab).

        2. ``xdotool search --name <pattern>``. Title-substring fallback
           used when the classname search returns nothing -- some
           builds of MuJoCo / GLFW set WM_CLASS to something other
           than the default ("python3", a custom value via
           ``glfwWindowHintString``, etc.). We post-filter the title
           hits to drop anything whose WM_CLASS sits in
           :attr:`EXCLUDED_WINDOW_CLASSES` (currently
           ``mutter-x11-frames``), so the wrapper window can never
           sneak in even if the operator narrowed the pattern.

        If either search returns multiple candidates after filtering
        we log them all and pick the first; the operator can always
        narrow further with ``--viewer-window-pattern`` /
        ``--viewer-window-classname`` on the manager CLI.
        """
        # 1) classname search — the precise path
        wids = self._xdotool_search(["--classname", self._classname])
        wids = self._drop_excluded_classes(wids)
        if wids:
            self._maybe_log_multiple(wids, source=f"--classname {self._classname!r}")
            return wids[0]

        # 2) name-substring fallback — the legacy path
        wids = self._xdotool_search(["--name", self._pattern])
        wids = self._drop_excluded_classes(wids)
        if not wids:
            return None
        self._maybe_log_multiple(wids, source=f"--name {self._pattern!r}")
        return wids[0]

    def _xdotool_search(self, search_args: list[str]) -> list[str]:
        """Run a single ``xdotool search ...`` and return the WIDs.

        Returns ``[]`` on failure / timeout / non-zero exit. Logs once
        on actual subprocess failure (timeout / OSError) but stays
        quiet on "no match" because the caller may have a fallback
        path it still wants to try.
        """
        try:
            out = subprocess.run(
                ["xdotool", "search", *search_args],
                capture_output=True, text=True, timeout=2.0,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning(
                "[viewer-cycler] xdotool search %s failed: %s",
                search_args, exc,
            )
            return []
        if out.returncode != 0 or not out.stdout.strip():
            return []
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]

    def _drop_excluded_classes(self, wids: list[str]) -> list[str]:
        """Filter out WIDs whose WM_CLASS is in
        :attr:`EXCLUDED_WINDOW_CLASSES`.

        We invoke ``xdotool getwindowclassname`` per WID rather than
        using ``xdotool search --class --classname`` jointly because
        ``--class`` and ``--classname`` are *intersected* with
        ``--name`` when given together, which would prevent the
        classname-first strategy from ever finding the GLFW window
        on builds where its title doesn't include "MuJoCo".
        """
        if not wids:
            return wids
        kept: list[str] = []
        dropped: list[str] = []
        for wid in wids:
            cls = self._get_window_class(wid)
            if cls in self.EXCLUDED_WINDOW_CLASSES:
                dropped.append(f"{wid} (class={cls!r})")
            else:
                kept.append(wid)
        if dropped:
            log.info(
                "[viewer-cycler] excluded %d compositor wrapper "
                "window(s) from candidate list: %s",
                len(dropped), dropped,
            )
        return kept

    @staticmethod
    def _get_window_class(wid: str) -> str:
        """Best-effort fetch of WM_CLASS for ``wid``.

        Returns ``""`` on any failure -- the caller treats unknown
        class as "not excluded" (defensive: we'd rather Tab a
        possibly-wrong window than skip the legitimate one because
        ``xprop`` was missing).
        """
        try:
            res = subprocess.run(
                ["xdotool", "getwindowclassname", wid],
                capture_output=True, text=True, timeout=1.0,
            )
        except (subprocess.TimeoutExpired, OSError):
            return ""
        if res.returncode != 0:
            return ""
        return res.stdout.strip()

    def _maybe_log_multiple(self, wids: list[str], *, source: str) -> None:
        """Log all candidate WIDs when the search returned >1, so
        the operator can narrow the pattern via the manager CLI if we
        picked the wrong one.
        """
        if len(wids) > 1:
            log.info(
                "[viewer-cycler] multiple windows match %s: %s; "
                "using the first (%s). If wrong, narrow with "
                "--viewer-window-classname / --viewer-window-pattern.",
                source, wids, wids[0],
            )

    def _send_tab(self, wid: str) -> bool:
        """Synthesize a Tab keypress on the cached window."""
        try:
            res = subprocess.run(
                ["xdotool", "key", "--window", wid, "Tab"],
                capture_output=True, text=True, timeout=2.0,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            log.warning("[viewer-cycler] xdotool key failed: %s", exc)
            return False
        if res.returncode != 0:
            # Most likely the window was closed under us (deploy
            # viewer killed). Drop the cache so the next press
            # re-locates whatever new window is up.
            log.warning(
                "[viewer-cycler] xdotool key exit=%d stderr=%r; "
                "window may have closed, will re-locate on next press.",
                res.returncode, res.stderr.strip(),
            )
            self._cached_wid = None
            self._warned_no_window = False
            return False
        return True


__all__ = ["ViewerCameraCycler"]
