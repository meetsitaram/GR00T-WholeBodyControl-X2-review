"""Tests for the right-thumbstick-click camera cycler.

Covers three layers, top-down:

1. ``Quest3Reader.get_stick_clicks`` parses the WebXR client's
   ``leftStickClick`` / ``rightStickClick`` payload fields and degrades
   gracefully on older clients that don't include them.

2. :class:`ViewerCameraCycler` shells out to ``xdotool`` correctly,
   honours the cooldown window, caches the window ID across calls,
   drops the cache when the window goes away, and never raises when
   xdotool / DISPLAY are missing.

3. ``Quest3ManagerX2`` wires the rising-edge detector + cycler into
   the main loop -- a single press fires one ``cycle()`` call, a
   sustained press doesn't fire repeatedly, and OFF mode suppresses
   the dispatch entirely.

These together make the ``TODO(unified-vr-input-topic)`` migration
mechanical: when the new topic lands, replace the cycler internals
and re-run these tests; the public surface (rising-edge + mode gate)
stays the same.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Layer 1: Quest3Reader.get_stick_clicks
# ---------------------------------------------------------------------------


def _make_reader_with_sample(sample):
    """Build a Quest3Reader with ``get_latest`` stubbed to return ``sample``.

    We don't need the full reader infrastructure (websocket server,
    HTTP server, gtts) to test a pure parsing method, so we patch
    ``__init__`` to a no-op and bolt on the minimal attributes
    ``get_stick_clicks`` reads.
    """
    from gear_sonic.utils.teleop.vr.quest3_reader import Quest3Reader

    reader = Quest3Reader.__new__(Quest3Reader)
    reader.get_latest = MagicMock(return_value=sample)  # type: ignore[method-assign]
    return reader


def test_get_stick_clicks_returns_false_when_no_sample():
    reader = _make_reader_with_sample(None)
    assert reader.get_stick_clicks() == (False, False)


def test_get_stick_clicks_parses_both_clicks():
    reader = _make_reader_with_sample(
        {"buttons": {"leftStickClick": True, "rightStickClick": True}}
    )
    assert reader.get_stick_clicks() == (True, True)


def test_get_stick_clicks_independent_of_face_buttons():
    """A and B held while neither stick clicks must NOT report clicks --
    proves we're reading the right keys (not falling back to
    ``buttons["a"]`` or similar)."""
    reader = _make_reader_with_sample(
        {"buttons": {"a": True, "b": True, "x": True, "y": True}}
    )
    assert reader.get_stick_clicks() == (False, False)


def test_get_stick_clicks_handles_old_client_gracefully():
    """WebXR clients that pre-date the stick-click forwarding patch
    just don't include the keys in the payload. We must default
    those to False instead of raising KeyError."""
    reader = _make_reader_with_sample(
        {"buttons": {"a": False}}  # no leftStickClick / rightStickClick keys
    )
    assert reader.get_stick_clicks() == (False, False)


def test_get_stick_clicks_handles_missing_buttons_dict():
    """Some early-frame samples arrive before the gamepad polling
    loop populates ``buttons``; the field is then absent entirely.
    Must not raise."""
    reader = _make_reader_with_sample({})
    assert reader.get_stick_clicks() == (False, False)


# ---------------------------------------------------------------------------
# Layer 2: ViewerCameraCycler -- xdotool plumbing
# ---------------------------------------------------------------------------


@pytest.fixture
def cycler():
    from gear_sonic.utils.teleop.vr.viewer_camera_cycler import (
        ViewerCameraCycler,
    )
    # Tiny cooldown so the per-test rate-limit-skip behaviour can be
    # exercised without time.sleep'ing in tests.
    return ViewerCameraCycler(
        window_search_pattern="MuJoCo",
        cooldown_s=0.05,
    )


def _xdotool_proc(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess:
    """Build a fake CompletedProcess for any xdotool invocation."""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr="",
    )


# Convenience aliases for readability in the side_effect lists below.
def _xdotool_search_ok(stdout: str = "12345678\n") -> subprocess.CompletedProcess:
    return _xdotool_proc(stdout=stdout)


def _xdotool_search_empty() -> subprocess.CompletedProcess:
    """``xdotool search`` exits 1 with empty stdout when nothing matches."""
    return _xdotool_proc(stdout="", returncode=1)


def _xdotool_getclass(cls: str) -> subprocess.CompletedProcess:
    return _xdotool_proc(stdout=f"{cls}\n")


def _xdotool_key_ok() -> subprocess.CompletedProcess:
    return _xdotool_proc(stdout="")


def _xdotool_key_fail() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=[], returncode=1, stdout="",
        stderr="X11 error: window does not exist\n",
    )


def _classname_match_flow(
    wid: str = "42", cls: str = "MuJoCo",
) -> list[subprocess.CompletedProcess]:
    """Side-effect list simulating a happy classname-search match.

    The cycler does, per uncached cycle():
      (1) xdotool search --classname <classname>
      (2) xdotool getwindowclassname <wid>           (per returned WID)
      (3) xdotool key --window <wid> Tab

    Returns a 3-element list ready for ``side_effect`` so tests don't
    have to repeat the boilerplate. Add additional flows as
    consecutive list entries for tests that exercise multiple cycles.
    """
    return [
        _xdotool_search_ok(f"{wid}\n"),
        _xdotool_getclass(cls),
        _xdotool_key_ok(),
    ]


def test_cycler_dispatches_xdotool_key_tab_to_found_window(cycler):
    """Happy path: classname search finds a window, class filter
    keeps it, Tab is sent to that WID."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=_classname_match_flow(wid="42", cls="MuJoCo"),
    ) as m_run:
        assert cycler.cycle() is True

    assert m_run.call_count == 3
    search_call, getclass_call, key_call = m_run.call_args_list
    # Primary search must be by --classname (the precise filter
    # that avoids GNOME mutter's frame wrapper).
    assert search_call.args[0] == ["xdotool", "search", "--classname", "MuJoCo"]
    assert getclass_call.args[0] == ["xdotool", "getwindowclassname", "42"]
    # Camera-cycle key must be ``bracketright`` (the X11 keysym for
    # ``]``, mujoco viewer's "next fixed camera"). This regression
    # pin exists because v7.1 originally synthesised ``Tab``, which
    # actually toggles the viewer's left UI panel and does NOT
    # change cameras -- the bug surfaced live in the deploy and
    # cost us a test session. See ViewerCameraCycler class docstring.
    assert key_call.args[0] == ["xdotool", "key", "--window", "42", "bracketright"]


def test_cycler_prefers_classname_over_name_fallback(cycler):
    """When the classname search returns a valid WID, the name
    fallback must NOT run (one X11 round-trip beats two)."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=_classname_match_flow(wid="42"),
    ) as m_run:
        assert cycler.cycle() is True

    args_seen = [c.args[0] for c in m_run.call_args_list]
    assert all(
        ["--name" not in a for a in args_seen]
    ), f"--name fallback should not have run; saw {args_seen}"


def test_cycler_excludes_mutter_x11_frames_wrapper(cycler):
    """The exact bug from 2026-05-13: ``xdotool search --name MuJoCo``
    on GNOME-Shell returns BOTH the GLFW window AND mutter's frame
    wrapper (class=mutter-x11-frames). The wrapper has the right
    title but doesn't process Tab events. The cycler must drop it
    and pick the real GLFW window.

    Here we force the classname search to return EMPTY (so the
    fallback name path runs), have the name search return two WIDs
    (the wrapper first, then the real GLFW window), and assert Tab
    lands on the real window.
    """
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=[
            _xdotool_search_empty(),                     # --classname empty
            _xdotool_search_ok("6295708\n77594631\n"),  # --name returns wrapper + real
            _xdotool_getclass("mutter-x11-frames"),      # 6295708 -> EXCLUDED
            _xdotool_getclass("MuJoCo"),                 # 77594631 -> kept
            _xdotool_key_ok(),
        ],
    ) as m_run:
        assert cycler.cycle() is True

    # Camera-cycle key must land on the REAL GLFW window, not the mutter wrapper.
    key_call = m_run.call_args_list[-1]
    assert key_call.args[0] == [
        "xdotool", "key", "--window", "77594631", "bracketright",
    ], (
        f"Camera-cycle key landed on the wrong window. Full call list:\n"
        f"{[c.args[0] for c in m_run.call_args_list]}"
    )


def test_cycler_falls_back_to_name_search_when_classname_empty(cycler):
    """Operator runs a custom MuJoCo build whose WM_CLASS isn't
    ``MuJoCo`` (e.g. a GLFW build that defaults to ``python3``).
    The classname search returns nothing; the name fallback finds
    the window by title substring."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=[
            _xdotool_search_empty(),       # --classname returns nothing
            _xdotool_search_ok("42\n"),    # --name returns the window
            _xdotool_getclass("python3"),  # not in EXCLUDED_WINDOW_CLASSES
            _xdotool_key_ok(),
        ],
    ) as m_run:
        assert cycler.cycle() is True

    # Verify both searches ran in the right order.
    args_seen = [c.args[0] for c in m_run.call_args_list]
    assert args_seen[0] == ["xdotool", "search", "--classname", "MuJoCo"]
    assert args_seen[1] == ["xdotool", "search", "--name", "MuJoCo"]
    assert args_seen[-1] == ["xdotool", "key", "--window", "42", "bracketright"]


def test_cycler_caches_window_id_across_calls(cycler):
    """Subsequent calls must reuse the cached WID and skip both the
    classname search AND the per-WID class lookup (the expensive
    parts of the X11 round-trip)."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=[
            *_classname_match_flow(wid="42"),  # first cycle: search + getclass + key
            _xdotool_key_ok(),                  # second cycle: cached -> only key
            _xdotool_key_ok(),                  # third  cycle: cached -> only key
        ],
    ) as m_run:
        cycler.cycle()
        import time
        time.sleep(0.06)
        cycler.cycle()
        time.sleep(0.06)
        cycler.cycle()

    # 1 search + 1 getclass + 3 keys = 5 calls. Without caching
    # we'd see 9 (3 searches + 3 getclasses + 3 keys).
    assert m_run.call_count == 5
    args_seen = [c.args[0] for c in m_run.call_args_list]
    assert args_seen[0][:3] == ["xdotool", "search", "--classname"]
    assert args_seen[1][:2] == ["xdotool", "getwindowclassname"]
    for a in args_seen[2:]:
        assert a[:2] == ["xdotool", "key"]


def test_cycler_cooldown_drops_back_to_back_calls(cycler):
    """Two ``cycle()`` calls within the cooldown must collapse to one
    real keystroke -- otherwise a noisy stick fires 5 Tabs and
    overshoots the camera the operator wanted."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=_classname_match_flow(wid="42"),
    ) as m_run:
        assert cycler.cycle() is True
        # Immediately call again -- inside cooldown, must short-circuit.
        assert cycler.cycle() is False

    # First cycle = 3 calls; cooldown shorts the second to 0 extra.
    assert m_run.call_count == 3


def test_cycler_drops_cache_on_xdotool_key_failure(cycler):
    """If xdotool key fails (most likely because the deploy viewer
    was killed), the cached WID must be invalidated so the next
    cycle() re-searches and finds the new window."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=[
            # First cycle: classname-search -> getclass -> key (FAILS)
            _xdotool_search_ok("42\n"),
            _xdotool_getclass("MuJoCo"),
            _xdotool_key_fail(),
            # Cache invalidated; second cycle re-runs the full flow
            # against the new window 99.
            _xdotool_search_ok("99\n"),
            _xdotool_getclass("MuJoCo"),
            _xdotool_key_ok(),
        ],
    ) as m_run:
        assert cycler.cycle() is False
        import time
        time.sleep(0.06)
        assert cycler.cycle() is True

    assert m_run.call_count == 6
    # The recovery key call must target the NEW window (99), not 42.
    final_key_call = m_run.call_args_list[-1]
    assert "99" in final_key_call.args[0]


def test_cycler_no_op_when_xdotool_missing(cycler):
    """Missing ``xdotool`` binary must short-circuit without raising
    (camera cycling is decorative; the manager loop must keep going)."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value=None,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
    ) as m_run:
        assert cycler.cycle() is False
    m_run.assert_not_called()


def test_cycler_no_op_when_no_display(cycler):
    """Headless / CI: no DISPLAY -> no-op, no subprocess call."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {}, clear=True,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
    ) as m_run:
        assert cycler.cycle() is False
    m_run.assert_not_called()


def test_cycler_no_op_when_window_not_found(cycler):
    """Both classname AND name searches return empty -> warn once,
    return False, never try to send Tab to a non-existent window."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=[_xdotool_search_empty(), _xdotool_search_empty()],
    ) as m_run:
        assert cycler.cycle() is False

    # Both classname + name searches ran; no getwindowclassname, no key.
    assert m_run.call_count == 2
    args_seen = [c.args[0] for c in m_run.call_args_list]
    assert args_seen[0][:3] == ["xdotool", "search", "--classname"]
    assert args_seen[1][:3] == ["xdotool", "search", "--name"]


def test_cycler_reset_clears_cache(cycler):
    """Operator workflow: deploy was restarted mid-session, the
    operator calls reset() (or it happens automatically), and the
    next cycle() re-runs the full search flow for the new window."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=[
            *_classname_match_flow(wid="42"),
            *_classname_match_flow(wid="99"),
        ],
    ) as m_run:
        cycler.cycle()
        cycler.reset()
        import time
        time.sleep(0.06)
        cycler.cycle()

    # Two full flows = 6 calls (2 searches + 2 getclasses + 2 keys).
    assert m_run.call_count == 6


def test_cycler_subprocess_timeout_logged_not_raised(cycler):
    """Whatever the OS throws -- TimeoutExpired, OSError -- must
    surface as a False return, never propagate."""
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=subprocess.TimeoutExpired(["xdotool"], 2.0),
    ):
        assert cycler.cycle() is False


# ---------------------------------------------------------------------------
# Layer 2b: ViewerCameraCycler -- periodic WARN re-emission
# ---------------------------------------------------------------------------


def _drain_cycle_inside_cooldown(cycler) -> None:
    """Sleep past the cycler's cooldown so the next cycle() runs.

    Lets us call cycle() repeatedly in tests without each call
    short-circuiting on the rate-limit gate.
    """
    import time
    time.sleep(cycler._cooldown_s + 0.01)


def test_cycler_warns_once_then_silences_within_reemit_period(cycler, caplog):
    """Two failures in quick succession must produce only ONE WARN
    (the existing one-shot behaviour). Without this, a noisy stick
    spamming clicks at 50 Hz would also spam the manager log with
    50 identical warnings per second."""
    import logging

    caplog.set_level(logging.WARNING, logger="gear_sonic.utils.teleop.vr.viewer_camera_cycler")
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value=None,  # xdotool missing -> WARN path
    ):
        assert cycler.cycle() is False
        _drain_cycle_inside_cooldown(cycler)
        assert cycler.cycle() is False
        _drain_cycle_inside_cooldown(cycler)
        assert cycler.cycle() is False

    warns = [r for r in caplog.records if "xdotool not installed" in r.getMessage()]
    assert len(warns) == 1, (
        f"Expected exactly 1 'xdotool not installed' WARN inside "
        f"WARN_REEMIT_PERIOD_S; got {len(warns)}."
    )


def test_cycler_reemits_warn_after_period_elapses(cycler, caplog):
    """Long-session diagnostic: if the failure persists past
    ``WARN_REEMIT_PERIOD_S``, the next cycle() WARN re-emits so the
    operator gets a fresh log line instead of the cycler going
    permanently silent. Critical for the "I started the stack
    before the deploy was up, then nothing ever logged again"
    failure mode."""
    import logging

    caplog.set_level(logging.WARNING, logger="gear_sonic.utils.teleop.vr.viewer_camera_cycler")
    # Force the period low so the test is fast.
    cycler.WARN_REEMIT_PERIOD_S = 0.01
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value=None,
    ):
        assert cycler.cycle() is False
        # Wait past BOTH the cooldown AND the re-emit period.
        import time
        time.sleep(max(cycler._cooldown_s, cycler.WARN_REEMIT_PERIOD_S) + 0.02)
        assert cycler.cycle() is False

    warns = [r for r in caplog.records if "xdotool not installed" in r.getMessage()]
    assert len(warns) == 2, (
        f"Expected 2 WARNs after period elapsed; got {len(warns)}. "
        "Re-emission gate is broken; long sessions will go silent "
        "after the first failure."
    )


def test_cycler_reset_re_arms_no_window_warn(cycler, caplog):
    """``reset()`` clears both the WID cache AND the no-window WARN
    gate so the post-reset cycle() can warn freshly if the new
    deploy still doesn't have a viewer up. Without this the
    operator would never see a follow-up warn after a deploy
    restart."""
    import logging

    caplog.set_level(logging.WARNING, logger="gear_sonic.utils.teleop.vr.viewer_camera_cycler")
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=[
            _xdotool_search_empty(), _xdotool_search_empty(),
            _xdotool_search_empty(), _xdotool_search_empty(),
        ],
    ):
        assert cycler.cycle() is False
        _drain_cycle_inside_cooldown(cycler)
        cycler.reset()
        _drain_cycle_inside_cooldown(cycler)
        assert cycler.cycle() is False

    no_window_warns = [
        r for r in caplog.records if "no window matching" in r.getMessage()
    ]
    assert len(no_window_warns) == 2, (
        f"Expected 2 no-window WARNs (one before reset, one after); "
        f"got {len(no_window_warns)}."
    )


def test_cycler_default_keysym_is_bracketright_not_tab():
    """REGRESSION: in v7.1 we initially used ``Tab`` because a stale
    cheatsheet said "Tab cycles cameras". It does NOT -- in
    mujoco.viewer.launch_passive Tab toggles the left UI panel and
    leaves the active camera unchanged. The actual next-fixed-camera
    key is ``]`` (X11 keysym ``bracketright``), per the upstream
    docs (https://mujoco.readthedocs.io/en/2.3.6/programming/visualization.html).

    This test pins the default so any future "let's go back to Tab"
    refactor surfaces immediately rather than at a live deploy.
    """
    from gear_sonic.utils.teleop.vr.viewer_camera_cycler import (
        ViewerCameraCycler,
    )

    assert ViewerCameraCycler.CYCLE_KEYSYM == "bracketright", (
        "ViewerCameraCycler.CYCLE_KEYSYM regressed to %r. The MuJoCo "
        "passive viewer cycles fixed cameras on `]` (X11 keysym "
        "`bracketright`); `Tab` only toggles the left UI panel and "
        "must NOT be used here." % ViewerCameraCycler.CYCLE_KEYSYM
    )


def test_cycler_honors_overridden_keysym(cycler):
    """Operator override path: setting CYCLE_KEYSYM on the instance
    redirects the synthesized keystroke. This lets us swap to
    ``bracketleft`` (cycle backwards) or ``Escape`` (snap to free
    camera) without subclassing."""
    cycler.CYCLE_KEYSYM = "bracketleft"
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":1"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=_classname_match_flow(wid="42"),
    ) as m_run:
        assert cycler.cycle() is True

    key_call = m_run.call_args_list[-1]
    assert key_call.args[0] == [
        "xdotool", "key", "--window", "42", "bracketleft",
    ], f"Override didn't take effect; saw: {key_call.args[0]}"


def test_cycler_no_window_warn_includes_display_for_diagnosis(cycler, caplog):
    """The no-window WARN must include the DISPLAY value -- this is
    the single most common operator-error cause (running on Wayland
    where xdotool returns empty for every search; or having
    DISPLAY=:0 when the deploy launched on :1). Without it the
    operator can't distinguish "viewer not up" from "wrong DISPLAY"."""
    import logging

    caplog.set_level(logging.WARNING, logger="gear_sonic.utils.teleop.vr.viewer_camera_cycler")
    with patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.shutil.which",
        return_value="/usr/bin/xdotool",
    ), patch.dict(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.os.environ",
        {"DISPLAY": ":7"}, clear=False,
    ), patch(
        "gear_sonic.utils.teleop.vr.viewer_camera_cycler.subprocess.run",
        side_effect=[_xdotool_search_empty(), _xdotool_search_empty()],
    ):
        assert cycler.cycle() is False

    no_window_warns = [
        r for r in caplog.records if "no window matching" in r.getMessage()
    ]
    assert len(no_window_warns) == 1
    msg = no_window_warns[0].getMessage()
    assert "DISPLAY=:7" in msg, (
        f"WARN must include the actual DISPLAY value for diagnosis; got: {msg}"
    )
