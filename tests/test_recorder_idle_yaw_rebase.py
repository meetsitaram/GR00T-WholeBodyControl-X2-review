"""Tests for the recorder's idle-frame yaw rebase.

The recorder publishes a "stand still" idle frame at ~50 Hz whenever
no body_pose source has come up yet (boot gap, kplanner warmup).
Historically that frame carried an identity ``root_quat_xyzw``, which
made the SONIC tracking policy briefly twist the body back toward
world +X via ``waist_yaw_joint`` -- a loud waist click at startup on
real robot. Fix (2026-06-02): rebase the idle frame's root_quat to the
live ``x2_debug`` ``base_quat`` (yaw-only re-projection), falling back
to identity when ``x2_debug`` is silent / never arrived.

The tests in this module exercise ``_compute_idle_root_quat_xyzw`` and
``_publish_idle`` via a thin harness that re-binds only those methods
onto a dummy holding the minimal state -- same pattern used in
:file:`test_recorder_gesture_gate.py` to avoid spinning up the full
:class:`X2DatasetRecorder` (which transitively pulls in the LeRobot
writer chain, ZMQ binds, manager subprocess, etc.).

Invariants pinned:

1. **Fresh x2_debug + non-trivial yaw -> R_z(yaw) on the wire.** The
   xy components of the quat stay 0 (pitch/roll dropped on purpose).
2. **Stale / never-received x2_debug -> identity.** Reverts to the
   pre-fix behaviour so the fix never regresses the wire shape.
3. **One-shot log gating.** The "ACTIVE" line fires once on the first
   successful rebase, the "stale" line fires once on the next stale
   tick, and 50 Hz of repeated activity does NOT spam logs.
4. **Round-trip recovery logs.** If x2_debug goes stale and comes
   back, the next ACTIVE line fires again (so the operator sees the
   recovery).
5. **`_publish_idle` glue.** When the helper returns a non-None quat,
   `_publish_pose` is called with that exact ``root_quat_xyzw=`` kwarg;
   when None, the kwarg is None and `_publish_pose` falls back to
   identity internally (its existing default).
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.live_vla_publish_motion_token import (  # noqa: E402
    _LatestState,
)
from gear_sonic.utils.teleop.x2_dataset_recorder import (  # noqa: E402
    X2DatasetRecorder,
)


class _IdleHarness:
    """Minimal stand-in for the bits of :class:`X2DatasetRecorder` the
    idle-yaw-rebase methods touch.

    No ZMQ, no threads, no writer. Re-binds the two real recorder
    methods so we test the production code directly (a regression in
    those methods immediately fails the tests here).
    """

    def __init__(self, *, latest_state: _LatestState, verbose: bool = False):
        self._latest_state = latest_state
        # Just needs ``.verbose``; anything else the methods reach for
        # will MagicMock-AttributeError loudly in tests, which is what
        # we want for catching scope creep.
        self._cfg = MagicMock(verbose=verbose)
        self._idle_yaw_rebase_logged_active = False
        self._idle_yaw_rebase_logged_fallback = False
        self._zero_motion_token = np.zeros(64, dtype=np.float64)
        self.publish_calls: list[dict[str, Any]] = []

    _compute_idle_root_quat_xyzw = (
        X2DatasetRecorder._compute_idle_root_quat_xyzw
    )
    _publish_idle = X2DatasetRecorder._publish_idle

    def _publish_pose(self, **kwargs: Any) -> None:
        # Capture verbatim so tests can assert what _publish_idle sent.
        self.publish_calls.append(kwargs)


def _state_with_yaw(yaw_rad: float) -> _LatestState:
    """Build a ``_LatestState`` already populated with one ``base_quat``
    representing a pure ``R_z(yaw_rad)`` rotation, marked alive."""
    state = _LatestState()
    half = 0.5 * yaw_rad
    state.update(
        body_q_mj=np.zeros(31, dtype=np.float64),
        base_quat_wxyz=np.array(
            [math.cos(half), 0.0, 0.0, math.sin(half)],
            dtype=np.float64,
        ),
        left_hand_q=np.zeros(7, dtype=np.float64),
        right_hand_q=np.zeros(7, dtype=np.float64),
    )
    return state


# ---------------------------------------------------------------------------
# 1. fresh x2_debug + non-trivial yaw -> R_z(yaw)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaw_deg", [-179.0, -90.0, -45.0, -10.0, 0.0, 10.0, 45.0, 90.0, 179.0]
)
def test_fresh_xdebug_returns_yaw_only_quat(yaw_deg: float) -> None:
    """A live x2_debug snapshot with pelvis yaw=Y must produce
    root_quat_xyzw = (0, 0, sin(Y/2), cos(Y/2)) (a pure world-Z rotation
    in xyzw order)."""
    yaw_rad = math.radians(yaw_deg)
    h = _IdleHarness(latest_state=_state_with_yaw(yaw_rad))

    quat = h._compute_idle_root_quat_xyzw()

    assert quat is not None, "fresh x2_debug must produce a non-None quat"
    assert quat.shape == (4,)
    assert quat.dtype == np.float32

    qx, qy, qz, qw = quat.tolist()
    half = 0.5 * yaw_rad
    expected_qz = math.sin(half)
    expected_qw = math.cos(half)

    assert qx == pytest.approx(0.0, abs=1e-6), (
        "pitch/roll must be dropped on purpose -- qx should be 0"
    )
    assert qy == pytest.approx(0.0, abs=1e-6), (
        "pitch/roll must be dropped on purpose -- qy should be 0"
    )
    assert qz == pytest.approx(expected_qz, abs=1e-5), (
        f"qz mismatch for yaw={yaw_deg}deg: got {qz}, want {expected_qz}"
    )
    assert qw == pytest.approx(expected_qw, abs=1e-5), (
        f"qw mismatch for yaw={yaw_deg}deg: got {qw}, want {expected_qw}"
    )


def test_fresh_xdebug_pitch_and_roll_are_dropped() -> None:
    """If the pelvis quat carries pitch/roll (e.g. mid-fall lean),
    the returned idle root_quat must STILL be a pure world-Z rotation
    -- a transient lean must not bleed into the upright training-
    distribution reference."""
    from scipy.spatial.transform import Rotation as Rot

    # Build a quat with non-trivial pitch + roll + yaw, in scipy
    # extrinsic ZYX convention (matches yaw_of_quat_xyzw).
    yaw_rad = math.radians(35.0)
    rot = Rot.from_euler("zyx", [yaw_rad, math.radians(15.0), math.radians(-10.0)])
    q_xyzw = rot.as_quat()
    state = _LatestState()
    state.update(
        body_q_mj=np.zeros(31, dtype=np.float64),
        base_quat_wxyz=np.array(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64
        ),
        left_hand_q=np.zeros(7, dtype=np.float64),
        right_hand_q=np.zeros(7, dtype=np.float64),
    )

    h = _IdleHarness(latest_state=state)
    quat = h._compute_idle_root_quat_xyzw()

    assert quat is not None
    qx, qy, qz, qw = quat.tolist()
    assert qx == pytest.approx(0.0, abs=1e-6)
    assert qy == pytest.approx(0.0, abs=1e-6)
    # Yaw component should match the scipy ZYX yaw exactly.
    half = 0.5 * yaw_rad
    assert qz == pytest.approx(math.sin(half), abs=1e-5)
    assert qw == pytest.approx(math.cos(half), abs=1e-5)


# ---------------------------------------------------------------------------
# 2. stale / never-received x2_debug -> None (identity fallback)
# ---------------------------------------------------------------------------


def test_never_received_xdebug_returns_none() -> None:
    """Fresh _LatestState defaults (received_any=False) -> alive=False
    -> helper returns None so _publish_pose falls back to identity."""
    h = _IdleHarness(latest_state=_LatestState())
    quat = h._compute_idle_root_quat_xyzw()
    assert quat is None


def test_stale_xdebug_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once last_update_monotonic ages past DEPLOY_ALIVE_STALE_THRESHOLD_S,
    is_alive flips to False and the helper returns None (matches the
    pre-fix wire behaviour exactly)."""
    state = _state_with_yaw(math.radians(45.0))
    # Force the stored last_update_monotonic into the deep past.
    state.last_update_monotonic = time.monotonic() - 1e6

    h = _IdleHarness(latest_state=state)
    quat = h._compute_idle_root_quat_xyzw()
    assert quat is None


# ---------------------------------------------------------------------------
# 3. log gating: ACTIVE fires once, fallback fires once, 50 Hz doesn't spam
# ---------------------------------------------------------------------------


def test_log_gates_active_message_fires_exactly_once(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The first successful rebase prints one ACTIVE line. Subsequent
    successful rebases at 50 Hz must NOT re-print -- the flag pins
    the gate one-way."""
    h = _IdleHarness(latest_state=_state_with_yaw(math.radians(30.0)))

    for _ in range(100):  # 100 ticks = 2s @ 50 Hz worth of idle frames
        h._compute_idle_root_quat_xyzw()

    captured = capsys.readouterr()
    active_lines = [
        l for l in captured.out.splitlines()
        if "idle yaw-rebase: ACTIVE" in l
    ]
    assert len(active_lines) == 1, (
        f"expected exactly one ACTIVE log line across 100 ticks; got "
        f"{len(active_lines)}: {active_lines}"
    )
    assert h._idle_yaw_rebase_logged_active is True


def test_log_gates_fallback_message_fires_only_after_active(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If x2_debug never came up at all, the helper should NOT spam a
    fallback message every tick. The fallback line is gated on
    ``_idle_yaw_rebase_logged_active`` so it only fires when the
    rebase was *previously* active and just went stale -- the operator-
    visible 'we lost the deploy' transition. Pre-active stale state is
    a silent no-op fallback (the boot gap)."""
    h = _IdleHarness(latest_state=_LatestState())

    for _ in range(100):
        h._compute_idle_root_quat_xyzw()

    captured = capsys.readouterr()
    fallback_lines = [
        l for l in captured.out.splitlines()
        if "idle yaw-rebase: x2_debug went stale" in l
    ]
    assert len(fallback_lines) == 0, (
        f"pre-active stale must not log a fallback line (would spam at "
        f"50 Hz during boot gap); got {fallback_lines}"
    )


def test_log_round_trip_active_then_stale_then_active(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A realistic operator-visible sequence: rebase comes up,
    deploy quits, deploy comes back. We want exactly:
      * 1 ACTIVE line on first success
      * 1 stale-fallback line on the next stale tick
      * 1 ACTIVE line again on recovery
    """
    state = _state_with_yaw(math.radians(20.0))
    h = _IdleHarness(latest_state=state)

    # Phase 1: first ACTIVE.
    assert h._compute_idle_root_quat_xyzw() is not None

    # Phase 2: force stale -> fallback line fires.
    state.last_update_monotonic = time.monotonic() - 1e6
    assert h._compute_idle_root_quat_xyzw() is None
    # Subsequent stale ticks must NOT re-log.
    for _ in range(10):
        h._compute_idle_root_quat_xyzw()

    # Phase 3: recovery -- bump update + re-mark alive.
    state.update(
        body_q_mj=np.zeros(31, dtype=np.float64),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        left_hand_q=np.zeros(7, dtype=np.float64),
        right_hand_q=np.zeros(7, dtype=np.float64),
    )
    assert h._compute_idle_root_quat_xyzw() is not None
    # And one more idle tick after recovery must NOT re-log.
    h._compute_idle_root_quat_xyzw()

    captured = capsys.readouterr()
    active_lines = [
        l for l in captured.out.splitlines()
        if "idle yaw-rebase: ACTIVE" in l
    ]
    fallback_lines = [
        l for l in captured.out.splitlines()
        if "idle yaw-rebase: x2_debug went stale" in l
    ]
    assert len(active_lines) == 2, (
        f"expected 2 ACTIVE lines (initial + recovery); got "
        f"{len(active_lines)}: {active_lines}"
    )
    assert len(fallback_lines) == 1, (
        f"expected 1 stale-fallback line on the active->stale "
        f"transition; got {len(fallback_lines)}: {fallback_lines}"
    )


# ---------------------------------------------------------------------------
# 4. _publish_idle glue: passes the helper's quat into _publish_pose
# ---------------------------------------------------------------------------


def test_publish_idle_forwards_yaw_rebased_quat() -> None:
    """When x2_debug is alive, _publish_idle calls _publish_pose with
    ``root_quat_xyzw=<helper output>`` and the rest of the wire shape
    unchanged (default stand body, zero motion_token, zero hands,
    tick=-1)."""
    yaw_rad = math.radians(60.0)
    h = _IdleHarness(latest_state=_state_with_yaw(yaw_rad))

    h._publish_idle()

    assert len(h.publish_calls) == 1
    kwargs = h.publish_calls[0]
    assert kwargs["tick"] == -1
    assert kwargs["root_quat_xyzw"] is not None
    qx, qy, qz, qw = kwargs["root_quat_xyzw"].tolist()
    half = 0.5 * yaw_rad
    assert qx == pytest.approx(0.0, abs=1e-6)
    assert qy == pytest.approx(0.0, abs=1e-6)
    assert qz == pytest.approx(math.sin(half), abs=1e-5)
    assert qw == pytest.approx(math.cos(half), abs=1e-5)


def test_publish_idle_falls_back_to_none_when_xdebug_silent() -> None:
    """When x2_debug never arrived, _publish_idle calls _publish_pose
    with ``root_quat_xyzw=None`` -- the deploy receives the recorder's
    pre-fix identity-quat wire shape verbatim (zero regression)."""
    h = _IdleHarness(latest_state=_LatestState())

    h._publish_idle()

    assert len(h.publish_calls) == 1
    kwargs = h.publish_calls[0]
    assert kwargs["tick"] == -1
    assert kwargs["root_quat_xyzw"] is None
