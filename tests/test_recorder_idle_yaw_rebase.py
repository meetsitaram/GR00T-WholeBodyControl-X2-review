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
2. **Never-received x2_debug -> identity (bootstrap).** The deploy
   hasn't booted yet / the SUB is still warming up; we have no
   measured value to hold, so the wire falls back to identity.
3. **Stale x2_debug AFTER receiving at least one packet -> cached
   R_z(yaw) on the wire (hold-last-good).** Mirrors the VLA bridge
   fix from 2026-06-23 (commit ``8eb3279``): a stale cached yaw is
   strictly better than identity, because identity = world +X = a
   known-wrong orientation the policy would actively twist toward.
   Fixes the direct-PKL-stack regression where wifi jitter to PC2
   flipped the wire to identity every >1 s gap.
4. **One-shot log gating.** The "ACTIVE" line fires once on the
   first successful rebase, the "live -> CACHED" line fires once on
   the next stale tick, and 50 Hz of repeated activity does NOT
   spam logs.
5. **Round-trip recovery logs.** If x2_debug goes stale and comes
   back, the next ACTIVE line fires again (so the operator sees the
   recovery).
6. **`_publish_idle` glue.** When the helper returns a non-None quat,
   `_publish_pose` is called with that exact ``root_quat_xyzw=`` kwarg;
   when None (bootstrap only), the kwarg is None and `_publish_pose`
   falls back to identity internally.
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
# 2. bootstrap vs stale: hold-last-good split
# ---------------------------------------------------------------------------


def test_never_received_xdebug_returns_none() -> None:
    """Bootstrap: fresh _LatestState (received_any=False) -> helper
    returns None so _publish_pose falls back to identity. There's no
    measured value to hold yet, and the deploy's own bootstrap-safe
    quat override carries the orientation reference until the SUB
    warms up."""
    h = _IdleHarness(latest_state=_LatestState())
    quat = h._compute_idle_root_quat_xyzw()
    assert quat is None


def test_stale_xdebug_holds_last_good_yaw(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hold-last-good (2026-06-24): once x2_debug has produced at
    least one frame, the helper keeps deriving root_quat from the
    cached base_quat across stalls of arbitrary length rather than
    reverting to identity. Mirrors the VLA bridge's
    ``_resolve_wire_rebase_source`` cached branch (commit ``8eb3279``).

    This is the fix that unblocks the direct-PKL stack against PC2
    over wifi: sub-second wifi jitter routinely flips is_alive False
    and the pre-fix helper used to publish identity = world +X every
    such gap, which the SONIC policy then tried to twist the body
    toward.
    """
    yaw_rad = math.radians(45.0)
    state = _state_with_yaw(yaw_rad)
    # Force the stored last_update_monotonic into the deep past so
    # is_alive flips False. received_any stays True (we DID receive
    # one packet) so the hold-last-good path kicks in.
    state.last_update_monotonic = time.monotonic() - 1e6
    assert state.received_any, "preconditions: we must have received once"

    h = _IdleHarness(latest_state=state)
    quat = h._compute_idle_root_quat_xyzw()

    assert quat is not None, (
        "stale but received-once must hold-last-good, not fall back "
        "to identity (= world +X = known-wrong orientation that the "
        "policy will actively twist the body toward)"
    )
    qx, qy, qz, qw = quat.tolist()
    half = 0.5 * yaw_rad
    assert qx == pytest.approx(0.0, abs=1e-6)
    assert qy == pytest.approx(0.0, abs=1e-6)
    assert qz == pytest.approx(math.sin(half), abs=1e-5), (
        "cached yaw must equal the last-received yaw bit-for-bit"
    )
    assert qw == pytest.approx(math.cos(half), abs=1e-5)


def test_stale_after_never_received_still_returns_none() -> None:
    """Combined gate: a default _LatestState (never received) plus a
    deep-past last_update_monotonic is still bootstrap, NOT
    hold-last-good. The cached base_quat_wxyz on a never-updated
    state is the identity default; we must NOT publish that as if
    it were a real measured yaw."""
    state = _LatestState()
    state.last_update_monotonic = time.monotonic() - 1e6
    assert not state.received_any, (
        "preconditions: never received, so hold-last-good must be off"
    )

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


def test_log_round_trip_active_then_cached_then_active(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A realistic operator-visible sequence: rebase comes up,
    wifi stalls (deploy still alive, x2_debug just hasn't reached us
    in >1 s), x2_debug comes back. We want exactly:
      * 1 ACTIVE line on first success
      * 1 "live -> CACHED" line on the next stale tick (the wire
        keeps publishing the last measured yaw, not identity)
      * 1 ACTIVE line again on recovery
    """
    yaw_rad = math.radians(20.0)
    state = _state_with_yaw(yaw_rad)
    h = _IdleHarness(latest_state=state)

    # Phase 1: first ACTIVE.
    quat_active = h._compute_idle_root_quat_xyzw()
    assert quat_active is not None

    # Phase 2: force stale -> live->CACHED line fires; the helper
    # MUST keep publishing the last measured yaw (not None / not
    # identity), so the cached quat byte-equals the previous tick.
    state.last_update_monotonic = time.monotonic() - 1e6
    quat_cached = h._compute_idle_root_quat_xyzw()
    assert quat_cached is not None, (
        "stale-after-received must hold-last-good, not return None"
    )
    np.testing.assert_array_equal(
        quat_cached, quat_active,
        err_msg="cached tick must publish exactly the last live yaw",
    )
    # Subsequent stale ticks must NOT re-log (and must still hold).
    for _ in range(10):
        q = h._compute_idle_root_quat_xyzw()
        assert q is not None
        np.testing.assert_array_equal(q, quat_active)

    # Phase 3: recovery -- bump update + re-mark alive (new yaw
    # value so we can prove the live branch took over).
    new_yaw_rad = math.radians(-30.0)
    half = 0.5 * new_yaw_rad
    state.update(
        body_q_mj=np.zeros(31, dtype=np.float64),
        base_quat_wxyz=np.array(
            [math.cos(half), 0.0, 0.0, math.sin(half)],
            dtype=np.float64,
        ),
        left_hand_q=np.zeros(7, dtype=np.float64),
        right_hand_q=np.zeros(7, dtype=np.float64),
    )
    quat_recovery = h._compute_idle_root_quat_xyzw()
    assert quat_recovery is not None
    # Recovery yaw should follow the new live measurement, not the
    # stale cache (sanity that the live branch is actually firing).
    assert quat_recovery[2] == pytest.approx(math.sin(half), abs=1e-5)
    assert quat_recovery[3] == pytest.approx(math.cos(half), abs=1e-5)
    # And one more idle tick after recovery must NOT re-log.
    h._compute_idle_root_quat_xyzw()

    captured = capsys.readouterr()
    active_lines = [
        l for l in captured.out.splitlines()
        if "idle yaw-rebase: ACTIVE" in l
    ]
    cached_lines = [
        l for l in captured.out.splitlines()
        if "idle yaw-rebase: live -> CACHED" in l
    ]
    assert len(active_lines) == 2, (
        f"expected 2 ACTIVE lines (initial + recovery); got "
        f"{len(active_lines)}: {active_lines}"
    )
    assert len(cached_lines) == 1, (
        f"expected 1 live->CACHED line on the active->stale "
        f"transition; got {len(cached_lines)}: {cached_lines}"
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


def test_publish_idle_falls_back_to_none_when_xdebug_never_received() -> None:
    """Bootstrap case (received_any=False): _publish_idle calls
    _publish_pose with ``root_quat_xyzw=None`` -- the deploy receives
    identity, matching the pre-fix wire shape verbatim. The hold-
    last-good branch only activates after the SUB has produced at
    least one packet."""
    h = _IdleHarness(latest_state=_LatestState())

    h._publish_idle()

    assert len(h.publish_calls) == 1
    kwargs = h.publish_calls[0]
    assert kwargs["tick"] == -1
    assert kwargs["root_quat_xyzw"] is None


def test_publish_idle_forwards_cached_quat_when_xdebug_stalls() -> None:
    """Hold-last-good through ``_publish_idle``: after one live tick
    the source can stall arbitrarily long and ``_publish_idle`` keeps
    forwarding the cached R_z(yaw) quat. This is the wire shape the
    direct-PKL stack relies on so PC2 wifi jitter doesn't twist the
    body back to world +X every >1 s gap."""
    yaw_rad = math.radians(75.0)
    state = _state_with_yaw(yaw_rad)
    h = _IdleHarness(latest_state=state)

    # Tick 1: live -> caches the ACTIVE log + drives publish.
    h._publish_idle()

    # Force stale and emit another idle frame.
    state.last_update_monotonic = time.monotonic() - 1e6
    h._publish_idle()

    assert len(h.publish_calls) == 2
    stale_kwargs = h.publish_calls[1]
    assert stale_kwargs["tick"] == -1
    assert stale_kwargs["root_quat_xyzw"] is not None, (
        "stale-after-received must publish a real quat, not None"
    )
    qx, qy, qz, qw = stale_kwargs["root_quat_xyzw"].tolist()
    half = 0.5 * yaw_rad
    assert qx == pytest.approx(0.0, abs=1e-6)
    assert qy == pytest.approx(0.0, abs=1e-6)
    assert qz == pytest.approx(math.sin(half), abs=1e-5)
    assert qw == pytest.approx(math.cos(half), abs=1e-5)
