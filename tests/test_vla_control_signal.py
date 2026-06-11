"""Unit tests for the live VLA bridge's manual-takeover signal.

The 2026-06-10 milestone wires a vla_control SUB into the bridge so
the proxy can drive a cold restart on operator release. The 2026-06-11
milestone moved the publisher from the PC2 proxy to the laptop-side
``x2_pose_mux`` (the bridge-side consumer contract is unchanged).
The actual SUB worker is gated on a running ZMQ peer (covered by the
dual-source smoke at tests/test_x2_pose_mux_dual_source.py) but the
``_VlaControlSignal`` state machine is pure-Python and trivially
testable here. These tests pin the engage/release/consume semantics
so a refactor of the publisher's cold-restart consumer can't silently
break the signal contract.

The signal class lives in
``gear_sonic/scripts/live_vla_publish_motion_token.py``; importing
that module loads a few hundred ms of optional deps (joblib, ZMQ,
warnings filters) but no GPU / model state -- safe in CI.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import path matches what main() uses: the module is run-as-script
# friendly so we import via its package path. ``importlib.util`` would
# work too but adds noise.
from gear_sonic.scripts.live_vla_publish_motion_token import (  # noqa: E402
    DEFAULT_HAND_DOF,
    NUM_BODY_DOFS,
    _VlaControlSignal,
)


# ===========================================================================
# Initial state
# ===========================================================================
def test_signal_starts_inactive_with_no_pending_restart() -> None:
    sig = _VlaControlSignal()
    override_active, cold_restart_pending = sig.snapshot()
    assert override_active is False
    assert cold_restart_pending is False
    engage_count, release_count, last_ts = sig.stats()
    assert engage_count == 0
    assert release_count == 0
    assert last_ts == -1.0


# ===========================================================================
# Engage / release edges
# ===========================================================================
def test_engage_sets_override_active_and_increments_count() -> None:
    sig = _VlaControlSignal()
    sig.engage(ts=12.5)
    override_active, cold_restart_pending = sig.snapshot()
    assert override_active is True
    # Engaging alone does NOT arm a cold restart -- the cold restart
    # is what the publisher does on RELEASE so the wire can hand back
    # cleanly. Engaging only suppresses chunk publication during the
    # takeover window.
    assert cold_restart_pending is False
    engage_count, release_count, last_ts = sig.stats()
    assert engage_count == 1
    assert release_count == 0
    assert last_ts == pytest.approx(12.5)


def test_release_clears_override_and_arms_cold_restart() -> None:
    sig = _VlaControlSignal()
    sig.engage(ts=10.0)
    sig.release(ts=11.25)
    override_active, cold_restart_pending = sig.snapshot()
    assert override_active is False
    assert cold_restart_pending is True
    engage_count, release_count, last_ts = sig.stats()
    assert engage_count == 1
    assert release_count == 1
    assert last_ts == pytest.approx(11.25)


def test_consume_cold_restart_is_one_shot() -> None:
    """The publisher should see exactly ONE cold-restart pending per
    release edge; subsequent ticks without a new release see False.
    Critical so the publisher doesn't repeatedly clobber its smoothing
    state every tick after a release."""
    sig = _VlaControlSignal()
    sig.engage(ts=0.0)
    sig.release(ts=1.0)
    pending, release_pose = sig.consume_cold_restart()
    assert pending is True
    # Legacy release (no release_pose arg) means the bridge falls
    # back to measured-pose hold for the cold-restart window.
    assert release_pose is None
    # After consumption the flag clears.
    pending, release_pose = sig.consume_cold_restart()
    assert pending is False
    assert release_pose is None
    pending, release_pose = sig.consume_cold_restart()
    assert pending is False
    assert release_pose is None
    # snapshot() should also report the cleared flag.
    _, cold_restart_pending = sig.snapshot()
    assert cold_restart_pending is False


def test_repeated_engage_without_release_does_not_arm_restart() -> None:
    """Multiple engage edges (rare in practice -- the proxy emits one
    engage per OVERRIDE entry -- but defensive): the override-active
    flag stays True, the count increments, but no cold restart fires
    until the operator actually releases."""
    sig = _VlaControlSignal()
    sig.engage(ts=1.0)
    sig.engage(ts=2.0)
    sig.engage(ts=3.0)
    override_active, cold_restart_pending = sig.snapshot()
    assert override_active is True
    assert cold_restart_pending is False
    engage_count, release_count, _ = sig.stats()
    assert engage_count == 3
    assert release_count == 0


def test_release_without_prior_engage_still_arms_restart() -> None:
    """Edge case: the SUB worker may miss an engage event (PUB/SUB
    starts and drops the very first frame). If we then see a release,
    we should still arm a cold restart so the bridge re-baselines --
    safer to over-restart than to keep streaming stale chunks."""
    sig = _VlaControlSignal()
    sig.release(ts=5.0)
    override_active, cold_restart_pending = sig.snapshot()
    assert override_active is False
    assert cold_restart_pending is True


# ===========================================================================
# Thread safety
# ===========================================================================
def test_concurrent_engage_release_consume_is_safe() -> None:
    """Hammer the signal from multiple threads simultaneously to
    surface any latent race in the lock-protected updates. We only
    assert that nothing crashes and that counts are internally
    consistent (engages + releases >= 1 each); the actual ordering
    is non-deterministic by design (proxy + publisher run in
    parallel)."""
    sig = _VlaControlSignal()
    n_iters = 2000

    def hammer_engage() -> None:
        for i in range(n_iters):
            sig.engage(ts=float(i))

    def hammer_release() -> None:
        for i in range(n_iters):
            sig.release(ts=float(i))

    def hammer_consume() -> None:
        for _ in range(n_iters):
            sig.consume_cold_restart()  # tuple return, discard
            sig.snapshot()
            sig.stats()

    threads = [
        threading.Thread(target=hammer_engage, name="engage"),
        threading.Thread(target=hammer_release, name="release"),
        threading.Thread(target=hammer_consume, name="consume"),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
        assert not t.is_alive(), f"{t.name} thread hung"

    engage_count, release_count, _ = sig.stats()
    assert engage_count == n_iters
    assert release_count == n_iters


# ===========================================================================
# Operator-pose handoff (2026-06-10 follow-up)
# ===========================================================================
def test_release_pose_is_stored_and_returned_by_consume() -> None:
    """release(release_pose=...) must round-trip through consume.

    The bridge's cold-restart path reads the release pose alongside
    the pending flag so it can hold the wire at the operator's last
    commanded pose (body + hands) for the bridging window. Without
    this round-trip the bridge falls back to x2_debug's lagged
    measured pose and the operator sees a visible "pose reset"
    on ARM_MANIPULATION -> LOCOMOTION handoff.
    """
    sig = _VlaControlSignal()
    body = np.full(NUM_BODY_DOFS, 0.42, dtype=np.float32)
    left = np.full(DEFAULT_HAND_DOF, 0.61, dtype=np.float32)
    right = np.full(DEFAULT_HAND_DOF, 0.73, dtype=np.float32)
    pose = {
        "joint_pos_mj": body,
        "left_hand_joints": left,
        "right_hand_joints": right,
    }
    sig.engage(ts=10.0)
    sig.release(ts=11.0, release_pose=pose)

    pending, returned = sig.consume_cold_restart()
    assert pending is True
    assert returned is not None
    # Round-trip identity check on all three fields. We compare by
    # value (not by reference) so the signal is free to defensively
    # copy in a future hardening pass.
    np.testing.assert_array_equal(
        returned["joint_pos_mj"], body
    )
    np.testing.assert_array_equal(
        returned["left_hand_joints"], left
    )
    np.testing.assert_array_equal(
        returned["right_hand_joints"], right
    )


def test_release_pose_clears_on_consume_so_next_release_starts_fresh() -> None:
    """Each release event must own its own pose snapshot.

    If consume left the cached pose around, a SECOND release that
    arrived from an older proxy (no payload) would incorrectly
    reuse the previous release's pose, holding the wire at a stale
    operator pose during the new cold-restart. Cleanup on consume
    is what makes this a one-shot like ``cold_restart_pending``.
    """
    sig = _VlaControlSignal()
    body = np.full(NUM_BODY_DOFS, 0.50, dtype=np.float32)
    sig.engage(ts=1.0)
    sig.release(ts=2.0, release_pose={"joint_pos_mj": body})
    # First consume: pose present.
    pending1, pose1 = sig.consume_cold_restart()
    assert pending1 is True
    assert pose1 is not None and "joint_pos_mj" in pose1
    # Second consume (no intervening release): pose cleared.
    pending2, pose2 = sig.consume_cold_restart()
    assert pending2 is False
    assert pose2 is None
    # Third event: release without payload (older proxy / legacy
    # smoke test). Must NOT replay the previous snapshot.
    sig.release(ts=3.0)
    pending3, pose3 = sig.consume_cold_restart()
    assert pending3 is True
    assert pose3 is None, (
        f"second release had no payload; consume must return None, "
        f"not the stale pose from the first release; got {pose3!r}"
    )


def test_release_pose_partial_fields_are_preserved() -> None:
    """release_pose with only a subset of fields (e.g. body but no
    hands) must round-trip through consume without filling in the
    missing keys. The bridge inspects each key individually and
    falls back to legacy behaviour per-field when absent."""
    sig = _VlaControlSignal()
    body = np.full(NUM_BODY_DOFS, 0.33, dtype=np.float32)
    sig.engage(ts=0.0)
    # Only body, no hands.
    sig.release(ts=1.0, release_pose={"joint_pos_mj": body})
    _, returned = sig.consume_cold_restart()
    assert returned is not None
    assert "joint_pos_mj" in returned
    assert "left_hand_joints" not in returned
    assert "right_hand_joints" not in returned
