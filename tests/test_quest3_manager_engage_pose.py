"""Regression pins for the OFF->non-OFF engage-pose snap (2026-06-10
follow-up 10).

The default OFF->non-OFF behaviour snaps the manager's frozen arm + hand
caches to ``self._retargeter._teleop.{left,right}_neutral_q`` + zeros.
That's the right call when the operator engages teleop from a cold
start, but it's the *wrong* call mid-VLA-run: VLA was driving the arms
to pose X, the operator presses A+B+X+Y to take over, and the manager
publishes ``arm_targets`` at X2 neutral. The recorder forwards those
into ``pose``, the pose-proxy engages override, and the proxy's
engagement clamp (follow-up 9b) slowly walks the wire from X down to
neutral over ~5 seconds. The robot's arms drift to default; the
operator sees "I activated teleop and the arms went to a parking
pose".

Follow-up 10 wires an optional SUB to the proxy/bridge pose wire on
the manager. When configured AND a fresh frame is cached, the snap
target becomes the wire's current ``joint_pos_mj`` (sliced at
``[15:22]`` / ``[22:29]`` for left / right arms; same canonical MJ
layout the recorder uses). Hands stay at ``fingers=open`` unless
``--engage-preserve-hands`` is also set.

This file pins:

  * ``_resolve_engage_freeze`` returns the cached arm slices when the
    cache is fresh (``source="wire"``) and falls back to
    ``source="neutral"`` when stale / empty / SUB disabled.
  * The full SUB -> cache -> slice path: a real ZMQ PUB on loopback
    sends one ``pose`` frame, the manager's background thread decodes
    + caches it, and ``_resolve_engage_freeze`` returns the arms.
  * Hands ride along with arms whenever the snap-to-wire path fires
    (single-flag UX: the user opted into "preserve in place" and the
    consistent expectation is the whole upper body stays put). A
    per-side cache miss (missing ``*_hand_joints``) falls back to
    fingers-open just for that side, not the whole engage path.
  * Cache freshness is enforced via ``engage_pose_sub_max_age_ms``.

These tests do NOT boot the Quest 3 reader; the manager fixture stops
it before yielding. ZMQ ports use a high range (25600+) to avoid
collisions with concurrent runs of the wire-format test.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import zmq

from gear_sonic.scripts.quest3_manager_x2 import (
    ManagerConfig,
    Quest3ManagerX2,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message


PLANNER_CMD_PORT = 25593
RECORDER_PORT = 25594
ENGAGE_POSE_PORT = 25595


def _require_default_calibration() -> Path:
    cal = Path("data/operator_calibrations/default.yaml").resolve()
    if not cal.is_file():
        pytest.skip(f"requires {cal}")
    return cal


def _make_cfg(
    *, preserve_arms_on_engage: bool = False, sub_port: int | None = None,
) -> ManagerConfig:
    cfg = ManagerConfig(calibration_path=_require_default_calibration())
    cfg.planner_cmd_port = PLANNER_CMD_PORT
    cfg.recorder_pub_port = RECORDER_PORT
    cfg.recorder_pub_host = "127.0.0.1"
    cfg.planner_cmd_host = "127.0.0.1"
    cfg.preserve_arms_on_engage = preserve_arms_on_engage
    if sub_port is not None:
        cfg.engage_pose_sub_port = sub_port
    cfg.engage_pose_sub_host = "127.0.0.1"
    cfg.engage_pose_sub_topic = "pose"
    cfg.engage_pose_sub_max_age_ms = 200
    return cfg


def _make_pose_pub(port: int) -> zmq.Socket:
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.bind(f"tcp://127.0.0.1:{port}")
    # ZMQ PUB-SUB slow-joiner: give the manager's SUB a moment to
    # complete the handshake before the first send, else the message
    # is dropped on the publisher side.
    time.sleep(0.15)
    return pub


def _send_pose_frame(
    pub: zmq.Socket,
    jpos: np.ndarray,
    *,
    left_hand: np.ndarray | None = None,
    right_hand: np.ndarray | None = None,
    repeat: int = 5,
) -> None:
    fields: dict[str, np.ndarray] = {
        "joint_pos_mj": jpos.astype(np.float32),
    }
    if left_hand is not None:
        fields["left_hand_joints"] = left_hand.astype(np.float32)
    if right_hand is not None:
        fields["right_hand_joints"] = right_hand.astype(np.float32)
    payload = pack_pose_message(fields, topic="pose")
    for _ in range(repeat):
        pub.send(payload)
        time.sleep(0.02)


# ---------------------------------------------------------------------------
# _resolve_engage_freeze (unit-level, no ZMQ)
# ---------------------------------------------------------------------------


def test_resolve_engage_freeze_disabled_returns_neutral():
    """preserve_arms_on_engage=False MUST yield the neutral fallback
    (source='neutral'). Pin the legacy default so existing operators
    see zero behaviour change without opting in."""
    cfg = _make_cfg(preserve_arms_on_engage=False)
    mgr = Quest3ManagerX2(cfg)
    try:
        left, right, lhand, rhand, src = mgr._resolve_engage_freeze()
        assert src == "neutral"
        assert left is None and right is None
        assert lhand is None and rhand is None
    finally:
        mgr.stop()


def test_resolve_engage_freeze_stale_cache_falls_back_to_neutral():
    """A populated-but-old cache MUST be rejected (operator opted into
    'preserve current wire pose', not 'whatever was on the wire 5
    seconds ago when the manager last saw a frame')."""
    cfg = _make_cfg(preserve_arms_on_engage=True, sub_port=ENGAGE_POSE_PORT)
    cfg.engage_pose_sub_max_age_ms = 50  # 50ms tolerance
    mgr = Quest3ManagerX2(cfg)
    try:
        # Inject a stale cache entry directly (bypass the SUB thread).
        jpos = np.linspace(-0.1, 0.5, 31).astype(np.float64)
        with mgr._engage_pose_lock:
            mgr._engage_pose_jpos = jpos
            mgr._engage_pose_last_ts = time.time() - 1.0   # 1s old
            mgr._engage_pose_msg_count = 1
        left, right, lhand, rhand, src = mgr._resolve_engage_freeze()
        assert src == "neutral", "stale cache must fall back to neutral"
        assert left is None and right is None
    finally:
        mgr.stop()


def test_resolve_engage_freeze_fresh_cache_returns_arms_and_hands():
    """A fresh cache with a 31-dim jpos + populated hand fields MUST
    yield jpos[15:22] / jpos[22:29] as the left/right arm targets AND
    the cached hand joints as the left/right hand targets. Pins the
    single-flag UX (the user opted into 'preserve arms on engage' and
    the consistent expectation is the whole upper body stays put,
    including the fingers -- no surprise drop / squeeze)."""
    cfg = _make_cfg(preserve_arms_on_engage=True, sub_port=ENGAGE_POSE_PORT)
    mgr = Quest3ManagerX2(cfg)
    try:
        jpos = np.linspace(-0.1, 0.5, 31).astype(np.float64)
        left_hand = (0.3 * np.ones(10))
        right_hand = (0.4 * np.ones(10))
        with mgr._engage_pose_lock:
            mgr._engage_pose_jpos = jpos
            mgr._engage_pose_left_hand = left_hand
            mgr._engage_pose_right_hand = right_hand
            mgr._engage_pose_last_ts = time.time()
            mgr._engage_pose_msg_count = 1
        left, right, lhand, rhand, src = mgr._resolve_engage_freeze()
        assert src == "wire"
        np.testing.assert_allclose(left, jpos[15:22])
        np.testing.assert_allclose(right, jpos[22:29])
        np.testing.assert_allclose(lhand, left_hand)
        np.testing.assert_allclose(rhand, right_hand)
    finally:
        mgr.stop()


def test_resolve_engage_freeze_missing_hands_falls_back_per_side():
    """If the cached wire frame omitted ``*_hand_joints`` (e.g., an
    older bridge or a buggy upstream) the arms still come through but
    that side's hand is None -- the caller is expected to fill in
    fingers-open fallback for any None returned here. Pin both
    branches (left present + right missing, and vice versa) so a
    refactor can't silently start raising on partial-hand frames."""
    cfg = _make_cfg(preserve_arms_on_engage=True, sub_port=ENGAGE_POSE_PORT)
    mgr = Quest3ManagerX2(cfg)
    try:
        jpos = np.linspace(-0.1, 0.5, 31).astype(np.float64)
        left_hand = (0.2 * np.ones(10))
        with mgr._engage_pose_lock:
            mgr._engage_pose_jpos = jpos
            mgr._engage_pose_left_hand = left_hand
            mgr._engage_pose_right_hand = None
            mgr._engage_pose_last_ts = time.time()
            mgr._engage_pose_msg_count = 1
        left, right, lhand, rhand, src = mgr._resolve_engage_freeze()
        assert src == "wire"
        np.testing.assert_allclose(left, jpos[15:22])
        np.testing.assert_allclose(right, jpos[22:29])
        np.testing.assert_allclose(lhand, left_hand)
        assert rhand is None
    finally:
        mgr.stop()


def test_resolve_engage_freeze_short_jpos_falls_back_to_neutral():
    """If the cached jpos is shorter than 29 we can't slice the arms,
    so we MUST fall back rather than IndexError."""
    cfg = _make_cfg(preserve_arms_on_engage=True, sub_port=ENGAGE_POSE_PORT)
    mgr = Quest3ManagerX2(cfg)
    try:
        short_jpos = np.zeros(20, dtype=np.float64)
        with mgr._engage_pose_lock:
            mgr._engage_pose_jpos = short_jpos
            mgr._engage_pose_last_ts = time.time()
            mgr._engage_pose_msg_count = 1
        _, _, _, _, src = mgr._resolve_engage_freeze()
        assert src == "neutral"
    finally:
        mgr.stop()


# ---------------------------------------------------------------------------
# End-to-end SUB path (real ZMQ on loopback)
# ---------------------------------------------------------------------------


def test_engage_pose_sub_decodes_and_caches_real_zmq_frame():
    """Bind a real PUB on loopback, send one ``pose`` frame, and wait
    for the manager's background SUB thread to land it in the cache.
    Then verify _resolve_engage_freeze returns the right arm slices.
    """
    cfg = _make_cfg(preserve_arms_on_engage=True, sub_port=ENGAGE_POSE_PORT)
    pub = _make_pose_pub(ENGAGE_POSE_PORT)
    mgr = Quest3ManagerX2(cfg)
    try:
        jpos = np.linspace(-0.2, 0.6, 31).astype(np.float32)
        left_hand = (0.5 * np.ones(10)).astype(np.float32)
        right_hand = (0.6 * np.ones(10)).astype(np.float32)
        # Re-send for up to 2s while the SUB drains (lossy PUB-SUB
        # handshake; one send isn't enough on a cold socket).
        deadline = time.time() + 2.0
        got = False
        while time.time() < deadline and not got:
            _send_pose_frame(pub, jpos, left_hand=left_hand, right_hand=right_hand,
                             repeat=3)
            time.sleep(0.05)
            with mgr._engage_pose_lock:
                got = mgr._engage_pose_msg_count > 0
        assert got, "manager engage-pose SUB never received a frame"
        left, right, lhand, rhand, src = mgr._resolve_engage_freeze()
        assert src == "wire"
        np.testing.assert_allclose(left, jpos[15:22], atol=1e-5)
        np.testing.assert_allclose(right, jpos[22:29], atol=1e-5)
        # Hands ride along with arms now (single-flag UX); validate
        # the cached values made it through the SUB -> decode -> cache
        # -> resolve path end-to-end.
        np.testing.assert_allclose(lhand, left_hand, atol=1e-5)
        np.testing.assert_allclose(rhand, right_hand, atol=1e-5)
    finally:
        pub.close(linger=0)
        mgr.stop()
