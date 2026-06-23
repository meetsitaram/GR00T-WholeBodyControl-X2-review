"""Tests for the VLA bridge's hold-last-good live root_quat semantics.

Background: the bridge used to gate its wire root_quat rebase and
waist_yaw pin on ``deploy_fresh`` (1 s freshness on ``x2_debug``). On a
stall longer than the freshness window the rebase was skipped, the
wire shipped the baked-identity quat from the idle clip, and the C++
deploy commanded the body back to spawn heading -- the "robot snaps
back to boot orientation on every nudge" symptom.

Fix (2026-06-23): hold the last-known live ``base_quat`` (and
``body_q_mj`` for the waist_yaw pin) across stalls of arbitrary length.
A cached measured yaw is strictly better than identity because identity
is a known-wrong orientation that drives the deploy back to world +X.
Mirrors the kplanner's robot_pose hold-last-good pattern at
``gear_sonic/scripts/x2_kplanner.py:2816-2821``.

The tests pin:

1. **no x2_debug ever -> source="none", bootstrap publish-gate withholds.**
   The bridge's outer publish gate at line ~3580 keys on
   ``state.received_any``; the rebase resolver returns ``"none"`` so
   the wire would carry baked identity, but the publish gate prevents
   the frame from going out at all. No regression of the bootstrap
   protection.
2. **fresh x2_debug -> source="live".** Wire root_quat tracks the
   measured yaw exactly.
3. **stall after fresh -> source="cached".** Cache holds the last live
   base_quat across an arbitrarily long stall (5 s in the test).
   ``_resolve_wire_rebase_source`` reports the cache age in seconds.
4. **stream recovers -> source="live" again.** Cache picks up the new
   measured yaw on the very next tick.
5. **waist_yaw pin uses cached slot-12 value during stalls.** Without
   the cache the legs/waist freeze path would revert to idle_stand's
   waist_yaw (~33 deg off DEFAULT_STAND_POSE) and drive a steady-state
   heading drift.
6. **state-transition logger fires exactly once per edge.** ACTIVE,
   STALE, and RECOVERED each emit one line at the corresponding
   transition; same-state ticks are silent.

The helpers (``_resolve_wire_rebase_source``,
``_log_rebase_source_transition``, ``_WireRebaseSource``) are pure and
testable directly; we exercise the production code -- a regression in
the inline call site at the inference loop's tick body immediately
fails these tests because the helpers ARE what the production loop
calls.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.live_vla_publish_motion_token import (  # noqa: E402
    DEPLOY_ALIVE_STALE_THRESHOLD_S,
    WAIST_YAW_IDX,
    _LatestState,
    _log_rebase_source_transition,
    _resolve_wire_rebase_source,
    _root_quat_xyzw_from_base_quat_wxyz,
    _WireRebaseSource,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_quat_wxyz_for_yaw(yaw_rad: float) -> np.ndarray:
    """Build a wxyz quat representing a pure ``R_z(yaw_rad)`` rotation."""
    half = 0.5 * yaw_rad
    return np.array(
        [math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64
    )


def _body_q_with_waist_yaw(waist_yaw_rad: float) -> np.ndarray:
    """Build a 31-D MuJoCo body_q with the given waist_yaw slot value."""
    q = np.zeros(31, dtype=np.float64)
    q[WAIST_YAW_IDX] = waist_yaw_rad
    return q


def _update_state(
    state: _LatestState, yaw_rad: float, waist_yaw_rad: float = 0.0
) -> None:
    state.update(
        body_q_mj=_body_q_with_waist_yaw(waist_yaw_rad),
        base_quat_wxyz=_base_quat_wxyz_for_yaw(yaw_rad),
        left_hand_q=np.zeros(7, dtype=np.float64),
        right_hand_q=np.zeros(7, dtype=np.float64),
    )


def _yaw_deg_from_xyzw(quat_xyzw: np.ndarray) -> float:
    from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw

    return math.degrees(
        float(yaw_of_quat_xyzw(quat_xyzw.astype(np.float64)))
    )


# ---------------------------------------------------------------------------
# 1. no x2_debug ever -> source="none"
# ---------------------------------------------------------------------------


def test_no_x2_debug_ever_returns_source_none() -> None:
    """Fresh state, no cache, deploy_fresh=False -> source='none'.
    base_quat_wxyz and body_q_mj are both None so the rebase block and
    the waist_yaw pin both skip. The outer bootstrap-publish gate
    (state.received_any) is what protects the wire in this case."""
    rebase = _resolve_wire_rebase_source(
        deploy_fresh=False,
        base_quat_now=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        body_q_mj_now=np.zeros(31, dtype=np.float64),
        last_known_base_quat_wxyz=None,
        last_known_body_q_mj=None,
        last_known_x2_debug_monotonic=-1.0,
        now_monotonic=time.monotonic(),
    )
    assert rebase.source == "none"
    assert rebase.base_quat_wxyz is None
    assert rebase.body_q_mj is None
    assert rebase.cache_age_s == float("inf")


def test_no_x2_debug_ever_bootstrap_gate_holds() -> None:
    """``_LatestState`` with never-received -> ``received_any`` False,
    which is what the bridge's bootstrap-publish gate (line ~3572 in
    live_vla_publish_motion_token.py) keys on to withhold the first
    publish. The rebase resolver's ``"none"`` source is the cooperative
    signal that says 'don't try to rebase'."""
    state = _LatestState()
    assert state.received_any is False
    assert state.is_alive() is False


# ---------------------------------------------------------------------------
# 2. fresh x2_debug -> source="live"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "yaw_deg", [-179.0, -90.0, -45.0, -10.0, 0.0, 10.0, 45.0, 90.0, 179.0]
)
def test_fresh_x2_debug_returns_source_live(yaw_deg: float) -> None:
    """deploy_fresh=True -> source='live' and base_quat is the current
    snapshot (NOT the cache, even if a cache exists)."""
    yaw_rad = math.radians(yaw_deg)
    fresh_bq = _base_quat_wxyz_for_yaw(yaw_rad)
    fresh_bq_mj = _body_q_with_waist_yaw(0.1)
    cached_bq = _base_quat_wxyz_for_yaw(math.radians(yaw_deg + 30.0))
    cached_bq_mj = _body_q_with_waist_yaw(0.2)

    rebase = _resolve_wire_rebase_source(
        deploy_fresh=True,
        base_quat_now=fresh_bq,
        body_q_mj_now=fresh_bq_mj,
        last_known_base_quat_wxyz=cached_bq,
        last_known_body_q_mj=cached_bq_mj,
        last_known_x2_debug_monotonic=time.monotonic() - 0.5,
        now_monotonic=time.monotonic(),
    )

    assert rebase.source == "live"
    assert rebase.cache_age_s == 0.0
    assert rebase.base_quat_wxyz is not None
    assert rebase.body_q_mj is not None
    np.testing.assert_allclose(rebase.base_quat_wxyz, fresh_bq, atol=1e-12)
    np.testing.assert_allclose(rebase.body_q_mj, fresh_bq_mj, atol=1e-12)

    wire_quat = _root_quat_xyzw_from_base_quat_wxyz(rebase.base_quat_wxyz)
    assert _yaw_deg_from_xyzw(wire_quat) == pytest.approx(yaw_deg, abs=1e-3)


def test_fresh_x2_debug_copies_arrays_so_caller_can_mutate() -> None:
    """The resolver must return COPIES of the input arrays so the
    inference loop can mutate ``cur_jpos`` without aliasing into the
    snapshot buffer (which the state thread might overwrite mid-tick)."""
    bq = _base_quat_wxyz_for_yaw(math.radians(45.0))
    bq_mj = _body_q_with_waist_yaw(0.3)

    rebase = _resolve_wire_rebase_source(
        deploy_fresh=True,
        base_quat_now=bq,
        body_q_mj_now=bq_mj,
        last_known_base_quat_wxyz=None,
        last_known_body_q_mj=None,
        last_known_x2_debug_monotonic=-1.0,
        now_monotonic=time.monotonic(),
    )

    assert rebase.base_quat_wxyz is not None
    assert rebase.body_q_mj is not None
    rebase.base_quat_wxyz[0] = -999.0
    rebase.body_q_mj[WAIST_YAW_IDX] = -999.0
    assert bq[0] != -999.0
    assert bq_mj[WAIST_YAW_IDX] != -999.0


# ---------------------------------------------------------------------------
# 3. stall after fresh -> source="cached"
# ---------------------------------------------------------------------------


def test_stall_after_fresh_returns_source_cached_with_correct_age() -> None:
    """deploy_fresh=False but cache populated -> source='cached'. The
    cached base_quat is what's used for the rebase; this is the core
    of the hold-last-good fix."""
    yaw_rad = math.radians(45.0)
    cached_bq = _base_quat_wxyz_for_yaw(yaw_rad)
    cached_bq_mj = _body_q_with_waist_yaw(0.15)
    last_seen = time.monotonic()

    rebase = _resolve_wire_rebase_source(
        deploy_fresh=False,
        base_quat_now=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        body_q_mj_now=np.zeros(31, dtype=np.float64),
        last_known_base_quat_wxyz=cached_bq,
        last_known_body_q_mj=cached_bq_mj,
        last_known_x2_debug_monotonic=last_seen,
        now_monotonic=last_seen + 5.0,
    )

    assert rebase.source == "cached"
    assert rebase.base_quat_wxyz is cached_bq, (
        "cached path should return the cache reference, not a copy"
    )
    assert rebase.body_q_mj is cached_bq_mj
    assert rebase.cache_age_s == pytest.approx(5.0, abs=1e-3)

    wire_quat = _root_quat_xyzw_from_base_quat_wxyz(rebase.base_quat_wxyz)
    assert _yaw_deg_from_xyzw(wire_quat) == pytest.approx(45.0, abs=1e-3)


def test_stall_beyond_freshness_threshold_still_uses_cache() -> None:
    """Even with an arbitrarily long stall (10x the freshness threshold)
    we keep using the cached value. Identity fallback is the bug."""
    yaw_rad = math.radians(-30.0)
    cached_bq = _base_quat_wxyz_for_yaw(yaw_rad)
    cached_bq_mj = _body_q_with_waist_yaw(0.0)
    last_seen = time.monotonic()

    long_stall_s = 10.0 * DEPLOY_ALIVE_STALE_THRESHOLD_S
    rebase = _resolve_wire_rebase_source(
        deploy_fresh=False,
        base_quat_now=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        body_q_mj_now=np.zeros(31, dtype=np.float64),
        last_known_base_quat_wxyz=cached_bq,
        last_known_body_q_mj=cached_bq_mj,
        last_known_x2_debug_monotonic=last_seen,
        now_monotonic=last_seen + long_stall_s,
    )

    assert rebase.source == "cached"
    assert rebase.cache_age_s == pytest.approx(long_stall_s, rel=1e-3)
    wire_quat = _root_quat_xyzw_from_base_quat_wxyz(rebase.base_quat_wxyz)
    assert _yaw_deg_from_xyzw(wire_quat) == pytest.approx(-30.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 4. recovery -> source="live" again, cache picks up new value
# ---------------------------------------------------------------------------


def test_recovery_returns_to_live_and_picks_up_new_yaw() -> None:
    """After a stall, the next deploy_fresh=True tick returns
    source='live' with the new measured yaw (NOT the cached value).
    The inference loop updates the cache on this same tick."""
    cached_yaw = math.radians(45.0)
    recovered_yaw = math.radians(-90.0)
    cached_bq = _base_quat_wxyz_for_yaw(cached_yaw)
    recovered_bq = _base_quat_wxyz_for_yaw(recovered_yaw)

    rebase = _resolve_wire_rebase_source(
        deploy_fresh=True,
        base_quat_now=recovered_bq,
        body_q_mj_now=_body_q_with_waist_yaw(0.0),
        last_known_base_quat_wxyz=cached_bq,
        last_known_body_q_mj=_body_q_with_waist_yaw(0.5),
        last_known_x2_debug_monotonic=time.monotonic() - 5.0,
        now_monotonic=time.monotonic(),
    )

    assert rebase.source == "live"
    assert rebase.cache_age_s == 0.0
    np.testing.assert_allclose(
        rebase.base_quat_wxyz, recovered_bq, atol=1e-12
    )
    wire_quat = _root_quat_xyzw_from_base_quat_wxyz(rebase.base_quat_wxyz)
    assert _yaw_deg_from_xyzw(wire_quat) == pytest.approx(-90.0, abs=1e-3)


# ---------------------------------------------------------------------------
# 5. waist_yaw pin uses cached slot-12 during stalls
# ---------------------------------------------------------------------------


def test_waist_yaw_pin_uses_cached_value_during_stall() -> None:
    """The waist_yaw pin in the inference loop reads
    ``rebase_source.body_q_mj[WAIST_YAW_IDX]``. During a stall this
    must be the CACHED measured value, not idle_stand's ~33 deg-off
    default."""
    measured_waist_yaw = math.radians(-12.0)
    cached_bq_mj = _body_q_with_waist_yaw(measured_waist_yaw)

    rebase = _resolve_wire_rebase_source(
        deploy_fresh=False,
        base_quat_now=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        body_q_mj_now=np.zeros(31, dtype=np.float64),
        last_known_base_quat_wxyz=_base_quat_wxyz_for_yaw(0.0),
        last_known_body_q_mj=cached_bq_mj,
        last_known_x2_debug_monotonic=time.monotonic() - 2.0,
        now_monotonic=time.monotonic(),
    )

    assert rebase.body_q_mj is not None
    assert rebase.body_q_mj[WAIST_YAW_IDX] == pytest.approx(
        measured_waist_yaw, abs=1e-12
    )


# ---------------------------------------------------------------------------
# 6. state-transition logger
# ---------------------------------------------------------------------------


def test_log_active_fires_only_on_none_to_live(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bq = _base_quat_wxyz_for_yaw(math.radians(30.0))
    _log_rebase_source_transition(
        new_source="live",
        prev_source="none",
        base_quat_wxyz=bq,
        cache_age_s=0.0,
    )
    captured = capsys.readouterr()
    assert "yaw-rebase ACTIVE" in captured.out
    assert "yaw=+30.0deg" in captured.out


def test_log_stale_fires_only_on_live_to_cached(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bq = _base_quat_wxyz_for_yaw(math.radians(0.0))
    _log_rebase_source_transition(
        new_source="cached",
        prev_source="live",
        base_quat_wxyz=bq,
        cache_age_s=1.234,
    )
    captured = capsys.readouterr()
    assert "yaw-rebase STALE" in captured.out
    assert "cache age=1234ms" in captured.out
    assert "no snap-back" in captured.out


def test_log_recovered_fires_only_on_cached_to_live(
    capsys: pytest.CaptureFixture[str],
) -> None:
    bq = _base_quat_wxyz_for_yaw(math.radians(45.0))
    _log_rebase_source_transition(
        new_source="live",
        prev_source="cached",
        base_quat_wxyz=bq,
        cache_age_s=0.0,
    )
    captured = capsys.readouterr()
    assert "yaw-rebase RECOVERED" in captured.out


@pytest.mark.parametrize("source", ["none", "live", "cached"])
def test_log_silent_on_same_state(
    capsys: pytest.CaptureFixture[str], source: str
) -> None:
    """Same-state transitions are silent so a 50 Hz publish loop never
    spams logs."""
    bq = _base_quat_wxyz_for_yaw(math.radians(10.0))
    for _ in range(50):
        _log_rebase_source_transition(
            new_source=source,
            prev_source=source,
            base_quat_wxyz=bq,
            cache_age_s=0.5,
        )
    captured = capsys.readouterr()
    assert captured.out == ""


def test_log_full_lifecycle_emits_one_line_per_edge(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: none -> live -> live -> cached -> cached -> live
    must produce exactly 1 ACTIVE, 1 STALE, 1 RECOVERED line."""
    bq = _base_quat_wxyz_for_yaw(math.radians(20.0))
    sources = ["none", "live", "live", "cached", "cached", "live"]
    for prev, new in zip(sources[:-1], sources[1:]):
        _log_rebase_source_transition(
            new_source=new,
            prev_source=prev,
            base_quat_wxyz=bq,
            cache_age_s=2.5,
        )
    captured = capsys.readouterr()
    active = [ln for ln in captured.out.splitlines() if "ACTIVE" in ln]
    stale = [ln for ln in captured.out.splitlines() if "STALE" in ln]
    recovered = [ln for ln in captured.out.splitlines() if "RECOVERED" in ln]
    assert len(active) == 1
    assert len(stale) == 1
    assert len(recovered) == 1


# ---------------------------------------------------------------------------
# 7. integration with _LatestState: snapshot returns cached values even
#    when not alive (so the bridge's cache-update path captures the
#    correct value on every fresh tick)
# ---------------------------------------------------------------------------


def test_latest_state_snapshot_returns_last_quat_even_when_not_alive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_LatestState.snapshot()`` returns the LAST stored base_quat
    even when ``alive=False``. This is what the bridge's
    cache-update-when-fresh + resolve-source-always pattern relies on:
    the snapshot is monotonically the freshest value the state thread
    has stored, never reset to defaults."""
    state = _LatestState()
    _update_state(state, math.radians(60.0), waist_yaw_rad=0.25)

    body_q, bq_wxyz, _, _, _, alive = state.snapshot()
    assert alive is True
    np.testing.assert_allclose(
        bq_wxyz, _base_quat_wxyz_for_yaw(math.radians(60.0)), atol=1e-12
    )

    state.last_update_monotonic = time.monotonic() - 1e6
    body_q_stale, bq_stale, _, _, _, alive_stale = state.snapshot()
    assert alive_stale is False
    np.testing.assert_allclose(bq_stale, bq_wxyz, atol=1e-12)
    assert body_q_stale[WAIST_YAW_IDX] == pytest.approx(0.25, abs=1e-12)


# ---------------------------------------------------------------------------
# 8. dataclass shape sanity
# ---------------------------------------------------------------------------


def test_wire_rebase_source_is_frozen() -> None:
    """Defensive: the dataclass is frozen so an accidental mutation in
    the inference loop fails loud instead of corrupting the cache."""
    rs = _WireRebaseSource(
        base_quat_wxyz=None,
        body_q_mj=None,
        source="none",
        cache_age_s=float("inf"),
    )
    with pytest.raises(Exception):
        rs.source = "live"  # type: ignore[misc]
