"""Unit tests for the x2_kplanner intent -> velocity dispatcher.

Lightweight (no torch / no checkpoints / no ZMQ); just imports
``gear_sonic.scripts.x2_kplanner`` and exercises ``intent_to_velocity``
+ ``_resolve_velocity`` against the full manager vocabulary. Lives
separately from ``test_x2_kplanner_zmq_publish.py`` so it can run on
machines without the MotionBricks training stack installed.

Three invariants covered:

1. **Parity** -- every (intent, magnitude) the legacy flat-table version
   of ``INTENT_VELOCITY_MAP`` mapped pre-refactor must still resolve to
   the same 4-tuple value. This is the regression net for the refactor
   from flat-table to direction-explicit-base + magnitude-scalar
   dispatcher.

2. **Manager vocabulary coverage** -- every (intent, magnitude) the
   manager's ``IntentDecoder._cmd_for_inputs`` can emit in LOCOMOTION
   must NOT idle (returning ``_IDLE_INTENT``) UNLESS the intent is
   intentionally a no-op for the kplanner (``hold_torso``, ``lean_*``,
   ``torso_*``, ``crouch``, ``idle``). This catches "stick pushed,
   planner stays frozen" regressions.

3. **Self-consistency** -- the precomputed ``INTENT_VELOCITY_MAP`` and
   the live ``intent_to_velocity`` dispatcher must agree on every key.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gear_sonic.scripts.x2_kplanner as kp  # noqa: E402
from gear_sonic.scripts.x2_kplanner import (  # noqa: E402
    INTENT_VELOCITY_MAP,
    _BACK_SPEED_MPS,
    _FAST_WALK_SPEED_MPS,
    _HIP_HEIGHT_M,
    _IDLE_INTENT,
    _SIDE_SPEED_MPS,
    _TURN_15_RAD_S,
    _TURN_30_RAD_S,
    _TURN_45_RAD_S,
    _TURN_90_RAD_S,
    _WALK_SPEED_MPS,
    _resolve_velocity,
    intent_to_velocity,
)
from gear_sonic.utils.planner.state_machine import LocomotionCommand  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_runtime_scales():
    """Reset module-level runtime tuning scalars to 1.0 around every test.

    The kplanner stashes ``--turn-left-scale`` and friends as mutable
    module-level globals (so the dispatcher can apply them without
    threading state through every call). Without an autouse reset a
    test that sets ``_RUNTIME_TURN_LEFT_SCALE = 1.5`` would leak into
    every subsequent test in alphabetical order; the parity tests
    would then start failing.
    """
    saved = (
        kp._RUNTIME_TURN_LEFT_SCALE,
        kp._RUNTIME_TURN_RIGHT_SCALE,
        kp._RUNTIME_FORWARD_SCALE,
        kp._RUNTIME_BACKWARD_SCALE,
        kp._RUNTIME_LATERAL_SCALE,
    )
    kp._RUNTIME_TURN_LEFT_SCALE = 1.0
    kp._RUNTIME_TURN_RIGHT_SCALE = 1.0
    kp._RUNTIME_FORWARD_SCALE = 1.0
    kp._RUNTIME_BACKWARD_SCALE = 1.0
    kp._RUNTIME_LATERAL_SCALE = 1.0
    yield
    (
        kp._RUNTIME_TURN_LEFT_SCALE,
        kp._RUNTIME_TURN_RIGHT_SCALE,
        kp._RUNTIME_FORWARD_SCALE,
        kp._RUNTIME_BACKWARD_SCALE,
        kp._RUNTIME_LATERAL_SCALE,
    ) = saved


# ---------------------------------------------------------------------------
# Pinned golden values: the literal 4-tuples the flat-table version of
# ``INTENT_VELOCITY_MAP`` mapped to before the refactor. Any change here
# is a deliberate semantics change to the velocity surface; bump
# carefully.
# ---------------------------------------------------------------------------

_LEGACY_GOLDEN: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("idle", "default"):       (0.0, 0.0, 0.0, _HIP_HEIGHT_M),
    ("idle", "stand"):         (0.0, 0.0, 0.0, _HIP_HEIGHT_M),
    ("walk", "forward"):       (0.0,  _WALK_SPEED_MPS, 0.0, _HIP_HEIGHT_M),
    ("walk", "fast"):          (0.0,  _FAST_WALK_SPEED_MPS, 0.0, _HIP_HEIGHT_M),
    ("fwd_step", "quarter_ft"): (0.0,  _WALK_SPEED_MPS * 0.5, 0.0, _HIP_HEIGHT_M),
    ("fwd_step", "half_ft"):   (0.0,  _WALK_SPEED_MPS, 0.0, _HIP_HEIGHT_M),
    ("fwd_step", "one_ft"):    (0.0,  _WALK_SPEED_MPS * 1.5, 0.0, _HIP_HEIGHT_M),
    ("back_step", "quarter_ft"): (0.0, -_BACK_SPEED_MPS * 0.5, 0.0, _HIP_HEIGHT_M),
    ("back_step", "half_ft"):  (0.0, -_BACK_SPEED_MPS, 0.0, _HIP_HEIGHT_M),
    ("side_left", "default"):  (0.0, 0.0,  _SIDE_SPEED_MPS, _HIP_HEIGHT_M),
    ("side_right", "default"): (0.0, 0.0, -_SIDE_SPEED_MPS, _HIP_HEIGHT_M),
    ("turn_left", "deg_15"):   ( _TURN_15_RAD_S, 0.0, 0.0, _HIP_HEIGHT_M),
    ("turn_left", "deg_30"):   ( _TURN_30_RAD_S, 0.0, 0.0, _HIP_HEIGHT_M),
    ("turn_left", "deg_45"):   ( _TURN_45_RAD_S, 0.0, 0.0, _HIP_HEIGHT_M),
    ("turn_left", "deg_90"):   ( _TURN_90_RAD_S, 0.0, 0.0, _HIP_HEIGHT_M),
    ("turn_right", "deg_15"):  (-_TURN_15_RAD_S, 0.0, 0.0, _HIP_HEIGHT_M),
    ("turn_right", "deg_30"):  (-_TURN_30_RAD_S, 0.0, 0.0, _HIP_HEIGHT_M),
    ("turn_right", "deg_45"):  (-_TURN_45_RAD_S, 0.0, 0.0, _HIP_HEIGHT_M),
    ("turn_right", "deg_90"):  (-_TURN_90_RAD_S, 0.0, 0.0, _HIP_HEIGHT_M),
}


# ---------------------------------------------------------------------------
# New entries added in the surgical fix (manager actually emits these
# from L-stick presses; previous flat-table missed them, causing the
# planner to silently idle on every forward / backward stride).
# ---------------------------------------------------------------------------

_SURGICAL_FIX_GOLDEN: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("fwd_step", "default"):   (0.0,  _WALK_SPEED_MPS, 0.0, _HIP_HEIGHT_M),
    ("back_step", "default"):  (0.0, -_BACK_SPEED_MPS, 0.0, _HIP_HEIGHT_M),
    ("walk", "backward"):      (0.0, -_BACK_SPEED_MPS, 0.0, _HIP_HEIGHT_M),
}


# ---------------------------------------------------------------------------
# Parity tests.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,expected",
    sorted(_LEGACY_GOLDEN.items()),
    ids=lambda v: f"{v[0]}+{v[1]}" if isinstance(v, tuple) and len(v) == 2 else None,
)
def test_legacy_values_preserved(key, expected):
    """Refactor must not change any previously-mapped (intent, magnitude)."""
    intent, magnitude = key
    cmd = LocomotionCommand(intent=intent, magnitude=magnitude)
    assert intent_to_velocity(cmd) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(
    "key,expected",
    sorted(_SURGICAL_FIX_GOLDEN.items()),
    ids=lambda v: f"{v[0]}+{v[1]}" if isinstance(v, tuple) and len(v) == 2 else None,
)
def test_surgical_fix_new_entries(key, expected):
    """The three entries added in the surgical fix must resolve to non-idle."""
    intent, magnitude = key
    cmd = LocomotionCommand(intent=intent, magnitude=magnitude)
    actual = intent_to_velocity(cmd)
    assert actual == pytest.approx(expected, abs=1e-9)
    assert actual != _IDLE_INTENT, (
        f"({intent}, {magnitude}) regressed to _IDLE_INTENT -- the kplanner "
        "would freeze the planner on this manager emission, breaking the "
        "L-stick single-stride and continuous-walk-backward UX."
    )


# ---------------------------------------------------------------------------
# Manager vocabulary coverage. Every label the IntentDecoder can emit in
# LOCOMOTION mode must either resolve to a non-idle velocity OR be one
# of the intentional no-op intents (lean / torso / crouch / hold_torso /
# idle, all of which have no velocity meaning for the kplanner).
# ---------------------------------------------------------------------------


# Intents the kplanner intentionally has no velocity meaning for. Source:
# ``gear_sonic.utils.teleop.vr.intent_decoder._cmd_for_inputs`` (legacy
# discrete-bin path; v7+ continuous torso path emits ``hold_torso``).
_NO_OP_LOCOMOTION_PAIRS: set[tuple[str, str]] = {
    ("idle", "default"),
    ("idle", "stand"),
    # Legacy R-stick discrete bins (only emitted when
    # enable_continuous_torso=False; the kplanner has no torso channel,
    # so these idle out by design).
    ("lean_fwd", "small"),
    ("lean_fwd", "medium"),
    ("lean_fwd", "large"),
    ("torso_left", "deg_30"),
    ("torso_right", "deg_30"),
    # v7+ continuous R-stick (only the manager + heuristic state
    # machine care about the waist_*_deg payload; the kplanner idles).
    ("hold_torso", "continuous"),
    # Y-held crouch (gated on enable_crouch; idles out for the
    # kplanner same as the lean/torso DOFs).
    ("crouch", "medium"),
}

# Every (intent, magnitude) the manager's ``IntentDecoder._cmd_for_inputs``
# can emit in LOCOMOTION mode. Keep this list in sync with that method.
_MANAGER_LOCOMOTION_VOCABULARY: list[tuple[str, str]] = [
    ("idle", "default"),
    ("crouch", "medium"),
    ("walk", "forward"),
    ("walk", "backward"),
    ("fwd_step", "default"),
    ("back_step", "default"),
    ("side_left", "default"),
    ("side_right", "default"),
    ("turn_left", "deg_45"),
    ("turn_right", "deg_45"),
    ("turn_left", "deg_90"),
    ("turn_right", "deg_90"),
    ("hold_torso", "continuous"),
    ("lean_fwd", "small"),
    ("lean_fwd", "medium"),
    ("lean_fwd", "large"),
    ("torso_left", "deg_30"),
    ("torso_right", "deg_30"),
]


@pytest.mark.parametrize(
    "intent,magnitude",
    _MANAGER_LOCOMOTION_VOCABULARY,
    ids=lambda v: v,
)
def test_manager_locomotion_vocabulary_does_not_silently_idle(intent, magnitude):
    """L-stick / R-stick events the manager actually emits resolve sanely.

    Either the kplanner has a non-idle velocity meaning, OR the pair is
    explicitly in ``_NO_OP_LOCOMOTION_PAIRS`` (intentional no-op intent
    with no velocity surface, e.g. torso lean).
    """
    cmd = LocomotionCommand(intent=intent, magnitude=magnitude)
    result = intent_to_velocity(cmd)
    if (intent, magnitude) in _NO_OP_LOCOMOTION_PAIRS:
        assert result == _IDLE_INTENT, (
            f"({intent}, {magnitude}) is in _NO_OP_LOCOMOTION_PAIRS "
            "but did not resolve to _IDLE_INTENT; either fix the "
            "dispatcher or remove the entry from the no-op set."
        )
    else:
        assert result != _IDLE_INTENT, (
            f"({intent}, {magnitude}) silently idled. The manager "
            "emits this and the kplanner should produce a non-zero "
            "velocity for it -- otherwise the operator pushes the "
            "stick and the robot freezes."
        )


# ---------------------------------------------------------------------------
# Unknown-magnitude fallback. New manager-side magnitude labels should
# default to scale=1.0 (= ``default`` intensity) rather than silently
# idling, so a vocabulary addition doesn't freeze the planner before the
# kplanner's table is updated.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "intent",
    ["fwd_step", "back_step", "side_left", "side_right", "turn_left", "turn_right"],
)
def test_unknown_magnitude_falls_back_to_default_scale(intent):
    """Unknown magnitude on a known direction-intent = scale 1.0, not idle."""
    cmd = LocomotionCommand(intent=intent, magnitude="bogus_future_label")
    default_cmd = LocomotionCommand(intent=intent, magnitude="default")
    assert intent_to_velocity(cmd) == pytest.approx(
        intent_to_velocity(default_cmd), abs=1e-9
    )
    assert intent_to_velocity(cmd) != _IDLE_INTENT


def test_walk_unknown_magnitude_idles():
    """Unlike direction-intents, ``walk`` encodes direction in magnitude.

    An unknown magnitude on ``walk`` has ambiguous direction, so safe
    behavior is to idle (avoid guessing forward vs backward).
    """
    cmd = LocomotionCommand(intent="walk", magnitude="bogus_future_label")
    assert intent_to_velocity(cmd) == _IDLE_INTENT


def test_unknown_intent_idles():
    cmd = LocomotionCommand(intent="lean_back", magnitude="default")
    assert intent_to_velocity(cmd) == _IDLE_INTENT


# ---------------------------------------------------------------------------
# Self-consistency: the precomputed map and the dispatcher agree.
# ---------------------------------------------------------------------------


def test_precomputed_map_matches_dispatcher():
    """Every (intent, magnitude) in INTENT_VELOCITY_MAP must agree with
    the live dispatcher. Catches drift between the two surfaces if
    someone updates one without the other."""
    for (intent, magnitude), expected in INTENT_VELOCITY_MAP.items():
        cmd = LocomotionCommand(intent=intent, magnitude=magnitude)
        actual = intent_to_velocity(cmd)
        assert actual == pytest.approx(expected, abs=1e-9), (
            f"INTENT_VELOCITY_MAP[({intent}, {magnitude})] = {expected} "
            f"but intent_to_velocity returned {actual}"
        )


def test_resolve_velocity_is_pure():
    """``_resolve_velocity`` should return the same tuple for the same
    inputs (no hidden state, no shared mutable defaults)."""
    a = _resolve_velocity("fwd_step", "default")
    b = _resolve_velocity("fwd_step", "default")
    assert a == b
    assert a is not _IDLE_INTENT  # not the literal idle sentinel
    assert a != _IDLE_INTENT


# ---------------------------------------------------------------------------
# Runtime tuning scales (--turn-left-scale / --turn-right-scale /
# --forward-scale / --backward-scale / --lateral-scale). These are
# module-level mutable globals set by ``run()`` from CLI args; the
# autouse fixture above pins them to 1.0 around each test so the
# dispatcher stays deterministic for the parity / vocabulary tests.
# ---------------------------------------------------------------------------


def test_turn_left_scale_multiplies_only_left_yaw():
    """Boosting --turn-left-scale must scale turn_left yaw rates only;
    turn_right + translational intents must be untouched.

    This is the operator's lever for compensating training-data L/R
    asymmetry without a model retrain.
    """
    kp._RUNTIME_TURN_LEFT_SCALE = 1.5

    left = intent_to_velocity(LocomotionCommand("turn_left", "deg_45"))
    right = intent_to_velocity(LocomotionCommand("turn_right", "deg_45"))
    fwd = intent_to_velocity(LocomotionCommand("fwd_step", "default"))

    assert left[0] == pytest.approx(_TURN_45_RAD_S * 1.5, abs=1e-9)
    assert (left[1], left[2], left[3]) == pytest.approx(
        (0.0, 0.0, _HIP_HEIGHT_M), abs=1e-9
    )
    # turn_right untouched.
    assert right[0] == pytest.approx(-_TURN_45_RAD_S, abs=1e-9)
    # Forward translation untouched (scale lives on the yaw axis only).
    assert fwd == pytest.approx(
        (0.0, _WALK_SPEED_MPS, 0.0, _HIP_HEIGHT_M), abs=1e-9
    )


def test_turn_right_scale_attenuates_only_right_yaw():
    """Symmetric verification of --turn-right-scale."""
    kp._RUNTIME_TURN_RIGHT_SCALE = 0.5

    right = intent_to_velocity(LocomotionCommand("turn_right", "deg_90"))
    left = intent_to_velocity(LocomotionCommand("turn_left", "deg_90"))

    assert right[0] == pytest.approx(-_TURN_90_RAD_S * 0.5, abs=1e-9)
    assert left[0] == pytest.approx(_TURN_90_RAD_S, abs=1e-9)


def test_forward_scale_affects_fwd_step_and_walk_forward():
    """--forward-scale must reach both ``fwd_step`` and ``walk/forward``
    (the manager's two forward emitters); must not touch backward or
    lateral.
    """
    kp._RUNTIME_FORWARD_SCALE = 0.6

    fwd_step = intent_to_velocity(LocomotionCommand("fwd_step", "default"))
    fwd_walk = intent_to_velocity(LocomotionCommand("walk", "forward"))
    back_step = intent_to_velocity(LocomotionCommand("back_step", "default"))
    back_walk = intent_to_velocity(LocomotionCommand("walk", "backward"))
    side = intent_to_velocity(LocomotionCommand("side_left", "default"))

    assert fwd_step[1] == pytest.approx(_WALK_SPEED_MPS * 0.6, abs=1e-9)
    assert fwd_walk[1] == pytest.approx(_WALK_SPEED_MPS * 0.6, abs=1e-9)
    # Backward + lateral untouched.
    assert back_step[1] == pytest.approx(-_BACK_SPEED_MPS, abs=1e-9)
    assert back_walk[1] == pytest.approx(-_BACK_SPEED_MPS, abs=1e-9)
    assert side[2] == pytest.approx(_SIDE_SPEED_MPS, abs=1e-9)


def test_backward_scale_affects_back_step_and_walk_backward():
    kp._RUNTIME_BACKWARD_SCALE = 0.5

    back_step = intent_to_velocity(LocomotionCommand("back_step", "default"))
    back_walk = intent_to_velocity(LocomotionCommand("walk", "backward"))
    fwd_step = intent_to_velocity(LocomotionCommand("fwd_step", "default"))

    assert back_step[1] == pytest.approx(-_BACK_SPEED_MPS * 0.5, abs=1e-9)
    assert back_walk[1] == pytest.approx(-_BACK_SPEED_MPS * 0.5, abs=1e-9)
    assert fwd_step[1] == pytest.approx(_WALK_SPEED_MPS, abs=1e-9)


def test_lateral_scale_affects_only_side_intents():
    kp._RUNTIME_LATERAL_SCALE = 0.7

    side_l = intent_to_velocity(LocomotionCommand("side_left", "default"))
    side_r = intent_to_velocity(LocomotionCommand("side_right", "default"))
    fwd = intent_to_velocity(LocomotionCommand("fwd_step", "default"))
    turn = intent_to_velocity(LocomotionCommand("turn_left", "deg_45"))

    assert side_l[2] == pytest.approx(_SIDE_SPEED_MPS * 0.7, abs=1e-9)
    assert side_r[2] == pytest.approx(-_SIDE_SPEED_MPS * 0.7, abs=1e-9)
    assert fwd[1] == pytest.approx(_WALK_SPEED_MPS, abs=1e-9)
    assert turn[0] == pytest.approx(_TURN_45_RAD_S, abs=1e-9)


def test_runtime_scales_do_not_reanimate_idle_intent():
    """Boosting a scale must not turn an _IDLE_INTENT into a non-idle.

    A regression here would mean the kplanner starts emitting velocity
    for a no-op intent like ``hold_torso``; the autouse fixture
    guarantees this state across tests.
    """
    kp._RUNTIME_TURN_LEFT_SCALE = 5.0
    kp._RUNTIME_FORWARD_SCALE = 5.0
    kp._RUNTIME_LATERAL_SCALE = 5.0

    for intent, magnitude in (
        ("hold_torso", "continuous"),
        ("lean_fwd", "medium"),
        ("torso_left", "deg_30"),
        ("crouch", "medium"),
        ("idle", "default"),
        ("unknown_intent", "default"),
    ):
        cmd = LocomotionCommand(intent=intent, magnitude=magnitude)
        assert intent_to_velocity(cmd) == _IDLE_INTENT, (
            f"({intent}, {magnitude}) leaked out of idle when runtime "
            "scales were boosted; the scale path should only multiply "
            "an already-resolved non-idle velocity."
        )
