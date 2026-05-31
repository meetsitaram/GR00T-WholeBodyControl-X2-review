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
    _CONTINUOUS_TURN_MAX_RAD_S,
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
        kp._CONTINUOUS_TURN_MAX_RAD_S,
    )
    kp._RUNTIME_TURN_LEFT_SCALE = 1.0
    kp._RUNTIME_TURN_RIGHT_SCALE = 1.0
    kp._RUNTIME_FORWARD_SCALE = 1.0
    kp._RUNTIME_BACKWARD_SCALE = 1.0
    kp._RUNTIME_LATERAL_SCALE = 1.0
    kp._CONTINUOUS_TURN_MAX_RAD_S = kp._DEFAULT_CONTINUOUS_TURN_MAX_RAD_S
    yield
    (
        kp._RUNTIME_TURN_LEFT_SCALE,
        kp._RUNTIME_TURN_RIGHT_SCALE,
        kp._RUNTIME_FORWARD_SCALE,
        kp._RUNTIME_BACKWARD_SCALE,
        kp._RUNTIME_LATERAL_SCALE,
        kp._CONTINUOUS_TURN_MAX_RAD_S,
    ) = saved


# ---------------------------------------------------------------------------
# Pinned golden values: the literal 4-tuples the flat-table version of
# ``INTENT_VELOCITY_MAP`` mapped to before the refactor. Any change here
# is a deliberate semantics change to the velocity surface; bump
# carefully.
# ---------------------------------------------------------------------------

# Velocity tuple layout: ``(yaw_rate, vel_x=lateral, vel_z=forward, hip_h)``.
# This matches ``_BASE_VELOCITY`` after the 2026-05-29 channel-swap
# bugfix (commit ``46bd017``) which moved forward speed into ``vel_z``
# and lateral speed into ``vel_x``. The pre-bugfix layout had them
# swapped; values below are the post-fix expected truth.
_LEGACY_GOLDEN: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("idle", "default"):       (0.0, 0.0, 0.0, _HIP_HEIGHT_M),
    ("idle", "stand"):         (0.0, 0.0, 0.0, _HIP_HEIGHT_M),
    ("walk", "forward"):       (0.0, 0.0,  _WALK_SPEED_MPS,        _HIP_HEIGHT_M),
    ("walk", "fast"):          (0.0, 0.0,  _FAST_WALK_SPEED_MPS,   _HIP_HEIGHT_M),
    ("fwd_step", "quarter_ft"): (0.0, 0.0,  _WALK_SPEED_MPS * 0.5, _HIP_HEIGHT_M),
    ("fwd_step", "half_ft"):   (0.0, 0.0,  _WALK_SPEED_MPS,        _HIP_HEIGHT_M),
    ("fwd_step", "one_ft"):    (0.0, 0.0,  _WALK_SPEED_MPS * 1.5,  _HIP_HEIGHT_M),
    ("back_step", "quarter_ft"): (0.0, 0.0, -_BACK_SPEED_MPS * 0.5, _HIP_HEIGHT_M),
    ("back_step", "half_ft"):  (0.0, 0.0, -_BACK_SPEED_MPS,        _HIP_HEIGHT_M),
    ("side_left", "default"):  (0.0,  _SIDE_SPEED_MPS, 0.0,        _HIP_HEIGHT_M),
    ("side_right", "default"): (0.0, -_SIDE_SPEED_MPS, 0.0,        _HIP_HEIGHT_M),
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
    ("fwd_step", "default"):   (0.0, 0.0,  _WALK_SPEED_MPS, _HIP_HEIGHT_M),
    ("back_step", "default"):  (0.0, 0.0, -_BACK_SPEED_MPS, _HIP_HEIGHT_M),
    ("walk", "backward"):      (0.0, 0.0, -_BACK_SPEED_MPS, _HIP_HEIGHT_M),
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
    # Tuple layout: (yaw_rate, vel_x=lateral, vel_z=forward, hip_h).
    assert fwd == pytest.approx(
        (0.0, 0.0, _WALK_SPEED_MPS, _HIP_HEIGHT_M), abs=1e-9
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

    # Tuple layout: (yaw_rate, vel_x=lateral, vel_z=forward, hip_h).
    assert fwd_step[2] == pytest.approx(_WALK_SPEED_MPS * 0.6, abs=1e-9)
    assert fwd_walk[2] == pytest.approx(_WALK_SPEED_MPS * 0.6, abs=1e-9)
    # Backward + lateral untouched.
    assert back_step[2] == pytest.approx(-_BACK_SPEED_MPS, abs=1e-9)
    assert back_walk[2] == pytest.approx(-_BACK_SPEED_MPS, abs=1e-9)
    assert side[1] == pytest.approx(_SIDE_SPEED_MPS, abs=1e-9)


def test_backward_scale_affects_back_step_and_walk_backward():
    kp._RUNTIME_BACKWARD_SCALE = 0.5

    back_step = intent_to_velocity(LocomotionCommand("back_step", "default"))
    back_walk = intent_to_velocity(LocomotionCommand("walk", "backward"))
    fwd_step = intent_to_velocity(LocomotionCommand("fwd_step", "default"))

    assert back_step[2] == pytest.approx(-_BACK_SPEED_MPS * 0.5, abs=1e-9)
    assert back_walk[2] == pytest.approx(-_BACK_SPEED_MPS * 0.5, abs=1e-9)
    assert fwd_step[2] == pytest.approx(_WALK_SPEED_MPS, abs=1e-9)


def test_lateral_scale_affects_only_side_intents():
    kp._RUNTIME_LATERAL_SCALE = 0.7

    side_l = intent_to_velocity(LocomotionCommand("side_left", "default"))
    side_r = intent_to_velocity(LocomotionCommand("side_right", "default"))
    fwd = intent_to_velocity(LocomotionCommand("fwd_step", "default"))
    turn = intent_to_velocity(LocomotionCommand("turn_left", "deg_45"))

    assert side_l[1] == pytest.approx(_SIDE_SPEED_MPS * 0.7, abs=1e-9)
    assert side_r[1] == pytest.approx(-_SIDE_SPEED_MPS * 0.7, abs=1e-9)
    assert fwd[2] == pytest.approx(_WALK_SPEED_MPS, abs=1e-9)
    assert turn[0] == pytest.approx(_TURN_45_RAD_S, abs=1e-9)


def test_continuous_locomotion_idle_when_all_sticks_zero():
    """``locomotion / continuous`` with all-zero sticks produces idle."""
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_fwd=0.0, stick_side=0.0, stick_yaw=0.0,
    )
    assert intent_to_velocity(cmd) == _IDLE_INTENT


def test_continuous_locomotion_full_fwd_stick_hits_walk_speed():
    """``stick_fwd=1`` -> shaped to 1 -> vel_z = _WALK_SPEED_MPS."""
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_fwd=1.0,
    )
    yaw, vx, vz, hip_h = intent_to_velocity(cmd)
    assert yaw == pytest.approx(0.0, abs=1e-9)
    assert vx  == pytest.approx(0.0, abs=1e-9)
    assert vz  == pytest.approx(_WALK_SPEED_MPS, abs=1e-9)
    assert hip_h == pytest.approx(_HIP_HEIGHT_M, abs=1e-9)


def test_continuous_locomotion_half_stick_linear_default():
    """At the default linear shaping (exp=1.0), ``stick_fwd=0.5`` -> shaped
    to 0.5 -> vel_z = 0.5 * _WALK_SPEED.

    Defaults to linear because the previous squared default produced
    only 25%% of walk speed at 50%% deflection, which the operator
    perceived as "robot won't move forward". Linear (50%% deflection ->
    50%% speed) is a much closer match to the bucketed-path muscle
    memory while still allowing fine creeping near zero.
    """
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_fwd=0.5,
    )
    _, _, vz, _ = intent_to_velocity(cmd)
    assert vz == pytest.approx(0.5 * _WALK_SPEED_MPS, abs=1e-9)


def test_continuous_locomotion_shape_exponent_tunable():
    """``--stick-shape-exp`` changes the curve. exp=2.0 reproduces the
    historical squared curve (0.5 stick -> 0.25 vel); exp=0.5 produces
    a bucketed-like fast feel (0.5 stick -> ~0.707 vel)."""
    import gear_sonic.scripts.x2_kplanner as kp
    original = kp._RUNTIME_STICK_SHAPING_EXPONENT
    try:
        kp._RUNTIME_STICK_SHAPING_EXPONENT = 2.0
        cmd = LocomotionCommand(
            intent="locomotion", magnitude="continuous", stick_fwd=0.5,
        )
        _, _, vz_sq, _ = intent_to_velocity(cmd)
        assert vz_sq == pytest.approx(0.25 * _WALK_SPEED_MPS, abs=1e-9)

        kp._RUNTIME_STICK_SHAPING_EXPONENT = 0.5
        _, _, vz_half, _ = intent_to_velocity(cmd)
        # 0.5 ** 0.5 = sqrt(0.5) = 0.7071...
        assert vz_half == pytest.approx(
            (0.5 ** 0.5) * _WALK_SPEED_MPS, abs=1e-9,
        )
    finally:
        kp._RUNTIME_STICK_SHAPING_EXPONENT = original


def test_continuous_locomotion_back_stick_uses_back_speed():
    """Negative stick_fwd is scaled by _BACK_SPEED_MPS (not _WALK_SPEED)."""
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_fwd=-1.0,
    )
    _, _, vz, _ = intent_to_velocity(cmd)
    assert vz == pytest.approx(-_BACK_SPEED_MPS, abs=1e-9)


def test_continuous_locomotion_side_stick_maps_to_vel_x():
    """``stick_side=+1`` (L-stick right) -> side_right -> -vel_x,
    matching the bucketed ``_BASE_VELOCITY['side_right']`` convention."""
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_side=1.0,
    )
    _, vx, _, _ = intent_to_velocity(cmd)
    assert vx == pytest.approx(-_SIDE_SPEED_MPS, abs=1e-9)
    cmd_left = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_side=-1.0,
    )
    _, vx_left, _, _ = intent_to_velocity(cmd_left)
    assert vx_left == pytest.approx(_SIDE_SPEED_MPS, abs=1e-9)


def test_continuous_locomotion_yaw_stick_sign_matches_bucketed_turn():
    """``stick_yaw=+1`` (R-stick right) -> turn-right -> negative yaw_rate,
    same sign convention as the bucketed ``turn_right`` path.

    The *magnitude* is intentionally decoupled: continuous mode caps at
    ``_CONTINUOUS_TURN_MAX_RAD_S`` (a gentler ceiling so analog R-stick
    turns don't overdrive a model trained mostly on forward walking),
    while the bucketed ``turn_right deg_45`` path keeps the legacy
    ``_TURN_45_RAD_S = 1.5 rad/s``. See the constant's comment block.
    """
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_yaw=1.0,
    )
    yaw, _, _, _ = intent_to_velocity(cmd)
    assert yaw == pytest.approx(-_CONTINUOUS_TURN_MAX_RAD_S, abs=1e-9)
    cmd_left = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_yaw=-1.0,
    )
    yaw_left, _, _, _ = intent_to_velocity(cmd_left)
    assert yaw_left == pytest.approx(_CONTINUOUS_TURN_MAX_RAD_S, abs=1e-9)


def test_continuous_locomotion_combined_axes():
    """All three sticks at full deflection -> full speed on all three
    axes simultaneously, with the bucketed-path sign conventions."""
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_fwd=1.0, stick_side=-1.0, stick_yaw=-1.0,
    )
    yaw, vx, vz, hip_h = intent_to_velocity(cmd)
    assert vz  == pytest.approx(_WALK_SPEED_MPS, abs=1e-9)
    # stick_side = -1 -> side_left -> +vel_x
    assert vx  == pytest.approx(_SIDE_SPEED_MPS, abs=1e-9)
    # stick_yaw = -1 -> turn_left -> +yaw_rate, capped at the continuous
    # ceiling (decoupled from bucketed _TURN_45_RAD_S).
    assert yaw == pytest.approx(_CONTINUOUS_TURN_MAX_RAD_S, abs=1e-9)
    assert hip_h == pytest.approx(_HIP_HEIGHT_M, abs=1e-9)


def test_continuous_locomotion_runtime_scale_applies():
    """``--kplanner-forward-scale 0.5`` (env-mutated _RUNTIME_FORWARD_SCALE)
    caps continuous mode the same way it caps the bucketed ``fwd_step``."""
    kp._RUNTIME_FORWARD_SCALE = 0.5
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous", stick_fwd=1.0,
    )
    _, _, vz, _ = intent_to_velocity(cmd)
    assert vz == pytest.approx(0.5 * _WALK_SPEED_MPS, abs=1e-9)


def test_continuous_turn_ceiling_decoupled_from_bucketed():
    """Bucketed ``turn_left deg_45`` keeps the legacy 1.5 rad/s ceiling
    even after the continuous knob is dialled way down.

    Regression guard for the 2026-05-30 fix: the analog R-stick turn
    rate was previously hard-wired to ``_TURN_45_RAD_S``, so any operator
    that dialled down the continuous-mode ceiling would also have nerfed
    button-driven pivots. The two paths must be independent.
    """
    kp._CONTINUOUS_TURN_MAX_RAD_S = 0.1  # essentially "no turn"
    bucketed = LocomotionCommand(
        intent="turn_left", magnitude="deg_45",
    )
    yaw_bucketed, _, _, _ = intent_to_velocity(bucketed)
    assert yaw_bucketed == pytest.approx(_TURN_45_RAD_S, abs=1e-9)
    continuous = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_yaw=-1.0,
    )
    yaw_continuous, _, _, _ = intent_to_velocity(continuous)
    assert yaw_continuous == pytest.approx(0.1, abs=1e-9)


def test_continuous_turn_ceiling_is_runtime_mutable():
    """Mutating ``_CONTINUOUS_TURN_MAX_RAD_S`` at runtime (as ``run()``
    does when ``--continuous-turn-max-rad-s`` is passed) immediately
    changes the yaw_rate the dispatcher emits for full R-stick.

    This is what plumbs the wrapper-script env var through to live
    behaviour without a kplanner restart-rebuild cycle.
    """
    kp._CONTINUOUS_TURN_MAX_RAD_S = 1.25
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous", stick_yaw=1.0,
    )
    yaw, _, _, _ = intent_to_velocity(cmd)
    assert yaw == pytest.approx(-1.25, abs=1e-9)
    kp._CONTINUOUS_TURN_MAX_RAD_S = 0.25
    yaw2, _, _, _ = intent_to_velocity(cmd)
    assert yaw2 == pytest.approx(-0.25, abs=1e-9)


def test_continuous_turn_ceiling_scales_partial_stick_linearly():
    """``shaped_yaw`` after shape_exp=1 is linear in the stick deflection;
    the ceiling scales the result, so 50%-stick yields 50% of the ceiling.

    Catches accidental introduction of a non-linear remap inside the
    yaw-stick path that could make small deflections feel jumpy.
    """
    kp._CONTINUOUS_TURN_MAX_RAD_S = 0.8
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous", stick_yaw=0.5,
    )
    yaw, _, _, _ = intent_to_velocity(cmd)
    # IntentDecoder owns deadzone rescaling, so stick_yaw arrives here
    # already in [-1, 1]. The kplanner only applies stick_shape_exp
    # (default 1.0 = identity). So shaped = 0.5, yaw = -0.5 * 0.8.
    assert yaw == pytest.approx(-0.5 * 0.8, abs=1e-9)


def test_continuous_locomotion_turn_scales_split_by_sign():
    """``stick_yaw > 0`` -> turn_right path -> _RUNTIME_TURN_RIGHT_SCALE.
    ``stick_yaw < 0`` -> turn_left path -> _RUNTIME_TURN_LEFT_SCALE."""
    kp._RUNTIME_TURN_LEFT_SCALE = 0.3
    kp._RUNTIME_TURN_RIGHT_SCALE = 0.7
    right = LocomotionCommand(
        intent="locomotion", magnitude="continuous", stick_yaw=1.0,
    )
    left = LocomotionCommand(
        intent="locomotion", magnitude="continuous", stick_yaw=-1.0,
    )
    yaw_right, _, _, _ = intent_to_velocity(right)
    yaw_left,  _, _, _ = intent_to_velocity(left)
    # Continuous-mode yaw ceiling is _CONTINUOUS_TURN_MAX_RAD_S (not
    # _TURN_45_RAD_S); per-side runtime scales multiply the ceiling.
    assert yaw_right == pytest.approx(-_CONTINUOUS_TURN_MAX_RAD_S * 0.7, abs=1e-9)
    assert yaw_left  == pytest.approx( _CONTINUOUS_TURN_MAX_RAD_S * 0.3, abs=1e-9)


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


# ---------------------------------------------------------------------------
# Direct-velocity passthrough (used by x2_pkl_command_source for replaying
# recorded motion clips through the kplanner -> deploy chain). The
# dispatcher must short-circuit and return the exact 4-tuple unchanged
# even when continuous-stick or runtime-scale fields are also set on
# the same command. Missing the field reverts to the normal path.
# ---------------------------------------------------------------------------


def test_direct_velocity_short_circuits_dispatcher():
    """direct_velocity != None -> returned verbatim; intent/magnitude ignored."""
    target = (0.42, -0.13, 0.97, 0.91)
    cmd = LocomotionCommand(
        intent="idle", magnitude="default",
        direct_velocity=target,
    )
    assert intent_to_velocity(cmd) == pytest.approx(target, abs=1e-9)


def test_direct_velocity_bypasses_runtime_scales():
    """A direct-velocity command must NOT pick up the forward / yaw scales.

    Replaying a recorded clip's velocity through the runtime scales
    would double-apply the operator's compensation knobs -- they were
    designed to cap live Quest3 stick output, not to re-shape PKL data.
    """
    kp._RUNTIME_FORWARD_SCALE = 0.5
    kp._RUNTIME_TURN_LEFT_SCALE = 2.0
    target = (0.3, 0.0, 0.6, 0.95)
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        direct_velocity=target,
    )
    assert intent_to_velocity(cmd) == pytest.approx(target, abs=1e-9)


def test_direct_velocity_bypasses_continuous_shaping():
    """A direct-velocity command must NOT go through stick shaping.

    Even when the same command also carries non-zero stick_fwd /
    stick_side / stick_yaw, the direct path wins so a poorly-formed
    payload from the transmitter doesn't accidentally mix in shaped
    velocity from the analog-stick branch.
    """
    target = (-0.5, 0.2, 0.4, 0.95)
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        stick_fwd=1.0, stick_side=1.0, stick_yaw=1.0,
        direct_velocity=target,
    )
    assert intent_to_velocity(cmd) == pytest.approx(target, abs=1e-9)


def test_direct_velocity_none_falls_back_to_bucketed():
    """Missing direct_velocity preserves the bucketed-dispatch behaviour."""
    cmd = LocomotionCommand(
        intent="fwd_step", magnitude="default", direct_velocity=None,
    )
    assert intent_to_velocity(cmd) == pytest.approx(
        (0.0, 0.0, _WALK_SPEED_MPS, _HIP_HEIGHT_M), abs=1e-9,
    )


def test_direct_velocity_idle_tuple_returned_verbatim():
    """direct_velocity equal to _IDLE_INTENT is honoured (e.g. replaying a
    standing PKL frame). The publisher's idle gate then freezes the wire
    on the warmup anchor; that's the correct behaviour for a stationary
    clip frame.
    """
    cmd = LocomotionCommand(
        intent="locomotion", magnitude="continuous",
        direct_velocity=_IDLE_INTENT,
    )
    assert intent_to_velocity(cmd) == _IDLE_INTENT
