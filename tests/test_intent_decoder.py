"""Behavior tests for the X2 IntentDecoder.

Pins the Quest-3-button-and-stick to ``(intent, magnitude)`` mapping
that ``quest3_manager_x2.py`` will publish on the ``planner_cmd`` ZMQ
topic. Locking these in here means a UX tweak to the manager script
can't silently change the planner-side semantics.
"""

from __future__ import annotations

import pytest

from gear_sonic.utils.teleop.vr.button_state_machine import ButtonEvents
from gear_sonic.utils.teleop.vr.intent_decoder import (
    IntentDecoder,
    LocomotionCmd,
    ModeTransition,
    StreamMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _no_buttons() -> ButtonEvents:
    return ButtonEvents(
        a_pressed=False, b_pressed=False, x_pressed=False, y_pressed=False,
        ab_pressed=False, xy_pressed=False, ax_pressed=False, by_pressed=False,
        abxy_pressed=False,
    )


def _abxy_chord() -> ButtonEvents:
    return ButtonEvents(
        a_pressed=True, b_pressed=True, x_pressed=True, y_pressed=True,
        ab_pressed=True, xy_pressed=True, ax_pressed=True, by_pressed=True,
        abxy_pressed=True,
    )


def _b_single() -> ButtonEvents:
    # Note: when B fires alone, the chord flags involving B are NOT also
    # set (they require the other button held); the abxy_pressed test
    # in test_vr_button_and_joystick_utilities.py covers the chord case.
    return ButtonEvents(
        a_pressed=False, b_pressed=True, x_pressed=False, y_pressed=False,
        ab_pressed=False, xy_pressed=False, ax_pressed=False, by_pressed=False,
        abxy_pressed=False,
    )


# ---------------------------------------------------------------------------
# Mode transitions
# ---------------------------------------------------------------------------


def test_starts_in_off_mode():
    dec = IntentDecoder()
    assert dec.mode is StreamMode.OFF


def test_abxy_chord_off_to_locomotion():
    dec = IntentDecoder()
    transition = dec.update_mode(_abxy_chord())
    assert transition == ModeTransition(
        previous=StreamMode.OFF, current=StreamMode.LOCOMOTION
    )
    assert dec.mode is StreamMode.LOCOMOTION


def test_abxy_chord_locomotion_to_off():
    dec = IntentDecoder()
    dec.update_mode(_abxy_chord())  # OFF -> LOCOMOTION
    transition = dec.update_mode(_abxy_chord())
    assert transition == ModeTransition(
        previous=StreamMode.LOCOMOTION, current=StreamMode.OFF
    )


def test_abxy_chord_arm_to_off():
    dec = IntentDecoder()
    dec.update_mode(_abxy_chord())  # OFF -> LOCOMOTION
    dec.update_mode(_b_single())     # LOCOMOTION -> ARM_MANIPULATION
    transition = dec.update_mode(_abxy_chord())
    assert transition == ModeTransition(
        previous=StreamMode.ARM_MANIPULATION, current=StreamMode.OFF
    )


def test_b_single_no_op_in_off_mode():
    """B-single is a no-op while in OFF mode (avoids accidental engage)."""
    dec = IntentDecoder()
    transition = dec.update_mode(_b_single())
    assert transition is None
    assert dec.mode is StreamMode.OFF


def test_b_single_toggles_locomotion_arm():
    dec = IntentDecoder()
    dec.update_mode(_abxy_chord())  # OFF -> LOCOMOTION
    transition = dec.update_mode(_b_single())
    assert transition == ModeTransition(
        previous=StreamMode.LOCOMOTION, current=StreamMode.ARM_MANIPULATION
    )
    transition = dec.update_mode(_b_single())
    assert transition == ModeTransition(
        previous=StreamMode.ARM_MANIPULATION, current=StreamMode.LOCOMOTION
    )


def test_no_button_event_is_no_transition():
    dec = IntentDecoder()
    dec.update_mode(_abxy_chord())  # OFF -> LOCOMOTION
    transition = dec.update_mode(_no_buttons())
    assert transition is None
    assert dec.mode is StreamMode.LOCOMOTION


# ---------------------------------------------------------------------------
# Locomotion decoding (LOCOMOTION mode only)
# ---------------------------------------------------------------------------


def _make_loco(
    *,
    enable_lean_fwd: bool = False,
    enable_torso: bool = False,
) -> IntentDecoder:
    dec = IntentDecoder(
        stick_deadzone=0.30,
        enable_lean_fwd=enable_lean_fwd,
        enable_torso=enable_torso,
    )
    dec.update_mode(_abxy_chord())
    assert dec.mode is StreamMode.LOCOMOTION
    return dec


def test_no_command_in_off_mode():
    dec = IntentDecoder()
    cmd = dec.decode_locomotion(lx=0.0, ly=1.0, rx=0.0, ry=0.0, y_held=False, now=0.0)
    assert cmd is None


def test_no_command_in_arm_mode():
    dec = _make_loco()
    dec.update_mode(_b_single())
    assert dec.mode is StreamMode.ARM_MANIPULATION
    cmd = dec.decode_locomotion(lx=0.0, ly=1.0, rx=0.0, ry=0.0, y_held=False, now=0.0)
    assert cmd is None


def test_neutral_sticks_emit_idle_once():
    dec = _make_loco()
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("idle", "default")
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, False, now=0.02)
    assert cmd is None, "duplicate idle should not re-fire"


def test_within_deadzone_treated_as_neutral():
    dec = _make_loco()
    cmd = dec.decode_locomotion(0.2, 0.2, 0.2, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("idle", "default")


@pytest.mark.parametrize(
    "lx,ly,rx,ry,y,expected",
    [
        # left stick cardinal directions
        (0.0,  0.9, 0.0, 0.0, False, LocomotionCmd("fwd_step",   "default")),
        (0.0, -0.9, 0.0, 0.0, False, LocomotionCmd("back_step",  "default")),
        ( 0.9, 0.0, 0.0, 0.0, False, LocomotionCmd("side_right", "default")),
        (-0.9, 0.0, 0.0, 0.0, False, LocomotionCmd("side_left",  "default")),

        # diagonals - whichever axis dominates wins; ties go to forward/back
        (0.4,  0.9, 0.0, 0.0, False, LocomotionCmd("fwd_step",   "default")),
        (0.9,  0.4, 0.0, 0.0, False, LocomotionCmd("side_right", "default")),
        (0.5,  0.5, 0.0, 0.0, False, LocomotionCmd("fwd_step",   "default")),

        # right stick yaw turn
        (0.0,  0.0,  0.9, 0.0, False, LocomotionCmd("turn_right", "deg_45")),
        (0.0,  0.0, -0.9, 0.0, False, LocomotionCmd("turn_left",  "deg_45")),

        # left stick wins over right when both are deflected
        (0.0,  0.9, 0.9, 0.0, False, LocomotionCmd("fwd_step",   "default")),

        # Y held with crouch disabled (default): falls through to whatever
        # the sticks would have emitted, never crouch.
        (0.0,  0.9, 0.9, 0.0, True,  LocomotionCmd("fwd_step",   "default")),
        (0.0,  0.0, 0.0, 0.0, True,  LocomotionCmd("idle",       "default")),
    ],
)
def test_locomotion_vocabulary(lx, ly, rx, ry, y, expected):
    dec = _make_loco()
    cmd = dec.decode_locomotion(lx=lx, ly=ly, rx=rx, ry=ry, y_held=y, now=0.0)
    assert cmd == expected


def test_y_held_with_crouch_disabled_does_not_emit_crouch():
    """Crouch is disabled by default; Y must not produce a crouch command
    even outside the chord-debounce window."""
    dec = IntentDecoder(stick_deadzone=0.30, chord_debounce_s=0.0)
    dec.update_mode(_abxy_chord(), now=0.0)
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=0.0)
    assert cmd == LocomotionCmd("idle", "default")


# ---------------------------------------------------------------------------
# A-held modifier: continuous walk
# ---------------------------------------------------------------------------


def test_a_held_plus_fwd_stick_emits_walk_forward():
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.9, rx=0.0, ry=0.0, y_held=False, now=0.0, a_held=True
    )
    assert cmd == LocomotionCmd("walk", "forward")


def test_a_held_plus_back_stick_emits_walk_backward():
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=-0.9, rx=0.0, ry=0.0, y_held=False, now=0.0, a_held=True
    )
    assert cmd == LocomotionCmd("walk", "backward")


def test_a_held_plus_neutral_stick_emits_idle_not_walk():
    """A held without stick deflection is just idle; the modifier only
    upgrades an already-active forward/back deflection."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.0, ry=0.0, y_held=False, now=0.0, a_held=True
    )
    assert cmd == LocomotionCmd("idle", "default")


def test_a_held_plus_side_stick_falls_through_to_side_step():
    """No walk_left / walk_right primitives exist; A modifier is ignored
    when the dominant axis is sideways."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.9, ly=0.0, rx=0.0, ry=0.0, y_held=False, now=0.0, a_held=True
    )
    assert cmd == LocomotionCmd("side_right", "default")


def test_a_held_default_false_preserves_legacy_single_step():
    """Existing callers that don't pass a_held continue to get
    fwd_step / back_step exactly as before."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("fwd_step", "default")


# ---------------------------------------------------------------------------
# X-held modifier: 90deg turn
# ---------------------------------------------------------------------------


def test_x_held_plus_right_turn_emits_90deg():
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.9, ry=0.0, y_held=False, now=0.0, x_held=True
    )
    assert cmd == LocomotionCmd("turn_right", "deg_90")


def test_x_held_plus_left_turn_emits_90deg():
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=-0.9, ry=0.0, y_held=False, now=0.0, x_held=True
    )
    assert cmd == LocomotionCmd("turn_left", "deg_90")


def test_x_held_default_false_preserves_legacy_45deg_turn():
    dec = _make_loco()
    cmd = dec.decode_locomotion(0.0, 0.0, 0.9, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("turn_right", "deg_45")


def test_x_held_with_no_right_stick_emits_idle():
    """Modifier only upgrades an already-active turn; X alone is a no-op
    in the locomotion vocabulary."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.0, ry=0.0, y_held=False, now=0.0, x_held=True
    )
    assert cmd == LocomotionCmd("idle", "default")


def test_a_and_x_held_left_stick_wins_over_x_modifier():
    """Left-stick precedence is unchanged: when both sticks are
    deflected and both modifiers are held, the walk command fires
    (left-stick) and the X-turn modifier is ignored."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.9, rx=0.9, ry=0.0,
        y_held=False, now=0.0, a_held=True, x_held=True,
    )
    assert cmd == LocomotionCmd("walk", "forward")


# ---------------------------------------------------------------------------
# Right-stick Y forward: graded lean_fwd_{small,medium,large}
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ry,expected_magnitude",
    [
        (0.35, "small"),    # deadzone(0.30) <= ry < medium(0.55)
        (0.50, "small"),
        (0.55, "medium"),   # medium(0.55) <= ry < large(0.80)
        (0.65, "medium"),
        (0.79, "medium"),
        (0.80, "large"),    # ry >= large(0.80)
        (1.00, "large"),
    ],
)
def test_right_stick_forward_emits_graded_lean(ry, expected_magnitude):
    dec = _make_loco(enable_lean_fwd=True)
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.0, ry=ry, y_held=False, now=0.0
    )
    assert cmd == LocomotionCmd("lean_fwd", expected_magnitude)


def test_right_stick_back_does_not_emit_lean():
    """No lean_back primitive exists; a back push stays at idle."""
    dec = _make_loco(enable_lean_fwd=True)
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.0, ry=-0.9, y_held=False, now=0.0
    )
    assert cmd == LocomotionCmd("idle", "default")


# ---------------------------------------------------------------------------
# Right-stick X: torso (soft) vs turn (hard) split
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rx,expected",
    [
        # Soft push (deadzone <= |rx| < turn_threshold(0.75)) -> torso
        (0.35,  LocomotionCmd("torso_right", "deg_30")),
        (0.74,  LocomotionCmd("torso_right", "deg_30")),
        (-0.35, LocomotionCmd("torso_left",  "deg_30")),
        (-0.74, LocomotionCmd("torso_left",  "deg_30")),
        # Hard push (|rx| >= 0.75) -> 45° turn (default magnitude)
        (0.75,  LocomotionCmd("turn_right", "deg_45")),
        (0.90,  LocomotionCmd("turn_right", "deg_45")),
        (-0.75, LocomotionCmd("turn_left",  "deg_45")),
        (-0.90, LocomotionCmd("turn_left",  "deg_45")),
    ],
)
def test_right_stick_x_torso_vs_turn_split(rx, expected):
    dec = _make_loco(enable_torso=True)
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=rx, ry=0.0, y_held=False, now=0.0
    )
    assert cmd == expected


def test_x_held_only_upgrades_hard_turn_not_soft_torso():
    """X held + soft rx push stays as torso; X modifier only takes
    effect once the operator commits to a real turn (|rx| >= 0.60)."""
    dec = _make_loco(enable_torso=True)
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.40, ry=0.0,
        y_held=False, now=0.0, x_held=True,
    )
    assert cmd == LocomotionCmd("torso_right", "deg_30")
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.90, ry=0.0,
        y_held=False, now=10.0, x_held=True,  # advance time + change cmd
    )
    assert cmd == LocomotionCmd("turn_right", "deg_90")


def test_right_stick_dominant_axis_y_wins_over_x_when_larger():
    dec = _make_loco(enable_lean_fwd=True, enable_torso=True)
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.50, ry=0.85, y_held=False, now=0.0
    )
    assert cmd == LocomotionCmd("lean_fwd", "large")


def test_right_stick_dominant_axis_x_wins_when_y_smaller():
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.85, ry=0.40, y_held=False, now=0.0
    )
    assert cmd == LocomotionCmd("turn_right", "deg_45")


def test_left_stick_still_wins_over_right_stick_lean():
    """Left-stick precedence is unchanged after lean/torso wiring."""
    dec = _make_loco(enable_lean_fwd=True)
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.9, rx=0.0, ry=0.9, y_held=False, now=0.0
    )
    assert cmd == LocomotionCmd("fwd_step", "default")


# ---------------------------------------------------------------------------
# Default-disabled lean / torso primitives. The curated planner bins for
# ``lean_fwd_*`` and ``torso_*_30deg`` are *replay* clips: the body
# leans / twists into the pose and immediately blends back to standing,
# instead of holding the static pose. That confuses operators (you push
# the stick and the body flicks then snaps back), so both are disabled
# by default. ``turn_*`` (hard rx) is intentionally unaffected because
# its bin actually completes a discrete yaw step.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ry", [0.35, 0.55, 0.85, 1.0])
def test_lean_fwd_disabled_by_default_emits_idle(ry):
    """With the default ``enable_lean_fwd=False``, even a hard forward
    R-stick push must land on idle (no spurious lean_fwd commands)."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.0, ry=ry, y_held=False, now=0.0
    )
    assert cmd == LocomotionCmd("idle", "default")


@pytest.mark.parametrize("rx", [0.35, 0.50, 0.59, -0.35, -0.50, -0.59])
def test_torso_disabled_by_default_emits_idle(rx):
    """With the default ``enable_torso=False``, soft R-stick X push
    falls through to idle instead of emitting torso_left/right_30deg."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=rx, ry=0.0, y_held=False, now=0.0
    )
    assert cmd == LocomotionCmd("idle", "default")


@pytest.mark.parametrize(
    "rx,expected",
    [
        # Hard rx still fires a turn even with torso disabled. This is
        # the operator's primary "pivot the robot" path for now.
        (0.75,  LocomotionCmd("turn_right", "deg_45")),
        (0.95,  LocomotionCmd("turn_right", "deg_45")),
        (-0.75, LocomotionCmd("turn_left",  "deg_45")),
        (-0.95, LocomotionCmd("turn_left",  "deg_45")),
    ],
)
def test_turns_unaffected_by_torso_disable(rx, expected):
    dec = _make_loco()  # torso disabled, lean disabled
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=rx, ry=0.0, y_held=False, now=0.0
    )
    assert cmd == expected


def test_x_held_90deg_turn_unaffected_by_torso_disable():
    """X + hard rx still upgrades to a 90deg turn when torso is disabled."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.0, rx=0.95, ry=0.0,
        y_held=False, now=0.0, x_held=True,
    )
    assert cmd == LocomotionCmd("turn_right", "deg_90")


def test_lean_disabled_does_not_steal_dominant_axis_from_turn():
    """When lean is disabled and ry dominates over rx, we must still
    emit idle (not silently fall through to the torso/turn branch).
    Otherwise an operator pushing the right stick toward the upper
    corner would unexpectedly trigger a turn."""
    dec = _make_loco()  # both disabled
    cmd = dec.decode_locomotion(
        # ry dominant, rx active hard, both above deadzone:
        # lean would have fired large; turn would have fired on rx.
        # With lean disabled, ry-dominant branch returns idle and
        # we do NOT fall through to rx -> turn.
        lx=0.0, ly=0.0, rx=0.65, ry=0.85, y_held=False, now=0.0,
    )
    assert cmd == LocomotionCmd("idle", "default")


def test_left_stick_still_wins_when_right_stick_lean_disabled():
    """Left-stick precedence still beats right-stick even if right's
    branches all collapse to idle (regression guard for the disable)."""
    dec = _make_loco()
    cmd = dec.decode_locomotion(
        lx=0.0, ly=0.9, rx=0.0, ry=0.9, y_held=False, now=0.0,
    )
    assert cmd == LocomotionCmd("fwd_step", "default")


# ---------------------------------------------------------------------------
# Threshold validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.10, 0.30, 1.0, 1.5])
def test_invalid_turn_threshold(bad):
    """turn_threshold must satisfy stick_deadzone(0.30) < t < 1.0."""
    with pytest.raises(ValueError):
        IntentDecoder(turn_threshold=bad)


@pytest.mark.parametrize(
    "medium,large",
    [
        (0.10, 0.50),  # medium <= deadzone
        (0.50, 0.40),  # medium > large
        (0.50, 1.0),   # large not strictly < 1
        (0.30, 0.50),  # medium == deadzone
    ],
)
def test_invalid_lean_thresholds(medium, large):
    with pytest.raises(ValueError):
        IntentDecoder(
            lean_medium_threshold=medium, lean_large_threshold=large
        )


def test_y_held_with_crouch_enabled_emits_crouch():
    """Opt-in flag re-enables the legacy Y-as-crouch mapping for offline
    planner experiments."""
    dec = IntentDecoder(
        stick_deadzone=0.30, chord_debounce_s=0.0, enable_crouch=True
    )
    dec.update_mode(_abxy_chord(), now=0.0)
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=0.0)
    assert cmd == LocomotionCmd("crouch", "medium")


def test_repeat_interval_zero_only_emits_on_change():
    dec = _make_loco()
    cmd = dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("fwd_step", "default")
    # held: no re-emit
    for t in (0.02, 0.5, 1.0, 5.0):
        assert dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=t) is None


def test_repeat_interval_positive_re_emits_after_threshold():
    dec = IntentDecoder(stick_deadzone=0.30, repeat_interval_s=0.5)
    dec.update_mode(_abxy_chord())
    cmd = dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("fwd_step", "default")
    assert dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.4) is None
    assert dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.5) == LocomotionCmd(
        "fwd_step", "default"
    )


def test_change_emits_and_resets_repeat_timer():
    dec = IntentDecoder(stick_deadzone=0.30, repeat_interval_s=0.5)
    dec.update_mode(_abxy_chord())
    dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    cmd = dec.decode_locomotion(0.9, 0.0, 0.0, 0.0, False, now=0.1)
    assert cmd == LocomotionCmd("side_right", "default")
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, False, now=0.2)
    assert cmd == LocomotionCmd("idle", "default")


def test_mode_flip_resets_emit_memory():
    """After a mode flip, the next entry into LOCOMOTION should re-emit
    even if the stick state matches the last command before the flip."""
    dec = _make_loco()
    dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    # ARM_MANIPULATION
    dec.update_mode(_b_single())
    # back to LOCOMOTION
    dec.update_mode(_b_single())
    cmd = dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=2.0)
    assert cmd == LocomotionCmd("fwd_step", "default")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5])
def test_invalid_deadzone(bad):
    with pytest.raises(ValueError):
        IntentDecoder(stick_deadzone=bad)


def test_invalid_repeat_interval():
    with pytest.raises(ValueError):
        IntentDecoder(repeat_interval_s=-0.1)


def test_invalid_chord_debounce():
    with pytest.raises(ValueError):
        IntentDecoder(chord_debounce_s=-0.1)


# ---------------------------------------------------------------------------
# Chord debounce: A+B+X+Y physically holds Y down. Without a quiet window
# after the OFF -> LOCOMOTION transition the operator's chord-release
# fires a crouch (Y mapped to crouch) on the very next tick and the robot
# tips over. These tests pin the suppression behavior in place.
# ---------------------------------------------------------------------------


def test_chord_debounce_suppresses_y_held_after_locomotion_entry():
    dec = IntentDecoder(stick_deadzone=0.30, chord_debounce_s=0.5)
    transition = dec.update_mode(_abxy_chord(), now=0.0)
    assert transition is not None
    assert dec.mode is StreamMode.LOCOMOTION
    # Within the quiet window: even with Y still held, decoder forces idle.
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=0.1)
    assert cmd == LocomotionCmd("idle", "default")
    # Sticks deflected during the chord release also get squashed.
    assert dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, y_held=False, now=0.2) is None


def test_chord_debounce_expires_and_y_held_resumes_crouch():
    """When crouch is explicitly enabled, the chord debounce only
    suppresses Y for its window; after expiry, Y-held emits crouch."""
    dec = IntentDecoder(
        stick_deadzone=0.30, chord_debounce_s=0.5, enable_crouch=True
    )
    dec.update_mode(_abxy_chord(), now=0.0)
    # Burn the quiet window with the forced idle emit.
    dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=0.1)
    # After the deadline (0.0 + 0.5), a deliberate Y-hold yields crouch.
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=0.6)
    assert cmd == LocomotionCmd("crouch", "medium")


def test_chord_debounce_not_armed_without_now():
    """Back-compat: callers that don't pass ``now`` get the legacy
    behavior (no quiet window). Existing tests in this file rely on
    that, so we lock it in explicitly."""
    dec = IntentDecoder(
        stick_deadzone=0.30, chord_debounce_s=0.5, enable_crouch=True
    )
    dec.update_mode(_abxy_chord())  # no now=
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=0.0)
    assert cmd == LocomotionCmd("crouch", "medium")


def test_chord_debounce_zero_disables_suppression():
    dec = IntentDecoder(
        stick_deadzone=0.30, chord_debounce_s=0.0, enable_crouch=True
    )
    dec.update_mode(_abxy_chord(), now=0.0)
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=0.0)
    assert cmd == LocomotionCmd("crouch", "medium")


def test_chord_debounce_re_arms_on_each_chord_transition():
    """Each chord transition (re-)arms the quiet window; a chord into
    OFF then back into LOCOMOTION must still suppress chord-Y."""
    dec = IntentDecoder(
        stick_deadzone=0.30, chord_debounce_s=0.5, enable_crouch=True
    )
    # OFF -> LOCOMOTION
    dec.update_mode(_abxy_chord(), now=0.0)
    dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=0.6)  # past window
    # LOCOMOTION -> OFF (chord again, debounce should re-arm even
    # though decode_locomotion will short-circuit on mode mismatch)
    dec.update_mode(_abxy_chord(), now=10.0)
    assert dec.mode is StreamMode.OFF
    # OFF -> LOCOMOTION
    dec.update_mode(_abxy_chord(), now=20.0)
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, y_held=True, now=20.1)
    assert cmd == LocomotionCmd("idle", "default")


# ---------------------------------------------------------------------------
# Sidecar introspection
# ---------------------------------------------------------------------------


def test_last_emitted_tracks_recent_command():
    dec = _make_loco()
    assert dec.last_emitted() is None
    dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    assert dec.last_emitted() == LocomotionCmd("fwd_step", "default")
    dec.decode_locomotion(-0.9, 0.0, 0.0, 0.0, False, now=0.1)
    assert dec.last_emitted() == LocomotionCmd("side_left", "default")


# ---------------------------------------------------------------------------
# Continuous-locomotion path (enable_continuous_locomotion=True)
# ---------------------------------------------------------------------------


def _make_loco_continuous(
    *,
    stick_deadzone: float = 0.30,
    continuous_stick_threshold: float = 0.02,
    continuous_yaw_max: float = 1.0,
) -> IntentDecoder:
    # NOTE: default ``continuous_yaw_max=1.0`` here so the existing
    # yaw-axis tests keep the un-clamped behaviour they were written
    # against. The new clamp-specific tests below pass an explicit
    # value to exercise the cap.
    dec = IntentDecoder(
        stick_deadzone=stick_deadzone,
        enable_continuous_locomotion=True,
        continuous_stick_threshold=continuous_stick_threshold,
        continuous_yaw_max=continuous_yaw_max,
    )
    dec.update_mode(_abxy_chord())
    assert dec.mode is StreamMode.LOCOMOTION
    return dec


def test_continuous_neutral_stick_emits_idle():
    """All three sticks inside the deadzone -> idle (not locomotion)."""
    dec = _make_loco_continuous()
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("idle", "default")


def test_continuous_fwd_stick_carries_normalised_deflection():
    """Forward L-stick beyond deadzone emits locomotion / continuous with
    a positive stick_fwd in (0, 1]. Lateral / yaw stay 0 because lx/rx
    are inside the deadzone."""
    dec = _make_loco_continuous()
    cmd = dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "locomotion"
    assert cmd.magnitude == "continuous"
    assert cmd.stick_fwd > 0.0
    assert cmd.stick_fwd <= 1.0
    assert cmd.stick_side == 0.0
    assert cmd.stick_yaw == 0.0


def test_continuous_back_stick_carries_negative_fwd():
    dec = _make_loco_continuous()
    cmd = dec.decode_locomotion(0.0, -0.9, 0.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "locomotion"
    assert cmd.stick_fwd < 0.0
    assert cmd.stick_fwd >= -1.0


def test_continuous_side_stick_carries_side_only():
    """Right L-stick deflection -> positive stick_side, others 0."""
    dec = _make_loco_continuous()
    cmd = dec.decode_locomotion(0.9, 0.0, 0.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "locomotion"
    assert cmd.stick_side > 0.0
    assert cmd.stick_fwd == 0.0
    assert cmd.stick_yaw == 0.0


def test_continuous_yaw_stick_carries_yaw_only():
    """R-stick deflection -> stick_yaw, sign matches rx direction."""
    dec = _make_loco_continuous()
    cmd = dec.decode_locomotion(0.0, 0.0, 0.9, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "locomotion"
    assert cmd.stick_yaw > 0.0
    cmd = dec.decode_locomotion(0.0, 0.0, -0.9, 0.0, False, now=0.1)
    assert cmd is not None
    assert cmd.intent == "locomotion"
    assert cmd.stick_yaw < 0.0


def test_continuous_deadzone_clamps_below_threshold():
    """|stick| <= deadzone produces 0 deflection on that axis."""
    dec = _make_loco_continuous(stick_deadzone=0.30)
    # 0.2 is inside the 0.3 deadzone on every axis; should remain idle.
    cmd = dec.decode_locomotion(0.2, 0.2, 0.2, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("idle", "default")


def test_continuous_just_past_deadzone_emits_small_deflection():
    """Outside the deadzone the rescaled deflection should be small but
    non-zero, NOT snap to 1.0 -- the operator's thumb owns analog
    control here."""
    dec = _make_loco_continuous(stick_deadzone=0.30)
    cmd = dec.decode_locomotion(0.0, 0.35, 0.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "locomotion"
    # (0.35 - 0.30) / (1 - 0.30) ~= 0.071
    assert 0.0 < cmd.stick_fwd < 0.2


def test_continuous_full_deflection_saturates_to_unity():
    """|stick| = 1.0 -> normalised deflection = 1.0 (full speed)."""
    dec = _make_loco_continuous(stick_deadzone=0.30)
    cmd = dec.decode_locomotion(0.0, 1.0, 0.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.stick_fwd == pytest.approx(1.0, abs=1e-6)


def test_continuous_combined_stick_carries_all_three_axes():
    """L-stick fwd + L-stick side + R-stick yaw should populate all
    three stick_* fields in a single locomotion command."""
    dec = _make_loco_continuous()
    cmd = dec.decode_locomotion(0.7, 0.7, 0.7, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "locomotion"
    assert cmd.stick_fwd > 0.0
    assert cmd.stick_side > 0.0
    assert cmd.stick_yaw > 0.0


def test_continuous_repeat_suppressed_when_stick_unchanged():
    """Steady-state stick -> first tick emits, subsequent ticks suppress."""
    dec = _make_loco_continuous()
    first = dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    assert first is not None
    repeat = dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.01)
    assert repeat is None


def test_continuous_micro_jitter_below_threshold_does_not_emit():
    """Sub-threshold stick changes are filtered by _is_significant_change."""
    dec = _make_loco_continuous(continuous_stick_threshold=0.05)
    dec.decode_locomotion(0.0, 0.90, 0.0, 0.0, False, now=0.0)
    # 0.91 is well within the 0.05 stick-delta threshold of 0.90
    repeat = dec.decode_locomotion(0.0, 0.91, 0.0, 0.0, False, now=0.02)
    assert repeat is None


def test_continuous_significant_change_re_emits():
    dec = _make_loco_continuous(continuous_stick_threshold=0.05)
    dec.decode_locomotion(0.0, 0.50, 0.0, 0.0, False, now=0.0)
    out = dec.decode_locomotion(0.0, 0.80, 0.0, 0.0, False, now=0.02)
    assert out is not None
    assert out.intent == "locomotion"


def test_continuous_off_mode_disables_continuous_path():
    """``enable_continuous_locomotion=False`` (default) keeps the
    bucketed fwd_step / side_* / turn_* path that the heuristic
    planner consumes. Regression for the heuristic-mode wrapper."""
    dec = _make_loco()  # default: continuous disabled
    cmd = dec.decode_locomotion(0.0, 0.9, 0.0, 0.0, False, now=0.0)
    assert cmd == LocomotionCmd("fwd_step", "default")


# ---------------------------------------------------------------------------
# continuous_yaw_max clamp -- operator-feel knob that caps how much of
# the planner's continuous-mode yaw-rate ceiling a full R-stick
# deflection actually requests. Lives on the teleop side so the planner
# doesn't have to know about operator-comfort settings.
# ---------------------------------------------------------------------------


def test_continuous_yaw_max_default_caps_full_stick_to_half():
    """ManagerConfig wires ``intent_continuous_yaw_max=0.5`` as the
    teleop-side default; a full R-stick deflection should request 0.5
    (half of the planner's continuous yaw ceiling). This is the
    primary fix for the 2026-05-30 'brief stick burst -> overshoot'
    operator report.
    """
    dec = _make_loco_continuous(continuous_yaw_max=0.5)
    cmd = dec.decode_locomotion(0.0, 0.0, 1.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "locomotion"
    assert cmd.stick_yaw == pytest.approx(0.5, abs=1e-6)
    cmd_left = dec.decode_locomotion(0.0, 0.0, -1.0, 0.0, False, now=0.05)
    assert cmd_left is not None
    assert cmd_left.stick_yaw == pytest.approx(-0.5, abs=1e-6)


def test_continuous_yaw_max_scales_linearly_with_stick():
    """Linear scaling of post-deadzone deflection by the cap means a
    half-stick command produces half the capped magnitude.

    This is what gives the operator analog control over turn rate
    *within* the safe envelope, rather than forcing them to slam the
    stick to get any motion or back off to avoid overshoot.
    """
    dec = _make_loco_continuous(stick_deadzone=0.30, continuous_yaw_max=0.5)
    # rx = 0.65 -> rescaled = (0.65 - 0.30) / 0.70 = 0.50 -> after cap = 0.25
    cmd = dec.decode_locomotion(0.0, 0.0, 0.65, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.stick_yaw == pytest.approx(0.25, abs=1e-6)


def test_continuous_yaw_max_does_not_affect_fwd_or_side():
    """The clamp is yaw-only: full L-stick forward still produces
    stick_fwd=1.0 even with a tight yaw cap. Forward / lateral
    motion doesn't suffer from the same 'brief burst -> overshoot'
    failure mode that prompted the yaw clamp, so we keep those axes
    at full operator authority.
    """
    dec = _make_loco_continuous(continuous_yaw_max=0.25)
    cmd = dec.decode_locomotion(1.0, 1.0, 0.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.stick_fwd == pytest.approx(1.0, abs=1e-6)
    assert cmd.stick_side == pytest.approx(1.0, abs=1e-6)


def test_continuous_yaw_max_unity_restores_legacy_full_deflection():
    """``continuous_yaw_max=1.0`` reproduces the pre-fix behaviour --
    full R-stick = full planner ceiling -- for A/B regression runs."""
    dec = _make_loco_continuous(continuous_yaw_max=1.0)
    cmd = dec.decode_locomotion(0.0, 0.0, 1.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.stick_yaw == pytest.approx(1.0, abs=1e-6)


def test_continuous_yaw_max_invalid_values_rejected():
    """Out-of-range values must fail loudly at construction time --
    catching typos like ``continuous_yaw_max=50`` (operator meant
    50% but typed an absolute number) rather than silently amplifying
    yaw 50x. ``0`` is also rejected because a clamp of zero would
    silently disable turns entirely; operators wanting that behaviour
    should pass ``--no-enable-continuous-locomotion``.
    """
    with pytest.raises(ValueError):
        IntentDecoder(enable_continuous_locomotion=True, continuous_yaw_max=0.0)
    with pytest.raises(ValueError):
        IntentDecoder(enable_continuous_locomotion=True, continuous_yaw_max=-0.5)
    with pytest.raises(ValueError):
        IntentDecoder(enable_continuous_locomotion=True, continuous_yaw_max=1.5)
    with pytest.raises(ValueError):
        IntentDecoder(enable_continuous_locomotion=True, continuous_yaw_max=50.0)


def test_continuous_yaw_max_clamps_each_tick_independently():
    """The clamp is stateless -- it's applied at the dispatch site, not
    as a running smoother -- so successive ticks with varying rx all
    see the same cap. Regression guard against accidentally moving
    the clamp into a stateful EWMA (which would change the operator
    contract from "max amplitude" to "max amplitude after a few
    frames" and re-introduce the very brief-burst problem this knob
    exists to solve).
    """
    dec = _make_loco_continuous(continuous_yaw_max=0.5)
    # Push hard, release, push hard -- each "push" tick should report
    # the same clamped magnitude.
    out_a = dec.decode_locomotion(0.0, 0.0, 1.0, 0.0, False, now=0.0)
    out_b = dec.decode_locomotion(0.0, 0.0, 1.0, 0.0, False, now=0.5)
    assert out_a is not None
    assert out_a.stick_yaw == pytest.approx(0.5, abs=1e-6)
    # ``out_b`` is None because _is_significant_change suppresses
    # identical repeats; we don't care about the emit gating here,
    # only that nothing inside the decoder has snuck in a smoother.
    # Force a re-emit with a different yaw magnitude:
    out_c = dec.decode_locomotion(0.0, 0.0, -1.0, 0.0, False, now=1.0)
    assert out_c is not None
    assert out_c.stick_yaw == pytest.approx(-0.5, abs=1e-6)
