"""Tests for the continuous-torso path in :mod:`intent_decoder`.

Pins the right-stick -> ``hold_torso`` mapping the Quest 3 manager
relies on to drive STATIC_HOLD. Covers:

  - ``ry`` -> positive ``waist_pitch_deg`` (forward lean only).
  - ``rx`` (no A) -> negative ``waist_yaw_deg`` (twist).
  - ``rx`` (A held) -> negative ``waist_roll_deg`` (lateral lean).
  - Deadzone yields neutral target.
  - Soft-band stick produces ``hold_torso`` (NOT discrete bins).
  - Hard ``rx`` deflection still wins as ``turn_*`` (operator wants to
    pivot, not lean).
  - Throttle: small noise around an existing target does NOT re-emit;
    a perceptible change DOES re-emit.
  - ``continuous_waist_target`` returns the same clamped (pitch, roll,
    yaw) tuple regardless of mode (used by the manager for B-press
    latching).
  - Backward-compat: ``enable_continuous_torso=False`` (default)
    preserves the legacy decoder semantics covered by
    ``test_intent_decoder.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.teleop.vr.button_state_machine import ButtonEvents  # noqa: E402
from gear_sonic.utils.teleop.vr.intent_decoder import (  # noqa: E402
    IntentDecoder,
    LocomotionCmd,
    StreamMode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _abxy_chord() -> ButtonEvents:
    return ButtonEvents(
        a_pressed=True, b_pressed=True, x_pressed=True, y_pressed=True,
        ab_pressed=True, xy_pressed=True, ax_pressed=True, by_pressed=True,
        abxy_pressed=True,
    )


def _make_decoder(
    *,
    deadzone: float = 0.30,
    enable_continuous_torso: bool = True,
    hold_target_threshold_deg: float = 0.5,
    max_pitch: float = 20.0,
    max_roll: float = 10.0,
    max_yaw: float = 40.0,
    chord_debounce_s: float = 0.0,
) -> IntentDecoder:
    dec = IntentDecoder(
        stick_deadzone=deadzone,
        chord_debounce_s=chord_debounce_s,
        enable_continuous_torso=enable_continuous_torso,
        hold_target_threshold_deg=hold_target_threshold_deg,
        max_waist_pitch_deg=max_pitch,
        max_waist_roll_deg=max_roll,
        max_waist_yaw_deg=max_yaw,
    )
    dec.update_mode(_abxy_chord(), now=0.0)
    assert dec.mode is StreamMode.LOCOMOTION
    return dec


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_max_waist_negative_pitch_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(max_waist_pitch_deg=-1.0)


def test_max_waist_negative_roll_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(max_waist_roll_deg=-1.0)


def test_max_waist_negative_yaw_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(max_waist_yaw_deg=-1.0)


def test_negative_threshold_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(hold_target_threshold_deg=-0.1)


# ---------------------------------------------------------------------------
# Stick -> waist target conventions
# ---------------------------------------------------------------------------


def test_neutral_sticks_emit_neutral_hold_torso_when_continuous_on() -> None:
    """Documented behavior: continuous mode emits ``hold_torso`` with the
    target at (0, 0, 0) when the left stick is neutral. The planner's
    STATIC_HOLD entry path then slews to neutral (which is the default
    stand pose), and a subsequent non-hold cmd blends back to idle."""
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, 0.0, 0.0, False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.magnitude == "continuous"
    assert (cmd.waist_pitch_deg, cmd.waist_roll_deg, cmd.waist_yaw_deg) == (
        0.0, 0.0, 0.0,
    )


def test_ry_positive_emits_pitch_only() -> None:
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=0.5, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.magnitude == "continuous"
    assert cmd.waist_pitch_deg > 0.0
    assert cmd.waist_roll_deg == 0.0
    assert cmd.waist_yaw_deg == 0.0


def test_ry_negative_does_not_lean_back() -> None:
    """Backward lean has no primitive; the decoder must clamp to 0."""
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=-0.9, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg == 0.0


def test_ry_full_deflection_clamped_to_max_pitch() -> None:
    dec = _make_decoder(max_pitch=20.0)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=1.0, y_held=False, now=0.0)
    assert cmd.waist_pitch_deg == pytest.approx(20.0)


def test_rx_positive_no_a_emits_negative_yaw() -> None:
    """Stick right -> torso_right -> negative waist_yaw (matches existing
    ``torso_right_*`` peak_deg sign convention)."""
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert cmd.intent == "hold_torso"
    assert cmd.waist_yaw_deg < 0.0
    assert cmd.waist_roll_deg == 0.0
    assert cmd.waist_pitch_deg == 0.0


def test_rx_negative_no_a_emits_positive_yaw() -> None:
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=-0.5, ry=0.0, y_held=False, now=0.0)
    assert cmd.intent == "hold_torso"
    assert cmd.waist_yaw_deg > 0.0


def test_rx_with_a_held_remaps_to_roll_not_yaw() -> None:
    dec = _make_decoder()
    cmd = dec.decode_locomotion(
        0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0, a_held=True
    )
    assert cmd.intent == "hold_torso"
    assert cmd.waist_roll_deg < 0.0
    assert cmd.waist_yaw_deg == 0.0


def test_rx_with_a_held_left_emits_positive_roll() -> None:
    dec = _make_decoder()
    cmd = dec.decode_locomotion(
        0.0, 0.0, rx=-0.5, ry=0.0, y_held=False, now=0.0, a_held=True
    )
    assert cmd.intent == "hold_torso"
    assert cmd.waist_roll_deg > 0.0


def test_combined_pitch_and_yaw_compose() -> None:
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.5, y_held=False, now=0.0)
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg > 0.0
    assert cmd.waist_yaw_deg < 0.0
    assert cmd.waist_roll_deg == 0.0


def test_rx_deadzone_yields_zero_yaw() -> None:
    """In-deadzone rx still produces hold_torso (continuous mode default),
    but with a zero yaw target."""
    dec = _make_decoder(deadzone=0.3)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.2, ry=0.0, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_yaw_deg == 0.0
    assert cmd.waist_roll_deg == 0.0
    assert cmd.waist_pitch_deg == 0.0


def test_max_clamping_per_axis() -> None:
    """Hard rx wins as turn_*; clamp test uses soft rx that bypasses the
    turn threshold."""
    dec = _make_decoder(max_pitch=15.0, max_roll=8.0, max_yaw=30.0)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=1.0, y_held=False, now=0.0)
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg == pytest.approx(15.0)
    cmd2 = dec.decode_locomotion(
        0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=1.0,
    )
    assert cmd2.intent == "hold_torso"
    assert cmd2.waist_yaw_deg < 0.0
    cmd3 = dec.decode_locomotion(
        0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=2.0, a_held=True
    )
    assert cmd3.intent == "hold_torso"
    assert cmd3.waist_roll_deg < 0.0


# ---------------------------------------------------------------------------
# Hard turn still pre-empts continuous hold
# ---------------------------------------------------------------------------


def test_hard_rx_right_still_emits_turn_right() -> None:
    """Past the turn threshold the operator wants to pivot, not lean."""
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.95, ry=0.0, y_held=False, now=0.0)
    assert cmd == LocomotionCmd("turn_right", "deg_45")


def test_hard_rx_left_still_emits_turn_left() -> None:
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=-0.95, ry=0.0, y_held=False, now=0.0)
    assert cmd == LocomotionCmd("turn_left", "deg_45")


def test_x_held_promotes_hard_rx_to_90() -> None:
    dec = _make_decoder()
    cmd = dec.decode_locomotion(
        0.0, 0.0, rx=0.95, ry=0.0, y_held=False, now=0.0, x_held=True
    )
    assert cmd == LocomotionCmd("turn_right", "deg_90")


def test_left_stick_walk_still_wins_in_continuous_mode() -> None:
    """Continuous torso must NOT block left-stick locomotion."""
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.9, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert cmd == LocomotionCmd("fwd_step", "default")


# ---------------------------------------------------------------------------
# Throttling / dedup
# ---------------------------------------------------------------------------


def test_identical_hold_target_throttled() -> None:
    dec = _make_decoder()
    first = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert first is not None
    repeat = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.02)
    assert repeat is None, "identical hold_torso target must not re-emit"


def test_subthreshold_hold_target_throttled() -> None:
    """Sub-threshold noise around an active target must not re-emit."""
    dec = _make_decoder(hold_target_threshold_deg=0.5)
    first = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert first is not None
    # 0.501 vs 0.500 -> tiny scaled-axis difference (~0.04 deg). Below 0.5.
    repeat = dec.decode_locomotion(
        0.0, 0.0, rx=0.501, ry=0.0, y_held=False, now=0.02
    )
    assert repeat is None


def test_significant_target_change_re_emits() -> None:
    dec = _make_decoder(hold_target_threshold_deg=0.5)
    first = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert first is not None
    later = dec.decode_locomotion(0.0, 0.0, rx=0.7, ry=0.0, y_held=False, now=0.02)
    assert later is not None
    assert later.waist_yaw_deg != first.waist_yaw_deg


def test_intent_change_re_emits_even_under_threshold() -> None:
    """Switching axes (yaw -> roll, etc.) must always emit."""
    dec = _make_decoder(hold_target_threshold_deg=10.0)  # huge threshold
    first = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert first is not None
    # Tiny rx change but A flips on -> roll axis instead of yaw.
    later = dec.decode_locomotion(
        0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.02, a_held=True
    )
    assert later is not None


def test_repeat_interval_re_emits_after_window_elapsed() -> None:
    dec = IntentDecoder(
        stick_deadzone=0.3,
        chord_debounce_s=0.0,
        enable_continuous_torso=True,
        repeat_interval_s=0.05,
        hold_target_threshold_deg=0.5,
    )
    dec.update_mode(_abxy_chord(), now=0.0)
    first = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert first is not None
    none1 = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.02)
    assert none1 is None
    repeat = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.10)
    assert repeat is not None


# ---------------------------------------------------------------------------
# continuous_waist_target API used for latching
# ---------------------------------------------------------------------------


def test_continuous_waist_target_returns_clamped_tuple() -> None:
    dec = _make_decoder(max_pitch=20.0, max_yaw=40.0)
    pitch, roll, yaw = dec.continuous_waist_target(rx=0.5, ry=0.5)
    assert pitch > 0.0
    assert roll == 0.0
    assert yaw < 0.0


def test_continuous_waist_target_neutral_for_in_deadzone() -> None:
    dec = _make_decoder(deadzone=0.3)
    pitch, roll, yaw = dec.continuous_waist_target(rx=0.1, ry=0.1)
    assert (pitch, roll, yaw) == (0.0, 0.0, 0.0)


def test_continuous_waist_target_a_held_swaps_roll_for_yaw() -> None:
    dec = _make_decoder()
    pitch_a, roll_a, yaw_a = dec.continuous_waist_target(
        rx=0.5, ry=0.0, a_held=True,
    )
    pitch_b, roll_b, yaw_b = dec.continuous_waist_target(
        rx=0.5, ry=0.0, a_held=False,
    )
    assert roll_a != 0.0
    assert yaw_a == 0.0
    assert roll_b == 0.0
    assert yaw_b != 0.0


def test_continuous_waist_target_works_independent_of_mode() -> None:
    """Used at B-press latch time: must NOT depend on decoder mode."""
    dec = IntentDecoder(
        stick_deadzone=0.3,
        chord_debounce_s=0.0,
        enable_continuous_torso=False,  # off!
    )
    assert dec.mode is StreamMode.OFF
    pitch, roll, yaw = dec.continuous_waist_target(rx=0.5, ry=0.5)
    assert pitch > 0.0
    assert yaw < 0.0


# ---------------------------------------------------------------------------
# Backward compatibility (continuous_torso=False)
# ---------------------------------------------------------------------------


def test_continuous_off_falls_back_to_legacy_idle_for_soft_rx() -> None:
    """With continuous off, soft rx (below turn_threshold) and torso
    disabled returns idle (legacy behavior)."""
    dec = _make_decoder(enable_continuous_torso=False)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert cmd == LocomotionCmd("idle", "default")


def test_continuous_off_hard_rx_still_emits_turn() -> None:
    dec = _make_decoder(enable_continuous_torso=False)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.95, ry=0.0, y_held=False, now=0.0)
    assert cmd == LocomotionCmd("turn_right", "deg_45")
