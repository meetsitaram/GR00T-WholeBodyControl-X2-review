"""Tests for the continuous-torso path in :mod:`intent_decoder`.

Pins the stick -> ``hold_torso`` mapping the Quest 3 manager relies on
to drive STATIC_HOLD / kplanner waist overlay. Covers:

  - ``ry`` -> signed ``waist_pitch_deg`` (forward + backward lean,
    v7.4 -- pre-v7.4 the negative side was clamped to 0).
  - ``rx`` -> negative ``waist_yaw_deg`` (twist). v7.2: A-held no
    longer remaps rx to roll because the operator's right thumb
    cannot reach A while driving the R-stick on the same controller.
  - Yaw-priority cone (v7.4): when |rx| dominates |ry| past
    ``pitch_dominance_ratio``, the pitch axis is suppressed so a
    near-pure twist doesn't accidentally lean the body.
  - Deadzone yields neutral target.
  - Soft-band stick produces ``hold_torso`` (NOT discrete bins).
  - Hard ``rx`` deflection still wins as ``turn_*`` (operator wants to
    pivot, not lean).
  - Throttle: small noise around an existing target does NOT re-emit;
    a perceptible change DOES re-emit.
  - ``continuous_waist_target`` returns the clamped 4-tuple ``(pitch,
    roll, yaw, hip_height_m)`` regardless of mode (used by the manager
    for B-press latching).
  - Mode gating (v7.2 + v7.4): in ARM_MANIPULATION the decoder allows
    ``hold_torso`` through (lean / twist / roll / squat all steer the
    waist for extra arm reach + reach envelope) but filters out walk /
    step / turn commands so the base never slides under the operator's
    IK targets. In LOCOMOTION the L-stick still owns step / side /
    walk; the L-stick contribution to roll + height is suppressed.
    In OFF nothing flows.
  - ARM_MAN L-stick (v7.4): roll (lx) + continuous hip height (ly,
    squat / stand). Roll-priority cone gates height against roll.
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
    pitch_dominance_ratio: float = 0.0,  # tests assume axes are independent
    height_dominance_ratio: float = 0.0,
    enable_arm_man_lstick: bool = True,
    max_height_down_m: float = 0.09,
    max_height_up_m: float = 0.04,
    default_hip_height_m: float = 0.687,
    hold_height_threshold_m: float = 0.005,
) -> IntentDecoder:
    """Build a decoder with the dominance cones disabled by default so
    legacy tests (which exercise individual axes one at a time) keep
    their semantics. Tests that exercise the cones pass non-zero
    ratios explicitly."""
    dec = IntentDecoder(
        stick_deadzone=deadzone,
        chord_debounce_s=chord_debounce_s,
        enable_continuous_torso=enable_continuous_torso,
        hold_target_threshold_deg=hold_target_threshold_deg,
        max_waist_pitch_deg=max_pitch,
        max_waist_roll_deg=max_roll,
        max_waist_yaw_deg=max_yaw,
        pitch_dominance_ratio=pitch_dominance_ratio,
        height_dominance_ratio=height_dominance_ratio,
        enable_arm_man_lstick=enable_arm_man_lstick,
        max_height_down_m=max_height_down_m,
        max_height_up_m=max_height_up_m,
        default_hip_height_m=default_hip_height_m,
        hold_height_threshold_m=hold_height_threshold_m,
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


def test_ry_negative_emits_negative_pitch_v7_4() -> None:
    """v7.4: backward lean is now bidirectional. ry < 0 produces
    negative ``waist_pitch_deg``; the kplanner's waist overlay (and
    the heuristic STATIC_HOLD path) accept signed pitch and lean the
    body the requested direction."""
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=-0.9, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg < 0.0


def test_ry_negative_full_deflection_clamped_to_minus_max_pitch() -> None:
    """Backward lean is symmetric with forward lean: full -ry deflection
    clamps to ``-max_waist_pitch_deg``."""
    dec = _make_decoder(max_pitch=20.0)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=-1.0, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.waist_pitch_deg == pytest.approx(-20.0)


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


def test_a_held_no_longer_remaps_rx_to_roll() -> None:
    """v7.2: the A-held -> roll modifier is gone. With A held, rx still
    drives yaw and roll stays at 0 (the operator's right thumb owns the
    R-stick; A on the same controller is unreachable mid-lean, so
    keeping the modifier in the decoder would be a footgun)."""
    dec = _make_decoder()
    cmd = dec.decode_locomotion(
        0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0, a_held=True
    )
    assert cmd.intent == "hold_torso"
    assert cmd.waist_roll_deg == 0.0
    assert cmd.waist_yaw_deg < 0.0  # rx still drives yaw, ignoring A


def test_a_held_left_no_longer_emits_roll() -> None:
    """Symmetric check for left-side rx with A held."""
    dec = _make_decoder()
    cmd = dec.decode_locomotion(
        0.0, 0.0, rx=-0.5, ry=0.0, y_held=False, now=0.0, a_held=True
    )
    assert cmd.intent == "hold_torso"
    assert cmd.waist_roll_deg == 0.0
    assert cmd.waist_yaw_deg > 0.0


def test_roll_axis_not_emitted_from_locomotion_rstick_path() -> None:
    """In LOCOMOTION mode, sweeping the R-stick alone (lx=ly=0) must not
    leak any roll target -- L-stick X-axis owns side-step there.

    v7.4 ARM_MANIPULATION decodes lx as roll, but that is a separate
    test (see ``test_arm_man_lstick_lx_emits_roll``). The wire format
    still reserves the field for scripted demos / future VLA outputs."""
    dec = _make_decoder()
    samples = [
        (0.0, 0.0, False, False),
        (0.5, 0.0, False, False),
        (-0.5, 0.0, False, False),
        (0.0, 0.5, False, False),
        (0.5, 0.5, False, False),
        (0.5, 0.0, True, False),
        (-0.5, 0.5, True, False),
        (0.5, 0.5, True, True),
    ]
    for i, (rx, ry, a_held, x_held) in enumerate(samples):
        cmd = dec.decode_locomotion(
            0.0, 0.0,
            rx=rx, ry=ry, y_held=False, now=float(i),
            a_held=a_held, x_held=x_held,
        )
        if cmd is None or cmd.intent != "hold_torso":
            continue
        assert cmd.waist_roll_deg == 0.0, (
            f"roll leaked from LOCO operator path: rx={rx} ry={ry} "
            f"a_held={a_held} x_held={x_held} -> {cmd}"
        )


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
    turn threshold. v7.2: roll axis no longer driven by the operator
    path so we only clamp-check pitch and yaw."""
    dec = _make_decoder(max_pitch=15.0, max_roll=8.0, max_yaw=30.0)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=1.0, y_held=False, now=0.0)
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg == pytest.approx(15.0)
    cmd2 = dec.decode_locomotion(
        0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=1.0,
    )
    assert cmd2.intent == "hold_torso"
    assert cmd2.waist_yaw_deg < 0.0


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
    """Switching from a discrete bin (turn_*) to hold_torso must always
    emit, regardless of the per-axis threshold (intent change beats
    waist-target dedup). v7.2: this used to test the yaw->roll axis
    flip via A-held; that path is gone, so we exercise the
    hold_torso<->turn_right boundary instead."""
    dec = _make_decoder(hold_target_threshold_deg=10.0)  # huge threshold
    # Soft rx -> hold_torso(yaw=...).
    first = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.0, y_held=False, now=0.0)
    assert first is not None
    assert first.intent == "hold_torso"
    # Hard rx -> turn_right (different intent, must re-emit even
    # though waist-target deltas are technically irrelevant here).
    later = dec.decode_locomotion(0.0, 0.0, rx=0.95, ry=0.0, y_held=False, now=0.02)
    assert later is not None
    assert later == LocomotionCmd("turn_right", "deg_45")


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


def test_continuous_waist_target_returns_clamped_4tuple() -> None:
    dec = _make_decoder(max_pitch=20.0, max_yaw=40.0)
    pitch, roll, yaw, hip_h = dec.continuous_waist_target(rx=0.5, ry=0.5)
    assert pitch > 0.0
    assert roll == 0.0
    assert yaw < 0.0
    # LOCOMOTION mode (default for _make_decoder): L-stick is silent
    # in the waist path, so hip_height_m must be None even though the
    # decoder accepts the lx / ly defaults.
    assert hip_h is None


def test_continuous_waist_target_neutral_for_in_deadzone() -> None:
    dec = _make_decoder(deadzone=0.3)
    pitch, roll, yaw, hip_h = dec.continuous_waist_target(rx=0.1, ry=0.1)
    assert (pitch, roll, yaw) == (0.0, 0.0, 0.0)
    assert hip_h is None


def test_continuous_waist_target_a_held_kwarg_is_ignored() -> None:
    """v7.2: ``continuous_waist_target`` still accepts ``a_held`` for
    backward compatibility (so older callers don't crash) but the
    parameter is now ignored. The result must match the call with the
    kwarg omitted."""
    dec = _make_decoder()
    pitch_a, roll_a, yaw_a, hip_a = dec.continuous_waist_target(
        rx=0.5, ry=0.0, a_held=True,
    )
    pitch_b, roll_b, yaw_b, hip_b = dec.continuous_waist_target(
        rx=0.5, ry=0.0, a_held=False,
    )
    assert (pitch_a, roll_a, yaw_a, hip_a) == (pitch_b, roll_b, yaw_b, hip_b)
    assert roll_a == 0.0
    assert yaw_a < 0.0  # rx still drives yaw, ignoring A


def test_continuous_waist_target_works_independent_of_mode() -> None:
    """Used at B-press latch time: must NOT depend on decoder mode."""
    dec = IntentDecoder(
        stick_deadzone=0.3,
        chord_debounce_s=0.0,
        enable_continuous_torso=False,  # off!
        pitch_dominance_ratio=0.0,
    )
    assert dec.mode is StreamMode.OFF
    pitch, roll, yaw, hip_h = dec.continuous_waist_target(rx=0.5, ry=0.5)
    assert pitch > 0.0
    assert yaw < 0.0
    # OFF mode is not ARM_MANIPULATION, so the L-stick height path
    # is gated off; hip_h must be None regardless of (lx, ly).
    assert hip_h is None


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


# ---------------------------------------------------------------------------
# Mode gating (v7.2): hold_torso flows in ARM_MAN; walk / turn don't
# ---------------------------------------------------------------------------


def _make_arm_man_decoder(**kwargs) -> IntentDecoder:
    """Helper: build a decoder, drive A+B+X+Y to leave OFF, then a single
    B-press to flip LOCOMOTION -> ARM_MANIPULATION. Lets each test pin
    behaviour with the manager-installed mode actually set."""
    dec = _make_decoder(**kwargs)
    assert dec.mode is StreamMode.LOCOMOTION
    b_only = ButtonEvents(
        a_pressed=False, b_pressed=True, x_pressed=False, y_pressed=False,
        ab_pressed=False, xy_pressed=False,
        ax_pressed=False, by_pressed=False,
        abxy_pressed=False,
    )
    transition = dec.update_mode(b_only, now=1.0)
    assert transition is not None
    assert dec.mode is StreamMode.ARM_MANIPULATION
    return dec


def test_arm_man_passes_hold_torso_through() -> None:
    """In ARM_MANIPULATION the right stick must still drive the planner's
    STATIC_HOLD target -- that's the whole point of v7.2 (lean to extend
    arm reach during manipulation)."""
    dec = _make_arm_man_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=0.5, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg > 0.0


def test_arm_man_passes_neutral_hold_torso_through() -> None:
    """Neutral R-stick in ARM_MAN still emits hold_torso(0,0,0). The
    planner slews back to neutral; operator uses R-click freeze if
    they want to lock a non-neutral pose while releasing the stick."""
    dec = _make_arm_man_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert (cmd.waist_pitch_deg, cmd.waist_roll_deg, cmd.waist_yaw_deg) == (
        0.0, 0.0, 0.0,
    )


def test_arm_man_lstick_y_emits_hold_torso_not_walk_v7_4() -> None:
    """v7.4: in ARM_MANIPULATION the L-stick Y axis is owned by the
    waist path (squat / stand) -- it must NEVER emit ``walk`` /
    ``fwd_step`` / ``back_step`` (which would slide the IK reference
    frame out from under the operator's hands). The decoder routes
    L-stick deflections in ARM_MAN through ``_continuous_hold_cmd``
    instead, producing a ``hold_torso`` with a hip-height override."""
    dec = _make_arm_man_decoder()
    cmd = dec.decode_locomotion(0.0, 0.9, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso", f"L-stick Y leaked discrete intent: {cmd}"
    assert cmd.hip_height_m is not None


def test_arm_man_lstick_x_emits_hold_torso_not_side_step_v7_4() -> None:
    """v7.4: L-stick X-axis owns roll in ARM_MANIPULATION (not
    side-step). Same rationale as the Y-axis: discrete locomotion
    bins would slide the IK frame."""
    dec = _make_arm_man_decoder()
    cmd = dec.decode_locomotion(0.9, 0.0, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso", f"L-stick X leaked discrete intent: {cmd}"
    assert cmd.waist_roll_deg > 0.0


def test_arm_man_filters_turn_command() -> None:
    """Hard R-stick X past the turn threshold is a pivot; pivots must
    not fire in ARM_MAN."""
    dec = _make_arm_man_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.95, ry=0.0, y_held=False, now=2.0)
    assert cmd is None, "turn_right leaked into ARM_MANIPULATION"


def test_arm_man_filters_x_held_90_turn() -> None:
    dec = _make_arm_man_decoder()
    cmd = dec.decode_locomotion(
        0.0, 0.0, rx=0.95, ry=0.0, y_held=False, now=2.0, x_held=True,
    )
    assert cmd is None, "X-held 90-deg turn leaked into ARM_MANIPULATION"


def test_arm_man_combined_pitch_yaw_passes() -> None:
    """Composite hold_torso (pitch + yaw together) must flow in ARM_MAN."""
    dec = _make_arm_man_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.5, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg > 0.0
    assert cmd.waist_yaw_deg < 0.0


def test_loco_still_passes_walk_command() -> None:
    """Sanity: relaxing the ARM_MAN gate must NOT regress LOCO behavior."""
    dec = _make_decoder()
    assert dec.mode is StreamMode.LOCOMOTION
    cmd = dec.decode_locomotion(0.0, 0.9, rx=0.0, ry=0.0, y_held=False, now=0.0)
    assert cmd == LocomotionCmd("fwd_step", "default")


def test_loco_still_passes_turn_command() -> None:
    dec = _make_decoder()
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.95, ry=0.0, y_held=False, now=0.0)
    assert cmd == LocomotionCmd("turn_right", "deg_45")


def test_off_blocks_everything() -> None:
    """OFF mode is the safety baseline: no command of any kind."""
    dec = IntentDecoder(
        stick_deadzone=0.3,
        chord_debounce_s=0.0,
        enable_continuous_torso=True,
    )
    assert dec.mode is StreamMode.OFF
    for rx, ry, lx, ly in [
        (0.0, 0.5, 0.0, 0.0),  # would-be hold_torso
        (0.95, 0.0, 0.0, 0.0),  # would-be turn_right
        (0.0, 0.0, 0.0, 0.9),  # would-be fwd_step
        (0.0, 0.0, 0.0, 0.0),  # would-be neutral hold_torso
    ]:
        assert dec.decode_locomotion(
            lx, ly, rx=rx, ry=ry, y_held=False, now=0.0,
        ) is None


# ---------------------------------------------------------------------------
# v7.4: dominance cones (R-stick yaw priority + L-stick roll priority)
# ---------------------------------------------------------------------------


def test_yaw_priority_cone_suppresses_pitch_when_rx_dominates() -> None:
    """Cone fires when |ry| past deadzone but |ry| < ratio*|rx|. Pick
    ratio=0.8 so the inequality has room: with rx=0.6 the threshold
    is 0.48; ry=0.35 (just past 0.3 deadzone) sits well below 0.48
    so the cone should suppress pitch."""
    dec = _make_decoder(pitch_dominance_ratio=0.8)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.6, ry=0.35, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg == 0.0
    assert cmd.waist_yaw_deg < 0.0


def test_yaw_priority_cone_allows_pitch_when_ry_dominates() -> None:
    """When ry dominates the cone allows both axes."""
    dec = _make_decoder(pitch_dominance_ratio=0.4)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.5, ry=0.7, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg > 0.0
    assert cmd.waist_yaw_deg < 0.0


def test_yaw_priority_cone_disabled_when_ratio_zero() -> None:
    """ratio=0 -> the cone never suppresses; both axes fire whenever
    they're past the deadzone."""
    dec = _make_decoder(pitch_dominance_ratio=0.0)
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.6, ry=0.35, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg > 0.0
    assert cmd.waist_yaw_deg < 0.0


def test_yaw_priority_cone_does_not_block_pure_pitch() -> None:
    """Pure forward lean (|rx| in deadzone) is never suppressed -- the
    cone only acts when both axes are active."""
    dec = _make_decoder(pitch_dominance_ratio=0.9)  # very strict
    cmd = dec.decode_locomotion(0.0, 0.0, rx=0.0, ry=0.5, y_held=False, now=0.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg > 0.0


# ---------------------------------------------------------------------------
# v7.4: ARM_MANIPULATION L-stick decoding (roll + continuous height)
# ---------------------------------------------------------------------------


def _make_arm_man_decoder_v74(**kwargs) -> IntentDecoder:
    dec = _make_decoder(**kwargs)
    b_only = ButtonEvents(
        a_pressed=False, b_pressed=True, x_pressed=False, y_pressed=False,
        ab_pressed=False, xy_pressed=False,
        ax_pressed=False, by_pressed=False,
        abxy_pressed=False,
    )
    transition = dec.update_mode(b_only, now=1.0)
    assert transition is not None
    assert dec.mode is StreamMode.ARM_MANIPULATION
    return dec


def test_arm_man_lstick_lx_emits_positive_roll() -> None:
    """v7.4: ARM_MAN L-stick X drives waist roll. lx > 0 -> positive
    waist_roll_deg (operator's right-side lean)."""
    dec = _make_arm_man_decoder_v74()
    cmd = dec.decode_locomotion(0.5, 0.0, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_roll_deg > 0.0
    assert cmd.waist_pitch_deg == 0.0
    assert cmd.waist_yaw_deg == 0.0


def test_arm_man_lstick_lx_negative_emits_negative_roll() -> None:
    dec = _make_arm_man_decoder_v74()
    cmd = dec.decode_locomotion(-0.5, 0.0, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.waist_roll_deg < 0.0


def test_arm_man_lstick_ly_negative_emits_squat_below_default() -> None:
    """Quest3 convention: ly < 0 = stick pushed forward (away from
    operator) -> hip DOWN (squat). Asymmetric clamp: full forward
    push gives default - max_height_down_m."""
    default_h = 0.687
    dec = _make_arm_man_decoder_v74(
        default_hip_height_m=default_h,
        max_height_down_m=0.09,
        max_height_up_m=0.04,
    )
    cmd = dec.decode_locomotion(0.0, -1.0, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.hip_height_m is not None
    assert cmd.hip_height_m == pytest.approx(default_h - 0.09)


def test_arm_man_lstick_ly_positive_emits_stand_above_default() -> None:
    default_h = 0.687
    dec = _make_arm_man_decoder_v74(
        default_hip_height_m=default_h,
        max_height_down_m=0.09,
        max_height_up_m=0.04,
    )
    cmd = dec.decode_locomotion(0.0, 1.0, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.hip_height_m is not None
    assert cmd.hip_height_m == pytest.approx(default_h + 0.04)


def test_arm_man_lstick_ly_in_deadzone_yields_no_height_override() -> None:
    """In-deadzone ly leaves hip_height_m at None (== "use planner
    default") so a tiny ly noise around neutral doesn't compete with
    the kplanner's idle hip-height target."""
    dec = _make_arm_man_decoder_v74(deadzone=0.3)
    cmd = dec.decode_locomotion(0.0, 0.1, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.hip_height_m is None


def test_locomotion_lstick_does_not_emit_height_or_roll() -> None:
    """Sanity: in LOCOMOTION mode the L-stick still owns step / side /
    walk; the v7.4 roll + height decoding does NOT activate. A neutral
    R-stick (so the discrete-stick path is a no-op) and a small soft
    L-stick lx that's past the deadzone should produce ``side_*`` (or
    ``hold_torso`` with roll=0 / height=None if the L-stick is in
    deadzone), never roll or height. Use a sub-deadzone L-stick to
    fall through to the hold_torso branch."""
    dec = _make_decoder()
    assert dec.mode is StreamMode.LOCOMOTION
    cmd = dec.decode_locomotion(
        0.1, 0.1, rx=0.0, ry=0.0, y_held=False, now=0.0,
    )
    assert cmd is not None
    if cmd.intent == "hold_torso":
        assert cmd.waist_roll_deg == 0.0
        assert cmd.hip_height_m is None


def test_arm_man_lstick_disabled_via_flag() -> None:
    """``enable_arm_man_lstick=False`` restores v7.3 behaviour:
    L-stick is silent in ARM_MAN. The R-stick lean still works."""
    dec = _make_arm_man_decoder_v74(enable_arm_man_lstick=False)
    cmd = dec.decode_locomotion(
        0.5, -0.5, rx=0.0, ry=0.5, y_held=False, now=2.0,
    )
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_pitch_deg > 0.0
    assert cmd.waist_roll_deg == 0.0
    assert cmd.hip_height_m is None


# ---------------------------------------------------------------------------
# v7.4: roll-priority cone (L-stick height suppression on roll dominance)
# ---------------------------------------------------------------------------


def test_roll_priority_cone_suppresses_height_when_lx_dominates() -> None:
    """Cone fires when |ly| past deadzone but |ly| < ratio*|lx|. Pick
    ratio=0.8 so the inequality has room: with lx=0.5 the threshold
    is 0.40; ly=0.35 (just past 0.3 deadzone) sits well below 0.40
    so the cone should suppress hip_height_m."""
    dec = _make_arm_man_decoder_v74(height_dominance_ratio=0.8)
    cmd = dec.decode_locomotion(0.5, 0.35, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.intent == "hold_torso"
    assert cmd.waist_roll_deg > 0.0
    assert cmd.hip_height_m is None


def test_roll_priority_cone_allows_height_when_ly_dominates() -> None:
    dec = _make_arm_man_decoder_v74(height_dominance_ratio=0.4)
    cmd = dec.decode_locomotion(0.5, -0.7, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.waist_roll_deg > 0.0
    assert cmd.hip_height_m is not None
    assert cmd.hip_height_m < 0.687  # squat


def test_roll_priority_cone_does_not_block_pure_height() -> None:
    """Pure ly (|lx| in deadzone) is never suppressed."""
    dec = _make_arm_man_decoder_v74(height_dominance_ratio=0.9)
    cmd = dec.decode_locomotion(0.0, -0.5, rx=0.0, ry=0.0, y_held=False, now=2.0)
    assert cmd is not None
    assert cmd.hip_height_m is not None
    assert cmd.hip_height_m < 0.687


# ---------------------------------------------------------------------------
# v7.4: hip_height_m wire-side hysteresis (significant-change throttling)
# ---------------------------------------------------------------------------


def test_hip_height_below_threshold_throttled() -> None:
    """ly noise smaller than ``hold_height_threshold_m`` must not
    re-emit a hold_torso command -- the decoder treats two finite
    hip_height_m targets within the threshold as equal."""
    dec = _make_arm_man_decoder_v74(hold_height_threshold_m=0.005)
    first = dec.decode_locomotion(
        0.0, -1.0, rx=0.0, ry=0.0, y_held=False, now=2.0,
    )
    assert first is not None
    # Tiny ly perturbation -> hip_height_m delta ~ max_height_down_m *
    # |delta_ly_normalized|. With max_height_down_m=0.09 and
    # rescale slope 1/0.7, lly delta of 0.001 is roughly 0.13mm,
    # well below 5mm threshold.
    repeat = dec.decode_locomotion(
        0.0, -0.999, rx=0.0, ry=0.0, y_held=False, now=2.02,
    )
    assert repeat is None


def test_hip_height_engagement_re_emits() -> None:
    """The transition None <-> finite is always significant -- the
    operator engaging or releasing the squat axis must re-emit."""
    dec = _make_arm_man_decoder_v74(hold_height_threshold_m=0.05)
    first = dec.decode_locomotion(
        0.0, 0.0, rx=0.0, ry=0.0, y_held=False, now=2.0,
    )
    assert first is not None
    assert first.hip_height_m is None
    later = dec.decode_locomotion(
        0.0, -0.5, rx=0.0, ry=0.0, y_held=False, now=2.02,
    )
    assert later is not None
    assert later.hip_height_m is not None


# ---------------------------------------------------------------------------
# v7.4: constructor validation
# ---------------------------------------------------------------------------


def test_pitch_dominance_ratio_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(pitch_dominance_ratio=-0.1)
    with pytest.raises(ValueError):
        IntentDecoder(pitch_dominance_ratio=1.1)


def test_height_dominance_ratio_out_of_range_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(height_dominance_ratio=-0.1)
    with pytest.raises(ValueError):
        IntentDecoder(height_dominance_ratio=1.1)


def test_max_height_negative_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(max_height_down_m=-0.01)
    with pytest.raises(ValueError):
        IntentDecoder(max_height_up_m=-0.01)


def test_default_hip_height_zero_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(default_hip_height_m=0.0)


def test_hold_height_threshold_negative_rejected() -> None:
    with pytest.raises(ValueError):
        IntentDecoder(hold_height_threshold_m=-1e-3)
