"""Behavior tests for the VR-shared button + joystick utilities.

These pin the exact behavior that was extracted from
``gear_sonic/scripts/quest3_manager_thread_server.py`` in Step 1 of the
Phase 0 plan, so the refactor can never silently regress.
"""

from __future__ import annotations

import numpy as np
import pytest

from gear_sonic.utils.teleop.vr.button_state_machine import (
    ButtonEvents,
    ButtonStateMachine,
)
from gear_sonic.utils.teleop.vr.joystick_mapping import (
    JOYSTICK_DEADZONE,
    YawAccumulator,
    apply_radial_deadzone,
)


# ---------------------------------------------------------------------------
# Backward-compat: common.py still re-exports the joystick utilities
# ---------------------------------------------------------------------------


def test_common_reexports_joystick_utilities():
    from gear_sonic.utils.teleop import common as common_mod

    assert common_mod.JOYSTICK_DEADZONE is JOYSTICK_DEADZONE
    assert common_mod.YawAccumulator is YawAccumulator


# ---------------------------------------------------------------------------
# ButtonStateMachine
# ---------------------------------------------------------------------------


def _make_sm() -> ButtonStateMachine:
    return ButtonStateMachine(log_prefix=None)


def test_first_tick_no_false_positive():
    sm = _make_sm()
    ev = sm.tick(False, False, False, False)
    for field in (
        "a_pressed", "b_pressed", "x_pressed", "y_pressed",
        "ab_pressed", "xy_pressed", "ax_pressed", "by_pressed", "abxy_pressed",
    ):
        assert getattr(ev, field) is False, f"{field} fired on no-input first tick"


def test_per_button_rising_edge_only():
    sm = _make_sm()
    sm.tick(False, False, False, False)
    ev = sm.tick(True, False, False, False)
    assert ev.a_pressed
    assert not ev.b_pressed
    ev = sm.tick(True, False, False, False)
    assert not ev.a_pressed, "a should not re-fire while held"
    ev = sm.tick(False, False, False, False)
    assert not ev.a_pressed, "a should not fire on release"
    ev = sm.tick(True, False, False, False)
    assert ev.a_pressed, "a should fire again after release + press"


@pytest.mark.parametrize(
    "chord_attr,inputs",
    [
        ("ab_pressed", (True, True, False, False)),
        ("xy_pressed", (False, False, True, True)),
        ("ax_pressed", (True, False, True, False)),
        ("by_pressed", (False, True, False, True)),
        ("abxy_pressed", (True, True, True, True)),
    ],
)
def test_chord_rising_edge_only(chord_attr, inputs):
    sm = _make_sm()
    sm.tick(False, False, False, False)
    ev = sm.tick(*inputs)
    assert getattr(ev, chord_attr), f"{chord_attr} did not fire on chord rise"
    ev = sm.tick(*inputs)
    assert not getattr(ev, chord_attr), f"{chord_attr} re-fired while chord held"
    sm.tick(False, False, False, False)
    ev = sm.tick(*inputs)
    assert getattr(ev, chord_attr), f"{chord_attr} did not re-fire after release"


def test_abxy_chord_does_not_double_fire_subchords():
    """A+B+X+Y must only fire ``abxy_pressed``; the subchords also fire
    on the same tick (because their two-button condition is also met),
    which is the original G1-manager behavior. Lock it in."""
    sm = _make_sm()
    sm.tick(False, False, False, False)
    ev = sm.tick(True, True, True, True)
    assert ev.abxy_pressed
    assert ev.ab_pressed
    assert ev.xy_pressed
    assert ev.ax_pressed
    assert ev.by_pressed


def test_log_prefix_silences_when_none(capsys):
    sm = ButtonStateMachine(log_prefix=None)
    sm.tick(True, False, False, False)
    captured = capsys.readouterr()
    assert captured.out == ""


def test_log_prefix_emits_per_button_lines(capsys):
    sm = ButtonStateMachine(log_prefix="Test")
    sm.tick(True, False, False, False)
    captured = capsys.readouterr()
    assert "[Test] A pressed" in captured.out


# ---------------------------------------------------------------------------
# apply_radial_deadzone parity with original inline math
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", [0.0, 0.05, 0.149, 0.15, 0.151, 0.3, 0.5, 0.99, 1.0, 1.5])
def test_apply_radial_deadzone_matches_original(raw):
    raw_clipped = float(np.clip(raw, 0.0, 1.0))
    if abs(raw_clipped) < JOYSTICK_DEADZONE:
        expected = 0.0
    else:
        expected = (raw_clipped - JOYSTICK_DEADZONE) / (1.0 - JOYSTICK_DEADZONE)
        if expected > 1.0:
            expected = 1.0
    assert apply_radial_deadzone(raw_clipped) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# YawAccumulator
# ---------------------------------------------------------------------------


def test_yaw_accumulator_initial_state():
    ya = YawAccumulator()
    assert ya.yaw_angle() == 0.0
    assert ya.yaw_angle_change() == 0.0
    assert ya.heading == [1.0, 0.0, 0.0]


def test_yaw_accumulator_ignores_input_inside_deadzone():
    ya = YawAccumulator()
    ya.update(rx=0.05, dt=0.05)
    assert ya.yaw_angle() == 0.0


def test_yaw_accumulator_integrates_above_deadzone():
    ya = YawAccumulator(yaw_gain=1.5, deadzone=JOYSTICK_DEADZONE)
    ya.update(rx=1.0, dt=0.1)
    expected = 1.5 * (-1.0) * 0.1
    assert ya.yaw_angle() == pytest.approx(expected, abs=1e-9)
    expected_heading = [np.cos(expected), np.sin(expected), 0.0]
    assert ya.heading == pytest.approx(expected_heading, abs=1e-9)


def test_yaw_accumulator_reset_clears_state():
    ya = YawAccumulator()
    ya.update(rx=1.0, dt=0.5)
    assert ya.yaw_angle() != 0.0
    ya.reset()
    assert ya.yaw_angle() == 0.0
    assert ya.heading == [1.0, 0.0, 0.0]


# ---------------------------------------------------------------------------
# Sanity: ButtonEvents is hashable / dataclass-like
# ---------------------------------------------------------------------------


def test_button_events_is_frozen_dataclass():
    ev = ButtonEvents(False, False, False, False, False, False, False, False, False)
    with pytest.raises(Exception):
        ev.a_pressed = True
