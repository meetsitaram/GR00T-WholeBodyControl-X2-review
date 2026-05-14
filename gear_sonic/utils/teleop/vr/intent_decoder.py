"""X2 intent decoder: Quest 3 buttons + thumbsticks -> planner_cmd vocabulary.

This module is the X2 counterpart to the (G1-specific) joystick-to-mode
mapping inside :mod:`gear_sonic.scripts.quest3_manager_thread_server`.
It is intentionally pure logic — no Quest 3 connection, no ZMQ
dependency — so it can be unit-tested directly and reused by the new
``quest3_manager_x2.py``.

Vocabulary
----------

The output of :meth:`IntentDecoder.decode_locomotion` is the same
``(intent, magnitude)`` tuple format that
:class:`gear_sonic.utils.planner.state_machine.LocomotionCommand`
expects, and matches the ``KEYBOARD_MAP`` defined in
:mod:`gear_sonic.scripts.x2_heuristic_planner` so the X2 planner consumes
manager-emitted commands with no wire-format change.

StreamMode
----------

The X2 manager only uses three modes in Phase 0:

- ``OFF`` — manager is idle; no ``planner_cmd`` is emitted; arms are
  held at their last commanded pose.
- ``LOCOMOTION`` — thumbsticks drive the planner; arm IK output is
  ignored and frozen to the last commanded pose.
- ``ARM_MANIPULATION`` — planner is held at ``idle_stand``; Quest 3
  hand poses drive the arm IK.

This is intentionally a *separate* enum from
:class:`gear_sonic.utils.teleop.common.StreamMode` (which carries
G1-specific PLANNER / PLANNER_FROZEN_UPPER_BODY / PLANNER_VR_3PT
values). Future modes (e.g. ``LOC_AND_ARM`` for parallel control) can
be added here without disturbing G1 callers.

Button vocabulary
-----------------

- ``A+B+X+Y`` chord — toggle ``OFF`` <-> ``LOCOMOTION``  (engage / E-stop)
- ``B`` single  — toggle ``LOCOMOTION`` <-> ``ARM_MANIPULATION`` (when not OFF)

Locomotion vocabulary (LOCOMOTION mode only)
--------------------------------------------

- Left stick forward          -> ``(fwd_step, default)``
- Left stick back             -> ``(back_step, default)``
- Left stick right            -> ``(side_right, default)``
- Left stick left             -> ``(side_left,  default)``
- Right stick hard left/right -> ``(turn_left/right, deg_45 [or deg_90])``
- Right stick (continuous)    -> ``(hold_torso, continuous, waist_*_deg)``
    * ``ry > 0`` => positive ``waist_pitch_deg`` (forward lean)
    * ``rx`` (no A held) => negative ``waist_yaw_deg`` for stick-right (twist)
    * ``rx`` (A held)    => negative ``waist_roll_deg`` for stick-right (lateral lean)
- ``Y`` held                  -> ``(crouch, medium)`` (only when enabled)
- All sticks neutral          -> ``(idle, default)`` (or ``hold_torso`` at 0/0/0
                                  while continuous mode is engaged)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from gear_sonic.utils.teleop.vr.button_state_machine import ButtonEvents


class StreamMode(Enum):
    """X2 manager operating modes for Phase 0.

    Distinct from :class:`gear_sonic.utils.teleop.common.StreamMode`
    (which is G1-shaped). See module docstring for details.
    """

    OFF = 0
    LOCOMOTION = 1
    ARM_MANIPULATION = 2


@dataclass(frozen=True)
class LocomotionCmd:
    """High-level command tuple: ``intent`` + ``magnitude``.

    Mirrors :class:`gear_sonic.utils.planner.state_machine.LocomotionCommand`
    but without the ``source`` field — the manager fills that in when
    serializing to JSON for the planner.

    The ``waist_*_deg`` fields carry continuous waist targets when
    ``intent == "hold_torso"`` (paired with ``magnitude == "continuous"``);
    they are ignored for every other intent so existing call sites remain
    valid. Defaults of 0.0 mean "neutral hold" -- a stick-released
    operator naturally produces a hold at the rest pose, which the
    planner blends back to ``idle_stand`` once a non-hold command lands.
    """

    intent: str
    magnitude: str
    waist_pitch_deg: float = 0.0
    waist_roll_deg: float = 0.0
    waist_yaw_deg: float = 0.0


@dataclass(frozen=True)
class ModeTransition:
    """Returned by :meth:`IntentDecoder.update_mode` when the mode flips."""

    previous: StreamMode
    current: StreamMode


class IntentDecoder:
    """Stateful Quest-3-button-and-stick to LocomotionCmd mapper.

    Args:
        stick_deadzone: Per-axis deflection threshold below which the
            stick is treated as neutral. Higher than the planner-side
            ``JOYSTICK_DEADZONE`` because operators rest their thumbs
            on the sticks (which would otherwise emit jittery
            ``fwd_step`` commands).
        repeat_interval_s: If > 0, re-emit the most recent command at
            this cadence even when the input has not changed; useful
            because the planner has a short queue and held thumbsticks
            will otherwise stall once the bin finishes. ``0.0``
            disables the repeat (commands fire only on change).
    """

    def __init__(
        self,
        stick_deadzone: float = 0.30,
        repeat_interval_s: float = 0.0,
        chord_debounce_s: float = 0.5,
        enable_crouch: bool = False,
        enable_lean_fwd: bool = False,
        enable_torso: bool = False,
        enable_continuous_torso: bool = False,
        turn_threshold: float = 0.75,
        lean_medium_threshold: float = 0.55,
        lean_large_threshold: float = 0.80,
        # Per-axis maximum deflection deg (clamps; defaults match the
        # planner's _WAIST_RAMP_CAP_DEG safety caps so the operator
        # cannot drive STATIC_HOLD past a trackable angle).
        max_waist_pitch_deg: float = 20.0,
        max_waist_roll_deg: float = 10.0,
        max_waist_yaw_deg: float = 40.0,
        # Hysteresis for continuous hold_torso emission. The decoder
        # only re-emits when ANY axis target moves by at least this
        # many degrees from the last sent target (or repeat_interval_s
        # elapses). Keeps the ZMQ topic from being spammed at every tick
        # by inevitable stick noise without losing perceived smoothness:
        # the planner-side _HoldTracker slews between coarse waypoints.
        hold_target_threshold_deg: float = 0.5,
    ) -> None:
        if stick_deadzone <= 0 or stick_deadzone >= 1:
            raise ValueError(f"stick_deadzone must be in (0, 1); got {stick_deadzone}")
        if repeat_interval_s < 0:
            raise ValueError(
                f"repeat_interval_s must be >= 0; got {repeat_interval_s}"
            )
        if chord_debounce_s < 0:
            raise ValueError(
                f"chord_debounce_s must be >= 0; got {chord_debounce_s}"
            )
        if not (stick_deadzone < turn_threshold < 1.0):
            raise ValueError(
                f"turn_threshold must satisfy stick_deadzone < t < 1.0; "
                f"got deadzone={stick_deadzone}, t={turn_threshold}"
            )
        if not (
            stick_deadzone < lean_medium_threshold < lean_large_threshold < 1.0
        ):
            raise ValueError(
                f"lean thresholds must satisfy "
                f"stick_deadzone < medium < large < 1.0; got "
                f"deadzone={stick_deadzone}, medium={lean_medium_threshold}, "
                f"large={lean_large_threshold}"
            )

        self._stick_deadzone = float(stick_deadzone)
        self._repeat_interval_s = float(repeat_interval_s)
        self._chord_debounce_s = float(chord_debounce_s)
        self._turn_threshold = float(turn_threshold)
        self._lean_medium_threshold = float(lean_medium_threshold)
        self._lean_large_threshold = float(lean_large_threshold)
        # Crouch is disabled by default: the X2 heuristic planner's
        # crouch primitive currently destabilizes the controller (the
        # SONIC tracking policy tips over partway through). Until the
        # planner-side fix lands, we drop Y-as-crouch on the floor and
        # log it as ignored. Pass enable_crouch=True to re-enable for
        # offline planner experiments.
        self._enable_crouch = bool(enable_crouch)
        # ``lean_fwd_*`` and ``torso_*_30deg`` are *replay* primitives:
        # the curated bins lean / twist into the pose and immediately
        # blend back to standing instead of holding. For interactive
        # teleop this looks like "the body flicks then snaps back",
        # which confuses operators (you push the stick and nothing
        # appears to stick). We disable both by default; the right-
        # stick X-axis hard-deflection still emits ``turn_*`` (which
        # the operator finds intuitive), and graded lean / soft torso
        # fall through to ``idle`` instead. Re-enable per primitive
        # once the planner can hold the static pose.
        #
        # ``enable_continuous_torso`` (default ON) supersedes the soft-
        # band discrete bins with a continuous hold_torso target. The
        # right-stick X-axis hard turn_* path is unchanged; pitch (ry)
        # / yaw (rx) / roll (A+rx) all map to a single hold_torso
        # command that the planner runs in STATIC_HOLD. Set False to
        # restore the legacy discrete-bin behavior.
        self._enable_lean_fwd = bool(enable_lean_fwd)
        self._enable_torso = bool(enable_torso)
        self._enable_continuous_torso = bool(enable_continuous_torso)
        if max_waist_pitch_deg < 0 or max_waist_roll_deg < 0 or max_waist_yaw_deg < 0:
            raise ValueError(
                "max_waist_*_deg must be non-negative; got "
                f"({max_waist_pitch_deg}, {max_waist_roll_deg}, {max_waist_yaw_deg})"
            )
        if hold_target_threshold_deg < 0:
            raise ValueError(
                f"hold_target_threshold_deg must be >= 0; "
                f"got {hold_target_threshold_deg}"
            )
        self._max_waist_pitch_deg = float(max_waist_pitch_deg)
        self._max_waist_roll_deg = float(max_waist_roll_deg)
        self._max_waist_yaw_deg = float(max_waist_yaw_deg)
        self._hold_target_threshold_deg = float(hold_target_threshold_deg)

        self._mode = StreamMode.OFF
        self._last_emitted: Optional[LocomotionCmd] = None
        self._last_emit_t: float = -1.0
        # Wall-clock deadline (monotonic) before which decode_locomotion
        # treats Y-as-crouch and all stick deflections as neutral. Armed
        # on any A+B+X+Y chord transition because the chord physically
        # requires Y to be held, and Y mapped naively to crouch was
        # tipping the robot over the instant the operator entered
        # LOCOMOTION mode.
        self._chord_quiet_until: float = -1.0

    # -- mode -----------------------------------------------------------------

    @property
    def mode(self) -> StreamMode:
        return self._mode

    def update_mode(
        self,
        ev: ButtonEvents,
        now: Optional[float] = None,
    ) -> Optional[ModeTransition]:
        """Apply the button-driven mode-toggle transitions.

        Returns the :class:`ModeTransition` when a transition fired this
        tick, or ``None`` when the mode is unchanged.

        ``now`` is the caller's monotonic clock; when supplied and the
        transition is driven by the A+B+X+Y chord, the decoder arms a
        short quiet window that suppresses Y-as-crouch and stick inputs
        so the operator's chord release does not immediately steer the
        planner.
        """
        prev = self._mode
        new = self._mode

        if ev.abxy_pressed:
            new = (
                StreamMode.LOCOMOTION
                if self._mode == StreamMode.OFF
                else StreamMode.OFF
            )
        elif ev.b_pressed and self._mode != StreamMode.OFF:
            # B-single only matters once we're already in an active mode.
            # In OFF mode, B is a no-op (avoids accidental engages).
            new = (
                StreamMode.ARM_MANIPULATION
                if self._mode == StreamMode.LOCOMOTION
                else StreamMode.LOCOMOTION
            )

        if new == prev:
            return None

        self._mode = new
        # On a mode change, drop the last-emitted memory so the very
        # first tick after the flip re-emits whatever the new mode
        # implies (e.g. an idle command when leaving LOCOMOTION, or a
        # held thumbstick re-firing on entering it).
        self._last_emitted = None

        # The A+B+X+Y chord physically holds Y down; without a debounce
        # the operator's chord-release fires a crouch the instant they
        # land in LOCOMOTION mode and the robot tips over. Arm a quiet
        # window so the next ~chord_debounce_s of decode_locomotion
        # sees Y as released and sticks as neutral. We also arm it on
        # OFF transitions to avoid the symmetric "chord into OFF then
        # straight back into LOCOMOTION" footgun.
        if ev.abxy_pressed and now is not None and self._chord_debounce_s > 0:
            self._chord_quiet_until = now + self._chord_debounce_s

        return ModeTransition(previous=prev, current=new)

    # -- locomotion -----------------------------------------------------------

    def decode_locomotion(
        self,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        y_held: bool,
        now: float,
        a_held: bool = False,
        x_held: bool = False,
    ) -> Optional[LocomotionCmd]:
        """Map this tick's stick + button state to a ``LocomotionCmd``.

        Returns ``None`` when the manager should NOT publish a command
        this tick (either because we're not in LOCOMOTION mode, or
        because the desired command is unchanged from the previous
        emit and the repeat interval has not elapsed).

        Modifier semantics (LOCOMOTION mode only):

        - ``a_held`` + forward/back stick → continuous walk
          (``walk / forward`` resp. ``walk / backward``) instead of
          a single-stride ``fwd_step`` / ``back_step``. Default
          (``a_held=False``) preserves the legacy single-step UX.
        - ``x_held`` + right-stick X → 90° turn
          (``turn_left / deg_90`` resp. ``turn_right / deg_90``)
          instead of the default 45° step.
        """
        if self._mode != StreamMode.LOCOMOTION:
            return None

        if now < self._chord_quiet_until:
            # Operator is still releasing the A+B+X+Y chord that put us
            # here. Force-quiet inputs so we don't crouch on chord-Y or
            # walk on a thumb that grazed the stick during the chord.
            cmd = LocomotionCmd("idle", "default")
        else:
            cmd = self._cmd_for_inputs(
                lx=lx, ly=ly, rx=rx, ry=ry,
                y_held=y_held, a_held=a_held, x_held=x_held,
            )
        return self._maybe_emit(cmd, now)

    def _cmd_for_inputs(
        self,
        *,
        lx: float,
        ly: float,
        rx: float,
        ry: float,
        y_held: bool,
        a_held: bool = False,
        x_held: bool = False,
    ) -> LocomotionCmd:
        """Pure stick + button -> LocomotionCmd mapping (no state)."""
        if y_held and self._enable_crouch:
            return LocomotionCmd("crouch", "medium")

        # Left stick: cardinal direction wins (whichever axis dominates).
        # We bias toward forward/backward when the deflections are
        # roughly equal so a slight diagonal still produces forward
        # walking instead of jittering to side-step on a small lateral
        # component.
        l_active = (
            abs(ly) >= self._stick_deadzone or abs(lx) >= self._stick_deadzone
        )
        if l_active:
            if abs(ly) >= abs(lx):
                # A held promotes single-stride to continuous walk.
                # ``walk / forward`` resolves to ``fwd_walk_standard``
                # in the planner; ``walk / backward`` resolves to
                # ``back_walk_standard``. Both are loop primitives, so
                # holding the stick keeps the robot walking until the
                # operator releases (decoder then emits idle and the
                # planner blends back to standing).
                if a_held:
                    if ly > 0:
                        return LocomotionCmd("walk", "forward")
                    return LocomotionCmd("walk", "backward")
                if ly > 0:
                    return LocomotionCmd("fwd_step", "default")
                return LocomotionCmd("back_step", "default")
            if lx > 0:
                return LocomotionCmd("side_right", "default")
            return LocomotionCmd("side_left", "default")

        # Right stick: continuous-hold path wins when enabled, with
        # hard-turn deflection still pre-empting (operator wants to
        # PIVOT, not lean). Otherwise fall back to the legacy
        # dominant-axis decoder so existing scripted regression tests
        # (test_intent_decoder.py) keep their semantics.
        if self._enable_continuous_torso:
            rx_active = abs(rx) >= self._stick_deadzone
            if rx_active and abs(rx) >= self._turn_threshold:
                magnitude = "deg_90" if x_held else "deg_45"
                if rx > 0:
                    return LocomotionCmd("turn_right", magnitude)
                return LocomotionCmd("turn_left", magnitude)
            return self._continuous_hold_cmd(rx=rx, ry=ry, a_held=a_held)

        # ----- Legacy discrete-bin path -----------------------------------
        # Y axis (forward) -> graded lean_fwd_{small,medium,large}
        #   (gated on ``enable_lean_fwd``).
        # X axis: hard -> turn_*; soft -> torso_* (gated on enable_torso).
        # Dominant-axis precedence preserves the legacy "ry > rx wins"
        # guard so a forward + slight-right push doesn't pivot.
        ry_active = abs(ry) >= self._stick_deadzone
        rx_active = abs(rx) >= self._stick_deadzone
        if ry_active and abs(ry) >= abs(rx):
            if ry > 0 and self._enable_lean_fwd:
                if ry >= self._lean_large_threshold:
                    return LocomotionCmd("lean_fwd", "large")
                if ry >= self._lean_medium_threshold:
                    return LocomotionCmd("lean_fwd", "medium")
                return LocomotionCmd("lean_fwd", "small")
            return LocomotionCmd("idle", "default")
        if rx_active:
            if abs(rx) >= self._turn_threshold:
                magnitude = "deg_90" if x_held else "deg_45"
                if rx > 0:
                    return LocomotionCmd("turn_right", magnitude)
                return LocomotionCmd("turn_left", magnitude)
            if self._enable_torso:
                if rx > 0:
                    return LocomotionCmd("torso_right", "deg_30")
                return LocomotionCmd("torso_left", "deg_30")
            return LocomotionCmd("idle", "default")

        return LocomotionCmd("idle", "default")

    @staticmethod
    def _scaled_axis(value: float, deadzone: float, max_deg: float) -> float:
        """Map a raw stick deflection in [-1, 1] to a clamped angle in degrees.

        Below |value| <= deadzone the result is 0 (so a thumb at rest
        produces a neutral target). Outside the deadzone (deadzone, 1)
        is rescaled to (0, 1) so that just past the deadzone yields a
        small angle and full deflection yields ``max_deg``. Sign is
        preserved.
        """
        sign = 1.0 if value >= 0 else -1.0
        mag = abs(float(value))
        if mag <= deadzone:
            return 0.0
        denom = max(1.0 - deadzone, 1e-6)
        norm = min(1.0, (mag - deadzone) / denom)
        return sign * norm * float(max_deg)

    def continuous_waist_target(
        self,
        rx: float,
        ry: float,
        a_held: bool = False,
    ) -> tuple[float, float, float]:
        """Compute the (pitch, roll, yaw) continuous waist target in degrees.

        Public companion to :meth:`_continuous_hold_cmd` for callers
        (e.g. the Quest 3 manager) that need the *target value* without
        the dedup / throttle gating done by ``decode_locomotion``. Used
        to latch the current operator-driven hold pose at the moment
        the manager flips ``LOCOMOTION -> ARM_MANIPULATION``.

        Always returns clamped, deadzone-aware angles regardless of the
        decoder's mode or the ``enable_continuous_torso`` flag.
        """
        cmd = self._continuous_hold_cmd(rx=rx, ry=ry, a_held=a_held)
        return (cmd.waist_pitch_deg, cmd.waist_roll_deg, cmd.waist_yaw_deg)

    def _continuous_hold_cmd(
        self,
        *,
        rx: float,
        ry: float,
        a_held: bool,
    ) -> LocomotionCmd:
        """Build a continuous-hold LocomotionCmd from the right-stick state.

        Stick conventions (validated with the operator on 2026-05-13):

        - ``ry > 0`` (stick up)   -> forward lean (positive pitch).
          Backward lean is unsupported (no primitive, single-sided cap).
        - ``rx`` (no A held)      -> twist (yaw). Stick right (rx > 0)
          maps to a NEGATIVE waist_yaw, matching ``torso_right_*deg``
          which uses negative peak_deg.
        - ``rx`` (A held)         -> lateral lean (roll). Stick right
          maps to a NEGATIVE waist_roll for the same sign convention as
          ``lean_right_*``.

        Released stick (in deadzone) yields a (0, 0, 0) hold target;
        the planner's STATIC_HOLD state slews back to neutral via the
        same ``_HoldTracker`` path that any other change would use.
        """
        # Pitch: clamp ry > 0 to (0, max_waist_pitch_deg]; ry <= 0 -> 0.
        pitch_deg = max(
            0.0,
            self._scaled_axis(
                max(0.0, ry),
                self._stick_deadzone,
                self._max_waist_pitch_deg,
            ),
        )
        if a_held:
            roll_deg = -self._scaled_axis(
                rx, self._stick_deadzone, self._max_waist_roll_deg,
            )
            yaw_deg = 0.0
        else:
            roll_deg = 0.0
            yaw_deg = -self._scaled_axis(
                rx, self._stick_deadzone, self._max_waist_yaw_deg,
            )
        return LocomotionCmd(
            intent="hold_torso",
            magnitude="continuous",
            waist_pitch_deg=float(pitch_deg),
            waist_roll_deg=float(roll_deg),
            waist_yaw_deg=float(yaw_deg),
        )

    def _maybe_emit(
        self,
        cmd: LocomotionCmd,
        now: float,
    ) -> Optional[LocomotionCmd]:
        """Apply de-duplication + optional repeat-interval gating.

        Behaviour:

        - First tick a command appears: emit.
        - Subsequent ticks with the SAME command: emit only if
          ``repeat_interval_s > 0`` AND the elapsed time since the
          last emit reaches the threshold.
        - First tick a NEW command appears: emit (resets the timer).

        For ``hold_torso`` commands, "SAME" means the waist targets are
        within ``hold_target_threshold_deg`` of the previous emit on
        every axis -- otherwise inevitable thumbstick noise would
        retrigger the planner at every tick. The planner-side
        ``_HoldTracker`` slews between coarse waypoints, so this
        threshold is purely a wire-bandwidth optimisation.
        """
        if self._last_emitted is None or self._is_significant_change(
            cmd, self._last_emitted
        ):
            self._last_emitted = cmd
            self._last_emit_t = now
            return cmd

        if (
            self._repeat_interval_s > 0
            and (now - self._last_emit_t) >= self._repeat_interval_s
        ):
            self._last_emit_t = now
            return cmd

        return None

    def _is_significant_change(
        self,
        new: LocomotionCmd,
        old: LocomotionCmd,
    ) -> bool:
        """True iff ``new`` should be re-emitted relative to ``old``."""
        if new.intent != old.intent or new.magnitude != old.magnitude:
            return True
        if new.intent == "hold_torso":
            thresh = self._hold_target_threshold_deg
            return (
                abs(new.waist_pitch_deg - old.waist_pitch_deg) >= thresh
                or abs(new.waist_roll_deg - old.waist_roll_deg) >= thresh
                or abs(new.waist_yaw_deg - old.waist_yaw_deg) >= thresh
            )
        # Discrete commands: any field change counts (intent/magnitude
        # already compared above; equality on the dataclass covers any
        # future fields too).
        return new != old

    # -- introspection (for sidecar logging) ----------------------------------

    def last_emitted(self) -> Optional[LocomotionCmd]:
        """Return the most recent emitted command, or ``None`` if never."""
        return self._last_emitted


__all__ = [
    "StreamMode",
    "LocomotionCmd",
    "ModeTransition",
    "IntentDecoder",
]
