"""Tests for the v7 continuous-waist B-press latch in quest3_manager_x2.

Locks in:

  - On ``LOCOMOTION -> ARM_MANIPULATION`` the manager publishes a
    ``hold_torso(latched)`` planner_cmd with the live waist target,
    populates ``self._latched_waist``, and (when the latched pose is
    non-neutral) plays the ``mode_torso_locked`` audio cue.
  - On ``ARM_MANIPULATION -> LOCOMOTION`` the manager clears
    ``self._latched_waist`` and publishes ``idle/default``.
  - The wire payload built by :func:`_planner_cmd_payload` carries the
    waist_*_deg fields ONLY for ``hold_torso`` commands (plain idle /
    walk / turn payloads stay pre-v7 byte-compatible).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.quest3_manager_x2 import (  # noqa: E402
    ManagerConfig,
    Quest3ManagerX2,
    _planner_cmd_payload,
)
from gear_sonic.utils.teleop.vr.intent_decoder import (  # noqa: E402
    LocomotionCmd,
    ModeTransition,
    StreamMode,
)


# ---------------------------------------------------------------------------
# Wire format unit tests (no manager)
# ---------------------------------------------------------------------------


def test_planner_cmd_payload_omits_waist_fields_for_idle() -> None:
    cmd = LocomotionCmd("idle", "default")
    payload = json.loads(_planner_cmd_payload(cmd).decode("utf-8"))
    assert payload == {"intent": "idle", "magnitude": "default"}
    assert "waist_pitch_deg" not in payload


def test_planner_cmd_payload_omits_waist_fields_for_walk() -> None:
    cmd = LocomotionCmd("walk", "forward")
    payload = json.loads(_planner_cmd_payload(cmd).decode("utf-8"))
    assert payload == {"intent": "walk", "magnitude": "forward"}


def test_planner_cmd_payload_includes_waist_fields_for_hold_torso() -> None:
    cmd = LocomotionCmd(
        intent="hold_torso",
        magnitude="continuous",
        waist_pitch_deg=12.5,
        waist_roll_deg=-3.0,
        waist_yaw_deg=25.0,
    )
    payload = json.loads(_planner_cmd_payload(cmd).decode("utf-8"))
    assert payload == {
        "intent": "hold_torso",
        "magnitude": "continuous",
        "waist_pitch_deg": 12.5,
        "waist_roll_deg": -3.0,
        "waist_yaw_deg": 25.0,
    }


def test_planner_cmd_payload_includes_zero_waist_fields_for_neutral_hold() -> None:
    cmd = LocomotionCmd(intent="hold_torso", magnitude="continuous")
    payload = json.loads(_planner_cmd_payload(cmd).decode("utf-8"))
    assert payload["waist_pitch_deg"] == 0.0
    assert payload["waist_roll_deg"] == 0.0
    assert payload["waist_yaw_deg"] == 0.0


# ---------------------------------------------------------------------------
# B-latch end-to-end (real manager, mocked publishers)
# ---------------------------------------------------------------------------


@pytest.fixture
def manager(tmp_path):
    real_cal = REPO_ROOT / "data" / "operator_calibrations" / "default.yaml"
    if not real_cal.is_file():
        pytest.skip(f"requires {real_cal}")
    cfg = ManagerConfig(calibration_path=real_cal)
    cfg.planner_cmd_port = 25571
    cfg.recorder_pub_port = 25572
    cfg.planner_cmd_host = "127.0.0.1"
    cfg.recorder_pub_host = "127.0.0.1"
    mgr = Quest3ManagerX2(cfg)
    yield mgr
    mgr.stop()


def _capture_planner_cmds(manager) -> list[LocomotionCmd]:
    """Replace the planner_cmd publisher with a recorder."""
    captured: list[LocomotionCmd] = []
    def _cap(cmd: LocomotionCmd) -> None:
        captured.append(cmd)
    manager._publish_planner_cmd = _cap
    return captured


def _capture_audio(manager) -> list[dict]:
    sent: list[dict] = []
    manager._quest.send_message = lambda payload: (sent.append(payload), True)[1]
    return sent


def test_locomotion_to_arm_man_with_neutral_waist_latches_zero_pose(manager):
    cmds = _capture_planner_cmds(manager)
    sent = _capture_audio(manager)

    transition = ModeTransition(
        previous=StreamMode.LOCOMOTION,
        current=StreamMode.ARM_MANIPULATION,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
        live_waist_target=(0.0, 0.0, 0.0),
    )

    assert manager._latched_waist == (0.0, 0.0, 0.0)
    hold_cmds = [c for c in cmds if c.intent == "hold_torso"]
    assert len(hold_cmds) == 1
    h = hold_cmds[0]
    assert h.magnitude == "continuous"
    assert (h.waist_pitch_deg, h.waist_roll_deg, h.waist_yaw_deg) == (0.0, 0.0, 0.0)
    # No torso_locked cue for neutral pose -- only the standard mode prompt.
    keys = [m.get("key") for m in sent if m.get("_type") == "play_audio"]
    assert "mode_torso_locked" not in keys
    assert "mode_arm_manipulation" in keys


def test_locomotion_to_arm_man_with_nonneutral_waist_latches_and_cues(manager):
    cmds = _capture_planner_cmds(manager)
    sent = _capture_audio(manager)

    transition = ModeTransition(
        previous=StreamMode.LOCOMOTION,
        current=StreamMode.ARM_MANIPULATION,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
        live_waist_target=(15.0, -5.0, 30.0),
    )

    assert manager._latched_waist == (15.0, -5.0, 30.0)
    hold_cmds = [c for c in cmds if c.intent == "hold_torso"]
    assert len(hold_cmds) == 1
    h = hold_cmds[0]
    assert (h.waist_pitch_deg, h.waist_roll_deg, h.waist_yaw_deg) == (15.0, -5.0, 30.0)

    keys = [m.get("key") for m in sent if m.get("_type") == "play_audio"]
    assert "mode_torso_locked" in keys
    assert "mode_arm_manipulation" in keys


def test_arm_man_to_locomotion_clears_latch_and_emits_idle(manager):
    cmds = _capture_planner_cmds(manager)
    _capture_audio(manager)

    # Pretend we're already latched.
    manager._latched_waist = (10.0, 0.0, 20.0)

    transition = ModeTransition(
        previous=StreamMode.ARM_MANIPULATION,
        current=StreamMode.LOCOMOTION,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
        live_waist_target=(0.0, 0.0, 0.0),
    )

    assert manager._latched_waist is None
    idle_cmds = [c for c in cmds if c.intent == "idle"]
    assert any(c.magnitude == "default" for c in idle_cmds)


def test_active_to_off_clears_latch(manager):
    cmds = _capture_planner_cmds(manager)
    _capture_audio(manager)

    manager._latched_waist = (8.0, 2.0, 5.0)
    transition = ModeTransition(
        previous=StreamMode.ARM_MANIPULATION,
        current=StreamMode.OFF,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
    )
    assert manager._latched_waist is None
    # An idle planner cmd must be issued so the planner clears its queue.
    assert any(c.intent == "idle" for c in cmds)


def test_off_to_locomotion_does_not_publish_hold_torso(manager):
    """Engage chord must not accidentally synthesize a hold_torso cmd."""
    cmds = _capture_planner_cmds(manager)
    _capture_audio(manager)

    transition = ModeTransition(
        previous=StreamMode.OFF, current=StreamMode.LOCOMOTION,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
        live_waist_target=(0.0, 0.0, 0.0),
    )
    hold_cmds = [c for c in cmds if c.intent == "hold_torso"]
    assert hold_cmds == []
    assert manager._latched_waist is None


# ---------------------------------------------------------------------------
# R-thumbstick-click waist freeze toggle
# ---------------------------------------------------------------------------


def test_toggle_waist_freeze_on_captures_live_target(manager):
    sent = _capture_audio(manager)

    assert manager._waist_frozen is False
    assert manager._latched_waist is None

    manager._toggle_waist_freeze((10.0, -2.5, 18.0))

    assert manager._waist_frozen is True
    assert manager._latched_waist == (10.0, -2.5, 18.0)
    keys = [m.get("key") for m in sent if m.get("_type") == "play_audio"]
    assert "torso_frozen" in keys


def test_toggle_waist_freeze_off_clears_latch(manager):
    sent = _capture_audio(manager)
    manager._waist_frozen = True
    manager._latched_waist = (12.0, 0.0, 0.0)

    manager._toggle_waist_freeze((0.0, 0.0, 0.0))

    assert manager._waist_frozen is False
    assert manager._latched_waist is None
    keys = [m.get("key") for m in sent if m.get("_type") == "play_audio"]
    assert "torso_released" in keys


def test_toggle_waist_freeze_round_trip(manager):
    """Two clicks cancel out -- back to the starting state."""
    _capture_audio(manager)
    manager._toggle_waist_freeze((5.0, 0.0, 0.0))
    assert manager._waist_frozen is True
    manager._toggle_waist_freeze((0.0, 0.0, 0.0))
    assert manager._waist_frozen is False
    assert manager._latched_waist is None


# ---------------------------------------------------------------------------
# Freeze interaction with B-press mode transitions
# ---------------------------------------------------------------------------


def test_loco_to_arm_with_active_freeze_uses_latched_pose(manager):
    """If the operator R-click-froze first, the B-press into ARM_MAN
    must republish the LATCHED pose (not whatever the live stick now
    samples) so the body pose doesn't drift across the transition."""
    cmds = _capture_planner_cmds(manager)
    _capture_audio(manager)

    manager._waist_frozen = True
    manager._latched_waist = (15.0, -3.0, 25.0)

    transition = ModeTransition(
        previous=StreamMode.LOCOMOTION,
        current=StreamMode.ARM_MANIPULATION,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
        live_waist_target=(2.0, 0.0, 5.0),  # different from the frozen pose
    )

    hold_cmds = [c for c in cmds if c.intent == "hold_torso"]
    assert len(hold_cmds) == 1
    h = hold_cmds[0]
    assert (h.waist_pitch_deg, h.waist_roll_deg, h.waist_yaw_deg) == (15.0, -3.0, 25.0)
    assert manager._latched_waist == (15.0, -3.0, 25.0)
    assert manager._waist_frozen is True


def test_arm_to_loco_with_active_freeze_keeps_hold(manager):
    """If the freeze is on, B-press back to LOCO must NOT publish idle
    and must NOT clear the latch -- the operator is walking with a
    locked-in lean and we want the body to stay there."""
    cmds = _capture_planner_cmds(manager)
    _capture_audio(manager)

    manager._waist_frozen = True
    manager._latched_waist = (12.0, 0.0, 20.0)

    transition = ModeTransition(
        previous=StreamMode.ARM_MANIPULATION,
        current=StreamMode.LOCOMOTION,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
        live_waist_target=(0.0, 0.0, 0.0),
    )

    idle_cmds = [c for c in cmds if c.intent == "idle"]
    assert idle_cmds == [], (
        "ARM->LOCO with active freeze should NOT publish an idle release; "
        f"got {idle_cmds}"
    )
    assert manager._waist_frozen is True
    assert manager._latched_waist == (12.0, 0.0, 20.0)


def test_arm_to_loco_without_freeze_releases_as_today(manager):
    """Without an explicit R-click freeze, the existing release path
    still fires (back-compat with the v7 B-latch behavior)."""
    cmds = _capture_planner_cmds(manager)
    _capture_audio(manager)

    manager._waist_frozen = False
    manager._latched_waist = (8.0, 0.0, 15.0)  # left over from B-latch entry

    transition = ModeTransition(
        previous=StreamMode.ARM_MANIPULATION,
        current=StreamMode.LOCOMOTION,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
        live_waist_target=(0.0, 0.0, 0.0),
    )

    assert manager._latched_waist is None
    assert any(c.intent == "idle" and c.magnitude == "default" for c in cmds)


def test_off_transition_clears_freeze(manager):
    """OFF resets all input state, including the R-click freeze."""
    cmds = _capture_planner_cmds(manager)
    _capture_audio(manager)

    manager._waist_frozen = True
    manager._latched_waist = (10.0, 0.0, 0.0)

    transition = ModeTransition(
        previous=StreamMode.LOCOMOTION,
        current=StreamMode.OFF,
    )
    manager._on_mode_transition(
        transition,
        vr_pose=np.zeros(9, dtype=np.float32),
        tick=0,
    )

    assert manager._waist_frozen is False
    assert manager._latched_waist is None
    assert any(c.intent == "idle" for c in cmds)
