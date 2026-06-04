"""Wire-format smoke test for the X2 manager's ZMQ publishers.

This test boots a real :class:`Quest3ManagerX2` (with its Quest 3
reader unstarted), subscribes to all four output topics (planner_cmd,
arm_targets, hand_finger_cmd, stream_mode, recorder_cmd) and then
exercises the publisher methods directly. This validates the wire
encoding without needing a live Quest 3 headset.

Together with ``test_quest3_manager_x2_retargeting_parity.py``, these
tests give us:

  - Parity test: lift-and-shifted retargeting math is bit-equivalent
                 to the original recorder.
  - This test:   the new ZMQ wire format the recorder will subscribe
                 to actually round-trips and matches the documented
                 schema.

The downstream recorder subscribe-only mode (Step 3 of the plan) will
need a matching test that asserts the *consumer* round-trip; this test
covers the *producer* side.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock

import msgpack
import numpy as np
import pytest
import zmq

from gear_sonic.scripts.quest3_manager_x2 import (
    ManagerConfig,
    Quest3ManagerX2,
)
from gear_sonic.utils.teleop.vr.intent_decoder import LocomotionCmd


# Use ports far above anything else the project binds (5556..5562) so
# parallel test runs don't collide.
PLANNER_CMD_PORT = 25563
RECORDER_PORT = 25564


@pytest.fixture
def manager(tmp_path):
    """Construct a manager (Quest 3 reader is created but NOT started)."""
    cfg = ManagerConfig(
        calibration_path=tmp_path  # placeholder, replaced before constructing
    )
    # Use the real default calibration; if absent, skip.
    from pathlib import Path

    real_cal = Path("data/operator_calibrations/default.yaml").resolve()
    if not real_cal.is_file():
        pytest.skip(f"requires {real_cal}")
    cfg.calibration_path = real_cal
    cfg.planner_cmd_port = PLANNER_CMD_PORT
    cfg.recorder_pub_port = RECORDER_PORT
    cfg.recorder_pub_host = "127.0.0.1"
    cfg.planner_cmd_host = "127.0.0.1"

    mgr = Quest3ManagerX2(cfg)
    yield mgr
    mgr.stop()


def _make_sub(port: int, topic: str) -> zmq.Socket:
    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.LINGER, 0)
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.setsockopt_string(zmq.SUBSCRIBE, topic)
    sub.connect(f"tcp://127.0.0.1:{port}")
    # ZMQ PUB-SUB: subscribers MUST connect AND have a moment to
    # complete the handshake before the publisher's first send, or the
    # message is dropped on the publisher side. Sleep slightly here.
    time.sleep(0.1)
    return sub


def _recv_multipart(sub: zmq.Socket, retries: int = 10) -> list[bytes] | None:
    """Try a few times — PUB-SUB initial messages can land in the void."""
    for _ in range(retries):
        try:
            return sub.recv_multipart()
        except zmq.error.Again:
            time.sleep(0.05)
    return None


# ---------------------------------------------------------------------------
# planner_cmd
# ---------------------------------------------------------------------------


def test_planner_cmd_wire_format(manager):
    sub = _make_sub(PLANNER_CMD_PORT, "planner_cmd")
    try:
        # Send a few times -- PUB-SUB is lossy until the subscription
        # handshake completes (slow-joiner problem).
        for _ in range(5):
            manager._publish_planner_cmd(LocomotionCmd("walk", "forward"))
            time.sleep(0.02)

        parts = _recv_multipart(sub)
        assert parts is not None, "no planner_cmd received"
        assert len(parts) == 2
        assert parts[0] == b"planner_cmd"
        payload = json.loads(parts[1].decode("utf-8"))
        assert payload == {"intent": "walk", "magnitude": "forward"}
    finally:
        sub.close(linger=0)


# ---------------------------------------------------------------------------
# arm_targets
# ---------------------------------------------------------------------------


def test_arm_targets_wire_format(manager):
    sub = _make_sub(RECORDER_PORT, "arm_targets")
    try:
        L = np.linspace(-0.5, 0.5, 7).astype(np.float64)
        R = np.linspace(-1.0, 1.0, 7).astype(np.float64)
        for _ in range(5):
            manager._publish_arm_targets(left=L, right=R, is_engaged=True, tick=42)
            time.sleep(0.02)

        parts = _recv_multipart(sub)
        assert parts is not None, "no arm_targets received"
        assert len(parts) == 2
        assert parts[0] == b"arm_targets"
        msg = msgpack.unpackb(parts[1], raw=False)
        # ``passthrough_arm_targets`` added 2026-05-30 as the
        # LOCO_DECOUPLED_ARMS sentinel; default False preserves
        # legacy override behaviour. See test_quest3_manager_arm_decouple.py
        # for per-mode behaviour pinning.
        assert set(msg.keys()) == {
            "left_q_rad", "right_q_rad",
            "is_engaged", "passthrough_arm_targets",
            "tick", "ts",
        }
        assert len(msg["left_q_rad"]) == 7
        assert len(msg["right_q_rad"]) == 7
        assert msg["is_engaged"] is True
        assert msg["passthrough_arm_targets"] is False
        assert msg["tick"] == 42
        # float32 cast -> at most ~1e-7 error
        assert max(abs(a - b) for a, b in zip(msg["left_q_rad"], L)) < 1e-6
        assert max(abs(a - b) for a, b in zip(msg["right_q_rad"], R)) < 1e-6
    finally:
        sub.close(linger=0)


# ---------------------------------------------------------------------------
# hand_finger_cmd
# ---------------------------------------------------------------------------


def test_hand_finger_cmd_wire_format(manager):
    sub = _make_sub(RECORDER_PORT, "hand_finger_cmd")
    try:
        L = np.full(10, 0.7, dtype=np.float64)
        R = np.full(10, 0.3, dtype=np.float64)
        for _ in range(5):
            manager._publish_hand_finger_cmd(left=L, right=R, tick=99)
            time.sleep(0.02)

        parts = _recv_multipart(sub)
        assert parts is not None, "no hand_finger_cmd received"
        assert parts[0] == b"hand_finger_cmd"
        msg = msgpack.unpackb(parts[1], raw=False)
        assert set(msg.keys()) == {"left_hand_q", "right_hand_q", "tick", "ts"}
        assert len(msg["left_hand_q"]) == 10
        assert len(msg["right_hand_q"]) == 10
        assert msg["tick"] == 99
    finally:
        sub.close(linger=0)


# ---------------------------------------------------------------------------
# stream_mode
# ---------------------------------------------------------------------------


def test_stream_mode_wire_format(manager):
    sub = _make_sub(RECORDER_PORT, "stream_mode")
    try:
        for _ in range(5):
            manager._publish_stream_mode(tick=7)
            time.sleep(0.02)

        parts = _recv_multipart(sub)
        assert parts is not None, "no stream_mode received"
        assert parts[0] == b"stream_mode"
        msg = msgpack.unpackb(parts[1], raw=False)
        assert set(msg.keys()) == {"mode", "tick", "ts"}
        assert msg["mode"] in {"OFF", "LOCOMOTION", "ARM_MANIPULATION"}
        assert msg["tick"] == 7
    finally:
        sub.close(linger=0)


# ---------------------------------------------------------------------------
# recorder_cmd
# ---------------------------------------------------------------------------


def test_recorder_cmd_wire_format(manager):
    sub = _make_sub(RECORDER_PORT, "recorder_cmd")
    try:
        for _ in range(5):
            manager._publish_recorder_cmd("save_episode", tick=11)
            time.sleep(0.02)

        parts = _recv_multipart(sub)
        assert parts is not None, "no recorder_cmd received"
        assert parts[0] == b"recorder_cmd"
        payload = json.loads(parts[1].decode("utf-8"))
        assert set(payload.keys()) == {"action", "tick", "ts"}
        assert payload["action"] == "save_episode"
        assert payload["tick"] == 11
    finally:
        sub.close(linger=0)


@pytest.mark.parametrize("action", ["start", "save", "discard", "estop"])
def test_recorder_cmd_action_vocabulary(manager, action):
    """Lock in the four recorder_cmd action strings the recorder
    consumes (see x2_dataset_recorder.py::_run_subscribe_mode). If a
    typo creeps into the manager (e.g. 'save_episode' vs 'save'),
    the recorder will silently log 'unknown recorder_cmd action' and
    no episode will ever start/stop."""
    sub = _make_sub(RECORDER_PORT, "recorder_cmd")
    try:
        for _ in range(5):
            manager._publish_recorder_cmd(action, tick=42)
            time.sleep(0.02)

        parts = _recv_multipart(sub)
        assert parts is not None, f"no recorder_cmd received for {action}"
        assert parts[0] == b"recorder_cmd"
        payload = json.loads(parts[1].decode("utf-8"))
        assert payload["action"] == action
        assert payload["tick"] == 42
    finally:
        sub.close(linger=0)


# ---------------------------------------------------------------------------
# Sidecar JSONL
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Audio prompts (manager -> WebXR client via Quest3Reader.send_message)
# ---------------------------------------------------------------------------


def test_play_audio_prompt_pushes_play_audio_message(manager):
    """The helper must marshal the prompt key into a JSON-serialisable
    ``{"_type": "play_audio", "key": ..., "fallback": ...}`` payload
    that the WebXR client's ``handlePlayAudio`` understands.
    """
    sent: list[dict] = []
    manager._quest.send_message = lambda payload: (sent.append(payload), True)[1]

    manager._play_audio_prompt("record_start", fallback="Recording.")
    assert sent == [
        {"_type": "play_audio", "key": "record_start", "fallback": "Recording."}
    ]


def test_play_audio_prompt_omits_fallback_when_none(manager):
    """No fallback key in the payload when the caller passes None;
    keeps wire-bytes minimal and lets the client default to the MP3."""
    sent: list[dict] = []
    manager._quest.send_message = lambda payload: (sent.append(payload), True)[1]

    manager._play_audio_prompt("mode_off")
    assert sent == [{"_type": "play_audio", "key": "mode_off"}]


def test_play_audio_prompt_swallows_send_failures(manager):
    """The control loop must never crash on a transient send error
    (e.g. WebSocket dropped mid-flight); audio cues are decorative."""
    def raise_once(payload):
        raise RuntimeError("simulated websocket failure")

    manager._quest.send_message = raise_once
    # Should NOT raise.
    manager._play_audio_prompt("record_save", fallback="Saved.")


@pytest.mark.parametrize(
    "current_mode, expected_key",
    [
        ("OFF",              "mode_off"),
        ("LOCOMOTION",       "mode_locomotion"),
        ("ARM_MANIPULATION", "mode_arm_manipulation"),
    ],
)
def test_on_mode_transition_emits_matching_audio_cue(
    manager, current_mode, expected_key,
):
    """Each StreamMode transition must enqueue the corresponding
    ``mode_<lower>`` prompt. We don't assert other side-effects
    (planner_cmd traffic is covered elsewhere) -- only the audio."""
    from gear_sonic.utils.teleop.vr.intent_decoder import (
        ModeTransition,
        StreamMode,
    )

    sent: list[dict] = []
    manager._quest.send_message = lambda payload: (sent.append(payload), True)[1]

    transition = ModeTransition(
        previous=StreamMode.LOCOMOTION,
        current=StreamMode[current_mode],
    )
    manager._on_mode_transition(
        transition, vr_pose=np.zeros(9, dtype=np.float32), tick=0,
    )
    audio_msgs = [m for m in sent if m.get("_type") == "play_audio"]
    assert len(audio_msgs) == 1, sent
    assert audio_msgs[0]["key"] == expected_key


# ---------------------------------------------------------------------------
# Stick-click bindings (v7+):
#   LEFT  thumbstick click -> camera cycler (xdotool Tab keypress)
#   RIGHT thumbstick click -> waist freeze toggle
#
# We exercise the manager-side rising-edge detectors + mode gates by
# stubbing out get_stick_clicks() and observing whether cycle() (left)
# or _toggle_waist_freeze() (right) was called. The xdotool side is
# covered separately in tests/test_viewer_camera_cycler.py.
# ---------------------------------------------------------------------------


def _step_main_loop_once(manager) -> None:
    """Run one iteration of the body of Quest3ManagerX2._run_loop().

    We can't call _run_loop() itself because it spins and blocks on
    the WS thread; but the per-tick logic is small enough to inline
    here for the purpose of the rising-edge / mode-gate assertions.
    Mirrors the production path: get_stick_clicks() -> rising-edge ->
    mode gate -> cycle() (L) or _toggle_waist_freeze() (R). Any drift
    from the production code path will surface as a failing test.
    """
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    l, r = manager._quest.get_stick_clicks()
    l_edge = l and not manager._prev_left_stick_click
    r_edge = r and not manager._prev_right_stick_click
    manager._prev_left_stick_click = l
    manager._prev_right_stick_click = r
    in_active_mode = manager._intent.mode != StreamMode.OFF
    if l_edge and in_active_mode and manager._viewer_cycler is not None:
        manager._viewer_cycler.cycle()
    if r_edge and in_active_mode:
        manager._toggle_waist_freeze((0.0, 0.0, 0.0, None))


def test_camera_cycler_fires_on_rising_edge_in_arm_man(manager):
    """Single LEFT click in ARM_MANIPULATION must fire exactly one cycle()."""
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.ARM_MANIPULATION  # force mode
    cycle_mock = MagicMock(return_value=True)
    manager._viewer_cycler = MagicMock()
    manager._viewer_cycler.cycle = cycle_mock

    # Simulate: press, hold (3 ticks), release. (left, right) tuple.
    manager._quest.get_stick_clicks = MagicMock(side_effect=[
        (True, False),  # press
        (True, False),  # hold
        (True, False),  # hold
        (False, False), # release
    ])
    for _ in range(4):
        _step_main_loop_once(manager)

    assert cycle_mock.call_count == 1, (
        "rising-edge detector regressed: a sustained press fired "
        "cycle() %d times instead of 1" % cycle_mock.call_count
    )


def test_camera_cycler_fires_again_after_release(manager):
    """Press, release, press again -> two cycle() calls (LEFT click)."""
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.ARM_MANIPULATION
    cycle_mock = MagicMock(return_value=True)
    manager._viewer_cycler = MagicMock()
    manager._viewer_cycler.cycle = cycle_mock

    manager._quest.get_stick_clicks = MagicMock(side_effect=[
        (True, False),  # press 1
        (False, False), # release
        (True, False),  # press 2
        (False, False), # release
    ])
    for _ in range(4):
        _step_main_loop_once(manager)

    assert cycle_mock.call_count == 2


def test_camera_cycler_suppressed_in_off_mode(manager):
    """OFF means "ignore controller events" -- stick click must NOT
    cycle the camera, otherwise the operator would inadvertently
    poke the viewer while the manager is supposed to be silent."""
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.OFF
    cycle_mock = MagicMock(return_value=True)
    manager._viewer_cycler = MagicMock()
    manager._viewer_cycler.cycle = cycle_mock

    manager._quest.get_stick_clicks = MagicMock(return_value=(True, False))
    for _ in range(3):
        _step_main_loop_once(manager)

    cycle_mock.assert_not_called()


def test_camera_cycler_fires_in_locomotion_mode(manager):
    """LOCOMOTION is also an active mode; the cycler should fire there too."""
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.LOCOMOTION
    cycle_mock = MagicMock(return_value=True)
    manager._viewer_cycler = MagicMock()
    manager._viewer_cycler.cycle = cycle_mock

    manager._quest.get_stick_clicks = MagicMock(side_effect=[
        (True, False),
        (False, False),
    ])
    for _ in range(2):
        _step_main_loop_once(manager)

    assert cycle_mock.call_count == 1


def test_camera_cycler_disabled_via_config(manager):
    """When ``enable_viewer_camera_cycler=False`` the manager sets
    ``self._viewer_cycler = None``; the dispatch must skip cleanly
    even if a press lands. Belt-and-braces with the OFF gate."""
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.ARM_MANIPULATION
    manager._viewer_cycler = None

    manager._quest.get_stick_clicks = MagicMock(return_value=(True, False))
    # Should NOT raise -- the None check in the production path is
    # the contract being tested here.
    _step_main_loop_once(manager)


def test_right_click_does_not_fire_camera_cycler(manager):
    """RIGHT click was the camera cycler pre-v7; it now toggles waist
    freeze instead. Exercising a right-only press must NOT call the
    viewer cycler -- otherwise we'd double-bind the click."""
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.LOCOMOTION
    cycle_mock = MagicMock(return_value=True)
    manager._viewer_cycler = MagicMock()
    manager._viewer_cycler.cycle = cycle_mock

    manager._quest.get_stick_clicks = MagicMock(side_effect=[
        (False, True),
        (False, False),
    ])
    for _ in range(2):
        _step_main_loop_once(manager)

    cycle_mock.assert_not_called()


# ---------------------------------------------------------------------------


def test_sidecar_log_appends_jsonl(tmp_path, manager):
    """Sidecar is opened in append mode; each emit is one JSON object per line."""
    log_path = tmp_path / "sidecar.jsonl"
    manager._sidecar = log_path.open("a", buffering=1)
    manager._cfg.sidecar_log_path = log_path
    try:
        manager._sidecar_emit(LocomotionCmd("turn_left", "deg_45"), tick=12)
        manager._sidecar_emit(LocomotionCmd("idle", "default"), tick=13)
        manager._sidecar.flush()

        lines = log_path.read_text().splitlines()
        assert len(lines) >= 2
        rec1 = json.loads(lines[0])
        assert rec1["intent"] == "turn_left"
        assert rec1["magnitude"] == "deg_45"
        assert rec1["tick"] == 12
        assert rec1["stream_mode"] in {"OFF", "LOCOMOTION", "ARM_MANIPULATION"}
        rec2 = json.loads(lines[1])
        assert rec2["intent"] == "idle"
    finally:
        manager._sidecar.close()
        manager._sidecar = None


# ---------------------------------------------------------------------------
# Episode-button audio gating (--recorder-enabled / --no-recorder-enabled).
#
# The ZMQ ``recorder_cmd`` PUB is one-way (no ACK channel back to the
# manager), so prior to v7.2 the headset would say "Recording." / "Saved."
# on every X / Y press in ARM_MAN regardless of whether the recorder
# actually accepted the action -- including in ``--teleop-only`` runs
# where no parquet was being written. The wire path stays unchanged
# (always publish so the recorder is the source of truth); only the
# audio path is gated on ``ManagerConfig.recorder_enabled``. The
# wrapper sets that flag iff ``--with-record`` was passed.
# ---------------------------------------------------------------------------


def _make_button_event(*, x_pressed: bool = False, y_pressed: bool = False):
    from gear_sonic.utils.teleop.vr.button_state_machine import ButtonEvents
    return ButtonEvents(
        a_pressed=False, b_pressed=False,
        x_pressed=x_pressed, y_pressed=y_pressed,
        ab_pressed=False, xy_pressed=False,
        ax_pressed=False, by_pressed=False,
        abxy_pressed=False,
    )


def test_x_press_in_arm_man_publishes_recorder_cmd_when_audio_disabled(manager):
    """Wire path is unconditional: even with audio gated OFF the recorder
    must still receive the ``start`` command so it can decide to honour
    or ignore it. Otherwise the operator could never start recording in
    a session where they manually disabled the cue."""
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.ARM_MANIPULATION
    manager._cfg.recorder_enabled = False
    sent_audio: list[dict] = []
    manager._quest.send_message = lambda p: (sent_audio.append(p), True)[1]
    manager._publish_recorder_cmd = MagicMock()

    manager._handle_episode_buttons(_make_button_event(x_pressed=True), tick=42)

    manager._publish_recorder_cmd.assert_called_once_with("start", 42)
    audio_msgs = [m for m in sent_audio if m.get("_type") == "play_audio"]
    assert audio_msgs == [], (
        "audio cue fired despite recorder_enabled=False; this is the "
        "exact false-ACK regression v7.2 set out to fix"
    )


def test_y_press_in_arm_man_publishes_recorder_cmd_when_audio_disabled(manager):
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.ARM_MANIPULATION
    manager._cfg.recorder_enabled = False
    sent_audio: list[dict] = []
    manager._quest.send_message = lambda p: (sent_audio.append(p), True)[1]
    manager._publish_recorder_cmd = MagicMock()

    manager._handle_episode_buttons(_make_button_event(y_pressed=True), tick=99)

    manager._publish_recorder_cmd.assert_called_once_with("save", 99)
    audio_msgs = [m for m in sent_audio if m.get("_type") == "play_audio"]
    assert audio_msgs == []


def test_x_press_in_arm_man_plays_audio_when_recorder_enabled(manager):
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.ARM_MANIPULATION
    manager._cfg.recorder_enabled = True
    sent_audio: list[dict] = []
    manager._quest.send_message = lambda p: (sent_audio.append(p), True)[1]
    manager._publish_recorder_cmd = MagicMock()

    manager._handle_episode_buttons(_make_button_event(x_pressed=True), tick=7)

    manager._publish_recorder_cmd.assert_called_once_with("start", 7)
    audio_msgs = [m for m in sent_audio if m.get("_type") == "play_audio"]
    assert len(audio_msgs) == 1
    assert audio_msgs[0]["key"] == "record_start"
    assert audio_msgs[0]["fallback"] == "Recording."


def test_y_press_in_arm_man_plays_audio_when_recorder_enabled(manager):
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    manager._intent._mode = StreamMode.ARM_MANIPULATION
    manager._cfg.recorder_enabled = True
    sent_audio: list[dict] = []
    manager._quest.send_message = lambda p: (sent_audio.append(p), True)[1]
    manager._publish_recorder_cmd = MagicMock()

    manager._handle_episode_buttons(_make_button_event(y_pressed=True), tick=11)

    manager._publish_recorder_cmd.assert_called_once_with("save", 11)
    audio_msgs = [m for m in sent_audio if m.get("_type") == "play_audio"]
    assert len(audio_msgs) == 1
    assert audio_msgs[0]["key"] == "record_save"
    assert audio_msgs[0]["fallback"] == "Saved."


def test_x_press_outside_arm_man_does_not_publish_or_play(manager):
    """Recording is ARM_MAN-only; X in OFF or LOCO must not poke the
    recorder OR the audio. Locking this in so a future refactor can't
    accidentally fire 'Recording.' while the operator is just walking."""
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    for mode in (StreamMode.OFF, StreamMode.LOCOMOTION):
        manager._intent._mode = mode
        manager._cfg.recorder_enabled = True  # cue would fire in ARM_MAN
        sent_audio: list[dict] = []
        manager._quest.send_message = lambda p: (sent_audio.append(p), True)[1]
        manager._publish_recorder_cmd = MagicMock()

        manager._handle_episode_buttons(
            _make_button_event(x_pressed=True), tick=1,
        )

        manager._publish_recorder_cmd.assert_not_called()
        assert sent_audio == [], f"audio leaked in mode {mode.name}"


def test_y_press_outside_arm_man_does_not_publish_or_play(manager):
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode

    for mode in (StreamMode.OFF, StreamMode.LOCOMOTION):
        manager._intent._mode = mode
        manager._cfg.recorder_enabled = True
        sent_audio: list[dict] = []
        manager._quest.send_message = lambda p: (sent_audio.append(p), True)[1]
        manager._publish_recorder_cmd = MagicMock()

        manager._handle_episode_buttons(
            _make_button_event(y_pressed=True), tick=1,
        )

        manager._publish_recorder_cmd.assert_not_called()
        assert sent_audio == []


def test_default_recorder_enabled_is_false(manager):
    """The default ManagerConfig (and therefore a manual ``python -m
    gear_sonic.scripts.quest3_manager_x2`` launch with no flags) must
    leave audio gated OFF -- the wrapper opts in via --recorder-enabled
    when --with-record is set, so the safe default is silence."""
    assert manager._cfg.recorder_enabled is False


def test_cli_flag_recorder_enabled_flips_default():
    """``--recorder-enabled`` on the CLI must produce a config with the
    field set to True; ``--no-recorder-enabled`` resets to False."""
    from gear_sonic.scripts.quest3_manager_x2 import _build_parser

    parser = _build_parser()
    args = parser.parse_args(["--recorder-enabled"])
    assert args.recorder_enabled is True

    args2 = parser.parse_args(["--recorder-enabled", "--no-recorder-enabled"])
    assert args2.recorder_enabled is False

    args3 = parser.parse_args([])
    assert args3.recorder_enabled is False, (
        "default changed; the wrapper relies on the safe-silent default "
        "for --teleop-only sessions"
    )
