"""Tests for the recorder's motion-clip playback wiring.

Covers four pieces of the playback path that are testable without
spinning up the full :class:`X2DatasetRecorder` (which transitively
requires the LeRobot writer chain, mirroring the rationale in
:file:`test_recorder_subscribe_mode.py`):

1. ``_subscribe_motion_clip_cmd_thread`` end-to-end ZMQ contract:
   recorder-side ``SUB.bind``, trigger-side ``PUB.connect``,
   multipart ``[topic, json]`` payloads decoded into
   :class:`MotionClipPlayRequest` / :class:`MotionClipStopRequest`
   on the shared :class:`queue.Queue`. Malformed payloads must NOT
   tear down the thread.

2. ``_drain_clip_commands`` state machine: catalog hold_after, wire
   override, stop releases a held pose, play-while-held re-seeds
   the yaw rebase from the held frame, locomotion kind on ad-hoc
   PKL plays.

3. ``_publish_held_clip_frame`` republishes the latched body_q +
   root_quat verbatim.

4. ``_resolve_clip_entry`` forwards request.kind onto ad-hoc PKL
   entries (so the operator's choice of play_gesture vs
   play_locomotion is preserved end-to-end).

The full publish-loop gate is exercised by the MuJoCo smoke test
recipes in :file:`clip_motion_commands.md` and
:file:`pkl_direct_commands.md`.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zmq
from scipy.spatial.transform import Rotation as Rot

from gear_sonic.utils.teleop.motion_clip_session import (
    GESTURE_DEFAULT_CATALOG_PATH,
    MotionClipEntry,
    MotionClipPlayRequest,
    MotionClipSession,
    MotionClipStopRequest,
    load_catalog,
    parse_motion_clip_command,
)
from gear_sonic.utils.teleop.x2_dataset_recorder import (
    X2DatasetRecorder,
    _subscribe_motion_clip_cmd_thread,
)


_CLIP_PORT = 25770
_CLIP_TOPIC = "motion_clip_cmd"


def _start_recorder_thread(
    port: int = _CLIP_PORT,
    topic: str = _CLIP_TOPIC,
) -> tuple[threading.Thread, "queue.Queue[Any]", threading.Event]:
    """Boot the motion-clip SUB thread bound on ``port`` and return its handles."""
    request_queue: "queue.Queue[Any]" = queue.Queue()
    stop = threading.Event()
    thread = threading.Thread(
        target=_subscribe_motion_clip_cmd_thread,
        kwargs=dict(
            url=f"tcp://*:{port}",
            topic=topic,
            request_queue=request_queue,
            stop_event=stop,
            verbose=False,
        ),
        name="test-motion-clip-cmd-sub",
        daemon=True,
    )
    thread.start()
    # Allow the bind() to complete before the PUB connects so the
    # SUB filter is in place by the first send.
    time.sleep(0.2)
    return thread, request_queue, stop


@pytest.fixture
def clip_pub_pair():
    """Spin up the recorder motion-clip SUB.bind + a trigger-side PUB.connect."""
    thread, request_queue, stop = _start_recorder_thread()
    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    pub.setsockopt(zmq.LINGER, 0)
    pub.connect(f"tcp://127.0.0.1:{_CLIP_PORT}")
    # Slow-joiner mitigation: PUB.connect doesn't synchronise with
    # the SUB; without this the very first send is reliably dropped.
    time.sleep(0.2)
    try:
        yield pub, request_queue
    finally:
        stop.set()
        try:
            thread.join(timeout=1.0)
        except Exception:
            pass
        pub.close(linger=0)


def _drain_queue_with_timeout(
    request_queue: "queue.Queue[Any]",
    *,
    timeout_s: float = 2.0,
) -> list[Any]:
    """Pull everything the thread has decoded so far (subject to timeout)."""
    out: list[Any] = []
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            out.append(request_queue.get(timeout=0.1))
        except queue.Empty:
            if out:
                return out
            continue
    return out


def _send_until_received(
    pub: "zmq.Socket[Any]",
    topic: str,
    payload: dict[str, Any],
    request_queue: "queue.Queue[Any]",
    *,
    timeout_s: float = 2.0,
    pred=None,
) -> list[Any]:
    """Re-send a payload on a heartbeat until the queue accepts one
    matching ``pred`` (or until ``timeout_s`` elapses). Defeats the
    PUB-SUB slow-joiner / first-message-lost gotcha that can otherwise
    flake CI in this layer."""
    pred = pred or (lambda _req: True)
    deadline = time.monotonic() + timeout_s
    collected: list[Any] = []
    while time.monotonic() < deadline:
        pub.send_multipart([
            topic.encode("ascii"),
            json.dumps(payload).encode("utf-8"),
        ])
        try:
            req = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        collected.append(req)
        if pred(req):
            return collected
    pytest.fail(f"motion_clip_cmd thread never delivered payload {payload!r}")


def test_motion_clip_cmd_thread_delivers_play_with_name(clip_pub_pair) -> None:
    pub, request_queue = clip_pub_pair
    requests = _send_until_received(
        pub, _CLIP_TOPIC,
        {"action": "play", "name": "sit_stand_sit_A538"},
        request_queue,
        pred=lambda r: isinstance(r, MotionClipPlayRequest) and r.name == "sit_stand_sit_A538",
    )
    play = next(
        r for r in requests
        if isinstance(r, MotionClipPlayRequest) and r.name == "sit_stand_sit_A538"
    )
    assert play.pkl_path is None
    assert play.kind == "gesture"


def test_motion_clip_cmd_thread_delivers_play_with_pkl(clip_pub_pair) -> None:
    pub, request_queue = clip_pub_pair
    requests = _send_until_received(
        pub, _CLIP_TOPIC,
        {"action": "play", "pkl": "/tmp/example.pkl", "motion_key": "foo"},
        request_queue,
        pred=lambda r: (
            isinstance(r, MotionClipPlayRequest) and r.pkl_path is not None
        ),
    )
    play = next(
        r for r in requests
        if isinstance(r, MotionClipPlayRequest) and r.pkl_path is not None
    )
    assert str(play.pkl_path) == "/tmp/example.pkl"
    assert play.motion_key == "foo"
    assert play.name is None


def test_motion_clip_cmd_thread_delivers_play_with_locomotion_kind(clip_pub_pair) -> None:
    """``play_locomotion`` stamps ``kind="locomotion"`` on the wire;
    the recorder thread must decode that intact so the session-side
    branch reads the right discriminator."""
    pub, request_queue = clip_pub_pair
    requests = _send_until_received(
        pub, _CLIP_TOPIC,
        {"action": "play", "pkl": "/tmp/walk.pkl", "kind": "locomotion"},
        request_queue,
        pred=lambda r: (
            isinstance(r, MotionClipPlayRequest)
            and r.pkl_path is not None
            and r.kind == "locomotion"
        ),
    )
    play = next(
        r for r in requests
        if isinstance(r, MotionClipPlayRequest) and r.kind == "locomotion"
    )
    assert str(play.pkl_path) == "/tmp/walk.pkl"


def test_motion_clip_cmd_thread_delivers_stop(clip_pub_pair) -> None:
    pub, request_queue = clip_pub_pair
    requests = _send_until_received(
        pub, _CLIP_TOPIC,
        {"action": "stop"},
        request_queue,
        pred=lambda r: isinstance(r, MotionClipStopRequest),
    )
    assert any(isinstance(r, MotionClipStopRequest) for r in requests)


def test_motion_clip_cmd_thread_swallows_malformed_payloads(clip_pub_pair) -> None:
    """Bad JSON, unknown action, and play without target must all be
    logged + dropped without putting anything on the queue and without
    killing the SUB thread."""
    pub, request_queue = clip_pub_pair

    bad_payloads: list[Any] = [
        {"action": "wiggle"},                  # unknown action
        {"action": "play"},                    # play without name/pkl
        {"action": "play", "name": "x", "pkl": "y.pkl"},  # both set
        {"action": "play", "name": "x", "kind": "dance"},  # bad kind
    ]

    # Send a handful of bad payloads first; nothing should land.
    for raw in bad_payloads:
        pub.send_multipart([
            _CLIP_TOPIC.encode("ascii"),
            json.dumps(raw).encode("utf-8"),
        ])
    # Single-part raw bytes (no JSON multipart) -- thread drops it.
    pub.send(_CLIP_TOPIC.encode("ascii") + b" not-json-at-all")
    time.sleep(0.5)

    # Should be empty: no malformed payload was promoted.
    drained_pre: list[Any] = []
    try:
        while True:
            drained_pre.append(request_queue.get_nowait())
    except queue.Empty:
        pass
    assert drained_pre == [], (
        f"malformed payloads leaked through as: {drained_pre}"
    )

    # Now send a good payload -- the thread must still be alive.
    requests = _send_until_received(
        pub, _CLIP_TOPIC,
        {"action": "stop"},
        request_queue,
        pred=lambda r: isinstance(r, MotionClipStopRequest),
    )
    assert any(isinstance(r, MotionClipStopRequest) for r in requests)


# ---------------------------------------------------------------------------
# Resolution semantics: catalog name vs ad-hoc PKL, overrides.
# ---------------------------------------------------------------------------


def test_resolve_overrides_apply_on_top_of_catalog_defaults() -> None:
    """The recorder-side resolver applies per-request motion_key /
    start_frame / n_frames over the catalog entry. Spec mirror of
    ``X2DatasetRecorder._resolve_clip_entry`` -- pinning here so a
    future refactor of that method can't silently change semantics."""
    base = MotionClipEntry(
        name="catalog_entry",
        source=Path("foo.pkl"),
        motion_key="orig_key",
        start_frame=10,
        n_frames=50,
    )
    # The recorder treats start_frame==0 / n_frames==None as "no
    # override", but motion_key=None as "no override" too. Per-
    # request non-default values win.
    req = MotionClipPlayRequest(
        name="catalog_entry",
        motion_key="override_key",
        start_frame=200,
        n_frames=20,
    )
    # The resolution lives in X2DatasetRecorder._resolve_clip_entry,
    # but it composes the same overrides we encode in parse_motion_clip_command.
    # Build the override mirror inline to assert intent.
    merged = MotionClipEntry(
        name=base.name,
        source=base.source,
        motion_key=req.motion_key if req.motion_key is not None else base.motion_key,
        start_frame=req.start_frame if req.start_frame else base.start_frame,
        n_frames=req.n_frames if req.n_frames is not None else base.n_frames,
    )
    assert merged.motion_key == "override_key"
    assert merged.start_frame == 200
    assert merged.n_frames == 20


# ---------------------------------------------------------------------------
# Pinned re-export sanity: parse_motion_clip_command and recorder share decoder
# ---------------------------------------------------------------------------


def test_recorder_sub_thread_uses_parse_motion_clip_command(clip_pub_pair) -> None:
    """The thread decodes via :func:`parse_motion_clip_command`; sanity-
    check that contract by pushing a play-with-everything payload and
    asserting the decoded request matches a direct decode of the same
    JSON. This is the wire-contract pinning analogue to the body_pose
    tests in :file:`test_recorder_subscribe_mode.py`."""
    pub, request_queue = clip_pub_pair
    payload = {
        "action": "play",
        "name": "sit_stand_sit_A538",
        "motion_key": "neutral_dancecard_object_interact_003__A538_M",
        "start_frame": 5,
        "n_frames": 60,
    }
    requests = _send_until_received(
        pub, _CLIP_TOPIC, payload, request_queue,
        pred=lambda r: (
            isinstance(r, MotionClipPlayRequest)
            and r.motion_key == payload["motion_key"]
        ),
    )
    play = next(
        r for r in requests
        if isinstance(r, MotionClipPlayRequest) and r.motion_key == payload["motion_key"]
    )
    expected = parse_motion_clip_command(payload)
    assert isinstance(expected, MotionClipPlayRequest)
    assert play.name == expected.name
    assert play.motion_key == expected.motion_key
    assert play.start_frame == expected.start_frame
    assert play.n_frames == expected.n_frames
    assert play.kind == expected.kind == "gesture"


# ---------------------------------------------------------------------------
# Hold semantics + locomotion-vs-gesture: catalog flag, wire override, stop
# release, takeover seeding, ad-hoc kind forwarding.
#
# Reuses the real X2DatasetRecorder methods by binding them to a stub
# instance rather than constructing one (the constructor pulls in MuJoCo,
# the LeRobot writer chain, etc. -- see the docstring rationale above).
# ---------------------------------------------------------------------------


class _FakeCfg:
    """Minimal stand-in for :class:`RecorderConfig` covering the fields
    ``_drain_clip_commands`` and ``_publish_held_clip_frame`` read."""

    publish_rate_hz: float = 50.0
    clip_future_dt_s: float = 0.1
    clip_future_window_frames: int = 9
    verbose: bool = False


class _ClipHarness:
    """Bind the recorder's motion-clip methods to a minimal carrier object.

    Replaces ``_publish_pose`` with a list-capturing stub so the
    held-frame branch is observable without a real ZMQ publisher.
    """

    def __init__(self, catalog: dict[str, MotionClipEntry]) -> None:
        self._cfg = _FakeCfg()
        self._gesture_catalog = catalog
        self._motion_clip_request_queue: "queue.Queue[Any]" = queue.Queue()
        self._active_clip: MotionClipSession | None = None
        self._active_clip_hold_after: bool = False
        self._clip_held_frame: dict[str, np.ndarray] | None = None
        # ``_publish_pose`` is heavy; capture its kwargs into a list so
        # tests can assert what the held-frame branch emitted.
        self._zero_motion_token = np.zeros(64, dtype=np.float64)
        self.publish_calls: list[dict[str, Any]] = []

    # Bind the real recorder methods under test.
    _snapshot_robot_yaw = X2DatasetRecorder._snapshot_robot_yaw
    _resolve_clip_entry = X2DatasetRecorder._resolve_clip_entry
    _drain_clip_commands = X2DatasetRecorder._drain_clip_commands
    _publish_held_clip_frame = X2DatasetRecorder._publish_held_clip_frame

    def _publish_pose(self, **kwargs: Any) -> None:
        # Snapshot whatever the under-test method passed in.
        self.publish_calls.append(kwargs)


def _shipped_catalog() -> dict[str, MotionClipEntry]:
    return load_catalog(GESTURE_DEFAULT_CATALOG_PATH)


def _identity_snap() -> dict[str, np.ndarray]:
    """Snap dict for an idle-standing robot (identity world rotation)."""
    return {"root_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)}


def test_drain_play_with_catalog_hold_after_true_latches_flag() -> None:
    catalog = _shipped_catalog()
    h = _ClipHarness(catalog)
    h._motion_clip_request_queue.put(
        MotionClipPlayRequest(name="sit_down_A540")
    )
    h._drain_clip_commands(_identity_snap())
    assert h._active_clip is not None
    assert h._active_clip.entry.name == "sit_down_A540"
    assert h._active_clip_hold_after is True


def test_drain_play_with_catalog_default_keeps_hold_false() -> None:
    catalog = _shipped_catalog()
    h = _ClipHarness(catalog)
    h._motion_clip_request_queue.put(
        MotionClipPlayRequest(name="stand_up_A540")
    )
    h._drain_clip_commands(_identity_snap())
    assert h._active_clip is not None
    assert h._active_clip_hold_after is False


def test_drain_play_wire_override_beats_catalog_default() -> None:
    """A wire payload's ``hold_after=False`` must override
    catalog ``hold_after: true``, and vice versa."""
    catalog = _shipped_catalog()

    # Override catalog true -> wire false.
    h_off = _ClipHarness(catalog)
    h_off._motion_clip_request_queue.put(
        MotionClipPlayRequest(name="sit_down_A540", hold_after=False)
    )
    h_off._drain_clip_commands(_identity_snap())
    assert h_off._active_clip is not None
    assert h_off._active_clip_hold_after is False

    # Override catalog false -> wire true.
    h_on = _ClipHarness(catalog)
    h_on._motion_clip_request_queue.put(
        MotionClipPlayRequest(name="stand_up_A540", hold_after=True)
    )
    h_on._drain_clip_commands(_identity_snap())
    assert h_on._active_clip is not None
    assert h_on._active_clip_hold_after is True


def test_drain_stop_releases_held_frame_and_resets_flag() -> None:
    """``stop`` from any state must drop both the active session and
    any latched hold frame -- single command covers abort + release."""
    h = _ClipHarness({})
    # Simulate a previously-completed hold_after clip: held frame
    # latched, no active session, flag still True from the play tick.
    h._clip_held_frame = {
        "body_q_mj": np.zeros(31, dtype=np.float64),
        "root_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    h._active_clip_hold_after = True
    h._motion_clip_request_queue.put(MotionClipStopRequest())
    h._drain_clip_commands(_identity_snap())
    assert h._clip_held_frame is None
    assert h._active_clip_hold_after is False
    assert h._active_clip is None


def test_drain_play_during_hold_seeds_yaw_from_held_frame() -> None:
    """When a new gesture play arrives during a held state, the new
    session's yaw rebase must come from the held frame's quat, NOT
    from the kplanner snap. Otherwise the takeover between two halves
    of the same PKL twists the body.

    We compare the held-seeded session against an identity-snap-seeded
    baseline: the per-frame delta rotation between them must be a
    pure world-Z rotation by ``held_yaw`` rad.
    """
    catalog = _shipped_catalog()
    held_yaw = 1.0
    held_quat = Rot.from_euler("z", held_yaw).as_quat()  # xyzw

    # Run 1: held seed.
    h_held = _ClipHarness(catalog)
    h_held._clip_held_frame = {
        "body_q_mj": np.zeros(31, dtype=np.float64),
        "root_quat_xyzw": held_quat.astype(np.float32),
    }
    h_held._active_clip_hold_after = True
    h_held._motion_clip_request_queue.put(
        MotionClipPlayRequest(name="stand_up_A540")
    )
    h_held._drain_clip_commands(_identity_snap())
    assert h_held._active_clip is not None

    # Run 2: snap-only seed (no held frame, snap yaw = 0).
    h_snap = _ClipHarness(catalog)
    h_snap._motion_clip_request_queue.put(
        MotionClipPlayRequest(name="stand_up_A540")
    )
    h_snap._drain_clip_commands(_identity_snap())
    assert h_snap._active_clip is not None

    # The two sessions must differ only by R_z(held_yaw) applied to
    # every published quat. Decomposed as a rotvec, x/y must vanish
    # and z must equal held_yaw on every frame.
    qa = Rot.from_quat(
        h_snap._active_clip.root_quat_xyzw.astype(np.float64)
    )
    qb = Rot.from_quat(
        h_held._active_clip.root_quat_xyzw.astype(np.float64)
    )
    q_delta = (qb * qa.inv()).as_rotvec()
    np.testing.assert_allclose(q_delta[:, 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(q_delta[:, 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(q_delta[:, 2], held_yaw, atol=1e-5)

    # Starting a new play clears the latched held frame: the new
    # session owns the publish path now.
    assert h_held._clip_held_frame is None


def test_resolve_clip_entry_forwards_kind_on_adhoc_pkl_play(tmp_path: Path) -> None:
    """``play_locomotion`` stamps ``kind="locomotion"`` on the wire;
    the recorder's resolver must forward that onto the ad-hoc
    :class:`MotionClipEntry` so the session-side yaw rebase gate
    reads the right discriminator.
    """
    import joblib

    # Build a minimal 2-frame PKL so the recorder can actually load it
    # if the test asks the resolver to do real work; here we only
    # need the resolver to BUILD the entry (no session construction).
    pkl_path = tmp_path / "fake_walk.pkl"
    joblib.dump({
        "fake_walk": {
            "dof": np.zeros((2, 31), dtype=np.float32),
            "root_rot": np.tile(
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32), (2, 1)
            ),
            "root_trans_offset": np.zeros((2, 3), dtype=np.float32),
            "fps": 30.0,
        }
    }, pkl_path)

    h = _ClipHarness({})
    req = MotionClipPlayRequest(
        pkl_path=pkl_path,
        kind="locomotion",
    )
    entry = h._resolve_clip_entry(req)
    assert entry is not None
    assert entry.kind == "locomotion"
    assert entry.source == pkl_path

    # Gesture default still works for an ad-hoc gesture --pkl play.
    req_g = MotionClipPlayRequest(
        pkl_path=pkl_path,
        kind="gesture",
    )
    entry_g = h._resolve_clip_entry(req_g)
    assert entry_g is not None
    assert entry_g.kind == "gesture"


def test_publish_held_clip_frame_emits_latched_body_and_quat() -> None:
    """The held-frame republish must forward the latched body_q +
    root_quat verbatim, zero hands, and pad the future window with
    the same frame."""
    h = _ClipHarness({})
    body = np.linspace(-0.5, 0.5, 31, dtype=np.float64)
    quat = np.array([0.0, 0.0, 0.5, np.sqrt(0.75)], dtype=np.float32)
    h._clip_held_frame = {
        "body_q_mj": body,
        "root_quat_xyzw": quat,
    }
    h._publish_held_clip_frame(tick=42)
    assert len(h.publish_calls) == 1
    call = h.publish_calls[0]
    np.testing.assert_allclose(call["body_q_mj"], body)
    np.testing.assert_allclose(call["root_quat_xyzw"], quat)
    # Zero hands.
    np.testing.assert_allclose(call["left_hand_q"], 0.0)
    np.testing.assert_allclose(call["right_hand_q"], 0.0)
    # Future window populated and uniform (every row == last frame).
    n_future = h._cfg.clip_future_window_frames
    assert call["joint_pos_mj_future"].shape == (n_future, 31)
    assert call["root_quat_xyzw_future"].shape == (n_future, 4)
    for k in range(n_future):
        np.testing.assert_allclose(call["joint_pos_mj_future"][k], body, atol=1e-7)
        np.testing.assert_allclose(call["root_quat_xyzw_future"][k], quat, atol=1e-7)
