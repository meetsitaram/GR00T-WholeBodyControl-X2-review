"""Unit tests for :mod:`gear_sonic.utils.teleop.motion_clip_session`.

Covers:

* Catalog YAML parsing (default catalog + malformed inputs).
* PKL load + resample math (length, dtype, shape).
* Yaw-only root rebase for ``kind="gesture"``: first emitted root_quat
  matches the operator-supplied robot yaw; roll/pitch pass through
  unchanged.
* Locomotion branch (``kind="locomotion"``) applies the SAME rigid
  yaw rebase as gesture (frame 0 yaw aligned with robot yaw) but
  preserves the authored relative yaw evolution across frames --
  so the takeover from idle-stand is C0-continuous in yaw while a
  walk-and-turn clip still turns by the authored amount.
* Future window padding past clip end.
* Session lifecycle (next_frame / is_done / StopIteration).
* JSON command parsing (play with name, play with pkl, stop, errors,
  kind discriminator).

The tests use the shipped sit_stand_sit catalog as the realistic PKL
fixture; if it ever moves, update :data:`_CATALOG_PATH`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from gear_sonic.utils.teleop.motion_clip_session import (
    GESTURE_DEFAULT_CATALOG_PATH,
    MotionClipEntry,
    MotionClipPlayRequest,
    MotionClipSession,
    MotionClipStopRequest,
    X2_NUM_BODY_DOFS,
    estimate_duration_s,
    load_catalog,
    parse_motion_clip_command,
)


_CATALOG_PATH = GESTURE_DEFAULT_CATALOG_PATH
_TARGET_RATE = 50.0


def _first_entry() -> MotionClipEntry:
    cat = load_catalog(_CATALOG_PATH)
    return next(iter(cat.values()))


# ---------------------------------------------------------------------------
# Catalog parsing
# ---------------------------------------------------------------------------


def test_load_default_catalog_has_at_least_one_resolvable_entry() -> None:
    cat = load_catalog(_CATALOG_PATH)
    assert cat, "default gesture catalog ships empty"
    for entry in cat.values():
        src = entry.resolved_source()
        assert src.is_file(), f"catalog entry {entry.name!r} points at missing PKL: {src}"
        # Catalog rows always load as gestures (locomotion uses --pkl).
        assert entry.kind == "gesture"


def test_load_catalog_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_catalog(tmp_path / "does_not_exist.yaml")


def test_load_catalog_rejects_non_mapping(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="top-level YAML must be a mapping"):
        load_catalog(p)


def test_load_catalog_rejects_empty_gestures(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("name: x\ngestures: []\n")
    with pytest.raises(ValueError, match="'gestures' must be a non-empty list"):
        load_catalog(p)


def test_load_catalog_rejects_duplicate_names(tmp_path: Path) -> None:
    p = tmp_path / "dup.yaml"
    p.write_text(
        "name: x\n"
        "gestures:\n"
        "  - {name: foo, source: a.pkl}\n"
        "  - {name: foo, source: b.pkl}\n"
    )
    with pytest.raises(ValueError, match="duplicate gesture name"):
        load_catalog(p)


def test_load_catalog_rejects_missing_required_keys(tmp_path: Path) -> None:
    p = tmp_path / "missing.yaml"
    p.write_text("name: x\ngestures:\n  - {source: a.pkl}\n")
    with pytest.raises(ValueError, match="missing required key"):
        load_catalog(p)


def test_load_catalog_picks_up_hold_after_field(tmp_path: Path) -> None:
    """The optional ``hold_after`` field should round-trip through the
    YAML loader; missing entries default to False so existing catalogs
    keep their auto-handback semantics."""
    p = tmp_path / "hold.yaml"
    p.write_text(
        "name: hold-test\n"
        "gestures:\n"
        "  - {name: holds_after, source: a.pkl, hold_after: true}\n"
        "  - {name: no_hold,     source: b.pkl, hold_after: false}\n"
        "  - {name: omitted,     source: c.pkl}\n"
    )
    cat = load_catalog(p)
    assert cat["holds_after"].hold_after is True
    assert cat["no_hold"].hold_after is False
    assert cat["omitted"].hold_after is False


def test_shipped_catalog_marks_sit_down_A540_hold_after() -> None:
    """Pin the contract of the bundled catalog so a careless YAML edit
    silently demoting sit_down_A540 to auto-handback gets caught."""
    cat = load_catalog(_CATALOG_PATH)
    assert "sit_down_A540" in cat, "expected sit_down_A540 in shipped catalog"
    assert "stand_up_A540" in cat, "expected stand_up_A540 in shipped catalog"
    assert cat["sit_down_A540"].hold_after is True
    assert cat["stand_up_A540"].hold_after is False


# ---------------------------------------------------------------------------
# Session: shape, resample, rebase
# ---------------------------------------------------------------------------


def test_session_loads_first_catalog_entry_and_resamples_to_target_rate() -> None:
    entry = _first_entry()
    sess = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=0.0,
    )
    assert sess.body_q_mj.shape == (sess.n_frames, X2_NUM_BODY_DOFS)
    assert sess.body_q_mj.dtype == np.float32
    assert sess.root_quat_xyzw.shape == (sess.n_frames, 4)
    assert sess.root_quat_xyzw.dtype == np.float32
    # Duration is consistent (frame count / target rate).
    assert sess.duration_s == pytest.approx(sess.n_frames / _TARGET_RATE)
    # Catalog default key is the first one in the PKL (per spec).
    assert sess.motion_key_resolved
    # Catalog entries ship as gestures.
    assert sess.kind == "gesture"


def test_estimate_duration_matches_session_duration() -> None:
    entry = _first_entry()
    est = estimate_duration_s(entry, _TARGET_RATE)
    sess = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=0.0,
    )
    # Match within one tick (estimator and session both floor()).
    assert est == pytest.approx(sess.duration_s, abs=1.0 / _TARGET_RATE)


def test_gesture_root_rebase_aligns_first_frame_yaw_with_robot_yaw() -> None:
    entry = _first_entry()
    target_yaw = 0.7  # rad, not aligned with PKL
    sess = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=target_yaw,
    )
    yaw0 = Rot.from_quat(sess.root_quat_xyzw[0].astype(np.float64)).as_euler("zyx")[0]
    # Tolerance reflects stacked float32 precision loss: PKL is stored
    # f32, the resampler runs in f32, then rotate_quats_yaw_only does
    # one more quat-multiply pass. Empirically ~6e-5 rad.
    assert yaw0 == pytest.approx(target_yaw, abs=1e-4)


def test_gesture_root_rebase_delta_is_pure_world_z_rotation() -> None:
    """Yaw-only rebase rotates every frame's quat by the same Rz(dyaw).

    We can't just compare ZYX Euler pitches because changing yaw on
    a non-aligned quat shuffles the pitch/roll components when
    re-decomposed in ZYX order. The physically meaningful check is
    that ``q_b = Rz(dyaw) * q_a`` for every frame, i.e. the delta
    rotation between the two sessions is a pure z-axis rotation.
    """
    entry = _first_entry()
    sess_a = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=0.0,
    )
    sess_b = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=1.2345,
    )
    qa = Rot.from_quat(sess_a.root_quat_xyzw.astype(np.float64))
    qb = Rot.from_quat(sess_b.root_quat_xyzw.astype(np.float64))
    # qb = q_delta * qa  ->  q_delta = qb * qa.inv()
    q_delta = (qb * qa.inv()).as_rotvec()
    # x/y components of the per-frame rotvec must be ~0 (pure z rot).
    np.testing.assert_allclose(q_delta[:, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(q_delta[:, 1], 0.0, atol=1e-4)
    # z component is the constant yaw delta the rebase applied.
    np.testing.assert_allclose(q_delta[:, 2], q_delta[0, 2], atol=1e-4)


def test_locomotion_kind_rebases_frame0_yaw_to_robot_yaw() -> None:
    """``kind="locomotion"`` applies the same single rigid Rz(dyaw)
    as gesture: frame 0's published yaw must equal the operator-
    supplied robot yaw, so the takeover from idle-stand doesn't
    teleport-rotate the body.
    """
    base_entry = _first_entry()
    loco_entry = MotionClipEntry(
        name=base_entry.name,
        source=base_entry.source,
        motion_key=base_entry.motion_key,
        start_frame=base_entry.start_frame,
        n_frames=base_entry.n_frames,
        hold_after=base_entry.hold_after,
        kind="locomotion",
    )
    target_yaw = 0.7
    sess = MotionClipSession(
        entry=loco_entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=target_yaw,
    )
    assert sess.kind == "locomotion"
    yaw0 = Rot.from_quat(sess.root_quat_xyzw[0].astype(np.float64)).as_euler("zyx")[0]
    # Same tolerance as the gesture variant: stacked f32 quat-multiply noise.
    assert yaw0 == pytest.approx(target_yaw, abs=1e-4)


def test_locomotion_kind_preserves_authored_relative_yaw_evolution() -> None:
    """The rebase is a single Rz(dyaw) shared across frames, so
    per-frame yaw DELTAS must be preserved bit-for-bit between two
    locomotion sessions with different robot_root_yaw seeds. (A
    walk-and-turn clip turning by 90deg authored must still turn
    by 90deg published, regardless of which heading the robot
    started from.)
    """
    base_entry = _first_entry()
    loco_entry = MotionClipEntry(
        name=base_entry.name,
        source=base_entry.source,
        motion_key=base_entry.motion_key,
        start_frame=base_entry.start_frame,
        n_frames=base_entry.n_frames,
        hold_after=base_entry.hold_after,
        kind="locomotion",
    )
    sess_a = MotionClipSession(
        entry=loco_entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=0.0,
    )
    sess_b = MotionClipSession(
        entry=loco_entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=1.7,
    )
    qa = Rot.from_quat(sess_a.root_quat_xyzw.astype(np.float64))
    qb = Rot.from_quat(sess_b.root_quat_xyzw.astype(np.float64))
    # qb = q_delta * qa  ->  q_delta = qb * qa.inv(), a pure world-Z
    # rotation by the robot-yaw difference, identical on every frame
    # (which means the per-frame deltas yaw_k - yaw_0 are the same
    # in both sessions, i.e. authored evolution preserved).
    q_delta = (qb * qa.inv()).as_rotvec()
    np.testing.assert_allclose(q_delta[:, 0], 0.0, atol=1e-4)
    np.testing.assert_allclose(q_delta[:, 1], 0.0, atol=1e-4)
    np.testing.assert_allclose(q_delta[:, 2], q_delta[0, 2], atol=1e-4)
    # Joint angles untouched by the rebase in either branch.
    np.testing.assert_array_equal(sess_a.body_q_mj, sess_b.body_q_mj)


# ---------------------------------------------------------------------------
# Session: next_frame + future window
# ---------------------------------------------------------------------------


def test_next_frame_advances_index_and_flips_is_done_after_exhaustion() -> None:
    entry = _first_entry()
    sess = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=0.0,
    )
    n = sess.n_frames
    assert not sess.is_done()
    for i in range(n):
        b, r = sess.next_frame()
        assert b.shape == (X2_NUM_BODY_DOFS,)
        assert r.shape == (4,)
        assert sess.current_index == i + 1
    assert sess.is_done()
    with pytest.raises(StopIteration):
        sess.next_frame()


def test_future_window_step_and_padding_at_end_of_clip() -> None:
    entry = _first_entry()
    sess = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=0.0,
        future_dt_s=0.1,  # 5 ticks @ 50 Hz
    )
    # Walk to within the last 5 frames so future window must pad.
    while sess.current_index < sess.n_frames - 1:
        sess.next_frame()
    jpos, rot = sess.future_window(9)
    assert jpos.shape == (9, X2_NUM_BODY_DOFS)
    assert rot.shape == (9, 4)
    # All padded rows should equal the final clip frame.
    last_body = sess.body_q_mj[-1]
    last_rot = sess.root_quat_xyzw[-1]
    np.testing.assert_allclose(jpos[-1], last_body)
    np.testing.assert_allclose(rot[-1], last_rot)


def test_future_window_step_matches_dt_and_rate() -> None:
    entry = _first_entry()
    sess = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=0.0,
        future_dt_s=0.1,  # 5 ticks
    )
    # At index 0 (no next_frame called), future_window samples
    # frames [1, 6, 11, ...] (anchor=-1 clamped to 0; step=5).
    jpos, _rot = sess.future_window(3)
    expected_indices = [
        min(sess.n_frames - 1, max(0, 0 - 1) + (k + 1) * 5)
        for k in range(3)
    ]
    for k, idx in enumerate(expected_indices):
        np.testing.assert_allclose(jpos[k], sess.body_q_mj[idx])


def test_future_window_zero_returns_empty() -> None:
    entry = _first_entry()
    sess = MotionClipSession(
        entry=entry,
        target_rate_hz=_TARGET_RATE,
        robot_root_yaw_rad=0.0,
    )
    jpos, rot = sess.future_window(0)
    assert jpos.shape == (0, X2_NUM_BODY_DOFS)
    assert rot.shape == (0, 4)


def test_session_rejects_too_short_clip(tmp_path: Path) -> None:
    """A 1-frame PKL slice can't be resampled (the helper requires >=2)."""
    import joblib

    one_frame = {
        "dof": np.zeros((1, X2_NUM_BODY_DOFS), dtype=np.float32),
        "root_rot": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
        "root_trans_offset": np.zeros((1, 3), dtype=np.float32),
        "fps": 30.0,
    }
    p = tmp_path / "tiny.pkl"
    joblib.dump({"only": one_frame}, p)
    entry = MotionClipEntry(name="tiny", source=p, motion_key=None)
    with pytest.raises(ValueError, match=r"sliced length must be >= 2"):
        MotionClipSession(
            entry=entry,
            target_rate_hz=_TARGET_RATE,
            robot_root_yaw_rad=0.0,
        )


# ---------------------------------------------------------------------------
# Command parsing
# ---------------------------------------------------------------------------


def test_parse_play_with_name() -> None:
    req = parse_motion_clip_command({"action": "play", "name": "foo"})
    assert isinstance(req, MotionClipPlayRequest)
    assert req.name == "foo"
    assert req.pkl_path is None
    # Default kind is gesture when the wire omits it.
    assert req.kind == "gesture"


def test_parse_play_with_pkl() -> None:
    req = parse_motion_clip_command({"action": "play", "pkl": "/tmp/a.pkl"})
    assert isinstance(req, MotionClipPlayRequest)
    assert req.name is None
    assert req.pkl_path == Path("/tmp/a.pkl")
    assert req.kind == "gesture"


def test_parse_play_kind_locomotion() -> None:
    req = parse_motion_clip_command(
        {"action": "play", "pkl": "/tmp/walk.pkl", "kind": "locomotion"}
    )
    assert isinstance(req, MotionClipPlayRequest)
    assert req.kind == "locomotion"


def test_parse_play_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="'kind' must be one of"):
        parse_motion_clip_command(
            {"action": "play", "name": "foo", "kind": "dance"}
        )


def test_parse_play_carries_overrides() -> None:
    req = parse_motion_clip_command({
        "action": "play",
        "name": "foo",
        "motion_key": "bar",
        "start_frame": 30,
        "n_frames": 100,
    })
    assert isinstance(req, MotionClipPlayRequest)
    assert req.motion_key == "bar"
    assert req.start_frame == 30
    assert req.n_frames == 100
    # No hold_after key on the wire -> None (defer to catalog).
    assert req.hold_after is None


def test_parse_play_carries_hold_after_true_and_false() -> None:
    req_true = parse_motion_clip_command(
        {"action": "play", "name": "foo", "hold_after": True}
    )
    req_false = parse_motion_clip_command(
        {"action": "play", "name": "foo", "hold_after": False}
    )
    assert isinstance(req_true, MotionClipPlayRequest)
    assert isinstance(req_false, MotionClipPlayRequest)
    assert req_true.hold_after is True
    assert req_false.hold_after is False


def test_parse_play_rejects_non_bool_hold_after() -> None:
    with pytest.raises(ValueError, match="hold_after"):
        parse_motion_clip_command(
            {"action": "play", "name": "foo", "hold_after": "yes"}
        )


def test_parse_stop() -> None:
    req = parse_motion_clip_command({"action": "stop"})
    assert isinstance(req, MotionClipStopRequest)


def test_parse_rejects_unknown_action() -> None:
    with pytest.raises(ValueError, match="unknown action"):
        parse_motion_clip_command({"action": "wiggle"})


def test_parse_rejects_play_without_target() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        parse_motion_clip_command({"action": "play"})


def test_parse_rejects_play_with_both_name_and_pkl() -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        parse_motion_clip_command({"action": "play", "name": "a", "pkl": "b.pkl"})


def test_parse_rejects_non_dict_payload() -> None:
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_motion_clip_command(["not", "a", "dict"])  # type: ignore[arg-type]
