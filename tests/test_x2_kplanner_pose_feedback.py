"""Unit tests for the x2_kplanner closed-loop pose-reseed hook.

These tests pin the contract of :func:`_reseed_root_from_observations`
without spinning up the full neural planner: a tiny fake
``planner_core`` exposes only the four attributes the hook reads
(``frames["mujoco_qpos"]``, ``_current_frame_idx``,
``NUM_FRAMES_PER_TOKEN``, ``PRED_OFFSETS``) and we drive a real
``collections.deque`` of :class:`PoseObservation` against it.

Invariants covered:

1. **Root-only overwrite** -- only slots ``[0:7]`` of the targeted
   context frames are mutated; joints ``[7:]`` stay byte-identical.
2. **Fills all NUM_FRAMES_PER_TOKEN frames** -- on success the 4
   context indices match
   ``NeuralPlannerCore.get_context_mujoco_qpos`` (mirrors the index
   math 1:1).
3. **Stale-obs fallback** -- when the newest observation is older than
   ``max_age_s`` the hook returns ``"stale_obs ..."`` and leaves the
   buffer untouched.
4. **Insufficient-obs fallback** -- when fewer than
   ``NUM_FRAMES_PER_TOKEN`` observations are queued the hook returns
   ``"insufficient_obs ..."`` and leaves the buffer untouched.
5. **wxyz slot ordering** -- ``pelvis_qpos_wxyz =
   [x, y, z, qw, qx, qy, qz]`` lands at slots ``[0..6]`` in the
   correct order (the planner stores quat as wxyz, same as the
   ``robot_pose`` wire format).
"""

from __future__ import annotations

import collections
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gear_sonic.scripts.x2_kplanner as kp  # noqa: E402
from gear_sonic.scripts.x2_kplanner import (  # noqa: E402
    PoseObservation,
    _reseed_root_from_observations,
)


# Match the production constants pulled from
# motionbricks.motion_backbone.inference.neural_planner.NeuralPlannerCore.
# These are pinned defaults that are hardcoded in the checkpoints; if
# they ever change upstream this fixture and the hook must change
# together.
_NUM_FT = 4
_PRED_OFFSETS = 4
_QPOS_DIM = 38


class _FakePlannerCore:
    """Minimal stand-in for NeuralPlannerCore.

    Mirrors the four attributes the reseed hook reads. Lets us run
    the hook without loading any checkpoints, importing the model, or
    touching cuda.
    """

    NUM_FRAMES_PER_TOKEN = _NUM_FT
    PRED_OFFSETS = _PRED_OFFSETS

    def __init__(
        self,
        buf: torch.Tensor | None,
        current_frame_idx: int = 0,
    ) -> None:
        self.frames = {"mujoco_qpos": buf, "model_features": None}
        self._current_frame_idx = current_frame_idx


def _make_buf(num_frames: int = 64, marker_value: float = 99.0) -> torch.Tensor:
    """Build a [1, T, 38] qpos buffer filled with a sentinel value.

    Joints (slots 7:) are seeded with ``marker_value`` so the test can
    assert they survive the reseed byte-identical (any overwrite would
    smear ``marker_value`` -> reseed value, easy to detect).
    """
    buf = torch.full((1, num_frames, _QPOS_DIM), fill_value=marker_value, dtype=torch.float32)
    # Root slots [0:7] -- seed each frame with its index so we can
    # detect "wrote to right frame" vs "wrote to wrong frame".
    for i in range(num_frames):
        buf[0, i, 0:7] = torch.tensor(
            [-1.0 - i, -2.0 - i, -3.0 - i, -1.0, 0.0, 0.0, 0.0],
            dtype=torch.float32,
        )
    return buf


def _make_obs(t_mono: float, x: float, y: float, z: float,
              qw: float = 1.0, qx: float = 0.0,
              qy: float = 0.0, qz: float = 0.0) -> PoseObservation:
    return PoseObservation(
        t_mono=float(t_mono),
        pelvis_qpos_wxyz=np.array([x, y, z, qw, qx, qy, qz], dtype=np.float32),
    )


def _expected_context_indices(num_frames: int, current_frame_idx: int) -> list[int]:
    """Mirror NeuralPlannerCore.get_context_mujoco_qpos exactly.

    We reimplement it here (rather than importing) so this test fails
    if the production hook ever drifts away from the upstream math.
    """
    last_idx = num_frames - 1
    return [
        max(0, min(current_frame_idx - _NUM_FT + i + _PRED_OFFSETS, last_idx))
        for i in range(_NUM_FT)
    ]


# ---------------------------------------------------------------------------
# 1. root-only overwrite
# ---------------------------------------------------------------------------


def test_reseed_overwrites_root_only_not_joints():
    """Slots [0:7] change; slots [7:] stay byte-identical."""
    buf = _make_buf(num_frames=32, marker_value=42.0)
    joints_before = buf[:, :, 7:].clone()

    core = _FakePlannerCore(buf, current_frame_idx=10)

    deque = collections.deque(maxlen=8)
    lock = threading.Lock()
    now = time.monotonic()
    for k in range(_NUM_FT):
        deque.append(_make_obs(now - 0.01 * (_NUM_FT - 1 - k),
                               x=1.0 + k, y=2.0 + k, z=3.0 + k))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
    )

    assert reason is None, f"expected success, got skip reason: {reason}"
    joints_after = buf[:, :, 7:]
    assert torch.equal(joints_before, joints_after), (
        "joint slots [7:] must NOT be touched by the root reseed; "
        "the policy injects high-frequency noise that would poison "
        "the planner's context if joints were reseeded too."
    )


# ---------------------------------------------------------------------------
# 2. fills 4 context indices (mirrors get_context_mujoco_qpos)
# ---------------------------------------------------------------------------


def test_reseed_fills_all_four_context_indices():
    """Each of the 4 indices get_context_mujoco_qpos will read gets written.

    The reseed samples observations at the model's training fps spacing
    (1/30 s = 33.3 ms apart), walking backwards from "now". So we
    populate the deque with observations at exactly that spacing and
    let the nearest-neighbor pick land on the expected k-th sample.
    Use num_frames+current_frame_idx values that yield 4 distinct
    context indices (so per-slot writes don't collapse onto each other).
    """
    model_fps = 30.0
    spacing_s = 1.0 / model_fps

    num_frames = 40
    # Pick current_idx so context_indices = [idx, idx+1, idx+2, idx+3]
    # are all in range and distinct. With NUM_FT=4 and PRED_OFFSETS=4
    # the formula collapses to idx+i, so any idx <= last_idx-3 works.
    current_idx = 12
    buf = _make_buf(num_frames=num_frames, marker_value=0.0)
    core = _FakePlannerCore(buf, current_frame_idx=current_idx)

    expected_indices = _expected_context_indices(num_frames, current_idx)
    assert len(set(expected_indices)) == _NUM_FT, (
        "test setup error: pick current_idx so the 4 context indices "
        "are distinct, otherwise the duplicate-collapse path runs and "
        "this test isn't measuring per-slot ordering"
    )

    deque = collections.deque(maxlen=16)
    lock = threading.Lock()
    now = time.monotonic()
    # Observations exactly at the model's training spacing. The k-th
    # context slot maps to "now - (NUM_FT-1-k) * spacing" so we layer
    # the values to match: obs at "now - (NUM_FT-1-k)*spacing" carries
    # value k. Note: deque order is append-order, oldest first.
    for k in range(_NUM_FT):
        deque.append(_make_obs(
            now - (_NUM_FT - 1 - k) * spacing_s,
            x=100.0 + k, y=200.0 + k, z=300.0 + k,
        ))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
    )

    assert reason is None
    for k, idx in enumerate(expected_indices):
        actual_xyz = buf[0, idx, 0:3].tolist()
        assert actual_xyz == [
            pytest.approx(100.0 + k),
            pytest.approx(200.0 + k),
            pytest.approx(300.0 + k),
        ], (
            f"context slot k={k} (buffer idx={idx}) should hold "
            f"the obs nearest to now-{(_NUM_FT-1-k)*spacing_s*1000:.1f}ms "
            f"(carries x={100.0 + k}); got {actual_xyz}"
        )


def test_reseed_collapses_when_deque_spacing_finer_than_model_fps():
    """Bridge publishes at 50 Hz; model wants 30 Hz spacing.

    When the deque holds many recent obs all within one model-frame
    duration of "now", multiple context slots can legitimately resolve
    to the *same* observation. Pin that behaviour so a future
    "deque must monotonically populate distinct slots" assertion
    doesn't sneak in and break the production path.
    """
    num_frames = 32
    buf = _make_buf(num_frames=num_frames)
    core = _FakePlannerCore(buf, current_frame_idx=10)

    deque = collections.deque(maxlen=16)
    lock = threading.Lock()
    now = time.monotonic()
    # 4 obs all within 60 ms of now (bridge at 50 Hz over the most
    # recent 80 ms). The model's training spacing is 33 ms; targets
    # are now, -33, -67, -100 ms. The nearest-neighbor picks will
    # collapse to "the newest" for the recent targets and "the oldest"
    # for the -100 ms target -- which IS the right semantic when only
    # 60 ms of history is on hand.
    for k in range(_NUM_FT):
        deque.append(_make_obs(now - 0.020 * (_NUM_FT - 1 - k),
                               x=1.0, y=2.0, z=3.0))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
    )
    assert reason is None, f"unexpected skip: {reason}"

    # Every targeted context index must hold one of the obs we
    # published (they're all identical here, so just verify x=1.0).
    expected_indices = _expected_context_indices(num_frames, 10)
    for idx in expected_indices:
        assert buf[0, idx, 0].item() == pytest.approx(1.0), (
            f"context slot at buf[{idx}] should be reseeded "
            f"(even if multiple slots collapse onto the same obs)"
        )


# ---------------------------------------------------------------------------
# 3. stale-obs fallback
# ---------------------------------------------------------------------------


def test_reseed_skips_when_observations_are_stale():
    """Newest obs older than max_age_s -> skip, no mutation."""
    buf = _make_buf(num_frames=32)
    buf_snapshot = buf.clone()
    core = _FakePlannerCore(buf, current_frame_idx=10)

    deque = collections.deque(maxlen=8)
    lock = threading.Lock()
    # All observations are 5 seconds old. With max_age_s=0.5 (production
    # default) the newest is stale and we must skip.
    stale_t = time.monotonic() - 5.0
    for k in range(_NUM_FT):
        deque.append(_make_obs(stale_t + 0.001 * k,
                               x=999.0, y=888.0, z=777.0))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=0.5,
    )

    assert reason is not None and reason.startswith("stale_obs"), (
        f"expected 'stale_obs ...' skip, got: {reason}"
    )
    assert torch.equal(buf, buf_snapshot), (
        "stale-obs skip must not mutate the buffer (the planner falls "
        "back to its own predictions)"
    )


# ---------------------------------------------------------------------------
# 4. insufficient-obs fallback (boot window)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_obs", [0, 1, 2, 3])
def test_reseed_skips_when_fewer_than_num_ft_observations(n_obs: int):
    """Boot window: not enough observations -> skip, no mutation."""
    buf = _make_buf(num_frames=32)
    buf_snapshot = buf.clone()
    core = _FakePlannerCore(buf, current_frame_idx=10)

    deque = collections.deque(maxlen=8)
    lock = threading.Lock()
    now = time.monotonic()
    for k in range(n_obs):
        deque.append(_make_obs(now - 0.001 * (n_obs - 1 - k),
                               x=1.0, y=2.0, z=3.0))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
    )

    assert reason is not None and reason.startswith("insufficient_obs"), (
        f"expected 'insufficient_obs ...' skip with n_obs={n_obs}, got: {reason}"
    )
    assert torch.equal(buf, buf_snapshot), (
        f"insufficient-obs skip must not mutate the buffer "
        f"(n_obs={n_obs} < NUM_FRAMES_PER_TOKEN={_NUM_FT})"
    )


# ---------------------------------------------------------------------------
# 5. wxyz slot ordering
# ---------------------------------------------------------------------------


def test_reseed_writes_wxyz_quat_in_correct_slot_order():
    """pelvis_qpos_wxyz [x,y,z,qw,qx,qy,qz] -> buf slots [0..6]."""
    buf = _make_buf(num_frames=32)
    core = _FakePlannerCore(buf, current_frame_idx=10)

    deque = collections.deque(maxlen=8)
    lock = threading.Lock()
    now = time.monotonic()

    # A distinctive non-identity quat so any axis swap would surface.
    # Numbers chosen to be unmistakable in tensor diffs.
    test_pose = (0.111, 0.222, 0.333, 0.444, 0.555, 0.666, 0.777)
    for k in range(_NUM_FT):
        # Same pose for every frame -- we're testing slot ordering,
        # not per-frame variation.
        deque.append(_make_obs(
            now - 0.001 * (_NUM_FT - 1 - k),
            *test_pose,
        ))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
    )
    assert reason is None

    expected_indices = _expected_context_indices(buf.shape[1], 10)
    for idx in expected_indices:
        slot = buf[0, idx, 0:7].tolist()
        assert slot == [
            pytest.approx(0.111),  # x
            pytest.approx(0.222),  # y
            pytest.approx(0.333),  # z
            pytest.approx(0.444),  # qw
            pytest.approx(0.555),  # qx
            pytest.approx(0.666),  # qy
            pytest.approx(0.777),  # qz
        ], (
            f"buf[0, {idx}, 0:7] should be [x,y,z,qw,qx,qy,qz] in "
            f"order, got {slot}. The robot_pose wire format is wxyz "
            f"and NeuralPlannerCore stores wxyz; any swap here means "
            f"the planner sees the wrong root orientation as context."
        )


# ---------------------------------------------------------------------------
# Bonus: buffer-uninitialized guard. Cheap to add and pins the
# pre-reset() boot path. Skipped from the "5 unit tests" headline count
# in the plan because it's a degenerate edge case.
# ---------------------------------------------------------------------------


def test_reseed_quat_only_scope_leaves_xyz_unchanged():
    """``scope='quat_only'`` rewrites slots [3:7], leaves [0:3] alone.

    Motivated by the validation run that showed ``full_root`` reseed
    regresses forward tracking (the planner's xy overshoot was helping
    the deploy track). ``quat_only`` preserves that overshoot while
    still anchoring heading.
    """
    buf = _make_buf(num_frames=32)
    xyz_before = buf[:, :, 0:3].clone()

    core = _FakePlannerCore(buf, current_frame_idx=10)
    deque = collections.deque(maxlen=8)
    lock = threading.Lock()
    now = time.monotonic()
    for k in range(_NUM_FT):
        deque.append(_make_obs(
            now - (1.0 / 30.0) * (_NUM_FT - 1 - k),
            x=99.0, y=98.0, z=97.0,  # values that would be obvious if written
            qw=0.7071, qx=0.0, qy=0.0, qz=0.7071,  # distinctive 90deg yaw quat
        ))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
        scope="quat_only",
    )

    assert reason is None
    # xyz channels must be byte-identical to before -- this is the
    # whole point of quat_only mode.
    assert torch.equal(buf[:, :, 0:3], xyz_before), (
        "scope='quat_only' must NOT touch xyz slots [0:3]; the open-loop "
        "diagnostic showed those overshoot values help forward tracking"
    )
    # Quat channels of the targeted context indices must hold the
    # observed quat.
    expected_indices = _expected_context_indices(buf.shape[1], 10)
    for idx in expected_indices:
        quat = buf[0, idx, 3:7].tolist()
        assert quat == [
            pytest.approx(0.7071),
            pytest.approx(0.0),
            pytest.approx(0.0),
            pytest.approx(0.7071),
        ], f"buf[0, {idx}, 3:7] should hold the observed quat, got {quat}"


def test_reseed_invalid_scope_raises():
    """Bad scope strings fail loudly at the boundary, not silently."""
    buf = _make_buf(num_frames=32)
    core = _FakePlannerCore(buf, current_frame_idx=10)
    deque = collections.deque(maxlen=8)
    lock = threading.Lock()
    now = time.monotonic()
    for k in range(_NUM_FT):
        deque.append(_make_obs(now, 0.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="unknown reseed scope"):
        _reseed_root_from_observations(
            core, deque, lock, max_age_s=1.0,
            scope="banana",
        )


def test_reseed_returns_buffer_uninitialized_when_frames_none():
    core = _FakePlannerCore(buf=None, current_frame_idx=0)
    deque = collections.deque(maxlen=8)
    lock = threading.Lock()
    now = time.monotonic()
    for k in range(_NUM_FT):
        deque.append(_make_obs(now, 0.0, 0.0, 0.0))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
    )

    assert reason == "buffer_uninitialized", (
        f"pre-reset() boot path should return the literal sentinel "
        f"'buffer_uninitialized'; got: {reason!r}"
    )


# ---------------------------------------------------------------------------
# scope='none' short-circuit
# ---------------------------------------------------------------------------


def test_reseed_scope_none_short_circuits_without_touching_buffer():
    """scope='none' must return 'disabled' immediately and leave the
    planner's neural buffer byte-identical.

    This is the real-robot path: the x2_debug bridge has no position
    sensor (xy=z=0), so the launcher pins scope=none and lets the
    yaw-only refreshes (IDLE_LOOP, IDLE->PLAYING, startup) carry the
    snap-back protection alone. The PLAYING-side reseed must NOT mutate
    the model's context buffer in that mode -- if it ever did, the model
    would see (0,0,0) position history every tick and walk/turn would
    regress.
    """
    buf = _make_buf(num_frames=32, marker_value=7.0)
    buf_before = buf.clone()

    core = _FakePlannerCore(buf, current_frame_idx=10)
    deque = collections.deque(maxlen=8)
    lock = threading.Lock()
    now = time.monotonic()
    # Stack the deque with fresh, valid observations -- even with
    # plenty of data the short-circuit must fire BEFORE the lock is
    # taken (no insufficient/stale fallback path leaking through).
    for k in range(_NUM_FT * 2):
        deque.append(_make_obs(now - 0.005 * k, x=1.0, y=2.0, z=0.85))

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
        scope="none",
    )

    assert reason == "disabled", (
        f"scope='none' must return the literal 'disabled' sentinel so "
        f"reseed_stats['skipped_disabled'] accounting works; got: {reason!r}"
    )
    assert torch.equal(buf, buf_before), (
        "scope='none' must leave the planner buffer untouched, byte-for-byte"
    )


def test_reseed_scope_none_short_circuits_even_without_observations():
    """The short-circuit must fire BEFORE the pose_deque length check.

    Confirms scope='none' returns 'disabled' (not 'insufficient_obs')
    even when the deque is empty or smaller than NUM_FT. Important on
    boot when pose_deque hasn't filled yet -- we want predictable
    'disabled' accounting from tick 1, not a transient 'insufficient'.
    """
    buf = _make_buf(num_frames=32, marker_value=11.0)
    buf_before = buf.clone()
    core = _FakePlannerCore(buf, current_frame_idx=5)
    deque = collections.deque(maxlen=8)
    lock = threading.Lock()

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
        scope="none",
    )

    assert reason == "disabled"
    assert torch.equal(buf, buf_before)


def test_reseed_scope_none_short_circuits_with_uninitialized_buffer():
    """The short-circuit must fire BEFORE the frames-None check too.

    Same rationale: predictable accounting from boot, even when the
    planner_core hasn't called reset() yet (frames['mujoco_qpos']=None).
    """
    core = _FakePlannerCore(buf=None, current_frame_idx=0)
    deque = collections.deque(maxlen=8)
    lock = threading.Lock()

    reason = _reseed_root_from_observations(
        core, deque, lock, max_age_s=1.0,
        scope="none",
    )

    assert reason == "disabled"


# ---------------------------------------------------------------------------
# Smoke test the PoseObservation dataclass shape contract. Keeps the
# wire-format spec discoverable from the tests file even if the
# dataclass moves.
# ---------------------------------------------------------------------------


def test_pose_observation_carries_length_7_wxyz_array():
    obs = PoseObservation(
        t_mono=1.23,
        pelvis_qpos_wxyz=np.array([0, 1, 2, 1, 0, 0, 0], dtype=np.float32),
    )
    assert obs.t_mono == pytest.approx(1.23)
    assert obs.pelvis_qpos_wxyz.shape == (7,)
    # wxyz convention: identity quat is [1, 0, 0, 0] at slots [3..6].
    assert obs.pelvis_qpos_wxyz[3] == pytest.approx(1.0)
