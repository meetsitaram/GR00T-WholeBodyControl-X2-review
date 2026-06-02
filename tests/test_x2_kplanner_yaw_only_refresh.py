"""Unit tests for ``x2_kplanner._yaw_only_wxyz_from_pelvis``.

The helper is the workhorse of the IDLE_LOOP yaw resync added to
``x2_kplanner.py`` to stop SONIC from twisting the body back to a
stale published yaw after fall recovery or sit/stand gesture
sequences. We test the function in isolation (pure numpy, no torch /
no neural planner) because the publish-loop wrapper that calls it is
threaded with a 50 Hz cadence and not easily unit-testable.

Invariants pinned:

1. Identity-in -> identity-out.
2. Yaw-only quat-in -> bit-for-bit-same yaw-only quat-out.
3. Pitch / roll components in the input are DROPPED -- output is a
   pure R_z(yaw) wxyz quat. (A fall-pitched pelvis must not bleed
   pitch into the published reference; SONIC's training distribution
   assumes upright references.)
4. Negative yaw preserves sign and stays in (-pi, pi].
5. Short arrays raise :class:`ValueError` (defensive guard against a
   half-decoded ``robot_pose`` payload making it into the publish
   path).

Symmetry note: yaw extraction follows the same ZYX Euler convention
as :func:`gear_sonic.utils.planner.blending.yaw_of_quat_xyzw` (the
helper used everywhere else in the stack); we cross-check against it
on the round-trip case.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.scripts.x2_kplanner import (  # noqa: E402
    _yaw_only_wxyz_from_pelvis,
)
from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw  # noqa: E402


def _make_pelvis(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Wrap a wxyz quat into the bridge wire format (xyz + qwxyz)."""
    return np.array([0.0, 0.0, 0.793, qw, qx, qy, qz], dtype=np.float32)


def _wxyz_to_xyzw(q_wxyz: np.ndarray) -> np.ndarray:
    return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]],
                    dtype=np.float64)


def test_identity_in_gives_identity_out() -> None:
    """A robot at world identity yaw + level pelvis must produce identity."""
    pelvis = _make_pelvis(1.0, 0.0, 0.0, 0.0)
    out = _yaw_only_wxyz_from_pelvis(pelvis)
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0, 0.0], atol=1e-7)


def test_pure_z_yaw_roundtrips_bit_for_bit_within_float_tolerance() -> None:
    """Input R_z(yaw) -> output R_z(yaw) for the full yaw range."""
    for yaw_rad in (-3.0, -1.5, -0.7, 0.0, 0.3, 1.2, 2.4, 3.0):
        half = 0.5 * yaw_rad
        pelvis = _make_pelvis(math.cos(half), 0.0, 0.0, math.sin(half))
        out = _yaw_only_wxyz_from_pelvis(pelvis)
        # Output must be unit norm.
        assert float(np.linalg.norm(out)) == pytest.approx(1.0, abs=1e-6)
        # Extracted yaw from output must match input yaw.
        out_yaw = yaw_of_quat_xyzw(_wxyz_to_xyzw(out))
        assert out_yaw == pytest.approx(yaw_rad, abs=1e-6)


def test_pitch_in_input_is_dropped_from_output() -> None:
    """A pelvis with pitch but no yaw -> identity output (yaw=0)."""
    # 25-degree pitch about world Y, no yaw.
    r = Rot.from_euler("y", 25.0, degrees=True)
    q_xyzw = r.as_quat()  # xyzw
    pelvis = _make_pelvis(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
    out = _yaw_only_wxyz_from_pelvis(pelvis)
    out_yaw = yaw_of_quat_xyzw(_wxyz_to_xyzw(out))
    assert out_yaw == pytest.approx(0.0, abs=1e-6)
    # And the output must be a pure yaw quat (qx == qy == 0).
    np.testing.assert_allclose(out[1:3], [0.0, 0.0], atol=1e-7)


def test_roll_in_input_is_dropped_from_output() -> None:
    """A pelvis with roll but no yaw -> identity output (yaw=0)."""
    r = Rot.from_euler("x", -18.0, degrees=True)
    q_xyzw = r.as_quat()
    pelvis = _make_pelvis(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
    out = _yaw_only_wxyz_from_pelvis(pelvis)
    out_yaw = yaw_of_quat_xyzw(_wxyz_to_xyzw(out))
    assert out_yaw == pytest.approx(0.0, abs=1e-6)
    np.testing.assert_allclose(out[1:3], [0.0, 0.0], atol=1e-7)


def test_yaw_extracted_from_combined_zyx_input() -> None:
    """Combined yaw+pitch+roll: only the yaw component survives."""
    yaw_deg = 35.0
    r = Rot.from_euler("zyx", [yaw_deg, 15.0, -8.0], degrees=True)
    q_xyzw = r.as_quat()
    pelvis = _make_pelvis(q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2])
    out = _yaw_only_wxyz_from_pelvis(pelvis)
    out_yaw = yaw_of_quat_xyzw(_wxyz_to_xyzw(out))
    assert out_yaw == pytest.approx(math.radians(yaw_deg), abs=1e-5)
    # Pure yaw output -> qx/qy must be zero.
    np.testing.assert_allclose(out[1:3], [0.0, 0.0], atol=1e-7)


def test_negative_yaw_preserves_sign() -> None:
    """A -90 degree pelvis must produce a -90 degree output (not +270)."""
    yaw_rad = -math.pi / 2
    half = 0.5 * yaw_rad
    pelvis = _make_pelvis(math.cos(half), 0.0, 0.0, math.sin(half))
    out = _yaw_only_wxyz_from_pelvis(pelvis)
    out_yaw = yaw_of_quat_xyzw(_wxyz_to_xyzw(out))
    assert out_yaw == pytest.approx(yaw_rad, abs=1e-6)


def test_output_dtype_is_float32() -> None:
    """``current_root_wxyz`` is typed float32 in the publish loop; the
    helper must match so the assignment doesn't silently upcast."""
    pelvis = _make_pelvis(1.0, 0.0, 0.0, 0.0)
    out = _yaw_only_wxyz_from_pelvis(pelvis)
    assert out.dtype == np.float32
    assert out.shape == (4,)


def test_short_array_raises_value_error() -> None:
    """Defensive: a half-decoded robot_pose payload (length<7) must
    raise rather than silently misinterpret bytes as yaw."""
    short = np.array([0.0, 0.0, 0.793, 1.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="pelvis_qpos_wxyz must be >= 7"):
        _yaw_only_wxyz_from_pelvis(short)


def test_accepts_extra_dimensions_via_reshape() -> None:
    """The helper reshapes input to 1-D, so a (1, 7) row from a
    PoseObservation buffer should work without an explicit squeeze."""
    pelvis_2d = np.array(
        [[0.0, 0.0, 0.793, math.cos(0.5), 0.0, 0.0, math.sin(0.5)]],
        dtype=np.float32,
    )
    out = _yaw_only_wxyz_from_pelvis(pelvis_2d)
    out_yaw = yaw_of_quat_xyzw(_wxyz_to_xyzw(out))
    assert out_yaw == pytest.approx(1.0, abs=1e-6)
