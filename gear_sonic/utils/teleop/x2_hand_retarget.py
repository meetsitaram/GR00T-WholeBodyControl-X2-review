"""Quest 3 controller -> X2 OmniHand 10-DOF retargeting (controllers-only).

The dataset recorder targets controllers-mode VR for v0 (no bare-hand
XRHand skeleton). The retargeting reduces to a single scalar grasp
ratio per side derived from the controller trigger / grip and a
linear interpolation between OPEN and CLOSED motor anchors.

The OPEN / CLOSED anchors below are vendored verbatim from
``agitbot-x2-record-and-replay``'s ``constants.py`` (branch
``quest3-bare-hand-control``). Joint order is the firmware-canonical
``HAND_FINGER_NAMES_PER_SIDE``::

    1 thumb_roll   2 thumb_abad   3 thumb_mcp
    4 index_abad   5 index_pip
    6 middle_pip
    7 ring_abad    8 ring_pip
    9 pinky_abad  10 pinky_pip

Asymmetric per side because the abad ranges are mirrored on left vs
right.
"""

from __future__ import annotations

import math

import numpy as np


NUM_HAND_DOF_PER_SIDE: int = 10


HAND_FINGER_NAMES_PER_SIDE: tuple[str, ...] = (
    "thumb_roll",
    "thumb_abad",
    "thumb_mcp",
    "index_abad",
    "index_pip",
    "middle_pip",
    "ring_abad",
    "ring_pip",
    "pinky_abad",
    "pinky_pip",
)


def _deg_to_rad(d: float) -> float:
    return d * math.pi / 180.0


def _deg_list_to_rad(lst: list[float]) -> list[float]:
    return [_deg_to_rad(d) for d in lst]


# ── OPEN / CLOSED anchor poses (degrees, then radians) ────────────────────


HAND_GRASP_OPEN_LEFT_DEG: list[float] = [
    0.0,    # thumb_roll  (range -50..+10)
    10.0,   # thumb_abad  (range   0..+100)
    -5.0,   # thumb_mcp   (range -49..   0)
    0.0,    # index_abad  (range   0..+12)
    5.0,    # index_pip   (range   0..+90)
    5.0,    # middle_pip
    0.0,    # ring_abad   (range -10..   0)
    5.0,    # ring_pip
    0.0,    # pinky_abad  (range -10..   0)
    5.0,    # pinky_pip
]

HAND_GRASP_CLOSED_LEFT_DEG: list[float] = [
    -30.0,  # thumb_roll
    60.0,   # thumb_abad
    -40.0,  # thumb_mcp
    6.0,    # index_abad
    80.0,   # index_pip
    80.0,   # middle_pip
    -5.0,   # ring_abad
    80.0,   # ring_pip
    -5.0,   # pinky_abad
    80.0,   # pinky_pip
]

HAND_GRASP_OPEN_RIGHT_DEG: list[float] = [
    0.0,    # thumb_roll  (range -10..+50)
    -10.0,  # thumb_abad  (range -100..  0)
    5.0,    # thumb_mcp   (range   0..+49)
    0.0,    # index_abad  (range -12..   0)
    5.0,    # index_pip
    5.0,    # middle_pip
    0.0,    # ring_abad   (range   0..+10)
    5.0,    # ring_pip
    0.0,    # pinky_abad  (range   0..+10)
    5.0,    # pinky_pip
]

HAND_GRASP_CLOSED_RIGHT_DEG: list[float] = [
    30.0,   # thumb_roll
    -60.0,  # thumb_abad
    40.0,   # thumb_mcp
    -6.0,   # index_abad
    80.0,   # index_pip
    80.0,   # middle_pip
    5.0,    # ring_abad
    80.0,   # ring_pip
    5.0,    # pinky_abad
    80.0,   # pinky_pip
]

HAND_GRASP_OPEN_RAD_LEFT: tuple[float, ...] = tuple(_deg_list_to_rad(HAND_GRASP_OPEN_LEFT_DEG))
HAND_GRASP_CLOSED_RAD_LEFT: tuple[float, ...] = tuple(_deg_list_to_rad(HAND_GRASP_CLOSED_LEFT_DEG))
HAND_GRASP_OPEN_RAD_RIGHT: tuple[float, ...] = tuple(_deg_list_to_rad(HAND_GRASP_OPEN_RIGHT_DEG))
HAND_GRASP_CLOSED_RAD_RIGHT: tuple[float, ...] = tuple(_deg_list_to_rad(HAND_GRASP_CLOSED_RIGHT_DEG))


# (lower, upper) per motor in radians; matches HAND_JOINT_RANGE_*_DEG.
HAND_JOINT_LIMITS_LEFT_RAD: tuple[tuple[float, float], ...] = tuple(
    (_deg_to_rad(lo), _deg_to_rad(hi))
    for lo, hi in [
        (-50.0, 10.0), (0.0, 100.0), (-49.0, 0.0),
        (0.0, 12.0), (0.0, 90.0),
        (0.0, 90.0),
        (-10.0, 0.0), (0.0, 90.0),
        (-10.0, 0.0), (0.0, 90.0),
    ]
)

HAND_JOINT_LIMITS_RIGHT_RAD: tuple[tuple[float, float], ...] = tuple(
    (_deg_to_rad(lo), _deg_to_rad(hi))
    for lo, hi in [
        (-10.0, 50.0), (-100.0, 0.0), (0.0, 49.0),
        (-12.0, 0.0), (0.0, 90.0),
        (0.0, 90.0),
        (0.0, 10.0), (0.0, 90.0),
        (0.0, 10.0), (0.0, 90.0),
    ]
)


# ── Public API ────────────────────────────────────────────────────────────


def grasp_command_from_ratio(side: str, ratio: float) -> np.ndarray:
    """Linearly interpolate between OPEN and CLOSED for a given side.

    Args:
        side: ``"left"`` or ``"right"``.
        ratio: 0.0 (open) .. 1.0 (closed). Clamped to [0, 1].

    Returns:
        ``np.ndarray`` of shape ``(NUM_HAND_DOF_PER_SIDE,)`` -- target
        motor positions in radians, in the canonical motor-axis order
        from :data:`HAND_FINGER_NAMES_PER_SIDE`. Output is clamped to
        the per-motor hardware limits as a final safety belt.
    """
    if side == "left":
        open_q = HAND_GRASP_OPEN_RAD_LEFT
        closed_q = HAND_GRASP_CLOSED_RAD_LEFT
        limits = HAND_JOINT_LIMITS_LEFT_RAD
    elif side == "right":
        open_q = HAND_GRASP_OPEN_RAD_RIGHT
        closed_q = HAND_GRASP_CLOSED_RAD_RIGHT
        limits = HAND_JOINT_LIMITS_RIGHT_RAD
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    r = float(np.clip(ratio, 0.0, 1.0))
    open_arr = np.asarray(open_q, dtype=np.float64)
    closed_arr = np.asarray(closed_q, dtype=np.float64)
    cmd = (1.0 - r) * open_arr + r * closed_arr
    lo = np.array([a for a, _ in limits], dtype=np.float64)
    hi = np.array([b for _, b in limits], dtype=np.float64)
    return np.clip(cmd, lo, hi)


def controller_grasp_ratio(
    left_trigger: float,
    right_trigger: float,
    left_grip: float,
    right_grip: float,
    *,
    mode: str = "trigger",
) -> tuple[float, float]:
    """Pick (left_ratio, right_ratio) in [0, 1] from controller analog inputs.

    Args:
        left_trigger / right_trigger / left_grip / right_grip: 0..1
            analog values from the Quest 3 controllers (already
            clamped on the WebXR side).
        mode: ``"trigger"`` (index trigger; default), ``"grip"``
            (middle-finger grip), or ``"max"`` (whichever is greater
            this frame).
    """
    def _pick(t: float, g: float) -> float:
        if mode == "trigger":
            v = t
        elif mode == "grip":
            v = g
        elif mode == "max":
            v = max(t, g)
        else:
            raise ValueError(f"mode must be trigger|grip|max, got {mode!r}")
        return float(np.clip(v, 0.0, 1.0))

    return _pick(left_trigger, left_grip), _pick(right_trigger, right_grip)


__all__ = [
    "HAND_FINGER_NAMES_PER_SIDE",
    "HAND_GRASP_CLOSED_RAD_LEFT",
    "HAND_GRASP_CLOSED_RAD_RIGHT",
    "HAND_GRASP_OPEN_RAD_LEFT",
    "HAND_GRASP_OPEN_RAD_RIGHT",
    "HAND_JOINT_LIMITS_LEFT_RAD",
    "HAND_JOINT_LIMITS_RIGHT_RAD",
    "NUM_HAND_DOF_PER_SIDE",
    "controller_grasp_ratio",
    "grasp_command_from_ratio",
]
