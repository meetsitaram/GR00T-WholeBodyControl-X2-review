"""Surgical wrist-/arm-target override for X2 SONIC deploy.

This is a Python port of
``gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp``;
keep the two implementations in lock-step.

Two override sets:

- :data:`BYPASSED_WRIST_MJ_DOFS` (4 DOFs, default for
  :func:`apply_wrist_bypass`): MJ {20, 21, 27, 28} = left/right
  wrist_pitch + wrist_roll. Used by ``--wrist-bypass=ik`` on the deploy.
- :data:`BYPASSED_ARM_MJ_DOFS` (14 DOFs): MJ 15..28 = both arms in full
  (shoulders + elbows + wrist_yaw + wrist_pitch + wrist_roll). Used by
  ``--wrist-bypass=ik-arms`` -- VR IK drives both arms direct to motors
  while SONIC keeps controlling legs+waist+head for balance. Call
  :func:`apply_arm_bypass` (or pass ``bypassed_dofs=BYPASSED_ARM_MJ_DOFS``
  to :func:`apply_wrist_bypass`).

Why the bypass exists
=====================

SONIC's training distribution does not include diverse wrist motion and the
small-motor wrist channels carry an ``x2_action_scale`` of just 0.0715 (vs
~0.42 elsewhere on the arm), so the policy outputs a near-static comfort
pose for ``*_wrist_pitch`` and pins ``*_wrist_roll`` at the asymmetric
joint-range tight side regardless of what the IK reference asks for.

Empirical proof: ``data/lerobot/x2_quest3_sonic_v2/data/chunk-000/
episode_000001.parquet`` shows ``corr(commanded, executed) ~ 0.0`` for both
wrist pitches and 98-99 % of frames pinned at +-41 deg for both wrist
rolls. The same behaviour reproduces on the iter-2k and iter-25k
checkpoints, ruling out a training regression.

The bypass replaces the SONIC wrist target with the IK reference for those
four DOFs *before* downstream safety stages (soft-start blend,
``--max-target-dev`` clamp, tilt-trip) run, so they apply uniformly to the
override too.

``wrist_yaw`` is intentionally LEFT under SONIC: telemetry shows it
tracks the IK reference cleanly (correlation ~0.8), unlike pitch / roll.

The MJ-order joint table for the X2 lives in
``gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp``;
the pitch/roll indices below are pinned against that table.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# MJ-order joint indices the wrist-only bypass overrides:
#   20 = left_wrist_pitch_joint
#   21 = left_wrist_roll_joint
#   27 = right_wrist_pitch_joint
#   28 = right_wrist_roll_joint
#
# IMPORTANT: keep in sync with ``policy_parameters.hpp``'s
# ``mujoco_joint_names`` ordering.
BYPASSED_WRIST_MJ_DOFS: Tuple[int, ...] = (20, 21, 27, 28)

# MJ-order joint indices the full-arm bypass overrides (matches
# ``kBypassedArmMjDofs`` in ``wrist_bypass.hpp``, used by the C++
# deploy binary when ``--wrist-bypass=ik-arms``):
#   15..21 = left  shoulder_pitch, shoulder_roll, shoulder_yaw,
#            elbow, wrist_yaw, wrist_pitch, wrist_roll
#   22..28 = right shoulder_pitch, shoulder_roll, shoulder_yaw,
#            elbow, wrist_yaw, wrist_pitch, wrist_roll
#
# This is the "operator drives both arms straight to the motors while
# SONIC keeps legs+waist+head" mode. See
# ``docs/source/user_guide/milestones/2026-06-08_arm_bypass_v1.md`` for
# the design notes + stability caveats.
BYPASSED_ARM_MJ_DOFS: Tuple[int, ...] = (
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
)

# X2 has 31 DOF in the SONIC MJ-order vector.
NUM_DOFS: int = 31


def apply_wrist_bypass(
    target_pos_mj: np.ndarray,
    ik_ref_pos_mj: Sequence[float] | np.ndarray,
    bypassed_dofs: Sequence[int] = BYPASSED_WRIST_MJ_DOFS,
) -> float:
    """Override the SONIC wrist target with the IK reference, in-place.

    Parameters
    ----------
    target_pos_mj
        Length-``NUM_DOFS`` MJ-ordered target vector (e.g. SONIC's policy
        output mapped through ``isaaclab_to_mujoco``). Modified in-place.
    ik_ref_pos_mj
        Length-``NUM_DOFS`` MJ-ordered IK reference vector for the same
        joints.
    bypassed_dofs
        MJ-order joint indices whose target should be overridden. Defaults
        to the four broken wrist DOFs identified by the deploy stack;
        passing a different set lets unit tests parametrise it.

    Returns
    -------
    float
        Largest absolute delta between the original target and the IK
        reference across the overridden slots. Useful as a periodic
        "how hard is SONIC being overruled" indicator.

    Notes
    -----
    * This function has no global state; safe to call from a 50 Hz
      control loop and from unit tests interchangeably.
    * Slots outside ``bypassed_dofs`` are left untouched.
    """
    target_arr = np.asarray(target_pos_mj)
    ref_arr = np.asarray(ik_ref_pos_mj)

    if target_arr.shape != (NUM_DOFS,):
        raise ValueError(
            f"target_pos_mj must be 1-D length {NUM_DOFS}, got shape {target_arr.shape}"
        )
    if ref_arr.shape != (NUM_DOFS,):
        raise ValueError(
            f"ik_ref_pos_mj must be 1-D length {NUM_DOFS}, got shape {ref_arr.shape}"
        )

    max_delta = 0.0
    for mj in bypassed_dofs:
        delta = float(abs(target_arr[mj] - ref_arr[mj]))
        if delta > max_delta:
            max_delta = delta
        target_arr[mj] = ref_arr[mj]
    return max_delta


def wrist_bypass_max_delta(
    target_pos_mj: np.ndarray,
    ik_ref_pos_mj: Sequence[float] | np.ndarray,
    bypassed_dofs: Sequence[int] = BYPASSED_WRIST_MJ_DOFS,
) -> float:
    """Pure-read variant: compute the max delta WITHOUT mutating anything.

    Useful in dry-run / monitoring code paths where the bypass is gated
    off but you still want to log how far SONIC's wrist target has
    drifted from the IK reference.
    """
    target_arr = np.asarray(target_pos_mj)
    ref_arr = np.asarray(ik_ref_pos_mj)
    deltas = np.abs(target_arr[list(bypassed_dofs)] - ref_arr[list(bypassed_dofs)])
    return float(deltas.max()) if deltas.size > 0 else 0.0


def apply_arm_bypass(
    target_pos_mj: np.ndarray,
    ik_ref_pos_mj: Sequence[float] | np.ndarray,
) -> float:
    """Full-arm bypass alias: override the 14 arm DOFs (MJ 15..28).

    Mirrors ``ApplyArmBypass`` in ``wrist_bypass.hpp``. Thin wrapper
    around :func:`apply_wrist_bypass` with ``BYPASSED_ARM_MJ_DOFS``.
    """
    return apply_wrist_bypass(
        target_pos_mj, ik_ref_pos_mj, bypassed_dofs=BYPASSED_ARM_MJ_DOFS
    )
