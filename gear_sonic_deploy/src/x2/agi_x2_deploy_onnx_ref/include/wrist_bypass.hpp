/**
 * @file wrist_bypass.hpp
 * @brief Surgical override of SONIC's wrist target for the 4 broken DOFs.
 *
 * SONIC's training distribution does not include diverse wrist motion and
 * the smallmotor wrist channels have an x2_action_scale of just 0.0715 (vs
 * ~0.42 elsewhere on the arm), so the policy outputs a near-static comfort
 * pose for ``*_wrist_pitch`` and pins ``*_wrist_roll`` at the asymmetric
 * joint-range tight side regardless of what the IK reference asks for.
 * Empirical proof: ``data/lerobot/x2_quest3_sonic_v2/data/chunk-000/
 * episode_000001.parquet`` shows ``corr(commanded, executed) ~ 0.0`` for
 * both wrist pitches and 98-99% of frames pinned at +-41 deg for both
 * wrist rolls. The same behaviour reproduces on the iter-2k and iter-25k
 * checkpoints, ruling out a training regression.
 *
 * The bypass is CLI-gated (``--wrist-bypass {off,ik}`` on the deploy
 * binary, default off). When ``ik`` is selected, ``OnControl`` calls
 * ``ApplyWristBypass`` BEFORE the safety stack so soft-start blend,
 * ``--max-target-dev`` clamp, and the tilt-trip force-to-default branch
 * still apply uniformly to the IK-driven target.
 *
 * ``wrist_yaw`` (MJ indices 19 and 26) is intentionally LEFT under SONIC:
 * v2 telemetry shows it tracks the IK reference cleanly (correlation
 * ~0.8), unlike pitch / roll. Override only the four broken DOFs.
 *
 * Lives in its own header so the unit test (``test/test_obs_builder.cpp``)
 * can exercise the override loop without pulling in ROS 2 / ONNX runtime.
 */

#ifndef AGI_X2_WRIST_BYPASS_HPP
#define AGI_X2_WRIST_BYPASS_HPP

#include "policy_parameters.hpp"
#include "reference_motion.hpp"

#include <array>
#include <cmath>

namespace agi_x2 {

/// MJ-order joint indices the wrist-only bypass overrides:
///   20 = left_wrist_pitch
///   21 = left_wrist_roll
///   27 = right_wrist_pitch
///   28 = right_wrist_roll
///
/// IMPORTANT: keep in sync with ``policy_parameters.hpp``'s
/// ``mujoco_joint_names`` ordering. The unit test pins these against
/// the live ``default_angles`` slot semantics (both pitches and both
/// rolls have ``default_angles == 0.0``, which is a slot-validity
/// invariant on top of the index numbers themselves).
inline constexpr std::array<int, 4> kBypassedWristMjDofs = {20, 21, 27, 28};

/// MJ-order joint indices the full-arm bypass overrides (used by
/// ``--wrist-bypass=ik-arms``):
///   15..21 = left  shoulder_pitch, shoulder_roll, shoulder_yaw,
///            elbow, wrist_yaw, wrist_pitch, wrist_roll
///   22..28 = right shoulder_pitch, shoulder_roll, shoulder_yaw,
///            elbow, wrist_yaw, wrist_pitch, wrist_roll
///
/// This is the "operator drives both arms straight to the motors while
/// SONIC keeps legs+waist+head" mode. The override semantics are
/// identical to ``kBypassedWristMjDofs`` -- only the set of slots
/// changes -- so the safety stack (soft-start blend, ``--max-target-dev``
/// clamp, tilt-trip force-to-default) and the tokenizer obs flow are
/// unchanged. The leg policy is still conditioned on the IK reference
/// for all 31 DOFs (we lie about nothing in tokenizer obs; only the
/// FINAL per-tick PD target gets the override).
///
/// IMPORTANT: keep in sync with ``policy_parameters.hpp``'s
/// ``mujoco_joint_names`` ordering.
inline constexpr std::array<int, 14> kBypassedArmMjDofs = {
    15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 26, 27, 28,
};

/// Override ``target_pos_mj[mj]`` with ``ref.joint_pos_mj[mj]`` for every
/// joint index in ``bypassed_dofs``. Returns the largest absolute delta
/// between the original target and the IK reference across the
/// overridden slots, so the deploy can surface it on the periodic
/// status line as a "how hard is SONIC being overruled" indicator.
///
/// All other slots of ``target_pos_mj`` are left untouched. The function
/// has no global state; safe to call from the deploy's 50 Hz control
/// loop and from unit tests interchangeably.
template <std::size_t N>
inline double ApplyIkBypass(std::array<double, NUM_DOFS>& target_pos_mj,
                            const ReferenceFrame&         ref,
                            const std::array<int, N>&     bypassed_dofs)
{
  double max_delta = 0.0;
  for (const int mj : bypassed_dofs) {
    const double delta = std::fabs(target_pos_mj[mj] - ref.joint_pos_mj[mj]);
    if (delta > max_delta) max_delta = delta;
    target_pos_mj[mj] = ref.joint_pos_mj[mj];
  }
  return max_delta;
}

/// Wrist-only bypass (4 DOFs: left/right wrist_pitch + wrist_roll).
/// Thin wrapper around ``ApplyIkBypass`` kept for backward compatibility
/// with the unit tests and any out-of-tree callers.
inline double ApplyWristBypass(std::array<double, NUM_DOFS>& target_pos_mj,
                               const ReferenceFrame&         ref)
{
  return ApplyIkBypass(target_pos_mj, ref, kBypassedWristMjDofs);
}

/// Full-arm bypass (14 DOFs: every shoulder/elbow/wrist slot, both sides).
/// Used by ``--wrist-bypass=ik-arms``.
inline double ApplyArmBypass(std::array<double, NUM_DOFS>& target_pos_mj,
                             const ReferenceFrame&         ref)
{
  return ApplyIkBypass(target_pos_mj, ref, kBypassedArmMjDofs);
}

}  // namespace agi_x2

#endif  // AGI_X2_WRIST_BYPASS_HPP
