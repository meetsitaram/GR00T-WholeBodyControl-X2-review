/**
 * @file safety.hpp
 * @brief Safety scaffolding for first-real-robot bring-up.
 *
 * Three layers, ordered most-passive to least-passive:
 *
 *   1. DryRun
 *      - Runs the full obs/policy/IO pipeline and PUBLISHES the commands,
 *        but with stiffness=0 and damping=0 so the motors do nothing. Used
 *        to verify wiring (topics, joint name remap, IMU sign convention)
 *        before any torque is applied.
 *
 *   2. SoftStartRamp
 *      - Once the operator says "go", blend the live policy target with the
 *        DEFAULT_DOF standing pose over `ramp_seconds` seconds. Prevents
 *        a sudden jump from the operator-selected start pose to whatever
 *        the policy thinks is "right" on the first inference.
 *
 *   3. TiltWatchdog
 *      - Monitors gravity[2] in body frame (== quat_rotate_inverse(quat,
 *        [0,0,-1])[2]). On the real robot, "upright" means gravity[2] is
 *        approximately -1.0. If it drifts above the configured threshold
 *        (default -0.3, matching eval_x2_mujoco --fall-tilt-cos), the
 *        watchdog trips and the deploy switches to "hold default angles
 *        with high damping" mode (no policy commands). The operator can
 *        then catch the robot manually.
 *
 * All three are pure CPU-side state machines; they do not own any ROS or
 * ONNX resources. The main loop calls them from the 50 Hz control timer.
 */

#ifndef AGI_X2_SAFETY_HPP
#define AGI_X2_SAFETY_HPP

#include "policy_parameters.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <string>

namespace agi_x2 {

/// Pure-data result of running the safety stack on one tick. Consumed by
/// the IO publisher to decide what to actually send to the joints.
struct SafeCommand {
  std::array<double, NUM_DOFS> target_pos_mj;
  std::array<double, NUM_DOFS> stiffness_mj;
  std::array<double, NUM_DOFS> damping_mj;
  bool                         dry_run    = false;  ///< if true, all kp/kd zeroed
  bool                         tilt_trip  = false;  ///< watchdog fired this tick
  double                       ramp_alpha = 1.0;    ///< 0=fully default, 1=fully policy
  std::string                  reason;              ///< human-readable status
};

/// Soft-start ramp: blends a freshly-computed policy target toward the
/// trained default standing pose during the first `ramp_seconds` seconds
/// of CONTROL state.
class SoftStartRamp {
 public:
  explicit SoftStartRamp(double ramp_seconds = 2.0)
      : ramp_seconds_(ramp_seconds) {}

  /// Reset on every entry to CONTROL state; alpha will start at 0.
  void Reset() { control_start_s_ = -1.0; }

  /// Compute alpha in [0, 1]. alpha=0 -> output = default, alpha=1 -> policy.
  double Alpha(double now_s);

  /// Blend in place: target_mj = (1 - alpha) * default + alpha * target_mj.
  void Apply(double alpha, std::array<double, NUM_DOFS>& target_mj) const;

 private:
  double ramp_seconds_;
  double control_start_s_ = -1.0;  ///< set on first Alpha() call after Reset()
};

/// Tilt watchdog: monitors body-frame gravity[2]. Once tripped, latches.
/// Use Reset() at INIT to clear.
class TiltWatchdog {
 public:
  /// @param fall_tilt_cos same sign convention as
  ///   gear_sonic/scripts/eval_x2_mujoco.py::--fall-tilt-cos. Default -0.3
  ///   means "trip when the body tilt exceeds ~72.5 degrees from upright".
  explicit TiltWatchdog(double fall_tilt_cos = -0.3)
      : fall_tilt_cos_(fall_tilt_cos) {}

  void   Reset()      { tripped_ = false; reason_.clear(); }
  bool   Tripped()    const { return tripped_; }
  double FallCos()    const { return fall_tilt_cos_; }
  const std::string& Reason() const { return reason_; }

  /// Returns true if the watchdog fires ON THIS TICK (transitioned from
  /// not-tripped to tripped). Subsequent calls return false but Tripped()
  /// stays true.
  bool Update(double gravity_body_z);

 private:
  double      fall_tilt_cos_;
  bool        tripped_ = false;
  std::string reason_;
};

/// Build the post-safety command for one control tick. Encapsulates the
/// dry-run / ramp / watchdog interaction so the main loop stays small.
///
/// Called from the 50 Hz control timer. The returned SafeCommand is then
/// published by the 500 Hz writer (which simply re-publishes the latest
/// SafeCommand without modification).
///
/// @param max_target_dev_rad  Per-joint hard clamp on |target - default|.
///   Applied AFTER the soft-start ramp blend, BEFORE the dry-run gain
///   zeroing. A non-positive value disables the clamp entirely (back to the
///   pre-clamp behaviour). Use a small value (e.g. 0.05 rad ~= 3 deg) for the
///   first powered bring-up runs so a divergent policy or obs-construction
///   bug cannot drive any joint more than `max_target_dev_rad` away from the
///   trained standing pose, regardless of what the ONNX session emits.
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             double max_target_dev_rad = -1.0);

}  // namespace agi_x2

#endif  // AGI_X2_SAFETY_HPP
