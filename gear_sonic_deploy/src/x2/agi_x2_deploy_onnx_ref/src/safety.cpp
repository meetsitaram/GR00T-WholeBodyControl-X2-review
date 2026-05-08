#include "safety.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace agi_x2 {

// ---------------------------------------------------------------------------
// SoftStartRamp
// ---------------------------------------------------------------------------
double SoftStartRamp::Alpha(double now_s)
{
  if (control_start_s_ < 0.0) {
    control_start_s_ = now_s;
    return 0.0;
  }
  if (ramp_seconds_ <= 0.0) return 1.0;
  const double a = (now_s - control_start_s_) / ramp_seconds_;
  return std::clamp(a, 0.0, 1.0);
}

void SoftStartRamp::Apply(double alpha,
                          std::array<double, NUM_DOFS>& target_mj) const
{
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    target_mj[i] = (1.0 - alpha) * default_angles[i] + alpha * target_mj[i];
  }
}

// ---------------------------------------------------------------------------
// TiltWatchdog
// ---------------------------------------------------------------------------
bool TiltWatchdog::Update(double gravity_body_z)
{
  if (tripped_) return false;
  // Upright: gravity[2] ~ -1.0. Trip when it climbs toward 0 or positive.
  // Same convention as eval_x2_mujoco.py::step_once: trip if grav_z > -0.3.
  if (gravity_body_z > fall_tilt_cos_) {
    tripped_ = true;
    const double g = std::clamp(-gravity_body_z, -1.0, 1.0);
    const double tilt_deg = std::acos(g) * 180.0 / 3.14159265358979323846;
    std::ostringstream os;
    os << "tilt watchdog tripped: gravity_body[z]=" << gravity_body_z
       << " > threshold " << fall_tilt_cos_
       << " (~" << static_cast<int>(tilt_deg) << " deg from upright)";
    reason_ = os.str();
    return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// ApplySafetyStack
// ---------------------------------------------------------------------------
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             double max_target_dev_rad)
{
  SafeCommand cmd{};

  // Default kp/kd come from the trained PD spec (codegen'd into policy
  // parameters). Dry-run zeros these later; tilt-trip overrides them.
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    cmd.stiffness_mj[i] = kps[i];
    cmd.damping_mj[i]   = kds[i];
  }

  // Tilt watchdog runs first so a freshly-tripped command goes to the safe
  // hold-default branch immediately.
  watchdog.Update(current_gravity_body_z);
  if (watchdog.Tripped()) {
    cmd.tilt_trip = true;
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      cmd.target_pos_mj[i] = default_angles[i];
      // Same kp as trained, but boost damping by 4x to gently slump back
      // to default. The trained kd is critically damped for the "follow
      // policy" use case; here we want to over-damp.
      cmd.damping_mj[i] = kds[i] * 4.0;
    }
    cmd.ramp_alpha = 0.0;
    cmd.reason     = watchdog.Reason();
  } else {
    // Normal path: soft-start blend.
    cmd.target_pos_mj = policy_target_mj;
    cmd.ramp_alpha    = ramp.Alpha(now_s);
    ramp.Apply(cmd.ramp_alpha, cmd.target_pos_mj);
    cmd.reason = "ok";
  }

  // Optional per-joint hard clamp on |target - default|. Applied AFTER the
  // ramp so a small max_target_dev_rad cleanly bounds the worst-case command
  // even when the policy is fully phased in (alpha = 1). Skipped on a
  // tilt-trip because that branch already pinned the target to default and
  // adding a clamp on top would be a no-op. Non-positive value = disabled.
  //
  // Rationale: a divergent policy (or an obs-construction bug that makes a
  // sane policy look divergent) can emit per-joint targets many radians from
  // the standing pose. With kp ~ 100 Nm/rad on the legs, that becomes
  // hundreds of Nm of impulse the moment MC steps aside and the gains come
  // back from zero. The clamp turns "policy can ask for anything" into
  // "policy can ask for at most max_target_dev_rad away from the trained
  // pose", which is the only safety property the C++ side can guarantee
  // without trusting the policy.
  if (!cmd.tilt_trip && max_target_dev_rad > 0.0) {
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      const double delta = cmd.target_pos_mj[i] - default_angles[i];
      const double clamped =
          std::max(-max_target_dev_rad, std::min(max_target_dev_rad, delta));
      cmd.target_pos_mj[i] = default_angles[i] + clamped;
    }
  }

  // Dry-run zeros the gains AFTER everything else, so the wiring (target
  // positions, joint name remap) is still exercised end-to-end while the
  // motors do nothing.
  if (dry_run) {
    cmd.dry_run = true;
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      cmd.stiffness_mj[i] = 0.0;
      cmd.damping_mj[i]   = 0.0;
    }
  }

  return cmd;
}

}  // namespace agi_x2
