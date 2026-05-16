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
// ApplySafetyStack -- full-control variant (preferred; takes per-DOF kp/kd)
// ---------------------------------------------------------------------------
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             const std::array<double, NUM_DOFS>& max_target_dev_per_dof,
                             const std::array<double, NUM_DOFS>& kp_per_dof,
                             const std::array<double, NUM_DOFS>& kd_per_dof)
{
  SafeCommand cmd{};

  // Effective kp/kd: caller passes the already-scaled values (e.g.
  // ``--kp-scale-ankle 1.5`` baked in upstream by BuildPdScalesPerDof).
  // Dry-run zeros these later; tilt-trip overrides damping below.
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    cmd.stiffness_mj[i] = kp_per_dof[i];
    cmd.damping_mj[i]   = kd_per_dof[i];
  }

  // Tilt watchdog runs first so a freshly-tripped command goes to the safe
  // hold-default branch immediately.
  watchdog.Update(current_gravity_body_z);
  if (watchdog.Tripped()) {
    cmd.tilt_trip = true;
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      cmd.target_pos_mj[i] = default_angles[i];
      // Same effective kp, but boost damping by 4x to gently slump back
      // to default. kd_per_dof[i] already incorporates any operator kd
      // scaling; the 4x is on top of that (the slump branch is the same
      // factor regardless of trim).
      cmd.damping_mj[i] = kd_per_dof[i] * 4.0;
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

  // Per-joint hard clamp on |target - default|, with PER-DOF thresholds.
  // Applied AFTER the ramp so a small clamp cleanly bounds the worst-case
  // command even when the policy is fully phased in (alpha = 1). Skipped
  // on a tilt-trip because that branch already pinned the target to
  // default and adding a clamp on top would be a no-op. An entry <= 0 in
  // max_target_dev_per_dof disables the clamp on that specific joint.
  //
  // Rationale: a divergent policy (or an obs-construction bug that makes a
  // sane policy look divergent) can emit per-joint targets many radians from
  // the standing pose. With kp ~ 99 Nm/rad on the legs, that becomes
  // hundreds of Nm of impulse the moment MC steps aside and the gains come
  // back from zero. With kp ~ 14 Nm/rad on the arms, the same nominal
  // deviation has 7x less torque, so arms can safely take 5x the angular
  // travel that legs can. Per-DOF clamps let us express that ratio
  // directly: legs/waist tight (e.g. 0.30 rad ~ 17 deg) for upright
  // stability, arms wide (e.g. 1.50 rad ~ 86 deg) for reach-target
  // following.
  if (!cmd.tilt_trip) {
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      const double clamp = max_target_dev_per_dof[i];
      if (clamp <= 0.0) continue;
      const double delta = cmd.target_pos_mj[i] - default_angles[i];
      const double clamped = std::max(-clamp, std::min(clamp, delta));
      cmd.target_pos_mj[i] = default_angles[i] + clamped;
    }
  }

  // Dry-run zeros the gains AFTER everything else, so the wiring (target
  // positions, joint name remap) is still exercised end-to-end while the
  // motors do nothing. Operator-supplied kp/kd scaling is honoured up to
  // this point; dry-run is the unconditional kill.
  if (dry_run) {
    cmd.dry_run = true;
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      cmd.stiffness_mj[i] = 0.0;
      cmd.damping_mj[i]   = 0.0;
    }
  }

  return cmd;
}

// ---------------------------------------------------------------------------
// ApplySafetyStack -- per-DOF clamp variant (uses trained kps/kds as-is)
// ---------------------------------------------------------------------------
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             const std::array<double, NUM_DOFS>& max_target_dev_per_dof)
{
  // Forward to the full-control overload with the constexpr trained PD as
  // the "effective" gains. Backward-compatible with callers that don't
  // care about deployment-time PD bumps.
  std::array<double, NUM_DOFS> kp_default{};
  std::array<double, NUM_DOFS> kd_default{};
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    kp_default[i] = kps[i];
    kd_default[i] = kds[i];
  }
  return ApplySafetyStack(policy_target_mj, current_gravity_body_z,
                          ramp, watchdog, dry_run, now_s,
                          max_target_dev_per_dof, kp_default, kd_default);
}

// ---------------------------------------------------------------------------
// ApplySafetyStack -- scalar clamp variant (legacy thin wrapper)
// ---------------------------------------------------------------------------
SafeCommand ApplySafetyStack(const std::array<double, NUM_DOFS>& policy_target_mj,
                             double current_gravity_body_z,
                             SoftStartRamp& ramp,
                             TiltWatchdog& watchdog,
                             bool dry_run,
                             double now_s,
                             double max_target_dev_rad)
{
  std::array<double, NUM_DOFS> per_dof{};
  per_dof.fill(max_target_dev_rad);
  return ApplySafetyStack(policy_target_mj, current_gravity_body_z,
                          ramp, watchdog, dry_run, now_s, per_dof);
}

}  // namespace agi_x2
