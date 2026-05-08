/**
 * @file x2_deploy_onnx_ref.cpp
 * @brief Reference real-robot deploy harness for the AgiBot X2 Ultra.
 *
 * Phase 3 of .cursor/plans/x2-ultra-onnx-deploy_9dde7da2.plan.md.
 *
 * ## Threading model
 *
 *   Thread / Timer    | Rate    | Responsibility
 *   ------------------|---------|----------------------------------------
 *   Subscriptions     | event   | Async ingest of leg/waist/arm/head/imu
 *                     |         | (mutex-guarded write into AimdkIo state)
 *   Control timer     | 50 Hz   | Snapshot state, build obs, run ONNX,
 *                     |         | apply safety stack, push SafeCommand to
 *                     |         | the latest_command_ slot.
 *   Command writer    | 500 Hz  | Read latest_command_ and PublishCommand.
 *                     |         | (10x oversampled re-publish; the policy
 *                     |         | target itself only changes at 50 Hz, but
 *                     |         | streaming it at the firmware's command
 *                     |         | rate keeps the on-bot PD loop fed.)
 *
 * All timers attach to a MultiThreadedExecutor so subscribers can deliver
 * messages while the control timer is mid-inference.
 *
 * ## State machine
 *
 *   STANDBY -> INIT -> WAIT_FOR_CONTROL -> CONTROL -> SAFE_HOLD
 *               ^                                       |
 *               +---------------------------------------+
 *
 *   STANDBY          : OPTIONAL pre-INIT state, only entered when
 *                       --start-trigger-sentinel is set. ROS subscribers
 *                       are active, ONNX is loaded, MC-takeover detectors
 *                       armed -- but the 500 Hz writer is GATED OFF (no
 *                       joint commands published). bash uses this to
 *                       launch the binary BEFORE stop_app + safety gate
 *                       so colcon build / DDS discovery / model load
 *                       overlap with the operator's "Y" decision. Exits
 *                       to INIT when bash touches the trigger sentinel.
 *   INIT             : waiting for first leg/waist/arm/head/IMU message
 *                       (AimdkIo::AllStateFresh(0.5)). Publishes nothing.
 *   WAIT_FOR_CONTROL : have valid state, waiting for operator "go" via
 *                       --autostart-after=N (N seconds delay) OR Ctrl-C-then-rerun
 *                       OR (future) the operator service. Publishes nothing.
 *   CONTROL          : runs the policy, applies the safety stack, publishes
 *                       commands. The 500 Hz writer is allowed to publish.
 *   SAFE_HOLD        : tilt watchdog tripped or fatal error. The 500 Hz
 *                       writer publishes "hold default angles, 4x damping"
 *                       indefinitely; operator must restart the binary.
 *
 * ## CLI (selected)
 *
 *   --model PATH               Path to fused g1+g1_dyn ONNX (required)
 *   --motion PATH              Optional X2M2 motion file for the tokenizer
 *                              reference. Default: StandStill.
 *   --autostart-after SECONDS        Auto-transition WAIT->CONTROL after N seconds.
 *   --dry-run                  Publish stiffness=0/damping=0 (no torque).
 *   --tilt-cos COS             Watchdog threshold (default -0.3).
 *   --ramp-seconds SECONDS     Soft-start blend duration (default 2.0).
 *   --log-dir PATH             Per-tick CSVs go here. Empty = no logs.
 */

#include "aimdk_io.hpp"
#include "deploy_logger.hpp"
#include "math_utils.hpp"
#include "onnx_actor.hpp"
#include "policy_parameters.hpp"
#include "proprioception_buffer.hpp"
#include "reference_motion.hpp"
#include "safety.hpp"
#include "stand_pose_loader.hpp"
#include "tokenizer_obs.hpp"

#include <aimdk_msgs/msg/joint_command_array.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/subscription_options.hpp>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

namespace agi_x2 {

// ---------------------------------------------------------------------------
// CLI parsing (single-purpose; lightweight; no third-party deps)
// ---------------------------------------------------------------------------
struct CliArgs {
  std::string model_path;
  std::string motion_path;             // empty -> StandStillReference
  std::string log_dir;                 // empty -> logging disabled
  double      autostart_seconds = -1.0;// negative -> wait for operator stdin "go"
  double      max_duration      = -1.0;// negative -> run until Ctrl-C; positive
                                       // -> shutdown N seconds after entering
                                       // CONTROL. Used by --dry-run smoke tests
                                       // to bound runtime so the operator isn't
                                       // expected to babysit Ctrl-C.
  bool        dry_run           = false;
  double      tilt_cos          = -0.3;
  double      ramp_seconds      = 2.0;
  // Per-joint hard clamp on |target - default_angles|, in radians. Negative
  // = disabled. See safety.hpp::ApplySafetyStack for rationale; intended for
  // first-powered-bring-up runs (e.g. 0.05 rad ~= 3 deg) so a divergent
  // policy or obs-construction bug cannot drive any joint more than this
  // many radians from the trained standing pose.
  double      max_target_dev    = -1.0;
  // Symmetric clamp on the raw ONNX action (action_il) BEFORE multiplying by
  // x2_action_scale. This mirrors IsaacLab's training-time clamp applied in
  // ManagerEnvWrapper.step (controlled by config ``action_clip_value`` in
  // gear_sonic/config/manager_env/base_env.yaml, default 20.0). Without this,
  // a saturated/diverged policy can emit O(100 rad) action_il, which the
  // ``--max-target-dev`` safety stack truncates -- but the truncated motor
  // command never matches what training observed, so the policy's
  // ``last_action`` proprioception feedback drifts away from training
  // distribution and the loop runs away. Default 20.0 matches training.
  // Set to a negative value to disable (only useful for parity tests).
  double      action_clip       = 20.0;
  // Soft-EXIT ramp (counterpart to --ramp-seconds, the soft-START). When
  // --max-duration trips we don't want to just stop publishing -- the policy
  // may have left the joints far from the trained standing pose (e.g. mid-
  // swing on a take_a_sip / hot_dog motion), and MC's PD-hold will snap
  // them back the moment start_app is POSTed during cleanup, which can fault
  // the MC unit (red-flashing-light territory on X2 Ultra). RAMP_OUT
  // linearly interpolates target_pos from the last policy command to
  // default_angles over `return_seconds` while keeping deploy-mode kp/kd
  // active, so we drive the joints back instead of letting them flop. Set
  // to <=0 to keep the legacy "shutdown immediately" behaviour.
  double      return_seconds    = 2.0;
  // Output-side first-order EMA on the published joint targets, applied
  // AFTER the policy has produced action_il and AFTER the safety stack has
  // computed sc.target_pos_mj. Purely a real-deploy jitter mitigation; the
  // policy still SEES the same observation it always has, so:
  //   * --obs-dump emits raw policy outputs (this LPF runs after the dump
  //     return path), keeping compare_deploy_vs_python_obs.py bit-exact;
  //   * sim profiles in deploy_x2.sh leave this at 0 (= bypass) so MuJoCo
  //     parity (eval_x2_mujoco.py) is preserved by construction;
  //   * RAMP_OUT and SAFE_HOLD bypass the LPF -- those states already
  //     produce a deliberate trajectory we don't want to attenuate.
  // alpha = 1 - exp(-2*pi*hz*dt) at the 50 Hz OnControl rate, so e.g.
  // hz=8 -> alpha~=0.63 (~5 Hz effective bandwidth). 0 = disabled.
  // Documented under gear_sonic_deploy/configs/real_deploy_tuning/.
  double      target_lpf_hz     = 0.0;
  int         intra_op_threads  = 1;
  // Optional topic overrides. Empty string -> use AimdkIo defaults
  // (which match the canonical names in the SDK `topics_and_services`
  // registry: /aima/hal/joint/{leg,waist,arm,head}/{state,command} and
  // /aima/hal/imu/torso/state).
  std::string imu_topic;
  // If non-empty, the first CONTROL tick will write the full inference
  // payload (tokenizer 680 + proprioception 990 + raw policy output 31)
  // to PATH as a binary blob and *immediately exit* (so we don't keep
  // commanding a robot whose obs we don't trust). The companion script
  // ``gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py``
  // diffs that blob against /tmp/x2_step0_isaaclab_lastpt.pt slot-by-slot.
  // Used for debugging policy divergence on the real robot. See
  // docs/source/references/x2_deployment_code.md for the file format.
  std::string obs_dump_path;
  // ────────────────────────────────────────────────────────────────────
  // End-of-run smooth handoff (HOLD_FOR_MC).
  //
  // Default policy: when --max-duration trips, after RAMP_OUT lerps back
  // to the trained ``default_angles``, the deploy node exits and bash
  // POSTs start_app to bring MC back. The robot is therefore in zero
  // torque (PASSIVE_DEFAULT) for the ~5-15 s MC takes to boot. To close
  // that window, the deploy node can stay alive in a ``HOLD_FOR_MC``
  // state holding MC's STAND_DEFAULT pose, while bash drives MC back.
  // The deploy node detects MC's takeover by listening on
  // /aima/hal/joint/{leg,waist}/command for any non-self publisher and
  // exits cleanly the moment MC's first message arrives. Sentinel file
  // is touched on entering HOLD_FOR_MC so bash can sequence start_app.
  // ────────────────────────────────────────────────────────────────────
  // Path to the stand-pose YAML produced by the capture script. Empty
  // = fall back to default_angles for both RAMP_OUT and HOLD_FOR_MC
  // (legacy behaviour). See configs/x2_stand_default_pose.yaml.
  std::string stand_pose_path;
  // Cap on time spent in HOLD_FOR_MC. <=0 disables HOLD_FOR_MC entirely
  // (RAMP_OUT exits the process directly, like before). When >0, the
  // deploy node holds MC's stand pose for up to N seconds waiting for
  // MC's bus takeover, then exits with a warning if MC never came back.
  double      hold_for_mc_timeout_s = 0.0;
  // If non-empty, touch this file when the deploy node enters
  // HOLD_FOR_MC. Lets bash know "policy phase is done; you can now
  // start_app + SetMcAction(STAND_DEFAULT) safely" without parsing
  // stdout. Cleared on clean exit.
  std::string hold_for_mc_sentinel;
  // If non-empty, HOLD_FOR_MC stops auto-exiting on the first detected
  // MC publish (because MC boots in PASSIVE_DEFAULT = zero torque, and
  // exiting then would let the robot go limp). Instead, deploy waits
  // for THIS sentinel to appear -- bash creates it after escalating MC
  // all the way to STAND_DEFAULT. The MC-takeover detector subscribers
  // are still armed (and logged for diagnostics), they just don't
  // trigger exit. The hold_for_mc_timeout_s safety cap still applies.
  std::string hold_for_mc_exit_sentinel;
  // If non-empty, deploy touches this file the moment it sees its
  // FIRST non-self publisher on /aima/hal/joint/{leg,waist}/command
  // (= MC has started publishing again, very likely in
  // PASSIVE_DEFAULT). Bash uses this as a fast signal to start the
  // SetMcAction(JOINT_DEFAULT) escalation immediately, BYPASSING the
  // slow mc_get_action poll loop (which can lag MC's actual first
  // publish by 0.5-0.8 s on the standing-gestures runs). Combined
  // with a fast-poll mode-check in bash, this compresses the
  // PASSIVE_DEFAULT dwell time (= dual-publisher whir window) from
  // ~1.5 s to roughly the SetMcAction round-trip latency. Independent
  // of hold_for_mc_exit_sentinel; touching mc_first_publish_sentinel
  // does NOT trigger exit on its own.
  std::string mc_first_publish_sentinel;
  // ────────────────────────────────────────────────────────────────────
  // Pre-launch STANDBY (cold-warm boot).
  //
  // Default flow without these flags is "deploy starts, immediately
  // tries to enter INIT, the writer streams the safe-hold latch from
  // the moment it has fresh state". That assumes MC is already stopped
  // before the binary launches -- otherwise MC + deploy would fight on
  // the bus during the startup window. The bash script enforces this
  // by stop_app'ing MC before `ros2 run`.
  //
  // The bash script can flip the order so the C++ binary starts BEFORE
  // the operator confirms the safety gate (cold-warm boot: build, ONNX
  // load, ROS subscribers settle while the operator is still reading
  // the prompt). For that, we need a "alive but silent" state where
  // the writer publishes nothing, so deploy doesn't dual-publish with
  // MC. STANDBY is that state, and start_trigger_sentinel is how bash
  // (post-stop_app + verify) tells us "you can take the bus now". When
  // start_trigger_sentinel is empty we skip STANDBY entirely and behave
  // exactly like before.
  // ────────────────────────────────────────────────────────────────────
  // If non-empty, deploy boots into STANDBY (no publishing) and stays
  // there until this file exists. Once it does, STANDBY -> INIT and
  // the rest of the state machine runs as normal. Bash is responsible
  // for stop_app'ing MC BEFORE touching this file.
  std::string start_trigger_sentinel;
  // If non-empty, deploy touches this file once it has reached its
  // ready point (subscribers up, ONNX loaded, MC-takeover detectors
  // armed) so bash can show its safety-gate prompt with confidence
  // that the next step (touch start_trigger_sentinel) is sub-second.
  // Independent of start_trigger_sentinel; safe to use on its own.
  std::string ready_sentinel;
};

void PrintUsage()
{
  std::cout
      << "Usage: x2_deploy_onnx_ref --model PATH [options]\n"
      << "  --model PATH               (required) fused g1+g1_dyn ONNX\n"
      << "  --motion PATH              X2M2 reference motion (else stand-still)\n"
      << "  --autostart-after SECONDS        auto-go after N seconds (else wait stdin)\n"
      << "  --max-duration SECONDS     auto-shutdown N seconds after entering CONTROL\n"
      << "                             (default: run until Ctrl-C). Useful for bounded\n"
      << "                             dry-run smoke tests.\n"
      << "  --dry-run                  publish stiffness=0/damping=0\n"
      << "  --tilt-cos COS             tilt watchdog threshold (default -0.3)\n"
      << "  --ramp-seconds SECONDS     soft-start ramp (default 2.0)\n"
      << "  --max-target-dev RAD       per-joint hard clamp on |target-default|,\n"
      << "                             in radians. Negative/omitted = disabled.\n"
      << "                             Use 0.05 (~3 deg) for first powered runs\n"
      << "                             so a divergent policy cannot drive any joint\n"
      << "                             more than RAD away from the standing pose.\n"
      << "  --action-clip RAD          symmetric clip on the raw ONNX action\n"
      << "                             (action_il) BEFORE x2_action_scale (default\n"
      << "                             20.0, matches training-time\n"
      << "                             config.action_clip_value in\n"
      << "                             gear_sonic/config/manager_env/base_env.yaml).\n"
      << "                             Negative = disabled (parity tests only).\n"
      << "  --return-seconds SECONDS   soft-exit ramp duration (default 2.0). When\n"
      << "                             --max-duration trips, lerp target_pos from\n"
      << "                             the last policy command to default_angles\n"
      << "                             over SECONDS (deploy-mode PD active) before\n"
      << "                             shutdown -- prevents MC from snapping joints\n"
      << "                             back at handoff (red-fault on X2 Ultra).\n"
      << "                             Set 0 to disable (legacy immediate-shutdown).\n"
      << "  --target-lpf-hz HZ         REAL-DEPLOY ONLY: first-order EMA cutoff\n"
      << "                             applied to the published joint targets to\n"
      << "                             tame leg/waist jitter caused by noisy real\n"
      << "                             sensor obs. Bypassed in RAMP_OUT/SAFE_HOLD.\n"
      << "                             Default 0 (disabled). Sim parity profiles\n"
      << "                             MUST leave this at 0 -- the LPF is invisible\n"
      << "                             to --obs-dump (raw target preserved) but it\n"
      << "                             changes what the bus sees, which would\n"
      << "                             diverge from eval_x2_mujoco.py's reference.\n"
      << "  --log-dir PATH             write per-tick CSVs to PATH\n"
      << "  --intra-op-threads N       ONNX session threads (default 1)\n"
      << "  --imu-topic NAME           override IMU topic (default /aima/hal/imu/torso/state;\n"
      << "                             use /aima/hal/imu/torse/state on firmware that\n"
      << "                             ships with the SDK-example typo)\n"
      << "  --obs-dump PATH            DEBUG: dump the first CONTROL-tick obs\n"
      << "                             (tokenizer + proprioception + raw action) to\n"
      << "                             PATH as a binary blob and exit immediately.\n"
      << "                             Pair with --dry-run + --autostart-after for a\n"
      << "                             deterministic capture from a known robot pose.\n"
      << "                             See compare_deploy_vs_isaaclab_obs.py.\n"
      << "  --stand-default-pose PATH  YAML file capturing MC's STAND_DEFAULT pose\n"
      << "                             (configs/x2_stand_default_pose.yaml). When\n"
      << "                             provided, RAMP_OUT lerps to *this* pose and\n"
      << "                             HOLD_FOR_MC publishes it -- so the joints land\n"
      << "                             exactly where MC will resume from, eliminating\n"
      << "                             the takeover step. Empty = use default_angles\n"
      << "                             (legacy: ~12-34 deg snap on takeover).\n"
      << "  --hold-for-mc-timeout-s SECONDS\n"
      << "                             Stay alive after RAMP_OUT for up to SECONDS,\n"
      << "                             holding MC's STAND_DEFAULT pose, until MC's\n"
      << "                             first joint command arrives on the bus (then\n"
      << "                             exit cleanly). 0 = disabled (legacy: exit at\n"
      << "                             end of RAMP_OUT, MC bus is silent until bash\n"
      << "                             POSTs start_app). Recommended: 15.\n"
      << "  --hold-for-mc-sentinel PATH\n"
      << "                             Touch PATH on entering HOLD_FOR_MC. Used by\n"
      << "                             deploy_x2.sh to sequence start_app +\n"
      << "                             SetMcAction(STAND_DEFAULT) at the right\n"
      << "                             moment. Empty = no sentinel.\n"
      << "  --hold-for-mc-exit-sentinel PATH\n"
      << "                             When set, HOLD_FOR_MC stops auto-exiting on\n"
      << "                             the first MC publish (because MC boots in\n"
      << "                             PASSIVE_DEFAULT = zero torque -- exiting then\n"
      << "                             would let the robot go limp). Instead, the\n"
      << "                             deploy node waits for PATH to appear; bash\n"
      << "                             creates it once MC has been escalated all the\n"
      << "                             way back to STAND_DEFAULT. The hold-for-mc\n"
      << "                             timeout is still a hard upper bound.\n"
      << "  --mc-first-publish-sentinel PATH\n"
      << "                             Touch PATH the moment the takeover detector\n"
      << "                             sees its first non-self publisher on\n"
      << "                             /aima/hal/joint/{leg,waist}/command (= MC\n"
      << "                             has come back online, likely in PASSIVE).\n"
      << "                             bash uses this as a fast trigger to start\n"
      << "                             SetMcAction(JOINT_DEFAULT) without waiting\n"
      << "                             for MC's mode service to respond. Does NOT\n"
      << "                             trigger exit on its own (see exit-sentinel).\n"
      << "  --start-trigger-sentinel PATH\n"
      << "                             Boot into STANDBY (writer suppressed) and wait\n"
      << "                             for PATH to exist before advancing to INIT.\n"
      << "                             Lets bash launch deploy AHEAD of stop_app +\n"
      << "                             safety gate to overlap colcon build / ONNX\n"
      << "                             load / DDS discovery with the operator's\n"
      << "                             confirmation. Empty = boot straight to INIT.\n"
      << "  --ready-sentinel PATH      Touch PATH on the first STANDBY tick to tell\n"
      << "                             bash that subscribers / ONNX / takeover\n"
      << "                             detectors are armed and the safety gate can\n"
      << "                             be shown. Independent of start-trigger; safe\n"
      << "                             to use on its own.\n"
      << "  --help, -h                 show this help\n";
}

CliArgs ParseCli(int argc, char** argv)
{
  CliArgs a;
  for (int i = 1; i < argc; ++i) {
    const std::string s = argv[i];
    auto next = [&](const char* flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + flag);
      }
      return argv[++i];
    };
    if (s == "--help" || s == "-h") { PrintUsage(); std::exit(0); }
    else if (s == "--model")             a.model_path        = next("--model");
    else if (s == "--motion")            a.motion_path       = next("--motion");
    else if (s == "--log-dir")           a.log_dir           = next("--log-dir");
    else if (s == "--autostart-after")         a.autostart_seconds = std::stod(next("--autostart-after"));
    else if (s == "--max-duration")      a.max_duration      = std::stod(next("--max-duration"));
    else if (s == "--dry-run")           a.dry_run           = true;
    else if (s == "--tilt-cos")          a.tilt_cos          = std::stod(next("--tilt-cos"));
    else if (s == "--ramp-seconds")      a.ramp_seconds      = std::stod(next("--ramp-seconds"));
    else if (s == "--max-target-dev")    a.max_target_dev    = std::stod(next("--max-target-dev"));
    else if (s == "--action-clip")       a.action_clip       = std::stod(next("--action-clip"));
    else if (s == "--return-seconds")    a.return_seconds    = std::stod(next("--return-seconds"));
    else if (s == "--target-lpf-hz")     a.target_lpf_hz     = std::stod(next("--target-lpf-hz"));
    else if (s == "--intra-op-threads")  a.intra_op_threads  = std::stoi(next("--intra-op-threads"));
    else if (s == "--imu-topic")         a.imu_topic         = next("--imu-topic");
    else if (s == "--obs-dump")          a.obs_dump_path     = next("--obs-dump");
    else if (s == "--stand-default-pose") a.stand_pose_path  = next("--stand-default-pose");
    else if (s == "--hold-for-mc-timeout-s")
      a.hold_for_mc_timeout_s = std::stod(next("--hold-for-mc-timeout-s"));
    else if (s == "--hold-for-mc-sentinel")
      a.hold_for_mc_sentinel = next("--hold-for-mc-sentinel");
    else if (s == "--hold-for-mc-exit-sentinel")
      a.hold_for_mc_exit_sentinel = next("--hold-for-mc-exit-sentinel");
    else if (s == "--mc-first-publish-sentinel")
      a.mc_first_publish_sentinel = next("--mc-first-publish-sentinel");
    else if (s == "--start-trigger-sentinel")
      a.start_trigger_sentinel = next("--start-trigger-sentinel");
    else if (s == "--ready-sentinel")
      a.ready_sentinel = next("--ready-sentinel");
    else {
      throw std::runtime_error("unknown argument: " + s);
    }
  }
  if (a.model_path.empty()) {
    throw std::runtime_error("--model is required");
  }
  return a;
}

// ---------------------------------------------------------------------------
// Top-level controller -- mirrors the G1Deploy class but vastly slimmer.
// ---------------------------------------------------------------------------
class X2Deploy {
 public:
  // STANDBY -> INIT -> WAIT_FOR_CONTROL -> CONTROL -> RAMP_OUT -> HOLD_FOR_MC
  //              ^                                       |
  //              +---------------------------------------+
  //                                                      |
  //                                                      v
  //                                                  SAFE_HOLD (terminal)
  //
  // STANDBY is an optional "alive but silent" state at the head of the
  // state machine, used when --start-trigger-sentinel is set. In STANDBY:
  //   * subscribers are active (ROS state is being collected so the
  //     INIT->WAIT freshness check passes immediately on transition)
  //   * the 500 Hz writer is GATED OFF (no joint commands published)
  //   * we poll the start-trigger sentinel each control tick (20 ms)
  // bash uses this to launch deploy in the background AHEAD of the
  // safety gate (parallelising colcon build / ONNX load / DDS discovery
  // with the operator reading the prompt). Once the operator confirms
  // and bash POSTs stop_app + verify, bash touches the trigger sentinel
  // and we move STANDBY -> INIT in the next tick.
  //
  // Without --start-trigger-sentinel deploy boots straight to INIT
  // (legacy path): bash must stop_app BEFORE launching the binary.
  //
  // HOLD_FOR_MC keeps the deploy node alive after RAMP_OUT publishing
  // MC's STAND_DEFAULT pose with MC-stand kp/kd (firmer than deploy
  // gains, matching MC's own stiffness for a stable static hold),
  // until any non-self publisher appears on /aima/hal/joint/{leg,waist}/
  // command (= MC has taken back over). The exit is fast: the DDS
  // callback for MC's first publish fires sub-ms after MC enters
  // JOINT_DEFAULT, and the next OnControl tick exits within <= 20 ms
  // (50 Hz). The exit-sentinel (touched by deploy_x2.sh once MC is in
  // STAND_DEFAULT) is honoured as a redundant backup so an external
  // mode switch (mobile app, ROS service from another shell) can free
  // us even if the takeover detector misses. On takeover (or on
  // hold-for-mc-timeout-s timeout) we shut down cleanly. If
  // --hold-for-mc-timeout-s is 0,
  // RAMP_OUT exits the process directly (legacy behaviour).
  enum class State {
    STANDBY,
    INIT,
    WAIT_FOR_CONTROL,
    CONTROL,
    RAMP_OUT,
    HOLD_FOR_MC,
    SAFE_HOLD,
  };

  X2Deploy(rclcpp::Node::SharedPtr node, const CliArgs& cli)
      : node_(node),
        cli_(cli),
        ramp_(cli.ramp_seconds),
        watchdog_(cli.tilt_cos),
        logger_(cli.log_dir.empty() ? std::string{} : cli.log_dir,
                /*enabled=*/!cli.log_dir.empty())
  {
    if (cli.imu_topic.empty()) {
      aimdk_io_ = std::make_unique<AimdkIo>(node_);
    } else {
      aimdk_io_ = std::make_unique<AimdkIo>(
          node_,
          /*leg=*/  "/aima/hal/joint/leg",
          /*waist=*/"/aima/hal/joint/waist",
          /*arm=*/  "/aima/hal/joint/arm",
          /*head=*/ "/aima/hal/joint/head",
          /*imu=*/  cli.imu_topic);
      RCLCPP_INFO(node_->get_logger(),
                  "AimdkIo: IMU topic overridden via CLI -> '%s'",
                  cli.imu_topic.c_str());
    }

    if (cli.motion_path.empty()) {
      ref_motion_ = std::make_unique<StandStillReference>();
      RCLCPP_INFO(node_->get_logger(),
                  "Reference motion: StandStill (default standing pose)");
    } else {
      ref_motion_ = PklMotionReference::Load(cli.motion_path);
      RCLCPP_INFO(node_->get_logger(),
                  "Reference motion: PklMotionReference '%s'",
                  cli.motion_path.c_str());
    }

    onnx_actor_ = std::make_unique<OnnxActor>(cli.model_path, cli.intra_op_threads);
    RCLCPP_INFO(node_->get_logger(),
                "Loaded ONNX: %s  (input='%s' [1, %ld])",
                onnx_actor_->model_path().c_str(),
                onnx_actor_->input_name().c_str(),
                static_cast<long>(onnx_actor_->expected_obs_dim()));

    // Loud-and-proud announcement of the safety knobs that materially affect
    // worst-case actuation. Operator should see these in the deploy log
    // before saying "go", so a missing --max-target-dev on a powered run is
    // visible at a glance instead of buried in the help text.
    if (cli_.max_target_dev > 0.0) {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: per-joint target clamp ENABLED at "
                  "|target - default| <= %.3f rad (%.1f deg)",
                  cli_.max_target_dev,
                  cli_.max_target_dev * 180.0 / 3.14159265358979323846);
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: per-joint target clamp DISABLED "
                  "(--max-target-dev not set). Policy can drive any joint "
                  "to any value the ONNX session emits.");
    }

    if (cli_.action_clip > 0.0) {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: raw action clip ENABLED at |action_il| <= %.3f "
                  "(matches training action_clip_value=20.0 from base_env.yaml)",
                  cli_.action_clip);
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: raw action clip DISABLED "
                  "(--action-clip <= 0). Deploy will diverge from training "
                  "wrapper behavior; only use for parity tests.");
    }

    if (cli_.return_seconds > 0.0) {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: soft-exit ramp ENABLED (--return-seconds %.2fs). "
                  "When --max-duration trips, joints will be lerped back to "
                  "default_angles before shutdown so MC handoff doesn't fault.",
                  cli_.return_seconds);
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "SAFETY: soft-exit ramp DISABLED (--return-seconds <= 0). "
                  "Deploy will shut down immediately on --max-duration; if the "
                  "policy left joints far from default_angles, the next MC "
                  "start_app POST may snap them back and trip a red fault.");
    }

    // ─── End-of-run smooth handoff: load MC's STAND_DEFAULT pose ────────
    // If --stand-default-pose was given, we use the captured pose+kp+kd
    // for both RAMP_OUT (lerp target) and HOLD_FOR_MC (static publish).
    // Otherwise we fall back to default_angles + deploy-mode kp/kd, which
    // mismatches MC's pose by up to 34 deg at the elbows -- the operator
    // sees an audible pop on takeover. Print a loud diff summary so the
    // operator notices when the YAML is missing.
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      stand_pose_target_[i]    = default_angles[i];
      stand_pose_stiffness_[i] = kps[i];
      stand_pose_damping_[i]   = kds[i];
    }
    if (!cli_.stand_pose_path.empty()) {
      try {
        const auto sp = LoadStandPose(cli_.stand_pose_path);
        double max_diff_rad = 0.0;
        std::size_t worst_idx = 0;
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          stand_pose_target_[i]    = sp.position[i];
          stand_pose_stiffness_[i] = sp.stiffness[i];
          stand_pose_damping_[i]   = sp.damping[i];
          const double d = std::abs(sp.position[i] - default_angles[i]);
          if (d > max_diff_rad) { max_diff_rad = d; worst_idx = i; }
        }
        RCLCPP_WARN(node_->get_logger(),
                    "HANDOFF: loaded MC STAND_DEFAULT pose from '%s' "
                    "(31 joints). Worst delta vs default_angles: %.3f rad "
                    "(%.1f deg) at '%s'. RAMP_OUT and HOLD_FOR_MC will "
                    "target this pose so MC takeover is step-free.",
                    cli_.stand_pose_path.c_str(),
                    max_diff_rad,
                    max_diff_rad * 180.0 / 3.14159265358979323846,
                    mujoco_joint_names[worst_idx]);
      } catch (const std::exception& e) {
        RCLCPP_FATAL(node_->get_logger(),
                     "HANDOFF: failed to load --stand-default-pose '%s': %s. "
                     "Aborting (refusing to start with an unknown handoff "
                     "target). Pass --stand-default-pose '' to fall back to "
                     "default_angles, or fix the YAML.",
                     cli_.stand_pose_path.c_str(), e.what());
        throw;
      }
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "HANDOFF: --stand-default-pose not provided; RAMP_OUT and "
                  "HOLD_FOR_MC will target default_angles. MC takeover may "
                  "snap up to 34 deg at the elbows. Pass "
                  "--stand-default-pose configs/x2_stand_default_pose.yaml "
                  "to eliminate the snap.");
    }

    if (cli_.hold_for_mc_timeout_s > 0.0) {
      if (cli_.hold_for_mc_exit_sentinel.empty()) {
        RCLCPP_WARN(node_->get_logger(),
                    "HANDOFF: HOLD_FOR_MC enabled (timeout %.1fs). After "
                    "RAMP_OUT, deploy will keep publishing MC's STAND_DEFAULT "
                    "pose until MC's first joint command arrives on the bus, "
                    "then exit cleanly (legacy fast-exit path).",
                    cli_.hold_for_mc_timeout_s);
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "HANDOFF: HOLD_FOR_MC enabled (timeout %.1fs). After "
                    "RAMP_OUT, deploy will keep publishing MC's STAND_DEFAULT "
                    "pose until bash touches the exit-sentinel '%s' (set after "
                    "MC has escalated all the way to STAND_DEFAULT). No zero-"
                    "torque window during MC boot's PASSIVE -> JOINT -> STAND.",
                    cli_.hold_for_mc_timeout_s,
                    cli_.hold_for_mc_exit_sentinel.c_str());
      }
      InitMcTakeoverDetectors();
    } else {
      RCLCPP_WARN(node_->get_logger(),
                  "HANDOFF: HOLD_FOR_MC disabled (--hold-for-mc-timeout-s "
                  "<= 0). RAMP_OUT will exit the process; the bus will be "
                  "silent until bash POSTs start_app.");
    }
    if (!cli_.hold_for_mc_sentinel.empty()) {
      RCLCPP_INFO(node_->get_logger(),
                  "HANDOFF: HOLD_FOR_MC sentinel = '%s' "
                  "(touched on entering HOLD_FOR_MC).",
                  cli_.hold_for_mc_sentinel.c_str());
    }

    // STANDBY support. If --start-trigger-sentinel is set, boot into
    // STANDBY (writer suppressed, no joint commands published) and wait
    // for bash to touch the file before advancing to INIT. This lets
    // bash launch the binary AHEAD of the safety gate to overlap colcon
    // build / ONNX load / DDS discovery with the operator's "Y" decision.
    if (!cli_.start_trigger_sentinel.empty()) {
      state_.store(State::STANDBY);
      RCLCPP_WARN(node_->get_logger(),
                  "STANDBY: --start-trigger-sentinel='%s' provided; deploy "
                  "is alive but the 500 Hz writer is GATED OFF. Will "
                  "advance STANDBY -> INIT when this file appears.",
                  cli_.start_trigger_sentinel.c_str());
      // Best-effort cleanup of any stale trigger left over from a prior
      // crash. If the file is owned by another user we may fail; that's
      // harmless -- bash recreates it fresh.
      std::remove(cli_.start_trigger_sentinel.c_str());
    }

    // Compute the EMA coefficient now so OnControl can apply it without
    // re-deriving every tick. dt is fixed at 1/50 s (the OnControl rate);
    // alpha = 1 - exp(-2*pi*hz*dt) is the standard discrete first-order
    // low-pass coefficient. hz<=0 -> alpha=0 (bypass).
    if (cli_.target_lpf_hz > 0.0) {
      const double dt = 1.0 / 50.0;
      const double pi = 3.14159265358979323846;
      target_lpf_alpha_ = 1.0 - std::exp(-2.0 * pi * cli_.target_lpf_hz * dt);
      RCLCPP_WARN(node_->get_logger(),
                  "REAL-DEPLOY: output target LPF ENABLED (--target-lpf-hz %.2f Hz, "
                  "alpha=%.3f at 50 Hz OnControl). Bypassed in RAMP_OUT/SAFE_HOLD. "
                  "Sim parity (eval_x2_mujoco.py) is preserved -- this filter "
                  "lives strictly downstream of the policy and never affects "
                  "--obs-dump output.",
                  cli_.target_lpf_hz, target_lpf_alpha_);
    } else {
      target_lpf_alpha_ = 0.0;
    }

    // Initial safe command: PASSIVE (kp=0, kd=0) until any state arrives.
    //
    // The 500 Hz writer starts publishing as soon as the timer fires, which
    // is *before* INIT clears (i.e. before any joint state has been
    // received from HAL). If we latched full kp/kd here against
    // default_angles, the writer would yank every joint toward
    // default_angles for the entire pre-INIT window -- which on the real
    // robot fights MC's standing pose, and in sim mode (with the bridge's
    // standby PD already holding a non-default pose, e.g. --init-pose=
    // gantry_hang) tips the body over before the policy ever gets a tick
    // (verified: 95 deg tilt in 3 s of autostart). Publishing kp=kd=0 at
    // startup means the writer applies zero torque, leaving HAL / MC / the
    // sim bridge fully in charge until INIT->WAIT_FOR_CONTROL.
    //
    // The actual safe-hold pose is latched in OnControl() at the
    // INIT->WAIT_FOR_CONTROL transition, using the *current* observed
    // joint pose (rs.joint_pos_mj). That way the deploy holds whatever
    // pose the operator/MC/bridge had at the moment of handoff, so
    // WAIT_FOR_CONTROL is genuinely safe regardless of how far the start
    // pose is from DEFAULT_DOF.
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      for (std::size_t i = 0; i < NUM_DOFS; ++i) {
        latest_cmd_.target_pos_mj[i] = default_angles[i];
        latest_cmd_.stiffness_mj[i]  = 0.0;
        latest_cmd_.damping_mj[i]    = 0.0;
      }
      latest_cmd_.dry_run    = cli_.dry_run;
      latest_cmd_.tilt_trip  = false;
      latest_cmd_.ramp_alpha = 0.0;
      latest_cmd_.reason     = "init_passive";
    }

    // Timers. Both attach to whatever executor spins this node.
    control_timer_ = node_->create_wall_timer(
        std::chrono::milliseconds(20),  // 50 Hz
        std::bind(&X2Deploy::OnControl, this));
    writer_timer_ = node_->create_wall_timer(
        std::chrono::milliseconds(2),   // 500 Hz
        std::bind(&X2Deploy::OnWriter, this));

    // Optional autostart watchdog: flips WAIT->CONTROL after N seconds.
    if (cli_.autostart_seconds >= 0.0) {
      autostart_target_s_ = SteadyNow() + cli_.autostart_seconds;
    }
  }

  State state() const { return state_.load(); }

  /// Operator-triggered transition (stdin "go" or future service call).
  /// Safe to call from any thread; takes effect on the next OnControl tick.
  void RequestGo()
  {
    if (state_.load() != State::WAIT_FOR_CONTROL) return;
    autostart_target_s_ = SteadyNow();
    RCLCPP_INFO(node_->get_logger(),
                "Operator GO received; transitioning on next tick.");
  }

 private:
  static double SteadyNow()
  {
    return std::chrono::duration<double>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
  }

  // ----- 50 Hz control loop -------------------------------------------------
  void OnControl()
  {
    const double now = SteadyNow();
    const State  cur = state_.load();

    RobotState rs{};
    const bool fresh = aimdk_io_->SnapshotState(rs);

    switch (cur) {
      case State::STANDBY: {
        // Touch the ready sentinel exactly once on the first STANDBY tick
        // so bash knows our subscribers/timers are live. ROS subscribers
        // are receiving state in the background while we sit here.
        if (!standby_ready_logged_) {
          standby_ready_logged_ = true;
          if (!cli_.ready_sentinel.empty()) {
            std::ofstream r(cli_.ready_sentinel, std::ios::trunc);
            if (!r) {
              RCLCPP_ERROR(node_->get_logger(),
                           "STANDBY: failed to touch ready-sentinel '%s' "
                           "(errno=%d). Bash may show its safety gate "
                           "without confirmation that deploy is ready.",
                           cli_.ready_sentinel.c_str(), errno);
            } else {
              r << "ready " << now << "\n";
              RCLCPP_WARN(node_->get_logger(),
                          "STANDBY: ready-sentinel touched at '%s'. "
                          "Writer is GATED OFF; waiting for trigger.",
                          cli_.ready_sentinel.c_str());
            }
          } else {
            RCLCPP_WARN(node_->get_logger(),
                        "STANDBY: ready (no ready-sentinel configured). "
                        "Writer is GATED OFF; waiting for trigger.");
          }
        }
        // Poll the trigger sentinel. std::ifstream open is the cheapest
        // portable existence check; no allocations on the steady-state
        // miss path.
        std::ifstream probe(cli_.start_trigger_sentinel);
        if (probe.good()) {
          RCLCPP_WARN(node_->get_logger(),
                      "STANDBY: start-trigger-sentinel '%s' detected; "
                      "advancing STANDBY -> INIT.",
                      cli_.start_trigger_sentinel.c_str());
          // Best-effort cleanup so a second invocation doesn't
          // immediately re-trigger.
          std::remove(cli_.start_trigger_sentinel.c_str());
          state_.store(State::INIT);
        }
        return;
      }
      case State::INIT: {
        if (aimdk_io_->AllStateFresh(0.5)) {
          // Latch the current observed joint pose as the safe-hold target
          // *before* WAIT_FOR_CONTROL begins. The 500 Hz writer will keep
          // publishing this latched command for the entire WAIT window
          // (and as the long-tail of SAFE_HOLD if we ever fall back).
          //
          // Capturing the actual pose -- rather than the constructor's
          // initial latch of target=default_angles, kp=0 -- means HAL
          // smoothly takes ownership at the operator's current pose with
          // no jolt:
          //   * Real robot: if MC has been driving firmware-stand (knees
          //     +28 deg, elbows -67 deg), the deploy now holds firmware-
          //     stand. When the operator stops MC and the deploy's
          //     commands actually take effect, there is no 33 deg elbow
          //     snap toward DEFAULT_DOF.
          //   * Sim --init-pose=gantry_hang / gantry_dangle / motion-RSI:
          //     the bridge has spawned the body at a non-default pose;
          //     the deploy now reads that pose from rs and latches it.
          //     The body stays put through the autostart window.
          //
          // The soft-start ramp at CONTROL still blends from default_angles
          // toward the policy output -- that is a separate concern and is
          // tracked in the deploy plan (Phase 4d follow-up). The fix here
          // only addresses the WAIT-time yank.
          if (fresh) {
            std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
            for (std::size_t i = 0; i < NUM_DOFS; ++i) {
              latest_cmd_.target_pos_mj[i] = rs.joint_pos_mj[i];
              latest_cmd_.stiffness_mj[i]  = cli_.dry_run ? 0.0 : kps[i];
              latest_cmd_.damping_mj[i]    = cli_.dry_run ? 0.0 : kds[i];
            }
            latest_cmd_.reason = "wait_for_control_hold";
          } else {
            RCLCPP_ERROR(
                node_->get_logger(),
                "INIT->WAIT advance fired but SnapshotState reported the "
                "state is stale; safe-hold latch left at constructor's "
                "passive default. Will retry on next tick if the state "
                "freshness re-asserts.");
            return;
          }
          state_.store(State::WAIT_FOR_CONTROL);
          control_entry_s_ = now;
          RCLCPP_INFO(node_->get_logger(),
                      "INIT -> WAIT_FOR_CONTROL (all state sources fresh; "
                      "safe-hold latched at current observed pose)");
        }
        return;
      }
      case State::WAIT_FOR_CONTROL: {
        if (autostart_target_s_ > 0.0 && now >= autostart_target_s_) {
          RCLCPP_WARN(node_->get_logger(),
                      "Autostart elapsed (%.2fs) -> CONTROL%s",
                      cli_.autostart_seconds,
                      cli_.dry_run ? " (DRY-RUN)" : "");
          ramp_.Reset();
          watchdog_.Reset();
          prop_buf_.Reset();
          last_action_il_.fill(0.0);
          control_entry_s_ = now;

          // Yaw-anchor the reference motion to the robot's current heading.
          // This is the C++ stand-in for IsaacLab's Reference State
          // Initialization (RSI): training teleports the simulated robot to
          // motion[0] every episode, so the policy never sees the motion's
          // absolute world yaw drift away from the robot's. On hardware the
          // robot is wherever the gantry left it, so without this call the
          // policy would see the recorded yaw of the .pkl file (often tens
          // of degrees off — e.g. ~96 deg for x2_ultra_idle_stand) as the
          // motion-anchor diff at t=0. See ReferenceMotion::Anchor() for
          // the full rationale and gear_sonic_deploy/scripts/
          // compare_deploy_vs_isaaclab_obs.py for the parity check that
          // surfaced the missing anchor.
          //
          // Only `fresh && AllStateFresh` could have made INIT advance us to
          // WAIT_FOR_CONTROL, so rs.base_quat_wxyz here is the same IMU
          // sample the operator sees on the robot — exactly what we want.
          if (fresh) {
            ref_motion_->Anchor(rs.base_quat_wxyz);
            RCLCPP_WARN(
                node_->get_logger(),
                "Reference motion '%s' yaw-anchored to robot heading "
                "(robot yaw = %.2f deg, applied Δyaw = %.2f deg). "
                "Motion pitch/roll left as recorded (gravity-grounded).",
                ref_motion_->Name().c_str(),
                yaw_from_quat_wxyz(rs.base_quat_wxyz) * 180.0 / 3.14159265358979323846,
                ref_motion_->yaw_anchor_delta() * 180.0 / 3.14159265358979323846);
          } else {
            RCLCPP_ERROR(
                node_->get_logger(),
                "WAIT->CONTROL fired on a STALE IMU sample; reference "
                "motion will NOT be yaw-anchored. Expect the motion-anchor "
                "diff to start far OOD. Aborting transition; will retry on "
                "next tick.");
            return;
          }

          state_.store(State::CONTROL);
        }
        return;
      }
      case State::SAFE_HOLD: {
        // Stay here forever; writer publishes the latched safe command.
        return;
      }
      case State::RAMP_OUT: {
        // Soft-EXIT ramp: linearly interpolate target_pos from the snapshot
        // we took on entering RAMP_OUT (last policy command) toward MC's
        // STAND_DEFAULT pose (or default_angles if --stand-default-pose was
        // not given) over cli_.return_seconds. We hold kp/kd at the
        // deploy-mode values the policy was trained with -- DO NOT lerp
        // kp/kd toward MC's stand-mode gains, even though we know them.
        //
        // Why: MC's STAND_DEFAULT gains have, per joint, very different
        // damping ratios than the deploy gains (e.g. elbow kp triples
        // 14 -> 50 while kd barely moves 0.9 -> 1.0; hip kd halves
        // 6.3 -> 4.0). Lerping kp/kd while the position target is also
        // moving by up to 1.2 rad over 2 s (elbows, when the policy left
        // arms extended) produces a transiently under-damped system
        // chasing a moving setpoint = motor whir + ringing. Verified the
        // hard way on 2026-05-03; reverted to deploy-mode gains here.
        // MC will switch to its own gains in a single message at the
        // takeover boundary, with the position already at the matching
        // pose, so the gain step is benign (low position error).
        // We always keep PD active -- never zero torque -- so the body
        // stays balanced through the whole ramp.
        // When the ramp completes we either:
        //   * --hold-for-mc-timeout-s > 0 -> transition to HOLD_FOR_MC
        //     and wait for MC to take back over the joint command bus;
        //   * otherwise -> request shutdown (legacy).
        const double T = std::max(cli_.return_seconds, 1e-6);
        const double t = now - ramp_out_entry_s_;
        const double alpha = std::clamp(t / T, 0.0, 1.0);  // 0=start, 1=done
        SafeCommand sc;
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          sc.target_pos_mj[i] = (1.0 - alpha) * ramp_out_start_pos_[i]
                                + alpha * stand_pose_target_[i];
          sc.stiffness_mj[i]  = cli_.dry_run ? 0.0 : kps[i];
          sc.damping_mj[i]    = cli_.dry_run ? 0.0 : kds[i];
        }
        sc.dry_run    = cli_.dry_run;
        sc.tilt_trip  = false;
        sc.ramp_alpha = 1.0 - alpha;  // for tick.csv: 1.0=just started, 0.0=done
        sc.reason     = "ramp_out";
        {
          std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
          latest_cmd_ = sc;
        }
        latest_cmd_ready_.store(true, std::memory_order_release);
        logger_.Log(now, rs.joint_pos_mj, rs.joint_vel_mj,
                    rs.base_quat_wxyz, rs.base_ang_vel,
                    last_action_il_, sc);
        if (alpha >= 1.0) {
          if (cli_.hold_for_mc_timeout_s > 0.0) {
            RCLCPP_WARN(node_->get_logger(),
                        "RAMP_OUT complete (%.2fs) -> HOLD_FOR_MC "
                        "(timeout %.1fs, waiting for MC to take back over the "
                        "joint command bus). Position now at MC STAND_DEFAULT "
                        "pose; HOLD_FOR_MC will step gains up to MC-stand "
                        "stiffness/damping for a firmer hold (legs, ankles, "
                        "waist). Position error at the gain step is ~0 so the "
                        "torque kick is negligible.",
                        cli_.return_seconds, cli_.hold_for_mc_timeout_s);
            EnterHoldForMc(now);
          } else {
            RCLCPP_WARN(node_->get_logger(),
                        "RAMP_OUT complete (%.2fs) -> shutting down. "
                        "Joints commanded back to STAND_DEFAULT pose; safe to "
                        "hand off to MC.", cli_.return_seconds);
            rclcpp::shutdown();
          }
        }
        return;
      }
      case State::HOLD_FOR_MC: {
        // Static-pose hold while MC restarts. We publish MC's STAND_DEFAULT
        // pose with MC-stand kp/kd (loaded from --stand-default-pose YAML;
        // falls back to deploy gains if the YAML wasn't supplied) so the
        // legs / ankles / waist are as stiff as MC-STAND would have them
        // (e.g. waist pitch kp 14 -> 40, knee kp 99 -> 150, ankle kp
        // 21 -> 30-40). This addresses the operator's "legs and waist
        // not holding" feedback during the post-policy hold: deploy-mode
        // gains were tuned for the *active* policy, not for a static
        // hold against gravity. The gain step at HOLD_FOR_MC entry is
        // safe because we just ramped position to stand_pose_target_,
        // so position error is ~0 and the torque kick from the kp step
        // is small. Note: deploy still has NO active balance controller
        // -- this hold is "joints stiff, body free to tilt" -- so a
        // strong perturbation can still tilt the torso. That's the
        // architectural ceiling until MC takes back over.
        // We exit when EITHER:
        //   (a) the MC-takeover detector callback (subscribed to
        //       /aima/hal/joint/leg/command and /aima/hal/joint/waist/command
        //       with ignore_local_publications=true) has seen its first
        //       message from a non-self publisher = MC is publishing again;
        //   (b) hold_for_mc_timeout_s elapses (MC didn't come back; we
        //       give up and let bash decide what to do next).
        SafeCommand sc;
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          sc.target_pos_mj[i] = stand_pose_target_[i];
          sc.stiffness_mj[i]  = cli_.dry_run ? 0.0 : stand_pose_stiffness_[i];
          sc.damping_mj[i]    = cli_.dry_run ? 0.0 : stand_pose_damping_[i];
        }
        sc.dry_run    = cli_.dry_run;
        sc.tilt_trip  = false;
        sc.ramp_alpha = 0.0;
        sc.reason     = "hold_for_mc";
        {
          std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
          latest_cmd_ = sc;
        }
        latest_cmd_ready_.store(true, std::memory_order_release);
        logger_.Log(now, rs.joint_pos_mj, rs.joint_vel_mj,
                    rs.base_quat_wxyz, rs.base_ang_vel,
                    last_action_il_, sc);
        // Exit policy (re-revised 2026-05-03 after operator observed
        // ~1.5 s of zero-torque after deploy released the bus -- the
        // FAST-EXIT-on-first-publish version had this bug because MC
        // publishes commands while still in PASSIVE_DEFAULT during boot):
        //
        //   PRIMARY GATE: --hold-for-mc-exit-sentinel.
        //     Bash touches the file after escalating MC all the way to
        //     STAND_DEFAULT. While the file is absent, deploy keeps
        //     publishing MC's stand pose with MC-stand gains -- so the
        //     legs/ankles/waist stay actively held throughout MC's
        //     PASSIVE -> JOINT -> STAND boot sequence. There IS a brief
        //     dual-publisher window once MC enters JOINT_DEFAULT (~1 s
        //     before STAND_DEFAULT activates), but both deploy and MC
        //     are PD-holding the same stand pose with similar gains, so
        //     the conflict is small and the robot stays under torque.
        //
        //   FALLBACK: first-MC-publish detection (legacy path, no
        //     exit-sentinel configured). Exits the moment the takeover
        //     detector fires. Use this only if you've arranged some
        //     other means of ensuring MC is in a holding mode (not
        //     PASSIVE) before its first publish.
        //
        //   BACKSTOP: hold_for_mc_timeout_s. Hard upper bound on the
        //     hold so deploy doesn't get stuck if bash crashes.
        //
        // First-publish detection is still LOGGED for tracing in both
        // modes -- it's the most useful timestamp for measuring the
        // STAND_DEFAULT settle latency from MC's side.
        const double held       = now - hold_for_mc_entry_s_;
        const bool   takeover   = mc_takeover_detected_.load(std::memory_order_acquire);
        if (takeover && !mc_takeover_logged_) {
          mc_takeover_logged_ = true;
          // ms-precision delta from HOLD_FOR_MC entry to actual DDS
          // callback firing. Captured inside the callback before any
          // locks, so it is independent of OnControl scheduling jitter.
          using ms_d = std::chrono::duration<double, std::milli>;
          double dt_ms_callback = -1.0;
          double dt_ms_now      = -1.0;
          {
            std::lock_guard<std::mutex> lk(mc_takeover_topic_mutex_);
            if (mc_takeover_steady_ts_.time_since_epoch().count() != 0
                && hold_for_mc_entry_steady_.time_since_epoch().count() != 0) {
              dt_ms_callback = ms_d(mc_takeover_steady_ts_
                                    - hold_for_mc_entry_steady_).count();
            }
            dt_ms_now = ms_d(std::chrono::steady_clock::now()
                             - hold_for_mc_entry_steady_).count();
          }
          if (cli_.hold_for_mc_exit_sentinel.empty()) {
            RCLCPP_WARN(node_->get_logger(),
                        "HOLD_FOR_MC: FIRST MC PUBLISH on '%s' at "
                        "callback=+%.3f ms, OnControl=+%.3f ms (entry+%.2fs). "
                        "Exiting -> MC takes the bus alone (no dual-publisher "
                        "fight). Total handoff latency = "
                        "callback->next OnControl tick (<= 20 ms).",
                        mc_takeover_topic_.c_str(),
                        dt_ms_callback, dt_ms_now, held);
          } else {
            RCLCPP_WARN(node_->get_logger(),
                        "HOLD_FOR_MC: FIRST MC PUBLISH on '%s' at "
                        "callback=+%.3f ms, OnControl=+%.3f ms (entry+%.2fs). "
                        "Continuing to publish STAND_DEFAULT pose (MC is "
                        "likely still in PASSIVE_DEFAULT during boot); "
                        "waiting for exit-sentinel to fire from bash after "
                        "MC reaches STAND_DEFAULT.",
                        mc_takeover_topic_.c_str(),
                        dt_ms_callback, dt_ms_now, held);
          }
        }

        // Exit policy:
        //   * If an exit-sentinel is configured, the sentinel is the ONLY
        //     gate. Deploy keeps publishing MC's STAND_DEFAULT pose until
        //     bash explicitly tells us "MC is in STAND_DEFAULT" by
        //     touching the file. This is the intended design from the
        //     original handoff plan: MC boots in PASSIVE_DEFAULT (zero
        //     torque), then escalates PASSIVE -> JOINT -> STAND. Its
        //     FIRST publish lands while it's still in PASSIVE; releasing
        //     the bus then would drop the robot to zero torque for the
        //     ~1-2 s it takes bash to escalate to STAND_DEFAULT (this
        //     was the bug the operator observed on 2026-05-03 20:33:33:
        //     "robot not in control for a couple of seconds before MC
        //     said switching to standing mode").
        //   * If no exit-sentinel was configured (legacy callers, or
        //     a future caller that arbitrates handoff via some other
        //     channel), fall back to first-MC-publish-detection.
        //   * The hold_for_mc_timeout_s cap is the ultimate backstop
        //     in either case.
        bool should_exit = false;
        const char* exit_reason = "";
        if (!cli_.hold_for_mc_exit_sentinel.empty()) {
          // Exit-sentinel mode (the path bash configures by default).
          // First-MC-publish detection is informational only here --
          // we logged it above but do NOT exit on it.
          std::ifstream probe(cli_.hold_for_mc_exit_sentinel);
          if (probe.good()) {
            should_exit = true;
            exit_reason = "exit-sentinel touched";
          }
        } else if (takeover) {
          should_exit = true;
          exit_reason = "first MC publish detected (no exit-sentinel configured)";
        }
        if (should_exit) {
          RCLCPP_WARN(node_->get_logger(),
                      "HOLD_FOR_MC: %s after %.2fs -> shutting down. "
                      "Robot stayed in STAND_DEFAULT pose throughout the "
                      "handoff (no DAMPING / PASSIVE window).",
                      exit_reason, held);
          ClearHoldForMcSentinel();
          rclcpp::shutdown();
          return;
        }
        if (held >= cli_.hold_for_mc_timeout_s) {
          RCLCPP_ERROR(node_->get_logger(),
                       "HOLD_FOR_MC: timed out after %.2fs without bash "
                       "creating the exit-sentinel. Shutting down anyway -- "
                       "the bus will go silent. Operator: confirm MC is "
                       "alive and in STAND_DEFAULT before re-enabling.",
                       held);
          ClearHoldForMcSentinel();
          rclcpp::shutdown();
        }
        return;
      }
      case State::CONTROL: break;
    }

    // Optional bounded-duration auto-shutdown. Triggered N seconds after we
    // entered CONTROL (control_entry_s_ is set in WAIT->CONTROL above). We
    // run this BEFORE the stale-state guard so a frozen robot still hits the
    // deadline -- the whole point of --max-duration is the operator can walk
    // away from a smoke test, and "stuck on stale state forever" defeats
    // that. Instead of shutting down immediately, transition to RAMP_OUT so
    // the joints get linearly returned to default_angles over
    // --return-seconds before MC takes back over. Set --return-seconds 0 to
    // get the legacy "shutdown immediately" behaviour.
    if (cli_.max_duration > 0.0 && (now - control_entry_s_) >= cli_.max_duration) {
      if (cli_.return_seconds > 0.0) {
        // Snapshot the current target as the ramp-out start pose and switch
        // states. Next OnControl tick will be served by the RAMP_OUT case
        // above.
        {
          std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
          ramp_out_start_pos_ = latest_cmd_.target_pos_mj;
        }
        ramp_out_entry_s_ = now;
        state_.store(State::RAMP_OUT);
        RCLCPP_WARN(node_->get_logger(),
                    "Max duration elapsed (%.2fs in CONTROL) -> RAMP_OUT "
                    "(%.2fs return-to-default)%s",
                    cli_.max_duration, cli_.return_seconds,
                    cli_.dry_run ? " (DRY-RUN)" : "");
      } else {
        RCLCPP_WARN(node_->get_logger(),
                    "Max duration elapsed (%.2fs in CONTROL) -> shutting down "
                    "(--return-seconds disabled)%s",
                    cli_.max_duration,
                    cli_.dry_run ? " (DRY-RUN)" : "");
        rclcpp::shutdown();
      }
      return;
    }

    if (!fresh) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 1000,
                           "CONTROL: stale or missing state, skipping tick");
      return;
    }

    // ---- Build observation -------------------------------------------------
    // 1. IL-remap measured joint pos/vel; subtract default for jpos_rel.
    std::array<double, NUM_DOFS> jpos_il{}, jvel_il{}, jpos_rel_il{};
    for (std::size_t il = 0; il < NUM_DOFS; ++il) {
      const std::size_t mj = static_cast<std::size_t>(isaaclab_to_mujoco[il]);
      jpos_il[il]     = rs.joint_pos_mj[mj];
      jvel_il[il]     = rs.joint_vel_mj[mj];
      jpos_rel_il[il] = jpos_il[il] - default_angles[mj];
    }
    // 2. Body-frame gravity from IMU quaternion.
    const auto grav = body_frame_gravity_from_quat_wxyz(rs.base_quat_wxyz);

    prop_buf_.Append(rs.base_ang_vel, jpos_rel_il, jvel_il, last_action_il_, grav);

    // 3. Tokenizer reference window.
    const double policy_time = now - control_entry_s_;
    const auto tok_obs = BuildTokenizerObs(*ref_motion_, policy_time, rs.base_quat_wxyz);
    const auto prop    = prop_buf_.GetFlat();

    // ---- Inference ---------------------------------------------------------
    std::array<double, NUM_DOFS> action_il;
    try {
      action_il = onnx_actor_->Infer(tok_obs, prop);
    } catch (const std::exception& e) {
      RCLCPP_FATAL(node_->get_logger(),
                   "ONNX inference failed: %s -> SAFE_HOLD", e.what());
      LatchSafeHold("onnx_failure");
      return;
    }

    // ---- Action clip (matches training-time ManagerEnvWrapper) -------------
    // IsaacLab's training wrapper applies torch.clip(env_actions, -C, C) with
    // C = config.action_clip_value (default 20.0) before stepping the sim.
    // The clipped value is then what env.action_manager.action returns, which
    // is what the ``last_action_wo_hand`` proprioception term sees on the
    // next tick. Deploying without this clip means: (a) target_pos_mj can
    // explode by O(C * action_scale) when the policy saturates, which the
    // ``--max-target-dev`` clamp truncates -- silently breaking the
    // physics<->command relationship -- and (b) ``last_action_il_`` drifts
    // outside the training distribution, accelerating divergence. See the
    // 16k checkpoint smoke test on 2026-04-22 for the gory details.
    std::size_t clipped_joint_count = 0;
    double max_pre_clip = 0.0;
    if (cli_.action_clip > 0.0) {
      const double clip = cli_.action_clip;
      for (std::size_t i = 0; i < NUM_DOFS; ++i) {
        const double a = action_il[i];
        if (std::abs(a) > max_pre_clip) max_pre_clip = std::abs(a);
        if (a >  clip) { action_il[i] =  clip; ++clipped_joint_count; }
        else if (a < -clip) { action_il[i] = -clip; ++clipped_joint_count; }
      }
      if (clipped_joint_count > 0) {
        ++action_clip_tick_count_;
        action_clip_max_pre_clip_ = std::max(action_clip_max_pre_clip_, max_pre_clip);
      }
    }
    last_action_il_ = action_il;

    // ---- Optional one-shot obs dump for offline parity-check ---------------
    // Dumps tokenizer, proprioception, raw policy output, and the robot state
    // that produced them on the first CONTROL tick, then asks the node to
    // shut down. The companion script
    //   gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py
    // diffs this against /tmp/x2_step0_isaaclab_lastpt.pt.
    if (!cli_.obs_dump_path.empty() && !obs_dumped_) {
      DumpObsBlob(cli_.obs_dump_path, tok_obs, prop, action_il, rs, policy_time);
      obs_dumped_ = true;
      RCLCPP_WARN(node_->get_logger(),
                  "--obs-dump fired; wrote %zu bytes to %s; requesting shutdown.",
                  static_cast<std::size_t>(
                    tok_obs.size() * sizeof(float)
                    + prop.size() * sizeof(float)
                    + NUM_DOFS * sizeof(double)),
                  cli_.obs_dump_path.c_str());
      // Returning here means we don't push any SafeCommand for this tick;
      // the writer keeps repeating the previous one (which in --dry-run mode
      // has zero gains, so the robot stays passive). We then ask the node
      // to shut down at the next executor turn.
      rclcpp::shutdown();
      return;
    }

    // ---- Action -> MJ-ordered PD target ------------------------------------
    std::array<double, NUM_DOFS> target_pos_mj{};
    for (std::size_t mj = 0; mj < NUM_DOFS; ++mj) {
      const std::size_t il = static_cast<std::size_t>(mujoco_to_isaaclab[mj]);
      target_pos_mj[mj] = default_angles[mj] + action_il[il] * x2_action_scale[mj];
    }

    // ---- Safety stack ------------------------------------------------------
    SafeCommand sc = ApplySafetyStack(target_pos_mj, grav[2],
                                      ramp_, watchdog_, cli_.dry_run, now,
                                      cli_.max_target_dev);

    // ---- Output-side target LPF (real-deploy only; bypassed by default) ----
    // The EMA runs strictly AFTER the safety stack, so:
    //   * --max-target-dev clamps still bound the PRE-filter target, and the
    //     filter then attenuates further (cannot make the published target
    //     exceed the clamp);
    //   * --obs-dump returned earlier in this tick (line ~672 above) before
    //     `target_pos_mj` was even computed, so the dumped raw policy output
    //     is identical with or without --target-lpf-hz set;
    //   * RAMP_OUT and SAFE_HOLD bypass: those states already produce a
    //     deliberately-shaped trajectory we don't want to attenuate.
    if (target_lpf_alpha_ > 0.0
        && state_.load() == State::CONTROL
        && !sc.tilt_trip) {
      if (!target_lpf_initialized_) {
        // First CONTROL tick: seed the EMA state to the current target so
        // we don't bias toward zero or any prior value.
        target_lpf_state_ = sc.target_pos_mj;
        target_lpf_initialized_ = true;
      } else {
        const double a = target_lpf_alpha_;
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          target_lpf_state_[i] =
              a * sc.target_pos_mj[i] + (1.0 - a) * target_lpf_state_[i];
        }
        sc.target_pos_mj = target_lpf_state_;
      }
    }

    if (sc.tilt_trip && state_.load() == State::CONTROL) {
      RCLCPP_FATAL(node_->get_logger(), "%s -> SAFE_HOLD", sc.reason.c_str());
      // Latch the safety command so the writer keeps holding it forever.
      {
        std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
        latest_cmd_ = sc;
      }
      latest_cmd_ready_.store(true, std::memory_order_release);
      state_.store(State::SAFE_HOLD);
      return;
    }

    // ---- Publish to writer slot --------------------------------------------
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      latest_cmd_ = sc;
    }
    latest_cmd_ready_.store(true, std::memory_order_release);

    // ---- Log ---------------------------------------------------------------
    logger_.Log(now, rs.joint_pos_mj, rs.joint_vel_mj,
                rs.base_quat_wxyz, rs.base_ang_vel,
                action_il, sc);

    // ---- Periodic status ---------------------------------------------------
    if (++control_tick_ % 50 == 0) {
      RCLCPP_INFO(node_->get_logger(),
                  "CONTROL tick=%lu policy_t=%.2fs alpha=%.2f grav_z=%+.2f "
                  "act_clip_ticks=%lu max_pre_clip=%.2f",
                  static_cast<unsigned long>(control_tick_),
                  policy_time, sc.ramp_alpha, grav[2],
                  static_cast<unsigned long>(action_clip_tick_count_),
                  action_clip_max_pre_clip_);
    }
  }

  // ----- 500 Hz writer loop ------------------------------------------------
  void OnWriter()
  {
    const State cur = state_.load();
    if (cur == State::STANDBY
        || cur == State::INIT
        || cur == State::WAIT_FOR_CONTROL) {
      // Don't publish anything in pre-control states. The robot's last-good
      // command (from whatever was running before us) keeps the joints held.
      // STANDBY in particular MUST be silent: bash launches us before
      // stop_app, so MC may still be publishing on this bus -- adding our
      // commands would make the firmware see a dual-publisher fight.
      return;
    }
    // Skip publishing until OnControl (or RAMP_OUT/SAFE_HOLD) has latched
    // a real command. Without this guard, the first ~15 ms after WAIT ->
    // CONTROL would publish a default-zero command (kp=0, kd=0, target=0),
    // which on the sim bridge cancels the pre-handoff freeze prematurely
    // and lets gravity perturb joint state before the policy ever sees it.
    // On hardware the symptom would be different (a no-op zero-gain PD
    // command) but it's still wrong: MC's last-good command is what should
    // hold the joints during this sub-tick window, not our zeroed one.
    if (!latest_cmd_ready_.load(std::memory_order_acquire)) {
      return;
    }
    SafeCommand sc;
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      sc = latest_cmd_;
    }
    // target_vel = zeros (the policy spec is pos-only; firmware integrates).
    static const std::array<double, NUM_DOFS> kZeroVel{};
    aimdk_io_->PublishCommand(sc.target_pos_mj, kZeroVel,
                              sc.stiffness_mj, sc.damping_mj);
  }

  void LatchSafeHold(const std::string& reason)
  {
    SafeCommand sc;
    for (std::size_t i = 0; i < NUM_DOFS; ++i) {
      sc.target_pos_mj[i] = default_angles[i];
      sc.stiffness_mj[i]  = cli_.dry_run ? 0.0 : kps[i];
      sc.damping_mj[i]    = cli_.dry_run ? 0.0 : kds[i] * 4.0;
    }
    sc.dry_run    = cli_.dry_run;
    sc.tilt_trip  = false;
    sc.ramp_alpha = 0.0;
    sc.reason     = reason;
    {
      std::lock_guard<std::mutex> lk(latest_cmd_mutex_);
      latest_cmd_ = sc;
    }
    latest_cmd_ready_.store(true, std::memory_order_release);
    state_.store(State::SAFE_HOLD);
  }

  // ─── HOLD_FOR_MC support ────────────────────────────────────────────
  // RAMP_OUT calls EnterHoldForMc() once the lerp completes.
  void EnterHoldForMc(double now)
  {
    hold_for_mc_entry_s_ = now;
    hold_for_mc_entry_steady_ = std::chrono::steady_clock::now();
    mc_takeover_detected_.store(false, std::memory_order_release);
    mc_takeover_logged_ = false;
    {
      std::lock_guard<std::mutex> lk(mc_takeover_topic_mutex_);
      mc_takeover_topic_.clear();
      mc_takeover_steady_ts_ = std::chrono::steady_clock::time_point{};
    }
    if (!cli_.hold_for_mc_sentinel.empty()) {
      // Touch the sentinel so deploy_x2.sh knows the policy phase is
      // done and it can fire start_app + SetMcAction(STAND_DEFAULT).
      // Use std::ofstream so we don't require <unistd.h>; trunc-creates
      // an empty file on every entry.
      std::ofstream sentinel(cli_.hold_for_mc_sentinel, std::ios::trunc);
      if (!sentinel) {
        RCLCPP_ERROR(node_->get_logger(),
                     "HOLD_FOR_MC: failed to touch sentinel '%s' (errno "
                     "= %d). Bash won't sequence start_app; operator "
                     "will need to bring MC back manually.",
                     cli_.hold_for_mc_sentinel.c_str(), errno);
      } else {
        sentinel << "hold_for_mc " << now << "\n";
      }
    }
    state_.store(State::HOLD_FOR_MC);
  }

  void ClearHoldForMcSentinel()
  {
    if (cli_.hold_for_mc_sentinel.empty()) return;
    // Best-effort cleanup; if remove fails there's nothing meaningful
    // we can do (the bash cleanup trap also rm -f's it as a backstop).
    std::remove(cli_.hold_for_mc_sentinel.c_str());
  }

  // Called from the constructor when --hold-for-mc-timeout-s > 0.
  // Subscribes to /aima/hal/joint/leg/command and /aima/hal/joint/waist/command
  // with ignore_local_publications=true, so OUR command writer's
  // traffic never trips the detector. The first non-self message on
  // either topic flips mc_takeover_detected_, which the HOLD_FOR_MC
  // case in OnControl reads on the next tick to exit cleanly.
  void InitMcTakeoverDetectors()
  {
    rclcpp::SubscriptionOptions opts;
    opts.ignore_local_publications = true;
    // Match the QoS of the joint command bus exactly. AimdkIo publishes
    // on these topics with rclcpp::SensorDataQoS() (best-effort, depth=10);
    // MC's HAL on PC1 also publishes best-effort. A default rclcpp::QoS()
    // here is RELIABLE, which is INCOMPATIBLE with MC's best-effort
    // publisher -- DDS refuses to match and our subscriber receives zero
    // messages from MC. That is the root cause of the ~1.7 s dual-
    // publisher whir at the end of the 2026-05-03 run: the takeover
    // detector never fired, deploy fell back to the slower exit-sentinel
    // path, and MC + deploy fought on the bus for the full duration of
    // bash's PASSIVE -> JOINT -> STAND escalation. With matched QoS the
    // detector callback fires sub-ms after MC's first publish.
    // ignore_local_publications still suppresses self-traffic at the
    // GID level, independent of QoS.
    auto qos = rclcpp::SensorDataQoS();
    auto on_takeover =
        [this](const std::string& topic) {
          // Capture the moment MC took the bus at the highest resolution
          // we can, BEFORE acquiring any locks, so the timestamp is as
          // close as possible to "DDS callback fired". steady_clock is
          // monotonic + ns-precision; we publish the delta vs HOLD_FOR_MC
          // entry so the operator can correlate against audible whirring.
          const auto now_steady = std::chrono::steady_clock::now();
          bool first = false;
          {
            std::lock_guard<std::mutex> lk(mc_takeover_topic_mutex_);
            if (mc_takeover_topic_.empty()) {
              mc_takeover_topic_ = topic;
              mc_takeover_steady_ts_ = now_steady;
              first = true;
            }
          }
          mc_takeover_detected_.store(true, std::memory_order_release);
          // Touch the early-signal sentinel ONCE on the first publish so
          // bash can start the JOINT_DEFAULT escalation immediately rather
          // than polling MC's mode service (which lags MC's actual first
          // publish by ~0.5-0.8 s). The std::ofstream truncate is fast (a
          // couple of ms even on EXT4); we do it inside the DDS callback
          // because the latency from this point to bash seeing the file
          // matters: every ms we shave here is a ms less of MC-PASSIVE +
          // deploy dual-publisher whir. Best-effort -- if the open fails
          // (e.g. dir missing), bash falls back to the mc_get_action poll
          // path automatically.
          if (first && !cli_.mc_first_publish_sentinel.empty()) {
            std::ofstream s(cli_.mc_first_publish_sentinel, std::ios::trunc);
            if (s) s << "first_publish " << topic << "\n";
          }
        };
    mc_takeover_leg_sub_ =
        node_->create_subscription<aimdk_msgs::msg::JointCommandArray>(
            "/aima/hal/joint/leg/command", qos,
            [on_takeover](aimdk_msgs::msg::JointCommandArray::ConstSharedPtr) {
              on_takeover("/aima/hal/joint/leg/command");
            },
            opts);
    mc_takeover_waist_sub_ =
        node_->create_subscription<aimdk_msgs::msg::JointCommandArray>(
            "/aima/hal/joint/waist/command", qos,
            [on_takeover](aimdk_msgs::msg::JointCommandArray::ConstSharedPtr) {
              on_takeover("/aima/hal/joint/waist/command");
            },
            opts);
    RCLCPP_INFO(node_->get_logger(),
                "HANDOFF: MC-takeover detectors armed on "
                "/aima/hal/joint/leg/command and "
                "/aima/hal/joint/waist/command "
                "(ignore_local_publications=true).");
  }

  // -------------------------------------------------------------------------
  rclcpp::Node::SharedPtr node_;
  CliArgs                 cli_;

  std::unique_ptr<AimdkIo>          aimdk_io_;
  std::unique_ptr<ReferenceMotion>  ref_motion_;
  std::unique_ptr<OnnxActor>        onnx_actor_;
  ProprioceptionBuffer              prop_buf_;
  std::array<double, NUM_DOFS>      last_action_il_{};

  SoftStartRamp                     ramp_;
  TiltWatchdog                      watchdog_;
  DeployLogger                      logger_;

  rclcpp::TimerBase::SharedPtr      control_timer_;
  rclcpp::TimerBase::SharedPtr      writer_timer_;

  std::mutex                        latest_cmd_mutex_;
  SafeCommand                       latest_cmd_;
  // True once OnControl has populated latest_cmd_ at least once (or
  // RAMP_OUT/SAFE_HOLD has explicitly latched one). Until this flips,
  // OnWriter must NOT publish, otherwise the bridge sees a default-zero
  // JointCommand (kp=0, kd=0, target=0), which (a) silently flips its
  // ``_first_command_received`` flag and unfreezes physics, and (b)
  // perturbs joint state in the ~15 ms between WAIT->CONTROL and the first
  // OnControl tick. That produced an apples-to-oranges first-tick obs
  // (e.g. cpp jvel[5] = -0.0077 vs python jvel[5] = -0.0265 for the same
  // motion frame) and was the source of the parity-profile fall-at-~5 s.
  // See gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py:_sim_step_once
  // for the freeze counterpart on the bridge side.
  std::atomic<bool>                 latest_cmd_ready_{false};

  std::atomic<State>                state_{State::INIT};
  // Latched true the first time OnControl runs in STANDBY -- prevents
  // the ready-sentinel from being touched + logged on every tick.
  bool                              standby_ready_logged_ = false;
  double                            control_entry_s_     = -1.0;
  double                            autostart_target_s_  = -1.0;
  std::uint64_t                     control_tick_        = 0;

  // RAMP_OUT bookkeeping: ramp_out_entry_s_ is the steady-clock time we
  // entered RAMP_OUT, and ramp_out_start_pos_ is the target_pos_mj snapshot
  // we took at that moment. The RAMP_OUT case in OnControl lerps from
  // ramp_out_start_pos_ -> stand_pose_target_ over cli_.return_seconds.
  double                            ramp_out_entry_s_    = -1.0;
  std::array<double, NUM_DOFS>      ramp_out_start_pos_{};

  // End-of-run handoff target (MC's STAND_DEFAULT pose). Initialised in
  // the constructor: defaults to default_angles + deploy-mode kp/kd, and
  // overwritten with the captured YAML values when --stand-default-pose
  // is provided. Used by RAMP_OUT (lerp toward) and HOLD_FOR_MC (publish
  // statically).
  std::array<double, NUM_DOFS>      stand_pose_target_{};
  std::array<double, NUM_DOFS>      stand_pose_stiffness_{};
  std::array<double, NUM_DOFS>      stand_pose_damping_{};

  // HOLD_FOR_MC bookkeeping. hold_for_mc_entry_s_ is the steady-clock
  // time we entered the state. mc_takeover_detected_ is flipped by the
  // detection subscribers below the *moment* an external publisher (MC)
  // shows up on /aima/hal/joint/{leg,waist}/command -- we then exit
  // cleanly on the next OnControl tick. mc_takeover_topic_ records which
  // topic saw the first non-self publish, purely for the log line.
  // hold_for_mc_entry_steady_ pairs with mc_takeover_steady_ts_ to
  // produce the ms-precision delta in the takeover log line, independent
  // of the ROS-clock rounding in hold_for_mc_entry_s_.
  double                            hold_for_mc_entry_s_ = -1.0;
  std::chrono::steady_clock::time_point hold_for_mc_entry_steady_{};
  std::atomic<bool>                 mc_takeover_detected_{false};
  // True after we've logged the "first MC publish observed" line once,
  // so we don't spam the log on every tick once MC is back on the bus.
  bool                              mc_takeover_logged_  = false;
  std::string                       mc_takeover_topic_;
  // ns-precision timestamp captured INSIDE the DDS callback (before any
  // locks), so the trace shows the actual moment MC took the bus rather
  // than the next OnControl tick (which can lag up to ~20 ms at 50 Hz).
  std::chrono::steady_clock::time_point mc_takeover_steady_ts_{};
  std::mutex                        mc_takeover_topic_mutex_;
  // Subscribers carry SubscriptionOptions::ignore_local_publications=true
  // so OUR command writer's traffic does not trip the detector. Kept as
  // members so they outlive the constructor.
  rclcpp::Subscription<aimdk_msgs::msg::JointCommandArray>::SharedPtr
      mc_takeover_leg_sub_;
  rclcpp::Subscription<aimdk_msgs::msg::JointCommandArray>::SharedPtr
      mc_takeover_waist_sub_;

  // Output-side target LPF state. target_lpf_alpha_ is computed once in
  // Run() from cli_.target_lpf_hz at the OnControl rate (50 Hz). When alpha
  // is zero the filter is fully bypassed -- no math, no allocations, and
  // the published target is the unmodified safety-clamped policy output.
  // target_lpf_initialized_ is reset implicitly by being default-false at
  // node startup; we also re-seed on the FIRST CONTROL tick (see OnControl)
  // so post-RAMP_OUT/SAFE_HOLD restarts (if we ever support them) behave.
  double                            target_lpf_alpha_         = 0.0;
  bool                              target_lpf_initialized_   = false;
  std::array<double, NUM_DOFS>      target_lpf_state_{};

  // Cumulative count of ticks where at least one joint hit the
  // ``--action-clip`` symmetric limit, plus the largest |action_il| we've
  // seen pre-clip across the whole run. Reported on the periodic status
  // line so the operator can tell at a glance whether the policy is
  // saturated (large numbers = bad: see action_clip explanation above).
  std::uint64_t                     action_clip_tick_count_   = 0;
  double                            action_clip_max_pre_clip_ = 0.0;

  // Set after --obs-dump fires so we don't accidentally dump a second time
  // if the executor manages to schedule another OnControl before shutdown.
  bool                              obs_dumped_          = false;

  // Write the first-tick inference payload to PATH as a binary blob, then
  // request shutdown. Layout (little-endian, no padding):
  //
  //   magic        : char[8]  = "X2OBSV01"
  //   tok_dim      : uint32_t = 680
  //   prop_dim     : uint32_t = 990
  //   action_dim   : uint32_t = 31
  //   policy_time  : float64
  //   tokenizer_obs: float32[tok_dim]
  //   proprioception: float32[prop_dim]
  //   action_il    : float64[action_dim]
  //   joint_pos_mj : float64[31]
  //   joint_vel_mj : float64[31]
  //   base_quat_wxyz: float64[4]
  //   base_ang_vel : float64[3]
  //
  // Total: 8 + 12 + 8 + (680+990)*4 + (31+31+31+4+3)*8 = 7508 bytes.
  // The companion script
  //   gear_sonic_deploy/scripts/compare_deploy_vs_isaaclab_obs.py
  // reads this and diffs each named slot against
  // /tmp/x2_step0_isaaclab_lastpt.pt.
  void DumpObsBlob(const std::string&                       path,
                   const std::vector<float>&                tok_obs,
                   const std::vector<float>&                prop,
                   const std::array<double, NUM_DOFS>&      action_il,
                   const RobotState&                        rs,
                   double                                   policy_time)
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
      RCLCPP_ERROR(node_->get_logger(),
                   "obs-dump: failed to open %s for writing", path.c_str());
      return;
    }
    constexpr char kMagic[8] = {'X','2','O','B','S','V','0','1'};
    const std::uint32_t tok_dim    = static_cast<std::uint32_t>(tok_obs.size());
    const std::uint32_t prop_dim   = static_cast<std::uint32_t>(prop.size());
    const std::uint32_t action_dim = static_cast<std::uint32_t>(NUM_DOFS);

    out.write(kMagic, sizeof(kMagic));
    out.write(reinterpret_cast<const char*>(&tok_dim),    sizeof(tok_dim));
    out.write(reinterpret_cast<const char*>(&prop_dim),   sizeof(prop_dim));
    out.write(reinterpret_cast<const char*>(&action_dim), sizeof(action_dim));
    out.write(reinterpret_cast<const char*>(&policy_time), sizeof(policy_time));

    out.write(reinterpret_cast<const char*>(tok_obs.data()),
              tok_obs.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(prop.data()),
              prop.size() * sizeof(float));
    out.write(reinterpret_cast<const char*>(action_il.data()),
              action_il.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(rs.joint_pos_mj.data()),
              rs.joint_pos_mj.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(rs.joint_vel_mj.data()),
              rs.joint_vel_mj.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(rs.base_quat_wxyz.data()),
              rs.base_quat_wxyz.size() * sizeof(double));
    out.write(reinterpret_cast<const char*>(rs.base_ang_vel.data()),
              rs.base_ang_vel.size() * sizeof(double));
  }
};

}  // namespace agi_x2

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
  using namespace agi_x2;

  rclcpp::init(argc, argv);

  CliArgs cli;
  try {
    cli = ParseCli(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << "Argument error: " << e.what() << "\n\n";
    PrintUsage();
    rclcpp::shutdown();
    return 2;
  }

  auto node = rclcpp::Node::make_shared("x2_deploy_onnx_ref");
  const std::string banner_dry  = cli.dry_run ? " [DRY-RUN]" : "";
  const std::string banner_auto = cli.autostart_seconds >= 0
      ? (" autostart=" + std::to_string(cli.autostart_seconds) + "s")
      : std::string(" (operator-go required: type 'go' on stdin)");
  const std::string banner_dur  = cli.max_duration > 0
      ? (" max-duration=" + std::to_string(cli.max_duration) + "s")
      : std::string("");
  RCLCPP_INFO(node->get_logger(),
              "x2_deploy_onnx_ref starting%s%s%s",
              banner_dry.c_str(), banner_auto.c_str(), banner_dur.c_str());

  std::unique_ptr<X2Deploy> deploy;
  try {
    deploy = std::make_unique<X2Deploy>(node, cli);
  } catch (const std::exception& e) {
    RCLCPP_FATAL(node->get_logger(), "Initialization failed: %s", e.what());
    rclcpp::shutdown();
    return 3;
  }

  // Operator-go gate: when --autostart-after is not used, read a single line from
  // stdin to advance from WAIT_FOR_CONTROL to CONTROL. We do this in a
  // dedicated thread so it doesn't block the executor.
  std::thread operator_thread;
  if (cli.autostart_seconds < 0.0) {
    operator_thread = std::thread([&deploy]() {
      std::cout << "\n[operator] Type 'go' + Enter to enter CONTROL state.\n"
                   "          (or Ctrl-C to abort)\n"
                << std::flush;
      std::string line;
      while (rclcpp::ok() && std::getline(std::cin, line)) {
        if (line == "go") {
          deploy->RequestGo();
          break;
        }
        std::cout << "[operator] (ignored '" << line
                  << "'; type 'go' to start)\n" << std::flush;
      }
    });
  }

  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  if (operator_thread.joinable()) operator_thread.join();
  rclcpp::shutdown();
  return 0;
}
