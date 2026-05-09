/**
 * @file zmq_pose_input_source.hpp
 * @brief ReferenceMotion drop-in that pulls 31-DOF body refs from a ZMQ feed.
 *
 * This is the M2 counterpart to ``StandStillReference`` and ``PklMotionReference``:
 * the X2 deploy harness consumes per-frame ``ReferenceFrame``s through the
 * abstract ``ReferenceMotion`` interface, and this class implements that
 * interface by subscribing to a ZMQ ``pose`` topic published by the VLA
 * Python process (or, during M2 bring-up, the
 * ``gear_sonic/scripts/mock_vla_publish_stand_token.py`` helper).
 *
 * ## Wire format (subset relevant to v0)
 *
 *   metadata: { v: 4, endian: "le", count: 1 }
 *   data:
 *     joint_pos_mj:   float32[31]   body reference pose, MuJoCo URDF order
 *     root_quat_xyzw: float32[4]    base link orientation, scipy convention
 *     left_hand_joints:  float32[10]  passthrough to AimDK HAL (not consumed
 *                                     by the policy ONNX -- the body-only
 *                                     decoder doesn't see fingers)
 *     right_hand_joints: float32[10]
 *     motion_token:   float32[64]   v1 hook for VLA-direct token streaming;
 *                                   currently logged but otherwise unused
 *                                   (see future-work note in
 *                                   docs/source/tutorials/vla_training.md).
 *     frame_index:    int64[1]      monotonic VLA tick counter
 *
 * Compatibility with the existing wire format
 * -------------------------------------------
 *
 * The Python helper ``pack_pose_message`` accepts a free-form payload dict;
 * this class consumes whichever fields are present and falls back to the
 * last-good cached value when an expected field is missing. That keeps M2
 * bring-up incremental:
 *
 *   - The token-only mock (``mock_vla_publish_stand_token.py``) keeps
 *     working -- when ``joint_pos_mj`` is absent, ``Sample(time)`` returns
 *     the trained ``default_angles`` stand pose.
 *   - The refframe mock (``mock_vla_publish_refframe.py``, added with this
 *     PR) drives the body to a real reference trajectory.
 *
 * ## Threading
 *
 * The subscriber runs a background thread (provided by
 * ``ZMQPackedMessageSubscriber::Start``); decoded payloads are copied into
 * ``cache_`` under ``cache_mutex_``. ``Sample(time)`` is called from the
 * 50 Hz deploy control thread and locks the same mutex for the duration
 * of a single struct copy (sub-microsecond on the X2 dev box).
 *
 * Joint velocity is reconstructed from the cached frames via finite
 * difference -- mirrors what ``PklMotionReference::Sample`` already does
 * (the file format also lacks velocity).
 */

#ifndef AGI_X2_ZMQ_POSE_INPUT_SOURCE_HPP
#define AGI_X2_ZMQ_POSE_INPUT_SOURCE_HPP

#include "policy_parameters.hpp"
#include "reference_motion.hpp"
#include "zmq/zmq_packed_message_subscriber.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace agi_x2 {

/// Hand-joint count exposed alongside the body reference. Defaults to the
/// full 10-DOF AgiBot OmniHand layout.
constexpr std::size_t DEFAULT_HAND_DOF_PER_SIDE = 10;

/// Latest hand-joint snapshot pulled from the ZMQ feed. Held as a fixed-size
/// vector so callers can pass it straight to ``aimdk_io`` without bounds
/// gymnastics. ``valid`` flips to ``true`` after the first successful decode;
/// callers that haven't seen a frame yet should fall back to "fingers open"
/// (zeros) as the deploy harness already does.
struct ZmqHandJointsSnapshot {
  std::array<double, DEFAULT_HAND_DOF_PER_SIDE> left{};
  std::array<double, DEFAULT_HAND_DOF_PER_SIDE> right{};
  bool   valid{false};
  /// Monotonic VLA tick that produced this snapshot.
  int64_t frame_index{0};
};

/**
 * @class ZmqPoseInputSource
 * @brief ReferenceMotion source that consumes body refs from a ZMQ ``pose`` topic.
 *
 * Construct via ``Connect``, then pass the returned ``unique_ptr`` to
 * ``X2Deploy`` exactly as you would a ``PklMotionReference``. Hand
 * snapshots are read out-of-band via ``LatestHandJoints()``.
 */
class ZmqPoseInputSource : public ReferenceMotion {
 public:
  /// Connect to ``host:port`` and start the background receive thread.
  /// Throws std::runtime_error on connect failure.
  static std::unique_ptr<ZmqPoseInputSource> Connect(
      const std::string& host,
      int                port,
      const std::string& topic = "pose",
      int                receive_timeout_ms = 200);

  ZmqPoseInputSource(const ZmqPoseInputSource&) = delete;
  ZmqPoseInputSource& operator=(const ZmqPoseInputSource&) = delete;

  ~ZmqPoseInputSource() override;

  // ---- ReferenceMotion interface -----------------------------------------
  ReferenceFrame Sample(double time) const override;
  std::string    Name() const override { return "zmq_pose"; }

  /// No-op: ZMQ frames are pre-anchored to whatever convention the VLA emits.
  /// Yaw alignment, if needed, lives on the publisher side (the VLA already
  /// has access to the robot's current heading via the ``x2_debug`` echo).
  void Anchor(const std::array<double, 4>& /*robot_quat_wxyz*/) override {}

  // ---- Side channel: hand joints + diagnostics ---------------------------

  /// Snapshot of the most recent hand-joint targets. Thread-safe.
  ZmqHandJointsSnapshot LatestHandJoints() const;

  /// Total decoded messages since Connect(). Useful for "no VLA frames yet"
  /// safety holds. Thread-safe.
  int64_t total_frames_received() const noexcept {
    return total_frames_received_.load(std::memory_order_acquire);
  }

  /// Whether at least one ``joint_pos_mj``-bearing frame has arrived. Until
  /// this returns true ``Sample()`` falls back to ``default_angles``.
  bool has_body_reference() const noexcept {
    return has_body_reference_.load(std::memory_order_acquire);
  }

 private:
  ZmqPoseInputSource(const std::string& host, int port, const std::string& topic);

  void HandleDecoded(
      const std::string& topic,
      const ZMQPackedMessageSubscriber::DecodedHeader& header,
      const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers);

  // Helpers for typed reads out of the BufferView descriptors.
  static bool CopyFloat32IntoDouble(
      const ZMQPackedMessageSubscriber::FieldInfo& field,
      const ZMQPackedMessageSubscriber::BufferView& buffer,
      double* out, std::size_t expected_count);

  static bool CopyInt64Scalar(
      const ZMQPackedMessageSubscriber::FieldInfo& field,
      const ZMQPackedMessageSubscriber::BufferView& buffer,
      int64_t* out);

  std::unique_ptr<ZMQPackedMessageSubscriber> subscriber_;

  mutable std::mutex          cache_mutex_;
  ReferenceFrame              latest_frame_{};
  ReferenceFrame              previous_frame_{};
  std::chrono::steady_clock::time_point latest_recv_{std::chrono::steady_clock::time_point::min()};
  std::chrono::steady_clock::time_point previous_recv_{std::chrono::steady_clock::time_point::min()};
  ZmqHandJointsSnapshot       latest_hand_{};
  std::array<double, 64>      latest_motion_token_{};

  std::atomic<int64_t>        total_frames_received_{0};
  std::atomic<bool>           has_body_reference_{false};
};

}  // namespace agi_x2

#endif  // AGI_X2_ZMQ_POSE_INPUT_SOURCE_HPP
