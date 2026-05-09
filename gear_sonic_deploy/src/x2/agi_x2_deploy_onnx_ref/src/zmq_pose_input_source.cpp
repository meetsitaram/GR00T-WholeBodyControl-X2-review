#include "zmq/zmq_pose_input_source.hpp"
// Pull the publisher header into the same TU so its compile errors surface
// in the offline syntax-check build (the ZmqDebugPublisher itself lives
// header-only; without this include it would only be checked against
// x2_deploy_onnx_ref.cpp during the full ROS 2 ament build).
#include "zmq/zmq_debug_publisher.hpp"  // NOLINT(misc-include-cleaner)

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace agi_x2 {

namespace {

/// Per-field byte-swap helper. Walks ``count`` little-endian floats sitting
/// at ``src`` and writes their host-byte-order values into ``dst`` (which is
/// allowed to alias ``src`` -- swap is in place when they match).
void DecodeF32Array(const void* src, std::size_t count, double* dst, bool needs_swap)
{
  const auto* in = static_cast<const float*>(src);
  for (std::size_t i = 0; i < count; ++i) {
    float v = in[i];
    if (needs_swap) {
      v = byte_swap(v);
    }
    dst[i] = static_cast<double>(v);
  }
}

void DecodeF32ArrayInto(const void* src, std::size_t count, double* dst, bool needs_swap)
{
  DecodeF32Array(src, count, dst, needs_swap);
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

std::unique_ptr<ZmqPoseInputSource> ZmqPoseInputSource::Connect(
    const std::string& host,
    int                port,
    const std::string& topic,
    int                receive_timeout_ms)
{
  auto source = std::unique_ptr<ZmqPoseInputSource>(
      new ZmqPoseInputSource(host, port, topic));
  source->subscriber_ = std::make_unique<ZMQPackedMessageSubscriber>(
      host, port, topic, receive_timeout_ms,
      /*verbose=*/false, /*conflate=*/true,
      /*rcv_hwm=*/1);
  source->subscriber_->SetOnDecodedMessage(
      [self = source.get()](
          const std::string& t,
          const ZMQPackedMessageSubscriber::DecodedHeader& h,
          const std::vector<ZMQPackedMessageSubscriber::BufferView>& b) {
        self->HandleDecoded(t, h, b);
      });

  if (!source->subscriber_->Connect()) {
    throw std::runtime_error(
        "ZmqPoseInputSource: failed to connect to tcp://" + host + ":" +
        std::to_string(port));
  }
  source->subscriber_->Start();

  // Pre-fill the cache with the trained stand pose so that the very first
  // call to Sample() (before any VLA frame arrives) still returns a
  // physically reasonable reference. The subscriber's first decode flips
  // has_body_reference_ to true and overwrites this.
  for (std::size_t i = 0; i < NUM_DOFS; ++i) {
    source->latest_frame_.joint_pos_mj[i] = default_angles[i];
    source->latest_frame_.joint_vel_mj[i] = 0.0;
    source->previous_frame_.joint_pos_mj[i] = default_angles[i];
    source->previous_frame_.joint_vel_mj[i] = 0.0;
  }
  source->latest_frame_.root_quat_xyzw = {0.0, 0.0, 0.0, 1.0};
  source->previous_frame_.root_quat_xyzw = {0.0, 0.0, 0.0, 1.0};
  return source;
}

ZmqPoseInputSource::ZmqPoseInputSource(
    const std::string& /*host*/, int /*port*/, const std::string& /*topic*/)
{
  // All real wiring happens in Connect(). The ctor exists purely so
  // std::unique_ptr<ZmqPoseInputSource> can be the return type of Connect().
}

ZmqPoseInputSource::~ZmqPoseInputSource()
{
  if (subscriber_) {
    subscriber_->Stop();
    subscriber_.reset();
  }
}

// ---------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------

bool ZmqPoseInputSource::CopyFloat32IntoDouble(
    const ZMQPackedMessageSubscriber::FieldInfo& field,
    const ZMQPackedMessageSubscriber::BufferView& buffer,
    double* out, std::size_t expected_count)
{
  if (field.dtype != "f32") {
    std::cerr << "[ZmqPoseInputSource] field '" << field.name
              << "' has unexpected dtype '" << field.dtype
              << "', want f32" << std::endl;
    return false;
  }
  std::size_t total = 1;
  for (auto d : field.shape) total *= d;
  if (total != expected_count) {
    std::cerr << "[ZmqPoseInputSource] field '" << field.name
              << "' has " << total << " elements, want " << expected_count
              << std::endl;
    return false;
  }
  if (buffer.size != total * sizeof(float)) {
    std::cerr << "[ZmqPoseInputSource] field '" << field.name
              << "' byte size mismatch: " << buffer.size
              << " vs " << total * sizeof(float) << std::endl;
    return false;
  }
  // We assume the publisher's byte order matches ours; ZMQPackedMessageSubscriber
  // exposes a byte-swap hint via DecodedHeader::NeedsByteSwap, but we don't
  // have it here. Pass needs_swap=false unconditionally; this matches every
  // x86_64/aarch64 host we ship to (all little-endian).
  DecodeF32Array(buffer.data, expected_count, out, /*needs_swap=*/false);
  return true;
}

bool ZmqPoseInputSource::CopyInt64Scalar(
    const ZMQPackedMessageSubscriber::FieldInfo& field,
    const ZMQPackedMessageSubscriber::BufferView& buffer,
    int64_t* out)
{
  if (field.dtype != "i64") {
    std::cerr << "[ZmqPoseInputSource] field '" << field.name
              << "' has unexpected dtype '" << field.dtype
              << "', want i64" << std::endl;
    return false;
  }
  if (buffer.size < sizeof(int64_t)) return false;
  std::memcpy(out, buffer.data, sizeof(int64_t));
  return true;
}

void ZmqPoseInputSource::HandleDecoded(
    const std::string& /*topic*/,
    const ZMQPackedMessageSubscriber::DecodedHeader& header,
    const std::vector<ZMQPackedMessageSubscriber::BufferView>& buffers)
{
  if (header.fields.size() != buffers.size()) return;

  ReferenceFrame next_frame{};
  bool got_body = false;
  bool got_quat = false;
  ZmqHandJointsSnapshot next_hand{};
  std::array<double, 64> next_motion_token{};
  bool got_motion_token = false;
  int64_t next_frame_index = -1;

  // Seed next_frame from cache so partial messages don't overwrite valid state.
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    next_frame = latest_frame_;
    next_hand = latest_hand_;
    next_motion_token = latest_motion_token_;
  }

  for (std::size_t i = 0; i < header.fields.size(); ++i) {
    const auto& f = header.fields[i];
    const auto& b = buffers[i];

    if (f.name == "joint_pos_mj") {
      if (CopyFloat32IntoDouble(f, b, next_frame.joint_pos_mj.data(), NUM_DOFS)) {
        got_body = true;
      }
    } else if (f.name == "root_quat_xyzw") {
      if (CopyFloat32IntoDouble(f, b, next_frame.root_quat_xyzw.data(), 4)) {
        got_quat = true;
      }
    } else if (f.name == "left_hand_joints") {
      double tmp[DEFAULT_HAND_DOF_PER_SIDE]{};
      if (CopyFloat32IntoDouble(f, b, tmp, DEFAULT_HAND_DOF_PER_SIDE)) {
        for (std::size_t k = 0; k < DEFAULT_HAND_DOF_PER_SIDE; ++k) {
          next_hand.left[k] = tmp[k];
        }
        next_hand.valid = true;
      }
    } else if (f.name == "right_hand_joints") {
      double tmp[DEFAULT_HAND_DOF_PER_SIDE]{};
      if (CopyFloat32IntoDouble(f, b, tmp, DEFAULT_HAND_DOF_PER_SIDE)) {
        for (std::size_t k = 0; k < DEFAULT_HAND_DOF_PER_SIDE; ++k) {
          next_hand.right[k] = tmp[k];
        }
        next_hand.valid = true;
      }
    } else if (f.name == "motion_token") {
      if (CopyFloat32IntoDouble(f, b, next_motion_token.data(), 64)) {
        got_motion_token = true;
      }
    } else if (f.name == "frame_index") {
      int64_t idx{};
      if (CopyInt64Scalar(f, b, &idx)) next_frame_index = idx;
    }
    // Unknown fields are silently ignored (forward compat).
  }

  // Joint velocity comes from finite differencing the last two body refs.
  // Mirrors PklMotionReference's strategy. We rely on the publisher running
  // at a roughly constant rate (mock-VLA at 50 Hz); rate jitter shows up as
  // velocity noise, which the policy tolerates.
  auto now = std::chrono::steady_clock::now();
  if (got_body) {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (has_body_reference_.load(std::memory_order_acquire)) {
      const auto dt = std::chrono::duration<double>(now - latest_recv_).count();
      if (dt > 0.0) {
        for (std::size_t i = 0; i < NUM_DOFS; ++i) {
          next_frame.joint_vel_mj[i] =
              (next_frame.joint_pos_mj[i] - latest_frame_.joint_pos_mj[i]) / dt;
        }
      } else {
        next_frame.joint_vel_mj = latest_frame_.joint_vel_mj;
      }
    } else {
      next_frame.joint_vel_mj.fill(0.0);
    }
    previous_frame_ = latest_frame_;
    previous_recv_  = latest_recv_;
    latest_frame_   = next_frame;
    latest_recv_    = now;
    if (next_hand.valid) {
      next_hand.frame_index = next_frame_index;
      latest_hand_ = next_hand;
    }
    if (got_motion_token) {
      latest_motion_token_ = next_motion_token;
    }
    has_body_reference_.store(true, std::memory_order_release);
  } else {
    // Token-only / hand-only frame: refresh side-channel caches without
    // touching the body reference (Sample() keeps returning the last
    // good body frame).
    std::lock_guard<std::mutex> lock(cache_mutex_);
    if (next_hand.valid) {
      next_hand.frame_index = next_frame_index;
      latest_hand_ = next_hand;
    }
    if (got_motion_token) {
      latest_motion_token_ = next_motion_token;
    }
    if (got_quat) {
      latest_frame_.root_quat_xyzw = next_frame.root_quat_xyzw;
    }
  }

  total_frames_received_.fetch_add(1, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------------

ReferenceFrame ZmqPoseInputSource::Sample(double /*time*/) const
{
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return latest_frame_;
}

ZmqHandJointsSnapshot ZmqPoseInputSource::LatestHandJoints() const
{
  std::lock_guard<std::mutex> lock(cache_mutex_);
  return latest_hand_;
}

}  // namespace agi_x2
