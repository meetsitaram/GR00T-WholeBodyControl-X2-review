#!/usr/bin/env python3
"""Bridge AgiBot HAL camera ROS topics → ``ComposedCameraClientSensor`` ZMQ.

Runs on PC2 (the Jetson Orin NX). Subscribes to one or more
``/aima/hal/sensor/.../rgb_image/compressed`` ROS 2 topics, optionally
resizes the JPEG frames, and re-publishes them as a single
``ImageMessageSchema`` payload over a ZMQ ``PUB`` socket.

This is the **server side** of the same wire protocol the existing
:class:`gear_sonic.camera.composed_camera.ComposedCameraClientSensor`
consumes. The X2 dataset recorder (laptop side) connects to it via
``ComposedCameraClientSensor(server_ip="10.0.1.41", port=5555)``.

Default plumbing matches the four AgiBot head cameras that come up
after ``./gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal``:

* ``head_front``   ← ``/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed``
                     (Orbbec Gemini 335 RGB, native 2688×1944)
* ``stereo_left``  ← ``/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed``
                     (IMX900, native 2064×1552)
* ``stereo_right`` ← ``/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed``
                     (IMX900, native 2064×1552)

These mount-key names line up 1:1 with the
``observation.images.{head_front,stereo_left,stereo_right}`` feature
keys declared by
:func:`gear_sonic.data.features_x2_vla.get_features_x2_vla` when
``include_head_cameras=True``, so the recorder can drop the merged
``images`` dict straight into the LeRobot frame.

The rear camera is intentionally skipped by default (operator-facing,
not useful for VLA). Pass ``--include-rear`` to also publish it as
``head_rear``.

Frames are decoded → resized → re-encoded as MJPEG once per stream
(quality 85 by default) before being packed into the ZMQ payload.
Re-encode + resize takes ~1.5 ms per IMX900 frame on the Orin NX, well
inside the 50 Hz dataset tick budget. The whole bridge runs at the
slowest publisher's frame rate (~15 Hz for the HAL today).

Typical usage (from the laptop, via the helper script)::

    ./gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve

…or directly on PC2::

    source /opt/ros/humble/setup.bash
    export FASTRTPS_DEFAULT_PROFILES_FILE=/agibot/software/entry/cfg/ros_dds_configuration.xml
    python3 /tmp/x2_pc2_camera_zmq_publisher.py --port 5555 --width 640 --height 480
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2  # noqa: F401 — imported early; some ROS builds segfault otherwise
import msgpack
import numpy as np
import zmq

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage


# ---------------------------------------------------------------------------
# Stream definitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamSpec:
    """A single ROS → ZMQ camera bridge stream."""

    mount_key: str
    """Key in the published ``ImageMessageSchema.images`` dict (e.g.
    ``"ego_view"``). Must match what the laptop-side recorder expects
    in :func:`gear_sonic.data.features_x2_vla.get_features_x2_vla`."""

    ros_topic: str
    """ROS 2 CompressedImage topic to subscribe to."""


DEFAULT_STREAMS: tuple[StreamSpec, ...] = (
    StreamSpec(
        mount_key="head_front",
        ros_topic="/aima/hal/sensor/rgb_head_front_center/rgb_image/compressed",
    ),
    StreamSpec(
        mount_key="stereo_left",
        ros_topic="/aima/hal/sensor/stereo_head_front_left/rgb_image/compressed",
    ),
    StreamSpec(
        mount_key="stereo_right",
        ros_topic="/aima/hal/sensor/stereo_head_front_right/rgb_image/compressed",
    ),
)


REAR_STREAM = StreamSpec(
    mount_key="head_rear",
    ros_topic="/aima/hal/sensor/rgb_head_rear/rgb_image/compressed",
)


# ---------------------------------------------------------------------------
# Per-stream worker
# ---------------------------------------------------------------------------


class _StreamWorker:
    """One ROS subscriber per stream, with its own latest-frame slot.

    Why one subscriber per stream (and not one node with many subs):
    we found empirically that on PC2's FastDDS profile a single
    multi-threaded executor with 4 subscriptions only ever gets
    scheduled for the first matched topic; per-process subscribers
    work robustly. See ``/tmp/x2_record_one_cam.py`` for the original
    test result.
    """

    def __init__(
        self,
        spec: StreamSpec,
        *,
        width: int,
        height: int,
        jpeg_quality: int,
    ) -> None:
        self.spec = spec
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality

        self._lock = threading.Lock()
        self._latest_jpeg: bytes | None = None
        self._latest_ts: float = 0.0
        self._n_received = 0
        self._n_dropped_decode = 0

        # Each worker gets its OWN rclpy Node so we can spin them in
        # separate threads / executors. rclpy supports multiple nodes
        # in one context as long as the names differ.
        self._node = Node(f"x2_pc2_camera_bridge_{spec.mount_key}")
        self._node.create_subscription(
            CompressedImage,
            spec.ros_topic,
            self._on_msg,
            self._build_qos(),
        )

    @staticmethod
    def _build_qos() -> QoSProfile:
        # Publishers use RELIABLE + TRANSIENT_LOCAL. Match RELIABLE on
        # our side so the QoS handshake actually completes (a
        # BEST_EFFORT subscriber against a RELIABLE publisher is
        # technically incompatible and FastDDS silently drops).
        return QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

    def _on_msg(self, msg: CompressedImage) -> None:
        try:
            raw = bytes(msg.data)
            # Resize + re-encode at the bridge so the wire-side stays
            # constant regardless of native sensor resolution
            # (Orbbec is 2688×1944, IMX900 is 2064×1552).
            arr = np.frombuffer(raw, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                self._n_dropped_decode += 1
                return
            h, w = bgr.shape[:2]
            if (w, h) != (self.width, self.height):
                bgr = cv2.resize(
                    bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
                )
            ok, jpg = cv2.imencode(
                ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
            )
            if not ok:
                self._n_dropped_decode += 1
                return
            jpg_bytes = jpg.tobytes()
            # Prefer the publisher's timestamp; fall back to wallclock
            # when the publisher left it empty.
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            if ts == 0:
                ts = time.time()
            with self._lock:
                self._latest_jpeg = jpg_bytes
                self._latest_ts = ts
                self._n_received += 1
        except Exception as exc:  # pragma: no cover — defensive
            print(
                f"[bridge:{self.spec.mount_key}] on_msg error: {exc}",
                flush=True,
            )
            self._n_dropped_decode += 1

    def snapshot(self) -> tuple[bytes | None, float]:
        with self._lock:
            return self._latest_jpeg, self._latest_ts

    def stats(self) -> tuple[int, int]:
        with self._lock:
            return self._n_received, self._n_dropped_decode

    def destroy(self) -> None:
        try:
            self._node.destroy_node()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Publisher main loop
# ---------------------------------------------------------------------------


def _spin_worker(worker: _StreamWorker, stop_event: threading.Event) -> None:
    """Spin a single rclpy node on its OWN executor until stop_event is set.

    Each thread needs its own executor; sharing rclpy's default global
    executor across threads triggers "generator already executing" errors
    because the underlying ``_wait_for_ready_callbacks`` generator is not
    re-entrant.
    """
    executor = SingleThreadedExecutor()
    executor.add_node(worker._node)
    try:
        while not stop_event.is_set():
            try:
                executor.spin_once(timeout_sec=0.05)
            except Exception as exc:  # pragma: no cover — defensive
                print(
                    f"[bridge:{worker.spec.mount_key}] spin error: {exc}",
                    flush=True,
                )
                time.sleep(0.1)
    finally:
        try:
            executor.remove_node(worker._node)
        except Exception:
            pass
        try:
            executor.shutdown()
        except Exception:
            pass


def run(
    streams: list[StreamSpec],
    *,
    port: int,
    width: int,
    height: int,
    jpeg_quality: int,
    publish_rate_hz: float,
    bind_host: str,
) -> int:
    rclpy.init()
    stop_event = threading.Event()

    def _sig(sig: int, frame: Any) -> None:  # noqa: ARG001
        print(f"[bridge] signal {sig} → shutting down", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    workers = [
        _StreamWorker(
            spec, width=width, height=height, jpeg_quality=jpeg_quality
        )
        for spec in streams
    ]
    spin_threads = [
        threading.Thread(target=_spin_worker, args=(w, stop_event), daemon=True)
        for w in workers
    ]
    for t in spin_threads:
        t.start()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDHWM, 20)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(f"tcp://{bind_host}:{port}")
    print(
        f"[bridge] zmq PUB bound tcp://{bind_host}:{port}, "
        f"streams={[s.mount_key for s in streams]}, "
        f"out={width}x{height}@q{jpeg_quality}, "
        f"publish_rate={publish_rate_hz:.1f}Hz",
        flush=True,
    )

    period = 1.0 / max(publish_rate_hz, 1e-3)
    next_tick = time.monotonic()
    last_log = time.monotonic()
    last_log_counts = [w.stats()[0] for w in workers]
    sent = 0
    dropped_zmq = 0

    try:
        while not stop_event.is_set():
            next_tick += period

            images: dict[str, bytes] = {}
            timestamps: dict[str, float] = {}
            any_image = False
            for w in workers:
                jpg, ts = w.snapshot()
                if jpg is not None:
                    images[w.spec.mount_key] = jpg
                    timestamps[w.spec.mount_key] = ts
                    any_image = True

            if any_image:
                payload = {"timestamps": timestamps, "images": images}
                packed = msgpack.packb(payload, use_bin_type=True)
                try:
                    sock.send(packed, flags=zmq.NOBLOCK)
                    sent += 1
                except zmq.Again:
                    dropped_zmq += 1

            now = time.monotonic()
            if now - last_log >= 5.0:
                cur_counts = [w.stats()[0] for w in workers]
                rates = [
                    (c - p) / (now - last_log)
                    for c, p in zip(cur_counts, last_log_counts, strict=True)
                ]
                rate_str = ", ".join(
                    f"{w.spec.mount_key}={r:.1f}Hz"
                    for w, r in zip(workers, rates, strict=True)
                )
                print(
                    f"[bridge] sent={sent} dropped_zmq={dropped_zmq} | "
                    f"in_rates: {rate_str}",
                    flush=True,
                )
                last_log = now
                last_log_counts = cur_counts

            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                # We're behind; reset baseline so we don't burst.
                next_tick = time.monotonic()
    finally:
        for w in workers:
            w.destroy()
        sock.close()
        ctx.term()
        try:
            rclpy.shutdown()
        except Exception:
            pass
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ PUB port (default 5555 to match composed_camera convention).",
    )
    ap.add_argument(
        "--bind-host",
        type=str,
        default="*",
        help="Interface to bind ZMQ PUB on. Default '*' = all.",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=640,
        help="Output frame width after resize (matches the default render_width "
        "in features_x2_vla.py).",
    )
    ap.add_argument(
        "--height",
        type=int,
        default=480,
        help="Output frame height after resize (matches the default "
        "render_height in features_x2_vla.py).",
    )
    ap.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG re-encode quality 1-100. 85 ≈ visually lossless at 640×480.",
    )
    ap.add_argument(
        "--publish-rate",
        type=float,
        default=50.0,
        help="ZMQ publish rate (Hz). Defaults to 50 to match the recorder tick. "
        "If a publisher streams slower than this rate, the bridge republishes "
        "the latest frame at the requested rate (recorder reads latest-only).",
    )
    ap.add_argument(
        "--include-rear",
        action="store_true",
        help="Also publish the rear head camera as the 'head_rear' stream "
        "(off by default; the recorder schema only declares the three "
        "front-facing slots).",
    )
    ap.add_argument(
        "--dds-profile",
        type=str,
        default="/agibot/software/entry/cfg/ros_dds_configuration.xml",
        help="FastRTPS profile XML (auto-exported via FASTRTPS_DEFAULT_PROFILES_FILE "
        "before rclpy.init when not already set). Pass an empty string to skip.",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dds_profile and "FASTRTPS_DEFAULT_PROFILES_FILE" not in os.environ:
        os.environ["FASTRTPS_DEFAULT_PROFILES_FILE"] = args.dds_profile
        print(
            f"[bridge] FASTRTPS_DEFAULT_PROFILES_FILE = {args.dds_profile}",
            flush=True,
        )

    streams = list(DEFAULT_STREAMS)
    if args.include_rear:
        streams.append(REAR_STREAM)
    return run(
        streams,
        port=args.port,
        width=args.width,
        height=args.height,
        jpeg_quality=args.jpeg_quality,
        publish_rate_hz=args.publish_rate,
        bind_host=args.bind_host,
    )


if __name__ == "__main__":
    sys.exit(main())
