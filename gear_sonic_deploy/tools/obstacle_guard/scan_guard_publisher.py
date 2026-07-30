#!/usr/bin/env python3
"""Publish the forward-obstacle guard state on ZMQ, and speak on the edge.

PUB binds ``tcp://127.0.0.1:5571``, topic ``b"scan_guard"``::

    {"blocked": bool, "dist": float, "ts": float}

``pc2_kplanner_onnx.py`` subscribes and zeroes forward/lateral while blocked.

Runs under SYSTEM python: kplanner and the pad bridge live in the gear_sonic
venv, which has no rclpy. Shipping a boolean over ZMQ keeps ROS out of the gait
process entirely.

Requires the ROS paths spelled out absolutely -- rclpy is in
``.../local/lib/python3.10/dist-packages`` while rpyutils is in
``site-packages``, and a fresh post-reboot login shell has no ROS env at all.
"""
import json
import threading
import time

import rclpy
import zmq
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

import scan_guard
from aimdk_msgs.srv import PlayAudioFile

PORT = 5571
SPEAK_COOLDOWN_S = 6.0
# Must live on SOC2 (10.0.1.42), which owns the audio hardware. A path that
# exists only on SOC1 returns SUCCESS and plays nothing. Not /tmp: it clears on
# reboot and the speech then dies silently.
WAV_DIR = "/agibot/data/home/agi/audio/"
WAV_NAME = "obstacle.wav"


class Speaker(Node):
    """Plays a pre-rendered English WAV.

    NOT PlayTts: the onboard TTS is a Chinese Volcano voice that reads English
    phonetically ("Hi" -> "hei"). voice_id 0-7 are one engine, voice_id is
    commented out of PlayTtsRequest, SetRobotLanguage("en") has no effect, and
    setting voice_type in BOTH tts_config.conf copies changed nothing.
    espeak-ng renders the phrase once and PlayAudioFile plays it.
    """

    def __init__(self) -> None:
        super().__init__("scan_guard_speaker")
        self.cli = self.create_client(
            PlayAudioFile, "/aimdk_5Fmsgs/srv/PlayAudioFile")
        self._last = 0.0

    def say(self) -> None:
        now = time.monotonic()
        if now - self._last < SPEAK_COOLDOWN_S:
            return
        self._last = now
        if not self.cli.service_is_ready():
            return
        r = PlayAudioFile.Request()
        r.request.header.stamp = self.get_clock().now().to_msg()
        r.file.pkg_name = "scan_guard"
        r.file.file_name = WAV_NAME
        r.file.file_path = WAV_DIR
        r.file.info.channels = 1
        r.file.info.sample_rate = 22050
        r.file.info.sample_format = "s16"
        r.file.info.coding_format = "wav"
        r.file.priority = 8
        self.cli.call_async(r)      # fire and forget: never block the loop


def main() -> None:
    # ONE executor for both nodes: two separate rclpy.spin() calls race on the
    # same global executor and die with "generator already executing".
    rclpy.init()
    guard = scan_guard.ScanGuard()
    spk = Speaker()
    ex = SingleThreadedExecutor()
    ex.add_node(guard)
    ex.add_node(spk)
    threading.Thread(target=ex.spin, daemon=True).start()

    sock = zmq.Context.instance().socket(zmq.PUB)
    sock.bind(f"tcp://127.0.0.1:{PORT}")
    print(f"[scan-guard] PUB bind tcp://127.0.0.1:{PORT} "
          f"stop={scan_guard.STOP_M}m arc=+/-{scan_guard.ARC_DEG}deg",
          flush=True)

    was = False
    while True:
        b = bool(guard.blocked())
        if b and not was:
            print(f"[scan-guard] OBSTACLE {guard.distance():.2f}m", flush=True)
            spk.say()
        was = b
        sock.send_multipart([b"scan_guard", json.dumps({
            "blocked": b,
            "dist": float(guard.distance()),
            "ts": time.time(),
        }).encode()])
        # Faster than the 10 Hz scan so the clamp sees fresh state on every
        # planner_cmd. Detection latency floors at the sensor's own 10 Hz.
        time.sleep(0.02)


if __name__ == "__main__":
    main()
