#!/usr/bin/env python3
"""x2_camera_recorder.py -- direct v4l2 camera recorder for PC2.

Captures the X2 head front-center RGB stream (Orbbec Gemini 335 over USB at
``/dev/video7``) straight from the kernel v4l2 device, no ROS / no aimrt /
no HAL bridge. Encodes to MP4 with NVENC H.264 on the Jetson so file sizes
stay small and the CPU stays cold.

Why direct v4l2 (and not the existing ROS topic):
    The ``orbbec_camera`` em-app publishes ``/aima/hal/sensor/rgb_head_
    front_center/rgb_image/compressed`` at ~28 Hz with QoS RELIABLE +
    TRANSIENT_LOCAL, but external (non-aimrt) ROS subscribers on this PC2
    only reliably receive the cached late-joiner frame; live frames after
    that drop silently (likely FastDDS SHM ring exhaustion when the
    publisher's writer history depth exceeds the subscriber buffer). Going
    straight to v4l2 sidesteps the whole DDS path.

Prerequisite: ``/dev/video7`` must be free. While the em-managed
``orbbec_camera`` app is up it holds the device exclusively, so before
running this recorder do::

    aima em stop-app orbbec_camera

and when you're done, restart the publisher with::

    aima em start-app orbbec_camera

The recorder will refuse to start if the device is held, with a one-line
hint pointing at the stop-app command.

Camera mount note: the Orbbec is physically inverted on the X2 head, so
the recorder rotates 180 degrees by default (``--rotate 180``). Pass
``--rotate 0`` for the raw sensor orientation.

Encoders:
    nvenc (default) -- hardware H.264 via ``nvv4l2h264enc``. Small files,
        plays anywhere. Requires gstreamer + nvargus plugins (preinstalled
        on JetPack).
    mjpg            -- MJPG passthrough into MKV. Zero transcoding, zero
        decode CPU. Files are ~5-10x larger than NVENC H.264 but the
        recorder is dead simple and the source bytes are preserved.

Typical usage on PC2::

    # 10 s clip, default 1280x720 @ 30 fps, NVENC H.264, MP4, 180-rotated
    python3 x2_camera_recorder.py --duration 10 --output /tmp/demo.mp4

    # continuous recording until Ctrl+C, MJPG passthrough into MKV
    python3 x2_camera_recorder.py --encoder mjpg --duration 0 \
        --output /tmp/demo.mkv

    # single still
    python3 x2_camera_recorder.py --snapshot /tmp/snap.jpg
"""

from __future__ import annotations

import argparse
import datetime
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

_DEFAULT_DEVICE = "/dev/video7"
_DEFAULT_WIDTH = 1280
_DEFAULT_HEIGHT = 720
_DEFAULT_FPS = 30
_DEFAULT_BITRATE = 4_000_000  # 4 Mbit/s, ~30 MB / minute at 720p30
_DEFAULT_ROTATE = 180  # Orbbec is mounted upside-down on the X2 head


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
def _check_device(device: str) -> None:
    if not Path(device).exists():
        sys.exit(
            f"ERROR: device {device} does not exist. "
            "Run `v4l2-ctl --list-devices` and pick the right /dev/video*."
        )
    try:
        fd = os.open(device, os.O_RDWR)
        os.close(fd)
    except OSError as e:
        if e.errno in (16, 11):  # EBUSY / EAGAIN
            sys.exit(
                f"ERROR: {device} is busy (errno={e.errno}).\n"
                "  Most likely the orbbec_camera em-app is holding it.\n"
                "  Free it with:  aima em stop-app orbbec_camera\n"
                "  Then re-run this recorder. When done, restart the\n"
                "  publisher with:  aima em start-app orbbec_camera"
            )
        sys.exit(f"ERROR: cannot open {device}: {e}")


def _check_gstreamer(encoder: str) -> None:
    if not shutil.which("gst-launch-1.0"):
        sys.exit("ERROR: gst-launch-1.0 not found (apt install gstreamer1.0-tools).")
    if encoder == "nvenc":
        # nvv4l2h264enc is JetPack-only; warn early rather than at runtime.
        rc = subprocess.run(
            ["gst-inspect-1.0", "nvv4l2h264enc"],
            capture_output=True,
            text=True,
        ).returncode
        if rc != 0:
            sys.exit(
                "ERROR: nvv4l2h264enc plugin not found. "
                "Either run on a Jetson with JetPack installed, or pass "
                "--encoder mjpg for the no-transcode passthrough path."
            )


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------
def _videoflip_method(rotate: int) -> str:
    # gstreamer videoflip / nvvidconv flip-method numeric codes differ;
    # we use videoflip's named methods which work for both software paths.
    return {
        0: "none",
        90: "clockwise",
        180: "rotate-180",
        270: "counterclockwise",
    }[rotate]


def _build_nvenc_pipeline(
    device: str,
    width: int,
    height: int,
    fps: int,
    rotate: int,
    bitrate: int,
    output: Path,
) -> list[str]:
    # MJPG capture -> CPU JPEG decode -> software rotate (cheap on Jetson
    # because the frame is already at display resolution; nvvidconv's
    # flip-method has historically had bugs with USB-cam pipelines so we
    # do it in videoflip pre-upload) -> upload to NVMM -> NVENC -> MP4.
    return [
        "gst-launch-1.0",
        "-e",  # send EOS on SIGINT so mp4mux finalizes the file
        "v4l2src",
        f"device={device}",
        "io-mode=2",
        "!",
        f"image/jpeg,width={width},height={height},framerate={fps}/1",
        "!",
        "jpegdec",
        "!",
        "videoflip",
        f"method={_videoflip_method(rotate)}",
        "!",
        "videoconvert",
        "!",
        "video/x-raw,format=NV12",
        "!",
        "nvvidconv",
        "!",
        "video/x-raw(memory:NVMM),format=NV12",
        "!",
        "nvv4l2h264enc",
        f"bitrate={bitrate}",
        "insert-sps-pps=1",
        "iframeinterval=30",
        "!",
        "h264parse",
        "!",
        "mp4mux",
        "!",
        "filesink",
        f"location={output}",
    ]


def _build_mjpg_pipeline(
    device: str,
    width: int,
    height: int,
    fps: int,
    rotate: int,
    output: Path,
) -> list[str]:
    # MJPG passthrough -- no decode, no encode, just rewrap into MKV.
    # NOTE: rotation can't be applied without decoding, so we ignore the
    # --rotate flag in this mode and warn the operator at startup.
    if rotate != 0:
        print(
            f"WARN: --encoder mjpg cannot rotate (would need to decode + "
            f"re-encode); ignoring --rotate {rotate}. Frames will be saved "
            f"in raw sensor orientation. Use --encoder nvenc to rotate.",
            file=sys.stderr,
        )
    return [
        "gst-launch-1.0",
        "-e",
        "v4l2src",
        f"device={device}",
        "io-mode=2",
        "!",
        f"image/jpeg,width={width},height={height},framerate={fps}/1",
        "!",
        "jpegparse",
        "!",
        "matroskamux",
        "!",
        "filesink",
        f"location={output}",
    ]


# ---------------------------------------------------------------------------
# Snapshot mode (single still, OpenCV-based -- no gstreamer needed)
# ---------------------------------------------------------------------------
def _snapshot(
    device: str,
    width: int,
    height: int,
    rotate: int,
    output: Path,
    warmup_frames: int = 8,
) -> int:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("ERROR: --snapshot requires python3-opencv (apt install python3-opencv).")
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        sys.exit(f"ERROR: cv2 could not open {device}.")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Auto-exposure can take ~5-10 frames to settle; throw the first few away.
    frame = None
    for _ in range(warmup_frames):
        ok, f = cap.read()
        if ok and f is not None:
            frame = f
    cap.release()
    if frame is None:
        sys.exit(f"ERROR: cv2 captured 0 frames from {device}.")
    if rotate == 90:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotate == 180:
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    elif rotate == 270:
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(str(output), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
    size_kb = output.stat().st_size // 1024
    print(f"snapshot: {output} ({frame.shape[1]}x{frame.shape[0]}, {size_kb} KB)")
    return 0


# ---------------------------------------------------------------------------
# Recording driver
# ---------------------------------------------------------------------------
def _run_recording(cmd: list[str], duration_s: float) -> int:
    # Pretty-print the pipeline once so the operator can replay it manually.
    print("pipeline: " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    t0 = time.monotonic()

    def _stop(signum, _frame):  # noqa: ANN001
        # SIGINT -> gst-launch's signal handler emits EOS, mp4mux finalizes.
        print(f"\n[recorder] received signal {signum}; sending SIGINT to gst-launch ...", flush=True)
        try:
            proc.send_signal(signal.SIGINT)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        if duration_s > 0:
            # Wait the requested duration, then send SIGINT for clean EOS.
            try:
                proc.wait(timeout=duration_s)
            except subprocess.TimeoutExpired:
                print(
                    f"[recorder] {duration_s:.1f}s elapsed; stopping ...",
                    flush=True,
                )
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=10)
        else:
            proc.wait()  # run until Ctrl+C
    except KeyboardInterrupt:
        # Defensive -- the signal handler above usually catches first.
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)

    rc = proc.returncode
    dt = time.monotonic() - t0
    print(f"[recorder] gst-launch exited rc={rc} after {dt:.1f}s", flush=True)
    return rc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--device",
        default=_DEFAULT_DEVICE,
        help=f"v4l2 device path (default {_DEFAULT_DEVICE}).",
    )
    p.add_argument("--width", type=int, default=_DEFAULT_WIDTH)
    p.add_argument("--height", type=int, default=_DEFAULT_HEIGHT)
    p.add_argument("--fps", type=int, default=_DEFAULT_FPS)
    p.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=_DEFAULT_ROTATE,
        help=(
            f"Rotation in degrees (default {_DEFAULT_ROTATE} -- the Orbbec "
            "is mounted upside-down on the X2 head). Set 0 for raw sensor "
            "orientation."
        ),
    )
    p.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Recording duration in seconds (0 = until Ctrl+C). Default 10.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output file. Default: /tmp/x2_cam_<YYYYMMDD_HHMMSS>.<ext> with "
            "extension picked from --encoder (mp4 / mkv / jpg)."
        ),
    )
    p.add_argument(
        "--encoder",
        choices=["nvenc", "mjpg"],
        default="nvenc",
        help=(
            "nvenc = NVENC H.264 in MP4 (default, recommended for sharing). "
            "mjpg = MJPG passthrough in MKV (no transcode, larger files, "
            "ignores --rotate)."
        ),
    )
    p.add_argument(
        "--bitrate",
        type=int,
        default=_DEFAULT_BITRATE,
        help=f"NVENC target bitrate in bits/sec (default {_DEFAULT_BITRATE}).",
    )
    p.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help=(
            "Single-frame mode: capture one still to the given path "
            "(JPEG, OpenCV-based) and exit. Ignores --duration / --encoder."
        ),
    )
    args = p.parse_args(argv)

    # ----- Preflight ------------------------------------------------------
    _check_device(args.device)

    # ----- Snapshot fast path --------------------------------------------
    if args.snapshot is not None:
        return _snapshot(
            args.device, args.width, args.height, args.rotate, args.snapshot
        )

    # ----- Pick output path ----------------------------------------------
    if args.output is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ext = "mp4" if args.encoder == "nvenc" else "mkv"
        args.output = Path(f"/tmp/x2_cam_{ts}.{ext}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    _check_gstreamer(args.encoder)

    # ----- Build + run pipeline ------------------------------------------
    if args.encoder == "nvenc":
        cmd = _build_nvenc_pipeline(
            args.device,
            args.width,
            args.height,
            args.fps,
            args.rotate,
            args.bitrate,
            args.output,
        )
    else:
        cmd = _build_mjpg_pipeline(
            args.device,
            args.width,
            args.height,
            args.fps,
            args.rotate,
            args.output,
        )

    print(
        f"[recorder] device={args.device} {args.width}x{args.height}@{args.fps} "
        f"rotate={args.rotate}deg encoder={args.encoder} duration={args.duration}s",
        flush=True,
    )
    print(f"[recorder] -> {args.output}", flush=True)

    rc = _run_recording(cmd, args.duration)

    if args.output.exists():
        size_mb = args.output.stat().st_size / (1024 * 1024)
        print(f"[recorder] wrote {args.output} ({size_mb:.2f} MB)")
    else:
        print(f"[recorder] WARN: output file {args.output} was not created")
    return rc


if __name__ == "__main__":
    sys.exit(main())
