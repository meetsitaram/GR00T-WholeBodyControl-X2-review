"""Launch SONIC sim on a stitched motion + record MuJoCo window + caption clip names.

Three-stage pipeline:

  1. Spawn ``deploy_x2.sh sim --motion <pkl> --sim-viewer`` as a background
     subprocess and wait for the MuJoCo passive_viewer window to appear
     on the host X server (we run on X11 with DISPLAY :1).
  2. Start ``ffmpeg -f x11grab`` capturing the MuJoCo window region into
     a raw MP4 (``<out>.raw.mp4``) at 30 fps.
  3. When deploy finishes (or --max-duration trips), stop ffmpeg, then
     post-process the raw MP4 with a chained ``drawtext`` filter so each
     clip's name is overlaid during its time range (read from the
     stitcher's manifest JSON).

Final output: ``<out>.mp4`` with the clip name as a caption.

Run::

    .venv/bin/python -m gear_sonic.scripts.record_sonic_review_video \\
        --motion data/.../x2_browser_side_walks_review.pkl \\
        --manifest data/.../x2_browser_side_walks_review.manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DEPLOY_SH = _REPO_ROOT / "gear_sonic_deploy" / "deploy_x2.sh"
_DEFAULT_CKPT_DIR = Path(
    "/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported"
)
_DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "sim_to_real_anchors" / "browse_sonic" / "videos"
_MUJOCO_WINDOW_NAME = "MuJoCo"  # passive_viewer window title prefix


def _resolve_onnx(ckpt_dir: Path) -> Path:
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"ckpt dir not found: {ckpt_dir}")
    onnx = sorted(ckpt_dir.glob("*.onnx"))
    if not onnx:
        raise FileNotFoundError(f"no .onnx in {ckpt_dir}")
    return onnx[0]


def _find_mujoco_window(display: str, timeout_s: float = 30.0) -> dict | None:
    """Poll xwininfo until a window whose name contains 'MuJoCo' shows up.

    Returns ``{"id": "0x...", "x": int, "y": int, "w": int, "h": int}`` or
    None if not found within the timeout.
    """
    deadline = time.monotonic() + timeout_s
    env = {**os.environ, "DISPLAY": display}
    while time.monotonic() < deadline:
        try:
            # ``xwininfo -root -tree`` lists every window with name + geometry.
            res = subprocess.run(
                ["xwininfo", "-root", "-tree"],
                env=env, capture_output=True, text=True, timeout=10,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            time.sleep(0.5)
            continue
        for line in res.stdout.splitlines():
            # Format: "     0xNNNNNNN "MuJoCo": ("python3" "python3")  WxH+X+Y  +X+Y"
            if _MUJOCO_WINDOW_NAME not in line:
                continue
            m = re.search(
                r"(0x[0-9a-fA-F]+).*\"([^\"]*MuJoCo[^\"]*)\".*?(\d+)x(\d+)\+(-?\d+)\+(-?\d+)",
                line,
            )
            if not m:
                continue
            wid, name, w, h, x, y = m.groups()
            return {"id": wid, "name": name, "x": int(x), "y": int(y), "w": int(w), "h": int(h)}
        time.sleep(0.5)
    return None


def _spawn_deploy(
    motion_pkl: Path,
    onnx: Path,
    max_duration_s: int,
    log_path: Path,
    cam_track_body: str | None = "pelvis",
    cam_distance: float = 3.5,
    cam_elevation: float = -12.0,
    cam_azimuth: float = 135.0,
) -> subprocess.Popen:
    cmd = [
        "bash", str(_DEFAULT_DEPLOY_SH), "sim", "--no-confirm",
        "--motion", str(motion_pkl),
        "--model", str(onnx),
        "--sim-viewer",
        "--max-duration", str(int(max_duration_s)),
    ]
    # Lock the MuJoCo viewer camera onto the pelvis so the robot stays in
    # frame as it walks (stitched reviews can run 5+ minutes and travel
    # several meters in world coords -- a free camera drifts off-screen).
    if cam_track_body:
        cmd += [
            "--sim-cam-track-body", cam_track_body,
            "--sim-cam-distance", f"{cam_distance:g}",
            "--sim-cam-elevation", f"{cam_elevation:g}",
            "--sim-cam-azimuth", f"{cam_azimuth:g}",
        ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "wb")
    print(f"[record] spawning deploy: {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"[record]                  log -> {log_path}")
    return subprocess.Popen(
        cmd,
        stdout=log_f, stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        cwd=str(_REPO_ROOT),
    )


def _spawn_ffmpeg_x11(
    win: dict, display: str, raw_mp4: Path, fps: int = 30
) -> subprocess.Popen:
    """ffmpeg x11grab targeting the MuJoCo window region."""
    raw_mp4.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-f", "x11grab",
        "-framerate", str(fps),
        "-video_size", f"{win['w']}x{win['h']}",
        "-i", f"{display}+{win['x']},{win['y']}",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(raw_mp4),
    ]
    print(f"[record] starting ffmpeg x11grab @ {win['w']}x{win['h']}+{win['x']}+{win['y']}")
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, env=os.environ.copy())


def _kill_subprocess(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    print(f"[record] stopping {name} (pid {proc.pid}) ...")
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    except PermissionError:
        proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.wait()


def _stop_ffmpeg(proc: subprocess.Popen) -> None:
    """ffmpeg responds to a 'q' on stdin or SIGINT for a clean MOOV write."""
    if proc.poll() is not None:
        return
    print(f"[record] stopping ffmpeg (pid {proc.pid}) -- sending 'q' for clean trailer")
    try:
        if proc.stdin:
            proc.stdin.write(b"q")
            proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
    try:
        proc.wait(timeout=10)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def _ffmpeg_caption_filter(manifest: dict, fontfile: str | None) -> str:
    """Build a ``drawtext`` chain enabling clip names in their time windows.

    Each manifest entry contributes one drawtext box visible from
    ``blend_start_s`` to ``pad_end_s``. Boxes use a black background +
    white text, anchored top-center.
    """
    parts: list[str] = []
    # Common prefix (font file optional; ffmpeg picks a default if missing).
    base = (
        "drawtext="
        "text='%TEXT%'"
        ":x=(w-text_w)/2"
        ":y=20"
        ":fontsize=28"
        ":fontcolor=white"
        ":box=1:boxcolor=black@0.65:boxborderw=8"
        ":enable='between(t,%T0%,%T1%)'"
    )
    if fontfile:
        base += f":fontfile={fontfile}"
    for entry in manifest["clips"]:
        # Escape ffmpeg-special chars in the clip name.
        text = entry["clip_name"]
        text = text.replace("\\", "\\\\").replace("'", "\\'").replace(":", r"\:")
        # Two-line caption: index/total + clip name fits one line on a 4K viewer.
        idx = entry["index"] + 1
        total = manifest["n_clips"]
        full = f"[{idx}/{total}] {text}"
        # Escape commas in case any clip name has them (none should).
        full = full.replace(",", r"\,")
        f = (
            base
            .replace("%TEXT%", full)
            .replace("%T0%", f"{entry['blend_start_s']:.3f}")
            .replace("%T1%", f"{entry['pad_end_s']:.3f}")
        )
        parts.append(f)
    return ",".join(parts)


def _post_process_captions(
    raw_mp4: Path, manifest: dict, out_mp4: Path, fontfile: str | None
) -> bool:
    if not raw_mp4.is_file() or raw_mp4.stat().st_size < 1024:
        print(f"[record] WARN: raw mp4 missing or empty: {raw_mp4}")
        return False
    filt = _ffmpeg_caption_filter(manifest, fontfile=fontfile)
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-i", str(raw_mp4),
        "-vf", filt,
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-crf", "23",
        str(out_mp4),
    ]
    print(f"[record] post-processing captions -> {out_mp4}")
    res = subprocess.run(cmd)
    return res.returncode == 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--motion", type=Path, required=True,
                   help="Stitched motion PKL from stitch_side_walks_for_review.py")
    p.add_argument("--manifest", type=Path, required=True,
                   help="Manifest JSON sidecar (clip name + time ranges)")
    p.add_argument("--ckpt-dir", type=Path, default=_DEFAULT_CKPT_DIR,
                   help=f"SONIC ONNX dir (default: {_DEFAULT_CKPT_DIR})")
    p.add_argument("--out", type=Path, default=None,
                   help="Output captioned MP4 path (default: data/.../videos/<motion-stem>.mp4)")
    p.add_argument("--display", type=str, default=os.environ.get("DISPLAY", ":1"),
                   help="X display to capture from (default: $DISPLAY or :1)")
    p.add_argument("--fps", type=int, default=30,
                   help="Recording fps (default 30)")
    p.add_argument("--window-wait-s", type=float, default=30.0,
                   help="Max seconds to wait for the MuJoCo window to appear")
    p.add_argument("--extra-tail-s", type=float, default=4.0,
                   help="Extra wait after motion duration before stopping (RAMP_OUT etc.)")
    p.add_argument("--font", type=str, default=None,
                   help="Optional path to a TTF for caption text. Default uses ffmpeg's "
                        "fontconfig default (DejaVu Sans on most Linux).")
    p.add_argument("--cam-track-body", type=str, default="pelvis",
                   help="MJCF body the viewer camera follows (default 'pelvis'). "
                        "Pass an empty string to disable tracking.")
    p.add_argument("--cam-distance", type=float, default=3.5,
                   help="Tracking-camera distance in meters (default 3.5).")
    p.add_argument("--cam-elevation", type=float, default=-12.0,
                   help="Tracking-camera elevation in degrees, negative looks "
                        "down (default -12).")
    p.add_argument("--cam-azimuth", type=float, default=135.0,
                   help="Tracking-camera azimuth in degrees, 0=+X, 90=+Y, "
                        "180=-X, 270=-Y (default 135 = 3/4 view from front-right).")
    args = p.parse_args()

    if not args.motion.is_file():
        print(f"motion not found: {args.motion}", file=sys.stderr)
        return 2
    if not args.manifest.is_file():
        print(f"manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    manifest = json.loads(args.manifest.read_text())
    total_s = float(manifest["total_seconds"])
    max_dur = int(round(total_s + 1.0))  # +1s tail for the final stand pad

    out_dir = (args.out.parent if args.out else _DEFAULT_OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.motion.stem
    raw_mp4 = out_dir / f"{stem}.raw.mp4"
    out_mp4 = args.out or (out_dir / f"{stem}.mp4")
    deploy_log = out_dir / f"{stem}.deploy.log"

    onnx = _resolve_onnx(args.ckpt_dir)
    print(f"[record] motion       = {args.motion}")
    print(f"[record] manifest     = {args.manifest}  ({manifest['n_clips']} clips, {total_s:.1f}s)")
    print(f"[record] ONNX         = {onnx}")
    print(f"[record] X display    = {args.display}")
    print(f"[record] raw mp4      = {raw_mp4}")
    print(f"[record] captioned    = {out_mp4}")
    print()

    deploy = _spawn_deploy(
        args.motion, onnx,
        max_duration_s=max_dur, log_path=deploy_log,
        cam_track_body=(args.cam_track_body or None),
        cam_distance=args.cam_distance,
        cam_elevation=args.cam_elevation,
        cam_azimuth=args.cam_azimuth,
    )

    print(f"[record] waiting up to {args.window_wait_s:.0f}s for MuJoCo viewer ...")
    win = _find_mujoco_window(display=args.display, timeout_s=args.window_wait_s)
    if win is None:
        print("[record] ERROR: MuJoCo window never appeared. Killing deploy.", file=sys.stderr)
        _kill_subprocess(deploy, "deploy")
        return 3
    print(f"[record] viewer found: {win}")

    # Tiny settle so the gantry-drop frame isn't part of the recording.
    time.sleep(0.5)

    ffmpeg = _spawn_ffmpeg_x11(win, args.display, raw_mp4, fps=args.fps)

    deadline = time.monotonic() + total_s + args.extra_tail_s + 30  # extra slack for boot
    try:
        while time.monotonic() < deadline:
            if deploy.poll() is not None:
                print(f"[record] deploy exited with code {deploy.returncode}")
                break
            time.sleep(0.5)
        else:
            print("[record] WARN: deploy ran past expected deadline; killing.")
            _kill_subprocess(deploy, "deploy")
    except KeyboardInterrupt:
        print("[record] interrupted -- stopping subprocesses")
        _kill_subprocess(deploy, "deploy")

    _stop_ffmpeg(ffmpeg)

    print()
    print(f"[record] raw recording: {raw_mp4} ({raw_mp4.stat().st_size / 1e6:.1f} MB)")
    ok = _post_process_captions(raw_mp4, manifest, out_mp4, fontfile=args.font)
    if not ok:
        print(f"[record] caption pass FAILED. Raw is at: {raw_mp4}", file=sys.stderr)
        return 4

    print(f"[record] DONE. Captioned video: {out_mp4} ({out_mp4.stat().st_size / 1e6:.1f} MB)")
    print(f"[record] deploy log: {deploy_log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
