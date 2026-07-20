"""Interactive recording SESSION — loop a clip-name list so you never type names.

Per clip:  ENTER/SPACE = start a fixed-duration capture (drive in the deploy
terminal while it records), then:  n = good, keep + next  |  r = redo this clip
|  s = skip  |  q = quit.  Every take accumulates into --out (the recorder
merges; a redo replaces just that clip and keeps a .prev backup).

    # Terminal 3 (recorder session):
    .venv/bin/python gear_sonic/scripts/record_session.py \
        --out gear_sonic/data/motions/g1_teleop_corpus.pkl --robot g1 \
        --fps 30,50 --duration 10 \
        --clips gear_sonic/data/motions/g1_teleop_clip_list.txt

Resume mid-list with --start-at <clip_key>.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import termios
import tty
from pathlib import Path

REC = "gear_sonic/scripts/record_motion_to_pkl.py"


def getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--robot", default="g1")
    ap.add_argument("--fps", default="30,50", help="output fps grid(s), e.g. 30,50")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--clips", default="gear_sonic/data/motions/g1_teleop_clip_list.txt",
                    help="file with one clip-key per line (# comments ok)")
    ap.add_argument("--start-at", default=None, help="resume from this clip key")
    args = ap.parse_args()

    clips = [l.strip() for l in Path(args.clips).read_text().splitlines()
             if l.strip() and not l.startswith("#")]
    if not clips:
        raise SystemExit(f"no clips in {args.clips}")
    if args.start_at:
        if args.start_at not in clips:
            raise SystemExit(f"--start-at {args.start_at!r} not in list")
        clips = clips[clips.index(args.start_at):]

    print(f"\n=== recording session: {len(clips)} clips -> {args.out} "
          f"({args.duration:g}s each, fps={args.fps}) ===")
    print("per clip:  ENTER/SPACE = start recording  |  s = skip  |  q = quit")
    print("after take: n = good/next  |  r = redo  |  s = skip  |  q = quit\n")

    i = 0
    while i < len(clips):
        key = clips[i]
        print(f"── [{i+1}/{len(clips)}] {key} ──  ENTER=record  s=skip  q=quit", flush=True)
        c = getch()
        if c in ("q", "\x03"):
            break
        if c == "s":
            print(f"   skipped {key}\n"); i += 1; continue
        # record loop (redo re-records without re-pressing ENTER)
        while True:
            print(f"   ● RECORDING {key} for {args.duration:g}s — DRIVE NOW in the deploy terminal...",
                  flush=True)
            rc = subprocess.run(
                [sys.executable, REC, "--robot", args.robot, "--out", args.out,
                 "--motion-key", key, "--fps", args.fps, "--duration", str(args.duration)]
            ).returncode
            status = "OK" if rc == 0 else f"FAILED(rc={rc})"
            print(f"   ▷ {key} recorded [{status}].  n=good/next  r=redo  s=skip  q=quit", flush=True)
            c = getch()
            if c in ("n", "\r", "\n"):
                i += 1; break
            if c == "r":
                print(f"   ↻ redo {key}"); continue
            if c == "s":
                i += 1; break
            if c in ("q", "\x03"):
                print("\nsession ended."); return
        print()
    print("\n=== session done ===")


if __name__ == "__main__":
    main()
