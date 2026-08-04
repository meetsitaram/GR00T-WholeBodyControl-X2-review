#!/usr/bin/env python3
"""Generate + stage the PC3 robot-speaker voice prompts.

The X2's PC3 interaction unit plays pre-baked WAVs via ``aplay -D
playback_def`` (the dmix device that coexists with the vendor audio
stack — see docs/source/references/x2_interaction_layer.md). This
script is the single source of truth for those prompts:

  * regenerates every WAV via gTTS -> ffmpeg (48 kHz stereo pcm_s16le,
    -3 dB: the dmix device is locked to rate 48000, pre-resample so
    ALSA never converts),
  * writes them to the repo at gear_sonic_deploy/data/pc3_audio/
    (committed — PC3 is not backed up),
  * with ``--stage``, pushes them to PC3 ``/opt/x2_interact/audio/``
    via the PC2 two-hop (laptop -> run@PC2 -> agi@PC3, sshpass) and
    md5-verifies.

Consumers (all fire-and-forget, never block a control path):
  * pc2_kplanner_onnx.py ``_pc3_announce_estop``: estop_activating /
    estop_damping on the operator e-stop phases.
  * thermal notifier (PC2): thermal_warning while any motor is hot.

Usage:
    python3 gear_sonic_deploy/scripts/gen_pc3_audio_prompts.py           # regen only
    python3 gear_sonic_deploy/scripts/gen_pc3_audio_prompts.py --stage  # + push to PC3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "gear_sonic_deploy" / "data" / "pc3_audio"

PC2 = "run@${X2_PC2_HOST}"
PC3 = "agi@${X2_PC3_HOST}"
PC3_PASS = os.environ.get("X2_PC3_PASS", "")
PC3_AUDIO_DIR = "/opt/x2_interact/audio"

PROMPTS: dict[str, str] = {
    "estop_activating": "Emergency stop activating.",
    "estop_damping": "Emergency stop. Pure damping engaged.",
    "thermal_warning": (
        "Warning. Motor temperature high. "
        "Warning. Motor temperature high."
    ),
    # Group-specific thermal cues (2026-08-04): legs/waist are the
    # critical locomotion joints (loud, repeated more often by the
    # notifier); upper body is informational.
    "thermal_legs": (
        "Warning. Leg motor temperature high. Consider resting the robot."
    ),
    "thermal_upper": (
        "Notice. Arm motor temperature elevated."
    ),
}


def generate() -> None:
    from gtts import gTTS

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, text in PROMPTS.items():
        mp3 = OUT_DIR / f"{key}.mp3"
        wav = OUT_DIR / f"{key}.wav"
        gTTS(text, lang="en").save(str(mp3))
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
             "-ar", "48000", "-ac", "2", "-c:a", "pcm_s16le",
             "-af", "volume=-3dB", str(wav)],
            check=True)
        mp3.unlink()
        print(f"  {wav.relative_to(REPO)}  ({wav.stat().st_size} bytes)")


def stage() -> None:
    wavs = sorted(OUT_DIR.glob("*.wav"))
    if not wavs:
        sys.exit("nothing to stage — run without --stage first")
    names = [w.name for w in wavs]
    subprocess.run(
        ["scp", "-q", *map(str, wavs), f"{PC2}:/tmp/"], check=True)
    remote_tmp = " ".join(f"/tmp/{n}" for n in names)
    remote_dst = " ".join(f"{PC3_AUDIO_DIR}/{n}" for n in names)
    subprocess.run(
        ["ssh", PC2,
         f"sshpass -p {PC3_PASS} scp -o StrictHostKeyChecking=no "
         f"{remote_tmp} {PC3}:/tmp/ && "
         f"sshpass -p {PC3_PASS} ssh {PC3} "
         f"\"sudo -S mkdir -p {PC3_AUDIO_DIR} <<< {PC3_PASS} 2>/dev/null; "
         f"sudo -S cp {remote_tmp} {PC3_AUDIO_DIR}/ <<< {PC3_PASS} "
         f"2>/dev/null; md5sum {remote_dst}\""],
        check=True)
    for w in wavs:
        subprocess.run(["md5sum", str(w)], check=True)
    print("compare the two md5 blocks above — they must match pairwise")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", action="store_true",
                    help="push the WAVs to PC3 via the PC2 two-hop")
    args = ap.parse_args()
    generate()
    if args.stage:
        stage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
