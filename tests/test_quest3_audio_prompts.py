"""Smoke tests for the Quest 3 prompt-audio cache.

The audio cache is what makes calibration prompts actually audible on
Quest 3 (the headset's ``speechSynthesis`` is unreliable -- see the
module docstring for context). These tests verify that

* the prompt-key registry contains every pose / status the WebXR
  client expects,
* MP3 keys are stable so renaming a pose doesn't silently produce a
  ``404`` on the headset,
* :func:`ensure_prompt_audio_files` is idempotent.

We intentionally skip the network-dependent gTTS generation step in CI;
generation is exercised end-to-end only when the cache is empty and a
network is available.
"""

from __future__ import annotations

import importlib

import pytest

from gear_sonic.utils.teleop.operator_calibration import CALIBRATION_POSE_IDS


def test_audio_prompts_module_importable() -> None:
    """Module imports cleanly without optional gTTS dependency."""
    mod = importlib.import_module(
        "gear_sonic.utils.teleop.vr.quest3_audio_prompts"
    )
    assert hasattr(mod, "PROMPT_TEXTS")
    assert hasattr(mod, "ensure_prompt_audio_files")
    assert hasattr(mod, "audio_dir")


def test_every_calibration_pose_has_a_show_prompt() -> None:
    """Every ``pose_id`` returned by the calibration module must have a
    matching ``show_<pose>`` MP3 key. Drift here = silent prompt on
    headset.
    """
    from gear_sonic.utils.teleop.vr.quest3_audio_prompts import PROMPT_TEXTS

    for pose_id in CALIBRATION_POSE_IDS:
        assert f"show_{pose_id}" in PROMPT_TEXTS, (
            f"missing show_{pose_id} prompt -- WebXR client will fall "
            f"back to speechSynthesis (unreliable on Quest 3)"
        )


def test_every_calibration_pose_has_a_recapture_prompt() -> None:
    """Same contract for recapture prompts."""
    from gear_sonic.utils.teleop.vr.quest3_audio_prompts import PROMPT_TEXTS

    for pose_id in CALIBRATION_POSE_IDS:
        assert f"recapture_{pose_id}" in PROMPT_TEXTS, (
            f"missing recapture_{pose_id} prompt"
        )


def test_status_prompts_present() -> None:
    """Captured / moved-too-much / done / failed statuses must all
    have audio.
    """
    from gear_sonic.utils.teleop.vr.quest3_audio_prompts import PROMPT_TEXTS

    for required in (
        "captured",
        "moved_too_much",
        "calibration_saved",
        "calibration_failed",
        "audio_test",
    ):
        assert required in PROMPT_TEXTS, f"missing {required} prompt"


def test_every_x2_stream_mode_has_a_prompt() -> None:
    """Every X2 StreamMode must have a matching ``mode_<lower>`` MP3
    key. The Quest3ManagerX2 plays one of these on every mode
    transition; a missing key means the operator gets silent audio
    feedback (and we'd only catch it at runtime by ear).
    """
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode
    from gear_sonic.utils.teleop.vr.quest3_audio_prompts import PROMPT_TEXTS

    for mode in StreamMode:
        key = f"mode_{mode.name.lower()}"
        assert key in PROMPT_TEXTS, (
            f"missing {key} prompt -- add it to PROMPT_TEXTS so the "
            f"WebXR client can play /audio/{key}.mp3 on transitions to "
            f"StreamMode.{mode.name}"
        )


def test_recording_lifecycle_prompts_present() -> None:
    """The manager fires ``record_start`` and ``record_save`` on
    X / Y press during ARM_MANIPULATION. Both keys must exist.
    """
    from gear_sonic.utils.teleop.vr.quest3_audio_prompts import PROMPT_TEXTS

    for required in ("record_start", "record_save"):
        assert required in PROMPT_TEXTS, f"missing {required} prompt"


def test_manager_prompt_keys_exported_and_consistent() -> None:
    """The exported ``MANAGER_PROMPT_KEYS`` tuple must match the
    union of mode + recording prompts (plus the modal ``mode_torso_locked``
    cue fired on LOCOMOTION -> ARM_MANIPULATION B-press latch) and stay
    in sync with PROMPT_TEXTS so callers can iterate without hard-coding
    names.
    """
    from gear_sonic.utils.teleop.vr.intent_decoder import StreamMode
    from gear_sonic.utils.teleop.vr.quest3_audio_prompts import (
        MANAGER_PROMPT_KEYS,
        PROMPT_TEXTS,
    )

    expected = {f"mode_{m.name.lower()}" for m in StreamMode} | {
        "mode_torso_locked",
        "torso_frozen",
        "torso_released",
        "record_start",
        "record_save",
    }
    assert set(MANAGER_PROMPT_KEYS) == expected, (
        "MANAGER_PROMPT_KEYS drifted from the StreamMode enum + "
        "recording lifecycle keys (plus the mode_torso_locked latch "
        "cue and the torso_frozen / torso_released R-click toggle "
        "cues); update either the enum or the exported tuple to match"
    )
    for key in MANAGER_PROMPT_KEYS:
        assert key in PROMPT_TEXTS, (
            f"MANAGER_PROMPT_KEYS lists {key} but it's not in "
            f"PROMPT_TEXTS"
        )


def test_audio_dir_under_webxr_app() -> None:
    """The audio cache lives under the WebXR HTTP root so the headset
    can fetch it via ``/audio/<key>.mp3`` from the same origin.
    """
    from gear_sonic.utils.teleop.vr.quest3_audio_prompts import audio_dir

    p = audio_dir()
    assert p.name == "audio"
    assert p.parent.name == "quest3_webxr_app"


def test_ensure_prompt_audio_files_is_idempotent_on_existing_cache(
    tmp_path, monkeypatch
) -> None:
    """If MP3s already exist on disk, a second call should not regenerate
    them (no network needed, runs in CI).
    """
    from gear_sonic.utils.teleop.vr import quest3_audio_prompts as q3a

    # Redirect the cache to a temp dir and pre-populate it with empty
    # placeholders for every key.
    monkeypatch.setattr(q3a, "audio_dir", lambda: tmp_path)
    for key in q3a.PROMPT_TEXTS:
        (tmp_path / q3a.filename_for(key)).write_bytes(b"")

    # Should NOT call gTTS (no network in CI).
    result = q3a.ensure_prompt_audio_files()
    assert set(result.keys()) == set(q3a.PROMPT_TEXTS.keys())
    for path in result.values():
        assert path.is_file()
