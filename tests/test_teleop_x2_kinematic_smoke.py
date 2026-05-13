"""
Smoke tests for ``gear_sonic.scripts.teleop_x2_kinematic``.

These are *offline* tests -- they never connect to a Quest 3, never open the
MuJoCo viewer, and never call any GPU/MuJoCo path. They exist to catch
breakage in the kinematic teleop's two failure-prone glue points:

* ``Gr00tDataExporter.create()`` kwargs match what the exporter actually
  accepts. The script went through two bring-up failures (one
  ``embodiment_tag`` typo, one half-broken-output-dir trap) before the
  operator could even press B; both would have been caught here.
* ``--output-dir`` preflight handles "fresh", "resume", "auto-clean
  stub", and "occupied non-dataset dir" cases without crashing the
  recorder mid-session.

Why it matters: a failed teleop bring-up costs the operator ~30 s of
headset boot + a partial dataset that often poisons the next attempt.
Cheap CI-able preflight tests are the right place to fail.
"""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

import pytest

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_x2_vla import (
    HAND_DOF_OMNI,
    get_features_x2_vla,
    get_modality_config_x2_vla,
    get_x2_robot_model,
)
from gear_sonic.scripts.teleop_x2_kinematic import _preflight_output_dir


# ---------------------------------------------------------------------------
# Exporter-call-site contract (would have caught the embodiment_tag typo)
# ---------------------------------------------------------------------------


def _exporter_kwargs_for_kinematic_teleop(output_dir: Path) -> dict:
    """Return the exact kwargs the live ``main()`` passes to ``Gr00tDataExporter.create``.

    Kept in lockstep with ``gear_sonic/scripts/teleop_x2_kinematic.py`` --
    if main() changes its kwargs, update this builder so the contract
    test below stays a meaningful guard.
    """
    rm = get_x2_robot_model(hand_variant="omnihand_10")
    features = get_features_x2_vla(rm, hand_dof_per_side=HAND_DOF_OMNI)
    modality_cfg = get_modality_config_x2_vla(rm, hand_dof_per_side=HAND_DOF_OMNI)
    return {
        "save_root": output_dir,
        "fps": 50,
        "features": features,
        "modality_config": modality_cfg,
        "task": "smoke-test",
        "script_config": {
            "robot_type": "agibot_x2_ultra",
            "embodiment_tag": "new_embodiment",
            "hand_variant": "omnihand_10",
            "num_body_joints": rm.num_joints,
            "hand_dof_per_side": HAND_DOF_OMNI,
            "fps": 50,
            "teleop_mode": "kinematic",
        },
        "robot_type": "agibot_x2_ultra",
    }


def test_exporter_kwargs_are_all_accepted(tmp_path: Path) -> None:
    """Every kwarg the kinematic teleop passes must be in ``create()``'s signature.

    This is the test that would have caught
    ``TypeError: Gr00tDataExporter.create() got an unexpected keyword
    argument 'embodiment_tag'`` at CI time instead of after a 30-second
    Quest 3 boot.
    """
    sig = inspect.signature(Gr00tDataExporter.create)
    accepted = set(sig.parameters.keys()) - {"cls"}
    used = set(_exporter_kwargs_for_kinematic_teleop(tmp_path / "out").keys())
    bad = used - accepted
    assert not bad, (
        f"teleop_x2_kinematic passes {bad} to Gr00tDataExporter.create() "
        f"but the exporter only accepts {sorted(accepted)}. The most "
        f"likely fix is to move the offending key into the script_config "
        f"dict (see build_x2_sample_episode.py for the canonical pattern)."
    )


def test_exporter_can_be_constructed_with_kinematic_teleop_kwargs(
    tmp_path: Path,
) -> None:
    """End-to-end: constructing the exporter with the script's exact kwargs
    must succeed against a fresh output directory.

    This simultaneously covers the embodiment_tag-style kwarg bugs AND
    the half-broken-output-dir trap (because we point at a non-existent
    leaf directory, the exporter's "fresh dataset" code path runs).
    """
    out_dir = tmp_path / "x2_kinematic_smoke"
    assert not out_dir.exists()  # critical: must NOT pre-create
    kwargs = _exporter_kwargs_for_kinematic_teleop(out_dir)

    exporter = Gr00tDataExporter.create(**kwargs)
    try:
        assert (out_dir / "meta" / "info.json").is_file(), (
            "Gr00tDataExporter.create() should have laid down meta/info.json "
            "for a fresh dataset; if this fails, the script_config kwargs "
            "didn't hit the create-fresh branch."
        )
    finally:
        # Clean up video writers / file handles so pytest tmp cleanup
        # works on every OS.
        try:
            exporter.stop_video_writers()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Output-dir preflight (would have caught the "corrupted dataset" trap)
# ---------------------------------------------------------------------------


def test_preflight_accepts_missing_output_dir(tmp_path: Path) -> None:
    """A non-existent path is fine; the exporter creates it on first write."""
    out = tmp_path / "fresh"
    assert not out.exists()
    _preflight_output_dir(out)
    assert not out.exists(), "preflight must NOT pre-create the leaf dir"


def test_preflight_accepts_valid_existing_dataset(tmp_path: Path) -> None:
    """A path with ``meta/info.json`` AND at least one parquet under
    ``data/`` is treated as resume-mode and left alone.
    """
    out = tmp_path / "real_dataset"
    (out / "meta").mkdir(parents=True)
    (out / "meta" / "info.json").write_text("{}")
    (out / "data" / "chunk-000").mkdir(parents=True)
    # Any parquet file under data/ counts as "at least one finalized
    # episode"; we don't need to write a valid one for the preflight.
    (out / "data" / "chunk-000" / "episode_000000.parquet").write_text("stub")

    _preflight_output_dir(out)

    assert (out / "meta" / "info.json").is_file()
    assert (out / "data" / "chunk-000" / "episode_000000.parquet").is_file()


def test_preflight_auto_cleans_half_init_stub_with_info_json(tmp_path: Path) -> None:
    """A path with ``meta/info.json`` but ZERO parquet under ``data/`` is
    a half-init stub from a previous run that crashed before the first
    episode finalized. Auto-clean instead of letting the exporter trip
    on its HF-Hub resume path (404 for ``tmp/tmp_dataset``).
    """
    out = tmp_path / "x2_quest3_kinematic_v0"
    (out / "meta").mkdir(parents=True)
    (out / "meta" / "info.json").write_text('{"codebase_version": "v2.1"}')
    (out / "meta" / "modality.json").write_text("{}")
    # Pre-created videos/ tree from exporter init, but no MP4s yet.
    (out / "videos" / "chunk-000" / "observation.images.ego_view").mkdir(
        parents=True
    )
    # Note: NO data/ subdir at all (that's how the real bug looks).

    _preflight_output_dir(out)
    assert not out.exists(), "half-init stub should be cleaned up"


def test_preflight_auto_cleans_half_init_stub_with_empty_data_dir(
    tmp_path: Path,
) -> None:
    """Same as above but ``data/`` exists and is empty (also half-init)."""
    out = tmp_path / "x2_quest3_kinematic_v0"
    (out / "meta").mkdir(parents=True)
    (out / "meta" / "info.json").write_text('{"codebase_version": "v2.1"}')
    (out / "data" / "chunk-000").mkdir(parents=True)
    # Empty data/chunk-000 -- no .parquet inside.

    _preflight_output_dir(out)
    assert not out.exists(), "half-init stub with empty data/ should be cleaned up"


def test_preflight_auto_cleans_empty_stub(tmp_path: Path) -> None:
    """An empty leaf dir (created by a previous failed bring-up) gets removed."""
    out = tmp_path / "x2_quest3_kinematic_v0"
    out.mkdir()

    _preflight_output_dir(out)
    assert not out.exists(), "empty stub dir should be cleaned up"


def test_preflight_auto_cleans_stale_debug_only_dir(tmp_path: Path) -> None:
    """A dir that contains ONLY a stale ``debug/`` is also auto-cleaned.

    Mirrors the real-world failure mode: previous crashed run wrote a
    debug NPZ but never reached exporter init.
    """
    out = tmp_path / "stale_run"
    (out / "debug").mkdir(parents=True)
    (out / "debug" / "teleop_episode_000000.npz").write_text("not really npz")

    _preflight_output_dir(out)
    assert not out.exists(), "debug-only dir should be cleaned up"


def test_preflight_refuses_unrecognized_dir_contents(tmp_path: Path) -> None:
    """A dir with real content but no ``meta/info.json`` is NOT auto-cleaned.

    We don't want the recorder to clobber the operator's data on accident.
    """
    out = tmp_path / "operator_workdir"
    (out / "data").mkdir(parents=True)
    (out / "important.txt").write_text("operator content")

    with pytest.raises(SystemExit) as excinfo:
        _preflight_output_dir(out)
    msg = str(excinfo.value)
    assert "not a valid LeRobot dataset" in msg
    assert "rm -rf" in msg, "preflight error should be actionable"
    # Side-effect free on the operator's content.
    assert (out / "important.txt").is_file()


# ---------------------------------------------------------------------------
# Argparse / --help (cheap import sanity)
# ---------------------------------------------------------------------------


def test_parse_args_help_does_not_crash() -> None:
    """``python -m gear_sonic.scripts.teleop_x2_kinematic --help`` must succeed."""
    import gear_sonic.scripts.teleop_x2_kinematic as mod

    with pytest.raises(SystemExit) as excinfo:
        mod._parse_args(["--help"])
    assert excinfo.value.code == 0


def test_parse_args_requires_task_with_output_dir(tmp_path: Path) -> None:
    """``--output-dir`` without ``--task`` should fail at main() preflight."""
    import gear_sonic.scripts.teleop_x2_kinematic as mod

    out = tmp_path / "smoke_out"

    with pytest.raises(SystemExit) as excinfo:
        mod.main(["--output-dir", str(out), "--rate", "50"])
    msg = str(excinfo.value)
    assert "task" in msg.lower(), f"unexpected error: {msg}"


# ---------------------------------------------------------------------------
# Calibration loading: missing file must fail fast with an actionable msg
# ---------------------------------------------------------------------------


def test_load_or_recalibrate_missing_calibration_raises(tmp_path: Path) -> None:
    """When ``--calibration`` points at a non-existent YAML and
    ``--recalibrate`` is NOT set, ``_load_or_recalibrate`` must raise
    SystemExit with a hint pointing the operator at
    ``vr_operator_calibrate.py``.

    This is the test that catches "operator runs teleop before
    capturing a calibration" before we boot Quest 3 / MuJoCo.
    """
    import gear_sonic.scripts.teleop_x2_kinematic as mod

    args = mod._parse_args(
        ["--calibration", str(tmp_path / "does_not_exist.yaml")]
    )
    assert args.recalibrate is False

    # Pass a dummy quest reader; load_or_recalibrate should never
    # touch it on the missing-file path.
    class _StubQuest:
        pass

    with pytest.raises(SystemExit) as excinfo:
        mod._load_or_recalibrate(_StubQuest(), args)
    msg = str(excinfo.value)
    assert "calibration file not found" in msg.lower()
    assert "vr_operator_calibrate" in msg, "error must point at the capture script"
