"""CLI-validation smoke tests for ``run_x2_quest3_planner_stack.sh``.

These tests don't launch the real stack -- the wrapper spawns a docker
container, an ONNX policy, a Quest 3 webserver, etc. -- so they can't
be run in a normal CI environment without a GPU + an Oculus headset.

Instead they exercise the wrapper's *argument-validation* layer, which
is pure bash and runs entirely on the host. They:

  * ``bash -n`` the script (catches syntax regressions immediately).
  * Pass deliberately bogus arguments and assert the wrapper bails
    out with the right exit code AND the right operator-facing error
    message in stderr (so the operator gets a fix-it hint, not just
    a stack trace).
  * Verify the new robocasa-mode plumbing (added 2026-05-13):
      - ``--robocasa-env BadName`` is rejected with the "must be one
        of ..." message,
      - ``--robocasa-env <env> --scene-xml-path /nonexistent`` is
        rejected with the build-script hint,
      - ``--with-record`` without ``--task`` AND without
        ``--robocasa-env`` is rejected with the "requires --task"
        message (existing behaviour, preserved through the refactor),
      - ``--with-record --robocasa-env <valid env>`` *without* a
        ``--task`` is **not** rejected at the task-validation step
        (the recorder auto-fills it from the scene metadata).

Why argparse-level coverage matters for a bash wrapper: the wrapper
is the operator's single entrypoint to the Phase 0 stack. A
copy-paste mistake (typo'd env name, swapped flag order, forgotten
``--task``) used to silently boot the stack into the wrong
configuration -- the deploy would launch with a flat-floor MJCF, the
recorder would attempt to instantiate a ``RobocasaTaskMirror`` for a
scene the deploy isn't simulating, and the operator would only notice
when the recorded ego-view frames showed the floor instead of a
table. These tests pin the validation contract so any future
refactor that breaks an error-message keyword fails loudly here
before it reaches an operator.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = REPO_ROOT / "gear_sonic" / "scripts" / "run_x2_quest3_planner_stack.sh"


def _run_wrapper(
    args: list[str], *, timeout_s: float = 30.0,
    extra_default_args: tuple[str, ...] = (
        "--no-deploy", "--no-sonic-checkpoint",
    ),
) -> subprocess.CompletedProcess[str]:
    """Invoke the wrapper with ``args`` and capture stdout/stderr.

    We always pass ``--no-deploy`` so the wrapper skips the model-file
    + ``deploy_x2.sh`` checks; those need real ONNX / docker assets we
    can't depend on in a unit test. We also default to
    ``--no-sonic-checkpoint`` so the SONIC tokenizer preflight (which
    requires the cloud-mirrored .pt next to the deploy ONNX) doesn't
    fail on hosts without the checkpoint. Tests that specifically
    exercise SONIC plumbing override ``extra_default_args`` to drop
    the opt-out flag and supply their own checkpoint path.

    The validation paths exercised in most tests all happen *after*
    these short-circuits, so the omissions are invisible to the test.

    A 30 s timeout guards against a regression where validation
    accidentally falls through into the port-check / docker-preflight
    paths -- if that ever happens, ``Popen`` would hang on the
    docker-stop calls. The timeout fails the test loudly instead.
    """
    cmd = ["bash", str(WRAPPER), *extra_default_args, *args]
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout_s,
        env={**os.environ, "X2_PLANNER_SMOKE_MODEL": ""},
    )


# ---------------------------------------------------------------------------
# Smoke / sanity
# ---------------------------------------------------------------------------


def test_wrapper_script_exists() -> None:
    """Sanity: the wrapper file is on disk and executable."""
    assert WRAPPER.is_file(), f"wrapper not found at {WRAPPER}"
    assert os.access(WRAPPER, os.X_OK), f"wrapper not executable: {WRAPPER}"


def test_wrapper_passes_bash_syntax_check() -> None:
    """``bash -n`` catches syntax errors / unterminated heredocs / etc.

    This is by far the most common regression mode for the wrapper:
    edits to the heredoc banner or the case statement can leave a
    dangling ``;;`` or an unmatched ``EOF`` and the wrapper still
    "parses" until shell tries to execute it. ``bash -n`` parses
    without running, so it catches those before the operator does.
    """
    res = subprocess.run(
        ["bash", "-n", str(WRAPPER)], capture_output=True, text=True,
    )
    assert res.returncode == 0, (
        f"bash -n exited {res.returncode}\nstderr:\n{res.stderr}"
    )


# ---------------------------------------------------------------------------
# Robocasa env validation
# ---------------------------------------------------------------------------


def test_unknown_robocasa_env_is_rejected_with_helpful_message() -> None:
    """``--robocasa-env BadName`` must exit non-zero and list the
    valid choices in stderr so the operator can fix the typo without
    grepping the source.
    """
    res = _run_wrapper(["--robocasa-env", "BadEnvName"])

    assert res.returncode != 0, (
        f"expected non-zero exit for bad env, got 0\n"
        f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    # The wrapper should explicitly enumerate the supported envs.
    # Don't pin the exact wording -- match on the stable keywords so
    # an editorial cleanup of the error message doesn't break the
    # test, but a regression that drops the env list does.
    assert "--robocasa-env" in res.stderr
    assert "X2PickPlaceCube" in res.stderr
    assert "X2PickPlaceBowl" in res.stderr
    assert "BadEnvName" in res.stderr  # echo the offender back


def test_missing_scene_xml_is_rejected_with_build_script_hint(
    tmp_path: Path,
) -> None:
    """``--robocasa-env <env> --scene-xml-path /nonexistent.xml``
    must reject and point the operator at ``build_x2_robocasa_scene_xml``.

    This catches the most common operator failure mode: forgetting
    to build the scene XML before the first record session. Without
    the hint the operator would otherwise hit a far less actionable
    error inside the recorder when it tries to load the MJCF.
    """
    bogus_xml = tmp_path / "definitely_not_a_real_scene.xml"
    res = _run_wrapper([
        "--robocasa-env", "X2PickPlaceCube",
        "--scene-xml-path", str(bogus_xml),
    ])

    assert res.returncode != 0
    assert str(bogus_xml) in res.stderr
    assert "build_x2_robocasa_scene_xml" in res.stderr


# ---------------------------------------------------------------------------
# --with-record / --task interaction
# ---------------------------------------------------------------------------


def test_with_record_requires_task_outside_robocasa_mode(
    tmp_path: Path,
) -> None:
    """Outside robocasa mode the operator MUST pass ``--task`` --
    the LeRobot v2.1 schema requires a task string per episode and
    we don't want to silently default it.
    """
    res = _run_wrapper([
        "--with-record",
        "--output-dir", str(tmp_path / "out"),
    ])

    assert res.returncode != 0
    # Stable substring: don't break the test if the message is
    # reworded for clarity, but do break if the entire requirement
    # is dropped.
    assert "--task" in res.stderr
    assert "--with-record" in res.stderr


def test_with_record_in_robocasa_mode_does_not_require_task(
    tmp_path: Path,
) -> None:
    """In robocasa mode the recorder auto-fills the task string from
    the env's canonical instruction (the success oracle is grading
    against that exact text), so the wrapper must NOT reject a
    ``--with-record`` without an explicit ``--task``.

    Uses ``--validate-only`` so the wrapper exits cleanly right after
    pre-flight, *before* it spawns the planner / manager / recorder.
    Without that flag the wrapper would race past validation, spawn
    three Python children with ``setsid``, and a 30 s pytest timeout
    would SIGKILL the bash parent -- leaving the children orphaned
    under PID 1, holding ports 5556/5560/5563/5564/5565 until the
    operator hunts them down with ``lsof``. (We learned this the
    hard way 2026-05-13 when three back-to-back launches all
    reported "port 5556 in use" because of exactly this leak from
    pre-``--validate-only`` runs of this very test.)

    On success the wrapper exits 0 with the ``validate-only:``
    banner; we assert the task-required error is NOT in stderr.
    """
    scene_xml = (
        REPO_ROOT / "gear_sonic" / "data" / "assets"
        / "robocasa_scenes" / "X2PickPlaceCube.xml"
    )
    if not scene_xml.is_file():
        pytest.skip(
            f"X2PickPlaceCube.xml not built locally ({scene_xml}); "
            "run `python -m gear_sonic.scripts.build_x2_robocasa_scene_xml "
            "--env X2PickPlaceCube` to enable this test."
        )

    res = _run_wrapper([
        "--validate-only",
        "--with-record",
        "--output-dir", str(tmp_path / "out"),
        "--robocasa-env", "X2PickPlaceCube",
    ])

    # Validate-only must exit 0 (validation passed) AND must not
    # have rejected on the task gate. We split these into two
    # assertions so a regression on either prong fails with a
    # specific message instead of "task gate maybe regressed,
    # also exit code wrong".
    assert "requires --task" not in res.stderr, (
        f"wrapper rejected --with-record + --robocasa-env without "
        f"--task -- task auto-fill from scene metadata regressed.\n"
        f"stderr was:\n{res.stderr}"
    )
    assert res.returncode == 0, (
        f"--validate-only exited {res.returncode} for a config we "
        f"expect to pass pre-flight.\nstderr:\n{res.stderr}"
    )
    # Sanity: the validate-only banner should fire, proving we
    # short-circuited at the right place (after banner, before any
    # spawn). If the message moves, update both this assertion and
    # the wrapper.
    assert "validate-only" in res.stdout or "validate-only" in res.stderr


# ---------------------------------------------------------------------------
# Help / banner / usage
# ---------------------------------------------------------------------------


def test_validate_only_exits_clean_without_spawning() -> None:
    """``--validate-only`` must exit 0 after pre-flight + banner
    without spawning any of the four child processes.

    This is the contract the rest of the test suite relies on (see
    ``test_with_record_in_robocasa_mode_does_not_require_task``).
    If the short-circuit ever moves so it runs *after* the first
    spawn, every test in this file that uses ``--validate-only``
    starts leaking processes and the next operator launch fails
    with "port in use". Pin the contract here.
    """
    res = _run_wrapper(["--validate-only"])

    assert res.returncode == 0, (
        f"--validate-only exited {res.returncode} on the simplest "
        f"happy path.\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    # The validate-only message itself proves we ran past pre-flight
    # and reached the short-circuit (rather than e.g. exiting 0 from
    # --cleanup-only by mistake).
    assert "validate-only" in res.stdout or "validate-only" in res.stderr


def test_help_mentions_robocasa_flag() -> None:
    """``-h`` / ``--help`` should advertise ``--robocasa-env`` so the
    operator can discover the new flag without reading the source.
    """
    res = subprocess.run(
        ["bash", str(WRAPPER), "--help"],
        capture_output=True, text=True, timeout=10.0,
    )
    # The wrapper prints help to stderr and exits 1 (matches the
    # rest of the help-on-bad-arg pattern in this codebase).
    assert "--robocasa-env" in res.stderr, (
        f"--help output missing --robocasa-env advertisement.\n"
        f"stderr:\n{res.stderr}"
    )
    # Sanity: examples block survived the edit.
    assert "X2PickPlaceCube" in res.stderr or "robocasa-env" in res.stderr


# ---------------------------------------------------------------------------
# v7.2: --recorder-enabled audio-cue gate plumbing.
#
# We can't introspect MANAGER_ARGS at runtime without spawning the
# manager (``--validate-only`` exits before the array is built), so
# these are source-level smoke tests: the wrapper must mention BOTH
# branches (``--recorder-enabled`` for --with-record runs, and
# ``--no-recorder-enabled`` for the teleop-only default), and they
# must be conditional on ``WITH_RECORD``. If a refactor drops one of
# these the manager will silently regress to "always play the audio
# cue", which is the false-ACK trap v7.2 set out to fix.
# ---------------------------------------------------------------------------


def test_wrapper_forwards_recorder_enabled_iff_with_record() -> None:
    """The wrapper must populate MANAGER_ARGS with --recorder-enabled
    when WITH_RECORD=1 and --no-recorder-enabled otherwise. Source-level
    pin; the runtime behaviour is covered by the manager unit tests
    (test_quest3_manager_x2_wire_format::test_*recorder_enabled*).
    """
    src = WRAPPER.read_text()
    assert "MANAGER_ARGS+=(--recorder-enabled)" in src, (
        "wrapper no longer forwards --recorder-enabled to the manager; "
        "the v7.2 audio-cue gate will silently break and the headset "
        "will start lying again in --teleop-only sessions."
    )
    assert "MANAGER_ARGS+=(--no-recorder-enabled)" in src, (
        "wrapper no longer explicitly passes --no-recorder-enabled in "
        "the teleop-only branch. The CLI default is False so behaviour "
        "is currently OK, but the explicit form makes the intent clear "
        "and protects against a future change to the manager default."
    )
    # The two forms must live under a WITH_RECORD conditional. Looking
    # for the literal block protects against a refactor that splits
    # them across two unrelated ifs.
    expect_block = (
        "if [[ \"${WITH_RECORD}\" -eq 1 ]]; then\n"
        "    MANAGER_ARGS+=(--recorder-enabled)\n"
        "else\n"
        "    MANAGER_ARGS+=(--no-recorder-enabled)\n"
        "fi"
    )
    assert expect_block in src, (
        "the WITH_RECORD->recorder-enabled forwarding block has been "
        "rewritten; double-check the v7.2 audio-gate semantics still "
        "hold and update this assertion to match the new layout."
    )


# ---------------------------------------------------------------------------
# Inline SONIC FSQ tokenizer plumbing.
#
# The recorder needs the SONIC tokenizer .pt to produce the
# ground-truth ``action.motion_token`` column the VLA trains against;
# without it every recorded frame is zeros and the dataset is not
# VLA-trainable. The wrapper auto-resolves the .pt as a sibling of the
# deploy ONNX. These tests pin (a) the auto-resolution string transform,
# (b) the missing-.pt preflight error, and (c) the ``--no-sonic-checkpoint``
# escape hatch so future refactors don't silently re-introduce the
# zero-token gap.
# ---------------------------------------------------------------------------


def test_no_sonic_checkpoint_passes_validate_only() -> None:
    """``--no-sonic-checkpoint`` must let the wrapper skip the .pt
    preflight and reach ``--validate-only``.

    The CLI test suite relies on this escape hatch so it doesn't have
    to ship the cloud-mirrored 398 MB SONIC checkpoint to CI hosts.
    The escape is also the documented smoke-test mode for kinematic-
    only recordings (they're explicit-non-VLA-trainable on purpose).
    """
    res = _run_wrapper(["--validate-only"])
    assert res.returncode == 0, (
        f"--validate-only --no-sonic-checkpoint must succeed; got "
        f"{res.returncode}.\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )


def test_no_sonic_checkpoint_banner_marks_dataset_disabled() -> None:
    """When ``--no-sonic-checkpoint`` is set the banner must say so.

    The banner is the operator's only feedback that the dataset will
    NOT be VLA-trainable. If the wording silently changes, an operator
    can produce hundreds of zero-token episodes thinking they're
    training-ready -- exactly the failure mode the wrapper preflight
    is supposed to make impossible.
    """
    res = _run_wrapper(["--validate-only"])
    assert "DISABLED" in res.stdout, (
        "banner must surface the DISABLED state for action.motion_token "
        "when --no-sonic-checkpoint is set; otherwise the operator has "
        "no warning that the dataset is not VLA-trainable.\n"
        f"stdout:\n{res.stdout}"
    )


def test_missing_sonic_checkpoint_in_auto_mode_is_rejected() -> None:
    """Auto-resolution failure must abort BEFORE ``--validate-only``
    runs, with a fix-it hint.

    We simulate the failure by pointing ``--model`` at a path whose
    sibling ``.pt`` does not exist (every path under /tmp qualifies),
    omitting ``--no-sonic-checkpoint``, and asserting the wrapper
    exits non-zero with the documented operator-facing message.
    """
    bogus_onnx = "/tmp/__never_exists__/exported/model_step_999_g1.onnx"
    res = _run_wrapper(
        [
            "--model", bogus_onnx,
            "--validate-only",
        ],
        # Drop the default --no-sonic-checkpoint so the preflight
        # actually fires.
        extra_default_args=("--no-deploy",),
    )
    assert res.returncode != 0, (
        "wrapper must reject auto-resolved .pt that doesn't exist; "
        f"got exit {res.returncode}.\nstdout:\n{res.stdout}\n"
        f"stderr:\n{res.stderr}"
    )
    # Operator-facing remediation. If any of these phrases moves the
    # operator stops getting an actionable error and the test starts
    # passing for the wrong reason -- pin the wording.
    err = res.stderr
    assert "SONIC tokenizer .pt not found" in err, (
        f"missing the SONIC preflight error preamble.\nstderr:\n{err}"
    )
    assert "--sonic-checkpoint" in err, (
        f"missing the --sonic-checkpoint fix-it hint.\nstderr:\n{err}"
    )
    assert "--no-sonic-checkpoint" in err, (
        f"missing the --no-sonic-checkpoint escape-hatch hint.\nstderr:\n{err}"
    )


def test_wrapper_source_auto_resolves_pt_from_onnx_path() -> None:
    """Source-level pin for the auto-resolution transform.

    Strip ``/exported/`` and replace the ``_g1.onnx`` suffix with
    ``.pt``. We can't introspect the runtime variable without going
    through ``--validate-only`` (which the auto-mode test above does
    indirectly), so this guards the bash transform itself against
    regression.
    """
    src = WRAPPER.read_text()
    # The transform must use both substitutions in the documented
    # order; if a refactor swaps either of them the auto-resolution
    # silently produces a wrong path that still 'looks plausible'.
    assert "SIM_MODEL/\\/exported\\//" in src, (
        "wrapper no longer strips /exported/ from the ONNX path; "
        "auto-resolution will fall back to the ONNX path and fail "
        "the preflight or (worse) load the ONNX as if it were a .pt."
    )
    assert "_g1.onnx}.pt" in src, (
        "wrapper no longer rewrites the _g1.onnx suffix to .pt; "
        "auto-resolution will produce e.g. *_g1.onnx.pt which is "
        "guaranteed to not exist on disk."
    )


def test_wrapper_forwards_sonic_flags_to_recorder() -> None:
    """When SONIC tokenizer is ON, both --sonic-checkpoint and
    --sonic-tokenizer-device must reach the recorder spawn args.

    Source-level pin; runtime behaviour is covered by
    ``tests/test_x2_dataset_recorder_motion_token.py``. If the
    forwarding block ever changes, the recorder will silently fall
    back to its CLI defaults (which are themselves correct, but the
    operator's --sonic-tokenizer-device cpu override would silently
    no-op -- exactly the kind of "looks like it worked" failure the
    wrapper is supposed to prevent).
    """
    src = WRAPPER.read_text()
    expect_block = (
        "if [[ -n \"${SONIC_CHECKPOINT}\" ]]; then\n"
        "    RECORDER_ARGS+=(\n"
        "        --sonic-checkpoint \"${SONIC_CHECKPOINT}\"\n"
        "        --sonic-tokenizer-device \"${SONIC_TOKENIZER_DEVICE}\"\n"
        "    )\n"
        "fi"
    )
    assert expect_block in src, (
        "the SONIC_CHECKPOINT->recorder forwarding block has been "
        "rewritten; double-check both --sonic-checkpoint AND "
        "--sonic-tokenizer-device still ride together so the device "
        "override isn't silently dropped."
    )
