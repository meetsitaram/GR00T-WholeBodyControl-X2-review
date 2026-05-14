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
) -> subprocess.CompletedProcess[str]:
    """Invoke the wrapper with ``args`` and capture stdout/stderr.

    We always pass ``--no-deploy`` so the wrapper skips the model-file
    + ``deploy_x2.sh`` checks; those need real ONNX / docker assets we
    can't depend on in a unit test. The validation paths exercised
    here all happen *after* the no-deploy short-circuit on the model
    file, so the omission is invisible to the test.

    A 30 s timeout guards against a regression where validation
    accidentally falls through into the port-check / docker-preflight
    paths -- if that ever happens, ``Popen`` would hang on the
    docker-stop calls. The timeout fails the test loudly instead.
    """
    cmd = ["bash", str(WRAPPER), "--no-deploy", *args]
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
