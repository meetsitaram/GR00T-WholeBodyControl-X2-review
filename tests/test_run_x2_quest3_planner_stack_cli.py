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
        "--no-deploy", "--no-sonic-checkpoint", "--no-x2-debug-bridge",
    ),
) -> subprocess.CompletedProcess[str]:
    """Invoke the wrapper with ``args`` and capture stdout/stderr.

    We always pass ``--no-deploy`` so the wrapper skips the model-file
    + ``deploy_x2.sh`` checks; those need real ONNX / docker assets we
    can't depend on in a unit test. We also default to
    ``--no-sonic-checkpoint`` so the SONIC tokenizer preflight (which
    requires the cloud-mirrored .pt next to the deploy ONNX) doesn't
    fail on hosts without the checkpoint. And we default to
    ``--no-x2-debug-bridge`` so the split-topology bridge-host gate
    (introduced 2026-06-10 alongside ``--pc2-host``) doesn't short-
    circuit the validation paths these tests actually pin. Tests that
    specifically exercise SONIC plumbing or the bridge-host gate
    override ``extra_default_args`` to drop the opt-out flag and
    supply their own checkpoint / bridge-host arg.

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
        # actually fires. Keep --no-x2-debug-bridge so the
        # split-topology bridge-host gate doesn't intercept first.
        extra_default_args=("--no-deploy", "--no-x2-debug-bridge"),
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


# ---------------------------------------------------------------------------
# Engage-pose preservation plumbing (2026-06-10 follow-up 10).
#
# When ENGAGE_POSE_SUB_PORT is set (env var OR --engage-pose-sub-port
# CLI flag), the wrapper MUST forward --engage-pose-sub-{port,host,
# topic,max-age-ms} (and optionally --engage-preserve-hands) to the
# manager so OFF -> non-OFF mid-VLA snaps the arm freeze to the wire's
# current pose instead of X2 neutral. The forwarding MUST exist in
# BOTH the non-VLA branch (planner+manager+recorder) AND the VLA
# branch (manager+bridge+recorder); the latter is the primary user of
# the feature (shared-autonomy takeover) but the non-VLA branch needs
# it too for the "run VLA in T2, run planner stack in T1" topology
# the operator actually uses.
# ---------------------------------------------------------------------------


def test_wrapper_accepts_preserve_arms_on_engage_cli_flag() -> None:
    """``--preserve-arms-on-engage`` MUST reach ``--validate-only``
    without falling into the unknown-arg branch.

    This is the single operator-facing boolean for the follow-up 10
    UX. Catches a regression where the case statement entry is
    dropped or its spelling drifts; the operator would then see
    ``unknown arg: --preserve-arms-on-engage`` and silently lose
    the arm + hand preservation behaviour for the rest of the
    session (the wrapper bails before reaching the manager).
    """
    res = _run_wrapper([
        "--preserve-arms-on-engage",
        "--no-x2-debug-bridge",
        "--validate-only",
    ])
    assert res.returncode == 0, (
        f"wrapper rejected --preserve-arms-on-engage; got exit "
        f"{res.returncode}.\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    assert "unknown arg" not in res.stderr, (
        f"wrapper case statement dropped --preserve-arms-on-engage.\n"
        f"stderr:\n{res.stderr}"
    )


def test_wrapper_forwards_preserve_arms_args_to_manager_in_both_branches() -> None:
    """Source-level pin: the wrapper MUST forward
    ``--preserve-arms-on-engage`` (plus the four advanced overrides)
    to BOTH manager spawn paths (non-VLA and VLA).

    The non-VLA branch is the primary one (the operator runs VLA in
    a sibling terminal); the VLA branch is the in-process VLA-bridge
    mode (less common but supported). If either branch loses the
    forwarding the operator's --preserve-arms-on-engage silently
    no-ops for that mode -- exactly the kind of "looks like it
    worked" failure the wrapper preflight is supposed to make
    impossible.
    """
    src = WRAPPER.read_text()

    # The gate is "PRESERVE_ARMS_ON_ENGAGE -eq 1" (env var or CLI
    # flag flipping it on); the port is auto-defaulted to 5558.
    # Pin that the gate string survives any future refactor.
    assert "PRESERVE_ARMS_ON_ENGAGE}\" -eq 1 ]]; then" in src, (
        "wrapper no longer gates the engage-pose forwarding behind "
        "PRESERVE_ARMS_ON_ENGAGE=1; the single-boolean UX would "
        "regress to needing a port number again."
    )

    # The numeric guard must still reject non-int / non-positive
    # port overrides so a typo'd ENGAGE_POSE_SUB_PORT env var bails
    # at the wrapper instead of the manager argparse.
    assert "ENGAGE_POSE_SUB_PORT}\" =~ ^[0-9]+$" in src, (
        "wrapper no longer numeric-guards ENGAGE_POSE_SUB_PORT; "
        "a typo'd env value would be passed verbatim to the manager "
        "and crash argparse with an unhelpful error."
    )

    # The two branches sit at different nesting depths (the VLA branch
    # is inside ``if VLA_MODE -eq 1; then``) so the indentation
    # differs. Normalize whitespace to a tight form and count
    # occurrences of the canonical 6-arg body.
    normalized = " ".join(src.split())
    expect_body = (
        "MANAGER_ARGS+=( "
        "--preserve-arms-on-engage "
        "--engage-pose-sub-host \"${ENGAGE_POSE_SUB_HOST}\" "
        "--engage-pose-sub-port \"${ENGAGE_POSE_SUB_PORT}\" "
        "--engage-pose-sub-topic \"${ENGAGE_POSE_SUB_TOPIC}\" "
        "--engage-pose-sub-max-age-ms \"${ENGAGE_POSE_SUB_MAX_AGE_MS}\" "
        ")"
    )
    occurrences = normalized.count(expect_body)
    assert occurrences == 2, (
        f"preserve-arms MANAGER_ARGS forwarding block must appear in BOTH "
        f"the non-VLA branch AND the VLA branch (expected 2, got "
        f"{occurrences}). If only one branch has it, the manager will "
        f"silently fall back to X2-neutral snapping in the missing mode."
    )

    # Pin the single-flag UX: the separate --engage-preserve-hands
    # flag was retired. If a refactor re-introduces it as a separate
    # opt-in the operator gets back the confusing two-flag surface.
    assert "--engage-preserve-hands" not in src, (
        "wrapper resurfaced --engage-preserve-hands as a separate "
        "opt-in; hands now ride along with --preserve-arms-on-engage "
        "(single-flag UX). Either remove the flag or update this "
        "pin if the two-flag surface is intentional."
    )
    assert "ENGAGE_PRESERVE_HANDS" not in src, (
        "wrapper resurfaced the ENGAGE_PRESERVE_HANDS env var; the "
        "single-flag UX retired this opt-in. Update this pin if the "
        "env var is intentional."
    )


# ---------------------------------------------------------------------------
# 2026-06-11 regression: --no-x2-debug-bridge must survive --pc2-host
# ---------------------------------------------------------------------------
def test_no_x2_debug_bridge_survives_pc2_host_auto_resolution() -> None:
    """Source-level pin for the 2026-06-11 fix to a UX bug where
    ``--no-x2-debug-bridge`` was silently flipped back on whenever
    ``--pc2-host`` was also passed.

    The bug:
      1. CLI parser correctly sets WITH_X2_DEBUG_BRIDGE=0 on
         ``--no-x2-debug-bridge``.
      2. Later, the ``--pc2-host`` fan-out block at ~line 1020
         unconditionally re-set WITH_X2_DEBUG_BRIDGE=1 whenever
         X2_DEBUG_BRIDGE_HOST was empty -- which it always is when
         the operator only passed --pc2-host (no explicit bridge
         host override).
      3. Net effect: ``--no-x2-debug-bridge --pc2-host PC2_IP`` spawned
         the x2_debug bridge anyway and clashed on :5570 with the
         VLA bridge's --enable-takeover internal port (back when
         BRIDGE_POSE_PORT_INTERNAL was 5570) -- producing
         ``zmq.error.ZMQError: Address already in use (addr='tcp://*:5570')``
         even though the operator had explicitly asked for the bridge
         to be skipped.

    Pin the fix: the auto-resolution must consult the explicit opt-out
    before flipping the flag back on. We grep for the guard rather than
    drive the wrapper to validate-only because the fan-out block runs
    early enough that the test harness doesn't reach the bridge spawn
    in unit-test mode anyway.
    """
    src = WRAPPER.read_text()
    expect = '"${WITH_X2_DEBUG_BRIDGE}" != "0"'
    assert expect in src, (
        "auto-resolution of X2_DEBUG_BRIDGE_HOST under --pc2-host must "
        "guard on WITH_X2_DEBUG_BRIDGE != 0; without that guard, "
        "--no-x2-debug-bridge silently no-ops when --pc2-host is also "
        "passed (2026-06-11 fix). Restore the guard."
    )
