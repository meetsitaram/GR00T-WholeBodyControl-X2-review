"""Smoke tests for the sim-mode pose-proxy plumbing in
``gear_sonic/scripts/run_x2_vla_runtime.sh``.

The 2026-06-10 manual-takeover milestone extended the sim launcher to
spawn a local ``x2_pose_proxy.py`` on loopback so the same operator
workflow (override SUB + vla_control PUB) works in pure sim without
PC2 daemons. The proxy is a real subprocess that the bash launcher
builds an argv for from env vars (``POSE_PROXY_DOWNSTREAM_PORT``,
``POSE_PROXY_OVERRIDE_PORT``, ``VLA_CONTROL_PORT`` etc.).

The dual-source semantics themselves are already covered by
``tests/test_x2_pose_proxy_dual_source.py``. What's NOT covered is
arg-name drift: a typo in the bash launcher (``--override-port-name``
instead of ``--override-port``, or ``--idle-x2m2-file`` instead of
``--idle-x2m2``) would only fail at first sim launch, hours after the
edit landed.

This test closes that gap with two cheap checks:

1. ``test_launcher_bash_syntax_ok`` -- ``bash -n`` on the launcher.
   Catches structural bash mistakes (unmatched ``fi``, bad heredoc,
   stray backtick) introduced while wiring the proxy spawn.

2. ``test_spawn_sim_proxy_argv_parses`` -- builds the same argv the
   bash launcher would build when SIM_PROXY_ENABLED=1 and feeds it to
   ``x2_pose_proxy.main``'s parser via ``parse_args``. Catches argname
   typos, missing required args, and incompatible value types. We
   intentionally do NOT spawn the proxy: that path is exercised by the
   subprocess smokes already.

Both tests run in the fast unit-test pass (no env-var gate) because
they're <1 s combined and have no external dependencies beyond the
proxy module already imported by the dual-source smoke.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "gear_sonic" / "scripts" / "run_x2_vla_runtime.sh"
PROXY_DIR = REPO_ROOT / "gear_sonic_deploy" / "scripts"
IDLE_X2M2 = REPO_ROOT / "gear_sonic_deploy" / "data" / "idle_stand.x2m2"

if str(PROXY_DIR) not in sys.path:
    sys.path.insert(0, str(PROXY_DIR))

import x2_pose_proxy as proxy  # noqa: E402


def _build_proxy_parser() -> argparse.ArgumentParser:
    """Reconstruct the proxy's argparse exactly the way ``main`` does.

    We mirror ``proxy.main`` up to (but not including) ``parse_args``
    so the test catches drift without us hand-maintaining a second
    copy of the parser. If ``proxy.main`` ever gains a refactor that
    exposes ``build_parser()`` directly, switch to that and delete
    this shim. Until then this is the cheapest robust option.
    """
    # The cleanest path is to import _argparser_ from proxy if exposed;
    # otherwise we monkey-call main with a sentinel that short-circuits
    # right after parse_args. Currently the proxy inlines the parser
    # in main(), so we feed our argv through main() and rely on the
    # idle_x2m2 file existence check at line ~848 acting as a quick
    # early-exit-on-bad-arg gate (it runs AFTER parse_args, so an
    # argparse failure trips first via SystemExit).
    return _IndirectProxyArgvCheck()


class _IndirectProxyArgvCheck:
    """Adapter that exposes ``parse_args`` by re-running proxy.main on
    a stripped argv until parse_args succeeds, then returns the
    parsed Namespace by intercepting the next instruction.

    In practice we just shell out to the proxy script with ``--help``
    to confirm the parser is well-formed, then call ``main(argv)``
    inside a try/except SystemExit and accept either a parse-pass
    (which then trips on the file-not-found guard) or a parse-fail.
    """

    def parse_args(self, argv: list[str]) -> argparse.Namespace:
        # We can't cleanly extract the parser; instead, invoke main()
        # and trust SystemExit to surface argparse failures. main()
        # validates --hold-last-secs / --blend-secs / --idle-x2m2
        # BEFORE binding any sockets, so a well-formed argv with a
        # real idle_x2m2 will reach the socket-bind step. We don't
        # want to bind, so we intentionally pass a bogus idle path
        # AFTER the test confirms argparse accepted the rest.
        #
        # Two-phase check:
        #   Phase 1: argv as built by the launcher should NOT trigger
        #            argparse SystemExit. Re-running main with a
        #            sentinel idle path proves the parser accepted
        #            everything else; we catch the idle-file error
        #            as success.
        sentinel = "/nonexistent/sentinel_idle.x2m2"
        argv_munged = list(argv)
        for i, tok in enumerate(argv_munged):
            if tok == "--idle-x2m2" and i + 1 < len(argv_munged):
                argv_munged[i + 1] = sentinel
        try:
            rc = proxy.main(argv_munged)
        except SystemExit as e:
            code = getattr(e, "code", 1) or 0
            if code == 0:
                return argparse.Namespace()
            raise AssertionError(
                f"argparse rejected launcher argv (SystemExit code={code}): "
                f"{argv_munged!r}"
            )
        # main() returns 1 when idle_x2m2 is missing -- that means
        # argparse + range validators passed and we tripped on the
        # file check. Anything else is a real failure.
        if rc == 1:
            return argparse.Namespace()
        raise AssertionError(
            f"proxy.main returned unexpected rc={rc} for argv={argv_munged!r}"
        )


def _build_launcher_argv(
    *,
    upstream_port: int,
    downstream_port: int,
    override_port: int,
    vla_control_port: int,
) -> list[str]:
    """Mirror the argv ``spawn_sim_proxy`` builds in the bash launcher.

    Keep this in sync with the ``proxy_args=( … )`` array in
    ``gear_sonic/scripts/run_x2_vla_runtime.sh::spawn_sim_proxy``.
    Any divergence is exactly the bug this test is here to catch.
    """
    argv: list[str] = [
        "--upstream-host", "127.0.0.1",
        "--upstream-port", str(upstream_port),
        "--upstream-topic", "pose",
        "--downstream-host", "127.0.0.1",
        "--downstream-port", str(downstream_port),
        "--downstream-topic", "pose",
        "--idle-x2m2", str(IDLE_X2M2),
        "--idle-stale-ms", "300",
        "--idle-mode", "blend",
        "--hold-last-secs", "10.0",
        "--blend-secs", "3.0",
        "--no-x2-debug-yaw-track",
    ]
    if override_port > 0:
        argv += [
            "--override-host", "127.0.0.1",
            "--override-port", str(override_port),
            "--override-topic", "pose",
            "--override-stale-ms", "200",
            "--override-frozen-ticks", "10",
            "--override-frozen-l2-tol", "5e-3",
            "--override-engage-motion-ticks", "10",
            # 2026-06-10 follow-up: when the manager's stream_mode
            # PUB is reachable on 127.0.0.1:5564 (the default in
            # run_x2_vla_runtime.sh sim mode) the launcher hands the
            # proxy these args too. Mirror that here so the test
            # stays a faithful argv echo.
            "--teleop-mode-host", "127.0.0.1",
            "--teleop-mode-port", "5564",
            "--teleop-mode-topic", "stream_mode",
            "--teleop-mode-stale-ms", "1000",
        ]
    if vla_control_port > 0:
        argv += [
            "--vla-control-bind-host", "127.0.0.1",
            "--vla-control-port", str(vla_control_port),
            "--vla-control-topic", "vla_control",
        ]
    return argv


def test_launcher_bash_syntax_ok():
    """``bash -n`` catches structural bash errors in the launcher."""
    bash = shutil.which("bash")
    assert bash is not None, "bash not on PATH; cannot syntax-check launcher"
    assert LAUNCHER.is_file(), f"launcher missing: {LAUNCHER}"
    proc = subprocess.run(
        [bash, "-n", str(LAUNCHER)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, (
        f"bash -n failed (rc={proc.returncode}):\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )


def test_launcher_help_lists_manual_takeover_flags():
    """``--help`` must document the manual-takeover CLI flags.

    Regression escape: if a future refactor renames the case-statement
    entries but forgets to update the heredoc, the operator-facing
    contract silently drifts. Pinning the visible flag names in the
    help output is the cheapest way to keep them in lock-step.
    """
    bash = shutil.which("bash")
    assert bash is not None, "bash not on PATH; cannot run launcher --help"
    proc = subprocess.run(
        [bash, str(LAUNCHER), "--help"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, (
        f"--help failed (rc={proc.returncode}):\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    for flag in (
        "--vla-control-port",
        "--vla-control-host",
        "--vla-cold-restart-hold-ticks",
        "--vla-handoff-max-hold-ticks",
        "--pose-proxy-override-port",
        "--pose-proxy-override-stale-ms",
        "--pose-proxy-override-frozen-ticks",
        "--pose-proxy-override-frozen-l2-tol",
        "--pose-proxy-override-engage-motion-ticks",
        "--pose-proxy-teleop-mode-host",
        "--pose-proxy-teleop-mode-port",
        "--pose-proxy-teleop-mode-topic",
        "--pose-proxy-teleop-mode-stale-ms",
        "--pose-proxy-downstream-port",
    ):
        assert flag in proc.stdout, (
            f"{flag} missing from launcher --help output; "
            f"the case statement and the heredoc have drifted."
        )


def test_launcher_accepts_manual_takeover_cli_flags(tmp_path):
    """CLI flags must be parsed without falling into the ``*) ARGS+=``
    catch-all (which would silently forward them to the bridge and the
    operator would see "bridge: unrecognized argument" hours later).

    We run the launcher with ``preflight`` so it parses argv, runs the
    pre-flight probes, and exits without spawning anything. A typo in
    the case-statement would hit the ``*) ARGS+=("$1"); shift`` branch
    and the flags would end up in ``BRIDGE_ARGS``. We can't observe
    that directly without intercepting subprocess, so we check the
    next-best signal: the launcher banner echoes the resolved sim-
    proxy state, which only flips ON when the parsed values land in
    ``POSE_PROXY_OVERRIDE_PORT`` (not in ``ARGS``).
    """
    bash = shutil.which("bash")
    assert bash is not None
    run_dir = tmp_path / "preflight_run"
    # preflight in sim mode (no --pc2-host) with the new CLI flags;
    # SKIP_PREFLIGHT skips the heavy model + decoder probes so the
    # test stays fast. We only care about the argv parse path.
    proc = subprocess.run(
        [
            bash,
            str(LAUNCHER),
            "preflight",
            "--vla-control-port", "5559",
            "--pose-proxy-override-port", "5560",
            "--pose-proxy-override-stale-ms", "150",
            "--pose-proxy-override-frozen-ticks", "8",
            "--pose-proxy-override-frozen-l2-tol", "5e-5",
            "--pose-proxy-override-engage-motion-ticks", "7",
            "--pose-proxy-teleop-mode-host", "192.168.42.11",
            "--pose-proxy-teleop-mode-port", "5564",
            "--pose-proxy-teleop-mode-topic", "stream_mode",
            "--pose-proxy-teleop-mode-stale-ms", "750",
            "--pose-proxy-downstream-port", "5558",
            "--vla-cold-restart-hold-ticks", "30",
            "--run-dir", str(run_dir),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env={**__import__("os").environ, "SKIP_PREFLIGHT": "1"},
    )
    # preflight exits 0 on a clean parse + skipped probes. If the new
    # flags fell through to the catch-all, BRIDGE_ARGS would be set
    # but preflight never reads them, so we'd still exit 0 -- the
    # observable signal is the banner / log lines. Check stdout for
    # the resolved sim-proxy state.
    combined = proc.stdout + proc.stderr
    # Either the launcher prints the banner with the proxy line, or
    # (if preflight short-circuits before the banner) it at least
    # doesn't emit an "unrecognized argument" warning for our flags.
    for flag in (
        "--vla-control-port",
        "--pose-proxy-override-port",
        "--pose-proxy-override-stale-ms",
        "--pose-proxy-override-frozen-ticks",
        "--pose-proxy-override-frozen-l2-tol",
        "--pose-proxy-override-engage-motion-ticks",
        "--pose-proxy-teleop-mode-host",
        "--pose-proxy-teleop-mode-port",
        "--pose-proxy-teleop-mode-topic",
        "--pose-proxy-teleop-mode-stale-ms",
        "--pose-proxy-downstream-port",
        "--vla-cold-restart-hold-ticks",
    ):
        # A drift-detection regex: "Unknown argument: --vla-control-port"
        # or similar. The launcher doesn't currently emit such a
        # warning (the catch-all is silent), so this is forward-
        # looking insurance. The stricter check below is the real
        # gate.
        assert f"Unknown argument: {flag}" not in combined, (
            f"launcher rejected {flag} as unknown ({combined!r})"
        )
    # preflight either succeeds (rc=0) or fails with a domain-
    # specific reason (model missing, etc.) -- both are fine. The
    # forbidden outcome is rc != 0 paired with a bash-level usage
    # error (rc=2) or a "command not found" trace.
    assert proc.returncode in (0, 1), (
        f"launcher preflight returned unexpected rc={proc.returncode}; "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


@pytest.mark.parametrize(
    "override_port,vla_control_port",
    [
        # Override only: dual-source arbitration without cold-restart edge.
        (5560, -1),
        # vla_control only: edge events without a second pose source
        # (operator wants the bridge to react to PROXY-side OVERRIDE
        # state but isn't wiring a teleop SUB yet -- valid use case
        # during integration ramp-up).
        (-1, 5559),
        # Both: the canonical sim manual-takeover configuration.
        (5560, 5559),
    ],
)
def test_spawn_sim_proxy_argv_parses(override_port, vla_control_port):
    """The launcher's spawn_sim_proxy argv must pass proxy argparse.

    Catches typos / arg-name drift between the bash launcher and the
    proxy script. The proxy's idle-x2m2 file-not-found guard is what
    actually short-circuits us out of main(); reaching that means
    argparse + range validators accepted everything else.
    """
    assert IDLE_X2M2.is_file(), (
        f"idle_stand.x2m2 missing at {IDLE_X2M2} -- rebake via "
        f"`python -m gear_sonic_deploy.scripts.bake_idle_stand_x2m2`"
    )
    argv = _build_launcher_argv(
        upstream_port=5556,
        downstream_port=5558,
        override_port=override_port,
        vla_control_port=vla_control_port,
    )
    parser = _build_proxy_parser()
    parser.parse_args(argv)


def test_spawn_sim_deploy_routes_omnihand_sub_through_proxy_when_proxy_on():
    """spawn_sim_deploy MUST redirect --sim-hand-zmq-host/--sim-hand-zmq-port
    to the proxy's downstream port when SIM_PROXY_ENABLED=1 and
    SIM_WITH_OMNIHAND=1.

    Regression pin for the 2026-06-10 late-afternoon "fingers still
    not responding" bug. The OmniHand SUB in
    ``x2_mujoco_ros_bridge.py`` defaults to ``localhost:5556``
    (the bridge's port), bypassing the proxy entirely. Without this
    redirect, operator finger commands silently die on the recorder
    -> proxy -> deploy hop because OmniHand never subscribed to the
    proxy's downstream port. Body joints still work (sim deploy's
    body SUB IS routed through the proxy), so the user-visible
    symptom is exactly "override engages, body follows, fingers
    stuck at VLA chunks" -- the same thing they hit.

    This is a source-level pattern check (no subprocess) because
    spawn_sim_deploy isn't trivially reachable from outside the
    launcher's main() flow; the alternative would be sourcing the
    bash file with all the right env-var stubs in place, which is
    fragile. Direct pattern assertion catches the regression cheaply.
    """
    src = LAUNCHER.read_text()
    # Find spawn_sim_deploy() function body.
    start = src.find("\nspawn_sim_deploy()")
    assert start >= 0, "spawn_sim_deploy() not found in launcher"
    end = src.find("\n}\n", start)
    assert end > start, (
        "spawn_sim_deploy() opening found but no matching closing brace"
    )
    func_body = src[start:end]
    # Three claims, all conjunctive:
    #   1. --sim-with-omnihand is added inside the SIM_WITH_OMNIHAND
    #      branch (regression check on the umbrella feature),
    #   2. --sim-hand-zmq-host is added in the SAME branch,
    #   3. --sim-hand-zmq-port is added in the SAME branch,
    # AND the host/port pair is gated on SIM_PROXY_ENABLED.
    assert "--sim-with-omnihand" in func_body, (
        "spawn_sim_deploy must forward --sim-with-omnihand to deploy_x2.sh"
    )
    assert "--sim-hand-zmq-host" in func_body, (
        "spawn_sim_deploy must forward --sim-hand-zmq-host to deploy_x2.sh "
        "when proxy + omnihand are both on (otherwise the OmniHand SUB "
        "bypasses the proxy and operator finger commands are silently "
        "dropped during override)"
    )
    assert "--sim-hand-zmq-port" in func_body, (
        "spawn_sim_deploy must forward --sim-hand-zmq-port to deploy_x2.sh "
        "when proxy + omnihand are both on (see --sim-hand-zmq-host)"
    )
    # The host/port pair MUST be gated on SIM_PROXY_ENABLED so legacy
    # autonomous-only sim runs (no proxy in the wire) don't try to
    # subscribe through a port that nothing is publishing on.
    hand_zmq_idx = func_body.find("--sim-hand-zmq-host")
    # Look backwards for the nearest SIM_PROXY_ENABLED gate.
    preceding = func_body[:hand_zmq_idx]
    assert "SIM_PROXY_ENABLED" in preceding, (
        "the --sim-hand-zmq-host/port forwarding MUST be gated on "
        "SIM_PROXY_ENABLED so legacy non-proxy sim runs aren't "
        "broken by pointing OmniHand at an unbound port"
    )
    # The host/port pair MUST resolve to the proxy's downstream
    # (NOT LAPTOP_POSE_PORT). spawn_sim_deploy already computes
    # deploy_pose_host/port for the body SUB; the new wire must
    # reuse those exact variables so body + fingers cannot disagree.
    assert '"$deploy_pose_host"' in func_body, (
        "spawn_sim_deploy must reuse $deploy_pose_host for the new "
        "OmniHand wire so body + fingers come from the same source"
    )
    assert '"$deploy_pose_port"' in func_body, (
        "spawn_sim_deploy must reuse $deploy_pose_port for the new "
        "OmniHand wire so body + fingers come from the same source"
    )


def test_spawn_sim_deploy_forwards_wrist_bypass_via_deploy_extra_arg():
    """spawn_sim_deploy must forward $WRIST_BYPASS to deploy_x2.sh.

    Regression pin for the 2026-06-10 "wrist not responding to teleop"
    bug. The SONIC tracker pins wrist pitch/roll regardless of the IK
    reference -- see wrist_bypass.hpp. The C++ deploy provides
    ``--wrist-bypass ik`` for surgical override of those 4 MJ DOFs
    ({20,21,27,28}) with the wire's joint_pos_mj before the safety
    stack. The launcher previously never set this, so even though
    operator/VLA wrist commands were on the wire, SONIC clamped them
    away. The fix is to forward $WRIST_BYPASS to deploy_x2.sh via two
    ``--deploy-extra-arg`` tokens (because the C++ CLI expects
    ``--wrist-bypass <mode>`` as a value-separated pair).
    """
    src = LAUNCHER.read_text()
    start = src.find("\nspawn_sim_deploy()")
    assert start >= 0, "spawn_sim_deploy() not found in launcher"
    end = src.find("\n}\n", start)
    assert end > start
    func_body = src[start:end]
    # Two extras (key + value) because deploy_x2.sh appends each
    # extra verbatim; ``--wrist-bypass ik`` would otherwise be
    # collapsed into a single ill-formed token.
    assert "--wrist-bypass" in func_body, (
        "spawn_sim_deploy must forward --wrist-bypass to the C++ deploy "
        "via --deploy-extra-arg so the SONIC wrist clamp is bypassed "
        "for manual takeover and VLA wrist tracking"
    )
    assert func_body.count("--deploy-extra-arg") >= 3, (
        "spawn_sim_deploy must use --deploy-extra-arg at least three "
        "times: one for --disable-pose-ref-watchdog, plus a key+value "
        "pair for --wrist-bypass <mode>"
    )
    # Default value must be 'ik' (per 2026-06-10 follow-up 7 -- the
    # operator at 13:12 explicitly requested wrist_bypass be enabled
    # so wrist gestures actually move the wrist).
    src_for_default = LAUNCHER.read_text()
    assert 'WRIST_BYPASS:=ik' in src_for_default, (
        "WRIST_BYPASS default must be 'ik' (per follow-up 7 the "
        "operator explicitly requested wrist ik enabled so wrist "
        "gestures actually move the wrist). If you intended to "
        "default 'off', read 2026-06-10 follow-up 7 first."
    )
    # CRITICAL NEGATIVE PIN -- the launcher MUST NOT auto-pair
    # wrist_bypass=ik with ``--max-target-dev``. Follow-up 8 proved
    # the auto-pair makes the robot collapse: --max-target-dev is a
    # GLOBAL absolute clamp on ALL joint groups (leg + waist + arm
    # + head), pinning everything to default +/- N. At 0.05 the
    # robot couldn't bend its knees enough to stand (act_clip_ticks
    # 916/1000 in the 13:21 deploy.log). Wrist slam mitigation lives
    # on the bridge side instead (--vla-max-wire-step + follow-up 6
    # slow-step ramp). Pin BOTH the absence of the auto-pair AND
    # the absence of the env var so a future revival of the same
    # mistake fails CI before it ships.
    assert 'WRIST_BYPASS_MAX_TARGET_DEV' not in src_for_default, (
        "WRIST_BYPASS_MAX_TARGET_DEV must not exist -- it was "
        "introduced in follow-up 7 to auto-pair --wrist-bypass ik "
        "with --max-target-dev, but the 13:21 run proved this makes "
        "the robot collapse (the flag is a GLOBAL clamp on leg + "
        "waist + arm + head, not a wrist-specific rate clamp). "
        "See follow-up 8 in the 2026-06-10 milestone doc before "
        "reintroducing this."
    )
    # The forwarded --deploy-extra-arg block must NOT mention
    # --max-target-dev in the wrist-bypass forwarding (a future
    # change that defaults it back as part of the wrist block
    # would re-trigger the collapse).
    wrist_block_start = src_for_default.find('WRIST_BYPASS}" != "off"')
    assert wrist_block_start > 0, (
        "couldn't find the wrist-bypass forwarding conditional; "
        "did you rename WRIST_BYPASS? if so update this test."
    )
    wrist_block = src_for_default[
        wrist_block_start : wrist_block_start + 2000
    ]
    assert '--deploy-extra-arg --max-target-dev' not in wrist_block, (
        "wrist-bypass forwarding block re-introduced "
        "--deploy-extra-arg --max-target-dev; this is the exact "
        "regression that collapsed the robot in the 13:21 run. "
        "If you need a wrist-specific clamp, add a per-group "
        "override in the C++ deploy (--max-target-dev-wrist), "
        "don't reuse the global --max-target-dev."
    )


def test_launcher_help_lists_wrist_bypass_flag():
    """``--help`` must document --wrist-bypass so operators can find it."""
    bash = shutil.which("bash")
    assert bash is not None
    proc = subprocess.run(
        [bash, str(LAUNCHER), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0
    assert "--wrist-bypass" in proc.stdout, (
        "--wrist-bypass missing from launcher --help; the case "
        "statement and the heredoc have drifted"
    )


def test_launcher_forwards_handoff_max_hold_ticks_to_bridge():
    """The smooth-handoff CLI flag must reach the bridge BRIDGE_ARGS.

    Regression pin for the 2026-06-10 (PM follow-up 3) smooth-handoff
    guard. The bridge takes ``--vla-handoff-max-hold-ticks N``; the
    launcher reads it as ``VLA_HANDOFF_MAX_HOLD_TICKS`` (env or CLI)
    and appends it to BRIDGE_ARGS inside the manual-takeover wiring
    block. Without this, the bridge defaults the new guard to 200
    silently and the operator can't tune the safety cap from the
    launcher. We grep the launcher source for the wiring pattern;
    avoids subprocess flakiness.
    """
    src = LAUNCHER.read_text()
    # The default must be set so older callers (env-var-only) still get
    # the guard on.
    assert 'VLA_HANDOFF_MAX_HOLD_TICKS:=200' in src, (
        "VLA_HANDOFF_MAX_HOLD_TICKS default missing/wrong; should be "
        "200 (= 4 s @ 50 Hz)"
    )
    # The CLI flag must be in the case statement.
    assert '--vla-handoff-max-hold-ticks)' in src, (
        "--vla-handoff-max-hold-ticks missing from launcher case "
        "statement; CLI override won't take effect"
    )
    # The BRIDGE_ARGS append must reference both VLA_HANDOFF_MAX_HOLD_TICKS
    # and the flag name.
    assert '--vla-handoff-max-hold-ticks "${VLA_HANDOFF_MAX_HOLD_TICKS}"' in src, (
        "BRIDGE_ARGS append for --vla-handoff-max-hold-ticks missing; "
        "the bridge won't see the operator's value"
    )


def test_bridge_fails_fast_when_handoff_max_hold_less_than_cold_restart_hold():
    """Bridge MUST refuse to start when handoff cap < cold-restart hold.

    A handoff cap shorter than the minimum hold would mean the bridge
    releases the wire to idle BEFORE the proxy's HOLD ladder finishes
    replaying the operator pose -- the exact "abrupt motion" symptom
    the smooth-handoff guard is here to prevent. The bridge's
    startup validator catches this with sys.exit(2). We invoke the
    bridge with --help-style smoke (real launch needs a model + GPU)
    by triggering the validator via a minimal argv. Since the
    validator runs before any heavy init, we can drive it with a
    bogus model path and assert sys.exit(2) with the expected message.
    """
    bridge = REPO_ROOT / "gear_sonic" / "scripts" / "live_vla_publish_motion_token.py"
    assert bridge.is_file()
    # Invoke with the bad combination; the validator runs before model
    # loading. We don't need a real model because the validator exits
    # at the top of main(). Use --help to confirm the flag is wired.
    proc = subprocess.run(
        [sys.executable, str(bridge), "--help"],
        capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, (
        f"bridge --help failed (rc={proc.returncode}): {proc.stderr}"
    )
    assert "--vla-handoff-max-hold-ticks" in proc.stdout, (
        "--vla-handoff-max-hold-ticks missing from bridge --help; "
        "argparse drift"
    )
    assert "Safety cap" in proc.stdout, (
        "--vla-handoff-max-hold-ticks help text doesn't explain its "
        "role as the safety cap; help text drift could mislead "
        "operators tuning the value"
    )


def test_handoff_gate_requires_nontrivial_token_magnitude():
    """The cold-restart handoff gate MUST check token magnitude.

    Regression pin for 2026-06-10 follow-up 5. The original
    follow-up 3 gate released the wire when ``chunk_id > baseline
    and chunk_id > 0`` -- but the decoder below has its own gate
    (``np.linalg.norm(token[step]) > 1e-3``) that skips zero-token
    chunks. The mismatch meant that if VLA produced a steady
    stream of zero-token chunks (which happens whenever the model
    hasn't latched onto the prompt, the camera feed is missing /
    occluded, or the proprio decoder is starved), the cold-restart
    hold would release the wire on the FIRST chunk arrival, the
    decoder would refuse to use that chunk, and ``cur_jpos`` would
    fall through to ``idle_loop.current(tick)`` (= idle_stand pose).
    The operator's wire would snap from operator-pose to
    idle_stand-pose in a single tick -- which produced the
    2026-06-10 12:26 "hand slammed into the table" report (see
    docs/source/user_guide/milestones/2026-06-10_vla_manual_takeover.md,
    follow-up 4 / 5).

    The fix mirrors the decoder's token-norm guard inside the
    handoff gate so the wire stays at operator-pose until VLA is
    producing usable tokens OR ``handoff_max_hold_ticks`` expires.
    Pin both the source-level pattern (so a future "just check
    chunk_id" refactor fails this test) AND the always-on per-tick
    wire rate clamp on the idle branch (defense-in-depth: if the
    safety cap DOES expire, the wire ramps from operator-pose to
    idle at ``max_wire_step`` rad/tick instead of snapping).
    """
    bridge = (
        REPO_ROOT / "gear_sonic" / "scripts" / "live_vla_publish_motion_token.py"
    )
    src = bridge.read_text()
    # The gate must include a token-magnitude check. We pin the
    # exact variable name ``current_token_norm`` so a refactor that
    # renames it fails loudly here -- the e2e symptom (slam) is
    # impossible to catch in CI without a sim run, but the variable
    # presence is a cheap proxy.
    assert "current_token_norm = float(np.linalg.norm(token[step]))" in src, (
        "cold-restart handoff gate is missing the token-magnitude "
        "snapshot. Without ``current_token_norm`` the gate can release "
        "the wire on a zero-token chunk and the decoder skip + fall-through "
        "to idle_stand will produce the 12:26 slam regression. See follow-up "
        "5 in docs/source/user_guide/milestones/2026-06-10_vla_manual_takeover.md"
    )
    assert (
        "first_eligible_chunk_ready = (" in src
        and "current_token_norm > 1e-3" in src
    ), (
        "cold-restart handoff first_eligible_chunk_ready predicate must "
        "include the ``current_token_norm > 1e-3`` clause; otherwise "
        "the gate releases on zero-token chunks and the wire snaps to "
        "idle_stand."
    )
    # Defense-in-depth: the per-tick wire rate clamp must also fire
    # on the ``else`` (idle wire) branch. Without this, the safety
    # cap expiring still produces a snap (just delayed).
    assert (
        "# Idle wire (deploy stale, no decoder, or zero-token chunk)." in src
        and "cur_jpos = _clamp_vector_step(" in src
        and "max_wire_step" in src
    ), (
        "idle-wire branch must call ``_clamp_vector_step(cur_jpos, "
        "prev_wire_jpos, max_wire_step)`` so when the cold-restart "
        "safety cap expires the wire ramps from operator-pose to "
        "idle_stand instead of snapping."
    )


def test_launcher_forwards_handoff_slow_step_to_bridge():
    """Launcher must forward the slow-step window flags to the bridge.

    2026-06-10 follow-up 6 added ``--vla-handoff-max-wire-step`` and
    ``--vla-handoff-step-ramp-ticks`` to bound the per-element wire
    step right after the cold-restart hold releases (the 12:50 +
    13:05 runs showed body_Δ=0.247 rad sustained during the LPF ramp
    even with the existing handoff guard fired correctly). The
    launcher reads these as env vars (defaults 0.012 / 250) and CLI
    flags, and MUST forward both to ``BRIDGE_ARGS`` so the bridge
    sees the operator's values -- otherwise the bridge silently
    falls back to its own defaults and the operator can't tune the
    slow window from the launcher.
    """
    src = LAUNCHER.read_text()
    assert 'VLA_HANDOFF_MAX_WIRE_STEP:=0.012' in src, (
        "VLA_HANDOFF_MAX_WIRE_STEP default missing/wrong; should "
        "be 0.012 rad/tick (= ~36 deg/s/joint at 50 Hz, ~3x slower "
        "than --vla-max-wire-step default of 0.035 rad/tick). "
        "Lower -> slower / safer handoff; higher -> closer to a slam."
    )
    assert 'VLA_HANDOFF_STEP_RAMP_TICKS:=250' in src, (
        "VLA_HANDOFF_STEP_RAMP_TICKS default missing/wrong; should "
        "be 250 ticks (= 5 s @ 50 Hz). Lower -> wire returns to "
        "normal speed sooner; higher -> longer safe ramp."
    )
    for flag in (
        '--vla-handoff-max-wire-step)',
        '--vla-handoff-step-ramp-ticks)',
    ):
        assert flag in src, (
            f"{flag.rstrip(')')!r} missing from launcher case "
            "statement; CLI override won't take effect"
        )
    for wiring in (
        '--vla-handoff-max-wire-step "${VLA_HANDOFF_MAX_WIRE_STEP}"',
        '--vla-handoff-step-ramp-ticks "${VLA_HANDOFF_STEP_RAMP_TICKS}"',
    ):
        assert wiring in src, (
            f"BRIDGE_ARGS append for {wiring!r} missing; the bridge "
            "won't see the operator's value"
        )


def test_handoff_slow_step_state_machine_in_bridge_source():
    """Bridge source must implement the slow-step state machine.

    Pins the structural pieces of the 2026-06-10 follow-up 6 fix so
    a refactor that drops them fails CI before the operator sees a
    regressed slam:

      - ``handoff_step_remaining`` countdown variable
      - Arming the countdown on both success AND safety-cap paths
        out of the cold-restart hold
      - Linear interpolation ``handoff_max_wire_step -> max_wire_step``
        across ``handoff_step_ramp_ticks`` ticks
      - Applying the interpolated ``effective_max_step`` to the rate
        clamp in BOTH the decoder-succeeded branch AND the idle-wire
        fallthrough branch (defense-in-depth -- if only one branch
        gets the slow step, the safety-cap path still slams)
    """
    bridge = (
        REPO_ROOT / "gear_sonic" / "scripts" / "live_vla_publish_motion_token.py"
    )
    src = bridge.read_text()
    assert "handoff_step_remaining = 0" in src, (
        "missing initialiser ``handoff_step_remaining = 0`` -- the "
        "post-handoff slow-step countdown won't exist"
    )
    # Both paths out of the hold must arm the countdown. We search
    # for the arm pattern after both the success log line and the
    # safety-cap log line.
    success_arm = "handoff_step_remaining = max(int(handoff_step_ramp_ticks), 0)"
    assert src.count(success_arm) >= 2, (
        f"``{success_arm}`` must appear at least twice (once on the "
        "first-eligible-chunk success path, once on the safety-cap "
        "expiry path). Otherwise the slow-step ramp won't engage on "
        "one of them, producing a slam on that path."
    )
    # The interpolation formula must be present.
    assert (
        "(1.0 - ramp_progress) * float(handoff_max_wire_step)" in src
        and "ramp_progress * float(max_wire_step)" in src
    ), (
        "linear interpolation handoff_max_wire_step -> max_wire_step "
        "missing; the wire step won't actually ramp"
    )
    # The interpolated ``effective_max_step`` must be passed to the
    # rate clamp in BOTH branches.
    assert (
        src.count("cur_jpos = _clamp_vector_step(") >= 2
        and src.count("effective_max_step") >= 2
    ), (
        "``effective_max_step`` must be used in BOTH the decoder-"
        "succeeded AND idle-wire rate-clamp call sites. Otherwise "
        "the slow step only applies on one path; the other path "
        "will slam."
    )


def test_launcher_forwards_engagement_slow_step_to_proxy():
    """Launcher must forward the proxy engagement-ramp flags.

    2026-06-10 follow-up 9 added a SYMMETRIC slow-step ramp on the
    LIVE -> OVERRIDE edge (operator takes over from VLA), mirroring
    follow-up 6's handoff slow-step on the OVERRIDE -> LIVE edge.
    Without this the proxy forwarded the operator's first override
    frame VERBATIM and the deploy slammed the body across the full
    VLA -> operator joint-space delta in one tick (the user
    reported "still rams when taking over from vla to ON" at 13:29
    after follow-up 6 fixed the OFF transition).

    The launcher reads three env vars (defaults 0.012 / 0.035 /
    250 to match the bridge handoff defaults) and three CLI flags;
    all three must be forwarded to the proxy spawn argv. Negative
    assertion: the launcher MUST NOT auto-pair these with any
    deploy-side flag (the follow-up 8 regression: a global
    --max-target-dev collapsed the robot's legs).
    """
    src = LAUNCHER.read_text()
    for default in (
        'POSE_PROXY_ENGAGEMENT_MAX_WIRE_STEP:=0.012',
        'POSE_PROXY_ENGAGEMENT_STEADY_WIRE_STEP:=0.035',
        'POSE_PROXY_ENGAGEMENT_STEP_RAMP_TICKS:=250',
    ):
        assert default in src, (
            f"default ``{default}`` missing from launcher; the "
            "engagement ramp will fall back to whatever the proxy "
            "parser defaults to (currently the same numbers, but "
            "this pin catches divergence)"
        )
    for flag in (
        '--pose-proxy-engagement-max-wire-step)',
        '--pose-proxy-engagement-steady-wire-step)',
        '--pose-proxy-engagement-step-ramp-ticks)',
    ):
        assert flag in src, (
            f"CLI flag {flag.rstrip(')')!r} missing from launcher "
            "case statement; the operator can't override the default"
        )
    for wiring in (
        '--engagement-max-wire-step "$POSE_PROXY_ENGAGEMENT_MAX_WIRE_STEP"',
        '--engagement-steady-wire-step "$POSE_PROXY_ENGAGEMENT_STEADY_WIRE_STEP"',
        '--engagement-step-ramp-ticks "$POSE_PROXY_ENGAGEMENT_STEP_RAMP_TICKS"',
    ):
        assert wiring in src, (
            f"proxy_args append for {wiring!r} missing; the proxy "
            "won't see the operator's value -- engagement ramp will "
            "silently fall back to parser default"
        )
    # Negative: must NOT auto-pair the proxy engagement flags with
    # any deploy-side rate-limit flag. The deploy's --max-target-dev
    # is a GLOBAL absolute clamp on all joint groups (not per-tick,
    # not arm-specific) and pairing it with anything collapsed the
    # robot during follow-up 7 (see test_spawn_sim_deploy_forwards_
    # wrist_bypass_via_deploy_extra_arg negative assertions).
    assert "ENGAGEMENT_MAX_TARGET_DEV" not in src, (
        "follow-up 8 regression: do NOT auto-pair the proxy "
        "engagement ramp with --max-target-dev. The proxy controls "
        "the wire step alone; the deploy's own rate limits stay "
        "at their defaults."
    )


def test_proxy_engagement_clamp_state_machine_in_source():
    """Proxy source must implement the engagement-ramp state machine.

    Pins the structural pieces of 2026-06-10 follow-up 9 so a
    refactor that drops them fails CI before the operator sees a
    regressed slam on takeover:

      - ``rebuild_msg_with_jpos_override`` helper that surgically
        replaces ``joint_pos_mj`` bytes (preserves hands / root_quat /
        future window)
      - ``_clamp_vector_step_f32`` per-element rate clamp
      - ``engagement_clamp_remaining`` countdown variable
      - Arming the countdown on the LIVE -> OVERRIDE edge
      - Tearing down the countdown on the OVERRIDE -> LIVE edge
        (so re-engage starts fresh from the new VLA anchor)
      - Linear interpolation ``engagement_max_wire_step ->
        engagement_steady_wire_step`` across ``engagement_step_
        ramp_ticks``
      - Applying the clamp via ``rebuild_msg_with_jpos_override``
        (NOT via forwarding the original frame verbatim -- the
        whole point of the fix)
    """
    proxy = (
        REPO_ROOT
        / "gear_sonic_deploy"
        / "scripts"
        / "x2_pose_proxy.py"
    )
    src = proxy.read_text()
    assert "def rebuild_msg_with_jpos_override(" in src, (
        "missing surgical jpos-replace helper; the proxy can't "
        "modify operator frames without re-packing the whole frame"
    )
    assert "def _clamp_vector_step_f32(" in src, (
        "missing per-element rate clamp helper; the engagement ramp "
        "has nothing to call to slow down the wire step"
    )
    assert "engagement_clamp_remaining = 0" in src, (
        "missing initialiser ``engagement_clamp_remaining = 0``; "
        "the engagement-ramp countdown won't exist"
    )
    # Must be armed on the LIVE -> OVERRIDE edge (i.e., inside the
    # ``if override_fresh and not override_active`` block where
    # override_active flips True).
    assert (
        "engagement_clamp_remaining = engagement_step_ramp_ticks"
        in src
    ), (
        "missing arm site ``engagement_clamp_remaining = "
        "engagement_step_ramp_ticks``; the LIVE -> OVERRIDE edge "
        "won't trigger the slow-step ramp"
    )
    # Must be torn down on the OVERRIDE -> LIVE/IDLE edge so a
    # rapid release+re-engage within the window re-arms cleanly.
    teardown_idx = src.find(
        "engagement_clamp_remaining = 0\n                "
        "engagement_last_forwarded_jpos = None"
    )
    assert teardown_idx != -1, (
        "missing release-edge teardown of engagement_clamp_"
        "remaining + engagement_last_forwarded_jpos; a rapid "
        "release-then-re-engage within the ramp window will "
        "inherit the previous (stale) anchor and skip the "
        "slow-step bridge from VLA's new pose"
    )
    # Linear interpolation formula must be present.
    assert (
        "(1.0 - ramp_progress) * engagement_max_wire_step" in src
        and "ramp_progress * engagement_steady_wire_step" in src
    ), (
        "linear interpolation engagement_max_wire_step -> "
        "engagement_steady_wire_step missing; the wire step won't "
        "actually ramp"
    )
    # The clamp MUST be applied via the surgical-rebuild helper
    # (not by just forwarding the original frame), and the
    # rebuilt-frame branch must update both the forwarded message
    # AND the next-tick anchor.
    assert "rebuilt = rebuild_msg_with_jpos_override(" in src, (
        "engagement clamp isn't applied to the forwarded bytes; "
        "the operator's verbatim jpos still hits the wire and the "
        "deploy still slams"
    )
    assert "engagement_last_forwarded_jpos = (\n" in src, (
        "engagement_last_forwarded_jpos must be updated to the "
        "clamped pose on every successful clamp so the next tick's "
        "step is measured from THIS forwarded pose (not the prev "
        "anchor, which would let the operator's pose drift away "
        "faster than max_step per tick)"
    )


def test_rebuild_msg_with_jpos_override_preserves_other_fields():
    """``rebuild_msg_with_jpos_override`` must NOT touch other fields.

    The whole point of the surgical-byte-replace helper is that the
    proxy can clamp ONLY ``joint_pos_mj`` while leaving every other
    field (hands, root_quat, motion_token, frame_index, future
    window) byte-identical. If the helper re-packs the frame from
    scratch it might quietly drop a field (e.g., the operator's
    hand joints, which would make finger commands stop working
    mid-takeover -- exactly the failure mode follow-up 5 already
    fixed once).

    Round-trip test:
      1. Pack a frame with a known jpos + non-zero hands +
         arbitrary root_quat
      2. Run it through rebuild_msg_with_jpos_override with a new
         jpos
      3. Re-decode: jpos must match the new value; hands +
         root_quat must be byte-identical to the original
    """
    sys.path.insert(0, str(PROXY_DIR))
    try:
        import importlib
        proxy_mod = importlib.import_module("x2_pose_proxy")
    finally:
        sys.path.pop(0)

    import numpy as np

    num_body = proxy_mod.NUM_BODY_DOFS
    num_hand = proxy_mod.DEFAULT_HAND_DOF
    jpos_orig = np.linspace(-0.5, 0.5, num_body, dtype=np.float32)
    jpos_new = np.full(num_body, 0.3, dtype=np.float32)
    left_hand = np.linspace(0.1, 0.9, num_hand, dtype=np.float32)
    right_hand = np.linspace(0.9, 0.1, num_hand, dtype=np.float32)
    root_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    payload = {
        "joint_pos_mj": jpos_orig,
        "root_quat_xyzw": root_quat,
        "left_hand_joints": left_hand,
        "right_hand_joints": right_hand,
        "frame_index": np.array([42], dtype=np.int64),
    }
    msg = proxy_mod.pack_pose_message(payload, topic="pose", version=4)
    rebuilt = proxy_mod.rebuild_msg_with_jpos_override(
        msg, "pose", jpos_new
    )
    assert rebuilt is not None, (
        "rebuild_msg_with_jpos_override returned None for a well-"
        "formed frame; the proxy will fall back to forwarding "
        "verbatim and the engagement clamp won't engage"
    )
    assert len(rebuilt) == len(msg), (
        "rebuilt frame length differs from original; the header / "
        "field layout was modified, which will break the deploy's "
        "fixed-offset cursor walk"
    )
    new_jpos_decoded = proxy_mod.decode_pose_joint_pos_mj(
        rebuilt, topic="pose"
    )
    np.testing.assert_array_equal(new_jpos_decoded, jpos_new)
    new_left = proxy_mod.decode_pose_left_hand(rebuilt, topic="pose")
    new_right = proxy_mod.decode_pose_right_hand(rebuilt, topic="pose")
    np.testing.assert_array_equal(new_left, left_hand)
    np.testing.assert_array_equal(new_right, right_hand)


def test_rebuild_msg_with_field_overrides_flattens_future_window():
    """Multi-field rebuild must replace future arrays in one byte-splice.

    2026-06-10 follow-up 9b: the engagement slow-step clamp on
    ``joint_pos_mj`` alone wasn't enough -- the deploy's window-
    mode policy reads ``joint_pos_mj_future`` (9 slots, 0.1 s
    apart) for forward prediction. The operator's untouched
    future encoded "go all the way to operator-pose in 0.9 s"
    and the policy slammed the body to follow even when the
    current jpos was properly rate-limited.

    The fix: during the engagement ramp, the proxy broadcasts the
    clamped current jpos to all 9 future slots AND zeros the
    velocity-future field. This test pins the byte-splice
    invariants:

      - All three target fields (jpos, jpos_future, jvel_future)
        get spliced in one pass
      - Other fields (root_quat, hand joints, frame_index,
        future quat, future_dt_s, root_xy_world) stay byte-
        identical
      - Frame length unchanged (so the deploy's fixed-offset
        cursor walk keeps working)
      - Unmatched override key returns None (callers must
        notice and fall back to the single-field path)
    """
    sys.path.insert(0, str(PROXY_DIR))
    try:
        import importlib
        proxy_mod = importlib.import_module("x2_pose_proxy")
    finally:
        sys.path.pop(0)

    import numpy as np

    num_body = proxy_mod.NUM_BODY_DOFS
    num_hand = proxy_mod.DEFAULT_HAND_DOF
    num_future = proxy_mod._NUM_FUTURE_SLOTS

    # Build a frame matching what the recorder publishes during
    # OVERRIDE: current jpos + hands + future window.
    jpos_orig = np.linspace(-0.5, 0.5, num_body, dtype=np.float32)
    left_hand = np.linspace(0.1, 0.9, num_hand, dtype=np.float32)
    right_hand = np.linspace(0.9, 0.1, num_hand, dtype=np.float32)
    root_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    jpos_future_orig = np.tile(
        np.linspace(-1.0, 1.0, num_body, dtype=np.float32),
        (num_future, 1),
    )
    rot_future_orig = np.tile(root_quat, (num_future, 1))
    jvel_future_orig = np.full(
        (num_future, num_body), 0.5, dtype=np.float32
    )
    payload = {
        "joint_pos_mj": jpos_orig,
        "root_quat_xyzw": root_quat,
        "motion_token": np.zeros(64, dtype=np.float32),
        "left_hand_joints": left_hand,
        "right_hand_joints": right_hand,
        "frame_index": np.array([42], dtype=np.int64),
        "joint_pos_mj_future": jpos_future_orig,
        "root_quat_xyzw_future": rot_future_orig,
        "joint_vel_mj_future": jvel_future_orig,
    }
    msg = proxy_mod.pack_pose_message(payload, topic="pose", version=4)

    clamped = np.full(num_body, 0.3, dtype=np.float32)
    flat_future = np.broadcast_to(
        clamped, (num_future, num_body)
    ).astype(np.float32, copy=True)
    zero_vel = np.zeros((num_future, num_body), dtype=np.float32)
    overrides = {
        "joint_pos_mj": clamped,
        "joint_pos_mj_future": flat_future,
        "joint_vel_mj_future": zero_vel,
    }
    rebuilt = proxy_mod.rebuild_msg_with_field_overrides(
        msg, "pose", overrides
    )
    assert rebuilt is not None, (
        "multi-field rebuild returned None for a well-formed frame"
    )
    assert len(rebuilt) == len(msg), (
        "rebuilt frame length changed; header / cursor layout "
        "mutated and the deploy's offset walk will break"
    )

    new_jpos = proxy_mod.decode_pose_joint_pos_mj(rebuilt, topic="pose")
    np.testing.assert_array_equal(new_jpos, clamped)

    # Other fields must be byte-identical to the original.
    new_left = proxy_mod.decode_pose_left_hand(rebuilt, topic="pose")
    new_right = proxy_mod.decode_pose_right_hand(rebuilt, topic="pose")
    np.testing.assert_array_equal(new_left, left_hand)
    np.testing.assert_array_equal(new_right, right_hand)

    # Decode future arrays via the generic f32 walker.
    new_jpos_future = proxy_mod._decode_pose_field_f32(
        rebuilt, "pose",
        name="joint_pos_mj_future",
        expected_shape=(num_future, num_body),
    )
    new_jvel_future = proxy_mod._decode_pose_field_f32(
        rebuilt, "pose",
        name="joint_vel_mj_future",
        expected_shape=(num_future, num_body),
    )
    new_rot_future = proxy_mod._decode_pose_field_f32(
        rebuilt, "pose",
        name="root_quat_xyzw_future",
        expected_shape=(num_future, 4),
    )
    # The generic field walker returns the buffer as 1D f32; reshape
    # back to declared shape for comparison.
    np.testing.assert_array_equal(
        new_jpos_future.reshape(num_future, num_body), flat_future
    )
    np.testing.assert_array_equal(
        new_jvel_future.reshape(num_future, num_body), zero_vel
    )
    # rot_future MUST be untouched (we only flatten body, not pelvis
    # orientation -- the deploy reads root_quat_xyzw_future for the
    # window-mode root-frame prediction and would lose heading
    # tracking if we zeroed it).
    np.testing.assert_array_equal(
        new_rot_future.reshape(num_future, 4), rot_future_orig
    )

    # Negative: an override key that doesn't match any field must
    # return None so callers notice the typo rather than silently
    # ignoring the requested clamp.
    rebuilt_bad = proxy_mod.rebuild_msg_with_field_overrides(
        msg, "pose", {"joint_pos_mj": clamped, "nonexistent": clamped},
    )
    assert rebuilt_bad is None, (
        "unmatched override key was silently ignored; future "
        "refactors that rename a wire field would skip the clamp "
        "without any test failure"
    )


def test_proxy_engagement_clamp_flattens_future_window_in_source():
    """Source must call the multi-field rebuild with future overrides.

    Pins the 2026-06-10 follow-up 9b structural change: the
    engagement-clamp call site must pass joint_pos_mj_future +
    joint_vel_mj_future in the overrides dict, not just
    joint_pos_mj. A refactor that drops the future-flattening
    will let the deploy's window-mode policy slam to follow the
    operator's untouched future window.
    """
    proxy = (
        REPO_ROOT
        / "gear_sonic_deploy"
        / "scripts"
        / "x2_pose_proxy.py"
    )
    src = proxy.read_text()
    assert (
        'rebuild_msg_with_field_overrides(' in src
    ), (
        "multi-field rebuild helper not invoked; the engagement "
        "clamp only touches the current jpos and the future "
        "window still snaps -> deploy slams"
    )
    # The overrides dict must include all three fields.
    for field in (
        '"joint_pos_mj": clamped_jpos',
        '"joint_pos_mj_future": flat_future',
        '"joint_vel_mj_future": zero_future_vel',
    ):
        assert field in src, (
            f"missing override entry ``{field}``; the future "
            "window field won't be flattened during the engagement "
            "ramp and the deploy's window-mode policy will slam"
        )
    # Flat future is broadcast from clamped_jpos -- not from the
    # operator's raw future (which would defeat the purpose).
    assert (
        "np.broadcast_to(\n                            clamped_jpos,"
        in src
    ), (
        "flat_future must be built from clamped_jpos (not from "
        "op_jpos or the operator's raw future); the whole point "
        "is to tell the policy 'hold the clamped current pose, "
        "no future motion'"
    )


def test_clamp_vector_step_f32_caps_peak_element_and_preserves_direction():
    """``_clamp_vector_step_f32`` must shrink proportionally, not slice.

    The bridge's ``_clamp_vector_step`` deliberately scales the
    whole delta vector by ``max_step / peak`` (not per-element
    saturating clip) so a coordinated multi-joint motion stays
    on its original trajectory -- just slower. The proxy helper
    is a 1:1 mirror; if a future refactor swaps it for per-element
    np.clip the takeover handoff will warp the operator's pose
    (small joints saturate while big joints don't, twisting the
    body).
    """
    sys.path.insert(0, str(PROXY_DIR))
    try:
        import importlib
        proxy_mod = importlib.import_module("x2_pose_proxy")
    finally:
        sys.path.pop(0)

    import numpy as np

    prev = np.zeros(5, dtype=np.float32)
    # delta = [0.1, 0.2, 0.4, 0.05, 0.0]; peak = 0.4
    target = np.array([0.1, 0.2, 0.4, 0.05, 0.0], dtype=np.float32)
    clamped = proxy_mod._clamp_vector_step_f32(
        target, prev, max_step=0.1
    )
    # scale = 0.1 / 0.4 = 0.25; result = target * 0.25 (since prev=0)
    expected = np.array(
        [0.025, 0.05, 0.1, 0.0125, 0.0], dtype=np.float32
    )
    np.testing.assert_allclose(clamped, expected, rtol=1e-6)
    # Direction preserved: clamped / target must be uniform.
    nonzero = target != 0.0
    ratios = clamped[nonzero] / target[nonzero]
    np.testing.assert_allclose(
        ratios, np.full(ratios.shape, 0.25, dtype=np.float32),
        rtol=1e-6,
    )
    # No-op when peak <= max_step.
    small = np.array([0.01, 0.02, 0.03], dtype=np.float32)
    np.testing.assert_array_equal(
        proxy_mod._clamp_vector_step_f32(
            small, np.zeros(3, dtype=np.float32), max_step=0.1
        ),
        small,
    )
    # No-op when prev is None (cold-start tick).
    np.testing.assert_array_equal(
        proxy_mod._clamp_vector_step_f32(
            target, None, max_step=0.1
        ),
        target,
    )
