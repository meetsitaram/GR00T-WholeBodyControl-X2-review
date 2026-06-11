"""Smoke tests for the pose-pipeline plumbing in
``gear_sonic/scripts/run_x2_vla_runtime.sh``.

The 2026-06-11 pose_mux_split refactor replaced the single-process
``x2_pose_proxy.py`` (spawned in sim, re-implemented on PC2) with two
purpose-built processes:

  * ``x2_pose_mux``      -- laptop-side N-to-1 pose merger (sim + real)
  * ``x2_pose_watchdog`` -- single-input fallback ladder (PC2 in real
                            mode; loopback in sim mode)

The dual-source semantics themselves are covered by
``tests/test_x2_pose_mux_dual_source.py`` (gated on X2_POSE_PROXY_SMOKE=1).
The fallback ladder + wire helpers are covered by the renamed
``tests/test_x2_pose_watchdog_fallback_ladder.py`` +
``tests/test_pose_pipeline_*.py`` unit tests.

What's NOT covered by those is arg-name drift between the bash
launcher and the new mux + watchdog argparse signatures. A typo in the
launcher (``--primary-port-name`` instead of ``--primary-port``,
``--out-port`` vs ``--downstream-port``) would only fail at first
real launch, hours after the edit landed.

This file closes that gap:

1. ``bash -n`` on the launcher (catches structural bash errors).
2. ``--help`` lists the canonical CLI flag surface (mux + legacy alias).
3. CLI flags parse without falling through to the bridge catch-all.
4. The argv ``spawn_pose_mux`` / ``spawn_sim_watchdog`` build is
   accepted by the actual mux / watchdog argparse parsers.
5. Source-level pattern pins for the bridge --pub-port flip on
   --enable-takeover and the sim-deploy reading from the watchdog.

The existing wrist-bypass / handoff-slow-step / tracking-feedback
launcher-forwarding tests are preserved verbatim because those concerns
are orthogonal to the mux split.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
LAUNCHER = REPO_ROOT / "gear_sonic" / "scripts" / "run_x2_vla_runtime.sh"
MUX_SCRIPT = REPO_ROOT / "gear_sonic" / "scripts" / "x2_pose_mux.py"
WATCHDOG_SCRIPT = (
    REPO_ROOT / "gear_sonic_deploy" / "scripts" / "x2_pose_watchdog.py"
)
IDLE_X2M2 = REPO_ROOT / "gear_sonic_deploy" / "data" / "idle_stand.x2m2"
BRIDGE_SCRIPT = (
    REPO_ROOT / "gear_sonic" / "scripts" / "live_vla_publish_motion_token.py"
)


# ===========================================================================
# Helpers
# ===========================================================================
def _bash() -> str:
    bash = shutil.which("bash")
    assert bash is not None, "bash not on PATH; cannot drive the launcher"
    return bash


def _build_mux_argv(
    *,
    primary_port: int = 5571,
    out_port: int = 5556,
    override_port: int = 5560,
    vla_control_port: int = -1,
    teleop_mode_port: int = -1,
) -> list[str]:
    """Mirror the argv ``spawn_pose_mux`` builds in the bash launcher.

    Keep this in sync with the ``mux_args=( … )`` array in
    ``gear_sonic/scripts/run_x2_vla_runtime.sh::spawn_pose_mux``. Any
    divergence is exactly the bug this test exists to catch.
    """
    argv = [
        "--primary-host", "127.0.0.1",
        "--primary-port", str(primary_port),
        "--primary-topic", "pose",
        "--out-host", "*",
        "--out-port", str(out_port),
        "--out-topic", "pose",
        "--override-host", "127.0.0.1",
        "--override-port", str(override_port),
        "--override-topic", "pose",
        "--override-stale-ms", "200",
        "--override-frozen-ticks", "10",
        "--override-frozen-l2-tol", "5e-3",
        "--override-engage-motion-ticks", "10",
        "--engagement-max-wire-step", "0.012",
        "--engagement-steady-wire-step", "0.035",
        "--engagement-step-ramp-ticks", "250",
        "--rate-hz", "50",
        "--status-every-s", "5.0",
    ]
    if teleop_mode_port > 0:
        argv += [
            "--teleop-mode-host", "127.0.0.1",
            "--teleop-mode-port", str(teleop_mode_port),
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


def _build_watchdog_argv(
    *,
    upstream_port: int = 5556,
    downstream_port: int = 5558,
    idle_x2m2: Path | None = None,
) -> list[str]:
    """Mirror the argv ``spawn_sim_watchdog`` builds in the launcher."""
    return [
        "--upstream-host", "127.0.0.1",
        "--upstream-port", str(upstream_port),
        "--upstream-topic", "pose",
        "--downstream-host", "127.0.0.1",
        "--downstream-port", str(downstream_port),
        "--downstream-topic", "pose",
        "--idle-x2m2", str(idle_x2m2 or IDLE_X2M2),
        "--idle-stale-ms", "300",
        "--idle-mode", "blend",
        "--hold-last-secs", "10.0",
        "--blend-secs", "3.0",
        "--no-x2-debug-yaw-track",
    ]


# ===========================================================================
# Launcher syntax + --help drift detection
# ===========================================================================
def test_launcher_bash_syntax_ok() -> None:
    """``bash -n`` catches structural bash errors in the launcher."""
    assert LAUNCHER.is_file(), f"launcher missing: {LAUNCHER}"
    proc = subprocess.run(
        [_bash(), "-n", str(LAUNCHER)],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, (
        f"bash -n failed (rc={proc.returncode}):\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )


def test_launcher_help_lists_takeover_and_legacy_flags() -> None:
    """``--help`` must document both the new --enable-takeover flag and
    the legacy --pose-proxy-* / --vla-control-* aliases that operators
    have in their existing runbooks. Pinning the visible surface in
    the help output is the cheapest way to keep the case-statement
    and the heredoc in lock-step."""
    proc = subprocess.run(
        [_bash(), str(LAUNCHER), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, (
        f"--help failed (rc={proc.returncode}):\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    for flag in (
        # New master switch
        "--enable-takeover",
        # vla_control (preserved through the refactor)
        "--vla-control-port",
        "--vla-control-host",
        "--vla-cold-restart-hold-ticks",
        "--vla-handoff-max-hold-ticks",
        # Legacy aliases -- preserved as pass-through to the mux so
        # operator runbooks don't break.
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
        # Tracking feedback (orthogonal to the split; preserved)
        "--vla-tracking-feedback",
        "--no-vla-tracking-feedback",
        "--vla-tracking-soft-rad",
        "--vla-tracking-hard-rad",
        "--vla-tracking-velocity-margin",
        "--vla-tracking-velocity-floor-rad-tick",
        "--vla-tracking-stale-ms",
        # Wrist bypass (orthogonal; preserved)
        "--wrist-bypass",
    ):
        assert flag in proc.stdout, (
            f"{flag} missing from launcher --help; the case statement "
            f"and the heredoc have drifted."
        )


def test_launcher_accepts_takeover_cli_flags(tmp_path: Path) -> None:
    """CLI flags must be parsed without falling into the ``*) ARGS+=``
    catch-all (which would silently forward them to the bridge and the
    operator would see "bridge: unrecognized argument" hours later).

    We run the launcher with ``preflight`` + ``SKIP_PREFLIGHT=1`` so it
    parses argv, runs no probes, and exits cheaply.
    """
    run_dir = tmp_path / "preflight_run"
    proc = subprocess.run(
        [
            _bash(),
            str(LAUNCHER),
            "preflight",
            "--enable-takeover",
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
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "SKIP_PREFLIGHT": "1"},
    )
    combined = proc.stdout + proc.stderr
    for flag in (
        "--enable-takeover",
        "--vla-control-port",
        "--pose-proxy-override-port",
        "--pose-proxy-teleop-mode-host",
        "--pose-proxy-downstream-port",
        "--vla-cold-restart-hold-ticks",
    ):
        assert f"Unknown argument: {flag}" not in combined, (
            f"launcher rejected {flag} as unknown ({combined!r})"
        )
    # preflight either succeeds (rc=0) or fails with a domain-specific
    # reason (model missing, etc.). The forbidden outcome is rc=2 (bash
    # usage error) which would indicate a syntax fault in the case
    # statement.
    assert proc.returncode != 2, (
        f"launcher preflight returned bash usage error rc=2; argv "
        f"didn't parse. stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )


# ===========================================================================
# Mux argparse + spawn argv parity
# ===========================================================================
def test_mux_help_lists_required_args() -> None:
    """The mux must accept the canonical CLI surface the launcher
    builds. Cheapest sanity check: --help completes cleanly with the
    expected flag set in the output."""
    proc = subprocess.run(
        [sys.executable, str(MUX_SCRIPT), "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"mux --help failed (rc={proc.returncode}): {proc.stderr!r}"
    )
    for flag in (
        "--primary-host", "--primary-port", "--primary-topic",
        "--out-host", "--out-port", "--out-topic",
        "--override-host", "--override-port", "--override-topic",
        "--override-stale-ms",
        "--override-frozen-ticks", "--override-frozen-l2-tol",
        "--override-engage-motion-ticks",
        "--engagement-max-wire-step", "--engagement-steady-wire-step",
        "--engagement-step-ramp-ticks",
        "--teleop-mode-host", "--teleop-mode-port",
        "--teleop-mode-topic", "--teleop-mode-stale-ms",
        "--vla-control-bind-host", "--vla-control-port",
        "--vla-control-topic",
        "--rate-hz", "--status-every-s",
    ):
        assert flag in proc.stdout, (
            f"{flag} missing from mux --help; CLI surface drift"
        )


@pytest.mark.parametrize(
    "vla_control_port,teleop_mode_port",
    [
        (-1, -1),     # arbitration only, no edge events, no strict mode
        (5559, -1),   # arbitration + edge events
        (5559, 5564), # full takeover topology
    ],
)
def test_spawn_pose_mux_argv_parses(
    vla_control_port: int, teleop_mode_port: int,
) -> None:
    """The launcher's spawn_pose_mux argv must pass mux argparse.

    Catches typos / arg-name drift between the bash launcher and the
    mux script. We run the mux with ``--help`` first to confirm the
    parser is well-formed, then re-invoke it with the launcher-style
    argv plus an explicit ``--help`` tail. argparse runs validation
    before --help, so a bogus arg trips before the help text prints.
    """
    argv = _build_mux_argv(
        vla_control_port=vla_control_port,
        teleop_mode_port=teleop_mode_port,
    )
    proc = subprocess.run(
        [sys.executable, str(MUX_SCRIPT)] + argv + ["--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"mux rejected launcher argv (rc={proc.returncode}):\n"
        f"argv={argv}\nstderr={proc.stderr!r}"
    )


# ===========================================================================
# Watchdog argparse + spawn argv parity
# ===========================================================================
def test_watchdog_help_lists_required_args() -> None:
    proc = subprocess.run(
        [sys.executable, str(WATCHDOG_SCRIPT), "--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"watchdog --help failed (rc={proc.returncode}): {proc.stderr!r}"
    )
    for flag in (
        "--upstream-host", "--upstream-port", "--upstream-topic",
        "--downstream-host", "--downstream-port", "--downstream-topic",
        "--idle-x2m2", "--idle-stale-ms", "--idle-mode",
        "--hold-last-secs", "--blend-secs",
        "--no-x2-debug-yaw-track",
    ):
        assert flag in proc.stdout, (
            f"{flag} missing from watchdog --help; CLI surface drift"
        )


def test_watchdog_rejects_legacy_takeover_flags() -> None:
    """The 2026-06-11 split moved manual-takeover args to the mux.
    Anyone running the watchdog with legacy --override-port /
    --vla-control-port flags MUST get a clean migration error pointing
    at the milestone doc (not a confusing argparse 'unrecognized
    argument' message).
    """
    proc = subprocess.run(
        [
            sys.executable, str(WATCHDOG_SCRIPT),
            "--upstream-host", "127.0.0.1",
            "--upstream-port", "5556",
            "--downstream-port", "5558",
            "--idle-x2m2", str(IDLE_X2M2),
            # The flag that should trigger the migration error:
            "--override-port", "5560",
        ],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 2, (
        f"watchdog should exit 2 on legacy takeover flag; got "
        f"rc={proc.returncode}, stderr={proc.stderr!r}"
    )
    assert "x2_pose_mux" in proc.stderr, (
        f"watchdog migration error should mention x2_pose_mux; "
        f"got stderr={proc.stderr!r}"
    )
    assert "2026-06-11" in proc.stderr, (
        f"watchdog migration error should reference the milestone "
        f"date; got stderr={proc.stderr!r}"
    )


@pytest.mark.skipif(
    not IDLE_X2M2.is_file(),
    reason=f"idle_stand.x2m2 missing at {IDLE_X2M2}",
)
def test_spawn_sim_watchdog_argv_parses() -> None:
    """The launcher's spawn_sim_watchdog argv must pass watchdog
    argparse. Same drift-prevention contract as the mux test above."""
    argv = _build_watchdog_argv()
    proc = subprocess.run(
        [sys.executable, str(WATCHDOG_SCRIPT)] + argv + ["--help"],
        capture_output=True, text=True, timeout=10,
    )
    assert proc.returncode == 0, (
        f"watchdog rejected launcher argv (rc={proc.returncode}):\n"
        f"argv={argv}\nstderr={proc.stderr!r}"
    )


# ===========================================================================
# Launcher source-level pattern pins
# ===========================================================================
def test_launcher_enable_takeover_auto_promotes_loopback_ports() -> None:
    """--enable-takeover MUST promote POSE_PROXY_OVERRIDE_PORT to 5560
    and VLA_CONTROL_PORT to 5559 when the operator left them at the
    legacy disabled defaults (-1). Both ports are now pure laptop
    loopback after the 2026-06-11 split, so requiring the operator
    to re-pass them on every invocation buys nothing -- and forgetting
    --vla-control-port specifically means the bridge will NOT cold-
    restart on release (the wire snaps to the mid-decode chunk).

    Pure source-level pattern check; spawning a real subprocess just
    to verify two env-var defaults is overkill.
    """
    src = LAUNCHER.read_text()

    # Anchor block: the promotion lives directly after the
    # TAKEOVER_ENABLED resolution. Keep the search narrow so a stray
    # constant elsewhere in the script can't fake the test green.
    anchor = "if [[ \"${TAKEOVER_ENABLED}\" -eq 1 ]]; then"
    anchor_idx = src.find(anchor)
    assert anchor_idx >= 0, "TAKEOVER_ENABLED auto-promotion block missing"
    # ~2 KB after the anchor should comfortably contain the whole
    # promotion stanza without dragging in unrelated downstream logic.
    block = src[anchor_idx : anchor_idx + 2048]

    assert "POSE_PROXY_OVERRIDE_PORT=5560" in block, (
        "--enable-takeover must auto-promote POSE_PROXY_OVERRIDE_PORT "
        "to the recorder's canonical PUB port (5560) when the operator "
        "didn't override it; otherwise the mux has no override SUB to "
        "listen on and takeover silently no-ops"
    )
    assert "VLA_CONTROL_PORT=5559" in block, (
        "--enable-takeover must auto-promote VLA_CONTROL_PORT to 5559 "
        "(the canonical loopback port for mux -> bridge edge events) "
        "when the operator didn't override it; otherwise the bridge "
        "won't cold-restart on operator release and the wire snaps "
        "back to whatever VLA chunk is mid-decode"
    )
    assert "VLA_CONTROL_HOST=127.0.0.1" in block, (
        "--enable-takeover must default VLA_CONTROL_HOST to loopback "
        "since the mux is co-located with the bridge after the "
        "2026-06-11 split (the pre-split default was --pc2-host, "
        "which would cross wifi for no reason)"
    )

    # The explicit-disable knob (--vla-control-port 0) MUST still
    # survive auto-promotion. The guard pattern is `... -lt 0` so a
    # zero value short-circuits the promotion.
    assert "POSE_PROXY_OVERRIDE_PORT}\" -lt 0" in block, (
        "auto-promotion guard must be -lt 0 (not -le 0 or != 0); "
        "operators rely on --pose-proxy-override-port 0 to opt out"
    )
    assert "VLA_CONTROL_PORT}\" -lt 0" in block, (
        "auto-promotion guard must be -lt 0 so --vla-control-port 0 "
        "remains a valid opt-out knob"
    )


def test_launcher_bridge_pub_port_uses_internal_port_on_takeover() -> None:
    """With --enable-takeover the bridge must publish on
    BRIDGE_POSE_PUB_PORT (= BRIDGE_POSE_PORT_INTERNAL, default 5571)
    instead of LAPTOP_POSE_PORT (5556). Otherwise the mux would clash
    with the bridge over the canonical pose port and one of them
    silently fails to bind.

    2026-06-11 follow-up: the internal port moved from 5570 -> 5571
    after we discovered :5570 was already owned by the kplanner stack's
    x2_debug_to_robot_pose_bridge (publishes 'robot_pose' topic). If
    this pin ever moves again, also update tests/_build_mux_argv()'s
    primary_port default, pick_place_commands.md, and the 2026-06-11
    pose-pipeline-split milestone doc.

    Pins the BRIDGE_ARGS slot for --pub-port. Source-level pattern
    check (no subprocess needed)."""
    src = LAUNCHER.read_text()
    assert "BRIDGE_POSE_PORT_INTERNAL:=5571" in src, (
        "BRIDGE_POSE_PORT_INTERNAL default missing/wrong; the bridge "
        "won't have a clean internal port to publish on when takeover "
        "is enabled (and :5570 collides with the kplanner stack's "
        "x2_debug_to_robot_pose_bridge per 2026-06-11 fix)"
    )
    assert '--pub-port "$BRIDGE_POSE_PUB_PORT"' in src, (
        "bridge --pub-port must use BRIDGE_POSE_PUB_PORT (not "
        "LAPTOP_POSE_PORT directly); otherwise the mux + bridge will "
        "fight over port 5556 when --enable-takeover is set"
    )


def test_launcher_spawn_pose_mux_uses_pipeline_topology() -> None:
    """spawn_pose_mux MUST plumb the right ports through:
      - --primary-port reads BRIDGE_POSE_PUB_PORT (bridge's internal port)
      - --out-port binds LAPTOP_POSE_PORT (the canonical pose port)
      - --override-port pulls from POSE_PROXY_OVERRIDE_PORT (with the
        recorder default 5560 when unset)
    so the same env-var surface that drove the old proxy keeps working.
    """
    src = LAUNCHER.read_text()
    start = src.find("\nspawn_pose_mux()")
    assert start >= 0, "spawn_pose_mux() not found in launcher"
    end = src.find("\n}\n", start)
    assert end > start
    body = src[start:end]
    assert '--primary-port "$BRIDGE_POSE_PUB_PORT"' in body, (
        "spawn_pose_mux --primary-port must use BRIDGE_POSE_PUB_PORT"
    )
    assert '--out-port "$LAPTOP_POSE_PORT"' in body, (
        "spawn_pose_mux --out-port must bind LAPTOP_POSE_PORT (so "
        "external consumers -- PC2 watchdog or sim watchdog -- can "
        "continue SUBing at the canonical port)"
    )
    assert "POSE_PROXY_OVERRIDE_PORT" in body, (
        "spawn_pose_mux must consume POSE_PROXY_OVERRIDE_PORT so the "
        "existing operator runbook keeps working"
    )
    # Engagement ramp args must be forwarded to the mux's matching
    # CLI surface (the mux took these over from the old proxy).
    for arg in (
        '--engagement-max-wire-step "$POSE_PROXY_ENGAGEMENT_MAX_WIRE_STEP"',
        '--engagement-steady-wire-step "$POSE_PROXY_ENGAGEMENT_STEADY_WIRE_STEP"',
        '--engagement-step-ramp-ticks "$POSE_PROXY_ENGAGEMENT_STEP_RAMP_TICKS"',
    ):
        assert arg in body, (
            f"spawn_pose_mux missing {arg!r}; the engagement ramp will "
            f"silently fall back to the mux's parser defaults"
        )


def test_launcher_spawn_sim_watchdog_uses_pipeline_topology() -> None:
    """spawn_sim_watchdog MUST read from the mux's output port
    (LAPTOP_POSE_PORT) and PUB to the sim deploy port
    (POSE_PROXY_DOWNSTREAM_PORT). Pins the dataflow:
    bridge -> mux *:5556 -> watchdog *:5558 -> sim deploy."""
    src = LAUNCHER.read_text()
    start = src.find("\nspawn_sim_watchdog()")
    assert start >= 0, "spawn_sim_watchdog() not found in launcher"
    end = src.find("\n}\n", start)
    assert end > start
    body = src[start:end]
    assert '--upstream-port "$LAPTOP_POSE_PORT"' in body, (
        "spawn_sim_watchdog --upstream-port must read LAPTOP_POSE_PORT "
        "(the mux's output, not the bridge's internal port)"
    )
    assert '--downstream-port "$POSE_PROXY_DOWNSTREAM_PORT"' in body, (
        "spawn_sim_watchdog --downstream-port must use "
        "POSE_PROXY_DOWNSTREAM_PORT (where the sim deploy SUBs from)"
    )
    assert "--no-x2-debug-yaw-track" in body, (
        "sim watchdog must disable yaw rebase (no deploy x2_debug PUB "
        "during sim warmup)"
    )


def test_launcher_pose_mux_spawned_in_both_sim_and_real_takeover() -> None:
    """spawn_pose_mux must be called in BOTH the sim and real
    branches when TAKEOVER_ENABLED=1, because the laptop-side mux is
    what replaces the PC2-side dual-source arbitration in either mode."""
    src = LAUNCHER.read_text()
    # The sim path spawns mux + sim watchdog inside the SIM_PROXY_ENABLED
    # block; the real path spawns mux on TAKEOVER_ENABLED right after
    # the bridge PUB binds.
    sim_block = src[src.find('"sim manual-takeover plumbing ON:"'):]
    sim_block = sim_block[:sim_block.find('spawn_sim_deploy')]
    assert "spawn_pose_mux" in sim_block, (
        "spawn_pose_mux must be invoked in the sim manual-takeover "
        "block (the mux is the merge/arbitration brain in sim too)"
    )
    real_block = src[src.find('"real-robot manual-takeover plumbing ON:"'):]
    real_block = real_block[:real_block.find("\nfi\n")]
    assert "spawn_pose_mux" in real_block, (
        "spawn_pose_mux must be invoked in the real-robot manual-"
        "takeover block (the mux runs on the LAPTOP next to the "
        "bridge; PC2 only has the watchdog)"
    )


def test_launcher_sim_watchdog_only_spawned_in_sim_mode() -> None:
    """The sim watchdog provides the fallback ladder when the sim
    deploy is colocated on the laptop. Real-robot deployments get
    their watchdog from PC2's x2_pc2_daemons.sh; the launcher MUST NOT
    spawn a second watchdog locally in real mode."""
    src = LAUNCHER.read_text()
    # The only spawn_sim_watchdog invocation must live inside the sim
    # branch (after "policy ready; spawning sim deploy"). The real-
    # robot block must NOT call spawn_sim_watchdog.
    real_block = src[src.find('"real-robot manual-takeover plumbing ON:"'):]
    real_block = real_block[:real_block.find("\nfi\n")]
    assert "spawn_sim_watchdog" not in real_block, (
        "spawn_sim_watchdog must NOT be invoked in the real-robot "
        "manual-takeover block -- PC2's x2_pose_watchdog handles "
        "fallback on the robot side"
    )


def test_legacy_spawn_sim_proxy_alias_exists() -> None:
    """Older runbooks / postmortem scripts may grep for the
    spawn_sim_proxy / stop_sim_proxy function names. Keep wrapper
    aliases that delegate to the new mux + watchdog spawns so the
    runbook stays buildable."""
    src = LAUNCHER.read_text()
    assert "stop_sim_proxy()" in src, (
        "stop_sim_proxy() alias must exist (delegates to "
        "stop_sim_watchdog + stop_pose_mux). Operator scripts grep "
        "for this name."
    )


def test_kill_stale_sim_processes_targets_new_scripts() -> None:
    """kill_stale_sim_processes must look for the new x2_pose_mux.py
    and x2_pose_watchdog.py paths -- a stale daemon from a previous
    run otherwise wedges the ports forever."""
    src = LAUNCHER.read_text()
    start = src.find("kill_stale_sim_processes()")
    assert start >= 0
    end = src.find("\n}\n", start)
    body = src[start:end]
    assert "gear_sonic/scripts/x2_pose_mux.py" in body, (
        "kill_stale_sim_processes must target the new x2_pose_mux.py "
        "path"
    )
    assert "gear_sonic_deploy/scripts/x2_pose_watchdog.py" in body, (
        "kill_stale_sim_processes must target the new "
        "x2_pose_watchdog.py path"
    )


# ===========================================================================
# spawn_sim_deploy: OmniHand redirect through the watchdog (preserved)
# ===========================================================================
def test_spawn_sim_deploy_routes_omnihand_sub_through_pipeline_when_proxy_on() -> None:
    """When the laptop mux + sim watchdog are in the wire, the sim
    deploy's OmniHand ZMQ SUB MUST go through the same wire as the
    body joints. Otherwise the operator's finger commands silently
    die on the recorder -> mux -> watchdog -> deploy hop (the body
    SUB is routed; the OmniHand SUB defaults to LAPTOP_POSE_PORT and
    misses the merged wire). This was the 2026-06-10 "fingers still
    not responding" regression."""
    src = LAUNCHER.read_text()
    start = src.find("\nspawn_sim_deploy()")
    assert start >= 0, "spawn_sim_deploy() not found in launcher"
    end = src.find("\n}\n", start)
    func_body = src[start:end]
    assert "--sim-with-omnihand" in func_body
    assert "--sim-hand-zmq-host" in func_body, (
        "spawn_sim_deploy must forward --sim-hand-zmq-host to deploy_x2.sh"
    )
    assert "--sim-hand-zmq-port" in func_body, (
        "spawn_sim_deploy must forward --sim-hand-zmq-port to deploy_x2.sh"
    )
    hand_zmq_idx = func_body.find("--sim-hand-zmq-host")
    preceding = func_body[:hand_zmq_idx]
    assert "SIM_PROXY_ENABLED" in preceding, (
        "the --sim-hand-zmq-host/port forwarding MUST be gated on "
        "SIM_PROXY_ENABLED (= sim + takeover) so legacy non-pipeline "
        "sim runs aren't broken by pointing OmniHand at an unbound port"
    )
    assert '"$deploy_pose_host"' in func_body, (
        "spawn_sim_deploy must reuse $deploy_pose_host for the "
        "OmniHand wire so body + fingers come from the same source"
    )
    assert '"$deploy_pose_port"' in func_body, (
        "spawn_sim_deploy must reuse $deploy_pose_port for the "
        "OmniHand wire so body + fingers come from the same source"
    )


def test_spawn_sim_deploy_forwards_wrist_bypass_via_deploy_extra_arg() -> None:
    """spawn_sim_deploy must forward $WRIST_BYPASS to deploy_x2.sh
    via two --deploy-extra-arg tokens. Same regression pin as
    2026-06-10 follow-up 7."""
    src = LAUNCHER.read_text()
    start = src.find("\nspawn_sim_deploy()")
    assert start >= 0
    end = src.find("\n}\n", start)
    func_body = src[start:end]
    assert "--wrist-bypass" in func_body
    assert func_body.count("--deploy-extra-arg") >= 3, (
        "spawn_sim_deploy must use --deploy-extra-arg at least three "
        "times: one for --disable-pose-ref-watchdog plus a key+value "
        "pair for --wrist-bypass <mode>"
    )
    assert 'WRIST_BYPASS:=ik' in src, (
        "WRIST_BYPASS default must be 'ik' (per 2026-06-10 follow-up 7)"
    )
    assert 'WRIST_BYPASS_MAX_TARGET_DEV' not in src, (
        "the global --max-target-dev auto-pair from follow-up 7 was "
        "reverted in follow-up 8 (it collapsed the robot's legs); do "
        "not reintroduce"
    )


# ===========================================================================
# Bridge-side handoff plumbing (preserved across the refactor)
# ===========================================================================
def test_launcher_forwards_handoff_max_hold_ticks_to_bridge() -> None:
    src = LAUNCHER.read_text()
    assert 'VLA_HANDOFF_MAX_HOLD_TICKS:=200' in src, (
        "VLA_HANDOFF_MAX_HOLD_TICKS default missing/wrong"
    )
    assert '--vla-handoff-max-hold-ticks)' in src, (
        "--vla-handoff-max-hold-ticks missing from launcher case "
        "statement; CLI override won't take effect"
    )
    assert (
        '--vla-handoff-max-hold-ticks "${VLA_HANDOFF_MAX_HOLD_TICKS}"'
        in src
    ), (
        "BRIDGE_ARGS append for --vla-handoff-max-hold-ticks missing"
    )


def test_launcher_forwards_handoff_slow_step_to_bridge() -> None:
    src = LAUNCHER.read_text()
    assert 'VLA_HANDOFF_MAX_WIRE_STEP:=0.012' in src
    assert 'VLA_HANDOFF_STEP_RAMP_TICKS:=250' in src
    for flag in (
        '--vla-handoff-max-wire-step)',
        '--vla-handoff-step-ramp-ticks)',
    ):
        assert flag in src, f"{flag} missing from case statement"
    for wiring in (
        '--vla-handoff-max-wire-step "${VLA_HANDOFF_MAX_WIRE_STEP}"',
        '--vla-handoff-step-ramp-ticks "${VLA_HANDOFF_STEP_RAMP_TICKS}"',
    ):
        assert wiring in src, f"BRIDGE_ARGS append {wiring!r} missing"


# ===========================================================================
# Bridge source-level pins (orthogonal to the mux split; preserved)
# ===========================================================================
def test_bridge_help_lists_handoff_max_hold_ticks() -> None:
    proc = subprocess.run(
        [sys.executable, str(BRIDGE_SCRIPT), "--help"],
        capture_output=True, text=True, timeout=20,
    )
    assert proc.returncode == 0, (
        f"bridge --help failed (rc={proc.returncode}): {proc.stderr}"
    )
    assert "--vla-handoff-max-hold-ticks" in proc.stdout
    assert "Safety cap" in proc.stdout


def test_handoff_gate_requires_nontrivial_token_magnitude() -> None:
    """Cold-restart handoff gate must include the token-norm clause
    (2026-06-10 follow-up 5; symptom is the 12:26 hand-into-table)."""
    src = BRIDGE_SCRIPT.read_text()
    assert "current_token_norm = float(np.linalg.norm(token[step]))" in src
    assert (
        "first_eligible_chunk_ready = (" in src
        and "current_token_norm > 1e-3" in src
    )
    assert (
        "# Idle wire (deploy stale, no decoder, or zero-token chunk)." in src
        and "cur_jpos = _clamp_vector_step(" in src
        and "max_wire_step" in src
    )


def test_handoff_slow_step_state_machine_in_bridge_source() -> None:
    """Bridge source must implement the 2026-06-10 follow-up 6 slow-
    step ramp + apply effective_max_step on both branches."""
    src = BRIDGE_SCRIPT.read_text()
    assert "handoff_step_remaining = 0" in src
    success_arm = "handoff_step_remaining = max(int(handoff_step_ramp_ticks), 0)"
    assert src.count(success_arm) >= 2
    assert (
        "(1.0 - ramp_progress) * float(handoff_max_wire_step)" in src
        and "ramp_progress * float(max_wire_step)" in src
    )
    assert (
        src.count("cur_jpos = _clamp_vector_step(") >= 2
        and src.count("effective_max_step") >= 2
    )


# ===========================================================================
# Tracking feedback (preserved across the refactor)
# ===========================================================================
def test_launcher_accepts_tracking_feedback_cli_flags(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "tracking_preflight"
    proc = subprocess.run(
        [
            _bash(), str(LAUNCHER),
            "preflight",
            "--vla-tracking-feedback",
            "--vla-tracking-soft-rad", "0.20",
            "--vla-tracking-hard-rad", "0.50",
            "--vla-tracking-velocity-margin", "2.0",
            "--vla-tracking-velocity-floor-rad-tick", "0.02",
            "--vla-tracking-stale-ms", "150",
            "--run-dir", str(run_dir),
        ],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "SKIP_PREFLIGHT": "1"},
    )
    combined = proc.stdout + proc.stderr
    for flag in (
        "--vla-tracking-feedback",
        "--vla-tracking-soft-rad",
        "--vla-tracking-hard-rad",
        "--vla-tracking-velocity-margin",
        "--vla-tracking-velocity-floor-rad-tick",
        "--vla-tracking-stale-ms",
    ):
        assert f"Unknown argument: {flag}" not in combined
    assert proc.returncode != 2, (
        f"launcher returned bash usage error (rc=2); flags didn't parse. "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert (
        "tracking feedback ENABLED" in combined
        or ("Tracking feedback" in combined and "ON" in combined)
    )


def test_launcher_no_tracking_feedback_default_off_in_banner(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "no_tracking_preflight"
    proc = subprocess.run(
        [
            _bash(), str(LAUNCHER),
            "preflight",
            "--run-dir", str(run_dir),
        ],
        capture_output=True, text=True, timeout=60,
        env={**os.environ, "SKIP_PREFLIGHT": "1"},
    )
    combined = proc.stdout + proc.stderr
    assert (
        "tracking feedback DISABLED" in combined
        or ("Tracking feedback" in combined and "OFF" in combined)
    )


def test_launcher_forwards_tracking_feedback_args_to_bridge() -> None:
    src = LAUNCHER.read_text()
    expected_fragment = (
        '--vla-tracking-feedback '
        '--vla-tracking-soft-rad "${VLA_TRACKING_SOFT_RAD}" '
        '--vla-tracking-hard-rad "${VLA_TRACKING_HARD_RAD}" '
        '--vla-tracking-velocity-margin "${VLA_TRACKING_VELOCITY_MARGIN}" '
        '--vla-tracking-velocity-floor-rad-tick "${VLA_TRACKING_VELOCITY_FLOOR_RAD_TICK}" '
        '--vla-tracking-velocity-max-rad-s "${VLA_TRACKING_VELOCITY_MAX_RAD_S}" '
        '--vla-tracking-overshoot-dq-rad-s "${VLA_TRACKING_OVERSHOOT_DQ_RAD_S}" '
        '--vla-tracking-damping-kd "${VLA_TRACKING_DAMPING_KD}" '
        '--vla-tracking-damping-shoulder-scale "${VLA_TRACKING_DAMPING_SHOULDER_SCALE}" '
        '--vla-tracking-stale-ms "${VLA_TRACKING_STALE_MS}"'
    )
    normalised_src = " ".join(src.split())
    normalised_expected = " ".join(expected_fragment.split())
    assert normalised_expected in normalised_src, (
        "launcher BRIDGE_ARGS+= block for tracking feedback is missing "
        "or has drifted; 2026-06-11 follow-up added "
        "--vla-tracking-velocity-max-rad-s (hard cap), "
        "--vla-tracking-overshoot-dq-rad-s (derivative gate), and "
        "--vla-tracking-damping-kd / -shoulder-scale (viscous damper)"
    )


def test_launcher_help_lists_wrist_bypass_flag() -> None:
    proc = subprocess.run(
        [_bash(), str(LAUNCHER), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0
    assert "--wrist-bypass" in proc.stdout
