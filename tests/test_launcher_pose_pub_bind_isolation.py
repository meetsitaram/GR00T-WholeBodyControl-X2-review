"""LAN-isolation pins for the laptop-side pose PUB bind.

Background (2026-06-23 incident)
================================
PC2's ``x2_pose_proxy`` is a long-lived daemon (started once via
``x2_pc2_daemons.sh start --laptop-host <LAPTOP_IP> ...`` and intentionally
kept alive across laptop sessions -- the operator never SSHes in to stop
it). It SUBs the laptop's pose PUB over wifi. ZMQ semantics: SUBs connect
out, our PUB just binds and accepts whatever attaches.

So if the laptop binds the pose PUB on ``'*'`` (= ``0.0.0.0``, all
interfaces) the wire is delivered to PC2 too. That's correct in real-robot
mode (the entire point); it's a silent safety bug in sim mode, because the
real robot starts tracking the sim wire even though the operator never
passed ``--pc2-host``. We hit this in 2026-06-23: the wrapper banner said
"sim run artifacts" + ``docker stop`` of the sim container, but the
physical robot moved.

Fix
===
Both ``run_x2_vla_runtime.sh`` and ``run_x2_replay_stack.sh`` now gate the
pose PUB bind host on the same flag they use to decide sim vs. real:
``SIM_MODE`` (derived from ``--pc2-host``) in the runtime, ``PC2_HOST``
(directly) in the replay. Sim mode binds loopback so the wire is
physically unreachable from PC2; real mode binds ``'*'`` so PC2 can SUB.
A ``PUB_BIND_HOST`` env-var escape hatch exists for the rare cross-host
sim-debug case.

What this file pins
===================
* The wrapper derives ``PUB_BIND_HOST`` from ``SIM_MODE`` / ``PC2_HOST``.
* The ``PUB_BIND_HOST`` env-var escape hatch wins (``:=`` semantics).
* Every ``--pub-host`` / ``--out-host`` arg site in the wrappers uses
  ``$PUB_BIND_HOST`` instead of a hardcoded ``'*'``.
* ``bash -n`` still passes (no structural fault from the new gating block).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNTIME_LAUNCHER = REPO_ROOT / "gear_sonic" / "scripts" / "run_x2_vla_runtime.sh"
REPLAY_LAUNCHER = REPO_ROOT / "gear_sonic" / "scripts" / "run_x2_replay_stack.sh"


def _bash() -> str:
    b = shutil.which("bash")
    assert b is not None, "bash missing from PATH; cannot drive the launchers"
    return b


# ===========================================================================
# Structural sanity (bash -n)
# ===========================================================================
@pytest.mark.parametrize(
    "launcher",
    [RUNTIME_LAUNCHER, REPLAY_LAUNCHER],
    ids=lambda p: p.name,
)
def test_launcher_bash_syntax_ok(launcher: Path) -> None:
    """The new ``PUB_BIND_HOST`` gating block must not break ``bash -n``."""
    assert launcher.is_file(), f"launcher missing: {launcher}"
    proc = subprocess.run(
        [_bash(), "-n", str(launcher)],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, (
        f"bash -n failed on {launcher.name}: rc={proc.returncode}\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )


# ===========================================================================
# Source-level pattern pins
# ===========================================================================
def test_runtime_launcher_derives_pub_bind_from_sim_mode() -> None:
    """``run_x2_vla_runtime.sh`` must derive ``PUB_BIND_HOST`` from
    ``SIM_MODE`` (which is itself derived from ``--pc2-host``). The
    ``:=`` form is mandatory -- an explicit ``PUB_BIND_HOST=*`` env
    must win as the documented cross-host sim-debug escape hatch."""
    src = RUNTIME_LAUNCHER.read_text()

    # The gating block lives directly after the SIM_MODE derivation.
    # Anchor on the start of the SIM_MODE block to keep the search
    # narrow -- a stray ``PUB_BIND_HOST=`` elsewhere in the script
    # could fake the test green.
    anchor = 'if [[ -n "$PC2_HOST" ]]; then\n    SIM_MODE=0'
    anchor_idx = src.find(anchor)
    assert anchor_idx >= 0, "SIM_MODE derivation block missing"
    # 4 KB after the anchor comfortably covers the gating stanza
    # without dragging in unrelated downstream config.
    block = src[anchor_idx : anchor_idx + 4096]

    assert 'if [[ "$SIM_MODE" -eq 1 ]]; then' in block, (
        "PUB_BIND_HOST gating must branch on SIM_MODE (so --pc2-host is "
        "the single source of truth for sim vs real)"
    )
    assert ': "${PUB_BIND_HOST:=127.0.0.1}"' in block, (
        "sim branch must default PUB_BIND_HOST to 127.0.0.1 (loopback) "
        "so the always-on PC2 pose proxy cannot reach the wire"
    )
    assert ': "${PUB_BIND_HOST:=*}"' in block, (
        "real branch must default PUB_BIND_HOST to '*' so PC2 can SUB "
        "the wire (and any debug SUB can attach)"
    )


def test_replay_launcher_derives_pub_bind_from_pc2_host() -> None:
    """``run_x2_replay_stack.sh`` does not have a separate SIM_MODE
    var; it gates on ``PC2_HOST`` directly. Same shape, same env
    escape hatch."""
    src = REPLAY_LAUNCHER.read_text()
    anchor = 'mkdir -p "${LOG_DIR}"'
    anchor_idx = src.find(anchor)
    assert anchor_idx >= 0, "expected mkdir anchor in replay launcher"
    block = src[anchor_idx : anchor_idx + 4096]

    assert 'if [[ -n "${PC2_HOST}" ]]; then' in block, (
        "PUB_BIND_HOST gating must branch on PC2_HOST"
    )
    assert ': "${PUB_BIND_HOST:=127.0.0.1}"' in block, (
        "no --pc2-host => loopback bind (LAN isolation)"
    )
    assert ': "${PUB_BIND_HOST:=*}"' in block, (
        "--pc2-host present => '*' bind so PC2 can SUB"
    )


def test_runtime_launcher_uses_pub_bind_host_at_all_call_sites() -> None:
    """Every laptop-side PUB bind in the runtime wrapper (bridge,
    mux, recorder) must read ``$PUB_BIND_HOST`` instead of a hardcoded
    ``'*'``. The recorder skips its bind in VLA subscribe-mode anyway,
    but we gate it on principle for the rare standalone-record-
    without-bridge case."""
    src = RUNTIME_LAUNCHER.read_text()

    # Bridge --pub-host (the primary leak vector -- this is what binds
    # LAPTOP_POSE_PORT when --enable-takeover is OFF).
    assert '--pub-host "$PUB_BIND_HOST"' in src, (
        "bridge BRIDGE_ARGS must use PUB_BIND_HOST"
    )

    # Mux --out-host (binds LAPTOP_POSE_PORT when --enable-takeover is
    # ON; the mux is the public PUB the PC2 watchdog SUBs).
    assert '--out-host "$PUB_BIND_HOST"' in src, (
        "spawn_pose_mux --out-host must use PUB_BIND_HOST"
    )

    # The previous hardcoded `'*'` for the bridge / mux / recorder
    # is gone. A stray `--pub-host '*'` would re-introduce the leak.
    assert "--pub-host '*'" not in src, (
        "no laptop-side --pub-host should still be hardcoded to '*' -- "
        "gate on PUB_BIND_HOST instead"
    )
    assert '--out-host "*"' not in src, (
        "no laptop-side --out-host should still be hardcoded to '*' -- "
        "gate on PUB_BIND_HOST instead"
    )


def test_replay_launcher_uses_pub_bind_host() -> None:
    src = REPLAY_LAUNCHER.read_text()
    assert '--pub-host "${PUB_BIND_HOST}"' in src, (
        "replay REPLAY_ARGS must use PUB_BIND_HOST"
    )
    assert '--pub-host "*"' not in src, (
        "no --pub-host '\"*\"' should remain hardcoded -- gate via PUB_BIND_HOST"
    )


# ===========================================================================
# Banner pins -- the bind decision must be operator-visible
# ===========================================================================
def test_runtime_launcher_banner_shows_pose_pub_bind() -> None:
    """The runtime wrapper's banner must echo the resolved
    ``PUB_BIND_HOST`` so the operator can spot a bind-host misuse
    without grepping log files."""
    src = RUNTIME_LAUNCHER.read_text()
    # Sim banner block + real banner block both must reference
    # ${PUB_BIND_HOST}:${LAPTOP_POSE_PORT}. Two separate cat heredocs.
    assert src.count("${PUB_BIND_HOST}:${LAPTOP_POSE_PORT}") >= 2, (
        "both sim and real banner blocks must echo the resolved "
        "pose PUB bind host:port to the operator"
    )


def test_replay_launcher_banner_shows_pose_pub_bind() -> None:
    src = REPLAY_LAUNCHER.read_text()
    assert "${PUB_BIND_HOST}:${POSE_PORT}" in src, (
        "replay banner must echo the resolved pose PUB bind host:port"
    )


# ===========================================================================
# End-to-end argv check (drives the launcher in preflight mode)
# ===========================================================================
def test_runtime_launcher_help_still_works() -> None:
    """The gating block must not break the launcher's ``--help`` path
    (which is the cheapest way to confirm the script still reaches
    argv parsing without aborting in the new derivation stanza)."""
    proc = subprocess.run(
        [_bash(), str(RUNTIME_LAUNCHER), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    assert proc.returncode == 0, (
        f"--help failed: rc={proc.returncode}\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )


def test_replay_launcher_help_still_works() -> None:
    proc = subprocess.run(
        [_bash(), str(REPLAY_LAUNCHER), "--help"],
        capture_output=True, text=True, timeout=15,
    )
    # The replay wrapper's usage() awk-extracts the heredoc and exits
    # rc=1 (it's a usage screen, not a "success" path).
    assert proc.returncode in (0, 1), (
        f"--help should print usage and exit 0 or 1, got rc={proc.returncode}\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )


# ===========================================================================
# Env-var override sanity (PUB_BIND_HOST=* in sim must win)
# ===========================================================================
# The ``: "${PUB_BIND_HOST:=...}"`` form is a bash-language guarantee
# (the variable's default is set ONLY if it's unset or empty), so an
# explicit ``PUB_BIND_HOST=*`` env override always wins. That semantics
# is pinned indirectly by the source-pattern tests above which assert
# the ``:=`` operator is present at the derivation site, so a runtime
# subprocess test would just re-assert bash itself. Skipping.
