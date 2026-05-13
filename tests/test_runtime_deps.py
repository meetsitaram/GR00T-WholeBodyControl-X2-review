"""Tests for ``gear_sonic.utils.install.runtime_deps``.

The helper auto-installs missing optional dependencies on first launch
of the teleop / record entry-point scripts. We don't run real pip
installs in CI; instead we monkeypatch ``subprocess.check_call`` and
``importlib.util.find_spec`` to exercise the decision logic.
"""

from __future__ import annotations

import sys
import types

import pytest

from gear_sonic.utils.install import runtime_deps


def test_dep_groups_have_expected_shape() -> None:
    """Sanity-check the package maps. Adding a dep to one of these
    dicts requires a matching entry in
    ``requirements-teleop-record.txt`` and ``gear_sonic[data_collection]``
    extras.
    """
    assert isinstance(runtime_deps.CALIBRATION_DEPS, dict)
    assert "gtts" in runtime_deps.CALIBRATION_DEPS

    assert isinstance(runtime_deps.RECORDER_DEPS, dict)
    # Recorder deps are a SUPERSET of calibration deps.
    for k in runtime_deps.CALIBRATION_DEPS:
        assert k in runtime_deps.RECORDER_DEPS
    # Recorder-specific items.
    assert "datasets" in runtime_deps.RECORDER_DEPS
    assert "av" in runtime_deps.RECORDER_DEPS
    assert "lerobot" in runtime_deps.RECORDER_DEPS


def test_already_installed_short_circuits(monkeypatch) -> None:
    """When every package is already importable, we should NOT shell
    out to pip (network / venv mutation is the user's biggest
    objection; we must respect it).
    """
    called = {"flag": False}

    def fake_check_call(*_a, **_kw):
        called["flag"] = True
        return 0

    monkeypatch.setattr(runtime_deps.subprocess, "check_call", fake_check_call)
    # Simulate "everything installed" by returning a fake spec for
    # every import name.
    monkeypatch.setattr(
        runtime_deps.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(name=name),
    )

    ok = runtime_deps.ensure_runtime_deps({"foo": "foo>=1", "bar": "bar"})
    assert ok is True
    assert called["flag"] is False


def test_missing_packages_trigger_pip_install(monkeypatch) -> None:
    """Missing import names should map to a single ``pip install`` call
    with the matching pip specs.
    """
    captured: dict[str, list[str]] = {"cmd": []}

    def fake_check_call(cmd, *_a, **_kw):
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr(runtime_deps.subprocess, "check_call", fake_check_call)
    # First call says "missing", second call (after install) says "installed".
    state = {"installed": False}

    def fake_find_spec(name):
        if state["installed"]:
            return types.SimpleNamespace(name=name)
        return None

    monkeypatch.setattr(runtime_deps.importlib.util, "find_spec", fake_find_spec)

    def install_then_flip_state(cmd, *_a, **_kw):
        state["installed"] = True
        captured["cmd"] = list(cmd)
        return 0

    monkeypatch.setattr(runtime_deps.subprocess, "check_call", install_then_flip_state)

    ok = runtime_deps.ensure_runtime_deps({"foo": "foo>=1", "bar": "bar"})
    assert ok is True
    # The pip command should include both pip-specs.
    assert sys.executable in captured["cmd"]
    assert "install" in captured["cmd"]
    assert "foo>=1" in captured["cmd"]
    assert "bar" in captured["cmd"]


def test_pip_failure_returns_false(monkeypatch) -> None:
    """A pip CalledProcessError should NOT propagate -- the script
    falls back to degraded behavior instead of crashing.
    """
    monkeypatch.setattr(runtime_deps.importlib.util, "find_spec", lambda _: None)

    def fake_check_call(*_a, **_kw):
        raise runtime_deps.subprocess.CalledProcessError(1, ["pip"])

    monkeypatch.setattr(runtime_deps.subprocess, "check_call", fake_check_call)

    ok = runtime_deps.ensure_runtime_deps({"foo": "foo>=1"})
    assert ok is False


def test_env_var_disables_auto_install(monkeypatch) -> None:
    """``GEAR_SONIC_NO_AUTO_INSTALL=1`` should turn the helper into a
    no-op even when packages are missing (used by reproducible-build
    setups that don't want surprise pip mutations).
    """
    monkeypatch.setenv("GEAR_SONIC_NO_AUTO_INSTALL", "1")
    monkeypatch.setattr(runtime_deps.importlib.util, "find_spec", lambda _: None)

    called = {"flag": False}

    def fake_check_call(*_a, **_kw):
        called["flag"] = True

    monkeypatch.setattr(runtime_deps.subprocess, "check_call", fake_check_call)

    ok = runtime_deps.ensure_runtime_deps({"foo": "foo>=1"})
    assert ok is False
    assert called["flag"] is False, "pip must not be invoked when opt-out is set"


def test_post_install_verification_catches_wrong_python(monkeypatch) -> None:
    """If pip claims success but the package still isn't importable
    (because the wheel landed in a different interpreter), we must
    return False to surface the issue.
    """
    monkeypatch.setattr(runtime_deps.importlib.util, "find_spec", lambda _: None)
    monkeypatch.setattr(runtime_deps.subprocess, "check_call", lambda *a, **kw: 0)

    ok = runtime_deps.ensure_runtime_deps({"foo": "foo>=1"})
    assert ok is False
