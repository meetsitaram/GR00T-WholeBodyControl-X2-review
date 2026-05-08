#!/usr/bin/env python3
"""YAML real-deploy tuning config -> CLI args translator for deploy_x2.sh.

Reads a YAML file from gear_sonic_deploy/configs/real_deploy_tuning/ and
prints the corresponding ``--flag VALUE`` tokens, one per line, on stdout.
The bash wrapper consumes them via ``mapfile -t`` and prepends them to the
ROS2_ARGS list -- explicit CLI flags on the deploy_x2.sh command line still
override anything the config file sets.

PARITY RULE
-----------
This translator is invoked ONLY from deploy_x2.sh in ``local`` / ``onbot``
mode. Sim profiles deliberately bypass it so the bit-exact C++<->Python
parity surface (eval_x2_mujoco.py vs the deploy binary in MuJoCo) cannot be
silently perturbed by a tuning preset. See gear_sonic_deploy/configs/
real_deploy_tuning/README.md for the rationale.

USAGE
-----
    python3 tuning_config_to_args.py PATH                # print flags
    python3 tuning_config_to_args.py PATH --validate     # parse + exit 0
    python3 tuning_config_to_args.py --schema            # print schema doc
    python3 tuning_config_to_args.py --list-keys         # print supported
                                                          # YAML keys
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Callable

# YAML key -> (CLI flag name, value formatter, optional validator).
# Adding a new tuning knob is a one-line change here PLUS a one-line entry
# in _schema.yaml + the matching --flag in x2_deploy_onnx_ref.cpp /
# deploy_x2.sh forwarding. Keep this table sorted by CLI flag.
def _fmt_float(v: Any) -> str:
    return f"{float(v):.6g}"


def _fmt_int(v: Any) -> str:
    return str(int(v))


def _check_nonneg(name: str, v: Any) -> None:
    if float(v) < 0:
        raise ValueError(f"{name}: expected >= 0, got {v!r}")


KEY_TO_FLAG: dict[str, tuple[str, Callable[[Any], str], Callable[[str, Any], None] | None]] = {
    # Existing C++ binary flags. Pulling them into the YAML lets one preset
    # capture an entire real-deploy scenario in a single file.
    "action_clip":     ("--action-clip",      _fmt_float, None),
    "max_target_dev":  ("--max-target-dev",   _fmt_float, None),
    "ramp_seconds":    ("--ramp-seconds",     _fmt_float, _check_nonneg),
    "return_seconds":  ("--return-seconds",   _fmt_float, _check_nonneg),
    "tilt_cos":        ("--tilt-cos",         _fmt_float, None),
    # New post-policy filters (parity-safe by construction; see C++ comment
    # next to CliArgs::target_lpf_hz for why this is dump-invisible).
    "target_lpf_hz":   ("--target-lpf-hz",    _fmt_float, _check_nonneg),
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "PyYAML is required to parse real-deploy tuning configs. "
            "Inside docker_x2 it ships pre-installed; on the host run "
            "`pip install pyyaml`. Original error: " + str(e)
        ) from e
    if not path.is_file():
        raise SystemExit(f"tuning config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(
            f"tuning config {path}: top-level must be a YAML mapping, "
            f"got {type(data).__name__}"
        )
    return data


def translate(path: Path) -> list[str]:
    """Read PATH, return the list of CLI tokens to forward to the binary."""
    data = _load_yaml(path)
    args: list[str] = []
    unknown: list[str] = []
    for key, value in data.items():
        # Tolerate descriptive metadata that doesn't map to a flag.
        if key in ("description", "name", "notes", "_schema_version"):
            continue
        if value is None:
            # Explicit null = "leave this knob at the binary's default".
            continue
        if key not in KEY_TO_FLAG:
            unknown.append(key)
            continue
        flag, fmt, validator = KEY_TO_FLAG[key]
        if validator is not None:
            validator(key, value)
        args.append(flag)
        args.append(fmt(value))
    if unknown:
        raise SystemExit(
            f"tuning config {path}: unknown keys {unknown!r}. "
            f"Supported keys: {sorted(KEY_TO_FLAG)} (plus 'description', "
            f"'name', 'notes', '_schema_version' which are ignored). "
            f"See gear_sonic_deploy/configs/real_deploy_tuning/_schema.yaml."
        )
    return args


def _print_schema() -> None:
    print("# Supported real-deploy tuning keys")
    print("# Each maps to a single deploy binary CLI flag.")
    print("# null / unset = use binary default (which matches sim parity).")
    print()
    for key, (flag, _fmt, _v) in sorted(KEY_TO_FLAG.items()):
        print(f"{key}: <number>   # -> {flag}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("path", nargs="?", help="Path to a YAML tuning config")
    p.add_argument("--validate", action="store_true",
                   help="Parse and validate without printing flags")
    p.add_argument("--schema", action="store_true",
                   help="Print the supported key set and exit")
    p.add_argument("--list-keys", action="store_true",
                   help="Print supported keys (one per line) and exit")
    args = p.parse_args()

    if args.schema:
        _print_schema()
        return 0
    if args.list_keys:
        for k in sorted(KEY_TO_FLAG):
            print(k)
        return 0
    if args.path is None:
        p.error("path is required (unless --schema / --list-keys)")
    tokens = translate(Path(args.path))
    if args.validate:
        return 0
    for tok in tokens:
        print(tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
