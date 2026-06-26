#!/usr/bin/env python3
"""Build a self-contained data bundle for the Xbox-controller demo.

Reads ``BINDINGS_LOCOMOTION``, ``BINDINGS_GESTURES`` and
``ESTOP_FOLLOWUP_PKL`` from
``gear_sonic/scripts/play_xbox_controller.py`` and copies every
referenced PKL into a target directory, **preserving the source-tree
relative paths** so the bundle can be rsynced directly over the
``gear_sonic/`` checkout on the demo machine without any further
reshuffling.

Optionally also bundles:

* the 4 locomotion playlist YAMLs (``relaxed_walk_*.yaml`` under
  ``gear_sonic/data/motions/playlists/``) for traceability;
* the launcher script itself
  (``gear_sonic/scripts/play_xbox_controller.py``);
* the operator cheatsheet (``xbox_controller_commands.md``).

The bundle layout becomes::

    demo_bundle/
      README.md                       <- this build emits, with rsync recipe
      manifest.json                   <- machine-readable inventory + sha256
      gear_sonic/
        data/motions/
          x2_ultra_relaxed_walk_*.pkl
          playlists/
            relaxed_walk_*.yaml       (--include-playlists, default on)
          x2_recorded/
            demo_gestures/*.pkl
            mc_gestures/*.pkl
        scripts/
          play_xbox_controller.py     (--include-script, default on)
      xbox_controller_commands.md     (--include-cheatsheet, default on)

Run it before each demo to make sure the bundle matches the current
binding map::

    .venv/bin/python -m gear_sonic.scripts.build_demo_bundle

Then sync to the demo machine (recipe printed in README.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]

# Playlist YAMLs that *generate* the 4 locomotion PKLs (so the demo
# machine has a record of what the binaries came from).
LOCOMOTION_PLAYLIST_YAMLS: tuple[str, ...] = (
    "gear_sonic/data/motions/playlists/relaxed_walk_forward_v1.yaml",
    "gear_sonic/data/motions/playlists/relaxed_walk_one_left_turn_v1.yaml",
    "gear_sonic/data/motions/playlists/relaxed_walk_one_right_turn_v1.yaml",
    "gear_sonic/data/motions/playlists/relaxed_walk_two_right_turns_v1.yaml",
)

# Reference assets the operator may want alongside the data.
LAUNCHER_SCRIPT_REL = "gear_sonic/scripts/play_xbox_controller.py"
CHEATSHEET_REL = "xbox_controller_commands.md"


@dataclass(frozen=True)
class BundleEntry:
    """One file copied into the bundle."""

    category: str  # 'locomotion' | 'gesture' | 'estop_followup' | 'playlist' | 'script' | 'doc'
    chord: Optional[str]  # e.g. 'UP', 'Y+R2', or None for non-binding entries
    src_rel: str  # path relative to repo root
    sha256: str
    size_bytes: int


def _sha256_of(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _copy_into_bundle(
    *, src_abs: Path, src_rel: str, bundle_root: Path
) -> Path:
    """Copy ``src_abs`` into ``bundle_root`` at the same relative path."""
    dst = bundle_root / src_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_abs, dst)
    return dst


def _resolve_or_die(rel_path: str, *, label: str) -> Path:
    abs_path = (REPO_ROOT / rel_path).resolve()
    if not abs_path.is_file():
        raise SystemExit(
            f"[bundle] FATAL: {label} not found at {abs_path}"
        )
    if REPO_ROOT not in abs_path.parents and abs_path != REPO_ROOT:
        # Defensive: never let a binding point outside the repo root.
        raise SystemExit(
            f"[bundle] FATAL: {label} {abs_path} is outside repo root "
            f"{REPO_ROOT}"
        )
    return abs_path


def _iter_binding_entries(
    *,
    bindings: dict[str, Optional[str]],
    category: str,
    bundle_root: Path,
) -> Iterable[BundleEntry]:
    for chord, rel in bindings.items():
        if rel is None:
            continue
        abs_path = _resolve_or_die(rel, label=f"{category} {chord}")
        _copy_into_bundle(src_abs=abs_path, src_rel=rel, bundle_root=bundle_root)
        yield BundleEntry(
            category=category,
            chord=chord,
            src_rel=rel,
            sha256=_sha256_of(abs_path),
            size_bytes=abs_path.stat().st_size,
        )


def _iter_extra_files(
    *,
    paths: Iterable[str],
    category: str,
    bundle_root: Path,
) -> Iterable[BundleEntry]:
    for rel in paths:
        abs_path = (REPO_ROOT / rel).resolve()
        if not abs_path.is_file():
            print(
                f"[bundle] WARN  skipping missing {category}: {rel}",
                file=sys.stderr,
            )
            continue
        _copy_into_bundle(src_abs=abs_path, src_rel=rel, bundle_root=bundle_root)
        yield BundleEntry(
            category=category,
            chord=None,
            src_rel=rel,
            sha256=_sha256_of(abs_path),
            size_bytes=abs_path.stat().st_size,
        )


def _format_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    val = float(n)
    for u in units:
        if val < 1024.0:
            return f"{val:.1f} {u}" if u != "B" else f"{int(val)} {u}"
        val /= 1024.0
    return f"{val:.1f} TB"


def _estimate_duration_s(rel_path: str) -> Optional[float]:
    """Best-effort PKL duration estimate, returns None if unavailable.

    Uses the same helper the launcher uses for its busy-gate window
    (``estimate_duration_s`` with ``target_rate_hz=50.0`` -- matches
    ``play_xbox_controller.py``'s default).
    """
    try:
        from gear_sonic.utils.teleop.motion_clip_session import (  # noqa: WPS433
            MotionClipEntry,
            estimate_duration_s,
        )
    except Exception:  # pragma: no cover -- soft dependency
        return None
    try:
        entry = MotionClipEntry(name="duration_probe", source=Path(rel_path))
        return float(estimate_duration_s(entry, target_rate_hz=50.0))
    except Exception:  # noqa: BLE001 -- best-effort
        return None


_VERIFY_SCRIPT_BODY = '''#!/usr/bin/env python3
"""Verify Xbox-controller demo bundle integrity against manifest.json.

By default, hashes every file inside this bundle directory (so you
can confirm rsync didn't corrupt anything mid-transfer).

Pass ``--against <dir>`` to point at the deployed gear_sonic tree on
the demo machine instead -- typically the repo root the bundle was
rsynced into. Verifies the files in their *deployed* location match
the bundle's recorded sha-256.

Usage:
    cd <bundle_dir> && python3 verify.py
    python3 <bundle_dir>/verify.py --against /path/to/GR00T-WholeBodyControl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for buf in iter(lambda: fh.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).resolve().parent / "manifest.json",
        help="Path to manifest.json (default: alongside this script).",
    )
    parser.add_argument(
        "--against",
        type=Path,
        default=None,
        help="Verify deployed copies under this root (default: verify "
        "files inside the bundle itself).",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"FAIL: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    manifest = json.loads(args.manifest.read_text())
    root = (args.against or args.manifest.parent).resolve()
    print(f"verifying {len(manifest['entries'])} entries against {root}")

    bad: list[tuple[str, str]] = []
    for e in manifest["entries"]:
        p = root / e["src_rel"]
        if not p.is_file():
            bad.append((e["src_rel"], "MISSING"))
            continue
        size = p.stat().st_size
        if size != e["size_bytes"]:
            bad.append(
                (e["src_rel"],
                 f"size {size} != manifest {e['size_bytes']}")
            )
            continue
        got = _sha256_of(p)
        if got != e["sha256"]:
            bad.append(
                (e["src_rel"],
                 f"sha mismatch (got {got[:12]} want {e['sha256'][:12]})")
            )

    if not bad:
        print(f"OK  all {len(manifest['entries'])} files match.")
        return 0
    print(f"FAIL  {len(bad)} mismatched / missing files:")
    for rel, why in bad:
        print(f"  {rel}: {why}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
'''


def _build_readme(
    *,
    entries: list[BundleEntry],
    bindings_loco: dict[str, Optional[str]],
    bindings_gest: dict[str, Optional[str]],
    estop_followup: Optional[str],
    bundle_dir_name: str,
    built_at: str,
    total_bytes: int,
) -> str:
    """Render the operator-facing README for the bundle."""
    sha_by_rel = {e.src_rel: e.sha256 for e in entries}
    size_by_rel = {e.src_rel: e.size_bytes for e in entries}

    def _row(chord: str, rel: Optional[str]) -> str:
        if rel is None:
            return f"| `{chord}` | *(free)* | — | — |"
        size = size_by_rel.get(rel, 0)
        sha = sha_by_rel.get(rel, "")
        dur = _estimate_duration_s(rel)
        dur_s = f"{dur:.1f} s" if dur is not None else "—"
        return (
            f"| `{chord}` | `{rel}` | {_format_bytes(size)} | {dur_s} "
            f"| `{sha[:12]}…` |"
        )

    lines: list[str] = []
    lines.append(f"# Xbox demo bundle ({bundle_dir_name})")
    lines.append("")
    lines.append(f"Built at: `{built_at}`")
    lines.append(f"Total payload: **{_format_bytes(total_bytes)}** "
                 f"across **{len(entries)}** files.")
    lines.append("")
    lines.append(
        "Self-contained bundle of every PKL referenced by "
        "`gear_sonic/scripts/play_xbox_controller.py` (locomotion + "
        "gestures + e-stop ack). The directory tree mirrors the repo "
        "layout, so you can rsync this folder directly on top of the "
        "`gear_sonic/` checkout on the demo machine without any path "
        "rewriting."
    )
    lines.append("")
    lines.append("## Sync to the demo machine")
    lines.append("")
    lines.append("Copy the whole bundle alongside the repo (preserves the")
    lines.append("manifest + README) so you can re-verify on-site:")
    lines.append("")
    lines.append("```bash")
    lines.append("# on the build machine (this one):")
    lines.append(
        f"rsync -avz --progress {bundle_dir_name}/ "
        f"<user>@<demo-host>:/path/to/GR00T-WholeBodyControl/"
        f"{bundle_dir_name}/"
    )
    lines.append("")
    lines.append("# on the demo machine, overlay the data into gear_sonic/")
    lines.append("# (rsync without --delete so unrelated PKLs already on the")
    lines.append("# demo box aren't wiped):")
    lines.append(
        f"rsync -av {bundle_dir_name}/gear_sonic/ gear_sonic/"
    )
    lines.append("```")
    lines.append("")
    lines.append(
        "If the demo machine doesn't have the `gear_sonic` checkout "
        "at all, just clone the repo first and then run the second "
        "`rsync` above."
    )
    lines.append("")
    lines.append("## Verify the payload on the demo machine")
    lines.append("")
    lines.append(
        "Re-hashes every file in the bundle against `manifest.json`. "
        "Run it from inside the bundle:"
    )
    lines.append("")
    lines.append("```bash")
    lines.append(f"cd {bundle_dir_name} && python3 verify.py")
    lines.append("```")
    lines.append("")
    lines.append(
        "Pass `--against <dir>` to verify the deployed gear_sonic tree "
        "instead (after the second rsync above):"
    )
    lines.append("")
    lines.append("```bash")
    lines.append(
        f"python3 {bundle_dir_name}/verify.py "
        f"--against /path/to/GR00T-WholeBodyControl"
    )
    lines.append("```")
    lines.append("")
    lines.append("## Locomotion (D-pad + L2+R2 deadman, L1+R1 released)")
    lines.append("")
    lines.append(
        "| chord | PKL | size | duration | sha-256 |"
    )
    lines.append(
        "|-------|-----|-----:|---------:|---------|"
    )
    for chord in ("UP", "LEFT", "RIGHT", "DOWN"):
        lines.append(_row(f"D-pad {chord}", bindings_loco.get(chord)))
    lines.append("")
    lines.append(
        "## Gestures (A/B/X/Y bare or + single modifier L1|R1|L2|R2)"
    )
    lines.append("")
    lines.append("Any other shoulder/trigger combo silences face buttons "
                 "(see launcher docstring + cheatsheet).")
    lines.append("")
    lines.append(
        "| chord | PKL | size | duration | sha-256 |"
    )
    lines.append(
        "|-------|-----|-----:|---------:|---------|"
    )
    for chord, rel in bindings_gest.items():
        lines.append(_row(chord, rel))
    lines.append("")
    lines.append("## E-stop acknowledgment")
    lines.append("")
    lines.append(
        "The `L1+R1+L2+R2` chord publishes a stop and then plays this "
        "PKL as a visible acknowledgment gesture (set "
        "`ESTOP_FOLLOWUP_PKL = None` in the launcher to skip)."
    )
    lines.append("")
    lines.append(
        "| chord | PKL | size | duration | sha-256 |"
    )
    lines.append(
        "|-------|-----|-----:|---------:|---------|"
    )
    lines.append(_row("L1+R1+L2+R2 ack", estop_followup))
    lines.append("")
    lines.append("## Other bundled files")
    lines.append("")
    extras = [e for e in entries
              if e.category in ("playlist", "script", "doc")]
    if not extras:
        lines.append("*(none)*")
    else:
        lines.append("| category | path | size |")
        lines.append("|----------|------|-----:|")
        for e in extras:
            lines.append(
                f"| {e.category} | `{e.src_rel}` "
                f"| {_format_bytes(e.size_bytes)} |"
            )
    lines.append("")
    lines.append("## Regenerating the bundle")
    lines.append("")
    lines.append("Re-run the builder whenever the binding map changes:")
    lines.append("")
    lines.append("```bash")
    lines.append(".venv/bin/python -m gear_sonic.scripts.build_demo_bundle")
    lines.append("```")
    lines.append("")
    lines.append(
        "Pass `--output <dir>` to stage somewhere other than "
        f"`./{bundle_dir_name}/`, `--no-playlists` / `--no-script` / "
        "`--no-cheatsheet` to slim the bundle, or `--tar` to also "
        "emit a gzipped tarball next to the bundle directory."
    )
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a self-contained Xbox-controller demo bundle.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "demo_bundle",
        help="Output directory (default: ./demo_bundle at the repo root).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete the output directory before building (default: "
        "merge into it).",
    )
    parser.add_argument(
        "--no-playlists",
        dest="include_playlists",
        action="store_false",
        help="Skip the 4 relaxed_walk_*.yaml playlist files.",
    )
    parser.add_argument(
        "--no-script",
        dest="include_script",
        action="store_false",
        help="Skip the play_xbox_controller.py launcher copy.",
    )
    parser.add_argument(
        "--no-cheatsheet",
        dest="include_cheatsheet",
        action="store_false",
        help="Skip the xbox_controller_commands.md cheatsheet copy.",
    )
    parser.add_argument(
        "--tar",
        action="store_true",
        help="Also emit <output>.tar.gz alongside the directory.",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output.resolve()
    if args.clean and output_dir.exists():
        print(f"[bundle] --clean: rm -rf {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Defer the import so --help / arg parsing don't pay the pygame
    # import cost; play_xbox_controller imports pygame at module load.
    from gear_sonic.scripts.play_xbox_controller import (  # noqa: WPS433
        BINDINGS_GESTURES,
        BINDINGS_LOCOMOTION,
        ESTOP_FOLLOWUP_PKL,
    )

    entries: list[BundleEntry] = []
    entries.extend(_iter_binding_entries(
        bindings=BINDINGS_LOCOMOTION,
        category="locomotion",
        bundle_root=output_dir,
    ))
    entries.extend(_iter_binding_entries(
        bindings=BINDINGS_GESTURES,
        category="gesture",
        bundle_root=output_dir,
    ))
    if ESTOP_FOLLOWUP_PKL is not None:
        entries.extend(_iter_binding_entries(
            bindings={"L1+R1+L2+R2 ack": ESTOP_FOLLOWUP_PKL},
            category="estop_followup",
            bundle_root=output_dir,
        ))

    if args.include_playlists:
        entries.extend(_iter_extra_files(
            paths=LOCOMOTION_PLAYLIST_YAMLS,
            category="playlist",
            bundle_root=output_dir,
        ))
    if args.include_script:
        entries.extend(_iter_extra_files(
            paths=(LAUNCHER_SCRIPT_REL,),
            category="script",
            bundle_root=output_dir,
        ))
    if args.include_cheatsheet:
        entries.extend(_iter_extra_files(
            paths=(CHEATSHEET_REL,),
            category="doc",
            bundle_root=output_dir,
        ))

    total_bytes = sum(e.size_bytes for e in entries)
    built_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")

    manifest = {
        "built_at": built_at,
        "repo_root": str(REPO_ROOT),
        "bundle_root": str(output_dir),
        "entry_count": len(entries),
        "total_bytes": total_bytes,
        "estop_followup": ESTOP_FOLLOWUP_PKL,
        "bindings_locomotion": BINDINGS_LOCOMOTION,
        "bindings_gestures": BINDINGS_GESTURES,
        "entries": [
            {
                "category": e.category,
                "chord": e.chord,
                "src_rel": e.src_rel,
                "size_bytes": e.size_bytes,
                "sha256": e.sha256,
            }
            for e in entries
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=False) + "\n"
    )

    readme = _build_readme(
        entries=entries,
        bindings_loco=BINDINGS_LOCOMOTION,
        bindings_gest=BINDINGS_GESTURES,
        estop_followup=ESTOP_FOLLOWUP_PKL,
        bundle_dir_name=output_dir.name,
        built_at=built_at,
        total_bytes=total_bytes,
    )
    (output_dir / "README.md").write_text(readme)

    verify_path = output_dir / "verify.py"
    verify_path.write_text(_VERIFY_SCRIPT_BODY)
    verify_path.chmod(0o755)

    print(f"[bundle] wrote {len(entries)} files "
          f"({_format_bytes(total_bytes)}) to {output_dir}")
    for e in entries:
        print(f"  {e.category:>14}  "
              f"{(e.chord or '-'):<16}  "
              f"{_format_bytes(e.size_bytes):>9}  "
              f"{e.src_rel}")
    print(f"[bundle] manifest: {manifest_path}")
    print(f"[bundle] readme:   {output_dir / 'README.md'}")
    print(f"[bundle] verify:   {verify_path}")

    if args.tar:
        tar_path = output_dir.with_suffix(".tar.gz")
        # Build relative archive so unpacking on the demo side puts the
        # bundle next to the existing tree, not inside an absolute path.
        archive_base = str(tar_path).rsplit(".tar.gz", 1)[0]
        archive_path = shutil.make_archive(
            base_name=archive_base,
            format="gztar",
            root_dir=str(output_dir.parent),
            base_dir=output_dir.name,
        )
        size = Path(archive_path).stat().st_size
        print(f"[bundle] tarball: {archive_path} ({_format_bytes(size)})")

    return 0


if __name__ == "__main__":  # pragma: no cover -- CLI entrypoint
    sys.exit(main())
