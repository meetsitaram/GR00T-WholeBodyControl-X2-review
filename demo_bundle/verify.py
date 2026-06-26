#!/usr/bin/env python3
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
