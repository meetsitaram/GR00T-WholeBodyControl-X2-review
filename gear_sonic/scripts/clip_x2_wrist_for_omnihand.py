"""Clip the X2 ``*_wrist_roll_link.STL`` mesh so the OmniHand palm mates cleanly.

The X2 Ultra URDF ships with ``meshes/{left,right}_wrist_roll_link.STL`` that
bakes in *both* the wrist-roll motor housing **and** a static dummy "fist"
stub at its tip (intended for renders that have no real hand attached). When
we attach the articulated OmniHand-2025 chain at the wrist, the dummy fist
sticks straight through the OmniHand palm and the result looks like two
hands stacked at the same wrist.

This vendor-step utility clips the wrist mesh at the natural neck where the
cylindrical motor housing transitions into the dummy fist (roughly z ≈
-0.055 m in the body local frame, where the cross-section radius narrows to
~0.029 m -- almost exactly the OmniHand palm cuff radius of 0.028 m). The
clipped mesh keeps everything *above* the cut and is written to
``gear_sonic/data/assets/robot_description/omnihand/meshes/
{side}_wrist_roll_clipped_link.STL``.

The clipping is exact: triangles fully above the cut are kept verbatim;
triangles straddling the cut plane are split along the intersection edges so
the resulting mesh has a clean cap-edge at the cut Z. We do not synthesise
an end-cap fan because the OmniHand palm cuff covers the remaining hole
seamlessly when ``compose_x2_with_omnihand`` mounts the palm at the same
cut Z.

Run as part of the vendoring step (idempotent; re-running overwrites)::

    .venv/bin/python gear_sonic/scripts/clip_x2_wrist_for_omnihand.py

The composer (``compose_x2_with_omnihand.py``) registers the clipped meshes
as a separate asset and swaps the wrist_roll *visual* geom to use them; the
collision geom keeps the original (full) mesh so the contact solver behaves
identically to the un-augmented X2 model.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# Default cut Z: where the wrist roll motor casing ends and the dummy fist
# starts. Identified by sweeping the cross-section radius along Z (see the
# docstring for the profile). The same value works for both sides since the
# meshes are mirrored.
DEFAULT_CUT_Z: float = -0.055


@dataclass(frozen=True)
class Triangle:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    normal: np.ndarray


def load_stl(path: Path) -> list[Triangle]:
    """Load a binary STL into a list of triangles (no header / topology)."""
    with open(path, "rb") as f:
        f.read(80)
        n = struct.unpack("<I", f.read(4))[0]
        out: list[Triangle] = []
        for _ in range(n):
            normal = np.array(struct.unpack("<fff", f.read(12)), dtype=np.float64)
            verts = np.empty((3, 3), dtype=np.float64)
            for k in range(3):
                verts[k] = struct.unpack("<fff", f.read(12))
            f.read(2)  # uint16 attribute byte count -- always 0 for binary STL
            out.append(Triangle(verts[0], verts[1], verts[2], normal))
    return out


def write_stl(path: Path, triangles: Iterable[Triangle]) -> int:
    """Write a binary STL. Returns triangle count actually written."""
    triangles = list(triangles)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        # 80-byte header (free-form text). Mark provenance so future you can
        # tell at a glance what this mesh is.
        header = b"X2 wrist_roll clipped for OmniHand mount"
        f.write(header.ljust(80, b" "))
        f.write(struct.pack("<I", len(triangles)))
        for tri in triangles:
            f.write(struct.pack("<fff", *tri.normal.astype(np.float32)))
            for v in (tri.a, tri.b, tri.c):
                f.write(struct.pack("<fff", *v.astype(np.float32)))
            f.write(struct.pack("<H", 0))
    return len(triangles)


def _interp_on_plane(p_above: np.ndarray, p_below: np.ndarray, cut_z: float) -> np.ndarray:
    """Linear interpolation along the edge ``p_above -> p_below`` to ``z = cut_z``."""
    t = (p_above[2] - cut_z) / (p_above[2] - p_below[2])
    return p_above + t * (p_below - p_above)


def clip_above(tris: list[Triangle], cut_z: float) -> list[Triangle]:
    """Return triangles representing the mesh portion with ``z >= cut_z``.

    Triangles fully above the plane are kept verbatim. Triangles fully below
    are dropped. Triangles straddling the plane are split along the
    intersection segment, producing 1 or 2 new triangles entirely on the
    above side. The original triangle's outward normal is reused for every
    sub-triangle (cheap approximation; a full retriangulation would require
    recomputing each normal but for our visual-only use case the original
    normal is close enough since the cut is near-planar).
    """
    out: list[Triangle] = []
    for tri in tris:
        verts = (tri.a, tri.b, tri.c)
        zs = np.array([v[2] for v in verts])
        above = zs >= cut_z
        n_above = int(above.sum())

        if n_above == 3:
            out.append(tri)
        elif n_above == 0:
            continue
        elif n_above == 2:
            # Split into a quad (two above-vertices + two interpolated points
            # on the cut plane) -> two triangles. Identify the single
            # below-vertex.
            below_idx = int(np.argmin(above.astype(int)))
            above_idxs = [i for i in range(3) if i != below_idx]
            v_below = verts[below_idx]
            v_a, v_b = verts[above_idxs[0]], verts[above_idxs[1]]
            p1 = _interp_on_plane(v_a, v_below, cut_z)
            p2 = _interp_on_plane(v_b, v_below, cut_z)
            # Preserve original winding: the original ordering is
            # (v_below, v_a, v_b) cycled to start at below_idx; we restore
            # the original orientation by inserting p1, p2 in place of v_below.
            order = [(below_idx, v_below), (above_idxs[0], v_a), (above_idxs[1], v_b)]
            order.sort(key=lambda x: x[0])
            ordered = [pair[1] for pair in order]
            sub_a = ordered[0]  # original index 0
            sub_b = ordered[1]  # original index 1
            sub_c = ordered[2]  # original index 2
            # Replace the below-vertex with its two interpolated counterparts.
            replacement = {below_idx: (p1, p2)}
            # Build polygon by walking original vertex order; insert (p1, p2)
            # in place of below-vertex.
            poly: list[np.ndarray] = []
            for i in range(3):
                if i == below_idx:
                    # Keep traversal direction consistent with edges:
                    # edge (above_a -> below) -> p_a, edge (below -> above_b) -> p_b.
                    poly.append(p1)
                    poly.append(p2)
                else:
                    poly.append(verts[i])
            # poly has 4 points; fan-triangulate.
            out.append(Triangle(poly[0], poly[1], poly[2], tri.normal))
            out.append(Triangle(poly[0], poly[2], poly[3], tri.normal))
        elif n_above == 1:
            # Single above-vertex -> one triangle.
            above_idx = int(np.argmax(above.astype(int)))
            v_above = verts[above_idx]
            below_idxs = [i for i in range(3) if i != above_idx]
            v_b1 = verts[below_idxs[0]]
            v_b2 = verts[below_idxs[1]]
            p1 = _interp_on_plane(v_above, v_b1, cut_z)
            p2 = _interp_on_plane(v_above, v_b2, cut_z)
            poly: list[np.ndarray] = []
            for i in range(3):
                if i == above_idx:
                    poly.append(v_above)
                elif i == below_idxs[0]:
                    poly.append(p1)
                else:
                    poly.append(p2)
            out.append(Triangle(poly[0], poly[1], poly[2], tri.normal))
    return out


def clip_one(src: Path, dst: Path, *, cut_z: float) -> tuple[int, int]:
    tris = load_stl(src)
    n_in = len(tris)
    clipped = clip_above(tris, cut_z)
    n_out = write_stl(dst, clipped)
    return n_in, n_out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cut-z", type=float, default=DEFAULT_CUT_Z,
        help=(
            "Z threshold (meters, wrist_roll body local frame) above which "
            "the mesh is kept. Default %(default)s -- the natural neck."
        ),
    )
    parser.add_argument(
        "--src-dir", type=Path,
        default=Path(__file__).resolve().parents[2]
        / "gear_sonic" / "data" / "assets" / "robot_description"
        / "urdf" / "x2_ultra" / "meshes",
        help="Source directory containing {side}_wrist_roll_link.STL files.",
    )
    parser.add_argument(
        "--dst-dir", type=Path,
        default=Path(__file__).resolve().parents[2]
        / "gear_sonic" / "data" / "assets" / "robot_description"
        / "omnihand" / "meshes",
        help="Output directory; files written as {side}_wrist_roll_clipped_link.STL.",
    )
    args = parser.parse_args(argv)

    args.dst_dir.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        src = args.src_dir / f"{side}_wrist_roll_link.STL"
        dst = args.dst_dir / f"{side}_wrist_roll_clipped_link.STL"
        if not src.is_file():
            print(f"  ERROR: missing {src}", file=sys.stderr)
            return 2
        n_in, n_out = clip_one(src, dst, cut_z=args.cut_z)
        print(
            f"  {side}: {n_in} -> {n_out} triangles  "
            f"(cut_z={args.cut_z:+.4f}m)  -> {dst.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_CUT_Z", "Triangle", "clip_above", "clip_one", "load_stl", "write_stl"]
