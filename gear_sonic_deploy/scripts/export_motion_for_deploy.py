#!/usr/bin/env python3
"""Convert a motion-lib source (PKL or warehouse playlist YAML) to the X2M2
binary format consumed by the C++ deploy package's ``PklMotionReference``
loader.

Format spec (from
``gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/reference_motion.hpp``)::

    uint32  magic       == 0x58324D32  ("X2M2")
    uint32  num_frames
    uint32  num_dofs    (must equal 31)
    double  fps
    For each frame f in [0, num_frames):
        double  joint_pos_mj[31]
        double  root_quat_xyzw[4]

Joint velocity is reconstructed at runtime via finite difference, so we
intentionally do NOT serialize qvel here.

The input is a motion-lib dict-of-motions of shape::

    {<motion_name>: {"dof": (T, 31), "root_rot": (T, 4) xyzw,
                     "fps": float, "root_trans_offset": (T, 3), ...}}

We accept it either as:
- ``.pkl`` produced by the training motion-lib pipeline, or
- ``.yaml`` warehouse playlist resolved through
  ``gear_sonic.scripts._warehouse_playlist.build_concat`` (same in-memory
  dict ``eval_x2_mujoco_onnx.py --playlist`` consumes for sim-to-sim eval).

The exporter takes the first (and typically only) entry.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

X2M2_MAGIC = 0x58324D32  # "X2M2" little-endian
NUM_DOFS = 31

# Source extensions we accept for ``bake_x2m2``. Anything else raises so the
# caller fails loudly instead of silently feeding the deploy node garbage.
SUPPORTED_SOURCE_SUFFIXES = (".pkl", ".yaml", ".yml")


def _load_motion_dict(source_path: Path) -> dict:
    """Resolve a PKL or YAML playlist into a motion-lib dict.

    Centralises the load so callers (this CLI + the wrapper's bake-on-the-fly
    path) can't accidentally diverge. We deliberately delegate to the same
    helpers ``eval_x2_mujoco_onnx.py`` uses for Python-driven sim-to-sim eval
    so the bridge's RSI init, the Python parity sweep, and the C++ deploy node
    all consume bit-identical motion data.
    """
    suffix = source_path.suffix.lower()
    if suffix == ".pkl":
        import joblib  # local import: only PKL path needs joblib at runtime
        return joblib.load(source_path)
    if suffix in (".yaml", ".yml"):
        # Reuse the same builder ``eval_x2_mujoco.load_playlist_motion_data``
        # uses. We import it lazily so the CLI stays cheap on --help and so
        # gear_sonic doesn't have to be importable for the PKL fast-path.
        repo_root = Path(__file__).resolve().parent.parent.parent
        scripts_dir = repo_root / "gear_sonic" / "scripts"
        if not scripts_dir.is_dir():
            raise RuntimeError(
                f"Cannot resolve YAML playlist {source_path}: "
                f"gear_sonic/scripts not found at {scripts_dir}. "
                f"Pass a baked PKL instead, or run from the repo root."
            )
        sys.path.insert(0, str(scripts_dir))
        from _warehouse_playlist import build_concat, load_playlist  # type: ignore
        pl = load_playlist(source_path)
        motion = build_concat(pl)
        return {pl.output_key: motion}
    raise ValueError(
        f"Unsupported motion source extension {suffix!r} for {source_path}; "
        f"expected one of {SUPPORTED_SOURCE_SUFFIXES}"
    )


def _take_first_motion(pkl_data: dict) -> tuple[str, dict]:
    if not isinstance(pkl_data, dict):
        raise ValueError(
            f"Expected a dict-of-motions, got {type(pkl_data).__name__}"
        )
    if not pkl_data:
        raise ValueError("Motion source contained an empty dict")
    name = next(iter(pkl_data))
    return name, pkl_data[name]


def _validate(motion: dict) -> None:
    for key in ("dof", "root_rot", "fps"):
        if key not in motion:
            raise KeyError(
                f"Motion missing required key {key!r}; "
                f"present keys: {sorted(motion.keys())}"
            )
    dof = np.asarray(motion["dof"])
    rot = np.asarray(motion["root_rot"])
    if dof.ndim != 2 or dof.shape[1] != NUM_DOFS:
        raise ValueError(
            f"motion['dof'] must be (T, {NUM_DOFS}); got {dof.shape}"
        )
    if rot.ndim != 2 or rot.shape[1] != 4:
        raise ValueError(
            f"motion['root_rot'] must be (T, 4) xyzw; got {rot.shape}"
        )
    if dof.shape[0] != rot.shape[0]:
        raise ValueError(
            f"frame count mismatch: dof has {dof.shape[0]}, "
            f"root_rot has {rot.shape[0]}"
        )


def bake_x2m2(in_path: Path, out_path: Path, verbose: bool = True) -> None:
    """Bake a PKL or YAML playlist source to an X2M2 binary at ``out_path``.

    Single source-of-truth entry point: the deploy_x2.sh wrapper calls this
    on every sim/onbot launch so the X2M2 the deploy binary loads is *always*
    derived from the same motion the bridge's RSI init reads, eliminating
    the silent-drift class of bug where motions_x2m2/<x>.x2m2 was baked from
    a stale PKL. Callers should pass a tempdir output for one-shot bakes.
    """
    if verbose:
        print(f"[export_motion] loading {in_path} ...", flush=True)
    data = _load_motion_dict(in_path)
    name, motion = _take_first_motion(data)
    _validate(motion)

    dof = np.asarray(motion["dof"], dtype=np.float64)
    rot = np.asarray(motion["root_rot"], dtype=np.float64)
    fps = float(motion["fps"])
    n_frames = int(dof.shape[0])

    # Sanity-check the quaternion is unit length (within tolerance) on the
    # first / last frame so we catch obvious xyzw-vs-wxyz swaps early.
    for label, q in (("first", rot[0]), ("last", rot[-1])):
        norm = float(np.linalg.norm(q))
        if not (0.95 <= norm <= 1.05):
            raise ValueError(
                f"{label} root_rot quaternion has |q|={norm:.4f}, expected ~1.0; "
                f"is the PKL really xyzw scipy convention?"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(struct.pack("<III", X2M2_MAGIC, n_frames, NUM_DOFS))
        f.write(struct.pack("<d", fps))
        for i in range(n_frames):
            f.write(dof[i].astype(np.float64).tobytes(order="C"))
            f.write(rot[i].astype(np.float64).tobytes(order="C"))

    if verbose:
        size = out_path.stat().st_size
        expected = 4 * 3 + 8 + n_frames * (8 * NUM_DOFS + 8 * 4)
        print(
            f"[export_motion] wrote {out_path}\n"
            f"                motion_name = {name!r}\n"
            f"                num_frames  = {n_frames}\n"
            f"                fps         = {fps:.3f}\n"
            f"                duration    = {n_frames / fps:.2f} s\n"
            f"                bytes       = {size} (expected {expected})",
            flush=True,
        )
        if size != expected:
            print(
                "[export_motion] WARNING: byte count mismatch — "
                "check struct alignment",
                file=sys.stderr,
            )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", required=True, type=Path,
                   help="Input motion source: .pkl (motion-lib) or .yaml "
                        "(warehouse playlist).")
    p.add_argument("--out", dest="out_path", required=True, type=Path,
                   help="Output .x2m2 path.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress informational output.")
    args = p.parse_args(argv)

    bake_x2m2(args.in_path, args.out_path, verbose=not args.quiet)
    return 0


# Back-compat alias: older call sites used ``export(...)``. Keep the old name
# pointing at the new entry point so we don't break any in-repo imports.
export = bake_x2m2


if __name__ == "__main__":
    raise SystemExit(main())
