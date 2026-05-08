"""Warehouse playlist loader / stitcher (shared between CLI + runtime eval).

A *playlist* is a YAML that points at N motion-lib PKL clips ("segments") and
asks the loader to chain them into a single continuous reference, with a
"rest layer" of held standing pose interleaved between every pair. This module
turns the YAML into a single ``{dof, root_rot, root_trans_offset, fps}`` dict
that drops cleanly into anything that already consumes a single-clip
motion-lib PKL (``play_motion_mujoco.py``, ``eval_x2_mujoco.py``,
``eval_x2_mujoco_onnx.py``, ``record_x2_eval_mujoco.py``).

Why a separate module: the CLI stitcher (``make_warehouse_motion.py``) and
the eval ``--playlist`` runtime path both call into the same
:func:`build_concat`, so the on-disk ``x2_ultra_warehouse_v1.pkl`` and the
runtime in-memory dict are byte-identical. That makes the runtime
``--playlist`` mode a free regression test on the stitcher (and vice versa).

What this is NOT:
    - A motion-lib loader. The output keeps only the four fields MuJoCo eval
      reads. ``pose_aa`` and ``smpl_joints`` from ``bones_seed.pkl`` are
      explicitly DROPPED, because (a) eval doesn't read them, (b) the
      yaw-cylinder chaining would silently desync ``pose_aa[:, 0]`` (the
      SMPL global root rotation) from ``root_rot``, and (c) the rest source
      ``x2_ultra_rest_loop_idle.pkl`` already has mismatched dof/pose_aa
      shapes, so the convention of dropping SMPL fields on non-training PKLs
      already exists in the repo.
    - A trainable artifact. Use the per-clip PKLs in
      ``gear_sonic/data/motions/`` for training; this is for evaluation /
      scripted demos only.

Algorithm summary (one pass over the playlist in playlist order):

    1.  Load each segment's slice (``dof``, ``root_rot``, ``root_trans_offset``)
        and the single-frame rest pose.
    2.  For each segment apply yaw-only rigid alignment so its frame 0
        starts at the running world cursor (``xy_world``, ``yaw_world``).
        Roll/pitch are preserved untouched (this is the "yaw-cylinder
        subgroup of SE(3)" wording from the plan).
    3.  Between segments, synthesize a rest layer of ``rest_frames``:
            - frames 0..blend_in:   SLERP joints + root_rot from
              prev_last -> rest_pose, root_trans pinned at prev_last XY.
            - middle:               hold rest_pose, root_trans pinned.
            - last blend_out:       SLERP rest_pose -> next_first (which
              has already been re-aligned so its frame 0 = pinned XY).
    4.  Concatenate everything in float32.

See ``[gear_sonic/scripts/play_motion_mujoco.py:55](play_motion_mujoco.py)``
for the consumer-side schema validation this output respects.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp


NUM_DOFS = 31


# --------------------------------------------------------------------- types


@dataclass
class SegmentSpec:
    name: str
    source: Path
    motion_key: str | None
    start_frame: int
    n_frames: int


@dataclass
class RestSpec:
    source: Path
    motion_key: str | None
    frame: int
    rest_frames: int
    blend_in_frames: int
    blend_out_frames: int


@dataclass
class Playlist:
    name: str
    output_key: str
    fps: float
    zero_xy: bool
    zero_yaw: bool
    segments: list[SegmentSpec]
    rest: RestSpec
    # Filled in by build_concat(); useful for diagnostics and the
    # runtime --playlist accessor (which segment owns frame i?).
    segment_frame_ranges: list[tuple[str, int, int]] = field(default_factory=list)
    rest_frame_ranges: list[tuple[int, int]] = field(default_factory=list)


# --------------------------------------------------------------------- yaml


def load_playlist(yaml_path: Path) -> Playlist:
    """Parse the warehouse playlist YAML into a :class:`Playlist`."""
    yaml_path = Path(yaml_path)
    with yaml_path.open("r") as f:
        raw = yaml.safe_load(f)

    base = yaml_path.parent.parent  # playlists/ lives under data/motions/
    repo = Path(__file__).resolve().parents[2]

    def _resolve(p: str) -> Path:
        pp = Path(p)
        if pp.is_absolute() and pp.exists():
            return pp
        for candidate in (Path.cwd() / pp, repo / pp, base / pp):
            if candidate.exists():
                return candidate
        # Last resort: return the cwd-rooted path so the joblib.load error
        # message is still informative.
        return Path.cwd() / pp

    rest_raw = raw["rest"]
    rest = RestSpec(
        source=_resolve(rest_raw["source"]),
        motion_key=rest_raw.get("motion_key"),
        frame=int(rest_raw.get("frame", 0)),
        rest_frames=int(rest_raw.get("rest_frames", 30)),
        blend_in_frames=int(rest_raw.get("blend_in_frames", 6)),
        blend_out_frames=int(rest_raw.get("blend_out_frames", 6)),
    )
    if rest.blend_in_frames + rest.blend_out_frames > rest.rest_frames:
        raise ValueError(
            f"rest.blend_in ({rest.blend_in_frames}) + blend_out "
            f"({rest.blend_out_frames}) exceeds rest_frames "
            f"({rest.rest_frames}); shorten the blends or grow rest_frames."
        )

    segments = [
        SegmentSpec(
            name=s["name"],
            source=_resolve(s["source"]),
            motion_key=s.get("motion_key"),
            start_frame=int(s.get("start_frame", 0)),
            n_frames=int(s["n_frames"]),
        )
        for s in raw["segments"]
    ]
    if not segments:
        raise ValueError(f"{yaml_path}: playlist has zero segments")

    return Playlist(
        name=str(raw.get("name", yaml_path.stem)),
        output_key=str(raw.get("output_key", yaml_path.stem)),
        fps=float(raw.get("fps", 30.0)),
        zero_xy=bool(raw.get("zero_xy", True)),
        zero_yaw=bool(raw.get("zero_yaw", True)),
        segments=segments,
        rest=rest,
    )


# ---------------------------------------------------------- motion-lib I/O


def _load_motion_dict(path: Path, key: str | None) -> tuple[str, dict]:
    """Load a motion-lib PKL and return (resolved_key, motion_dict)."""
    data = joblib.load(path)
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{path}: not a non-empty dict-of-motions")
    name = key if key is not None else next(iter(data))
    if name not in data:
        sample = list(data)[:5]
        raise KeyError(
            f"{path}: motion key {name!r} not found "
            f"(available e.g. {sample}, total={len(data)})"
        )
    return name, data[name]


def _slice_segment(seg: SegmentSpec) -> dict[str, np.ndarray]:
    """Return float64 (dof, root_rot xyzw, root_trans) sliced from seg."""
    name, m = _load_motion_dict(seg.source, seg.motion_key)
    dof = np.asarray(m["dof"], dtype=np.float64)
    rot = np.asarray(m["root_rot"], dtype=np.float64)
    pos = np.asarray(m["root_trans_offset"], dtype=np.float64)
    if dof.shape[1] != NUM_DOFS:
        raise ValueError(
            f"{seg.name}: dof has {dof.shape[1]} cols, expected {NUM_DOFS} "
            f"(source={seg.source}, key={name})"
        )
    T = dof.shape[0]
    s = int(seg.start_frame)
    e = s + int(seg.n_frames)
    if s < 0 or e > T:
        raise ValueError(
            f"{seg.name}: requested frames [{s}:{e}] but source has {T} "
            f"(source={seg.source}, key={name})"
        )
    return {
        "dof": dof[s:e].copy(),
        "root_rot": rot[s:e].copy(),  # xyzw
        "root_trans": pos[s:e].copy(),
        "_resolved_key": name,
    }


def _load_rest_pose(rest: RestSpec) -> dict[str, np.ndarray]:
    """Load a single-frame "held" pose for the rest layer."""
    name, m = _load_motion_dict(rest.source, rest.motion_key)
    dof = np.asarray(m["dof"], dtype=np.float64)
    rot = np.asarray(m["root_rot"], dtype=np.float64)
    pos = np.asarray(m["root_trans_offset"], dtype=np.float64)
    f = int(rest.frame) % dof.shape[0]
    return {
        "dof": dof[f].copy(),
        "root_rot": rot[f].copy(),
        "root_trans": pos[f].copy(),
        "_resolved_key": name,
    }


# ----------------------------------------------------- yaw-cylinder helpers


def _yaw_of(quat_xyzw: np.ndarray) -> float:
    """Yaw (rad) about world Z. Uses zyx Euler so yaw is the first element.

    Returned yaw is in (-pi, pi]; absolute scale matches
    ``Rotation.from_quat([...]).as_euler("zyx")[0]`` to machine precision."""
    return float(Rot.from_quat(quat_xyzw).as_euler("zyx")[0])


def _rotate_yaw_only(
    quat_xyzw: np.ndarray, dyaw: float
) -> np.ndarray:
    """Left-multiply ``quat_xyzw`` by Rz(dyaw). Preserves roll/pitch by
    construction (Rz commutes through z but the body's own roll/pitch live
    in the residual rotation).

    Output shape = input shape (supports both (4,) and (T, 4))."""
    rz = Rot.from_euler("z", dyaw)
    return (rz * Rot.from_quat(quat_xyzw)).as_quat()


def _align_segment_yaw_only(
    seg: dict[str, np.ndarray],
    xy_world: np.ndarray,
    yaw_world: float,
) -> dict[str, np.ndarray]:
    """Rigid yaw-only re-alignment so seg.frame_0 starts at (xy_world, yaw_world).

    XY: rotate the segment's XY trajectory (relative to its own frame 0) by
        ``dyaw``, then translate to ``xy_world``.
    Z:  untouched (preserves natural pelvis bob).
    Rotation: left-multiply every frame's quaternion by Rz(dyaw). Roll/pitch
        are preserved verbatim because Rz only affects the world-yaw component.

    Mutates ``seg`` in-place and returns it.
    """
    yaw0 = _yaw_of(seg["root_rot"][0])
    dyaw = yaw_world - yaw0
    rz = Rot.from_euler("z", dyaw)

    # Rotate root_trans relative to frame 0, then translate.
    pos = seg["root_trans"]
    rel = pos - pos[0]
    rel_rot = rz.apply(rel)
    pos[:, 0] = rel_rot[:, 0] + xy_world[0]
    pos[:, 1] = rel_rot[:, 1] + xy_world[1]
    # Z is rel_rot[:, 2] + pos[0, 2]; but rz only rotates about Z, so
    # rel_rot[:, 2] == rel[:, 2], i.e. pos[:, 2] is unchanged. We assign it
    # explicitly for clarity / to defend against future Rz changes.
    pos[:, 2] = rel[:, 2] + pos[0, 2]

    seg["root_rot"] = (rz * Rot.from_quat(seg["root_rot"])).as_quat()
    seg["root_trans"] = pos
    return seg


def _slerp_quats(
    q_start_xyzw: np.ndarray, q_end_xyzw: np.ndarray, n: int
) -> np.ndarray:
    """SLERP from q_start to q_end over ``n`` frames (inclusive of start &
    end). Returns (n, 4) xyzw. n must be >= 2."""
    if n < 2:
        raise ValueError(f"_slerp_quats needs n>=2, got {n}")
    times = np.array([0.0, 1.0])
    rots = Rot.from_quat(np.stack([q_start_xyzw, q_end_xyzw]))
    slerp = Slerp(times, rots)
    samples = np.linspace(0.0, 1.0, n)
    return slerp(samples).as_quat()


def _lerp_dof(
    dof_start: np.ndarray, dof_end: np.ndarray, n: int
) -> np.ndarray:
    """Linear interp from dof_start to dof_end over n inclusive frames."""
    if n < 2:
        raise ValueError(f"_lerp_dof needs n>=2, got {n}")
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return (1.0 - t) * dof_start[None, :] + t * dof_end[None, :]


# --------------------------------------------------------- build pipeline


def _build_rest_layer(
    rest: RestSpec,
    rest_pose: dict[str, np.ndarray],
    prev_last: dict[str, np.ndarray],   # keys: dof, root_rot (xyzw), root_trans
    next_first: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Synthesize a rest-layer slice between two segments.

    Schema:
        frames 0 .. blend_in-1                : prev_last -> rest_pose SLERP
        frames blend_in .. T-blend_out-1      : hold rest_pose
        frames T-blend_out .. T-1             : rest_pose -> next_first SLERP

    All XY translation is pinned at ``prev_last['root_trans'][:2]`` (the
    handover XY); Z is interpolated/held at ``rest_pose['root_trans'][2]``
    so the held pose's pelvis-z is consistent. ``next_first`` should already
    have its frame 0 yaw-cylinder-aligned to the same handover XY/yaw, so
    the final blend-out lands smoothly.
    """
    T = rest.rest_frames
    bi = rest.blend_in_frames
    bo = rest.blend_out_frames
    if T <= 0:
        # Caller may set rest_frames=0 to disable rest layer entirely (just a
        # hard cut). We still return an empty array of the right shape.
        return {
            "dof": np.zeros((0, NUM_DOFS), dtype=np.float64),
            "root_rot": np.zeros((0, 4), dtype=np.float64),
            "root_trans": np.zeros((0, 3), dtype=np.float64),
        }

    out_dof = np.empty((T, NUM_DOFS), dtype=np.float64)
    out_rot = np.empty((T, 4), dtype=np.float64)
    out_pos = np.empty((T, 3), dtype=np.float64)

    pinned_xy = prev_last["root_trans"][:2].copy()
    rest_z = float(rest_pose["root_trans"][2])

    # Build the held rest pose as a yaw-aligned variant of the source rest
    # pose so its yaw matches the handover heading (otherwise SLERP would do
    # a global yaw flip mid-rest).
    handover_yaw = _yaw_of(prev_last["root_rot"])
    rest_yaw = _yaw_of(rest_pose["root_rot"])
    rest_root_rot_aligned = _rotate_yaw_only(
        rest_pose["root_rot"], handover_yaw - rest_yaw
    )

    # blend_in: prev_last.dof -> rest.dof, prev_last.root_rot -> rest_aligned.
    if bi >= 2:
        out_dof[:bi] = _lerp_dof(prev_last["dof"], rest_pose["dof"], bi)
        out_rot[:bi] = _slerp_quats(
            prev_last["root_rot"], rest_root_rot_aligned, bi
        )
    elif bi == 1:
        out_dof[:1] = prev_last["dof"]
        out_rot[:1] = prev_last["root_rot"]
    # bi == 0: skip

    # hold rest_pose between [bi, T-bo)
    hold_lo, hold_hi = bi, T - bo
    if hold_hi > hold_lo:
        out_dof[hold_lo:hold_hi] = rest_pose["dof"]
        out_rot[hold_lo:hold_hi] = rest_root_rot_aligned

    # blend_out: rest_aligned -> next_first
    if bo >= 2:
        out_dof[T - bo:] = _lerp_dof(
            rest_pose["dof"], next_first["dof"], bo
        )
        # next_first yaw is already aligned to handover_yaw by the caller
        # (yaw-cylinder chaining), so this SLERP stays in-plane.
        out_rot[T - bo:] = _slerp_quats(
            rest_root_rot_aligned, next_first["root_rot"], bo
        )
    elif bo == 1:
        out_dof[T - 1:T] = next_first["dof"]
        out_rot[T - 1:T] = next_first["root_rot"]

    out_pos[:, 0] = pinned_xy[0]
    out_pos[:, 1] = pinned_xy[1]
    out_pos[:, 2] = rest_z

    return {"dof": out_dof, "root_rot": out_rot, "root_trans": out_pos}


def build_concat(playlist: Playlist) -> dict[str, np.ndarray | float]:
    """Resolve, chain, rest-interleave, concat. Return motion-lib-shaped dict.

    Output schema (consumed by ``play_motion_mujoco`` /
    ``eval_x2_mujoco`` / ``eval_x2_mujoco_onnx``):

        dof               : (T, 31) float32
        root_rot          : (T, 4)  float32   xyzw
        root_trans_offset : (T, 3)  float32
        fps               : float

    Mutates ``playlist.segment_frame_ranges`` and
    ``playlist.rest_frame_ranges`` so the caller can print diagnostics or do
    runtime "which segment owns frame i" lookups.
    """
    segs = [_slice_segment(s) for s in playlist.segments]
    rest_pose = _load_rest_pose(playlist.rest)

    # Yaw-cylinder chain. Walk segments in playlist order, keeping a rolling
    # (xy_world, yaw_world) cursor that points at where the next segment's
    # frame 0 should land.
    if playlist.zero_xy:
        xy_world = np.zeros(2, dtype=np.float64)
    else:
        xy_world = segs[0]["root_trans"][0, :2].copy()
    if playlist.zero_yaw:
        yaw_world = 0.0
    else:
        yaw_world = _yaw_of(segs[0]["root_rot"][0])

    aligned: list[dict[str, np.ndarray]] = []
    for seg in segs:
        _align_segment_yaw_only(seg, xy_world, yaw_world)
        aligned.append(seg)
        # Advance world cursor to this segment's last frame.
        xy_world = seg["root_trans"][-1, :2].copy()
        yaw_world = _yaw_of(seg["root_rot"][-1])

    # Now interleave: [seg0, rest, seg1, rest, ..., segN-1].
    # Every rest layer is built AFTER both adjacent segments are
    # yaw-aligned, so prev_last and next_first are already in the same
    # world frame.
    chunks_dof: list[np.ndarray] = []
    chunks_rot: list[np.ndarray] = []
    chunks_pos: list[np.ndarray] = []
    cursor = 0
    seg_ranges: list[tuple[str, int, int]] = []
    rest_ranges: list[tuple[int, int]] = []

    for i, seg in enumerate(aligned):
        chunks_dof.append(seg["dof"])
        chunks_rot.append(seg["root_rot"])
        chunks_pos.append(seg["root_trans"])
        n = seg["dof"].shape[0]
        seg_ranges.append((playlist.segments[i].name, cursor, cursor + n))
        cursor += n

        if i < len(aligned) - 1:
            prev_last = {
                "dof": seg["dof"][-1],
                "root_rot": seg["root_rot"][-1],
                "root_trans": seg["root_trans"][-1],
            }
            next_first = {
                "dof": aligned[i + 1]["dof"][0],
                "root_rot": aligned[i + 1]["root_rot"][0],
                "root_trans": aligned[i + 1]["root_trans"][0],
            }
            rest_layer = _build_rest_layer(
                playlist.rest, rest_pose, prev_last, next_first
            )
            chunks_dof.append(rest_layer["dof"])
            chunks_rot.append(rest_layer["root_rot"])
            chunks_pos.append(rest_layer["root_trans"])
            r = rest_layer["dof"].shape[0]
            rest_ranges.append((cursor, cursor + r))
            cursor += r

            # The next segment was already aligned to *its own* prior cursor
            # (i.e. seg[i].last). The rest layer happens AT seg[i].last, so
            # seg[i+1] still needs to start at seg[i].last_xy/yaw. Already
            # holds: nothing to fix.

    out_dof = np.concatenate(chunks_dof, axis=0).astype(np.float32)
    out_rot = np.concatenate(chunks_rot, axis=0).astype(np.float32)
    out_pos = np.concatenate(chunks_pos, axis=0).astype(np.float32)

    playlist.segment_frame_ranges = seg_ranges
    playlist.rest_frame_ranges = rest_ranges

    return {
        "dof": out_dof,
        "root_rot": out_rot,
        "root_trans_offset": out_pos,
        "fps": float(playlist.fps),
    }


def write_pkl(playlist: Playlist, motion: dict[str, np.ndarray | float], out: Path) -> None:
    """Dump ``{playlist.output_key: motion}`` to ``out`` (creates parent dir)."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({playlist.output_key: motion}, out)


# ---------------------------------------------------------- diagnostics


def _arm_summary(dof_block: np.ndarray) -> str:
    means = dof_block.mean(axis=0)
    return (
        f"L_sh[p={means[15]:+.2f} r={means[16]:+.2f}] L_el={means[18]:+.2f}  "
        f"R_sh[p={means[22]:+.2f} r={means[23]:+.2f}] R_el={means[25]:+.2f}"
    )


def diagnostics_lines(
    playlist: Playlist, motion: dict[str, np.ndarray | float]
) -> Iterable[str]:
    """Yield one line per segment + per seam + an end-to-end summary.

    Side-effect free; safe to call as ``print('\n'.join(diagnostics_lines(...)))``.
    """
    dof = np.asarray(motion["dof"])
    rot = np.asarray(motion["root_rot"])
    pos = np.asarray(motion["root_trans_offset"])
    fps = float(motion["fps"])

    yield f"[playlist] {playlist.name!r} -> output_key={playlist.output_key!r}"
    yield (
        f"           {dof.shape[0]} frames @ {fps:g} fps "
        f"({dof.shape[0] / fps:.2f} s), {len(playlist.segments)} segments + "
        f"{len(playlist.rest_frame_ranges)} rest layers"
    )
    yield ""
    yield "[segments]"
    for (name, lo, hi), spec in zip(playlist.segment_frame_ranges, playlist.segments):
        block = dof[lo:hi]
        yield (
            f"  {name:30s} key={spec.motion_key!s:50s} "
            f"frames=[{lo:4d}..{hi:4d}) ({(hi - lo) / fps:.2f}s) "
            f"pelvis_z_mean={pos[lo:hi, 2].mean():+.3f}  "
            f"arms={_arm_summary(block)}"
        )

    if playlist.rest_frame_ranges:
        yield ""
        yield "[seams]  (between consecutive segments, going through rest layer)"
        for i, (lo, hi) in enumerate(playlist.rest_frame_ranges):
            seam_xy = pos[lo - 1, :2] if lo > 0 else pos[0, :2]
            yaw_pre = _yaw_of(rot[lo - 1]) if lo > 0 else 0.0
            yaw_post = _yaw_of(rot[hi]) if hi < rot.shape[0] else yaw_pre
            # Max joint jump per control tick within the rest layer (proxy
            # for how violent the SLERP looks to the policy).
            if hi > lo + 1:
                jumps = np.abs(np.diff(dof[lo:hi], axis=0))
                max_jump = float(jumps.max())
            else:
                max_jump = 0.0
            yield (
                f"  seam[{i}]: rest=[{lo:4d}..{hi:4d}) ({(hi - lo) / fps:.2f}s)  "
                f"pre_yaw={np.degrees(yaw_pre):+6.1f}deg  post_yaw={np.degrees(yaw_post):+6.1f}deg  "
                f"max|d_dof|/tick={max_jump:.3f}rad  pinned_xy=({seam_xy[0]:+.2f},{seam_xy[1]:+.2f})"
            )

    yield ""
    yield "[end-to-end]"
    yaw0 = _yaw_of(rot[0])
    yawN = _yaw_of(rot[-1])
    xy_path = float(np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1).sum())
    yield (
        f"  total_frames     = {dof.shape[0]}"
    )
    yield (
        f"  total_duration   = {dof.shape[0] / fps:.2f} s"
    )
    yield (
        f"  start_xy_yaw     = ({pos[0, 0]:+.2f}, {pos[0, 1]:+.2f}, {np.degrees(yaw0):+6.1f}deg)"
    )
    yield (
        f"  end_xy_yaw       = ({pos[-1, 0]:+.2f}, {pos[-1, 1]:+.2f}, {np.degrees(yawN):+6.1f}deg)"
    )
    yield (
        f"  XY_path_length   = {xy_path:.2f} m"
    )
    yield (
        f"  pelvis_z_range   = [{pos[:, 2].min():.3f}, {pos[:, 2].max():.3f}] m"
    )
    yield (
        f"  dof_range        = [{dof.min():+.2f}, {dof.max():+.2f}] rad"
    )


__all__ = [
    "NUM_DOFS",
    "Playlist",
    "RestSpec",
    "SegmentSpec",
    "build_concat",
    "diagnostics_lines",
    "load_playlist",
    "write_pkl",
]
