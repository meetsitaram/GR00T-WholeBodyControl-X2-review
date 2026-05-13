"""Mine ``x2_ultra_bones_seed.pkl`` for X2 heuristic-planner motion primitives.

Re-runnable: each invocation rebuilds the primitive PKL + registry YAML +
markdown report from the current ``x2_planner_bins.yaml`` and source motion
library. Pinned rows in the registry (``pinned: true``) are preserved
verbatim and re-measured but never overwritten by candidate search.

Run from the repo root::

    .venv/bin/python -m gear_sonic.scripts.curate_x2_primitives \\
        --source gear_sonic/data/motions/x2_ultra_bones_seed.pkl \\
        --bins   gear_sonic/data/motions/x2_planner_bins.yaml \\
        --out    gear_sonic/data/motions

For per-bin debugging::

    .venv/bin/python -m gear_sonic.scripts.curate_x2_primitives \\
        --bins-only fwd_step_half_ft turn_left_45deg lean_fwd_medium \\
        --top-k 10

Outputs (in ``--out`` dir):

  - ``x2_planner_primitives.pkl``    : ``{bin_name: {dof, root_rot, root_trans, fps, source_pkl, motion_key, start_frame, n_frames}}``
  - ``x2_planner_primitives.yaml``   : registry (hand-editable; pin with ``pinned: true``)
  - ``x2_planner_primitives_report.md`` : top-K candidates per bin

Acceptance criteria for "perfect" vs "partial" matches are encoded in
``x2_planner_bins.yaml`` (per-bin tolerances + score gates). The CLI prints
a one-line summary at the end so CI can grep for failures.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np

# Make ``import gear_sonic.utils.planner`` work even when running as a script.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.utils.planner.constants import (  # noqa: E402
    DEFAULT_STAND_POSE_NP,
    HEAD_INDICES,
    LEFT_ARM_INDICES,
    RIGHT_ARM_INDICES,
)
from gear_sonic.utils.planner.metrics import (  # noqa: E402
    WindowMetrics,
    compute_window_metrics,
)
from gear_sonic.utils.planner.registry import (  # noqa: E402
    BinSpec,
    CandidateRow,
    PrimitiveEntry,
    load_bin_specs,
    load_primitive_registry,
    write_primitive_registry,
)


# Indices that get pinned to DEFAULT_STAND_POSE when a bin sets
# ``freeze_arms_to_default: true``. Arm + head only — we deliberately leave
# legs and waist alone so the static-upper-body waist twist/lean reaches the
# planner. The head pin is just so VLA / future neck-IK has full ownership.
_FREEZE_INDICES: tuple[int, ...] = (
    *LEFT_ARM_INDICES, *RIGHT_ARM_INDICES, *HEAD_INDICES,
)


# ---------------------------------------------------------------------------
# Motion-lib I/O
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class MotionClip:
    """One clip from the source motion library, normalized to numpy."""

    motion_key: str
    dof: np.ndarray  # (T, 31) float32
    root_rot_xyzw: np.ndarray  # (T, 4) float32
    root_trans: np.ndarray  # (T, 3) float32
    fps: float


def load_motion_library(path: Path) -> dict[str, MotionClip]:
    """Load and normalize the bones-seed-style PKL into ``{key: MotionClip}``.

    The library is structured as ``{motion_key: {dof, root_rot, root_trans_offset, fps, ...}}``.
    Extra keys (``pose_aa``, ``smpl_joints``, ...) are ignored.
    """
    raw = joblib.load(path)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected dict, got {type(raw).__name__}")
    out: dict[str, MotionClip] = {}
    for key, entry in raw.items():
        try:
            dof = np.asarray(entry["dof"], dtype=np.float32)
            root_rot = np.asarray(entry["root_rot"], dtype=np.float32)
            root_trans = np.asarray(entry["root_trans_offset"], dtype=np.float32)
            fps = float(entry.get("fps", 30))
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{path}: clip {key!r} missing field {exc}") from exc
        if dof.ndim != 2 or dof.shape[1] != 31:
            raise ValueError(
                f"{path}: clip {key!r} dof shape {dof.shape}, expected (T, 31)"
            )
        if root_rot.shape[0] != dof.shape[0] or root_rot.shape[1] != 4:
            raise ValueError(
                f"{path}: clip {key!r} root_rot shape {root_rot.shape}, "
                f"expected ({dof.shape[0]}, 4)"
            )
        if root_trans.shape[0] != dof.shape[0] or root_trans.shape[1] != 3:
            raise ValueError(
                f"{path}: clip {key!r} root_trans shape {root_trans.shape}, "
                f"expected ({dof.shape[0]}, 3)"
            )
        out[str(key)] = MotionClip(
            motion_key=str(key),
            dof=dof,
            root_rot_xyzw=root_rot,
            root_trans=root_trans,
            fps=fps,
        )
    return out


# ---------------------------------------------------------------------------
# Window enumeration
# ---------------------------------------------------------------------------


def _window_lengths_for_bin(spec: BinSpec) -> list[int]:
    """Sample 4 lengths between min and max, inclusive (or fewer if range is small)."""
    if spec.window_frames_min == spec.window_frames_max:
        return [spec.window_frames_min]
    n = 4
    return sorted(
        set(
            int(round(x))
            for x in np.linspace(spec.window_frames_min, spec.window_frames_max, n)
        )
    )


def _window_stride_for_length(length: int) -> int:
    return max(2, length // 4)


def enumerate_windows(
    clip: MotionClip, lengths: Sequence[int]
) -> Iterable[tuple[int, int]]:
    """Yield ``(start_frame, n_frames)`` pairs covering one clip with a given length set.

    Stride is ``max(2, length // 4)`` so windows of the same length overlap
    by ~75% — enough to catch the "stop" frame of a stride wherever it sits.
    """
    T = clip.dof.shape[0]
    for length in lengths:
        if length > T or length < 2:
            continue
        stride = _window_stride_for_length(length)
        for start in range(0, T - length + 1, stride):
            yield start, length


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ScoreResult:
    """Output of ``score_window`` — gates + sub-scores + total."""

    score: float  # 0..1, product of gate sub-scores; 1.0 = perfect
    passes: bool  # all hard gates pass AND all sub-scores >= 0.5
    sub_scores: dict[str, float]
    pass_fail: dict[str, bool]
    cross_axis_xy_m: float
    measured_xy_m: tuple[float, float]
    measured_yaw_deg: float


def _intent_axis_for_bin(spec: BinSpec) -> tuple[float, float] | None:
    """Unit vector for the bin's intended XY axis in body frame, or None."""
    if not spec.is_locomotion():
        return None
    norm = float(np.linalg.norm(spec.target_xy_m))
    if norm < 1e-3:
        return None
    return (spec.target_xy_m[0] / norm, spec.target_xy_m[1] / norm)


def _gauss_score(error: float, tol: float) -> float:
    """Smooth score: 1.0 at error=0, ~0.5 at error=tol, ~0.13 at error=2*tol."""
    if tol <= 0:
        return 1.0 if abs(error) < 1e-9 else 0.0
    return float(np.exp(-((error / tol) ** 2)))


def score_window(spec: BinSpec, m: WindowMetrics) -> ScoreResult:
    """Score one window against one bin spec; produce gates + total."""
    sub: dict[str, float] = {}
    pf: dict[str, bool] = {}

    # ---- pelvis-z band (hard gate, all families)
    z_min, z_max = spec.pelvis_z_band_m
    pf["pelvis_z_band"] = (m.pelvis_z_min_m >= z_min) and (m.pelvis_z_max_m <= z_max)
    # Soft sub-score: distance from the band center as a fraction of half-width
    z_center = 0.5 * (z_min + z_max)
    z_half = max(1e-3, 0.5 * (z_max - z_min))
    z_drift = max(
        abs(m.pelvis_z_min_m - z_center),
        abs(m.pelvis_z_max_m - z_center),
    )
    sub["pelvis_z_band"] = _gauss_score(max(0.0, z_drift - z_half), z_half)

    if spec.family == "idle":
        # Idle: should be loopable — start ≈ end, low XY/yaw drift.
        sub["loop_dof"] = _gauss_score(m.loop_dof_drift, 0.05)
        pf["loop_dof"] = m.loop_dof_drift < 0.10
        sub["loop_quat"] = _gauss_score(m.loop_quat_distance, 0.02)
        pf["loop_quat"] = m.loop_quat_distance < 0.05
        xy_norm = float(np.linalg.norm(m.net_xy_body_m))
        sub["xy_drift"] = _gauss_score(xy_norm, spec.tol_xy_m)
        pf["xy_drift"] = xy_norm < spec.tol_xy_m
        sub["yaw_drift"] = _gauss_score(abs(m.net_yaw_deg), spec.tol_yaw_deg)
        pf["yaw_drift"] = abs(m.net_yaw_deg) < spec.tol_yaw_deg
        sub["feet_planted"] = m.feet_planted_score
        pf["feet_planted"] = m.feet_planted_score >= spec.feet_planted_min
        cross = xy_norm  # idle has no preferred axis
    elif spec.family in ("locomotion", "continuous_walk"):
        intent = _intent_axis_for_bin(spec)
        if intent is None:
            # Pure rotation bin: target_xy_m == (0, 0). Cross axis = full XY.
            cross = float(np.linalg.norm(m.net_xy_body_m))
            along = 0.0
        else:
            ax, ay = intent
            along = float(m.net_xy_body_m[0] * ax + m.net_xy_body_m[1] * ay)
            cross = float(abs(m.net_xy_body_m[0] * ay - m.net_xy_body_m[1] * ax))
        target_along = float(np.hypot(spec.target_xy_m[0], spec.target_xy_m[1]))
        if intent is None:
            sub["xy_drift"] = _gauss_score(cross, spec.cross_axis_max_m)
            pf["xy_drift"] = cross < spec.cross_axis_max_m
        else:
            sub["xy_along"] = _gauss_score(abs(along - target_along), spec.tol_xy_m)
            pf["xy_along"] = abs(along - target_along) < spec.tol_xy_m
        sub["cross_axis"] = _gauss_score(cross, spec.cross_axis_max_m)
        pf["cross_axis"] = cross < spec.cross_axis_max_m
        yaw_err = abs(m.net_yaw_deg - spec.target_yaw_deg)
        sub["yaw"] = _gauss_score(yaw_err, spec.tol_yaw_deg)
        pf["yaw"] = yaw_err < spec.tol_yaw_deg
        if spec.family == "locomotion":
            sub["end_at_square"] = m.end_at_square_score
            pf["end_at_square"] = m.end_at_square_score >= spec.end_at_square_min
            if spec.stride_count_target is not None:
                sub["stride_count"] = (
                    1.0 if m.stride_count == spec.stride_count_target else 0.4
                )
                pf["stride_count"] = m.stride_count == spec.stride_count_target
            # Hard floor on stride count: if set, drop windows that don't
            # actually lift a foot enough times. Multiplies sub-score to 0
            # so the candidate is filtered out before ranking.
            if spec.stride_count_min is not None:
                meets = m.stride_count >= spec.stride_count_min
                sub["stride_count_min"] = 1.0 if meets else 0.0
                pf["stride_count_min"] = meets
    elif spec.family == "static_upper_body":
        cross = 0.0
        # Pick dominant axis by spec target.
        target_pitch = abs(spec.target_waist_pitch_deg)
        target_yaw = abs(spec.target_waist_yaw_deg)
        if target_pitch > 1e-3 and target_pitch >= target_yaw:
            err = abs(m.waist_pitch_apex_deg - target_pitch)
            sub["waist_axis"] = _gauss_score(err, spec.tol_waist_deg)
            pf["waist_axis"] = err < spec.tol_waist_deg
            # Wrong-axis penalty.
            sub["off_axis_yaw"] = _gauss_score(m.waist_yaw_apex_deg, 8.0)
            pf["off_axis_yaw"] = m.waist_yaw_apex_deg < 12.0
        elif target_yaw > 1e-3:
            # Signed yaw target — direction matters for torso_left vs _right.
            signed_apex = (
                m.waist_yaw_apex_deg
                if spec.target_waist_yaw_deg >= 0
                else -m.waist_yaw_apex_deg
            )
            err = abs(signed_apex - spec.target_waist_yaw_deg)
            # Note: waist_yaw_apex_deg is unsigned. To detect direction, fall
            # back to the registry's pin if direction is wrong (handled below).
            sub["waist_axis"] = _gauss_score(err, spec.tol_waist_deg)
            pf["waist_axis"] = err < spec.tol_waist_deg
            sub["off_axis_pitch"] = _gauss_score(m.waist_pitch_apex_deg, 8.0)
            pf["off_axis_pitch"] = m.waist_pitch_apex_deg < 12.0
        else:
            sub["waist_axis"] = 0.5
            pf["waist_axis"] = False
        sub["end_at_apex"] = m.end_at_apex_score
        pf["end_at_apex"] = m.end_at_apex_score >= spec.end_at_apex_min
        sub["feet_planted"] = m.feet_planted_score
        pf["feet_planted"] = m.feet_planted_score >= spec.feet_planted_min
    else:
        cross = 0.0
        sub["unknown_family"] = 0.0
        pf["unknown_family"] = False

    score = float(np.prod(list(sub.values()))) if sub else 0.0
    passes = all(pf.values()) and all(v >= 0.4 for v in sub.values())
    return ScoreResult(
        score=score,
        passes=passes,
        sub_scores=sub,
        pass_fail=pf,
        cross_axis_xy_m=cross,
        measured_xy_m=(float(m.net_xy_body_m[0]), float(m.net_xy_body_m[1])),
        measured_yaw_deg=float(m.net_yaw_deg),
    )


# ---------------------------------------------------------------------------
# Direction-aware bookkeeping for static_upper_body / lateral bins
# ---------------------------------------------------------------------------


def _signed_waist_yaw_apex(dof: np.ndarray, idx: int) -> float:
    """Signed peak (degrees) — sign = direction of the largest |drift|."""
    drift = dof[:, idx] - dof[0, idx]
    apex_idx = int(np.argmax(np.abs(drift)))
    return float(np.degrees(drift[apex_idx]))


def _direction_matches_static(spec: BinSpec, dof: np.ndarray) -> bool:
    """For ``torso_left/right_*`` bins, verify the dominant axis sign matches."""
    if spec.family != "static_upper_body":
        return True
    if abs(spec.target_waist_yaw_deg) > 1e-3:
        signed = _signed_waist_yaw_apex(dof, 12)  # WAIST_YAW_IDX
        return (signed * spec.target_waist_yaw_deg) > 0
    if abs(spec.target_waist_pitch_deg) > 1e-3:
        signed = _signed_waist_yaw_apex(dof, 13)  # WAIST_PITCH_IDX
        return (signed * spec.target_waist_pitch_deg) > 0
    return True


# ---------------------------------------------------------------------------
# Curator main
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class CuratedBin:
    """All output state for one bin."""

    spec: BinSpec
    selected: PrimitiveEntry | None
    candidates: list[CandidateRow]  # top-K, sorted by score desc


def _filter_clip_keys(
    motion_keys: Iterable[str], name_regex: str | None
) -> list[str]:
    if not name_regex:
        return list(motion_keys)
    pat = re.compile(name_regex)
    return [k for k in motion_keys if pat.search(k)]


def _candidate_from_window(
    clip: MotionClip,
    start: int,
    length: int,
    spec: BinSpec,
    m: WindowMetrics,
    score: ScoreResult,
    source_pkl: str,
) -> CandidateRow:
    return CandidateRow(
        motion_key=clip.motion_key,
        source_pkl=source_pkl,
        start_frame=start,
        n_frames=length,
        score=score.score,
        measured_xy_m=score.measured_xy_m,
        measured_yaw_deg=score.measured_yaw_deg,
        measured_waist_pitch_deg=m.waist_pitch_apex_deg,
        measured_waist_yaw_deg=m.waist_yaw_apex_deg,
        end_at_square_score=m.end_at_square_score,
        end_at_apex_score=m.end_at_apex_score,
        feet_planted_score=m.feet_planted_score,
        pelvis_z_min_m=m.pelvis_z_min_m,
        pelvis_z_max_m=m.pelvis_z_max_m,
        stride_count=m.stride_count,
        pass_fail=dict(score.pass_fail),
    )


def _measure_existing(
    clip: MotionClip, entry: PrimitiveEntry, spec: BinSpec
) -> tuple[PrimitiveEntry, ScoreResult, WindowMetrics]:
    """Recompute metrics for an existing/pinned registry row and update fields."""
    s, e = entry.start_frame, entry.start_frame + entry.n_frames
    if e > clip.dof.shape[0] or s < 0:
        raise ValueError(
            f"pinned entry for {entry.bin_name!r} ({entry.motion_key} "
            f"[{s}:{e}]) is out of range (clip length {clip.dof.shape[0]})"
        )
    m = compute_window_metrics(
        clip.dof[s:e],
        clip.root_rot_xyzw[s:e],
        clip.root_trans[s:e],
        clip.fps,
    )
    score = score_window(spec, m)
    new_entry = PrimitiveEntry(
        bin_name=entry.bin_name,
        source_pkl=entry.source_pkl,
        motion_key=entry.motion_key,
        start_frame=entry.start_frame,
        n_frames=entry.n_frames,
        fps=clip.fps,
        measured_xy_m=score.measured_xy_m,
        measured_yaw_deg=score.measured_yaw_deg,
        measured_waist_pitch_deg=m.waist_pitch_apex_deg,
        measured_waist_yaw_deg=m.waist_yaw_apex_deg,
        end_at_square_score=m.end_at_square_score,
        end_at_apex_score=m.end_at_apex_score,
        feet_planted_score=m.feet_planted_score,
        pelvis_z_min_m=m.pelvis_z_min_m,
        pelvis_z_max_m=m.pelvis_z_max_m,
        stride_count=m.stride_count,
        partial=not score.passes,
        pinned=entry.pinned,
        notes=entry.notes,
    )
    return new_entry, score, m


def curate_one_bin(
    spec: BinSpec,
    clips: dict[str, MotionClip],
    pinned: PrimitiveEntry | None,
    source_pkl: str,
    top_k: int,
) -> CuratedBin:
    """Score every candidate window in the corpus for one bin and pick the best."""
    if pinned is not None and pinned.pinned:
        clip = clips.get(pinned.motion_key)
        if clip is None:
            raise ValueError(
                f"pinned entry for {spec.name!r} references unknown motion_key "
                f"{pinned.motion_key!r}"
            )
        new_entry, score, m = _measure_existing(clip, pinned, spec)
        cand = _candidate_from_window(
            clip,
            new_entry.start_frame,
            new_entry.n_frames,
            spec,
            m,
            score,
            source_pkl,
        )
        return CuratedBin(spec=spec, selected=new_entry, candidates=[cand])

    candidate_keys = _filter_clip_keys(clips.keys(), spec.name_regex)
    lengths = _window_lengths_for_bin(spec)
    rows: list[CandidateRow] = []
    for key in candidate_keys:
        clip = clips[key]
        for start, length in enumerate_windows(clip, lengths):
            m = compute_window_metrics(
                clip.dof[start : start + length],
                clip.root_rot_xyzw[start : start + length],
                clip.root_trans[start : start + length],
                clip.fps,
            )
            # Direction filter for static_upper_body: skip windows whose
            # dominant signed waist axis points the wrong way.
            if not _direction_matches_static(spec, clip.dof[start : start + length]):
                continue
            score = score_window(spec, m)
            if score.score <= 0.0:
                continue
            rows.append(
                _candidate_from_window(
                    clip, start, length, spec, m, score, source_pkl
                )
            )

    rows.sort(key=lambda r: r.score, reverse=True)
    rows = rows[:top_k]
    if not rows:
        return CuratedBin(spec=spec, selected=None, candidates=[])

    best = rows[0]
    selected = PrimitiveEntry(
        bin_name=spec.name,
        source_pkl=source_pkl,
        motion_key=best.motion_key,
        start_frame=best.start_frame,
        n_frames=best.n_frames,
        fps=clips[best.motion_key].fps,
        measured_xy_m=best.measured_xy_m,
        measured_yaw_deg=best.measured_yaw_deg,
        measured_waist_pitch_deg=best.measured_waist_pitch_deg,
        measured_waist_yaw_deg=best.measured_waist_yaw_deg,
        end_at_square_score=best.end_at_square_score,
        end_at_apex_score=best.end_at_apex_score,
        feet_planted_score=best.feet_planted_score,
        pelvis_z_min_m=best.pelvis_z_min_m,
        pelvis_z_max_m=best.pelvis_z_max_m,
        stride_count=best.stride_count,
        partial=not all(best.pass_fail.values()),
        pinned=False,
        notes="auto-selected by curator",
    )
    return CuratedBin(spec=spec, selected=selected, candidates=rows)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_primitives_pkl(
    path: Path,
    curated: dict[str, CuratedBin],
    clips: dict[str, MotionClip],
    source_pkl: str,
) -> int:
    """Pickle the actual sliced motion data the runtime planner consumes.

    Applies ``freeze_arms_to_default`` per-bin: when set, arm + head DOFs in
    the curated slice are overwritten with ``DEFAULT_STAND_POSE_NP`` for every
    frame. The legs and waist are untouched, so the static-upper-body waist
    twist/lean still reaches the runtime, but the arms stay neutral so the
    downstream VLA / teleop owns them without interference.
    """
    freeze_template = np.asarray(DEFAULT_STAND_POSE_NP, dtype=np.float32)
    out: dict[str, dict[str, Any]] = {}
    for bin_name, cb in curated.items():
        if cb.selected is None:
            continue
        clip = clips[cb.selected.motion_key]
        s = cb.selected.start_frame
        e = s + cb.selected.n_frames
        dof = clip.dof[s:e].copy()
        if cb.spec.freeze_arms_to_default:
            for idx in _FREEZE_INDICES:
                dof[:, idx] = freeze_template[idx]
        out[bin_name] = {
            "dof": dof,
            "root_rot_xyzw": clip.root_rot_xyzw[s:e].copy(),
            "root_trans": clip.root_trans[s:e].copy(),
            "fps": clip.fps,
            "source_pkl": source_pkl,
            "motion_key": cb.selected.motion_key,
            "start_frame": int(s),
            "n_frames": int(cb.selected.n_frames),
            "partial": bool(cb.selected.partial),
            "pinned": bool(cb.selected.pinned),
            "freeze_arms_to_default": bool(cb.spec.freeze_arms_to_default),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, path)
    return len(out)


def write_report(
    path: Path,
    curated: dict[str, CuratedBin],
    source_pkl: str,
    started_at: float,
) -> None:
    """Markdown report: per-bin section with selected + top-K candidates."""
    lines: list[str] = []
    perfect = sum(
        1 for cb in curated.values() if cb.selected and not cb.selected.partial
    )
    partial = sum(
        1 for cb in curated.values() if cb.selected and cb.selected.partial
    )
    missing = sum(1 for cb in curated.values() if cb.selected is None)
    lines.append("# X2 planner primitive curator report")
    lines.append("")
    lines.append(f"- Source: `{source_pkl}`")
    lines.append(f"- Bins: {len(curated)}")
    lines.append(f"- Perfect matches: **{perfect}**")
    lines.append(f"- Partial matches: **{partial}** (review and pin)")
    lines.append(f"- Missing: **{missing}** (no candidate had score > 0)")
    lines.append(f"- Curator wall time: {time.time() - started_at:.1f}s")
    lines.append("")
    lines.append("## Bin status summary")
    lines.append("")
    lines.append("| Bin | Status | Best score | Selected motion_key | Frames |")
    lines.append("|---|---|---|---|---|")
    for name in sorted(curated):
        cb = curated[name]
        if cb.selected is None:
            lines.append(f"| `{name}` | MISSING | — | — | — |")
            continue
        status = "PIN" if cb.selected.pinned else (
            "PARTIAL" if cb.selected.partial else "OK"
        )
        score = cb.candidates[0].score if cb.candidates else 0.0
        frames = f"{cb.selected.start_frame}..{cb.selected.start_frame + cb.selected.n_frames}"
        lines.append(
            f"| `{name}` | {status} | {score:.3f} "
            f"| `{cb.selected.motion_key}` | {frames} |"
        )
    lines.append("")
    lines.append("## Per-bin candidates (top-K)")
    for name in sorted(curated):
        cb = curated[name]
        spec = cb.spec
        lines.append("")
        lines.append(f"### `{name}` ({spec.family})")
        if spec.is_locomotion():
            lines.append(
                f"- target_xy_m: `{spec.target_xy_m}` "
                f"target_yaw_deg: `{spec.target_yaw_deg}`"
            )
            lines.append(
                f"- tol_xy_m: `{spec.tol_xy_m}` "
                f"tol_yaw_deg: `{spec.tol_yaw_deg}` "
                f"cross_axis_max_m: `{spec.cross_axis_max_m}`"
            )
        if spec.is_static():
            lines.append(
                f"- target_waist_pitch_deg: `{spec.target_waist_pitch_deg}` "
                f"target_waist_yaw_deg: `{spec.target_waist_yaw_deg}`"
            )
            lines.append(
                f"- tol_waist_deg: `{spec.tol_waist_deg}` "
                f"end_at_apex_min: `{spec.end_at_apex_min}` "
                f"feet_planted_min: `{spec.feet_planted_min}`"
            )
        if not cb.candidates:
            lines.append("")
            lines.append("_no candidates with score > 0 — relax tolerances or pin a clip manually._")
            continue
        lines.append("")
        lines.append(
            "| Rank | Score | motion_key | start | N | xy_m | yaw_deg | "
            "waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |"
        )
        lines.append(
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
        )
        for rank, c in enumerate(cb.candidates):
            gates_failing = [k for k, v in c.pass_fail.items() if not v]
            gates_str = ",".join(gates_failing) if gates_failing else "all-pass"
            xy_str = f"({c.measured_xy_m[0]:.3f},{c.measured_xy_m[1]:.3f})"
            lines.append(
                f"| {rank + 1} | {c.score:.3f} | `{c.motion_key}` | "
                f"{c.start_frame} | {c.n_frames} | {xy_str} | "
                f"{c.measured_yaw_deg:.1f} | {c.measured_waist_pitch_deg:.1f} | "
                f"{c.measured_waist_yaw_deg:.1f} | "
                f"{c.end_at_square_score:.2f} | {c.end_at_apex_score:.2f} | "
                f"{c.feet_planted_score:.2f} | {c.stride_count} | {gates_str} |"
            )
    lines.append("")
    lines.append("---")
    lines.append("Re-run the curator after editing the bins YAML or pinning rows in the registry.")
    lines.append(
        "To pin a candidate: copy its `motion_key`, `start_frame`, `n_frames` "
        "into the registry YAML row for that bin and set `pinned: true`."
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="curate_x2_primitives",
        description=(
            "Mine x2_ultra_bones_seed.pkl for X2 heuristic-planner primitives. "
            "Re-runnable; respects pinned registry rows."
        ),
    )
    p.add_argument(
        "--source",
        type=Path,
        default=Path("gear_sonic/data/motions/x2_ultra_bones_seed.pkl"),
        help="Source motion library PKL.",
    )
    p.add_argument(
        "--bins",
        type=Path,
        default=Path("gear_sonic/data/motions/x2_planner_bins.yaml"),
        help="Bin spec YAML (hand-edited).",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("gear_sonic/data/motions"),
        help="Output dir for primitives PKL + registry YAML + report.",
    )
    p.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Number of top candidates to keep per bin in the markdown report.",
    )
    p.add_argument(
        "--bins-only",
        nargs="*",
        default=None,
        help="Restrict curation to these bin names (others remain unchanged in the registry).",
    )
    p.add_argument(
        "--no-write",
        action="store_true",
        help="Compute everything but do not write outputs (dry run).",
    )
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    started_at = time.time()

    print(f"[curator] Loading bin spec from {args.bins} ...")
    bin_specs = load_bin_specs(args.bins)
    if args.bins_only:
        unknown = set(args.bins_only) - set(bin_specs)
        if unknown:
            print(f"[curator] ERROR: unknown bin names: {sorted(unknown)}", file=sys.stderr)
            return 2
        bin_specs = {k: v for k, v in bin_specs.items() if k in set(args.bins_only)}
    print(f"[curator] {len(bin_specs)} bins to curate.")

    print(f"[curator] Loading motion library {args.source} ...")
    clips = load_motion_library(args.source)
    print(f"[curator] {len(clips)} clips loaded.")

    registry_path = args.out / "x2_planner_primitives.yaml"
    pinned = load_primitive_registry(registry_path)
    print(
        f"[curator] {sum(1 for e in pinned.values() if e.pinned)} pinned rows "
        f"will be preserved."
    )

    curated: dict[str, CuratedBin] = {}
    for bin_name, spec in bin_specs.items():
        existing = pinned.get(bin_name)
        bin_started = time.time()
        cb = curate_one_bin(
            spec,
            clips,
            existing if existing and existing.pinned else None,
            source_pkl=str(args.source),
            top_k=args.top_k,
        )
        elapsed = time.time() - bin_started
        if cb.selected is None:
            print(f"  [{bin_name:>26}] MISSING (no candidates) ({elapsed:.1f}s)")
        else:
            tag = "PIN" if cb.selected.pinned else (
                "PARTIAL" if cb.selected.partial else "OK"
            )
            best_score = cb.candidates[0].score if cb.candidates else 0.0
            print(
                f"  [{bin_name:>26}] {tag:7s} score={best_score:.3f} "
                f"key={cb.selected.motion_key} "
                f"frames={cb.selected.start_frame}..{cb.selected.start_frame + cb.selected.n_frames} "
                f"({elapsed:.1f}s)"
            )
        curated[bin_name] = cb

    # Merge into existing registry: preserve un-curated bins (subset runs)
    # and keep pinned rows verbatim with refreshed measurements.
    merged: dict[str, PrimitiveEntry] = dict(pinned)
    for bin_name, cb in curated.items():
        if cb.selected is not None:
            merged[bin_name] = cb.selected

    if args.no_write:
        print("[curator] --no-write specified; skipping outputs.")
        return 0

    write_primitive_registry(registry_path, merged)
    print(f"[curator] Wrote registry: {registry_path}")
    pkl_path = args.out / "x2_planner_primitives.pkl"
    n_written = write_primitives_pkl(pkl_path, curated, clips, str(args.source))
    print(f"[curator] Wrote {n_written} primitive slices to {pkl_path}")
    report_path = args.out / "x2_planner_primitives_report.md"
    write_report(report_path, curated, str(args.source), started_at)
    print(f"[curator] Wrote report: {report_path}")

    n_perfect = sum(
        1 for cb in curated.values() if cb.selected and not cb.selected.partial
    )
    n_partial = sum(
        1 for cb in curated.values() if cb.selected and cb.selected.partial
    )
    n_missing = sum(1 for cb in curated.values() if cb.selected is None)
    print(
        f"[curator] DONE in {time.time() - started_at:.1f}s — "
        f"perfect={n_perfect}, partial={n_partial}, missing={n_missing}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
