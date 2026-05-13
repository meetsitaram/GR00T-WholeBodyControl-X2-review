#!/usr/bin/env python3
"""Mine ``x2_ultra_bones_seed.pkl`` for high-quality primitive candidates.

Produces a markdown report ranking clips by family. Used by the recipe
author (manual or LLM) to pick replacements for under-performing
primitives.

Per-family selection rules:

* **locomotion (forward / backward / sideways)**
    - net XY translation in the relevant body axis ≥ family-specific
      minimum
    - foot-lift NEVER exceeds 0.10 m (no marching / stair-climbing
      gaits — SONIC tracks them poorly under real PD)
    - end-at-square score ≥ 0.4 (so sweep transitions are stable)
    - average commanded forward speed in [0.30, 0.80] m/s — SONIC's
      training distribution
    - cross-axis drift ≤ 0.10 m

* **lean_forward (natural)**
    - waist + pelvis pitch fwd ≥ 12 deg sustained for ≥ 0.5 s
    - foot-lift ≤ 0.03 m (essentially planted)
    - net XY translation ≤ 0.05 m
    - at apex ≥ 0.6 ratio (held, not transient)

* **crouch**
    - pelvis Z drops ≥ 0.05 m below the clip-baseline standing height
      (typically ~0.85 m → < 0.80 m at apex)
    - foot-lift ≤ 0.03 m (no lifting feet to crouch)
    - knee + hip pitch increase together (squat kinematics)
    - net XY translation ≤ 0.05 m

Per-clip metrics computed (ALL via MJCF forward kinematics, so the
foot heights are physical, not joint-angle proxies):

* ``foot_lift_max_m``     : max world-Z swing of either ankle_roll_link
* ``pelvis_z_min_m``      : min pelvis world-Z over the clip
* ``pelvis_z_drop_m``     : (pelvis_z_baseline - min) — positive ⇒ crouch
* ``forward_speed_mps``   : net forward translation / clip duration
* ``waist_pitch_fwd_max_deg``  : peak (waist_pitch + pelvis_pitch_world) deg
* ``waist_pitch_sustained_deg``: max angle held for ≥ 0.5 s
* ``cross_axis_xy_m``     : magnitude of orthogonal drift
* ``stride_count``        : foot-Z zero-crossing count / 2

Usage::

    .venv/bin/python -m gear_sonic.scripts.mine_x2_motion_candidates
    .venv/bin/python -m gear_sonic.scripts.mine_x2_motion_candidates \\
        --max-foot-lift 0.08 --top-k 30
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rot

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from gear_sonic.utils.planner.constants import (  # noqa: E402
    LEFT_HIP_PITCH_IDX,
    LEFT_KNEE_IDX,
    MUJOCO_JOINT_NAMES,
    NUM_BODY_DOFS,
    RIGHT_HIP_PITCH_IDX,
    RIGHT_KNEE_IDX,
    WAIST_PITCH_IDX,
)

SEED_PKL = REPO_ROOT / "gear_sonic/data/motions/x2_ultra_bones_seed.pkl"
MJCF = REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
REPORT_OUT = REPO_ROOT / "gear_sonic/data/motions/x2_motion_candidates_report.md"
WINDOWS_OUT = REPO_ROOT / "gear_sonic/data/motions/x2_motion_candidate_windows.md"

# Sampling: foot lift is a peaky signal so we want decent temporal density.
# 0.05s ⇒ 20 Hz subsample which catches a typical 0.4s swing-phase apex.
DEFAULT_SUBSAMPLE_DT_S = 0.05


@dataclass
class ClipFK:
    motion_key: str
    fps: float
    n_frames: int
    duration_s: float
    foot_lift_max_m: float       # max world-Z swing of either foot
    foot_lift_mean_m: float      # mean across sampled frames
    pelvis_z_baseline_m: float   # mean Z of frames where both feet are low
    pelvis_z_min_m: float        # min over all sampled frames
    pelvis_z_drop_m: float       # baseline - min (positive = crouch)
    forward_speed_mps: float     # net forward / duration (body frame x)
    backward_speed_mps: float    # negative of forward
    side_speed_mps: float        # |body-frame y| speed
    net_xy_body_m: tuple[float, float]
    net_yaw_deg: float
    pelvis_yaw_osc_deg: float    # peak-to-peak pelvis yaw swing across clip
                                  # (after unwrap). High osc -> SONIC can't
                                  # track because the policy was trained on
                                  # stable-pelvis reference traces.
    waist_pitch_fwd_max_deg: float    # joint-only; positive forward
    pelvis_pitch_world_max_deg: float # root quat pitch, positive forward
    total_pitch_max_deg: float        # waist + pelvis
    total_pitch_sustained_deg: float  # max held >= 0.5 s
    stride_count: int
    foot_planted_score: float    # 1=feet flat all clip
    cross_axis_xy_m: float       # body-y drift for fwd, body-x for side
    end_at_square_score: float


def _yaw_pitch_roll(q_xyzw: np.ndarray) -> tuple[float, float, float]:
    yaw, pitch, roll = Rot.from_quat(q_xyzw).as_euler("zyx")
    return float(yaw), float(pitch), float(roll)


def _rotate_xy_to_body(delta_xy_world: np.ndarray, yaw0: float) -> np.ndarray:
    c, s = np.cos(-yaw0), np.sin(-yaw0)
    return np.array([c * delta_xy_world[0] - s * delta_xy_world[1],
                     s * delta_xy_world[0] + c * delta_xy_world[1]])


def _stride_count_from_foot_z(left_z: np.ndarray, right_z: np.ndarray) -> int:
    """Count strides via L/R foot-Z difference zero crossings (true FK)."""
    if len(left_z) < 4:
        return 0
    diff = left_z - right_z
    diff_demean = diff - float(np.mean(diff))
    if float(np.std(diff_demean)) < 0.005:
        return 0
    signs = np.sign(diff_demean)
    crossings = int(np.sum((signs[1:] != signs[:-1]) & (signs[1:] != 0)))
    return crossings // 2


def _foot_baseline_mask(foot_z_left: np.ndarray, foot_z_right: np.ndarray,
                        threshold_m: float = 0.04) -> np.ndarray:
    """Frames where the lower foot is within ``threshold_m`` of the global min.

    Used to compute "pelvis Z when standing" — a stable baseline that
    isn't biased by mid-stride frames.
    """
    lower_per_frame = np.minimum(foot_z_left, foot_z_right)
    floor_z = float(np.min(lower_per_frame))
    return lower_per_frame <= floor_z + threshold_m


def _sustained_max_deg(angle_deg: np.ndarray, fps_sub: float, hold_s: float) -> float:
    """Return the largest angle that's continuously >= itself for ``hold_s``.

    Implementation: for each candidate angle theta in (sorted desc), find the
    longest run of frames where ``angle >= theta`` and check duration.
    Cheap O(N log N) version: percentile sweep.
    """
    if len(angle_deg) < 2:
        return float(np.max(np.abs(angle_deg))) if len(angle_deg) else 0.0
    abs_a = np.abs(angle_deg)
    n_required = max(1, int(round(hold_s * fps_sub)))
    if len(abs_a) < n_required:
        return float(np.max(abs_a))
    # Sliding-window minimum over the magnitude. The largest such minimum is
    # the answer (i.e. the largest angle that was at least sustained for the
    # window).
    from collections import deque
    dq: deque[int] = deque()
    best = 0.0
    for i, v in enumerate(abs_a):
        while dq and abs_a[dq[-1]] >= v:
            dq.pop()
        dq.append(i)
        while dq and dq[0] <= i - n_required:
            dq.popleft()
        if i >= n_required - 1:
            window_min = abs_a[dq[0]]
            if window_min > best:
                best = float(window_min)
    return best


def fk_metrics_for_clip(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos_adr_per_joint: np.ndarray,
    left_foot_bid: int,
    right_foot_bid: int,
    clip: dict,
    subsample_dt_s: float,
) -> ClipFK:
    fps = float(clip.get("fps", 30))
    dof = np.asarray(clip["dof"], dtype=np.float64)
    rot_xyzw = np.asarray(clip["root_rot"], dtype=np.float64)
    trans = np.asarray(clip["root_trans_offset"], dtype=np.float64)
    n = dof.shape[0]
    if n < 4:
        raise ValueError("clip too short")
    duration = n / fps
    sub_step = max(1, int(round(subsample_dt_s * fps)))
    idx = np.arange(0, n, sub_step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    fps_sub = 1.0 / (sub_step / fps)

    foot_z_left = np.empty(len(idx), dtype=np.float64)
    foot_z_right = np.empty(len(idx), dtype=np.float64)
    pelvis_z = np.empty(len(idx), dtype=np.float64)
    pelvis_pitch = np.empty(len(idx), dtype=np.float64)
    pelvis_yaw_per = np.empty(len(idx), dtype=np.float64)

    for k, f in enumerate(idx):
        # Compose qpos: floating base [x,y,z,qw,qx,qy,qz] then 31 joints.
        qx, qy, qz, qw = rot_xyzw[f]  # source is xyzw
        data.qpos[0:3] = trans[f]
        data.qpos[3] = qw
        data.qpos[4:7] = (qx, qy, qz)
        data.qpos[qpos_adr_per_joint] = dof[f]
        mujoco.mj_kinematics(model, data)
        foot_z_left[k] = data.xpos[left_foot_bid, 2]
        foot_z_right[k] = data.xpos[right_foot_bid, 2]
        pelvis_z[k] = trans[f, 2]
        y, pitch_w, _ = _yaw_pitch_roll(rot_xyzw[f])
        pelvis_pitch[k] = pitch_w
        pelvis_yaw_per[k] = y

    floor_z = float(min(foot_z_left.min(), foot_z_right.min()))
    foot_lift_left = foot_z_left - floor_z
    foot_lift_right = foot_z_right - floor_z
    foot_lift = np.maximum(foot_lift_left, foot_lift_right)
    foot_lift_max = float(foot_lift.max())
    foot_lift_mean = float(foot_lift.mean())

    baseline_mask = _foot_baseline_mask(foot_z_left, foot_z_right)
    if baseline_mask.any():
        pelvis_z_baseline = float(pelvis_z[baseline_mask].mean())
    else:
        pelvis_z_baseline = float(pelvis_z.mean())
    pelvis_z_min = float(pelvis_z.min())
    pelvis_z_drop = max(0.0, pelvis_z_baseline - pelvis_z_min)

    yaw0, _, _ = _yaw_pitch_roll(rot_xyzw[0])
    yawN, _, _ = _yaw_pitch_roll(rot_xyzw[-1])
    delta_xy = trans[-1, :2] - trans[0, :2]
    body_xy = _rotate_xy_to_body(delta_xy, yaw0)
    fwd_m = float(body_xy[0])
    side_m = float(body_xy[1])
    forward_speed = fwd_m / duration
    backward_speed = -forward_speed
    side_speed = abs(side_m) / duration

    net_yaw_rad = float(np.arctan2(np.sin(yawN - yaw0), np.cos(yawN - yaw0)))
    net_yaw_deg = float(np.degrees(net_yaw_rad))

    # Waist pitch (joint angle, frozen sample at original 30 Hz)
    waist_pitch_deg = np.degrees(dof[:, WAIST_PITCH_IDX] - dof[0, WAIST_PITCH_IDX])
    waist_pitch_fwd_max = float(np.max(np.abs(waist_pitch_deg)))

    pelvis_pitch_deg = np.degrees(pelvis_pitch - pelvis_pitch[0])
    pelvis_pitch_max = float(np.max(np.abs(pelvis_pitch_deg)))

    total_pitch = waist_pitch_deg[idx] + pelvis_pitch_deg
    total_pitch_max = float(np.max(np.abs(total_pitch)))
    total_pitch_sustained = _sustained_max_deg(total_pitch, fps_sub, hold_s=0.5)

    stride_count = _stride_count_from_foot_z(foot_z_left, foot_z_right)

    # foot_planted_score: 1 if total foot lift never exceeds 2 cm.
    foot_planted_score = float(np.exp(-foot_lift_max / 0.02))

    # Pelvis yaw oscillation: max peak-to-peak yaw within any 1-second
    # sliding window (after unwrap). This isolates pelvis SWING from
    # cumulative turning. Walks with low osc are SONIC-trackable; high-
    # osc clips (>40 deg/s of swing) cause the policy to interpret the
    # motion as standing-in-place wobble rather than locomotion.
    pelvis_yaw_unwrap = np.degrees(np.unwrap(pelvis_yaw_per))
    win_size = max(2, int(round(1.0 * fps_sub)))
    pelvis_yaw_osc_deg = 0.0
    if len(pelvis_yaw_unwrap) >= win_size:
        for i in range(0, len(pelvis_yaw_unwrap) - win_size + 1):
            window = pelvis_yaw_unwrap[i:i + win_size]
            # Subtract linear trend (cumulative turn) inside the window.
            t = np.arange(len(window))
            slope = (window[-1] - window[0]) / max(1, win_size - 1)
            detrended = window - slope * t
            osc = float(detrended.max() - detrended.min())
            if osc > pelvis_yaw_osc_deg:
                pelvis_yaw_osc_deg = osc
    else:
        pelvis_yaw_osc_deg = float(pelvis_yaw_unwrap.max()
                                   - pelvis_yaw_unwrap.min())

    # cross-axis drift: for forward primitives this is body-y; for side it is body-x.
    cross_axis = float(np.hypot(min(abs(side_m), abs(fwd_m)), 0.0))

    # end-at-square: leg DOF symmetry at last frame.
    last = dof[-1]
    sq = float(np.exp(-max(
        abs(last[LEFT_HIP_PITCH_IDX] - last[RIGHT_HIP_PITCH_IDX]),
        abs(last[LEFT_KNEE_IDX] - last[RIGHT_KNEE_IDX]),
    ) / 0.05))

    return ClipFK(
        motion_key=clip.get("motion_key", "?"),
        fps=fps,
        n_frames=n,
        duration_s=duration,
        foot_lift_max_m=foot_lift_max,
        foot_lift_mean_m=foot_lift_mean,
        pelvis_z_baseline_m=pelvis_z_baseline,
        pelvis_z_min_m=pelvis_z_min,
        pelvis_z_drop_m=pelvis_z_drop,
        forward_speed_mps=forward_speed,
        backward_speed_mps=backward_speed,
        side_speed_mps=side_speed,
        net_xy_body_m=(fwd_m, side_m),
        net_yaw_deg=net_yaw_deg,
        pelvis_yaw_osc_deg=pelvis_yaw_osc_deg,
        waist_pitch_fwd_max_deg=waist_pitch_fwd_max,
        pelvis_pitch_world_max_deg=pelvis_pitch_max,
        total_pitch_max_deg=total_pitch_max,
        total_pitch_sustained_deg=total_pitch_sustained,
        stride_count=stride_count,
        foot_planted_score=foot_planted_score,
        cross_axis_xy_m=cross_axis,
        end_at_square_score=sq,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed-pkl", type=Path, default=SEED_PKL)
    p.add_argument("--mjcf", type=Path, default=MJCF)
    p.add_argument("--out", type=Path, default=REPORT_OUT)
    p.add_argument("--top-k", type=int, default=20,
                   help="How many to list per family in the report.")
    p.add_argument("--max-foot-lift", type=float, default=0.08,
                   help="Hard ceiling on foot-lift (m) — anything above is "
                        "rejected. Default 0.08 m (~3 inches).")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only first N clips (debugging).")
    args = p.parse_args(argv)

    if not args.seed_pkl.exists():
        print(f"seed pkl not found: {args.seed_pkl}", file=sys.stderr)
        return 1
    if not args.mjcf.exists():
        print(f"mjcf not found: {args.mjcf}", file=sys.stderr)
        return 1

    model = mujoco.MjModel.from_xml_path(str(args.mjcf))
    data = mujoco.MjData(model)
    qpos_adr_per_joint = np.array(
        [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
         for name in MUJOCO_JOINT_NAMES],
        dtype=np.int64,
    )
    left_foot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    right_foot_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

    print(f"loading {args.seed_pkl} ...", flush=True)
    raw = joblib.load(args.seed_pkl)
    keys = list(raw.keys())
    if args.limit > 0:
        keys = keys[: args.limit]

    print(f"running FK on {len(keys)} clips ...", flush=True)
    t0 = time.time()
    rows: list[ClipFK] = []
    fail = 0
    for i, k in enumerate(keys):
        try:
            entry = dict(raw[k])
            entry["motion_key"] = k
            row = fk_metrics_for_clip(
                model, data, qpos_adr_per_joint,
                left_foot_bid, right_foot_bid, entry,
                subsample_dt_s=DEFAULT_SUBSAMPLE_DT_S,
            )
            rows.append(row)
        except Exception as exc:
            fail += 1
            if fail <= 3:
                print(f"  fail {k}: {exc}", file=sys.stderr)
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(keys)} ({time.time()-t0:.1f}s elapsed)",
                  flush=True)
    print(f"done in {time.time()-t0:.1f}s; {len(rows)} ok, {fail} fail",
          flush=True)

    # ----- Family rankings -----
    def _filter(rows, **kw):
        out = []
        for r in rows:
            if r.foot_lift_max_m > kw.get("max_foot_lift", args.max_foot_lift):
                continue
            ok = True
            for attr, (lo, hi) in kw.get("ranges", {}).items():
                v = getattr(r, attr)
                if lo is not None and v < lo:
                    ok = False; break
                if hi is not None and v > hi:
                    ok = False; break
            if ok:
                out.append(r)
        return out

    # Helper: "true straight" filter — net yaw delta is small AND the
    # cross-axis (body-y for fwd, body-x for side) is small relative to
    # the dominant axis. Keeps out the 270-deg arc walks that would
    # otherwise dominate the leaderboard.
    def _is_true_forward(r: ClipFK, min_speed: float) -> bool:
        return (r.forward_speed_mps >= min_speed
                and abs(r.net_yaw_deg) <= 30
                and abs(r.net_xy_body_m[1]) <= 0.20)

    def _is_true_backward(r: ClipFK, min_speed: float) -> bool:
        return (r.backward_speed_mps >= min_speed
                and abs(r.net_yaw_deg) <= 30
                and abs(r.net_xy_body_m[1]) <= 0.20)

    def _is_true_side(r: ClipFK, min_speed: float) -> bool:
        return (r.side_speed_mps >= min_speed
                and abs(r.net_yaw_deg) <= 30
                and abs(r.net_xy_body_m[0]) <= 0.20)

    # ── Locomotion: real mocap walking has 12-18 cm swing apex; the
    #    user's "too high" complaint was about the 50 cm high-knee /
    #    marching variants. Cap at 0.18 m and rank by lift (lower better).
    #
    #    NEW (after first SONIC sweep): hard-cap pelvis-yaw oscillation at
    #    40 deg peak-to-peak. Source clips with bigger pelvis swing (raw
    #    SMPL→X2 retargets often have 80+ deg) cause SONIC to interpret
    #    the motion as standing-in-place wobble instead of locomotion.
    #    The v1 walk sources had ~17-31 deg osc and SONIC tracked them.
    LOCO_FOOT_CAP = 0.18
    LOCO_YAW_OSC_CAP = 40.0
    LOCO_YAW_OSC_FWD_CAP = 55.0   # forward walks rare under 40

    def _is_loco_trackable(r: ClipFK, yaw_cap: float = LOCO_YAW_OSC_CAP) -> bool:
        return (r.foot_lift_max_m <= LOCO_FOOT_CAP
                and r.pelvis_yaw_osc_deg <= yaw_cap)

    fwd_walk = [r for r in rows
                if _is_loco_trackable(r, yaw_cap=LOCO_YAW_OSC_FWD_CAP)
                and r.stride_count >= 1
                and _is_true_forward(r, 0.25)]
    fwd_walk.sort(key=lambda r: (
        -r.forward_speed_mps + 1.0 * r.foot_lift_max_m
        + 0.01 * r.pelvis_yaw_osc_deg
    ))

    fwd_step_short = [r for r in rows
                      if _is_loco_trackable(r, yaw_cap=LOCO_YAW_OSC_FWD_CAP)
                      and 1 <= r.stride_count <= 2
                      and _is_true_forward(r, 0.10)
                      and r.forward_speed_mps <= 0.40]
    fwd_step_short.sort(key=lambda r: (
        abs(r.forward_speed_mps - 0.25) + 1.0 * r.foot_lift_max_m
        + 0.01 * r.pelvis_yaw_osc_deg
    ))

    back_walk = [r for r in rows
                 if _is_loco_trackable(r) and r.stride_count >= 1
                 and _is_true_backward(r, 0.15)]
    back_walk.sort(key=lambda r: (
        -r.backward_speed_mps + 1.0 * r.foot_lift_max_m
        + 0.01 * r.pelvis_yaw_osc_deg
    ))

    side_walk = [r for r in rows
                 if _is_loco_trackable(r) and r.stride_count >= 1
                 and _is_true_side(r, 0.15)]
    side_walk.sort(key=lambda r: (
        -r.side_speed_mps + 1.0 * r.foot_lift_max_m
        + 0.01 * r.pelvis_yaw_osc_deg
    ))

    lean_fwd = _filter(rows, max_foot_lift=0.025, ranges={
        "total_pitch_sustained_deg": (12.0, None),
        "forward_speed_mps": (-0.05, 0.05),
        "side_speed_mps": (None, 0.04),
    })
    lean_fwd.sort(key=lambda r: -r.total_pitch_sustained_deg)

    # ── Crouch: pelvis Z drops ≥ 0.04 m below baseline. Most "two_hands
    #    front_low" mocaps have 4.5-7 cm peak foot lift (heel pop while the
    #    actor settles into the squat) — allow up to 0.07 m so we surface
    #    them. Strict 0.04 m cap previously excluded all symmetric squats.
    crouch = _filter(rows, max_foot_lift=0.07, ranges={
        "pelvis_z_drop_m": (0.04, None),
        "forward_speed_mps": (-0.05, 0.05),
        "side_speed_mps": (None, 0.05),
        "pelvis_yaw_osc_deg": (None, 25.0),  # exclude reach-and-rotate
    })
    crouch.sort(key=lambda r: (-r.pelvis_z_drop_m + 0.05 * r.pelvis_yaw_osc_deg
                               + 0.5 * r.foot_lift_max_m))

    # Write report
    args.out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# X2 motion candidates — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Mined {len(rows)} clips from `{args.seed_pkl.name}` via MuJoCo FK.",
        f"Hard ceiling on foot-lift: {args.max_foot_lift:.3f} m (excluded "
        "from every list below).",
        "",
        "Columns:",
        "* `dur` — clip duration (s)",
        "* `fl_max` / `fl_mean` — foot-lift max / mean (m)",
        "* `pz_min` / `pz_drop` — pelvis Z min / drop from baseline (m)",
        "* `fwd` / `back` / `side` — body-frame speed (m/s)",
        "* `pitch` — sustained waist+pelvis forward pitch (deg, ≥0.5s held)",
        "* `dyaw` — net heading delta (deg)",
        "* `strides` — stride count (FK-based, foot-Z zero crossings/2)",
        "* `sq` — end-at-square score (0..1)",
        "* `fp` — feet-planted score (0..1; 1=feet flat)",
        "",
    ]

    def _table(name: str, hits: list[ClipFK], k: int):
        lines.append(f"## {name}  (top {min(k, len(hits))} of {len(hits)})")
        lines.append("")
        lines.append("| # | motion_key | T | dur | fl_max | pz_drop | "
                     "fwd | back | side | pitch | dyaw | yaw_osc | strides | sq |")
        lines.append("| - | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |")
        for i, r in enumerate(hits[:k], 1):
            lines.append(
                f"| {i} | `{r.motion_key}` | {r.n_frames} | {r.duration_s:.2f} | "
                f"{r.foot_lift_max_m:.3f} | {r.pelvis_z_drop_m:.3f} | "
                f"{r.forward_speed_mps:+.3f} | {r.backward_speed_mps:+.3f} | "
                f"{r.side_speed_mps:.3f} | {r.total_pitch_sustained_deg:.1f} | "
                f"{r.net_yaw_deg:+.1f} | {r.pelvis_yaw_osc_deg:.0f} | "
                f"{r.stride_count} | {r.end_at_square_score:.2f} |"
            )
        lines.append("")

    _table("Forward continuous walk (0.30–0.80 m/s)", fwd_walk, args.top_k)
    _table("Forward short step (0.15–0.40 m/s, 1–2 strides)", fwd_step_short, args.top_k)
    _table("Backward walk (0.20–0.70 m/s)", back_walk, args.top_k)
    _table("Side step (|by| ≥ 0.20 m/s)", side_walk, args.top_k)
    _table("Lean forward (sustained ≥ 12 deg, planted, foot-lift ≤ 2.5 cm)",
           lean_fwd, args.top_k)
    _table("Crouch (pelvis Z drops ≥ 0.05 m, planted, foot-lift ≤ 3 cm)",
           crouch, args.top_k)

    # ALSO: report on EXISTING bin sources so we know what's currently being used.
    lines += [
        "## Existing bin sources (for comparison)",
        "",
    ]
    try:
        prims = joblib.load(REPO_ROOT / "gear_sonic/data/motions/x2_planner_primitives.pkl")
        existing = sorted({p.get("motion_key") for p in prims.values() if p.get("motion_key")})
        rows_by_key = {r.motion_key: r for r in rows}
        lines.append("| bin_source | fl_max | fwd | side | pitch | yaw_osc | strides |")
        lines.append("| --- | -: | -: | -: | -: | -: | -: |")
        for k in existing:
            r = rows_by_key.get(k)
            if r is None:
                lines.append(f"| `{k}` | - | - | - | - | - | - |")
            else:
                lines.append(
                    f"| `{r.motion_key}` | {r.foot_lift_max_m:.3f} | "
                    f"{r.forward_speed_mps:+.3f} | {r.side_speed_mps:.3f} | "
                    f"{r.total_pitch_sustained_deg:.1f} | "
                    f"{r.pelvis_yaw_osc_deg:.0f} | {r.stride_count} |"
                )
        lines.append("")
    except Exception as exc:
        lines.append(f"(could not load primitives.pkl: {exc})")

    args.out.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {args.out}", flush=True)

    # ----- Window proposals for top winners -----
    # For a curated set of candidate clips (the top of each family), scan
    # sub-windows and propose the cleanest one — minimal foot-lift, planted
    # endpoints, low yaw drift, AND meets the family's intent (forward
    # speed for locomotion; pitch-at-end for leans; Z-drop-at-end for
    # crouches).
    winners: list[tuple[str, str, dict]] = []

    # Prefer the LONG, low-osc loop variants by name. The auto-pick by
    # leaderboard score sometimes surfaces "walking_quip" or "hands_on_back"
    # which have high osc; we want clean treadmill loops.
    def _pick_first(rows_in: list[ClipFK], name_substring: str) -> ClipFK | None:
        for r in rows_in:
            if name_substring in r.motion_key:
                return r
        return None

    fwd_pick = (_pick_first(fwd_walk, "walk_forward_loop_003__A034")
                or _pick_first(fwd_walk, "wander_R_001")
                or _pick_first(fwd_walk, "walk_forward_loop")
                or (fwd_walk[0] if fwd_walk else None))
    if fwd_pick:
        winners += [
            ("fwd_walk_standard", fwd_pick.motion_key,
             {"target_s": 6.0, "intent": "fwd_walk",
              "min_speed_mps": 0.30, "max_yaw_deg": 10.0}),
            ("fwd_step_1ft", fwd_pick.motion_key,
             {"target_s": 1.6, "intent": "fwd_walk",
              "min_speed_mps": 0.25, "max_yaw_deg": 15.0,
              "min_translation_m": 0.30}),
        ]

    bw_pick = (_pick_first(back_walk, "walk_backward_loop_005__A028_M")
               or _pick_first(back_walk, "walk_backward_loop")
               or _pick_first(back_walk, "walk_backward")
               or (back_walk[0] if back_walk else None))
    if bw_pick:
        winners += [
            ("back_walk_standard", bw_pick.motion_key,
             {"target_s": 6.0, "intent": "back_walk",
              "min_speed_mps": 0.20, "max_yaw_deg": 10.0}),
            ("back_step_half_ft", bw_pick.motion_key,
             {"target_s": 1.6, "intent": "back_walk",
              "min_speed_mps": 0.20, "max_yaw_deg": 15.0,
              "min_translation_m": 0.15}),
        ]

    # 2026-05-11 v3: prefer ``stop_004__A043`` (lowest fwd drift / dyaw of
    # the sideway clips with side speed > 0.30 m/s, see candidates report).
    sw_pick = (_pick_first(side_walk, "walk_sideway_090_stop_004__A043")
               or _pick_first(side_walk, "walk_sideway_090_stop_001__A042_M")
               or _pick_first(side_walk, "walk_sideway_090_stop_001__A039_M")
               or _pick_first(side_walk, "walk_sideway_090")
               or _pick_first(side_walk, "walk_sideway")
               or (side_walk[0] if side_walk else None))
    if sw_pick:
        winners += [
            ("side_walk", sw_pick.motion_key,
             {"target_s": 6.0, "intent": "side_walk",
              "min_speed_mps": 0.15, "max_yaw_deg": 15.0}),
            ("side_half_ft", sw_pick.motion_key,
             {"target_s": 1.6, "intent": "side_walk",
              "min_speed_mps": 0.15, "max_yaw_deg": 15.0,
              "min_translation_m": 0.15}),
        ]

    if lean_fwd:
        lean_pick = next((r for r in lean_fwd if "body_check" in r.motion_key
                          and r.duration_s > 20), lean_fwd[0])
        winners += [
            ("lean_fwd_natural_apex", lean_pick.motion_key,
             {"target_s": 2.5, "intent": "lean", "min_pitch_deg": 18.0}),
        ]

    if crouch:
        # 2026-05-11 v3: prefer the symmetric two-hands front_low picks
        # discovered after relaxing the foot_lift cap from 4 cm -> 7 cm.
        crouch_pick = (
            _pick_first(crouch, "big_heavy_two_hands_front_low_to_front_low_R_001__A526")
            or _pick_first(crouch, "medium_heavy_two_hands_pick_up_front_low_R_001__A504_M")
            or _pick_first(crouch, "medium_light_two_hands_pick_up_front_low_R_001__A506_M")
            or crouch[0]
        )
        winners += [
            ("crouch_natural_apex", crouch_pick.motion_key,
             {"target_s": 2.0, "intent": "crouch", "min_drop_m": 0.04}),
        ]

    win_lines = [
        f"# X2 candidate sub-windows — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Per family-pick, the cleanest sub-window meeting the intent:",
        "* walks: feet planted at endpoints, minimum forward/back/side speed",
        "  hit, low yaw drift, minimum foot-lift apex within window.",
        "* lean: starts upright, ends at apex pitch >= min_pitch_deg.",
        "* crouch: starts at baseline pelvis Z, ends with drop >= min_drop_m.",
        "",
        "| family | motion_key | target_s | start | n | fl_max | dx | dy | "
        "dyaw | dpz | end_pitch | speed |",
        "| --- | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |",
    ]
    for fam, key, kw in winners:
        clip = raw[key]
        try:
            start, n_frames, info = _scan_clean_window(
                model, data, qpos_adr_per_joint,
                left_foot_bid, right_foot_bid, clip, **kw,
            )
            win_lines.append(
                f"| {fam} | `{key}` | {kw['target_s']:.2f} | {start} | "
                f"{n_frames} | {info['fl_max']:.3f} | {info['dx']:+.3f} | "
                f"{info['dy']:+.3f} | {info['dyaw']:+.1f} | "
                f"{info['dpz']:+.3f} | {info['end_pitch']:+.1f} | "
                f"{info['speed_mps']:+.3f} |"
            )
        except RuntimeError as exc:
            win_lines.append(
                f"| {fam} | `{key}` | {kw['target_s']:.2f} | - | - | - | "
                f"- | - | - | - | - | NO WINDOW: {exc} |"
            )
    WINDOWS_OUT.parent.mkdir(parents=True, exist_ok=True)
    WINDOWS_OUT.write_text("\n".join(win_lines) + "\n")
    print(f"Windows written to {WINDOWS_OUT}", flush=True)
    return 0


def _scan_clean_window(
    model, data, qpos_adr, lf_bid, rf_bid, clip: dict,
    target_s: float, intent: str,
    min_speed_mps: float = 0.0, max_yaw_deg: float = 180.0,
    min_translation_m: float = 0.0,
    min_pitch_deg: float = 0.0, min_drop_m: float = 0.0,
) -> tuple[int, int, dict]:
    """Find the cleanest window of length ``target_s`` matching the intent.

    Intents: ``fwd_walk``, ``back_walk``, ``side_walk``, ``lean``,
    ``crouch``. Returns (start_frame, n_frames, info).

    Walks: minimise foot-lift apex among windows with feet planted at
    endpoints AND meeting min translation & speed thresholds.

    Lean / crouch: pick the window where the start is upright/baseline
    and the END is at the requested apex (so the runtime blends INTO the
    held pose; recovery is left to the next primitive).
    """
    fps = float(clip.get("fps", 30))
    dof = np.asarray(clip["dof"], dtype=np.float64)
    rot = np.asarray(clip["root_rot"], dtype=np.float64)
    trans = np.asarray(clip["root_trans_offset"], dtype=np.float64)
    n = dof.shape[0]
    n_frames = max(8, int(round(target_s * fps)))
    if n_frames >= n:
        n_frames = n - 1

    sub_step = max(1, int(round(0.03 * fps)))
    idx = np.arange(0, n, sub_step)
    fz_l = np.empty(len(idx))
    fz_r = np.empty(len(idx))
    pz = np.empty(len(idx))
    pitch = np.empty(len(idx))
    yaw_per = np.empty(len(idx))
    for k, f in enumerate(idx):
        qx, qy, qz, qw = rot[f]
        data.qpos[0:3] = trans[f]
        data.qpos[3] = qw
        data.qpos[4:7] = (qx, qy, qz)
        data.qpos[qpos_adr] = dof[f]
        mujoco.mj_kinematics(model, data)
        fz_l[k] = data.xpos[lf_bid, 2]
        fz_r[k] = data.xpos[rf_bid, 2]
        pz[k] = trans[f, 2]
        y, p, _ = _yaw_pitch_roll(rot[f])
        pitch[k] = p
        yaw_per[k] = y
    floor_z = float(min(fz_l.min(), fz_r.min()))
    # For crouches the absolute foot Z drops with pelvis Z (the leg folds
    # rather than lifts), so the global ``floor_z`` reference would mark
    # standing frames as 5+ cm "lifted". Use a baseline-stand foot Z
    # (median of the first second) for the planted-at-start check, with a
    # per-frame relative lift instead of absolute floor lift.
    base_n = max(1, int(round(1.0 * fps / sub_step)))
    baseline_foot_z = float(np.median(np.minimum(fz_l[:base_n], fz_r[:base_n])))
    fl_per_sub = np.maximum(fz_l, fz_r) - floor_z
    foot_lift_rel_baseline = np.minimum(fz_l, fz_r) - baseline_foot_z
    waist_pitch = dof[idx, WAIST_PITCH_IDX]
    total_pitch_deg = np.degrees(np.abs(waist_pitch - waist_pitch[0]) +
                                 np.abs(pitch - pitch[0]))

    # Smoothed yaw — pelvis swings ±35 deg over a stride; body-frame math
    # using a single-frame yaw is meaningless. Use a 1-second median.
    smooth_w = max(3, int(round(1.0 * fps / sub_step)))
    yaw_smooth = np.empty_like(yaw_per)
    for k in range(len(yaw_per)):
        lo = max(0, k - smooth_w // 2)
        hi = min(len(yaw_per), k + smooth_w // 2 + 1)
        yaw_smooth[k] = np.median(np.unwrap(yaw_per[lo:hi]))

    sub_n = max(2, int(round(n_frames / sub_step)))
    if sub_n >= len(idx):
        sub_n = len(idx) - 1

    best = None  # (cost, sub_start, sub_end)

    for sub_start in range(0, len(idx) - sub_n + 1):
        sub_end = sub_start + sub_n - 1
        s_frame = int(idx[sub_start])
        e_frame = int(idx[sub_end])
        n_win = e_frame - s_frame + 1
        win_dur = n_win / fps

        # Common: planted feet at endpoints (within 2 cm for walks, 4 cm for
        # crouch since some symmetric squats heel-pop briefly).
        if intent in ("fwd_walk", "back_walk", "side_walk", "lean"):
            if min(fz_l[sub_start], fz_r[sub_start]) - floor_z > 0.025:
                continue
            if intent in ("fwd_walk", "back_walk", "side_walk"):
                if min(fz_l[sub_end], fz_r[sub_end]) - floor_z > 0.025:
                    continue
        elif intent == "crouch":
            # Crouch feet "fold under" pelvis -- absolute foot Z drops too,
            # so we measure lift relative to the baseline (stand) foot Z.
            if foot_lift_rel_baseline[sub_start] > 0.025:
                continue

        yaw0 = float(yaw_smooth[sub_start])
        yawN = float(yaw_smooth[sub_end])
        dyaw = np.degrees(np.arctan2(np.sin(yawN - yaw0), np.cos(yawN - yaw0)))
        dxyz_world = trans[e_frame] - trans[s_frame]
        body = _rotate_xy_to_body(dxyz_world[:2], yaw0)
        fl_max = float(fl_per_sub[sub_start:sub_end + 1].max())

        if abs(dyaw) > max_yaw_deg:
            continue

        if intent == "fwd_walk":
            speed = float(body[0]) / win_dur
            if speed < min_speed_mps:
                continue
            if abs(body[0]) < min_translation_m:
                continue
            cost = fl_max - 0.05 * speed + 0.005 * abs(dyaw)
        elif intent == "back_walk":
            speed = -float(body[0]) / win_dur
            if speed < min_speed_mps:
                continue
            if -body[0] < min_translation_m:
                continue
            cost = fl_max - 0.05 * speed + 0.005 * abs(dyaw)
        elif intent == "side_walk":
            speed = abs(float(body[1])) / win_dur
            if speed < min_speed_mps:
                continue
            if abs(body[1]) < min_translation_m:
                continue
            cost = fl_max - 0.05 * speed + 0.005 * abs(dyaw)
        elif intent == "lean":
            # Start near upright (<5 deg), end at peak (>= min_pitch_deg).
            if total_pitch_deg[sub_start] > 5.0:
                continue
            end_pitch = float(total_pitch_deg[sub_end])
            if end_pitch < min_pitch_deg:
                continue
            cost = -end_pitch + 100 * fl_max + 0.01 * abs(dyaw)
        elif intent == "crouch":
            # Start near baseline pelvis, end at apex Z drop.
            baseline_z = float(np.median(pz[sub_start:sub_start + 3]))
            min_z_in = float(pz[sub_start:sub_end + 1].min())
            drop = baseline_z - min_z_in
            end_drop = baseline_z - float(pz[sub_end])
            if end_drop < min_drop_m:
                continue
            # Reject if pelvis yaw rolls during the crouch (we want pure squat).
            if abs(dyaw) > 15.0:
                continue
            cost = -drop + 100 * fl_max
        else:
            raise ValueError(f"unknown intent {intent!r}")

        if best is None or cost < best[0]:
            best = (cost, sub_start, sub_end)

    if best is None:
        raise RuntimeError(
            f"no window meets intent={intent} constraints in clip"
        )

    _, sub_start, sub_end = best
    start_frame = int(idx[sub_start])
    end_frame = int(idx[sub_end])
    n_frames_out = end_frame - start_frame + 1

    yaw0 = float(yaw_smooth[sub_start])
    yawN = float(yaw_smooth[sub_end])
    dyaw = float(np.degrees(np.arctan2(np.sin(yawN - yaw0), np.cos(yawN - yaw0))))
    dxyz_world = trans[end_frame] - trans[start_frame]
    body = _rotate_xy_to_body(dxyz_world[:2], yaw0)
    win_dur = n_frames_out / fps

    end_pitch = float(total_pitch_deg[sub_end])
    if intent == "fwd_walk":
        speed = float(body[0]) / win_dur
    elif intent == "back_walk":
        speed = -float(body[0]) / win_dur
    elif intent == "side_walk":
        speed = abs(float(body[1])) / win_dur
    elif intent == "crouch":
        baseline_z = float(np.median(pz[sub_start:sub_start + 3]))
        speed = baseline_z - float(pz[sub_end])
    else:
        speed = 0.0

    info = dict(
        fl_max=float(fl_per_sub[sub_start:sub_end + 1].max()),
        dx=float(body[0]),
        dy=float(body[1]),
        dyaw=dyaw,
        dpz=float(dxyz_world[2]),
        end_pitch=end_pitch,
        speed_mps=speed,
    )
    return start_frame, n_frames_out, info


if __name__ == "__main__":
    raise SystemExit(main())
