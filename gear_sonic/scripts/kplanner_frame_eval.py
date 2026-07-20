#!/usr/bin/env python3
"""Per-FRAME evaluation of planner output. Answers "which frame went wrong".

WHY
---
The repo has ~23 evaluation tools and every one of them reduces to a scalar at
the point of computation. ``motion_deviation.py`` and ``score_quality.py`` both
build full (T, B) per-frame error tensors and then ``.mean()`` them away.
``kplanner_gait_metrics.py`` computes a per-frame foot-contact mask and a
per-frame planar foot speed -- then exports only the mean over contact frames.

That reduction is why the 30->50 Hz bug survived as an unexplained "model
problem" for weeks: a reference sweeping 1.67x too fast is obvious in a
per-frame velocity trace and invisible in a per-clip average.

So this tool does not invent metrics. It keeps the ones we already had, at
frame resolution, and adds the one class we never implemented at all: joint
velocity/jerk against the limits that already live in
``gear_sonic/utils/mujoco_sim/configs.py`` (arm 25.0, lower body 20.0 rad/s)
but which no evaluator has ever checked a candidate reference against.

CRITERIA AND WHERE THEY COME FROM
  foot slide   -- planar foot speed while in contact (skating). The old
                  hand-trimmed relaxed-walk clips were selected on exactly this
                  measure: the shipped forward clip passed at 0.090/0.021 m of
                  slide. That is a real, previously-cleared bar, not an
                  invented threshold.
  joint vel    -- vs configs.py limits. Catches "reference outside the
                  tracker's bandwidth", the documented root cause of the
                  handoff regression.
  jerk         -- third difference. Seam discontinuities at replan boundaries
                  show up here before they show up in position.
  foot z       -- ground penetration / float, per frame per foot.
  yaw rate     -- per frame. A limit cycle is a frame-level signature that any
                  aggregate hides completely.

Everything is exported as a (T,)-indexed series so two runs can be diffed
frame by frame, which is the point.

    python gear_sonic/scripts/kplanner_frame_eval.py --npz run.npz
    python gear_sonic/scripts/kplanner_frame_eval.py --npz a.npz --baseline b.npz
    python gear_sonic/scripts/kplanner_frame_eval.py --npz a.npz --save-series s.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
ARM_TOKENS = ("shoulder", "elbow", "wrist")
ARM_VEL_LIMIT = 25.0          # configs.py:190
LOWER_VEL_LIMIT = 20.0        # configs.py:196
# The bar the old relaxed-walk clips actually cleared (see playlist YAML).
CONTACT_Z_M = 0.02        # foot counts as planted within 2 cm of the
                          # clip's own registered floor
# Calibration bar, MEASURED not invented: the loosest of the four
# hand-trimmed relaxed-walk clips that drove the real robot reliably.
SLIDE_REF_M_S = 0.12


def _mjcf_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    return str(REPO / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml")


def _fk_feet(qpos: np.ndarray, model, data) -> np.ndarray:
    """(T, 2, 3) world foot positions. Same FK as kplanner_gait_metrics."""
    bids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
            for b in FOOT_BODIES]
    if any(b < 0 for b in bids):
        raise RuntimeError(f"foot bodies {FOOT_BODIES} not in model")
    T = qpos.shape[0]
    nq = min(model.nq, qpos.shape[1])
    out = np.zeros((T, 2, 3))
    for t in range(T):
        data.qpos[:nq] = qpos[t, :nq]
        mujoco.mj_kinematics(model, data)
        for i, b in enumerate(bids):
            out[t, i] = data.xpos[b]
    return out


def _dof_is_arm(model, ndof: int) -> np.ndarray:
    """Classify DOFs by JOINT NAME rather than hardcoded index ranges -- the
    index layout has bitten this repo before (gather vs scatter vs verbatim)."""
    is_arm = np.zeros(ndof, dtype=bool)
    for j in range(model.njnt):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        adr = model.jnt_qposadr[j] - 7          # free joint occupies qpos[0:7]
        if 0 <= adr < ndof and any(tok in nm for tok in ARM_TOKENS):
            is_arm[adr] = True
    return is_arm


def _wxyz_yaw(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def frame_series(qpos: np.ndarray, mjcf: str, fps: float) -> dict:
    """Every metric as a (T,)- or (T,D)-indexed array. Nothing reduced."""
    model = mujoco.MjModel.from_xml_path(mjcf)
    data = mujoco.MjData(model)
    T = qpos.shape[0]
    dof = qpos[:, 7:]
    ndof = dof.shape[1]
    feet = _fk_feet(qpos, model, data)
    is_arm = _dof_is_arm(model, ndof)

    # --- foot contact + slide, per frame per foot ------------------------
    # GROUND REGISTRATION, then an ABSOLUTE threshold.
    #
    # kplanner_gait_metrics.py:74 uses an adaptive threshold (p20 of foot z
    # + 2 cm). That silently assumes the clip's feet actually reach the floor.
    # They often do not: walk_circle_001 -- a clip that walked the real robot
    # cleanly -- floats 6.2 cm at its lowest, so the adaptive threshold landed
    # mid-swing and scored SWING SPEED as foot slide, ranking our best clip
    # worst (0.372 vs 0.115 m/s). Registering to the clip's own lowest foot
    # point and thresholding at a fixed 2 cm makes the measure comparable
    # across clips with different root-height offsets.
    # PER-FOOT registration. A single global floor is not enough: the
    # ankle_roll_link body origin sits at a different height above the sole on
    # X2 vs G1 (registered median foot z 0.010 vs 0.026 m), so any ABSOLUTE
    # threshold calibrated on one robot reports nonsense on the other -- G1
    # scored "both feet airborne 87.5% of frames", which no gait can be.
    # Referencing each foot to its own lowest achieved point is invariant to
    # both whole-clip float and cross-robot link geometry.
    for f in range(2):
        feet[:, f, 2] -= float(feet[:, f, 2].min())
    contact = np.zeros((T, 2), dtype=bool)
    slide = np.zeros((T, 2))
    for f in range(2):
        z = feet[:, f, 2]
        contact[:, f] = z < CONTACT_Z_M
        v = np.zeros(T)
        v[:-1] = np.linalg.norm(np.diff(feet[:, f, :2], axis=0), axis=1) * fps
        v[-1] = v[-2] if T > 1 else 0.0
        slide[:, f] = np.where(contact[:, f], v, 0.0)

    # --- joint velocity / jerk vs the limits already in configs.py -------
    jvel = np.zeros((T, ndof))
    jvel[:-1] = np.diff(dof, axis=0) * fps
    if T > 1:
        jvel[-1] = jvel[-2]
    jacc = np.zeros((T, ndof))
    jacc[:-1] = np.diff(jvel, axis=0) * fps
    jerk = np.zeros((T, ndof))
    jerk[:-1] = np.diff(jacc, axis=0) * fps

    limit = np.where(is_arm, ARM_VEL_LIMIT, LOWER_VEL_LIMIT)
    vel_ratio = np.abs(jvel) / limit                    # 1.0 == at the limit
    vel_worst = vel_ratio.max(axis=1)
    vel_worst_dof = vel_ratio.argmax(axis=1)

    # --- root / yaw ------------------------------------------------------
    yaw = _wxyz_yaw(qpos[:, 3:7])
    yaw_u = np.unwrap(yaw)
    yaw_rate = np.zeros(T)
    yaw_rate[:-1] = np.diff(yaw_u) * fps
    if T > 1:
        yaw_rate[-1] = yaw_rate[-2]
    root_xy = qpos[:, :2]
    root_speed = np.zeros(T)
    root_speed[:-1] = np.linalg.norm(np.diff(root_xy, axis=0), axis=1) * fps
    if T > 1:
        root_speed[-1] = root_speed[-2]

    # per-frame joint step (the "click" precursor: what kp turns into a slam)
    jump = np.zeros(T)
    jump[:-1] = np.abs(np.diff(dof, axis=0)).max(axis=1)

    return {
        "T": T, "fps": fps,
        "contact": contact, "slide": slide, "foot_z": feet[:, :, 2],
        "jvel": jvel, "jerk": jerk, "vel_ratio": vel_ratio,
        "vel_worst": vel_worst, "vel_worst_dof": vel_worst_dof,
        "yaw": yaw_u, "yaw_rate": yaw_rate,
        "root_speed": root_speed, "jump": jump,
        "is_arm": is_arm,
    }


def summarize(s: dict, label: str) -> dict:
    T, fps = s["T"], s["fps"]
    slide_c = s["slide"][s["contact"]]
    yr = s["yaw_rate"]
    rev = int((np.diff(np.sign(yr)) != 0).sum())
    m = {
        "label": label, "frames": T, "dur_s": T / fps,
        "slide_mean": float(slide_c.mean()) if slide_c.size else 0.0,
        "slide_p95": float(np.percentile(slide_c, 95)) if slide_c.size else 0.0,
        "slide_max": float(slide_c.max()) if slide_c.size else 0.0,
        "slide_over_ref_pct": float(100.0 * (slide_c > SLIDE_REF_M_S).mean())
                              if slide_c.size else 0.0,
        "vel_ratio_p95": float(np.percentile(s["vel_worst"], 95)),
        "vel_ratio_max": float(s["vel_worst"].max()),
        "vel_over_limit_pct": float(100.0 * (s["vel_worst"] > 1.0).mean()),
        "jerk_p95": float(np.percentile(np.abs(s["jerk"]).max(axis=1), 95)),
        "jump_max": float(s["jump"].max()),
        "yaw_reversals_per_s": rev / max(1e-6, T / fps),
        "yaw_rate_p95": float(np.percentile(np.abs(yr), 95)),
        "root_speed_mean": float(s["root_speed"].mean()),
        "foot_z_min": float(s["foot_z"].min()),
        "both_feet_air_pct": float(100.0 * (~s["contact"]).all(axis=1).mean()),
    }
    return m


def show(m: dict, s: dict) -> None:
    print(f"\n=== {m['label']} ===")
    print(f"  {m['frames']} frames @ {s['fps']:g} fps = {m['dur_s']:.1f}s")
    print(f"  foot slide  : mean {m['slide_mean']:.3f}  p95 {m['slide_p95']:.3f}"
          f"  max {m['slide_max']:.3f} m/s"
          f"   ({m['slide_over_ref_pct']:.1f}% of contact frames > "
          f"{SLIDE_REF_M_S:.2f} ref bar)")
    print(f"  joint vel   : p95 {m['vel_ratio_p95']:.2f}x limit  "
          f"max {m['vel_ratio_max']:.2f}x   "
          f"({m['vel_over_limit_pct']:.2f}% of frames OVER limit)")
    print(f"  jerk p95    : {m['jerk_p95']:.1f} rad/s^3    "
          f"max per-frame joint step {m['jump_max']:.3f} rad")
    print(f"  yaw         : {m['yaw_reversals_per_s']:.2f} reversals/s   "
          f"p95 rate {np.degrees(m['yaw_rate_p95']):.1f} deg/s")
    print(f"  root speed  : {m['root_speed_mean']:.3f} m/s mean")
    print(f"  feet        : min foot z {m['foot_z_min']:+.3f} m   "
          f"both-airborne {m['both_feet_air_pct']:.1f}% of frames")

    # worst frames -- the actual point of the tool
    k = min(5, s["T"])
    bad = np.argsort(-s["vel_worst"])[:k]
    print(f"  worst frames by joint velocity:")
    for t in sorted(bad):
        print(f"    f{t:<5d} t={t/s['fps']:6.2f}s  "
              f"vel {s['vel_worst'][t]:.2f}x (dof {s['vel_worst_dof'][t]})  "
              f"slide {s['slide'][t].max():.3f}  jump {s['jump'][t]:.3f}")


def load_qpos(p: Path, key: str | None) -> tuple[np.ndarray, float]:
    """NPZ (planner output) or PKL (motion-lib clip) -> qpos (T, 7+ndof).

    PKL support matters for calibration: the hand-trimmed relaxed-walk clips
    are the only motion we KNOW walked the real robot reliably, so they are
    what defines a passing score. Thresholds should be measured, not invented.
    """
    if p.suffix == ".pkl":
        import joblib
        lib = joblib.load(p)
        if key and key in lib:
            clip = lib[key]
        else:
            if key:
                raise KeyError(f"{key!r} not in {p}; keys={list(lib)[:5]}")
            clip = lib[next(iter(lib))]
        dof = np.asarray(clip["dof"], dtype=np.float64)
        rot = np.asarray(clip["root_rot"], dtype=np.float64)      # xyzw
        tr = np.asarray(clip["root_trans_offset"], dtype=np.float64)
        quat_wxyz = rot[:, [3, 0, 1, 2]]                          # -> wxyz
        return (np.concatenate([tr, quat_wxyz, dof], axis=1),
                float(clip.get("fps", 30.0)))
    d = np.load(p, allow_pickle=True)
    if key:
        return np.asarray(d[key], dtype=np.float64), float(d.get("fps", 30.0))
    for k in ("pred", "qpos", "qpos_traj", "actual"):
        if k in d:
            a = np.asarray(d[k], dtype=np.float64)
            # run_scripted_demo saves (n_trials, T, 38); squeeze a single trial
            # rather than silently scoring trial 0 of many.
            while a.ndim > 2:
                if a.shape[0] != 1:
                    raise ValueError(f"{k} has {a.shape[0]} trials; pass --key "
                                     f"or split them first (shape {a.shape})")
                a = a[0]
            return a, float(d.get("fps", 30.0))
    raise KeyError(f"no qpos-like array in {p}; keys={list(d.keys())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, type=Path)
    ap.add_argument("--key", default=None, help="array name (default: pred/qpos/actual)")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="second run to diff against, frame by frame")
    ap.add_argument("--baseline-key", default=None)
    ap.add_argument("--mjcf", default=None)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--save-series", type=Path, default=None,
                    help="write all per-frame arrays to NPZ")
    args = ap.parse_args()

    mjcf = _mjcf_path(args.mjcf)
    qpos, fps = load_qpos(args.npz, args.key)
    fps = args.fps or fps
    s = frame_series(qpos, mjcf, fps)
    m = summarize(s, args.npz.stem)
    show(m, s)

    if args.save_series:
        np.savez_compressed(args.save_series,
                            **{k: v for k, v in s.items()
                               if isinstance(v, np.ndarray)})
        print(f"\n  per-frame series -> {args.save_series}")

    if args.baseline:
        qb, fb = load_qpos(args.baseline, args.baseline_key)
        sb = frame_series(qb, mjcf, args.fps or fb)
        mb = summarize(sb, args.baseline.stem)
        show(mb, sb)
        print("\n=== side by side (lower is better on all rows) ===")
        rows = [("slide_mean", "foot slide mean (m/s)"),
                ("slide_p95", "foot slide p95 (m/s)"),
                ("vel_ratio_p95", "joint vel p95 (x limit)"),
                ("vel_over_limit_pct", "% frames over vel limit"),
                ("jerk_p95", "jerk p95"),
                ("jump_max", "max joint step (rad)"),
                ("yaw_reversals_per_s", "yaw reversals/s")]
        w = 22
        print(f"  {'metric':<26}{m['label'][:w]:>{w}}{mb['label'][:w]:>{w}}")
        for k, lbl in rows:
            print(f"  {lbl:<26}{m[k]:>{w}.3f}{mb[k]:>{w}.3f}")
        print(f"  {'root speed mean (m/s)':<26}"
              f"{m['root_speed_mean']:>{w}.3f}{mb['root_speed_mean']:>{w}.3f}"
              f"   (higher = travels more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
