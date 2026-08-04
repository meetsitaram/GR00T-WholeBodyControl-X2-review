#!/usr/bin/env python3
"""Maneuver sweep for the SHIPPED kplanner graph: kitchen-teleop tuning.

Drives the deployed ONNX planner backend in-process (same serve path as
pc2_kplanner_onnx / kplanner_turnrate_sweep) through scripted maneuvers:

  straight     6 s forward walk
  circle       constant (fwd, yaw) arc — wide + narrow commanded radii
  turn90       walk 3 s -> 90 deg arc turn -> walk 3 s (heading capture)
  uturn        walk 2 s -> 180 deg arc -> walk 2 s

Reports, per (fwd_mps, arc_rad_s, fwd_boost):
  R_cmd / R_fit    commanded vs achieved turn radius (circle fit on xy)
  v_path           achieved path speed on the arc (under-translation check)
  yaw_ratio        achieved / commanded heading rate on the arc
  still%           fraction of served ticks with a still pose chunk
  stepP95          p95 per-tick heading step deg (smoothness; SONIC tracks it)
  accP95           p95 root linear accel (m/s^2) at 50 Hz (abruptness)

Cadence study (--cadence): emulates PC2 inference latency by delaying
replan commits N served ticks; sweeps --replan-threshold-frames and
measures ring starvation + intent-change response latency.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "gear_sonic/scripts"))

import pc2_kplanner_onnx as k  # noqa: E402


def _yaw_of_wxyz(q):
    w, x, y, z = q
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def serve(backend, warm, schedule, seconds, latency_ticks=0):
    """Serve `seconds` of frames. schedule = [(t_start_s, target4), ...].

    latency_ticks emulates PC2 replan inference latency: a prepared replan
    only commits after that many served ticks. Intent changes force a
    replan (mirrors the deploy's replan_event).
    """
    backend.reset(warm)
    n = int(seconds * 50)
    frames, stills_mask, pending, pend_age = [], [], None, 0
    sched = sorted(schedule)
    cur_target = sched[0][1]
    forced = True
    switch_ticks = []
    for i in range(n):
        t = i / 50.0
        while len(sched) > 1 and t >= sched[1][0]:
            sched.pop(0)
            cur_target = sched[0][1]
            forced = True
            switch_ticks.append(i)
        if pending is not None:
            pend_age += 1
            if pend_age >= latency_ticks:
                pred, npf = backend.replan_infer(pending)
                hp_std = max(float(np.std(pred[:npf, 7])),
                             float(np.std(pred[:npf, 13])))
                backend.replan_commit(pred, npf)
                stills_mask.append((i, hp_std <= 0.045))
                pending, pend_age = None, 0
        if pending is None and (forced or backend.should_replan()):
            forced = False
            pending = backend.replan_prepare(cur_target)
            pend_age = 0
        frames.append(backend.get_next_frame_resampled(50.0))
    q = np.asarray(frames, dtype=np.float64)
    return q, stills_mask, switch_ticks


_FK = {"model": None, "data": None, "feet": None}


def paper_metrics(q, t0=1.0):
    """MotionBricks-paper-aligned metrics on a served [T,38] qpos stream:
    Root Jitter + Joint Jitter (m/s^2, accel magnitude of positions) and
    Foot Skate (m/frame, horizontal foot travel while in ground contact).
    FK through the X2 MJCF (free-joint root + 31 dof)."""
    import mujoco
    if _FK["model"] is None:
        mjcf = str(REPO / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml")
        _FK["model"] = mujoco.MjModel.from_xml_path(mjcf)
        _FK["data"] = mujoco.MjData(_FK["model"])
        feet = [i for i in range(_FK["model"].nbody)
                if "ankle_roll" in (mujoco.mj_id2name(
                    _FK["model"], mujoco.mjtObj.mjOBJ_BODY, i) or "")]
        _FK["feet"] = feet
    m, d = _FK["model"], _FK["data"]
    seg = q[int(t0 * 50):]
    fps = 50.0
    joints = []
    feet_pos = []
    for f in seg:
        d.qpos[:38] = f
        mujoco.mj_kinematics(m, d)
        joints.append(d.xpos[1:].copy())
        feet_pos.append(d.xpos[_FK["feet"]].copy())
    J = np.asarray(joints)          # [T, nb, 3]
    F = np.asarray(feet_pos)        # [T, nfeet, 3]
    jvel = np.diff(J, axis=0) * fps
    jacc = np.linalg.norm(np.diff(jvel, axis=0) * fps, axis=-1)
    root_acc = np.linalg.norm(
        np.diff(np.diff(q[int(t0*50):, 0:3], axis=0) * fps, axis=0) * fps,
        axis=-1)
    # foot skate: horizontal travel per frame while foot near ground
    z_thresh = F[:, :, 2].min() + 0.03
    contact = F[:, :, 2] < z_thresh
    dxy = np.linalg.norm(np.diff(F[:, :, :2], axis=0), axis=-1)
    both = contact[1:] & contact[:-1]
    skate = float(dxy[both].mean()) if both.any() else 0.0
    return {
        "joint_jitter": float(np.mean(jacc)),
        "root_jitter": float(np.mean(root_acc)),
        "foot_skate": skate,
    }


def metrics(q, t0=1.0, t1=None):
    """Geometry metrics over [t0, t1] seconds of a [T,38] qpos stream."""
    i0 = int(t0 * 50)
    i1 = int(t1 * 50) if t1 else len(q)
    seg = q[i0:i1]
    xy = seg[:, 0:2]
    yaws = np.unwrap([_yaw_of_wxyz(f[3:7]) for f in seg])
    d_xy = np.diff(xy, axis=0)
    speed = np.linalg.norm(d_xy, axis=1) * 50.0
    acc = np.abs(np.diff(speed)) * 50.0
    steps = np.degrees(np.abs(np.diff(yaws)))
    out = {
        "yaw_rate_dps": float(np.degrees(yaws[-1] - yaws[0])
                              / max(1e-6, (i1 - i0) / 50.0)),
        "v_path": float(np.mean(speed)),
        "stepP95": float(np.percentile(steps, 95)),
        "accP95": float(np.percentile(acc, 95)),
    }
    # circle fit (algebraic, Kasa) for radius
    A = np.c_[2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))]
    b = (xy ** 2).sum(axis=1)
    try:
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        out["R_fit"] = float(np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2))
    except np.linalg.LinAlgError:
        out["R_fit"] = float("nan")
    return out


def still_frac(stills_mask):
    if not stills_mask:
        return 0.0
    return sum(1 for _, s in stills_mask if s) / len(stills_mask)


def boot(onnx_dir: Path, threshold: int):
    onnx = onnx_dir / "x2_kplanner_template.onnx"
    sidecars = sorted(onnx_dir.glob("*.json"))
    contract = k._load_onnx_contract(onnx, sidecars[0] if sidecars else None)
    backend = k.OnnxPlannerBackend(
        onnx_path=onnx, contract=contract,
        replan_threshold_frames=threshold, planner_mode="slow_walk")
    warm_path = REPO / "gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl"
    warm = k._load_warmup_qpos(warm_path if warm_path.exists() else None)
    return backend, warm, float(warm[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", type=Path,
                    default=Path.home() / ".cache/sonic/x2/kplanner_onnx")
    ap.add_argument("--fwd", default="0.3,0.4,0.5")
    ap.add_argument("--arc", default="0.4,0.55,0.7,0.9")
    ap.add_argument("--boost", default="1.0,1.3")
    ap.add_argument("--threshold", type=int, default=32)
    ap.add_argument("--cadence", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("/tmp/kplanner_maneuver_results.json"))
    args = ap.parse_args()

    backend, warm, hip = boot(args.onnx_dir, args.threshold)
    results = []

    if args.cadence:
        # threshold x latency: starvation + response latency to a turn command
        for thr in (16, 24, 32, 48):
            backend.REPLAN_THRESHOLD_FRAMES = thr
            for lat_s in (0.05, 0.45, 0.6):
                lt = int(lat_s * 50)
                sched = [(0.0, (0.0, 0.0, 0.3, hip)),
                         (4.0, (0.55, 0.0, 0.3, hip))]
                q, sm, sw = serve(backend, warm, sched, 8.0, latency_ticks=lt)
                yaws = np.unwrap([_yaw_of_wxyz(f[3:7]) for f in q])
                rate = np.degrees(np.abs(np.diff(yaws))) * 50.0
                sw_i = sw[0] if sw else 200
                # response: ticks from switch until heading rate exceeds
                # 50% of commanded for 10 consecutive ticks
                thr_dps = 0.5 * np.degrees(0.55)
                resp = None
                run = 0
                for j in range(sw_i, len(rate)):
                    run = run + 1 if rate[j] >= thr_dps else 0
                    if run >= 10:
                        resp = (j - 9 - sw_i) / 50.0
                        break
                # starvation: served frame repeats (zero root+joint delta)
                rep = (np.abs(np.diff(q, axis=0)).max(axis=1) < 1e-9).sum()
                results.append({
                    "kind": "cadence", "threshold": thr, "latency_s": lat_s,
                    "resp_s": resp, "repeat_frames": int(rep),
                    "still": still_frac(sm)})
                print(f"thr={thr:3d} lat={lat_s:.2f}s resp="
                      f"{resp if resp is not None else 'NONE'}s "
                      f"repeats={rep} still={still_frac(sm)*100:.0f}%")
    else:
        fwds = [float(x) for x in args.fwd.split(",")]
        arcs = [float(x) for x in args.arc.split(",")]
        boosts = [float(x) for x in args.boost.split(",")]
        hdr = (f'{"fwd":>4s} {"arc":>5s} {"boost":>5s} | {"R_cmd":>5s} '
               f'{"R_fit":>5s} {"v_path":>6s} {"yawX":>5s} {"still%":>6s} '
               f'{"stpP95":>6s} {"accP95":>6s}')
        print("=== circle (10 s constant arc) ===")
        print(hdr)
        for v, w, b in itertools.product(fwds, arcs, boosts):
            v_cmd = v * b
            sched = [(0.0, (w, 0.0, v_cmd, hip))]
            q, sm, _ = serve(backend, warm, sched, 12.0)
            m = metrics(q, t0=2.0)
            m.update(paper_metrics(q, t0=2.0))
            r_cmd = v / w
            row = {"kind": "circle", "fwd": v, "arc": w, "boost": b,
                   "R_cmd": r_cmd, **m, "still": still_frac(sm)}
            results.append(row)
            print(f"{v:4.1f} {w:5.2f} {b:5.1f} | {r_cmd:5.2f} "
                  f"{m['R_fit']:5.2f} {m['v_path']:6.2f} "
                  f"{m['yaw_rate_dps']/np.degrees(w):5.2f} "
                  f"{still_frac(sm)*100:5.0f}% "
                  f"{m['stepP95']:6.2f} {m['accP95']:6.2f} "
                  f"| jJit {m['joint_jitter']:5.2f} rJit {m['root_jitter']:5.2f} "
                  f"skate {m['foot_skate']*1000:5.2f}mm")
        print("=== turn90 (walk 3s -> arc to 90 -> walk 3s) ===")
        for v, w, b in itertools.product(fwds, arcs, boosts):
            t_turn = (np.pi / 2) / w
            sched = [(0.0, (0.0, 0.0, v, hip)),
                     (3.0, (w, 0.0, v * b, hip)),
                     (3.0 + t_turn, (0.0, 0.0, v, hip))]
            total = 6.0 + t_turn
            q, sm, _ = serve(backend, warm, sched, total)
            yaw0 = _yaw_of_wxyz(q[int(2.5 * 50)][3:7])
            yaw1 = _yaw_of_wxyz(q[-1][3:7])
            turned = np.degrees(abs(np.unwrap([yaw0, yaw1])[1] - yaw0))
            m = metrics(q, t0=2.0)
            row = {"kind": "turn90", "fwd": v, "arc": w, "boost": b,
                   "turned_deg": float(turned), **m,
                   "still": still_frac(sm)}
            results.append(row)
            print(f"{v:4.1f} {w:5.2f} {b:5.1f} | turned {turned:6.1f}° "
                  f"(want 90) v_path={m['v_path']:.2f} "
                  f"still={still_frac(sm)*100:.0f}% acc={m['accP95']:.2f}")
    args.out.write_text(json.dumps(results, indent=1))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
