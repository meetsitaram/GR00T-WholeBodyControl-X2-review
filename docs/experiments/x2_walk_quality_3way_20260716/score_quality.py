#!/usr/bin/env python3
"""Walk-quality metrics from im_eval IM_EVAL_DUMP_TRAJ trajectory dumps.

Per clip: root-relative MPJPE (all/upper/lower), root net-displacement error,
heading error, XY path-length ratio (executed/ref), final-yaw error
(from the L/R shoulder or hip cross-body vector).
"""
import json, math, os, pickle, sys
import numpy as np

SCRATCH = os.path.dirname(os.path.abspath(__file__))

CATS = ["slow_walk", "slow_walk_turns", "slow_walk_back", "walk", "run"]


def cat_of(name):
    if name.startswith("slow_walk_turns"): return "slow_walk_turns"
    if name.startswith("slow_walk_back"): return "slow_walk_back"
    if name.startswith("slow_walk"): return "slow_walk"
    if name.startswith("run"): return "run"
    if name.startswith("walk"): return "walk"
    return "other"


def find_idx(names, *subs, exclude=()):
    for i, n in enumerate(names):
        ln = n.lower()
        if all(s in ln for s in subs) and not any(e in ln for e in exclude):
            return i
    return None


def classify_bodies(names):
    upper, lower = [], []
    for i, n in enumerate(names):
        ln = n.lower()
        if any(s in ln for s in ("shoulder", "elbow", "wrist", "torso", "hand", "arm")):
            upper.append(i)
        elif any(s in ln for s in ("hip", "knee", "ankle", "foot", "leg")):
            lower.append(i)
    return upper, lower


def yaw_of_pair(pos, li, ri):
    """Heading yaw (rad) from the left-right cross-body vector at each frame.
    Facing direction = left-to-right vector rotated -90 deg about z:
    if v = left - right (XY), facing = (v_y, -v_x) rotated... use atan2(-v_x? )
    Convention-free: we only take DIFFERENCES pred-gt so any fixed convention ok.
    """
    v = pos[:, li, :2] - pos[:, ri, :2]
    return np.arctan2(v[:, 1], v[:, 0])


def ang_diff(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def path_len(root_xy, step=10):
    p = root_xy[::step]
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def clip_metrics(pred, gt, names, root_i, li, ri, upper, lower):
    T = min(len(pred), len(gt))
    pred, gt = pred[:T], gt[:T]
    pr = pred[:, root_i, :]
    gr = gt[:, root_i, :]
    # root-relative MPJPE
    rel_p = pred - pr[:, None, :]
    rel_g = gt - gr[:, None, :]
    err = np.linalg.norm(rel_p - rel_g, axis=-1)  # (T, n_bodies)
    m = {}
    m["mpjpe_rr_mm"] = float(err.mean() * 1000)
    m["mpjpe_rr_upper_mm"] = float(err[:, upper].mean() * 1000) if upper else float("nan")
    m["mpjpe_rr_lower_mm"] = float(err[:, lower].mean() * 1000) if lower else float("nan")
    # global root drift at end + net displacement error
    net_p = pr[-1, :2] - pr[0, :2]
    net_g = gr[-1, :2] - gr[0, :2]
    m["ref_travel_m"] = float(np.linalg.norm(net_g))
    m["net_disp_err_m"] = float(np.linalg.norm(net_p - net_g))
    # heading error (direction of net displacement), gated on ref travel
    if np.linalg.norm(net_g) > 0.3 and np.linalg.norm(net_p) > 0.05:
        h = math.degrees(ang_diff(math.atan2(net_p[1], net_p[0]),
                                  math.atan2(net_g[1], net_g[0])))
        m["heading_err_deg"] = float(abs(h))
    else:
        m["heading_err_deg"] = float("nan")
    # path-length ratio (0.2 s decimation to suppress jitter)
    pl_p = path_len(pr[:, :2])
    pl_g = path_len(gr[:, :2])
    m["path_ratio"] = float(pl_p / pl_g) if pl_g > 0.2 else float("nan")
    # final-yaw error via cross-body vector
    if li is not None and ri is not None:
        yp = yaw_of_pair(pred, li, ri)
        yg = yaw_of_pair(gt, li, ri)
        m["final_yaw_err_deg"] = float(abs(math.degrees(ang_diff(yp[-1], yg[-1]))))
        m["mean_yaw_err_deg"] = float(np.abs(np.degrees(ang_diff(yp, yg))).mean())
    else:
        m["final_yaw_err_deg"] = float("nan")
        m["mean_yaw_err_deg"] = float("nan")
    # global root position error mean (drift along the clip)
    m["root_gpe_m"] = float(np.linalg.norm(pr[:, :2] - gr[:, :2], axis=1).mean())
    return m


def load_sweep(traj_dir, metrics_json, ref_pkl=None):
    files = sorted(f for f in os.listdir(traj_dir) if f.startswith("traj_rank"))
    assert files, f"no traj dumps in {traj_dir}"
    d = pickle.load(open(os.path.join(traj_dir, files[0]), "rb"))
    keys = d["motion_keys"]
    names = d["body_names"]
    # NOTE: manager_env_wrapper.get_env_data() maps "ref_body_pos_extend" ->
    # motion_command.robot_body_pos_w (the ROBOT) and "rigid_body_pos_extend" ->
    # motion_command.body_pos_w (the REFERENCE), i.e. the dump's field names are
    # swapped: dump["pred_pos"] = reference, dump["gt_pos"] = executed robot.
    preds, gts = d["gt_pos"], d["pred_pos"]  # pred=executed robot, gt=reference
    n = min(len(keys), len(preds))
    keys, preds, gts = keys[:n], preds[:n], gts[:n]
    # sanity: dump reference net-travel should match the motion-lib pkl
    if ref_pkl and os.path.exists(ref_pkl):
        import joblib
        ref = joblib.load(ref_pkl)
        diffs = []
        for k, g in zip(keys, gts):
            if k in ref:
                r = np.asarray(ref[k]["root_trans_offset"])
                g = np.asarray(g)
                diffs.append(abs(np.linalg.norm(g[-1, 0, :2] - g[0, 0, :2])
                                 - np.linalg.norm(r[-1, :2] - r[0, :2])))
        print(f"  [sanity] mean |dump-ref-net - pkl-ref-net| = {np.mean(diffs):.4f} m "
              f"(max {np.max(diffs):.4f})", file=sys.stderr)
    term = {}
    mj = json.load(open(metrics_json))
    md = mj["eval/all_metrics_dict"]
    for k, t in zip(md["motion_keys"], md["terminated"]):
        term[k] = bool(t)
    root_i = find_idx(names, "pelvis")
    if root_i is None:
        root_i = find_idx(names, "torso") or 0
    li = find_idx(names, "left", "shoulder")
    ri = find_idx(names, "right", "shoulder")
    if li is None or ri is None:
        li = find_idx(names, "left", "hip"); ri = find_idx(names, "right", "hip")
    upper, lower = classify_bodies(names)
    out = {}
    for k, p, g in zip(keys, preds, gts):
        p = np.asarray(p, np.float64); g = np.asarray(g, np.float64)
        m = clip_metrics(p, g, names, root_i, li, ri, upper, lower)
        m["terminated"] = term.get(k, False)
        out[k] = m
    return out, names, root_i, (li, ri)


def agg(rows, key):
    v = [r[key] for r in rows if not math.isnan(r.get(key, float("nan")))]
    return (sum(v) / len(v)) if v else float("nan")


def main():
    REPO = "/home/stickbot/Projects/GR00T-WholeBodyControl"
    g1_pkl = os.path.join(REPO, "gear_sonic/data/motions/g1_teleop_corpus_50fps.pkl")
    x2_pkl = os.path.join(REPO, "gear_sonic/data/motions/x2_g1teleop_50fps.pkl")
    # (name, [(traj_dir, metrics_json), ...], ref_pkl); later pairs OVERRIDE
    # earlier ones per clip (run-only 4-env passes override the big-batch runs).
    sweeps = [
        ("G1-stock", [(os.path.join(SCRATCH, "g1_traj"), os.path.join(SCRATCH, "g1_stock_rec", "metrics_eval.json")),
                      (os.path.join(SCRATCH, "g1_runs_traj"), os.path.join(SCRATCH, "g1_runs_rec", "metrics_eval.json"))], g1_pkl),
        ("base-3k", [(os.path.join(SCRATCH, "base3k_traj"), os.path.join(SCRATCH, "base_3k_rec", "metrics_eval.json")),
                     (os.path.join(SCRATCH, "base3k_runs_traj"), os.path.join(SCRATCH, "base3k_runs_rec", "metrics_eval.json"))], x2_pkl),
        ("FT-1144", [(os.path.join(SCRATCH, "ft1144_traj"), os.path.join(SCRATCH, "ft_1144_rec", "metrics_eval.json")),
                     (os.path.join(SCRATCH, "ft1144_runs_traj"), os.path.join(SCRATCH, "ft1144_runs_rec", "metrics_eval.json"))], x2_pkl),
    ]
    res = {}
    for name, parts, rpkl in sweeps:
        merged = {}
        for tdir, mfile in parts:
            if not os.path.isdir(tdir) or not os.path.exists(mfile):
                print(f"[warn] {name}: part missing ({tdir})", file=sys.stderr)
                continue
            print(f"[{name}] {os.path.basename(tdir)}", file=sys.stderr)
            part_res, names, root_i, pair = load_sweep(tdir, mfile, rpkl)
            merged.update(part_res)
        if merged:
            res[name] = merged
            print(f"[{name}] root={names[root_i]} yaw-pair={names[pair[0]]}/{names[pair[1]]} n={len(merged)}", file=sys.stderr)

    with open(os.path.join(SCRATCH, "walk_quality_perclip.json"), "w") as f:
        json.dump(res, f, indent=1)

    METS = [("mpjpe_rr_mm", "root-rel MPJPE (mm)", "{:.1f}"),
            ("mpjpe_rr_upper_mm", "  upper-body (mm)", "{:.1f}"),
            ("mpjpe_rr_lower_mm", "  lower-body (mm)", "{:.1f}"),
            ("net_disp_err_m", "net-displacement err (m)", "{:.3f}"),
            ("root_gpe_m", "mean root pos err (m)", "{:.3f}"),
            ("path_ratio", "path-length ratio (exe/ref)", "{:.3f}"),
            ("heading_err_deg", "heading err (deg)", "{:.1f}"),
            ("final_yaw_err_deg", "final-yaw err (deg)", "{:.1f}"),
            ("mean_yaw_err_deg", "mean yaw err (deg)", "{:.1f}")]

    lines = ["# Walk quality — 61 teleop clips, executed vs reference (im_eval traj dumps)",
             "",
             "Terminated clips excluded from means (run_004 in all sweeps).", ""]
    order = [s[0] for s in sweeps if s[0] in res]
    for cat in CATS + ["ALL"]:
        lines.append(f"## {cat}")
        lines.append("| metric | " + " | ".join(order) + " |")
        lines.append("|---|" + "---|" * len(order))
        for mk, label, fmt in METS:
            row = [label]
            for n in order:
                rows = [r for k, r in res[n].items()
                        if (cat == "ALL" or cat_of(k) == cat) and not r["terminated"]]
                v = agg(rows, mk)
                row.append(fmt.format(v) if not math.isnan(v) else "—")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    # worst offenders per sweep
    lines.append("## Worst offenders (per sweep, feasible clips)")
    for n in order:
        lines.append(f"### {n}")
        rows = sorted(((k, r) for k, r in res[n].items() if not r["terminated"]),
                      key=lambda kv: -(kv[1]["mpjpe_rr_mm"]))
        lines.append("worst root-rel MPJPE: " + ", ".join(f"{k} {r['mpjpe_rr_mm']:.0f}mm" for k, r in rows[:5]))
        rows2 = sorted(((k, r) for k, r in res[n].items() if not r["terminated"] and not math.isnan(r["path_ratio"])),
                       key=lambda kv: abs(math.log(max(kv[1]["path_ratio"], 1e-3))))[::-1]
        lines.append("worst path-ratio: " + ", ".join(f"{k} {r['path_ratio']:.2f}" for k, r in rows2[:5]))
        rows3 = sorted(((k, r) for k, r in res[n].items() if not r["terminated"] and not math.isnan(r["heading_err_deg"])),
                       key=lambda kv: -kv[1]["heading_err_deg"])
        lines.append("worst heading err: " + ", ".join(f"{k} {r['heading_err_deg']:.0f}deg" for k, r in rows3[:5]))
        rows4 = sorted(((k, r) for k, r in res[n].items() if not r["terminated"]),
                       key=lambda kv: -kv[1]["net_disp_err_m"])
        lines.append("worst net-disp err: " + ", ".join(f"{k} {r['net_disp_err_m']:.2f}m" for k, r in rows4[:5]))
        lines.append("")

    out = "\n".join(lines)
    with open(os.path.join(SCRATCH, "WALK_QUALITY.md"), "w") as f:
        f.write(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
