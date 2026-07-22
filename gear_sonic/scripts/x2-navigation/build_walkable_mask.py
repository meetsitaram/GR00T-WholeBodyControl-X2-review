#!/usr/bin/env python3
"""Build the walkable-occupancy grid + ESDF from the kitchen collision USD.

Outputs (into --out-dir, default x2-kitchen-sim/assets):
  walkable_mask.png   free-space grid (255 = walkable), robot-radius eroded
  nav_grid.npz        walkable bool, esdf (m to nearest obstacle), origin, res
  walkable_preview.png  mask + obstacles + waypoints overlay (visual check)

Frames: the collision USD's local frame == the "kitchen frame" used by
configs/waypoints.json (world minus world_pos), so waypoints overlay directly.

Method: gather all mesh triangles (world xform applied); surface-sample them;
obstacle cells = samples in the robot body band [0.10, 1.60] m; floor cells =
samples below 0.10 m; walkable = floor AND NOT dilated-obstacle, eroded by
--robot-radius. ESDF from scipy distance transform (meters).

Run inside env_isaaclab (needs isaacsim's pxr):
    python build_walkable_mask.py [--collision-usd ...] [--res 0.02]
"""
import argparse
import json
import os

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collision-usd",
                    default="/home/stickbot/projects/x2-kitchen-sim/assets/kitchen/kitchen_collision.usd")
    ap.add_argument("--out-dir",
                    default="/home/stickbot/projects/x2-kitchen-sim/assets")
    ap.add_argument("--waypoints",
                    default="/home/stickbot/projects/x2-kitchen-sim/configs/waypoints.json")
    ap.add_argument("--res", type=float, default=0.02, help="grid meters/cell")
    ap.add_argument("--robot-radius", type=float, default=0.35)
    ap.add_argument("--band", type=float, nargs=2, default=[0.10, 1.60],
                    help="obstacle z band (robot body)")
    args = ap.parse_args()

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from pxr import Usd, UsdGeom  # noqa: E402  (importable only after app)

    stage = Usd.Stage.Open(args.collision_usd)
    tris = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        xf = np.asarray(
            UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()), dtype=np.float64)
        ptsw = pts @ xf[:3, :3] + xf[3, :3]
        o = 0
        for c in counts:                     # fan-triangulate any polygon
            for k in range(1, c - 1):
                tris.append((ptsw[idx[o]], ptsw[idx[o + k]],
                             ptsw[idx[o + k + 1]]))
            o += c
    # NOTE: app.close() terminates the PROCESS — it must be the very last
    # statement; all computation and file writes happen before it.
    tris = np.asarray(tris)                  # (T, 3, 3)
    print(f"triangles: {len(tris)}", flush=True)

    # surface-sample triangles proportional to area (~4 samples / cell area)
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    per_cell = args.res * args.res
    n_samp = np.maximum(1, (4.0 * area / per_cell)).astype(int)
    reps = np.repeat(np.arange(len(tris)), n_samp)
    r1 = np.sqrt(np.random.default_rng(0).random(len(reps)))
    r2 = np.random.default_rng(1).random(len(reps))
    samp = ((1 - r1)[:, None] * a[reps]
            + (r1 * (1 - r2))[:, None] * b[reps]
            + (r1 * r2)[:, None] * c[reps])
    print(f"surface samples: {len(samp)}")

    lo = samp[:, :2].min(0) - 0.2
    hi = samp[:, :2].max(0) + 0.2
    nx, ny = np.ceil((hi - lo) / args.res).astype(int)
    def to_cell(p):
        return np.clip(((p[:, :2] - lo) / args.res).astype(int),
                       0, [nx - 1, ny - 1])

    floor = np.zeros((nx, ny), bool)
    obst = np.zeros((nx, ny), bool)
    zlo, zhi = args.band
    fsel = samp[:, 2] < zlo
    osel = (samp[:, 2] >= zlo) & (samp[:, 2] <= zhi)
    fc = to_cell(samp[fsel]); floor[fc[:, 0], fc[:, 1]] = True
    oc = to_cell(samp[osel]); obst[oc[:, 0], oc[:, 1]] = True

    from scipy import ndimage
    obst_closed = ndimage.binary_closing(obst, iterations=2)
    esdf = ndimage.distance_transform_edt(~obst_closed) * args.res
    # Splat scans under-sample floors (fuzzy, patchy under furniture), so
    # floor-coverage detection is unreliable. Instead: free space = cells
    # with >= robot_radius clearance, and walkable = the connected region
    # REACHABLE FROM THE WAYPOINTS — the scanned walls bound it, and
    # exterior/void space (also high-clearance) is excluded automatically.
    free = esdf > args.robot_radius
    lab, nl = ndimage.label(free)
    wps_j = json.load(open(args.waypoints)) if os.path.exists(args.waypoints) else {}
    seeds = set()
    for w in wps_j.values():
        cx, cy = np.clip(((np.array(w["xy"]) - lo) / args.res).astype(int),
                         0, [nx - 1, ny - 1])
        if lab[cx, cy] > 0:
            seeds.add(int(lab[cx, cy]))
    if seeds:
        walkable = np.isin(lab, sorted(seeds))
    else:  # no waypoints available: largest free component
        walkable = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    print(f"floor-sampled cells: {int(floor.sum())} (unused; reachability "
          f"method), free comps: {nl}, seed comps: {sorted(seeds)}",
          flush=True)
    print(f"grid {nx}x{ny} @ {args.res} m | walkable cells: {walkable.sum()} "
          f"({walkable.sum()*per_cell:.1f} m^2)")

    os.makedirs(args.out_dir, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.out_dir, "nav_grid.npz"),
        walkable=walkable, esdf=esdf.astype(np.float32),
        origin=lo, res=args.res)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    Image.fromarray((walkable.T[::-1] * 255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "walkable_mask.png"))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(np.where(obst_closed.T, 0.2, np.where(walkable.T, 1.0, 0.6)),
              origin="lower", cmap="gray",
              extent=[lo[0], lo[0] + nx * args.res,
                      lo[1], lo[1] + ny * args.res])
    if os.path.exists(args.waypoints):
        wps = json.load(open(args.waypoints))
        for name, w in wps.items():
            x, y = w["xy"]
            ax.plot(x, y, "r*", ms=14)
            ax.arrow(x, y, 0.3 * np.cos(w["yaw"]), 0.3 * np.sin(w["yaw"]),
                     head_width=0.07, color="orange")
            ax.annotate(name, (x, y), textcoords="offset points",
                        xytext=(8, 6), color="red", fontsize=10)
    ax.set_title("kitchen walkable mask + waypoints (kitchen frame)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    fig.savefig(os.path.join(args.out_dir, "walkable_preview.png"), dpi=110,
                bbox_inches="tight")
    print("wrote walkable_mask.png / nav_grid.npz / walkable_preview.png",
          flush=True)
    app.close()   # kills the process — keep last
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
