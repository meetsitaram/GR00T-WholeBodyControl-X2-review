"""Offscreen side-by-side MP4 of X2 vs G1 planner output, auto-cycling speeds.

Renders the qpos trajectories captured by ``test_e2e_velocity_tracking.py``
(one NPZ per ckpt-set, same sweep grid) into a single MP4 -- X2 on the left,
G1 on the right -- automatically playing through every aligned trial (e.g.
forward speeds 0.0..0.6 m/s) with the commanded velocity burned into each
frame. Capture-for-analysis companion to the interactive
``view_e2e_x2_vs_g1.py`` (no display needed; writes a file).

Each robot is rendered in ITS OWN native scene (its MJCF already ships a
floor + skybox) and the two frames are stitched horizontally. This avoids
the z-fighting flicker you get from stacking both robots' coplanar floor
planes into one combined scene.

Usage::

    PYTHONPATH="${PWD}/motionbricks:${PWD}" MUJOCO_GL=egl python \\
      motionbricks/scripts/render_x2_vs_g1_video.py \\
        --x2-npz out/e2e_headtohead/e2e_x2_forward.npz \\
        --g1-npz out/e2e_headtohead/e2e_g1_forward.npz \\
        --out out/e2e_headtohead/x2_vs_g1_forward.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import view_e2e_x2_vs_g1 as viewer  # noqa: E402  (reuse NPZ + canonicalize helpers)


def _load_robot_model(mjcf: Path, offw: int, offh: int) -> mujoco.MjModel:
    """Load a single robot MJCF with an offscreen buffer + multisampling."""
    spec = mujoco.MjSpec.from_file(str(mjcf))
    try:
        spec.visual.global_.offwidth = int(offw)
        spec.visual.global_.offheight = int(offh)
        spec.visual.quality.offsamples = 8  # MSAA -> kills checker shimmer
    except Exception as e:  # pragma: no cover - version-dependent attrs
        print(f"[render] warn: could not set visual quality ({e})")
    return spec.compile()


def _root_addr(model: mujoco.MjModel) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
    if jid < 0:
        raise KeyError("joint 'floating_base_joint' not found")
    return int(model.jnt_qposadr[jid])


def _overlay(
    img: np.ndarray,
    title: str,
    lines: list[tuple[str, tuple[int, int, int]]],
) -> np.ndarray:
    out = img.copy()
    w = out.shape[1]
    # Centered title banner at the very top.
    if title:
        scale = 1.2
        (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, scale, 3)
        tx = max(10, (w - tw) // 2)
        cv2.putText(out, title, (tx, 44), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(out, title, (tx, 44), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    y = 92 if title else 36
    for text, rgb in lines:
        bgr = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        cv2.putText(out, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2, cv2.LINE_AA)
        y += 40
    return out


class _RobotRenderer:
    def __init__(self, mjcf: Path, w: int, h: int) -> None:
        self.model = _load_robot_model(mjcf, w, h)
        self.data = mujoco.MjData(self.model)
        self.addr = _root_addr(self.model)
        self.renderer = mujoco.Renderer(self.model, height=h, width=w)
        self.cam = mujoco.MjvCamera()
        self.cam.azimuth = 140.0
        self.cam.elevation = -18.0
        self.cam.distance = 4.2

    def render(self, qpos: np.ndarray) -> np.ndarray:
        dim = qpos.shape[0]
        self.data.qpos[self.addr : self.addr + dim] = qpos
        mujoco.mj_forward(self.model, self.data)
        # Track root x,y so forward walking AND turns/arcs stay framed.
        self.cam.lookat[:] = [float(qpos[0]), float(qpos[1]), 0.75]
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--x2-npz", type=Path, required=True)
    p.add_argument("--g1-npz", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--x2-mjcf", type=Path, default=None)
    p.add_argument("--g1-mjcf", type=Path, default=None)
    p.add_argument("--width", type=int, default=1280, help="TOTAL width (split across 2 panes)")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=float, default=0.0, help="0 = use NPZ fps")
    p.add_argument("--hold-frames", type=int, default=12)
    p.add_argument(
        "--right-label", type=str, default="G1",
        help="Label for the right pane (e.g. 'G1', or 'X2 g1-style' for a clip A/B).",
    )
    p.add_argument(
        "--title", type=str, default="Motionbricks K-Planner validation",
        help="Centered title banner at the top of every frame.",
    )
    p.add_argument("--no-canonicalize", action="store_true")
    args = p.parse_args(argv)

    x2 = viewer._load_npz(args.x2_npz)
    g1 = viewer._load_npz(args.g1_npz)
    if not args.no_canonicalize:
        x2["qpos_traj"] = viewer._canonicalize_trial_qpos(x2["qpos_traj"])
        g1["qpos_traj"] = viewer._canonicalize_trial_qpos(g1["qpos_traj"])
    aligned = viewer._align_trials(x2, g1)
    fps = args.fps if args.fps > 0 else float(x2["fps"])
    horizon = int(x2["qpos_traj"].shape[1])

    pane_w = args.width // 2
    x2_mjcf = args.x2_mjcf or viewer._DEFAULT_X2_MJCF
    g1_mjcf = args.g1_mjcf or viewer._DEFAULT_G1_MJCF
    left = _RobotRenderer(x2_mjcf, pane_w, args.height)
    right = _RobotRenderer(g1_mjcf, pane_w, args.height)

    print(f"[render] {len(aligned)} trials  horizon={horizon}  fps={fps:.1f}  "
          f"pane={pane_w}x{args.height}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=fps, macro_block_size=None)

    def _frame(x2_i: int, g1_i: int, f: int, lines) -> np.ndarray:
        lf = left.render(x2["qpos_traj"][x2_i, f])
        rf = right.render(g1["qpos_traj"][g1_i, f])
        combined = np.concatenate([lf, rf], axis=1)
        return _overlay(combined, args.title, lines)

    # Scripted-routine NPZs carry a per-frame segment schedule; sweep NPZs
    # don't (one constant intent per trial).
    segments = x2.get("segments") or g1.get("segments")

    def _seg_at(frame: int):
        if not segments:
            return None
        for s in segments:
            start = int(s["start_frame"])
            if start <= frame < start + int(s["n_frames"]):
                return s
        return segments[-1]

    n = 0
    for t, (x2_i, g1_i, axis, intent) in enumerate(aligned):
        vz, vx, yaw = float(intent[2]), float(intent[1]), float(intent[0])

        def _lines(frame: int):
            seg = _seg_at(frame)
            if seg is not None:
                head = (f"{seg['label']}   vz(fwd)={float(seg['vel_z']):+.2f}  "
                        f"vx={float(seg['vel_x']):+.2f}  yaw={float(seg['yaw_rate']):+.2f}")
            else:
                head = (f"[{t + 1}/{len(aligned)}]  {axis}  vz(fwd)={vz:+.2f} m/s  "
                        f"vx={vx:+.2f}  yaw={yaw:+.2f}")
            return [
                (head, (255, 235, 120)),
                (f"X2 (left)   vs   {args.right_label} (right)", (180, 220, 255)),
            ]

        for _ in range(args.hold_frames):
            writer.append_data(_frame(x2_i, g1_i, 0, _lines(0))); n += 1
        for f in range(horizon):
            writer.append_data(_frame(x2_i, g1_i, f, _lines(f))); n += 1
        print(f"  trial {t + 1}/{len(aligned)}  frames={horizon}")

    writer.close()
    print(f"[render] wrote {args.out}  ({n} frames, {n / fps:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
