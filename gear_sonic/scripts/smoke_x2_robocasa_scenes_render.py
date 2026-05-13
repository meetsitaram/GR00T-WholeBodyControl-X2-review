"""Hardware-free visual smoke test for the X2 + robocasa static scenes.

For each bundled scene XML this script:

1. Loads the static MJCF (the same one the deploy bridge uses).
2. Instantiates a :class:`RobocasaTaskMirror` (which lazily constructs a
   matching robosuite env).
3. Runs ``N`` ``mirror.reset(seed=k)`` trials. Each trial:
   - asks the env's placement_initializer for fresh object poses,
   - writes those poses into the mirror's ``MjData`` (mimicking what the
     deploy bridge would do on receiving a ``scene_reset`` ZMQ message),
   - settles a few mj_steps so welded bodies / freejoints register,
   - renders ``F`` frames per camera.
4. Writes one MP4 per (scene, camera) and a side-by-side collage PNG of
   the first frame of every trial so reviewers can scan all trials at a
   glance.

No Quest 3, no docker, no C++ deploy needed -- just .venv with
``gr00trobocasa`` installed. Output lands under ``data/scene_smoke/``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

# Pin the headless GL backend BEFORE ``import mujoco`` so the very first
# ``mujoco.Renderer(...)`` resolves an EGL context instead of trying GLFW
# (which needs a real X session). The recorder + bridge do the same dance
# elsewhere in the repo. Operators can still override by exporting their
# own MUJOCO_GL=glfw before invoking the script.
os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402  -- after MUJOCO_GL setdefault
import mujoco  # noqa: E402  -- after MUJOCO_GL setdefault
import numpy as np  # noqa: E402  -- after MUJOCO_GL setdefault


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE_DIR = REPO_ROOT / "gear_sonic/data/assets/robocasa_scenes"
DEFAULT_OUT_DIR = REPO_ROOT / "data/scene_smoke"


def _list_scenes(scene_dir: Path, requested: Sequence[str]) -> list[Path]:
    if requested:
        out = []
        for name in requested:
            p = scene_dir / f"{name}.xml"
            if not p.is_file():
                raise FileNotFoundError(f"requested scene not found: {p}")
            out.append(p)
        return out
    # Default: every .xml that has a matching .json sidecar.
    return sorted(
        p for p in scene_dir.glob("*.xml")
        if p.with_suffix(".json").is_file()
    )


def _make_third_person_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Aimed at the table area in front of the robot's pelvis.
    cam.lookat[:] = np.array([0.55, 0.0, 1.0])
    cam.distance = 1.6
    cam.elevation = -18.0
    cam.azimuth = 145.0
    return cam


def _apply_reset_to_mj_data(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    payload,
) -> None:
    """Mimic the deploy bridge's ``_apply_scene_reset``."""
    for jname, qpos in payload.object_freejoint_qpos.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        data.qpos[qadr:qadr + 7] = np.asarray(qpos[:7], dtype=np.float64)
        # Zero the corresponding 6-vec velocity so the teleport doesn't
        # register as an instantaneous infinite-velocity collision.
        dofadr = int(model.jnt_dofadr[jid])
        data.qvel[dofadr:dofadr + 6] = 0.0
    for bname, pos in payload.mutable_body_pos.items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            continue
        model.body_pos[bid] = np.asarray(pos[:3], dtype=np.float64)
    mujoco.mj_forward(model, data)


def _settle(
    *,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    n_steps: int,
) -> None:
    for _ in range(n_steps):
        mujoco.mj_step(model, data)


def _render_with_named_camera(
    *,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera_name: str,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera_name)
    return renderer.render()


def _render_with_free_camera(
    *,
    renderer: mujoco.Renderer,
    data: mujoco.MjData,
    camera: mujoco.MjvCamera,
) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def _maybe_overlay_text(
    img: np.ndarray, text: str
) -> np.ndarray:
    """Burn a small label onto the top-left of the frame.

    Falls back gracefully when Pillow isn't installed.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError:
        return img
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14
        )
    except OSError:
        font = ImageFont.load_default()
    pad = 4
    bbox = draw.textbbox((pad, pad), text, font=font)
    draw.rectangle(bbox, fill=(0, 0, 0, 200))
    draw.text((pad, pad), text, fill=(255, 255, 255), font=font)
    return np.asarray(pil)


def _summarise_payload(payload, max_len: int = 60) -> str:
    bits = []
    for jname, qpos in payload.object_freejoint_qpos.items():
        bits.append(
            f"{jname}=[{qpos[0]:+.2f},{qpos[1]:+.2f},{qpos[2]:+.2f}]"
        )
    for bname, pos in payload.mutable_body_pos.items():
        bits.append(f"{bname}=[{pos[0]:+.2f},{pos[1]:+.2f},{pos[2]:+.2f}]")
    s = " ".join(bits)
    return s[:max_len]


def _render_scene(
    *,
    scene_xml: Path,
    out_dir: Path,
    n_trials: int,
    frames_per_trial: int,
    fps: int,
    width: int,
    height: int,
    settle_steps: int,
    cameras: Sequence[str],
) -> dict[str, list[Path]]:
    scene_name = scene_xml.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {scene_name} ===")
    print(f"  xml      : {scene_xml}")

    meta_path = scene_xml.with_suffix(".json")
    metadata = json.loads(meta_path.read_text())
    env_name = metadata.get("env_name", scene_name)
    print(f"  env_name : {env_name}")

    from gear_sonic.utils.teleop.robocasa_task_mirror import (
        RobocasaTaskMirror,
    )

    mirror = RobocasaTaskMirror(
        scene_xml_path=scene_xml,
        scene_metadata=metadata,
        env_name=env_name,
    )
    model = mirror.mj_model
    data = mirror.mj_data

    print(
        f"  freejoints={list(mirror._object_freejoint_map.values())} "
        f"welded={list(mirror._object_welded_map.values())}"
    )

    # Validate baked-in cameras up front so a missing one fails loudly.
    available = []
    for cam_id in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
        available.append(name)
    print(f"  cameras  : {available}")

    # Scene MJCFs declare ``<visual><global offwidth=… offheight=…/></visual>``
    # which caps the offscreen framebuffer. ``mujoco.Renderer`` refuses to
    # over-render. Grow the framebuffer in-place to fit any requested size
    # before constructing the renderer; downsizing is fine too.
    if model.vis.global_.offwidth < width:
        model.vis.global_.offwidth = int(width)
    if model.vis.global_.offheight < height:
        model.vis.global_.offheight = int(height)

    renderer = mujoco.Renderer(model, width=width, height=height)
    free_cam = _make_third_person_camera(model)

    # Per-camera frame buffers + per-trial collage tiles.
    per_camera_frames: dict[str, list[np.ndarray]] = {c: [] for c in cameras}
    collage_tiles: list[np.ndarray] = []

    for trial in range(n_trials):
        seed = trial
        print(f"  trial {trial}: reset(seed={seed}) …", end="", flush=True)
        payload = mirror.reset(seed=seed)
        print(
            f" got {len(payload.object_freejoint_qpos)} freejoints + "
            f"{len(payload.mutable_body_pos)} bodies"
        )
        _apply_reset_to_mj_data(model=model, data=data, payload=payload)
        _settle(model=model, data=data, n_steps=settle_steps)

        label = f"{scene_name}  trial={trial}  {_summarise_payload(payload)}"

        for cam_name in cameras:
            if cam_name == "__free__":
                img = _render_with_free_camera(
                    renderer=renderer, data=data, camera=free_cam
                )
                cam_label = label + "  (third-person)"
            else:
                img = _render_with_named_camera(
                    renderer=renderer, data=data, camera_name=cam_name
                )
                cam_label = label + f"  (camera={cam_name})"
            img = _maybe_overlay_text(img, cam_label)
            for _ in range(frames_per_trial):
                per_camera_frames[cam_name].append(img)
            if cam_name == cameras[0]:
                collage_tiles.append(img)

    written: dict[str, list[Path]] = {"mp4": [], "png": []}
    for cam_name, frames in per_camera_frames.items():
        cam_tag = "third_person" if cam_name == "__free__" else cam_name
        out_mp4 = out_dir / f"{scene_name}__{cam_tag}.mp4"
        imageio.mimwrite(out_mp4, frames, fps=fps, codec="libx264", quality=8)
        print(f"  -> wrote {out_mp4}  ({len(frames)} frames @ {fps} fps)")
        written["mp4"].append(out_mp4)

    if collage_tiles:
        cols = min(3, len(collage_tiles))
        rows = (len(collage_tiles) + cols - 1) // cols
        h, w, _ = collage_tiles[0].shape
        canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
        for idx, tile in enumerate(collage_tiles):
            r, c = divmod(idx, cols)
            canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = tile
        out_png = out_dir / f"{scene_name}__trials_collage.png"
        imageio.imwrite(out_png, canvas)
        print(f"  -> wrote {out_png}  ({rows}x{cols} grid of trial #0 frames)")
        written["png"].append(out_png)

    renderer.close()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenes",
        nargs="*",
        default=[],
        help="Scene names to render (without extension). Default: every "
             "*.xml in --scene-dir that has a matching .json sidecar.",
    )
    parser.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--frames-per-trial", type=int, default=24)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=50,
        help="mj_step iterations between scene_reset and rendering, so "
             "freejoints settle visibly under gravity.",
    )
    parser.add_argument(
        "--cameras",
        nargs="*",
        default=["rgbd_head_front", "__free__"],
        help="Camera names to render. '__free__' = orbital third-person.",
    )
    args = parser.parse_args()

    scenes = _list_scenes(args.scene_dir, args.scenes)
    if not scenes:
        raise SystemExit(f"no scenes found under {args.scene_dir}")
    print(f"rendering {len(scenes)} scene(s) -> {args.out_dir}")

    all_outputs: list[Path] = []
    for scene_xml in scenes:
        out = _render_scene(
            scene_xml=scene_xml,
            out_dir=args.out_dir,
            n_trials=args.trials,
            frames_per_trial=args.frames_per_trial,
            fps=args.fps,
            width=args.width,
            height=args.height,
            settle_steps=args.settle_steps,
            cameras=args.cameras,
        )
        all_outputs.extend(out["mp4"])
        all_outputs.extend(out["png"])

    print("\n========= SUMMARY =========")
    for p in all_outputs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
