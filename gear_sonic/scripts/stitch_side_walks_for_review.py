"""Stitch all side-walk source clips into one PKL for SONIC review.

Concatenates every ``loco__*sideway*`` / ``*lateral*`` / ``*strafing*``
clip from ``x2_ultra_bones_seed.pkl`` into a single deploy-format
PKL motion. Each clip is windowed (default 4 s), yaw+XY-aligned to
the previous clip's last frame for continuous travel, and bracketed
by short stand-pose pads + joint blends so SONIC can recover between
clips.

Outputs:
  * ``<out>.pkl``   single-motion PKL keyed by ``side_walks_review``,
                    schema matches ``deploy_x2.sh sim --motion``
                    (``dof``, ``root_rot`` xyzw, ``root_trans_offset``,
                    ``fps``).
  * ``<out>.manifest.json`` per-clip frame ranges + timestamps so a
                    captioner can stamp the clip name on the recorded
                    MuJoCo video.

Run::

    .venv/bin/python -m gear_sonic.scripts.stitch_side_walks_for_review

then feed the PKL into ``deploy_x2.sh sim --motion <pkl> --sim-viewer``.
The companion ``record_sonic_review_video.py`` script wraps that launch
with screen recording + post-process captioning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gear_sonic.utils.planner.blending import (
    build_blend_window,
    yaw_align_segment,
    yaw_of_quat_xyzw,
)
from gear_sonic.utils.planner.constants import (
    DEFAULT_PELVIS_Z_M,
    DEFAULT_STAND_POSE_NP,
    NUM_BODY_DOFS,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_BONES_SEED = _REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_ultra_bones_seed.pkl"
_DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "sim_to_real_anchors" / "browse_sonic" / "baked_pkls"
_DEFAULT_OUT_NAME = "x2_browser_side_walks_review.pkl"


def _identity_quat() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _stand_pose_64() -> np.ndarray:
    return DEFAULT_STAND_POSE_NP.astype(np.float64)


def collect_side_walk_keys(src: dict[str, dict]) -> list[str]:
    """All side-walk source-library keys, sorted by family."""
    keys = [
        k for k in src
        if k.startswith("loco__")
        and any(tok in k.lower() for tok in ("sideway", "lateral", "strafing"))
    ]
    return sorted(keys)


def _stand_pad_at(
    xy: np.ndarray, yaw: float, n: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stand-pose hold of ``n`` frames anchored at ``(xy, yaw)``."""
    dof = np.broadcast_to(_stand_pose_64(), (n, NUM_BODY_DOFS)).copy()
    quat = Rot.from_euler("z", yaw).as_quat()
    rot = np.broadcast_to(quat, (n, 4)).copy()
    trans = np.empty((n, 3), dtype=np.float64)
    trans[:, 0] = xy[0]
    trans[:, 1] = xy[1]
    trans[:, 2] = DEFAULT_PELVIS_Z_M
    return dof, rot, trans


def stitch_side_walks(
    src: dict[str, dict],
    keys: list[str],
    *,
    max_clip_seconds: float = 3.0,
    pad_seconds: float = 1.2,
    blend_seconds: float = 0.3,
    target_fps: float = 30.0,
) -> tuple[dict, list[dict], float]:
    """Build the stitched PKL contents and the manifest.

    Returns ``(motion_dict, manifest_entries, total_seconds)``.
    """
    pad_frames = max(2, int(round(pad_seconds * target_fps)))
    blend_frames = max(2, int(round(blend_seconds * target_fps)))

    parts_dof: list[np.ndarray] = []
    parts_rot: list[np.ndarray] = []
    parts_trans: list[np.ndarray] = []
    manifest: list[dict] = []

    cursor_xy = np.zeros(2, dtype=np.float64)
    cursor_yaw = 0.0
    cursor_frame = 0

    # Initial pre-roll stand pad so SONIC has time to anchor RSI.
    pre_dof, pre_rot, pre_trans = _stand_pad_at(cursor_xy, cursor_yaw, pad_frames)
    parts_dof.append(pre_dof)
    parts_rot.append(pre_rot)
    parts_trans.append(pre_trans)
    cursor_frame += pad_frames

    skipped: list[tuple[str, str]] = []
    for clip_idx, k in enumerate(keys):
        m = src[k]
        c_dof = np.asarray(m["dof"], dtype=np.float64)
        c_rot = np.asarray(m["root_rot"], dtype=np.float64)
        c_trans = np.asarray(m["root_trans_offset"], dtype=np.float64)
        c_fps = float(m.get("fps", 30.0))

        # Resample to target fps if mismatched (rare; most clips are 30 fps).
        if abs(c_fps - target_fps) > 0.01:
            from gear_sonic.utils.planner.blending import resample_motion_30_to_50hz
            try:
                c_dof, c_rot, c_trans = resample_motion_30_to_50hz(
                    c_dof, c_rot, c_trans, c_fps, target_fps=target_fps
                )
            except Exception as exc:
                skipped.append((k, f"resample failed: {exc}"))
                continue

        if c_dof.shape[0] < 4:
            skipped.append((k, f"too short ({c_dof.shape[0]} frames)"))
            continue

        # Window: take first max_clip_seconds.
        max_frames = int(round(max_clip_seconds * target_fps))
        n_clip = min(max_frames, c_dof.shape[0])
        c_dof = c_dof[:n_clip]
        c_rot = c_rot[:n_clip]
        c_trans = c_trans[:n_clip]

        # Yaw + XY-align frame 0 to the cursor (continuous travel).
        c_dof, c_rot, c_trans = yaw_align_segment(
            c_dof, c_rot, c_trans, xy_world=cursor_xy, yaw_world=cursor_yaw
        )

        # Blend from the trailing pad's stand pose to the clip's frame 0.
        # build_blend_window holds XY at the start endpoint (no foot-skate);
        # we want it at the cursor so the blend stays anchored at the cursor.
        pad_quat = Rot.from_euler("z", cursor_yaw).as_quat()
        pad_xyz = np.array([cursor_xy[0], cursor_xy[1], DEFAULT_PELVIS_Z_M])
        bl_dof, bl_rot, bl_trans = build_blend_window(
            dof_start=_stand_pose_64(),
            rot_start_xyzw=pad_quat,
            trans_start_xyz=pad_xyz,
            dof_end=c_dof[0],
            rot_end_xyzw=c_rot[0],
            trans_end_xyz=c_trans[0],
            n_frames=blend_frames,
        )

        # Append: blend, then clip body.
        parts_dof.append(bl_dof)
        parts_rot.append(bl_rot)
        parts_trans.append(bl_trans)
        clip_start_frame = cursor_frame + blend_frames

        parts_dof.append(c_dof)
        parts_rot.append(c_rot)
        parts_trans.append(c_trans)
        clip_end_frame = clip_start_frame + n_clip

        # Update cursor to clip's last frame so the next clip starts there.
        cursor_xy = c_trans[-1, :2].copy()
        cursor_yaw = float(yaw_of_quat_xyzw(c_rot[-1]))

        # Trailing pad: stand pose at the new cursor.
        post_dof, post_rot, post_trans = _stand_pad_at(cursor_xy, cursor_yaw, pad_frames)
        parts_dof.append(post_dof)
        parts_rot.append(post_rot)
        parts_trans.append(post_trans)

        # Record manifest before bumping cursor_frame past the pad.
        manifest.append(
            {
                "index": clip_idx,
                "clip_name": k,
                "blend_start_frame": cursor_frame,
                "clip_start_frame": clip_start_frame,
                "clip_end_frame": clip_end_frame,
                "pad_end_frame": clip_end_frame + pad_frames,
                "blend_start_s": cursor_frame / target_fps,
                "clip_start_s": clip_start_frame / target_fps,
                "clip_end_s": clip_end_frame / target_fps,
                "pad_end_s": (clip_end_frame + pad_frames) / target_fps,
                "n_clip_frames": n_clip,
                "src_fps": c_fps,
            }
        )
        cursor_frame = clip_end_frame + pad_frames

    final_dof = np.concatenate(parts_dof, axis=0)
    final_rot = np.concatenate(parts_rot, axis=0)
    final_trans = np.concatenate(parts_trans, axis=0)
    total_frames = final_dof.shape[0]
    total_seconds = total_frames / target_fps

    motion_dict = {
        "side_walks_review": {
            "dof": final_dof,
            "root_rot": final_rot,
            "root_trans_offset": final_trans,
            "fps": float(target_fps),
        }
    }

    if skipped:
        print(f"[stitch] skipped {len(skipped)} clips:")
        for k, reason in skipped:
            print(f"           {k}  ({reason})")

    return motion_dict, manifest, total_seconds


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--source-pkl", type=Path, default=_DEFAULT_BONES_SEED,
                   help=f"Bones-seed PKL (default: {_DEFAULT_BONES_SEED.relative_to(_REPO_ROOT)})")
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT_DIR / _DEFAULT_OUT_NAME,
                   help=f"Output PKL path (default: {(_DEFAULT_OUT_DIR / _DEFAULT_OUT_NAME).relative_to(_REPO_ROOT)})")
    p.add_argument("--max-clip-seconds", type=float, default=3.0,
                   help="Max seconds per clip (longer clips are truncated; default 3.0)")
    p.add_argument("--pad-seconds", type=float, default=1.2,
                   help="Stand-pose hold between clips (default 1.2 s)")
    p.add_argument("--blend-seconds", type=float, default=0.3,
                   help="Joint blend in/out of pad (default 0.3 s)")
    p.add_argument("--target-fps", type=float, default=30.0,
                   help="Output fps (default 30; matches source)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only stitch the first N clips (debug aid)")
    p.add_argument("--keys", type=str, nargs="+", default=None,
                   help="If set, stitch ONLY these clip keys (full names from "
                        "the bones-seed). Skips the auto side-walk filter -- "
                        "use to bake a single clip for focused SONIC review.")
    args = p.parse_args()

    print(f"[stitch] loading source library from {args.source_pkl}")
    src = joblib.load(args.source_pkl)

    if args.keys:
        missing = [k for k in args.keys if k not in src]
        if missing:
            print(f"[stitch] ERROR: keys not found in source: {missing}")
            return 2
        keys = list(args.keys)
        print(f"[stitch] --keys -> using {len(keys)} explicit clip(s)")
    else:
        keys = collect_side_walk_keys(src)
        print(f"[stitch] {len(keys)} candidate side-walk clips")
    if args.limit is not None:
        keys = keys[: args.limit]
        print(f"[stitch] --limit -> using first {len(keys)} clips")

    motion_dict, manifest, total_s = stitch_side_walks(
        src, keys,
        max_clip_seconds=args.max_clip_seconds,
        pad_seconds=args.pad_seconds,
        blend_seconds=args.blend_seconds,
        target_fps=args.target_fps,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(motion_dict, args.out)

    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "pkl": str(args.out),
                "fps": args.target_fps,
                "total_frames": int(total_s * args.target_fps),
                "total_seconds": total_s,
                "max_clip_seconds": args.max_clip_seconds,
                "pad_seconds": args.pad_seconds,
                "blend_seconds": args.blend_seconds,
                "n_clips": len(manifest),
                "clips": manifest,
            },
            indent=2,
        )
    )

    print()
    print(f"[stitch] wrote {args.out}")
    print(f"          {len(manifest)} clips stitched, total = {total_s:.1f} s")
    print(f"          manifest = {manifest_path}")
    print()
    print("# Launch with screen recording + auto-caption:")
    print(
        f"  .venv/bin/python -m gear_sonic.scripts.record_sonic_review_video \\\n"
        f"    --motion {args.out} \\\n"
        f"    --manifest {manifest_path}"
    )
    print()
    print("# Or just play in SONIC without recording:")
    print(
        f"  bash gear_sonic_deploy/deploy_x2.sh sim --no-confirm \\\n"
        f"    --motion {args.out} \\\n"
        f"    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \\\n"
        f"    --sim-viewer \\\n"
        f"    --max-duration {int(total_s + 5)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
