"""Bake one curated primitive into the PKL format ``deploy_x2.sh --motion`` expects.

Diagnostic helper for the planner primitive bring-up. The browser
(`browse_x2_planner_primitives.py --with-sonic`) drives the deploy via
ZMQ (`--vla` flag → `ZmqPoseInputSource` in the C++ deploy). When a
primitive looks fine in the kinematic viewer but produces no movement
under SONIC, we want to rule out the ZMQ wire path entirely by feeding
the **same** clip bytes through the proven `--motion <pkl>`
(`PklMotionReference` + `--sim-profile parity` RSI) path that
``deploy_x2.sh sim --motion x2_ultra_walk_forward.pkl`` uses.

The runtime primitives PKL (``gear_sonic/data/motions/x2_planner_primitives.pkl``)
stores each bin as ``{"dof": (T,31), "root_rot_xyzw": (T,4),
"root_trans": (T,3), "fps": float, ...}`` (see
``build_x2_planner_primitives.write_primitives_pkl``). The deploy's
PKL loader (see
``gear_sonic_deploy/scripts/export_motion_for_deploy.py``) wants
``{"dof": (T,31), "root_rot": (T,4) xyzw, "root_trans_offset": (T,3),
"fps": float}`` keyed by motion name. This script does that key
remap and writes a single-motion PKL ready to feed straight into
``deploy_x2.sh sim --motion <out.pkl>``.

Example::

    .venv/bin/python -m gear_sonic.scripts.bake_primitive_for_deploy \\
        fwd_step_half_ft

prints the absolute path of the baked PKL plus a ready-to-paste
``deploy_x2.sh sim --motion ... --model ...`` command (assuming the
default SONIC checkpoint path).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_PRIMS_PKL = _REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_planner_primitives.pkl"
_DEFAULT_BONES_SEED = _REPO_ROOT / "gear_sonic" / "data" / "motions" / "x2_ultra_bones_seed.pkl"
_DEFAULT_CKPT = Path(
    "/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/"
    "model_step_025000.pt"
)
_DEPLOY_SH = _REPO_ROOT / "gear_sonic_deploy" / "deploy_x2.sh"
# deploy_x2.sh auto-re-execs inside the docker_x2/x2sim container, which
# only bind-mounts $HOME -- /tmp on the host is NOT visible. Stash the
# baked PKLs under the repo (always inside $HOME) so the container can
# see them. Output is gitignored by default since the dir doesn't exist
# yet; add it to .gitignore if you want to be explicit.
_DEFAULT_BAKED_DIR = _REPO_ROOT / "data" / "sim_to_real_anchors" / "browse_sonic" / "baked_pkls"


def _resolve_onnx(checkpoint: Path) -> Path:
    """Same resolution rule as ``browse_x2_planner_primitives._resolve_deploy_model_onnx``.

    The deploy needs the ONNX bundle next to the .pt, under ``exported/``.
    """
    exported = checkpoint.parent / "exported"
    if not exported.is_dir():
        raise FileNotFoundError(
            f"No 'exported/' dir next to checkpoint: {checkpoint}\n"
            f"  expected: {exported}\n"
            f"  Run the SONIC export step or pass --model explicitly."
        )
    onnx_files = sorted(exported.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files in {exported}")
    return onnx_files[0]


def _slerp_quats(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Vectorised SLERP for shape (T, 4) -> (T, 4) result.

    q0, q1 are (4,) arrays in xyzw order. t is (T,) in [0, 1].
    """
    from scipy.spatial.transform import Rotation as Rot, Slerp
    times = np.array([0.0, 1.0])
    rots = Rot.from_quat(np.stack([q0, q1]))
    slerp = Slerp(times, rots)
    return slerp(np.clip(t, 0.0, 1.0)).as_quat()


def _splice_blend(
    A_dof: np.ndarray, A_rot: np.ndarray, A_trans: np.ndarray,
    B_dof: np.ndarray, B_rot: np.ndarray, B_trans: np.ndarray,
    blend_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate A then B with a smooth transition over ``blend_frames``.

    The blend window is *inserted* between A and B (so output length is
    len(A) + blend_frames + len(B)). Each blend frame i interpolates:
      - dof    : (1 - alpha) * A[-1] + alpha * B[0]
      - rot    : SLERP(A[-1], B[0], alpha)
      - trans  : (1 - alpha) * A[-1] + alpha * B[0]
    where alpha = (i + 1) / (blend_frames + 1) (excludes 0 and 1 since
    those are A's last frame and B's first frame respectively).
    """
    if blend_frames <= 0:
        dof = np.concatenate([A_dof, B_dof], axis=0)
        rot = np.concatenate([A_rot, B_rot], axis=0)
        trans = np.concatenate([A_trans, B_trans], axis=0)
        return dof, rot, trans

    alphas = np.arange(1, blend_frames + 1) / float(blend_frames + 1)
    dof_blend = (1.0 - alphas)[:, None] * A_dof[-1:] + alphas[:, None] * B_dof[0:1]
    rot_blend = _slerp_quats(A_rot[-1], B_rot[0], alphas)
    trans_blend = (1.0 - alphas)[:, None] * A_trans[-1:] + alphas[:, None] * B_trans[0:1]
    dof = np.concatenate([A_dof, dof_blend, B_dof], axis=0)
    rot = np.concatenate([A_rot, rot_blend, B_rot], axis=0)
    trans = np.concatenate([A_trans, trans_blend, B_trans], axis=0)
    return dof, rot, trans


def bake_window_from_source(
    motion_key: str,
    start: int,
    n: int | None,
    source_pkl: Path,
    out_path: Path,
    prepend_stand_s: float = 0.0,
    append_stand_s: float = 0.0,
    prepend_source_frame: int | None = None,
    waist_roll_deg: float = 0.0,
    waist_pitch_deg: float = 0.0,
    splice_motion_key: str | None = None,
    splice_start: int = 0,
    splice_n: int | None = None,
    splice_blend_frames: int = 6,
    no_leg_back: bool = False,
    leg_back_baseline_source_frame: int | None = None,
) -> dict:
    """Bake an ad-hoc [start:start+n] window from a bones-seed clip.

    Diagnostic-only: lets us test candidate source windows for new
    primitives without round-tripping through the recipe YAML +
    ``build_x2_planner_primitives.py``. Use ``bake_one`` once a
    promising window is locked into a recipe.
    """
    if not source_pkl.is_file():
        raise FileNotFoundError(f"source PKL not found: {source_pkl}")
    raw = joblib.load(source_pkl)
    if motion_key not in raw:
        # Be helpful: show prefix matches.
        prefix_matches = sorted(k for k in raw if k.startswith(motion_key[:20]))[:10]
        raise KeyError(
            f"no such motion_key {motion_key!r} in {source_pkl}\n"
            f"  similar prefixes: {prefix_matches}"
        )
    m = raw[motion_key]
    dof_full = np.asarray(m["dof"], dtype=np.float64)
    rot_full = np.asarray(m["root_rot"], dtype=np.float64)
    trans_full = np.asarray(m["root_trans_offset"], dtype=np.float64)
    fps = float(m.get("fps", 30.0))

    T_full = dof_full.shape[0]
    if start < 0 or start >= T_full:
        raise ValueError(f"start={start} out of range [0,{T_full}) for {motion_key!r}")
    end = T_full if n is None else min(start + n, T_full)
    if end - start < 2:
        raise ValueError(f"window [{start}:{end}] is < 2 frames")

    dof = dof_full[start:end].copy()
    rot = rot_full[start:end].copy()
    trans = trans_full[start:end].copy()

    # Optional: clamp leg hip-pitch joints so neither leg sweeps behind
    # a reference baseline. Used for side-step primitives where the
    # natural side-walk gait has the trailing leg's hip_pitch dip
    # negative (foot moves toward -Y of the robot frame, "behind" the
    # body). Clamping both legs to baseline lets feet lift forward but
    # never trail back. Joint indices: LEFT_HIP_PITCH=0,
    # RIGHT_HIP_PITCH=6 (see gear_sonic/utils/planner/constants.py).
    if no_leg_back:
        bf = (leg_back_baseline_source_frame
              if leg_back_baseline_source_frame is not None
              else start)
        if bf < 0 or bf >= T_full:
            raise ValueError(
                f"--leg-back-baseline-source-frame {bf} out of range "
                f"[0,{T_full})"
            )
        baseline = dof_full[bf]
        # Clamp both leg hip_pitch values to be >= baseline.
        for idx in (0, 6):
            dof[:, idx] = np.maximum(dof[:, idx], baseline[idx])

    # Optional: overlay a half-sine torso tilt on top of the motion
    # frames. Physically: counter-shift the torso to load the support
    # leg, freeing the swing leg; then return to neutral as the swing
    # leg lands. Profile starts at 0, peaks at the middle, returns to
    # 0 -- so it can splice cleanly into a neutral-pose prepend.
    # Indices: WAIST_PITCH_IDX=13, WAIST_ROLL_IDX=14 (see
    # gear_sonic/utils/planner/constants.py).
    n_motion = dof.shape[0]
    if abs(waist_roll_deg) > 1e-6 or abs(waist_pitch_deg) > 1e-6:
        # Half-sine profile across the motion: 0 at i=0, peak at i=mid,
        # 0 at i=n_motion-1 (so the splice into prepend / append-zero
        # is continuous).
        t_phase = np.linspace(0.0, np.pi, n_motion)
        envelope = np.sin(t_phase)
        if abs(waist_roll_deg) > 1e-6:
            dof[:, 14] += np.deg2rad(waist_roll_deg) * envelope
        if abs(waist_pitch_deg) > 1e-6:
            dof[:, 13] += np.deg2rad(waist_pitch_deg) * envelope

    # Re-zero only X/Y so the bridge RSI doesn't try to spawn the robot
    # 5 m down the y-axis. Z is the *absolute* pelvis height (~0.68 m
    # for a stand) -- subtracting it sub-floors the spawn and physics
    # catapults the robot. Orientation is left alone too;
    # PklMotionReference yaw-anchors at runtime ("applied Δyaw = -0.00
    # deg" in its log).
    trans[:, 0] -= trans[0, 0]
    trans[:, 1] -= trans[0, 1]

    # Optional splice: chain a second window onto the end of this one
    # with a small joint-angle blend. Used to attach the natural
    # "settle-to-stop" tail of the source clip onto a primer window
    # that builds gait momentum, so the policy gets:
    #   primer (rest -> stride momentum)
    #   blend (smooth joint transition)
    #   tail (gait deceleration ending feet-together)
    # WITHOUT having to play all the intermediate strides that take
    # the robot too far. The splice tail's XY/Z are recentered so it
    # starts continuously from the primer's last frame; the joint
    # angles, however, jump unless blend_frames > 0.
    splice_info: dict | None = None
    if splice_motion_key is not None:
        skey = splice_motion_key if splice_motion_key else motion_key
        if skey not in raw:
            raise KeyError(f"splice motion_key {skey!r} not found")
        sm = raw[skey]
        s_dof_full = np.asarray(sm["dof"], dtype=np.float64)
        s_rot_full = np.asarray(sm["root_rot"], dtype=np.float64)
        s_trans_full = np.asarray(sm["root_trans_offset"], dtype=np.float64)
        s_T = s_dof_full.shape[0]
        if splice_start < 0 or splice_start >= s_T:
            raise ValueError(f"splice_start={splice_start} out of range for {skey!r}")
        s_end = s_T if splice_n is None else min(splice_start + splice_n, s_T)
        if s_end - splice_start < 1:
            raise ValueError(f"splice window [{splice_start}:{s_end}] too short")
        s_dof = s_dof_full[splice_start:s_end].copy()
        s_rot = s_rot_full[splice_start:s_end].copy()
        s_trans = s_trans_full[splice_start:s_end].copy()
        # Recenter splice's XY/Z so its first frame sits exactly where
        # the primary window's last frame is. (Same Z too -- if the
        # splice clip's pelvis is at a slightly different height, we
        # don't want a spawn-time foot-skip.)
        s_trans[:, 0] += trans[-1, 0] - s_trans[0, 0]
        s_trans[:, 1] += trans[-1, 1] - s_trans[0, 1]
        s_trans[:, 2] += trans[-1, 2] - s_trans[0, 2]
        # Splice in with joint blend.
        dof, rot, trans = _splice_blend(
            dof, rot, trans, s_dof, s_rot, s_trans,
            blend_frames=int(splice_blend_frames),
        )
        splice_info = {
            "key": skey,
            "start": int(splice_start),
            "end": int(s_end),
            "blend_frames": int(splice_blend_frames),
        }

    # Optional: repeat a held pose for prepend_stand_s seconds at the
    # start (resp. append_stand_s at the end). Joint pose source:
    #   --prepend-source-frame N : use source frame N's joint angles
    #                              (e.g. a natural pre-gait stand pose)
    #   default                  : reuse window[0]'s joint angles
    # Either way, the held pose's root_trans is window[0]'s recentered
    # value (no teleport) and root_rot is window[0]'s quat. If the held
    # pose's joints differ from window[0]'s joints there will be a
    # one-frame splice at the prepend->window boundary; SONIC's
    # action_clip handles small splices, but if you see a "snap" pick
    # a held pose closer to window[0]'s joints.
    n_pre = int(round(prepend_stand_s * fps))
    n_post = int(round(append_stand_s * fps))
    if n_pre > 0:
        if prepend_source_frame is None:
            held_dof = dof_full[start]
        else:
            if prepend_source_frame < 0 or prepend_source_frame >= T_full:
                raise ValueError(
                    f"--prepend-source-frame {prepend_source_frame} out of range "
                    f"[0,{T_full}) for {motion_key!r}"
                )
            held_dof = dof_full[prepend_source_frame]
        dof = np.concatenate([np.broadcast_to(held_dof, (n_pre,) + held_dof.shape).copy(), dof], axis=0)
        rot = np.concatenate([np.broadcast_to(rot[0:1], (n_pre,) + rot.shape[1:]).copy(), rot], axis=0)
        trans = np.concatenate([np.broadcast_to(trans[0:1], (n_pre,) + trans.shape[1:]).copy(), trans], axis=0)
    if n_post > 0:
        dof = np.concatenate([dof, np.broadcast_to(dof[-1:], (n_post,) + dof.shape[1:]).copy()], axis=0)
        rot = np.concatenate([rot, np.broadcast_to(rot[-1:], (n_post,) + rot.shape[1:]).copy()], axis=0)
        trans = np.concatenate([trans, np.broadcast_to(trans[-1:], (n_post,) + trans.shape[1:]).copy()], axis=0)

    nice_name = f"{motion_key}__win{start}_{end}"
    if splice_info is not None:
        nice_name += f"__splice{splice_info['start']}_{splice_info['end']}b{splice_info['blend_frames']}"
    if n_pre > 0 or n_post > 0:
        nice_name += f"__pad{n_pre}_{n_post}"
    out = {
        nice_name: {
            "dof": dof,
            "root_rot": rot,
            "root_trans_offset": trans,
            "fps": fps,
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, out_path)
    return {
        "bin": nice_name,
        "out_path": str(out_path),
        "n_frames": int(dof.shape[0]),
        "fps": fps,
        "duration_s": float(dof.shape[0]) / fps,
        "motion_key": motion_key,
        "recipe_ops": [f"clip_window[{start}:{end}]", "recenter_root"],
    }


def bake_one(
    bin_name: str,
    primitives_pkl: Path,
    out_path: Path,
) -> dict:
    """Read one bin from the runtime primitives PKL, write a deploy-format PKL."""
    if not primitives_pkl.is_file():
        raise FileNotFoundError(f"primitives PKL not found: {primitives_pkl}")
    bins = joblib.load(primitives_pkl)
    if bin_name not in bins:
        raise KeyError(
            f"no such bin {bin_name!r} in {primitives_pkl}\n"
            f"  available: {sorted(bins.keys())}"
        )
    entry = bins[bin_name]
    dof = np.asarray(entry["dof"], dtype=np.float64)
    rot = np.asarray(entry["root_rot_xyzw"], dtype=np.float64)
    trans = np.asarray(entry["root_trans"], dtype=np.float64)
    fps = float(entry["fps"])

    if dof.ndim != 2 or dof.shape[1] != 31:
        raise ValueError(f"bin {bin_name!r} dof shape {dof.shape}; expected (T, 31)")
    if rot.shape != (dof.shape[0], 4):
        raise ValueError(f"bin {bin_name!r} root_rot shape {rot.shape}; expected ({dof.shape[0]}, 4)")
    if trans.shape != (dof.shape[0], 3):
        raise ValueError(f"bin {bin_name!r} root_trans shape {trans.shape}; expected ({dof.shape[0]}, 3)")

    # deploy_x2.sh feeds this through PklMotionReference / RSI:
    #   - field renames: root_rot_xyzw -> root_rot, root_trans -> root_trans_offset
    #   - keyed by motion name (deploy takes the first entry of the dict)
    out = {
        bin_name: {
            "dof": dof,
            "root_rot": rot,
            "root_trans_offset": trans,
            "fps": fps,
        }
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, out_path)
    return {
        "bin": bin_name,
        "out_path": str(out_path),
        "n_frames": int(dof.shape[0]),
        "fps": fps,
        "duration_s": float(dof.shape[0]) / fps,
        "motion_key": entry.get("motion_key"),
        "recipe_ops": entry.get("recipe_ops"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("bin_name", nargs="?", default=None,
                   help="Primitive bin to bake (e.g. fwd_step_half_ft). "
                        "Required unless --list, --motion-key, or --list-source is given.")
    p.add_argument("--primitives-pkl", type=Path, default=_DEFAULT_PRIMS_PKL,
                   help=f"Runtime primitives PKL (default: {_DEFAULT_PRIMS_PKL.relative_to(_REPO_ROOT)})")
    p.add_argument("--out", type=Path, default=None,
                   help=f"Output PKL path (default: {_DEFAULT_BAKED_DIR.relative_to(_REPO_ROOT)}/x2_browser_<name>.pkl). "
                        "Must live under $HOME so the deploy docker container can see it.")
    p.add_argument("--checkpoint", type=Path, default=_DEFAULT_CKPT,
                   help=f"SONIC .pt checkpoint to derive ONNX from (default: {_DEFAULT_CKPT})")
    p.add_argument("--list", action="store_true",
                   help="List available primitive bins and exit.")
    # ---- ad-hoc source-window mode (diagnostic) -------------------------
    src = p.add_argument_group("ad-hoc source window (bypass recipes for testing)")
    src.add_argument("--motion-key", type=str, default=None,
                     help="Bones-seed motion key to slice (e.g. loco__walk_forward_loop_003__A034). "
                          "When set, --start/--n carve a window and we bake straight from the source "
                          "without going through the recipe. The recipe rebuild step is the right "
                          "answer once you've found a window that walks well.")
    src.add_argument("--start", type=int, default=0,
                     help="Window start frame (inclusive; default 0).")
    src.add_argument("--n", type=int, default=None,
                     help="Window length in frames (default = to end of clip).")
    src.add_argument("--source-pkl", type=Path, default=_DEFAULT_BONES_SEED,
                     help=f"Bones-seed PKL to slice from (default: {_DEFAULT_BONES_SEED.relative_to(_REPO_ROOT)})")
    src.add_argument("--list-source", action="store_true",
                     help="Search bones-seed for motion_key prefix (use with --motion-key).")
    src.add_argument("--prepend-stand-s", type=float, default=0.0,
                     help="Hold a stand pose for N seconds at the start. By default "
                          "the held pose is window[0]'s joint angles (no splice "
                          "discontinuity); pair with --prepend-source-frame to use a "
                          "different (e.g. more natural) source frame as the held pose.")
    src.add_argument("--prepend-source-frame", type=int, default=None,
                     help="Source-frame index whose joint angles are held during the "
                          "prepend (e.g. a frame from the clip's natural pre-gait stand). "
                          "If omitted, the window's frame 0 is held.")
    src.add_argument("--append-stand-s", type=float, default=0.0,
                     help="Repeat the last frame of the window for N seconds at the end. "
                          "Useful for letting SONIC settle into the landing pose.")
    src.add_argument("--waist-roll-deg", type=float, default=0.0,
                     help="Overlay a half-sine waist_roll tilt on the motion frames "
                          "(peak in middle, 0 at start/end). Positive vs negative = tilt "
                          "direction; convention depends on URDF (try +10 first, flip if "
                          "wrong way). Use to counter-shift the torso so the support leg "
                          "loads and the swing leg can step. Applies to motion frames "
                          "only, not the prepend.")
    src.add_argument("--waist-pitch-deg", type=float, default=0.0,
                     help="Same as --waist-roll-deg but for waist_pitch (lean forward/back). "
                          "Useful for forward-step momentum priming.")
    # ---- splice (chain a tail window with joint blend) ------------------
    sp = p.add_argument_group("splice (chain a tail window with joint blend)")
    sp.add_argument("--splice-motion-key", type=str, default=None,
                    help="Motion key for the splice tail (default: same as --motion-key). "
                         "Use to attach the source clip's natural settle-to-stop tail "
                         "after a primer window built rest->stride momentum.")
    sp.add_argument("--splice-start", type=int, default=0,
                    help="Start frame of the splice tail (in its source clip).")
    sp.add_argument("--splice-n", type=int, default=None,
                    help="Length of the splice tail in frames (default: to end).")
    sp.add_argument("--splice-blend-frames", type=int, default=6,
                    help="Number of blend frames inserted between primary window and "
                         "splice tail (default 6 = 0.2s @ 30fps). 0 = hard splice.")
    # ---- joint-space cleanup --------------------------------------------
    cl = p.add_argument_group("joint-space cleanup")
    cl.add_argument("--no-leg-back", action="store_true",
                    help="Clamp BOTH legs' hip_pitch joints to >= the baseline "
                         "value (default: source clip's first frame). Prevents "
                         "either foot from sweeping behind the body. Useful for "
                         "killing the scissor/trailing-leg-behind pattern in "
                         "side-step primitives.")
    cl.add_argument("--leg-back-baseline-source-frame", type=int, default=None,
                    help="Source-clip frame whose hip_pitch values are used as the "
                         "no-leg-back baseline. Default: --start frame.")
    args = p.parse_args()

    if args.list:
        bins = joblib.load(args.primitives_pkl)
        # Group by family for easier scanning during locomotion debug.
        rows = []
        for name in sorted(bins.keys()):
            e = bins[name]
            n = int(np.asarray(e["dof"]).shape[0])
            fps = float(e["fps"])
            rows.append((
                str(e.get("recipe_family", "?")),
                name,
                n,
                fps,
                str(e.get("motion_key", "?")),
            ))
        rows.sort()
        cur_fam = None
        for fam, name, n, fps, mk in rows:
            if fam != cur_fam:
                print(f"\n[{fam}]")
                cur_fam = fam
            print(f"  {name:<28} n={n:>4} dur={n/fps:>5.2f}s  src={mk}")
        return 0

    if args.list_source:
        if not args.motion_key:
            p.error("--list-source requires --motion-key <prefix>")
        raw = joblib.load(args.source_pkl)
        matches = sorted(k for k in raw if args.motion_key in k)
        if not matches:
            print(f"no matches for {args.motion_key!r} in {args.source_pkl}")
            return 1
        for k in matches[:30]:
            T = int(np.asarray(raw[k]["dof"]).shape[0])
            fps = float(raw[k].get("fps", 30.0))
            print(f"  {k:<60} n={T:>5} dur={T/fps:>6.2f}s")
        if len(matches) > 30:
            print(f"  ... ({len(matches)-30} more)")
        return 0

    if args.motion_key:
        # Ad-hoc source-window bake (diagnostic).
        slug = f"{args.motion_key}_win{args.start}"
        if args.n is not None:
            slug += f"_{args.start + args.n}"
        out_path = args.out or (_DEFAULT_BAKED_DIR / f"x2_browser_{slug}.pkl")
        info = bake_window_from_source(
            args.motion_key, args.start, args.n,
            args.source_pkl, out_path,
            prepend_stand_s=args.prepend_stand_s,
            append_stand_s=args.append_stand_s,
            prepend_source_frame=args.prepend_source_frame,
            waist_roll_deg=args.waist_roll_deg,
            waist_pitch_deg=args.waist_pitch_deg,
            splice_motion_key=args.splice_motion_key,
            splice_start=args.splice_start,
            splice_n=args.splice_n,
            splice_blend_frames=args.splice_blend_frames,
            no_leg_back=args.no_leg_back,
            leg_back_baseline_source_frame=args.leg_back_baseline_source_frame,
        )
    else:
        if args.bin_name is None:
            p.error("bin_name is required (or pass --list / --motion-key)")
        out_path = args.out or (_DEFAULT_BAKED_DIR / f"x2_browser_{args.bin_name}.pkl")
        info = bake_one(args.bin_name, args.primitives_pkl, out_path)

    print(f"[bake] wrote {info['out_path']}")
    print(f"       bin       = {info['bin']}")
    print(f"       n_frames  = {info['n_frames']}")
    print(f"       fps       = {info['fps']:.2f}")
    print(f"       duration  = {info['duration_s']:.2f} s")
    print(f"       source    = {info['motion_key']}")
    if info["recipe_ops"]:
        print(f"       recipe    = {info['recipe_ops']}")

    try:
        onnx = _resolve_onnx(args.checkpoint)
    except FileNotFoundError as exc:
        print(f"\n[bake] WARN: could not resolve ONNX: {exc}", file=sys.stderr)
        print(
            "\n# Run deploy manually with your ONNX:\n"
            f"bash {_DEPLOY_SH} sim --no-confirm \\\n"
            f"    --motion {info['out_path']} \\\n"
            "    --model <PATH_TO_DEPLOY.onnx>",
            flush=True,
        )
        return 0

    # Cap --max-duration close to the clip length. PklMotionReference
    # *holds the last frame* after the clip ends, so a short locomotion
    # clip (e.g. 2.3 s fwd_step) followed by 12 s of "frozen mid-stride"
    # tells us nothing about the clip and just lets the policy drift.
    # Add a small tail (1 s) so we can see the landing, plus the
    # deploy's --return-seconds (default 2 s) ramp-out.
    clip_dur = float(info["duration_s"])
    max_dur = max(4, int(round(clip_dur + 1.0)))
    print(
        "\n# Test the same primitive through the proven --motion <pkl> path\n"
        "# (PklMotionReference + parity RSI; bypasses our ZMQ publisher).\n"
        "# --sim-viewer launches MuJoCo passive_viewer; close the window or\n"
        "# wait for --max-duration to terminate.\n"
        f"bash {_DEPLOY_SH} sim --no-confirm \\\n"
        f"    --motion {info['out_path']} \\\n"
        f"    --model {onnx} \\\n"
        "    --sim-viewer \\\n"
        f"    --max-duration {max_dur}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
