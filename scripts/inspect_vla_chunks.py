"""Summarise per-inference VLA I/O dumps from the live bridge.

The bridge (``gear_sonic.scripts.live_vla_publish_motion_token``) writes
``chunk_{NNNNN}.npz`` files when ``--dump-chunks-dir`` is set. Each file
contains both the *input* it sent to GR00T (ego camera RGB, body_q in
MJ order, base quat, hand state) and the *output* it produced (motion
token chunk, left/right hand chunks). This script prints a one-line
summary per chunk plus aggregate stats, optionally exporting the ego
frames as a video so we can eyeball what the VLA actually saw.

Use case
--------
The robot stands but doesn't move under live VLA: is the model
producing dead/repeating tokens, or are downstream consumers (deploy
SONIC tracker) ignoring them? This script answers the first half.

Examples
--------

    .venv/bin/python scripts/inspect_vla_chunks.py \\
        /tmp/x2_vla_viewer2/vla_chunks

    .venv/bin/python scripts/inspect_vla_chunks.py \\
        /tmp/x2_vla_viewer2/vla_chunks --first 5 --last 5 \\
        --export-video /tmp/x2_vla_viewer2/vla_egos.mp4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _summarise_chunk(idx: int, path: Path) -> dict:
    d = np.load(path, allow_pickle=False)
    token = d["token"].astype(np.float32)            # (T, 64)
    left = d["left_hand"].astype(np.float32)         # (T, 10)
    right = d["right_hand"].astype(np.float32)       # (T, 10)
    body_q = d["body_q_mj"].astype(np.float32)       # (31,)
    base_q = d["base_quat_wxyz"].astype(np.float32)  # (4,)
    left_obs = d["left_hand_q_obs"].astype(np.float32)
    right_obs = d["right_hand_q_obs"].astype(np.float32)
    elapsed = float(d["elapsed_ms"][0])
    n_inf = int(d["n_inference"][0])
    wall_t = float(d["wall_t_s"][0])

    # First-step vs last-step delta -- a "live" chunk should change
    # along the horizon; a dead chunk repeats the same target.
    token_traj = float(np.linalg.norm(token[-1] - token[0]))
    left_traj = float(np.linalg.norm(left[-1] - left[0]))
    right_traj = float(np.linalg.norm(right[-1] - right[0]))

    return {
        "idx": idx,
        "n_inf": n_inf,
        "wall_t": wall_t,
        "elapsed_ms": elapsed,
        "token_norm0": float(np.linalg.norm(token[0])),
        "token_normL": float(np.linalg.norm(token[-1])),
        "token_traj": token_traj,
        "left_norm0": float(np.linalg.norm(left[0])),
        "left_traj": left_traj,
        "right_norm0": float(np.linalg.norm(right[0])),
        "right_traj": right_traj,
        "body_q_p2p": float(np.ptp(body_q)),
        "base_q": base_q.copy(),
        "left_obs_p2p": float(np.ptp(left_obs)),
        "right_obs_p2p": float(np.ptp(right_obs)),
        "ego_view_shape": tuple(d["ego_view"].shape),
        "_path": path,
    }


def _print_chunk(s: dict) -> None:
    print(
        f"  #{s['n_inf']:04d} t={s['wall_t']:.2f}  "
        f"infer={s['elapsed_ms']:6.1f}ms  "
        f"|tok[0]|={s['token_norm0']:.3f}  "
        f"|tok[-1]|={s['token_normL']:.3f}  "
        f"tok_traj={s['token_traj']:.3f}  "
        f"|L|={s['left_norm0']:.3f}/Δ{s['left_traj']:.3f}  "
        f"|R|={s['right_norm0']:.3f}/Δ{s['right_traj']:.3f}  "
        f"body_p2p={s['body_q_p2p']:.3f}"
    )


def _export_video(chunks: list[dict], out_path: Path, fps: float = 4.0) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print(
            "imageio not available; install via 'pip install imageio[ffmpeg]'"
            " to enable --export-video",
            file=sys.stderr,
        )
        return
    print(f"writing {len(chunks)} ego frames -> {out_path} @ {fps} fps")
    with imageio.get_writer(out_path, fps=fps) as w:
        for s in chunks:
            d = np.load(s["_path"], allow_pickle=False)
            ego = d["ego_view"]
            w.append_data(ego)
    print(f"  wrote {out_path}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump_dir", type=Path)
    p.add_argument("--first", type=int, default=3, help="Pretty-print first N chunks.")
    p.add_argument("--last", type=int, default=3, help="Pretty-print last N chunks.")
    p.add_argument(
        "--export-video", type=Path, default=None,
        help="Optional .mp4 path; concatenates ego_view frames at 4 fps.",
    )
    p.add_argument(
        "--video-fps", type=float, default=4.0,
        help="FPS for --export-video (default 4 = 1 frame per chunk dump).",
    )
    args = p.parse_args()

    if not args.dump_dir.is_dir():
        print(f"not a directory: {args.dump_dir}", file=sys.stderr)
        return 1

    paths = sorted(args.dump_dir.glob("chunk_*.npz"))
    if not paths:
        print(f"no chunk_*.npz files under {args.dump_dir}", file=sys.stderr)
        return 1

    print(f"found {len(paths)} chunk dumps in {args.dump_dir}")
    chunks = [_summarise_chunk(i, p) for i, p in enumerate(paths)]

    if args.first > 0:
        print(f"\nfirst {min(args.first, len(chunks))} chunks:")
        for s in chunks[: args.first]:
            _print_chunk(s)
    if args.last > 0 and len(chunks) > args.first:
        print(f"\nlast {min(args.last, len(chunks) - args.first)} chunks:")
        for s in chunks[-args.last :]:
            _print_chunk(s)

    # Aggregates
    tok_norms = np.array([s["token_norm0"] for s in chunks])
    tok_trajs = np.array([s["token_traj"] for s in chunks])
    left_norms = np.array([s["left_norm0"] for s in chunks])
    right_norms = np.array([s["right_norm0"] for s in chunks])
    elapsed = np.array([s["elapsed_ms"] for s in chunks])
    body_p2p = np.array([s["body_q_p2p"] for s in chunks])

    # Inter-chunk drift in the *output*: how much does the head of
    # consecutive chunks change? If this is ~0 the VLA is rehearsing
    # the same motion every chunk regardless of input.
    inter_token_drift = []
    for a, b in zip(chunks[:-1], chunks[1:]):
        ta = np.load(a["_path"])["token"][0].astype(np.float32)
        tb = np.load(b["_path"])["token"][0].astype(np.float32)
        inter_token_drift.append(float(np.linalg.norm(tb - ta)))
    inter_token_drift_arr = np.array(inter_token_drift) if inter_token_drift else np.array([0.0])

    print("\naggregates over all chunks:")
    print(
        f"  inference time       : mean={elapsed.mean():6.1f} ms"
        f"  median={np.median(elapsed):6.1f} ms  max={elapsed.max():6.1f} ms"
    )
    print(
        f"  |motion_token[0]|    : mean={tok_norms.mean():.3f}"
        f"  min={tok_norms.min():.3f}  max={tok_norms.max():.3f}"
        f"  std={tok_norms.std():.3f}"
    )
    print(
        f"  intra-chunk traj     : mean={tok_trajs.mean():.3f}"
        f"  max={tok_trajs.max():.3f}   (||token[-1]-token[0]||)"
    )
    print(
        f"  inter-chunk drift    : mean={inter_token_drift_arr.mean():.3f}"
        f"  max={inter_token_drift_arr.max():.3f}   (||token_n[0]-token_{{n-1}}[0]||)"
    )
    print(
        f"  |left_hand[0]|       : mean={left_norms.mean():.3f}"
        f"  min={left_norms.min():.3f}  max={left_norms.max():.3f}"
    )
    print(
        f"  |right_hand[0]|      : mean={right_norms.mean():.3f}"
        f"  min={right_norms.min():.3f}  max={right_norms.max():.3f}"
    )
    print(
        f"  input body_q peak2pk : mean={body_p2p.mean():.3f}"
        f"  max={body_p2p.max():.3f}    (rad span across 31 DOF)"
    )

    print("\ninterpretation:")
    if tok_norms.max() < 1e-3:
        print("  ALL tokens ~0 -> bridge stuck on safe-idle (no inference completed?)")
    elif inter_token_drift_arr.max() < 0.05:
        print(
            "  inter-chunk drift ~0 -> VLA is producing the SAME token chunk"
            " every inference, regardless of input (model wedged or not"
            " responding to ego_view changes)"
        )
    elif tok_trajs.max() < 0.05:
        print(
            "  intra-chunk traj ~0 -> tokens within a chunk barely change"
            " (model is asking SONIC to hold pose)"
        )
    else:
        print(
            "  tokens are non-trivial AND change across chunks -> VLA is"
            " producing a live signal; the lack of robot motion is downstream"
            " (SONIC tracker policy, action clipping, or wrist bypass)"
        )

    if args.export_video is not None:
        _export_video(chunks, args.export_video, fps=args.video_fps)

    return 0


if __name__ == "__main__":
    sys.exit(main())
