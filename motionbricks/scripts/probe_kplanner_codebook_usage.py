"""Probe the VQVAE codebook usage during a kplanner replay run.

This is the production-path version of the wandb ``perplexity_pose``
metric. The wandb summary from the VQVAE training run (5b0wayyd)
reported ``perplexity_pose = 9.39`` against a 1024-code codebook --
i.e. the VQVAE collapsed to ~1% of its capacity. This script
verifies that the same collapse manifests at INFERENCE time by
recording every pose token the pose model picks while driving the
kplanner with a real walking clip's velocity intent.

What the probe records:

  - Total tokens emitted across all replans
  - Distinct token ids used (vs. 1024 available)
  - Empirical perplexity = exp(H) where H is entropy of the
    observed code-usage distribution
  - Top-K most-frequently emitted code ids and their fractions
  - Histogram bucketed by 10% of the codebook

For each test clip the probe seeds with the clip's first 4 frames,
drives the planner with the clip's rolling-mean velocity intent
(same logic as ``replay_pkl_through_kplanner.py``), and emits ~5
seconds of motion. The number of unique codes used over that
window is the production-path answer to "is the codebook collapsed
in real inference?".

Usage::

  source .venv/bin/activate && \\
    PYTHONPATH="${PWD}/motionbricks:${PWD}" \\
    python motionbricks/scripts/probe_kplanner_codebook_usage.py
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "motionbricks") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "motionbricks"))


# Reuse the same intent / qpos helpers as the replay script.
from replay_pkl_through_kplanner import (  # noqa: E402
    _build_clip_qpos,
    _instant_intent_from_clip,
)


# ---------------------------------------------------------------------------
# Test clips. Drawn from x2_ultra_locowalk.pkl by inspection. Each chosen
# clip is a long (>=150-frame), un-mirrored exemplar of the intended
# motion category so the rolling-intent yields a clean signal.
# ---------------------------------------------------------------------------

DEFAULT_TEST_CLIPS: dict[str, str] = {
    "fwd_walk":     "Loop_Forward_Walk_001__A018",
    "back_walk":    "Loop_Backward_Walk_001__A020",
    # Turn naming convention is 0=stationary heading, 0090=+90deg
    # (right), 0270=-90deg (left). Both 0090 and 0270 produce roughly
    # the same magnitude of yaw_rate but opposite signs.
    "turn_right":   "Turn_Start_Walk_0090_001__A018",
    "turn_left":    "Turn_Start_Walk_0270_001__A017",
}


def _perplexity(counts: Counter) -> float:
    if not counts:
        return 0.0
    arr = np.array(list(counts.values()), dtype=np.float64)
    probs = arr / arr.sum()
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    return float(np.exp(entropy))


def _summarize_token_usage(
    label: str,
    tokens_3d: np.ndarray,
    nb_code_per_head: int,
    top_k: int = 5,
) -> dict:
    """Summarize per-head AND joint multi-head token usage.

    Args:
        tokens_3d: flattened to (T_total, num_heads) of per-head code ids.

    Per-head perplexity (max = ``nb_code_per_head`` = 10 for this VQVAE)
    tells us how well each head exercises its individual codebook.
    Joint-tuple perplexity (max = ``nb_code_per_head ** num_heads`` =
    1e8) tells us whether the heads vary independently or lock-step.
    """
    if tokens_3d.size == 0:
        return {"label": label, "total": 0, "distinct_joint": 0,
                "perplexity_joint": 0.0, "perplexity_per_head_mean": 0.0,
                "per_head_distinct": [], "top_joint": []}
    if tokens_3d.ndim == 1:
        tokens_3d = tokens_3d[:, None]
    total = int(tokens_3d.shape[0])
    num_heads = int(tokens_3d.shape[1])

    # Per-head metrics.
    per_head_distinct = []
    per_head_perp = []
    for h in range(num_heads):
        c = Counter(int(x) for x in tokens_3d[:, h])
        per_head_distinct.append(len(c))
        per_head_perp.append(_perplexity(c))

    # Joint-tuple metrics: treat the 8-tuple as a single discrete code.
    joint_counts: Counter = Counter(map(tuple, tokens_3d.tolist()))
    top_joint = joint_counts.most_common(top_k)
    top_joint = [(t, v, 100.0 * v / total) for (t, v) in top_joint]

    return {
        "label": label,
        "total": total,
        "num_heads": num_heads,
        "nb_code_per_head": nb_code_per_head,
        "per_head_distinct": per_head_distinct,
        "per_head_perp": per_head_perp,
        "perplexity_per_head_mean": float(np.mean(per_head_perp)),
        "distinct_joint": len(joint_counts),
        "perplexity_joint": _perplexity(joint_counts),
        "top_joint": top_joint,
    }


def _install_token_recorder(inferencer) -> list[np.ndarray]:
    """Monkey-patch _predict_pose_tokens to capture multi-head token ids.

    Returns a list populated with one ndarray per replan; each ndarray
    has shape ``(B, num_tokens, num_heads)`` containing the per-head
    code id selected. Multi-head: each head is an independent
    ``nb_code_per_head``-way categorical.
    """
    captured: list[np.ndarray] = []
    original = inferencer._predict_pose_tokens

    def wrapped(batch, config, info):
        tokens, cond, has_cond = original(batch, config, info)
        try:
            arr = tokens.detach().cpu().numpy()
        except Exception:
            arr = np.asarray(tokens)
        captured.append(arr)
        return tokens, cond, has_cond

    inferencer._predict_pose_tokens = wrapped
    return captured


def _resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if not torch.cuda.is_available():
        print(f"[device] {requested!r} unavailable, falling back to cpu")
        return "cpu"
    try:
        _ = (torch.zeros(1, device=requested) + 1).cpu()
    except RuntimeError as exc:
        print(f"[device] CUDA probe failed ({exc}); cpu")
        return "cpu"
    return requested


def _run_probe_on_clip(
    planner,
    inferencer_token_log: list[np.ndarray],
    qpos_clip: np.ndarray,
    fps: float,
    duration_s: float,
    n_context: int,
    device: str,
) -> np.ndarray:
    """Drive the planner over ``duration_s`` seconds of the clip and
    return all emitted pose tokens flattened to ``(T_total, num_heads)``.
    """
    inferencer_token_log.clear()
    n_predict = int(round(duration_s * fps))
    n_predict = min(n_predict, qpos_clip.shape[0] - n_context)

    seed = torch.from_numpy(qpos_clip[:n_context])
    planner.reset(seed)
    intent0 = _instant_intent_from_clip(qpos_clip, fps, n_context)
    planner.replan_with_velocity(
        torch.tensor(list(intent0), dtype=torch.float32, device=device)
    )

    for i in range(n_predict):
        playback_frame = n_context + i
        if planner.should_replan():
            intent = _instant_intent_from_clip(
                qpos_clip, fps, playback_frame,
            )
            planner.replan_with_velocity(
                torch.tensor(list(intent), dtype=torch.float32, device=device)
            )
        _ = planner.get_next_frame()

    if not inferencer_token_log:
        return np.zeros((0, 1), dtype=np.int64)
    # Each replan returns (B, num_tokens, num_heads). Stack along time.
    parts = []
    for arr in inferencer_token_log:
        a = np.asarray(arr)
        if a.ndim == 2:
            a = a[:, :, None]
        # Squash batch dim and time dim into one axis.
        parts.append(a.reshape(-1, a.shape[-1]))
    return np.concatenate(parts, axis=0).astype(np.int64)


def _print_summary(rows: list[dict], nb_code_per_head: int, num_heads: int) -> None:
    joint_max = nb_code_per_head ** num_heads
    print()
    print("=" * 100)
    print(f"VQVAE multi-head codebook probe  "
          f"(heads = {num_heads}, codes/head = {nb_code_per_head}, "
          f"joint space = {nb_code_per_head}^{num_heads} = {joint_max:.0e})")
    print("=" * 100)

    print("\nPer-head usage (max perplexity per head = "
          f"{nb_code_per_head}; wandb 'perplexity_pose' is the MEAN of these):")
    print(f"{'category':<12} " + "  ".join(
        f"h{i}_perp" for i in range(num_heads)
    ) + "    mean")
    for row in rows:
        cells = "  ".join(f"{p:6.2f}" for p in row["per_head_perp"])
        print(f"{row['label']:<12} {cells}    {row['perplexity_per_head_mean']:6.2f}")

    print("\nPer-head DISTINCT codes used (out of "
          f"{nb_code_per_head}):")
    print(f"{'category':<12} " + "  ".join(
        f"h{i}" for i in range(num_heads)
    ))
    for row in rows:
        cells = "  ".join(f"{d:>2d}" for d in row["per_head_distinct"])
        print(f"{row['label']:<12} {cells}")

    print("\nJoint multi-head tuple usage  "
          f"(distinct 8-tuples observed; joint space = {joint_max:.0e}):")
    print(f"{'category':<12} {'tokens':>8} {'distinct_tuples':>16} "
          f"{'joint_perp':>12}  top 3 tuples (%)")
    for row in rows:
        top_str = "  ".join(
            f"{t}={pct:.1f}%" for (t, _v, pct) in row["top_joint"][:3]
        )
        print(
            f"{row['label']:<12} {row['total']:>8d} "
            f"{row['distinct_joint']:>16d} {row['perplexity_joint']:>12.1f}  "
            f"{top_str}"
        )

    print("\nCross-intent overlap (top-K joint tuples shared across categories):")
    all_sets = [
        set(tuple(t) for t, _, _ in r["top_joint"])
        for r in rows if r["distinct_joint"] > 0
    ]
    if all_sets:
        common = set.intersection(*all_sets) if len(all_sets) > 1 else all_sets[0]
        union = set.union(*all_sets)
        print(f"  union of top-K tuples across categories: {len(union)}")
        print(f"  common to ALL categories               : {len(common)}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--motion-lib-pkl", type=Path,
        default=REPO_ROOT / "gear_sonic" / "data" / "motions"
                / "x2_ultra_locowalk.pkl",
    )
    p.add_argument(
        "--duration", type=float, default=5.0,
        help="seconds of motion to probe per clip",
    )
    p.add_argument(
        "--n-context", type=int, default=4,
        help="seed-window length in frames",
    )
    p.add_argument(
        "--clip-overrides", type=str, default=None,
        help="comma-separated label=key pairs to override the default clip "
             "selection, e.g. fwd_walk=Loop_Forward_Walk_001__A018",
    )
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    overrides = dict(DEFAULT_TEST_CLIPS)
    if args.clip_overrides:
        for piece in args.clip_overrides.split(","):
            if "=" in piece:
                lbl, key = piece.split("=", 1)
                overrides[lbl.strip()] = key.strip()

    print(f"PKL                   = {args.motion_lib_pkl}")
    print(f"duration / category   = {args.duration:.1f}s")
    print(f"n_context             = {args.n_context}")
    print(f"test categories       = {list(overrides.keys())}")

    device = _resolve_device(args.device)
    print(f"device                = {device}")

    raw = joblib.load(args.motion_lib_pkl)

    from motionbricks.motion_backbone.inference.load_x2_planner import (
        X2PlannerPaths, load_x2_planner,
    )
    paths = X2PlannerPaths.default()
    print(f"vqvae_ckpt            = {paths.vqvae_ckpt}")
    print(f"pose_ckpt             = {paths.pose_ckpt}")
    print(f"root_ckpt             = {paths.root_ckpt}")

    planner = load_x2_planner(paths, device=device, replan_threshold_frames=16)

    # Discover the actual multi-head codebook shape from the loaded
    # quantizer. The stored tensor is _codebook.embed with shape
    # (num_heads, nb_code_per_head, code_dim_per_head).
    pose_vqvae = planner._inferencer._vqvae_pose_model
    embed_shape = None
    try:
        embed_shape = tuple(
            pose_vqvae.quantizer.vq._codebook.embed.shape
        )
    except Exception:
        try:
            embed_shape = tuple(
                pose_vqvae.quantizer.codebook.shape
            )
        except Exception:
            embed_shape = None
    if embed_shape is not None and len(embed_shape) == 3:
        num_heads = int(embed_shape[0])
        nb_code_per_head = int(embed_shape[1])
    else:
        num_heads = 8
        nb_code_per_head = 10
    joint_max = nb_code_per_head ** num_heads
    print(f"num_heads             = {num_heads}")
    print(f"nb_code_per_head      = {nb_code_per_head}")
    print(f"joint code space      = {joint_max:.0e}  "
          f"(= {nb_code_per_head}^{num_heads})")

    token_log = _install_token_recorder(planner._inferencer)

    rows = []
    for label, key in overrides.items():
        if key not in raw:
            print(f"[skip] {label}: clip {key!r} not in PKL")
            continue
        qpos_clip, fps = _build_clip_qpos(raw[key])
        if qpos_clip.shape[0] < args.n_context + 8:
            print(f"[skip] {label}: clip too short")
            continue
        intent0 = _instant_intent_from_clip(qpos_clip, fps, args.n_context)
        print(f"\n--- {label} = {key} (fps={fps:.0f}, T={qpos_clip.shape[0]}) ---")
        print(f"     intent@t=0   = (yaw_rate={intent0[0]:+.3f}, "
              f"vel_x={intent0[1]:+.3f}, vel_z={intent0[2]:+.3f}, "
              f"hip_h={intent0[3]:.3f})")
        tokens = _run_probe_on_clip(
            planner, token_log, qpos_clip, fps, args.duration,
            args.n_context, device,
        )
        summary = _summarize_token_usage(label, tokens, nb_code_per_head)
        print(f"     positions emitted = {summary['total']},  "
              f"distinct joint tuples = {summary['distinct_joint']},  "
              f"joint perplexity = {summary['perplexity_joint']:.1f},  "
              f"mean per-head perp = "
              f"{summary['perplexity_per_head_mean']:.2f}/"
              f"{nb_code_per_head}")
        rows.append(summary)

    if rows:
        _print_summary(rows, nb_code_per_head, num_heads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
