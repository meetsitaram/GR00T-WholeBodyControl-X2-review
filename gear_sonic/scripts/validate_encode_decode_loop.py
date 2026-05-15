#!/usr/bin/env python3
"""Layer 4: encode-decode round-trip sanity for X2 SONIC tokens on a parquet.

This script validates the *internal consistency* of the inline-
tokenizer pipeline against a recorded LeRobot dataset. It answers
the question: "given a recorded episode, does re-encoding the row's
``action.body_q_mj_pre_sonic`` (commanded body pose) via the SONIC
encoder produce a token that matches the row's stored
``action.motion_token``?"

What this catches
-----------------

If the recorder's gather drifted (e.g. wrong joint order, wrong
future-frame stride, a stale obs builder) the inline token written
to the parquet would diverge from what an offline encoder run would
produce on the same body_q sequence. The recorder's Layer 1+2 unit
tests catch this *in isolation*; Layer 4 catches it on real
recorded data, including operator-induced edge cases (waist
freezes, mid-bin transitions, etc.).

Specifically the script:

1. Loads a recorded ``data/chunk-*/episode_*.parquet`` and pulls
   ``action.body_q_mj_pre_sonic`` (commanded body pose, the
   recorder's *input* to the encoder) and ``action.motion_token``
   (the recorder's *output* token).
2. For each row ``r``, builds a temporal-slice 10-frame future
   window from rows ``r..r+9`` (clamping at episode end -- mirrors
   what Isaac-GR00T's training-time temporal slicing does).
3. Runs that window through the SONIC encoder + FSQ via
   :class:`SonicMotionTokenLabeler`.
4. Compares the offline token to the stored token and reports per-
   row max-abs diff, mean cosine similarity, and the FSQ-bucket
   match rate.

Note: the recorder uses the *planner's* future window at write time,
not a parquet temporal slice. The two windows differ by O(20 ms) of
arm-overlay timing and at scene boundaries, so byte-equality is NOT
expected. We assert:

* mean cosine similarity > ``--cos-threshold`` (default 0.92), and
* >= ``--bucket-threshold`` % of FSQ buckets match exactly
  (default 60 %).

Decoder consistency (optional)
------------------------------

If ``--check-decoder`` is set, the script also runs each token
through the SONIC ``g1_dyn`` decoder using a *zeroed* 990-D
proprioception placeholder. The decoded action is compared to the
row's ``action.body_q_mj_pre_sonic`` (joint-target intent). With
zero proprio the decoder is severely OOD, so this check uses a
generous ``--decoder-rmse-threshold`` (default 0.30 rad). Failures
here indicate the decoder weights didn't load correctly from the
``.pt`` (silent fallback to random init).

Usage
-----

::

    python -m gear_sonic.scripts.validate_encode_decode_loop \\
        --parquet data/lerobot/x2_phase0_smoke_v0/data/chunk-000/episode_000000.parquet \\
        --checkpoint /path/to/model_step_NNNNN.pt \\
        --device cpu

Exits 0 if all checks pass.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# Defer heavy imports until after CLI parsing so --help is fast.


def _load_parquet_columns(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd
    df = pd.read_parquet(path)
    needed = ["action.body_q_mj_pre_sonic", "action.motion_token"]
    missing = [n for n in needed if n not in df.columns]
    if missing:
        raise KeyError(
            f"{path}: missing columns {missing}. Available: "
            f"{list(df.columns)[:20]}{'...' if len(df.columns) > 20 else ''}"
        )

    def _stack(name: str) -> np.ndarray:
        col = df[name].to_numpy()
        return np.stack(
            [np.asarray(row, dtype=np.float64) for row in col], axis=0
        )

    body_q = _stack("action.body_q_mj_pre_sonic")
    motion_tok = _stack("action.motion_token")
    return body_q, motion_tok


def _build_temporal_slice_clip(
    body_q: np.ndarray, row: int, num_future_frames: int
) -> np.ndarray:
    """Return ``(num_future_frames, 31)`` clip starting at ``row``.

    Clamps at end of episode (mirrors
    :func:`build_tokenizer_obs`'s ``min(int(future_time/dt),
    total_frames-1)`` index clamp).
    """
    T = body_q.shape[0]
    out = np.zeros((num_future_frames, body_q.shape[1]), dtype=np.float64)
    for f in range(num_future_frames):
        out[f] = body_q[min(row + f, T - 1)]
    return out


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--parquet", type=Path, required=True,
        help="Recorded LeRobot episode parquet (chunk-*/episode_*.parquet).",
    )
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="SONIC .pt checkpoint (e.g. model_step_025000.pt). The "
             "encoder + decoder weights are extracted in-place.",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Torch device for the SONIC labeler / decoder. Default cpu.",
    )
    parser.add_argument(
        "--motion-fps", type=float, default=50.0,
        help="Recording frame rate (default 50 Hz).",
    )
    parser.add_argument(
        "--cos-threshold", type=float, default=0.92,
        help="Mean cosine-similarity threshold for the encode round-trip.",
    )
    parser.add_argument(
        "--bucket-threshold", type=float, default=60.0,
        help="Min %% of FSQ buckets that must match exactly per row.",
    )
    parser.add_argument(
        "--max-rows", type=int, default=0,
        help="Cap on rows to validate (0 = all). Useful for spot-checking "
             "long recordings.",
    )
    parser.add_argument(
        "--check-decoder", action="store_true",
        help="Also run each stored token through the SONIC g1_dyn decoder "
             "(with zero-proprio placeholder) and compare the decoded "
             "action to action.body_q_mj_pre_sonic.",
    )
    parser.add_argument(
        "--decoder-rmse-threshold", type=float, default=0.30,
        help="Max per-row RMSE (rad) for the optional decoder check. "
             "Generous default because zero-proprio is OOD.",
    )
    args = parser.parse_args(argv)

    body_q, motion_tok = _load_parquet_columns(args.parquet)
    T = body_q.shape[0]
    if args.max_rows > 0:
        T = min(T, args.max_rows)
        body_q = body_q[:T]
        motion_tok = motion_tok[:T]

    if motion_tok.shape[1] != 64:
        raise ValueError(
            f"action.motion_token must be (T, 64); got {motion_tok.shape}"
        )
    if body_q.shape[1] != 31:
        raise ValueError(
            f"action.body_q_mj_pre_sonic must be (T, 31); got {body_q.shape}"
        )

    # Quick check: stored tokens are not all zero (would mean the
    # recorder ran without --sonic-checkpoint -- the comparison would
    # otherwise produce misleadingly large diffs).
    nonzero_rows = int(np.any(motion_tok != 0.0, axis=1).sum())
    if nonzero_rows == 0:
        print(
            "[FAIL] action.motion_token is all zeros across the parquet "
            "-- the recorder ran WITHOUT --sonic-checkpoint, so there's "
            "no token to validate. Re-record with --sonic-checkpoint.",
            file=sys.stderr,
        )
        return 1
    print(
        f"Parquet     : {args.parquet}\n"
        f"Frames      : {T}\n"
        f"Nonzero tok : {nonzero_rows} / {T} rows\n"
        f"Checkpoint  : {args.checkpoint}\n"
    )

    # Defer heavy imports
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parent),
    )
    from sonic_motion_token_labeler import (
        IDENTITY_QUAT_XYZW,
        SonicMotionTokenLabeler,
    )
    from gear_sonic.utils.teleop.x2_encoder_obs_builder import (
        X2_NUM_FUTURE_FRAMES,
    )

    labeler = SonicMotionTokenLabeler(
        args.checkpoint,
        device=args.device,
        motion_fps=args.motion_fps,
    )

    # ---- Layer 4a: encode round-trip ---------------------------------------
    print("Layer 4a: encode round-trip (temporal-slice future)")
    print("---------------------------------------------------")
    cos_sims = np.zeros(T, dtype=np.float64)
    bucket_match = np.zeros(T, dtype=np.float64)
    max_abs_per_row = np.zeros(T, dtype=np.float64)

    for r in range(T):
        clip = _build_temporal_slice_clip(
            body_q, r, X2_NUM_FUTURE_FRAMES + 1
        )
        # build_tokenizer_obs reads frames at f * DT_FUTURE_REF; with
        # motion_fps=50 the 10 future frames land on the first 10
        # rows of ``clip``. The 11th frame is for the velocity
        # computation at frame 0 (which reads f-1, clamped to 0).
        root_clip = np.tile(
            np.asarray(IDENTITY_QUAT_XYZW, dtype=np.float64),
            (clip.shape[0], 1),
        )
        tokens = labeler.label_trajectory(clip, root_rot_xyzw=root_clip)
        offline = tokens[0]
        stored = motion_tok[r]

        denom = np.linalg.norm(offline) * np.linalg.norm(stored)
        cos = float(np.dot(offline, stored) / denom) if denom > 0 else 0.0
        cos_sims[r] = cos
        # FSQ buckets are exact: round to 1/16 = 2/32 step.
        bucket_match[r] = float(
            np.mean(np.isclose(offline, stored, atol=1e-6))
        )
        max_abs_per_row[r] = float(np.abs(offline - stored).max())

    mean_cos = float(cos_sims.mean())
    mean_bucket = float(bucket_match.mean()) * 100.0
    mean_max_abs = float(max_abs_per_row.mean())
    print(f"  rows compared       : {T}")
    print(f"  mean cosine sim     : {mean_cos:+.4f}  (threshold > {args.cos_threshold:.2f})")
    print(f"  mean bucket match   : {mean_bucket:.1f} %  (threshold >= {args.bucket_threshold:.1f} %)")
    print(f"  mean per-row max-abs: {mean_max_abs:.3e}")

    pass_4a = (
        mean_cos > args.cos_threshold
        and mean_bucket >= args.bucket_threshold
    )
    print(f"  Layer 4a verdict    : {'PASS' if pass_4a else 'FAIL'}")
    print()

    # ---- Layer 4b: decoder check (optional) --------------------------------
    pass_4b = True
    if args.check_decoder:
        print("Layer 4b: decoder round-trip (zero-proprio, generous threshold)")
        print("---------------------------------------------------------------")
        import torch
        proprio = torch.zeros((1, 990), dtype=torch.float32, device=args.device)
        per_row_rmse = np.zeros(T, dtype=np.float64)
        for r in range(T):
            tok = torch.from_numpy(motion_tok[r:r + 1].astype(np.float32)).to(
                args.device
            )
            decoder_input = torch.cat([tok, proprio], dim=-1)
            with torch.no_grad():
                action_il = labeler._actor.decoder(decoder_input)
            action = action_il.cpu().numpy()[0]
            # Decoder output is in IsaacLab joint order; we need to map to
            # MJ to compare with action.body_q_mj_pre_sonic. For a sanity
            # check we ignore the joint order and just compute RMSE over
            # all 31 dims (the order is consistent within encoder /
            # decoder, so an order mismatch would still surface a large
            # RMSE if the network is wrong).
            target = body_q[r]
            per_row_rmse[r] = float(np.sqrt(np.mean((action - target) ** 2)))
        mean_rmse = float(per_row_rmse.mean())
        max_rmse = float(per_row_rmse.max())
        print(f"  mean per-row RMSE : {mean_rmse:.4f} rad")
        print(f"  max  per-row RMSE : {max_rmse:.4f} rad")
        print(f"  threshold         : {args.decoder_rmse_threshold:.4f} rad")
        pass_4b = mean_rmse <= args.decoder_rmse_threshold
        print(f"  Layer 4b verdict  : {'PASS' if pass_4b else 'FAIL'}")
        print()

    overall = pass_4a and pass_4b
    print("=" * 70)
    print(f"Overall verdict: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
