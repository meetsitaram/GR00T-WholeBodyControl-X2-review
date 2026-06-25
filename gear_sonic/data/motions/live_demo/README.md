# Live demo motion snapshot

Deploy-ready stitched motion PKLs used during live-demo runs of the
fine-tuned SONIC X2 Ultra policies. Each file is a **copy** of the
canonical source under `gear_sonic/data/motions/x2_ultra_*.pkl` (those
canonical files are gitignored as regenerable build outputs). The copies
live here so that a fresh clone of this branch is self-contained for a
demo without re-running the warehouse / chain-matched motion builders.

Companion docs:

- [`../demo_v1_sources/README.md`](../demo_v1_sources/README.md) — fine-tune
  motion staging for the demo_v1 / demo_v2 SONIC training corpus.
- [`../demo_v1_sources/SKILL.md`](../demo_v1_sources/SKILL.md) — operator
  playbook to spin up a new SONIC fine-tune on this corpus.
- [`../demo_v1_sources/ITERATE.md`](../demo_v1_sources/ITERATE.md) — what
  to change to go from one fine-tune to the next (demo_v1 → demo_v2).
- [`../../../../docs/source/user_guide/finetune-x2-on-new-corpus.md`](../../../../docs/source/user_guide/finetune-x2-on-new-corpus.md)
  — higher-level fine-tune walkthrough.
- [`../../../../play_pkl_motions_commands.md`](../../../../play_pkl_motions_commands.md)
  — runtime cheatsheet (which ONNX × which PKL).

## Contents

All 30 fps, single-key stitched-loop PKLs (motion_lib schema). Each
already has the home-loop choreography baked in (anchor → walk → pivot →
home), so you can pass any one of these straight to
`gear_sonic_deploy/deploy_x2.sh sim/real --motion <pkl>`.

| File | Frames | Duration | What it is |
|---|---:|---:|---|
| `x2_ultra_in_place_turns_v1_chain_matched.pkl` | — | — | In-place turn primitives (chain-matched retarget). |
| `x2_ultra_walk_demo_v6.pkl` | 1325 | 44.2 s | v6 4-pivot home-loop (walk → +90 → walk → +90 → return → +90 → return → +90), original (legacy) retargeter. |
| `x2_ultra_walk_demo_v6_chain_matched.pkl` | 1325 | 44.2 s | Same v6 choreography, rebuilt with the new SOMA chain-matched retargeter (`soma_to_x2_ultra_chain_matched_retargeter_config.json`). |
| `x2_ultra_relaxed_walk_loop_v1.pkl` | 1565 | 52.2 s | Closed-loop relaxed walk (uniform_h14 retarget): anchor → 90° pivot → 3.5 m out → 180° pivot → 3.5 m back → 90° pivot → anchor. Lands within ~21 cm of origin. |
| `x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl` | 1565 | 52.2 s | Same closed-loop choreography, chain-matched retarget. |
| `x2_ultra_relaxed_walk_loop_v1_halfspeed_walks.pkl` | 1941 | 64.7 s | v1 closed-loop with 0.5× speed multipliers on the walk segments (pivots/anchors at full speed); uniform_h14 retarget. |
| `x2_ultra_relaxed_walk_loop_v1_halfspeed_walks_chain_matched.pkl` | 1941 | 64.7 s | Halfspeed walks variant, chain-matched retarget. |

Dance / gesture motions for the demo live elsewhere and stay there
because they're already tracked at their canonical locations:

- Dances: `gear_sonic/data/motions/dance_singles/dance_*__A*.pkl`
  (shortlist below; full 34-clip bundle is
  `gear_sonic/data/motions/x2_ultra_dances.pkl`, gitignored).
- MC gestures: `gear_sonic/data/motions/x2_recorded/mc_gestures/*.pkl`
  (51 clips, tracked in repo).

## Demo policies (lives on PC2 at `/home/run/getsolo/policies/`)

| Policy slot | Behavior | Best for | Source (in this repo) |
|---|---|---|---|
| `agibot_x2_sonic_base_version.onnx` | H200 25k sphere-feet baseline | clean fallback | `/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx` |
| `agibot_x2_sonic.onnx` | demo_v1 fine-tune (model_step_004000) | walking + general | `logs_rl/TRL_X2Ultra_DemoV1/.../sonic_x2_ultra_demo_v1_demo_v1-20260623_231221/exported/model_step_004000_g1.onnx` |
| `agibot_x2_sonic_dance_fine_tuned.onnx` | demo_v2 fine-tune (model_step_004000, 2× obs noise + ±15 % KP/KD DR) | dances + gestures | `logs_rl/TRL_X2Ultra_DemoV2/.../sonic_x2_ultra_demo_v2_demo_v2-20260624_130315/exported/model_step_004000_g1.onnx` |

## Demo shortlists

```
walks (demo day)
  in_place_turns_v1_chain_matched
  walk_demo_v6 / walk_demo_v6_chain_matched
  relaxed_walk_loop_v1 / chain_matched / halfspeed{,_chain_matched}

dances (demo day)
  easy   - dance_singles/dance_hiphop_stick_n_roll_dancehall_R_loop_003__A324.pkl
  medium - dance_singles/dance_western_horse_step_with_leg_undercut_R_loop_002__A324.pkl
  hard   - dance_singles/dance_latino_kick_kick_R_001__A313.pkl

gestures (demo day)
  e.g.   - gear_sonic/data/motions/x2_recorded/mc_gestures/left_kiss_001.pkl
```

## Quick deploy examples

Local MuJoCo sim parity:

```bash
# walk — use demo_v1 (or base) policy
gear_sonic_deploy/deploy_x2.sh sim \
    --model logs_rl/TRL_X2Ultra_DemoV1/manager/universal_token/all_modes/sonic_x2_ultra_demo_v1_demo_v1-20260623_231221/exported/model_step_004000_g1.onnx \
    --motion gear_sonic/data/motions/live_demo/x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl \
    --sim-viewer --no-confirm \
    --max-duration 60

# dance — use demo_v2 policy
gear_sonic_deploy/deploy_x2.sh sim \
    --model logs_rl/TRL_X2Ultra_DemoV2/manager/universal_token/all_modes/sonic_x2_ultra_demo_v2_demo_v2-20260624_130315/exported/model_step_004000_g1.onnx \
    --motion gear_sonic/data/motions/dance_singles/dance_latino_kick_kick_R_001__A313.pkl \
    --sim-viewer --no-confirm \
    --max-duration 15
```

Real robot (PC2):

```bash
# walk-leaning policy
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

# dance-fine-tuned policy (swap --model only)
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic_dance_fine_tuned.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight
```

> If you see false-positive `SAFE_HOLD` trips on the dance policy during
> kicks/turns, the `walking_recovery.yaml` tilt thresholds may be too
> tight for explosive dance moves — relax `--tilt-cos` first.

## Regenerating these files

Each PKL is rebuilt from a playlist YAML via the warehouse-stitcher path
(`gear_sonic/utils/motion_lib/make_warehouse_motion.py`). The playlists
live at `../playlists/<base_name>.yaml`. To rebuild after a retargeter or
choreography change:

```bash
# example for the relaxed-walk closed loop
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/utils/motion_lib/make_warehouse_motion.py \
        --playlist gear_sonic/data/motions/playlists/relaxed_walk_loop_v1_chain_matched.yaml \
        --out      gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl
# then copy the new PKL over the snapshot in this dir:
cp gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl \
   gear_sonic/data/motions/live_demo/
```

The canonical source under `gear_sonic/data/motions/x2_ultra_*.pkl` is
gitignored (`*.pkl` rule at line 264 of `.gitignore`), but
`live_demo/*.pkl` is **not** caught by that rule because the gitignore
pattern is `gear_sonic/data/motions/*.pkl` (top-level only, not
recursive).
