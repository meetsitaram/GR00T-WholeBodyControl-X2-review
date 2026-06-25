# 2026-06-23 — Motion dataset rebuild on chain_matched + rest-pose match for stitched loops

> **Status:** all artifacts landed on disk; previewed in MuJoCo kinematic
> viewer. No SONIC sim / real-robot validation yet — these are reference
> motions that get consumed downstream by `make_warehouse_motion.py`,
> `deploy_x2.sh sim --motion`, and the SONIC training data pipeline.

> **Session focus.** Three intertwined data-production threads,
> sharing the same retarget pipeline and the same stitcher
> (`make_warehouse_motion.py` + `_warehouse_playlist.py`):
>
> 1. **Re-retarget the walk primitives on the corrected `chain_matched`
>    config.** A separate session committed
>    `soma_to_x2_ultra_chain_matched_retargeter_config.json` to the
>    `soma-retargeter` repo to fix the "Groucho-Marx" deep-crouch bug
>    (pelvis Z floored at ~0.51 m, knees jammed at ~70°) that the
>    earlier scratch `chain_matched_v4` config produced. This milestone
>    pulled that config, re-retargeted every walk-related BVH on both
>    tracks (`uniform_h14` and `chain_matched`), rebuilt the bundle
>    PKLs, and regenerated all stitched walk loops that source from
>    them.
> 2. **Fix the visible "snap to default pose between segments"** that
>    every stitched loop showed at its rest seams. Root cause: the
>    legacy rest pose (`x2_ultra_stitched_idle_relaxed_arms.pkl`) was
>    built from `idle_hands_on_back_loop_001` retargeted by an older
>    config, so wrists were ~80° rotated relative to the new-retarget
>    walk poses. Cure: build two new track-matched rest PKLs and
>    re-point the relevant playlists at them.
> 3. **Add a 3-clip sit/stand primitive bundle on `chain_matched`,
>    packaged at 1.0× and 0.5× playback.** Three BVHs
>    (`sit_on_chair_start/loop/stop_R_*__A244`) retargeted on the
>    chain_matched track, merged into one 6-entry bundle PKL using the
>    same `__speed_X.X` key convention as the walks bundle.
>
> No infrastructure code was touched in this session. The stitcher,
> stitched-rest builder, and bundle builder all date back to early May
> 2026 — this session is data production only. The one upstream code
> dependency (`soma_to_x2_ultra_chain_matched_retargeter_config.json`)
> was authored in a separate session and merely picked up here.

---

## TL;DR

| Thread | Before | After |
|---|---|---|
| Walk-primitive retarget on `chain_matched` | Used scratch `_v4` scaler with uniform 0.6732 leg ratios → pelvis Z 0.509 m / knee flex 70° (permanent Groucho crouch) | Committed config with per-segment leg scales (LeftLeg=0.541, LeftShin=0.81, LeftFoot=0.75, LeftToe=0.78, LeftToeBase=0.66) → pelvis Z 0.634-0.681 m / knee flex ≤23° at idle, walks back to natural 3.50 m per stride |
| Stitched-loop rest seam | Each rest seam SLERPed wrists ~80° to "hands-on-back" pose and back, in addition to whatever pose alignment the legs needed. Anchor seams traversed 0.30 rad of leg motion before holding. | Per-track rest PKLs match each segment's track. Wrist L2 vs walk-end mid-rest drops 22% (0.40 → 0.31 rad). Anchor seams now traverse 0.012 rad lower-body (-96%) because the anchor segments and the rest source share `neutral_idle_loop_002__A074`. |
| Sit/stand bundle | None — the 3 chair sit/stand CSVs existed in `soma-retargeter/scratch/sitstand_chain_matched/` from an earlier retarget pass but had no motion-lib PKL packaging. | `x2_ultra_sitstand_chain_matched.pkl` (6 entries = 3 clips × {1.0×, 0.5×}), 30 fps, ready to drop into any future `chair_visit_v*.yaml` playlist. Pelvis-Z verifies semantics: start descends 0.68 → 0.39 m (sitting down), loop holds 0.39 m (sit still), stop rises 0.39 → 0.68 m (stand up). |

---

## What got generated

### 1. Source retarget CSVs (44 new — 22 walk-related BVHs × 2 tracks)

Output dirs:

```
agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/
├── chain_matched/   *.csv (22 files)
└── uniform_h14/     *.csv (22 files)
```

Per-track contents include the walk_forward primitives we added today
(`walk_forward_loop_{001,002,003}__A021{,_M}` × 6 BVHs) plus the
pre-existing `Relaxed_walk_forward_*`, `idle_turn_*`,
`neutral_idle_loop_002__A074`, `neutral_walk_180_R_002__A105`,
`step_rotate_idle_090_*`, `Turn_Start_Walk_*` primitives.

The chain_matched outputs were produced by the corrected scaler config
([`soma_to_x2_ultra_chain_matched_retargeter_config.json`](https://github.com/) committed 2026-06-23 21:02), invoked via
`scripts/retarget_one.py --retargeter-config <CFG>`. The uniform_h14
outputs were produced with `--model-height 1.40` against the default
scaler.

```bash
PY=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter/.venv/bin/python
CFG_CHAIN=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter/soma_retargeter/configs/agibot_x2_ultra/soma_to_x2_ultra_chain_matched_retargeter_config.json
BVH_DIR=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed/extracted/_retarget_compare
OUT_CHAIN=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/chain_matched
OUT_UNI=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/uniform_h14
cd $HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter
for BVH in "$BVH_DIR"/*.bvh ; do
  name=$(basename "$BVH" .bvh)
  "$PY" scripts/retarget_one.py --retargeter-config "$CFG_CHAIN" --bvh "$BVH" --out "$OUT_CHAIN/${name}.csv"
  "$PY" scripts/retarget_one.py --model-height 1.40              --bvh "$BVH" --out "$OUT_UNI/${name}.csv"
done
```

The 3 sit/stand CSVs live at
`agibot-x2-references/soma-retargeter/scratch/sitstand_chain_matched/sit_on_chair_{start_R_001,loop_R_002,stop_R_002}__A244__x2_chain_matched.csv`
(retargeted earlier in the day before this session).

### 2. Bundle PKLs (3 new files, 138 total motion entries)

| PKL | Entries | Composition |
|---|---|---|
| `gear_sonic/data/motions/x2_ultra_retarget_uniform_h14.pkl` | **66** | 22 clips × 3 speeds (`__speed_{1.0, 0.5, 0.25}`) via `--fps-source {120,60,30}` decimation tricks |
| `gear_sonic/data/motions/x2_ultra_retarget_chain_matched.pkl` | **66** | same shape, chain_matched track |
| `gear_sonic/data/motions/x2_ultra_sitstand_chain_matched.pkl` | **6** | 3 clips × 2 speeds (`__speed_{1.0, 0.5}`) — sit DOWN, sit STILL, stand UP |

Bundles built via `gear_sonic/data_process/build_x2_bones_seed_motion_lib.py`
(one invocation per speed; merged into the final bundle by a 10-line
post-process that suffixes each key with `__speed_X.X`).

### 3. Track-matched rest PKLs (2 new — for use as `rest.source` in stitching YAMLs)

Built with `gear_sonic/scripts/make_stitched_motion.py` using the body
partition (lower from `neutral_idle_loop_002__A074`, upper from
`Relaxed_walk_forward_002__A057`), sourced from each new bundle:

| PKL | Lower-body source | Upper-body source |
|---|---|---|
| `x2_ultra_stitched_idle_relaxed_arms_uniform_h14.pkl` | `neutral_idle_loop_002__A074__speed_1.0` (uniform_h14) | `Relaxed_walk_forward_002__A057__speed_1.0` (uniform_h14) |
| `x2_ultra_stitched_idle_relaxed_arms_chain_matched.pkl` | `neutral_idle_loop_002__A074__speed_1.0` (chain_matched) | `Relaxed_walk_forward_002__A057__speed_1.0` (chain_matched) |

The legacy `x2_ultra_stitched_idle_relaxed_arms.pkl` is **kept** for
back-compat — older playlists (`walk_demo_v1..v5`, `casual_walk_v*`,
`warehouse_v*`, `showcase_v1`, `standing_gestures_v1`, `one_foot_v*`,
`minimal_v1`, `walk_demo_v6.yaml` original) still reference it and
their segment-end poses match its older retarget.

### 4. Stitched closed-loop walk PKLs (8 rebuilt with the new rest pose)

| PKL | Layout | Walk clip | Per-walk dist | Duration | Closure XY / yaw |
|---|---|---|---|---|---|
| `x2_ultra_walk_loop_v1_uniform_h14.pkl` | 4-pivot rectangle | `walk_forward_loop_001__A021` | ~3.30 m | 50.17 s | (-0.30, +0.10) m / -5.4° |
| `x2_ultra_walk_loop_v1_chain_matched.pkl` | 4-pivot rectangle | `walk_forward_loop_001__A021` | ~3.49 m | 50.17 s | (-0.44, +0.12) m / -9.0° |
| `x2_ultra_walk_demo_v6_uniform_h14.pkl` | 4-pivot rectangle | `neutral_walk_180_R_002__A105` (68 fr) | ~0.85 m | 44.17 s | (-0.04, +0.09) m / +4.6° |
| `x2_ultra_walk_demo_v6_chain_matched.pkl` | 4-pivot rectangle | `neutral_walk_180_R_002__A105` (68 fr) | ~0.90 m | 44.17 s | (-0.11, +0.11) m / **+2.4°** ← tightest yaw closure of any loop |
| `x2_ultra_relaxed_walk_loop_v1.pkl` | 4-pivot rectangle | `Relaxed_walk_forward_001__A057` (188 fr) | ~3.50 m | 52.17 s | (-0.21, +0.03) m / -2.1° |
| `x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl` | 4-pivot rectangle | `Relaxed_walk_forward_001__A057` (188 fr) | ~3.50 m | 52.17 s | (-0.47, -0.01) m / -9.7° |
| `x2_ultra_relaxed_walk_loop_v1_halfspeed_walks.pkl` | 4-pivot rectangle | `Relaxed_walk_forward_001__A057__speed_0.5` | ~3.50 m, 0.5× walks | 64.70 s | (-0.20, +0.04) m / -1.8° |
| `x2_ultra_relaxed_walk_loop_v1_halfspeed_walks_chain_matched.pkl` | 4-pivot rectangle | `Relaxed_walk_forward_001__A057__speed_0.5` | ~3.50 m, 0.5× walks | 64.70 s | (-0.47, -0.01) m / -9.6° |

### 5. YAML playlists touched (8 — `rest.source` rewires + walk-loop additions)

```
gear_sonic/data/motions/playlists/
├── walk_loop_v1_uniform_h14.yaml                       (NEW — regular walk 4-pivot rectangle, uniform_h14)
├── walk_loop_v1_chain_matched.yaml                     (NEW — same, chain_matched)
├── walk_demo_v6_uniform_h14.yaml                       (NEW — v6 short-step 4-pivot rectangle, uniform_h14)
├── walk_demo_v6_chain_matched.yaml                     (NEW — same, chain_matched)
├── relaxed_walk_loop_v1.yaml                           (rest.source rewired to *_uniform_h14.pkl)
├── relaxed_walk_loop_v1_chain_matched.yaml             (rest.source rewired to *_chain_matched.pkl)
├── relaxed_walk_loop_v1_halfspeed_walks.yaml           (rest.source rewired to *_uniform_h14.pkl)
└── relaxed_walk_loop_v1_halfspeed_walks_chain_matched.yaml  (rest.source rewired to *_chain_matched.pkl)
```

---

## Why the rest-pose match matters (the visible fix)

Every `make_warehouse_motion.py` reel inserts a 75-frame (2.5 s) rest
layer between every pair of segments:

```
segment_end → SLERP 30 fr in → hold rest pose 15 fr → SLERP 30 fr out → next_segment_start
```

Before: the rest source was retargeted with an older config whose
`R_wr_yaw` was at +47° (wrists rotated as if hands behind back). Every
new-retarget walk clip has `R_wr_yaw` at -19° (hands hanging at
sides). At every seam, both wrists rotated ~80° to "hands-on-back",
held for 0.5 s, then rotated back. The user described this as the
"robot going to default pose and coming back between motion clips".

After: the rest source's wrists match the segment-end wrists; the
mid-rest L2 drop in wrist-only joints is **22%** (0.397 → 0.310 rad on
the walk_demo_v6 reel). Anchor seams (open/close — where `anchor_open`
and `anchor_close` segments are themselves `neutral_idle_loop_002__A074`)
go from 0.295 rad of lower-body traverse to **0.012 rad** because the
anchor segments and the rest source are now the same clip.

| Seam in walk_demo_v6 | Wrist L2 mid-rest (OLD) | Wrist L2 mid-rest (NEW uniform_h14) | Wrist L2 mid-rest (NEW chain_matched) |
|---|---|---|---|
| anchor_open → pivot_left_1 | 0.297 | 0.284 | 0.245 |
| pivot_left_1 → walk_3_steps_a | 0.396 | 0.256 | 0.184 |
| walk_3_steps_a → pivot_right_1 | 0.542 | 0.319 | 0.429 |
| pivot_right_1 → pivot_right_2 | 0.255 | 0.377 | 0.351 |
| pivot_right_2 → walk_3_steps_b | 0.255 | 0.377 | 0.351 |
| walk_3_steps_b → pivot_left_2 | 0.542 | 0.319 | 0.429 |
| pivot_left_2 → anchor_close | 0.396 | 0.256 | 0.184 |
| **mean** | **0.397** | **0.313** (-21%) | **0.310** (-22%) |

The chain_matched track has slightly more **lower-body** traverse at
mid-seam (since its idle stance has a +21° knee bend the walks also
use, vs uniform_h14's +5° knee bend), but the result is visually more
consistent — the legs no longer straighten and re-bend during the rest
seam.

---

## Why the chain_matched retarget rebuild matters (the deeper fix)

The earlier scratch `chain_matched_v4` config in this repo used a
uniform 0.6732 scaling factor for every leg-chain joint, which forced
the IK to a permanent ~70° knee flex (the source human's leg was
shorter than the X2 Ultra's, and IK had to fold the knee to make the
foot reach). Result: pelvis Z floored at 0.509 m for the whole motion
(Groucho-Marx crouch). Per-walk distance also shrunk from 3.50 m
(uniform_h14 reference) to 2.89 m.

The committed `soma_to_x2_ultra_chain_matched_retargeter_config.json`
replaces the uniform leg scales with per-segment ratios that respect
the X2 leg geometry (long thigh, short shin, normal foot):

| Joint | OLD `_v4` scale | NEW committed scale |
|---|---|---|
| LeftLeg (thigh) | 0.6732 | **0.541** |
| LeftShin | 0.6732 | **0.81** |
| LeftFoot | 0.6732 | **0.75** |
| LeftToe | 0.6732 | **0.78** |
| LeftToeBase | 0.6732 | **0.66** |
| (Hips `r_weight`) | 1.0 | **10.0** |
| (wrist smoothing) | none | added |

With the fix the chain_matched walks come back to natural posture
(pelvis Z 0.634-0.681 m, knee flex max ~23° at idle) and per-walk
distance recovers to 3.50 m, matching the uniform_h14 reference.

| Metric over `relaxed_walk_loop_v1` (1565 frames) | uniform_h14 | chain_matched (NEW) | chain_matched_v4 (OLD, deleted) |
|---|---|---|---|
| pelvis_z min / mean / max | 0.591 / 0.639 / 0.673 m | 0.634 / 0.669 / 0.681 m | 0.509 / 0.565 / 0.673 m |
| L knee flex mean / max | 0.18 rad / 1.29 (10° / 74°) | 0.41 rad / 1.60 (23° / 92°) | 1.21 rad / 2.01 (70° / 115°) |
| per-walk distance | 3.50 m | 3.50 m | 2.89 m |
| loop yaw closure | -2.1° | -9.7° | -11.1° |

`chain_matched` still has slightly looser yaw closure than
`uniform_h14` (-9.7° vs -2.1° on the relaxed-walk loop) because its
walks have asymmetric stride rotations baked in by the per-chain
ratios — not a bug, just a different trade-off (per-chain accuracy
beats yaw consistency).

---

## How to preview / consume

All commands documented in detail at `play_pkl_motions_commands.md`
under three new sections:

* "regular walk closed loop v1 — walk_forward_loop_001__A021 + v6 turn primitives"
* "rest-pose match for stitched loops"
* "sit/stand chain_matched motions @ 1.0× + 0.5×"

Quick reference:

```bash
# Preview the tightest-yaw-closure loop in the repo
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_walk_demo_v6_chain_matched.pkl \
    --no-loop

# Preview a sit-down at half speed
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_sitstand_chain_matched.pkl \
    --motion-key sit_on_chair_start_R_001__A244__speed_0.5 \
    --no-loop

# Rebuild everything if any source CSV / config / YAML changes
for VAR in walk_loop_v1_uniform_h14 walk_loop_v1_chain_matched \
            walk_demo_v6_uniform_h14 walk_demo_v6_chain_matched \
            relaxed_walk_loop_v1 relaxed_walk_loop_v1_chain_matched \
            relaxed_walk_loop_v1_halfspeed_walks relaxed_walk_loop_v1_halfspeed_walks_chain_matched ; do
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/${VAR}.yaml \
    --out      gear_sonic/data/motions/x2_ultra_${VAR}.pkl
done
```

---

## Open follow-ups

* **SONIC sim validation.** None of these PKLs have been run through
  `deploy_x2.sh sim --motion <pkl> --sim-viewer` yet. Sanity-checking
  the half-speed relaxed walks under the current SONIC policy is the
  next step; the half-speed builds were produced specifically to give
  SONIC twice the frames per stride to track.
* **`chain_matched` loop closure.** The chain_matched relaxed-walk
  loop drifts -9.7° in yaw over the closed loop vs uniform_h14's
  -2.1°. If a tighter closure on chain_matched is needed, re-run the
  FK inter-step sweep against the new chain_matched bundle and
  re-trim `n_frames` per walk segment.
* **Sit/stand stitching playlist.** The sit/stand bundle exists but
  no `chair_visit_v1.yaml` playlist references it yet. A natural
  sequence would be: `anchor_open → sit_down → sit_for_5s → stand_up
  → anchor_close`, with a short or zero `rest_frames` between the
  sit/stand segments (the loop segment already holds the seated pose).
* **uniform_h14 sit/stand variant.** The sit/stand BVHs were not
  retargeted on the `uniform_h14` track in this session (operator
  asked for chain_matched only). If A/B parity with the walks bundle
  is wanted, repeat the retarget loop with `--model-height 1.40` on
  the same three BVHs and merge into a parallel
  `x2_ultra_sitstand_uniform_h14.pkl`.
* **`x2_ultra_demo_v1.pkl` migration.** The new
  `build_x2_demo_motion_lib.py` (also created today, in a parallel
  session) merges per-task PKLs under
  `gear_sonic/data/motions/demo_v1_sources/<subdir>/` into a single
  `x2_ultra_demo_v1.pkl`. The bundles produced in this session are
  drop-in compatible — symlinking them under
  `demo_v1_sources/retargeted/` / `demo_v1_sources/sitstand_chain_matched/`
  will pull them into that aggregate bundle automatically.

---

## What was NOT touched

* Any stitching / bundle / retarget infrastructure code. Everything
  used today (`make_warehouse_motion.py`, `_warehouse_playlist.py`,
  `make_stitched_motion.py`, `build_x2_bones_seed_motion_lib.py`,
  `scripts/retarget_one.py`) was authored on 2026-05-07 or
  2026-05-13 in prior sessions.
* SONIC policy / training pipeline. These are reference motions for
  downstream consumers, not policy code.
* PC2 deploy or the live VLA stack — separate session today owned
  those concerns (see
  [`2026-06-23_blend_future_window_continuity`](2026-06-23_blend_future_window_continuity.md)
  and
  [`2026-06-23_pose_pub_lan_isolation`](2026-06-23_pose_pub_lan_isolation.md)
  and
  [`2026-06-23_vla_bridge_yaw_hold_last_good`](2026-06-23_vla_bridge_yaw_hold_last_good.md)).
* `x2_planner_bins.yaml` and `x2_planner_primitives_recipes.yaml`
  were modified today by a parallel session and are unrelated to this
  data-generation work.
