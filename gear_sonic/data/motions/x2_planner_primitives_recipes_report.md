# X2 planner primitive **recipe** build report

- Recipes: `/home/stickbot/Projects/GR00T-WholeBodyControl/gear_sonic/data/motions/x2_planner_primitives_recipes.yaml`
- Source: `/home/stickbot/Projects/GR00T-WholeBodyControl/gear_sonic/data/motions/x2_ultra_bones_seed.pkl`
- Bins built: **30**
- Build wall time: 1.3s

## Bin summary

| Bin | Family | Frames | fps | Recipe ops | Frozen arms+head |
|---|---|---|---|---|---|
| `back_step_half_ft` | locomotion | 48 | 30.0 | `clip_window -> freeze` | yes |
| `back_step_quarter_ft` | locomotion | 48 | 30.0 | `derive_from:back_step_half_ft -> clip_window -> freeze -> scale_magnitude` | yes |
| `back_walk_standard` | continuous_walk | 180 | 30.0 | `clip_window -> freeze` | yes |
| `crouch_large` | static_upper_body | 65 | 50.0 | `synthesize_crouch_ramp -> freeze` | yes |
| `crouch_medium` | static_upper_body | 125 | 30.0 | `clip_window -> freeze` | yes |
| `crouch_small` | static_upper_body | 65 | 50.0 | `synthesize_crouch_ramp -> freeze` | yes |
| `fwd_step_1ft` | locomotion | 48 | 30.0 | `clip_window -> freeze` | yes |
| `fwd_step_half_ft` | locomotion | 48 | 30.0 | `derive_from:fwd_step_1ft -> clip_window -> freeze -> scale_magnitude` | yes |
| `fwd_step_quarter_ft` | locomotion | 48 | 30.0 | `derive_from:fwd_step_1ft -> clip_window -> freeze -> scale_magnitude` | yes |
| `fwd_walk_standard` | continuous_walk | 180 | 30.0 | `clip_window -> freeze` | yes |
| `idle_stand` | idle | 45 | 30.0 | `clip_window -> freeze` | yes |
| `lean_fwd_large` | static_upper_body | 75 | 30.0 | `derive_from:lean_fwd_medium -> clip_window -> freeze -> scale_magnitude` | yes |
| `lean_fwd_medium` | static_upper_body | 75 | 30.0 | `clip_window -> freeze` | yes |
| `lean_fwd_small` | static_upper_body | 75 | 30.0 | `derive_from:lean_fwd_medium -> clip_window -> freeze -> scale_magnitude` | yes |
| `side_left_step` | locomotion | 90 | 30.0 | `derive_from:side_right_step -> clip_window -> freeze -> mirror_lr` | yes |
| `side_right_step` | locomotion | 90 | 30.0 | `clip_window -> freeze` | yes |
| `torso_left_15deg` | static_upper_body | 70 | 50.0 | `synthesize_waist_ramp -> freeze` | yes |
| `torso_left_30deg` | static_upper_body | 70 | 50.0 | `synthesize_waist_ramp -> freeze` | yes |
| `torso_left_45deg` | static_upper_body | 70 | 50.0 | `synthesize_waist_ramp -> freeze` | yes |
| `torso_right_15deg` | static_upper_body | 70 | 50.0 | `derive_from:torso_left_15deg -> synthesize_waist_ramp -> freeze -> mirror_lr` | yes |
| `torso_right_30deg` | static_upper_body | 70 | 50.0 | `derive_from:torso_left_30deg -> synthesize_waist_ramp -> freeze -> mirror_lr` | yes |
| `torso_right_45deg` | static_upper_body | 70 | 50.0 | `derive_from:torso_left_45deg -> synthesize_waist_ramp -> freeze -> mirror_lr` | yes |
| `turn_left_15deg` | locomotion | 60 | 30.0 | `clip_window -> freeze` | yes |
| `turn_left_30deg` | locomotion | 70 | 30.0 | `clip_window -> freeze` | yes |
| `turn_left_45deg` | locomotion | 105 | 30.0 | `clip_window -> freeze` | yes |
| `turn_left_90deg` | locomotion | 90 | 30.0 | `clip_window -> freeze` | yes |
| `turn_right_15deg` | locomotion | 60 | 30.0 | `derive_from:turn_left_15deg -> clip_window -> freeze -> mirror_lr` | yes |
| `turn_right_30deg` | locomotion | 70 | 30.0 | `derive_from:turn_left_30deg -> clip_window -> freeze -> mirror_lr` | yes |
| `turn_right_45deg` | locomotion | 105 | 30.0 | `derive_from:turn_left_45deg -> clip_window -> freeze -> mirror_lr` | yes |
| `turn_right_90deg` | locomotion | 90 | 30.0 | `derive_from:turn_left_90deg -> clip_window -> freeze -> mirror_lr` | yes |

## Per-bin sources

### `back_step_half_ft` (locomotion)
- notes: v3 source: walk_backward_loop_005__A028_M frames [292, 340] (1.6s,
dx=-0.52m, dy=+0.017m clean, dyaw -0.3 deg, fl_max=8.5cm).
Lower yaw_osc than v2 source.

- frames: 48 @ 30.0 fps
- sources:
  - `loco__walk_backward_loop_005__A028_M[292:340]`
  - `op:freeze(arms,head)`

### `back_step_quarter_ft` (locomotion)
- notes: derived: 0.5 * back_step_half_ft
- frames: 48 @ 30.0 fps
- sources:
  - `loco__walk_backward_loop_005__A028_M[292:340]`
  - `op:freeze(arms,head)`
  - `derive_from:back_step_half_ft`
  - `op:scale_magnitude(0.5)`

### `back_walk_standard` (continuous_walk)
- notes: v3 source: walk_backward_loop_005__A028_M. Whole-clip 19.87s,
-0.34 m/s straight, fl_max 12.7 cm, dyaw -1.5 deg, yaw_osc 39 deg.
Window [299, 479] = 6 s, dx=-1.89 m, dy=+0.06 m, dyaw -0.1 deg,
fl_max 10.3 cm.

- frames: 180 @ 30.0 fps
- sources:
  - `loco__walk_backward_loop_005__A028_M[299:479]`
  - `op:freeze(arms,head)`

### `crouch_large` (static_upper_body)
- notes: v4 synthesized: 10 cm pelvis drop. UNREACHABLE via planner --
crouch+large is aliased to crouch_medium (mocap squat). Kept in
the build so existing test fixtures and the curator inventory
continue to resolve.

- frames: 65 @ 50.0 fps
- sources:
  - `synth:crouch_ramp(peak_drop=0.100m,hip=0.505,knee=1.011)`
  - `op:freeze(arms,head)`

### `crouch_medium` (static_upper_body)
- notes: v5 source: medium_big_light_two_hands_front_medium_to_front_low__A530_M
frames [10, 135] (125f @ 30 fps = 4.17 s).
Hand-picked from 22 symmetric-squat candidates as the one with
the LOWEST torso lean while still bending knees deeply: at apex
(around frame 85, ~2.50 s) L/R knee +96/+100 deg (asym 4 deg),
L/R hip -97/-95 deg (asym 2 deg), **waist_pitch ~0 deg** (torso
stays upright relative to pelvis throughout), pelvis Z drop
27 cm. Endpoints are square stance with zero knee delta and hip
delta within +/-2 deg. ~13 cm forward XY drift at apex (actor
shifts CG forward to balance), recovers by end of window.
Replaces the first v5 attempt (A516 [10,75]) which had a
similar squat shape but +18 deg waist pitch at apex -- visually
a "deep stoop", not a "true squat". A530_M's flat waist + same
deep knee bend gives the upright crouch the user asked for.

- frames: 125 @ 30.0 fps
- sources:
  - `loco__medium_big_light_two_hands_front_medium_to_front_low_R_001__A530_M[10:135]`
  - `op:freeze(arms,head)`

### `crouch_small` (static_upper_body)
- notes: v4 synthesized: 3 cm pelvis drop. UNREACHABLE via planner --
crouch+small is aliased to crouch_medium (mocap squat). Kept in
the build so existing test fixtures and the curator inventory
continue to resolve.

- frames: 65 @ 50.0 fps
- sources:
  - `synth:crouch_ramp(peak_drop=0.030m,hip=0.275,knee=0.549)`
  - `op:freeze(arms,head)`

### `fwd_step_1ft` (locomotion)
- notes: v5 source: walk_forward_loop_003__A034 frames [279, 327] (1.6s
at 30fps, body-frame dx=+0.45m, lat=+0.01m, dyaw +0.3 deg).
Both endpoints in symmetric stance (knee asym start/end =
0.0/0.1 deg), so seam-blend into idle preserves the stride.
Same source clip as the proven-tracking fwd_walk_standard.

Replaces v4 source (walk_randdir_relax_001__A005 [0, 68]) which
only produced +20 cm of body-frame translation in 2.3 s (~9 cm/s);
after scale_magnitude 0.5 / 0.25 the half_ft / quarter_ft
derivatives became ~10 / ~5 cm and the policy treated them as
noise. v5 base ~45 cm gives half_ft ~22 cm and quarter_ft ~11 cm,
both clearly above the policy noise floor.

- frames: 48 @ 30.0 fps
- sources:
  - `loco__walk_forward_loop_003__A034[279:327]`
  - `op:freeze(arms,head)`

### `fwd_step_half_ft` (locomotion)
- notes: derived: 0.5 * fwd_step_1ft
- frames: 48 @ 30.0 fps
- sources:
  - `loco__walk_forward_loop_003__A034[279:327]`
  - `op:freeze(arms,head)`
  - `derive_from:fwd_step_1ft`
  - `op:scale_magnitude(0.5)`

### `fwd_step_quarter_ft` (locomotion)
- notes: derived: 0.25 * fwd_step_1ft
- frames: 48 @ 30.0 fps
- sources:
  - `loco__walk_forward_loop_003__A034[279:327]`
  - `op:freeze(arms,head)`
  - `derive_from:fwd_step_1ft`
  - `op:scale_magnitude(0.25)`

### `fwd_walk_standard` (continuous_walk)
- notes: v3 source: walk_forward_loop_003__A034. Whole-clip 11.17s,
0.51 m/s straight, fl_max 14.6 cm, dyaw -0.1 deg, yaw_osc 53 deg
(acceptable, <55 SONIC-trackable bound). Window [82, 262] = 6 s,
dx=+4.09 m, dyaw -0.3 deg, fl_max 14.6 cm, planted endpoints.

- frames: 180 @ 30.0 fps
- sources:
  - `loco__walk_forward_loop_003__A034[82:262]`
  - `op:freeze(arms,head)`

### `idle_stand` (idle)
- notes: curator pick: best feet-planted slice; arms+head pinned to neutral stand
- frames: 45 @ 30.0 fps
- sources:
  - `loco__idle_vigilance_start_R_001__A502[33:78]`
  - `op:freeze(arms,head)`

### `lean_fwd_large` (static_upper_body)
- notes: derived: 1.4 * lean_fwd_medium (~31 deg apex)
- frames: 75 @ 30.0 fps
- sources:
  - `loco__body_check_001__A474_M[752:827]`
  - `op:freeze(arms,head)`
  - `derive_from:lean_fwd_medium`
  - `op:scale_magnitude(1.4)`

### `lean_fwd_medium` (static_upper_body)
- notes: natural lean: body_check_001__A474_M frames [752, 827] (~22 deg apex)
- frames: 75 @ 30.0 fps
- sources:
  - `loco__body_check_001__A474_M[752:827]`
  - `op:freeze(arms,head)`

### `lean_fwd_small` (static_upper_body)
- notes: derived: 0.5 * lean_fwd_medium (~11 deg apex)
- frames: 75 @ 30.0 fps
- sources:
  - `loco__body_check_001__A474_M[752:827]`
  - `op:freeze(arms,head)`
  - `derive_from:lean_fwd_medium`
  - `op:scale_magnitude(0.5)`

### `side_left_step` (locomotion)
- notes: derived: mirror_lr(side_right_step) -> back-left diagonal at ~45deg
- frames: 90 @ 30.0 fps
- sources:
  - `loco__walk_sideway_045_stop_001__A038_M[0:90]`
  - `op:freeze(arms,head)`
  - `derive_from:side_right_step`
  - `op:mirror_lr`

### `side_right_step` (locomotion)
- notes: Mocap source: walk_sideway_045_stop_001__A038_M frames [0, 90]
(3.0s @ 30Hz). Body-frame delta dx=-0.25m back, dy=-0.27m right
-- a back-right diagonal slide at ~45deg, ~37 cm total. Yaw
drift -15 deg over 3s, waist_yaw essentially fixed (span 0.9 deg),
planted endpoints. Selected from the 99-clip side-walk SONIC
review as clip 35/99 (eyeballed clean by user 2026-05-11).

KNOWN LIMITATION: this 3s window is biomechanically a knees-
locked pelvis-slide -- L knee span 3.8 deg, R knee span 0.1 deg
over the window -- so visible foot lift under SONIC is minimal.
The pelvis travels but the feet skate. We chose simplicity here
("just keep it simple for now given how difficult it is for us
to get any good moves" -- user 2026-05-11) over a longer window
with real stepping. If foot-lift becomes important later, see
A038_M frames [60, 150] (L knee span 12.9 deg) or revisit the
99-clip review video for a different source.

- frames: 90 @ 30.0 fps
- sources:
  - `loco__walk_sideway_045_stop_001__A038_M[0:90]`
  - `op:freeze(arms,head)`

### `torso_left_15deg` (static_upper_body)
- notes: synthesized: waist_yaw ramp 0->+15deg->0
- frames: 70 @ 50.0 fps
- sources:
  - `synth:waist_yaw_ramp(peak=0.2618rad)`
  - `op:freeze(arms,head)`

### `torso_left_30deg` (static_upper_body)
- notes: synthesized: waist_yaw ramp 0->+30deg->0
- frames: 70 @ 50.0 fps
- sources:
  - `synth:waist_yaw_ramp(peak=0.5236rad)`
  - `op:freeze(arms,head)`

### `torso_left_45deg` (static_upper_body)
- notes: synthesized: waist_yaw ramp 0->+45deg->0
- frames: 70 @ 50.0 fps
- sources:
  - `synth:waist_yaw_ramp(peak=0.7854rad)`
  - `op:freeze(arms,head)`

### `torso_right_15deg` (static_upper_body)
- notes: derived: mirror_lr(torso_left_15deg) -> waist_yaw -15deg
- frames: 70 @ 50.0 fps
- sources:
  - `synth:waist_yaw_ramp(peak=0.2618rad)`
  - `op:freeze(arms,head)`
  - `derive_from:torso_left_15deg`
  - `op:mirror_lr`

### `torso_right_30deg` (static_upper_body)
- notes: derived: mirror_lr(torso_left_30deg) -> waist_yaw -30deg
- frames: 70 @ 50.0 fps
- sources:
  - `synth:waist_yaw_ramp(peak=0.5236rad)`
  - `op:freeze(arms,head)`
  - `derive_from:torso_left_30deg`
  - `op:mirror_lr`

### `torso_right_45deg` (static_upper_body)
- notes: derived: mirror_lr(torso_left_45deg) -> waist_yaw -45deg
- frames: 70 @ 50.0 fps
- sources:
  - `synth:waist_yaw_ramp(peak=0.7854rad)`
  - `op:freeze(arms,head)`
  - `derive_from:torso_left_45deg`
  - `op:mirror_lr`

### `turn_left_15deg` (locomotion)
- notes: curator pick (score 0.745)
- frames: 60 @ 30.0 fps
- sources:
  - `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M[1350:1410]`
  - `op:freeze(arms,head)`

### `turn_left_30deg` (locomotion)
- notes: curator pick (score 0.610)
- frames: 70 @ 30.0 fps
- sources:
  - `loco__idle_turn_360_R_003__A265_M[85:155]`
  - `op:freeze(arms,head)`

### `turn_left_45deg` (locomotion)
- notes: curator pick (score 0.685)
- frames: 105 @ 30.0 fps
- sources:
  - `loco__Step_Rotate_Reaction_Idle_0135_001__A019[1040:1145]`
  - `op:freeze(arms,head)`

### `turn_left_90deg` (locomotion)
- notes: curator pick (score 0.734)
- frames: 90 @ 30.0 fps
- sources:
  - `loco__idle_turn_270_R_003__A235_M[0:90]`
  - `op:freeze(arms,head)`

### `turn_right_15deg` (locomotion)
- notes: derived: mirror_lr(turn_left_15deg)
- frames: 60 @ 30.0 fps
- sources:
  - `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M[1350:1410]`
  - `op:freeze(arms,head)`
  - `derive_from:turn_left_15deg`
  - `op:mirror_lr`

### `turn_right_30deg` (locomotion)
- notes: derived: mirror_lr(turn_left_30deg)
- frames: 70 @ 30.0 fps
- sources:
  - `loco__idle_turn_360_R_003__A265_M[85:155]`
  - `op:freeze(arms,head)`
  - `derive_from:turn_left_30deg`
  - `op:mirror_lr`

### `turn_right_45deg` (locomotion)
- notes: derived: mirror_lr(turn_left_45deg)
- frames: 105 @ 30.0 fps
- sources:
  - `loco__Step_Rotate_Reaction_Idle_0135_001__A019[1040:1145]`
  - `op:freeze(arms,head)`
  - `derive_from:turn_left_45deg`
  - `op:mirror_lr`

### `turn_right_90deg` (locomotion)
- notes: derived: mirror_lr(turn_left_90deg)
- frames: 90 @ 30.0 fps
- sources:
  - `loco__idle_turn_270_R_003__A235_M[0:90]`
  - `op:freeze(arms,head)`
  - `derive_from:turn_left_90deg`
  - `op:mirror_lr`

---
Edit the recipes YAML and re-run ``gear_sonic.scripts.build_x2_planner_primitives`` to regenerate.
