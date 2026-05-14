# X2 planner primitive curator report

- Source: `gear_sonic/data/motions/x2_ultra_bones_seed.pkl`
- Bins: 36
- Perfect matches: **15**
- Partial matches: **21** (review and pin)
- Missing: **0** (no candidate had score > 0)
- Curator wall time: 84.1s

## Bin status summary

| Bin | Status | Best score | Selected motion_key | Frames |
|---|---|---|---|---|
| `back_step_half_ft` | OK | 0.531 | `loco__Loop_Backward_Walk_001__A019` | 1326..1396 |
| `back_step_quarter_ft` | OK | 0.155 | `loco__walk_backward_loop_007__A026` | 658..716 |
| `back_walk_standard` | OK | 0.721 | `loco__walk_backward_start_001__A039_M` | 0..90 |
| `crouch_large` | PARTIAL | 0.461 | `loco__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189..219 |
| `crouch_medium` | PARTIAL | 0.461 | `loco__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189..219 |
| `crouch_small` | PARTIAL | 0.461 | `loco__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189..219 |
| `fwd_step_1ft` | OK | 0.519 | `loco__walk_180_R_003__A333_M` | 240..290 |
| `fwd_step_half_ft` | PARTIAL | 0.048 | `loco__walk_forward_confident_impro_001__A002_M` | 629..699 |
| `fwd_step_quarter_ft` | OK | 0.352 | `loco__walk_backward_loop_007__A028` | 470..512 |
| `fwd_walk_standard` | OK | 0.991 | `loco__walk_180_R_001__A331` | 242..332 |
| `idle_stand` | PARTIAL | 0.846 | `loco__idle_vigilance_start_R_001__A502` | 33..78 |
| `lean_fwd_large` | PARTIAL | 0.003 | `loco__walk_forward_weakend_sick_impro_002__A001_M` | 0..50 |
| `lean_fwd_medium` | PARTIAL | 0.088 | `loco__medium_big_heavy_two_hands_pick_up_front_high_R_001__A523_M` | 77..107 |
| `lean_fwd_small` | PARTIAL | 0.236 | `loco__walk_forward_relax_003__A006_M` | 0..30 |
| `lean_left_large` | PARTIAL | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21..51 |
| `lean_left_medium` | PARTIAL | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21..51 |
| `lean_left_small` | PARTIAL | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21..51 |
| `lean_right_large` | PARTIAL | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21..51 |
| `lean_right_medium` | PARTIAL | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21..51 |
| `lean_right_small` | PARTIAL | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21..51 |
| `side_left_step` | PARTIAL | 0.019 | `loco__walk_sideway_045_stop_003__A044` | 165..225 |
| `side_right_step` | OK | 0.243 | `loco__Sideway_Walk_Left_001__A017_M` | 795..855 |
| `torso_left_15deg` | PARTIAL | 0.048 | `loco__mosquito_squash_neck_left_R_001__A502_M` | 126..201 |
| `torso_left_30deg` | PARTIAL | 0.000 | `loco__arc_walk_left_stop_004__A037` | 245..275 |
| `torso_left_40deg` | PARTIAL | 0.000 | `loco__arc_walk_left_stop_004__A037` | 245..275 |
| `torso_right_15deg` | PARTIAL | 0.309 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 126..156 |
| `torso_right_30deg` | PARTIAL | 0.003 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 108..158 |
| `torso_right_40deg` | PARTIAL | 0.000 | `loco__small_heavy_two_hands_put_down_right_side_medium_R_001__A508_M` | 63..93 |
| `turn_left_15deg` | OK | 0.745 | `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M` | 1350..1410 |
| `turn_left_30deg` | OK | 0.610 | `loco__idle_turn_360_R_003__A265_M` | 85..155 |
| `turn_left_45deg` | OK | 0.685 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 1040..1145 |
| `turn_left_90deg` | OK | 0.734 | `loco__idle_turn_270_R_003__A235_M` | 0..90 |
| `turn_right_15deg` | OK | 0.733 | `loco__Turn_Start_Walk_0045_001__A020_M` | 2037..2067 |
| `turn_right_30deg` | OK | 0.672 | `loco__big_heavy_one_hand_right_side_high_to_behind_high_R_002__A524_M` | 224..254 |
| `turn_right_45deg` | OK | 0.731 | `loco__Step_Rotate_Reaction_Idle_0135_001__A017_M` | 1456..1521 |
| `turn_right_90deg` | OK | 0.865 | `loco__step_rotate_idle_090_003__A023_M` | 37..187 |

## Per-bin candidates (top-K)

### `back_step_half_ft` (locomotion)
- target_xy_m: `(-0.1524, 0.0)` target_yaw_deg: `0.0`
- tol_xy_m: `0.05` tol_yaw_deg: `4.0` cross_axis_max_m: `0.05`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.531 | `loco__Loop_Backward_Walk_001__A019` | 1326 | 70 | (-0.167,-0.009) | -0.7 | 1.7 | 19.4 | 0.61 | 0.12 | 0.00 | 1 | all-pass |
| 2 | 0.133 | `loco__Loop_Backward_Walk_001__A020` | 1337 | 30 | (-0.106,-0.036) | -1.7 | 2.1 | 14.5 | 0.64 | 0.40 | 0.00 | 1 | all-pass |
| 3 | 0.057 | `loco__walk_backward_loop_003__A035_M` | 228 | 50 | (-0.222,-0.013) | 1.0 | 6.5 | 14.7 | 0.44 | 0.65 | 0.00 | 1 | xy_along |
| 4 | 0.043 | `loco__Loop_Backward_Walk_001__A020` | 255 | 70 | (-0.088,0.038) | 3.1 | 1.6 | 14.4 | 0.76 | 0.26 | 0.00 | 1 | xy_along |
| 5 | 0.009 | `loco__walk_backward_stop_003__A042` | 147 | 30 | (-0.179,-0.078) | -2.6 | 2.1 | 17.9 | 0.21 | 0.31 | 0.00 | 1 | cross_axis,end_at_square |
| 6 | 0.006 | `loco__Loop_Backward_Walk_001__A020` | 2275 | 30 | (-0.111,0.013) | -7.7 | 4.4 | 8.2 | 0.50 | 0.45 | 0.00 | 1 | yaw |
| 7 | 0.003 | `loco__walk_backward_loop_007__A026` | 660 | 50 | (-0.039,-0.016) | -1.8 | 2.8 | 5.0 | 0.75 | 0.01 | 0.00 | 1 | xy_along |
| 8 | 0.002 | `loco__Loop_Backward_Walk_001__A019` | 1836 | 70 | (-0.049,-0.018) | -4.8 | 4.8 | 13.9 | 0.80 | 0.74 | 0.00 | 1 | xy_along,yaw |

### `back_step_quarter_ft` (locomotion)
- target_xy_m: `(-0.0762, 0.0)` target_yaw_deg: `0.0`
- tol_xy_m: `0.04` tol_yaw_deg: `4.0` cross_axis_max_m: `0.04`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.155 | `loco__walk_backward_loop_007__A026` | 658 | 58 | (-0.039,0.011) | -3.2 | 3.1 | 6.1 | 0.76 | 0.26 | 0.00 | 1 | all-pass |
| 2 | 0.112 | `loco__walk_backward_loop_007__A028` | 820 | 42 | (-0.025,-0.012) | 0.4 | 4.9 | 7.6 | 0.66 | 0.56 | 0.00 | 1 | xy_along |
| 3 | 0.096 | `loco__walk_backward_loop_007__A026` | 864 | 75 | (-0.024,-0.013) | -1.3 | 0.4 | 7.7 | 0.64 | 0.05 | 0.00 | 1 | xy_along |
| 4 | 0.091 | `loco__walk_backward_loop_007__A026` | 882 | 58 | (-0.025,-0.012) | -1.8 | 0.4 | 7.5 | 0.62 | 0.01 | 0.00 | 1 | xy_along |
| 5 | 0.073 | `loco__Loop_Backward_Walk_001__A017` | 1848 | 25 | (-0.076,-0.011) | 5.6 | 7.0 | 15.8 | 0.56 | 0.06 | 0.00 | 1 | yaw |
| 6 | 0.072 | `loco__Loop_Backward_Walk_001__A019` | 1836 | 75 | (-0.049,-0.017) | -5.3 | 4.8 | 13.9 | 0.76 | 0.71 | 0.00 | 1 | yaw |
| 7 | 0.066 | `loco__walk_backward_loop_007__A026` | 882 | 75 | (-0.018,-0.010) | -1.2 | 0.4 | 7.5 | 0.62 | 0.13 | 0.00 | 1 | xy_along |
| 8 | 0.064 | `loco__Loop_Backward_Walk_001__A019` | 3122 | 58 | (-0.073,-0.031) | -2.9 | 1.1 | 9.3 | 0.20 | 1.00 | 0.00 | 1 | end_at_square |

### `back_walk_standard` (continuous_walk)
- target_xy_m: `(-0.5, 0.0)` target_yaw_deg: `0.0`
- tol_xy_m: `0.2` tol_yaw_deg: `8.0` cross_axis_max_m: `0.1`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.721 | `loco__walk_backward_start_001__A039_M` | 0 | 90 | (-0.456,-0.047) | -2.0 | 3.0 | 33.3 | 0.00 | 0.07 | 0.00 | 1 | all-pass |
| 2 | 0.714 | `loco__walk_backward_loop_002__A031_M` | 240 | 120 | (-0.386,-0.010) | -0.4 | 8.7 | 41.9 | 0.53 | 0.15 | 0.00 | 1 | all-pass |
| 3 | 0.626 | `loco__walk_backward_start_002__A041` | 0 | 90 | (-0.410,0.046) | -1.8 | 3.8 | 24.4 | 0.01 | 0.06 | 0.00 | 1 | all-pass |
| 4 | 0.252 | `loco__walk_backward_loop_002__A039_M` | 330 | 120 | (-0.399,-0.080) | 5.5 | 5.3 | 32.5 | 0.32 | 0.09 | 0.00 | 1 | all-pass |
| 5 | 0.236 | `loco__walk_backward_loop_002__A039_M` | 330 | 90 | (-0.393,-0.080) | 5.8 | 5.3 | 32.5 | 0.33 | 0.11 | 0.00 | 1 | all-pass |
| 6 | 0.225 | `loco__walk_backward_loop_005__A028_M` | 506 | 90 | (-0.739,0.009) | 1.8 | 3.5 | 26.8 | 0.75 | 0.43 | 0.00 | 1 | xy_along |
| 7 | 0.225 | `loco__walk_backward_loop_005__A028` | 506 | 90 | (-0.739,-0.008) | -1.9 | 2.9 | 27.1 | 0.77 | 0.43 | 0.00 | 1 | xy_along |
| 8 | 0.215 | `loco__walk_backward_loop_003__A036_M` | 264 | 90 | (-0.363,-0.045) | 7.4 | 4.9 | 23.5 | 0.48 | 0.51 | 0.00 | 0 | all-pass |

### `crouch_large` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `12.0` end_at_apex_min: `0.5` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.461 | `loco__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189 | 30 | (-0.000,-0.000) | 0.0 | 0.2 | 0.2 | 0.36 | 1.00 | 0.92 | 0 | waist_axis |
| 2 | 0.461 | `standing__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189 | 30 | (-0.000,-0.000) | 0.0 | 0.2 | 0.2 | 0.36 | 1.00 | 0.92 | 0 | waist_axis |
| 3 | 0.436 | `loco__swollowing_pills_R_001__A502_M` | 21 | 30 | (-0.003,-0.001) | 0.0 | 0.1 | 0.1 | 0.27 | 1.00 | 0.87 | 0 | waist_axis |
| 4 | 0.436 | `standing__swollowing_pills_R_001__A502_M` | 21 | 30 | (-0.003,-0.001) | 0.0 | 0.1 | 0.1 | 0.27 | 1.00 | 0.87 | 0 | waist_axis |
| 5 | 0.431 | `loco__medium_big_heavy_one_hand_put_down_front_low_R_001__A507` | 140 | 30 | (0.002,-0.000) | 0.2 | 1.2 | 0.5 | 0.42 | 1.00 | 0.86 | 0 | waist_axis |
| 6 | 0.431 | `standing__medium_big_heavy_one_hand_put_down_front_low_R_001__A507` | 140 | 30 | (0.002,-0.000) | 0.2 | 1.2 | 0.5 | 0.42 | 1.00 | 0.86 | 0 | waist_axis |
| 7 | 0.431 | `loco__swollowing_pills_R_001__A502_M` | 7 | 30 | (-0.002,-0.000) | -0.0 | 0.1 | 0.1 | 0.27 | 0.95 | 0.91 | 0 | waist_axis |
| 8 | 0.431 | `standing__swollowing_pills_R_001__A502_M` | 7 | 30 | (-0.002,-0.000) | -0.0 | 0.1 | 0.1 | 0.27 | 0.95 | 0.91 | 0 | waist_axis |

### `crouch_medium` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `10.0` end_at_apex_min: `0.5` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.461 | `loco__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189 | 30 | (-0.000,-0.000) | 0.0 | 0.2 | 0.2 | 0.36 | 1.00 | 0.92 | 0 | waist_axis |
| 2 | 0.461 | `standing__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189 | 30 | (-0.000,-0.000) | 0.0 | 0.2 | 0.2 | 0.36 | 1.00 | 0.92 | 0 | waist_axis |
| 3 | 0.436 | `loco__swollowing_pills_R_001__A502_M` | 21 | 30 | (-0.003,-0.001) | 0.0 | 0.1 | 0.1 | 0.27 | 1.00 | 0.87 | 0 | waist_axis |
| 4 | 0.436 | `standing__swollowing_pills_R_001__A502_M` | 21 | 30 | (-0.003,-0.001) | 0.0 | 0.1 | 0.1 | 0.27 | 1.00 | 0.87 | 0 | waist_axis |
| 5 | 0.431 | `loco__medium_big_heavy_one_hand_put_down_front_low_R_001__A507` | 140 | 30 | (0.002,-0.000) | 0.2 | 1.2 | 0.5 | 0.42 | 1.00 | 0.86 | 0 | waist_axis |
| 6 | 0.431 | `standing__medium_big_heavy_one_hand_put_down_front_low_R_001__A507` | 140 | 30 | (0.002,-0.000) | 0.2 | 1.2 | 0.5 | 0.42 | 1.00 | 0.86 | 0 | waist_axis |
| 7 | 0.431 | `loco__swollowing_pills_R_001__A502_M` | 7 | 30 | (-0.002,-0.000) | -0.0 | 0.1 | 0.1 | 0.27 | 0.95 | 0.91 | 0 | waist_axis |
| 8 | 0.431 | `standing__swollowing_pills_R_001__A502_M` | 7 | 30 | (-0.002,-0.000) | -0.0 | 0.1 | 0.1 | 0.27 | 0.95 | 0.91 | 0 | waist_axis |

### `crouch_small` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `8.0` end_at_apex_min: `0.5` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.461 | `loco__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189 | 30 | (-0.000,-0.000) | 0.0 | 0.2 | 0.2 | 0.36 | 1.00 | 0.92 | 0 | waist_axis |
| 2 | 0.461 | `standing__big_light_one_hand_pick_up_right_side_low_R_002__A511` | 189 | 30 | (-0.000,-0.000) | 0.0 | 0.2 | 0.2 | 0.36 | 1.00 | 0.92 | 0 | waist_axis |
| 3 | 0.436 | `loco__swollowing_pills_R_001__A502_M` | 21 | 30 | (-0.003,-0.001) | 0.0 | 0.1 | 0.1 | 0.27 | 1.00 | 0.87 | 0 | waist_axis |
| 4 | 0.436 | `standing__swollowing_pills_R_001__A502_M` | 21 | 30 | (-0.003,-0.001) | 0.0 | 0.1 | 0.1 | 0.27 | 1.00 | 0.87 | 0 | waist_axis |
| 5 | 0.431 | `loco__medium_big_heavy_one_hand_put_down_front_low_R_001__A507` | 140 | 30 | (0.002,-0.000) | 0.2 | 1.2 | 0.5 | 0.42 | 1.00 | 0.86 | 0 | waist_axis |
| 6 | 0.431 | `standing__medium_big_heavy_one_hand_put_down_front_low_R_001__A507` | 140 | 30 | (0.002,-0.000) | 0.2 | 1.2 | 0.5 | 0.42 | 1.00 | 0.86 | 0 | waist_axis |
| 7 | 0.431 | `loco__swollowing_pills_R_001__A502_M` | 7 | 30 | (-0.002,-0.000) | -0.0 | 0.1 | 0.1 | 0.27 | 0.95 | 0.91 | 0 | waist_axis |
| 8 | 0.431 | `standing__swollowing_pills_R_001__A502_M` | 7 | 30 | (-0.002,-0.000) | -0.0 | 0.1 | 0.1 | 0.27 | 0.95 | 0.91 | 0 | waist_axis |

### `fwd_step_1ft` (locomotion)
- target_xy_m: `(0.3048, 0.0)` target_yaw_deg: `0.0`
- tol_xy_m: `0.06` tol_yaw_deg: `4.0` cross_axis_max_m: `0.05`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.519 | `loco__walk_180_R_003__A333_M` | 240 | 50 | (0.323,-0.007) | -2.2 | 5.6 | 19.0 | 0.79 | 0.56 | 0.00 | 1 | all-pass |
| 2 | 0.245 | `loco__Turn_Start_Walk_0000_001__A017_M` | 516 | 50 | (0.347,0.008) | -3.3 | 7.7 | 25.0 | 0.81 | 0.47 | 0.00 | 1 | all-pass |
| 3 | 0.176 | `loco__Turn_Start_Walk_0360_001__A019_M` | 187 | 70 | (0.329,-0.055) | -0.5 | 2.6 | 28.0 | 0.71 | 0.54 | 0.00 | 1 | cross_axis |
| 4 | 0.171 | `loco__Turn_Start_Walk_0360_001__A019` | 187 | 70 | (0.329,0.055) | 0.5 | 3.6 | 28.1 | 0.69 | 0.54 | 0.00 | 1 | cross_axis |
| 5 | 0.168 | `loco__street_avoid_obstacle_180_walk_R_002__A428_M` | 60 | 50 | (0.338,0.038) | 1.0 | 4.7 | 31.3 | 0.43 | 0.49 | 0.00 | 1 | all-pass |
| 6 | 0.140 | `loco__walk_hands_on_back_180_stop_R_002__A217_M` | 105 | 30 | (0.286,0.047) | 0.2 | 6.7 | 16.2 | 0.37 | 0.06 | 0.00 | 1 | end_at_square |
| 7 | 0.119 | `loco__walk_forward_stop_002__A028_M` | 136 | 70 | (0.314,-0.014) | 5.1 | 5.8 | 25.3 | 0.69 | 0.08 | 0.00 | 1 | yaw |
| 8 | 0.078 | `loco__walk_hands_on_back_loop_003__A030` | 217 | 30 | (0.374,-0.025) | 0.6 | 7.5 | 32.1 | 0.40 | 0.58 | 0.00 | 1 | xy_along,end_at_square |

### `fwd_step_half_ft` (locomotion)
- target_xy_m: `(0.1524, 0.0)` target_yaw_deg: `0.0`
- tol_xy_m: `0.05` tol_yaw_deg: `4.0` cross_axis_max_m: `0.05`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.048 | `loco__walk_forward_confident_impro_001__A002_M` | 629 | 70 | (0.141,-0.069) | -3.3 | 11.6 | 12.6 | 0.69 | 0.37 | 0.00 | 1 | cross_axis |
| 2 | 0.042 | `loco__street_avoid_obstacle_360_walk_R_002__A432` | 22 | 90 | (0.142,-0.023) | -1.4 | 10.3 | 42.9 | 0.16 | 0.43 | 0.00 | 2 | end_at_square,stride_count |
| 3 | 0.035 | `loco__walk_the_dog_ff_180_pull_leash_R_001__A495_M` | 84 | 50 | (0.082,0.042) | -2.5 | 3.9 | 19.9 | 0.76 | 1.00 | 0.00 | 1 | xy_along |
| 4 | 0.023 | `loco__walk_forward_confident_impro_001__A002_M` | 624 | 50 | (0.168,-0.028) | 7.2 | 11.0 | 12.9 | 0.86 | 0.45 | 0.00 | 1 | yaw |
| 5 | 0.020 | `loco__walk_sideway_045_stop_001__A026_M` | 105 | 30 | (0.070,-0.007) | -0.9 | 2.9 | 8.0 | 0.33 | 0.95 | 0.00 | 1 | xy_along,end_at_square |
| 6 | 0.019 | `loco__walk_randdir_relax_001__A005` | 1360 | 70 | (0.090,-0.014) | -1.9 | 17.7 | 49.7 | 0.31 | 0.69 | 0.00 | 2 | xy_along,end_at_square,stride_count |
| 7 | 0.010 | `loco__turn_start_walk_0000_001__A023` | 140 | 30 | (0.209,0.086) | 0.9 | 1.4 | 10.6 | 0.77 | 0.09 | 0.00 | 1 | xy_along,cross_axis |
| 8 | 0.010 | `loco__walk_backward_loop_007__A028` | 476 | 70 | (0.058,0.003) | 3.4 | 1.9 | 6.8 | 0.69 | 0.60 | 0.00 | 1 | xy_along |

### `fwd_step_quarter_ft` (locomotion)
- target_xy_m: `(0.0762, 0.0)` target_yaw_deg: `0.0`
- tol_xy_m: `0.04` tol_yaw_deg: `4.0` cross_axis_max_m: `0.04`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.352 | `loco__walk_backward_loop_007__A028` | 470 | 42 | (0.048,-0.021) | 1.0 | 1.6 | 7.7 | 0.81 | 0.60 | 0.00 | 1 | all-pass |
| 2 | 0.304 | `loco__Loop_Backward_Walk_001__A020` | 486 | 75 | (0.047,-0.022) | -0.3 | 1.3 | 8.2 | 0.71 | 0.51 | 0.00 | 1 | all-pass |
| 3 | 0.148 | `loco__Loop_Backward_Walk_001__A020` | 504 | 58 | (0.030,-0.009) | -0.9 | 1.7 | 6.5 | 0.65 | 0.40 | 0.00 | 1 | xy_along |
| 4 | 0.145 | `loco__walk_sideway_135_stop_002__A040` | 140 | 58 | (0.039,-0.018) | 0.3 | 2.2 | 9.0 | 0.42 | 0.54 | 0.00 | 1 | all-pass |
| 5 | 0.139 | `loco__Loop_Backward_Walk_001__A020` | 2720 | 42 | (0.048,0.026) | -1.6 | 1.5 | 10.1 | 0.40 | 0.75 | 0.00 | 1 | end_at_square |
| 6 | 0.127 | `loco__walk_backward_loop_007__A028` | 476 | 58 | (0.051,0.034) | 0.7 | 1.9 | 6.8 | 0.41 | 0.37 | 0.00 | 1 | all-pass |
| 7 | 0.124 | `loco__Loop_Backward_Walk_001__A020` | 2716 | 58 | (0.037,-0.013) | -3.0 | 0.8 | 12.8 | 0.61 | 0.77 | 0.00 | 1 | all-pass |
| 8 | 0.094 | `loco__walk_ff_stop_225_002__A060` | 90 | 42 | (0.053,0.013) | -4.4 | 4.7 | 13.0 | 0.49 | 0.81 | 0.00 | 1 | yaw |

### `fwd_walk_standard` (continuous_walk)
- target_xy_m: `(0.5, 0.0)` target_yaw_deg: `0.0`
- tol_xy_m: `0.2` tol_yaw_deg: `8.0` cross_axis_max_m: `0.1`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.991 | `loco__walk_180_R_001__A331` | 242 | 90 | (0.487,-0.006) | 0.2 | 10.6 | 32.8 | 0.48 | 0.45 | 0.00 | 1 | all-pass |
| 2 | 0.835 | `loco__neutral_dancecard_strafing_walk_007__A536_M` | 374 | 90 | (0.467,0.016) | -2.8 | 6.5 | 37.9 | 0.00 | 0.14 | 0.00 | 0 | all-pass |
| 3 | 0.738 | `loco__walk_hands_on_back_loop_003__A030` | 185 | 150 | (0.568,-0.020) | 3.1 | 5.5 | 42.8 | 0.00 | 0.08 | 0.00 | 2 | all-pass |
| 4 | 0.685 | `loco__walk_hands_on_back_stop_004__A061_M` | 88 | 90 | (0.444,-0.051) | 1.6 | 5.0 | 35.7 | 0.45 | 0.49 | 0.00 | 1 | all-pass |
| 5 | 0.552 | `loco__walk_forward_relax_impro_001__A001_M` | 22 | 90 | (0.456,-0.044) | -4.8 | 10.1 | 30.9 | 0.00 | 0.29 | 0.00 | 1 | all-pass |
| 6 | 0.409 | `loco__walk_forward_hips_amplified_001__A001` | 22 | 90 | (0.462,-0.028) | 7.1 | 9.6 | 25.8 | 0.00 | 1.00 | 0.00 | 0 | all-pass |
| 7 | 0.391 | `loco__medium_big_light_two_hands_walk_ff_stop_180_R_002__A508_M` | 132 | 90 | (0.376,-0.002) | -6.0 | 3.6 | 27.8 | 0.21 | 0.52 | 0.00 | 1 | all-pass |
| 8 | 0.391 | `standing__medium_big_light_two_hands_walk_ff_stop_180_R_002__A508_M` | 132 | 90 | (0.376,-0.002) | -6.0 | 3.6 | 27.8 | 0.21 | 0.52 | 0.00 | 1 | all-pass |

### `idle_stand` (idle)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `0.0`
- tol_xy_m: `0.04` tol_yaw_deg: `3.0` cross_axis_max_m: `0.04`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.846 | `loco__idle_vigilance_start_R_001__A502` | 33 | 45 | (-0.001,-0.000) | -0.0 | 0.1 | 0.1 | 0.43 | 1.00 | 0.93 | 0 | pelvis_z_band |
| 2 | 0.846 | `standing__idle_vigilance_start_R_001__A502` | 33 | 45 | (-0.001,-0.000) | -0.0 | 0.1 | 0.1 | 0.43 | 1.00 | 0.93 | 0 | pelvis_z_band |
| 3 | 0.824 | `loco__idle_vigilance_start_R_001__A502` | 77 | 45 | (0.001,-0.000) | 0.1 | 0.1 | 0.1 | 0.43 | 0.03 | 0.91 | 0 | pelvis_z_band |
| 4 | 0.824 | `standing__idle_vigilance_start_R_001__A502` | 77 | 45 | (0.001,-0.000) | 0.1 | 0.1 | 0.1 | 0.43 | 0.03 | 0.91 | 0 | pelvis_z_band |
| 5 | 0.815 | `loco__pull_shoudler_270_standing_R_002__A432_M` | 143 | 45 | (-0.001,-0.000) | 0.0 | 0.1 | 0.1 | 0.57 | 0.59 | 0.90 | 0 | pelvis_z_band |
| 6 | 0.812 | `loco__idle_vigilance_start_R_001__A502` | 55 | 45 | (0.001,-0.000) | -0.0 | 0.1 | 0.1 | 0.43 | 0.79 | 0.89 | 0 | pelvis_z_band |
| 7 | 0.812 | `standing__idle_vigilance_start_R_001__A502` | 55 | 45 | (0.001,-0.000) | -0.0 | 0.1 | 0.1 | 0.43 | 0.79 | 0.89 | 0 | pelvis_z_band |
| 8 | 0.811 | `loco__idle_vigilance_start_R_001__A502` | 44 | 45 | (0.000,-0.000) | -0.0 | 0.2 | 0.0 | 0.43 | 0.87 | 0.89 | 0 | pelvis_z_band |

### `lean_fwd_large` (static_upper_body)
- target_waist_pitch_deg: `20.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `6.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.003 | `loco__walk_forward_weakend_sick_impro_002__A001_M` | 0 | 50 | (0.019,-0.024) | 2.1 | 8.2 | 2.1 | 0.32 | 1.00 | 0.15 | 0 | waist_axis,feet_planted |
| 2 | 0.003 | `loco__walk_forward_weakend_sick_impro_002__A001_M` | 21 | 30 | (0.013,-0.028) | 2.2 | 8.2 | 1.8 | 0.37 | 1.00 | 0.13 | 0 | waist_axis,feet_planted |
| 3 | 0.002 | `loco__walk_forward_shoulder_amplified_002__A001` | 7 | 30 | (-0.005,0.001) | 0.4 | 6.4 | 1.8 | 0.39 | 1.00 | 0.30 | 0 | waist_axis,feet_planted |
| 4 | 0.002 | `loco__walk_forward_relax_003__A006_M` | 0 | 30 | (-0.000,0.007) | -1.1 | 5.9 | 2.7 | 0.20 | 1.00 | 0.46 | 0 | waist_axis |
| 5 | 0.001 | `loco__walk_forward_relax_003__A006_M` | 7 | 30 | (0.001,0.017) | -0.9 | 6.0 | 2.7 | 0.18 | 1.00 | 0.40 | 0 | waist_axis |
| 6 | 0.001 | `loco__walk_fast_forward_grab_normal_002__A008_M` | 14 | 30 | (-0.002,0.005) | 0.9 | 6.0 | 1.9 | 0.35 | 1.00 | 0.33 | 0 | waist_axis,feet_planted |
| 7 | 0.001 | `loco__walk_fast_forward_grab_normal_002__A008_M` | 0 | 50 | (-0.001,0.008) | 1.4 | 6.6 | 2.0 | 0.44 | 0.93 | 0.21 | 0 | waist_axis,feet_planted |
| 8 | 0.001 | `loco__walk_forward_shoulder_amplified_002__A001` | 0 | 30 | (-0.004,0.001) | 0.1 | 5.5 | 1.6 | 0.39 | 1.00 | 0.43 | 0 | waist_axis |

### `lean_fwd_medium` (static_upper_body)
- target_waist_pitch_deg: `14.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `5.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.088 | `loco__medium_big_heavy_two_hands_pick_up_front_high_R_001__A523_M` | 77 | 30 | (-0.006,0.003) | 0.8 | 14.8 | 1.0 | 0.01 | 0.82 | 0.11 | 0 | pelvis_z_band,feet_planted |
| 2 | 0.083 | `loco__medium_big_heavy_two_hands_pick_up_front_high_R_001__A523_M` | 72 | 50 | (-0.012,0.009) | -0.1 | 16.0 | 2.1 | 0.01 | 0.94 | 0.11 | 0 | pelvis_z_band,feet_planted |
| 3 | 0.069 | `loco__medium_big_heavy_two_hands_pick_up_front_high_R_001__A523_M` | 70 | 30 | (-0.029,0.014) | -0.5 | 15.8 | 2.0 | 0.01 | 0.86 | 0.10 | 0 | pelvis_z_band,feet_planted |
| 4 | 0.045 | `loco__medium_big_heavy_two_hands_pick_up_front_high_R_001__A523_M` | 68 | 70 | (-0.018,0.017) | -0.7 | 15.5 | 2.4 | 0.01 | 0.82 | 0.07 | 0 | pelvis_z_band,feet_planted |
| 5 | 0.036 | `loco__medium_heavy_two_hands_pick_up_front_low_R_001__A504_M` | 0 | 50 | (0.003,-0.001) | 0.1 | 8.6 | 0.5 | 0.46 | 1.00 | 0.12 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 6 | 0.035 | `loco__walk_forward_weakend_sick_impro_002__A001_M` | 0 | 50 | (0.019,-0.024) | 2.1 | 8.2 | 2.1 | 0.32 | 1.00 | 0.15 | 0 | waist_axis,feet_planted |
| 7 | 0.034 | `loco__medium_big_heavy_two_hands_pick_up_front_high_R_001__A523_M` | 66 | 90 | (-0.027,0.021) | -1.5 | 15.4 | 2.6 | 0.01 | 1.00 | 0.04 | 0 | pelvis_z_band,feet_planted |
| 8 | 0.032 | `loco__walk_forward_weakend_sick_impro_002__A001_M` | 21 | 30 | (0.013,-0.028) | 2.2 | 8.2 | 1.8 | 0.37 | 1.00 | 0.13 | 0 | waist_axis,feet_planted |

### `lean_fwd_small` (static_upper_body)
- target_waist_pitch_deg: `8.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `3.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.236 | `loco__walk_forward_relax_003__A006_M` | 0 | 30 | (-0.000,0.007) | -1.1 | 5.9 | 2.7 | 0.20 | 1.00 | 0.46 | 0 | pelvis_z_band |
| 2 | 0.215 | `loco__walk_forward_shoulder_amplified_002__A001` | 7 | 30 | (-0.005,0.001) | 0.4 | 6.4 | 1.8 | 0.39 | 1.00 | 0.30 | 0 | pelvis_z_band,feet_planted |
| 3 | 0.211 | `loco__walk_forward_relax_003__A006_M` | 7 | 30 | (0.001,0.017) | -0.9 | 6.0 | 2.7 | 0.18 | 1.00 | 0.40 | 0 | pelvis_z_band |
| 4 | 0.201 | `loco__walk_forward_shoulder_amplified_002__A001` | 0 | 30 | (-0.004,0.001) | 0.1 | 5.5 | 1.6 | 0.39 | 1.00 | 0.43 | 0 | pelvis_z_band |
| 5 | 0.169 | `loco__walk_fast_forward_grab_normal_002__A008_M` | 14 | 30 | (-0.002,0.005) | 0.9 | 6.0 | 1.9 | 0.35 | 1.00 | 0.33 | 0 | pelvis_z_band,feet_planted |
| 6 | 0.168 | `loco__walk_forward_professional_001__A001` | 7 | 30 | (-0.003,-0.000) | 0.1 | 4.7 | 1.7 | 0.35 | 1.00 | 0.62 | 0 | pelvis_z_band,waist_axis |
| 7 | 0.147 | `loco__walk_forward_professional_001__A001` | 0 | 50 | (-0.007,-0.000) | 0.0 | 5.9 | 1.5 | 0.36 | 1.00 | 0.25 | 0 | pelvis_z_band,feet_planted |
| 8 | 0.136 | `loco__walk_forward_weakend_sick_impro_002__A001_M` | 0 | 50 | (0.019,-0.024) | 2.1 | 8.2 | 2.1 | 0.32 | 1.00 | 0.15 | 0 | pelvis_z_band,feet_planted |

### `lean_left_large` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `3.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21 | 30 | (-0.002,0.000) | -0.0 | 0.3 | 0.1 | 0.59 | 0.99 | 0.90 | 0 | pelvis_z_band,waist_axis |
| 2 | 0.412 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 3 | 0.412 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 4 | 0.411 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 5 | 0.411 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 6 | 0.409 | `loco__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 7 | 0.409 | `standing__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 8 | 0.404 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 147 | 30 | (-0.000,-0.000) | -0.1 | 0.0 | 0.3 | 0.73 | 1.00 | 0.93 | 0 | pelvis_z_band,waist_axis |

### `lean_left_medium` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `3.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21 | 30 | (-0.002,0.000) | -0.0 | 0.3 | 0.1 | 0.59 | 0.99 | 0.90 | 0 | pelvis_z_band,waist_axis |
| 2 | 0.412 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 3 | 0.412 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 4 | 0.411 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 5 | 0.411 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 6 | 0.409 | `loco__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 7 | 0.409 | `standing__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 8 | 0.404 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 147 | 30 | (-0.000,-0.000) | -0.1 | 0.0 | 0.3 | 0.73 | 1.00 | 0.93 | 0 | pelvis_z_band,waist_axis |

### `lean_left_small` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `2.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21 | 30 | (-0.002,0.000) | -0.0 | 0.3 | 0.1 | 0.59 | 0.99 | 0.90 | 0 | pelvis_z_band,waist_axis |
| 2 | 0.412 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 3 | 0.412 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 4 | 0.411 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 5 | 0.411 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 6 | 0.409 | `loco__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 7 | 0.409 | `standing__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 8 | 0.404 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 147 | 30 | (-0.000,-0.000) | -0.1 | 0.0 | 0.3 | 0.73 | 1.00 | 0.93 | 0 | pelvis_z_band,waist_axis |

### `lean_right_large` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `3.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21 | 30 | (-0.002,0.000) | -0.0 | 0.3 | 0.1 | 0.59 | 0.99 | 0.90 | 0 | pelvis_z_band,waist_axis |
| 2 | 0.412 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 3 | 0.412 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 4 | 0.411 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 5 | 0.411 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 6 | 0.409 | `loco__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 7 | 0.409 | `standing__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 8 | 0.404 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 147 | 30 | (-0.000,-0.000) | -0.1 | 0.0 | 0.3 | 0.73 | 1.00 | 0.93 | 0 | pelvis_z_band,waist_axis |

### `lean_right_medium` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `3.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21 | 30 | (-0.002,0.000) | -0.0 | 0.3 | 0.1 | 0.59 | 0.99 | 0.90 | 0 | pelvis_z_band,waist_axis |
| 2 | 0.412 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 3 | 0.412 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 4 | 0.411 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 5 | 0.411 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 6 | 0.409 | `loco__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 7 | 0.409 | `standing__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 8 | 0.404 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 147 | 30 | (-0.000,-0.000) | -0.1 | 0.0 | 0.3 | 0.73 | 1.00 | 0.93 | 0 | pelvis_z_band,waist_axis |

### `lean_right_small` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `0.0`
- tol_waist_deg: `2.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.416 | `loco__walk_sideway_090_stop_001__A039_M` | 21 | 30 | (-0.002,0.000) | -0.0 | 0.3 | 0.1 | 0.59 | 0.99 | 0.90 | 0 | pelvis_z_band,waist_axis |
| 2 | 0.412 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 3 | 0.412 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 175 | 30 | (-0.000,0.000) | -0.0 | 0.2 | 0.2 | 0.73 | 1.00 | 0.95 | 0 | pelvis_z_band,waist_axis |
| 4 | 0.411 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 5 | 0.411 | `standing__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 168 | 30 | (-0.001,0.000) | -0.1 | 0.2 | 0.2 | 0.73 | 1.00 | 0.94 | 0 | pelvis_z_band,waist_axis |
| 6 | 0.409 | `loco__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 7 | 0.409 | `standing__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 217 | 30 | (0.002,0.000) | 0.0 | 0.4 | 0.3 | 0.24 | 1.00 | 0.85 | 0 | pelvis_z_band,waist_axis |
| 8 | 0.404 | `loco__big_light_two_hands_put_down_right_side_medium_R_001__A510_M` | 147 | 30 | (-0.000,-0.000) | -0.1 | 0.0 | 0.3 | 0.73 | 1.00 | 0.93 | 0 | pelvis_z_band,waist_axis |

### `side_left_step` (locomotion)
- target_xy_m: `(-0.25, 0.27)` target_yaw_deg: `0.0`
- tol_xy_m: `0.1` tol_yaw_deg: `8.0` cross_axis_max_m: `0.1`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.019 | `loco__walk_sideway_045_stop_003__A044` | 165 | 60 | (-0.129,0.261) | -0.7 | 3.6 | 7.9 | 0.08 | 0.93 | 0.00 | 1 | end_at_square |
| 2 | 0.019 | `loco__walk_sideway_045_stop_003__A044` | 30 | 60 | (-0.269,0.227) | -1.8 | 2.3 | 16.6 | 0.02 | 0.37 | 0.00 | 1 | end_at_square |
| 3 | 0.014 | `loco__walk_sideway_045_loop_005__A037` | 0 | 60 | (-0.173,0.221) | 3.2 | 3.0 | 13.1 | 0.04 | 1.00 | 0.00 | 1 | end_at_square |
| 4 | 0.009 | `loco__walk_sideway_045_stop_003__A044` | 160 | 80 | (-0.245,0.238) | 12.9 | 2.4 | 26.2 | 0.14 | 0.65 | 0.00 | 1 | yaw,end_at_square |
| 5 | 0.002 | `loco__walk_sideway_045_stop_001__A037` | 45 | 60 | (-0.274,0.217) | -4.1 | 8.0 | 25.4 | 0.00 | 1.00 | 0.00 | 1 | end_at_square |
| 6 | 0.001 | `loco__big_light_two_hands_right_side_high_to_behind_high_R_001__A525` | 90 | 120 | (-0.226,0.103) | 1.7 | 8.4 | 22.3 | 0.06 | 0.50 | 0.00 | 2 | xy_along,end_at_square,stride_count |
| 7 | 0.001 | `standing__big_light_two_hands_right_side_high_to_behind_high_R_001__A525` | 90 | 120 | (-0.226,0.103) | 1.7 | 8.4 | 22.3 | 0.06 | 0.50 | 0.00 | 2 | xy_along,end_at_square,stride_count |
| 8 | 0.001 | `loco__Sideway_Walk_Left_001__A019` | 480 | 60 | (-0.165,0.424) | 11.6 | 2.5 | 26.8 | 0.09 | 0.99 | 0.00 | 1 | cross_axis,yaw,end_at_square |

### `side_right_step` (locomotion)
- target_xy_m: `(-0.25, -0.27)` target_yaw_deg: `0.0`
- tol_xy_m: `0.1` tol_yaw_deg: `8.0` cross_axis_max_m: `0.1`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.243 | `loco__Sideway_Walk_Left_001__A017_M` | 795 | 60 | (-0.199,-0.260) | 6.1 | 8.1 | 23.3 | 0.57 | 0.88 | 0.00 | 1 | all-pass |
| 2 | 0.019 | `loco__Sideway_Walk_Left_001__A018_M` | 1095 | 60 | (-0.335,-0.275) | 6.3 | 4.9 | 46.4 | 0.19 | 0.81 | 0.00 | 2 | end_at_square,stride_count |
| 3 | 0.017 | `loco__Sideway_Walk_Left_001__A017_M` | 1365 | 60 | (-0.288,-0.305) | 13.7 | 6.3 | 35.7 | 0.41 | 0.75 | 0.00 | 1 | yaw |
| 4 | 0.014 | `loco__walk_sideway_090_loop_001__A029_M` | 135 | 60 | (-0.242,-0.443) | 3.6 | 8.1 | 19.8 | 0.35 | 0.28 | 0.00 | 1 | xy_along,cross_axis,end_at_square |
| 5 | 0.009 | `loco__big_light_one_hand_right_side_high_to_right_side_low_R_001__A521_M` | 45 | 60 | (-0.132,-0.216) | 2.9 | 13.2 | 23.0 | 0.05 | 0.67 | 0.00 | 1 | xy_along,end_at_square |
| 6 | 0.009 | `standing__big_light_one_hand_right_side_high_to_right_side_low_R_001__A521_M` | 45 | 60 | (-0.132,-0.216) | 2.9 | 13.2 | 23.0 | 0.05 | 0.67 | 0.00 | 1 | xy_along,end_at_square |
| 7 | 0.009 | `loco__walk_sideway_045_stop_002__A037_M` | 225 | 100 | (-0.245,-0.241) | 2.1 | 3.6 | 13.4 | 0.03 | 0.90 | 0.00 | 3 | end_at_square,stride_count |
| 8 | 0.006 | `loco__walk_sideway_045_stop_002__A037_M` | 400 | 100 | (-0.425,-0.256) | 7.3 | 3.3 | 24.5 | 0.74 | 0.25 | 0.00 | 2 | xy_along,cross_axis,stride_count |

### `torso_left_15deg` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `15.0`
- tol_waist_deg: `5.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.048 | `loco__mosquito_squash_neck_left_R_001__A502_M` | 126 | 75 | (0.003,-0.002) | 1.3 | 0.9 | 8.1 | 0.24 | 0.92 | 0.39 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 2 | 0.048 | `standing__mosquito_squash_neck_left_R_001__A502_M` | 126 | 75 | (0.003,-0.002) | 1.3 | 0.9 | 8.1 | 0.24 | 0.92 | 0.39 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 3 | 0.048 | `loco__mosquito_squash_neck_left_R_001__A502_M` | 120 | 60 | (0.007,-0.004) | 1.3 | 1.0 | 7.7 | 0.24 | 1.00 | 0.44 | 0 | pelvis_z_band,waist_axis |
| 4 | 0.048 | `standing__mosquito_squash_neck_left_R_001__A502_M` | 120 | 60 | (0.007,-0.004) | 1.3 | 1.0 | 7.7 | 0.24 | 1.00 | 0.44 | 0 | pelvis_z_band,waist_axis |
| 5 | 0.046 | `loco__mosquito_squash_neck_left_R_001__A502_M` | 135 | 60 | (0.001,-0.001) | 1.3 | 0.8 | 7.7 | 0.26 | 0.94 | 0.46 | 0 | pelvis_z_band,waist_axis |
| 6 | 0.046 | `standing__mosquito_squash_neck_left_R_001__A502_M` | 135 | 60 | (0.001,-0.001) | 1.3 | 0.8 | 7.7 | 0.26 | 0.94 | 0.46 | 0 | pelvis_z_band,waist_axis |
| 7 | 0.036 | `loco__mosquito_squash_neck_left_R_001__A502_M` | 132 | 45 | (0.004,-0.000) | 1.0 | 0.8 | 7.0 | 0.24 | 0.99 | 0.52 | 0 | pelvis_z_band,waist_axis |
| 8 | 0.036 | `standing__mosquito_squash_neck_left_R_001__A502_M` | 132 | 45 | (0.004,-0.000) | 1.0 | 0.8 | 7.0 | 0.24 | 0.99 | 0.52 | 0 | pelvis_z_band,waist_axis |

### `torso_left_30deg` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `30.0`
- tol_waist_deg: `6.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.000 | `loco__arc_walk_left_stop_004__A037` | 245 | 30 | (0.030,-0.036) | -19.4 | 5.0 | 31.0 | 0.78 | 0.92 | 0.00 | 0 | pelvis_z_band,feet_planted |
| 2 | 0.000 | `loco__arc_walk_left_stop_004__A037` | 252 | 30 | (0.046,0.011) | -16.8 | 1.5 | 19.1 | 0.68 | 0.82 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 3 | 0.000 | `loco__arc_walk_left_stop_004__A037` | 252 | 50 | (0.049,0.009) | -14.1 | 1.5 | 19.1 | 0.71 | 0.88 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 4 | 0.000 | `loco__idle_one_foot_left_001__A043_M` | 132 | 90 | (0.011,-0.005) | -20.0 | 1.0 | 22.7 | 0.00 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 5 | 0.000 | `standing__idle_one_foot_left_001__A043_M` | 132 | 90 | (0.011,-0.005) | -20.0 | 1.0 | 22.7 | 0.00 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 6 | 0.000 | `loco__idle_one_foot_left_004__A034_M` | 231 | 30 | (0.013,-0.012) | -16.8 | 0.4 | 18.0 | 0.01 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 7 | 0.000 | `standing__idle_one_foot_left_004__A034_M` | 231 | 30 | (0.013,-0.012) | -16.8 | 0.4 | 18.0 | 0.01 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 8 | 0.000 | `loco__Sideway_Walk_Left_001__A017_M` | 3213 | 30 | (-0.139,-0.041) | -4.5 | 5.1 | 19.0 | 0.42 | 0.89 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |

### `torso_left_40deg` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `40.0`
- tol_waist_deg: `6.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.000 | `loco__arc_walk_left_stop_004__A037` | 245 | 30 | (0.030,-0.036) | -19.4 | 5.0 | 31.0 | 0.78 | 0.92 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 2 | 0.000 | `loco__step_rotate_idle_000_001__A031` | 91 | 30 | (-0.027,-0.072) | 23.2 | 2.6 | 26.5 | 0.03 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 3 | 0.000 | `loco__step_rotate_idle_000_001__A031` | 51 | 70 | (-0.023,-0.079) | 23.7 | 2.4 | 26.6 | 0.03 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 4 | 0.000 | `loco__idle_one_foot_left__003__A024_M` | 102 | 70 | (0.010,-0.035) | -29.3 | 3.6 | 31.0 | 0.00 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 5 | 0.000 | `standing__idle_one_foot_left__003__A024_M` | 102 | 70 | (0.010,-0.035) | -29.3 | 3.6 | 31.0 | 0.00 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 6 | 0.000 | `loco__idle_one_foot_left__003__A024_M` | 120 | 50 | (-0.002,-0.005) | -21.0 | 1.9 | 25.9 | 0.00 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 7 | 0.000 | `standing__idle_one_foot_left__003__A024_M` | 120 | 50 | (-0.002,-0.005) | -21.0 | 1.9 | 25.9 | 0.00 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 8 | 0.000 | `loco__idle_one_foot_left_004__A034_M` | 238 | 30 | (0.029,-0.020) | -33.2 | 1.4 | 33.6 | 0.01 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |

### `torso_right_15deg` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `-15.0`
- tol_waist_deg: `5.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.309 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 126 | 30 | (0.013,-0.006) | 0.2 | 3.8 | 14.1 | 0.47 | 0.93 | 0.47 | 0 | pelvis_z_band |
| 2 | 0.220 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 119 | 30 | (0.010,0.006) | 0.4 | 5.3 | 14.9 | 0.46 | 0.97 | 0.38 | 0 | pelvis_z_band,feet_planted |
| 3 | 0.220 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 112 | 30 | (0.006,0.013) | 0.3 | 5.2 | 14.1 | 0.40 | 1.00 | 0.38 | 0 | pelvis_z_band,feet_planted |
| 4 | 0.210 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 99 | 45 | (-0.006,0.013) | 1.3 | 3.4 | 16.5 | 0.41 | 1.00 | 0.30 | 0 | pelvis_z_band,feet_planted |
| 5 | 0.194 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 105 | 30 | (-0.007,0.018) | 0.2 | 4.1 | 12.5 | 0.40 | 1.00 | 0.35 | 0 | pelvis_z_band,feet_planted |
| 6 | 0.190 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 110 | 45 | (0.008,0.006) | -1.3 | 5.0 | 16.6 | 0.47 | 0.95 | 0.36 | 0 | pelvis_z_band,feet_planted |
| 7 | 0.054 | `loco__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 112 | 30 | (-0.003,-0.001) | -3.7 | 2.4 | 10.3 | 0.01 | 1.00 | 0.15 | 0 | pelvis_z_band,feet_planted |
| 8 | 0.054 | `standing__medium_big_light_one_hand_right_side_high_to_behind_high_R_001__A526` | 112 | 30 | (-0.003,-0.001) | -3.7 | 2.4 | 10.3 | 0.01 | 1.00 | 0.15 | 0 | pelvis_z_band,feet_planted |

### `torso_right_30deg` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `-30.0`
- tol_waist_deg: `6.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.003 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 108 | 50 | (0.006,0.005) | -1.3 | 4.9 | 17.7 | 0.48 | 0.92 | 0.35 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 2 | 0.001 | `loco__small_heavy_two_hands_put_down_right_side_medium_R_001__A508_M` | 70 | 30 | (0.024,0.005) | -11.0 | 3.7 | 25.7 | 0.54 | 1.00 | 0.00 | 0 | pelvis_z_band,feet_planted |
| 3 | 0.001 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 96 | 50 | (-0.005,0.010) | 1.4 | 3.2 | 15.7 | 0.43 | 1.00 | 0.30 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 4 | 0.001 | `loco__big_heavy_one_hand_pick_up_right_side_medium_R_002__A510_M` | 49 | 30 | (-0.014,-0.003) | -7.6 | 4.0 | 21.2 | 0.41 | 1.00 | 0.01 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 5 | 0.001 | `loco__small_light_two_hands_right_side_high_to_right_side_medium_R_001__A518_M` | 63 | 30 | (0.007,-0.010) | -7.4 | 4.4 | 24.4 | 0.41 | 0.93 | 0.00 | 1 | pelvis_z_band,feet_planted |
| 6 | 0.001 | `standing__small_light_two_hands_right_side_high_to_right_side_medium_R_001__A518_M` | 63 | 30 | (0.007,-0.010) | -7.4 | 4.4 | 24.4 | 0.41 | 0.93 | 0.00 | 1 | pelvis_z_band,feet_planted |
| 7 | 0.000 | `loco__small_heavy_one_hand_right_side_high_to_front_high_R_001__A526` | 119 | 30 | (0.010,0.006) | 0.4 | 5.3 | 14.9 | 0.46 | 0.97 | 0.38 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 8 | 0.000 | `loco__big_heavy_one_hand_pick_up_right_side_medium_R_002__A510_M` | 56 | 30 | (-0.023,-0.001) | -11.8 | 8.5 | 27.6 | 0.25 | 1.00 | 0.00 | 0 | pelvis_z_band,feet_planted |

### `torso_right_40deg` (static_upper_body)
- target_waist_pitch_deg: `0.0` target_waist_yaw_deg: `-40.0`
- tol_waist_deg: `6.0` end_at_apex_min: `0.7` feet_planted_min: `0.4`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.000 | `loco__small_heavy_two_hands_put_down_right_side_medium_R_001__A508_M` | 63 | 30 | (0.024,0.007) | -18.6 | 8.1 | 36.1 | 0.57 | 1.00 | 0.00 | 0 | pelvis_z_band,feet_planted |
| 2 | 0.000 | `loco__big_heavy_one_hand_pick_up_right_side_medium_R_002__A510_M` | 17 | 70 | (-0.023,-0.001) | -11.5 | 7.5 | 28.4 | 0.23 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 3 | 0.000 | `loco__small_heavy_two_hands_put_down_right_side_medium_R_001__A508_M` | 70 | 30 | (0.024,0.005) | -11.0 | 3.7 | 25.7 | 0.54 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 4 | 0.000 | `loco__big_heavy_one_hand_pick_up_right_side_medium_R_002__A510_M` | 56 | 30 | (-0.023,-0.001) | -11.8 | 8.5 | 27.6 | 0.25 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 5 | 0.000 | `loco__medium_heavy_one_hand_right_side_high_to_front_high_R_003__A524_M` | 91 | 30 | (-0.002,-0.003) | -16.6 | 3.2 | 29.4 | 0.12 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 6 | 0.000 | `loco__medium_big_heavy_two_hands_right_side_low_to_right_side_medium_R_001__A523_M` | 126 | 30 | (-0.009,-0.082) | -12.2 | 8.0 | 27.4 | 0.42 | 1.00 | 0.00 | 0 | pelvis_z_band,waist_axis,feet_planted |
| 7 | 0.000 | `loco__medium_heavy_two_hands_put_down_right_side_high_R_001__A523_M` | 63 | 30 | (-0.013,0.020) | -21.9 | 5.2 | 37.9 | 0.34 | 1.00 | 0.00 | 0 | pelvis_z_band,feet_planted |
| 8 | 0.000 | `loco__medium_heavy_one_hand_right_side_high_to_front_high_R_003__A524_M` | 84 | 50 | (0.013,0.002) | -20.6 | 7.3 | 37.2 | 0.09 | 1.00 | 0.00 | 0 | pelvis_z_band,feet_planted |

### `turn_left_15deg` (locomotion)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `15.0`
- tol_xy_m: `0.05` tol_yaw_deg: `3.0` cross_axis_max_m: `0.05`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.745 | `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M` | 1350 | 60 | (0.004,-0.004) | 15.6 | 2.8 | 5.9 | 0.79 | 1.00 | 0.00 | 0 | all-pass |
| 2 | 0.630 | `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M` | 1071 | 30 | (0.001,-0.013) | 13.8 | 1.7 | 8.7 | 0.85 | 0.77 | 0.00 | 0 | all-pass |
| 3 | 0.608 | `loco__Step_Rotate_Reaction_Idle_0315_001__A017_M` | 238 | 30 | (-0.010,0.019) | 14.7 | 1.5 | 12.0 | 0.88 | 0.93 | 0.00 | 0 | all-pass |
| 4 | 0.581 | `loco__idle_turn_045_R_long_002__A548` | 352 | 45 | (-0.009,0.002) | 14.8 | 4.1 | 3.5 | 0.62 | 0.59 | 0.00 | 0 | all-pass |
| 5 | 0.531 | `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M` | 1350 | 75 | (-0.003,-0.018) | 13.9 | 2.8 | 5.9 | 0.80 | 0.64 | 0.00 | 0 | all-pass |
| 6 | 0.522 | `loco__Step_Rotate_Reaction_Idle_0315_001__A017_M` | 1947 | 45 | (0.004,0.021) | 15.2 | 4.9 | 15.9 | 0.75 | 0.82 | 0.00 | 0 | all-pass |
| 7 | 0.491 | `loco__Turn_Start_Walk_0225_003__A020_M` | 483 | 30 | (-0.022,-0.013) | 15.4 | 2.2 | 5.6 | 0.84 | 0.89 | 0.00 | 0 | all-pass |
| 8 | 0.474 | `loco__Turn_Start_Walk_0045_001__A018` | 203 | 30 | (-0.011,-0.017) | 13.4 | 0.8 | 3.9 | 0.86 | 0.10 | 0.00 | 0 | all-pass |

### `turn_left_30deg` (locomotion)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `30.0`
- tol_xy_m: `0.06` tol_yaw_deg: `4.0` cross_axis_max_m: `0.06`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.610 | `loco__idle_turn_360_R_003__A265_M` | 85 | 70 | (-0.009,-0.003) | 30.1 | 1.7 | 3.0 | 0.64 | 0.40 | 0.00 | 0 | all-pass |
| 2 | 0.556 | `loco__Step_Rotate_Reaction_Idle_0315_001__A017_M` | 889 | 30 | (0.014,-0.007) | 31.5 | 0.7 | 2.0 | 0.73 | 0.56 | 0.00 | 0 | all-pass |
| 3 | 0.504 | `loco__idle_turn_360_R_003__A265_M` | 84 | 50 | (-0.005,-0.009) | 31.9 | 1.8 | 4.2 | 0.66 | 0.77 | 0.00 | 0 | all-pass |
| 4 | 0.499 | `loco__Turn_Start_Walk_0045_001__A018` | 1479 | 70 | (0.004,0.010) | 30.3 | 1.1 | 7.0 | 0.53 | 0.34 | 0.00 | 0 | all-pass |
| 5 | 0.439 | `loco__Turn_Start_Walk_0045_001__A018` | 1500 | 50 | (0.011,0.004) | 29.8 | 1.2 | 6.1 | 0.47 | 0.46 | 0.00 | 0 | all-pass |
| 6 | 0.434 | `loco__idle_turn_360_R_003__A265_M` | 84 | 30 | (-0.006,-0.006) | 32.3 | 1.7 | 4.2 | 0.62 | 0.84 | 0.00 | 0 | all-pass |
| 7 | 0.417 | `loco__step_rotate_idle_000_002__A037` | 132 | 50 | (0.002,-0.023) | 29.2 | 1.2 | 6.5 | 0.58 | 0.60 | 0.00 | 0 | all-pass |
| 8 | 0.403 | `loco__Step_Rotate_Reaction_Idle_0315_001__A017_M` | 594 | 90 | (-0.033,-0.002) | 28.5 | 2.9 | 15.4 | 0.84 | 0.98 | 0.00 | 0 | all-pass |

### `turn_left_45deg` (locomotion)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `45.0`
- tol_xy_m: `0.08` tol_yaw_deg: `5.0` cross_axis_max_m: `0.08`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.685 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 1040 | 105 | (0.004,0.006) | 45.8 | 1.6 | 10.7 | 0.72 | 0.21 | 0.00 | 1 | all-pass |
| 2 | 0.665 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 1050 | 85 | (-0.022,0.022) | 44.9 | 1.7 | 11.1 | 0.89 | 0.27 | 0.00 | 2 | all-pass |
| 3 | 0.634 | `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M` | 1342 | 45 | (-0.023,-0.022) | 45.5 | 2.1 | 6.2 | 0.88 | 0.08 | 0.00 | 0 | all-pass |
| 4 | 0.547 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 1456 | 105 | (-0.018,0.022) | 44.3 | 1.8 | 7.3 | 0.71 | 0.06 | 0.00 | 1 | all-pass |
| 5 | 0.528 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 1534 | 105 | (0.031,-0.005) | 45.6 | 1.6 | 6.2 | 0.72 | 0.12 | 0.00 | 0 | all-pass |
| 6 | 0.481 | `loco__Step_Rotate_Reaction_Idle_0315_001__A017_M` | 1176 | 85 | (0.023,0.009) | 45.5 | 1.4 | 4.7 | 0.59 | 0.40 | 0.00 | 1 | all-pass |
| 7 | 0.450 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 748 | 45 | (-0.022,0.006) | 43.4 | 1.1 | 8.9 | 0.59 | 0.06 | 0.00 | 1 | all-pass |
| 8 | 0.404 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 1470 | 85 | (-0.024,0.030) | 43.7 | 2.2 | 6.9 | 0.69 | 0.31 | 0.00 | 1 | all-pass |

### `turn_left_90deg` (locomotion)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `90.0`
- tol_xy_m: `0.1` tol_yaw_deg: `8.0` cross_axis_max_m: `0.1`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.734 | `loco__idle_turn_270_R_003__A235_M` | 0 | 90 | (0.008,0.017) | 90.0 | 7.6 | 7.4 | 0.79 | 0.09 | 0.00 | 2 | all-pass |
| 2 | 0.708 | `loco__idle_turn_270_R_003__A235_M` | 0 | 120 | (0.017,0.013) | 89.0 | 7.6 | 7.4 | 0.79 | 0.04 | 0.00 | 2 | all-pass |
| 3 | 0.648 | `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M` | 770 | 90 | (0.027,0.029) | 91.0 | 3.6 | 8.0 | 0.91 | 0.16 | 0.00 | 1 | all-pass |
| 4 | 0.633 | `loco__step_rotate_idle_090_002__A035` | 0 | 150 | (0.015,0.031) | 89.7 | 3.2 | 12.1 | 0.81 | 0.02 | 0.00 | 3 | all-pass |
| 5 | 0.619 | `loco__idle_turn_270_001__A056_M` | 0 | 120 | (0.021,-0.002) | 87.3 | 1.2 | 11.6 | 0.76 | 0.00 | 0.00 | 2 | all-pass |
| 6 | 0.574 | `loco__idle_turn_270_R_003__A235_M` | 22 | 90 | (0.021,0.029) | 87.5 | 7.0 | 6.2 | 0.82 | 0.14 | 0.00 | 2 | all-pass |
| 7 | 0.572 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 1320 | 120 | (-0.026,0.025) | 90.3 | 1.9 | 6.5 | 0.74 | 0.26 | 0.00 | 2 | all-pass |
| 8 | 0.563 | `loco__Step_Rotate_Reaction_Idle_0315_001__A017_M` | 1260 | 120 | (0.024,-0.009) | 88.4 | 4.2 | 13.0 | 0.66 | 0.14 | 0.00 | 1 | all-pass |

### `turn_right_15deg` (locomotion)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `-15.0`
- tol_xy_m: `0.05` tol_yaw_deg: `3.0` cross_axis_max_m: `0.05`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.733 | `loco__Turn_Start_Walk_0045_001__A020_M` | 2037 | 30 | (-0.002,-0.014) | -14.9 | 2.3 | 12.0 | 0.86 | 1.00 | 0.00 | 0 | all-pass |
| 2 | 0.589 | `loco__small_light_two_hands_behind_high_to_right_side_high_R_001__A527_M` | 144 | 75 | (0.004,0.002) | -16.2 | 6.3 | 24.5 | 0.70 | 0.95 | 0.00 | 0 | all-pass |
| 3 | 0.589 | `standing__small_light_two_hands_behind_high_to_right_side_high_R_001__A527_M` | 144 | 75 | (0.004,0.002) | -16.2 | 6.3 | 24.5 | 0.70 | 0.95 | 0.00 | 0 | all-pass |
| 4 | 0.586 | `loco__small_heavy_two_hands_pick_up_right_side_high_R_001__A520_M` | 98 | 30 | (-0.010,0.011) | -16.0 | 5.1 | 13.2 | 0.78 | 0.99 | 0.00 | 0 | all-pass |
| 5 | 0.560 | `loco__Turn_Start_Walk_0045_001__A020_M` | 150 | 60 | (-0.004,-0.005) | -13.8 | 3.4 | 14.6 | 0.67 | 0.92 | 0.00 | 0 | all-pass |
| 6 | 0.559 | `loco__Turn_Start_Walk_0045_001__A020_M` | 1476 | 75 | (0.007,-0.006) | -14.6 | 1.8 | 11.4 | 0.61 | 0.96 | 0.00 | 0 | all-pass |
| 7 | 0.533 | `loco__idle_one_foot_right_003__A025` | 465 | 60 | (-0.004,0.002) | -14.2 | 2.8 | 14.7 | 0.58 | 0.74 | 0.00 | 1 | all-pass |
| 8 | 0.533 | `standing__idle_one_foot_right_003__A025` | 465 | 60 | (-0.004,0.002) | -14.2 | 2.8 | 14.7 | 0.58 | 0.74 | 0.00 | 1 | all-pass |

### `turn_right_30deg` (locomotion)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `-30.0`
- tol_xy_m: `0.06` tol_yaw_deg: `4.0` cross_axis_max_m: `0.06`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.672 | `loco__big_heavy_one_hand_right_side_high_to_behind_high_R_002__A524_M` | 224 | 30 | (-0.010,0.020) | -30.5 | 6.7 | 19.6 | 0.89 | 0.97 | 0.00 | 0 | all-pass |
| 2 | 0.569 | `loco__small_heavy_two_hands_pick_up_right_side_high_R_001__A520_M` | 51 | 70 | (-0.022,-0.004) | -31.6 | 12.5 | 30.8 | 0.88 | 1.00 | 0.00 | 0 | all-pass |
| 3 | 0.566 | `loco__small_heavy_two_hands_pick_up_right_side_high_R_001__A520_M` | 44 | 90 | (-0.024,0.011) | -30.6 | 12.2 | 29.7 | 0.84 | 1.00 | 0.00 | 0 | all-pass |
| 4 | 0.484 | `loco__medium_big_light_two_hands_front_high_to_right_side_high_R_001__A527_M` | 112 | 30 | (-0.007,0.021) | -28.3 | 7.6 | 32.0 | 0.76 | 1.00 | 0.00 | 0 | all-pass |
| 5 | 0.484 | `standing__medium_big_light_two_hands_front_high_to_right_side_high_R_001__A527_M` | 112 | 30 | (-0.007,0.021) | -28.3 | 7.6 | 32.0 | 0.76 | 1.00 | 0.00 | 0 | all-pass |
| 6 | 0.480 | `loco__small_heavy_two_hands_right_side_high_to_right_side_high_R_001__A526_M` | 88 | 90 | (0.003,0.004) | -30.2 | 8.1 | 33.5 | 0.49 | 0.91 | 0.00 | 0 | all-pass |
| 7 | 0.480 | `loco__small_light_two_hands_behind_high_to_right_side_high_R_001__A527_M` | 119 | 70 | (-0.003,0.010) | -30.3 | 7.3 | 39.2 | 0.51 | 0.99 | 0.00 | 0 | all-pass |
| 8 | 0.480 | `standing__small_light_two_hands_behind_high_to_right_side_high_R_001__A527_M` | 119 | 70 | (-0.003,0.010) | -30.3 | 7.3 | 39.2 | 0.51 | 0.99 | 0.00 | 0 | all-pass |

### `turn_right_45deg` (locomotion)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `-45.0`
- tol_xy_m: `0.08` tol_yaw_deg: `5.0` cross_axis_max_m: `0.08`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.731 | `loco__Step_Rotate_Reaction_Idle_0135_001__A017_M` | 1456 | 65 | (-0.007,-0.004) | -45.2 | 3.4 | 6.2 | 0.75 | 0.36 | 0.00 | 1 | all-pass |
| 2 | 0.714 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019_M` | 1040 | 105 | (0.004,-0.006) | -45.8 | 1.8 | 10.8 | 0.75 | 0.20 | 0.00 | 1 | all-pass |
| 3 | 0.691 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019_M` | 1050 | 85 | (-0.022,-0.022) | -45.0 | 1.9 | 11.2 | 0.93 | 0.26 | 0.00 | 2 | all-pass |
| 4 | 0.571 | `loco__Step_Rotate_Reaction_Idle_0135_001__A017_M` | 1197 | 85 | (-0.018,0.005) | -42.9 | 2.4 | 6.4 | 0.76 | 0.06 | 0.00 | 0 | all-pass |
| 5 | 0.561 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019_M` | 1456 | 105 | (-0.018,-0.022) | -44.3 | 1.8 | 7.3 | 0.73 | 0.06 | 0.00 | 1 | all-pass |
| 6 | 0.552 | `loco__Step_Rotate_Reaction_Idle_0135_001__A017_M` | 78 | 105 | (-0.027,-0.006) | -43.9 | 4.1 | 6.4 | 0.74 | 0.32 | 0.00 | 1 | all-pass |
| 7 | 0.546 | `loco__Step_Rotate_Reaction_Idle_0135_001__A019_M` | 1534 | 105 | (0.031,0.005) | -45.5 | 1.3 | 6.3 | 0.74 | 0.13 | 0.00 | 0 | all-pass |
| 8 | 0.524 | `loco__Step_Rotate_Reaction_Idle_0135_001__A017_M` | 231 | 85 | (-0.032,-0.003) | -45.5 | 2.7 | 5.5 | 0.73 | 0.17 | 0.00 | 0 | all-pass |

### `turn_right_90deg` (locomotion)
- target_xy_m: `(0.0, 0.0)` target_yaw_deg: `-90.0`
- tol_xy_m: `0.1` tol_yaw_deg: `8.0` cross_axis_max_m: `0.1`

| Rank | Score | motion_key | start | N | xy_m | yaw_deg | waist_pitch_deg | waist_yaw_deg | end_sq | end_apex | feet | strides | gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.865 | `loco__step_rotate_idle_090_003__A023_M` | 37 | 150 | (-0.004,0.019) | -89.8 | 3.8 | 9.1 | 0.93 | 0.06 | 0.00 | 2 | all-pass |
| 2 | 0.799 | `loco__step_rotate_idle_090_003__A023_M` | 0 | 120 | (-0.006,0.030) | -89.4 | 4.1 | 8.5 | 0.97 | 0.07 | 0.00 | 3 | all-pass |
| 3 | 0.774 | `loco__step_rotate_idle_090_003__A023_M` | 44 | 90 | (0.013,0.010) | -87.5 | 2.7 | 7.8 | 0.90 | 0.13 | 0.00 | 2 | all-pass |
| 4 | 0.770 | `loco__step_rotate_idle_090_003__A023_M` | 0 | 150 | (-0.007,0.020) | -88.6 | 4.1 | 8.5 | 0.87 | 0.05 | 0.00 | 2 | all-pass |
| 5 | 0.768 | `loco__Step_Rotate_Reaction_Idle_0270_001__A018` | 444 | 150 | (-0.003,-0.030) | -90.1 | 3.6 | 8.9 | 0.92 | 0.08 | 0.00 | 0 | all-pass |
| 6 | 0.762 | `loco__step_rotate_idle_090_003__A023_M` | 30 | 120 | (-0.002,0.022) | -88.5 | 4.0 | 8.3 | 0.87 | 0.01 | 0.00 | 3 | all-pass |
| 7 | 0.753 | `loco__step_rotate_idle_090_003__A023_M` | 22 | 90 | (-0.000,0.030) | -88.8 | 4.0 | 7.9 | 0.92 | 0.06 | 0.00 | 2 | all-pass |
| 8 | 0.745 | `loco__Step_Rotate_Reaction_Idle_0270_001__A018` | 990 | 120 | (-0.027,-0.005) | -90.2 | 2.9 | 12.1 | 0.86 | 0.27 | 0.00 | 0 | all-pass |

---
Re-run the curator after editing the bins YAML or pinning rows in the registry.
To pin a candidate: copy its `motion_key`, `start_frame`, `n_frames` into the registry YAML row for that bin and set `pinned: true`.
