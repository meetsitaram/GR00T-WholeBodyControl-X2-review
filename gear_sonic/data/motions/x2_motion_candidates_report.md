# X2 motion candidates — 2026-05-11 12:07:45

Mined 2550 clips from `x2_ultra_bones_seed.pkl` via MuJoCo FK.
Hard ceiling on foot-lift: 0.080 m (excluded from every list below).

Columns:
* `dur` — clip duration (s)
* `fl_max` / `fl_mean` — foot-lift max / mean (m)
* `pz_min` / `pz_drop` — pelvis Z min / drop from baseline (m)
* `fwd` / `back` / `side` — body-frame speed (m/s)
* `pitch` — sustained waist+pelvis forward pitch (deg, ≥0.5s held)
* `dyaw` — net heading delta (deg)
* `strides` — stride count (FK-based, foot-Z zero crossings/2)
* `sq` — end-at-square score (0..1)
* `fp` — feet-planted score (0..1; 1=feet flat)

## Forward continuous walk (0.30–0.80 m/s)  (top 12 of 12)

| # | motion_key | T | dur | fl_max | pz_drop | fwd | back | side | pitch | dyaw | yaw_osc | strides | sq |
| - | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 1 | `loco__walk_hands_on_back_180_loop_R_001__A422_M` | 161 | 5.37 | 0.157 | 0.029 | +0.623 | -0.623 | 0.028 | 21.5 | -5.1 | 46 | 3 | 0.21 |
| 2 | `loco__walk_hands_on_back_180_stop_R_001__A267_M` | 159 | 5.30 | 0.169 | 0.029 | +0.583 | -0.583 | 0.015 | 24.5 | -2.1 | 48 | 4 | 0.62 |
| 3 | `loco__walk_hands_on_back_start_003__A096` | 236 | 7.87 | 0.159 | 0.030 | +0.615 | -0.615 | 0.011 | 4.7 | +0.2 | 54 | 5 | 0.59 |
| 4 | `loco__walk_hands_on_back_180_start_R_001__A215` | 205 | 6.83 | 0.149 | 0.025 | +0.472 | -0.472 | 0.010 | 8.0 | +19.2 | 41 | 5 | 0.00 |
| 5 | `loco__walk_hands_on_back_180_start_R_002__A267` | 192 | 6.40 | 0.172 | 0.027 | +0.545 | -0.545 | 0.007 | 4.2 | -19.4 | 49 | 4 | 0.00 |
| 6 | `loco__walk_hands_on_back_180_start_R_002__A214_M` | 191 | 6.37 | 0.155 | 0.027 | +0.468 | -0.468 | 0.002 | 8.2 | -7.2 | 45 | 3 | 0.01 |
| 7 | `loco__walk_forward_loop_003__A034` | 335 | 11.17 | 0.146 | 0.026 | +0.513 | -0.513 | 0.012 | 3.7 | -0.1 | 53 | 8 | 0.76 |
| 8 | `loco__neutral_stoop_down_R_007__A098_M` | 180 | 6.00 | 0.173 | 0.053 | +0.420 | -0.420 | 0.025 | 9.1 | +1.0 | 44 | 3 | 0.44 |
| 9 | `loco__walking_quip_180_R_002__A431_M` | 250 | 8.33 | 0.158 | 0.025 | +0.458 | -0.458 | 0.016 | 7.3 | +2.0 | 50 | 5 | 0.83 |
| 10 | `loco__walk_forward_stop_001__A028` | 134 | 4.47 | 0.180 | 0.030 | +0.454 | -0.454 | 0.009 | 2.9 | +3.1 | 52 | 2 | 0.89 |
| 11 | `loco__big_heavy_two_hands_walk_ff_stop_180_R_001__A511` | 177 | 5.90 | 0.168 | 0.025 | +0.424 | -0.424 | 0.003 | 13.9 | -3.4 | 52 | 3 | 0.47 |
| 12 | `loco__wander_R_001__A432` | 554 | 18.47 | 0.137 | 0.018 | +0.294 | -0.294 | 0.008 | 5.9 | +0.6 | 48 | 12 | 0.93 |

## Forward short step (0.15–0.40 m/s, 1–2 strides)  (top 1 of 1)

| # | motion_key | T | dur | fl_max | pz_drop | fwd | back | side | pitch | dyaw | yaw_osc | strides | sq |
| - | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 1 | `loco__walk_forward_stop_002__A026_M` | 274 | 9.13 | 0.171 | 0.028 | +0.167 | -0.167 | 0.002 | 6.8 | +2.3 | 54 | 2 | 0.87 |

## Backward walk (0.20–0.70 m/s)  (top 10 of 10)

| # | motion_key | T | dur | fl_max | pz_drop | fwd | back | side | pitch | dyaw | yaw_osc | strides | sq |
| - | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 1 | `loco__walk_ff_start_360_R_normal_pace_001__A443` | 207 | 6.90 | 0.106 | 0.021 | -0.453 | +0.453 | 0.011 | 3.9 | -22.4 | 38 | 4 | 0.00 |
| 2 | `loco__medium_heavy_two_hands_walk_ff_start_360_R_001__A504_M` | 256 | 8.53 | 0.143 | 0.024 | -0.353 | +0.353 | 0.001 | 10.3 | +25.0 | 30 | 5 | 0.01 |
| 3 | `loco__small_light_one_hand_walk_ff_start_360_R_001__A507` | 214 | 7.13 | 0.124 | 0.022 | -0.403 | +0.403 | 0.018 | 6.1 | +14.6 | 38 | 4 | 0.00 |
| 4 | `standing__small_light_one_hand_walk_ff_start_360_R_001__A507` | 214 | 7.13 | 0.124 | 0.022 | -0.403 | +0.403 | 0.018 | 6.1 | +14.6 | 38 | 4 | 0.00 |
| 5 | `loco__walk_backward_start_001__A028` | 138 | 4.60 | 0.126 | 0.025 | -0.328 | +0.328 | 0.005 | 4.2 | +21.9 | 35 | 3 | 0.01 |
| 6 | `loco__walk_backward_start_002__A028_M` | 123 | 4.10 | 0.100 | 0.024 | -0.322 | +0.322 | 0.026 | 3.6 | -8.7 | 38 | 3 | 0.32 |
| 7 | `loco__walk_backward_stop_003__A042` | 193 | 6.43 | 0.108 | 0.020 | -0.329 | +0.329 | 0.018 | 2.7 | +2.9 | 38 | 4 | 0.92 |
| 8 | `loco__walk_backward_loop_005__A028_M` | 596 | 19.87 | 0.127 | 0.023 | -0.341 | +0.341 | 0.007 | 5.1 | -1.5 | 39 | 12 | 0.75 |
| 9 | `loco__walk_backward_loop_005__A028` | 596 | 19.87 | 0.128 | 0.023 | -0.341 | +0.341 | 0.007 | 8.2 | +1.5 | 39 | 12 | 0.77 |
| 10 | `loco__proud_walk_ff_360_R_001__A545` | 997 | 33.23 | 0.179 | 0.022 | -0.200 | +0.200 | 0.002 | 13.5 | -7.2 | 32 | 12 | 0.85 |

## Side step (|by| ≥ 0.20 m/s)  (top 10 of 10)

| # | motion_key | T | dur | fl_max | pz_drop | fwd | back | side | pitch | dyaw | yaw_osc | strides | sq |
| - | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 1 | `loco__walk_ff_start_270_R_001__A234` | 176 | 5.87 | 0.093 | 0.022 | +0.004 | -0.004 | 0.545 | 15.9 | +10.3 | 39 | 5 | 0.47 |
| 2 | `loco__walk_sideway_090_stop_001__A042_M` | 138 | 4.60 | 0.086 | 0.020 | +0.034 | -0.034 | 0.424 | 6.8 | +6.7 | 33 | 3 | 0.53 |
| 3 | `loco__walk_sideway_090_start_001__A039` | 261 | 8.70 | 0.087 | 0.030 | -0.005 | +0.005 | 0.376 | 3.2 | -17.6 | 35 | 4 | 0.12 |
| 4 | `loco__walk_sideway_090_stop_004__A043` | 144 | 4.80 | 0.096 | 0.020 | -0.017 | +0.017 | 0.382 | 1.5 | -0.6 | 38 | 3 | 0.78 |
| 5 | `loco__walk_ff_start_270_R_slow_001__A446` | 319 | 10.63 | 0.129 | 0.018 | -0.006 | +0.006 | 0.277 | 3.0 | +10.0 | 26 | 5 | 0.21 |
| 6 | `loco__walk_sideway_090_stop_002__A028` | 276 | 9.20 | 0.089 | 0.027 | -0.017 | +0.017 | 0.239 | 10.2 | -4.3 | 28 | 4 | 0.53 |
| 7 | `loco__walk_ff_start_270_R_very_slow_001__A444_M` | 260 | 8.67 | 0.086 | 0.015 | -0.020 | +0.020 | 0.268 | 9.7 | -8.8 | 33 | 4 | 0.05 |
| 8 | `loco__Sideway_Walk_Right_001__A017` | 144 | 4.80 | 0.069 | 0.016 | +0.035 | -0.035 | 0.186 | 14.9 | -13.1 | 33 | 3 | 0.10 |
| 9 | `loco__walk_ff_start_270_R_slow_001__A443` | 325 | 10.83 | 0.098 | 0.019 | -0.011 | +0.011 | 0.271 | 3.6 | +15.0 | 40 | 5 | 0.01 |
| 10 | `loco__walk_sideway_090_stop_001__A039_M` | 271 | 9.03 | 0.077 | 0.029 | -0.015 | +0.015 | 0.206 | 8.0 | -1.5 | 39 | 5 | 0.91 |

## Lean forward (sustained ≥ 12 deg, planted, foot-lift ≤ 2.5 cm)  (top 25 of 101)

| # | motion_key | T | dur | fl_max | pz_drop | fwd | back | side | pitch | dyaw | yaw_osc | strides | sq |
| - | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 1 | `loco__big_heavy_one_hand_front_medium_to_front_low_R_001__A527_M` | 230 | 7.67 | 0.023 | 0.008 | -0.001 | +0.001 | 0.002 | 50.3 | +1.5 | 10 | 0 | 0.40 |
| 2 | `loco__body_check_001__A474_M` | 1421 | 47.37 | 0.020 | 0.022 | +0.000 | -0.000 | 0.000 | 41.7 | -1.1 | 6 | 0 | 0.81 |
| 3 | `standing__body_check_001__A474_M` | 1421 | 47.37 | 0.020 | 0.022 | +0.000 | -0.000 | 0.000 | 41.7 | -1.1 | 6 | 0 | 0.81 |
| 4 | `loco__medium_big_light_one_hand_front_low_to_behind_low_R_001__A524` | 151 | 5.03 | 0.024 | 0.033 | +0.005 | -0.005 | 0.001 | 38.5 | -2.8 | 18 | 0 | 0.61 |
| 5 | `standing__medium_big_light_one_hand_front_low_to_behind_low_R_001__A524` | 151 | 5.03 | 0.024 | 0.033 | +0.005 | -0.005 | 0.001 | 38.5 | -2.8 | 18 | 0 | 0.61 |
| 6 | `loco__body_check_001__A120_M` | 1615 | 53.83 | 0.025 | 0.028 | +0.000 | -0.000 | 0.000 | 38.4 | +3.7 | 8 | 0 | 0.91 |
| 7 | `standing__body_check_001__A120_M` | 1615 | 53.83 | 0.025 | 0.028 | +0.000 | -0.000 | 0.000 | 38.4 | +3.7 | 8 | 0 | 0.91 |
| 8 | `loco__body_check_002__A410_M` | 770 | 25.67 | 0.023 | 0.019 | -0.000 | +0.000 | 0.000 | 38.4 | -1.3 | 6 | 0 | 0.48 |
| 9 | `standing__body_check_002__A410_M` | 770 | 25.67 | 0.023 | 0.019 | -0.000 | +0.000 | 0.000 | 38.4 | -1.3 | 6 | 0 | 0.48 |
| 10 | `loco__body_check_001__A166` | 830 | 27.67 | 0.020 | 0.017 | -0.000 | +0.000 | 0.001 | 37.6 | +0.5 | 9 | 0 | 0.66 |
| 11 | `standing__body_check_001__A166` | 830 | 27.67 | 0.020 | 0.017 | -0.000 | +0.000 | 0.001 | 37.6 | +0.5 | 9 | 0 | 0.66 |
| 12 | `loco__body_check_001__A121_M` | 1603 | 53.43 | 0.015 | 0.012 | -0.000 | +0.000 | 0.000 | 34.8 | +0.3 | 2 | 0 | 0.73 |
| 13 | `standing__body_check_001__A121_M` | 1603 | 53.43 | 0.015 | 0.012 | -0.000 | +0.000 | 0.000 | 34.8 | +0.3 | 2 | 0 | 0.73 |
| 14 | `loco__body_check_001__A451_M` | 1210 | 40.33 | 0.020 | 0.015 | +0.000 | -0.000 | 0.000 | 34.5 | +0.4 | 3 | 0 | 0.77 |
| 15 | `standing__body_check_001__A451_M` | 1210 | 40.33 | 0.020 | 0.015 | +0.000 | -0.000 | 0.000 | 34.5 | +0.4 | 3 | 0 | 0.77 |
| 16 | `loco__body_check_001__A131_M` | 1080 | 36.00 | 0.022 | 0.020 | +0.000 | -0.000 | 0.000 | 33.4 | +1.6 | 4 | 0 | 0.52 |
| 17 | `standing__body_check_001__A131_M` | 1080 | 36.00 | 0.022 | 0.020 | +0.000 | -0.000 | 0.000 | 33.4 | +1.6 | 4 | 0 | 0.52 |
| 18 | `loco__body_check_01__A287` | 1087 | 36.23 | 0.019 | 0.012 | -0.001 | +0.001 | 0.000 | 33.0 | -2.0 | 5 | 0 | 0.97 |
| 19 | `standing__body_check_01__A287` | 1087 | 36.23 | 0.019 | 0.012 | -0.001 | +0.001 | 0.000 | 33.0 | -2.0 | 5 | 0 | 0.97 |
| 20 | `loco__body_check_002__A497` | 827 | 27.57 | 0.024 | 0.015 | +0.000 | -0.000 | 0.000 | 31.9 | -3.5 | 9 | 0 | 0.81 |
| 21 | `standing__body_check_002__A497` | 827 | 27.57 | 0.024 | 0.015 | +0.000 | -0.000 | 0.000 | 31.9 | -3.5 | 9 | 0 | 0.81 |
| 22 | `loco__big_heavy_two_hands_right_side_medium_to_right_side_medium_R_001__A527` | 189 | 6.30 | 0.012 | 0.008 | -0.002 | +0.002 | 0.003 | 31.1 | +1.9 | 10 | 0 | 0.81 |
| 23 | `loco__body_check_001__A338_M` | 717 | 23.90 | 0.024 | 0.020 | +0.000 | -0.000 | 0.000 | 30.3 | -0.9 | 5 | 0 | 0.78 |
| 24 | `standing__body_check_001__A338_M` | 717 | 23.90 | 0.024 | 0.020 | +0.000 | -0.000 | 0.000 | 30.3 | -0.9 | 5 | 0 | 0.78 |
| 25 | `loco__body_check_001__A131` | 1080 | 36.00 | 0.022 | 0.020 | +0.000 | -0.000 | 0.000 | 29.7 | -1.6 | 4 | 0 | 0.50 |

## Crouch (pelvis Z drops ≥ 0.05 m, planted, foot-lift ≤ 3 cm)  (top 25 of 82)

| # | motion_key | T | dur | fl_max | pz_drop | fwd | back | side | pitch | dyaw | yaw_osc | strides | sq |
| - | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| 1 | `loco__small_light_two_hands_front_low_to_front_medium_R_001__A519_M` | 111 | 3.70 | 0.054 | 0.156 | -0.001 | +0.001 | 0.000 | 39.2 | -0.4 | 4 | 0 | 0.72 |
| 2 | `standing__small_light_two_hands_front_low_to_front_medium_R_001__A519_M` | 111 | 3.70 | 0.054 | 0.156 | -0.001 | +0.001 | 0.000 | 39.2 | -0.4 | 4 | 0 | 0.72 |
| 3 | `loco__medium_big_light_two_hands_front_low_to_front_medium_R_002__A529_M` | 187 | 6.23 | 0.057 | 0.092 | +0.003 | -0.003 | 0.000 | 11.5 | +0.0 | 2 | 0 | 0.56 |
| 4 | `standing__medium_big_light_two_hands_front_low_to_front_medium_R_002__A529_M` | 187 | 6.23 | 0.057 | 0.092 | +0.003 | -0.003 | 0.000 | 11.5 | +0.0 | 2 | 0 | 0.56 |
| 5 | `loco__medium_light_two_hands_pick_up_front_low_R_001__A506_M` | 224 | 7.47 | 0.053 | 0.075 | -0.001 | +0.001 | 0.000 | 5.3 | -0.3 | 2 | 0 | 0.85 |
| 6 | `standing__medium_light_two_hands_pick_up_front_low_R_001__A506_M` | 224 | 7.47 | 0.053 | 0.075 | -0.001 | +0.001 | 0.000 | 5.3 | -0.3 | 2 | 0 | 0.85 |
| 7 | `loco__medium_light_two_hands_front_high_to_front_low_R_001__A526` | 169 | 5.63 | 0.060 | 0.181 | +0.002 | -0.002 | 0.001 | 13.4 | +0.8 | 4 | 0 | 0.79 |
| 8 | `standing__medium_light_two_hands_front_high_to_front_low_R_001__A526` | 169 | 5.63 | 0.060 | 0.181 | +0.002 | -0.002 | 0.001 | 13.4 | +0.8 | 4 | 0 | 0.79 |
| 9 | `loco__big_heavy_two_hands_front_low_to_front_low_R_001__A526` | 204 | 6.80 | 0.053 | 0.079 | -0.001 | +0.001 | 0.001 | 18.4 | -0.4 | 3 | 0 | 0.94 |
| 10 | `loco__medium_big_light_two_hands_front_medium_to_front_low_R_001__A530_M` | 136 | 4.53 | 0.057 | 0.128 | +0.002 | -0.002 | 0.000 | 33.9 | +0.1 | 4 | 0 | 0.61 |
| 11 | `standing__medium_big_light_two_hands_front_medium_to_front_low_R_001__A530_M` | 136 | 4.53 | 0.057 | 0.128 | +0.002 | -0.002 | 0.000 | 33.9 | +0.1 | 4 | 0 | 0.61 |
| 12 | `loco__big_heavy_two_hands_pick_up_front_low_R_001__A508` | 156 | 5.20 | 0.069 | 0.194 | -0.005 | +0.005 | 0.000 | 8.8 | +2.9 | 5 | 1 | 0.84 |
| 13 | `loco__body_check_001__A428_M` | 959 | 31.97 | 0.055 | 0.057 | +0.000 | -0.000 | 0.000 | 20.6 | -0.3 | 3 | 0 | 0.55 |
| 14 | `standing__body_check_001__A428_M` | 959 | 31.97 | 0.055 | 0.057 | +0.000 | -0.000 | 0.000 | 20.6 | -0.3 | 3 | 0 | 0.55 |
| 15 | `loco__big_heavy_two_hands_put_down_front_low_R_001__A508` | 187 | 6.23 | 0.056 | 0.121 | +0.002 | -0.002 | 0.001 | 9.1 | -1.9 | 5 | 0 | 0.75 |
| 16 | `loco__medium_big_heavy_two_hands_front_low_to_front_low_R_001__A521_M` | 182 | 6.07 | 0.057 | 0.137 | -0.005 | +0.005 | 0.001 | 15.3 | +0.7 | 5 | 0 | 0.72 |
| 17 | `loco__neutral_item_pick_up_floor_R_002__A542_M` | 94 | 3.13 | 0.053 | 0.042 | -0.005 | +0.005 | 0.004 | 11.0 | +1.4 | 4 | 0 | 0.59 |
| 18 | `loco__neutral_item_pick_up_floor_R_002__A542` | 94 | 3.13 | 0.053 | 0.042 | -0.005 | +0.005 | 0.004 | 9.9 | -1.3 | 4 | 0 | 0.58 |
| 19 | `loco__medium_light_two_hands_front_medium_to_front_low_R_001__A527_M` | 148 | 4.93 | 0.059 | 0.130 | -0.004 | +0.004 | 0.001 | 37.7 | +2.0 | 6 | 0 | 0.95 |
| 20 | `standing__medium_light_two_hands_front_medium_to_front_low_R_001__A527_M` | 148 | 4.93 | 0.059 | 0.130 | -0.004 | +0.004 | 0.001 | 37.7 | +2.0 | 6 | 0 | 0.95 |
| 21 | `loco__body_check_001__A463_M` | 1081 | 36.03 | 0.052 | 0.046 | +0.001 | -0.001 | 0.000 | 31.6 | +0.8 | 5 | 0 | 0.94 |
| 22 | `standing__body_check_001__A463_M` | 1081 | 36.03 | 0.052 | 0.046 | +0.001 | -0.001 | 0.000 | 31.6 | +0.8 | 5 | 0 | 0.94 |
| 23 | `loco__pick_up_little_value_bill_R_003__A459` | 288 | 9.60 | 0.064 | 0.067 | +0.003 | -0.003 | 0.000 | 10.2 | +2.3 | 5 | 0 | 0.64 |
| 24 | `standing__pick_up_little_value_bill_R_003__A459` | 288 | 9.60 | 0.064 | 0.067 | +0.003 | -0.003 | 0.000 | 10.2 | +2.3 | 5 | 0 | 0.64 |
| 25 | `loco__neutral_item_put_down_floor_R_002__A542` | 104 | 3.47 | 0.053 | 0.040 | -0.008 | +0.008 | 0.003 | 8.5 | -0.7 | 6 | 0 | 0.76 |

## Existing bin sources (for comparison)

| bin_source | fl_max | fwd | side | pitch | yaw_osc | strides |
| --- | -: | -: | -: | -: | -: | -: |
| `loco__Step_Rotate_Reaction_Idle_0135_001__A019` | 0.045 | +0.002 | 0.001 | 7.6 | 16 | 20 |
| `loco__Step_Rotate_Reaction_Idle_0360_002__A019_M` | 0.104 | -0.001 | 0.001 | 5.9 | 26 | 27 |
| `loco__body_check_001__A474_M` | 0.020 | +0.000 | 0.000 | 41.7 | 6 | 0 |
| `loco__idle_turn_270_R_003__A235_M` | 0.042 | +0.004 | 0.003 | 7.7 | 12 | 0 |
| `loco__idle_turn_360_R_003__A265_M` | 0.071 | +0.019 | 0.038 | 3.1 | 14 | 1 |
| `loco__idle_vigilance_start_R_001__A502` | 0.018 | +0.000 | 0.001 | 10.6 | 24 | 0 |
| `loco__medium_big_light_one_hand_front_low_to_behind_low_R_001__A527` | 0.037 | -0.003 | 0.002 | 46.7 | 26 | 0 |
| `loco__walk_backward_loop_002__A035_M` | 0.133 | -0.635 | 0.019 | 6.7 | 52 | 7 |
| `loco__walk_forward_loop_004__A042_M` | 0.160 | +0.670 | 0.000 | 4.4 | 75 | 6 |
| `loco__walk_sideway_right_loop_001__A032` | 0.117 | +0.011 | 0.698 | 5.9 | 64 | 9 |
| `synth:waist_yaw_ramp(peak=0.2618rad)` | - | - | - | - | - | - |
| `synth:waist_yaw_ramp(peak=0.5236rad)` | - | - | - | - | - | - |
| `synth:waist_yaw_ramp(peak=0.7854rad)` | - | - | - | - | - | - |

