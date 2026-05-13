# X2 candidate sub-windows — 2026-05-11 12:07:45

Per family-pick, the cleanest sub-window meeting the intent:
* walks: feet planted at endpoints, minimum forward/back/side speed
  hit, low yaw drift, minimum foot-lift apex within window.
* lean: starts upright, ends at apex pitch >= min_pitch_deg.
* crouch: starts at baseline pelvis Z, ends with drop >= min_drop_m.

| family | motion_key | target_s | start | n | fl_max | dx | dy | dyaw | dpz | end_pitch | speed |
| --- | --- | -: | -: | -: | -: | -: | -: | -: | -: | -: | -: |
| fwd_walk_standard | `loco__walk_forward_loop_003__A034` | 6.00 | 82 | 180 | 0.146 | +4.090 | -0.622 | -0.3 | -0.040 | +5.2 | +0.682 |
| fwd_step_1ft | `loco__walk_forward_loop_003__A034` | 1.60 | 229 | 48 | 0.139 | +0.994 | -0.189 | -0.2 | +0.041 | +25.1 | +0.621 |
| back_walk_standard | `loco__walk_backward_loop_005__A028_M` | 6.00 | 299 | 180 | 0.103 | -1.893 | +0.060 | -0.1 | +0.018 | +9.0 | +0.316 |
| back_step_half_ft | `loco__walk_backward_loop_005__A028_M` | 1.60 | 292 | 48 | 0.085 | -0.522 | +0.017 | -0.3 | +0.002 | +12.3 | +0.326 |
| side_walk | `loco__walk_sideway_090_stop_004__A043` | 6.00 | 0 | 143 | 0.098 | +0.039 | +1.835 | -4.6 | -0.000 | +2.2 | +0.385 |
| side_half_ft | `loco__walk_sideway_090_stop_004__A043` | 1.60 | 60 | 48 | 0.098 | -0.259 | +0.773 | -0.0 | -0.002 | +7.3 | +0.483 |
| lean_fwd_natural_apex | `loco__body_check_001__A474_M` | 2.50 | 752 | 75 | 0.020 | -0.098 | -0.016 | +1.9 | -0.009 | +21.6 | +0.000 |
| crouch_natural_apex | `loco__big_heavy_two_hands_front_low_to_front_low_R_001__A526` | 2.00 | 5 | 60 | 0.053 | -0.079 | -0.014 | -1.7 | -0.046 | +18.5 | +0.046 |
