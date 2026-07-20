# Walk quality — 61 teleop clips, executed vs reference (im_eval traj dumps)

Terminated clips excluded from means (run_004 in all sweeps).

## slow_walk
| metric | G1-stock | base-3k | FT-1144 |
|---|---|---|---|
| root-rel MPJPE (mm) | 12.1 | 26.9 | 26.0 |
|   upper-body (mm) | 8.5 | 28.3 | 28.6 |
|   lower-body (mm) | 18.3 | 29.8 | 27.4 |
| net-displacement err (m) | 0.186 | 0.187 | 0.154 |
| mean root pos err (m) | 0.100 | 0.099 | 0.082 |
| path-length ratio (exe/ref) | 0.970 | 1.061 | 1.079 |
| heading err (deg) | 1.7 | 2.6 | 2.8 |
| final-yaw err (deg) | 1.0 | 1.1 | 1.3 |
| mean yaw err (deg) | 1.0 | 1.6 | 1.5 |

## slow_walk_turns
| metric | G1-stock | base-3k | FT-1144 |
|---|---|---|---|
| root-rel MPJPE (mm) | 13.7 | 27.9 | 27.2 |
|   upper-body (mm) | 9.7 | 29.1 | 28.8 |
|   lower-body (mm) | 20.8 | 31.1 | 30.0 |
| net-displacement err (m) | 0.193 | 0.149 | 0.122 |
| mean root pos err (m) | 0.108 | 0.094 | 0.079 |
| path-length ratio (exe/ref) | 0.957 | 1.062 | 1.075 |
| heading err (deg) | 1.8 | 3.0 | 2.3 |
| final-yaw err (deg) | 2.0 | 2.1 | 1.5 |
| mean yaw err (deg) | 1.7 | 2.0 | 1.8 |

## slow_walk_back
| metric | G1-stock | base-3k | FT-1144 |
|---|---|---|---|
| root-rel MPJPE (mm) | 13.2 | 26.5 | 25.9 |
|   upper-body (mm) | 8.6 | 26.4 | 26.3 |
|   lower-body (mm) | 20.8 | 31.1 | 29.6 |
| net-displacement err (m) | 0.183 | 0.225 | 0.188 |
| mean root pos err (m) | 0.100 | 0.132 | 0.103 |
| path-length ratio (exe/ref) | 0.969 | 1.119 | 1.148 |
| heading err (deg) | 5.4 | 9.4 | 5.8 |
| final-yaw err (deg) | 1.0 | 1.1 | 0.9 |
| mean yaw err (deg) | 1.2 | 1.9 | 1.8 |

## walk
| metric | G1-stock | base-3k | FT-1144 |
|---|---|---|---|
| root-rel MPJPE (mm) | 13.9 | 28.8 | 27.3 |
|   upper-body (mm) | 10.1 | 29.7 | 28.3 |
|   lower-body (mm) | 20.8 | 32.7 | 30.6 |
| net-displacement err (m) | 0.246 | 0.155 | 0.137 |
| mean root pos err (m) | 0.154 | 0.125 | 0.096 |
| path-length ratio (exe/ref) | 0.971 | 0.985 | 1.005 |
| heading err (deg) | 2.6 | 1.7 | 1.5 |
| final-yaw err (deg) | 1.6 | 2.2 | 2.2 |
| mean yaw err (deg) | 1.7 | 2.3 | 2.0 |

## run
| metric | G1-stock | base-3k | FT-1144 |
|---|---|---|---|
| root-rel MPJPE (mm) | 16.5 | 29.6 | 27.3 |
|   upper-body (mm) | 12.3 | 27.5 | 25.8 |
|   lower-body (mm) | 24.2 | 36.9 | 33.5 |
| net-displacement err (m) | 0.598 | 1.584 | 1.373 |
| mean root pos err (m) | 0.321 | 0.854 | 0.726 |
| path-length ratio (exe/ref) | 0.938 | 0.786 | 0.805 |
| heading err (deg) | 1.3 | 1.9 | 0.7 |
| final-yaw err (deg) | 1.3 | 0.3 | 1.1 |
| mean yaw err (deg) | 1.8 | 2.4 | 2.2 |

## ALL
| metric | G1-stock | base-3k | FT-1144 |
|---|---|---|---|
| root-rel MPJPE (mm) | 13.5 | 27.7 | 26.8 |
|   upper-body (mm) | 9.5 | 28.6 | 28.2 |
|   lower-body (mm) | 20.3 | 31.3 | 29.5 |
| net-displacement err (m) | 0.226 | 0.241 | 0.204 |
| mean root pos err (m) | 0.127 | 0.143 | 0.118 |
| path-length ratio (exe/ref) | 0.963 | 1.040 | 1.058 |
| heading err (deg) | 2.3 | 3.3 | 2.6 |
| final-yaw err (deg) | 1.5 | 1.7 | 1.5 |
| mean yaw err (deg) | 1.5 | 2.0 | 1.8 |

## Worst offenders (per sweep, feasible clips)
### G1-stock
worst root-rel MPJPE: slow_walk_turns_0.4_001 20mm, walk_003 19mm, slow_walk_back_0.5_001 18mm, run_002 18mm, slow_walk_turns_0.6_003 17mm
worst path-ratio: slow_walk_turns_0.2_005 0.51, slow_walk_back_0.2_001 0.66, slow_walk_0.2_001 0.67, slow_walk_0.2_002 0.70, slow_walk_back_0.2_002 0.70
worst heading err: slow_walk_back_0.2_001 18deg, slow_walk_turns_0.2_002 6deg, walk_circle_003 6deg, walk_circle_002 6deg, slow_walk_back_0.3_002 5deg
worst net-disp err: run_002 0.87m, walk_002 0.86m, run_003 0.63m, slow_walk_turns_0.2_005 0.59m, run_001 0.51m

### base-3k
worst root-rel MPJPE: slow_walk_turns_0.2_005 33mm, slow_walk_0.8_002 33mm, walk_turn_003 32mm, walk_circle_003 31mm, slow_walk_turns_0.8_002 31mm
worst path-ratio: slow_walk_back_0.3_001 1.68, run_001 0.75, slow_walk_back_0.2_001 0.77, run_003 0.78, slow_walk_0.2_001 1.25
worst heading err: slow_walk_back_0.3_001 19deg, slow_walk_back_0.3_002 17deg, slow_walk_turns_0.2_004 13deg, slow_walk_turns_0.2_002 12deg, slow_walk_back_0.2_001 11deg
worst net-disp err: run_001 1.90m, run_003 1.52m, run_002 1.33m, slow_walk_0.8_002 0.78m, walk_002 0.41m

### FT-1144
worst root-rel MPJPE: slow_walk_turns_0.2_005 34mm, slow_walk_0.8_002 34mm, walk_turn_003 32mm, slow_walk_turns_0.2_002 32mm, slow_walk_turns_0.8_002 31mm
worst path-ratio: slow_walk_back_0.3_001 1.74, slow_walk_0.2_001 1.36, run_001 0.79, run_002 0.81, slow_walk_turns_0.2_005 1.23
worst heading err: slow_walk_turns_0.2_002 20deg, slow_walk_turns_0.2_005 9deg, slow_walk_back_0.3_001 9deg, slow_walk_back_0.3_002 8deg, slow_walk_back_0.4_001 6deg
worst net-disp err: run_001 1.56m, run_002 1.38m, run_003 1.18m, slow_walk_back_0.2_001 0.32m, slow_walk_0.3_003 0.26m

## G1 feasibility_report cross-check (doc-standard Metric-2, FK local MPJPE + stride)

| category | n | upper local (mm) | lower local (mm) | root drift (m) | stride ratio |
|---|---|---|---|---|---|
| slow_walk | 16 | 8.7 | 20.6 | 0.186 | 0.908 |
| slow_walk_turns | 23 | 10.1 | 23.6 | 0.194 | 0.874 |
| slow_walk_back | 7 | 8.9 | 21.2 | 0.183 | 0.803 |
| walk | 11 | 11.6 | 26.1 | 0.248 | 0.918 |
| run | 4 | 15.5 | 30.6 | 0.598 | 0.890 |
| ALL | 61 | 10.2 | 23.4 | 0.227 | 0.884 |

All 61 clips labeled CLEAN (with runs measured at 4 envs). Overall stride 0.884 matches
the historical G1 reference (~0.91, MPJPE ~19 mm). Deep understep concentrated at 0.2 m/s:
slow_walk_back_0.2_001 stride 0.20, slow_walk_turns_0.2_005 0.35, slow_walk_0.2_001 0.53 —
the known slow-walk dead-band, present already in the G1 source executions.

## Method + caveats

- Executed/reference trajectories from `IM_EVAL_DUMP_TRAJ` per-clip dumps (im_eval_callback.py:159).
  NOTE: manager_env_wrapper.get_env_data() maps "ref_body_pos_extend"->robot and
  "rigid_body_pos_extend"->reference, so the dump's pred/gt field names are SWAPPED;
  verified by matching reference net-travel to the motion pkls (median diff 0.0 m).
- 14 tracked bodies; root = pelvis; yaw from the L/R shoulder cross-body vector;
  path-length on 0.2 s-decimated root XY.
- G1 CSV recorder (G1_SHIM_RECORD_DIR) requires num_envs >= num clips (global motion ids
  indexed into the chunk-local motion buffer -> CUDA assert otherwise), so the G1 sweep ran
  at 64 envs; X2 sweeps at 32 envs (2 loops).
- ENV-COUNT ARTIFACT (feasibility): the marginal run clips' pass/fail depends on batch size.
  G1-stock run_002/003/004: pass@4env(prog 1.0) / run_004-fails@32env(0.66) / all-3-fail@64env(0.016).
  Isolated with a no-capture 64-env run: capture is innocent; the batch composition itself changes it.
  Therefore ALL run-category quality metrics (all 3 sweeps) come from dedicated 4-env run-only passes.
- run_004 (fastest run): G1-stock tracks it fully at 4 envs; BOTH X2 models fall at ~5% progress
  even at 4 envs -> real capability gap, excluded from X2 run means (X2 run n=3, G1 run n=4).
- Terminated clips excluded from quality means.
