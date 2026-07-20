# Sim full-stack session (T2) — smooth walks, instrumented

Operator ran `sim_onnx_planner.sh` (same `pc2_kplanner_onnx.py` runtime the
robot uses, deployed planner graphs, deployed sonic `softland_4800`, MuJoCo
plant, gamepad) with `KPLANNER_TAPE_DIR` set. 6 forward walks + one 9.3 s
backward walk, operator-attested smooth. Diff against the robot session
`20260719_223057_walks_residual_stumbles`:

| metric | sim stack (this run) | robot session |
|---|---|---|
| inference latency | median 66 ms, max 95 | 270–620 ms |
| replan threshold | 2 frames (launcher default!) | 32 |
| commit fast-forward | median 1.8 frames | 10.8–20.4 |
| onset anchor gap | ~0.07 s | ~0.3–0.6 s |
| starvation | 34 single-tick blips at seams (invisible) | 0 |
| chunk quality (walking hip-std) | med 0.194, 1 still | med ~0.19, 4 stills (turn) |
| wire foot-skate | 0.16–0.18 m/s (~1.5× good clip) | 0.20–0.31 (clean-replay basis) |
| swing events | healthy both directions incl. BACK 4/5 | healthy |
| stumbles | none (operator) | 12 spike episodes |

Notable: the sim ran at the HARSHEST seam margin (threshold 2 — the launcher
default was never bumped to 32) and still walked smooth, because 66 ms
inference makes seams/onset gaps negligible. The same runtime at PC2 latency
needs threshold 32 and still pays a 0.3–0.6 s onset gap.

## Combined verdict (with the tuned-harness offline result)

Same runtime + same models + near-same reference quality:
- MuJoCo plant, fast inference → smooth (this run)
- MuJoCo plant, robot's exact recorded wire, deploy-tuned harness → 0 falls,
  0 spikes (offline)
- real robot, PC2 latency → 12 brief stumble episodes, 0 falls

Remaining causal axes for the robot's residual stumbles, in order:
1. **Physical layer** (contact/surface, state estimation + IMU noise,
   actuator friction/thermals) — the only axis present exclusively where
   stumbles happen.
2. **PC2 inference latency** — doesn't corrupt the wire (fast-forward keeps
   it continuous) but scales the walk-onset anchor gap (0.48 s vs 0.07 s);
   robot stumbles cluster at onsets/transitions. Orin NX GPU inference (ORT
   CUDA/TensorRT EP) attacks this directly.

Sim-stack launcher note: `run_x2_quest3_planner_stack.sh` default
`KPLANNER_REPLAN_THRESHOLD_FRAMES=2` is fine for workstation latency but
must never be copied to a PC2 launch.
