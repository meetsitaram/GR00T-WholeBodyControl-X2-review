# Planner mode-template clips — the ONE place for this dependency

`x2_planner_mode_clips.pkl` (≈0.5 MB, CHECKED IN) contains exactly the four
motion clips the X2 planner's pose-template path depends on. Everything the
deployed planner "knows" about template poses traces back to this file:

| mode idx | mode | clip key | baked window | provenance |
|---|---|---|---|---|
| 0 | idle | `Idle_Right_001__A019` | [0:200) | `x2_ultra_locowalk.pkl` (1.5 GB corpus, untracked) |
| 1 | slow_walk (deployed) | `slow_walk_0.3_001` | [61:230) | `x2_g1teleop_30fps.pkl` — our G1-teleop drives retargeted to X2 (2026-07-16, untracked) |
| 2 | walk | `walk_002` | [47:247) | same |
| 3 | run_proxy | `run_001` | [45:238) | same |

Full clips are stored (not just the windows) so windows stay adjustable in
`build_x2_planner_clips.py` without re-extraction. The `g1teleop_stance`
table cuts the same clips to their longest fully-grounded runs.

## How the dependency flows

```
planner_deps/x2_planner_mode_clips.pkl
  └─ motionbricks/scripts/build_x2_planner_clips.py   (--modes g1teleop | g1teleop_stance)
       └─ motionbricks/out/X2-clip*.ckpt (+ .modes.json sidecar)
            └─ motionbricks/scripts/export_x2_planner_onnx.py --clip-ckpt ...
                 └─ x2_planner_template.onnx   (clips baked as graph CONSTANTS)
                      └─ PC2 planner_stack/models/planner_onnx/   (robot)
```

At each replan the graph samples a uniform-random 4-frame window from the
active mode's baked frames (`start = seed % (n-4)`) as the target keyframe.
Consequences of that sampling are documented in
`docs/experiments/kplanner_tuning_20260721.md` (mid-air-anchor rates, the
stance variant, the g1style negative result).

Clips are baked into the ONNX because the PC2 runtime is torch-free and
cannot load pkls; changing clips therefore requires rebuild + re-export
(~2 min total). Future improvement: make the target window a graph input
sampled daemon-side.

## Editing clips / windows

1. Edit the mode table (or add one) in `build_x2_planner_clips.py`.
2. Rebuild:  `PYTHONPATH="$PWD/motionbricks:$PWD" .venv/bin/python \
   motionbricks/scripts/build_x2_planner_clips.py --modes g1teleop`
3. Re-export the graph (env_isaaclab python, `export_x2_planner_onnx.py`,
   same model ckpts, new `--clip-ckpt`).
4. A/B in the sim stack before any robot deploy
   (`kplanner_template_sweep.py` scores candidates offline).
