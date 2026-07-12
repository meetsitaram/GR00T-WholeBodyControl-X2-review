# K-Planner Validation

Tools to validate the **motionbricks kinematic planner** (k-planner) output — the
generated reference motion, *without* the SONIC tracking controller in the loop.
Drive a fixed routine (speeds + turns, mode-switched) through a checkpoint, dump
the raw qpos, and render X2 vs G1 side-by-side so you can judge a training run.

> Scope: this validates the **planner's kinematic reference only**. The SONIC
> policy/controller (`policy/release/model_*.onnx`) is never run here.

All commands assume:

```bash
cd <repo-root>
source .venv/bin/activate
export PYV="PYTHONPATH=${PWD}/motionbricks:${PWD}"
export KV=motionbricks/scripts/kplanner_validation
```

## Files

| Script | Role |
|---|---|
| `run_scripted_demo.py` | Drive a checkpoint through a scripted schedule (`--schedule validation` = fixed 8-phase battery); dump qpos NPZ + segment log |
| `render_x2_vs_g1_video.py` | Offscreen side-by-side MP4 (X2 left / G1 right), title + per-phase labels, camera-tracked. No display needed |
| `view_e2e_x2_vs_g1.py` | Interactive MuJoCo side-by-side viewer; `--auto-cycle` walks through trials hands-free |
| `test_e2e_velocity_tracking.py` | Constant-velocity sweep driver + tracking-ratio metrics (forward/lateral/yaw); emits NPZ for the renderer/viewer |

Shared, still in `motionbricks/scripts/`: `build_x2_planner_clips.py` (bakes the
pose-template clip library), `compare_e2e_reports.py` (parity table for two sweep
reports).

## The validation routine (`--schedule validation`)

Fixed 8-phase, mode-driven battery (pose-template path). Turns are done in
slow_walk mode before ramping up:

| # | phase | mode | vz | yaw |
|---|---|---|---|---|
| 1 | idle | idle | 0 | 0 |
| 2 | slow_walk_0.2 | slow_walk | 0.20 | 0 |
| 3 | slow_walk_0.3 | slow_walk | 0.30 | 0 |
| 4 | turn_left | slow_walk | 0.30 | +0.4 |
| 5 | turn_right | slow_walk | 0.30 | −0.4 |
| 6 | slow_walk_0.5 | slow_walk | 0.50 | 0 |
| 7 | walk_1.0 | walk | 1.00 | 0 |
| 8 | run_1.5 | run_proxy | 1.50 | 0 |

G1 has no run clip, so the driver remaps run→walk for the G1 reference. Both
robots seed from `neutral_idle_loop_001__A076` for a matched start.

## Generate + render (X2 vs G1)

```bash
D=out/e2e_headtohead

# X2 (current planner)
env $PYV python $KV/run_scripted_demo.py --ckpt-set x2 --schedule validation \
  --x2-seed-clip-key neutral_idle_loop_001__A076 \
  --save-npz $D/validation_x2.npz --save-schedule-json $D/validation_schedule.json

# G1 reference
env $PYV python $KV/run_scripted_demo.py --ckpt-set g1 --schedule validation \
  --seed-clip-idx 2 --save-npz $D/validation_g1.npz

# Side-by-side MP4
env $PYV MUJOCO_GL=egl python $KV/render_x2_vs_g1_video.py \
  --x2-npz $D/validation_x2.npz --g1-npz $D/validation_g1.npz \
  --out $D/validation_x2_vs_g1.mp4
```

## Regression test for a new training run

Point the tool at the retrained checkpoint with `--{vqvae,pose,root}-ckpt`
(each override auto-derives its version dir for hparams as `<ckpt>/../..`), then
render **new vs old** (or vs G1):

```bash
NEW=motionbricks/out/<retrained_run>/version_1/checkpoints
env $PYV python $KV/run_scripted_demo.py --ckpt-set x2 --schedule validation \
  --x2-seed-clip-key neutral_idle_loop_001__A076 \
  --vqvae-ckpt $NEW/model-step=NNNNNN.ckpt \
  --pose-ckpt  $NEW/model-step=NNNNNN.ckpt \
  --root-ckpt  $NEW/model-step=NNNNNN.ckpt \
  --save-npz $D/validation_x2_new.npz

env $PYV MUJOCO_GL=egl python $KV/render_x2_vs_g1_video.py \
  --x2-npz $D/validation_x2_new.npz --g1-npz $D/validation_x2.npz \
  --right-label "X2 old" --out $D/validation_progress.mp4
```

Omit the `*-ckpt` flags to validate the pinned default checkpoint.

## Constant-velocity sweep (optional)

```bash
env $PYV python $KV/test_e2e_velocity_tracking.py --ckpt-set x2 --sweep forward \
  --horizon-s 4.0 --x2-seed-clip-key neutral_idle_loop_001__A076 \
  --save-npz $D/e2e_x2_forward.npz --report-json $D/e2e_x2_forward.json
# repeat --ckpt-set g1 --seed-clip-idx 2 ; then render / view the two NPZs.
```

## What "good" looks like

The routine renders the **planner reference**, so gait differentiation depends on
the planner backbone. On the current locowalk-only X2 planner the modes look
flat (glides at low speed, no run). After a broad-corpus (BONES-SEED) retrain the
*same* routine should start showing slow≠walk≠run — that differentiation is the
pass signal. The deployed G1 SONIC (mode × velocity gaits) is the target behavior.
