# Standard Commands

Top-level cheat-sheet for the X2 + Quest 3 teleop / record / replay
loop. Defaults assume a `uv venv` at `.venv/` and the dataset
landing under `data/lerobot/`.

## SONIC-stabilised teleop + record (recommended for v1+ datasets)

The full deploy-in-the-loop path. Co-launches the C++ deploy (running
the SONIC 25k tracking policy) with the Python recorder. The trained
policy enforces lower-body stability + joint-limit / collision
feasibility on top of operator IK output. **This is what you want
for VLA training data.**

### Sanity-check the loop without writing data

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --teleop-only \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

### Full record session

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_quest3_sonic_v1 \
    --task "wave hello with both hands" \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

The wrapper forwards every flag below to `record_x2_dataset.py`:

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--sonic-correction-warn-rad FLOAT` | `0.05` | Threshold for the once-per-second log when SONIC pushes back on operator commands. |
| `--no-sonic-correction-log` | off | Suppress that log. The `action.sonic_correction_max_rad` parquet column is still populated. |
| `--no-finger-filter` / `--finger-filter-*` | filter ON | Same v0.6 finger-smoothing knobs as kinematic teleop. |
| `--calibration PATH` / `--recalibrate` / `--operator-id NAME` | — | Operator calibration plumbing. |

The recorded dataset uses the **v1 schema**: `action.body_q_mj` is the
**post-SONIC** executed q (what the trained policy actually achieved
and what the MuJoCo viewer shows); the operator's pre-SONIC X2 joint
command is preserved as `action.body_q_mj_pre_sonic` for retargeter /
SONIC-correction analysis (debug-only, not pulled into training
batches). See [docs/source/tutorials/x2_dataset_record_and_replay.md](docs/source/tutorials/x2_dataset_record_and_replay.md)
for the schema spec.

## Calibration (one-time per operator)

```sh
.venv/bin/python -m gear_sonic.scripts.vr_operator_calibrate \
    --operator-id <name>
```

Captures the 4-pose calibration (arms-down / T-pose / arms-forward /
namaste) and writes `data/operator_calibrations/<name>.yaml`. Re-run
when switching operators or after sessions where the wrist mapping
felt off.

## Kinematic teleop + record (no SONIC, debug only)

The fastest debug loop for retargeting / filter A/B with **no
feasibility enforcement**. Pure VR → IK → MuJoCo with the lower body
pinned to the SONIC stand pose; no deploy, no policy, no ZMQ. Use
this for retargeter debug; **do not use it to capture training data
for VLA fine-tuning** (operator commands can drive the robot through
joint limits / self-collisions and the trainer will then learn those
infeasible targets).

```sh
.venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic \
    --output-dir data/lerobot/x2_quest3_kinematic_v6 \
    --task "wave hello with both hands" \
    --rate 50 \
    --hand-input max
```

Operator buttons (Quest 3): **A** = engage IK, **B** = start episode,
**X** = save episode, **Y** = discard episode.

### Useful flags

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--task STR` | required with `--output-dir` | Language label stamped on every frame. |
| `--hand-input {trigger,grip,max}` | `trigger` | Which controller analog drives finger curl. |
| `--calibration PATH` | `data/operator_calibrations/default.yaml` | Operator YAML. |
| `--recalibrate` | off | Run the 4-pose calibration inline before teleop starts. |
| `--ik-rotation-weight 0.3` | 0.3 | Position+orientation IK. Set 0.0 for position-only. |

### Finger smoothing filter (v0.6, **on by default**)

The Quest 3 hand-curl / thumb-oppose / finger-tip-oppose streams go
through a per-side EMA + rolling-median deadband-hold before they
reach the retargeter. Calibrated against the v5/ep1 recording;
delivers ~50 % max-single-frame-jump reduction and bridges 1–3 frame
XRHand re-acquire blips invisibly. Live + recorded both run through it.

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--no-finger-filter` | off (filter ON) | Disable the smoother. The debug NPZ still persists raw + filtered channels for offline A/B. |
| `--finger-filter-alpha FLOAT` | `0.5` | EMA alpha. Lower = smoother + more lag; `1.0` disables EMA. |
| `--finger-filter-hold-window INT` | `8` (160 ms) | Rolling-window length for the deadband-hold. |
| `--finger-filter-hold-std FLOAT` | `0.005` | Std threshold to enter the held-pose latch. Raise to ~0.008 if rest tremor still leaks through. |

A/B in two consecutive sessions:

```sh
# session 1: filter ON (default)
.venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic \
    --output-dir data/lerobot/x2_quest3_kinematic_v6 \
    --task "filter_on_v6"

# session 2: filter OFF for direct comparison
.venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic \
    --output-dir data/lerobot/x2_quest3_kinematic_v6 \
    --task "filter_off_v6" \
    --no-finger-filter
```

## Replay

### Kinematic MuJoCo replay (fastest)

Plays the recorded `action.body_q_mj` (v1 schema, post-SONIC executed
q in SONIC datasets / commanded q in kinematic datasets) straight
from the parquet — no Quest 3, no IK, no policy. Auto-falls-back to
the legacy `action.commanded_body_q_mj` for v0 datasets.

```sh
.venv/bin/python -m gear_sonic.scripts.replay_x2_kinematic \
    --dataset x2_quest3_kinematic_v6 \
    --episode 0
```

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--dataset NAME-OR-PATH` | — | Dataset root under `data/lerobot/` or any path. |
| `--episode INT` | — | Zero-indexed. Use `--parquet PATH` to bypass resolution. |
| `--rate FLOAT` | `50.0` | Playback rate. Match the recorder for real-time. |
| `--start-frame INT` / `--end-frame INT` / `--loop` | — | Window + loop the playback. |
| `--no-omnihand` | off | Drop the OmniHand mesh (debug only). |

### Offline retargeting replay (re-derive parquet from the debug NPZ)

Useful for tuning new retargeter / calibration / filter params against
a recording without restrapping the headset.

```sh
.venv/bin/python -m gear_sonic.scripts.replay_recorded_dataset \
    --npz     data/lerobot/x2_quest3_kinematic_v6/debug/teleop_episode_000000.npz \
    --parquet data/lerobot/x2_quest3_kinematic_v6/data/chunk-000/episode_000000.parquet \
    --output-dir /tmp/replay_v6_default
```

Filter A/B from a single recording:

```sh
# Use the FILTERED channels from the NPZ (default 'auto').
.venv/bin/python -m gear_sonic.scripts.replay_recorded_dataset \
    --npz <ep>.npz --parquet <ep>.parquet \
    --output-dir /tmp/v6_filter_on  --apply-finger-filter auto

# Replay RAW signals as recorded (no filter).
.venv/bin/python -m gear_sonic.scripts.replay_recorded_dataset \
    --npz <ep>.npz --parquet <ep>.parquet \
    --output-dir /tmp/v6_filter_off --apply-finger-filter never

# Force-apply the offline filter even when *_filtered keys are present.
.venv/bin/python -m gear_sonic.scripts.replay_recorded_dataset \
    --npz <ep>.npz --parquet <ep>.parquet \
    --output-dir /tmp/v6_filter_offline --apply-finger-filter always
```

Then point the kinematic replay at each output to view side-by-side:

```sh
.venv/bin/python -m gear_sonic.scripts.replay_x2_kinematic \
    --parquet /tmp/v6_filter_on/episode_000000_replay_baseline.parquet
```

### SONIC corrective-delta diagnostic

Runs offline against any v1 SONIC dataset to surface frames where
the trained policy had to override operator commands.

```sh
.venv/bin/python -m gear_sonic.scripts.inspect_sonic_correction \
    --dataset x2_quest3_sonic_v1 --episode 0
```

Outputs a per-arm-joint delta summary table and writes a 4-panel
time-series PNG to `<dataset>/debug/sonic_correction_ep<N>.png`.
Falls back to commanded-trajectory stats on legacy v0 datasets.

## Build bones-seed motion library

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/data_process/build_x2_bones_seed_motion_lib.py
```

## Standing-gestures showcase in sim

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
conda run -n env_isaaclab --no-capture-output python gear_sonic/scripts/eval_x2_mujoco.py \
    --checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.pt \
    --playlist gear_sonic/data/motions/playlists/showcase_v1.yaml
```
