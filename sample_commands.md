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

### Full record session (recommended VLA recipe)

`--sim-omnihand` loads the OmniHand-augmented MJCF so the fingers are
in scope for the dataset; `--wrist-bypass ik` overrides SONIC's wrist
attractor with the operator's IK reference (the policy pins those
four DOFs without it). Both flags are made explicit here for
discoverability — `--wrist-bypass ik` is the wrapper default.

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_quest3_sonic_v3 \
    --task "wave hello with both hands" \
    --sim-omnihand \
    --wrist-bypass ik \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

Sanity-check the bypass is firing once SONIC reaches `CONTROL` state:
the deploy log's periodic status line should end with
`wrist_bypass_ticks=<N> wrist_bypass_max_dev_rad=<X.XXX>`, with `N`
incrementing every second. After the session, run
`/tmp/wrist_sign_probe.py` against the resulting parquet — both
`*_wrist_pitch` and `*_wrist_roll` should show `corr > 0.9` and no
limit-pinning. See
[tutorial §3.5](docs/source/tutorials/x2_dataset_record_and_replay.md#35-wrist-bypass-honest-vr-wrist-tracking-on-top-of-sonic)
for the operator workflow and
[§8 SONIC pins the wrist DOFs](docs/source/tutorials/x2_dataset_record_and_replay.md#sonic-pins-the-wrist-dofs--and-why-we-bypass-them-in-c)
for the root-cause post-mortem.

### Sim-to-real fidelity probe (wrist bypass OFF)

Reproduces the v2 baseline (pinned `wrist_roll`, flat `wrist_pitch`)
so you can numerically diff against a wrist-bypass-on recording. Use
this when validating SONIC behaviour, not for VLA training data.

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_quest3_sonic_baseline \
    --task "wrist bypass OFF baseline" \
    --sim-omnihand \
    --wrist-bypass off \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

The wrapper forwards every flag below to `record_x2_dataset.py`:

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--wrist-bypass {off,ik}` | `ik` | Override SONIC's wrist target with the operator's IK reference for the 4 broken DOFs (`*_wrist_pitch`, `*_wrist_roll`). Keep `ik` for VLA recordings; flip to `off` only for sim-to-real fidelity probes. See [tutorial Section 3.5](docs/source/tutorials/x2_dataset_record_and_replay.md#35-wrist-bypass-honest-vr-wrist-tracking-on-top-of-sonic). |
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

## Robocasa scene mode (G1 architecture, hardware-required smoke)

Adds a tabletop + cube + bowl scene to the deploy's MuJoCo and writes
per-tick `task.success` / `task.reward` / `task.subtask_<name>` columns
into the LeRobot dataset. The recorder loads the same static scene XML
the deploy bridge sees, so the ego-view renders the table + objects
and the `RobocasaTaskMirror` can grade success purely by mirroring the
bridge's scene state over ZMQ. See
[`decoupled_wbc/dexmg/gr00trobocasa/X2_INTEGRATION_NOTES.md`](decoupled_wbc/dexmg/gr00trobocasa/X2_INTEGRATION_NOTES.md)
for the full architecture write-up.

### Build the scene XMLs (one-time, after a fresh checkout)

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceCube && \
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceBowl
```

Output lands in `gear_sonic/data/assets/robocasa_scenes/<env>.xml`
plus a `<env>.json` metadata sidecar. Both processes auto-discover the
sidecar at startup.

### Hardware-required smoke #1 — open the scene in deploy without recording

Confirms the bridge boots with `--sim-mjcf` pointed at the scene and the
operator can teleop arms freely without writing data. No Quest 3
required if you skip teleop calibration; the bridge will still load and
render the scene.

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --teleop-only \
    --robocasa-env X2PickPlaceCube \
    --wrist-bypass ik \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

What to look for in the deploy log:

* `[bridge] loaded scene metadata: …/X2PickPlaceCube.json`
* `[bridge] scene_state PUB bound at tcp://*:5559`
* `[bridge] scene_reset SUB connected to tcp://localhost:5560`
* MuJoCo viewer shows the X2 standing in front of a small lab table
  with a red cube + blue bowl on top.

### Hardware-required smoke #2 — record one robocasa episode end-to-end

Drops the language-instruction requirement (the recorder picks it up
from the scene metadata: "pick up the red cube and drop it into the
blue bowl") and writes a real LeRobot v2.1 dataset with the
`task.success` / `task.reward` / `task.subtask_grasp_cube` columns.

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_robocasa_pnp_smoke_v0 \
    --robocasa-env X2PickPlaceCube \
    --wrist-bypass ik \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

Operator buttons (Quest 3): **A** engages IK, **B** starts a fresh
episode (this is when the mirror calls `reset()` and randomizes the
cube), **X** saves, **Y** discards.

#### Tight finger-closure variant (recommended for cube grasps)

The default live finger pipeline is **affine-normalised only** so
deliberate intermediate gestures (half-grasp, soft pinch) preserve
their amplitude. On a tight power-grasp pick-and-place that means a
~70-85 % squeeze (typical operator effort) maps to ~70-85 % of the
way to the OmniHand CLOSED anchor — visually the fingers don't fully
wrap a 4 cm cube. To fix this, layer the **opt-in stretch curves**
on top of the affine normalisation so mid-range curls saturate
toward CLOSED:

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_robocasa_pnp_v1 \
    --robocasa-env X2PickPlaceCube \
    --wrist-bypass ik \
    --apply-curl-compensation \
    --apply-oppose-compensation \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

The recorder banner at startup confirms the active mode:

```
Finger control:
  smoothing filter (v0.6) : ON
  curl compensation       : ON  (--apply-curl-compensation)
  oppose compensation     : ON  (--apply-oppose-compensation)
```

The same flags are also exposed on `gear_sonic.scripts.teleop_x2_kinematic`
for kinematic-only sessions. Background and tuning rationale:
[`x2_dataset_record_and_replay.md` → "Why we abandoned the global
power-curve compensation"](docs/source/tutorials/x2_dataset_record_and_replay.md).

Post-flight check on the saved parquet (covers all six phases of the
shaped-reward ladder):

```sh
.venv/bin/python - <<'PY'
import pandas as pd, glob
parquet = sorted(glob.glob("data/lerobot/x2_robocasa_pnp_smoke_v0/data/chunk-*/*.parquet"))[-1]
df = pd.read_parquet(parquet)

def _scalar(s):
    return s.apply(lambda v: float(v[0]) if hasattr(v, '__len__') else float(v))

print(f"=== {parquet} ===")
print(f"frames: {len(df)}  duration ~{len(df)/50:.1f}s")
task_cols = sorted(c for c in df.columns if c.startswith("task"))
print(f"task columns: {task_cols}")
print()

if "task.success" in df.columns:
    s = _scalar(df["task.success"])
    print(f"task.success    : {int(s.sum())}/{len(s)} ticks "
          f"(first_on_tick={int((s>0).idxmax()) if s.any() else 'never'})")
if "task.reward" in df.columns:
    r = _scalar(df["task.reward"])
    print(f"task.reward     : sum={r.sum():.2f} max={r.max():.2f} mean={r.mean():.3f}")
print()
print("phase ladder ticks (any-on across episode):")
for col in task_cols:
    if col.startswith("task.subtask_"):
        v = _scalar(df[col])
        on = int(v.sum())
        first = int((v>0).idxmax()) if v.any() else None
        print(f"  {col:<32} {on:>5} ticks  first_on={first}")
PY
```

Expected pattern on a clean cube-in-bowl demo (numbers are
illustrative, not exact):

```
task.success    : 142/2079 ticks (first_on_tick=1937)
task.reward     : sum=312.45 max=1.00 mean=0.150
phase ladder ticks (any-on across episode):
  task.subtask_approach_cube       1421 ticks  first_on=180
  task.subtask_touch_cube           742 ticks  first_on=410
  task.subtask_grasp_cube           526 ticks  first_on=510
  task.subtask_cube_off_table       497 ticks  first_on=580
  task.subtask_cube_above_bowl      214 ticks  first_on=1850
  task.subtask_cube_in_bowl         142 ticks  first_on=1937
```

The `first_on` values should be strictly increasing through the
ladder. If `task.success` stays at 0 the most likely culprits are:

1. The bridge's `scene_state` PUB never came up — check the deploy log
   for the `scene plumbing: N freejoints, M welded bodies, K object
   collision geoms, L:N R:N hand geoms` line at boot.
2. The mirror's mirror is reading the wrong body name — re-run
   `.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --all`
   and confirm the JSON sidecar contains `object_contact_geoms`,
   `hand_root_bodies`, and `fingertip_bodies`.
3. The fingers aren't actually colliding with the cube — check the
   scene XML doesn't contain `contype="0" conaffinity="0"` on the
   fingertip cylinders (it shouldn't if it was rebuilt with the
   `disable_hand_collisions=False` knob).

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

## X2 heuristic locomotion planner

The planner publishes 50 Hz pose refs over ZMQ that the SONIC deploy
(sim or real) consumes via `--vla --vla-zmq-host/port`. Same planner +
same deploy config drives both sim and real robot. Architecture and
all knobs documented in
[`docs/source/references/x2_heuristic_planner.md`](docs/source/references/x2_heuristic_planner.md).

### Closed-loop sim with keyboard control

Brings up `deploy_x2.sh sim --vla` (subscribes to ZMQ pose), the MuJoCo
viewer (camera tracking pelvis), and the planner with keyboard intake,
all under one trap-cleaned wrapper. Type keys in the same terminal to
drive the robot.

```sh
gear_sonic/scripts/run_planner_smoke.sh --with-deploy --keyboard --duration 120
```

Each keypress **replaces** any pending commands (latest press wins);
the currently-playing primitive finishes naturally. See `KEYBOARD_HELP`
printed at startup for the full key map.

### Closed-loop sim with a scripted demo

```sh
gear_sonic/scripts/run_planner_smoke.sh \
    --demo gear_sonic/data/scripted_demos/eleven_motion_sequence.yaml \
    --with-deploy --duration 60
```

Available demos in `gear_sonic/data/scripted_demos/`:

| Demo | Bins exercised |
| --- | --- |
| `eleven_motion_sequence.yaml` | All 11 working canonical bins (smoke for every family) |
| `gallery_fwd_back_shuffle.yaml` | fwd_step + back_step variants |
| `gallery_crouch.yaml` | crouch_medium |
| `six_motion_smoke.yaml` | fwd_step + side steps + turns + back_step |
| `side_steps_only_smoke.yaml` | side_left_step + side_right_step |
| `forward_back_turn.yaml` | continuous walk + turns |
| `static_reach.yaml` | leans + torso twists |
| `manipulation_approach.yaml` | locomanipulation approach + reach |

### Replay a baked planner trajectory directly (no ZMQ, no planner)

Useful as a ground-truth reference: if a motion looks fine here but
breaks through the planner path, the bug is in the planner / wire /
future window, not in the bin or the policy.

```sh
bash gear_sonic_deploy/deploy_x2.sh sim --no-confirm \
    --motion data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_demo_eleven_motion_sequence.pkl \
    --sim-profile parity \
    --sim-viewer --max-duration 60
```

### Re-bake a demo PKL after a recipe / state-machine change

```sh
.venv/bin/python -m gear_sonic.scripts.bake_planner_demo_to_pkl \
    --demo gear_sonic/data/scripted_demos/eleven_motion_sequence.yaml \
    --out  data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_demo_eleven_motion_sequence.pkl
```

### Kinematic-only viewer (fastest iteration, no policy / no docker)

```sh
.venv/bin/python -m gear_sonic.scripts.view_x2_planner_mujoco \
    --demo gear_sonic/data/scripted_demos/eleven_motion_sequence.yaml
```

### Recovery: kill orphan planner / free the publish port

```sh
gear_sonic/scripts/run_planner_smoke.sh --cleanup-only
```

### Build / rebuild planner primitives PKL from recipes

Defaults already point at the canonical paths; pass `--bins-only NAME ...`
to rebuild a single bin during iteration.

```sh
.venv/bin/python -m gear_sonic.scripts.build_x2_planner_primitives
```

### Planner-vs-PKL parity check (catches future-window regressions)

```sh
.venv/bin/python -m gear_sonic.scripts.compare_planner_vs_motion \
    --motion-pkl   data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_planner_demo_side_steps_only_smoke.pkl \
    --planner-demo gear_sonic/data/scripted_demos/side_steps_only_smoke.yaml \
    --duration 14 --no-sim-viewer
```

Compares pelvis displacement and per-joint ranges between the
PklMotionReference path (ground truth) and the ZmqPoseInputSource
path (planner). Big gaps indicate the deploy isn't decoding the v5
future window correctly.
