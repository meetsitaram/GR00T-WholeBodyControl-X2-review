# Standard Commands

## custom quick commands
- record
```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 1200 --with-record \
    --output-dir data/lerobot/x2_pick_place_apple_v1 \
    --robocasa-env X2PickPlaceApple \
    --sonic-tokenizer-device cpu
```

- replay videos:
```sh
xdg-open data/lerobot/x2_pick_place_apple_v1/videos/chunk-000/observation.images.ego_view/episode_000005.mp4 &
xdg-open data/lerobot/x2_pick_place_apple_v1/videos/chunk-000/observation.images.front_cam/episode_000005.mp4 &
```

- train
```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl

deactivate              # leave .venv (uv-managed)
conda deactivate         # leave whatever conda env is currently on top
conda activate env_isaaclab

# Verify python now resolves to env_isaaclab
which python
# Should print: /home/stickbot/miniconda3/envs/env_isaaclab/bin/python

PYTHONPATH=external_dependencies/Isaac-GR00T:. python \
    external_dependencies/Isaac-GR00T/gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path data/lerobot/x2_pick_place_apple_v1 \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path gear_sonic/data/x2_modality_config_10dof.py \
    --num-gpus 1 \
    --output-dir /tmp/x2_pick_place_apple_v1_run1 \
    --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08
```

- real-robot autonomous VLA (laptop-side bridge only; assumes
  `x2_pc2_daemons.sh start` is already running on PC2 — same SONIC
  deploy you use for teleop and recording).

  One command. Preflight (PC2 ping + x2_debug + cameras + model files
  + bridge python) runs inside the launcher; auto-SSHes to PC2 to
  start the camera bridge if it's silent. See
  [`docs/source/tutorials/x2_vla_runtime.md`](docs/source/tutorials/x2_vla_runtime.md)
  for the full operator runbook and
  [`docs/source/references/x2_sonic_runtime_architecture.md`](docs/source/references/x2_sonic_runtime_architecture.md)
  for how this launcher relates to the teleop + recording stack.

```sh
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model data/checkpoints/x2_grab_a_drink_n17_30k_v1/checkpoint-25000 \
    --motion-token-decoder $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "grab the can from the table"
```

> Legacy alias: `--sonic-checkpoint` / `SONIC_CHECKPOINT` still works
> (one-shot deprecation warning printed). Use
> `--motion-token-decoder` going forward.

Knobs you'll actually touch (matching `--FLAG VALUE` form — env-var
names are accepted as fallbacks, see the full runbook):

| Flag                          | Default                                                       | What it does                                                    |
| ----------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------- |
| `--model PATH`                | required                                                      | Fine-tuned GR00T-N1.7 checkpoint dir (`model.safetensors` + `experiment_cfg/` + `processor/`) |
| `--motion-token-decoder PATH` | auto-resolve next to `--model`, fallback 25k canonical        | Body-pose decoder `.pt`; empty = body holds idle, fingers move only. Legacy alias: `--sonic-checkpoint`. |
| `--prompt STR`                | `"grab a drink"`                                              | Language instruction. Use exactly the training prompt for first runs |
| `--max-duration SEC`          | `30`                                                          | Increase only after a successful bounded run                    |
| `--pc2-host HOST`             | `10.0.1.41` (wired LAN)                                       | WiFi address differs — check `x2_pc2_daemons.sh print-env`     |
| `--modality-config PATH`      | `gear_sonic/data/x2_modality_config_omnihand_stereo.py`       | Uses `stereo_left` + `stereo_right` from PC2 cameras            |
| `--bridge-py PATH`            | `~/miniconda3/envs/env_isaaclab/bin/python`                   | Override for non-Blackwell GPUs                                 |
| `--inference-min-period-s S`  | `0.8`                                                         | Matches the 40-step chunk horizon at 50 Hz                      |
| `--skip-preflight`            | off                                                           | Bypass connectivity probes. Do NOT use on a powered robot.      |
| `--no-cameras-autostart`      | off                                                           | Skip the auto-SSH `x2_pc2_cameras.sh serve` if you manage the camera bridge manually |

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

### Full record session WITH X2 head cameras (Orbbec + IMX900 stereo)

Adds the three real PC2 head-camera streams to the dataset alongside
the MuJoCo `ego_view`. Auto-starts the ROS→ZMQ bridge on PC2 over SSH
before launching the recorder; tear it down later with
`gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-stop`. Resulting
parquets carry the extra video features
`observation.images.{head_front,stereo_left,stereo_right}`.

Pre-flight (once per robot boot): make sure all three head cameras
came up properly on PC2 — there is a known boot-time Argus race that
sometimes loses the IMX900 stereo pair, fixed by bouncing the HAL:

```sh
gear_sonic_deploy/scripts/x2_pc2_cameras.sh status   # see what pubs exist
gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal  # bounce hal_sensor_orin if stereo pubs=0
gear_sonic_deploy/scripts/x2_pc2_cameras.sh grab     # save one JPEG per cam for visual check
```

Then record:

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_quest3_sonic_cams_v1 \
    --task "wave hello with both hands" \
    --sim-omnihand \
    --wrist-bypass ik \
    --head-cameras \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

Replay all four video tracks side-by-side:

```sh
DS=data/lerobot/x2_quest3_sonic_cams_v1
EP=000000
for cam in ego_view head_front stereo_left stereo_right; do
    xdg-open ${DS}/videos/chunk-000/observation.images.${cam}/episode_${EP}.mp4 &
done
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
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --all
```

Or build them one at a time:

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceCube && \
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceBowl && \
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceApple
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

## X2 Quest 3 planner stack (locomotion + arms + record)

`gear_sonic/scripts/run_x2_quest3_planner_stack.sh` is the all-in-one
wrapper that brings up the heuristic locomotion **planner**, the Quest 3
**manager**, the C++ **deploy** (sim or real), and (optionally) the
LeRobot **recorder** under one trap-cleaned shell with one Ctrl-C. Use
this for **mobile-manipulation** episodes where the operator drives the
lower body via the left thumbstick (planner) while their arms / hands
track in VR.

Difference vs `record_x2_dataset.sh` (the section just above): that
script runs the recorder + deploy in *teleop-only* mode where the lower
body holds the SONIC stand pose for the whole episode. The planner stack
adds locomotion on top, so the operator can walk into / around the
robocasa scene before manipulating.

For the full operator cheat sheet (button mappings, mode transitions,
audio cues, camera cycling, recording controls), see
[`docs/source/tutorials/x2_quest3_planner_stack_cheatsheet.md`](docs/source/tutorials/x2_quest3_planner_stack_cheatsheet.md).
For the engineering architecture (process topology, ZMQ port + topic
catalogue, CONFLATE/HWM matrix, boot/shutdown sequencing, complete
invocation matrix incl. tests), see
[`docs/source/references/x2_quest3_planner_stack_architecture.md`](docs/source/references/x2_quest3_planner_stack_architecture.md).

### Quick teleop only (no recording, flat floor)

Smoke-test the 4-process stack without writing data; useful first time
after a fresh checkout to confirm planner / manager / deploy come up.

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh --duration 300
```

### Auto-play a scripted demo at startup, then idle for VR takeover

Pre-loads a scripted YAML demo into the planner's command queue at
boot. The planner plays through the sequence (each `intent: ...`
entry runs through the FSM blend / play / blend cycle), drains back
to `idle_stand` when the queue empties, and sits in `IDLE_LOOP`
indefinitely waiting for the operator. The first VR-driven
`planner_cmd` (operator straps on the headset, chord-presses
**A+B+X+Y** to enter LOCOMOTION, then nudges a stick) calls
`replace_pending` and preempts whatever's still queued, so a
half-played demo can be interrupted mid-flight without restarting the
stack.

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 0 \
    --planner-demo gear_sonic/data/scripted_demos/eleven_motion_sequence.yaml
```

Available demos (full inventory in
[`x2_heuristic_planner.md` § Scripted demo gallery](docs/source/references/x2_heuristic_planner.md#scripted-demo-gallery)):

| Demo | What it plays |
| --- | --- |
| `eleven_motion_sequence.yaml` | 11-bin smoke covering every working family (fwd_step, turns, side steps, crouch, leans, torso twists). |
| `forward_back_turn.yaml` | continuous walk fwd / back + two turns. |
| `static_reach.yaml` | full lean + torso-twist reach ladder, no locomotion. |
| `gallery_crouch.yaml` | crouch_medium ×3. |
| `gallery_fwd_back_shuffle.yaml` | fwd_step + back_step variants. |
| `manipulation_approach.yaml` | locomanipulation approach + reach. |
| `side_steps_only_smoke.yaml` | side_left_step + side_right_step. |
| `six_motion_smoke.yaml` | shorter 6-bin smoke. |

Mutually exclusive with `--vla-bridge` / `--vla-no-policy` (those
modes replace the heuristic planner with the live VLA bridge and have
no command queue to seed). Compatible with `--with-record` if you
want the auto-played intro stitched into the same LeRobot episode the
operator records afterward.

### Record mobile-manipulation episodes (flat floor)

Same scene as the kinematic / SONIC recorders above (no table, no
objects), but the planner is alive so the operator can walk the robot
between captures.

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 600 --with-record \
    --output-dir data/lerobot/x2_planner_stack_v0 \
    --task "wave hello while walking forward"
```

### Robocasa scene + record (recommended VLA recipe)

The wrapper auto-resolves the scene XML from `--robocasa-env`, forwards
`--sim-mjcf` to the deploy bridge, forwards `--robocasa-env` + scene
ports (`5559` PUB / `5560` SUB) to the recorder, and **auto-enables
`--apply-curl-compensation` + `--apply-oppose-compensation`** (defaults
that match the recorder's robocasa flow above). `--task` is optional in
robocasa mode — the recorder auto-fills the language label from the
scene metadata (e.g. *"pick up the red cube and drop it into the blue
bowl"*).

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 1200 --with-record \
    --output-dir data/lerobot/x2_pick_place_cube_v0 \
    --robocasa-env X2PickPlaceCube
```

The startup banner echoes the resolved configuration:

```
scene            : X2PickPlaceCube -> .../robocasa_scenes/X2PickPlaceCube.xml
finger comp      : curl=on  oppose=on  (robocasa default; pass --no-apply-{curl,oppose}-compensation to override)
ports            : pose=5556  x2_debug=5557  planner_cmd=5563
                   arm/hands=5564  body_pose=5565
                   scene_state=5559  scene_reset=5560
```

Operator buttons (Quest 3): **B** toggles `OFF → LOCOMOTION → ARM_MAN`,
**A** engages IK in `ARM_MAN`, **X** starts a fresh episode (this is
when the mirror calls `reset()` and randomises the cube), **Y** saves.

### Reproducible per-episode object placement

Pass `--episode-seed` to lock the random cube / bowl pose to a known
value. Useful for A/B comparisons across recording sessions.

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 600 --with-record \
    --output-dir data/lerobot/x2_pick_place_cube_seed42 \
    --robocasa-env X2PickPlaceCube \
    --episode-seed 42
```

### Custom scene XML override

If you've built a tweaked scene (e.g. moved the table, swapped the
object), point both deploy and recorder at it explicitly:

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 600 --with-record \
    --output-dir data/lerobot/x2_pick_place_cube_custom \
    --robocasa-env X2PickPlaceCube \
    --scene-xml-path /path/to/custom_scene.xml
```

### Override finger compensations

Force-disable in robocasa mode (e.g. while debugging the OmniHand
finger pipeline — see also
[`bug-tracker/thumb-closing-bug.md`](bug-tracker/thumb-closing-bug.md)):

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 600 --with-record \
    --output-dir data/lerobot/x2_pick_place_cube_no_comp \
    --robocasa-env X2PickPlaceCube \
    --no-apply-curl-compensation --no-apply-oppose-compensation
```

Force-enable in flat-floor mode (the inverse — same compensation knobs
the robocasa default would have given you, but on the bare-floor
scene):

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --apply-curl-compensation --apply-oppose-compensation
```

### Sanity-check args without launching anything

`--validate-only` runs the full pre-flight (port collision, calibration
YAML, scene MJCF existence, ONNX model presence) then exits 0. Used by
CI and by the `tests/test_run_x2_quest3_planner_stack_cli.py` smoke
suite to validate the wrapper's CLI contract without spawning Quest 3 /
deploy children.

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --validate-only --robocasa-env X2PickPlaceCube --with-record \
    --output-dir /tmp/dryrun
```

### Recovery — kill orphans, free ports

```sh
gear_sonic/scripts/run_x2_quest3_planner_stack.sh --cleanup-only
```

Frees ZMQ ports `5556` / `5557` / `5563` / `5564` / `5565` (plus
`5559` / `5560` if a previous robocasa run leaked), kills orphan
planner / manager / recorder processes by PID file + name, and stops
any leftover `gr00t-x2sim` / `x2sim-run` Docker containers from a
crashed deploy.

### Useful flags

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--duration N` | `0` (unlimited; Ctrl-C to stop) | Auto-shutdown after N seconds (deploy `--max-duration`). Pass `0` (or omit) for no limit. |
| `--with-record` | off | Spawn the LeRobot recorder. Requires `--output-dir`. |
| `--output-dir PATH` | — | Required with `--with-record`. |
| `--task STR` | required in flat-floor; **optional in robocasa** | Language instruction stamped on every frame. Robocasa mode auto-fills from scene metadata. |
| `--robocasa-env {none,X2PickPlaceCube,X2PickPlaceBowl,X2PickPlaceApple}` | `none` | Load a Robocasa scene; flat floor when `none`. Build the XMLs first via `python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env <ENV>` (or `--all`). |
| `--scene-xml-path PATH` | auto-resolved from env | Override `gear_sonic/data/assets/robocasa_scenes/<env>.xml`. |
| `--episode-seed INT` | (numpy global RNG) | Reproducible per-`start` object placement. |
| `--apply-curl-compensation` / `--no-apply-curl-compensation` | **ON in robocasa**, OFF otherwise | Per-finger curl stretch (boosts mid-range curls toward CLOSED). Applied on the manager side (which owns the Retargeter in subscribe-mode). |
| `--apply-oppose-compensation` / `--no-apply-oppose-compensation` | **ON in robocasa**, OFF otherwise | Thumb-oppose stretch (suppresses 5–25 % rest-bleed at the open hand). |
| `--operator-id NAME` | `default` | Loads `data/operator_calibrations/<NAME>.yaml`. |
| `--calibration PATH` | auto from `--operator-id` | Explicit calibration override. |
| `--wrist-bypass {off,ik}` | `ik` | Override SONIC's wrist target with the operator's IK reference. Same semantics as `record_x2_dataset.sh`. |
| `--planner-demo PATH.yaml` | (none) | Pre-load a scripted-demo YAML into the planner's command queue at startup. The planner plays through it, returns to `idle_stand`, and waits for the operator's first VR `planner_cmd` (which preempts via `replace_pending`). Same YAML schema as `gear_sonic/data/scripted_demos/*.yaml`. Mutually exclusive with `--vla-bridge` / `--vla-no-policy`. |
| `--no-deploy` | off | Skip launching `deploy_x2.sh` (assume external). |
| `--no-sim-viewer` | off | Run the deploy headless (no MuJoCo viewer; headset still required). |
| `--sim-profile {parity,manual}` | `parity` | Deploy SONIC profile. `parity` matches the bake-vs-planner reference; `manual` skips the RSI anchor. |
| `--model PATH` | env `X2_PLANNER_SMOKE_MODEL` or h200-iter-25000 | ONNX policy. |
| `--scene-state-port INT` / `--scene-reset-port INT` | `5559` / `5560` | Override scene-mirroring ports if 5559 / 5560 are already bound by another stack. |
| `--validate-only` | off | Pre-flight + exit 0 (no children spawned). For CI / smoke validation. |
| `--cleanup-only` | off | Free leaked ports / kill orphan processes / stop containers, then exit. |

### One-time setup before first robocasa launch

If you haven't built the scene XMLs yet (or pulled a fresh checkout
that doesn't have them committed), the wrapper will refuse to start
with a helpful error pointing at the build command. Run it once:

```sh
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --all
```

Output lands in `gear_sonic/data/assets/robocasa_scenes/<env>.xml`
plus a `<env>.json` metadata sidecar (the wrapper passes the XML path;
both the deploy bridge and the recorder auto-discover the JSON sidecar
at startup).

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

### SONIC-loop replay (full deploy + v5 future-window wire)

Plays a recorded episode through the **live SONIC deploy** (sim or
real robot), so the body physically tracks the recording instead of
just viewing the parquet in a passive MuJoCo viewer. Single-shell
launcher; spawns the deploy + the replay client and tears them down
in reverse order on Ctrl-C. Required reading on the wire contract:
[2026-06-22 milestone](docs/source/user_guide/milestones/2026-06-22_dataset_replay_v5_wire.md).

```sh
# Sim (default): brings up deploy_x2.sh sim --vla --sim-with-omnihand
# --sim-viewer alongside the replay client.
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0

# Sim + recorded-camera rerun viewer side-by-side.
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 --with-rerun

# Real-robot first pass: half-speed, e-stop in reach. Requires
# x2_pc2_daemons.sh start to be running on PC2 already.
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 \
    --pc2-host 192.168.86.32 --rate-scale 0.5 --with-rerun

# Free :5556 + kill orphan deploy/replay processes after a crashed run.
./gear_sonic/scripts/run_x2_replay_stack.sh --cleanup-only
```

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `--dataset NAME-OR-PATH` | — | Dataset root under `data/lerobot/` or any path. |
| `--episode INT` | `0` | Zero-indexed episode. |
| `--rate FLOAT` | dataset native fps | Publish rate override. Future-window stride uses native fps (independent of this knob). |
| `--rate-scale FLOAT` | `1.0` | Multiplier on rate. `0.5` = half-speed wall-clock for cautious first passes. |
| `--loop` | off | Loop the episode indefinitely. |
| `--countdown S` | `3.0` | Hold-frame-0 warm-up before the trajectory starts (gives the deploy's handoff ramp time to settle). |
| `--hold-on-exit S` | `0.5` | Hold-the-last-frame soft stop. SONIC decays PD gains in ~200 ms. |
| `--no-deploy` / `--pc2-host HOST` | — | Skip spawning a sim deploy (external deploy / real robot, respectively). |
| `--with-rerun` | off | Spawn `view_x2_recorded_dataset.sh` for the same episode; GUI outlives this wrapper. |
| `--no-sim-viewer` / `--sim-profile {handoff,parity,manual}` | viewer on, `handoff` | Sim deploy knobs forwarded to `deploy_x2.sh`. |
| `--duration S` | `0` | Wall-clock cap; `0` = run until the replay's own end-of-episode signal. |
| `--cleanup-only` | — | Free `:5556`, sweep stale `x2sim` docker containers, exit. |

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
