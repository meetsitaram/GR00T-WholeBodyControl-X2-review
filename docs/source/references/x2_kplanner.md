# X2 Neural Kinematic Locomotion Planner (kplanner)

The **kplanner** is a trained-checkpoint replacement for the curated
`x2_heuristic_planner`. It streams 31-DOF body refs + root quaternion
at 50 Hz on the same `body_pose` / `pose` ZMQ wire — so the rest of the
X2 stack (`record_x2_dataset`, MuJoCo viewer, C++ deploy) consumes it
without any plumbing changes — but the frames come from a MotionBricks
VQVAE + pose + root model triple instead of a primitives PKL and a state
machine.

Why we built it: the heuristic planner stitches pre-baked clips with
linear blends, and the available bin matrix is small, hand-curated, and
struggles to express continuous velocity targets (Quest 3 thumbsticks
are inherently continuous). The kplanner takes a 4-D velocity intent
`[yaw_rate_rad_s, vel_x_m_s, vel_z_m_s, hip_height_m]` and conditions
`motion_inference.predict()` on it directly, generating a fresh future
window every replan. The bins concept is no longer load-bearing.

The kplanner became the **default** planner in
`run_x2_quest3_planner_stack.sh` as of this rollout. The heuristic
planner stays available behind `--planner heuristic` for fallback /
A-B comparison.

## Where this fits in the X2 stack

```
   Quest 3 (WebXR)
        │
        ▼
   quest3_manager_x2 ─── planner_cmd (intent + magnitude) ───▶ x2_kplanner.py  ◀── this doc
                                                                    │
                                            intent_to_velocity()     │
                                            [yaw_rate, vx, vz, h]    │
                                                                    │
                                            NeuralPlannerCore.       │
                                            replan_with_velocity()   │
                                            (worker thread, ~60 ms   │
                                            on CPU / <15 ms GPU)     │
                                                                    │
                                            ring buffer of           │
                                            48 future mujoco_qpos    │
                                            frames                   │
                                                                    │
                                            50 Hz publish loop ──────┘
                                                                    │
                                                       body_pose ZMQ │ (+ 9-frame future window)
                                                                    │
                                                                    ▼
                                                       record_x2_dataset (merge)
                                                                    │
                                                                pose ZMQ
                                                                    ▼
                                                         X2 C++ deploy → SONIC → robot
```

The wire format on `body_pose` is **identical** to the heuristic planner
— same `pack_pose_message` payload (`joint_pos_mj`, `root_quat_xyzw`,
`motion_token`, `left_hand_joints`, `right_hand_joints`, `frame_index`,
plus the v4 future window `joint_pos_mj_future` / `root_quat_xyzw_future`
/ `joint_vel_mj_future` / `frame_index_future` / `future_dt_s`). The
recorder + deploy do not know which planner is upstream and do not need
to.

## Components

### 1. `NeuralPlannerCore` (robot-agnostic)

Lives in
[`motionbricks/motion_backbone/inference/neural_planner.py`](../../../motionbricks/motionbricks/motion_backbone/inference/neural_planner.py).
This is a slim refactor of the G1 demo's
[`full_navigation_agent`](../../../motionbricks/motionbricks/motion_backbone/demo/full_agent.py),
stripped of:

- the G1-specific clip-holder pickle (the blendspace of pre-baked clips
  the demo uses to pick a target end-pose),
- the spring-damped world-frame target-position model,
- the WASD controller adapter and discrete `mode` channel.

What's left is the predict-and-decode core:

- **Ring buffer** of decoded `mujoco_qpos` frames (default capacity
  ~48 frames after each replan; 50 Hz → ~0.96 s of headroom).
- `reset(init_mujoco_qpos)` seeds the buffer with a stand pose (used at
  daemon boot before the first predict completes).
- `replan_with_velocity(target_local_root_values)` takes the last 4
  qpos frames as past context, builds the 8-frame constraint window
  (4 past + 4 target), and calls
  [`motion_inference.predict(...)`](../../../motionbricks/motionbricks/motion_backbone/inference/motion_inference.py)
  with the velocity broadcast onto the target frames.
- Constraint masks (`has_global_root_values`, `has_local_root_values`,
  `has_local_poses`) match the G1 demo's "no target world position, no
  target pose, just velocity" mode:
  - `has_local_root_values[:, 3] = False` (last past-frame velocity
    is invalid; no `t+1` to differentiate against).
  - `has_global_root_values[:, -4:] = False` (no target world XY).
  - `has_local_poses[:, -4:] = False` (no target joint pose — let the
    VQVAE free-sample locomotion poses consistent with the velocity).

Decoded `[B, T, feat_dim]` motion features are converted to MuJoCo qpos
via the robot-specific converter (the only robot-specific dependency).

### 2. `X2PlannerPaths` + `load_x2_planner()` (X2-specific)

Lives in
[`motionbricks/motion_backbone/inference/load_x2_planner.py`](../../../motionbricks/motionbricks/motion_backbone/inference/load_x2_planner.py).

- Reads each stage's `hparams.yaml` from
  `motionbricks/out/motionbricks_{vqvae,pose,root}_x2/version_1/` and
  re-instantiates the pose + root LightningModules via Hydra. Mirrors
  the construction patterns in `train_pose_x2.py` / `train_root_x2.py`
  but skips the dataset / Trainer / wandb plumbing.
- Loads `state_dict` from each `.ckpt`. Tolerates VQVAE keys missing
  from the pose checkpoint (they're re-loaded by
  `MotionModel._load_vqvae_models` from `args.vqvae_model_ckpt_path`).
- Wraps the pose + root models in `motion_inference` (the predict
  entry point) and pairs them with `X2MujocoQposConverter` (38-col
  qpos: 3 root_xyz + 4 root_quat + 31 hinges).
- Returns a ready-to-stream `NeuralPlannerCore`.

The on-disk default is a **pinned step checkpoint** from each
`version_1/checkpoints/` directory rather than `last.ckpt`, so a fresh
training run doesn't silently re-point inference at an unverified
checkpoint. Source of truth:
[`X2PlannerPaths.default()`](../../../motionbricks/motionbricks/motion_backbone/inference/load_x2_planner.py).
Current pins (mirrored in `x2_kplanner.py`'s argparse, the stack
wrapper's preflight, and the kplanner pytest skip-gate):

| Stage | Default checkpoint |
| --- | --- |
| VQVAE | `motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0200000.ckpt` |
| Pose | `motionbricks/out/motionbricks_pose_x2_v2/version_1/checkpoints/model-step=0250000.ckpt` |
| Root | `motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0235000.ckpt` |

When you promote a newer checkpoint, update all four call sites in the
same commit (the wrapper preflights against its own paths; if they
drift you'll get a confusing "checkpoint not found" before the daemon
ever runs). Or override at runtime via
`--kplanner-{vqvae,pose,root}-ckpt` in the stack wrapper / `--*-ckpt`
on the daemon directly.

### 3. `x2_kplanner.py` daemon (the entry point)

Lives in
[`gear_sonic/scripts/x2_kplanner.py`](../../../gear_sonic/scripts/x2_kplanner.py).

Forks the heuristic daemon's process hygiene, signal handling, PID file
ownership, scripted-demo + keyboard + ZMQ command sources, and 50 Hz
publish loop. The differences from the heuristic:

- No primitives PKL / bins YAML — the model is the motion source.
- Default PID file `/tmp/x2_kplanner.pid` (heuristic uses
  `x2_heuristic_planner.pid`); the stack wrapper cleans both.
- Default publish topic `body_pose` (same as heuristic Phase 0 mode).
- A dedicated **worker thread** runs `predict()` so the publish thread
  stays on its 20 ms tick even while predict is in flight. The worker
  wakes every 50 ms or when the publish thread signals "ring buffer is
  getting empty" (`should_replan()`); see `_planner_worker` in the
  daemon source.
- An **`intent_to_velocity()`** dispatcher translates the
  `(intent, magnitude)` pairs that come over `planner_cmd` (Quest 3 or
  scripted YAML) into the 4-D velocity vector.

#### Intent → velocity dispatcher

The manager's `IntentDecoder` was authored for the **heuristic**
planner's curated motion bins (`deg_15/30/45/90` turns,
`quarter_ft / half_ft / one_ft` strides). The kplanner does **not**
honour those labels as literal angles or distances — for the neural
model they are intensity buckets scaling a continuous velocity. To
keep the vocabulary contract with the manager intact while not
locking the kplanner into the heuristic's labels, the dispatcher
splits the table into two pieces:

1. **Direction-explicit base vectors** (`_BASE_VELOCITY`): one row per
   intent name. Direction is in the intent (`fwd_step` vs `back_step`,
   `turn_left` vs `turn_right`, `side_left` vs `side_right`).
2. **Magnitude → scalar multiplier** (`_TRANSLATIONAL_SCALE`,
   `_TURN_SCALE`): `default = 1.0`; the legacy `*_ft` / `deg_*` labels
   are mapped to fixed multipliers. **Unknown magnitudes fall back to
   1.0** rather than idling, so a manager-side vocabulary addition
   doesn't silently freeze the planner before the kplanner table
   catches up.
3. The `walk` intent is a manager-side legacy where direction is in
   the **magnitude** (`forward` / `backward` / `fast`), not the intent
   name; resolved as a separate small table.

The precomputed `INTENT_VELOCITY_MAP` (every key the dispatcher
explicitly recognises) is rebuilt from the dispatcher at import time;
the two surfaces are guaranteed to agree.

Constants (`gear_sonic/scripts/x2_kplanner.py`):

```
_WALK_SPEED_MPS              = 0.5    # fwd_step / walk-forward 1× speed
_FAST_WALK_SPEED_MPS         = 0.9    # walk-fast
_BACK_SPEED_MPS              = 0.35   # back_step / walk-backward 1× speed
_SIDE_SPEED_MPS              = 0.4    # side_left / side_right 1× speed
_TURN_45_RAD_S               = 1.5    # bucketed turn_left/turn_right deg_45
                                      # yaw rate (button-driven pivots only)
_CONTINUOUS_TURN_MAX_RAD_S   = 0.75   # Quest 3 R-stick full-deflection yaw
                                      # ceiling (analog turns). Decoupled
                                      # from _TURN_45_RAD_S because the
                                      # current X2 root model is OOD for
                                      # yaw_rate > ~1.0 rad/s. Tune with
                                      # --continuous-turn-max-rad-s.
_HIP_HEIGHT_M                = 0.687  # target pelvis Y (channel 3) -- must
                                      # match training distribution (PKL
                                      # pelvis_z mean ~0.66 m); 0.95 was
                                      # OOD for current model
```

| `(intent, magnitude)` | Resolved value | Source |
| --- | --- | --- |
| `idle / default` or `idle / stand` | `(0, 0, 0, 0.687)` | idle |
| `walk / forward` | `(0,  0.5, 0, 0.687)` | walk table |
| `walk / backward` | `(0, -0.35, 0, 0.687)` | walk table |
| `walk / fast` | `(0,  0.9, 0, 0.687)` | walk table |
| `fwd_step / default` | `(0,  0.5, 0, 0.687)` | base × 1.0 |
| `fwd_step / quarter_ft` | `(0,  0.25, 0, 0.687)` | base × 0.5 |
| `fwd_step / half_ft` | `(0,  0.5, 0, 0.687)` | base × 1.0 |
| `fwd_step / one_ft` | `(0,  0.75, 0, 0.687)` | base × 1.5 |
| `back_step / default` | `(0, -0.35, 0, 0.687)` | base × 1.0 |
| `back_step / quarter_ft` | `(0, -0.175, 0, 0.687)` | base × 0.5 |
| `back_step / half_ft` | `(0, -0.35, 0, 0.687)` | base × 1.0 |
| `side_left / default` | `(0, 0,  0.40, 0.687)` | base × 1.0 |
| `side_right / default` | `(0, 0, -0.40, 0.687)` | base × 1.0 |
| `turn_left / deg_45` |  `( 1.5, 0, 0, 0.687)` | base × 1.0 (= baseline) |
| `turn_left / deg_15` |  `( 0.5, 0, 0, 0.687)` | base × ⅓ |
| `turn_left / deg_30` |  `( 1.0, 0, 0, 0.687)` | base × ⅔ |
| `turn_left / deg_90` |  `( 3.0, 0, 0, 0.687)` | base × 2 |
| `turn_right / *` | mirror with negated yaw | base × scale |

Channels match the MotionBricks `LocalRootLocalBody` representation
(`local_root_rot_vel`, `local_root_vel_x`, `local_root_vel_z`,
`global_root_y`). In MotionBricks motion-rep coordinates Y is the
gravity axis, so the lateral channel is `vel_z` — not `vel_y`. See
[`motionbricks/motionlib/core/motion_reps/motion_reps_base/local_root_local_body.py`](../../../motionbricks/motionbricks/motionlib/core/motion_reps/motion_reps_base/local_root_local_body.py)
for the canonical channel-order definition.

#### Yaw lock and runtime velocity scales

Two operator-tunable knobs sit downstream of the dispatcher, both
addressed at issues the static table can't fix:

- **Yaw lock** (`--yaw-lock-epsilon`, default `0.05` rad/s). When the
  commanded `yaw_rate` is below this threshold the kplanner publishes
  the **persisted world-frame `root_quat`** instead of the model's
  per-frame prediction. The neural model has small per-frame yaw
  deltas even when the operator commanded `yaw_rate = 0` (pure
  forward / back / sidestep / idle); without the lock those deltas
  compound across replans into a visible spin because the SONIC
  policy tracks the published yaw. Joint angles still come from the
  model so the gait keeps stepping — only the root orientation
  reference is pinned. Pass `0` to disable the lock entirely.
- **Runtime velocity scales** (`--turn-left-scale`,
  `--turn-right-scale`, `--forward-scale`, `--backward-scale`,
  `--lateral-scale`, all default `1.0`). Multipliers applied **after**
  the dispatcher resolves the velocity, before the worker thread
  feeds it to the model. Two intended uses:
  1. Compensate model-side L/R asymmetry without retraining
     (`--turn-left-scale 1.5` if left turns are weaker than right).
  2. Shave the global walk magnitude (`--forward-scale 0.6`) when the
     SONIC policy can't track the dispatcher's baseline 0.5 m/s on
     this hardware / firmware combo.

  Scales never reanimate a `_IDLE_INTENT` — the multiplier path runs
  only on resolved non-idle velocities, so `hold_torso` / `lean_*` /
  `crouch` stay idle regardless of scale. The unit test
  `test_runtime_scales_do_not_reanimate_idle_intent` pins this
  contract.

All six knobs forward through `run_x2_quest3_planner_stack.sh` as
`--kplanner-yaw-lock-epsilon`, `--kplanner-turn-left-scale`, etc.,
and via environment variables (`KPLANNER_YAW_LOCK_EPSILON`,
`KPLANNER_TURN_LEFT_SCALE`, ...).

**No-op intents.** Manager labels that have no locomotion meaning to
the kplanner — `hold_torso / continuous`, `lean_fwd / *`,
`torso_left|right / *`, `crouch / *` — resolve to `_IDLE_INTENT`. The
heuristic planner handles them inside its state machine; the kplanner
does not currently expose a separate torso / crouch channel. If you
need upper-body / torso work, run with `--planner heuristic`.

**Tuning.** Edit the constants above and the two scale tables; values
take effect on the next daemon restart, no model retrain needed. The
parity / vocabulary regression net for this surface is
`tests/test_x2_kplanner_intent_velocity.py` (50 fast assertions; runs
without the MotionBricks training stack installed).

## CLI surface

```
python -m gear_sonic.scripts.x2_kplanner \
    --device cuda                                          # or cpu
    --body-pose-port 5565                                  # phase-0 recorder-merge mode
    --warmup-quiet-stand-s 0.5                             # frozen anchor pre-roll
    --zmq-cmd-host 127.0.0.1 --zmq-cmd-port 5563           # Quest 3 / external drive
    --zmq-cmd-topic planner_cmd
    --replan-threshold-frames 16                           # refill when buffer < this
    --vqvae-ckpt /...last.ckpt                             # optional overrides
    --pose-ckpt /...last.ckpt
    --root-ckpt /...last.ckpt
    --warmup-qpos-path /...balanced_stand.pkl              # optional stand seed
```

Most operators won't run the daemon directly; the stack wrapper
(`run_x2_quest3_planner_stack.sh`, default `--planner kplanner`) is the
intended entry point.

### Stack wrapper integration

`run_x2_quest3_planner_stack.sh --planner kplanner` (the default) does
all of:

- Skips the heuristic primitives / bins preflight (kplanner doesn't
  consume them).
- Bakes a **kplanner-specific** parity RSI anchor PKL via
  `bake_kplanner_rsi_anchor.py` (see "MuJoCo spawn pose" below).
  Default `--sim-profile parity` is honored just like the heuristic
  flow — the robot spawns on the floor at the kplanner's warmup
  quiet-stand qpos, not in mid-air on an elastic band.
- Spawns the kplanner with the right ZMQ ports / PID file path.
- Waits for the `"first replan complete"` log line (up to 90 s — first
  replan is the slowest the daemon will ever do because torch graphs
  are compiled on demand).
- Cleanup-only mode (`--cleanup-only`) sweeps both heuristic and
  kplanner PID files.

To fall back to the heuristic:

```bash
gear_sonic/scripts/run_x2_quest3_planner_stack.sh --planner heuristic \
    --duration 0
```

### MuJoCo spawn pose (parity RSI contract)

The X2 sim deploy supports two MuJoCo spawn profiles:

| Profile | Initial pose | Used by |
| --- | --- | --- |
| `--sim-profile parity --motion <PKL>` | Bridge RSIs from frame 0 of the PKL. Robot spawns **on the floor** in the exact pose the policy is about to track. | Default for both heuristic and kplanner |
| `--sim-profile manual` | Pelvis spawns at z=0.85 m on an elastic band that auto-releases after ~2 s. Robot drops if no pose ref arrives during cold-start. | Opt-in only via explicit `--sim-profile manual` |

For the kplanner the parity anchor PKL is produced by
[`bake_kplanner_rsi_anchor.py`](../../../gear_sonic/scripts/bake_kplanner_rsi_anchor.py),
which is the kplanner-side analogue of
[`bake_planner_rsi_anchor.py`](../../../gear_sonic/scripts/bake_planner_rsi_anchor.py).
The anchor frame is resolved by the daemon's
[`_build_default_warmup_qpos()`](../../../gear_sonic/scripts/x2_kplanner.py)
to the deploy's **`training_default_angles`** stand pose — the same
31-DOF + hip-Z configuration the C++ deploy holds during `SAFE_IDLE`
(see `policy_parameters.hpp::default_angles`). Floor-contact hip Z =
0.636 m. Knees +0.669 rad (38° bend), hips −0.312 rad, ankles
−0.363 rad, elbows −0.6 rad. The hip Z value comes from forward
kinematics on the canonical X2 MJCF (mirrored by
`x2_pelvis_z_from_capture.py`'s baseline column).

Why this exact pose is the default: the deploy's pose-ref starvation
watchdog trips ~0.5 s after CONTROL when the recorder isn't yet
merging `pose:5556` frames (the recorder is Step 4/4 in the wrapper
and comes up ~10 s after the deploy enters CONTROL). On trip the
deploy enters `SAFE_IDLE` and PD-controls every DOF to
`training_default_angles` with 4× kd. RSI'ing the bridge to the same
pose means SAFE_IDLE produces **zero** joint-target delta — the robot
sits perfectly still through the multi-second cold-start window
instead of squatting / wobbling / collapsing under the SAFE_IDLE
pull. This was the root cause of the "robot started on the ground
but collapsed without loading any model" failure mode in the original
kplanner integration (a `balanced_stand.pkl` anchor with 0.16 rad
knee delta and 0.57 rad elbow delta from `default_angles` was being
yanked toward the SAFE_IDLE target with 4× kd and toppled).

Recorded real-robot stands such as
`gear_sonic/data/motions/x2_recorded/balanced_stand.pkl` (a 4001-frame,
30 fps IMU-derived recording of the X2 settling into a balanced
bent-knee stand at hip Z = 0.655 m) are kinematically valid but
**don't match `default_angles`**, so they trigger SAFE_IDLE drift
during cold-start. They remain available as an explicit
`--kplanner-warmup-qpos` override when the operator can guarantee the
deploy reaches CONTROL with fresh pose frames (recorder up first)
**before** the watchdog trips.

Override either default with `--kplanner-warmup-qpos PATH.pkl` (forwarded
to both the daemon's `--warmup-qpos-path` and the bake step) so the
bridge RSI pose and the daemon's first publish-tick remain bit-identical.
The loader accepts three schemas:

- Raw `[38]` or `[T, 38]` array (mujoco qpos, wxyz quaternion).
- Dict `{'mujoco_qpos' or 'qpos': [38] | [T, 38]}`.
- Deploy-PKL nested schema
  `{'<name>': {'root_trans_offset': [T, 3], 'root_rot': [T, 4] xyzw, 'dof': [T, 31], ...}}`
  — same shape as `bake_planner_rsi_anchor.py` output and the on-disk
  `x2_recorded/*.pkl` recordings.

Output PKL location:
`data/sim_to_real_anchors/browse_sonic/baked_pkls/x2_kplanner_rsi_anchor.pkl`.

The wrapper re-bakes on every kplanner run (it's ~200 ms and depends
only on the operator's `--kplanner-warmup-qpos` override). The bake
runs **before** Step 1 spawns the deploy, so the deploy gets
`--motion <PKL>` on its very first invocation — there's no "first run
spawns in mid-air, second run spawns on the floor" footgun.

Why this matters: the daemon's first replan takes ~5 s on GPU
(checkpoint load + first torch graph compilation), during which the C++
deploy is already running and waiting for pose refs. Manual profile
would drop the robot in that window even if the kplanner came up
perfectly. Parity profile spawns the robot on the floor in a stable
pose; the kplanner then publishes the same pose during its warmup
quiet-stand (default 2 s in the wrapper, configurable via
`--warmup-quiet-stand-s`); only then does the neural planner take over.

## Runbook

### Kinematic-only smoke (no deploy, no Quest 3)

Verify the daemon publishes a clean 50 Hz `body_pose` stream:

```bash
# Terminal 1: spawn the kplanner on a non-default port
python -m gear_sonic.scripts.x2_kplanner \
    --device cuda --body-pose-port 25565 \
    --pid-file /tmp/x2_kplanner_smoke.pid --duration-s 30

# Terminal 2: subscribe and verify cadence + wire format
python -c "
import sys, time, zmq
sys.path.insert(0, '.')
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message
ctx = zmq.Context.instance()
sub = ctx.socket(zmq.SUB)
sub.setsockopt_string(zmq.SUBSCRIBE, 'body_pose')
sub.connect('tcp://127.0.0.1:25565')
t0 = time.monotonic()
for i in range(100):
    raw = sub.recv()
    msg = unpack_message(raw, expected_topic='body_pose')
print(f'100 frames in {time.monotonic()-t0:.2f}s')
print('fields:', list(msg.fields))
"
```

Expected (CPU): ~2 seconds for 100 frames (50 Hz). On a Blackwell GPU
with a torch wheel built for sm_120 the cadence is identical, but the
first replan is much faster (<15 ms vs ~60 ms on CPU).

For a live MuJoCo render:

```bash
# Terminal 1 (same as above)

# Terminal 2: open the viewer subscribed to the kplanner's body_pose topic
python -m gear_sonic.scripts.view_x2_planner_mujoco \
    --from-zmq 127.0.0.1:25565 --topic body_pose
```

### Full Quest 3 stack

See the canonical operator runbook in
[`docs/source/tutorials/x2_quest3_planner_stack_cheatsheet.md`](../tutorials/x2_quest3_planner_stack_cheatsheet.md);
the default planner is now kplanner, no extra flag needed.

### Tuning the intent map

After running the full stack with `--with-record` enabled and Quest 3
driving, the operator typically wants to nudge the per-intent
velocities (e.g. "forward walk feels too slow"). Edit the
`_WALK_SPEED_MPS` / `_BACK_SPEED_MPS` / `_SIDE_SPEED_MPS` /
`_TURN_45_RAD_S` constants (or the `_TRANSLATIONAL_SCALE` /
`_TURN_SCALE` multipliers) in
[`gear_sonic/scripts/x2_kplanner.py`](../../../gear_sonic/scripts/x2_kplanner.py)
and restart the daemon — no model retrain needed. The model itself was
trained on the full X2 motion library (BONES seed) so the per-intent
velocities are continuous-domain inputs, not discrete bin selectors.

When you find values that feel right, update the table above in this
doc so the operator runbook reflects the live numbers.

## Failure modes + diagnostics

### Daemon fails to load

If `load_x2_planner()` raises at startup:

- `FileNotFoundError: vqvae_ckpt=...` → the default checkpoint path
  doesn't exist; pass `--kplanner-vqvae-ckpt` / `--kplanner-pose-ckpt`
  / `--kplanner-root-ckpt`, or train fresh checkpoints via
  `train_vqvae_x2.py` / `train_pose_x2.py` / `train_root_x2.py`.
- `ModuleNotFoundError: vector_quantize_pytorch` or
  `pytorch_lightning` → install the MotionBricks deps:
  `pip install vector-quantize-pytorch pytorch-lightning adam-atan2-pytorch colorlog`.
- `omegaconf.errors.InterpolationKeyError: 'trainer.log_every_n_steps' not found`
  → an older copy of `hparams.yaml` had a stale interpolation; pull the
  latest from the corresponding `train_*_x2.py` run, or set the
  missing key explicitly inside `_patch_hparams` in
  `load_x2_planner.py`.

### CUDA "no kernel image is available for execution on the device"

The base miniconda env's torch (2.6.0+cu124) is built for sm_50..sm_90
and crashes on Blackwell (RTX 5090, sm_120). The fix is to run under
`env_isaaclab`, which ships torch 2.7.0+cu128 with `sm_120` in its
compiled arch list. The kplanner dependencies are already installed
there as of 2026-05:

```bash
# Daemon directly
PYTHONPATH=motionbricks:$PWD \
    ~/miniconda3/envs/env_isaaclab/bin/python -m gear_sonic.scripts.x2_kplanner \
    --device cuda --body-pose-port 25565

# pytest under env_isaaclab
~/miniconda3/envs/env_isaaclab/bin/python -m pytest \
    tests/test_x2_kplanner_zmq_publish.py -v
```

Route the full stack's kplanner subprocess to env_isaaclab without
moving anything else off the default `.venv` python via the wrapper's
`--kplanner-python` knob (or the matching `KPLANNER_PYTHON` env var):

```bash
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --kplanner-python ~/miniconda3/envs/env_isaaclab/bin/python \
    --duration 600
```

On a 5090 with this routing the worker thread's per-replan latency
drops from ~500–800 ms (CPU) to ~14 ms, and the publish ring buffer
never starves. The wrapper logs the resolved kplanner python +
device on the boot banner so you can confirm the routing took.

If you really need to run on CPU (e.g. CI without a GPU), pass
`--device cpu`. The test fixture auto-detects this and loosens the
cadence band; set `KPLANNER_TEST_FORCE_CPU=1` to force the CPU path
even on a Blackwell-capable env.

### Publish cadence dips below 50 Hz

The daemon logs `loop fell behind by Xms; resyncing` when a tick slips
more than 5 ticks. If you see this frequently:

- Check the worker-thread predict latency in DEBUG mode (`-v`):
  `worker: replan done in Xms`. Sustained > 250 ms on GPU is a sign
  the model is paging or the host is under load.
- Verify the ring buffer isn't pinned to 1 frame (publisher continually
  popping the last frame because the worker can't keep up). DEBUG logs
  show `frames_remaining=N` after each replan.

## Relationship to the heuristic planner

| Aspect | Heuristic | kplanner |
| --- | --- | --- |
| Motion source | Curated primitives PKL | Trained MotionBricks (VQVAE + pose + root) ckpts |
| Wire format | `body_pose` / `pose` ZMQ (v4) | Identical |
| Default in stack wrapper | No (since this rollout) | Yes |
| Continuous velocity input | No (discrete bin selection) | Yes (4-D vector per replan) |
| Cold-start latency | <0.1 s (PKL load) | ~5–10 s (model load + first replan) |
| Predict per replan | n/a (table lookup) | ~10 ms GPU / ~60 ms CPU |
| Out-of-distribution behavior | Falls back to nearest bin | Generates novel poses; quality scales with training data |
| Configuration knobs | `x2_planner_bins.yaml` | `_BASE_VELOCITY` + scale tables in daemon source |

Use the heuristic planner when you want bit-identical replay of
curated demos (e.g. data-collection sessions that must be aligned with
the parity-RSI anchor), or when the kplanner's checkpoints aren't on
disk. Use the kplanner for everything else.
