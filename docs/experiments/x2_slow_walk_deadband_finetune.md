# X2 Slow-Walk / Micro-Adjustment Fine-Tune — Dead-Band Diagnosis & Fix

**Goal:** give X2 SONIC a slow, deliberate stepping gait (~0.1–0.3 m/s) for
micro-positioning during manipulation. Base and the executed-feasible 2k
checkpoints **stall below ~0.2 m/s**: they hold a good pose but do not step
forward. Target motions = the 34 recorded teleop clips
(`slow_walk_slow_keyboard_*`, `slow_walk_manipulation_vr_*`, `slow_walk_medium_*`,
keyboard+VR), retargeted G1→X2 (`gear_sonic/data/motions/g1_recorded_x2/`).

## Root cause: a REWARD dead-band (not data, not sim2sim)

Measured on the executed-feasible reward config:

- **`tracking_body_linvel` std = 1.0 m/s.** Reward is `exp(−err²/std²)`. At a
  0.12 m/s target, a robot that **stands still** scores `exp(−0.12²/1.0²) = 0.986`
  vs `1.000` for tracking. The velocity kernel is so wide that standing gets
  ~99% of the reward — **near-zero gradient to move slowly.**
- **`tracking_anchor_pos` weight = 0.5, std = 0.3 m.** The only term that
  punishes world-position lag is the lowest-weighted, and its wide kernel lets
  the robot drift ~0.3 m cheaply.
- The dominant rewards (relative body pos/ori, anchor_ori, VR 5-point) all reward
  **posture/orientation**, which a standing robot nails. So the reward-optimal
  low-speed policy is: hold a good pose, don't translate.

No amount of fine-tuning on the same reward fixes this — the gradient isn't there.

## The fix

| reward term | was | now | why |
|---|---|---|---|
| `tracking_body_linvel` std | 1.0 | **0.25** | at 0.12 m/s, standing drops 0.986 → ~0.79 (real gradient to move) |
| `tracking_anchor_pos` weight | 0.5 | **2.0** | make world-position lag expensive (the catch-up signal) |
| `tracking_anchor_pos` std | 0.3 | **0.15** | sharpen so small lag isn't free |

Config: `sonic_x2_ultra_slow_manip_focus.yaml` (focused validation) and
`sonic_x2_ultra_slow_manip_rebalance.yaml` (full-corpus deploy run).

## Results (focused validation run: 34 clips, warm-start from 2k, local RTX 5090)

Fine-tuned on **just the 34 slow clips** (`x2_slow_manip_focus.pkl`) — a deliberate
small-subset test of "point at a task-relevant motion set → fine-tune → capability,
no per-task oversampling."

**IsaacLab (training sim):** dead-band broke by **iter 506** — robot takes steps
where base/2k froze. NOTE: aggregate metrics (`error_body_lin_vel` etc.) went flat
after warm-start settling and did **not** reflect this — they average limb velocity,
so a stepping-in-place stall looks like a translating walk. Only the visual/translation
test reveals it.

**MuJoCo deploy (raw PD, no ElasticBand) — 8–15 s slow window, ref net travel 0.87 m:**

| checkpoint | robot travel | understep | peak speed | note |
|---|---|---|---|---|
| base 001376 | 0.06 m | 93% | 0.19 m/s | hard stall |
| ef 2k | 0.04 m | 96% | 0.22 m/s | hard stall |
| focus 506 | 0.04 m | 96% | 0.19 m/s | IsaacLab steps, MuJoCo still stalls |
| **focus 3000** | **0.63 m** | **27%** | **0.57 m/s** | **WALKS in MuJoCo, no fall** |

**Knob #1 (more training) closed most of the sim2sim gap.** The reward fix broke the
dead-band in IsaacLab by iter 506, but the slow gait only became robust enough to
**transfer to MuJoCo deploy** by iter 3000. The reward fix is necessary but not
sufficient — training duration made it transfer.

## Instrument lessons (which eval measures what)

| instrument | base | 2k | focus 3000 | measures |
|---|---|---|---|---|
| General feasibility (512 clips) | 85.9% | 92.2% | **62.7%** | general competence (the *cost*) |
| Slow feasibility (34 clips) | 94.1% | 97.1% | 97.1% | **nothing** — blind to the stall |
| Slow deploy translation (understep) | 93% | 96% | **27%** | the slow-walk *gain* |

- **IsaacLab feasibility is blind to the slow-walk stall.** Base "passes" 94% of the
  slow clips — the anchor gate tolerates a lagging robot (slow refs stay within the
  0.15 m gate; manip clips barely translate). Use **deploy understep**
  (`DEPLOY_SCORECARD.md`, `record_x2_eval_mujoco.py --traj-csv`) for slow-walk, not
  the feasibility sweep.
- **The focused checkpoint is a validation ckpt, not deploy** — general feasibility
  cratered 92.2 → 62.7 (fixable-gap 20 → 1). It traded broad competence for the slow
  skill by overfitting the 34 clips.

## Next: the rebalance (deploy) run

Apply the **same reward fix to the FULL executed-feasible corpus**, warm-started from
2k. With the sharpened `anchor_pos`, a stalling robot exceeds the 0.15 m anchor gate on
the slow clips → they **terminate → register as failures → the adaptive sampler
auto-concentrates on them** (no manual oversampling — capped at 200× mean failure rate,
so the ~34 slow clips draw a meaningful sample share while general clips stay covered).

Judge on **both** instruments:
- general feasibility ≥ ~92 (hold the line vs 2k), **and**
- slow-walk understep in the low-20s / single digits (the new skill).

## Tooling added

- `record_x2_eval_mujoco.py --onscreen` — live `launch_passive` MuJoCo viewer of the
  `.pt` policy (no ONNX/docker; `MUJOCO_GL=glfw DISPLAY=:1`). Fast way to eyeball deploy.
- `play_x2_motion_mujoco.py --start-sec/--end-sec/--world-cam` — windowed kinematic replay.
- `deploy_understep.py` / `record_x2_eval_mujoco.py --traj-csv` — deploy translation metric.
- Interactive stack for ONNX deploy: `run_x2_pkl_direct_stack.sh --model <onnx>` (band
  auto-releases at 3 s; `SIM_BAND_RELEASE_S=0` to disable) + `play_locomotion --pkl`.
