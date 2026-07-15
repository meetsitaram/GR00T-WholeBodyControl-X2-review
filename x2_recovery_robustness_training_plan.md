# X2 Recovery-Robustness Training Plan (future phase)

Design notes for a dedicated **disturbance-robustness** training phase — to fix
the real robot's **forward fall** and the sim behaviors we saw in MuJoCo push
testing (2026-07-14). This is a *next-run* plan, NOT a mid-warmstart tweak
(compute + destabilization risk). Validate each change with the sim push tool
before committing to a long run.

## Observations that motivate this
- Real robot: **stable in sim, falls forward** on hardware.
- MuJoCo head-push test (`eval_x2_mujoco.py`, `P` key, force at `head_pitch_link`):
  policy **collapses** under a moderate sustained force.
- It recovers with **many tiny shuffle steps**, not a few **decisive strides**.
- Nothing trains **steady CoM bias / payload** compensation.

## Root causes (verified in the configs/code)
1. **Push training is gentle + wrong form.** `push_robot` = *velocity impulse on
   the BASE* (±0.5 m/s x/y, ±0.52 rad/s pitch, interval 4-6 s @ level0_4). No
   upper-body *force* push → a hard head-level force is out-of-distribution.
2. **Tiny steps, not strides**, from three compounding forces:
   - it's a **reference tracker** → stays near the dance/walk reference, which has
     no recovery stride in its vocabulary;
   - **smoothness penalties** (`action_rate_l2`, `anti_shake_ang_vel`,
     `feet_acc`/`joint_acc_l2`) are **global (legs too)** and punish the fast,
     large, high-accel action a stride requires;
   - gentle velocity-impulse pushes never *force* a stride, so it never learns one.
3. **No steady-bias training.** `randomize_rigid_body_mass` is on (global ±20%
   scale) but `randomize_rigid_body_com` **exists yet is unused**; no hand-payload
   term; no constant-force term.

## Key insight: TWO disturbance types, TWO skills
- **Sudden burst** → *transient recovery* (catch a shove).
- **Steady force / CoM offset** → *steady-state compensation* (hold a persistent
  lean). A policy good at one is NOT automatically good at the other.
- The **steady/CoM** case is probably closer to the real forward-fall (persistent
  CoM gap / cabling / payload), so don't skip it in favor of only impulsive pushes.

## Changes to make (the robustness phase)

### A. Disturbance events (make recovery necessary)
1. **Strengthen impulsive push** — `push_robot`: velocity x/y ±0.5 → ±1.0-1.5 m/s,
   pitch ±0.52 → ~1.0 rad/s, interval 4-6 → 2-3 s.
2. **Add force-based upper-body push** — new event `apply_external_force_torque`
   on torso/head, curriculum-ramped magnitude. Analog of the real chest-nudge and
   the MuJoCo `P` test. (Does not exist yet — only base-velocity push.)
3. **Steady-state / CoM bias**:
   - **Wire in `randomize_rigid_body_com`** (already implemented,
     `events.py:106`) — randomize base/torso CoM ±few cm per episode (esp.
     fwd/back). Teaches steady posture compensation, robust to the *unknown* real
     offset (better than hand-fixing the URDF CoM).
   - **Hand payload** — targeted `randomize_rigid_body_mass` on wrist/hand bodies
     (larger add). Models "carrying something" that tilts CoM forward.
   - **Optional constant force** — reset-mode `apply_external_force_torque` held
     for the whole episode. Proxy for persistent bias without changing inertia.

### B. Reward changes (allow decisive strides)
1. **Relax LEG smoothness penalties** — lower `action_rate_l2`,
   `anti_shake_ang_vel`, `feet_acc` weights on the legs (arm-dynamics config
   already relaxed them for arms). THE lever for big strides over tiny steps.
2. **Reduce off-reference / foot-placement penalty during/after a push** — let it
   step off-reference to catch itself.

### C. Curriculum
- Ramp disturbance magnitude over training (gentle → strong) so early learning
  isn't destabilized.

## Tradeoffs / caveats
- **Decisive recovery vs smoothness**: relaxing smoothness → jittery dances +
  harder on real actuators. Tune the balance, don't zero them.
- **DR softens nominal precision** slightly — worth it for a must-not-fall robot.
- **Pure-tracker ceiling**: strong sustained pushes + relaxed smoothness gets far;
  the deepest fix is a dedicated balance/recovery objective, not just tracking.
- **Dedicated phase only** — meaningful reward/event change; needs its own run.

## Validation
- **Sim**: `eval_x2_mujoco.py` `P`-key head push (`--push-force`/`--push-body`).
  Metrics: recovery threshold (N), strides vs shuffles, steady-lean compensation
  under a sustained push / CoM offset. A/B new vs old checkpoint.
- **Real**: the push + recovery tests in `x2_real_robot_test_plan.md` (Phase 2/5).

## File / code pointers
- Push event: `gear_sonic/config/manager_env/events/terms/push_robot.yaml`
  (func `push_by_setting_velocity`)
- Active push level: `gear_sonic/config/manager_env/events/tracking/level0_4.yaml`
- Mass DR: `.../events/terms/randomize_rigid_body_mass.yaml` (global ±20% scale)
- CoM fn (UNUSED, wire it in): `gear_sonic/envs/manager_env/mdp/events.py:106`
  `randomize_rigid_body_com`
- Smoothness terms: `rewards/terms/{action_rate_l2,anti_shake_ang_vel,feet_acc}.yaml`
  (global; arm-dyn config relaxes for arms — need leg relaxation)
- Sim push tool: `eval_x2_mujoco.py` (`P` push, `[`/`]` magnitude, `--push-body`,
  `--push-force`, `--push-duration`)
- Related: `x2_real_robot_test_plan.md`, effort/motor findings in
  memory `project_x2_motor_datasheet_effort_fix`.
