# X2 Manipulation Robustness — Payload / Rated-Capacity DR (plan, revisit later)

_Status: PLAN ONLY — not yet run. Companion to the arm-dynamics work
([x2_shadow_boxing_arm_dynamics_plan.md](x2_shadow_boxing_arm_dynamics_plan.md)):
that plan makes the arm move *fast*; this one makes it hold position *under load*.
Together = manipulation-ready arms._

## Problem

The policy is trained with effectively **empty hands**, so it is not robust to carrying
a real object. Deployed with a payload (dead weight in the hand), the loaded arm will
**sag / undershoot** because the policy never learned to compensate the extra gravity +
inertia torque.

**Key point: this is a domain-randomization / physics gap, NOT a reward gap.** "Dead
weight at the wrist" is a physical condition (added mass), not a behavior to reward. The
existing position-tracking rewards (`tracking_relative_body_pos`, `tracking_vr_5point_local`)
already reward holding the wrist at the reference — but they only teach load compensation
*if the load is present in sim*. DR is the enabler.

## What training does today (a weak proxy, measured on x2_ultra.xml)

There IS a `randomize_rigid_body_mass` event, but it is an actuator-robustness perturbation,
not a payload:

- Scales **`wrist_yaw` + `torso`** link masses by **[0.8, 2.5]×**, `operation: scale`.
- `wrist_yaw_link` nominal mass = **0.36 kg** → scaled range 0.29–0.90 kg → **≤ ~0.54 kg
  added at the wrist.**

Why that does NOT cover rated payload:
1. **Magnitude:** ≤0.54 kg vs the arm's rated carry (kg-scale — `TODO(spec)`: fill in X2
   rated per-arm payload). Nowhere near.
2. **Location / lever arm:** it scales the *wrist link's own COM mass*. A carried object
   sits at the **grasp point** — past `wrist_pitch`/`wrist_roll`, ~8–16 cm further out.
   That longer lever arm produces **much more elbow/shoulder torque** than scaling the
   wrist link. There is **no hand/palm/EE body** in the MJCF; payload should attach at
   `wrist_roll_link` (the arm tip).

Good news — **actuator torque limits are real** in the MJCF (`wrist_yaw` ±24 Nm, ankle
±24–36, hip/knee ±118 Nm). So a properly-added payload will **correctly reveal whether the
arm can physically hold rated weight** — no cheating. This is exactly what validates
"rated capacity."

## Proposed change — payload DR (no new reward)

Add an *absolute* payload mass at the arm tip, randomized 0 → rated capacity, at the correct
grasp offset. Config sketch (new event, alongside the existing mass scale):

```yaml
# manager_env.events
add_wrist_payload:
  func: isaaclab.envs.mdp:randomize_rigid_body_mass
  mode: reset                       # re-draw payload each episode
  params:
    asset_cfg: {name: robot, body_names: ".*wrist_roll.*"}   # arm tip, not wrist_yaw
    mass_distribution_params: [0.0, RATED_CAPACITY_KG]        # TODO(spec): fill in
    operation: add                                            # ABSOLUTE add, not scale
    # NOTE on lever arm: MuJoCo adds mass at the body COM. To model a payload held
    # ~8-16cm beyond the wrist tip, either (a) attach at a body whose COM is at the
    # grasp point, or (b) add a small massless "grasp" site/body at the palm offset
    # and put the payload there, so the elbow/shoulder torque matches reality.
```

Randomize per-arm independently (left/right) so it also learns asymmetric loads. Keep the
existing wrist-mass *scale* (actuator robustness) — they're complementary.

## Curriculum option

Payloads make everything harder, so consider a curriculum: start 0 kg, ramp the upper bound
toward rated capacity as tracking holds up (same spirit as the feasibility-gated curricula
elsewhere). Warm-start from a good general checkpoint.

## Evaluation

- **Tracking-under-load metric:** run a hold/reach clip with a fixed payload at the arm tip
  and measure wrist **position droop** (steady-state error vs reference) at
  {0, ¼, ½, ¾, rated} capacity. A robust policy keeps droop bounded up to rated; it should
  degrade gracefully (not collapse) beyond.
- **Torque headroom:** log `wrist`/`elbow`/`shoulder` actuator torque vs the MJCF limits
  under load — confirms whether rated capacity is within the arm's physical envelope or the
  arm hardware itself is the limiter.
- Extend `record_x2_eval_mujoco.py` to (a) attach a payload mass at the arm tip and
  (b) log wrist position error + arm torques, analogous to the `--traj-csv` deploy metric.
- Guardrail: no regression on general feasibility (512 sweep) or locomotion — same
  two-instrument discipline as the rebalance run.

## Open questions / TODO

- `TODO(spec)`: X2 rated per-arm payload capacity (sets the mass range).
- Grasp offset: exact palm/EE point past `wrist_roll_link` for a realistic lever arm.
- Does IsaacLab support per-arm independent payload draws in one event, or do we need two
  events (left/right)?
- Interaction with the arm-dynamics fix: does holding-under-load fight fast-arm rewards?
  Likely fine (different clips), but watch for the policy over-stiffening the arm.
- Actuator model fidelity: are the sim `actuatorfrcrange` values the *real* motor limits?
  If optimistic, the policy may learn to hold loads the hardware can't.
