# X2 Shadow-Boxing / Arm-Dynamics Fine-Tune — Plan (revisit later)

_Status: PLAN ONLY — not yet run. Parked while the slow-walk rebalance run trains.
Same playbook as the slow-walk dead-band fix
(see [x2_slow_walk_deadband_finetune.md](x2_slow_walk_deadband_finetune.md))._

## Problem

Dynamic upper-body motions (shadow boxing) track very poorly in X2 deploy. Observed
(2k model, `play_locomotion` in MuJoCo vs kinematic reference):

- **Reference (raw retarget):** bent-knee stance, good balance, **fast arm movements**.
- **Executed-feasible reference (what G1-SONIC actually did, in the corpus):** only
  **slightly** watered down — leg stance still good, arms only a touch damped. So the
  motion **is achievable and IS in the training corpus.**
- **X2 2k deploy:** **completely damped** — barely any leg stance, arm movement almost
  absent. Does not resemble the motion.

**Conclusion: this is a genuine X2 tracking/reward gap, NOT a data ceiling.** The target
exists in the corpus; the policy fails to reproduce it.

## Root cause (why arms come out completely damped)

1. **The arm reward is position-only.** `tracking_vr_5point_local`
   (`tracking_local_vr_5point_error`, rewards.py:336) tracks the reward-point body
   POSITIONS (torso + both wrists) relative to the anchor — no velocity. Weight 2.0
   (highest). A damped arm drifting slowly toward the target position still scores well,
   so it does not force fast punches. **Position tracking is satisfiable by a damped
   mean pose** (minimize average error by sitting mid-range instead of overshooting on
   fast extremes).
2. **Smoothness penalties actively suppress fast arm motion** — `action_rate_l2` (−0.1)
   and `anti_shake_ang_vel` (−0.02) penalize exactly the rapid actuation a punch needs.
   Prime suspect for "completely" damped.
3. **Arm-dynamic clips are ~0.06 % of the corpus** (23 / 35,974). The policy optimizes
   the common case (posture/locomotion) and treats fast arms as noise to damp.

The generic `tracking_body_linvel`/`body_angvel` DO cover the arm bodies, but (a) they
average over all 14 bodies so the arm is diluted, and (b) std 1.0 is too wide to penalize
arm-velocity lag.

## Proposed fix — focused combat fine-tune (warm-start from 2k)

Mirror the slow-walk approach. Levers, in priority order:

| lever | change | rationale |
|---|---|---|
| arm smoothness penalties | reduce / **exempt arm joints** from `action_rate_l2` & `anti_shake_ang_vel` | biggest single lever for un-damping; stop punishing fast actuation |
| arm-velocity reward | **arm-weighted** `tracking_body_angvel`/`linvel` (emphasize wrist/elbow) + **sharper std** (e.g. angvel/linvel ~0.25) | reward matching the fast arm VELOCITY, not just position |
| `tracking_vr_5point_local` | optionally sharpen std (not raise weight) | tighter arm POSITION precision — secondary, won't drive speed alone |
| data | focused corpus of combat/arm clips (23 shadow-boxing + ROM_Box) oversampled | rare-in-corpus regime, under-trained; focused fine-tune amplifies it |
| warm-start | from executed-feasible **2k** (best general) | keep general competence, add the arm skill |

Note: the slow-walk **rebalance** run already sharpened `tracking_body_linvel` std
1.0→0.25 (covers arm bodies too) — **re-check shadow boxing on the rebalance checkpoint
first**, it may partially improve arms as a side effect before we add arm-specific terms.

## Caveat to rule out first (deploy-side, no retrain)

Verify X2 **arm KP/KD** physically allow fast movement. If arm gains are too low / damping
too high, no reward change fixes it. Since it's *completely* damped (not "tries and falls
short"), reward suppression is the likelier cause — but check the actuator gains before
spending a training run.

## Clips & commands (for eval)

Executed-feasible clips extracted to `gear_sonic/data/motions/shadow_boxing_executed/`
(23 clips @ 50 fps): `shadow_boxing_R_00{1,2,3}__A{359..362}[_M].pkl`,
`ROM_Box_..._002__A5{20,23}[_M].pkl`. Raw retargets:
`gear_sonic/data/motions/demo_v1_sources/combat_chain_matched/shadow_boxing_R_*__x2_chain_matched.pkl`.

Three-way compare on the SAME clip:
```bash
# 1. kinematic reference (executed-feasible target)
export MUJOCO_GL=glfw DISPLAY=:1
python gear_sonic/scripts/play_x2_motion_mujoco.py \
  --motion gear_sonic/data/motions/shadow_boxing_executed/shadow_boxing_R_003__A359.pkl

# 2. deploy (Terminal 1: stack w/ model; Terminal 2: feed clip)
gear_sonic/scripts/run_x2_pkl_direct_stack.sh --model <checkpoint_g1.onnx>
python -m gear_sonic.scripts.play_locomotion \
  --pkl gear_sonic/data/motions/shadow_boxing_executed/shadow_boxing_R_003__A359.pkl
```

## Success criteria

- Deploy arm reaches near the reference extension **at roughly the reference speed**
  (not damped to mid-range), and the bent-knee stance appears.
- No regression on general feasibility (512 sweep ≥ ~92) or the slow-walk understep —
  same two-instrument guardrail as the rebalance.
- Quantify with an **arm end-effector velocity / extension** metric (extend
  `record_x2_eval_mujoco.py --traj-csv` to log wrist positions + speeds; understep-style
  ratio of arm-travel or peak wrist speed vs reference).

## Open questions

- Is per-joint penalty exemption supported cleanly, or do we need a new arm-specific
  reward term / body-weighting in the tracking funcs?
- Does sharpening `body_angvel` globally hurt smooth/slow motions (over-twitchy)? May
  need arm-weighting rather than a global std change.
- Actuator: are X2 arm KP/KD the limiter? (check first).
