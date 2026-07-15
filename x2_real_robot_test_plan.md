# X2 Real-Robot Test & Verification Plan

Consolidates the sim2real threads from the 2026-07-14 session. Goal: fix the
**forward fall** (robot stable in sim, falls forward on real) + the **forward
waist-pitch wobble** (small nudge; roll is rock-solid), and validate the deploy
+ training changes.

## What we're validating (this session's changes)
- **Sim effort fix**: `x2_ultra.py` waist p/r 48→36 (motor physical peak); ankle
  reverted to URDF 36/24 (enforced limit) pending the torque-clamp answer.
- **Deploy loose preset**: `walking_recovery_loose.yaml` — leg clamp 0.50→0.70,
  waist clamp 0.30→0.45, **waist LPF off** (`target_lpf_hz_waist=0`).
- **Per-axis waist damping**: `kd_scale_waist_pitch` (deploy knob, currently 1.0).
- **Wrist**: MuJoCo deploy twist (proprioception runaway); bypass vs raw.
- **New skills** trained in run 175030: forward-fall recovery (faint_stand_up),
  crawl (start/loop/stop), kneel (start/loop/stop).

## Prerequisites / blockers
- [ ] **PC2 reachable** (robogym wifi; `ssh run@<ip>`). *Currently OFFLINE.*
- [ ] Decide deploy policy: current `agibot_x2_sonic.onnx` (baseline) vs a fresh
      ONNX export from run 175030 (once trained further).
- [ ] Safety: spotter/harness for all push + recovery tests. Start conservative.

---

## Phase 0 — Sim gate (before the robot)
Verify the deploy candidate in sim first so we don't burn robot time on a bad policy.
- [ ] MuJoCo eval `eval_x2_mujoco.py --motions x2_all_dances_finetune.pkl` (+ recovery/crawl/kneel pkls): tracks without falling. `--wrist-ref` for clean run; `--no-wrist-fix` to judge wrist.
- [ ] IsaacLab feasibility eval: success rates ≥ baseline; no forgetting on the general yardstick.

## Phase 1 — Read-only hardware verification (no motion, zero risk)
| Item | How | Settles |
|---|---|---|
| Motor firmware torque/current limits (**ankle 36 vs 60**, all joints) | PC2 boot log `hal_ethercat_x2.log` + motor params (Thread A) | sim `effort_limit_sim` (ankle/waist true ceiling) |
| Joint angle limits vs official | our URDF vs `joint_name_and_limit` page | clamp/limit mismatch |
| CoM / inertia sim-vs-real | URDF mass props vs robot spec / measured CoM | forward-CoM gap (a forward-wobble suspect) |

## Phase 2 — Baseline characterization (current policy + `walking_recovery.yaml`)
Record numbers for later A/B. Tool: `x2_scan_mc_motors.sh --duration 30` + `motor_monitor`.
- [ ] **Idle stand** (60 s): rocking sway? drift? (freq / amplitude)
- [ ] **Static nudge** fwd / back / lateral: per-joint tracking err, effort, wobble freq/amp
- [ ] **Firm FORWARD push** → `ankle_pitch` + `waist_pitch` `effort_max_abs`
      - ankle_pitch **>36** ⇒ motor cap (sim→60); plateau **~36** ⇒ clamped at 36
- [ ] **SIDEWAYS push** → `ankle_roll` + `waist_roll` `effort_max_abs` (settles ankle-roll 24 vs 36)
- [ ] **Wrist** deploy behavior: twist / bypass working? time-to-saturation

## Phase 3 — Deploy-tuning experiments (ONE knob at a time; re-scan after each)
1. [ ] **Forward wobble**: `kd_scale_waist_pitch` ladder 1.0 → 1.3 → 1.6.
       Pass: pitch wobble damps, **roll unaffected**, forward recovery does NOT lock.
2. [ ] **Waist LPF off** (`walking_recovery_loose.yaml`): A/B vs the kd approach.
       Watch for re-exposed >25 Hz waist jitter → walk back toward 16 Hz if it appears.
3. [ ] **Loosened clamps** (leg 0.70, waist 0.45): bigger recovery stride, no OOD/instability.
4. [ ] **Sagittal kd stays MC-match** (hip 0.476 / knee 0.792 / ankle_pitch 3.31):
       verify forward recovery keeps torque headroom (no "lean forward and lock").

## Phase 4 — New-policy deploy (when run 175030 gives a good checkpoint)
- [ ] Export ONNX (encoder/decoder), deploy.
- [ ] Progressive presets: `conservative` → `expressive` → `walking_recovery` → `walking_recovery_loose`. Confirm stand/hold before advancing each.
- [ ] Wrist `--no-wrist-fix`: is the twist gone / much slower? (A/B time-to-saturation + peak |wrist action| vs the 3250 baseline)

## Phase 5 — Target-behavior verification ("is it fixed?")
- [ ] **Forward push recovery**: recovers from a forward chest-nudge; NO lean-forward-and-lock (the main blocker)
- [ ] **Forward waist wobble** gone / reduced on a small nudge
- [ ] **Lateral stability** still rock-solid
- [ ] **Fall recovery**: get up from a forward fall (faint_stand_up)
- [ ] **Crawl** start/loop/stop; **Kneel** start/loop/stop
- [ ] **Walking**: forward stride + turns
- [ ] **No new failures**: hip-yaw shimmy at heel-strike, 5–7 Hz ankle resonance under static nudge

## Open questions this plan resolves
- Ankle torque ceiling **36 vs 60** → Phase 1 (firmware limit) + Phase 2 (push measurement). Deploy is position-control w/ no torque clamp, so the cap = motor firmware limit.
- Forward wobble root cause: **PD damping** (Phase 3 kd) vs **CoM gap** (Phase 1) vs **LPF lag** (Phase 3).
- Whether waist 48→36 (training) + kd_scale_waist_pitch (deploy) together kill the forward instability.
