# X2 Real-Robot Push-Test Results

Append-only log of push/recovery tests. Compare future runs (new policy, tuning
tweaks) against these baselines. Companion to `x2_real_robot_test_plan.md`.

---

## Run 1 — 2026-07-14 · loose preset · arm-dynamics model

**Config**
- Model: `agibot_x2_sonic.onnx` = **arm_dynamics_v3_1500** (NOT a dance/recovery model)
- Tuning: **`walking_recovery_loose.yaml`** (leg clamp 0.70, waist clamp 0.45, **waist LPF off**; action_clip 20, tilt_cos -0.3, sagittal kd at MC-match)
- `--wrist-bypass ik`, `--lock-head-straight`, `--max-target-dev-head 0.01`
- Instrument: `motor_monitor` JSONL @ **1 Hz** (group effort + top per-joint tracking err). NOTE: 1 Hz cannot resolve the ~1-2 Hz trailing oscillation — use `x2_scan_mc_motors` (~50 Hz) for that.

**Idle stand (baseline)**
- leg eff ~7-15 N·m (varies w/ stance), vel ~0.01 rad/s; waist eff ~2-3, trk ~0.05-0.08; arm ~2.7
- steady offsets (not diverging): shoulder_roll ~0.20 rad, ankle_roll ~0.17 rad

**Pushes — ALL directions recovered, no fall.**
- Correct strategy by direction: fwd/back → ankle_pitch + waist_pitch (+ hip on hard); left/right → ankle_roll + hip_roll.
- Recovery is **decisive** (real hip-strategy steps, leg vel up to ~7-8 rad/s) — NOT the tiny-shuffle seen in MuJoCo sim. Loose preset's wider leg clamp (0.70) likely enables the bigger stride.
- Peak effort observed: **leg 79.7 N·m (66% of 120)**, **waist 21.9 N·m (61% of 36)** — hard pushes near the envelope, still caught, lots of torque headroom.

**Chest vs pelvis forward push (clean moment-arm A/B):**
| | Chest push | Pelvis push |
|---|---|---|
| leg eff / vel | 79.7 N·m / 7.16 rad/s | 76.1 N·m / 7.89 rad/s |
| **waist_pitch trk** | **0.70** (railed ±0.31 range) | **0.00** (none) |
| hip_pitch trk | 0.53 | **0.82** (takes over) |
| ankle_pitch trk | 0.76 | 0.77 |
- Confirms: policy recruits the **waist only when the disturbance creates a waist moment** (high/chest push); a pelvis (low) push engages hip+ankle instead. => waist is NOT over-relied; recruitment is geometry-correct. Waist is range-limited (±18°) so it rails on hard chest pushes but stays within torque budget.

**Trailing oscillation (operator felt back-and-forth after pushes):**
- Visible as decaying effort bounce post-push (e.g. after pelvis push: waist eff 6.2→1.5→5.3→0.6→6.2→0.8→8.7; ankle_pitch trk 0.34→0.13→0.22→0.15→0.30 over ~4 s).
- Mild **under-damped sagittal (waist/ankle) ring**, decaying. Frequency NOT resolved at 1 Hz.
- Candidate fixes (only if it becomes a problem): small ankle/waist sagittal kd bump (careful — over-damping caused the historical "lean-forward-and-lock"), OR restore waist LPF toward 16 (the loose preset removed it).

**Verdict:** loose preset + arm-dynamics model = stable stand, recovers pushes in all directions with decisive motion, geometry-correct joint recruitment, well within torque limits. Mild trailing ring is the only blemish (open to characterize @ 50 Hz).

**Open follow-ups:** (1) `x2_scan_mc_motors` @ 50 Hz to resolve the trailing ring; (2) push harder to find the fall threshold; (3) re-run with a dance/recovery-trained model when available.
