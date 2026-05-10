# Milestones

A chronological log of significant integration milestones for the X2
Ultra + GR00T N1.7 + SONIC stack. Each entry is a session-level wrap-up
with the run command, headline numbers, code changes, issues
encountered + fixes applied, and what's left for the next session.

```{toctree}
:maxdepth: 1

2026-05-02_first_iter4000_powered_run
2026-05-02_post_deploy_tuning
2026-05-03_first_iter22000_powered_walk
2026-05-08_live_vla_sonic_sim_v0
2026-05-10_omnihand_finger_tuning
2026-05-11_finger_tip_oppose_signal
2026-05-12_finger_signal_filter
2026-05-10_sonic_loop_v1_schema
```

## At a glance

| Date | Milestone | Surface | Outcome |
|------|-----------|---------|---------|
| 2026-05-02 | First iter-4000 powered run | Real X2 + SONIC | First-ever powered policy on hardware. Tilt/balance only — no walking. |
| 2026-05-02 | Post-deploy tuning | Real X2 + SONIC | KP/KD sweep, target-LPF and clamp tuning gauntlet. |
| 2026-05-03 | First iter-22000 powered walk | Real X2 + SONIC | First out-and-back walking cycle, 36.75 s wall-time, clean MC handoff. |
| 2026-05-08 | Live VLA → SONIC sim (v0) | **Sim-only**, X2 + SONIC + N1.7 | Closed-loop VLA → SONIC pipeline runs end to end. Visible motion is mode-collapsed; full triage is documented in the milestone page. |
| 2026-05-10 | OmniHand finger-tuning iteration | **Sim-only**, Quest 3 → X2 kinematic | Thumb-fingertip-touch gesture now closes correctly. Anchor expansion + 3-motor opposition fold-in. Topology mismatch on non-thumb tips filed for v1. |
| 2026-05-11 | Per-finger fingertip-to-thumb proximity (v0.5) | **Sim-only**, Quest 3 → X2 kinematic | New JS `computeFingerTipOppose` 4-vector + Python `max(curl, finger_tip_oppose)` drive on non-thumb pips + pip CLOSED anchor 80° → 88°. Wired end-to-end (JS → reader → retargeter → debug NPZ → replay); needs a fresh recording to verify visually. |
| 2026-05-12 | Finger-signal smoothing filter (v0.6) | **Sim-only**, Quest 3 → X2 kinematic | EMA(α=0.5) + rolling-median deadband-hold on the 10 per-side hand inputs. Calibrated against v5/ep1: held-pose tremor reduced 20–40 % on the worst fingers, +20 ms motion lag, 0 ms touch-onset lag. Live + record + replay paths all wired; debug NPZ persists raw + filtered for offline A/B. |
| 2026-05-10 | SONIC-loop v1 dataset schema | **Sim-only**, X2 + SONIC 25k + Quest 3 | Canonical training-target columns flip from operator-commanded to **post-SONIC executed** q. Operator command preserved as `_pre_sonic` siblings (debug-only, training-invisible). New `meta/dataset_format_version.json` marker, `inspect_sonic_correction.py` offline diagnostic, live `--sonic-correction-warn-rad` operator log. `record_x2_dataset.sh` is now the recommended path for VLA captures. |
