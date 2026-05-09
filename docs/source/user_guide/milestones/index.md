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
```

## At a glance

| Date | Milestone | Surface | Outcome |
|------|-----------|---------|---------|
| 2026-05-02 | First iter-4000 powered run | Real X2 + SONIC | First-ever powered policy on hardware. Tilt/balance only — no walking. |
| 2026-05-02 | Post-deploy tuning | Real X2 + SONIC | KP/KD sweep, target-LPF and clamp tuning gauntlet. |
| 2026-05-03 | First iter-22000 powered walk | Real X2 + SONIC | First out-and-back walking cycle, 36.75 s wall-time, clean MC handoff. |
| 2026-05-08 | Live VLA → SONIC sim (v0) | **Sim-only**, X2 + SONIC + N1.7 | Closed-loop VLA → SONIC pipeline runs end to end. Visible motion is mode-collapsed; full triage is documented in the milestone page. |
