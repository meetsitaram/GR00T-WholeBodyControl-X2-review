# Anchor D -- iter-22k + walk_demo_v6

**Headline:** turn-walk-turn-walk-return locomotion, 47 s of CONTROL,
sim-vs-real `state_pos` diff = **5.03 deg RMS**, net yaw drift 0.1 deg
(real) vs 2.9 deg (sim) -- both close the loop and end near the
starting heading.

## Inputs

- ONNX: `~/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx`
- Motion: `gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl`
  - key `walk_demo__v6__home_loop`
  - 1325 frames @ 30 fps = 44.17 s
  - choreography: anchor -> pivot left 90 deg -> walk 3 steps -> pivot
    right 90 deg -> pivot right 90 deg (180 deg total turnaround) ->
    walk 3 steps -> pivot left 90 deg -> anchor
  - source playlist: `gear_sonic/data/motions/playlists/walk_demo_v6.yaml`
- Tuning: `gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml`
- LPF: 5 Hz on policy targets
- max-target-dev: 1.50 rad
- max-duration: 50 s

## Real-robot deploy command

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx \
    --motion ./gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --max-target-dev 1.50 \
    --target-lpf-hz 5.0 \
    --max-duration 50 \
    --record
```

Recorded at 2026-05-04 00:41 ai-gym, on-disk path
`scratch/runs/x2_run_20260504_004150/run.npz` -> mirrored here as `real.npz`.

## MuJoCo deploy command (parity profile)

```bash
./gear_sonic_deploy/deploy_x2.sh sim \
    --sim-profile parity \
    --no-confirm \
    --no-build \
    --model  $HOME/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx \
    --motion ./gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl \
    --max-target-dev 1.50 \
    --target-lpf-hz 5.0 \
    --max-duration 50 \
    --log-dir /workspace/sonic/scratch/runs/sim_anchor_d_$(date +%Y%m%d_%H%M%S) \
    --record
```

Recorded at 2026-05-04 20:09 ai-gym, on-disk path
`scratch/runs/sim_anchor_d_20260504_200943/run.npz` -> mirrored here as `sim.npz`.

## Headline metrics

| Metric | Real | Sim |
|---|---:|---:|
| CONTROL window | 48.99 s | 52.02 s |
| IMU yaw range | 264.1 deg | 253.0 deg |
| IMU net yaw drift | -0.1 deg | +2.9 deg |
| Tracking error RMS (state - cmd) | 13.12 deg | 12.95 deg |
| IMU angvel RMS | 0.447 rad/s | 0.436 rad/s |

| Sim-vs-real diff | RMS |
|---|---:|
| `state_pos` real - sim | **5.03 deg** |
| `cmd_pos`   real - sim | 11.51 deg |

After skipping the first 1.0 s of CONTROL (which captures the
initial-pose mismatch between real STAND_DEFAULT and MuJoCo init):

| Metric | Value |
|---|---:|
| Compared duration | 48.0 s |
| Whole-body state-diff RMS | **4.70 deg** |
| Max single-frame whole-body diff | 23.5 deg |

## Findings

- **The walk transfers cleanly.** Both worlds traverse the entire
  turn-walk-turn-walk-return choreography (264 vs 253 deg of total
  yaw range) and both close the loop, returning within 3 deg of
  starting heading. This is the strongest sim-to-real result in the
  archive for locomotion.
- **Arms / waist agree tightly across all DoFs**, mirroring the
  Anchor C pattern: feed-forward upper-body transfers nearly
  perfectly. cmd ranges agree within 0-20 deg, state ranges within
  0-15 deg.
- **Closed-loop balance signature shows up on the legs.** Knees and
  ankle-roll diverge most: real `left_knee` cmd 43 deg vs sim 73
  deg, real `right_knee` cmd 80 deg vs sim 42 deg -- *opposite
  directions* on the two legs, exactly what you expect from the
  policy reacting to slightly different IMU/state observations and
  emitting different per-leg balance corrections.

## "Wobbly steps" -- per-frame divergence localised

The user reported "one or two out-of-balance steps" qualitatively.
Per-frame whole-body state-diff (`real - sim` resampled @ 50 Hz)
shows that the run sits at a 3-6 deg baseline most of the time, with
a few isolated knee/ankle spikes:

| Time (s) | Worst DoF | Diff (deg) | Interpretation |
|---:|---|---:|---|
| 13.3 | left_knee | -24.5 | step in first walk segment |
| 18.5 | right_knee | +34.6 | step during/around right-pivot transition |
| 33.6 | right_ankle_pitch | -24.2 | ankle correction during return walk |

Wrist-yaw spikes near t=48 s are end-of-run handoff transients
(`HOLD_FOR_MC` window approaching), not gait events.

The IMU-yaw overlay in
`plots/real_vs_sim_diff_with_yaw.png` lets us read which choreography
phase each spike falls in. Spikes at t=13.3 / 18.5 / 33.6 s are during
the walk segments, exactly where the qualitative wobbles were
observed; baseline elsewhere is tight (3-6 deg per-DoF state diff).

## Caveats / known issues

- `meta_json` in the recorder npz does NOT yet capture
  `--model`/`--motion`/`--tuning-config`. Provenance comes from this
  doc and from `sim_to_real_recordings_inventory.md`.
- Sim run captured 52.02 s of CONTROL vs real's 48.99 s; the comparator
  trims to the shorter window before computing RMS.
- Real run is gantry-tethered (one-way rail). Sim has no such
  constraint, but both runs end at near-zero net yaw drift because
  the choreography is symmetric (turn-walk-counterturn-walk-back) by
  design.

## Reproducing this comparison

### Quantitative metrics + plots

Re-runs the CONTROL-window detection, resamples both recordings onto a
common 50 Hz grid, and writes all the figures + `summary.{json,txt}`
into `plots/`:

```bash
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py \
    --real data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/real.npz \
    --sim  data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/sim.npz \
    --out  data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/plots \
    --label-real "iter-22k real (walk_demo_v6)" \
    --label-sim  "iter-22k MuJoCo (walk_demo_v6)"
```

### Visual side-by-side (dual kinematic viewer)

Two robots in the same MuJoCo scene, gravity off, replaying the
recorded joint state and IMU quaternion. Real is rendered in original
X2 colours, sim is a translucent ghost in white + blue feet. No
end-trim is needed for this anchor -- `walk_demo_v6` ends in a stable
home-loop pose, so the deploy harness's `RAMP_OUT` window is uneventful
in both worlds:

```bash
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic_deploy/scripts/play_npz_dual_kinematic.py \
    --real data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/real.npz \
    --sim  data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/sim.npz \
    --separation 1.0 \
    --speed 1.0
```

For a fully overlaid "ghost" view (sim translucent on top of real),
pass `--separation 0`. The home-loop choreography means both robots
return to within ~3 deg of starting heading, so the overlay stays
clean throughout the run.

## Plot files in `plots/`

- `dof_pos_grid.png` -- 31-DoF cmd / state for both runs over time
- `dof_tracking_error.png` -- 31-DoF |state - cmd| traces
- `dof_l2_bar.png` -- per-DoF tracking-error L2, real vs sim, sorted
- `imu_overlay.png` -- IMU roll/pitch/yaw + angular velocity overlay
- `cmd_diff_heatmap.png` -- time x DoF heatmap of cmd_real - cmd_sim
- `real_vs_sim_diff_with_yaw.png` -- per-frame whole-body diff with
  IMU yaw overlay; spikes annotated with time + worst-DoF
- `real_vs_sim_state_diff_heatmap.png` -- time x DoF heatmap of
  |state_real - state_sim|
- `summary.json`, `summary.txt` -- machine- and human-readable stats
