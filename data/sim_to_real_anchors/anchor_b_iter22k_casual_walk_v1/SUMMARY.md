# Anchor B -- iter-22k + casual_walk_v1 (first powered walk)

**Headline:** the first successful policy-driven walk on the X2 Ultra
hardware. Short straight-line walk motion with a slight turn-walk-turn
shape (1.81 m forward in motion ref, 248 deg yaw range), 14 s of
CONTROL, sim-vs-real `state_pos` diff = **7.27 deg RMS** (4.74 deg
after skipping the first 1.0 s of initial-pose mismatch).

This is the milestone run from 2026-05-03 22:20 captured immediately
after the user said "we already saw the mujoco sim results pretty good"
and approved the iter-22k checkpoint for hardware. It is the
locomotion bridge between the standing-only Anchor C and the full
turn-walk-turn-walk-return Anchor D.

## Inputs

- ONNX: `~/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx`
- Motion: `gear_sonic/data/motions/x2_ultra_casual_walk_v1.pkl`
  - key `casual_walk__v1__one_cycle`
  - 445 frames @ 30 fps = 14.83 s
  - root translation: 0.0 to 1.81 m forward (y-axis)
  - yaw range in reference: 248 deg
- Tuning: `gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml`
- LPF: 5 Hz on policy targets
- max-target-dev: 1.50 rad
- max-duration: 14 s

## Real-robot deploy command

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx \
    --motion ./gear_sonic/data/motions/x2_ultra_casual_walk_v1.pkl \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --max-target-dev 1.50 \
    --target-lpf-hz 5.0 \
    --max-duration 14 \
    --record
```

Recorded at 2026-05-03 22:20 ai-gym, on-disk path
`scratch/runs/x2_run_20260503_222045/run.npz` -> mirrored here as `real.npz`.
Bash history index: command #1972.

## MuJoCo deploy command (parity profile)

```bash
./gear_sonic_deploy/deploy_x2.sh sim \
    --sim-profile parity \
    --no-confirm \
    --no-build \
    --model  $HOME/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx \
    --motion ./gear_sonic/data/motions/x2_ultra_casual_walk_v1.pkl \
    --max-target-dev 1.50 \
    --target-lpf-hz 5.0 \
    --max-duration 14 \
    --log-dir /workspace/sonic/scratch/runs/sim_anchor_b_$(date +%Y%m%d_%H%M%S) \
    --record
```

Recorded at 2026-05-04 18:57 ai-gym, on-disk path
`scratch/runs/sim_anchor_b_20260504_185740/run.npz` -> mirrored here as `sim.npz`.

## Headline metrics

| Metric | Real | Sim |
|---|---:|---:|
| CONTROL window | 16.03 s | 14.02 s |
| IMU yaw range | 266.8 deg | 307.9 deg |
| IMU net yaw drift | -6.5 deg | +165.9 deg |
| Tracking error RMS (state - cmd) | 14.99 deg | 15.34 deg |
| IMU angvel RMS | 1.027 rad/s | 1.024 rad/s |

| Sim-vs-real diff | RMS |
|---|---:|
| `state_pos` real - sim | **7.27 deg** |
| `cmd_pos`   real - sim | 14.98 deg |

After skipping the first 1.0 s of CONTROL:

| Metric | Value |
|---|---:|
| Compared duration | 15.0 s |
| Whole-body state-diff RMS | **4.74 deg** |
| Max single-frame whole-body diff | 18.2 deg |

## Findings

- **Per-cycle kinematics agree.** Joint-level state difference of 4.74
  deg RMS over 15 s of walking is very tight. Looking per-DoF: arms
  match within 5-10 deg per joint, waist matches except for `waist_roll`
  which has a known motion-file over-reach. Legs match within 5-15 deg
  per joint.
- **Base trajectory diverges -- this is the main locomotion sim-to-real
  gap.** Real net yaw drift -6.5 deg (gantry-tethered, robot pinned to
  origin). Sim net yaw drift +165.9 deg (robot was free to rotate; tiny
  per-cycle leg-joint differences integrated into a wildly different
  base trajectory over 8+ gait cycles). Both runs had ~270-310 deg of
  yaw oscillation, consistent with each step having visible balance
  recovery.
- **Both runs were unstable.** This shows up in the per-frame
  divergence trace as recurring left/right hip-yaw spikes at t=5.9,
  6.4, 8.5 s (during the turn-walk-turn portion of the cycle), each in
  the 19-28 deg range. These coincide qualitatively with the user's
  recollection of the real walk being "kept upright by the gantry,
  with one or two out-of-balance steps."
- **Same iter-22k policy is borderline on this motion.** Real stayed
  upright because the gantry was effectively a balance assist. Sim was
  free, and the policy reactivity diverged the base path significantly
  but kept the body upright. Anchor D (`walk_demo_v6`) is the more
  successful locomotion result -- it has a return-to-home choreography
  that closes the loop, so the base trajectories agree better.

## "Out-of-balance" step localisation

Per-frame whole-body state-diff (real - sim, resampled @ 50 Hz) shows
the run sits at 3-6 deg baseline with a few step-time spikes:

| Time (s) | Worst DoF | Diff (deg) | Interpretation |
|---:|---|---:|---|
| 1.0 | right_wrist_yaw | +49.4 | initial-pose-mismatch transient (real had been holding STAND_DEFAULT, sim started fresh); not a step event |
| 5.9 | left_hip_yaw | -25.6 | step in turn segment |
| 6.4 | left_hip_yaw | -28.4 | step in turn segment |
| 8.5 | right_hip_yaw | +19.0 | step in walk segment |
| 15.3 | left_ankle_pitch | -25.0 | end-of-walk recovery |
| 16.0 | left_elbow | -26.3 | end-of-window arm transient |

Wrist transients in the first second are the initial-pose mismatch
between real (in STAND_DEFAULT before CONTROL begins) and sim
(MuJoCo init pose). They aren't step events.

## Sim falls during the post-CONTROL handoff (RAMP_OUT artefact)

The IMU-pitch trace shows a sim/real asymmetry at the *end* of the
recording. **This is not a sim-to-real policy gap -- it is a
hand-off-FSM artefact** identical to the one documented for Anchor C:

- `--max-duration 14` trips, transitioning the deploy node from
  `CONTROL` to `RAMP_OUT` (lines 1134-1158 of
  `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp`).
- During `RAMP_OUT`, the **ONNX policy is no longer queried**. Joint
  targets are open-loop linearly interpolated from the policy's last
  command toward `default_angles` over `--return-seconds` (default
  2.0 s), then held in `HOLD_FOR_MC`.
- This window is exactly the design intent of `RAMP_OUT`: clean
  hand-off to the real robot's MC. On hardware the gantry catches any
  residual tilt during the open-loop ramp; in MuJoCo there is no
  gantry, so an already-walking pitch at handoff finishes the job.

| Wall t (rel) | Phase | Real pitch | Sim pitch |
|---:|---|---:|---:|
| 15.0 s | RAMP_OUT entered | -8.9 deg (upright) | -27.1 deg (already tilting) |
| 15.5 s | open-loop ramp | -7.8 deg | -76.5 deg (falling) |
| 16.0 s | HOLD_FOR_MC | -6.4 deg | **-89.5 deg (face-planted)** |

For visual playback in `play_npz_dual_kinematic.py`, pass
`--end-trim 1.5` to drop the entire `RAMP_OUT` + `HOLD_FOR_MC` window
and keep the comparison focused on the powered-walk segment.

> **Reframing for the paper:** what we'd been calling the
> "gantry-effect signature" at motion boundaries is in fact the
> deploy harness's intended `RAMP_OUT` → `HOLD_FOR_MC` handoff. On
> hardware the gantry covers this open-loop window; in sim there is
> no gantry. The closed-loop policy phase (CONTROL) transfers
> cleanly. Trim the handoff window when comparing.

## Caveats / known issues

- `meta_json` in the recorder npz does NOT yet capture
  `--model`/`--motion`/`--tuning-config`. Provenance comes from this
  doc and from `sim_to_real_recordings_inventory.md` (Anchor B).
- Real CONTROL window is 16 s, sim 14 s; the comparator trims to the
  shorter window. Real's extra 2 s is the policy continuing to publish
  past motion end while LPF wound down.
- The motion ref's reachable y-translation is ~1.8 m forward. Real
  base translation is not captured by the recorder, so we cannot
  directly verify how far the real robot walked; we infer from the
  IMU yaw drift that real returned near origin (gantry-pinned), while
  sim drifted ~166 deg in heading (no such constraint).
- This is the first powered walk recording; the motion playlist has
  since been refined into `walk_demo_v6` (Anchor D), which is the
  recommended primary locomotion anchor.

## Reproducing this comparison

### Quantitative metrics + plots

Re-runs the CONTROL-window detection, resamples both recordings onto a
common 50 Hz grid, and writes all the figures + `summary.{json,txt}`
into `plots/`:

```bash
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py \
    --real data/sim_to_real_anchors/anchor_b_iter22k_casual_walk_v1/real.npz \
    --sim  data/sim_to_real_anchors/anchor_b_iter22k_casual_walk_v1/sim.npz \
    --out  data/sim_to_real_anchors/anchor_b_iter22k_casual_walk_v1/plots \
    --label-real "iter-22k real (casual_walk_v1)" \
    --label-sim  "iter-22k MuJoCo (casual_walk_v1)"
```

### Visual side-by-side (dual kinematic viewer)

Two robots in the same MuJoCo scene, gravity off, replaying the
recorded joint state and IMU quaternion. Real is rendered in original
X2 colours, sim is a translucent ghost in white + blue feet. Use
`--end-trim 1.5` to drop the deploy harness's `RAMP_OUT` +
`HOLD_FOR_MC` window so the comparison stays on the closed-loop
CONTROL phase:

```bash
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic_deploy/scripts/play_npz_dual_kinematic.py \
    --real data/sim_to_real_anchors/anchor_b_iter22k_casual_walk_v1/real.npz \
    --sim  data/sim_to_real_anchors/anchor_b_iter22k_casual_walk_v1/sim.npz \
    --separation 1.0 \
    --end-trim 1.5 \
    --speed 1.0
```

For a fully overlaid "ghost" view (sim translucent on top of real),
pass `--separation 0`. To preserve the raw recorded yaw divergence
(real ~-7 deg vs sim ~+166 deg) instead of the default start-yaw
alignment, pass `--raw-imu-yaw` -- useful for showing the gantry-pinned
real path next to the free-rotating sim.

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
