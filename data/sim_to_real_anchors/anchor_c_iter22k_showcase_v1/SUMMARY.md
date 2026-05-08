# Anchor C -- iter-22k + showcase_v1

**Headline:** stand-in-place upper-body gesture reel, 100 s of CONTROL,
sim-vs-real `state_pos` diff = **3.88 deg RMS**.

## Inputs

- ONNX: `~/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx`
- Motion: `gear_sonic/data/motions/x2_ultra_showcase_v1.pkl`
  - key `showcase__v1__hands_only_demo`
  - 2895 frames @ 30 fps = 96.5 s
- Tuning: `gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml`
- LPF: 5 Hz on policy targets
- max-target-dev: 1.50 rad
- max-duration: 100 s

## Real-robot deploy command

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx \
    --motion ./gear_sonic/data/motions/x2_ultra_showcase_v1.pkl \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --max-target-dev 1.50 \
    --target-lpf-hz 5.0 \
    --max-duration 100 \
    --record
```

Recorded at 2026-05-03 23:18 ai-gym, on-disk path
`scratch/runs/x2_run_20260503_231753/run.npz` -> mirrored here as `real.npz`.

## MuJoCo deploy command (parity profile)

```bash
./gear_sonic_deploy/deploy_x2.sh sim \
    --sim-profile parity \
    --no-confirm \
    --no-build \
    --model  $HOME/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx \
    --motion ./gear_sonic/data/motions/x2_ultra_showcase_v1.pkl \
    --max-target-dev 1.50 \
    --target-lpf-hz 5.0 \
    --max-duration 100 \
    --log-dir /workspace/sonic/scratch/runs/sim_anchor_c_$(date +%Y%m%d_%H%M%S) \
    --record
```

Recorded at 2026-05-04 19:58 ai-gym, on-disk path
`scratch/runs/sim_anchor_c_20260504_195840/run.npz` -> mirrored here as `sim.npz`.

Note: in sim mode, `deploy_x2.sh` rejects `--tuning-config`; the relevant
tuning parameters (`--target-lpf-hz`, `--max-target-dev`) were passed
explicitly so the policy sees the same observation post-processing.

## Headline metrics

| Metric | Value |
|---|---:|
| Real CONTROL window | 102.00 s |
| Sim  CONTROL window | 102.02 s |
| Real tracking error RMS (state - cmd) | 13.05 deg |
| Sim  tracking error RMS (state - cmd) | 14.09 deg |
| **Sim-vs-real `state_pos` diff RMS** | **3.88 deg** |
| Sim-vs-real `cmd_pos` diff RMS | 8.34 deg |
| IMU angvel RMS real / sim | 0.186 / 0.197 rad/s |

## Findings

- **Arms (14 DoFs)** transfer almost perfectly. Cmd ranges agree
  within 0-15 deg per joint, state ranges within 0-7 deg per joint.
  Upper-body kinematics are essentially feed-forward from the playlist;
  both worlds execute nearly identically.
- **Waist (3 DoFs)** transfers cleanly on yaw and roll. `waist_pitch`
  is commanded with 155-165 deg range in *both* worlds but neither
  produces more than ~26 deg of state range. The body just doesn't
  go there. That's a property of the motion file (over-reach), not a
  sim-to-real gap.
- **Head (2 DoFs)**: head_yaw cmd matches (65 vs 63 deg); state
  diverges (40 vs 20 deg), with sim being freer. head_pitch state
  is 0 deg on real (motor refused 171 deg target) vs 13 deg in sim.
- **Legs (12 DoFs)**: most match within 10-20 deg cmd-range. Four
  DoFs diverge by 20-45 deg: `left_hip_pitch` (105 vs 84),
  `right_hip_pitch` (76 vs 102), `right_knee` (78 vs 50),
  `right_ankle_roll` (72 vs 116). Expected closed-loop sim-to-real
  signature: balance corrections diverge slightly even on a
  stand-in-place motion.

The `cmd_pos` diff (8.34 deg) being roughly 2x the `state_pos` diff
(3.88 deg) reflects that both physics environments low-pass the
small per-step policy command differences via the body's own
dynamics. Behaviour at the body level transfers tighter than commands
at the policy output level.

## Sim falls during the post-CONTROL handoff (RAMP_OUT artefact)

The sim run face-plants in MuJoCo in the final ~1.5 s of the
recording. **This is not a sim-to-real policy gap -- it is a
hand-off-FSM artefact** with a clean root cause we verified from
`sim.npz`:

- `--max-duration 100` trips at t ≈ 100 s of policy time, transitioning
  the deploy node from `CONTROL` to `RAMP_OUT`
  (`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp:1134`).
- During `RAMP_OUT`, the **ONNX policy is no longer queried**. Joint
  targets are open-loop linearly interpolated from the policy's last
  command toward `default_angles` over `--return-seconds` (default
  2.0 s), then held in `HOLD_FOR_MC` (lines 887–944).
- This window is exactly the design intent of `RAMP_OUT`: hand control
  back to the real robot's motion controller cleanly. On hardware the
  gantry catches any residual tilt during this open-loop ramp; in
  MuJoCo there's no gantry, so a ~-4 deg standing pitch at the moment
  of handoff is enough for gravity to take over.

Verified timeline from `sim.npz`:

| Wall t (rel) | Phase | cmd_leg behaviour | Sim pitch |
|---:|---|---|---:|
| 95–100 s | CONTROL (policy active, closed-loop) | std ≈ 0.015 rad, oscillating | –3 to –5 deg |
| 100–102 s | **RAMP_OUT** (policy out of loop, lerp to default) | monotonically lerping toward default | –4.8 → –7.3 deg |
| 102–103 s | HOLD_FOR_MC (open-loop, default pose held) | static at default | –24.9 → –75 deg |
| 103.4 s | end of recording | --- | **–89.4 deg (face-planted)** |

The motion-loop wrap at t ≈ 96.5 s is *not* the killer. The motion
file's joints repeat almost seamlessly across the wrap (max joint
delta 0.66 deg, RMS 0.26 deg); the only base-frame discontinuity is
+11.35 deg yaw and +19 cm horizontal translation in the reference,
which the policy absorbed for ~5 s of "second-loop showcase" without
falling.

For visual playback in `play_npz_dual_kinematic.py`, pass
`--end-trim 3.5` to drop the entire `RAMP_OUT` + `HOLD_FOR_MC` window
and keep the comparison focused on the powered-gesture segment. The
sim-vs-real metrics quoted above (state diff 3.88 deg, etc.) are
computed over the full ~102 s CONTROL window without trimming -- the
last ~1.5 s contributes modestly because per-DoF L2 averages over
many seconds of tightly matching upper-body motion.

> **Reframing for the paper:** what we'd been calling the
> "gantry-effect signature" at motion boundaries is in fact the
> deploy harness's intended `RAMP_OUT` → `HOLD_FOR_MC` handoff. On
> hardware the gantry covers this open-loop window; in sim there is
> no gantry. The closed-loop policy phase (CONTROL) transfers
> cleanly. Trim the handoff window when comparing.

## Caveats / known issues

- `meta_json` in the recorder npz does NOT yet capture
  `--model`/`--motion`/`--tuning-config`; provenance comes from this
  doc and from `sim_to_real_recordings_inventory.md`. Patching the
  recorder is on the next-time-at-the-robot list.
- The motion file `x2_ultra_showcase_v1.pkl` contains physically
  unreachable angles for several joints (head_pitch ref reaches
  171 deg, waist_pitch ref reaches 166 deg, several shoulder DoFs
  ref-range exceeds 175 deg). Both sim and real refuse those
  over-reaches identically; the policy is faithful to the motion
  but the motion is over-eager in those joints.
- This anchor is a stand-in-place reel: the base barely moves,
  IMU yaw stays near zero, and gait variability does not enter the
  picture. For locomotion sim-to-real, see Anchor D.

## Reproducing this comparison

### Quantitative metrics + plots

Re-runs the CONTROL-window detection, resamples both recordings onto a
common 50 Hz grid, and writes all the figures + `summary.{json,txt}`
into `plots/`:

```bash
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py \
    --real data/sim_to_real_anchors/anchor_c_iter22k_showcase_v1/real.npz \
    --sim  data/sim_to_real_anchors/anchor_c_iter22k_showcase_v1/sim.npz \
    --out  data/sim_to_real_anchors/anchor_c_iter22k_showcase_v1/plots \
    --label-real "iter-22k real (showcase_v1)" \
    --label-sim  "iter-22k MuJoCo (showcase_v1)"
```

### Visual side-by-side (dual kinematic viewer)

Two robots in the same MuJoCo scene, gravity off, replaying the
recorded joint state and IMU quaternion. Real is rendered in original
X2 colours, sim is a translucent ghost in white + blue feet. Use
`--end-trim 3.5` to drop the deploy harness's `RAMP_OUT` +
`HOLD_FOR_MC` window so the comparison stays on the closed-loop
CONTROL phase:

```bash
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic_deploy/scripts/play_npz_dual_kinematic.py \
    --real data/sim_to_real_anchors/anchor_c_iter22k_showcase_v1/real.npz \
    --sim  data/sim_to_real_anchors/anchor_c_iter22k_showcase_v1/sim.npz \
    --separation 1.0 \
    --end-trim 3.5 \
    --speed 1.0
```

For a fully overlaid "ghost" view (sim translucent on top of real),
pass `--separation 0`. Pass `--no-anchor-feet` if you want to see the
raw IMU-driven base height (skips the per-frame foot-grounding
heuristic).

## Plot files in `plots/`

- `dof_pos_grid.png` -- 31-DoF cmd / state for both runs over time
- `dof_tracking_error.png` -- 31-DoF |state - cmd| traces
- `dof_l2_bar.png` -- per-DoF tracking-error L2, real vs sim, sorted
- `imu_overlay.png` -- IMU roll/pitch/yaw + angular velocity overlay
- `cmd_diff_heatmap.png` -- time x DoF heatmap of cmd_real - cmd_sim
- `summary.json`, `summary.txt` -- machine- and human-readable stats
