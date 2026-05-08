# Sim-to-Real Anchor Archive

This folder contains the matched real-robot and MuJoCo recordings used as
sim-to-real anchors for the paper. Each anchor is a pair of `run.npz`
files produced by `deploy_x2.sh --record`:

- `real.npz` -- recorded on the X2 Ultra hardware (`mode=local`)
- `sim.npz`  -- recorded in the same iter-22k MuJoCo bridge (`mode=sim --sim-profile parity`)

Both runs use the same ONNX checkpoint, the same motion playlist/PKL, the
same `--target-lpf-hz 5.0`, and the same `--max-target-dev 1.50`. The only
difference between them is the physics environment: real robot vs MuJoCo.

## Anchors

| Anchor | Motion | Iter | Real duration | Sim duration | state-diff RMS | Headline |
|---|---|---|---:|---:|---:|---|
| B | `x2_ultra_casual_walk_v1.pkl` (short straight-walk-turn, 14.8 s)    | 22000 | 16.03 s  | 14.02 s  | **7.27 deg** (4.74 after t>1s) | **first powered walk on hardware**; base trajectories diverge by 166 deg yaw because real was gantry-pinned and sim was free. Trim the post-CONTROL handoff with `--end-trim 1.5` in the viewer. |
| C | `x2_ultra_showcase_v1.pkl` (stand-in-place upper-body reel, 96.5 s) | 22000 | 102.00 s | 102.02 s | **3.88 deg** | tightest sim-to-real of the three; upper-body kinematics transfer near-perfectly. Trim the post-CONTROL handoff with `--end-trim 3.5` in the viewer. |
| D | `x2_ultra_walk_demo_v6.pkl` (turn-walk-turn-walk-return, 44.2 s)    | 22000 | 48.99 s  | 52.02 s  | **5.03 deg** | strongest locomotion result; both worlds traverse the home-loop choreography, returning within 3 deg of starting heading. No end-trim needed. |

**Post-CONTROL handoff caveat (Anchors B & C):** both runs end with
the sim robot face-planting in MuJoCo (pitch –85 deg) while real
stays upright (pitch –2 deg). This is **not** a sim-to-real policy
gap — it is the deploy harness's intended `CONTROL → RAMP_OUT →
HOLD_FOR_MC` finite-state-machine handoff. After `--max-duration`
trips, the ONNX policy is taken out of the loop and joint targets
are open-loop linearly interpolated to `default_angles` over
`--return-seconds` (default 2.0 s) before being held static for the
real motion controller to take back over. On hardware the gantry
catches any residual tilt during this open-loop window; in MuJoCo
there is no gantry. **Trim the handoff window with `--end-trim`
when comparing.** Anchor D doesn't show this because
`walk_demo_v6` ends in a stable home-loop pose, so the open-loop
ramp is uneventful in both worlds. See the per-anchor `SUMMARY.md`
for the verified phase timeline.

All anchors use `iter-22000` ONNX (`h200-iter-22000-sphere-feet-20260501`).

## Digital-twin fidelity: what these anchors confirm and what they don't

The anchors are matched closed-loop runs (same policy, same motion command,
gantry-tethered base on the real side). They give us **indirect** evidence
about specific parts of the URDF/MJCF digital twin, and there are several
modeling components they cannot exercise.

### Confirmed (with confidence)

| Model component | Evidence | Confidence |
|---|---|---|
| Kinematic chain (link lengths, joint axes, parent/child orientations) | Per-DoF joint state agrees within **3.88-5.03 deg RMS** over 30+ s of motion across all anchors | High -- large kinematic errors would compound visibly over this duration |
| Torso mass distribution / inertia | IMU angular-velocity RMS matches within ~5%: `0.186/0.197` (C), `0.447/0.436` (D), `1.027/1.024` rad/s (B) | Medium-high -- bounded to the gantry-tethered base configuration |
| Joint PD response under reference tracking | Tracking error (state - cmd) RMS within ~1 deg between sim and real at the same kp/kd: `13.05/14.09` (C), `13.12/12.95` (D), `14.99/15.34` (B) | Medium -- mostly tells us actuator response time-constants are similar at the gains we tested |
| Joint-limit / saturation behaviour | "Physically impossible" motion targets (head_pitch 171 deg, waist_pitch 166 deg, several shoulder DoFs > 175 deg) get refused identically by sim and real | Medium -- agreement could partly be coincidence; both saturate but for potentially different reasons |
| Closed-loop choreography phasing | Both worlds traverse the same playlist phases at the same wall-clock times (Anchor D: both close the loop within 3 deg of starting heading) | High for repeatable in-distribution motions |
| Deploy-harness bit-exactness | Same ONNX, same observation builder, same LPF + max-target-dev clamp produce indistinguishable command streams modulo the physics environment | High |

### Not yet verified (modeling components these anchors can't discriminate)

| Model component | Why these anchors miss it |
|---|---|
| **Foot contact geometry** (sphere-feet trained vs flat-soled hardware vs whatever the deploy MJCF uses) | Anchor C is stand-in-place; Anchors B/D have the gantry pinning the base and absorbing lateral contact forces |
| **Floor friction coefficient** | Same -- gantry takes lateral forces |
| **Actuator torque saturation curves** | Neither sim nor real hit hard torque limits during these gentle gestures; the policy stayed comfortably in the linear regime |
| **Joint friction / damping at low velocity** | A reactive PD policy continuously excites every joint, masking static-friction differences |
| **Cable / battery / harness mass and CoM offsets** | Bundled into the torso inertia term; can be off by O(10%) and still match IMU angvel within our noise floor |
| **IMU latency + noise model** | We match signal RMS, not phase response. A 5-10 ms latency difference would change the policy's effective phase margin without showing up here |
| **Joint encoder noise / quantisation** | Buried below the policy's LPF |
| **Backlash, soft-body / cable compliance** | Hard for any rigid-body URDF; closed-loop tracking hides residuals |
| **Free-base dynamics** (untethered standing or walking) | All real recordings have the gantry strap engaged |

### Specific signs of modeling divergence the anchors *do* show

Two findings that point at unmodeled or mis-modeled physics, even though
per-DoF joint state agrees tightly:

1. **Anchor B: 166 deg sim yaw drift vs -6.5 deg real yaw drift.** Per-cycle
   leg kinematics agree to 4.74 deg RMS, but the integrated base trajectory
   diverges wildly. Classic signature of contact / friction / floor-model
   error integrated over many gait cycles, partially masked in real by
   the gantry pinning the base.
2. **Anchor C: per-leg `right_knee` cmd-range diverges by ~30 deg between
   sim and real** despite the motion being stand-in-place. The policy is
   reacting to *different* IMU/state observations between worlds, which
   means the underlying body dynamics produced different sensor readings --
   a sign that **inertia or contact response models** are not perfectly
   matched.

### What would close the remaining gaps (policy-free bench tests)

The cleanest validations of URDF/MJCF fidelity are *passive* tests where
no policy is in the loop, isolating the physics model from the controller:

1. **Passive sway / drop test** (motors off, gantry slack): push body 10 deg, log IMU. Tests inertia + joint friction in isolation.
2. **Open-loop joint sweep** with low PD, no policy: sinusoidal target on one joint at a time, log torque + position. Tests per-actuator dynamics.
3. **Hanging-arm gravity test**: motors off on one arm, measure equilibrium angle vs MuJoCo. Tests per-link mass distribution.
4. **Foot contact characterisation**: lift and drop the robot from 5 cm, compare ground-reaction profile. Tests contact geometry + floor model.
5. **Per-joint step response**: step input (0 -> 30 deg) with policy off, log overshoot + settling. Tests PD gains, motor inertia, friction in isolation.

These are 1-2 hour bench tests each. A round of these would convert the
indirect, closed-loop evidence above into direct, isolated component-level
validation of the digital twin.

### Calibrated bottom-line claim

What these anchors support saying:

> *On the gantry-tethered X2 Ultra, the URDF/MJCF reproduces real-robot
> joint kinematics within ~5 deg per-DoF RMS and base IMU angular-velocity
> within ~5% during closed-loop reference-tracking of in-distribution
> motions, validating the kinematic chain and torso inertia of the model.*

What they do **not** support saying (yet):

> ~~The contact model, actuator saturation, joint friction, free-base
> dynamics, and sensor noise models of the digital twin are validated.~~

The Anchor B yaw-drift finding is direct evidence that at least one of
those untested components (most likely contact / friction / free-base
integration) is meaningfully off in the current MJCF.

## What's in each anchor folder

```
anchor_<id>_<name>/
  real.npz         deploy_x2.sh local --record output (X2 Ultra hardware)
  sim.npz          deploy_x2.sh sim   --record output (MuJoCo, parity profile)
  SUMMARY.md       commands used, comparison results, key findings
  plots/           PNG figures + summary.json/txt produced by the comparator
```

## Reproducing the comparison

Use [`gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py`](../../gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py):

```bash
python gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py \
    --real data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/real.npz \
    --sim  data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/sim.npz \
    --out  /tmp/anchor_d_redo \
    --label-real "iter-22k real (walk_demo_v6)" \
    --label-sim  "iter-22k MuJoCo (walk_demo_v6)"
```

The script auto-detects the CONTROL window in each run (using the
leg-knee `kp` schedule), resamples both onto a common 50 Hz grid, and
emits per-DoF tracking error, sim-vs-real `cmd_pos`/`state_pos`
diffs, IMU overlays, and a heatmap of the cmd diff over time.

## Visualising both runs side-by-side (no gravity)

Use [`gear_sonic_deploy/scripts/play_npz_dual_kinematic.py`](../../gear_sonic_deploy/scripts/play_npz_dual_kinematic.py):

```bash
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic_deploy/scripts/play_npz_dual_kinematic.py \
    --real data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/real.npz \
    --sim  data/sim_to_real_anchors/anchor_d_iter22k_walk_demo_v6/sim.npz \
    --separation 1.5 \
    --speed 1.0
```

Both robots share the same MJCF, are placed `--separation` metres apart in
the x-axis at z=0.95 m (no gravity), and play back the recorded joint
state trajectories from each npz. IMU quaternion drives the base
orientation so torso tilt and full-body yaw are visible. Use SPACE to
pause, R to restart, and arrow keys to scrub by ten frames.

## Recorder npz schema

See `docs/source/paper/sim_to_real_recordings_inventory.md`, section 1.
Key fact: the four-group recordings carry `cmd_pos_<group>`,
`state_pos_<group>`, plus `t_cmd_<group>` / `t_state_<group>` at
~500/~1000 Hz, and `imu_quat_wxyz` / `imu_angvel` at ~500 Hz.

## Why these recordings are gold

1. **Shared inputs.** Same checkpoint, same playlist, same tuning;
   the only difference is the physics environment.
2. **Provenance is unambiguous.** Mapping from disk artefact back to
   deploy command is in `sim_to_real_recordings_inventory.md` and in
   each anchor's `SUMMARY.md`. The recorder `meta_json` does NOT yet
   bake `--model` / `--motion` (TODO at the host).
3. **Reproducible.** Sim half can be regenerated from `deploy_x2.sh sim`
   in a few minutes; real half cannot, which is why we mirror it here.
4. **Apples-to-apples.** Both runs traverse the same playlist phases
   (the controller is deterministic given the playlist), so the
   `compare_sim_vs_real_npz.py` script can align them on a common time
   grid for stride-by-stride analysis.
