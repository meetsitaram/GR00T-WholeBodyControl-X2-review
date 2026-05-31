# X2 PD Tuning via MC Motor Scan + Nudge Tests

> **Audience.** Operators bringing a fresh SONIC checkpoint up on the real
> AgiBot X2 Ultra who need to set the deployment-time PD scale knobs
> (`kp_scale_*`, `kd_scale_*`) and the per-group target clamps
> (`max_target_dev_*`) in `gear_sonic_deploy/configs/real_deploy_tuning/`.
>
> This is the practical procedure that produced the values shipped in
> `expressive.yaml`. It assumes you already have the bring-up checklist in
> [`x2_first_real_robot.md`](x2_first_real_robot.md) green and that the
> `dry-run` flow in [`x2_sonic_deploy_real.md`](x2_sonic_deploy_real.md)
> works end to end.

## Why we need this

The trained policy in `policy_parameters.hpp` ships with PD gains
(`kps[31]`, `kds[31]`) calibrated against IsaacLab's **implicit**
integrator. On the real X2 Ultra the motion controller (`mc_app_main` on
PC1, the motion-control unit) drives joints with **explicit** torque, so
the deployed loop gain is ~1.3-1.5x lower than what training optimised
for. The robot stands fine in static equilibrium but wobbles on nudge or
disturbance.

The fix is the standard G16b workaround: keep the trained policy
unchanged and bump deployed PD per joint family with multiplicative scale
factors. The deploy binary exposes them as `--kp-scale-*` /
`--kd-scale-*` flags; the YAML presets in
`gear_sonic_deploy/configs/real_deploy_tuning/` are how operators check
in known-good combinations.

The question this doc answers is: **what numbers do you put in those
fields?** Not "1.5 because someone said so", but "1.87 on
`kp_scale_ankle_pitch` because MC publishes 40 N·m/rad on
`left_ankle_pitch_joint` while training used 21.38".

## Tools you will use

| Tool | Purpose |
|---|---|
| `gear_sonic_deploy/scripts/x2_scan_mc_motors.sh` | Pure subscriber to `/aima/hal/joint/{leg,waist,arm,head}/{state,command}` topics. Captures MC's published kp/kd/target plus measured pos/vel/eff. Writes `mc_motor_scan_<unix_ts>.jsonl` and prints two summary tables (per-joint baseline + oscillation analysis). |
| `gear_sonic_deploy/scripts/x2_scan_mc_motors.py --replay PATH` | Re-analyse an existing JSONL with different filter settings (`--osc-lpf-hz`, `--osc-vel-dead-zone`) without re-running the live scan. |
| `gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml` | The shipped preset that already encodes the discovered MC-match values. Read its inline comments for the rationale behind every constant. |
| `gear_sonic_deploy/configs/real_deploy_tuning/_schema.yaml` | The full set of supported tuning knobs with descriptions. Authoritative reference; any new knob must land here AND in `tuning_config_to_args.py` AND as a CLI flag in `x2_deploy_onnx_ref.cpp`. |

The scan tool is **read-only**: it never publishes on the bus, so it is
safe to run any time, including while MC is actively holding the robot
up in `STAND_DEFAULT` or while a deploy is running.

## Quick start (the entire loop in 5 commands)

```bash
# 0) Hoist robot on gantry. MC in STAND_DEFAULT.

# 1) Capture MC's stock PD baseline (~30 s; nudge during the window)
./gear_sonic_deploy/scripts/x2_scan_mc_motors.sh --duration 30
# -> reads MC's kp/kd/target on every joint group; prints the table.

# 2) Stop MC, start the deploy with your candidate tuning preset
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/.../exported/model_step_NNNNNN_g1.onnx \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --log-dir /tmp/x2_run_$(date +%Y%m%d_%H%M%S)

# 3) In a second terminal: scan again WHILE the policy is in CONTROL,
#    nudge the robot 1-2 times during the window
./gear_sonic_deploy/scripts/x2_scan_mc_motors.sh --duration 30

# 4) Compare the two scans. Tune. Re-run.

# 5) Re-analyse the last scan with tighter LPF / dead-zone if you want
#    to confirm a borderline ringing finding without re-nudging:
./gear_sonic_deploy/scripts/x2_scan_mc_motors.py \
    --replay mc_motor_scan_<ts>.jsonl --osc-lpf-hz 6.0 --osc-vel-dead-zone 0.03
```

## Step 1 — Capture MC's stock PD baseline

Run the scanner with MC actively holding the robot in `STAND_DEFAULT`:

```bash
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic_deploy/scripts/x2_scan_mc_motors.sh --duration 30
```

The shim script re-execs inside `gear_sonic_deploy/docker_x2/x2sim` so
the scanner has `aimdk_msgs` on the Python path and a real-mode DDS
discovery setting. From the host you do not need to source ROS 2.

What you should see at startup, before the nudge window opens:

```
[x2_scan_mc_motors] Subscribed to /aima/hal/joint/{leg,waist,arm,head}/{state,command}.
[x2_scan_mc_motors] DDS publisher discovery on /command topics:
  /aima/hal/joint/leg/command:   1 publisher(s)
    - node=/mc/...  reliability=BEST_EFFORT durability=VOLATILE depth=1
  /aima/hal/joint/waist/command: 1 publisher(s)
  ...
[x2_scan_mc_motors] >>> NUDGE NOW <<<  (recording for 30 s)
```

Push the robot once or twice during the recording window — gentle nudges
on the chest in roughly orthogonal directions (forward, lateral). The
oscillation analysis at the end needs at least one transient to score
ring-down; without a nudge the table just shows that everything is at
baseline. Mark the wall-clock time of each nudge in your bring-up log.

If discovery returns `NO publishers` for any `/command` topic, **MC isn't
in control**. The most common cause is that the deploy is still up and
holding the bus; stop the deploy (`Ctrl-C`, wait for the EM handoff to
restart MC), confirm `STAND_DEFAULT` via the operator console, and
re-run the scanner.

The two summary tables that print at the end are the entire output that
matters for tuning — see [Reading the per-joint table](#reading-the-per-joint-table)
and [Reading the oscillation table](#reading-the-oscillation-table)
below.

The raw timeseries lands in `mc_motor_scan_<unix_ts>.jsonl` in the
working directory. Every line is one DDS callback (`state` or
`command`), with a monotonic timestamp so you can crop the nudge window
post-hoc with the `--replay` flag.

### Reading the per-joint table

```
joint                          mc_kp  mc_kd   mc_tgt      def     pos_med  tgt-def  pos-tgt_p95  |vel|_p95  |eff|_p95
left_hip_pitch_joint          100.00   3.00   -0.312   -0.312    -0.310    +0.000      0.0040     0.0210      6.450
left_knee_joint               150.00   5.00   +0.669   +0.669    +0.671    +0.000      0.0058     0.0245      8.910
left_ankle_pitch_joint         40.00   3.00   -0.363   -0.363    -0.358    +0.000      0.0125     0.1450      4.330
left_ankle_roll_joint          30.00   2.00   +0.000   +0.000    +0.001    +0.000      0.0085     0.0790      2.180
waist_yaw_joint                40.00   8.00   +0.000   +0.000    +0.001    +0.000      0.0042     0.0220      0.910
waist_pitch_joint              40.00   5.00   +0.000   +0.000    +0.000    +0.000      0.0050     0.0260      1.140
waist_roll_joint               40.00   5.00   +0.000   +0.000    +0.000    +0.000      0.0048     0.0240      1.090
left_shoulder_pitch_joint      14.00   1.00   +0.200   +0.200    +0.200    +0.000      0.0021     0.0080      0.230
...
```

Columns:

- **`mc_kp`, `mc_kd`** — MEDIAN MC-published stiffness / damping over
  the whole scan. If MC modulates these (variable-stiffness modes), the
  median hides that; cross-check the JSONL.
- **`mc_tgt`** — MEDIAN MC-published target. Should equal `def` (the
  codegen baseline from `policy_parameters.hpp::DEFAULT_ANGLES`) in
  `STAND_DEFAULT`. If `tgt-def` is non-zero, MC is actively servoing
  somewhere else and the rest of the row should be interpreted relative
  to `mc_tgt`, not `def`.
- **`pos_med`** — MEDIAN measured position. Compared to `mc_tgt` it
  tells you the static tracking error of MC's PD on real hardware.
- **`pos-tgt_p95`** — 95th-percentile `|pos − tgt|`. During a NUDGE this
  is the disturbance amplitude MC is rejecting — the bigger this is for
  a given nudge, the more compliant MC's PD is on that joint.
- **`|vel|_p95`, `|eff|_p95`** — 95th-percentile speed and effort. A
  barely-perceptible chest push typically shows up as 0.05-0.20 rad/s
  velocity and a few N·m of effort on legs/waist; arms barely move at
  all under that nudge.

This table gives you the **MC-match denominator** for the PD scale
knobs. To pick `kp_scale_X`, take MC's published `mc_kp` for joints in
group `X` and divide by the trained `kps[i]` from
`policy_parameters.hpp` (also dumped in the YAML comments).

Worked example, ankle_pitch:

```
mc_kp (from this table)           = 40.00 N·m/rad
trained kps["left_ankle_pitch"]   = 21.38 N·m/rad   (policy_parameters.hpp)
=> kp_scale_ankle_pitch (MC match) = 40.00 / 21.38 = 1.87
```

The `expressive.yaml` shipped today encodes exactly this calculation in
its inline comments for every joint family.

### Why ankle and waist need split knobs

Two joint families publish ASYMMETRIC PD across their sub-axes; matching
MC requires splitting the legacy single-knob into per-subgroup knobs.

| Family | Sub-axis | MC kp | MC kd | trained kp | trained kd | MC-match scale |
|---|---|---|---|---|---|---|
| Ankle | pitch | 40 | 3.0 | 21.38 | 0.907 | `kp 1.87 / kd 3.31` |
| Ankle | roll | 30 | 2.0 | 21.38 | 0.907 | `kp 1.40 / kd 2.20` |
| Waist | yaw | 40 | 8.0 | 40.18 | 2.56 | `kp 1.00 / kd 3.13` |
| Waist | pitch + roll | 40 | 5.0 | 14.25 | 0.907 | `kp 2.81 / kd 5.51` |

If you only set the legacy `kp_scale_ankle` you are forced to compromise
between the pitch and roll axes. Same for `kp_scale_waist` between the
yaw axis and the pitch/roll pair. The split knobs
(`kp_scale_ankle_pitch` / `..._roll`, `kp_scale_waist_yaw` / `..._pr`)
solve that. Backward-compat: the legacy aliases still apply
multiplicatively on top of the split knobs, so older presets keep
working.

## Step 2 — Run the policy with your candidate preset

The preset that produced the shipped numbers is
`gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml`. Its
inline comments document every choice; for the rest of this doc we
treat it as the working set.

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/.../exported/model_step_NNNNNN_g1.onnx \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --log-dir /tmp/x2_run_$(date +%Y%m%d_%H%M%S)
```

Wait until the operator console shows `CONTROL` and the robot is
holding `STAND_DEFAULT` cleanly. **Do not skip the dry-run sequence
described in [`x2_first_real_robot.md`](x2_first_real_robot.md) on a
fresh checkpoint** — these PD trims do not protect against a divergent
policy, only against marginal stability of a healthy one.

The deploy binary applies the YAML preset's flags first, so anything
you pass explicitly on the command line wins. Typical iteration pattern:

```bash
# Start with the shipped preset
... --tuning-config .../expressive.yaml

# Try a tighter waist KD without editing the YAML
... --tuning-config .../expressive.yaml --kd-scale-waist-pr 4.5
```

## Step 3 — Scan again with the policy in control

```bash
./gear_sonic_deploy/scripts/x2_scan_mc_motors.sh --duration 30
```

This time the `command` topics will show the **deploy's** publishers
(under `/x2_deploy_onnx_ref/...` instead of MC). The kp/kd values in
the per-joint table are now the trained `kps[i] * kp_scale_*` product
that the deploy is actually emitting — a free sanity check that your
tuning preset is being applied.

Nudge the robot 1-2 times during the window. The oscillation analysis
at the end is the primary signal you want.

### Reading the oscillation table

```
=== Oscillation analysis (sliding 1.0 s window, step 0.25 s) ===
    Velocity LPF cutoff: 10.0 Hz   |   dead-zone: 0.050 rad/s

  joint                       peak_osc  peak@t  flip_hz  vel_rms  pos_p2p  baseline   ratio
  ----------------------------------------------------------------------------------------------
  left_ankle_pitch_joint        0.083    14.50     2.00   0.0415   0.1900    0.0040    20.7x  <- RINGING
  right_ankle_pitch_joint       0.078    14.75     2.00   0.0390   0.1850    0.0042    18.6x  <- RINGING
  left_knee_joint               0.062    14.75     1.50   0.0413   0.2500    0.0030    20.7x  <- RINGING
  right_knee_joint              0.058    14.75     1.50   0.0387   0.2400    0.0029    20.0x  <- RINGING
  waist_pitch_joint             0.012    14.50     1.00   0.0120   0.0240    0.0008    15.0x  <- RINGING
  ...
  left_shoulder_pitch_joint     0.0006    8.20     0.00   0.0006   0.0010    0.0005     1.2x  <- background
  head_pitch_joint              0.0004   12.10     0.00   0.0004   0.0008    0.0004     1.0x  <- background
```

Columns:

- **`peak_osc`** = `flip_hz * vel_rms` in the worst-offender 1 s
  window. The composite metric: a single nudge that decays without
  ringing scores low (high `vel_rms`, near-zero `flip_hz`); a joint
  oscillating at a few Hz with sustained speed scores high. **Above
  0.05 with `ratio` ≥ 5x is a clear ring-down.**
- **`peak@t`** — elapsed seconds when the worst window started. Should
  line up with when you nudged plus a brief settle.
- **`flip_hz`** — sign-flip rate of the LPF'd velocity (1 oscillation =
  2 flips, so this is already in physical Hz). Ankle ring-downs are
  typically 2-5 Hz; knees / waist 1-3 Hz; > 8 Hz post-LPF means a very
  stiff joint or a closed-loop instability.
- **`vel_rms`** — RMS LPF'd velocity in the peak window. Magnitude of
  the actual motion; pairs with `pos_p2p` to estimate the size of the
  ring-down.
- **`pos_p2p`** — peak-to-peak position swing. The most operator-legible
  number on the table; use it to compare nudge intensity across runs.
- **`baseline`** — median `osc_power` excluding the peak ±2 windows.
  Tells you whether the peak is a true transient or chronic background
  jitter.
- **`ratio`** = `peak_osc / baseline_osc`. **`< 2x` = background
  jitter (joint never really oscillated)**. **`>= 5x` = real
  ring-down**. The `<- RINGING` / `<- background` markers are emitted
  using these thresholds.

### Why velocity gets LPF'd before flip-counting

The HAL state stream comes through at ~1067 Hz on legs/waist/arm. At
that rate, encoder-tick differencing, motor commutation, and structural
micro-vibration produce dozens of spurious sign flips per second on a
joint that is, physically, rotating smoothly. Counting those raw flips
gave nonsense `flip_hz` values like 25-38 Hz on top of an underlying
0.5 rad smooth motion in the first cut of the metric.

The clean-up is two-stage:

1. **LPF (boxcar moving average)** at `--osc-lpf-hz` (default 10 Hz).
   The 3 dB cutoff of an N-tap boxcar sits near `fs / (2N)`, so the
   default cuts everything above ~9 Hz. Real underdamped ankle / knee /
   waist ring-downs (1-5 Hz) are preserved with margin; the high-freq
   sensor floor collapses to ~zero.
2. **Dead-zone** at `--osc-vel-dead-zone` (default 0.05 rad/s). Below
   this magnitude the joint is "effectively still" and we set velocity
   to 0, so micro-reversals around the quiescent point do not get
   counted as oscillation flips.

The flip counter then uses a small state machine that tracks the *last
non-zero* velocity sign. A flip is counted only when we transition from
one sign to the opposite via any number of dead-zone (zero) samples in
between, so a joint that "settles into stillness" does not register as
oscillating.

## Step 4 — Tune from the table

The general rule is: **KD bumps matter MORE than KP bumps for nudge
rejection specifically.** Our trained PD is critically-damped against
IsaacLab's implicit model, which leaves it heavily under-damped on real
hardware. MC's KD is 2-5x ours on every trunk joint. The waist
pitch/roll KD bump (5.51x = MC-match) is the single biggest knob for
forward/back wobble, because it kills the torso-leaning mode before
ankle/hip have to spend torque fighting the gravitational moment of the
leaned trunk.

Suggested tuning order:

1. **Match MC on every joint family first.** Use the values in the
   table above. This is the floor — the policy was trained against an
   implicit integrator that effectively gave it more damping than
   `kds[i]` would suggest, so MC-match is a known-safe starting point.
2. **Re-scan with the policy in control.** Look at the oscillation
   table. Anything ringing > 5x is an undertuned joint family; anything
   ringing < 2x is fine.
3. **For each ringing family, lift KD first.** Start at +50% above
   MC-match and re-test. Ankle pitch ended up at `kd_scale_ankle_pitch
   = 5.0` (vs MC-match 3.31) for exactly this reason — the MC-match
   value was still leaving 0.30 rad ankle swing on a forward push. The
   final 5.0 dropped that to 0.19 rad and the secondary knee swing
   from 0.45 → 0.25 rad.
4. **Lift KP only if KD alone cannot kill the ring AND `pos-tgt_p95`
   in the per-joint table is large.** A high `pos-tgt_p95` means the
   joint can't even hold target under static load; that's a stiffness
   problem, not a damping one.
5. **Verify idle is still safe.** After each KD bump, run a 10 s scan
   with NO nudge and check that the ringing joint shows `pos_p2p < 0.1
   deg` in the oscillation table. If idle ringing has appeared, you've
   gone too high on KD and the policy's internal noise is being
   amplified into a limit cycle.

The cycle ends when:

- No joint marks `<- RINGING` after a nudge of comparable intensity to
  what you'll see in real teleop / VLA usage.
- Idle scans show every joint at `pos_p2p < 0.2 deg` over 10 s.
- The 95th-percentile effort on any single joint stays bounded across
  multiple nudges — values bouncing 5x between scans means you're at
  the edge of stability for that joint family.

## Step 5 — Replay mode for filter sweeps

If you want to confirm a borderline finding without re-running the
hardware test (e.g., "did that ankle really ring at 2 Hz, or am I
seeing the LPF response?"), use `--replay`:

```bash
./gear_sonic_deploy/scripts/x2_scan_mc_motors.py \
    --replay mc_motor_scan_1778884089.jsonl \
    --osc-lpf-hz 6.0 \
    --osc-vel-dead-zone 0.03
```

This skips the live ROS scan entirely, reads the captured JSONL back
in, and re-runs the same `_summarize` + `_oscillation_summary` pipeline
with whatever filter settings you want to try. The ankle-resonance
finding is robust if it survives both `--osc-lpf-hz 10` (default) and
`--osc-lpf-hz 6` (more aggressive). If the peak vanishes at 6 Hz it
was actually a 7-9 Hz artefact of the default filter, not a real
mechanical resonance.

## Per-group `max_target_dev` clamps

The PD scale knobs determine *how stiff* the joint is; the
`max_target_dev_*` knobs determine *how far* the policy is allowed to
push the target away from `default_angles`. They protect against a
divergent policy generating torque spikes via the high-stiffness leg
chain.

The 4 groups map to MuJoCo joint indices (see
`policy_parameters.hpp::mujoco_joint_names`):

| Group | MJ indices | What's in it |
|---|---|---|
| `leg` | 0..11 | hip pitch/roll/yaw, knee, ankle pitch/roll, both sides |
| `waist` | 12..14 | yaw, pitch, roll |
| `arm` | 15..28 | shoulder pitch/roll/yaw, elbow, wrist yaw/pitch/roll, both sides |
| `head` | 29..30 | yaw, pitch |

For teleop / VLA use the natural setting is **tight legs/waist/head,
wide arms**:

- Legs run `kp ~99 N·m/rad`. Same nominal travel produces ~7x the
  torque of the arms (`kp ~14`). Keep legs at 0.30-0.50 rad max
  deviation.
- Waist similarly stays at 0.30 — the upper-body posture is mostly
  driven by torso reference quaternions, not large waist commands.
- Arms run wide (1.50 rad ≈ 86 °) so IK-driven wrist targets and
  reach-envelope tracking are not truncated.
- Head 0.50 covers VR look-at sweeps.

The `expressive.yaml` ships exactly this split. `conservative.yaml`
falls back to a uniform 0.30 rad clamp for first powered runs with a
new checkpoint, where the scenario is "if the policy is going to
misbehave, the tighter clamp catches it first".

## Logged history of every tuning decision

The shipped `expressive.yaml` has inline comments next to every PD
scale and clamp explaining:

- the MC-published value it derives from (and which `mc_motor_scan_*.jsonl`
  recorded that value),
- the operator nudge-test result that prompted any deviation from
  MC-match, and
- the failure mode you'd hit if you went outside the documented range.

When you change one of these values in a new preset, please mirror that
discipline: future you (or the next operator) needs to know why the
number is what it is, not just what it is.

## See also

- [`gear_sonic_deploy/configs/real_deploy_tuning/README.md`](../../../gear_sonic_deploy/configs/real_deploy_tuning/README.md)
  — the operator-facing README for the YAML presets.
- [`_schema.yaml`](../../../gear_sonic_deploy/configs/real_deploy_tuning/_schema.yaml)
  — authoritative list of supported tuning keys and their CLI mapping.
- [`x2_sonic_deploy_real.md`](x2_sonic_deploy_real.md) — end-to-end
  deploy runbook (covers the bring-up flow this doc plugs into).
- [`x2_first_real_robot.md`](x2_first_real_robot.md) — operator safety
  checklist; do not skip on a fresh checkpoint.
- [`sim2sim_mujoco.md`](sim2sim_mujoco.md), section "G16b deployment
  PD trim" — the upstream rationale for why MuJoCo / real X2 need PD
  bumps relative to IsaacLab training.
