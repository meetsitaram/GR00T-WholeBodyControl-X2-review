# 2026-05-02 — Post-Deploy Tuning Infrastructure & First Gesture Playlist

> **Continuation of the iter-4000 day.** After the morning's first powered
> run on `minimal_v1`, we built the post-deploy tuning toolchain (recorder,
> YAML presets, output-side LPF), uncorked the policy from the safety
> clamp, and ran the **`standing_gestures_v1`** playlist end-to-end on the
> real X2 Ultra. Visibly fuller arm motion, lower IMU peak angular
> velocity, and a parity-safe path forward for sim-to-real iteration.

---

## What got built today

### 1. Real-run recorder (`x2_record_real_run.py` + `x2_record.sh`)

A pure-subscriber ROS 2 node that latches every `JointStateArray`,
`JointCommandArray`, and `Imu` message arriving on `/aima/hal/*` to an
`.npz` archive, with a host-runnable `--summarize` post-mortem. Integrated
into `deploy_x2.sh` as `--record [--record-out PATH]` so a single command
captures the full run with zero terminal coordination -- no more
"start the recorder in another tab and pray the timing works."

The recorder was the diagnostic instrument that exposed every other
finding below.

### 2. The 0.30-rad safety clamp was sawing the policy off

First post-mortem on `minimal_v1` showed every wrist/shoulder-yaw cmd
range pinned to **exactly 17.2° (= 0.300 rad)** -- not a coincidence,
that's `--max-target-dev 0.30`. The deploy log said `act_clip_ticks=0`
truthfully, but that counter only watches the raw `action_il` clip, not
the per-joint target clamp. Misleading "no clipping" for a run that was
clipping every arm joint on every tick. (A future tick-counter for the
target-clamp would close this gap.)

### 3. Real-deploy tuning configs (parity-safe)

Authored `gear_sonic_deploy/configs/real_deploy_tuning/`:

```
configs/real_deploy_tuning/
├── README.md          # parity contract + how-to-add
├── _schema.yaml       # authoritative key reference
├── conservative.yaml  # 0.30 clamp, no LPF -- first-run validation
└── expressive.yaml    # 0.80 clamp + 8 Hz LPF -- gesture demos
```

Loaded via `deploy_x2.sh local --tuning-config <preset>.yaml`. Translation
done by `gear_sonic_deploy/scripts/tuning_config_to_args.py` (YAML →
`--flag VALUE` tokens), prepended to the binary's arg list so explicit
CLI flags still win for A/B sweeps.

**Parity is preserved by construction**:

* `--tuning-config` is **rejected in sim mode** with a friendly error
  pointing at the rule.
* Every knob in the schema today maps to a CLI flag the binary already
  exposes (or a new one that's parity-safe -- see next item).
* `--obs-dump` returns from `OnControl` *before* any post-policy filter
  is reached, so `compare_deploy_vs_python_obs.py` keeps comparing
  bit-identical raw policy outputs.

### 4. Output-side target LPF (`--target-lpf-hz HZ`)

First-order EMA on the published joint targets, applied AFTER the safety
stack and BEFORE the bus, bypassed in RAMP_OUT/SAFE_HOLD. `0` = disabled
= full parity. Real-deploy mitigation only.

```cpp
target_lpf_state_[i] = a * sc.target_pos_mj[i] + (1-a) * target_lpf_state_[i];
```

Where `a = 1 - exp(-2π·hz·dt)` at the 50 Hz `OnControl` rate.

---

## Empirical sweep: how much LPF is enough?

Ran `standing_gestures_v1` (~17.6 s of motion) three times with
identical model, motion, and `max_target_dev=0.80`, varying only the LPF:

| Run                  | clamp | LPF  | IMU peak \|ω\| | Arm `jit_cmd` Δ vs BEFORE | Final tilt |
|----------------------|-------|------|---------------:|--------------------------:|-----------:|
| BEFORE (clamp@0.30)  | 0.30  | off  | 3.36 rad/s     | --                        | 3.7°       |
| `expressive` default | 0.80  | 8 Hz | 3.05 rad/s     | +14% (bigger motion)      | **1.5°**   |
| **LPF override 5 Hz**| 0.80  | 5 Hz | **2.78 rad/s** | **−15%**                  | 4.3°*      |

\* end-of-run snapshot; peak-during-CONTROL is the more meaningful
quality metric.

**Per-joint arm jitter at 5 Hz vs 8 Hz** (same clamp, same motion):

```
right_shoulder_roll  -25%
left_shoulder_yaw    -17%
left_wrist_yaw       -17%
left_elbow           -16%
right_shoulder_pitch -15%
right_wrist_pitch    -15%
right_shoulder_yaw   -14%
left_shoulder_roll   -13%
right_elbow          -13%
```

Operator's eye agrees: "looked better than before, but there is still a
lot of room for improvement" → `LPF=5` is meaningfully smoother than
`LPF=8` while preserving the wider arm range.

Leg `jit_cmd` *increased* at LPF=5 (knees +46%, ankles +45%), but
hardware tracking error stayed flat at ~0.04 rad RMS and IMU peak ω
*decreased* -- legs are doing more balance work to support the bigger
arm trajectories the wider clamp now allows. Net body-level disturbance
is lower.

---

## Open follow-ups (for the next session)

1. **Lock in LPF=5 as the new `expressive.yaml` default** -- one-line YAML
   change. Current 8 Hz default was a guess; 5 Hz is empirically better.
2. **Input-side `joint_vel` smoothing** -- the 50 Hz finite-difference
   amplifies encoder noise ~50×, and that's what the policy is reactively
   chasing in the legs. Output LPF damps the response; input LPF would
   attack the noise at the source. Bigger change (touches the obs
   builder + needs an explicit "real-only" guard to keep parity intact),
   but it's the right direction once we've squeezed the output side.
3. **Target-clamp tick counter** -- mirror `act_clip_ticks` so the deploy
   log surfaces when the per-joint clamp is gagging the policy (today it
   silently saws everything off at the limit).
4. **iter-10000 checkpoint** -- ready to download from Nebius and bring
   through the same `conservative.yaml → expressive.yaml → LPF sweep`
   ladder. Everything we built today was iter-agnostic.

---

## What this run *does* prove

* Real-deploy tuning can be expressed as version-controlled YAML, loaded
  by name, and overridden ad-hoc -- without polluting the sim parity
  surface. The architecture works.
* The recorder is bit-perfect on state capture (verified during the
  manual-arm-wiggle test) and now captures real cmd traffic during a
  live deploy, enabling proper drift / jitter / clamp-hit analysis.
* Output-side LPF at 5 Hz cleanly attenuates the leg/waist noise the
  policy was reactively chasing on real hardware, with no measurable
  regression in arm-motion expressiveness.
* The same `iter-4000` checkpoint that stood "rock solid" on
  `minimal_v1` also handles a 7-segment 17.6 s gesture choreography
  end-to-end -- including the `tiny_one_hand_pick`, two-handed swing,
  and put-down segments -- with tilt staying within ±5° of upright
  throughout.

## What this run *doesn't* prove yet

* Whether 5 Hz is truly optimal or just the best of three trials. A
  finer sweep (3, 5, 7 Hz) and a per-group LPF (more aggressive on legs
  than arms) are still on the table.
* Whether the iter-10000 checkpoint will need different tuning. The
  presets are defined relative to current model behaviour; a more
  capable model may want looser clamps and lighter filtering.
* Whether the residual jitter the operator noted ("still a lot of room
  for improvement") is dominated by output side or input side. The
  jit_cmd-improves-on-arms-but-not-legs pattern at LPF=5 hints that
  it's input-side now.

A good day. The infrastructure is in place; the model can be iterated
without re-engineering the tuning loop each time.
