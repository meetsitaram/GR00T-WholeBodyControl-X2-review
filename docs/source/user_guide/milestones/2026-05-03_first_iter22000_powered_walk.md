# 2026-05-03 — First Powered Walk on the Real X2 Ultra

> **A cherished moment.** First powered walking run on the real X2 Ultra.
> The iter-22000 sphere-foot checkpoint tracked a `Turn_Start_Walk_0090_003`
> reference for a full out-and-back cycle on a one-way rail gantry — feet
> stepping, hips rotating ±55°, torso held, no fall, no SAFE_HOLD, no
> RAMP_OUT trip. Total run wall-time 36.75 s including pre/post anchors,
> with the cleanest MC handoff to date (0.20 s `JOINT_DEFAULT` dwell).

---

## The run

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

A single command from the host shell. The script auto-relaunched into the
`docker_x2/x2sim` container (real-robot DDS overlay), built the deploy
binary in the background, brought up the recorder before the safety
gate, ran the cold-warm STANDBY → trigger sequence, stopped MC, ran 14 s
of CONTROL on the casual_walk reference, executed RAMP_OUT to MC's
`STAND_DEFAULT` pose, held it through the MC restart, and handed back to
`STAND_DEFAULT` cleanly. Operator never left the gantry.

## What the deploy saw

```
[22:20:46.125] [handoff] deploy is in STANDBY and ready.
[22:20:50.412] [SAFETY GATE] Stop MC and launch policy? [y/N]: y
[22:20:50.413] [handoff] stopping MC via PC1 EM HTTP API ...
[22:20:52.193] [handoff] start-trigger sentinel touched -> deploy entering CONTROL.
[22:20:52.301] CONTROL tick=000  policy_t=0.00s  alpha=0.00
[22:20:53.301] CONTROL tick=050  policy_t=1.00s  alpha=0.49
[22:21:06.301] CONTROL tick=700  policy_t=14.00s alpha=1.00
[22:21:06.321] Max duration elapsed (14.00s) -> RAMP_OUT (2.00s)
[22:21:08.341] RAMP_OUT complete -> HOLD_FOR_MC
[22:21:08.341] [handoff] HOLD_FOR_MC sentinel detected -- POSTing start_app
[22:21:09.501] [post-handoff] escalator launched (hammering SetMcAction(JOINT_DEFAULT) at 20 Hz)
[22:21:11.685] [post-handoff] -> JOINT_DEFAULT confirmed by escalator (active PD; releasing deploy now)
[22:21:11.886] HOLD_FOR_MC: exit-sentinel touched after 3.36s -> shutting down
[22:21:14.412] [post-handoff] STAND_DEFAULT confirmed via mc_get_action.
[22:21:14.512] [handoff] deploy exited with code 0.
[22:21:27.045] [cleanup] recorder finalized: scratch/runs/x2_run_20260503_222045/run.npz
```

## Headline numbers

| | Value |
|---|---|
| CONTROL window | **14.00 s clean**, then 2.00 s RAMP_OUT |
| Tilt during run | **2.6° median, 18.8° peak** at the turn apex |
| Max angular velocity | 4.89 rad/s (~280 °/s) at the turn moment |
| Pelvis stability | no fall, no SAFE_HOLD, no tilt-watchdog trip |
| MC `JOINT_DEFAULT` dwell | **0.20 s** (was 1.60 s on the previous gestures run — 8× shorter) |
| Topic rates | 447 Hz cmd / 942 Hz state across leg/waist/arm; no drops |

## Tracking — legs actually moved with the reference

| Joint | cmd range (deg) | state range (deg) | tracking |
|---|---:|---:|---:|
| `left_hip_yaw`     | 116.7 | 113.6 | **97 %** |
| `right_hip_yaw`    | 110.7 | 107.3 | **97 %** |
| `waist_yaw`        |  84.7 |  82.9 | **98 %** |
| `left_knee`        |  46.2 |  37.4 | 81 % |
| `right_knee`       |  45.2 |  37.8 | 84 % |
| `left_hip_pitch`   |  57.5 |  37.8 | 66 % |
| `right_hip_pitch`  |  57.2 |  44.4 | 78 % |

Hip_yaw at 97% on **both** legs means the policy actually rotated each
foot to where it asked — that's the turn working mechanically. waist_yaw
at 98% means the torso followed. The 116° hip_yaw state range is the
robot's feet rotating ±55-60° each, consistent with a real walking
pivot.

Two joints commanded harder than the body could deliver:

- **Waist roll/pitch**: 168°/85° peak commands vs 45°/19° peak state
  (26%/22% tracking). The policy was asking for a "lean into the turn"
  the harness wouldn't allow.
- **Ankle pitch/roll**: ±70°/±60° peak commands vs ±25°/±9° peak state
  (~30%/~16% tracking). Expected for walking — ankles always saturate
  on contact. Consistent with the sphere-foot policy's known authority
  limit on real ankles.

These weren't destabilizing — they were the gantry-constrained
operating regime, not policy misbehaviour.

## What had to land before this run could happen

1. **G20-G23 sim-to-sim parity** (2026-05-01 / 02). Without bit-exact
   parity between the C++ deploy and the IsaacLab Python eval, every
   real-robot symptom would be ambiguous between "policy bug", "sim
   gap", and "deploy bug". G20 (6D rotation channel order), G21 (ONNX
   tokenizer layout), G22 (sim-bridge pre-control physics leak), and
   G23 (IMU `qvel` parity) all closed before any locomotion was
   attempted on hardware.

2. **Sphere-foot training run** (h200-iter-{2k…22k}-sphere-feet-
   20260501). Mesh-foot policies failed `walk_forward` in MuJoCo at
   1-2 s; sphere-foot saturated the 30 s eval gate from iter-2000
   onwards. Without this collider change, no walk attempt would have
   survived the first ground contact.

3. **Smooth MC handoff infrastructure** (2026-05-03):
   - **HOLD_FOR_MC + ramp-out** in `x2_deploy_onnx_ref.cpp`. After
     CONTROL ends, deploy ramps targets to MC's `STAND_DEFAULT` pose
     (loaded from `configs/x2_stand_default_pose.yaml`), holds with
     MC-stand stiffness, and only releases the bus after MC has
     escalated all the way back to `STAND_DEFAULT`. No zero-torque
     window, no DAMPING fallback.
   - **`x2_mc_escalator.py`** persistent-client node hammering
     `SetMcAction(JOINT_DEFAULT)` at 20 Hz, using `GetMcAction` as
     ground-truth for success — the previous "trust `code=0`" path
     silently left MC in `PASSIVE_DEFAULT` (motors limp) for 1-2 s.
     Dropped JOINT_DEFAULT dwell from 1.60 s to 0.20 s.
   - **QoS fix** for the takeover detector (sensor data QoS to match
     MC's publishers).
   - **Recorder cancel-trap flush.** Ctrl-C at the safety gate now
     SIGINTs the recorder before the parent shell exits, so an
     aborted run still produces an npz for post-mortem.

4. **`casual_walk_v1.yaml` playlist + bake.** A 13.5-second garage-
   friendly clip wrapping one out-and-back cycle (`anchor_open` →
   `Turn_Start_Walk_0090_003__A020[0:345]` → `anchor_close`) of the
   115.77 s multi-take recording in `bones_seed.pkl`. Robot returns to
   within 12 cm of starting position so the rail gantry doesn't run
   out of travel.

## What this run does NOT prove yet

- **Off-gantry walking is still untested.** The harness suppressed
  waist_roll/pitch and provided yaw resistance at the turn apex. Free
  walking — even forward-only — is the next ladder rung.
- **Reproducibility is N=1.** Walking is high-variance; a single good
  run is encouraging, three is a baseline, ten is a regression target.
- **The 180° turn on the rail.** The gantry held the torso through
  the turn moment, which is exactly what we wanted on a first walk —
  but it also means we don't know yet how the policy handles the
  rotational degree of freedom unconstrained.
- **Only `casual_walk_v1` tested.** `walk_forward` (straight) and
  `relaxed_walk_postfix` (continuous forward) need their own first-runs
  before the catalog is "covered".

## Suggested ladder for the next session

```
Run 1-3: repeat the exact command above (reproducibility, N=3)
Run 4:   --max-target-dev 1.80  (let waist_roll actually reach the policy's commands)
Run 5:   --motion x2_ultra_relaxed_walk_postfix.pkl --max-duration 6  (forward-only, longer cycle test)
Run 6:   --motion x2_ultra_walk_forward.pkl --max-duration 6  (straight walk reference, unfamiliar to the model on hardware)
Run 7:   reduced harness support (gantry slack), same casual_walk_v1
Run 8:   off-gantry forward-only relaxed_walk_postfix (the real test)
```

## See also

- {doc}`2026-05-02_first_iter4000_powered_run` — first stand on the
  iter-4000 checkpoint that opened this whole sequence.
- {doc}`2026-05-02_post_deploy_tuning` — the recorder + tuning preset
  toolchain that surfaced every concrete number above.
- `gear_sonic_deploy/docs/SMOOTH_HANDOFF.md` — architecture of the
  STANDBY → CONTROL → RAMP_OUT → HOLD_FOR_MC → STAND_DEFAULT pipeline
  this run exercised.
- `gear_sonic_deploy/scripts/x2_record_real_run.py --summarize PATH.npz`
  — the recorder summary tool that produced the headline numbers.

---

*"i have a one way rail attached gantry. so the robot can move."* —
operator, 2026-05-03, just before the first powered walk.

It walked.
