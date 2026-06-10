# 2026-06-09 — VLA bridge wire-shaping iteration & limit-cycle diagnosis

> **Session focus.** Pick-and-place soda-can VLA on real X2 was producing
> visible arm oscillations late in each run. Goal: characterize the
> oscillation, find a bridge-side wire-shaping config that suppresses it
> without killing the legitimate arm-rise motion, and capture what was
> learned for follow-up tuning. Code changes were small; the bulk of
> the session was empirical tuning with offline FFT analysis of the
> recorded LeRobot rollouts.

---

## TL;DR

| Aspect | Status |
|---|---|
| Discovered a slowly-exponential 2.5 Hz limit cycle in the bridge wire that grows after the arm reaches a holding pose (doubling time ~2-3 s). Source is **internal to the bridge** (pre-SONIC wire shows the growth), not SONIC. | Diagnosed (FFT + spectrogram of recorded rollouts). |
| Introduced `--vla-raw` launcher flag to disable wire-shaping for diagnosis; then **excluded `--vla-max-action-il`** from the raw zeroing list after the first raw run blew the wire to 90+ rad (proprio `last_action` echo runaway in <10 chunks). | Fixed in `gear_sonic/scripts/run_x2_vla_runtime.sh`; documented in `pick_place_commands.md`. |
| Tuning sweep: v1 (LPF=10, step=0.10) → limit cycle; v2 (LPF=3, step=0.04) → bridge stable but **PC2 SONIC tracker independently unstable** when fed near-constant wire; v3 (LPF=5, step=0.07) → both stable but holding pose is "too quiet". | Three runs recorded as ep1–ep3 in `data/lerobot/x2_pick_and_place_soda_can_n17_50k_v1_rollouts`; manual perturbation test on v3 confirmed loop gain < 1 (arms damp back when pushed). |
| Recorder data-fidelity caveat surfaced: in `--vla-subscribe-mode`, `observation.state` is just a copy of the wire, **not** the measured body. Useful for analyzing the bridge but cannot answer "did the real robot move." | Filed as a follow-up; recorder needs a parallel subscriber to PC2's `x2_debug` stream to capture measured body in VLA rollouts. |

---

## Run command (current best, "v3 balanced")

```bash
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --prompt "pick up the mini soda can with your left hand and place it in the open black container on the right" \
    --vla-max-wire-dev-from-body 1.5 \
    --vla-target-lpf-hz 5.0 \
    --vla-future-lpf-hz 5.0 \
    --vla-hand-lpf-hz 10.0 \
    --vla-max-wire-step 0.07
```

Add `--with-record --output-dir data/lerobot/x2_pick_and_place_soda_can_n17_50k_v1_rollouts --task "<prompt>"` to capture for offline FFT.

---

## The limit-cycle finding

Recorded `ep1` at config v1 (LPF=10, step=0.10) for ~15 s, then ran
offline FFT on the **pre-SONIC** bridge wire (`action.body_q_mj_pre_sonic`
column in the parquet). Spectrogram:

* Time-domain trace on `left_shoulder_roll_joint` starts at ~0.30 rad
  with mild noise; after `t=8 s` a clean sinusoid emerges and grows
  exponentially to ±0.10 rad amplitude by `t=14 s`.
* FFT shows a **single dominant peak at 2.5 Hz** that brightens in
  every successive 2-s rolling window — doubling time ~2-3 s.
* Chunk cadence in this run was 121 ticks/chunk = 0.41 Hz, so 2.5 Hz
  is roughly the 6th harmonic of the chunk rate — **inside** the
  policy's 40-step decoded chunk, not a chunk-boundary fingerprint.

The growth is present in pre-SONIC data, so the source is upstream
of PC2's SONIC tracker. Most likely path: the policy is producing
chunks containing ~6 cycles of 2.5 Hz oscillation, the bridge's
10 Hz LPF passes them through almost unattenuated (-0.25 dB at
2.5 Hz), and the action-IL echo into proprio's `last_action`
slowly amplifies the AC component over many chunks. The arm's
"holding pose" phase has small commanded-motion energy, so the
limit-cycle eventually dominates.

## The `--vla-raw` near-miss

To prove the limit cycle was bridge-shaping vs. policy-intrinsic, we
added `--vla-raw` to the launcher to zero out every wire-shaping
knob. The first raw run (`/tmp/x2_vla_runtime-20260609_090317`) blew
up immediately: wire commanded joints to 90+ rad within 50 chunks.
Root cause: the raw flag was zeroing **`--vla-max-action-il`** as
well, which is not a wire shaper — it's a clamp on the policy's
`last_action` proprio echo that matches training's
`action_clip_value=20` headroom. With it disabled, the proprio
feedback loop diverged within milliseconds.

Fix in `run_x2_vla_runtime.sh`: `VLA_MAX_ACTION_IL` is excluded
from the `--vla-raw` zero list and stays at its 8.0 default. The
safety banner now reads `[--vla-raw: WIRE FILTERS OFF; action-IL
clamp KEPT to prevent proprio runaway]`. Verified the next raw
run stayed sane (wire deltas in single-digit rad range).

---

## Wire-shaping tuning sweep

All four runs same checkpoint + same prompt + recorded as episodes
0–3 in the rollouts dataset. Analyzed offline with
`/tmp/x2_vla_analysis/sweep_v1_v2_v3.png`.

| Run | Config | Bridge wire | PC2 SONIC | Verdict |
|---|---|---|---|---|
| 091400 (ep0) | LPF=10, step=0.10 | calm early (7.7 s) | calm | OK but too short to trigger limit cycle |
| 091517 (ep1) | LPF=10, step=0.10 | **2.5 Hz limit cycle grows 23× over 15 s** | amplifies (×1.6) | unstable |
| 092504 (ep2) | LPF=3, step=0.04 | flat (limit cycle killed) | **independently unstable**, post-SONIC RMS = 0.07 rad on L_shR while pre-SONIC RMS = 0 | new failure mode |
| 093004 (ep3) | LPF=5, step=0.07 | stable, low motion | stable | best so far but operator reports holding pose lower-amplitude than v1 |

### Lessons

* The 10 Hz LPF was leaving the closed-loop bridge-policy gain > 1 at
  the natural feedback frequency (2.5 Hz). LPF cutoff is the lever
  for loop gain.
* The per-tick velocity cap (`--vla-max-wire-step`) is an **amplitude
  backstop**, not a loop-gain knob. Tightening it from 0.1 → 0.04
  capped the limit-cycle peak velocity from 7 rad/s to 3.9 rad/s
  but didn't prevent the cycle from forming.
* SONIC's PD tracker on PC2 has its own stability margin that
  depends on the wire's update bandwidth. v2's 3 Hz LPF produced a
  near-constant wire that SONIC's controller couldn't track
  smoothly, so it started overshooting on its own — a new failure
  mode not present in v1.
* v3's 5 Hz LPF is the current best compromise: at 2.5 Hz it
  attenuates by -1 dB (loop gain ~0.89, stable margin), but still
  has enough bandwidth that SONIC tracks normally.
* Manual perturbation test on v3 (operator physically pushed the
  arm to swing it): oscillation increased momentarily and subsided
  — confirms loop gain < 1.

---

## Recorder data-fidelity caveat (filed for follow-up)

While investigating, found that `observation.state` in
`--vla-subscribe-mode` rollouts is **byte-equal to the wire**
(`body_RMS == wire_RMS == 0.0738`, correlation = +1.000, lag = 0
on `L_shR`). The recorder doesn't subscribe to PC2's `x2_debug`
stream in this mode, so it has nothing to put in
`observation.state` except the bridge's own commanded value.

This means:
* The rollouts dataset captures **what the bridge commanded**, not
  what the robot did.
* Cannot use the rollouts to compare "wire vs. measured" or to
  verify that SONIC's correction landed.
* Loop-shape diagnosis (limit-cycle FFT) still works because the
  bridge wire shows the growth on its own.

Fix is small and lives in `gear_sonic/utils/teleop/x2_dataset_recorder.py`:
in `_run_subscribe_mode`, also subscribe to the `x2_debug` topic
on PC2 (`tcp://<pc2>:7891`?) so `observation.state` reflects
measured body. Out of scope for this session.

---

## What's left for the next session

1. **Confirm v3 is the right baseline.** Record a longer (30+ s)
   run with `--with-record`, FFT the result. If 2-3 Hz peak energy
   stays flat or trends down, lock v3 as the working config.
2. **The "arm doesn't reach the can" problem is separate from
   wire shaping.** All four runs above had the arm raise to ~25-30°
   shoulder pitch and then hold; none progressed to the descent
   phase. Likely a training-data or prompt issue. Suggested
   experiments:
   - Manually move the can into the arm's pre-grasp position and
     see if the policy completes the grasp.
   - Replay the training episodes to confirm the typical
     descend-and-grasp trajectory exists in the data.
   - Try a more imperative prompt ("reach down and grab the can
     NOW") once the arm is at hold-pose.
3. **Recorder data-fidelity fix.** Wire `x2_debug` subscriber into
   the VLA subscribe-mode recorder so `observation.state` reflects
   measured body. Needed before we can quantitatively answer
   "did the real arm follow the wire."
4. **Optional: harder action-IL clamp** (`--vla-max-action-il 4.0`
   instead of 8.0) as another loop-gain reduction lever, in case
   the LPF alone proves marginal in longer runs.

---

## Files touched

* `gear_sonic/scripts/run_x2_vla_runtime.sh` — `--vla-raw` flag,
  conflict detection on operator-set wire knobs, exclude
  `VLA_MAX_ACTION_IL` from the raw zeroing list, safety banner
  update.
* `pick_place_commands.md` — three recipe variants (v1 loosened,
  v2 anti-limit-cycle aggressive, v3 balanced) plus `--vla-raw`
  reference with explicit list of knobs kept ON.
* `/tmp/x2_vla_analysis/sweep_v1_v2_v3.png` — comparison plot
  across all four episodes.
* `/tmp/x2_vla_analysis/spectrogram.png` — proof-of-instability
  spectrogram for the ep1 (v1) limit cycle.
