# Quest 3 stick-input smoothing

End-to-end runbook for diagnosing, characterising, and fixing the "robot is stable in idle but lurches on forward push" failure mode when teleoping the X2 over VR. Reads top-to-bottom; you can land on the **Tuned defaults & enabling on the rig** section at the end if you only need to flip the switch.

> **Status:** Part 1 (recording tooling) and Part 2 (analyzer, training-distribution extractor, `StickFilter` + wiring, offline sweep, tuned defaults) shipped. The tuned default is `tau = 0.10 s` LPF, no slew clamp, no asymmetric release. See **Tuned defaults** below.

## Why this exists

The kplanner/SONIC stack consumes a 4-D velocity intent `(yaw_rate, vel_x, vel_z, hip_h)` regardless of the source. Two live sources today:

- **Quest VR** ([`quest3_manager_x2.py`](../../../gear_sonic/scripts/quest3_manager_x2.py)) — emits raw deadzone-rescaled stick deflections at 50 Hz. A single thumb flick is a step input into a model that was trained only on smooth human motion.
- **PKL replay** ([`x2_pkl_command_source.py`](../../../gear_sonic/scripts/x2_pkl_command_source.py)) — derives velocity from an 8-frame rolling window over 30 fps mocap. Band-limited, bounded acceleration, smooth.

The only existing input-side smoothing is `_ColdStartVelocityRamp` in [`x2_kplanner.py`](../../../gear_sonic/scripts/x2_kplanner.py) (EWMA `tau ≈ 0.2 s`) — but it resets only on idle→playing transitions, so steady-state stick changes pass through verbatim. That is why the robot is stable in idle but lurches when commanded to move forward.

The fix lives at the manager. Before we add it we need a measurement: capture one operator session with enough maneuver coverage to characterise the *current* VR input statistics, then we can tell whether any candidate filter actually closes the gap to PKL.

## Capture tooling shipped in Part 1

| File | Purpose |
| ---- | ------- |
| [`gear_sonic/scripts/quest3_manager_x2.py`](../../../gear_sonic/scripts/quest3_manager_x2.py) `--quest3-record-to FILE` | New flag. Writes one JSONL row per 50 Hz manager tick capturing the **raw Quest 3 inputs** the manager actually consumed (post-invert axes + buttons + 3pt pose + hand curls + reader fps). One row per tick where a Quest packet was available. Default off. |
| [`gear_sonic/scripts/record_planner_cmd_jsonl.py`](../../../gear_sonic/scripts/record_planner_cmd_jsonl.py) | New script. ZMQ SUB on `planner_cmd:5563`. One JSONL row per published payload, including the **resolved 4-D velocity** computed by the kplanner's `intent_to_velocity` (mirrors `debug_planner_cmd.py` semantics). Source-tags each row as `vr` or `pkl`. |

Both writers fsync on close, so a Ctrl-C in tmux still leaves a structurally valid JSONL.

## Maneuver script (~75 seconds)

The maneuvers below are chosen to exercise the failure modes — especially the **forward-with-twist** segment, which is the case where today's unfiltered stream destabilises the robot the worst. If you have to deviate, prioritise t = 25-35 (fast forward) and t = 60-70 (mixed).

| t (s) | Stage | What the operator does |
| ----- | ----- | ---------------------- |
| 0-10  | Settle / idle | Sticks centered. Robot should hold `idle_stand`. |
| 10-20 | Slow forward | L-stick fwd to ~30%, hold steady. |
| 20-25 | Stop | Sticks centered, observe recovery. |
| 25-35 | Fast forward | L-stick fwd to ~80%, hold steady. |
| 35-40 | Stop | Sticks centered. |
| 40-45 | Lateral L | L-stick left to ~50%, hold. |
| 45-50 | Lateral R | L-stick right to ~50%, hold. |
| 50-55 | Turn L | R-stick left to ~50%, hold. |
| 55-60 | Turn R | R-stick right to ~50%, hold. |
| 60-70 | Mixed (worst case) | L-stick fwd to ~50% while flicking R-stick L↔R. This is the maneuver that lurches the robot today; we want it captured. |
| 70-75 | Stop / safe | Sticks centered, A+B+X+Y chord to disengage. |

There is no rigid timing requirement — the analyzer slices on `mode` and `intent` rather than wall-clock. If you finish in 60 s or 90 s, that is fine.

## Live capture procedure (MuJoCo sim, single host)

This is a **sim-only** runbook: the wrapper brings up MuJoCo viewer + deploy + kplanner + manager + recorder in-process; you only need a second pane for the planner-cmd recorder so it has its own Ctrl-C.

Two panes total. Pick the same `${STAMP}` in both so the filenames line up.

### Pane 1 — Wrapper (sim deploy + planner stack + Quest3 raw capture)

The wrapper reads `QUEST3_RECORD_TO` and forwards it to the manager spawn as `--quest3-record-to`. Set it on the wrapper invocation:

```bash
mkdir -p out/intent_reference/live
STAMP=$(date +%Y%m%d_%H%M%S)
QUEST3_RECORD_TO="out/intent_reference/live/quest3_raw_${STAMP}.jsonl" \
    bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
        --planner kplanner
```

(The wrapper auto-enables continuous-locomotion when `--planner kplanner` is selected; no extra flag needed. If your normal sim invocation has extra flags — e.g. `--mode robocasa-kitchen`, a non-default tuning yaml, etc. — just append them as usual; the `QUEST3_RECORD_TO` env var is independent.)

Confirm the sidecar is open by watching the manager log lines on the wrapper stdout:

```
[quest3-raw] capture -> out/intent_reference/live/quest3_raw_*.jsonl ...
[quest3-raw] N rows captured (mode=LOCOMOTION, fps=72.0)
```

The second line repeats every ~5 s; that is your liveness signal.

### Pane 2 — planner_cmd recorder

```bash
STAMP=$(date +%Y%m%d_%H%M%S)   # set to the same value as pane 1
.venv/bin/python -m gear_sonic.scripts.record_planner_cmd_jsonl \
    --out "out/intent_reference/live/planner_cmd_vr_${STAMP}.jsonl" \
    --rate-hz 40 --print-every 250
```

`--rate-hz 40` arms the watchdog: if the manager goes quiet (below 40 Hz over a 5 s window), the recorder logs a WARN. The manager normally emits at 50 Hz, so 40 is a comfortable floor that catches drops without firing on noise.

### Recording

1. Don the headset, open the WebXR page at `https://<laptop-LAN>:8443/quest3_webxr_app/`.
2. Wait for the `[manager-x2] running. mode=OFF` line in pane 1 — the cue the manager is ticking.
3. Press the A+B+X+Y chord to enter LOCOMOTION.
4. Execute the maneuver script above (~75 s).
5. A+B+X+Y again to disengage.
6. Ctrl-C pane 2 (recorder) first, then Ctrl-C pane 1 (wrapper).

The recorder Ctrl-Cs cleanly via SIGINT and fsyncs the file. The wrapper's shutdown path closes the manager (which also fsyncs the `quest3_raw_*.jsonl`).

Both JSONL files now live in `out/intent_reference/live/`.

## Acceptance check before going to bed

Run these one-liners to confirm the recording is usable:

```bash
# Both files exist and have rows.
wc -l out/intent_reference/live/quest3_raw_*.jsonl \
       out/intent_reference/live/planner_cmd_vr_*.jsonl

# At 50 Hz × ~75 s ≈ 3750 rows each. Anything > 3000 is fine; anything
# < 1500 means a recorder dropped early.

# Timestamps are monotonic in each file.
.venv/bin/python -c "
import json, sys
for p in sys.argv[1:]:
    with open(p) as f:
        ts = [json.loads(l)['t_mono'] for l in f]
    print(p, 'rows=', len(ts), 'monotonic=', all(b >= a for a, b in zip(ts, ts[1:])))
" out/intent_reference/live/quest3_raw_*.jsonl \
   out/intent_reference/live/planner_cmd_vr_*.jsonl

# The planner_cmd JSONL has at least one non-zero forward velocity row.
.venv/bin/python -c "
import json, sys
nz_vz = 0
for line in open(sys.argv[1]):
    rec = json.loads(line)
    if abs(rec['resolved_velocity'][2]) > 0.05:
        nz_vz += 1
print('rows with |vel_z| > 0.05:', nz_vz)
" out/intent_reference/live/planner_cmd_vr_*.jsonl
```

If `wc -l` >= 3000 for each file, all timestamps are monotonic, and `|vel_z| > 0.05` is true for at least ~500 rows, the capture is good. Go to bed.

## Part 2 — Reference distributions and offline sweep

The capture from Part 1 is the only live data this part needs. Everything below runs offline against it.

### Step 1: Extract the training-distribution reference

[`motionbricks/scripts/extract_training_intent_stats.py`](../../../motionbricks/scripts/extract_training_intent_stats.py) computes the per-frame velocity intent for a **curated subset of PKL clips** that covers the maneuver script above 1:1. The subset lives in [`gear_sonic/data/motions/x2_intent_reference_subset.yaml`](../../../gear_sonic/data/motions/x2_intent_reference_subset.yaml) (7 clips spanning forward / backward / lateral / turn / idle).

Two reference flavors are emitted in one pass:

- **Flavor A** — per-frame raw finite difference (`window=2`), matching `NeuralPlannerCore._predict_with_velocity`. This is "what the training data carries per tick."
- **Flavor B** — 8-frame rolling-window intent, matching what `x2_pkl_command_source.py` ships on the wire during live PKL replay. Slightly smoother, the same path the kplanner sees from the PKL source.

```bash
.venv/bin/python motionbricks/scripts/extract_training_intent_stats.py \
    --out out/intent_reference/training_intent_stats.json
```

That writes `out/intent_reference/training_intent_stats.json` containing per-clip + aggregate per-channel statistics (mean, std, all common percentiles, plus per-channel `|d/dt|` percentiles — the step-input metric).

Numbers from the canonical 7-clip subset (aggregate, 840 frames):

| Channel | Flavor A `\|d/dt\|` p99 | Flavor B `\|d/dt\|` p99 | Units |
| ------- | ---:| ---:| --- |
| `vel_z` (forward) | **3.241** | **2.481** | m/s² |
| `vel_x` (lateral) | 6.044 | 5.284 | m/s² |
| `yaw_rate` | 44.632 | 29.634 | rad/s² |

`vel_z` is the channel that matters here. The lateral and yaw channels are already bounded by `continuous_yaw_max = 0.5` on the manager side, so the live VR stream cannot exceed the training band on those even unfiltered (verified below).

### Step 2: Run the offline analyzer + sweep

[`scripts/analyze_planner_cmd_jsonl.py`](../../../scripts/analyze_planner_cmd_jsonl.py) loads the captured `quest3_raw_*.jsonl`, reconstructs the dense 50 Hz velocity stream the kplanner would see (deadzone → continuous sticks → `intent_to_velocity`), sweeps over a `(τ_LPF, slew_max)` grid, and picks the config whose VR `\|d(vel_z)/dt\|` p99 lands closest to the midpoint of the Flavor-A/B band — subject to an operator-feel cap (`τ ∈ [0.05, 0.30] s`, `τ · slew ≥ 0.5` to avoid over-clamping the engaged-stick response).

```bash
.venv/bin/python scripts/analyze_planner_cmd_jsonl.py \
    --vr-raw out/intent_reference/live/quest3_raw_20260531.jsonl \
    --training-stats out/intent_reference/training_intent_stats.json \
    --out-dir out/intent_reference/analysis_20260531
```

Outputs land in `--out-dir`:

| File | Contents |
| ---- | -------- |
| `intent_overlay.png` | Three-row multi-panel: per-channel stick + velocity time series (raw vs recommended-filtered), bottom row shows `\|d/dt\|` p99 for every sweep config with the PKL Flavor A and Flavor B horizontal references overlaid. |
| `intent_stats.md` | Markdown table of per-config percentiles plus the recommended config block. |
| `recommended_config.json` | Machine-readable `StickFilterConfig` for the chosen tuning. |
| `sweep_summary.json` | Every sweep config with full per-channel stats; the data-source-of-truth if the doc tables ever go stale. |

### Step 3: Baseline + tuning result

VR baseline (no filter) for the canonical 2026-05-31 session:

| Channel | VR raw `\|d/dt\|` p99 | PKL Flavor A p99 | Multiplier |
| ------- | ---:| ---:| ---:|
| `vel_z` | **8.904** m/s² | 3.241 m/s² | **2.7×** out of band |
| `vel_x` | 0.509 m/s² | 6.044 m/s² | 0.08× (well under) |
| `yaw_rate` | 2.115 rad/s² | 44.632 rad/s² | 0.05× (well under) |

VR is only out-of-distribution on the **forward channel**. That matches the live failure mode exactly: forward push lurches, lateral and yaw don't.

The sweep's recommended config:

```json
{
  "tau_lpf_fwd_s":  0.10,  "slew_max_fwd_per_s":  "inf",
  "tau_lpf_side_s": 0.10,  "slew_max_side_per_s": "inf",
  "tau_lpf_yaw_s":  0.10,  "slew_max_yaw_per_s":  "inf",
  "return_to_zero_tau_*_s": null
}
```

Effect on the live fixture:

| Channel | VR raw p99 | VR filtered p99 | PKL Flavor A | PKL Flavor B |
| ------- | ---:| ---:| ---:| ---:|
| `vel_z` | 8.904 | **3.144** | 3.241 | 2.481 |
| `vel_x` | 0.509 | 0.319 | 6.044 | 5.284 |
| `yaw_rate` | 2.115 | 1.822 | 44.632 | 29.634 |

Filtered `vel_z` p99 lands at 3.144 m/s² — within the PKL training band (2.48 - 3.24). Lateral and yaw stay well under the band (they were never the problem). Operator-side: a 0.10 s LPF on a 50 Hz stick stream costs ~100 ms of perceptual lag, which is below the threshold where teleop operators report a "sluggish" feel.

### Why LPF-only and not slew-only

Slew-only configurations (`tau = 0`, `slew = 2..6`) over-clamp the forward channel — they put `\|d(vel_z)/dt\|` p99 at 1-3 m/s², below even the smoother Flavor-B reference, which means the operator's full-deflection push is throttled to a slower-than-training acceleration. LPF-only matches the training band almost exactly without artificially capping the operator's intent. Slew clamp remains in the codebase as a backstop (per-channel `--stick-slew-max`); we just don't need it at these tunings.

### Why a `0.10 s` τ and not `0.15` or `0.20`

Inspecting the sweep summary:

| Config | `\|d(vel_z)/dt\|` p99 | Comment |
| ------ | ---:| --- |
| `tau=0.10, slew=inf` | 3.144 | matches PKL Flavor A (3.241) almost exactly |
| `tau=0.15, slew=inf` | 2.322 | undershoots Flavor B (2.481) slightly |
| `tau=0.20, slew=inf` | 1.844 | under-band; over-smoothed |
| `tau=0.30, slew=inf` | 1.311 | well under-band; lag becomes noticeable |

`τ = 0.10 s` is the sweet spot. We could go to 0.15 if the on-robot test still feels jerky and we want to err on the smoother side — Flavor B (the live-PKL-replay path the kplanner already handles cleanly) sits at 2.48, so anywhere in `[0.10, 0.15]` is in-band.

## Tuned defaults & enabling on the rig

The filter ships **disabled by default** (legacy unfiltered behaviour) so this change is risk-free for non-VR launches. Operators who want to validate the tuned config flip it on via three env vars consumed by [`run_x2_quest3_planner_stack.sh`](../../../gear_sonic/scripts/run_x2_quest3_planner_stack.sh):

```bash
QUEST3_STICK_LPF_TAU=0.10 \
QUEST3_STICK_SLEW_MAX=inf \
    bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
        --planner kplanner
```

The wrapper plumbs them through to the manager as `--stick-lpf-tau` / `--stick-slew-max` / `--stick-return-tau`. The manager logs

```
[stick-filter] enabled: tau_fwd=0.100s tau_side=0.100s tau_yaw=0.100s ...
```

on startup; absence of that line means the env var path didn't fire (typo, etc.).

### When to use each knob

| Knob | When to touch |
| ---- | ------------- |
| `QUEST3_STICK_LPF_TAU` | Primary control. `0.10` is the tuned default; raise to `0.15` if forward push still feels jerky on robot (Flavor-B-like response). Don't go above `0.30` — operators report noticeable lag. |
| `QUEST3_STICK_SLEW_MAX` | Backstop. Leave at `inf` unless you observe high-frequency spikes the LPF doesn't cap; then set to `~5.0` to clamp peak acceleration. The current sweep showed no need. |
| `QUEST3_STICK_RETURN_TAU` | Asymmetric "snappy push, gentle release" feel. Default disabled (symmetric). Try `0.30 s` if release-induced overshoot becomes a complaint. |

### Validation procedure (next-day on-robot)

1. Run the stack with the tuned config:
   ```bash
   QUEST3_STICK_LPF_TAU=0.10 \
       bash gear_sonic/scripts/run_x2_quest3_planner_stack.sh --planner kplanner
   ```
2. Capture a second VR fixture (same maneuver script as Part 1) using the same `QUEST3_RECORD_TO` env var. The capture now records the **post-filter** axes because the manager's raw-capture path snapshots `axes_post_invert` before the filter (the filter is applied at the decode site downstream). However, the resolved `planner_cmd_vr_*.jsonl` *does* reflect the filtered stream, so you compare on that file rather than the raw one.
3. Re-run the analyzer:
   ```bash
   .venv/bin/python scripts/analyze_planner_cmd_jsonl.py \
       --vr-raw out/intent_reference/live/quest3_raw_<NEW_STAMP>.jsonl \
       --training-stats out/intent_reference/training_intent_stats.json \
       --out-dir out/intent_reference/validation_<NEW_STAMP>
   ```
   The `vr_raw` row in the sweep output is now the actual on-robot baseline-with-filter. It should match the recommended config's `vel_z` p99 (~3.1 m/s²) within ±0.5 m/s²; if it doesn't, the StickFilter wiring isn't engaging — check the manager log for the `[stick-filter] enabled` line.
4. **Subjective check**: forward-with-twist maneuver (t=60-70 in the script). The lurching should be gone; the gait should look like a smoother continuous walk under twist.

## Follow-ups (deferred)

- **`Quest3Replayer` + `--quest3-replay-from`** — a getter-compatible replacement for `Quest3Reader` that drives the manager from a recorded JSONL, no headset. Would enable repeated `(τ, slew)` sweeps through the full manager+kplanner+SONIC+MuJoCo stack offline. Not needed for the current tuning because the analyzer's offline pipeline (replay → filter → `intent_to_velocity` → metrics) already captures the relevant per-channel dynamics. Worth picking up if a future tuning iteration needs full-stack physics in the loop.
- **(B) live PKL-replay wire capture via `x2_pkl_command_source` + `record_planner_cmd_jsonl.py`** — the analyzer currently reads Flavor B from the same per-frame derivation as Flavor A. A live wire capture (`x2_pkl_command_source --pkl ... | record_planner_cmd_jsonl.py`) would confirm that the kplanner's actual on-wire intent matches the offline-computed Flavor B byte-for-byte. Useful regression check; not strictly required for tuning since the kplanner's `intent_to_velocity` short-circuits on `target_velocity` (i.e., the PKL source ships the same numbers the analyzer extracts offline).
- **Per-channel tuning** — current defaults apply the same `τ` uniformly across `fwd / side / yaw`. The analyzer already shows lateral and yaw are well under the PKL band even unfiltered, so per-channel `τ` could reduce lag on those without changing the forward fix. Easy to add (`StickFilterConfig` already carries per-channel fields; only the CLI plumbing needs splitting).
- **Cold-start ramp Flavor C** — the existing `_ColdStartVelocityRamp` in `x2_kplanner.py` produces a third intent flavor (post-LPF on idle→play transition). The analyzer doesn't currently overlay it because it operates downstream of `planner_cmd` and is event-triggered; reproducing it offline requires modeling the manager's emit pattern. Worth adding to the analyzer if we ever need to compare the StickFilter against the existing ramp directly.

## Artifacts shipped this session

| File | Role |
| ---- | ---- |
| [`gear_sonic/utils/teleop/vr/stick_smoother.py`](../../../gear_sonic/utils/teleop/vr/stick_smoother.py) | `StickFilter` + `StickFilterConfig` (per-channel LPF + slew + optional release tau) |
| [`tests/test_stick_smoother.py`](../../../tests/test_stick_smoother.py) | 14 unit tests covering filter math, edge cases, reset semantics, multi-channel independence |
| [`gear_sonic/data/motions/x2_intent_reference_subset.yaml`](../../../gear_sonic/data/motions/x2_intent_reference_subset.yaml) | Curated 7-clip subset of `x2_ultra_locowalk.pkl` covering the maneuver script |
| [`motionbricks/scripts/extract_training_intent_stats.py`](../../../motionbricks/scripts/extract_training_intent_stats.py) | Flavors A + B extractor → `training_intent_stats.json` |
| [`scripts/analyze_planner_cmd_jsonl.py`](../../../scripts/analyze_planner_cmd_jsonl.py) | Offline (τ, slew) sweep + PNG + markdown + recommended config |
| [`gear_sonic/scripts/quest3_manager_x2.py`](../../../gear_sonic/scripts/quest3_manager_x2.py) | `--stick-lpf-tau` / `--stick-slew-max` / `--stick-return-tau` CLI flags; filter applied in LOCOMOTION mode only, reset on mode transition |
| [`gear_sonic/scripts/run_x2_quest3_planner_stack.sh`](../../../gear_sonic/scripts/run_x2_quest3_planner_stack.sh) | `QUEST3_STICK_LPF_TAU` / `QUEST3_STICK_SLEW_MAX` / `QUEST3_STICK_RETURN_TAU` env vars |
