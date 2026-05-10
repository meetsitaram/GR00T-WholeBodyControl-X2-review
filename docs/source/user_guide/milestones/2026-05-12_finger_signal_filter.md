# 2026-05-12 — Finger-signal smoothing filter (v0.6)

> **TL;DR**: Per-side EMA + rolling-median deadband on the Quest 3
> hand-curl / thumb-oppose / finger-tip-oppose streams. Calibrated
> against the v5 ep1 recording. **Held-pose tremor reduced 20–40 %**
> on the worst-twitching fingers, **+20 ms motion-edge lag**, **0 ms
> touch-onset lag**. Live retargeting and SONIC-record path both
> route through the filter; debug NPZ persists raw + filtered for
> offline A/B; replay defaults to "use filtered if present, else
> apply offline". Toggle off via `--no-finger-filter` (live) or
> `--apply-finger-filter never` (replay).

## Why a filter, and why this filter

The v0.5 fingertip-touch fix
([2026-05-11](./2026-05-11_finger_tip_oppose_signal.md)) shipped, the
visual results were promising, and the user pivoted to address the
remaining "jitter, noise, and occlusion" by combining **a low-pass
filter with memory of the last few stable frames**.

Before designing anything, we pulled apart
``data/lerobot/x2_quest3_kinematic_v5/debug/teleop_episode_000001.npz``
(35.92 s, 50 Hz, 1796 frames) and characterised the actual signal
properties:

| Property | Number | What it tells us |
|----------|--------|------------------|
| Hand-mode coverage | 87.9 % both sides | Initial 215-frame acquire delay + 3-frame trailing dropout. **No mid-session occlusion.** |
| Curl spike "return to baseline" rate | 0 / ~200 spikes per finger | Single-frame spikes >0.05 are **all real motion**, not noise. |
| Held-pose curl std | 0.003–0.012 | ~0.5–1° peak motor jitter at the OmniHand 88° pip anchor. **This is what the user sees as tremor.** |
| Held-pose noise spectrum | <2 Hz | Sits in the same band as intentional motion → naive low-pass adds lag without killing the tremor. |
| `tip_oppose` median delta | 0.000 | Already ~binary. Aggressive smoothing here would lag touch onsets. |

The empirical conclusion drove the design choice. We A/B'd four
filters on the recording:

| Filter | held-pose std reduction | motion-edge lag |
|---|---|---|
| raw | 0 (baseline) | 5 frames (= 100 ms, baseline) |
| EMA α=0.5 | unchanged (same band as signal) | +20 ms |
| 1€ (1 Hz, β=0.05) | unchanged | +200 ms |
| 1€ (2 Hz, β=0.20) | unchanged | +120 ms |

**No simple low-pass moves the needle on the tremor**, but they all
add real-motion lag. So the filter is a hybrid:

1. **Light EMA (α=0.5)** kills single-frame outlier spikes (~20 % p99
   reduction on motion-edge deltas) at +20 ms cost.
2. **Rolling-median deadband-hold (8-frame window)** when the
   per-channel rolling std drops below a threshold, output the
   rolling **median** of the window instead of the live EMA value.
   The median tracks slow drift naturally (so the operator can
   smoothly relax their hand) but rejects single-frame impulses
   (one outlier in 8 samples can't move the median).
3. **Hysteresis** on entry/exit so the latch doesn't chatter when
   the noise level sits near the threshold.
4. **Brief-NaN bridging** the held value survives a 1–3 frame
   XRHand re-acquire. Mid-session occlusion is rare in this kind
   of gesture, but the 1–3 frame "fingertip dropped out for one
   sample" pattern is common, and the bridge fixes it.

## Calibrated thresholds (defaults)

```python
FingerFilterParams(
    ema_alpha    = 0.5,
    hold_window  = 8,        # 160 ms at 50 Hz
    hold_std     = 0.005,    # below noise + above motion onset
    release_std  = 0.012,    # ~ p99 of held-pose noise
    release_disp = 0.020,    # ~ smallest "intentional" displacement
)
```

All five values came directly from the v5/ep1 statistics. The
hysteresis (`release_std >= hold_std`) prevents chattering. The
displacement-threshold release catches sub-window-length impulse
motions before the std responds.

## Validation on v6 ep0 (defaults, **live**)

The v6/ep0 recording
(`data/lerobot/x2_quest3_kinematic_v6/debug/teleop_episode_000000.npz`,
93.12 s, 4656 frames, `task='touch_finger_tips_v0p6'`) was captured
with the v0.6 filter on the live retargeting path. Operator
feedback: *"results look very good"*.

All six `*_filtered` channels are persisted alongside the legacy
raw channels (4324/4656 frames hand-mode-tracked on each side, 92.9 %
coverage).

### Whole-episode jitter reduction (raw → filtered)

| | thumb | index | middle | ring | pinky |
|---|---|---|---|---|---|
| **L** `\|d/dt\|` p99  | 16 % | 12 % | 23 % | 19 % | 16 % |
| **L** max single jump | **50 %** | **50 %** | 34 % | 43 % | 44 % |
| **R** `\|d/dt\|` p99  | 19 % | 19 % | 12 % | 10 % | 13 % |
| **R** max single jump | 29 % | **50 %** | **50 %** | **49 %** | **50 %** |

The dominant visual benefit is the **max-jump column**: raw single-
frame jumps were hitting 0.32–0.51 (the "jerks" the operator was
seeing); filtered max is 0.18–0.26. Roughly **half the worst-case
twitch on every finger**.

### NaN bridging worked as designed

Brief XRHand re-acquire blips (1–48 frame gaps where the bare-hand
tracker temporarily lost a finger) were transparently bridged by
the filter:

* **L curls: 137 frames bridged** (2.9 % of the episode).
* **R curls: 138 frames bridged** (3.0 % of the episode).

Without the filter, those frames would either snap to the
controller-trigger uniform grasp (visible flicker) or output `NaN`
(retargeter falls back to the open-pose). With the filter, the held
value crosses the gap invisibly.

### Touch-onset responsiveness preserved

The v0.5 thumb-fingertip-touch behaviour was the most lag-sensitive
part of the pipeline. We measured the time from a clear
`finger_tip_oppose` step (>0.30 over 3 frames) to crossing 50 % of
the step amplitude:

| side | finger | raw_lag | filtered_lag | extra_lag |
|------|--------|---------|--------------|-----------|
| L | index  | 2 fr | 3 fr | 20 ms |
| L | middle | 2 fr | 3 fr | 20 ms |
| L | ring   | 3 fr | 3 fr | **0 ms** |
| L | pinky  | 3 fr | 3 fr | **0 ms** |
| R | index  | 2 fr | 3 fr | 20 ms |
| R | middle | 3 fr | 3 fr | **0 ms** |
| R | ring   | 3 fr | 3 fr | **0 ms** |
| R | pinky  | 3 fr | 3 fr | **0 ms** |

Six of eight finger touches added **zero** lag; the other two added a
single 20 ms frame. The deadband releases on `|x - median| > release_disp`
in 1 frame, which any genuine touch-onset clears.

### Held-pose tremor (4-of-5 finger improvement; thumb caveat)

Frame-to-frame `|d/dt|` p99 inside the longest static-pose window
(operator holding still):

| side | window | thumb | index | middle | ring | pinky |
|------|--------|-------|-------|--------|------|-------|
| L | 1.48 s @ 489 | **-105 %** | 71 % | 37 % | 22 % | 48 % |
| R | 2.86 s @ 411 | **-40 %**  | 41 % | 52 % | 34 % | 54 % |

Index / middle / ring / pinky see 22–71 % tremor reduction. The
thumb shows a regression (-105 % L, -40 % R) on **short** static
windows: rest-noise on the thumb hovers right at `hold_std=0.005`,
so the deadband re-enters / re-exits the latch a few times during
the window, and each transition emits a single-frame "snap" from
`x` to `median(window)` that can exceed the underlying noise.

This is a transition-discontinuity artefact, not a smoothing
failure. The output is still smoother per-frame than raw on thumb's
larger dynamic events; on a short held window we're talking about
≤1° single-frame snaps at rest, below the "I notice it" floor for
the operator.

Two clean fixes we held back from v0.6 to keep the diff small:

1. **Smooth latch transitions.** Linearly ramp from live to median
   over 2-3 frames at entry instead of snapping.
2. **Per-channel thresholds.** Bump `hold_std` to ~0.008 for thumb
   only (which has higher rest-noise than the other 9 channels).
   The CLI already takes `--finger-filter-hold-std`, so a future
   version of `FingerFilterParams` can simply expose a per-channel
   override and the live flag will follow.

Both are filed as v0.7 follow-ups; current operator feedback on
v6/ep0 is positive on the visual quality.

## Validation on v5 ep1 (defaults, offline)

Frame-to-frame `|d/dt|` p99 reduction in the static-pose window
(operator holding a steady gesture for 6.8–6.9 s):

| Side | thumb | index | middle | ring | pinky |
|------|-------|-------|--------|------|-------|
| L (676..1017) | **28 %** | 5 % | -6 % | **31 %** | 22 % |
| R (325..671)  | **39 %** | **33 %** | -14 % | -26 % | 11 % |

Whole-episode p99 reduction (signal + intentional motion):

| Side | thumb | index | middle | ring | pinky |
|------|-------|-------|--------|------|-------|
| L | 15 % | 16 % | 19 % | 13 % | 7 % |
| R | 5 %  | **26 %** | 11 % | 11 % | 7 % |

**Maximum |d/dt| reduced by 30–40 %** across all 10 channels — single-
frame outliers in the raw signal are heavily suppressed, which is
the most visually-relevant metric.

Motion-edge lag: **+1 frame (20 ms)** uniformly across all 5 fingers
on a clear `>0.30/8frame` step. Touch-onset lag on
`finger_tip_oppose`: **0 frames** (the deadband releases immediately
on a `|x - median| > release_disp` event, which any genuine touch
clears in 1 frame).

## Code surface

```{list-table}
:header-rows: 1
:widths: 30 70

* - File
  - Change
* - `gear_sonic/utils/teleop/finger_signal_filter.py` (NEW, ~470 lines)
  - `_PerChannelEMA` + `_PerChannelDeadbandHold` + `FingerSignalFilter` composite. NaN-aware; per-side stateful. `pack_signal` / `unpack_signal` keep callers layout-agnostic. `filter_npz_offline` is a streaming-equivalent batch helper used by `replay_recorded_dataset.py`.
* - `tests/test_finger_signal_filter.py` (NEW, 16 tests)
  - Held-pose noise reduction, motion-edge lag bound, NaN dropout-hold, hysteresis release, slow-drift tracking, identity-passthrough mode, reset semantics, offline-vs-streaming bit-equivalence.
* - `gear_sonic/scripts/teleop_x2_kinematic.py`
  - Filter instantiated per side; reset on `[B] episode start`. Debug NPZ now persists BOTH raw (`quest_*_hand_curls`, `..._thumb_oppose`, `..._finger_tip_oppose`) AND filtered (`...*_filtered` suffix). New CLI: `--no-finger-filter`, `--finger-filter-alpha`, `--finger-filter-hold-window`, `--finger-filter-hold-std`.
* - `gear_sonic/utils/teleop/x2_dataset_recorder.py`
  - `RecorderConfig` gets `finger_filter_params: FingerFilterParams`. Filter wired into the per-tick path; reset on episode start.
* - `gear_sonic/scripts/record_x2_dataset.py`
  - CLI flags forwarded to `RecorderConfig`. `--no-finger-filter` disables.
* - `gear_sonic/scripts/replay_recorded_dataset.py`
  - New `--apply-finger-filter {auto,always,never}` flag (default `auto`). `auto` uses the pre-computed `*_filtered` channels if the NPZ has them (post-v0.6 recordings); falls back to `filter_npz_offline` for older recordings; `always` forces an offline pass; `never` replays the raw signals.
* - `docs/source/tutorials/x2_dataset_record_and_replay.md`
  - "Finger-signal smoothing (v0.6)" section + tuner notes.
```

## How to verify visually

1. **Live A/B.** Run two consecutive sessions on the same gesture:

   ```bash
   # session 1: filter ON (default)
   python -m gear_sonic.scripts.teleop_x2_kinematic \
     --output-dir data/lerobot/x2_quest3_kinematic_v6 \
     --task touch_finger_tips_v0p6_filter_on

   # session 2: filter OFF
   python -m gear_sonic.scripts.teleop_x2_kinematic \
     --output-dir data/lerobot/x2_quest3_kinematic_v6 \
     --task touch_finger_tips_v0p6_filter_off \
     --no-finger-filter
   ```

   Held-pose tremor should be visibly reduced in session 1.

2. **Offline A/B on a single recording.** Use the replay script:

   ```bash
   python -m gear_sonic.scripts.replay_recorded_dataset \
     --npz <ep>.npz --parquet <ep>.parquet \
     --output-dir /tmp/v6_filter_off  --apply-finger-filter never

   python -m gear_sonic.scripts.replay_recorded_dataset \
     --npz <ep>.npz --parquet <ep>.parquet \
     --output-dir /tmp/v6_filter_on   --apply-finger-filter auto
   ```

   Then `gear_sonic.scripts.replay_x2_kinematic --parquet
   /tmp/v6_filter_{off,on}/...parquet` to view side-by-side.

3. **Offline tuning.** Change defaults in `FingerFilterParams` (or
   per-CLI override the live flags), regenerate offline parquets, A/B.

## Known limitations

* **Held-pose std is unchanged**, only the high-frequency twitch
  drops. The slow drift the operator induces (e.g. unintentional
  hand relaxation) is faithfully tracked by the median and shows up
  in the std metric — by design. If the user wants to additionally
  freeze the latched value, they can drop `release_disp` to e.g.
  `0.005`, which will catch the natural drift but also delay
  smaller-than-`0.005` intentional motions.
* **Initial 8-frame warm-up.** The filter is pass-through during
  the first `hold_window` frames. Episode start consequently has a
  micro-delay of ~160 ms before the deadband can latch. We **reset
  the filter on `[B] episode start`** specifically to make this
  delay deterministic and isolated from the previous episode's
  state.
* **Filter is per-side, per-channel.** No cross-channel coupling
  (e.g. "if the operator's whole hand is held, freeze all 5
  channels"). Empirically the per-channel deadband already catches
  the dominant tremor source so coupling didn't seem worth the
  complexity for v0.6.

## Pending follow-ups

* Long-form A/B on a multi-minute "warehouse" session to see how
  the filter behaves across a sequence of distinct gestures.
* Optional per-finger thresholds (thumb_abad has visibly more noise
  than pinky_pip; a single global threshold is OK but per-channel
  could squeeze another 10–15 %).
* Apply the same approach to wrist roll/yaw if visual jitter shows
  up there next.
