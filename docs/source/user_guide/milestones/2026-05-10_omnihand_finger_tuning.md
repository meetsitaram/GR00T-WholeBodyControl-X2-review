# 2026-05-10 — OmniHand finger-tuning iteration (Quest 3 → X2 in MuJoCo)

> **Session wrap-up.** Iterated on bare-hand XRHand → OmniHand
> retargeting until the robot's thumb visually meets the operator's
> thumb during a fingertip-touch gesture. Closed three layers of
> problems (ad-hoc anchor stops, single-signal thumb_mcp drive,
> Quest 3 hand-prior coupling) and surfaced a fourth (non-thumb
> fingertip topology mismatch) for v1. All work was iterated
> against an existing recorded NPZ (`x2_quest3_kinematic_v4`)
> without re-recording.

---

## TL;DR

| Aspect | Before this session | After this session |
|---|---|---|
| Right-thumb closure on touch frames (mean / max) | thumb_roll 46 % / 66 %, thumb_abad 44 % / 63 %, thumb_mcp **28 % / 30 %** | thumb_roll **98 % / 100 %**, thumb_abad **98 % / 100 %**, thumb_mcp **98 % / 100 %** |
| Per-finger PIP closure on full fist | 97–99 % | unchanged (97–99 %) |
| Live default mapping | per-finger affine `hand_range` normalization | unchanged — only thumb anchors and thumb_mcp drive changed |
| Bit-identical to prior parquet on non-thumb motors? | — | **Yes** (max diff = 0.0 rad on index/middle/ring/pinky) |
| Tests | 50 / 50 hand-retarget unit tests passing | 50 / 50 (one renamed and one extended for new behaviour) |

The visual change is summarised in two sentences: **the thumb now
swings across the palm and bends at the knuckle on a deliberate
thumb-fingertip gesture**, where previously it swung but stayed
straight. The four non-thumb fingers still don't reach all the way
to where the thumb-tip lands — that's the topology mismatch
documented at the bottom of this file.

---

## What changed

### 1. Thumb CLOSED-anchor expansion — 50 % → 80 % hardware travel

`gear_sonic/utils/teleop/x2_hand_retarget.py`:

| Anchor | Pre-v0.4 (°) | v0.4 (°) | Hardware range (°) | Travel utilised |
|---|---|---|---|---|
| LEFT  `thumb_roll` CLOSED | -30 | **-40** | -50 .. +10 | 50 % → **83 %** |
| LEFT  `thumb_abad` CLOSED | +60 | **+80** |   0 .. +100 | 60 % → **80 %** |
| RIGHT `thumb_roll` CLOSED | +30 | **+40** | -10 .. +50 | 50 % → **83 %** |
| RIGHT `thumb_abad` CLOSED | -60 | **-80** | -100 .. 0  | 60 % → **80 %** |

Rationale: the upstream agitbot `quest3-bare-hand-control`
constants stopped the thumb at half-travel. Even with a perfect
opposition signal of 1.0, the robot's thumb stayed visibly short
of the other fingertips because the URDF was being commanded only
half-way through its physical range. 80 % (rather than 100 %)
keeps a small mechanical margin for finger / thumb interference
at extreme poses.

### 2. `thumb_mcp` joins `thumb_roll` and `thumb_abad` on the combined drive

Previously `thumb_mcp` lerped on `thumb_flex_curl` (the per-finger
thumb curl), while `thumb_roll` and `thumb_abad` lerped on
`max(thumb_oppose, thumb_flex_curl)`. This was the right
specialisation for fist gestures (where `thumb_flex` is high), but
it failed on thumb-fingertip touches — Quest 3 reports `thumb_flex
≈ 0.30` even on a clear thumb-pad-touches-index-pad gesture
because the IP joint barely folds during opposition (the bend is
in the CMC, which isn't in the XRHand chain).

Changed `_THUMB_OPPOSITION_MOTORS` (now renamed
`_THUMB_COMBINED_DRIVE_MOTORS`) to include `thumb_mcp`. Verbatim
backwards-compatible alias kept for tests and external callers.

### 3. Diagnostic loop — `replay_recorded_dataset.py --calibration`

`gear_sonic/scripts/replay_recorded_dataset.py` already supported
`--calibration` to read a YAML and reproduce the live retargeting
on a recorded NPZ; this session used it heavily to A/B mappings
without putting the headset back on. Workflow:

```bash
# 1. Re-derive a parquet from the existing NPZ + current code:
python -m gear_sonic.scripts.replay_recorded_dataset \
    --npz data/lerobot/x2_quest3_kinematic_v4/debug/teleop_episode_000000.npz \
    --parquet data/lerobot/x2_quest3_kinematic_v4/data/chunk-000/episode_000000.parquet \
    --calibration data/operator_calibrations/default.yaml \
    --output-suffix _hand_range_thumb_oppose_fix \
    --hand-input max

# 2. Replay it visually in the kinematic MuJoCo viewer:
python -m gear_sonic.scripts.replay_x2_kinematic \
    --parquet data/lerobot/x2_quest3_kinematic_v4/data/chunk-000/episode_000000_hand_range_thumb_oppose_fix.parquet \
    --rate 30
```

Ad-hoc analysis script (one-off, not committed) sliced the right
hand's "thumb touch" frames out of the NPZ via raw signal masks —
`oppose ≥ 0.4 AND thumb_flex ≤ 0.3` — and reported per-motor
closure ratios across each candidate parquet. That's where the
98 % numbers in the TL;DR came from.

### 4. Unit-test updates

`tests/test_teleop_v2_dropout_and_orientation.py`:

* Renamed `test_oppose_drives_thumb_roll_and_thumb_abad_independent_of_curl`
  → `test_oppose_drives_all_three_thumb_motors_independent_of_curl`
  and updated the assertion set: pure-opposition gestures now move
  all three thumb motors, including `thumb_mcp`.
* Renamed `test_oppose_high_with_low_thumb_curl_drives_opposition_motors_via_oppose`
  → `_drives_all_thumb_motors_via_oppose` and rewrote the assertion
  to check all three motors close to CLOSED on a low-flex / high-oppose
  signal.
* Added `test_zero_oppose_with_thumb_flex_drives_thumb_mcp_via_flex_curl`
  to cover the symmetric back-compat path: when oppose = 0, all
  three thumb motors still lerp on the raw flex curl alone.
* All 50 tests pass on the kept-and-renamed suite. Comparison and
  renderer suites (`test_replay_finger_curl_comparison.py`,
  `test_x2_omnihand_renderer.py`) pass unchanged.

---

## What we re-confirmed (from data, not code)

* **Quest 3 occludes the thumb when the operator makes a fist.**
  Tight fists report `thumb_flex` of 0.5–0.6 even with the thumb
  fully tucked. The May 10 `max(oppose, thumb_flex)` fold-in
  absorbs most of the under-shoot because a tight fist still gives
  `oppose ≈ 0.5` (thumb-tip near palm centre, close to fingertips).

* **Per-finger affine normalization is the right live default.**
  The earlier `stretch_finger_curls` path maximised bimodality
  (98 %+ frames at <0.05 or >0.95) but operators reported losing
  smooth intermediate variation. Per-operator
  `(floor[i], ceiling[i])` from p05 / p95 of a recorded NPZ
  preserves the operator's reach envelope without pushing
  intermediate gestures toward the endpoints. Confirmed against
  `x2_quest3_kinematic_v4/debug/teleop_episode_000000.npz`.

* **`replay_recorded_dataset.py` is bit-equivalent to live.** When
  pointed at the same NPZ + calibration the live recorder used,
  the regenerated parquet matches the live one to <1e-6 rad on
  every motor. This is the contract that lets us iterate on the
  retargeting without re-recording.

---

## Known open issues (filed, not fixed this session)

### Non-thumb fingertip-to-thumb touch geometry

Even with the thumb fix above, a deliberate thumb-to-index-tip
touch lands the robot's index fingertip somewhere near where the
operator's PIP would be — not where their tip would be. Two
factors stack:

1. **Topology mismatch.** Operator finger has 3 cascaded knuckles
   (MCP + PIP + DIP) summing to ~240 ° at full curl. OmniHand
   non-thumb finger has 1 active flexion DOF (`*_pip` 0..90 °)
   driving 2 mimic-coupled segments (`*_dip = 1.097 × pip`). At
   hardware-max pip the total bend is ~189 °. The robot tip arc is
   geometrically narrower than the operator tip arc.
2. **Quest 3's "fingers move together" prior.** Pairwise curl
   correlations of +0.99 between index/middle/ring/pinky in the v3
   data; isolated single-finger curls cap at ~0.30–0.40 raw. Per-finger
   normalization rescales to the operator's actual range, but raw
   data with limited variance can't be conjured up.

Planned v1 fix: emit a per-finger tip-to-thumb proximity scalar in
the WebXR client (mirror of the existing `thumb_oppose`), and on
the Python side drive each `*_pip` motor on
`max(curls[i], finger_tip_oppose[i])`. Same shape as the May 10
thumb-mcp fix; closes the loop on intentional thumb-finger touches
without compressing intermediate curls.

### Thumb under-shoot at full-fist `thumb_mcp`

Same Quest 3 occlusion issue as above — when the thumb is tucked
into a fist the IP joint reads ~0.5 raw instead of ~1.0. The
opposition fold-in absorbs the visual under-shoot but doesn't
produce a kinematically perfect fist on the robot. Probably not
worth a v1 fix: the fold-in puts the thumb tip against the palm
which is what the gesture is communicating anyway.

### Wrist roll/yaw glitch on controller↔hand mode switch

When the operator picks up or sets down a controller mid-session,
the wrist visibly snaps because the head-frame anchor used by IK
is computed once per pose source. Filed as a v1 task; not
attempted this session.

---

## Files touched

| Surface | Path | Nature of change |
|---|---|---|
| Live retargeting | `gear_sonic/utils/teleop/x2_hand_retarget.py` | Anchor constants, `_THUMB_COMBINED_DRIVE_MOTORS` set, comments |
| Tests | `tests/test_teleop_v2_dropout_and_orientation.py` | Renamed + extended thumb-opposition assertions |
| Tutorial | `docs/source/tutorials/x2_dataset_record_and_replay.md` | §8 hand-retargeting journey log; §11 next steps |
| Milestone | `docs/source/user_guide/milestones/2026-05-10_omnihand_finger_tuning.md` | (this file) |

No changes to: deploy code, ZMQ contract, parquet schema, calibration
YAML format, WebXR client, or Sonic / VLA paths. The diff is
deliberately narrow and confined to the kinematic teleop path.

---

## How to re-verify visually

```bash
cd /home/stickbot/Projects/GR00T-WholeBodyControl
source .venv/bin/activate

# Compare side-by-side with the prior fix (thumb_mcp stays at ~10 %):
python -m gear_sonic.scripts.replay_x2_kinematic \
    --parquet data/lerobot/x2_quest3_kinematic_v4/data/chunk-000/episode_000000_hand_range_calibrated.parquet \
    --rate 30

# This session's mapping (thumb_mcp goes to ~100 % on touch frames):
python -m gear_sonic.scripts.replay_x2_kinematic \
    --parquet data/lerobot/x2_quest3_kinematic_v4/data/chunk-000/episode_000000_hand_range_thumb_oppose_fix.parquet \
    --rate 30
```

Watch the right hand specifically when the operator does the
"thumb touches each finger" gesture. The robot's thumb should
visibly swing across the palm and bend at the knuckle to meet
each fingertip rather than swinging out and staying straight.
