# 2026-05-11 — Per-finger fingertip-to-thumb proximity signal (v0.5)

> **Sequel to [2026-05-10](2026-05-10_omnihand_finger_tuning.md).**
> Yesterday's fix closed the *thumb side* of a thumb-fingertip
> touch gesture. The receiving finger was still under-reporting
> on isolated single-finger curls, so the operator's index pad
> would meet their thumb pad while the robot's index sat halfway
> closed. Today: emit a per-finger thumb-tip-to-fingertip
> proximity signal from the WebXR client, fold it into the
> non-thumb pip drive, and bump the pip CLOSED anchor to its
> hardware ceiling so the kinematic envelope is wide enough to
> consume the new signal.

---

## TL;DR

| Aspect | After yesterday's session | After today's session |
|---|---|---|
| Right thumb closure on touch frames | 98 % (thumb_roll/abad/mcp all at CLOSED) | unchanged |
| Right pinky pip closure on **thumb-pinky touch** frames (raw `pinky_curl ≈ 0.37`) | ~36 % via curl-only drive | 100 % via `max(curl, finger_tip_oppose)` (when fresh recording carries the new field) |
| Non-thumb pip CLOSED anchor | 80° (~89 % of 0..90° hardware travel) | 88° (~98 % of hardware travel) |
| New WebXR payload field | — | `hands.<side>.finger_tip_oppose: [4 floats in [0,1] | nulls]` |
| New debug NPZ key | — | `quest_left_finger_tip_oppose`, `quest_right_finger_tip_oppose` (shape `(N, 4)`) |
| Back-compat on existing v4 NPZ replay | bit-equivalent to live | regen anchors itself to **88°** instead of 80° (~10 % more closure on peak-curl frames); older NPZ has no `finger_tip_oppose` so non-thumb pip stays curl-driven, no kinematic regression |
| Tests | 50 / 50 | 55 / 55 (added 4 dedicated coverage cases for `finger_tip_oppose`, anchor, NaN per-finger, shape validation) |

The visual change: deliberate thumb-to-fingertip touches now
close the receiving finger to the OmniHand tip arc, while
intermediate non-touch frames still show the same smooth curl
proportional control (because `finger_tip_oppose ≈ 0` on those
frames and `max(curl, 0) = curl`).

The verification step requires a **fresh test recording** — v4
NPZ doesn't carry `finger_tip_oppose`, so we can't iterate
visually offline like we did for the thumb fix. The scaffolding
to record + replay is already in place; the user just needs to
run a 30-second test session with a thumb-tip-to-fingertip
sequence.

---

## What changed

### 1. JS-side: `computeFingerTipOppose` in `index.html`

`gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html`:

* New `computeFingerTipOppose(hand, frame, refSpace)` returns a
  4-vector `[index, middle, ring, pinky]` of per-finger
  `dist / palm_width` proximities, normalized through the same
  thresholds as the existing `computeThumbOpposition` (touch =
  d_norm < 0.06 ≈ 0.5 cm; far = d_norm > 0.45 ≈ 3.5 cm). Returns
  `null` if the thumb-tip / palm-width landmarks aren't tracked;
  individual entries can be `NaN` if a specific fingertip dropped
  out for the frame.
* Wired into the per-frame loop alongside `computeThumbOpposition`
  and emitted on the WebSocket payload as
  `hands.<side>.finger_tip_oppose`.

### 2. Python-side: signal parsing + persistence + drive

`gear_sonic/utils/teleop/vr/quest3_reader.py`:

* New `Quest3Reader.get_finger_tip_oppose() -> (left, right)`
  returning numpy `(4,)` arrays in `[0, 1]` (or `None` when the
  hand is untracked, or for old WebXR clients that don't emit
  the field). NaN entries are preserved so callers can fall
  back to the curl path per finger.

`gear_sonic/utils/teleop/x2_hand_retarget.py`:

* `per_finger_grasp_command_from_curls_and_oppose(...)` gains a
  `finger_tip_oppose: tuple[float, float, float, float] | None`
  argument. When provided, each non-thumb pip motor (and the
  matching abad for index/ring/pinky) is driven on
  `max(curls[finger_idx], finger_tip_oppose[finger_idx-1])`.
  `None` is fully back-compat. NaN per-finger entries fall back
  to the curl signal for that finger only.
* `HAND_GRASP_CLOSED_LEFT_DEG[4..9]` and `_RIGHT_DEG[4..9]`:
  index/middle/ring/pinky pip CLOSED anchors moved from 80° to
  **88°** (98 % of the 0..90° hardware travel).

`gear_sonic/scripts/teleop_x2_kinematic.py`:

* Reads `quest.get_finger_tip_oppose()` once per tick.
* Forwards into the retargeter and persists into debug NPZ as
  `quest_left_finger_tip_oppose` / `quest_right_finger_tip_oppose`
  (shape `(N, 4)`).

`gear_sonic/utils/teleop/x2_dataset_recorder.py`:

* Same forwarding for the SONIC-record path so downstream
  recorders pick up the new field too.

`gear_sonic/scripts/replay_recorded_dataset.py`:

* Reads the new NPZ keys when present (logs `finger_tip_oppose
  present in NPZ` with valid-row counts) and falls back to a
  per-finger NaN row when the field is missing (older NPZs).
* Forwards through `_replay_hand_q` so iterating offline against
  a fresh recording produces the same hand q the live recorder
  would have written.

`gear_sonic/scripts/tune_finger_curl_compensation.py`:

* Stale mirrored anchor lists replaced with imports from
  `x2_hand_retarget`. Yesterday's thumb-anchor expansion and
  today's pip expansion now propagate automatically.

### 3. Tests

`tests/test_teleop_v2_dropout_and_orientation.py`:

* `test_finger_tip_oppose_drives_pinky_motors_independent_of_curl`
  — pinky curl 0.37, tip_oppose 1.0 → pinky pip lands at the
  88° CLOSED anchor on both sides; index/middle/ring untouched.
* `test_finger_tip_oppose_zero_matches_curl_only_baseline`
  — passing all-zero `finger_tip_oppose` is bit-equivalent to
  passing `None`.
* `test_finger_tip_oppose_nan_falls_back_to_curl_per_finger`
  — NaN entries fall back to the curl signal per finger; finite
  entries still drive the corresponding finger.
* `test_finger_tip_oppose_invalid_shape_raises` — `(3,)`, `(5,)`,
  `(4, 1)` all raise `ValueError`.
* `test_non_thumb_pip_closed_anchors_at_88_degrees` — guards the
  anchor bump.

All 55 retargeter unit tests pass. Replay / recorder / tuner
suites still pass after the cross-file anchor sync.

---

## How to verify visually

The v4 NPZ doesn't have `finger_tip_oppose`, so we can't iterate
offline against existing recordings. A fresh ~30 s session does
the job:

```bash
cd /home/stickbot/Projects/GR00T-WholeBodyControl
source .venv/bin/activate

# 1. **Reload** the WebXR page on the Quest 3 — a hard refresh
#    is required for the new computeFingerTipOppose JS to take
#    effect.

# 2. Record a short test session with a thumb-to-fingertip touch
#    sequence (index, middle, ring, pinky in turn):
python -m gear_sonic.scripts.teleop_x2_kinematic \
    --output-dir data/lerobot/x2_quest3_kinematic_v5 \
    --calibration data/operator_calibrations/default.yaml \
    --task touch_finger_tips_v0p5

# 3. Confirm the new field exists in the saved debug NPZ:
python -c "import numpy as np; d = np.load('data/lerobot/x2_quest3_kinematic_v5/debug/teleop_episode_000000.npz'); print('finger_tip_oppose' in '\n'.join(d.files))"
# expect: True

# 4. Visually replay against the kinematic viewer:
python -m gear_sonic.scripts.replay_x2_kinematic \
    --dataset data/lerobot/x2_quest3_kinematic_v5 --episode 0
```

Watch the right hand on each thumb-to-fingertip touch. The
receiving finger should close all the way to the OmniHand tip
arc. Intermediate non-touch frames should look the same as
before (smooth proportional control, no binary on/off snap).

---

## Why this isn't fully solved offline

Yesterday's thumb fix was iterable against existing data because
`thumb_oppose` was already in the May v4 NPZ — we could replay
the same recorded frames through different retargeting code and
A/B the results. `finger_tip_oppose` is a brand-new signal: it
needs the WebXR-side computation, which needs the headset to be
held while operating, so iterating visually requires a new
recording.

The full pipeline is wired: JS → Quest3Reader → retargeter →
recorder NPZ → replay. So once a session is recorded, every
subsequent retargeting tweak (e.g. per-operator tip_oppose
calibration, anchor refinement, alternative blend functions)
*can* be iterated offline against that NPZ.

---

## Open follow-ups

* **Per-operator `finger_tip_oppose` calibration.** Hold until
  we have ≥ 2 operators worth of recorded data — the current
  code uses the JS-side normalized signal directly, which
  saturates at ~0.5 cm contact regardless of operator hand
  geometry. Different palm widths could produce different rest-
  bleed levels (the same problem we solved for `thumb_oppose`
  with `oppose_floor` / `oppose_ceiling` in `HandRangeCalibration`).
  We need population data first.
* **Sonic-enabled record + replay.** Still pending (mentioned in
  yesterday's milestone). The kinematic backend now has full
  end-to-end coverage; the SONIC-tracking variant is the
  remaining gap in the teleop / record / replay matrix.
* **Wrist orientation in calibration.** Filed yesterday, still
  open. Independent of the finger work.

---

## Files touched

| Surface | Path | Nature of change |
|---|---|---|
| WebXR client | `gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html` | New `computeFingerTipOppose` + payload wiring |
| WS reader | `gear_sonic/utils/teleop/vr/quest3_reader.py` | New `get_finger_tip_oppose()` |
| Live retargeting | `gear_sonic/utils/teleop/x2_hand_retarget.py` | New arg in `per_finger_grasp_command_from_curls_and_oppose`, pip CLOSED anchor 80° → 88° |
| Live teleop loop | `gear_sonic/scripts/teleop_x2_kinematic.py` | Forward + persist `finger_tip_oppose` |
| SONIC-record loop | `gear_sonic/utils/teleop/x2_dataset_recorder.py` | Forward `finger_tip_oppose` |
| Offline replay | `gear_sonic/scripts/replay_recorded_dataset.py` | Read new NPZ keys; forward to retargeter |
| Tuner | `gear_sonic/scripts/tune_finger_curl_compensation.py` | Replace stale mirrored anchors with import |
| Tests | `tests/test_teleop_v2_dropout_and_orientation.py` | 4 new tests + 1 anchor guard |
| Tutorial | `docs/source/tutorials/x2_dataset_record_and_replay.md` | §8 status update; §11 next-steps + verification checklist |
| Milestone | `docs/source/user_guide/milestones/2026-05-11_finger_tip_oppose_signal.md` | (this file) |
