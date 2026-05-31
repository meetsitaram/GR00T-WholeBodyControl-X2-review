# 2026-05-13 — Robocasa hand-retargeting unblock (Quest 3 → X2 OmniHand)

> **Session wrap-up.** Four independent bugs were preventing the X2's
> OmniHand from working as a teleop end-effector inside the
> `gr00trobocasa` scene-mode pipeline. They stacked: the wrist
> collision shell silently blocked finger contact, mode switches
> froze trigger input, hand-tracking compensation snapped fingers
> from open to closed across a 5 % raw-curl window, and the resting
> thumb stuck out perpendicular to the palm. Each was reproducible
> in isolation and each is fixed end-to-end with regression
> coverage. After this session the operator can do proportional
> bare-hand teleop in both bare-sim and `--robocasa-env` mode, with
> a natural-looking resting hand pose.

---

## TL;DR

| Symptom | Root cause | Fix |
|---|---|---|
| Fingers don't curl in robocasa scenes (work fine in bare sim) | X2's vestigial `wrist_roll_link` collision mesh — the *pre-OmniHand* "fist" shell — physically blocked OmniHand fingers in MuJoCo | Programmatically disable that mesh in the composed scene XML; verify post-compile that all OmniHand palm primitives still have collision enabled |
| Controller triggers stop closing fingers after switching from hand-tracking → controller mode and back | `FingerSignalFilter`'s NaN-tolerant EMA kept emitting stale XRHand curls when `Quest3Reader.get_hand_curls()` switched to `None`; dispatch stayed on the XRHand path forever | Track `hands.*.source` per side; reset the per-side filter on each `hand ↔ controller` transition |
| Hand-tracking finger curl is bang-bang: 0 % robot motion until ~55-60 % real curl, then snap to 100 % closure with the slightest extra curl | `--apply-curl-compensation` enabled `stretch_finger_curls` with the **bimodal** defaults (`dz=0.35, full=0.40, gamma=5`), which composed on top of the already-proportional `HandRangeCalibration` and re-collapsed it to a binary detector | Flip the compensation defaults to **smooth proportional** (`dz=0.05, full=0.95, gamma=1`) for both `stretch_finger_curls` and `stretch_thumb_oppose`. Bimodal still accessible via explicit per-finger params for the original isolated-curl-detection use case |
| Thumb visually sticks straight out to the side at rest, perpendicular to the fingers — uncanny in the kitchen viewer | OPEN anchors for `thumb_roll` (0°) and `thumb_abad` (10° L / −10° R) sat right next to the hardware hardstop that produces the perpendicular pose | Bias OPEN further into the natural-rest direction: `thumb_roll` 0° → ±12°, `thumb_abad` ±10° → ±35°. CLOSED anchors **unchanged** — no extra hardware-hardstop risk |

The visual change is summarised in two sentences: **the X2's bare
hand now lives in robocasa kitchens with finger contact, smooth
proportional curling, and a natural-looking rest pose.** The
operator can do controller teleop and hand-tracking teleop
interchangeably in the same session without any mid-session
gestures freezing the hand.

---

## What changed

### 1. Disable the pre-OmniHand X2 fist collision mesh in scene XMLs

`gear_sonic/scripts/build_x2_robocasa_scene_xml.py`:

* New helper `_disable_pre_omnihand_x2_fist_collision_mesh(side, root)`
  walks the composed MJCF and zeroes `contype` / `conaffinity` on the
  X2's original `wrist_roll_link` collision geom (the legacy "closed
  fist" primitive shipped with the bare X2 model). The OmniHand
  palm and finger primitives — which live in the per-side OmniHand
  subtree — keep their collision flags untouched.
* The post-compile verification block now asserts (a) every
  OmniHand palm primitive on each side still has `contype != 0`,
  and (b) the X2 `wrist_roll_link` mesh is disabled (`contype == 0
  && conaffinity == 0`). If a future X2 URDF revision changes the
  body name or adds another vestigial collision mesh, the assertion
  fires loudly instead of silently blocking the fingers again.
* All bundled scene XMLs (`X2PickPlaceCube.xml`,
  `X2PickPlaceBowl.xml`; the later-added `X2PickPlaceApple.xml` picks
  up the disable automatically when built via the same helper) were
  rebuilt to embed the disable. Diff is
  6 attribute changes per scene (`contype="1" conaffinity="1"` → `"0"
  "0"` on the two wrist meshes per side).

`tests/test_x2_robocasa_scene_mode.py`:

* New test `test_scene_xml_disables_pre_omnihand_x2_fist_collision_mesh`
  pins the mesh-disable behaviour against a freshly-built scene.
* `test_scene_xml_hand_geoms_use_self_collision_filter` updated to
  assert the X2 wrist mesh is now `(contype=0, conaffinity=0)` (was
  `(1, 1)` in the pre-fix state).

Why this was the right level: doing it in the scene-build step
rather than at MJCF runtime means the disable is **visible in the
XML**, the C++ deploy + the Python ego renderer + the MuJoCo
viewer all get the same disabled mesh, and there's no per-tick
runtime cost. It also leaves the upstream `x2_ultra.xml` untouched,
so the SONIC training distribution is preserved.

### 2. Reset finger filter on Quest `hands.*.source` transitions

`gear_sonic/utils/teleop/x2_retarget_pipeline.py`,
`gear_sonic/scripts/quest3_manager_x2.py`,
`gear_sonic/utils/teleop/x2_dataset_recorder.py`,
`gear_sonic/scripts/teleop_x2_kinematic.py`:

* New `RetargetTickInput.left_hand_source` /
  `right_hand_source` fields plumb the per-side
  `hands.*.source` (`"hand"`, `"controller"`, or `None`) reported
  by `Quest3Reader.get_hand_curls()` through to the retargeter.
* `Retargeter.step()` resets `FingerSignalFilter` on the affected
  side whenever the source string changes from the previous tick.
  Same logic added to `X2DatasetRecorder` (which has its own
  filter instance) and `teleop_x2_kinematic.py` (standalone path).
* `Retargeter.reset_finger_filter()` now also clears the prev-source
  trackers so a manual reset behaves like a fresh start.

`tests/test_retargeter_hand_source_filter_reset.py` (new): exercises
the exact failure mode that prompted the fix — a `"hand"` frame
with finite curls followed by a `"controller"` frame with `None`
curls — and asserts the controller triggers correctly drive
`hand_q` after the transition.

Why this was the bug: `FingerSignalFilter` deliberately tolerates
short XRHand dropouts by holding the last finite EMA value through
NaN frames. When the operator switches to controller mode and the
WebXR client stops emitting curls (`get_hand_curls()` returns
`(None, None, …)`), the filter kept emitting stale curls forever.
That made `if l_curls is not None` evaluate true on every tick, so
the retargeter took the XRHand path with frozen inputs and ignored
trigger / grip from the controllers. The reset on source change is
the cleanest fix because it keeps the dropout-bridging behaviour
intact for genuine 1-3 frame XRHand re-acquires.

### 3. Smooth-proportional compensation defaults

`gear_sonic/utils/teleop/x2_hand_retarget.py`:

| Constant | Old (bimodal) | New (smooth proportional) |
|---|---|---|
| `DEFAULT_CURL_DEADZONE_PER_FINGER` | `(0.25, 0.35, 0.35, 0.35, 0.35)` | `(0.05, 0.05, 0.05, 0.05, 0.05)` |
| `DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER` | `(0.27, 0.40, 0.40, 0.40, 0.40)` | `(0.95, 0.95, 0.95, 0.95, 0.95)` |
| `DEFAULT_CURL_GAMMA_PER_FINGER` | `(5, 5, 5, 5, 5)` | `(1, 1, 1, 1, 1)` |
| `DEFAULT_OPPOSE_DEADZONE / FULL_THRESHOLD / GAMMA` | `0.25 / 0.40 / 3.0` | `0.05 / 0.95 / 1.0` |

With the new defaults `--apply-curl-compensation` /
`--apply-oppose-compensation` apply only a tiny rest-noise gate at
the bottom and a small saturation cushion at the top — linear in
between. So:

* **Operator at half curl → robot at half closure** (instead of 0
  % then 100 %).
* The bimodal "isolated-curl detector" is still accessible by
  passing explicit per-finger `deadzone` / `full_threshold` /
  `gamma` arguments — see
  `test_stretch_finger_curls_bimodal_via_explicit_params` for the
  recipe.
* `tune_finger_curl_compensation.py` still works unchanged; its
  output is just no longer the live default.

Why this was the bug: the bimodal defaults were tuned by maximising
bimodality of the post-stretch motor-command distribution on
2026-05 v3 episodes (a binary "intentional curl yes/no" classifier
job). They composed on top of the per-operator
`HandRangeCalibration.normalize_finger_curls` affine remap and
silently destroyed its proportional response. Calibrated operator
50 % curl → ~36 % raw → below the 0.40 full_threshold → 100 %
robot closure. Operator-perceived "no robot motion until 55-60 %,
then snap" matched the math exactly.

`gear_sonic/scripts/record_x2_dataset.py`: `--help` text for both
compensation flags rewritten to describe the new smooth-proportional
default (and how to recover bimodal explicitly).

`tests/test_teleop_v2_dropout_and_orientation.py`:

* Tests that pinned bimodal behaviour against the live defaults
  rewritten to either (a) pass explicit bimodal params and keep
  asserting bimodal behaviour, or (b) assert the new smooth-
  proportional behaviour. None deleted.
* New `test_default_compensation_no_bang_bang_with_default_calibration`
  is the regression that fails if anyone ever flips the defaults
  back to bimodal: it sweeps the operator's normalised curl 0 → 1
  through `default.yaml`'s `hand_range` and asserts a strictly
  monotone, near-linear motor sweep with half-curl ⇒ half-closure
  within ±12 %.

### 4. Natural resting thumb pose (OPEN anchor shift)

`gear_sonic/utils/teleop/x2_hand_retarget.py`:

| Side | Motor | Old OPEN (°) | New OPEN (°) | CLOSED (°, **unchanged**) |
|---|---|---|---|---|
| LEFT  | `thumb_roll` | 0   | **−12** | −40 |
| LEFT  | `thumb_abad` | 10  | **35**  | 80  |
| RIGHT | `thumb_roll` | 0   | **+12** | +40 |
| RIGHT | `thumb_abad` | −10 | **−35** | −80 |

`thumb_mcp` and all four non-thumb fingers untouched. **No CLOSED
anchor moved**, so real-hardware hardstop risk is identical to
today (every motor still has the same 4–20° headroom from its
hardstop). The OPEN→CLOSED span shrinks (`thumb_abad` 70° → 45°,
`thumb_roll` 40° → 28°), but the operator perceives more visible
thumb-toward-palm motion at half drive because the rest pose is
already biased that way.

`tests/test_quest3_manager_x2_retargeting_parity.py` updated to
mask `thumb_roll` and `thumb_abad` columns when comparing against
the v6 NPZ baseline (the OPEN-anchor shift produces an intentional
~25° per-frame delta on those motors). The other 8 motors still
have to match bit-for-bit, so any *real* regression on the
unchanged motors is still caught by the parity assertion.

Why this was the bug: the previous OPEN values put the thumb 0–10°
across the palm — visually almost perpendicular to the four
fingers, like a hitchhiker. The viewer made it obvious in the
robocasa kitchen scenes (no other props to hide the thumb). Moving
OPEN partway across the palm gives a relaxed natural rest pose and
costs only span, not safety.

---

## Verify in the viewer

Same `--teleop-only` commands as before. The first thing to watch
is the **resting** thumb pose with no input — it should look like
a relaxed hand, not a perpendicular thumb-out-to-the-side.

```bash
# Bare sim, no scene
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --teleop-only \
    --wrist-bypass ik \
    --sim-omnihand \
    --apply-curl-compensation \
    --apply-oppose-compensation \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

```bash
# Robocasa scene
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --teleop-only \
    --robocasa-env X2PickPlaceCube \
    --task "pick up the red cube and drop it into the blue bowl" \
    --wrist-bypass ik \
    --apply-curl-compensation \
    --apply-oppose-compensation \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

What to check, in order:

1. **At rest** (no oppose, no curl): thumb sits ~35° across the
   palm, pad rotated slightly inward. Looks like a relaxed hand.
2. **Slow real-hand curl from open to fist**: robot fingers follow
   continuously, no dead zone at the start, no snap to closed near
   the middle. 50 % real curl → ~50 % robot closure.
3. **Mode switch**: controller → hand tracking → controller. Keys
   keep working after every switch.
4. **In robocasa scene**: fingers actually contact the table /
   props. Thumb tip can reach the index pad on a deliberate
   opposition gesture without "ghosting" through the X2 wrist
   collision shell.

---

## Tests

* `tests/test_teleop_v2_dropout_and_orientation.py`: 58 / 58 pass.
* `tests/test_x2_robocasa_scene_mode.py`: pre-OmniHand mesh-disable
  test green; existing self-collision-filter test updated and
  green.
* `tests/test_quest3_manager_x2_retargeting_parity.py`: 17 / 17
  pass with the thumb-anchor mask in place. Other 8 motors still
  match v6 bit-for-bit.
* `tests/test_retargeter_hand_source_filter_reset.py`: green (new
  regression for the mode-switch lockup).
* Hand-retarget + parity full sweep: 100 / 100 pass together.
* Broader `tests/` sweep: 637 / 637 pass; the one pre-existing
  failure (`test_x2_planner_zmq_publish.py::test_frame_index_is_monotonic_no_drops`)
  reproduces on `main` without any of the above changes — unrelated
  ZMQ timing test, filed separately.

---

## Files touched

| Surface | Path | Nature of change |
|---|---|---|
| Live retargeting | `gear_sonic/utils/teleop/x2_hand_retarget.py` | Smooth-proportional defaults; thumb OPEN anchor shift; comments |
| Retargeter pipeline | `gear_sonic/utils/teleop/x2_retarget_pipeline.py` | `RetargetTickInput.{left,right}_hand_source`; filter reset on source change |
| Quest 3 manager | `gear_sonic/scripts/quest3_manager_x2.py` | Pass `hands.*.source` into `RetargetTickInput` |
| Dataset recorder | `gear_sonic/utils/teleop/x2_dataset_recorder.py` | Track per-side `hands.*.source`; reset filter on transitions |
| Kinematic teleop | `gear_sonic/scripts/teleop_x2_kinematic.py` | Same filter-reset logic for the standalone path |
| Scene builder | `gear_sonic/scripts/build_x2_robocasa_scene_xml.py` | `_disable_pre_omnihand_x2_fist_collision_mesh` + post-compile verifications |
| Recorder CLI | `gear_sonic/scripts/record_x2_dataset.py` | `--apply-*-compensation` `--help` updated for smooth-proportional defaults |
| Robocasa scenes | `gear_sonic/data/assets/robocasa_scenes/X2PickPlace{Cube,Bowl}.xml` | Rebuilt with X2 wrist-mesh collision disabled |
| Tests | `tests/test_teleop_v2_dropout_and_orientation.py` | Smooth-default + bimodal-via-explicit-params split; new bang-bang regression |
| Tests | `tests/test_x2_robocasa_scene_mode.py` | New mesh-disable assertion; updated self-collision test |
| Tests | `tests/test_quest3_manager_x2_retargeting_parity.py` | Mask thumb-anchor-shift columns in v6 parity |
| Tests | `tests/test_retargeter_hand_source_filter_reset.py` | New — pins the mode-switch fix |
| Tutorial | `docs/source/tutorials/x2_dataset_record_and_replay.md` | Anchor table updated; journey log entry pointing here |
| Milestone | `docs/source/user_guide/milestones/2026-05-13_robocasa_finger_fixes.md` | (this file) |

No changes to: deploy code, ZMQ contract, parquet schema,
calibration YAML format, WebXR client, SONIC checkpoints, or VLA
training paths. The diff stays inside the kinematic teleop / scene-
build / hand-retargeting surface area.

---

## Known follow-ups

* **Iterate thumb OPEN by eye.** The `−12° / 35°` values are a
  starting point. If the resting thumb still looks too perpendicular,
  bump `thumb_abad` OPEN further (e.g. 45° L / −45° R). If it looks
  pre-closed, dial back to 25° L / −25° R. Same for `thumb_roll`.
* **Span vs naturalness vs hardware safety is a 2-of-3 trade-off.**
  Today's edit prioritises naturalness + hardware safety at the
  cost of some span. If we ever want to recover that span without
  the natural-rest sacrifice, the next dial is a small `thumb_*`
  CLOSED push (e.g. `thumb_abad` 80° → 88°, leaving ~12° of
  hardstop headroom — roughly matching the `*_pip` 88° / 98 %
  precedent). Not worth doing without a real-hardware soak test.
* **Re-record `default.yaml` `hand_range` periodically.** The
  bundled `oppose_ceiling=0.51 / 0.54` is the operator's *peak*
  oppose value during the v4 calibration session. If most actual
  teleop is "approach, hover, grasp" rather than "literal pinch",
  lowering the ceiling to ~0.30 makes the thumb saturate faster
  on near-contact gestures. Pure YAML edit, zero code risk.
* **Consider revisiting the `*_pip` 88° (~98 % of hardware) anchor
  on real hardware.** That precedent is what the table above leans
  on for the "small CLOSED push is OK" follow-up. Worth a long
  teleop-session soak before we add more motors at that aggressive
  fraction of hardware.
