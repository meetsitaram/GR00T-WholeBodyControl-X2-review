# Thumb does not close in robocasa-scene teleop mode

**Status:** OPEN — first attempted fix reverted (did not resolve the symptom).
**First reported:** 2026-05-13, during `--robocasa-env X2PickPlaceCube` recording.
**Reporter:** operator wearing Quest 3, using `gear_sonic/scripts/run_x2_quest3_planner_stack.sh`.

---

## Symptom

When the operator closes their hand into a fist while teleoping the X2 in
robocasa-scene mode (`--robocasa-env X2PickPlaceCube`), the four non-thumb
OmniHand fingers (`index`, `middle`, `ring`, `pinky`) close as expected, but
the **thumb** stays at (or very near) the OPEN anchor. This breaks
power-grasp pick-and-place because the thumb cannot oppose the cube.

Operator confirmed:

* "the 4 fingers close, so the omnihand data is going, only the thumb has
  issue, and it worked before we enabled robocasa scenes"
* The same gesture in flat-floor (non-robocasa) mode produced acceptable
  thumb closure.

The thumb regression appeared in the same time window we made the robocasa
wrapper auto-enable `--apply-curl-compensation` and
`--apply-oppose-compensation` (see the 2026-05-13 entry in
`docs/source/tutorials/x2_quest3_planner_stack_cheatsheet.md`).

## What we know works in the pipeline

The 4-finger closure proves the end-to-end transport is alive:

```
WebXR (Quest 3 XRHand)
  → quest3_reader.get_hand_curls / get_thumb_opposition / get_finger_tip_oppose
  → quest3_manager_x2._build_retarget_input
  → Retargeter.step  (gear_sonic/utils/teleop/x2_retarget_pipeline.py)
  → per_finger_grasp_command_from_curls_and_oppose
      (gear_sonic/utils/teleop/x2_hand_retarget.py)
  → manager publishes hand_finger_cmd (ZMQ topic)
  → x2_dataset_recorder subscribes, merges into final_pose payload as
    left_hand_joints / right_hand_joints (10-D each) (port 5556)
  → C++ ZmqPoseInputSource reads left_hand_joints/right_hand_joints
      (gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/zmq_pose_input_source.cpp)
  → x2_mujoco_ros_bridge._omnihand_zmq_thread copies into _hand_left/right_active
      (gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py)
  → _apply_omnihand_qpos → apply_active_hand_ctrl writes data.ctrl[thumb_roll/abad/mcp]
      (gear_sonic/scripts/compose_x2_with_omnihand.py)
  → MuJoCo position actuators close the PD loop
```

If any link were dead, all 10 motors (including the 7 non-thumb actuators)
would freeze. Only the thumb is affected, so the bug is *thumb-specific*
within an otherwise-healthy pipeline.

## Hypothesis #1 (TRIED, DID NOT FIX): aggressive thumb stretch zeroes the curl

Inside `per_finger_grasp_command_from_curls_and_oppose`
(`gear_sonic/utils/teleop/x2_hand_retarget.py`), the three thumb actuators
are driven by:

```python
oppose_motor_signal = max(stretched_oppose, stretched_thumb_flex)
```

The per-finger defaults are:

```python
DEFAULT_CURL_DEADZONE_PER_FINGER     = (0.25, 0.35, 0.35, 0.35, 0.35)
DEFAULT_CURL_FULL_THRESHOLD_PER_FINGER = (0.27, 0.40, 0.40, 0.40, 0.40)
DEFAULT_CURL_GAMMA_PER_FINGER         = (5.0,  5.0,  5.0,  5.0,  5.0)
DEFAULT_OPPOSE_DEADZONE        = 0.25
DEFAULT_OPPOSE_FULL_THRESHOLD  = 0.40
DEFAULT_OPPOSE_GAMMA           = 3.0
```

Quest 3's XRHand thumb chain bend tops out around ~0.10–0.20 even on a
tight fist (the thumb's CMC opposition is *not* in the chain). My
hypothesis was:

* `apply_curl_compensation=True` runs `stretch_finger_curls`. Thumb raw
  ~0.20 < deadzone 0.25 → `stretched_thumb_flex = 0`.
* `apply_oppose_compensation=True` runs `stretch_thumb_oppose`. For a
  casual fist (no explicit thumb-tip touch) `oppose ~ 0.10` < deadzone
  0.25 → `stretched_oppose = 0`.
* `max(0, 0) = 0` → all three thumb motors pinned at OPEN.

This *would* explain why flat-floor mode (where both compensations default
OFF) worked: raw signals reach the `max()`, giving `max(0.10, 0.20) = 0.20`
~ 20 % closure.

### Fix attempted

Added a `stretch_thumb: bool = True` knob to `stretch_finger_curls` and
called it with `stretch_thumb=False` from
`per_finger_grasp_command_from_curls_and_oppose` and
`per_finger_grasp_command_from_curls`, so the thumb passes through raw
while the four fingers are still amplified.

### Result

**Did not fix the symptom.** Operator re-tested in robocasa mode and
reported the thumb still does not close. Reverted via `git checkout` on
`gear_sonic/utils/teleop/x2_hand_retarget.py` and
`tests/test_teleop_v2_dropout_and_orientation.py`. The cheat-sheet
paragraph documenting the (now-reverted) exemption was also removed.

This means the thumb-suppression cause is **somewhere else**, or there is
a *second* failure mode in series that we have to clear first.

## Things to investigate next

Order roughly by what's cheapest to verify.

### 1. Confirm what the manager is actually publishing for the thumb

Add a one-shot dump on the manager side of the per-tick `left_hand_q` /
`right_hand_q` (the 10-D `hand_finger_cmd` payload, indices 0/1/2 are
`thumb_roll`/`thumb_abad`/`thumb_mcp`) when the operator gestures.

* If thumb indices are non-zero → the bug is *downstream* of the manager
  (recorder merge, deploy bridge, MuJoCo actuator wiring).
* If thumb indices are zero → the bug is in the retargeter or upstream
  (WebXR raw values, `quest3_reader`, `_build_retarget_input`).

The manager already has `--verbose` but it doesn't dump per-tick hand
slots. Cheap one-liner: log
`hand_finger_cmd[left][0:3]` and `hand_finger_cmd[right][0:3]` once per
second from `quest3_manager_x2.py` near the `self._publish_hand_cmd(...)`
call (or wherever it sends the topic).

### 2. Confirm the deploy bridge actually applies thumb actuator commands

The bridge does:
```python
apply_active_hand_ctrl(self.mj_data, self._omnihand_layout,
                       left_active=left, right_active=right)
```
which iterates `layout.active_actadr[side]` and writes `data.ctrl[aid]`.
`ACTIVE_FINGER_JOINTS` in `gear_sonic/scripts/compose_x2_with_omnihand.py`
puts thumb at indices 0/1/2 of the 10-D vector.

Verify with `mj_name2id` that the thumb actuators (`left_thumb_roll_act`,
`left_thumb_abad_act`, `left_thumb_mcp_act`, plus right side) actually
exist in the **augmented robocasa scene MJCF** that the bridge compiles
when given `--sim-mjcf <robocasa_scene>.xml`. If the robocasa scene XML
overrides or replaces the OmniHand actuator block, the thumb actuators
might be missing or bound to a different body. The 4 fingers working would
still be consistent with that if the scene XML happened to keep
non-thumb actuators intact.

Quick check: dump `layout.active_actadr['left']` in
`_omnihand_zmq_thread` once after the layout is resolved. The list should
have 10 valid actuator IDs (>= 0). If thumb IDs are -1 or missing entirely
this is the bug.

### 3. Confirm the WebXR is sending thumb curls > 0

Open the WebXR client in the headset and inspect what it sends for the
thumb element of `hands.left.curls` / `hands.right.curls` and the
`hands.{left,right}.oppose` scalar. If the thumb chain bend is being
reported as 0 even when the operator visibly bends their thumb, the bug
lives in `computeFingerCurls` / `computeThumbOpposition` in
`gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html`, not in any
backend code.

### 4. Per-operator calibration may be silently zero-ing the thumb

`per_finger_grasp_command_from_curls_and_oppose` accepts `curl_floor` /
`curl_ceiling` / `oppose_floor` / `oppose_ceiling` from
`OperatorCalibration` (`gear_sonic/utils/teleop/operator_calibration.py`).
If the loaded calibration has the thumb's `floor` >= the operator's
typical raw thumb_flex, `normalize_finger_curls` will push the thumb to 0
**before** the (now-irrelevant) stretch step. Inspect
`data/calibration/<operator_id>.yaml` and check the thumb row of
`hand_range.left/right.floor` / `.ceiling` / `.oppose_floor` /
`.oppose_ceiling`.

### 5. The recorder might be overwriting thumb slots

`x2_dataset_recorder.py` handles `hand_finger_cmd` in
`update_hand_finger_cmd` and re-publishes via `_publish_pose` as
`left_hand_joints` / `right_hand_joints`. Confirm those payloads keep all
10 elements intact (specifically indices 0/1/2 for the thumb). A previous
revision of the recorder zero-padded or dropped slots in some code paths;
re-grep for any `[3:]` or `[1:]` slicing on the hand vectors.

### 6. The wrapper's flag forwarding may be inconsistent

`run_x2_quest3_planner_stack.sh` forwards `--apply-curl-compensation` /
`--apply-oppose-compensation` to *both* the manager and the recorder
(2026-05-13 fix). Re-confirm the manager log line at startup actually
shows `apply_curl_compensation=true` / `apply_oppose_compensation=true`
when launched in robocasa mode. If for some reason these are still OFF
on the manager side, the failure mode is different from what we
hypothesised in #1 above and we'd need to look at why a casual fist on
the operator's hand does not produce *any* thumb closure even with
compensations OFF. (Pre-robocasa baseline produced ~20 %.)

## Reproduction recipe

```bash
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
  --robocasa-env X2PickPlaceCube --with-record \
  --output-dir data/lerobot/x2_pick_place_cube_v2
```

1. Don the headset, calibrate as usual.
2. Toggle `B + A` to enter `ARM_MANIPULATION`.
3. Make a fist with either hand. Watch the MuJoCo viewer window.
4. **Expected:** all 5 robot fingers curl toward the palm, thumb included.
5. **Observed:** index/middle/ring/pinky curl; thumb stays at the OPEN
   anchor.

For comparison the pre-regression baseline can be reproduced by dropping
the `--robocasa-env` flag (and either omitting `--with-record` or also
passing `--task casual_walk`):

```bash
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh
```

Same fist gesture should produce ~20 % thumb closure.

## Files touched (and reverted) on the first attempt

* `gear_sonic/utils/teleop/x2_hand_retarget.py` — added
  `stretch_thumb: bool = True` knob to `stretch_finger_curls`; called
  with `stretch_thumb=False` from
  `per_finger_grasp_command_from_curls_and_oppose` and
  `per_finger_grasp_command_from_curls`. **Reverted.**
* `tests/test_teleop_v2_dropout_and_orientation.py` — added two
  regression-pin tests
  (`test_stretch_finger_curls_thumb_exemption_passes_thumb_through_raw`,
  `test_per_finger_oppose_command_thumb_closes_with_modest_flex_under_compensation`)
  and adjusted two pre-existing tests whose inputs incidentally relied on
  the old (regression-driven) behaviour. **Reverted.**
* `docs/source/tutorials/x2_quest3_planner_stack_cheatsheet.md` —
  appended a "Thumb-flex stretch exemption (2026-05-13)" paragraph.
  **Removed** (rest of file untouched, file was already untracked from a
  previous session).

To recover the (reverted) implementation if you want to compare against a
new attempt, pull from the prior assistant turn in this chat session or
re-derive the patch from the description above.
