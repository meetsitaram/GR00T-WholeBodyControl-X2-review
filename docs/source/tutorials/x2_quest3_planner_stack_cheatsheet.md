# X2 Quest 3 Planner Stack — Operator Cheat Sheet

Quick-reference for the **planner-driven Quest 3 → X2** stack launched
by `gear_sonic/scripts/run_x2_quest3_planner_stack.sh`.

This page is the at-a-glance card for everyday teleop and dataset
recording. For the deeper architecture, see:

- [`X2 Quest 3 Planner Stack — System Architecture`](../references/x2_quest3_planner_stack_architecture.md)
  — end-to-end engineering reference: mermaid topology, full ZMQ port +
  topic catalogue, CONFLATE/HWM matrix, boot/shutdown sequencing, and
  the invocation matrix covering wrappers / individual-component launchers / test groups.
- [`X2 Heuristic Locomotion Planner`](../references/x2_heuristic_planner.md)
  — what the planner actually plays under the hood.
- [`X2 Dataset Record and Replay`](x2_dataset_record_and_replay.md)
  — the v0 stationary-arms-only recorder (predecessor to this stack).

---

## TL;DR launch line

```bash
cd ~/Projects/GR00T-WholeBodyControl

# Teleop only (no dataset writes)
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --duration 600

# Record episodes
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 600 \
    --with-record \
    --output-dir data/quest3_x2_planner_phase0/v1 \
    --task "phase 0 stand-and-manipulate"
```

The wrapper boots **deploy → planner → manager → recorder** in order,
prints this cheat sheet to stdout, mirrors manager `[mgr]` log lines
to the foreground, and traps Ctrl-C to do an ordered reverse-shutdown
(recorder flushes parquet → manager → planner → deploy → docker).

---

## Mode chord (any time)

| Chord | Effect |
|---|---|
| **A + B + X + Y** simultaneously | OFF ↔ LOCOMOTION (also returns to OFF from any active mode) |
| **B alone** in LOCOMOTION | → ARM_MANIPULATION |
| **B alone** in ARM_MANIPULATION | → LOCOMOTION |

When the stack is freshly booted the manager is in **OFF**: the planner
is publishing `idle_stand` so the robot just stands. You must press the
4-button chord at least once to enter `LOCOMOTION` before sticks /
arm IK do anything. After the chord, a 0.5 s quiet window suppresses
inputs so the chord-release doesn't kick the robot.

---

## LOCOMOTION mode — sticks drive the planner, no recording

> **Operator-facing direction**: "push the stick the way you want the
> robot to move". With the default polarity (`invert_ly=False`) and the
> default RSI anchor, pushing the L stick forward emits a `back_step`
> command in the planner log — that is **expected**: the curated bins
> are authored in a body frame rotated 180° from the bridge's RSI init,
> so the bin labelled `back_step_half_ft` actually translates the body
> forward in world. See "Stick polarity" below for why.

| Input | Command in planner log | Planner bin | World-frame motion |
|---|---|---|---|
| **L stick fwd** | `back_step / default` | `back_step_half_ft` | One stride **forward** |
| **L stick back** | `fwd_step / default` | `fwd_step_1ft` | One stride **backward** |
| **L stick fwd + A held** | `walk / backward` | `back_walk_standard` | **Continuous walk forward**; release to stop |
| **L stick back + A held** | `walk / forward` | `fwd_walk_standard` | **Continuous walk backward**; release to stop |
| **L stick L / R** | `side_left` / `side_right` | `side_left_step` / `side_right_step` | Single side stride |
| **R stick L / R**, hard (\|rx\| ≥ 0.60) | `turn_left / deg_45` / `turn_right / deg_45` | `turn_*_45deg` | Yaw step |
| **R stick L / R**, hard + **X held** | `turn_* / deg_90` | `turn_*_90deg` | Bigger yaw step |
| Sticks neutral | `idle / default` | `idle_stand` (loops) | |

> **Disabled by default**: `lean_fwd_*` (R stick fwd) and
> `torso_*_30deg` (R stick L/R **soft**) are *replay* primitives —
> the curated bin leans / twists into the pose and immediately blends
> back to standing instead of holding. That made the body flicker
> when operators tried to use them. They're now silently ignored
> (fall through to `idle / default`) unless you re-enable them on
> the manager:
>
> ```bash
> --enable-lean-fwd    # restore R stick fwd graded lean
> --enable-torso       # restore R stick L/R soft torso twist
> ```
>
> The hard turn (R stick L/R, |rx| ≥ 0.60) is **always on** because
> the `turn_*` bins commit to a real discrete yaw step.

### Precedence rules (locked in by `tests/test_intent_decoder.py`)

1. **Y held** → would emit `crouch / medium`, but **crouch is currently disabled** (X2 SONIC tips over on the crouch primitive). Y held is silently ignored.
2. **L stick** wins over R stick: any active L stick beats any R stick.
3. On L stick, **dominant axis** wins: \|ly\| ≥ \|lx\| → fwd/back; otherwise side-step.
4. On R stick, dominant axis wins: \|ry\| ≥ \|rx\|. With lean/torso disabled (the default), the ry-dominant branch returns `idle` and we **do not** fall through to the rx branch — operators pushing the right stick toward an upper corner won't accidentally trigger a turn.
5. **A held** only modifies fwd/back to walk; it has no effect on side / lean / torso / turn.
6. **X held** only modifies hard rx (\|rx\| ≥ 0.60) to a 90° turn; soft rx is ignored (or emits torso when `--enable-torso` is set), regardless of X.

### What "hold the stick" does

Most planner primitives are **single-stride**: one push of the L stick = one `fwd_step`. Holding the stick does **not** keep stepping — you must release and re-push to take another step. The exceptions are:

- `walk / forward` and `walk / backward` (A + ly): planner loops the continuous walk primitive until you release.
- `lean_fwd_*`, `torso_*_30deg`: static pose primitives; the planner blends in and holds. Releasing the stick blends back to standing.

---

## ARM_MANIPULATION mode — arms track VR + recording controls

| Input | Action |
|---|---|
| **A press** | toggle arm IK engage / disengage (your VR wrists drive the X2 arms) |
| **X press** | start episode  *(`--with-record` only; no-op if already recording)* |
| **Y press** | stop & save episode  *(`--with-record` only; no-op if no open episode)* |
| **R thumbstick click** | cycle deploy MuJoCo viewer's fixed cameras (Tab-equivalent; works in LOCOMOTION too) |

**All recording triggers are gated on ARM_MANIPULATION mode.** A held + walk in LOCOMOTION can never accidentally start an episode; X held + 90° turn in LOCOMOTION never fires start. The episode lifecycle (`start` → frames → `save`) is owned by the manager and forwarded to the recorder over ZMQ topic `recorder_cmd`.

**The R thumbstick click is *not* gated on ARM_MAN** — it's active in any non-OFF mode so you can also re-frame the viewer while walking the robot into position. It's idle in OFF (which is consistent with the rest of the manager: OFF means "ignore controller events").

> **Why no chord for start?** The previous `A+B same tick = start` design collided with the `B-single = mode toggle` rule: the same press both opened an episode AND immediately flipped to LOCOMOTION, yanking the planner reference out from under the new episode. Splitting `start` and `save` onto separate one-shot buttons makes `B` unambiguously a mode toggle.

> **Discard mid-episode** is intentionally not bound to any button right now. If you record a bad take you have two options: (a) press **Y** to save it then delete the parquet/mp4 from disk, or (b) Ctrl-C the wrapper and re-engage. The recorder still understands the `discard` wire action, so we can rebind it (e.g. to `A+Y` chord) if it becomes a frequent need.

### Recording workflow

1. Engage stack: **A + B + X + Y** → enter LOCOMOTION
2. (Optional) walk / turn into position with the L / R sticks
3. **B** → switch to ARM_MAN
4. **A** → engage arm IK; your VR wrists now drive the X2 arms
5. **X** → start episode → `[mgr] [X] start episode forwarded to recorder`
6. Demonstrate the task
7. **Y** → stop & save → `[mgr] [Y] save episode forwarded to recorder`
8. Repeat steps 5–7 for more episodes
9. Ctrl-C the wrapper or **A + B + X + Y** to return to OFF; any open episode auto-saves on shutdown

---

## What you'll see in the foreground

The wrapper mirrors the manager's `INFO`-level logs to your terminal as `[mgr] …`:

```
[mgr] [A] arm tracking -> ACTIVE
[mgr] [X] start episode forwarded to recorder
[mgr] [Y] save episode forwarded to recorder
[mgr] mode transition: LOCOMOTION -> ARM_MANIPULATION
```

The four full logs live under `/tmp/x2_quest3_planner_stack-<TIMESTAMP>/`:

- `deploy.log` — MuJoCo + SONIC policy
- `planner.log` — interactive command receipts + bin resolution
- `manager.log` — full button + retargeter trace
- `recorder.log` — episode start/save/discard, parquet write paths
- `manager_sidecar.jsonl` — every `planner_cmd` emitted, with tick + timestamp

Tail any of them in another terminal while the stack is running for live debugging.

---

## Stick polarity (if a direction is wrong)

The operator-facing contract is "push the stick the way you want the
robot to move". That contract is shaped by **two** layers:

1. **Hardware** — Quest 3 controllers report `ly < 0` when the stick is
   pushed forward (away from operator).
2. **Bin world frame** — the curated planner bins were authored in a
   body frame rotated 180° from the bridge's RSI init orientation, so
   the bin labelled `fwd_step` actually translates the body **backward**
   in world; the `back_step_half_ft` bin translates it **forward** in
   world.

Stack defaults are tuned so that the two flips cancel out: forward
stick push → `ly < 0` → `IntentDecoder` emits `back_step` → planner
plays `back_step_half_ft` → robot walks **forward** in world. You will
see the chord `(back_step, default)` in the planner log when you push
the stick forward — that is correct in this configuration.

If you switch to a different RSI anchor / bin set that does **not**
have the world-frame inversion, pass `--invert-ly` to restore the
literal "+ly emits `fwd_step`" mapping:

```bash
./run_x2_quest3_planner_stack.sh ... \
    -- \
    --invert-ly         # bins where fwd_step actually moves +X world
    --invert-lx         # if side-step direction is wrong
    --invert-rx         # if turn / torso direction is wrong
    --invert-ry         # if lean direction is wrong
```

Each manager startup logs the active polarity:

```
[manager-x2] stick polarity: invert_lx=False invert_ly=False invert_rx=False invert_ry=False
```

---

## Tunable thresholds

Set on the manager (passed through to `IntentDecoder`):

| Parameter | Default | Meaning |
|---|---|---|
| `stick_deadzone` | 0.30 | Per-axis deflection treated as neutral |
| `turn_threshold` | 0.60 | Soft / hard rx boundary (torso vs turn) |
| `lean_medium_threshold` | 0.55 | ry boundary between `lean_fwd_small` and `lean_fwd_medium` |
| `lean_large_threshold` | 0.80 | ry boundary between `lean_fwd_medium` and `lean_fwd_large` |
| `chord_debounce_s` | 0.5 | Quiet window after the A+B+X+Y chord puts you in LOCOMOTION |
| `enable_crouch` | `False` | Y-held → `crouch / medium` (currently destabilizes the policy; do not enable in sim until the planner-side fix lands) |

These live in `gear_sonic/utils/teleop/vr/intent_decoder.py::IntentDecoder.__init__`. To expose them as CLI flags on the manager, add them to `ManagerConfig` and the argparse group in `quest3_manager_x2.py`.

---

## Known footguns

- **B in ARM_MAN** flips you back to LOCOMOTION before any per-tick decoder work runs, so the same B press cannot emit anything to the planner — you'll see the mode transition log line then the next tick's idle.
- **X press in ARM_MAN with no open episode** fires `start` (the happy path). **X press in ARM_MAN while already recording** fires another `start`; the recorder logs `[recorder] [X] ignored: already recording` and continues — harmless.
- **Y press in ARM_MAN with no open episode** fires `save` against an empty buffer; the recorder logs `[recorder] [Y] ignored: no active episode` and continues — harmless.

When a save succeeds the wrapper foreground prints **three** `[recorder]` lines so the operator can locate the data without grepping `find`:

```
[recorder] [Y] episode saved: 3977 frames (total saved=1)
[recorder]     parquet -> data/lerobot/x2_pick_place_cube_v0/data/chunk-000/episode_000000.parquet
[recorder]     mp4     -> data/lerobot/x2_pick_place_cube_v0/videos/chunk-000/observation.images.ego_view/episode_000000.mp4
```

The label `[Y]` matches the button you pressed. (Pre-2026-05-13 these said `[X]` because the original chord was X-for-save; the rebind to X=start, Y=save left the recorder labels misaligned for one revision and was cleaned up in the same commit that added the foreground mirror.)
- **X or Y press in OFF or LOCOMOTION** is a no-op. Both buttons only forward `recorder_cmd` when in ARM_MANIPULATION.
- **First push of the chord on a freshly booted stack** triggers an audio-prompt + WAV "engaged" cue if your headset has speakers; this is normal.

---

## Headset audio cues

The manager pushes a short voice prompt to the headset speakers on every mode transition and every recording-lifecycle action. They're rendered as MP3 files in `gear_sonic/utils/teleop/vr/quest3_webxr_app/audio/` and played via a regular `<audio>` element (Quest 3's `speechSynthesis` is unreliable, so we ship pre-rendered audio).

| Event                                    | Spoken prompt        | MP3 key                  |
|------------------------------------------|----------------------|--------------------------|
| Mode → OFF (incl. estop chord)           | "Off."               | `mode_off`               |
| Mode → LOCOMOTION                        | "Locomotion."        | `mode_locomotion`        |
| Mode → ARM_MANIPULATION                  | "Arm manipulation."  | `mode_arm_manipulation`  |
| X press in ARM_MAN — start episode       | "Recording."         | `record_start`           |
| Y press in ARM_MAN — stop & save episode | "Saved."             | `record_save`            |

To regenerate the cache (e.g. after editing `PROMPT_TEXTS`), delete the per-key MP3s under `quest3_webxr_app/audio/` and either restart the manager (which calls `ensure_prompt_audio_files` on Quest3Reader startup) or run:

```bash
python -c "from gear_sonic.utils.teleop.vr.quest3_audio_prompts import ensure_prompt_audio_files; ensure_prompt_audio_files()"
```

If a cue is missing on the headset, check (1) the headset isn't muted in Quest Settings → Sound, (2) the WebXR debug overlay shows `playPromptAudio('mode_*'/'record_*')` log lines on the relevant action, (3) the MP3 file exists on disk under `quest3_webxr_app/audio/`. If gTTS / network are unavailable when the cache is built the WebXR client falls back to `speechSynthesis` (less reliable) for any missing key.

---

## Camera cycling (right thumbstick click → deploy viewer Tab)

Manipulation work benefits a lot from re-framing — picking up a small object reads better from `obj_left` or `obj_right` than from `ego_view`. Pressing the **right thumbstick click** sends a synthetic `Tab` keystroke to the deploy MuJoCo viewer's GLFW window, cycling the same fixed cameras you'd cycle from the workstation keyboard.

| Camera (default order) | What it shows |
|------------------------|---------------|
| `obj_left`             | Wrist-side view biased to the robot's left hand |
| `obj_right`            | Wrist-side view biased to the robot's right hand |
| `rgbd_head_front`      | Head-mounted, near identical to the recorded `ego_view` |
| free orbit             | Mouse-controlled (after Tab cycles past the fixed cameras) |

Active in **LOCOMOTION** and **ARM_MANIPULATION**, idle in OFF. Rate-limited to ~250 ms so a noisy stick can't fire 5 Tabs and overshoot the camera you wanted.

### Setup (one-time)

```bash
sudo apt install xdotool
```

The manager checks for `xdotool` on the first press and prints a one-shot `WARN` if it's missing — your control loop keeps running, the camera cycle just no-ops. Same for missing `DISPLAY` (e.g. `--no-sim-viewer` runs).

### Window-pattern override

The cycler locates the deploy viewer with `xdotool search --name "MuJoCo"`. If you have multiple MuJoCo windows open (e.g. you're running the planner stack and a separate `visualize_motion.py` at the same time), pin the right one with:

```bash
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 600 \
    --with-record \
    --output-dir data/lerobot/x2_phase0_v0 \
    --task "phase 0 demo" \
    -- --viewer-window-pattern "MuJoCo : x2_ultra"
```

(The `--` separates wrapper flags from passthrough flags to `quest3_manager_x2`.)

### Disable for headless runs

```bash
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --no-sim-viewer \
    -- --no-viewer-camera-cycler
```

### "I press the click and nothing happens"

The most common cause is a **stale WebXR client**. The 2026-05-13 patch added `leftStickClick` / `rightStickClick` to the JSON the headset sends each tick; if the headset's Quest Browser is still on a cached pre-patch `index.html` it sends only the 8 face-button keys (no stick clicks) and the manager's rising-edge detector can never fire. Confirm by reading the manager log:

```bash
tail -F /tmp/x2_quest3_planner_stack-*/manager.log | grep -E 'First tracking|R-click|L-click'
```

If the `First tracking data received!` line shows `buttons: {'leftTrigger': ..., 'rightTrigger': ..., ..., 'a': False, 'b': False, 'x': False, 'y': False}` — **eight keys, no `leftStickClick` / `rightStickClick`** — your browser is on the cached old build. Fix: take the headset off, **fully close the Quest Browser tab** (long-press → close, not just background it), reopen `https://<workstation-ip>:8443`, hit "Start VR" again. The Python HTTP server already sends `Cache-Control: no-cache, no-store, must-revalidate + Pragma: no-cache` so a fresh tab reload fetches the new JS.

If the `buttons:` line **does** include `leftStickClick / rightStickClick` keys but pressing the click still doesn't log `R-stick click` in the manager log, the headset / browser combo isn't exposing `gpad.buttons[3]` (rare on older Quest Browser builds). Workaround: use the workstation `Tab` key directly — the cycler is purely additive, it doesn't disable the keyboard binding.

### "manager logs `cycled deploy viewer camera (Tab)` but the camera doesn't actually move"

This is the GNOME-Shell mutter-frames bug fixed 2026-05-13. On GNOME (and a few other compositors that wrap X11 clients in a decorative window) `xdotool search --name "MuJoCo"` returns **two** windows — the real GLFW viewer (class `MuJoCo`) and the compositor's frame wrapper (class `mutter-x11-frames`). The wrapper has the right title but doesn't have a GLFW event loop, so synthetic Tab events to it land in the void. Pre-fix, the cycler picked the wrapper; post-fix it filters by `--classname MuJoCo` first and falls back to a class-filtered name search that excludes `mutter-x11-frames` explicitly.

If you ever hit this again on a fresh compositor / window manager, confirm by running on the workstation:

```bash
xdotool search --name "MuJoCo"
# multiple WIDs returned -> check the class of each:
for wid in $(xdotool search --name "MuJoCo"); do
    printf "  wid=%s  class=%s  name=%q\n" \
        "$wid" "$(xdotool getwindowclassname $wid)" \
        "$(xdotool getwindowname $wid)"
done
```

If the class of the WID the cycler is targeting (look for `[viewer-cycler] excluded ... compositor wrapper window(s)` in `manager.log`) is unfamiliar and you want to add it to the deny-list, add it to `EXCLUDED_WINDOW_CLASSES` in `gear_sonic/utils/teleop/vr/viewer_camera_cycler.py`. If you've rebuilt MuJoCo with a custom WM_CLASS, override the search via `--viewer-window-classname` on the manager.

### Why xdotool (and what's next)

xdotool is the **MVP** path: zero changes to the C++ deploy, zero Docker rebuild. The real fix is a unified `vr_input` ZMQ topic (manager publishes the full controller surface, deploy viewer subscribes and updates `mjvCamera` in-process). Tracked in [`planner_driven_quest3_recorder_mvp_4b00fdab.plan.md`](.cursor/plans/planner_driven_quest3_recorder_mvp_4b00fdab.plan.md) under "Phase 0 follow-ups" → "Unified `vr_input` ZMQ topic". Search the codebase for `TODO(unified-vr-input-topic)` to find the call sites that should migrate.

---

## Robocasa scene mode (record on a table, not a flat floor)

The wrapper can launch the deploy with a pre-built **robocasa scene XML** so the MuJoCo viewer (and recorded `ego_view` frames) shows the table + cube + bowl that manipulation episodes need, instead of the bare flat floor. This piggybacks on the same `--sim-mjcf` / `RobocasaTaskMirror` plumbing `record_x2_dataset.sh` already uses, so the deploy bridge, the recorder, and the renderer all load the **same** MJCF and stay in lock-step on object qpos addresses.

### One-time: build the scene XML

The XML lives at `gear_sonic/data/assets/robocasa_scenes/<env>.xml`. Build it once per env (or after editing the underlying robocasa env):

```bash
# Pick-place cube on a table:
python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceCube

# Pick-place bowl on a table:
python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceBowl
```

The build script writes both the `.xml` and a `.json` sidecar (joint/body/site names, freejoint qpos addresses, instruction text). The wrapper, the deploy bridge, and the recorder all read the sidecar — don't edit the `.xml` by hand.

### Launch

```bash
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 1200 \
    --with-record \
    --output-dir data/lerobot/x2_pick_place_cube_v0 \
    --robocasa-env X2PickPlaceCube
```

Note that `--task` is **omitted**: the recorder auto-fills it from the scene metadata (the env's canonical instruction is what the success oracle is grading against, so we want the dataset to record that exact string). Pass `--task "..."` to override.

For reproducible per-episode object placement (useful for smoke tests / regression diffs):

```bash
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 600 --with-record \
    --output-dir data/lerobot/x2_pick_place_cube_seed42 \
    --robocasa-env X2PickPlaceCube \
    --episode-seed 42
```

### What changes vs. flat-floor mode

| Behaviour                       | Flat floor (`--robocasa-env none`, default) | Robocasa scene                                           |
|---------------------------------|---------------------------------------------|----------------------------------------------------------|
| Deploy MJCF                     | `x2_ultra.xml` (+ OmniHand augmentation)    | `robocasa_scenes/<env>.xml` (table + objects on top)     |
| `--sim-mjcf` forwarded to deploy| no                                          | yes                                                      |
| Recorder loads scene XML        | no                                          | yes — `ego_view` frames show the table + objects         |
| `RobocasaTaskMirror` instantiated | no                                        | yes — per-tick `task.success / task.reward / task.subtask_*` columns |
| Per-episode object randomisation | n/a                                         | yes — recorder PUBs `scene_reset` with fresh qpos at every `start` |
| ZMQ ports bound                 | 5556, 5557, 5563, 5564, 5565                | + 5559 (scene_state PUB) + 5560 (scene_reset SUB)        |
| `--task` required               | yes (always)                                | optional — auto-fills from env's instruction             |
| Finger curl compensation        | OFF (pass `--apply-curl-compensation` to enable) | **ON** (pass `--no-apply-curl-compensation` to disable) |
| Finger thumb-oppose compensation| OFF (pass `--apply-oppose-compensation` to enable) | **ON** (pass `--no-apply-oppose-compensation` to disable) |

**Finger compensations.** Power-grasp pick-and-place on a small cube / bowl is exactly the workload the OmniHand curl + thumb-oppose compensations were tuned for, so the wrapper auto-enables both when you pass `--robocasa-env`. The banner shows the resolved values (`finger comp: curl=on  oppose=on  (robocasa default; pass --no-apply-{curl,oppose}-compensation to override)`) so the operator never has to guess. The compensations apply on the **manager** side (the Retargeter that builds `hand_finger_cmd` lives in the manager in subscribe-mode), not the recorder; previously the wrapper forwarded the flags only to the recorder which silently no-op'd them — fixed 2026-05-13.

Pre-flight refuses to start if `robocasa_scenes/<env>.xml` is missing or if 5559 / 5560 are already bound (a previous run leaked, or an unrelated bridge is up). The error message points at the build command and `--cleanup-only` respectively.

### Port-collision overrides

If you're running two robocasa stacks side-by-side (you probably aren't, but the wrapper supports it), bump the scene ports on the wrapper side; they get forwarded to both the deploy bridge and the recorder so the contract stays internally consistent:

```bash
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --robocasa-env X2PickPlaceCube \
    --scene-state-port 5659 \
    --scene-reset-port 5660 \
    ...
```

### Picking the right `--robocasa-env`

`X2PickPlaceCube` and `X2PickPlaceBowl` both put a single object on a low table within reach of the X2's pre-calibrated arm workspace. They're the only two envs we've validated end-to-end with the deploy bridge + recorder; adding a new env means writing a `RobocasaTaskMirror` subclass, building its scene XML, and re-running the in-sim object-placement sanity check before recording any episodes.

### Custom scene XMLs (rare)

If you're iterating on a scene that lives outside the `robocasa_scenes/` directory:

```bash
gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --robocasa-env X2PickPlaceCube \
    --scene-xml-path /path/to/my_dev_scene.xml \
    ...
```

`--scene-xml-path` overrides the auto-resolved location but the env name is still required so the recorder picks the matching `RobocasaTaskMirror` (the success oracle, subtask schema, and placement initializer all key off the env name, not the XML path).

---

## Bug-hunting checklist if a button doesn't seem to do anything

1. Look at `manager.log` for the rising-edge log: `[Input] A pressed`, `[Input] B pressed`, etc. If the line isn't there, the manager isn't receiving the button — check the WebXR app on the headset.
2. Look for `[mgr] [<button>] <action> forwarded to recorder` in the foreground. If absent, the manager saw the button but the mode gate suppressed it.
3. Look at `planner.log` for `interactive command (<intent>, <magnitude>) from zmq`. If absent, the manager didn't publish; if present, the planner accepted it.
4. Look for `WARNING ... no primitive for command (X,Y) -> bin 'X_Y'; falling back to idle_stand` in `planner.log` — that means the bin name doesn't exist in the curated YAML and the planner is silently skipping the request. (Historical example: `back_step / default → back_step_default` was unmapped pre-2026-05-13.)
5. Look at `deploy.log` for `act_clip_ticks` — if it's keeping pace with `tick`, the policy is at its safety cap most of the time and the apparent immobility is a tracking issue downstream of the planner.

---

## Layout of the running stack

```
   ┌─────────────────────────┐  ┌─────────────────────────┐
   │  Quest 3 (WebXR)        │  │  MuJoCo + SONIC (sim)   │
   │  buttons + sticks       │  │  + viewer               │
   │  + 3-pt pose            │  │                         │
   └────────────┬────────────┘  └────────────▲────────────┘
                │ WS                          │ ZMQ pose 5556
                ▼                             │  (recorder publishes
   ┌─────────────────────────┐                │   the merged stream)
   │  quest3_manager_x2.py   │                │
   │  ─ IntentDecoder        │                │
   │  ─ Retargeter (arm IK)  │                │
   └──────┬──────┬───────────┘                │
          │      │                            │
planner_cmd│      │arm_targets/hand_finger_cmd│
   tcp:5563│      │stream_mode/recorder_cmd   │
          ▼      │ tcp:5564                   │
   ┌─────────────────────────┐                │
   │  x2_heuristic_planner.py│body_pose       │
   │  -> body_pose tcp:5565  ├────────────────┤
   └─────────────────────────┘                │
                                              │
                       ┌──────────────────────┴───┐
                       │  x2_dataset_recorder.py  │
                       │  (subscribe-mode merger) │
                       │  body_pose + arm_targets │
                       │  + hand_finger_cmd       │
                       │  → 'pose' on tcp:5556    │
                       │  → parquet / mp4 writer  │
                       └──────────────────────────┘
```

The wrapper (`run_x2_quest3_planner_stack.sh`) handles port allocation, readiness markers between each spawn, and shutdown ordering.
