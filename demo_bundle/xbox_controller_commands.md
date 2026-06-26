# Xbox controller demo launcher

Sibling cheat-sheet to [`mc_gesture_capture_commands.md`](mc_gesture_capture_commands.md)
and [`play_pkl_motions_commands.md`](play_pkl_motions_commands.md). Lets
the operator drive locomotion / gesture PKLs from a wired or wireless
Xbox controller during a live demo, with a safety deadman on
locomotion and a single-flight gate that prevents a second clip from
snap-cutting an in-flight one.

The launcher is a thin PUB-side client over the existing
`motion_clip_cmd` ZMQ wire (port `5568`), so it works against either
of the two stacks listed in the prereqs without any wrapper-side
change.

## Prereqs (assumed running in other shells)

```bash
# 1. PC2 daemons (Robogym WiFi, walking-recovery tuning, head locked)
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

# 2. Either the direct-PKL stack ...
./gear_sonic/scripts/run_x2_pkl_direct_stack.sh --pc2-host 192.168.86.32

# ... or the Quest3 / kplanner stack (head cameras optional):
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pc2-host 192.168.86.32
```

Both wrappers bind the `motion_clip_cmd` SUB on `tcp://*:5568`, which
is what the launcher PUB-connects to.

## Inspect controller (verify mapping before the demo)

```bash
.venv/bin/python -m gear_sonic.scripts.play_xbox_controller --list
```

Prints joystick names, axis / button / hat counts, then streams the
live raw state at ~5 Hz. Press each chord button and verify the
expected index lights up (the defaults at the top of the script are
the standard Linux `xpad` driver mapping; swap if your pad differs):

| label  | const         | default |
|--------|---------------|---------|
| A      | `BTN_A`       | button 0 |
| B      | `BTN_B`       | button 1 |
| X      | `BTN_X`       | button 2 |
| Y      | `BTN_Y`       | button 3 |
| L1 (LB)| `BTN_LB`      | button 4 |
| R1 (RB)| `BTN_RB`      | button 5 |
| L2 (LT)| `AXIS_LT`     | axis 2 (>0.5 = held) |
| R2 (RT)| `AXIS_RT`     | axis 5 (>0.5 = held) |
| D-pad  | `HAT_DPAD`    | hat 0 |

## Run the launcher

```bash
# Default: listens on localhost:5568, joystick 0, 2 s calibration window.
.venv/bin/python -m gear_sonic.scripts.play_xbox_controller

# Dry-run (no PUB traffic; just verify chord -> log lines work).
.venv/bin/python -m gear_sonic.scripts.play_xbox_controller --dry-run

# Custom host (e.g. recorder running on another machine):
.venv/bin/python -m gear_sonic.scripts.play_xbox_controller \
    --host 192.168.86.22 --port 5568
```

Once it prints `listener live`, the chord bindings are armed. Ctrl-C
exits cleanly and publishes a defensive `stop` so a mid-clip launcher
crash never leaves the recorder driving the robot off a captured
trajectory.

## Bindings

### Locomotion — D-pad direction + L2+R2 deadman (L1+R1 released)

The L2+R2 "deadman" requirement is the safety against an accidental
D-pad bump. The robot will not walk unless both back triggers are
squeezed.

| chord            | PKL                                                                 | duration |
|------------------|---------------------------------------------------------------------|----------|
| D-pad UP + L2+R2 | `gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl`      | ~13.3 s  |
| D-pad LEFT + L2+R2 | `gear_sonic/data/motions/x2_ultra_relaxed_walk_one_left_turn_v1.pkl`  | varies   |
| D-pad RIGHT + L2+R2 | `gear_sonic/data/motions/x2_ultra_relaxed_walk_one_right_turn_v1.pkl` | varies   |
| D-pad DOWN + L2+R2 | `gear_sonic/data/motions/x2_ultra_relaxed_walk_two_right_turns_v1.pkl` | ~19.8 s  |

D-pad DOWN is the two-right-turns about-face — we don't have a
backwards-walk PKL on the new retarget tracks (see "relaxed-walk v1 —
split into walk-forward + two-right-turns" in
[`play_pkl_motions_commands.md`](play_pkl_motions_commands.md)).

### Gestures — A/B/X/Y with 5 modifier slots each

Each face button has 5 chord slots (bare + 4 single-modifier
variants). Modifier must be held **alone** — never combined:

* **bare** — no shoulders, no triggers held
* **+L1** — L1 held, no R1, no triggers
* **+R1** — R1 held, no L1, no triggers
* **+L2** — L2 held, no R2, no shoulders (locomotion deadman needs *both* triggers)
* **+R2** — R2 held, no L2, no shoulders

Any other combo (`L2+R2`, `L1+R1`, `L1+L2`, etc.) silences face
buttons — so the locomotion deadman (`L2+R2`) and the e-stop chord
(`L1+R1+L2+R2`) can never accidentally fire a gesture.

20-slot grid (currently 9 bound, 11 free):

| button | bare | +L1 | +R1 | +L2 | +R2 |
|--------|------|------|------|------|------|
| **A** | `demo_gestures/hug3_001.pkl` | (free) | (free) | (free) | (free) |
| **B** | `demo_gestures/hand_on_shoulder_001.pkl` | (free) | (free) | (free) | (free) |
| **X** | `demo_gestures/what_can_i_do_001.pkl` | (free) | (free) | (free) | `demo_gestures/chicken_001.pkl` |
| **Y** | `demo_gestures/come_here_001.pkl` | `mc_gestures/bow_001.pkl` | `mc_gestures/right_shake_001.pkl` | `demo_gestures/left_wave_high_001.pkl` | `demo_gestures/right_wave_001.pkl` |

### Emergency stop — L1+R1+L2+R2 (all four)

Always live, even while a clip is busy. Publishes **two** payloads
back-to-back:

1. `{"action": "stop"}` — drops the in-flight clip on the recorder.
2. `{"action": "play", "kind": "gesture", "pkl": "shake_head_001.pkl"}`
   — short head-shake gesture as a visible operator-facing
   acknowledgment that the e-stop landed.

The busy gate is then re-armed for the shake_head duration (~5.7 s)
so a stray chord during the acknowledgment doesn't trample it. A
second e-stop chord during the shake still bypasses the gate and
fires another stop + shake (always-live property).

When the shake_head finishes, the recorder's built-in
`DEFAULT_STAND_POSE_MUJOCO_RAD` idle-stand fallback takes over
automatically (see `_publish_idle` in
[gear_sonic/utils/teleop/x2_dataset_recorder.py](gear_sonic/utils/teleop/x2_dataset_recorder.py)).

To change the acknowledgment clip (or disable it and rely on pure
stop + recorder idle fallback), edit `ESTOP_FOLLOWUP_PKL` at the top
of [gear_sonic/scripts/play_xbox_controller.py](gear_sonic/scripts/play_xbox_controller.py):

```python
ESTOP_FOLLOWUP_PKL = "gear_sonic/data/motions/x2_recorded/mc_gestures/shake_head_001.pkl"
# Set to None to skip the gesture and go straight to recorder idle stand.
```

## Filling the 11 free gesture slots

Edit the `BINDINGS_GESTURES` dict at the top of
[`gear_sonic/scripts/play_xbox_controller.py`](gear_sonic/scripts/play_xbox_controller.py).
The dict has 20 keys (`A`, `A+L1`, `A+R1`, `A+L2`, `A+R2`, then the
same for `B`, `X`, `Y`); 9 are bound, 11 are `None`:

```python
BINDINGS_GESTURES = {
    ...
    "A+L1": None,  # -> e.g. "gear_sonic/data/motions/x2_recorded/mc_gestures/salute_001.pkl"
    "X+L1": None,  # -> e.g. "gear_sonic/data/motions/x2_recorded/mc_gestures/wave_002.pkl"
    "B+R2": None,
    ...
}
```

Paths are repo-relative (resolved against the repo root at startup).
Missing files fail fast at launch via the pre-flight check, so you
won't discover a typo mid-demo.

Other suggested gestures in `gear_sonic/data/motions/x2_recorded/`:

```bash
ls gear_sonic/data/motions/x2_recorded/demo_gestures/
ls gear_sonic/data/motions/x2_recorded/mc_gestures/
```

## Single-flight behavior

While a clip is playing, the launcher ignores any new locomotion /
gesture chord edges and prints a `BUSY t-2.3s -- ignored ...` line so
you can see what was dropped. **The controller also buzzes briefly
(~150 ms) on every BUSY-ignored chord** so you get a tactile reject
without having to look at the terminal. Successful fires and e-stops
are intentionally silent so the *only* haptic cue is "your press was
dropped". Pass `--no-rumble` to disable.

The e-stop chord is the only chord that bypasses the gate: it cancels
the busy timer and re-arms the launcher immediately.

The busy window is sized via `estimate_duration_s` (same helper
`play_locomotion.py` and `play_gesture.py` use to size their own
`time.sleep`) plus a 0.4 s tail buffer to cover the recorder's
snap-back-to-idle settle. Status prints once per second:

```
[xbox 14:32:08] armed
[xbox 14:32:11] FIRE locomotion D-pad-UP -> x2_ultra_relaxed_walk_forward_v1.pkl (~13.3s; gate armed until clip+buffer)
[xbox 14:32:12] BUSY  x2_ultra_relaxed_walk_forward_v1.pkl (t-12.4s)
[xbox 14:32:13] BUSY  t-11.4s -- ignored Y+R2 -> right_wave_001.pkl
[xbox 14:32:25] armed
```

## Troubleshooting

- **`No joysticks detected`** — the OS doesn't see the pad. Plug in a
  USB cable (or pair the wireless dongle), check `ls /dev/input/js*`
  on Linux, and re-run `--list`. The pad must be enumerated *before*
  you launch the script (no hot-plug recovery).
- **D-pad fires but L2+R2 doesn't gate it** — your pad reports the
  triggers on different axes than the default. Run `--list`, press
  the trigger, and note which axis index swings from -1.0 to +1.0.
  Update `AXIS_LT` / `AXIS_RT` at the top of the script.
- **E-stop doesn't fire** — most common cause: one of the shoulder
  buttons (`L1`/`R1`) is on a non-standard index for your pad. Run
  `--list`, hold L1 and R1 one at a time, confirm which button index
  lights up, and update `BTN_LB` / `BTN_RB`.
- **Locomotion fires before I'm ready** — the calibration window
  arms the loop a couple of seconds after launch. Bump
  `--calibration-secs 5` (or any longer value) to give yourself more
  time to walk over to the robot before the loop starts processing
  chord edges.
- **Two clips back-to-back snap-cut anyway** — confirm the launcher
  is the only thing PUB-sending to port 5568. A leftover
  `play_locomotion` or `play_gesture` from another shell will bypass
  the launcher's single-flight gate (the gate is launcher-local).
  `lsof -nPi :5568` or `ss -tnp | grep 5568` to spot stragglers.
- **`STARTUP FAIL: ... missing`** — one of the bound PKL paths
  doesn't exist on disk. Either the file moved or your checkout is
  missing LFS objects; `git lfs pull` or correct the binding.

## What the launcher does NOT do

- Spawn `x2_pc2_daemons.sh` or any stack wrapper — you bring those up
  yourself; the launcher only PUB-sends.
- Hot-plug recovery — if the controller is unplugged mid-session, the
  script will likely hang on a pygame read; Ctrl-C and re-run.
- Per-pad axis remapping for non-Xbox controllers — the constants at
  the top of the script are tuned for `xpad` / SDL2 Xbox controller
  mapping. Use `--list` to identify the right indices and edit.
