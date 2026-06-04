# X2 Quest 3 Planner Stack — Operator Cheat Sheet

Quick-reference for the **planner-driven Quest 3 → X2** stack launched
by `gear_sonic/scripts/run_x2_quest3_planner_stack.sh`.

This page is the at-a-glance card for everyday teleop and dataset
recording. For the deeper architecture, see:

- [`X2 Quest 3 Planner Stack — System Architecture`](../references/x2_quest3_planner_stack_architecture.md)
  — end-to-end engineering reference: mermaid topology, full ZMQ port +
  topic catalogue, CONFLATE/HWM matrix, boot/shutdown sequencing, and
  the invocation matrix covering wrappers / individual-component launchers / test groups.
- [`X2 Neural Kinematic Planner (kplanner)`](../references/x2_kplanner.md)
  — the **default** planner since 2026-05: trained MotionBricks
  checkpoints driving `motion_inference.predict()` on a 4-D velocity
  intent. Same wire format as the heuristic. Selected via
  `--planner kplanner` (default).
- [`X2 Heuristic Locomotion Planner`](../references/x2_heuristic_planner.md)
  — fallback planner. Still the canonical reference for the v4 future
  window contract + curated primitives library. Available behind
  `--planner heuristic`.
- [`X2 Dataset Record and Replay`](x2_dataset_record_and_replay.md)
  — the v0 stationary-arms-only recorder (predecessor to this stack).
- [`X2 VLA motion_token Decoder`](../references/x2_vla_motion_token_decoder.md)
  — only relevant in `--vla-bridge` mode: why the bridge has to decode
  SONIC tokens itself before the body will move under VLA authority,
  and the operator runbook for the auto-resolved `--vla-bridge-sonic-checkpoint`.

---

## TL;DR launch line

```bash
cd ~/Projects/GR00T-WholeBodyControl

# Teleop only (no dataset writes) — kplanner is the default
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --duration 600

# Record episodes
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 600 \
    --with-record \
    --output-dir data/quest3_x2_planner_phase0/v1 \
    --task "phase 0 stand-and-manipulate"

# Force the heuristic planner (fallback / A-B comparison)
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --planner heuristic --duration 600
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

| Input | Command in planner log | Planner bin / state | World-frame motion |
|---|---|---|---|
| **L stick fwd** | `back_step / default` | `back_step_half_ft` | One stride **forward** |
| **L stick back** | `fwd_step / default` | `fwd_step_1ft` | One stride **backward** |
| **L stick fwd + A held** | `walk / backward` | `back_walk_standard` | **Continuous walk forward**; release to stop |
| **L stick back + A held** | `walk / forward` | `fwd_walk_standard` | **Continuous walk backward**; release to stop |
| **L stick L / R** | `side_left` / `side_right` | `side_left_step` / `side_right_step` | Single side stride |
| **R stick L / R**, hard (\|rx\| ≥ 0.75) | `turn_left / deg_45` / `turn_right / deg_45` | `turn_*_45deg` | Yaw step |
| **R stick L / R**, hard + **X held** | `turn_* / deg_90` | `turn_*_90deg` | Bigger yaw step |
| **R stick fwd / back (ry ≠ 0)** *(v7.4: bidirectional)* | `hold_torso / continuous` (waist_pitch_deg signed) | `STATIC_HOLD` | Continuous **forward / backward lean** (±20° clamp; pre-v7.4 only forward worked) |
| **R stick L / R, soft (\|rx\| < 0.75)** | `hold_torso / continuous` (waist_yaw_deg ≠ 0) | `STATIC_HOLD` | Continuous **torso twist** (up to ±25.7° at the threshold; ±40° hard cap) |
| Sticks neutral | `hold_torso / continuous` (0, 0, 0) | `STATIC_HOLD` (slewed back to neutral) | Stand still |
| **R thumbstick CLICK** | (no `planner_cmd`; manager toggles `_waist_frozen`) | `STATIC_HOLD` pinned at click-time pose | Freeze the current lean / twist; release with another click |

> **v7 continuous torso hold**: the right stick is position-mapped
> to a continuous `(pitch, yaw)` target (gated by the manager's
> default `--intent-enable-continuous-torso`). The decoder throttles
> emissions to a 0.5° change-or-50 ms cadence, the planner runs the
> per-axis 60 °/s slew limit, and `_WAIST_RAMP_CAP_DEG` (pitch 20°,
> roll 10°, yaw 40°) clamps the target so an extreme stick yank
> can't drive SONIC out of distribution. **v7.4** flipped pitch
> bidirectional — backward lean (`ry < 0`) now decodes to negative
> `waist_pitch_deg` (clamped to `-20°`); pre-v7.4 the negative side
> was clamped to 0.
>
> **v7.1 R-click waist freeze**: pressing the **right thumbstick
> down** toggles `_waist_frozen`. While frozen the manager drops
> live `hold_torso` updates from the right stick — the planner stays
> in `STATIC_HOLD` at whatever pose was active at click time. Other
> commands (walk, turn, idle) still flow through, so you can lean,
> click to freeze, then walk around with the body locked at the lean
> using the left stick. Click again to release. The freeze persists
> across B-press mode flips, including ARM_MANIPULATION → LOCOMOTION
> (you can do arm work, B-flip back, and keep walking with the
> torso still frozen). Going to OFF clears the freeze.
>
> **v7.2 changes (2026-05-14)**: two operator-facing tweaks landed
> together:
>
> - **R-stick now drives the waist in ARM_MANIPULATION too**, not
>   just LOCOMOTION. The arm IK targets are computed in the robot's
>   torso frame, so leaning the body during arm work cleanly
>   extends the reachable envelope (the arms ride along with the
>   torso). Walk / step / turn commands are still filtered out in
>   ARM_MAN — the operator's IK targets must not slide out from
>   under their hands. R-click freeze and the B-press latch behave
>   identically in both modes.
> - **Lateral lean (roll) is removed from the operator vocabulary.**
>   v7.0 used "A held + R-stick X → roll", but A and the R-stick
>   share the operator's right thumb on the same controller and
>   the modifier was unreachable mid-lean in practice. The wire
>   format still carries `waist_roll_deg` (scripted demos and
>   future VLA outputs can emit it), the planner still slews to a
>   non-zero roll target if asked, but the right thumbstick can
>   no longer steer it. Get sideways reach by chaining a brief
>   yaw-twist with the forward lean, or pre-pose with a step
>   before B-pressing into ARM_MAN.
>
> The legacy discrete `lean_fwd_*` and `torso_*_30deg` bins remain
> reachable via `--enable-lean-fwd` / `--enable-torso` (and via the
> scripted YAML demos), but the VR right-stick path is now the
> continuous one. Pass `--no-intent-enable-continuous-torso` on the
> manager to fall back to the v6 dominant-axis behavior.
>
> **v7.4 (2026-06-03): bidirectional pitch + ARM_MAN L-stick squat /
> roll + dominance cones**:
>
> 1. **R-stick is bidirectional**: forward push leans forward, back
>    push leans backward, all the way to ±20°. The wire format
>    already allowed signed pitch — the decoder just stopped
>    clamping the negative half.
> 2. **L-stick repurposed in ARM_MANIPULATION** (LOCO unchanged).
>    LX → continuous **roll** (`waist_roll_deg`, ±10° hard cap).
>    LY → **continuous hip-height squat / stand** (`hip_height_m`,
>    asymmetric cap: ~9 cm down, ~4 cm up — the kplanner's training
>    distribution covers crouch-down further than stand-up before
>    going OOD). The hip-height path is **kplanner only**; the
>    heuristic planner ignores `hip_height_m` and synthesises a
>    frozen-feet pose at the default pelvis height. For squat under
>    the heuristic, fall back to `crouch_medium`. Disable the
>    L-stick repurpose with `--no-enable-arm-man-lstick`.
> 3. **Dominance cones**: pitch fires only when `|ry| ≥
>    pitch_dominance_ratio · |rx|` (yaw-priority cone, R-stick); a
>    similar `height_dominance_ratio` gates the L-stick LY squat
>    against LX roll (roll-priority cone). 0.4 default. Lets the
>    operator twist while leaning slightly without accidentally
>    cranking lean / squat from a yaw-or-roll-intent stick wobble.
> 4. **Why no bins**: with the trained kplanner now the default
>    backend, continuous channels (`hip_height_m`, signed
>    `waist_pitch_deg`) are first-class inputs — the manager no
>    longer needs to map operator intent through the heuristic's
>    discrete primitive labels. The kplanner's `intent_to_velocity`
>    substitutes `hip_height_m` into channel-3 of the velocity
>    intent for `hold_torso` commands and the publish loop runs a
>    separate kinematic waist overlay (slew-limited `_HoldTracker`,
>    same `make_waist_pose_frame` helper as the heuristic
>    `STATIC_HOLD`) so the three waist DOFs ride on top of the
>    neural model's leg motion. See [`x2_kplanner.md`](../references/x2_kplanner.md)
>    ("Continuous waist hold (v7.4)" callout) for the full
>    description.
>
> The hard turn (R stick L/R, |rx| ≥ 0.75) is **always on** in
> LOCOMOTION and pre-empts the continuous hold — the operator wants
> to PIVOT at the end of the stick, not just lean further. In
> ARM_MANIPULATION the hard-turn fires as `turn_*` inside the
> decoder but is then filtered out by the mode gate, so a hard
> R-stick deflection during arm work is a no-op.

### Precedence rules (locked in by `tests/test_intent_decoder.py` + `tests/test_intent_decoder_continuous_torso.py`)

1. **Y held** → would emit `crouch / medium`, but **crouch is currently disabled** (X2 SONIC tips over on the crouch primitive). Y held is silently ignored. v7.4 supersedes the discrete crouch with a continuous L-stick LY hip-height path under ARM_MANIPULATION (kplanner only).
2. **In LOCOMOTION mode, L stick wins over R stick** for locomotion (walk / step). v7.4 changed nothing here.
3. On L stick (LOCOMOTION), **dominant axis** wins: \|ly\| ≥ \|lx\| → fwd/back; otherwise side-step.
4. On R stick (continuous mode, the default): hard \|rx\| ≥ 0.75 → discrete `turn_*` (operator wants a real pivot). Otherwise both `ry` and `rx` are read simultaneously and composed into a single `hold_torso / continuous` target (`pitch` from `ry` — **signed in both directions** since v7.4; `yaw` from `rx`; `roll` from the operator R-stick path is always 0 — see v7.2 callout for why and v7.4 callout for where roll moved). Stick fully released → `(0, 0, 0)` neutral hold.
5. **Yaw-priority cone (v7.4, R-stick)**: pitch is suppressed unless `|ry| ≥ pitch_dominance_ratio · |rx|` (default 0.4). Soft yaw + tiny pitch deflection → pure twist. Lets the operator turn the torso a few degrees off-axis without accidentally adding lean.
6. **In ARM_MANIPULATION mode (v7.4)**: the **L-stick is repurposed for waist control** (NOT walk). LX → `waist_roll_deg` (continuous, ±10° hard cap). LY → `hip_height_m` (continuous squat / stand, asymmetric cap ~9 cm down / ~4 cm up; kplanner only). The R-stick still drives pitch + yaw exactly as in LOCOMOTION. This is mode-gated: in LOCOMOTION the L-stick still drives walk / step / side. Pass `--no-enable-arm-man-lstick` to revert to the v7.3 behaviour (L-stick fully filtered out in ARM_MAN).
7. **Roll-priority cone (v7.4, ARM_MAN L-stick)**: hip-height is suppressed unless `|ly| ≥ height_dominance_ratio · |lx|` (default 0.4). Soft roll + tiny LY deflection → pure roll. Same idea as the yaw-priority cone, in the orthogonal axis.
8. **A held** in LOCOMOTION promotes fwd/back L stick → `walk` (continuous gait). The legacy "A held + R-stick X → roll" path was removed in v7.2 (right-thumb ergonomics). In ARM_MANIPULATION A toggles arm IK; it does **not** modify any stick path.
9. **X held** only modifies hard rx (\|rx\| ≥ 0.75) to a 90° turn; soft rx is unaffected.
10. **B press** in LOCOMOTION → ARM_MANIPULATION **latches** the current `(pitch, roll, yaw, hip_height_m)` waist target so the planner enters `STATIC_HOLD` (heuristic) / continuous-hold (kplanner) at that pose with no jump. The latch is just a *seed*: the operator's R-stick (and L-stick in v7.4) are still live in ARM_MAN and will slew the planner away from the latched value (or back to neutral on release). **B press** in ARM_MANIPULATION → LOCOMOTION clears the latch and emits `idle / default`.
11. **Mode gate (v7.2 + v7.4)**: in ARM_MANIPULATION the decoder allows `hold_torso` through (lean / twist / roll / squat all still steer the waist for extra arm reach) but filters out walk / step / turn commands so the base never slides under the operator's IK targets. In LOCOMOTION the full vocabulary flows. In OFF nothing flows.

### What "hold the stick" does

Most planner primitives are **single-stride**: one push of the L stick = one `fwd_step`. Holding the stick does **not** keep stepping — you must release and re-push to take another step. The exceptions are:

- `walk / forward` and `walk / backward` (A + ly): planner loops the continuous walk primitive until you release.
- **R stick (continuous)**: pitch / yaw / roll are *position-mapped*; holding the stick at +0.5 ry holds the body at the equivalent forward lean angle. Releasing slews the target back to neutral at 60 °/s. This is the v7 `hold_torso` path, not the legacy `lean_fwd_*` discrete bins.

---

## ARM_MANIPULATION mode — arms track VR + recording controls + R-stick still steers waist

| Input | Action |
|---|---|
| **A press** | toggle arm IK engage / disengage (your VR wrists drive the X2 arms) |
| **B press** | back to LOCOMOTION; clears the latched waist hold so the right stick takes over again |
| **X press** | start episode. The `recorder_cmd` ZMQ message is always published; the recorder no-ops it in `--teleop-only` (foreground prints `[recorder] [X] ignored: --teleop-only mode (no dataset writes)`) and ignores it if already recording. The "Recording." headset cue is gated on `--recorder-enabled` *(v7.2)* so it only plays when the wrapper was launched with `--with-record`. |
| **Y press** | stop & save episode. Same pattern: wire path always fires, recorder logs `ignored: no active episode` if there's nothing to save, and the "Saved." headset cue is gated on `--recorder-enabled` *(v7.2)*. |
| **L thumbstick click** | cycle deploy MuJoCo viewer's fixed cameras (synthesises a `]` keystroke; works in LOCOMOTION too). Pre-v7.1 this was the right click; v7.1 also briefly used `Tab` here, but `Tab` only toggles the viewer's left UI panel — `]` is the real "next fixed camera" key. |
| **R thumbstick click** | toggle waist freeze on / off — same control as in LOCOMOTION. Useful to release a freeze without flipping back to LOCOMOTION first. |
| **R stick fwd / back (ry ≠ 0)** *(v7.4 bidirectional)* | `hold_torso / continuous` (waist_pitch_deg signed) → continuous **forward / backward lean** while doing arm work, ±20° clamp. The arm IK rides the torso so this directly extends arm reach. |
| **R stick L / R, soft** | `hold_torso / continuous` (waist_yaw_deg ≠ 0) → continuous **torso twist**, same ±25.7° / ±40° caps as in LOCOMOTION. *(v7.2)* |
| **R stick L / R, hard** | **No-op** — the decoder fires `turn_*` internally but the ARM_MAN mode gate filters it out. Hard pivots while doing arm IK would slide your hands' reference frame; if you want to pivot, B-press to LOCOMOTION first. |
| **L stick L / R (lx ≠ 0)** *(v7.4)* | `hold_torso / continuous` (waist_roll_deg ≠ 0) → continuous **lateral lean** for sideways reach. ±10° hard cap. Roll-priority cone gates LY squat against LX roll. |
| **L stick fwd / back (ly ≠ 0)** *(v7.4, kplanner only)* | `hold_torso / continuous` (`hip_height_m` set) → **continuous squat / stand**. Asymmetric cap: ~9 cm down (squat / push), ~4 cm up (stand / lift) from the kplanner's default hip height. The heuristic planner ignores `hip_height_m` and stays at default pelvis height; for a heuristic-mode squat fall back to the discrete `crouch_medium` primitive in a scripted demo. |
| **L stick (any) — v7.3 and earlier** | **No-op** — walk / side / step commands were filtered out by the mode gate. v7.4 reroutes the L-stick to waist control (above); to restore the v7.3 behaviour pass `--no-enable-arm-man-lstick`. |

> **Torso latch (v7) + live R-stick (v7.2) + live L-stick (v7.4)**: when the operator B-presses LOCOMOTION → ARM_MANIPULATION, the manager samples the live `(pitch, roll, yaw, hip_height_m)` continuous waist target (v7.4 added the 4th component) and pins the planner to `STATIC_HOLD(latched)`. The headset announces it with the `mode_torso_locked` audio cue (only when the latched pose is non-neutral; neutral latches just play the standard `mode_arm_manipulation` cue). The latch is a *no-jump seed* — the planner enters `STATIC_HOLD` at exactly the pre-flip pose so there's no transient — but **both** sticks remain live in ARM_MAN: the R-stick continues to slew pitch / yaw, and the L-stick (v7.4) drives roll / squat. Pre-pose with the R-stick in LOCOMOTION (`pitch=15°, yaw=20°` extends arm reach by ~15 cm in the chosen direction), B-press to lock-and-go, then either keep the sticks deflected or use the **R-thumbstick click** to hard-freeze the pose if you want to release the sticks and use both hands on arm IK without the torso slewing back to neutral. The reverse B-press clears the latch and emits `idle / default` so the planner cleanly blends back to standing.

**All recording triggers are gated on ARM_MANIPULATION mode.** A held + walk in LOCOMOTION can never accidentally start an episode; X held + 90° turn in LOCOMOTION never fires start. The episode lifecycle (`start` → frames → `save`) is owned by the manager and forwarded to the recorder over ZMQ topic `recorder_cmd`.

**The L thumbstick click is *not* gated on ARM_MAN** — it's active in any non-OFF mode so you can re-frame the viewer while walking the robot into position. It's idle in OFF (which is consistent with the rest of the manager: OFF means "ignore controller events"). The R thumbstick click is similarly active in both LOCOMOTION and ARM_MANIPULATION, idle in OFF.

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

> **Datasets are training-ready as written.** Every recorded
> `action.motion_token` is the SONIC FSQ encoding of the **planner's
> real 10-frame future window** at that tick — produced inline by
> the recorder, no offline labeling step required. The wrapper auto-
> resolves the SONIC `.pt` sibling of `--model` and the YAML encoder
> config (`gear_sonic/data/encoder/x2_observation_config.yaml`), and
> fails pre-flight if either is missing (see the "Knobs" table for
> the `--no-sonic-checkpoint` and `--encoder-config ''` smoke-test
> escapes). Look for the banner lines:
>
> ```
> motion_token     : ON  (/.../model_step_025000.pt, cuda:0)
> encoder_config   : .../x2_observation_config.yaml, modes=[retargeted_body_q], multi-frame 10x68 -> 680-D
> ```
>
> to confirm both loaded. If `motion_token` says `DISABLED`, the
> parquet's `motion_token` column will be all zeros; if
> `encoder_config` says `DEPRECATED freeze-pose`, the column will be
> tokens that encode static intent (every row tiled from the current
> body_q). Both are smoke-test-only states — re-record with the
> defaults restored before training a VLA on it.
>
> **VLA-grade recording requires this stack.** The single-process
> direct-mode loop (no planner, no manager) intentionally falls back
> to the freeze-pose path because there's no real future window to
> source — that path emits a one-shot deprecation warning and is
> **not** suitable for VLA datasets. Use this wrapper or another
> subscribe-mode pipeline for any episode you intend to train on.

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
| `turn_threshold` | 0.75 | Soft / hard rx boundary (continuous torso twist below; discrete `turn_*_45deg` at or above). At 0.75 the soft band reaches ~25.7° of continuous waist yaw before the discrete pivot fires. |
| `lean_medium_threshold` | 0.55 | ry boundary between `lean_fwd_small` and `lean_fwd_medium` (legacy discrete bin path; bypassed by the v7 continuous hold) |
| `lean_large_threshold` | 0.80 | ry boundary between `lean_fwd_medium` and `lean_fwd_large` (legacy discrete bin path) |
| `chord_debounce_s` | 0.5 | Quiet window after the A+B+X+Y chord puts you in LOCOMOTION |
| `enable_crouch` | `False` | Y-held → `crouch / medium` (currently destabilizes the policy; superseded by the v7.4 continuous L-stick LY squat under ARM_MAN + kplanner) |
| `--pitch-dominance-ratio` | 0.4 | **v7.4** R-stick yaw-priority cone: pitch fires only when `|ry| ≥ ratio · |rx|`. Set to 0 to disable the cone (always emit pitch from `ry`). |
| `--height-dominance-ratio` | 0.4 | **v7.4** L-stick (ARM_MAN) roll-priority cone: hip-height fires only when `|ly| ≥ ratio · |lx|`. |
| `--max-height-down-m` | 0.09 | **v7.4** Maximum continuous squat below kplanner default hip height (m). Asymmetric: training distribution covers crouch-down further than stand-up. |
| `--max-height-up-m` | 0.04 | **v7.4** Maximum continuous stand above kplanner default hip height (m). Lower than `--max-height-down-m` because the kplanner's training distribution is asymmetric. |
| `--hold-height-threshold-m` | 0.005 | **v7.4** Hysteresis for re-emitting `hip_height_m` updates (0.5 cm). Prevents noisy stick chatter from spamming `hold_torso` commands. |
| `--no-enable-arm-man-lstick` | (off; L-stick repurpose ON by default via `--enable-arm-man-lstick`) | **v7.4** Revert ARM_MAN L-stick to v7.3 behaviour (filtered out, no waist control). Useful for A/B comparison or operators who prefer the original feel. |

These live in `gear_sonic/utils/teleop/vr/intent_decoder.py::IntentDecoder.__init__`. The v7.4 flags are exposed on the manager (`gear_sonic/scripts/quest3_manager_x2.py`); the legacy ones still need to be added to `ManagerConfig` and the argparse group manually if you want to override them.

### Inline SONIC FSQ tokenizer (for `action.motion_token` in the LeRobot dataset)

Wrapper-level flags that control whether `action.motion_token` is the
real FSQ encoding of the planner's 10-frame future window (training-
ready) or one of the two fallback paths (smoke-test only):

| Flag | Default | Meaning |
|---|---|---|
| `--sonic-checkpoint PATH` | auto-resolved sibling of `--model` | Override the SONIC `.pt` used to encode `action.motion_token`. The auto-resolve strips `/exported/` and the `_g1.onnx` suffix and appends `.pt`. |
| `--no-sonic-checkpoint` | (unset) | Skip the tokenizer entirely. The recorder will write **zeros** for `action.motion_token` and the dataset is **NOT** VLA-trainable. Only use for kinematic-only smoke tests. |
| `--sonic-tokenizer-device DEV` | `cuda:0` | Torch device for the inline tokenizer. Use `cpu` if `cuda:0` is contended by the deploy / VLA on the same GPU (~1 ms/tick CPU overhead, well within the 20 ms 50 Hz budget). |
| `--encoder-config PATH` | `gear_sonic/data/encoder/x2_observation_config.yaml` | YAML that drives the inline tokenizer's gather (the 680-D 10-frame future window the SONIC encoder was trained on). The default mirrors G1's `observation_config.yaml` schema. Pass `--encoder-config ''` to fall back to the **deprecated freeze-pose** path (current body_q tiled 11 times) — semantically incorrect for VLA training. |

The wrapper banner shows the resolved state on every launch:

```
motion_token     : ON  (/.../model_step_025000.pt, cuda:0)
encoder_config   : /.../x2_observation_config.yaml, modes=[retargeted_body_q], multi-frame 10x68 -> 680-D
```

or, when the tokenizer is off entirely:

```
motion_token     : DISABLED (action.motion_token = zeros; dataset will NOT be VLA-trainable)
encoder_config   : (unused; tokenizer DISABLED above)
```

or, when the operator opted out of the multi-frame path:

```
motion_token     : ON  (/.../model_step_025000.pt, cuda:0)
encoder_config   : DEPRECATED freeze-pose (--encoder-config '' was passed)
```

Verify the banner before pressing `X` — `DISABLED` is the operator's
last chance to abort before producing a zero-token episode, and
`DEPRECATED freeze-pose` is the last chance to abort before producing
a static-intent episode.

### Validating the inline tokenizer (Layers 3 + 4)

The unit tests (`tests/test_x2_encoder_obs_builder.py` and
`tests/test_x2_dataset_recorder_real_future_token.py`) cover Layers
1 + 2 of the validation pyramid — those run on every push. Layers 3
+ 4 require a live planner stack and a recorded parquet
respectively, so they're operator-driven incantations.

**Layer 3 — byte-parity between recorder Python obs and deploy C++
obs.** Run the planner stack with the deploy's `--obs-dump` enabled
and the recorder's `--obs-dump-recorder` flag set, then diff the
two snapshots:

```bash
# Terminal 1: planner stack with deploy obs-dump enabled
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 30 \
    --extra-deploy-args "--obs-dump /tmp/x2_deploy_obs.bin"

# Terminal 2: recorder side, same wire
python -m gear_sonic.scripts.record_x2_dataset \
    --teleop-only --body-pose-source zmq \
    --sonic-checkpoint /path/to/model_step_NNNNN.pt \
    --encoder-config gear_sonic/data/encoder/x2_observation_config.yaml \
    --obs-dump-recorder /tmp/x2_recorder_obs.pt \
    --duration 10

# After both run, diff:
python gear_sonic_deploy/scripts/compare_recorder_vs_deploy_obs.py \
    --recorder /tmp/x2_recorder_obs.pt \
    --deploy   /tmp/x2_deploy_obs.bin
```

Expect `max-abs diff < 1e-5`. Anything larger means the recorder's
gather diverged from the deploy's `ZmqPoseInputSource`.

**Layer 4 — encode-decode consistency on a recorded parquet.** Spot-
check that re-encoding the parquet's `action.body_q_mj_pre_sonic` via
the same `.pt` reproduces (close to) the stored `action.motion_token`:

```bash
python gear_sonic/scripts/validate_encode_decode_loop.py \
    --parquet data/lerobot/x2_phase0_smoke_v0/data/chunk-000/episode_000000.parquet \
    --checkpoint /path/to/model_step_NNNNN.pt \
    --device cpu
```

Expect mean cosine similarity > 0.92 and bucket match >= 60 %. The
recorder uses the planner future, while the validator uses a
temporal-slice future from the parquet; they're not byte-equal but
should be highly correlated.

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

| Event                                    | Spoken prompt        | MP3 key                  | Gated on              |
|------------------------------------------|----------------------|--------------------------|-----------------------|
| Mode → OFF (incl. estop chord)           | "Off."               | `mode_off`               | always                |
| Mode → LOCOMOTION                        | "Locomotion."        | `mode_locomotion`        | always                |
| Mode → ARM_MANIPULATION                  | "Arm manipulation."  | `mode_arm_manipulation`  | always                |
| LOCO → ARM_MAN with non-neutral waist    | "Torso locked."      | `mode_torso_locked`      | always                |
| R-thumbstick click — freeze ON           | "Torso frozen."      | `torso_frozen`           | always                |
| R-thumbstick click — freeze OFF          | "Torso released."    | `torso_released`         | always                |
| X press in ARM_MAN — start episode       | "Recording."         | `record_start`           | `--recorder-enabled` *(v7.2)* |
| Y press in ARM_MAN — stop & save episode | "Saved."             | `record_save`            | `--recorder-enabled` *(v7.2)* |

> **v7.2 false-ACK fix (2026-05-14)**: pre-v7.2 the `Recording.` / `Saved.` cues fired on the **rising edge of X / Y**, with no awareness of whether the recorder was actually writing parquet. Real-world consequence (caught in the v7.2 verification session): the operator launched the wrapper without `--with-record` (so the recorder was in `--teleop-only` mode), pressed Y twice during ARM_MAN, heard the headset say "Saved." both times, and walked away thinking two episodes were on disk. They weren't. The recorder log line — `[recorder] [Y] ignored: --teleop-only mode (no dataset writes)` — was the ground truth, but the headset cue had already lied.
>
> Since v7.2 the `record_start` / `record_save` audio cues are **gated on the manager's `--recorder-enabled` flag** (default OFF). The `recorder_cmd` ZMQ message is *still* published on every X / Y press — the recorder is the source of truth and its log line is unchanged — but the headset stays silent in `--teleop-only` runs. The wrapper (`run_x2_quest3_planner_stack.sh`) sets `--recorder-enabled` automatically iff `--with-record` was passed; manual `python -m gear_sonic.scripts.quest3_manager_x2` launches must opt in explicitly.
>
> **Known remaining footgun**: the gate covers the most common case (operator forgot `--with-record`) but doesn't yet help with `--with-record` + bad sequencing. In a `--with-record` session, an X press while *already recording*, or a Y press *with no active episode*, will still fire the cue — the recorder logs `ignored: ...` either way, but the headset can't see that. Tracked as the **"recorder-ack topic"** follow-up: recorder PUBs `recorder_ack` after each `recorder_cmd`, manager waits ~150 ms for the ACK before playing the cue, and a new "Save failed." / "Already recording." cue fires on rejection. Until that lands, **trust the foreground `[recorder] ...` lines, not the headset audio**, when you need to know whether a parquet hit disk.

To regenerate the cache (e.g. after editing `PROMPT_TEXTS`), delete the per-key MP3s under `quest3_webxr_app/audio/` and either restart the manager (which calls `ensure_prompt_audio_files` on Quest3Reader startup) or run:

```bash
python -c "from gear_sonic.utils.teleop.vr.quest3_audio_prompts import ensure_prompt_audio_files; ensure_prompt_audio_files()"
```

If a cue is missing on the headset, check (1) the headset isn't muted in Quest Settings → Sound, (2) the WebXR debug overlay shows `playPromptAudio('mode_*'/'record_*')` log lines on the relevant action, (3) the MP3 file exists on disk under `quest3_webxr_app/audio/`. If gTTS / network are unavailable when the cache is built the WebXR client falls back to `speechSynthesis` (less reliable) for any missing key.

---

## Camera cycling (left thumbstick click → deploy viewer `]`)

Manipulation work benefits a lot from re-framing — picking up a small object reads better from `obj_left` or `obj_right` than from `ego_view`. Pressing the **left thumbstick click** sends a synthetic `]` keystroke to the deploy MuJoCo viewer's GLFW window, cycling the same fixed cameras you'd cycle from the workstation keyboard. Pre-v7.1 this was on the right click, but the right click is now reserved for the waist freeze toggle so the operator can keep their right thumb on the lean / twist stick.

> **Why `]` and not `Tab`?** v7.1 originally synthesised `Tab` because that key was named in an early cheatsheet draft, but `Tab` actually toggles the viewer's **left UI panel** in `mujoco.viewer.launch_passive` and leaves the active camera unchanged. The next-fixed-camera key is `]` (xdotool keysym `bracketright`); the previous is `[` (`bracketleft`); `Esc` snaps back to the free orbit camera. Verified live on `mujoco==3.5.0`. Override via `ViewerCameraCycler.CYCLE_KEYSYM` if you want to cycle backwards.

| Camera (default order) | What it shows |
|------------------------|---------------|
| `obj_left`             | Wrist-side view biased to the robot's left hand |
| `obj_right`            | Wrist-side view biased to the robot's right hand |
| `front_cam`            | Wide-angle (120° vertical FoV) world-fixed witness view, 3 ft in front of the robot launch position at chest height. Stays put when the robot walks (no targetbody tracking) so it doubles as the second video track baked into the dataset (`observation.images.front_cam`, see "Front-cam witness view" below) |
| `rgbd_head_front`      | Head-mounted, near identical to the recorded `ego_view` |
| free orbit             | Mouse-controlled (cycle past all fixed cameras with `]`, or press `Esc` directly) |

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

If the `buttons:` line **does** include `leftStickClick / rightStickClick` keys but pressing the click still doesn't log `L-stick click` in the manager log, the headset / browser combo isn't exposing `gpad.buttons[3]` (rare on older Quest Browser builds). Workaround: use the workstation `]` key directly on the focused MuJoCo viewer window — the cycler is purely additive, it doesn't disable the keyboard binding.

### "manager logs `[L-click] camera cycle: ok` but the camera doesn't actually move"

Two known causes:

1. **Wrong keysym** — the cycler is dispatching a key but the viewer doesn't bind it to camera-cycle. v7.1 originally sent `Tab`, which only toggles the left UI panel; the fix is to send `]` (xdotool keysym `bracketright`). Pinned by `test_cycler_default_keysym_is_bracketright_not_tab`. If your fork uses a custom MuJoCo build that rebinds `]`, override `ViewerCameraCycler.CYCLE_KEYSYM` (e.g. to `F2` or whatever the build uses).

2. **GNOME mutter wrapper swallowing the keystroke** (fixed 2026-05-13). On GNOME (and a few other compositors that wrap X11 clients in a decorative window) `xdotool search --name "MuJoCo"` returns **two** windows — the real GLFW viewer (class `MuJoCo`) and the compositor's frame wrapper (class `mutter-x11-frames`). The wrapper has the right title but doesn't have a GLFW event loop, so synthetic key events to it land in the void. Pre-fix, the cycler picked the wrapper; post-fix it filters by `--classname MuJoCo` first and falls back to a class-filtered name search that excludes `mutter-x11-frames` explicitly.

To diagnose without putting the headset on, run the standalone CLI:

```bash
.venv/bin/python -m gear_sonic.utils.teleop.vr.viewer_camera_cycler --repeat 3 -v
```

It prints the resolved DISPLAY, xdotool path, search pattern, and the keysym it'll send, then dispatches `cycle()` three times against the live viewer. If you see `OK ('bracketright' dispatched)` three times AND the viewer rotates through three cameras, the path is healthy. If the dispatcher reports OK but the viewer doesn't move, you're hitting one of the two causes above.

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
# Pick-place cube on a table (red primitive cube → blue bowl):
python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceCube

# Pick-place bowl on a table (blue primitive bowl → green target zone):
python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceBowl

# Pick-place apple on a table (real-mesh apple → blue bowl):
python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --env X2PickPlaceApple

# Or build all bundled scenes in one go:
python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --all
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
| `front_cam` witness video track | no — single-camera schema (`ego_view` only) | **yes by default *(v7.3)*** — schema gains `observation.images.front_cam`; pass `--no-front-cam` to opt out |
| `robot_pose` SUB → live `root_pos_xyz` for the renderer | no | **yes *(v7.3)*** — recorder subscribes to bridge port 5570 so both ego_view and front_cam see the robot wherever it actually is, instead of the renderer's hardcoded `(0, 0, 0.793)` |
| ZMQ ports bound                 | 5556, 5557, 5563, 5564, 5565                | + 5559 (scene_state PUB) + 5560 (scene_reset SUB) + 5570 (robot_pose SUB) |
| `--task` required               | yes (always)                                | optional — auto-fills from env's instruction             |
| Finger curl compensation        | OFF (pass `--apply-curl-compensation` to enable) | **ON** (pass `--no-apply-curl-compensation` to disable) |
| Finger thumb-oppose compensation| OFF (pass `--apply-oppose-compensation` to enable) | **ON** (pass `--no-apply-oppose-compensation` to disable) |

**Finger compensations.** Power-grasp pick-and-place on a small cube / bowl is exactly the workload the OmniHand curl + thumb-oppose compensations were tuned for, so the wrapper auto-enables both when you pass `--robocasa-env`. The banner shows the resolved values (`finger comp: curl=on  oppose=on  (robocasa default; pass --no-apply-{curl,oppose}-compensation to override)`) so the operator never has to guess. The compensations apply on the **manager** side (the Retargeter that builds `hand_finger_cmd` lives in the manager in subscribe-mode), not the recorder; previously the wrapper forwarded the flags only to the recorder which silently no-op'd them — fixed 2026-05-13.

#### Front-cam witness view *(v7.3, 2026-05-13)*

Robocasa scenes now bake in a **second camera** named `front_cam`: a wide-angle (120° vertical FoV) world-fixed view 3 ft (~0.91 m) in front of the robot launch position at chest height (`z = 1.10 m`), looking back along world `-x`. The XML lives next to `obj_left` / `obj_right` in `_WORKSPACE_CAMERAS` (see `gear_sonic/scripts/build_x2_robocasa_scene_xml.py`), and the recorder writes its frames as a second LeRobot video track — `observation.images.front_cam` — alongside the existing `observation.images.ego_view`. Both tracks are 640×480 / 50 Hz / mp4-encoded by the same exporter, so the on-disk layout becomes:

```
videos/chunk-000/
├── observation.images.ego_view/episode_000000.mp4   # head-mounted, what the policy sees
└── observation.images.front_cam/episode_000000.mp4  # wide-angle witness, what a human sees
```

The `meta/modality.json` `video` block gains a matching `front_cam` entry so the trainer can attend to both views.

**Why a second camera?** Manipulation episodes are easier to label, easier to debug, and easier to diff between operators when there's a fixed third-person reference. The wide 120° FoV keeps the entire X2 + the table in frame even when the robot leans forward to grasp something — at the cost of barrel distortion at the edges, which the trainer learns through anyway.

**Wiring:**

* **Build script** — the camera is an entry in `_WORKSPACE_CAMERAS` with `mode="fixed"` + explicit `xyaxes`, same dict format as the existing `obj_left` / `obj_right` (which use `mode="targetbody"`). Re-running `build_x2_robocasa_scene_xml.py --all` regenerates all three scene XMLs with the camera baked in.
* **Recorder** — `record_x2_dataset.py --front-cam` toggles a second `MujocoFrameRenderer` pinned to `camera="front_cam"` plus the second LeRobot video feature. The wrapper resolves the flag automatically: omitted + `--robocasa-env != none` → ON; omitted + flat floor → OFF; explicit `--front-cam` / `--no-front-cam` always wins. ~20-40 MB extra GPU memory + one extra `render()` call per tick (sub-millisecond at 640×480 on a modern GPU).
* **Live `root_pos_xyz`** — the bridge already publishes `pelvis_qpos[0:7]` on the `robot_pose` topic (port 5570; sim-only ground truth, no real-robot equivalent). The recorder now subscribes and feeds the live `(x, y, z)` into both renderers' `root_pos_xyz` argument so a fixed camera actually sees the robot translate when it walks. `ego_view` is rigidly attached to the head and is invariant under root translation, but using the live value here keeps both renderers consistent and removes the previous "hardcoded `(0, 0, 0.793)`" footgun for any future world-fixed cameras.
* **Schema gating** — `get_features_x2_vla(include_front_cam=True)` adds `observation.images.front_cam`; `get_modality_config_x2_vla(include_front_cam=True)` adds `video.front_cam`. The recorder passes `include_front_cam=cfg.record_front_cam` to both so they can't drift, and the LeRobot exporter's per-frame validator catches any mismatch on the first frame.
* **Cycling** — appears between `obj_right` and `rgbd_head_front` in the L-thumbstick `]` cycle order (see camera-cycling table above).

**Known footgun.** The `front_cam` camera does **not** exist in the legacy flat-floor MJCF (it's only added by `_WORKSPACE_CAMERAS` injection during scene-XML build). Passing `--front-cam` without a scene XML logs a one-shot `WARN` and silently keeps the single-camera schema; the dataset still lands without `observation.images.front_cam` so existing flat-floor recordings stay schema-compatible. Conversely, **appending** to a pre-existing single-camera dataset directory while in robocasa mode will trip the LeRobot exporter at first frame (`KeyError: observation.images.front_cam` in the meta/info.json check) — pass `--no-front-cam` for that one append session, or start a fresh `--output-dir`.

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

`X2PickPlaceCube`, `X2PickPlaceBowl`, and `X2PickPlaceApple` all put a single object on a low table within reach of the X2's pre-calibrated arm workspace. They're the only three envs we've validated end-to-end with the deploy bridge + recorder; adding a new env means writing a `RobocasaTaskMirror` subclass (or, for "object → bowl" siblings of the cube task, registering a new oracle entry that reuses `_phase_pick_place_apple`-style helpers), building its scene XML, and re-running the in-sim object-placement sanity check before recording any episodes.

| `--robocasa-env`     | Manipulable object             | Receptacle                | Success criterion                                        | Notes |
|----------------------|--------------------------------|---------------------------|----------------------------------------------------------|-------|
| `X2PickPlaceCube`    | `PrimitiveCube` (red, ~4.4 cm) | `PrimitiveBowl` (blue)    | Cube xy inside bowl footprint, z within wall window, upright | Original validation scene. Sharp corners are easy for the OmniHand to "trap". |
| `X2PickPlaceBowl`    | `PrimitiveBowl` (blue)         | `PrimitiveFixture` (green target zone) | Bowl xy / z inside target zone, upright             | The bowl is bigger than the cube; tests the OmniHand's ability to grasp a thin-walled receptacle by its rim. |
| `X2PickPlaceApple`   | `apple_0` real-mesh (~7.4 cm)  | `PrimitiveBowl` (blue)    | Apple xy inside bowl footprint, z within wall window. **No uprightness check** (apple is roughly spherical). | Real-mesh sibling of the cube task. Same bowl + table + spawn ranges; mix-and-match with cube data for VLA training without renormalising rewards. The apple's curved sides + stem indent stress non-convex grasping that a cube-only policy never sees. |

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

## Common errors

### `SONIC tokenizer .pt not found at auto-resolved path`

The wrapper failed to find the SONIC `.pt` next to the deploy ONNX
during pre-flight, before any process was spawned. This is the
guardrail that prevents you from silently producing a zero-token
(non-VLA-trainable) dataset. Two valid fixes:

* **You DO want the dataset to be VLA-trainable** (the usual case):
  point at a real `.pt` explicitly via
  `--sonic-checkpoint /path/to/model_step_NNNNN.pt`. The convention
  is the `.pt` lives one directory above the ONNX (alongside the
  Hydra config and `last.pt`); see
  `~/x2_cloud_checkpoints/<run>/model_step_NNNNN.pt` on the dev box.
* **You're doing a kinematic-only smoke test** and accept that
  `action.motion_token` will be all zeros: pass `--no-sonic-checkpoint`.
  The wrapper banner will switch to
  `motion_token : DISABLED (action.motion_token = zeros; ...)`.

The recorder also logs the resolved state at startup; look for
`[recorder] motion_token tokenizer ready (...)` (success) or
`[recorder] WARNING: no --sonic-checkpoint provided; ...` (zeros
fallback) in `recorder.log`.

### `Encoder-observation YAML not found at: ...`

Pre-flight could not find the YAML config that drives the inline
tokenizer's gather. The default lives at
`gear_sonic/data/encoder/x2_observation_config.yaml` (it ships in
the repo). Two valid fixes:

* **You want the canonical multi-frame path** (the usual case):
  restore the file from git (`git checkout
  gear_sonic/data/encoder/x2_observation_config.yaml`) or point at
  your own copy with `--encoder-config /path/to/your.yaml`.
* **You're doing a smoke test and want the deprecated freeze-pose
  path**: pass `--encoder-config ''` (empty string). The wrapper
  banner will switch to `encoder_config   : DEPRECATED freeze-pose
  (--encoder-config '' was passed)` and the recorder will write
  static-intent tokens. **Not** suitable for VLA training.

### `[recorder] DEPRECATED encode() called` warning in `recorder.log`

The recorder's tokenizer fell back to the freeze-pose path (current
body_q tiled 11 times) because either:

* you passed `--encoder-config ''` to opt out of the multi-frame
  path, **or**
* the recorder is in **direct mode** (no `--body-pose-source zmq`,
  no planner snapshot to source a real future window from).

In both cases the resulting `action.motion_token` labels encode
static intent — the VLA will only learn to predict "stay where you
are". Re-record via the wrapper (which forces subscribe mode + the
default YAML) to fix.

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
