# Plan: auto-regenerate G1 locomotion clips from the captured command log

**Status:** PLAN ONLY — not executed. Source log: `sample_planner_logs`.

## Goal
Recover/regenerate the ~64 G1 keyboard-driven locomotion clips that were lost to the
recorder overwrite bug, **without manual re-driving**, by replaying the captured *intent*
stream. Same reconstructed intents can then drive the **X2 kplanner** for the A/B.

## What the log actually contains (verified)
- **Intent/command stream**, one line per replan:
  `mode: {SLOW_WALK, WALK, RUN, IDLE, FORWARD_JUMP}, target_vel, movement[x,y,z], facing[x,y,z]`.
- **Segment boundaries = `R` presses** — logged as `Emergency Stop! Movement momentum reset.`
  (`R` in planner mode = emergency-stop/momentum-reset per the keyboard map). ~110 segments.
- Appended **shell history** → 64 ordered `--motion-key` names.
- Driving vocabulary was **W=forward, Q=heading-left, E=heading-right, S=backward, R=stop**.
- **NO timestamps, NO qpos/body_pose.** => cannot recover exact motion; must *regenerate*
  by replaying intents (the kplanner is near-deterministic given an intent stream).
- ~110 segments vs 64 keys => ~46+ **extra, unrecorded exploratory drives** interleaved;
  alignment must skip them, not assume 1:1.

## Phase 1 — Parse & align (offline, no sim)
1. Split the replan stream on `R`/Emergency-Stop → segments; per segment collapse consecutive
   duplicate intents into "hold" spans; record replan-count per span (duration proxy).
2. Parse the 64 `--motion-key` names → expected signature:
   mode (prefix `slow_walk`/`walk`/`run`), target_vel (digits in name), turn (`turns`/`circle`),
   back (`back`).
3. **Order-preserving DP alignment** (Needleman–Wunsch style, NOT greedy) mapping each key →
   best-fit segment, anchored on target_vel (exact) + mode + back/turn flags + monotonic order,
   with a no-double-assign constraint. Extras (unmatched segments) drop out; ambiguous matches
   get flagged for human review. (A first greedy pass already matched 48/64 — DP + the W/Q/E/R/S
   vocabulary + dedup should push this higher.)
4. Output: `{motion_key: intent_schedule}` + a review table (matched / ambiguous / unmatched).

## Phase 2 — Intent timeline + timing template
- Convert each `(mode, target_vel, movement, facing)` → kplanner 4-D `[yaw_rate, vel_x, vel_z, hip_h]`:
  W→+vel_z, S→−vel_z, Q/E→±yaw_rate, magnitude from target_vel; hip_h = `_HIP_HEIGHT_M`.
- Timing (no timestamps → use the recording rhythm the operator described):
  **1 s idle lead-in → ~7 s active (spans distributed by replan-count proportion) → stop**, on a 50 fps grid.

## Phase 3 — Auto-replay + record (batch, hands-free)
- **Feasibility gate (check first):** the deploy's `InterfaceManager` exposes a ZMQ input (`#`).
  Confirm its message schema accepts `mode+target_vel+movement+facing`; if yes, build a small
  publisher that streams each clip's timeline. (If not, fall back to feeding the reconstructed
  4-D intents through the X2 kplanner via an `x2_pkl_command_source`-style injector.)
- Loop over all keys: publish schedule → deploy runs the motion → the (now-fixed, merging)
  `record_motion_to_pkl.py` captures ~7 s (auto-start after the 1 s settle) → all clips
  accumulate into ONE pkl.
- **Dual-fps output (30 + 50):** the kplanner corpus MUST be exactly **30 fps** (no internal
  reframing); the SONIC/deploy side needs **50 fps**. `build_entry` already resamples from the
  RAW timestamped ZMQ samples (linear + Slerp), so both grids are faithful, not one derived from
  the other. Small recorder enhancement: accept `--fps 30,50` and emit both from ONE capture —
  `<out>` at 50 + sibling `<out>_30fps.pkl` at 30 (or a `--also-fps 30`). Do this once so every
  regenerated clip lands in both a 30 fps planner-corpus pkl and a 50 fps deploy pkl.
  NOTE: `--fps 30` works TODAY on the current tool (default is 50); only the *simultaneous* dual
  write needs the small change.

## Phase 4 — Verify + reuse for the A/B
- Inspect the regenerated pkl: clip count, per-clip root x/y paths, measured speed vs the
  target_vel label.
- Feed the SAME intent schedules through the **X2 kplanner + latest root** → capture X2 root
  track → compare vs the G1 regeneration (root x/y-from-origin error).

## Honest limitations
- Timing is approximate (no timestamps); sustained-speed clips reconstruct faithfully,
  turn/circle transition timing is estimated from replan-count order.
- Regenerated clips are **intent-equivalent**, not the exact lost takes (kplanner target-segment
  seed varies) — fine for a locomotion corpus.
- The ~46 extra segments are unrecorded exploratory drives; keep only the 64 that map to the
  keys (optionally salvage clean extras as bonus clips).

## Single open question before building
The deploy ZMQ command schema (Phase-3 feasibility gate). Everything else is mechanical.
