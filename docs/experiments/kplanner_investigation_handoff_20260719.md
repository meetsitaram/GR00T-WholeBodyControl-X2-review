# kplanner Regression Investigation — Handoff (2026-07-19)

**Read the "Reliability" section first.** This session produced many wrong
intermediate results. Facts below are tagged with how they were established.

---

## The problem

X2 walks poorly under kplanner: barely two steps at low speed, turns unstable,
cannot walk out and return. A month ago the robot walked reliably — but see
"What the old system actually was" below, because that comparison is not what
it appears.

## Hardware ground truth (operator-observed, most reliable evidence we have)

1. **PKL → SONIC works.** Real clips streamed to SONIC walk the robot cleanly.
2. **`walk_circle_001` → SONIC works.** Walked beautifully on the real robot.
3. **X2 kplanner → SONIC fails.**

(2) is critical: `walk_circle_001` is **G1-stock-planner output retargeted to
X2** (see `gear_sonic/scripts/kplanner_gen_from_log_g1.py` docstring). So it is
*generated* motion, not captured. Generated motion works — if it comes from the
G1 planner. That kills "generative planning is inherently hard for SONIC".

## What the old working system actually was — VERIFIED (code + git)

Not a better kplanner. **No kplanner at all.** It was PKL-direct clip replay:
`gear_sonic/scripts/play_xbox_controller.py:137-141` maps D-pad to four
hand-trimmed clips (`x2_ultra_relaxed_walk_{forward,one_left_turn,
one_right_turn,two_right_turns}_v1.pkl`), streamed via
`run_x2_pkl_direct_stack.sh`. Commit `7ce6e1d`.

Reliability came from constraints since removed: every clip starts and ends at
standstill (idle anchor + 30-frame SLERP), a single-flight timer made mid-stride
cuts impossible, and trim points were chosen by FK foot-slide sweep. It also had
**no steering** — four canned trajectories only. Operator has explicitly decided
**not** to go back to this.

---

## ESTABLISHED — high confidence (code/git/file reading, no invented metrics)

- **kplanner is autoregressive and fully open-loop on the robot.**
  `pc2_kplanner_onnx.py:863-871` — context is read from the buffer that *is* the
  previous prediction. No robot state enters. Deliberate, documented at `:69-72`.
  Closed-loop reseed exists **only** in `x2_kplanner.py:1186` (torch), never
  ported to PC2.
- **Deployed config** (md5 `48a8e3551cd6`, matches robot): threshold 16,
  buffer 64, MODEL_FPS 30, OUTPUT_FPS 50, resample active, 8-frame blend.
  Ritual passes no overrides. Replans ≈ every 1.60 s, ~75% of prediction
  consumed.
- **Commit `9d1798f` (07-17) swapped the pose-template table.** Deployed mode is
  `slow_walk`, whose template went **0.40 → 0.22 m/s**; `walk` went 0.27 → 0.51.
  The relaxed-walk clip was removed from the table entirely.
  Backup at `motionbricks/out/X2-clip.{ckpt,modes.json}.bak_default`.
- **kplanner training corpus** (from `scratchpad/kplanner_g1teleop_chain.sh`):
  base `x2_ultra_bones_seed_g1_retarget_feasible.pkl` (33,206 @ 30fps) +
  priority `x2_g1teleop_30fps_windowed.pkl` (122 @ 30fps) at **30% sampling**,
  applied to all three stages (vqvae/root/pose).
  `x2_sonic_executed_feasible.pkl` (50fps) is SONIC's corpus, NOT kplanner's.
- **Corpus speed distribution** (root translation math only): base corpus has
  only **6.0%** of clips in the 0.1–0.3 m/s deploy band; 28.5% near-stationary,
  40% at 0.5–0.8. The fine-tune set is well-targeted (27.9% in-band) but is 122
  clips against 33,206.
- **TRAINING CORPUS CONTAINS CORRUPTION.** `np.diff` on stored `dof` arrays:
  median max per-frame joint jump 0.293 rad, **p99 = 2.917**, max 3.127 rad
  (179° in one 33 ms frame). **4.6% of sampled clips exceed 1.0 rad**; 47.7%
  exceed 0.3. Priority set is much cleaner (0.8% > 1.0 rad). The `_feasible`
  filter was for *dynamic* feasibility (G1-SONIC executed it), **not kinematic
  continuity**. Operator visually confirmed a broken pose.
  → **This is the strongest open lead.**
- **G1→X2 retarget deflates, does not inflate**: speed ratio 0.820× (tight,
  unimodal), yaw ratio 0.406× median but **bimodal** (clusters at ~0.25 and
  ~1.5). A geometric retarget should be a consistent transform; yaw is not.
- **fps/horizon identical between X2 and G1 stacks** (30, 2.0 s, 4 frames/token)
  — so intent encoding and units are not the differentiator.

## MEASURED but condition-dependent — treat with care

- **Yaw over-tracking.** Under *constant synthetic* yaw intent, X2 produces
  ~2.8–3.4× commanded; G1 produces 1.01× on the identical harness. Independently
  corroborated by `run_scripted_demo.py` docstring ("X2 over-tracks ~2.35×,
  G1 ~1.27×"), written before this session.
  **BUT** under *real clip intent* (chunked, from `idle_turn_270_002`), X2
  measured **0.71× — it UNDER-rotates.** Which condition resembles the gamepad
  is unknown. The pad sends fixed ±0.30 via a sign resolver (closer to
  synthetic), but the synthetic test also used a mismatched walking seed.
- **Root and pose agree with each other** (0.07 rad divergence over 6 s of
  turning), so the defect does not isolate to one model.
- **Legacy vs deployed path agree within 2%** (2.50/2.98/3.09 vs 2.49/3.03/3.01)
  — the 30→50 resample is NOT the source of yaw error.
- **Replan cadence sweep** (harness only, forced timing): over-rotation fell
  3.11× → 1.77× by consuming the whole prediction instead of ~28% of it.
  Deployed already consumes ~75%, so real headroom is much smaller.
  X2's prediction opens with a *wrong-direction* yaw transient (first 25% at
  −0.18 rad/s for a +0.40 command); G1's is correct from frame 0.

## DEAD — do not re-explore (each was tested and refuted)

| hypothesis | how it died |
|---|---|
| Replay-vs-generative is the root cause | G1-planner output works on hardware |
| Mode-table swap caused it | Restoring `.bak_default` made it **worse** (0.37 vs 0.26) |
| Data starvation at low speed | Fine-tune set is 27.9% in-band and upweighted 30% |
| Retarget inflates stride/speed | Measured 0.820× — it deflates |
| fps mislabel (120/30, 50/30, 90/30) | All corpora uniformly 30.0 |
| Height-convention mismatch on robot | Deployed anchor + clips both new convention |
| Root model in isolation | Root and pose agree to 0.07 rad |
| Chain-matched corpus had feet-in-air | Measured: 100% median planted |
| Foot slip predicts hardware success | `walk_circle_001` has the *highest* slip and works |
| `ref-smoother halfcos` | Real bug (I added it; now reverted locally) but too recent to explain a month-old regression |

## RETRACTED findings — I claimed these, they were artifacts

All caused by **one** error class: choosing a floor/contact datum from a clip's
extremes, where a transient frame sets it.

- "constant 0.18 m/s foot-slide floor" — computed over 3–5 contact frames
- "feet never plant" — frame-0 datum artifact
- "feet never lift" — inverted the above, also unreliable
- "planner makes contact 14× less" — became ~100% vs 12–29% with a robust datum
- the 0.33 slide-ratio bar, X2-vs-G1 slide comparison, per-segment slide table

**Every foot-contact/foot-slide number from this session should be discarded.**
Four metric versions produced four different answers, and in the final version
the known-good `relaxed_walk` clip ranked *worst*.

---

## Tooling built this session — trust levels

| file | trust | note |
|---|---|---|
| `gear_sonic/scripts/kplanner_frame_eval.py` | **LOW** | contact/slide logic unreliable; yaw/root-speed/jump parts are fine (no datum) |
| `gear_sonic/scripts/kplanner_intent_probe.py` | medium | drives planner from a clip's chunked intent; the intent extraction is sound |
| `gear_sonic/scripts/plot_planner_timeline.py` | medium | seam detection is INVALID (12 false seams on a clip with zero replans) |
| `gear_sonic/scripts/view_multi_qpos.py` | unverified | N robots in one viewer; operator does not trust it |
| `gear_sonic/scripts/play_motion_mujoco.py` | **modified** | added `--key` (substring match). Additive; no-key path byte-identical. Pre-existing script otherwise. |

Scratch scripts in the session scratchpad: `cadence.py`, `yaw_char.py`,
`buf2.py`, `deployed2.py`, `root_vs_pose.py`, `retarget_speed.py`.
**Watch for shared-planner state contamination** — `_resample_active` latches;
always construct a fresh planner per measurement cell.

## THE DISCIPLINE RULE (violated repeatedly this session)

**No metric may be interpreted until it scores known-good motion correctly.**
`relaxed_walk_forward_v1` and `walk_circle_001` both walked the real robot. Any
metric ranking them badly is broken — the metric, not the motion. And when the
operator's eyes disagree with a number, the eyes have won every single time.

## Next steps, in priority order

1. **Chase the corpus corruption.** Check whether discontinuities exist in
   `x2_ultra_bones_seed.pkl` (pre-retarget, matching keys minus `_M`). If the
   retarget *manufactures* them → fixable script bug. If inherited → data
   problem. This is cheap and is the strongest lead.
2. **Determine which yaw condition matches the gamepad** — synthetic constant
   intent (3×) vs real clip intent (0.71×).
3. **Stage-2 evaluation, never run**: feed planner output through SONIC in
   MuJoCo and measure achieved vs reference. All of this session was stage 1
   (planner alone). `eval_x2_mujoco.py` has `--clip` and `--motion` but requires
   a `.pt` checkpoint; only the ONNX is local.

## State

- Robot **powered down** by operator on a **hip joint temperature** warning
  after a full-day stress test. Nothing pending on it.
- `ref-smoother halfcos` removed from `x2_pc2/ritual_start_demo.sh` **locally
  only** — not yet copied to PC2.
- Rumble fix (`pad_locomotion_bridge.py` + `pc2_pad_daemon.py`, PULL port 5570)
  complete locally, **not copied to PC2**.
- Pad bindings working on hardware, incl. **L2+R2+Y held 2 s → `walk_circle_001`**.
  D-pad does not reach the daemon on this pad/driver — do not spend time on it.
- **Nothing committed this session.**
- `motionbricks/out/X2-clip.{ckpt,modes.json}.SAFETY_current` left in place;
  clip table restored and md5-verified (`1c5f5752d159`).

---

## ADDENDUM (follow-up session) — corpus corruption ROOT-CAUSED

**The `x2_from_g1` retarget run manufactured the corruption.** Evidence chain,
each step verified against local files:

1. **Retarget vs seed, same frames**: of 313 corrupt retargeted clips (sample
   of 6011), 100% have seed counterparts; at the exact frames where the
   retarget jumps 3–4 rad, the seed moves 0.06–0.45 rad. Only 6% of seeds
   are also dirty.
2. **Isolated to one retarget variant**: for
   `arc_walk_left_start_001__A032_M`, the G1 source CSV and all three sibling
   retargets (`x2`, `x2_uniform_h14`, `x2_chain_matched`) have max jump
   0.10–0.13 rad; **only `retargeted/x2_from_g1/` has the 3.10 rad jump**.
3. **Mechanism — wrong IK basin, not spikes**: frames 0–253 of that clip sit
   in a garbage branch (root rolled 158°, hip pitches −154°, waist pinned at
   its ±18°/±28° soft limits), then snap to sane at frame 254 (root yaw jumps
   116° in one frame). The solver is a warm-started LM tracking IK
   (`g1_to_x2_pipeline.py` `_inject`, init = 10 repeats of frame 0, soft joint
   limits w=10) run via `batched_g1x2_driver.py` with no output sanity check.
   Once an env converges into a folded basin it tracks keypoints from inside
   it for hundreds of frames.
4. **Jump counting badly undercounts**: clips that never leave the bad branch
   show ZERO jumps. Direct residency metric (root tilt >60° from upright OR
   |hip pitch| >2.0 rad): several `walk_ff_*_270` / `crouch_ff_loop_270`
   clips are **100% bad** — entire walking-turn clips that are pure garbage.
   **Operator visually confirmed** `walk_ff_loop_270_001__A065` in MuJoCo:
   the whole clip is bad, per the discipline rule the eyes agree with this
   metric.
5. **Differential quantification** (bad in retarget AND seed-clean, so
   excluding genuine handstand/sitting content): **808 clips, 2830 s of
   garbage reference**, list written to
   `docs/experiments/x2_from_g1_bad_branch_clips.txt`. **52% are
   turn/arc/270/360-named** — the corruption concentrates precisely in
   turning clips.
6. **What it teaches the model**: root yaw-rate inside bad frames has
   p99 = 26.3 rad/s (max 94) vs 3.5 rad/s in clean frames. The corpus tells
   the root model that, in turning contexts, the root can spin at 10–25×
   physical rates while the legs do something unrelated. Clean-frame max is
   still 87 rad/s → the residency mask is a FLOOR; upright-preserving branch
   flips exist beyond it. Add a |yaw-rate| gate (e.g. >6 rad/s) to any
   corpus filter.

**Can 7.5% of clips explain a 3× yaw error? Measured, not argued:**

- Full gate set (residency + jump>1.0 rad + yaw-rate>6 rad/s vs seed control)
  drops **2,493 clips (7.5%)** — `filter_kinematic_continuity.py`, output
  `x2_ultra_bones_seed_g1_retarget_feasible_kinclean.pkl` + `.dropped.txt`.
  The yaw-rate gate alone caught 690 clips invisible to both the jump scan
  and the tilt metric (upright-preserving discontinuities).
- **Refuted**: contamination does NOT concentrate in the deploy regime —
  deploy-band(0.1–0.3 m/s)+turning clips are 2.6% bad, same as the corpus
  average. "It's concentrated where we deploy" is wrong.
- **Confirmed**: the dropped 7.5% of clips carry **72.4% of the corpus's
  total squared root yaw-rate mass** (26.0% of squared dof-velocity mass).
  Under any L2-style objective on root motion, the yaw-rate signal the root
  model fit was ~3:1 dominated by garbage. This is the quantitative bridge
  from "7.5% corrupt" to "learned yaw dynamics are wrong" (assumes a
  continuous-regression root loss; weaker if the objective is tokenized or
  huber).

**This coherently explains** open items 1, 3, 5, and 8 (yaw over-rotation
learned from corrupt turning clips; wrong-direction opening transients =
learned branch-snap reversals; rotation-without-stepping = literally present
in the training references; G1 stock is clean because it never saw this
corpus). It also explains the bimodal retarget yaw ratio (0.25 / 1.5
clusters = branch-flipped vs clean clips). It does NOT yet *prove* the 3×
magnitude — that proof is a retrain on a filtered corpus.

**Fix options** (in effort order):
- Rebuild the fine-tune corpus dropping the 808 listed clips (+ yaw-rate
  gate), re-run the vqvae/root/pose fine-tune. Cheapest falsifiable test.
- Re-retarget the affected clips per-clip (G1 sources are clean); add a
  post-solve sanity check (tilt/limit/yaw-rate) to `retarget_batch` that
  re-solves diverged envs.
- Permanent: add the kinematic-continuity + residency gate to the corpus
  build next to the dynamic `_feasible` filter, which cannot see this (it
  filtered on G1-side executability, never inspected the X2 output).

---

## Reference clips (all in `x2_ultra_bones_seed_g1_retarget_feasible.pkl`)

View with:
`play_motion_mujoco.py --motion <that pkl> --key <substring>`

**CORRUPTED — visually confirmed broken by operator:**

| key substring | jump | at frame |
|---|---|---|
| `medium_big_heavy_two_hands_behind_low_to_behind_medium_R_001__A520_M` | ~1.85 rad | — |
| `big_light_one_hand_behind_low_to_behind_medium_R_001__A520_M` | 3.60 rad | 40 of 198 |

Others >3.1 rad: `small_light_two_hands_turn_walk_270_R_001__A505` (3.23 @145),
`locobal__body_search_002__A412_M` (3.22 @331),
`turn_walk_big_dog_ff_270_R_002__A493` (3.20 @57),
`arc_walk_left_start_001__A032_M` (3.15 @63).

**CLEAN in-place turning manipulation — good baselines:**

| key substring | jump | dyaw | travel | dur |
|---|---|---|---|---|
| `medium_heavy_two_hands_idle_turn_270_R_002` | 0.108 | 94° | 0.18 m | 5.8 s |
| `small_heavy_two_hands_idle_turn_360_R_001` | 0.143 | 174° | 0.24 m | 7.6 s |
| `medium_big_heavy_one_hand_idle_turn_270_R_001` | 0.144 | 91° | 0.06 m | 6.5 s |

Clean control (non-manip): `idle_turn_045_R_long_002__A549_M` (jump 0.23).

**Corruption rate by family** (8000 clips sampled), % with max per-frame jump
>1.0 rad: `locobal` 9.9%, `locowalk` 5.0%, `locopost` 4.1%, `locomanip` 2.7%.
`locowalk` — the walking clips, most relevant to the regression — is the
worst of the large families: only 43.6% are clean (<0.3 rad).

**NOTE:** several clips named `idle_turn_270` measure only 90-98° of actual
yaw. Possible trim artifact or label mismatch — unverified, but anything
trusting those labels should check.
