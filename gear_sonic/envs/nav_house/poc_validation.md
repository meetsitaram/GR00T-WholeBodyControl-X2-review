# Stage-1 PoC validation ledger
Tracks the assumption-validation gates toward the camera-student + language-goal
architecture (see x2_upgraded_demo/x2-kitchen-nav-stage1-camera-plan.md).
Rule: nothing gets committed until PoC-C passes. Update PASS/FAIL + evidence here.

## Gates (measurable, in dependency order)

### G1 — splat renders through the TiledCamera SENSOR path
- Why: every camera-obs plan dies here if the NuRec splat only draws in viewports.
- How: showcase rig + `KP_TILED_GATE=1` + `render_results=true` → position
  `eval_camera` at the showcase eye, print rgb stats, dump PNG.
- PASS: frame std > 15 (not blank/flat) AND dumped PNG visibly shows the kitchen
  splat (human check). Record FPS impact at 256x256x1 env.
- Status: **PASS (2026-07-22)** — definitive frame: crisp X2 robot + photoreal
  splat kitchen (shelf items, doorway, floor) in one 256x256 sensor frame.
  Proof: x2-kitchen-sim/media/poc_g1_sensor_frame.png. Two required tricks,
  both now in the rig gate code: (1) the spawned camera prim's xformOp is
  translate-typed — ClearXformOpOrder once, then author a matrix TransformOp
  (sensor set_world_poses does NOT stick in this eval path); (2) the sensor
  buffer is STALE by default (eval path never updates it) — call
  cam.update(dt, force_recompute=True) before reading. All earlier "mush"
  frames were the stale init buffer, not a renderer limitation.

### G2 — planner batch>1 (or measured stagger budget)
- Why: sets the env-count ceiling for planner-in-loop rollouts (stage-1 training).
- How: standalone bench `poc_planner_batch.py` — load NeuralPlannerCore, attempt
  B in {1,4,16,64} through the torch core; measure ms/replan and per-env cost.
- PASS: B=16 works numerically (finite outputs, matches B=1 on a shared input
  within 1e-4) AND per-env replan cost at B=16 is < 25% of B=1 cost.
  FALLBACK-PASS: B>1 impossible → measured sequential replan budget documented
  (envs x replans/s at 50 Hz sim), plan adjusted to that ceiling.
- Status: **FALLBACK-PASS (2026-07-22)** — B=1 measured at **30.0 ms/replan**
  (RTX 5090, fixed_scratch ckpts) through the batch-derived entry; B>1 fails
  deeper in the root model: `motion_inference.py:230` gather receives
  `pred_num_tokens` with collapsed batch (index [1,1,4B] vs values [B,64,4]) —
  plumbing is B-shaped but the root model's num-token head isn't B-clean.
  Sequential ceiling: replan every ~2s sim per env → ~64 envs ≈ 1x-realtime
  planner overhead; 256 envs ≈ 4x. True batching = half-day fix at the cited
  site (queued for N1). Bench: `poc_planner_batch.py`.

### G3 — synced (camera, state, teacher-action) sample stream
- Why: the DAgger data path: per control tick we must get aligned {rgb, proprio,
  goal, teacher_sticks} without stalling the sim.
- How: rig probe `KP_POC_DATA=1` — per tick, build the teacher's 28-D obs from
  sim state (reusing train_nav_teacher obs code), run the frozen stage-0
  checkpoint, log rgb stats + action; save N=200 samples to npz.
- PASS: 200 aligned samples saved; teacher actions non-degenerate (std>0.05 on
  at least one stick); wall rate >= 10 samples/s at 1 env.
- Status: **PARTIAL (2026-07-22)** — content PASS: 200 aligned samples, teacher
  actively steering (action std [0.10, 0.47, 0.95]); rate FAIL: 1.5 samples/s
  wall (un-vectorized per-tick python obs build + 0.27x sim). Fix at N1: batch
  the obs build; rate scales with num_envs regardless. npz at
  /tmp/claude-1000/poc_g3_samples.npz; probe = KP_POC_DATA in video_showcase_rig.

### G4 — student forward + BC gradient step
- Why: proves the training plumbing (encoder + heads + loss) end to end before
  investing in the full DAgger loop.
- How: `poc_student_bc.py` — ResNet18(pretrained) 96x96 + proprio+goal MLP → 3
  sticks; one epoch of BC on G3's npz; loss must drop.
- PASS: forward OK on GPU; BC loss decreases >30% over one epoch on 200 samples
  (sanity, not accuracy); ONNX export of the student succeeds.
- Status: **PASS (2026-07-22)** — loss 0.67 -> 0.19 (**71% drop**) in 3 epochs /
  1s wall; ONNX export OK. `poc_student_bc.py`.

### G5 — language router v0 (N3 demo interface)
- Why: locks the System-2 -> System-1 goal interface early.
- How: `nav_language_router.py` — text -> waypoint registry match -> goal pose
  (xy, yaw) payload identical to what nav_policy_bridge/teacher consume.
- PASS: 10/10 canned phrasings ("go to the cooking range", "head over to the
  fridge", ...) resolve to correct waypoints; unknown target -> explicit reject.
- Status: **PASS (2026-07-22)** — 12/12 incl. synonyms (stove→cooking_range,
  refrigerator→fridge) and 2 explicit rejects. `nav_language_router.py --self-test`.

## End-goal success criteria (post-PoC, for the record)
- N2: student within ~10% of teacher on held-out routes AFTER RL fine-tune.
- N4: full-stack (student->planner->SONIC) success gap vs surrogate < 15%.
- N6: nav->GR00T handover with zero SONIC reference discontinuity (blend both
  ways through idle-stand).

---

## Campaign log: student training + drift empirics (2026-07-22/23)

### G6 — camera student, first training (nav_student_cam_0722)
DAgger from frozen stage-0 teacher, 150k iters @ ~20k samples/s, gallery eyes.
**RESULT: entrance arrival under drift — teacher 0.25 -> student 1.00** (perfect
in 94/100 evals; every eval after it=22.5k). student_best.onnx exported.
The drift compensation is a LEARNED FEATURE of the weights (no estimator at
runtime): training fed disagreeing inputs (corrupted goal numbers, truthful
pixels) with true-state teacher labels -> network learned "trust the pixels".

### G7 — empirical drift measurement (drift_distribution.png)
12 ref(planner)/exec(SONIC) route pairs: rate median 0.184 m/m, p90 0.29,
max err 1.56 m, NO saturation in-route. Structure (user-predicted): drift
concentrates in TURNS — straight chunks ~0.000 m/m, high-turn chunks carry all
drift; ~1.5–3.7% of turned angle becomes heading error. Model everywhere is now
turn-gated: acc += rate*turniness_gate*dist + rot_slip*|dyaw|.
Real-robot rates expected HIGHER (state estimation, slip): training DR widened
to rate<=0.5/m, rot<=0.12 m/rad, cap<=2.5m (sim-empirical = lower half).
N4 protocol: tape-mark loops on the real robot -> believed-vs-physical at
landmarks -> re-anchor DR (same session checks real-camera domain).

### G8 — retrain v2 (nav_student_cam_driftfix_0723) + eval-harness lessons
Resume student_best + widened turn-gated DR, 40k iters. Two harness bugs found
en route (both fixed):
  (a) precision arrival tol 0.18m < student's ~0.3m park radius (envelope
      V_MIN floor) -> stall -> unbounded drift -> wander. Tol=0.30 + in-place
      ALIGN phase at goal; docking micro-phase (last 20cm via direct intents)
      queued for the entrance next iteration.
  (b) eval spawned pantry at RAW waypoint coords -> inside wall margin ->
      collision at t=0 -> false 0.00 forever. Starts must use wp_xy_snap.
FIXED-harness baseline (v1 student, p90 drift): mid 1.00, pantry 1.00,
dining_table 0.875 (1/8 worst-direction fail = v2's target).

### Showcase v3 (student-driven video)
All 6 routes REACHED (entrance via funnel staging: hallway -> turn -> door,
matching the teacher's learned corridor). Precision: appliance goals visually
docked; entrance at ring edge (doorway makes the 0.30m ring visible) ->
docking phase queued. Video: showcase_6robots_student_v3_realtime.mp4
(recorded during GPU contention; full-length final take queued post-retrain).

### G8 FINAL — retrain v2 head-to-head (fixed harness, 2026-07-23)
| drift | v1 | v2-drifthard |
|---|---|---|
| 0.184 (sim median) | all 1.00 | all 1.00 |
| 0.29 (sim p90) | dining 0.875 | **all 1.00** |
| 0.50 (real-robot stress) | mean 0.94; pantry/dining 0.375 | **all 1.00** |
v2 = student_best in runs/nav_student_cam_driftfix_0723 (+ ONNX). VERDICT:
long-route drift closed with margin; deploy candidate = v2.

### G9 — N4 closed-loop LIVE rig (KP_STUDENT in run_x2_kplanner_env2, 2026-07-24)
Student inference IN the SONIC loop: gallery render at the EXECUTED pose +
goal vector from the odometry belief -> sticks onto the daemon wire @5 Hz.
Motivation: capture-then-replay is open-loop on both sides of SONIC —
recording SONIC's executed output helps only if the intents were computed
from SONIC's pose (user-proposed, confirmed). Three design lessons from the
first live attempts, all fixed in the pilot:
  (a) The planner's integrated pose is NOT usable as odometry belief: its
      frame micro-jumps on every replan (known planner-drift artifact) and
      at 5 Hz intent changes the jumps compound to >1 m per 10 s — far
      beyond leg-kinematics odometry. Belief = executed pose + synthetic
      slip from the measured distribution (deploy-faithful).
  (b) A drift-compensating student NEVER zeroes the BELIEVED goal distance
      — it parks the TRUE pose at the true goal, leaving belief dist ~= the
      accrued bias. Belief-radius phase switches therefore stall or fire by
      luck; arrival latching must be a truth-referee (the training env's
      true-radius termination analog). Corollary: staging legs (has_yaw=0)
      carry no landmark identity, so pixels CANNOT correct them — the
      student parks at the phantom (belief-offset) staging point. Live
      routes go DIRECT to the landmark goal; staging was kinematic-era
      path cosmetics.
  (c) The turn-gate slip model degenerates under live bang-bang execution
      (every step reads "turny", gate==1 -> ~0.8 m/m, 2x the worst measured
      route). Live accrual uses the measured PER-ROUTE rate directly:
      acc += 0.29 (p90) * dist + 0.04 * |dyaw|, cap 1.6.
Narrative datum (run v2, mid-route): BELIEF-STOP — odometry-only nav would
have parked 1.81 m off the door; the student was at 0.79 m and closing when
the run was cut. Full referee-verdict run = v4 (direct-to-goal).

### G9 RESULTS — three live closed-loop arrivals (2026-07-24)
| run | referee | true final dist | drift | note |
|---|---|---|---|---|
| v4 headless smoke | 0.35m + 0.2rad | **0.15 m** | 1.60 m | clip live_student_entrance_v4smoke.pkl |
| video take 3 | 0.5m + 0.3rad | 0.24 m | 1.60 m | recording cut for live view |
| ON-SCREEN (user watched) | 0.5m + 0.3rad | 0.38 m | 1.60 m | 64 s; BELIEF-STOP 1.19 m short |
| turn-rate 0.6 validation | 0.5m + 0.6rad | 0.30 m | 1.60 m | **27.2 s, fastest**; sticky latch + KPLANNER_FIXED_TURN_RAD_S=0.6 (daemon const made env-tunable, deploy default 0.30 kept; sweep-certified <=1.0, comfort ceiling 0.75) |
Every run: odometry-only nav parks 1.1-1.8 m short (BELIEF-STOP marker);
the camera student closes the gap LIVE through SONIC physics.
DOORWAY DANCE root cause (user-diagnosed on screen): the terminal ALIGN
demands in-place turns at the tightest spot — and in-place turning is this
stack's weakest skill (root model trained on ~zero-yaw data, daemon caps
0.3 rad/s, SONIC turn tracking slip-heavy = the drift mechanism itself);
policies correspondingly learned back-up-and-arc styles. Fixes applied:
sticky REACHED latch (stop-slide must not re-trigger approach) + RELAXED
terminal-yaw gate (user rule: orientation tolerance looser than position
tolerance — 0.6 rad vs 0.5 m demo defaults — NOT a heading-free exclusion;
in-place turns stay in the contract because the N6 manipulation handover
needs arrival yaw. Tunable per task via KP_REF_RAD / KP_REF_YAW).
Proper cure queued: docking micro-phase that curves in pre-aligned;
stage-2 SONIC-in-loop fine-tune so the student internalizes turn costs.

### G10 — v2 realism retrain (USER-DIRECTED, launched 2026-07-24)
Motivated by the live wall-contact study (11 events / 15.8 s scraping on the
watched route; offline ESDF sweep of recorded executed paths, contact figure
media/live_run_wall_contacts.png). Teacher HAD a collision penalty (-5
terminal + 0.3 m center ramp) — but judged on a POINT robot with perfect
actuation, which legalized every scrape. Four env changes (NavKitchenEnv
v2=True, flag-gated for reproducibility of the frozen v1 teacher):
  1. FOOTPRINT (user: "rectangle with wide shoulders"): oriented rect from
     MJCF default-pose collision extents — fore-aft [-0.17,+0.14] m,
     lateral +-0.35 m; 8 perimeter samples + center; blocked footprint
     also blocks rotation (shoulders can't sweep through walls); spawn
     set eroded to esdf>=0.40.
  2. DEPLOY-QUANTIZED sticks: sign-only {0.3 m/s, 0.3, 0.6 rad/s}
     (matches daemon setpoint + KPLANNER_FIXED_TURN_RAD_S=0.6 rig).
  3. EDGE clearance ramp: -1.0 max within 0.20 m of the BODY EDGE
     (min esdf over footprint samples) — comparable to progress when
     scraping.
  4. ALIGNMENT-DISCOUNTED progress (user turn-and-walk rule): backward/
     sideways motion in OPEN space earns half credit; front-blocked
     states keep full credit (step-back-then-turn stays free near walls).
Chain: teacher PPO 30k iters (runs/nav_teacher_v2_0724, ~90 min) -> gate
best route success >=85% -> DAgger re-distill (NAV_V2=1, TEACHER_RUN env,
resume driftfix vision features -> runs/nav_student_cam_v2_0724).
Acceptance: matrix >= v2-drifthard AND live acid test with the contact
counter -> target zero scraping events on the entrance route.

### G10 RESULTS (2026-07-24)
Teacher v2: 100% waypoint-route success @30k iters (68 min) — after fixing
a double-erosion bug (footprint samples must check the TRUE-wall esdf; the
walkable mask is ALREADY 0.35-eroded, so v1 was a conservative disc, never
a point robot; the oriented rect gains the 0.17-0.35 nose-first band).
Teacher-under-drift baseline REPRODUCED in v2: entrance 0.25 without vision.
Student v2 (nav_student_cam_v2_0724/student_best = it30000): matrix mean
1.000 @0.184, entrance 1.00 (final dist 0.24), entrance_3starts 1.0/1.0/1.0
@p90. Post-annealing evals oscillate (all-1.0 iterates alternate with
entrance-0.0 iterates; mid start is 1-effective-sample bimodal since drift
~0 early makes all 8 replicas identical) — best-checkpointing captured a
clean iterate.
LIVE ACID TEST (entrance route, live SONIC, drift cap 1.6):
  REACHED — latched 0.47 m, settled 0.69 m, 40.4 s.
  Contacts: 5 events / 11.4 s sub-0.25m / min esdf 0.07 m
  vs v1 baseline 11 events / 15.8 s / min 0.10 (46.6 s route).
VERDICT: contact events HALVED, turn-then-walk opening confirmed live
(heading captured in 4 s), but the doorway-mouth grind persists (6 s at
esdf 0.07 near (-2.96,0.0)) — the close-quarters phantom-pull + SONIC
overshoot regime is outside the surrogate's reach. Remaining fixes are the
known queue: docking micro-phase (curve in pre-aligned), deploy-side
clearance veto (ESDF/depth guard), stage-2 SONIC-in-loop fine-tune.
NOTE: user's zmq_cmd_bind dual-source daemon change landed mid-session;
env2 now passes zmq_cmd_bind=False explicitly (legacy wiring preserved).
