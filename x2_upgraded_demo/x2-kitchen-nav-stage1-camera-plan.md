# Stage 1: Cameras in the pipeline — teacher-student curriculum + language goals
2026-07-22. Research-backed plan for the next phase of x2-kitchen-sim navigation.
Sources: repo audit (below), Flexion/Niantic/NVIDIA publications, NaVILA (RSS'25),
GR00T N1.5–N1.7 documentation. Decision needed from user on Phase ordering only;
architecture recommendation is firm.

## 1. The core question, answered

**"Is pure RL needed, or should the student be a GR00T VLA that takes 'go to the
cooking range'?" → Both, but at different layers. Neither replaces the other.**

The evidence says split the system NaVILA-style (RSS'25, 88% real-world success on
legged robots including humanoids):

- **System 1 — local visual navigation (5 Hz, reactive):** an RGB point-goal policy.
  This should be a small trained policy (RL and/or distillation). Honest note on
  rates: GR00T N1.7 with TensorRT reaches ~36 Hz (H100) / 50–80 Hz (RTX 5090
  community), so raw speed is NOT the blocker — the case rests on (a) DualVLN's
  head-to-head win of modular over end-to-end, (b) **zero published precedent** for
  GR00T emitting base-nav velocity commands (every documented G1 deployment routes
  the VLA through SONIC WBC — the `velocity_commands [vx,vy,yaw]` interface is the
  WBC/kinematic-planner side, i.e. literally our kplanner intent wire), and
  (c) training-data economics (a small student trains on millions of cheap sim
  rollouts; a VLA fine-tune needs curated episodes). Flexion validates the
  component in exactly our world type: RGB-only, goal-in-robot-frame, **direct RL
  on rendered images (no teacher-student)**, frozen offline-trained encoders,
  millions of rollouts, DR — 97.8%/75.0% sim success (1,024-rollout evals),
  zero-shot real transfer (no quantitative real numbers published).
- **System 2 — language goal layer (0.1 Hz, deliberative):** maps "go to the cooking
  range" → a goal pose for System 1. For our 8 named waypoints this needs **zero
  training** (instruction → waypoint-registry lookup, any small VLM or even a keyword
  router). A GR00T N1.7 fine-tune becomes worthwhile only for *open-ended* language
  ("check what's on the stove", "go to the thing that's beeping") — and our repo
  already contains the full N1.7 fine-tune + X2/SONIC integration to do it later.

Key literature caveats that shaped this:
- Generalist ViTs (DINOv2/v3) **underperform for navigation** as frozen encoders;
  encoders distilled from heterogeneous teachers, or robust offline-trained encoders
  (Flexion's choice), do better (arXiv 2606.21216). Don't default to DINOv2.
- Flexion used **pure RL** (no teacher-student) — but they paid "millions of
  rollouts". We have something they didn't: a 99.96% state-based teacher. DAgger
  distillation into the RGB student is strictly cheaper than visual RL from scratch;
  keep a short on-policy RL fine-tune of the student as the polish pass.
- GR00T N1.7's native output is **latent motion tokens decoded by SONIC-class WBC**
  (that's literally this repo's lineage) — a nav fine-tune would have it emit nav
  *intents* instead; that's a new action head, not free.

## 1.5 The frozen contract (nothing below the intent wire changes)

```text
 "go to the cooking range"
 +--------------------------------------------------+
 | System 2: language layer                         |
 | router -> VLMaps -> (stretch) GR00T N1.7         |
 +-------------------------+------------------------+
                           | goal pose (robot frame)
                           v
 ...........................................................................
 :  DAgger TRAINING LOOP (fast surrogate; planner NOT in loop)             :
 :                                                                         :
 :  +---------------------+  true state   +--------------------------+     :
 :  | STAGE-0 RL TEACHER  |<--------------| surrogate rollout        |     :
 :  | nav_teacher_0722c   |               | (2D unicycle, kitchen    |     :
 :  | MLP on 28-D priv.   |               |  walkable map, 512 envs) |     :
 :  | state (goal+vel+    |               | + drift DR (15%/m bias)  |     :
 :  | 16 ESDF rays)       |               | + push DR + visual DR    |     :
 :  | 99.96% - FROZEN     |               +------------+-------------+     :
 :  +----------+----------+                            |                   :
 :             | label: "what I would do"              | pose (x,y,yaw)    :
 :             | (MSE target, 40k replay buffer)       v                   :
 :             |                       +--------------------------------+  :
 :             |                       | BAKED GALLERY (7392 sensor     |  :
 :             |                       | renders of the splat kitchen,  |  :
 :             |                       | 0.2m x 16-heading lookup)      |  :
 :             |                       +---------------+----------------+  :
 :             v                                       | 96x96 RGB         :
 :  +=====================================================================:
 :  |                      RGB STUDENT (~11.7M params)                    |:
 :  |                                                                     |:
 :  |   96x96 RGB --> [ ResNet18 encoder ] --> 512 --> [proj] --> 128 \   |:
 :  |   (ImageNet init,      pretrained,                          [FUSION]|:
 :  |    trainable)                                               [ HEAD ]|:
 :  |   12-D state --> [ state MLP 128x128 ] ------------> 128 /  3-layer |:
 :  |   (goal-in-body + dist + goal-yaw (6)                       MLP+tanh|:
 :  |    believed vel (3), prev action (3)                           |    |:
 :  |    -- DRIFT-CORRUPTED, NO rays)                                |    |:
 :  +=================================================================|===+:
 :....................................................................|....:
                                                                      |
                              3 sticks (fwd, lat, turn) @ 2-5 Hz      |
                              SAME action space as teacher            |
                              planner_cmd:5563                        v
 +---------------------------------------------------------------------+
 |  FIXED - never retrained, out of the training loop                  |
 |  (re-enters only at N4 full-stack eval; optional M3.5 polish)       |
 |                                                                     |
 |     kplanner  ------>  SONIC  ------>  robot                        |
 +---------------------------------------------------------------------+

 DAgger mechanics: student drives the rollouts (teacher-driven fraction
 anneals 1->0 over first 3k iters); teacher labels EVERY visited state
 from true state; student regresses to labels (MSE) — learning to output
 teacher-optimal sticks from drifted belief + pixels. RL fine-tune stage
 (evidence: closes distill gap and exceeds teacher) reserved as follow-up.
 First result 2026-07-23: entrance arrival under 15%/m drift —
 teacher 0.25 -> student 1.00 (it=9000, 8 drift directions).
```
The student swaps only the teacher's *inputs* (privileged ESDF rays + exact pose →
camera RGB + proprio + goal). Output wire identical → drop-in replacement in
nav_policy_bridge and on deploy. Distillation and the student RL fine-tune run in
the same cheap surrogate rollout the teacher trained in; planner+SONIC re-enter at
N4 as the full-stack deploy-parity eval, exactly like stage 0's live demo.

## 2. What we already own (repo audit, 2026-07-22)

| Piece | Where | State |
|---|---|---|
| DAgger distill trainer (teacher rollout scheduling, BC loss, aux heads) | `external_dependencies/GR00T-VisualSim2Real/gr00t/rl/trl/trainer/distill_trainer_obj_pred_homie_api.py` | Working, G1 loco-manip, not nav |
| RGB student encoders: ResNet18/34/… pretrained + DINOv3 ViT family | `gr00t/rl/agents/modules/modules.py:303-474` | Working |
| RGB-delay obs buffer (1–5 frame latency DR) | `gr00t/rl/config/obs/.../obs_..._distill_rgb_delay.yaml` | Working |
| Ego TiledCamera wiring + rgb/depth getters | `gr00t/rl/simulator/isaacsim/isaacsim.py:1381-1823` | Working |
| Student ONNX export | `gr00t/rl/eval_agent_trl.py:343-402` | Working |
| Stage-0 nav teacher (28-D state, 3 sticks, 99.96%) | `gear_sonic/scripts/x2-navigation/train_nav_teacher.py` + `runs/nav_teacher_hardened_0722c` | Done |
| Kitchen splat world loader (NuRec visual + collision) | `gear_sonic/envs/manager_env/modular_tracking_env_cfg.py:357-374` | Working (flagged "num_envs=1 eval") |
| Camera obs group + tiled-image normalizer | `gear_sonic/envs/manager_env/mdp/observations.py:452-460, 2258-2317` | Working, unused |
| OpenCV fisheye→pinhole camera spawner (real-camera match) | `modular_tracking_env_cfg.py:83-183` | Working (needs xvfb headful) |
| GR00T N1.7 VLA + X2 fine-tune/inference plumbing | `external_dependencies/Isaac-GR00T` + `gear_sonic/utils/inference/vla_utils.py`, `launch_finetune_x2.py` | Working for manipulation motion-tokens |
| Frozen nav env design | `gear_sonic/envs/nav_house/nav_kitchen_v1.yaml` | Spec only, no code |

**The three missing connectors** (all already flagged in nav_kitchen_v1.yaml):
1. A `nav_house` IsaacLab task package that instantiates splat + ego camera + teacher obs.
2. A nav distillation loop (VIRAL's trainer is bound to G1 loco-manip HOMIE plumbing).
3. Batched planner rollout — `NeuralPlannerCore` is batch-shaped internally but
   hardcodes B=1 (`neural_planner.py:755,993`); the ONNX export is static B=1.

## 3. The plan

### Phase N0 — gates (≤1 day, all measurable)
- **TiledCamera × NuRec splat**: render the splat through the *sensor* path (not
  viewport) at 96×96×N envs; confirm the splat appears and measure FPS vs env count.
  This is the #1 risk and is explicitly unverified ("sensor path = gate artifact").
  Known NuRec gotchas (Isaac Sim docs): opening a .usdz as the ROOT stage blocks
  adding references/payloads (use .usd/.usda wrappers — we already reference it, ✓);
  DLSS Frame Generation causes splat artifacts (disable); Isaac Sim 6.0 moves NuRec
  to Fabric Scene Delegate with multi-GPU 3DGS if we ever upgrade.
  Fallback ladder (from the approved plan): plain per-env Camera at low env count →
  bake splat→textured mesh → external gsplat renderer.
- **Planner batching benchmark**: lift the B=1 hardcode in `NeuralPlannerCore`
  (tensors are already batch-shaped) OR measure staggered sequential replans;
  sets the env-count ceiling for planner-in-loop training.
- **Multi-env world layout**: splat is a global prim ("num_envs=1 eval only") —
  decide robots-share-one-kitchen (proven by today's 6-robot scene, needs disjoint
  spawn cells) vs per-env cloned splat (memory cost unknown — measure).

### Phase N1 — nav_house task package (2–3 days)
Implement `gear_sonic/envs/nav_house/` per the frozen yaml: kitchen splat world,
2D-unicycle or planner-kinematic rollout (stage-0 model first — cheap), virtual
gamepad action space, goal sampling from walkable mask, ego TiledCamera on the
head link (96×96, rectified-pinhole matching stereo_head_front_left), obs groups:
`teacher_privileged` (stage-0's 28-D) + `policy_student` (RGB + proprio + goal) +
`camera_rgb`. Reuse today's hard-won knowledge: origin pinning, boundary-velocity
hygiene, debug_vis off (markers are visible to the policy camera!).

### Phase N2 — DAgger distillation to the RGB student (3–5 days incl. training)
Adapt VIRAL's distill trainer to nav (strip HOMIE plumbing; teacher = frozen
stage-0 checkpoint; student = encoder + proprio + goal → 3 sticks; RGB-delay
buffer 1–5 frames). Literature-calibrated expectations (see sources):
- **DAgger-style on-policy distillation is the 2024–26 standard** (Learning by
  Cheating; Humanoid Parkour: from-scratch vision RL got 0–10% success vs
  80–100% for the DAgger student). Expect the student to land **5–25% below the
  teacher** after distillation alone (Parkour-in-the-Wild: −10.4% avg).
- **RL fine-tune AFTER distillation is not optional polish — it's the step that
  closes (and can exceed) the teacher** (+3.1% above experts in Parkour in the
  Wild; Distillation-PPO). Budget for it.
- **Visual DR recipe for splat worlds** (VR-Robo, GS worlds, 93–100% real
  success): camera-extrinsics noise, brightness/contrast/saturation/hue jitter,
  Gaussian blur, additive noise, 0–1 frame image delay. Splat lighting is baked
  → augment in image space.
- **Encoder ladder** (evidence-ordered): (1) pretrained-trainable ResNet18
  (VIRAL default) or shallow from-scratch CNN + heavy augmentation (Hansen et
  al. — competitive with frozen PVRs), (2) frozen robust/multi-teacher-distilled
  encoder (Theia-style; VR-Robo's frozen ViT+PPO worked on GS), (3) DINOv2/v3
  last — generalist ViTs underperform for navigation.
Success gate: student within ~10% of teacher on held-out spawn/goal routes
(teacher = 99.96%) after the RL fine-tune stage.

### Phase N3 — language layer v0: zero-training demo (half a day)
"go to the cooking range" → waypoint registry lookup → goal into the student (or
into today's teacher via `nav_policy_bridge` — works before N2 even finishes).
Router options: keyword match (works today) or any local VLM for phrasing
robustness. This delivers the user-facing language demo immediately and defines
the interface System 2 must satisfy.

### Phase N4 — full-stack eval + SONIC-in-loop polish (per approved M4/M3.5)
Student ONNX drives planner_cmd → kplanner → SONIC → PhysX (the deploy-parity
path we demonstrated in stage 0). Quantify the sim-success vs full-stack gap;
run stage-2 fine-tune only if the gap warrants it.

### Interlude — why the hybrid is now the evidenced default (2025-26 SOTA)
- **DualVLN (arXiv:2512.08186, Dec 2025)**: dual-system — 2 Hz Qwen2.5-VL-7B
  grounds a *pixel waypoint*, 30 Hz diffusion local policy executes — **beats the
  end-to-end VLAs head-to-head** (R2R-CE 64.3 vs StreamVLN 56.9 / NaVILA 54.0),
  and its ablation shows the explicit modular interface is itself load-bearing.
  Deployed zero-shot on Unitree G1. As of mid-2026 no pure end-to-end language-nav
  VLA holds the R2R-CE lead.
- **For a KNOWN, mapped, kitchen-scale environment the literature says VLN models
  are unnecessary**: VLMaps (ICRA'23/IJRR'25) and OK-Robot show semantic-map +
  language grounding + point-nav suffices — in OK-Robot's real-home failure
  breakdown navigation contributed only 4–15% of failures (grounding/retrieval
  dominated); SayNav's low-level point-nav alone hits 98.5% SR in-room. Our
  waypoint router (N3) is exactly this architecture with grounding pre-solved.
- **VAMOS (2026) — the strongest head-to-head**: hierarchical VLA nav (VLM plans
  2D pixel paths, sim-trained affordance MLP enforces embodiment feasibility):
  **90% success vs end-to-end NoMaD 27% / NaVILA 10% / classical stack 53%** on
  real legged+wheeled robots. And across EVERY shipped 2024-26 system (Figure
  Helix-02's 1 kHz/200 Hz/slow-VLM stack, GR00T N1's internal 10/120 Hz split,
  Gemini Robotics-ER emitting points/trajectories for existing low-level
  controllers), the slow-semantic/fast-motor rate hierarchy is universal — the
  only open debate is explicit interface vs jointly-trained latent, and for
  navigation the published quantitative evidence favors explicit. No source
  found claiming end-to-end VLA beats hybrid on real-robot navigation.
- **Flexion Reflect v1.0 (their full-mission demo video) is explicitly modular,
  not one RL model** — despite appearances: (1) VLM mission controller replanning
  live, (2) motion layer = real-data VLA + separate RL skills (nav = global path
  planning + local adaptation; manipulation = contact-rich skills from teleop/RL),
  (3) Reflex RL whole-body controller, (4) FlexComm runtime. The seamless look
  comes from continuous VLM replanning + learned local recovery, not from a
  unified policy. Maps 1:1 onto this plan (VLM↔language layer, motion↔nav student
  + GR00T manipulation, Reflex↔SONIC). Bonus datum: RL-fine-tuning the VLM
  coordinator itself lifted 16-step mission success 38%→90% (SFT vs SFT+RL) —
  a future lever for our language layer. Full layer-by-layer mapping to our
  stack: see §7. https://flexion.ai/news/flexion-reflect-v1.0
- **AgiBot has published NO navigation architecture for X2** (SDK exposes
  locomotion commands, SLAM optional; GO-1 VLA is manipulation-first) — so
  there's no vendor stack to conflict with; our kplanner intent wire is the
  natural low-level interface.
- **Upgrade path between router and VLA (optional N3.5)**: a VLMaps-style
  open-vocabulary semantic map built over our kitchen splat — enables "go to the
  microwave" for objects NOT in the waypoint registry, still zero policy training,
  still feeding point-goals to the same student. Cheaper and better-evidenced than
  a VLA fine-tune for object goals; the VLA (N5) remains the answer only for
  instruction-following ("past the couch then left") and truly open-ended tasks.

### Phase N6 — manipulation handover (USER-DEFINED TARGET ARCHITECTURE)
The system's end state (user decision, 2026-07-22): **nav drives to the waypoint,
then hands control to GR00T N1.7 for manipulation at the destination** — the one
role where the VLA is unambiguously evidence-backed (its documented lane, and our
repo already runs N1.7 as a motion-token commander through SONIC via vla_utils/
run_vla_inference; fine-tune via launch_finetune_x2 on the 5090).
Key insight: both controllers already speak SONIC — nav via kplanner intents, VLA
via decoded motion tokens — so handover is a *reference-stream source switch* on
the same tracker, not a controller swap:
  nav → arrive (goal radius + heading tolerance; waypoint yaws face the
  appliance, so the robot arrives pre-posed) → kplanner settles to idle-stand
  (well-defined common ground state) → blended switch to VLA reference stream →
  manipulate → hand back to idle → next goal.
Engineering item: the handover protocol — SONIC must never see a reference
discontinuity (cf. the boundary-velocity lesson), so define the stand posture as
the shared ground state with a short blend on both transitions (same discipline
as the PKL→CSV shim's 8-frame blends).

### Phase N5 (stretch) — GR00T N1.7 as the open-language NAV commander
Only after N3's router shows its limits. Two credible designs:
- **(a) VLA as goal-picker**: fine-tune N1.7 (via existing `launch_finetune_x2.py`
  machinery) on (ego RGB, instruction) → waypoint/goal-pose token episodes
  generated *by our own stack* (the executed-clip recorder + teacher routes give
  unlimited synthetic episodes). Cheap head, keeps System 1 intact. NVIDIA's own
  G1 workflow (teleop → N1.7 fine-tune → SONIC deploy, May 2026) is the template.
- **(b) VLA end-to-end nav intents**: N1.7 emits nav intents directly at ~1 Hz with
  System 1 reduced to a safety layer — higher risk, data-hungry, only worth it if
  (a) plateaus.

## 4. Recommendation

Run N0→N3 in order (N3 can start day 1 against the stage-0 teacher). The student
is **distilled-then-RL-polished** (not pure RL — we'd be throwing away a 99.96%
teacher; not pure BC — on-policy correction matters for reactive vision). The
language layer starts as a free router and graduates to a GR00T N1.7 fine-tune
only when open-ended language is actually required. This keeps every component on
the deploy-parity path we validated in stage 0 and reuses ~80% existing code.

## 6.5 Results so far (2026-07-22/23 — surrogate + gallery)

| eval (8 wp x 8 drift dirs) | teacher clean | teacher @drift | student @drift |
|---|---|---|---|
| all-waypoint mean | 1.00 | 0.80 | **1.00** |
| entrance | 1.00 | 0.25 | **1.00** |
| hallway | 1.00 | 0.12 | **1.00** |

- Drift measured from 12 planner/SONIC executions: median 0.184 m/m, p90
  0.29, max 1.56 m, concentrated in TURNS (straights ~free) — model is
  turn-gated everywhere; training DR widened for real-robot margin.
- The compensation is a learned feature of the student weights (no runtime
  estimator): corrupted numbers + truthful pixels + true-state labels ->
  "trust the pixels". Ablation: same drift, no camera = 0.25; camera = 1.00.
- Long-route residual: dining_table->entrance 7/8 @p90 (v1); drift-hardened
  retrain v2 targets the last direction. Full ledger:
  gear_sonic/envs/nav_house/poc_validation.md (G6-G8).

## 7. Reference: our stack vs Flexion Reflect v1.0, layer by layer

Reflect v1.0 (the "single AI does a whole mission" demo, 2026-06) is the closest
shipped system to our target architecture. Their published pipeline:

```text
Mission prompt
   → AGENT            (mission → structured subtasks; TOOL-CALLS a Semantic Map)
   → MOTION GENERATOR (subtask → short-horizon trajectory)
   → WHOLE-BODY CTRL  (trajectory → motor commands)
        ▲ both fed by NEURAL PERCEPTION (shared 3D reconstruction stream)
```

| Reflect v1.0 layer | Ours (exists today) | Ours (planned) | Notes |
|---|---|---|---|
| Mission prompt → **Agent** (VLM, structured subtasks, live replanning; SFT+RL 38%→90% on 16-step missions) | — | Language layer: waypoint router (N3) → open-vocab map (N3.5) → GR00T/VLM agent (N5); RL on the coordinator = future lever | Their agent's visible plan list alternates "Navigate to X" / manipulation steps as **discrete subtasks** — our nav↔manip handover (N6) made explicit |
| **Semantic Map** (a TOOL the agent queries for grounding) | `configs/waypoints.json` (8 labeled poses) | VLMaps-style open-vocab map over the kitchen splat (N3.5) | Identical pattern at different scale; validates map-as-tool over end-to-end language nav |
| **Motion Generator** (subtask → short-horizon trajectory; VLA + RL skills) | kplanner (nav intents → kinematic motion); GR00T N1.7 motion-token path (manipulation, already integrated) | RGB nav student feeding kplanner (N2); GR00T fine-tune for kitchen manipulation (post-N6) | Same split: per-skill backends behind one trajectory interface |
| **Whole-Body Controller** ("Reflex", RL, closes the loop in real time) | **SONIC** (softland_4800 + successors) | unchanged — frozen below the wire | Direct equivalent; both RL, both robot-agnostic by design |
| **Neural Perception** (shared 3D reconstruction feeding motion gen + WBC) | — (split: camera→policy obs; proprio→SONIC) | not needed for nav; add for contact-rich manipulation later | Our only structural gap vs their stack |
| **FlexComm** (runtime: comms, isolation, latency, safety) | ZMQ/DDS deploy stack (planner_cmd:5563, robot_pose, MC safety rules) | unchanged | Equivalent role |
| Training worlds | Scaniverse splat kitchen (NuRec, IsaacLab) | same, + camera-in-loop (N0–N2) | They train in splat twins of real sites — same recipe (their Niantic collab) |

Takeaways for later reference:
1. The seamless "one model" look is produced by continuous VLM replanning +
   learned local recovery — the stack underneath is modular, like ours.
2. Their two RL insertion points beyond skills: recovery behaviors inside the
   motion layer, and RL on the VLM coordinator (the 38→90 jump). Both are
   candidate upgrades for us after N4.
3. Nothing in their stack contradicts a single choice in this plan; the deltas
   are scale (building vs kitchen) and the shared perception stream.

## Sources
- Flexion × Niantic × NVIDIA: https://flexion.ai/news/niantic-spatial-flexion-and-nvidia-closing-the-sim2real-gap-for-humanoids ; https://www.nianticspatial.com/blog/usdz-scaniverse ; https://www.nianticspatial.com/robotics
- NaVILA (RSS'25): https://arxiv.org/abs/2412.04453 ; https://navila-bot.github.io/
- GR00T N1.5/N1.6/N1.7: https://huggingface.co/nvidia/GR00T-N1.5-3B ; https://github.com/Nvidia/Isaac-GR00T ; https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-gr00t-n1-6-using-a-sim-to-real-workflow/ ; https://research.nvidia.com/labs/gear/gr00t-n1_5/
- Encoder caveat (generalist ViTs underperform for nav): https://arxiv.org/html/2606.21216
- GR00T-WholeBodyControl platform: https://github.com/NVlabs/GR00T-WholeBodyControl
- Distillation practice: Learning by Cheating https://arxiv.org/abs/1912.12294 ;
  Humanoid Parkour (DAgger vs from-scratch vision RL) https://arxiv.org/html/2406.10759v1 ;
  Parkour in the Wild (distill gap −10.4%, RL fine-tune +3.1%) https://arxiv.org/html/2505.11164v1 ;
  Distillation-PPO https://arxiv.org/abs/2503.08299 ;
  To Distill or Decide? (theory) https://arxiv.org/abs/2510.03207
- Encoders: Hansen et al. from-scratch baseline https://arxiv.org/abs/2212.05749 ;
  Theia multi-teacher distilled encoder https://arxiv.org/abs/2407.20179 ;
  VC-1/CortexBench https://arxiv.org/abs/2303.18240
- Hybrid validation: VAMOS https://vamos-vla.github.io/ ; LeVERB https://arxiv.org/abs/2506.13751 ;
  Figure Helix/Helix-02 https://www.figure.ai/news/helix-02 ;
  Gemini Robotics-ER https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/ ;
  Helpful DoggyBot https://arxiv.org/abs/2410.00231
- Language-nav SOTA: DualVLN https://arxiv.org/abs/2512.08186 ;
  StreamVLN https://arxiv.org/abs/2507.05240 ; CorrectNav https://arxiv.org/abs/2508.10416 ;
  VLMaps https://vlmaps.github.io/ ; OK-Robot https://arxiv.org/abs/2401.12202 ;
  SayNav https://arxiv.org/abs/2309.04077 ; LeLaN https://arxiv.org/abs/2410.03603
- GS-world DR + camera-in-loop: VR-Robo https://arxiv.org/html/2502.01536v2 ;
  DextrAH-RGB (NVIDIA; still chose DAgger over direct RL) https://arxiv.org/html/2412.01791v1 ;
  LucidSim generative augmentation https://arxiv.org/abs/2411.00083 ;
  Isaac Lab tiled rendering https://isaac-sim.github.io/IsaacLab/v1.2.0/source/features/tiled_rendering.html ;
  NVIDIA NuRec https://developer.nvidia.com/omniverse/nurec
