# Deep-research notes: cameras + language in the navigation pipeline
2026-07-22/23. The full evidence base behind `x2-kitchen-nav-stage1-camera-plan.md`,
written up for reading. Sources inline; items marked (med) came from search
snippets rather than fetched primary pages.

---

## 0. The question we asked

Stage-0 gave us a 99.96% state-based RL nav teacher driving kplanner+SONIC.
Next: cameras in the pipeline. Should the visual student be (a) pure RL/BC,
(b) a GR00T-style VLA fine-tuned to take "go to the cooking range", or
(c) a hybrid? Five research threads + a full repo audit answered it.

---

## 1. Flexion × Niantic × NVIDIA — the recipe closest to ours

Primary: https://flexion.ai/news/niantic-spatial-flexion-and-nvidia-closing-the-sim2real-gap-for-humanoids (2026-07-21!)
and https://www.nianticspatial.com/blog/usdz-scaniverse

- RGB-only **local** navigation policy (goal pose in robot frame, no global
  map, no language), trained **entirely in simulation** inside Gaussian-splat
  digital twins of two real offices; **zero-shot** transfer to the real robot;
  robust to rearranged furniture; on par with a depth policy (ZED X neural).
- Method: **direct RL** (no teacher-student mentioned), domain randomization,
  **frozen "large-scale offline-trained image encoders"** (architecture
  undisclosed), "millions of rollouts", massively parallel on a single GPU in
  Isaac Lab. Sim results: 97.8% (Flexion office), 75.0% (Niantic office),
  1,024 rollouts per condition. No real-world quantitative numbers published.
- Scaniverse pipeline: ~5-min 360° capture → splat + auto-aligned mesh
  (MVSAnywhere) in one USDZ matching NVIDIA's NuRec volume spec — the exact
  format our kitchen world already uses.
- Their **Reflect v1.0** full-mission stack (the "one AI does everything"
  demo video) is explicitly modular: VLM mission agent (tool-calls a
  **semantic map**) → motion generator (VLA + RL skills per subtask) →
  Reflex whole-body controller → FlexComm runtime. The agent's plan list
  alternates "Navigate to X" and manipulation steps as discrete subtasks.
  RL-fine-tuning the VLM coordinator lifted 16-step mission success
  **38% → 90%** (SFT vs SFT+RL). https://flexion.ai/news/flexion-reflect-v1.0

Takeaway: the industry's flagship result in our exact world type is a small
RGB point-goal policy under a language layer — not an end-to-end VLA.

## 2. GR00T N1 → N1.7 — what the VLA actually is and isn't

- Architecture: dual-system VLA — System 2 VLM (Eagle-2 → Cosmos-Reason2/
  Qwen3-VL by N1.7) + System 1 flow-matching DiT action head, jointly
  trained. N1: 10 Hz VLM / 120 Hz actions internally. N1.7: 3B params,
  action horizon 40, state/action dims 132, pretrained on 20,854 h of
  egocentric human video. https://arxiv.org/abs/2503.14734 ,
  https://huggingface.co/blog/nvidia/gr00t-n1-7
- **Navigation: zero precedent.** The N1 paper is tabletop-manipulation-only
  by its own scope statement. Every documented legged deployment (incl.
  NVIDIA's official G1 workflow: teleop data → N1.7 fine-tune → SONIC WBC
  deploy) routes the VLA **through a whole-body controller** — the
  `velocity_commands [vx,vy,yaw]` interface belongs to the WBC/kinematic
  planner side, i.e. literally our kplanner intent wire.
  https://github.com/NVlabs/GR00T-WholeBodyControl
- Fine-tuning: EmbodimentTag system (`NEW_EMBODIMENT` + modality config);
  LoRA (r64) fits in ~31 GB on an RTX 5090 at batch 8; demo counts from
  ~30 demos (RoboCasa low-data) to 15 min–3 h teleop per task.
  Inference with TensorRT: ~36 Hz (H100), community 50–80 Hz (RTX 5090) —
  so **latency is NOT the disqualifier**; the absence of any nav precedent
  and the data economics are. N1.7 is the first commercially-licensed GR00T.
- No "N2" exists; GR00T-Dreams is a synthetic-data blueprint, not a model.

Takeaway: GR00T is a manipulation commander that already speaks SONIC —
its proven role in our architecture is **manipulation at the destination**
(the user's handover design), not driving the base.

## 3. Language-conditioned navigation SOTA (2024–2026)

- **NaVILA** (RSS'25): VLA emits mid-level spatial language ("move forward
  75 cm") at ~1 Hz; an RL vision locomotion policy executes. Unitree Go2/G1,
  Booster T1; 88% real-world SR on 25 instructions; R2R-CE val-unseen 54.0.
  Its own ablation: vision-based low-level beats blind by 14–21 SR.
  https://arxiv.org/abs/2412.04453
- **DualVLN** ("Ground Slow, Move Fast", Dec 2025): 2 Hz Qwen2.5-VL-7B
  grounds a **pixel waypoint**; 30 Hz diffusion local policy executes.
  **Beats end-to-end VLAs head-to-head**: R2R-CE 64.3 vs StreamVLN 56.9 /
  NaVILA 54.0; zero-shot on a real G1. Ablation: removing the explicit
  modular interface degrades both sides — the interface itself is
  load-bearing. https://arxiv.org/abs/2512.08186
- **VAMOS** (2026): hierarchical VLM planner (pixel paths) + sim-trained
  affordance model: **90% vs end-to-end NoMaD 27% / NaVILA 10%** on real
  legged+wheeled robots; the planner trains on non-robot-specific data.
  https://vamos-vla.github.io/
- End-to-end monocular track progression for reference: NaVid 37.0 →
  NaVILA 54.0 → StreamVLN 56.4 (ICRA'26) → CorrectNav 65.1 (self-correction
  post-training) — still at/below the hierarchical systems.
  https://arxiv.org/abs/2508.10416 , https://arxiv.org/abs/2507.05240
- **For known, mapped, room-scale spaces** (our kitchen): VLN models are
  unnecessary. VLMaps (open-vocab semantic map queried by language,
  ICRA'23/IJRR'25) https://vlmaps.github.io/ ; OK-Robot (1-min scan +
  CLIP map + point-nav: nav contributed only 4–15% of failures — grounding
  dominates) https://arxiv.org/abs/2401.12202 ; SayNav (LLM plans over scene
  graph; its low-level point-nav alone: **98.5% SR in-room**)
  https://arxiv.org/abs/2309.04077 ; LeLaN (language object-nav from
  YouTube-labeled video, short-range) https://arxiv.org/abs/2410.03603
- ViNT/NoMaD: image-goal foundation policies (no language natively);
  LeLaN/OmniVLA extend the lineage to language/omni-modal goals.
  https://arxiv.org/abs/2306.14846 , https://arxiv.org/abs/2310.07896

Takeaway: as of mid-2026 **no pure end-to-end language-nav VLA holds the
lead anywhere**; the best numbers are hierarchical (slow grounder → fast
local policy), and for an 8-waypoint mapped kitchen a registry/map lookup
is the demonstrated-sufficient System 2.

## 4. Teacher-student distillation — best practice + hard numbers

- Canon: DAgger (2011) https://arxiv.org/abs/1011.0686 ; Learning by
  Cheating https://arxiv.org/abs/1912.12294 ; Lee/Miki ANYmal privileged
  learning; RMA. Modern vision students (Extreme Parkour
  https://arxiv.org/abs/2309.14341 , egocentric-vision quadrupeds
  https://arxiv.org/abs/2211.07638 ) are all on-policy supervised
  distillation — not offline BC, not RL-from-scratch.
- **Humanoid Parkour Learning**: vision RL from scratch 0–10% success vs
  **80–100%** for the DAgger student; on-policy data volume matters
  (single-GPU student collapses to 25–65%). https://arxiv.org/html/2406.10759v1
- **Parkour in the Wild** (ANYmal, 4 depth cams): distill-only student
  −10.4% avg vs experts (worst 73.0 vs 98.8); generalizes badly (14.9% on
  unseen meshes); **RL fine-tune after distillation → +3.1% ABOVE experts**.
  https://arxiv.org/html/2505.11164v1 . Distillation-PPO (teacher-regularized
  RL) same conclusion. https://arxiv.org/abs/2503.08299
- Theory: "To Distill or Decide?" (NeurIPS'25) — the best teacher to distill
  is not always the optimal one; less observation-dependent teachers distill
  better. https://arxiv.org/abs/2510.03207
- **DextrAH-RGB** (NVIDIA's flagship photoreal RGB manipulation): even with
  tiled ray-traced rendering at scale, they chose **online DAgger** over
  direct pixel RL. https://arxiv.org/html/2412.01791v1
- Rule of thumb: well-tuned DAgger student lands 5–25% below the privileged
  teacher; the RL fine-tune stage closes it (and can exceed).

Takeaway: distill-then-RL-fine-tune is the 2024–26 standard. Our teacher is
measured (99.96%), so we skip Flexion's millions-of-rollouts RL cost.

## 5. Encoders for the student

- Frozen generalist ViTs (DINOv2/v3) **underperform for navigation**
  (https://arxiv.org/html/2606.21216); heterogeneous-teacher-distilled
  encoders do better (Theia, https://arxiv.org/abs/2407.20179).
- From-scratch shallow CNN + strong augmentation is competitive with frozen
  PVRs (Hansen et al., https://arxiv.org/abs/2212.05749); no PVR dominates
  across tasks (VC-1/CortexBench, https://arxiv.org/abs/2303.18240).
- Working datapoints: VR-Robo used a frozen pretrained ViT + PPO in
  Gaussian-splat worlds → 93–100% real success
  (https://arxiv.org/html/2502.01536v2); VIRAL's default is pretrained-
  trainable ResNet18; Flexion used frozen robust encoders.
- Our ladder: ResNet18-trainable / shallow CNN → Theia-style frozen →
  DINO-class last.

## 6. Visual domain randomization for splat worlds

- **VR-Robo recipe** (GS worlds, the closest published): camera-extrinsics
  noise, brightness/contrast/saturation/hue jitter, Gaussian blur, additive
  noise, 0–1-frame image delay. https://arxiv.org/html/2502.01536v2
- DextrAH-RGB (photoreal ray-tracing): HDRI background swaps, material
  randomization, color jitter 100%, motion blur, light intensity ranges.
- LucidSim: generative image augmentation beats classical DR for RGB parkour.
  https://arxiv.org/abs/2411.00083
- NuRec notes: splats are USD volumes rendered natively by RTX (Isaac Sim
  5.0+; 6.0 moves to Fabric Scene Delegate w/ multi-GPU); .usdz can't be a
  root stage if you need to add references; DLSS Frame-Gen causes artifacts.
  https://developer.nvidia.com/omniverse/nurec ,
  https://docs.isaacsim.omniverse.nvidia.com/latest/assets/usd_assets_nurec.html

## 7. Hybrid vs end-to-end across shipped systems

- **Figure Helix / Helix-02**: 7–9 Hz VLM + 200 Hz visuomotor policy
  (latent interface); Helix-02 adds a 1 kHz whole-body layer. Rate-separated
  layers, imitation-learned. https://www.figure.ai/news/helix-02
- **Gemini Robotics-ER**: VLM emits points/trajectories "for your existing
  low-level controllers" — the productized explicit interface.
  https://deepmind.google/blog/gemini-robotics-brings-ai-into-the-physical-world/
- **GR00T N1** itself: internal 10/120 Hz split. **LeVERB**: latent "verbs"
  into an RL WBC, 7.8× over a naive hierarchical whole-body VLA.
  https://arxiv.org/abs/2506.13751
- Universal pattern: slow-semantic / fast-motor hierarchy in EVERY shipped
  2024–26 system; the only debate is explicit vs latent interface. For
  navigation, published quantitative evidence (VAMOS, DualVLN) favors
  explicit. No source found where end-to-end beats hybrid on real-robot nav.
- AgiBot (our robot's vendor): GO-1/ViLLA VLA is manipulation-first; **no
  navigation architecture published for X2 at all** — nothing to conflict
  with. https://agibot-world.com/blog/go1

## 8. What we already had in-tree (repo audit)

- **VIRAL / GR00T-VisualSim2Real**: complete teacher-PPO + DAgger-vision
  distillation stack (G1 loco-manip flavored): distill trainers, ResNet18 +
  DINOv3 encoders, RGB-delay buffers, ego TiledCamera wiring, student ONNX
  export. `external_dependencies/GR00T-VisualSim2Real/gr00t/rl/...`
- **GR00T N1.7 + X2 integration**: full fine-tune/inference plumbing
  (`launch_finetune_x2.py`, `vla_utils.py`) running the VLA as a motion-token
  commander through SONIC — the manipulation-handover path is pre-built.
- Kitchen splat world loader, camera_rgb obs group, OpenCV fisheye spawner,
  nav_kitchen_v1.yaml frozen spec. Missing connectors: nav_house task
  package, nav distill loop, planner batching (B=1 hardcode;
  `neural_planner.py:755` + root-model head at `motion_inference.py:230`).

## 9. Where it landed (and same-day empirical confirmation)

Architecture: **hybrid** — language router/semantic-map → RGB point-goal
student (DAgger from the stage-0 teacher + later RL fine-tune) → frozen
kplanner+SONIC; GR00T N1.7 reserved for manipulation at the destination
(handover = SONIC reference-stream switch through idle-stand).

Validated within 24h of the research (PoC gates + first training):
- Splat renders through the TiledCamera sensor path (after stale-buffer +
  xform-op fixes) — crisp robot-in-kitchen sensor frame.
- Teacher quality confirmed: 1.00 on all 8 waypoints (clean state).
- The user's field observation reproduced in sim: under bipedal-realistic
  odometry drift (15% of distance walked), teacher entrance arrivals drop
  to **0.25** (hallway 0.12).
- First DAgger student (ResNet18 + state fusion, baked-gallery eyes):
  entrance **1.00 under the same drift** by iteration 9k (~5 min at 20k
  samples/s). Vision closes the drift gap, as hypothesized.

Caveats that remain: surrogate-level result (planner/SONIC dynamics not in
loop), gallery-trained (same-world memorization acceptable for a fixed
kitchen, unproven beyond it), generic pinhole camera (real stereo-left
fisheye matching deferred to the task package).
