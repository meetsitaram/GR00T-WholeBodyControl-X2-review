# Asimov v1 SONIC bring-up plan

Branch: `asimov`. Goal (phase 1): **stable idle stand + stable forward / left /
right / backward walks (+ a few loops)** on Asimov v1 in IsaacLab SONIC,
trained overnight on the local 5090. Modeled on the G1→X2 port; retargeting
side already done (`agibot-x2-references/soma-retargeter`, branch `asimov`,
`docs/g1_to_asimov.md` there).

Canonical recipe: `docs/source/user_guide/new_embodiments.md` (H2 worked
example). X2 files to mirror: `gear_sonic/envs/manager_env/robots/x2_ultra.py`,
`gear_sonic/trl/utils/order_converter.py`, `robot_mapping` in
`modular_tracking_env_cfg.py` (~line 1026), exp yaml
`config/exp/manager/universal_token/all_modes/sonic_x2_ultra.yaml`.

## Asimov embodiment facts (from the retarget work — measured, not vendor tables)

- 23 actuated DOF, strict subset of G1's 29 (no waist pitch/roll, one wrist DOF
  per arm). MJCF truth: `asimov-references/mjlab/src/mjlab/asset_zoo/robots/asimov_1/`
  (`asimov_1_constant.py` = hardware-characterized kp/kd/effort/armature/friction).
- MJCF joint order: legs L(6), legs R(6), waist_yaw, **RIGHT arm(5), LEFT arm(5)**.
- Right side is sign-mirrored vs left in the MJCF conventions.
- Elbow MJCF `ref` = ±45°; qpos=0 = straight arm. Standing keyframe: pelvis z
  0.639, hip_pitch ±0.1, elbows ±0.87 (qpos), shoulders slightly adducted.
- No `torso_link` (torso = `waist_yaw_link`); neck bodies welded
  (`neck_yaw_link`, `neck_pitch_link` exist as bodies, no joints); terminal arm
  body = `*_wrist_yaw_link`; feet = 4 collision spheres r=0.005, sole 3.4 cm
  below ankle. Parallel RSU ankle is UNMODELED (serial approximation, ±5.7°
  ankle_roll — saturates on most dynamic content; known accepted risk).

## X2-port lessons this plan must not re-learn (from docs/experiments + conventions)

1. **Joint order is the #1 silent killer.** IsaacLab = BFS + alphabetical
   siblings; MuJoCo = DFS XML order. Get IsaacLab order empirically from
   runtime `robot.joint_names`, never from the URDF. PKL `dof`/`pose_aa` are
   MuJoCo order; maps must be generated, not hand-typed (see unification).
2. **fps discipline.** bones-seed CSVs are 120 fps; motion-lib target 50;
   never let a non-integer stride silently skip resampling; SONIC is a
   finite-bandwidth tracker — a too-fast reference is smeared, not sped up.
3. **Quats wxyz everywhere** except scipy in data_process (`w_last=False`).
4. **Obs conventions are load-bearing** at export/deploy time (deploy obs
   order `[base_ang_vel|joint_pos|joint_vel|actions|gravity_dir]`×10) — not a
   phase-1 concern but recorded here so the exp yaml isn't "simplified".
5. **Body-name compatibility**: reward/termination yamls name
   `torso_link`, `head_*_link`, `*_wrist_*_link`, `*_ankle_roll_link` —
   Asimov needs overrides (torso→`waist_yaw_link`, head→`neck_pitch_link`,
   wrist→`*_wrist_yaw_link`). Run `num_envs=1` first for clear name errors.
6. **KP/KD**: X2/H2 derive from armature (KP=armature·ω², ω=2π·10 Hz, ζ=2) —
   but Asimov has REAL hardware-characterized kp/kd in `asimov_1_constant.py`
   (hip 150/5, ankle 440/20, waist 65/5, shoulders 57–96/5, elbow/wrist 40/2).
   Use the real ones; effort_limit = the hard clamp (NOT saturation_effort).
   Action scale: mjlab uses 0.30·effort/kp (vs our 0.25·effort/kp) — start
   with the repo's 0.25 convention, note the delta.
7. **Default pose / spawn height**: spawn z from standing keyframe (0.639 +
   margin), default joint_pos = mjlab KNEES_STAND_KEYFRAME (do NOT re-learn
   the X2 straight-knee-idle-outside-operating-point lesson — keep default
   pose consistent with what the corpus actually does).
8. **Test the artifact the robot runs** (ONNX gate, later phases) and keep a
   visual gate: numeric parity can pass while motion is broken.

## Phase plan

### P0 — assets + embodiment + registration (with unification)
- [ ] URDF: **generate from the mjlab MJCF** (public asimovinc/asimov-1 has no
  URDF; its MJCF is deprecated). Traps (audit plan §7): bake elbow `ref` ±45°
  into joint origins (fails silently), re-add IMU/feet sites as fixed links,
  explicit `<inertial>` (never recompute from meshes). Script:
  `gear_sonic/scripts/dev/mjcf_to_urdf_asimov.py`.
- [ ] **Parity gate A (kinematics)**: FK the same qpos through MuJoCo
  (asimov_1.xml) and the URDF (IsaacLab/pinocchio) — body-position parity
  <1 mm on a pose sweep incl. elbows (the ref bake-in test).
- [ ] MJCF for motion_lib: `gear_sonic/data/assets/robot_description/mjcf/asimov.xml`
  (copy of the mjlab MJCF + assets).
- [ ] `robots/asimov.py`: joints list (empirical), CFG (real kp/kd/effort from
  asimov_1_constant.py), ACTION_SCALE, init state.
- [ ] **Unification** (the generalize-where-possible ask): add
  `robots/mapping_utils.py::build_isaaclab_mujoco_maps(il_names, mj_names)`
  that GENERATES the 4 index arrays from name lists (kills the hand-typed-map
  bug class); use it for asimov, leave g1/x2/h2 arrays in place but add an
  assert that regenerated maps match the existing constants (free regression
  test on all three). Same for `order_converter.py`: one
  `make_converter(joint_names, vr3_bodies, foot_bodies)` factory; register
  asimov through it.
- [ ] Register: `robots/__init__.py`, `robot_mapping["asimov"]`,
  `_CONVERTER_REGISTRY["asimov"]`.

### P1 — motion corpus (~2k locomotion)
- [ ] Select ≈2,000 locomotion clips from bones-seed G1 CSVs: idle/stand +
  walk fwd/back/lateral/turn + loops (filter the locowalk filename list by
  name patterns; mirror the spirit of the ~2,550-clip X2 warm-start corpus).
- [ ] Retarget G1→Asimov with the soma-retargeter pipeline (batch), run the
  eval gate, drop FAILs, record warn stats.
- [ ] Extend `data_process/convert_soma_csv_to_motion_lib.py` with
  `--robot asimov` (23 DOF, MuJoCo order, 120→lib fps handled explicitly) →
  `gear_sonic/data/motions/asimov_loco_2k.pkl` (+ mirrored `_M` if cheap).

### P2 — pkl-through-stack evaluation (BEFORE any training)
The user-specified gate: play the pkl directly through the SONIC stack and
compare source vs result, to catch joint order / frame buffer / obs
convention errors.
- [ ] **Parity gate B (kinematic replay)**: IsaacLab env with the Asimov
  robot kinematically driven from motion_lib (`num_envs=1 headless=False`) —
  visually correct + FK body positions vs MuJoCo FK of the same pkl frames
  (<5 mm). This exercises motion_lib load, order converter, body maps.
- [ ] **Parity gate C (zero-policy sanity)**: env steps with zero/default
  actions — robot stands at default pose, obs finite, no NaNs, joint order
  spot-check (command a single joint offset, verify the right joint moves).
- [ ] Later (post-training): im_eval + `feasibility_report.py` source-vs-
  executed comparison, same as X2's `_feasible` flow.

### P3 — experiment yaml + smoke train
- [ ] `sonic_asimov_loco.yaml`: copy `sonic_x2_ultra.yaml`; `robot.type:
  asimov`, `assetFileName: asimov.xml`, `motion_file: asimov_loco_2k.pkl`,
  body-name overrides (reward_point_body `[waist_yaw_link, left/right_
  wrist_yaw_link]`, anti_shake `[*_wrist_yaw_link, neck_pitch_link]`), keep
  the locomotion reward set (NOT the arm-dynamics moderated weights — X2's
  elbow-offset lesson).
- [ ] Smoke: `num_envs=1` (body names) → `num_envs=16 headless=False`
  (visual: standing, then early tracking) → fix fallout.

### P4 — overnight training on the 5090
- [ ] `num_envs` sized to 32 GB (start 2048, try 4096), `headless=True`,
  from scratch (no G1 warm start — different DOF), W&B logging, checkpoints
  every 500 iters, tmux + watcher. Overnight ≈ 8–20k iters (X2 walked at
  iter-22k on 8 GPU; local single-GPU expectation: standing + early gait by
  morning — judge with im_eval on a 5–10 clip walk subset + visual).
- [ ] Morning: im_eval metrics (progress, mpjpe) on idle/fwd/back/left/right
  clips; visual gate; decide continue vs iterate.

## Success criteria (phase 1)
- Parity gates A/B/C pass (mm-level FK parity, correct visual replay).
- Training stable (no explosion at spawn — else KP/KD/action-scale wrong).
- im_eval on the walk subset: success on idle + 4-direction walk clips,
  mpjpe in family with early X2 numbers; visually stable stand and walks.
