# X2 + GR00T + RoboCasa: Kitchen MuJoCo Pick-and-Place Stack

This page is the architectural overview of how AgiBot **X2 Ultra** (with
**OmniHand** dexterous hands) was ported into NVIDIA's **gr00trobocasa**
fork of RoboCasa, and how the resulting MuJoCo kitchen-style scenes are
**driven by the SONIC tracking policy + operator IK + Quest 3 VR** to
record GR00T-VLA training data in **LeRobot v2.1** format.

It is a companion to two adjacent docs:

* [`mujoco_kitchen_humanoid_notes.md`](../../../mujoco_kitchen_humanoid_notes.md)
  — the prior-art landscape (Franka Kitchen, BiGym, HumanoidBench,
  RoboCasa, LW-BenchHub) and why we picked RoboCasa.
* [`X2_INTEGRATION_NOTES.md`](../../../decoupled_wbc/dexmg/gr00trobocasa/X2_INTEGRATION_NOTES.md)
  — the operator runbook (Quickstart, button cheat-sheet, post-flight
  verification, troubleshooting, every CLI flag).

Read this page first to understand the **why** and the **shape** of the
system; jump to the integration notes when you need to actually record.

---

## Why this exists

Our overarching goal is to fine-tune a **GR00T VLA (N1.7)** on
dexterous pick-and-place demonstrations recorded with our own X2 Ultra
+ OmniHand robot in **kitchen-style MuJoCo scenes**. NVIDIA ships two
relevant pieces but neither one solves the whole problem on its own:

1. **gr00trobocasa** is RoboCasa wired up for the GR-1 humanoid + VLA
   training. Its tabletop tasks (`PnPCubeToBowl_*`, `PnPBowlToTarget_*`)
   are exactly the difficulty band we want, but it assumes a robosuite-
   native robot, a robosuite controller, and a robosuite teleop loop.
2. The **X2 deploy stack** (C++ deploy + SONIC tracking policy +
   MuJoCo bridge) already records VR teleop episodes flawlessly on a
   flat floor. It owns physics, contact resolution, and rendering, and
   it is what GR00T will close the loop against on the real robot.

The integration in this page glues them together so that **the C++
deploy + SONIC stays in the loop unmodified**, RoboCasa is reduced to
a *task wrapper* (per-episode randomization, shaped per-tick reward,
subtask labels), and Quest 3 / IK / finger retargeting drive the robot
exactly the way they do on the real X2. This is the "**G1 plan**"
referenced throughout the codebase.

---

## High-level system diagram

```
                            ┌────────────────────────────────────────────┐
                            │          Quest 3 headset (WebXR)            │
                            │   • two 6-DoF wrist poses                   │
                            │   • 5×curl + 1×oppose per hand              │
                            │   • A / B / X / Y buttons                   │
                            └───────────────┬────────────────────────────┘
                                            │  WebSocket  wss://host:8443
                                            ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │  X2DatasetRecorder            (gear_sonic.utils.teleop, host venv)   │
   │                                                                      │
   │  ┌──────────────┐   ┌──────────────┐   ┌────────────────────────────┐│
   │  │ DLS arm IK   │   │ finger       │   │ RobocasaTaskMirror          ││
   │  │ (per-arm     │   │ retarget +   │   │  • lazy robosuite env (reset)││
   │  │  6-DoF)      │   │ filter +     │   │  • pure-MuJoCo oracle:      ││
   │  │              │   │ stretch      │   │      success / shaped reward ││
   │  └──────┬───────┘   └──────┬───────┘   │      / subtask ladder        ││
   │         │                  │           └─────┬──────────────▲────────┘│
   │         ▼                  ▼                 │              │         │
   │  ┌──────────────────────────────────────┐    │ scene_reset  │ scene_  │
   │  │ pose ZMQ PUB     :5556  (50 Hz)      │    │ PUB :5560    │ state   │
   │  │  - 31-DOF body refs + root quat      │    │ (B button,   │ SUB :5559│
   │  │  - per-finger curl/oppose targets    │    │  ad-hoc)     │ (50 Hz) │
   │  └──────────────┬───────────────────────┘    │              │         │
   │                 │                            │              │         │
   │  ┌──────────────▼───────────────────────┐    │              │         │
   │  │ MujocoFrameRenderer  (offscreen EGL) │    │              │         │
   │  │  - loads SAME scene XML as bridge    │    │              │         │
   │  │  - renders ego_view per frame        │    │              │         │
   │  └──────────────────────────────────────┘    │              │         │
   │                                              │              │         │
   │  ┌──────────────────────────────────────┐    │              │         │
   │  │ LeRobot v2.1 exporter                │    │              │         │
   │  │   parquet + mp4 + meta/info.json     │    │              │         │
   │  │   + task.success / .reward / .subtask│    │              │         │
   │  └──────────────────────────────────────┘    │              │         │
   └──────────┬───────────────────▲───────────────┼──────────────┼─────────┘
              │                   │               │              │
              │ pose SUB :5556    │ x2_debug      │              │
              │                   │ SUB :5557     │              │
              ▼                   │ (proprio @    │              │
   ┌──────────────────────────────┴───────────────┴──────────────┴────────┐
   │  x2_mujoco_ros_bridge.py    (lives inside docker_x2/x2sim container) │
   │                                                                      │
   │  ┌──────────────────────────────────────────────────────────────┐   │
   │  │  C++ deploy (x2_deploy_onnx_ref)                              │   │
   │  │   • SONIC tracking policy (ONNX, ~50 Hz inference)           │   │
   │  │   • Wrist Bypass for the 4 broken wrist DOFs                  │   │
   │  │   • OmniHand controller (24 finger DOFs from ZMQ setpoints)   │   │
   │  └────────────────────────┬──────────────────────────────────────┘   │
   │                           │ joint_target_pos                          │
   │                           ▼                                           │
   │  ┌──────────────────────────────────────────────────────────────┐   │
   │  │  MuJoCo physics @ 1 kHz (mujoco==3.5.0)                       │   │
   │  │   • Loads <env>.xml = X2 + OmniHand + RoboCasa table/objects  │   │
   │  │   • Hand contact pairs ENABLED (disable_hand_collisions=False)│   │
   │  │   • Passive viewer + offscreen rgbd_head_front camera         │   │
   │  └──────────────────────────────────────────────────────────────┘   │
   │                                                                      │
   │  ┌──────────────────────────────────────────────────────────────┐   │
   │  │  scene_state PUB / scene_reset SUB (JSON-on-ZMQ)              │   │
   │  │   • Publishes object qpos, mutable body pos,                   │   │
   │  │     grasp_contacts (left/right/any), fingertip_pos             │   │
   │  │   • Receives ResetObjects payload from recorder on B press     │   │
   │  └──────────────────────────────────────────────────────────────┘   │
   └──────────────────────────────────────────────────────────────────────┘
```

Two new ZMQ topics carry the **scene** state across the process boundary
(the `pose` and `x2_debug` topics already existed for SONIC):

| Topic          | Direction         | Cadence    | Default port | Wire format          |
|----------------|-------------------|------------|--------------|----------------------|
| `scene_state`  | bridge → recorder | ~50 Hz     | 5559         | JSON via `pack_json` |
| `scene_reset`  | recorder → bridge | ad-hoc (B) | 5560         | JSON via `pack_json` |

---

## Embodiment porting: X2 + OmniHand → gr00trobocasa

### Step 1 — compose the canonical 31-DOF X2

The deployed robot is **X2 Ultra (29 body DOFs) + OmniHand (12 finger
DOFs/side)** = 31 body DOFs (lower body, waist, arms, head) +
24 hand DOFs. The composition lives in
`gear_sonic/scripts/compose_x2_with_omnihand.py` and is shared by every
downstream consumer (deploy bridge, robocasa builder, viewer).

```
   x2_ultra.xml (MJCF)               omnihand_left.urdf + omnihand_right.urdf
   ─ 29 body joints                  ─ 12 active finger joints / side
   ─ wrist stubs (placeholder)       ─ 7 mm × 17 mm fingertip cylinders
            │                                       │
            └──────────────────┬────────────────────┘
                               ▼
              build_x2_with_omnihand_spec()      ← single source of truth
              ┌──────────────────────────────┐
              │  • mounts hands at wrist     │
              │  • adds hand <actuator>s     │
              │  • optional:                 │
              │     disable_hand_collisions  │← True = SONIC training,
              │                              │  False = robocasa scenes
              └──────────────┬───────────────┘
                             ▼
              composed MjSpec → MJCF on disk
```

The `disable_hand_collisions` knob is the one critical fork point:

* **`True` (default)** — every fingertip geom gets `contype="0"
  conaffinity="0"`, so MuJoCo never considers finger-vs-anything pairs.
  This is byte-identical to the SONIC training setup and avoids
  spurious finger-vs-body / finger-vs-floor / finger-vs-finger forces
  in the proprio stream.
* **`False` (robocasa scenes)** — keeps the URDF-derived collision
  primitives live so the bridge's broad-phase considers
  finger-vs-cube and finger-vs-bowl pairs. Without this the
  subtask-grasp signal stays at 0 forever even when the operator is
  obviously squeezing the cube. To stop the OmniHand collision
  primitives from pinning the fingers half-open against each other or
  against the palm, every hand collision geom is re-classed via
  `_filter_hand_self_collisions` to
  `(contype=2, conaffinity=1)`. The MuJoCo contact rule
  `(c1 & a2) || (c2 & a1)` then evaluates to:

  | Pair | Result |
  |------|--------|
  | Hand vs cube / bowl / table / floor / X2 body (default `1, 1`) | **contact ✓** |
  | Hand vs hand (`2, 1` × `2, 1`) | filtered (palm-vs-finger AND finger-vs-finger) |

  Touchable scene geoms keep their default `(1, 1)` -- no scene-side
  patching is required for new tasks. (See
  `X2_INTEGRATION_NOTES.md` → "Hand self-collision filter" for the
  full bitmask derivation, the X2 wrist-mesh discriminator that
  preserves the X2 collision model, and the regression test set.)

### Step 2 — wrap as a robosuite robot inside the gr00trobocasa fork

robosuite needs the robot to live inside its `RobotModel` /
`GripperModel` taxonomy. We added three classes inside the
gr00trobocasa fork:

| Class | File | Role |
|-------|------|------|
| `X2Ultra` (+ fixed-lower-body variants) | `decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/robots/manipulators/x2_ultra_robot.py` | `LeggedManipulatorModel` subclass; declares the X2's joint groups, default qpos, gripper mount frames |
| `OmniHandLeft`, `OmniHandRight` | `decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/grippers/omnihand_grippers.py` | `GripperModel` subclasses — robosuite's hook for "the thing on the wrist" |
| `X2UltraFixedLowerBodyKeyConverter` | `decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/robots/__init__.py` | Maps robosuite obs/action arrays into the GR00T schema |

Plus a custom controller bundle at
`decoupled_wbc/dexmg/gr00trobocasa/robocasa/controllers/config/default/composite/x2_ultra_default.json`
(`JOINT_POSITION` for the body groups + `GRIP` for the OmniHand
grippers). A symlink in the host robosuite install lets `robosuite.make`
find it without patching robosuite itself.

### Step 3 — define the kitchen tasks

`decoupled_wbc/dexmg/gr00trobocasa/robocasa/environments/locomanipulation/x2_tabletop_pnp.py`
declares:

* `PrimitiveCube`, `PrimitiveBowl`, `PrimitiveFixture` — minimal
  hand-coded objects (no kitchen-asset library lookup) so the scenes
  build offline and don't pull in a multi-GB texture set.
* `LMTabletopFixedBase` — base env that pins the X2's floating base
  via a fixed frame, so the SONIC policy doesn't need to learn to
  walk while we're focused on upper-body manipulation. Exposes a
  `_build_mujoco_objects()` hook so subclasses that need real-mesh
  free-joint props (e.g. the apple) can extend the merge list before
  `super()._load_model()` runs.
* `X2PickPlaceCube` — pick a red cube, drop it into a blue bowl.
* `X2PickPlaceBowl` — pick a blue bowl, place it on a green target zone.
* `X2PickPlaceApple` — real-mesh sibling of `X2PickPlaceCube`. Loads
  the upstream `apple_0` MJCFObject (textured visual mesh + 5-fragment
  convex-decomposition collision shell) instead of a primitive box.
  Same bowl, table, and per-episode placement ranges as the cube
  variant so episodes from both scenes can be co-trained on the same
  VLA backbone without renormalising rewards.

Two version-bump tweaks were also applied:

* `robocasa/__init__.py` — relaxed the MuJoCo `(3.2.6, 3.3.2)` and
  robosuite `1.5.x` version assertions to accept our pinned `3.5.0` /
  `1.5.2`.
* `robocasa/environments/locomanipulation/base.py` — set
  `ROBOT_POS_OFFSETS = [0, 0, 1.355]` for every X2 variant so the
  robot stands on the floor instead of clipping through it.

### Step 4 — bake out a static scene XML

For runtime we don't load the live robocasa env in the deploy. Instead,
`gear_sonic/scripts/build_x2_robocasa_scene_xml.py`:

1. Composes X2 + OmniHand (with `disable_hand_collisions=False`).
2. Spins up a transient robocasa env, scrapes the table + cube + bowl
   bodies (and assets) from its compiled XML.
3. Bakes the `rgbd_head_front` head camera, two close-up
   `obj_left` / `obj_right` workspace cameras (`mode="targetbody"`,
   each tracking the env's manipulable object), and one wide-angle
   world-fixed witness camera `front_cam` (`mode="fixed"`, 120°
   vertical FoV, sat at `(0.9144, 0, 1.10)` looking back along world
   `-x`). All three workspace cameras live in the
   `_WORKSPACE_CAMERAS` tuple in
   `build_x2_robocasa_scene_xml.py`; per-camera `mode` + optional
   `xyaxes` controls whether the camera tracks an object or stays
   pinned to the launch framing.
4. Absolutizes every `<mesh file="…">` path so the result is a
   single self-contained XML.
5. Writes `<env>.xml` plus a `<env>.json` sidecar listing freejoints,
   mutable welded bodies, the canonical task instruction, scene
   object collision geoms, hand root bodies, fingertip bodies, and
   initial poses.

The bundled outputs live at
`gear_sonic/data/assets/robocasa_scenes/X2PickPlaceCube.{xml,json}`,
`X2PickPlaceBowl.{xml,json}`, and `X2PickPlaceApple.{xml,json}`.
**Both the bridge and the recorder load the same XML**; that is what
keeps qpos addresses, body IDs, and freejoint layouts byte-identical
across processes.

```
   gr00trobocasa env class                 compose_x2_with_omnihand
        ─────┬─────                              ─────┬─────
             │                                        │
             ▼                                        ▼
        robosuite scene XML  ◀──── extract ────▶   X2+OmniHand spec
             │                                        │
             └────────────────  merge  ───────────────┘
                                  │
                                  ▼
              robocasa_scenes/<Env>.xml          ← bridge --sim-mjcf
                       +
              robocasa_scenes/<Env>.json         ← recorder mirror, both
                                                   ego renderer & bridge
                                                   metadata sidecar
```

---

## Control architecture: SONIC + IK + Wrist Bypass + fingers

The X2 deploy was designed around SONIC tracking, not VR teleop. To get
honest per-DOF authority during recording we layer four control
sources, in priority order, all inside the C++ deploy:

```
   ┌─────────────────────────────────────────────────────────────────┐
   │ Per-tick, in the C++ deploy (50 Hz):                            │
   │                                                                 │
   │   1.  SONIC tracking policy → target_pos_mj for ALL 31 DOFs     │
   │           (stand-still reference unless --vla streams refs)     │
   │                                                                 │
   │   2.  Wrist Bypass override (if --wrist-bypass=ik):             │
   │       overwrite target_pos_mj for the 4 broken wrist DOFs       │
   │       (left/right wrist_pitch + wrist_roll) with the IK ref     │
   │       from the recorder's ZMQ pose stream.  wrist_yaw stays     │
   │       under SONIC.                                              │
   │                                                                 │
   │   3.  Safety stack:                                             │
   │       per-joint clamp (--max-target-dev), raw action clip,      │
   │       soft-exit ramp on --max-duration.                         │
   │                                                                 │
   │   4.  PD torque to MuJoCo joints (handed off from bridge once   │
   │       the first deploy command arrives).                        │
   │                                                                 │
   │   5.  OmniHand controller (separate ZMQ ingest path):           │
   │       per-finger curl / oppose setpoints from the recorder      │
   │       drive the 24 hand DOFs directly via PD.                   │
   └─────────────────────────────────────────────────────────────────┘
```

### Wrist Bypass in one paragraph

SONIC pins `wrist_pitch` and `wrist_roll` at its trained comfort pose
for both the 2k and 25k checkpoints — the operator's wrist motion
literally never reaches the robot through SONIC alone. The bypass,
implemented in C++ (see
[`x2_zmq_cpp_port_plan.md`](x2_zmq_cpp_port_plan.md) for the wire-level
details), reads the IK reference straight off the `pose` ZMQ topic and
splices it into `target_pos_mj` for those 4 DOFs **before** the safety
stack. The Python port at `gear_sonic/utils/teleop/wrist_bypass.py`
mirrors the same logic for replay tooling. Operators can disable it
via `--wrist-bypass off` for sim-to-real fidelity probes, but VLA
recording defaults to `--wrist-bypass ik`.

### Finger retargeting + curl/oppose compensation

```
   Quest 3                             RecorderConfig knobs
   ─────────                           ─────────────────────
   per-finger curl ∈ [0, 1]            apply_curl_compensation: bool
   thumb-oppose ∈ [0, 1]               apply_oppose_compensation: bool
                                                │
                ┌───────────────────────────────┘
                ▼
   FingerSignalFilter   (low-pass + dead-zone, smooths Quest 3 jitter)
                │
                ▼
   OperatorCalibration  (per-operator min/max range mapping)
                │
                ▼
   per_finger_grasp_command_from_curls_and_oppose()
        │   │
        │   └── if apply_curl_compensation:
        │         power-curve stretch pushes mid-range curls toward
        │         full closure (operators rarely physically reach
        │         curl=1.0 on the headset; this is the same fix that
        │         landed in replay tooling earlier in the project)
        │
        └── if apply_oppose_compensation:
              same stretch on thumb oppose
                │
                ▼
   24 finger setpoints (12 left + 12 right, in radians)
                │
                ▼
   ZMQ PUB on `pose` topic ─► deploy OmniHand controller ─► PD torque
```

Both compensation flags are CLI-exposed (`--apply-curl-compensation`,
`--apply-oppose-compensation`) on the recorder
(`gear_sonic/scripts/record_x2_dataset.py`) and on the standalone
kinematic teleop tool
(`gear_sonic/scripts/teleop_x2_kinematic.py`), and surfaced in the
recorder's startup banner so the operator sees which mode is active.

---

## Quest 3 → LeRobot recording pipeline

### Operator-facing button map

The recorder reads buttons over WebSocket from the headset. Typing
into the host terminal does nothing — only the physical right-controller
buttons fire the lifecycle.

| Button | Action |
|--------|--------|
| **A** | Toggle arm IK engagement (idle ↔ active wrist tracking). |
| **B** | Start a new episode. Calls `mirror.reset()` to randomize cube/bowl, PUBs `scene_reset` to the bridge, opens a fresh LeRobot episode buffer. |
| **X** | Stop and **save** the current episode. |
| **Y** | Stop and **discard** the current episode (no disk write). |

### Per-episode timeline

```
Time ─────────────────────────────────────────────────────────────►

deploy boot   wrapper bring-up ── docker compose up ── ros2 launch ── bridge start
              ─────── ~30 s on cold image (one-time pyzmq self-install) ───────

ready state   bridge SCENE PUB online ─ recorder SCENE SUB online ─ Quest 3 connected
                                       ▲
                                       └── wrapper waits for first
                                           "CONTROL tick=" line in deploy
                                           log; aborts on "failed to bind"

idle          [waiting for B] ─ scene_state @ 50 Hz, no recording

B pressed     mirror.reset(seed) ─ scene_reset PUB ─► bridge mj_data.qpos overwritten
                                                       cube/bowl jump to new poses
              episode buffer reset ─ task.* columns start filling

50 Hz loop    mirror.sync_from_state(state) ─► task.success / task.reward
                                                / task.subtask_<phase>
              joint_pos_mj PUB ─► SONIC actor ─► PD torques ─► MuJoCo physics
              x2_debug SUB ─► proprio ─► LeRobot frame
              ego_view render ─► observation.images.ego_view ─► LeRobot frame

X pressed     episode buffer flushed to LeRobot exporter (parquet + mp4 + meta)

Y pressed     episode buffer dropped (no disk write)

deploy dies   deploy-watchdog detects PID exit ─► loud red banner with last 20
              log lines ─► SIGINT to recorder process group ─► clean dataset flush

Ctrl-C        recorder stop ─ deploy SIGINT ─ docker stop ─ logs preserved at
              /tmp/deploy_x2_record_*.log
```

### Lifecycle hardening (May 2026)

The wrapper at `gear_sonic/scripts/record_x2_dataset.sh` was
historically fragile because it didn't track the deploy's actual
liveness. Three defensive layers were added:

1. **Pre-flight cleanup** — sweep any leftover `docker_x2-x2sim-run-*`
   container via `docker stop`, then probe ZMQ ports 5557 / 5559 / 5570
   with `ss` and abort with a clear error if anything is still bound.
   Without this, a previous run's container could silently hold
   `x2_debug` (5557) so the new bridge bound nothing on it; the
   recorder would then show `deploy_alive=False` for the entire
   session and the operator would teleop into the void.
2. **Real readiness gate** — wait for the deploy's first
   `CONTROL tick=` log line (proves the bridge has bound, the deploy
   is running, and the policy is publishing). The old gate matched on
   `Launching ...` — but `deploy_x2.sh` prints that string in its own
   banner *before* forking the ros2 process, so the wait fired on
   iteration 0 and handed off while docker was still spinning.
3. **Background deploy watchdog** — polls `kill -0` on the deploy PID
   once a second; the first time it disappears it prints a loud red
   banner with the last 20 lines of the deploy log AND `kill -INT`s
   the recorder's process group so the LeRobot dataset gets flushed
   cleanly instead of leaving the operator staring at a frozen viewer.

To make #3 work the recorder is launched under `setsid` so it has a
dedicated PGID the watchdog can target.

### What gets written

A LeRobot v2.1 dataset directory after a few episodes:

```
<output-dir>/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet     ← per-frame action, observation.state,
│       ├── episode_000001.parquet       task.success, task.reward,
│       └── …                            task.subtask_<phase>, …
├── videos/
│   └── chunk-000/
│       └── observation.images.ego_view/
│           ├── episode_000000.mp4
│           └── …
└── meta/
    ├── info.json     ← features, fps, robot_type=x2_ultra
    ├── tasks.jsonl   ← language instructions per episode
    └── episodes.jsonl
```

The full feature schema is locked down in
`gear_sonic/data/features_x2_vla.py` and unit-tested by
`tests/test_x2_robocasa_scene_mode.py`.

### Per-tick shaped reward

The cube oracle returns a phased reward in `[0, 1]` (NOT a sparse 0/1
indicator), so a partially-successful demo gets credit for the deepest
phase reached on each tick:

| Phase    | Trigger                                          | Subtask column                  | Reward weight |
|----------|--------------------------------------------------|---------------------------------|---------------|
| Approach | `min(fingertip-cube dist) ≤ 8 cm`                | `task.subtask_approach_cube`    | 0.10          |
| Touch    | any hand-vs-cube contact (either side)           | `task.subtask_touch_cube`       | 0.25          |
| Grasp    | right-hand-vs-cube contact (mirrors upstream env)| `task.subtask_grasp_cube`       | 0.45          |
| Lift     | grasp + cube z > table_top + 2.5 cm              | `task.subtask_cube_off_table`   | 0.65          |
| Carry    | cube xy inside bowl footprint AND z above rim    | `task.subtask_cube_above_bowl`  | 0.80          |
| Place    | cube settled in bowl, upright (== `task.success`)| `task.subtask_cube_in_bowl`     | 1.00          |

Phase weights live in `_PICK_PLACE_CUBE_PHASE_REWARDS`
(`gear_sonic/utils/teleop/robocasa_task_mirror.py`). Threshold
constants (8 cm approach distance, 2.5 cm lift height, etc.) live in
`_PickPlaceCubeConstants` in the same file. Tune one place; the test
`tests/test_x2_robocasa_scene_mode.py::test_mirror_phased_reward_climbs_through_stages`
keeps the ladder monotonic.

---

## Tooling reference

### Build / inspect scenes

| Script | Purpose |
|--------|---------|
| `gear_sonic/scripts/build_x2_robocasa_assets.py` | Generates robocasa-compatible MJCFs for X2 (floating-base + fixed-lower-body variants) and OmniHand. Run once after upstream URDF changes. |
| `gear_sonic/scripts/build_x2_robocasa_scene_xml.py` | Bake `<env>.xml` + `<env>.json` for one or all known scenes. Re-run after editing `x2_tabletop_pnp.py`. |
| `gear_sonic/scripts/view_x2_robocasa_scene.py` | Standalone interactive MuJoCo viewer for a built scene XML. Hardware-free. |
| `gear_sonic/scripts/smoke_x2_robocasa_scenes_render.py` | Hardware-free EGL smoke test that renders multiple `RobocasaTaskMirror.reset()` trials per scene as MP4 + collage PNG, with pose overlays. Auto-sets `MUJOCO_GL=egl` and adjusts the offscreen framebuffer to match the requested resolution. |

### Live recording / teleop

| Script | Purpose |
|--------|---------|
| `gear_sonic/scripts/record_x2_dataset.sh` | The operator entry point. Co-launches deploy + bridge + recorder + Quest 3 server. Handles `--robocasa-env` plumbing, pre-flight cleanup, readiness gate, and deploy watchdog (see above). |
| `gear_sonic/scripts/record_x2_dataset.py` | Python recorder CLI. Forwarded to by the wrapper; can be invoked directly when the deploy is already running elsewhere. |
| `gear_sonic/scripts/teleop_x2_kinematic.py` | Live kinematic teleop without dataset writes. Same finger / IK plumbing as the recorder; useful for VR debugging. |
| `gear_sonic/utils/teleop/x2_dataset_recorder.py` | The `X2DatasetRecorder` orchestrator. Owns the Quest 3 reader, IK, finger retarget, ZMQ sockets, ego-view renderer, LeRobot exporter, and the deploy-silent watchdog. |
| `gear_sonic/utils/teleop/robocasa_task_mirror.py` | `RobocasaTaskMirror` (lazy robosuite env for reset, pure-MuJoCo oracle for success/reward/subtasks). Defines `SceneState` / `ResetObjects`. |

### Bridge side (inside docker_x2 container)

| Script | Purpose |
|--------|---------|
| `gear_sonic_deploy/deploy_x2.sh` | The ros2 deploy wrapper. Forwards `--sim-mjcf`, `--sim-with-omnihand`, `--vla`, `--wrist-bypass` to the C++ binary. Throttles `[deploy] CONTROL tick=` chatter to one line per 5 s on the way to stdout. |
| `gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py` | The MuJoCo bridge. Loads `<env>.xml`, auto-discovers the `.json` sidecar, hosts the passive viewer, lazy-installs pyzmq if missing, runs the contact-walker (`_collect_grasp_contacts`), publishes `scene_state`, and applies `scene_reset` payloads through a thread-safe pending queue (so the main sim loop owns all `mj_data` mutations). |

### Tests

* `tests/test_x2_robocasa_scene_mode.py` — hardware-free pytest gate
  for the whole RoboCasa scene path: builder, mirror oracle, ZMQ
  serialization, phased reward monotonicity, scene XML invariants.
* `tests/test_x2_camera_plumbing.py` — verifies `rgbd_head_front` +
  `obj_left` / `obj_right` cameras land in the built scene XML.
* `tests/test_record_x2_dataset_schema.py::test_front_cam_baked_into_robocasa_scene_xmls`
  — locks down the world-fixed `front_cam` (120° FoV, fixed pose,
  3 ft / chest-height) across all three bundled robocasa scene XMLs;
  paired with `test_front_cam_include_adds_video_feature` /
  `test_front_cam_default_off_keeps_legacy_schema` /
  `test_front_cam_resolver_default_in_record_cli` for the recorder
  feature schema + CLI default.
* `tests/test_x2_lerobot_exporter.py` — schema / column-name lock-down
  for the LeRobot v2.1 writer.

---

## Status & known limits

* **SONIC tracking policy was trained without table contact.** The
  25k *_g1 checkpoint we ship doesn't know about the table or any
  scene object. The Wrist Bypass override compensates for the broken
  wrist DOFs but is not contact-aware. Sim-to-real fidelity for
  *collision events* is therefore not preserved by this dataset. Use
  these recordings to teach the GR00T VLA the **intent** of pick and
  place, not the precise impedance behaviour at table contact. Plan A
  is to retrain SONIC with the static scene fixtures present in the
  world model; Plan B is to swap in a contact-aware tracking policy.

* **Per-episode randomization fires only on B in record mode.** In
  `--teleop-only` mode `_start_episode` early-returns, so
  `mirror.reset()` is never called and the cube/bowl sit at their
  baked-in initial pose for the whole session. To get randomized
  objects without writing data, record into a throwaway `--output-dir`
  and discard episodes with Y.

* **Mirror oracle is a port, not a delegate.** `robocasa_task_mirror.py`
  hand-translates the success / reward / subtask helpers from the
  upstream env classes. If upstream env logic changes the mirror will
  drift; smoke tests lock down the *shape* of the oracle's output,
  not its agreement with the live env.

* **Lower body is pinned.** All tasks today use
  `LMTabletopFixedBase`, which fixes the X2's floating base. Walking
  + manipulation will land once the
  [X2 heuristic locomotion planner](x2_heuristic_planner.md) is wired
  through `record_x2_dataset.py` (planned but not implemented).

* **Three scenes shipped.** `X2PickPlaceCube`, `X2PickPlaceBowl`, and
  `X2PickPlaceApple`. The full RoboCasa365 kitchen library (2,500+
  scenes, 365 tasks) is not auto-imported; each new scene is a 3-file
  change documented in
  [`X2_INTEGRATION_NOTES.md`](../../../decoupled_wbc/dexmg/gr00trobocasa/X2_INTEGRATION_NOTES.md#adding-a-new-scene).
  Apple is the first real-mesh manipulable in the bundle (the other
  two use synthesised primitives) — use it as the template when
  importing more upstream `MJCFObject` assets.

---

## Where to go next

* **To actually record an episode** → jump to the Quickstart in
  [`X2_INTEGRATION_NOTES.md`](../../../decoupled_wbc/dexmg/gr00trobocasa/X2_INTEGRATION_NOTES.md#quickstart-record-one-episode).
* **To debug a recording session** → the troubleshooting tree in
  [`X2_INTEGRATION_NOTES.md`](../../../decoupled_wbc/dexmg/gr00trobocasa/X2_INTEGRATION_NOTES.md#troubleshooting)
  covers "viewer terminated as soon as I pressed A", "B does nothing",
  empty `task.subtask_grasp_cube`, and friends.
* **To add a new scene** → §"Adding a new scene" in the integration
  notes plus
  [`tests/test_x2_robocasa_scene_mode.py`](../../../tests/test_x2_robocasa_scene_mode.py)
  as a copy-paste template.
* **To understand the wrist bypass at the byte level** →
  [`x2_zmq_cpp_port_plan.md`](x2_zmq_cpp_port_plan.md) and
  [`x2_zmq_protocol.md`](x2_zmq_protocol.md).
* **To understand the locomotion planner that will eventually unfix
  the lower body** → [`x2_heuristic_planner.md`](x2_heuristic_planner.md).
