# X2 + gr00trobocasa integration notes (G1 architecture)

This document is the reference for recording X2 dexterous pick-and-place
datasets on top of robocasa's tabletop tasks. It serves a triple
purpose:

1. **Operator runbook** — how to record, what the buttons do, how to add
   a new scene, how to verify the dataset (§Quickstart, §CLI reference,
   §Post-flight verification, §Troubleshooting).
2. **Architecture reference** — what the moving pieces are, why they
   are split this way, and where in the source tree each lives
   (§Architecture overview, §Static scene XML, §Recorder lifecycle).
3. **Caveat sheet** — what the system does NOT do, and what would
   break if you tried to lean on it (§Caveats and known limitations).

If you only need the recording command, skip to **Quickstart**.

---

## TL;DR

> The C++ deploy + SONIC tracking policy stay in the loop unmodified.
> Robocasa is loaded inside the Python recorder process **only as a
> task wrapper** (per-episode object randomization, per-tick
> `task.success` / `task.reward` / `task.subtask_*` labels). Scene state
> crosses the process boundary over two JSON-on-ZMQ topics so neither
> side has to subprocess-spawn the other.

---

## Quickstart: record one episode

This is the canonical, hardware-required path. It needs:

* a SONIC checkpoint with an `exported/*.onnx` bundle;
* docker + the `docker_x2/x2sim` image (the wrapper bring-up handles
  this; on a stale image the bridge will self-install pyzmq once and
  then run);
* a Quest 3 headset on the same WiFi as the workstation, with the
  WebXR app reachable at `https://<workstation-ip>:8443`;
* the bundled scene XML for the env you want — already committed under
  `gear_sonic/data/assets/robocasa_scenes/`. (Re-build only if you've
  changed the env class; see §Adding new scenes.)

### One terminal, one command

```bash
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_robocasa_pnp_smoke_v0 \
    --robocasa-env X2PickPlaceCube \
    --wrist-bypass ik \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

The wrapper co-launches the C++ deploy, the MuJoCo bridge (with
`--sim-mjcf` set to the matching scene), the recorder, and the Quest 3
WebSocket / WebXR servers — all in one shell. Logs are interleaved
with `[deploy] …`, `[bridge] …`, `[recorder] …`, `[live-VLA] …` and
`[Quest3Reader] …` prefixes.

The `--robocasa-env` flag forces three things automatically:

| Implication | Why |
|-------------|-----|
| `deploy_x2.sh` gets `--sim-mjcf <scene>.xml` | both processes load the same MuJoCo |
| `--sim-with-omnihand` is forced ON | every robocasa scene is built on the OmniHand-augmented robot |
| `--task` defaults to the env's canonical instruction string from the `.json` sidecar | the LeRobot `task` column matches what the mirror's success oracle is grading |

You can still pass `--task "..."` explicitly to override the language
instruction (e.g. for paraphrase-augmentation experiments).

### Operator buttons (Quest 3 right controller)

The recorder reads buttons over WebSocket from the headset. Typing
into the host terminal does nothing — only the physical controller
buttons fire the lifecycle.

| Button | Action |
|--------|--------|
| **A** | Toggle arm IK engagement (idle ↔ active wrist tracking). |
| **B** | Start a new episode. Calls `mirror.reset()` to randomize cube/bowl, PUBs `scene_reset` to the bridge, opens a fresh LeRobot episode buffer. |
| **X** | Stop and **save** the current episode. |
| **Y** | Stop and **discard** the current episode (deletes the in-flight buffer; nothing is written to disk). |

The expected log sequence on a successful **B** press:

```
[recorder] [B] scene_reset sent: freejoints=['cube_joint'] welded=['bowl_body', 'table_body_main']
[recorder] [B] episode start (task='pick up the red cube and drop it into the blue bowl', episode=0)
```

If those lines do NOT appear when you press B, see §Troubleshooting →
*B does nothing*.

### What "ready" looks like in the log

Healthy boot sequence (in order):

```
[bridge] loaded scene metadata: …/X2PickPlaceCube.json
[bridge] scene plumbing: 1 freejoints, 2 welded bodies, 6 object collision geoms, L:42 R:42 hand geoms, L:5 R:5 fingertips
[bridge] scene_state PUB bound at tcp://*:5559 (1 freejoints, 2 bodies).
[bridge] scene_reset SUB connected at tcp://localhost:5560 (topic='scene_reset').
[recorder] robocasa scene mode: env='X2PickPlaceCube' xml=…/X2PickPlaceCube.xml
[recorder] task.subtask_* columns registered: ['task.subtask_approach_cube', 'task.subtask_touch_cube', 'task.subtask_grasp_cube', 'task.subtask_cube_off_table', 'task.subtask_cube_above_bowl', 'task.subtask_cube_in_bowl']
[recorder] scene_reset PUB bound at tcp://*:5560
[recorder] scene_state SUB connected at tcp://localhost:5559
[recorder] scene_state SUB: first message (t=…s freejoints=['cube_joint'])  ← scene state is flowing
[Quest3Reader] Serving WebXR app at https://<ip>:8443
[recorder] renderer ready (640x480, omnihand=True)
[recorder] waiting for first Quest 3 packet …
```

The bridge's `scene plumbing:` line is the most informative one in the
boot sequence: any zero in there ("0 object collision geoms" or
"L:0 R:0 hand geoms") means the contact-walker won't be able to fire,
so the recorder's `task.subtask_grasp_cube` will stay at 0 even if you
physically touch the cube. The right-side hand-geom count is roughly
40 (one fingertip cylinder + thumb collision boxes per finger × 5
fingers, plus a few wrist primitives).

After the operator opens the WebXR app on the headset and taps *Start
XR*, the wait line stops re-printing and B/X/Y become live.

---

## Architecture overview

### Topology

```
                                 ┌────────────────────────────────┐
                                 │ Quest 3 headset (WebXR)        │
                                 └──────────────┬─────────────────┘
                                                │ WebSocket (wss:8765)
                                                ▼
        ┌─────────────────────────────┐ ZMQ:5556 ┌────────────────────────────┐
        │ X2DatasetRecorder           ├────────► │ x2_mujoco_ros_bridge       │
        │  • DLS arm IK               │  pose    │  (lives in docker_x2)      │
        │  • finger retarget          │   PUB    │  • MuJoCo physics @ 1 kHz  │
        │  • RobocasaTaskMirror       │          │  • SONIC actor (ONNX)      │
        │     (robosuite for reset    │ ZMQ:5557 │  • OmniHand control        │
        │      + pure-MuJoCo oracle)  ◄─────────┤  • offscreen camera        │
        │  • MujocoFrameRenderer      │ x2_debug │    (rgbd_head_front)       │
        │      (loads SAME scene XML) │   PUB    │                            │
        │  • LeRobot v2.1 exporter    │          │                            │
        └─────────┬─────────▲─────────┘          └──────┬─────────────▲──────┘
                  │         │                           │             │
       ZMQ:5560   │         │ scene_state               │ scene_state │
       scene_reset│         │   SUB @ 50 Hz             │ PUB @ 50 Hz │
        PUB ad-hoc▼         │ (cube qpos, bowl pos,     │             │
                  │         │  grasp_contacts)          │             │
                  └─────────┴───────────────────────────┘             │
                                                                       │
                            scene_reset SUB (ad-hoc)                   │
                  ────────────────────────────────────────────────────►┘
                                                                  ZMQ:5560
```

Two new ZMQ topics carry the scene state across the process boundary:

| Topic          | Direction         | Cadence       | Default port | Wire format       |
|----------------|-------------------|---------------|--------------|-------------------|
| `scene_state`  | bridge → recorder | ~50 Hz        | 5559         | JSON via `pack_json` |
| `scene_reset`  | recorder → bridge | ad-hoc (B)    | 5560         | JSON via `pack_json` |

Schemas live in
`gear_sonic/utils/teleop/robocasa_task_mirror.py`
(`SceneState`, `ResetObjects`) and the ZMQ envelope helpers in
`gear_sonic/utils/teleop/zmq/scene_state_zmq.py`. JSON is intentional
— these messages are tiny (a handful of floats per object) and
infrequent relative to the proprio stream, so the simplicity of
`recv() -> print` debugging beats any wire-size win we'd get from
msgpack or protobuf.

### Why this split (the "G1 plan")

The C++ deploy already owns a MuJoCo replica. Adding a second
authoritative MuJoCo on the recorder side would force us to keep two
state machines in lock-step and would defeat the whole point of
recording on the same physics the SONIC tracking policy was trained
against.

So we let the bridge own physics, and use robosuite + the
gr00trobocasa fork only for what robosuite is uniquely good at:

* **Per-episode randomization** — robocasa's `placement_initializer`
  knows the task-specific safe placement region for the cube + bowl.
  Reimplementing that by hand in pure MuJoCo would be tedious and
  brittle.
* **Per-tick success / reward / subtask labels** — porting every
  `_check_success` / `_check_grasp` / `get_subtask_term_signals`
  method from `gr00trobocasa` to a hand-rolled oracle is a lot of
  surface area; mirroring is cheaper than re-deriving.

Everything else — physics, contact resolution, controller execution,
camera rendering — happens inside the deploy bridge's MuJoCo, exactly
as it does in flat-floor recording today.

### Episode timeline

```
Time ─────────────────────────────────────────────────────────────►

deploy boot   wrapper bring-up ── docker compose up ── ros2 launch ── bridge start
              ─────── ~30 s on cold image (one-time pyzmq self-install) ───────

ready state   bridge SCENE PUB online ─ recorder SCENE SUB online ─ Quest 3 connected

idle          [waiting for B] ─ scene_state @ 50 Hz, no recording

B pressed     mirror.reset(seed) ─ scene_reset PUB ─► bridge mj_data.qpos overwritten
                                                       cube/bowl jump to new poses
              episode buffer reset ─ task.* columns start filling

50 Hz loop    mirror.sync_from_state(state) ─► task.success / task.reward / task.subtask_grasp_cube
              joint_pos_mj PUB ─► SONIC actor ─► PD torques ─► MuJoCo physics
              x2_debug SUB ─► proprio ─► LeRobot frame
              ego_view render ─► observation.images.ego_view ─► LeRobot frame

X pressed     episode buffer flushed to LeRobot exporter (parquet + mp4 + meta/info.json)

Y pressed     episode buffer dropped (no disk write)

Ctrl-C        recorder stop ─ deploy SIGINT ─ docker stop ─ logs preserved at /tmp/deploy_x2_record_*.log
```

---

## Static scene XML (the linchpin)

Both processes load the **same** static `.xml` file. The recorder uses
it for the `RobocasaTaskMirror`'s MuJoCo state mirror AND for the
`MujocoFrameRenderer`'s ego-view camera. The bridge uses it as its
authoritative MuJoCo via `--sim-mjcf`. Because both sides parse the
same XML, joint qpos addresses, body IDs, and freejoint layouts line
up byte-for-byte; the per-tick state copy is just a
`mj_data.qpos[adr:adr+7] = …` slice on each side, no name-translation
table needed.

Bundled scenes live in `gear_sonic/data/assets/robocasa_scenes/`:

| File | Env | Task instruction |
|------|-----|------------------|
| `X2PickPlaceCube.xml` + `.json` | `X2PickPlaceCube` | pick up the red cube and drop it into the blue bowl |
| `X2PickPlaceBowl.xml` + `.json` | `X2PickPlaceBowl` | pick up the blue bowl and place it on the green target zone |

The `.json` sidecar is the source of truth for which freejoints /
welded bodies are scene-mutable; the bridge auto-discovers it from the
XML stem.

### Building / rebuilding the scene XML

You only need to do this when you've changed the env class
(`x2_tabletop_pnp.py`) or the X2 / OmniHand composition; the bundled
files are committed.

```bash
# Rebuild a single scene
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml \
    --env X2PickPlaceCube

# Rebuild every known scene
.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --all
```

Under the hood the builder:

1. Composes X2 + OmniHand via `compose_x2_with_omnihand` to get the
   canonical 31-DOF robot the deploy expects.
2. Spins up a transient robocasa env and scrapes the table + cube +
   bowl bodies (and their assets) from its compiled XML.
3. Bakes the `rgbd_head_front` head camera into the merged spec so the
   recorder's offscreen ego-view renders the same field of view the
   deploy bridge would render.
4. Absolutizes every `<mesh file="…">` path so the resulting XML is
   self-contained (no `<compiler meshdir>` needed).
5. Writes `<env>.xml` plus a `<env>.json` sidecar listing freejoints,
   mutable welded bodies, the canonical task instruction, and initial
   poses.

`.venv_sim` is required for the build (it has the gr00trobocasa fork +
robosuite + `compose_x2_with_omnihand`'s MuJoCo 3.5.0 features).
`.venv` lacks the fork — see §Caveats.

### Adding a new scene

Three files to touch, in order:

1. **Define the env class.** Add a new `X2PickPlace<Name>` subclass to
   `decoupled_wbc/dexmg/gr00trobocasa/robocasa/environments/locomanipulation/x2_tabletop_pnp.py`
   (you can copy the cube class as a template). Make sure it ends up
   exported from the package's `__init__.py`.
2. **Register it in the builder.** Add an entry to `_KNOWN_ENVS` in
   `gear_sonic/scripts/build_x2_robocasa_scene_xml.py` with the env
   name, the canonical task string, and any extra metadata fields.
3. **Register the oracle.** Add a `_TaskOracle` to `_ORACLES` in
   `gear_sonic/utils/teleop/robocasa_task_mirror.py` with the
   pure-MuJoCo `success_fn` / `reward_fn` / `subtasks_fn` for that
   task. Mirror the upstream env's `_check_success` /
   `get_subtask_term_signals` exactly.

Then build the XML once (`build_x2_robocasa_scene_xml --env <Name>`),
extend the `--robocasa-env` choices in
`gear_sonic/scripts/record_x2_dataset.py` and
`gear_sonic/scripts/record_x2_dataset.sh` (search for the existing
choices list — it's small), and add a smoke test in
`tests/test_x2_robocasa_scene_mode.py`.

---

## Recorder lifecycle in scene mode

When `record_x2_dataset.sh --robocasa-env X2PickPlaceCube` is invoked:

1. **Wrapper resolves paths.** `record_x2_dataset.sh` resolves the
   scene XML in `gear_sonic/data/assets/robocasa_scenes/`, forwards
   `--sim-mjcf <path>` to `deploy_x2.sh`, sets `--sim-with-omnihand`,
   and passes `--robocasa-env` through to the recorder.
2. **Bridge boot.** The bridge loads the scene XML, auto-discovers the
   `.json` sidecar, resolves freejoint qpos / dof addresses, lazy-
   installs pyzmq if missing, then arms the `scene_state` PUB and
   `scene_reset` SUB.
3. **Recorder boot.** The recorder constructs a `RobocasaTaskMirror`
   from the same XML + sidecar, opens its own `scene_state` SUB on a
   background thread, and binds a `scene_reset` PUB. The mirror's
   `task_string` becomes the LeRobot `task` field for every episode in
   this session.
4. **Quest 3 handshake.** The recorder waits for the first WebXR
   packet. Until then B/X/Y are inert (they're not connected to the
   keyboard; they only fire on Quest 3 controller buttons).
5. **Episode start (B).** The recorder calls `mirror.reset(seed)`
   which lazily compiles a robosuite env on the first call (~3-5 s,
   then reused), runs `env.reset()` to roll new object poses, copies
   them into the mirror's MuJoCo, and PUBs a `ResetObjects` payload.
   The bridge slams the new poses into its own `mj_data.qpos`, zeroes
   the freejoint velocities, and calls `mj_forward`.
6. **Per-tick (50 Hz).** The recorder pulls the latest `SceneState`
   from its background thread (which now carries `grasp_contacts`
   bucketed by side as well as `fingertip_pos` for the approach phase),
   runs `mirror.sync_from_state(state)`, then queries
   `mirror.check_success()`, `mirror.compute_reward()`, and
   `mirror.subtask_signals()`. These become `task.success`,
   `task.reward`, and `task.subtask_<name>` columns appended to the
   LeRobot frame. The full subtask column set for `X2PickPlaceCube` is
   `task.subtask_{approach,touch,grasp,off_table,above_bowl,in_bowl}_cube`
   (six 0/1 columns); the names are pre-registered in
   `RecorderConfig.__init__` from `mirror.static_subtask_names` so the
   LeRobot exporter's `validate_frame` accepts them.
7. **Episode end (X / Y).** X flushes the in-memory episode buffer to
   the LeRobot exporter (parquet + mp4 + chunk meta). Y drops it.

### Shaped reward and the subtask ladder

The cube oracle returns a phased reward in `[0, 1]` (NOT a sparse 0/1
indicator), so a partially-successful demo gets credit for the deepest
phase reached on each tick. Phases are independent indicators; the
reward at each tick is the **maximum weight** of all currently-active
phases:

| Phase | Trigger | Subtask column | Reward weight |
|-------|---------|----------------|---------------|
| Approach | min(fingertip-cube dist) ≤ 8 cm | `task.subtask_approach_cube` | 0.10 |
| Touch | any hand-vs-cube contact (either side) | `task.subtask_touch_cube` | 0.25 |
| Grasp | right-hand-vs-cube contact (mirrors upstream env) | `task.subtask_grasp_cube` | 0.45 |
| Lift | grasp + cube z > table_top + 2.5 cm | `task.subtask_cube_off_table` | 0.65 |
| Carry | cube xy inside bowl footprint AND cube z above rim | `task.subtask_cube_above_bowl` | 0.80 |
| Place | cube settled in bowl, upright (== `task.success`) | `task.subtask_cube_in_bowl` | 1.00 |

Phase weights live in `_PICK_PLACE_CUBE_PHASE_REWARDS`
(`gear_sonic/utils/teleop/robocasa_task_mirror.py`). The threshold
constants (8 cm approach distance, 2.5 cm lift height, etc.) live in
`_PickPlaceCubeConstants` in the same file. Tune one place; the
test
`tests/test_x2_robocasa_scene_mode.py::test_mirror_phased_reward_climbs_through_stages`
keeps the ladder monotonic.

### Bridge-side contact-walker

`_collect_grasp_contacts()` in `x2_mujoco_ros_bridge.py` walks
`mj_data.contact[:ncon]` once per scene_state publish (50 Hz) and
buckets each contact into `{logical_object: {left: bool, right: bool,
any: bool}}`. The `left`/`right` sets are populated at startup by
walking from each `*_wrist_roll_link` and collecting every descendant
geom ID; the per-object set is read from the metadata's
`object_contact_geoms`. A contact is attributed to side `S` only if
exactly one of its two geoms is in side `S`'s hand-geom set AND the
other is in the object's contact-geom set; cube-vs-table contacts go
to `any` so the oracle can distinguish "cube touched something" from
"cube grasped". Cost is O(ncon) with O(1) per-contact lookups.

`_collect_fingertip_pos()` returns
`{side: [[x, y, z], … five entries]}` from the bridge's
`mj_data.xpos[fingertip_bid]`. Five fingertips per side: thumb /
index / middle / ring / pinky distal phalanges (`*_dip` bodies). The
mirror computes `min(distance(tip, cube))` over all 10 fingertips for
the approach-phase indicator.

### Finger collisions on / off

`compose_x2_with_omnihand.build_x2_with_omnihand_spec` accepts
`disable_hand_collisions: bool = True`. The default keeps the SONIC
recording pipeline byte-identical (every fingertip geom gets
`contype=0 conaffinity=0` so MuJoCo never even considers
finger-vs-anything contact pairs, avoiding spurious finger-vs-body /
finger-vs-floor / finger-vs-finger forces in the proprio stream).

`build_x2_robocasa_scene_xml._build_compose_xml` passes
`disable_hand_collisions=False` so the bundled robocasa scene XMLs
ship with the URDF-derived collision primitives (7 mm × 17 mm
cylinders on each fingertip, plus thumb MCP/PIP collision boxes)
LIVE. With them enabled the bridge's broad-phase considers
finger-vs-cube and finger-vs-bowl contact pairs, which is exactly
what the contact-walker above needs. Cost in the bridge is
single-digit-ms per tick on the X2 + tabletop scene.

### What lives where

| File | Role |
|------|------|
| `gear_sonic/scripts/record_x2_dataset.sh` | Operator entry point. Co-launches deploy + recorder. Handles `--robocasa-env` → `--sim-mjcf` plumbing and forces `--sim-with-omnihand` on for scene mode. |
| `gear_sonic/scripts/record_x2_dataset.py` | Recorder CLI. Parses scene mode flags, instantiates `RecorderConfig`. |
| `gear_sonic/utils/teleop/x2_dataset_recorder.py` | `X2DatasetRecorder` orchestrator. Owns the Quest 3 reader, IK/finger retarget, ZMQ PUB/SUB sockets, and the LeRobot exporter. Adds `task.*` columns when scene mode is active. |
| `gear_sonic/utils/teleop/robocasa_task_mirror.py` | `RobocasaTaskMirror` (lazy robosuite env for reset, pure-MuJoCo oracle for success/reward/subtasks). Defines `SceneState` / `ResetObjects` dataclasses. |
| `gear_sonic/utils/teleop/zmq/scene_state_zmq.py` | Wire format helpers (`pack_json` / `unpack_json` / `serialize_*` / `parse_*`). |
| `gear_sonic/scripts/build_x2_robocasa_scene_xml.py` | Static scene XML builder (compose-driven). |
| `gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py` | MuJoCo bridge that lives inside `docker_x2`. Handles `--mjcf`, scene metadata auto-discovery, scene_state PUB, scene_reset SUB, and pyzmq self-install. |
| `decoupled_wbc/dexmg/gr00trobocasa/robocasa/environments/locomanipulation/x2_tabletop_pnp.py` | The X2-specific robocasa env classes (`X2PickPlaceCube`, `X2PickPlaceBowl`, primitives). |

---

## CLI reference

### `record_x2_dataset.sh` — scene-mode flags

| Flag | Default | Effect |
|------|---------|--------|
| `--robocasa-env {none, X2PickPlaceCube, X2PickPlaceBowl}` | `none` | Switch into scene mode. Resolves scene XML, forwards `--sim-mjcf`, forces `--sim-with-omnihand`, fills `--task` from sidecar. |
| `--output-dir <path>` | required (unless `--teleop-only`) | LeRobot dataset destination. Pre-flight cleanup runs if a stub is present. |
| `--task "<instruction>"` | optional in scene mode (auto-filled) | Override the language instruction for this session. |
| `--sonic-checkpoint <path-to-.pt>` | required (or `--deploy-model-dir`) | Used to derive the deploy's `exported/*.onnx` bundle path. |
| `--wrist-bypass {off, ik}` | `ik` | Override SONIC's wrist target with operator IK refs for the 4 broken wrist DOFs. Keep `ik` for VLA recordings. |
| `--teleop-only` | off | Spin the loop, don't write disk. Useful for verifying bring-up. **B/X/Y are no-ops in this mode**, so the cube does NOT randomize. |
| `--no-vla` | off | Run deploy in motion-replay mode (StandStillReference); for IK / finger debug only. |
| `--sim-viewer` / `--no-sim-viewer` | viewer ON | Toggle the deploy's MuJoCo passive viewer. |

### `record_x2_dataset.py` — extra recorder flags

These are forwarded by the wrapper as `EXTRA_ARGS`, so you can append
them after the wrapper-level flags.

| Flag | Default | Effect |
|------|---------|--------|
| `--episode-seed <int>` | `None` | Seed the mirror's `placement_initializer`. Episode N uses `seed + N` so each B press is still distinct, but the whole sequence is reproducible. |
| `--scene-xml-path <path>` | resolved from `--robocasa-env` | Override the scene MJCF path (for developing custom scenes outside the bundled dir). |
| `--scene-state-sub-host <host>` / `--scene-state-sub-port <port>` | `localhost` / `5559` | Override the bridge's `scene_state` PUB endpoint. |
| `--scene-reset-pub-host <host>` / `--scene-reset-pub-port <port>` | `*` / `5560` | Override the recorder's `scene_reset` PUB endpoint. |
| `--no-omnihand` | off | Hardware debug only; do NOT use with scene recording (scenes assume OmniHand fingers). |

### `build_x2_robocasa_scene_xml.py`

| Flag | Effect |
|------|--------|
| `--env <name>` | Build one scene. |
| `--all` | Build every scene listed in `_KNOWN_ENVS`. |
| `--seed <int>` | RNG seed for the transient robocasa env used during scrape. Doesn't affect runtime randomization (the recorder's mirror re-rolls per episode). |
| `--output <path>` | Override the XML output path (default = bundled location). |
| `--no-verify` | Skip the post-merge MuJoCo compile check (debug only). |

---

## Selecting different scenes

Just change `--robocasa-env`:

```bash
# Cube → bowl pick-and-place (default)
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_pnp_cube_v0 \
    --robocasa-env X2PickPlaceCube \
    --sonic-checkpoint <path>.pt

# Bowl → green target placement
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_place_bowl_v0 \
    --robocasa-env X2PickPlaceBowl \
    --sonic-checkpoint <path>.pt
```

To make a session deterministic for debugging:

```bash
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_pnp_cube_seeded \
    --robocasa-env X2PickPlaceCube \
    --sonic-checkpoint <path>.pt \
    --episode-seed 42
```

To explicitly override the language instruction (e.g. for paraphrase
augmentation):

```bash
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_pnp_cube_paraphrased \
    --robocasa-env X2PickPlaceCube \
    --task "place the cube inside the bowl" \
    --sonic-checkpoint <path>.pt
```

---

## Post-flight verification

After saving an episode (X), verify the `task.*` columns landed and
have plausible values:

```bash
.venv/bin/python - <<'PY'
import pandas as pd, glob
parquet = sorted(glob.glob(
    "data/lerobot/x2_robocasa_pnp_smoke_v0/data/chunk-*/*.parquet"
))[-1]
df = pd.read_parquet(parquet)

def _scalar(s):
    return s.apply(lambda v: float(v[0]) if hasattr(v, '__len__') else float(v))

print(f"frames: {len(df)}  duration ~{len(df)/50:.1f}s")
task_cols = sorted(c for c in df.columns if c.startswith("task"))
print(f"task columns: {task_cols}")
print()
s = _scalar(df["task.success"]); r = _scalar(df["task.reward"])
print(f"task.success    : {int(s.sum())}/{len(s)} ticks "
      f"(first_on={int((s>0).idxmax()) if s.any() else 'never'})")
print(f"task.reward     : sum={r.sum():.2f} max={r.max():.2f} mean={r.mean():.3f}")
print()
print("phase ladder (any-on across episode):")
for col in task_cols:
    if col.startswith("task.subtask_"):
        v = _scalar(df[col])
        on = int(v.sum())
        first = int((v>0).idxmax()) if v.any() else None
        print(f"  {col:<32} {on:>5} ticks  first_on={first}")
PY
```

Expected pattern on a clean cube-in-bowl demo:

* `task.success` ticks > 0 in the last few seconds of the episode
  (once the cube is sitting in the bowl).
* `task.reward` mean is ~0.15-0.30 — most of the episode lives in the
  touch / grasp / lift bands, only the final seconds reach 1.00. A
  purely sparse-success demo would have `mean ≈ 0.07`.
* `first_on` for the six subtask columns should be **strictly
  increasing** through the ladder (approach → touch → grasp →
  off_table → above_bowl → in_bowl). Out-of-order firing usually
  means the corresponding threshold in `_PickPlaceCubeConstants` is
  too tight or too loose.

If `task.success` ticks == 0 despite a clearly successful demo, the
most likely culprits are:

1. The bridge's `scene_state` PUB never came up — check the deploy log
   for the `scene plumbing: …` line at boot (it should report
   non-zero `object collision geoms` and `L:N R:M hand geoms`).
2. The mirror is reading the wrong body name — diff the `.json`
   sidecar against the env's `_check_success` and make sure cube/bowl
   body IDs resolve in the mirror's MJCF.

If `task.subtask_grasp_cube` stays at 0 even though you were clearly
squeezing the cube, see the dedicated entry in §Troubleshooting.

---

## Troubleshooting

### "Viewer terminated as soon as I pressed A" / "deploy_alive=False forever"

Almost always one root cause: **a previous deploy container is still
holding the bridge's ZMQ ports** (5557 `x2_debug`, 5559 `scene_state`,
5570 `robot_pose`). The new bridge silently fails the colliding bind
but keeps stepping MuJoCo, so:

* The viewer renders normally and accepts commands.
* The recorder never sees a `x2_debug` heartbeat → `deploy_alive=False`
  for the entire session.
* Pressing A engages wrist-bypass with no observation feedback. The
  wrist target snaps the arm; the operator perceives this as "the
  viewer crashed" and Ctrl-Cs.

Diagnostic line in `${DEPLOY_LOG}`:

```
[ERROR] [...] [x2_deploy_onnx_ref]: x2_debug PUB failed to bind on port 5557: Address already in use
[bridge] robot_pose PUB setup failed: Address already in use (addr='tcp://*:5570')
```

`record_x2_dataset.sh` now defends against this in three layers:

1. **Pre-flight cleanup** (top of the script) — sweeps any leftover
   `docker_x2-x2sim-run-*` container and aborts with a clear error if
   ports 5557 / 5559 / 5570 are still bound after the sweep.
2. **Real readiness gate** — waits for the C++ deploy's first
   `CONTROL tick=` line (proves the bridge has bound, the deploy is
   running, and the policy is publishing) instead of the old
   `Launching ...` grep, which matched on `deploy_x2.sh`'s banner
   BEFORE the actual ros2 process started. Aborts immediately if
   `failed to bind` appears.
3. **Background deploy watchdog** — polls `kill -0` on the deploy PID
   once a second; the first time it disappears it prints a loud red
   banner with the last 20 lines of the deploy log AND `kill -INT`s
   the recorder process group so the operator gets immediate feedback
   instead of staring at a frozen viewer.

If you see the watchdog banner without having Ctrl-C'd, the deploy
really did die — the tail in the banner shows why (most often
`--max-duration` tripped, the MuJoCo viewer X-button was clicked, or a
sibling `docker kill --filter ancestor=x2sim` (e.g. from
`run_live_vla_demo.sh`) tore the container down).

### "B does nothing"

Most common: the recorder hasn't received its first Quest 3 packet
yet. Look for either of these in the log:

```
[recorder] waiting for first Quest 3 packet …
```

Until that line stops re-printing, B/X/Y are inert. Open the WebXR
app on the headset (`https://<workstation-ip>:8443`), tap *Start XR*,
and accept the WebXR + camera prompts. The recorder will log
`Quest3Reader connected …` and the wait line will stop.

Note: typing `b` into the host terminal does NOT press B. Only the
physical right-controller B button does.

### "scene_state PUB setup failed: No module named 'zmq'"

Stale `docker_x2` image. The bridge will auto-recover by lazy-
installing pyzmq inside the container; you'll see this sequence in the
log:

```
[bridge] scene_state PUB: pyzmq not present in this container; attempting one-shot 'pip3 install pyzmq'.
[bridge] pyzmq installed at runtime; scene_state PUB online.
[bridge] scene_state PUB bound at tcp://*:5559 (1 freejoints, 2 bodies).
[bridge] scene_reset SUB connected at tcp://localhost:5560 (topic='scene_reset').
```

If the auto-install fails (no network in the container, etc.), the
bridge will keep running but the cube will sit at its baked-in initial
pose forever and the recorder will never receive `scene_state`. To
bake pyzmq into the image permanently:

```bash
cd /home/stickbot/Projects/GR00T-WholeBodyControl/gear_sonic_deploy/docker_x2 && \
docker compose build x2sim
```

### "Cube doesn't move when I press B"

Check, in this order:

1. The recorder logged `[recorder] [B] scene_reset sent: …` — if not,
   B isn't reaching the recorder (see *B does nothing* above).
2. The bridge logged `[bridge] scene_state PUB bound …` — if not,
   pyzmq wasn't available (see previous entry).
3. The bridge actually applied the reset — look for `[bridge] applied
   scene_reset` (or warnings about unknown joint names). A name
   mismatch means the scene XML and the `.json` sidecar are out of
   sync; rebuild via `build_x2_robocasa_scene_xml --env <Name>`.

### "task.success stays 0 even though the cube is in the bowl"

Either the scene_state PUB isn't flowing (see above), or the oracle's
geometric thresholds are off for your env. The cube oracle uses
constants from `_PickPlaceCubeConstants` in
`robocasa_task_mirror.py`; tune `bowl_half_size_xy` /
`bowl_wall_height` / `cube_half_size` if your env's mesh dimensions
differ.

### "task.subtask_grasp_cube stays 0 even though I'm clearly squeezing the cube"

This means the bridge isn't seeing finger-vs-cube contacts. Check, in
order:

1. The scene XML shows finger collisions enabled. The fingertip
   cylinder geoms must NOT have `contype="0" conaffinity="0"`. If
   they do, the scene XML was built with the SONIC-era default;
   rebuild with `.venv_sim/bin/python -m
   gear_sonic.scripts.build_x2_robocasa_scene_xml --all` (the builder
   passes `disable_hand_collisions=False` automatically).
2. The bridge's boot log includes `L:N R:M hand geoms` with N, M
   non-zero (typically ≥40 each, depending on URDF version). If you
   see `L:0 R:0` the metadata sidecar is stale or doesn't include
   `hand_root_bodies` — rebuild scenes.
3. The bridge's contact-walker is firing: peek at the scene_state
   payload by running
   ```sh
   .venv/bin/python - <<'PY'
   import zmq, json, time
   ctx = zmq.Context.instance()
   s = ctx.socket(zmq.SUB); s.connect('tcp://localhost:5559'); s.subscribe(b'scene_state')
   topic, _, payload = s.recv().partition(b' ')
   d = json.loads(payload)
   print('grasp_contacts:', d.get('grasp_contacts'))
   print('fingertip_pos sides:', list(d.get('fingertip_pos', {}).keys()))
   PY
   ```
   With the fingers near the cube, `grasp_contacts['cube']['right']`
   should flip to `True`. If `fingertip_pos` is empty or absent the
   bridge is running an older build — restart the deploy.

### "task.reward stays at 0 throughout"

This is the older sparse 0/1 behaviour. The current shaped reward in
`compute_reward()` returns `[0.10, 0.25, 0.45, 0.65, 0.80, 1.00]` for
each phase reached. If you're getting 0.00 across the board:

1. **fingertip_pos is empty** — the approach phase needs fingertip
   world positions from the bridge. See "task.subtask_grasp_cube …"
   above for the diagnostic.
2. **You're running an older mirror** that hadn't been updated to use
   `fingertip_pos`. Pull latest `gear_sonic/utils/teleop/robocasa_task_mirror.py`
   and confirm `_phase_pick_place_cube` exists.

### Quest 3 WebXR shows "Not Secure" / "Mixed content"

WebXR refuses non-secure contexts. The Quest 3 reader uses a self-
signed cert by default; accept it once on the headset's browser. Do
NOT pass `--quest3-no-ssl` unless you're on a fully trusted local
network — WebXR will refuse the WebSocket otherwise.

### "RuntimeError: coroutine ignored GeneratorExit" on Ctrl-C

Cosmetic. The Quest 3 WebSocket coroutine doesn't shut down cleanly
in Python 3.10. Doesn't affect the dataset.

---

## Caveats and known limitations

* **SONIC tracking policy was trained without table contact in the
  loop.** The 25k *_g1 checkpoint we're shipping does not know about
  the table or any scene object. Expect the policy to push the wrists
  through obstacles when commanded; the Wrist Bypass override (default
  `--wrist-bypass ik`) compensates somewhat by routing the IK
  reference directly past SONIC for the broken wrist DOFs, but it is
  not a contact-aware controller. **Sim-to-real fidelity for collision
  events is therefore not preserved by this dataset.** Plan A is to
  retrain SONIC with the static scene fixtures present in the world
  model; Plan B is to swap in a contact-aware tracking policy. Until
  then, use these recordings to teach the GR00T VLA the *intent* of
  pick and place, not the precise impedance behaviour at table
  contact.

* **Per-episode randomization fires only on B in record mode.** In
  `--teleop-only` mode `_start_episode` early-returns, so
  `mirror.reset()` is never called and the cube/bowl sit at the
  baked-in initial pose for the whole session. To get randomized
  objects without writing data, run in record mode with a throwaway
  `--output-dir` and discard episodes with Y.

* **Mirror oracle is a port, not a delegate.** The success / reward /
  subtask helpers in `robocasa_task_mirror.py` are hand-translated
  from the corresponding methods in
  `robocasa/environments/locomanipulation/x2_tabletop_pnp.py`. If the
  upstream env logic changes (e.g. new pose-tolerance, new subtask
  signal), the mirror will drift. Tests in
  `tests/test_x2_robocasa_scene_mode.py` lock down the oracle's
  behaviour at the construction layer, but they don't compare against
  the env directly. When porting a new task, keep the oracle's
  pure-MuJoCo signature and document the source method it mirrors.

* **`gripper0_left_*` joint prefixing is intentionally avoided.** The
  robocasa env loads the X2 robot via `X2UltraFixedLowerBody`, which
  prefixes hand joints with the robosuite gripper namespace. The
  static scene XML built by `build_x2_robocasa_scene_xml.py` instead
  composes the canonical X2 + OmniHand layout (via
  `compose_x2_with_omnihand`) and grafts the env's table + objects
  onto that. Joint names match the deploy's expected
  `compose_x2_with_omnihand` layout, so address-based state copies
  between deploy and mirror are byte-compatible.

* **MuJoCo version pinning.** The gr00trobocasa fork was originally
  pinned to MuJoCo 3.2.6 / 3.3.2; we relaxed the assert in
  `robocasa/__init__.py` to also accept 3.5.0 (the version
  `compose_x2_with_omnihand` and the `MujocoFrameRenderer` need). If
  you bump MuJoCo further, smoke-test the bundled scene loads first
  and add the new version to `_SUPPORTED_MUJOCO`.

* **No camera observations from the mirror.** The
  `MujocoFrameRenderer` renders ego-view from its own copy of the
  scene XML. The mirror does not render cameras; it only runs
  `mj_forward` to support oracle queries. If you need ground-truth
  RGB-D from the bridge's authoritative state, add an offscreen
  renderer to the bridge and publish images on a third ZMQ topic — do
  NOT try to render from the mirror, its qpos for the X2 itself is
  stale (only scene objects are mirrored).

* **Both venvs need the gr00trobocasa fork.** The `RobocasaTaskMirror`
  lazy-imports `robocasa.models.grippers.omnihand_grippers` on the
  first **B** press to spin up a robosuite env for object
  randomization. That module exists ONLY in the gr00trobocasa fork
  (upstream `robocasa==1.0.0` from PyPI doesn't have it). Both
  `.venv_sim` and `.venv` therefore need the fork installed:

  ```bash
  # First time (or when the fork's setup.py changes):
  VIRTUAL_ENV=$PWD/.venv uv pip install -e decoupled_wbc/dexmg/gr00trobocasa
  VIRTUAL_ENV=$PWD/.venv uv pip install 'mujoco==3.5.0'  # see version note below
  ```

  The fork's `setup.py` pins `mujoco==3.2.6`, but our scene XMLs use
  the `colorspace` texture attribute (introduced in MuJoCo 3.3). The
  assert in `decoupled_wbc/dexmg/gr00trobocasa/robocasa/__init__.py`
  was relaxed to also accept 3.5.0, so re-pinning to 3.5.0 after the
  install is mandatory for `.venv`. (`.venv_sim` already does this via
  `gear_sonic`'s `[sim]` extra.)

---

## Smoke-test gates

### Hardware-free pytest gate

`tests/test_x2_robocasa_scene_mode.py` covers every Python-only seam
in the architecture (scene XML loading, ZMQ wire format, mirror
oracles, recorder argparse). 20 tests in `.venv_sim`, 18 + 2 skipped
in `.venv` (the deterministic-reset tests need the gr00trobocasa
fork).

```bash
.venv_sim/bin/python -m pytest tests/test_x2_robocasa_scene_mode.py -v
.venv/bin/python     -m pytest tests/test_x2_robocasa_scene_mode.py -v
```

### Hardware-required smoke

Two sequenced commands, both documented in the operator-facing
`sample_commands.md`:

* **Smoke #1 (no recording).** `--teleop-only --robocasa-env
  X2PickPlaceCube` — confirms the bridge boots with the scene MJCF and
  the operator can teleop arms freely. Quest 3 optional (the bridge
  will load and render even without the headset).
* **Smoke #2 (one-episode write).** Same command sans `--teleop-only`
  but plus `--output-dir`. Expected to write a real LeRobot v2.1
  dataset with non-zero `task.subtask_grasp_cube` ticks during the
  carry phase and a non-zero `task.success` window once the cube lands
  in the bowl.

The post-flight parquet snippet in §Post-flight verification is the
end-to-end validator for smoke #2.
