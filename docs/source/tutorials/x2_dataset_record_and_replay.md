# X2 Dataset Record and Replay (Quest 3 → SONIC → LeRobot)

This page is the operator runbook for the **Quest 3 → X2 closed-loop
dataset recorder**: a single 50 Hz pipeline that lets a VR-wearing
operator drive a SONIC-stabilised X2 in MuJoCo and record everything
they do as a [LeRobot](https://github.com/huggingface/lerobot) v2.1
dataset suitable for fine-tuning Isaac-GR00T N1.7.

It also covers **replay** — three ways to play a recorded session
back: as a parquet inspection, as an MP4 of `observation.images.ego_view`,
and as a "re-publish the saved motion tokens to a fresh deploy" loop
that re-creates the on-robot trajectory in sim.

```{admonition} Where this fits in the M-series
:class: note
This pipeline replaces the synthetic M3 / mujoco-replay M5 datasets
with operator-in-the-loop demonstrations for **M6+**. It does **not**
replace the SONIC training data — SONIC is unchanged; we just sample
from its action distribution while the operator drives the upper body.
```

```{admonition} v0 scope
:class: warning
* **Stationary robot only.** Lower body, waist, and head are pinned to
  the trained X2 stand pose for every frame. Only the 14 arm DOFs and
  20 OmniHand DOFs follow operator input.
* **Quest 3 controllers, not bare hands.** Trigger / grip drive a
  uniform finger curl. Bare-hand XRHand skeleton retargeting is a v1
  follow-up.
* **One `--task` string per session.** All episodes recorded by one
  recorder process share the same language instruction.
* **Closed-loop "freeze-pose" tokenization.** Every tick we build an
  11-frame virtual clip where every frame equals the current `body_q`
  and feed it through the SONIC encoder + FSQ. The deploy chases that
  "stay here" token until the next 50 Hz tick re-tokenizes against the
  operator's new wrist position.
```

---

## 1. Architecture

A single recorder process owns the VR ingress, IK, online tokenization,
ZMQ pub/sub, MuJoCo render and LeRobot writer; the C++ deploy + MuJoCo
bridge run as a co-launched sibling.

```text
Quest 3 (WebXR https://<host>:8443)
        │  (3-pt pose + buttons + triggers, WebSocket)
        ▼
[recorder process: gear_sonic/scripts/record_x2_dataset.py]
   • Quest3Reader          ── raw VR
   • VRArmTeleop (DLS IK)  ── 7+7 arm joint targets
   • compose body_q        ── trained stand pose for legs/waist/head
   • OnlineSonicTokenizer  ── 64-D motion_token (freeze-pose strategy)
   • ZMQ PUB :5556 'pose'  ── 50 Hz, idle stand token while VR is silent
        │
        ▼  pose msg = (joint_pos_mj, root_quat, motion_token, hand_q, frame_idx)
[deploy_x2.sh sim --vla --sim-profile gantry --sim-with-omnihand --sim-viewer]
   • C++ deploy reads ZMQ 'pose'   ── caches last token (no watchdog)
   • SONIC ONNX policy steps       ── 22-DOF body action
   • x2_mujoco_ros_bridge.py --viewer
        ─ steps the X2 + OmniHand sim
        ─ MuJoCo passive viewer window  ◄── the operator's monitor
        ─ publishes x2_debug :5557 (proprio feedback)
        │
        ▼
[recorder process again]
   • subscribes x2_debug :5557 for ground-truth proprio
   • when recording: MujocoFrameRenderer → ego_view (640×480) →
     Gr00tDataExporter → LeRobot v2.1 episodes (parquet + mp4)
```

```{admonition} Why two MuJoCo processes?
:class: tip
The deploy bridge **simulates the robot** (kinematics + dynamics + the
SONIC tracking policy in the loop). The recorder's `MujocoFrameRenderer`
is **just a renderer** — it consumes the deploy's published
`x2_debug` proprio and renders an `ego_view` image off-screen for the
LeRobot dataset. They never collide, because the renderer doesn't
step physics.
```

---

## 2. One-time setup

### 2.1 Workstation prerequisites

The recorder uses the standard GR00T training venv (`.venv/`), not the
data-collection venv. Make sure:

* You can already run `gear_sonic_deploy/deploy_x2.sh sim` end-to-end
  (see [Quickstart](../getting_started/quickstart.md) and
  [VR Teleop Setup](../getting_started/vr_teleop_setup.md)).
* The X2 SONIC checkpoint is on disk, with the `exported/` ONNX bundle
  next to the `.pt` (the recorder needs the `.pt` for the encoder/FSQ;
  the deploy needs the ONNX bundle for the tracking policy):

  ```bash
  ls /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/
  # config.yaml exported/ last.pt meta.yaml model_step_025000.pt
  ```

* `.venv/bin/python` resolves to the GR00T env with `zmq`, `mujoco`,
  `lerobot`, and `websockets` installed.

### 2.2 Quest 3 prerequisites

* The Quest 3 is on the same LAN as the workstation.
* The Meta Quest Browser trusts your workstation's self-signed cert
  (the recorder boots its own HTTPS server on port 8443 — see the
  initial banner output for the exact URL).
* Operator wears the headset, holds **both** controllers, and starts
  in a relaxed shoulder-down posture for the engage-time calibration.

```{tip}
The recorder embeds its own Quest 3 WebSocket / HTTPS server, so you
do **not** need to start `run_quest3_server.sh` separately.
```

---

## 3. Launch (one command)

The wrapper script `gear_sonic/scripts/record_x2_dataset.sh`
co-launches the deploy in the background and the recorder in the
foreground. It already passes `--sim-profile gantry`,
`--sim-with-omnihand`, and `--sim-viewer` so the MuJoCo window opens
automatically.

### 3.1 Pure VR teleop (no dataset writes) — validate first

Use this before your first recording session to make sure the
SONIC + IK + viewer chain feels right. The recorder still does the
full Quest 3 → IK → SONIC token → ZMQ pub at 50 Hz; it just skips the
exporter and the ego renderer.

```bash
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --teleop-only \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

In `--teleop-only` mode, the **B / X / Y** buttons become no-ops. **A**
still engages IK calibration.

### 3.2 Full record session

```bash
cd /home/stickbot/Projects/GR00T-WholeBodyControl && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_quest3_v0 \
    --task "wave hello with both hands" \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

### 3.3 Wrapper flags

| Flag | Default | Description |
| ---- | ------- | ----------- |
| `--sonic-checkpoint PATH` | required | `.pt` checkpoint; the recorder loads encoder + FSQ. The deploy's ONNX bundle is auto-derived from this path's directory unless you override with `--deploy-model-dir`. |
| `--output-dir DIR` | required (unless `--teleop-only`) | LeRobot v2.1 dataset root. Created if missing. |
| `--task STRING` | required (unless `--teleop-only`) | Language instruction stamped on every episode. |
| `--teleop-only` | off | Skip exporter + ego renderer + dataset writes. Ideal for sanity-checking the loop. |
| `--sim-viewer` / `--no-sim-viewer` | `--sim-viewer` | Open / suppress the MuJoCo passive viewer in the deploy. |
| `--deploy-model-dir DIR` | `dirname(--sonic-checkpoint)` | Override the ONNX bundle dir for the deploy. |
| `--sim-duration SECS` | `3600` | Auto-stop the deploy after N seconds. |
| anything else | — | Forwarded verbatim to `record_x2_dataset.py`. |

### 3.4 Useful pass-through flags

These go on the same `record_x2_dataset.sh` line and the wrapper
forwards them to the Python recorder:

| Flag | Notes |
| ---- | ----- |
| `--tokenizer-device cuda` | Run the SONIC encoder + FSQ on GPU. CPU is fine for 50 Hz on a recent workstation. |
| `--hand-input {trigger,grip,max}` | Which controller analog drives finger curl. Default `trigger`. `max` picks whichever analog is greater this frame. |
| `--rate 50` | Publish + record cadence. Match `FPS` in the dataset features (default 50). |
| `--no-omnihand` | Debug only. The trained M5/M6 datasets all carry the OmniHand mesh, so don't use this unless you know why. |
| `--quest3-no-ssl` | Disable TLS on the WebXR server. WebXR refuses non-secure contexts so this is for debugging only on a trusted LAN. |
| `--ik-damping 0.08`<br>`--ik-rotation-weight 0.5`<br>`--ik-position-scale 1.0`<br>`--ik-per-tick-step-rad 0.30` | DLS IK tuning. The defaults track ~5 cm/s wrist motions cleanly without joint-limit thrashing. |

---

## 4. Operator workflow

### 4.1 Quest 3 controller cheat-sheet

| Button | Action |
| ------ | ------ |
| **A** | Engage / re-calibrate. Snapshots the current wrist anchors against the X2 neutral arm pose. Press once at session start, then any time your "rest" pose drifts (e.g. you took the headset off and put it back on). |
| **B** | Start a fresh episode. No-op if one is already recording or if `--teleop-only` is set. |
| **X** | Stop and *save* the current episode → writes a parquet shard + mp4 chunk to `--output-dir`. |
| **Y** | Stop and *discard* the current episode → drops the in-memory frame buffer; on-disk dataset is unchanged. |
| **Trigger / Grip** | Per-side analog finger curl. See `--hand-input`. |

### 4.2 Step-by-step session

1. Run the wrapper from your interactive terminal (it needs `DISPLAY`
   for the MuJoCo viewer).
2. Read the banner:
   ```text
   ─────────────────────────────────────────────────
     X2 Dataset Recorder + MuJoCo Deploy (VLA mode)
   ─────────────────────────────────────────────────
     output_dir        : data/lerobot/x2_quest3_v0
     task              : wave hello with both hands
     sonic_checkpoint  : /home/.../model_step_025000.pt
     deploy_model_dir  : /home/.../h200-iter-25000-...
     sim_duration      : 3600s
     sim_viewer        : true
     deploy_log        : /tmp/deploy_x2_record_XXXXXX.log
     Quest 3 WebXR URL : https://10.0.0.42:8443
   ─────────────────────────────────────────────────
   ```
3. Watch `[deploy] …` startup lines stream in. The MuJoCo viewer pops
   open at the `gantry_hang` initial pose, then ramps to `DEFAULT_DOF`
   over ~2 s as the recorder's idle stand tokens come in.
4. On the Quest 3, open the WebXR URL, accept the cert, hit
   **Connect WS** + **Start VR**.
5. Recorder log shows `Quest 3 connected; first packet received`.
6. Stand in your neutral pose, arms relaxed, then squeeze **A** on
   either controller. The recorder logs `[A] engaged: wrist anchors
   captured`.
7. Move your arms. The MuJoCo X2 should follow within ~50 ms. Use the
   `--sim-viewer` window to confirm the policy is tracking and the
   gantry strap is keeping the robot upright (it should be).
8. Press **B** to start an episode, perform the task, press **X** to
   save it. The recorder logs the episode index and frame count.
9. Repeat **B → demo → X** as many times as you want — they all land in
   the same `--output-dir` with the same `--task` tag.
10. Press **Ctrl-C** in the terminal to shut down. Any open episode
    that was started but not closed with X is auto-saved on shutdown so
    a stray Ctrl-C doesn't lose the last 30 s of work.

```{admonition} Safety: the robot can never fall while idle
:class: tip
Three independent layers keep the X2 upright when nothing is
happening:

1. The recorder publishes a "stay at trained stand pose" token at
   50 Hz even before VR connects (see `_publish_idle()` in
   `x2_dataset_recorder.py`).
2. The C++ deploy caches the last received `pose` message
   indefinitely and re-uses it if the stream goes silent — no
   watchdog cliff (`zmq_pose_input_source.cpp`).
3. `--sim-profile gantry` keeps the elastic band ON forever at
   `gantry_hang` length (~88 % body weight off the legs).

The combination means you can start the recorder, walk away, come
back hours later, and the robot will still be in the trained stand
pose with the gantry strap engaged.
```

---

## 5. What lands on disk

Every saved episode appends to a standard LeRobot v2.1 layout under
`--output-dir`:

```text
data/lerobot/x2_quest3_v0/
├── meta/
│   ├── info.json
│   ├── modality.json          ← the GR00T modality config
│   ├── episodes.jsonl         ← one row per saved episode
│   ├── episodes_stats.jsonl
│   ├── stats.json
│   ├── tasks.jsonl            ← maps task index → language string
│   └── ...
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── videos/
    └── chunk-000/
        └── observation.images.ego_view/
            ├── episode_000000.mp4
            ├── episode_000001.mp4
            └── ...
```

The parquet rows contain (per frame, all `float64` unless noted):

| Field | Shape | Source |
| ----- | ----- | ------ |
| `observation.state` | `(N_body + 2 * N_hand,)` | Pinocchio-ordered body + omnihand joints from the deploy's `x2_debug`. Falls back to the commanded body_q if the deploy hasn't published yet. |
| `observation.projected_gravity` | `(3,)` | Body-frame gravity from the deploy's base quaternion. |
| `observation.images.ego_view` | `(480, 640, 3)` uint8 | Off-screen render of the *observed* body_q (not the commanded one) — keeps image and proprio aligned to ground truth. |
| `action.motion_token` | `(64,)` | The commanded SONIC FSQ token for this tick. |
| `action.left_hand_joints` | `(10,)` | Commanded left OmniHand joints. |
| `action.right_hand_joints` | `(10,)` | Commanded right OmniHand joints. |
| `task` | string | The session's `--task` value. |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | scalars | LeRobot bookkeeping (filled in by `Gr00tDataExporter`). |

```{admonition} Discarded episodes leave no on-disk trace
:class: note
`Gr00tDataExporter` only writes a parquet shard on `save_episode()`.
The `Y`-button path simply drops the in-memory buffer, so the
on-disk dataset shape (parquet count, episode indices, video files)
is unchanged.
```

---

## 6. Replay

There are three useful "replay" recipes. They all consume the same
on-disk dataset and require no Quest 3.

### 6.1 Quick parquet inspection

For a session smell-test (frame counts, action ranges, NaNs):

```python
import pandas as pd
import pyarrow.parquet as pq

df = pq.read_table(
    "data/lerobot/x2_quest3_v0/data/chunk-000/episode_000000.parquet"
).to_pandas()

print(df.shape)                          # (T, ~12)
print(df.columns.tolist())
print(df["task"].iloc[0])
print(df["action.motion_token"].iloc[0]) # 64-D vector
```

For session-wide stats:

```python
import json
from pathlib import Path

eps = [
    json.loads(line)
    for line in Path("data/lerobot/x2_quest3_v0/meta/episodes.jsonl").read_text().splitlines()
]
print(f"{len(eps)} episodes, {sum(e['length'] for e in eps)} total frames")
```

### 6.2 Re-watch the ego_view footage

The recorder writes one MP4 per episode under
`videos/chunk-000/observation.images.ego_view/`. Open the MP4 in any
video player; that's exactly the frame stream the trained policy will
see at deploy time (the same camera, same resolution, same FPS).

Quick CLI peek:

```bash
ls data/lerobot/x2_quest3_v0/videos/chunk-000/observation.images.ego_view/
mpv data/lerobot/x2_quest3_v0/videos/chunk-000/observation.images.ego_view/episode_000000.mp4
```

### 6.3 Re-publish the recorded motion tokens to a fresh deploy

This is the strongest acceptance gate: it proves the saved
`action.motion_token` stream alone is enough to re-create the
on-robot trajectory, end-to-end, with no operator in the loop. We do
not yet ship a one-line script for this, but the recipe is short:

1. Launch a fresh deploy (no recorder, no VR):

   ```bash
   gear_sonic_deploy/deploy_x2.sh sim \
       --vla \
       --sim-profile gantry \
       --sim-with-omnihand \
       --sim-viewer \
       --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501
   ```

2. In a second shell, stream the saved tokens at the original 50 Hz
   over the same `pose` ZMQ topic the recorder used:

   ```python
   # gear_sonic/scripts/replay_x2_dataset.py  (sketch)
   import time
   import numpy as np
   import pyarrow.parquet as pq
   import zmq
   from gear_sonic.scripts.live_vla_publish_motion_token import (
       DEFAULT_STAND_POSE_MUJOCO_RAD,
   )
   from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message

   df = pq.read_table(
       "data/lerobot/x2_quest3_v0/data/chunk-000/episode_000000.parquet"
   ).to_pandas()

   ctx = zmq.Context.instance()
   pub = ctx.socket(zmq.PUB)
   pub.bind("tcp://*:5556")
   time.sleep(0.5)  # wait for the deploy SUB to wire up

   body = np.asarray(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32)
   for frame_idx, row in df.iterrows():
       msg = pack_pose_message(
           dict(
               joint_pos_mj=body,
               root_quat_xyzw=np.array([0, 0, 0, 1], dtype=np.float32),
               motion_token=row["action.motion_token"].astype(np.float32),
               left_hand_joints=row["action.left_hand_joints"].astype(np.float32),
               right_hand_joints=row["action.right_hand_joints"].astype(np.float32),
               frame_index=np.array([frame_idx], dtype=np.int64),
           ),
           topic="pose",
           version=4,
       )
       pub.send(msg)
       time.sleep(1 / 50.0)
   ```

3. Watch the MuJoCo viewer. The robot should re-execute the demo
   identically (within tracking-policy noise). If it doesn't, the
   dataset isn't faithful — the recorder is doing something you
   didn't see, or the FSQ codebook drifted between sessions.

```{admonition} Roadmap
:class: note
A `replay_x2_dataset.py` CLI that wraps recipe 6.3 + a
multi-episode sequencer + a `--start-frame` / `--end-frame` window
selector is on the v1 backlog. Today, copy-paste the snippet.
```

---

## 7. Acceptance gates and unit tests

Run these before merging any change to the recorder:

```bash
.venv/bin/python -m pytest tests/test_x2_arm_ik_smoke.py -x
.venv/bin/python -m pytest tests/test_record_x2_dataset_schema.py -x
```

* [`tests/test_x2_arm_ik_smoke.py`](../../../tests/test_x2_arm_ik_smoke.py)
  pins the DLS IK round-trip (FK → target → IK → FK should match
  within sub-mm) and the joint-limit clamp invariant for both the
  left and right 7-DOF arm chains.
* [`tests/test_record_x2_dataset_schema.py`](../../../tests/test_record_x2_dataset_schema.py)
  pins the hand-retargeter shapes, the `VRArmTeleop` step's
  neutral-pose fallback (no Quest data ⇒ neutral arm `q`), the
  LeRobot feature schema, and the
  `OnlineSonicTokenizer` FSQ-lattice property (the encoded token
  must lie on the FSQ codebook). The tokenizer test skips
  cleanly when the SONIC checkpoint isn't on disk.

---

## 8. Pointers into the implementation

| File | Role |
| ---- | ---- |
| [`gear_sonic/utils/teleop/solver/arm/x2_arm_fk.py`](../../../gear_sonic/utils/teleop/solver/arm/x2_arm_fk.py) | Pure-numpy FK + analytical Jacobian for the 7-DOF X2 arm chain. |
| [`gear_sonic/utils/teleop/solver/arm/x2_arm_ik.py`](../../../gear_sonic/utils/teleop/solver/arm/x2_arm_ik.py) | Single-step DLS IK solver, joint-limit clamped. |
| [`gear_sonic/utils/teleop/vr_arm_teleop.py`](../../../gear_sonic/utils/teleop/vr_arm_teleop.py) | Engage-time calibration + per-tick IK over the Quest 3 3-pt pose. |
| [`gear_sonic/utils/teleop/x2_hand_retarget.py`](../../../gear_sonic/utils/teleop/x2_hand_retarget.py) | Trigger/grip → 10-DOF OmniHand command (open/closed motor anchors). |
| [`gear_sonic/utils/teleop/online_sonic_tokenizer.py`](../../../gear_sonic/utils/teleop/online_sonic_tokenizer.py) | Per-frame `body_q` → 64-D motion token via the freeze-pose virtual clip. |
| [`gear_sonic/utils/teleop/x2_dataset_recorder.py`](../../../gear_sonic/utils/teleop/x2_dataset_recorder.py) | Top-level orchestrator: ZMQ pub/sub, MuJoCo render, button state machine, LeRobot writer. |
| [`gear_sonic/scripts/record_x2_dataset.py`](../../../gear_sonic/scripts/record_x2_dataset.py) | CLI shim. |
| [`gear_sonic/scripts/record_x2_dataset.sh`](../../../gear_sonic/scripts/record_x2_dataset.sh) | Co-launches the deploy + recorder. |
| [`gear_sonic/scripts/process_dataset.py`](../../../gear_sonic/scripts/process_dataset.py) | Post-process / merge / clean LeRobot datasets. |
| [`gear_sonic/scripts/mock_vla_publish_stand_token.py`](../../../gear_sonic/scripts/mock_vla_publish_stand_token.py) | Reference for "publish a constant stand-still token over `pose`" — same wire format the recorder's idle path uses. |

---

## 9. Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Recorder logs `waiting for first Quest 3 packet …` forever | Headset can't reach the WebXR HTTPS server | Open the URL in the Quest 3 browser, accept the self-signed cert, then tap **Connect WS** + **Start VR**. Verify the workstation's firewall allows ports 8443 (HTTPS) and 8765 (WebSocket). |
| Robot jolts on engage | Operator wrists were far from the X2 neutral elbow pose at the **A** press | Stand with elbows relaxed at your sides before pressing **A**. Re-press any time you drift. |
| MuJoCo viewer doesn't open | The deploy was launched headless | Pass `--sim-viewer` (default in the wrapper). If you used the wrapper and still don't see a window, ensure `DISPLAY` is set in the shell that launched the wrapper. |
| Saved episode has 0 frames | Pressed **X** before any tick advanced (e.g. before VR connected) | Check the recorder log for `[X] dropping 0 frames (no frames)`. Press **B** again, wait for the recorder to log non-zero frame counts in its periodic status, then press **X**. |
| Tracking policy lags by ~100 ms | The CPU-side encoder is too slow | Pass `--tokenizer-device cuda`. The recorder ships its own `OnlineSonicTokenizer` so this is a one-flag change. |
| `[recorder] render warn (frame skipped): …` | EGL renderer hiccup; recorder drops the frame and continues | Safe to ignore unless it happens > 1 % of frames. If it does, drop the resolution (`--render-width 320 --render-height 240`) or move the renderer to a different GPU. |
| Deploy log spams `tilt watchdog` errors | The recorded body_q drifted out of the trained distribution (e.g. lower body got modified) | The recorder pins legs/waist/head to `DEFAULT_STAND_POSE_MUJOCO_RAD`. If you patched that, revert. |

---

## 10. Next steps

* **Train Isaac-GR00T N1.7 on the recorded dataset.** See
  [VLA Training](vla_training.md) for the fine-tuning recipe — point
  `--dataset_path` at `--output-dir` and you're done.
* **Merge multiple sessions before training.** Use
  `gear_sonic/scripts/process_dataset.py` to merge sessions that
  share the same task or to clean up partially-recorded ones.
* **Run a closed-loop sim eval against the recorded checkpoint.**
  See [VLA Inference](vla_inference.md). The recorder writes the
  same `observation.state` schema the inference path consumes, so
  no manual mapping is required.
