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
  20 OmniHand DOFs follow operator input. **The X2 waist is held at
  neutral**: torso pitch / roll / yaw are not driven from VR (3-point
  Quest 3 tracking can't disambiguate operator torso tilt from arm
  reach without an extra chest tracker or the Meta Movement SDK).
  See [Section 8](#8-v0-limitations).
* **Quest 3 controllers AND bare-hand XRHand are both supported.**
  Trigger / grip drive a uniform finger curl per side. When the
  WebXR `XRHand` skeleton is reported, per-finger curls retarget
  the OmniHand on a per-finger basis with operator-specific affine
  normalization (see [Section 2.3](#23-vr-operator-calibration) for
  capture; [Section 8](#8-v0-limitations) for the hand-tracking
  journey log). The two input sources are multimodal — operators
  can pick up / set down a controller mid-session and the recorder
  switches transparently.
* **One `--task` string per session.** All episodes recorded by one
  recorder process share the same language instruction.
* **Stateless head-relative wrist mapping.** Operator wrist positions
  are converted into the head-yaw frame and passed through a per-arm
  affine map fitted offline by `vr_operator_calibrate.py`. There is no
  engage-anchor and the robot wrist target is invariant to operator
  body rotation in place. See [Section 2.3](#23-vr-operator-calibration).
```

---

## 0. Quick command reference

The full operator runbook is in [Section 3 (Launch)](#3-launch-one-command),
[Section 4 (Operator workflow)](#4-operator-workflow) and
[Section 6 (Replay)](#6-replay). This section is a one-screen cheat-
sheet for the three verbs and the most common flags.

A condensed version with every command on one page also lives in
[`sample_commands.md`](../../../sample_commands.md) at the repo root.

### Calibrate (one-time per operator)

```bash
.venv/bin/python -m gear_sonic.scripts.vr_operator_calibrate \
    --operator-id <name>
```

Writes `data/operator_calibrations/<name>.yaml`. Re-run when switching
operators or after a session where the wrist mapping felt off.

### Teleop (kinematic, no SONIC) — fastest debug loop

```bash
.venv/bin/python -m gear_sonic.scripts.teleop_x2_kinematic \
    --output-dir data/lerobot/x2_quest3_kinematic_v6 \
    --task "<task string>" \
    --rate 50 --hand-input max
```

Record by pressing **B** on the Quest 3 controller; **X** saves, **Y**
discards. Drop `--output-dir` for pure-viewer teleop with no disk
writes. Per-finger smoothing filter is enabled by default (see
[v0.6 below](#finger-signal-smoothing-v06-may-12)); pass
`--no-finger-filter` to disable for an A/B baseline.

### Record (full SONIC-stabilised loop)

```bash
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_quest3_v0 \
    --task "<task string>" \
    --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

The wrapper co-launches the C++ deploy + the Python recorder. Same
operator buttons as kinematic teleop. All filter flags pass through.
For a pure connectivity smoke-test that doesn't write to disk, add
`--teleop-only` and drop `--output-dir` / `--task`.

### Replay (kinematic, from parquet)

```bash
.venv/bin/python -m gear_sonic.scripts.replay_x2_kinematic \
    --dataset x2_quest3_kinematic_v6 --episode 0
```

Opens a passive MuJoCo viewer on the recorded `action.body_q_mj` +
hand columns (v1 schema; auto-falls-back to `action.commanded_body_q_mj`
for legacy v0 datasets). No Quest 3, no IK, no policy. See
[Section 6.4](#64-kinematic-mujoco-replay-replay_x2_kinematicpy) for
windowing + loop options.

### Offline retargeting replay (re-derive parquet from the debug NPZ)

When you want to test a new retargeter / calibration / filter without
restrapping the headset:

```bash
.venv/bin/python -m gear_sonic.scripts.replay_recorded_dataset \
    --npz     <episode>.npz \
    --parquet <episode>.parquet \
    --output-dir /tmp/replay_out
```

Add `--apply-finger-filter {auto,always,never}` to A/B the v0.6
smoothing filter against the raw signals; `auto` is the default and
uses the pre-computed `*_filtered` channels if the NPZ has them
(post-v0.6 recordings) or applies the filter offline otherwise.

### Common shared flags

| Flag | Effect |
| ---- | ------ |
| `--no-finger-filter` | Disable the v0.6 EMA + deadband-hold smoother on the hand inputs. Both teleop_x2_kinematic and record_x2_dataset accept this. |
| `--finger-filter-alpha FLOAT` | Override the EMA alpha (default 0.5). |
| `--finger-filter-hold-window INT` | Override the deadband-hold window (default 8 frames = 160 ms at 50 Hz). |
| `--finger-filter-hold-std FLOAT` | Override the std threshold for entering the held-pose latch (default 0.005). |
| `--apply-finger-filter {auto,always,never}` | Replay only. `auto` = use NPZ's `*_filtered` if present, else apply offline. `never` = use raw. `always` = force offline pass. |
| `--hand-input {trigger,grip,max}` | Which controller analog drives the uniform fallback grasp when XRHand isn't reported. |
| `--calibration PATH` / `--recalibrate` / `--operator-id NAME` | Operator calibration plumbing. |
| `--task STR` | Required when `--output-dir` is set. Stamped on every recorded frame. |

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

* (Optional, but recommended once you start recording datasets)
  the Rerun viewer for multi-camera replay lives in a separate
  `.venv-viewer/` to avoid clashing with the planner's `pinocchio`
  pin on `numpy<2`. Bootstrap it once with:

  ```bash
  bash install_scripts/install_viewer.sh
  ```

  See § 6.2.1 for why it's a separate venv and how to invoke it
  via the [`view_x2_recorded_dataset.sh`](../../../../gear_sonic/scripts/view_x2_recorded_dataset.sh)
  wrapper. Pins live in [`requirements-viewer.txt`](../../../../requirements-viewer.txt).

  :::{tip}
  **Optional dependencies are auto-installed on first launch.**
  The teleop / calibration / record entry-points call
  `ensure_runtime_deps()` before doing anything else, which
  pip-installs any missing optional packages (`gtts` for Quest 3
  audio prompts; `datasets` / `av` / `lerobot` for the recorder's
  dataset writer). You'll see a one-line log on first run:
  `[runtime-deps] N optional package(s) for ... missing: [...]. Auto-installing into /…/python.`
  Subsequent runs are no-ops because the packages are already
  importable.

  Two opt-out paths if you don't want surprise pip mutations
  (reproducible-build setups, locked Docker images):

  1. Set `GEAR_SONIC_NO_AUTO_INSTALL=1`. The script will print which
     packages are missing and exit with the degraded fallback (no
     audio, or a hard ImportError for the recorder).
  2. Pre-install everything with
     `pip install -e gear_sonic[data_collection]`, OR
     `pip install -r requirements-teleop-record.txt` for a thin
     standalone install. Both lists are kept in sync with
     `gear_sonic.utils.install.runtime_deps`.
  :::

### 2.2 Quest 3 prerequisites

* The Quest 3 is on the same LAN as the workstation.
* The Meta Quest Browser trusts your workstation's self-signed cert
  (the recorder boots its own HTTPS server on port 8443 — see the
  initial banner output for the exact URL).
* Operator wears the headset, holds **both** controllers, and is
  comfortable getting into the three calibration poses described
  below.

```{tip}
The recorder embeds its own Quest 3 WebSocket / HTTPS server, so you
do **not** need to start `run_quest3_server.sh` separately.
```

### 2.3 VR operator calibration

The recorder retargets operator wrist motion to robot wrist targets via
a **per-operator calibration YAML** that captures three things:

* **Anatomy** — arm length, shoulder width, height all differ between
  operators.
* **Pose habits** — how relaxed the elbows are, whether the wrists
  flex inward, etc.
* **Reach envelope** — the working volume the operator can naturally
  cover with comfortable arm motion.

The calibration is fit from three static poses. You capture once per
operator (or once per setup change) and reuse the same YAML for every
subsequent session.

#### Capture flow

1. Boot only the calibration script (no MuJoCo, no deploy, no
   recorder):

   ```bash
   .venv/bin/python -m gear_sonic.scripts.vr_operator_calibrate \
       --operator-id stickbot
   ```

   The default output path is
   `data/operator_calibrations/<operator-id>.yaml`. Override with
   `--output PATH` if you want a different name.

2. The Quest 3 WebXR URL prints in the banner. Open it on the headset,
   accept the cert, hit **Connect WS**.

   ##### Audio + UI sanity check (do this before entering VR)

   Before you press **Start VR**, click the **Test audio** button on
   the WebXR page. The headset should speak *"Audio test successful.
   Calibration prompts will speak through this audio device."* If
   you don't hear it:

   * **Headset volume** is the most common culprit — turn it up using
     the volume rocker on the right side of the headset.
   * The browser status row at the bottom of the page logs
     `TTS voices loaded: N` once the speech engine is ready, and
     `TTS prime started` / `TTS prime ended` when the priming
     utterance plays. If you see `TTS prime error: …`, the headset
     browser blocked TTS — close the tab, reopen, click **Test
     audio** *first*, then **Start VR**.
   * Quest 3 Browser requires a real user gesture to "unlock"
     `speechSynthesis`. The Test audio button provides exactly that
     gesture. Subsequent server-driven prompts (during calibration)
     reuse the unlocked engine, so once the test works the rest
     speaks automatically.

   The dom-overlay carrying the calibration UI is requested as a
   WebXR feature in `requestSession()`; if the runtime denies it
   (older browser, missing permissions), the page logs
   `domOverlayState: …` in the status row. In that case, use a
   hand-mirror or the desktop preview window to read the prompts
   while wearing the headset.

3. The headset browser shows a stick-figure overlay and **plays a
   pre-rendered MP3 audio prompt** for **Pose 1 of 4 — Arms fully
   straight down**:
   *"Stand relaxed with both arms hanging fully straight down at your
   sides. Do not bend your elbows. Press A on either controller when
   ready."*
   Get into the pose (no bent elbows), hold steady, press **A** on
   either controller.

   The audio is played via a regular `<audio>` element from
   `/audio/show_<pose>.mp3` rather than the headset's built-in
   `speechSynthesis` engine — Quest 3 Browser's TTS is unreliable in
   immersive-ar (gesture-locked, sometimes silent even when it
   reports success). MP3 playback travels the same plumbing as
   YouTube/Spotify, which the headset always plumbs through to the
   speakers. A `🔊 SPEAKING` badge pulses in the top-right of the
   overlay every time a prompt plays, so you can tell at-a-glance
   whether the client is actually firing audio (vs. the volume just
   being muted).

4. The script samples wrist positions for 1 second and gates the
   capture on **cluster spread** — the 80th-percentile distance of
   samples from the median wrist position must be ≤ 6 cm
   (`--spread-threshold-m`, default `0.06`). The reported wrist mean
   uses inliers only, so a stray Quest 3 tracking spike can't bias
   your calibration.

   :::{note}
   Frame-to-frame **velocity is no longer used as a gate** (it's
   logged as `mm/frame` jitter for diagnostics only). Quest 3
   inside-out tracking has 1–3 mm of phantom drift at arm extension,
   which a naïve `||p[i+1] - p[i]|| / dt` metric reports as 5–15 cm/s
   "velocity" even when you're holding perfectly still — so we
   measure the size of the cloud of samples instead.
   :::

5. Repeat for **Pose 2 — T-pose** (arms straight out sideways),
   **Pose 3 — Arms forward** (parallel, at shoulder height,
   *roughly shoulder-width apart — not wider*), and **Pose 4 —
   Namaste** (palms together at chest, forearms vertical).

   :::{note}
   The 4th pose ("namaste") was added in v2 of the calibration schema
   to anchor the y-axis fit at the body centerline. Without it, the
   3-pose v1 calibration was structurally biased: even when the
   operator brought their hands together right in front of their
   chest, the per-axis affine fit predicted robot wrists ~50 cm apart
   (because no calibration data point was near `op_y = 0`). The v2
   namaste pose puts the robot's reference wrists 0.2 cm apart at
   chest height, so the fit knows what "hands together" should map
   to.

   The arms-down robot reference also changed in v2: previously it
   used the SONIC stand-pose default `(0.2, 0.2, 0, -0.6, ...)` which
   has bent elbows, so even when the operator fully extended their
   arms downward, the IK kept the robot's elbows bent to lift the
   wrist up to the (incorrect) bent-arm target. v2 uses fully
   straight arms (every joint at `0`) as both the calibration
   reference *and* the IK null-space preferred posture, so full arm
   extension on the operator now produces full arm extension on the
   robot.
   :::

6. The script fits per-arm scale + translation (least squares,
   per-axis) and prints residuals. Per-pose residuals must be under
   the matching threshold (defaults: `arms_down`, `t_pose`,
   `arms_forward` ≤ **10 cm**; `namaste` ≤ **18 cm**) or the fit
   is rejected.

   :::{note}
   **Why namaste is looser**: the operator holds the controllers, so
   the controller-grip is offset ~5–7 cm from the actual palm
   position. A perfectly-executed namaste *with controllers in hand*
   will already have several cm of residual baked into the
   per-axis affine fit. The 18 cm gate exists so a clean
   capture isn't rejected for a structural offset that has nothing
   to do with how well the operator held the pose.

   The other three poses use 10 cm because the X2's per-axis affine
   model can't get below that on a typical human-vs-X2 anatomy
   mapping anyway — the pre-v2 5 cm gate was unattainable in
   practice and produced spurious rejections. If you want stricter
   gates for a particular setup, use `--reject-m 0.08`
   (uniform) or per-pose flags `--t-pose-reject-m 0.08`,
   `--namaste-reject-m 0.12`, etc.
   :::

   On a rejection the script does **not** crash. Instead it identifies
   the worst-contributing pose+arm (e.g. "T-pose right arm, residual
   10.5 cm"), speaks a coaching line through the headset (e.g.
   *"Recapture t pose (right arm): fit error was 11 cm. Stretch the
   right arm STRAIGHT SIDEWAYS at shoulder height; do NOT angle it
   forward."*), shows the same line in the dom-overlay, and lets the
   operator press **A** to recapture only that pose. The fit retries
   automatically. Up to `--max-fit-recaptures` (default `4`) retries
   are allowed before the script gives up.

7. On success, the YAML lands at the configured path. The teleop and
   recorder scripts then load it via `--calibration <path>`.

#### YAML contents (debugging-friendly)

```yaml
schema_version: 2
operator_id: stickbot
created_utc: "2026-05-09T22:30:11.512Z"
units: meters
poses:
  arms_down:    {left_wrist: [...], right_wrist: [...], samples: 50, ...}
  t_pose:       {...}
  arms_forward: {...}
  namaste:      {...}
robot_reference_q_rad:
  arms_down:    {left: [0,0,0,0,0,0,0],         right: [0,0,0,0,0,0,0]}        # v2: STRAIGHT arms (was bent stand pose)
  t_pose:       {left: [...],                    right: [...]}
  arms_forward: {left: [...],                    right: [...]}
  namaste:      {left: [-1.1,0,-1.2,-1.6,0,0,0], right: [-1.1,0,1.2,-1.6,0,0,0]}  # v2: hands at centerline
fit:
  left:  {scale: [sx,sy,sz], translation: [tx,ty,tz], residual_m: 0.018}
  right: {scale: [sx,sy,sz], translation: [tx,ty,tz], residual_m: 0.022}
```

The `poses` block + `robot_reference_q_rad` block let you re-fit the
calibration later if the math evolves, without re-recording the
operator.

#### When to recalibrate

* Every new operator (one YAML per person).
* The operator gives radically different wrist heights — e.g. wearing
  thicker gloves, standing on a riser.
* You see > 10 cm IK pos_err in the recorded debug NPZ during T-pose
  or arms-forward style motions.

#### Inline calibration during teleop

Both the kinematic teleop and the full recorder accept `--recalibrate`,
which runs the same 4-pose flow inline before teleop starts. Useful
the first time a new operator sits down.

#### Tuning the stability gate

The capture is gated on cluster spread. Defaults work for most
setups; reach for these only if calibration keeps rejecting your
poses or you want stricter quality.

| Flag | Default | Effect |
| ---- | ------- | ------ |
| `--spread-threshold-m` | `0.06` (6 cm) | Per-arm 80th-pct distance from the median sample, in meters. Lower = stricter, higher = more permissive. |
| `--sample-window-s` | `1.0` | Length of the sample window in seconds. Longer = more averaging, but operator has to hold the pose longer. |
| `--max-retries-per-pose` | `3` | How many times the operator can retry a pose before the script gives up. |
| `--reject-m` | unset | **Uniform** per-pose residual ceiling. When set, applies to every pose (overrides per-pose defaults). When unset, the per-pose defaults below are used instead. Use this for fast experimentation; per-pose flags below for production. |
| `--arms-down-reject-m` | `0.10` (10 cm) | Residual ceiling for the arms-down pose only. |
| `--t-pose-reject-m` | `0.10` (10 cm) | Residual ceiling for the T-pose only. |
| `--arms-forward-reject-m` | `0.10` (10 cm) | Residual ceiling for the arms-forward pose only. |
| `--namaste-reject-m` | `0.18` (18 cm) | Residual ceiling for the namaste pose only. **Looser than the others** to account for the ~5–7 cm controller-grip offset baked into every namaste capture (see the per-pose note in the workflow above). |

Common tuning recipes:

* **Shaky setup / bumping the headset / pets in the room**:
  `--spread-threshold-m 0.10 --sample-window-s 0.7`.
* **Lab-quality calibration**: `--spread-threshold-m 0.03`.
* **Operator can't get T-pose to register**: most common cause is
  holding only one controller, or one controller's batteries are dead
  and it dropped out. The script's per-attempt log line shows
  `dropouts skipped: N` — if `N > 20%` of the sample count, that's the
  culprit, not the threshold.

#### Why velocity isn't a gate

Earlier versions of this script used a `--vel-threshold-mps` flag that
tried to reject captures based on RMS frame-to-frame velocity. That
flag is now silently ignored (a deprecation warning prints if you pass
it) because the metric was fundamentally misleading at Quest 3's
mm-level precision. Quick math:

* Quest 3 reports controller pose at ~50 Hz → `dt = 20 ms`.
* Inside-out tracking has **1–3 mm of phantom drift per frame** at arm
  extension (T-pose, arms-forward), where the controller is at the
  periphery of the camera FOV and sometimes occluded.
* `||p[i+1] - p[i]|| / dt` for 1.5 mm jitter = `0.0015 / 0.02` =
  **7.5 cm/s** of reported "velocity" while the operator is perfectly
  still.

The cluster-spread metric we use instead measures the actual size of
the cloud of samples in 3D space — held wrists produce 1–3 cm clouds,
moving wrists produce 10+ cm clouds, and a single jitter spike adds
one outlier sample that barely moves the 80th-percentile distance.

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
| `--wrist-bypass {off,ik}` | `ik` | Override the C++ deploy's wrist target with the IK reference. See [Section 3.5](#35-wrist-bypass-honest-vr-wrist-tracking-on-top-of-sonic). |
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
| `--ik-damping 0.08`<br>`--ik-rotation-weight 0.0`<br>`--ik-per-tick-step-rad 0.30` | DLS IK tuning. v0 default `rotation_weight=0` runs **position-only IK** since wrist orientation is not calibrated. |
| `--calibration PATH` | YAML produced by `vr_operator_calibrate.py`. Required unless `--recalibrate` is passed. Defaults to `data/operator_calibrations/default.yaml`. |
| `--recalibrate` | Run the 4-pose calibration inline before recording starts. Use for the first session with a new operator. Writes the YAML to `--calibration` (or the operator-id default). |
| `--operator-id NAME` | Free-form operator label stamped into the calibration YAML when `--recalibrate` is set. |

### 3.5 Wrist bypass — honest VR wrist tracking on top of SONIC

```{admonition} TL;DR
:class: important
The default `--wrist-bypass ik` overwrites SONIC's `wrist_pitch` and
`wrist_roll` targets with the operator's IK reference. Keep it on for
**every VLA dataset recording**. Pass `--wrist-bypass off` only if you
are running a sim-to-real fidelity probe and want the policy's own
commands to reach every joint.
```

#### Why the bypass exists

Empirical analysis of `data/lerobot/x2_quest3_sonic_v2/data/chunk-000/episode_000001.parquet`
(recorded with the iter-25k checkpoint, then re-confirmed with iter-2k):

* `*_wrist_pitch` (`x2_action_scale = 0.0715`, ~8.8x smaller than the
  rest of the arm): correlation between commanded and executed is
  ~0.0; the executed angle sits in the -8 to -20 deg comfort band the
  policy learned to converge to, regardless of operator input.
* `*_wrist_roll` (asymmetric joint range): pinned at the +/-41 deg
  tight-side limit in 98-99% of frames.

`wrist_yaw` tracks fine (correlation ~0.8) and is left under SONIC.

Root cause is SONIC's training distribution (no diverse wrist motion)
combined with the smallmotor `x2_action_scale`, **not** an axis-sign
mismatch nor a deploy regression — see
`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp`
for the full audit trail. Re-fine-tuning SONIC on diverse wrist motion
is the long-term fix; the bypass is the unblocker for VLA dataset
recording in the meantime.

#### What the bypass does

When `--wrist-bypass ik` is set (and `--vla` so the deploy is
subscribed to the recorder's ZMQ pose feed), the C++ deploy
`OnControl()` step calls
[`ApplyWristBypass`](https://github.com/agibot/gear_sonic_deploy/blob/main/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp)
on `target_pos_mj` BEFORE the safety stack:

```text
target_pos_mj[20]  ← ref.joint_pos_mj[20]   # left_wrist_pitch
target_pos_mj[21]  ← ref.joint_pos_mj[21]   # left_wrist_roll
target_pos_mj[27]  ← ref.joint_pos_mj[27]   # right_wrist_pitch
target_pos_mj[28]  ← ref.joint_pos_mj[28]   # right_wrist_roll
```

Every other DOF (legs, waist, shoulders, elbows, `wrist_yaw`, head,
hand fingers) is still 100% under SONIC. The override sits **before**
soft-start blending, the `--max-target-dev` clamp, and the
tilt-watchdog force-to-default branch, so all existing safety
behaviour applies uniformly to the IK-driven targets.

The tokenizer observation is unchanged — SONIC still sees the IK
reference for all 31 DOFs as the future window. We only swap the
final per-tick PD target. SONIC therefore continues to drive a
self-consistent whole-body posture; only the wrist motors are
released from its attractor.

#### Operator-visible signals

The deploy logs two extra counters on the periodic `CONTROL tick=...`
status line so you can see the bypass firing:

```text
CONTROL tick=500 policy_t=10.00s alpha=1.00 grav_z=-1.00
        act_clip_ticks=0 max_pre_clip=2.31
        wrist_bypass_ticks=500 wrist_bypass_max_dev_rad=0.842
```

* `wrist_bypass_ticks` — number of OnControl steps where the override
  fired (== total ticks while a body-bearing ZMQ frame was available).
* `wrist_bypass_max_dev_rad` — running max of
  `|policy_target - ik_target|` across the bypassed DOFs. Large
  numbers (~0.5-1.0 rad) are normal; they're exactly why the bypass
  exists. Compare against an `--wrist-bypass off` baseline to quantify.

#### When to use which mode

| Scenario | Recommended setting |
| -------- | ------------------- |
| VR teleop / dataset recording for VLA fine-tune | `--wrist-bypass ik` (default) |
| Sim-to-real fidelity probe — want every joint to follow SONIC | `--wrist-bypass off` |
| Hand-only / no-wrist tasks where the policy is fine | either; default still safe |
| `--no-vla` kinematic loop (no SONIC) | flag is suppressed automatically (it would be a no-op) |

#### Validating a session

After a recording made with `--wrist-bypass ik`, regenerate the wrist
correlation table:

```bash
python /tmp/wrist_sign_probe.py \
    --parquet data/lerobot/<your_dataset>/data/chunk-000/episode_000000.parquet
```

Expect `corr(commanded, executed) > 0.9` and `alpha ~ 1.0` for both
`*_wrist_pitch` and `*_wrist_roll`, with no pinning at the joint-range
limits. Sanity-check by re-running the same trajectory under
`--wrist-bypass off` — you should reproduce the v2 baseline (~0.0
correlation on pitch, ~98% pinning on roll).

---

## 4. Operator workflow

### 4.1 Quest 3 controller cheat-sheet

| Button | Action |
| ------ | ------ |
| **A** | Toggle active arm tracking on / off. **Stateless** — calibration is loaded once at startup and applied every tick; A only gates whether the IK solver runs vs holds the last commanded q. Press once to start; press again to "park" the arms at their last pose without disconnecting. |
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
6. Squeeze **A** on either controller. The recorder logs
   `[A] arm tracking -> ACTIVE`. There is no posture you need to be in
   when you press A — the calibration was already loaded at startup.
   Press A again any time you want to "park" the arms at their last
   pose without disconnecting (`-> IDLE`).
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
| `action.motion_token` | `(64,)` | Reserved for offline FSQ-labeling pass. The live recorder writes zeros. |
| `action.body_q_mj` | `(N_body,)` | **Canonical training target.** Post-SONIC executed body q (MuJoCo joint order) — what the trained tracking policy actually achieved and what the MuJoCo viewer shows. In kinematic-only datasets this is just the commanded q (no policy in the loop). |
| `action.left_hand_joints` | `(10,)` | Canonical post-deploy left OmniHand joints (URDF-clipped). |
| `action.right_hand_joints` | `(10,)` | Canonical post-deploy right OmniHand joints. |
| `action.body_q_mj_pre_sonic` | `(N_body,)` | **Debug-only sibling.** Operator's X2 joint command sent on the wire to the deploy, *before* SONIC and MuJoCo physics. SONIC-recorded datasets only. |
| `action.left_hand_joints_pre_sonic` | `(10,)` | **Debug-only.** Pre-deploy left hand q (raw retargeter output). |
| `action.right_hand_joints_pre_sonic` | `(10,)` | **Debug-only.** Pre-deploy right hand q. |
| `action.sonic_correction_max_rad` | `(1,)` float32 | **Debug-only.** Per-frame `max_arms |body_q_mj − body_q_mj_pre_sonic|` summary scalar. |
| `task` | string | The session's `--task` value. |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | scalars | LeRobot bookkeeping (filled in by `Gr00tDataExporter`). |

```{admonition} Canonical action vs debug-only siblings
:class: note
The bare-canonical columns (`action.body_q_mj`, `action.left_hand_joints`,
`action.right_hand_joints`) are the *only* action columns surfaced as
training targets via `get_modality_config_x2_vla` (see
[gear_sonic/data/features_x2_vla.py](../../../gear_sonic/data/features_x2_vla.py)).
The `_pre_sonic` siblings + `action.sonic_correction_max_rad` live on
disk for retargeter / SONIC-correction analysis but the GR00T trainer
never pulls them into batches — they're invisible to the policy.
```

```{admonition} v0 → v1 schema migration
:class: warning
The v1 schema was introduced alongside the SONIC-loop recorder. It
**renames** the body action column and **flips semantics** on the
hand columns:

| | v0 datasets | v1 datasets |
|---|---|---|
| Body action column name | `action.commanded_body_q_mj` | `action.body_q_mj` |
| Body action semantics | Pre-SONIC operator command | **Post-SONIC executed q** (canonical); pre-SONIC preserved as `action.body_q_mj_pre_sonic` |
| Hand action column names | `action.left_hand_joints` / `action.right_hand_joints` | Same names |
| Hand action semantics | Pre-deploy operator retarget | **Post-deploy URDF-clipped q** (canonical); pre-deploy preserved as `_pre_sonic` siblings |
| `meta/dataset_format_version.json` | absent | `{"version": 1, "post_sonic_canonical": true}` (SONIC) or `false` (kinematic) |

**Do not mix v0 and v1 in a single training run.** The hand columns
have the same name but different semantics, so a mixed-version
training set silently teaches the model on inconsistent data. The
GR00T trainer can safely consume either schema *individually* via
the modality config; mixing happens at the dataset-aggregation step.

The replay tools (`replay_x2_kinematic.py`,
`inspect_sonic_correction.py`) auto-fall-back to the v0 column name
when the v1 column is absent, so old recordings still play back
without modification.
```

```{admonition} Discarded episodes leave no on-disk trace
:class: note
`Gr00tDataExporter` only writes a parquet shard on `save_episode()`.
The `Y`-button path simply drops the in-memory buffer, so the
on-disk dataset shape (parquet count, episode indices, video files)
is unchanged.
```

### 5.1 SONIC corrective-delta observability

Two complementary tools surface the gap between operator intent and
SONIC-stabilised output:

* **Live operator log** — once-per-second print when the trained
  policy pushes back on operator commands by more than
  `--sonic-correction-warn-rad` (default 0.05 rad ≈ 2.9°). Suppress
  with `--no-sonic-correction-log`. Helps the operator notice
  unreachable poses in real time.
* **Offline diagnostic** — `inspect_sonic_correction.py` runs
  against any saved SONIC episode and prints a per-arm-joint
  `|delta_q|` summary plus a 4-panel time-series PNG under
  `<dataset>/debug/sonic_correction_ep<N>.png`. Frames where
  `|delta_q|max > 0.15 rad` (~8.6°) are flagged as candidate
  "infeasible" events worth reviewing in `replay_x2_kinematic.py`.

Both signals derive from the same `|action.body_q_mj − action.body_q_mj_pre_sonic|`
arm-joint subset (the lower body is pinned to the stand pose so its
delta is uninteresting noise).

```bash
.venv/bin/python -m gear_sonic.scripts.inspect_sonic_correction \
    --dataset x2_quest3_sonic_v1 --episode 0
```

The diagnostic also recognises legacy v0 datasets (no `_pre_sonic`
columns) and falls back to a per-joint range stat on the commanded
trajectory itself, since there is nothing to compare against.

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

The recorder writes one MP4 per episode per camera under
`videos/chunk-000/observation.images.<camera>/`. Open the MP4 in any
video player; that's exactly the frame stream the trained policy will
see at deploy time (the same camera, same resolution, same FPS).

Quick CLI peek:

```bash
ls data/lerobot/x2_quest3_v0/videos/chunk-000/observation.images.ego_view/
mpv data/lerobot/x2_quest3_v0/videos/chunk-000/observation.images.ego_view/episode_000000.mp4
```

For a full multi-camera replay with joint plots and state timelines,
use the dedicated Rerun viewer instead -- see § 6.2.1.

#### 6.2.1 Rerun viewer (multi-camera + scalar plots)

```bash
./gear_sonic/scripts/view_x2_recorded_dataset.sh \
    --dataset x2_grab_a_drink --episode 6
```

This spawns the system `rerun-cli 0.31.4` viewer and streams every
camera (ego_view, head_front, stereo_left, stereo_right when
recorded) plus the per-joint scalar plots in a single window. The
viewer uses native H.264 decode on the GPU and a columnar rerun
log path, so even an 8 K-frame, 4-camera episode is interactive in
< 2 s with no broken-pipe / "Dropping messages" warnings.

##### Why this needs its own venv

This is the most common foot-gun in this repo right now, so it's
worth understanding once:

| Stack | numpy | pinocchio (`pin`) | rerun-sdk | What it can do |
|---|---|---|---|---|
| `.venv/` (planner) | 1.26.4 | 2.7.0 (cmeel wheel) | 0.21.0 | Runs the kinematic planner. **Cannot view recorded datasets** (0.21 .rrd files are rejected by the conda 0.31.4 viewer with `Codec error: Invalid encoding options`). |
| `.venv-viewer/` (viewer) | ≥2 | not installed | 0.31.4 | Matches the conda `rerun-cli 0.31.4` on the user's PATH; decodes H.264 natively. **Cannot import `pinocchio`** because pin 2.7's pre-built wheel needs numpy<2. |

Upgrading rerun-sdk in `.venv/` would pull `numpy>=2`, which immediately
breaks `pin.buildSampleModelHumanoid` with:

```
ImportError: A module that was compiled using NumPy 1.x cannot be run
in NumPy 2.x ... downgrade to 'numpy<2' or upgrade the affected module.
```

Downgrading the conda viewer to 0.21 would let `.venv/` talk to it,
but 0.21's native viewer only decodes **AV1** (`Only MP4 containers
with AV1 are generally supported`, per `rr.AssetVideo`'s own
docstring) -- our recorder writes H.264, so all camera panes would
render blank.

The two-venv setup avoids both traps: keep the planner pinned, ship
a thin (~600 MB) parallel venv just for the viewer, dispatch via a
wrapper script. **The wrapper handles the interpreter selection;
operators never have to know which venv to activate.**

##### Installing or repairing the viewer venv

If `.venv-viewer/` is missing or broken, the wrapper prints exit
code 2 with recovery instructions. The idempotent installer is:

```bash
bash install_scripts/install_viewer.sh
```

Pinned versions live in [`requirements-viewer.txt`](../../../../requirements-viewer.txt)
at the repo root. The installer also verifies the columnar rerun
API surface (`TimeColumn`, `Scalars.columns`, etc.) at install
time so a future SDK bump that drops one of those symbols fails
at `bash install_scripts/install_viewer.sh` instead of mid-replay.

##### Viewer CLI flags

```bash
./gear_sonic/scripts/view_x2_recorded_dataset.sh --help
```

Useful selectors:

| Flag | Effect |
|---|---|
| `--dataset NAME` | Resolve `data/lerobot/NAME/` automatically. |
| `--root PATH` | Use an absolute dataset path instead. |
| `--episode N` | Required; the on-disk `episode_index`. Read off the recorder log (`on-disk episode_index=N`) or from `ls data/lerobot/.../episode_NNNNNN.parquet`. |
| `--save FILE.rrd` | Dump a self-contained recording instead of spawning the viewer. Replay later with `rerun FILE.rrd`. |
| `--skip-scalars` | Videos only -- fastest cold load when you just want a visual eyeball pass. |
| `--scalar-decimate N` | Subsample scalars to every Nth frame; useful for > 30 K-frame sessions where chart panes get sluggish. |
| `--max-scalar-dims K` | Per-column cap on vector expansion. Default 64 covers `body_q_mj` (30) + omnihand (10 + 10) + projected gravity (3). Set to 0 to skip scalars (same as `--skip-scalars`). |

### 6.3 Replay a recorded episode through the live deploy

This is the strongest acceptance gate the dataset path has: it proves
the saved `action.body_q_mj` (plus the OmniHand finger columns) is
enough to re-create the on-robot trajectory end-to-end, with no
operator and no policy in the loop. The full sim or real-robot stack
is one command:

```bash
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0
```

The wrapper brings up `deploy_x2.sh sim --vla --sim-with-omnihand
--sim-viewer` on localhost, waits for it to log `Launching ...`,
spawns the replay client against it, and tears everything down in
reverse order on Ctrl-C (replay gets `SIGINT` first so its
`hold_on_exit` ramp-down completes against a still-alive deploy
before the sim container goes away). See the
[2026-06-22 milestone](../user_guide/milestones/2026-06-22_dataset_replay_v5_wire.md)
for the topology diagram and the v5 wire contract this depends on.

```{admonition} Why the **body** moves now (and didn't before)
:class: important
The C++ deploy on PC2 (`agi_x2_deploy_onnx_ref`) **ignores**
`motion_token` on the wire and re-tokenizes the trajectory each tick
from the v5 future window (`joint_pos_mj_future`, 9 slots × 31 DOFs at
0.1 s spacing). The pre-2026-06-22 replay tool published only the v4
envelope; the deploy back-filled the future window with the trained
`default_angles` stand pose and the body held `idle_stand` while only
the OmniHand fingers tracked (different code path, no future-window
dependency). The new replay payload includes the v5 future-window
fields — sourced from `body_q_mj[f+5, f+10, ..., f+45]` and tail-tiled
past episode end — and the body now tracks the recording.
```

#### Three deploy modes

| Mode | Flag | What happens |
|---|---|---|
| **Sim (default)** | (none) | Wrapper spawns `deploy_x2.sh sim --vla --sim-profile handoff --sim-with-omnihand --sim-viewer --wrist-bypass ik`. Safe for first-pass validation. |
| **External deploy** | `--no-deploy` | Skips the spawn; assume you brought up a deploy in another shell. Useful for inspecting the wire with a separate `pose` SUB. |
| **Real robot** | `--pc2-host <PC2_IP>` | Skips the spawn; assume `x2_pc2_daemons.sh start` is already running on PC2. The replayer's PUB still binds locally (`*:5556`); PC2 connects out. |

#### Useful flags (all forward to `replay_x2_dataset` 1-for-1)

| Flag | Default | Effect |
|---|---|---|
| `--dataset <name-or-path>` | — | Short name under `data/lerobot/` or absolute path. **Required.** |
| `--episode <int>` | `0` | Zero-indexed episode within the dataset. |
| `--rate <Hz>` | dataset native fps | Override the publish rate. The v5 future-window stride is always derived from the dataset's NATIVE fps, so `--rate` only changes the publish cadence, not the per-slot lookahead. |
| `--rate-scale <float>` | `1.0` | Multiplier on rate. `0.5` = half-speed wall-clock playback. The future window still represents 0.1 s per slot of the source's recorded dynamics (correct for the deploy's tokenizer). |
| `--loop` | off | Loop the episode indefinitely. |
| `--countdown <s>` | `3.0` | Seconds of "hold frame 0" warm-up before the trajectory starts. Gives the deploy's handoff ramp time to transition from default angles to the recording's starting pose. |
| `--hold-on-exit <s>` | `0.5` | Seconds of "hold the last frame" before exit. SONIC's safety stack decays PD gains in ~200 ms; 0.5 s is the soft-stop window. |
| `--with-rerun` | off | Also spawn the [recorded-camera viewer](#62-rerun-viewer-for-the-recorded-dataset-view_x2_recorded_datasetsh) for the same `--dataset` / `--episode`. The rerun GUI process is spawned by `rr.init(spawn=True)` and **outlives this wrapper** so you can scrub the recording after the live run ends. Requires the dedicated `.venv-viewer/`; see `install_scripts/install_viewer.sh`. |
| `--no-sim-viewer` | off | Headless variant (no MuJoCo window). Useful for CI / capture sweeps. |
| `--duration <s>` | `0` | Wall-clock cap; `0` = run until the replay's own end-of-episode signal. |
| `--cleanup-only` | — | Free `:5556`, sweep stale `x2sim` docker containers from a crashed run, exit. |

#### Example: sim + recorded cameras side-by-side

```bash
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 --with-rerun
```

The MuJoCo viewer shows the live deploy executing the recorded
`body_q_mj`; the rerun GUI shows the operator's original 4 camera
streams (`ego_view`, `head_front`, `stereo_left`, `stereo_right`)
plus the per-joint scalar timeline and 3-D wrist-FK trace for the
same episode. Eyeball both panes side-by-side to confirm the live
playback matches what you expected.

#### Example: real-robot first pass (half-speed, e-stop in reach)

```bash
# On PC2 (separate shell):
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start \
    --attach --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

# On the laptop:
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 \
    --pc2-host 192.168.86.32 --rate-scale 0.5 --with-rerun
```

```{admonition} Safety
:class: warning
* Object position on the table MUST match the recording. The replayer
  has no perception loop; it commands the same wrist trajectory
  regardless of where the soda can / apple actually is.
* The first real-robot run should always use `--rate-scale 0.5` and an
  operator hand on the e-stop. After one clean half-speed pass, drop
  the scale.
* The recorded `action.body_q_mj` is what the operator **commanded**,
  not what the robot **achieved**. SONIC's tracking-policy will still
  apply its safety filters; expect ~1–2 mm of arm tracking error per
  joint relative to the recording.
```

#### What `motion_token` is doing (or rather, isn't)

The replay still copies `action.motion_token` from the parquet into
the wire envelope, but the C++ deploy **ignores it** — it re-tokenizes
the trajectory each tick from `joint_pos_mj_future`. The token field
on the wire is a debug echo only. The actual lever that drives the
body is the v5 future-window; if the body isn't moving but the fingers
are, you're looking at a v4-only envelope (likely because the wrapper
spawned an outdated replay client that hasn't been rebuilt — see
`tests/test_replay_x2_dataset_future_window.py::test_payload_packs_and_unpacks_byte_for_byte`,
the byte-roundtrip pin that gates this contract).

### 6.4 Kinematic MuJoCo replay (`replay_x2_kinematic.py`)

For everything except SONIC closed-loop verification, the
purpose-built replay CLI is faster and more legible than the snippet
above. It opens a passive MuJoCo viewer and faithfully replays the
recorded `action.body_q_mj` plus `action.left_hand_joints` /
`action.right_hand_joints` straight out of the parquet (auto-falling-
back to the legacy `action.commanded_body_q_mj` for pre-v1 datasets) —
no Quest 3, no IK, no policy, no ZMQ.

```bash
.venv/bin/python -m gear_sonic.scripts.replay_x2_kinematic \
    --dataset x2_quest3_kinematic_v4 \
    --episode 0
```

The first positional-style flag accepts either a short name (resolved
under `data/lerobot/<name>/`) or any absolute / relative path to a
LeRobot v2.1 dataset root. Useful options:

| Flag | Default | Effect |
|------|---------|--------|
| `--dataset <name-or-path>` | — | Dataset root (name or path). Required unless `--parquet` is set. |
| `--episode <int>` | — | Zero-indexed episode number. Required unless `--parquet` is set. |
| `--parquet <path>` | — | Direct path to a parquet file. Bypasses `--dataset` / `--episode` resolution; useful for replaying re-derived variants (e.g. `episode_000000_hand_range_calibrated.parquet`) without renaming the canonical file. |
| `--robot {x2,g1}` | `x2` | Embodiment dispatched through the registry. `g1` is a stub today. |
| `--rate <Hz>` | `50.0` | Playback rate. Match the recorder's 50 Hz to see real-time motion. |
| `--start-frame <int>` | `0` | First frame to play (inclusive). |
| `--end-frame <int>` | `-1` | Last frame, exclusive. `-1` means end-of-episode. |
| `--loop` | off | Restart from `--start-frame` after `--end-frame`. |
| `--with-omnihand` / `--no-omnihand` | on | Augment the MJCF with OmniHand and apply hand columns. |
| `--quiet` | off | Suppress per-second progress logs. |

**What you see:** the exact joint trajectory the recorder wrote to
disk. If the recording captured weird arm positions or twitchy hand
commands, this viewer will faithfully reproduce them. Use it to
diagnose retargeting and recording issues without strapping the
headset back on.

**What you do NOT see:** SONIC tracking-policy output. The commanded
`body_q_mj` is what the operator told the lower body to do, not what
a stabilised deploy would have actually executed. For SONIC-loop
replay, see recipe 6.3 above (and the planned `replay_x2_sonic.py`).

**Tips:**

- The viewer never modifies the dataset. Quitting (close window or
  `Ctrl-C`) leaves disk untouched.
- Pair this with [`replay_recorded_dataset.py`](../../../gear_sonic/scripts/replay_recorded_dataset.py)
  to first regenerate the parquet with new retargeting code, then
  point `replay_x2_kinematic` at the new parquet to compare side-by-
  side.
- For a subset playback, `--start-frame` / `--end-frame` slice the
  episode; `--loop` keeps it cycling.

### 6.5 Architecture: teleop / record / replay matrix

The X2 stack treats teleop, record, and replay as three distinct
verbs that each have a kinematic and a SONIC backend. This is the
current state of the matrix:

| | **kinematic backend** | **SONIC backend** |
|---|---|---|
| **teleop** (no recording) | [`teleop_x2_kinematic.py`](../../../gear_sonic/scripts/teleop_x2_kinematic.py) | [`record_x2_dataset.py --teleop-only`](../../../gear_sonic/scripts/record_x2_dataset.py) |
| **record** (with dataset writes) | [`teleop_x2_kinematic.py --output-dir <path>`](../../../gear_sonic/scripts/teleop_x2_kinematic.py) | [`record_x2_dataset.py`](../../../gear_sonic/scripts/record_x2_dataset.py) |
| **replay** (no operator) | [`replay_x2_kinematic.py`](../../../gear_sonic/scripts/replay_x2_kinematic.py) (this section) | [`run_x2_replay_stack.sh`](../../../gear_sonic/scripts/run_x2_replay_stack.sh) → [`replay_x2_dataset.py`](../../../gear_sonic/scripts/replay_x2_dataset.py) ([§6.3](#63-replay-a-recorded-episode-through-the-live-deploy)) |

Two scripts double-duty as both `teleop` and `record` today via a
mode flag (`--output-dir` for the kinematic side, `--teleop-only` for
the SONIC side). Splitting them into single-purpose files is on the
v1 backlog; the public surface (CLI flags) will keep working through
that refactor.

The **kinematic** column has zero deploy / SONIC / ZMQ in the loop —
just MuJoCo `mj_forward` writes from a passive viewer. The **SONIC**
column runs the C++ deploy plus the SONIC tracking policy plus the
motion-token publisher; expect a 30 s startup per session.

### 6.6 Embodiment registry (`gear_sonic.utils.embodiment`)

Both `teleop_x2_kinematic.py` and `replay_x2_kinematic.py` accept
`--robot x2|g1` (default `x2`) and dispatch through a small
registry under [`gear_sonic/utils/embodiment/`](../../../gear_sonic/utils/embodiment/).
Today only `x2` resolves to a real
[`EmbodimentConfig`](../../../gear_sonic/utils/embodiment/config.py);
[`g1`](../../../gear_sonic/utils/embodiment/g1.py) registers a stub
whose factories raise `NotImplementedError` so misuse fails fast at
the point a real model would be needed.

**Filename convention** for new scripts: `<verb>_<robot>_<backend>.py`.
The X2 entries follow it (`teleop_x2_kinematic.py`,
`replay_x2_kinematic.py`); `record_x2_dataset.py` keeps its legacy
name to avoid breaking the bash wrapper but is the canonical
"X2 + SONIC + record" path.

**Adding a new embodiment** (e.g. when G1 lands):

1. Edit [`gear_sonic/utils/embodiment/g1.py`](../../../gear_sonic/utils/embodiment/g1.py)
   and replace the `_g1_build_kinematic_model` /
   `_g1_apply_dexhand_fn` stubs with real implementations
   (MJCF loader analogous to X2's `build_model_with_camera`, hand
   applier analogous to `apply_active_hand_qpos`).
2. Update the constants (`_G1_NUM_BODY_DOFS`, pelvis pose, stand
   pose) to match the real G1 URDF / MJCF.
3. Add `tests/test_embodiment_registry.py::test_g1_*` cases that
   exercise the new factories with a real MuJoCo load.

No CLI changes are needed: every existing `--robot g1` invocation
will start working as soon as the stub is replaced.

---

## 7. Acceptance gates and unit tests

Run these before merging any change to the recorder:

```bash
.venv/bin/python -m pytest tests/test_x2_arm_ik_smoke.py -x
.venv/bin/python -m pytest tests/test_record_x2_dataset_schema.py -x
.venv/bin/python -m pytest tests/test_operator_calibration.py -x
.venv/bin/python -m pytest tests/test_vr_arm_teleop_v2_smoke.py -x
.venv/bin/python -m pytest tests/test_teleop_x2_kinematic_smoke.py -x
.venv/bin/python -m pytest tests/test_embodiment_registry.py -x
.venv/bin/python -m pytest tests/test_x2_kinematic_view.py -x
.venv/bin/python -m pytest tests/test_replay_x2_kinematic.py -x
.venv/bin/python -m pytest tests/test_replay_x2_dataset_future_window.py -x
```

* [`tests/test_x2_arm_ik_smoke.py`](../../../tests/test_x2_arm_ik_smoke.py)
  pins the DLS IK round-trip (FK → target → IK → FK should match
  within sub-mm) and the joint-limit clamp invariant for both the
  left and right 7-DOF arm chains.
* [`tests/test_record_x2_dataset_schema.py`](../../../tests/test_record_x2_dataset_schema.py)
  pins the hand-retargeter shapes, the LeRobot feature schema, and
  the (legacy) `VRArmTeleop` step's neutral-pose fallback.
* [`tests/test_operator_calibration.py`](../../../tests/test_operator_calibration.py)
  pins the per-axis fit math, residual-reject threshold, YAML
  round-trip, and head-yaw-frame transform.
* [`tests/test_vr_arm_teleop_v2_smoke.py`](../../../tests/test_vr_arm_teleop_v2_smoke.py)
  is the regression suite for the head-yaw-relative wrist mapping —
  including the test that pinned the "operator rotates 90 deg in
  place ⇒ robot wrist target stays put" invariant the engage-anchor
  solver violated.
* [`tests/test_teleop_x2_kinematic_smoke.py`](../../../tests/test_teleop_x2_kinematic_smoke.py)
  pins `Gr00tDataExporter.create()` kwargs, the half-init preflight
  cleanup paths, and the missing-calibration error message.
* [`tests/test_embodiment_registry.py`](../../../tests/test_embodiment_registry.py)
  pins the `--robot x2|g1` dispatch contract: registry lookup,
  `EmbodimentConfig` shape validation, and the G1 stub's
  `NotImplementedError` failure mode.
* [`tests/test_x2_kinematic_view.py`](../../../tests/test_x2_kinematic_view.py)
  pins the lifted `build_kinematic_model` / `set_kinematic_pose`
  helpers shared by live teleop and offline replay.
* [`tests/test_replay_x2_kinematic.py`](../../../tests/test_replay_x2_kinematic.py)
  pins the kinematic-replay CLI's pure helpers (arg parsing, dataset
  resolution, chunk-path math, parquet schema validation) without
  launching the MuJoCo viewer.
* [`tests/test_replay_x2_dataset_future_window.py`](../../../tests/test_replay_x2_dataset_future_window.py)
  pins the SONIC-loop replay's v5 future-window helper + payload
  schema (see [§6.3](#63-replay-a-recorded-episode-through-the-live-deploy)
  and the [2026-06-22 milestone](../user_guide/milestones/2026-06-22_dataset_replay_v5_wire.md)).
  The critical test (`test_payload_packs_and_unpacks_byte_for_byte`)
  asserts that `pack_pose_message(..., version=4)` →
  `unpack_message()` roundtrips every byte of every v5 field, so the
  C++ deploy on PC2 is guaranteed to promote the replay's frames to
  v5 mode (= the body actually moves, not just the fingers).

---

## 8. v0 limitations

### Torso tilt is not captured

The X2 has a 3-DOF waist (`waist_pitch`, `waist_roll`, `waist_yaw`),
but v0 holds it locked at neutral for every recorded frame. With only
3 tracked rigid bodies (head + 2 wrists), it's mathematically
impossible to disambiguate "operator bent forward at the waist" from
"operator looked down with arms reaching" — both produce the same
head-displacement + wrist-displacement signature.

If you need to capture torso tilt for a task (e.g. picking from the
floor, leaning over a table), the upgrade options are:

1. **Joystick-driven waist control** — map the right thumbstick to
   `waist_pitch` / `waist_roll`. Crude but immediate. Already streamed
   over the WS as `axes.{rx, ry}`.
2. **Meta Movement SDK / WebXR body tracking** — Quest 3 estimates
   full-body keypoints from the headset + controllers via built-in
   upper-body IK. Natural, no extra hardware, ~250 LOC of WebXR
   work, reliability varies.
3. **Vive Tracker on the chest** — best 6-DOF chest pose, but adds
   SteamVR base-station setup time per session.

For the same reason, **wrist orientation is also not calibrated** in
v0 (`--ik-rotation-weight 0` by default). The IK runs position-only.
Adding wrist orientation requires fitting an operator-to-robot
rotation map alongside the position map; tracked as a v1 follow-up.

### SONIC pins the wrist DOFs — and why we bypass them in C++

```{admonition} Status
:class: note
Landed May 10, 2026. Default `ik` mode in
`record_x2_dataset.sh`. Quick reference + operator workflow lives
in [Section 3.5](#35-wrist-bypass-honest-vr-wrist-tracking-on-top-of-sonic);
this section is the longer-form post-mortem so the next person
investigating "why don't my wrists move?" doesn't have to redo the
diagnostic loop.
```

#### The symptom

When we first ran VR teleop with the SONIC 25k checkpoint in the loop
(`x2_quest3_sonic_v2`), the operator could move shoulders / elbows /
wrist_yaw freely on the robot, but `wrist_pitch` and `wrist_roll`
**did not respond**. The IK output looked correct in the recorder
debug log; the executed angles in the parquet did not.

#### What the data said

Two diagnostic scripts on `data/lerobot/x2_quest3_sonic_v2/data/chunk-000/episode_000001.parquet`
([`/tmp/wrist_sign_probe.py`](/tmp/wrist_sign_probe.py),
[`/tmp/wrist_diagnostic_plot.py`](/tmp/wrist_diagnostic_plot.py),
both pure-pandas, run against any v1+ recording):

| DOF | `corr(commanded, executed)` | `alpha` (linear fit slope) | Pinning |
| --- | --------------------------- | -------------------------- | ------- |
| `*_wrist_yaw`   | ~0.8 | ~1.0 | none |
| `*_wrist_pitch` | ~0.0 | undefined (executed flat at -8 to -20°) | n/a |
| `*_wrist_roll`  | ~0.0 | undefined | **98–99% of frames at the asymmetric tight-side limit (±41°)** |

Re-running with the **iter-2k** checkpoint reproduced the same numbers,
confirming this is **not** a late-training regression — the policy
has been pinning these DOFs since iteration 2000. The other 27 DOFs
(legs, waist, shoulders, elbows, `wrist_yaw`, head) all track the IK
reference cleanly.

#### What we ruled out

* **Axis sign / convention mismatch.** Walked through
  [`gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml`](../../../gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml)
  and the URDF on both sides; joint axes match between deploy and
  recorder. Swapping the `corr+` / `corr-` columns in
  `wrist_sign_probe.py` produces mirror-image numbers (both ~0), so
  there's no hidden sign flip — the executed signal just doesn't
  move with the input regardless of polarity.
* **`default_angles` mismatch between deploy and recorder.** Identical
  per [`gear_sonic_deploy/.../include/policy_parameters.hpp`](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp);
  both pitches and both rolls have `default_angles == 0.0`.
* **Joint-range clipping in the recorder.** Wrists in the parquet
  reach ±40°+ during operator sweeps. They get clipped at the joint
  range *after* SONIC, not before.

#### What's actually wrong

Two compounding factors in the SONIC training distribution:

1. **`x2_action_scale` is ~8.8× smaller for wrist DOFs.** From
   [`policy_parameters.hpp`](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/policy_parameters.hpp):

   ```cpp
   x2_action_scale[20] = 0.0715;   // left_wrist_pitch
   x2_action_scale[21] = 0.0715;   // left_wrist_roll
   x2_action_scale[27] = 0.0715;   // right_wrist_pitch
   x2_action_scale[28] = 0.0715;   // right_wrist_roll
   // vs ~0.42 for shoulders / elbows / wrist_yaw
   ```

   So even at maximum policy authority (action_il = ±action_clip ≈
   ±20), the wrist target can only deviate ~1.4 rad from
   `default_angles[mj]`. The other arm joints get ~8.4 rad of authority.

2. **The training motion library has almost no diverse wrist motion.**
   The X2M2 PKL clips that fed SONIC training are dominated by
   stand-still / locomotion / standing-gestures — none of which
   exercise wrist pitch or roll through their full range. With low
   per-tick authority *and* a thin training distribution, the policy
   learned to converge to a single comfort pose and ignore the
   reference window for those four DOFs. The asymmetric `wrist_roll`
   joint range (`(-41°, +57°)` left, `(-57°, +41°)` right) means that
   single comfort pose lands directly on the tight-side limit, which
   is why the executed angle reads as "pinned at the limit" rather
   than "stuck at zero."

`wrist_yaw` survives both of these — its `action_scale` is the same
0.42 as the rest of the arm, *and* the training motions exercise it
because it tracks head turns through the kinematic tree.

#### Solution alternatives we considered

| Option | Idea | Why we didn't pick it |
| ------ | ---- | --------------------- |
| **A. Kinematic-only mode** | Run VR teleop without SONIC; pin lower body to stand pose by hand. | Loses every other benefit of SONIC (joint-limit feasibility, collision avoidance, balance). Also disconnects the data from the deployment loop, so VLA fine-tunes on a different distribution from inference. |
| **B. Fine-tune SONIC on diverse wrist motion** | Augment the training motion library with full wrist sweeps; bump `x2_action_scale` for wrist DOFs; retrain. | Days of compute. Right thing to do long-term but blocks VLA recording today. |
| **C. Surgical bypass in the C++ deploy** (chosen) | Inside `OnControl()`, after SONIC produces `target_pos_mj` but **before** the safety stack, overwrite the four wrist DOF targets with the IK reference from the ZMQ pose feed. | Conceptually simple, ~30 lines of C++, no model retraining. SONIC keeps full authority over every other joint and over the tokenizer observation; the bypass only swaps the final per-tick PD targets for `{20, 21, 27, 28}`. Stability risk is low because wrist mass is negligible compared to the upper arm. |

#### Why not "spoof the proprioception too"?

A second variant we considered (Option 2 in the planning doc): not
just override `target_pos_mj` but *also* lie to the policy about
where the wrists are by replacing `joint_pos_mj` in the proprio
observation. That would keep the policy's internal state self-
consistent — it would never see a wrist position that disagrees
with what it commanded.

We rejected this because:

* **Honest observation matters more than commanded-vs-executed
  consistency** for the policy's body planning. SONIC reasons about
  the upper-arm posture from the *full* joint vector; if we lied
  about the wrist, the elbow / shoulder controller could plan
  motions that assume a wrist orientation that doesn't exist.
* **Wrist mass is tiny.** Even a 30° wrist deviation from the policy
  expectation contributes negligible torque to the rest of the body.
  Empirically: in the v3 wrist-bypass session, gravity z-component
  stays at -1.00 ± 0.005 throughout (no torso tilt response from the
  wrist disagreement).
* **Domain randomization in training already covers it.** SONIC was
  trained with joint-position perturbations on the order of ±5°
  (kinematics noise) and physics-side mass scaling, so a few tens of
  degrees of "I commanded one thing but observed another" on a low-
  mass DOF is well inside its robustness envelope.

If a future task surfaces torso twitchiness when the bypass fires,
revisit this trade-off. Until then, simple-bypass wins on
cognitive load.

#### Where it lives

| File | Role |
| ---- | ---- |
| [`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp`](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp) | The override loop and the `kBypassedWristMjDofs = {20, 21, 27, 28}` constant. Inline so the unit test can call it without ROS / ONNX. |
| [`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp`](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp) | `CliArgs::WristBypass` enum, `--wrist-bypass {off,ik}` parser, the `OnControl()` call site, and the periodic `wrist_bypass_ticks` / `wrist_bypass_max_dev_rad` log line. |
| [`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/test/test_obs_builder.cpp`](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/test/test_obs_builder.cpp) | `TestWristBypassOverridesExactly4Slots` + `TestWristBypassZeroDeltaWhenAligned`. |
| [`gear_sonic_deploy/deploy_x2.sh`](../../../gear_sonic_deploy/deploy_x2.sh) | `--wrist-bypass` passthrough plumbing. |
| [`gear_sonic/scripts/record_x2_dataset.sh`](../../../gear_sonic/scripts/record_x2_dataset.sh) | Default `--wrist-bypass ik` for VLA dataset recording; suppressed automatically under `--no-vla` (would be a no-op). |

### Hand retargeting journey log (v0 → v0.7)

The next several subsections (`OmniHand vs human hand` through
`Open: non-thumb fingertip-to-thumb touch`) describe the chain of
problems we hit moving the OmniHand mapping from "vendored upstream
constants + uniform trigger" to "per-finger XRHand retargeting
that survives a thumb-fingertip touch gesture, works inside a
robocasa kitchen scene, and stays proportional in hand-tracking
mode". Each subsection follows the same template — what the
operator saw, what the data showed, what we changed, and what's
still imperfect — so the log doubles as a debugging crib sheet
for future hand work.

The relevant code is in
[`gear_sonic/utils/teleop/x2_hand_retarget.py`](../../../gear_sonic/utils/teleop/x2_hand_retarget.py)
on the Python side and in
[`gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html`](../../../gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html)
(`computeHandCurls` / `computeThumbOpposition`) on the WebXR side.
Self-contained session wrap-ups of the major iterations live at:

* [`milestones/2026-05-10_omnihand_finger_tuning.md`](../user_guide/milestones/2026-05-10_omnihand_finger_tuning.md)
  — May 10 thumb opposition / anchor expansion / `thumb_mcp` join.
* [`milestones/2026-05-12_finger_signal_filter.md`](../user_guide/milestones/2026-05-12_finger_signal_filter.md)
  — May 12 EMA + deadband-hold signal smoothing.
* [`milestones/2026-05-13_robocasa_finger_fixes.md`](../user_guide/milestones/2026-05-13_robocasa_finger_fixes.md)
  — May 13 robocasa unblock: pre-OmniHand wrist mesh disable,
  hand-source filter reset on mode switches, smooth-proportional
  compensation defaults (kills the "fingers are bang-bang in
  hand-tracking mode" report), and natural resting thumb pose.

### OmniHand has fewer flexion DOFs than the human hand

This is the foundational mismatch every retargeting decision in the
following subsections has to work around: **a human hand has more
finger joints than the X2 OmniHand can move**. The retargeting
collapses 3 cascaded human knuckles per finger onto a single
robot DOF that drives 2 mimic-coupled segments. Some of that lost
geometric range is recoverable (anchor expansion, opposition
fold-ins), some isn't.

```text
Operator finger (Quest 3 XRHand 25-joint chain, non-thumb):
   metacarpal ── proximal ── intermediate ── distal ── tip
                  ▲           ▲              ▲
                MCP knuckle  PIP knuckle    DIP knuckle
                ~80° max     ~80° max       ~80° max
   total bend at full curl ≈ 240° = 1.33 π rad

OmniHand finger (URDF, non-thumb):
   *_abad ── *_pip ──────────── *_dip (mimic = 1.097 × pip) ── tip
              ▲                  ▲
            active flexion     coupled passive flexion
            0..90°             0..~99° (when pip = 90°)
   total bend at hardware-max pip ≈ 189° = 1.05 π rad
```

| | Operator (Quest 3) | OmniHand (URDF) |
|---|---|---|
| **Thumb knuckles** | 2 in chain (MCP, IP) — CMC at base is *not* in the XRHand chain | 1 active flexion DOF (`thumb_mcp`) drives 2 mimic-coupled segments (`thumb_pip = 1.33 × mcp`, `thumb_dip = 1.30 × mcp`) → effective multiplier ~3.6 |
| **Index / ring / pinky** | 3 cascaded knuckles (MCP + PIP + DIP), each ~80° | 1 active flexion (`*_pip` 0..90°) + 1 mimic (`*_dip = 1.097 × pip`) — effective multiplier ~2.1 |
| **Middle** | Same as above | 1 active flexion (`middle_pip`) + 1 mimic (`middle_dip`); `middle_abad` is **fixed in the URDF** (no lateral spread) |
| **Lateral spread** | Real per-knuckle XRHand metacarpal-to-metacarpal angles | `*_abad` motors with very narrow ~5–12° hardware ranges; v0 lerps them on the same curl signal as the matching `_pip` |

The thumb has a comfortable headroom (one input angle covers
3.6× the angular span at the tip, very close to a fully-curled
human thumb). The non-thumb fingers do not — at hardware-max pip
the robot fingertip lands somewhere between where the operator's
PIP and DIP would, never at the operator's tip. This is the
underlying cause of the open issue documented at the bottom of
this section.

Joint hardware ranges (per
[`gear_sonic/data/assets/robot_description/omnihand/omnihand_*.urdf`](../../../gear_sonic/data/assets/robot_description/omnihand/), live in `HAND_FINGER_NAMES_PER_SIDE` order):

| # | Active joint    | Range (°)            | OPEN (°)        | CLOSED (°)        |
|--:|-----------------|----------------------|----------------:|------------------:|
| 1 | `thumb_roll`    | (-50, +10) L / (-10, +50) R | **−12 L / +12 R** | -40 L / +40 R |
| 2 | `thumb_abad`    | (0, +100) L / (-100, 0) R   | **+35 L / −35 R** | +80 L / -80 R |
| 3 | `thumb_mcp`     | (-49, 0) L / (0, +49) R     | ∓5             | -40 L / +40 R |
| 4 | `index_abad`    | (0, +12) L / (-12, 0) R     | 0              | ±6 |
| 5 | `index_pip`     | (0, +90)             | 5               | 88                |
| 6 | `middle_pip`    | (0, +90)             | 5               | 88                |
| 7 | `ring_abad`     | (-10, 0) L / (0, +10) R     | 0              | ∓5 |
| 8 | `ring_pip`      | (0, +90)             | 5               | 88                |
| 9 | `pinky_abad`    | (-10, 0) L / (0, +10) R     | 0              | ∓5 |
| 10| `pinky_pip`     | (0, +90)             | 5               | 88                |

Where the OPEN / CLOSED columns have two values, left vs right is
mirrored because the abad / roll ranges are physically mirrored
across the body midline. Where there's a single value, both sides
are identical.

Notes on the bolded `thumb_roll` / `thumb_abad` OPEN values
(2026-05-13): these were biased into a natural resting pose
(thumb sits ~35° across the palm at rest, pad rolled slightly
inward) instead of the previous near-perpendicular `0° / ±10°`.
CLOSED anchors did not move, so hardware hardstop headroom is
unchanged. See
[2026-05-13 robocasa finger-fix milestone](../user_guide/milestones/2026-05-13_robocasa_finger_fixes.md)
§4 for the rationale and trade-off (visible-motion span shrinks
slightly; iterate visually in the MuJoCo viewer if these need
further tuning).

### Thumb opposition: fingertip-proximity signal + anchor expansion

This subsection covers three layered fixes that together let the
robot's thumb actually meet the operator's thumb-pose during both
fist gestures and thumb-fingertip touches:

1. **Independent thumb-opposition signal** (the lateral CMC swing
   isn't recoverable from per-finger curls).
2. **Folding `oppose` into all three thumb motors** via
   `max(thumb_oppose, thumb_flex_curl)` (the May 10 update extended
   this from 2 motors to 3).
3. **Pushing the thumb CLOSED anchors from 50 % to 80 % of hardware
   travel** so the robot can physically reach the touch pose.

#### 1. Why we need a separate opposition signal

The thumb's CMC joint — the joint that swings the thumb laterally
across the palm — is **not in the XRHand chain** (the chain starts
at `thumb-metacarpal`, which sits *at* the CMC pivot). Driving
`thumb_roll` and `thumb_abad` from the per-finger thumb curl alone
only swings them to ~50 % of their travel during a thumb-finger
touch, because Quest 3's chain only sees MCP + IP flexion (~30–50 °
total during a touch) and that's all the curl signal can encode.

The opposition signal is **fingertip proximity**: the minimum
distance from the operator's `thumb-tip` to any of the four
fingertips (index, middle, ring, pinky), normalized by palm width
(distance from `index-finger-metacarpal` to
`pinky-finger-metacarpal`). The mapping saturates at `s = 1` when
`dMin / palm_width < 0.06` (≈0.5 cm, in contact) and falls to
`s = 0` for `dMin / palm_width > 0.45` (≈3.5 cm, controller grip or
hand neutral). It directly captures intent ("thumb is in contact
with another finger"), saturates regardless of *which* finger is
touched, and distinguishes a thumbs-up gesture (thumb extended away
from palm) from a real touch.

This proximity signal replaced an earlier *lateral-projection*
signal (thumb-tip projected onto the index-MCP → pinky-MCP axis).
That earlier signal saturated only when the thumb crossed *well
past* palm centre toward the pinky-MCP — an anatomical extreme
almost no operator reaches. Cross-correlation analysis on
`data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_000000.npz`
showed **zero** frames out of 666 where `thumb_flex >= 0.85` while
other fingers had average curl `<= 0.30` — Quest 3 effectively never
reports a "thumb-only-curled" hand pose, so the lateral-projection
fallback never triggered cleanly.

#### 2. Folding `oppose` into all 3 thumb motors

All three thumb motors (`thumb_roll`, `thumb_abad`, `thumb_mcp`)
lerp on `max(thumb_oppose, thumb_flex_curl)` rather than picking
one signal per motor.

The asymmetry between the two source signals is what justifies the
fold-in:

| Gesture                       | Quest 3 `thumb_oppose` | Quest 3 `thumb_flex` | Robot needs |
|-------------------------------|------------------------|----------------------|-------------|
| Closed fist (thumb tucked)    | ~0.5 (palm centre)     | high (curl chain)    | All 3 thumb motors closed |
| Thumb-fingertip touch         | high (proximity)       | low (IP doesn't fold) | All 3 thumb motors closed |
| Thumbs-up                     | 0 (thumb far from tips)| ~0 (extended)         | All 3 thumb motors at OPEN |
| Open palm                     | 0                      | 0                     | All 3 thumb motors at OPEN |

`max()` reduces to whichever input is high. Concretely the May 10
sweep (right-thumb-touch frames in v4 ep0, raw `oppose ≥ 0.4`,
raw `thumb_flex ≤ 0.3`, n = 12) shows what each fix bought:

| Variant | thumb_roll | thumb_abad | thumb_mcp |
|---|---|---|---|
| Recorded baseline (live) | 46 % / 66 % | 44 % / 63 % | **28 % / 30 %** |
| `hand_range_calibrated` (per-finger normalization only) | 73 % / 75 % | 70 % / 71 % | **10 % / 13 %** |
| **Anchor-expansion + 3-motor fold-in (live default)** | **98 % / 100 %** | **98 % / 100 %** | **98 % / 100 %** |

(Format: mean / max % closure across the 12 touch frames.)

The v0 → v0.3 path drove `thumb_mcp` from `thumb_flex` alone, which
gave only ~22 % closure on touch frames — exactly what the operator
saw as "the robot's thumb sweeps across the palm but stays straight
and ends up alongside the fingers". v0.4 (May 10) folded `oppose`
into `thumb_mcp` as well and the mean closure jumped to 98 %.

The fold-in also serves as a robustness backstop: when XRHand
momentarily drops the fingertip joints, the `thumb_oppose`
proximity signal returns `None` for that frame, and `max()` falls
back to driving all three motors from the flex curl alone.

#### 3. Anchor expansion to 80 % hardware travel

Even with both `thumb_oppose = 1` and `thumb_flex_curl = 1`, the
upstream agitbot `quest3-bare-hand-control` constants stopped the
thumb at half-travel:

| Anchor | Pre-v0.4 (%travel) | v0.4 (%travel) |
|---|---|---|
| `thumb_roll` CLOSED | 50 % (LEFT -30°, RIGHT +30°) | 80 % (LEFT -40°, RIGHT +40°) |
| `thumb_abad` CLOSED | 60 % (LEFT +60°, RIGHT -60°) | 80 % (LEFT +80°, RIGHT -80°) |

20 % of additional travel was sitting unused on the URDF at every
"closed thumb" frame. Pushing the anchors closer to the hardware
extremes lets the robot's thumb actually swing across the palm so
a thumb-to-fingertip gesture from the operator translates to a
thumb-to-fingertip pose on the robot. The 80 % cap (rather than
100 %) keeps a small mechanical margin for finger / thumb
interference at extreme poses.

The constants live as
`HAND_GRASP_CLOSED_LEFT_DEG` / `HAND_GRASP_CLOSED_RIGHT_DEG`
in `gear_sonic/utils/teleop/x2_hand_retarget.py` and are exercised
by the `test_oppose_*` suite in
`tests/test_teleop_v2_dropout_and_orientation.py`.

### Per-finger curl normalization (the live default)

Quest 3's hand-pose estimator has two structural quirks that the
retargeter must compensate for:

1. **Operator-specific resting and saturation values.** "Open hand"
   doesn't read as raw 0 — Quest 3 reports a baseline curl of
   roughly 0.05–0.20 per finger even when the operator's hand is
   relaxed flat, with the exact value depending on hand size and
   resting tone. "Full fist" caps at ~0.85–0.95, never quite 1.0,
   because Quest 3's tracking under-estimates joint flexion when
   the fingers occlude each other.
2. **A "fingers move together" prior.** Pairwise correlations of
   +0.99–+1.00 between index/middle/ring/pinky curls in v3 ep0
   (666 hand-mode frames). When the operator curls a single finger
   alone, the other three report partial curls too. Maximum
   observed isolated curl was ~0.30 for index/middle/pinky and
   0.30 for ring.

The live retargeter handles both quirks via **per-finger affine
normalization** parameterised by an operator-specific
`HandRangeCalibration` baked into the calibration YAML.

#### How it works

For each finger `i ∈ {thumb, index, middle, ring, pinky}`, the
calibration stores a `(floor[i], ceiling[i])` pair. A separate
`(oppose_floor, oppose_ceiling)` pair handles the thumb-opposition
signal. At runtime:

```text
normalized_curl[i] = clip( (raw_curl[i] - floor[i]) / (ceiling[i] - floor[i]),
                            0.0, 1.0 )
```

Then the normalized curl drives a linear lerp between the OPEN and
CLOSED anchors for each motor (with the thumb-motor fold-in
described above). Concretely, with the v4 ep0 calibration:

```text
RIGHT calibration (operator: stickbot, v4 capture):
  floor   = [thumb 0.198, index 0.055, middle 0.084, ring 0.062, pinky 0.095]
  ceiling = [thumb 0.989, index 0.881, middle 0.894, ring 0.877, pinky 0.869]
  oppose: floor=0.000  ceiling=0.544
```

So an operator who never quite reaches raw 1.0 still hits the
robot's CLOSED anchor on a deliberate fist, and the relaxed-hand
values map exactly to OPEN. No deadzone, no power curve — the
mapping stays linear *between* the operator's actual extremes,
which is what preserves smooth intermediate variation.

#### Where the calibration values come from

`gear_sonic/scripts/fit_hand_range_from_npz.py` reads a debug NPZ
from a previous teleop session and writes the per-finger
`(p05, p95)` percentiles plus the oppose `(p05, p95)` into the
operator's calibration YAML under `hand_range:`. p05 / p95 (rather
than min / max) absorb tracker noise and brief frame dropouts. The
script enforces a minimum spread of 0.05 to prevent fingers that
never closed during the capture session from collapsing to a
single point.

The fit is **operator-specific**: a different person, or even the
same person on a different headset session, can have a 10–20 %
shift in the resting / saturation values. Re-running
`fit_hand_range_from_npz.py` after every meaningful change of
operator or headset adjustment is the recommended workflow. See
[Section 2.3](#23-vr-operator-calibration) for how the
`HandRangeCalibration` block fits into the YAML schema.

#### Why we abandoned the global power-curve compensation

Through ~v0.2 the live default was a piecewise power-curve stretch
(`stretch_finger_curls`) with five hand-tuned `(deadzone,
full_threshold, gamma)` triples. It maximised bimodality (98.5–
99.6 % of post-stretch outputs at <0.05 or >0.95) but operators
consistently reported the same complaint: *"the robot's fingers
just snap from open to closed; I lose all smooth variation."*

The trade-off is structural: any compensation that pushes Quest 3's
0.20–0.80 mid-range curls toward 0 or 1 also pushes deliberately
intermediate gestures (a half-grasp, a soft pinch) into the same
endpoints. The affine-normalization approach above instead targets
the operator's *own* range without compressing what's between, so
both endpoints saturate cleanly **and** intermediate gestures
preserve their amplitude.

`stretch_finger_curls` is still in `x2_hand_retarget.py` as an
opt-in tool (`apply_curl_compensation=True`,
`apply_oppose_compensation=True`) — useful when an operator
wants explicit binary close/open feel for tasks like a tight
power-grasp pick-and-place. The script
`gear_sonic/scripts/tune_finger_curl_compensation.py` and the
visualiser
`gear_sonic/scripts/replay_finger_curl_comparison.py` still work
against it. The defaults (`DEFAULT_APPLY_CURL_COMPENSATION = False`,
`DEFAULT_APPLY_OPPOSE_COMPENSATION = False`) reflect the live
linear path.

### Re-tuning against fresh recordings

The live path (per-operator affine normalization) re-fits from any
recorded debug NPZ:

```bash
# Re-fit the operator's per-finger floor / ceiling from a recorded
# debug NPZ and write into the calibration YAML. p05 / p95 by
# default; --p-low / --p-high override.
python -m gear_sonic.scripts.fit_hand_range_from_npz \
    --calibration data/operator_calibrations/default.yaml \
    --npz data/lerobot/x2_quest3_kinematic_v4/debug/teleop_episode_000000.npz
```

The opt-in `stretch_finger_curls` path still has its own tuning
helpers (only useful if you've explicitly enabled curl
compensation):

```bash
# Pool all available episodes, find best global stretch params:
python -m gear_sonic.scripts.tune_finger_curl_compensation \
    --mode global \
    data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_*.npz

# Find best per-finger stretch params:
python -m gear_sonic.scripts.tune_finger_curl_compensation \
    --mode per-finger \
    data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_*.npz

# Render visual before/after comparison plots:
python -m gear_sonic.scripts.replay_finger_curl_comparison \
    data/lerobot/x2_quest3_kinematic_v3/debug/teleop_episode_000000.npz
```

To explore "what would this NPZ look like with a different mapping
config?" without re-recording, see
`gear_sonic/scripts/replay_recorded_dataset.py`. It re-runs the
current retargeting pipeline over a recorded NPZ and writes a new
parquet alongside the original — useful for A/B-ing a calibration
file or stretch parameter change before wiring it into the live
defaults. The May 10 thumb-fix work used this loop to iterate
without touching the headset.

### Thumb-opposition rest-bleed suppression

The same piecewise-power-curve stretch is also applied to the
JS-side thumb-opposition signal (`computeThumbOpposition` in
`index.html`). The opposition signal is a normalised
thumb-tip-to-nearest-fingertip proximity score in [0, 1] with
saturation at 1.0 for any clear thumb-finger touch. At rest hand
pose the signal drifts to **0.05–0.25** because the thumb tip
naturally sits a few cm from the index fingertip even when the
operator isn't intentionally opposing. Without compensation,
this leaks 5–25 % spurious closure into `thumb_roll` /
`thumb_abad` at rest, which the operator perceives as "the
robot's thumb starts moving by itself".

Live defaults (`DEFAULT_OPPOSE_*`):

| Param | Value |
|-------|------:|
| deadzone | 0.25 |
| full_threshold | 0.40 |
| gamma | 3.0 |

This suppresses anything below 0.25 to 0 (covers rest-bleed) and
saturates anything above 0.40 to 1.0 (covers any clear
thumb-finger touch). Pass `apply_oppose_compensation=False` to
`per_finger_grasp_command_from_curls_and_oppose(...)` to opt out
(legacy direct-lerp behaviour).

### Non-thumb fingertip-to-thumb touch (v0.5: in progress, needs new recording to verify)

This is the current edge of the quality envelope. The thumb fix
above closed the **thumb side** of a thumb-fingertip touch
gesture — but the **other finger's side** of the same gesture
is still imperfect: when the operator brings their index pad to
their thumb pad, the robot's index fingertip ends up roughly
where the operator's PIP would be, not where their tip would be.

Two factors stack:

1. **Topology mismatch** (see "OmniHand has fewer flexion DOFs"
   above): the operator has 3 cascaded knuckles bending ~80° each
   for a total of ~240°; the OmniHand has one active flexion DOF
   driving 2 mimic-coupled segments, with ~189° of total bend at
   hardware-max pip. Even at `curl = 1.0` and `pip = 90°`, the
   robot's tip arc is geometrically narrower than the operator's.
2. **Quest 3 under-reports isolated finger curls.** Per the v4 ep0
   data, the right pinky on dedicated thumb-to-pinky touch frames
   reports raw `pinky_curl ≈ 0.37` — the "fingers move together"
   prior caps a single-finger bend signal well below 1.0 even when
   the operator's tip is clearly bent toward palm. After
   normalization the signal reads ~0.36, so the robot pinky pip
   only travels ~36 % of the way from OPEN to CLOSED while the
   thumb has already been pushed to 100 %.

**v0.5 fix (May 11)**: mirror of the thumb-opposition treatment.
Two changes land together so the kinematic ceiling is raised at
the same time as the new signal that lifts isolated fingers up
to that ceiling:

* **Per-finger tip-proximity in JS** (`computeFingerTipOppose`
  in `gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html`):
  same `dist / palm_width` formulation as `computeThumbOpposition`
  but applied per non-thumb fingertip, returning a 4-vector
  `[index, middle, ring, pinky]`. Saturates at literal contact
  (d_norm < 0.06 ≈ 0.5 cm), zero past d_norm > 0.45 (≈ 3.5 cm).
  Emitted alongside the existing `thumb_oppose` in
  `hands.<side>.finger_tip_oppose` on the WebSocket payload, and
  persisted to debug NPZ as
  `quest_left_finger_tip_oppose` / `quest_right_finger_tip_oppose`
  (shape `(N, 4)`, NaN row for pre-May-2026 schemas).
* **Python combined drive** in
  `per_finger_grasp_command_from_curls_and_oppose`: each non-thumb
  `*_pip` motor (and the matching `*_abad` for index/ring/pinky)
  is now driven on `max(curls[i+1], finger_tip_oppose[i])` for
  `i ∈ {0..3}`. On non-touch frames `finger_tip_oppose ≈ 0` and
  `max` reduces to the existing curl path so smooth proportional
  variation is preserved. NaN entries (a fingertip dropped out
  this frame) fall back to the curl signal per finger.
* **Pip CLOSED anchor pushed from 80° to 88°**
  (`HAND_GRASP_CLOSED_LEFT_DEG[4,5,7,9]` and
  `HAND_GRASP_CLOSED_RIGHT_DEG[4,5,7,9]`). The pip range is
  `0..90°`, so 88° is ~98 % of hardware travel — the closest
  we can land to the geometric tip arc without bumping the
  hardware limit. Without this bump the JS signal would lift
  the robot to its old ~89 % ceiling on touch, leaving a
  visible gap.

The thumb-side coverage stays bit-identical: thumb_oppose still
saturates on the same touch frames, the `_THUMB_COMBINED_DRIVE_MOTORS`
set is unchanged, and the four thumb anchors are still at their
~80 % hardware travel from the previous fix.

**Verification status**: code lands behind a back-compat-safe API
(`finger_tip_oppose=None` is identical to the previous behaviour,
and tests assert this). The v4 NPZ doesn't contain the new field
— we can't iterate offline against existing recordings — so a
fresh test session is required to visually confirm the
thumb-fingertip touch now closes the receiving finger. See the
"How to verify" checklist in [Section 11](#11-next-steps).

A more ambitious alternative (filed but not planned) would do
**fingertip-position IK** on the per-finger 2-link chain: solve
for the pip angle that lands the robot tip closest to the
operator's tip in palm frame. Kinematically correct but a much
larger change to live retargeting math and end-to-end timing
budget.

### Finger-signal smoothing (v0.6, May 12)

`v5/ep1` showed clean signal acquisition and the `finger_tip_oppose`
plumbing landed cleanly, but a fine ~2–4° peak-to-peak finger
tremor was still visible during held poses. We characterised it on
the recording:

- **Held-pose curl std = 0.003–0.012** (~ 0.3–1° at the 88° pip
  anchor); spectrum sits at 1–2 Hz.
- **0 % of single-frame `|d/dt| > 0.05` curl spikes return to
  baseline** — they're real intentional motion, not noise.
- **`finger_tip_oppose` median delta = 0.000**, but max delta
  0.5–0.9 — signal is event-like (touch-onset spikes).

This means the right shape is **NOT** a low-pass filter: any
cutoff that catches the 1–2 Hz tremor band also lags real motion
by 100–200 ms. Instead we use a hybrid:

1. **Light EMA (α=0.5)** for single-frame outliers (+20 ms lag).
2. **Rolling-median deadband-hold** (8-frame window, ~160 ms): when
   the per-channel rolling std drops below `hold_std=0.005`, the
   output snaps to the rolling median (which still tracks slow
   drift). Releases on `std > release_std=0.012` OR
   `|x - median| > release_disp=0.020`.
3. **Hysteresis** keeps the latch from chattering near threshold.
4. **Brief-NaN bridging** lets the held value survive a 1-3 frame
   XRHand re-acquire.

Validated on `v5/ep1`: held-pose `|d/dt|` p99 reduced **20-40 %**
on the worst-twitching fingers, **+20 ms** motion-edge lag, **0 ms**
touch-onset lag, max single-frame jump reduced ~30-40 % across all
10 channels.

The filter is wired into:

- `teleop_x2_kinematic.py` — live kinematic teleop + record.
- `x2_dataset_recorder.py` — SONIC-record path.
- `replay_recorded_dataset.py` — offline replay (with `auto`
  / `always` / `never` mode flag).

The debug NPZ now persists **both** raw and filtered channels
(`quest_*_hand_curls` + `quest_*_hand_curls_filtered`, etc.),
so offline A/B is just a flag flip.

CLI flags (all opt-in to **disable** or **tweak** — defaults are
on, calibrated):

```text
--no-finger-filter
--finger-filter-alpha       <float>     # default 0.5
--finger-filter-hold-window <int>       # default 8 (160 ms at 50 Hz)
--finger-filter-hold-std    <float>     # default 0.005
```

For `replay_recorded_dataset.py`:

```text
--apply-finger-filter {auto,always,never}
   auto    : use NPZ's pre-computed *_filtered if present (post-v0.6),
             else apply offline. Default.
   always  : force an offline pass even when filtered keys are present
             (useful for tuning a different filter cfg).
   never   : replay raw signals as recorded (matches pre-v0.6 behaviour).
```

Full design + tuning rationale, and the complete code surface, are
in
[2026-05-12 finger-signal smoothing milestone](../user_guide/milestones/2026-05-12_finger_signal_filter.md).

### Other v0 hand limitations

1. **Quest 3 occludes the thumb when curled into the palm.** Tight
   fists report a thumb-flex curl of ~0.5–0.6 even when the
   operator's thumb is fully tucked. The opposition signal is
   unaffected (it's a fingertip-proximity score, not flexional), but
   the `thumb_mcp` knuckle still under-reports during a fist. In
   practice the May 10 fold-in (`max(oppose, thumb_flex)` driving
   all 3 thumb motors) absorbs most of the residual under-shoot,
   because a tight fist still gives a moderate `oppose` (thumb-tip
   sits near palm centre, close to fingertips).
2. **Per-finger abduction is locked to per-finger flexion** (not
   tracked independently). The X2 omnihand's `index_abad`,
   `ring_abad`, `pinky_abad` motors have very narrow ~5–10° hardware
   ranges and currently lerp from OPEN→CLOSED on the same curl as
   the matching `_pip` motor. So on the robot, fingers spread/close
   together with their flexion. Adding an independent finger-spread
   signal from XRHand metacarpal-to-metacarpal angles is a v1
   follow-up; the marginal gain is small given the hardware range.

### OmniHand sim stability (armature + damping + locked passives)

```{admonition} Status
:class: note
Landed May 10, 2026 alongside the `--sim-omnihand` deploy support.
Lives entirely in
[`gear_sonic/scripts/compose_x2_with_omnihand.py`](../../../gear_sonic/scripts/compose_x2_with_omnihand.py)
— the bare `x2_ultra.xml` is unchanged so the host SONIC training
distribution is preserved.
```

The OmniHand-augmented MJCF (`--sim-omnihand`) glues the OmniHand
URDFs onto `x2_ultra.xml` at runtime, adding 24+ finger DOFs to the
22-DOF body. The first integrations exposed three independent sim-
stability bugs that all show up as "fingers go crazy in MuJoCo":

#### 1. `middle_abad` was a free-floating, unranged hinge

The middle finger has only one active flexion DOF (`middle_pip`) by
hardware design — unlike index / ring / pinky, it has no abduction
motor. But the upstream URDF still ships a `middle_abad` hinge
declaration (looks copy-pasted from the other fingers). With:

* **No actuator** (not in `ACTIVE_FINGER_JOINTS`)
* **No equality coupling** (not in `PASSIVE_MIMIC_RULES`)
* **No range** (`model.jnt_range[middle_abad] == (0, 0)` so MuJoCo
  doesn't even clamp it)

…any tiny numerical perturbation from the equality solver accumulates
as runaway angular velocity around the sideways axis, and the visual
mesh punches through the palm collision shell. Visually: the middle
finger spins in a plane that no human finger can.

**Fix.** [`compose_x2_with_omnihand.py`](../../../gear_sonic/scripts/compose_x2_with_omnihand.py)
declares
[`LOCKED_PASSIVE_JOINTS`](../../../gear_sonic/scripts/compose_x2_with_omnihand.py)
and adds a one-coefficient `mjEQ_JOINT` equality (`passive - 0 = 0`)
for every entry. This is the equality-constraint analogue of welding
the joint shut without changing the URDF topology the SDK expects.

```python
LOCKED_PASSIVE_JOINTS: tuple[str, ...] = (
    "middle_abad",
)
```

If a future OmniHand revision adds a similar vestigial hinge, just
append it to the tuple and the compose script will pin it on the
next sim launch.

#### 2. Finger joints had no armature, no passive damping

Even with `middle_abad` welded shut, all 32 remaining finger joints
(active + mimic-passive) **continued to wiggle nonstop** in the
SONIC bridge sim (`mj_step` path), while the kinematic renderer
(`mj_forward` only, no integration) looked fine. Smoking gun: the
problem was numerical, not retargeting.

The OmniHand URDF link inertias are extremely small — PIP segments
are ~1.2e-5 kg·m², DIP segments are ~1.5e-6 kg·m² (lighter than a
paper clip). With:

* The **soft mimic equality solver** (`solimp[0] = 0.9`) injecting
  tiny but non-zero constraint forces every step at 1 kHz, AND
* **Zero passive joint damping** to absorb that energy at the
  joint level,

…the integrator turns those forces into 70°+ of finger drift in 5
seconds even when the position actuators are commanded to *hold*
the rest pose.

**Fix.** Add a small armature (rotor inertia at the joint) and a
small passive damping to every OmniHand finger joint. Armature
inflates the diagonal of the mass matrix without changing the link
inertia, so pose / FK / contacts are unchanged but the joints become
insensitive to small constraint impulses. Passive damping bleeds off
any residual oscillation between actuator updates.

Empirical sweep (5 s of `mj_step` with `ctrl` frozen at rest pose,
max over all 32 non-locked finger joints):

| config | drift (°) | jitter (°/s) | max deviation (°) |
| ------ | --------: | -----------: | ----------------: |
| baseline (zero, zero) | 75.5 | 3393 | 100.5 |
| `armature=1e-3` only | 0.9 | 13.0 | 10.2 |
| `armature=1e-3` + `damping=0.05` | 0.9 | **3.6** | **2.6** |

Armature alone gives a 90× drift / 260× jitter improvement; adding
the damping further smooths residual oscillation to <0.1° per
second. The chosen values live as
[`_FINGER_ARMATURE = 1e-3`](../../../gear_sonic/scripts/compose_x2_with_omnihand.py)
and `_FINGER_DAMPING = 0.05` in the compose script. Tracking
response with these values: at `kp=20`, the closed-loop time
constant is ~50 ms — well under the bridge's 1 kHz integration step
and indistinguishable at the 50 Hz teleop command rate. Human
finger motion lives at <5 Hz; the operator does not perceive the
added inertia.

```{admonition} MuJoCo 3.5 vs 3.7 gotcha
:class: warning
The fix patches `model.dof_armature[dofadr]` and
`model.dof_damping[dofadr]` on the *compiled* `MjModel` rather than
going through the `MjSpec.joint(name).damping` setter. The
`MjsJoint.damping` Python attribute is **scalar-typed in MuJoCo
3.5.x** but became a **3-element NDArray in 3.7.x** (per-DOF for
compound joints). Our deploy Docker image and host venv are pinned
to different minor versions; the compiled-model arrays are scalar-
per-DOF on every MuJoCo version, so the patch is forward / backward
compatible.
```

#### 3. `pyzmq` was missing in the bridge container

With the dynamics fixes in place, fingers held the rest pose
cleanly — but **didn't respond to operator input**. Symptom: every
finger sat at its rest angle, no matter what the operator did with
the controller triggers or hand-tracking curl.

Root cause: the OmniHand finger commands flow over a separate ZMQ
SUB socket from the recorder to the bridge (see
[`x2_mujoco_ros_bridge.py`](../../../gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py)
`apply_active_hand_rest_pose`). The bridge ran as `python3` inside
the `docker_x2/x2sim` container, which historically didn't have
`pyzmq` baked in. The `import zmq` raised, the bridge silently fell
through to "rest pose forever", and the only signal in the deploy
log was the absence of the SUB-online line.

**Fix.** Two layers:

1. **Permanently bake `pyzmq` into the Dockerfile.** On the next
   `cd gear_sonic_deploy/docker_x2 && docker compose build` of
   `x2sim`, the dependency is cached.
2. **Best-effort runtime install fallback** in
   [`x2_mujoco_ros_bridge.py`](../../../gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py).
   If `import zmq` fails at bridge startup (older container layer),
   the bridge runs `pip3 install pyzmq` once and retries. Logs:

   ```text
   [bridge] OmniHand ZMQ subscriber: pyzmq not present in this
   container; attempting one-shot 'pip3 install pyzmq' …
   [bridge] pyzmq installed at runtime; OmniHand SUB online.
   ```

   If the install also fails (no network, container immutable, etc.)
   the bridge prints a loud single-line warning and continues with
   fingers stuck at rest pose — useful so the rest of the loop still
   exercises the body policy.

#### Smoke test

After all three fixes, run:

```bash
bash gear_sonic/scripts/record_x2_dataset.sh \
    --teleop-only --sim-omnihand --no-vla \
    --sonic-checkpoint <path> \
    --sim-duration 30
```

Watch the MuJoCo viewer with no operator engagement: every finger
should sit at its rest pose for the full 30 s with no visible
oscillation. Then engage VR and squeeze a controller trigger — the
matching robot fingers should curl in within ~50 ms.

---

## 9. Pointers into the implementation

| File | Role |
| ---- | ---- |
| [`gear_sonic/utils/teleop/solver/arm/x2_arm_fk.py`](../../../gear_sonic/utils/teleop/solver/arm/x2_arm_fk.py) | Pure-numpy FK + analytical Jacobian for the 7-DOF X2 arm chain. |
| [`gear_sonic/utils/teleop/solver/arm/x2_arm_ik.py`](../../../gear_sonic/utils/teleop/solver/arm/x2_arm_ik.py) | Single-step DLS IK solver, joint-limit clamped. |
| [`gear_sonic/utils/teleop/operator_calibration.py`](../../../gear_sonic/utils/teleop/operator_calibration.py) | Per-operator calibration: dataclass, fit, YAML I/O, head-yaw frame transform. |
| [`gear_sonic/utils/teleop/vr_arm_teleop_v2.py`](../../../gear_sonic/utils/teleop/vr_arm_teleop_v2.py) | Stateless head-relative wrist retargeter that consumes `OperatorCalibration`. v0 default. |
| [`gear_sonic/utils/teleop/vr_arm_teleop.py`](../../../gear_sonic/utils/teleop/vr_arm_teleop.py) | Legacy engage-anchor retargeter (deprecated; kept for tests + legacy recordings). |
| [`gear_sonic/utils/teleop/x2_hand_retarget.py`](../../../gear_sonic/utils/teleop/x2_hand_retarget.py) | Trigger/grip → 10-DOF OmniHand command (open/closed motor anchors). |
| [`gear_sonic/utils/teleop/x2_dataset_recorder.py`](../../../gear_sonic/utils/teleop/x2_dataset_recorder.py) | Top-level orchestrator: ZMQ pub/sub, MuJoCo render, button state machine, LeRobot writer. |
| [`gear_sonic/data/dataset_output_dir.py`](../../../gear_sonic/data/dataset_output_dir.py) | Shared `--output-dir` preflight (auto-cleans empty / half-init stubs). |
| [`gear_sonic/scripts/vr_operator_calibrate.py`](../../../gear_sonic/scripts/vr_operator_calibrate.py) | Standalone 4-pose calibration CLI. |
| [`gear_sonic/utils/teleop/vr/quest3_audio_prompts.py`](../../../gear_sonic/utils/teleop/vr/quest3_audio_prompts.py) | Generates the calibration audio cache (gTTS-rendered MP3s served at `/audio/<key>.mp3`). |
| [`gear_sonic/scripts/record_x2_dataset.py`](../../../gear_sonic/scripts/record_x2_dataset.py) | Recorder CLI shim. |
| [`gear_sonic/scripts/record_x2_dataset.sh`](../../../gear_sonic/scripts/record_x2_dataset.sh) | Co-launches the deploy + recorder. |
| [`gear_sonic/scripts/teleop_x2_kinematic.py`](../../../gear_sonic/scripts/teleop_x2_kinematic.py) | Pure-kinematic VR teleop (no SONIC, no deploy) — fastest debug loop. |
| [`gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html`](../../../gear_sonic/utils/teleop/vr/quest3_webxr_app/index.html) | WebXR client with calibration overlay + TTS. |
| [`gear_sonic/scripts/process_dataset.py`](../../../gear_sonic/scripts/process_dataset.py) | Post-process / merge / clean LeRobot datasets. |
| [`gear_sonic/scripts/compose_x2_with_omnihand.py`](../../../gear_sonic/scripts/compose_x2_with_omnihand.py) | Programmatically composes `x2_ultra.xml` with the OmniHand URDFs at runtime. Owns `LOCKED_PASSIVE_JOINTS`, `_FINGER_ARMATURE`, `_FINGER_DAMPING`, and the mimic-equality solver. |
| [`gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py`](../../../gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py) | Bridges the C++ deploy to the MuJoCo sim. Subscribes to the OmniHand ZMQ command topic; runtime-installs `pyzmq` if missing. |
| [`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp`](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp) | The `--wrist-bypass ik` override loop and `kBypassedWristMjDofs` constant. |
| [`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp`](../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/x2_deploy_onnx_ref.cpp) | C++ deploy main: SONIC inference loop, safety stack, wrist-bypass call site, periodic status log. |

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| Recorder logs `waiting for first Quest 3 packet …` forever | Headset can't reach the WebXR HTTPS server | Open the URL in the Quest 3 browser, accept the self-signed cert, then tap **Connect WS** + **Start VR**. Verify the workstation's firewall allows ports 8443 (HTTPS) and 8765 (WebSocket). |
| `Error: calibration file not found at …` | No `--calibration` YAML on disk | Run `vr_operator_calibrate.py --operator-id <id>` to capture one, OR pass `--recalibrate` to capture inline before recording. See [Section 2.3](#23-vr-operator-calibration). |
| `ValueError: Failed to resume from corrupted dataset …` | A previous run wrote `meta/info.json` and crashed before any episode finalized | The preflight in `gear_sonic/data/dataset_output_dir.py` should auto-clean half-init stubs on the next launch. If you still hit this, `rm -rf <output-dir>` and retry. |
| Robot's hands end up "behind the body" when you turn your body | You're using the legacy engage-anchor solver (`vr_arm_teleop.py`). The current default is the calibrated head-yaw mapping which is rotation-invariant. | Verify the recorder banner says `loaded calibration …`. If it doesn't, your code path is still on `VRArmTeleop`; rebase / re-import. |
| Calibration capture rejects every pose with "Wrist moved too much" | Cluster spread (80th-pct distance from median) > 6 cm — usually a controller dropping out mid-capture, or one controller not actually held. | First, **make sure you're holding both controllers** with the headset in line-of-sight of both. The script logs `dropouts skipped: N` per attempt; a healthy capture has `N` close to 0. If you genuinely need a looser gate (e.g. handheld walking-around setup), pass `--spread-threshold-m 0.10` (10 cm). Tighter (precise lab setup): `--spread-threshold-m 0.03`. Longer averaging window: `--sample-window-s 2.0`. |
| Calibration captures all 4 poses fine but **fit residual exceeds the per-pose threshold** | One pose was captured at a geometrically inconsistent location (e.g. T-pose right arm angled forward by 17 cm instead of straight sideways), OR the operator's reach envelope is unusually different from the X2's. | The script no longer crashes here — it identifies the worst-contributing pose+arm, speaks a coaching line through the headset, and lets you press **A** to recapture only that pose. Just follow the prompt. Up to `--max-fit-recaptures` (default `4`) recaptures are allowed. If a particular pose legitimately needs a looser gate, pass the per-pose flag (e.g. `--t-pose-reject-m 0.15` for a 15 cm T-pose ceiling, or `--namaste-reject-m 0.22` for a 22 cm namaste ceiling). The default `namaste` gate is already looser (18 cm) than the others (10 cm) to absorb the controller-grip offset. |
| Robot's hands stay shoulder-width apart even when operator brings their hands together | v1 calibration (3-pose, no centerline anchor) is loaded. | Re-run `vr_operator_calibrate.py` to regenerate the YAML with the v2 4-pose schema (now includes a **namaste** pose at `op_y = 0`). The recorder won't load v1 YAMLs; it logs `schema_version mismatch` if it sees one. |
| Robot's elbows stay bent even when operator's arms are fully extended down | Same root cause as above — v1 used the bent-arm SONIC stand pose as the calibration `arms_down` reference, so the IK was instructed to keep elbows bent. | Re-calibrate. v2 uses fully-straight arms (`q = 0`) as both the calibration reference *and* the IK null-space preferred posture, so full operator extension produces full robot extension. |
| No audio in the headset during calibration | Headset volume muted, or the gTTS audio cache failed to populate (no internet at first run, or `gtts` not installed). | (1) Click **Test audio** on the WebXR page *before* pressing **Start VR** — you should hear *"Audio test successful."* and see the `🔊 SPEAKING` badge pulse on the overlay. If you see the badge but hear nothing, **the headset volume is the issue** (use the side rocker). (2) If you hear silence AND see `Audio: audio/audio_test.mp3 failed to play, falling back to TTS` in the page status row, the MP3 cache is empty: run `pip install gtts && python -c "from gear_sonic.utils.teleop.vr.quest3_audio_prompts import ensure_prompt_audio_files; ensure_prompt_audio_files(force_regenerate=True)"` on the workstation, then refresh the page. (3) The `Quest3Reader` regenerates the cache on every server boot, so a stable installation only needs step 2 the first time. |
| Calibration prompts not visible in immersive-ar view | The runtime denied the dom-overlay feature, or the page lost it during a session restart | The WebXR client requests the `dom-overlay` feature in `requestSession()` and the status row logs `domOverlayState: …`. If it says `not supported` or similar, close and reopen the WebXR page (cold session) and try again. |
| MuJoCo viewer doesn't open | The deploy was launched headless | Pass `--sim-viewer` (default in the wrapper). If you used the wrapper and still don't see a window, ensure `DISPLAY` is set in the shell that launched the wrapper. |
| Saved episode has 0 frames | Pressed **X** before any tick advanced (e.g. before VR connected) | Check the recorder log for `[X] dropping 0 frames (no frames)`. Press **B** again, wait for the recorder to log non-zero frame counts in its periodic status, then press **X**. |
| `[recorder] render warn (frame skipped): …` | EGL renderer hiccup; recorder drops the frame and continues | Safe to ignore unless it happens > 1 % of frames. If it does, drop the resolution (`--render-width 320 --render-height 240`) or move the renderer to a different GPU. |
| Deploy log spams `tilt watchdog` errors | The recorded body_q drifted out of the trained distribution (e.g. lower body got modified) | The recorder pins legs/waist/head to `DEFAULT_STAND_POSE_MUJOCO_RAD`. If you patched that, revert. |
| Wrists don't move on the robot even though IK ref does | SONIC pins `wrist_pitch` / `wrist_roll` at its trained comfort pose / asymmetric joint-range limit. See [§8 deep dive](#sonic-pins-the-wrist-dofs--and-why-we-bypass-them-in-c). | Use `--wrist-bypass ik` (default in `record_x2_dataset.sh`). Verify `wrist_bypass_ticks` increments on the deploy log's periodic `CONTROL tick=…` line. If you explicitly passed `--wrist-bypass off`, drop the flag. |
| Deploy refuses to start with `--wrist-bypass=ik requires --input-type=zmq` | You passed the bypass flag without `--vla` / on the motion-file replay path. The bypass would be a no-op there. | Either add `--vla` (so the deploy subscribes to the recorder's ZMQ pose feed) or drop the bypass flag. The recorder wrapper auto-suppresses the flag under `--no-vla`. |
| Middle finger spins through the palm collision shell | Vestigial `middle_abad` hinge in the upstream OmniHand URDF was free-floating + unranged. See [§8 OmniHand sim stability](#omnihand-sim-stability-armature--damping--locked-passives) §1. | Should be fixed for you by the equality lock in `compose_x2_with_omnihand.py::LOCKED_PASSIVE_JOINTS`. If a future OmniHand revision adds a similar joint, append it to that tuple. |
| Fingers jitter / wiggle constantly even with no operator input (`--sim-omnihand` mode) | Tiny URDF link inertias + soft mimic equality solver inject constraint forces with no joint damping to absorb. See [§8 OmniHand sim stability](#omnihand-sim-stability-armature--damping--locked-passives) §2. | Verify `_FINGER_ARMATURE` and `_FINGER_DAMPING` are non-zero in `compose_x2_with_omnihand.py`. The `mj_forward`-only kinematic renderer hides this — only `mj_step` paths (the SONIC bridge) expose it. |
| `--sim-omnihand` runs but fingers never move with operator input | `pyzmq` missing in the bridge container; the OmniHand ZMQ SUB silently failed to bind. | `cd gear_sonic_deploy/docker_x2 && docker compose build` to bake the dependency in permanently. The bridge has a runtime `pip3 install pyzmq` fallback that should auto-recover; check the log for `[bridge] OmniHand ZMQ subscriber: pyzmq not present …` lines. |
| Fingers refuse to close in `--robocasa-env` scenes but curl fine in bare sim (`--robocasa-env none`) | The X2's vestigial pre-OmniHand `wrist_roll_link` collision mesh — the legacy "closed fist" primitive shipped with the bare X2 model — physically blocks the OmniHand fingers from contacting anything inside the composed scene MJCF. The bare-sim path doesn't hit this because nothing is in reach of the fist shell; the moment a robocasa scene puts a table / cube / bowl into the workspace, the shell catches them first. | Rebuild the affected scene XML with the current `gear_sonic/scripts/build_x2_robocasa_scene_xml.py` — the `_disable_pre_omnihand_x2_fist_collision_mesh` helper zeroes `contype` / `conaffinity` on that mesh per side, and the post-compile verification asserts every OmniHand palm primitive still has collision. All three bundled scenes (`X2PickPlaceCube.xml`, `X2PickPlaceBowl.xml`, `X2PickPlaceApple.xml`) already ship with the disable. If you author a custom scene XML by hand, mirror the same flag flip on `*_wrist_roll_link` collision geoms. Full root-cause walk-through: [2026-05-13 robocasa finger-fix milestone §1](../user_guide/milestones/2026-05-13_robocasa_finger_fixes.md#1-disable-the-pre-omnihand-x2-fist-collision-mesh-in-scene-xmls). |

---

## 11. Next steps

### Hand-retargeting follow-ups

* **Per-finger tip-to-thumb proximity**: **landed in v0.5** (May
  11). Wires `finger_tip_oppose` end-to-end and bumps the non-thumb
  pip CLOSED anchor from 80° to 88°. See "Non-thumb fingertip-to-
  thumb touch" in §8 for the kinematic motivation. Visually
  verified on `data/lerobot/x2_quest3_kinematic_v5/debug/teleop_episode_000001.npz`
  ("very promising" feedback).
* **Finger-signal smoothing**: **landed in v0.6** (May 12). Per-side
  EMA + rolling-median deadband-hold on the 10 hand-input channels.
  Kills 20–40 % of held-pose tremor, +20 ms motion lag, 0 ms touch-
  onset lag. Enabled by default; live + record + replay all wired;
  debug NPZ persists raw + filtered. See the
  ["Finger-signal smoothing (v0.6)"](#finger-signal-smoothing-v06-may-12)
  section in §8 for the full design and tuning rationale.
* **Sonic-enabled record + replay path.** The teleop / record /
  replay matrix currently has only the kinematic backend wired
  end-to-end. Add Sonic-backed variants of `record_x2_dataset.py`
  and a Sonic replayer to fill out the 2 × 3 grid in
  [Section 6.5](#65-architecture-teleop--record--replay-matrix).
* **Per-operator `finger_tip_oppose` calibration**. The v0.5 wire
  layout uses the JS-side proximity output directly (saturates at
  d_norm < 0.06 ≈ 0.5 cm). Different operators have different
  finger lengths and palm widths, so the proximity-to-touch
  threshold may need a per-operator stretch the same way
  `thumb_oppose` got `oppose_floor` / `oppose_ceiling` in
  `HandRangeCalibration`. Hold this until we have ≥ 2 operators
  worth of recorded data to compare distributions; before then
  any normalization curve we choose is over-fitting to a sample
  size of one.
* **Wrist orientation in calibration.** v0 runs IK position-only
  (`--ik-rotation-weight 0`). Adding an operator-to-robot rotation
  map alongside the position map would let elbow circumduction and
  wrist roll/yaw carry through cleanly, fixing the residual
  "elbows facing in reverse" complaint when the operator switches
  between controller and hand-tracking mid-session.

### Downstream consumers of the recorded dataset

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
