# Dev Notes

## Local Development Environment Setup

**Date:** 2026-03-04
**Host OS:** Ubuntu 24.04 (x86_64, kernel 6.17.0)
**GPU:** NVIDIA (driver 580.126.09, CUDA 13.0)

---

### Prerequisites Installed

| Component | Version | Location |
|-----------|---------|----------|
| NVIDIA Driver | 580.126.09 | system |
| CUDA Toolkit | 13.0.88 | `/usr/local/cuda` (via apt: `cuda-compiler-13-0`, `cuda-cudart-dev-13-0`) |
| TensorRT | 10.15.1 | apt packages (`libnvinfer-dev`, etc.) with symlink tree at `~/TensorRT` |
| ONNX Runtime | 1.16.3 | `/opt/onnxruntime` |
| Python | 3.10.19 | uv-managed, in `.venv/` |
| uv | 0.10.0 | `~/.local/bin/uv` |
| just | 1.43.0 | `/usr/local/bin/just` |
| cmake | system | `/usr/bin/cmake` |
| clang | system | `/usr/bin/clang` |
| git-lfs | 3.4.1 | system |

### TensorRT Setup (apt-based)

The docs recommend downloading the TensorRT TAR package, but on this system TensorRT was installed via apt from the NVIDIA CUDA repository. Because `FindTensorRT.cmake` expects a `TensorRT_ROOT/include/` + `TensorRT_ROOT/lib/` layout, a symlink directory was created:

```bash
mkdir -p ~/TensorRT/include ~/TensorRT/lib
ln -sf /usr/include/x86_64-linux-gnu/NvInfer*.h ~/TensorRT/include/
ln -sf /usr/include/x86_64-linux-gnu/NvOnnx*.h ~/TensorRT/include/
ls /usr/include/x86_64-linux-gnu/Nv*.h | xargs -I {} ln -sf {} ~/TensorRT/include/
ln -sf /usr/lib/x86_64-linux-gnu/libnvinfer*.so* ~/TensorRT/lib/
ln -sf /usr/lib/x86_64-linux-gnu/libnvonnxparser*.so* ~/TensorRT/lib/
ln -sf /usr/lib/x86_64-linux-gnu/libtensorrt*.so* ~/TensorRT/lib/
```

The environment variable is set in `~/.bashrc`:

```bash
export TensorRT_ROOT=$HOME/TensorRT
```

> **Note:** The docs recommend TensorRT **10.13** for x86_64. This system has **10.15** (matched to CUDA 13.0). Different TensorRT versions *may* cause inference issues — test in simulation first before deploying to hardware.

---

### Python Virtual Environment

Created with `uv` using Python 3.10 (required by both `gear_sonic` and `decoupled_wbc` which specify `requires-python = "~=3.10.0"`):

```bash
uv venv --python 3.10 .venv
```

#### Installed packages

```bash
# gear_sonic with teleop and simulation extras
uv pip install -e "gear_sonic/[teleop,sim]"

# decoupled_wbc with full and dev extras
# Requires --no-build-isolation due to lerobot's poetry build backend
# and temporarily patching pyproject.toml paths (see workaround below)
uv pip install setuptools wheel poetry-core poetry
uv pip install --no-build-isolation -e "decoupled_wbc/[full,dev]"
```

#### decoupled_wbc install workaround

`decoupled_wbc/pyproject.toml` references `readme = "../README.md"` and `license = {file = "../LICENSE"}` which newer versions of setuptools reject (cannot access files outside the package directory). To install:

1. Create temporary symlinks inside `decoupled_wbc/`:
   ```bash
   cd decoupled_wbc
   ln -sf ../README.md README.md
   ln -sf ../LICENSE LICENSE
   ```
2. Temporarily edit `decoupled_wbc/pyproject.toml`:
   - Change `readme = "../README.md"` → `readme = "README.md"`
   - Change `license = {file = "../LICENSE"}` → `license = {file = "LICENSE"}`
3. Run the install (see above)
4. Revert `pyproject.toml` and remove symlinks

Additionally, `lerobot` (a dependency) uses `poetry-core` as its build backend, so `poetry-core` and `poetry` must be installed in the venv before running with `--no-build-isolation`.

---

### C++ Build (gear_sonic_deploy)

Built using `just` from inside `gear_sonic_deploy/`:

```bash
cd gear_sonic_deploy
export TensorRT_ROOT=$HOME/TensorRT
source scripts/setup_env.sh
just build
```

#### Built targets (in `gear_sonic_deploy/target/release/`):

| Binary | Description |
|--------|-------------|
| `g1_deploy_onnx_ref` | Main deployment executable |
| `freq_test` | Inference frequency test |
| `run_tests` | Unit tests (GTest) |
| `zmq_pose_subscriber_test` | ZMQ pose subscriber test |
| `zmq_python_sender_test` | ZMQ Python sender test |

#### Build notes

- ROS2 is **not installed** — the build skips `ROS2InputHandler` support (this is optional)
- DLA support is disabled on x86_64 (DLA is Jetson-only)
- cppzmq headers were vendored to `gear_sonic_deploy/third_party/cppzmq/` (not available via apt on Ubuntu 24.04)

---

### Environment Variables (in ~/.bashrc)

```bash
export TensorRT_ROOT=$HOME/TensorRT
```

### Useful Commands

```bash
# Activate Python venv
source .venv/bin/activate

# Set up C++ build environment (CUDA, TensorRT, ONNX Runtime paths)
cd gear_sonic_deploy && source scripts/setup_env.sh

# Build C++ project
just build          # Release build
just build Debug    # Debug build
just clean          # Clean build artifacts
just --list         # Show all available commands

# Run unit tests
./gear_sonic_deploy/target/release/run_tests

# Run inference frequency test
./gear_sonic_deploy/target/release/freq_test policy/release/model_decoder.onnx
```

---

## Quest 3 VR Teleop Setup

**Date:** 2026-03-05

### Overview

Quest 3 VR teleop uses WebXR + WebSocket to stream head + controller tracking data to the GEAR-SONIC pipeline. The sim2sim setup requires **3 terminals**.

### Additional Dependencies

Installed into the existing `.venv` (no separate venv needed):

```bash
source .venv/bin/activate
uv pip install websockets
```

A self-signed TLS certificate is generated at `gear_sonic/utils/teleop/vr/quest3_certs/`:

```bash
mkdir -p gear_sonic/utils/teleop/vr/quest3_certs
openssl req -x509 -newkey rsa:2048 \
    -keyout gear_sonic/utils/teleop/vr/quest3_certs/key.pem \
    -out gear_sonic/utils/teleop/vr/quest3_certs/cert.pem \
    -days 365 -nodes -subj "/CN=quest3-teleop"
```

Firewall ports must be open:

```bash
sudo ufw allow 8443/tcp   # HTTPS for WebXR app
sudo ufw allow 8765/tcp   # WSS for WebSocket data
```

### Running Sim2Sim with Quest 3

```bash
# Terminal 1 — MuJoCo Simulator
bash run_sim.sh

# Terminal 2 — C++ Deployment (with ZMQ manager input)
bash deploy_sonic_with_zmq.sh

# Terminal 3 — Quest 3 Teleop Manager
bash run_quest3_server.sh
```

### Quest 3 Headset Steps

1. Ensure Quest 3 is on the **same Wi-Fi** as the workstation
2. Open **Meta Quest Browser**
3. Navigate to `https://<workstation-ip>:8443`
4. Accept the self-signed certificate warning (Advanced → Proceed)
5. Also visit `https://<workstation-ip>:8765` and accept that cert too
6. Go back to `https://<workstation-ip>:8443`
7. Tap **"Connect WS"** (status turns green)
8. Tap **"Start VR"** to begin streaming

### Engaging the Robot

1. In **Terminal 2**, wait for `Init Done`
2. In **MuJoCo viewer**, press **`9`** to drop the robot
3. On **Quest 3 controllers**, press **A+B+X+Y** together to engage teleop (starts in PLANNER mode)
4. Press **A+B** to cycle from IDLE to SLOW_WALK / WALK / RUN
5. Press **A+X** to toggle VR 3PT (upper body tracking)

### Controls Reference

| Input | Action |
|---|---|
| A+B+X+Y | Start / Emergency Stop |
| A+X | Toggle VR 3PT (upper body tracking) |
| Left Stick | Move direction |
| Right Stick | Yaw / heading |
| A+B | Next locomotion mode |
| X+Y | Previous locomotion mode |
| Triggers | Hand grasp (VR 3PT mode) |
| Grips | Hand grip (VR 3PT mode) |

### Locomotion Modes

| ID | Mode |
|---|---|
| 0 | Idle (default) |
| 1 | Slow Walk |
| 2 | Walk |
| 3 | Run |
| 4 | Squat |
| 5-6 | Kneel |
| 7 | Lying face-down |
| 8 | Crawling |
| 17 | Forward Jump |
| 18 | Stealth Walk |
| 19 | Injured Walk |

---

### Issues Encountered & Fixes

#### 1. CycloneDDS multicast error on loopback

**Symptom:** Sim loop exits immediately with CycloneDDS domain creation failure.

**Fix:** Enable multicast on the loopback interface (already handled by `run_sim.sh`):
```bash
sudo ip link set lo multicast on
```

#### 2. WebXR "request failed" error on Start VR

**Symptom:** Clicking "Start VR" shows "request failed" on the Quest 3 browser.

**Cause:** The self-signed certificate for the WebSocket port (8765) was not accepted. The browser blocks WSS connections to untrusted certs.

**Fix:** Navigate to `https://<workstation-ip>:8765` in the Quest 3 browser, accept the certificate warning, then go back to the main page and retry.

#### 3. WebXR "device does not support requestReferenceSpace"

**Symptom:** Start VR fails with "failed to execute requestReferenceSpace on XRSession — device does not support".

**Cause:** Quest 3 guardian boundary not set up. The `local-floor` reference space requires a floor level and play boundary.

**Fix:** On Quest 3: Settings → Physical Space → tap **"Set Floor"** (point controller at floor, confirm) → tap **"Create Boundary"** (draw play area, confirm). The WebXR app now falls back through `local-floor` → `local` → `viewer` if needed.

#### 4. Controllers not detected (hand tracking mode)

**Symptom:** Head position updates in logs but all buttons/axes/triggers are zeros. Server shows `Input sources (controllers): 0` or `hand-tracking (gamepad=NO)`.

**Cause:** Quest 3 defaults to hand tracking when not holding physical controllers. WebXR hand tracking sources don't have gamepad buttons/axes.

**Fix:** **Pick up the physical Quest 3 controllers.** The headset auto-switches to controller mode. If it doesn't switch, go to Settings → Movement Tracking → Hand and Body Tracking → **turn off Hand Tracking** to force controller mode.

#### 5. Black screen in VR immersive mode

**Symptom:** Clicking Start VR enters immersive mode but shows a black screen.

**Cause:** The `immersive-vr` session renders to a blank WebGL canvas with no 3D scene.

**Fix:** The WebXR app now requests `immersive-ar` mode first (passthrough), falling back to `immersive-vr`. The canvas clears to transparent so the passthrough camera feed is visible.

#### 6. Browser caching old HTML page

**Symptom:** Code changes to the WebXR HTML page don't take effect after restarting the server.

**Cause:** Quest 3 browser caches the page aggressively.

**Fix:** Clear cache in Quest Browser (three dots menu → Settings → Clear Browsing Data → Cached images and files → Clear). The HTTP server now sends `Cache-Control: no-cache` headers to prevent future caching.

#### 7. Terminal 2 doesn't accept keyboard input with `--input-type zmq_manager`

**Symptom:** Pressing `]` in Terminal 2 does nothing.

**Cause:** With `--input-type zmq_manager`, all input comes via ZMQ from the Quest 3 manager — keyboard input is disabled.

**Fix:** This is expected. Use the Quest 3 controllers instead: **A+B+X+Y** to start the policy (replaces `]`), **A+B+X+Y** again for emergency stop (replaces `O`).

#### 8. Robot collapses / doesn't move after engaging

**Symptom:** Robot stands after A+B+X+Y but joystick has no effect.

**Cause:** Locomotion mode starts at **IDLE** (mode 0), which ignores joystick input.

**Fix:** Press **A+B** to cycle to SLOW_WALK (mode 1) or WALK (mode 2), then use the joystick.

#### 9. `websockets.exceptions.InvalidUpgrade: invalid Connection header: keep-alive`

**Symptom:** Error spam in Terminal 3 when accepting certs.

**Cause:** Navigating directly to the WebSocket port (`https://<ip>:8765`) sends a regular HTTPS request instead of a WebSocket upgrade. The `websockets` library (v16) rejects it.

**Fix:** This error is harmless — the certificate is still accepted. The actual WebSocket connection from the WebXR page works fine afterward.

---

## Heuristic Planner — Curated Motion Bins (X2 Ultra)

**Date:** 2026-05-11
**Scope:** All work to date on building the X2 Ultra heuristic motion planner that streams 50 Hz pose references over ZMQ for SONIC to track. Mirrors the wire format of Unitree G1's `LocalMotionPlannerTensorRT` but is built from primitive **bins** (not a learned planner) since NVIDIA never released that model's training/training data.

### Architecture (3 layers)

1. **Source library** — `gear_sonic/data/motions/x2_ultra_bones_seed.pkl` (~2550 SMPL→X2 retargeted clips, joblib-compressed, NOT standard pickle).
2. **Recipe DSL** — `gear_sonic/utils/planner/x2_recipes.py` + `gear_sonic/data/motions/x2_planner_primitives_recipes.yaml`. A bin is built from an ordered list of ops:
   - **Producers**: `clip_window`, `synthesize_waist_ramp`, `synthesize_crouch_ramp`, (deprecated) `synthesize_side_step_ramp`
   - **Transforms**: `freeze` (zero arms/head/etc to `DEFAULT_STAND_POSE`), `mirror_lr`, `scale_magnitude`, `recenter_root`, `pad_idle`, splice/blend
   - Bins can `derive_from` another bin — e.g. all `*_right_*` mirror their `*_left_*` sibling so we only hand-curate one side.
3. **Bin spec** — `gear_sonic/data/motions/x2_planner_bins.yaml` — per-bin metadata: target intent, target magnitude, target XY/yaw, tolerances, pelvis-z band, stride-count gates, frame-window range, name regex.

**Build pipeline:** `gear_sonic/scripts/build_x2_planner_primitives.py` reads recipes + source PKL → writes `gear_sonic/data/motions/x2_planner_primitives.pkl` (runtime artifact, currently 30 bins) + a Markdown report.

**Runtime:** `gear_sonic/utils/planner/state_machine.py`'s `HeuristicPlanner` loads the primitives PKL, resamples 30→50 Hz on load, yaw-aligns segments to the planner's current world pose at runtime, and blends seams. `gear_sonic/scripts/x2_heuristic_planner.py` is the executable that wraps it with ZMQ publisher + keyboard/scripted/ZMQ command sources.

**Wrapper:** `gear_sonic/scripts/run_planner_smoke.sh` spawns the planner + (optionally) `deploy_x2.sh sim --vla` + (optionally) the live MuJoCo viewer, all under a trap-based cleanup.

### Bins shipped (30 total)

| Family | Bins |
|---|---|
| Idle | `idle_stand` |
| Continuous walk | `fwd_walk_standard`, `back_walk_standard` |
| Forward/back steps | `fwd_step_1ft` (parent), `fwd_step_half_ft`, `fwd_step_quarter_ft`, `back_step_half_ft` (parent), `back_step_quarter_ft` |
| Side steps | `side_left_step`, `side_right_step` (collapsed from 4 variants → 2 canonical) |
| Pivot turns | `turn_{left,right}_{15,30,45,90}deg` (left = mocap; right = `mirror_lr`) |
| Torso twists (synthesized) | `torso_{left,right}_{15,30,45}deg` |
| Forward leans | `lean_fwd_{small,medium,large}` |
| Crouches (synthesized) | `crouch_{small,medium,large}` |

### Per-family source choices and rationale

- **`idle_stand`** — `loco__idle_vigilance_start_R_001__A502 [33:78]`. Tiny pelvis sway; carrier for the planner's IDLE_LOOP state.
- **`fwd_step_*` / `back_step_*` (v4)** — `walk_randdir_relax_001__A005 [0:68]`. Square→step→square→step→square. Net dx ≈ +0.20 m, yaw_osc 19.7°. Replaces the v3 source which was a one-stride excerpt from a continuous walk — its mid-stride staggered endpoints got chewed up by the runtime's 16-frame seam blend, so SONIC just shook in place.
- **`side_left_step` / `side_right_step`** — `loco__walk_sideway_045_stop_001__A038_M [0:215]` (the **full** 7.17 s clip). Earlier we used [0:90] but that 3 s window is essentially a knees-locked pelvis-slide (L knee span 3.8°, R knee 0.1°) so neither foot lifted under SONIC. The full clip contains real stepping in frames 60–175 (L knee span 12.9°, hip-pitch 35–40°, ~228 cm back-right diagonal travel).
- **Turn bins** — picked by the curator from `Step_Rotate_Reaction_Idle_*` clips, all scoring ≥ 0.6, hitting target heading ±1–3°.
- **Torso twists** — fully synthesized via `synthesize_waist_ramp(axis: yaw, peak_deg, ramp_in/hold/ramp_out)`; mocap had too few clean isolated-twist examples.
- **Crouches** — fully synthesized via `synthesize_crouch_ramp(peak_drop_m, ramp_in/hold/ramp_out)` with knee = 2 × hip and ankle = -hip so feet stay flat (geometrically self-consistent, no toe-pop).

### Locomotion-mining filters (added in v3)

`gear_sonic/scripts/mine_x2_motion_candidates.py`:
- `pelvis_yaw_osc_deg ≤ 40°` (55° for forward) per 1-second detrended window. The v2 walks had 60–80° yaw oscillation in the raw retarget; SONIC interpreted that as "stand-in-place wobble" instead of gait. After the filter, `fwd_walk_standard` walks 0.51 m/s under SONIC.
- Crouch-family: `foot_lift ≤ 7 cm` (was 4 cm — symmetric two-hand squats heel-pop briefly), and switched the foot-planted check to use a **baseline-stand foot Z reference** rather than the absolute clip floor — fixed the bug where standing frames were being marked "+5 cm lifted" because the ankle drops as the leg folds during a squat.

### Challenges and resolutions

#### Forward stepping looked fine kinematically but didn't move under SONIC

**Diagnosis:** High pelvis-yaw oscillation in the retarget. **Fix:** Yaw-osc filter in the miner (above) + replaced sources with the v3 picks.

#### `fwd_step_*` still looked like "shake without stepping" in v3

**Diagnosis:** v3 short-step source was a one-stride excerpt of continuous walk → endpoints were mid-stride staggered. **Fix:** Switched to `walk_randdir_relax_001__A005 [0:68]` which begins from a >0.6 s feet-side-by-side rest (v4).

#### Side stepping never worked from raw mocap

Tried in order, all failed:
- **Multiple mocap candidates** (`Sideway_Walk_Right_001__A017`, `walk_sideway_090_loop_001__A029_M`, etc.) → all exhibited "scissor" gait (trailing leg crossing behind body) and waist twist.
- **`--no-leg-back` clamp** (clip both legs' hip_pitch ≥ baseline in `bake_primitive_for_deploy.py`) → killed the scissor *and* the step.
- **`--waist-roll-deg` half-sine overlay** to load the support leg → policy ignored it, no movement.
- **Reducing the primer-momentum window** progressively → either still scissored or didn't move.
- **Synthesizing from scratch** via `synthesize_side_step_ramp` (mirror of crouch synthesis) — v1 had a half-sine hip-roll envelope (foot returned to start, no net translation); v2 with monotonic ramp envelope still didn't move because there was no waist counter-shift. User said "stop."

**What finally worked:** Stitched **all 99** side-walk clips from the bones-seed into a single PKL via `gear_sonic/scripts/stitch_side_walks_for_review.py` (with stand-pose pads + 0.3 s blends + a JSON manifest). Recorded MuJoCo with caption overlays via `gear_sonic/scripts/record_sonic_review_video.py` (uses `xwininfo` + `ffmpeg -f x11grab` + `drawtext`). User eyeballed the 99 clips and identified `walk_sideway_045_stop_001__A038_M` as the cleanest. Locked in.

#### MuJoCo camera didn't track the robot during the 99-clip review

The MuJoCo passive viewer defaults to a fixed free camera; over a multi-meter side-walk traversal the robot drifted out of frame. **Fix:** Added `--cam-track-body`, `--cam-distance`, `--cam-elevation`, `--cam-azimuth` to `gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py`; plumbed through `gear_sonic_deploy/deploy_x2.sh` (`--sim-cam-track-body` etc.) and `record_sonic_review_video.py` and `run_planner_smoke.sh`. Default tracks `pelvis`.

#### Robot launching from height instead of standing on the floor

Two distinct bugs hit at different times:

1. **Pure kinematic viewer** (`view_x2_planner_mujoco.py`) pinned `pelvis_z` to a hardcoded `DEFAULT_PELVIS_Z_M = 0.788 m`, ignoring each clip's own Z → robot floated. **Fix:** Use the clip's own `root_trans[:, 2]`. Also, kinematic viewer is `mj_forward`-only — no gravity, no contact resolution; it's a "do the joints look right" tool, not a "see the robot walk" tool. Use `deploy_x2.sh sim` for the latter.

2. **Under SONIC**, the bridge spawns at `z = 0.85 m` by default (`_reset_to_default_pose()` in `x2_mujoco_ros_bridge.py` deliberately adds slack for the elastic band) and the band auto-releases 1 s after the first deploy command. The deploy works fine *if nothing is on the wire during boot* — the policy uses its internal `default_angles` target, the band drops the robot gently. **The browse path (`browse_x2_planner_primitives.py --with-sonic`) was rewritten to mirror `record_x2_dataset.sh` exactly:**
   - Spawn `deploy_x2.sh` first
   - Wait for `Launching ...` marker in the deploy log
   - Sleep 2 s settle
   - **Only then** start publishing — and only with frozen `DEFAULT_STAND_POSE` + identity quat until the user presses `N` to fire a primitive.

   The smoke runner (`run_planner_smoke.sh`) **still spawns the planner first**, so it streams `idle_stand` (with sway/drift) before deploy finishes booting → policy tries to track a moving reference while the band releases → drop-from-height. **Open item — not yet ported into the smoke runner.**

#### Robot spinning on band release

The browse path used to publish a quat derived from observed yaw during boot pump. The bridge spawned at yaw ≈ −97° (because of `gantry_hang` profile or just because of how X2 Ultra's MJCF orients), our publisher said yaw = 0°, the policy slammed a 97° rotation correction during the band drop. **Fix:** Mirrored `x2_dataset_recorder._publish_idle`: identity quat everywhere idle, no observed-yaw tracking during boot/warmup/pre-launch.

#### `hold_last_pose` between motions

User wanted pauses between primitives in scripted demos that hold the *last frame* of the prior primitive, not blend back to `idle_stand`. **Implementation:**
- New `PlannerState.HOLDING` enum value.
- `LocomotionCommand.duration_s` field; `intent="hold_last_pose"` resolves to a `HOLDING` segment.
- `_start_hold()` snapshots the last emitted frame; `_ActiveSegment.is_hold = True`, `hold_total_frames` set.
- Indefinite hold (`duration_s == 0`): rewinds the cursor each tick so the same frame re-emits forever, gets interrupted on next command.
- Finite hold: counts down via `remaining`.
- `commands_from_yaml()` interprets `hold_seconds` on `hold_last_pose` as `duration_s` (instead of expanding to N idle commands like it does for `intent: idle`).
- Keyboard binding `h` in `x2_heuristic_planner.py`.
- 3 new unit tests (indefinite, finite, yaml-roundtrip) in `tests/test_x2_planner_seam_continuity.py`.

#### Six-motion smoke sequence

`gear_sonic/data/scripted_demos/six_motion_smoke.yaml` — `idle 2s → fwd_step + hold 1s → side_left + hold 1s → turn_left_45 + hold 1s → side_right + hold 1s → turn_right_45 + hold 1s → back_step + hold 1.5s`. `run_planner_smoke.sh` extended with `--model`, `--sim-cam-track-body`, etc., to drive end-to-end through `deploy_x2.sh sim --vla`.

### Tooling built along the way

| Script | Purpose |
|---|---|
| `gear_sonic/scripts/build_x2_planner_primitives.py` | Build runtime PKL from recipes YAML + source library. |
| `gear_sonic/scripts/curate_x2_primitives.py` | Legacy curator (score → best-clip per bin); largely superseded by recipes. |
| `gear_sonic/scripts/mine_x2_motion_candidates.py` | Source-clip miner with pelvis-yaw-osc / foot-lift / planted-foot filters. |
| `gear_sonic/scripts/browse_x2_planner_primitives.py` | Interactive browser for kinematic + SONIC playback of bins / raw windows. The "drops nicely on the floor" path. |
| `gear_sonic/scripts/bake_primitive_for_deploy.py` | Bake one bin (or ad-hoc source window) into a `deploy_x2.sh --motion <pkl>`-compatible PKL. Bypasses the ZMQ wire. |
| `gear_sonic/scripts/stitch_side_walks_for_review.py` | Concatenate all 99 side-walk clips with pads + blends + manifest. |
| `gear_sonic/scripts/record_sonic_review_video.py` | Spawn SONIC playback + `xwininfo` + `ffmpeg -f x11grab` + caption overlay. |
| `gear_sonic/scripts/x2_heuristic_planner.py` | The planner executable. |
| `gear_sonic/scripts/run_planner_smoke.sh` | Wrapper: planner + (optional) deploy + (optional) viewer + cleanup trap. |

### Test surface (27 tests, all passing)

- `tests/test_x2_planner_curator.py` — required-bins-exist gate, schema validation. Updated when we collapsed side bins to 2.
- `tests/test_x2_planner_seam_continuity.py` — state machine: bin-name resolution per intent/magnitude, blend continuity, hold semantics (3 new), command-queue interruption.

### Format gotchas

- `x2_ultra_bones_seed.pkl` is `joblib`-compressed (not standard pickle) → use `joblib.load()`.
- `bones-seed['<key>']['dof']` is a 2-D `np.ndarray` of shape `(T, 31)`, not a tuple.
- `x2_planner_primitives.pkl` schema uses `root_trans` (not `root_trans_offset`); deploy PKLs use `root_trans_offset` (not `root_trans`). `bake_primitive_for_deploy.py` does the rename.
- `bake_primitive_for_deploy.py`'s `bin_name` is **positional**, not a flag — `python -m gear_sonic.scripts.bake_primitive_for_deploy fwd_step_half_ft`, not `--bin-name fwd_step_half_ft`.

### Open items (paused on)

1. **Smoke runner boot order** — port the `record_x2_dataset.sh` boot pattern (deploy first → wait `Launching ...` → 2 s settle → publish DEFAULT_STAND_POSE + identity quat) into `run_planner_smoke.sh`. Browse path is fixed; smoke runner is not, hence the height-drop user observed.
2. **Stitched 6-motion PKL** — build a single deploy PKL of `fwd_step → hold → side_left → hold → turn_left_45 → hold → side_right → hold → turn_right_45 → hold → back_step` so the user can verify side-step foot lifts under the proven `deploy_x2.sh sim --motion <pkl>` (`PklMotionReference` + RSI) path, bypassing ZMQ entirely. Helpers (`bake_one`, `_splice_blend`, `build_blend_window`, `yaw_align_segment`) already exist in `bake_primitive_for_deploy.py` and `gear_sonic/utils/planner/blending.py`; just needs wiring.
