# Sim-to-Real Recordings Inventory

**Date:** 2026-05-04
**Purpose:** Preserve every real-robot deploy run we have on disk and the deploy
command that produced each one, so that when we get back to the robot we can
(a) reproduce the same conditions, and (b) generate a matching sim-side
recording for sim-to-real overlay analysis.

**Why this matters:** Today the recorder does not bake the playlist or ONNX
checkpoint into the `meta_json` blob -- recordings are self-describing only on
host/timing. Without this doc, mapping a `run.npz` back to "what the policy was
trying to do" requires bash-history archaeology, and bash history will roll
over.

---

## 1. Recorder npz schema

Every `run.npz` produced by `deploy_x2.sh --record` (real or sim mode) has the
same schema, written by
[`gear_sonic_deploy/scripts/x2_record_real_run.py`](../../../gear_sonic_deploy/scripts/x2_record_real_run.py):

| Block | Keys | Shape | Source topic | Rate |
|---|---|---|---|---|
| Commanded targets (per limb group) | `cmd_pos_<group>`, `cmd_vel_<group>`, `cmd_kp_<group>`, `cmd_kd_<group>` + `t_cmd_<group>` | `(N_cmd, dof)` | `/aima/hal/joint/<group>/command` | ~500 Hz |
| Measured state (per limb group) | `state_pos_<group>`, `state_vel_<group>`, `state_eff_<group>` + `t_state_<group>` | `(N_state, dof)` | `/aima/hal/joint/<group>/state` | ~1 kHz |
| Joint names (per limb group) | `joint_names_<group>` | `(dof,)` object | from `aimdk_msgs` | static |
| IMU | `imu_quat_wxyz`, `imu_angvel`, `imu_linacc` + `t_imu` | `(N_imu, {3,4})` | `/aima/hal/imu/torso/state` | ~500 Hz |
| MC mode trace (newer runs) | `mc_mode_str`, `t_mc_mode` | `(N_mc,)` object | `/aima/mc/mode` | ~event |
| Meta | `meta_json` | scalar string | hostname + timing only | -- |

`<group>` is one of `arm` (14 DoF), `head` (2 DoF), `leg` (12 DoF), `waist`
(3 DoF). Total = 31 DoF for X2 Ultra.

### What `meta_json` does **not** capture (recorder hole)

```json
{
  "out_path": "...", "duration_s_requested": 0.0, "duration_s_actual": 84.93,
  "started_at_wall": ..., "started_at_iso": "2026-05-04T00:41:54",
  "hostname": "ai-gym", "ros_domain_id": 0, "ros_localhost_only": 0,
  "rmw_implementation": null, "imu_topic": "/aima/hal/imu/torso/state",
  "git_sha": null, "note": "deploy_x2.sh local @ ..."
}
```

Missing (worth fixing in `deploy_x2.sh` next time at the robot):
- `--model` path / ONNX SHA256
- `--motion` / playlist path
- `--tuning-config` path
- `--target-lpf-hz`, `--max-target-dev`, `--ramp-seconds`, `--tilt-cos`
- `mode` (sim vs local vs onbot)

Until that's fixed, this document is the source of truth for command
provenance.

---

## 2. Recordings inventory (27 runs on disk)

All runs live under `scratch/runs/` in the repo. Sizes and durations come
from the `imu` time-series; the durations include WAIT_FOR_CONTROL,
CONTROL, RAMP_OUT, and (for newer runs) HOLD_FOR_MC.

### iter-4k era (mesh-warm-start ONNX, May 2)

| Run dir | Date | Duration | Hist# | Checkpoint | Playlist | Notable flags |
|---|---|---|---|---|---|---|
| `minv1_20260502_094743` | 5/2 09:48 | 12 s | 1918-1921 | iter-4k | `minimal_v1.yaml` | 5 s control, autostart 5, dev 0.30, ramp 2.0 |
| `minv1_full_20260502_094935` | 5/2 09:49 | 20 s | 1922 | iter-4k | `minimal_v1.yaml` | (same) |
| `minv1_full_20260502_095337` | 5/2 09:55 | 38 s | 1923 | iter-4k | `minimal_v1.yaml` | **12 s control** longer test |
| `gestures_expressive_20260502_102121` | 5/2 10:21 | 30 s | 1925 | iter-4k | `standing_gestures_v1.yaml` | `expressive.yaml`, **no LPF** |
| `gestures_lpf5_20260502_102753` | 5/2 10:27 | 30 s | 1926/1927 | iter-4k | `standing_gestures_v1.yaml` | `expressive.yaml` + `--target-lpf-hz 5` |

Pre-`stand_default` handoff era. Recordings have no `mc_mode_str` field.
Tuning config introduced at #1925.

### iter-16k era (sphere-feet, May 3 morning + day)

All used `--tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml`,
`--max-target-dev 1.80`, `--target-lpf-hz 5.0`, `--max-duration 22`,
default `--log-dir`. Most are repeated tuning iterations on the standing
gesture playlist; useful as a *consistency band* for that motion.

| Run dir | Date | Duration | Hist# | Checkpoint | Playlist |
|---|---|---|---|---|---|
| `x2_run_20260503_053447` | 5/3 05:34 | 30 s | 1945+ | iter-16k | `standing_gestures_v1.yaml` |
| `x2_run_20260503_184204` | 5/3 18:42 | 30 s | 1947-1955 cluster | iter-16k | `standing_gestures_v1.yaml` |
| `x2_run_20260503_192038` | 5/3 19:20 | 39 s | (same cluster) | iter-16k | `standing_gestures_v1.yaml` |
| `x2_run_20260503_193517` | 5/3 19:35 | 37 s | (same cluster) | iter-16k | `standing_gestures_v1.yaml` |
| `x2_run_20260503_200208` | 5/3 20:02 | 37 s | (same cluster) | iter-16k | `standing_gestures_v1.yaml` |
| `x2_run_20260503_203205` | 5/3 20:32 | 67 s | extended | iter-16k? | gestures (`--max-duration` bumped) |
| `x2_run_20260503_204158` | 5/3 20:41 | 164 s | extended | iter-16k? | gestures (long-form) |
| `x2_run_20260503_205242` | 5/3 20:52 | 43 s | handoff-debug | iter-16k? | gestures (this is the **5-second window** handoff incident; useful for §4) |
| `x2_run_20260503_213002` | 5/3 21:30 | 38 s | tuning-sweep | iter-?? | gestures |
| `x2_run_20260503_213810` | 5/3 21:38 | 40 s | tuning-sweep | iter-?? | gestures |
| `x2_run_20260503_214804` | 5/3 21:48 | 36 s | tuning-sweep | iter-?? | gestures |
| `x2_run_20260503_214932` | 5/3 21:49 | 42 s | tuning-sweep | iter-?? | gestures |
| `x2_run_20260503_220054` | 5/3 22:00 | 24 s | aborted/short | -- | gestures (likely cancelled) |

### iter-22k era (May 3 evening)

| Run dir | Date | Duration | Hist# | Checkpoint | Playlist | Verdict |
|---|---|---|---|---|---|---|
| `x2_run_20260503_221706` | 5/3 22:17 | 29 s | warmup | iter-22k | gestures (pre-walk) | warmup |
| `x2_run_20260503_221800` | 5/3 22:18 | 37 s | warmup | iter-22k | gestures | warmup |
| `x2_run_20260503_221845` | 5/3 22:18 | 31 s | warmup | iter-22k | gestures | warmup |
| **`x2_run_20260503_222045`** | **5/3 22:20** | **36 s** | **#1972** | **iter-22k** | **`x2_ultra_casual_walk_v1.pkl`** | **first powered walk** |
| **`x2_run_20260503_231753`** | **5/3 23:17** | **125 s** | **#1978** | **iter-22k** | **`x2_ultra_showcase_v1.pkl`** | **showcase reel for the overlords** |

### Post-1981 era (recordings made after the bash-history paste; provenance TBD)

The bash history shared on 2026-05-04 stopped at command #1981. The
following recordings were made after that and are still on disk -- their
checkpoint and playlist have to be filled in from memory or by re-running
`history` on the host.

| Run dir | Date | Duration | Likely contents |
|---|---|---|---|
| `x2_run_20260503_233140` | 5/3 23:31 | 43 s | `walk_demo` iteration on iter-22k |
| `x2_run_20260503_235329` | 5/3 23:53 | 51 s | `walk_demo` iteration on iter-22k |
| `x2_run_20260504_004004` | 5/4 00:40 | 61 s | `walk_demo_v6` or `one_foot_v1` final |
| `x2_run_20260504_004150` | 5/4 00:41 | 84 s | likely `walk_demo_v6` final |

**Action item:** when next at the host, run `history | grep 'deploy_x2.sh local'`
and capture commands #1982 onward into this doc.

---

## 3. Three certain anchors (use these for sim-to-real comparison first)

These three have unambiguous mapping between disk artifact and the deploy
command used to produce them. They are the highest-priority targets for
sim-to-real overlay figures in §7 of the paper.

### Anchor A -- iter-4k gestures (low-skill baseline)

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model  ~/x2_cloud_checkpoints/h200-iter-4000-20260501/model_step_004000.onnx \
    --motion ./gear_sonic/data/motions/playlists/standing_gestures_v1.yaml \
    --autostart-after 5 --max-duration 22 \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --target-lpf-hz 5 \
    --log-dir /workspace/sonic/scratch/runs/gestures_lpf5_$(date +%Y%m%d_%H%M%S) \
    --record
```

Disk artifact: `scratch/runs/gestures_lpf5_20260502_102753/run.npz` (5.7 MB,
30 s actual).

### Anchor B -- iter-22k first powered walk

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx \
    --motion ./gear_sonic/data/motions/x2_ultra_casual_walk_v1.pkl \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --max-target-dev 1.50 \
    --target-lpf-hz 5.0 \
    --max-duration 14 \
    --record
```

Disk artifact: `scratch/runs/x2_run_20260503_222045/run.npz` (5.7 MB, 36 s
actual).

### Anchor C -- iter-22k showcase reel

```bash
./gear_sonic_deploy/deploy_x2.sh local \
    --model $HOME/x2_cloud_checkpoints/h200-iter-22000-sphere-feet-20260501/model_step_022000.onnx \
    --motion ./gear_sonic/data/motions/x2_ultra_showcase_v1.pkl \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --max-target-dev 1.50 \
    --target-lpf-hz 5.0 \
    --max-duration 100 \
    --record
```

Disk artifact: `scratch/runs/x2_run_20260503_231753/run.npz` (20 MB, 125 s
actual).

---

## 4. Producing matching sim-side recordings

`deploy_x2.sh sim --record` writes the **same npz schema** as `local --record`,
so producing a sim counterpart for any anchor is just a mode swap. From
`deploy_x2.sh:397-411,1841-`, the recorder is mode-agnostic; the MuJoCo-ROS
bridge in sim mode republishes joint state and IMU on the same topic names
the recorder subscribes to.

### Recipe

```bash
# Anchor A sim counterpart
./gear_sonic_deploy/deploy_x2.sh sim \
    --sim-profile parity \
    --model  ~/x2_cloud_checkpoints/h200-iter-4000-20260501/model_step_004000.onnx \
    --motion ./gear_sonic/data/motions/playlists/standing_gestures_v1.yaml \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml \
    --target-lpf-hz 5.0 \
    --max-duration 22 \
    --log-dir /tmp/anchor_a_sim_$(date +%Y%m%d_%H%M%S) \
    --record
```

Notes:
- Use `--sim-profile parity` so the bridge does not apply gantry-band or
  ramp-modifying behaviours that the real run did not have.
- Do **not** use `--sim-viewer` if you want max-speed evaluation; only useful
  for visual inspection.
- Sim runs do not need `--autostart-after` -- the deploy node enters CONTROL
  immediately, which the comparison script will detect and crop.
- Sim duration tends to be cleaner than real because there is no MC handoff;
  the recorder will still write equivalent timestamps.

Bookkeeping: keep sim runs under `scratch/runs_sim/` (or pass `--log-dir
/tmp/...`) so they do not clutter the real-robot inventory.

---

## 5. Comparing real vs sim

Use the script
[`gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py`](../../../gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py).

```bash
python gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py \
    --real scratch/runs/x2_run_20260503_222045/run.npz \
    --sim  /tmp/anchor_b_sim_<...>/run.npz \
    --out  /tmp/sim_vs_real_anchor_b/ \
    --label-real "iter-22k real (casual_walk_v1)" \
    --label-sim  "iter-22k MuJoCo (casual_walk_v1)"
```

What the script does:

1. **Loads both npz** and verifies joint-name alignment across the four
   limb groups.
2. **Auto-detects the CONTROL window** by trimming everything before the
   first published joint command (`t_cmd_leg[0]`) and optionally
   trimming the tail (`--end-trim`) to drop RAMP_OUT and HOLD_FOR_MC.
3. **Resamples** both runs onto a common 50 Hz time grid (configurable
   via `--resample-hz`).
4. **Computes deltas:**
   - per-DoF tracking error `state_pos - cmd_pos`
   - per-DoF L2 over the run
   - sim-vs-real cmd diff and state diff (the actual transfer gap)
   - IMU base orientation (roll, pitch, yaw from quaternion)
5. **Writes:**
   - `summary.json`: numeric stats
   - `summary.txt`: human-readable table
   - `dof_pos_grid.png`: 31-DoF figure with cmd / state for real and sim
   - `dof_tracking_error.png`: 31-DoF figure of tracking errors
   - `dof_l2_bar.png`: per-DoF L2 bars (sim vs real)
   - `imu_overlay.png`: IMU traces (orientation + angular velocity)
   - `cmd_diff_heatmap.png`: time x DoF heatmap of `cmd_real - cmd_sim`

See `--help` for all flags.

---

## 5b. Anchor C sim-vs-real findings (showcase_v1 + iter-22k, 2026-05-04)

Real:  `scratch/runs/x2_run_20260503_231753/run.npz` (102.00 s CONTROL window)
Sim :  `scratch/runs/sim_anchor_c_20260504_195840/run.npz` (102.02 s CONTROL window,
       deploy_x2.sh sim --sim-profile parity, same model + motion + LPF + max-target-dev)
Compared via `gear_sonic_deploy/scripts/compare_sim_vs_real_npz.py`,
results in `scratch/sim_vs_real_results/anchor_c_iter22k_showcase/`.

Headline numbers:

| Metric | Value |
|---|---:|
| Real tracking error RMS (state - cmd) | 13.05 deg |
| Sim  tracking error RMS (state - cmd) | 14.09 deg |
| Sim-vs-real `state_pos` diff RMS | **3.88 deg** |
| Sim-vs-real `cmd_pos`   diff RMS | 8.34 deg |
| IMU angvel RMS real / sim | 0.186 / 0.197 rad/s |

Per-DoF / per-group breakdown:

- **Arms (14 DoFs)**: cmd ranges agree between real and sim within 0-15 deg per joint.
  State ranges agree within 0-7 deg. Upper-body kinematics are essentially feed-forward
  from the playlist, so both worlds execute nearly identically.
- **Waist (3 DoFs)**: cmd ranges within 5-10 deg, state within 0-18 deg. Notably
  `waist_pitch` is commanded 155-165 deg range in *both* worlds but neither produces
  more than 26 deg of state range -- the body just doesn't go there. This is a property
  of the motion file (over-reach), not a sim-to-real gap.
- **Head (2 DoFs)**: head_yaw cmd matched (65 vs 63 deg), state diverges (40 vs 20 deg
  -- sim was freer than real). head_pitch state was 0 deg on real (motor refused) vs
  13 deg in sim, again an artefact of the motion commanding 171 deg.
- **Legs (12 DoFs)**: most match within 10-20 deg cmd-range. Four DoFs diverge by
  20-45 deg: `left_hip_pitch` (105 vs 84), `right_hip_pitch` (76 vs 102),
  `right_knee` (78 vs 50), `right_ankle_roll` (72 vs 116). This is the expected
  closed-loop sim-to-real signature: the policy reacts to slightly different
  IMU/state observations and emits different leg balance corrections.

Verdict: showcase_v1 is essentially feed-forward upper-body, transfers very cleanly
(~4 deg per-DoF state diff over 100 s). The motion file itself contains physically
impossible angles for several joints (head_pitch ref reaches 171 deg, waist_pitch ref
reaches 166 deg, several shoulder DoFs ref-range exceeds 175 deg), and both sim and
real refuse to follow those over-reaches identically. Not a transfer bug.

---

## 6. Open questions / next time at the robot

1. Patch `deploy_x2.sh` and `x2_record_real_run.py` to bake `--model`,
   `--motion`, `--tuning-config`, and any obs/policy hashes into
   `meta_json`. Five-line change, eliminates this whole mapping doc going
   forward.
2. Recover provenance for `x2_run_20260503_233140`,
   `x2_run_20260503_235329`, `x2_run_20260504_004004`, and
   `x2_run_20260504_004150` from the host's bash history (commands #1982+).
3. Decide whether sim recordings should be checked into the repo
   alongside real ones, or just regenerated on demand. Current policy:
   real-only in `scratch/runs/`, sim under `/tmp/` for now.

---

## 7. Reference -- raw bash history (commands 1918-1978)

```
1918  ./gear_sonic_deploy/deploy_x2.sh local   --model ~/x2_cloud_checkpoints/h200-iter-4000-20260501/model_step_004000.onnx   --motion ./gear_sonic/data/motions/playlists/minimal_v1.yaml   --autostart-after 5 --max-duration 5   --max-target-dev 0.30 --ramp-seconds 2.0 --tilt-cos -0.3   --return-seconds 2.0   --log-dir /workspace/sonic/scratch/runs/minv1_$(date +%Y%m%d_%H%M%S)   --record
1919  (same)
1921-1922  (same)
1923  (same with --max-duration 12, --log-dir minv1_full)
1924  (--motion standing_gestures_v1.yaml --max-duration 30 -- log-dir minv1_full)
1925  (gestures + expressive.yaml, no LPF, log-dir gestures_expressive)
1926-1927  (gestures + expressive.yaml + --target-lpf-hz 5, log-dir gestures_lpf5)
1928,1930  (gestures + expressive.yaml + --max-target-dev 1.80, /tmp/gestures_dev180_*)
1933,1936  ./gear_sonic_deploy/deploy_x2.sh sim --sim-profile parity --model ...iter-10000... gestures (sim, log-dir /tmp/iter10k_sim_*)
1937  ./gear_sonic_deploy/deploy_x2.sh local ...iter-10000... gestures + expressive + dev1.80 + lpf5 (/tmp/iter10k_dev180_lpf5_*)
1940  ./gear_sonic_deploy/deploy_x2.sh sim --sim-profile parity --model ...iter-16000... gestures (sim, /tmp/iter16k_sim_*)
1941  ./gear_sonic_deploy/deploy_x2.sh local ...iter-16000... gestures + expressive + dev1.80 + lpf5 (/tmp/iter16k_dev180_lpf5_*)
1945-1956  iter-16k or iter-22k + gestures + expressive + dev1.80 + lpf5 (no --log-dir, scratch/runs/x2_run_*)
1957  ./gear_sonic_deploy/deploy_x2.sh sim --model ...iter-22000... --motion x2_ultra_relaxed_walk_postfix.pkl --sim-viewer --max-duration 30 --record
1972  ./gear_sonic_deploy/deploy_x2.sh local --model ...iter-22000... --motion x2_ultra_casual_walk_v1.pkl --tuning-config expressive.yaml --max-target-dev 1.50 --target-lpf-hz 5.0 --max-duration 14 --record   <-- ANCHOR B
1978  ./gear_sonic_deploy/deploy_x2.sh local --model ...iter-22000... --motion x2_ultra_showcase_v1.pkl --tuning-config expressive.yaml --max-target-dev 1.50 --target-lpf-hz 5.0 --max-duration 100 --record   <-- ANCHOR C
1981  history | grep record  (= when this paste was generated)
```
