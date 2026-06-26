# =============== do not auto edit this section =============== #

### start sonic on PC2 : Robogym Wifi
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

### stop sonic
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32

### run the pkl stack
./gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
    --pc2-host 192.168.86.32

### xbox controller
.venv/bin/python -m gear_sonic.scripts.play_xbox_controller

### play pkl motion
python -m gear_sonic.scripts.play_locomotion \
    --pkl gear_sonic/data/motions/x2_ultra_in_place_turns_v1_chain_matched.pkl

python -m gear_sonic.scripts.play_locomotion \
    --pkl gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl

python -m gear_sonic.scripts.play_locomotion \
    --pkl gear_sonic/data/motions/x2_ultra_relaxed_walk_two_right_turns_v1.pkl

python -m gear_sonic.scripts.play_locomotion \
    --pkl gear_sonic/data/motions/x2_ultra_relaxed_walk_one_left_turn_v1.pkl

python -m gear_sonic.scripts.play_locomotion \
    --pkl gear_sonic/data/motions/x2_ultra_relaxed_walk_one_right_turn_v1.pkl

python -m gear_sonic.scripts.play_locomotion --pkl gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1.pkl 

### run in kinematic viewer
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl \
    --no-loop

x2_ultra_relaxed_walk_loop_v1.pkl
x2_ultra_relaxed_walk_loop_v1_halfspeed_walks.pkl

### run with sonic in sim directly (without full stack piipeline)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/chain_matched_v2_iter_004000/exported/model_step_004000_g1.onnx  \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1.pkl \
    --sim-viewer --no-confirm \
    --max-duration 60

### run with sonic in sim and full stack
gear_sonic/scripts/run_x2_pkl_direct_stack.sh --model /home/stickbot/x2_cloud_checkpoints/chain_matched_v2_iter_004000/exported/model_step_004000_g1.onnx

python -m gear_sonic.scripts.play_locomotion --pkl gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl


### trained sonic models
M25K=/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx
M3339=/home/stickbot/x2_cloud_checkpoints/chain_matched_v2_iter_003339/exported/model_step_003339_g1.onnx
M4000=/home/stickbot/x2_cloud_checkpoints/chain_matched_v2_iter_004000/exported/model_step_004000_g1.onnx

-- base model
/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx
backed up as agibot_x2_sonic_base_version.onnx

-- good for walks
<!-- logs_rl/TRL_X2Ultra_DemoV1/manager/universal_token/all_modes/sonic_x2_ultra_demo_v1_demo_v1-20260623_231221/exported/model_step_004000_g1.onnx
copied as /home/run/getsolo/policies/agibot_x2_sonic.onnx -->
/home/stickbot/x2_cloud_checkpoints/chain_matched_v2_iter_004000/exported/model_step_004000_g1.onnx 
copied as
/home/run/getsolo/policies/agibot_x2_sonic.onnx

gear_sonic_deploy/deploy_x2.sh sim     --model logs_rl/TRL_X2Ultra_DemoV1/manager/universal_token/all_modes/sonic_x2_ultra_demo_v1_demo_v1-20260623_231221/exported/model_step_004000_g1.onnx     --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_halfspeed_walks_chain_matched.pkl     --sim-viewer --no-confirm     --max-duration 60

-- good for dances
logs_rl/TRL_X2Ultra_DemoV2/manager/universal_token/all_modes/sonic_x2_ultra_demo_v2_demo_v2-20260624_130315/exported/model_step_004000_g1.onnx 
copied as /home/run/getsolo/policies/agibot_x2_sonic_dance_fine_tuned.onnx


gear_sonic_deploy/deploy_x2.sh sim     --model logs_rl/TRL_X2Ultra_DemoV2/manager/universal_token/all_modes/sonic_x2_ultra_demo_v2_demo_v2-20260624_130315/exported/model_step_004000_g1.onnx     --motion gear_sonic/data/motions/dance_singles/dance_latino_kick_kick_R_001__A313.pkl     --sim-viewer --no-confirm     --max-duration 60

### trained models v2
M25K=/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx
M3339=/home/stickbot/x2_cloud_checkpoints/chain_matched_v2_iter_003339/exported/model_step_003339_g1.onnx
M4000=/home/stickbot/x2_cloud_checkpoints/chain_matched_v2_iter_004000/exported/model_step_004000_g1.onnx


##### v2 — walk forward -> turn 90 -> turn 90 -> walk back (~34.5 s; ends ~29 cm off origin)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_walk_demo_v6.pkl \
    --sim-viewer --no-confirm \
    --max-duration 40

gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1.pkl \
    --sim-viewer --no-confirm \
    --max-duration 60

gear_sonic_deploy/deploy_x2.sh sim \
    --model logs_rl/TRL_X2Ultra_DemoV1/manager/universal_token/all_modes/sonic_x2_ultra_demo_v1_demo_v1-20260623_231221/exported/model_step_004000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1.pkl \
    --sim-viewer --no-confirm \
    --max-duration 60

### gestures
gear_sonic/data/motions/x2_recorded/mc_gestures/left_kiss_001.pkl

### short listed walking motions for demo day
x2_ultra_in_place_turns_v1_chain_matched.pkl
x2_ultra_walk_demo_v6.pkl
x2_ultra_walk_demo_v6_chain_matched.pkl
x2_ultra_relaxed_walk_loop_v1.pkl
x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl
x2_ultra_relaxed_walk_loop_v1_halfspeed_walks_chain_matched.pkl
x2_ultra_relaxed_walk_loop_v1_halfspeed_walks.pkl


### short listed dance motions for demo day
easy - dance_singles/dance_hiphop_stick_n_roll_dancehall_R_loop_003__A324.pkl
medium - dance_singles/dance_western_horse_step_with_leg_undercut_R_loop_002__A324.pkl
hard - dance_singles/dance_latino_kick_kick_R_001__A313.pkl

gear_sonic_deploy/deploy_x2.sh sim \
    --model logs_rl/TRL_X2Ultra_DemoV2/manager/universal_token/all_modes/sonic_x2_ultra_demo_v2_demo_v2-20260624_130315/exported/model_step_004000_g1.onnx \
    --motion gear_sonic/data/motions/dance_singles/dance_latino_kick_kick_R_001__A313.pkl \
    --sim-viewer --no-confirm \
    --max-duration 15


### lower robot volume
./gear_sonic_deploy/scripts/x2_pc3_audio.sh volume 70 --pc2-host 192.168.86.32

# ================ end of manual edit section ===================== #

> **Demo motion snapshot dir**:
> [`gear_sonic/data/motions/live_demo/`](gear_sonic/data/motions/live_demo/README.md)
> holds tracked copies of every walking-motion PKL listed in the "short
> listed walking motions for demo day" section above. Use those paths
> (e.g. `gear_sonic/data/motions/live_demo/x2_ultra_relaxed_walk_loop_v1.pkl`)
> on a fresh clone, since the canonical
> `gear_sonic/data/motions/x2_ultra_*.pkl` files are gitignored as
> regenerable warehouse-stitcher outputs. The README also documents the
> demo policy slots on PC2 (`agibot_x2_sonic{,_base_version,_dance_fine_tuned}.onnx`)
> and which one to swap in for walks vs dances.

### relaxed-walk closed loop v1 — re-retargeted relaxed-walk + v6 turn primitives

PKL: `gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1.pkl` (1565 frames @ 30 fps = 52.17 s, single key `relaxed_walk__v1__home_loop`).

Choreography (closed loop, lands within (-0.21 m, +0.03 m, -2.1 deg) of origin):

1. anchor_open (1.0 s idle)
2. pivot_left_1 (+90 deg)
3. relaxed_walk_out (Relaxed_walk_forward_001__A057 frames [0..188), ~3.50 m, starts from true rest, ends at f187 inter-step double-support)
4. pivot_right_1 (-90 deg)
5. pivot_right_2 (-90 deg, cumulative -180 from prev — now facing home)
6. relaxed_walk_back (same trim, ~3.50 m back toward home tile)
7. pivot_left_2 (+90 deg, restores original heading)
8. anchor_close (1.0 s idle)

Trim was picked via FK foot-slide profile (see top-of-yaml table). Mid-stride heel-strike trims (e.g. f95) leave the actor with one foot 21 cm ahead at the rest seam, and the warehouse stitcher pins pelvis XY for 2.5 s while leg dofs SLERP to centered idle → ~30 cm of horizontal foot drag at each post-walk seam. Trimming at f187 (inter-step double-support, both feet planted side-by-side) drops the worst-foot slide to ~9.5 cm. Speed multipliers don't help — they're frame subsamples of the same gait, so the stance at any given gait phase is identical.

All sources point at the new re-retarget bundle `gear_sonic/data/motions/x2_ultra_retarget_uniform_h14.pkl` (uniform h=1.4 SOMA scaler track, 48 entries = 16 clips × 3 speeds). The chain-matched alternative is at `gear_sonic/data/motions/x2_ultra_retarget_chain_matched.pkl` (rebuilt 2026-06-24 with the committed `soma_to_x2_ultra_chain_matched_retargeter_config.json` — the earlier `_v4` artifact had a deep-crouch retarget bug and is gone); swap the `source:` lines in the YAML to A/B the two tracks.

#### kinematic viewer (no anchor: see closure)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1.pkl \
    --no-loop

#### sonic in sim
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1.pkl \
    --sim-viewer --no-confirm \
    --max-duration 50

#### rebuild from the YAML (e.g. after editing trim, swapping retarget track, or picking a different WINNER variant)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/relaxed_walk_loop_v1.yaml \
    --out      gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1.pkl \
    --check-runtime-parity

#### preview every retargeted Relaxed_walk_forward variant (60 viewer windows: 2 tracks × 10 variants × 3 speeds) before committing the WINNER
bash ~/relaxed_walk_retarget_build/preview_windows.sh

### relaxed-walk closed loop v1 — half-speed walks variant

Same choreography as v1, but the two relaxed-walk segments play at 0.5x (turns + idles stay at 1.0x). Useful for letting the SONIC policy track each step over twice as many frames; same stride length, same closure, same foot slide — only per-frame pelvis velocity changes (~0.28 m/s vs ~0.56 m/s during walks).

PKL: `gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_halfspeed_walks.pkl` (1941 frames @ 30 fps = 64.70 s, single key `relaxed_walk__v1_halfspeed_walks__home_loop`). Loop closes at (-0.20 m, +0.04 m, -1.8 deg).

#### kinematic viewer
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_halfspeed_walks.pkl \
    --no-loop

#### sonic in sim (--max-duration bumped to 68 to clear the longer reel)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_halfspeed_walks.pkl \
    --sim-viewer --no-confirm \
    --max-duration 68

#### rebuild from the YAML
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/relaxed_walk_loop_v1_halfspeed_walks.yaml \
    --out      gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_halfspeed_walks.pkl \
    --check-runtime-parity

### relaxed-walk v1 — split into walk-forward + two-right-turns (2026-06-26)

Two standalone PKLs carved out of the parent `relaxed_walk_loop_v1.yaml` so each motion can be played on its own via `play_locomotion`. Same `uniform_h14` retarget track, same track-matched rest seams, same primitives — just smaller subsets wrapped in the same neutral_idle anchors the parent uses.

| Variant | PKL | Frames @ 30 fps | Duration | End XY | End yaw | XY path | Worst seam max\|d_dof\|/tick |
|---|---|---|---|---|---|---|---|
| walk forward only | `x2_ultra_relaxed_walk_forward_v1.pkl` | 398 | 13.27 s | (+3.36, +0.98) m | -0.4 deg | 3.58 m | 0.027 rad (walk -> rest) |
| two right turns (about-face) | `x2_ultra_relaxed_walk_two_right_turns_v1.pkl` | 595 | 19.83 s | (+0.02, -0.13) m | +177.2 deg (= -182.8, about-face -2.8 deg drift) | 0.84 m | 0.011 rad |

Single-key bundles: `relaxed_walk__v1__walk_forward` and `relaxed_walk__v1__two_right_turns`. Both built with the 2026-06-24 track-matched rest source `x2_ultra_stitched_idle_relaxed_arms_uniform_h14.pkl` (no "snap to default pose" between anchors and the motion segment).

The two-right-turns PKL gets much cleaner seams (0.010-0.011 rad) than the walk-forward PKL (0.027 rad on the walk-end seam) because both pivots start and end in idle pose, while the walk-end always leaves the legs mid-gait relative to idle stance. That 0.027 rad is the same value the parent loop hits at the same seam — it's the foot-slide cost the parent's YAML header comment calls out.

#### kinematic viewer
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl \
    --no-loop

conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_two_right_turns_v1.pkl \
    --no-loop

#### sonic in sim (--max-duration sized to ~1 s margin)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl \
    --sim-viewer --no-confirm \
    --max-duration 15

gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_two_right_turns_v1.pkl \
    --sim-viewer --no-confirm \
    --max-duration 22

#### real-robot playback
python -m gear_sonic.scripts.play_locomotion --pkl gear_sonic/data/motions/x2_ultra_relaxed_walk_forward_v1.pkl

python -m gear_sonic.scripts.play_locomotion --pkl gear_sonic/data/motions/x2_ultra_relaxed_walk_two_right_turns_v1.pkl

#### rebuild both from the YAMLs (e.g. after editing trim or swapping retarget track)
for NAME in relaxed_walk_forward_v1 relaxed_walk_two_right_turns_v1 ; do
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
      --playlist gear_sonic/data/motions/playlists/${NAME}.yaml \
      --out      gear_sonic/data/motions/x2_ultra_${NAME}.pkl \
      --check-runtime-parity
done

For a `chain_matched` variant: copy each YAML to `<name>_chain_matched.yaml`, swap `x2_ultra_retarget_uniform_h14.pkl` -> `x2_ultra_retarget_chain_matched.pkl` and the rest source to `x2_ultra_stitched_idle_relaxed_arms_chain_matched.pkl`, then rebuild.

### relaxed-walk closed loop v1 — chain_matched retarget A/B  (rebuilt 2026-06-24)

Direct counterparts of the two uniform_h14 builds above, but sourced from the committed chain-matched SOMA scaler track instead of the whole-body @ h=1.4 track. Same trim windows and segment shapes; the only delta in each YAML is the source PKL filename. Useful for A/B-ing foot-floor contact and hand rest height between the two retargeters.

History note: an earlier build dated 2026-06-23 sourced from a scratch `chain_matched_v4` scaler that uniformly shortened every leg-chain joint by 33% — visually that produced a Groucho-Marx permanent ~70° knee crouch (pelvis Z floored at 0.509 m). The committed `soma_to_x2_ultra_chain_matched_config.json` replaces those uniform 0.6732 leg scales with per-segment ratios that match the X2 Ultra leg geometry (LeftLeg=0.541, LeftShin=0.81, LeftFoot=0.75, LeftToe=0.78, LeftToeBase=0.66). The rebuilt PKLs below have pelvis Z back to 0.634-0.681 m and walks back to 3.50 m, in line with uniform_h14.

| metric (over full 1565-frame loop) | uniform_h14, 1.0x | chain_matched, 1.0x (NEW) | chain_matched_v4, 1.0x (OLD, deleted) |
|---|---|---|---|
| pelvis_z min / mean / max | 0.591 / 0.639 / 0.673 m | **0.634 / 0.669 / 0.681 m** | 0.509 / 0.565 / 0.673 m |
| L knee flex mean (max) | +0.18 rad / +1.29 (10° / 74°) | **+0.41 rad / +1.60 (23° / 92°)** | +1.21 rad / +2.01 (70° / 115°) |
| per-walk distance | 3.50 m | **3.50 m** | 2.89 m |
| max foot slide at post-walk seam | L=0.110 R=0.031 m | **L=0.133 R=0.037 m** | (similar magnitude but on a crouched stance) |
| loop yaw closure | -2.1° | **-9.7°** | -11.1° |
| loop XY closure | (-0.21, +0.03) m | **(-0.47, -0.01) m** | similar |

The chain_matched track still has a slightly worse loop closure (-9.7° yaw vs -2° for uniform_h14) and a slightly larger left-foot drift at the post-walk seam (13 cm vs 11 cm). The pelvis stance is correct (no more crouch), so the visible delta is now legitimately just per-chain scaling differences (which arm/leg ratios are preserved at the cost of yaw drift in the gait). If you need cleaner closure for chain_matched specifically, re-run the FK inter-step sweep against this new retarget bundle and re-trim `n_frames`.

| Variant | PKL | Total | Loop XY | End yaw |
|---|---|---|---|---|
| chain, 1.0x walks | `x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl` | 52.17 s | (-0.47, -0.01) m | -9.7° |
| chain, 0.5x walks | `x2_ultra_relaxed_walk_loop_v1_halfspeed_walks_chain_matched.pkl` | 64.70 s | (-0.47, -0.01) m | -9.6° |

#### kinematic viewer (chain_matched, 1.0x walks)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl \
    --no-loop

#### kinematic viewer (chain_matched, 0.5x walks)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_halfspeed_walks_chain_matched.pkl \
    --no-loop

#### sonic in sim (chain_matched, 1.0x walks)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_chain_matched.pkl \
    --sim-viewer --no-confirm \
    --max-duration 55

#### sonic in sim (chain_matched, 0.5x walks)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_relaxed_walk_loop_v1_halfspeed_walks_chain_matched.pkl \
    --sim-viewer --no-confirm \
    --max-duration 68

#### rebuild both chain_matched stitched variants (assumes the chain_matched bundle PKL is fresh)
for VAR in relaxed_walk_loop_v1_chain_matched relaxed_walk_loop_v1_halfspeed_walks_chain_matched ; do
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/${VAR}.yaml \
    --out      gear_sonic/data/motions/x2_ultra_${VAR}.pkl \
    --check-runtime-parity
done

### walk_demo_v6 rebuilt on the new retarget bundles (2026-06-24)

Re-retargeted counterparts of `x2_ultra_walk_demo_v6.pkl`. Same choreography, same trim windows, same n_frames -- only the source PKL differs (each segment now points at one of the new uniform_h14 / chain_matched bundles instead of the older `x2_ultra_bones_seed.pkl`). Useful when you want the v6 4-pivot home-loop on the cleaner new retargets.

| PKL | duration | XY closure | yaw closure | per-walk dist | pelvis_z mean | L knee max |
|---|---|---|---|---|---|---|
| `x2_ultra_walk_demo_v6.pkl` (original) | 44.17 s | 0.13 m | +11.2° | ~1.05 m | 0.677 m | +19.3° |
| `x2_ultra_walk_demo_v6_uniform_h14.pkl` (NEW) | 44.17 s | **0.10 m** | +4.6° | ~0.85 m | 0.644 m | +65.3° |
| `x2_ultra_walk_demo_v6_chain_matched.pkl` (NEW) | 44.17 s | **0.11 m** | **+2.4°** | ~0.90 m | 0.672 m | +82.5° |

The chain_matched variant has the **tightest yaw closure of any walk loop in the repo** (+2.4°). Per-walk distance is slightly shorter than the original (~0.85-0.90 m vs 1.05 m) because the new retargets scale stride length differently. Pose delta from the original: the new variants have more anatomically correct knee flex during stride (peak 65-82° vs the original's 19° — the original retarget kept the robot quite upright; the new retargets honor the source human's knee swing more faithfully).

#### kinematic viewer (uniform_h14)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_walk_demo_v6_uniform_h14.pkl \
    --no-loop

#### kinematic viewer (chain_matched -- best closure of any loop)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_walk_demo_v6_chain_matched.pkl \
    --no-loop

#### sonic in sim (either)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_walk_demo_v6_chain_matched.pkl \
    --sim-viewer --no-confirm \
    --max-duration 47

#### rebuild both v6 variants from the YAMLs
for VAR in walk_demo_v6_uniform_h14 walk_demo_v6_chain_matched ; do
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/${VAR}.yaml \
    --out      gear_sonic/data/motions/x2_ultra_${VAR}.pkl \
    --check-runtime-parity
done

### sit/stand chain_matched motions @ 1.0x + 0.5x (2026-06-24)

Three chair sit/stand primitives, retargeted on the chain_matched track and packaged at two playback speeds. Frame-decimation only (no interpolation): the source BVH is 120 fps; the builder is told `--fps-source 120` (real rate) for 1.0x and `--fps-source 60` (the lie) for 0.5x, then emits the 30 fps PKL. Keys are suffixed `__speed_1.0` / `__speed_0.5` to match the walks bundle naming convention.

PKL: `gear_sonic/data/motions/x2_ultra_sitstand_chain_matched.pkl` (6 entries, fps=30)

| key | frames | duration | pelvis_z (min → max) | semantics |
|---|---|---|---|---|
| `sit_on_chair_start_R_001__A244__speed_1.0` | 142 | 4.73 s | 0.391 → 0.681 m | sit DOWN onto chair (real-time) |
| `sit_on_chair_start_R_001__A244__speed_0.5` | 283 | **9.43 s** | 0.391 → 0.681 m | sit DOWN onto chair (half speed) |
| `sit_on_chair_loop_R_002__A244__speed_1.0`  | 274 | 9.13 s | 0.388 → 0.389 m | sitting STILL on chair (real-time) |
| `sit_on_chair_loop_R_002__A244__speed_0.5`  | 548 | **18.27 s** | 0.388 → 0.389 m | sitting STILL on chair (half speed) |
| `sit_on_chair_stop_R_002__A244__speed_1.0`  | 80  | 2.67 s | 0.392 → 0.682 m | stand UP from chair (real-time) |
| `sit_on_chair_stop_R_002__A244__speed_0.5`  | 160 | **5.33 s** | 0.392 → 0.682 m | stand UP from chair (half speed) |

#### preview each motion in the kinematic viewer
for KEY in \
    sit_on_chair_start_R_001__A244__speed_1.0 \
    sit_on_chair_start_R_001__A244__speed_0.5 \
    sit_on_chair_loop_R_002__A244__speed_1.0  \
    sit_on_chair_loop_R_002__A244__speed_0.5  \
    sit_on_chair_stop_R_002__A244__speed_1.0  \
    sit_on_chair_stop_R_002__A244__speed_0.5  ; do
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_sitstand_chain_matched.pkl \
    --motion-key $KEY \
    --no-loop
done

#### preview the source BVH side-by-side with the chain_matched retarget (for retarget QA — runs in the SOMA project's uv .venv)
cd $HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter
CHAIN=scratch/sitstand_chain_matched
BVH=../bones-seed/extracted_sitstand/soma_uniform/bvh/230306
for CLIP in sit_on_chair_start_R_001__A244 sit_on_chair_loop_R_002__A244 sit_on_chair_stop_R_002__A244 ; do
  uv run python app/play_csvs_with_human.py \
    --csv-a $CHAIN/${CLIP}__x2_chain_matched.csv --label-a chain_matched \
    --csv-b $CHAIN/${CLIP}__x2_chain_matched.csv --label-b chain_matched \
    --bvh   $BVH/${CLIP}.bvh
done

#### rebuild both speeds and merge them into one bundle
for SPEED in 1.0 0.5 ; do
  case $SPEED in 1.0) FPS_SRC=120 ;; 0.5) FPS_SRC=60 ;; esac
  mkdir -p $HOME/sitstand_retarget_build/build_speed_${SPEED}/
  conda run -n env_isaaclab --no-capture-output python \
      gear_sonic/data_process/build_x2_bones_seed_motion_lib.py \
      --retargeted-root agibot-x2-references/soma-retargeter/scratch \
      --subsets sitstand_chain_matched \
      --out-dir $HOME/sitstand_retarget_build/build_speed_${SPEED}/ \
      --fps-source $FPS_SRC --fps 30 --workers 4
done
conda run -n env_isaaclab --no-capture-output python - <<'PY'
import joblib
from pathlib import Path
BUILD = Path('/home/stickbot/sitstand_retarget_build')
merged = {}
for speed in (1.0, 0.5):
    src = joblib.load(BUILD / f'build_speed_{speed}' / 'x2_ultra_sitstand_chain_matched.pkl')
    for k, v in src.items():
        merged[f'{k.replace("__x2_chain_matched", "")}__speed_{speed}'] = v
joblib.dump(merged, 'gear_sonic/data/motions/x2_ultra_sitstand_chain_matched.pkl', compress=3)
print(f'Saved {len(merged)} entries')
PY

Notes:
- For a 0.25x build add `--fps-source 30` (no decimation) → ~565 / 1095 / 320 frames (4x slower than real-time). Not bundled yet — add if needed.
- This PKL doesn't yet appear in any stitching playlist. To use the sit → loop → stand sequence inside a `make_warehouse_motion.py` reel, add a YAML with three segments pointing at this PKL and the three keys, with a short or zero `rest_frames` between them (the loop segment already holds the seated pose).

### in-place turns v1 — R(-90) → L(+90) → L(+90) → R(-90) (2026-06-24)

Pure turn-only choreography on the `chain_matched` track. No walks — just the two pivot primitives from `walk_demo_v6_chain_matched` (idle_turn_270 = +90° left, step_rotate_idle_090 = -90° right) arranged so cumulative yaw closes back to start heading.

PKL: `gear_sonic/data/motions/x2_ultra_in_place_turns_v1_chain_matched.pkl` (single key `in_place_turns__v1__chain_matched__RLLR`, 1039 frames @ 30 fps = **34.63 s**)

| segment | source key | duration | per-segment yaw |
|---|---|---|---|
| anchor_open  | `neutral_idle_loop_002__A074__speed_1.0` | 1.00 s | hold 0° |
| pivot_right_1 | `step_rotate_idle_090_002__A026_M__speed_1.0` | 5.17 s | -90° (drift: +0.0 → -91.8°) |
| pivot_left_1  | `idle_turn_270_003__A149_M__speed_1.0` | 4.90 s | +90° (drift: -91.7° → -1.2°) |
| pivot_left_2  | `idle_turn_270_003__A149_M__speed_1.0` | 4.90 s | +90° (drift: -1.3° → +89.6°) |
| pivot_right_2 | `step_rotate_idle_090_002__A026_M__speed_1.0` | 5.17 s | -90° (drift: +89.6° → -2.2°) |
| anchor_close | `neutral_idle_loop_002__A074__speed_1.0` | 1.00 s | hold -2.2° |

End-to-end metrics: **XY closure (-0.03, +0.09) m**, **yaw closure -2.1°** (on par with `walk_demo_v6_chain_matched`'s +2.4° — among the tightest closures in the repo). pelvis_z stays in [0.659, 0.681] m (no bobbing). XY path length is 1.77 m total — the individual turns drift ~30 cm during pivot, but the symmetric layout cancels it back to origin.

Seam smoothness (`rest_frames: 75`, track-matched rest pose): max\|d_dof\|/tick is 0.011-0.022 rad across all 5 seams — clean blends, no visible jitter.

#### kinematic viewer
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_in_place_turns_v1_chain_matched.pkl \
    --no-loop

#### sonic in sim
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_in_place_turns_v1_chain_matched.pkl \
    --sim-viewer --no-confirm \
    --max-duration 37

#### rebuild from the YAML
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/in_place_turns_v1_chain_matched.yaml \
    --out      gear_sonic/data/motions/x2_ultra_in_place_turns_v1_chain_matched.pkl

To produce a `uniform_h14` variant, copy the YAML and swap every `chain_matched` token (in both the segment `source:` lines and the `rest.source:` line) to `uniform_h14`. The two pivot keys (`step_rotate_idle_090_002__A026_M__speed_1.0`, `idle_turn_270_003__A149_M__speed_1.0`) are the same in both bundles.

### chair visit v1 — sit DOWN → sit STILL → stand UP (2026-06-24)

Three-segment chair sit/stand sequence on the `chain_matched` track. Robot starts standing, sits down onto the chair, holds the seated pose for ~9 s, stands back up, and ends standing. No rest layer between segments (the standard rest pose is standing, so any rest blend would interpolate the robot back up to standing and back down to sitting at every mid-sequence seam — visually "stand up and sit back down" between the chair clips).

PKL: `gear_sonic/data/motions/x2_ultra_chair_visit_v1_chain_matched.pkl` (single key `chair_visit__v1__chain_matched__sit_down_then_stand_up`, 496 frames @ 30 fps = **16.53 s**)

| segment | source key | frames | duration | pelvis_z |
|---|---|---|---|---|
| sit_down  | `sit_on_chair_start_R_001__A244__speed_1.0` | 142 | 4.73 s | 0.68 → 0.39 m |
| sit_still | `sit_on_chair_loop_R_002__A244__speed_1.0`  | 274 | 9.13 s | held at 0.39 m |
| stand_up  | `sit_on_chair_stop_R_002__A244__speed_1.0`  | 80  | 2.67 s | 0.39 → 0.68 m |

Seam discontinuities (`rest_frames: 0` = hard-cut after yaw-XY alignment):

| seam | max \|d_dof\| | L2 \|d_dof\| | pelvis_xy drift | pelvis_z drift |
|---|---|---|---|---|
| sit_down → sit_still | 0.105 rad (6.0°) | 0.236 rad | 0.0 cm | -0.2 cm |
| sit_still → stand_up | 0.211 rad (12.1°) | 0.405 rad | 0.0 cm | +0.5 cm |

The 12° jump at the sit_still → stand_up seam is the only visible 1-frame blip (~2× the peak joint velocity of the stand_up clip itself). If that's too jarring, the fix is to add a 3-5 frame manual linear-interp blend across the seam — say the word and I'll iterate.

#### kinematic viewer
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_chair_visit_v1_chain_matched.pkl \
    --no-loop

#### sonic in sim (--max-duration sized to ~1 s margin)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_chair_visit_v1_chain_matched.pkl \
    --sim-viewer --no-confirm \
    --max-duration 18

#### rebuild from the YAML (e.g. after swapping in 0.5x source keys for SONIC training)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/chair_visit_v1_chain_matched.yaml \
    --out      gear_sonic/data/motions/x2_ultra_chair_visit_v1_chain_matched.pkl

To produce a half-speed variant, copy the YAML to `chair_visit_v1_halfspeed_chain_matched.yaml`, swap the three `motion_key: ...__speed_1.0` lines for `...__speed_0.5`, double each `n_frames` (284 / 548 / 160 from the 6-entry sit/stand bundle), and rebuild. Total run-time would be 33.07 s.

### PC2 watchdog HOLD yaw-rebase (2026-06-24) — fixes "robot snaps back to previous orientation when I Ctrl+C the stack"

If you've been observing that the robot **springs back to its last-published heading for ~5 s** after killing `run_x2_pkl_direct_stack.sh` (or that pushing the body by hand while idle feels stiff), that was a known PC2-side bug. The `x2_pose_watchdog`'s `HOLD` state was re-publishing the cached upstream frame byte-for-byte — including a stale `root_quat_xyzw` — for the full `--hold-last-secs` (default 5 s) window, so SONIC kept commanding the body back to the cached yaw.

Fix landed: HOLD now splices `R_z(measured_yaw)` (from the live `x2_debug` cache) into the cached frame's `root_quat_xyzw` and `root_quat_xyzw_future` fields. Joint targets are still bit-identical (body pose still freezes exactly where upstream left it), so operators get the original HOLD wifi-blip safety **plus** free-yaw behaviour on intentional shutdowns. Falls back to verbatim re-publish if `x2_debug` is stale or the splice fails.

#### deploying the fix to PC2 (required before the next session)

The watchdog runs on PC2, so the laptop edit isn't live until you rsync the updated files and restart the daemon:

```bash
# 1. rsync the updated pose_pipeline + watchdog onto PC2 (re-runs the same step pc2_bringup.sh uses)
./gear_sonic_deploy/scripts/pc2_bringup.sh \
    --skip-onnx --skip-venv --skip-build --skip-model
# 2. restart the PC2 daemons so the watchdog picks up the new code
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh restart --pc2-host 192.168.86.32
```

After the restart, look for these new lines in the watchdog log:

```
[pose_watchdog] HOLD yaw-rebase:  ENABLED (cached root_quat_xyzw replaced with R_z(measured_yaw) at every HOLD tick; falls back to verbatim if x2_debug stale or splice fails)
[pose_watchdog] tick=… state=HOLD … hold=N hold_rebased=N …    # N == N when the rebase is engaged
[pose_watchdog] HOLD yaw-rebase: ACTIVE -- cached frame's root_quat_xyzw replaced with R_z(measured_yaw=+45.30deg); operator can freely rotate the body during HOLD without policy spring-back
```

To opt out (regression escape only): `--no-x2-debug-yaw-track` on the watchdog disables yaw tracking entirely (then both HOLD and IDLE_CLIP republish baked `R_z(0)` and the body snaps to world +X on every stale tick — the pre-2026-06-23 behaviour).

#### known follow-up (NOT fixed in this turn)

The "in-stack stiffness" — body feels springy when you push it by hand DURING normal stack operation (not just after Ctrl+C) — is a separate symptom caused by the ~50–100 ms wifi RTT between robot rotating, `x2_debug` reporting the new yaw, the recorder re-publishing, and the deploy reading it. The policy spends that ~100 ms with a stale yaw target and provides proportional restoring torque. Three workarounds available today:

1. **Tolerate it** — the body still settles at the new heading once you stop pushing; it just resists during the rotation itself.
2. **`--no-idle-publish`** — already wired into `run_x2_pkl_direct_stack.sh` via the same-named flag. When idle (no clip playing) the recorder stays silent on the pose wire and the PC2 watchdog's `IDLE_CLIP` (which also yaw-rebases per tick from `x2_debug`) takes over. Acceptable for preview-only sessions; the joint targets come from the baked idle stand instead of the recorder's `DEFAULT_STAND_POSE_MUJOCO_RAD` (visually similar, semantically slightly different).
3. **Future: yaw lookahead** — predict yaw forward by `measured_yaw_rate * lag_compensation_s` in the recorder. Not implemented yet. Track via [`2026-06-24_pose_watchdog_hold_yaw_rebase`](docs/source/user_guide/milestones/2026-06-24_pose_watchdog_hold_yaw_rebase.md) "Open follow-ups".

### play_gesture yaw rebase parity (2026-06-26) — fixes "every gesture teleports the robot to world +X"

Companion recorder-side fix to the watchdog HOLD-rebase above. The PC2 watchdog patch handled stale-frame republishing; this one closes the remaining gap on the laptop side: **gesture playback used to teleport the body to the PKL's authored heading at takeover**, even when the robot was facing some arbitrary live yaw. (Locomotion clips already got this right since 2026-06; gestures had been intentionally left on the legacy code path.)

Root cause: `X2DatasetRecorder._drain_clip_commands` picked its yaw rebase target from one of three sources — held-frame, live `x2_debug` `base_quat`, or kplanner `body_pose` snap. The locomotion branch tried the live deploy yaw first and fell back to the kplanner snap. The gesture branch went **straight to the kplanner snap**, which is empty in the direct-PKL stack (no kplanner publishing). Empty snap silently fell back to `yaw=0` → every gesture rebased onto world +X → teleport at takeover.

Fix: gesture branch now follows the same ladder as locomotion — `held-frame > live x2_debug > kplanner snap (stale-x2_debug fallback)`. Held-frame still wins so chained `hold_after` PKLs stay continuous in body frame across the handoff. The new log line at PLAY time tells you which source fed the rebase seed:

```
[recorder] motion-clip PLAY (kind=gesture) 'wave_R_001__A074': 632 frames @ 50.0 Hz (~12.6s) rebased_yaw=-58.4deg [x2_debug-base_quat]
[recorder] motion-clip PLAY (kind=gesture) 'sit_down_A540':    214 frames @ 50.0 Hz (~4.3s)  rebased_yaw=+12.1deg [kplanner-snap-fallback]   # only when x2_debug SUB is stale
```

The yaw-rebase math itself is unchanged (single rigid `Rz(dyaw)` over every frame), so authored arm/wrist/hip motion is bit-identical to before — only the world frame shifts. Tests pin the new behaviour for both kinds in `tests/test_recorder_motion_clip_gate.py::test_drain_play_prefers_live_deploy_yaw_over_kplanner_snap` (parametrized over gesture/locomotion) plus `test_drain_play_held_frame_beats_live_deploy_yaw` to lock in the chained-PKL semantics.

No PC2 redeploy needed — this fix is purely on the laptop recorder, takes effect on the next `run_x2_pkl_direct_stack.sh` (or any other launcher that runs the recorder).

### rest-pose match for stitched loops (2026-06-24) — fixes "snap to default pose" between segments

Every stitched playlist (`walk_loop_v1_*`, `walk_demo_v6_*`, `relaxed_walk_loop_v1*`) inserts a 75-frame (2.5 s) rest layer between segments. Inside that layer the warehouse stitcher SLERPs each joint from the segment-end pose, holds the rest pose for 15 frames, then SLERPs out to the next segment-start. Previously the rest pose came from `x2_ultra_stitched_idle_relaxed_arms.pkl` whose lower-body was retargeted by the **older** `idle_hands_on_back_loop_001` recipe (wrists rotated ~80°, knees +5°), so during every seam the robot visibly rotated both wrists ~80° to a "hands-on-back" pose and back — the user-reported "going to default pose and coming back" effect.

Fix: produced retarget-track-matched rest PKLs and re-pointed all eight stitched playlists at them.

| Rest PKL | Lower body source | Upper body source | Wrist L2 vs walk-end (mean across 7 seams) |
|---|---|---|---|
| `x2_ultra_stitched_idle_relaxed_arms.pkl` (legacy, kept for back-compat) | `idle_hands_on_back_loop_001__A051_M` (older retarget) | `Relaxed_walk_forward_002__A057` (older retarget) | 0.397 rad |
| `x2_ultra_stitched_idle_relaxed_arms_uniform_h14.pkl` (NEW) | `neutral_idle_loop_002__A074__speed_1.0` (uniform_h14) | `Relaxed_walk_forward_002__A057__speed_1.0` (uniform_h14) | **0.313 rad** (-21%) |
| `x2_ultra_stitched_idle_relaxed_arms_chain_matched.pkl` (NEW) | `neutral_idle_loop_002__A074__speed_1.0` (chain_matched) | `Relaxed_walk_forward_002__A057__speed_1.0` (chain_matched) | **0.310 rad** (-22%) |

Anchor seams (`anchor_open` → first segment, last segment → `anchor_close`) get the biggest win: lower-body L2 traverse drops from 0.295 rad to **0.012 rad** since the anchor segments themselves use `neutral_idle_loop_002__A074` — they are now a *zero-mismatch* match to the rest source. The robot opens/closes the loop without any visible knee or pelvis jitter.

Track assignment:
- `walk_loop_v1_uniform_h14.yaml`, `walk_demo_v6_uniform_h14.yaml`, `relaxed_walk_loop_v1.yaml`, `relaxed_walk_loop_v1_halfspeed_walks.yaml` → `..._idle_relaxed_arms_uniform_h14.pkl`
- `walk_loop_v1_chain_matched.yaml`, `walk_demo_v6_chain_matched.yaml`, `relaxed_walk_loop_v1_chain_matched.yaml`, `relaxed_walk_loop_v1_halfspeed_walks_chain_matched.yaml` → `..._idle_relaxed_arms_chain_matched.pkl`

The legacy `x2_ultra_stitched_idle_relaxed_arms.pkl` is still referenced by older playlists (`walk_demo_v1..v5`, `casual_walk_v*`, `warehouse_v*`, `showcase_v1`, `standing_gestures_v1`, `one_foot_v*`, `minimal_v1`, `walk_demo_v6.yaml`, `relaxed_walk_loop_v1_*` pre-fix). Those weren't migrated because they're not on the new retarget tracks — keep them on the legacy rest pose so their lower-body still matches their stride pose.

#### rebuild both track-matched rest PKLs (when source bundles or `make_stitched_motion.py` change)
for TRACK in uniform_h14 chain_matched ; do
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_stitched_motion.py \
      --lower-source gear_sonic/data/motions/x2_ultra_retarget_${TRACK}.pkl \
      --lower-key    neutral_idle_loop_002__A074__speed_1.0 \
      --upper-source gear_sonic/data/motions/x2_ultra_retarget_${TRACK}.pkl \
      --upper-key    Relaxed_walk_forward_002__A057__speed_1.0 \
      --frames 60 --zero-xy \
      --out          gear_sonic/data/motions/x2_ultra_stitched_idle_relaxed_arms_${TRACK}.pkl \
      --out-key      stitched__lower=neutral_idle_loop_002__A074__upper=Relaxed_walk_forward_002__A057__${TRACK}
done

#### rebuild the eight track-matched stitched loops
for YAML in walk_loop_v1_uniform_h14 walk_loop_v1_chain_matched \
             walk_demo_v6_uniform_h14 walk_demo_v6_chain_matched \
             relaxed_walk_loop_v1 relaxed_walk_loop_v1_chain_matched \
             relaxed_walk_loop_v1_halfspeed_walks relaxed_walk_loop_v1_halfspeed_walks_chain_matched ; do
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/${YAML}.yaml \
    --out      gear_sonic/data/motions/x2_ultra_${YAML}.pkl \
    --check-runtime-parity
done

### regular walk closed loop v1 — walk_forward_loop_001__A021 + v6 turn primitives (2026-06-24)

Closed-loop choreography that uses the "regular" walk_forward primitive `walk_forward_loop_001__A021` (canonical SOMA walk-forward clip) instead of the swaggering `Relaxed_walk_forward_*` family. Same v6-style 4-pivot rectangle shape as `relaxed_walk_loop_v1`. Two variants (one per retarget track).

| Variant | PKL | Total | Loop XY | End yaw | Walk dist (each) | pelvis_z mean |
|---|---|---|---|---|---|---|
| uniform_h14 | `x2_ultra_walk_loop_v1_uniform_h14.pkl` | 50.17 s | (-0.30, +0.10) m | -5.4° | ~3.30 m | 0.642 m |
| chain_matched | `x2_ultra_walk_loop_v1_chain_matched.pkl` | 50.17 s | (-0.44, +0.12) m | -9.0° | ~3.49 m | 0.671 m |

Picked `n_frames=158` (5.27 s, ~3.3-3.5 m) for the two walks via FK gap-profile sweep. `walk_forward_loop_001__A021` has a "continuous-stagger" gait (the left foot is held ~14 cm ahead of the right throughout the walk; feet never come together along the walk axis), so the "double-support inter-step" trim trick that worked for `Relaxed_walk_forward_001__A057` doesn't fully apply — f157 is the best local minimum of |gap_walk| AND |gap_lat| in the 2-5 m range. Foot slide at the post-walk rest seam ends up at 11-16 cm (vs 9 cm on the relaxed-walk loop), driven by the residual stagger.

Alternative trims if you re-tune:
- f110 -> 1.94 m walked, yaw -0.71° per walk (shorter, off-axis lateral gap_lat=-0.18)
- f176 -> 3.86 m walked, yaw -0.42° per walk (longer, straightest yaw)

#### kinematic viewer (uniform_h14)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_walk_loop_v1_uniform_h14.pkl \
    --no-loop

#### kinematic viewer (chain_matched)
conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/play_x2_motion_mujoco.py \
    --motion gear_sonic/data/motions/x2_ultra_walk_loop_v1_chain_matched.pkl \
    --no-loop

#### sonic in sim (uniform_h14)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_walk_loop_v1_uniform_h14.pkl \
    --sim-viewer --no-confirm \
    --max-duration 53

#### sonic in sim (chain_matched)
gear_sonic_deploy/deploy_x2.sh sim \
    --model /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx \
    --motion gear_sonic/data/motions/x2_ultra_walk_loop_v1_chain_matched.pkl \
    --sim-viewer --no-confirm \
    --max-duration 53

#### rebuild both walk_loop_v1 stitched variants
for VAR in walk_loop_v1_uniform_h14 walk_loop_v1_chain_matched ; do
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/scripts/make_warehouse_motion.py \
    --playlist gear_sonic/data/motions/playlists/${VAR}.yaml \
    --out      gear_sonic/data/motions/x2_ultra_${VAR}.pkl \
    --check-runtime-parity
done

#### regenerate the walk_forward primitive CSVs (6 BVHs, both tracks, ~6 min on RTX 5090)
PY=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter/.venv/bin/python
CFG_CHAIN=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter/soma_retargeter/configs/agibot_x2_ultra/soma_to_x2_ultra_chain_matched_retargeter_config.json
BVH_DIR=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed/extracted/_retarget_compare
OUT_CHAIN=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/chain_matched
OUT_UNI=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/uniform_h14
cd $HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter
for V in 001 002 003 ; do for A in A021 A021_M ; do
  BVH="$BVH_DIR/walk_forward_loop_${V}__${A}.bvh" ; name=$(basename "$BVH" .bvh)
  "$PY" scripts/retarget_one.py --retargeter-config "$CFG_CHAIN" --bvh "$BVH" --out "$OUT_CHAIN/${name}.csv"
  "$PY" scripts/retarget_one.py --model-height 1.40       --bvh "$BVH" --out "$OUT_UNI/${name}.csv"
done ; done

#### then rebuild both 66-entry bundles (chain_matched + uniform_h14, 22 clips x 3 speeds)
cd $HOME/Projects/GR00T-WholeBodyControl
for SPEED in 1.0 0.5 0.25 ; do
  case $SPEED in 1.0) FPS_SRC=120 ;; 0.5) FPS_SRC=60 ;; 0.25) FPS_SRC=30 ;; esac
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/data_process/build_x2_bones_seed_motion_lib.py \
    --retargeted-root agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/ \
    --subsets chain_matched uniform_h14 \
    --out-dir $HOME/relaxed_walk_retarget_build/build_speed_${SPEED}/ \
    --fps-source $FPS_SRC --fps 30
done
conda run -n env_isaaclab --no-capture-output python - <<'PY'
import joblib
from pathlib import Path
BUILD = Path('/home/stickbot/relaxed_walk_retarget_build')
for track in ['chain_matched', 'uniform_h14']:
    merged = {}
    for speed, d in [(1.0, 'build_speed_1.0'), (0.5, 'build_speed_0.5'), (0.25, 'build_speed_0.25')]:
        for k, v in joblib.load(BUILD / d / f'x2_ultra_{track}.pkl').items():
            merged[f'{k}__speed_{speed}'] = v
    joblib.dump(merged, f'gear_sonic/data/motions/x2_ultra_retarget_{track}.pkl', compress=3)
    print(f'  {track}: {len(merged)} entries')
PY

#### regenerate the chain_matched CSVs from scratch (only needed when the SOMA chain_matched config changes)
# Run via the soma_retargeter project's uv .venv (pulls in `pxr`/usd-core which env_isaaclab
# does not have). Output goes to <repo>/agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/chain_matched/
# NOTE: scripts/retarget_one.py is single-BVH; bvh_to_csv_converter.py cannot inject a custom
# retargeter config (only the default `soma_to_x2_ultra_retargeter_config.json`).
PY=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter/.venv/bin/python
CFG=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter/soma_retargeter/configs/agibot_x2_ultra/soma_to_x2_ultra_chain_matched_retargeter_config.json
BVH_DIR=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed/extracted/_retarget_compare
OUT_DIR=$HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/chain_matched
mkdir -p "$OUT_DIR"
cd $HOME/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter
for BVH in "$BVH_DIR"/*.bvh ; do
  name=$(basename "$BVH" .bvh)
  "$PY" scripts/retarget_one.py --retargeter-config "$CFG" --bvh "$BVH" --out "$OUT_DIR/${name}.csv"
done

#### rebuild the chain_matched bundle PKL (3 speed variants × 16 clips = 48 entries)
cd $HOME/Projects/GR00T-WholeBodyControl
for SPEED in 1.0 0.5 0.25 ; do
  case $SPEED in 1.0) FPS_SRC=120 ;; 0.5) FPS_SRC=60 ;; 0.25) FPS_SRC=30 ;; esac
  conda run -n env_isaaclab --no-capture-output python \
    gear_sonic/data_process/build_x2_bones_seed_motion_lib.py \
    --retargeted-root agibot-x2-references/bones-seed/retargeted/x2_retarget_compare/ \
    --subsets chain_matched \
    --out-dir $HOME/relaxed_walk_retarget_build/build_speed_${SPEED}/ \
    --fps-source $FPS_SRC --fps 30
done
conda run -n env_isaaclab --no-capture-output python - <<'PY'
import joblib
from pathlib import Path
BUILD = Path('/home/stickbot/relaxed_walk_retarget_build')
merged = {}
for speed, d in [(1.0, 'build_speed_1.0'), (0.5, 'build_speed_0.5'), (0.25, 'build_speed_0.25')]:
    for k, v in joblib.load(BUILD / d / 'x2_ultra_chain_matched.pkl').items():
        merged[f'{k}__speed_{speed}'] = v
joblib.dump(merged, 'gear_sonic/data/motions/x2_ultra_retarget_chain_matched.pkl', compress=3)
print(f'Saved {len(merged)} entries')
PY
