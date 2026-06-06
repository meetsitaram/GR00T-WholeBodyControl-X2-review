# Clip Motion Commands

Workflow for slicing BONES-SEED motions into trimmed PKLs and playing them on the X2 robot in MuJoCo (kinematic preview, SOMA viewer side-by-side, or SONIC tracking via `deploy_x2.sh --motion`).

## Current sit-stand-sit clips

| PKL (in `gear_sonic/data/motions/`) | Source | Range | Frames @ 30 fps |
|---|---|---|---|
| `x2_ultra_sit_stand_sit_003__A538_M_19s_43s.pkl` | dancecard 003 A538 mirror | 19-43 s | 720 |
| `x2_ultra_sit_stand_sit_003__A540_M_23s_38s.pkl` | dancecard 003 A540 mirror | 23-38 s | 450 |
| `x2_ultra_sit_stand_sit_003__A541_25s_49s.pkl`   | dancecard 003 A541        | 25-49 s | 720 |
| `x2_ultra_sit_stand_sit_003__A541_M_25s_49s.pkl` | dancecard 003 A541 mirror | 25-49 s | 720 |
| `x2_ultra_sit_stand_sit_004__A538_M_22s_44s.pkl` | dancecard 004 A538 mirror | 22-44 s | 660 |

## Edit time ranges

`CLIPS = [Clip(name, start_s, end_s), ...]` at top of:

```
agibot-x2-references/bones-seed/scripts/slice_sit_stand_sit.py
```

## Re-slice + re-bake (after editing CLIPS)

Slicer cuts both BVH (human) and CSV (X2 retargeted) at 120 fps and embeds the range in the output filename (`_<start>s_<end>s`).

```sh
# 1) Remove stale slices for the clip(s) you re-timed (replace 19s_43s suffix)
rm -f \
  agibot-x2-references/bones-seed/extracted/sit-stand-sit/<NAME>_<old>s_<old>s.bvh \
  agibot-x2-references/bones-seed/retargeted/x2/sit-stand-sit/<NAME>_<old>s_<old>s.csv \
  gear_sonic/data/motions/x2_ultra_sit_stand_sit_<NAME>_<old>s_<old>s.pkl

# 2) Re-slice (other clips re-slice to same paths, no-op)
python3 agibot-x2-references/bones-seed/scripts/slice_sit_stand_sit.py

# 3) Re-bake to a throwaway dir (keeps the real x2_ultra_bones_seed.pkl untouched)
mkdir -p /tmp/sssbake
conda run -n env_isaaclab --no-capture-output python \
  gear_sonic/data_process/build_x2_bones_seed_motion_lib.py \
  --subsets sit-stand-sit --out-dir /tmp/sssbake

# 4) Explode the multi-clip PKL into 5 single-clip PKLs at the canonical path
python3 - <<'PY'
import joblib, pathlib
src = pathlib.Path("/tmp/sssbake/x2_ultra_sit_stand_sit.pkl")
dst_dir = pathlib.Path("gear_sonic/data/motions")
data = joblib.load(src)
PFX = "neutral_dancecard_object_interact_"
for k, v in data.items():
    short = k[len(PFX):] if k.startswith(PFX) else k
    out = dst_dir / f"x2_ultra_sit_stand_sit_{short}.pkl"
    joblib.dump({k: v}, out, compress=3)
    print(f"wrote {out.name}  ({v['dof'].shape[0]} frames @ {v['fps']} fps)")
PY

rm -rf /tmp/sssbake
```

## Play one clip via SONIC (deploy `--motion` bypass; no kplanner)

Same path as the walk-demo PKLs. Auto-bakes the PKL to a temp X2M2 inside the deploy container; C++ `PklMotionReference` feeds it as the reference SONIC tracks.

```sh
export X2_SIM_MODEL=/home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx

gear_sonic_deploy/deploy_x2.sh sim \
  --model "$X2_SIM_MODEL" \
  --motion gear_sonic/data/motions/x2_ultra_sit_stand_sit_003__A538_M_19s_43s.pkl \
  --sim-viewer --no-confirm
```

Swap `--motion` to any other clip PKL to compare.

## Kinematic preview in MuJoCo (no deploy, no docker, no SONIC)

```sh
python gear_sonic/scripts/play_x2_motion_mujoco.py \
  --motion gear_sonic/data/motions/x2_ultra_sit_stand_sit_003__A538_M_19s_43s.pkl
```

## SOMA viewer: human BVH + retargeted X2 CSV side-by-side

```sh
SR=/home/stickbot/Projects/GR00T-WholeBodyControl/agibot-x2-references/soma-retargeter
BS=/home/stickbot/Projects/GR00T-WholeBodyControl/agibot-x2-references/bones-seed
SUB=sit-stand-sit

"$SR/.venv/bin/python" "$SR/app/bvh_to_csv_converter.py" \
  --config "$SR/assets/x2_ultra_bvh_to_csv_config.json" --viewer gl \
  --bvh "$BS/extracted/$SUB/neutral_dancecard_object_interact_003__A538_M_19s_43s.bvh" \
  --csv "$BS/retargeted/x2/$SUB/neutral_dancecard_object_interact_003__A538_M_19s_43s.csv"
```

## Play a gesture mid-VR-session

Live PKL takeover during a running Quest3 planner stack. The recorder owns a `gesture_cmd` SUB (binds on `:5568`); `play_gesture` PUB-connects, sends `{"action": "play", "name": ...}`, then BLOCKS for the clip's estimated duration so a single Ctrl-C tears it down cleanly. On SIGINT, the script publishes `{"action": "stop"}` before exiting (code 130) and the recorder snaps back to forwarding kplanner frames. Hands are zeroed during gesture; motion_token is zeroed; the PKL's root yaw is rebased to match the robot's current world yaw (pitch/roll/Z pass through, so a sit-down actually lowers the pelvis).

| File | Purpose |
|---|---|
| `gear_sonic/data/motions/gestures/gestures_v1.yaml` | Named-gesture catalog (5 sit-stand-sit clips seeded). |
| `gear_sonic/utils/teleop/gesture_session.py` | PKL load + resample + yaw rebase + future window. |
| `gear_sonic/scripts/play_gesture.py` | Trigger CLI (blocking; Ctrl-C = stop). |

### Add a gesture

Append to `gear_sonic/data/motions/gestures/gestures_v1.yaml`, restart the recorder:

```yaml
  - name: my_new_gesture
    source: gear_sonic/data/motions/x2_ultra_<your_clip>.pkl
    motion_key: null            # null = first key in the PKL
    start_frame: 0
    n_frames: null              # null = play to end of clip
```

### Trigger commands

```sh
# List the catalog (no ZMQ traffic).
python -m gear_sonic.scripts.play_gesture --list

# Play a named gesture; blocks for ~24 s. Ctrl-C stops mid-clip.
python -m gear_sonic.scripts.play_gesture sit_stand_sit_A538

# Ad-hoc PKL (bypasses the catalog).
python -m gear_sonic.scripts.play_gesture \
  --pkl gear_sonic/data/motions/x2_ultra_sit_stand_sit_004__A538_M_22s_44s.pkl
```

### End-to-end MuJoCo smoke test (no Quest3 hardware required)

Two terminals. The wrapper spawns deploy + kplanner + manager + recorder itself (defaults: `WITH_DEPLOY=1`, `SIM_MODEL=...sphere-feet-20260501/.../model_step_025000_g1.onnx`); kplanner emits idle-stand `body_pose` continuously even without a `planner_cmd`, and the recorder forwards those between gestures.

```sh
# T1 — full stack (deploy sim --vla + kplanner + manager + recorder)
gear_sonic/scripts/run_x2_quest3_planner_stack.sh

# T2 — fire a gesture. MuJoCo SONIC sits, then stands.
python -m gear_sonic.scripts.play_gesture sit_stand_sit_A538
```

Expected: idle-stand from kplanner → `play_gesture` block overrides for ~24 s while SONIC tracks the sit-stand-sit → snap back to kplanner idle on completion. Ctrl-C during the block aborts and snaps back early. Pass `--no-deploy` to the wrapper if you want to start the deploy yourself in T1 against a non-default model.
