# X2 MotionBricks: walk-and-turn planner training

This path trains a **kinematic motion generator** for AgiBot X2 Ultra using the
MotionBricks stack (same family as the G1 planner Zen Luo pointed to), scoped to
**walking and turning** clips only — enough for manipulation setups where the
robot moves around a room and reaches, without stylistic gaits or running.

## Why not reuse the G1 planner ONNX?

| | G1 | X2 |
|--|----|-----|
| Body DOFs | 29 | 31 |
| Skeleton | `G1Skeleton34` | `X2Skeleton34` (new) |
| Shipped planner | `planner_sonic.onnx` | none |
| Feature dim | 418 | **418** (same joint count with toe dummies) |

Weights are **not interchangeable** (different kinematics and data), but network
**widths match** so you can bootstrap hyperparameters from G1 checkpoints.

## Components added in this repo

| Path | Role |
|------|------|
| `motionlib/core/skeletons/x2.py` | `X2Skeleton34` + MuJoCo body map |
| `data/x2_loco_filters.py` | Walk/turn key filter (excludes run, crawl, manip, styles) |
| `data/x2_pkl_to_motion.py` | PKL → FK → `input_tensor_dict` |
| `data/x2_bones_seed_dataset.py` | PyTorch `Dataset` + feature cache |
| `scripts/build_x2_skeleton_assets.py` | `joints.p`, `parents.p`, stats, bootstrap hparams |
| `scripts/train_vqvae_x2.py` | VQVAE training on filtered X2 clips |

## Data

Default source: `gear_sonic/data/motions/x2_ultra_bones_seed.pkl` (joblib, ~2.5k
retargeted BONES-SEED clips).

**Included** (examples): `walk`, `stride`, `turn`, `sideway`, `loco__*`, `idle`/`stand`.

**Excluded**: `run`, `crawl`, `crouch`, `pick`, `dance`, `standing__*` manip subset, etc.
See `DEFAULT_*_PATTERNS` in `x2_loco_filters.py`.

To tighten or loosen filters, pass custom regex lists into `X2LocoMotionDataset`.

## Quick start

```bash
cd motionbricks
conda activate motionbricks  # python 3.10, pip install -e .

# 1) Skeleton + normalization stats (needs repo-root x2_ultra.xml + bones PKL)
python scripts/build_x2_skeleton_assets.py \
  --out-dir out/motionbricks_vqvae_x2/version_1 \
  --pkl ../gear_sonic/data/motions/x2_ultra_bones_seed.pkl

# 2) Train VQVAE on walk/turn features
python scripts/train_vqvae_x2.py \
  --result_dir ./out \
  --max_steps 500 \
  --batch_size 4
```

Pose and root models follow the G1 scripts (`train_pose.py`, `train_root.py`) once
you point their hparams at `out/motionbricks_vqvae_x2/version_1` and swap the
skeleton target to `X2Skeleton34`.

## Deploy (not wired yet)

Today X2 uses the **heuristic planner** (`x2_heuristic_planner.py`) streaming ZMQ
poses. A trained MotionBricks model should run as a **Python inference daemon**
with the same ZMQ `pose` layout until ONNX export + C++ integration land.

## G1 → X2 “miracle” transfer

There is no safe weight transfer. Do train on X2-retargeted data. Optional future
work: initialize small layers from G1 after verifying feature alignment on a
single clip.
