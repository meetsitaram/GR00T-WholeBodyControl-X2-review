# Demo v1 motion staging

Staging folder for the SONIC fine-tune motion library. **Every entry is a
symlink to a canonical PKL** under `gear_sonic/data/motions/x2_recorded/` (or
its sibling `soma_csvs_converted/`), so any rebuild of the upstream PKL flows
through automatically. Staging is **uniform `.pkl` (motion_lib schema)** —
no raw CSVs live here; CSV sources are pre-converted into per-clip PKLs.

Plan that produces this folder: `~/.cursor/plans/sonic_demo_finetune_nebius_42a7d24f.plan.md`.

## Contents (as staged, all PKL)

| Subdir | Count | Symlinks point at | What it is |
|---|---|---|---|
| `body_check/` | 1 bundle PKL -> 1 entry | `gear_sonic/data/motions/x2_ultra_body_check.pkl` | Single 28.9 s body-range / joint-sweep validation motion (`loco__body_check_001__A271_M`). Used as a sanity sequence — wide DoF excursions, hands-down. 230 KB, fps=30. Merger prefix: `bodycheck__`. |
| `combat_chain_matched/` | 3 PKLs | `gear_sonic/data/motions/x2_recorded/soma_csvs_converted/combat_chain_matched/shadow_boxing_R_{001,002,003}__A35{9,0,9}__x2_chain_matched.pkl` | Chain-matched retargeted shadow-boxing (alternating straight punches, 8.6 / 11.4 / 8.6 s). Pre-converted from `agibot-x2-references/soma-retargeter/scratch/combat_chain_matched/*.csv` via `convert_soma_csv_to_motion_lib` (120->30 fps downsample). |
| `dances/` | 1 bundle PKL -> **34 entries** | `gear_sonic/data/motions/x2_ultra_dances.pkl` | Diverse dance corpus (latino, hiphop, western, vogue, jazz, krakowiak, retro disco, basic turn, victory dance, etc.). 34 clips at 3.0-9.7 s each (~170 s total). 1.5 MB, fps=30. **This bundle subsumes the 2 latino-kick clips that used to live in `fighting_chain_matched/`** — `dance_latino_kick_kick_R_001__A313` and `dance_latino_chase_mambo_kicks_R_fast_001__A314` — so that subdir was removed to avoid double-weighting. Merger prefix: `dance__`. |
| `sitstand_chain_matched/` | 1 bundle PKL -> **6 entries** | `gear_sonic/data/motions/x2_ultra_sitstand_chain_matched.pkl` | Chain-matched retargeted chair sit/loop/stand at 2 speeds (3 clips × {speed_1.0, speed_0.5} using the standard `__speed_X.X` key convention): sit-down (4.7 / 9.4 s), sitting-loop (9.1 / 18.3 s), stand-up (2.7 / 5.3 s). 354 KB, fps=30. The per-clip PKLs under `gear_sonic/data/motions/x2_recorded/soma_csvs_converted/sitstand_chain_matched/` are superseded by this bundle and no longer linked from staging (kept on disk as audit trail). |
| `mc_gestures/` | 51 PKLs + README | `gear_sonic/data/motions/x2_recorded/mc_gestures/*.pkl` | MC-mode gestures captured 2026-06-23 from real X2 via the mobile app, foot-flat root rot, motion_lib schema (byte-compatible with `x2_ultra_bones_seed.pkl`). |
| `retargeted/` | 6 PKLs | `gear_sonic/data/motions/x2_ultra_{relaxed_walk_loop_v1*, walk_demo_v6_*}.pkl` | 4 relaxed-walk closed-loop variants (1x/0.5x walks x uniform_h14/chain_matched retargeters) + 2 walk_demo_v6 variants used in earlier demos. Built 2026-06-23 via `make_warehouse_motion.py`. The 48-entry `x2_ultra_retarget_*` bundles were dropped 2026-06-23 to keep the demo corpus tight. |
| `teleop_kinematic/` | TBD | (will be) `data/lerobot/x2_quest3_kinematic_demo/debug/teleop_episode_*.npz` -> PKL via new `convert_kinematic_teleop_to_motion_lib.py` | Fresh kinematic-MuJoCo Quest 3 teleop captures, to be recorded next. The converter will also drop per-episode PKLs into a canonical location (sibling of `soma_csvs_converted/`) and these will be symlinked in. |

**Current total: 63 staging symlinks -> 101 motion entries across 6 staged
subdirs** (2 of the symlinks are multi-entry bundles: `sitstand` -> 6 entries,
`dances` -> 34 entries, `body_check` -> 1 entry; everything else is
single-entry-per-symlink). `teleop_kinematic/` adds more once recorded.
Post-prefix collision audit: 0 collisions, 101 unique merged keys.

A "motion entry" is one `{key: motion_lib_dict}` pair inside a PKL. Bundle
PKLs (multiple entries per file) and single-key PKLs are both fine — the
merger will iterate the dict items in each staged PKL, not assume one
entry per file.

## How to filter out a clip

`rm <symlink>` — does NOT touch the upstream canonical PKL. The merger
(`build_x2_demo_motion_lib.py`, Phase 1c of the plan) only walks this folder,
so anything you delete here is dropped from the demo corpus.

## How to add a clip

Symlink a canonical PKL into the matching subdir:

```bash
ln -sf "$(realpath path/to/source.pkl)" \
       gear_sonic/data/motions/demo_v1_sources/<subdir>/<name>.pkl
```

For new SOMA chain-matched CSVs, first run the converter once so a canonical
per-clip PKL exists, then symlink. Example pattern used for the 8 SOMA CSVs
already staged here (combat + fighting + sitstand):

```python
# inside env_isaaclab
from gear_sonic.data_process.convert_soma_csv_to_motion_lib import (
    set_robot, load_bones_csv, convert_sequence, downsample_sequence,
)
import joblib
set_robot("x2_ultra")
seq = load_bones_csv("<path>.csv")
entry = downsample_sequence(convert_sequence(seq, 120), 120, 30)
joblib.dump({"<stem>": entry},
            "gear_sonic/data/motions/x2_recorded/soma_csvs_converted/"
            "<subdir>/<stem>.pkl", compress=True)
```

## Disk footprint

This staging folder itself is ~0 bytes on disk (only symlinks + this README).
Real bytes referenced:

- body_check bundle PKL: ~230 KB (1 entry, 28.9 s)
- combat PKLs: ~240 KB (3 clips, 72-94 KB each)
- dances bundle PKL: ~1.5 MB (34 entries, 3-10 s each, ~170 s total)
- sitstand bundle PKL: ~354 KB (6 entries, 3 clips × 2 speeds)
- mc_gestures PKLs: ~10 MB
- retargeted PKLs: ~1.5 MB (6 small stitched-loop PKLs, ~200-300 KB each)

Everything together rounds to ~14 MB — still tiny.

## What happens next

1. Record kinematic-MuJoCo teleop episodes via
   `python -m gear_sonic.scripts.teleop_x2_kinematic --output-dir data/lerobot/x2_quest3_kinematic_demo`,
   then convert the resulting `debug/teleop_episode_*.npz` files into
   per-episode PKLs (new `convert_kinematic_teleop_to_motion_lib.py`) and
   symlink into `teleop_kinematic/`.
2. Run `gear_sonic/data_process/build_x2_demo_motion_lib.py` (new merger,
   ~60 LOC per the plan — simpler now that staging is uniform PKL). For each
   `<subdir>/*.pkl`, `joblib.load` it (some are single-entry, some are
   bundles like `dances` (34 entries), `sitstand` (6), `body_check` (1)),
   iterate `dict.items()`, prefix each key with the subdir tag (e.g.
   `combat__shadow_boxing_R_001...`, `gesture__left_wave_001`,
   `sitstand__sit_on_chair_start_R_001...__speed_1.0`,
   `dance__dance_latino_kick_kick_R_001__A313`,
   `bodycheck__loco__body_check_001__A271_M`,
   `walk__relaxed_walk_loop_v1`, `teleop__episode_001`), assert no key
   collisions (verified: 0 collisions today across the 101 staged entries),
   and `joblib.dump` the merged dict to
   `gear_sonic/data/motions/x2_ultra_demo_v1.pkl` (compress=3).
3. Author `sonic_x2_ultra_demo_v1.yaml` (inherits sphere-feet bones-seed
   config, points at the new PKL, `num_learning_iterations=4000`).
4. Launch the ~4k-iter local fine-tune on the RTX 5090 off the H200 25k
   sphere-feet checkpoint with `+checkpoint=...`.
