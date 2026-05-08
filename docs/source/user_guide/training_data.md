# Training Data

## BONES-SEED

[BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) is an open dataset of **142,220 annotated human motion animations** for humanoid robotics, created by [Bones Studio](https://bones.studio/datasets). It provides motion capture data in SOMA and Unitree G1 formats with natural language descriptions, temporal segmentation labels, and detailed skeletal metadata.

| | |
|---|---|
| **Total motions** | 142,220 (71,132 original + 71,088 mirrored) |
| **Total duration** | ~288 hours (@ 120 fps) |
| **Performers** | 522 actors (253 F / 269 M) |
| **Age range** | 17–71 years |
| **Height range** | 145–199 cm |
| **Weight range** | 38–145 kg |
| **Output formats** | SOMA Uniform · SOMA Proportional · Unitree G1 MuJoCo-compatible |
| **Annotations** | Up to 6 NL descriptions per motion + temporal segmentation + skeletal metadata |

### Relevance to SONIC

BONES-SEED a large subset of SONIC training data:

- **Unitree G1 joint trajectories** — retargeted for MuJoCo, directly usable for motion tracking training
- **Broad motion coverage** — locomotion, manipulation, dance, sports, communication, and everyday activities across 8 categories and 20 sub-categories
- **Rich language annotations** — up to 6 natural language descriptions per motion, enabling language-conditioned policy learning
- **Temporal segmentation** — per-motion phase labels with timestamps for structured skill decomposition
- **Performer diversity** — 522 actors spanning a wide range of body types, ages, and movement styles

### Motion Categories

| Package       | Motions | Description                                                             |
|---------------|---------|-------------------------------------------------------------------------|
| Locomotion    | 74,488  | Walking, jogging, jumping, climbing, crawling, turning, and transitions |
| Communication | 21,493  | Gestures, pointing, looking, and communicative body language            |
| Interactions  | 14,643  | Object manipulation, pick-and-place, carrying, and tool use             |
| Dances        | 11,006  | Full-body dance performances across multiple styles                     |
| Gaming        | 8,700   | Game-inspired actions and dynamic movements                             |
| Everyday      | 5,816   | Household tasks, consuming, sitting, reading, and daily activities      |
| Sport         | 3,993   | Athletic movements and sports-specific actions                          |
| Other         | 2,081   | Stunts, martial arts, and edge-case motions                             |

### Data Formats

Every motion is available in three formats:

- **SOMA Proportional (BVH)** — per-actor skeleton preserving original body proportions
- **SOMA Uniform (BVH)** — standardized skeleton shared across all motions for batch processing
- **Unitree G1 (CSV)** — joint-angle trajectories retargeted to the Unitree G1 humanoid

### Download

```bash
# Using the Hugging Face CLI
pip install huggingface_hub
huggingface-cli download bones-studio/seed --repo-type dataset --local-dir ./bones-seed
```

```python
# Using Python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="bones-studio/seed",
    repo_type="dataset",
    local_dir="./bones-seed"
)
```

After downloading, extract the motion archives. The full `soma_uniform.tar.gz`
is ~43 GB compressed and ~277 GB uncompressed (142,220 BVH files). Use `pigz`
for parallel decompression:

```bash
mkdir -p ./bones-seed/extracted/all
pigz -dc ./bones-seed/soma_uniform.tar.gz | tar -xf - -C ./bones-seed/extracted/all
```

Allow ~5 minutes on a fast NVMe SSD with 32 cores.

## Curating a Training Subset

For a new embodiment, you typically don't want all 142K motions — most policies
only need a few thousand well-chosen clips for the target skill set (locomotion,
manipulation, posture transitions, etc.). The repo ships an example curation
pipeline under `agibot-x2-references/bones-seed/scripts/` that:

1. Reads `metadata/seed_metadata_v004.parquet`
2. Applies tag-based filters (movement type, body position, props, complexity)
3. Stratified-samples to balance the resulting subset
4. Writes a CSV of motion metadata + a `.txt` list of filenames

Adapt `curate_bones.py` and `build_subset.py` to your robot's needs. The X2
example produces two subsets:

| Subset                    | Motions | Composition |
|---------------------------|--------:|-------------|
| `loco-manipulation`       |  2,000  | Walking, turning, standing-idle + light/heavy object manipulation |
| `standing-manipulation`   |    550  | Pure standing + light-prop manipulation only |

After curating, materialize the subset directories with the chosen BVH files
(see `extract_all.sh` + `materialize_and_clean.sh` for the X2 example flow).

## Retargeting to Your Robot

Once you have a subset of BVH files on disk, retarget them to your robot using
the [SOMA Retargeter](https://github.com/NVlabs/SOMA-Retargeter). The repo
includes the retargeter as a submodule under
`agibot-x2-references/soma-retargeter/`, pre-configured for both Unitree G1 and
Agibot X2 Ultra. See its `README.md` for one-time setup (`uv sync` + apt install
`python3.12-tk` for the GUI).

### Single motion (sanity check)

Before batching thousands of clips, retarget one in the interactive viewer to
confirm the IK solver, joint limits, and scaler are configured correctly:

```bash
cd agibot-x2-references/soma-retargeter
uv run python ./app/bvh_to_csv_converter.py \
  --config ./assets/x2_ultra_bvh_to_csv_config.json \
  --viewer gl
```

In the right panel, click **BVH Motion → Load**, pick a `.bvh` file, then
**Retarget**. The source SOMA actor and the retargeted robot animate side by
side. Save the result with **CSV Motion → Save** if you want to keep it.

### Batch (single process)

To convert a folder of BVH files headlessly, edit a config to set
`import_folder` and `export_folder`, then:

```bash
cd agibot-x2-references/soma-retargeter
uv run python ./app/bvh_to_csv_converter.py \
  --config ./assets/x2_loco_manipulation_config.json \
  --viewer null
```

Throughput is ~30 motions/min on an RTX 5090. For ~2,500 motions that is roughly
~90 minutes. The retargeter does not skip already-completed files in this mode —
killing it loses unfinished batches.

### Batch (parallel + resumable)

For larger workloads use `agibot-x2-references/bones-seed/scripts/retarget_x2_parallel.py`.
It shards the remaining work across N processes, writes directly to the
canonical output directory, and **skips any file that already has a matching CSV**
— so you can interrupt and rerun freely:

```bash
# default: 4 shards per subset
python3 agibot-x2-references/bones-seed/scripts/retarget_x2_parallel.py

# detached, log to file
nohup python3 agibot-x2-references/bones-seed/scripts/retarget_x2_parallel.py \
  > agibot-x2-references/bones-seed/retargeted-driver.log 2>&1 &

# more shards (5090 fits 6-8 comfortably; each shard uses ~2 GB VRAM)
PARALLEL_SHARDS=8 python3 agibot-x2-references/bones-seed/scripts/retarget_x2_parallel.py
```

Measured speedup with 4 parallel shards: **~3.2×** over single-process. The
remaining gap from the ideal 4× is one-time CUDA/Warp init per shard (~2 min);
the overhead amortizes on longer runs.

Per-shard logs are written to `agibot-x2-references/bones-seed/retargeted/logs/`.
Monitor progress with:

```bash
ls agibot-x2-references/bones-seed/retargeted/x2/<subset>/ | wc -l
tail -c 4000 agibot-x2-references/bones-seed/retargeted/logs/<subset>_shard0.log \
  | tr '\r' '\n' | tail -5
```

The script names two subsets (`loco-manipulation`, `standing-manipulation`) by
default. Edit the `SUBSETS` list in the script for your own splits.

### Output Format

The retargeter emits one CSV per motion. Columns are: `Frame`, root pose
(`root_translateX/Y/Z`, `root_rotateX/Y/Z` in cm and degrees), then one column
per robot DOF in URDF order. These CSVs are the input to the SONIC motion
library — see [Training on New Embodiments](new_embodiments.md) for the
PKL conversion and Isaac Lab integration steps.
