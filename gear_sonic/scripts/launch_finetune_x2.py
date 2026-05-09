"""X2-specific entrypoint mirroring Isaac-GR00T's ``launch_finetune.py``.

Why this file exists
--------------------

Upstream ``launch_finetune.py`` doesn't expose two ``TrainingConfig``
fields via its CLI surface even though they're already present in
``gr00t.configs.training.training_config.TrainingConfig`` -- it
hardcodes ``config.training.optim = "adamw_torch"`` and never sets
``config.training.gradient_checkpointing``. For RTX 5090 / 32 GB
fine-tuning of N1.7-3B these two knobs are the difference between a
training run that fits and one that crashes Adam at the optimizer step,
so we expose them in a thin local launcher rather than vendor-patching
``external_dependencies/Isaac-GR00T``.

This script:

* Subclasses upstream ``FinetuneConfig`` to add ``--optim`` and
  ``--gradient-checkpointing`` flags.
* Re-implements the small config-wiring block from upstream's
  ``__main__`` (so we don't have to monkey-patch the upstream
  module body), with the two extra assignments inline.
* Delegates the heavy lifting (modality config side-loading, dataset
  config, default config resolution, ``run()``) to the upstream
  helpers.

If upstream ever adds these fields to its own CLI, delete this file
and point ``gear_sonic/scripts/train_groot_vla.sh`` back at
``external_dependencies/Isaac-GR00T/gr00t/experiment/launch_finetune.py``.

Usage
-----

Drop-in replacement for upstream ``launch_finetune.py``::

    python gear_sonic/scripts/launch_finetune_x2.py \
        --base-model-path nvidia/GR00T-N1.7-3B \
        --dataset-path /path/to/x2_lerobot_dataset \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path gear_sonic/data/x2_modality_config_10dof.py \
        --output-dir /tmp/x2_n17_finetune \
        --num-gpus 1 \
        --global-batch-size 2 \
        --gradient-accumulation-steps 2 \
        --max-steps 3000 \
        --no-tune-llm --no-tune-visual \
        --tune-projector --tune-diffusion-model \
        --gradient-checkpointing \
        --optim adamw_torch
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


def _load_modality_config(modality_config_path: str) -> None:
    """Side-load the user-provided modality config module.

    Lifted verbatim from upstream's ``load_modality_config``; replicated
    here so we don't depend on a private helper that may move.
    """
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


@dataclasses.dataclass
class X2FinetuneConfig(FinetuneConfig):
    """Upstream ``FinetuneConfig`` plus the two extra fields we need to
    pipe through to ``TrainingConfig``.

    Both fields already exist on upstream
    ``gr00t.configs.training.training_config.TrainingConfig`` -- we're
    just exposing them on the CLI without modifying upstream's source.
    """

    optim: str = "adamw_torch"
    """HuggingFace TrainingArguments.optim string. ``adamw_torch`` (default)
    uses fp32 Adam state; ``paged_adamw_8bit`` cuts optimizer state by ~4x
    via bitsandbytes and is the difference between fitting / not fitting
    the 1.09 B-param N1.7 DiT into 32 GB of VRAM during fine-tuning."""

    gradient_checkpointing: bool = False
    """If True, enable HF Trainer's gradient checkpointing. Trades extra
    forward-pass compute for ~2-3x lower activation memory; useful when
    combined with ``--optim paged_adamw_8bit`` on consumer GPUs."""


def main() -> None:
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    ft_config = tyro.cli(X2FinetuneConfig, description=__doc__)

    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    if ft_config.modality_config_path is not None:
        _load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [ft_config.dataset_path],
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            }
        }
    )
    config.load_config_path = None

    # ---- Mirror upstream launch_finetune.py's wiring block ----
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    # The two X2-specific overrides (the rest is identical to upstream).
    config.training.optim = ft_config.optim
    config.training.gradient_checkpointing = ft_config.gradient_checkpointing
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    config.training.save_only_model = ft_config.save_only_model
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)


if __name__ == "__main__":
    main()
