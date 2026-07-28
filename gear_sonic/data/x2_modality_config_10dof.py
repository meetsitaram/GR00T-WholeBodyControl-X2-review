"""
Side-loadable Isaac-GR00T modality config for X2 Ultra (10-DOF hand variant).

Pass this file to ``launch_finetune.py --modality-config-path`` along with
``--embodiment-tag NEW_EMBODIMENT``:

    python gr00t/experiment/launch_finetune.py \\
        --base-model-path nvidia/GR00T-N1.7-3B \\
        --dataset-path /path/to/x2_lerobot_dataset \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path gear_sonic/data/x2_modality_config_10dof.py

The 10-DOF hand layout exposes the full AgiBot OmniHand action space and is
the v0 target for autonomous X2 manipulation.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag

from gear_sonic.data._x2_groot_compat import apply_all_x2_groot_compat
from gear_sonic.data.x2_modality_config import make_x2_modality_config


# Side-effect: install all X2 -> Isaac-GR00T compatibility patches
# *before* Isaac-GR00T's model classes are instantiated. Three patches:
#   1. ``Qwen3VLForConditionalGeneration.{language_model,visual}`` property
#      shim (transformers 4.x layout fallback for transformers 5.x).
#   2. ``Qwen3Backbone.__init__`` CPU-device guard (escapes transformers
#      5.0 / accelerate 1.13 meta-device context for the nested
#      ``from_pretrained`` call).
#   3. ``Gr00tN1d7.post_init`` freeze fix (re-applies
#      ``set_trainable_parameters`` after post_init resets
#      ``requires_grad=True`` for every parameter, otherwise
#      ``--no-tune-llm`` / ``--no-tune-visual`` silently turn into
#      end-to-end training of the entire 3.45 B model and OOM Adam at
#      the first optimizer step).
# All three are idempotent and no-ops where not needed. See
# ``gear_sonic/data/_x2_groot_compat.py`` for the full rationale of each.
apply_all_x2_groot_compat()


X2_MODALITY_CONFIG_10DOF = make_x2_modality_config(hand_dof=10)

register_modality_config(
    X2_MODALITY_CONFIG_10DOF,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
