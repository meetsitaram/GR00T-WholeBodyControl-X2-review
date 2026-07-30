"""
Side-loadable Isaac-GR00T modality config for X2 Ultra with the full
OmniHand 10-DOF-per-hand action surface and **dual head stereo cameras**
as the visual modality.

Use this with datasets recorded by
``run_x2_quest3_planner_stack.sh --head-cameras`` and then sliced to
stereo-only via
``lerobot_slice_episodes.py --cameras stereo_left,stereo_right``
(see ``data/lerobot/x2_grab_a_drink_train_v1`` for the canonical example).

Naming note
-----------
This file's sibling ``x2_modality_config_10dof.py`` uses the legacy
``_10dof`` suffix to mean *"per-hand action-target DOF = 10 (full
OmniHand)"*, NOT total body DOF. The state surface is always the full
51-D X2 proprio (31 body joints + 10 left + 10 right hand joints) and
nothing is dropped from observation regardless of the suffix; only the
hand *action* target width changes (7 = G1-compatible ThreeFinger
subset; 10 = full OmniHand). To avoid the "10dof = total DOF"
confusion that suffix invites, this file names itself after the hand
mechanism (``omnihand``) rather than the numeric DOF.

Usage::

    MODALITY=gear_sonic/data/x2_modality_config_omnihand_stereo.py \\
    bash gear_sonic/scripts/vla/train_groot_vla.sh \\
        data/lerobot/x2_grab_a_drink_train_v1 \\
        data/checkpoints/x2_grab_a_drink_n17_v1

Or directly with the upstream launcher::

    python gear_sonic/scripts/vla/launch_finetune_x2.py \\
        --base-model-path nvidia/GR00T-N1.7-3B \\
        --dataset-path data/lerobot/x2_grab_a_drink_train_v1 \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path gear_sonic/data/x2_modality_config_omnihand_stereo.py

Multi-view tolerance is exercised upstream by the
``oxe_droid_relative_eef_relative_joint`` config (two video keys); see
``external_dependencies/Isaac-GR00T/gr00t/configs/data/embodiment_configs.py:29-32``.
If the GR00T visual encoder ever rejects a 2-key video config for this
embodiment, the safe fallback is to set
``modality_keys=["stereo_left"]`` (single-view) and re-launch.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig

from gear_sonic.data._x2_groot_compat import apply_all_x2_groot_compat
from gear_sonic.data.x2_modality_config import make_x2_modality_config


apply_all_x2_groot_compat()


X2_MODALITY_CONFIG_OMNIHAND_STEREO = make_x2_modality_config(hand_dof=10)
X2_MODALITY_CONFIG_OMNIHAND_STEREO["video"] = ModalityConfig(
    delta_indices=[0],
    modality_keys=["stereo_left", "stereo_right"],
)

register_modality_config(
    X2_MODALITY_CONFIG_OMNIHAND_STEREO,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
