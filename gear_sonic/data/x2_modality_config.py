"""
X2 Ultra SONIC modality config factory for Isaac-GR00T fine-tuning.

Two variants of the action space share a single recorded dataset:

* **7-DOF hands** — keeps Isaac-GR00T's `unitree_g1_sonic` action layout
  (ThreeFinger), so the dataset can also be replayed against the upstream
  G1 SONIC posttrain checkpoint for cross-embodiment ablations. The X2
  parquet stores 10 DOF per hand and the 7-DOF view is constructed by the
  exporter via the active-finger subset (see Item 5 in the plan).

* **10-DOF hands** — the full OmniHand action surface. This is the v0
  target for autonomous X2 manipulation.

This file does **not** register itself with the upstream registry. Two
thin side-loaders sit next to it (``x2_modality_config_7dof.py`` and
``x2_modality_config_10dof.py``) that get passed to
``launch_finetune.py --modality-config-path``; they call
``register_modality_config(..., embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)``
which mutates ``gr00t.configs.data.embodiment_configs.MODALITY_CONFIGS`` in-process.

Reference: `docs/source/references/x2_isaac_groot_data_contract.md`
"""

from __future__ import annotations

from typing import Literal


SONIC_MOTION_TOKEN_DIM: int = 64
"""Dimensionality of the SONIC latent motion token streamed from the VLA decoder."""

DEFAULT_ACTION_HORIZON: int = 40
"""Action chunk length in frames. Matches `unitree_g1_sonic` upstream (= 0.8s @ 50 Hz)."""

DEFAULT_STATE_GROUPS: tuple[str, ...] = (
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
    "projected_gravity",
)
"""State joint groups exposed to the VLA backbone. Mirrors `unitree_g1_sonic`.

The (start, end) parquet slices for these groups are written by the gear_sonic
exporter into `meta/modality.json` and derived from the X2 RobotModel's joint
ordering (see `gear_sonic/data/robot_model/instantiation/x2_ultra.py`). The
upstream loader doesn't validate dimensions here — it only checks that the
group names exist on disk — so this tuple is the source of truth for *which*
groups the trainer attends to, not their numeric shapes.
"""


def make_x2_modality_config(
    hand_dof: Literal[7, 10],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    motion_token_dim: int = SONIC_MOTION_TOKEN_DIM,
) -> dict:
    """Build the Isaac-GR00T `ModalityConfig` dict for the X2 Ultra SONIC embodiment.

    Args:
        hand_dof: 7 (G1-compatible ThreeFinger view) or 10 (full OmniHand).
        action_horizon: number of future steps the policy predicts in each
            inference call. Default 40 matches `unitree_g1_sonic`.
        motion_token_dim: SONIC latent dimensionality. Default 64 matches the
            shipping 22k checkpoint.

    Returns:
        A dict suitable for `register_modality_config(...)`. Top-level keys are
        ``video``, ``state``, ``action``, ``language``.

    Notes:
        - State and action *names* must match the keys written to
          ``meta/modality.json`` by the gear_sonic exporter.
        - All hand/motion_token actions are ``ABSOLUTE / NON_EEF / DEFAULT``
          to mirror `unitree_g1_sonic` and play well with SONIC's tracking
          policy, which consumes absolute targets, not deltas.
    """
    if hand_dof not in (7, 10):
        raise ValueError(f"hand_dof must be 7 or 10, got {hand_dof}")

    # Imported lazily so this module can be inspected in environments that
    # don't have gr00t available (e.g. read-the-docs builds).
    from gr00t.data.types import (
        ActionConfig,
        ActionFormat,
        ActionRepresentation,
        ActionType,
        ModalityConfig,
    )

    action_keys = ["motion_token", "left_hand_joints", "right_hand_joints"]
    action_configs = [
        ActionConfig(
            rep=ActionRepresentation.ABSOLUTE,
            type=ActionType.NON_EEF,
            format=ActionFormat.DEFAULT,
        )
        for _ in action_keys
    ]

    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=list(DEFAULT_STATE_GROUPS),
        ),
        "action": ModalityConfig(
            delta_indices=list(range(action_horizon)),
            modality_keys=action_keys,
            action_configs=action_configs,
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.task_description"],
        ),
    }


__all__ = [
    "DEFAULT_ACTION_HORIZON",
    "DEFAULT_STATE_GROUPS",
    "SONIC_MOTION_TOKEN_DIM",
    "make_x2_modality_config",
]
