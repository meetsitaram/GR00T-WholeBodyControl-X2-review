"""
Full list of loco-manipulation tasks.

GroundOnly - ground only environments

locomanip_pnp - factory environments, pick and place tasks:
LMBottlePnP
LMBoxPnP
"""

from .base import REGISTERED_LOCOMANIPULATION_ENVS

# Trigger metaclass-driven registration of the X2 tabletop tasks so they
# become discoverable via ``robosuite.make(env_name="X2PickPlaceCube", ...)``.
from . import x2_tabletop_pnp  # noqa: F401

ALL_LOCOMANIPULATION_ENVIRONMENTS = REGISTERED_LOCOMANIPULATION_ENVS.keys()
