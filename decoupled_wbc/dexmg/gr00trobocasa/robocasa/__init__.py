from robocasa.environments.locomanipulation.base import (
    PnPBottle,
    PickBottleShelf,
    PnPBottleHigh,
    NavPickBottle,
    PnPBottleRandRobotPose,
    VisualReach,
    PnPBottleFixtureToFixture,
    PnPBottleFixtureToFixtureSourceDemo,
    PnPBottleShelfToTable,
    PnPBottleTableToTable,
    PickBottleGround,
    PickBottles,
    NavPickBottles,
    PnPBottlesTableToTable,
)
from robocasa.environments.locomanipulation.locomanip_basic import (
    LMPickBottle,
    LMPickBottleHigh,
    LMNavPickBottle,
    LMPickBottleGround,
    LMPnPBottle,
    LMPickMultipleBottles,
    LMPnPMultipleBottles,
    LMPickBottleShelf,
    LMNavPickBottleShelf,
    LMPickBottleShelfLow,
    LMNavPickBottleShelfLow,
    LMPnPBottleToPlate,
    LMPnPAppleToPlate,
)
from robocasa.environments.locomanipulation.locomanip_pnp import (
    LMBottlePnP,
    LMBoxPnP,
)

from robocasa.environments.locomanipulation.locomanip_dc import (
    LMNavPickBottleDC,
    LMPnPAppleToPlateDC,
)

from robocasa.environments.locomanipulation.x2_tabletop_pnp import (
    LMTabletopFixedBase,
    X2PickPlaceCube,
    X2PickPlaceBowl,
    X2PickPlaceApple,
)

# from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
from robosuite.controllers import ALL_PART_CONTROLLERS, load_composite_controller_config
from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.models.grippers import ALL_GRIPPERS
from robosuite.robots import ALL_ROBOTS


import mujoco

_SUPPORTED_MUJOCO = ("3.2.6", "3.3.2", "3.3.7", "3.5.0", "3.7.0")
assert mujoco.__version__ in _SUPPORTED_MUJOCO, (
    f"MuJoCo version must be one of {_SUPPORTED_MUJOCO}; "
    f"got {mujoco.__version__}. (Add yours to _SUPPORTED_MUJOCO in "
    f"robocasa/__init__.py once you've smoke-tested it.)"
)

import numpy

assert numpy.__version__ in [
    "1.23.2",
    "1.23.3",
    "1.23.5",
    "1.26.4",
    "2.2.5",
    "2.2.6",
], "numpy version must be either 1.23.{2,3,5}, 1.26.4 or 2.2.{5,6}. Please install one of these versions."

import robosuite

# Upstream pins to 1.5.{0,1}; 1.5.2 ships a bug-fix release with a compatible
# API surface for the X2 + tabletop scenes we use. Smoke-tested via
# tests/test_x2_robocasa_scene_mode.py before adding to this list.
assert robosuite.__version__ in [
    "1.5.0",
    "1.5.1",
    "1.5.2",
], "robosuite version must be one of 1.5.{0,1,2}. Please install the correct version"

__version__ = "0.2.0"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
