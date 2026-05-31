"""Tabletop pick-and-place tasks tuned for the AgiBot X2 Ultra in Phase 1.

Phase 1 of the X2 + GR00T integration uses a *fixed-lower-body* X2 (legs and
pelvis welded out by ``X2UltraFixedLowerBody``).  We therefore want simple,
single-table workspaces that sit directly in front of the robot at a height
the OmniHand can reach, with one or two free-floating objects to manipulate.

Two starter tasks live here:

``X2PickPlaceCube``
    Pick a small red cube from a randomised pose on the table and drop it
    into a fixed blue tray on the same table.  The bowl/tray is modelled as
    a low-walled primitive box so the contact / containment check is
    cheap and deterministic.

``X2PickPlaceBowl``
    Same single-table layout, but the bowl itself becomes the manipulation
    target -- the cube acts as a visible reference object that must remain
    on the source side of the table.

Both tasks reuse :class:`LMTabletopFixedBase`, which mirrors the
``PnPBottle`` table-loading pattern from ``locomanip.py`` but trims the
default base position so the X2 OmniHand wrist sits above the work surface.
The classes intentionally do not depend on any omniverse assets so the
environments load even on a clean ``gr00trobocasa`` install.
"""

from __future__ import annotations

from typing import Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from robosuite.utils.mjcf_utils import array_to_string, xml_path_completion

import robocasa
from robocasa.environments.locomanipulation.base import (
    LocoManipulationEnv,
    PrimitiveBottle,  # re-exported for downstream subclasses that may want it
    PrimitiveFixture,
)
from robocasa.models.objects.objects import MJCFObject
from robocasa.utils.dexmg_utils import DexMGConfigHelper
from robocasa.utils.object_utils import check_obj_upright
from robocasa.utils.visuals_utls import Gradient, randomize_materials_rgba

# ``X2PickPlaceApple`` reuses the upstream ``apple_0`` real-mesh asset
# from the locomanip object library. Unlike ``PrimitiveCube`` /
# ``PrimitiveBowl`` (which we synthesize inline from raw boxes), the
# apple ships with a textured visual mesh + 5-fragment convex
# decomposition and is loaded via ``MJCFObject``. Centralising the
# mesh-relative path here keeps the asset reference next to the env
# class that consumes it instead of hidden in a per-method literal.
_APPLE_MJCF_PATH: str = "objects/omniverse/locomanip/apple_0/model.xml"


# ---------------------------------------------------------------------------
# Primitive props (kept self-contained so X2 tasks load on a clean install)
# ---------------------------------------------------------------------------


class PrimitiveCube:
    """A small free-floating cube prop with matching visual + collision geoms."""

    DEFAULT_RGB = [0.85, 0.15, 0.15]

    def __init__(
        self,
        name: str = "cube",
        half_size: float = 0.025,
        rgb: Optional[list[float]] = None,
        density: float = 100.0,
    ):
        self.name = name
        self.half_size = half_size

        rgb_str = " ".join(map(str, self.DEFAULT_RGB if rgb is None else rgb))
        self.assets = [
            ET.Element(
                "texture",
                type="2d",
                name=f"{name}_tex",
                builtin="flat",
                rgb1=rgb_str,
                width="64",
                height="64",
            ),
            ET.Element(
                "material",
                name=f"{name}_mat",
                texture=f"{name}_tex",
                texuniform="true",
                reflectance="0.05",
            ),
        ]

        self.body = ET.Element("body", name=f"{self.name}_body", pos="0.45 0 0.85")
        # Visual-only geom (group=1, no contact, no inertia contribution).
        self.body.append(
            ET.Element(
                "geom",
                name=f"{self.name}_vis",
                pos="0 0 0",
                size=f"{half_size} {half_size} {half_size}",
                type="box",
                material=f"{name}_mat",
                group="1",
                conaffinity="0",
                contype="0",
            )
        )
        # Collision-and-inertia geom (group=0 so robosuite's
        # ``inertiagrouprange="0 0"`` compiler picks it up).
        self.contact_geoms = [f"{self.name}_collider"]
        self.body.append(
            ET.Element(
                "geom",
                name=self.contact_geoms[0],
                type="box",
                pos="0 0 0",
                size=f"{half_size} {half_size} {half_size}",
                group="0",
                solimp="0.998 0.998 0.001",
                solref="0.001 2",
                density=str(density),
                friction="1.0 0.3 0.05",
            )
        )
        self.body.append(
            ET.Element(
                "joint",
                name=f"{self.name}_joint",
                type="free",
                damping="0.0005",
            )
        )


class PrimitiveBowl:
    """A simple "bowl"/tray prop -- a low-walled open box.

    Modelled as four thin walls + a thin floor so dropping a cube inside
    produces stable contacts.  The whole prop is welded to the world (no
    free joint) so the robot has a stationary placement target.
    """

    DEFAULT_RGB = [0.20, 0.55, 0.85]
    WALL_THICKNESS = 0.005

    def __init__(
        self,
        name: str = "bowl",
        half_size_xy: float = 0.075,
        wall_height: float = 0.035,
        rgb: Optional[list[float]] = None,
    ):
        self.name = name
        self.half_size_xy = half_size_xy
        self.wall_height = wall_height
        self.contact_geoms = [
            f"{self.name}_floor",
            f"{self.name}_wall_xp",
            f"{self.name}_wall_xn",
            f"{self.name}_wall_yp",
            f"{self.name}_wall_yn",
        ]

        rgb_str = " ".join(map(str, self.DEFAULT_RGB if rgb is None else rgb))
        self.assets = [
            ET.Element(
                "texture",
                type="2d",
                name=f"{name}_tex",
                builtin="flat",
                rgb1=rgb_str,
                width="64",
                height="64",
            ),
            ET.Element(
                "material",
                name=f"{name}_mat",
                texture=f"{name}_tex",
                texuniform="true",
                reflectance="0.05",
            ),
        ]

        self.body = ET.Element("body", name=f"{self.name}_body", pos="0.45 0 0.85")

        # Each wall + floor gets a *visual* geom (material, group=1, no
        # contact, no inertia contribution) and a co-located *collision*
        # geom (group=0, so robosuite's ``inertiagrouprange="0 0"`` picks
        # it up for body mass).
        floor_half_z = self.WALL_THICKNESS
        wall_half_z = wall_height / 2.0
        wall_thick = self.WALL_THICKNESS

        def _add_pair(name_suffix: str, pos: str, size: str, density: str = "500"):
            # Visual geom -- shows the bowl's colour.
            self.body.append(
                ET.Element(
                    "geom",
                    name=f"{self.name}_{name_suffix}_vis",
                    type="box",
                    pos=pos,
                    size=size,
                    material=f"{name}_mat",
                    group="1",
                    conaffinity="0",
                    contype="0",
                )
            )
            # Collision + inertia geom -- same shape, group=0.
            collider_name = f"{self.name}_{name_suffix}"
            self.body.append(
                ET.Element(
                    "geom",
                    name=collider_name,
                    type="box",
                    pos=pos,
                    size=size,
                    group="0",
                    solimp="0.998 0.998 0.001",
                    solref="0.001 2",
                    density=density,
                    friction="1.0 0.5 0.1",
                )
            )

        # Floor (sits at body origin so prop position = floor top).
        _add_pair("floor", f"0 0 {-floor_half_z}", f"{half_size_xy} {half_size_xy} {floor_half_z}")

        # +x / -x walls
        for sign, suffix in ((1.0, "xp"), (-1.0, "xn")):
            _add_pair(
                f"wall_{suffix}",
                f"{sign * (half_size_xy - wall_thick)} 0 {wall_half_z}",
                f"{wall_thick} {half_size_xy} {wall_half_z}",
            )
        # +y / -y walls
        for sign, suffix in ((1.0, "yp"), (-1.0, "yn")):
            _add_pair(
                f"wall_{suffix}",
                f"0 {sign * (half_size_xy - wall_thick)} {wall_half_z}",
                f"{half_size_xy} {wall_thick} {wall_half_z}",
            )


# ---------------------------------------------------------------------------
# Base tabletop env (single-table layout in front of fixed-base X2)
# ---------------------------------------------------------------------------


class LMTabletopFixedBase(LocoManipulationEnv):
    """Single-table workspace placed in front of the (fixed-base) robot.

    Subclasses populate :attr:`mujoco_objects` and create their own props
    (cube, bowl, …) on top of the table.  Only one table is added; for
    table-to-table layouts, use :class:`PnPBottleTableToTable` from
    ``locomanip.py`` instead.
    """

    TABLE_GRADIENT: Gradient = Gradient(
        np.array([0.66, 0.46, 0.30, 1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    )
    # Table center is pushed forward to ``x=0.62`` (was 0.55) so the table's
    # back edge lands at ``x = 0.62 - 0.368 = 0.252`` -- about 5 cm ahead of
    # the X2's right OmniHand idle reach (``x_max ~ 0.20``).  Without this
    # shift the OmniHand thumb spawns inside the table top slab, producing
    # a non-physical "pop" on episode start and spurious force-torque
    # readings on the wrist sensor.  The 90deg rotation puts the lab table's
    # short side (0.74 m) toward the robot so the workspace depth stays
    # within comfortable arm-IK reach (cube spawn lives at ``table_center +
    # (-0.12, -0.04)`` -> ``x_world in [0.50, 0.58]`` m).
    TABLE_POS: list[float] = [0.62, 0.0, 0.0]
    TABLE_EULER: list[float] = [0.0, 0.0, np.pi / 2]

    def __init__(self, *args, **kwargs):
        self.objects: dict[str, dict] = {}
        super().__init__(*args, **kwargs)

    def _load_model(self):
        self.mujoco_objects = self._build_mujoco_objects()
        super()._load_model()

    def _build_mujoco_objects(self) -> list:
        """Hook returning the MJCFObjects merged BEFORE ``super()._load_model()``.

        Default: just the lab table. Subclasses that need additional
        real-mesh free-joint props (e.g. an upstream ``apple_0`` /
        ``jug_a01`` / ``plate_1`` from the locomanip asset library)
        should extend this list -- those props carry their own
        ``<asset>`` sections (textures + meshes) which only get merged
        into ``self.model`` if the object is in ``self.mujoco_objects``
        when robosuite's ``ManipulationTask`` is constructed in
        ``LocoManipulationEnv._load_model``.

        Primitive props (``PrimitiveCube``, ``PrimitiveBowl``,
        ``PrimitiveFixture``) don't need this hook -- they're inlined
        directly into ``self.model.worldbody`` by the subclass AFTER
        ``super()`` runs because their asset elements are tiny and
        don't depend on robosuite's per-object naming machinery.
        """
        return [self._create_table("table_body", self.TABLE_POS, self.TABLE_EULER)]

    @staticmethod
    def _create_table(name: str, position: list[float], euler: list[float]) -> MJCFObject:
        """Reuse the omniverse lab table that ``PnPBottle`` uses; gives us a
        consistent visual style across X2 / GR1 / G1 tabletop tasks."""
        table = MJCFObject(
            name=name,
            mjcf_path=xml_path_completion(
                "objects/omniverse/locomanip/lab_table/model.xml",
                root=robocasa.models.assets_root,
            ),
            scale=1.0,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.001, 1),
            density=10,
            friction=(1, 1, 1),
            static=True,
        )
        table.set_pos(position)
        table.set_euler(euler)
        return table

    def _table_top_z(self) -> float:
        """Runtime table-top z (handles dynamic table re-placement)."""
        body_id = self.sim.model.body_name2id("table_body_main")
        base_z = float(self.sim.data.body_xpos[body_id][2])
        top_offset_z = float(self.mujoco_objects[0].top_offset[2])
        return base_z + top_offset_z

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {}
        for prop_name, model in self.objects.items():
            self.obj_body_id[prop_name] = self.sim.model.body_name2id(model["name"])

    def _randomize_table_texture(self):
        randomize_materials_rgba(
            rng=self.rng, mjcf_obj=self.mujoco_objects[0], gradient=self.TABLE_GRADIENT, linear=True
        )


# ---------------------------------------------------------------------------
# X2 task variants
# ---------------------------------------------------------------------------


class X2PickPlaceCube(LMTabletopFixedBase, DexMGConfigHelper):
    """Pick the small red cube on the table and drop it inside the blue bowl.

    The cube spawn is randomised every reset within a tight box on the
    *source* side of the table; the bowl stays at a fixed pose on the
    *target* side.  Success is declared when the cube comes to rest above
    the bowl floor and inside the bowl walls (containment check).
    """

    CUBE_HALF_SIZE = 0.022
    BOWL_HALF_SIZE_XY = 0.075
    BOWL_WALL_HEIGHT = 0.04

    # Spawn ranges relative to the table top.  X is forward (away from
    # robot), Y is left/right.  The robot's right OmniHand reaches the
    # right side of the table comfortably, so the cube spawns slightly
    # right-of-center and the bowl stays slightly left.
    _CUBE_X_RANGE = (-0.12, -0.04)  # m, w.r.t. table center
    _CUBE_Y_RANGE = (-0.20, -0.10)
    _BOWL_X = -0.08  # fixed
    _BOWL_Y = +0.15

    def _load_model(self):
        super()._load_model()

        # Cube (free joint) -- gets re-placed in _reset_internal.
        self._cube = PrimitiveCube(name="cube", half_size=self.CUBE_HALF_SIZE)
        self.model.asset.extend(self._cube.assets)
        self.model.worldbody.append(self._cube.body)
        self.objects["cube"] = {"name": f"{self._cube.name}_body"}

        # Bowl (welded -- no joint).
        self._bowl = PrimitiveBowl(
            name="bowl",
            half_size_xy=self.BOWL_HALF_SIZE_XY,
            wall_height=self.BOWL_WALL_HEIGHT,
        )
        self.model.asset.extend(self._bowl.assets)
        self.model.worldbody.append(self._bowl.body)
        self.objects["bowl"] = {"name": f"{self._bowl.name}_body"}

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return

        # Place the bowl on the table top (welded body -- write into model).
        bowl_id = self.sim.model.body_name2id("bowl_body")
        x_table, y_table = float(self.sim.data.body_xpos[self.obj_body_id["bowl"]][0]), float(
            self.sim.data.body_xpos[self.obj_body_id["bowl"]][1]
        )
        # We don't actually want to drift from the table center -- recompute
        # from the table body so the bowl stays put even if the table moves.
        table_id = self.sim.model.body_name2id("table_body_main")
        table_x, table_y = (
            float(self.sim.data.body_xpos[table_id][0]),
            float(self.sim.data.body_xpos[table_id][1]),
        )
        z_top = self._table_top_z()
        self.sim.model.body_pos[bowl_id] = np.array(
            [table_x + self._BOWL_X, table_y + self._BOWL_Y, z_top]
        )

        # Place the cube within its sampling range, centred above the table.
        cube_x = self.rng.uniform(*self._CUBE_X_RANGE)
        cube_y = self.rng.uniform(*self._CUBE_Y_RANGE)
        # Add a small clearance so MuJoCo doesn't penetrate on first step.
        cube_z = z_top + self.CUBE_HALF_SIZE + 0.005

        qpos = self.sim.data.get_joint_qpos("cube_joint").copy()
        qpos[:3] = np.array([table_x + cube_x, table_y + cube_y, cube_z])
        # Random yaw only -- keep the cube aligned with the world frame.
        yaw = self.rng.uniform(-np.pi, np.pi)
        qpos[3:7] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.sim.data.set_joint_qpos("cube_joint", qpos)

        self._randomize_table_texture()

    # --- success ---

    def _check_success(self) -> bool:
        cube_pos = self.sim.data.body_xpos[self.obj_body_id["cube"]]
        bowl_pos = self.sim.data.body_xpos[self.obj_body_id["bowl"]]
        # Check cube is centred above the bowl (XY) and resting above the floor.
        in_xy = (
            abs(cube_pos[0] - bowl_pos[0])
            <= self.BOWL_HALF_SIZE_XY - self.CUBE_HALF_SIZE
        ) and (
            abs(cube_pos[1] - bowl_pos[1])
            <= self.BOWL_HALF_SIZE_XY - self.CUBE_HALF_SIZE
        )
        # Cube z must be inside [bowl_floor + cube_half, bowl_top + slack]
        bowl_floor_z = float(bowl_pos[2])
        cube_z = float(cube_pos[2])
        in_z = bowl_floor_z + self.CUBE_HALF_SIZE * 0.5 <= cube_z <= bowl_floor_z + self.BOWL_WALL_HEIGHT + 0.01
        # Make sure the cube is roughly upright (avoid tip-over false positives).
        upright = check_obj_upright(self, "cube", threshold=0.5, symmetric=True)
        return bool(in_xy and in_z and upright)

    # --- DexMG integration ---

    def get_object(self) -> dict:
        return dict(
            cube=dict(obj_name=self.objects["cube"]["name"], obj_type="body"),
            bowl=dict(obj_name=self.objects["bowl"]["name"], obj_type="body"),
        )

    def get_subtask_term_signals(self) -> dict[str, int]:
        # Right OmniHand grasps the cube.
        check_grasp_right = self._check_grasp(
            self.robots[0].gripper["right"], self._cube.contact_geoms
        )
        return {"grasp_cube": int(check_grasp_right)}

    @staticmethod
    def task_config() -> dict:
        task = DexMGConfigHelper.AttrDict()
        # Subtask 1 -- right hand grasps the cube.
        task.task_spec_0.subtask_1 = dict(
            object_ref="cube",
            subtask_term_signal="grasp_cube",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Subtask 2 -- right hand drops the cube into the bowl.
        task.task_spec_0.subtask_2 = dict(
            object_ref="bowl",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Idle filler for the (unused) left arm spec.
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


class X2PickPlaceBowl(LMTabletopFixedBase, DexMGConfigHelper):
    """Pick the bowl from the table and place it onto a marked target zone.

    The bowl sits on the source side of the table at start; the target
    fixture (an invisible flat box) sits on the opposite side.  Success
    is declared when the bowl rests on the target fixture and is upright.
    The cube remains as a visual distractor (no joint action on it).
    """

    BOWL_HALF_SIZE_XY = 0.07
    BOWL_WALL_HEIGHT = 0.035
    TARGET_HALF_SIZE = np.array([0.08, 0.08, 0.001])

    _BOWL_X_RANGE = (-0.12, -0.04)
    _BOWL_Y_RANGE = (-0.18, -0.10)
    _TARGET_X_RANGE = (-0.10, 0.0)
    _TARGET_Y_RANGE = (0.10, 0.18)

    def _load_model(self):
        super()._load_model()

        # Bowl: this version *does* get a free joint so the policy can
        # pick it up.  We accomplish this by adding a free joint at the
        # bowl body level.
        self._bowl = PrimitiveBowl(
            name="bowl",
            half_size_xy=self.BOWL_HALF_SIZE_XY,
            wall_height=self.BOWL_WALL_HEIGHT,
        )
        self._bowl.body.append(
            ET.Element("joint", name="bowl_joint", type="free", damping="0.0005")
        )
        self.model.asset.extend(self._bowl.assets)
        self.model.worldbody.append(self._bowl.body)
        self.objects["bowl"] = {"name": f"{self._bowl.name}_body"}

        # Target zone (invisible flat box, just for contact-based success).
        self._target = PrimitiveFixture(
            name="target",
            pos=np.array([0.0, 0.0, 0.0]),
            half_size=self.TARGET_HALF_SIZE,
            rgb=[0.10, 0.65, 0.20],
        )
        # Keep it dim-but-visible so demos look sensible.
        self.model.asset.extend(self._target.assets)
        self.model.worldbody.append(self._target.body)
        self.objects["target"] = {"name": "target_body"}

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return

        table_id = self.sim.model.body_name2id("table_body_main")
        table_x, table_y = (
            float(self.sim.data.body_xpos[table_id][0]),
            float(self.sim.data.body_xpos[table_id][1]),
        )
        z_top = self._table_top_z()

        # Place the target fixture (welded -- write into model.body_pos).
        target_id = self.sim.model.body_name2id("target_body")
        tx = self.rng.uniform(*self._TARGET_X_RANGE)
        ty = self.rng.uniform(*self._TARGET_Y_RANGE)
        self.sim.model.body_pos[target_id] = np.array([table_x + tx, table_y + ty, z_top])

        # Place the bowl with a small clearance so MuJoCo doesn't penetrate.
        bx = self.rng.uniform(*self._BOWL_X_RANGE)
        by = self.rng.uniform(*self._BOWL_Y_RANGE)
        bowl_z = z_top + self.PrimitiveBowl_floor_thickness() + 0.002
        qpos = self.sim.data.get_joint_qpos("bowl_joint").copy()
        qpos[:3] = np.array([table_x + bx, table_y + by, bowl_z])
        qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.sim.data.set_joint_qpos("bowl_joint", qpos)

        self._randomize_table_texture()

    @staticmethod
    def PrimitiveBowl_floor_thickness() -> float:
        return PrimitiveBowl.WALL_THICKNESS

    # --- success ---

    def _check_success(self) -> bool:
        bowl_on_target = self.check_contact(
            self._bowl.contact_geoms, ["target_collider"]
        )
        bowl_upright = check_obj_upright(self, "bowl", threshold=0.7, symmetric=True)
        return bool(bowl_on_target and bowl_upright)

    def get_object(self) -> dict:
        return dict(
            bowl=dict(obj_name=self.objects["bowl"]["name"], obj_type="body"),
            target=dict(obj_name=self.objects["target"]["name"], obj_type="body"),
        )

    def get_subtask_term_signals(self) -> dict[str, int]:
        bowl_z = self.sim.data.body_xpos[self.obj_body_id["bowl"]][2]
        return {"bowl_off_table": int(bowl_z > self._table_top_z() + 0.10)}

    @staticmethod
    def task_config() -> dict:
        task = DexMGConfigHelper.AttrDict()
        task.task_spec_0.subtask_1 = dict(
            object_ref="bowl",
            subtask_term_signal="bowl_off_table",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_0.subtask_2 = dict(
            object_ref="target",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()


class X2PickPlaceApple(LMTabletopFixedBase, DexMGConfigHelper):
    """Pick the red apple on the table and drop it inside the blue bowl.

    Real-mesh sibling of :class:`X2PickPlaceCube`. Where the cube is a
    flat-shaded primitive box, the apple is the upstream ``apple_0``
    locomanip asset -- a textured visual mesh wrapped around a
    5-fragment convex-decomposition collision shell. That makes this
    scene useful for:

    * **Real-mesh visual variety in the recorded ego frames.** The cube
      task records a flat red blob; the apple records a textured object
      with proper shading + albedo. VLA training can mix the two cleanly
      because the table layout, bowl, and success criterion are
      identical -- only the manipulable object's geometry / texture
      differs.
    * **Non-convex grasp object.** The apple's curved sides + stem
      indent are harder to "trap" between OmniHand fingertips than the
      sharp-cornered cube, so a policy trained on cube + apple sees a
      richer grasp-pose distribution and is less likely to overfit to
      "just close fingers around any small object on the table".

    The apple body is loaded via :class:`MJCFObject` (added to
    ``self.mujoco_objects`` before ``super()._load_model()`` so robosuite
    merges its asset section into ``self.model``); the bowl is the same
    welded :class:`PrimitiveBowl` used by :class:`X2PickPlaceCube`.

    Conventions (mirrored from :class:`X2PickPlaceCube` so per-episode
    placements + success criteria stay comparable across the two
    scenes):

    * Apple spawn is randomised every reset within a tight box on the
      *source* (right) side of the table.
    * Bowl is welded at a fixed pose on the *target* (left) side of
      the table.
    * Success fires when the apple comes to rest above the bowl floor
      and inside the bowl walls. Uprightness is *not* enforced -- the
      apple is roughly spherical, so any settled orientation that
      keeps it inside the bowl footprint is acceptable.
    """

    # Apple geometry. The asset's intrinsic mesh scale is 0.075 so a
    # MJCFObject(scale=1.0) gives an apple ~7.4 cm tall and ~7.4 cm
    # across (top_site - bottom_site = 0.0738 m, horizontal_radius
    # ~0.052 m projected). We store the half-extents below for the
    # success-check footprint math; both default to 0.037 because we
    # treat the apple as roughly spherical.
    APPLE_HALF_HEIGHT: float = 0.037
    APPLE_HALF_WIDTH: float = 0.037

    # Bowl geometry. Wall height bumped from the cube's 0.04 to 0.045
    # because the apple is ~3x taller than the cube and would otherwise
    # roll out of the bowl on a soft drop. Half-size matches
    # X2PickPlaceCube so the operator's calibrated hand-to-bowl reach
    # transfers across scenes without re-tuning.
    BOWL_HALF_SIZE_XY: float = 0.075
    BOWL_WALL_HEIGHT: float = 0.045

    # Spawn ranges relative to the table top.  X is forward (away from
    # robot), Y is left/right.  Mirror X2PickPlaceCube so the per-
    # operator calibration that works for cube spawns also works here.
    _APPLE_X_RANGE: tuple[float, float] = (-0.12, -0.04)
    _APPLE_Y_RANGE: tuple[float, float] = (-0.20, -0.10)
    _BOWL_X: float = -0.08
    _BOWL_Y: float = +0.15

    # Approximate world-z of the lab-table top with TABLE_POS[2]=0 and
    # the omniverse lab_table mesh's top_offset. Used ONLY for the
    # apple's static-XML default pose (so the apple is visible at
    # viewer startup before any per-episode scene_reset has fired).
    # Per-episode placement uses the runtime ``self._table_top_z()``,
    # which queries the actual MuJoCo body position; that path stays
    # robust to future table-pose changes.
    _STATIC_TABLE_TOP_Z: float = 0.684446

    def _build_mujoco_objects(self) -> list:
        # Append the apple to the table list returned by the base class.
        # The apple has to land in ``self.mujoco_objects`` (and stay
        # there) before ``super()._load_model()`` runs so robosuite
        # merges its <asset> section (textures + 6 meshes for the
        # convex decomposition) into ``self.model`` AND wraps its body
        # with the auto-generated free joint ``apple_joint0``. If we
        # tried to append the apple AFTER super (the way the cube /
        # bowl primitives are appended), the asset references would be
        # dangling.
        self._apple = self._create_apple()

        # Set a sensible static-XML default pose so the apple is
        # visible on the table top from the moment MuJoCo finishes
        # compiling the model -- not just after the first per-episode
        # scene_reset has fired.  Without this, ``MJCFObject`` leaves
        # the body's ``pos`` attribute unset, MuJoCo defaults the
        # free-joint qpos to ``[0, 0, 0, 1, 0, 0, 0]``, and the apple
        # ends up buried at world origin (under the floor) until the
        # operator presses B to start an episode.  The cube / bowl
        # primitives don't show this failure mode because their
        # ``__init__`` hardcodes ``pos="0.45 0 0.85"`` /
        # ``pos="0.54 0.15 …"`` directly into the body element.
        #
        # The default position is the midpoint of the spawn range so
        # the very first viewer frame shows the apple sitting at the
        # centre of the per-episode randomisation window, on top of
        # the lab table.  Per-episode resets in
        # :meth:`_reset_internal` then jitter the apple within
        # ``_APPLE_X_RANGE`` / ``_APPLE_Y_RANGE``.
        ax_mid = 0.5 * (self._APPLE_X_RANGE[0] + self._APPLE_X_RANGE[1])
        ay_mid = 0.5 * (self._APPLE_Y_RANGE[0] + self._APPLE_Y_RANGE[1])
        default_pos = [
            float(self.TABLE_POS[0] + ax_mid),
            float(self.TABLE_POS[1] + ay_mid),
            float(self._STATIC_TABLE_TOP_Z + self.APPLE_HALF_HEIGHT + 0.005),
        ]
        self._apple.set_pos(default_pos)

        return [*super()._build_mujoco_objects(), self._apple]

    def _load_model(self):
        super()._load_model()

        # Bowl is a Primitive (welded -- no joint), inlined directly into
        # the world body the same way ``X2PickPlaceCube`` inlines its
        # primitive cube + bowl. Asset elements live in the bowl object
        # itself, so no robosuite-side merge is needed.
        self._bowl = PrimitiveBowl(
            name="bowl",
            half_size_xy=self.BOWL_HALF_SIZE_XY,
            wall_height=self.BOWL_WALL_HEIGHT,
        )
        self.model.asset.extend(self._bowl.assets)
        self.model.worldbody.append(self._bowl.body)
        self.objects["bowl"] = {"name": f"{self._bowl.name}_body"}

        # Robosuite gives the MJCFObject the body name ``<name>_main`` and
        # a single auto-named free joint ``<name>_joint0`` (because we
        # passed ``static=False``). Cache both so the success checks +
        # SceneEnvSpec metadata don't have to recompute them.
        self.objects["apple"] = {
            "name": self._apple.root_body,
            "joint": self._apple.joints[0],
        }

    @staticmethod
    def _create_apple() -> MJCFObject:
        return MJCFObject(
            name="apple",
            mjcf_path=xml_path_completion(
                _APPLE_MJCF_PATH, root=robocasa.models.assets_root
            ),
            scale=1.0,
            # Real apple density is ~800 kg/m^3; we use 80 here to match
            # the ``LMPnPAppleToPlate`` upstream env's apple-mass tuning,
            # which keeps the OmniHand from spiking the position-actuator
            # force budget on a power grasp.
            density=80,
            friction=(1.0, 0.3, 0.05),
            static=False,
        )

    def _reset_internal(self):
        super()._reset_internal()
        if self.deterministic_reset:
            return

        # Place the bowl on the table top (welded body -- write into model).
        bowl_id = self.sim.model.body_name2id("bowl_body")
        table_id = self.sim.model.body_name2id("table_body_main")
        table_x = float(self.sim.data.body_xpos[table_id][0])
        table_y = float(self.sim.data.body_xpos[table_id][1])
        z_top = self._table_top_z()
        self.sim.model.body_pos[bowl_id] = np.array(
            [table_x + self._BOWL_X, table_y + self._BOWL_Y, z_top]
        )

        # Place the apple within its sampling range, centred above the table.
        ax = self.rng.uniform(*self._APPLE_X_RANGE)
        ay = self.rng.uniform(*self._APPLE_Y_RANGE)
        # Add a small clearance so MuJoCo doesn't penetrate on first step.
        # The apple's body origin sits at the asset's mesh centre, so a
        # z offset of (half_height + clearance) leaves the bottom-most
        # collision fragment ~5 mm above the table top.
        apple_z = z_top + self.APPLE_HALF_HEIGHT + 0.005

        joint_name = self.objects["apple"]["joint"]
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:3] = np.array([table_x + ax, table_y + ay, apple_z])
        # Random yaw only -- keep the apple's stem aligned with world +z so
        # its first-contact pose is repeatable across episodes.
        yaw = self.rng.uniform(-np.pi, np.pi)
        qpos[3:7] = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        self.sim.data.set_joint_qpos(joint_name, qpos)

        self._randomize_table_texture()

    # --- success ---

    def _check_success(self) -> bool:
        apple_pos = self.sim.data.body_xpos[self.obj_body_id["apple"]]
        bowl_pos = self.sim.data.body_xpos[self.obj_body_id["bowl"]]
        # Apple xy must sit inside the bowl footprint (with the apple's
        # own horizontal half-width subtracted so the test still passes
        # for an apple resting against a wall).
        in_xy = (
            abs(apple_pos[0] - bowl_pos[0])
            <= self.BOWL_HALF_SIZE_XY - self.APPLE_HALF_WIDTH
        ) and (
            abs(apple_pos[1] - bowl_pos[1])
            <= self.BOWL_HALF_SIZE_XY - self.APPLE_HALF_WIDTH
        )
        # Apple z must be inside [bowl_floor + apple_half, bowl_top + slack].
        # The +0.04 slack on the upper bound is generous because a real
        # apple settles on its side with the stem poking above the rim.
        bowl_floor_z = float(bowl_pos[2])
        apple_z = float(apple_pos[2])
        in_z = (
            bowl_floor_z + self.APPLE_HALF_HEIGHT * 0.5
            <= apple_z
            <= bowl_floor_z + self.BOWL_WALL_HEIGHT + 0.04
        )
        # Apple is roughly spherical -- skip uprightness check (any
        # settled orientation inside the bowl is acceptable).
        return bool(in_xy and in_z)

    # --- DexMG integration ---

    def get_object(self) -> dict:
        return dict(
            apple=dict(obj_name=self.objects["apple"]["name"], obj_type="body"),
            bowl=dict(obj_name=self.objects["bowl"]["name"], obj_type="body"),
        )

    def get_subtask_term_signals(self) -> dict[str, int]:
        # Right OmniHand grasps the apple. ``MJCFObject.contact_geoms``
        # returns the auto-named collision-class geoms (``apple_g1`` ..
        # ``apple_g5`` for the 5 convex-decomposition fragments).
        check_grasp_right = self._check_grasp(
            self.robots[0].gripper["right"], self._apple.contact_geoms
        )
        return {"grasp_apple": int(check_grasp_right)}

    @staticmethod
    def task_config() -> dict:
        task = DexMGConfigHelper.AttrDict()
        # Subtask 1 -- right hand grasps the apple.
        task.task_spec_0.subtask_1 = dict(
            object_ref="apple",
            subtask_term_signal="grasp_apple",
            subtask_term_offset_range=(5, 10),
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Subtask 2 -- right hand drops the apple into the bowl.
        task.task_spec_0.subtask_2 = dict(
            object_ref="bowl",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        # Idle filler for the (unused) left arm spec -- mirrors the
        # X2PickPlaceCube layout so DexMG demo collation treats the two
        # scenes interchangeably.
        task.task_spec_1.subtask_1 = dict(
            object_ref=None,
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="random",
            selection_strategy_kwargs=None,
            action_noise=0.05,
            num_interpolation_steps=5,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        return task.to_dict()
