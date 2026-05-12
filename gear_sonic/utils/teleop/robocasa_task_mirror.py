"""Per-tick task oracle for X2 + robocasa scene recording.

Architecture
------------

The G1 architecture (see ``X2_INTEGRATION_NOTES.md``) keeps the C++ deploy
bridge as the only process that runs MuJoCo physics. Robosuite is loaded
inside the recorder process for two purposes only:

1. **Per-episode randomization**:  At episode start we call
   :meth:`RobocasaTaskMirror.reset` which steps the matching robosuite env
   forward by one ``env.reset()`` (running its full
   :class:`placement_initializer` / per-task ``_reset_internal`` logic),
   reads the resulting object freejoint qpos + welded body positions,
   then publishes those to the deploy bridge over the ``reset_objects``
   ZMQ topic.

2. **Per-tick success / reward / subtask labelling**: The bridge
   publishes a small ``scene_state`` snapshot (cube qpos, bowl qpos,
   contact summary) at 50 Hz. The recorder feeds those into
   :meth:`RobocasaTaskMirror.sync_from_state`, then queries
   :meth:`check_success` / :meth:`compute_reward` / :meth:`subtask_signals`.
   These methods read **the mirror's** ``MjModel`` / ``MjData`` (which
   loaded the same static scene XML the bridge sees) so contact / geom
   / body addresses match wholesale.

The mirror **never** calls ``env.step()`` -- it's a stateless oracle. We
keep the robosuite env handle around only because :meth:`reset` borrows
its placement_initializer; everything else is a thin pure-MuJoCo helper.

Why the mirror loads its own MJCF instead of just using the env's
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The robocasa env uses ``X2UltraFixedLowerBody`` (legs welded out) and
robosuite's gripper attachment (``gripper0_left_L_*`` joint prefixing).
The deploy bridge expects the canonical 31-DOF ``compose_x2_with_omnihand``
joint layout. The static scene XML (built by
``gear_sonic/scripts/build_x2_robocasa_scene_xml.py``) has the deploy's
expected layout PLUS the env's table + cube + bowl bodies grafted in. By
loading that same XML on the mirror side we guarantee that address-based
state copies between deploy and mirror are byte-compatible -- no name
translation table needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
from typing import Any, Callable, Mapping, Optional

import mujoco
import numpy as np


_LOG = logging.getLogger(__name__)


# ── Wire schema (mirror <-> deploy bridge) ────────────────────────────────


@dataclass
class SceneState:
    """Per-tick snapshot of the scene-only state in the deploy bridge.

    The bridge publishes this on the ``scene_state`` ZMQ topic at the
    same rate as ``x2_debug`` (50 Hz). Only the scene-object freejoints
    + mutable welded-body positions live here -- the X2 / OmniHand
    proprio is already covered by ``x2_debug``.
    """

    sim_time: float = 0.0
    """Bridge sim time at the moment of publish (seconds)."""

    object_freejoint_qpos: dict[str, list[float]] = field(default_factory=dict)
    """``{joint_name: 7-vec qpos}`` for every scene freejoint
    (e.g. ``"cube_joint": [x, y, z, qw, qx, qy, qz]``)."""

    mutable_body_pos: dict[str, list[float]] = field(default_factory=dict)
    """``{body_name: 3-vec world pos}`` for welded bodies whose position
    can change at episode reset (e.g. ``"bowl_body"`` for X2PickPlaceCube)."""

    grasp_contacts: dict[str, dict[str, bool]] = field(default_factory=dict)
    """``{logical_object_name: {"left": bool, "right": bool, "any": bool}}``
    summarising per-side OmniHand-vs-object contact. The bridge fills
    this in by walking ``mj_data.contact[:ncon]`` and matching geoms
    against the object's contact_geoms list and the side's hand-geom set.
    ``"any"`` is True when the object touched anything at all (table,
    floor, other object included) -- useful for the "object resting
    against world" check independent of which hand holds it."""

    fingertip_pos: dict[str, list[list[float]]] = field(default_factory=dict)
    """``{side: [[x, y, z], …]}`` for each fingertip body listed in the
    scene metadata's ``fingertip_bodies``. Empty when the bridge isn't
    a robocasa-aware build. The mirror's shaped reward uses
    ``min(distance(tip, cube))`` for the ``approach_cube`` phase."""

    @classmethod
    def from_dict(cls, d: dict) -> "SceneState":
        return cls(
            sim_time=float(d.get("sim_time", 0.0)),
            object_freejoint_qpos={
                k: list(map(float, v))
                for k, v in d.get("object_freejoint_qpos", {}).items()
            },
            mutable_body_pos={
                k: list(map(float, v))
                for k, v in d.get("mutable_body_pos", {}).items()
            },
            grasp_contacts={
                k: {side: bool(b) for side, b in v.items()}
                for k, v in d.get("grasp_contacts", {}).items()
            },
            fingertip_pos={
                k: [list(map(float, p)) for p in v]
                for k, v in d.get("fingertip_pos", {}).items()
            },
        )

    def to_dict(self) -> dict:
        return dict(
            sim_time=float(self.sim_time),
            object_freejoint_qpos={
                k: list(v) for k, v in self.object_freejoint_qpos.items()
            },
            mutable_body_pos={
                k: list(v) for k, v in self.mutable_body_pos.items()
            },
            grasp_contacts={
                k: dict(v) for k, v in self.grasp_contacts.items()
            },
            fingertip_pos={
                k: [list(p) for p in v]
                for k, v in self.fingertip_pos.items()
            },
        )


@dataclass
class ResetObjects:
    """Wire payload published from the recorder to the bridge at episode start.

    The bridge subscribes on the ``reset_objects`` ZMQ topic and writes
    the per-joint qpos / per-body pos into its own ``mj_model`` /
    ``mj_data`` at receipt, then calls ``mj_forward`` so subsequent
    bridge ticks see the new initial state. See
    ``gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py`` for the
    receiver side.
    """

    object_freejoint_qpos: dict[str, list[float]] = field(default_factory=dict)
    mutable_body_pos: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return dict(
            object_freejoint_qpos={
                k: list(map(float, v)) for k, v in self.object_freejoint_qpos.items()
            },
            mutable_body_pos={
                k: list(map(float, v)) for k, v in self.mutable_body_pos.items()
            },
        )


# ── Per-env success / reward / subtask oracles ────────────────────────────
#
# Each task gets a tiny module of pure-Python helpers that read the
# mirror's MuJoCo state (post-sync_from_state). These are direct ports of
# the corresponding ``_check_success`` / ``reward`` / ``get_subtask_term_signals``
# methods in
# ``decoupled_wbc/dexmg/gr00trobocasa/robocasa/environments/locomanipulation/x2_tabletop_pnp.py``
# but rewritten to operate on a plain ``MjModel`` / ``MjData`` pair so
# they don't drag the robosuite env class along at runtime.
#
# Keep the per-env constants here (cube half-size, bowl wall height, …)
# in sync with the env class -- a unit test compares the two sources
# (see ``tests/test_robocasa_task_mirror.py``).


def _world_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"body {body_name!r} not in mirror model")
    return np.asarray(data.xpos[bid], dtype=np.float64)


def _world_quat_wxyz(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise KeyError(f"body {body_name!r} not in mirror model")
    return np.asarray(data.xquat[bid], dtype=np.float64)


def _is_upright(quat_wxyz: np.ndarray, axis: int = 2, threshold: float = 0.5,
                symmetric: bool = True) -> bool:
    """Mirror of ``robocasa.utils.object_utils.check_obj_upright``."""
    qw, qx, qy, qz = (float(quat_wxyz[i]) for i in range(4))
    # Rotate the world up vector (0,0,1) into the body frame? Actually
    # check_obj_upright tests that the object's local +z (or chosen axis)
    # aligns with world +z. The world Z component of the body's local
    # axis is the (2, axis) entry of the rotation matrix. For wxyz
    # quats the matrix entry R[2, 2] = 1 - 2*(qx^2 + qy^2).
    if axis == 2:
        z_world = 1.0 - 2.0 * (qx * qx + qy * qy)
    elif axis == 1:
        z_world = 2.0 * (qy * qz + qw * qx)
    elif axis == 0:
        z_world = 2.0 * (qx * qz - qw * qy)
    else:
        raise ValueError(f"axis must be 0|1|2, got {axis}")
    if symmetric:
        return abs(z_world) >= float(threshold)
    return z_world >= float(threshold)


@dataclass(frozen=True)
class _PickPlaceCubeConstants:
    cube_body: str = "cube_body"
    bowl_body: str = "bowl_body"
    table_body: str = "table_body_main"
    cube_half_size: float = 0.022
    bowl_half_size_xy: float = 0.075
    bowl_wall_height: float = 0.04
    # Approach phase fires when the closest fingertip is within this
    # distance of the cube centre. 8 cm is roughly an open-hand radius
    # plus a small slack -- "the operator clearly intends to grab".
    approach_distance: float = 0.08
    # Cube counts as "off the table" when its centre is at least this
    # far above the table top. Half-size + 1.5 cm gives a small but
    # unambiguous lift threshold.
    lift_above_table: float = 0.025
    # Cube counts as "above the bowl" when xy is inside (bowl half - cube half)
    # AND z is above bowl_floor + bowl_wall_height (i.e. cube has cleared
    # the rim from above).
    above_bowl_z_clearance: float = 0.0


def _min_fingertip_distance_to_cube(
    cube_pos: np.ndarray,
    fingertip_pos: Mapping[str, list[list[float]]],
) -> float:
    """Min Euclidean distance from any fingertip (either side) to the cube
    centre. Returns ``+inf`` when no fingertips were reported (e.g. the
    bridge isn't a robocasa-aware build)."""
    best = float("inf")
    for tips in fingertip_pos.values():
        for tip in tips:
            if len(tip) < 3:
                continue
            dx = float(tip[0]) - float(cube_pos[0])
            dy = float(tip[1]) - float(cube_pos[1])
            dz = float(tip[2]) - float(cube_pos[2])
            d = (dx * dx + dy * dy + dz * dz) ** 0.5
            if d < best:
                best = d
    return best


def _phase_pick_place_cube(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_contacts: Mapping[str, Mapping[str, bool]],
    fingertip_pos: Mapping[str, list[list[float]]],
    *,
    constants: _PickPlaceCubeConstants = _PickPlaceCubeConstants(),
) -> dict[str, bool]:
    """Compute the boolean state of every shaping phase for the cube task.

    Returned dict has stable keys ``approach_cube``, ``touch_cube``,
    ``grasp_cube``, ``cube_off_table``, ``cube_above_bowl``,
    ``cube_in_bowl``. Each is an independent indicator -- the helpers
    that turn this into a scalar reward / subtask map below are the ones
    that enforce monotonicity.
    """
    cube_pos = _world_pos(model, data, constants.cube_body)
    bowl_pos = _world_pos(model, data, constants.bowl_body)
    table_pos = _world_pos(model, data, constants.table_body)
    cube_grasp = grasp_contacts.get("cube", {})

    # Approach: closest fingertip near the cube.
    nearest = _min_fingertip_distance_to_cube(cube_pos, fingertip_pos)
    approach = nearest <= constants.approach_distance

    # Touch: any contact with the cube on either side.
    touch = bool(
        cube_grasp.get("left", False)
        or cube_grasp.get("right", False)
    )

    # Grasp: contact on the right side specifically. We mirror the
    # upstream env's "right-hand grasp" definition; if a future task
    # uses the left hand, broaden this to either-side or task-specific.
    grasp = bool(cube_grasp.get("right", False))

    # Lift: cube centre is above table_top + lift threshold AND the cube
    # is currently being grasped (otherwise a knock-over would count).
    table_top_z = float(table_pos[2])
    cube_off_table = (
        grasp
        and float(cube_pos[2]) > table_top_z + constants.lift_above_table
    )

    # Above-bowl: cube xy inside the bowl footprint and cube z above the
    # rim. Unlike "in bowl" this fires while the operator is still
    # carrying the cube, not waiting for it to settle.
    in_xy = (
        abs(float(cube_pos[0]) - float(bowl_pos[0]))
        <= constants.bowl_half_size_xy - constants.cube_half_size
    ) and (
        abs(float(cube_pos[1]) - float(bowl_pos[1]))
        <= constants.bowl_half_size_xy - constants.cube_half_size
    )
    bowl_floor_z = float(bowl_pos[2])
    above_bowl = bool(
        in_xy
        and float(cube_pos[2]) >= bowl_floor_z + constants.bowl_wall_height
        - constants.above_bowl_z_clearance
    )

    # In-bowl (== success): cube settled inside, upright.
    in_z = (
        bowl_floor_z + constants.cube_half_size * 0.5
        <= float(cube_pos[2])
        <= bowl_floor_z + constants.bowl_wall_height + 0.01
    )
    upright = _is_upright(_world_quat_wxyz(model, data, constants.cube_body))
    in_bowl = bool(in_xy and in_z and upright)

    return {
        "approach_cube": bool(approach),
        "touch_cube": bool(touch),
        "grasp_cube": bool(grasp),
        "cube_off_table": bool(cube_off_table),
        "cube_above_bowl": bool(above_bowl),
        "cube_in_bowl": bool(in_bowl),
    }


# Cumulative per-phase rewards, monotonic and bounded in [0, 1]. The
# reward at each tick is the maximum over all currently-active phases'
# weights; a successful demo therefore traces 0.00 -> 0.10 -> 0.25 ->
# 0.45 -> 0.65 -> 0.80 -> 1.00 across approach/touch/grasp/lift/carry/
# place. Tuning rationale:
#   * Approach is a small bonus (0.10) -- "you got close".
#   * Touch/grasp jumps reward a lot (0.25 -> 0.45) -- physical contact
#     is the hardest part for the VLA to learn.
#   * Lift + carry are the "transport" credit (0.65, 0.80).
#   * Success (1.00) is the only state with full reward.
_PICK_PLACE_CUBE_PHASE_REWARDS: tuple[tuple[str, float], ...] = (
    ("approach_cube", 0.10),
    ("touch_cube", 0.25),
    ("grasp_cube", 0.45),
    ("cube_off_table", 0.65),
    ("cube_above_bowl", 0.80),
    ("cube_in_bowl", 1.00),
)


def _success_pick_place_cube(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_contacts: Mapping[str, Mapping[str, bool]],
    fingertip_pos: Mapping[str, list[list[float]]],
    *,
    constants: _PickPlaceCubeConstants = _PickPlaceCubeConstants(),
) -> bool:
    """Mirror of ``X2PickPlaceCube._check_success``: cube centred inside
    the bowl footprint, vertically inside the wall-height window, and
    upright. Independent of the contact map -- a knocked-in cube counts."""
    phases = _phase_pick_place_cube(
        model, data, grasp_contacts, fingertip_pos, constants=constants
    )
    return phases["cube_in_bowl"]


def _reward_pick_place_cube(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_contacts: Mapping[str, Mapping[str, bool]],
    fingertip_pos: Mapping[str, list[list[float]]],
    *,
    constants: _PickPlaceCubeConstants = _PickPlaceCubeConstants(),
) -> float:
    """Phase-shaped reward in ``[0, 1]``. See
    :data:`_PICK_PLACE_CUBE_PHASE_REWARDS` for the per-phase weights."""
    phases = _phase_pick_place_cube(
        model, data, grasp_contacts, fingertip_pos, constants=constants
    )
    score = 0.0
    for phase, weight in _PICK_PLACE_CUBE_PHASE_REWARDS:
        if phases.get(phase, False):
            if weight > score:
                score = weight
    return float(score)


def _subtasks_pick_place_cube(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_contacts: Mapping[str, Mapping[str, bool]],
    fingertip_pos: Mapping[str, list[list[float]]],
    *,
    constants: _PickPlaceCubeConstants = _PickPlaceCubeConstants(),
) -> dict[str, int]:
    """Per-tick subtask termination signals for X2PickPlaceCube.

    Returns the same keys that ``_phase_pick_place_cube`` produces (plus
    ``grasp_cube`` is the upstream-env signal mirrored here for
    schema-compatibility with the original
    ``X2PickPlaceCube.get_subtask_term_signals``). Each value is 0 or 1.
    """
    phases = _phase_pick_place_cube(
        model, data, grasp_contacts, fingertip_pos, constants=constants
    )
    return {name: int(active) for name, active in phases.items()}


@dataclass(frozen=True)
class _PickPlaceBowlConstants:
    bowl_body: str = "bowl_body"
    target_body: str = "target_body"
    table_top_z: float = 0.74
    """Approximate table-top height in world coordinates -- used by the
    bowl-off-table subtask signal. The static scene XML pins the table
    rigid to the world, so this is constant."""


def _success_pick_place_bowl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_contacts: Mapping[str, Mapping[str, bool]],
    *,
    constants: _PickPlaceBowlConstants = _PickPlaceBowlConstants(),
) -> bool:
    """Mirror of ``X2PickPlaceBowl._check_success``.

    The original env tests ``check_contact(bowl_geoms, ["target_collider"])``
    which requires live contact arrays -- those are forwarded by the bridge
    in :attr:`SceneState.grasp_contacts` under the logical key ``"bowl_target"``.
    Geometric distance falls back as a heuristic if the bridge didn't
    publish that signal (e.g. older bridge build).
    """
    bowl_pos = _world_pos(model, data, constants.bowl_body)
    target_pos = _world_pos(model, data, constants.target_body)
    contact_signal = grasp_contacts.get("bowl_target", {}).get("any", None)
    if contact_signal is None:
        # Fallback: bowl is "on target" if its xy is within target xy
        # bounds and its z is within ~5cm of the target z.
        bowl_on_target = (
            abs(float(bowl_pos[0]) - float(target_pos[0])) <= 0.08
            and abs(float(bowl_pos[1]) - float(target_pos[1])) <= 0.08
            and abs(float(bowl_pos[2]) - float(target_pos[2])) <= 0.05
        )
    else:
        bowl_on_target = bool(contact_signal)
    bowl_upright = _is_upright(
        _world_quat_wxyz(model, data, constants.bowl_body), threshold=0.7
    )
    return bool(bowl_on_target and bowl_upright)


def _reward_pick_place_bowl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_contacts: Mapping[str, Mapping[str, bool]],
    *,
    constants: _PickPlaceBowlConstants = _PickPlaceBowlConstants(),
) -> float:
    return 1.0 if _success_pick_place_bowl(model, data, grasp_contacts,
                                           constants=constants) else 0.0


def _subtasks_pick_place_bowl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    grasp_contacts: Mapping[str, Mapping[str, bool]],
    *,
    constants: _PickPlaceBowlConstants = _PickPlaceBowlConstants(),
) -> dict[str, int]:
    bowl_pos = _world_pos(model, data, constants.bowl_body)
    bowl_off_table = int(float(bowl_pos[2]) > constants.table_top_z + 0.10)
    return {"bowl_off_table": bowl_off_table}


# ── Per-env oracle dispatch table ─────────────────────────────────────────


@dataclass(frozen=True)
class _TaskOracle:
    """Per-env collection of pure-Python success/reward/subtask functions."""

    success_fn: Callable[..., bool]
    reward_fn: Callable[..., float]
    subtasks_fn: Callable[..., dict[str, int]]

    needs_contacts: bool = False
    """If True, the success/reward/subtask functions read
    :attr:`SceneState.grasp_contacts` and the recorder should
    forward those from the bridge."""


_ORACLES: dict[str, _TaskOracle] = {
    "X2PickPlaceCube": _TaskOracle(
        success_fn=_success_pick_place_cube,
        reward_fn=_reward_pick_place_cube,
        subtasks_fn=_subtasks_pick_place_cube,
        needs_contacts=True,  # all phases consume the contact map
    ),
    "X2PickPlaceBowl": _TaskOracle(
        success_fn=_success_pick_place_bowl,
        reward_fn=_reward_pick_place_bowl,
        subtasks_fn=_subtasks_pick_place_bowl,
        needs_contacts=True,  # success 'bowl_on_target' uses contacts
    ),
}


# Static list of the subtask signals each env publishes -- the recorder
# pre-registers matching ``task.subtask_<name>`` columns at construction
# time so the LeRobot exporter's frame validator accepts them.
#
# Keep these in sync with what ``_subtasks_*`` actually returns. If you
# rename a phase, update both this table and the per-env ``_subtasks_*``
# function -- the unit test in
# ``tests/test_x2_robocasa_scene_mode.py::test_mirror_advertises_static_subtask_signals``
# will catch drift.
_STATIC_SUBTASK_SIGNALS: dict[str, tuple[str, ...]] = {
    "X2PickPlaceCube": (
        "approach_cube",
        "touch_cube",
        "grasp_cube",
        "cube_off_table",
        "cube_above_bowl",
        "cube_in_bowl",
    ),
    "X2PickPlaceBowl": (
        "bowl_off_table",
    ),
}


# ── The mirror itself ─────────────────────────────────────────────────────


class RobocasaTaskMirror:
    """See module docstring for the role this class plays.

    Construction is fast (just loads a MJCF). The first
    :meth:`reset` call lazily instantiates a robosuite env, which is
    relatively slow (~3-5 seconds, robosuite + robocasa import + scene
    compile) -- subsequent resets reuse the same env handle.
    """

    def __init__(
        self,
        *,
        scene_xml_path: Path,
        scene_metadata: Mapping[str, Any],
        env_name: Optional[str] = None,
    ) -> None:
        self._scene_xml_path = Path(scene_xml_path)
        if not self._scene_xml_path.is_file():
            raise FileNotFoundError(
                f"scene XML not found: {self._scene_xml_path}. "
                f"Build it first via "
                f"gear_sonic/scripts/build_x2_robocasa_scene_xml.py."
            )
        self._scene_metadata = dict(scene_metadata)
        self._env_name = env_name or self._scene_metadata.get("env_name")
        if self._env_name is None:
            raise ValueError("env_name must be set (in metadata or via kwarg)")
        if self._env_name not in _ORACLES:
            raise ValueError(
                f"no task oracle registered for env {self._env_name!r}. "
                f"Add one to robocasa_task_mirror._ORACLES."
            )
        self._oracle = _ORACLES[self._env_name]

        self._task_string = str(self._scene_metadata.get("task_string", ""))

        # Mirror's MuJoCo (loaded from same XML the deploy bridge sees).
        self.mj_model = mujoco.MjModel.from_xml_path(str(self._scene_xml_path))
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        # Resolve per-object joint / body addresses ONCE.
        self._object_freejoint_map = dict(
            self._scene_metadata.get("object_freejoint_map", {})
        )
        self._object_welded_map = dict(
            self._scene_metadata.get("object_welded_map", {})
        )
        self._freejoint_qadr: dict[str, int] = {}
        for logical, jname in self._object_freejoint_map.items():
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(
                    f"freejoint {jname!r} (logical={logical!r}) missing from "
                    f"mirror MJCF {self._scene_xml_path}"
                )
            self._freejoint_qadr[logical] = int(self.mj_model.jnt_qposadr[jid])
        self._welded_bid: dict[str, int] = {}
        for logical, bname in self._object_welded_map.items():
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                raise RuntimeError(
                    f"welded body {bname!r} (logical={logical!r}) missing "
                    f"from mirror MJCF {self._scene_xml_path}"
                )
            self._welded_bid[logical] = int(bid)

        # Last seen scene state (for diagnostics).
        self._last_state: Optional[SceneState] = None
        self._latest_grasp_contacts: dict[str, dict[str, bool]] = {}
        self._latest_fingertip_pos: dict[str, list[list[float]]] = {}

        # Lazy robosuite env (only built when reset() is called).
        self._robosuite_env: Any | None = None

    # ──────────────────────────────────────────────────────────────────
    # Read-only surface
    # ──────────────────────────────────────────────────────────────────

    @property
    def env_name(self) -> str:
        return self._env_name

    @property
    def task_string(self) -> str:
        return self._task_string

    @property
    def needs_contacts(self) -> bool:
        return self._oracle.needs_contacts

    @property
    def freejoint_names(self) -> list[str]:
        return list(self._object_freejoint_map.values())

    @property
    def mutable_body_names(self) -> list[str]:
        return list(self._object_welded_map.values())

    @property
    def static_subtask_names(self) -> tuple[str, ...]:
        """Names of the subtask signals this env always reports.

        The recorder pre-registers a ``task.subtask_<name>`` column per
        entry so the LeRobot exporter accepts the rows even when the
        bridge is mid-startup and the mirror returns empty oracle output.
        """
        return _STATIC_SUBTASK_SIGNALS.get(self._env_name, ())

    # ──────────────────────────────────────────────────────────────────
    # Episode lifecycle
    # ──────────────────────────────────────────────────────────────────

    def _ensure_robosuite_env(self) -> None:
        if self._robosuite_env is not None:
            return
        import os
        os.environ.setdefault("MUJOCO_GL", "egl")

        import robosuite as suite
        import robocasa  # noqa: F401  -- side-effect registration
        import robocasa.models.robots  # noqa: F401  -- registers X2 robots
        import robocasa.models.grippers  # noqa: F401  -- registers OmniHand
        from robocasa.models.grippers.omnihand_grippers import (
            load_x2_default_controller_config,
        )
        _LOG.info("instantiating robosuite env %s for randomization", self._env_name)
        self._robosuite_env = suite.make(
            env_name=self._env_name,
            robots="X2UltraFixedLowerBody",
            controller_configs=load_x2_default_controller_config(),
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            horizon=1000,  # large horizon -- we never let the env truncate us
        )

    def reset(self, seed: Optional[int] = None) -> ResetObjects:
        """Sample fresh object poses; return them so the recorder can push
        them to the deploy bridge AND apply them to the mirror's MuJoCo
        in one go.

        ``seed=None`` lets the env's RNG draw from the global numpy state
        (default robosuite behaviour). Pass an int for reproducible
        episode setups (e.g. unit tests).
        """
        self._ensure_robosuite_env()
        env = self._robosuite_env
        if seed is not None:
            np.random.seed(int(seed))
            env.rng = np.random.RandomState(int(seed))
        env.reset()

        reset_payload = ResetObjects()
        for logical, jname in self._object_freejoint_map.items():
            try:
                qpos = env.sim.data.get_joint_qpos(jname).copy().tolist()
            except (KeyError, ValueError):
                _LOG.warning("env did not produce freejoint %r at reset", jname)
                continue
            reset_payload.object_freejoint_qpos[jname] = qpos
        for logical, bname in self._object_welded_map.items():
            try:
                bid = env.sim.model.body_name2id(bname)
                pos = env.sim.data.body_xpos[bid].copy().tolist()
            except Exception:
                _LOG.warning("env did not produce welded body %r at reset", bname)
                continue
            reset_payload.mutable_body_pos[bname] = pos

        # Apply to mirror so its initial state matches what the deploy
        # will see after the reset_objects message lands.
        self._apply_reset_to_mirror(reset_payload)

        return reset_payload

    def _apply_reset_to_mirror(self, reset_payload: ResetObjects) -> None:
        for jname, qpos in reset_payload.object_freejoint_qpos.items():
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                _LOG.warning("mirror missing freejoint %r; skipping", jname)
                continue
            qadr = int(self.mj_model.jnt_qposadr[jid])
            self.mj_data.qpos[qadr:qadr + 7] = np.asarray(qpos, dtype=np.float64)
        for bname, pos in reset_payload.mutable_body_pos.items():
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                _LOG.warning("mirror missing welded body %r; skipping", bname)
                continue
            self.mj_model.body_pos[bid] = np.asarray(pos, dtype=np.float64)
        # Zero scene-object velocities so the static state is consistent.
        for jname in reset_payload.object_freejoint_qpos:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            vadr = int(self.mj_model.jnt_dofadr[jid])
            self.mj_data.qvel[vadr:vadr + 6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # ──────────────────────────────────────────────────────────────────
    # Per-tick state sync
    # ──────────────────────────────────────────────────────────────────

    def sync_from_state(self, state: SceneState) -> None:
        """Copy a deploy-side scene snapshot into the mirror.

        After this returns, :meth:`check_success`, :meth:`compute_reward`,
        and :meth:`subtask_signals` reflect the deploy's current scene
        configuration.
        """
        for jname, qpos in state.object_freejoint_qpos.items():
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            qadr = int(self.mj_model.jnt_qposadr[jid])
            if len(qpos) >= 7:
                self.mj_data.qpos[qadr:qadr + 7] = np.asarray(qpos[:7], dtype=np.float64)
        for bname, pos in state.mutable_body_pos.items():
            bid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                continue
            if len(pos) >= 3:
                self.mj_model.body_pos[bid] = np.asarray(pos[:3], dtype=np.float64)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._last_state = state
        self._latest_grasp_contacts = {
            k: dict(v) for k, v in state.grasp_contacts.items()
        }
        self._latest_fingertip_pos = {
            k: [list(p) for p in v] for k, v in state.fingertip_pos.items()
        }

    # ──────────────────────────────────────────────────────────────────
    # Oracle queries
    # ──────────────────────────────────────────────────────────────────

    def check_success(self) -> bool:
        """Return True iff the operator just satisfied the task spec."""
        if self._env_name == "X2PickPlaceCube":
            return bool(self._oracle.success_fn(
                self.mj_model, self.mj_data,
                self._latest_grasp_contacts, self._latest_fingertip_pos,
            ))
        if self._env_name == "X2PickPlaceBowl":
            return bool(self._oracle.success_fn(
                self.mj_model, self.mj_data, self._latest_grasp_contacts
            ))
        return bool(self._oracle.success_fn(self.mj_model, self.mj_data))

    def compute_reward(self) -> float:
        if self._env_name == "X2PickPlaceCube":
            return float(self._oracle.reward_fn(
                self.mj_model, self.mj_data,
                self._latest_grasp_contacts, self._latest_fingertip_pos,
            ))
        if self._env_name == "X2PickPlaceBowl":
            return float(self._oracle.reward_fn(
                self.mj_model, self.mj_data, self._latest_grasp_contacts
            ))
        return float(self._oracle.reward_fn(self.mj_model, self.mj_data))

    def subtask_signals(self) -> dict[str, int]:
        """Per-tick subtask termination signals (e.g. ``grasp_cube``).

        The bridge fills in :attr:`SceneState.grasp_contacts` and
        :attr:`SceneState.fingertip_pos`; the rest are pure-geometry
        queries against the mirror's ``mj_data``.
        """
        if self._env_name == "X2PickPlaceCube":
            return self._oracle.subtasks_fn(
                self.mj_model, self.mj_data,
                self._latest_grasp_contacts, self._latest_fingertip_pos,
            )
        return self._oracle.subtasks_fn(
            self.mj_model, self.mj_data, self._latest_grasp_contacts
        )

    # ──────────────────────────────────────────────────────────────────
    # Convenience helpers
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_metadata_path(
        cls, metadata_path: Path, *, env_name: Optional[str] = None
    ) -> "RobocasaTaskMirror":
        """Construct a mirror given the JSON sidecar produced by
        ``gear_sonic/scripts/build_x2_robocasa_scene_xml.py``.

        The XML path is taken from the metadata (the build script writes
        it as ``scene_xml_path``) so the caller doesn't have to keep both
        paths in sync.
        """
        meta = json.loads(Path(metadata_path).read_text())
        scene_xml = Path(meta.get("scene_xml_path", ""))
        if not scene_xml.is_file():
            # Fallback: assume the XML lives next to the JSON with the
            # same stem (build script's default layout).
            scene_xml = Path(metadata_path).with_suffix(".xml")
        return cls(
            scene_xml_path=scene_xml,
            scene_metadata=meta,
            env_name=env_name,
        )

    def close(self) -> None:
        if self._robosuite_env is not None:
            try:
                self._robosuite_env.close()
            except Exception:
                pass
            self._robosuite_env = None

    def __enter__(self) -> "RobocasaTaskMirror":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
