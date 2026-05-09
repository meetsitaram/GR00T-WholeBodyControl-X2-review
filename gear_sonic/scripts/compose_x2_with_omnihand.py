"""Compose the X2 Ultra MJCF with two articulated OmniHand-2025 chains.

This module exists purely to support **rendering and inspection videos**. The
production training MJCF (``gear_sonic/data/assets/robot_description/mjcf/
x2_ultra.xml``) deliberately ends each arm at ``*_wrist_roll_link`` -- 31 DOFs
total -- to match the SONIC tracking decoder, the deploy ONNX, and the AimDK
ROS 2 HAL contract. Hand commands flow out-of-band via ``/aima/hal/joint/hand/
command`` on the real robot.

When we want to *visually* render what those hand commands would have produced
kinematically (for the M3 inspection video, for the M5 camera-plumbing
pipeline that bakes camera frames into the LeRobot dataset, etc.), we compose
the X2 spec with two ``omnihand_{left,right}.urdf`` chains and recreate the
six URDF ``<mimic>`` relationships per side as MJCF ``<equality joint>``
constraints (URDF mimic semantics are not preserved by MuJoCo's URDF parser).

The composition is performed entirely in memory through ``mujoco.MjSpec``;
no static composite XML file is written to disk. This keeps the renderer
asset path stable and avoids divergence between the body MJCF and the
augmented MJCF.

Public entry points
-------------------

``build_x2_with_omnihand_spec``
    Returns a compiled ``MjSpec`` with both hand chains attached.

``HAND_QPOS_LAYOUT``
    Static description of where each side's active / passive hand qpos slots
    end up in the augmented model, plus the prefix that ``MjSpec.attach``
    applies to joint names.

``apply_active_hand_qpos``
    Helper that writes a 10-DOF (per side) active-joint vector into
    ``data.qpos``. The MJCF equality constraints then snap the 6 passive
    DOFs into place on the next ``mj_forward`` call.

The 10 active joints per side, in order, match exactly:

- ``OMNIHAND_FINGER_NAMES_PER_SIDE`` in
  ``gear_sonic/data/robot_model/supplemental_info/x2_ultra/x2_ultra_supplemental_info.py``
- The motor index order in the AgiBot OmniHand-2025 SDK Python API
- The 10-D layout of ``action.left_hand_joints`` / ``action.right_hand_joints``
  in our LeRobot v2.1 datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import mujoco
import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# Static contract (must stay in sync with x2_ultra_supplemental_info.py)
# ────────────────────────────────────────────────────────────────────────────


# 10 active joints per side, in canonical OmniHand-2025 motor index order.
ACTIVE_FINGER_JOINTS: tuple[str, ...] = (
    "thumb_roll",
    "thumb_abad",
    "thumb_mcp",
    "index_abad",
    "index_pip",
    "middle_pip",
    "ring_abad",
    "ring_pip",
    "pinky_abad",
    "pinky_pip",
)


# 6 passive (mimic) relationships per side encoded in the SDK URDF as
#   <mimic joint="<active>" multiplier="m" offset="0"/>.
# We re-create each as an MJCF equality constraint: passive = active * mult.
@dataclass(frozen=True)
class _MimicRule:
    passive: str
    active: str
    multiplier: float


PASSIVE_MIMIC_RULES: tuple[_MimicRule, ...] = (
    _MimicRule(passive="thumb_pip",  active="thumb_mcp", multiplier=1.33),
    _MimicRule(passive="thumb_dip",  active="thumb_mcp", multiplier=1.30),
    _MimicRule(passive="index_dip",  active="index_pip", multiplier=1.097),
    _MimicRule(passive="middle_dip", active="middle_pip", multiplier=1.097),
    _MimicRule(passive="ring_dip",   active="ring_pip",  multiplier=1.097),
    _MimicRule(passive="pinky_dip",  active="pinky_pip", multiplier=1.097),
)


# Default URDF root names (produced by ``MjSpec.from_file`` for the SDK URDFs).
# In the upstream right URDF active joint names look like ``R_thumb_roll_joint``
# (capitalised side prefix). After ``MjSpec.attach(prefix=...)`` they become
# ``<prefix>R_thumb_roll_joint``. We use ``left_/right_`` as our prefix so
# attached joint names land at ``left_R_thumb_roll_joint`` / ``right_L_*``,
# which is unambiguous when we later look them up.
@dataclass(frozen=True)
class _SideConfig:
    side: str  # "left" or "right"
    sdk_prefix: str  # "L_" or "R_" (upstream URDF naming)
    urdf_path: Path
    parent_body: str  # X2 body to attach the SDK base_link onto
    mount_pos: tuple[float, float, float]
    mount_quat_wxyz: tuple[float, float, float, float]


# ────────────────────────────────────────────────────────────────────────────
# Mount transforms
# ────────────────────────────────────────────────────────────────────────────


# Mount transforms tuned against the X2 wrist_roll mesh extents (see
# ``gear_sonic/data/assets/robot_description/urdf/x2_ultra/meshes/`` STL bbox
# probe -- the wrist_roll mesh extends from z=0 down to z≈-0.182m in the body
# local frame, i.e. the wrist tip points along -Z).  The OmniHand palm has its
# origin at the cuff-mounting cylinder with fingers extending along +Z. So the
# palm origin should land at z=-0.182m in the wrist_roll body frame, with the
# palm rotated 180° about the local Y axis so its +Z aligns with the wrist's
# -Z (and the SDK's "thumb-out" Y direction is preserved across hands).
#
# These defaults are visual-only; the renderer accepts overrides via
# ``mount_offset_z`` / extra rotation knobs. Tune by re-rendering the
# ego-view inspection video.

# Mount rotation: per-side composition of two rotations applied in body-
# frame order.
#
# Step 1 (BOTH sides): 180° about wrist_roll local Y. This flips the
# OmniHand palm's +Z (fingertip direction) from wrist +Z to wrist -Z
# (away from the elbow), so fingers extend along the forearm.
#
# Step 2: roll about wrist_roll local Z (the wrist long axis). The X2
# left and right wrist_roll bodies are mirrored about the body
# centerline -- so the SAME world-frame "palm down" orientation
# corresponds to OPPOSITE local-Z rolls on the two sides:
#
# * Right wrist (``_RIGHT_MOUNT_QUAT_WXYZ``): +90° about local Z.
#   With the SDK's right palm placing thumb at palm +Y, this swings
#   the thumb to wrist +X (the right wrist's medial side toward the
#   body centerline) and leaves the back of the hand facing up
#   (palm-down).
#
# * Left wrist (``_LEFT_MOUNT_QUAT_WXYZ``): -90° about local Z.
#   The SDK's left palm places thumb at palm -Y (mirrored from the
#   right). The -90° roll swings the thumb to the LEFT wrist's medial
#   side too, and -- crucially -- inverts the palm-up that the right-
#   wrist roll would otherwise produce on the mirrored body.
#
# Without the sign flip, both hands look anatomically correct
# *individually*, but the left palm faces up while the right palm
# faces down, which breaks the piano-down posture rubric.
#
# To re-derive::
#
#     from scipy.spatial.transform import Rotation as R
#     base = R.from_quat([0, 1, 0, 0])  # 180° about Y, xyzw
#     right = (R.from_euler("z", +90, degrees=True) * base).as_quat()
#     left  = (R.from_euler("z", -90, degrees=True) * base).as_quat()
#     # convert each xyzw -> wxyz
_RIGHT_MOUNT_QUAT_WXYZ: tuple[float, float, float, float] = (
    0.0, 0.7071067811865476, 0.7071067811865476, 0.0
)
_LEFT_MOUNT_QUAT_WXYZ: tuple[float, float, float, float] = (
    0.0, 0.7071067811865476, -0.7071067811865476, 0.0
)
# Backwards-compatibility alias retained for the M3.5 acceptance test
# that pinned a single "default" mount quaternion. Now bound to the
# right-side value because that was the original published convention.
_DEFAULT_MOUNT_QUAT_WXYZ: tuple[float, float, float, float] = _RIGHT_MOUNT_QUAT_WXYZ

# Mount Z for the OmniHand palm in the wrist_roll body local frame, in meters.
#
# This MUST match the cut_z used by ``clip_x2_wrist_for_omnihand.py`` -- the
# OmniHand palm cuff (radius 0.028 m) mates seamlessly with the clipped X2
# wrist neck (radius ~0.029 m at z ≈ -0.055 m). Mounting deeper than the cut
# leaves a gap where the palm cuff is suspended below the trimmed wrist
# tube; mounting shallower clips the palm cuff into the wrist roll motor
# casing.
_DEFAULT_MOUNT_Z = -0.055


def _vendor_root() -> Path:
    return Path(__file__).resolve().parents[2] / "gear_sonic" / "data" / "assets" / "robot_description"


def _default_side_configs(asset_root: Path | None = None) -> tuple[_SideConfig, _SideConfig]:
    if asset_root is None:
        # gear_sonic/scripts/.. -> repo root
        asset_root = Path(__file__).resolve().parents[2] / "gear_sonic" / "data" / "assets" / "robot_description"
    return (
        _SideConfig(
            side="left",
            sdk_prefix="L_",
            urdf_path=asset_root / "omnihand" / "omnihand_left.urdf",
            parent_body="left_wrist_roll_link",
            mount_pos=(0.0, 0.0, _DEFAULT_MOUNT_Z),
            mount_quat_wxyz=_LEFT_MOUNT_QUAT_WXYZ,
        ),
        _SideConfig(
            side="right",
            sdk_prefix="R_",
            urdf_path=asset_root / "omnihand" / "omnihand_right.urdf",
            parent_body="right_wrist_roll_link",
            mount_pos=(0.0, 0.0, _DEFAULT_MOUNT_Z),
            mount_quat_wxyz=_RIGHT_MOUNT_QUAT_WXYZ,
        ),
    )


# ────────────────────────────────────────────────────────────────────────────
# Public layout description
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class HandQposLayout:
    """Maps the 10 active OmniHand DOFs (per side) into augmented ``qpos``.

    ``active_qposadr[side][i]`` is the qpos index of the *i*-th active joint
    (where ``i`` indexes ``ACTIVE_FINGER_JOINTS``). ``passive_qposadr[side]``
    is keyed by passive joint short name (e.g. ``"thumb_pip"``).

    The layout is a function of the augmented model only -- it is computed
    once at compose time and surfaced so callers can write hand commands
    without re-resolving joint names every frame.
    """

    body_qposadr_start: int
    """First qpos index occupied by the X2 body floating base (always 0)."""

    body_njnt: int
    """Number of body joints (free joint contributes 7, hinges contribute 1)."""

    active_qposadr: Mapping[str, list[int]]
    """``side -> [qposadr_per_active_joint_in_canonical_order]``."""

    passive_qposadr: Mapping[str, Mapping[str, int]]
    """``side -> {passive_joint_short_name -> qposadr}``."""

    full_joint_names: Mapping[str, list[str]]
    """``side -> [<prefix>L/R_<short>_joint, ...]`` after ``attach``."""


# ────────────────────────────────────────────────────────────────────────────
# Composition
# ────────────────────────────────────────────────────────────────────────────


def _attach_hand_spec(
    parent: mujoco.MjSpec,
    side: _SideConfig,
) -> None:
    """Attach a single ``omnihand_{side}`` URDF onto the given X2 wrist body."""
    child = mujoco.MjSpec.from_file(str(side.urdf_path))
    # The vendored omnihand URDF declares ``<compiler meshdir="meshes"/>`` AND
    # references each mesh as ``filename="meshes/foo.STL"``. Older mujoco
    # (<= 3.5) silently strips the redundant ``meshes/`` prefix from each
    # filename and resolves to ``<urdf_dir>/meshes/foo.STL``; newer mujoco
    # (>= 3.7) literally joins the two and tries to open
    # ``<urdf_dir>/meshes/meshes/foo.STL``, which doesn't exist.
    #
    # Try the original (3.5-friendly) behaviour first. If it fails with the
    # "meshes/meshes" path, force ``meshdir`` to the URDF dir and retry. This
    # keeps the M5 acceptance gate (which runs against mujoco 3.5 in .venv)
    # green while letting the live VLA bridge (env_isaaclab @ mujoco 3.7) work.
    try:
        child.compile()  # resolves mesh files relative to the URDF dir
    except ValueError as exc:
        if "meshes/meshes/" not in str(exc):
            raise
        child.meshdir = ""
        child.compile()

    # Find the X2 wrist body and add a frame at the desired mount pose.
    parent_body = parent.body(side.parent_body)
    frame = parent_body.add_frame(
        name=f"{side.side}_omnihand_mount",
        pos=list(side.mount_pos),
        quat=list(side.mount_quat_wxyz),
    )

    # ``MjSpec.attach`` mounts the child spec at the given frame and prepends
    # ``prefix`` to every named element. The SDK URDF uses 'L_' / 'R_' prefixes
    # already so we add a 'left_' / 'right_' outer prefix for unambiguous
    # disambiguation with X2's own ``left_*`` / ``right_*`` body names.
    parent.attach(child, prefix=f"{side.side}_", frame=frame)


def _add_mimic_equalities(spec: mujoco.MjSpec) -> None:
    """Recreate URDF ``<mimic>`` rules as MJCF ``<equality joint>`` constraints.

    MuJoCo's ``equality`` of type ``mjEQ_JOINT`` evaluates
    ``q1 - polycoef[0] - polycoef[1]*q2 - polycoef[2]*q2^2 - ...``
    where ``q1`` is ``name1`` and ``q2`` is ``name2``. Setting
    ``polycoef = [0, multiplier, 0, 0, 0]`` enforces ``q1 = multiplier * q2``.
    """
    for side_cfg in _default_side_configs():
        for rule in PASSIVE_MIMIC_RULES:
            j_passive = f"{side_cfg.side}_{side_cfg.sdk_prefix}{rule.passive}_joint"
            j_active = f"{side_cfg.side}_{side_cfg.sdk_prefix}{rule.active}_joint"
            # MjSpec.add_equality requires an 11-element data buffer:
            # for mjEQ_JOINT, the first 5 slots are polycoef[0..4], slots
            # 5-9 are unused, slot 10 is a default scaler (1.0).
            # polycoef = [0, multiplier, 0, 0, 0] enforces
            #     q_passive - 0 - multiplier * q_active = 0  ==>  passive = mult * active
            spec.add_equality(
                name=f"{side_cfg.side}_mimic_{rule.passive}",
                type=mujoco.mjtEq.mjEQ_JOINT,
                name1=j_passive,
                name2=j_active,
                data=[0.0, float(rule.multiplier), 0.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            )


def _swap_wrist_roll_visual_to_clipped(
    spec: mujoco.MjSpec,
    asset_root: Path,
) -> None:
    """Swap each wrist_roll body's *visual* geom to the clipped (no-fist) mesh.

    The X2 ``*_wrist_roll_link.STL`` mesh bakes in a static dummy "fist" stub
    at its tip that overlaps the OmniHand palm. ``clip_x2_wrist_for_omnihand``
    produces a trimmed mesh ending at the wrist neck (z ≈ ``_DEFAULT_MOUNT_Z``);
    this function registers that trimmed mesh as a new asset and points the
    wrist_roll body's visual geom at it. The collision geom keeps the
    original (full) mesh so contact behaviour is unchanged.

    Visual vs collision are distinguished by MJCF group: ``group=1`` is the
    visual class, ``group=3`` is the collision class. The X2 MJCF authors
    each ``*_wrist_roll_link`` body with two geoms following that convention.
    """
    clip_dir = asset_root / "omnihand" / "meshes"
    for side in ("left", "right"):
        clipped_stl = clip_dir / f"{side}_wrist_roll_clipped_link.STL"
        if not clipped_stl.is_file():
            raise FileNotFoundError(
                f"Clipped wrist mesh not found: {clipped_stl}. "
                "Run gear_sonic/scripts/clip_x2_wrist_for_omnihand.py first."
            )

        # Register a new mesh asset with an absolute path so we don't have
        # to fight the X2 MJCF's compiler.meshdir relative-path resolution.
        new_mesh_name = f"{side}_wrist_roll_clipped_link"
        spec.add_mesh(
            name=new_mesh_name,
            file=str(clipped_stl.resolve()),
        )

        body = spec.body(f"{side}_wrist_roll_link")
        if body is None:
            raise RuntimeError(f"X2 body {side}_wrist_roll_link not found in spec")

        swapped = False
        for geom in body.geoms:
            if geom.group == 1 and geom.meshname == f"{side}_wrist_roll_link":
                geom.meshname = new_mesh_name
                swapped = True
        if not swapped:
            raise RuntimeError(
                f"Could not find wrist_roll visual geom on {side}_wrist_roll_link "
                "to swap; X2 MJCF layout may have changed."
            )


def _disable_hand_collisions(spec: mujoco.MjSpec, side_cfgs: tuple[_SideConfig, ...]) -> None:
    """Mark all attached hand geoms as visual-only (contype=0, conaffinity=0).

    The renderer is purely kinematic -- we never want hand fingers to collide
    with the X2 body, the floor, or each other. URDF→MJCF conversion encodes
    the collision boxes/cylinders from the SDK URDF as separate geoms with
    ``contype=1, conaffinity=1`` by default; zero them out so the dynamics
    engine ignores them.

    URDF-derived geoms have empty names -- name-based filtering does NOT
    work. We instead walk the spec's body tree starting at each side's
    attach point (``{side}_wrist_roll_link``) and recurse into every
    OmniHand descendant body, zeroing every geom we find.
    """
    for side_cfg in side_cfgs:
        root = spec.body(side_cfg.parent_body)
        if root is None:
            raise RuntimeError(
                f"_disable_hand_collisions: {side_cfg.parent_body} not found"
            )
        # Walk only descendants newly attached by the OmniHand chain. The
        # wrist_roll_link itself owns the X2 wrist visual + collision geoms;
        # those must keep their original contype/conaffinity so the X2
        # collision model is unchanged.  Traverse children only.
        stack: list = list(root.bodies)
        while stack:
            body = stack.pop()
            for geom in body.geoms:
                geom.contype = 0
                geom.conaffinity = 0
            stack.extend(body.bodies)


def build_x2_with_omnihand_spec(
    *,
    asset_root: Path | None = None,
    mount_offset_z: float | None = None,
) -> tuple[mujoco.MjSpec, mujoco.MjModel, HandQposLayout]:
    """Compose the X2 + OmniHand augmented model.

    Parameters
    ----------
    asset_root:
        Override the ``robot_description`` directory (defaults to the
        repo-vendored path).
    mount_offset_z:
        Override the wrist→palm Z offset in the wrist_roll local frame
        (defaults to ``-0.182m`` -- the bottom of the wrist_roll mesh bbox).

    Returns
    -------
    (spec, model, layout)
        ``spec`` is the live ``MjSpec`` -- handy if the caller wants to
        further mutate it before re-compiling.

        ``model`` is the compiled ``MjModel``, ready to allocate ``MjData``
        against. We return the model from this call rather than re-parsing
        ``spec.to_xml()`` because the latter forces relative mesh paths to be
        resolved against the working directory, which breaks our vendored
        layout.

        ``layout`` describes where every hand qpos lives, so callers can
        write 10-D active vectors without re-resolving joint names.
    """
    if asset_root is None:
        asset_root = _vendor_root()

    x2_mjcf = asset_root / "mjcf" / "x2_ultra.xml"
    if not x2_mjcf.is_file():
        raise FileNotFoundError(f"x2_ultra.xml not found at {x2_mjcf}")

    spec = mujoco.MjSpec.from_file(str(x2_mjcf))

    # Replace the wrist_roll visual mesh with the clipped (no-fist) variant
    # before attaching the OmniHand chains, so the seam between the X2 wrist
    # tube and the OmniHand palm cuff lines up at z = _DEFAULT_MOUNT_Z.
    _swap_wrist_roll_visual_to_clipped(spec, asset_root)

    side_cfgs = list(_default_side_configs(asset_root))
    if mount_offset_z is not None:
        side_cfgs = [
            _SideConfig(
                side=s.side,
                sdk_prefix=s.sdk_prefix,
                urdf_path=s.urdf_path,
                parent_body=s.parent_body,
                mount_pos=(s.mount_pos[0], s.mount_pos[1], float(mount_offset_z)),
                mount_quat_wxyz=s.mount_quat_wxyz,
            )
            for s in side_cfgs
        ]

    for side in side_cfgs:
        _attach_hand_spec(spec, side)

    _add_mimic_equalities(spec)
    # First compile sets up internal indices; second compile picks up the
    # contype/conaffinity tweaks we apply below.
    spec.compile()
    _disable_hand_collisions(spec, tuple(side_cfgs))
    model = spec.compile()
    if model is None:
        raise RuntimeError("MjSpec.compile() returned None after composition")

    layout = _build_layout(model, tuple(side_cfgs))
    return spec, model, layout


def _build_layout(model: mujoco.MjModel, side_cfgs: tuple[_SideConfig, ...]) -> HandQposLayout:
    active_qposadr: dict[str, list[int]] = {}
    passive_qposadr: dict[str, dict[str, int]] = {}
    full_joint_names: dict[str, list[str]] = {}

    for side_cfg in side_cfgs:
        active_idx: list[int] = []
        active_names: list[str] = []
        for short in ACTIVE_FINGER_JOINTS:
            jname = f"{side_cfg.side}_{side_cfg.sdk_prefix}{short}_joint"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(
                    f"Active OmniHand joint '{jname}' not found in augmented MJCF "
                    f"(model.njnt={model.njnt}). Composition likely failed."
                )
            active_idx.append(int(model.jnt_qposadr[jid]))
            active_names.append(jname)
        active_qposadr[side_cfg.side] = active_idx

        passive_map: dict[str, int] = {}
        for rule in PASSIVE_MIMIC_RULES:
            jname = f"{side_cfg.side}_{side_cfg.sdk_prefix}{rule.passive}_joint"
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"Passive joint '{jname}' missing.")
            passive_map[rule.passive] = int(model.jnt_qposadr[jid])
        passive_qposadr[side_cfg.side] = passive_map
        full_joint_names[side_cfg.side] = active_names + [
            f"{side_cfg.side}_{side_cfg.sdk_prefix}{r.passive}_joint" for r in PASSIVE_MIMIC_RULES
        ]

    return HandQposLayout(
        body_qposadr_start=0,
        body_njnt=model.njnt,  # informational only
        active_qposadr=active_qposadr,
        passive_qposadr=passive_qposadr,
        full_joint_names=full_joint_names,
    )


# ────────────────────────────────────────────────────────────────────────────
# qpos write helper
# ────────────────────────────────────────────────────────────────────────────


def apply_active_hand_qpos(
    data: mujoco.MjData,
    layout: HandQposLayout,
    *,
    left_active: np.ndarray | None = None,
    right_active: np.ndarray | None = None,
) -> None:
    """Write 10-D active hand vectors into ``data.qpos`` and project mimic DOFs.

    Both the 10 active joints and the 6 passive (mimic) joints per side are
    written directly using the multipliers from ``PASSIVE_MIMIC_RULES``.  We
    project explicitly because ``mj_forward`` does not snap ``qpos`` onto
    equality constraints -- those drive constraint *forces* during dynamics
    integration, not kinematic projection.  The MJCF equality constraints
    remain in the model as a safety net for callers that step dynamics
    instead of taking the kinematic shortcut.

    Either side can be left ``None`` to leave that hand's qpos untouched.
    """
    if left_active is not None:
        _write_side(data, layout, "left", np.asarray(left_active, dtype=np.float64))
    if right_active is not None:
        _write_side(data, layout, "right", np.asarray(right_active, dtype=np.float64))


def _write_side(
    data: mujoco.MjData,
    layout: HandQposLayout,
    side: str,
    active: np.ndarray,
) -> None:
    if active.shape != (len(ACTIVE_FINGER_JOINTS),):
        raise ValueError(
            f"{side} hand active vector must have shape ({len(ACTIVE_FINGER_JOINTS)},), got {active.shape}"
        )

    # Write active joints in canonical order.
    active_qpos_idxs = layout.active_qposadr[side]
    active_lookup: dict[str, float] = {}
    for k, qadr in enumerate(active_qpos_idxs):
        v = float(active[k])
        data.qpos[qadr] = v
        active_lookup[ACTIVE_FINGER_JOINTS[k]] = v

    # Project mimic relationships: passive = multiplier * active.
    passive_map = layout.passive_qposadr[side]
    for rule in PASSIVE_MIMIC_RULES:
        qadr = passive_map[rule.passive]
        data.qpos[qadr] = float(rule.multiplier) * active_lookup[rule.active]


__all__ = [
    "ACTIVE_FINGER_JOINTS",
    "PASSIVE_MIMIC_RULES",
    "HandQposLayout",
    "build_x2_with_omnihand_spec",
    "apply_active_hand_qpos",
]
