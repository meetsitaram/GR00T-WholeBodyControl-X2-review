#!/usr/bin/env python3
"""Build a static MJCF combining compose-driven X2 + OmniHand with a robocasa
tabletop env's scene objects (table, cube, bowl, …).

Why this exists
---------------

The C++ deploy bridge (``gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py``)
already accepts ``--mjcf <path>`` to load an arbitrary X2 world XML at startup.
For Phase-1 robocasa data collection we want the bridge to spawn the X2 inside
a tabletop scene with a cube and bowl in front of it so the SONIC body policy
keeps running in the loop while the operator's hands actually interact with
something.

We *cannot* just dump the robocasa env's own MJCF and pass it to the bridge
because:

* The robocasa fork uses ``X2UltraFixedLowerBody`` which welds the legs out
  (16 body joints instead of the 31 the deploy / SONIC ONNX expects).
* Robosuite prefixes every robot joint with ``robot0_`` and every gripper
  joint with ``gripper0_left_L_`` / ``gripper0_right_R_`` -- the deploy looks
  joints up by their canonical names (``waist_yaw_joint``,
  ``left_L_thumb_roll_joint``, …) so the prefixed XML would fail the
  ``_resolve_actuator_indices`` handshake.

The fix is to start from the canonical compose spec
(``gear_sonic.scripts.compose_x2_with_omnihand.build_x2_with_omnihand_spec``)
which produces a 31-DOF X2 + OmniHand model with the exact joint / actuator
names the deploy expects, then graft the robocasa env's *scene* bodies
(table + cube + bowl + their assets) into its worldbody.

The result is a static MJCF on disk that

* the deploy can load via ``--mjcf <path>``;
* the recorder's ``MujocoFrameRenderer`` can also load via the same path so
  the ego-view image matches what the deploy is simulating;
* the ``RobocasaTaskMirror`` can write fresh per-episode object poses into
  by name (cube_joint, bowl_body, …).

Usage
-----

Build a single env::

    .venv_sim/bin/python -u gear_sonic/scripts/build_x2_robocasa_scene_xml.py \\
        --env X2PickPlaceCube \\
        --output gear_sonic/data/assets/robocasa_scenes/X2PickPlaceCube.xml

Build all known envs at once::

    .venv_sim/bin/python -u gear_sonic/scripts/build_x2_robocasa_scene_xml.py --all

Each invocation also writes a ``<scene>.json`` sidecar with metadata the
recorder needs (env name, task instruction string, list of object freejoint
names + the qpos initial values written into the XML).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ── Per-env build descriptors ─────────────────────────────────────────────


@dataclass(frozen=True)
class SceneEnvSpec:
    """Static metadata for a robocasa env we know how to scene-build."""

    env_name: str
    """``suite.make`` env name (must match the ``register_env`` decorator)."""

    robot_class: str = "X2UltraFixedLowerBody"
    """Robosuite robot class -- always the fixed-lower-body X2 for Phase 1."""

    task_string: str = ""
    """Default natural-language instruction stamped into LeRobot metadata."""

    scene_body_names: tuple[str, ...] = ()
    """Worldbody children to graft into the composed spec, in declaration
    order. ``robot0_pelvis`` and the OSC mocap targets (``*_eef_target``)
    are filtered automatically -- only true scene props belong here."""

    object_freejoint_map: dict[str, str] = field(default_factory=dict)
    """``{logical_object_name: freejoint_name}``. Used by the
    :class:`RobocasaTaskMirror` to write fresh per-episode poses into the
    deploy bridge via the ``reset_objects`` ZMQ topic."""

    object_welded_map: dict[str, str] = field(default_factory=dict)
    """``{logical_object_name: body_name}`` for objects welded to the world
    (no freejoint). Mirrored as ``model.body_pos`` writes at episode start."""

    object_contact_geoms: dict[str, tuple[str, ...]] = field(default_factory=dict)
    """``{logical_object_name: (geom_name, geom_name, …)}`` of the contact
    geoms that belong to each scene object. The bridge uses these to
    classify ``mj_data.contact[:ncon]`` entries into per-object grasp
    contacts (forwarded over the ``scene_state`` ZMQ topic). The names
    must match the geoms defined in
    ``decoupled_wbc/dexmg/gr00trobocasa/robocasa/environments/locomanipulation/x2_tabletop_pnp.py``
    (the ``contact_geoms`` attribute on each PrimitiveCube/PrimitiveBowl
    instance)."""

    manipulable_target_body: str = ""
    """Body name of the freely-moving object the operator is supposed to
    manipulate (the cube in pick-place, the bowl in bowl-place). Used as
    the ``targetbody`` for the bake-time ``obj_left`` / ``obj_right``
    inspection cameras so they keep the object in frame even after the
    operator moves it. Empty disables the inspection cameras for envs
    where no single freely-moving target makes sense."""


# Minimal registry. Add new tabletop envs here as we bring them online.
_KNOWN_ENVS: dict[str, SceneEnvSpec] = {
    "X2PickPlaceCube": SceneEnvSpec(
        env_name="X2PickPlaceCube",
        task_string="pick up the red cube and drop it into the blue bowl",
        scene_body_names=("table_body_main", "cube_body", "bowl_body"),
        object_freejoint_map={"cube": "cube_joint"},
        object_welded_map={"bowl": "bowl_body", "table": "table_body_main"},
        manipulable_target_body="cube_body",
        object_contact_geoms={
            "cube": ("cube_collider",),
            # PrimitiveBowl emits five colliders: the floor + four walls.
            # See PrimitiveBowl.contact_geoms in x2_tabletop_pnp.py.
            "bowl": (
                "bowl_floor",
                "bowl_wall_xp",
                "bowl_wall_xn",
                "bowl_wall_yp",
                "bowl_wall_yn",
            ),
        },
    ),
    "X2PickPlaceBowl": SceneEnvSpec(
        env_name="X2PickPlaceBowl",
        task_string="pick up the blue bowl and place it on the green target zone",
        scene_body_names=("table_body_main", "bowl_body", "target_body"),
        object_freejoint_map={"bowl": "bowl_joint"},
        object_welded_map={"target": "target_body", "table": "table_body_main"},
        manipulable_target_body="bowl_body",
        object_contact_geoms={
            "bowl": (
                "bowl_floor",
                "bowl_wall_xp",
                "bowl_wall_xn",
                "bowl_wall_yp",
                "bowl_wall_yn",
            ),
            "target": ("target_collider",),
        },
    ),
}


# OmniHand topology constants (mirrored from compose_x2_with_omnihand for
# the bridge contact-walker + the mirror oracle). Names must match what
# the composer emits into the static scene XML.
_HAND_ROOT_BODIES: dict[str, str] = {
    "left": "left_wrist_roll_link",
    "right": "right_wrist_roll_link",
}
_FINGER_NAMES: tuple[str, ...] = (
    "thumb", "index", "middle", "ring", "pinky",
)


def _fingertip_body_names() -> dict[str, list[str]]:
    """Return ``{side: [<side>_<sdk_prefix><finger>_dip, …]}`` for both hands.

    Distal phalanx (``*_dip``) bodies are the leaf links of each finger
    chain. Their world positions feed the shaped-reward "approach" phase
    in :class:`RobocasaTaskMirror`.
    """
    out: dict[str, list[str]] = {"left": [], "right": []}
    for side, sdk_prefix in (("left", "L_"), ("right", "R_")):
        for finger in _FINGER_NAMES:
            out[side].append(f"{side}_{sdk_prefix}{finger}_dip")
    return out


# ── Scene extraction (from a transient robocasa env) ──────────────────────


def _make_robosuite_env(env_spec: SceneEnvSpec, seed: int):
    """Instantiate the robocasa env briefly so we can scrape its scene XML."""
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")

    import robosuite as suite
    import robocasa  # noqa: F401  -- registers env classes
    import robocasa.models.robots  # noqa: F401  -- registers X2UltraFixedLowerBody
    import robocasa.models.grippers  # noqa: F401  -- registers OmniHand grippers
    from robocasa.models.grippers.omnihand_grippers import (
        load_x2_default_controller_config,
    )

    # Seed BEFORE construction so the env's internal ``self.rng`` (a
    # ``np.random.RandomState`` initialised in ``MujocoEnv.__init__``)
    # picks up our value. Robosuite envs draw all per-reset object
    # placements through ``self.rng``, so seeding the global numpy RNG
    # alone is not enough.
    np.random.seed(seed)
    env = suite.make(
        env_name=env_spec.env_name,
        robots=env_spec.robot_class,
        controller_configs=load_x2_default_controller_config(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=200,
    )
    # Override the env's RNG with a deterministic stream so successive
    # invocations of this build script produce the same canonical scene.
    env.rng = np.random.RandomState(seed)
    env.reset()
    return env


def _gather_assets_referenced_by(
    bodies: list[ET.Element], asset_section: ET.Element
) -> tuple[list[ET.Element], set[str]]:
    """Collect <texture>/<material>/<mesh> elements referenced by *bodies*.

    Walks each body subtree, collects the values of every ``material=`` and
    ``mesh=`` attribute, then transitively collects every ``texture=``
    referenced by one of those materials. Returns ``(asset_elements,
    asset_names)`` where ``asset_elements`` is a deduplicated list of
    ``ET.Element`` references into ``asset_section`` (we keep references so
    re-parenting respects DOM identity).
    """
    referenced_materials: set[str] = set()
    referenced_meshes: set[str] = set()
    for body in bodies:
        for el in body.iter():
            mat = el.get("material")
            if mat is not None:
                referenced_materials.add(mat)
            msh = el.get("mesh")
            if msh is not None:
                referenced_meshes.add(msh)

    # Pass 1: pull materials, capturing the textures they reference.
    referenced_textures: set[str] = set()
    asset_elements: list[ET.Element] = []
    for child in asset_section:
        name = child.get("name", "")
        if child.tag == "material" and name in referenced_materials:
            asset_elements.append(child)
            tex = child.get("texture")
            if tex is not None:
                referenced_textures.add(tex)
        elif child.tag == "mesh" and name in referenced_meshes:
            asset_elements.append(child)

    # Pass 2: pull the textures referenced by the chosen materials.
    for child in asset_section:
        if child.tag == "texture" and child.get("name", "") in referenced_textures:
            asset_elements.append(child)

    asset_names = {e.get("name", "") for e in asset_elements}
    return asset_elements, asset_names


def extract_scene_fragment(
    env, env_spec: SceneEnvSpec
) -> tuple[list[ET.Element], list[ET.Element], dict]:
    """Pull the scene-only bodies + assets + initial-state metadata from *env*.

    Returns ``(scene_bodies, scene_assets, metadata)`` where ``metadata`` is a
    dict with per-object freejoint qpos values (so the static XML can be
    initialised with deterministic poses if the recorder wants them).
    """
    env_xml = env.sim.model.get_xml()
    root = ET.fromstring(env_xml)

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("env XML has no <worldbody>")
    asset_section = root.find("asset")
    if asset_section is None:
        raise RuntimeError("env XML has no <asset>")

    # Pull the named scene bodies, preserving declaration order.
    name_to_body: dict[str, ET.Element] = {}
    for body in worldbody.findall("body"):
        n = body.get("name", "")
        if n:
            name_to_body[n] = body

    scene_bodies: list[ET.Element] = []
    for n in env_spec.scene_body_names:
        if n not in name_to_body:
            raise RuntimeError(
                f"env {env_spec.env_name!r} did not produce expected scene body {n!r}; "
                f"available worldbody children: {sorted(name_to_body)}"
            )
        scene_bodies.append(name_to_body[n])

    # Walk the bodies for asset deps.
    scene_assets, asset_names = _gather_assets_referenced_by(
        scene_bodies, asset_section
    )

    # Snapshot per-object freejoint qpos at reset-time so the static XML
    # ships with a sensible initial state (the recorder will overwrite
    # these via mj_data.qpos before each episode anyway, but having a
    # non-zero initial pose makes ``mujoco.viewer`` previews look right).
    object_initial_qpos: dict[str, list[float]] = {}
    for logical_name, joint_name in env_spec.object_freejoint_map.items():
        try:
            qpos = env.sim.data.get_joint_qpos(joint_name).tolist()
        except (KeyError, ValueError):
            qpos = []
        object_initial_qpos[logical_name] = qpos

    # Snapshot per-welded-object world position so the bridge can re-place
    # them via ``model.body_pos`` writes at episode start.
    object_welded_pos: dict[str, list[float]] = {}
    for logical_name, body_name in env_spec.object_welded_map.items():
        try:
            bid = env.sim.model.body_name2id(body_name)
            object_welded_pos[logical_name] = env.sim.data.body_xpos[bid].tolist()
        except Exception:
            object_welded_pos[logical_name] = []

    metadata = dict(
        env_name=env_spec.env_name,
        task_string=env_spec.task_string,
        scene_body_names=list(env_spec.scene_body_names),
        object_freejoint_map=dict(env_spec.object_freejoint_map),
        object_welded_map=dict(env_spec.object_welded_map),
        object_contact_geoms={
            k: list(v) for k, v in env_spec.object_contact_geoms.items()
        },
        hand_root_bodies=dict(_HAND_ROOT_BODIES),
        fingertip_bodies=_fingertip_body_names(),
        object_initial_qpos=object_initial_qpos,
        object_welded_pos=object_welded_pos,
        scene_asset_names=sorted(asset_names),
    )
    return scene_bodies, scene_assets, metadata


# ── Compose XML production + merge ────────────────────────────────────────


def _build_compose_xml(*, baked_cameras: tuple[str, ...] = ("ego_view",)) -> str:
    """Produce the canonical X2 + OmniHand MJCF as an XML string with the
    head camera(s) attached.

    Going through ``MjSpec.to_xml()`` rather than reading
    ``compose_x2_with_omnihand`` source means we automatically pick up any
    future composer tweaks (mount transforms, finger actuator gains, …)
    without having to mirror them here.

    *baked_cameras* names a set of cameras to attach via
    :func:`render_smoketest_episode_video.add_camera_to_spec` before
    serialisation. The default ``("ego_view",)`` is the front-facing
    RGB-D camera the recorder's ``MujocoFrameRenderer`` looks for. Baking
    them into the static XML means both the deploy bridge AND the
    recorder's renderer load the same MJCF and see identical scene
    geometry, with no ``add_camera_to_spec`` divergence between them.

    We pass ``disable_hand_collisions=False`` here because robocasa
    pick-and-place tasks need the OmniHand fingertips to be solid against
    the cube / bowl / table -- the SONIC-era default of ghosted fingers
    would let everything pass through. See ``X2_INTEGRATION_NOTES.md``
    for the rationale.
    """
    from gear_sonic.scripts.compose_x2_with_omnihand import (
        build_x2_with_omnihand_spec,
    )
    from gear_sonic.scripts.render_smoketest_episode_video import (
        add_camera_to_spec,
        resolve_camera_spec,
    )
    spec, _, _ = build_x2_with_omnihand_spec(disable_hand_collisions=False)
    for cam_name in baked_cameras:
        cam_spec = resolve_camera_spec(cam_name)
        add_camera_to_spec(spec, cam_spec)
    return spec.to_xml()


_MESH_SEARCH_DIRS: tuple[Path, ...] = (
    REPO_ROOT / "gear_sonic" / "data" / "assets" / "robot_description" / "urdf" / "x2_ultra" / "meshes",
    REPO_ROOT / "gear_sonic" / "data" / "assets" / "robot_description" / "omnihand" / "meshes",
)


def _absolutize_meshes(root: ET.Element) -> None:
    """Rewrite every ``<mesh file=...>`` to use an absolute, individually-
    resolved path, then drop the global ``<compiler meshdir>``.

    The composer's ``MjSpec.to_xml()`` collapses two sub-spec meshdirs
    (X2 + OmniHand) into a single ``meshdir`` attribute, but each sub-
    spec's mesh files actually live in different directories on disk.
    A single relative meshdir cannot resolve both sets of basenames at
    once. We side-step the problem by hard-coding the absolute path for
    every individual mesh (looking each basename up in
    ``_MESH_SEARCH_DIRS``) and removing the now-redundant ``meshdir``.
    """
    asset = root.find("asset")
    if asset is not None:
        unresolved: list[str] = []
        for mesh in asset.findall("mesh"):
            file_attr = mesh.get("file")
            if file_attr is None:
                continue
            f = Path(file_attr)
            if f.is_absolute():
                continue
            basename = f.name
            for cand_dir in _MESH_SEARCH_DIRS:
                cand = cand_dir / basename
                if cand.is_file():
                    mesh.set("file", str(cand))
                    break
            else:
                unresolved.append(basename)
        if unresolved:
            raise RuntimeError(
                f"could not resolve mesh basenames {unresolved} against "
                f"{[str(d) for d in _MESH_SEARCH_DIRS]}"
            )

    compiler = root.find("compiler")
    if compiler is not None and compiler.get("meshdir") is not None:
        del compiler.attrib["meshdir"]


def merge_scene_into_compose(
    compose_xml: str,
    scene_bodies: list[ET.Element],
    scene_assets: list[ET.Element],
) -> str:
    """Inject scene bodies + their assets into the composed MJCF, return XML."""
    root = ET.fromstring(compose_xml)
    # The composer's meshdir is relative to gear_sonic/.../mjcf/, AND
    # the OmniHand sub-spec's meshes live in a different directory than
    # the X2 meshes. Resolve each mesh basename individually so the
    # merged XML doesn't depend on a single shared meshdir.
    _absolutize_meshes(root)

    asset_section = root.find("asset")
    if asset_section is None:
        asset_section = ET.SubElement(root, "asset")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("composed MJCF unexpectedly has no <worldbody>")

    # Detect collisions so we fail loudly (vs. silently overwriting an
    # X2 / OmniHand asset by accident).
    existing_asset_names = {
        e.get("name", "")
        for e in asset_section
        if e.get("name") is not None
    }
    for asset in scene_assets:
        nm = asset.get("name", "")
        if nm in existing_asset_names:
            raise RuntimeError(
                f"asset name collision while injecting scene asset {nm!r} -- "
                "compose XML already declares an asset with this name"
            )

    existing_body_names: set[str] = set()
    for body in worldbody.iter("body"):
        bn = body.get("name", "")
        if bn:
            existing_body_names.add(bn)
    for body in scene_bodies:
        nm = body.get("name", "")
        if nm in existing_body_names:
            raise RuntimeError(
                f"body name collision while injecting scene body {nm!r} -- "
                "compose XML already declares a body with this name"
            )

    # Merge.
    for asset in scene_assets:
        asset_section.append(asset)
    for body in scene_bodies:
        worldbody.append(body)

    # Pretty-print would reindent everything; we just emit a single line
    # per element. For human inspection callers can ``xmllint --format``.
    return ET.tostring(root, encoding="unicode")


# ── Workspace inspection cameras ──────────────────────────────────────────


# Bake-time camera placements for the close-up "left" and "right" views
# of the manipulable object. World-frame coordinates assume the X2
# pelvis is at roughly (0, 0, 0.68), the table top hovers around z≈0.7
# in front of the robot at x≈0.5, and the manipulable object spawns
# somewhere over the table top.
#
# Convention: ``+y`` is the robot's left (humanoid right-hand rule, +x
# forward, +z up), so ``obj_left`` sits on the +y side of the cube.
# The ``targetbody`` mode keeps the optical axis pointed at the body
# every frame regardless of where it gets randomized to or where the
# operator drags it -- so these cameras stay useful through grasps,
# lifts, and drops without any per-episode bookkeeping.
_WORKSPACE_CAMERAS: tuple[dict, ...] = (
    {
        "name": "obj_left",
        "pos": (0.50, 0.45, 0.95),
        "fovy": "50",
    },
    {
        "name": "obj_right",
        "pos": (0.50, -0.45, 0.95),
        "fovy": "50",
    },
)


def _inject_workspace_cameras(root: ET.Element, env_spec: SceneEnvSpec) -> int:
    """Inject ``obj_left`` and ``obj_right`` worldbody cameras targeting
    the env's manipulable object body.

    Returns the number of cameras actually appended (0 when the env
    declares no ``manipulable_target_body``). Cameras are added at the
    end of ``<worldbody>`` so the operator can press ``[`` / ``]`` in
    the live MuJoCo viewer to cycle to them after the head camera.
    """
    target_body = env_spec.manipulable_target_body
    if not target_body:
        return 0
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError(
            "scene XML missing <worldbody> after merge -- cannot inject "
            "workspace cameras"
        )
    # Defensive: only inject if the named body actually exists in the
    # merged worldbody, otherwise mujoco compile would fail later with
    # an opaque "target body not found" error.
    body_names = {b.get("name", "") for b in worldbody.iter("body")}
    if target_body not in body_names:
        raise RuntimeError(
            f"manipulable_target_body {target_body!r} not present in merged "
            f"worldbody (have: {sorted(b for b in body_names if b)})"
        )
    n_added = 0
    for cam_spec in _WORKSPACE_CAMERAS:
        cam = ET.SubElement(worldbody, "camera")
        cam.set("name", cam_spec["name"])
        cam.set("pos", " ".join(f"{v:.4f}" for v in cam_spec["pos"]))
        cam.set("fovy", cam_spec["fovy"])
        cam.set("mode", "targetbody")
        cam.set("target", target_body)
        n_added += 1
    return n_added


# ── End-to-end entry point ────────────────────────────────────────────────


def build_scene_xml(
    env_spec: SceneEnvSpec,
    output_path: Path,
    *,
    seed: int = 0,
    metadata_path: Optional[Path] = None,
    verify: bool = True,
) -> dict:
    """Build a single scene XML, write it to disk, and return its metadata."""
    print(f"[build_scene] env={env_spec.env_name} seed={seed}", flush=True)
    print("[build_scene] composing X2 + OmniHand …", flush=True)
    compose_xml = _build_compose_xml()
    print(f"[build_scene] compose XML: {len(compose_xml)} chars", flush=True)

    print("[build_scene] instantiating robocasa env to scrape scene …", flush=True)
    env = _make_robosuite_env(env_spec, seed=seed)
    try:
        scene_bodies, scene_assets, metadata = extract_scene_fragment(
            env, env_spec
        )
        print(
            f"[build_scene] extracted {len(scene_bodies)} scene bodies, "
            f"{len(scene_assets)} assets",
            flush=True,
        )
    finally:
        env.close()

    print("[build_scene] merging scene fragment into composed MJCF …", flush=True)
    final_xml = merge_scene_into_compose(compose_xml, scene_bodies, scene_assets)
    print(f"[build_scene] final XML: {len(final_xml)} chars", flush=True)

    # Bake the close-up workspace cameras (obj_left / obj_right) AFTER
    # the merge so they can declare ``targetbody=cube_body`` (or the
    # bowl equivalent) -- the manipulable target body only exists in
    # the merged worldbody, not in the bare compose spec.
    final_root = ET.fromstring(final_xml)
    n_cams = _inject_workspace_cameras(final_root, env_spec)
    if n_cams > 0:
        print(
            f"[build_scene] injected {n_cams} workspace cameras "
            f"(targetbody={env_spec.manipulable_target_body!r}); "
            "press [ / ] in the MuJoCo viewer to cycle to them.",
            flush=True,
        )
        final_xml = ET.tostring(final_root, encoding="unicode")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(final_xml)
    print(f"[build_scene] wrote {output_path}", flush=True)

    if verify:
        print("[build_scene] verifying with mujoco.MjModel.from_xml_path …", flush=True)
        model = mujoco.MjModel.from_xml_path(str(output_path))
        if model is None:
            raise RuntimeError(
                f"static scene XML failed to compile: {output_path}"
            )
        print(
            f"[build_scene] ✓ model OK: nq={model.nq}, nv={model.nv}, "
            f"nu={model.nu}, njnt={model.njnt}",
            flush=True,
        )

        # Verify the canonical X2 body joints + OmniHand active joints
        # all survived the merge.
        from gear_sonic.scripts.compose_x2_with_omnihand import (
            ACTIVE_FINGER_JOINTS,
            _default_side_configs,
        )
        from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
            X2_BODY_JOINT_NAMES,
        )

        for jname in X2_BODY_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(
                    f"X2 body joint {jname!r} missing from final scene MJCF"
                )
        for side in _default_side_configs():
            for short in ACTIVE_FINGER_JOINTS:
                jname = f"{side.side}_{side.sdk_prefix}{short}_joint"
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    raise RuntimeError(
                        f"OmniHand joint {jname!r} missing from final scene MJCF"
                    )
        # Also check the scene-object freejoints survived (the recorder
        # writes per-episode object poses by name into mj_data.qpos).
        for logical, joint_name in env_spec.object_freejoint_map.items():
            jid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if jid < 0:
                raise RuntimeError(
                    f"scene-object freejoint {joint_name!r} (logical "
                    f"name={logical!r}) missing from final scene MJCF"
                )
        print("[build_scene] ✓ all canonical joints + scene freejoints present", flush=True)

    metadata["scene_xml_path"] = str(output_path)
    if metadata_path is None:
        metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"[build_scene] wrote metadata sidecar: {metadata_path}", flush=True)
    return metadata


# ── CLI ───────────────────────────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--env",
        choices=sorted(_KNOWN_ENVS),
        help="Robocasa env to build a static scene XML for.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Build XMLs for every known env into the default output dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output XML path (defaults to "
             "gear_sonic/data/assets/robocasa_scenes/<env>.xml).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the post-merge MuJoCo compile check.",
    )
    return parser.parse_args(argv)


def _default_output_path(env_name: str) -> Path:
    return (
        REPO_ROOT
        / "gear_sonic"
        / "data"
        / "assets"
        / "robocasa_scenes"
        / f"{env_name}.xml"
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.all:
        if args.output is not None:
            raise SystemExit("--output is incompatible with --all")
        for name, env_spec in sorted(_KNOWN_ENVS.items()):
            build_scene_xml(
                env_spec,
                _default_output_path(name),
                seed=args.seed,
                verify=not args.no_verify,
            )
        return 0

    if args.env is None:
        raise SystemExit("must pass --env <name> or --all")

    env_spec = _KNOWN_ENVS[args.env]
    out_path = args.output or _default_output_path(args.env)
    build_scene_xml(
        env_spec,
        out_path,
        seed=args.seed,
        verify=not args.no_verify,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
