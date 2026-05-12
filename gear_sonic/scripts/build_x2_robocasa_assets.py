"""Generate robosuite-compatible X2 Ultra + OmniHand MJCF assets for gr00trobocasa.

Produces three MJCFs under
``decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/assets/robots/x2_ultra/``:

* ``x2_ultra_robocasa.xml``           -- floating-base body (Phase 2 default).
* ``x2_ultra_fixed_lower_body.xml``   -- pelvis welded to world (Phase 1 default).
* ``omnihand_left.xml`` / ``omnihand_right.xml`` -- robosuite-style gripper
  MJCFs with mount transforms, mimic equality constraints, locked-passive
  equalities, and 10 active position actuators per side.

A single source of truth for OmniHand mount transforms / mimic rules is the
existing :mod:`gear_sonic.scripts.compose_x2_with_omnihand` module -- this
script imports its constants and re-emits them in static MJCFs so the
robosuite arena composer can parse them at env-creation time without needing
to call ``mujoco.MjSpec`` at boot.

Usage::

    .venv_sim/bin/python gear_sonic/scripts/build_x2_robocasa_assets.py [--force]

Re-running is idempotent. Pass ``--force`` to overwrite even if the existing
files would otherwise be identical.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from typing import Iterable

# Resolve repo root (gear_sonic/scripts/ -> repo root).
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from gear_sonic.scripts.compose_x2_with_omnihand import (  # noqa: E402
    ACTIVE_FINGER_JOINTS,
    LOCKED_PASSIVE_JOINTS,
    PASSIVE_MIMIC_RULES,
    _DEFAULT_MOUNT_Z,
    _LEFT_MOUNT_QUAT_WXYZ,
    _RIGHT_MOUNT_QUAT_WXYZ,
)

# Source assets.
_SRC_X2_MJCF = _REPO_ROOT / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
_SRC_X2_MESHES = _REPO_ROOT / "gear_sonic/data/assets/robot_description/urdf/x2_ultra/meshes"
_SRC_OMNIHAND_DIR = _REPO_ROOT / "gear_sonic/data/assets/robot_description/omnihand"
_SRC_OMNIHAND_MESHES = _SRC_OMNIHAND_DIR / "meshes"
_SRC_OMNIHAND_LEFT_URDF = _SRC_OMNIHAND_DIR / "omnihand_left.urdf"
_SRC_OMNIHAND_RIGHT_URDF = _SRC_OMNIHAND_DIR / "omnihand_right.urdf"

# Destination inside the gr00trobocasa fork.
_DST_ROOT = (
    _REPO_ROOT / "decoupled_wbc/dexmg/gr00trobocasa/robocasa/models/assets/robots/x2_ultra"
)
_DST_MESHES = _DST_ROOT / "meshes"
_DST_OMNIHAND_MESHES = _DST_ROOT / "omnihand_meshes"

_DST_BODY_FLOAT = _DST_ROOT / "x2_ultra_robocasa.xml"
_DST_BODY_FIXED = _DST_ROOT / "x2_ultra_fixed_lower_body.xml"
_DST_OMNIHAND_LEFT = _DST_ROOT / "omnihand_left.xml"
_DST_OMNIHAND_RIGHT = _DST_ROOT / "omnihand_right.xml"


# ----------------------------------------------------------------------------
# X2 body MJCF generation
# ----------------------------------------------------------------------------


def _build_x2_body_xml(*, fixed_base: bool) -> str:
    """Strip the X2 MJCF for robosuite arena composition.

    Removes world-level decoration (skybox, groundplane, light, statistic),
    rewrites ``meshdir`` to the local ``meshes/`` next to the output XML,
    optionally welds the pelvis to the world, and adds two ``<site>`` entries
    on the wrist roll links so robosuite can attach OmniHand grippers there.
    """
    tree = ET.parse(_SRC_X2_MJCF)
    root = tree.getroot()

    # 1. Drop ``<compiler meshdir>``: robosuite's ``resolve_asset_dependency``
    # ignores compiler.meshdir and always joins file paths to the XML's
    # folder, so the merged X2 + gripper model resolves consistently only
    # if every mesh ``file=`` is prefixed with ``meshes/`` directly.
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("expected <compiler> tag in X2 MJCF")
    if "meshdir" in compiler.attrib:
        del compiler.attrib["meshdir"]
    # The X2 MJCF has TWO ``<asset>`` blocks (textures + meshes) -- iterate
    # both so every file=path gets the meshes/ prefix.
    for asset in root.findall("asset"):
        for node in asset.findall("./*[@file]"):
            f = node.get("file")
            if f and "/" not in f:
                node.set("file", f"meshes/{f}")

    # 1b. Flatten nested <default> blocks. The original X2 MJCF wraps
    # everything under ``<default class="x2">`` and the pelvis carries
    # ``childclass="x2"`` to cascade it. Robosuite's ``_replace_defaults_inline``
    # only walks one level under ``<default>``, AND robosuite's prefixing
    # logic mangles ``class``/``childclass`` references in surprising ways.
    # So:
    #   * direct children of ``<default class="x2">`` (joint/site etc.)
    #     are promoted to root-level globals (unscoped),
    #   * nested ``<default class="...">`` blocks are hoisted to top level,
    #   * the now-empty ``<default class="x2">`` wrapper is dropped,
    #   * any ``childclass="x2"`` attribute is stripped from bodies.
    root_default = root.find("default")
    if root_default is not None:
        # Helper: recursively pull out nested ``<default>`` blocks (preserving
        # their non-default children in place) and return them flat.
        def _extract_nested_defaults(node: ET.Element) -> list[ET.Element]:
            collected: list[ET.Element] = []
            for child in list(node.findall("default")):
                node.remove(child)
                collected.extend(_extract_nested_defaults(child))
                collected.append(child)
            return collected

        # The wrapper layer in the X2 MJCF is ``<default class="x2">``.
        # Treat it specially: lift its primitive defaults (joint/site/etc.)
        # to root-level globals so X2 joints keep their armature/damping
        # without needing any childclass attribute.
        wrapper = None
        for top in list(root_default.findall("default")):
            if top.get("class") == "x2":
                wrapper = top
                break

        if wrapper is not None:
            root_default.remove(wrapper)
            nested = _extract_nested_defaults(wrapper)
            # All non-default children of the x2 wrapper become root-level globals.
            for prim in list(wrapper):
                wrapper.remove(prim)
                root_default.append(prim)
            for cls in nested:
                root_default.append(cls)
        else:
            # Generic case: still hoist all nested defaults regardless.
            for top in list(root_default.findall("default")):
                nested = _extract_nested_defaults(top)
                for cls in nested:
                    root_default.append(cls)

        # Drop any ``childclass="x2"`` (or any other now-defunct childclass).
        for body in root.iter("body"):
            if body.get("childclass") is not None:
                del body.attrib["childclass"]

        # Strip ``class="x2"`` from any actuator (motor/position) etc. -- the
        # wrapper class no longer exists. The actuator's ctrlrange / forcerange
        # are written explicitly on each motor in the X2 MJCF, so dropping
        # the class is a no-op physically.
        for elem in root.iter():
            if elem.get("class") == "x2":
                del elem.attrib["class"]

    # 2. Drop visual/asset chrome that conflicts with the robocasa arena.
    visual = root.find("visual")
    if visual is not None:
        root.remove(visual)
    statistic = root.find("statistic")
    if statistic is not None:
        root.remove(statistic)

    # 3. Strip skybox/groundplane textures and consolidate ALL ``<asset>``
    # blocks into a single one. The X2 MJCF originally has two assets
    # (textures/materials + meshes) but robosuite's
    # ``resolve_asset_dependency`` only sees the first ``<asset>`` block
    # so we must merge them or it will skip the meshes entirely.
    assets = root.findall("asset")
    if assets:
        primary = assets[0]
        # Merge all subsequent assets into the primary block.
        for extra in assets[1:]:
            for child in list(extra):
                primary.append(child)
            root.remove(extra)
        # Drop textures/materials (handled by the arena instead).
        for child in list(primary):
            if child.tag in ("texture", "material"):
                primary.remove(child)

    # 4. Worldbody surgery: drop floor + tracking light. Pelvis stays as the
    #    only top-level body. Optionally turn its freejoint into a weld.
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("expected <worldbody> tag in X2 MJCF")

    for child in list(worldbody):
        if child.tag == "geom" and child.get("name") == "floor":
            worldbody.remove(child)
        elif child.tag == "light":
            worldbody.remove(child)

    pelvis = worldbody.find("body[@name='pelvis']")
    if pelvis is None:
        raise RuntimeError("expected pelvis body in X2 MJCF worldbody")

    # Remove the per-pelvis tracking light too.
    for light in pelvis.findall("light"):
        pelvis.remove(light)

    if fixed_base:
        # Drop freejoint + the pelvis pos so the arena's robot-mount
        # transform decides the absolute world pose. (Robosuite welds
        # fixed-base robots to the arena via xpos_offset.)
        for fj in pelvis.findall("freejoint"):
            pelvis.remove(fj)
        # Lower the height a touch so the pelvis sits at typical mount height
        # when fixed; the robot class re-publishes a `base_xpos_offset` knob
        # for fine-tuning per arena.
        pelvis.set("pos", "0 0 0")

    # 4b. Auto-name nameless geoms (and sites) so we never trip robosuite's
    # ``_add_default_name_filter`` -- it only knows ``group="0"|"1"`` and
    # crashes on the X2 foot spheres which use ``group="3"``. Naming them
    # deterministically avoids the codepath entirely.
    for body in pelvis.iter("body"):
        body_name = body.get("name", "x2body")
        geom_idx = 0
        for geom in body.findall("geom"):
            if geom.get("name") is None:
                geom.set("name", f"{body_name}_g{geom_idx}")
            geom_idx += 1
        site_idx = 0
        for site in body.findall("site"):
            if site.get("name") is None:
                site.set("name", f"{body_name}_s{site_idx}")
            site_idx += 1

    # 4c. Add ``<site name="<side>_center">`` at the shoulder pitch link --
    # the robosuite composite controller uses this as the arm's base reference
    # site (``self.sim.data.site_xpos[<naming_prefix>{side}_center]``).
    for side in ("left", "right"):
        shoulder = pelvis.find(f".//body[@name='{side}_shoulder_pitch_link']")
        if shoulder is None:
            raise RuntimeError(f"expected {side}_shoulder_pitch_link body")
        center = ET.SubElement(shoulder, "site")
        center.set("name", f"{side}_center")
        center.set("pos", "0 0 0")
        center.set("size", "0.01")
        center.set("rgba", "1 0.3 0.3 1")
        center.set("group", "2")

    # 5. Add eef bodies on wrist roll links (robosuite gripper attach points).
    # Robosuite's ``ManipulatorModel.add_gripper`` calls ``self.merge(gripper,
    # merge_body=eef_name)``, which expects to find a *body* with that name
    # (not a site). The eef body's pos/quat IS the OmniHand mount transform
    # in the wrist-roll local frame -- matching ``_DEFAULT_MOUNT_Z`` and the
    # per-side mount quaternion from gear_sonic.scripts.compose_x2_with_omnihand
    # so the static gripper MJCF root body can sit at the local origin and
    # still mount correctly on the X2 wrist.
    for side in ("left", "right"):
        wrist = pelvis.find(f".//body[@name='{side}_wrist_roll_link']")
        if wrist is None:
            raise RuntimeError(f"expected {side}_wrist_roll_link body")
        eef_body = ET.SubElement(wrist, "body")
        eef_body.set("name", f"{side}_eef")
        eef_body.set("pos", f"0 0 {_DEFAULT_MOUNT_Z}")
        quat = _LEFT_MOUNT_QUAT_WXYZ if side == "left" else _RIGHT_MOUNT_QUAT_WXYZ
        eef_body.set("quat", " ".join(f"{c:.8f}" for c in quat))
        # Add a tiny visualisation site so the mount is visible in the viewer.
        site = ET.SubElement(eef_body, "site")
        site.set("name", f"{side}_eef_site")
        site.set("pos", "0 0 0")
        site.set("size", "0.005")
        site.set("rgba", "0 1 0 0.6")
        site.set("group", "1")

    # 6. Pretty-print + tag with provenance comment.
    ET.indent(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode")
    header = (
        "<!--\n"
        "  AUTO-GENERATED by gear_sonic/scripts/build_x2_robocasa_assets.py\n"
        f"  Source: gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml\n"
        f"  Variant: {'fixed_lower_body (pelvis welded)' if fixed_base else 'floating_base'}\n"
        "  DO NOT EDIT BY HAND, regenerate via the build script.\n"
        "-->\n"
    )
    return header + xml_str + "\n"


# ----------------------------------------------------------------------------
# OmniHand gripper MJCF generation
# ----------------------------------------------------------------------------


def _build_omnihand_xml(*, side: str) -> str:
    """Convert an OmniHand URDF into a robosuite gripper MJCF.

    Uses ``mujoco.MjSpec.from_file`` to parse the URDF (which honors the
    embedded ``<mujoco>`` block's ``meshdir``), then injects:

    * 10 active position actuators (matching ``ACTIVE_FINGER_JOINTS``).
    * 6 passive equality constraints (``PASSIVE_MIMIC_RULES``).
    * 1 locked-passive equality (``LOCKED_PASSIVE_JOINTS``: middle_abad).
    * The robosuite gripper boilerplate ``<body name='eef'>`` + ``ft_frame``
      / ``grip_site`` / ``ee_x``/``ee_y``/``ee_z`` / ``grip_site_cylinder``.

    Mount transform (pos/quat) is left as identity here; the X2 robot MJCF
    has already encoded the per-side mount via the ``<site name='*_eef'>``
    that robosuite uses to attach the gripper. The gripper's root body is
    re-rooted at that site by the robosuite arena composer.
    """
    import mujoco  # local import; keeps the venv check at top-level cleaner

    if side == "left":
        urdf_path = _SRC_OMNIHAND_LEFT_URDF
        sdk_prefix = "L_"
    elif side == "right":
        urdf_path = _SRC_OMNIHAND_RIGHT_URDF
        sdk_prefix = "R_"
    else:
        raise ValueError(side)

    spec = mujoco.MjSpec.from_file(str(urdf_path))
    # Compile first so URDF parse errors surface here -- the original meshdir
    # encoded in the URDF resolves correctly relative to the source file.
    spec.compile()
    raw_xml = spec.to_xml()
    # Patch the meshdir AFTER serialising so the emitted XML points at the
    # robot-local meshes/ dir.  Robosuite merges robot + gripper MJCFs at
    # runtime and only one ``<compiler meshdir>`` survives -- the X2 robot's
    # ``meshes/``.  We pre-symlink the OmniHand STLs into that same directory
    # (see ``_link_omnihand_meshes``) so all mesh references resolve under a
    # single meshdir.

    # Parse, surgically modify, re-serialise.
    root = ET.fromstring(raw_xml)
    # Drop ``<compiler meshdir>``: robosuite ignores compiler.meshdir at
    # asset-resolve time, so we prefix every mesh file with ``meshes/``
    # directly (matching the robocasa G1 convention).
    compiler = root.find("compiler")
    if compiler is not None:
        if "meshdir" in compiler.attrib:
            del compiler.attrib["meshdir"]
        if "texturedir" in compiler.attrib:
            del compiler.attrib["texturedir"]
    asset_root = root.find("asset")
    if asset_root is not None:
        for node in asset_root.findall("./*[@file]"):
            f = node.get("file")
            if f and "/" not in f:
                node.set("file", f"meshes/{f}")

    # The model name from URDF is generic ("omnihand"); make it side-specific.
    root.set("model", f"omnihand_{side}")

    # Drop any visual / statistic blocks that the URDF parser may have
    # synthesized; the arena owns those.
    for tag in ("visual", "statistic"):
        node = root.find(tag)
        if node is not None:
            root.remove(node)

    # Locate the worldbody and the URDF root body (URDF base_link becomes
    # the only top-level body in the converted MJCF).
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("expected <worldbody> in OmniHand MJCF after URDF conversion")

    # The URDF base_link is mass-less; mujoco's URDF parser collapses it
    # and the 5 finger root links surface as siblings in <worldbody>. We
    # re-parent them under a synthetic palm root so we have a single body
    # to attach via the robosuite gripper-mount machinery.
    base_bodies = list(worldbody.findall("body"))
    if not base_bodies:
        raise RuntimeError("OmniHand URDF -> MJCF produced no top-level bodies")

    palm_root = ET.Element("body", attrib={"name": f"{side}_omnihand_palm", "pos": "0 0 0", "quat": "1 0 0 0"})
    # Force-zero-mass site for force/torque sensors (matches robosuite convention).
    ET.SubElement(palm_root, "site", attrib={
        "name": "ft_frame",
        "pos": "0 0 0",
        "size": "0.01 0.01 0.01",
        "rgba": "1 0 0 0",
        "type": "sphere",
        "group": "1",
    })

    # Robosuite expects an `<body name="eef">` with `grip_site` etc. so the
    # arena composer can wire up grasp success checks and end-effector sites.
    eef_body = ET.SubElement(palm_root, "body", attrib={"name": "eef", "pos": "0 0 0", "quat": "1 0 0 0"})
    for site_attribs in (
        {"name": "grip_site", "pos": "0 0 0", "size": "0.01 0.01 0.01", "rgba": "1 1 0 1", "type": "sphere", "group": "2"},
        {"name": "ee_x", "pos": "0.1 0 0", "size": "0.005 .1", "quat": "0.707105 0 0.707108 0", "rgba": "1 0 0 0", "type": "cylinder", "group": "1"},
        {"name": "ee_y", "pos": "0 0.1 0", "size": "0.005 .1", "quat": "0.707105 0.707108 0 0", "rgba": "0 1 0 0", "type": "cylinder", "group": "1"},
        {"name": "ee_z", "pos": "0 0 0.1", "size": "0.005 .1", "quat": "1 0 0 0", "rgba": "0 0 1 0", "type": "cylinder", "group": "1"},
        {"name": "grip_site_cylinder", "pos": "0 0 0", "quat": "-0.5 -0.5 -0.5 0.5", "size": "0.005 0.5", "rgba": "0 1 0 0.3", "type": "cylinder", "group": "1"},
    ):
        ET.SubElement(eef_body, "site", attrib=site_attribs)

    # Move all finger root bodies underneath palm_root.
    for fb in base_bodies:
        worldbody.remove(fb)
        palm_root.append(fb)
    worldbody.append(palm_root)

    # Mark all hand collision geoms as kinematic only (contype=0, conaffinity=0)
    # so they do not interact with the world dynamics. compose_x2_with_omnihand
    # does the same: hands track the active joint targets and never push the
    # world via contact (objects move because the X2 wrist drives them through
    # equality constraints + tendon control, not finger-object contact).
    for body in palm_root.iter("body"):
        for geom in body.findall("geom"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")

    # Find or create the actuator and equality blocks.
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")
    else:
        # Drop any actuators the URDF parser autogenerated; we own them.
        for child in list(actuator):
            actuator.remove(child)

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    else:
        for child in list(equality):
            equality.remove(child)

    # The URDF joint names use the ``<sdk_prefix><name>_joint`` convention.
    def jname(short: str) -> str:
        return f"{sdk_prefix}{short}_joint"

    # 10 active position actuators (matches ACTIVE_FINGER_JOINTS).
    for short in ACTIVE_FINGER_JOINTS:
        joint_name = jname(short)
        ET.SubElement(actuator, "position", attrib={
            "name": f"{joint_name}_drive",
            "joint": joint_name,
            "kp": "5",
            "ctrlrange": "-1.6 1.6",
            "forcelimited": "true",
            "forcerange": "-2.0 2.0",
        })

    # 6 passive (mimic) equalities: passive = active * mult.
    for rule in PASSIVE_MIMIC_RULES:
        ET.SubElement(equality, "joint", attrib={
            "name": f"mimic_{jname(rule.passive)}",
            "joint1": jname(rule.passive),
            "joint2": jname(rule.active),
            "polycoef": f"0 {rule.multiplier} 0 0 0",
        })

    # 1 locked-passive equality per locked joint: passive = 0 * (anything).
    # MuJoCo's joint equality with no joint2 collapses joint1 to its qpos0.
    for short in LOCKED_PASSIVE_JOINTS:
        joint_name = jname(short)
        ET.SubElement(equality, "joint", attrib={
            "name": f"lock_{joint_name}",
            "joint1": joint_name,
            "polycoef": "0 0 0 0 0",
        })

    # Sensors: matches the F/T pair seen in the existing fourier hand MJCF.
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    else:
        for child in list(sensor):
            sensor.remove(child)
    # Sensor names must be ``force_ee`` / ``torque_ee`` -- robosuite prepends
    # the gripper naming prefix (``gripper0_right_`` / ``gripper0_left_``).
    ET.SubElement(sensor, "force", attrib={"name": "force_ee", "site": "ft_frame"})
    ET.SubElement(sensor, "torque", attrib={"name": "torque_ee", "site": "ft_frame"})

    ET.indent(root, space="  ")
    xml_str = ET.tostring(root, encoding="unicode")
    header = (
        "<!--\n"
        "  AUTO-GENERATED by gear_sonic/scripts/build_x2_robocasa_assets.py\n"
        f"  Source: gear_sonic/data/assets/robot_description/omnihand/omnihand_{side}.urdf\n"
        f"  Side: {side}  (sdk_prefix='{sdk_prefix}')\n"
        f"  Active joints (10): {', '.join(ACTIVE_FINGER_JOINTS)}\n"
        f"  Passive mimics (6): {', '.join(r.passive for r in PASSIVE_MIMIC_RULES)}\n"
        f"  Locked passive (1): {', '.join(LOCKED_PASSIVE_JOINTS)}\n"
        "  DO NOT EDIT BY HAND, regenerate via the build script.\n"
        "-->\n"
    )
    return header + xml_str + "\n"


# ----------------------------------------------------------------------------
# Mesh symlinks
# ----------------------------------------------------------------------------


def _symlink_dir(src: Path, dst: Path, *, force: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve() and not force:
            return
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src.resolve())


def _populate_combined_meshes(*, force: bool) -> None:
    """Populate ``x2_ultra/meshes/`` with both X2 and OmniHand STL symlinks.

    Robosuite merges robot + gripper MJCFs at runtime and only honours one
    ``<compiler meshdir>``.  We co-locate every mesh the X2 body and the
    OmniHand grippers reference under a single directory so the merged
    model can resolve them via ``meshdir="meshes"``.
    """
    _DST_MESHES.mkdir(parents=True, exist_ok=True)

    # If a stale top-level symlink exists from a previous build, remove it
    # so we can convert this path into a real directory of per-file links.
    if _DST_MESHES.is_symlink():
        _DST_MESHES.unlink()
        _DST_MESHES.mkdir()

    def _link_each(src_dir: Path) -> None:
        for stl in sorted(src_dir.iterdir()):
            if stl.is_dir():
                continue
            if stl.suffix.upper() not in (".STL", ".OBJ", ".PLY", ".GLB", ".GLTF"):
                continue
            link = _DST_MESHES / stl.name
            if link.exists() or link.is_symlink():
                if not force and link.is_symlink() and link.resolve() == stl.resolve():
                    continue
                link.unlink()
            link.symlink_to(stl.resolve())

    _link_each(_SRC_X2_MESHES)
    _link_each(_SRC_OMNIHAND_MESHES)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def _write_if_changed(path: Path, content: str, *, force: bool) -> bool:
    if path.exists() and not force:
        try:
            existing = path.read_text()
        except Exception:
            existing = None
        if existing == content:
            return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return True


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Always rewrite outputs even if unchanged.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    print(f"[build_x2_robocasa_assets] dst dir: {_DST_ROOT}")
    _DST_ROOT.mkdir(parents=True, exist_ok=True)

    # 1. Body MJCFs.
    body_float = _build_x2_body_xml(fixed_base=False)
    body_fixed = _build_x2_body_xml(fixed_base=True)
    if _write_if_changed(_DST_BODY_FLOAT, body_float, force=args.force):
        print(f"  wrote {_DST_BODY_FLOAT.name}")
    if _write_if_changed(_DST_BODY_FIXED, body_fixed, force=args.force):
        print(f"  wrote {_DST_BODY_FIXED.name}")

    # 2. Hand MJCFs.
    left_xml = _build_omnihand_xml(side="left")
    right_xml = _build_omnihand_xml(side="right")
    if _write_if_changed(_DST_OMNIHAND_LEFT, left_xml, force=args.force):
        print(f"  wrote {_DST_OMNIHAND_LEFT.name}")
    if _write_if_changed(_DST_OMNIHAND_RIGHT, right_xml, force=args.force):
        print(f"  wrote {_DST_OMNIHAND_RIGHT.name}")

    # 3. Combined meshes/ dir (X2 body STLs + OmniHand STLs).
    _populate_combined_meshes(force=args.force)
    print(f"  populated {_DST_MESHES.name}/ with X2 + OmniHand mesh symlinks")

    # 4. Sanity-compile each MJCF in mujoco to catch parse errors immediately.
    import mujoco
    for path in (_DST_BODY_FLOAT, _DST_BODY_FIXED, _DST_OMNIHAND_LEFT, _DST_OMNIHAND_RIGHT):
        try:
            model = mujoco.MjModel.from_xml_path(str(path))
        except Exception as exc:
            print(f"  COMPILE FAIL {path.name}: {exc}")
            return 2
        print(f"  compiled {path.name} OK  (nbody={model.nbody}, njnt={model.njnt}, neq={model.neq}, nu={model.nu})")

    print("[build_x2_robocasa_assets] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
