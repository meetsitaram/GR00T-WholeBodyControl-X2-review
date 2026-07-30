"""
Render an X2 smoke-test episode through MuJoCo's native renderer.

This is the M3/M5 bridge utility. The smoke-test orchestrator
(``record_synthetic_smoketest_dataset.py``) historically filled
``observation.images.ego_view`` with a deterministic gradient frame
because it didn't load MuJoCo. This script replays one of those
recorded episodes through MuJoCo and renders an actual head-mounted
camera view -- the same camera-frame the trained model will see at
deploy time.

M5 (camera plumbing) elevates the per-frame renderer logic into a
reusable service (:class:`MujocoFrameRenderer`) so the smoketest
orchestrator can plug a real MuJoCo render in place of the gradient
on demand (``--camera-source mujoco``). The CLI in this file
remains the canonical "render one recording to MP4" entry-point and
is now a thin loop on top of that service.

Camera model
------------

The X2 head carries four sensors per the URDF
(``gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra.urdf``):

* ``rgbd_head_front`` -- the AimDK ego-view RGB-D camera (the v0 target).
* ``stereo_head_front`` -- forehead stereo pair.
* ``rgb_head_center`` -- center-mount RGB.
* ``rgb_head_rear`` -- rear-facing RGB.

Each camera's mounting frame (``xyz`` + ``rpy``) is taken straight
from the URDF, and the optical axis is derived from the STL panel
geometry: ``rgbd_head_front_link.STL`` is a flat ~83 × 19 × 1 mm
panel whose mesh ``+Z`` is the panel normal -- i.e. the optical
forward direction. The camera's "up" axis is anchored to ``+Z`` of
``head_pitch_link`` (projected perpendicular to the look direction)
so the rendered image is right-side-up regardless of how the URDF
orients the panel.

Inputs
------

The script consumes the per-episode reference recording written by
``record_synthetic_smoketest_dataset.py`` to
``<dataset_dir>__recorded/episode_NNNN_recorded.npz`` -- an .npz with
keys including ``body_trajectory`` ``(T, 31)``. The body trajectory
is in MuJoCo joint order (legs, waist, arms, head) and is written
into ``mj_data.qpos[7:38]`` directly; root pose stays at the world
origin with identity orientation, matching the v0 mock-VLA convention
where the SONIC tracking decoder owns root motion.

Hand finger motion -- ``--with-omnihand`` (M3.5)
-----------------------------------------------

By default the rendered video only shows the 31 body DOFs (legs, waist,
arms, head). Hand finger motion is not in the X2 URDF kinematic chain --
on the real robot it flows out-of-band through the AimDK HAL
(``/aima/hal/joint/hand/command``).

Pass ``--with-omnihand`` to instead load the augmented MJCF assembled by
``gear_sonic/scripts/compose_x2_with_omnihand.py``: it attaches two
articulated 10-active-DOF hand chains (sourced from the upstream
[``AgibotTech/Omnihand-2025-SDK``](https://github.com/AgibotTech/Omnihand-2025-SDK)
URDFs we vendor under ``gear_sonic/data/assets/robot_description/omnihand/``)
to ``left_wrist_roll_link`` and ``right_wrist_roll_link``, recreates the 6
URDF mimic relationships per side as MJCF ``<equality joint>`` constraints,
and disables collisions on every hand geom (renderer is purely kinematic).

When ``--with-omnihand`` is set the renderer expects the recording to also
carry a ``hand_trajectory`` array of shape ``(T, 20)`` -- the canonical
left+right active-joint layout produced by ``record_synthetic_smoketest_
dataset.py``. The 6 passive (mimic) DOFs per side are projected on the fly
via ``apply_active_hand_qpos``.

The training MJCF and the modality config are *not* affected; this is a
renderer-only path.

Usage
-----

Render the first episode of a smoketest dataset to MP4::

    .venv/bin/python gear_sonic/scripts/render_smoketest_episode_video.py \\
        --recording /tmp/x2_smoketest_dryrun__recorded/episode_0000_recorded.npz \\
        --output /tmp/x2_smoketest_dryrun__recorded/episode_0000_ego.mp4 \\
        --camera ego_view --fps 50

Render the rear camera at half resolution (faster, useful for spot checks)::

    .venv/bin/python gear_sonic/scripts/render_smoketest_episode_video.py \\
        --recording /tmp/x2_smoketest_dryrun__recorded/episode_0000_recorded.npz \\
        --output /tmp/x2_smoketest_dryrun__recorded/episode_0000_rear.mp4 \\
        --camera rgb_head_rear --width 320 --height 240
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MJCF_PATH = (
    REPO_ROOT / "gear_sonic" / "data" / "assets"
    / "robot_description" / "mjcf" / "x2_ultra.xml"
)


# Default floating-base pose for the offline dataset rendering path. Live
# callers (e.g. the VLA bridge) pass real ``qpos[0:3]`` / ``qpos[3:7]`` so
# the rendered robot tips and translates the way it actually does in
# MuJoCo. Module-level constants keep ``render_frame`` allocation-free.
_DEFAULT_ROOT_POS_XYZ = np.array([0.0, 0.0, 0.793], dtype=np.float64)
_DEFAULT_ROOT_QUAT_WXYZ = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class HeadCameraSpec:
    """One head-mounted camera, parsed from the URDF and STL.

    The mounting frame fields ``pos`` and ``rpy_xyz`` come from the
    URDF joint origin. ``mesh_optical_axis_in_mesh_frame`` records
    which mesh axis is the optical forward; for the X2 head sensors
    that's the panel normal (mesh ``+Z``). Use :func:`build_camera_quat`
    to convert this spec into a MuJoCo camera quaternion.
    """

    name: str
    parent_link: str
    pos: tuple[float, float, float]
    rpy_xyz: tuple[float, float, float]
    fovy: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExternalCameraSpec:
    """A free-floating worldbody camera. Two flavours:

    * ``target_body`` set + ``xyaxes`` ``None`` -> ``mode="targetbody"``:
      the optical axis tracks the named MJCF body frame-by-frame so
      the framing stays useful no matter where the robot drifts. Used
      by the close-up workspace cameras (``obj_left`` / ``obj_right``).

    * ``target_body`` ``None`` + ``xyaxes`` set -> ``mode="fixed"``:
      both position and orientation are nailed down at scene-load
      time. Used by ``front_cam`` (a wide-angle witness camera that
      stays put even when the robot walks across the scene).

    Unlike :class:`HeadCameraSpec` (rigidly attached to a head link),
    these all live on the MuJoCo ``worldbody`` and never appear on the
    real robot -- they're inspection / dataset auxiliaries only.

    Attributes:
        name: MJCF camera name (must be unique within the model).
        pos: world-frame ``(x, y, z)`` camera position in metres.
            Convention: ``+x`` forward of the robot at episode start,
            ``+z`` up. The pelvis sits near the world origin at
            ``z = 0.793``.
        target_body: when set, the MJCF body the camera should track
            (``mode="targetbody"``). The X2 floating-base root body in
            the compiled MJCF is ``pelvis`` (the URDF ``base_link`` is
            collapsed during MuJoCo's URDF import). Mutually exclusive
            with ``xyaxes``.
        fovy: vertical field-of-view in degrees.
        xyaxes: when set, six floats ``(x_x, x_y, x_z, y_x, y_y, y_z)``
            describing the camera's local +x and +y axes in world
            coordinates. The +z axis (and therefore the look direction,
            which is local -z) is derived as the cross product. Selects
            ``mode="fixed"``. Mutually exclusive with ``target_body``.
        aliases: alternative names ``resolve_camera_spec`` will accept.
    """

    name: str
    pos: tuple[float, float, float]
    target_body: str | None
    fovy: float
    xyaxes: tuple[float, float, float, float, float, float] | None = None
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Either tracking OR fixed -- never both, never neither. Ban
        # ambiguous specs early so misconfigs surface at module import
        # rather than at MJCF compile time inside a Docker container.
        has_target = self.target_body is not None
        has_axes = self.xyaxes is not None
        if has_target == has_axes:
            raise ValueError(
                f"ExternalCameraSpec {self.name!r}: exactly one of "
                f"'target_body' (targetbody mode) or 'xyaxes' (fixed "
                f"mode) must be set; got target_body={self.target_body!r}, "
                f"xyaxes={self.xyaxes!r}."
            )


# Camera mounting frames -- copy-pasted from
# gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra.urdf
# (lines 968-1029). Keeping them in this file rather than re-parsing
# the URDF at import time keeps the renderer dependency-light.
HEAD_CAMERAS: dict[str, HeadCameraSpec] = {
    "rgbd_head_front": HeadCameraSpec(
        name="rgbd_head_front",
        parent_link="head_pitch_link",
        pos=(0.05761, -0.011183, -0.04837),
        rpy_xyz=(2.2689, 0.0, 1.5708),
        # 60° vertical FoV approximates the AimDK RGB-D module
        # (Intel RealSense D435i family that the X2 head ships with).
        fovy=60.0,
        aliases=("ego_view", "rgbd"),
    ),
    "stereo_head_front": HeadCameraSpec(
        name="stereo_head_front",
        parent_link="head_pitch_link",
        pos=(0.067995, 0.029784, 0.05),
        rpy_xyz=(-1.5708, 0.0, -1.574),
        fovy=70.0,
        aliases=("stereo",),
    ),
    "rgb_head_center": HeadCameraSpec(
        name="rgb_head_center",
        parent_link="head_pitch_link",
        pos=(0.0684, -0.00021713, 0.05),
        rpy_xyz=(-1.5708, 0.0, -1.574),
        fovy=60.0,
        aliases=("rgb_center",),
    ),
    "rgb_head_rear": HeadCameraSpec(
        name="rgb_head_rear",
        parent_link="head_pitch_link",
        pos=(-0.0834, 0.00026495, 0.0),
        rpy_xyz=(-1.5708, 0.0, 1.5676),
        fovy=60.0,
        aliases=("rear",),
    ),
}


# Free-floating world-frame cameras for "spectator" / debug renderings.
# These do not exist on the real X2 robot -- they're inspection tools
# only. They use MuJoCo's ``targetbody`` projection so the optical axis
# always points at the named body, no matter where the robot drifts.
EXTERNAL_CAMERAS: dict[str, ExternalCameraSpec] = {
    "third_person_front": ExternalCameraSpec(
        name="third_person_front",
        pos=(1.6, -1.2, 1.4),
        target_body="pelvis",
        fovy=45.0,
        aliases=("third_person", "spectator"),
    ),
    "third_person_side": ExternalCameraSpec(
        name="third_person_side",
        pos=(0.0, -2.0, 1.2),
        target_body="pelvis",
        fovy=45.0,
        aliases=("side_view",),
    ),
    "third_person_above": ExternalCameraSpec(
        name="third_person_above",
        pos=(0.6, -0.8, 2.0),
        target_body="pelvis",
        fovy=55.0,
        aliases=("overhead", "top_down"),
    ),
    # Wide-angle, world-fixed witness camera baked into the robocasa
    # scene XMLs (see ``_WORKSPACE_CAMERAS`` in
    # ``gear_sonic/scripts/build_x2_robocasa_scene_xml.py``). Sits 3
    # ft (~0.91 m) in front of the robot's launch position at chest
    # height, looking back along world -x with world +z as up. 120°
    # vertical FoV is wide enough to keep the entire X2 + the table
    # in frame even when the robot leans forward to grasp something.
    # Recorded into the LeRobot dataset as
    # ``observation.images.front_cam`` whenever the recorder is
    # invoked with ``--front-cam`` (default in robocasa scene mode).
    "front_cam": ExternalCameraSpec(
        name="front_cam",
        pos=(0.9144, 0.0, 1.10),
        target_body=None,
        xyaxes=(0.0, 1.0, 0.0,  0.0, 0.0, 1.0),
        fovy=120.0,
        aliases=("front", "witness"),
    ),
}


CameraSpec = HeadCameraSpec | ExternalCameraSpec


def resolve_camera_spec(name_or_alias: str) -> CameraSpec:
    """Look up a camera by canonical name or alias.

    Searches both :data:`HEAD_CAMERAS` (rigidly mounted to head links,
    visible to the policy) and :data:`EXTERNAL_CAMERAS` (free-floating
    spectator views, inspection only). Returns the matching dataclass.
    """
    if name_or_alias in HEAD_CAMERAS:
        return HEAD_CAMERAS[name_or_alias]
    if name_or_alias in EXTERNAL_CAMERAS:
        return EXTERNAL_CAMERAS[name_or_alias]
    for spec in HEAD_CAMERAS.values():
        if name_or_alias in spec.aliases:
            return spec
    for spec in EXTERNAL_CAMERAS.values():
        if name_or_alias in spec.aliases:
            return spec
    available = ", ".join(
        sorted(
            {n for spec in HEAD_CAMERAS.values()
             for n in (spec.name, *spec.aliases)}
            | {n for spec in EXTERNAL_CAMERAS.values()
               for n in (spec.name, *spec.aliases)}
        )
    )
    raise ValueError(
        f"Unknown camera name {name_or_alias!r}. Available: {available}"
    )


def build_camera_quat(rpy_xyz: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """Compute a MuJoCo camera ``quat (wxyz)`` from a URDF mesh ``rpy``.

    The MuJoCo camera convention is ``-Z`` forward, ``+Y`` up. The X2
    URDF mounts every head sensor with a mesh whose ``+Z`` is the
    panel normal (the optical "forward" direction), so:

    1. Apply the URDF ``rpy`` (extrinsic XYZ) to get the panel normal
       in the parent link frame -- that is the desired ``-Z_cam``.
    2. Anchor ``+Y_cam`` to ``+Z`` of the parent link projected
       perpendicular to the look direction. This keeps the rendered
       image right-side-up regardless of the panel's roll.
    3. Build the rotation matrix ``[right | up | -look]`` and read off
       the quaternion.

    Returns: ``(w, x, y, z)`` -- MuJoCo's quaternion convention.
    """
    from scipy.spatial.transform import Rotation as R

    R_mesh = R.from_euler("xyz", list(rpy_xyz))
    look = R_mesh.apply([0.0, 0.0, 1.0])
    look /= np.linalg.norm(look)

    world_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(world_up, look)) > 0.999:
        # Degenerate: panel normal is nearly vertical. Fall back to
        # head_pitch_link +X (forward) as the "up reference" so the
        # cross product is well-defined.
        world_up = np.array([1.0, 0.0, 0.0])
    up = world_up - np.dot(look, world_up) * look
    up /= np.linalg.norm(up)

    right = np.cross(up, -look)
    right /= np.linalg.norm(right)

    M = np.column_stack([right, up, -look])
    q_xyzw = R.from_matrix(M).as_quat()
    return float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])


def add_camera_to_spec(spec, mjcf_camera: CameraSpec) -> None:
    """Programmatically add a named camera to a loaded ``MjSpec``.

    Supports both :class:`HeadCameraSpec` (rigidly attached to a head
    link with an explicit URDF rpy orientation) and
    :class:`ExternalCameraSpec` (free-floating world camera that uses
    MuJoCo's ``targetbody`` projection to always point at a chosen
    body). The latter is useful for spectator / inspection videos.
    """
    import mujoco

    if isinstance(mjcf_camera, HeadCameraSpec):
        parent = spec.body(mjcf_camera.parent_link)
        if parent is None:
            raise RuntimeError(
                f"parent body {mjcf_camera.parent_link!r} not found in MJCF"
            )
        cam = parent.add_camera()
        cam.name = mjcf_camera.name
        cam.pos = list(mjcf_camera.pos)
        cam.quat = list(build_camera_quat(mjcf_camera.rpy_xyz))
        cam.fovy = float(mjcf_camera.fovy)
        return

    if isinstance(mjcf_camera, ExternalCameraSpec):
        worldbody = spec.worldbody
        if worldbody is None:
            raise RuntimeError("MjSpec has no worldbody (cannot anchor external camera)")
        cam = worldbody.add_camera()
        cam.name = mjcf_camera.name
        cam.pos = list(mjcf_camera.pos)
        cam.fovy = float(mjcf_camera.fovy)
        # ``mode=targetbody`` keeps the optical axis locked onto the
        # named body's origin every time MuJoCo renders. This is what
        # lets the spectator camera follow the robot through episode
        # motion without baking a per-frame quaternion.
        cam.mode = mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODY
        cam.targetbody = mjcf_camera.target_body
        return

    raise TypeError(
        f"add_camera_to_spec: unsupported camera type {type(mjcf_camera).__name__}"
    )


def build_model_with_camera(
    camera: HeadCameraSpec,
    *,
    with_omnihand: bool = False,
    offwidth: int | None = None,
    offheight: int | None = None,
):
    """Load the X2 MJCF, attach the head camera, and optionally augment with OmniHand.

    Returns ``(model, layout, body_qposadr)``:

    * ``model``: the compiled :class:`mujoco.MjModel`.
    * ``layout``: a ``compose_x2_with_omnihand.HandQposLayout`` when
      ``with_omnihand`` is True, otherwise ``None``.
    * ``body_qposadr``: a length-31 ``np.ndarray[int64]`` mapping each
      slot of the canonical body trajectory (``X2_BODY_JOINT_NAMES``) to
      its ``qposadr`` in the compiled model.

    Why the per-name address table?  ``MjSpec.attach()`` inserts the
    OmniHand finger hinges immediately after the parent ``*_wrist_roll``
    joint -- which means in the augmented model the right-arm joints are
    pushed past the left-hand finger qpos slots.  Callers that assume
    ``qpos[7:38]`` is the 31 contiguous body slots silently corrupt the
    right arm into hand qpos slots and freeze the right arm in place.
    Returning a name-resolved address table forces the renderer to use
    per-joint addresses, regardless of how MuJoCo laid out the
    augmented model.
    """
    import mujoco

    from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
        X2_BODY_JOINT_NAMES,
    )

    if with_omnihand:
        from gear_sonic.scripts.compose_x2_with_omnihand import (
            build_x2_with_omnihand_spec,
        )

        spec, _, layout = build_x2_with_omnihand_spec()
        add_camera_to_spec(spec, camera)
        # MuJoCo's offscreen framebuffer defaults to 640x480; rendering at
        # higher resolutions silently fails on EGL with
        # ``ValueError: Image width N exceeds offwidth 640``. Resize the
        # spec's <visual><global offwidth=... offheight=.../> *before*
        # compile so the framebuffer matches the requested camera output.
        if offwidth is not None:
            spec.visual.global_.offwidth = int(offwidth)
        if offheight is not None:
            spec.visual.global_.offheight = int(offheight)
        model = spec.compile()
        if model is None:
            raise RuntimeError(
                "augmented (X2 + OmniHand) MJCF failed to compile after camera attach"
            )
    else:
        spec = mujoco.MjSpec.from_file(str(MJCF_PATH))
        add_camera_to_spec(spec, camera)
        if offwidth is not None:
            spec.visual.global_.offwidth = int(offwidth)
        if offheight is not None:
            spec.visual.global_.offheight = int(offheight)
        model = spec.compile()
        layout = None

    body_qposadr = np.empty(len(X2_BODY_JOINT_NAMES), dtype=np.int64)
    for i, name in enumerate(X2_BODY_JOINT_NAMES):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise RuntimeError(
                f"X2 body joint {name!r} missing from compiled model "
                f"(with_omnihand={with_omnihand})"
            )
        body_qposadr[i] = int(model.jnt_qposadr[jid])
    return model, layout, body_qposadr


class MujocoFrameRenderer:
    """Per-frame MuJoCo render service for the X2 + OmniHand spec.

    Build once, render N frames, close. This encapsulates the
    model+renderer+EGL setup so it can be reused as both:

    * the inner loop of :func:`render_episode` (single-recording -> MP4),
    * the camera-plumbing path of
      :func:`gear_sonic.scripts.record_synthetic_smoketest_dataset.build_smoketest_dataset`
      when ``--camera-source mujoco`` is requested. There the renderer
      is built once per dataset and called per frame so the LeRobot
      ``observation.images.ego_view`` tensor carries native MuJoCo
      pixels instead of the M3 gradient placeholder.

    The renderer is purely kinematic: the floating base sits at the
    nominal stand pose ``(0, 0, 0.793)`` with identity orientation,
    body joints are written by *named* qposadr (the OmniHand augmented
    model fragments the contiguous body block), and finger mimic DOFs
    are projected on the fly via
    :func:`gear_sonic.scripts.compose_x2_with_omnihand.apply_active_hand_qpos`.
    """

    def __init__(
        self,
        *,
        camera: str | CameraSpec = "ego_view",
        width: int = 640,
        height: int = 480,
        with_omnihand: bool = True,
        egl: bool = True,
        scene_xml_path: str | os.PathLike | None = None,
    ) -> None:
        """Construct the renderer.

        ``scene_xml_path`` (Phase-1 robocasa addition): when set, load
        the MJCF from disk via :func:`mujoco.MjModel.from_xml_path`
        instead of programmatically composing X2 + OmniHand. The static
        XML must already include the camera named *camera* (the build
        script in ``gear_sonic/scripts/build_x2_robocasa_scene_xml.py``
        bakes ``ego_view`` in by default). This path lets the recorder
        render against the same scene the deploy bridge sees -- including
        table + cube + bowl -- without rebuilding the spec on the
        recorder side.
        """
        if egl:
            os.environ.setdefault("MUJOCO_GL", "egl")

        import mujoco

        self._mujoco = mujoco
        self._cam_spec: CameraSpec = (
            camera if isinstance(camera, (HeadCameraSpec, ExternalCameraSpec))
            else resolve_camera_spec(str(camera))
        )
        self._with_omnihand = bool(with_omnihand)

        if scene_xml_path is not None:
            # Static-MJCF path. Resolve body/hand qposadrs by joint name
            # so the rest of the renderer (which addresses qpos by
            # logical name) still works regardless of how the scene XML
            # numbered things.
            self._model = mujoco.MjModel.from_xml_path(str(scene_xml_path))
            # Bump the offscreen framebuffer to match the requested
            # output size (the build script may have left it at MuJoCo's
            # 640x480 default).
            try:
                if int(width) > self._model.vis.global_.offwidth:
                    self._model.vis.global_.offwidth = int(width)
                if int(height) > self._model.vis.global_.offheight:
                    self._model.vis.global_.offheight = int(height)
            except Exception:
                pass
            from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
                X2_BODY_JOINT_NAMES,
            )
            self._body_qposadr = np.empty(len(X2_BODY_JOINT_NAMES), dtype=np.int64)
            for i, jname in enumerate(X2_BODY_JOINT_NAMES):
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                if jid < 0:
                    raise RuntimeError(
                        f"X2 body joint {jname!r} missing from scene MJCF "
                        f"{scene_xml_path}"
                    )
                self._body_qposadr[i] = int(self._model.jnt_qposadr[jid])
            self._hand_layout = None
            if self._with_omnihand:
                from gear_sonic.scripts.compose_x2_with_omnihand import (
                    _build_layout, _default_side_configs,
                )
                self._hand_layout = _build_layout(
                    self._model, _default_side_configs()
                )
        else:
            self._model, self._hand_layout, self._body_qposadr = (
                build_model_with_camera(
                    self._cam_spec,
                    with_omnihand=self._with_omnihand,
                    offwidth=int(width),
                    offheight=int(height),
                )
            )
        self._data = mujoco.MjData(self._model)
        self._cam_id = mujoco.mj_name2id(
            self._model, mujoco.mjtObj.mjOBJ_CAMERA, self._cam_spec.name
        )
        if self._cam_id < 0:
            raise RuntimeError(
                f"camera {self._cam_spec.name!r} did not survive MJCF compile"
            )

        self._width = int(width)
        self._height = int(height)
        self._renderer = mujoco.Renderer(self._model, height=self._height, width=self._width)

        self._apply_hand = None
        if self._with_omnihand:
            from gear_sonic.scripts.compose_x2_with_omnihand import (
                apply_active_hand_qpos as _apply,
            )
            self._apply_hand = _apply

    # ------------------------------------------------------------------
    # Read-only surface (acceptance gates exercise these)
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def with_omnihand(self) -> bool:
        return self._with_omnihand

    @property
    def camera_spec(self) -> CameraSpec:
        return self._cam_spec

    @property
    def body_qposadr(self) -> np.ndarray:
        """Length-31 ``int64`` array mapping each canonical body slot to a qposadr."""
        return self._body_qposadr

    # ------------------------------------------------------------------
    # Per-frame render
    # ------------------------------------------------------------------

    def render_frame(
        self,
        body_q: np.ndarray,
        *,
        left_active: np.ndarray | None = None,
        right_active: np.ndarray | None = None,
        root_pos_xyz: np.ndarray | None = None,
        root_quat_wxyz: np.ndarray | None = None,
    ) -> np.ndarray:
        """Render one frame and return a ``(H, W, 3)`` uint8 RGB array.

        Args:
            body_q: ``(31,)`` canonical X2 body joint vector
                (legs/waist/arms/head, MuJoCo joint order).
            left_active: ``(10,)`` active OmniHand joint vector for the
                left side. Required when ``with_omnihand=True``; ignored
                otherwise. Shape mismatches raise ``ValueError`` (via
                ``apply_active_hand_qpos``).
            right_active: ``(10,)`` active OmniHand joint vector for the
                right side. Same semantics as ``left_active``.
            root_pos_xyz: optional ``(3,)`` world-frame pelvis position.
                Defaults to ``(0, 0, 0.793)`` (the X2 nominal stand
                pose), which is what the offline dataset rendering code
                path uses. Pass a live MuJoCo ``qpos[0:3]`` here to
                visualize translation (e.g. the live VLA bridge).
            root_quat_wxyz: optional ``(4,)`` world-frame pelvis
                orientation in MuJoCo's ``wxyz`` order. Defaults to
                identity, matching the offline path. Pass a live
                ``qpos[3:7]`` (or the deploy's ``base_quat`` from the
                ``x2_debug`` ZMQ stream) to visualize tilt / fall.

        Returns:
            ``np.ndarray`` of shape ``(self.height, self.width, 3)`` and
            dtype ``uint8`` — ready to feed straight into the LeRobot
            exporter as ``observation.images.ego_view``.
        """
        body_q = np.asarray(body_q, dtype=np.float64)
        if body_q.shape != (self._body_qposadr.shape[0],):
            raise ValueError(
                f"body_q must have shape ({self._body_qposadr.shape[0]},); got {body_q.shape}"
            )

        if root_pos_xyz is None:
            root_pos_xyz = _DEFAULT_ROOT_POS_XYZ
        else:
            root_pos_xyz = np.asarray(root_pos_xyz, dtype=np.float64).reshape(-1)
            if root_pos_xyz.shape != (3,):
                raise ValueError(
                    f"root_pos_xyz must have shape (3,); got {root_pos_xyz.shape}"
                )
        if root_quat_wxyz is None:
            root_quat_wxyz = _DEFAULT_ROOT_QUAT_WXYZ
        else:
            root_quat_wxyz = np.asarray(root_quat_wxyz, dtype=np.float64).reshape(-1)
            if root_quat_wxyz.shape != (4,):
                raise ValueError(
                    f"root_quat_wxyz must have shape (4,); got {root_quat_wxyz.shape}"
                )
            n = float(np.linalg.norm(root_quat_wxyz))
            if n < 1e-9:
                root_quat_wxyz = _DEFAULT_ROOT_QUAT_WXYZ
            else:
                root_quat_wxyz = root_quat_wxyz / n

        d = self._data
        d.qpos[0] = float(root_pos_xyz[0])
        d.qpos[1] = float(root_pos_xyz[1])
        d.qpos[2] = float(root_pos_xyz[2])
        d.qpos[3] = float(root_quat_wxyz[0])
        d.qpos[4] = float(root_quat_wxyz[1])
        d.qpos[5] = float(root_quat_wxyz[2])
        d.qpos[6] = float(root_quat_wxyz[3])
        d.qpos[self._body_qposadr] = body_q

        if (
            self._with_omnihand
            and self._apply_hand is not None
            and self._hand_layout is not None
            and (left_active is not None or right_active is not None)
        ):
            self._apply_hand(
                d,
                self._hand_layout,
                left_active=left_active,
                right_active=right_active,
            )

        d.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, d)

        self._renderer.update_scene(d, camera=self._cam_id)
        return self._renderer.render()

    # ------------------------------------------------------------------
    # Resource management
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the EGL render context. Idempotent."""
        renderer = getattr(self, "_renderer", None)
        if renderer is not None:
            try:
                renderer.close()
            except AttributeError:
                pass
            del self._renderer
            self._renderer = None  # type: ignore[assignment]

    def __enter__(self) -> "MujocoFrameRenderer":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def _load_recording(path: Path) -> dict[str, np.ndarray]:
    """Load a per-episode reference recording from
    ``record_synthetic_smoketest_dataset.py``."""
    if not path.is_file():
        raise FileNotFoundError(
            f"recording not found: {path}. Expected a .npz produced by "
            "gear_sonic/scripts/record_synthetic_smoketest_dataset.py."
        )
    out: dict[str, np.ndarray] = {}
    with np.load(path) as data:
        for k in data.files:
            out[k] = np.asarray(data[k])
    if "body_trajectory" not in out:
        raise KeyError(
            f"recording {path} is missing key 'body_trajectory'; "
            f"got {list(out.keys())}"
        )
    return out


def render_episode(
    recording_path: Path,
    output_path: Path,
    *,
    camera: str = "ego_view",
    width: int = 640,
    height: int = 480,
    fps: float = 50.0,
    max_frames: int | None = None,
    egl: bool = True,
    with_omnihand: bool = False,
) -> dict:
    """Render one smoketest episode to an MP4.

    Args:
        recording_path: ``.npz`` from ``record_synthetic_smoketest_dataset``.
        output_path: where the MP4 will be written. Parent dirs are created.
        camera: name or alias of the head camera (default ``ego_view``).
        width, height: pixel resolution. ``observation.images.ego_view`` in
            ``features_x2_vla`` is 640x480; matching that here keeps the
            rendered video drop-in compatible with the dataset schema.
        fps: video frame rate. Default 50 (matches the X2 control loop).
        max_frames: cap the number of frames rendered (``None`` = no cap).
        egl: if True, set ``MUJOCO_GL=egl`` before importing MuJoCo so
            offscreen rendering works on a headless host. Set False if
            the caller has already configured the GL backend.
        with_omnihand: if True, load the augmented MJCF that adds two
            articulated OmniHand-2025 chains and use ``hand_trajectory``
            from the recording to drive 10 active joints per side.

    Returns:
        A summary dict with rendered frame count, output path, and
        camera metadata.
    """
    import imageio.v2 as imageio

    cam_spec = resolve_camera_spec(camera)
    rec = _load_recording(recording_path)
    body = np.asarray(rec["body_trajectory"], dtype=np.float64)
    if body.ndim != 2 or body.shape[1] != 31:
        raise ValueError(
            f"body_trajectory must have shape (T, 31); got {body.shape}"
        )

    hand: np.ndarray | None = None
    if with_omnihand:
        # ``record_synthetic_smoketest_dataset.py`` writes split per-side
        # arrays (canonical 10-D each). Older / external recordings may
        # carry a single 20-D ``hand_trajectory`` instead. Accept both.
        if "left_hand_trajectory" in rec and "right_hand_trajectory" in rec:
            left = np.asarray(rec["left_hand_trajectory"], dtype=np.float64)
            right = np.asarray(rec["right_hand_trajectory"], dtype=np.float64)
            if left.ndim != 2 or left.shape[1] != 10:
                raise ValueError(f"left_hand_trajectory must be (T, 10); got {left.shape}")
            if right.ndim != 2 or right.shape[1] != 10:
                raise ValueError(f"right_hand_trajectory must be (T, 10); got {right.shape}")
            if left.shape[0] != right.shape[0]:
                raise ValueError(
                    f"left_hand_trajectory T={left.shape[0]} != right_hand_trajectory T={right.shape[0]}"
                )
            hand = np.concatenate([left, right], axis=1)
        elif "hand_trajectory" in rec:
            hand = np.asarray(rec["hand_trajectory"], dtype=np.float64)
            if hand.ndim != 2 or hand.shape[1] != 20:
                raise ValueError(f"hand_trajectory must have shape (T, 20); got {hand.shape}")
        else:
            raise KeyError(
                "--with-omnihand requested but recording is missing hand data. "
                "Expected either left_hand_trajectory + right_hand_trajectory "
                "(canonical), or a single hand_trajectory of shape (T, 20)."
            )

        if hand.shape[0] != body.shape[0]:
            raise ValueError(
                f"body_trajectory T={body.shape[0]} but hand_trajectory T={hand.shape[0]}; "
                "should be equal."
            )

    if max_frames is not None:
        body = body[:max_frames]
        if hand is not None:
            hand = hand[:max_frames]
    T = body.shape[0]

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(output_path),
        fps=float(fps),
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )

    with MujocoFrameRenderer(
        camera=cam_spec,
        width=width,
        height=height,
        with_omnihand=with_omnihand,
        egl=egl,
    ) as renderer:
        try:
            for f in range(T):
                if with_omnihand and hand is not None:
                    frame = renderer.render_frame(
                        body[f],
                        left_active=hand[f, 0:10],
                        right_active=hand[f, 10:20],
                    )
                else:
                    frame = renderer.render_frame(body[f])
                writer.append_data(frame)
        finally:
            writer.close()

    summary: dict = {
        "output_path": str(output_path),
        "num_frames": int(T),
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "camera": cam_spec.name,
        "camera_pos": list(cam_spec.pos),
        "with_omnihand": bool(with_omnihand),
    }
    if isinstance(cam_spec, HeadCameraSpec):
        summary["camera_kind"] = "head"
        summary["camera_parent_link"] = cam_spec.parent_link
        summary["camera_rpy_xyz"] = list(cam_spec.rpy_xyz)
    else:
        summary["camera_kind"] = "external"
        summary["camera_target_body"] = cam_spec.target_body
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recording", type=Path, required=True,
        help=".npz recording from record_synthetic_smoketest_dataset.py",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output MP4 path.",
    )
    parser.add_argument(
        "--camera", default="ego_view",
        help=(
            "Camera name or alias. One of: "
            + ", ".join(sorted({n for spec in HEAD_CAMERAS.values()
                                for n in (spec.name, *spec.aliases)}))
            + ". Default 'ego_view' (=rgbd_head_front)."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Cap rendered frames (default: render the whole recording).",
    )
    parser.add_argument(
        "--no-egl", action="store_true",
        help=(
            "Skip the MUJOCO_GL=egl env var. Use when the host already "
            "has a GL context configured (e.g. interactive desktop)."
        ),
    )
    parser.add_argument(
        "--with-omnihand", action="store_true",
        help=(
            "Augment the X2 MJCF with two articulated OmniHand-2025 chains "
            "and animate fingers from the recording's hand_trajectory[0:10] "
            "(left) and [10:20] (right). Requires the recording to carry a "
            "hand_trajectory (T, 20) array; record_synthetic_smoketest_dataset.py "
            "always writes one. See gear_sonic/scripts/compose_x2_with_omnihand.py "
            "for the augmented MJCF assembly."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = render_episode(
        recording_path=args.recording,
        output_path=args.output,
        camera=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        max_frames=args.max_frames,
        egl=not args.no_egl,
        with_omnihand=args.with_omnihand,
    )
    print(
        f"[render_smoketest_episode_video] wrote {summary['num_frames']} frames "
        f"@ {summary['width']}x{summary['height']} ({summary['fps']:g} fps) "
        f"-> {summary['output_path']}"
    )
    if summary.get("camera_kind") == "head":
        orient = f"rpy_xyz={summary['camera_rpy_xyz']}"
    else:
        orient = f"target_body={summary['camera_target_body']}"
    print(
        f"  camera: {summary['camera']} ({summary.get('camera_kind', '?')})  "
        f"pos={summary['camera_pos']}  {orient}  "
        f"with_omnihand={summary['with_omnihand']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXTERNAL_CAMERAS",
    "ExternalCameraSpec",
    "HEAD_CAMERAS",
    "HeadCameraSpec",
    "MujocoFrameRenderer",
    "add_camera_to_spec",
    "build_camera_quat",
    "build_model_with_camera",
    "render_episode",
    "resolve_camera_spec",
]
