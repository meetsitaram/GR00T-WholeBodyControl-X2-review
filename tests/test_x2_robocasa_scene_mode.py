"""
Hardware-free smoke tests for the X2 + robocasa scene-mode pipeline (G1).

Why these exist
---------------

The full pick-and-place loop spans three processes that each need physical
hardware to exercise end-to-end:

* the C++ deploy + MuJoCo bridge (needs the deploy docker container),
* the recorder (needs a Quest 3 headset on the network),
* and the SONIC tracking policy (needs an exported .onnx checkpoint).

Bringing all three up just to verify a typo in the recorder argparse or a
silly off-by-one in the ``RobocasaTaskMirror`` is overkill, so this gate
exercises every Python-only seam in the G1 architecture from a single
``.venv`` pytest run with **no hardware required**:

1. **Bundled scene MJCF + JSON sidecar** load cleanly into MuJoCo and
   contain every freejoint / welded body listed in the metadata. This
   guards against a stale ``build_x2_robocasa_scene_xml`` rebuild
   producing a scene whose oracle keys don't match the on-disk MJCF.

2. **scene_state / scene_reset wire format** round-trips through the
   ``pack_json``/``unpack_json`` helpers AND the
   ``serialize_*``/``parse_*`` envelope helpers in
   ``gear_sonic.utils.teleop.zmq.scene_state_zmq``. This catches schema
   drift between the bridge PUB side and the recorder SUB side.

3. **RobocasaTaskMirror** can be constructed against the bundled scenes
   (without ever calling its lazy ``_ensure_robosuite_env``), and its
   pure-MuJoCo oracle helpers behave correctly when a synthetic
   ``SceneState`` puts the cube inside / outside the bowl.

4. **Recorder argparse + scene-XML resolution** maps
   ``--robocasa-env <name>`` to the matching MJCF in
   ``gear_sonic/data/assets/robocasa_scenes/`` and surfaces a clear
   error when the asset is missing.

The robosuite-side ``reset()`` path (which lazily compiles a full
robocasa env) is gated behind ``pytest.importorskip`` on the
``gr00trobocasa`` fork-only ``X2PickPlaceCube`` symbol so we don't false-
fail in a venv that only has the upstream ``robocasa==1.0.0``.

Run via::

    .venv/bin/python -m pytest tests/test_x2_robocasa_scene_mode.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENES_DIR = REPO_ROOT / "gear_sonic" / "data" / "assets" / "robocasa_scenes"
BUNDLED_SCENES = ("X2PickPlaceCube", "X2PickPlaceBowl")


def _ensure_repo_on_path() -> None:
    s = str(REPO_ROOT)
    if s not in sys.path:
        sys.path.insert(0, s)


_ensure_repo_on_path()


# ── Layer 1: bundled scene MJCF + JSON sidecar self-consistency ──────────


def _scene_paths(env_name: str) -> tuple[Path, Path]:
    return (SCENES_DIR / f"{env_name}.xml", SCENES_DIR / f"{env_name}.json")


@pytest.mark.parametrize("env_name", BUNDLED_SCENES)
def test_scene_xml_is_loadable_into_mujoco(env_name: str) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml_path, json_path = _scene_paths(env_name)
    assert xml_path.is_file(), f"missing scene XML: {xml_path}"
    assert json_path.is_file(), f"missing metadata sidecar: {json_path}"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    assert model.nq > 0, "scene model has no qpos addresses"
    assert model.nbody > 1, "scene model has only the worldbody"

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # mj_forward should not produce NaN qpos / qvel on a freshly-loaded
    # scene; if it does, the bridge will crash on first publish.
    assert np.all(np.isfinite(data.qpos)), "qpos has NaN after mj_forward"
    assert np.all(np.isfinite(data.qvel)), "qvel has NaN after mj_forward"


@pytest.mark.parametrize("env_name", BUNDLED_SCENES)
def test_scene_metadata_freejoints_and_bodies_resolve_in_xml(
    env_name: str,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    xml_path, json_path = _scene_paths(env_name)
    meta = json.loads(json_path.read_text())

    assert meta.get("env_name") == env_name, (
        f"metadata env_name mismatch: {meta.get('env_name')!r} vs {env_name!r}"
    )
    assert meta.get("task_string"), "metadata is missing a task_string"

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    for logical, jname in meta.get("object_freejoint_map", {}).items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, (
            f"metadata declares freejoint {jname!r} (logical={logical!r}) "
            f"but it does not exist in {xml_path}"
        )
    for logical, bname in meta.get("object_welded_map", {}).items():
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        assert bid >= 0, (
            f"metadata declares welded body {bname!r} (logical="
            f"{logical!r}) but it does not exist in {xml_path}"
        )


@pytest.mark.parametrize("env_name", BUNDLED_SCENES)
def test_scene_xml_bakes_in_ego_view_camera(env_name: str) -> None:
    """The scene must contain the head camera that the recorder's
    ``MujocoFrameRenderer`` points at when rendering
    ``observation.images.ego_view``. The recorder uses ``ego_view`` as a
    user-facing alias for the canonical MJCF camera ``rgbd_head_front``
    (see ``gear_sonic/scripts/render_smoketest_episode_video.py``), so
    only the canonical name needs to live in the static MJCF.
    """
    mujoco = pytest.importorskip("mujoco")
    xml_path, _ = _scene_paths(env_name)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    cam_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "rgbd_head_front"
    )
    assert cam_id >= 0, (
        "scene XML must bake in the 'rgbd_head_front' camera (the "
        "canonical MJCF name behind the 'ego_view' alias) so the "
        "recorder's MujocoFrameRenderer matches the deploy bridge's view"
    )


# ── Layer 2: scene_state / scene_reset ZMQ wire format round-trip ────────


def test_pack_unpack_json_round_trip_preserves_payload() -> None:
    from gear_sonic.utils.teleop.zmq.scene_state_zmq import (
        pack_json, unpack_json,
    )
    payload = {"sim_time": 1.234, "objects": {"cube": [1.0, 2.0, 3.0]}}
    raw = pack_json("scene_state", payload)
    assert raw.startswith(b"scene_state ")
    topic, decoded = unpack_json(raw, expected_topic="scene_state")
    assert topic == "scene_state"
    assert decoded == payload


def test_unpack_json_rejects_topic_mismatch() -> None:
    from gear_sonic.utils.teleop.zmq.scene_state_zmq import (
        pack_json, unpack_json,
    )
    raw = pack_json("scene_state", {"sim_time": 0.0})
    with pytest.raises(ValueError):
        unpack_json(raw, expected_topic="scene_reset")


def test_serialize_parse_scene_state_preserves_object_qpos() -> None:
    pytest.importorskip("mujoco")  # SceneState lives in robocasa_task_mirror
    from gear_sonic.utils.teleop.robocasa_task_mirror import SceneState
    from gear_sonic.utils.teleop.zmq.scene_state_zmq import (
        serialize_scene_state, parse_scene_state,
    )

    state = SceneState(
        sim_time=0.42,
        object_freejoint_qpos={
            "cube_joint": [0.5, -0.1, 0.71, 1.0, 0.0, 0.0, 0.0],
        },
        mutable_body_pos={
            "bowl_body": [0.54, 0.15, 0.68],
        },
        grasp_contacts={"cube": {"left": False, "right": True}},
    )
    raw = serialize_scene_state(state)
    decoded = parse_scene_state(raw)

    assert decoded.sim_time == pytest.approx(0.42)
    assert decoded.object_freejoint_qpos["cube_joint"] == pytest.approx(
        state.object_freejoint_qpos["cube_joint"]
    )
    assert decoded.mutable_body_pos["bowl_body"] == pytest.approx(
        state.mutable_body_pos["bowl_body"]
    )
    assert decoded.grasp_contacts == state.grasp_contacts


def test_serialize_parse_reset_objects_preserves_payload() -> None:
    pytest.importorskip("mujoco")
    from gear_sonic.utils.teleop.robocasa_task_mirror import ResetObjects
    from gear_sonic.utils.teleop.zmq.scene_state_zmq import (
        serialize_reset_objects, parse_reset_objects,
    )

    payload = ResetObjects(
        object_freejoint_qpos={"cube_joint": [0.5, -0.1, 0.7, 1.0, 0.0, 0.0, 0.0]},
        mutable_body_pos={"bowl_body": [0.54, 0.15, 0.68]},
    )
    raw = serialize_reset_objects(payload)
    decoded = parse_reset_objects(raw)
    assert decoded.object_freejoint_qpos["cube_joint"] == pytest.approx(
        payload.object_freejoint_qpos["cube_joint"]
    )
    assert decoded.mutable_body_pos["bowl_body"] == pytest.approx(
        payload.mutable_body_pos["bowl_body"]
    )


# ── Layer 3: RobocasaTaskMirror oracle behaviour (no robosuite needed) ───


@pytest.fixture
def cube_mirror():
    pytest.importorskip("mujoco")
    from gear_sonic.utils.teleop.robocasa_task_mirror import RobocasaTaskMirror

    xml_path, json_path = _scene_paths("X2PickPlaceCube")
    meta = json.loads(json_path.read_text())
    mirror = RobocasaTaskMirror(
        scene_xml_path=xml_path,
        scene_metadata=meta,
        env_name="X2PickPlaceCube",
    )
    yield mirror
    mirror.close()


def test_mirror_construction_does_not_instantiate_robosuite_env(cube_mirror) -> None:
    # The whole point of the lazy-init is that pytest can exercise
    # the oracle layer without paying the ~5 s robosuite import cost
    # (and without needing the gr00trobocasa fork installed).
    assert cube_mirror._robosuite_env is None  # noqa: SLF001 -- intentional


def test_mirror_advertises_static_subtask_signals(cube_mirror) -> None:
    sigs = cube_mirror.subtask_signals()
    # The cube task exposes a full phase ladder: approach/touch/grasp/
    # off-table/above-bowl/in-bowl. All values are 0 on a fresh mirror
    # (no contacts, no fingertips).
    expected_phases = {
        "approach_cube",
        "touch_cube",
        "grasp_cube",
        "cube_off_table",
        "cube_above_bowl",
        "cube_in_bowl",
    }
    assert set(sigs.keys()) == expected_phases
    for name in expected_phases:
        assert sigs[name] == 0, f"phase {name} should be 0 at default pose"
    # Static name list must match the live signature -- this is what
    # the recorder uses to pre-register task.subtask_<name> columns.
    assert set(cube_mirror.static_subtask_names) == expected_phases


def test_mirror_check_success_is_false_at_default_pose(cube_mirror) -> None:
    # Default scene XML places the cube at table-rest height and the
    # bowl off to the side; the success oracle must return False on
    # this initial state, otherwise the recorder would falsely label
    # the very first frame of every episode as a win.
    assert cube_mirror.check_success() is False
    assert cube_mirror.compute_reward() == 0.0


def test_mirror_check_success_flips_when_cube_dropped_into_bowl(
    cube_mirror,
) -> None:
    """End-to-end happy path: synthesise a SceneState that places the
    cube inside the bowl, sync it into the mirror, and assert the
    oracle declares success."""
    pytest.importorskip("mujoco")
    import mujoco

    from gear_sonic.utils.teleop.robocasa_task_mirror import SceneState

    bowl_bid = mujoco.mj_name2id(
        cube_mirror.mj_model, mujoco.mjtObj.mjOBJ_BODY, "bowl_body"
    )
    assert bowl_bid >= 0
    bowl_pos = np.array(cube_mirror.mj_model.body_pos[bowl_bid], dtype=float)
    # Place the cube directly above the bowl floor (within bowl_wall_height)
    # and upright (identity quat). Success criterion: cube xy ≈ bowl xy and
    # cube_z within bowl-height window above bowl_pos[2].
    cube_qpos = [
        float(bowl_pos[0]),
        float(bowl_pos[1]),
        float(bowl_pos[2]) + 0.02,
        1.0, 0.0, 0.0, 0.0,
    ]

    cube_mirror.sync_from_state(SceneState(
        sim_time=1.0,
        object_freejoint_qpos={"cube_joint": cube_qpos},
        mutable_body_pos={"bowl_body": bowl_pos.tolist()},
    ))
    assert cube_mirror.check_success() is True
    assert cube_mirror.compute_reward() == 1.0


def test_mirror_subtask_grasp_cube_reflects_contact_signal(cube_mirror) -> None:
    pytest.importorskip("mujoco")
    from gear_sonic.utils.teleop.robocasa_task_mirror import SceneState

    cube_mirror.sync_from_state(SceneState(
        sim_time=0.0,
        grasp_contacts={"cube": {"left": False, "right": True, "any": True}},
    ))
    sigs = cube_mirror.subtask_signals()
    assert sigs["grasp_cube"] == 1
    assert sigs["touch_cube"] == 1  # right-hand contact also flips touch

    cube_mirror.sync_from_state(SceneState(
        sim_time=0.1,
        grasp_contacts={"cube": {"left": False, "right": False, "any": False}},
    ))
    sigs = cube_mirror.subtask_signals()
    assert sigs["grasp_cube"] == 0
    assert sigs["touch_cube"] == 0


def test_mirror_left_only_contact_fires_touch_but_not_grasp(cube_mirror) -> None:
    """Grasp signal is right-hand specific (mirrors upstream env). Left-only
    contact should still fire ``touch_cube`` so the phase ladder credits the
    operator for getting in contact with either hand."""
    pytest.importorskip("mujoco")
    from gear_sonic.utils.teleop.robocasa_task_mirror import SceneState

    cube_mirror.sync_from_state(SceneState(
        sim_time=0.0,
        grasp_contacts={"cube": {"left": True, "right": False, "any": True}},
    ))
    sigs = cube_mirror.subtask_signals()
    assert sigs["touch_cube"] == 1
    assert sigs["grasp_cube"] == 0


def test_mirror_phased_reward_climbs_through_stages(cube_mirror) -> None:
    """End-to-end check that ``compute_reward`` exposes a partial credit
    ladder rather than a sparse 0/1. We synthesise three states (touch,
    grasp+lift, success) and verify the reward is monotonically
    non-decreasing across them."""
    pytest.importorskip("mujoco")
    import mujoco

    from gear_sonic.utils.teleop.robocasa_task_mirror import SceneState

    bowl_bid = mujoco.mj_name2id(
        cube_mirror.mj_model, mujoco.mjtObj.mjOBJ_BODY, "bowl_body"
    )
    table_bid = mujoco.mj_name2id(
        cube_mirror.mj_model, mujoco.mjtObj.mjOBJ_BODY, "table_body_main"
    )
    bowl_pos = np.array(cube_mirror.mj_model.body_pos[bowl_bid], dtype=float)
    table_top_z = float(cube_mirror.mj_data.xpos[table_bid][2]) + 0.37  # ~table height

    # State A -- cube on table, left-hand touch only (no grasp, no lift).
    cube_at_rest = [0.45, 0.0, table_top_z + 0.025, 1.0, 0.0, 0.0, 0.0]
    cube_mirror.sync_from_state(SceneState(
        sim_time=0.0,
        object_freejoint_qpos={"cube_joint": cube_at_rest},
        grasp_contacts={"cube": {"left": True, "right": False, "any": True}},
    ))
    r_touch = cube_mirror.compute_reward()
    assert 0.0 < r_touch < 0.45, (
        f"touch-only reward should sit on the touch rung (~0.25), got {r_touch}"
    )

    # State B -- cube grasped by right hand AND lifted ~6 cm above the table.
    cube_lifted = [0.45, 0.0, table_top_z + 0.06, 1.0, 0.0, 0.0, 0.0]
    cube_mirror.sync_from_state(SceneState(
        sim_time=0.5,
        object_freejoint_qpos={"cube_joint": cube_lifted},
        grasp_contacts={"cube": {"left": False, "right": True, "any": True}},
    ))
    r_lift = cube_mirror.compute_reward()
    assert r_lift >= r_touch
    assert r_lift >= 0.45  # grasp rung at least

    # State C -- cube settled inside the bowl (success).
    cube_in_bowl = [
        float(bowl_pos[0]),
        float(bowl_pos[1]),
        float(bowl_pos[2]) + 0.02,
        1.0, 0.0, 0.0, 0.0,
    ]
    cube_mirror.sync_from_state(SceneState(
        sim_time=1.0,
        object_freejoint_qpos={"cube_joint": cube_in_bowl},
        mutable_body_pos={"bowl_body": bowl_pos.tolist()},
    ))
    r_success = cube_mirror.compute_reward()
    assert r_success == 1.0
    assert r_success >= r_lift


# ── Layer 4: recorder CLI argparse + scene XML resolution ────────────────


def test_record_x2_dataset_argparse_resolves_robocasa_env_to_xml(
    tmp_path: Path,
) -> None:
    from gear_sonic.scripts.record_x2_dataset import _parse_args

    args = _parse_args([
        "--robocasa-env", "X2PickPlaceCube",
        "--output-dir", str(tmp_path / "ds"),
    ])
    assert args.robocasa_env == "X2PickPlaceCube"
    # Recorder leaves --scene-xml-path unset so main() resolves it; we
    # mirror that resolution here using the same helper attribute.
    expected = args._robocasa_scenes_dir / "X2PickPlaceCube.xml"  # noqa: SLF001
    assert expected.is_file()


def test_record_x2_dataset_argparse_accepts_episode_seed_override() -> None:
    from gear_sonic.scripts.record_x2_dataset import _parse_args

    args = _parse_args([
        "--robocasa-env", "X2PickPlaceBowl",
        "--output-dir", "/tmp/x2_robocasa_smoke",
        "--episode-seed", "1234",
    ])
    assert args.episode_seed == 1234


def test_record_x2_dataset_argparse_rejects_unknown_robocasa_env() -> None:
    from gear_sonic.scripts.record_x2_dataset import _parse_args
    # argparse prints to stderr and exits with SystemExit on bad choices.
    with pytest.raises(SystemExit):
        _parse_args([
            "--robocasa-env", "NotARealEnv",
            "--output-dir", "/tmp/x2_robocasa_smoke",
        ])


# ── Layer 5: gr00trobocasa fork integration (skipped in upstream venv) ────


def _gr00trobocasa_available() -> bool:
    try:
        from robocasa import X2PickPlaceCube  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(
    not _gr00trobocasa_available(),
    reason="gr00trobocasa fork not on this venv -- run in .venv_sim or skip.",
)
@pytest.mark.parametrize("env_name", BUNDLED_SCENES)
def test_mirror_reset_is_deterministic_for_seed(env_name: str) -> None:
    """Same seed -> identical object freejoint qpos. Catches regressions
    in the env's placement_initializer plumbing."""
    pytest.importorskip("mujoco")
    from gear_sonic.utils.teleop.robocasa_task_mirror import RobocasaTaskMirror

    xml_path, json_path = _scene_paths(env_name)
    meta = json.loads(json_path.read_text())

    seed = 4242
    poses_a: dict[str, list[float]] = {}
    poses_b: dict[str, list[float]] = {}
    for sink in (poses_a, poses_b):
        m = RobocasaTaskMirror(
            scene_xml_path=xml_path,
            scene_metadata=meta,
            env_name=env_name,
        )
        try:
            payload = m.reset(seed=seed)
            sink.update({k: list(v) for k, v in payload.object_freejoint_qpos.items()})
        finally:
            m.close()

    assert poses_a.keys() == poses_b.keys() and poses_a, (
        "mirror.reset() did not emit any freejoint qpos entries"
    )
    for jname in poses_a:
        np.testing.assert_allclose(
            poses_a[jname], poses_b[jname], atol=1e-9,
            err_msg=f"freejoint {jname!r} diverged across same-seed resets",
        )


# ── Layer 6: hand contact filter (regression: fingers don't close) ────────


def _walk_hand_geom_collision_attrs(model, hand_root_body_names):
    """Yield ``(contype, conaffinity, group, type, meshname_or_empty)`` for every geom in the hand subtree.

    Walks the body tree ROOTED at each ``*_wrist_roll_link`` (inclusive
    of the root body itself) so the OmniHand palm primitives -- which
    live directly under the wrist body via the ``omnihand_mount`` frame
    -- are visited too. The wrist body also carries X2's own collision
    mesh; the test filters those out by checking ``geom_dataid >= 0``
    AND ``type == mjGEOM_MESH``.
    """
    import mujoco  # local import: pytest.importorskip handles availability

    for root_name in hand_root_body_names:
        root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
        if root_id < 0:
            continue
        # BFS INCLUSIVE of the root body so the OmniHand palm primitives
        # (attached via the omnihand_mount frame, parented directly to
        # the wrist body, NOT under any child body) get visited.
        stack: list[int] = [int(root_id)]
        while stack:
            bid = stack.pop()
            geomadr = int(model.body_geomadr[bid])
            geomnum = int(model.body_geomnum[bid])
            for gid in range(geomadr, geomadr + geomnum):
                meshid = int(model.geom_dataid[gid])
                meshname = ""
                if meshid >= 0:
                    meshname = mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_MESH, meshid
                    ) or ""
                yield (
                    int(model.geom_contype[gid]),
                    int(model.geom_conaffinity[gid]),
                    int(model.geom_group[gid]),
                    int(model.geom_type[gid]),
                    meshname,
                )
            for cbid in range(model.nbody):
                if int(model.body_parentid[cbid]) == bid:
                    stack.append(cbid)


def _is_x2_wrist_collision_mesh(meshname: str) -> bool:
    """X2's own wrist_roll collision mesh -- must remain at default 1/1."""
    return meshname.endswith("_wrist_roll_link")


# Bitmask channels used by the OmniHand self-collision filter. Mirror
# the production constants in
# ``gear_sonic.scripts.compose_x2_with_omnihand``. Hard-coded here so a
# divergence in production code surfaces as a test failure (rather than
# the test silently re-importing the broken value).
_HAND_COLLISION_CHANNEL: int = 2
_WORLD_CHANNEL: int = 1


@pytest.mark.parametrize("env_name", BUNDLED_SCENES)
def test_scene_xml_hand_geoms_use_self_collision_filter(env_name: str) -> None:
    """Bundled scene XMLs must ship with the hand-vs-hand contact filter.

    Regression guard for the bug where ``disable_hand_collisions=False``
    left every URDF-derived collision primitive at MuJoCo's default
    ``contype=1, conaffinity=1`` -- which made the palm cylinder, the
    thumb_mcp / thumb_pip / thumb_dip primitives, and the index /
    middle / ring / pinky pip+dip cylinders all collide with each other
    when the operator commanded a curl. The resulting constraint forces
    routinely exceeded the 3 Nm position-actuator limit and physically
    pinned the fingers half-open; "fully close" never reached the
    commanded angle in robocasa scene mode even though it worked
    perfectly with the bare X2 + OmniHand MJCF.

    The current fix in
    ``compose_x2_with_omnihand._filter_hand_self_collisions`` re-classes
    each hand collision geom to ``(contype=_HAND_COLLISION_CHANNEL=2,
    conaffinity=_WORLD_CHANNEL=1)``. Net effect:

    * fingers / palm collide with cube, bowl, table, floor, X2 body
      (anything at the default (1, 1)); and
    * fingers / palm do NOT collide with each other (palm-vs-finger
      AND finger-vs-finger uniformly filtered).

    Visual geoms (``contype=0, conaffinity=0``) are tolerated and
    skipped: the URDF import emits one mesh visual per link with
    explicit zeros, and we don't want to promote those into the active
    contact set.
    """
    mujoco = pytest.importorskip("mujoco")
    xml_path, json_path = _scene_paths(env_name)
    meta = json.loads(json_path.read_text())
    hand_roots = list(meta.get("hand_root_bodies", {}).values())
    assert hand_roots, (
        "metadata is missing 'hand_root_bodies' -- rebuild scenes via "
        "build_x2_robocasa_scene_xml --all"
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    seen_collision_geoms = 0
    seen_palm_primitives = 0
    seen_x2_wrist_mesh_disabled = False
    for (contype, conaffinity, _group, _gtype, meshname) in (
        _walk_hand_geom_collision_attrs(model, hand_roots)
    ):
        # The X2's pre-OmniHand fist collision mesh (the legacy
        # boxing-glove shell that lives on ``*_wrist_roll_link`` from
        # before the OmniHand was bolted on) must be DISABLED
        # (ct=ca=0) in the scene MJCF -- see the long block-comment
        # above
        # ``build_x2_robocasa_scene_xml._disable_pre_omnihand_x2_fist_collision_mesh``
        # for why. Previously this mesh was kept at the default (1, 1)
        # world channel, but that left it extending past the OmniHand
        # palm mount and physically blocked finger curl past
        # q≈+0.67 rad. The OmniHand palm primitives (cylinder + box,
        # on the (2, 1) hand channel) provide the actual graspable
        # contact surface, so disabling the legacy fist mesh doesn't
        # remove the hand from physics -- it just removes the obstacle
        # the fingers were punching into.
        if _is_x2_wrist_collision_mesh(meshname):
            assert (contype, conaffinity) == (0, 0), (
                f"pre-OmniHand X2 fist collision mesh {meshname!r} has "
                f"({contype}, {conaffinity}); expected (0, 0) so the "
                "OmniHand fingers can curl past the wrist cuff. "
                "Either the scene-build step's "
                "_disable_pre_omnihand_x2_fist_collision_mesh did not "
                "run, or the X2 vendor MJCF schema for wrist_roll_link "
                "changed -- rebuild scenes via "
                "``.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --all``."
            )
            seen_x2_wrist_mesh_disabled = True
            continue
        if contype == 0 and conaffinity == 0:
            continue  # visual-only geom, ignore
        seen_collision_geoms += 1
        # Track palm primitives separately -- these are the ones the
        # original (incomplete) walk missed because they live in the
        # wrist body, not under a child body.
        if not meshname:  # non-mesh primitive (cylinder/box) = palm
            seen_palm_primitives += 1
        assert contype == _HAND_COLLISION_CHANNEL and conaffinity == _WORLD_CHANNEL, (
            f"hand collision geom in {xml_path.name} has "
            f"(contype={contype}, conaffinity={conaffinity}); expected "
            f"({_HAND_COLLISION_CHANNEL}, {_WORLD_CHANNEL}) so finger-vs-hand "
            "contacts are filtered but finger-vs-everything-else still fires. "
            "Rebuild scenes via "
            "``.venv_sim/bin/python -m gear_sonic.scripts.build_x2_robocasa_scene_xml --all`` "
            "after pulling the latest compose_x2_with_omnihand."
        )
    assert seen_collision_geoms > 0, (
        f"{xml_path.name}: expected at least one hand collision geom "
        "(palm cylinder / thumb boxes / fingertip cylinders) but found "
        "none. The URDF import or the merge step likely stripped them."
    )
    assert seen_palm_primitives >= 4, (  # 2 sides × (cylinder + box)
        f"{xml_path.name}: expected at least 4 OmniHand palm primitives "
        f"(left/right × cylinder/box) but found {seen_palm_primitives}. "
        "Either the palm collision was lost, or the filter is missing the "
        "wrist body's own non-mesh geoms (the regression mode)."
    )
    assert seen_x2_wrist_mesh_disabled, (
        f"{xml_path.name}: expected to find X2's wrist_roll collision "
        "mesh DISABLED at (contype=0, conaffinity=0). The walk did not "
        "encounter it -- either the scene XML lost the X2 collision "
        "mesh entirely, or the wrist-collision disable helper did not "
        "run, or this test's discriminator is wrong. Either way, "
        "OmniHand fingers will stall at q≈+0.67 if this assertion is "
        "silently bypassed."
    )


@pytest.mark.parametrize("env_name", BUNDLED_SCENES)
def test_scene_xml_touchable_geoms_keep_default_channel(env_name: str) -> None:
    """Touchable scene-object colliders must keep their default (1, 1) bitmask.

    The (2, 1) self-collision filter on the hand only works because the
    cube / bowl / table / floor are all on the default world channel
    (1, 1) -- the contact rule then evaluates to ``(2 & 1) | (1 & 1) =
    1`` and the hand collides with them. If a future patch starts
    rewriting touchable scene-object channels (the way the abandoned
    "whitelist" prototype did), the filter math breaks and fingers
    silently stop gripping the cube.

    This guard asserts that for every collider declared in
    ``env_spec.object_contact_geoms`` (cube_collider, bowl_floor, …)
    the bitmask is still default (1, 1) and the contact rule against a
    hand geom (2, 1) evaluates to True.
    """
    mujoco = pytest.importorskip("mujoco")
    xml_path, json_path = _scene_paths(env_name)
    meta = json.loads(json_path.read_text())
    object_contact_geoms = meta.get("object_contact_geoms", {})
    assert object_contact_geoms, (
        f"{xml_path.name}: metadata is missing 'object_contact_geoms' -- "
        "rebuild scenes via build_x2_robocasa_scene_xml --all"
    )

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    def _collide(c1: int, a1: int, c2: int, a2: int) -> bool:
        return bool((c1 & a2) | (c2 & a1))

    expected_geom_names: list[str] = []
    for _logical, geoms in object_contact_geoms.items():
        expected_geom_names.extend(geoms)
    assert expected_geom_names, (
        f"{xml_path.name}: object_contact_geoms is empty; nothing for "
        "the hand to grasp"
    )

    for geom_name in expected_geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        assert gid >= 0, (
            f"{xml_path.name}: touchable-object geom {geom_name!r} "
            "(declared in object_contact_geoms) is missing from the "
            "compiled MJCF."
        )
        contype = int(model.geom_contype[gid])
        conaffinity = int(model.geom_conaffinity[gid])
        assert (contype, conaffinity) == (_WORLD_CHANNEL, _WORLD_CHANNEL), (
            f"{xml_path.name}: geom {geom_name!r} has "
            f"(contype={contype}, conaffinity={conaffinity}); expected the "
            f"default ({_WORLD_CHANNEL}, {_WORLD_CHANNEL}). "
            "The (2, 1) hand filter assumes touchable scene objects keep "
            "the default world channel -- if you intentionally re-classed "
            "this geom, update the hand filter to match."
        )
        # Direct rule evaluation: hand vs this geom must produce contact.
        assert _collide(contype, conaffinity, _HAND_COLLISION_CHANNEL, _WORLD_CHANNEL), (
            f"{xml_path.name}: geom {geom_name!r} ({contype}, {conaffinity}) "
            "fails to collide with a hand geom "
            f"({_HAND_COLLISION_CHANNEL}, {_WORLD_CHANNEL}) -- "
            "fingers will pass through it."
        )


def test_compose_filter_keeps_world_contacts_but_drops_self_contacts() -> None:
    """Direct unit test on the composer: no scene XML on disk required.

    Builds the augmented MJCF in-process with
    ``disable_hand_collisions=False`` and verifies the contact-rule
    ``(c1 & a2) || (c2 & a1)`` evaluates to:

    * ``True``  for hand vs default-world (1, 1) -- cube, bowl, table,
      floor, X2 body all collide with fingers / palm
    * ``False`` for hand vs hand (2, 1) x (2, 1) -- self contacts
      uniformly filtered (palm-vs-finger AND finger-vs-finger)

    These are the exact properties the bug fix must guarantee: the
    MuJoCo contact resolver applies that rule per candidate pair, so
    asserting it directly catches regressions in the bitmask choice
    even without a robocasa scene loaded.
    """
    pytest.importorskip("mujoco")
    from gear_sonic.scripts.compose_x2_with_omnihand import (
        build_x2_with_omnihand_spec,
    )

    _, model, _ = build_x2_with_omnihand_spec(disable_hand_collisions=False)
    hand_roots = ("left_wrist_roll_link", "right_wrist_roll_link")

    def _collide(c1: int, a1: int, c2: int, a2: int) -> bool:
        return bool((c1 & a2) | (c2 & a1))

    saw_one = False
    for (contype, conaffinity, _group, _gtype, meshname) in (
        _walk_hand_geom_collision_attrs(model, hand_roots)
    ):
        if contype == 0 and conaffinity == 0:
            continue
        if _is_x2_wrist_collision_mesh(meshname):
            continue  # X2's own wrist mesh stays at (1, 1)
        saw_one = True
        assert (contype, conaffinity) == (_HAND_COLLISION_CHANNEL, _WORLD_CHANNEL), (
            f"hand collision geom is ({contype}, {conaffinity}); the "
            f"self-collision filter should rewrite it to "
            f"({_HAND_COLLISION_CHANNEL}, {_WORLD_CHANNEL})."
        )
        # Hand vs default-world geom (cube, bowl, table, floor, X2 body): contact ✓
        assert _collide(contype, conaffinity, _WORLD_CHANNEL, _WORLD_CHANNEL), (
            "hand geom must collide with default-world geoms "
            "(contype=1, conaffinity=1) -- otherwise fingers pass "
            "through the cube AND the table AND the floor"
        )
        # Hand vs hand (same bitmask on both sides): no contact ✓
        assert not _collide(contype, conaffinity, contype, conaffinity), (
            "hand geom must NOT collide with another hand geom -- "
            "palm-vs-finger / finger-vs-finger pinning was the original bug"
        )
    assert saw_one, (
        "no hand collision geoms found under the wrist roots -- did the "
        "spec composition lose the OmniHand collision primitives?"
    )


# ── Pre-OmniHand X2 fist collision-mesh disable (scene-build only) ───────


@pytest.mark.parametrize("env_name", BUNDLED_SCENES)
def test_scene_xml_disables_pre_omnihand_x2_fist_collision_mesh(
    env_name: str,
) -> None:
    """Bundled scene XMLs must ship with the pre-OmniHand X2 fist
    collision mesh disabled (contype=0, conaffinity=0) on both
    ``*_wrist_roll_link`` bodies. Otherwise the OmniHand fingers
    physically punch into the legacy fist shell on every full-grasp
    curl and stall at q≈+0.67 rad regardless of friction / actuator
    settings.

    Standalone MuJoCo proves this is the actual root cause: with the
    pre-OmniHand fist mesh disabled, every finger joint reaches its
    commanded curl target on a fresh ``mj_step`` loop. With the legacy
    mesh enabled, PIPs saturate at +0.67 and the thumb sits near zero.
    See the long block-comment above
    ``build_x2_robocasa_scene_xml._disable_pre_omnihand_x2_fist_collision_mesh``
    for the full diagnostic walkthrough -- including why "increase
    friction" / "lower friction" / "boost actuator torque" workarounds
    don't help, and why this fix is intentionally scoped to the scene
    XML rather than the bare compose path.

    This test loads the bundled scene MJCF into MuJoCo, walks each
    wrist body's geoms, and asserts:

      * at least one geom on the body has ``contype=0, conaffinity=0``
        (the disabled legacy fist collision mesh), AND
      * at least two geoms on the body still have
        ``(contype=2, conaffinity=1)`` (the OmniHand palm primitives --
        cylinder + box -- which provide the actual graspable surface
        for the cube/bowl/table on the hand channel).

    If either invariant breaks, the operator will see fingers stalled
    again at recording time. The build-time verifier in
    ``build_scene_xml`` raises the same way; this test catches a stale
    bundled scene that hasn't been rebuilt after the fix landed.
    """
    mujoco = pytest.importorskip("mujoco")
    xml_path, _ = _scene_paths(env_name)
    model = mujoco.MjModel.from_xml_path(str(xml_path))

    for side in ("left", "right"):
        wrist_body = f"{side}_wrist_roll_link"
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist_body)
        assert bid >= 0, (
            f"{xml_path.name}: missing expected body {wrist_body!r} -- "
            "rebuild scenes via build_x2_robocasa_scene_xml --all"
        )
        n_disabled = 0
        n_palm = 0
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != bid:
                continue
            ct = int(model.geom_contype[g])
            ca = int(model.geom_conaffinity[g])
            if ct == 0 and ca == 0:
                n_disabled += 1
            elif (ct, ca) == (_HAND_COLLISION_CHANNEL, _WORLD_CHANNEL):
                n_palm += 1
        assert n_disabled >= 1, (
            f"{xml_path.name}: {wrist_body!r} has {n_disabled} disabled "
            "(ct=0, ca=0) geoms; expected at least 1 (the pre-OmniHand "
            "X2 fist collision shell). Without this, OmniHand fingers "
            "will stall at q≈+0.67 rad on full-grasp curls. Rebuild "
            "scenes via ``.venv_sim/bin/python -m "
            "gear_sonic.scripts.build_x2_robocasa_scene_xml --all`` "
            "after pulling the latest build_x2_robocasa_scene_xml."
        )
        assert n_palm >= 2, (
            f"{xml_path.name}: {wrist_body!r} only has {n_palm} OmniHand "
            "palm primitives left on the hand channel "
            f"({_HAND_COLLISION_CHANNEL}, {_WORLD_CHANNEL}); expected "
            "at least 2 (cylinder + box). The compose script's palm "
            "primitives are the OmniHand's actual graspable surface; "
            "without them the cube falls through the hand. Either the "
            "scene-build pre-OmniHand-fist-disable helper went too "
            "broad, or the compose pipeline stopped attaching palm "
            "collision."
        )
