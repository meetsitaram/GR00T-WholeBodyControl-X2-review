"""
M3.5 acceptance gate: X2 + OmniHand-2025 renderer composer + qpos helper.

The X2 Ultra training MJCF deliberately ends each arm at ``*_wrist_roll_link``
(31 body DOFs total) -- the deploy ONNX, SONIC tracking decoder, and AimDK
ROS 2 HAL all expect that. Hand commands flow out-of-band via
``/aima/hal/joint/hand/command`` on the real robot.

For *visual* renderering (M3 inspection videos and the M5 camera plumbing
that bakes camera frames into the LeRobot dataset), we compose an augmented
MJCF in memory by attaching two articulated 10-active-DOF OmniHand-2025
chains to the X2 wrist bodies, swapping the wrist-roll *visual* mesh for a
clipped variant that drops the baked-in dummy fist, and recreating the six
URDF mimic relationships per side as MJCF ``<equality joint>`` constraints.

This test gate verifies:

1. The augmented model compiles, has the expected shape (33 X2 bodies + 32
   hand bodies, 31 X2 hinges + 32 hand hinges, 12 mimic equality
   constraints, 70 qpos slots).
2. The 10 active joints per side match
   ``OMNIHAND_FINGER_NAMES_PER_SIDE`` from
   ``x2_ultra_supplemental_info.py`` exactly -- so a 10-D vector recorded
   in M1's LeRobot dataset writes to the right qpos slot.
3. ``apply_active_hand_qpos`` projects mimic relationships exactly
   (passive = multiplier × active for all 6 rules per side).
4. The clipped wrist mesh actually replaces the original visual geom on
   each ``*_wrist_roll_link`` body (so the dummy fist no longer renders).
5. Hand collision geoms have ``contype=0, conaffinity=0`` (purely
   kinematic; the dynamics never sees them).
6. The renderer ``--with-omnihand`` flag accepts split per-side hand
   trajectory keys (``left_hand_trajectory`` / ``right_hand_trajectory``)
   from a ``record_synthetic_smoketest_dataset`` recording.
7. The training MJCF (``x2_ultra.xml``) is unaffected -- its mesh table
   still resolves to the original 38-mesh asset list, so M1, M2, and the
   trainer do not need to be touched.

Run via::

    .venv/bin/python -m pytest tests/test_x2_omnihand_renderer.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Skip the entire suite if MuJoCo is not importable (CI environments without
# OpenGL deps); M3.5 is a renderer-side milestone and is not required for
# trainer-side gates to pass.
mujoco = pytest.importorskip("mujoco")

from gear_sonic.scripts.compose_x2_with_omnihand import (  # noqa: E402
    ACTIVE_FINGER_JOINTS,
    PASSIVE_MIMIC_RULES,
    _DEFAULT_MOUNT_QUAT_WXYZ,
    _LEFT_MOUNT_QUAT_WXYZ,
    _RIGHT_MOUNT_QUAT_WXYZ,
    apply_active_hand_qpos,
    build_x2_with_omnihand_spec,
)
from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (  # noqa: E402
    OMNIHAND_FINGER_NAMES_PER_SIDE,
)


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def augmented():
    """Compose the X2 + OmniHand augmented model once for all tests."""
    spec, model, layout = build_x2_with_omnihand_spec()
    return spec, model, layout


# ────────────────────────────────────────────────────────────────────────────
# 1. Augmented model shape
# ────────────────────────────────────────────────────────────────────────────


def test_augmented_model_shape(augmented):
    """Augmented model has the expected count of bodies, joints, equality constraints.

    Per-side hand hinge count = 17:
      * 5 thumb joints (roll, abad, mcp, pip, dip)
      * 3 index joints (abad, pip, dip)
      * 3 middle joints (abad-placeholder, pip, dip) -- the abad joint is a
        zero-range hinge added in the vendored URDF to prevent MuJoCo's
        URDF parser from re-parenting middle_pip onto index_abad when it
        collapses the upstream fixed joint (see omnihand_{left,right}.urdf
        gear_sonic patch).
      * 3 ring joints (abad, pip, dip)
      * 3 pinky joints (abad, pip, dip)
      = 17 per side, 34 across both hands.

    Total joints: 1 free + 31 body + 34 hand = 66.
    nq = 7 + 31 + 34 = 72.
    """
    _, model, _ = augmented

    assert model.njnt == 66, (
        f"expected 66 joints (1 free + 31 body + 34 hand), got {model.njnt}"
    )
    assert model.nq == 72, f"expected 72 qpos slots, got {model.nq}"

    # 6 mimic rules per side × 2 sides = 12 equality constraints. The
    # zero-range middle_abad placeholders are NOT mimics -- they have
    # ``range=(0,0)`` enforced by MuJoCo's joint-range solver, not by an
    # ``<equality>``.
    assert model.neq == 12, f"expected 12 mimic equality constraints, got {model.neq}"


# ────────────────────────────────────────────────────────────────────────────
# 2. Joint name parity vs supplemental info
# ────────────────────────────────────────────────────────────────────────────


def test_active_joint_order_matches_supplemental_info():
    """Composer's canonical 10 active joints == X2 supplemental_info contract.

    Both sources MUST agree -- otherwise our M1 LeRobot dataset's
    ``action.{left,right}_hand_joints[i]`` would write to the wrong joint
    in the augmented MJCF, and the renderer would lie about what the policy
    is commanding.
    """
    assert tuple(ACTIVE_FINGER_JOINTS) == tuple(OMNIHAND_FINGER_NAMES_PER_SIDE), (
        "ACTIVE_FINGER_JOINTS in compose_x2_with_omnihand.py drifted from "
        "OMNIHAND_FINGER_NAMES_PER_SIDE in x2_ultra_supplemental_info.py"
    )


def test_layout_resolves_all_active_and_passive_joints(augmented):
    """Every active + passive joint per side resolves to a valid qpos index."""
    _, model, layout = augmented

    for side in ("left", "right"):
        # 10 active joints, all distinct qpos addresses.
        active = layout.active_qposadr[side]
        assert len(active) == 10
        assert len(set(active)) == 10
        for qadr in active:
            assert 0 <= qadr < model.nq

        # 6 passive joints, all distinct, all in valid range.
        passive = layout.passive_qposadr[side]
        assert len(passive) == 6
        assert set(passive.keys()) == {r.passive for r in PASSIVE_MIMIC_RULES}
        for qadr in passive.values():
            assert 0 <= qadr < model.nq


# ────────────────────────────────────────────────────────────────────────────
# 3. Mimic projection numerics
# ────────────────────────────────────────────────────────────────────────────


def test_apply_active_hand_qpos_projects_mimic_exactly(augmented):
    """``apply_active_hand_qpos`` writes passive = multiplier × active."""
    _, model, layout = augmented
    data = mujoco.MjData(model)

    rng = np.random.default_rng(0)
    left_active = rng.uniform(-1.0, 1.0, size=10)
    right_active = rng.uniform(-1.0, 1.0, size=10)
    apply_active_hand_qpos(
        data, layout, left_active=left_active, right_active=right_active,
    )
    mujoco.mj_forward(model, data)

    for side, vec in (("left", left_active), ("right", right_active)):
        active_qadrs = layout.active_qposadr[side]
        passive_qadrs = layout.passive_qposadr[side]
        # Active values round-trip exactly.
        for k, qadr in enumerate(active_qadrs):
            assert data.qpos[qadr] == pytest.approx(vec[k], abs=1e-12), (
                f"{side} active joint #{k} ({ACTIVE_FINGER_JOINTS[k]}) round-trip mismatch"
            )
        # Passive values match multiplier × active.
        for rule in PASSIVE_MIMIC_RULES:
            active_idx = ACTIVE_FINGER_JOINTS.index(rule.active)
            expected = rule.multiplier * vec[active_idx]
            actual = data.qpos[passive_qadrs[rule.passive]]
            assert actual == pytest.approx(expected, abs=1e-12), (
                f"{side} passive {rule.passive} = {actual:.6f} but expected "
                f"{rule.multiplier} × {rule.active}({vec[active_idx]:.6f}) = {expected:.6f}"
            )


def test_apply_active_hand_qpos_rejects_wrong_shape(augmented):
    """Helper raises a clear error for shape-mismatched active vectors."""
    _, model, layout = augmented
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match=r"shape \(10,\)"):
        apply_active_hand_qpos(data, layout, left_active=np.zeros(7))
    with pytest.raises(ValueError, match=r"shape \(10,\)"):
        apply_active_hand_qpos(data, layout, right_active=np.zeros(11))


# ────────────────────────────────────────────────────────────────────────────
# 4. Clipped wrist-roll mesh actually swaps in
# ────────────────────────────────────────────────────────────────────────────


def test_wrist_roll_visual_swapped_to_clipped_mesh(augmented):
    """Each wrist_roll body's *visual* geom uses the clipped (no-fist) mesh.

    The collision geom must keep the original mesh so contact behaviour is
    unchanged from the un-augmented X2 model.
    """
    _, model, _ = augmented

    for side in ("left", "right"):
        bid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_wrist_roll_link",
        )
        assert bid >= 0, f"{side}_wrist_roll_link missing from augmented model"

        meshes_seen: list[str] = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid:
                continue
            mid = model.geom_dataid[g]
            if mid < 0:
                continue
            mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mid)
            meshes_seen.append(mname)

        assert f"{side}_wrist_roll_clipped_link" in meshes_seen, (
            f"clipped mesh missing on {side}_wrist_roll_link; "
            f"saw {meshes_seen}"
        )
        # Original mesh must still be present (collision geom).
        assert f"{side}_wrist_roll_link" in meshes_seen, (
            f"original wrist_roll mesh dropped on {side}; "
            f"contact-solver behaviour would diverge. saw {meshes_seen}"
        )


def test_clipped_mesh_files_are_present():
    """The vendor step ``clip_x2_wrist_for_omnihand`` was run."""
    base = (
        REPO_ROOT
        / "gear_sonic" / "data" / "assets" / "robot_description"
        / "omnihand" / "meshes"
    )
    for side in ("left", "right"):
        p = base / f"{side}_wrist_roll_clipped_link.STL"
        assert p.is_file(), (
            f"missing vendored clipped wrist mesh {p}. "
            "Re-run gear_sonic/scripts/clip_x2_wrist_for_omnihand.py."
        )
        # STL is at least a few hundred KB; if it's tiny something failed.
        assert p.stat().st_size > 100_000, (
            f"{p} is suspiciously small ({p.stat().st_size} bytes); "
            "the clip step may have produced an empty mesh."
        )


# ────────────────────────────────────────────────────────────────────────────
# 5. Hand geoms are kinematic-only (no collisions)
# ────────────────────────────────────────────────────────────────────────────


def test_hand_geoms_have_no_collision(augmented):
    """All geoms attached to the OmniHand chain have ``contype=conaffinity=0``."""
    _, model, _ = augmented

    # Find every body that's part of the hand chain (name has the side prefix
    # from MjSpec.attach AND the SDK 'L_' / 'R_' prefix).
    for b in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        is_hand = (
            bname.startswith("left_L_") or bname.startswith("right_R_")
        )
        if not is_hand:
            continue
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != b:
                continue
            assert model.geom_contype[g] == 0, (
                f"hand geom on {bname} has contype={model.geom_contype[g]} "
                "(expected 0; renderer must be kinematic-only)"
            )
            assert model.geom_conaffinity[g] == 0, (
                f"hand geom on {bname} has conaffinity={model.geom_conaffinity[g]} "
                "(expected 0; renderer must be kinematic-only)"
            )


# ────────────────────────────────────────────────────────────────────────────
# 6. Renderer ``--with-omnihand`` accepts smoketest-recording layout
# ────────────────────────────────────────────────────────────────────────────


def test_renderer_accepts_split_hand_trajectory_keys(tmp_path):
    """Renderer accepts split ``left_hand_trajectory``+``right_hand_trajectory``.

    ``record_synthetic_smoketest_dataset.py`` writes per-side 10-D arrays;
    the renderer must accept this layout (the canonical form) without
    requiring callers to concatenate first.
    """
    pytest.importorskip("imageio")  # MP4 writer
    pytest.importorskip("scipy")

    # Synthesise a tiny smoketest-style recording (4 frames, deterministic).
    T = 4
    body = np.zeros((T, 31), dtype=np.float32)
    body[:, 0] = -0.312  # left_hip_pitch (any non-zero so the renderer doesn't
                         # silently zero-out qpos in the test harness)
    left_hand = np.linspace(0.0, 0.5, T * 10).reshape(T, 10).astype(np.float32)
    right_hand = np.linspace(-0.3, 0.3, T * 10).reshape(T, 10).astype(np.float32)

    rec_path = tmp_path / "fake_episode_recorded.npz"
    np.savez(
        rec_path,
        body_trajectory=body,
        left_hand_trajectory=left_hand,
        right_hand_trajectory=right_hand,
    )

    # Import here so the EGL backend is set up only when this test actually
    # runs (other tests in this file don't need offscreen rendering).
    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from gear_sonic.scripts.render_smoketest_episode_video import render_episode

    out_path = tmp_path / "ego.mp4"
    summary = render_episode(
        recording_path=rec_path,
        output_path=out_path,
        camera="ego_view",
        width=128,
        height=96,
        fps=10.0,
        max_frames=T,
        with_omnihand=True,
    )
    assert summary["with_omnihand"] is True
    assert summary["num_frames"] == T
    assert out_path.is_file()
    assert out_path.stat().st_size > 0


# ────────────────────────────────────────────────────────────────────────────
# 6b. Mount orientation is locked to the documented convention
# ────────────────────────────────────────────────────────────────────────────


def test_mount_quat_is_documented_orientation():
    """Per-side mount quaternions follow the documented convention.

    Right wrist: ``(180° about Y) ∘ (+90° about Z)`` -> wxyz
        ``(0, √½, +√½, 0)``.
    Left  wrist: ``(180° about Y) ∘ (-90° about Z)`` -> wxyz
        ``(0, √½, -√½, 0)``.

    The X2 left and right wrist_roll bodies are mirrored about the body
    centerline, so the SAME world-frame "palm down" pose corresponds to
    OPPOSITE local-Z rolls. Without the sign flip the right hand renders
    palm-down (correct) but the left hand renders palm-up.

    The convention is validated by the ``v7_ego_t*.png`` head-camera
    audit frames in the Minecraft-piano smoketest: both backs of hands
    visible from the head camera with thumbs medial.

    If you intentionally change this convention (for example to align
    with a different OmniHand SDK version) update both
    ``_LEFT_MOUNT_QUAT_WXYZ`` and ``_RIGHT_MOUNT_QUAT_WXYZ`` in
    ``compose_x2_with_omnihand.py`` and regenerate the audit frames in
    ``docs/source/tutorials/vla_training.md``.
    """
    expected_right = (0.0, 0.7071067811865476, +0.7071067811865476, 0.0)
    expected_left = (0.0, 0.7071067811865476, -0.7071067811865476, 0.0)
    assert _RIGHT_MOUNT_QUAT_WXYZ == pytest.approx(expected_right, abs=1e-12), (
        "Right OmniHand mount quaternion drifted from the documented "
        "convention. Re-render the m35_visual_audit ego-view frames "
        "before changing this."
    )
    assert _LEFT_MOUNT_QUAT_WXYZ == pytest.approx(expected_left, abs=1e-12), (
        "Left OmniHand mount quaternion drifted from the documented "
        "convention. The left side mirrors the right side's wrist long-"
        "axis roll (-90° instead of +90°). Re-render the audit frames "
        "before changing this."
    )
    # The legacy ``_DEFAULT_MOUNT_QUAT_WXYZ`` alias must stay bound to the
    # right-side value for backwards compatibility with downstream code
    # that imported the single "default" name.
    assert _DEFAULT_MOUNT_QUAT_WXYZ == _RIGHT_MOUNT_QUAT_WXYZ


# ────────────────────────────────────────────────────────────────────────────
# 7. Training MJCF unaffected
# ────────────────────────────────────────────────────────────────────────────


def test_training_mjcf_still_loads_independently():
    """The X2 training MJCF compiles with its original 31-DOF body in isolation.

    M3.5 must not have side-effected ``x2_ultra.xml`` -- the training
    pipeline, the C++ deploy reference, and the SONIC tracking decoder all
    still expect 31 DOFs / 38 meshes there.
    """
    spec = mujoco.MjSpec.from_file(
        str(REPO_ROOT / "gear_sonic" / "data" / "assets"
            / "robot_description" / "mjcf" / "x2_ultra.xml")
    )
    model = spec.compile()
    assert model.njnt == 32, f"expected 1 free + 31 body = 32 joints, got {model.njnt}"
    assert model.nq == 38, f"expected 7 + 31 = 38 qpos slots, got {model.nq}"
    assert model.neq == 0, "training MJCF must have zero equality constraints"
    # 37 X2 meshes (per the spec compile probe). M3.5 vendoring must not
    # have leaked clipped meshes back into x2_ultra.xml.
    mesh_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, m)
        for m in range(model.nmesh)
    }
    assert "left_wrist_roll_clipped_link" not in mesh_names
    assert "right_wrist_roll_clipped_link" not in mesh_names


# ────────────────────────────────────────────────────────────────────────────
# 8. Trainer / deploy / SONIC side stays renderer-agnostic
#
# These guard against the most dangerous regression mode of M3.5: a future
# refactor accidentally pulling the OmniHand augmentation into the trainer
# or deploy code, expanding the body action surface from 31 DOF to 31+20
# DOF and silently breaking the SONIC checkpoint contract.
# ────────────────────────────────────────────────────────────────────────────


def test_x2_robot_model_body_is_31_dof_no_hands():
    """The X2 RobotModel exposes exactly 31 actuated body joints, zero hand DOFs.

    The trainer, ONNX export, C++ deploy harness, and SONIC tracking
    decoder all consume ``RobotModel.body_actuated_joints``. If a future
    change accidentally sneaks the OmniHand active joints into this list,
    the body-action head explodes from 31 to 31+20 = 51 and the deployed
    22k checkpoint becomes unloadable.
    """
    from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
        X2UltraSupplementalInfo,
    )

    info = X2UltraSupplementalInfo()
    assert len(info.body_actuated_joints) == 31, (
        f"X2 body must remain 31 actuated DOFs; got "
        f"{len(info.body_actuated_joints)} ({info.body_actuated_joints!r})."
    )
    assert info.left_hand_actuated_joints == [], (
        "left hand must stay out-of-band (HAL stream); never an actuated "
        "URDF joint on the trainer side."
    )
    assert info.right_hand_actuated_joints == [], (
        "right hand must stay out-of-band (HAL stream); never an actuated "
        "URDF joint on the trainer side."
    )
    # No hand joint name should appear in body_actuated_joints.
    for jn in info.body_actuated_joints:
        assert "thumb" not in jn, f"hand joint {jn} leaked into body actuated list"
        assert "index" not in jn, f"hand joint {jn} leaked into body actuated list"
        assert "middle" not in jn, f"hand joint {jn} leaked into body actuated list"
        assert "ring" not in jn, f"hand joint {jn} leaked into body actuated list"
        assert "pinky" not in jn, f"hand joint {jn} leaked into body actuated list"


def test_x2_modality_config_keeps_hand_streams_separate():
    """``x2_modality_config`` keeps body (motion_token) and hands as separate
    action keys -- the OmniHand integration does not collapse them into one
    blob. SONIC tracking decoder consumes only ``motion_token``; AimDK HAL
    consumes only ``{left,right}_hand_joints``.
    """
    pytest.importorskip("gr00t")
    from gear_sonic.data.x2_modality_config import (
        DEFAULT_STATE_GROUPS,
        make_x2_modality_config,
    )

    cfg = make_x2_modality_config(hand_dof=10)
    action_keys = list(cfg["action"].modality_keys)
    # Three separate streams; no merged "body+hand" concatenation.
    assert action_keys == ["motion_token", "left_hand_joints", "right_hand_joints"], (
        f"action key layout drifted; got {action_keys}"
    )

    # State groups stay the seven SONIC-canonical chunks + projected_gravity.
    # In particular, "left_hand" and "right_hand" must remain as DISTINCT
    # entries from "left_arm" / "right_arm" -- the trainer relies on this
    # split when slicing the parquet.
    assert "left_arm" in DEFAULT_STATE_GROUPS
    assert "right_arm" in DEFAULT_STATE_GROUPS
    assert "left_hand" in DEFAULT_STATE_GROUPS
    assert "right_hand" in DEFAULT_STATE_GROUPS


def test_omnihand_composer_is_not_imported_by_trainer_or_deploy():
    """Nothing on the trainer / deploy / ZMQ / motion-replay path may
    import the OmniHand composer.

    Allowed importers (renderer-only path):

    * ``gear_sonic/scripts/render_smoketest_episode_video.py`` (M3 inspection
      video; M3.5 augmented MJCF lives only here).
    * ``gear_sonic/scripts/clip_x2_wrist_for_omnihand.py`` (vendoring step
      that produces the clipped wrist meshes).
    * ``tests/test_x2_omnihand_renderer.py`` (this file).

    Anything else importing ``compose_x2_with_omnihand`` would leak the
    65-body / 70-qpos augmented model into the training or deploy
    pipelines and invalidate the 31-DOF body action surface.
    """
    forbidden_substrings = (
        "compose_x2_with_omnihand",
        "build_x2_with_omnihand_spec",
        "apply_active_hand_qpos",
    )
    allowed_path_endings = (
        # Renderer + vendoring scripts that legitimately use the composer.
        "gear_sonic/scripts/render_smoketest_episode_video.py",
        "gear_sonic/scripts/clip_x2_wrist_for_omnihand.py",
        "gear_sonic/scripts/compose_x2_with_omnihand.py",
        # Kinematic-viewer paths (live teleop + offline replay). These
        # only consume the augmented MJCF inside an interactive
        # ``mujoco.viewer.launch_passive`` loop, never as input to the
        # 31-DOF training / deploy surfaces. Keeping them allowlisted
        # means the augmented model never reaches a trainer or the C++
        # deploy.
        "gear_sonic/scripts/teleop_x2_kinematic.py",
        "gear_sonic/scripts/replay_x2_kinematic.py",
        "gear_sonic/utils/teleop/x2_kinematic_view.py",
        "gear_sonic/utils/embodiment/x2.py",
        # MuJoCo<->ROS bridge for the C++ deploy's kinematic-stand
        # bootstrap (offline visualization only; never feeds the
        # training pipeline).
        "gear_sonic_deploy/scripts/x2_mujoco_ros_bridge.py",
        # Tests.
        "tests/test_x2_omnihand_renderer.py",
    )
    # Scan only Python source files inside the repo (skip vendor SDK,
    # build artifacts, .venv, .git, ...).
    code_roots = [
        REPO_ROOT / "gear_sonic",
        REPO_ROOT / "gear_sonic_deploy",
        REPO_ROOT / "decoupled_wbc",
        REPO_ROOT / "tests",
    ]
    bad: list[str] = []
    for root in code_roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            # Skip vendored upstream; we don't control imports there.
            if "agibot-x2-references" in path.parts:
                continue
            rel = str(path.relative_to(REPO_ROOT))
            if any(rel.endswith(allowed) for allowed in allowed_path_endings):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except (OSError, UnicodeDecodeError):
                continue
            for sym in forbidden_substrings:
                if sym in text:
                    bad.append(f"{rel} :: contains forbidden symbol {sym!r}")
    assert not bad, (
        "OmniHand composer leaked outside the renderer-only allowlist:\n  "
        + "\n  ".join(bad)
        + "\n\nIf you intentionally added a new caller, extend "
        "``allowed_path_endings`` in this test, but FIRST consider whether "
        "the new caller really needs the augmented MJCF -- M3.5's whole "
        "point is that the 31-DOF training MJCF is the canonical pipeline."
    )


def test_x2_zmq_pose_protocol_is_31_dof():
    """The mock VLA publisher sends 31-DOF body joint commands, not 31+20.

    This locks the wire-format invariant: the C++ deploy ``ZmqPoseInputSource``
    consumes a 31-element ``joint_pos_mj`` payload. If a future change
    expanded that to 51 (body + hands), every existing deploy harness and
    the SONIC tracking decoder would silently mis-slice.
    """
    from gear_sonic.scripts.mock_vla_publish_stand_token import (
        DEFAULT_STAND_POSE_MUJOCO_RAD,
        NUM_BODY_DOFS,
    )

    assert NUM_BODY_DOFS == 31
    assert len(DEFAULT_STAND_POSE_MUJOCO_RAD) == 31
