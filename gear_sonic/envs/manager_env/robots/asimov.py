"""Asimov v1 (Menlo Research) embodiment for SONIC training.

23 actuated DOF, strict subset of Unitree G1's 29 (no waist pitch/roll, one
wrist DOF per arm, welded neck). Model truth: mjlab
``feature/asimov-velocity-training-cfg`` head ``f5c4caa`` — MJCF copied to
``data/assets/robot_description/mjcf/asimov.xml``; URDF GENERATED from it by
``gear_sonic/scripts/dev/mjcf_to_urdf_asimov.py`` (FK-verified to 1e-6 m,
elbow ``ref`` ±45° baked — URDF zero == MJCF qpos=0, the retarget-CSV/PKL
convention).

Unlike X2/H2 (armature-derived estimated gains), Asimov has HARDWARE-
CHARACTERIZED kp/kd/effort/armature from ``asimov_1_constant.py`` — used
verbatim. ``effort`` is the hard clamp (NOT ``saturation_effort``, which
feeds the torque-speed derating curve; using it would be phantom torque —
the same class of bug as X2's waist 48-vs-36).

Asimov quirks that differ from G1/X2 and are easy to get wrong:
- The MJCF right side is SIGN-MIRRORED vs the left (right_hip_pitch axis
  0,-1,0; right knee range -86..0; right elbow -140..0). Default pose values
  below are therefore mirrored per side, not shared regexes.
- MJCF joint order puts the RIGHT arm before the LEFT arm.
- Training assets RENAME pelvis_link->pelvis and waist_yaw_link->torso_link
  (framework body-name conventions: motion.yaml defaults, base_com DR). Welded neck bodies
  are FUSED into waist_yaw_link (scripts/dev/fuse_asimov_neck.py) so bodies ==
  1 + num_dof == 24, the pose_aa convention every consumer assumes.
- Parallel RSU ankle is unmodeled (serial approximation); ankle_roll ROM is
  only ±5.7°.
"""

from typing import Literal

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

from . import mapping_utils

ASSET_DIR = "gear_sonic/data/assets"

# Hardware-characterized per-joint parameters from mjlab asimov_1_constant.py:
# joint basename -> (kp, kd, effort_hard_clamp, velocity_limit, armature)
ASIMOV_JOINT_PARAMS = {
    "hip_pitch": (150.0, 5.0, 40.0, 12.57, 0.0698),
    "hip_roll": (150.0, 5.0, 30.0, 3.98, 0.1400),
    "hip_yaw": (150.0, 5.0, 20.0, 5.45, 0.0687),
    "knee": (150.0, 5.0, 25.0, 12.25, 0.0330),
    "ankle_pitch": (440.0, 20.0, 40.0, 9.32, 0.0484),
    "ankle_roll": (440.0, 20.0, 17.0, 9.32, 0.0484),
    "waist_yaw": (65.0, 5.0, 40.0, 12.57, 0.0698),
    "shoulder_pitch": (57.0, 5.0, 30.0, 3.98, 0.1400),
    "shoulder_roll": (86.0, 5.0, 25.0, 12.25, 0.0330),
    "shoulder_yaw": (96.0, 5.0, 20.0, 5.45, 0.0687),
    "elbow": (40.0, 2.0, 12.0, 9.32, 0.0242),
    "wrist_yaw": (40.0, 2.0, 12.0, 9.32, 0.0242),
}

# MuJoCo orders, straight from mjcf/asimov.xml (DFS in XML order).
ASIMOV_MUJOCO_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_yaw_joint",
]
ASIMOV_MUJOCO_BODIES = [
    "pelvis",
    "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "torso_link",
    "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_yaw_link",
    "left_shoulder_pitch_link", "left_shoulder_roll_link",
    "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_yaw_link",
]

# IsaacLab orders: BFS with alphabetical sibling sort (per-parent groups in
# parent order), PREDICTED from the URDF tree. The prediction is asserted
# against runtime robot.joint_names at env construction (modular_tracking_env
# smoke with num_envs=1) — conventions.md says never trust a derivation, so
# any mismatch is a hard error that prints the actual order to paste here.
ASIMOV_ISAACLAB_JOINTS_ORDER = [
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
]
ASIMOV_ISAACLAB_BODIES = [
    "pelvis",
    "left_hip_pitch_link", "right_hip_pitch_link", "torso_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_knee_link", "right_knee_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
    "left_ankle_pitch_link", "right_ankle_pitch_link",
    "left_elbow_link", "right_elbow_link",
    "left_ankle_roll_link", "right_ankle_roll_link",
    "left_wrist_yaw_link", "right_wrist_yaw_link",
]

# Generated, never hand-typed (mapping_utils regression-tested against the
# X2 hand-typed constants).
ASIMOV_ISAACLAB_TO_MUJOCO_MAPPING = mapping_utils.build_isaaclab_mujoco_mapping(
    il_body_names=ASIMOV_ISAACLAB_BODIES,
    mj_body_names=ASIMOV_MUJOCO_BODIES,
    il_joint_names=ASIMOV_ISAACLAB_JOINTS_ORDER,
    mj_joint_names=ASIMOV_MUJOCO_JOINTS,
)


def _group(basenames, field):
    """Per-joint dict for an actuator group from ASIMOV_JOINT_PARAMS."""
    i = {"kp": 0, "kd": 1, "effort": 2, "velocity": 3, "armature": 4}[field]
    return {f".*_{b}_joint" if b != "waist_yaw" else "waist_yaw_joint":
            ASIMOV_JOINT_PARAMS[b][i] for b in basenames}


def make_asimov_cfg(
    actuator_regime: Literal["implicit", "explicit"] = "implicit",
) -> ArticulationCfg:
    ActuatorCls = ImplicitActuatorCfg if actuator_regime == "implicit" else IdealPDActuatorCfg

    legs = ["hip_pitch", "hip_roll", "hip_yaw", "knee"]
    feet = ["ankle_pitch", "ankle_roll"]
    arms = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_yaw"]

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=f"{ASSET_DIR}/robot_description/urdf/asimov/asimov.urdf",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # mjlab KNEES_STAND_KEYFRAME: pelvis 0.639 m; spawn slightly high.
            # Right side is sign-mirrored — per-side values, NOT shared regexes.
            pos=(0.0, 0.0, 0.70),
            joint_pos={
                "left_hip_pitch_joint": 0.1,
                "right_hip_pitch_joint": -0.1,
                "left_shoulder_pitch_joint": -0.35,
                "right_shoulder_pitch_joint": 0.35,
                "left_shoulder_roll_joint": -0.18,
                "right_shoulder_roll_joint": 0.18,
                "left_elbow_joint": 0.87,
                "right_elbow_joint": -0.87,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ActuatorCls(
                joint_names_expr=[f".*_{b}_joint" for b in legs],
                effort_limit_sim=_group(legs, "effort"),
                velocity_limit_sim=_group(legs, "velocity"),
                stiffness=_group(legs, "kp"),
                damping=_group(legs, "kd"),
                armature=_group(legs, "armature"),
            ),
            "feet": ActuatorCls(
                joint_names_expr=[f".*_{b}_joint" for b in feet],
                effort_limit_sim=_group(feet, "effort"),
                velocity_limit_sim=_group(feet, "velocity"),
                stiffness=_group(feet, "kp"),
                damping=_group(feet, "kd"),
                armature=_group(feet, "armature"),
            ),
            "waist_yaw": ActuatorCls(
                joint_names_expr=["waist_yaw_joint"],
                effort_limit_sim=40.0,
                velocity_limit_sim=12.57,
                stiffness=65.0,
                damping=5.0,
                armature=0.0698,
            ),
            "arms": ActuatorCls(
                joint_names_expr=[f".*_{b}_joint" for b in arms],
                effort_limit_sim=_group(arms, "effort"),
                velocity_limit_sim=_group(arms, "velocity"),
                stiffness=_group(arms, "kp"),
                damping=_group(arms, "kd"),
                armature=_group(arms, "armature"),
            ),
        },
    )


ASIMOV_CFG = make_asimov_cfg()

# Action scale: repo convention 0.25 * effort / kp per joint (mjlab itself
# trains with 0.30 * effort / kp — noted delta, revisit if tracking is too
# timid). Keyed by joint name pattern like the other robots.
ASIMOV_ACTION_SCALE = {
    (f".*_{b}_joint" if b != "waist_yaw" else "waist_yaw_joint"):
        0.25 * p[2] / p[0]
    for b, p in ASIMOV_JOINT_PARAMS.items()
}
