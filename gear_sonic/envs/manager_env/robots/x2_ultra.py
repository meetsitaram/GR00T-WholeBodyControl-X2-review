from typing import Literal

from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import isaaclab.sim as sim_utils

ASSET_DIR = "gear_sonic/data/assets"

# Agibot X2 Ultra motor parameters.
# Exact rotor inertia (armature) values are not available from the vendor;
# using H2 motor constants as starting estimates grouped by torque class.
# These MUST be tuned once real motor datasheets are obtained.
ARMATURE_HIP_KNEE = 0.025101925  # 120 N-m class (hip pitch/roll/yaw, knee)
ARMATURE_WAIST_YAW = 0.010177520  # 120 N-m waist yaw
ARMATURE_WAIST_PR = 0.003609725  # 48 N-m waist pitch/roll
ARMATURE_ANKLE = 0.003609725  # 36/24 N-m ankle
ARMATURE_SHOULDER_ELBOW = 0.003609725  # 36/24 N-m shoulder/elbow
ARMATURE_WRIST = 0.00425  # 4.8 N-m wrist pitch/roll
ARMATURE_WRIST_YAW = 0.003609725  # 24 N-m wrist yaw
ARMATURE_HEAD = 0.00425  # 2.6/0.6 N-m head

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_HIP_KNEE = ARMATURE_HIP_KNEE * NATURAL_FREQ**2
STIFFNESS_WAIST_YAW = ARMATURE_WAIST_YAW * NATURAL_FREQ**2
STIFFNESS_WAIST_PR = ARMATURE_WAIST_PR * NATURAL_FREQ**2
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ**2
STIFFNESS_SHOULDER_ELBOW = ARMATURE_SHOULDER_ELBOW * NATURAL_FREQ**2
STIFFNESS_WRIST = ARMATURE_WRIST * NATURAL_FREQ**2
STIFFNESS_WRIST_YAW = ARMATURE_WRIST_YAW * NATURAL_FREQ**2
STIFFNESS_HEAD = ARMATURE_HEAD * NATURAL_FREQ**2

DAMPING_HIP_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_HIP_KNEE * NATURAL_FREQ
DAMPING_WAIST_YAW = 2.0 * DAMPING_RATIO * ARMATURE_WAIST_YAW * NATURAL_FREQ
DAMPING_WAIST_PR = 2.0 * DAMPING_RATIO * ARMATURE_WAIST_PR * NATURAL_FREQ
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ
DAMPING_SHOULDER_ELBOW = 2.0 * DAMPING_RATIO * ARMATURE_SHOULDER_ELBOW * NATURAL_FREQ
DAMPING_WRIST = 2.0 * DAMPING_RATIO * ARMATURE_WRIST * NATURAL_FREQ
DAMPING_WRIST_YAW = 2.0 * DAMPING_RATIO * ARMATURE_WRIST_YAW * NATURAL_FREQ
DAMPING_HEAD = 2.0 * DAMPING_RATIO * ARMATURE_HEAD * NATURAL_FREQ

# Body names in IsaacLab BFS traversal order (32 bodies including root pelvis).
# Verified against runtime robot.joint_names.  Isaac Lab sorts children
# alphabetically within each BFS level, so head_yaw_link ("h") precedes
# left/right_shoulder_pitch_link ("l"/"r") at the same depth.
X2_ULTRA_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_pitch_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "head_yaw_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "head_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
]

# DOF index mappings between IsaacLab and MuJoCo orderings (31 DOF).
# isaaclab_to_mujoco[i] = MuJoCo index of the joint at IsaacLab index i.
X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF = [
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 29, 15, 22, 4, 10,
    30, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
]
X2_ULTRA_MUJOCO_TO_ISAACLAB_DOF = [
    0, 3, 6, 9, 14, 19, 1, 4, 7, 10, 15, 20, 2, 5, 8, 12,
    17, 21, 23, 25, 27, 29, 13, 18, 22, 24, 26, 28, 30, 11, 16,
]

# Body index mappings between IsaacLab and MuJoCo orderings (32 bodies).
X2_ULTRA_ISAACLAB_TO_MUJOCO_BODY = [
    0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 30, 16, 23, 5,
    11, 31, 17, 24, 6, 12, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29,
]
X2_ULTRA_MUJOCO_TO_ISAACLAB_BODY = [
    0, 1, 4, 7, 10, 15, 20, 2, 5, 8, 11, 16, 21, 3, 6, 9,
    13, 18, 22, 24, 26, 28, 30, 14, 19, 23, 25, 27, 29, 31, 12, 17,
]

X2_ULTRA_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": X2_ULTRA_ISAACLAB_JOINTS,
    # The motion_lib code uses these as numpy/torch gather indices:
    #   dof_il = dof_mj[mapping[i]]  →  mapping[i] must be the MJ source for IL position i
    # G1 follows this convention; X2 arrays are swapped to match.
    "isaaclab_to_mujoco_dof": X2_ULTRA_MUJOCO_TO_ISAACLAB_DOF,
    "mujoco_to_isaaclab_dof": X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF,
    "isaaclab_to_mujoco_body": X2_ULTRA_MUJOCO_TO_ISAACLAB_BODY,
    "mujoco_to_isaaclab_body": X2_ULTRA_ISAACLAB_TO_MUJOCO_BODY,
}


def make_x2_ultra_cfg(
    actuator_regime: Literal["implicit", "explicit"] = "implicit",
    frictionloss: float = 0.0,
    foot: Literal["mesh", "sphere"] = "mesh",
    ankle_kp_scale: float = 1.0,
) -> ArticulationCfg:
    """Build an X2 Ultra ``ArticulationCfg`` with optional MuJoCo-mirroring tweaks.

    All defaults reproduce the long-standing ``X2_ULTRA_CFG`` exactly so this
    factory is safe to use as a drop-in replacement. The non-default values are
    used by the ``isaaclab_mujoco_mirror`` diagnostic
    (``docs/source/user_guide/sim2sim_mujoco.md`` G18) to reproduce the
    MuJoCo deployment regime inside IsaacLab.

    Args:
        actuator_regime: ``"implicit"`` (default) keeps PD inside the PhysX
            implicit integrator (training-equivalent). ``"explicit"`` uses
            ``IdealPDActuatorCfg`` so PD runs as ``ctrl``-driven torque,
            mirroring MuJoCo's deploy regime (sim2sim_mujoco.md G5).
        frictionloss: Per-joint Coulomb friction in N.m. Default 0 (matches
            training). Set to 0.3 to mirror the MJCF ``frictionloss="0.3"``.
        foot: ``"mesh"`` (default) loads the standard URDF with mesh foot
            colliders. ``"sphere"`` loads ``x2_ultra_sphere_feet.urdf`` with
            12 sphere collisions per foot at the exact MJCF positions.
        ankle_kp_scale: Multiplier on ankle pitch/roll KP only. Default 1.0
            (training-equivalent). Set to 1.5 to mirror the deployed
            ``DEPLOYMENT_KP_SCALE["ankle"]`` baked into ``eval_x2_mujoco.py``
            (G16b).
    """

    if actuator_regime == "implicit":
        ActuatorCls = ImplicitActuatorCfg
    elif actuator_regime == "explicit":
        ActuatorCls = IdealPDActuatorCfg
    else:
        raise ValueError(f"Unknown actuator_regime={actuator_regime!r}")

    if foot == "mesh":
        urdf_name = "x2_ultra.urdf"
    elif foot == "sphere":
        urdf_name = "x2_ultra_sphere_feet.urdf"
    else:
        raise ValueError(f"Unknown foot={foot!r}")

    fric_kw = {"friction": frictionloss} if frictionloss > 0.0 else {}
    ankle_kp = STIFFNESS_ANKLE * float(ankle_kp_scale)

    return ArticulationCfg(
        spawn=sim_utils.UrdfFileCfg(
            fix_base=False,
            replace_cylinders_with_capsules=True,
            asset_path=f"{ASSET_DIR}/robot_description/urdf/x2_ultra/{urdf_name}",
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
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # X2 Ultra pelvis height is ~0.68m in MJCF default pose;
            # spawn slightly higher to avoid ground clipping with bent knees.
            pos=(0.0, 0.0, 0.78),
            joint_pos={
                ".*_hip_pitch_joint": -0.312,
                ".*_knee_joint": 0.669,
                ".*_ankle_pitch_joint": -0.363,
                ".*_elbow_joint": -0.6,
                "left_shoulder_roll_joint": 0.2,
                "left_shoulder_pitch_joint": 0.2,
                "right_shoulder_roll_joint": -0.2,
                "right_shoulder_pitch_joint": 0.2,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "legs": ActuatorCls(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                effort_limit_sim={
                    ".*_hip_yaw_joint": 120.0,
                    ".*_hip_roll_joint": 120.0,
                    ".*_hip_pitch_joint": 120.0,
                    ".*_knee_joint": 120.0,
                },
                velocity_limit_sim={
                    ".*_hip_yaw_joint": 11.936,
                    ".*_hip_roll_joint": 11.936,
                    ".*_hip_pitch_joint": 11.936,
                    ".*_knee_joint": 11.936,
                },
                stiffness={
                    ".*_hip_pitch_joint": STIFFNESS_HIP_KNEE,
                    ".*_hip_roll_joint": STIFFNESS_HIP_KNEE,
                    ".*_hip_yaw_joint": STIFFNESS_HIP_KNEE,
                    ".*_knee_joint": STIFFNESS_HIP_KNEE,
                },
                damping={
                    ".*_hip_pitch_joint": DAMPING_HIP_KNEE,
                    ".*_hip_roll_joint": DAMPING_HIP_KNEE,
                    ".*_hip_yaw_joint": DAMPING_HIP_KNEE,
                    ".*_knee_joint": DAMPING_HIP_KNEE,
                },
                armature={
                    ".*_hip_pitch_joint": ARMATURE_HIP_KNEE,
                    ".*_hip_roll_joint": ARMATURE_HIP_KNEE,
                    ".*_hip_yaw_joint": ARMATURE_HIP_KNEE,
                    ".*_knee_joint": ARMATURE_HIP_KNEE,
                },
                **fric_kw,
            ),
            "feet": ActuatorCls(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                effort_limit_sim={
                    ".*_ankle_pitch_joint": 36.0,
                    ".*_ankle_roll_joint": 24.0,
                },
                velocity_limit_sim={
                    ".*_ankle_pitch_joint": 13.088,
                    ".*_ankle_roll_joint": 15.077,
                },
                stiffness=ankle_kp,
                damping=DAMPING_ANKLE,
                armature=ARMATURE_ANKLE,
                **fric_kw,
            ),
            "waist_yaw": ActuatorCls(
                joint_names_expr=["waist_yaw_joint"],
                effort_limit_sim=120.0,
                velocity_limit_sim=11.936,
                stiffness=STIFFNESS_WAIST_YAW,
                damping=DAMPING_WAIST_YAW,
                armature=ARMATURE_WAIST_YAW,
                **fric_kw,
            ),
            "waist": ActuatorCls(
                joint_names_expr=["waist_pitch_joint", "waist_roll_joint"],
                effort_limit_sim=48.0,
                velocity_limit_sim=13.088,
                stiffness=STIFFNESS_WAIST_PR,
                damping=DAMPING_WAIST_PR,
                armature=ARMATURE_WAIST_PR,
                **fric_kw,
            ),
            "head": ActuatorCls(
                joint_names_expr=["head_yaw_joint", "head_pitch_joint"],
                effort_limit_sim={
                    "head_yaw_joint": 2.6,
                    "head_pitch_joint": 0.6,
                },
                velocity_limit_sim={
                    "head_yaw_joint": 6.019,
                    "head_pitch_joint": 6.28,
                },
                stiffness=STIFFNESS_HEAD,
                damping=DAMPING_HEAD,
                armature=ARMATURE_HEAD,
                **fric_kw,
            ),
            "arms": ActuatorCls(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_yaw_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_roll_joint",
                ],
                effort_limit_sim={
                    ".*_shoulder_pitch_joint": 36.0,
                    ".*_shoulder_roll_joint": 36.0,
                    ".*_shoulder_yaw_joint": 24.0,
                    ".*_elbow_joint": 24.0,
                    ".*_wrist_yaw_joint": 24.0,
                    ".*_wrist_pitch_joint": 4.8,
                    ".*_wrist_roll_joint": 4.8,
                },
                velocity_limit_sim={
                    ".*_shoulder_pitch_joint": 13.088,
                    ".*_shoulder_roll_joint": 13.088,
                    ".*_shoulder_yaw_joint": 15.077,
                    ".*_elbow_joint": 15.077,
                    ".*_wrist_yaw_joint": 15.077,
                    ".*_wrist_pitch_joint": 4.188,
                    ".*_wrist_roll_joint": 4.188,
                },
                stiffness={
                    ".*_shoulder_pitch_joint": STIFFNESS_SHOULDER_ELBOW,
                    ".*_shoulder_roll_joint": STIFFNESS_SHOULDER_ELBOW,
                    ".*_shoulder_yaw_joint": STIFFNESS_SHOULDER_ELBOW,
                    ".*_elbow_joint": STIFFNESS_SHOULDER_ELBOW,
                    ".*_wrist_yaw_joint": STIFFNESS_WRIST_YAW,
                    ".*_wrist_pitch_joint": STIFFNESS_WRIST,
                    ".*_wrist_roll_joint": STIFFNESS_WRIST,
                },
                damping={
                    ".*_shoulder_pitch_joint": DAMPING_SHOULDER_ELBOW,
                    ".*_shoulder_roll_joint": DAMPING_SHOULDER_ELBOW,
                    ".*_shoulder_yaw_joint": DAMPING_SHOULDER_ELBOW,
                    ".*_elbow_joint": DAMPING_SHOULDER_ELBOW,
                    ".*_wrist_yaw_joint": DAMPING_WRIST_YAW,
                    ".*_wrist_pitch_joint": DAMPING_WRIST,
                    ".*_wrist_roll_joint": DAMPING_WRIST,
                },
                armature={
                    ".*_shoulder_pitch_joint": ARMATURE_SHOULDER_ELBOW,
                    ".*_shoulder_roll_joint": ARMATURE_SHOULDER_ELBOW,
                    ".*_shoulder_yaw_joint": ARMATURE_SHOULDER_ELBOW,
                    ".*_elbow_joint": ARMATURE_SHOULDER_ELBOW,
                    ".*_wrist_yaw_joint": ARMATURE_WRIST_YAW,
                    ".*_wrist_pitch_joint": ARMATURE_WRIST,
                    ".*_wrist_roll_joint": ARMATURE_WRIST,
                },
                **fric_kw,
            ),
        },
    )


X2_ULTRA_CFG = make_x2_ultra_cfg()

# Action scale: effort_limit / stiffness * 0.25 (same formula as H2)
# Built from STIFFNESS_* constants directly — independent of ankle_kp_scale so
# the policy's [-1, 1] -> joint-target-offset mapping stays training-equivalent
# even when the deployed PD is bumped (mirrors eval_x2_mujoco.py G16b note).
X2_ULTRA_ACTION_SCALE = {}
for a in X2_ULTRA_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = dict.fromkeys(names, e)
    if not isinstance(s, dict):
        s = dict.fromkeys(names, s)
    for n in names:
        if n in e and n in s and s[n]:
            X2_ULTRA_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
