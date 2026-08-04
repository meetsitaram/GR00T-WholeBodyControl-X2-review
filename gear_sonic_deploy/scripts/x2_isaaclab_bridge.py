#!/usr/bin/env python3
"""
IsaacLab world bridge for the X2 deploy stack — the IsaacLab twin of
``x2_mujoco_ros_bridge.py``.

Modes
-----
standalone (default, M0 gate):
    No DDS, no deploy node. Loads a world (NuRec splat USDZ from
    x2-kitchen-sim + invisible flat GroundPlane at z=0), spawns the X2 from
    the canonical ``make_x2_ultra_cfg`` (sphere feet), ramps to the default
    pose while hanging, drops, and PD-holds. Prints a PASS/FAIL gate on
    pelvis height stability. Optionally captures a viewport screenshot.

--dds (M1, not yet implemented here):
    Speaks the MuJoCo bridge's exact contract: /aima/hal/joint/* +
    /aima/hal/imu/torso/state publishers, /aima/hal/joint/*/command
    subscribers, robot_pose ZMQ PUB :5570 — so ``deploy_x2.sh sim
    --world isaaclab`` swaps this in for the MuJoCo bridge with the C++
    deploy node unchanged.

Usage (standalone M0):
    conda activate env_isaaclab
    python gear_sonic_deploy/scripts/x2_isaaclab_bridge.py \
        --scene-usdz ~/projects/x2-kitchen-sim/assets/kitchen/kitchen_splat.usdz \
        --duration 15 --screenshot /tmp/m0_kitchen.png --headless --enable_cameras
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import sys
import threading
import time

print = functools.partial(print, flush=True)  # results must survive buffered redirects

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)  # gear_sonic asset paths are repo-relative

from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(description="X2 IsaacLab world bridge")
parser.add_argument("--scene-usdz", type=str, default=None,
                    help="NuRec splat USDZ (or any USD) world file; omit for plain ground")
parser.add_argument("--collision-usd", type=str, default=None,
                    help="collision mesh USD (default: <scene>_collision.usd next to the splat)")
parser.add_argument("--floor-z", type=float, default=0.0,
                    help="world floor height (x2-kitchen-sim exports put the floor at z=0)")
parser.add_argument("--spawn", type=float, nargs=4, default=[0.0, 0.0, 0.66, 0.0],
                    metavar=("X", "Y", "Z", "YAW_DEG"),
                    help="pelvis spawn pose relative to floor (default 0.66 = feet just touching "
                         "in the bent-knee default pose; 0.78 leaves feet dangling)")
parser.add_argument("--hold-kp-mult", type=float, default=20.0,
                    help="stiffness multiplier for the standalone PD hold. Training gains are "
                         "deliberately soft (SONIC stabilizes actively); naked PD cannot stand "
                         "statically at 1x (verified: collapses in 0.3 s; 20x holds with 0.2 cm "
                         "drift). Same idea as the MuJoCo bridge's --hold-stiffness-mult. "
                         "Ignored in --dds mode, which always runs true deploy gains.")
parser.add_argument("--foot", choices=["sphere", "mesh"], default="sphere")
parser.add_argument("--sim-dt", type=float, default=0.005, help="physics dt (200 Hz default)")
parser.add_argument("--hang-steps", type=int, default=300,
                    help="INIT steps with the root pinned while ramping to default pose")
parser.add_argument("--settle-steps", type=int, default=200,
                    help="steps after release before the stability gate starts scoring")
parser.add_argument("--duration", type=float, default=15.0, help="total seconds after release")
parser.add_argument("--daemon-in-proc", action="store_true",
                    help="With --pose-port: host the UNMODIFIED x2_kplanner "
                         "daemon on a sim-clocked thread in this process "
                         "(required for windowed viewing at RTF<1; the only "
                         "adaptation is a virtualized time module).")
parser.add_argument("--pose-port", type=int, default=None,
                    help="Direct-wire mode for --sonic-onnx: consume the external "
                         "x2_kplanner daemon's pose payload on this ZMQ port "
                         "(stack default 5556) as the reference stream — no "
                         "motion pkl, no motion lib. Publishes robot_pose:5570 "
                         "feedback and throttles to wall-clock 50 Hz (RTF≈1).")
parser.add_argument("--snap-dir", type=str, default=None,
                    help="Headless camera: save a PNG here every second and on "
                         "falls (both operator and agent can review them).")
parser.add_argument("--screenshot", type=str, default=None,
                    help="capture a viewport PNG here at the end (needs --enable_cameras)")
parser.add_argument("--shot-eye", type=float, nargs=3, default=None,
                    help="screenshot camera eye offset dx dy dz relative to spawn "
                    "(default 2.4 -2.4 1.2 — the proven kitchen viewpoint)")
parser.add_argument("--num-robots", type=int, default=1,
                    help="spawn N X2s at preset offsets around --spawn (max 8)")
parser.add_argument("--show-collision", action="store_true",
                    help="debug: make the collision mesh visible")
parser.add_argument("--no-paint", action="store_true",
                    help="skip applying MJCF-matched link colors to the visuals")
parser.add_argument("--sonic-onnx", type=str, default=None,
                    help="in-process SONIC mode: fused model_step_*_g1.onnx, stepped at 50 Hz "
                         "SIM time in lockstep (RTF-independent — viewer can stay open)")
parser.add_argument("--motion", type=str,
                    default="gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl",
                    help="reference motion PKL for --sonic-onnx (default: idle stand anchor)")
parser.add_argument("--dds", action="store_true",
                    help="M1 mode: IsaacLab world driven by the C++ deploy node via "
                         "x2_dds_zmq_adapter.py (state PUB :5581, cmd SUB :5582, "
                         "robot_pose PUB :5570). Single robot, deploy-exact torque PD.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.spawners.from_files import UsdFileCfg  # noqa: E402

from gear_sonic.envs.manager_env.robots.x2_ultra import make_x2_ultra_cfg  # noqa: E402

# Per-link visual colors lifted from the MJCF (gear_sonic/data/assets/robot_description/
# mjcf/x2_ultra.xml <geom class="visual"> rgba) so IsaacLab matches the MuJoCo look.
# The URDF's STL visual meshes carry no material at all -> ghost-white without this.
X2_LINK_COLORS = {
    "pelvis": (0.298, 0.298, 0.298), "waist_yaw_link": (0.298, 0.298, 0.298),
    "waist_pitch_link": (0.298, 0.298, 0.298), "torso_link": (1.0, 1.0, 1.0),
    "head_yaw_link": (0.867, 0.867, 0.89), "head_pitch_link": (0.2, 0.2, 0.2),
    "imu_in_torso_link": (0.647, 0.62, 0.588), "imu_in_head_link": (0.647, 0.62, 0.588),
    "RGBD_in_head_link": (0.647, 0.62, 0.588), "RGB_binocular_link": (0.267, 0.267, 0.267),
    "RGB_rear_link": (0.867, 0.867, 0.89), "RGB_Forward_link": (0.502, 0.502, 0.502),
}
for side in ("left", "right"):
    X2_LINK_COLORS.update({
        f"{side}_hip_pitch_link": (0.298, 0.298, 0.298),
        f"{side}_hip_roll_link": (1.0, 1.0, 1.0),
        f"{side}_hip_yaw_link": (1.0, 1.0, 1.0),
        f"{side}_knee_link": (1.0, 1.0, 1.0),
        f"{side}_ankle_pitch_link": (0.298, 0.298, 0.298),
        f"{side}_ankle_roll_link": (1.0, 0.5, 0.0),          # the orange feet
        f"{side}_shoulder_pitch_link": (0.298, 0.298, 0.298),
        f"{side}_shoulder_roll_link": (1.0, 1.0, 1.0),
        f"{side}_shoulder_yaw_link": (1.0, 1.0, 1.0),
        f"{side}_elbow_link": (1.0, 1.0, 1.0),
        f"{side}_wrist_yaw_link": (1.0, 1.0, 1.0),
        f"{side}_wrist_pitch_link": (0.3, 0.3, 0.3),
        f"{side}_wrist_roll_link": (0.898, 0.918, 0.929),
    })

# Extra spawn offsets (dx, dy, dyaw_deg) for --num-robots > 1.
# Kitchen positions are clearance-verified against kitchen_collision.obj
# (>=1.0 m to any geometry between 0.15-1.7 m height, >=0.75 m apart).
ROBOT_OFFSETS = [
    (0.18, -0.67, 0.0), (-0.18, -1.24, 40.0), (0.57, -0.07, 160.0),
    (-2.05, -0.35, -60.0), (0.92, 0.49, -100.0), (-0.25, 0.05, -160.0),
    (0.0, 0.0, 0.0), (1.80, 2.55, -100.0),
]


def paint_x2(stage, robot_root: str):
    """Bind MJCF-matched UsdPreviewSurface materials to each link's visuals."""
    from pxr import Gf, Sdf, UsdShade

    def ensure_mat(name, rgb):
        path = f"/World/Looks/x2_{name}"
        if stage.GetPrimAtPath(path):
            return UsdShade.Material(stage.GetPrimAtPath(path))
        mat = UsdShade.Material.Define(stage, path)
        sh = UsdShade.Shader.Define(stage, path + "/shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.05)
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    root = stage.GetPrimAtPath(robot_root)
    if not root:
        return 0
    painted = 0
    for link in root.GetChildren():
        rgb = X2_LINK_COLORS.get(link.GetName())
        if rgb is None:
            continue
        vis = stage.GetPrimAtPath(f"{link.GetPath()}/visuals")
        if not vis:
            continue
        api = UsdShade.MaterialBindingAPI.Apply(vis)
        api.Bind(ensure_mat(link.GetName(), rgb), UsdShade.Tokens.strongerThanDescendants)
        painted += 1
    return painted


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=args.sim_dt, render_interval=4)
    # Static triangle-mesh world + many sphere-feet robots: GPU contact buffers
    # overflow SILENTLY (no warning) and drop contacts for late articulations —
    # observed as one random robot free-falling. Oversize them.
    sim_cfg.physx.gpu_max_rigid_contact_count = 2 ** 24
    sim_cfg.physx.gpu_max_rigid_patch_count = 2 ** 19
    sim_cfg.physx.gpu_collision_stack_size = 2 ** 28
    # The decisive ones: with a 55k-tri static mesh + several articulations, the
    # LAST-spawned robot silently loses its contacts when aggregate-pair
    # capacities run out (observed: X2_5 always falls ~0.6 s post-release,
    # regardless of position; 6/6 stand without the mesh).
    sim_cfg.physx.gpu_found_lost_pairs_capacity = 2 ** 23
    sim_cfg.physx.gpu_found_lost_aggregate_pairs_capacity = 2 ** 26
    sim_cfg.physx.gpu_total_aggregate_pairs_capacity = 2 ** 24
    sim = SimulationContext(sim_cfg)

    # --- world ---------------------------------------------------------------
    ground = sim_utils.GroundPlaneCfg(
        size=(40.0, 40.0), visible=args.scene_usdz is None,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.5, dynamic_friction=1.5, restitution=0.0),
    )
    ground.func("/World/GroundPlane", ground, translation=(0.0, 0.0, args.floor_z))

    if args.scene_usdz:
        scene_path = os.path.expanduser(args.scene_usdz)
        if not os.path.exists(scene_path):
            sys.exit(f"[bridge] world file not found: {scene_path}")
        UsdFileCfg(usd_path=scene_path).func("/World/Splat", UsdFileCfg(usd_path=scene_path))
        print(f"[bridge] world: {scene_path}")
    else:
        print("[bridge] world: plain ground plane")

    dome = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.92, 0.92, 0.88))
    dome.func("/World/DomeLight", dome)
    dist = sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 1.0, 0.95), angle=0.53)
    dist.func("/World/DistantLight", dist)

    # --- robots --------------------------------------------------------------
    x, y, dz, yaw_deg = args.spawn
    spawn_z = args.floor_z + dz
    n_robots = max(1, min(args.num_robots, len(ROBOT_OFFSETS)))
    base_cfg = make_x2_ultra_cfg(foot=args.foot)
    robots, poses = [], []
    for i in range(n_robots):
        dx, dy, dyaw = ROBOT_OFFSETS[i]
        yaw_i = math.radians(yaw_deg + dyaw)
        cfg = base_cfg.replace(prim_path=f"/World/X2_{i}")
        cfg.init_state.pos = (x + dx, y + dy, spawn_z)
        cfg.init_state.rot = (math.cos(yaw_i / 2), 0.0, 0.0, math.sin(yaw_i / 2))
        robots.append(Articulation(cfg))
        poses.append((x + dx, y + dy, yaw_i))
        print(f"[bridge] X2_{i} spawned at ({x+dx:.2f}, {y+dy:.2f}, {spawn_z:.2f}) "
              f"yaw={yaw_deg+dyaw:.0f}")
    robot = robots[0]  # reference robot for gate telemetry

    # Collision mesh via IsaacLab's TerrainImporter — the supported path for
    # articulations on static trimesh (raw UsdFileCfg spawning of a collision
    # USD made one robot per run progressively sink through the floor, on both
    # CPU and GPU PhysX, regardless of buffers/order/tiling).
    # NOTE: auto-loading disabled — pass --collision-usd explicitly to opt in.
    # OPEN ISSUE (see x2-kitchen-sim docs/issues.md): with the static collision
    # mesh present and >=2 robots, exactly one robot per run progressively sinks
    # through the floor ~0.6 s after release. Reproduced on CPU AND GPU PhysX,
    # with raw UsdFileCfg and TerrainImporter, any spawn order, tiled or
    # monolithic mesh, degenerate faces scrubbed, all GPU buffers oversized,
    # robot-robot collisions filtered. Single robot + mesh is stable. Deferred;
    # revisit with a minimal repro (2 robots + 1 m box) before the RL phase.
    if args.scene_usdz and args.collision_usd:
        col_path = args.collision_usd
        if os.path.exists(os.path.expanduser(col_path)):
            col_path = os.path.expanduser(col_path)
            from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
            terrain_cfg = TerrainImporterCfg(
                prim_path="/World/Collision",
                terrain_type="usd",
                usd_path=col_path,
                collision_group=-1,
                env_spacing=0.0,  # single shared world; origins unused
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            )
            importer = TerrainImporter(terrain_cfg)
            importer.configure_env_origins = lambda *a, **k: None  # not cloning envs
            print(f"[bridge] collision mesh via TerrainImporter: {col_path}")
            if args.show_collision:
                import omni.usd
                from pxr import Usd, UsdGeom
                _st = omni.usd.get_context().get_stage()
                for _p in Usd.PrimRange(_st.GetPrimAtPath("/World/Collision")):
                    if _p.IsA(UsdGeom.Imageable):
                        UsdGeom.Imageable(_p).MakeVisible()
        else:
            print("[bridge] collision mesh: none (GroundPlane only — walls won't collide)")

    if not args.no_paint:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        total = sum(paint_x2(stage, f"/World/X2_{i}") for i in range(n_robots))
        print(f"[bridge] painted {total} links with MJCF colors")

    if n_robots > 1 and os.environ.get("X2_FILTER_ROBOT_COLLISION"):  # OFF by default: this CollisionGroup approach also kills robot-GROUND contacts (issues.md #9)
        # Robots pass through each other (user decision — simplicity this phase)
        # but still collide with the world: one collision group holding all
        # robots, filtered against itself.
        import omni.usd
        from pxr import UsdPhysics
        stage = omni.usd.get_context().get_stage()
        cg = UsdPhysics.CollisionGroup.Define(stage, "/World/RobotMutualFilter")
        cg.CreateFilteredGroupsRel().AddTarget("/World/RobotMutualFilter")
        col = cg.GetCollidersCollectionAPI()
        for i in range(n_robots):
            col.CreateIncludesRel().AddTarget(f"/World/X2_{i}")
        print(f"[bridge] robot-robot collisions filtered off ({n_robots} robots)")

    shot_cam = None
    if args.screenshot:
        from isaaclab.sensors import Camera, CameraCfg
        shot_cam = Camera(CameraCfg(
            prim_path="/World/ShotCam",
            width=1600, height=1000, data_types=["rgb"], update_period=0.0,
            spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.05, 100.0)),
        ))

    # EXACT proven viewpoint from the M0 hero shot — raising or pulling back
    # lands inside counter/cabinet splats (fog). Adjust only with a clearance
    # check against the collision mesh. --shot-eye dx dy dz overrides the
    # offset (relative to spawn) for worlds where this default sits in fuzz.
    _se = args.shot_eye if args.shot_eye else (2.4, -2.4, 1.2)
    cam_eye = (x + _se[0], y + _se[1], spawn_z + _se[2])
    sim.set_camera_view(eye=cam_eye, target=(x, y, spawn_z))
    sim.reset()
    for r in robots:
        r.update(args.sim_dt)

    stiff, damp = robot.data.joint_stiffness[0], robot.data.joint_damping[0]
    print(f"[bridge] joint stiffness: min {float(stiff.min()):.1f} max {float(stiff.max()):.1f} "
          f"mean {float(stiff.mean()):.1f} | damping mean {float(damp.mean()):.2f} | "
          f"zeros {(stiff == 0).sum().item()}/{stiff.numel()}")
    if args.hold_kp_mult != 1.0:
        for r in robots:
            r.write_joint_stiffness_to_sim(r.data.joint_stiffness * args.hold_kp_mult)
            r.write_joint_damping_to_sim(r.data.joint_damping * math.sqrt(args.hold_kp_mult))
        print(f"[bridge] standalone hold gains: kp x{args.hold_kp_mult}, kd x{math.sqrt(args.hold_kp_mult):.2f}")
        for i, r in enumerate(robots):
            eff = r.root_physx_view.get_dof_stiffnesses()
            print(f"[bridge]   X2_{i} physx stiffness readback: mean {float(eff.mean()):.1f} "
                  f"max {float(eff.max()):.1f}")

    default_pos = robot.data.default_joint_pos.clone()
    hang_poses = [torch.tensor([[px, py, spawn_z, math.cos(pyaw / 2), 0.0, 0.0, math.sin(pyaw / 2)]],
                               device=robot.device) for px, py, pyaw in poses]
    zero_vel = torch.zeros((1, 6), device=robot.device)

    # --- INIT: hang + ramp to default pose ----------------------------------
    start_pos = [r.data.joint_pos.clone() for r in robots]
    for i in range(args.hang_steps):
        alpha = min(1.0, i / max(args.hang_steps - 50, 1))
        for r, sp, hp in zip(robots, start_pos, hang_poses):
            r.write_root_pose_to_sim(hp)
            r.write_root_velocity_to_sim(zero_vel)
            r.set_joint_position_target(sp * (1.0 - alpha) + default_pos * alpha)
            r.write_data_to_sim()
        sim.step()
        for r in robots:
            r.update(args.sim_dt)
    err = (robot.data.joint_pos[0] - default_pos[0]).abs()
    print(f"[bridge] INIT done — joint tracking err: mean {float(err.mean()):.4f} rad, "
          f"max {float(err.max()):.4f} rad — releasing {n_robots} robot(s)")

    # --- release + PD hold ---------------------------------------------------
    n_steps = int(args.duration / args.sim_dt)
    heights, fell = [], False
    for i in range(n_steps):
        for r in robots:
            r.set_joint_position_target(default_pos)
            r.write_data_to_sim()
        sim.step()
        for r in robots:
            r.update(args.sim_dt)
        hs_now = [float(r.data.root_pos_w[0, 2]) for r in robots]
        h = hs_now[0]
        if i >= args.settle_steps:
            heights.append(h)
        low = min(hs_now)
        if low < args.floor_z + 0.35:
            fell = True
            who = int(np.argmin(hs_now))
            print(f"[bridge] FELL at t={i*args.sim_dt:.2f}s: X2_{who} at "
                  f"({poses[who][0]:.2f}, {poses[who][1]:.2f}), pelvis {low:.2f} m")
            break
        if i < 200 and i % 10 == 0:
            for j in (0, len(robots) - 1):
                r = robots[j]
                vz = float(r.data.root_lin_vel_w[0, 2])
                jerr = float((r.data.joint_pos[0] - default_pos[0]).abs().mean())
                print(f"[bridge] trace t={i*args.sim_dt:.2f} X2_{j}: h={hs_now[j]:.3f} "
                      f"vz={vz:+.2f} jerr={jerr:.3f}")
        if i % 400 == 0:
            print(f"[bridge] t={i*args.sim_dt:5.1f}s pelvis(ref)={h - args.floor_z:.3f} m "
                  f"lowest={low - args.floor_z:.3f} m")

    # --- gate ----------------------------------------------------------------
    ok, result = False, {"gate": "FAIL", "fell": fell}
    if heights and not fell:
        hs = np.array(heights) - args.floor_z
        drift = float(hs.max() - hs.min())
        nan_free = bool(torch.isfinite(robot.data.joint_pos).all())
        ok = 0.55 < float(hs.mean()) < 0.80 and drift < 0.08 and nan_free
        result.update(gate="PASS" if ok else "FAIL",
                      pelvis_mean_m=round(float(hs.mean()), 3),
                      pelvis_min_m=round(float(hs.min()), 3),
                      pelvis_max_m=round(float(hs.max()), 3),
                      drift_cm=round(drift * 100, 1), nan_free=nan_free)
        print(f"[bridge] pelvis height above floor: mean {hs.mean():.3f} m | "
              f"min {hs.min():.3f} | max {hs.max():.3f} | drift {drift*100:.1f} cm")
    print(f"[bridge] M0 GATE: {result['gate']}")

    # Persist the verdict BEFORE any shutdown path — close() can hang or kill us.
    result_path = (os.path.splitext(args.screenshot)[0] + "_result.json"
                   if args.screenshot else "/tmp/x2_bridge_m0_result.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[bridge] result: {result_path}")

    # --- screenshot (Camera sensor — deterministic, headless-safe) -----------
    if args.screenshot and shot_cam is not None:
        try:
            import torch as _t
            eye = _t.tensor([list(cam_eye)], device=robot.device)
            tgt = _t.tensor([[x, y, spawn_z - 0.1]], device=robot.device)
            shot_cam.set_world_poses_from_view(eye, tgt)
            for _ in range(60):  # let RTX accumulate samples
                sim.render()
            shot_cam.update(args.sim_dt)
            rgb = shot_cam.data.output["rgb"][0].cpu().numpy()
            from PIL import Image
            dest = os.path.abspath(args.screenshot)
            Image.fromarray(rgb[..., :3].astype(np.uint8)).save(dest)
            print(f"[bridge] screenshot: {dest} ok ({rgb.shape[1]}x{rgb.shape[0]})")
        except Exception as e:  # best-effort — the gate result stands regardless
            print(f"[bridge] screenshot failed: {e}")

    # close() is known to hang with cameras enabled — watchdog it, then hard-exit.
    closer = threading.Thread(target=simulation_app.close, daemon=True)
    closer.start()
    closer.join(timeout=20.0)
    if closer.is_alive():
        print("[bridge] close() hung — hard exit")
    sys.stdout.flush()
    os._exit(0 if ok else 2)


def dds_main():
    """M1 mode: IsaacLab world driven by the C++ deploy node via the
    x2_dds_zmq_adapter (which runs inside the deploy docker).

    Wire (all loopback ZMQ, msgpack):
      PUB :5581 "il_state"  {t, qpos[31], qvel[31], effort[31], quat_wxyz, angvel}
      SUB :5582 "il_cmd"    {t, pos, vel, ff, kp, kd}   (from adapter/deploy)
      PUB :5570 robot_pose  (pack_robot_pose — kplanner yaw re-anchor + eval tools)

    Torque path is deploy-exact: tau = ff + kp*(q*-q) + kd*(dq*-dq), clipped to
    the MJCF torque limits, applied at every physics step (500 Hz default) with
    actuator gains zeroed (explicit-torque regime, same as the MuJoCo bridge).
    Standby before the first deploy command: root hung + soft PD to default.
    """
    import importlib.util

    import msgpack
    import zmq

    # eval_x2_mujoco is the single source of truth for KP/KD/DEFAULT_DOF/limits
    spec = importlib.util.spec_from_file_location(
        "eval_x2", os.path.join(REPO_ROOT, "gear_sonic", "scripts", "eval_x2_mujoco.py"))
    eval_x2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_x2)
    from gear_sonic.envs.manager_env.robots.x2_ultra import (
        X2_ULTRA_ISAACLAB_TO_MUJOCO_DOF as IL2MJ,
        X2_ULTRA_MUJOCO_TO_ISAACLAB_DOF as MJ2IL,
    )
    from gear_sonic.utils.teleop.zmq.robot_pose_zmq import pack_robot_pose

    n = eval_x2.NUM_DOFS
    kp0 = np.asarray(eval_x2.KP, dtype=np.float64)
    kd0 = np.asarray(eval_x2.KD, dtype=np.float64)
    default_mj = np.asarray(eval_x2.DEFAULT_DOF, dtype=np.float64)
    # Torque limits from the MJCF actuator force ranges — same source as the
    # MuJoCo bridge (its self.tau_limit), so clipping is deploy-identical.
    import mujoco as _mj
    _mjm = _mj.MjModel.from_xml_path(str(eval_x2.MJCF_PATH))
    _fr = _mjm.actuator_forcerange[:n]
    tau_limit = np.maximum(np.abs(_fr[:, 0]), np.abs(_fr[:, 1]))
    tau_limit = np.where(tau_limit > 0.0, tau_limit, 1e6)

    sim_dt = args.sim_dt                                       # 200 Hz default
    sim_cfg = sim_utils.SimulationCfg(dt=sim_dt, render_interval=4)
    sim = SimulationContext(sim_cfg)

    ground = sim_utils.GroundPlaneCfg(size=(40.0, 40.0), visible=args.scene_usdz is None)
    ground.func("/World/GroundPlane", ground, translation=(0.0, 0.0, args.floor_z))
    if args.scene_usdz:
        scene_path = os.path.expanduser(args.scene_usdz)
        UsdFileCfg(usd_path=scene_path).func("/World/Splat", UsdFileCfg(usd_path=scene_path))
        print(f"[bridge:dds] world: {scene_path}")
        if args.collision_usd and os.path.exists(os.path.expanduser(args.collision_usd)):
            col = os.path.expanduser(args.collision_usd)
            UsdFileCfg(usd_path=col).func("/World/Collision", UsdFileCfg(usd_path=col))
            print(f"[bridge:dds] collision mesh: {col} (single robot — stable config)")
    dome = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.92, 0.92, 0.88))
    dome.func("/World/DomeLight", dome)

    x, y, dz, yaw_deg = args.spawn
    yaw = math.radians(yaw_deg)
    spawn_z = args.floor_z + dz
    cfg = make_x2_ultra_cfg(foot=args.foot).replace(prim_path="/World/X2")
    for act in cfg.actuators.values():           # explicit-torque regime
        act.stiffness = 0.0
        act.damping = 0.0
    cfg.init_state.pos = (x, y, spawn_z)
    cfg.init_state.rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    robot = Articulation(cfg)
    print(f"[bridge:dds] X2 spawned at ({x:.2f}, {y:.2f}, {spawn_z:.2f}) yaw={yaw_deg:.0f}")

    if not args.no_paint:
        import omni.usd
        paint_x2(omni.usd.get_context().get_stage(), "/World/X2")

    sim.set_camera_view(eye=(x + 2.4, y - 2.4, spawn_z + 1.2), target=(x, y, spawn_z))
    sim.reset()
    robot.update(sim_dt)
    dev = robot.device
    default_il = robot.data.default_joint_pos.clone()
    # MC-MATCH (deploy) gains: walking_recovery_loose.yaml documents that the
    # policy runs against MC-match stiffness on the robot — raw training gains
    # make it "sag at idle" (exactly the failure we saw). ev.KP/ev.KD carry the
    # deployment scaling; this is the first test of them WITH clean obs.
    kp_il = torch.tensor(kp[il2mj_g], dtype=torch.float32, device=dev).unsqueeze(0)
    kd_il = torch.tensor(kd[il2mj_g], dtype=torch.float32, device=dev).unsqueeze(0)
    robot.write_joint_stiffness_to_sim(kp_il)
    robot.write_joint_damping_to_sim(kd_il)
    print(f"[bridge:sonic] implicit PD with MC-match deploy gains: "
          f"kp mean {float(kp_il.mean()):.1f}, kd mean {float(kd_il.mean()):.2f}")

    il2mj = torch.tensor(IL2MJ, dtype=torch.long, device=dev)
    mj2il = torch.tensor(MJ2IL, dtype=torch.long, device=dev)
    tau_lim_t = torch.tensor(tau_limit, dtype=torch.float32, device=dev)

    # command state (MJ order), guarded by a lock against the SUB thread
    lock = threading.Lock()
    target_pos = default_mj.copy()
    target_vel = np.zeros(n)
    effort_ff = np.zeros(n)
    kp = kp0 * 2.0        # standby hold gains until the deploy takes over
    kd = kd0 * 2.0
    first_cmd = {"seen": False, "mono": 0.0}

    ctx = zmq.Context.instance()
    state_pub = ctx.socket(zmq.PUB)
    state_pub.bind("tcp://*:5581")
    pose_pub = ctx.socket(zmq.PUB)
    pose_pub.bind("tcp://*:5570")
    cmd_sub = ctx.socket(zmq.SUB)
    cmd_sub.connect("tcp://127.0.0.1:5582")
    cmd_sub.setsockopt(zmq.SUBSCRIBE, b"il_cmd")
    cmd_sub.setsockopt(zmq.RCVTIMEO, 100)

    stop = threading.Event()

    def cmd_thread():
        nonlocal target_pos, target_vel, effort_ff, kp, kd
        while not stop.is_set():
            try:
                _, payload = cmd_sub.recv_multipart()
            except zmq.Again:
                continue
            c = msgpack.unpackb(payload)
            with lock:
                target_pos = np.asarray(c["pos"], dtype=np.float64)
                target_vel = np.asarray(c["vel"], dtype=np.float64)
                effort_ff = np.asarray(c["ff"], dtype=np.float64)
                kp = np.asarray(c["kp"], dtype=np.float64)
                kd = np.asarray(c["kd"], dtype=np.float64)
                if not first_cmd["seen"]:
                    first_cmd["seen"] = True
                    first_cmd["mono"] = time.monotonic()
                    print("[bridge:dds] first deploy command — PD handed to the deploy")

    threading.Thread(target=cmd_thread, daemon=True).start()

    hang_pose = torch.tensor([[x, y, spawn_z, math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]],
                             device=dev)
    zero_vel = torch.zeros((1, 6), device=dev)
    band_release_s = 1.0
    step = 0
    rehang_until = 0.0
    loop_t0 = time.monotonic()
    print(f"[bridge:dds] loop up at {1/sim_dt:.0f} Hz — waiting for the deploy "
          f"(state PUB :5581, cmd SUB :5582, robot_pose PUB :5570)")
    try:
        while simulation_app.is_running():
            hanging = (not first_cmd["seen"]) or \
                (time.monotonic() - first_cmd["mono"] < band_release_s) or \
                (time.monotonic() < rehang_until)
            if hanging:
                # Kinematic freeze at the default pose: re-pinning only the root
                # while soft PD fights gravity pumps energy into the limbs
                # (violent pendulum flailing around the pelvis). Full state
                # freeze is visually calm and hands over cleanly.
                robot.write_root_pose_to_sim(hang_pose)
                robot.write_root_velocity_to_sim(zero_vel)
                robot.write_joint_state_to_sim(default_il, torch.zeros_like(default_il))

            q_il = robot.data.joint_pos[0]
            dq_il = robot.data.joint_vel[0]
            # Table convention is scatter (mj[IL2MJ[i]] = il[i]), so the
            # GATHER to MJ order uses MJ2IL, and the gather back uses IL2MJ.
            q_mj = q_il[mj2il].cpu().numpy()
            dq_mj = dq_il[mj2il].cpu().numpy()

            with lock:
                tp, tv, ff, k_p, k_d = (target_pos.copy(), target_vel.copy(),
                                        effort_ff.copy(), kp.copy(), kd.copy())
            tau_mj = ff + k_p * (tp - q_mj) + k_d * (tv - dq_mj)
            tau_il = torch.tensor(tau_mj, dtype=torch.float32, device=dev)[il2mj]
            tau_il = torch.clamp(tau_il, -tau_lim_t[il2mj], tau_lim_t[il2mj]).unsqueeze(0)
            robot.set_joint_effort_target(tau_il)
            robot.write_data_to_sim()
            sim.step()
            robot.update(sim_dt)

            root = robot.data.root_state_w[0]
            quat = root[3:7].cpu().numpy()          # wxyz
            angvel_b = robot.data.root_ang_vel_b[0].cpu().numpy()
            state_pub.send_multipart([b"il_state", msgpack.packb({
                "t": step * sim_dt,
                "qpos": q_mj.tolist(), "qvel": dq_mj.tolist(),
                "effort": tau_mj.tolist(),
                "quat_wxyz": quat.tolist(), "angvel": angvel_b.tolist(),
            })])
            if step % 10 == 0:                       # robot_pose at ~50 Hz
                pelvis = [float(v) for v in root[:3].cpu().numpy()] + \
                         [float(v) for v in quat]
                pose_pub.send(pack_robot_pose(step * sim_dt, pelvis))
            # Fall auto-reset: pelvis below threshold -> teleport to spawn and
            # freeze 2 s, then release again. NOTE the deploy's policy state
            # does NOT reset — restart the deploy for a clean re-engage.
            if not hanging and float(root[2]) < args.floor_z + 0.35:
                print(f"[bridge:dds] FELL (pelvis {float(root[2]):.2f} m) — auto-reset to spawn")
                rehang_until = time.monotonic() + 2.0

            # Real-time pacing: SONIC's obs history assumes wall-clock rates.
            # Sleep off any surplus; report the achieved real-time factor so a
            # too-slow sim is loud instead of a silent policy-breaker.
            deadline = loop_t0 + (step + 1) * sim_dt
            now = time.monotonic()
            if now < deadline:
                time.sleep(deadline - now)
            if step % 1000 == 0 and step > 0:
                h = float(root[2])
                mode = "HANG" if hanging else "DEPLOY"
                rtf = (step * sim_dt) / (time.monotonic() - loop_t0)
                print(f"[bridge:dds] t={step*sim_dt:7.1f}s {mode} pelvis={h:.3f} m RTF={rtf:.2f}")
                try:
                    # Dump the user's viewport camera so a hand-tuned view can
                    # be harvested and baked in as the default framing.
                    from omni.kit.viewport.utility import get_active_viewport
                    import omni.usd as _ou
                    _cam_path = get_active_viewport().camera_path
                    _cam = _ou.get_context().get_stage().GetPrimAtPath(_cam_path)
                    from pxr import UsdGeom as _ug
                    _m = _ug.Xformable(_cam).ComputeLocalToWorldTransform(0)
                    _eye = list(_m.ExtractTranslation())
                    with open("/tmp/x2_viewport_cam.json", "w") as _f:
                        json.dump({"eye": [round(v, 3) for v in _eye],
                                   "matrix": [[round(float(_m[i][j]), 5) for j in range(4)]
                                              for i in range(4)]}, _f)
                except Exception:
                    pass
                if rtf < 0.9:
                    print("[bridge:dds] WARNING: sim slower than real time — SONIC will misbehave "
                          "(reduce rendering: close viewport / raise render_interval)")
            step += 1
    except KeyboardInterrupt:
        print("[bridge:dds] stopped")
    finally:
        stop.set()
        sys.stdout.flush()
        os._exit(0)


def sonic_main():
    """In-process SONIC (lockstep, sim-time). Reference comes from a motion PKL
    (idle anchor = active standing); the kplanner slots in as the reference
    source in step 2. Starvation is impossible by construction: physics only
    steps after the reference frame + obs + action for that tick exist.
    Faithful port of eval_x2_mujoco_onnx.py's obs/action path.
    """
    import importlib.util

    import onnxruntime as ort

    spec = importlib.util.spec_from_file_location(
        "eval_x2", os.path.join(REPO_ROOT, "gear_sonic", "scripts", "eval_x2_mujoco.py"))
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)

    n = ev.NUM_DOFS
    default_mj = np.asarray(ev.DEFAULT_DOF, dtype=np.float64)
    kp = np.asarray(ev.KP, dtype=np.float64)
    kd = np.asarray(ev.KD, dtype=np.float64)
    act_scale = np.asarray(ev.ACTION_SCALE, dtype=np.float64)
    il2mj_g = np.asarray(ev.IL_TO_MJ_DOF)   # GATHER mj->il (eval convention)
    mj2il_g = np.asarray(ev.MJ_TO_IL_DOF)   # GATHER il->mj

    import mujoco as _mj
    _mjm = _mj.MjModel.from_xml_path(str(ev.MJCF_PATH))
    _fr = _mjm.actuator_forcerange[:n]
    tau_limit = np.maximum(np.abs(_fr[:, 0]), np.abs(_fr[:, 1]))
    tau_limit = np.where(tau_limit > 0.0, tau_limit, 1e6)

    wire = None
    if args.pose_port:
        # Direct-wire reference (user architecture, 2026-07-21): the external
        # x2_kplanner daemon streams pose payloads (current frame + 9 future
        # frames @0.1 s). First frame (idle anchor post-warmup) = RSI
        # reference; per tick the latest payload becomes a 10-frame @10fps
        # mini-clip fed UNCHANGED to ev.build_tokenizer_obs — the
        # deploy-verified obs path, zero translation layers.
        import zmq
        from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (
            unpack_message,
        )
        from gear_sonic.utils.teleop.zmq.robot_pose_zmq import pack_robot_pose

        zctx = zmq.Context.instance()
        wire_sub = zctx.socket(zmq.SUB)
        wire_sub.setsockopt(zmq.SUBSCRIBE, b"")
        wire_sub.setsockopt(zmq.RCVHWM, 100)
        wire_sub.connect(f"tcp://127.0.0.1:{args.pose_port}")
        fb_pub = zctx.socket(zmq.PUB)
        fb_pub.bind("tcp://127.0.0.1:5570")

        def _wire_recv(block_s=None):
            deadline = (time.monotonic() + block_s) if block_s else None
            latest = None
            while True:
                try:
                    parts = wire_sub.recv_multipart(flags=zmq.NOBLOCK)
                    try:
                        f = unpack_message(parts[-1],
                                           expected_topic="pose").fields
                    except Exception:
                        try:
                            f = unpack_message(parts[-1]).fields
                        except Exception:
                            continue
                    if "joint_pos_mj" in f:
                        latest = f
                except zmq.Again:
                    if latest is not None or deadline is None:
                        return latest
                    if time.monotonic() > deadline:
                        return None
                    time.sleep(0.02)

        def _wire_motion_dict(f):
            cur_j = np.asarray(f["joint_pos_mj"], np.float64).reshape(1, 31)
            cur_q = np.asarray(f["root_quat_xyzw"], np.float64).reshape(1, 4)
            xy = np.asarray(f.get("root_xy_world", [0.0, 0.0]),
                            np.float64).reshape(-1)[:2]
            z = float(np.asarray(f.get("root_z_world", [0.665]),
                                 np.float64).reshape(-1)[0])
            jf = f.get("joint_pos_mj_future")
            qf = f.get("root_quat_xyzw_future")
            if jf is not None:
                dof = np.concatenate(
                    [cur_j, np.asarray(jf, np.float64).reshape(-1, 31)], axis=0)
                rot = np.concatenate(
                    [cur_q, np.asarray(qf, np.float64).reshape(-1, 4)], axis=0)
            else:
                dof = np.tile(cur_j, (10, 1))
                rot = np.tile(cur_q, (10, 1))
            n = dof.shape[0]
            return {"wire": {
                "dof": dof,
                "root_rot": rot,
                "root_trans_offset": np.tile([xy[0], xy[1], z], (n, 1)),
                "fps": 10.0,   # frames 0.1 s apart == DT_FUTURE_REF
            }}

        if args.daemon_in_proc:
            # Host the unmodified daemon on a sim-clocked thread (proven
            # component from run_x2_kplanner_env2): monotonic/sleep are
            # virtualized; env loop advances the clock 0.02/policy tick.
            import importlib.util as _ilu
            import threading as _th
            from pathlib import Path as _P

            class _SimClock:
                def __init__(self):
                    self._t = 0.0
                    self._cv = _th.Condition()

                def monotonic(self):
                    with self._cv:
                        return self._t

                time = monotonic
                perf_counter = monotonic

                def sleep(self, dt):
                    with self._cv:
                        dl = self._t + max(0.0, float(dt))
                        while self._t < dl:
                            self._cv.wait(timeout=0.5)

                def advance(self, dt):
                    with self._cv:
                        self._t += float(dt)
                        self._cv.notify_all()

            sim_clock = _SimClock()

            class _TimeProxy:
                def __init__(self, real, clk):
                    self._r, self._c = real, clk

                def monotonic(self):
                    return self._c.monotonic()

                def time(self):
                    return self._c.monotonic()

                def perf_counter(self):
                    return self._c.monotonic()

                def sleep(self, dt):
                    self._c.sleep(dt)

                def __getattr__(self, n):
                    return getattr(self._r, n)

            import signal as _rs

            class _SigProxy:
                def signal(self, *a, **k):
                    return None

                def __getattr__(self, n):
                    return getattr(_rs, n)

            _spec = _ilu.spec_from_file_location(
                "x2_kplanner", os.path.join(REPO_ROOT, "gear_sonic",
                                            "scripts", "x2_kplanner.py"))
            _kp = _ilu.module_from_spec(_spec)
            sys.modules["x2_kplanner"] = _kp
            _mb = os.path.join(REPO_ROOT, "motionbricks")
            if _mb not in sys.path:
                sys.path.insert(0, _mb)
            _spec.loader.exec_module(_kp)
            _kp.time = _TimeProxy(time, sim_clock)
            _kp.signal = _SigProxy()
            _kw = dict(
                vqvae_ckpt=_P(f"{REPO_ROOT}/motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0500000.ckpt"),
                pose_ckpt=_P(f"{REPO_ROOT}/motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/model-step=0500000.ckpt"),
                root_ckpt=_P(f"{REPO_ROOT}/motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0315000.ckpt"),
                warmup_qpos_path=_P(f"{REPO_ROOT}/gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl"),
                pub_host="127.0.0.1", pub_port=args.pose_port,
                pid_file=_P("/tmp/claude-1000/kplanner_bridge_daemon.pid"),
                demo_yaml=None, enable_keyboard=False,
                zmq_cmd_host="127.0.0.1", zmq_cmd_port=6563,
                zmq_cmd_topic="planner_cmd",
                duration_s=0.0, hand_dof=10, verbose=True,
                warmup_quiet_stand_s=2.0, body_pose_port=None, device="cuda",
                replan_threshold_frames=2, planner_mode=None,
                pose_feedback_host="127.0.0.1", pose_feedback_port=5570,
                pose_reseed_scope="none",
                ref_smoother_ms=300.0, ref_smoother_trigger_rad=0.05,
                ref_smoother_shape="halfcos",
                ref_smoother_joints="lower_body",
            )
            # planner_cmd forwarder: pad bridge PUB-connects :5563 and the
            # daemon SUB-connects — someone must bind both ends. SUB-bind
            # 5563 (pad side) -> PUB-bind 6563 (daemon side), wall-clock
            # (commands are real user input at wall rate).
            def _cmd_fwd():
                import zmq as _z
                c = _z.Context.instance()
                s_in = c.socket(_z.SUB)
                s_in.setsockopt(_z.SUBSCRIBE, b"")
                s_in.bind("tcp://127.0.0.1:5563")
                s_out = c.socket(_z.PUB)
                s_out.bind("tcp://127.0.0.1:6563")
                while True:
                    try:
                        s_out.send_multipart(s_in.recv_multipart())
                    except Exception:
                        break

            _th.Thread(target=_cmd_fwd, name="cmd_fwd", daemon=True).start()
            _th.Thread(target=lambda: _kp.run(**_kw),
                       name="x2_kplanner_daemon", daemon=True).start()
            print("[bridge:sonic] sim-clocked x2_kplanner daemon thread "
                  "started (pose:5556, pad->5563->fwd->6563, fb:5570)",
                  flush=True)
        else:
            sim_clock = None

        print(f"[bridge:sonic] DIRECT WIRE mode: waiting for daemon pose "
              f"payload on :{args.pose_port} ...", flush=True)
        # While waiting for the first payload, keep the daemon's virtual
        # clock flowing (its ckpt-load + quiet-stand warmup need time to
        # pass; the sim loop isn't advancing it yet) — else deadlock.
        first = None
        _dl = time.monotonic() + 180.0
        while first is None and time.monotonic() < _dl:
            if sim_clock is not None:
                sim_clock.advance(0.05)
            first = _wire_recv()
            if first is None:
                time.sleep(0.02)
        if first is None:
            print("[bridge:sonic] FATAL: no pose payload in 120 s — is "
                  "x2_kplanner running?", flush=True)
            simulation_app.close()
            return
        motion_data = _wire_motion_dict(first)
        wire = (fb_pub, _wire_recv, _wire_motion_dict, pack_robot_pose)
        motion_path = f"<pose wire :{args.pose_port}>"
    elif args.motion == "default":
        # Synthetic reference: hold the URDF default pose — the one configuration
        # proven collision-clean in this sim (M0 statues never exploded). Used to
        # isolate reference-pose problems (e.g. a teleop-derived anchor whose limbs
        # interpenetrate our collider set under enabled_self_collisions).
        N = 300
        motion_data = {"synthetic_default": {
            "dof": np.tile(np.asarray(ev.DEFAULT_DOF, dtype=np.float64), (N, 1)),
            "root_trans_offset": np.tile(np.array([0.0, 0.0, 0.68]), (N, 1)),
            "root_rot": np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (N, 1)),
            "fps": 30.0,
        }}
        motion_path = "<synthetic default pose>"
    else:
        motion_path = os.path.join(REPO_ROOT, args.motion) if not os.path.isabs(args.motion) else args.motion
        import joblib
        motion_data = joblib.load(motion_path)
    motion_fps = ev.get_motion_fps(motion_data)
    motion_frames = ev.get_total_frames(motion_data)
    motion_dur = motion_frames / motion_fps
    print(f"[bridge:sonic] motion: {os.path.basename(motion_path)} "
          f"({motion_frames} frames @ {motion_fps:.0f} fps, {motion_dur:.1f}s loop)")

    sess = ort.InferenceSession(os.path.expanduser(args.sonic_onnx),
                                providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"[bridge:sonic] ONNX: {args.sonic_onnx} (CPU)")

    # --- world + robot (explicit torque, deploy gains) -----------------------
    SIM_DT = 0.005                 # TRAINING timing: sim_dt 0.005, decimation 4
    POLICY_EVERY = 4               # 50 Hz policy in sim time
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, render_interval=POLICY_EVERY)
    sim = SimulationContext(sim_cfg)
    ground = sim_utils.GroundPlaneCfg(size=(40.0, 40.0), visible=args.scene_usdz is None)
    ground.func("/World/GroundPlane", ground, translation=(0.0, 0.0, args.floor_z))
    if args.scene_usdz:
        sp = os.path.expanduser(args.scene_usdz)
        UsdFileCfg(usd_path=sp).func("/World/Splat", UsdFileCfg(usd_path=sp))
        print(f"[bridge:sonic] world: {sp}")
        if args.collision_usd and os.path.exists(os.path.expanduser(args.collision_usd)):
            cp = os.path.expanduser(args.collision_usd)
            UsdFileCfg(usd_path=cp).func("/World/Collision", UsdFileCfg(usd_path=cp))
            print(f"[bridge:sonic] collision mesh: {cp}")
    dome = sim_utils.DomeLightCfg(intensity=1200.0, color=(0.92, 0.92, 0.88))
    dome.func("/World/DomeLight", dome)

    x, y, dz, yaw_deg = args.spawn
    yaw = math.radians(yaw_deg)
    spawn_z = args.floor_z + dz
    cfg = make_x2_ultra_cfg(foot=args.foot).replace(prim_path="/World/X2")
    # IMPLICIT PD (training-faithful): SONIC was trained against IsaacLab's
    # implicit actuators + position targets. The explicit-torque port produced
    # constraint yanks and wrapped velocities (exact 2*pi/pi artifacts) here.
    # Self-collisions OFF: our converted colliders interpenetrate slightly at
    # the default pose, injecting constant impulses (7 rad/s phantom arm
    # velocities on an airborne, torque-free robot). Statues masked it with
    # 20x gains; SONIC amplifies the obs noise into a fall. Parity with
    # training's collider set to be restored later.
    # Training parity: x2_ultra.py sets enabled_self_collisions=True and the
    # ground-truth eval stands with it. (The phantom-impulse episode that led
    # to disabling this happened under the removed explicit-torque mode.)
    cfg.init_state.pos = (x, y, spawn_z)
    cfg.init_state.rot = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    robot = Articulation(cfg)
    if not args.no_paint:
        import omni.usd
        paint_x2(omni.usd.get_context().get_stage(), "/World/X2")

    cam_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "..", "..", "user_cam_pose.json")
    sim.set_camera_view(eye=(x + 2.4, y - 2.4, spawn_z + 1.2), target=(x, y, spawn_z))
    sim.reset()
    robot.update(SIM_DT)
    dev = robot.device
    default_il = robot.data.default_joint_pos.clone()

    # RSI: start (and reset) EXACTLY in the reference's frame-0 state, like the
    # eval scripts and the real ritual start (idle anchor == actual stand pose).
    prop_buf = ev.ProprioceptionBuffer()
    last_action_mj = np.zeros(n, dtype=np.float32)
    motion_time = 0.0
    target_mj = default_mj.copy()

    def full_reset():
        nonlocal motion_time, last_action_mj, target_mj, prop_buf
        st = ev.compute_motion_state(motion_data, 0, motion_fps)
        rq = np.asarray(st["root_quat_w_wxyz"], dtype=np.float64)
        rz = float(np.asarray(st["root_pos_w"], dtype=np.float64)[2])
        jp_mj = np.asarray(st["joint_pos_mj"], dtype=np.float64)
        jv_mj = np.asarray(st["joint_vel_mj"], dtype=np.float64)
        # Ground-snap RSI: the anchor's recorded pelvis height doesn't match this
        # world (near-straight legs at z=0.665 put the feet ~10 cm underground ->
        # PhysX depenetration explosion: jvel +/-15 rad/s within 20 ms, pelvis
        # slammed down 8x faster than gravity). Place high, measure the lowest
        # body point, then re-place so the feet just kiss the floor.
        jp_il = torch.tensor(jp_mj[il2mj_g], dtype=torch.float32, device=dev).unsqueeze(0)
        jv_il = torch.tensor(jv_mj[il2mj_g], dtype=torch.float32, device=dev).unsqueeze(0)
        probe_z = args.floor_z + 1.2
        probe = torch.tensor([[x, y, probe_z, rq[0], rq[1], rq[2], rq[3]]],
                             device=dev, dtype=torch.float32)
        robot.write_root_pose_to_sim(probe)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        robot.write_joint_state_to_sim(jp_il, jv_il)
        # PD target must follow the RSI pose BEFORE any step: the actuators
        # still hold the pre-reset target, and a reference pose 1 rad away
        # (idle_stand arms) gets yanked to ~14 rad/s during the probe/settle
        # steps — the policy then wakes to an off-distribution obs and falls.
        robot.set_joint_position_target(jp_il)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        min_body_z = float(robot.data.body_pos_w[0, :, 2].min())
        # body_pos_w is LINK ORIGINS — the lowest is the ankle, whose foot
        # spheres extend ~0.07 m further down. Snap so the sphere bottoms
        # (not the ankle origin) clear the floor, with a small hover margin;
        # a few-mm drop onto the plane is gentle, an embedded foot explodes.
        FOOT_BELOW_ANKLE = 0.075
        z_root = probe_z - (min_body_z - args.floor_z) + FOOT_BELOW_ANKLE + 0.02
        pose = torch.tensor([[x, y, z_root, rq[0], rq[1], rq[2], rq[3]]],
                            device=dev, dtype=torch.float32)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=dev))
        robot.write_joint_state_to_sim(jp_il, jv_il)
        # No settle: at 1x training gains naked PD cannot hold through a settle
        # (robot starts falling before the policy wakes). Training starts the
        # policy INSTANTLY on the written state — do the same.
        prop_buf = ev.ProprioceptionBuffer()
        last_action_mj = np.zeros(n, dtype=np.float32)
        motion_time = 0.0
        target_mj = jp_mj.copy()

    full_reset()
    for _ in range(2):     # settle physics buffers on the RSI state
        sim.step()
        robot.update(SIM_DT)
    step = 0
    falls = 0
    t0 = time.monotonic()
    print("[bridge:sonic] lockstep loop up — RSI-initialized, 1 kHz PD, "
          "SONIC at 50 Hz sim time, RTF-independent")
    try:
        while simulation_app.is_running():
            q_il = robot.data.joint_pos[0].cpu().numpy()
            dq_il = robot.data.joint_vel[0].cpu().numpy()
            q_mj = q_il[mj2il_g]
            dq_mj = dq_il[mj2il_g]
            root = robot.data.root_state_w[0].cpu().numpy()
            quat = root[3:7]
            angvel_b = robot.data.root_ang_vel_b[0].cpu().numpy()

            if step % POLICY_EVERY == 0:           # 50 Hz policy, sim time
                gravity = ev.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
                jpos_rel_il = (q_mj - default_mj)[il2mj_g]
                jvel_il = dq_mj[il2mj_g]
                action_il_prev = last_action_mj[il2mj_g]
                prop_buf.append(gravity, angvel_b, jpos_rel_il, jvel_il, action_il_prev)
                if wire is not None:
                    fb_pub, _wire_recv, _wire_motion_dict, pack_robot_pose = wire
                    if sim_clock is not None:
                        sim_clock.advance(0.02)   # daemon runs on sim time
                    # publish robot pose feedback (daemon yaw refresh/reseed)
                    fb_pub.send(pack_robot_pose(step * SIM_DT, [
                        float(root[0]), float(root[1]), float(root[2]),
                        float(quat[0]), float(quat[1]), float(quat[2]),
                        float(quat[3])]))
                    latest = _wire_recv()
                    if latest is not None:
                        motion_data = _wire_motion_dict(latest)
                    motion_time = 0.0          # current frame is clip frame 0
                elif args.motion == "default":
                    # Idle yaw re-anchor (mirrors the PC2 kplanner idle resync):
                    # the stand reference follows the robot's current heading so
                    # yaw drift never becomes a phantom orientation error.
                    yaw_r = math.atan2(2.0 * (quat[0] * quat[3] + quat[1] * quat[2]),
                                       1.0 - 2.0 * (quat[2] ** 2 + quat[3] ** 2))
                    mkey = list(motion_data.keys())[0]
                    motion_data[mkey]["root_rot"][:] = np.array(
                        [0.0, 0.0, math.sin(yaw_r / 2), math.cos(yaw_r / 2)])  # xyzw
                tok = ev.build_tokenizer_obs(motion_data, motion_time % motion_dur,
                                             quat, motion_fps)
                obs = np.concatenate([tok.astype(np.float32),
                                      prop_buf.get_flat().astype(np.float32)])[None]
                action_il = sess.run([out_name], {in_name: obs})[0][0]
                if step < POLICY_EVERY * 3:
                    print(f"[bridge:sonic] TERMS tick{step//POLICY_EVERY}: "
                          f"angvel|max|={np.abs(angvel_b).max():.2f} "
                          f"jpos_rel|max|={np.abs(jpos_rel_il).max():.2f} "
                          f"jvel|max|={np.abs(jvel_il).max():.2f} "
                          f"jvel_argmax={int(np.abs(jvel_il).argmax())} "
                          f"lastact|max|={np.abs(action_il_prev).max():.2f}")
                    print(f"[bridge:sonic] DBG tick{step//POLICY_EVERY}: "
                          f"tok[{tok.min():+.2f},{tok.max():+.2f}] "
                          f"prop[{prop_buf.get_flat().min():+.2f},{prop_buf.get_flat().max():+.2f}] "
                          f"|act|max={np.abs(action_il).max():.2f} "
                          f"grav={np.round(gravity,2).tolist()} h={float(robot.data.root_pos_w[0,2]):.3f}")
                action_il = np.clip(action_il, -20.0, 20.0)
                action_mj = action_il[mj2il_g]
                last_action_mj = action_mj.astype(np.float32)
                # Training semantics: JointPositionActionCfg(use_default_offset,
                # scale=1.0) with the cfg's implicit-PD gains. ev.ACTION_SCALE
                # belongs to the deploy pair (MC-match KP/KD on the real MC) —
                # applying it here under training gains under-actuates every
                # joint 1.2-100x (ankle roll 0.009). Verified against
                # dump_isaaclab_step0: obs path exact, so this was the gap.
                target_mj = default_mj + action_mj
                motion_time += 0.02

            target_il = torch.tensor(target_mj[il2mj_g], dtype=torch.float32,
                                     device=dev).unsqueeze(0)
            robot.set_joint_position_target(target_il)
            robot.write_data_to_sim()
            sim.step()
            robot.update(SIM_DT)

            h = float(robot.data.root_pos_w[0, 2])
            grav_z = float(ev.quat_rotate_inverse(
                robot.data.root_state_w[0, 3:7].cpu().numpy(),
                np.array([0.0, 0.0, -1.0]))[2])
            if h < args.floor_z + 0.35 or grav_z > -0.3:
                falls += 1
                print(f"[bridge:sonic] FELL #{falls} (pelvis {h:.2f} m, grav_z {grav_z:.2f}) "
                      f"at t={step*SIM_DT:.1f}s — full reset")
                full_reset()
            if step % 10000 == 0 and step > 0:
                rtf = (step * SIM_DT) / (time.monotonic() - t0)
                print(f"[bridge:sonic] t={step*SIM_DT:7.1f}s pelvis={h:.3f} m "
                      f"falls={falls} RTF={rtf:.2f} (RTF is cosmetic in lockstep)")
            step += 1
    except KeyboardInterrupt:
        print("[bridge:sonic] stopped")
    finally:
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    if args.sonic_onnx:
        sonic_main()
    elif args.dds:
        dds_main()
    else:
        main()
