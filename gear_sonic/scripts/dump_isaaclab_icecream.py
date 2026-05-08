#!/usr/bin/env python3
"""Dump per-step IsaacLab state for the icecream motion.

Wraps ``gear_sonic.eval_agent_trl`` similar to ``dump_isaaclab_step0.py`` but:

  * Forces the motion library to load only the icecream motion (via
    ``filter_motion_keys``) so num_envs=1 always plays it.
  * Starts at frame 0 (RSI to motion start).
  * Wraps the inner env's ``step`` to capture, every control step, the same
    fields the MuJoCo diagnostic dumps:
        - sim_time, motion_time
        - root pos / quat (wxyz) / lin_vel_w / ang_vel_w
        - joint_pos, joint_vel (IsaacLab DOF order)
        - per-foot link z (left/right ankle_roll)
        - per-body net contact force from ContactSensor
        - per-foot floor contact force (z component, sum)
        - applied joint torque (from articulation.data.applied_torque)
        - whether the env terminated this step
  * After ``DUMP_STEPS`` control steps (default 150 = 3 s @ 50 Hz), writes a
    JSON file and exits.

Static config captured once at the start:
  * sim.dt, sim.render_interval, sim.gravity
  * physx solver type, position/velocity iteration counts
  * terrain physics_material (static/dynamic friction, restitution, combine)
  * per-body friction values (DR-sampled at startup, before first reset)
  * articulation joint armature/damping/stiffness/effort/velocity limits

Usage:
    python gear_sonic/scripts/dump_isaaclab_icecream.py \\
        +checkpoint=/path/model_step_002000.pt \\
        +headless=True ++num_envs=1 ++run_eval_loop=true \\
        ++max_render_steps=200 \\
        ++manager_env.commands.motion.motion_lib_cfg.filter_motion_keys="[standing__eat_icecream_fall_standing_R_001__A456_M]" \\
        ++manager_env.commands.motion.start_from_first_frame=true \\
        ++dump_path=/tmp/x2_icecream_isaaclab.json \\
        ++dump_steps=150
"""

import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


# ---------------- CLI scrape (before Hydra owns sys.argv) ----------------

_DUMP_PATH = "/tmp/x2_icecream_isaaclab.json"
_DUMP_STEPS = 150
_remaining = []
for arg in sys.argv:
    if arg.startswith("++dump_path=") or arg.startswith("dump_path="):
        _DUMP_PATH = arg.split("=", 1)[1]
    elif arg.startswith("++dump_steps=") or arg.startswith("dump_steps="):
        _DUMP_STEPS = int(arg.split("=", 1)[1])
    else:
        _remaining.append(arg)
sys.argv = _remaining
print(f"[dump_isaaclab_icecream] DUMP_PATH={_DUMP_PATH}  DUMP_STEPS={_DUMP_STEPS}", flush=True)


_RECORDS = []
_STATIC = {"captured": False}
_STEP_COUNT = {"n": 0}


def _tensor_to_list(t):
    """Detach and convert torch tensor to nested python list (env 0 only)."""
    import torch
    if isinstance(t, torch.Tensor):
        if t.dim() == 0:
            return float(t.item())
        return t.detach().cpu().float().numpy().tolist()
    if isinstance(t, (list, tuple)):
        return list(t)
    if hasattr(t, "tolist"):
        return t.tolist()
    return t


def _capture_static_config(env):
    import torch  # noqa: F401
    out = {}
    inner = env.env if hasattr(env, "env") else env
    sim = inner.sim
    cfg = sim.cfg
    out["sim_dt"] = float(getattr(cfg, "dt", 0.0))
    out["render_interval"] = int(getattr(cfg, "render_interval", 0))
    out["gravity"] = list(map(float, getattr(cfg, "gravity", (0, 0, -9.81))))
    out["device"] = str(getattr(cfg, "device", "?"))
    physx = getattr(cfg, "physx", None)
    if physx is not None:
        out["physx"] = {
            "solver_type": int(getattr(physx, "solver_type", -1)),
            "min_position_iteration_count": int(
                getattr(physx, "min_position_iteration_count", -1)
            ),
            "max_position_iteration_count": int(
                getattr(physx, "max_position_iteration_count", -1)
            ),
            "min_velocity_iteration_count": int(
                getattr(physx, "min_velocity_iteration_count", -1)
            ),
            "max_velocity_iteration_count": int(
                getattr(physx, "max_velocity_iteration_count", -1)
            ),
            "enable_ccd": bool(getattr(physx, "enable_ccd", False)),
            "enable_stabilization": bool(getattr(physx, "enable_stabilization", True)),
            "bounce_threshold_velocity": float(
                getattr(physx, "bounce_threshold_velocity", 0.0)
            ),
            "friction_offset_threshold": float(
                getattr(physx, "friction_offset_threshold", 0.0)
            ),
            "friction_correlation_distance": float(
                getattr(physx, "friction_correlation_distance", 0.0)
            ),
        }
    pm = getattr(cfg, "physics_material", None)
    if pm is not None:
        out["physics_material"] = {
            "static_friction": float(getattr(pm, "static_friction", -1)),
            "dynamic_friction": float(getattr(pm, "dynamic_friction", -1)),
            "restitution": float(getattr(pm, "restitution", -1)),
            "friction_combine_mode": str(getattr(pm, "friction_combine_mode", "?")),
            "restitution_combine_mode": str(getattr(pm, "restitution_combine_mode", "?")),
        }

    scene = inner.scene
    robot = scene["robot"]
    out["robot"] = {
        "joint_names": list(robot.data.joint_names),
        "body_names": list(robot.data.body_names),
        "default_joint_pos": _tensor_to_list(robot.data.default_joint_pos[0]),
        "joint_armature": _tensor_to_list(robot.data.joint_armature[0]),
        "joint_damping": _tensor_to_list(robot.data.joint_damping[0]),
        "joint_stiffness": _tensor_to_list(robot.data.joint_stiffness[0]),
        "joint_friction_coeff": _tensor_to_list(
            getattr(robot.data, "joint_friction_coeff", robot.data.joint_armature)[0]
        ),
        "default_joint_limits": _tensor_to_list(robot.data.default_joint_limits[0]),
    }

    # Per-body friction (after material DR randomization)
    try:
        from isaacsim.core.utils.types import (  # noqa: F401
            ArticulationActions,
        )
    except Exception:  # noqa: BLE001
        pass
    try:
        # IsaacLab Articulation exposes physics_view; per-body materials
        # live there as (num_bodies, num_materials, 3) → static/dynamic/restitution
        view = robot.root_physx_view
        materials = view.get_material_properties()
        out["robot"]["body_material_shape"] = list(materials.shape)
        out["robot"]["body_material_env0"] = _tensor_to_list(materials[0])
    except Exception as exc:  # noqa: BLE001
        out["robot"]["body_material_error"] = str(exc)

    # Contact sensor info
    try:
        cs = scene["contact_forces"]
        out["contact_sensor"] = {
            "body_names": list(cs.body_names),
            "force_threshold": float(getattr(cs.cfg, "force_threshold", 0)),
            "track_air_time": bool(getattr(cs.cfg, "track_air_time", False)),
            "history_length": int(getattr(cs.cfg, "history_length", 0)),
        }
    except Exception as exc:  # noqa: BLE001
        out["contact_sensor_error"] = str(exc)

    # Terrain info
    try:
        terrain = scene["terrain"]
        out["terrain"] = {
            "type": str(type(terrain).__name__),
            "env_origins": _tensor_to_list(terrain.env_origins[:1])
            if hasattr(terrain, "env_origins") else None,
        }
    except Exception as exc:  # noqa: BLE001
        out["terrain_error"] = str(exc)
    return out


def _capture_step_record(env, sim_time):
    inner = env.env if hasattr(env, "env") else env
    scene = inner.scene
    robot = scene["robot"]

    rec = {
        "step": int(_STEP_COUNT["n"]),
        "sim_time": float(sim_time),
        "joint_pos": _tensor_to_list(robot.data.joint_pos[0]),
        "joint_vel": _tensor_to_list(robot.data.joint_vel[0]),
        "applied_torque": _tensor_to_list(robot.data.applied_torque[0]),
        "computed_torque": _tensor_to_list(robot.data.computed_torque[0])
        if hasattr(robot.data, "computed_torque") else None,
        "root_pos_w": _tensor_to_list(robot.data.root_pos_w[0]),
        "root_quat_w_wxyz": _tensor_to_list(robot.data.root_quat_w[0]),
        "root_lin_vel_w": _tensor_to_list(robot.data.root_lin_vel_w[0]),
        "root_ang_vel_w": _tensor_to_list(robot.data.root_ang_vel_w[0]),
        "root_ang_vel_b": _tensor_to_list(robot.data.root_ang_vel_b[0]),
        "projected_gravity_b": _tensor_to_list(
            robot.data.projected_gravity_b[0]
        ),
    }

    # Foot heights
    body_names = list(robot.data.body_names)
    for side in ("left", "right"):
        bname = f"{side}_ankle_roll_link"
        if bname in body_names:
            bidx = body_names.index(bname)
            rec[f"{side}_ankle_z"] = float(
                robot.data.body_pos_w[0, bidx, 2].item()
            )
            rec[f"{side}_ankle_lin_vel_w"] = _tensor_to_list(
                robot.data.body_lin_vel_w[0, bidx]
            )

    # Contact forces (per-body net force)
    try:
        cs = scene["contact_forces"]
        # net_forces_w is (num_envs, num_bodies, 3); take env 0
        net_forces = cs.data.net_forces_w[0]  # (B, 3)
        cs_body_names = list(cs.body_names)
        rec["body_contact_forces"] = {}
        for side in ("left", "right"):
            bname = f"{side}_ankle_roll_link"
            if bname in cs_body_names:
                bidx = cs_body_names.index(bname)
                f = net_forces[bidx].detach().cpu().float().numpy().tolist()
                rec["body_contact_forces"][bname] = f
        # Also dump pelvis + torso for sanity (should be near zero)
        for bname in ("pelvis", "torso_link", "head_pitch_link"):
            if bname in cs_body_names:
                bidx = cs_body_names.index(bname)
                f = net_forces[bidx].detach().cpu().float().numpy().tolist()
                rec["body_contact_forces"][bname] = f
        # Aggregate foot floor force (just the magnitude of net_force)
        import math
        rec["foot_floor_force_mag"] = {}
        for side in ("left", "right"):
            bname = f"{side}_ankle_roll_link"
            if bname in cs_body_names:
                bidx = cs_body_names.index(bname)
                f = net_forces[bidx]
                rec["foot_floor_force_mag"][side] = float(
                    math.sqrt(float(f[0])**2 + float(f[1])**2 + float(f[2])**2)
                )
    except Exception as exc:  # noqa: BLE001
        rec["contact_sensor_step_error"] = str(exc)

    # Motion command time
    try:
        cmd = inner.command_manager.get_term("motion")
        rec["motion_id"] = int(cmd.motion_ids[0].item())
        rec["motion_time_step"] = int(
            (cmd.motion_start_time_steps[0]
             + getattr(cmd, "motion_step", torch.tensor(0))[0]
             if hasattr(cmd, "motion_step") else cmd.motion_start_time_steps[0]
             ).item()
        )
        if hasattr(cmd, "motion_times"):
            rec["motion_time_s"] = float(cmd.motion_times[0].item())
    except Exception:  # noqa: BLE001
        pass

    return rec


def _install_patches():
    """Patch env construction + step + reset to dump diagnostics."""
    from gear_sonic import train_agent_trl

    original_create = train_agent_trl.create_manager_env

    def patched_create(*args, **kwargs):
        env = original_create(*args, **kwargs)
        original_step = env.step
        original_reset_all = env.reset_all
        sim_time_holder = {"t": 0.0}

        def patched_reset_all(*a, **kw):
            obs = original_reset_all(*a, **kw)
            sim_time_holder["t"] = 0.0
            if not _STATIC["captured"]:
                try:
                    static_cfg = _capture_static_config(env)
                    _STATIC.update(static_cfg)
                    _STATIC["captured"] = True
                    print(f"[dump_isaaclab_icecream] Captured static config. "
                          f"sim_dt={static_cfg.get('sim_dt')} "
                          f"physx={static_cfg.get('physx')} "
                          f"physics_material={static_cfg.get('physics_material')}", flush=True)
                    inner = env.env if hasattr(env, "env") else env
                    cmd = inner.command_manager.get_term("motion")
                    print(f"[dump_isaaclab_icecream] Motion ids: {cmd.motion_ids.tolist()} "
                          f"start_time_steps: {cmd.motion_start_time_steps.tolist()}", flush=True)
                    motion_keys = cmd.motion_lib._motion_data_keys  # noqa: SLF001
                    for mid in cmd.motion_ids.tolist():
                        print(f"[dump_isaaclab_icecream]   motion_id={mid} -> {motion_keys[mid]}",
                              flush=True)
                except Exception as exc:  # noqa: BLE001
                    import traceback; traceback.print_exc()
                    print(f"[dump_isaaclab_icecream] static cfg error: {exc}", flush=True)
            return obs

        def patched_step(actor_state, *a, **kw):
            results = original_step(actor_state, *a, **kw)

            # Time advance: control_dt = sim_dt * decimation
            inner = env.env if hasattr(env, "env") else env
            try:
                sim_dt = float(inner.sim.cfg.dt)
                decimation = int(inner.cfg.decimation)
            except Exception:  # noqa: BLE001
                sim_dt, decimation = 0.005, 4
            sim_time_holder["t"] += sim_dt * decimation
            _STEP_COUNT["n"] += 1
            try:
                rec = _capture_step_record(env, sim_time_holder["t"])
                # Also flag termination (dones)
                dones = results[2] if len(results) > 2 else None
                if dones is not None:
                    try:
                        rec["done"] = bool(dones[0].item()) if hasattr(dones, "item") else bool(dones[0])
                    except Exception:  # noqa: BLE001
                        rec["done"] = None
                _RECORDS.append(rec)

                if _STEP_COUNT["n"] % 25 == 0:
                    print(f"[dump_isaaclab_icecream] step {_STEP_COUNT['n']}: "
                          f"t={sim_time_holder['t']:.3f}s "
                          f"pelvis_z={rec['root_pos_w'][2]:.3f} "
                          f"l_ankle_z={rec.get('left_ankle_z', 0):.3f} "
                          f"r_ankle_z={rec.get('right_ankle_z', 0):.3f} "
                          f"l_force={rec.get('foot_floor_force_mag', {}).get('left', 0):.1f} "
                          f"r_force={rec.get('foot_floor_force_mag', {}).get('right', 0):.1f}",
                          flush=True)
            except Exception as exc:  # noqa: BLE001
                import traceback; traceback.print_exc()
                print(f"[dump_isaaclab_icecream] step capture error: {exc}", flush=True)

            if _STEP_COUNT["n"] >= _DUMP_STEPS:
                _flush_and_exit()
            return results

        env.step = patched_step
        env.reset_all = patched_reset_all
        return env

    train_agent_trl.create_manager_env = patched_create
    print("[dump_isaaclab_icecream] Installed env construction + step + reset patches.",
          flush=True)


def _flush_and_exit():
    import torch  # noqa: F401
    out = {
        "static_config": _STATIC,
        "n_steps": len(_RECORDS),
        "step_records": _RECORDS,
    }
    os.makedirs(os.path.dirname(_DUMP_PATH) or ".", exist_ok=True)
    with open(_DUMP_PATH, "w") as f:
        json.dump(out, f, indent=1)
    sz = os.path.getsize(_DUMP_PATH) / 1024
    print(f"[dump_isaaclab_icecream] Wrote {len(_RECORDS)} steps to {_DUMP_PATH} ({sz:.1f} KB)",
          flush=True)
    os._exit(0)


def main():
    import torch  # noqa: F401
    global torch_module  # noqa: PLW0603
    _install_patches()

    # We need ``torch`` reachable from inside _capture_step_record (it builds
    # a sentinel tensor for cmd.motion_step access). Pre-import here.
    import torch
    globals()["torch"] = torch

    import runpy
    eval_script_path = os.path.join(_REPO_ROOT, "gear_sonic", "eval_agent_trl.py")
    print(f"[dump_isaaclab_icecream] Handing off to {eval_script_path}", flush=True)
    sys.argv[0] = eval_script_path
    runpy.run_path(eval_script_path, run_name="__main__")


if __name__ == "__main__":
    main()
