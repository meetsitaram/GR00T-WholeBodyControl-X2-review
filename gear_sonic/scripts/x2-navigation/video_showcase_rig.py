#!/usr/bin/env python3
"""Six-robot showcase rig for video recording.

Native multi-env playback of policy-generated route clips (kp6_showcase.pkl)
with presentation extras — NO daemon, NO ring writes; the env tracks each
clip natively (envs map 1:1 to motions):
  - main viewport pinned to a static interior camera (KPFixedCam)
  - "KP Chase View" sub-window, bottom-right, chasing one robot
    (--chase-env, default = the entrance-bound robot) facing its heading
  - yellow feet (MuJoCo look), warm light matched to the splat tone
  - color-coded goal orbs: one floating above each robot, its twin at the
    robot's destination

Launch (from repo root):
    gear_sonic/scripts/x2-navigation/launch_showcase.sh
"""
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_OWN = {"--onnx", "--run-dir", "--chase-env"}


def _pop_own(argv):
    own, rest, i = {}, [], 1
    while i < len(argv):
        tok = argv[i]
        head = tok.split("=", 1)[0]
        if head in _OWN:
            if "=" in tok:
                own[head] = tok.split("=", 1)[1]
                i += 1
            else:
                own[head] = argv[i + 1]
                i += 2
        else:
            rest.append(tok)
            i += 1
    return own, rest


_own, _rest = _pop_own(sys.argv)
ONNX_PATH = os.path.abspath(os.path.expanduser(_own["--onnx"]))
RUN_DIR = os.path.expanduser(_own["--run-dir"])
CHASE_ENV = int(_own.get("--chase-env", "1"))
WORLD_POS = [float(v) for v in os.environ.get(
    "KP_WORLD_POS", "-19.99,-75.96,0").split(",")]

GOAL_COLORS = [           # per-env orb colors (envs map to motions in order)
    (0.95, 0.25, 0.20),   # red
    (0.20, 0.45, 0.95),   # blue
    (0.20, 0.85, 0.35),   # green
    (1.00, 0.65, 0.10),   # orange
    (0.85, 0.25, 0.85),   # magenta
    (0.15, 0.85, 0.85),   # cyan
]

_DRIVER = None


class _ShowDriver:
    def __init__(self, cmd):
        self.cmd = cmd
        self.ticks = 0
        self.setup_done = False
        self.chase_op = None
        self.chase_eye = None
        self.chase_lk = None
        self.orb_ops = []
        # ALL ROBOTS AT THE SAME SPOT: env origins come from terrain tiles
        # (no hydra key moves them) — overwrite every origin with env 0's
        # BEFORE the first RSI so all six robots spawn identically.
        try:
            scene = cmd._env.scene
            # Clips carry ABSOLUTE kitchen coords (they place robots
            # themselves). Correct origins = world_pos for EVERY env, so
            # refs + origin = true world position aligned with the kitchen
            # mesh. (Sidecar origins double-shifted -> off-world falls.)
            # replicate_physics=false adds a THIRD origins tensor
            # (scene._default_env_origins, grid-spaced 2.5m) — the reset
            # path read it unpinned, spawning env1+ robots meters east of
            # their references. Pin every origins tensor there is.
            for t in (getattr(getattr(scene, "terrain", None),
                              "env_origins", None),
                      getattr(scene, "_default_env_origins", None),
                      getattr(scene, "env_origins", None)):
                if t is not None and hasattr(t, "__setitem__"):
                    t[:, 0] = WORLD_POS[0]
                    t[:, 1] = WORLD_POS[1]
                    t[:, 2] = 0.0
            print("[showcase] env origins pinned to world_pos", flush=True)
        except Exception as e:
            print(f"[showcase] origin collapse failed: {e}", flush=True)
        # Overlapping robots COLLIDE (cross-env filtering is off in this
        # eval config) and PhysX blasts them apart at spawn. Filter
        # explicitly so they coexist as ghosts.
        try:
            # KP_SPAWN=stack (default): superimposed at env0 (stable while
            # motions are identical — the 'one robot' shot). huddle: 0.55m
            # circle for divergent motions (solid robots never overlap).
            if os.environ.get("KP_SPAWN", "stack") != "huddle":
                raise StopIteration   # keep collapsed origins from above
            raise RuntimeError("huddle requested")
        except StopIteration:
            print("[showcase] spawn mode: STACK (superimposed)", flush=True)
        except Exception:
            try:
                import math as _hm
                import torch as _ht
                scene = cmd._env.scene
                for holder in (getattr(scene, "terrain", None), scene):
                    t = getattr(holder, "env_origins", None) if holder else None
                    if t is not None and hasattr(t, "__setitem__"):
                        for i in range(t.shape[0]):
                            a = 2 * _hm.pi * i / max(1, t.shape[0])
                            t[i, 0] = t[0, 0] + 0.55 * _hm.cos(a)
                            t[i, 1] = t[0, 1] + 0.55 * _hm.sin(a)
                print("[showcase] huddle origins set (r=0.55m)", flush=True)
            except Exception as e2:
                print(f"[showcase] huddle failed: {e2}", flush=True)
        # Generated clips carry pose_aa=zeros, so the loader's SMPL-FK body
        # tensors (incl. the tokenizer's hand/head targets) come out STATIC
        # -> robots obediently stand still while joint refs walk. Rebuild
        # body families from MuJoCo FK of dof+root (yesterday's proven
        # conventions), and re-apply after the eval reload wipes them.
        try:
            self._fix_bodies()
            lib = cmd.motion_lib
            orig_reload = lib.load_motions_for_evaluation

            def reload_and_fix(*a, **k):
                r = orig_reload(*a, **k)
                self._fix_bodies()
                return r

            lib.load_motions_for_evaluation = reload_and_fix
        except Exception:
            import traceback
            traceback.print_exc()

    def _fix_bodies(self):
        import numpy as np
        import torch
        import mujoco
        import os as _os
        lib = self.cmd.motion_lib
        dev = lib._device
        mj2il_dof = np.asarray(lib.m_cfg.mujoco_to_isaaclab_dof)
        il2mj_dof = np.argsort(mj2il_dof)
        self._il2mj_dof = il2mj_dof  # for the KP_RECORD executed-motion path
        b_mj2il = np.asarray(lib.m_cfg.mujoco_to_isaaclab_body)
        bi = getattr(lib, "body_indexes", None)
        bi = (bi.cpu().numpy() if hasattr(bi, "cpu") else np.asarray(bi))             if bi is not None else None
        if not hasattr(self, "_mjm"):
            self._mjm = mujoco.MjModel.from_xml_path(_os.path.join(
                REPO_ROOT, "gear_sonic", "data", "assets",
                "robot_description", "mjcf", "x2_ultra.xml"))
            self._mjd = mujoco.MjData(self._mjm)
        m, d = self._mjm, self._mjd
        # ground truth from the pkl itself (loader-derived tensors are
        # pose_aa-contaminated): concat fields in the lib's motion order,
        # then add each motion's env-origin/init offset the loader applied
        import joblib
        pkl_path = os.environ.get("KP_MOTION_FILE", os.path.join(
            REPO_ROOT, "gear_sonic", "data", "motions", "kp6_showcase.pkl"))
        pkl = joblib.load(pkl_path)
        keys = [str(k) for k in lib.curr_motion_keys]
        starts = lib.length_starts.cpu().numpy()
        nums = lib._motion_num_frames.cpu().numpy()
        n_rows = lib.dof_pos.shape[0]
        pos_full = np.zeros((n_rows, len(b_mj2il), 3), np.float32)
        quat_full = np.zeros((n_rows, len(b_mj2il), 4), np.float32)
        for mi, key in enumerate(keys):
            rec = pkl[key]
            trans = np.asarray(rec["root_trans_offset"], np.float64)
            rq = np.asarray(rec["root_rot"], np.float64)      # xyzw
            dof = np.asarray(rec["dof"], np.float64)          # MJ order
            f0, nf = int(starts[mi]), int(nums[mi])
            nf = min(nf, trans.shape[0])
            # The loader keeps clips ABSOLUTE (probe: teleop clips spawn
            # at recorded coords) — FK must match: NO offset. Env origins
            # (collapsed to env0 = world_pos) are added downstream.
            off = np.array([0.0, 0.0, 0.0])
            # SMOKING GUN (probe 2026-07-22): loader derives dof from the
            # SMPL chain (pose_aa) and IGNORES the file's dof field —
            # zero pose_aa => ALL-ZERO joint refs. Overwrite dof_pos and
            # dof_vel from the pkl ground truth (IL order, fdiff vels).
            dof_il = dof[:, mj2il_dof].astype(np.float32)
            fps_m = float(1.0 / lib._motion_dt[0].item())
            dvel = np.zeros_like(dof_il)
            dvel[1:] = (dof_il[1:] - dof_il[:-1]) * fps_m
            lib.dof_pos[f0:f0 + nf] = torch.tensor(
                dof_il[:nf], device=dev)
            lib.dof_vel[f0:f0 + nf] = torch.tensor(
                dvel[:nf], device=dev)
            for j in range(nf):
                r = f0 + j
                d.qpos[0:3] = trans[j] + off
                d.qpos[3:7] = [rq[j, 3], rq[j, 0], rq[j, 1], rq[j, 2]]
                d.qpos[7:38] = dof[j]
                mujoco.mj_kinematics(m, d)
                pos_full[r] = np.asarray(d.xpos[1:])[b_mj2il]
                quat_full[r] = np.asarray(d.xquat[1:])[b_mj2il]  # wxyz
        tp = torch.tensor(pos_full, device=dev)
        tq = torch.tensor(quat_full, device=dev)
        lib.body_pos_w_full[:] = tp
        lib.body_quat_w_full[:] = tq
        if bi is not None:
            lib.body_pos_w[:] = tp[:, bi]
            lib.body_quat_w[:] = tq[:, bi]
        # velocities: finite difference for linear, zeros for angular
        fps = float(1.0 / lib._motion_dt[0].item())
        lv = torch.zeros_like(tp)
        lv[1:] = (tp[1:] - tp[:-1]) * fps
        # CRITICAL: zero each clip's first row — the global finite diff
        # spans clip boundaries, giving motion i>0 a frame-0 velocity of
        # (start_i - end_{i-1})*fps (~100 m/s). The RESET writes that into
        # the robot -> PhysX blast. This was the day's phantom "ejection".
        lv[lib.length_starts.long()] = 0
        # safety net: no reference linear velocity may exceed 2 m/s — any
        # remaining data artifact gets capped instead of blasting the robot
        over = (lv.abs() > 2.0)
        if bool(over.any()):
            rows = torch.unique(over.any(dim=2).any(dim=1).nonzero()[:, 0])
            vmax = float(lv.abs().max())
            print(f"[showcase] VEL-CLAMP HIT: {int(over.sum())} components "
                  f"over 2 m/s (max {vmax:.1f}) in rows "
                  f"{rows[:10].tolist()}{'...' if rows.numel() > 10 else ''}",
                  flush=True)
        lv.clamp_(-2.0, 2.0)
        lib.body_lin_vel_w_full[:] = lv
        lib.body_ang_vel_w_full[:] = 0
        if bi is not None:
            lib.body_lin_vel_w[:] = lv[:, bi]
            lib.body_ang_vel_w[:] = 0
        print(f"[showcase] body families rebuilt via MuJoCo FK "
              f"({n_rows} rows)", flush=True)

    def _setup(self):
        import math
        from pxr import Gf, Sdf, UsdGeom
        import omni.usd
        from omni.kit.viewport.utility import (
            create_viewport_window, get_active_viewport,
        )
        import omni.ui as ui
        stage = omni.usd.get_context().get_stage()
        env = self.cmd._env

        # hide the gray terrain (z-fights the splat floor)
        prim = stage.GetPrimAtPath("/World/ground")
        if prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()

        # warm light matched to the splat's baked tone
        lp = stage.GetPrimAtPath("/World/light")
        if lp.IsValid():
            lp.GetAttribute("inputs:intensity").Set(
                float(os.environ.get("KP_LIGHT_INT", "1800")))
            lp.GetAttribute("inputs:color").Set(
                Gf.Vec3f(1.0, 0.93, 0.80))
        dome = UsdGeom.DomeLight = None  # noqa: F841 (distant+ambient enough)

        # MuJoCo-style yellow feet on every robot (broad filter + path
        # dump so a naming mismatch is fixable from the log)
        n_feet = 0
        sample = []
        for p in stage.Traverse():
            path = str(p.GetPath())
            if "obot" not in path:
                continue
            low = path.lower()
            if len(sample) < 60 and "/env_0/" in low and p.IsA(UsdGeom.Gprim):
                sample.append(path)
            if ("ankle" in low or "foot" in low) and p.IsA(UsdGeom.Gprim):
                UsdGeom.Gprim(p).GetDisplayColorAttr().Set(
                    [Gf.Vec3f(0.95, 0.85, 0.10)])
                n_feet += 1
        if n_feet == 0:
            print("[showcase] FEET MISS — sample gprims:", flush=True)
            for sp in sample[:40]:
                print("   ", sp, flush=True)

        # main viewport -> dedicated static camera
        eye = [float(v) for v in os.environ.get(
            "KP_CAM_EYE", "-18.6,-78.3,2.1").split(",")]
        lk = [float(v) for v in os.environ.get(
            "KP_CAM_LOOKAT", "-20.6,-75.6,0.7").split(",")]
        fcam = UsdGeom.Camera.Define(stage, "/World/KPFixedCam")
        fcam.GetFocalLengthAttr().Set(14.0)
        view = Gf.Matrix4d()
        view.SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*lk), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(fcam.GetPrim()).AddTransformOp().Set(
            view.GetInverse())
        get_active_viewport().camera_path = "/World/KPFixedCam"

        # chase camera + bottom-right sub-window (~80% of previous size)
        ccam = UsdGeom.Camera.Define(stage, "/World/KPChaseCam")
        ccam.GetFocalLengthAttr().Set(16.0)
        self.chase_op = UsdGeom.Xformable(ccam.GetPrim()).AddTransformOp()
        self.Gf = Gf
        w, h = 690, 430
        vp2 = create_viewport_window("KP Chase View", width=w, height=h)
        vp2.viewport_api.camera_path = "/World/KPChaseCam"
        try:
            vw = ui.Workspace.get_window("Viewport")
            vp2.position_x = int(vw.position_x + vw.width - w - 16)
            vp2.position_y = int(vw.position_y + vw.height - h - 16)
        except Exception:
            try:
                mw = ui.Workspace.get_main_window_width()
                vp2.position_x = int(mw - w - 14)
                vp2.position_y = 60
            except Exception:
                pass

        # color-coded orbs: above each robot + at its destination
        wps = json.load(open(
            "/home/stickbot/projects/x2-kitchen-sim/configs/waypoints.json"))
        keys = list(self.cmd.motion_lib.curr_motion_keys)
        num = min(env.num_envs, len(GOAL_COLORS))
        for i in range(num):
            col = Gf.Vec3f(*GOAL_COLORS[i])
            goal = str(keys[i]).replace("route_", "") if i < len(keys) else None
            # destination orb (static)
            if goal in wps:
                gx = wps[goal]["xy"][0] + WORLD_POS[0]
                gy = wps[goal]["xy"][1] + WORLD_POS[1]
                s = UsdGeom.Sphere.Define(stage, f"/World/KPGoalOrb_{i}")
                s.GetRadiusAttr().Set(0.07)
                s.GetDisplayColorAttr().Set([col])
                UsdGeom.Xformable(s.GetPrim()).AddTranslateOp().Set(
                    Gf.Vec3d(gx, gy, 1.75))
            # robot orb (tick-updated)
            s2 = UsdGeom.Sphere.Define(stage, f"/World/KPBotOrb_{i}")
            s2.GetRadiusAttr().Set(0.055)
            s2.GetDisplayColorAttr().Set([col])
            self.orb_ops.append(
                UsdGeom.Xformable(s2.GetPrim()).AddTranslateOp())
        print(f"[showcase] setup done: feet colored={n_feet}, orbs={num}, "
              f"chase env={CHASE_ENV}, keys={[str(k) for k in keys]}",
              flush=True)

    def _record_tick(self, path):
        """Capture the EXECUTED robot motion (not the reference) for one full
        clip cycle — cycle 2, so launch transients are excluded — then dump a
        clip-format pkl and optionally exit (KP_RECORD_EXIT=1). Executed clips
        replay through SONIC almost exactly (it tracked them once already),
        which makes multi-robot path-separation gates trustworthy."""
        import numpy as np
        ts = int(self.cmd.time_steps[0])
        prev = getattr(self, "_rec_prev_ts", None)
        self._rec_prev_ts = ts
        if getattr(self, "_rec_done", False):
            return
        boundary = prev is not None and ts < prev
        frames = getattr(self, "_rec_frames", None)
        if frames is not None and boundary and len(frames) > 50:
            rp = np.stack([f[0] for f in frames])
            rq = np.stack([f[1] for f in frames])
            jp = np.stack([f[2] for f in frames])
            rp[:, 0] -= WORLD_POS[0]
            rp[:, 1] -= WORLD_POS[1]
            n = rp.shape[0]
            key = os.environ.get("KP_RECORD_KEY", "route_exec")
            import joblib
            joblib.dump({key: {
                "root_trans_offset": rp.astype(np.float32),
                "root_rot": rq[:, [1, 2, 3, 0]].astype(np.float32),  # ->xyzw
                "dof": jp[:, self._il2mj_dof].astype(np.float32),
                "fps": 50.0,
                "pose_aa": np.zeros((n, 32, 3), np.float32),
            }}, path, compress=3)
            print(f"[record] wrote {path}: {n} frames ({n/50.0:.1f}s)",
                  flush=True)
            self._rec_done = True
            if os.environ.get("KP_RECORD_EXIT"):
                os._exit(0)
            return
        if frames is None:
            if not boundary:
                return
            frames = self._rec_frames = []
            print("[record] cycle boundary — recording started", flush=True)
        data = self.cmd._env.scene["robot"].data
        frames.append((data.root_pos_w[0].cpu().numpy().copy(),
                       data.root_quat_w[0].cpu().numpy().copy(),
                       data.joint_pos[0].cpu().numpy().copy()))

    def tick(self):
        self.ticks += 1
        if True:
            # idempotent re-pin EVERY tick: something restores stale origins
            # between resets (reset z lands +0.4 above reference)
            try:
                scene = self.cmd._env.scene
                for t in (getattr(getattr(scene, "terrain", None),
                                  "env_origins", None),
                          getattr(scene, "_default_env_origins", None),
                          getattr(scene, "env_origins", None)):
                    if t is not None and hasattr(t, "__setitem__"):
                        t[:, 0] = WORLD_POS[0]
                        t[:, 1] = WORLD_POS[1]
                        t[:, 2] = 0.0
            except Exception:
                pass
        if not getattr(self, "_term_spy", False) and self.ticks >= 2:
            self._term_spy = True
            self._spy_n = 0
            try:
                tm = self.cmd._env.termination_manager
                names = list(getattr(tm, "_term_names", []))
                cfgs = list(getattr(tm, "_term_cfgs", []))
                for name, cfg in zip(names, cfgs):
                    if name != "anchor_pos":
                        continue
                    orig_fn = cfg.func

                    def _spy(env, *a, __orig=orig_fn, **kw):
                        out = __orig(env, *a, **kw)
                        try:
                            if out.shape[0] > 1 and bool(out[1]) \
                                    and self._spy_n < 6:
                                self._spy_n += 1
                                c = env.command_manager.get_term("motion")
                                print(
                                    f"[spy] anchor_pos FIRED env1: "
                                    f"ref_z={float(c.anchor_pos_w[1, 2]):.3f} "
                                    f"rob_z={float(c.robot_anchor_pos_w[1, 2]):.3f} "
                                    f"ema={float(c.running_ref_root_height[1]):.3f} "
                                    f"ts={int(c.time_steps[1])}", flush=True)
                        except Exception:
                            pass
                        return out

                    cfg.func = _spy
                    print("[showcase] anchor_pos spy armed", flush=True)
            except Exception as e:
                print(f"[showcase] spy arm failed: {e}", flush=True)
        if getattr(self, "_ej_n", 0) < 30:
            try:
                ts1 = int(self.cmd.time_steps[1])
                if ts1 <= 3:
                    self._ej_n = getattr(self, "_ej_n", 0) + 1
                    rs = self.cmd._env.scene["robot"].data.root_state_w[1]
                    print(f"[eject] ts={ts1} pos=({float(rs[0]):.2f},"
                          f"{float(rs[1]):.2f},{float(rs[2]):.3f}) "
                          f"vz={float(rs[9]):.2f}", flush=True)
            except Exception:
                self._ej_n = 99
        rec_path = os.environ.get("KP_RECORD")
        if rec_path:
            try:
                self._record_tick(rec_path)
            except Exception:
                import traceback
                traceback.print_exc()
                os.environ.pop("KP_RECORD", None)
        if self.ticks % 150 == 0:
            try:
                ts = self.cmd.time_steps[:3].tolist()
                rp = self.cmd._env.scene["robot"].data.root_pos_w
                r0 = rp[0].tolist()
                zmin = float(rp[:, 2].min())
                zs = " ".join(f"{float(z):.2f}" for z in rp[:, 2])
                xys = " ".join(
                    f"({float(p[0]):+.1f},{float(p[1]):+.1f})"
                    for p in rp[:, :2])
                print(f"[showcase] xy=[{xys}]", flush=True)
                import time as _time
                print(f"[showcase] hb tick={self.ticks} time_steps={ts} "
                      f"r0=({r0[0]:.2f},{r0[1]:.2f},{r0[2]:.2f}) "
                      f"zmin={zmin:.2f} z=[{zs}] "
                      f"wall={_time.time():.2f}", flush=True)
                # PIPELINE PROBE (user methodology): per-env reference
                # stream at the playhead — arriving? distinct? moving?
                lib = self.cmd.motion_lib
                gs = (lib.length_starts[self.cmd.motion_ids]
                      + self.cmd.time_steps).long()
                dref = lib.dof_pos[gs]              # (N,31) joint refs
                dvel = lib.dof_vel[gs]              # (N,31) ref velocities
                mst = self.cmd.motion_start_time_steps.tolist()
                mid = self.cmd.motion_ids.tolist()
                tot = lib._motion_num_frames.tolist()
                print(f"[probe] motion_ids={mid} start_steps={mst} "
                      f"num_frames={tot}", flush=True)
                try:
                    tm = self.cmd._env.termination_manager
                    dones = {n: tm.get_term(n).int().tolist()
                             for n in tm.active_terms}
                    elb = self.cmd._env.episode_length_buf.tolist()
                    print(f"[probe] dones={dones} episode_len={elb} "
                          f"max_ep={self.cmd._env.max_episode_length}",
                          flush=True)
                    sc = self.cmd._env.scene
                    o1 = sc.env_origins[1].tolist() if sc.env_origins is not None else None
                    d1 = (sc._default_env_origins[1].tolist()
                          if getattr(sc, "_default_env_origins", None) is not None else None)
                    tr = getattr(sc, "terrain", None)
                    t1 = (tr.env_origins[1].tolist()
                          if tr is not None and tr.env_origins is not None else None)
                    print(f"[probe] origins env1: scene={o1} default={d1} "
                          f"terrain={t1}", flush=True)
                except Exception as te:
                    print(f"[probe] term probe err: {te}", flush=True)
                ap = getattr(self.cmd, "anchor_pos_w", None)
                rap = getattr(self.cmd, "robot_anchor_pos_w", None)
                for i in range(min(6, dref.shape[0])):
                    a = (f"({float(ap[i,0]):.2f},{float(ap[i,1]):.2f},"
                         f"{float(ap[i,2]):.2f})"
                         if ap is not None else "n/a")
                    rz = (f"{float(rap[i,2]):.2f}" if rap is not None
                          else "n/a")
                    print(f"[probe] env{i} anchor={a} robot_anchor_z={rz} "
                          f"dof_mean={float(dref[i].mean()):+.3f} "
                          f"dof[3]={float(dref[i,3]):+.3f} "
                          f"|dvel|={float(dvel[i].abs().mean()):.3f}",
                          flush=True)
            except Exception as e:
                print(f"[showcase] hb err: {e}", flush=True)
        if not self.setup_done and self.ticks >= 10:
            self.setup_done = True
            try:
                self._setup()
            except Exception:
                import traceback
                traceback.print_exc()
        if not self.setup_done or self.ticks % 5:
            return
        try:
            import math
            Gf = self.Gf
            robot = self.cmd._env.scene["robot"]
            roots = robot.data.root_state_w
            # robot orbs
            for i, op in enumerate(self.orb_ops):
                r = roots[i].cpu().numpy()
                op.Set(Gf.Vec3d(float(r[0]), float(r[1]), float(r[2]) + 0.62))
            # chase cam behind + facing the chase robot
            r = roots[min(CHASE_ENV, roots.shape[0] - 1)].cpu().numpy()
            ryaw = math.atan2(2 * (r[3] * r[6] + r[4] * r[5]),
                              1 - 2 * (r[5] ** 2 + r[6] ** 2))
            back, hh, ahead = 1.3, 0.65, 1.5
            cx, sx = math.cos(ryaw), math.sin(ryaw)
            eye_t = [r[0] - back * cx, r[1] - back * sx, r[2] + hh]
            lk_t = [r[0] + ahead * cx, r[1] + ahead * sx, r[2] + 0.2]
            a = 0.12
            if self.chase_eye is None:
                self.chase_eye, self.chase_lk = eye_t, lk_t
            else:
                self.chase_eye = [p + a * (t - p) for p, t in
                                  zip(self.chase_eye, eye_t)]
                self.chase_lk = [p + a * (t - p) for p, t in
                                 zip(self.chase_lk, lk_t)]
            view = Gf.Matrix4d()
            view.SetLookAt(Gf.Vec3d(*self.chase_eye),
                           Gf.Vec3d(*self.chase_lk), Gf.Vec3d(0, 0, 1))
            self.chase_op.Set(view.GetInverse())
        except Exception as e:
            print(f"[showcase] tick error: {e}", flush=True)


def _wrap_commands_module(mod):
    TrackingCommand = mod.TrackingCommand
    orig_init = TrackingCommand.__init__
    orig_compute = TrackingCommand.compute

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        global _DRIVER
        if _DRIVER is None:
            _DRIVER = _ShowDriver(self)

    def compute(self, dt):
        orig_compute(self, dt)
        if _DRIVER is not None:
            _DRIVER.tick()

    TrackingCommand.__init__ = __init__
    TrackingCommand.compute = compute
    print("[showcase] TrackingCommand wrapped", flush=True)


class _PostImportHook(importlib.abc.MetaPathFinder):
    TARGET = "gear_sonic.envs.manager_env.mdp.commands"

    def __init__(self):
        self._busy = False

    def find_spec(self, fullname, path, target=None):
        if fullname != self.TARGET or self._busy:
            return None
        self._busy = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._busy = False
        if spec is None or spec.loader is None:
            return None
        orig_exec = spec.loader.exec_module

        def exec_module(module):
            orig_exec(module)
            _wrap_commands_module(module)

        spec.loader.exec_module = exec_module  # type: ignore[method-assign]
        return spec


sys.meta_path.insert(0, _PostImportHook())

from gear_sonic.scripts.eval_x2_isaacsim_onnx import (  # noqa: E402
    _ensure_checkpoint_override,
    _patch_actor_with_onnx,
)

_rest = _ensure_checkpoint_override(_rest, RUN_DIR)
sys.argv = [sys.argv[0]] + _rest
_patch_actor_with_onnx(onnx_path=ONNX_PATH, encoder_name="g1",
                       compare=False, diff_csv="/dev/null")

if os.environ.get("KP_VERBOSE"):
    # Hydra owns argv at process start, so --verbose can't ride sys.argv from
    # the shell. Inject it when AppLauncher inits — after hydra parsing, at
    # the exact moment SimulationApp reads sys.argv for its log level.
    import isaaclab.app as _ila  # noqa: E402

    _orig_al_init = _ila.AppLauncher.__init__

    def _verbose_init(self, *a, **k):
        if "--verbose" not in sys.argv:
            sys.argv.append("--verbose")
        _orig_al_init(self, *a, **k)

    _ila.AppLauncher.__init__ = _verbose_init

import runpy  # noqa: E402

runpy.run_module("gear_sonic.eval_agent_trl", run_name="__main__")
