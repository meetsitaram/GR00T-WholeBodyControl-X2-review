#!/usr/bin/env python3
"""Drive SONIC with kplanner-generated references inside the ACTUAL training env.

x2-kitchen-sim M1 step 2. Architecture:

    intent schedule (RL/pad stand-in: velocity+facing, held >=200ms)
        -> TorchPlannerBackend (kplanner: vqvae+pose+root ckpts)
        -> reference frames written IN PLACE into the loaded MotionLib ring
        -> TrackingCommand / obs / SONIC ONNX / PD / PhysX all unchanged.

We deliberately do NOT re-implement any env machinery (see 2026-07-21 debug
night: the hand-rolled bridge lost to the env on every reset/assembly detail).
Instead we load ``x2_ultra_idle_stand.pkl`` normally (valid shapes/FK), then
grow motion 0 into a ring buffer and overwrite rows ahead of the playhead:

    dof_pos / dof_vel      exact from planner frames  (all SONIC sees)
    root pose / velocities exact from planner frames  (anchor obs)
    other body rows        yaw-rotated frame-0 offsets (feeds only critic /
                           tracking-reward terms -- unused at eval)

Usage (viewer on, kitchen world):
    python -m gear_sonic.scripts.run_x2_kplanner_env \
        --run-dir  $HOME/x2_cloud_checkpoints/h200-...-20260501 \
        --onnx     .../exported/model_step_025000_g1.onnx \
        +num_envs=1 +headless=False \
        ++manager_env.commands.motion.motion_lib_cfg.motion_file=gear_sonic/data/motions/x2_ultra_idle_stand.pkl \
        ++manager_env.config.world_usd=.../kitchen_splat.usdz \
        ++manager_env.config.world_collision_usd=.../kitchen_collision.usd \
        '++manager_env.config.world_pos=[-19.99,-75.96,0.0]'
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# --------------------------------------------------------------------------
# Own CLI (everything else forwarded to Hydra, mirroring eval_x2_isaacsim_onnx)
# --------------------------------------------------------------------------
_OWN = {"--onnx", "--run-dir", "--vqvae-ckpt", "--pose-ckpt", "--root-ckpt",
        "--ring-seconds", "--lookahead-s", "--hip"}


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
VQVAE = os.path.expanduser(_own.get(
    "--vqvae-ckpt", f"{REPO_ROOT}/motionbricks/out/motionbricks_vqvae_x2/version_1/checkpoints/model-step=0500000.ckpt"))
POSE = os.path.expanduser(_own.get(
    "--pose-ckpt", f"{REPO_ROOT}/motionbricks/out/motionbricks_pose_x2/version_1/checkpoints/model-step=0500000.ckpt"))
ROOT = os.path.expanduser(_own.get(
    "--root-ckpt", f"{REPO_ROOT}/motionbricks/out/motionbricks_root_x2/version_1/checkpoints/model-step=0315000.ckpt"))
RING_SECONDS = float(_own.get("--ring-seconds", "120.0"))
LOOKAHEAD_S = float(_own.get("--lookahead-s", "1.5"))
HIP_TARGET = float(_own.get("--hip", "0.68"))

# Intent schedule: (duration_s, yaw_rate, vx_lateral, vz_forward). Certified
# magnitudes per plan (KPLANNER_FIXED_FWD_MPS=0.5, TURN=1.0), held >=200ms --
# this is the RL/pad stand-in until the nav policy provides the stream.
INTENT_SCHEDULE = [
    (3.0, 0.0, 0.0, 0.0),      # stand
    (3.0, 0.0, 0.0, 0.3),      # forward 0.3 m/s, facing straight (user spec)
    (6.0, 0.0, 0.0, 0.0),      # stand
]
_SCHED_TOTAL = sum(s[0] for s in INTENT_SCHEDULE)


_ZERO_ONLY = os.environ.get("KP_ZERO_ONLY", "0") == "1"
_ALWAYS_PLANNER = os.environ.get("KP_ALWAYS_PLANNER", "0") == "1"


def _intent_at(t: float):
    if _ZERO_ONLY:
        return (0.0, 0.0, 0.0, HIP_TARGET)
    t = t % _SCHED_TOTAL
    for dur, yaw, vx, vz in INTENT_SCHEDULE:
        if t < dur:
            return (yaw, vx, vz, HIP_TARGET)
        t -= dur
    return (0.0, 0.0, 0.0, HIP_TARGET)


# --------------------------------------------------------------------------
# Planner driver: owns backend + ring writing. Built lazily on first
# TrackingCommand.compute (isaaclab app is up by then).
# --------------------------------------------------------------------------
class _PlannerDriver:
    def __init__(self, cmd):
        import numpy as np
        import torch

        self.np, self.torch = np, torch
        self.cmd = cmd
        lib = cmd.motion_lib
        self.lib = lib
        dev = lib._device  # noqa: SLF001
        self.dev = dev

        fps = float(1.0 / lib._motion_dt[0].item())  # noqa: SLF001
        self.fps = fps
        self.ring_n = int(RING_SECONDS * fps)
        self.look_n = int(LOOKAHEAD_S * fps)

        f0 = int(lib.length_starts[0].item())

        def grow(t, fill_row):
            new = fill_row.unsqueeze(0).repeat(
                (self.ring_n,) + (1,) * (t.dim() - 1)).clone()
            return new

        # The lib keeps TWO body-tensor families: mujoco-order xyzw
        # (body_*_w) and IsaacLab-order wxyz (body_*_w_full). Obs getters
        # read the _full family; grow+write BOTH, each in its own
        # convention (quats: xyzw for base, wxyz for _full).
        self.families = []
        for pos_n, quat_n, lv_n, av_n, conv in (
            ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "xyzw"),
            ("body_pos_w_full", "body_quat_w_full",
             "body_lin_vel_w_full", "body_ang_vel_w_full", "wxyz"),
        ):
            if getattr(lib, pos_n, None) is None:
                continue
            self.families.append({
                "pos": pos_n, "quat": quat_n, "lv": lv_n, "av": av_n,
                "conv": conv,
                "f0_pos": getattr(lib, pos_n)[f0].clone(),
                "f0_quat": getattr(lib, quat_n)[f0].clone(),
            })

        # Bisect mode (KP_RING_IDLE=1): replay the ORIGINAL clip rows through
        # the same ring machinery, planner bypassed. Stand => machinery OK,
        # planner-frame content is the issue; drift => ring writes themselves.
        self.ring_idle = os.environ.get("KP_RING_IDLE", "0") == "1"
        self.orig = {}
        snap_names = ["dof_pos", "dof_vel", "root_linv_vel_w", "root_ang_vel_w",
                      "feet_l", "feet_r"]
        for fam in self.families:
            snap_names += [fam["pos"], fam["quat"], fam["lv"], fam["av"]]
        # Always snapshot the original clip rows: the hybrid driver replays
        # them during idle intents (PC2 pattern — zero intent never reaches
        # the planner), and KP_RING_IDLE=1 replays them exclusively.
        for nme in snap_names:
            t = getattr(lib, nme, None)
            if t is not None:
                self.orig[nme] = t.clone()
        self.orig_len = lib.dof_pos.shape[0]

        self.f0_dof = lib.dof_pos[f0].clone()                  # [31] IL order
        self.f0_root = self.families[0]["f0_pos"][0].clone()
        self.f0_feet_l = lib.feet_l[f0].clone()
        self.f0_feet_r = lib.feet_r[f0].clone()
        # root quat wxyz for planner seeding (from _full if present else convert)
        if len(self.families) > 1:
            self.f0_root_quat_wxyz = self.families[1]["f0_quat"][0].cpu().numpy()
        else:
            q = self.families[0]["f0_quat"][0].cpu().numpy()  # xyzw
            self.f0_root_quat_wxyz = np.array([q[3], q[0], q[1], q[2]])

        grow_names = ["body_pos_b", "root_linv_vel_w", "root_ang_vel_w",
                      "dof_vel", "dof_pos", "feet_l", "feet_r"]
        for fam in self.families:
            grow_names += [fam["pos"], fam["quat"], fam["lv"], fam["av"]]
        for name in grow_names:
            t = getattr(lib, name, None)
            if t is None:
                continue
            setattr(lib, name, grow(t, t[f0]))
        # velocities start at zero
        for name in ("root_linv_vel_w", "root_ang_vel_w", "dof_vel"):
            getattr(lib, name).zero_()
        for fam in self.families:
            getattr(lib, fam["lv"]).zero_()
            getattr(lib, fam["av"]).zero_()

        lib.length_starts = torch.zeros_like(lib.length_starts)
        lib._motion_num_frames[:] = self.ring_n           # noqa: SLF001
        lib._motion_lengths[:] = self.ring_n / fps        # noqa: SLF001
        if hasattr(lib, "_motion_num_steps"):
            lib._motion_num_steps[:] = self.ring_n        # noqa: SLF001

        # mujoco->isaaclab dof gather map (planner emits mujoco order)
        self.mj2il = np.asarray(lib.m_cfg.mujoco_to_isaaclab_dof)

        # ---- planner backend ------------------------------------------------
        mb_root = os.path.join(REPO_ROOT, "motionbricks")
        if mb_root not in sys.path:
            sys.path.insert(0, mb_root)
        sys.path.insert(0, os.path.join(REPO_ROOT, "gear_sonic", "scripts"))
        spec = importlib.util.spec_from_file_location(
            "pc2_kplanner_onnx",
            os.path.join(REPO_ROOT, "gear_sonic", "scripts", "pc2_kplanner_onnx.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("pc2_kplanner_onnx", mod)
        spec.loader.exec_module(mod)
        self.backend = mod.TorchPlannerBackend(
            vqvae_ckpt=Path(VQVAE), pose_ckpt=Path(POSE), root_ckpt=Path(ROOT),
            device="cuda", replan_threshold_frames=8, planner_mode=None)
        # Deploy-canonical reference smoothing (pc2 runs every planner frame
        # through this before SONIC): 300 ms halfcos ramp on lower-body
        # joints whenever the reference steps >0.05 rad — covers replan
        # boundaries and frame jitter. Ring-idle bisect proved the ring
        # machinery clean; raw planner frames destabilize without this.
        self.smoother = mod.ReferenceStepSmoother()
        # Deploy-canonical command smoothing: the pad bridge never sends the
        # planner a step-function velocity — tau-ramp intents like PC2 does.
        self.cmd_ramp = mod.ColdStartVelocityRamp(tau_s=0.35)

        qpos0 = np.zeros(38, dtype=np.float32)
        qpos0[0:3] = self.f0_root.cpu().numpy()
        qpos0[3:7] = self.f0_root_quat_wxyz
        self.il2mj_inv = self._il2mj()
        qpos0[7:38] = self.f0_dof.cpu().numpy()[self.il2mj_inv]
        self.backend.reset(qpos0)
        self.backend.replan(_intent_at(0.0))
        # Timeline audit log (user request): wall vs sim time, per second
        self.wall0 = time.monotonic()
        self.n_replans = 0
        self.tl_path = os.path.join(
            os.environ.get("KP_OBS_DUMP_DIR",
                           "/tmp/claude-1000/-home-stickbot-Projects-GR00T-"
                           "WholeBodyControl/fa9112a6-98d6-4d9b-9c15-"
                           "3150190f6aa2/scratchpad"),
            "kp_timeline.jsonl")
        # Hybrid idle/motion state (PC2 pattern)
        self.idle_eps = 0.05
        self.was_idle = True
        self.idle_src = 0
        self.qpos0 = qpos0.copy()
        self.hold_trans = qpos0[0:3].copy()
        self.hold_quat_wxyz = qpos0[3:7].copy()
        self.last_dof_mj = qpos0[7:38].copy()
        # idle-anchor transform, identity at start (user spec: resets anchor
        # to the STARTING position, not where the robot stopped)
        self.ia_c, self.ia_s = 1.0, 0.0
        self.ia_q = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32,
                                 device=dev)
        self.ia_trans = torch.tensor(qpos0[0:3], dtype=torch.float32,
                                     device=dev)
        self.ia_root0 = self.families[0]["f0_pos"][0].clone()

        self.write_head = 0
        self.prev_dof_il = None
        self.prev_trans = None
        self.prev_yaw = None
        self.ticks = 0
        print(f"[kplanner-env] driver up: ring {self.ring_n} rows @ {fps:.0f} Hz, "
              f"lookahead {self.look_n}, backend {self.backend.describe()}",
              flush=True)

    def _il2mj(self):
        # inverse gather of mj2il: il2mj[j] gives mujoco index of IL joint j
        import numpy as np
        inv = np.empty_like(self.mj2il)
        inv[self.mj2il] = np.arange(len(self.mj2il))
        return inv

    def _yaw_of(self, quat_wxyz):
        import math
        w, x, y, z = quat_wxyz
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def tick(self):
        """Called after TrackingCommand.compute each env step."""
        import math

        np, torch = self.np, self.torch
        lib, dev = self.lib, self.dev
        self.ticks += 1
        # playhead in ring rows (lib frame steps @ lib fps)
        play = int((self.cmd.motion_start_time_steps[0] + self.cmd.time_steps[0]).item())
        # aggregate reset forensics across the heartbeat window
        if not hasattr(self, "_last_play"):
            self._last_play, self._resets, self._fired_agg = play, 0, {}
        if play < self._last_play:
            self._resets += 1
        self._last_play = play
        try:
            tm = self.cmd._env.termination_manager
            if bool(tm.dones[0]):
                for k in tm.active_terms:
                    if bool(tm.get_term(k)[0]):
                        self._fired_agg[k] = self._fired_agg.get(k, 0) + 1
                if bool(getattr(tm, "time_outs", [False])[0]):
                    self._fired_agg["<time_out>"] = self._fired_agg.get("<time_out>", 0) + 1
        except Exception:
            pass
        t_sim = self.ticks * 0.02

        wrote = 0
        while self.write_head < play + self.look_n and wrote < 4 * self.look_n:
            if self.ring_idle:
                r = self.write_head % self.ring_n
                src = self.write_head % self.orig_len
                for nme, t in self.orig.items():
                    getattr(lib, nme)[r] = t[src]
                self.write_head += 1
                wrote += 1
                continue
            target = _intent_at(t_sim)
            r = self.write_head % self.ring_n
            idle_now = (not _ALWAYS_PLANNER) and max(
                abs(target[0]), abs(target[1]),
                abs(target[2])) < self.idle_eps
            if idle_now:
                # PC2 idle pattern: zero intent NEVER reaches the planner
                # (locomotion model, OOD at v=0 -> stepping-in-place output;
                # proven by the KP_RING_IDLE bisect: identical machinery
                # replaying the clip stands rock-solid). Write the idle
                # clip's own rows, root held at the last reference pose.
                # Idle reference = the original clip rows (live sway — the
                # frozen-root variant made SONIC march), RE-ANCHORED to the
                # held pose (PC2 idle-resync): translate + yaw-rotate every
                # pos/quat channel so "stand here" means where the robot
                # stopped, not the clip's origin.
                if not self.was_idle:
                    yaw_h = self._yaw_of(self.hold_quat_wxyz)
                    yaw_0 = self._yaw_of(self.f0_root_quat_wxyz)
                    d = yaw_h - yaw_0
                    self.ia_c = math.cos(d)
                    self.ia_s = math.sin(d)
                    self.ia_q = torch.tensor(
                        [math.cos(d / 2), 0.0, 0.0, math.sin(d / 2)],
                        dtype=torch.float32, device=dev)
                    self.ia_trans = torch.tensor(
                        self.hold_trans, dtype=torch.float32, device=dev)
                    self.ia_root0 = self.orig[self.families[0]["pos"]][0, 0].clone()
                src = self.idle_src % self.orig_len
                self.idle_src += 1

                def _yawrot(p):
                    # rotate XY of [.,3] rows by the anchor yaw delta
                    out = p.clone()
                    out[..., 0] = self.ia_c * p[..., 0] - self.ia_s * p[..., 1]
                    out[..., 1] = self.ia_s * p[..., 0] + self.ia_c * p[..., 1]
                    return out

                def _yawq(q, conv):
                    # premultiply rows [.,4] by the pure-yaw anchor quat
                    qw = q if conv == "wxyz" else q[..., [3, 0, 1, 2]]
                    w1, z1 = self.ia_q[0], self.ia_q[3]
                    w2, x2, y2, z2 = qw[..., 0], qw[..., 1], qw[..., 2], qw[..., 3]
                    res = torch.stack([w1 * w2 - z1 * z2, w1 * x2 - z1 * y2,
                                       w1 * y2 + z1 * x2, w1 * z2 + z1 * w2],
                                      dim=-1)
                    return res if conv == "wxyz" else res[..., [1, 2, 3, 0]]

                for nme, t in self.orig.items():
                    getattr(lib, nme)[r] = t[src]
                for fam in self.families:
                    p = self.orig[fam["pos"]][src]
                    getattr(lib, fam["pos"])[r] = (
                        _yawrot(p - self.ia_root0) + self.ia_trans)
                    getattr(lib, fam["quat"])[r] = _yawq(
                        self.orig[fam["quat"]][src], fam["conv"])
                    getattr(lib, fam["lv"])[r] = _yawrot(self.orig[fam["lv"]][src])
                    getattr(lib, fam["av"])[r] = self.orig[fam["av"]][src]
                self.last_dof_mj = self.orig["dof_pos"][src].cpu().numpy()[
                    self.il2mj_inv].astype(np.float32)
                self.n_idle = getattr(self, "n_idle", 0) + 1
                self.was_idle = True
                self.write_head += 1
                wrote += 1
                continue
            else:
                if self.was_idle:
                    # idle -> motion: reseed planner from the held pose
                    qpos = np.zeros(38, dtype=np.float32)
                    qpos[0:3] = self.hold_trans
                    qpos[3:7] = self.hold_quat_wxyz
                    qpos[7:38] = self.last_dof_mj
                    self.backend.reset(qpos)
                    self.cmd_ramp.reset_idle()
                    self.backend.replan(self.cmd_ramp.step(target, 1.0 / self.fps))
                    self.n_replans += 1
                    self.was_idle = False
                target = self.cmd_ramp.step(target, 1.0 / self.fps)
                if self.backend.frames_remaining <= 1:
                    self.backend.replan(target)
                    self.n_replans += 1
                f = self.backend.get_next_frame_resampled(self.fps)  # qpos[38]
                trans = f[0:3]
                quat_wxyz = f[3:7]
                dof_mj = self.smoother.update(
                    np.asarray(f[7:38], dtype=np.float64),
                    t_now=self.write_head / self.fps)
                self.hold_trans = np.asarray(trans, dtype=np.float32).copy()
                self.hold_quat_wxyz = np.asarray(quat_wxyz,
                                                 dtype=np.float32).copy()
                self.n_plan = getattr(self, "n_plan", 0) + 1
            self.last_dof_mj = np.asarray(dof_mj, dtype=np.float32).copy()
            dof_il = dof_mj[self.mj2il]

            t_dof = torch.tensor(dof_il, dtype=torch.float32, device=dev)
            t_trans = torch.tensor(trans, dtype=torch.float32, device=dev)
            t_quat = torch.tensor(quat_wxyz, dtype=torch.float32, device=dev)

            # Handoff blend (mirrors deploy's ReferenceStepSmoother): the
            # planner's first frames settle from the seed pose into its own
            # stance with up to ~3 rad/s joint velocities; tracked raw they
            # lurch the robot into recovery-stepping. Ramp the first 50 rows
            # (1 s) from the RSI pose into planner frames.
            BLEND_N = 50
            if self.write_head < BLEND_N:
                a = self.write_head / float(BLEND_N)
                t_dof = (1.0 - a) * self.f0_dof + a * t_dof
            lib.dof_pos[r] = t_dof
            yaw = self._yaw_of(quat_wxyz)
            yaw0 = self._yaw_of(self.f0_root_quat_wxyz)
            if self.prev_dof_il is not None:
                lib.dof_vel[r] = (t_dof - self.prev_dof_il) * self.fps
                lib.root_linv_vel_w[r] = (t_trans - self.prev_trans) * self.fps
                dyaw = math.atan2(math.sin(yaw - self.prev_yaw),
                                  math.cos(yaw - self.prev_yaw))
                lib.root_ang_vel_w[r] = torch.tensor(
                    [0.0, 0.0, dyaw * self.fps], device=dev)
            c, s = math.cos(yaw - yaw0), math.sin(yaw - yaw0)
            q_xyzw = torch.tensor(
                [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
                dtype=torch.float32, device=dev)
            q_wxyz = torch.tensor(quat_wxyz, dtype=torch.float32, device=dev)
            for fam in self.families:
                off = fam["f0_pos"] - fam["f0_pos"][0]  # [nb,3]
                rot = off.clone()
                rot[:, 0] = c * off[:, 0] - s * off[:, 1]
                rot[:, 1] = s * off[:, 0] + c * off[:, 1]
                getattr(lib, fam["pos"])[r] = t_trans.unsqueeze(0) + rot
                getattr(lib, fam["pos"])[r, 0] = t_trans
                getattr(lib, fam["quat"])[r] = fam["f0_quat"]
                getattr(lib, fam["quat"])[r, 0] = (
                    q_xyzw if fam["conv"] == "xyzw" else q_wxyz)
                getattr(lib, fam["lv"])[r] = lib.root_linv_vel_w[r].unsqueeze(0)
                getattr(lib, fam["av"])[r] = lib.root_ang_vel_w[r].unsqueeze(0)
            lib.feet_l[r] = self.f0_feet_l
            lib.feet_r[r] = self.f0_feet_r

            self.prev_dof_il = t_dof
            self.prev_trans = t_trans
            self.prev_yaw = yaw
            self.write_head += 1
            wrote += 1

        # Fall reset (anchor terminations are disabled in planner mode):
        # pelvis below 0.35 m => force truncation so the env RSIs instead of
        # the robot lying crashed until the episode timeout.
        try:
            rob_z = float(self.cmd._env.scene["robot"].data.root_pos_w[0, 2])
            if rob_z < 0.35:
                # Jam the motion cursor to the clip end: the env's own
                # tracking_time_out termination fires and resets through the
                # normal path. (episode_length_buf writes do nothing here —
                # this env's timeout is motion-completion-based.)
                total = int(self.cmd.motion_lib.get_time_step_total(
                    self.cmd.motion_ids)[0])
                self.cmd.time_steps[:] = total
        except Exception:
            pass
        # env reset rewinds the playhead -> reseed driver state
        if play + self.look_n < self.write_head - 2 * self.look_n:
            print("[kplanner-env] playhead rewound (env reset) — reseeding "
                  "driver state", flush=True)
            self.write_head = play
            self.was_idle = True
            self.idle_src = 0
            self.smoother._last_q = None
            self.prev_dof_il = None

        if self.ticks % 50 == 0:
            try:
                _rob = self.cmd._env.scene["robot"].data.root_pos_w[0].tolist()
                with open(self.tl_path, "a") as _fh:
                    _fh.write(json.dumps({
                        "wall_s": round(time.monotonic() - self.wall0, 2),
                        "sim_s": round(t_sim, 2),
                        "play": play, "head": self.write_head,
                        "idle_writes": getattr(self, "n_idle", 0),
                        "plan_writes": getattr(self, "n_plan", 0),
                        "replans": self.n_replans,
                        "intent": list(_intent_at(t_sim)),
                        "robot": [round(v, 3) for v in _rob],
                    }) + "\n")
            except Exception:
                pass
            tgt = _intent_at(t_sim)
            c = self.cmd
            tl = float(c.time_left[0].item()) if hasattr(c, "time_left") else -1
            print(f"[kplanner-env] t={t_sim:6.1f}s play={play} head={self.write_head} "
                  f"writes(idle={getattr(self, 'n_idle', 0)}, "
                  f"plan={getattr(self, 'n_plan', 0)}) "
                  f"intent(yaw={tgt[0]:+.1f}, fwd={tgt[2]:+.1f}) "
                  f"[steps={int(c.time_steps[0])} start={int(c.motion_start_time_steps[0])} "
                  f"time_left={tl:.2f} total={int(c.motion_lib.get_time_step_total(c.motion_ids)[0])}]",
                  flush=True)
            try:
                tm = c._env.termination_manager
                fired = [k for k in tm.active_terms if bool(tm.get_term(k)[0])]
                print(f"[kplanner-env]   window: resets={self._resets} "
                      f"fired_agg={self._fired_agg} "
                      f"ep_len={int(c._env.episode_length_buf[0])}/"
                      f"{int(c._env.max_episode_length)}", flush=True)
                self._resets, self._fired_agg = 0, {}
                rob = c._env.scene["robot"].data.root_pos_w[0].tolist()
                pos_t = getattr(c.motion_lib, self.families[-1]["pos"])
                rows = {ri: pos_t[ri, 0].tolist()
                        for ri in (0, 1, 2, 5, 10, 25, 50)}
                rows_s = " ".join(
                    f"r{ri}=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})"
                    for ri, p in rows.items())
                anchor = getattr(c, "anchor_pos_w", None)
                anc_s = ("anchor=(%+.2f,%+.2f,%+.2f)" % tuple(anchor[0].tolist())
                         if anchor is not None else "anchor=n/a")
                dof_diff = (c.motion_lib.dof_pos[1] - self.f0_dof).abs()
                print(f"[kplanner-env]   terminated={bool(tm.terminated[0])} "
                      f"fired={fired} robot=({rob[0]:+.2f},{rob[1]:+.2f},{rob[2]:+.2f}) "
                      f"{anc_s} {rows_s} "
                      f"dofdiff(max={float(dof_diff.max()):.2f},"
                      f"argmax={int(dof_diff.argmax())})", flush=True)
                qf = getattr(c.motion_lib, self.families[-1]["quat"])
                print(f"[kplanner-env]   quat_full r1={[round(v,3) for v in qf[1,0].tolist()]} "
                      f"f0={[round(v,3) for v in self.families[-1]['f0_quat'][0].tolist()]} "
                      f"dofvel_max={float(c.motion_lib.dof_vel[1:80].abs().max()):.2f} "
                      f"rootlv_max={float(c.motion_lib.root_linv_vel_w[1:80].abs().max()):.2f}",
                      flush=True)
            except Exception as e:
                print(f"[kplanner-env]   term-introspect failed: {e}", flush=True)
        # env reset jumped the playhead backwards -> reseed
        if play + self.look_n < self.write_head - 2 * self.look_n:
            pass  # reseed handled earlier in tick (with driver-state reset)


_DRIVER = None


def _wrap_commands_module(mod):
    TrackingCommand = mod.TrackingCommand
    orig_compute = TrackingCommand.compute

    def compute(self, dt):
        global _DRIVER
        orig_compute(self, dt)
        if _DRIVER is None:
            try:
                _DRIVER = _PlannerDriver(self)
            except Exception as e:  # loud, once
                import traceback
                traceback.print_exc()
                raise
        _DRIVER.tick()

    TrackingCommand.compute = compute
    print("[kplanner-env] TrackingCommand.compute wrapped", flush=True)


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

# --------------------------------------------------------------------------
# ONNX actor patch + checkpoint override, then run eval_agent_trl as __main__
# --------------------------------------------------------------------------
from gear_sonic.scripts.eval_x2_isaacsim_onnx import (  # noqa: E402
    _ensure_checkpoint_override,
    _patch_actor_with_onnx,
)

_rest = _ensure_checkpoint_override(_rest, RUN_DIR)
sys.argv = [sys.argv[0]] + _rest
_patch_actor_with_onnx(onnx_path=ONNX_PATH, encoder_name="g1",
                       compare=False, diff_csv="/dev/null")

# Dump the first 3 raw ONNX inputs (the exact 1670-d obs SONIC sees) for
# offline diff against the verified-good step-0 dump. Crude but decisive:
# wrap InferenceSession.run globally.
import numpy as _np  # noqa: E402
import onnxruntime as _ort  # noqa: E402

_OBS_DUMP_DIR = os.environ.get(
    "KP_OBS_DUMP_DIR",
    "/tmp/claude-1000/-home-stickbot-Projects-GR00T-WholeBodyControl/"
    "fa9112a6-98d6-4d9b-9c15-3150190f6aa2/scratchpad")
_orig_ort_run = _ort.InferenceSession.run
_obs_cnt = {"n": 0}


def _dumping_run(self, out_names, feed, *a, **k):
    if _obs_cnt["n"] < 3:
        for _kk, _vv in feed.items():
            _np.save(os.path.join(_OBS_DUMP_DIR, f"kp_obs{_obs_cnt['n']}.npy"),
                     _np.asarray(_vv))
        _obs_cnt["n"] += 1
        if _obs_cnt["n"] == 3:
            print("[kplanner-env] first-3 ONNX inputs dumped", flush=True)
    return _orig_ort_run(self, out_names, feed, *a, **k)


_ort.InferenceSession.run = _dumping_run

import runpy  # noqa: E402

runpy.run_module("gear_sonic.eval_agent_trl", run_name="__main__")
