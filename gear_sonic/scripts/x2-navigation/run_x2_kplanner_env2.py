#!/usr/bin/env python3
"""SONIC + the REAL x2_kplanner daemon inside the training env (lockstep).

Maximal-reuse successor to run_x2_kplanner_env.py: instead of re-implementing
planner orchestration, this embeds the battle-tested torch daemon
``x2_kplanner.py`` (the quest stack's default planner: idle anchor, quiet-stand
warmup, IDLE_LOOP, slow_walk template, command shaping, ref smoothing) with
exactly ONE adaptation: its ``time`` module is swapped for a SimClock so the
identical code paces by sim ticks instead of wall clock (RTF-independent).

Wires stay the stack's real ZMQ contract, on localhost:
  driver PUB  robot_pose  -> daemon pose-feedback SUB   (pack_robot_pose)
  driver PUB  planner_cmd -> daemon cmd SUB             (pad JSON schema)
  daemon PUB  pose stream -> driver SUB                 (build_pose_payload)

The driver writes each daemon frame (current + 9-frame future window) into
the motion-lib ring; env obs/reset/SONIC machinery untouched (bisect-proven).

Usage: same CLI as run_x2_kplanner_env.py.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.util
import json
import os
import sys
import threading
import time
from pathlib import Path

# x2-navigation/ is one level deeper than gear_sonic/scripts/ — repo root
# is FOUR dirnames up. Launch by PATH (not -m): the directory has a dash.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

_OWN = {"--onnx", "--run-dir", "--vqvae-ckpt", "--pose-ckpt", "--root-ckpt",
        "--planner-mode"}


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
PLANNER_MODE = _own.get("--planner-mode", None)  # None -> daemon default

SCRATCH = os.environ.get(
    "KP_OBS_DUMP_DIR",
    "/tmp/claude-1000/-home-stickbot-Projects-GR00T-WholeBodyControl/"
    "fa9112a6-98d6-4d9b-9c15-3150190f6aa2/scratchpad")

# Intent schedule as PAD STICK deflections (the daemon shapes/scales them
# exactly as it does for the real gamepad): (duration_s, fwd, side, yaw).
INTENT_SCHEDULE = [
    (5.0, 0.0, 0.0, 0.0),      # stand (daemon IDLE_LOOP streams the anchor)
    (3.0, 1.0, 0.0, 0.0),      # full forward stick (slow_walk template pace)
    (8.0, 0.0, 0.0, 0.0),      # stand
]
_SCHED_TOTAL = sum(s[0] for s in INTENT_SCHEDULE)

# KP_PAD=1: sticks come from a live pad_locomotion_bridge.py (unmodified,
# its default target 127.0.0.1:5563 — the driver binds a SUB there and
# relays onto the sim-clocked daemon wire). Schedule ignored; no pad -> 0.
_PAD_MODE = os.environ.get("KP_PAD", "0") == "1"
PAD_IN_PORT = 5563


def _sticks_at(t: float):
    t = t % _SCHED_TOTAL
    for dur, fwd, side, yaw in INTENT_SCHEDULE:
        if t < dur:
            return (fwd, side, yaw)
        t -= dur
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# SimClock: virtualized time for the daemon thread. monotonic() returns sim
# time; sleep() blocks until the env advances past the deadline.
# ---------------------------------------------------------------------------
class SimClock:
    def __init__(self):
        self._t = 0.0
        self._cv = threading.Condition()

    def monotonic(self) -> float:
        with self._cv:
            return self._t

    def time(self) -> float:  # some helpers use time.time for logging only
        with self._cv:
            return self._t

    def sleep(self, dt: float) -> None:
        with self._cv:
            deadline = self._t + max(0.0, float(dt))
            while self._t < deadline:
                # timeout so daemon shutdown (stop_event) isn't starved
                self._cv.wait(timeout=0.5)

    def advance(self, dt: float) -> None:
        with self._cv:
            self._t += float(dt)
            self._cv.notify_all()


SIM_CLOCK = SimClock()


class _TimeProxy:
    """Drop-in for the ``time`` module inside x2_kplanner: monotonic/sleep/
    time are simulated; everything else passes through."""

    def __init__(self, real, clock):
        self._real = real
        self._clock = clock

    def monotonic(self):
        return self._clock.monotonic()

    def time(self):
        return self._clock.time()

    def sleep(self, dt):
        self._clock.sleep(dt)

    def perf_counter(self):
        return self._clock.monotonic()

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# ZMQ ports (localhost, mirroring the quest stack's defaults but offset to
# avoid clashing with a real stack if one is running)
# ---------------------------------------------------------------------------
POSE_PUB_PORT = 6556      # daemon pose stream (stack: 5556)
CMD_PORT = 6563           # planner_cmd intents (stack: 5563)
FEEDBACK_PORT = 6570      # robot_pose feedback  (stack: 5570)


def _start_daemon_thread():
    """Import x2_kplanner with virtualized time and run() it on a thread."""
    spec = importlib.util.spec_from_file_location(
        "x2_kplanner", os.path.join(REPO_ROOT, "gear_sonic", "scripts",
                                    "x2_kplanner.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["x2_kplanner"] = mod
    mb_root = os.path.join(REPO_ROOT, "motionbricks")
    if mb_root not in sys.path:
        sys.path.insert(0, mb_root)
    spec.loader.exec_module(mod)
    mod.time = _TimeProxy(time, SIM_CLOCK)  # adaptation 1: sim-paced clock

    # adaptation 2: signal handlers are main-thread-only; the env process
    # owns SIGINT/SIGTERM, so the daemon's installer becomes a no-op.
    import signal as _real_signal

    class _SignalProxy:
        def signal(self, *_a, **_k):
            return None

        def __getattr__(self, name):
            return getattr(_real_signal, name)

    mod.signal = _SignalProxy()

    kw = dict(
        vqvae_ckpt=Path(VQVAE), pose_ckpt=Path(POSE), root_ckpt=Path(ROOT),
        warmup_qpos_path=Path(
            f"{REPO_ROOT}/gear_sonic/data/motions/kplanner_idle_anchor_g1teleop_v3.pkl"),
        pub_host="127.0.0.1", pub_port=POSE_PUB_PORT,
        pid_file=Path(SCRATCH) / "kplanner_env_daemon.pid",
        demo_yaml=None, enable_keyboard=False,
        zmq_cmd_host="127.0.0.1", zmq_cmd_port=CMD_PORT,
        zmq_cmd_topic="planner_cmd",
        duration_s=0.0, hand_dof=10, verbose=True,
        warmup_quiet_stand_s=2.0,
        body_pose_port=None, device="cuda",
        # KP_REPLAN_THRESH: frames remaining when a continuation replan
        # fires. 2 = stack default (consume ~full 64-frame chunk, ~2.1 s
        # commitment); 32 = replan at half-chunk (~1.07 s commitment,
        # fresher conditioning). Intent changes always replan immediately.
        replan_threshold_frames=int(os.environ.get("KP_REPLAN_THRESH", "2")),
        planner_mode=PLANNER_MODE,
        pose_feedback_host="127.0.0.1", pose_feedback_port=FEEDBACK_PORT,
        # Stack-parity: PLAYING reseed OFF (open-loop integration; IDLE yaw
        # refresh still active). full_root reseed dragged the planner to the
        # robot's lagging pose every couple of steps -> sideways lurch.
        pose_reseed_scope="none",
        # deploy-parity reference smoothing (the stack passes these; module
        # defaults leave it OFF -> unsmoothed walk->freeze boundary lurch)
        ref_smoother_ms=300.0, ref_smoother_trigger_rad=0.05,
        ref_smoother_shape="halfcos", ref_smoother_joints="lower_body",
    )
    th = threading.Thread(target=lambda: mod.run(**kw),
                          name="x2_kplanner_daemon", daemon=True)
    th.start()
    print(f"[kp-env2] x2_kplanner daemon thread started (sim-clocked); "
          f"pose:{POSE_PUB_PORT} cmd:{CMD_PORT} feedback:{FEEDBACK_PORT}",
          flush=True)
    return th


# ---------------------------------------------------------------------------
# Driver: ring writes from daemon payloads, robot-pose feedback, intents.
# ---------------------------------------------------------------------------
_DRIVER = None


class _Driver:
    def __init__(self, cmd):
        import numpy as np
        import torch
        import zmq

        self.np, self.torch = np, torch
        self.cmd = cmd
        lib = cmd.motion_lib
        self.lib = lib
        dev = lib._device  # noqa: SLF001
        self.dev = dev
        self.fps = float(1.0 / lib._motion_dt[0].item())  # noqa: SLF001
        self.ring_n = int(120.0 * self.fps)

        self.mj2il = np.asarray(lib.m_cfg.mujoco_to_isaaclab_dof)
        self.b_mj2il = np.asarray(lib.m_cfg.mujoco_to_isaaclab_body)

        # KP_HIDE_TERRAIN=1: hide the env terrain's VISUAL (gray plane
        # z-fights the kitchen splat floor) while keeping its collision —
        # the robot still stands on it. Runner-owned; training untouched.
        if os.environ.get("KP_HIDE_TERRAIN"):
            try:
                import omni.usd
                from pxr import UsdGeom
                stage = omni.usd.get_context().get_stage()
                prim = stage.GetPrimAtPath("/World/ground")
                if prim.IsValid():
                    UsdGeom.Imageable(prim).MakeInvisible()
                    print("[kp-env2] terrain visual hidden "
                          "(/World/ground, collision kept)", flush=True)
                else:
                    print("[kp-env2] KP_HIDE_TERRAIN: /World/ground not "
                          "found", flush=True)
            except Exception as e:
                print(f"[kp-env2] KP_HIDE_TERRAIN failed: {e}", flush=True)

        # KP_HEAD_CAM=1: head-mounted camera snapshots (M3 policy-cam
        # smoke: stereo_head_front_left stand-in — wide pinhole on the
        # head_pitch_link, slight down-pitch, per plan). Runner-owned.
        # Needs ++manager_env.config.enable_cameras=true on the CLI.
        self.head_cam = None
        if os.environ.get("KP_HEAD_CAM"):
            try:
                from isaaclab.sensors import Camera, CameraCfg
                import isaaclab.sim as _sim_utils
                self._snap_dir = os.environ.get(
                    "KP_SNAP_DIR", "/tmp/claude-1000/kp_head_cam")
                os.makedirs(self._snap_dir, exist_ok=True)
                self.head_cam = Camera(CameraCfg(
                    prim_path="/World/envs/env_0/Robot/head_pitch_link/kp_head_cam",
                    update_period=0.0, height=600, width=960,
                    data_types=["rgb"],
                    spawn=_sim_utils.PinholeCameraCfg(
                        focal_length=12.0, horizontal_aperture=36.0,
                        clipping_range=(0.05, 40.0)),
                    offset=CameraCfg.OffsetCfg(
                        pos=(0.07, 0.0, 0.05),
                        rot=(0.9848, 0.0, 0.1736, 0.0),  # 20 deg down
                        convention="world"),
                ))
                print(f"[kp-env2] head camera armed -> {self._snap_dir}",
                      flush=True)
            except Exception as e:
                self.head_cam = None
                print(f"[kp-env2] head cam failed: {e}", flush=True)
        # Real per-frame FK on the same MJCF skeleton the compare tooling
        # uses — replaces the frame-0-offset approximation whose stride-pose
        # marker constellation read as a facing mismatch.
        import mujoco
        self.mj = mujoco
        self.mj_model = mujoco.MjModel.from_xml_path(os.path.join(
            REPO_ROOT, "gear_sonic", "data", "assets", "robot_description",
            "mjcf", "x2_ultra.xml"))
        self.mj_data = mujoco.MjData(self.mj_model)
        self._setup_ring()

        # The eval wrapper reloads the motion lib (load_motions_for_
        # evaluation) AFTER env construction, clobbering the ring — rebuild
        # it on every reload so no scaffold-clip pose ever reaches an RSI.
        _orig_lme = lib.load_motions_for_evaluation

        def _reload_and_rebuild(*a, **k):
            out = _orig_lme(*a, **k)
            print("[kp-env2] motion lib reloaded — rebuilding ring + anchor "
                  "pre-fill", flush=True)
            self._setup_ring()
            return out

        lib.load_motions_for_evaluation = _reload_and_rebuild

        # ZMQ wiring (driver side)
        ctx = zmq.Context.instance()
        self.sub = ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.sub.setsockopt(zmq.RCVHWM, 1000)
        self.sub.connect(f"tcp://127.0.0.1:{POSE_PUB_PORT}")
        self.cmd_pub = ctx.socket(zmq.PUB)
        self.cmd_pub.bind(f"tcp://127.0.0.1:{CMD_PORT}")
        self.fb_pub = ctx.socket(zmq.PUB)
        self.fb_pub.bind(f"tcp://127.0.0.1:{FEEDBACK_PORT}")
        from gear_sonic.utils.teleop.zmq.robot_pose_zmq import pack_robot_pose
        self.pack_robot_pose = pack_robot_pose
        from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import (
            unpack_message,
        )
        self.unpack_pose = unpack_message

        _start_daemon_thread()

        # Adaptive sampling bins were sized for the ORIGINAL clip length at
        # load; ring-grown time steps overflow torch.bincount -> CUDA
        # device-side assert (root cause of the kpe2_2/3/4 + kp24 crashes).
        # It's a training-curriculum feature — off for planner-driven eval.
        cmd.use_adaptive_sampling = False

        self.write_head = 0
        self.prev_dof_il = None
        self.prev_trans = None
        self.ticks = 0
        self.n_frames = 0
        self._pb = None          # last raw payload (x, y, yaw) for rebase
        self._base_dyaw = 0.0
        self._base_xy = (0.0, 0.0)
        # gamepad input relay (KP_PAD=1): SUB bound where the unmodified
        # pad_locomotion_bridge PUB-connects by default
        self.pad_sticks = (0.0, 0.0, 0.0)
        self.pad_wall = 0.0
        if _PAD_MODE:
            self.pad_sub = ctx.socket(zmq.SUB)
            self.pad_sub.setsockopt(zmq.SUBSCRIBE, b"planner_cmd")
            self.pad_sub.bind(f"tcp://127.0.0.1:{PAD_IN_PORT}")
            print(f"[kp-env2] PAD MODE: listening for pad_locomotion_bridge "
                  f"on :{PAD_IN_PORT} (schedule disabled)", flush=True)

        self.last_sticks = None
        self.wall0 = time.monotonic()
        self.tl_path = os.path.join(SCRATCH, "kp_timeline.jsonl")
        print(f"[kp-env2] driver up: ring {self.ring_n} rows @ {self.fps:.0f} Hz",
              flush=True)

    # -- ring construction (re-run after every motion-lib reload) ----------
    def _setup_ring(self):
        np, torch = self.np, self.torch
        lib = self.lib
        f0 = int(lib.length_starts[0].item())

        # Snapshot the scaffold clip's NATIVE lib rows (already resampled to
        # lib fps, body tensors from the lib's own loader — the exact data
        # that stood +-1cm in the ring-idle bisect) BEFORE ring growth and
        # anchor pre-fill wipe them. The idle-writer replays these verbatim,
        # yaw/xy-shifted — no hand-rolled FK anywhere in the idle path.
        if not hasattr(self, "_idle_snap"):
            n0 = int(lib._motion_num_frames[0].item())    # noqa: SLF001
            snap = {}
            for name in ("dof_pos", "dof_vel", "root_linv_vel_w",
                         "root_ang_vel_w", "body_pos_b", "feet_l", "feet_r",
                         "body_pos_w", "body_quat_w", "body_lin_vel_w",
                         "body_ang_vel_w", "body_pos_w_full",
                         "body_quat_w_full", "body_lin_vel_w_full",
                         "body_ang_vel_w_full"):
                t = getattr(lib, name, None)
                if t is not None:
                    snap[name] = t[f0:f0 + n0].clone()
            self._idle_snap, self._idle_n0 = snap, n0
            print(f"[kp-env2] idle snapshot: {n0} native rows "
                  f"({len(snap)} tensors)", flush=True)
            # ONE-TIME CONVENTION DIAGNOSTIC: recompute frame 0 by MuJoCo FK
            # from the scaffold pkl and diff against the lib's NATIVE rows
            # under each body-order / quat-convention hypothesis. Prints the
            # truth; no more guessed conventions in _write_row.
            try:
                import joblib as _jl
                cfg_pkl = ""
                for attr in ("motion_lib_cfg", "cfg", "_cfg"):
                    c = getattr(lib, attr, None)
                    if c is not None and getattr(c, "motion_file", None):
                        cfg_pkl = str(c.motion_file)
                        break
                if not cfg_pkl or not os.path.exists(cfg_pkl):
                    cfg_pkl = os.path.join(
                        REPO_ROOT, "gear_sonic", "data", "motions",
                        "x2_sonic_feasible_stand_single.pkl")
                _mrec = next(iter(_jl.load(cfg_pkl).values()))
                jp0 = np.asarray(_mrec["dof"], np.float64)[0]
                tr0 = np.asarray(_mrec["root_trans_offset"], np.float64)[0]
                q0 = np.asarray(_mrec["root_rot"], np.float64)[0]   # xyzw
                d = self.mj_data
                d.qpos[0:3] = tr0
                d.qpos[3:7] = [q0[3], q0[0], q0[1], q0[2]]          # wxyz
                d.qpos[7:38] = jp0
                self.mj.mj_kinematics(self.mj_model, d)
                xpos = np.asarray(d.xpos[1:])
                xquat = np.asarray(d.xquat[1:])                     # wxyz, MJ order
                # The eval loader re-bases motions (env-origin translation +
                # yaw seen as ~20m offsets in diag v1), so compare
                # translation-invariantly and fit the yaw (Procrustes about
                # z) per order hypothesis; then test quat conventions with
                # the fitted yaw composed in.
                import math as _dm

                def _yawfit(fk, nat):
                    a = fk[:, :2] - fk[:, :2].mean(0)
                    b = nat[:, :2] - nat[:, :2].mean(0)
                    yw = _dm.atan2(float((a[:, 0] * b[:, 1]
                                          - a[:, 1] * b[:, 0]).sum()),
                                   float((a[:, 0] * b[:, 0]
                                          + a[:, 1] * b[:, 1]).sum()))
                    c, s = _dm.cos(yw), _dm.sin(yw)
                    rot = np.stack((c * a[:, 0] - s * a[:, 1],
                                    s * a[:, 0] + c * a[:, 1]), axis=1)
                    res = float(np.abs(rot - b).max())
                    ez = float(np.abs((fk[:, 2] - fk[:, 2].mean())
                                      - (nat[:, 2] - nat[:, 2].mean())).max())
                    return yw, max(res, ez)

                for fam_name, conv_lbl in (("body_pos_w", "base"),
                                           ("body_pos_w_full", "full")):
                    if fam_name not in snap:
                        continue
                    nat_p = snap[fam_name][0].cpu().numpy()
                    nb = nat_p.shape[0]
                    yw_mj, e_mj = _yawfit(xpos[:nb], nat_p)
                    yw_il, e_il = _yawfit(xpos[self.b_mj2il[:nb]], nat_p)
                    use_il = e_il < e_mj
                    sel = self.b_mj2il[:nb] if use_il else np.arange(nb)
                    yw = yw_il if use_il else yw_mj
                    hzc, bzc = _dm.cos(yw / 2.0), _dm.sin(yw / 2.0)
                    xq = xquat[sel]                          # wxyz, FK
                    # Rz(yw) ⊗ q in wxyz
                    xqr = np.stack(
                        (hzc * xq[:, 0] - bzc * xq[:, 3],
                         hzc * xq[:, 1] - bzc * xq[:, 2],
                         hzc * xq[:, 2] + bzc * xq[:, 1],
                         hzc * xq[:, 3] + bzc * xq[:, 0]), axis=1)
                    qn = snap[fam_name.replace("pos", "quat")][0].cpu().numpy()
                    e_wxyz = min(float(np.abs(xqr - qn).max()),
                                 float(np.abs(-xqr - qn).max()))
                    e_xyzw = min(
                        float(np.abs(xqr[:, [1, 2, 3, 0]] - qn).max()),
                        float(np.abs(-xqr[:, [1, 2, 3, 0]] - qn).max()))
                    print(f"[kp-env2] DIAG {conv_lbl}: order="
                          f"{'IL' if use_il else 'MJ'} "
                          f"(res il={e_il:.4f} mj={e_mj:.4f}) loader_yaw="
                          f"{yw:+.3f} quat="
                          f"{'wxyz' if e_wxyz < e_xyzw else 'xyzw'} "
                          f"(err wxyz={e_wxyz:.4f} xyzw={e_xyzw:.4f})",
                          flush=True)
            except Exception as e:
                print(f"[kp-env2] DIAG failed: {e}", flush=True)

        def grow(t):
            return t[f0].unsqueeze(0).repeat(
                (self.ring_n,) + (1,) * (t.dim() - 1)).clone()

        self.families = []
        for pos_n, quat_n, lv_n, av_n, conv in (
            ("body_pos_w", "body_quat_w",
             "body_lin_vel_w", "body_ang_vel_w", "xyzw"),
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
        for name in (["body_pos_b", "root_linv_vel_w", "root_ang_vel_w",
                      "dof_vel", "dof_pos", "feet_l", "feet_r"]
                     + [k for fam in self.families
                        for k in (fam["pos"], fam["quat"],
                                  fam["lv"], fam["av"])]):
            t = getattr(lib, name, None)
            if t is not None:
                setattr(lib, name, grow(t))
        lib.length_starts = torch.zeros_like(lib.length_starts)
        lib._motion_num_frames[:] = self.ring_n           # noqa: SLF001
        lib._motion_lengths[:] = self.ring_n / self.fps   # noqa: SLF001

        # anchor pre-fill: every row = the canonical hands-at-hips idle
        # anchor, so any RSI (first episode included) lands on it.
        if not hasattr(self, "_anchor"):
            import joblib
            anchor = joblib.load(os.path.join(
                REPO_ROOT, "gear_sonic", "data", "motions",
                "kplanner_idle_anchor_g1teleop_v3.pkl"))
            ainner = anchor[list(anchor.keys())[0]]
            self._anchor = (
                np.asarray(ainner["dof"]).reshape(-1, 31)[0],
                np.asarray(ainner["root_trans_offset"]).reshape(-1, 3)[0],
                np.asarray(ainner["root_rot"]).reshape(-1, 4)[0],  # xyzw
            )
        a_dof, a_trans, a_rot = self._anchor
        self.prev_dof_il = None
        self.prev_trans = None
        self._write_row(0, a_dof, a_trans, a_rot, 0.0)
        for name in (["dof_pos", "dof_vel", "root_linv_vel_w",
                      "root_ang_vel_w"]
                     + [k for fam in self.families
                        for k in (fam["pos"], fam["quat"],
                                  fam["lv"], fam["av"])]):
            t = getattr(lib, name)
            t[:] = t[0]
        self.prev_dof_il = None
        self.prev_trans = None
        self._chain_cur = None
        self.write_head = 0
        print("[kp-env2] ring built + anchor pre-fill (hands-at-hips)",
              flush=True)

    # -- helpers -----------------------------------------------------------
    def _yaw_of(self, q_wxyz):
        import math
        w, x, y, z = q_wxyz
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _write_row(self, row, dof_mj, trans, quat_xyzw, dt, prev="LEGACY"):
        """prev: None (no velocity), or (t_dof, t_trans) of the frame this
        one follows at spacing dt. Returns (t_dof, t_trans) for chaining.
        Separate chains for current-rate rows vs 0.1s future rows — a single
        shared chain made every current row's dof_vel = (cur - future_0.9s)
        / 0.02 = ~45x garbage in SONIC's velocity channel (user-suspected
        'kplanner output not reaching SONIC properly'; confirmed)."""
        import math
        np, torch = self.np, self.torch
        lib, dev = self.lib, self.dev
        r = row % self.ring_n
        dof_il = np.asarray(dof_mj, dtype=np.float32)[self.mj2il]
        t_dof = torch.tensor(dof_il, dtype=torch.float32, device=dev)
        t_trans = torch.tensor(np.asarray(trans, dtype=np.float32), device=dev)
        q_xyzw = torch.tensor(np.asarray(quat_xyzw, dtype=np.float32), device=dev)
        q_wxyz = q_xyzw[[3, 0, 1, 2]]
        lib.dof_pos[r] = t_dof
        if prev == "LEGACY":
            prev = (self.prev_dof_il, self.prev_trans)
        if prev is not None and prev[0] is not None and dt > 0:
            lib.dof_vel[r] = (t_dof - prev[0]) / dt
            lib.root_linv_vel_w[r] = (t_trans - prev[1]) / dt
        # True FK on the deploy MJCF: exact reference body poses per frame
        # (the frame-0-offset approximation put ref bodies in a stride
        # constellation facing off from the root).
        d = self.mj_data
        d.qpos[0:3] = np.asarray(trans, dtype=np.float64)
        d.qpos[3:7] = q_wxyz.cpu().numpy().astype(np.float64)
        d.qpos[7:38] = np.asarray(dof_mj, dtype=np.float64)
        self.mj.mj_kinematics(self.mj_model, d)
        xpos = torch.tensor(np.asarray(d.xpos[1:], dtype=np.float32),
                            device=dev)
        xquat = torch.tensor(np.asarray(d.xquat[1:], dtype=np.float32),
                             device=dev)  # wxyz, mujoco body order
        # Loader ground truth (motion_lib_base.py:1729): _full = ALL bodies
        # in IL order, wxyz; base = _full sliced by m_cfg.body_indexes_data
        # (tracked subset), ALSO IL-order/wxyz. The old base write (first-nb
        # mujoco bodies, xyzw) fed garbage tracked-body refs — hands/head
        # vr_3point targets included — into the tokenizer on every
        # planner-path row (root cause of arms-folding + idle collapse).
        pos_full = xpos[self.b_mj2il]
        quat_full = xquat[self.b_mj2il]            # wxyz
        if not hasattr(self, "_body_idx"):
            bi = getattr(lib, "body_indexes", None)
            if bi is None:
                self._body_idx = None
            else:
                self._body_idx = np.asarray(
                    bi.cpu().numpy() if hasattr(bi, "cpu") else bi)
        for fam in self.families:
            if fam["pos"].endswith("_full"):
                pos, quat = pos_full, quat_full
            elif self._body_idx is not None:
                pos = pos_full[self._body_idx]
                quat = quat_full[self._body_idx]
            else:
                pos, quat = pos_full, quat_full
            getattr(lib, fam["pos"])[r] = pos
            getattr(lib, fam["quat"])[r] = quat
        self.prev_dof_il = t_dof
        self.prev_trans = t_trans
        return (t_dof, t_trans)

    # -- per-env-step tick -------------------------------------------------
    def tick(self):
        import zmq
        np = self.np
        self.ticks += 1
        t_sim = self.ticks * 0.02
        play = int((self.cmd.motion_start_time_steps[0]
                    + self.cmd.time_steps[0]).item())

        # 1. publish robot pose feedback (daemon reseeds/anchors from this)
        try:
            robot = self.cmd._env.scene["robot"]
            root = robot.data.root_state_w[0].cpu().numpy()
            org = self.cmd._env.scene.env_origins[0].cpu().numpy()
            pelvis = [float(root[0] - org[0]), float(root[1] - org[1]),
                      float(root[2] - org[2]), float(root[3]), float(root[4]),
                      float(root[5]), float(root[6])]  # x,y,z,qw,qx,qy,qz
            self.fb_pub.send(
                self.pack_robot_pose(t_sim, pelvis))
        except Exception:
            pass

        # 2. publish scheduled sticks (pad wire schema), on change + 1 Hz beat
        # Env reset detection (playhead rewound): fire a 0.3 s micro
        # yaw-stick pulse. The daemon's own full_root closed-loop reseed
        # (active during PLAYING) snaps its planner pose to the robot's
        # respawn pose from our feedback; the following freeze re-anchors
        # IDLE right there. Daemon-native re-anchoring, no custom math.
        if not hasattr(self, "_last_play"):
            self._last_play = play
            self._pulse_until = -1
        # play only ever advances +1/tick; ANY rewind is an env reset. (The
        # old -100 threshold missed back-to-back early resets -> stale held
        # pose -> silent reset loop.)
        if play < self._last_play:
            self._pulse_until = self.ticks + 15
            self.write_head = play
            self.prev_dof_il = None
            self.prev_trans = None
            self._chain_cur = None
            # Full re-anchor at the RESPAWN pose (user rule): ring rows —
            # and therefore the yellow markers — must restart where the
            # robot actually is, not at the crash site. Place the anchor
            # pose at the robot's spawn xy/yaw and flood the ring; force
            # the next payload to re-capture the world base mapping.
            try:
                import math as _m
                rob = self.cmd._env.scene["robot"].data.root_state_w[
                    0].cpu().numpy()
                org = self.cmd._env.scene.env_origins[0].cpu().numpy()
                r_xy = rob[0:2] - org[0:2]
                r_yaw = _m.atan2(2.0 * (rob[3] * rob[6] + rob[4] * rob[5]),
                                 1.0 - 2.0 * (rob[5] ** 2 + rob[6] ** 2))
                a_dof, a_trans, a_rot = self._anchor
                tr = self.np.array([r_xy[0], r_xy[1], float(a_trans[2])],
                                   dtype=self.np.float32)
                qx = self.np.array([0.0, 0.0, _m.sin(r_yaw / 2.0),
                                    _m.cos(r_yaw / 2.0)],
                                   dtype=self.np.float32)
                self._write_row(0, a_dof, tr, qx, 0.0, prev=None)
                lib = self.lib
                for name in (["dof_pos", "dof_vel", "root_linv_vel_w",
                              "root_ang_vel_w"]
                             + [k for fam in self.families
                                for k in (fam["pos"], fam["quat"],
                                          fam["lv"], fam["av"])]):
                    t = getattr(lib, name)
                    t[:] = t[0]
                self._pb = None          # force base re-capture next payload
                self._quiet_ticks = 0
                self._idle_active = False
                print(f"[kp-env2] env reset (play {self._last_play}->{play})"
                      f" — ring+markers re-anchored at spawn "
                      f"({r_xy[0]:+.2f},{r_xy[1]:+.2f}, yaw {r_yaw:+.2f})",
                      flush=True)
            except Exception as e:
                print(f"[kp-env2] reset re-anchor failed: {e}", flush=True)
        self._last_play = play

        if _PAD_MODE:
            # drain latest pad frame(s); pad liveness judged on wall clock
            # (it's a real input device at wall rate; sim consumes latest)
            import zmq as _zmq
            while True:
                try:
                    p = self.pad_sub.recv_multipart(flags=_zmq.NOBLOCK)
                except _zmq.Again:
                    break
                try:
                    d = json.loads(p[-1].decode())
                    self.pad_sticks = (float(d.get("stick_fwd", 0.0)),
                                       float(d.get("stick_side", 0.0)),
                                       float(d.get("stick_yaw", 0.0)))
                    self.pad_wall = time.monotonic()
                except Exception:
                    continue
            fresh = (time.monotonic() - self.pad_wall) < 1.0
            sticks = self.pad_sticks if fresh else (0.0, 0.0, 0.0)
        else:
            sticks = _sticks_at(t_sim)
        if self.ticks < self._pulse_until:
            sticks = (0.0, 0.0, 0.15)
        if sticks != self.last_sticks or self.ticks % 50 == 0:
            payload = {"intent": "locomotion", "magnitude": "continuous",
                       "stick_fwd": sticks[0], "stick_side": sticks[1],
                       "stick_yaw": sticks[2]}
            self.cmd_pub.send_multipart(
                [b"planner_cmd", json.dumps(payload).encode()])
            if sticks != self.last_sticks:
                print(f"[kp-env2] CMD @sim={t_sim:.2f}s -> {payload}",
                      flush=True)
            self.last_sticks = sticks

        # 3. advance the daemon's clock one env step
        SIM_CLOCK.advance(0.02)

        # 3b. IDLE LIVE-CLIP WRITER: the daemon's IDLE_LOOP emits one frozen
        # anchor frame forever; Isaac-SONIC on a statue reference enters a
        # lateral weight-shift limit cycle and slowly sinks (root-caused
        # 2026-07-21: live-clip refs stand +-1cm, frozen refs drift). While
        # the pad is quiet, stream the planner's own idle template clip
        # (Idle_Right_001__A019 — same clip the deployed planner anchors
        # idle on, and this env's spawn scaffold) through the ring at the
        # held pose. Any stick input hands the ring back to daemon frames.
        import math as _m2
        quiet = all(abs(s) < 0.02 for s in sticks)
        self._quiet_ticks = (getattr(self, "_quiet_ticks", 0) + 1) if quiet else 0
        if not quiet:
            if getattr(self, "_idle_active", False):
                self._chain_cur = None   # payload rows restart their vel chain
            self._idle_active = False
        elif self._quiet_ticks == 75:            # 1.5 s quiet -> take over
            if not hasattr(self, "_idle_clip"):
                import joblib as _jl
                _mrec = next(iter(_jl.load(
                    f"{REPO_ROOT}/gear_sonic/data/motions/"
                    f"x2_sonic_feasible_stand_single.pkl").values()))
                q0 = np.asarray(_mrec["root_rot"], np.float32)[0]  # xyzw
                self._idle_clip = dict(
                    dof=np.asarray(_mrec["dof"], dtype=np.float32),
                    quat=np.asarray(_mrec["root_rot"], dtype=np.float32),
                    trans=np.asarray(_mrec["root_trans_offset"],
                                     dtype=np.float32),
                    fps=float(_mrec.get("fps", 50.0)),
                    yaw0=_m2.atan2(
                        2.0 * (q0[3] * q0[2] + q0[0] * q0[1]),
                        1.0 - 2.0 * (q0[1] ** 2 + q0[2] ** 2)))
            rob = self.cmd._env.scene["robot"].data.root_state_w[0].cpu().numpy()
            org = self.cmd._env.scene.env_origins[0].cpu().numpy()
            h_yaw = _m2.atan2(2.0 * (rob[3] * rob[6] + rob[4] * rob[5]),
                              1.0 - 2.0 * (rob[5] ** 2 + rob[6] ** 2))
            self._idle_held = (float(rob[0] - org[0]), float(rob[1] - org[1]),
                               h_yaw)
            self._idle_row0 = play + 1
            self._idle_next = play + 1
            self._idle_active = True
            print(f"[kp-env2] idle-writer ON (SONIC-executed stand; held xy="
                  f"({self._idle_held[0]:+.2f},{self._idle_held[1]:+.2f}) "
                  f"yaw={h_yaw:+.2f} dyaw="
                  f"{h_yaw - self._idle_clip['yaw0']:+.2f})", flush=True)
        if getattr(self, "_idle_active", False):
            c = self._idle_clip
            n_c = c["dof"].shape[0]
            dyaw = self._idle_held[2] - c["yaw0"]
            cd, sd = _m2.cos(dyaw), _m2.sin(dyaw)
            hz, bz = _m2.cos(dyaw / 2.0), _m2.sin(dyaw / 2.0)
            self._idle_next = max(self._idle_next, play + 1)
            while self._idle_next <= play + 50:
                row = self._idle_next
                tf = ((row - self._idle_row0) / 50.0 * c["fps"]) % (n_c - 1)
                i0 = int(tf)
                a = tf - i0
                dof = (1 - a) * c["dof"][i0] + a * c["dof"][i0 + 1]
                trn = (1 - a) * c["trans"][i0] + a * c["trans"][i0 + 1]
                q = c["quat"][i0 if a < 0.5 else i0 + 1]
                dx = trn[0] - c["trans"][0][0]
                dy = trn[1] - c["trans"][0][1]
                tr = np.array([self._idle_held[0] + cd * dx - sd * dy,
                               self._idle_held[1] + sd * dx + cd * dy,
                               trn[2]], dtype=np.float32)
                qr = np.array([hz * q[0] - bz * q[1], hz * q[1] + bz * q[0],
                               hz * q[2] + bz * q[3], hz * q[3] - bz * q[2]],
                              dtype=np.float32)   # Rz(dyaw) ⊗ q, xyzw
                self._chain_cur = self._write_row(row, dof, tr, qr, 0.02,
                                                  prev=self._chain_cur)
                self.write_head = max(self.write_head, row)
                self._idle_next += 1

        # 4. drain daemon frames -> ring (current row + future window rows)
        while True:
            try:
                parts = self.sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                fields = self.unpack_pose(parts[-1],
                                          expected_topic="pose").fields
            except Exception:
                try:
                    fields = self.unpack_pose(parts[-1]).fields
                except Exception:
                    continue
            if "joint_pos_mj" not in fields:
                continue
            self.n_frames += 1
            jp = np.asarray(fields["joint_pos_mj"], dtype=np.float32).reshape(-1)
            qx = np.asarray(fields["root_quat_xyzw"],
                            dtype=np.float32).reshape(-1)
            xy = np.asarray(fields.get("root_xy_world", [0.0, 0.0]),
                            dtype=np.float32).reshape(-1)
            z = float(np.asarray(fields.get("root_z_world", [0.665]),
                                 dtype=np.float32).reshape(-1)[0])

            # REBASE: the planner's pose fields live in ITS OWN integrated
            # frame ("advisory, for diagnostics" per StreamFrame) — the
            # deploy chain never notices (tokenizer eats diffs) but the
            # env's anchor obs compare absolutely. Capture a yaw+XY base
            # from the ROBOT's actual pose whenever the payload frame
            # jumps (PLAYING seed / reset), compose every frame through it.
            import math as _m
            p_yaw = _m.atan2(2.0 * (qx[3] * qx[2] + qx[0] * qx[1]),
                             1.0 - 2.0 * (qx[1] ** 2 + qx[2] ** 2))
            jump = (self._pb is None
                    or abs(_m.atan2(_m.sin(p_yaw - self._pb[2]),
                                    _m.cos(p_yaw - self._pb[2]))) > 1.2
                    or _m.hypot(xy[0] - self._pb[0], xy[1] - self._pb[1]) > 0.5)
            if jump:
                rob = self.cmd._env.scene["robot"].data.root_state_w[
                    0].cpu().numpy()
                org = self.cmd._env.scene.env_origins[0].cpu().numpy()
                r_xy = rob[0:2] - org[0:2]
                r_yaw = _m.atan2(2.0 * (rob[3] * rob[6] + rob[4] * rob[5]),
                                 1.0 - 2.0 * (rob[5] ** 2 + rob[6] ** 2))
                self._base_dyaw = _m.atan2(_m.sin(r_yaw - p_yaw),
                                           _m.cos(r_yaw - p_yaw))
                c0, s0 = _m.cos(self._base_dyaw), _m.sin(self._base_dyaw)
                self._base_xy = (r_xy[0] - (c0 * xy[0] - s0 * xy[1]),
                                 r_xy[1] - (s0 * xy[0] + c0 * xy[1]))
                print(f"[kp-env2] REBASE: payload jump -> base_dyaw="
                      f"{self._base_dyaw:+.2f} base_xy=({self._base_xy[0]:+.2f},"
                      f"{self._base_xy[1]:+.2f})", flush=True)
            self._pb = (float(xy[0]), float(xy[1]), float(p_yaw))

            if getattr(self, "_idle_active", False):
                continue   # ring belongs to the idle-writer; the daemon is
                           # only re-sending its frozen anchor frame anyway

            def _rebase(xy_p, q_xyzw_p):
                c0, s0 = _m.cos(self._base_dyaw), _m.sin(self._base_dyaw)
                wx = c0 * xy_p[0] - s0 * xy_p[1] + self._base_xy[0]
                wy = s0 * xy_p[0] + c0 * xy_p[1] + self._base_xy[1]
                h = 0.5 * self._base_dyaw
                bw, bz = _m.cos(h), _m.sin(h)
                x2, y2, z2, w2 = q_xyzw_p
                qn = np.array([bw * x2 - bz * y2, bw * y2 + bz * x2,
                               bw * z2 + bz * w2, bw * w2 - bz * z2],
                              dtype=np.float32)  # yawq ⊗ q, xyzw
                return np.array([wx, wy], dtype=np.float32), qn

            xy_w, qx = _rebase(xy, qx)
            tr = np.array([xy_w[0], xy_w[1], z], dtype=np.float32)
            self.write_head = max(self.write_head, play + 1)
            # Two velocity chains (see _write_row docstring): current rows
            # chain against the previous current row at ring rate; future
            # rows chain current->f0->f1... at their true 0.1 s spacing.
            cur = self._write_row(self.write_head, jp, tr, qx,
                                  1.0 / self.fps,
                                  prev=getattr(self, "_chain_cur", None))
            self._chain_cur = cur
            jw = fields.get("joint_pos_mj_future")
            qw = fields.get("root_quat_xyzw_future")
            if jw is not None:
                jw = np.asarray(jw, dtype=np.float32).reshape(-1, 31)
                qw = np.asarray(qw, dtype=np.float32).reshape(-1, 4)
                step5 = int(round(0.1 * self.fps))
                pv = cur
                for k in range(jw.shape[0]):
                    _, qk = _rebase(np.zeros(2, dtype=np.float32), qw[k])
                    pv = self._write_row(self.write_head + (k + 1) * step5,
                                         jw[k], tr, qk, 0.1, prev=pv)
            self.write_head += 1

        # 5. fall reset via the env's own motion timeout
        try:
            rob_z = float(self.cmd._env.scene["robot"].data.root_pos_w[0, 2])
            if rob_z < 0.35:
                total = int(self.cmd.motion_lib.get_time_step_total(
                    self.cmd.motion_ids)[0])
                # -46 keeps same-step future queries in-bounds (see init)
                self.cmd.time_steps[:] = max(0, total - 46)
        except Exception:
            pass

        # 5b. head camera snapshot every 2 s (and every tick is too costly)
        if self.head_cam is not None and self.ticks % 100 == 0:
            try:
                self.head_cam.update(0.02)
                rgb = self.head_cam.data.output["rgb"][0].cpu().numpy()
                from PIL import Image
                Image.fromarray(rgb[..., :3]).save(
                    f"{self._snap_dir}/cam_{t_sim:07.1f}s.png")
            except Exception as e:
                print(f"[kp-env2] head cam capture failed: {e}", flush=True)
                self.head_cam = None

        # 5c. waypoint labeling: `echo <name> > /tmp/claude-1000/kp_label`
        # captures the robot's CURRENT pose (kitchen-frame: world minus
        # world_pos so labels survive env-origin changes) into the
        # x2-kitchen-sim waypoint registry. Teleop-as-labeler workflow.
        if self.ticks % 50 == 0:
            try:
                lf = "/tmp/claude-1000/kp_label"
                mt = os.path.getmtime(lf) if os.path.exists(lf) else 0
                if mt > getattr(self, "_label_mtime", 0):
                    self._label_mtime = mt
                    with open(lf) as fh:
                        wp_name = fh.read().strip()
                    if wp_name:
                        import math as _lm
                        rob = self.cmd._env.scene["robot"].data.root_state_w[
                            0].cpu().numpy()
                        wyaw = _lm.atan2(
                            2.0 * (rob[3] * rob[6] + rob[4] * rob[5]),
                            1.0 - 2.0 * (rob[5] ** 2 + rob[6] ** 2))
                        wp_path = ("/home/stickbot/projects/x2-kitchen-sim/"
                                   "configs/waypoints.json")
                        os.makedirs(os.path.dirname(wp_path), exist_ok=True)
                        try:
                            with open(wp_path) as fh:
                                reg = json.load(fh)
                        except Exception:
                            reg = {}
                        wpos = [float(v) for v in os.environ.get(
                            "KP_WORLD_POS", "-19.99,-75.96,0").split(",")]
                        reg[wp_name] = {
                            "xy": [round(float(rob[0]) - wpos[0], 3),
                                   round(float(rob[1]) - wpos[1], 3)],
                            "yaw": round(wyaw, 3), "radius": 0.4,
                            "frame": "kitchen(world-world_pos)"}
                        with open(wp_path, "w") as fh:
                            json.dump(reg, fh, indent=2, sort_keys=True)
                        print(f"[kp-env2] WAYPOINT '{wp_name}' captured: "
                              f"xy={reg[wp_name]['xy']} "
                              f"yaw={reg[wp_name]['yaw']:+.2f} -> {wp_path}",
                              flush=True)
            except Exception as e:
                print(f"[kp-env2] waypoint label failed: {e}", flush=True)

        # 6. heartbeat + timeline
        if self.ticks % 50 == 0:
            try:
                rob = self.cmd._env.scene["robot"].data.root_pos_w[0].tolist()
                rq = self.cmd._env.scene["robot"].data.root_state_w[
                    0, 3:7].tolist()  # wxyz
                fam = self.families[-1]
                ring_q = getattr(self.lib, fam["quat"])[play, 0].tolist()
                if fam["conv"] == "xyzw":
                    ring_q = [ring_q[3], ring_q[0], ring_q[1], ring_q[2]]
                rec = {"wall_s": round(time.monotonic() - self.wall0, 2),
                       "sim_s": round(t_sim, 2), "play": play,
                       "head": self.write_head, "frames": self.n_frames,
                       "sticks": list(sticks),
                       "robot": [round(v, 3) for v in rob],
                       "yaw_robot": round(self._yaw_of(rq), 2),
                       "yaw_ref": round(self._yaw_of(ring_q), 2)}
                print(f"[kp-env2] {json.dumps(rec)}", flush=True)
                with open(self.tl_path, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            except Exception:
                pass


def _wrap_commands_module(mod):
    TrackingCommand = mod.TrackingCommand
    orig_init = TrackingCommand.__init__
    orig_compute = TrackingCommand.compute

    def __init__(self, *a, **k):
        # Build the driver HERE: motion lib is loaded but the env has not
        # yet run its first RSI, so the ring's anchor pre-fill is what the
        # very first episode spawns on (no hands-behind pose flash).
        orig_init(self, *a, **k)
        global _DRIVER
        if _DRIVER is None:
            try:
                _DRIVER = _Driver(self)
            except Exception:
                import traceback
                traceback.print_exc()
                raise

    def compute(self, dt):
        orig_compute(self, dt)
        if _DRIVER is not None:
            _DRIVER.tick()

    TrackingCommand.__init__ = __init__
    TrackingCommand.compute = compute
    print("[kp-env2] TrackingCommand init+compute wrapped", flush=True)


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

import runpy  # noqa: E402

runpy.run_module("gear_sonic.eval_agent_trl", run_name="__main__")
