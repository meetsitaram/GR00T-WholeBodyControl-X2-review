#!/usr/bin/env python3
"""Browse curated X2 planner primitives in a MuJoCo viewer.

This is the "soma retarget viewer" for the heuristic planner: load any
curated primitive (or any candidate from the report), play it kinematically
on the X2 Ultra MJCF, and flip through them with N / P like a slideshow.

Three input sources, all backed by the same player loop:

  1. ``--bin BIN_NAME``       : play the curator-selected primitive for one bin.
  2. ``--all``                : cycle through every primitive alphabetically;
                                press ``N`` / ``P`` in the viewer to step.
  3. ``--motion-key KEY``     : raw clip from the source bones-seed PKL,
       ``--start S --n N``      typically used to preview a candidate listed
                                in ``x2_planner_primitives_report.md`` before
                                pinning it into the registry.

By default the pelvis follows the clip's actual world XY so you see real
distance crossed (matches what SONIC would track at deploy). ``--anchor-xy``
locks XY to origin if a clip drifts off-screen.

Examples (from the repo root)::

    # Preview the curated 90-degree right turn (kinematic only):
    .venv/bin/python -m gear_sonic.scripts.browse_x2_planner_primitives \\
        --bin turn_right_90deg

    # Cycle through every primitive (N/P navigates):
    .venv/bin/python -m gear_sonic.scripts.browse_x2_planner_primitives --all

    # Audition a candidate from the markdown report before pinning:
    .venv/bin/python -m gear_sonic.scripts.browse_x2_planner_primitives \\
        --motion-key loco__walk_backward_loop_007__A026 --start 658 --n 58

    # Preview a candidate listed in the report by report rank:
    .venv/bin/python -m gear_sonic.scripts.browse_x2_planner_primitives \\
        --bin fwd_step_half_ft --candidate 3

    # Browse one primitive at a time WITH SONIC PHYSICS in the loop. The
    # browser becomes a ZMQ pose publisher; the SONIC sim deploy is spawned
    # as a child docker container and its MuJoCo window IS the visual.
    # N/P/R/SPACE/L/X navigate from the terminal you launched this in.
    # --sonic-checkpoint is REQUIRED so we can locate the .onnx model bundle.
    .venv/bin/python -m gear_sonic.scripts.browse_x2_planner_primitives \\
        --all --with-sonic \\
        --sonic-checkpoint ~/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt

    # Same, but auto-advance through every primitive instead of looping one:
    .venv/bin/python -m gear_sonic.scripts.browse_x2_planner_primitives \\
        --all --with-sonic --sonic-auto-advance \\
        --sonic-checkpoint ~/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt

Kinematic viewer keys (default mode):
  SPACE       pause / resume
  R           restart current clip
  N / P       next / previous clip (only in --all mode)
  LEFT/RIGHT  scrub by 10 frames (paused only)
  X / ESC     quit

SONIC sim mode keys (--with-sonic; type in the launching terminal):
  N / P       next / previous primitive
  R           restart current primitive
  SPACE       pause publishing (SONIC tracks the held frame)
  L           toggle loop-one vs auto-advance-all
  X / ESC     quit (cleans up the docker child)
  ?           re-print help
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

GEAR_SONIC_ROOT = Path(__file__).resolve().parents[2]
if str(GEAR_SONIC_ROOT) not in sys.path:
    sys.path.insert(0, str(GEAR_SONIC_ROOT))

from gear_sonic.utils.planner.constants import (  # noqa: E402
    DEFAULT_PELVIS_Z_M,
    NUM_BODY_DOFS,
)


MJCF_PATH = str(
    GEAR_SONIC_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/x2_ultra.xml"
)
PRIMITIVES_PKL = (
    GEAR_SONIC_ROOT / "gear_sonic/data/motions/x2_planner_primitives.pkl"
)
SOURCE_PKL = (
    GEAR_SONIC_ROOT / "gear_sonic/data/motions/x2_ultra_bones_seed.pkl"
)
REPORT_MD = (
    GEAR_SONIC_ROOT / "gear_sonic/data/motions/x2_planner_primitives_report.md"
)


@dataclass
class Clip:
    """Single thing the player can render. Wraps either a primitive or a raw window."""

    label: str  # short string shown in stdout when navigating
    dof: np.ndarray  # (T, 31) float32
    root_rot_xyzw: np.ndarray  # (T, 4) float32, scipy xyzw
    root_trans: np.ndarray  # (T, 3) float32
    fps: float
    notes: str = ""  # free-form, printed when this clip is selected


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_primitives() -> dict[str, dict]:
    if not PRIMITIVES_PKL.exists():
        raise FileNotFoundError(
            f"Primitives PKL not found: {PRIMITIVES_PKL}\n"
            "Run: .venv/bin/python -m gear_sonic.scripts.curate_x2_primitives"
        )
    return joblib.load(PRIMITIVES_PKL)


def clip_from_primitive(bin_name: str, p: dict) -> Clip:
    recipe_family = p.get("recipe_family")
    recipe_ops = p.get("recipe_ops") or []
    recipe_sources = p.get("recipe_sources") or []
    if recipe_family:
        ops_str = " -> ".join(str(o) for o in recipe_ops)
        sources_str = ", ".join(str(s) for s in recipe_sources)
        notes = (
            f"bin={bin_name}  family={recipe_family} (RECIPE-BUILT)\n"
            f"  recipe ops: {ops_str}\n"
            f"  source label: {p['motion_key']}\n"
            f"  buffer sources: {sources_str}\n"
            f"  source_pkl={p['source_pkl']}\n"
            f"  frames={p['n_frames']} @ {p['fps']:g} fps "
            f"= {p['n_frames'] / p['fps']:.2f}s\n"
            f"  pinned={'yes' if p.get('pinned') else 'no'}"
            f"  arms_frozen={'yes' if p.get('freeze_arms_to_default') else 'no'}"
        )
    else:
        notes = (
            f"bin={bin_name}  motion_key={p['motion_key']}\n"
            f"  source_pkl={p['source_pkl']}\n"
            f"  frames={p['start_frame']}..{p['start_frame'] + p['n_frames']} "
            f"({p['n_frames']} @ {p['fps']:g} fps = {p['n_frames'] / p['fps']:.2f}s)\n"
            f"  status={'PARTIAL' if p.get('partial') else 'OK'}"
            f"  pinned={'yes' if p.get('pinned') else 'no'}"
            f"  arms_frozen={'yes' if p.get('freeze_arms_to_default') else 'no'}"
        )
    return Clip(
        label=bin_name,
        dof=np.asarray(p["dof"], dtype=np.float64),
        root_rot_xyzw=np.asarray(p["root_rot_xyzw"], dtype=np.float64),
        root_trans=np.asarray(p["root_trans"], dtype=np.float64),
        fps=float(p["fps"]),
        notes=notes,
    )


def clip_from_raw_window(
    motion_key: str,
    start: int,
    n: int,
    freeze_groups: tuple[str, ...] = (),
) -> Clip:
    """Load an arbitrary [start:start+n] window from the source bones-seed PKL.

    ``freeze_groups`` (e.g. ``("arms", "head")``) pins the listed DOF groups
    to ``DEFAULT_STAND_POSE_NP`` -- same semantics as the recipe DSL's
    ``freeze`` op. Useful for previewing what a candidate clip looks like
    AFTER the planner-recipe arm-strip will be applied.
    """
    if not SOURCE_PKL.exists():
        raise FileNotFoundError(f"Source PKL not found: {SOURCE_PKL}")
    raw = joblib.load(SOURCE_PKL)
    if motion_key not in raw:
        matches = [k for k in raw if motion_key in k]
        if len(matches) == 1:
            motion_key = matches[0]
        else:
            preview = ", ".join(list(raw.keys())[:6])
            raise KeyError(
                f"motion_key {motion_key!r} not found. "
                f"First few: {preview} ..."
            )
    entry = raw[motion_key]
    fps = float(entry.get("fps", 30))
    dof = np.asarray(entry["dof"], dtype=np.float64)
    rot = np.asarray(entry["root_rot"], dtype=np.float64)
    trans = np.asarray(entry["root_trans_offset"], dtype=np.float64)
    if start < 0 or start + n > dof.shape[0]:
        raise ValueError(
            f"window [{start}:{start + n}] out of bounds for clip with "
            f"{dof.shape[0]} frames"
        )
    dof_window = dof[start : start + n].copy()
    freeze_summary = ""
    if freeze_groups:
        # Resolve groups -> DOF indices via the same map the recipe DSL uses
        # so the preview matches what the runtime PKL will produce.
        from gear_sonic.utils.planner.constants import DEFAULT_STAND_POSE_NP
        from gear_sonic.utils.planner.x2_recipes import _GROUP_INDICES

        indices: list[int] = []
        for g in freeze_groups:
            g = str(g).lower()
            if g not in _GROUP_INDICES:
                raise ValueError(
                    f"--freeze: unknown group {g!r}. "
                    f"Allowed: {sorted(_GROUP_INDICES)}"
                )
            indices.extend(_GROUP_INDICES[g])
        indices = sorted(set(indices))
        template = DEFAULT_STAND_POSE_NP.astype(np.float64)
        for i in indices:
            dof_window[:, i] = template[i]
        freeze_summary = f"  freeze={','.join(sorted(set(freeze_groups)))}\n"
    notes = (
        f"raw  motion_key={motion_key}\n"
        f"  source_pkl={SOURCE_PKL}\n"
        f"  frames={start}..{start + n} ({n} @ {fps:g} fps = {n / fps:.2f}s)\n"
        f"{freeze_summary}".rstrip()
    )
    label_freeze = f"+freeze({','.join(sorted(set(freeze_groups)))})" if freeze_groups else ""
    return Clip(
        label=f"{motion_key}[{start}:{start + n}]{label_freeze}",
        dof=dof_window,
        root_rot_xyzw=rot[start : start + n],
        root_trans=trans[start : start + n],
        fps=fps,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# Report parser (for ``--candidate`` flag)
# ---------------------------------------------------------------------------


def candidate_from_report(bin_name: str, rank: int) -> Clip:
    """Parse `x2_planner_primitives_report.md` for the Nth candidate of `bin_name`.

    Falls back with a clear error if the report is missing or the bin isn't
    found. Used to audition top-K alternatives before pinning.
    """
    if not REPORT_MD.exists():
        raise FileNotFoundError(f"Report not found: {REPORT_MD}")
    text = REPORT_MD.read_text().splitlines()

    # Find ``### `bin_name`` header, then the next 8-row markdown table.
    header_re = re.compile(rf"^###\s+`{re.escape(bin_name)}`")
    in_section = False
    rows: list[str] = []
    for line in text:
        if header_re.match(line):
            in_section = True
            continue
        if in_section:
            if line.startswith("### "):
                break
            if (
                line.startswith("| ")
                and not line.startswith("|---")
                and "Rank" not in line
            ):
                rows.append(line)
    if not rows:
        raise ValueError(f"no candidates found for bin {bin_name!r} in report")
    if rank < 1 or rank > len(rows):
        raise ValueError(
            f"rank {rank} out of range; report has {len(rows)} candidates "
            f"for bin {bin_name!r}"
        )
    cells = [c.strip() for c in rows[rank - 1].split("|") if c.strip()]
    # Expected columns: Rank, Score, motion_key, start, N, ...
    rank_str, score_str, key_cell, start_cell, n_cell = cells[0:5]
    motion_key = key_cell.strip("`")
    start = int(start_cell)
    n = int(n_cell)
    clip = clip_from_raw_window(motion_key, start, n)
    clip.label = f"{bin_name}#cand{rank}: {motion_key}[{start}:{start + n}] (score={score_str})"
    clip.notes = (
        f"candidate rank {rank} for bin {bin_name}\n"
        f"  score={score_str}\n  " + clip.notes
    )
    return clip


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


def _apply_frame(
    mj_data,
    clip: Clip,
    f: int,
    anchor_xy: bool,
) -> None:
    f = int(f) % clip.dof.shape[0]
    if anchor_xy:
        mj_data.qpos[0] = 0.0
        mj_data.qpos[1] = 0.0
    else:
        mj_data.qpos[0] = float(clip.root_trans[f, 0])
        mj_data.qpos[1] = float(clip.root_trans[f, 1])
    mj_data.qpos[2] = float(clip.root_trans[f, 2])
    # MuJoCo qpos[3:7] is wxyz; PKL stores xyzw -> reorder.
    mj_data.qpos[3] = float(clip.root_rot_xyzw[f, 3])
    mj_data.qpos[4] = float(clip.root_rot_xyzw[f, 0])
    mj_data.qpos[5] = float(clip.root_rot_xyzw[f, 1])
    mj_data.qpos[6] = float(clip.root_rot_xyzw[f, 2])
    mj_data.qpos[7 : 7 + NUM_BODY_DOFS] = clip.dof[f]
    mj_data.qvel[:] = 0.0
    mj_data.xfrc_applied[:] = 0


def play_clips(
    clips: list[Clip],
    anchor_xy: bool,
    speed: float = 1.0,
    no_loop: bool = False,
) -> int:
    """Open the MuJoCo viewer and step through ``clips``. Returns 0 on clean exit."""
    import mujoco
    import mujoco.viewer

    if not clips:
        print("[browse] no clips to show", file=sys.stderr)
        return 1

    mj_model = mujoco.MjModel.from_xml_path(MJCF_PATH)
    mj_data = mujoco.MjData(mj_model)
    pelvis_id = mj_model.body("pelvis").id

    state = {
        "idx": 0,
        "frame": 0,
        "paused": False,
        "quit": False,
        "wall_start": time.time(),
        "frame_origin": 0,
    }

    def _announce(idx: int) -> None:
        c = clips[idx]
        print("\n" + "=" * 70, flush=True)
        print(f"[{idx + 1}/{len(clips)}] {c.label}", flush=True)
        print(c.notes, flush=True)

    def _switch(new_idx: int) -> None:
        new_idx = new_idx % len(clips)
        state["idx"] = new_idx
        state["frame"] = 0
        state["wall_start"] = time.time()
        state["frame_origin"] = 0
        _apply_frame(mj_data, clips[new_idx], 0, anchor_xy)
        mujoco.mj_forward(mj_model, mj_data)
        _announce(new_idx)

    def key_callback(keycode: int) -> None:
        import glfw

        if keycode in (glfw.KEY_X, glfw.KEY_ESCAPE):
            state["quit"] = True
        elif keycode == glfw.KEY_SPACE:
            state["paused"] = not state["paused"]
            print("[browse] paused" if state["paused"] else "[browse] resumed", flush=True)
        elif keycode == glfw.KEY_R:
            state["frame"] = 0
            state["wall_start"] = time.time()
            state["frame_origin"] = 0
            _apply_frame(mj_data, clips[state["idx"]], 0, anchor_xy)
            print("[browse] restart", flush=True)
        elif keycode == glfw.KEY_N:
            _switch(state["idx"] + 1)
        elif keycode == glfw.KEY_P:
            _switch(state["idx"] - 1)
        elif keycode == glfw.KEY_LEFT and state["paused"]:
            state["frame"] = max(0, state["frame"] - 10)
            _apply_frame(mj_data, clips[state["idx"]], state["frame"], anchor_xy)
        elif keycode == glfw.KEY_RIGHT and state["paused"]:
            n = clips[state["idx"]].dof.shape[0]
            state["frame"] = min(n - 1, state["frame"] + 10)
            _apply_frame(mj_data, clips[state["idx"]], state["frame"], anchor_xy)

    _apply_frame(mj_data, clips[0], 0, anchor_xy)
    mujoco.mj_forward(mj_model, mj_data)
    _announce(0)

    print(
        "\n=== X2 Planner Primitive Browser ===\n"
        "  SPACE pause | R restart | N/P next/prev clip | "
        "LEFT/RIGHT scrub (when paused) | X/ESC quit\n",
        flush=True,
    )

    with mujoco.viewer.launch_passive(
        mj_model,
        mj_data,
        key_callback=key_callback,
        show_left_ui=False,
        show_right_ui=False,
    ) as viewer:
        viewer.cam.azimuth = 120
        viewer.cam.elevation = -20
        viewer.cam.distance = 3.0
        viewer.cam.lookat[:] = [0.0, 0.0, DEFAULT_PELVIS_Z_M]
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = pelvis_id

        while viewer.is_running() and not state["quit"]:
            cur = clips[state["idx"]]
            n_frames = cur.dof.shape[0]
            frame_dt = 1.0 / (cur.fps * max(speed, 1e-6))

            if state["paused"]:
                viewer.sync()
                time.sleep(0.02)
                continue

            elapsed = time.time() - state["wall_start"]
            target = state["frame_origin"] + int(elapsed / frame_dt)
            if no_loop and target >= n_frames:
                state["paused"] = True
                # Pin to last frame so the viewer shows the landing pose,
                # not whatever wrapped frame would render next.
                last = n_frames - 1
                if state["frame"] != last:
                    state["frame"] = last
                    _apply_frame(mj_data, cur, last, anchor_xy)
                    mujoco.mj_forward(mj_model, mj_data)
                print(
                    f"[browse] end of clip ({n_frames} frames) — paused on last frame. "
                    "Press R to replay, LEFT/RIGHT to scrub, X to quit.",
                    flush=True,
                )
                continue
            target = target % n_frames

            if target != state["frame"]:
                state["frame"] = target
                _apply_frame(mj_data, cur, target, anchor_xy)
                mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()
            time.sleep(min(frame_dt, 0.02))

    return 0


# ---------------------------------------------------------------------------
# SONIC-in-the-loop player
# ---------------------------------------------------------------------------
#
# When --with-sonic is set, the browser drops its own MuJoCo viewer and
# becomes a ZMQ pose publisher. The SONIC sim deploy is spawned as a child
# (docker container, ``deploy_x2.sh sim --vla --sim-viewer``) and tracks the
# stream we publish. You navigate primitives from stdin (N/P/R/space/X);
# the visual is SONIC's MuJoCo window, so what you see is what the trained
# policy actually does with the recipe-built target stream.

PUBLISHER_FPS: float = 50.0  # matches state_machine.OUTPUT_FPS

SONIC_KEYBOARD_HELP = """
=== X2 Planner Primitive Browser (SONIC sim) ===
  N / P     next / previous primitive
  R         restart current primitive at frame 0
  SPACE     pause / resume publishing (last frame held)
  L         toggle: loop one primitive  vs  walk through all (auto-advance)
  X / ESC   quit
  ?         re-print this help
"""


class _PosePublisher:
    """Minimal ZMQ publisher matching the planner's wire format.

    Inlined (vs. importing ``x2_heuristic_planner.PosePublisher``) so the
    browser doesn't pull in the planner script's argparse / signal setup.
    """

    def __init__(self, host: str, port: int, topic: str = "pose") -> None:
        import zmq

        from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
            pack_pose_message,
        )

        self._pack = pack_pose_message
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(f"tcp://{host}:{port}")
        self._topic = topic
        # PUB-SUB warmup: the SONIC subscriber needs ~100ms to connect.
        time.sleep(0.1)

    def publish(
        self,
        joint_pos_mj: np.ndarray,
        root_quat_xyzw: np.ndarray,
        frame_index: int,
        motion_token_dim: int = 64,
        hand_dof: int = 10,
    ) -> None:
        payload = {
            "joint_pos_mj": np.asarray(joint_pos_mj, dtype=np.float32),
            "root_quat_xyzw": np.asarray(root_quat_xyzw, dtype=np.float32),
            "motion_token": np.zeros(motion_token_dim, dtype=np.float32),
            "left_hand_joints": np.zeros(hand_dof, dtype=np.float32),
            "right_hand_joints": np.zeros(hand_dof, dtype=np.float32),
            "frame_index": np.array([int(frame_index)], dtype=np.int64),
        }
        self._sock.send(self._pack(payload, topic=self._topic, version=4))

    def close(self) -> None:
        self._sock.close(linger=0)


class _RobotPoseSubscriber:
    """Background SUB on the bridge's ``robot_pose`` topic.

    The bridge publishes ground-truth pelvis qpos at state-rate. We SUB
    here so the browser can capture per-primitive XY distance and yaw
    delta without re-deriving it from the kinematic clip (which is what
    we *commanded* — not what SONIC actually tracked).
    """

    def __init__(self, host: str, port: int) -> None:
        import threading

        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.LINGER, 0)
        # Conflate so we always read the freshest pose; the browser only
        # needs the latest, not a stream.
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"robot_pose")
        self._sock.connect(f"tcp://{host}:{port}")
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="browse-sonic-pose-sub", daemon=True,
        )
        self._thread.start()

    def _loop(self) -> None:
        import zmq

        from gear_sonic.utils.teleop.zmq.robot_pose_zmq import unpack_robot_pose

        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        while not self._stop.is_set():
            socks = dict(poller.poll(timeout=200))
            if self._sock in socks:
                try:
                    msg = self._sock.recv(flags=zmq.NOBLOCK)
                    payload = unpack_robot_pose(msg)
                except Exception:
                    continue
                with self._lock:
                    self._latest = payload

    def latest(self) -> dict | None:
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def close(self) -> None:
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        except Exception:
            pass
        try:
            self._sock.close(linger=0)
        except Exception:
            pass


def _pelvis_xy_yaw(payload: dict | None) -> tuple[float, float, float] | None:
    """Extract (x, y, yaw_rad) from a robot_pose payload, or ``None`` if invalid."""
    if not payload:
        return None
    qpos = payload.get("pelvis_qpos_wxyz")
    if not qpos or len(qpos) != 7:
        return None
    from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw

    x, y, _z, qw, qx, qy, qz = (float(v) for v in qpos)
    quat_xyzw = np.array([qx, qy, qz, qw], dtype=np.float64)
    yaw = yaw_of_quat_xyzw(quat_xyzw)
    return x, y, yaw


def _short_angle(da: float) -> float:
    """Wrap angle delta to (-pi, pi]."""
    while da > np.pi:
        da -= 2.0 * np.pi
    while da <= -np.pi:
        da += 2.0 * np.pi
    return da


def _idle_first(clips: list["Clip"]) -> list["Clip"]:
    """Return ``clips`` with any ``idle_stand`` entry moved to position 0.

    Without this the alphabetically-first primitive (``back_step_half_ft``)
    plays on load, which is a poor first impression and biases the
    measurement of every subsequent primitive (the carry-over starts from
    a half-stride pose instead of square stance).
    """
    if not clips:
        return clips
    for k, c in enumerate(clips):
        if c.label == "idle_stand":
            if k == 0:
                return clips
            return [clips[k]] + clips[:k] + clips[k + 1:]
    return clips


def _write_eval_report(rows: list[dict], path: Path) -> None:
    """Persist per-primitive distance/heading deltas as a markdown table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "primitive", "duration_s", "dist_m", "dx_m", "dy_m",
        "dyaw_deg", "end_xy", "end_heading_deg", "completed",
    ]
    lines = [
        f"# browse_x2_planner_primitives eval — {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Captured {len(rows)} primitive(s) under SONIC sim.",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for row in rows:
        cells: list[str] = []
        for c in cols:
            val = row.get(c, "")
            if isinstance(val, tuple):
                cells.append("(" + ",".join(f"{v:+.3f}" for v in val) + ")")
            elif isinstance(val, float):
                cells.append(f"{val:+.3f}")
            else:
                cells.append(str(val))
        lines.append("| " + " | ".join(cells) + " |")
    path.write_text("\n".join(lines) + "\n")


def _aligned_clip(
    clip: "Clip",
    xy_world: tuple[float, float],
    yaw_world: float,
) -> "Clip":
    """Return a copy of ``clip`` with frame-0 yaw-aligned to ``(xy_world, yaw_world)``.

    Wraps :func:`gear_sonic.utils.planner.blending.yaw_align_segment` so each
    primitive is published from wherever the robot currently stands, instead
    of from the clip's own native heading. Without this, pressing N teleports
    the reference 90 deg / arbitrary distance away from the current state and
    SONIC tries to chase a discontinuous setpoint.
    """
    from gear_sonic.utils.planner.blending import yaw_align_segment

    dof, new_rot, new_trans = yaw_align_segment(
        clip.dof,
        clip.root_rot_xyzw,
        clip.root_trans,
        np.asarray(xy_world, dtype=np.float64),
        float(yaw_world),
    )
    return Clip(
        label=clip.label,
        dof=dof,
        root_rot_xyzw=new_rot,
        root_trans=new_trans,
        fps=clip.fps,
        notes=clip.notes,
    )

    def publish(
        self,
        joint_pos_mj: np.ndarray,
        root_quat_xyzw: np.ndarray,
        frame_index: int,
        motion_token_dim: int = 64,
        hand_dof: int = 10,
    ) -> None:
        payload = {
            "joint_pos_mj": np.asarray(joint_pos_mj, dtype=np.float32),
            "root_quat_xyzw": np.asarray(root_quat_xyzw, dtype=np.float32),
            "motion_token": np.zeros(motion_token_dim, dtype=np.float32),
            "left_hand_joints": np.zeros(hand_dof, dtype=np.float32),
            "right_hand_joints": np.zeros(hand_dof, dtype=np.float32),
            "frame_index": np.array([int(frame_index)], dtype=np.int64),
        }
        self._sock.send(self._pack(payload, topic=self._topic, version=4))

    def close(self) -> None:
        self._sock.close(linger=0)


def _resolve_deploy_model_onnx(
    sonic_checkpoint: Path | None,
    deploy_model: Path | None,
    deploy_model_dir: Path | None,
) -> Path:
    """Mirror ``record_x2_dataset.sh`` resolution of the .onnx model path.

    Precedence: ``deploy_model`` > ``deploy_model_dir/exported/*.onnx`` >
    ``dirname(sonic_checkpoint)/exported/*.onnx``. Raises a clear error if
    none resolves.
    """
    if deploy_model is not None:
        if not deploy_model.is_file():
            raise FileNotFoundError(f"--deploy-model not found: {deploy_model}")
        if deploy_model.suffix != ".onnx":
            raise ValueError(
                f"--deploy-model must be a .onnx file, got {deploy_model}"
            )
        return deploy_model

    candidate_dir: Path | None = None
    if deploy_model_dir is not None:
        candidate_dir = deploy_model_dir
    elif sonic_checkpoint is not None:
        if not sonic_checkpoint.is_file():
            raise FileNotFoundError(
                f"--sonic-checkpoint not found: {sonic_checkpoint}"
            )
        candidate_dir = sonic_checkpoint.parent
    else:
        raise ValueError(
            "SONIC sim mode requires one of: --sonic-checkpoint, "
            "--deploy-model-dir, or --deploy-model. "
            "Example: --sonic-checkpoint /home/stickbot/x2_cloud_checkpoints/"
            "h200-iter-25000-sphere-feet-20260501/model_step_025000.pt"
        )

    exported = candidate_dir / "exported"
    if not exported.is_dir():
        raise FileNotFoundError(
            f"no 'exported/' directory under {candidate_dir} -- re-export the "
            "SONIC checkpoint to ONNX or pass --deploy-model directly."
        )
    onnx_files = sorted(exported.glob("*.onnx"))
    if not onnx_files:
        raise FileNotFoundError(
            f"no .onnx file in {exported} -- re-export the SONIC checkpoint "
            "or pass --deploy-model <path.onnx> directly."
        )
    return onnx_files[0]


def _spawn_sonic_deploy(
    pub_host: str,
    pub_port: int,
    sim_profile: str,
    max_duration_s: float,
    log_path: Path,
    show_viewer: bool,
    model_onnx: Path,
    wrist_bypass: str,
) -> "subprocess.Popen[bytes]":
    """Spawn ``deploy_x2.sh sim --vla --sim-viewer`` as a backgrounded child."""
    import subprocess

    deploy_sh = GEAR_SONIC_ROOT / "gear_sonic_deploy" / "deploy_x2.sh"
    if not deploy_sh.exists():
        raise FileNotFoundError(f"deploy_x2.sh not found: {deploy_sh}")
    # Mirror ``record_x2_dataset.sh`` verbatim:
    #   * --sim-init-pose default           # explicit DEFAULT_DOF spawn
    #   * NO --sim-profile                  # bridge resolves to 'manual'
    #   * --vla --vla-zmq-host/port         # we publish on this socket
    #   * --no-confirm --autostart-after 0  # no prompts, no WAIT phase
    #   * --wrist-bypass ${WRIST_BYPASS}    # only meaningful in --vla mode
    # The recorder script has been driving this exact deploy invocation in
    # production for months; do NOT add new flags here without verifying
    # against record_x2_dataset.sh first.
    cmd = [
        str(deploy_sh), "sim",
        "--no-confirm",
        "--vla",
        "--vla-zmq-host", pub_host,
        "--vla-zmq-port", str(pub_port),
        "--sim-init-pose", "default",
        "--autostart-after", "0",
        "--max-duration", str(int(max_duration_s)),
        "--model", str(model_onnx),
        "--wrist-bypass", wrist_bypass,
    ]
    # ``sim_profile`` is kept as a CLI knob but only forwarded when the
    # caller explicitly asked for something OTHER than the recorder's
    # default ('manual' resolved by deploy_x2.sh when no profile is given
    # AND no --motion is passed). Forwarding ``manual`` explicitly is a
    # no-op but adds one more divergence-from-recorder we don't need.
    if sim_profile and sim_profile.lower() not in ("", "manual"):
        cmd.extend(["--sim-profile", sim_profile])
    if show_viewer:
        cmd.append("--sim-viewer")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(log_path, "wb")  # noqa: SIM115 -- closed via Popen lifetime
    print(f"[browse-sonic] spawning deploy -> {log_path}", flush=True)
    print(f"[browse-sonic]   model={model_onnx}", flush=True)
    print(f"[browse-sonic]   {' '.join(cmd)}", flush=True)
    return subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,  # don't share our cbreak TTY with docker
        start_new_session=True,
    )


def _wait_for_deploy_ready(
    proc: "subprocess.Popen[bytes]",
    log_path: Path,
    timeout_s: float = 180.0,
    ready_marker: str = "Launching ...",
) -> bool:
    """Poll the deploy log for the 'Launching ...' marker. Returns True on ready.

    Mirrors the wait loop in record_x2_dataset.sh (tail-based, ~1 Hz log
    polling). We deliberately do NOT pump anything on the pose wire during
    this wait -- the deploy uses its internal default-angles target while
    it boots; the recorder script doesn't pump either. Streaming references
    here just adds a divergence vector vs. the proven path.
    """
    start = time.time()
    last_size = 0
    while time.time() - start < timeout_s:
        if proc.poll() is not None:
            print(
                f"\n[browse-sonic] deploy died during bring-up "
                f"(rc={proc.returncode}). Last 30 lines of log:",
                flush=True,
            )
            try:
                tail = log_path.read_text().splitlines()[-30:]
                for line in tail:
                    print(f"  {line}", flush=True)
            except Exception:
                pass
            return False
        if log_path.exists():
            try:
                size = log_path.stat().st_size
                if size > last_size:
                    chunk = log_path.read_bytes()[last_size:size]
                    last_size = size
                    text = chunk.decode("utf-8", errors="replace")
                    for line in text.splitlines():
                        print(f"  [deploy] {line}", flush=True)
                    if ready_marker in text:
                        return True
            except Exception:
                pass
        time.sleep(1.0)
    print(
        f"\n[browse-sonic] deploy didn't reach '{ready_marker}' within "
        f"{timeout_s:.0f}s; killing.",
        flush=True,
    )
    return False


def _kill_child(
    proc: "subprocess.Popen[bytes] | None",
    grace_s: float = 5.0,
    deploy_log: Path | None = None,
) -> None:
    """SIGINT->SIGTERM->SIGKILL the deploy child + ``docker stop`` the
    container it spawned.

    Mirrors the cleanup() function in record_x2_dataset.sh. Two-stage
    teardown is required because ``deploy_x2.sh`` auto-relaunches inside
    a ``docker_x2-x2sim`` container via ``docker compose run`` and
    signals from the host bash do NOT reliably propagate into the
    container -- the compose-run shell can race with our SIGTERM and
    exit before delivering the signal to PID 1 inside the container,
    leaking the deploy + MuJoCo bridge as host-visible orphans that
    hold ports 5556/5557/5570 forever (and reject ``pkill`` from this
    UID because container processes run as root). We therefore:

      (1) SIGINT the host bash so its own cleanup paths (RAMP_OUT,
          MuJoCo viewer teardown) get a chance to fire,
      (2) wait briefly,
      (3) escalate to SIGTERM/SIGKILL on the process group if needed,
      (4) parse the deploy log for the ``docker_x2-x2sim-run-<hex>``
          container name and ``docker stop`` it explicitly.
    Step (4) succeeds even if (1)-(3) failed entirely.
    """
    import signal as _signal
    import shutil
    import subprocess as _subprocess

    def _docker_stop_from_log(log_path: Path) -> None:
        """Find the docker container name in the deploy log and stop it."""
        if log_path is None or not log_path.exists():
            return
        if shutil.which("docker") is None:
            return
        try:
            text = log_path.read_text(errors="replace")
        except Exception:
            return
        # Match the bring-up line, e.g.
        # ``Container docker_x2-x2sim-run-126c526bb9d3 Creating``.
        import re
        matches = re.findall(r"docker_x2-x2sim-run-[a-f0-9]+", text)
        if not matches:
            return
        container = matches[-1]
        try:
            running = _subprocess.run(
                ["docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True, timeout=5,
            ).stdout.splitlines()
        except Exception:
            return
        if container not in running:
            return
        print(
            f"[browse-sonic] stopping deploy container {container} "
            "(docker stop --timeout 5)",
            flush=True,
        )
        try:
            _subprocess.run(
                ["docker", "stop", "--timeout", "5", container],
                capture_output=True, timeout=10,
            )
        except Exception as exc:
            print(
                f"[browse-sonic] docker stop {container} failed: {exc}",
                flush=True,
            )

    if proc is not None and proc.poll() is None:
        # (1) SIGINT first so deploy_x2.sh can run its own cleanup paths.
        print(
            f"[browse-sonic] stopping deploy child pid={proc.pid} (SIGINT)",
            flush=True,
        )
        try:
            os.killpg(os.getpgid(proc.pid), _signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=grace_s)
        except Exception:
            # (3a) Escalate to SIGTERM.
            print(
                f"[browse-sonic] SIGINT didn't drain pid={proc.pid}; "
                "escalating to SIGTERM",
                flush=True,
            )
            try:
                os.killpg(os.getpgid(proc.pid), _signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=grace_s)
            except Exception:
                # (3b) Final escalation to SIGKILL.
                print(
                    f"[browse-sonic] SIGTERM didn't drain pid={proc.pid}; "
                    "escalating to SIGKILL",
                    flush=True,
                )
                try:
                    os.killpg(os.getpgid(proc.pid), _signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass

    # (4) ALWAYS try to docker-stop the container, even if (1)-(3)
    # appear to have succeeded -- the host bash may exit cleanly while
    # leaving the container running. This step is idempotent.
    _docker_stop_from_log(deploy_log)


def play_clips_via_sonic(
    clips: list[Clip],
    pub_host: str,
    pub_port: int,
    sim_profile: str,
    max_duration_s: float,
    show_viewer: bool,
    auto_advance: bool,
    model_onnx: Path,
    wrist_bypass: str,
    boot_timeout_s: float = 180.0,
    *,
    pose_sub_host: str = "127.0.0.1",
    pose_sub_port: int = 5570,
    sweep_mode: bool = False,
    sweep_hold_s: float = 4.0,
    sweep_settle_s: float = 1.5,
    loop_gap_s: float = 1.0,
    initial_settle_s: float = 1.5,
    play_once: bool = False,
    eval_report_path: Path | None = None,
) -> int:
    """Publish the current primitive at 50 Hz to a child SONIC sim deploy.

    Drops the local kinematic MuJoCo viewer; SONIC's window is the visual.
    Stdin (cbreak-mode TTY) drives navigation.
    """
    import select
    import signal as _signal
    import subprocess  # noqa: F401 -- type alias only
    import sys as _sys
    import termios
    import tty

    if not clips:
        print("[browse-sonic] no clips to show", file=_sys.stderr)
        return 1
    if not sweep_mode and not _sys.stdin.isatty():
        print(
            "[browse-sonic] stdin is not a TTY; refusing to start "
            "(use --sweep for unattended auto-cycle runs, or "
            "scripted_demo + run_planner_smoke.sh --with-deploy)",
            file=_sys.stderr,
        )
        return 2

    log_dir = GEAR_SONIC_ROOT / "data" / "sim_to_real_anchors" / "browse_sonic"
    deploy_log = log_dir / time.strftime("deploy_%Y%m%d_%H%M%S.log")
    if eval_report_path is None:
        eval_report_path = log_dir / time.strftime("eval_%Y%m%d_%H%M%S.md")

    # Reorder so idle_stand is always first (calm starting state for SONIC).
    clips = _idle_first(clips)

    deploy_proc = None
    publisher: _PosePublisher | None = None
    pose_sub: _RobotPoseSubscriber | None = None
    eval_rows: list[dict] = []
    fd = _sys.stdin.fileno()
    old_termios = termios.tcgetattr(fd) if _sys.stdin.isatty() else None
    quit_flag = {"q": False}

    def _signal_handler(signum, _frame):  # noqa: ANN001 -- signal handler
        print(f"[browse-sonic] received signal {signum}, shutting down...",
              flush=True)
        quit_flag["q"] = True

    _signal.signal(_signal.SIGINT, _signal_handler)
    _signal.signal(_signal.SIGTERM, _signal_handler)

    try:
        publisher = _PosePublisher(pub_host, pub_port)

        # Spin up the robot_pose SUB FIRST -- before we spawn the deploy
        # container -- so we start receiving pelvis pose the instant the
        # bridge binds its PUB socket inside Docker. This is critical: the
        # bridge spawns the robot at a non-zero yaw (the .x2_init_pose YAML
        # picks a random / configured spawn heading, often around -90deg
        # for ``gantry_hang``), and if the boot pump streams identity quat
        # while the robot is being held by the gantry, the moment SONIC
        # enters CONTROL it tries to slam-track the legs to yaw=0 -- a 90+
        # degree swing -- and the robot crumples on first contact with the
        # floor. Connecting the SUB early lets the boot pump pivot to the
        # OBSERVED yaw the instant the first pose message arrives, so the
        # commanded reference always matches the spawn heading. (SUB
        # sockets auto-reconnect, so connecting before the PUB binds is
        # safe.)
        try:
            pose_sub = _RobotPoseSubscriber(pose_sub_host, pose_sub_port)
        except Exception as exc:
            print(
                f"[browse-sonic] robot_pose SUB connect failed "
                f"({exc}); per-primitive distance/heading capture disabled.",
                flush=True,
            )

        # NOTE: we deliberately do NOT pump anything on the pose wire
        # during deploy boot. The working record_x2_dataset.sh /
        # heuristic-planner flows don't either; the deploy holds its
        # internal default-angles target while it boots, then we start
        # publishing the recorder-style idle (identity quat +
        # DEFAULT_STAND_POSE) once the bridge is alive and the SUB has
        # had 2s to bind. See the warmup loop below for the publish.
        deploy_proc = _spawn_sonic_deploy(
            pub_host, pub_port, sim_profile, max_duration_s,
            deploy_log, show_viewer, model_onnx, wrist_bypass,
        )

        # Hold off on cbreak until SONIC is actually live so users see deploy
        # boot logs in cooked mode and any failure is obvious. NOTE: we do
        # NOT pump anything on the pose wire during deploy boot -- the
        # working ``record_x2_dataset.sh`` / heuristic-planner flows don't
        # either, and the deploy uses its INTERNAL default-angles target
        # while it boots. Pumping here just adds a divergence vector from
        # the proven path.
        ready = _wait_for_deploy_ready(
            deploy_proc, deploy_log, timeout_s=boot_timeout_s,
        )
        if not ready:
            print(
                f"[browse-sonic] deploy failed to start; full log at:\n"
                f"  {deploy_log}",
                flush=True,
            )
            return 3

        # Brief settle for the ZMQ pose SUB to bind before we start
        # publishing -- mirrors record_x2_dataset.sh (line 437):
        #   ``# Brief settle for the ZMQ pose SUB to bind before we start blasting.``
        #   ``sleep 2``
        print(
            "[browse-sonic] deploy ready; sleeping 2s for ZMQ SUB to bind "
            "(mirrors record_x2_dataset.sh).",
            flush=True,
        )
        time.sleep(2.0)

        if old_termios is not None:
            tty.setcbreak(fd)
        print(SONIC_KEYBOARD_HELP, flush=True)

        idx = 0
        frame = 0
        global_frame_index = 0
        paused = False
        # Sweep mode auto-advances; otherwise default to loop-one.
        loop_one = (not auto_advance) and (not sweep_mode)

        # ── Carry-over state ────────────────────────────────────────────
        # Each new primitive is yaw-aligned so frame 0 lands at the last
        # frame of the previous primitive, instead of at the clip's
        # native (0,0)/native-yaw. Without this, pressing N teleports the
        # reference target across the floor.
        carry_xy: tuple[float, float] = (0.0, 0.0)
        carry_yaw: float = 0.0

        # Per-primitive measurement state (updated each time we enter a
        # new primitive in the timeline).
        seg_t_start: float = time.time()
        seg_pose_start: tuple[float, float, float] | None = None
        seg_announced_idx: int = -1
        play_once_announced: list[bool] = [False]

        def _measure_now() -> tuple[float, float, float] | None:
            if pose_sub is None:
                return None
            return _pelvis_xy_yaw(pose_sub.latest())

        def _record_segment_end(end_idx: int, completed: bool) -> None:
            """Snapshot pelvis pose at end of a primitive, log + append to report."""
            nonlocal carry_xy, carry_yaw
            label = clips[end_idx].label
            now_pose = _measure_now()
            if seg_pose_start is None or now_pose is None:
                # No telemetry; we still update carry from the clip's
                # commanded final frame so the next primitive at least
                # starts from where we *asked* SONIC to be.
                last = clips[end_idx]
                last_xy = (float(last.root_trans[-1, 0]), float(last.root_trans[-1, 1]))
                from gear_sonic.utils.planner.blending import yaw_of_quat_xyzw
                last_yaw = yaw_of_quat_xyzw(last.root_rot_xyzw[-1])
                carry_xy = last_xy
                carry_yaw = last_yaw
                return
            x0, y0, yaw0 = seg_pose_start
            x1, y1, yaw1 = now_pose
            dx, dy = x1 - x0, y1 - y0
            dist = float(np.hypot(dx, dy))
            dyaw = _short_angle(yaw1 - yaw0)
            heading_deg = float(np.degrees(yaw1))
            duration = time.time() - seg_t_start
            row = {
                "primitive": label,
                "duration_s": round(duration, 2),
                "dx_m": round(dx, 3),
                "dy_m": round(dy, 3),
                "dist_m": round(dist, 3),
                "dyaw_deg": round(float(np.degrees(dyaw)), 2),
                "end_xy": (round(x1, 3), round(y1, 3)),
                "end_heading_deg": round(heading_deg, 2),
                "completed": completed,
            }
            eval_rows.append(row)
            print(
                f"[browse-sonic] eval[{label}]: dt={duration:.2f}s "
                f"dist={dist:.3f}m (dx={dx:+.3f}, dy={dy:+.3f}) "
                f"dyaw={np.degrees(dyaw):+.1f}deg "
                f"end_xy=({x1:+.3f},{y1:+.3f}) heading={heading_deg:+.1f}deg "
                f"completed={completed}",
                flush=True,
            )
            carry_xy = (x1, y1)
            carry_yaw = yaw1

        def _hold_stand_pose(duration_s: float) -> bool:
            """Publish recorder-style idle (DEFAULT_STAND_POSE + identity
            quat) for ``duration_s``.

            Returns True if the hold completed normally, False if the user
            requested quit during the hold (caller should bail).
            Publishes at PUBLISHER_FPS so SONIC always has a fresh ref.
            Uses IDENTITY quat (same as ``x2_dataset_recorder._publish_idle``);
            do NOT change this to track observed yaw.
            """
            nonlocal global_frame_index
            if duration_s <= 0.0:
                return True
            t_end = time.time() + duration_s
            while time.time() < t_end and not quit_flag["q"]:
                publisher.publish(
                    joint_pos_mj=warmup_dof,
                    root_quat_xyzw=warmup_quat,
                    frame_index=global_frame_index,
                )
                global_frame_index += 1
                time.sleep(period_s)
            return not quit_flag["q"]

        def _enter_segment(i: int, *, settle_s: float) -> None:
            """Yaw-align clip ``i`` to the carried (xy,yaw) and announce it.

            ``settle_s`` holds the canonical stand pose (NOT the previous
            clip's mid-stride end frame) so SONIC always re-stabilises to a
            known good standing state before the next primitive begins.
            Without this, after several N-presses the robot accumulates
            drift / instability from each primitive's end pose and starts
            failing to track new commands ("turn_right_45deg won't actually
            rotate after navigating through 30 bins").
            """
            nonlocal frame, seg_t_start, seg_pose_start, seg_announced_idx
            nonlocal carry_xy, carry_yaw
            _hold_stand_pose(settle_s)
            # After the settle, re-snapshot the observed pelvis pose so we
            # carry from where SONIC actually settled (vs. wherever the
            # previous primitive's commanded end frame placed the
            # reference).
            observed_after_settle = _measure_now()
            if observed_after_settle is not None:
                carry_xy = (observed_after_settle[0], observed_after_settle[1])
                carry_yaw = observed_after_settle[2]
            # Replace clip entry with a yaw-aligned copy.
            aligned = _aligned_clip(clips[i], carry_xy, carry_yaw)
            clips[i] = aligned
            frame = 0
            seg_t_start = time.time()
            seg_pose_start = _measure_now()
            seg_announced_idx = i
            play_once_announced[0] = False
            _announce(i, start_pose=seg_pose_start)

        def _announce(i: int, *, start_pose=None) -> None:
            c = clips[i]
            print("\n" + "=" * 70, flush=True)
            if sweep_mode:
                mode = "SWEEP"
            elif play_once:
                mode = "play-once+hold"
            elif loop_one:
                mode = "loop=one"
            else:
                mode = "loop=all"
            print(f"[{i + 1}/{len(clips)}] {c.label}  "
                  f"({mode} "
                  f"{'PAUSED' if paused else 'PLAYING'})",
                  flush=True)
            print(c.notes, flush=True)
            if start_pose is not None:
                print(
                    f"  start_xy=({start_pose[0]:+.3f},{start_pose[1]:+.3f}) "
                    f"start_heading={np.degrees(start_pose[2]):+.1f}deg "
                    f"carry_yaw={np.degrees(carry_yaw):+.1f}deg",
                    flush=True,
                )
            if not sweep_mode:
                print("(N=next P=prev R=restart SPACE=pause L=loop-mode X=quit ?=help)",
                      flush=True)

        period_s = 1.0 / PUBLISHER_FPS
        next_tick = time.time()

        # ── Warmup: idle-publish (recorder-style) until telemetry streaks ─
        # We now publish the SAME wire format the recorder uses while
        # waiting for VR (DEFAULT_STAND_POSE + identity quat). We still
        # poll pose_sub to know the robot's xy/yaw at warmup end so the
        # FIRST primitive can be carry-aligned, but we DO NOT use observed
        # yaw to derive the published quat -- the policy was trained on
        # identity-quat references and any other quat will cause spin.
        warmup_s = 8.0
        warmup_end = time.time() + warmup_s
        warmup_pose_seen_at: float | None = None
        from gear_sonic.utils.planner.constants import DEFAULT_STAND_POSE_NP

        warmup_dof = DEFAULT_STAND_POSE_NP.astype(np.float32, copy=True)
        warmup_quat = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        live = None
        warmup_t0 = time.time()
        last_warmup_log = 0.0
        print(
            f"[browse-sonic] WARMUP start (idle-publish, up to {warmup_s:.1f}s, "
            f"need 3s of continuous telemetry)",
            flush=True,
        )
        while time.time() < warmup_end and not quit_flag["q"]:
            publisher.publish(
                joint_pos_mj=warmup_dof,
                root_quat_xyzw=warmup_quat,
                frame_index=global_frame_index,
            )
            global_frame_index += 1
            cur_live = _measure_now() if pose_sub is not None else None
            if cur_live is not None:
                if warmup_pose_seen_at is None:
                    warmup_pose_seen_at = time.time()
                live = cur_live
            now_ts = time.time()
            if (now_ts - last_warmup_log) >= 0.5:
                last_warmup_log = now_ts
                if cur_live is not None:
                    seen_dur = (
                        now_ts - warmup_pose_seen_at
                        if warmup_pose_seen_at is not None
                        else 0.0
                    )
                    print(
                        f"[browse-sonic] WARMUP t+{now_ts - warmup_t0:5.2f}s "
                        f"obs_yaw={np.degrees(cur_live[2]):+.1f}deg "
                        f"obs_xy=({cur_live[0]:+.3f},{cur_live[1]:+.3f})m "
                        f"telem_streak={seen_dur:.2f}s",
                        flush=True,
                    )
                else:
                    print(
                        f"[browse-sonic] WARMUP t+{now_ts - warmup_t0:5.2f}s "
                        f"obs=NONE (no telemetry yet)",
                        flush=True,
                    )
            if (
                cur_live is not None
                and warmup_pose_seen_at is not None
                and (now_ts - warmup_pose_seen_at) >= 3.0
            ):
                print(
                    f"[browse-sonic] WARMUP done (3s telemetry streak) "
                    f"obs_yaw={np.degrees(cur_live[2]):+.1f}deg "
                    f"obs_xy=({cur_live[0]:+.3f},{cur_live[1]:+.3f})m",
                    flush=True,
                )
                break
            time.sleep(period_s)
        if pose_sub is not None and live is None:
            print(
                "[browse-sonic] WARN: no robot_pose telemetry within "
                f"{warmup_s:.1f}s warmup; eval rows will be dropped.",
                flush=True,
            )
        if live is not None:
            carry_xy = (live[0], live[1])
            carry_yaw = live[2]
        # ── Pre-launch hold ────────────────────────────────────────────
        # Identical wire format to the recorder's _publish_idle: trained
        # stand pose + IDENTITY root quat + zero motion_token. We carry
        # the observed pelvis xy/yaw locally so the first primitive can
        # be world-aligned to where the robot actually settled, but we
        # NEVER feed observed yaw into the published quat -- doing that
        # is what caused the gantry-spin we just spent an hour debugging.
        print(
            "[browse-sonic] gantry drop complete; holding stand pose at "
            f"obs=({carry_xy[0]:+.3f},{carry_xy[1]:+.3f})m / "
            f"yaw={np.degrees(carry_yaw):+.1f}deg "
            "(publishing identity-quat idle, recorder-style).\n"
            "  press N to launch the first primitive, X to quit, ? for help.",
            flush=True,
        )
        # Optional auto-arm after ``initial_settle_s`` for the sweep / auto-
        # advance flows that explicitly want unattended runs. Interactive
        # single-clip and bin-list views require an explicit N press.
        auto_arm = sweep_mode or auto_advance
        auto_arm_deadline = (
            time.time() + initial_settle_s
            if (auto_arm and initial_settle_s > 0.0)
            else None
        )
        armed = False
        prelaunch_t0 = time.time()
        last_prelaunch_log = 0.0
        prelaunch_carry_yaw_at_t0 = carry_yaw
        prelaunch_carry_xy_at_t0 = carry_xy
        while not armed and not quit_flag["q"]:
            if deploy_proc.poll() is not None:
                print(
                    f"\n[browse-sonic] deploy child exited rc={deploy_proc.returncode} "
                    f"during pre-launch hold; see {deploy_log}.",
                    flush=True,
                )
                return 4
            publisher.publish(
                joint_pos_mj=warmup_dof,
                root_quat_xyzw=warmup_quat,
                frame_index=global_frame_index,
            )
            global_frame_index += 1
            # Continually refresh carry from observed pose so the first
            # clip launches from where the robot ACTUALLY is, not where it
            # was the instant we left warmup.
            obs = _measure_now()
            if obs is not None:
                carry_xy = (obs[0], obs[1])
                carry_yaw = obs[2]
            now_ts = time.time()
            if (now_ts - last_prelaunch_log) >= 1.0:
                last_prelaunch_log = now_ts
                if obs is not None:
                    drift_yaw = np.degrees(
                        _short_angle(carry_yaw - prelaunch_carry_yaw_at_t0)
                    )
                    drift_xy = np.hypot(
                        carry_xy[0] - prelaunch_carry_xy_at_t0[0],
                        carry_xy[1] - prelaunch_carry_xy_at_t0[1],
                    )
                    print(
                        f"[browse-sonic] PRE-LAUNCH t+{now_ts - prelaunch_t0:5.2f}s "
                        f"obs_xy=({carry_xy[0]:+.3f},{carry_xy[1]:+.3f})m "
                        f"obs_yaw={np.degrees(carry_yaw):+.1f}deg "
                        f"drift_yaw={drift_yaw:+.1f}deg "
                        f"drift_xy={drift_xy:.3f}m",
                        flush=True,
                    )
                else:
                    print(
                        f"[browse-sonic] PRE-LAUNCH t+{now_ts - prelaunch_t0:5.2f}s "
                        f"obs=NONE (telemetry stalled)",
                        flush=True,
                    )
            if old_termios is not None:
                r, _, _ = select.select([_sys.stdin], [], [], 0.0)
                if r:
                    ch = _sys.stdin.read(1)
                    if ch in ("x", "X", "\x1b", "\x03"):
                        quit_flag["q"] = True
                        break
                    elif ch in ("n", "N", "\r", "\n", " "):
                        armed = True
                        break
                    elif ch == "?":
                        print(SONIC_KEYBOARD_HELP, flush=True)
                        print(
                            "  (currently in PRE-LAUNCH hold: press N / "
                            "ENTER / SPACE to start the first primitive, "
                            "X to quit.)",
                            flush=True,
                        )
            if auto_arm_deadline is not None and time.time() >= auto_arm_deadline:
                print(
                    f"[browse-sonic] auto-arm: {initial_settle_s:.2f}s "
                    "elapsed in {sweep|auto-advance} mode -> launching first "
                    "primitive without N.",
                    flush=True,
                )
                armed = True
                break
            time.sleep(period_s)
        if quit_flag["q"]:
            return 0
        # Final snapshot RIGHT BEFORE we hand control to the clip.
        launch_obs = _measure_now()
        print(
            "[browse-sonic] LAUNCH (user pressed N) -> handing control to "
            "primitive 0:\n"
            f"  carry_xy=({carry_xy[0]:+.3f},{carry_xy[1]:+.3f})m  "
            f"carry_yaw={np.degrees(carry_yaw):+.1f}deg",
            flush=True,
        )
        if launch_obs is not None:
            launch_dyaw = np.degrees(_short_angle(launch_obs[2] - carry_yaw))
            print(
                f"  observed at launch: xy=({launch_obs[0]:+.3f},"
                f"{launch_obs[1]:+.3f})m yaw={np.degrees(launch_obs[2]):+.1f}deg "
                f"d(obs-carry)={launch_dyaw:+.2f}deg",
                flush=True,
            )
        # Now actually start the first primitive. settle_s=0 here because
        # we already held the stand pose for as long as the user wanted.
        _enter_segment(idx, settle_s=0.0)

        while not quit_flag["q"]:
            # Bail if SONIC died.
            if deploy_proc.poll() is not None:
                print(
                    f"\n[browse-sonic] deploy child exited rc={deploy_proc.returncode}; "
                    f"see {deploy_log} for details.",
                    flush=True,
                )
                break

            # Sweep-mode auto-advance: hold each primitive for ~sweep_hold_s
            # past its kinematic end, then move on.
            if sweep_mode and not paused:
                seg_elapsed = time.time() - seg_t_start
                cur = clips[idx]
                kinematic_dur = cur.dof.shape[0] / float(PUBLISHER_FPS)
                if seg_elapsed >= kinematic_dur + sweep_hold_s:
                    _record_segment_end(idx, completed=True)
                    if idx + 1 >= len(clips):
                        print("[browse-sonic] sweep complete.", flush=True)
                        break
                    idx += 1
                    _enter_segment(idx, settle_s=sweep_settle_s)

            # Non-blocking stdin poll: 1 tick = 20ms (50 Hz pub period).
            if old_termios is not None:
                r, _, _ = select.select([_sys.stdin], [], [], 0.0)
                if r:
                    ch = _sys.stdin.read(1)
                    if ch in ("x", "X", "\x1b", "\x03"):  # x, X, ESC, Ctrl-C
                        print("[browse-sonic] quit", flush=True)
                        _record_segment_end(idx, completed=False)
                        break
                    elif ch in ("n", "N"):
                        _record_segment_end(idx, completed=False)
                        idx = (idx + 1) % len(clips)
                        _enter_segment(idx, settle_s=sweep_settle_s)
                    elif ch in ("p", "P"):
                        _record_segment_end(idx, completed=False)
                        idx = (idx - 1) % len(clips)
                        _enter_segment(idx, settle_s=sweep_settle_s)
                    elif ch in ("r", "R"):
                        _record_segment_end(idx, completed=False)
                        _enter_segment(idx, settle_s=sweep_settle_s)
                        print("[browse-sonic] restart", flush=True)
                    elif ch == " ":
                        paused = not paused
                        print("[browse-sonic] " + ("paused" if paused else "resumed"),
                              flush=True)
                    elif ch in ("l", "L"):
                        loop_one = not loop_one
                        print(f"[browse-sonic] loop mode = "
                              f"{'one' if loop_one else 'all (auto-advance)'}",
                              flush=True)
                    elif ch == "?":
                        print(SONIC_KEYBOARD_HELP, flush=True)

            if not paused:
                cur = clips[idx]
                publisher.publish(
                    joint_pos_mj=cur.dof[frame],
                    root_quat_xyzw=cur.root_rot_xyzw[frame],
                    frame_index=global_frame_index,
                )
                global_frame_index += 1
                frame += 1
                if frame >= cur.dof.shape[0]:
                    if play_once:
                        # Hold the LAST commanded clip frame indefinitely so
                        # the user can watch SONIC track / settle the landing
                        # pose. We do not loop, do not advance, do not snap
                        # the robot back to a stand pose -- we keep streaming
                        # the same end-of-clip target so the policy can finish
                        # planting the foot under physics. Falls through to
                        # the rate-limiter at the bottom of the loop so we
                        # don't busy-spin.
                        if not play_once_announced[0]:
                            play_once_announced[0] = True
                            held_pose = _measure_now()
                            if held_pose is not None:
                                pose_str = (
                                    f"  observed pelvis: "
                                    f"xy=({held_pose[0]:+.3f},{held_pose[1]:+.3f})m "
                                    f"yaw={np.degrees(held_pose[2]):+.1f}deg"
                                )
                            else:
                                pose_str = ""
                            print(
                                "[browse-sonic] end of clip reached "
                                "(--play-once / no --loop): holding LAST "
                                "commanded frame so SONIC can settle the "
                                "landing pose."
                                + ("\n" + pose_str if pose_str else "")
                                + "\n  press R to replay, N/P to advance/back, "
                                "X to quit, ? for help.",
                                flush=True,
                            )
                        frame = cur.dof.shape[0] - 1
                    elif loop_one:
                        # Snapshot observed pose IMMEDIATELY at end-of-clip
                        # (before the gap) so the next iteration's carry
                        # reflects how far SONIC actually got, not where it
                        # was commanded to be.
                        from gear_sonic.utils.planner.blending import (
                            yaw_of_quat_xyzw,
                        )
                        observed = _measure_now()
                        if observed is not None:
                            carry_xy = (observed[0], observed[1])
                            carry_yaw = observed[2]
                        else:
                            carry_xy = (
                                float(cur.root_trans[-1, 0]),
                                float(cur.root_trans[-1, 1]),
                            )
                            carry_yaw = yaw_of_quat_xyzw(
                                cur.root_rot_xyzw[-1]
                            )
                        # Hold STAND pose at the new carry yaw for
                        # ``loop_gap_s`` so SONIC re-stabilises before the
                        # next iteration. This is the key knob that lets
                        # turn primitives keep adding rotation across
                        # loops -- each iteration starts from a clean
                        # standing state at the previously-achieved yaw,
                        # instead of from whatever mid-stride pose the
                        # last frame happened to be. Stdin keys still
                        # interrupt cleanly.
                        gap_end = time.time() + loop_gap_s
                        # Identity quat (recorder-style idle) -- do NOT
                        # re-derive from carry_yaw; that's what caused the
                        # gantry-spin bug.
                        gap_aborted = False
                        while time.time() < gap_end and not quit_flag["q"]:
                            publisher.publish(
                                joint_pos_mj=warmup_dof,
                                root_quat_xyzw=warmup_quat,
                                frame_index=global_frame_index,
                            )
                            global_frame_index += 1
                            if old_termios is not None:
                                r2, _, _ = select.select(
                                    [_sys.stdin], [], [], 0.0
                                )
                                if r2:
                                    ch2 = _sys.stdin.read(1)
                                    if ch2 in ("x", "X", "\x1b", "\x03"):
                                        quit_flag["q"] = True
                                        break
                                    if ch2 in ("n", "N", "p", "P", "r", "R", " ", "l", "L"):
                                        gap_aborted = True
                                        if ch2 in ("n", "N"):
                                            _record_segment_end(idx, completed=True)
                                            idx = (idx + 1) % len(clips)
                                            _enter_segment(idx, settle_s=sweep_settle_s)
                                        elif ch2 in ("p", "P"):
                                            _record_segment_end(idx, completed=True)
                                            idx = (idx - 1) % len(clips)
                                            _enter_segment(idx, settle_s=sweep_settle_s)
                                        elif ch2 in ("r", "R"):
                                            _record_segment_end(idx, completed=True)
                                            _enter_segment(idx, settle_s=sweep_settle_s)
                                        elif ch2 == " ":
                                            paused = not paused
                                            print(
                                                "[browse-sonic] " + ("paused" if paused else "resumed"),
                                                flush=True,
                                            )
                                        elif ch2 in ("l", "L"):
                                            loop_one = not loop_one
                                            print(
                                                f"[browse-sonic] loop mode = "
                                                f"{'one' if loop_one else 'all (auto-advance)'}",
                                                flush=True,
                                            )
                                        break
                            time.sleep(period_s)
                        if quit_flag["q"] or gap_aborted:
                            next_tick = time.time()
                            continue
                        # After the stand-pose hold, snapshot observed
                        # pose AGAIN -- this is what SONIC actually
                        # achieved after settling. Re-align the clip to
                        # this fresh carry so the next iteration starts
                        # from a known-good stable state.
                        observed_after = _measure_now()
                        if observed_after is not None:
                            carry_xy = (observed_after[0], observed_after[1])
                            carry_yaw = observed_after[2]
                        clips[idx] = _aligned_clip(
                            clips[idx], carry_xy, carry_yaw
                        )
                        print(
                            f"[browse-sonic] loop end -> "
                            f"carry_xy=({carry_xy[0]:+.3f},{carry_xy[1]:+.3f}) "
                            f"carry_yaw={np.degrees(carry_yaw):+.1f}deg "
                            f"(source: {'observed' if observed_after is not None else 'commanded'}, "
                            f"gap={loop_gap_s:.2f}s, stand-hold)",
                            flush=True,
                        )
                        # Reset segment timing.
                        seg_t_start = time.time()
                        seg_pose_start = _measure_now()
                        next_tick = time.time()
                        frame = 0
                    else:
                        # Auto-advance at end-of-clip (non-sweep mode).
                        if not sweep_mode:
                            _record_segment_end(idx, completed=True)
                            idx = (idx + 1) % len(clips)
                            _enter_segment(idx, settle_s=sweep_settle_s)
                        else:
                            # Sweep mode handles advancement above; just hold.
                            frame = cur.dof.shape[0] - 1
            else:
                cur = clips[idx]
                hold_frame = max(0, min(frame, cur.dof.shape[0] - 1))
                publisher.publish(
                    joint_pos_mj=cur.dof[hold_frame],
                    root_quat_xyzw=cur.root_rot_xyzw[hold_frame],
                    frame_index=global_frame_index,
                )
                global_frame_index += 1

            next_tick += period_s
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()

        return 0
    finally:
        try:
            if old_termios is not None:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_termios)
        except Exception:
            pass
        if pose_sub is not None:
            pose_sub.close()
        if publisher is not None:
            publisher.close()
        _kill_child(deploy_proc, deploy_log=deploy_log)
        if eval_rows:
            try:
                _write_eval_report(eval_rows, eval_report_path)
                print(f"[browse-sonic] eval report -> {eval_report_path}",
                      flush=True)
            except Exception as exc:
                print(f"[browse-sonic] eval report write failed: {exc}",
                      flush=True)
        print(f"[browse-sonic] deploy log preserved at: {deploy_log}", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_bins(primitives: dict[str, dict]) -> None:
    print(f"\nCurated primitives in {PRIMITIVES_PKL}:\n")
    print(f"  {'bin':<25} {'frames':>7} {'fps':>5} {'status':<8} {'arms_frozen':<11} motion_key")
    print(f"  {'-' * 25} {'-' * 7} {'-' * 5} {'-' * 8} {'-' * 11} {'-' * 40}")
    for name in sorted(primitives):
        p = primitives[name]
        status = "PARTIAL" if p.get("partial") else "OK"
        frozen = "yes" if p.get("freeze_arms_to_default") else "no"
        print(
            f"  {name:<25} {p['n_frames']:>7} {p['fps']:>5g} "
            f"{status:<8} {frozen:<11} {p['motion_key']}"
        )
    print("")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="browse_x2_planner_primitives",
        description=(
            "Browse curated X2 planner primitives or arbitrary clip windows "
            "in a MuJoCo viewer. Use to review what the curator picked, "
            "audition candidates from the markdown report, and decide what "
            "to pin into the registry."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--bin", default=None,
        help="Curated bin name (e.g. fwd_walk_standard). Use --list to see all.",
    )
    src.add_argument(
        "--all", action="store_true",
        help="Cycle through every curated primitive; N/P navigates in viewer.",
    )
    src.add_argument(
        "--motion-key", default=None,
        help="Raw motion_key from the source bones-seed PKL "
             "(must use --start and --n with this).",
    )
    src.add_argument(
        "--list", action="store_true",
        help="Print the curated primitives table to stdout and exit.",
    )
    p.add_argument(
        "--candidate", type=int, default=None, metavar="RANK",
        help="With --bin BIN: play the RANK-th candidate listed in the "
             "markdown report (1-indexed). Use to audition alternatives.",
    )
    p.add_argument(
        "--start", type=int, default=None,
        help="Start frame for --motion-key.",
    )
    p.add_argument(
        "--n", type=int, default=None,
        help="Window length (frames) for --motion-key.",
    )
    p.add_argument(
        "--freeze", default=None, metavar="GROUPS",
        help="Comma-separated DOF groups to pin to DEFAULT_STAND_POSE on a "
             "raw --motion-key view (e.g. 'arms', 'arms,head', 'arms,head,waist'). "
             "Same semantics as the recipe DSL's freeze op. Use to preview a "
             "candidate clip with arm/head/waist swing stripped before pinning "
             "it into the registry. Allowed groups: "
             "left_leg, right_leg, legs, left_arm, right_arm, arms, head, "
             "waist, waist_yaw, waist_pitch.",
    )
    p.add_argument(
        "--anchor-xy", action="store_true",
        help="Lock pelvis XY at origin (default: follow clip's world XY).",
    )
    p.add_argument(
        "--speed", type=float, default=1.0,
        help="Playback speed multiplier (default: 1.0).",
    )
    loop_grp = p.add_mutually_exclusive_group()
    loop_grp.add_argument(
        "--no-loop", action="store_true",
        help="Stop at end of clip instead of looping. This is the DEFAULT for "
             "single-clip views (--bin / --motion-key) so you can study the "
             "landing pose without it whisking back to frame 0. --all ignores "
             "this and always cycles.",
    )
    loop_grp.add_argument(
        "--loop", action="store_true",
        help="Force looping a single-clip view. In kinematic mode this rewinds "
             "to frame 0; in --with-sonic mode this enables the loop-one path "
             "(stand-pose recovery between iterations, --loop-gap-s pause). "
             "Default for single-clip views is play-once-then-hold so you can "
             "watch SONIC settle the landing pose without rewinding.",
    )
    sonic = p.add_argument_group(
        "SONIC sim mode (publish to deploy_x2.sh sim --vla --sim-viewer)",
        description=(
            "When --with-sonic is set, the local kinematic viewer is dropped "
            "and the browser becomes a ZMQ pose publisher. The SONIC sim "
            "deploy is spawned as a child docker container and its MuJoCo "
            "window IS the visual. Stdin (cbreak TTY) drives navigation."
        ),
    )
    sonic.add_argument(
        "--with-sonic", action="store_true",
        help="Stream the current primitive into a child SONIC sim deploy.",
    )
    sonic.add_argument(
        "--sonic-port", type=int, default=5556,
        help="ZMQ pose port the deploy subscribes to (default: 5556).",
    )
    sonic.add_argument(
        "--sonic-host", default="127.0.0.1",
        help="ZMQ bind host (default: 127.0.0.1).",
    )
    sonic.add_argument(
        "--sonic-profile", default="manual",
        choices=("manual", "gantry", "handoff", "parity"),
        help="Sim profile passed to deploy_x2.sh (default: manual).",
    )
    sonic.add_argument(
        "--sonic-duration-s", type=float, default=3600.0,
        help="--max-duration handed to the deploy child (default: 3600s).",
    )
    sonic.add_argument(
        "--no-sonic-viewer", action="store_true",
        help="Run SONIC headless (no MuJoCo window). Useful for log capture.",
    )
    sonic.add_argument(
        "--sonic-auto-advance", action="store_true",
        help="Default loop mode = walk-through-all (auto-advance at end of "
             "each primitive). Without this flag the current primitive loops "
             "until you press N/P. Toggle anytime in-session with L.",
    )
    sonic.add_argument(
        "--sonic-checkpoint", type=Path, default=None,
        help="Path to SONIC .pt checkpoint (e.g. .../model_step_025000.pt). "
             "Used only to derive '<dir>/exported/*.onnx' for deploy --model. "
             "Required unless --deploy-model or --deploy-model-dir is set.",
    )
    sonic.add_argument(
        "--deploy-model", type=Path, default=None,
        help="Explicit path to the deploy ONNX bundle (overrides "
             "--sonic-checkpoint resolution).",
    )
    sonic.add_argument(
        "--deploy-model-dir", type=Path, default=None,
        help="Directory containing 'exported/*.onnx' (overrides "
             "--sonic-checkpoint resolution).",
    )
    sonic.add_argument(
        "--wrist-bypass", default="ik",
        choices=("ik", "off", "passthrough"),
        help="Wrist bypass policy passed to deploy_x2.sh (default: ik).",
    )
    sonic.add_argument(
        "--sonic-boot-timeout-s", type=float, default=180.0,
        help="Seconds to wait for SONIC deploy to print 'Launching ...' "
             "(default: 180).",
    )
    sonic.add_argument(
        "--robot-pose-port", type=int, default=5570,
        help="Port the bridge's robot_pose PUB binds on (default 5570). "
             "Used for per-primitive distance/heading capture.",
    )
    sonic.add_argument(
        "--robot-pose-host", default="127.0.0.1",
        help="Host the browser SUBs to for robot_pose telemetry.",
    )
    sonic.add_argument(
        "--sweep", action="store_true",
        help="Auto-cycle through every selected primitive without TTY input. "
             "Each primitive plays kinematically, holds for --sweep-hold-s, "
             "then advances. Per-primitive XY/yaw deltas are captured and "
             "written to a markdown report. Use to evaluate motion quality "
             "without manual stepping.",
    )
    sonic.add_argument(
        "--sweep-hold-s", type=float, default=4.0,
        help="Seconds to hold each primitive's last frame after kinematic "
             "playback ends, giving SONIC time to settle (default 4.0).",
    )
    sonic.add_argument(
        "--sweep-settle-s", type=float, default=1.5,
        help="Seconds to hold the previous primitive's last frame BEFORE "
             "snapshotting start pose for the next (default 1.5).",
    )
    sonic.add_argument(
        "--initial-settle-s", type=float, default=1.5,
        help="In --sweep / --sonic-auto-advance mode, seconds to hold the "
             "canonical stand pose AFTER gantry drop before auto-launching "
             "the first primitive (default 1.5). In INTERACTIVE mode the "
             "browser holds stand pose INDEFINITELY until you press N, so "
             "this flag has no effect there -- you're always in control of "
             "when the first move fires.",
    )
    sonic.add_argument(
        "--loop-gap-s", type=float, default=1.0,
        help="In loop-one mode, hold the last published frame for this "
             "many seconds between loop iterations (default 1.0). Gives "
             "SONIC time to fully settle at the commanded end pose so "
             "the next loop's carry snapshot reflects the actual robot "
             "state, and makes per-loop progress visually obvious. "
             "Set to 0.0 for back-to-back replays.",
    )
    sonic.add_argument(
        "--eval-report", type=Path, default=None,
        help="Markdown path for the per-primitive eval report. Defaults to "
             "data/sim_to_real_anchors/browse_sonic/eval_<timestamp>.md.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    primitives = load_primitives()

    if args.list or (
        not args.bin and not args.all and not args.motion_key
    ):
        list_bins(primitives)
        if args.list:
            return 0
        # Bare invocation -- list and exit so the user knows the names.
        print(
            "Pick one of: --bin NAME    --all    --motion-key KEY --start S --n N",
            file=sys.stderr,
        )
        return 0

    # Resolve the clip list once; the player choice (kinematic vs SONIC)
    # is independent of which clips were selected.
    clips: list[Clip] = []
    if args.motion_key is not None:
        if args.start is None or args.n is None:
            print(
                "[browse] --motion-key requires --start and --n",
                file=sys.stderr,
            )
            return 2
        freeze_groups: tuple[str, ...] = ()
        if args.freeze:
            freeze_groups = tuple(
                g.strip() for g in args.freeze.split(",") if g.strip()
            )
        clips = [
            clip_from_raw_window(
                args.motion_key, args.start, args.n,
                freeze_groups=freeze_groups,
            )
        ]
    elif args.bin is not None:
        if args.candidate is not None:
            clips = [candidate_from_report(args.bin, args.candidate)]
        else:
            if args.bin not in primitives:
                print(
                    f"[browse] unknown bin {args.bin!r}. Run with --list to "
                    "see available bins.",
                    file=sys.stderr,
                )
                return 2
            clips = [clip_from_primitive(args.bin, primitives[args.bin])]
    elif args.all:
        clips = [
            clip_from_primitive(name, primitives[name])
            for name in sorted(primitives)
        ]

    if not clips:
        return 0

    if args.with_sonic:
        try:
            model_onnx = _resolve_deploy_model_onnx(
                args.sonic_checkpoint, args.deploy_model, args.deploy_model_dir,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[browse-sonic] {exc}", file=sys.stderr)
            return 2
        # SONIC single-clip view defaults to play-once-then-hold so the user
        # can observe how the policy settles the landing pose. Pass --loop to
        # get the previous continuous-loop behaviour (with stand-pose gap).
        # --all / --sonic-auto-advance / --sweep all keep their own end-of-clip
        # handling and disable play-once.
        sonic_play_once = (
            len(clips) == 1
            and not args.loop
            and not args.all
            and not args.sweep
            and not args.sonic_auto_advance
        )
        return play_clips_via_sonic(
            clips,
            pub_host=args.sonic_host,
            pub_port=args.sonic_port,
            sim_profile=args.sonic_profile,
            max_duration_s=args.sonic_duration_s,
            show_viewer=not args.no_sonic_viewer,
            auto_advance=args.sonic_auto_advance,
            model_onnx=model_onnx,
            wrist_bypass=args.wrist_bypass,
            boot_timeout_s=args.sonic_boot_timeout_s,
            pose_sub_host=args.robot_pose_host,
            pose_sub_port=args.robot_pose_port,
            sweep_mode=args.sweep,
            sweep_hold_s=args.sweep_hold_s,
            sweep_settle_s=args.sweep_settle_s,
            loop_gap_s=args.loop_gap_s,
            initial_settle_s=args.initial_settle_s,
            play_once=sonic_play_once,
            eval_report_path=args.eval_report,
        )

    # Single-clip kinematic view defaults to stopping at end-of-clip so the
    # user can study the landing pose. --all keeps cycling. --loop overrides
    # the new default; --no-loop is now redundant for single-clip but kept
    # for backward compat.
    if args.all:
        no_loop = False
    elif args.loop:
        no_loop = False
    else:
        no_loop = True
    return play_clips(clips, args.anchor_xy, args.speed, no_loop)


if __name__ == "__main__":
    raise SystemExit(main())
