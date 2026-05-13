"""Runtime state machine for the X2 heuristic planner.

Pure logic: no ZMQ, no threads, no time.sleep. The CLI script
``gear_sonic/scripts/x2_heuristic_planner.py`` drives this with a fixed-
period loop and a command queue.

State machine:

  - IDLE_LOOP   : looping the ``idle_stand`` primitive at 50 Hz while the
                  command queue is empty.
  - PLAYING     : stepping forward through a non-idle primitive.
  - BLENDING    : synthesizing the blend window between two segments
                  (current-pose-frozen XY, SLERP'd root rot, LERP'd dof).

Transitions:

  IDLE_LOOP   -- enqueue(cmd) --> BLENDING(idle, cmd_primitive)
  BLENDING    -- frames done  --> PLAYING(cmd_primitive)
  PLAYING     -- frames done  --> BLENDING(cmd_primitive, idle)
                                   OR BLENDING(cmd_primitive, next_cmd)
                                   if another command arrived while playing.
  BLENDING    -- frames done  --> IDLE_LOOP / PLAYING
"""

from __future__ import annotations

import dataclasses
import enum
import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .blending import (
    build_blend_window,
    resample_motion_30_to_50hz,
    yaw_align_segment,
    yaw_of_quat_xyzw,
)
from .constants import (
    DEFAULT_PELVIS_Z_M,
    DEFAULT_STAND_POSE_NP,
    IDENTITY_ROOT_QUAT_XYZW,
    NUM_BODY_DOFS,
)


log = logging.getLogger("x2_planner")


# Blend windows in frames at the OUTPUT rate (50 Hz).
BLEND_FRAMES_END_AT_SQUARE: int = 6
BLEND_FRAMES_CONTINUOUS_WALK: int = 12
BLEND_FRAMES_STATIC_UPPER_BODY: int = 16

OUTPUT_FPS: float = 50.0


# ---------------------------------------------------------------------------
# Command interface
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LocomotionCommand:
    """High-level command emitted by command sources (scripted, ZMQ, keyboard)."""

    intent: str  # idle / walk / fwd_step / back_step
                  # / side_left / side_right / turn_left / turn_right
                  # / lean_fwd / torso_left / torso_right / crouch
    magnitude: str  # default / forward / backward / one_ft / half_ft / quarter_ft
                    # / deg_15 / deg_30 / deg_45 / deg_90 / small / medium / large
    source: str = "scripted"  # for logging only

    def as_bin_name(self) -> str:
        """Map (intent, magnitude) to a registry bin name."""
        if self.intent == "idle":
            return "idle_stand"
        if self.intent == "walk" and self.magnitude == "forward":
            return "fwd_walk_standard"
        if self.intent == "walk" and self.magnitude == "backward":
            return "back_walk_standard"
        # Side steps collapse to a single canonical bin per direction
        # regardless of magnitude (the policy could not reliably track
        # smaller / larger scaled variants of the A038_M mocap source,
        # so we keep one bin per side and ignore the magnitude token).
        if self.intent in ("side_left", "side_right"):
            return f"{self.intent}_step"
        # Forward shuffle currently collapses to fwd_step_1ft for ALL
        # magnitudes. The half_ft / quarter_ft scaled derivatives
        # (~22 cm / ~11 cm body-frame translation) sit too close to the
        # policy's tracking noise floor on the v5 A034 forward base, so
        # the smaller variants barely move under SONIC. Rather than ship
        # bins the policy ignores, we route every fwd_step magnitude to
        # the fully tracked 1ft step. Re-introduce magnitude branching
        # once we have per-magnitude mocap sources (or a faster source
        # that survives 0.5x / 0.25x scaling) instead of one base + scale.
        if self.intent == "fwd_step":
            return "fwd_step_1ft"
        # Back step has NO 1ft variant in the current primitive library
        # (only back_step_half_ft and back_step_quarter_ft), and the
        # default magnitude was previously falling through to the
        # generic ``{intent}_{suffix}`` path which produced the
        # nonexistent bin name ``back_step_default`` -- so every Quest 3
        # back_step request silently degraded to idle_stand and the
        # robot looked frozen. Until a back_step_1ft mocap lands, route
        # every back_step magnitude to back_step_half_ft (the larger of
        # the two existing variants); per the comment above this is
        # near the tracking noise floor but is at least a real reverse
        # primitive instead of a no-op.
        if self.intent == "back_step":
            return "back_step_half_ft"
        # Crouch family currently collapses to crouch_medium for ALL
        # magnitudes. The v4 synthesized small / medium / large
        # variants (3 / 6 / 10 cm pelvis drop, 70-96 deg added knee
        # bend) were under-tracked so heavily that the robot looked
        # like it was doing a low forward lean instead of a squat.
        # crouch_medium was rebased on a real squat-from-stand mocap
        # (~110 deg knee bend, ~34 cm pelvis drop) -- a stronger
        # reference signal that survives policy under-tracking. Add
        # back per-magnitude branching once we have proven mocap
        # candidates for shallower / deeper crouches.
        if self.intent == "crouch":
            return "crouch_medium"
        # Most bins follow {intent}_{magnitude} naming directly:
        #   fwd_step + one_ft     -> fwd_step_one_ft? (NO: bin is fwd_step_1ft)
        # Map magnitude tokens to bin suffixes.
        magnitude_map = {
            "one_ft": "1ft",
            "half_ft": "half_ft",
            "quarter_ft": "quarter_ft",
            "deg_15": "15deg",
            "deg_30": "30deg",
            "deg_45": "45deg",
            "deg_90": "90deg",
            "small": "small",
            "medium": "medium",
            "large": "large",
        }
        suffix = magnitude_map.get(self.magnitude, self.magnitude)
        return f"{self.intent}_{suffix}"


# ---------------------------------------------------------------------------
# Primitive library
# ---------------------------------------------------------------------------


@dataclass
class Primitive:
    """One bin of motion data, resampled to ``OUTPUT_FPS``."""

    bin_name: str
    family: str  # idle / continuous_walk / locomotion / static_upper_body
    dof: np.ndarray  # (T, 31) float32, OUTPUT_FPS
    root_rot_xyzw: np.ndarray  # (T, 4) float32
    root_trans: np.ndarray  # (T, 3) float64
    fps: float  # always OUTPUT_FPS after resample
    loopable: bool
    partial: bool
    motion_key: str

    @property
    def n_frames(self) -> int:
        return int(self.dof.shape[0])


def load_primitives_pkl(
    primitives_pkl: Path, bin_family_lookup: dict[str, str]
) -> dict[str, Primitive]:
    """Load curator-written primitives PKL, resample to 50 Hz, return ``{bin: Primitive}``.

    Args:
      primitives_pkl: path written by ``curate_x2_primitives``.
      bin_family_lookup: ``{bin_name: family_string}`` from the bins YAML.
    """
    raw = joblib.load(primitives_pkl)
    if not isinstance(raw, dict):
        raise ValueError(f"{primitives_pkl}: expected dict")

    out: dict[str, Primitive] = {}
    for bin_name, payload in raw.items():
        family = bin_family_lookup.get(bin_name, "locomotion")
        dof_src = np.asarray(payload["dof"], dtype=np.float32)
        rot_src = np.asarray(payload["root_rot_xyzw"], dtype=np.float32)
        trans_src = np.asarray(payload["root_trans"], dtype=np.float64)
        src_fps = float(payload["fps"])
        if abs(src_fps - OUTPUT_FPS) < 0.5:
            dof_out, rot_out, trans_out = dof_src, rot_src, trans_src
        else:
            dof_out, rot_out, trans_out = resample_motion_30_to_50hz(
                dof_src, rot_src, trans_src, src_fps, OUTPUT_FPS
            )
        out[bin_name] = Primitive(
            bin_name=bin_name,
            family=family,
            dof=dof_out.astype(np.float32),
            root_rot_xyzw=rot_out.astype(np.float32),
            root_trans=trans_out.astype(np.float64),
            fps=OUTPUT_FPS,
            loopable=(family == "idle"),
            partial=bool(payload.get("partial", False)),
            motion_key=str(payload.get("motion_key", bin_name)),
        )
    return out


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class PlannerState(enum.Enum):
    IDLE_LOOP = "idle_loop"
    PLAYING = "playing"
    BLENDING = "blending"


@dataclass
class StreamFrame:
    """One output frame emitted by the state machine."""

    joint_pos_mj: np.ndarray  # (31,) float32
    root_quat_xyzw: np.ndarray  # (4,) float32
    root_xy_world: np.ndarray  # (2,) float64 (advisory, for diagnostics)
    yaw_world_deg: float  # advisory, for diagnostics
    state: PlannerState
    bin_name: str  # current primitive bin
    frame_index: int  # global tick (0-based)
    seam_blend: bool  # True when this frame came from a blend window


@dataclass
class _ActiveSegment:
    """Internal record of the segment currently being streamed."""

    primitive: Primitive
    aligned_dof: np.ndarray  # (T, 31)
    aligned_rot: np.ndarray  # (T, 4)
    aligned_trans: np.ndarray  # (T, 3)
    cursor: int = 0  # next frame index to emit
    is_blend: bool = False
    bin_name_for_log: str = ""

    @property
    def n_frames(self) -> int:
        return int(self.aligned_dof.shape[0])

    @property
    def remaining(self) -> int:
        return max(0, self.n_frames - self.cursor)


def _pick_blend_frames(family_a: str, family_b: str) -> int:
    """Blend window length in 50 Hz frames between two segment families."""
    if family_a == "static_upper_body" or family_b == "static_upper_body":
        return BLEND_FRAMES_STATIC_UPPER_BODY
    if family_a == "continuous_walk" or family_b == "continuous_walk":
        return BLEND_FRAMES_CONTINUOUS_WALK
    return BLEND_FRAMES_END_AT_SQUARE


@dataclass
class HeuristicPlanner:
    """Heuristic motion planner state machine.

    Construct with ``HeuristicPlanner(primitives, ...)`` then call ``step()``
    once per output tick (50 Hz). ``enqueue(cmd)`` is thread-safe (callers must
    hold the deque mutex if they share queues across threads — usually the CLI
    just uses one source thread per priority).
    """

    primitives: dict[str, Primitive]
    initial_xy_world: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    initial_yaw_world: float = 0.0
    log_every_n_ticks: int = 100

    def __post_init__(self) -> None:
        if "idle_stand" not in self.primitives:
            raise ValueError(
                "HeuristicPlanner requires an 'idle_stand' primitive in the registry"
            )
        self._state: PlannerState = PlannerState.IDLE_LOOP
        self._tick: int = 0
        self._cur_xy: np.ndarray = self.initial_xy_world.astype(np.float64).copy()
        self._cur_yaw: float = float(self.initial_yaw_world)
        self._last_dof: np.ndarray = DEFAULT_STAND_POSE_NP.copy()
        self._last_rot: np.ndarray = np.asarray(
            IDENTITY_ROOT_QUAT_XYZW, dtype=np.float32
        )
        self._last_trans: np.ndarray = np.array(
            [self._cur_xy[0], self._cur_xy[1], DEFAULT_PELVIS_Z_M],
            dtype=np.float64,
        )
        self._cmd_queue: deque[LocomotionCommand] = deque()
        self._active: _ActiveSegment | None = None
        self._next_after_active: LocomotionCommand | None = None
        # Set to True by ``step_with_lookahead`` while it's peeking ahead;
        # used to silence the per-tick debug log so a single real tick
        # doesn't emit (num_future * step_ticks + 1) log lines.
        self._in_lookahead: bool = False
        self._start_idle_loop()

    # ------------------------------------------------------------------ API

    def enqueue(self, cmd: LocomotionCommand) -> None:
        """Append a command. Picked up after the active segment finishes."""
        self._cmd_queue.append(cmd)

    def replace_pending(self, cmd: LocomotionCommand) -> int:
        """Drop every pending command, then enqueue ``cmd``.

        Use this for *interactive* sources (keyboard, ZMQ teleop) where the
        user's latest intent should win over any backlog they accumulated by
        mashing keys: when 'a' then 'd' arrive in the same 20 ms tick, the
        backlog is exactly one command long and we want 'd' to win.

        What this method does NOT touch:

        - ``_active``: the currently-playing or currently-blending segment
          continues to its natural end. Preempting a blend or a stride
          mid-flight risks a fall (the policy assumes the next reference is
          a small delta from the previous one), so we never interrupt.
        - ``_next_after_active``: when the active segment is a blend, this
          is the command the blend is *targeting*. Clearing it would leave
          the blend with no destination on completion, which is invalid.

        Returns the number of pending commands that were dropped (0 when the
        queue was already empty). Useful for logging.
        """
        dropped = len(self._cmd_queue)
        self._cmd_queue.clear()
        self._cmd_queue.append(cmd)
        return dropped

    @property
    def state(self) -> PlannerState:
        return self._state

    @property
    def tick(self) -> int:
        return self._tick

    @property
    def queue_depth(self) -> int:
        return len(self._cmd_queue)

    def current_anchor_frame(self) -> StreamFrame:
        """Snapshot the active segment's frame 0 without advancing the cursor.

        This is the canonical "first frame the planner would emit if you
        called ``step()`` right now". It is meant for two callers:

        1. The CLI warmup loop in ``x2_heuristic_planner.py`` -- so that the
           quiet-stand pump publishes the *exact* joints/quat/trans that the
           state machine will switch to on the first real tick. Without this,
           DEFAULT_STAND_POSE_NP + identity quat could be ~30 deg off from
           ``idle_stand[0]`` on some joints (e.g. waist_yaw), causing a jolt
           when warmup ends.

        2. ``bake_planner_rsi_anchor.py`` -- so the bridge can RSI from this
           exact pose under ``--sim-profile parity``, giving us a bit-identical
           spawn = wire-content match (no air-drop, no orientation fight).

        The cursor is **not** advanced; the next ``step()`` call still returns
        frame 0.
        """
        if self._active is None:
            raise RuntimeError(
                "current_anchor_frame() called before _start_idle_loop() ran; "
                "this should be impossible after __post_init__."
            )
        seg = self._active
        return StreamFrame(
            joint_pos_mj=seg.aligned_dof[0].astype(np.float32),
            root_quat_xyzw=seg.aligned_rot[0].astype(np.float32),
            root_xy_world=seg.aligned_trans[0, :2].astype(np.float64).copy(),
            yaw_world_deg=float(np.degrees(self._cur_yaw)),
            state=self._state,
            bin_name=seg.bin_name_for_log or "anchor",
            frame_index=0,
            seam_blend=False,
        )

    def step_with_lookahead(
        self,
        num_future: int = 10,
        step_ticks: int = 5,
    ) -> tuple[StreamFrame, list[StreamFrame]]:
        """Emit the next frame AND peek ``num_future`` future frames.

        The first returned value is exactly what ``step()`` would return
        (live state advances by one tick, identically). The second value
        is a list of ``num_future`` future frames sampled at
        ``step_ticks`` tick spacing, starting from the same emit point:

            future[0] = the frame that step() will return on its NEXT call
                        (i.e. tick + 1 ticks ahead of right now)        ... not quite -- read on.

        Indexing convention (matches the C++ tokenizer obs window):

            future[k] = the frame at tick (current_tick + (k + 1) * step_ticks)

        i.e. future[0] is at tick + step_ticks, future[1] at tick + 2*step_ticks,
        etc. With ``step_ticks = OUTPUT_FPS * DT_FUTURE_REF`` (= 5 at
        50 Hz / 0.1 s), ``future[k]`` lines up with the policy's k-th
        future reference at ``current_time + (k+1) * 0.1 s``.

        The C++ tokenizer at ``tokenizer_obs.cpp`` uses
        ``t_k = current_time + k * DT_FUTURE_REF`` for k=0..9, so frame 0
        IS "current". The publisher should send the current frame
        separately (or in slot 0 of a parallel array); this function
        intentionally returns ONLY the strictly-future frames so callers
        can decide the layout.

        Implementation: snapshot live state, advance by N*step_ticks
        ticks while collecting the future frames, restore live state,
        then call ``step()`` once for real. Future state changes (queue
        pops, segment transitions, blends) are **observed** during the
        peek but **not committed** to live state.

        Args:
          num_future: Number of future frames to collect.
          step_ticks: Tick spacing between consecutive future samples.

        Returns:
          ``(current_frame, future_frames)``. ``current_frame`` is what a
          regular ``step()`` would emit; ``future_frames`` has length
          ``num_future``.
        """
        if num_future < 0 or step_ticks < 1:
            raise ValueError(
                f"num_future={num_future} step_ticks={step_ticks}: "
                f"need num_future>=0 and step_ticks>=1"
            )
        snap = self._snapshot()
        # Suppress per-tick log spam during the peek pass.
        prev_lookahead = self._in_lookahead
        self._in_lookahead = True
        try:
            # Burn the current tick first so the future window starts at
            # tick + step_ticks (matching the indexing contract above).
            _ = self.step()
            future: list[StreamFrame] = []
            for _ in range(num_future):
                # Step (step_ticks - 1) silent ticks, then capture one.
                for _ in range(step_ticks - 1):
                    self.step()
                future.append(self.step())
        finally:
            self._restore(snap)
            self._in_lookahead = prev_lookahead
        # Now actually emit the current frame and commit live state by
        # exactly one tick. This is what the caller would have gotten
        # from a plain step() call.
        actual_current = self.step()
        return actual_current, future

    # ----- snapshot / restore (used by step_with_lookahead) ------------

    def _snapshot(self) -> dict[str, Any]:
        """Capture all mutable state needed to roll back a peek pass.

        Shallow-copies the active segment via ``dataclasses.replace``;
        the underlying ``aligned_*`` numpy arrays are treated as
        immutable (they're written once at segment construction and
        only read thereafter). Cursor is an int and is captured
        by-value via the replace.
        """
        return {
            "state": self._state,
            "tick": self._tick,
            "cur_xy": self._cur_xy.copy(),
            "cur_yaw": float(self._cur_yaw),
            "last_dof": self._last_dof.copy(),
            "last_rot": self._last_rot.copy(),
            "last_trans": self._last_trans.copy(),
            "active": (
                dataclasses.replace(self._active) if self._active is not None else None
            ),
            "cmd_queue": deque(self._cmd_queue),
            "next_after_active": self._next_after_active,
        }

    def _restore(self, snap: dict[str, Any]) -> None:
        self._state = snap["state"]
        self._tick = snap["tick"]
        self._cur_xy = snap["cur_xy"]
        self._cur_yaw = snap["cur_yaw"]
        self._last_dof = snap["last_dof"]
        self._last_rot = snap["last_rot"]
        self._last_trans = snap["last_trans"]
        self._active = snap["active"]
        self._cmd_queue = snap["cmd_queue"]
        self._next_after_active = snap["next_after_active"]

    def step(self) -> StreamFrame:
        """Emit the next frame and advance one tick."""
        # Interrupt the cheap idle loop as soon as a command arrives;
        # otherwise a gait command would have to wait up to one full idle
        # clip (~1.5 s) to take effect.
        if (
            self._state == PlannerState.IDLE_LOOP
            and self._cmd_queue
            and self._active is not None
        ):
            self._advance_segment()
        elif self._active is None or self._active.remaining == 0:
            self._advance_segment()
        assert self._active is not None  # noqa: S101 — invariant after _advance_segment

        seg = self._active
        i = seg.cursor
        dof = seg.aligned_dof[i].astype(np.float32)
        rot = seg.aligned_rot[i].astype(np.float32)
        trans = seg.aligned_trans[i].astype(np.float64)
        seg.cursor += 1
        self._last_dof = dof
        self._last_rot = rot
        self._last_trans = trans
        self._cur_xy = trans[:2].copy()
        self._cur_yaw = yaw_of_quat_xyzw(rot)

        frame = StreamFrame(
            joint_pos_mj=dof,
            root_quat_xyzw=rot,
            root_xy_world=self._cur_xy.copy(),
            yaw_world_deg=float(np.degrees(self._cur_yaw)),
            state=self._state,
            bin_name=seg.bin_name_for_log,
            frame_index=self._tick,
            seam_blend=seg.is_blend,
        )
        self._tick += 1
        if (
            not self._in_lookahead
            and self.log_every_n_ticks
            and self._tick % self.log_every_n_ticks == 0
        ):
            log.debug(
                "tick=%d state=%s bin=%s queue=%d xy=(%.2f,%.2f) yaw=%.1f",
                self._tick,
                self._state.value,
                seg.bin_name_for_log,
                len(self._cmd_queue),
                self._cur_xy[0],
                self._cur_xy[1],
                np.degrees(self._cur_yaw),
            )
        return frame

    # ---------------------------------------------------------------- internals

    def _resolve_command(
        self, cmd: LocomotionCommand
    ) -> Primitive:
        bin_name = cmd.as_bin_name()
        prim = self.primitives.get(bin_name)
        if prim is None:
            log.warning(
                "no primitive for command (%s,%s) -> bin %r; falling back to idle_stand",
                cmd.intent, cmd.magnitude, bin_name,
            )
            return self.primitives["idle_stand"]
        if prim.partial:
            log.warning(
                "command (%s,%s) -> bin %r is PARTIAL; using anyway",
                cmd.intent, cmd.magnitude, bin_name,
            )
        return prim

    def _advance_segment(self) -> None:
        """Pick the next segment to play (or blend), updating state."""
        if self._active is not None and self._active.is_blend:
            # Blend just finished -> play the queued primitive that triggered it.
            cmd = self._next_after_active
            self._next_after_active = None
            assert cmd is not None  # noqa: S101 — set when blend was queued
            prim = self._resolve_command(cmd)
            self._start_play(prim)
            return

        # We just finished a non-blend segment (or starting cold).
        was_idle_loop = (
            self._active is not None
            and self._active.bin_name_for_log == "idle_stand"
        )

        if self._cmd_queue:
            cmd = self._cmd_queue.popleft()
            prim = self._resolve_command(cmd)
            if prim.bin_name == "idle_stand":
                # Blend into idle, then loop idle.
                self._start_blend_to(prim, after_command=None)
            else:
                self._start_blend_to(prim, after_command=cmd)
            return

        # Queue empty: either keep looping idle (cheap) or blend back into idle.
        if was_idle_loop:
            self._start_idle_loop()
            return
        idle = self.primitives["idle_stand"]
        self._start_blend_to(idle, after_command=None)

    def _start_idle_loop(self) -> None:
        idle = self.primitives["idle_stand"]
        # If we're already on idle_stand and finished a loop, restart in place.
        if self._active is not None and self._active.bin_name_for_log == "idle_stand":
            self._active.cursor = 0
            self._state = PlannerState.IDLE_LOOP
            return
        # First time, or coming from another segment (this happens via blend).
        aligned_dof, aligned_rot, aligned_trans = yaw_align_segment(
            idle.dof, idle.root_rot_xyzw, idle.root_trans,
            xy_world=self._cur_xy, yaw_world=self._cur_yaw,
        )
        self._active = _ActiveSegment(
            primitive=idle,
            aligned_dof=aligned_dof,
            aligned_rot=aligned_rot,
            aligned_trans=aligned_trans,
            bin_name_for_log="idle_stand",
            is_blend=False,
        )
        self._state = PlannerState.IDLE_LOOP

    def _start_play(self, prim: Primitive) -> None:
        aligned_dof, aligned_rot, aligned_trans = yaw_align_segment(
            prim.dof, prim.root_rot_xyzw, prim.root_trans,
            xy_world=self._cur_xy, yaw_world=self._cur_yaw,
        )
        self._active = _ActiveSegment(
            primitive=prim,
            aligned_dof=aligned_dof,
            aligned_rot=aligned_rot,
            aligned_trans=aligned_trans,
            bin_name_for_log=prim.bin_name,
            is_blend=False,
        )
        self._state = (
            PlannerState.IDLE_LOOP if prim.bin_name == "idle_stand"
            else PlannerState.PLAYING
        )

    def _start_blend_to(
        self, target_prim: Primitive, after_command: LocomotionCommand | None
    ) -> None:
        """Synthesize a blend window from the current pose to target frame 0."""
        # Align target prim so frame 0 sits at (cur_xy, cur_yaw).
        aligned_dof, aligned_rot, aligned_trans = yaw_align_segment(
            target_prim.dof, target_prim.root_rot_xyzw, target_prim.root_trans,
            xy_world=self._cur_xy, yaw_world=self._cur_yaw,
        )
        from_family = (
            self._active.primitive.family if self._active else "idle"
        )
        n_blend = _pick_blend_frames(from_family, target_prim.family)

        blend_dof, blend_rot, blend_trans = build_blend_window(
            self._last_dof,
            self._last_rot,
            self._last_trans,
            aligned_dof[0],
            aligned_rot[0],
            aligned_trans[0],
            n_blend,
        )
        self._active = _ActiveSegment(
            primitive=target_prim,
            aligned_dof=blend_dof.astype(np.float32),
            aligned_rot=blend_rot.astype(np.float32),
            aligned_trans=blend_trans,
            bin_name_for_log=f"blend->{target_prim.bin_name}",
            is_blend=True,
        )
        self._next_after_active = after_command if after_command is not None else (
            LocomotionCommand(intent="idle", magnitude="default", source="auto-idle")
        )
        self._state = PlannerState.BLENDING


# ---------------------------------------------------------------------------
# Standalone helpers used by the CLI script
# ---------------------------------------------------------------------------


def build_pose_payload(
    frame: StreamFrame,
    motion_token_dim: int = 64,
    hand_dof: int = 10,
    future_frames: list[StreamFrame] | None = None,
    future_dt_s: float = 0.1,
) -> dict[str, np.ndarray]:
    """Build the per-tick pose payload dict the C++ deploy expects.

    Wire-format design notes
    ------------------------
    Single-frame fields (``joint_pos_mj``, ``root_quat_xyzw``,
    ``frame_index``) are kept for backward compatibility with v4
    subscribers that don't understand the future window. New
    ``*_window`` fields carry the strictly-future-frame trajectory the
    C++ tokenizer needs (see ``tokenizer_obs.cpp`` -- the policy uses
    ``t_k = current_time + k * DT_FUTURE_REF`` for k = 0..NUM_FUTURE-1,
    with k=0 == current pose). To keep the wire compact, ``*_window``
    arrays carry only the strictly-future frames; the C++ side
    synthesizes k=0 from ``joint_pos_mj`` / ``root_quat_xyzw``.

    ``future_frames`` is optional: when omitted, the payload contains
    only the legacy single-frame fields, and v5 deploys fall back to
    the v4 single-frame Sample() behaviour. This keeps the planner
    interoperable with old deploy binaries while we roll out the fix.

    Args:
      frame: The current emit frame (k=0 in the policy's tokenizer).
      motion_token_dim: Token dim (zeros until VLA is wired).
      hand_dof: Per-hand DoF (zeros until hand teleop is wired).
      future_frames: Optional list of strictly-future frames at
        ``future_dt_s`` spacing. ``len(future_frames) ==
        NUM_FUTURE_FRAMES - 1 == 9`` is the canonical case for the
        X2 policy.
      future_dt_s: Spacing between consecutive future samples (seconds).
        Must match ``DT_FUTURE_REF`` in the C++ deploy (default 0.1 s).
    """
    payload: dict[str, np.ndarray] = {
        "joint_pos_mj": frame.joint_pos_mj.astype(np.float32),
        "root_quat_xyzw": frame.root_quat_xyzw.astype(np.float32),
        "motion_token": np.zeros(motion_token_dim, dtype=np.float32),
        "left_hand_joints": np.zeros(hand_dof, dtype=np.float32),
        "right_hand_joints": np.zeros(hand_dof, dtype=np.float32),
        "frame_index": np.array([frame.frame_index], dtype=np.int64),
    }
    if future_frames:
        n_future = len(future_frames)
        n_dofs = int(frame.joint_pos_mj.shape[0])
        # joint_pos_mj_future[k] -> joint pos at t + (k+1) * future_dt_s
        jpos_future = np.stack(
            [f.joint_pos_mj.astype(np.float32) for f in future_frames]
        )
        rot_future = np.stack(
            [f.root_quat_xyzw.astype(np.float32) for f in future_frames]
        )
        # joint_vel computed by backward finite-diff at future_dt_s
        # spacing between consecutive future samples; the k=0 future
        # frame uses (future[0] - current) / dt. This matches what the
        # C++ tokenizer expects in its ``frame.joint_vel_mj`` slot.
        prev_jpos = frame.joint_pos_mj.astype(np.float32)[None, :]
        all_jpos = np.concatenate([prev_jpos, jpos_future], axis=0)
        jvel_future = (
            (all_jpos[1:] - all_jpos[:-1]) / max(float(future_dt_s), 1e-6)
        ).astype(np.float32)
        # frame_index is global tick; compute consistent index for each future.
        step_ticks = int(round(future_dt_s * OUTPUT_FPS))
        frame_idx_future = np.array(
            [frame.frame_index + (k + 1) * step_ticks for k in range(n_future)],
            dtype=np.int64,
        )
        assert jpos_future.shape == (n_future, n_dofs)  # noqa: S101
        assert rot_future.shape == (n_future, 4)         # noqa: S101
        payload["joint_pos_mj_future"] = jpos_future
        payload["root_quat_xyzw_future"] = rot_future
        payload["joint_vel_mj_future"] = jvel_future
        payload["frame_index_future"] = frame_idx_future
        payload["future_dt_s"] = np.array([float(future_dt_s)], dtype=np.float32)
    return payload


def commands_from_yaml(yaml_path: Path) -> list[LocomotionCommand]:
    """Parse a scripted-demo YAML into a list of commands.

    Schema::

        commands:
          - intent: walk
            magnitude: forward
          - intent: turn_left
            magnitude: deg_45
          - intent: lean_fwd
            magnitude: medium
          - intent: idle
            magnitude: default
            hold_seconds: 1.5     # optional: sit on idle for N seconds before next cmd

    ``hold_seconds`` (any intent) is sugar for "queue ``round(hold_s)``
    extra idle commands after this one". Each idle command costs ~1 idle
    clip pass (idle_stand is 1.5 s @ 50 Hz), so ``hold_seconds: 1.0``
    yields roughly 1.5 s of stand pose between gait segments. We
    intentionally do NOT freeze the previous segment's last pose
    (that proved error-prone -- side-step clips end mid-stride and
    freezing them looked broken; see DEV_NOTES 2026-05-11).
    """
    import yaml

    raw = yaml.safe_load(Path(yaml_path).read_text())
    if not isinstance(raw, dict) or "commands" not in raw:
        raise ValueError(f"{yaml_path}: expected dict with 'commands' list")
    out: list[LocomotionCommand] = []
    for entry in raw["commands"]:
        intent = str(entry["intent"])
        hold_s = float(entry.get("hold_seconds", 0.0))
        cmd = LocomotionCommand(
            intent=intent,
            magnitude=str(entry.get("magnitude", "default")),
            source=str(entry.get("source", "yaml")),
        )
        out.append(cmd)
        n_idle = int(round(hold_s))  # idle_stand is 1.5 s; use 1-per-second sugar
        for _ in range(n_idle):
            out.append(
                LocomotionCommand(
                    intent="idle", magnitude="default", source="yaml-hold",
                )
            )
    return out


def collect_command_iter(it: Iterable[LocomotionCommand]) -> list[LocomotionCommand]:
    """Materialize an iterator of commands, used by tests."""
    return list(it)


def required_bins_summary(prims: dict[str, Primitive]) -> dict[str, Any]:
    """Summarize which bins are loaded and which are partial."""
    summary: dict[str, Any] = {
        "n_total": len(prims),
        "n_partial": sum(1 for p in prims.values() if p.partial),
        "missing_idle_stand": "idle_stand" not in prims,
        "by_family": {},
    }
    for prim in prims.values():
        summary["by_family"].setdefault(prim.family, []).append(prim.bin_name)
    return summary


__all__ = [
    "BLEND_FRAMES_CONTINUOUS_WALK",
    "BLEND_FRAMES_END_AT_SQUARE",
    "BLEND_FRAMES_STATIC_UPPER_BODY",
    "HeuristicPlanner",
    "LocomotionCommand",
    "OUTPUT_FPS",
    "PlannerState",
    "Primitive",
    "StreamFrame",
    "build_pose_payload",
    "commands_from_yaml",
    "load_primitives_pkl",
    "required_bins_summary",
]
