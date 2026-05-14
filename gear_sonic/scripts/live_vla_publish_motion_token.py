"""Live VLA → SONIC bridge for the X2 MuJoCo deploy (Stage C4 / M7).

This is the production-ish drop-in replacement for
:mod:`mock_vla_publish_stand_token`. It owns the full
"camera + state → VLA → motion-token chunk" loop expected by
``gear_sonic_deploy/deploy_x2.sh sim --vla``.

Architecture
------------

::

    ┌────────────────────┐                 ┌──────────────────────────┐
    │ x2_deploy_onnx_ref │                 │ live_vla_publish_motion_ │
    │ (C++, in container)│  x2_debug PUB   │ token.py (this script)   │
    │                    ├────tcp:5557────►│                          │
    │  MuJoCo + SONIC    │                 │  ┌────────────────────┐  │
    │   tracking policy  │                 │  │ ghost MuJoCo       │  │
    │                    │  pose SUB       │  │ renderer           │  │
    │                    ◄────tcp:5556─────┤  └────────────────────┘  │
    │                    │                 │  ┌────────────────────┐  │
    │                    │                 │  │ Isaac-GR00T N1.7   │  │
    │                    │                 │  │ Gr00tPolicy        │  │
    └────────────────────┘                 │  └────────────────────┘  │
                                           └──────────────────────────┘

Design notes
------------

* The C++ deploy runs MuJoCo in-process. We don't have direct access to its
  rendered ego camera, but the deploy publishes ``body_q`` (31 DOFs in
  MuJoCo joint order) + ``base_quat`` (wxyz) + ``left/right_hand_q`` over
  ``x2_debug``. We mirror that into a *passive* (kinematic) MuJoCo via
  :class:`gear_sonic.scripts.render_smoketest_episode_video.MujocoFrameRenderer`
  and re-render the ``ego_view`` camera. This is the same frame source
  that produced the M5/M6 training data so the VLA sees in-distribution
  pixels.

* Inference latency on a 32 GB RTX 5090 is ~150–300 ms / chunk (much
  slower than the 50 Hz publish cadence the C++ deploy expects). We
  therefore decouple the two: a worker thread runs the VLA as fast as
  it can, posting fresh 40-step action chunks; the main thread is a
  steady 50 Hz publisher that walks the *current* chunk timestep by
  timestep, rolling over to the freshest available chunk whenever the
  worker finishes another inference.

* The state vector handed to ``Gr00tPolicy`` is in **Pinocchio URDF order**
  (matches ``meta/modality.json``: legs / waist / head / arms / hands),
  while ``body_q`` arriving over ``x2_debug`` is in **MuJoCo joint order**
  (legs / waist / arms / head). We pre-compute a ``MJ_TO_PIN`` permutation
  once at startup.

Acceptance gate (Stage C5)
--------------------------

Run the deploy under ``deploy_x2.sh sim --vla``, point this script at
the just-trained checkpoint, and dump telemetry with
``dump_x2_debug.py``. The token norms produced here will be non-zero
(unlike the mock publisher's stand-still latent) and the deploy should
remain upright + tilt-free for at least 10s of policy time.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import signal
import sys
import threading
import time
from typing import Any, Optional

import numpy as np
import zmq

# Allow running this script without `pip install -e gear_sonic` --
# also pulls Isaac-GR00T onto sys.path so ``import gr00t`` resolves
# without the caller having to set PYTHONPATH manually.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
ISAAC_GROOT_ROOT = REPO_ROOT / "external_dependencies" / "Isaac-GR00T"
if ISAAC_GROOT_ROOT.is_dir() and str(ISAAC_GROOT_ROOT) not in sys.path:
    sys.path.insert(0, str(ISAAC_GROOT_ROOT))

from gear_sonic.utils.teleop.zmq.zmq_planner_sender import pack_pose_message  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_packed_message_decoder import unpack_message  # noqa: E402


SONIC_MOTION_TOKEN_DIM: int = 64
DEFAULT_HAND_DOF: int = 10
DEFAULT_PUB_RATE_HZ: float = 50.0
NUM_BODY_DOFS: int = 31  # MuJoCo joint count


# Stand-pose fallback (radians, MuJoCo body order). Mirrors the trained
# default in ``policy_parameters.hpp`` and the mock publisher.
DEFAULT_STAND_POSE_MUJOCO_RAD: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # left leg (6)
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # right leg (6)
    0.0, 0.0, 0.0,                                # waist (3)
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,           # left arm (7)
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,          # right arm (7)
    0.0, 0.0,                                     # head (2)
)
assert len(DEFAULT_STAND_POSE_MUJOCO_RAD) == NUM_BODY_DOFS


# Permutation MJ_TO_PIN[i] = j means: the i-th joint in Pinocchio order
# is the j-th joint in MuJoCo order. The dataset's ``observation.state``
# is written in Pinocchio order, but the C++ deploy publishes ``body_q``
# in MuJoCo order. Both are fixed at build-time so we can hard-code the
# index list (verified against ``rm.joint_names`` and X2_BODY_JOINT_NAMES
# at module import time below).
MJ_TO_PIN: tuple[int, ...] = (
    0, 1, 2, 3, 4, 5,           # left_leg  (pin 0..5  ← mj 0..5)
    6, 7, 8, 9, 10, 11,         # right_leg (pin 6..11 ← mj 6..11)
    12, 13, 14,                 # waist     (pin 12..14 ← mj 12..14)
    29, 30,                     # head      (pin 15..16 ← mj 29..30)
    15, 16, 17, 18, 19, 20, 21, # left arm  (pin 17..23 ← mj 15..21)
    22, 23, 24, 25, 26, 27, 28, # right arm (pin 24..30 ← mj 22..28)
)
assert len(MJ_TO_PIN) == NUM_BODY_DOFS


def _quat_wxyz_to_projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Rotate world ``[0, 0, -1]`` into the body frame using a wxyz quaternion.

    Mirrors the projection used by the IsaacLab observation pipeline:
    ``projected_gravity = R(quat).T @ [0, 0, -1]``.
    """
    w, x, y, z = (float(v) for v in quat_wxyz.reshape(-1))
    # Standard quaternion -> 3x3 rotation matrix (active, world<-body).
    # We need its transpose to bring the world gravity into the body frame.
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - z * w)
    r02 = 2.0 * (x * z + y * w)
    r10 = 2.0 * (x * y + z * w)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - x * w)
    r20 = 2.0 * (x * z - y * w)
    r21 = 2.0 * (y * z + x * w)
    r22 = 1.0 - 2.0 * (x * x + y * y)
    # R.T @ [0, 0, -1] picks out -1 * column 2.
    return np.array([-r02, -r12, -r22], dtype=np.float64)


# If we haven't seen a fresh ``x2_debug`` packet within this many seconds
# we declare the deploy *not* alive and stop both inference + video
# recording. The deploy publishes at 50 Hz, so anything past ~4 ticks
# (80 ms) is already abnormal; we use a generous 1.0 s to absorb
# transient stalls (long colcon build sleeping the bridge thread, GPU
# rendering hiccups, etc.) without falsely declaring death.
DEPLOY_ALIVE_STALE_THRESHOLD_S: float = 1.0


@dataclass
class _LatestState:
    """Thread-safe snapshot of the freshest ``x2_debug`` frame.

    We don't try to maintain a full queue: the inference loop always wants
    the *latest* frame, and a queue would just buffer staleness. A monotonic
    ``revision`` counter lets the inference thread block until something new
    arrives without polling.

    Liveness is tracked via :attr:`last_update_monotonic` (set every time
    ``update`` is called) and queried via :meth:`is_alive` against
    :data:`DEPLOY_ALIVE_STALE_THRESHOLD_S`. ``received_any`` is *only* a
    "have we ever seen a packet" flag and is deliberately one-way; the
    is_alive check is what callers should actually gate on, otherwise the
    bridge keeps recording forever after the deploy exits (the original
    sticky-True behaviour).
    """

    body_q_mj: np.ndarray = field(
        default_factory=lambda: np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float64)
    )
    base_quat_wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    left_hand_q: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_HAND_DOF))
    right_hand_q: np.ndarray = field(default_factory=lambda: np.zeros(DEFAULT_HAND_DOF))
    revision: int = 0
    received_any: bool = False
    last_update_monotonic: float = 0.0
    cv: threading.Condition = field(default_factory=threading.Condition)

    def update(
        self,
        *,
        body_q_mj: np.ndarray,
        base_quat_wxyz: np.ndarray,
        left_hand_q: np.ndarray,
        right_hand_q: np.ndarray,
    ) -> None:
        with self.cv:
            self.body_q_mj = body_q_mj.astype(np.float64, copy=False)
            self.base_quat_wxyz = base_quat_wxyz.astype(np.float64, copy=False)
            self.left_hand_q = left_hand_q.astype(np.float64, copy=False)
            self.right_hand_q = right_hand_q.astype(np.float64, copy=False)
            self.revision += 1
            self.received_any = True
            self.last_update_monotonic = time.monotonic()
            self.cv.notify_all()

    def is_alive(
        self,
        *,
        stale_threshold_s: float = DEPLOY_ALIVE_STALE_THRESHOLD_S,
        now_monotonic: float | None = None,
    ) -> bool:
        """Return True iff we have ever received an ``x2_debug`` packet
        AND the most recent one is younger than ``stale_threshold_s``.

        Note: ``last_update_monotonic`` defaults to 0.0 in the dataclass,
        which means the very first ``is_alive`` call before any packet
        arrives returns False (since ``now - 0.0 >> threshold``). After
        the deploy goes quiet (process exits / Ctrl-C), this flips back
        to False ``stale_threshold_s`` later, so the video recorder can
        gracefully close the writer instead of recording forever.
        """
        if not self.received_any:
            return False
        if now_monotonic is None:
            now_monotonic = time.monotonic()
        return (now_monotonic - self.last_update_monotonic) <= stale_threshold_s

    def snapshot(
        self, *, stale_threshold_s: float = DEPLOY_ALIVE_STALE_THRESHOLD_S
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, bool]:
        """Atomic snapshot. The bool return is ``is_alive``, NOT the
        sticky ``received_any`` (the latter caused the bridge to keep
        recording forever after the deploy exited)."""
        with self.cv:
            now = time.monotonic()
            alive = self.received_any and (
                (now - self.last_update_monotonic) <= stale_threshold_s
            )
            return (
                self.body_q_mj.copy(),
                self.base_quat_wxyz.copy(),
                self.left_hand_q.copy(),
                self.right_hand_q.copy(),
                self.revision,
                alive,
            )

    def wait_for_new(self, last_revision: int, timeout: float) -> int:
        """Block until ``revision`` advances past ``last_revision``."""
        with self.cv:
            self.cv.wait_for(lambda: self.revision > last_revision, timeout=timeout)
            return self.revision


@dataclass
class _LatestChunk:
    """Thread-safe latest-action chunk buffer.

    The publisher walks ``token[step]`` / ``left_hand[step]`` / ``right_hand[step]``
    one tick at a time. When the inference worker posts a new chunk
    (``chunk_id`` increments), the publisher resets ``step = 0``.
    """

    token: np.ndarray = field(default_factory=lambda: np.zeros((40, SONIC_MOTION_TOKEN_DIM), dtype=np.float32))
    left_hand: np.ndarray = field(default_factory=lambda: np.zeros((40, DEFAULT_HAND_DOF), dtype=np.float32))
    right_hand: np.ndarray = field(default_factory=lambda: np.zeros((40, DEFAULT_HAND_DOF), dtype=np.float32))
    body_pose: np.ndarray = field(
        default_factory=lambda: np.tile(np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32), (1, 1))
    )
    chunk_id: int = 0
    inference_count: int = 0  # how many inferences have completed
    last_inference_ms: float = 0.0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def post(
        self,
        *,
        token: np.ndarray,
        left_hand: np.ndarray,
        right_hand: np.ndarray,
        body_pose: np.ndarray,
        last_inference_ms: float,
    ) -> int:
        with self.lock:
            self.token = token.astype(np.float32, copy=False)
            self.left_hand = left_hand.astype(np.float32, copy=False)
            self.right_hand = right_hand.astype(np.float32, copy=False)
            self.body_pose = body_pose.astype(np.float32, copy=False)
            self.chunk_id += 1
            self.inference_count += 1
            self.last_inference_ms = float(last_inference_ms)
            return self.chunk_id

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        with self.lock:
            return (
                self.token,
                self.left_hand,
                self.right_hand,
                self.body_pose,
                self.chunk_id,
            )


def _build_observation(
    *,
    body_q_mj: np.ndarray,
    base_quat_wxyz: np.ndarray,
    left_hand_q: np.ndarray,
    right_hand_q: np.ndarray,
    ego_view: np.ndarray,
    prompt: str,
) -> dict[str, Any]:
    """Build the (B=1, T=1)-batched observation dict expected by ``Gr00tPolicy``.

    The slot layout matches the M5/M6 dataset's ``meta/modality.json``:

    * ``state.left_leg``           = pin[0:6]
    * ``state.right_leg``          = pin[6:12]
    * ``state.waist``              = pin[12:15]
    * ``state.left_arm``           = pin[17:24]
    * ``state.right_arm``          = pin[24:31]
    * ``state.left_hand``          = left_hand_q[:10]
    * ``state.right_hand``         = right_hand_q[:10]
    * ``state.projected_gravity``  = R(base_quat).T @ [0, 0, -1]
    * ``video.ego_view``           = (1, 1, H, W, 3) uint8
    * ``language.annotation.human.task_description`` = [[prompt]]

    Note: head joints (pin[15:17]) are *not* exposed to the policy; they
    are intentionally omitted from ``modality.json``.
    """
    body_q_pin = np.asarray(body_q_mj, dtype=np.float32)[list(MJ_TO_PIN)]
    proj_grav = _quat_wxyz_to_projected_gravity(base_quat_wxyz).astype(np.float32)
    left_h = np.asarray(left_hand_q, dtype=np.float32).reshape(-1)[:DEFAULT_HAND_DOF]
    right_h = np.asarray(right_hand_q, dtype=np.float32).reshape(-1)[:DEFAULT_HAND_DOF]

    def _b1(arr: np.ndarray) -> np.ndarray:
        return arr.reshape(1, 1, -1)

    state = {
        "left_leg": _b1(body_q_pin[0:6]),
        "right_leg": _b1(body_q_pin[6:12]),
        "waist": _b1(body_q_pin[12:15]),
        "left_arm": _b1(body_q_pin[17:24]),
        "right_arm": _b1(body_q_pin[24:31]),
        "left_hand": _b1(left_h),
        "right_hand": _b1(right_h),
        "projected_gravity": _b1(proj_grav),
    }
    return {
        "video": {"ego_view": ego_view.reshape(1, 1, *ego_view.shape).astype(np.uint8)},
        "state": state,
        "language": {"annotation.human.task_description": [[prompt]]},
    }


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------


def _video_recorder(
    *,
    renderer_factory: Any,
    state: _LatestState,
    output_path: str,
    fps: float,
    stop_event: threading.Event,
    chunk: _LatestChunk,
    verbose: bool = False,
) -> None:
    """Thread D: render the live deploy state into an MP4 at ``fps`` Hz.

    Runs *separately* from the inference worker so the recorded video is
    smooth (50 Hz / 25 Hz, decoupled from the ~3 Hz VLA cadence) and so
    its EGL context doesn't share with the inference worker (mujoco's EGL
    backend uses thread-local GL contexts).

    Each frame is the same ``ego_view`` the VLA actually sees during
    inference, so the recorded clip is the policy's first-person camera.
    """
    print(f"[live-VLA] video thread: building MujocoFrameRenderer for {output_path}", flush=True)
    try:
        renderer = renderer_factory()
    except Exception as exc:
        print(f"[live-VLA] WARN: video renderer init failed: {exc}", flush=True)
        return

    # Lazy import keeps the bridge's hard-dep surface small for test
    # environments that don't ship pyav.
    from gear_sonic.data.video_writer import VideoWriter

    writer: Any = None
    period = 1.0 / max(fps, 1e-6)
    next_tick = time.monotonic()
    n_written = 0
    last_revision = -1
    last_tilt_warned = False
    waiting_logged = False
    try:
        while not stop_event.is_set():
            (
                body_q_mj, base_quat_wxyz, left_hq, right_hq, revision, alive
            ) = state.snapshot()

            # Three liveness states for the recorder:
            #   1. Pre-deploy: never received x2_debug. Hold writer closed,
            #      keep polling so the renderer's EGL context stays warm.
            #   2. Deploy alive: ``alive=True`` (recent packet within
            #      DEPLOY_ALIVE_STALE_THRESHOLD_S). Open writer (lazy) and
            #      record.
            #   3. Deploy died: writer is open but ``alive=False`` again
            #      (deploy stopped publishing). Close writer to flush the
            #      MP4, then exit. Without this branch the bridge kept
            #      recording forever after the deploy exited because
            #      ``received_any`` was sticky-True; the file stayed open
            #      and unflushable.
            if not alive:
                if writer is not None:
                    if verbose:
                        print(
                            f"[live-VLA] video: deploy went quiet "
                            f"(no x2_debug for >{DEPLOY_ALIVE_STALE_THRESHOLD_S}s); "
                            f"flushing {n_written} frames to {output_path}",
                            flush=True,
                        )
                    try:
                        # ``VideoWriter.stop()`` finalises the muxer (writes
                        # the moov atom on MP4) and closes the file. The
                        # outer ``finally`` will skip its own stop() call
                        # because we set ``writer = None`` here.
                        writer.stop()
                    except Exception as exc:
                        print(f"[live-VLA] video: writer.stop() warn: {exc}", flush=True)
                    writer = None
                    return  # Exit the recorder thread; nothing more to do.
                if verbose and not waiting_logged:
                    print(
                        f"[live-VLA] video: waiting for first x2_debug frame "
                        f"before opening {output_path} …",
                        flush=True,
                    )
                    waiting_logged = True
                time.sleep(period)
                next_tick = time.monotonic() + period
                continue

            try:
                frame = renderer.render_frame(
                    body_q=body_q_mj,
                    left_active=left_hq.astype(np.float64),
                    right_active=right_hq.astype(np.float64),
                    root_quat_wxyz=base_quat_wxyz,
                )
            except Exception as exc:
                if not last_tilt_warned:
                    print(f"[live-VLA] video render warn (will keep trying): {exc}", flush=True)
                    last_tilt_warned = True
                time.sleep(period)
                continue
            last_tilt_warned = False

            if writer is None:
                # PyAV's add_stream(rate=...) expects an int / Fraction
                # (a plain float trips
                #   AttributeError: 'float' object has no attribute 'numerator'
                # at the first encode call).
                writer = VideoWriter(
                    output_path=output_path,
                    width=renderer.width,
                    height=renderer.height,
                    fps=int(round(fps)),
                    codec="h264",
                    buffer_size=200,
                )
                if verbose:
                    print(
                        f"[live-VLA] video: deploy alive, recording -> "
                        f"{output_path} ({renderer.width}x{renderer.height} "
                        f"@ {int(round(fps))} fps)",
                        flush=True,
                    )

            writer.add_frame(np.ascontiguousarray(frame))
            n_written += 1
            last_revision = revision

            if verbose and n_written % int(max(fps, 1)) == 0:
                _, _, _, _, chunk_id_now = chunk.read()
                print(
                    f"[live-VLA] video: {n_written:5d} frames written  "
                    f"(rev={revision} chunk_id={chunk_id_now} alive={alive})",
                    flush=True,
                )

            next_tick += period
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    finally:
        if writer is not None:
            try:
                writer.stop()
            except Exception as exc:
                print(f"[live-VLA] video writer.stop error: {exc}", flush=True)
        try:
            renderer.close()
        except Exception:
            pass
        print(f"[live-VLA] video thread done; wrote {n_written} frames -> {output_path}", flush=True)


def _x2_debug_subscriber(
    *,
    sub_url: str,
    topic: str,
    state: _LatestState,
    stop_event: threading.Event,
) -> None:
    """Thread A: SUB to ``x2_debug`` and update :class:`_LatestState`."""
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)
    sock.setsockopt(zmq.RCVHWM, 5)
    sock.setsockopt(zmq.LINGER, 0)
    sock.connect(sub_url)
    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    print(f"[live-VLA] x2_debug SUB connected to {sub_url} (topic={topic!r})", flush=True)

    try:
        while not stop_event.is_set():
            events = dict(poller.poll(200))
            if sock not in events:
                continue
            try:
                raw = sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                continue
            try:
                msg = unpack_message(raw, expected_topic=topic)
            except ValueError as exc:
                print(f"[live-VLA] x2_debug decode error: {exc}", flush=True)
                continue

            body_q = np.asarray(msg.fields.get("body_q", DEFAULT_STAND_POSE_MUJOCO_RAD), dtype=np.float64).reshape(-1)
            if body_q.shape[0] != NUM_BODY_DOFS:
                continue
            base_quat = np.asarray(msg.fields.get("base_quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64).reshape(-1)
            if base_quat.shape[0] != 4:
                base_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            left_hq = np.asarray(msg.fields.get("left_hand_q", np.zeros(DEFAULT_HAND_DOF)), dtype=np.float64).reshape(-1)[:DEFAULT_HAND_DOF]
            right_hq = np.asarray(msg.fields.get("right_hand_q", np.zeros(DEFAULT_HAND_DOF)), dtype=np.float64).reshape(-1)[:DEFAULT_HAND_DOF]
            if left_hq.shape[0] < DEFAULT_HAND_DOF:
                left_hq = np.pad(left_hq, (0, DEFAULT_HAND_DOF - left_hq.shape[0]))
            if right_hq.shape[0] < DEFAULT_HAND_DOF:
                right_hq = np.pad(right_hq, (0, DEFAULT_HAND_DOF - right_hq.shape[0]))

            state.update(
                body_q_mj=body_q,
                base_quat_wxyz=base_quat,
                left_hand_q=left_hq,
                right_hand_q=right_hq,
            )
    finally:
        try:
            sock.close(linger=0)
        except Exception:
            pass


def _inference_worker(
    *,
    policy: Any,
    renderer_factory: Any,
    state: _LatestState,
    chunk: _LatestChunk,
    prompt: str,
    stop_event: threading.Event,
    min_period_s: float,
    verbose: bool = False,
    dump_chunks_dir: str | None = None,
    dump_chunks_every: int = 5,
) -> None:
    """Thread B: render + run VLA continuously, post fresh chunks.

    We don't run faster than ``1/min_period_s`` Hz to avoid burning GPU
    cycles when the deploy hasn't even consumed the previous chunk yet.

    The MuJoCo renderer is constructed *inside* this thread because EGL
    contexts are thread-local: a renderer instantiated in the main thread
    raises ``EGLError(EGL_BAD_DISPLAY)`` the first time another thread
    tries to call ``render_frame``. Building it here owns the EGL context
    on the same thread that will use it for the entire process lifetime.
    """
    print("[live-VLA] inference thread: building MujocoFrameRenderer …", flush=True)
    try:
        renderer = renderer_factory()
    except Exception as exc:
        print(f"[live-VLA] FATAL: renderer init failed in worker: {exc}", flush=True)
        stop_event.set()
        return
    print(
        f"[live-VLA] inference thread: renderer ready "
        f"({renderer.width}x{renderer.height}, omnihand={renderer.with_omnihand})",
        flush=True,
    )

    last_revision = -1
    n_inferences = 0
    dump_dir_path: Path | None = None
    if dump_chunks_dir:
        dump_dir_path = Path(dump_chunks_dir)
        dump_dir_path.mkdir(parents=True, exist_ok=True)
        print(
            f"[live-VLA] inference thread: chunk dump enabled "
            f"(dir={dump_dir_path}, every={dump_chunks_every})",
            flush=True,
        )
    while not stop_event.is_set():
        # Block until a new x2_debug frame arrives (don't run inference on
        # stale state).
        rev = state.wait_for_new(last_revision, timeout=0.5)
        if rev <= last_revision:
            continue  # timeout, retry

        body_q_mj, base_quat_wxyz, left_hq, right_hq, revision, received = state.snapshot()
        if not received:
            continue
        last_revision = revision

        t0 = time.monotonic()
        try:
            # The inference ego-view deliberately uses the renderer's
            # default identity root quat -- the VLA training set
            # (record_synthetic_smoketest_dataset.py) renders frames the
            # same way, so feeding a live ``base_quat`` here would
            # inject a tilted horizon the policy has never seen and
            # cause OOD behaviour mid-rollout. The *video recorder*
            # thread does pass the live ``base_quat`` so the recording
            # reflects physical reality (tip / fall) -- those two
            # renderers are intentionally asymmetric.
            ego_view = renderer.render_frame(
                body_q=body_q_mj,
                left_active=left_hq.astype(np.float64),
                right_active=right_hq.astype(np.float64),
            )
        except Exception as exc:
            print(f"[live-VLA] render error: {exc}", flush=True)
            time.sleep(0.05)
            continue

        observation = _build_observation(
            body_q_mj=body_q_mj,
            base_quat_wxyz=base_quat_wxyz,
            left_hand_q=left_hq,
            right_hand_q=right_hq,
            ego_view=ego_view,
            prompt=prompt,
        )

        try:
            action, _info = policy.get_action(observation)
        except Exception as exc:
            print(f"[live-VLA] inference error: {exc}", flush=True)
            time.sleep(0.05)
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        # Action shape: (B=1, T_horizon, D). Drop the batch axis.
        token = np.asarray(action["motion_token"], dtype=np.float32)[0]   # (T, 64)
        left = np.asarray(action["left_hand_joints"], dtype=np.float32)[0]  # (T, 10)
        right = np.asarray(action["right_hand_joints"], dtype=np.float32)[0]  # (T, 10)

        # The C++ deploy doesn't actually consume body_pose when motion_token
        # is the live source-of-truth, but we keep DEFAULT_STAND_POSE on the
        # wire so legacy tooling (dump_x2_debug.py / parity scripts) still
        # see a valid "joint_pos_mj" field.
        body_pose = np.array(DEFAULT_STAND_POSE_MUJOCO_RAD, dtype=np.float32)

        chunk.post(
            token=token,
            left_hand=left,
            right_hand=right,
            body_pose=body_pose,
            last_inference_ms=elapsed_ms,
        )
        n_inferences += 1
        if dump_dir_path is not None and (n_inferences - 1) % max(dump_chunks_every, 1) == 0:
            try:
                out_path = dump_dir_path / f"chunk_{n_inferences - 1:05d}.npz"
                np.savez_compressed(
                    out_path,
                    token=token,
                    left_hand=left,
                    right_hand=right,
                    body_q_mj=body_q_mj.astype(np.float32),
                    base_quat_wxyz=base_quat_wxyz.astype(np.float32),
                    left_hand_q_obs=left_hq.astype(np.float32),
                    right_hand_q_obs=right_hq.astype(np.float32),
                    ego_view=ego_view.astype(np.uint8),
                    elapsed_ms=np.array([elapsed_ms], dtype=np.float32),
                    revision=np.array([revision], dtype=np.int64),
                    n_inference=np.array([n_inferences - 1], dtype=np.int64),
                    wall_t_s=np.array([time.time()], dtype=np.float64),
                )
            except Exception as exc:
                print(f"[live-VLA] chunk dump failed: {exc}", flush=True)
        if verbose:
            print(
                f"[live-VLA] inference #{n_inferences:04d}  "
                f"elapsed={elapsed_ms:6.1f} ms  "
                f"|token[0]|={float(np.linalg.norm(token[0])):.3f}  "
                f"|left[0]|={float(np.linalg.norm(left[0])):.3f}  "
                f"horizon={token.shape[0]}",
                flush=True,
            )

        # Pace ourselves to honour --inference-min-period-s. The publisher
        # walks the chunk at 50 Hz so a horizon=40 chunk represents 0.8 s
        # of motion; running inference faster than that means each new
        # chunk yanks the publisher back to step 0 mid-chunk and the
        # remaining steps are discarded (visible as a 1/min_period_s-Hz
        # spike pattern in the video).
        #
        # Loop the sleep in <=100 ms slices so we stay responsive to
        # stop_event during long pauses (the previous single-shot sleep
        # had a hard-coded 0.5 s cap which silently broke any min_period_s
        # > 0.5 s -- the L3 fix was a no-op as a result).
        if min_period_s > 0:
            while not stop_event.is_set():
                slack = min_period_s - (time.monotonic() - t0)
                if slack <= 0:
                    break
                time.sleep(min(slack, 0.1))

    try:
        renderer.close()
    except Exception:
        pass


def _publisher(
    *,
    pub_sock: zmq.Socket,
    topic: str,
    rate_hz: float,
    chunk: _LatestChunk,
    state: _LatestState,
    duration_s: float,
    stop_event: threading.Event,
    protocol_version: int = 4,
    print_every: int = 50,
) -> int:
    """Thread C (= main thread): publish action[step] at ``rate_hz``.

    Always publishes *something*, even before the first inference completes.
    The bootstrap chunk is zeros + DEFAULT_STAND_POSE — exactly what the
    mock VLA helper publishes — so the deploy's safety stack can keep the
    robot upright until the policy warms up.
    """
    period = 1.0 / max(rate_hz, 1e-6)
    next_tick = time.monotonic()
    deadline = float("inf") if duration_s <= 0 else time.monotonic() + duration_s

    last_chunk_id = -1
    chunk_step = 0
    horizon = 40
    tick = 0

    while not stop_event.is_set() and time.monotonic() < deadline:
        token, left, right, body_pose, chunk_id = chunk.read()
        horizon = int(token.shape[0])
        if chunk_id != last_chunk_id:
            chunk_step = 0
            last_chunk_id = chunk_id

        step = min(chunk_step, horizon - 1)
        payload = {
            "joint_pos_mj": body_pose,
            "root_quat_xyzw": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "motion_token": token[step],
            "left_hand_joints": left[step],
            "right_hand_joints": right[step],
            "frame_index": np.array([tick], dtype=np.int64),
        }
        msg = pack_pose_message(payload, topic=topic, version=protocol_version)
        try:
            pub_sock.send(msg, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

        if tick % print_every == 0:
            _, _, _, _, _, alive = state.snapshot()
            print(
                f"[live-VLA] pub tick={tick:6d} "
                f"chunk_id={chunk_id:4d} step={step:2d}/{horizon} "
                f"|token|={float(np.linalg.norm(token[step])):.3f} "
                f"|left|={float(np.linalg.norm(left[step])):.3f} "
                f"deploy_alive={alive}",
                flush=True,
            )

        chunk_step = min(chunk_step + 1, horizon - 1)
        tick += 1
        next_tick += period
        sleep_for = next_tick - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        else:
            next_tick = time.monotonic()
    return tick


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------


def _validate_pin_order_or_die() -> None:
    """Best-effort runtime check that ``MJ_TO_PIN`` matches the X2 schema.

    Falls back to a warning if ``gear_sonic.data.features_x2_vla`` can't be
    imported (e.g. minimal-deps test env): the mapping is hard-coded so a
    crash at runtime would be visibly miscalibrated joints anyway.
    """
    try:
        from gear_sonic.data.features_x2_vla import get_x2_robot_model
        from gear_sonic.data.robot_model.supplemental_info.x2_ultra.x2_ultra_supplemental_info import (
            X2_BODY_JOINT_NAMES,
        )
    except Exception as exc:
        print(f"[live-VLA] WARN: could not validate joint order: {exc}", flush=True)
        return

    rm = get_x2_robot_model()
    pin_names = list(rm.joint_names)
    mj_names = list(X2_BODY_JOINT_NAMES)
    if len(pin_names) != NUM_BODY_DOFS or len(mj_names) != NUM_BODY_DOFS:
        raise RuntimeError(
            f"unexpected joint count: pin={len(pin_names)} mj={len(mj_names)} (want {NUM_BODY_DOFS})"
        )
    for i, mj_idx in enumerate(MJ_TO_PIN):
        if pin_names[i] != mj_names[mj_idx]:
            raise RuntimeError(
                f"MJ_TO_PIN mismatch at i={i}: pin[{i}]={pin_names[i]!r} "
                f"!= mj[{mj_idx}]={mj_names[mj_idx]!r}"
            )
    print(f"[live-VLA] joint-order check OK ({NUM_BODY_DOFS} body DOFs)", flush=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-path", required=True,
        help="Path to the fine-tuned Isaac-GR00T checkpoint directory "
             "(expects model.safetensors + processor/ + experiment_cfg/).",
    )
    parser.add_argument(
        "--prompt", default="play minecraft music on piano",
        help="Language instruction passed to the VLA on every inference.",
    )
    parser.add_argument(
        "--embodiment-tag", default="new_embodiment",
        help="Embodiment tag registered for the X2 modality config.",
    )
    parser.add_argument(
        "--modality-config", default="gear_sonic/data/x2_modality_config_10dof.py",
        help="Side-loadable modality config that registers NEW_EMBODIMENT.",
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="Torch device for the VLA.",
    )
    parser.add_argument("--pub-host", default="*", help="Bind iface for the pose PUB.")
    parser.add_argument("--pub-port", type=int, default=5556, help="Port for the pose PUB (matches deploy --vla-zmq-port).")
    parser.add_argument("--pub-topic", default="pose")
    parser.add_argument("--sub-host", default="localhost", help="Host of the deploy's x2_debug PUB.")
    parser.add_argument("--sub-port", type=int, default=5557, help="Port of the deploy's x2_debug PUB (matches --vla-debug-port).")
    parser.add_argument("--sub-topic", default="x2_debug")
    parser.add_argument("--rate", type=float, default=DEFAULT_PUB_RATE_HZ, help="Publish rate (Hz). 50 matches the deploy control loop.")
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="Total run time in seconds (0 = run until Ctrl-C).",
    )
    parser.add_argument(
        "--inference-min-period-s", type=float, default=0.4,
        help="Lower bound on time between successive inferences (s). The "
             "publisher always advances at --rate; this just throttles the GPU.",
    )
    parser.add_argument(
        "--dump-chunks-dir", type=str, default=None,
        help="If set, dump each Nth full action chunk (token + hand joints + "
             "ego_view + observation snapshot) to this directory as .npz files. "
             "Diagnostic use only -- helps verify whether the model predicts a "
             "full gesture across the whole horizon or just spits out a single "
             "near-constant frame.",
    )
    parser.add_argument(
        "--dump-chunks-every", type=int, default=5,
        help="When --dump-chunks-dir is set, save every Nth chunk (1 = every "
             "chunk; 5 means save chunk_id 0,5,10,...). Keeps disk usage sane "
             "during a long probe run.",
    )
    parser.add_argument(
        "--render-width", type=int, default=640,
        help="Ego-view width (must match the dataset M5 used).",
    )
    parser.add_argument(
        "--render-height", type=int, default=480,
        help="Ego-view height (must match the dataset M5 used).",
    )
    parser.add_argument(
        "--no-omnihand", action="store_true",
        help="Disable the OmniHand mesh fragment (debug only; M5/M6 trained "
             "with OmniHand on).",
    )
    parser.add_argument(
        "--protocol-version", type=int, choices=(3, 4), default=4,
        help="Pose wire protocol version. v4 includes 'count'.",
    )
    parser.add_argument("--quiet", action="store_true", help="Cut down log spam.")
    parser.add_argument(
        "--print-every", type=int, default=50,
        help="Log every Nth pose publish.",
    )
    parser.add_argument(
        "--video-out", type=str, default=None,
        help=(
            "Optional MP4 output path for the ego-view video. When set, a "
            "separate thread mirrors the deploy state and records the live "
            "``ego_view`` (same camera the VLA sees) at --video-fps."
        ),
    )
    parser.add_argument(
        "--video-fps", type=float, default=25.0,
        help="Video recording frame rate. 25 Hz is a smooth wall-clock playback; "
             "set to 50 for 1:1 with the deploy control loop.",
    )
    parser.add_argument(
        "--video-front-out", type=str, default=None,
        help=(
            "Optional MP4 output path for a third-person front view. "
            "Independent of --video-out: each runs in its own thread with "
            "its own EGL context, mirroring the same _LatestState."
        ),
    )
    parser.add_argument(
        "--video-front-camera", type=str, default="third_person_front",
        help="MuJoCo camera spec for the front view "
             "(third_person_front | third_person_side | third_person_above).",
    )
    parser.add_argument(
        "--video-front-width", type=int, default=1280,
        help="Front-view width. Independent of the VLA's --render-width.",
    )
    parser.add_argument(
        "--video-front-height", type=int, default=720,
        help="Front-view height. Independent of the VLA's --render-height.",
    )
    return parser.parse_args(argv)


def _load_modality_config(path_or_module: str) -> None:
    """Side-load the X2 modality config so ``NEW_EMBODIMENT`` resolves.

    Supports either a path to a .py file or a dotted module name.
    """
    if path_or_module.endswith(".py") or "/" in path_or_module:
        spec_path = (REPO_ROOT / path_or_module).resolve() if not Path(path_or_module).is_absolute() else Path(path_or_module).resolve()
        if not spec_path.is_file():
            raise FileNotFoundError(f"--modality-config not found: {spec_path}")
        import importlib.util
        spec = importlib.util.spec_from_file_location("_x2_modality_config_loader", spec_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load module from {spec_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    else:
        import importlib
        importlib.import_module(path_or_module)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    _validate_pin_order_or_die()
    _load_modality_config(args.modality_config)

    # Lazy import — pulls in torch + transformers + Isaac-GR00T, which must
    # only happen after the modality-config side-load.
    print("[live-VLA] loading Gr00tPolicy …", flush=True)
    from gr00t.policy.gr00t_policy import Gr00tPolicy
    policy = Gr00tPolicy(
        embodiment_tag=args.embodiment_tag,
        model_path=args.model_path,
        device=args.device,
    )
    print(f"[live-VLA] policy ready (device={args.device})", flush=True)

    # The MuJoCo renderer is constructed inside the inference thread (see
    # ``_inference_worker``): MuJoCo's EGL backend uses thread-local GL
    # contexts, so a renderer built here would raise EGLError the first
    # time the worker tried to render. We hand workers a zero-arg factory
    # instead. Each video thread / inference thread gets its own EGL
    # context via its own factory invocation.
    from gear_sonic.scripts.render_smoketest_episode_video import MujocoFrameRenderer

    def _make_renderer_factory(
        *, camera: str, width: int, height: int
    ):
        def _factory() -> MujocoFrameRenderer:
            return MujocoFrameRenderer(
                camera=camera,
                width=width,
                height=height,
                with_omnihand=not args.no_omnihand,
                egl=True,
            )
        return _factory

    # Inference-side renderer must match the dataset camera + resolution
    # exactly (the VLA's vision encoder is dimension-locked).
    _renderer_factory = _make_renderer_factory(
        camera="ego_view",
        width=args.render_width,
        height=args.render_height,
    )

    state = _LatestState()
    chunk = _LatestChunk()
    stop_event = threading.Event()

    ctx = zmq.Context.instance()
    pub_sock = ctx.socket(zmq.PUB)
    pub_sock.setsockopt(zmq.SNDHWM, 10)
    pub_sock.setsockopt(zmq.LINGER, 0)
    pub_url = f"tcp://{args.pub_host}:{args.pub_port}"
    pub_sock.bind(pub_url)
    print(f"[live-VLA] pose PUB bound on {pub_url} (topic={args.pub_topic!r})", flush=True)
    sub_url = f"tcp://{args.sub_host}:{args.sub_port}"

    video_threads: list[threading.Thread] = []
    if args.video_out:
        video_threads.append(threading.Thread(
            target=_video_recorder,
            kwargs=dict(
                renderer_factory=_renderer_factory,
                state=state,
                output_path=args.video_out,
                fps=args.video_fps,
                stop_event=stop_event,
                chunk=chunk,
                verbose=not args.quiet,
            ),
            name="vla-video-ego",
            daemon=True,
        ))
    if args.video_front_out:
        front_factory = _make_renderer_factory(
            camera=args.video_front_camera,
            width=args.video_front_width,
            height=args.video_front_height,
        )
        video_threads.append(threading.Thread(
            target=_video_recorder,
            kwargs=dict(
                renderer_factory=front_factory,
                state=state,
                output_path=args.video_front_out,
                fps=args.video_fps,
                stop_event=stop_event,
                chunk=chunk,
                verbose=not args.quiet,
            ),
            name="vla-video-front",
            daemon=True,
        ))

    sub_thread = threading.Thread(
        target=_x2_debug_subscriber,
        kwargs=dict(
            sub_url=sub_url, topic=args.sub_topic,
            state=state, stop_event=stop_event,
        ),
        name="x2_debug-sub",
        daemon=True,
    )
    inf_thread = threading.Thread(
        target=_inference_worker,
        kwargs=dict(
            policy=policy, renderer_factory=_renderer_factory,
            state=state, chunk=chunk,
            prompt=args.prompt, stop_event=stop_event,
            min_period_s=args.inference_min_period_s, verbose=not args.quiet,
            dump_chunks_dir=args.dump_chunks_dir,
            dump_chunks_every=args.dump_chunks_every,
        ),
        name="vla-inference",
        daemon=True,
    )

    def _on_signal(signum: int, _frame: Any) -> None:
        print(f"[live-VLA] caught signal {signum}, shutting down…", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    sub_thread.start()
    inf_thread.start()
    for vt in video_threads:
        vt.start()
    # PUB-SUB late-join: give the deploy a moment to wire up its SUB before
    # we start blasting messages it can't keep up with.
    time.sleep(0.2)

    n_ticks = 0
    try:
        n_ticks = _publisher(
            pub_sock=pub_sock, topic=args.pub_topic, rate_hz=args.rate,
            chunk=chunk, state=state, duration_s=args.duration,
            stop_event=stop_event,
            protocol_version=args.protocol_version,
            print_every=args.print_every,
        )
    finally:
        stop_event.set()
        sub_thread.join(timeout=1.0)
        inf_thread.join(timeout=2.0)
        # Video threads may need an extra beat to flush the encoder queue.
        for vt in video_threads:
            vt.join(timeout=15.0)
        try:
            pub_sock.close(linger=0)
        except Exception:
            pass
        print(
            f"[live-VLA] done after {n_ticks} pub ticks, "
            f"{chunk.inference_count} inferences, "
            f"last_inference_ms={chunk.last_inference_ms:.1f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
