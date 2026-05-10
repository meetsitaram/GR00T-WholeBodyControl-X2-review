"""
M3 autoencoder smoke-test orchestrator.

Builds a small LeRobot v2.1 dataset from a single base motion (the
Minecraft theme song piano performance, sourced from the production
``agitbot-x2-record-and-replay`` repo) by:

1. Loading the recorded 14-DOF arm trajectory + 20-DOF hand-target
   stream + their native control rate (50 Hz).
2. Composing the 14-DOF arm trajectory onto a 31-DOF X2 standing pose
   so the legs/waist/head stay anchored in their trained-default
   configuration -- the SONIC tracking decoder needs a full 31-DOF
   pose stream and would diverge if we left those slots at zero.
3. Splitting the 20-DOF hand vector into the L/R 10-DOF hand-joint
   action streams expected by the X2 modality config.
4. Running ``generate_motion_variations.generate_variations`` to emit
   N independently-sampled variations of the base trajectory (time
   stretch, gaussian noise, phase shift, optional L/R mirror).
5. Writing each variation as one LeRobot v2.1 episode using the same
   ``Gr00tDataExporter`` + ``features_x2_vla`` schema as the M1
   sample episode -- so the smoke-test dataset and any future Quest3
   teleop dataset are loader-byte-compatible.
6. Running ``gr00t.data.stats.generate_stats`` to populate
   ``meta/stats.json`` so Isaac-GR00T's loader is satisfied.

This is the canonical "autoencoder smoke test" pattern from ML-systems
engineering: feed a known signal into the pipeline and check the output
reproduces it. See
[``docs/source/tutorials/vla_training.md``](../../docs/source/tutorials/vla_training.md)
section 2 for the methodology.

The orchestrator supports two camera sources for
``observation.images.ego_view``:

* ``--camera-source gradient`` (**default**): a deterministic
  gradient frame, same as the M1 sample episode. The orchestrator
  stays simulator-free and the M3 acceptance gate runs on hosts
  without OpenGL / MuJoCo rendering.
* ``--camera-source mujoco`` (M5 camera plumbing): per-frame native
  MuJoCo render through
  :class:`gear_sonic.scripts.render_smoketest_episode_video.MujocoFrameRenderer`,
  using the OmniHand-augmented MJCF from M3.5 (head ``ego_view``
  camera at 640×480, finger DOFs animated from the recorded hand
  trajectory). The renderer is built once per dataset and reused
  across all episodes so the EGL context is paid for once.

The dataset *schema* is unchanged regardless of camera source --
``meta/info.json`` features, modality config, episode/task layout,
and ``observation.images.ego_view`` shape/dtype all match between the
two. Only the pixel content of the video frames differs. This is
what lets the M5 mujoco-backed dataset be a drop-in replacement for
the M3 gradient-backed dataset in the Isaac-GR00T ``LeRobotEpisodeLoader``.

For an inspection video of what the deploy-time ego camera *would*
see when the recorded body trajectory is replayed (the URDF
``rgbd_head_front`` mount, optical axis derived from the panel STL),
``gear_sonic/scripts/render_smoketest_episode_video.py`` remains the
canonical "render one recording to MP4" entry-point.

Source asset
------------

The orchestrator looks for the Minecraft recording at
``$AGITBOT_RECORD_REPLAY_REPO/song_paths/minecraft_theme_song/path_omni.npz``
where ``$AGITBOT_RECORD_REPLAY_REPO`` defaults to
``../agitbot-x2-record-and-replay`` (sibling of GR00T-WholeBodyControl).
If the asset is not present, the orchestrator falls back to a
deterministic synthetic trajectory (multi-frequency sinusoids over the
arm DOFs + slow finger sweep). This keeps the M3 acceptance gate
runnable on a machine without the sibling repo (e.g. CI).

Usage
-----

Build a tiny gate-sized dataset (4 episodes, suitable for ``pytest``)::

    timeout 120 .venv/bin/python \\
        gear_sonic/scripts/record_synthetic_smoketest_dataset.py \\
        --output-dir /tmp/x2_smoketest_v0 \\
        --num-episodes 4 --max-frames 200 --seed 42

Build a full LoRA-fine-tunable dataset (~30 episodes)::

    timeout 600 .venv/bin/python \\
        gear_sonic/scripts/record_synthetic_smoketest_dataset.py \\
        --output-dir /path/to/x2_smoketest_lora --num-episodes 30 --seed 0

Build a MuJoCo-backed dataset (M5 camera plumbing)::

    timeout 900 .venv/bin/python \\
        gear_sonic/scripts/record_synthetic_smoketest_dataset.py \\
        --output-dir /path/to/x2_smoketest_lora_mujoco --num-episodes 30 --seed 0 \\
        --camera-source mujoco

Acceptance:

* The dataset directory must round-trip through Isaac-GR00T's
  ``LeRobotEpisodeLoader`` (gate: ``tests/test_x2_smoketest_pipeline.py``).
* Per-episode ``recorded.npz`` files (written next to the LeRobot
  dataset) are usable as the *reference* trajectory in
  ``compare_motion_trajectories.py`` after a closed-loop rollout.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Protocol

import numpy as np

from gear_sonic.data.exporter import Gr00tDataExporter
from gear_sonic.data.features_x2_vla import (
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
    FPS,
    HAND_DOF_OMNI,
    SONIC_MOTION_TOKEN_DIM,
    assemble_observation_state,
    get_features_x2_vla,
    get_modality_config_x2_vla,
    get_x2_robot_model,
)
from gear_sonic.scripts.generate_motion_variations import (
    VariationParams,
    apply_lr_mirror,
    generate_variations,
    variation_summary,
)


# Camera source identifiers exposed via the CLI / public API. The
# default ("gradient") keeps the orchestrator simulator-free so the M3
# acceptance gate runs without MuJoCo. "mujoco" opts into the M5
# camera-plumbing path documented at the top of this module.
CAMERA_SOURCE_GRADIENT: str = "gradient"
CAMERA_SOURCE_MUJOCO: str = "mujoco"
CAMERA_SOURCES: tuple[str, ...] = (CAMERA_SOURCE_GRADIENT, CAMERA_SOURCE_MUJOCO)


# Motion-token label sources (M6 / Stage C1). The default ("zeros") matches
# the M3 placeholder so existing acceptance gates keep passing without a
# SONIC checkpoint on disk; "sonic_g1" runs each recorded body trajectory
# through the SONIC ``g1`` encoder + FSQ to produce deploy-aligned 64-D
# motion tokens. See gear_sonic/scripts/sonic_motion_token_labeler.py.
MOTION_TOKEN_SOURCE_ZEROS: str = "zeros"
MOTION_TOKEN_SOURCE_SONIC_G1: str = "sonic_g1"
MOTION_TOKEN_SOURCES: tuple[str, ...] = (
    MOTION_TOKEN_SOURCE_ZEROS,
    MOTION_TOKEN_SOURCE_SONIC_G1,
)

# Default SONIC checkpoint used when --motion-token-source=sonic_g1. The
# h200 sphere-feet 25k-step checkpoint is the latest production-quality
# encoder available locally; callers can override via --sonic-checkpoint.
DEFAULT_SONIC_CHECKPOINT: Path = Path(
    "/home/stickbot/x2_cloud_checkpoints/"
    "h200-iter-25000-sphere-feet-20260501/model_step_025000.pt"
)


# ---------------------------------------------------------------------------
# X2 31-DOF body layout constants (must match
# gear_sonic/data/robot_model/supplemental_info/x2_ultra/...).
# Keeping them in this module rather than re-deriving from RobotModel
# keeps the orchestrator import-light and lets the M3 acceptance gate
# pin the indices in a single place.
# ---------------------------------------------------------------------------

X2_BODY_DOF: int = 31

LEFT_LEG_INDICES: tuple[int, ...] = tuple(range(0, 6))
RIGHT_LEG_INDICES: tuple[int, ...] = tuple(range(6, 12))
WAIST_INDICES: tuple[int, ...] = tuple(range(12, 15))
LEFT_ARM_INDICES: tuple[int, ...] = tuple(range(15, 22))   # 7 DOFs
RIGHT_ARM_INDICES: tuple[int, ...] = tuple(range(22, 29))  # 7 DOFs
HEAD_INDICES: tuple[int, ...] = (29, 30)

# Indices in the recorded "trajectory" array
# (left arm 0..6, right arm 7..13). Source:
# /home/stickbot/Projects/agitbot-x2-record-and-replay/src/x2_recorder/constants.py
RECORDED_LEFT_ARM_SLICE = slice(0, 7)
RECORDED_RIGHT_ARM_SLICE = slice(7, 14)

# X2 trained stand pose (MuJoCo joint order, radians). Mirrors
# ``policy_parameters.hpp::default_angles``. Hard-coded here rather
# than re-parsed from C++ to keep the orchestrator dependency-light.
DEFAULT_STAND_POSE_MJ_RAD: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # left leg
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,         # right leg
    0.0, 0.0, 0.0,                                # waist
    0.2, 0.2, 0.0, -0.6, 0.0, 0.0, 0.0,           # left arm
    0.2, -0.2, 0.0, -0.6, 0.0, 0.0, 0.0,          # right arm
    0.0, 0.0,                                     # head
)
assert len(DEFAULT_STAND_POSE_MJ_RAD) == X2_BODY_DOF


DEFAULT_TASK = "play minecraft music on piano"
DEFAULT_NUM_EPISODES = 4
DEFAULT_MAX_FRAMES = 200  # tiny by default, M3 gate scope


# ---------------------------------------------------------------------------
# Augmentation presets
# ---------------------------------------------------------------------------
#
# Each preset packages a coherent set of augmentation knobs for
# ``generate_variations``. The CLI exposes ``--preset <name>``; explicit
# ``--stretch-range`` / ``--bias-std-arm`` / etc. flags override the
# preset on a per-knob basis. Adding a new preset means adding an entry
# here and (optionally) bumping ``--preset`` choices.
#
#   * ``m3-legacy`` -- the original M3 gate defaults. Heavy time stretch,
#     phase shift, 50/50 L/R mirror, per-frame i.i.d. noise. Useful when
#     building a generic LoRA dataset (1k+ episodes, multi-task) or for
#     reproducing the legacy gate.
#
#   * ``gentle`` -- single-gesture over-fit recipe. No time stretch, no
#     phase shift, no L/R mirror, NO per-frame jitter. Per-episode joint
#     bias offsets only: σ_arm = 0.010 rad, σ_hand = 0.020 rad, clipped
#     at ±2σ. The trajectory shape stays bit-exact; only the absolute
#     joint home pose drifts a few mrad per episode. The "play one
#     gesture flawlessly" preset.
PRESETS: dict[str, dict[str, object]] = {
    "m3-legacy": {
        "stretch_range": (0.85, 1.15),
        "noise_std_range": (0.0, 0.02),
        "phase_shift_frac_range": (-0.25, 0.25),
        "lr_mirror_prob": 0.5,
        "bias_std_range": (0.0, 0.0),
        "bias_std_hand_range": (0.0, 0.0),
        "bias_clip_sigmas": 2.0,
    },
    "gentle": {
        "stretch_range": (1.0, 1.0),
        "noise_std_range": (0.0, 0.0),
        "phase_shift_frac_range": (0.0, 0.0),
        "lr_mirror_prob": 0.0,
        # σ_arm = 10 mrad ≈ 0.57°; below servo dead-band, ~6 mm
        # fingertip drift at 1σ on a 0.6 m arm. Per-DoF independent
        # draws at this magnitude already compound across the 7-DoF
        # arm chain to ~12 mm at 2σ -- safe for over-fit, not enough
        # to change which piano key the fingertip lands on.
        "bias_std_range": (0.010, 0.010),
        # σ_hand = 20 mrad ≈ 1.15°; hands have larger ROM and aren't
        # part of SONIC's body-token loop, so they tolerate more.
        "bias_std_hand_range": (0.020, 0.020),
        "bias_clip_sigmas": 2.0,
    },
}
DEFAULT_PRESET = "m3-legacy"


# ---------------------------------------------------------------------------
# Source asset loading
# ---------------------------------------------------------------------------


@dataclass
class SourceMotion:
    """Holds the base recording the orchestrator turns into episodes."""

    arm_trajectory: np.ndarray
    """Arm joint trajectory, ``(T, 14)``: 7 left + 7 right DOFs."""

    hand_trajectory: np.ndarray
    """Hand target trajectory, ``(T, 20)``: 10 left + 10 right DOFs."""

    fps: float
    """Native control rate of the recording (Hz)."""

    source_label: str
    """Where the trajectory came from. Stored in the dataset's
    ``script_config`` for provenance."""


def _default_record_replay_root() -> Path:
    """Return the canonical sibling-repo path for ``agitbot-x2-record-and-replay``."""
    env = os.environ.get("AGITBOT_RECORD_REPLAY_REPO")
    if env:
        return Path(env).expanduser()
    return Path(__file__).resolve().parents[2].parent / "agitbot-x2-record-and-replay"


def _make_synthetic_arm_trajectory(num_frames: int) -> np.ndarray:
    """Deterministic 14-DOF arm trajectory used when the recording is unavailable.

    Produces a multi-frequency sinusoid in each arm DOF so the columns
    are non-degenerate for the LeRobot stats validator. Amplitudes are
    bounded well within the URDF limits (~0.4 rad) to keep the
    trajectory plausible without depending on per-joint URDF reads.
    """
    t = np.linspace(0.0, 2.0 * np.pi, num_frames, dtype=np.float64)
    out = np.zeros((num_frames, 14), dtype=np.float64)
    for d in range(14):
        freq = 1.0 + 0.25 * d
        phase = 0.31 * d
        amp = 0.30 + 0.05 * (d % 4)
        out[:, d] = amp * np.sin(freq * t + phase)
    return out


def _make_synthetic_hand_trajectory(num_frames: int) -> np.ndarray:
    """Deterministic 20-DOF hand trajectory matching the OmniHand limits."""
    t = np.linspace(0.0, 2.0 * np.pi, num_frames, dtype=np.float64)
    out = np.zeros((num_frames, 20), dtype=np.float64)
    for d in range(20):
        freq = 0.5 + 0.1 * d
        amp = 0.20 + 0.02 * (d % 5)
        out[:, d] = amp * (0.5 + 0.5 * np.sin(freq * t))
    return out


def load_source_motion(
    record_replay_root: Path | None = None,
    *,
    fallback_synthetic_frames: int = 600,
) -> SourceMotion:
    """Load the Minecraft recording, falling back to a synthetic trace.

    Args:
        record_replay_root: path to the ``agitbot-x2-record-and-replay``
            checkout. Defaults to the sibling-repo convention.
        fallback_synthetic_frames: frames to synthesise when the
            recording is missing. ~12 s at 50 Hz keeps the trajectory
            interesting without being huge.

    Returns:
        :class:`SourceMotion`.
    """
    if record_replay_root is None:
        record_replay_root = _default_record_replay_root()
    asset = (
        record_replay_root
        / "song_paths"
        / "minecraft_theme_song"
        / "path_omni.npz"
    )
    if asset.exists():
        with np.load(asset, allow_pickle=True) as data:
            arm = np.asarray(data["trajectory"], dtype=np.float64)
            hand = np.asarray(data["hand_targets"], dtype=np.float64)
            dt = float(np.asarray(data["dt"]))
        if arm.shape[1] != 14:
            raise ValueError(
                f"Recorded arm trajectory has {arm.shape[1]} DOFs; expected 14. "
                f"Asset: {asset}"
            )
        if hand.shape[1] != 20:
            raise ValueError(
                f"Recorded hand trajectory has {hand.shape[1]} DOFs; expected 20. "
                f"Asset: {asset}"
            )
        if arm.shape[0] != hand.shape[0]:
            T = min(arm.shape[0], hand.shape[0])
            arm = arm[:T]
            hand = hand[:T]
        fps = 1.0 / dt if dt > 0 else float(FPS)
        return SourceMotion(
            arm_trajectory=arm,
            hand_trajectory=hand,
            fps=fps,
            source_label=str(asset),
        )

    # Fallback: synthetic trace. Logged via source_label so a dataset
    # built without the sibling repo never silently looks "real".
    arm = _make_synthetic_arm_trajectory(fallback_synthetic_frames)
    hand = _make_synthetic_hand_trajectory(fallback_synthetic_frames)
    return SourceMotion(
        arm_trajectory=arm,
        hand_trajectory=hand,
        fps=float(FPS),
        source_label=f"synthetic_fallback(frames={fallback_synthetic_frames})",
    )


# ---------------------------------------------------------------------------
# Trajectory composition
# ---------------------------------------------------------------------------


def _stand_pose() -> np.ndarray:
    """Return the 31-DOF X2 trained stand pose."""
    return np.asarray(DEFAULT_STAND_POSE_MJ_RAD, dtype=np.float64)


def compose_body_trajectory(
    arm_trajectory: np.ndarray,
    *,
    stand_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Overlay a 14-DOF arm trajectory onto the X2 stand pose.

    Returns a ``(T, 31)`` array where legs/waist/head columns hold the
    constant stand pose and arm columns track the recording. The arm
    indices are pinned to ``LEFT_ARM_INDICES`` / ``RIGHT_ARM_INDICES``
    so the resulting layout matches MuJoCo joint order.
    """
    arm = np.asarray(arm_trajectory, dtype=np.float64)
    if arm.ndim != 2 or arm.shape[1] != 14:
        raise ValueError(f"arm_trajectory must be (T, 14); got shape {arm.shape}")
    T = arm.shape[0]
    if stand_pose is None:
        stand_pose = _stand_pose()
    if stand_pose.shape != (X2_BODY_DOF,):
        raise ValueError(
            f"stand_pose must be ({X2_BODY_DOF},); got {stand_pose.shape}"
        )

    body = np.broadcast_to(stand_pose, (T, X2_BODY_DOF)).copy()
    body[:, list(LEFT_ARM_INDICES)] = arm[:, RECORDED_LEFT_ARM_SLICE]
    body[:, list(RIGHT_ARM_INDICES)] = arm[:, RECORDED_RIGHT_ARM_SLICE]
    return body


def split_hand_trajectory(
    hand_trajectory: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a ``(T, 20)`` hand-target stream into L/R 10-DOF halves."""
    arr = np.asarray(hand_trajectory, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 20:
        raise ValueError(f"hand_trajectory must be (T, 20); got shape {arr.shape}")
    return arr[:, :10].copy(), arr[:, 10:].copy()


# ---------------------------------------------------------------------------
# Variation -> episode plumbing
# ---------------------------------------------------------------------------


def _arm_lr_indices_with_signs() -> tuple[
    tuple[int, ...], tuple[int, ...], tuple[int, ...]
]:
    """L/R arm column indices + sign-flip mask for the joined 14-DOF stream.

    Returns indices into a 14-DOF arm trajectory (left=0..6, right=7..13).
    The sign-flip mask flips roll/yaw axes when mirroring; pitch axes
    keep their sign. Matches the X2 URDF's mirror convention (verified
    against ``X2_BODY_JOINT_LIMITS`` in the supplemental info).
    """
    # Order of the 7 arm DOFs (matches ARM_JOINTS in agitbot recorder):
    # shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
    # wrist_yaw, wrist_pitch, wrist_roll
    left = (0, 1, 2, 3, 4, 5, 6)
    right = (7, 8, 9, 10, 11, 12, 13)
    # Pitch (idx 0=shoulder_pitch, 5=wrist_pitch, 3=elbow) and pip-style
    # bend axes keep their sign; roll (1, 6) and yaw (2, 4) flip across L/R.
    sign_mask = (1, -1, -1, 1, -1, 1, -1)
    return left, right, sign_mask


def _hand_lr_indices() -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Index pair to mirror a 20-DOF hand stream (left=0..9, right=10..19)."""
    return tuple(range(10)), tuple(range(10, 20))


def apply_variation_to_arm_and_hand(
    arm: np.ndarray,
    hand: np.ndarray,
    params: VariationParams,
    *,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one variation jointly to an arm trajectory and the matched hand stream.

    Hands and arms must stay temporally aligned, so we time-stretch and
    phase-shift them with the same parameters.

    Augmentation order per stream (matches :func:`apply_variation` for
    parity):

    1. Time stretch (shared parameter -- arm and hand interpolate to
       the same length).
    2. Optional L/R mirror (hands use a separate index map than arms).
    3. Phase shift (shared frame count).
    4. Joint-bias noise: one per-episode constant offset added to every
       frame. Drawn independently for arms (``params.bias_std``) and
       hands (``params.bias_std_hand``); both clipped at
       ``params.bias_clip_sigmas`` σ. Preserves velocity / acceleration
       profile; only the absolute home pose drifts. This is the gentle-
       preset workhorse.
    5. Per-frame i.i.d. Gaussian noise (``params.noise_std``). Drawn
       independently for each stream. Off by default in the gentle
       preset; on for back-compat with the M3 gate.
    """
    from gear_sonic.scripts.generate_motion_variations import (
        gaussian_noise,
        joint_bias_noise,
        phase_shift,
        time_stretch,
    )

    arm_out = time_stretch(arm, params.stretch)
    hand_out = time_stretch(hand, params.stretch)

    if params.lr_mirror:
        left, right, sign_mask = _arm_lr_indices_with_signs()
        arm_out = apply_lr_mirror(arm_out, left, right, sign_mask)
        hand_left, hand_right = _hand_lr_indices()
        hand_out = apply_lr_mirror(hand_out, hand_left, hand_right)

    arm_out = phase_shift(arm_out, params.phase_shift_frames)
    hand_out = phase_shift(hand_out, params.phase_shift_frames)

    if params.bias_std > 0.0:
        arm_out = joint_bias_noise(
            arm_out, params.bias_std,
            rng=rng, clip_sigmas=params.bias_clip_sigmas,
        )
    if params.bias_std_hand > 0.0:
        hand_out = joint_bias_noise(
            hand_out, params.bias_std_hand,
            rng=rng, clip_sigmas=params.bias_clip_sigmas,
        )

    arm_out = gaussian_noise(arm_out, params.noise_std, rng=rng)
    hand_out = gaussian_noise(hand_out, params.noise_std, rng=rng)
    return arm_out, hand_out


def _make_synthetic_ego_view(frame_idx: int, num_frames: int) -> np.ndarray:
    """Same deterministic gradient as the M1 sample episode."""
    frame = np.zeros((EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3), dtype=np.uint8)
    red = int(255 * frame_idx / max(num_frames - 1, 1))
    blue = int(255 * (1 - frame_idx / max(num_frames - 1, 1)))
    frame[..., 0] = red
    frame[..., 2] = blue
    yy, xx = np.mgrid[: EGO_VIEW_HEIGHT, : EGO_VIEW_WIDTH]
    frame[..., 1] = ((xx + yy + frame_idx * 7) % 256).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Frame providers (M5 camera plumbing)
# ---------------------------------------------------------------------------
#
# A "frame provider" is the indirection that lets us swap the gradient
# placeholder for a real MuJoCo render without touching the orchestrator
# control flow. Both providers expose the same shape/dtype contract so
# the LeRobot dataset schema is identical regardless of which one is in
# use; only the pixel content of ``observation.images.ego_view`` differs.


class FrameProvider(Protocol):
    """Stateless-from-the-orchestrator's-POV per-frame ego-view provider.

    Implementations are responsible for any internal state (e.g. an EGL
    render context). The orchestrator owns lifecycle: it builds the
    provider once at the start of a dataset run and ``close()``s it at
    the end, calling ``frame()`` once per dataset frame in between.
    """

    def frame(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        body_q: np.ndarray,
        left_active: np.ndarray,
        right_active: np.ndarray,
    ) -> np.ndarray: ...

    def close(self) -> None: ...


class _GradientFrameProvider:
    """M3 gradient-placeholder frames (no MuJoCo, no GPU)."""

    name: str = CAMERA_SOURCE_GRADIENT

    def frame(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        body_q: np.ndarray,
        left_active: np.ndarray,
        right_active: np.ndarray,
    ) -> np.ndarray:
        del body_q, left_active, right_active  # gradient ignores motion
        return _make_synthetic_ego_view(frame_idx, num_frames)

    def close(self) -> None:
        return None


class _MujocoFrameProvider:
    """M5 MuJoCo-native ego-view frames using the OmniHand-augmented MJCF.

    Builds one :class:`MujocoFrameRenderer` and reuses it across every
    frame of every episode in the dataset run, so the EGL context and
    model compile cost are paid once. The renderer always uses the
    ``ego_view`` head camera at the dataset's ``EGO_VIEW_{WIDTH,HEIGHT}``
    so the returned tensor matches the LeRobot feature shape exactly.
    """

    name: str = CAMERA_SOURCE_MUJOCO

    def __init__(self, *, with_omnihand: bool = True, egl: bool = True) -> None:
        # Late import: the M3 acceptance gate must keep working on hosts
        # without MuJoCo / OpenGL. Importing the renderer here means
        # ``from gear_sonic.scripts.record_synthetic_smoketest_dataset
        # import build_smoketest_dataset`` is safe on those hosts as
        # long as the caller leaves ``camera_source="gradient"``.
        from gear_sonic.scripts.render_smoketest_episode_video import (
            MujocoFrameRenderer,
        )

        self._renderer = MujocoFrameRenderer(
            camera="ego_view",
            width=EGO_VIEW_WIDTH,
            height=EGO_VIEW_HEIGHT,
            with_omnihand=with_omnihand,
            egl=egl,
        )

    def frame(
        self,
        *,
        frame_idx: int,
        num_frames: int,
        body_q: np.ndarray,
        left_active: np.ndarray,
        right_active: np.ndarray,
    ) -> np.ndarray:
        del frame_idx, num_frames  # MuJoCo provider doesn't need them
        return self._renderer.render_frame(
            body_q,
            left_active=left_active,
            right_active=right_active,
        )

    def close(self) -> None:
        self._renderer.close()


def make_frame_provider(camera_source: str) -> FrameProvider:
    """Build the frame provider implementing ``camera_source``.

    Raises:
        ValueError: if ``camera_source`` isn't one of
            :data:`CAMERA_SOURCES`.
    """
    if camera_source == CAMERA_SOURCE_GRADIENT:
        return _GradientFrameProvider()
    if camera_source == CAMERA_SOURCE_MUJOCO:
        return _MujocoFrameProvider()
    raise ValueError(
        f"Unknown camera_source {camera_source!r}; expected one of {CAMERA_SOURCES}."
    )


def _build_one_episode(
    exporter: Gr00tDataExporter,
    robot_model,
    body_trajectory: np.ndarray,
    left_hand_trajectory: np.ndarray,
    right_hand_trajectory: np.ndarray,
    task: str,
    *,
    frame_provider: FrameProvider,
    motion_tokens: np.ndarray | None = None,
) -> int:
    """Stream one composed trajectory into the exporter; return frame count.

    Args:
        motion_tokens: optional ``(T, SONIC_MOTION_TOKEN_DIM)`` per-frame
            motion-token labels. When None, every frame is labeled with
            zeros (matches the M3 placeholder for backward compatibility).
            When provided, the array's ``T`` must match the trajectory
            length; this is the M6 / Stage C1 signal that lets the VLA
            actually learn deploy-aligned motion tokens.
    """
    T = body_trajectory.shape[0]
    if (
        left_hand_trajectory.shape[0] != T
        or right_hand_trajectory.shape[0] != T
    ):
        raise ValueError(
            "body / hand trajectories must share T; got "
            f"body={T}, left={left_hand_trajectory.shape[0]}, "
            f"right={right_hand_trajectory.shape[0]}"
        )

    projected_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    if motion_tokens is None:
        # Backward-compatible zero placeholder. Used by the M3 acceptance
        # gate and any caller that hasn't opted into the SONIC labeler yet.
        zero_token = np.zeros(SONIC_MOTION_TOKEN_DIM, dtype=np.float64)
        token_provider = (lambda _f: zero_token)
    else:
        if motion_tokens.shape != (T, SONIC_MOTION_TOKEN_DIM):
            raise ValueError(
                f"motion_tokens must be (T={T}, "
                f"{SONIC_MOTION_TOKEN_DIM}); got shape "
                f"{motion_tokens.shape}."
            )
        if motion_tokens.dtype != np.float64:
            # Cast once up-front so per-frame access is a clean view.
            motion_tokens = motion_tokens.astype(np.float64, copy=False)
        token_provider = (lambda f: motion_tokens[f])

    for f in range(T):
        body_q = body_trajectory[f]
        left_q = left_hand_trajectory[f]
        right_q = right_hand_trajectory[f]

        observation_state = assemble_observation_state(
            robot_model, body_q, left_q, right_q
        )
        ego_view = frame_provider.frame(
            frame_idx=f,
            num_frames=T,
            body_q=body_q,
            left_active=left_q,
            right_active=right_q,
        )
        if ego_view.dtype != np.uint8 or ego_view.shape != (
            EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3,
        ):
            raise RuntimeError(
                f"frame_provider {type(frame_provider).__name__} returned "
                f"shape={ego_view.shape} dtype={ego_view.dtype}; expected "
                f"({EGO_VIEW_HEIGHT}, {EGO_VIEW_WIDTH}, 3) uint8."
            )

        # body_q is in Pinocchio order; the dataset's action.body_q_mj
        # column expects MuJoCo order. The synthetic smoke episode
        # keeps the body in the default stand pose so we zero-fill at
        # the canonical 31 DOF -- downstream tests only care that the
        # column has the right shape and appears in every frame.
        # v1 schema: synthetic data has no SONIC rollout so executed ==
        # pre_sonic == zeros, sonic_correction_max_rad == 0.
        commanded_body_q_mj = np.zeros(robot_model.num_joints, dtype=np.float64)
        left_q_arr = left_q.copy()
        right_q_arr = right_q.copy()

        frame_data = {
            "observation.state": observation_state,
            "observation.projected_gravity": projected_gravity,
            "action.motion_token": token_provider(f),
            "action.body_q_mj": commanded_body_q_mj,
            "action.left_hand_joints": left_q_arr,
            "action.right_hand_joints": right_q_arr,
            "action.body_q_mj_pre_sonic": commanded_body_q_mj.copy(),
            "action.left_hand_joints_pre_sonic": left_q_arr.copy(),
            "action.right_hand_joints_pre_sonic": right_q_arr.copy(),
            "action.sonic_correction_max_rad": np.zeros(1, dtype=np.float32),
            "observation.images.ego_view": ego_view,
            "task": task,
        }
        exporter.add_frame(frame_data)

    exporter.save_episode()
    return T


# ---------------------------------------------------------------------------
# Public entry-points
# ---------------------------------------------------------------------------


@dataclass
class SmoketestRunSummary:
    """Returned by :func:`build_smoketest_dataset` for downstream gates."""

    output_dir: Path
    """Where the LeRobot dataset was written."""

    num_episodes: int
    """Episodes actually written."""

    per_episode_frames: list[int]
    """Frame count of every episode (post-stretch + truncation)."""

    variation_params: list[VariationParams]
    """Variation params for every episode (parallel to ``per_episode_frames``)."""

    source_label: str
    """Pass-through of :class:`SourceMotion.source_label`."""

    base_recordings_path: Path
    """Where the per-variation .npz reference trajectories were written."""

    camera_source: str = CAMERA_SOURCE_GRADIENT
    """Which provider populated ``observation.images.ego_view``. One of
    :data:`CAMERA_SOURCES`. The dataset schema is identical regardless;
    only the pixel content of the video frames differs."""

    motion_token_source: str = MOTION_TOKEN_SOURCE_ZEROS
    """Which source produced ``action.motion_token``. One of
    :data:`MOTION_TOKEN_SOURCES`.
    :data:`MOTION_TOKEN_SOURCE_ZEROS` (default) reproduces the M3
    placeholder; :data:`MOTION_TOKEN_SOURCE_SONIC_G1` runs each recorded
    body trajectory through the SONIC ``g1`` encoder + FSQ to produce
    deploy-aligned 64-D tokens (M6 / Stage C1)."""

    sonic_checkpoint_path: Optional[Path] = None
    """Path to the SONIC .pt checkpoint used when
    ``motion_token_source == MOTION_TOKEN_SOURCE_SONIC_G1``. ``None``
    when the dataset was built with zero motion tokens."""


def build_smoketest_dataset(
    output_dir: Path,
    *,
    num_episodes: int = DEFAULT_NUM_EPISODES,
    max_frames: int | None = DEFAULT_MAX_FRAMES,
    seed: int = 0,
    task: str = DEFAULT_TASK,
    record_replay_root: Path | None = None,
    overwrite: bool = True,
    skip_stats: bool = False,
    camera_source: str = CAMERA_SOURCE_GRADIENT,
    frame_provider: FrameProvider | None = None,
    motion_token_source: str = MOTION_TOKEN_SOURCE_ZEROS,
    sonic_checkpoint_path: Path | None = None,
    motion_token_labeler: "SonicMotionTokenLabeler | None" = None,
    motion_token_device: str = "cpu",
    stretch_range: tuple[float, float] = (0.85, 1.15),
    noise_std_range: tuple[float, float] = (0.0, 0.02),
    phase_shift_frac_range: tuple[float, float] = (-0.25, 0.25),
    lr_mirror_prob: float = 0.5,
    bias_std_range: tuple[float, float] = (0.0, 0.0),
    bias_std_hand_range: tuple[float, float] = (0.0, 0.0),
    bias_clip_sigmas: float = 2.0,
) -> SmoketestRunSummary:
    """Materialise the M3 smoke-test dataset on disk.

    Args:
        output_dir: where to write the LeRobot v2.1 dataset (the
            exporter manages the directory; if ``overwrite`` is True
            and it exists, it gets wiped first).
        num_episodes: variations to draw + write. Default 4 (gate-sized).
        max_frames: if set, cap each variation at this many frames after
            stretching. Default 200 keeps the gate fast.
        seed: deterministic variation seed.
        task: language prompt stored in ``meta/tasks.jsonl``.
        record_replay_root: optional override for the sibling repo
            location; falls back to ``$AGITBOT_RECORD_REPLAY_REPO`` or
            the canonical sibling path.
        overwrite: wipe ``output_dir`` before writing if it exists.
        skip_stats: if True, skip the
            ``gr00t.data.stats.generate_stats(output_dir)`` call. This
            is exposed only for tests that already invoke the stats
            generator on their own; production callers should keep the
            default of False.
        camera_source: which :class:`FrameProvider` builds the
            ``observation.images.ego_view`` tensor. One of
            :data:`CAMERA_SOURCES`. Default
            :data:`CAMERA_SOURCE_GRADIENT` keeps the orchestrator
            simulator-free (the M3 contract);
            :data:`CAMERA_SOURCE_MUJOCO` opts into the M5 camera
            plumbing (per-frame native MuJoCo render with the
            OmniHand-augmented MJCF). Ignored when ``frame_provider``
            is supplied.
        frame_provider: optional pre-built provider. When supplied it
            takes precedence over ``camera_source`` and is **not**
            closed by this function -- ownership stays with the caller
            (typical use: pytest fixtures that share one provider
            across multiple ``build_smoketest_dataset`` calls). When
            None, the function builds and tears down the provider
            implied by ``camera_source``.
        motion_token_source: which source builds the
            ``action.motion_token`` field. One of
            :data:`MOTION_TOKEN_SOURCES`.
            :data:`MOTION_TOKEN_SOURCE_ZEROS` (default) reproduces the
            M3 placeholder so existing acceptance gates keep passing
            without a SONIC checkpoint on disk.
            :data:`MOTION_TOKEN_SOURCE_SONIC_G1` runs each recorded
            body trajectory through the SONIC ``g1`` encoder + FSQ to
            produce deploy-aligned 64-D tokens (M6 / Stage C1).
        sonic_checkpoint_path: optional override for the SONIC .pt
            checkpoint used when
            ``motion_token_source == MOTION_TOKEN_SOURCE_SONIC_G1``.
            Defaults to :data:`DEFAULT_SONIC_CHECKPOINT`. Ignored when
            ``motion_token_labeler`` is supplied or the source is
            :data:`MOTION_TOKEN_SOURCE_ZEROS`.
        motion_token_labeler: optional pre-built
            :class:`SonicMotionTokenLabeler`. When supplied it takes
            precedence over ``sonic_checkpoint_path`` (typical use:
            pytest fixtures that share one labeler across multiple
            ``build_smoketest_dataset`` calls). The caller owns the
            labeler's lifecycle.
        motion_token_device: ``torch.device`` string for the labeler.
            Defaults to ``"cpu"``; pass ``"cuda"`` for very large
            datasets where CPU encoding becomes the bottleneck.

    Returns:
        :class:`SmoketestRunSummary`.
    """
    if camera_source not in CAMERA_SOURCES:
        raise ValueError(
            f"camera_source must be one of {CAMERA_SOURCES}; got {camera_source!r}."
        )
    if motion_token_source not in MOTION_TOKEN_SOURCES:
        raise ValueError(
            f"motion_token_source must be one of {MOTION_TOKEN_SOURCES}; "
            f"got {motion_token_source!r}."
        )
    output_dir = Path(output_dir).resolve()
    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    source = load_source_motion(record_replay_root)
    base_arm = source.arm_trajectory
    base_hand = source.hand_trajectory

    # Use the arm trajectory as the variation driver. Hands ride along
    # via ``apply_variation_to_arm_and_hand`` so they stay temporally
    # aligned per variation.
    variations = generate_variations(
        base_arm,
        num_variations=num_episodes,
        seed=seed,
        stretch_range=stretch_range,
        noise_std_range=noise_std_range,
        phase_shift_frac_range=phase_shift_frac_range,
        lr_mirror_prob=lr_mirror_prob,
        bias_std_range=bias_std_range,
        bias_std_hand_range=bias_std_hand_range,
        bias_clip_sigmas=bias_clip_sigmas,
    )

    rng = np.random.default_rng(seed + 17)  # disjoint stream for the noise applied below

    robot_model = get_x2_robot_model("omnihand_10")
    features = get_features_x2_vla(robot_model)
    modality_config = get_modality_config_x2_vla(robot_model)

    base_recordings_dir = output_dir.parent / f"{output_dir.name}__recorded"
    if overwrite and base_recordings_dir.exists():
        shutil.rmtree(base_recordings_dir)
    base_recordings_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the SONIC labeler upfront when caller opts into the
    # encoder-backed labels. We tolerate both pre-built labelers (caller-
    # owned, e.g. pytest fixtures sharing one across N dataset builds)
    # and the lazy "build from a checkpoint path" mode.
    resolved_sonic_checkpoint: Path | None = None
    used_labeler: "SonicMotionTokenLabeler | None" = None
    if motion_token_source == MOTION_TOKEN_SOURCE_SONIC_G1:
        if motion_token_labeler is not None:
            used_labeler = motion_token_labeler
            resolved_sonic_checkpoint = None  # caller-owned, path unknown
        else:
            from gear_sonic.scripts.sonic_motion_token_labeler import (
                SonicMotionTokenLabeler,
            )

            ckpt_path = (
                Path(sonic_checkpoint_path)
                if sonic_checkpoint_path is not None
                else DEFAULT_SONIC_CHECKPOINT
            )
            if not ckpt_path.exists():
                raise FileNotFoundError(
                    "SONIC checkpoint not found at "
                    f"{ckpt_path}. Pass --sonic-checkpoint to override or "
                    "fall back to --motion-token-source zeros."
                )
            used_labeler = SonicMotionTokenLabeler(
                ckpt_path,
                device=motion_token_device,
                motion_fps=float(FPS),
            )
            resolved_sonic_checkpoint = ckpt_path
    sonic_ckpt_str = (
        str(resolved_sonic_checkpoint)
        if resolved_sonic_checkpoint is not None
        else None
    )

    exporter = Gr00tDataExporter.create(
        save_root=output_dir,
        fps=FPS,
        features=features,
        modality_config=modality_config,
        task=task,
        script_config={
            "robot_type": "agibot_x2_ultra",
            "embodiment_tag": "new_embodiment",
            "hand_variant": "omnihand_10",
            "num_body_joints": robot_model.num_joints,
            "hand_dof_per_side": HAND_DOF_OMNI,
            "fps": FPS,
            "smoketest": True,
            "source_label": source.source_label,
            "source_fps": source.fps,
            "seed": seed,
            "variations_planned": num_episodes,
            "camera_source": camera_source,
            "motion_token_source": motion_token_source,
            "sonic_checkpoint_path": sonic_ckpt_str,
        },
        robot_type="agibot_x2_ultra",
    )

    # Frame provider lifecycle: when the caller passed one in, we don't
    # own it (no close() on the way out). When we built it ourselves we
    # release it in a finally block so the EGL context (if any) is freed
    # even if a downstream call (e.g. generate_stats) raises.
    caller_owns_provider = frame_provider is not None
    provider: FrameProvider = frame_provider or make_frame_provider(camera_source)
    print(
        f"[record_synthetic_smoketest_dataset] camera_source={camera_source} "
        f"provider={type(provider).__name__} "
        f"motion_token_source={motion_token_source} "
        f"sonic_checkpoint={sonic_ckpt_str}"
    )

    per_episode_frames: list[int] = []
    variation_params: list[VariationParams] = []
    try:
        for ep_idx, (params, _arm_traj_unused) in enumerate(variations):
            # Re-derive arm + hand jointly from base arrays per variation
            # so the hand stream stays aligned with the arm trajectory --
            # ``generate_variations`` only knows about the arm tensor.
            arm_var, hand_var = apply_variation_to_arm_and_hand(
                base_arm, base_hand, params, rng=rng
            )
            if max_frames is not None:
                arm_var = arm_var[:max_frames]
                hand_var = hand_var[:max_frames]

            body_traj = compose_body_trajectory(arm_var)
            left_hand, right_hand = split_hand_trajectory(hand_var)

            # Persist the recorded trajectory (post-variation) next to the
            # LeRobot dataset so compare_motion_trajectories.py can pick it
            # up as the reference stream after a closed-loop rollout.
            np.savez_compressed(
                base_recordings_dir / f"episode_{ep_idx:04d}_recorded.npz",
                body_trajectory=body_traj.astype(np.float32),
                left_hand_trajectory=left_hand.astype(np.float32),
                right_hand_trajectory=right_hand.astype(np.float32),
                arm_trajectory=arm_var.astype(np.float32),
                params_stretch=np.float64(params.stretch),
                params_noise_std=np.float64(params.noise_std),
                params_phase_shift_frames=np.int64(params.phase_shift_frames),
                params_lr_mirror=np.bool_(params.lr_mirror),
                params_bias_std=np.float64(params.bias_std),
                params_bias_std_hand=np.float64(params.bias_std_hand),
                params_bias_clip_sigmas=np.float64(params.bias_clip_sigmas),
                # ``trajectory`` is what compare_motion_trajectories.py
                # expects by default. Mirrors the body trajectory so a
                # default-CLI run of the comparator works without flag
                # tweaking.
                trajectory=body_traj.astype(np.float32),
            )

            episode_motion_tokens: np.ndarray | None = None
            if used_labeler is not None:
                episode_motion_tokens = used_labeler.label_trajectory(
                    body_traj
                )

            T = _build_one_episode(
                exporter,
                robot_model,
                body_traj,
                left_hand,
                right_hand,
                task,
                frame_provider=provider,
                motion_tokens=episode_motion_tokens,
            )
            per_episode_frames.append(T)
            variation_params.append(params)
            print(
                f"  episode {ep_idx:04d}: T={T:4d} | "
                f"{variation_summary(params)}",
                flush=True,
            )
    finally:
        if not caller_owns_provider:
            provider.close()

    if not skip_stats:
        try:
            from gr00t.data.stats import generate_stats
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError(
                "Isaac-GR00T (gr00t.data.stats) is required to finalise the "
                "X2 smoke-test dataset. Install via "
                "`uv pip install -e external_dependencies/Isaac-GR00T --no-deps "
                "--python .venv/bin/python`."
            ) from exc
        generate_stats(output_dir)

    print(
        f"[record_synthetic_smoketest_dataset] wrote {num_episodes} episode(s) "
        f"to {output_dir}",
        flush=True,
    )
    print(
        f"[record_synthetic_smoketest_dataset] reference recordings -> "
        f"{base_recordings_dir}",
        flush=True,
    )

    return SmoketestRunSummary(
        output_dir=output_dir,
        num_episodes=num_episodes,
        per_episode_frames=per_episode_frames,
        variation_params=variation_params,
        source_label=source.source_label,
        base_recordings_path=base_recordings_dir,
        camera_source=camera_source,
        motion_token_source=motion_token_source,
        sonic_checkpoint_path=resolved_sonic_checkpoint,
    )


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="LeRobot dataset output dir. Default: outputs/x2_smoketest_<timestamp>.",
    )
    parser.add_argument(
        "--num-episodes", type=int, default=DEFAULT_NUM_EPISODES,
        help=f"Variations to emit. Default {DEFAULT_NUM_EPISODES}.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
        help=(
            f"Cap each episode at this many frames after time-stretch "
            f"(default {DEFAULT_MAX_FRAMES}). Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for variation parameter draws (default 0).",
    )
    parser.add_argument(
        "--task", type=str, default=DEFAULT_TASK,
        help="Language prompt stored in meta/tasks.jsonl.",
    )
    parser.add_argument(
        "--record-replay-root", type=Path, default=None,
        help=(
            "Override for the agitbot-x2-record-and-replay sibling-repo path. "
            "Defaults to $AGITBOT_RECORD_REPLAY_REPO or the canonical sibling location."
        ),
    )
    parser.add_argument(
        "--no-overwrite", action="store_true",
        help="Do not wipe output-dir before writing.",
    )
    parser.add_argument(
        "--camera-source",
        choices=CAMERA_SOURCES,
        default=CAMERA_SOURCE_GRADIENT,
        help=(
            "Pixel source for observation.images.ego_view. "
            f"{CAMERA_SOURCE_GRADIENT!r} (default) keeps the orchestrator "
            "simulator-free with a deterministic gradient frame, matching "
            "the M3 acceptance contract. "
            f"{CAMERA_SOURCE_MUJOCO!r} renders each frame natively through "
            "the OmniHand-augmented MJCF (M5 camera plumbing); requires "
            "MuJoCo + an offscreen GL context."
        ),
    )
    parser.add_argument(
        "--motion-token-source",
        choices=MOTION_TOKEN_SOURCES,
        default=MOTION_TOKEN_SOURCE_ZEROS,
        help=(
            "Source of action.motion_token labels. "
            f"{MOTION_TOKEN_SOURCE_ZEROS!r} (default) writes the M3 "
            "placeholder all-zero token for backward compatibility. "
            f"{MOTION_TOKEN_SOURCE_SONIC_G1!r} runs each recorded body "
            "trajectory through the SONIC g1 encoder + FSQ to produce "
            "deploy-aligned 64-D tokens (M6 / Stage C1)."
        ),
    )
    parser.add_argument(
        "--sonic-checkpoint",
        type=Path,
        default=None,
        help=(
            "SONIC .pt checkpoint used when "
            "--motion-token-source sonic_g1. Defaults to the local h200 "
            "iter-25k sphere-feet checkpoint at "
            f"{DEFAULT_SONIC_CHECKPOINT}."
        ),
    )
    parser.add_argument(
        "--motion-token-device",
        type=str,
        default="cpu",
        help=(
            "torch device for the SONIC labeler. CPU is plenty fast (sub-"
            "millisecond per frame) for the gate dataset; pass 'cuda' if "
            "you're labeling a much larger corpus."
        ),
    )
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(PRESETS.keys()),
        default=DEFAULT_PRESET,
        help=(
            "Augmentation preset (per-episode knob defaults). "
            "'m3-legacy' (default) is the original M3 gate recipe -- "
            "heavy time-stretch / phase / mirror / per-frame noise, "
            "intended for diverse multi-task LoRA datasets. 'gentle' "
            "is the single-gesture over-fit recipe: no temporal "
            "transforms, no L/R mirror, joint-bias-only at σ_arm=0.010 "
            "rad, σ_hand=0.020 rad clipped at ±2σ. Each --*-range / "
            "--*-prob flag below overrides the preset on a per-knob "
            "basis."
        ),
    )
    # Augmentation overrides. Each is None by default and falls back to
    # the resolved preset; pass a value to override that knob only.
    parser.add_argument(
        "--stretch-range",
        type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Override preset's stretch_range. Pass '1.0 1.0' to disable.",
    )
    parser.add_argument(
        "--noise-std-range",
        type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Override preset's per-frame Gaussian noise std range (rad). "
             "Pass '0.0 0.0' to disable jitter (recommended for over-fit).",
    )
    parser.add_argument(
        "--phase-shift-frac-range",
        type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Override preset's phase-shift fraction range. Pass '0.0 0.0' to disable.",
    )
    parser.add_argument(
        "--lr-mirror-prob",
        type=float, default=None,
        help="Override preset's L/R mirror probability. 0.0 disables mirroring.",
    )
    parser.add_argument(
        "--bias-std-arm-range",
        type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Per-episode joint-bias std range for the 14-DoF arm (rad). "
             "Each episode draws ONE constant offset added to every frame. "
             "Pass '0.010 0.010' for the gentle-preset value, '0.0 0.0' to disable.",
    )
    parser.add_argument(
        "--bias-std-hand-range",
        type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
        help="Per-episode joint-bias std range for the 20-DoF hand (rad). "
             "Independent draw from --bias-std-arm-range. "
             "Pass '0.020 0.020' for the gentle-preset value, '0.0 0.0' to disable.",
    )
    parser.add_argument(
        "--bias-clip-sigmas",
        type=float, default=None,
        help="Symmetric clip on every bias draw, in units of σ. Default 2.0 "
             "(caps worst-case home-pose offset at 2σ). Pass a large value "
             "(e.g. 100) to effectively disable clipping.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _resolve_aug(args: argparse.Namespace) -> dict[str, object]:
    """Merge --preset defaults with explicit --*-range / --*-prob overrides.

    Explicit CLI args win; ``None`` means "fall back to the preset".
    Returns a dict of kwargs ready to splat into
    :func:`build_smoketest_dataset` (and ultimately
    :func:`generate_variations`).
    """
    resolved = dict(PRESETS[args.preset])
    if args.stretch_range is not None:
        resolved["stretch_range"] = tuple(args.stretch_range)
    if args.noise_std_range is not None:
        resolved["noise_std_range"] = tuple(args.noise_std_range)
    if args.phase_shift_frac_range is not None:
        resolved["phase_shift_frac_range"] = tuple(args.phase_shift_frac_range)
    if args.lr_mirror_prob is not None:
        resolved["lr_mirror_prob"] = float(args.lr_mirror_prob)
    if args.bias_std_arm_range is not None:
        resolved["bias_std_range"] = tuple(args.bias_std_arm_range)
    if args.bias_std_hand_range is not None:
        resolved["bias_std_hand_range"] = tuple(args.bias_std_hand_range)
    if args.bias_clip_sigmas is not None:
        resolved["bias_clip_sigmas"] = float(args.bias_clip_sigmas)
    return resolved


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"x2_smoketest_{ts}"

    max_frames: int | None = args.max_frames if args.max_frames > 0 else None
    aug = _resolve_aug(args)
    build_smoketest_dataset(
        output_dir=output_dir,
        num_episodes=args.num_episodes,
        max_frames=max_frames,
        seed=args.seed,
        task=args.task,
        record_replay_root=args.record_replay_root,
        overwrite=not args.no_overwrite,
        camera_source=args.camera_source,
        motion_token_source=args.motion_token_source,
        sonic_checkpoint_path=args.sonic_checkpoint,
        motion_token_device=args.motion_token_device,
        **aug,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CAMERA_SOURCES",
    "CAMERA_SOURCE_GRADIENT",
    "CAMERA_SOURCE_MUJOCO",
    "DEFAULT_SONIC_CHECKPOINT",
    "DEFAULT_STAND_POSE_MJ_RAD",
    "FrameProvider",
    "HEAD_INDICES",
    "LEFT_ARM_INDICES",
    "LEFT_LEG_INDICES",
    "MOTION_TOKEN_SOURCES",
    "MOTION_TOKEN_SOURCE_SONIC_G1",
    "MOTION_TOKEN_SOURCE_ZEROS",
    "RIGHT_ARM_INDICES",
    "RIGHT_LEG_INDICES",
    "SmoketestRunSummary",
    "SourceMotion",
    "WAIST_INDICES",
    "X2_BODY_DOF",
    "apply_variation_to_arm_and_hand",
    "build_smoketest_dataset",
    "compose_body_trajectory",
    "load_source_motion",
    "make_frame_provider",
    "split_hand_trajectory",
]
