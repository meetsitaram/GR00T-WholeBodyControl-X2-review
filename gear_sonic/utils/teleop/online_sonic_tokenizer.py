"""Online wrapper around :class:`SonicMotionTokenLabeler` for closed-loop teleop.

The base labeler in :mod:`gear_sonic.scripts.sonic_motion_token_labeler`
expects a *full* recorded trajectory and emits one 64-D motion token
per frame. The dataset recorder runs at 50 Hz with no future
visibility -- we have only the *current* operator-driven body pose,
not the next 10 frames.

For closed-loop teleop with the SONIC tracking decoder, the
operationally correct approach is "freeze-pose tokenization":

1. Build a virtual 11-frame motion clip where every frame equals the
   current ``body_q`` (with zero velocity, identity root rotation).
2. Run the labeler at frame ``f=0``. ``build_tokenizer_obs`` then
   sees ten future frames that all match ``body_q`` and a 6D
   rotation-diff that is identity for every step, producing the
   token whose decoded action keeps the robot *at* ``body_q``.

The tracking policy then drives towards that pose, and the next
50 Hz tick re-tokenizes against the operator's updated wrist
positions, closing the loop. This works because the SONIC tracking
encoder was trained on continuous motion clips: feeding it a
"stay here" target at every tick is the exact pattern used by the
``mock_vla_publish_stand_token`` baseline (which the deploy is
already happy to consume).

Usage::

    tokenizer = OnlineSonicTokenizer.from_checkpoint(
        "/path/to/model_step_025000.pt"
    )
    tok = tokenizer.encode(body_q)   # (64,) float64

The first call performs the (possibly slow) checkpoint load. Each
subsequent ``encode`` is a single 11-frame batched encoder pass --
sub-millisecond on CPU, sub-100 microseconds on GPU.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional

import numpy as np

# Importing the labeler implicitly pulls in eval_x2_mujoco via its
# own sys.path manipulation. We do the same here defensively so the
# recorder script can use this module without adding extra paths.
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from gear_sonic.scripts.sonic_motion_token_labeler import (  # noqa: E402
    DEFAULT_MOTION_FPS,
    IDENTITY_QUAT_XYZW,
    SONIC_MOTION_TOKEN_DIM,
    SonicMotionTokenLabeler,
)


# A short virtual clip: the labeler only reads frames 0..NUM_FUTURE_FRAMES-1
# (= 10 future frames at 0.1s spacing), so 11 frames is enough headroom
# without being wasteful. ``build_tokenizer_obs`` clamps the fetch index
# to ``total_frames - 1`` so even 1 frame would technically work, but
# the velocity computation reads ``f - 1`` and we want it to land at
# index 0 with zero velocity rather than borrowing a non-existent
# previous frame.
_VIRTUAL_CLIP_LEN: int = 11


class OnlineSonicTokenizer:
    """Per-frame ``body_q`` -> 64-D motion token encoder.

    Wraps :class:`SonicMotionTokenLabeler` so the recorder can call
    ``encode(body_q)`` once per 50 Hz tick.

    Args:
        labeler: Pre-loaded :class:`SonicMotionTokenLabeler`.
            Use :meth:`from_checkpoint` if you'd rather the wrapper
            construct it for you.
        clip_len: Length of the virtual frozen-pose clip used to
            satisfy the labeler's 10-frame future window. Defaults
            to 11; values <11 risk index-clamp boundary effects.
    """

    def __init__(
        self,
        labeler: SonicMotionTokenLabeler,
        *,
        clip_len: int = _VIRTUAL_CLIP_LEN,
    ) -> None:
        self._labeler = labeler
        self._clip_len = int(clip_len)
        if self._clip_len < 2:
            raise ValueError("clip_len must be >= 2")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        device: str = "cpu",
        motion_fps: float = DEFAULT_MOTION_FPS,
        clip_len: int = _VIRTUAL_CLIP_LEN,
    ) -> "OnlineSonicTokenizer":
        labeler = SonicMotionTokenLabeler(
            checkpoint_path,
            device=device,
            motion_fps=motion_fps,
        )
        return cls(labeler, clip_len=clip_len)

    @property
    def labeler(self) -> SonicMotionTokenLabeler:
        return self._labeler

    @property
    def device(self) -> str:
        return self._labeler.device

    @property
    def motion_fps(self) -> float:
        return self._labeler.motion_fps

    def encode(
        self,
        body_q: np.ndarray,
        *,
        root_rot_xyzw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Encode one body pose into a 64-D SONIC motion token.

        Args:
            body_q: ``(31,)`` body joint vector in MuJoCo joint order
                (legs / waist / arms / head). Same layout the live
                VLA bridge publishes on the ``pose`` topic.
            root_rot_xyzw: optional ``(4,)`` xyzw quaternion for the
                root. Defaults to identity (the gantry profile keeps
                the pelvis upright). Pass measured ``base_quat`` if
                you want the token to reflect a non-trivial pelvis
                orientation.

        Returns:
            ``(SONIC_MOTION_TOKEN_DIM,)`` float64 token vector that
            lies on the FSQ lattice.
        """
        body_q = np.asarray(body_q, dtype=np.float64).reshape(-1)
        if body_q.shape[0] != 31:
            raise ValueError(
                f"body_q must be (31,) MuJoCo joint order; got {body_q.shape}"
            )

        if root_rot_xyzw is None:
            root_xyzw = np.asarray(IDENTITY_QUAT_XYZW, dtype=np.float64)
        else:
            root_xyzw = np.asarray(root_rot_xyzw, dtype=np.float64).reshape(-1)
            if root_xyzw.shape[0] != 4:
                raise ValueError(
                    f"root_rot_xyzw must be (4,) xyzw; got {root_xyzw.shape}"
                )

        # Tile body_q + root quat across the virtual clip. This makes
        # ``build_tokenizer_obs`` see "stay at this pose" for every
        # future frame.
        body_clip = np.tile(body_q, (self._clip_len, 1))
        root_clip = np.tile(root_xyzw, (self._clip_len, 1))

        tokens = self._labeler.label_trajectory(
            body_clip, root_rot_xyzw=root_clip
        )
        # ``label_trajectory`` emits (T, 64); we just need frame 0.
        return tokens[0]

    def encode_with_horizon(
        self,
        body_q: np.ndarray,
        *,
        horizon: int,
        root_rot_xyzw: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Convenience: emit ``(horizon, 64)`` by tiling :meth:`encode`.

        Useful when downstream consumers (e.g. the deploy's pose-chunk
        protocol) want a horizon of identical motion tokens.
        """
        if horizon <= 0:
            raise ValueError(f"horizon must be > 0, got {horizon}")
        token = self.encode(body_q, root_rot_xyzw=root_rot_xyzw)
        return np.tile(token, (horizon, 1)).astype(np.float64, copy=False)


__all__ = [
    "OnlineSonicTokenizer",
    "SONIC_MOTION_TOKEN_DIM",
]
