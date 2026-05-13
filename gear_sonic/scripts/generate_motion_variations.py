"""
Pure-numpy variation generator for the M3 autoencoder smoke test.

The smoke test orchestrator (``record_synthetic_smoketest_dataset.py``)
needs ~30 episodes to give a tiny LoRA fine-tune enough diversity to
overfit cleanly without ballooning compute. This module generates those
variations from a single (T, D) base trajectory using four cheap,
deterministic transforms applied independently:

* **time_stretch** -- resample to a different episode length (linear
  interpolation on the temporal axis). Stretch factor s>1 slows the
  motion down (more frames, same range), s<1 speeds it up.
* **gaussian_noise** -- add per-frame i.i.d. Gaussian noise to the
  joint columns. Stationary across time, so the resulting trajectory
  is jittery but does not drift.
* **phase_shift** -- cyclic roll along the time axis. Useful when the
  base trajectory is approximately periodic (the Minecraft piano
  performance has a clear repeating melody phrase) so the variation
  starts/ends at a different point in the loop.
* **lr_mirror** -- swap left- and right-side joint columns, with
  optional sign-flip on roll/yaw axes. Doubles the effective dataset
  size without any runtime work; the ML systems literature calls this
  a "horizontal-flip" augmentation.

The transforms compose: each variation in
``generate_variations(...)`` is a randomly-sampled triple
``(stretch, noise, phase, mirror)`` drawn from the configured ranges,
so the orchestrator can request N variations and get a mostly-balanced
spread.

Determinism
-----------

All randomness flows through a single ``numpy.random.Generator``
created from the user-supplied seed. ``generate_variations(seed=k)``
always yields byte-identical output across runs. This is non-negotiable
for the M3 acceptance gate -- the test pins a specific seed and asserts
on the exact frame counts and per-variation tags, so a flake here
would make the gate untrustworthy.

Out of scope for M3
-------------------

* Joint-limit clamping. Variations that exceed the URDF limits are
  passed through unchanged; the dataset writer is expected to clamp
  before writing if needed (``record_synthetic_smoketest_dataset.py``
  uses ``RobotModel.compensate_joint_limits``).
* Velocity-aware augmentations. We do not consider joint velocities
  in M3 because the SONIC trainer ignores them on the action surface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class VariationParams:
    """Parameters for a single deterministic motion variation.

    Pure data; no randomness here. ``generate_variations`` is the
    entry-point that draws these from the configured ranges.
    """

    stretch: float
    """Temporal scale factor. ``s > 1`` slows down (more frames),
    ``s < 1`` speeds up (fewer frames). ``1.0`` is a no-op."""

    noise_std: float
    """Per-frame i.i.d. Gaussian std applied independently to every
    ``(t, j)`` cell. ``0.0`` is a no-op. Adds high-frequency jitter
    that destroys the trajectory's velocity profile, so the gentle
    preset keeps this at 0 -- prefer :attr:`bias_std` for "the home
    pose has slack" augmentation. Retained for back-compat with
    older recipes / the M3 gate."""

    phase_shift_frames: int
    """Cyclic frame roll. Positive shifts the start later in the loop;
    negative shifts it earlier. ``0`` is a no-op."""

    lr_mirror: bool
    """If ``True``, swap left/right halves of the trajectory. See
    :func:`apply_lr_mirror` for the column convention."""

    bias_std: float = 0.0
    """Per-episode joint-bias Gaussian std (radians) used by callers
    that pass a single trajectory through :func:`apply_variation`. For
    each episode we draw **one** offset vector ``delta ~
    N(0, bias_std^2 I_D)``, clip every component to ``[-bias_clip_sigmas
    * bias_std, +bias_clip_sigmas * bias_std]``, and add the same
    ``delta`` to every frame. The trajectory's velocities,
    accelerations, and curvatures are bit-exact to the base; only the
    absolute home pose drifts. This is the "fingertips end up at
    slightly different XYZ across episodes" augmentation. ``0.0`` is a
    no-op.

    The smoketest orchestrator passes the arm 14-D trajectory through
    :func:`apply_variation` (so it sees ``bias_std`` directly) and the
    hand 20-D trajectory through a separate path that reads
    :attr:`bias_std_hand`. Logging code should display both."""

    bias_std_hand: float = 0.0
    """Per-episode joint-bias Gaussian std (radians) for the hand
    trajectory only. Independent draw from :attr:`bias_std` so the arm
    and hand see uncorrelated home-pose offsets. ``0.0`` is a no-op.
    Read by ``record_synthetic_smoketest_dataset.apply_variation_to_arm_and_hand``;
    not used by :func:`apply_variation`."""

    bias_clip_sigmas: float = 2.0
    """Symmetric clip on every per-episode bias draw, in units of the
    relevant std (:attr:`bias_std` for arms, :attr:`bias_std_hand` for
    hands). ``2.0`` (default) clips the long tail at 2σ so no single
    episode gets a pathologically large home-pose offset."""


def time_stretch(
    trajectory: np.ndarray, factor: float, *, fps: float = 50.0
) -> np.ndarray:
    """Resample ``trajectory`` along the time axis by ``factor``.

    Linear interpolation between the existing samples. Returns a new
    ``(T_new, D)`` array with ``T_new = max(2, round(T * factor))``.

    Edge cases:

    * ``factor == 1.0`` returns a copy of the input (so callers can
      always rely on a fresh array).
    * ``factor <= 0`` raises ``ValueError`` -- a non-positive stretch
      is meaningless and almost certainly a bug at the caller.
    * ``trajectory.shape[0] < 2`` raises ``ValueError`` -- need at
      least two samples to interpolate between.
    """
    if factor <= 0:
        raise ValueError(f"stretch factor must be positive; got {factor}")
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"trajectory must be 2-D (T, D); got shape {arr.shape}")
    T, D = arr.shape
    if T < 2:
        raise ValueError(f"need at least 2 frames to time-stretch; got T={T}")

    if factor == 1.0:
        return arr.copy()

    T_new = max(2, int(round(T * factor)))
    src_idx = np.linspace(0.0, T - 1.0, T_new)
    out = np.empty((T_new, D), dtype=np.float64)
    base = np.arange(T)
    for d in range(D):
        out[:, d] = np.interp(src_idx, base, arr[:, d])
    return out


def gaussian_noise(
    trajectory: np.ndarray,
    std: float,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise (per-element) to ``trajectory``.

    Per-frame, per-dof independent Gaussian draw -- adds high-frequency
    jitter that destroys the trajectory's velocity profile (each tick's
    delta gets ±std random offset). For most augmentation recipes you
    actually want :func:`joint_bias_noise` instead, which draws a single
    constant offset vector per episode and adds it to every frame --
    same gesture shape, slightly different home pose. This function
    stays for back-compat with the M3 acceptance gate.

    A new array is always returned; the input is never mutated. ``std=0``
    short-circuits to a copy.
    """
    arr = np.asarray(trajectory, dtype=np.float64)
    if std == 0.0:
        return arr.copy()
    if std < 0:
        raise ValueError(f"noise std must be >= 0; got {std}")
    return arr + rng.normal(loc=0.0, scale=std, size=arr.shape)


def joint_bias_noise(
    trajectory: np.ndarray,
    std: float,
    *,
    rng: np.random.Generator,
    clip_sigmas: float = 2.0,
) -> np.ndarray:
    """Add a *single* per-episode constant offset to ``trajectory``.

    For a ``(T, D)`` input we draw one ``D``-dim offset vector
    ``delta ~ N(0, std^2 I_D)``, clip every component to
    ``[-clip_sigmas * std, +clip_sigmas * std]``, and add the same
    ``delta`` to every frame::

        out[t, j] = trajectory[t, j] + delta[j]      (for all t)

    Properties:

    * Velocities, accelerations and curvatures are bit-exact to the
      input -- only the absolute joint home pose shifts. SONIC's FSQ
      encoder, which tokenises via per-frame target-pose deltas, sees
      the same motion tokens up to a tiny DC bias.
    * The trajectory remains smooth (no frame-to-frame jitter).
    * Different joints are perturbed independently (no joint-joint
      correlation), so e.g. shoulder_pitch can shift up while elbow
      shifts down -- still anatomically plausible at small ``std``.

    Defaults of ``std=0.010`` rad (arms) and ``std=0.020`` rad (hands)
    with ``clip_sigmas=2.0`` are the "barely noticeable" preset used by
    :func:`record_synthetic_smoketest_dataset.build_smoketest_dataset`.

    Args:
        trajectory: ``(T, D)`` array.
        std: Per-DoF Gaussian std (radians). ``0.0`` is a no-op.
        rng: NumPy ``Generator`` for the offset draw. Determinism flows
            from this object.
        clip_sigmas: Symmetric clip on each component of ``delta``, in
            units of ``std``. ``2.0`` (default) caps the worst-case home
            offset at 2σ. Pass ``np.inf`` to disable clipping.

    Returns:
        ``(T, D)`` float64 array with the per-episode offset applied.
    """
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"trajectory must be 2-D (T, D); got shape {arr.shape}")
    if std == 0.0:
        return arr.copy()
    if std < 0:
        raise ValueError(f"bias std must be >= 0; got {std}")
    if clip_sigmas <= 0:
        raise ValueError(
            f"clip_sigmas must be > 0 (use np.inf to disable); got {clip_sigmas}"
        )
    D = arr.shape[1]
    delta = rng.normal(loc=0.0, scale=std, size=(D,))
    if np.isfinite(clip_sigmas):
        bound = clip_sigmas * std
        np.clip(delta, -bound, bound, out=delta)
    return arr + delta[None, :]


def phase_shift(trajectory: np.ndarray, frames: int) -> np.ndarray:
    """Cyclically roll ``trajectory`` along the time axis by ``frames``.

    Positive ``frames`` makes the new trajectory start ``frames`` ticks
    *later* in the original loop; negative makes it start earlier. Using
    cyclic roll (rather than zero-pad) preserves the trajectory's
    energy and is appropriate for the periodic piano melody.
    """
    arr = np.asarray(trajectory, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"trajectory must be 2-D (T, D); got shape {arr.shape}")
    if frames == 0:
        return arr.copy()
    return np.roll(arr, shift=frames, axis=0)


def apply_lr_mirror(
    trajectory: np.ndarray,
    left_indices: Sequence[int],
    right_indices: Sequence[int],
    flip_signs: Sequence[int] | None = None,
) -> np.ndarray:
    """Swap left- and right-side columns of ``trajectory``.

    Args:
        trajectory: ``(T, D)`` array.
        left_indices: column indices of the left-side joints.
        right_indices: column indices of the right-side joints. Must
            be the same length and order as ``left_indices`` (the
            i-th left joint is paired with the i-th right joint).
        flip_signs: optional sign-flip mask of length
            ``len(left_indices)`` -- 1 to keep the value, -1 to flip.
            Used for axes whose sign convention is mirrored across
            sides (e.g. shoulder roll on most humanoids). ``None``
            means no flip; the caller is responsible for supplying
            the correct mask for the embodiment.

    Returns:
        A new ``(T, D)`` array with the swap applied.

    Raises:
        ValueError: on mismatched left/right index lengths or if
            ``flip_signs`` has the wrong length.
    """
    arr = np.asarray(trajectory, dtype=np.float64)
    left_indices = list(left_indices)
    right_indices = list(right_indices)
    if len(left_indices) != len(right_indices):
        raise ValueError(
            f"left ({len(left_indices)}) and right ({len(right_indices)}) "
            "indices must have the same length for L/R mirror"
        )
    if flip_signs is not None:
        flip_signs = list(flip_signs)
        if len(flip_signs) != len(left_indices):
            raise ValueError(
                f"flip_signs length {len(flip_signs)} != "
                f"index length {len(left_indices)}"
            )
        for s in flip_signs:
            if s not in (-1, 1):
                raise ValueError(f"flip_signs entries must be ±1; got {s}")
    out = arr.copy()
    if flip_signs is None:
        flip_arr = np.ones(len(left_indices), dtype=np.float64)
    else:
        flip_arr = np.asarray(flip_signs, dtype=np.float64)
    out[:, left_indices] = arr[:, right_indices] * flip_arr
    out[:, right_indices] = arr[:, left_indices] * flip_arr
    return out


def apply_variation(
    trajectory: np.ndarray,
    params: VariationParams,
    *,
    rng: np.random.Generator,
    left_indices: Sequence[int] | None = None,
    right_indices: Sequence[int] | None = None,
    flip_signs: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply one :class:`VariationParams` instance to ``trajectory``.

    The transforms compose in the order **stretch -> mirror -> phase
    shift -> bias -> noise**. This ordering is intentional:

    1. Stretch first to set the final episode length.
    2. Mirror is shape-preserving and commutes with phase shift, so
       its position in the chain is interchangeable; it goes second
       only because the L/R swap changes which columns the noise hits
       (irrelevant for i.i.d. noise but kept stable for determinism).
    3. Phase shift is the last temporal transform so the bias / noise
       stay aligned with the (already shifted) trajectory.
    4. Joint-bias next: a single per-episode constant offset across all
       frames. Smooth, preserves velocity profile.
    5. Per-frame Gaussian noise last so it never gets resampled or
       rolled. Adds high-frequency jitter (intended off in gentle
       presets); kept here for back-compat with the M3 gate.
    """
    out = time_stretch(trajectory, params.stretch)
    if params.lr_mirror:
        if left_indices is None or right_indices is None:
            raise ValueError(
                "apply_variation: lr_mirror=True requires left_indices and right_indices"
            )
        out = apply_lr_mirror(out, left_indices, right_indices, flip_signs)
    out = phase_shift(out, params.phase_shift_frames)
    if params.bias_std > 0.0:
        out = joint_bias_noise(
            out, params.bias_std,
            rng=rng, clip_sigmas=params.bias_clip_sigmas,
        )
    out = gaussian_noise(out, params.noise_std, rng=rng)
    return out


def generate_variations(
    base_trajectory: np.ndarray,
    *,
    num_variations: int,
    seed: int = 0,
    stretch_range: tuple[float, float] = (0.85, 1.15),
    noise_std_range: tuple[float, float] = (0.0, 0.02),
    phase_shift_frac_range: tuple[float, float] = (-0.25, 0.25),
    lr_mirror_prob: float = 0.5,
    bias_std_range: tuple[float, float] = (0.0, 0.0),
    bias_std_hand_range: tuple[float, float] = (0.0, 0.0),
    bias_clip_sigmas: float = 2.0,
    include_identity: bool = True,
) -> list[tuple[VariationParams, np.ndarray]]:
    """Generate ``num_variations`` independently-sampled variations.

    The first variation, when ``include_identity=True``, is always the
    no-op (identity) variation. This is deliberate: the M3 acceptance
    gate uses the identity variation to assert that the round-trip
    through the LeRobot exporter is byte-clean before introducing any
    augmentation noise.

    Args:
        base_trajectory: ``(T, D)`` array.
        num_variations: total variations to emit (including identity
            if ``include_identity=True``).
        seed: NumPy ``Generator`` seed. Determines all sampling.
        stretch_range: ``(min, max)`` uniform draw for time-stretch.
        noise_std_range: ``(min, max)`` uniform draw for noise std.
        phase_shift_frac_range: ``(min, max)`` uniform draw, expressed
            as a fraction of the (post-stretch) episode length.
        lr_mirror_prob: probability of mirroring per variation.
        include_identity: if True, the first emitted variation is the
            unmodified base trajectory.

    Returns:
        A list of ``(params, trajectory)`` pairs, length
        ``num_variations``. Trajectories are float64.

    Note:
        L/R mirror requires the orchestrator to plumb ``left_indices`` /
        ``right_indices`` through :func:`apply_variation` -- this
        function only samples the *flag*. We can't sample the indices
        because they are embodiment-specific and live in the
        ``RobotModel``. The orchestrator builds the index map and
        re-applies via :func:`apply_variation` per emitted variation.
    """
    if num_variations < 1:
        raise ValueError(f"num_variations must be >= 1; got {num_variations}")
    if not (0.0 <= lr_mirror_prob <= 1.0):
        raise ValueError(f"lr_mirror_prob must be in [0, 1]; got {lr_mirror_prob}")
    if stretch_range[0] <= 0 or stretch_range[1] <= 0 or stretch_range[0] > stretch_range[1]:
        raise ValueError(f"stretch_range must be positive and ordered; got {stretch_range}")
    if noise_std_range[0] < 0 or noise_std_range[1] < noise_std_range[0]:
        raise ValueError(f"noise_std_range must be non-negative and ordered; got {noise_std_range}")
    if bias_std_range[0] < 0 or bias_std_range[1] < bias_std_range[0]:
        raise ValueError(f"bias_std_range must be non-negative and ordered; got {bias_std_range}")
    if bias_std_hand_range[0] < 0 or bias_std_hand_range[1] < bias_std_hand_range[0]:
        raise ValueError(
            f"bias_std_hand_range must be non-negative and ordered; got {bias_std_hand_range}"
        )
    if bias_clip_sigmas <= 0:
        raise ValueError(
            f"bias_clip_sigmas must be > 0 (use np.inf to disable); got {bias_clip_sigmas}"
        )

    rng = np.random.default_rng(seed)
    base = np.asarray(base_trajectory, dtype=np.float64)
    out: list[tuple[VariationParams, np.ndarray]] = []

    for i in range(num_variations):
        if include_identity and i == 0:
            params = VariationParams(
                stretch=1.0,
                noise_std=0.0,
                phase_shift_frames=0,
                lr_mirror=False,
                bias_std=0.0,
                bias_std_hand=0.0,
                bias_clip_sigmas=bias_clip_sigmas,
            )
            out.append((params, base.copy()))
            continue

        stretch = float(rng.uniform(*stretch_range))
        noise_std = float(rng.uniform(*noise_std_range))
        bias_std = float(rng.uniform(*bias_std_range))
        bias_std_hand = float(rng.uniform(*bias_std_hand_range))
        # Phase shift is sampled in *fractional episode* units, then
        # converted to integer frames against the post-stretch length
        # so the same fractional spec yields perceptually-similar
        # offsets across stretch factors.
        T_post_stretch = max(2, int(round(base.shape[0] * stretch)))
        frac = float(rng.uniform(*phase_shift_frac_range))
        phase_frames = int(round(frac * T_post_stretch))
        lr_mirror = bool(rng.uniform() < lr_mirror_prob)

        params = VariationParams(
            stretch=stretch,
            noise_std=noise_std,
            phase_shift_frames=phase_frames,
            lr_mirror=lr_mirror,
            bias_std=bias_std,
            bias_std_hand=bias_std_hand,
            bias_clip_sigmas=bias_clip_sigmas,
        )
        # NB: ``apply_variation`` would re-do the stretch from scratch.
        # Cache the stretched base to avoid that. ``include_identity``
        # path above already returns ``base`` directly.
        stretched = time_stretch(base, params.stretch)
        traj = stretched
        # Mirror is applied by the orchestrator that knows the
        # embodiment's index map; we mark the params and pass through.
        traj = phase_shift(traj, params.phase_shift_frames)
        if params.bias_std > 0.0:
            traj = joint_bias_noise(
                traj, params.bias_std,
                rng=rng, clip_sigmas=params.bias_clip_sigmas,
            )
        traj = gaussian_noise(traj, params.noise_std, rng=rng)
        out.append((params, traj))

    return out


def variation_summary(params: VariationParams) -> str:
    """Single-line human-readable summary of a variation. Used in logs."""
    pieces = [f"stretch={params.stretch:.3f}"]
    if params.bias_std > 0 or params.bias_std_hand > 0:
        pieces.append(
            f"bias=arm:{params.bias_std:.4f} hand:{params.bias_std_hand:.4f}"
        )
    if params.noise_std > 0:
        pieces.append(f"noise={params.noise_std:.4f}")
    if params.phase_shift_frames != 0:
        pieces.append(f"phase={params.phase_shift_frames:+d}")
    if params.lr_mirror:
        pieces.append("mirror")
    return ", ".join(pieces)


def iter_variations(
    base_trajectory: np.ndarray,
    **kwargs,
) -> Iterable[tuple[VariationParams, np.ndarray]]:
    """Convenience iterator wrapper around :func:`generate_variations`."""
    yield from generate_variations(base_trajectory, **kwargs)


__all__ = [
    "VariationParams",
    "apply_lr_mirror",
    "apply_variation",
    "gaussian_noise",
    "generate_variations",
    "iter_variations",
    "joint_bias_noise",
    "phase_shift",
    "time_stretch",
    "variation_summary",
]
