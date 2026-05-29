"""PyTorch dataset: X2 motion-lib PKL(s) -> MotionBricks global motion features.

Default behaviour matches SONIC's BONES-SEED training: no name filter, all
2,584 clips in ``x2_ultra_bones_seed.pkl`` are used. Pass an explicit set of
include/exclude regex patterns (or use the helpers in
``motionbricks.data.x2_loco_filters``) to drop down to walk/turn-only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import joblib
import numpy as np
import torch
from torch.utils.data import Dataset

from motionbricks.data.x2_loco_filters import filter_motion_keys
from motionbricks.data.x2_pkl_to_motion import (
  X2MujocoFkExtractor,
  assert_x2_mjcf_coherent,
  default_x2_mjcf_path,
)
from motionbricks.helper.pl_util import load_motion_rep


PathLike = Union[str, Path]


class X2MotionDataset(Dataset):
  """Load X2 motion-lib clip(s) from one or more PKLs and emit normalized features.

  Each ``__getitem__`` returns ``{"keyid": str, "motion": Tensor[T, D]}`` where
  ``D = 418`` for ``X2Skeleton34``.

  Parameters
  ----------
  pkl_paths
      One PKL path or a sequence of PKL paths. Multiple PKLs are concatenated
      with their per-key namespaces preserved (e.g. ``loco__*`` from one and
      ``stable-loco__*`` from another can coexist).
  motion_rep
      Instantiated MotionBricks motion-rep (carries skeleton + stats).
  include_patterns / exclude_patterns
      Regex filters applied to motion keys. Both default to ``None`` =
      keep every clip (matches SONIC). Pass
      ``motionbricks.data.x2_loco_filters.DEFAULT_INCLUDE_PATTERNS`` /
      ``DEFAULT_EXCLUDE_PATTERNS`` to opt into walk/turn filtering.
  """

  def __init__(
    self,
    pkl_paths: Union[PathLike, Sequence[PathLike]],
    motion_rep,
    *,
    mjcf_path: Optional[Path] = None,
    cache_dir: Optional[Path] = None,
    min_frames: int = 80,
    max_frames: int = 300,
    subsample: int = 1,
    include_patterns: Optional[Sequence[str]] = None,
    exclude_patterns: Optional[Sequence[str]] = None,
    max_clips: Optional[int] = None,
    recompute_cache: bool = False,
    normalize: bool = True,
    verify_mjcf: bool = True,
  ):
    self.motion_rep = motion_rep
    self._normalize = normalize
    self.min_frames = min_frames
    self.max_frames = max_frames
    self.subsample = subsample

    if isinstance(pkl_paths, (str, Path)):
      pkl_list: List[Path] = [Path(pkl_paths)]
    else:
      pkl_list = [Path(p) for p in pkl_paths]
    if not pkl_list:
      raise ValueError("X2MotionDataset requires at least one PKL path")

    mjcf = Path(mjcf_path) if mjcf_path is not None else default_x2_mjcf_path()
    if verify_mjcf:
      assert_x2_mjcf_coherent(mjcf)

    self._keys: List[str] = []
    self._features: List[torch.Tensor] = []

    cache_dir = Path(cache_dir) if cache_dir else None

    # Loud "what am I loading" summary — paranoia after a smoke-PKL was
    # accidentally used for full training in a prior run. We print BOTH the
    # PKL identity AND the cache_dir path so any mismatch is human-visible
    # before training step 1.
    print("=== X2MotionDataset ===")
    for p in pkl_list:
      try:
        size_gb = p.stat().st_size / (1024**3)
        print(f"  PKL:        {p}  ({size_gb:.2f} GB)")
      except OSError:
        print(f"  PKL:        {p}  (size unavailable)")
    if cache_dir is None:
      print("  cache_dir:  <none — in-process FK extraction>")
    else:
      hit = (cache_dir / "manifest.json").is_file() and not recompute_cache
      print(f"  cache_dir:  {cache_dir}  ({'HIT' if hit else 'MISS / will build'})")
      # Cross-check: the cache_dir basename should match (one of) the PKL
      # stems under the per-PKL convention. If it doesn't, warn loudly.
      pkl_stems = {p.stem for p in pkl_list}
      cache_basename = cache_dir.name
      legacy_flat = cache_basename == "feature_cache"
      if not legacy_flat and cache_basename not in pkl_stems and cache_basename != "__".join(sorted(pkl_stems)):
        print(
          f"  WARNING:    cache_dir basename {cache_basename!r} does not match any "
          f"PKL stem {sorted(pkl_stems)!r}. If this is unintentional, training will "
          "use features from a DIFFERENT PKL than the one passed in."
        )

    manifest_path = None
    if cache_dir is not None:
      cache_dir.mkdir(parents=True, exist_ok=True)
      manifest_path = cache_dir / "manifest.json"
      if manifest_path.is_file() and not recompute_cache:
        if self._load_from_cache(manifest_path, cache_dir):
          print(f"  loaded:     {len(self._keys):,} clips from cache")
          return

    extractor = X2MujocoFkExtractor(mjcf)

    for pkl_path in pkl_list:
      lib = joblib.load(pkl_path)
      keys = self._select_keys(
        lib.keys(),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
      )
      keys = sorted(keys)
      if max_clips is not None:
        remaining = max_clips - len(self._keys)
        if remaining <= 0:
          break
        keys = keys[:remaining]

      for key in keys:
        clip = lib[key]
        n_raw = int(clip["dof"].shape[0])
        n_eff = (n_raw + subsample - 1) // subsample
        if n_eff < min_frames or n_eff > max_frames:
          continue
        feat = self._clip_to_features(extractor, clip)
        if feat is None:
          continue
        self._keys.append(key)
        self._features.append(feat)
        if cache_dir is not None:
          torch.save(feat, cache_dir / f"{_safe_key(key)}.pt")

      del lib

    if cache_dir is not None and manifest_path is not None:
      manifest_path.write_text(
        json.dumps({"keys": self._keys, "count": len(self._keys)}, indent=2)
      )

    if not self._keys:
      raise RuntimeError(
        "No clips passed filters across "
        f"{[str(p) for p in pkl_list]} (min_frames={min_frames}, max_frames={max_frames})"
      )

  @staticmethod
  def _select_keys(
    raw_keys,
    *,
    include_patterns: Optional[Sequence[str]],
    exclude_patterns: Optional[Sequence[str]],
  ) -> List[str]:
    """Apply optional include/exclude regex filtering. ``None`` = keep all."""
    if include_patterns is None and exclude_patterns is None:
      return list(raw_keys)
    return filter_motion_keys(
      raw_keys,
      include_patterns=include_patterns or (r".",),
      exclude_patterns=exclude_patterns or (),
    )

  def _load_from_cache(self, manifest_path: Path, cache_dir: Path) -> bool:
    manifest = json.loads(manifest_path.read_text())
    loaded = 0
    for key in manifest["keys"]:
      path = cache_dir / f"{_safe_key(key)}.pt"
      if not path.is_file():
        continue
      self._keys.append(key)
      self._features.append(torch.load(path, weights_only=True))
      loaded += 1
    return loaded > 0

  def _clip_to_features(self, extractor: X2MujocoFkExtractor, clip: dict) -> Optional[torch.Tensor]:
    try:
      inp = extractor.clip_to_input_dict(clip, subsample=self.subsample)
      feat = self.motion_rep(inp, to_normalize=self._normalize, return_numpy=False)
      if feat.dim() == 3:
        feat = feat.squeeze(0)
      return feat
    except Exception:
      return None

  def __len__(self) -> int:
    return len(self._keys)

  def __getitem__(self, idx: int) -> Dict:
    return {"keyid": self._keys[idx], "motion": self._features[idx]}


# Backwards compatibility shim — old call sites import X2LocoMotionDataset.
# Defaults *match the new semantics* (no filter), so callers that relied on the
# implicit walk/turn filter must now pass ``include_patterns=DEFAULT_INCLUDE_PATTERNS``
# explicitly.
X2LocoMotionDataset = X2MotionDataset


def default_cache_dir_for(version_dir: PathLike, pkl_paths: Union[PathLike, Sequence[PathLike]]) -> Path:
  """Return the per-PKL feature-cache directory.

  Convention: ``<version_dir>/feature_cache/<pkl_stem>/`` for a single PKL,
  or ``<version_dir>/feature_cache/<stem1>__<stem2>__.../`` for multiple PKLs.
  Putting the cache under a PKL-named subdirectory makes cross-contamination
  (e.g. smoke-PKL cache used for full-PKL training) physically impossible —
  the directory paths simply don't collide.

  Use this helper from training scripts and the bundle/build pipeline so the
  builder, dataset class, and bundle script all agree on where the cache
  lives.
  """
  if isinstance(pkl_paths, (str, Path)):
    stems = [Path(pkl_paths).stem]
  else:
    stems = sorted({Path(p).stem for p in pkl_paths})
  if not stems:
    raise ValueError("default_cache_dir_for requires at least one PKL path")
  subdir = stems[0] if len(stems) == 1 else "__".join(stems)
  return Path(version_dir) / "feature_cache" / subdir


def _safe_key(key: str) -> str:
  return key.replace("/", "__").replace("\\", "__")


def build_x2_motion_rep(
  skeleton_folder: Path,
  stats_folder: Path,
  *,
  fps: float = 30.0,
  load_stats: bool = True,
):
  """Instantiate X2 global motion rep (loads skeleton + optional stats from checkpoint dirs)."""
  from omegaconf import OmegaConf

  conf = OmegaConf.create(
    {
      "fps": fps,
      "skeleton": {
        "name": "x2skel34",
        "base_name": "x2skel34",
        "folder": str(skeleton_folder),
        "t_pose": "capture",
        "_target_": "motionbricks.motionlib.core.skeletons.x2.X2Skeleton34",
      },
      "motion_rep": {
        "name": "x2skel34_dual_root_global_joints",
        "stats": {
          "_target_": "motionbricks.motionlib.core.utils.stats.Stats",
          "folder": str(stats_folder),
          "load": load_stats,
        },
        "_target_": (
          "motionbricks.motionlib.core.motion_reps.dual_root_global_joints."
          "GlobalRootGlobalJoints"
        ),
      },
    }
  )
  return load_motion_rep(conf)
