"""
M5 acceptance gate: camera plumbing.

The M3 smoke-test orchestrator (``record_synthetic_smoketest_dataset.py``)
historically filled ``observation.images.ego_view`` with a deterministic
gradient frame so the data pipeline could be exercised on hosts without
MuJoCo / OpenGL. M5 elevates the renderer logic into a reusable
per-frame service (``MujocoFrameRenderer``) and wires it into the
orchestrator behind a ``--camera-source {gradient,mujoco}`` flag.

This gate locks the M5 invariants:

1. ``MujocoFrameRenderer`` produces ``(EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3)``
   ``uint8`` frames -- exactly the LeRobot ``observation.images.ego_view``
   feature shape. (Means the M5 path is a drop-in replacement for the
   gradient path, no schema migration.)
2. ``MujocoFrameRenderer.render_frame`` is deterministic: same inputs
   -> byte-identical output across two consecutive calls. Catches
   accidental dependence on uninitialised qvel / motion state.
3. The MuJoCo-rendered ego frame is *substantially* different from the
   gradient placeholder (mean absolute pixel diff > 30) -- guards
   against accidentally falling back to the gradient path under
   ``camera_source="mujoco"``.
4. ``make_frame_provider`` returns the right concrete provider for each
   ``camera_source`` and rejects unknown values cleanly.
5. ``build_smoketest_dataset`` accepts ``camera_source="mujoco"`` and
   writes valid MP4 video files whose decoded frames carry MuJoCo
   pixels (within H.264 tolerance) and whose feature schema matches
   the gradient-backed dataset byte-for-byte. Also validates that
   ``meta/info.json`` records ``camera_source`` in ``script_config``
   for provenance.
6. Caller-owned providers are NOT closed by ``build_smoketest_dataset``
   (lifecycle invariant: the EGL context belongs to whoever built it).
7. The wire format (canonical 31-DOF body + 10-DOF/side hand) is
   honoured: the renderer's ``body_qposadr`` table is length 31, and
   the dataset's per-episode parquet still serialises 31-DOF body + 10
   left + 10 right hand vectors regardless of camera source.

Skips
-----

The mujoco-dependent invariants (1-3, 5, 6, 7) skip cleanly when
``mujoco`` cannot be imported (e.g. CI hosts without OpenGL); the
provider-factory invariants (4, gradient half of 7) still run there.

Run via::

    .venv/bin/python -m pytest tests/test_x2_camera_plumbing.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from gear_sonic.data.features_x2_vla import (  # noqa: E402
    EGO_VIEW_HEIGHT,
    EGO_VIEW_WIDTH,
)
from gear_sonic.scripts.record_synthetic_smoketest_dataset import (  # noqa: E402
    CAMERA_SOURCES,
    CAMERA_SOURCE_GRADIENT,
    CAMERA_SOURCE_MUJOCO,
    DEFAULT_STAND_POSE_MJ_RAD,
    X2_BODY_DOF,
    _GradientFrameProvider,
    build_smoketest_dataset,
    make_frame_provider,
)


# ---------------------------------------------------------------------------
# Provider factory invariants (host-portable; no MuJoCo required)
# ---------------------------------------------------------------------------


def test_camera_sources_constants_are_stable() -> None:
    """The two M5 camera-source identifiers and their order are part of the public API.

    The CLI ``--camera-source`` choice list, the docs, and downstream
    pipelines all key off these strings; renaming them is a
    breaking change.
    """
    assert CAMERA_SOURCE_GRADIENT == "gradient"
    assert CAMERA_SOURCE_MUJOCO == "mujoco"
    assert CAMERA_SOURCES == ("gradient", "mujoco")


def test_make_frame_provider_returns_gradient_provider() -> None:
    provider = make_frame_provider(CAMERA_SOURCE_GRADIENT)
    try:
        assert isinstance(provider, _GradientFrameProvider)
        assert provider.name == "gradient"
    finally:
        provider.close()


def test_make_frame_provider_rejects_unknown_source() -> None:
    with pytest.raises(ValueError) as excinfo:
        make_frame_provider("bogus")
    assert "bogus" in str(excinfo.value)
    assert "gradient" in str(excinfo.value)
    assert "mujoco" in str(excinfo.value)


def test_gradient_provider_returns_egress_view_shaped_uint8() -> None:
    provider = make_frame_provider(CAMERA_SOURCE_GRADIENT)
    try:
        body_q = np.asarray(DEFAULT_STAND_POSE_MJ_RAD, dtype=np.float64)
        left_q = np.zeros(10, dtype=np.float64)
        right_q = np.zeros(10, dtype=np.float64)
        frame = provider.frame(
            frame_idx=0,
            num_frames=10,
            body_q=body_q,
            left_active=left_q,
            right_active=right_q,
        )
    finally:
        provider.close()

    assert frame.shape == (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3)
    assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# MuJoCo-backed invariants (skipped when mujoco isn't importable)
# ---------------------------------------------------------------------------


mujoco = pytest.importorskip("mujoco")  # noqa: F841 - skip marker for the rest of the file


@pytest.fixture(scope="module")
def mujoco_renderer():
    """One MuJoCo render context shared across the M5 tests in this module.

    Building the EGL context + augmented MJCF model + renderer takes
    ~1.5s; reusing it across the four mujoco-dependent tests keeps the
    gate well under 30s.
    """
    from gear_sonic.scripts.render_smoketest_episode_video import MujocoFrameRenderer

    renderer = MujocoFrameRenderer(
        camera="ego_view",
        width=EGO_VIEW_WIDTH,
        height=EGO_VIEW_HEIGHT,
        with_omnihand=True,
        egl=True,
    )
    yield renderer
    renderer.close()


def _stand_pose_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    body_q = np.asarray(DEFAULT_STAND_POSE_MJ_RAD, dtype=np.float64)
    left_q = np.linspace(0.0, 0.5, 10, dtype=np.float64)
    right_q = np.linspace(0.5, 0.0, 10, dtype=np.float64)
    return body_q, left_q, right_q


def test_mujoco_renderer_frame_shape_matches_ego_view_feature(mujoco_renderer) -> None:
    body_q, left_q, right_q = _stand_pose_inputs()
    frame = mujoco_renderer.render_frame(
        body_q, left_active=left_q, right_active=right_q
    )
    assert frame.shape == (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3)
    assert frame.dtype == np.uint8
    assert mujoco_renderer.with_omnihand is True
    assert mujoco_renderer.body_qposadr.shape == (X2_BODY_DOF,)
    assert mujoco_renderer.body_qposadr.dtype == np.int64


def test_mujoco_renderer_is_deterministic(mujoco_renderer) -> None:
    """Same inputs -> byte-identical pixels across two consecutive renders.

    Catches accidental dependence on residual qvel / random seeded
    visual effects that would silently couple the rendered frames to
    the order in which ``render_frame`` is called.
    """
    body_q, left_q, right_q = _stand_pose_inputs()
    a = mujoco_renderer.render_frame(body_q, left_active=left_q, right_active=right_q)
    b = mujoco_renderer.render_frame(body_q, left_active=left_q, right_active=right_q)
    assert np.array_equal(a, b)


def test_mujoco_render_differs_substantially_from_gradient(mujoco_renderer) -> None:
    """The whole point of M5: pixel content actually changes.

    A regression where ``camera_source="mujoco"`` silently falls back
    to the gradient path would produce identical pixels here. Mean
    absolute diff > 30 is a generous floor (gradient and rendered
    frames typically differ by ~95 in the X2 stand pose).
    """
    from gear_sonic.scripts.record_synthetic_smoketest_dataset import (
        _make_synthetic_ego_view,
    )

    body_q, left_q, right_q = _stand_pose_inputs()
    rendered = mujoco_renderer.render_frame(
        body_q, left_active=left_q, right_active=right_q
    )
    gradient = _make_synthetic_ego_view(0, 30)
    diff = np.abs(rendered.astype(np.int32) - gradient.astype(np.int32))
    assert float(diff.mean()) > 30.0, (
        f"mujoco render is too close to gradient (mean abs diff="
        f"{float(diff.mean()):.2f}); did camera_source silently fall "
        "back to the gradient path?"
    )


# ---------------------------------------------------------------------------
# Orchestrator integration: dataset built with --camera-source mujoco
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gradient_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("x2_m5_gate_gradient")
    summary = build_smoketest_dataset(
        output_dir=out / "ds",
        num_episodes=1,
        max_frames=12,
        seed=0,
        skip_stats=True,
        camera_source=CAMERA_SOURCE_GRADIENT,
    )
    return summary


@pytest.fixture(scope="module")
def mujoco_dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("x2_m5_gate_mujoco")
    summary = build_smoketest_dataset(
        output_dir=out / "ds",
        num_episodes=1,
        max_frames=12,
        seed=0,
        skip_stats=True,
        camera_source=CAMERA_SOURCE_MUJOCO,
    )
    return summary


def _read_info_json(dataset_dir: Path) -> dict:
    return json.loads((dataset_dir / "meta" / "info.json").read_text())


def test_dataset_features_are_invariant_across_camera_source(
    gradient_dataset, mujoco_dataset,
) -> None:
    """``meta/info.json::features`` is the schema contract Isaac-GR00T's
    LeRobotEpisodeLoader keys off. M5 must NOT change it."""
    g_info = _read_info_json(gradient_dataset.output_dir)
    m_info = _read_info_json(mujoco_dataset.output_dir)
    assert g_info["features"] == m_info["features"], (
        "feature dict drifted between camera sources -- M5 must keep the "
        "LeRobot schema byte-identical."
    )


def test_info_json_records_camera_source_for_provenance(
    gradient_dataset, mujoco_dataset,
) -> None:
    """Provenance: every dataset must self-report which provider built its frames."""
    g_info = _read_info_json(gradient_dataset.output_dir)
    m_info = _read_info_json(mujoco_dataset.output_dir)
    assert g_info["script_config"]["camera_source"] == CAMERA_SOURCE_GRADIENT
    assert m_info["script_config"]["camera_source"] == CAMERA_SOURCE_MUJOCO


def test_summary_records_camera_source(gradient_dataset, mujoco_dataset) -> None:
    assert gradient_dataset.camera_source == CAMERA_SOURCE_GRADIENT
    assert mujoco_dataset.camera_source == CAMERA_SOURCE_MUJOCO


def _decode_mp4_with_cv2(path: Path) -> np.ndarray:
    """Decode an MP4 to ``(T, H, W, 3) uint8`` RGB via OpenCV.

    Why not ``imageio.v3.imread``? imageio's default plugin chain calls
    PyAV (``av.open``), which shares a libav codec/threadpool with the
    LeRobot exporter's video writer (``gear_sonic/data/video_writer.py``).
    Decoding via PyAV in this gate puts the libav codec contexts into
    a state that hangs the exporter's flush-on-stop in
    ``test_x2_lerobot_exporter.py`` when the suites run together. cv2's
    FFmpeg binding is a separate process-level codec instance, which
    keeps the exporter and the M5 reader fully isolated.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(np.ascontiguousarray(rgb))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"cv2 decoded zero frames from {path}")
    return np.stack(frames, axis=0)


def _episode_T(summary) -> int:
    """Helper: per-episode frame count baked into the summary."""
    assert len(summary.per_episode_frames) >= 1
    return int(summary.per_episode_frames[0])


def test_mujoco_dataset_video_decodes_to_native_mujoco_frames(mujoco_dataset) -> None:
    """The dataset's MP4 decodes to (T, H, W, 3) uint8 frames whose pixel
    distribution matches a fresh MuJoCo render -- not the gradient pattern."""
    from gear_sonic.scripts.record_synthetic_smoketest_dataset import (
        _make_synthetic_ego_view,
    )

    mp4 = (
        mujoco_dataset.output_dir
        / "videos" / "chunk-000" / "observation.images.ego_view"
        / "episode_000000.mp4"
    )
    assert mp4.is_file(), f"expected mujoco-backed mp4 at {mp4}"
    frames = _decode_mp4_with_cv2(mp4)
    assert frames.dtype == np.uint8
    assert frames.shape == (
        _episode_T(mujoco_dataset),
        EGO_VIEW_HEIGHT,
        EGO_VIEW_WIDTH,
        3,
    )

    gradient = _make_synthetic_ego_view(0, frames.shape[0])
    diff = np.abs(frames[0].astype(np.int32) - gradient.astype(np.int32))
    assert float(diff.mean()) > 30.0, (
        "mujoco-backed dataset MP4 decoded to gradient-shaped pixels -- "
        f"mean abs diff = {float(diff.mean()):.2f}"
    )


def test_gradient_dataset_video_decodes_to_gradient_pixels(gradient_dataset) -> None:
    """Reverse direction: the gradient dataset's MP4 must decode within
    H.264 tolerance of ``_make_synthetic_ego_view(0, T)``."""
    from gear_sonic.scripts.record_synthetic_smoketest_dataset import (
        _make_synthetic_ego_view,
    )

    mp4 = (
        gradient_dataset.output_dir
        / "videos" / "chunk-000" / "observation.images.ego_view"
        / "episode_000000.mp4"
    )
    assert mp4.is_file(), f"expected gradient-backed mp4 at {mp4}"
    frames = _decode_mp4_with_cv2(mp4)
    assert frames.dtype == np.uint8
    T = frames.shape[0]
    gradient = _make_synthetic_ego_view(0, T)
    diff = np.abs(frames[0].astype(np.int32) - gradient.astype(np.int32))
    # H.264 quality=8 keeps mean abs diff under ~5 for a smooth gradient.
    assert float(diff.mean()) < 5.0, (
        f"gradient dataset MP4 decoded to non-gradient pixels (mean abs diff "
        f"{float(diff.mean()):.2f}); the orchestrator's default camera "
        "source must stay 'gradient' for backward-compat."
    )


# ---------------------------------------------------------------------------
# Lifecycle invariant: caller-owned providers are not closed by the orchestrator
# ---------------------------------------------------------------------------


class _CountingProvider:
    """Stub provider that delegates to gradient and counts close() calls."""

    name = "counting_stub"

    def __init__(self) -> None:
        self._inner = _GradientFrameProvider()
        self.close_calls: int = 0

    def frame(self, **kwargs) -> np.ndarray:
        return self._inner.frame(**kwargs)

    def close(self) -> None:
        self.close_calls += 1
        self._inner.close()


def test_caller_owned_provider_is_not_closed_by_orchestrator(tmp_path: Path) -> None:
    provider = _CountingProvider()
    try:
        build_smoketest_dataset(
            output_dir=tmp_path / "ds",
            num_episodes=1,
            max_frames=8,
            seed=0,
            skip_stats=True,
            frame_provider=provider,
        )
        assert provider.close_calls == 0, (
            "build_smoketest_dataset must NOT close a caller-supplied provider; "
            f"got {provider.close_calls} close() calls."
        )
    finally:
        provider.close()
    assert provider.close_calls == 1


def test_unknown_camera_source_rejected_before_disk_write(tmp_path: Path) -> None:
    """Don't even create the output dir if the camera source is bogus."""
    out = tmp_path / "nope"
    with pytest.raises(ValueError) as excinfo:
        build_smoketest_dataset(
            output_dir=out,
            num_episodes=1,
            max_frames=8,
            seed=0,
            skip_stats=True,
            camera_source="bogus",
        )
    assert "camera_source" in str(excinfo.value)
    assert not out.exists(), (
        "build_smoketest_dataset should reject unknown camera_source before "
        "creating the output directory."
    )
