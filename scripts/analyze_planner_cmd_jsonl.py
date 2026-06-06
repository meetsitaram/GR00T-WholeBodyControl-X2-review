"""Offline analyzer: VR vs training-distribution intent + StickFilter sweep.

Three jobs in one script:

1. **Baseline characterisation.** Load the live Quest 3 raw fixture
   captured by ``quest3_manager_x2 --quest3-record-to`` and reconstruct
   the dense 50 Hz velocity-intent stream the kplanner consumes
   (deadzone -> continuous sticks -> ``intent_to_velocity``). Plot it
   alongside the curated PKL primitive distributions extracted by
   ``motionbricks/scripts/extract_training_intent_stats.py``.

2. **StickFilter sweep.** For a small grid of ``(tau, slew)`` configs
   (uniform across the three stick channels), re-run the same VR
   stream through ``StickFilter`` before the deadzone -> velocity step
   and tabulate per-channel ``|d/dt|`` percentiles. The goal is to
   bring the VR p99 close to the PKL training-distribution p99 without
   adding objectionable operator lag.

3. **Recommendation.** Pick the (tau, slew) config whose ``|d(vel_z)/dt|
   p99`` lands closest to the PKL Flavor-A reference (the rawest
   training-time finite difference), subject to a soft cap on operator
   lag. Write the chosen config to ``recommended_config.json`` so the
   manager-side defaults (and the wrapper env vars) can pick it up
   without a human re-doing the algebra.

Inputs
------

* ``--vr-raw``          : path to ``quest3_raw_*.jsonl`` captured by the manager
* ``--training-stats``  : path to ``training_intent_stats.json`` from the extractor
* ``--out-dir``         : directory to write PNGs + markdown table + recommended config

Outputs
-------

* ``intent_overlay.png``      time-series + histogram + |d/dt| panels comparing
                              raw VR vs each sweep config vs PKL flavors A & B
* ``intent_stats.md``         markdown table of per-channel percentiles
* ``recommended_config.json`` chosen ``(tau_fwd, slew_fwd, ...)`` triple

Run from repo root::

  .venv/bin/python scripts/analyze_planner_cmd_jsonl.py \\
      --vr-raw out/intent_reference/live/quest3_raw_20260531.jsonl \\
      --training-stats out/intent_reference/training_intent_stats.json \\
      --out-dir out/intent_reference/analysis_20260531
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import gear_sonic.scripts.x2_kplanner as kp  # noqa: E402
from gear_sonic.scripts.x2_kplanner import intent_to_velocity  # noqa: E402
from gear_sonic.utils.planner.state_machine import LocomotionCommand  # noqa: E402
from gear_sonic.utils.teleop.vr.stick_smoother import (  # noqa: E402
    StickFilter,
    StickFilterConfig,
)


log = logging.getLogger("analyze_planner_cmd_jsonl")


# ---------------------------------------------------------------------------
# Stick decode (mirror IntentDecoder._continuous_stick_targets)
# ---------------------------------------------------------------------------


def _rescaled_axis(value: float, deadzone: float) -> float:
    """Deadzoned rescale to [-1, 1], sign-preserving. Mirrors
    ``IntentDecoder._rescaled_axis`` so the offline replay agrees with
    the live manager exactly."""
    sign = 1.0 if value >= 0 else -1.0
    mag = abs(float(value))
    if mag <= deadzone:
        return 0.0
    denom = max(1.0 - deadzone, 1e-6)
    return sign * min(1.0, (mag - deadzone) / denom)


def _continuous_sticks_from_axes(
    *, lx: float, ly: float, rx: float,
    deadzone: float, yaw_max: float,
) -> tuple[float, float, float]:
    """Replay ``IntentDecoder._continuous_stick_targets`` semantics.

    Sign convention (matches the production decoder + the kplanner's
    ``_BASE_VELOCITY`` table):
      * ``ly > 0`` -> forward -> ``stick_fwd > 0``
      * ``lx > 0`` -> right step -> ``stick_side > 0``
      * ``rx > 0`` -> turn right -> ``stick_yaw > 0`` (clamped by yaw_max)
    """
    return (
        _rescaled_axis(ly, deadzone),
        _rescaled_axis(lx, deadzone),
        _rescaled_axis(rx, deadzone) * yaw_max,
    )


# ---------------------------------------------------------------------------
# Dense velocity reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_dense_velocity(
    rows: list[dict],
    deadzone: float,
    yaw_max: float,
    stick_filter: StickFilter | None,
) -> dict[str, np.ndarray]:
    """Walk the VR raw fixture, optionally apply a StickFilter, and
    return the dense per-tick streams the analyzer plots / aggregates.

    Returns a dict with keys (length = number of LOCOMOTION-mode ticks):
      * ``t``                : seconds since first LOCOMOTION tick
      * ``stick_raw``        : (N, 3) post-deadzone (fwd, side, yaw)
      * ``stick_filtered``   : (N, 3) post-filter (fwd, side, yaw)
      * ``velocity``         : (N, 4) (yaw_rate, vel_x, vel_z, hip_h)
      * ``axes_raw``         : (N, 4) (lx, ly, rx, ry) for diagnostics

    Non-LOCOMOTION ticks are dropped. The StickFilter state survives
    across non-LOCOMOTION gaps inside a single session (so a brief
    chord-quiet does not reset the filter), but ``reset()`` is called
    on the transition INTO the first LOCOMOTION tick of a run so the
    filter does not carry stale state.
    """
    t = []
    raw_sticks = []
    filt_sticks = []
    velocities = []
    axes_raw = []

    if stick_filter is not None:
        stick_filter.reset()

    t0_mono: float | None = None
    t_prev_mono: float | None = None
    prev_was_locomotion = False

    for row in rows:
        if row.get("mode") != "LOCOMOTION":
            prev_was_locomotion = False
            continue
        # Reset the filter on a fresh LOCOMOTION entry so the first
        # post-mode-flip tick is a pass-through.
        if not prev_was_locomotion and stick_filter is not None:
            stick_filter.reset()
        prev_was_locomotion = True

        t_mono = float(row["t_mono"])
        if t0_mono is None:
            t0_mono = t_mono
            dt = 0.02
        else:
            dt = max(0.0, t_mono - (t_prev_mono if t_prev_mono is not None else t_mono))
        t_prev_mono = t_mono

        ax = row["axes_post_invert"]
        lx, ly, rx, ry = ax["lx"], ax["ly"], ax["rx"], ax["ry"]
        s_fwd_raw, s_side_raw, s_yaw_raw = _continuous_sticks_from_axes(
            lx=lx, ly=ly, rx=rx, deadzone=deadzone, yaw_max=yaw_max,
        )

        if stick_filter is not None:
            s_fwd_out, s_side_out, s_yaw_out = stick_filter.step(
                stick_fwd=s_fwd_raw,
                stick_side=s_side_raw,
                stick_yaw=s_yaw_raw,
                dt=dt,
            )
        else:
            s_fwd_out, s_side_out, s_yaw_out = s_fwd_raw, s_side_raw, s_yaw_raw

        cmd = LocomotionCommand(
            intent="locomotion",
            magnitude="continuous",
            stick_fwd=s_fwd_out,
            stick_side=s_side_out,
            stick_yaw=s_yaw_out,
        )
        yr, vx, vz, hh = intent_to_velocity(cmd)

        t.append(t_mono - t0_mono)
        raw_sticks.append([s_fwd_raw, s_side_raw, s_yaw_raw])
        filt_sticks.append([s_fwd_out, s_side_out, s_yaw_out])
        velocities.append([yr, vx, vz, hh])
        axes_raw.append([lx, ly, rx, ry])

    if not t:
        return {
            "t": np.zeros(0),
            "stick_raw": np.zeros((0, 3)),
            "stick_filtered": np.zeros((0, 3)),
            "velocity": np.zeros((0, 4)),
            "axes_raw": np.zeros((0, 4)),
        }
    return {
        "t": np.asarray(t),
        "stick_raw": np.asarray(raw_sticks),
        "stick_filtered": np.asarray(filt_sticks),
        "velocity": np.asarray(velocities),
        "axes_raw": np.asarray(axes_raw),
    }


def _velocity_derivative(velocity: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return (N, 4) per-tick finite difference in velocity-channel-units/s.

    ``t`` is a monotonic seconds vector. First row is 0 by convention.
    Uses adjacent-tick diffs (i.e. dt ~= 0.02 s @ 50 Hz) so the result
    has the same time alignment as the input.
    """
    if velocity.shape[0] < 2:
        return np.zeros_like(velocity)
    dt = np.diff(t)
    dt = np.where(dt <= 0, 1.0 / 50.0, dt)  # guard against zero-dt rows
    diff = velocity[1:, :] - velocity[:-1, :]
    out = np.zeros_like(velocity)
    out[1:, :] = diff / dt[:, None]
    return out


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


_PCT_LIST = [50, 75, 95, 99]


def _percentiles(x: np.ndarray, abs_value: bool = False) -> dict[str, float]:
    if x.shape[0] == 0:
        return {f"p{p:02d}": 0.0 for p in _PCT_LIST} | {"max": 0.0, "mean": 0.0}
    v = np.abs(x) if abs_value else x
    out = {f"p{p:02d}": float(np.percentile(v, p)) for p in _PCT_LIST}
    out["max"] = float(np.max(v))
    out["mean"] = float(np.mean(v))
    return out


def _summarize_config(
    label: str,
    cfg: StickFilterConfig | None,
    streams: dict[str, np.ndarray],
    raw_streams: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Build a stats row for one filter configuration."""
    vel = streams["velocity"]
    t = streams["t"]
    dvel = _velocity_derivative(vel, t)

    summary: dict[str, Any] = {
        "label": label,
        "filter": asdict(cfg) if cfg is not None else None,
        "ticks": int(vel.shape[0]),
        # Velocity ranges by channel
        "velocity_abs": {
            "yaw_rate": _percentiles(vel[:, 0], abs_value=True),
            "vel_x":    _percentiles(vel[:, 1], abs_value=True),
            "vel_z":    _percentiles(vel[:, 2], abs_value=True),
        },
        # Velocity rate-of-change (the step-input metric)
        "dvelocity_abs": {
            "yaw_rate": _percentiles(dvel[:, 0], abs_value=True),
            "vel_x":    _percentiles(dvel[:, 1], abs_value=True),
            "vel_z":    _percentiles(dvel[:, 2], abs_value=True),
        },
    }

    # Lag proxy: mean abs deviation between filtered and raw sticks. The
    # tighter this is, the less the operator "feels" the filter. Only
    # meaningful when raw_streams is provided (the baseline summary
    # passes the same streams in twice so lag == 0).
    if (
        raw_streams is not None
        and raw_streams["stick_raw"].shape == streams["stick_filtered"].shape
        and streams["stick_filtered"].shape[0] > 0
    ):
        delta = streams["stick_filtered"] - raw_streams["stick_raw"]
        summary["stick_lag_proxy_abs"] = {
            "fwd":  _percentiles(delta[:, 0], abs_value=True),
            "side": _percentiles(delta[:, 1], abs_value=True),
            "yaw":  _percentiles(delta[:, 2], abs_value=True),
        }
    return summary


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------


def _sweep_grid(
    rows: list[dict],
    deadzone: float,
    yaw_max: float,
    tau_grid: list[float],
    slew_grid: list[float],
    release_tau: float | None,
) -> list[dict[str, Any]]:
    """For each (tau, slew) pair, reconstruct the dense stream and
    return a per-config stats record. ``tau == 0`` + ``slew == inf``
    is the no-filter baseline.
    """
    baseline_streams = _reconstruct_dense_velocity(
        rows, deadzone=deadzone, yaw_max=yaw_max, stick_filter=None,
    )
    summaries = [
        _summarize_config(
            "vr_raw",
            None,
            baseline_streams,
            baseline_streams,
        )
    ]

    for tau in tau_grid:
        for slew in slew_grid:
            cfg = StickFilterConfig(
                tau_lpf_fwd_s=tau, slew_max_fwd_per_s=slew,
                tau_lpf_side_s=tau, slew_max_side_per_s=slew,
                tau_lpf_yaw_s=tau, slew_max_yaw_per_s=slew,
                return_to_zero_tau_fwd_s=release_tau,
                return_to_zero_tau_side_s=release_tau,
                return_to_zero_tau_yaw_s=release_tau,
            )
            label = f"tau{tau:.2f}_slew{slew:g}"
            streams = _reconstruct_dense_velocity(
                rows, deadzone=deadzone, yaw_max=yaw_max,
                stick_filter=StickFilter(cfg),
            )
            summaries.append(
                _summarize_config(label, cfg, streams, baseline_streams)
            )
    return summaries


def _pick_recommendation(
    summaries: list[dict[str, Any]],
    pkl_a_p99_vel_z: float,
    pkl_b_p99_vel_z: float,
    tau_min: float = 0.05,
    tau_max: float = 0.30,
) -> dict[str, Any]:
    """Pick the best (tau, slew) config by matching VR vel_z |d/dt| p99
    to the PKL training-distribution band.

    Operator-feel bounds (hard constraints):
      * ``tau >= tau_min`` -- below this the filter does nothing useful.
      * ``tau <= tau_max`` -- above this the operator notices the lag.
      * slew on stick channels must not over-clamp the operator's full
        deflection; we require slew*tau >= 0.5 (= LPF can swing half
        the dynamic range within one tau). slew=inf trivially passes.

    Among configs that meet the constraints, the score is
    ``|vel_z_p99 - target|`` where ``target`` is the midpoint of the
    PKL Flavor-A and Flavor-B p99s. We aim for the middle of the
    training band rather than the rawest edge -- the kplanner already
    smooths internally (window=8 in the live replay path), so landing
    near Flavor B is the better operational target.

    Returns ``{"label", "filter", "vel_z_p99_filtered", "vel_z_p99_target", ...}``.
    """
    target = 0.5 * (pkl_a_p99_vel_z + pkl_b_p99_vel_z)
    best = None
    best_score = float("inf")
    for s in summaries:
        if s["label"] == "vr_raw" or s["filter"] is None:
            continue
        cfg = s["filter"]
        tau = float(cfg["tau_lpf_fwd_s"])
        slew = float(cfg["slew_max_fwd_per_s"])
        if not (tau_min <= tau <= tau_max):
            continue
        # Slew adequacy: tau-bandwidth filter can't reach half the
        # dynamic range within tau if slew is too low. Skip configs
        # that over-clamp the engaged-stick response.
        if math.isfinite(slew) and (slew * tau) < 0.5:
            continue
        vel_z_p99 = s["dvelocity_abs"]["vel_z"]["p99"]
        score = abs(vel_z_p99 - target)
        # Tiebreak: prefer smaller tau (less lag) when scores are within 0.05
        if abs(score - best_score) < 0.05 and best is not None:
            if tau < float(best["filter"]["tau_lpf_fwd_s"]):
                best = s
                best_score = score
            continue
        if score < best_score:
            best = s
            best_score = score

    if best is None:
        # No config met the constraints. Fall through to the smallest
        # non-trivial tau in the sweep (the doc generator will flag).
        for s in summaries[1:]:  # skip vr_raw
            if s["filter"] is not None and float(s["filter"]["tau_lpf_fwd_s"]) > 0:
                best = s
                break

    if best is None:
        return {
            "label": "vr_raw",
            "filter": None,
            "vel_z_p99_filtered": summaries[0]["dvelocity_abs"]["vel_z"]["p99"],
            "vel_z_p99_pkl_a": pkl_a_p99_vel_z,
            "vel_z_p99_pkl_b": pkl_b_p99_vel_z,
            "vel_z_p99_target": target,
            "stick_lag_p99": 0.0,
            "note": "no config met constraints; falling back to raw",
        }

    lag_dict = best.get("stick_lag_proxy_abs", {})
    return {
        "label": best["label"],
        "filter": best["filter"],
        "vel_z_p99_filtered": best["dvelocity_abs"]["vel_z"]["p99"],
        "vel_z_p99_pkl_a": pkl_a_p99_vel_z,
        "vel_z_p99_pkl_b": pkl_b_p99_vel_z,
        "vel_z_p99_target": target,
        "stick_lag_p99": max(
            lag_dict.get("fwd",  {"p99": 0.0})["p99"],
            lag_dict.get("side", {"p99": 0.0})["p99"],
            lag_dict.get("yaw",  {"p99": 0.0})["p99"],
        ),
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def _plot(
    *,
    rows: list[dict],
    summaries: list[dict[str, Any]],
    training_stats: dict,
    recommended: dict[str, Any] | None,
    out_png: Path,
    deadzone: float,
    yaw_max: float,
) -> None:
    """Multi-panel PNG: time series, histograms, |d/dt| percentiles.

    Panels (3 rows x 3 cols):
      Row 1: time series of stick_fwd / side / yaw (raw vs recommended filtered)
      Row 2: time series of resolved vel_z / vel_x / yaw_rate (same overlay)
      Row 3: |d(vel)/dt| p50/p95/p99 bars for each sweep config + PKL refs
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Re-reconstruct the streams we want to plot (baseline + recommended).
    base = _reconstruct_dense_velocity(
        rows, deadzone=deadzone, yaw_max=yaw_max, stick_filter=None,
    )
    rec_streams = None
    rec_label = None
    if recommended is not None and recommended.get("filter") is not None:
        rec_label = recommended["label"]
        rec_cfg_dict = recommended["filter"]
        rec_cfg = StickFilterConfig(**rec_cfg_dict)
        rec_streams = _reconstruct_dense_velocity(
            rows, deadzone=deadzone, yaw_max=yaw_max,
            stick_filter=StickFilter(rec_cfg),
        )

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(
        f"VR stick-smoothing analysis (rec={rec_label or 'none'})",
        fontsize=14,
    )
    channel_titles = ["fwd (ly post-deadzone)", "side (lx)", "yaw (rx * yaw_max)"]
    vel_titles = ["vel_z (forward, m/s)", "vel_x (lateral, m/s)", "yaw_rate (rad/s)"]
    vel_channels_idx = [2, 1, 0]

    # Row 1: stick time series.
    for j in range(3):
        ax = axes[0, j]
        if base["stick_raw"].shape[0] > 0:
            ax.plot(base["t"], base["stick_raw"][:, j],
                    color="tab:blue", linewidth=0.8, label="raw")
        if rec_streams is not None and rec_streams["stick_filtered"].shape[0] > 0:
            ax.plot(rec_streams["t"], rec_streams["stick_filtered"][:, j],
                    color="tab:orange", linewidth=1.2, label=f"filtered ({rec_label})")
        ax.set_title(f"stick {channel_titles[j]}")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)

    # Row 2: velocity time series.
    for j, idx in enumerate(vel_channels_idx):
        ax = axes[1, j]
        if base["velocity"].shape[0] > 0:
            ax.plot(base["t"], base["velocity"][:, idx],
                    color="tab:blue", linewidth=0.8, label="raw")
        if rec_streams is not None and rec_streams["velocity"].shape[0] > 0:
            ax.plot(rec_streams["t"], rec_streams["velocity"][:, idx],
                    color="tab:orange", linewidth=1.2, label=f"filtered ({rec_label})")
        ax.set_title(vel_titles[j])
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("t (s)")

    # Row 3: |d/dt| bars (per-channel) for each sweep config + PKL refs.
    pkl_A = training_stats["aggregate"]["dA_dt_stats"]
    pkl_B = training_stats["aggregate"]["dB_dt_stats"]
    for j, ch in enumerate(["vel_z", "vel_x", "yaw_rate"]):
        ax = axes[2, j]
        labels = [s["label"] for s in summaries]
        p99s = [s["dvelocity_abs"][ch]["p99"] for s in summaries]
        colors = [
            "tab:red" if s["label"] == "vr_raw"
            else ("tab:orange" if s["label"] == rec_label else "tab:gray")
            for s in summaries
        ]
        ax.bar(range(len(labels)), p99s, color=colors)
        ax.axhline(pkl_A[ch]["abs_p99"],
                   color="tab:green", linestyle="--", label="PKL Flavor A p99")
        ax.axhline(pkl_B[ch]["abs_p99"],
                   color="tab:purple", linestyle=":", label="PKL Flavor B p99")
        ax.set_title(f"|d({ch})/dt| p99 by config")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=80, fontsize=7)
        ax.set_ylabel(
            "m/s^2" if ch in ("vel_x", "vel_z") else "rad/s^2"
        )
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out_png)


# ---------------------------------------------------------------------------
# Markdown table
# ---------------------------------------------------------------------------


def _write_markdown_table(
    *,
    summaries: list[dict[str, Any]],
    training_stats: dict,
    out_md: Path,
    recommended: dict[str, Any] | None,
) -> None:
    """Per-config + PKL reference summary table."""
    pkl_A = training_stats["aggregate"]["dA_dt_stats"]
    pkl_B = training_stats["aggregate"]["dB_dt_stats"]
    rec_label = recommended["label"] if recommended else None

    lines = []
    lines.append("# VR stick-smoothing analysis\n")
    lines.append("## |d(velocity)/dt| percentiles (step-input metric)\n")
    lines.append(
        "| config | vel_z p95 | vel_z p99 | vel_z max | "
        "vel_x p99 | yaw_rate p99 | stick_lag p99 |\n"
    )
    lines.append(
        "| --- | ---:| ---:| ---:| ---:| ---:| ---:|\n"
    )
    for s in summaries:
        marker = "**[REC]** " if s["label"] == rec_label else ""
        dv = s["dvelocity_abs"]
        lag = s.get("stick_lag_proxy_abs")
        lag_p99 = (
            max(lag["fwd"]["p99"], lag["side"]["p99"], lag["yaw"]["p99"])
            if lag else 0.0
        )
        lines.append(
            f"| {marker}{s['label']} "
            f"| {dv['vel_z']['p95']:.3f} "
            f"| {dv['vel_z']['p99']:.3f} "
            f"| {dv['vel_z']['max']:.3f} "
            f"| {dv['vel_x']['p99']:.3f} "
            f"| {dv['yaw_rate']['p99']:.3f} "
            f"| {lag_p99:.4f} "
            f"|\n"
        )
    lines.append(
        f"| **PKL Flavor A (training, window=2)** "
        f"| {pkl_A['vel_z']['abs_p95']:.3f} "
        f"| {pkl_A['vel_z']['abs_p99']:.3f} "
        f"| {pkl_A['vel_z']['abs_max']:.3f} "
        f"| {pkl_A['vel_x']['abs_p99']:.3f} "
        f"| {pkl_A['yaw_rate']['abs_p99']:.3f} "
        f"| - |\n"
    )
    lines.append(
        f"| **PKL Flavor B (live replay, window=8)** "
        f"| {pkl_B['vel_z']['abs_p95']:.3f} "
        f"| {pkl_B['vel_z']['abs_p99']:.3f} "
        f"| {pkl_B['vel_z']['abs_max']:.3f} "
        f"| {pkl_B['vel_x']['abs_p99']:.3f} "
        f"| {pkl_B['yaw_rate']['abs_p99']:.3f} "
        f"| - |\n"
    )

    if recommended:
        lines.append(
            "\n## Recommended config\n\n"
            f"- **Label**: `{recommended['label']}`\n"
            f"- **Filter**: `{recommended['filter']}`\n"
            f"- VR-filtered vel_z |d/dt| p99: "
            f"`{recommended['vel_z_p99_filtered']:.3f}` m/s^2\n"
            f"- Target band (PKL): "
            f"`{recommended['vel_z_p99_pkl_b']:.3f}` (Flavor B, window=8) ... "
            f"`{recommended['vel_z_p99_pkl_a']:.3f}` (Flavor A, window=2); "
            f"midpoint `{recommended['vel_z_p99_target']:.3f}` m/s^2\n"
            f"- Operator stick-lag proxy p99: "
            f"`{recommended['stick_lag_p99']:.4f}` (stick-units)\n"
        )
        if recommended.get("note"):
            lines.append(f"\n> Note: {recommended['note']}\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(lines))
    log.info("wrote %s", out_md)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--vr-raw", type=Path, required=True,
        help="Path to quest3_raw_*.jsonl from quest3_manager_x2 --quest3-record-to.",
    )
    p.add_argument(
        "--training-stats", type=Path, required=True,
        help="Path to training_intent_stats.json from "
             "motionbricks/scripts/extract_training_intent_stats.py.",
    )
    p.add_argument(
        "--out-dir", type=Path, required=True,
        help="Output directory; created on demand. Contains intent_overlay.png, "
             "intent_stats.md, recommended_config.json.",
    )
    p.add_argument(
        "--deadzone", type=float, default=0.30,
        help="Manager IntentDecoder.stick_deadzone (default 0.30).",
    )
    p.add_argument(
        "--yaw-max", type=float, default=0.5,
        help="Manager intent_continuous_yaw_max (default 0.5).",
    )
    p.add_argument(
        "--tau-grid", type=str,
        default="0.00,0.10,0.15,0.20,0.30",
        help="Comma-separated LPF tau values (s).",
    )
    p.add_argument(
        "--slew-grid", type=str,
        default="2.0,4.0,6.0,inf",
        help="Comma-separated slew values (per-sec). 'inf' disables slew.",
    )
    p.add_argument(
        "--release-tau", type=float, default=None,
        help="Optional asymmetric release tau (s). When set, the filter "
             "uses this tau on stick release. None = symmetric.",
    )
    p.add_argument(
        "--tau-min", type=float, default=0.05,
        help="Minimum LPF tau (s) the recommender will accept. Configs "
             "below this are filtering too little to matter. Default 0.05.",
    )
    p.add_argument(
        "--tau-max", type=float, default=0.30,
        help="Maximum LPF tau (s) the recommender will accept. Configs "
             "above this introduce operator-noticeable lag. Default 0.30.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.DEBUG if verbose else logging.INFO,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _setup_logging(args.verbose)
    # Ensure kplanner runtime scales are at unity so the analyzer's
    # intent_to_velocity output matches the daemon defaults (the
    # wrapper does not pass any scale flags by default).
    kp._RUNTIME_FORWARD_SCALE = 1.0
    kp._RUNTIME_BACKWARD_SCALE = 1.0
    kp._RUNTIME_LATERAL_SCALE = 1.0
    kp._RUNTIME_TURN_LEFT_SCALE = 1.0
    kp._RUNTIME_TURN_RIGHT_SCALE = 1.0

    log.info("loading VR raw: %s", args.vr_raw)
    rows = [json.loads(l) for l in args.vr_raw.read_text().splitlines()]
    log.info("  %d rows", len(rows))

    log.info("loading training stats: %s", args.training_stats)
    training_stats = json.loads(args.training_stats.read_text())
    pkl_a_p99 = training_stats["aggregate"]["dA_dt_stats"]["vel_z"]["abs_p99"]
    pkl_b_p99 = training_stats["aggregate"]["dB_dt_stats"]["vel_z"]["abs_p99"]
    log.info(
        "  PKL Flavor A p99 |d(vel_z)/dt| = %.3f m/s^2", pkl_a_p99
    )
    log.info(
        "  PKL Flavor B p99 |d(vel_z)/dt| = %.3f m/s^2", pkl_b_p99
    )

    def _parse_grid(s: str) -> list[float]:
        out = []
        for tok in s.split(","):
            tok = tok.strip().lower()
            if tok in ("inf", "infinity", "+inf"):
                out.append(float("inf"))
            else:
                out.append(float(tok))
        return out

    tau_grid = _parse_grid(args.tau_grid)
    slew_grid = _parse_grid(args.slew_grid)
    log.info("sweep: tau_grid=%s  slew_grid=%s", tau_grid, slew_grid)

    summaries = _sweep_grid(
        rows,
        deadzone=args.deadzone,
        yaw_max=args.yaw_max,
        tau_grid=tau_grid,
        slew_grid=slew_grid,
        release_tau=args.release_tau,
    )

    log.info("baseline (vr_raw): vel_z |d/dt| p99 = %.3f m/s^2",
             summaries[0]["dvelocity_abs"]["vel_z"]["p99"])

    recommended = _pick_recommendation(
        summaries,
        pkl_a_p99_vel_z=pkl_a_p99,
        pkl_b_p99_vel_z=pkl_b_p99,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
    )
    log.info(
        "recommended: %s -> vel_z |d/dt| p99 = %.3f m/s^2 (target band "
        "%.3f..%.3f midpoint %.3f), stick_lag p99 = %.4f",
        recommended["label"],
        recommended.get("vel_z_p99_filtered") or -1.0,
        recommended.get("vel_z_p99_pkl_b") or -1.0,
        recommended.get("vel_z_p99_pkl_a") or -1.0,
        recommended.get("vel_z_p99_target") or -1.0,
        recommended.get("stick_lag_p99") or -1.0,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _plot(
        rows=rows,
        summaries=summaries,
        training_stats=training_stats,
        recommended=recommended,
        out_png=args.out_dir / "intent_overlay.png",
        deadzone=args.deadzone,
        yaw_max=args.yaw_max,
    )
    _write_markdown_table(
        summaries=summaries,
        training_stats=training_stats,
        out_md=args.out_dir / "intent_stats.md",
        recommended=recommended,
    )
    with (args.out_dir / "recommended_config.json").open("w") as f:
        # Use _json_sanitize so math.inf serialises as "inf" instead of
        # Python's non-standard `Infinity` literal (downstream parsers
        # such as the wrapper env-var generator must round-trip cleanly).
        json.dump(_json_sanitize(recommended), f, indent=2)
    log.info("wrote %s", args.out_dir / "recommended_config.json")

    # Also dump the full per-config summary as JSON so downstream code
    # (the doc generator, future regressions) can grep it.
    with (args.out_dir / "sweep_summary.json").open("w") as f:
        json.dump(
            _json_sanitize({
                "vr_raw_jsonl": str(args.vr_raw),
                "training_stats_json": str(args.training_stats),
                "deadzone": args.deadzone,
                "yaw_max": args.yaw_max,
                "tau_grid": tau_grid,
                "slew_grid": slew_grid,
                "release_tau": args.release_tau,
                "tau_min": args.tau_min,
                "tau_max": args.tau_max,
                "pkl_a_vel_z_p99": pkl_a_p99,
                "pkl_b_vel_z_p99": pkl_b_p99,
                "configs": summaries,
                "recommended": recommended,
            }),
            f, indent=2,
        )
    log.info("wrote %s", args.out_dir / "sweep_summary.json")
    return 0


def _json_sanitize(obj):
    """Recursively replace ``math.inf`` / ``-math.inf`` with the string
    sentinel ``"inf"`` / ``"-inf"`` and NaN with ``None``, so the
    output is RFC-8259 compliant. Used right before ``json.dump``."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        if math.isnan(obj):
            return None
    return obj


if __name__ == "__main__":
    sys.exit(main())
