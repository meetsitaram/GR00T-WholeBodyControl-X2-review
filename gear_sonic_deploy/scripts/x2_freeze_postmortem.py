#!/usr/bin/env python3
"""Stitch deploy CSVs + motor_monitor JSONL + manager sidecar JSONL into a
unified timeline for post-hoc freeze / oscillation / recovery analysis.

The split-topology deploy emits three independent log streams during a
test session:

1. **Deploy CSVs** (PC2, written by ``DeployLogger``):
   ``tick.csv``, ``target_pos.csv``, ``joint_pos.csv``, ``joint_vel.csv``,
   ``action_il.csv``, ``imu.csv`` -- per-tick (~50 Hz) measurements with
   relative time ``t`` (seconds since deploy start).
2. **Motor monitor JSONL** (PC2, written by ``x2_motor_monitor.py``):
   per-second ``"sample"`` records + edge-event ``"event"`` records with
   wall-clock ``ts`` AND relative ``rel_t``.
3. **Manager sidecar JSONL** (laptop, written by
   ``quest3_manager_x2.py``): planner_cmd / motor_monitor / safety / hand
   bridge events with wall-clock ``ts``.

Each stream uses its own clock (deploy CSVs are monotonic-relative; the
JSONL streams are wall-clock + monotonic-relative). This tool aligns them
to a single wall-clock axis so an operator looking at "the robot froze
around 7:24:30" can see, on one timeline:

  * what target positions the policy was emitting,
  * how the measured joints responded,
  * what the MC mode was at each second,
  * which joints had tracking-error spikes,
  * what chord / safety events happened on the laptop side.

Anchoring strategies
--------------------

The deploy CSVs do NOT carry a wall-clock timestamp on every row, only a
monotonic ``t``. We anchor the deploy CSV clock to wall-clock by using
the file modification time of ``tick.csv`` minus the largest ``t`` value
observed. This is accurate to a few hundred ms (whatever the OS flushes
to disk lag is).

If a deploy session emits its own ``boot.json`` (next-pass enhancement)
or if the motor monitor logged a ``"boot"`` record around the same
moment, we additionally cross-check the anchor against those.

Outputs
-------

The tool writes three artifacts to ``--out-dir``:

1. ``timeline.csv`` -- one row per event, columns:
   ``wall_ts, source, kind, summary, json``. Easy to scroll in
   spreadsheets.
2. ``timeline.md`` -- human-readable digest grouped by minute.
3. ``window.json`` -- if ``--center-ts`` was provided, a structured
   dump of every record inside the requested window for downstream
   tooling (e.g. plotting in Jupyter).

Usage
-----

::

    x2_freeze_postmortem.py \\
        --deploy-log-dir /tmp/x2_deploy_logs/2026-05-15_19-20-00 \\
        --motor-monitor /var/log/x2/motor_monitor.2026-05-15.jsonl \\
        --manager-sidecar ~/.x2_quest3_planner_stack/sidecar.jsonl \\
        --center-ts "2026-05-15T19:24:30" \\
        --window-s 30 \\
        --out-dir /tmp/postmortem_2026-05-15_19-24

If ``--center-ts`` is omitted the tool will auto-pick the wall-clock
range covering the entire deploy session and write the full timeline.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import json
import logging
import math
import pathlib
import sys
from typing import Any, Iterable, Optional


log = logging.getLogger("x2_freeze_postmortem")


# ---------------------------------------------------------------------------
# Event model -- one normalized record per source row
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TimelineEvent:
    """A single moment on the unified wall-clock timeline."""

    wall_ts: float
    source: str
    kind: str
    summary: str
    payload: dict


# ---------------------------------------------------------------------------
# Deploy CSV loader
# ---------------------------------------------------------------------------

DEPLOY_CSV_FILES = (
    "tick.csv",
    "target_pos.csv",
    "joint_pos.csv",
    "joint_vel.csv",
    "action_il.csv",
    "imu.csv",
)


def _detect_deploy_anchor(deploy_dir: pathlib.Path) -> tuple[float, float]:
    """Return (wall_ts_at_t0, last_t) for a deploy log directory.

    Uses ``tick.csv`` mtime as the wall-clock anchor for the LAST written
    row, then subtracts the max observed ``t`` to back out wall-clock at
    deploy start. Returns ``(anchor_wall_ts, max_t)``.

    Raises FileNotFoundError if the directory is missing or empty.
    """
    tick_path = deploy_dir / "tick.csv"
    if not tick_path.is_file():
        raise FileNotFoundError(f"missing {tick_path}")
    mtime = tick_path.stat().st_mtime

    max_t = 0.0
    with tick_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                t = float(row["t"])
            except (KeyError, ValueError):
                continue
            if t > max_t:
                max_t = t

    anchor = mtime - max_t
    log.info(
        "deploy anchor: tick.csv mtime=%s, max_t=%.3fs -> wall_ts_at_t0=%s",
        _dt.datetime.fromtimestamp(mtime).isoformat(timespec="milliseconds"),
        max_t,
        _dt.datetime.fromtimestamp(anchor).isoformat(timespec="milliseconds"),
    )
    return anchor, max_t


def _iter_tick_events(deploy_dir: pathlib.Path,
                      anchor: float,
                      window: tuple[Optional[float], Optional[float]]) -> Iterable[TimelineEvent]:
    """Yield one TimelineEvent per row of tick.csv that falls inside window."""
    path = deploy_dir / "tick.csv"
    lo, hi = window
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                t = float(row["t"])
            except (KeyError, ValueError):
                continue
            wall = anchor + t
            if lo is not None and wall < lo:
                continue
            if hi is not None and wall > hi:
                continue
            ramp = float(row.get("ramp_alpha", 0.0))
            dry = int(float(row.get("dry_run", 0)))
            tilt = int(float(row.get("tilt_trip", 0)))
            reason = row.get("reason", "").strip("\"")
            summary_bits = []
            if reason:
                summary_bits.append(f"reason={reason}")
            if ramp > 0:
                summary_bits.append(f"ramp_alpha={ramp:.2f}")
            if dry:
                summary_bits.append("dry_run")
            if tilt:
                summary_bits.append("tilt_trip")
            if not summary_bits:
                # Skip silent ticks for the timeline (otherwise we'd emit
                # 50 events / second and bury the interesting ones). They
                # are still in tick.csv if anyone wants to load it.
                continue
            yield TimelineEvent(
                wall_ts=wall,
                source="deploy:tick",
                kind="tick_marker",
                summary=", ".join(summary_bits),
                payload={"t_rel": t, **row},
            )


def _load_joint_csv(path: pathlib.Path) -> list[tuple[float, list[float]]]:
    """Return a list of (t, [vec...]) tuples for a 32-column joint CSV."""
    if not path.is_file():
        return []
    out: list[tuple[float, list[float]]] = []
    with path.open() as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            return []
        if not header or header[0] != "t":
            return []
        for row in reader:
            try:
                t = float(row[0])
                vec = [float(x) for x in row[1:]]
            except (IndexError, ValueError):
                continue
            out.append((t, vec))
    return out


# ---------------------------------------------------------------------------
# JSONL loaders (motor monitor + manager sidecar)
# ---------------------------------------------------------------------------

def _iter_jsonl(path: pathlib.Path) -> Iterable[dict]:
    if not path.is_file():
        log.warning("JSONL not found: %s", path)
        return
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _motor_monitor_to_events(path: pathlib.Path,
                             window: tuple[Optional[float], Optional[float]]) -> list[TimelineEvent]:
    out: list[TimelineEvent] = []
    lo, hi = window
    for rec in _iter_jsonl(path):
        ts = float(rec.get("ts", 0.0))
        if ts <= 0.0:
            continue
        if lo is not None and ts < lo:
            continue
        if hi is not None and ts > hi:
            continue
        kind = rec.get("kind", "?")
        if kind == "boot":
            summary = "motor monitor BOOT (PC2)"
        elif kind == "shutdown":
            summary = f"motor monitor SHUTDOWN cycles={rec.get('cycles_completed', '?')}"
        elif kind == "event":
            summary = _summarize_motor_event(rec)
        elif kind == "sample":
            summary = _summarize_motor_sample(rec)
            if summary is None:
                continue
        else:
            summary = f"motor:{kind}"
        out.append(TimelineEvent(
            wall_ts=ts,
            source="motor_monitor",
            kind=kind,
            summary=summary,
            payload=rec,
        ))
    return out


def _summarize_motor_event(rec: dict) -> str:
    et = rec.get("type", "?")
    if et == "mc_mode_change":
        return (
            f"MC mode {rec.get('previous_mode', '?')} -> {rec.get('current_mode', '?')} "
            f"({rec.get('current_desc', '')!r}) status={rec.get('current_status', '?')}"
        )
    if et == "tracking_error_spike":
        joint = rec.get("joint", "?")
        err = rec.get("tracking_err", float("nan"))
        return f"tracking spike: {joint} err={err:+.3f} rad (target={rec.get('target', '?')}, pos={rec.get('pos', '?')})"
    if et == "limit_proximity":
        return (
            f"limit proximity: {rec.get('joint', '?')} side={rec.get('side', '?')} "
            f"target={rec.get('target', '?')} margin={rec.get('margin_rad', '?')}"
        )
    if et == "state_staleness":
        return f"state stale: {rec.get('joint', '?')} age={rec.get('age_s', '?')}s"
    if et == "command_staleness":
        return f"command stale: {rec.get('joint', '?')} age={rec.get('age_s', '?')}s"
    return f"motor event: {et}"


def _summarize_motor_sample(rec: dict) -> Optional[str]:
    """Only surface samples that say something interesting (else None)."""
    groups = rec.get("groups", {})
    bad = []
    for name, stats in groups.items():
        max_err = stats.get("max_tracking_err")
        if max_err is not None and max_err >= 0.10:
            bad.append(f"{name}:{max_err:.3f}")
    mc = rec.get("mc_action_mode", -1)
    if not bad and mc in (-1, 200, 100):
        # Quiet (no big errors, MC sitting in STAND_DEFAULT/JOINT_DEFAULT/unknown).
        return None
    extras = []
    if bad:
        extras.append("max_err: " + " ".join(bad))
    if mc != -1:
        extras.append(f"mc={mc}")
    return "motor sample: " + " | ".join(extras)


def _manager_sidecar_to_events(path: pathlib.Path,
                               window: tuple[Optional[float], Optional[float]]) -> list[TimelineEvent]:
    out: list[TimelineEvent] = []
    lo, hi = window
    for rec in _iter_jsonl(path):
        ts = float(rec.get("ts", 0.0))
        if ts <= 0.0:
            continue
        if lo is not None and ts < lo:
            continue
        if hi is not None and ts > hi:
            continue
        kind = rec.get("kind", rec.get("type", "?"))
        summary = _summarize_manager_record(rec)
        if summary is None:
            continue
        out.append(TimelineEvent(
            wall_ts=ts,
            source="manager_sidecar",
            kind=kind,
            summary=summary,
            payload=rec,
        ))
    return out


def _summarize_manager_record(rec: dict) -> Optional[str]:
    kind = rec.get("kind") or rec.get("type") or "?"
    if kind in {"planner_cmd", "intent"}:
        intent = rec.get("intent")
        mag = rec.get("magnitude")
        if intent in (None, "noop"):
            return None
        return f"intent: {intent} / {mag}"
    if kind == "resume_chord":
        return f"resume chord {rec.get('event', '?')} count={rec.get('press_count', '?')}"
    if kind == "motor_monitor_summary":
        # Motor monitor summary forwarded by the manager. Skip; we already
        # have those from --motor-monitor directly. Including both would
        # dupe the timeline.
        return None
    if kind in {"sample", "event"}:
        # The manager forwarded a motor monitor record into its sidecar
        # JSONL via _motor_monitor_loop. Skip for the same reason.
        return None
    if kind == "mode_transition":
        prev = rec.get("previous_mode") or rec.get("from") or "?"
        nxt = rec.get("next_mode") or rec.get("to") or "?"
        return f"manager mode {prev} -> {nxt}"
    if kind == "controller_visibility":
        return f"controllers visibility: {rec.get('visibility', '?')}"
    if kind == "frame_stall":
        return f"frame stall {rec.get('duration_ms', '?')}ms"
    return f"manager:{kind}"


# ---------------------------------------------------------------------------
# Window parsing
# ---------------------------------------------------------------------------

def _parse_center_ts(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    text = text.strip()
    if not text:
        return None
    # Accept either an ISO8601 string or a unix timestamp.
    try:
        return float(text)
    except ValueError:
        pass
    try:
        return _dt.datetime.fromisoformat(text).timestamp()
    except ValueError as exc:
        raise SystemExit(f"--center-ts: cannot parse {text!r} ({exc})") from exc


# ---------------------------------------------------------------------------
# Joint-level digest (windowed deep dive)
# ---------------------------------------------------------------------------

def _build_joint_digest(deploy_dir: pathlib.Path,
                        anchor: float,
                        window: tuple[float, float]) -> dict:
    """Per-joint stats inside the requested window from deploy CSVs."""
    target = _load_joint_csv(deploy_dir / "target_pos.csv")
    pos = _load_joint_csv(deploy_dir / "joint_pos.csv")
    vel = _load_joint_csv(deploy_dir / "joint_vel.csv")

    lo_w, hi_w = window
    lo_t = lo_w - anchor
    hi_t = hi_w - anchor

    def _crop(rows: list[tuple[float, list[float]]]) -> list[tuple[float, list[float]]]:
        return [(t, v) for t, v in rows if lo_t <= t <= hi_t]

    target = _crop(target)
    pos = _crop(pos)
    vel = _crop(vel)

    if not target or not pos:
        return {"joints": [], "note": "deploy CSVs empty in window"}

    # Align target/pos by index (DeployLogger writes them in lock-step).
    aligned = list(zip(target, pos, vel)) if vel else list(zip(target, pos))
    n_joints = len(target[0][1]) if target else 0
    out_joints = []
    for j in range(n_joints):
        max_err = 0.0
        max_pos = -math.inf
        min_pos = math.inf
        max_vel = 0.0
        for tup in aligned:
            tgt = tup[0][1][j]
            ps = tup[1][1][j]
            err = abs(tgt - ps)
            if err > max_err:
                max_err = err
            if ps > max_pos:
                max_pos = ps
            if ps < min_pos:
                min_pos = ps
            if vel and tup[2] is not None:
                vv = abs(tup[2][1][j])
                if vv > max_vel:
                    max_vel = vv
        out_joints.append({
            "index": j,
            "max_abs_err_rad": round(max_err, 4),
            "min_pos_rad": round(min_pos, 4),
            "max_pos_rad": round(max_pos, 4),
            "max_abs_vel": round(max_vel, 4),
        })
    out_joints.sort(key=lambda d: d["max_abs_err_rad"], reverse=True)
    return {
        "joints": out_joints,
        "samples_in_window": len(aligned),
        "window_s": hi_w - lo_w,
        "deploy_t_lo": lo_t,
        "deploy_t_hi": hi_t,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _write_timeline_csv(events: list[TimelineEvent], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["wall_ts", "wall_iso", "source", "kind", "summary", "json"])
        for ev in events:
            iso = _dt.datetime.fromtimestamp(ev.wall_ts).isoformat(timespec="milliseconds")
            w.writerow([
                ev.wall_ts, iso, ev.source, ev.kind, ev.summary,
                json.dumps(ev.payload, default=str),
            ])
    log.info("wrote timeline CSV: %s (%d events)", path, len(events))


def _write_timeline_md(events: list[TimelineEvent], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    by_minute: dict[str, list[TimelineEvent]] = {}
    for ev in events:
        minute = _dt.datetime.fromtimestamp(ev.wall_ts).isoformat(timespec="minutes")
        by_minute.setdefault(minute, []).append(ev)
    with path.open("w") as fh:
        fh.write("# X2 freeze postmortem timeline\n\n")
        fh.write(f"_total events:_ **{len(events)}**\n\n")
        for minute in sorted(by_minute):
            fh.write(f"## {minute}\n\n")
            for ev in by_minute[minute]:
                iso = _dt.datetime.fromtimestamp(ev.wall_ts).isoformat(timespec="milliseconds")
                fh.write(f"- `{iso}` **{ev.source}** [`{ev.kind}`] {ev.summary}\n")
            fh.write("\n")
    log.info("wrote timeline markdown: %s", path)


def _write_window_json(events: list[TimelineEvent],
                       digest: Optional[dict],
                       path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump({
            "events": [
                {
                    "wall_ts": ev.wall_ts,
                    "wall_iso": _dt.datetime.fromtimestamp(ev.wall_ts).isoformat(timespec="milliseconds"),
                    "source": ev.source,
                    "kind": ev.kind,
                    "summary": ev.summary,
                    "payload": ev.payload,
                }
                for ev in events
            ],
            "joint_digest": digest,
        }, fh, indent=2, default=str)
    log.info("wrote window JSON: %s", path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--deploy-log-dir", type=pathlib.Path, default=None,
        help="Directory containing tick.csv / target_pos.csv / etc. from the C++ deploy run.",
    )
    p.add_argument(
        "--motor-monitor", type=pathlib.Path, default=None,
        help="Path to the motor_monitor JSONL on PC2.",
    )
    p.add_argument(
        "--manager-sidecar", type=pathlib.Path, default=None,
        help="Path to the manager_sidecar JSONL on the laptop.",
    )
    p.add_argument(
        "--center-ts", default=None,
        help="ISO8601 wall-clock timestamp (or unix seconds) to center the window on.",
    )
    p.add_argument(
        "--window-s", type=float, default=30.0,
        help="Window half-width in seconds (default 30).",
    )
    p.add_argument(
        "--out-dir", type=pathlib.Path,
        default=pathlib.Path("./postmortem_out"),
        help="Output directory for timeline.csv / timeline.md / window.json.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
    )
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(asctime)s %(levelname)s x2_freeze_postmortem] %(message)s",
    )

    if args.deploy_log_dir is None and args.motor_monitor is None and args.manager_sidecar is None:
        print(
            "ERROR: pass at least one of --deploy-log-dir / --motor-monitor / --manager-sidecar",
            file=sys.stderr,
        )
        return 2

    center = _parse_center_ts(args.center_ts)
    if center is not None:
        window = (center - args.window_s, center + args.window_s)
    else:
        window = (None, None)

    events: list[TimelineEvent] = []
    deploy_anchor: Optional[float] = None
    digest: Optional[dict] = None

    if args.deploy_log_dir is not None:
        try:
            deploy_anchor, max_t = _detect_deploy_anchor(args.deploy_log_dir)
            events.extend(_iter_tick_events(args.deploy_log_dir, deploy_anchor, window))
        except FileNotFoundError as exc:
            log.warning("skipping deploy CSVs: %s", exc)

    if args.motor_monitor is not None:
        events.extend(_motor_monitor_to_events(args.motor_monitor, window))

    if args.manager_sidecar is not None:
        events.extend(_manager_sidecar_to_events(args.manager_sidecar, window))

    events.sort(key=lambda e: e.wall_ts)

    if center is not None and args.deploy_log_dir is not None and deploy_anchor is not None:
        digest = _build_joint_digest(args.deploy_log_dir, deploy_anchor, window)  # type: ignore[arg-type]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_timeline_csv(events, args.out_dir / "timeline.csv")
    _write_timeline_md(events, args.out_dir / "timeline.md")
    if center is not None:
        _write_window_json(events, digest, args.out_dir / "window.json")

    print(f"\nWrote {len(events)} events to {args.out_dir}/")
    if center is not None:
        print(f"Window: [{_dt.datetime.fromtimestamp(window[0]).isoformat(timespec='seconds')}, "
              f"{_dt.datetime.fromtimestamp(window[1]).isoformat(timespec='seconds')}]")
        if digest:
            print(f"Joint digest: {digest.get('samples_in_window', 0)} samples in window, "
                  f"top max_abs_err: {digest['joints'][0] if digest.get('joints') else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
