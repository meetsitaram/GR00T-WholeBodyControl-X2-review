#!/usr/bin/env python3
"""Replay a captured Quest3 session through the e-stop gesture detector.

Purpose (2026-08-03, after repeated dead-on-arrival VR e-stop tests):
the operator should never have to re-don the headset to debug a gesture.
Every stack session now records ``quest3_raw.jsonl`` (raw buttons +
trigger analogs per manager tick); this tool replays that stream through
the EXACT chord + EstopGesture path the manager runs and reports every
soft/damp firing — plus WHY nothing fired when nothing fired (chord
never held / triggers never pumped / input stream dead).

Usage:
    python3 gear_sonic/scripts/replay_quest3_estop.py \
        /tmp/x2_quest3_planner_stack-*/quest3_raw.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "utils" / "teleop"))
from estop_gesture import EstopGesture  # noqa: E402


def replay(path: Path) -> int:
    g = EstopGesture()
    rows = 0
    chord_ticks = 0
    max_lt = max_rt = 0.0
    lt_cycles = rt_cycles = 0
    lt_was = rt_was = False
    fires: list[tuple[float, int]] = []
    t0 = None
    max_pumps = 0
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        rows += 1
        t = float(row.get("t_mono", 0.0))
        if t0 is None:
            t0 = t
        btn = row.get("buttons") or {}
        a = bool(btn.get("a", False))
        x = bool(btn.get("x", False))
        lt = float(btn.get("leftTrigger") or 0.0)
        rt = float(btn.get("rightTrigger") or 0.0)
        chord = a and x
        chord_ticks += chord
        max_lt, max_rt = max(max_lt, lt), max(max_rt, rt)
        # raw cycle counting (independent of chord) for diagnostics
        if not lt_was and lt >= 0.6:
            lt_was = True
        elif lt_was and lt <= 0.3:
            lt_was = False
            lt_cycles += 1
        if not rt_was and rt >= 0.6:
            rt_was = True
        elif rt_was and rt <= 0.3:
            rt_was = False
            rt_cycles += 1
        ph = g.tick(lt, rt, chord, now=t)
        max_pumps = max(max_pumps, len(g._pump_win))
        if ph:
            fires.append((t - t0, ph))

    dur = 0.0 if t0 is None else t - t0
    print(f"rows={rows} span={dur:.1f}s chord_held_ticks={chord_ticks} "
          f"max_lt={max_lt:.2f} max_rt={max_rt:.2f} "
          f"raw_cycles L={lt_cycles} R={rt_cycles} max_pumps_1s={max_pumps}")
    for t_rel, ph in fires:
        print(f"  +{t_rel:7.2f}s  {'SOFT (idle stand)' if ph == 1 else 'DAMP (terminal)'}")
    if not fires:
        print("  NO e-stop fired. Why:")
        if rows == 0:
            print("   - capture file is empty (was the session recorded?)")
        elif chord_ticks == 0:
            print("   - A+X chord was NEVER held simultaneously in this capture")
        elif max_lt < 0.6 or max_rt < 0.6:
            print(f"   - trigger analogs never reached press threshold "
                  f"(max L={max_lt:.2f} R={max_rt:.2f}; need >=0.6): "
                  f"input link likely DEAD (controller detach/battery)")
        elif min(lt_cycles, rt_cycles) < 3:
            print(f"   - too few full press->release cycles "
                  f"(L={lt_cycles} R={rt_cycles}; a pump needs both)")
        else:
            print(f"   - cycles present but pumps never reached 3-in-1s "
                  f"(max {max_pumps}); pumping too slow or chord dropped "
                  f"between pumps")
    return 0 if fires else 1


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    if not path.exists():
        print(f"no such file: {path}")
        return 2
    return replay(path)


if __name__ == "__main__":
    raise SystemExit(main())
