#!/usr/bin/env python3
"""N3 language router v0 — "go to the cooking range" -> goal pose.

Zero-training System-2: resolves a natural-language target against the
waypoint registry and emits the exact goal payload the nav policy consumes
(goal xy + yaw + radius, kitchen frame). Unknown targets are rejected
explicitly (never guess a goal).

Usage:
    python nav_language_router.py "head over to the fridge"
    python nav_language_router.py --self-test
"""
import argparse
import json
import re
import sys

WAYPOINTS_JSON = "/home/stickbot/projects/x2-kitchen-sim/configs/waypoints.json"

# synonyms per registry waypoint — extend freely; VLMaps replaces this at N3.5
ALIASES = {
    "cooking_range": ["cooking range", "range", "stove", "cooktop", "burner",
                      "oven"],
    "dining_table": ["dining table", "table", "dinner table"],
    "dishwasher": ["dishwasher", "dish washer"],
    "entrance": ["entrance", "door", "front door", "doorway", "exit"],
    "fridge": ["fridge", "refrigerator", "freezer"],
    "hallway": ["hallway", "hall", "corridor"],
    "pantry": ["pantry", "food closet"],
    "sink": ["sink", "faucet", "tap", "basin"],
}


def route(text: str, registry: dict) -> dict | None:
    """Return {name, xy, yaw, radius} or None if no confident match."""
    t = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    t = f" {' '.join(t.split())} "
    best, best_len = None, 0
    for name, words in ALIASES.items():
        if name not in registry:
            continue
        for w in words:
            if f" {w} " in t and len(w) > best_len:
                best, best_len = name, len(w)
    if best is None:
        return None
    wp = registry[best]
    return {"name": best, "xy": wp["xy"], "yaw": wp["yaw"],
            "radius": wp["radius"]}


SELF_TEST = [
    ("go to the cooking range", "cooking_range"),
    ("head over to the fridge", "fridge"),
    ("walk to the sink please", "sink"),
    ("can you go to the dishwasher", "dishwasher"),
    ("move to the entrance", "entrance"),
    ("go stand by the dining table", "dining_table"),
    ("go to the pantry", "pantry"),
    ("into the hallway", "hallway"),
    ("go to the stove", "cooking_range"),
    ("walk to the refrigerator", "fridge"),
    # explicit rejects
    ("go to the bathroom", None),
    ("fly to the moon", None),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--waypoints", default=WAYPOINTS_JSON)
    args = ap.parse_args()
    registry = json.load(open(args.waypoints))

    if args.self_test:
        ok = 0
        for text, want in SELF_TEST:
            got = route(text, registry)
            name = got["name"] if got else None
            mark = "PASS" if name == want else "FAIL"
            ok += name == want
            print(f"[router] {mark}  {text!r} -> {name}")
        print(f"[router] VERDICT: {'PASS' if ok == len(SELF_TEST) else 'FAIL'}"
              f" ({ok}/{len(SELF_TEST)})")
        return 0 if ok == len(SELF_TEST) else 1

    text = " ".join(args.text)
    goal = route(text, registry)
    if goal is None:
        print(json.dumps({"error": "no waypoint matched", "text": text}))
        return 1
    print(json.dumps(goal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
