"""Filter BONES-SEED / X2 motion-lib keys to walk-and-turn locomotion only.

Scope: manipulation tasks that need the robot to walk around a room and
reach — not stylistic gaits, running, crawling, or object-interaction clips.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

# Keys must match at least one include pattern (case-insensitive).
DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = (
    r"(?i)\b(walk|stride|pace|gait)\b",
    r"(?i)\b(forward|backward|back_step|fwd_step)\b",
    r"(?i)\b(sideway|sideways|lateral|straf)\b",
    r"(?i)\b(turn|pivot|rotate)\b",
    r"(?i)\b(idle|stand|rest|stationary)\b",
    r"(?i)^loco__",
)

# Reject even if an include matched.
DEFAULT_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"(?i)\b(run|sprint|jog|dash)\b",
    r"(?i)\b(crawl|kneel|crouch|squat|sit|lying|prone)\b",
    r"(?i)\b(jump|hop|leap|vault|fall)\b",
    r"(?i)\b(box|punch|kick|dance|zombie|injured|stealth|happy|scared|gun)\b",
    r"(?i)\b(pick|place|grasp|carry|lift|throw|push|pull|open|close)\b",
    r"(?i)\b(climb|stair|ladder|handstand|cartwheel)\b",
    r"(?i)\b(manip|object|prop|tool|sword|bench|chair)\b",
    r"(?i)\b(stoop|bend|reach|wave|gesture|point)\b",
    r"(?i)standing__",  # standing-manipulation subset
)


def compile_patterns(patterns: Sequence[str]) -> list[re.Pattern[str]]:
    return [re.compile(p) for p in patterns]


def motion_key_passes_loco_filter(
    key: str,
    *,
    include_patterns: Sequence[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> bool:
    """Return True if ``key`` is kept for X2 walk/turn MotionBricks training."""
    inc = compile_patterns(include_patterns)
    exc = compile_patterns(exclude_patterns)
    if not any(p.search(key) for p in inc):
        return False
    if any(p.search(key) for p in exc):
        return False
    return True


def filter_motion_keys(
    keys: Iterable[str],
    *,
    include_patterns: Sequence[str] = DEFAULT_INCLUDE_PATTERNS,
    exclude_patterns: Sequence[str] = DEFAULT_EXCLUDE_PATTERNS,
) -> list[str]:
    return [
        k
        for k in keys
        if motion_key_passes_loco_filter(
            k,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )
    ]
