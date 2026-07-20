"""Parse the captured kplanner command log and align it to the recorded
--motion-key names (order-preserving DP, skipping unrecorded extra drives).

Input:  sample_planner_logs   (repo root)
Output: out/kplanner_gen_proof/alignment.json  + a printed review table.

The log has NO timestamps and NO qpos. It carries the replan/intent stream
(`Replanning with mode: {..}, target_vel, movement[x,y,z], facing[x,y,z]`),
segmented by `Emergency Stop! Movement momentum reset.` lines (= operator `R`).
The appended shell history lists the ordered `--motion-key` names.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG = REPO / "sample_planner_logs"

_REPLAN_RE = re.compile(
    r"Replanning with mode:\s*([A-Z_]+),\s*target_height:\s*(-?[\d.eE+-]+),\s*"
    r"target_vel:\s*(-?[\d.eE+-]+),\s*movement:\s*\[([^\]]*)\],\s*facing:\s*\[([^\]]*)\]"
)
_ESTOP = "Emergency Stop! Movement momentum reset."
_MOTIONKEY_RE = re.compile(r"--motion-key\s+(\S+)")


def _vec(s: str):
    return [float(x) for x in s.split(",")]


@dataclass
class Replan:
    mode: str
    target_vel: float
    movement: list
    facing: list


@dataclass
class Segment:
    idx: int
    replans: list = field(default_factory=list)   # collapsed (mode,tvel,mv,fc) holds
    n_raw: int = 0

    # derived signature
    mode: str = "IDLE"
    target_vel: float = -1.0
    is_turn: bool = False
    is_back: bool = False
    is_circle: bool = False
    facing_sweep_deg: float = 0.0
    turn_sign: int = 0   # +1 left (ccw), -1 right (cw)

    def sig(self):
        return dict(mode=self.mode, target_vel=self.target_vel,
                    turn=self.is_turn, back=self.is_back, circle=self.is_circle,
                    sweep_deg=round(self.facing_sweep_deg, 1), turn_sign=self.turn_sign,
                    n_replans=len(self.replans), n_raw=self.n_raw)


def parse_segments(text: str):
    segs, cur = [], []
    for line in text.splitlines():
        if _ESTOP in line:
            segs.append(cur)
            cur = []
            continue
        m = _REPLAN_RE.search(line)
        if m:
            mode, _th, tvel, mv, fc = m.groups()
            cur.append(Replan(mode, float(tvel), _vec(mv), _vec(fc)))
    if cur:
        segs.append(cur)

    out = []
    for i, replans in enumerate(segs):
        if not replans:
            continue
        # collapse consecutive duplicate replans
        collapsed = []
        for r in replans:
            key = (r.mode, round(r.target_vel, 4),
                   tuple(round(x, 4) for x in r.movement),
                   tuple(round(x, 4) for x in r.facing))
            if not collapsed or collapsed[-1][0] != key:
                collapsed.append([key, r])
            else:
                collapsed[-1][0] = key
        seg = Segment(idx=i, replans=[c[1] for c in collapsed], n_raw=len(replans))
        _derive_signature(seg)
        out.append(seg)
    return out


def _heading_angle(vec):
    x, y = vec[0], vec[1]
    if abs(x) < 1e-9 and abs(y) < 1e-9:
        return None
    return math.atan2(y, x)


def _derive_signature(seg: Segment):
    # active (non-IDLE) replans define the drive
    active = [r for r in seg.replans if r.mode != "IDLE"]
    if not active:
        seg.mode = "IDLE"
        return
    # dominant mode = most frequent non-idle mode
    modes = {}
    for r in active:
        modes[r.mode] = modes.get(r.mode, 0) + 1
    seg.mode = max(modes, key=modes.get)
    mode_rs = [r for r in active if r.mode == seg.mode]

    # target_vel: dominant positive value (ignore -1 sentinels)
    tvels = [r.target_vel for r in mode_rs if r.target_vel > 0]
    if tvels:
        # most common
        vc = {}
        for v in tvels:
            vc[round(v, 3)] = vc.get(round(v, 3), 0) + 1
        seg.target_vel = max(vc, key=vc.get)
    else:
        seg.target_vel = -1.0

    # back: movement opposite facing (dot < 0) in the active window
    dots = []
    for r in mode_rs:
        mv, fc = r.movement, r.facing
        n = math.hypot(mv[0], mv[1]) * math.hypot(fc[0], fc[1])
        if n > 1e-6:
            dots.append((mv[0]*fc[0]+mv[1]*fc[1]) / n)
    seg.is_back = bool(dots) and (sum(1 for d in dots if d < -0.3) > 0.5*len(dots))

    # turn/circle: sweep of facing angle across active window (unwrapped)
    angs = [a for a in (_heading_angle(r.facing) for r in mode_rs) if a is not None]
    if len(angs) >= 2:
        unwrapped = [angs[0]]
        for a in angs[1:]:
            prev = unwrapped[-1]
            d = ((a - prev + math.pi) % (2*math.pi)) - math.pi
            unwrapped.append(prev + d)
        total = unwrapped[-1] - unwrapped[0]
        seg.facing_sweep_deg = math.degrees(total)
        seg.turn_sign = 1 if total > 0 else (-1 if total < 0 else 0)
        sweep_abs = abs(seg.facing_sweep_deg)
        seg.is_turn = sweep_abs > 25.0
        seg.is_circle = sweep_abs > 200.0


# ------------------------- expected key signatures -------------------------

@dataclass
class KeySig:
    name: str
    mode: str
    target_vel: float
    turn: bool
    back: bool
    circle: bool


def parse_key(name: str) -> KeySig:
    n = name.lower()
    if n.startswith("slow_walk"):
        mode = "SLOW_WALK"
    elif n.startswith("walk"):
        mode = "WALK"
    elif n.startswith("run"):
        mode = "RUN"
    else:
        mode = "SLOW_WALK"
    m = re.search(r"(\d\.\d)", n)
    tvel = float(m.group(1)) if m else -1.0
    turn = ("turn" in n) or ("turns" in n) or ("circle" in n)
    circle = "circle" in n
    back = "back" in n
    return KeySig(name, mode, tvel, turn, back, circle)


def extract_keys(text: str):
    keys = []
    for line in text.splitlines():
        m = _MOTIONKEY_RE.search(line)
        if m:
            keys.append(m.group(1))
    return keys


# ------------------------- order-preserving DP alignment -------------------------

def match_score(k: KeySig, s: Segment) -> float:
    """Higher = better. -inf-ish for hard mode mismatch handled via penalty."""
    score = 0.0
    # mode
    if k.mode == s.mode:
        score += 3.0
    elif s.mode == "IDLE":
        score -= 2.0
    else:
        score -= 3.0
    # target_vel (exact anchor for slow_walk)
    if k.target_vel > 0 and s.target_vel > 0:
        if abs(k.target_vel - s.target_vel) < 0.001:
            score += 4.0
        else:
            score -= 2.0 * min(3.0, abs(k.target_vel - s.target_vel) / 0.1)
    # turn flag
    score += 1.5 if (k.turn == s.is_turn) else -1.5
    # back flag
    score += 1.5 if (k.back == s.is_back) else -1.5
    # circle
    if k.circle:
        score += 1.5 if s.is_circle else -0.5
    return score


def align(keys, segs, skip_cost=0.4):
    """Needleman-Wunsch: every key must map to a segment (in order); segments
    may be skipped (extras). Only segment-side gaps allowed."""
    K, S = len(keys), len(segs)
    NEG = -1e9
    dp = [[NEG]*(S+1) for _ in range(K+1)]
    bt = [[None]*(S+1) for _ in range(K+1)]
    dp[0][0] = 0.0
    for j in range(1, S+1):          # skip leading segments free-ish
        dp[0][j] = -skip_cost*j
        bt[0][j] = ("skip", 0, j-1)
    for i in range(1, K+1):
        for j in range(1, S+1):
            # match key i-1 with seg j-1
            best = NEG; b = None
            cand = dp[i-1][j-1] + match_score(parse_key(keys[i-1]), segs[j-1])
            if cand > best:
                best, b = cand, ("match", i-1, j-1)
            # skip segment j-1
            cand = dp[i][j-1] - skip_cost
            if cand > best:
                best, b = cand, ("skip", i, j-1)
            dp[i][j] = best; bt[i][j] = b
    # backtrack from best over last row (any number of trailing skips)
    j = max(range(S+1), key=lambda jj: dp[K][jj])
    i = K
    pairs = []   # (key_idx, seg_idx or None)
    while i > 0 or j > 0:
        step = bt[i][j]
        if step is None:
            break
        kind, ni, nj = step
        if kind == "match":
            pairs.append((ni, nj))
            i, j = ni, nj
        else:
            i, j = ni, nj
    pairs.reverse()
    return pairs, dp[K][j]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=LOG)
    ap.add_argument("--out", type=Path, default=REPO / "out/kplanner_gen_proof/alignment.json")
    args = ap.parse_args()

    text = args.log.read_text()
    segs = parse_segments(text)
    keys = extract_keys(text)
    # de-dup adjacent identical shell-history retries? keep raw order but the
    # 64 count is what we align. Keep as-is (retries are legit re-records).
    print(f"[parse] {len(segs)} non-empty segments, {len(keys)} motion-key invocations")

    pairs, total = align(keys, segs)
    matched_segidx = {sj for _, sj in pairs}

    # review table
    rows = []
    ki_to_seg = {ki: sj for ki, sj in pairs}
    print("\n=== ALIGNMENT REVIEW TABLE ===")
    print(f"{'#':>3} {'motion_key':<26} {'keysig(mode,vel,t,b,c)':<28} "
          f"{'->seg':>6} {'segsig(mode,vel,turn,back,sweep,nrep)':<44} {'score':>6} {'flag'}")
    for ki, name in enumerate(keys):
        ks = parse_key(name)
        ksstr = f"{ks.mode[:4]},{ks.target_vel},{int(ks.turn)}{int(ks.back)}{int(ks.circle)}"
        if ki in ki_to_seg:
            sj = ki_to_seg[ki]
            s = segs[sj]
            sc = match_score(ks, s)
            segstr = (f"{s.mode[:4]},{s.target_vel},t{int(s.is_turn)},b{int(s.is_back)},"
                      f"sw{s.facing_sweep_deg:+.0f},n{len(s.replans)}")
            flag = "" if sc >= 4.5 else ("AMBIG" if sc >= 0 else "WEAK")
            rows.append(dict(key_idx=ki, motion_key=name, key_sig=asdict(ks),
                             seg_idx=s.idx, seg_sig=s.sig(), score=round(sc, 2), flag=flag))
            print(f"{ki:>3} {name:<26} {ksstr:<28} {s.idx:>6} {segstr:<44} {sc:>6.1f} {flag}")
        else:
            rows.append(dict(key_idx=ki, motion_key=name, key_sig=asdict(ks),
                             seg_idx=None, seg_sig=None, score=None, flag="UNMATCHED"))
            print(f"{ki:>3} {name:<26} {ksstr:<28} {'--':>6} {'(no segment)':<44} {'--':>6} UNMATCHED")

    extras = [segs[j].idx for j in range(len(segs)) if j not in matched_segidx]
    print(f"\n[align] total_score={total:.1f}  matched={len(matched_segidx)}  "
          f"extras(skipped segments)={len(extras)}")
    n_weak = sum(1 for r in rows if r['flag'] in ('WEAK', 'AMBIG', 'UNMATCHED'))
    print(f"[align] clean={len(rows)-n_weak}  needs-review={n_weak}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        n_segments=len(segs), n_keys=len(keys),
        rows=rows, extras_seg_idx=extras,
        segments={s.idx: {"sig": s.sig(),
                          "replans": [asdict(r) for r in s.replans]} for s in segs},
    )
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[out] wrote {args.out}")


if __name__ == "__main__":
    main()
