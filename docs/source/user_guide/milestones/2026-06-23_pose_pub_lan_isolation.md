# 2026-06-23 — Pose PUB LAN isolation (sim runs can't reach PC2 anymore)

> **Session focus.** Plug a silent safety leak where sim-mode runs of
> [`run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh)
> and [`run_x2_replay_stack.sh`](../../../../gear_sonic/scripts/run_x2_replay_stack.sh)
> could unintentionally drive the real robot whenever PC2's
> `x2_pose_proxy` daemons were already running (which by design they
> always are). The fix is a one-variable gating change in both wrappers
> — `PUB_BIND_HOST` derives from `SIM_MODE` / `PC2_HOST` so sim binds
> loopback and real binds `*`. Restores the invariant: `--pc2-host` is
> the single source of truth for whether the laptop's pose wire reaches
> the real robot.

---

## TL;DR

| Symptom (before) | Cause | Fix |
|---|---|---|
| Running [`run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh) **without `--pc2-host`** spawned the local sim stack as expected — wrapper banner said "sim run artifacts", `docker stop` on the sim container, etc. — **and the physical robot moved too**. Same with [`run_x2_replay_stack.sh`](../../../../gear_sonic/scripts/run_x2_replay_stack.sh) sim runs. The operator's intent ("just sim, please") was ignored at the wire level. | The bridge / mux / replay all hardcoded `--pub-host '*'` (= bind on `0.0.0.0:5556`, all network interfaces) regardless of mode. PC2's `x2_pose_proxy` is a long-lived daemon — started once via `x2_pc2_daemons.sh start --laptop-host <LAPTOP_IP> ...` and intentionally kept alive across laptop sessions because the operator never SSHes in to bounce it. It SUBs `<LAPTOP_IP>:5556` over wifi (ZMQ SUBs connect out actively). So every sim run that bound `*:5556` was visible to PC2 — and the real robot executed the same wire the local sim container did. The wrapper's `SIM_MODE=1` / `PC2_HOST=""` state was internal bookkeeping for what to launch *locally*; it never propagated to the bridge's bind argument. | New `PUB_BIND_HOST` env var derived from `SIM_MODE` (runtime) / `PC2_HOST` (replay). Sim → `127.0.0.1`, real → `*`. Plumbed into the bridge's `--pub-host`, the mux's `--out-host`, the recorder's `--pub-host`, and the replay's `--pub-host` — every laptop-side PUB bind that could touch `LAPTOP_POSE_PORT`. `:=` assignment so an explicit `PUB_BIND_HOST=*` env override still wins for the rare cross-host sim-debug case (a colleague SUBing the laptop wire from another machine on the LAN). |
| No operator-visible signal of which bind decision was made — the launcher banner showed `PC2 host:` only on real runs, and never echoed the actual PUB bind. The 2026-06-23 incident was diagnosed by `sudo ss -tnp 'sport = :5556'`, not by reading the wrapper output. | The banner blocks predated the dual-mode bind concern. | New banner line in both wrappers: `Pose PUB bind  : <host>:<port>  (LAN-isolated: PC2 cannot SUB ...)` for sim and `(LAN-visible: PC2 pose proxy SUBs ... over wifi)` for real. Loud `WARNING:` if the bind doesn't match the mode (e.g., `PUB_BIND_HOST=*` in sim, or loopback in real). |
| No regression test, so the next refactor that touched `--pub-host` could silently re-introduce the leak. | The 2026-06-11 pose_mux_split test file pinned every spawn-side argv except this one (because at the time `*` was the only correct value). | New `tests/test_launcher_pose_pub_bind_isolation.py` (10 tests): pins the gating block, the `:=` form, every call site, the banner, and `bash -n` + `--help` paths on both wrappers. |

---

## The leak in pictures

ZMQ semantics: **PUB binds passively, SUB connects actively.** The
laptop's bridge / mux / replay just opens a socket on its bind host
and waits. Anyone with TCP routing to that socket can attach.

### Before (broken — sim run leaks to PC2)

```
   Laptop -- ran ``run_x2_vla_runtime.sh ... <no --pc2-host>``
   ┌───────────────────────────────────────────┐
   │ wrapper: SIM_MODE=1, PC2_HOST=""          │
   │   ⇒ spawn local sim docker (deploy_x2.sh) │
   │   ⇒ DEBUG_SUB_HOST=localhost              │
   │   ⇒ banner: "sim run artifacts ..."       │
   │                                           │
   │ bridge: --pub-host '*'    ← HARDCODED     │
   │   binds 0.0.0.0:5556  ← all interfaces    │
   └─────────────┬─────────────────────────────┘
                 │                ▲
   loopback ─────┘ (sim docker)   │ wifi
                                  │
   PC2 (always-on daemons)        │
   ┌──────────────────────────────┴────────────┐
   │ x2_pose_proxy SUB <LAPTOP_IP>:5556        │
   │   (started once with --laptop-host;       │
   │    keeps trying to connect forever)       │
   │                                           │
   │ deploy receives the wire ⇒ MOTORS MOVE    │
   └───────────────────────────────────────────┘
```

### After (gated — sim run can't reach PC2)

```
   Laptop -- ran ``run_x2_vla_runtime.sh ... <no --pc2-host>``
   ┌───────────────────────────────────────────┐
   │ wrapper: SIM_MODE=1, PC2_HOST=""          │
   │   ⇒ PUB_BIND_HOST = 127.0.0.1   ← gated   │
   │   ⇒ banner: "Pose PUB bind: 127.0.0.1:5556│
   │             (LAN-isolated: PC2 cannot SUB │
   │              even if x2_pose_proxy is up)"│
   │                                           │
   │ bridge: --pub-host "$PUB_BIND_HOST"       │
   │   binds 127.0.0.1:5556  ← loopback only   │
   └─────────────┬─────────────────────────────┘
                 │
   loopback ─────┘ (sim docker — only consumer)

   PC2 (always-on daemons)
   ┌───────────────────────────────────────────┐
   │ x2_pose_proxy keeps trying to connect to  │
   │ <LAPTOP_IP>:5556 ⇒ ECONNREFUSED forever   │
   │   (kernel rejects -- nothing bound on the │
   │    LAN-facing interface)                  │
   │                                           │
   │ deploy receives NO wire ⇒ idle_stand      │
   └───────────────────────────────────────────┘
```

### Real-mode bind (unchanged behavior; explicit intent)

```
   Laptop -- ran ``run_x2_vla_runtime.sh ... --pc2-host 192.168.86.32``
   ┌───────────────────────────────────────────┐
   │ wrapper: SIM_MODE=0, PC2_HOST=...         │
   │   ⇒ PUB_BIND_HOST = '*'         ← gated   │
   │   ⇒ banner: "Pose PUB bind: *:5556        │
   │             (LAN-visible: PC2 pose proxy  │
   │              SUBs over wifi)"             │
   │                                           │
   │ bridge: --pub-host "$PUB_BIND_HOST"       │
   │   binds 0.0.0.0:5556  ← all interfaces    │
   └─────────────┬─────────────────────────────┘
                 │                ▲
   loopback ─────┘                │ wifi
                                  │
   PC2 ──────────────────────────┘
     x2_pose_proxy connects, deploy executes wire ⇒ MOTORS MOVE
```

---

## Bind decision table

| Mode | `--pc2-host` | `SIM_MODE` | `PUB_BIND_HOST` (default) | Wire reaches PC2? | Wire reaches sim docker? |
|---|---|---|---|---|---|
| Sim, no takeover | absent | 1 | `127.0.0.1` | NO (LAN-isolated) | yes (loopback) |
| Sim, takeover (`--enable-takeover`) | absent | 1 | `127.0.0.1` | NO (LAN-isolated) | yes (loopback, via mux) |
| Real, no takeover | present | 0 | `*` | yes (LAN, via wifi) | n/a |
| Real, takeover (`--enable-takeover`) | present | 0 | `*` | yes (LAN, via wifi, via mux) | n/a |
| Override (any mode, `PUB_BIND_HOST=*` env) | either | either | `*` (env wins) | yes | yes |

The override row is the **only** way sim mode can publish on the LAN
after this milestone, and it requires an explicit env var that doesn't
exist in any documented runbook — so it's an opt-in for the rare
cross-host sim-debug case (e.g., a colleague SUBing the sim wire from
another machine on the LAN for side-by-side viz), never an accidental
leak.

---

## Files

### Modified

- [`gear_sonic/scripts/run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh) — new `PUB_BIND_HOST` derivation block right after `SIM_MODE` is resolved; three call sites (`--pub-host` for the bridge at the `BRIDGE_ARGS` block, `--out-host` for the mux at `spawn_pose_mux()`, `--pub-host` for the recorder at `spawn_recorder()`) now read `$PUB_BIND_HOST` instead of hardcoded `'*'`; banner blocks for both sim and real branches gained a `Pose PUB bind ...` line that echoes the resolved host:port plus a colored isolation/warning suffix.
- [`gear_sonic/scripts/run_x2_replay_stack.sh`](../../../../gear_sonic/scripts/run_x2_replay_stack.sh) — same shape, derivation gated on `PC2_HOST` directly (replay doesn't have a separate `SIM_MODE` var); the single `--pub-host` call site for `replay_x2_dataset` now reads `$PUB_BIND_HOST`; banner gained the same `pose PUB bind ...` line.
- [`docs/source/user_guide/milestones/2026-06-22_dataset_replay_v5_wire.md`](2026-06-22_dataset_replay_v5_wire.md) — corrected the "PUB binds locally on `*:5556` ... doesn't change the bind topology" paragraph, which is no longer true.

### Created

- [`tests/test_launcher_pose_pub_bind_isolation.py`](../../../../tests/test_launcher_pose_pub_bind_isolation.py) — 10 tests pinning the gating block, the `:=` form, every `--pub-host`/`--out-host` call site, the banner pattern, and `bash -n` + `--help` paths on both wrappers. The `:=` form is critical — it's what makes the explicit `PUB_BIND_HOST=*` env override win as a documented escape hatch, vs. the bash `?=` form which would refuse to be overridden once set.

---

## Operator workflow (unchanged from operator perspective)

```bash
# Sim run -- now physically can't reach the real robot, regardless of
# PC2's daemon state. Banner confirms `Pose PUB bind: 127.0.0.1:5556`.
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model /path/to/checkpoint \
    --motion-token-decoder /path/to/decoder.pt \
    --prompt "..."

# Real-robot run -- explicit --pc2-host opts into LAN-visible binding.
# Banner confirms `Pose PUB bind: *:5556`.
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --pc2-host 192.168.86.32 \
    --model /path/to/checkpoint \
    --motion-token-decoder /path/to/decoder.pt \
    --prompt "..."

# Replay -- same pattern.
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0
# vs
./gear_sonic/scripts/run_x2_replay_stack.sh \
    --dataset x2_reach_and_retract_v1 --episode 0 --pc2-host 192.168.86.32

# Cross-host sim debug (colleague SUBing from another machine on the LAN).
# Explicit opt-in only -- not in any runbook by default.
PUB_BIND_HOST='*' ./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model ... --motion-token-decoder ... --prompt "..."
# Banner WILL show a red WARNING in this case because the operator
# overrode the safe default.
```

---

## How to verify the leak is closed

```bash
# Run a sim launcher (any of the wrappers, no --pc2-host).
./gear_sonic/scripts/run_x2_replay_stack.sh --dataset ... --episode 0 &

# Inspect the PUB bind from another shell:
sudo ss -tnp 'sport = :5556'
# Expected after fix: only 127.0.0.1:* peers (sim docker / loopback).
# Before fix: would include 192.168.86.32:* (PC2 over wifi) if PC2
# daemons were up.

# Bonus -- on PC2, the pose proxy log should show ECONNREFUSED to
# <LAPTOP_IP>:5556 during the sim run (which is the correct response;
# the laptop is intentionally not advertising the wire on the LAN).
ssh ubuntu@192.168.86.32 'tail -F /tmp/x2_pc2_daemons-*/pose_proxy.log'
```

---

## What this milestone does **not** do

- It does not change PC2's `x2_pose_proxy` or any daemon behavior on the
  robot side. The proxy still SUBs `<LAPTOP_IP>:5556` and reconnects on
  any laptop run, exactly as before. The change is purely on the
  laptop-side bind.
- It does not affect the
  [2026-06-22 dataset-replay v5 wire fix](2026-06-22_dataset_replay_v5_wire.md).
  The two milestones are independent: 06-22 fixed *what* was on the
  wire; 06-23 fixed *who can see* the wire. Both are required for safe
  sim-vs-real iteration.
- It does not gate the recorder's `--pub-host` aggressively — the
  recorder skips binding in VLA subscribe-mode anyway (because the
  bridge owns the wire there), and the rare standalone-record-
  without-bridge case is still defensively routed through
  `$PUB_BIND_HOST`.
- It does not touch the `--out-host` of the
  [`run_x2_quest3_planner_stack.sh`](../../../../gear_sonic/scripts/run_x2_quest3_planner_stack.sh)
  flow. That launcher already has the right gating pattern (its
  `PLANNER_PUB_HOST` derivation at lines 2275–2282 is the template
  this milestone's fix borrows from). Other `--pub-host '*'` sites in
  that launcher are scoped to manager / recorder paths that don't bind
  `LAPTOP_POSE_PORT`; they're left as-is.

---

## Follow-ups

- Apply the same `PUB_BIND_HOST` pattern to the
  [`run_x2_quest3_planner_stack.sh`](../../../../gear_sonic/scripts/run_x2_quest3_planner_stack.sh)
  remaining `*` binds for symmetry (lines 2808, 3039, 3124). Low
  priority — those PUBs don't target `LAPTOP_POSE_PORT` so they don't
  cross the safety boundary, but consistency would make the gating
  pattern uniform across the launcher family.
- Add a preflight that probes `<PC2_HOST>:22` reachability in sim mode
  and warns (or refuses) when PC2 is on the LAN — defense in depth on
  top of this bind-level fix. Useful if a future refactor accidentally
  re-introduces the `*` bind; the preflight would catch it operationally
  even if the source-level pins regressed.
- Promote the `--pc2-host` decision (and the resulting `PUB_BIND_HOST`)
  to a single shared library function in `gear_sonic_deploy/scripts/`
  so future launchers can't drift from the gating pattern. Today every
  wrapper has its own copy of the derivation block.
