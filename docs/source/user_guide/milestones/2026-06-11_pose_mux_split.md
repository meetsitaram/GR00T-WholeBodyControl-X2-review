# 2026-06-11 — Pose pipeline split (`x2_pose_mux` + `x2_pose_watchdog`)

> **Session focus.** Refactor the manual-takeover plumbing introduced
> in the [2026-06-10 manual takeover milestone](2026-06-10_vla_manual_takeover.md)
> by **splitting the single-process `x2_pose_proxy.py` into two
> purpose-built processes**: a laptop-side mux (`x2_pose_mux.py`) that
> owns dual-source arbitration / engagement ramp / `vla_control` edge
> events, and a slim PC2-side watchdog (`x2_pose_watchdog.py`, renamed
> from `x2_pose_proxy.py`) that owns only the staged fallback ladder
> from the [2026-06-08 milestone](2026-06-08_arm_freeze_on_upstream_stall.md).
> No behaviour changes for autonomous-only runs; manual-takeover gets
> moved off PC2 onto the laptop where it belongs.

---

## TL;DR

| Symptom (before) | Cause | Fix |
|---|---|---|
| Manual-takeover plumbing was tangled into PC2's safety-critical fallback ladder. A typo in the arbitration state machine could crash the watchdog and SAFE_IDLE the robot mid-session; a typo in the fallback ladder could leak into the merge logic and produce undefined behaviour on engagement. | The single `x2_pose_proxy.py` ran both concerns in one process tree. The fallback ladder is the LAST line of defense before the C++ deploy's `--disable-pose-ref-watchdog`; the arbitration code is the FIRST line of mutation. Putting them in one process violated the "stable vs. evolving" boundary. | Split the single process into two: laptop-side `x2_pose_mux` (merge + ramp + edges; mutates often) and PC2-side `x2_pose_watchdog` (fallback only; rarely changes). The PC2 watchdog has exactly one upstream SUB and one downstream PUB; it cannot crash on arbitration bugs because it doesn't see them. |
| Sim and real-robot deployments had different topologies for manual takeover: sim spawned `x2_pose_proxy` locally; real ran the same script on PC2 with a different env-var profile. This was a known maintenance burden — every takeover-related CLI flag had to be plumbed through two launchers with two slightly different defaults. | The original design put the proxy "next to the deploy", which forced a per-mode location decision. | The mux always runs on the laptop (sim and real). In sim, the launcher also spawns a local watchdog so the sim deploy gets the same fallback ladder the real deploy does. The two modes now share exactly one topology: `bridge → mux → watchdog → deploy`. |
| Operator override traffic crossed the wifi link between laptop (Quest 3 recorder) and PC2 (proxy). On a marginal link this added jitter to the engagement edge timing. | Recorder PUBed override pose on `laptop:5560`; PC2 proxy SUBed it over wifi. | Mux runs on the laptop. Operator pose stays on loopback (the recorder still PUBs on `:5560` but the mux is on `127.0.0.1:5560` now). Only the merged wire crosses wifi to PC2. |
| Adding a third pose source (e.g. a scripted gesture playback during VLA) required modifying PC2 code, retesting fallback, and re-staging via `pc2_bringup.sh`. | Arbitration logic lived on PC2. | Adding a third source is now a single SUB + arbiter entry on the laptop. PC2 stays out of the loop. |

---

## Architecture

```
LAPTOP                                           PC2
══════                                           ═══

bridge :5571 (internal)  ─┐
                          ├─► mux  ═══wifi═══►  watchdog  ──► deploy
recorder :5560 (override) ─┘   *:5556          *:5558      :5558
                              (LAPTOP_POSE_PORT)
                               │
                               ▼
                          vla_control PUB :5559
                          (loopback; bridge SUBs)
```

**Process roles:**

| Process | Host | Concern | Owns |
|---|---|---|---|
| `live_vla_publish_motion_token` (the bridge) | laptop | inference only | model loading, chunk decode, wire-shape filters, `vla_control` SUB for cold restart |
| `x2_pose_mux` | laptop | merge + ramp + edges | dual-source SUB, arbitration state machine (debounce / frozen / hysteresis / strict mode gate), engagement slow-step clamp, `vla_control` edge PUB |
| `x2_pose_watchdog` | PC2 (real) / laptop loopback (sim) | safety fallback only | upstream SUB, byte-verbatim forward when live, HOLD → BLEND → IDLE_CLIP staged ladder when upstream stalls, optional yaw rebase against deploy's `x2_debug` IMU quat |
| C++ deploy | PC2 (real) / laptop docker (sim) | SONIC tracker + safety stack | reads pose @ `:5558`, runs the policy, drives the motors |

**Sim mode** is identical except the watchdog runs on `127.0.0.1` next
to the mux instead of crossing wifi to PC2. The sim deploy reads pose
from `127.0.0.1:5558`.

**Autonomous-only mode** (no `--enable-takeover`): the mux is not
spawned. The bridge publishes directly to `*:5556`. The PC2 watchdog
(or, in sim, the bridge writing direct to `127.0.0.1:5556`) is the
only pose consumer. **Byte-for-byte unchanged from before the split.**

---

## What moved where

| Original code in `x2_pose_proxy.py` | New home | Notes |
|---|---|---|
| Wire-format constants, `x2_debug` decoder, pose field decoders, yaw extraction / rebase math, pose message packer | `gear_sonic/utils/pose_pipeline/wire.py` | Shared library; both mux and watchdog import from here. |
| `_clamp_vector_step_f32` per-element rate clamp | `gear_sonic/utils/pose_pipeline/clamp.py` | Same clamp the bridge uses (now imported, not duplicated). |
| `IdleStandReplay`, `build_idle_frame_msg`, `decide_fallback_state` | `gear_sonic/utils/pose_pipeline/fallback.py` | Used only by the watchdog. |
| Engagement / debounce / frozen-detect / hysteresis / strict-mode arbitration state machine | `gear_sonic/utils/pose_pipeline/arbitrate.py` (the `TakeoverArbiter` class) | Used only by the mux. |
| Main loop with dual SUB + fallback ladder + edge PUB | Split into two main loops: one in `gear_sonic/scripts/x2_pose_mux.py` (merge + edges), one in `gear_sonic_deploy/scripts/x2_pose_watchdog.py` (fallback only). | — |

---

## Topology details

### Bridge → mux port handshake

When `--enable-takeover` is set, the laptop launcher (`run_x2_vla_runtime.sh`)
flips the bridge's `--pub-port` from `LAPTOP_POSE_PORT` (5556) to
`BRIDGE_POSE_PORT_INTERNAL` (5571) so the mux can bind `*:5556` for
its merged output. Without `--enable-takeover`, the bridge publishes
directly on `*:5556` (legacy behaviour preserved).

> **2026-06-11 follow-up.** `BRIDGE_POSE_PORT_INTERNAL` was originally
> `5570`. We discovered that port was already owned by the kplanner
> stack's `x2_debug_to_robot_pose_bridge` (publishes `robot_pose`
> topic for the kplanner's pose-feedback). Running
> `run_x2_vla_runtime.sh --enable-takeover` alongside
> `run_x2_quest3_planner_stack.sh` produced a `zmq.error.ZMQError:
> Address already in use (addr='tcp://*:5570')` and the kplanner
> stack's bridge silently failed to start. Moved to `5571` (next
> free slot in the sonic port range); both endpoints are pure
> laptop loopback so there is no wire-level impact.

This matters because the canonical pose port external consumers
listen on is `LAPTOP_POSE_PORT=5556` — the PC2 watchdog, the recorder
in subscribe mode, and any future external SUB all SUB at `:5556`.
By having the mux take over that port, the rest of the system stays
oblivious to whether takeover is active.

### Mux loopback semantics

The mux is **always** on the laptop. Its SUBs are loopback:

```
mux --primary-host 127.0.0.1 --primary-port 5571  (bridge internal)
mux --override-host 127.0.0.1 --override-port 5560 (recorder loopback)
mux --teleop-mode-host 127.0.0.1 --teleop-mode-port 5564 (Quest3 manager mode)
```

Its PUB binds the external interface so PC2 can SUB over wifi:

```
mux --out-host '*' --out-port 5556 (LAPTOP_POSE_PORT)
mux --vla-control-bind-host 127.0.0.1 --vla-control-port 5559 (bridge SUBs)
```

Operator pose **never** crosses wifi as a separate stream. Only the
merged wire does.

### Watchdog narrowing

The watchdog dropped these flags entirely (the mux owns them now):

- `--override-host` / `--override-port` / `--override-topic` / `--override-stale-ms`
- `--override-frozen-ticks` / `--override-frozen-l2-tol` / `--override-engage-motion-ticks`
- `--engagement-max-wire-step` / `--engagement-steady-wire-step` / `--engagement-step-ramp-ticks`
- `--teleop-mode-host` / `--teleop-mode-port` / `--teleop-mode-topic` / `--teleop-mode-stale-ms`
- `--vla-control-bind-host` / `--vla-control-port` / `--vla-control-topic`

Passing any of these to the watchdog now triggers a **migration error**
with a pointer to this doc (instead of a confusing argparse "unrecognized
argument" message). The watchdog's surviving flag surface is:

```
--upstream-host / --upstream-port / --upstream-topic
--downstream-host / --downstream-port / --downstream-topic
--idle-x2m2 / --idle-stale-ms / --idle-mode
--hold-last-secs / --blend-secs
--x2-debug-host / --x2-debug-port / --x2-debug-topic
--no-x2-debug-yaw-track
```

Same `POSE_PROXY_*` env var names the original proxy used continue to
work (the launchers map them through to the new flag names).

---

## Operator workflow

### Autonomous-only (no change)

```bash
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model /path/to/hf_ckpt \
    --pc2-host 10.0.1.41 \
    --prompt "pick up the red cube"
```

The laptop runs only the bridge. PC2's `x2_pose_watchdog` SUBs the
bridge directly at `LAPTOP:5556`. No mux involved.

### Manual takeover (new)

Two terminals on the laptop:

```bash
# Terminal 1: VLA runtime + mux on the laptop, watchdog on PC2
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model /path/to/hf_ckpt \
    --pc2-host 10.0.1.41 \
    --prompt "pick up the red cube" \
    --enable-takeover \
    --vla-control-port 5559 \
    --pose-proxy-override-port 5560

# Terminal 2: Quest stack pointed at the mux's override port
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 0 \
    --remote-deploy 10.0.1.41 \
    --takeover
```

The `--takeover` shortcut sets `--pose-port 5560` so the recorder
PUBs operator pose into the mux's override SUB instead of straight
at the deploy.

### Sim mode with takeover

```bash
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --model /path/to/hf_ckpt \
    --prompt "pick up the red cube" \
    --enable-takeover \
    --vla-control-port 5559 \
    --pose-proxy-override-port 5560
# (no --pc2-host -> sim mode; launcher spawns mux + local watchdog +
#  sim deploy on loopback)

./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 0 \
    --no-deploy \
    --takeover
```

---

## Backward compatibility

| Surface | Old behaviour | New behaviour |
|---|---|---|
| `POSE_PROXY_*` env vars (operator runbooks) | Honored by `x2_pose_proxy.py` directly | Honored by `run_x2_vla_runtime.sh` and forwarded to `x2_pose_mux.py` 1-for-1 |
| `x2_pose_proxy.py` file on disk | Existed | **Deleted.** Any direct invocation will hit `No such file` |
| `x2_pc2_daemons.sh` env vars (`POSE_PROXY_OVERRIDE_*`, `POSE_PROXY_TELEOP_*`, `POSE_PROXY_CONTROL_*`) | Forwarded to PC2 proxy | **Hard error** with migration pointer (these moved to the laptop) |
| `--pose-port 5558` from old quest3_planner_stack runbooks | Pointed recorder at PC2 proxy override SUB | Still works but is non-canonical; new canonical is `--takeover` (= `--pose-port 5560`, mux loopback) |
| `--vla-control-host` defaulting to `PC2_HOST` in real-robot mode | Bridge SUBed `vla_control` from PC2 over wifi | Default is now `127.0.0.1` (mux is local). Existing `--vla-control-host <PC2_IP>` calls keep working but cross wifi for no reason. |
| Bridge `--pub-port` always `LAPTOP_POSE_PORT=5556` | Always 5556 | 5556 unless `--enable-takeover`, then `BRIDGE_POSE_PORT_INTERNAL=5571` (was 5570 in initial cut; moved to 5571 in 2026-06-11 follow-up to avoid colliding with the kplanner stack's `x2_debug_to_robot_pose_bridge` on :5570). Operator never sees this unless they're debugging the wire directly. |

---

## Tests added / moved

| Test | Status |
|---|---|
| `tests/test_pose_pipeline_clamp.py` | NEW. Unit tests for the per-element rate clamps in the shared library. |
| `tests/test_pose_pipeline_arbitrate.py` | NEW. Unit tests for the `TakeoverArbiter` state machine. |
| `tests/test_pose_pipeline_wire_yaw.py` | RENAMED from `tests/test_x2_pose_proxy_yaw_rebase.py`. Same tests, imports from `pose_pipeline.wire`. |
| `tests/test_x2_pose_watchdog_fallback_ladder.py` | RENAMED from `tests/test_x2_pose_proxy_fallback_ladder.py`. Same tests, exercises the new watchdog script. |
| `tests/test_x2_pose_watchdog_smoke.py` | RENAMED from `tests/test_x2_pose_proxy_smoke.py`. Same smoke, targets the new watchdog path. |
| `tests/test_x2_pose_mux_dual_source.py` | RENAMED + REWRITTEN from `tests/test_x2_pose_proxy_dual_source.py`. CLI changed from `--upstream/--downstream` to `--primary/--out`; removed all fallback-related setup. |
| `tests/test_run_x2_vla_runtime_sim_proxy.py` | REWRITTEN. Pins the new mux + watchdog spawn argv against their argparse parsers, the bridge `--pub-port` flip on `--enable-takeover`, and the sim deploy reading from the watchdog. |

---

## Files

**Created:**

- `gear_sonic/utils/pose_pipeline/__init__.py`
- `gear_sonic/utils/pose_pipeline/wire.py`
- `gear_sonic/utils/pose_pipeline/clamp.py`
- `gear_sonic/utils/pose_pipeline/fallback.py`
- `gear_sonic/utils/pose_pipeline/arbitrate.py`
- `gear_sonic/scripts/x2_pose_mux.py`
- `gear_sonic_deploy/scripts/x2_pose_watchdog.py`
- `tests/test_pose_pipeline_clamp.py`
- `tests/test_pose_pipeline_arbitrate.py`

**Deleted:**

- `gear_sonic_deploy/scripts/x2_pose_proxy.py`

**Renamed:**

- `tests/test_x2_pose_proxy_*` → `tests/test_x2_pose_watchdog_*` (fallback + smoke) and `tests/test_x2_pose_mux_*` (dual-source) and `tests/test_pose_pipeline_*` (wire).

**Modified:**

- `gear_sonic/scripts/run_x2_vla_runtime.sh` — `--enable-takeover` master switch; spawn `x2_pose_mux` (sim + real) + `x2_pose_watchdog` (sim only); bridge `--pub-port` flip; banner; preflight port set; stop ordering.
- `gear_sonic_deploy/scripts/x2_pc2_daemons.sh` — spawn `x2_pose_watchdog` instead of `x2_pose_proxy`; hard-fail on legacy `POSE_PROXY_OVERRIDE_*` / `POSE_PROXY_TELEOP_*` / `POSE_PROXY_CONTROL_*` env vars with migration pointer.
- `gear_sonic/scripts/run_x2_quest3_planner_stack.sh` — `--takeover` convenience flag (= `--pose-port 5560`) plus a prominent log line warning the operator that the mux must also be running.
- `gear_sonic/scripts/live_vla_publish_motion_token.py` (the bridge) — unchanged in this milestone; the `vla_control` SUB it added in the 2026-06-10 milestone still works as-is.

---

## What this milestone does **not** do

- It does not change any wire-shaping numbers (LPF cutoffs, ramp ticks, step caps, etc.). All defaults from the 2026-06-09 / 2026-06-10 milestones carry through unchanged.
- It does not change the `vla_control` event protocol or the bridge's cold-restart behaviour.
- It does not change the fallback ladder (HOLD → BLEND → IDLE_CLIP). The watchdog runs the exact same ladder the old proxy did, with the exact same defaults.
- It does not change how the operator engages / releases override (still motion-hysteresis or strict mode gate, whichever they configured).
- It does not change the C++ deploy in any way.

---

## Follow-ups

- Once the dust settles, consider letting the mux **register** new sources at runtime (e.g. a transient `play_gesture` script PUBs override pose, then disconnects) so gesture playback during VLA doesn't require any laptop-side ZMQ wiring beyond the gesture's own PUB.
- Add a `--mux-status` flag that prints arbitration state (current source, debounce countdown, engagement clamp progress) so the operator can introspect "why isn't override engaging?" without reading mux logs.
- Move the bridge's `--vla-control-host` default discussion (now always 127.0.0.1) into the `x2_vla_runtime` reference doc.
