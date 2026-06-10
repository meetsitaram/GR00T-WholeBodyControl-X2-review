# 2026-06-10 — Manual takeover during VLA inference (no restart)

> **Session focus.** Give the operator a way to nudge the X2's arms out
> of a stuck VLA pose using VR teleop **without** restarting the bridge
> or any other process, and without modifying the Quest 3 manager (the
> VR controller code stays unaware of the override). The mechanism is
> dual-source arbitration in the PC2 pose proxy plus a one-shot edge
> event that drives a cold-restart in the VLA bridge. Builds on the
> [2026-06-08 fallback ladder milestone](2026-06-08_arm_freeze_on_upstream_stall.md).

---

## TL;DR

| Symptom (before) | Cause | Fix |
|---|---|---|
| VLA gets stuck (e.g. arm hovers forward-up over the soda can but won't descend). Only recovery is Ctrl-C and restart the runtime — model + cameras re-init takes ~30 s, and the operator loses all session state. | The bridge's pose PUB is the only upstream the proxy listens to; the operator has no way to inject a competing reference without taking the bridge down. The bridge has no notion of an external takeover signal, so even if you could inject one, the next decoded chunk would slam the arm back from the operator's hand-off pose to a chunk decoded against pre-takeover observations. | Two-part change:<br/>1) Proxy gains an optional second SUB (`--override-port`) plus an edge-triggered control PUB (`--vla-control-port`). It arbitrates frame-by-frame: override frames win whenever fresh; primary frames fill in between override ticks; the existing fallback ladder kicks in only when both go silent.<br/>2) Bridge gains a `vla_control` SUB. On `override_engaged` it stops shipping decoded chunks; on `override_released` it cold-restarts (clears ramp / LPF / chunk-blend state, pins a chunk-id baseline so any in-flight pre-override chunk is discarded, hold-publishes the operator's measured pose for `--vla-cold-restart-hold-ticks`). The next freshly decoded chunk ramps in from the operator's hand-off pose. |
| Quest 3 manager would need a new "intercept VLA" mode and the proxy a new "is-teleop-active?" sidechannel for the operator to be able to engage. | The naive design would require coupling the VR controller code to the VLA control plane. | The Quest 3 manager is **not modified**. The dual-source design just observes ZMQ wire activity on the override port — whatever process publishes there is "the operator", regardless of whether it came from VR, a script, a scripted demo, or a unit test. Engagement is implicit: start publishing, you win. Stop publishing for `--override-stale-ms`, you lose. |
| Bridge cold-restart would either (a) snap from the operator's pose to the bridge's idle stand reference, or (b) require the bridge to be aware of the operator's pose. | No common rebase point. | The bridge consumes the released edge and ships the **live measured pose** (read from `x2_debug`) as its wire for `--vla-cold-restart-hold-ticks` ticks (default 25 = 500 ms). The proxy's HOLD ladder (from the 2026-06-08 milestone) bridges any residual gap. The next decoded chunk ramps in from this measured pose. No coupling to the operator process. |

---

## Architecture

```{mermaid}
flowchart LR
  subgraph Laptop
    bridge[live_vla_publish_motion_token<br/>bridge<br/>--vla-control-port 5559]
    quest[quest3_manager_x2<br/><i>unchanged</i>]
    planner[x2_kplanner / heuristic]
    recorder[record_x2_dataset.py<br/>--pub-port 5560]
  end
  subgraph PC2
    proxy[x2_pose_proxy<br/>--override-port 5560<br/>--vla-control-port 5559]
    deploy[C++ deploy]
  end
  bridge -- "tcp://laptop:5556 (primary)" --> proxy
  recorder -- "tcp://laptop:5560 (override)" --> proxy
  quest -- "intent" --> planner
  planner -- "body_pose :5565" --> recorder
  quest -- "arm/hands :5564" --> recorder
  proxy -- "tcp://localhost:5558 (pose)" --> deploy
  proxy -- "tcp://0.0.0.0:5559 (vla_control)" --> bridge
```

**Engagement timeline** (tick = 20 ms @ 50 Hz):

| Tick | Wire on `:5558` | Bridge behaviour | Proxy state |
|---|---|---|---|
| -10 | bridge frame | shipping decoded chunks | LIVE |
| 0 | operator frame | first override frame seen — proxy fires `override_engaged` | OVERRIDE |
| 1 | operator frame | bridge SUB receives event; **stops** sending decoded chunks; ships measured pose | OVERRIDE |
| 2..N | operator frame | bridge ships measured pose (ignored by proxy) | OVERRIDE |
| N+1 | <silence> | operator releases (stops moving) | OVERRIDE (within debounce) |
| N+10 | <silence> | proxy crosses `--override-stale-ms`; fires `override_released` | LIVE / HOLD |
| N+11 | bridge frame | bridge consumes released edge: clears all smoothing state, pins chunk-id baseline, starts hold-publishing **measured pose** for `--vla-cold-restart-hold-ticks` | LIVE |
| N+11..N+35 | bridge frame (= measured pose) | hold window | LIVE |
| N+36 | bridge frame (= ramped decoded) | first decoded chunk after baseline ramps in from measured pose | LIVE |

---

## What landed

### Primary edits

- [`gear_sonic_deploy/scripts/x2_pose_proxy.py`](../../../../gear_sonic_deploy/scripts/x2_pose_proxy.py): added `--override-{host,port,topic,stale-ms}` (second SUB) and `--vla-control-{bind-host,port,topic}` (edge-event PUB). New `STATE_OVERRIDE`. Main loop drains override BEFORE primary, prefers override whenever fresh, caches override frames into the existing `last_upstream_msg` slot so the HOLD fallback replays the operator's pose (not the stale pre-override bridge frame) on the post-override silence window. Engage and release edges emit one-shot JSON events on the control PUB. All new sockets close cleanly on SIGINT; final stats line reports `override_fwd`, `override_engaged`, `override_released` counts. Backwards-compatible: when `--override-port` is unset (default `-1`), the proxy is byte-for-byte identical to the 2026-06-08 single-source proxy. **2026-06-10 follow-up**: added `--override-frozen-ticks` (default 10) + `--override-frozen-l2-tol` (default 5e-3 ~ 0.3°; **bumped from 1e-4** after sim repros showed sub-degree controller-rest jitter tripped repeated single-frame engage/release cycles) for frame-equality release detection. The Quest3 manager publishes the FROZEN last commanded pose every tick when teleop mode is OFF or LOCOMOTION (manager lines 1221-1229), so the override SUB never goes silent across an A+B+X+Y disengage gesture and the original silence-only release only fired on full Ctrl-C teardown. Frame-equality counts consecutive override frames within tolerance and latches a `override_frozen_detected` flag once the streak crosses the threshold; the latched flag forces `override_fresh = False` through the SAME edge handler as silence-based release (zero downstream-consumer impact). The latch clears the moment a non-frozen frame arrives, so re-engagement is automatic when the operator drives again. Status line gains `frozen(det=… streak=N/M rel=K)` telemetry; one-shot log line `override frozen detected (streak=N >= threshold=M, L2=… <= tol=…)` fires on the latch transition for postmortem clarity. **2026-06-10 (afternoon) follow-up**: added symmetric engage-side hysteresis via `--override-engage-motion-ticks` (default 10 = 200 ms @ 50 Hz, mirrors `--override-frozen-ticks`). Without it, sim repros showed a single one-tick controller jiggle (operator resting hand on the Quest 3 controller in OFF mode) triggered `engage -> 10-tick-frozen-release` every second, each cycle firing a heavy VLA cold-restart. The new counter (`override_motion_count`) increments on each non-frozen frame and resets on each frozen frame; engage now requires `override_motion_count >= engage_motion_threshold` in addition to the existing fresh + not-frozen-latched conditions. Status line gains `moving(streak=N/M sustained=BOOL)` telemetry. Same follow-up also added `release_pose` to the `override_released` event payload: the proxy snapshots the operator's last commanded body (`joint_pos_mj`, 31-D) and hand joints (`left_hand_joints`, `right_hand_joints`, both 10-D) from every override frame and embeds them in the released event JSON, so the bridge can hold the wire at the operator's exact commanded pose during cold-restart instead of x2_debug's lagged measured pose. Each field is optional in the payload; missing fields fall back to the legacy measured-pose hold on the bridge side. New helpers `decode_pose_left_hand` / `decode_pose_right_hand` / `_decode_pose_field_f32` make the body decoder generic to any named f32 field.
- [`gear_sonic/scripts/live_vla_publish_motion_token.py`](../../../../gear_sonic/scripts/live_vla_publish_motion_token.py): added `_VlaControlSignal` (thread-safe shared state) and `_run_vla_control_sub` (poll-with-timeout SUB worker, runs in its own daemon thread). New CLI args `--vla-control-{host,port,topic}` + `--vla-cold-restart-hold-ticks`. Publisher loop now consumes the signal at the top of each tick: when override is active, replaces the idle-clip pose with the live measured pose AND broadcasts it across the future window so the wire stays stationary; when a cold restart is pending, clears `ramp_from / ramp_remaining / lpf_state / lpf_future_state / hand_lpf_{left,right} / chunk_blend_{from,remaining} / hand_blend_from_{left,right} / hand_chunk_blend_remaining / decoded_was_active / prev_wire_{jpos,left,right} / last_wire_{chunk,hand_chunk}_id`, pins `cold_restart_chunk_baseline = chunk_id` so any in-flight pre-override chunk is rejected, and arms a `hold_at_measured_remaining` countdown. The existing decode gate gains `chunk_id > cold_restart_chunk_baseline and not in_hold_window` conditions, so the policy can never re-engage until a fresh post-restart chunk arrives. Backwards-compatible: when `--vla-control-port` is unset (default `-1`), the signal stays None and every new code path is a no-op. **2026-06-10 (afternoon) follow-up**: `_VlaControlSignal.release()` now accepts an optional `release_pose: dict[str, np.ndarray]` argument; `consume_cold_restart()` signature changed to `(pending, release_pose)` tuple so the bridge atomically reads both. SUB worker parses the new `release_pose` field from the `override_released` event JSON (tolerant of missing keys and bad dtypes/shapes) and forwards to `signal.release(release_pose=…)`. Publisher loop: on `consume_cold_restart()` returning a release pose, seeds `prev_wire_{jpos,left,right}` from that snapshot (instead of resetting to None) so the post-hold chunk-blend ramps cleanly FROM operator pose; replaces `cur_jpos = measured` with `cur_jpos = operator_pose["joint_pos_mj"]` in the hold-window block; overrides `left_step` / `right_step` to the operator's hand snapshot during hold (otherwise the existing hand chunk-blend would still ramp into the in-flight VLA chunk's fingers, which is what produced the "only VLA controls fingers" symptom). All operator-pose paths fall back per-field to the legacy measured-pose hold when the proxy ships no payload (older proxy / smoke tests with `--override-engage-motion-ticks 0`).
- [`gear_sonic_deploy/scripts/x2_pc2_daemons.sh`](../../../../gear_sonic_deploy/scripts/x2_pc2_daemons.sh): added `POSE_PROXY_OVERRIDE_{HOST,PORT,TOPIC,STALE_MS}` + `POSE_PROXY_CONTROL_{BIND,PORT,TOPIC}` env vars that gate the new proxy flags. Defaults `OVERRIDE_PORT=-1` and `CONTROL_PORT=-1` keep the existing PC2 bringup byte-for-byte unchanged; operators opt in by exporting positive ports before `start`. **2026-06-10 follow-ups**: added `POSE_PROXY_OVERRIDE_FROZEN_{TICKS,L2_TOL}` (defaults 10 and 5e-3, both forwarded to the proxy's `--override-frozen-{ticks,l2-tol}` when override is enabled) and `POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS` (default 10, forwarded to `--override-engage-motion-ticks`). The status log line at proxy spawn time now also includes the new hysteresis values for postmortem clarity.
- [`gear_sonic/scripts/run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh): added `VLA_CONTROL_{HOST,PORT,TOPIC}` + `VLA_COLD_RESTART_HOLD_TICKS` env vars. `VLA_CONTROL_HOST` defaults to `PC2_HOST` if unset (works for the standard split topology with no extra env vars). **Sim path extension**: when `SIM_MODE=1` AND (`VLA_CONTROL_PORT > 0` OR `POSE_PROXY_OVERRIDE_PORT > 0`), spawns a local `x2_pose_proxy.py` on loopback (between bridge :5556 and sim deploy :5558) via new `spawn_sim_proxy` helper. New env vars `POSE_PROXY_{DOWNSTREAM_HOST,DOWNSTREAM_PORT,OVERRIDE_HOST,OVERRIDE_PORT,OVERRIDE_TOPIC,OVERRIDE_STALE_MS,IDLE_X2M2,IDLE_STALE_MS,IDLE_MODE,HOLD_LAST_SECS,BLEND_SECS}` mirror the daemon defaults so the same operator runbook works in sim and real-robot. `stop_all` tears the proxy down after the deploy + bridge so loopback ports come back cleanly. `kill_stale_sim_processes` extended to `pkill` the proxy and `fuser -k` the downstream / override / control ports if anything leaks. Default `SIM_PROXY_ENABLED=0` keeps legacy autonomous-only sim runs byte-for-byte unchanged. **2026-06-10 follow-ups**: env-var + CLI passthrough for `POSE_PROXY_OVERRIDE_FROZEN_{TICKS,L2_TOL}` (new defaults 10 and **5e-3 bumped from 1e-4**), `POSE_PROXY_OVERRIDE_ENGAGE_MOTION_TICKS` (new, default 10); `--pose-proxy-override-engage-motion-ticks` CLI flag with --help documentation matching the production semantics.
- [`gear_sonic/scripts/run_x2_quest3_planner_stack.sh`](../../../../gear_sonic/scripts/run_x2_quest3_planner_stack.sh): added `--pose-port` flag to override the recorder's hardcoded `--pub-port 5556`. Numeric guard rejects non-integer values. Defaults to 5556 (legacy single-source teleop wiring); set to 5560 (or similar) to publish the operator's wire to the proxy's override port.

### Tests

- [`tests/test_x2_pose_proxy_dual_source.py`](../../../../tests/test_x2_pose_proxy_dual_source.py) (NEW): five end-to-end subprocess smokes. `test_proxy_dual_source_override_takes_priority_and_emits_events` spawns the proxy with all three sockets wired (primary + override + control), verifies the override frames take priority on the wire, the engage event fires on first override frame, and the release event fires after the debounce window. `test_proxy_dual_source_disabled_by_default` verifies the proxy does NOT bind the control port when `--vla-control-port` is unset. `test_proxy_override_frozen_release_after_manager_freeze` exercises the manager-freeze release path end-to-end (engage on varying frames → release on constant frames → re-engage on varying again). All gated on `X2_POSE_PROXY_SMOKE=1`. **2026-06-10 (afternoon) follow-ups**: two new smokes. `test_proxy_override_engage_hysteresis_blocks_single_frame_flicker` sends 1 motion frame in a sea of frozen frames and asserts zero engage / release events — pins the symmetric hysteresis fix. `test_proxy_override_released_payload_carries_operator_pose` drives a 0.6 s motion + 0.6 s freeze sequence with distinct body / left_hand / right_hand values, captures the `override_released` event, and asserts all three field arrays are present in the new `release_pose` payload with the expected values. The original silence-based test now passes `--override-engage-motion-ticks 0` to preserve its single-frame-engage contract (its constant-value frames would never accumulate motion under the new default).
- [`tests/test_vla_control_signal.py`](../../../../tests/test_vla_control_signal.py) (NEW + extended): ten pure-Python tests pinning the `_VlaControlSignal` engage / release / consume / threading semantics. Hammer test with 6000 concurrent updates across 3 threads confirms the lock-protected state stays consistent. **2026-06-10 (afternoon) follow-ups**: `consume_cold_restart()` signature change (`bool` → `(bool, release_pose)` tuple) propagated through all existing tests. Three new tests cover the operator-pose handoff: `test_release_pose_is_stored_and_returned_by_consume` (round-trip body + hand arrays), `test_release_pose_clears_on_consume_so_next_release_starts_fresh` (one-shot semantics — a follow-up release without payload must not replay the previous snapshot), and `test_release_pose_partial_fields_are_preserved` (per-field optional — body-only payload returns body-only dict, not synthesized hands).
- [`tests/test_run_x2_vla_runtime_sim_proxy.py`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) (NEW): six fast tests pinning the sim launcher's pose-proxy plumbing. `test_launcher_bash_syntax_ok` runs `bash -n` on `run_x2_vla_runtime.sh`. `test_launcher_help_lists_manual_takeover_flags` and `test_launcher_accepts_manual_takeover_cli_flags` pin the visible CLI surface (now including `--pose-proxy-override-engage-motion-ticks`). `test_spawn_sim_proxy_argv_parses` (3 parametrized cases) reconstructs the launcher's `spawn_sim_proxy` argv and feeds it to `x2_pose_proxy.main`'s argparse for arg-name drift detection.

### Docs

- [`pick_place_commands.md`](../../../pick_place_commands.md): added "Manual takeover during VLA" section with the operator workflow, env-var setup for the PC2 daemons + VLA runtime, the new `--pose-port` flag on the teleop launcher, what to look for in the proxy and bridge logs, and how to disable the new code path with a single env-var unset. Includes a "Sim-only variant" subsection with the two-terminal recipe for running the same takeover loop on the laptop without PC2 or the real robot (sim launcher spawns the proxy on loopback). **2026-06-10 follow-up**: Step 5 in the operator workflow now documents the two release detection paths (frame-equality vs silence) and clarifies that frame-equality is the normal disengage trigger while silence is only a Ctrl-C backstop; the log-walkthrough subsection adds the new `[pose_proxy] override frozen detected …` line so operators know what to look for. **2026-06-10 (afternoon) follow-up**: Step 5 now also documents the symmetric engage hysteresis (`--override-engage-motion-ticks 10` requires sustained motion before engage to prevent controller-rest-jitter cycles), the bumped `--override-frozen-l2-tol 5e-3` default, and the operator-pose handoff (bridge holds at operator's exact commanded body+hands instead of x2_debug measured pose); a new "Engage / release tuning" subsection under the sim recipe surfaces the three knobs for runtime tweaking, and the bridge log walkthrough now shows the `release_pose has joint_pos_mj+left_hand_joints+right_hand_joints` confirmation and how to detect the legacy-fallback path.

---

## Operator workflow recap

1. **One-time PC2 setup**: `export POSE_PROXY_OVERRIDE_PORT=5560 POSE_PROXY_CONTROL_PORT=5559` before `x2_pc2_daemons.sh start`. Persists in the proxy tmux session across deploy restarts.
2. **Start bridge with cold restart**: `VLA_CONTROL_PORT=5559 ./run_x2_vla_runtime.sh ...` (host defaults to `--pc2-host`, no extra env needed).
3. **Start teleop alongside**: `./run_x2_quest3_planner_stack.sh --pose-port 5560 --no-deploy --pc2-host ...`.
4. **Engage**: hold the usual Quest 3 mode chord; recorder packs the operator's wire on `:5560`; proxy switches to OVERRIDE.
5. **Release**: stop publishing on the override port (drop out of teleop mode, or just stop moving for >200 ms); proxy fires `override_released`; bridge cold-restarts; VLA resumes from the operator's hand-off pose.

---

## What's intentionally NOT in scope

| Out of scope | Why |
|---|---|
| Modifying `quest3_manager_x2.py` | User constraint: VR controller code should not know about the override. Engagement is detected by the proxy purely from ZMQ wire activity on the override port. |
| Tearing down the VLA process on operator engage | The whole point is to keep the bridge alive (model loaded, cameras attached, inference loop warm) across multiple operator interventions. Cold-restart only clears smoothing state, not the policy or its weights. |
| Replaying VLA chunks that were decoded against pre-override observations | These are stale by definition (operator's takeover changed the observation distribution). The chunk-id baseline gate ensures they're dropped on the cold restart even if they finished decoding while override was active. |
| Detecting "VLA is stuck" automatically | The operator is the supervisor here. A future milestone could add a heuristic (e.g. wire velocity below a threshold for N seconds + camera-frame change above a threshold) but it's not required for the manual-takeover loop to work. |

---

## 2026-06-10 (late afternoon) — STRICT mode-gated engagement

The earlier afternoon follow-ups (engage-side hysteresis, bumped L2 tolerance) made the motion-heuristic path *better* but still left it as a heuristic on top of joint-space deltas. The first sim retry exposed the underlying issue: holding the controller still while in ARM_MANIPULATION is **indistinguishable from "operator dropped to OFF"** through the pose-delta lens, because the Quest3 manager keeps publishing the frozen last commanded pose every tick in both states. Hysteresis just slowed the flicker; it didn't fix the architectural mismatch.

The fix is to subscribe the proxy directly to the manager's `stream_mode` topic (msgpack `{mode: "OFF" | "LOCOMOTION" | "ARM_MANIPULATION", tick, ts}`, already published on the recorder PUB for the recorder's episode-lifecycle bookkeeping) and gate engagement on **`mode != "OFF"`** instead of on motion. The operator's A+B+X+Y button press is now the single source of truth for "is teleop on?".

### What landed (strict mode)

- [`gear_sonic_deploy/scripts/x2_pose_proxy.py`](../../../../gear_sonic_deploy/scripts/x2_pose_proxy.py): new CLI args `--teleop-mode-{host,port,topic,stale-ms}` open an optional second SUB on the manager's PUB. When `--teleop-mode-port > 0`, the main loop drains the mode SUB each tick, decodes the msgpack payload, and tracks `current_teleop_mode` + `last_teleop_mode_s`. The `override_fresh` calculation branches on `teleop_mode_enabled`: in strict mode, engagement requires the mode signal to be fresh (`< --teleop-mode-stale-ms`, default 1 s) AND `current_teleop_mode != "OFF"`; motion-hysteresis / frozen-detection are **completely bypassed**. On `mode == "OFF"` (or stale signal) the proxy fires `override_released` on the **next tick** — no debounce. Soft msgpack dependency: missing msgpack triggers a warn-and-disable that falls back to the legacy motion path so older deployments still boot. Status line gains a `gate(mode=… age=Xms msgs=N fail=K [STALE|OFF])` cluster that replaces the motion/frozen counters when strict mode is active.
- [`gear_sonic/scripts/run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh): new env vars `POSE_PROXY_TELEOP_MODE_{HOST,PORT,TOPIC,STALE_MS}` (defaults `127.0.0.1:5564` matching the manager's `--recorder-pub-port`) + matching `--pose-proxy-teleop-mode-*` CLI flags. `spawn_sim_proxy` forwards the new flags when `POSE_PROXY_OVERRIDE_PORT > 0` AND `POSE_PROXY_TELEOP_MODE_PORT > 0` (both opt-in to keep the legacy heuristic path one env-var away). Help text now warns that motion-hysteresis is IGNORED in mode-gated mode.
- [`gear_sonic_deploy/scripts/x2_pc2_daemons.sh`](../../../../gear_sonic_deploy/scripts/x2_pc2_daemons.sh): mirror env vars (`POSE_PROXY_TELEOP_MODE_{HOST,PORT,TOPIC,STALE_MS}`) so PC2 operators can switch on strict mode by exporting two extra vars before `start`. The startup log line now explicitly says "engage gate: STRICT stream_mode …; motion-hysteresis IGNORED" or "engage gate: LEGACY motion-hysteresis; will flicker if operator holds controller still" so postmortems can immediately distinguish the two paths.
- [`tests/test_x2_pose_proxy_dual_source.py`](../../../../tests/test_x2_pose_proxy_dual_source.py): two new end-to-end smokes. `test_proxy_strict_mode_gate_blocks_off_engages_arm_and_holds_through_freeze` drives the exact user-observed sequence (mode=OFF + override → no engage; mode=ARM_MANIPULATION + frozen override → engage and **stay engaged** through the freeze that would flicker the legacy path; mode=OFF → release with `release_pose` payload), with motion-hysteresis values that would absolutely flicker under the legacy path (`--override-frozen-ticks 3 --override-frozen-l2-tol 1e-6`) so a regression to the heuristic path would visibly fail the test. `test_proxy_strict_mode_gate_fails_closed_on_stale_signal` binds the mode PUB but never publishes; asserts zero engagement events even with override frames streaming, pinning the "fail closed when manager dies" contract.
- [`tests/test_run_x2_vla_runtime_sim_proxy.py`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py): launcher CLI + argv-echo tests extended for the new `--pose-proxy-teleop-mode-{host,port,topic,stale-ms}` flags; `_build_launcher_argv` now mirrors the new `spawn_sim_proxy` defaults (sim default `127.0.0.1:5564`).
- [`pick_place_commands.md`](../../../pick_place_commands.md): operator workflow rewritten around strict mode-gated engagement as the canonical path. Demoted motion-hysteresis to a legacy fallback (replay smoke tests only). Added log examples for both paths with the new `gate(mode=…)` status line cluster; added a strict-mode tuning subsection alongside the (now-legacy) motion-hysteresis flags. PC2 daemon recipe updated with the two new env vars (`POSE_PROXY_TELEOP_MODE_HOST`, `POSE_PROXY_TELEOP_MODE_PORT`) and a side-by-side startup-log comparison for the "enabled" vs "fell back to legacy" cases.

### Test results

- Fast tests (16 across `test_run_x2_vla_runtime_sim_proxy.py`, `test_vla_control_signal.py`, `test_x2_pose_proxy_dual_source.py` non-smoke): all pass in ~1.3 s.
- Smoke tests (7 in `test_x2_pose_proxy_dual_source.py`, `X2_POSE_PROXY_SMOKE=1`): all pass in ~15 s. Critically the new `test_proxy_strict_mode_gate_blocks_off_engages_arm_and_holds_through_freeze` passes with `--override-frozen-ticks 3 --override-frozen-l2-tol 1e-6` (settings that absolutely would have triggered flicker under the legacy path).

## 2026-06-10 (evening) — OmniHand SUB was bypassing the proxy

User reported override + body now work great but **fingers still don't respond to teleop**. Tracing the sim wire surfaced the actual cause: the MuJoCo bridge's `_omnihand_zmq_thread` (which writes finger qpos into the sim) has its own ZMQ SUB pinned to `--hand-zmq-host/--hand-zmq-port`, defaulting to `localhost:5556` -- the **VLA bridge's port**, NOT the proxy's downstream port (`:5558`). So in the takeover wire:

- **Body** joints: `bridge :5556 -> proxy :5558 -> sim deploy body SUB` (arbitrated, override works)
- **Hand** joints: `bridge :5556 -> OmniHand SUB (bypasses proxy entirely)` -- always VLA, regardless of override state

The bridge keeps publishing VLA hand chunks at full rate the whole time, so the OmniHand subscriber never sees a single operator finger command no matter what the proxy decides about the body wire. The operator's hand intent flows recorder -> :5560 -> proxy -> :5558 just fine; nothing was listening for it.

### Fix

[`gear_sonic/scripts/run_x2_vla_runtime.sh::spawn_sim_deploy`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh): when `SIM_WITH_OMNIHAND=1` AND `SIM_PROXY_ENABLED=1`, also forward `--sim-hand-zmq-host "$deploy_pose_host"` and `--sim-hand-zmq-port "$deploy_pose_port"` to `deploy_x2.sh`. These reuse the exact same host/port the body SUB is already pointed at, so body + fingers come from the same arbitrated source by construction -- they can never disagree about which upstream is driving. Gated on `SIM_PROXY_ENABLED` so legacy autonomous-only sim runs (no proxy in the wire) aren't broken by pointing OmniHand at an unbound port.

### Test

[`tests/test_run_x2_vla_runtime_sim_proxy.py::test_spawn_sim_deploy_routes_omnihand_sub_through_proxy_when_proxy_on`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py): source-level pattern check on `spawn_sim_deploy`'s function body asserting (a) `--sim-with-omnihand` is forwarded, (b) `--sim-hand-zmq-host/port` are forwarded too, (c) the host/port pair is gated on `SIM_PROXY_ENABLED`, and (d) the values come from `$deploy_pose_host`/`$deploy_pose_port` (the same vars the body SUB uses). Avoids the fragility of sourcing the bash launcher with full env-var stubs while still failing loudly on regression.

## 2026-06-10 (evening, follow-up 2) — Wrist + finger expectations: the launcher was never enabling `--wrist-bypass ik`

After the OmniHand-through-proxy fix landed and we confirmed in `deploy.log` that the bridge was now subscribing on `:5558` and the PD loop was tracking finger setpoints to within 0.01 rad, the user reported a fresh symptom: "**both wrist and hand finger joints are not responding**". Diagnosing this took two steps:

### Wrist: SONIC is structurally incapable of tracking, and the launcher never enabled the override

The deploy CONTROL log lines from the user's run all showed `wrist_bypass_ticks=0 wrist_bypass_max_dev_rad=0.000` — meaning the wrist bypass was **off the entire run**. Per [`gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp`](../../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/include/wrist_bypass.hpp):

> SONIC's training distribution does not include diverse wrist motion and the smallmotor wrist channels have an `x2_action_scale` of just 0.0715 (vs ~0.42 elsewhere on the arm), so the policy outputs a near-static comfort pose for `*_wrist_pitch` and pins `*_wrist_roll` at the asymmetric joint-range tight side regardless of what the IK reference asks for.

So even though the operator's wrist commands were correctly flowing through the recorder → :5560 → proxy → :5558 → deploy wire (and the same is true for VLA-decoded chunks in autonomous mode), SONIC was clamping `target_pos_mj[20,21,27,28]` (left/right wrist pitch + roll) back to its comfort pose every tick. `wrist_yaw` (MJ indices 19, 26) is deliberately left under SONIC because v2 telemetry shows it tracks the IK reference cleanly (correlation ~0.8).

The C++ team already built the surgical override for this — `--wrist-bypass ik` — which overwrites just those 4 MJ slots with the wire's `joint_pos_mj` BEFORE the safety stack runs (so soft-start, `--max-target-dev` clamp, and tilt-trip force-to-default still apply uniformly). The unit test [`test_obs_builder.cpp`](../../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/test/test_obs_builder.cpp) pins the MJ indices against `policy_parameters.hpp::mujoco_joint_names` to prevent the override from misaligning if the joint order is ever rearranged.

The bug was in `run_x2_vla_runtime.sh`: it had no `--wrist-bypass` flag and never set `WRIST_BYPASS=ik`, so the deploy binary fell through to its `off` default. Real-robot teleop runs that pass `--wrist-bypass ik` directly to `deploy_x2.sh` worked fine, but the VLA runtime path was silently broken since the wrist_bypass landed.

### Fingers: the operator's `hand_finger_cmd` is what's on the wire, not VLA's stuck pose

After the previous fix the OmniHand SUB does read from the proxy downstream, and during override the proxy forwards the recorder's pose frames verbatim — which contain `left_hand_joints` / `right_hand_joints` filled in from the manager's `hand_finger_cmd` topic. But the manager's retargeter only emits non-zero finger curls when the operator **actively pulls the VR triggers**. If the operator entered ARM_MANIPULATION without pulling triggers, the manager would publish `left_hand_q = right_hand_q ≈ zero` (open hand), and the OmniHand would faithfully snap fingers to the open pose and stay there — which looks indistinguishable from "fingers not responding" if the operator was expecting them to stay in whatever closed pose VLA had previously commanded.

This isn't a bug; it's the design. But "did the manager actually publish a non-zero curl?" was previously invisible from the recorder's status line, so it was impossible to tell apart from a real wiring fault.

### Attempted fix (REVERTED -- defaulted to `ik`, caused slam; see follow-up 4 below)

[`gear_sonic/scripts/run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh):

- New `WRIST_BYPASS` env var (initially defaulted to `ik`; reverted to `off` in 2026-06-10 follow-up 4 below after empirical slam at startup) and `--wrist-bypass {off,ik}` CLI flag.
- `spawn_sim_deploy` now appends `--deploy-extra-arg --wrist-bypass --deploy-extra-arg "$WRIST_BYPASS"` (two extras because `deploy_x2.sh` appends each verbatim and the C++ binary expects `--wrist-bypass <mode>` as a value-separated pair).
- Banner line `Wrist bypass: <mode>` with explicit DANGER text when `ik` is selected.
- `--help` documents the flag with a forward reference to `wrist_bypass.hpp` so the rationale is one click away, AND with the per-tick-clamp dependency that the original `wrist_bypass.hpp` commentary buries deeper in the file.

[`gear_sonic/utils/teleop/x2_dataset_recorder.py::_run_subscribe_mode`](../../../../gear_sonic/utils/teleop/x2_dataset_recorder.py): the subscribe-mode status line now also reports `hand|L|=… hand|R|=…` with a `(manager|zero-fallback)` source tag. Reading "hand|L|=0.000 (zero-fallback)" tells the operator the manager isn't pushing a finger command yet (= they need to pull the triggers); "hand|L|=0.452 (manager)" tells them the manager IS pushing a command and any disconnect must be downstream.

### Test

[`tests/test_run_x2_vla_runtime_sim_proxy.py::test_spawn_sim_deploy_forwards_wrist_bypass_via_deploy_extra_arg`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py): source-level pattern check on `spawn_sim_deploy`'s function body asserting (a) `--wrist-bypass` is forwarded, (b) at least 3 `--deploy-extra-arg` occurrences are present (one for `--disable-pose-ref-watchdog`, plus a key+value pair for `--wrist-bypass <mode>`), and (c) the default `WRIST_BYPASS:=ik` is present in the launcher source so operators get the working configuration by default.

[`tests/test_run_x2_vla_runtime_sim_proxy.py::test_launcher_help_lists_wrist_bypass_flag`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py): pins `--wrist-bypass` in the `--help` heredoc so the case statement and the help text can't drift.

## 2026-06-10 (evening, follow-up 3) — Smooth handoff: bridge holds operator pose until first eligible chunk

Once strict mode gating + wrist bypass + finger telemetry landed and the operator could cleanly override and release, the next observable issue was an **abrupt step** at the OVERRIDE → LIVE seam. Two distinct mechanisms were producing it:

1. **The minimum hold (`--vla-cold-restart-hold-ticks`, default 25 = 500 ms) could expire before the first eligible chunk arrived.** Inference cadence is ~15 Hz (`--inference-min-period-s 0.4`); wire cadence is 50 Hz. First-chunk decode time is empirically ~480 ms (see the [`bridge.log`](../../../tutorials/x2_vla_runtime.md) `inference #0000 elapsed=480ms` line from the user's runs). So in the worst case the post-hold tick lands BEFORE the next chunk has been written to the shared slot. With nothing eligible, the publisher fell through to `cur_jpos = idle_loop_pose`, producing a visible snap from the operator's hand-off pose to the idle_stand clip's pose for as many wire ticks as the inference latency ate.

2. **Even with an eligible chunk available, the ramp-in seeded from `idle_baseline`**, not from the operator pose. The ramp went `idle_clip_pose → decoded_now` over 75 ticks (1.5 s), and the chunk_blend interpolated `prev_wire_jpos → ramped_now` over 40 ticks (0.8 s). With `prev_wire_jpos = operator_pose` the chunk_blend smoothed the transition, but the LPF state was ALSO seeded from `idle_baseline` (line 2594 of `live_vla_publish_motion_token.py` pre-fix), so the first filtered tick had a per-LPF-step kick that the chunk_blend couldn't hide.

### Fix

[`gear_sonic/scripts/live_vla_publish_motion_token.py`](../../../../gear_sonic/scripts/live_vla_publish_motion_token.py): two-layer smooth-handoff guard.

**Layer 1 — extended hold until first eligible chunk.** New state vars `cold_restart_awaiting_first_chunk: bool` and `cold_restart_max_hold_remaining: int`. The cold-restart trigger arms both. The hold check (`in_hold_window`) now ORs in `cold_restart_awaiting_first_chunk`, so the wire stays pinned at the operator pose past the minimum 25-tick dwell **until** `chunk_id > cold_restart_chunk_baseline` (a fresh chunk decoded after the operator released the wire). A safety cap (`--vla-handoff-max-hold-ticks`, default 200 = 4 s) bounds the wait so the operator isn't stuck at their hand-off pose forever if the decoder hangs / starves / produces only zero-token chunks; the cap fires with an explicit `WARNING` log line so postmortems are unambiguous.

**Layer 2 — operator-pose-seeded ramp + LPF.** The ramp-init block (line ~2589) now checks if `operator_hold_pose` is still alive (kept alive by the await-first-chunk gate). When it is, the ramp `from`, body LPF state, AND both hand LPF states are seeded from the operator's snapshot instead of `idle_baseline`. Net result: the wire interpolates `operator_pose → decoded_now` over `--vla-ramp-in-ticks` (default 75 = 1.5 s), with chunk_blend smoothing the per-chunk seam on top, and the LPFs operating from a state that matches the wire's current output (no kick on tick 1).

### New CLI flag

- `--vla-handoff-max-hold-ticks N` on the bridge (default 200); `--vla-handoff-max-hold-ticks` on `run_x2_vla_runtime.sh` (forwards to the bridge); `VLA_HANDOFF_MAX_HOLD_TICKS` env var fallback. MUST be >= `--vla-cold-restart-hold-ticks` (the bridge's startup validator exits 2 with a clear message otherwise — fails fast at launch rather than at the first handoff).
- New banner line: `Handoff guard: cold_restart_hold=25t max_hold=200t (wire stays at operator pose until first eligible chunk, capped by max_hold)`.

### Log signal

Three new log lines surface the guard's state:

```
[live-VLA] cold-restart fired tick=NNN baseline_chunk=K min_hold_ticks=25 max_hold_ticks=200; will hold wire at operator's last commanded pose (body+left_hand+right_hand) until first eligible chunk (chunk_id > K) decodes, capped by max-hold safety
[live-VLA] cold-restart handoff: first eligible chunk decoded (chunk_id=K+1 > baseline=K); releasing wire hold at tick=NNN+M, ramping into VLA from operator pose
[live-VLA] cold-restart handoff: ramp + LPF seeded from operator pose (ramp_ticks=75); VLA wire re-engaging from hand-off pose without idle-clip detour
```

Or, on the safety-cap path:

```
[live-VLA] WARNING: cold-restart handoff safety cap reached at tick=NNN (handoff_max_hold_ticks elapsed without chunk_id > baseline=K); releasing wire to idle. Check the decoder (stuck inference? proprio starvation? zero-token chunks?).
```

### Tests

[`tests/test_run_x2_vla_runtime_sim_proxy.py`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py):

- `test_launcher_help_lists_manual_takeover_flags` — pin `--vla-handoff-max-hold-ticks` in the `--help` heredoc so case-statement / heredoc drift fails loudly.
- `test_launcher_forwards_handoff_max_hold_ticks_to_bridge` — source-level pattern check on `BRIDGE_ARGS` wiring + `VLA_HANDOFF_MAX_HOLD_TICKS:=200` default.
- `test_bridge_fails_fast_when_handoff_max_hold_less_than_cold_restart_hold` — runs the bridge with `--help`, confirms argparse exposes the flag with the expected "Safety cap" doc text. (The deeper `--vla-handoff-max-hold-ticks 10 --vla-cold-restart-hold-ticks 25` -> sys.exit(2) path is exercised by the startup validator at first launch and is too expensive to e2e in unit tests; the docstring + help-text pinning is the practical proxy.)

All 11 tests pass in <2 s.

## 2026-06-10 (evening, follow-up 4) — Wrist-bypass default was unsafe; reverted

Follow-up 2 above defaulted `WRIST_BYPASS=ik` to make manual takeover and VLA wrist tracking actually move the wrist. The operator restarted the sim and reported "**the hand slammed into the table**". The `deploy.log` from that 12:26 run had unambiguous telemetry on every CONTROL line:

```text
CONTROL tick=2550 ... act_clip_ticks=2518 max_pre_clip=24.75 wrist_bypass_ticks=2550 wrist_bypass_max_dev_rad=1.796 pose_ref_age=-1.000s mc_mode=-1
```

`wrist_bypass_max_dev_rad=1.796` = ~103° of deviation between what SONIC's tracker wanted to do for the 4 bypassed DOFs and what the bypass force-wrote (= the wire's `joint_pos_mj` values). Every tick. The bridge log shows the wire was in idle mode the entire time (`|token|=0.000 |left|=0.000 idle-pose`), so the source was the `idle_stand` clip's wrist values -- which by design differ ~100° from SONIC's natural pinned wrist (the whole reason SONIC pins them is that the training distribution clusters there).

Without `--max-target-dev` configured (default in our launcher: empty), the only per-tick rate limit on `target_pos_mj` is the soft-start blend, which is measured in seconds. The wrist swung the full 100° gap at the soft-start rate -- looks/feels like a slam given the OmniHand mounts to the wrist roll link and follows it through the workspace.

### Fix

Reverted `WRIST_BYPASS:=off` in [`gear_sonic/scripts/run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh). The launcher banner now shows DANGER text when `ik` is selected explicitly, and the `--help` for `--wrist-bypass` explicitly calls out the `--max-target-dev` pairing requirement. The pinning test in [`test_run_x2_vla_runtime_sim_proxy.py::test_spawn_sim_deploy_forwards_wrist_bypass_via_deploy_extra_arg`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) was updated to assert the safe default with a comment referencing the empirical evidence (terminal 2, 12:26 deploy.log) so a future "let's just make it work out of the box" attempt fails the regression pin before it gets the chance to slam anything again.

### What this means for operator workflow

- Default behaviour: wrist pitch/roll stay pinned at SONIC's comfort pose. Operator + VLA wrist commands on the wire are IGNORED for those 4 DOFs. The hand visibly does not respond to wrist gestures (this was the original symptom that motivated the bypass). Operator can still pick + place via shoulder + elbow + wrist_yaw (SONIC tracks wrist_yaw cleanly, correlation ~0.8 per wrist_bypass.hpp).
- Opt-in `ik`: pass `--wrist-bypass ik` AND `--deploy-extra-arg --max-target-dev <rad>` (e.g. `0.05` for ~3°/tick) together. The launcher does not couple these automatically because the right value of `max-target-dev` depends on the wire's content (operator IK retargeter output vs VLA chunk output have different statistics), so an automatic pairing would be a foot-gun in the other direction.

### Follow-up TODO (next session)

The right end-state is to make `--wrist-bypass ik` safe by construction:

1. Have the launcher set a sensible `--max-target-dev` default when `WRIST_BYPASS=ik` is selected (or refuse to start without it). 0.05 rad/tick at 50 Hz is 2.5 rad/s -- one full wrist swing per ~1.3 s -- aggressive but not slam-fast.
2. Match the bridge's `idle_stand` clip's wrist values to SONIC's natural pinned pose (or write a dedicated `idle_stand_for_wrist_bypass` clip). This would let `ik` be the default without the startup swing.
3. Add per-tick `wrist_bypass_step_rad` telemetry to the deploy's status line so operators can see the rate-limit kicking in.

None of those landed today; the safe-default revert was the right immediate move.

## 2026-06-10 (evening, follow-up 5) — Smooth handoff gate let zero-token chunks slam the wire

After follow-up 4 reverted the wrist-bypass default, the operator restarted and reported "**hand still slams instead of moving slowly to its position**" along with "**fingers stopped working again**". Pulled the run log (`/tmp/x2_vla_runtime-20260610_124920/`) and ground-truthed:

**Slam root cause (orthogonal to follow-up 4):**

The bridge log showed the smooth-handoff guard firing correctly per follow-up 3:

```text
[live-VLA] cold-restart fired tick=3951 baseline_chunk=499 min_hold_ticks=25 max_hold_ticks=200; will hold wire at operator's last commanded pose ...
[live-VLA] cold-restart handoff: first eligible chunk decoded (chunk_id=503 > baseline=499); releasing wire hold at tick=3976, ramping into VLA from operator pose
```

Looks fine. BUT every chunk in this run was reading `|token|=0.000 |left|=0.000 idle-pose` -- VLA was producing zero-token output the entire run. The decoder gate further down had its own guard (`np.linalg.norm(token[step]) > 1e-3`) that correctly refused zero-token chunks. Result:

1. `chunk_id=503 > baseline=499` -> the handoff gate fires, `cold_restart_awaiting_first_chunk = False`
2. The decoder gate at line 2569 rejects `|token|=0.000` -> `decoded_now = None`
3. `cur_jpos` falls through to `idle_loop.current(tick)` = idle_stand pose
4. Previous tick's `cur_jpos` was operator hold pose
5. The per-tick `_clamp_vector_step` rate clamp was ONLY applied inside the decoder-succeeded branch
6. Wire snaps from operator pose to idle_stand pose in one tick = SLAM

So the slam is "follow-up 3's handoff gate was too permissive" + "rate clamp wasn't defense-in-depth on the idle branch". Two fixes:

#### Fix 1: Token-norm guard on the handoff gate

[`gear_sonic/scripts/live_vla_publish_motion_token.py`](../../../../gear_sonic/scripts/live_vla_publish_motion_token.py): the `first_eligible_chunk_ready` predicate now mirrors the decoder's `np.linalg.norm(token[step]) > 1e-3` guard. The gate stays armed (= wire pinned at operator pose) until VLA is actually producing usable tokens OR `handoff_max_hold_ticks` expires. The log line now also reports the live `|token|` magnitude on both the success and safety-cap paths so an operator can immediately see "VLA is producing zero-token output, the handoff is correctly stalling" vs "VLA is alive, handoff completing normally".

#### Fix 2: Always-on per-tick wire rate clamp

Same file: `_clamp_vector_step(cur_jpos, prev_wire_jpos, max_wire_step)` now also runs in the `else` (idle wire) branch. Numpy `abs+max` cost; no-op on normal idle frames where the per-element delta is below `max_wire_step` (0.035 rad/tick = 1.75 rad/s at 50 Hz). This is the safety net: if `handoff_max_hold_ticks` DOES expire (VLA still zero-token after 4s), the wire ramps from operator pose toward idle_stand at bounded rate instead of snapping. Combined with fix 1, the worst case is now "4s hold at operator pose, then a 2-3s ramp from operator pose to idle_stand" -- visibly smooth, no slam.

#### Test coverage

[`tests/test_run_x2_vla_runtime_sim_proxy.py::test_handoff_gate_requires_nontrivial_token_magnitude`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py): source-level pattern pin for both the `current_token_norm` variable AND the `_clamp_vector_step(cur_jpos, prev_wire_jpos, max_wire_step)` call inside the idle branch. A future refactor that drops either falls this test before the operator hits another slam in the field.

**Finger root cause (input-side, not wire-side):**

The recorder log showed `hand|L|=0.000(manager) hand|R|=0.000(manager)` for every tick across both OFF and ARM_MANIPULATION windows. The `(manager)` tag means the recorder IS receiving `hand_finger_cmd` from the manager -- the published value just happens to be 0.000. The OmniHand wire path is healthy (deploy.log shows `[bridge] OmniHand ZMQ SUB connected at tcp://127.0.0.1:5558` + first finger setpoint `|L|=0.000 |R|=0.000`).

So the issue is that the manager isn't producing non-zero hand_q. From the [`Quest3Reader`](../../../../gear_sonic_deploy/scripts/quest3_reader.py) log: `right: controller (gamepad=yes grip=yes hand=NO)` -- XRHand tracking is OFF, so [`X2RetargetPipeline`](../../../../gear_sonic/utils/teleop/x2_retarget_pipeline.py) lines 363-371 take the controller-trigger fallback path (`controller_grasp_ratio(left_trigger, right_trigger, left_grip, right_grip, mode=trigger)`). If the operator's trigger pulls were reaching the manager, hand_q should be non-zero. But the operator's ARM_MANIPULATION windows in the manager log are 1-5 s each (they were testing handoff, not trigger pulls).

To close the diagnosis loop, [`gear_sonic/scripts/quest3_manager_x2.py`](../../../../gear_sonic/scripts/quest3_manager_x2.py) now emits a periodic ARM_MAN-gated telemetry line every 250 ticks (= 5 s @ 50 Hz):

```text
[manager-x2] tick=N ARM_MAN triggers=lt=0.85 rt=0.00 lg=0.00 rg=0.00 published hand_q|L|=2.413 |R|=0.000
```

This lets the operator distinguish:
- `triggers=lt=0.00 rt=0.00 lg=0.00 rg=0.00` -> the controller isn't producing trigger values (battery? wrong axis? XRHand mode confused?)
- `triggers=lt=0.85 ... published hand_q|L|=0.000` -> retargeter is broken (calibration? hand_input_mode mismatch?)
- `triggers=lt=0.85 ... hand_q|L|=2.413` BUT OmniHand still doesn't curl -> wire is broken (proxy override forwarding regression, OmniHand SUB host/port wiring)

If the next field test still shows fingers not responding, the new telemetry line will pinpoint which of those three layers is broken.

## 2026-06-10 (afternoon, follow-up 6) — Smooth handoff worked but body still slammed: per-tick rate clamp was wrong-magnitude

Follow-up 5 tightened the handoff eligibility gate and added the always-on idle-branch rate clamp. The operator re-ran (`/tmp/x2_vla_runtime-20260610_130545/`) and reported "**smooth transition seems to work sometimes, but still slamming some other time**". Pulled the new bridge log and the handoff guard was firing cleanly on all three release events:

```text
[live-VLA] cold-restart handoff: first eligible chunk decoded (chunk_id=200 > baseline=199, |token|=4.362 > 1e-3); releasing wire hold at tick=15959, ramping into VLA from operator pose
[live-VLA] pub tick= 16000 chunk_id= 200 step=39/40 ... VLA-pose raw_Δ=3.720rad wire_Δ=1.165rad body_Δ=0.247rad ramp=33/75 hand_Δ=0.012rad
```

Token magnitudes are healthy. The gate stays armed only as long as it should. So the slam isn't the gate -- it's the **per-tick wire-step magnitude itself**.

Decoded the log fields:

- `raw_Δ=3.720 rad` = L_inf distance between VLA's decoded body and the idle baseline (= ~213°, VLA wants the arm in a very different place than where the operator left it)
- `wire_Δ=1.165 rad` = L_inf distance the wire has actually moved from idle so far (rate-clamped)
- `body_Δ=0.247 rad` = L_inf distance between the wire's setpoint and the measured body (= sustained 14° tracking error)
- `ramp=33/75` = inside the 75-tick LPF blend

`max_wire_step=0.035 rad/tick` is a **per-element** rate clamp -- not an L2 clamp. So all 31 joints can step `0.035` rad simultaneously in one tick. The resulting L2 step magnitude is `sqrt(31) * 0.035 ≈ 0.195 rad/tick = 9.75 rad/s coordinated whole-body motion`. That's what the operator visually reports as a slam, even though no single joint exceeds the limit. The 75-tick LPF doesn't help once it's blended past ~30% of the way -- the residual step is still per-element clamped at `max_wire_step`.

### Fix

[`gear_sonic/scripts/live_vla_publish_motion_token.py`](../../../../gear_sonic/scripts/live_vla_publish_motion_token.py) gains two new parameters wired through `_publisher`:

- **`--vla-handoff-max-wire-step`** (default `0.012` rad/tick = ~36 deg/s/joint): per-element rate clamp applied for the first ticks AFTER the cold-restart hold releases. ~3x slower than the steady-state `--vla-max-wire-step` (0.035).
- **`--vla-handoff-step-ramp-ticks`** (default `250` = 5 s @ 50 Hz): linearly interpolates the rate clamp from `handoff_max_wire_step` -> `max_wire_step` over this many ticks, so the wire starts very slow right after handoff (when `raw_Δ` is largest and a single tick could whip the body) and accelerates back to normal as VLA stabilises.

`handoff_step_remaining` is armed in BOTH the success path (`first_eligible_chunk_ready`) and the safety-cap path so post-cap "ramp to idle_stand" also uses the slow step (the safety cap's idle delta is comparable in magnitude). The interpolated `effective_max_step` is applied to the rate clamp call in BOTH the decoder-succeeded branch and the idle-wire fallthrough branch -- defense-in-depth from follow-up 5 -- so neither path can slam.

### Tests

[`tests/test_run_x2_vla_runtime_sim_proxy.py`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) gains two pins:

- `test_launcher_forwards_handoff_slow_step_to_bridge` — env-var defaults + case-statement + `BRIDGE_ARGS` wiring for both new flags.
- `test_handoff_slow_step_state_machine_in_bridge_source` — source-level pin for `handoff_step_remaining`, the success+safety-cap arming pattern (appears `>= 2` times), the linear interpolation formula, and the dual-branch `effective_max_step` application.

All 14 tests in the file pass in <2 s.

### Launcher surfacing

[`gear_sonic/scripts/run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh) banner now shows the slow-step config alongside the hold caps:

```text
Handoff guard   : cold_restart_hold=25t max_hold=200t slow_step=0.012rad/t for 250t ...
```

## 2026-06-10 (afternoon, follow-up 7) — Wrist bypass back ON with auto-paired --max-target-dev

The operator's same 13:12 message: "**wrist not responding. we need the wrist ik enabled i think**". Follow-up 4 reverted `WRIST_BYPASS=off` after the 12:26 slam, but the wrist being completely SONIC-pinned was its own usability bug: operator wrist gestures and VLA wrist tokens both got discarded.

The 12:26 slam wasn't `wrist_bypass=ik` ON its own -- it was `ik` WITHOUT a `--max-target-dev` per-tick clamp on the deploy side. The deploy's `wrist_bypass.hpp` explicitly says soft-start + max-target-dev + tilt-trip all apply to the bypassed wrist target. We just never set max-target-dev.

### Fix

[`gear_sonic/scripts/run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh):

- `WRIST_BYPASS:=ik` (back to ik, the value the operator explicitly wants)
- NEW `WRIST_BYPASS_MAX_TARGET_DEV:=0.05` env var (0.05 rad/tick = ~2.5 rad/s per joint) -- bounds the idle_stand -> default wrist delta (~1.8 rad) to ~720 ms instead of the 12:26 slam
- `spawn_sim_deploy` automatically appends `--deploy-extra-arg --max-target-dev --deploy-extra-arg "$WRIST_BYPASS_MAX_TARGET_DEV"` whenever `wrist_bypass != off`. The pairing is wired structurally so it can't be skipped by accident.
- Operators who want to tune their own `--max-target-dev` can pass it explicitly via `--deploy-extra-arg --max-target-dev <N>` AFTER the launcher's forwarded value (argparse takes the last instance), or unset `WRIST_BYPASS_MAX_TARGET_DEV` to opt out of the auto-pair entirely (NOT recommended).

### Test pin update

[`tests/test_run_x2_vla_runtime_sim_proxy.py::test_spawn_sim_deploy_forwards_wrist_bypass_via_deploy_extra_arg`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) now asserts BOTH the `ik` default AND the auto-paired `--max-target-dev` wiring. A future change that flips back to off OR drops the auto-pair fails this test before it ships.

## 2026-06-10 (afternoon, follow-up 8) — `--max-target-dev` auto-pair from follow-up 7 collapsed the robot

The operator restarted after follow-up 7 (wrist_bypass=ik re-enabled with auto-paired `--max-target-dev 0.05`) and reported "**the robot just collapsed and tilted on the table in front**". The 13:21 deploy.log spells it out at startup:

```text
SAFETY: per-joint target clamp ENABLED. Effective per-group |target - default| limits:
  leg=0.050 rad (2.9 deg), waist=0.050 rad (2.9 deg), arm=0.050 rad (2.9 deg), head=0.050 rad (2.9 deg)
  (global default --max-target-dev=0.050 rad (2.9 deg); per-group overrides win when > 0)
```

Then on every CONTROL line:

```text
CONTROL tick=1000 ... act_clip_ticks=916 max_pre_clip=22.25 wrist_bypass_ticks=1000 wrist_bypass_max_dev_rad=1.426 ...
```

`act_clip_ticks=916/1000` = 92% of policy outputs clamped because the safety stack was fighting the policy on every tick. With every joint group pinned to +/-2.9 deg of `default_angles`, the robot literally couldn't bend its knees enough to stand. It collapsed forward onto the table.

### What I got wrong about `--max-target-dev`

I read the `wrist_bypass.hpp` comment ("soft-start blend, ``--max-target-dev`` clamp, and the tilt-trip force-to-default branch still apply uniformly to the IK-driven target") as "`--max-target-dev` is a per-tick rate clamp on the bypassed wrist target". WRONG on two counts:

1. **`--max-target-dev` is an ABSOLUTE deviation clamp**, not a per-tick step clamp. It enforces `|target_pos - default_angles| <= N rad` every tick. A small value pins the joint near default forever; it doesn't slow how fast the joint can move per tick.
2. **`--max-target-dev` is GLOBAL** -- applied to leg + waist + arm + head joint groups, NOT wrist-specific. The C++ deploy DOES expose per-group overrides (`--max-target-dev-{leg,waist,arm,head}`), but there's no `--max-target-dev-wrist` group; the "arm" group covers MJ joints 15..28 which includes shoulder + elbow + wrist_yaw too. So even `--max-target-dev-arm 0.05` would break shoulder/elbow tracking.

`--max-target-dev` is the wrong tool for rate-limiting the wrist bypass.

### Fix

[`gear_sonic/scripts/run_x2_vla_runtime.sh`](../../../../gear_sonic/scripts/run_x2_vla_runtime.sh):

- Removed `WRIST_BYPASS_MAX_TARGET_DEV` env var entirely.
- Removed the auto-pair `--deploy-extra-arg --max-target-dev <N>` from the wrist-bypass forwarding block. The block now ONLY forwards `--wrist-bypass <mode>`.
- Banner + help text updated to explicitly call out that the launcher does NOT auto-pair, and that wrist slam mitigation lives on the bridge side instead (via `--vla-max-wire-step` + follow-up 6's slow-step ramp).

### Why the bridge-side rate clamps are sufficient

`wrist_bypass=ik` force-writes `target_pos_mj[{20,21,27,28}]` from the wire's `joint_pos_mj`. So whatever rate-limits the WIRE's `joint_pos_mj` also rate-limits the bypassed wrist target -- transitively, via the bypass copy. The bridge already enforces:

1. `--vla-max-wire-step` (0.035 rad/tick = 1.75 rad/s/joint, steady state)
2. `--vla-handoff-max-wire-step` (0.012 rad/tick = 0.6 rad/s/joint for the first 5 s after handoff, per follow-up 6)

So the post-handoff wrist transition is bounded to the same rate as the rest of the body. No deploy-side clamp needed for that path.

The remaining wrist slam risk is the STARTUP transition: at first bridge launch, the wire ships the `idle_stand` clip's wrist values, which differ from `default_angles` by ~1.8 rad. The deploy's soft-start blend ramps `target_pos` from `default_angles` to the wire's target over a few seconds -- visibly a wrist swing, not a snap. Acceptable trade-off given the alternative is a collapsed robot.

### Test regression pin (NEGATIVE)

[`tests/test_run_x2_vla_runtime_sim_proxy.py::test_spawn_sim_deploy_forwards_wrist_bypass_via_deploy_extra_arg`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) now contains a NEGATIVE assertion: `WRIST_BYPASS_MAX_TARGET_DEV` must NOT appear in the launcher source, AND the wrist-bypass forwarding block must NOT contain `--deploy-extra-arg --max-target-dev`. A future revival of the same mistake fails CI before it ships.

### Future TODO (if we still need a deploy-side wrist clamp)

Plumb a new per-group override `--max-target-dev-wrist` through the C++ deploy (MJ joints {20, 21, 27, 28}). Then we could safely pair it with `wrist_bypass=ik` at the launcher level without touching legs/waist/shoulder/elbow. Not urgent: the bridge-side rate clamps already cover the post-handoff path, and the startup path is acceptable.

## 2026-06-10 (afternoon, follow-up 9) — Symmetric slow-step ramp on the LIVE → OVERRIDE edge

Follow-up 6 fixed the slam during the OVERRIDE → LIVE handoff (operator releases A+B+X+Y; VLA takes back over). The operator confirmed at 13:29:

> "smooth transition worked from arm manipulation to off transition. but it still rams when taking over from vla to ON. can you apply in that path as well?"

So the OPPOSITE direction is still unfixed. When the proxy fires the LIVE → OVERRIDE edge (operator presses A+B+X+Y, manager's `stream_mode` flips to non-`OFF`), the proxy starts forwarding the operator's pose frames **verbatim**. If the operator's pose is far from VLA's last commanded pose (typical: the soda-can pick scenario leaves VLA holding the arm forward-up, then the operator's pose maps to the controller's rest position with arms down at the sides), the first OVERRIDE frame steps the wire by ~3 rad L∞ across the body in one tick. The deploy's safety stack tries to track and slams.

### Diagnosis

Walked the proxy's main loop ([`gear_sonic_deploy/scripts/x2_pose_proxy.py`](../../../../gear_sonic_deploy/scripts/x2_pose_proxy.py)) and confirmed the LIVE → OVERRIDE branch is literally:

```python
if latest_override is not None:
    pub.send(latest_override, zmq.NOBLOCK)   # verbatim forward
    ...
cur_state = STATE_OVERRIDE
```

No clamp, no blend, no ramp. The bridge's rate limits don't apply here because the bridge isn't producing wire content during OVERRIDE (it's pausing decoded chunks). The proxy is the only thing in the path between the operator's pose source and the deploy.

### Fix

Implemented a SYMMETRIC slow-step ramp in the proxy, mirroring the bridge's follow-up 6 slow-step ramp. Three new flags (defaults match the bridge's handoff defaults for symmetry):

- `--engagement-max-wire-step 0.012` — per-element max joint step (rad/tick) during the engagement ramp (~0.6 rad/s/joint at 50 Hz)
- `--engagement-steady-wire-step 0.035` — steady-state per-element step the ramp converges to (matches `--vla-max-wire-step`)
- `--engagement-step-ramp-ticks 250` — ticks to linearly interpolate slow → steady (= 5.0 s at 50 Hz, matches `--vla-handoff-step-ramp-ticks`)

State machine in `_run_main_loop`:

1. **At the LIVE → OVERRIDE edge** (`override_fresh and not override_active`): arm `engagement_clamp_remaining = engagement_step_ramp_ticks` and snapshot `engagement_last_forwarded_jpos = last_upstream_jpos` (the last VLA pose forwarded on the wire). Log the arm event.
2. **On every override-state forward** while the countdown is positive: decode the operator's `joint_pos_mj` from the override frame, compute `ramp_progress = 1 - remaining / total`, lerp `effective_max_step` between the slow and steady values, run the operator's pose through `_clamp_vector_step_f32(target=op_jpos, prev=engagement_last_forwarded_jpos, max_step=effective_max_step)` (proportional-shrink, NOT per-element clip, so the operator's intended motion direction is preserved), then surgically splice the clamped jpos back into the override frame's bytes via `rebuild_msg_with_jpos_override(msg, topic, clamped_jpos)` (preserves hand joints + root_quat + future window byte-for-byte). Forward the rebuilt frame. Update `engagement_last_forwarded_jpos = clamped_jpos`. Decrement.
3. **After the countdown hits 0**: forward verbatim (operator's controller motion is the rate limit at that point).
4. **At the OVERRIDE → LIVE edge** (`not override_fresh and override_active`): tear down both `engagement_clamp_remaining = 0` and `engagement_last_forwarded_jpos = None` so a rapid release+re-engage within the window re-arms cleanly from the new VLA anchor (otherwise we'd inherit the previous stale operator pose).

Two new helpers in the proxy module:

- `rebuild_msg_with_jpos_override(msg, topic, new_jpos)` — surgical byte-replace, walks the v4 header field list to find `joint_pos_mj`'s offset/length and splices new bytes in place. Same shape + same dtype = same byte count = no header rewrite needed. Returns `None` if the frame's header is malformed (caller falls back to forwarding the original message verbatim).
- `_clamp_vector_step_f32(target, prev, max_step)` — exact mirror of the bridge's `_clamp_vector_step`: proportional-shrink delta vector so direction is preserved. No-op when `max_step <= 0` or `prev is None` (cold-start tick).

### Symmetry with the bridge's follow-up 6

| Edge | Component | Mechanism | Slow window | Steady step |
|---|---|---|---|---|
| LIVE → OVERRIDE (operator takes over) | proxy (this fix) | `_clamp_vector_step_f32` on operator's jpos vs last-forwarded VLA pose | 5 s | 0.035 rad/tick |
| OVERRIDE → LIVE (VLA takes back) | bridge (follow-up 6) | `_clamp_vector_step` on wire jpos vs last-forwarded operator pose | 5 s | 0.035 rad/tick |

Both ramps start at 0.012 rad/tick and linearly relax to 0.035 over the first 250 ticks of the new owner's wire. From the deploy's perspective the two takeover directions are now indistinguishable: the wire's `joint_pos_mj` steps bounded ≤ 0.012 rad/tick for the first 5 s of any ownership change.

### Why surgical byte-replace, not full re-pack

The override frame carries operator-side fields the proxy MUST NOT alter:

- `left_hand_joints`, `right_hand_joints` — operator's finger commands. Tampering would lose grasps mid-takeover (the exact failure mode follow-up 5 already fixed once).
- `root_quat_xyzw` — operator's pelvis orientation reference.
- `motion_token` — operator-side tokenizer output, fed downstream as language conditioning.
- `joint_pos_mj_future` + `root_quat_xyzw_future` + `frame_index` — operator's advisory horizon.

A full `pack_pose_message(decode(msg))` round-trip would drop any field whose decoder we don't run. Surgical splice is O(message size) and touches only the `joint_pos_mj` payload bytes.

### Test regression pins

[`tests/test_run_x2_vla_runtime_sim_proxy.py`](../../../../tests/test_run_x2_vla_runtime_sim_proxy.py) gained four new tests:

- `test_launcher_forwards_engagement_slow_step_to_proxy` — three default env vars present, three CLI flags wired, three `proxy_args` append lines present, plus a NEGATIVE pin that `ENGAGEMENT_MAX_TARGET_DEV` doesn't reappear (follow-up 8 lesson: no auto-paired deploy-side clamps).
- `test_proxy_engagement_clamp_state_machine_in_source` — pins the seven structural pieces (helpers, init, arm, teardown, interp formula, rebuild call, anchor update). Catches refactors that drop any one piece.
- `test_rebuild_msg_with_jpos_override_preserves_other_fields` — round-trip pack → rebuild with new jpos → decode; asserts hands + root_quat are byte-identical. Catches a regression where someone "simplifies" the helper to a full re-pack and silently drops fields.
- `test_clamp_vector_step_f32_caps_peak_element_and_preserves_direction` — verifies proportional-shrink semantics (not per-element clip) and the `prev is None` / `max_step <= 0` cold-start escape hatches.

### Operator runbook (no command change)

The new defaults activate automatically — the launcher banner and `--help` document the three flags but the operator doesn't need to set anything. To tune:

```bash
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    ... \
    --pose-proxy-engagement-max-wire-step 0.008 \   # even slower start
    --pose-proxy-engagement-step-ramp-ticks 500     # 10 s ramp instead of 5
```

Set ramp-ticks to 0 to fully disable (returns to pre-follow-up-9 verbatim-forward behaviour; not recommended outside debugging).

## 2026-06-10 (afternoon, follow-up 9b) — Future window still snapping; clamp body-only wasn't enough

The operator restarted with follow-up 9 in place and reported at 14:58:

> "it still slams when i switch from off to on/locomotion."

[`/tmp/x2_vla_runtime-20260610_145624/sim_proxy.log`](../../../../tmp/x2_vla_runtime-20260610_145624/sim_proxy.log) confirmed the new code WAS running — every OFF → ARM_MAN edge logged:

```text
[pose_proxy] engagement slow-step ramp armed (window=250 ticks; max_step 0.012 -> 0.035 rad/tick; anchor=last_VLA_pose)
[pose_proxy] state: LIVE -> OVERRIDE (operator teleop override engaged; ...)
```

So the clamp on `joint_pos_mj` was active. But the slam was still happening.

### Diagnosis

Walked the deploy's ZMQ pose decoder ([`zmq_pose_input_source.cpp`](../../../../gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/src/zmq_pose_input_source.cpp) lines 218–249). The deploy reads **two** body-pose fields off the wire:

1. `joint_pos_mj` (shape `(31,)`) — the current commanded pose
2. `joint_pos_mj_future` (shape `(9, 31)`) — 9 future slots, 0.1 s apart, **used by the window-mode policy for forward prediction**

The recorder publishes both ([`x2_dataset_recorder.py:3263`](../../../../gear_sonic/utils/teleop/x2_dataset_recorder.py)). Follow-up 9 clamped only `joint_pos_mj`; the future window passed through untouched.

What the deploy saw at the engage tick:

| Field | Pre-clamp value | Post-clamp value (follow-up 9) | Effect |
|---|---|---|---|
| `joint_pos_mj` | operator pose (~3 rad from VLA) | VLA + 0.012 rad/tick step toward operator | smooth ✓ |
| `joint_pos_mj_future` (9 slots) | "go all the way to operator pose in 0.9 s" | **unchanged** — still pointing at full operator pose | **policy slams to follow the future ✗** |
| `joint_vel_mj_future` | ~0.33 rad/slot velocity | **unchanged** | reinforces the slam |

The window-mode policy is trained to anticipate the next 0.9 s of motion. A future window encoding ~3 rad of body motion in 0.9 s ≈ 3.3 rad/s/joint coordinated motion. The policy slams the body to track that future even though the wire's current `joint_pos_mj` was being rate-limited to 0.6 rad/s.

### Fix

Promoted the surgical-byte-replace helper to multi-field (`rebuild_msg_with_field_overrides`) and extended the engagement clamp to splice **three** fields per override tick:

| Field | Override value during engagement ramp |
|---|---|
| `joint_pos_mj` | clamped per-tick from previous forwarded pose (unchanged from follow-up 9) |
| `joint_pos_mj_future` (9 slots) | **broadcast of the clamped current jpos** — flat future, no motion |
| `joint_vel_mj_future` (9 slots) | **zeros** — cancels velocity prediction |

After the engagement-ramp window completes (250 ticks = 5 s), the operator's actual future window flows through untouched. During the ramp, the policy reads "operator wants to hold this clamped pose; no future motion" → no future-driven slam.

The kept-untouched fields ensure no other behaviour regresses:

- `root_quat_xyzw_future` — still flows through (window-mode root-frame prediction needs it; flattening would lose heading tracking)
- Hand joints, motion_token, frame_index — fully transparent (operator's finger commands must work mid-ramp)

### Why broadcast instead of per-slot clamp

We considered three options for the future window during engagement:

1. **Per-slot clamp** (each future slot at most `max_step * (k+1)` from anchor): preserves operator's intended future motion direction but tracks 9 separate clamp states; complex, easy to refactor wrong.
2. **Drop future fields entirely**: would force the deploy to fall back to single-frame mode (`has_future_window_=false`), but stripping fields requires re-packing the header, which defeats the surgical-splice design.
3. **Broadcast clamped current** (chosen): simplest invariant — "during engagement, the future == clamped current, no velocity". Provably no future-driven motion. The 5 s ramp window is short enough that suppressing operator's future prediction during it has negligible UX cost.

### Single-field fallback

Kept `rebuild_msg_with_jpos_override` as a back-compat wrapper around `rebuild_msg_with_field_overrides`. When an override frame doesn't carry the full v5 future window (e.g. a legacy v4 token-only frame from an older publisher), the multi-field rebuild returns `None` and the call site falls back to clamping just `joint_pos_mj`. Better than verbatim forward; still safe for v4-only producers.

### Test regression pins (2 new)

- `test_rebuild_msg_with_field_overrides_flattens_future_window` — round-trip: pack a v5 frame, multi-field-rebuild with new jpos + flat future + zero vel, assert the three target fields match and all other fields (root_quat_future, hands, frame_index) stay byte-identical. Also pins the negative case (unmatched override key returns `None`).
- `test_proxy_engagement_clamp_flattens_future_window_in_source` — pins that the engagement clamp call site invokes `rebuild_msg_with_field_overrides` with all three field keys (`joint_pos_mj`, `joint_pos_mj_future`, `joint_vel_mj_future`) and that `flat_future` is broadcast from `clamped_jpos` (not from the operator's raw future, which would defeat the fix).

### Operator runbook (no command change)

Same as follow-up 9. Restart the VLA runtime to pick up the multi-field clamp; the engagement ramp now bounds both the current pose AND the future-window prediction.

## 2026-06-10 (afternoon, follow-up 5b) — Recorder silently zeroing finger commands at 50 Hz

The operator confirmed at 15:07 that the body slam was fixed:

> "looks better. fingers not working again."

The manager telemetry showed it was producing non-zero hand commands (`hand_q|L|=3.519` when triggers squeezed; `hand_q|L|=1.002` at the open-hand rest pose). But the recorder telemetry showed `hand|L|=0.000(manager)` on **every** status print — meaning the recorder thought it had received hand state from the manager but the value was always 0.

### Diagnosis

Walked the recorder's subscribe-mode message handlers ([`x2_dataset_recorder.py`](../../../../gear_sonic/utils/teleop/x2_dataset_recorder.py) lines 920–1043):

The recorder has TWO subscribe threads in teleop mode, polled side-by-side at 50 Hz each:

1. **Planner SUB** (port 5565, topic `body_pose`) → `_handle_body_pose_msg`
2. **Manager SUB** (port 5564, topic `hand_finger_cmd` among others) → `_handle_arm_and_hands_msg`

Both threads write to the same shared state slot via `update_hand_finger_cmd`. The manager's handler writes the operator's real finger commands. The body_pose handler **ALSO** updates the hand slot — extracting `left_hand_joints` / `right_hand_joints` from the planner's payload.

The planner's body_pose payload ([`state_machine.py:1105`](../../../../gear_sonic/utils/planner/state_machine.py)):

```python
"left_hand_joints":  np.zeros(hand_dof, dtype=np.float32),
"right_hand_joints": np.zeros(hand_dof, dtype=np.float32),
```

So the planner publishes all-zero hand fields in every body_pose tick at 50 Hz, for legacy wire-format compatibility. The recorder's body_pose handler **silently overwrote the manager's just-written non-zero values with zeros** within ~20 ms. The race-condition winner was always the planner (its zeros were published at the same 50 Hz as the manager's real values, and the threads were interleaved). Net effect: the recorder's `_left_hand_q` slot effectively never held a non-zero value, the recorder's published `'pose'` frame on :5560 had zero `left_hand_joints` / `right_hand_joints`, OmniHand received zero finger targets, **fingers never moved**.

This bug was present BEFORE follow-up 9/9b — it had been silently dropping operator finger commands all along; the body-slam fixes just made it visible by removing the other failure modes.

### The kicker

The existing test [`test_handle_body_pose_msg_planner_payload_leaves_hands_untouched`](../../../../tests/test_recorder_subscribe_mode.py) had **the right intent in its name and docstring** but the wrong assertion:

```python
"""Decoding the planner payload must NOT touch the existing hand slot
(since the manager's hand frame is the source of truth in that pipeline)."""

# ... seeded with manager-supplied hand pose 0.42 ...

# Zero-shaped hand was forwarded (since planner does include the
# field, even at zeros). This matches the existing fan-in behaviour;
# the loop then merges with manager hand if it arrives later.
np.testing.assert_allclose(snap["left_hand_q"], np.zeros(...))  # ← bug
```

The test verified the BUG (planner's zeros silently overwriting seeded manager values) while documenting the CORRECT behaviour (planner payload must not touch hand slot). Classic test-the-implementation-not-the-spec failure mode; the docstring was the source of truth, the assertion was a regression pin on the bug.

### Fix

Gated the embedded-hand-joints extraction in `_handle_body_pose_msg` on a new `vla_mode: bool = False` keyword argument:

```python
if (
    vla_mode
    and "left_hand_joints" in fields
    and "right_hand_joints" in fields
):
    # ... shape gate + state.update_hand_finger_cmd(lh, rh)
```

The flag is plumbed through from the subscribe thread (which already has `vla_mode` as a parameter — controls whether to connect to the manager URL at all):

| Mode | Hand source of truth | Bridge embeds hands? | Planner stamps zeros? | `vla_mode` flag |
|---|---|---|---|---|
| Teleop (`--no-vla`) | manager's `hand_finger_cmd` on :5564 | no | YES (gated out by `vla_mode=False`) | `False` |
| VLA (bridge running) | bridge's `pose` payload (embedded) | yes | n/a (recorder skips planner SUB entirely) | `True` |

After the gate, the manager's `hand_finger_cmd` writes are the sole producer of `_left_hand_q` / `_right_hand_q` in teleop mode. The planner's zero hand fields are decoded but ignored.

### Regression pins (3 updates / 1 new)

[`tests/test_recorder_subscribe_mode.py`](../../../../tests/test_recorder_subscribe_mode.py):

- `test_handle_body_pose_msg_extracts_vla_hand_joints` — updated to pass `vla_mode=True` explicitly so the VLA-mode extraction path is still covered.
- `test_handle_body_pose_msg_planner_payload_leaves_hands_untouched` — **assertion flipped** from "manager's seeded values get overwritten with planner zeros" to "manager's seeded values survive the planner update". Docstring updated to note the previous assertion encoded the bug.
- New `test_handle_body_pose_msg_planner_mode_ignores_nonzero_hands_too` — symmetric pin: even if a FUTURE planner starts publishing non-zero hand fields, teleop mode still ignores them. Prevents a planner change from silently re-introducing the race.

### Operator runbook

Restart the recorder (or the full quest3 planner stack) to pick up the fix. No command-line changes needed.

### Why this took 4 hours to find

The recorder's status print runs every 5 s and reads `_left_hand_q` at one instant — so even if the manager's writes occasionally won the race, the print sampled at a single point in time. The user always saw zeros, the manager log always showed non-zero, and they looked like contradictory truths instead of a 50-Hz race condition. The fix that surfaced the bug was the body-slam smoothing (follow-up 9/9b): once the body was clearly moving smoothly under operator override, the finger silence became diagnostic on its own rather than "another thing not working alongside body slam".

## 2026-06-10 (evening, follow-up 10) — Keep arms in place across OFF → ON during a VLA run

The operator's report after follow-up 9b landed:

> *"fixed now. can you do one more thing — when teleop is activated with a b x y from OFF to ON, how can we keep the arms from going to default pose, when we are in the middle of vla run. any extra arg to keep the arms in place during turn off and turn on?"*

So the body-slam was gone (proxy engagement clamp doing its job) but the arms still **drifted** to the X2 neutral stand pose over the engagement ramp window. That drift was the proxy faithfully ramping from VLA's pose `X` down to the operator's freeze `N_neutral`. The fix has to land on the **manager** side: the operator's freeze should be `X` (the wire's current pose), not `N_neutral`.

### Root cause

`Quest3ManagerX2._on_mode_transition` (line ~1373) explicitly snaps the arm + hand freeze caches to neutral on every OFF → non-OFF transition:

```python
if transition.previous == StreamMode.OFF:
    self._publish_planner_cmd(LocomotionCmd("idle", "default"))
    self._frozen_left_arm_q = self._retargeter._teleop.left_neutral_q
    self._frozen_right_arm_q = self._retargeter._teleop.right_neutral_q
    self._frozen_left_hand_q = np.zeros(10, dtype=np.float64)
    self._frozen_right_hand_q = np.zeros(10, dtype=np.float64)
```

That snap is the **right call** for cold-start teleop (the prior session's frozen values might be a mid-grasp wrist twist that the wrist-bypass=ik then "sticks to" forever). It's the **wrong call** mid-VLA-run because the recorder forwards the manager's frozen arms into the `pose` PUB, the proxy engages override, and the proxy's engagement clamp (follow-up 9b) walks the wire smoothly from `X` to `N_neutral` over ~5 seconds. The deploy follows the wire faithfully; the arms drift to the parking pose.

### Fix

The manager gains a single boolean opt-in: **`--preserve-arms-on-engage`**. When set, the manager spawns a SUB that connects to the canonical wire driving the deploy — the `x2_pose_proxy` downstream PUB. Host / port default to `tcp://127.0.0.1:5558` (the proxy's standard downstream on loopback) so the common SIM-on-PC1 case only needs the boolean; for split-topology / real-robot the wrapper auto-fans `--pc2-host` into the SUB host. A background thread decodes every frame (one 1280-byte JSON header parse + a few `np.frombuffer` calls per frame, ~50 Hz, negligible CPU) and caches the latest `joint_pos_mj` + `left_hand_joints` + `right_hand_joints` under a lock.

The flag preserves **both arms and hands** as a single concept. The user opted into "keep the robot where it is across the takeover boundary" — preserving the arms while force-opening the fingers would break that promise on every transition where VLA was holding something. The operator can still open fingers normally via the VR trigger once engaged, so there's no need for a separate hand-only opt-out.

On every OFF → non-OFF transition the manager now calls `_resolve_engage_freeze()` which returns either:

- `source="wire"` + arm slices `jpos[15:22]` / `jpos[22:29]` (canonical MJ layout, same as the recorder's body assembly) **plus** the cached `*_hand_joints` (when present in the wire frame) when the cache is configured AND non-empty AND fresh (`engage_pose_sub_max_age_ms`, default 200 ms = 10 ticks @ 50 Hz), OR
- `source="neutral"` + all-`None` otherwise (SUB disabled, cache empty, frame stale, or `jpos` shorter than 29).

The snap site applies both arm and hand slices in one shot; a per-side missing hand cache falls back to fingers-open just for that side (the other side, and the arms, still come through from the wire):

```python
wire_left, wire_right, wire_lhand, wire_rhand, src = self._resolve_engage_freeze()
if src == "wire" and wire_left is not None and wire_right is not None:
    self._frozen_left_arm_q = wire_left
    self._frozen_right_arm_q = wire_right
    self._frozen_left_hand_q  = wire_lhand if wire_lhand is not None else np.zeros(10)
    self._frozen_right_hand_q = wire_rhand if wire_rhand is not None else np.zeros(10)
else:
    # legacy neutral snap (arms=neutral, hands=fingers-open)
    ...
```

With the SUB configured at `tcp://127.0.0.1:5558`, the engagement timeline collapses to:

| Tick | Manager `arm_targets` | Proxy wire | Robot pose |
|---|---|---|---|
| t-1 | (held over from prior OFF state) | VLA pose `X` (LIVE) | `X` |
| t (A+B+X+Y → LOCO) | snap to `X` (wire cache) | engagement clamp activates with `last_upstream_jpos = X`, anchor = `X` | `X` |
| t+1..t+5 | `X` (operator hasn't engaged A yet) | proxy clamps operator's `X` → wire `X` (no delta to ramp) | `X` |
| t+N (operator engages A) | IK output from retargeter takes over freeze | proxy continues clamping; ramp window may still be running | smoothly slides from `X` to IK output |

The deploy never sees a discontinuity. The arms stay exactly where VLA had them at the moment of the chord.

### One flag, both arms and hands

The two-flag surface (`--preserve-arms-on-engage` + `--engage-preserve-hands`) was retired after the user pushed back on the split: "we have two options now? why?" The operator's mental model is "keep the robot where it is across the takeover" — splitting that into arm-only vs arm+hand was a false dichotomy. If VLA had inherited a stale grasp, the operator can still release the fingers with a relaxed VR trigger pull the moment they engage; the manager's `_publish_hand_finger_cmd` path is unaffected by the freeze (the freeze only seeds the first frame after the transition; subsequent ticks track the retargeter's live output). So preserving hands by default costs nothing safety-wise and avoids the surprise drop in the common case where VLA was actually holding the can.

### Files touched (follow-up 10)

[`gear_sonic/scripts/quest3_manager_x2.py`](../../../../gear_sonic/scripts/quest3_manager_x2.py):

- `ManagerConfig`: five new fields (`preserve_arms_on_engage` + `engage_pose_sub_{host,port,topic,max_age_ms}`). The port defaults to 5558 (the canonical proxy downstream) so the opt-in is genuinely a single boolean for the SIM-on-PC1 case.
- New SUB socket + background thread (`_engage_pose_loop`) using the same `unpack_message` decoder the recorder uses for packed-pose frames. Locked cache (`_engage_pose_jpos`, `_engage_pose_left_hand`, `_engage_pose_right_hand`, `_engage_pose_last_ts`).
- New helper `_resolve_engage_freeze()` returning `(left_arm, right_arm, left_hand, right_hand, source)`. `source ∈ {"wire", "neutral"}` so log messages can surface which branch fired.
- OFF → non-OFF snap site re-wired: try the wire-cache path first, fall through to the legacy neutral snap when unconfigured / stale / short. Both branches log distinctly so a one-line `grep` against `manager.log` tells you which path won every transition.
- CLI: new `engage-pose preservation` argparse group. Single flag `--preserve-arms-on-engage` (single boolean) preserves arms AND hands. Advanced overrides `--engage-pose-sub-host`, `--engage-pose-sub-port`, `--engage-pose-sub-topic`, `--engage-pose-sub-max-age-ms` are all no-ops unless the boolean is set.

[`gear_sonic/scripts/run_x2_quest3_planner_stack.sh`](../../../../gear_sonic/scripts/run_x2_quest3_planner_stack.sh):

- Five new env vars (`PRESERVE_ARMS_ON_ENGAGE` + `ENGAGE_POSE_SUB_{HOST,PORT,TOPIC,MAX_AGE_MS}`).
- Six new CLI flags (`--preserve-arms-on-engage` + `--no-preserve-arms-on-engage` + four `--engage-pose-sub-*` overrides).
- The forwarding block in **both** manager spawn paths (non-VLA at ~line 2448 and VLA at ~line 2889) is gated on `PRESERVE_ARMS_ON_ENGAGE -eq 1` so the four port/host/topic/age args don't leak unless the boolean is on. Numeric-guarded port (rejects typos at the wrapper layer, not at the manager argparse). One-liner status log so the manager startup banner surfaces "engage-pose preservation ON -> ..." with the configured host / port / topic.
- `--pc2-host` auto-fans `ENGAGE_POSE_SUB_HOST` to PC2's IP (mirrors how it already auto-fans `X2_DEBUG_BRIDGE_HOST`, `REMOTE_DEPLOY_HOST`, and `CAMERA_HOST`), so the real-robot operator's command stays a single `--preserve-arms-on-engage` flag on top of their existing `--pc2-host ...`.
- Usage banner updated to advertise the new flag.

### Tests added (follow-up 10)

[`tests/test_quest3_manager_engage_pose.py`](../../../../tests/test_quest3_manager_engage_pose.py) — six new pins:

- `test_resolve_engage_freeze_disabled_returns_neutral` — SUB-off boots the manager normally and returns `source="neutral"` so legacy users see no behaviour change.
- `test_resolve_engage_freeze_stale_cache_falls_back_to_neutral` — populated-but-old cache (`max_age_ms=50`, frame inserted with `last_ts = now - 1s`) correctly rejects and falls back. Catches a regression where the freshness check is dropped or inverted.
- `test_resolve_engage_freeze_fresh_cache_returns_arms_and_hands` — direct cache injection + `_resolve_engage_freeze` returns `jpos[15:22]` / `jpos[22:29]` AND both cached hand-joint vectors. Pins the single-flag UX (arms + hands ride together).
- `test_resolve_engage_freeze_missing_hands_falls_back_per_side` — when one side's `*_hand_joints` is missing from the wire frame, that side returns `None` (caller fills in fingers-open) while the other side and both arms still come through.
- `test_resolve_engage_freeze_short_jpos_falls_back_to_neutral` — defensive: a malformed wire frame with `jpos.shape[0] < 29` MUST fall back rather than raise `IndexError`.
- `test_engage_pose_sub_decodes_and_caches_real_zmq_frame` — end-to-end: real ZMQ PUB on loopback, real `pack_pose_message` payload, real background SUB thread, real decoder. Verifies the full path the operator will exercise in production.

[`tests/test_run_x2_quest3_planner_stack_cli.py`](../../../../tests/test_run_x2_quest3_planner_stack_cli.py) — two new pins:

- `test_wrapper_accepts_preserve_arms_on_engage_cli_flag` — the new CLI flag reaches `--validate-only` without falling into the unknown-arg branch.
- `test_wrapper_forwards_preserve_arms_args_to_manager_in_both_branches` — source-level pin: the `PRESERVE_ARMS_ON_ENGAGE -eq 1` gate is in place, the numeric port guard survives, the canonical 6-arg `MANAGER_ARGS+=(...)` block (`--preserve-arms-on-engage` + four overrides) appears exactly **twice** (once per spawn branch), and the retired `--engage-preserve-hands` / `ENGAGE_PRESERVE_HANDS` surface stays gone (catches a future regression that would re-introduce the two-flag confusion).

### Operator runbook (follow-up 10)

Two terminals as before. The only change is one flag on the planner stack invocation — no port number to remember, no relationship to the existing `--pose-port` recorder wire:

**Terminal 1 — planner + manager + recorder with engage-pose preservation:**

```bash
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --preserve-arms-on-engage
```

Or via env var (handy if you set it once in your shell rc):

```bash
PRESERVE_ARMS_ON_ENGAGE=1 ./gear_sonic/scripts/run_x2_quest3_planner_stack.sh
```

For real-robot / split-topology you don't need to add anything: `--pc2-host <PC2_IP>` (which you already pass) auto-targets `ENGAGE_POSE_SUB_HOST` at PC2. The port (5558) and topic (`pose`) default to the canonical proxy-downstream wire; only override if you started the proxy with a custom `--downstream-port`.

**Terminal 2 — VLA runtime (unchanged):**

```bash
./gear_sonic/scripts/run_x2_vla_runtime.sh \
    --vla-control-port 5559 \
    --pose-proxy-override-port 5560 \
    --model data/checkpoints/x2_pick_and_place_soda_can_n17_50k_v1/checkpoint-50000 \
    --motion-token-decoder /home/stickbot/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt \
    --robocasa-env X2PickPlaceFromCounterToBoxL \
    --prompt "pick up the mini soda can with your left hand and place it in the open black container on the right" \
    --sim-viewer --sim-with-omnihand
```

The runtime's proxy publishes the wire to `tcp://*:5558`, the manager subscribes there from T1. Verify on the next OFF → LOCOMOTION chord: the manager log should read

```
[manager-x2] OFF -> LOCOMOTION: snap arm+hand freeze to WIRE pose (arms=jpos[15:29], hands=from-wire)
```

instead of the legacy neutral-snap line. `hands=from-wire` means both `*_hand_joints` came through from the cached frame; `hands=partial-wire+fingers=open-fallback` would indicate one side was missing (fingers open just on that side); `hands=fingers=open-fallback` would mean neither side had hand fields (fingers open on both). If you see `engage-pose SUB stale/empty, falling back to NEUTRAL snap` either the SUB isn't getting frames (T2 not running yet, host mismatch on split-topology, firewall) or the cache is older than 200 ms (bump `--engage-pose-sub-max-age-ms` if your wire genuinely ticks slower than 5 Hz). Note this is independent of the existing `--pose-port` flag, which configures the **recorder's PUB port** (manager → recorder → proxy), not the **proxy's downstream port** (proxy → deploy, which the manager now reads from).

## What's left for the next session

- **Re-run with the safe wrist-bypass default.** Confirm the slam is gone and the fingers respond to VR triggers when the operator actually pulls them. The 12:26 recorder log shows `hand|L|=0.000(manager)` for the whole run during ARM_MANIPULATION — that's the manager publishing zero curls, i.e. no trigger pulls were detected. After the wrist swing is gone the operator should retry trigger pulls and confirm `hand|L|=N.NNN(manager)` with non-zero N appears in the recorder telemetry.
- **End-to-end sim smoke RE-RUN** with strict mode-gated engagement (`--pose-proxy-teleop-mode-port 5564`, the default when the planner stack is up on loopback). Expected: (1) zero engage events when the operator hasn't pressed A+B+X+Y, regardless of how they're holding the controller; (2) one engage event on the A+B+X+Y → LOCOMOTION/ARM_MANIPULATION transition; (3) zero release events as long as the operator stays in any non-OFF mode, even when holding the controller perfectly still; (4) one release event on the A+B+X+Y → OFF transition, with `release_pose` payload carrying body + hands; (5) clean fingers-follow-teleop during ARM_MANIPULATION; (6) smooth handoff with no visible pose reset.
- **End-to-end smoke on the real robot.** All component tests pass and both the bridge and proxy parse their new CLI cleanly, but the full takeover loop has only been exercised in the proxy subprocess smoke. The bridge cold-restart path needs at least one powered run to confirm:
  - The "hold at operator pose" wire is byte-identical to what the proxy's HOLD ladder is replaying (= no observable step at the OVERRIDE -> LIVE handoff).
  - The pinned `cold_restart_chunk_baseline` actually rejects pre-override chunks in the wild (the inference thread keeps producing chunks during override; we expect the first post-restart chunk to skip 0..N entries).
  - The 500 ms hold window is long enough at the production `--idle-stale-ms 300` (we use 100 ms in tests for speed); may need bumping to 750 ms or 1000 ms once we measure the real proxy -> bridge round-trip.
- **Telemetry export**: the proxy already counts `override_fwd / override_engaged / override_released` + the new `frozen(...) moving(...)` streak telemetry; the next iteration could surface those into `x2_debug_summary.json` for postmortem analysis.
- Optional: an automatic stuck-detector that synthesises override-engaged when the wire velocity flatlines + camera frames change (so the operator doesn't have to babysit). Explicitly punted — wants its own design pass.
