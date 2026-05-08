# Smooth MC ↔ policy handoff

How `deploy_x2.sh` hands the joint-command bus from the on-board Motion
Controller (MC) to the Sonic ONNX policy and back, without zero-torque
windows. Captures the architecture, the wire-level gotchas, and the
phase-2 direction (input-source arbitration) so we don't re-derive any
of this next time.

## Goal

The robot is gantry-supported but eventually runs free. Across the full
lifecycle of a deploy run

```
operator hits 'y'  →  policy CONTROL  →  RAMP_OUT  →  HOLD_FOR_MC  →  MC takes over
```

we want **active PD on every joint at every instant**. No `kp=0`
intervals, no "fall and catch" recovery, no motor whir from
dual-publisher conflict. Single unbroken chain of authority.

## What's running today (2026-05-03)

We *cycle* MC across each run: `stop_app` before deploy starts,
`start_app` afterwards. While MC is down, deploy is the sole publisher
on `/aima/hal/joint/{leg,waist,arm,head}/command`. While MC is up, deploy
is silent.

This is **phase 1** -- a working baseline that runs on the gantry. Phase
2 (below) replaces the cycle with input-source arbitration so MC stays
alive throughout.

### State machine (deploy node, phase 1)

```
STANDBY                 # cold-warm boot: build, ONNX load, ROS attach.
                        # writer is GATED OFF; bash drives the operator
                        # safety prompt while we silently warm up.
  ↓  start-trigger-sentinel touched (operator said 'y' AND bash POSTed
                                     stop_app on PC1)
INIT
  ↓  subscribers settled, all topics live
WAIT_FOR_CONTROL
  ↓  motion ready
CONTROL                 # 50 Hz policy publishes via 500 Hz writer
  ↓  max_duration elapsed
RAMP_OUT (2 s)          # interpolate target -> MC's STAND_DEFAULT pose
                        # gains stay at deploy-mode kp/kd (NOT lerped)
  ↓  ramp complete
HOLD_FOR_MC             # static publish of STAND_DEFAULT pose with MC-stand
                        # gains (loaded from x2_stand_default_pose.yaml).
                        # Touches HOLD_FOR_MC_SENTINEL so bash can
                        # POST start_app + start the escalator.
  ↓  HOLD_FOR_MC_EXIT_SENTINEL touched by bash
SHUTDOWN                # MC owns the bus alone in STAND_DEFAULT
```

### State machine (bash, phase 1)

```
build (colcon)                                    # stays at deploy-build
spawn deploy in STANDBY (writer gated off)        # silent warm-up
wait for ready-sentinel (deploy is ready)         # 30 s timeout
operator safety gate ('y' / 'n')
  ↓  'y'
POST stop_app on PC1                              # MC goes silent
touch start-trigger-sentinel                      # deploy enters INIT
  ↓
deploy CONTROL → RAMP_OUT → HOLD_FOR_MC → touches HOLD_FOR_MC_SENTINEL
  ↓
POST start_app on PC1                             # MC begins boot (~3-5 s)
spawn x2_mc_escalator.py &                        # 20 Hz hammer SetMcAction(JOINT_DEFAULT)
                                                  # success = GetMcAction reports JOINT_DEFAULT
  ↓  escalator success-sentinel touched
touch HOLD_FOR_MC_EXIT_SENTINEL                   # deploy releases on next OnControl tick
SetMcAction(STAND_DEFAULT) + verify via GetMcAction
  ↓
done. MC is in STAND_DEFAULT, robot held.
```

## Sentinel files

All sentinels live in `$RUN_LOG_DIR` if a per-run log dir was created,
else `/tmp/x2_*.<pid>.sentinel`.

| Sentinel | Created by | Read by | Meaning |
|---|---|---|---|
| `ready.sentinel` | deploy (in STANDBY, on first OnControl tick) | bash | "C++ binary is fully warmed up; safe to show the operator gate." |
| `start-trigger.sentinel` | bash (after `stop_app` returns) | deploy | "Operator confirmed; MC is stopped; you may publish." |
| `hold_for_mc.sentinel` | deploy (entering HOLD_FOR_MC) | bash | "Policy phase done; you may now POST start_app." |
| `hold_for_mc_exit.sentinel` | bash (after escalator confirms JOINT_DEFAULT) | deploy | "MC owns the bus; release on your next tick." |
| `mc_first_publish.sentinel` | deploy (takeover-detector callback fires) | bash (informational) | "MC's first publish has landed -- diagnostic timestamp." |
| `mc_escalator_ok.sentinel` | escalator (GetMcAction confirms target mode) | bash | "MC is *truly* in JOINT_DEFAULT, not just 'request was accepted'." |

## Wire-level gotchas (do not regress these)

### 1. Takeover-detector QoS

Subscribe with `rclcpp::SensorDataQoS()` (BEST_EFFORT, depth=10). MC
publishes BEST_EFFORT; a default RELIABLE QoS will silently drop every
message and the detector will never fire. Set
`ignore_local_publications=true` so we don't trigger on our own
publishes.

### 2. `SetMcAction` response codes are unreliable during MC boot

In the cold-boot window (~3-5 s after `start_app`), MC's `SetMcAction`
service comes up *before* its mode-arbitration subsystem is fully ready.
During that overlap MC accepts the request and replies `code=0`, but
**silently ignores it** and stays in `PASSIVE_DEFAULT`. Verified
empirically on 2026-05-03 (run `x2_run_20260503_213002`): bash logged
"STAND_DEFAULT confirmed" and the recorder showed MC stayed in
`PASSIVE_DEFAULT` for the full window.

**Rule**: `GetMcAction` is the only ground truth. The escalator and the
final `SetMcAction(STAND_DEFAULT)` both verify by polling
`GetMcAction` until the reported `action_desc` matches the requested
mode.

### 3. RAMP_OUT does NOT lerp gains

Lerping kp/kd toward MC-stand gains while the position target is also
moving by up to ~1.2 rad over 2 s produces a transiently
under-damped system chasing a moving setpoint = motor whir + ringing.
Verified the hard way; reverted to "lerp position only, hold deploy-mode
gains, snap to MC-stand gains at HOLD_FOR_MC entry where the position
error is ~0".

### 4. HOLD_FOR_MC needs MC-stand stiffness, not deploy gains

Deploy-mode gains were tuned for the *active* policy, not for a static
hold against gravity. Legs/ankles/waist felt soft when we used deploy
gains in HOLD_FOR_MC (e.g. waist pitch kp=14 vs MC's 40, knee kp=99 vs
MC's 150). The pose YAML (`configs/x2_stand_default_pose.yaml`) ships
the MC-stand kp/kd captured from a live `STAND_DEFAULT` run; deploy
loads it at startup and uses those values throughout HOLD_FOR_MC.

The kp step at HOLD_FOR_MC entry is benign because RAMP_OUT just landed
the pose at the matching joint angles -- position error is ~0, so the
torque kick from a kp step is small.

### 5. Bash's `ros2 service call` has ~300 ms of Python startup overhead

That's the entire reason `x2_mc_escalator.py` exists. It opens
`SetMcAction` + `GetMcAction` clients *once*, then fires at 20 Hz with
sub-ms RTT per call. A bash-only "loop and call mc_set_action" can fire
at most ~3 Hz because each iteration pays the rclpy startup cost.

## Diagnostics: what to read in the logs

Look for these lines in order during a clean run:

```
[handoff] MC start_app POSTed.
[post-handoff] escalator launched (pid X): hammering SetMcAction(JOINT_DEFAULT) at 20Hz
[escalator] GetMcAction first response: mode='PASSIVE_DEFAULT'
[escalator] GROUND-TRUTH SUCCESS: GetMcAction reports 'JOINT_DEFAULT' after N attempts
[post-handoff] -> JOINT_DEFAULT confirmed by escalator (active PD; releasing deploy now).
[handoff] exit-sentinel touched (MC is in JOINT_DEFAULT); deploy will release the bus on its next tick.
HOLD_FOR_MC: exit-sentinel touched after X.XXs -> shutting down.
[post-handoff] -> STAND_DEFAULT confirmed via mc_get_action.
[handoff] deploy exited with code 0.
```

If MC ends up in `PASSIVE_DEFAULT`, look at `mc_escalator.log`:
- if `GetMcAction first response` never appears -> MC's services never came up; check `start_app` POST + PC1 health
- if many `set_codes={0: N}` but no `modes_seen` includes `JOINT_DEFAULT` -> the silent-ignore boot bug is back; either the gantry pose violates MC's posture-detector preconditions or there's a deeper API regression. Switch to the mobile app to escalate manually.

The recorder's MC mode timeline is the post-mortem ground truth. If the
final segment is anything other than `STAND_DEFAULT`, the handoff did
not complete -- bash logs are not authoritative. Cross-reference
`cmd_kp_*` and `state_eff_*` traces to confirm whether deploy or MC was
publishing during each segment.

## Validated handoff numbers (run-to-run)

| Date | Motion | Notes | `JOINT_DEFAULT` dwell | Final mode |
|---|---|---|---:|---|
| 2026-05-03 (gestures) | `standing_gestures_v1`, iter-16k | Pre-escalator (one-shot bash `mc_set_action`) | 1.60 s | STAND_DEFAULT |
| 2026-05-03 (walk) | `casual_walk_v1`, iter-22k | Persistent-client escalator + GetMcAction ground-truth | **0.20 s** | STAND_DEFAULT |

The 8× drop in dwell time is the dual-publisher whir window
proportionally shrinking. With deploy still publishing during the
`JOINT_DEFAULT` window (until exit-sentinel fires), shorter dwell ==
shorter time both sides drive the bus. 0.20 s is operator-imperceptible
in the audio signature; 1.60 s was the "couple of seconds of whirring"
the operator reported on 2026-05-03 morning.

## Phase 2 direction: input-source arbitration

We have a *better* path that we deliberately deferred. MC exposes an
`InputManager` with priority-based source arbitration via
`SetMcInputSource`. Empirically validated 2026-05-03 (probe
`mc_input_source_20260503_090334`):

| | Phase 1 (today) | Phase 2 (planned) |
|---|---|---|
| MC across the run | restarted | stays alive in STAND_DEFAULT |
| Pre-deploy | `stop_app` HTTP POST | `SetMcInputSource(MODIFY+ENABLE, name=pnc, prio=40, timeout=200ms)` |
| During deploy | MC dead, deploy alone | MC + deploy both alive; arbitration says deploy wins |
| Post-deploy | `start_app` + boot wait + escalator | `SetMcInputSource(DISABLE, pnc)`; MC's 200 ms watchdog reclaims |
| Zero-torque window | ~1 s during MC PASSIVE boot phase | none |
| Off-gantry safe? | no (process crash = no controller) | yes (200 ms watchdog auto-reclaims) |

Phase 2 needs ~80 lines of C++ (`CLAIM_BUS` / `RELEASE_BUS` states +
service clients) and ~50 lines of bash. The probes already validated
that:

- `SetMcInputSource(MODIFY/ENABLE/DISABLE, pnc)` works on the live MC
  (every reply `code=0`)
- Publishing on `/aima/hal/joint/*/command` *is* the heartbeat -- no
  separate keep-alive RPC
- The 200 ms `expiration_time` watchdog reclaims silently when we
  stop publishing
- `mc.yaml`'s `pnc` slot (priority 40) is the canonical autonomy slot;
  `ADD` of a fresh source name is rejected, so we use `pnc`

See `scratch/probes/mc_introspect_20260502_233237/FINDINGS.md` and
`scratch/probes/mc_input_source_20260503_090334/FINDINGS_addendum.md`
for the full empirical validation.
