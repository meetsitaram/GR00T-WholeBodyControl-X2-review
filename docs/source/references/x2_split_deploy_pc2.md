# X2 split-topology deploy: PC2 + laptop architecture

This document describes the **split-topology** deployment, where the
hard-real-time C++ deploy and motor-state monitoring run on the
robot's on-board Jetson Orin NX (PC2), while the operator-side stack
(Quest 3 manager, planner, recorder) runs on a laptop. The two
machines communicate over ZMQ on the WiFi LAN.

The non-split topology (everything on the laptop) is documented in
`x2_quest3_planner_stack_architecture.md`.

## Why split

The motivating incident (2026-05-14) was a robot freeze where the
laptop's WebXR manager dropped a few frames, the pose stream stalled
~150 ms, and the C++ deploy on the laptop kept the most recent
target latched while it waited. Because the laptop is also the only
machine that can issue the safety-recovery commands, an operator
panicking through a 1-second freeze had no way to safely halt the
robot from outside the (frozen) chain.

Splitting the deploy onto PC2 fixes the root cause:

1. The C++ deploy runs adjacent to the motor controller (PC1) on the
   wired SDK ethernet. Even if the WiFi LAN goes away entirely, the
   deploy keeps the bus alive.
2. A **PC2-local pose proxy** (`x2_pose_proxy.py`) sits between the
   laptop publisher and the C++ deploy. It SUBs to the laptop's
   `pose` stream and re-PUBs to PC2 loopback. When the laptop wire
   is fresh, frames are forwarded byte-for-byte. When the laptop
   goes silent for >100 ms (wifi drop, laptop crash, planner not
   yet started), the proxy switches to publishing `idle_stand`
   frames from a locally-staged `idle_stand.x2m2` so the deploy
   always sees a continuous 50 Hz reference stream. The same idle
   clip is what `live_vla_publish_motion_token.py --no-policy`
   would publish on the laptop in idle mode -- one consistent
   reference distribution for the policy.
3. With the proxy in play, the legacy **pose-ref starvation
   watchdog** (`safety.hpp`) is no longer needed and is
   automatically disabled via `--disable-pose-ref-watchdog` from
   the daemon wrapper. The historical `SAFE_IDLE` path remains in
   the binary for non-split deployments and for legacy `--no-pose-proxy`
   runs, where it still pins targets to `default_angles` with 4×
   damping on starvation.
4. Recovery from wifi blips is now **automatic**: the proxy bridges
   the silence, the policy keeps tracking idle_stand, and operator
   pose resumes the instant the laptop publisher comes back. The
   `pose_resume` chord (A+B on the Quest 3 right controller, held
   1 s, X+Y NOT pressed) is still wired in case a future
   non-proxy path needs it, but the operator does not need to
   issue it for a wifi blip in proxy mode.

Independently of the freeze recovery, splitting also lets the
**motor-state monitor** subscribe to MC's `JointStateArray` and
`JointCommandArray` topics directly on PC2, where DDS discovery is
guaranteed to land (the laptop's WiFi DDS reach is unreliable). The
monitor writes structured JSONL forensic logs and PUBs a compact
summary back to the laptop sidecar at 1 Hz so every event lands in
both places.

## Process layout

```
                         ┌────────── Laptop (operator side) ──────────┐
                         │                                            │
   Quest 3 (WebXR)  ──── │  quest3_manager_x2  ┐                       │
                         │  --resume-pub-port  │ planner_cmd:5563      │
                         │  --motor-monitor-* ─┘                       │
                         │     │  ▲ ┌───────── motor_monitor:5567 (SUB)│
                         │     ▼  │ │   ┌─── pose_resume:5566 (PUB)    │
                         │  x2_heuristic_planner ─body_pose:5565──►   │
                         │     │                                       │
                         │     ▼                                       │
                         │  record_x2_dataset ─pose:5556 (PUB)─►──────┐│
                         │     ▲ x2_debug:5557 (SUB)──────────────────┘│
                         └────────────────────────────────────────────┘
                                            ▲ ▲ ▲
                                            │ │ │  WiFi LAN
                                            ▼ ▼ ▼
                         ┌──── PC2 (Jetson Orin NX, on-bot) ──────────┐
                         │                                             │
                         │ tmux x2_pose_proxy:                         │
                         │   x2_pose_proxy.py                          │
                         │     --upstream-host LAPTOP --upstream-port 5556│
                         │     --downstream-port 5558                  │
                         │     --idle-x2m2 .../data/idle_stand.x2m2    │
                         │     --idle-stale-ms 100                     │
                         │   (forwards laptop frames when fresh;       │
                         │    publishes idle_stand frames when wire    │
                         │    is silent > 100 ms; the deploy SUBs      │
                         │    here, not directly to the laptop)        │
                         │                                             │
                         │ tmux x2_deploy:                             │
                         │   deploy_x2.sh onbot                        │
                         │     --vla --vla-zmq-host localhost --vla-zmq-port 5558│
                         │     --vla-resume-host LAPTOP --vla-resume-port 5566│
                         │     --vla-debug-port 5557                   │
                         │     --deploy-extra-arg --disable-pose-ref-watchdog│
                         │     --log-dir /home/run/getsolo/log/        │
                         │       deploy_<ts>/                          │
                         │   (Y/n safety gate, MC stop/start via PC1 EM│
                         │    HTTP or aima em CLI fallback, sentinel   │
                         │    handoff, RAMP_OUT trap, all native on PC2)│
                         │                                             │
                         │ tmux x2_hand_bridge:                        │
                         │   x2_hand_zmq_to_aimdk_bridge.py            │
                         │     --zmq-host LAPTOP --zmq-port 5564       │
                         │                                             │
                         │ tmux x2_motor_monitor:                      │
                         │   x2_motor_monitor.py                       │
                         │     --jsonl /home/run/getsolo/log/          │
                         │       motor_monitor.jsonl                   │
                         │     --zmq-port 5567                         │
                         │     ─────────► /aima/hal/joint/*/{state,command}│
                         │     ─────────► /x2/em/.../GetMcAction (1Hz) │
                         └────────────────────────────────────────────┘
                                            ▲
                                            │ wired SDK ethernet
                                            ▼
                          PC1 (10.0.1.40, motion controller, EM HTTP)
```

## Wire contract

| ZMQ topic | PUB | SUB | Port | Notes |
|---|---|---|---|---|
| `pose` | recorder (laptop) | pose proxy (PC2) | 5556 | 50 Hz reference from operator stack |
| `pose` | pose proxy (PC2) | deploy (PC2) | 5558 | proxied: forwarded when laptop is fresh, idle_stand replay when laptop silent >100 ms |
| `x2_debug` | deploy (PC2) | recorder (laptop) | 5557 | per-tick telemetry; v5 carries `pose_ref_age_s`, `in_safe_idle`, `mc_action_mode` etc. |
| `body_pose` | planner (laptop) | recorder (laptop) | 5565 | local merge into `pose` |
| `arm_targets`, `hand_finger_cmd`, `stream_mode` | manager (laptop) | recorder + hand bridge | 5564 | multi-topic |
| `pose_resume` | manager (laptop) | deploy (PC2) | 5566 | A+B chord; legacy SAFE_IDLE exit (not used in proxy mode -- proxy auto-recovers) |
| `motor_monitor` | monitor (PC2) | manager (laptop) | 5567 | JSONL summary every 1 s, written into `manager_sidecar.jsonl` |

The `--remote-deploy HOST` flag on `run_x2_quest3_planner_stack.sh`
auto-redirects the recorder's `x2_debug` SUB and the manager's
`motor_monitor` SUB at `HOST`, and enables the manager's `pose_resume`
PUB. Otherwise the wire layout is identical to the all-on-laptop path
documented in `x2_quest3_planner_stack_architecture.md`.

## Safety contract

1. **PC2 pose proxy** (`x2_pose_proxy.py`) is the first line of
   defense. It sits between the laptop publisher and the deploy on
   PC2 loopback, forwards laptop frames byte-for-byte when fresh,
   and seamlessly switches to publishing `idle_stand` frames from
   the locally-staged `idle_stand.x2m2` when the laptop is silent
   for >100 ms. The deploy's pose-ref watchdog is automatically
   disabled in proxy mode (`--disable-pose-ref-watchdog`); the
   proxy guarantees the wire is never silent from the deploy's
   perspective, so the watchdog (and the historical `SAFE_IDLE`
   whir bug that triggered when it tripped onto a robot already
   leaning away from `default_angles`) becomes unreachable. A
   laptop wifi blip or planner restart is invisible to the policy
   -- it just sees the operator's pose temporarily replaced by an
   idle stand reference, and back again the instant the laptop
   recovers.
2. **PoseRefStarvationWatchdog** (in `safety.hpp/cpp`) remains in
   the binary for non-proxy paths (`--no-pose-proxy` on the daemon
   wrapper, or any non-split deployment). When active, it runs
   every tick: if `now - last_pose_rx > 0.5 s` it trips and
   transitions `CONTROL -> SAFE_IDLE`, which holds a latched
   `SafeCommand` whose `target_pos_mj` is `default_angles` and
   whose `kd` is 4× the trained value. The 500 Hz writer keeps
   publishing this command so MC never sees a silent bus.
3. **Manual resume** (legacy / non-proxy path) requires *both* (a)
   `ReadyToResume()` (pose stream fresh for ≥1 s after the trip)
   and (b) a recent `pose_resume` message (Quest 3 A+B chord on
   the right controller, held 1 s, with X+Y NOT pressed -- avoids
   the existing A+B+X+Y engage chord). In proxy mode the operator
   does not need to issue this chord for a wifi blip; the proxy
   bridges the silence automatically.
4. **No auto-trips from the monitor.** The motor monitor is
   intentionally read-only; it logs events (tracking error spikes,
   limit proximity, MC mode changes, drive faults) but never issues
   a SetMcAction or any other torque-affecting call. The deploy's
   real-time safety path is the only authoritative actor on the
   bus.
5. **MC mode observability.** The deploy polls `GetMcAction` every
   second via the ROS service and republishes the value on
   `x2_debug` so postmortem can see the MC action transitions
   alongside everything else without an extra subscriber.

## Lifecycle

The complete day-to-day workflow:

```bash
# Sanity check PC2 + workspace + topics + ports.
./gear_sonic_deploy/scripts/pc2_preflight.sh

# Bring up the three tmux sessions on PC2.
# The deploy session shells through `deploy_x2.sh onbot` (PC2-native,
# staged by pc2_bringup.sh at /home/run/getsolo/gear_sonic_deploy/),
# so the operator gets the full safety scaffold:
#   - Y/n safety gate (operator confirms BEFORE MC is stopped)
#   - mc_em_post stop_app (HTTP first, then aima em stop-app mc CLI)
#   - sentinel-driven STANDBY -> CONTROL handoff (default_angles
#     hold while operator considers; ~20 ms to CONTROL after Y)
#   - RAMP_OUT -> HOLD_FOR_MC -> start_app(mc) graceful shutdown
#     on Ctrl-C / SIGINT (robot ends in STAND_DEFAULT, no zero-torque)
# All orchestration runs ON PC2 in the deploy tmux pane, so it
# survives a laptop-side WiFi disconnect mid-run.
#
# No --motion: the deploy runs --vla (== --input-type=zmq), so the
# tokenizer reference window comes from the laptop's ZMQ pose stream
# rather than a PklMotionReference. SAFE_IDLE always uses the
# hardcoded default_angles constant, so no .x2m2 fallback is needed
# either.
#
# Pass --attach to land directly on the deploy pane's Y/n prompt, or
# attach later with `attach deploy`. The in-repo default policy at
# data/policies/agibot_x2_sonic.onnx is staged to
# /home/run/getsolo/policies/agibot_x2_sonic.onnx by pc2_bringup.sh.
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml

# Verify they're up.
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh status

# Start the operator-side stack on the laptop.
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 0 --remote-deploy 10.0.1.41

# (Test the robot ...)

# After the run, pull logs back + analyse.
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh postmortem \
    --center-ts "2026-05-15T19:24:30" --window-s 30

# Stop the daemons. SIGINTs the deploy session, which triggers
# deploy_x2.sh's restart_mc_on_exit trap: RAMP_OUT -> HOLD_FOR_MC ->
# start_app(mc) -> SetMcAction(STAND_DEFAULT), all on PC2.
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop
```

### Equivalence to the old laptop-driven workflow

In split topology the `deploy_x2.sh onbot` invocation inside the
PC2 deploy pane is the same shape as the laptop-side
`deploy_x2.sh local` you used in non-split runs:

| Phase                    | Laptop-driven (local)         | Split (onbot on PC2)              |
| ------------------------ | ----------------------------- | --------------------------------- |
| Build                    | `colcon build` on laptop      | `colcon build` on PC2 (bringup)   |
| Preflight                | `x2_preflight.py` on laptop   | `x2_preflight.py` on PC2          |
| Y/n safety gate          | laptop terminal               | PC2 deploy tmux pane              |
| MC stop_app              | curl PC1 EM (HTTP)            | curl PC1 EM, fallback `aima em`   |
| Sentinel STANDBY handoff | sentinels in `scratch/runs/…` | sentinels under `$RUN_LOG_DIR/…`  |
| RAMP_OUT trap            | laptop bash trap              | PC2 bash trap (survives WiFi off) |
| Hand bridge              | spawned by `deploy_x2.sh`     | own tmux session (`x2_hand_bridge`)|
| Recorder                 | spawned by `deploy_x2.sh`     | runs on laptop with planner stack |

For initial PC2 setup see `docs/source/getting_started/pc2_jetson_bringup.md`.
For the motor monitor's JSONL schema + thresholds see
`docs/source/references/x2_motor_monitoring.md`.
