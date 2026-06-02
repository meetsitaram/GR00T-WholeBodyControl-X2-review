=============== do not auto edit this section ===============
### start local stack
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --no-deploy 
or 
./gear_sonic/scripts/run_x2_quest3_wholebody_walk.sh --no-deploy 

### start sonic on PC2 : Robogym Wifi
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml

### stop sonic
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32

### start sonic on PC2 : Express Auto Wifi
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.4.79 --laptop-host 192.168.4.91 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml

### stop sonic
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.4.79

=============== end of pure manual notes section ===============

## Topology A — split (laptop = planner stack; PC2 = deploy + idle proxy)

The PC2 side runs four tmux sessions: `x2_pose_proxy`, `x2_deploy`,
`x2_hand_bridge`, `x2_motor_monitor`. The deploy SUBs to the pose
proxy on PC2 loopback (not to the laptop directly) so the wire is
always flowing: the proxy forwards laptop pose frames when they
arrive and switches to local idle_stand frames within 100 ms if the
laptop goes silent (planner not yet started, wifi drop, laptop
crash). The C++ pose-ref starvation watchdog is automatically
disabled in this mode — the proxy makes it irrelevant.

### one-time per session: laptop publisher first, then PC2 daemons

`x2_deploy` will run a heuristic policy as soon as MC stops, so the
laptop publisher MUST be running before you hit the Y/n gate. With
`--no-deploy` the planner stack runs everything *except* the local
deploy and publishes pose frames on `tcp://*:5556` for PC2 to pick up:

```sh
# Terminal 1 (laptop) — start the planner stack; leaves pose flowing.
# --no-deploy skips the local deploy spawn (deploy runs on PC2 instead);
# no --vla-* flags are needed because we're using the heuristic planner +
# recorder path (recorder publishes pose on :5556 for PC2 to subscribe).
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --no-deploy
```

The deploy session shells through `deploy_x2.sh onbot` on PC2 (staged
under `/home/run/getsolo/gear_sonic_deploy/` by `pc2_bringup.sh`), so
inside the tmux pane you get the full safety scaffold:

  - Wire-freshness probe: SUBs to the deploy's actual pose input
    (`localhost:5558` in proxy mode) for 1 s right before the Y/n
    gate. If zero frames arrive, the launch aborts WITHOUT stopping
    MC — catches "proxy crashed" / "planner stack never started" /
    "wrong host" before the robot is committed. Skip with
    `--no-wire-probe` only if you really know why.
  - Y/n safety gate (operator confirms BEFORE MC stops)
  - MC stop_app via PC1 EM HTTP, with `aima em stop-app mc` CLI fallback
  - Sentinel-driven STANDBY → CONTROL handoff
  - RAMP_OUT → HOLD_FOR_MC → start_app(mc) on Ctrl-C or `stop`
  - All on PC2 — survives laptop WiFi disconnects

```sh
# Terminal 2 (laptop) — start the PC2 daemons and attach to the deploy prompt.
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.21 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml

# When you land on the prompt you'll see the deploy run a wire-freshness
# probe before showing the Y/n safety gate. Expected sequence:
#   [handoff] probing pose wire on tcp://localhost:5558 (topic=pose, 1.0s) ...
#   [handoff] wire probe OK: OK frames=50 dt_first_ms=...
#   SAFETY GATE -- STOP MC + LAUNCH POLICY
#   Stop MC and launch policy? [y/N]:
#
# If the probe says "wire probe FAILED ... SILENT frames=0", the deploy
# aborts WITHOUT stopping MC. Most common causes:
#   (1) pose proxy died on PC2 -> ./x2_pc2_daemons.sh logs proxy --pc2-host ...
#   (2) laptop planner stack not running -> start ./run_x2_quest3_planner_stack.sh
# Fix the upstream, then `./x2_pc2_daemons.sh start --attach ...` again.
#
# Type `y` + Enter at the safety gate to commit. Skip the operator
# confirmation (CI only) with `--no-confirm`; skip the wire probe with
# `--no-wire-probe` (do NOT use this on a live robot).

# reconnect later (WiFi blip, ssh dropped, etc.)
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh attach deploy \
    --pc2-host 192.168.86.21

# Ctrl-B then d (release Ctrl before d) detaches without killing.
# If your terminal eats Ctrl-B, you can also detach with:
#   ssh run@192.168.86.21 'tmux detach -s x2_deploy'
```

### one-time bringup (copy and rebuild)
```sh
git lfs pull --include="data/policies/agibot_x2_sonic.onnx"

./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host 192.168.86.21 
```

### reduce volumn on PC3
```sh
./gear_sonic_deploy/scripts/x2_pc3_audio.sh           # mute (default)
./gear_sonic_deploy/scripts/x2_pc3_audio.sh unmute
./gear_sonic_deploy/scripts/x2_pc3_audio.sh status
./gear_sonic_deploy/scripts/x2_pc3_audio.sh volume 60 # PCM to 60%
```

### ontime ssh key setup
```ssh
# Over the wire (SDK ethernet):
ssh-copy-id run@10.0.1.41
# Over the robogym WiFi (now that both are on it):
ssh-copy-id run@192.168.86.21
# Or by hostname/alias if you've put one in ~/.ssh/config
```

### pre-flight checks 
```sh
./gear_sonic_deploy/scripts/pc2_preflight.sh  \
   --pc2-host 192.168.86.21 --pc2-user run --pc1-host 10.0.1.40
```


===================

### terminal 1 (when on wire)
- until Ctrl-C (no auto-shutdown)
```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --no-deploy

```

### terminal 2 (when on wire)

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic_deploy/deploy_x2.sh local \
    --vla \
    --vla-zmq-host 127.0.0.1 --vla-zmq-port 5556 --vla-zmq-topic pose \
    --vla-debug-port 5557 --vla-debug-topic x2_debug \
    --model ./data/policies/agibot_x2_sonic.onnx \
    --wrist-bypass ik \
    --tuning-config gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml
```

### terminal 2 (on wifi)
```sh
# Discover / sanity-check the resolved IPs before starting:
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh print-env \
    --pc2-host 192.168.86.21 \
    --laptop-host 192.168.86.22

# Same flags work on every subcommand:
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start \
    --pc2-host 192.168.86.21 \
    --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml
```




### one-time PC2 bringup (per fresh Orin NX flash, or after `/home/run/getsolo` is wiped)

Lays down `/home/run/getsolo/{onnxruntime,venv,ws,gear_sonic_deploy,policies,log}`
and builds the C++ deploy. Idempotent, safe to re-run. Nothing lands
outside `/home/run/getsolo/`, the system Python stays untouched.

The default deploy policy is in-repo at `data/policies/agibot_x2_sonic.onnx`
(LFS-tracked, 56 MB), so `--model` resolves locally without any
`$HOME/x2_cloud_checkpoints/...` detour:

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl

# Once per Orin NX (so the script can SSH non-interactively):
ssh-copy-id run@10.0.1.41      # password 'run / 1' on a fresh flash

# Make sure LFS has pulled the real ONNX bytes (not the pointer file):
git lfs pull --include="data/policies/agibot_x2_sonic.onnx"

# Lay out + build (~80 s on Orin NX after onnxruntime download).
# Stages the .onnx at /home/run/getsolo/policies/agibot_x2_sonic.onnx
# (filename preserved by rsync).
./gear_sonic_deploy/scripts/pc2_bringup.sh \
    --pc2-host 10.0.1.41 \
    --model ./data/policies/agibot_x2_sonic.onnx
```

Add `--dry-run` for a preview, `--force-build` to wipe the colcon
overlay and rebuild from scratch, `--force-onnx` to redownload ONNX
Runtime.

### terminal 0 -- one-time preflight (run from laptop)

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic_deploy/scripts/pc2_preflight.sh
```

### terminal 1 -- start PC2 daemons (run from laptop)

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml

./gear_sonic_deploy/scripts/x2_pc2_daemons.sh status
```

Notes:

- No `--motion <x2m2>` is needed. The PC2 deploy runs with
  `--input-type zmq`, so its tokenizer reference window is fed by the
  laptop's Quest 3 manager / recorder over ZMQ. `SAFE_IDLE` always
  falls back to the hardcoded `default_angles` constant in
  `policy_parameters.hpp`, so no motion clip needs to be staged on PC2
  either.
- `--tuning` accepts the same YAML files as
  `deploy_x2.sh --tuning-config` (the daemon wrapper re-uses
  `scripts/tuning_config_to_args.py` to expand the preset into native
  C++ `--flag VALUE` tokens). Path is resolved on the laptop, not on
  PC2.
- All PC2-side artifacts live under `/home/run/getsolo/`
  (`onnxruntime/`, `venv/` with pyzmq, `ws/` colcon overlay,
  `policies/` for ONNX checkpoints, `log/` for runtime CSVs +
  JSONL). Override the root with `--prefix /other/path` (default is
  `/home/run/getsolo`). System Python, `/opt`, and the
  `/agibot/software/` tree are never written to.
- IP defaults assume SDK-ethernet (`PC2_HOST=10.0.1.41`,
  `PC1_HOST=10.0.1.40`, `LAPTOP_HOST=hostname -I`). To discover the
  WiFi IPs of both the laptop and PC2 in one shot **while still
  connected on the wire**, run the discovery script:

  ```sh
  ./gear_sonic_deploy/scripts/x2_discover_network.sh \
      --write-env-dir ~/.x2
  ```

  Output: a table of every IPv4 interface on the laptop and PC2
  (ethernet / wifi / virtual), the candidate routes pairing them by
  /24 prefix, a 1-ping reachability probe per route, and two
  copy/paste env blocks. With `--write-env-dir ~/.x2` it also writes
  `~/.x2/env.wired` and `~/.x2/env.wifi` ready to `source`.

  Prerequisite: one-time `ssh-copy-id run@10.0.1.41` so the script
  can read PC2's interface list non-interactively (password is
  `run / 1` on a fresh Orin NX). The discovery script is read-only
  and doesn't touch any running daemons.

  After that, switch network just by sourcing:

  ```sh
  source ~/.x2/env.wired   # or ~/.x2/env.wifi
  ./gear_sonic_deploy/scripts/x2_pc2_daemons.sh print-env  # sanity check
  ./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start \
      --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
      --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml
  ```

  (`PC1_HOST` rarely changes — PC1 is reached from PC2 over the
  robot's internal SDK ethernet, not over WiFi from the laptop.)

### terminal 2 -- operator-side stack (run from laptop)

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 0 \
    --pc2-host ${PC2_HOST:-10.0.1.41}
```

`--pc2-host` takes PC2's IP from the laptop's perspective (same value you
passed `--pc2-host` to the daemon script). It implies `--no-deploy`,
wires the recorder / manager split-topology SUBs, and stands up the
`x2_debug -> robot_pose` bridge (so the kplanner pose-feedback sees
measured yaw on the very first published frame). The older
`--remote-deploy HOST` is kept as a legacy alias.

### terminal 3 -- (optional) follow-tail PC2 logs

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh logs deploy
# (or: logs hand | logs monitor | logs all)
```

### post-run -- pull logs back + run postmortem

```sh
cd /home/stickbot/Projects/GR00T-WholeBodyControl
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh postmortem \
    --center-ts "2026-05-15T19:24:30" --window-s 30
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop
```

### Recovering from SAFE_IDLE on the headset

When the C++ deploy detects a stale `pose` stream (>0.5 s) it enters
`SAFE_IDLE` and pins the robot at `default_angles` with 4× damping.
To re-engage:

1. Make sure the laptop is back on WiFi and the operator-side stack
   is alive (`./run_x2_quest3_planner_stack.sh ...`).
2. On the Quest 3, hold **A and B together on the RIGHT controller**
   for at least 1 second, with X and Y on the left controller NOT
   pressed. The manager publishes a `pose_resume` ZMQ message.
3. The deploy re-enters `CONTROL` once it has seen both (a) a fresh
   pose stream for ≥1 s AND (b) a recent `pose_resume`. The headset
   plays an audio cue confirming the chord was registered.
