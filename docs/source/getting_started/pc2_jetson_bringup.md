# PC2 (Jetson Orin NX) bring-up for the X2 split-topology deploy

This page is the one-time setup checklist for the robot's on-board
Jetson Orin NX (PC2) so it can host the long-running C++ deploy + hand
bridge + motor monitor as tmux sessions, while the laptop runs only
the operator-side teleop / planner / recorder stack.

All PC2-side artifacts (ONNX Runtime, Python venv, colcon overlay,
ONNX checkpoints, runtime logs) live under a single prefix
(`/home/run/getsolo` by default) so the system Python, `/opt`, and the
AgiBot `/agibot/software/` tree are never written to. Wipe the prefix
and re-run `pc2_bringup.sh` to start over from scratch.

Once PC2 is set up, the day-to-day workflow is just:

```bash
./gear_sonic_deploy/scripts/pc2_preflight.sh
# `start --attach` lands you directly on the deploy tmux pane's Y/n
# safety gate. `deploy_x2.sh onbot` (PC2-native) owns the whole
# lifecycle inside that pane, so MC stop_app / start_app /
# RAMP_OUT / HOLD_FOR_MC all run on PC2 and survive a WiFi blip.
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 0 --remote-deploy 10.0.1.41
```

See `docs/source/references/x2_split_deploy_pc2.md` for the architecture
diagram + safety contract once you've finished the bring-up below.

## 1. Network layout

The X2 has two physical networks plus an optional carry-along WiFi:

| Network | Hosts | Use |
|---|---|---|
| `robogym` WiFi | Laptop ↔ PC2 | ZMQ pose stream, Quest 3 manager → deploy, monitor sidecar (carry-along during the demo) |
| SDK ethernet | Laptop ↔ PC2 ↔ PC1 (`10.0.1.40`) | Wired option for low-latency dev; also how PC2 reaches PC1's ROS 2 / DDS, EM HTTP API, and MC services |

PC2's stable address depends on which network you're on:

| Network | PC2 address |
|---|---|
| SDK ethernet (wired) | `10.0.1.41` |
| `robogym` WiFi | `192.168.86.21` |

`pc2_bringup.sh` will work over either network. The wired link is
~1000× lower latency (0.25 ms vs 250 ms), so use it whenever possible
for the bringup (large ONNX Runtime download, colcon build).

## 2. SSH key

Drop your SSH key once so subsequent scripts (`pc2_bringup.sh`,
`x2_pc2_daemons.sh`, `x2_discover_network.sh`, `pc2_preflight.sh`) can
log in non-interactively:

```bash
ssh-copy-id run@10.0.1.41         # default password 'run / 1' on a fresh flash
```

## 3. One-shot install: `pc2_bringup.sh`

Everything below (ONNX Runtime download + extract, Python venv with
pyzmq, colcon workspace rsync + build, prefix directory layout) is
automated by `pc2_bringup.sh`. It's idempotent and safe to re-run.

```bash
cd /home/stickbot/Projects/GR00T-WholeBodyControl

# Preview without changing anything on PC2:
./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host 10.0.1.41 --dry-run

# Run the full install (~3 min total on Orin NX, dominated by the
# colcon build at ~80 s):
./gear_sonic_deploy/scripts/pc2_bringup.sh --pc2-host 10.0.1.41

# Same, but also rsync the in-repo deploy policy. The default
# checkpoint is LFS-tracked at data/policies/agibot_x2_sonic.onnx;
# run `git lfs pull --include=data/policies/agibot_x2_sonic.onnx`
# first if you only have the LFS pointer file locally.
./gear_sonic_deploy/scripts/pc2_bringup.sh \
    --pc2-host 10.0.1.41 \
    --model ./data/policies/agibot_x2_sonic.onnx

# Or, for a checkpoint that isn't in the repo (e.g. a fresh cloud
# train you haven't promoted yet), point --model at any local path:
./gear_sonic_deploy/scripts/pc2_bringup.sh \
    --pc2-host 10.0.1.41 \
    --model $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/exported/model_step_025000_g1.onnx
```

What it does (each step is independently re-runnable):

| Step | Result on PC2 |
|---|---|
| 0. ssh sanity | Aborts early with a `ssh-copy-id` hint if the key isn't set up. |
| 1. prefix dirs | `mkdir -p /home/run/getsolo/{onnxruntime,venv,ws,policies,log}` |
| 2. aimdk_msgs check | Verifies `/agibot/software/housekeeper/bin/aimdk_msgs/` exists (this is the canonical install with both the CMake config and the Python bindings — we link/import from it but never write into it). |
| 3. ONNX Runtime | Downloads `onnxruntime-linux-aarch64-1.16.3.tgz` locally, scp's it to PC2, extracts to `/home/run/getsolo/onnxruntime/`. |
| 4. Python venv | `python3 -m venv --system-site-packages /home/run/getsolo/venv` so the venv inherits `rclpy` + numpy from system, then `pip install pyzmq` into the venv. |
| 5. workspace sources | Rsyncs `gear_sonic_deploy/src/x2/agi_x2_deploy_onnx_ref/` + `gear_sonic_deploy/src/common/` + `gear_sonic/utils/teleop/zmq/zmq_packed_message_decoder.py` into `/home/run/getsolo/ws/src/`. |
| 6. colcon build | Builds `agi_x2_deploy_onnx_ref` with `-DONNXRUNTIME_ROOT=/home/run/getsolo/onnxruntime` and `aimdk_msgs` on `AMENT_PREFIX_PATH`. Produces `/home/run/getsolo/ws/install/agi_x2_deploy_onnx_ref/lib/agi_x2_deploy_onnx_ref/x2_deploy_onnx_ref`. |
| 7. deploy_x2.sh + helpers | Rsyncs `gear_sonic_deploy/deploy_x2.sh`, the python helpers it shells out to (`x2_preflight.py`, `tuning_config_to_args.py`, `x2_mc_escalator.py`, `export_motion_for_deploy.py`, `x2_hand_zmq_to_aimdk_bridge.py`), and the `configs/real_deploy_tuning/*.yaml` presets to `/home/run/getsolo/gear_sonic_deploy/`. This is what `x2_pc2_daemons.sh start` invokes in the deploy tmux pane (so the operator gets the full Y/n gate + MC stop/start + sentinel handoff + RAMP_OUT trap on PC2 itself, instead of a bare `ros2 run`). |
| 8. ONNX model | If `--model PATH` was passed, rsyncs it to `/home/run/getsolo/policies/`. |
| 9. summary | Prints copy/paste env vars + the daemon start command. |

Useful flags:

| Flag | What it does |
|---|---|
| `--dry-run` | Print intended actions, change nothing on PC2. |
| `--prefix DIR` | Stage under a different root (default `/home/run/getsolo`). |
| `--skip-onnx` / `--skip-venv` / `--skip-source` / `--skip-build` / `--skip-model` | Run only the remaining steps. Handy for source-refresh-only iterations. |
| `--force-onnx` / `--force-venv` / `--force-build` | Wipe and redo that step (otherwise it short-circuits if already done). |
| `--onnx-url URL` | Use a different ONNX Runtime build (e.g. CUDA / TensorRT). |

## 4. Verify

After the bringup completes:

```bash
# 1. Verify the laptop can resolve PC2 + see the new prefix:
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh print-env --pc2-host 10.0.1.41

# 2. Verify the standard preflight passes (MC publishers / EM HTTP / etc.):
./gear_sonic_deploy/scripts/pc2_preflight.sh \
    --pc2-host 10.0.1.41 --pc2-user run \
    --pc1-host 10.0.1.40
```

Expected: `print-env` lists the new `/home/run/getsolo/{ws,venv,onnxruntime,log}`
paths and reachability probes are green; `pc2_preflight.sh` reports 8
sections all green with `pass=N warn=0 fail=0`.

If the MC publishers section reports `publishers=0`, MC is not running
or is in an idle / passive state — start the MC app from PC1
(`curl -X POST 'http://10.0.1.40:50080/x2/em/start_app?app=mc' -d '{}' \
-H 'Content-Type: application/json'`) and re-run the preflight.

## 5. Running the daemons

```bash
# On the laptop:
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/expressive.yaml
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh status

# Then start the operator-side stack:
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 0 --remote-deploy 10.0.1.41

# After the test, postmortem (rsync logs back + run analysis):
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh postmortem
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop
```

All three daemons survive any laptop SSH disconnect. If the WiFi
bridge between the laptop and PC2 blinks for >0.5 s, the deploy's
pose-ref starvation watchdog trips into `SAFE_IDLE`, the robot holds
default angles with damping, and the operator can hit the **A+B
chord** (right controller, no X+Y) on the Quest 3 to re-engage once
the connection is back.

See `docs/source/references/x2_split_deploy_pc2.md` for the full
contract.

## 6. Updating the C++ deploy after a code change

Re-run only the source + build steps; the ONNX Runtime, venv, and
model checkpoint stay untouched:

```bash
./gear_sonic_deploy/scripts/pc2_bringup.sh \
    --pc2-host 10.0.1.41 \
    --skip-onnx --skip-venv --skip-model
```

Or, for a "wipe and rebuild from scratch" loop after a structural
CMakeLists change:

```bash
./gear_sonic_deploy/scripts/pc2_bringup.sh \
    --pc2-host 10.0.1.41 \
    --skip-onnx --skip-venv --skip-model \
    --force-build
```

## 7. Starting over from scratch

On the Orin NX:

```bash
rm -rf /home/run/getsolo
```

Then re-run `pc2_bringup.sh` from the laptop. Nothing else on the
filesystem (the system Python, `/opt/ros`, `/agibot/software/`) is
touched by this teardown, so a re-flashed or borrowed Orin NX can be
brought up with the same one command.
