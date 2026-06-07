# Pick and Place Commands

=============== do not auto edit this section ===============
### start sonic on PC2 : Robogym Wifi
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml

### start local stack with recording enabled
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --with-record \
    --head-cameras \
    --output-dir data/lerobot/x2_grab_a_drink \
    --task "grab a drink"

### camera access on PC2
gear_sonic_deploy/scripts/x2_pc2_cameras.sh status --host 192.168.86.32
gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal --host 192.168.86.32

### replay episode
./gear_sonic/scripts/view_x2_recorded_dataset.sh --dataset x2_grab_a_drink --episode 6

### stop sonic
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32


=============== end of pure manual notes section ===============

---

## Recording variants (drop-in replacements for the "start local stack" step)

Your usual `run_x2_quest3_planner_stack.sh --pc2-host …` is teleop-only
(no parquet writes). To record a LeRobot dataset for VLA training,
replace that one line with one of the variants below. The other two
lines in your runbook (the SONIC start on PC2 + the SONIC stop) stay
exactly the same.

### Record without head cameras (legacy single-camera schema)

Writes `observation.images.ego_view` (MuJoCo render) only.

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --with-record \
    --output-dir data/lerobot/x2_pick_place_v1 \
    --task "pick up the apple and place it in the bowl"
```

### Record WITH the three real PC2 head cameras (recommended)

Writes four video tracks per episode:
`observation.images.{ego_view,head_front,stereo_left,stereo_right}`.
`--head-cameras` auto-launches the PC2 ROS→ZMQ bridge over SSH
against `--pc2-host` before spawning the recorder; `--camera-host`
defaults to `--pc2-host` so no extra plumbing.

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --with-record \
    --head-cameras \
    --output-dir data/lerobot/x2_pick_place_cams_v1 \
    --task "pick up the apple and place it in the bowl"
```

### Robocasa scene mode (auto-fills the task from the scene)

Drops the `--task` flag — `RobocasaTaskMirror` pulls the canonical
instruction from the scene metadata. Add `--head-cameras` here too if
you want the four-track schema.

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 \
    --with-record \
    --head-cameras \
    --robocasa-env X2PickPlaceApple \
    --output-dir data/lerobot/x2_pick_place_apple_v1
```

---

## Pre-flight if the PC2 head cameras came up missing

The IMX900 stereo pair sometimes loses to `orbbec_camera` in the boot
Argus race. Check + restart-hal + verify, in that order:

```sh
gear_sonic_deploy/scripts/x2_pc2_cameras.sh status      --host 192.168.86.32
gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal --host 192.168.86.32   # only if stereo pubs=0
gear_sonic_deploy/scripts/x2_pc2_cameras.sh grab        --host 192.168.86.32   # one JPEG per cam back to laptop
```

You only need to do this on first boot of the day (or after a manual
`aima em` bounce). Once `status` shows `pubs=1` on all four head
topics, the bridge will Just Work from then on.

## Camera bridge cleanup

`--head-cameras` deliberately leaves the bridge running on PC2 after
the recorder exits so back-to-back record sessions don't pay the
cold-start. Tear it down at end of day with:

```sh
gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-stop --host 192.168.86.32
```