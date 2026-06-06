# Gesture Commands

=============== do not auto edit this section ===============
### start sonic on PC2 : Robogym Wifi
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml

### start local stack
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pc2-host 192.168.86.32

### play gesture to sit and stand
python -m gear_sonic.scripts.play_gesture sit_down_A540

python -m gear_sonic.scripts.play_gesture stand_up_A540 --delay 5

### stop sonic
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh stop --pc2-host 192.168.86.32


=============== end of pure manual notes section ===============


## List

```sh
python -m gear_sonic.scripts.play_gesture --list
```

## MuJoCo

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh

python -m gear_sonic.scripts.play_gesture sit_stand_sit_A540
```

## Real robot

```sh
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --pc2-host 192.168.86.32

python -m gear_sonic.scripts.play_gesture sit_stand_sit_A540
```

## Split sit + hold + stand

```sh
python -m gear_sonic.scripts.play_gesture sit_down_A540

python -m gear_sonic.scripts.play_gesture stand_up_A540 --delay 5
```

## Release a held pose

```sh
python -m gear_sonic.scripts.play_gesture --release
```
