
## SIM Only Mode

### launch pkl direct stack
gear_sonic/scripts/run_x2_pkl_direct_stack.sh --model ~/getsolo/policies/agibot_x2_sonic.onnx

## Launch Gamepad Controler
.venv/bin/python -m gear_sonic.scripts.play_xbox_controller

================================================================

## Real Robot Mode

### Start sonic on robot pc2
./gear_sonic_deploy/scripts/x2_pc2_daemons.sh start --attach \
    --pc2-host 192.168.86.32 --laptop-host 192.168.86.22 \
    --model /home/run/getsolo/policies/agibot_x2_sonic.onnx \
    --tuning gear_sonic_deploy/configs/real_deploy_tuning/walking_recovery.yaml \
    --lock-head-straight

### run the pkl stack
./gear_sonic/scripts/run_x2_pkl_direct_stack.sh \
    --pc2-host 192.168.86.32

### run the vr planner stack
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --pc2-host 192.168.86.32 

### launch xbox controller run
.venv/bin/python -m gear_sonic.scripts.play_xbox_controller

### To Stop sonic on robot (*** this will collapse the robot and needs to be held)
          
===================================================================








