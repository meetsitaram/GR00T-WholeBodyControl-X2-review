./gear_sonic/scripts/run_x2_quest3_planner_stack.sh --cleanup-only
./gear_sonic/scripts/run_x2_quest3_planner_stack.sh

./gear_sonic/scripts/run_x2_quest3_planner_stack.sh \
    --duration 1200 --with-record \
    --output-dir data/lerobot/x2_pick_place_apple_20260603_v2 \
    --robocasa-env X2PickPlaceApple \
    --sonic-tokenizer-device cpu