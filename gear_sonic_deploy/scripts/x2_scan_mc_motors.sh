#!/usr/bin/env bash
#
# Thin wrapper for x2_scan_mc_motors.py that re-executes inside the
# docker_x2/x2sim container with the real-mode overlay (so ROS sees the
# robot at ROS_DOMAIN_ID=0, ROS_LOCALHOST_ONLY=0, and aimdk_msgs is on
# the Python path).
#
# Forwards every CLI arg through to the Python script. Default scan
# duration is 30 s; pass e.g. --duration 60 to extend.
#
# Run pattern (from host, MC must be in STAND_DEFAULT actively holding):
#
#   ./gear_sonic_deploy/scripts/x2_scan_mc_motors.sh --duration 30
#
# Safe to run any time: pure subscriber, never publishes on the bus.

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
COMPOSE_DIR="$SCRIPT_DIR/../docker_x2"

# If we're already inside the container, just exec the Python directly.
if [[ -n "${X2_DEPLOY_IN_DOCKER:-}" ]]; then
    exec python3 "$SCRIPT_DIR/x2_scan_mc_motors.py" "$@"
fi

# Otherwise, re-exec inside docker_x2/x2sim with real-mode env. Mirrors
# the pattern in deploy_x2.sh::maybe_relaunch_in_docker. We use
# docker-compose.yml + docker-compose.real.yml when the file exists so
# we don't accidentally inherit sim DDS isolation; if the overlay isn't
# there (older clones), the base compose file already has the right
# defaults for real-mode aimdk_msgs.
cd "$COMPOSE_DIR"
script_in_container="/workspace/sonic/gear_sonic_deploy/scripts/x2_scan_mc_motors.py"

# FASTRTPS_BUILTIN_TRANSPORTS=UDPv4: force FastDDS to use UDP for ALL
# traffic, including same-host. FastDDS' default is SHM+UDP (shared
# memory for same-host, UDP across hosts). Same-host SHM uses /dev/shm,
# which is PER-CONTAINER in docker -- even with network_mode: host. So
# the deploy container's /dev/shm and the scanner container's /dev/shm
# are isolated, and the deploy's published commands never reach the
# scanner over SHM. Discovery (multicast) still works across containers
# because that's UDP, which is why our publisher-discovery probe sees
# the deploy's command publishers but the actual command messages never
# arrive. Forcing UDPv4 here makes the scanner use the same transport
# that crosses container boundaries cleanly.
#
# Only apply this to the SCANNER -- the deploy container can keep using
# SHM internally for its 500 Hz writer (lower latency for the inner
# control loop). We just need the scanner to see what the deploy is
# publishing.
exec docker compose run --rm --service-ports \
    -e "X2_DEPLOY_IN_DOCKER=1" \
    -e "ROS_DOMAIN_ID=${X2_REAL_DOMAIN_ID:-0}" \
    -e "ROS_LOCALHOST_ONLY=0" \
    -e "FASTRTPS_BUILTIN_TRANSPORTS=UDPv4" \
    -v "$HOME:$HOME:rw" \
    -w "$HOME/Projects/GR00T-WholeBodyControl" \
    x2sim \
    bash -lc 'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec python3 "$@"' \
    bash "$script_in_container" "$@"
