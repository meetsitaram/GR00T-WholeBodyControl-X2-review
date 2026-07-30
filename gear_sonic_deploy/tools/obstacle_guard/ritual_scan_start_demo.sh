# Added to ritual_start_demo.sh.
#
# Placement: the lumi call goes BEFORE `start_tmux pc2_kplanner`, the guard
# block goes AFTER it. lumi owns the PointCloud2 -> LaserScan converter;
# duplicating that converter here gave two publishers on /scan and the guard
# read inf.
#
# kplanner's ZMQ SUB connects to a socket that does not exist yet when the
# guard starts second. ZMQ reconnects when the publisher appears, so the clamp
# arms a few seconds later -- and the pad bridge starts after the guard, so the
# robot is not drivable during that window.

# --- before start_tmux pc2_kplanner ------------------------------------------

bash /home/run/getsolo/lumi.sh >> /home/run/getsolo/log/scan_guard.log 2>&1
sleep 8

# --- after the pc2_kplanner block --------------------------------------------

# SYSTEM python with absolute ROS paths: the gear_sonic venv has no rclpy, and
# a fresh post-reboot shell has no ROS env, so without these spelled out the
# guard dies silently and the robot drives unguarded.
#
# rclpy lives in .../local/lib/python3.10/dist-packages; rpyutils lives in
# .../lib/python3.10/site-packages. Both are required.
start_tmux scan_guard "LD_LIBRARY_PATH=/agibot/software/common/lib:/opt/ros/humble/lib \
  AMENT_PREFIX_PATH=/agibot/software/common:/opt/ros/humble \
  PYTHONPATH=/agibot/software/common/local/lib/python3.10/dist-packages:/opt/ros/humble/local/lib/python3.10/dist-packages:/opt/ros/humble/lib/python3.10/site-packages \
  stdbuf -oL -eL python3 /home/run/getsolo/scan_guard_pub.py \
  2>&1 | tee -a /home/run/getsolo/log/scan_guard.log"
sleep 4
