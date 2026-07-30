# RViz on the X2

Live visualisation of the robot: URDF driven by real joint angles, LiDAR
pointcloud, LaserScan, cameras. Rendered on the robot and viewed over VNC,
because the X2 has no display and X11 forwarding does not work (see below).

`lumi.sh` brings the whole thing up in one command.

```bash
/home/run/getsolo/lumi.sh          # visualiser only
/home/run/getsolo/lumi.sh stop     # tear down, including the VNC server
/home/run/getsolo/lumi.sh rc       # visualiser + locomotion (ritual)
```

Then connect a VNC client to `192.168.86.32:1`.

---

## What lumi.sh starts

| Piece | Why |
|---|---|
| TigerVNC on `:1` | X11 forwarding cannot give RViz a GL context — see below |
| URDF rewrite → `/tmp/x2_rviz.urdf` | mesh paths must be absolute `file://` URIs |
| `robot_state_publisher` | URDF → `/tf` |
| `joint_bridge.py` | HAL joint states → `sensor_msgs/JointState` |
| `pointcloud_to_laserscan` | 3D cloud → `/scan`, also feeds the obstacle guard |
| `rviz2 -d lumi.rviz` | the saved display config |

---

## Why VNC and not X11 forwarding

`ssh -Y` plus XQuartz gets you a working X connection, and 2D apps are fine —
but RViz dies:

```
Failed to create an OpenGL context. BadValue
RenderingAPIException: Unable to create a suitable GLXContext
... Unable to create the rendering window after 100 tries
```

Indirect GLX from macOS to an aarch64 host does not provide a usable context.
`LIBGL_ALWAYS_INDIRECT` and `LIBGL_ALWAYS_SOFTWARE` do not help. A local X
server on the robot does, so TigerVNC renders with Mesa software GL and ships
pixels instead.

Consequence: RViz is rendered by the robot's CPU — the same CPU running the
gait controller. It measured ~150% CPU on its own. Close it when not in use,
and expect it to be usable rather than smooth.

For anything better, move the rendering off the robot: RViz built natively on
the client, or Foxglove (the robot already runs `cobridge`, so the robot side
of that is done).

---

## The joint bridge

`robot_state_publisher` needs `sensor_msgs/JointState`, but the X2 publishes
`aimdk_msgs/JointStateArray` on `/aima/hal/joint/{leg,arm,waist,head}/state`.
`joint_bridge.py` translates.

HAL joint names match the URDF exactly (`left_hip_pitch_joint`,
`left_knee_joint`, …), so names pass straight through — no lookup table, and
therefore no chance of a silently wrong pose from a mis-ordered index. 31
joints total.

Without it, RViz shows the model in its zero pose regardless of what the robot
is doing. There is no `joint_state_publisher` package installed on this robot,
so a zero-pose stand-in has to be written by hand if you want one.

**Only ever run one publisher on `/joint_states`.** Two of them (a leftover
zero-pose publisher plus the bridge) makes the model flicker between upright
and the real pose, and `ros2 topic hz` shows a negative `min` — out-of-order
timestamps from the two sources interleaving.

---

## URDF and mesh paths

The URDF at
`planner_stack/gear_sonic/data/assets/robot_description/urdf/x2_ultra/x2_ultra.urdf`
uses relative mesh paths (`meshes/pelvis.STL`). RViz treats those as URLs:

```
Error retrieving file [meshes/head_pitch_link.STL]: Could not resolve host: meshes
```

`lumi.sh` rewrites all 69 of them to absolute `file://` URIs into
`/tmp/x2_rviz.urdf` at startup. The TF tree and camera/LiDAR frames resolve;
only the geometry is affected by this.

---

## Two TF trees

`robot_state_publisher` (from the URDF) and the vendor's `joint_tf` app both
publish. `tf2_echo` reports:

```
Could not find a connection between 'base_link' and 'lidar_chest_front'
because they are not part of the same tree. Tf has two or more unconnected trees.
```

The tree does actually connect
(`base_link → pelvis → waist_yaw → waist_pitch → torso_link → lidar_chest_front`),
but the static links carry **timestamp 0** while the joint transforms carry real
stamps at ~12 Hz. A lookup at a sensor message's stamp can therefore fail.

`transform_tolerance:=1.0` on the scan converter covers it. Expect intermittent
"unconnected trees" warnings that resolve on their own.

---

## Displays worth adding

Set **Fixed Frame** to `base_link`. Every display below needs
**Reliability → Best Effort** (expand the `Topic` entry) — see the QoS note.

| Display | Topic |
|---|---|
| RobotModel | Description Topic `/robot_description` |
| PointCloud2 | `/aima/hal/sensor/lidar_chest_front/lidar_pointcloud_down_sampling` |
| LaserScan | `/scan` |
| PointCloud2 | `/aima/hal/sensor/rgbd_head_front/depth_pointcloud` |
| Image | `/aima/hal/sensor/rgb_head_front_center/rgb_image`, transport `compressed` |

Cameras, all live: front centre RGB (~30 Hz), rear RGB (~10 Hz), RGB-D colour,
stereo left/right. Prefer the `compressed` transport — raw frames over VNC are
unusable.

Save with **File → Save Config As** → `/home/run/getsolo/lumi.rviz`, which is
what `lumi.sh` loads. Without saving, every restart comes up with RViz defaults
and Fixed Frame `map`, which does not exist:

```
Frame [map] does not exist
```

---

## Performance

In rough order of impact:

1. **PointCloud2 Style → Points.** "Flat Squares" is much heavier, and a
   stale `Size (m)` of 3 draws three-metre quads per point.
2. **Global Options → Frame Rate → 10.** The default 30 will never be reached
   and the attempts cost CPU.
3. **Uncheck TF.** 30+ frames redrawing with arrows and labels.
4. **Uncheck what you are not looking at** — RobotModel is 69 STL meshes.
5. **Lower VNC colour depth** to `-depth 16`. Halves bandwidth without
   shrinking the window.

Ethernet helps the link but not the bottleneck: the robot's CPU is doing the
rasterising either way.

## Final Outcome
---

<img width="960" height="1234" alt="image" src="https://github.com/user-attachments/assets/2db4c112-e1b9-402d-babd-f9c8f3d191ce" />



## Gotchas

**QoS.** Every sensor topic is `BEST_EFFORT`; RViz defaults to `RELIABLE` and
shows `showing 0 points from 0 messages` with a green "Ok" status. Expand the
display's `Topic` entry and set Reliability to Best Effort. This bites on every
display, every time.

**`--once` lies.** `pointcloud_to_laserscan` only subscribes to the cloud once
something subscribes to `/scan`. `ros2 topic echo /scan --once` connects, takes
the first (empty) message, and disconnects — reporting all-`inf` and looking
broken when the pipeline is fine. Use a persistent subscriber.

**Scan slice geometry.** `min_height`/`max_height` apply in `target_frame`.
`lidar_chest_front` is mounted rotated (`rpy="-1.5708 0 0"`), so its local Z is
not vertical and a slice there produces a *vertical* fan. Target `base_link`
instead, where Z is genuinely up. Forward also does not land at 0° in
`base_link` — see `OBSTACLE_GUARD.md`.

**Shell environment.** `lumi.sh` exports `PYTHONPATH` for the aimdk messages,
which clobbers `ros2cli` in the shell that ran it:

```
PackageNotFoundError: No package metadata was found for ros2cli
```

Run `ros2` commands from a separate SSH session.

**`aimdk_msgs` is not on the default path.** It lives in
`/agibot/software/common/`, and without it every custom type fails with
`The message type '...' is invalid`:

```bash
export LD_LIBRARY_PATH=/agibot/software/common/lib:$LD_LIBRARY_PATH
export AMENT_PREFIX_PATH=/agibot/software/common:$AMENT_PREFIX_PATH
export PYTHONPATH=/agibot/software/common/local/lib/python3.10/dist-packages:$PYTHONPATH
```

A fresh login shell has none of this. `aima em load-env --source-msgs` is the
vendor's equivalent.

**Do not pip install into the getsolo venv.** `mediapipe` pulls `numpy>=2` and
`opencv-contrib-python`, which uninstalls numpy 1.26.4 and breaks
`onnxruntime` — the planner then cannot start at all. Use a separate venv for
anything experimental.

---

## The model is pinned at the origin

Joint angles are live, but the robot does not translate across the grid. The
URDF root is `pelvis`, and nothing publishes a floating-base transform.

That would need `odom → pelvis` from a localisation source, and
`/slam/localization/odometry` has zero publishers on this unit — the SLAM
module is not installed (see `OBSTACLE_GUARD.md`). Correct joint angles in
place is what is available.
