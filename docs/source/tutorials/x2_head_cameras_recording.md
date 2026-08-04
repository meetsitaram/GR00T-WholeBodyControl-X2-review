# X2 head-camera recording (Orbbec + IMX900 stereo)

End-to-end runbook for collecting an X2 dataset that includes the three
real PC2-attached head cameras alongside the existing MuJoCo-rendered
`ego_view`. After this you'll have a LeRobot v2.1 dataset with four
synchronized video tracks per episode:

* `observation.images.ego_view` — MuJoCo render of the head-mounted
  camera (this is what the X2 record pipeline has always written).
* `observation.images.head_front` — physical **Orbbec Gemini 335** RGB
  on the head center.
* `observation.images.stereo_left` / `observation.images.stereo_right`
  — physical **Sony IMX900** GMSL cameras on the head stereo rig.

The three real-camera streams are downscaled to 640×480 RGB at the PC2
bridge so they line up bit-for-bit with the MuJoCo render's shape; no
per-tick resize on the laptop.

---

## 1. One-time PC2 setup

The bridge depends on `pyzmq`, `msgpack`, and `msgpack-numpy` in PC2's
system Python (so it shares an interpreter with `rclpy`). The
`x2_pc2_cameras.sh serve` action installs them via `pip3 install --user`
on first run, but you can pre-stage:

```sh
ssh agibot-pc2 'pip3 install --user pyzmq msgpack msgpack-numpy'
```

PC2 host/user defaults are `run@${X2_PC2_HOST}`; override with
`X2_PC2_HOST` / `X2_PC2_USER` environment variables or the
`--host` / `--user` flags on the helper scripts.

---

## 2. Bring the cameras up (every boot)

PC2 has a known boot-time Argus race between `orbbec_camera` and
`hal_sensor_orin` that sometimes leaves the three IMX900 GMSL cameras
unregistered (you'll see `pubs=0` on the stereo + rear topics).
Bouncing `hal_sensor_orin` after Orbbec stabilises wins the race
deterministically.

```sh
# What's currently publishing?
gear_sonic_deploy/scripts/x2_pc2_cameras.sh status

# If stereo / rear pubs are 0, bounce HAL:
gear_sonic_deploy/scripts/x2_pc2_cameras.sh restart-hal

# Sanity-check by grabbing one JPEG per camera back to the laptop:
gear_sonic_deploy/scripts/x2_pc2_cameras.sh grab
ls -lh /tmp/x2_cam_samples/
```

Expected after a clean `status`:

```
/aima/hal/sensor/rgb_head_front_center/rgb_image     pubs=1
/aima/hal/sensor/stereo_head_front_left/rgb_image    pubs=1
/aima/hal/sensor/stereo_head_front_right/rgb_image   pubs=1
/aima/hal/sensor/rgb_head_rear/rgb_image             pubs=1
```

---

## 2b. Sensor discovery reference (audited 2026-08-04 on PC2)

Ground truth for "which device is which" lives in the HAL config on PC2:
`/agibot/software/hal_sensor_orin/bin/cfg/hal_sensor_module_config_orin.yaml`
(selected by `AGIBOT_SOC_INDEX=1` in `em_run.sh`; the app runs as
`aima-hal_sensor-app-main` under `aima em`).

| Sensor | Device on PC2 | Notes |
|---|---|---|
| `stereo_head_front_left` | `/dev/cam1` → `/dev/video9` | IMX900 GMSL, 2064×1552 BG12 @30 fps; carries the stereo IMU (`enable_imu: true`, `stereo_imu_head_front`). The policy camera. |
| `stereo_head_front_right` | `/dev/cam2` → `/dev/video10` | IMX900, same mode, no IMU. |
| `rgb_head_rear` | `/dev/cam0` → `/dev/video0` | Third IMX900 — the rear-view camera. |
| Orbbec Gemini 335 (RGBD) | USB `/dev/video1–8` (depth=video1 Z16, IR=video3, color=video7 YUYV/MJPG) | The HAL's `rgbd_camera_module` is **commented out**; the RGBD runs via the standalone ROS 2 driver `ros2 launch orbbec_camera gemini_330_series.launch.py` instead. `rgb_head_front_center` topics come from this path. |
| `chest_lidar` (vertical) | Ethernet `10.11.1.100` on PC2's `sensor0` net (`10.11.1.1/24`) | Topic namespace `/aima/hal/sensor/lidar_chest_front` (+ `lidar_imu_chest_front`). |

Access notes:

* PC2 is reachable off-wire via its WiFi leg — `run@${X2_PC2_HOST}`
  (`X2_PC2_HOST=${X2_PC2_HOST}` for these scripts). Wired remains
  `run@${X2_PC2_HOST}`. PC2 nets: `develop0` ${X2_PC2_HOST}, `sensor0` 10.11.1.1,
  `ssh0` 10.0.200.41.
* Raw `v4l2-ctl` captures of the IMX900s return nothing and
  `nvarguscamerasrc` fails with "Failed to create CaptureSession" while
  the HAL owns the sensors — always go through the HAL topics.
* Image topics ride aimrt's ROS 2 backend; the config's
  `pub_topics_options` only override QoS for `camera_info`/`tf_static`/
  `module_info` patterns.
* Bouncing the HAL also brings up the chest lidar topics
  (`/aima/hal/sensor/lidar_chest_front/{lidar_pointcloud,imu,lidar_status,...}`).
* **RGBD mount inversion (verified 2026-08-04)**: the Orbbec is mounted
  upside-down at the chin, pointing forward-down ~40°. The URDF encodes the
  physical (inverted) mount — `rgbd_head_front` rpy `(2.2689, 0, 1.5708)`
  has image-up pointing world-down, while stereo/center/rear all follow the
  upright optical convention. The ROS driver un-flips the image in software,
  so live streams are upright. Consequence: a SIM camera placed at the raw
  URDF `rgbd_head_front` frame renders upside-down vs the real stream —
  either roll the sim camera frame 180° about its optical axis, or add a
  corrected child frame; do NOT "fix" the URDF itself (it matches physical
  reality and the real TF chain).

---

## 3. Start the ROS → ZMQ camera bridge on PC2

The bridge subscribes to the three front-facing head topics on PC2,
resizes each frame to 640×480, JPEG-encodes at quality 85, and
republishes them as a merged `ImageMessageSchema` ZMQ PUB on
`tcp://*:5555`. The dataset recorder on the laptop consumes this with
`ComposedCameraClientSensor(server_ip='${X2_PC2_HOST}', port=5555)`.

```sh
# Ship + (re-)launch in background. Idempotent.
gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve

# Tail the bridge log (Ctrl-C to detach; bridge keeps running):
gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-log

# Quick laptop-side connectivity probe:
.venv/bin/python -c "
from gear_sonic.camera.composed_camera import ComposedCameraClientSensor
import time
c = ComposedCameraClientSensor(server_ip='${X2_PC2_HOST}', port=5555)
t0 = time.time(); n = 0
while time.time() - t0 < 3:
    m = c.read(blocking=False)
    if m and m.get('images'):
        n += 1
print(f'received {n} bundles in 3s')
c.close()
"
```

To stop the bridge later (recommend doing this between robots, not
between back-to-back recording sessions):

```sh
gear_sonic_deploy/scripts/x2_pc2_cameras.sh serve-stop
```

---

## 4. Record with the VR planner + SONIC pipeline + cameras

Two flags are added to `record_x2_dataset.sh`:

* `--head-cameras` enables ingestion of the three PC2 camera streams.
  When set, the wrapper auto-runs `x2_pc2_cameras.sh serve` first.
* `--no-camera-autostart` skips that auto-launch (use when you've
  already started the bridge by hand and want to avoid a re-ship).

Recommended VLA recipe (sim deploy + Quest 3 + 4 video tracks per
episode):

```sh
cd <repo> && \
bash gear_sonic/scripts/record_x2_dataset.sh \
    --output-dir data/lerobot/x2_quest3_sonic_cams_v1 \
    --task "wave hello with both hands" \
    --sim-omnihand \
    --wrist-bypass ik \
    --head-cameras \
    --sonic-checkpoint $HOME/x2_cloud_checkpoints/h200-iter-25000-sphere-feet-20260501/model_step_025000.pt
```

Operator buttons in the Quest 3 controllers (unchanged):

* **A** — toggle active arm tracking.
* **B** — start a fresh episode.
* **X** — stop and *save* the current episode.
* **Y** — stop and *discard*.

What the recorder prints at startup when `--head-cameras` is active:

```
[recorder] head cameras ENABLED -> connecting to tcp://${X2_PC2_HOST}:5555
Initialized composed camera client sensor
[recorder] head-camera bridge ready: keys=['head_front', 'stereo_left', 'stereo_right'] shapes={...}
```

If the bridge isn't up or only publishes some of the three keys, the
recorder fails fast with a clear error message and never starts the
exporter — so we don't accidentally write a partial-schema parquet.

While recording, head-camera ticks older than 500 ms (default
`--camera-max-staleness`) are dropped from the dataset with a
rate-limited warning. The body-state stream (deploy `x2_debug`) keeps
running independently; only the affected ticks are skipped.

---

## 5. Inspect what landed

```sh
DS=data/lerobot/x2_quest3_sonic_cams_v1
EP=000000

# meta/info.json now lists 4 video features
jq '.features | with_entries(select(.value.dtype == "video"))' ${DS}/meta/info.json

# Play all 4 streams side-by-side
for cam in ego_view head_front stereo_left stereo_right; do
    xdg-open ${DS}/videos/chunk-000/observation.images.${cam}/episode_${EP}.mp4 &
done
```

Per-track frame count + duration should match (the exporter writes one
frame per recorded tick, identical across keys, by construction of the
strict schema validator).

---

## 6. Troubleshooting

| Symptom                                                                  | Likely cause / fix                                                                                                                                |
|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
| `status` shows pubs=0 on stereo / rear topics                            | Argus boot race. Run `x2_pc2_cameras.sh restart-hal`.                                                                                              |
| Recorder errors `head-camera bridge ... did NOT publish a complete bundle` | Bridge not running on PC2, or only the Orbbec topic is publishing. Run `x2_pc2_cameras.sh status` then `serve` (and `restart-hal` if necessary).   |
| Bridge log shows `dropped_zmq>0` accumulating                            | Laptop subscriber lagging (rare). Reduce `--publish-rate` on the bridge or use a wired connection.                                                |
| Bridge `in_rates` shows IMX900 at <2 Hz                                  | `hal_sensor_orin` may be throttled by another consumer. Try `restart-hal` again; if persistent, check `aima em doctor`.                            |
| Stale-frame warnings in the recorder                                     | Bridge is alive but slow. Increase `--camera-max-staleness` (default 0.5 s) or `restart-hal` to reset HAL.                                         |
| Recorder schema validator rejects a frame                                 | Schema flag mismatch between training-config consumers. Make sure the same `include_head_cameras` value flows into `get_features_x2_vla` everywhere. |

---

## 7. Where the wire spec is defined

* **PC2 bridge** —
  [`gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py`](../../../gear_sonic_deploy/scripts/x2_pc2_camera_zmq_publisher.py).
  ROS subscribers per stream (one node each, single-threaded executor
  per node), JPEG re-encode + resize, msgpack publish at 50 Hz.
* **Laptop client** —
  [`gear_sonic/camera/composed_camera.py::ComposedCameraClientSensor`](../../../gear_sonic/camera/composed_camera.py).
  Unmodified — the same client the G1 `run_data_exporter.py` uses.
* **Schema** —
  [`gear_sonic/data/features_x2_vla.py::HEAD_CAM_KEYS`](../../../gear_sonic/data/features_x2_vla.py).
  Single source of truth for mount-key names; the bridge and the
  recorder both import from here.
* **Recorder integration** —
  [`gear_sonic/utils/teleop/x2_dataset_recorder.py::_init_head_cameras`](../../../gear_sonic/utils/teleop/x2_dataset_recorder.py)
  / `_head_camera_subscriber` / `_format_head_camera_frame_data`.
