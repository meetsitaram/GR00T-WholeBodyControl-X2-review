# X2 Ultra MuJoCo-sim container
#
# Sibling of gear_sonic_deploy/docker/Dockerfile.ros2 (which is the heavy
# CUDA + TensorRT + ONNX Runtime image used by the G1 deploy). This image is
# intentionally lighter -- ROS 2 Humble + aimdk_msgs + the Python deps the
# X2 MuJoCo<->ROS bridge needs.
#
# Modeled on the lean Dockerfile from agitbot-x2-record-and-replay (which is
# already battle-tested for talking to the real X2 Orin via DDS), with three
# changes for our use-case:
#   * vendors aimdk_msgs from agibot-x2-references/lx2501_3-v0.9.0.4 (so
#     this repo is self-contained -- no need to side-clone the SDK)
#   * drops the song-playback Python deps (lerobot, ruckig)
#   * adds the bridge's deps: mujoco, scipy, joblib, torch (CPU). torch is
#     CPU-only because eval_x2_mujoco.py imports it at module level for its
#     policy class -- the bridge itself never runs the policy, so we don't
#     need a GPU build.
#
# Build context: repo root (set by docker_x2/docker-compose.yml).
FROM ros:humble-ros-base

SHELL ["/bin/bash", "-c"]

# Apt deps:
#   python3-pip                          -- bridge's pip deps
#   ros-humble-sensor-msgs / geometry-msgs -- bridge pubs Imu, deploy eats them
#   ros-humble-rclcpp                    -- C++ deploy (build + link)
#   ros-humble-ament-cmake-gtest         -- C++ deploy unit tests
#   python3-colcon-common-extensions     -- `colcon build` the deploy
#   build-essential / cmake              -- C++ deploy compile chain
#   wget / ca-certificates               -- ONNX Runtime tarball download
#   curl / procps                        -- in-container debugging
#   libgl1 / libegl1 / libglu1-mesa      -- libGL.so.1 family (for OpenGL libs
#                                           when --sim-viewer is used; with
#                                           NVIDIA_DRIVER_CAPABILITIES=all the
#                                           NVIDIA driver libs override these
#                                           on the LD path -- mesa is the
#                                           graceful CPU fallback when there's
#                                           no NVIDIA runtime)
#   libglfw3                             -- GLFW window system used by
#                                           mujoco.viewer.launch_passive
#   libxcursor1 / libxinerama1 /         -- GLFW's X11 runtime deps; missing
#   libxrandr2 / libxi6 / libxext6 /        any of these and GLFW silently
#   libxrender1 / libxfixes3                fails to create a window
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        ros-humble-sensor-msgs \
        ros-humble-geometry-msgs \
        ros-humble-rclcpp \
        ros-humble-ament-cmake-gtest \
        python3-colcon-common-extensions \
        build-essential \
        cmake \
        wget \
        ca-certificates \
        curl \
        procps \
        libgl1 \
        libegl1 \
        libglu1-mesa \
        libglfw3 \
        libxcursor1 \
        libxinerama1 \
        libxrandr2 \
        libxi6 \
        libxext6 \
        libxrender1 \
        libxfixes3 \
    && rm -rf /var/lib/apt/lists/*

# ONNX Runtime 1.16.3 (x86_64 prebuilt tarball). Same version + install path
# `install_deps.sh` uses on the real X2 dev box, so the C++ deploy's CMake
# `find_path(... HINTS ${ONNXRUNTIME_ROOT}/include ...)` lands on the same
# layout in both worlds. Keeps `-DONNXRUNTIME_ROOT=/opt/onnxruntime` (the
# default `deploy_x2.sh` passes) hitting the right files.
ARG ONNXRUNTIME_VERSION=1.16.3
RUN cd /tmp && \
    wget -q --tries=3 \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz" \
        -O onnxruntime.tgz && \
    tar xzf onnxruntime.tgz && \
    mv "onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}" /opt/onnxruntime && \
    rm onnxruntime.tgz && \
    echo /opt/onnxruntime/lib > /etc/ld.so.conf.d/onnxruntime.conf && \
    ldconfig
ENV ONNXRUNTIME_ROOT=/opt/onnxruntime
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

# Build aimdk_msgs from the SDK source vendored in this repo. We use the
# upstream layout (interface/ + package.xml + CMakeLists.txt) verbatim.
WORKDIR /ros2_ws/src
COPY agibot-x2-references/lx2501_3-v0.9.0.4/src/aimdk_msgs/ aimdk_msgs/
WORKDIR /ros2_ws
RUN source /opt/ros/humble/setup.bash && \
    colcon build --packages-select aimdk_msgs \
        --cmake-args -DCMAKE_BUILD_TYPE=Release

# Python deps for the bridge.
#   mujoco           -- physics
#   numpy / scipy    -- math; scipy needed by eval_x2_mujoco for Rotation
#   joblib           -- motion-lib PKL loader (used when --sim-motion is set)
#   torch (CPU)      -- imported at module level by eval_x2_mujoco. The
#                       bridge does not run the policy; CPU build keeps the
#                       image ~1.5 GB lighter than the CUDA one.
RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    pip3 install --no-cache-dir \
        mujoco \
        numpy \
        scipy \
        joblib && \
    pip3 install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch

# iputils-ping for deploy_x2.sh's pre-flight reachability check. Kept in its
# own RUN layer (rather than folded into the big apt-get above) so adding it
# does NOT invalidate the ONNX Runtime / aimdk_msgs / torch layers cached
# below it -- a full rebuild of those takes ~20 minutes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Default workdir is the bind-mounted monorepo (set in docker-compose.yml).
WORKDIR /workspace/sonic

# Drop the user into a shell with both ROS 2 and the aimdk_msgs overlay
# sourced. Override with `docker compose run x2sim <cmd>` for one-shots.
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && exec bash"]
