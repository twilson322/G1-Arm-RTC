# ROS 2 Jazzy build of the node plus an end-to-end check of its topics and services.
#   docker build -f docker/ros2.Dockerfile -t g1-arm-rtc-ros .
#   docker run --rm g1-arm-rtc-ros
FROM ros:jazzy-ros-base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip libeigen3-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --break-system-packages mujoco==3.17.0
WORKDIR /ws/src/g1-arm-rtc
COPY . .
RUN [ -d models/assets ] || ./tools/fetch_assets.sh
WORKDIR /ws
RUN . /opt/ros/jazzy/setup.sh && colcon build --base-paths src/g1-arm-rtc/ros2 \
      --cmake-args -DCMAKE_BUILD_TYPE=Release
COPY docker/ros2_check.sh /ros2_check.sh
CMD ["bash", "/ros2_check.sh"]
