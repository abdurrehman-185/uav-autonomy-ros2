#!/usr/bin/env bash
# GPS-denied lidar-inertial flight.
#   bash run.sh               flight with the Gazebo window, camera recording and video
#   HEADLESS=1 bash run.sh    no Gazebo window
#   NO_VIDEO=1 bash run.sh    skip cameras and video
cd "$(dirname "$0")" || exit 1
WORLD=worlds/gps_denied.sdf
OUT=results
BRIDGE_TOPICS=(
    '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'
    '/range@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'
    '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
    '/model/x3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
    '/X3/gazebo/command/twist@geometry_msgs/msg/Twist]gz.msgs.Twist'
)
source ../common/launch_sim.sh

python3 gps_denied_nav.py "$OUT" &
NAV_PID=$!
sleep 20
check_frames
wait "$NAV_PID"
stop_all
trap - EXIT

python3 plot_results.py "$OUT"
if [ -z "${NO_VIDEO:-}" ]; then
    python3 ../common/compose_video.py video.json
fi
