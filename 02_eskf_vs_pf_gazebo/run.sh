#!/usr/bin/env bash
# 1) Fly a figure eight in Gazebo and record raw IMU, GNSS and ground truth.
# 2) Replay the recording through DR, EKF and three particle filters with a 60 s GNSS outage.
# 3) Plot, score and render the video.
#   HEADLESS=1 bash run.sh    no Gazebo window
#   NO_VIDEO=1 bash run.sh    skip cameras and video
#   REPLAY_ONLY=1 bash run.sh reuse the last recording
cd "$(dirname "$0")" || exit 1
WORLD=worlds/outdoor_flight.sdf
OUT=results

if [ -z "${REPLAY_ONLY:-}" ]; then
    python3 make_world.py
    BRIDGE_TOPICS=(
        '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
        '/navsat@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat'
        '/model/x3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        '/X3/gazebo/command/twist@geometry_msgs/msg/Twist]gz.msgs.Twist'
    )
    source ../common/launch_sim.sh
    python3 fly_and_record.py "$OUT" &
    NAV_PID=$!
    sleep 25
    check_frames
    wait "$NAV_PID"
    stop_all
    trap - EXIT
fi

python3 replay_filters.py "$OUT"
python3 plot_results.py "$OUT"
if [ -z "${NO_VIDEO:-}" ]; then
    python3 ../common/compose_video.py video.json
fi
