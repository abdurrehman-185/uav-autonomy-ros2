#!/usr/bin/env bash
# Flies the same delivery route three times under the same GNSS spoofing attack:
#   naive (GNSS+IMU), gated (GNSS+IMU with chi-square gate), lidar (GNSS+IMU+lidar with integrity monitor)
#   HEADLESS=1 bash run.sh        no Gazebo window
#   NO_VIDEO=1 bash run.sh        skip cameras and video
#   MODES="lidar" bash run.sh     run a subset
cd "$(dirname "$0")" || exit 1
WORLD=worlds/street.sdf
python3 make_world.py

for MODE in ${MODES:-naive gated lidar}; do
    echo "=== $MODE ==="
    OUT=results/$MODE
    BRIDGE_TOPICS=(
        '/imu@sensor_msgs/msg/Imu[gz.msgs.IMU'
        '/navsat@sensor_msgs/msg/NavSatFix[gz.msgs.NavSat'
        '/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'
        '/range@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'
        '/model/x3/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        '/X3/gazebo/command/twist@geometry_msgs/msg/Twist]gz.msgs.Twist'
    )
    source ../common/launch_sim.sh
    python3 spoofer.py > "$OUT/spoofer.log" 2>&1 &
    SPOOF_PID=$!
    python3 spoof_nav.py "$MODE" "$OUT" &
    NAV_PID=$!
    sleep 25
    check_frames
    wait "$NAV_PID"
    kill "$SPOOF_PID" 2>/dev/null
    stop_all
    trap - EXIT
    sleep 3
done

python3 plot_results.py results
if [ -z "${NO_VIDEO:-}" ]; then
    for MODE in ${MODES:-naive gated lidar}; do
        python3 ../common/compose_video.py "results/video_$MODE.json"
    done
fi
