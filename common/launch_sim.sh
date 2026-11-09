#!/usr/bin/env bash
# Shared launcher: starts the Gazebo server, an optional Gazebo window, the ROS-Gazebo bridge
# and the camera recorder. Source it from a project's run.sh after setting WORLD, OUT and
# BRIDGE_TOPICS.
#   HEADLESS=1  no Gazebo window (the window runs as a separate client, so closing it
#               never stops the simulation)
#   NO_VIDEO=1  skip camera recording

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set +u
source /opt/ros/lyrical/setup.bash

rm -rf "$OUT"
mkdir -p "$OUT"

stop_all() {
    [ -n "${REC_PID:-}" ] && kill -INT "$REC_PID" 2>/dev/null && wait "$REC_PID" 2>/dev/null
    kill "${NAV_PID:-}" "${BRIDGE_PID:-}" "${GUI_PID:-}" "${GZ_PID:-}" 2>/dev/null
    sleep 1
    pkill -f "gz-sim-main -s -r $WORLD" 2>/dev/null
    pkill -f "gz-sim-main -g" 2>/dev/null
    pkill -f gz-sim-gui-client 2>/dev/null
    pkill -f parameter_bridge 2>/dev/null
    REC_PID= NAV_PID= BRIDGE_PID= GUI_PID= GZ_PID=
}
trap stop_all EXIT

gz sim -s -r "$WORLD" > "$OUT/gazebo.log" 2>&1 &
GZ_PID=$!
sleep 3
if [ -z "${HEADLESS:-}" ]; then
    gz sim -g > "$OUT/gazebo_gui.log" 2>&1 &
    GUI_PID=$!
fi

if [ -z "${NO_VIDEO:-}" ]; then
    BRIDGE_TOPICS+=('/chase_cam@sensor_msgs/msg/Image[gz.msgs.Image' '/overhead_cam@sensor_msgs/msg/Image[gz.msgs.Image')
fi
ros2 run ros_gz_bridge parameter_bridge "${BRIDGE_TOPICS[@]}" > "$OUT/bridge.log" 2>&1 &
BRIDGE_PID=$!

start_recorder() {
    [ -n "${NO_VIDEO:-}" ] && return 0
    python3 "$COMMON_DIR/frame_recorder.py" "$OUT/frames" /chase_cam /overhead_cam >> "$OUT/recorder.log" 2>&1 &
    REC_PID=$!
}
start_recorder

# Restarts the camera recorder once if it is not saving frames.
check_frames() {
    [ -n "${NO_VIDEO:-}" ] && return 0
    local n
    n=$(find "$OUT/frames" -name '*.jpg' 2>/dev/null | wc -l)
    if [ "$n" -lt 10 ]; then
        echo "camera recorder stalled ($n frames); restarting it" >&2
        kill -9 "$REC_PID" 2>/dev/null
        start_recorder
        sleep 8
        n=$(find "$OUT/frames" -name '*.jpg' 2>/dev/null | wc -l)
    fi
    echo "camera frames so far: $n"
}
