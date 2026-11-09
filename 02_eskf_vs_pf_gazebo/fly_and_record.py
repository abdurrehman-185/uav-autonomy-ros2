#!/usr/bin/env python3
"""Data-collection flight: a figure-eight at 6 m while recording raw IMU, GNSS and ground truth.

The pilot follows the reference using the simulator's ground truth, like a test pilot flying
a survey pattern. The estimators are not in the loop; replay_filters.py runs them afterwards
on the recorded sensor data only.
"""
import csv
import math
import os
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, NavSatFix

LAT0, LON0 = 0.35, 32.58      # Must match <spherical_coordinates> in the world.
A_WGS, E2 = 6378137.0, 6.69437999014e-3
ALT, AMP, PERIOD = 6.0, 18.0, 50.0
TAKEOFF_TIME = 10.0
MISSION_TIME = 150.0


def lla_to_enu(lat, lon):
    phi = math.radians(LAT0)
    s = math.sin(phi) ** 2
    rn = A_WGS * (1 - E2) / (1 - E2 * s) ** 1.5
    re = A_WGS / math.sqrt(1 - E2 * s)
    return math.radians(lon - LON0) * re * math.cos(phi), math.radians(lat - LAT0) * rn


def reference(t):
    """Figure eight; returns position and velocity at mission time t."""
    w = 2 * math.pi / PERIOD
    ramp = min(1.0, t / 6.0)
    x, y = AMP * math.sin(w * t), AMP / 2 * math.sin(2 * w * t)
    vx, vy = AMP * w * math.cos(w * t) * ramp, AMP * w * math.cos(2 * w * t) * ramp
    return x, y, vx, vy


class FlyAndRecord(Node):
    def __init__(self, out):
        super().__init__('fly_and_record')
        os.makedirs(out, exist_ok=True)
        self.files = {k: open(os.path.join(out, f'{k}.csv'), 'w', newline='') for k in ('imu', 'gps', 'truth')}
        self.w = {k: csv.writer(f) for k, f in self.files.items()}
        self.w['imu'].writerow(['t', 'ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        self.w['gps'].writerow(['t', 'e', 'n', 'u', 'lat', 'lon', 'alt'])
        self.w['truth'].writerow(['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'])
        self.truth = None
        self.t_start = None
        self.done = False
        self.pub = self.create_publisher(Twist, '/X3/gazebo/command/twist', 10)
        self.create_subscription(Imu, '/imu', self.on_imu, 1000)
        self.create_subscription(NavSatFix, '/navsat', self.on_gps, 50)
        self.create_subscription(Odometry, '/model/x3/odometry', self.on_truth, 200)

    @staticmethod
    def stamp(m):
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    def on_imu(self, m):
        a, g = m.linear_acceleration, m.angular_velocity
        self.w['imu'].writerow([f'{self.stamp(m):.4f}', a.x, a.y, a.z, g.x, g.y, g.z])

    def on_gps(self, m):
        e, n = lla_to_enu(m.latitude, m.longitude)
        self.w['gps'].writerow([f'{self.stamp(m):.4f}', f'{e:.4f}', f'{n:.4f}', f'{m.altitude:.4f}', m.latitude, m.longitude, m.altitude])

    def on_truth(self, m):
        t = self.stamp(m)
        p, q = m.pose.pose.position, m.pose.pose.orientation
        self.w['truth'].writerow([f'{t:.4f}', p.x, p.y, p.z, q.x, q.y, q.z, q.w])
        self.truth = (p.x, p.y, p.z, math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
        self.fly(t)

    def fly(self, t):
        if self.t_start is None:
            self.t_start = t
        x, y, z, yaw = self.truth
        tm = t - self.t_start - TAKEOFF_TIME
        if tm < 0:
            vx_w, vy_w, yaw_rate = 0.0, 0.0, -0.5 * yaw
        else:
            rx, ry, rvx, rvy = reference(tm)
            vx_w = rvx + 1.0 * (rx - x)
            vy_w = rvy + 1.0 * (ry - y)
            heading = math.atan2(rvy, rvx) if math.hypot(rvx, rvy) > 0.3 else yaw
            yaw_rate = max(-0.8, min(0.8, 1.5 * math.atan2(math.sin(heading - yaw), math.cos(heading - yaw))))
        c, s = math.cos(yaw), math.sin(yaw)
        cmd = Twist()
        cmd.linear.x, cmd.linear.y = c * vx_w + s * vy_w, -s * vx_w + c * vy_w
        cmd.linear.z = max(-1.0, min(1.0, 1.0 * (ALT - z)))
        cmd.angular.z = yaw_rate
        self.pub.publish(cmd)
        if tm > MISSION_TIME and not self.done:
            self.done = True
            for f in self.files.values():
                f.close()
            self.get_logger().info(f'recording complete at sim t={t:.1f}s')
        elif tm >= 0 and int(tm * 50) % 1000 == 0:
            self.get_logger().info(f'mission t={tm:5.1f}s pos=({x:.1f},{y:.1f},{z:.1f})')


def main():
    rclpy.init()
    node = FlyAndRecord(sys.argv[1] if len(sys.argv) > 1 else 'results')
    while rclpy.ok() and not node.done:
        rclpy.spin_once(node, timeout_sec=0.5)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
