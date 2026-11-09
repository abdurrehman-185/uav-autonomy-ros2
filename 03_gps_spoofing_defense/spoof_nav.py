#!/usr/bin/env python3
"""Delivery flight down a street, navigating on its own estimate while a spoofer attacks GNSS.

Modes (same airframe and route, different navigation):
  naive  GNSS + IMU EKF, every fix accepted.
  gated  GNSS + IMU EKF with a per-fix chi-square innovation gate (99%).
  lidar  GNSS + IMU + lidar scan-to-map EKF with a windowed NIS integrity monitor.
         When the monitor alarms, GNSS is isolated and the flight continues on lidar + IMU.

Ground truth is subscribed only for logging and crash detection. Nothing in the
navigation or control path reads it.
"""
import csv
import json
import math
import os
import sys
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan, NavSatFix
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from nav_core import AttitudeFilter, PlanarEkf, ScanMapper, lla_to_enu, tilt_matrix, wrap  # noqa: E402

CRUISE_ALT = 3.0
CRUISE_SPEED = 1.5
WP_RADIUS = 0.6
CALIB_TIME = 3.0
MAX_TIME = 200.0
WAYPOINTS = [(60.0, 0.0), (0.0, 0.0)]
GPS_SIGMA = 1.2
GATE_99 = 9.21           # chi-square, 2 dof, 99%
WINDOW = 25              # 5 s of 5 Hz fixes
WINDOW_LIMIT = 86.66 / 25  # chi-square, 50 dof, 99.9%, divided by window length
ALIGN_FIXES = 40         # 8 s of fixes used once to align the lidar map with the GNSS frame


class SpoofNav(Node):
    def __init__(self, mode, out):
        super().__init__('spoof_nav')
        self.mode, self.out = mode, out
        self.att = AttitudeFilter()
        self.ekf = None
        self.mapper = ScanMapper() if mode == 'lidar' else None
        self.t = self.t0 = None
        self.calib_g, self.calib_a, self.calib_gps = [], [], []
        self.bias_g = None
        self.alt = 0.0
        self.truth = None
        self.state = 'CALIB'
        self.wp = 0
        self.next_ctrl = self.next_log = self.next_print = 0.0
        self.gps_xy = (float('nan'), float('nan'))
        self.spoof = (0.0, 0.0)
        self.nis = float('nan')
        self.nis_window = deque(maxlen=WINDOW)
        self.gps_used = 0
        self.gps_isolated = False
        self.align = []
        self.aligned = mode != 'lidar'
        self.align_offset = (0.0, 0.0)
        self.alarm_t = None
        self.attack_t = None
        self.crash_t = None
        self.land_t = None
        self.rows = []
        self.done = False
        self.outcome = None
        self.pub = self.create_publisher(Twist, '/X3/gazebo/command/twist', 10)
        self.create_subscription(Imu, '/imu', self.on_imu, 1000)
        self.create_subscription(NavSatFix, '/navsat/received', self.on_gps, 20)
        self.create_subscription(LaserScan, '/range', self.on_range, 20)
        self.create_subscription(Odometry, '/model/x3/odometry', self.on_truth, 50)
        self.create_subscription(Float64MultiArray, '/spoofer/offset', self.on_spoof, 20)
        if self.mapper is not None:
            self.create_subscription(LaserScan, '/scan', self.on_scan, 5)

    @staticmethod
    def stamp(m):
        return m.header.stamp.sec + m.header.stamp.nanosec * 1e-9

    # Logging-only inputs ---------------------------------------------------
    def on_truth(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        tilt = math.acos(max(-1.0, min(1.0, 1 - 2 * (q.x * q.x + q.y * q.y))))
        self.truth = (p.x, p.y, p.z, tilt)

    def on_spoof(self, m):
        self.spoof = (m.data[1], m.data[2])
        if self.attack_t is None and math.hypot(*self.spoof) > 0:
            self.attack_t = m.data[0]
            self.get_logger().warning(f'spoofing attack started at t={m.data[0] - (self.t0 or 0):.1f}s')

    # Navigation inputs ---------------------------------------------------------
    def on_range(self, m):
        r = m.ranges[0] if len(m.ranges) else float('inf')
        if math.isfinite(r):
            self.alt = float(r) * math.cos(self.att.roll) * math.cos(self.att.pitch)

    def on_gps(self, m):
        e, n = lla_to_enu(m.latitude, m.longitude)
        self.gps_xy = (e, n)
        if self.state == 'CALIB':
            self.calib_gps.append((e, n))
            return
        if self.ekf is None or self.gps_isolated:
            return
        if not self.aligned and self.mapper.map is not None:
            # Map frame alignment: GNSS is assumed honest during takeoff. Average GNSS minus
            # lidar position, then shift the map and state once. Monitoring starts after this.
            self.align.append((e - self.ekf.x[0], n - self.ekf.x[1]))
            if len(self.align) >= ALIGN_FIXES:
                dx, dy = np.mean(self.align, axis=0)
                self.mapper.map = self.mapper.map + np.array([dx, dy])
                self.mapper.last_kf = (self.mapper.last_kf[0] + dx, self.mapper.last_kf[1] + dy)
                self.ekf.x[0] += dx
                self.ekf.x[1] += dy
                self.aligned = True
                self.align_offset = (float(dx), float(dy))
                self.get_logger().info(f'[lidar] map aligned to GNSS frame: shift ({dx:.2f},{dy:.2f}) m from {len(self.align)} fixes')
            return
        _, _, nis = self.ekf.innovation((e, n), GPS_SIGMA)
        self.nis = nis
        accept = True
        if self.mode in ('gated', 'lidar') and nis > GATE_99:
            accept = False
        if self.mode == 'lidar':
            self.nis_window.append(nis)
            if len(self.nis_window) == WINDOW and np.mean(self.nis_window) > WINDOW_LIMIT:
                self.gps_isolated = True
                self.alarm_t = self.t
                self.get_logger().error(f'GNSS INTEGRITY ALARM at t={self.t - self.t0:.1f}s: '
                                        f'mean NIS {np.mean(self.nis_window):.1f} over {WINDOW} fixes. GNSS isolated.')
                return
        if accept:
            self.ekf.update_position((e, n), GPS_SIGMA)
            self.gps_used += 1

    def on_scan(self, m):
        if self.ekf is None or self.state in ('CALIB', 'CRASHED') or self.done:
            return
        pts = ScanMapper.scan_points(m, self.att.roll, self.att.pitch)
        if pts is None:
            return
        pose = (self.ekf.x[0], self.ekf.x[1], self.ekf.x[4])
        if self.mapper.map is None:
            if self.alt > 2.0:
                self.mapper.start(pts, pose)
                self.get_logger().info(f'lidar map started with {len(self.mapper.map)} points')
            return
        res = self.mapper.localize(pts, pose)
        if res is None:
            return
        nx, ny, nyaw, ratio, rms, ref = res
        if ratio < 0.5 or rms > 0.2 or math.hypot(nx - pose[0], ny - pose[1]) > 1.0 or abs(wrap(nyaw - pose[2])) > 0.25:
            return
        self.ekf.update_pose((nx, ny, nyaw), sigma_xy=0.05, sigma_yaw=0.01)
        self.mapper.grow(pts, (self.ekf.x[0], self.ekf.x[1], self.ekf.x[4]), ref)

    def on_imu(self, m):
        t = self.stamp(m)
        g = np.array([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z])
        a = np.array([m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z])
        if self.t is None:
            self.t = self.t0 = t
            return
        dt = t - self.t
        self.t = t
        if dt <= 0 or dt > 0.5:
            return
        if self.state == 'CALIB':
            self.calib_g.append(g)
            self.calib_a.append(a)
            self.command(0, 0, 0, 0)
            if t - self.t0 >= CALIB_TIME and len(self.calib_gps) >= 5:
                self.bias_g = np.mean(self.calib_g, axis=0)
                self.att.init(np.mean(self.calib_a, axis=0))
                x0, y0 = np.mean(self.calib_gps, axis=0)
                self.ekf = PlanarEkf(x0, y0, pos_std=GPS_SIGMA / math.sqrt(len(self.calib_gps)))
                self.state = 'TAKEOFF'
                self.get_logger().info(f'[{self.mode}] initialised at ({x0:.2f},{y0:.2f}) from {len(self.calib_gps)} fixes; taking off')
            return
        w = g - self.bias_g
        self.att.update(w, a, dt)
        a_level = (tilt_matrix(self.att.roll, self.att.pitch) @ a)[:2]
        self.ekf.predict(a_level, w[2], dt)

        if self.truth is not None and self.state in ('MISSION', 'LAND') and self.crash_t is None:
            if self.truth[3] > 1.0 or (self.state == 'MISSION' and self.truth[2] < 0.8):
                self.crash_t = t
                self.state = 'CRASHED'
                self.get_logger().error(f'CRASH at t={t - self.t0:.1f}s, true position ({self.truth[0]:.1f},{self.truth[1]:.1f})')
        if t >= self.next_ctrl:
            self.next_ctrl = t + 0.05
            self.control(t)
        if t >= self.next_log and self.truth is not None:
            self.next_log = t + 0.1
            e = self.ekf.x
            self.rows.append([round(t - self.t0, 2), *[round(v, 3) for v in self.truth[:3]], round(e[0], 3), round(e[1], 3),
                              round(self.gps_xy[0], 3), round(self.gps_xy[1], 3), round(self.spoof[0], 3), round(self.spoof[1], 3),
                              round(self.nis, 3), int(self.gps_isolated), self.state])
        if t >= self.next_print:
            self.next_print = t + 10.0
            if self.truth:
                self.get_logger().info(f'[{self.mode}] t={t - self.t0:5.1f}s {self.state} est=({self.ekf.x[0]:.1f},{self.ekf.x[1]:.1f}) '
                                       f'true=({self.truth[0]:.1f},{self.truth[1]:.1f}) spoof={self.spoof[1]:.1f} nis={self.nis:.1f}')
        if (self.crash_t is not None and t - self.crash_t > 4.0) or t - self.t0 > MAX_TIME:
            self.finish('CRASHED' if self.crash_t is not None else 'TIMEOUT')

    # Guidance --------------------------------------------------------------------
    def command(self, vx_w, vy_w, vz, yaw_rate):
        yaw = self.ekf.x[4] if self.ekf is not None else 0.0
        c, s = math.cos(yaw), math.sin(yaw)
        msg = Twist()
        msg.linear.x, msg.linear.y = c * vx_w + s * vy_w, -s * vx_w + c * vy_w
        msg.linear.z, msg.angular.z = float(vz), float(yaw_rate)
        self.pub.publish(msg)

    def control(self, t):
        if self.state == 'CRASHED':
            self.command(0, 0, -0.5, 0)
            return
        e = self.ekf.x
        vz = float(np.clip(1.0 * (CRUISE_ALT - self.alt), -0.6, 0.6))
        yaw_rate = float(np.clip(-0.8 * e[4], -0.5, 0.5))
        if self.state == 'TAKEOFF':
            self.command(0, 0, vz, yaw_rate)
            ready = self.mapper is None or self.mapper.map is not None
            if abs(self.alt - CRUISE_ALT) < 0.15 and ready:
                self.state = 'MISSION'
                self.get_logger().info(f'[{self.mode}] mission started')
        elif self.state == 'MISSION':
            wx, wy = WAYPOINTS[self.wp]
            dx, dy = wx - e[0], wy - e[1]
            d = math.hypot(dx, dy)
            if d < WP_RADIUS:
                self.wp += 1
                self.get_logger().info(f'[{self.mode}] waypoint {self.wp}/{len(WAYPOINTS)} at t={t - self.t0:.1f}s')
                if self.wp >= len(WAYPOINTS):
                    self.state = 'LAND'
                return
            v = min(CRUISE_SPEED, 0.8 * d)
            self.command(v * dx / d, v * dy / d, vz, yaw_rate)
        elif self.state == 'LAND':
            self.command(0.4 * (WAYPOINTS[-1][0] - e[0]), 0.4 * (WAYPOINTS[-1][1] - e[1]), -0.4, yaw_rate)
            if self.alt < 0.15:
                self.land_t = self.land_t or t
                if t - self.land_t > 1.0:
                    self.finish('COMPLETED')

    def finish(self, outcome):
        if self.done:
            return
        self.done = True
        self.outcome = outcome
        self.command(0, 0, 0, 0)
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, 'run.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'true_x', 'true_y', 'true_z', 'est_x', 'est_y', 'gps_x', 'gps_y', 'spoof_x', 'spoof_y',
                        'nis', 'gps_isolated', 'state'])
            w.writerows(self.rows)
        summary = {
            'mode': self.mode,
            'outcome': outcome,
            'attack_start_s': round(self.attack_t - self.t0, 1) if self.attack_t else None,
            'alarm_s': round(self.alarm_t - self.t0, 1) if self.alarm_t else None,
            'detection_delay_s': round(self.alarm_t - self.attack_t, 1) if (self.alarm_t and self.attack_t) else None,
            'crash_s': round(self.crash_t - self.t0, 1) if self.crash_t else None,
            'gps_fixes_fused': self.gps_used,
            'map_alignment_shift_m': [round(v, 2) for v in self.align_offset],
            'waypoints_reached': self.wp,
        }
        with open(os.path.join(self.out, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        self.get_logger().info(f'[{self.mode}] finished: {summary}')


def main():
    mode = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else f'results/{mode}'
    rclpy.init()
    node = SpoofNav(mode, out)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        node.finish('INTERRUPTED')
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
