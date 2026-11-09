#!/usr/bin/env python3
"""GPS-denied drone navigation: IMU + 2D lidar EKF with scan-to-map matching.

The flight controller below is steered only by the estimate. Ground truth is subscribed
solely for logging, so it can score the run afterward. An IMU-only estimate is logged too,
to show what happens without the lidar.
"""
import csv
import math
import os
import sys

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan

G = 9.80665
CRUISE_ALT = 1.5
CRUISE_SPEED = 0.8
WP_RADIUS = 0.35
MAX_SIM_TIME = 420.0
CALIB_TIME = 3.0
# Out through the S-shaped route, then back to the start.
OUTBOUND = [(3.0, 4.5), (9.0, 4.5), (9.0, -4.5), (15.0, -4.5), (15.0, 3.0)]
WAYPOINTS = OUTBOUND + OUTBOUND[-2::-1] + [(0.0, 0.0)]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def tilt_matrix(roll, pitch):
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]])


def rot2(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


class Ekf:
    """State: x, y, vx, vy, yaw, gyro-z bias."""

    def __init__(self):
        self.x = np.zeros(6)
        self.P = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-6])

    def predict(self, a_level, gz, dt):
        px, py, vx, vy, yaw, bg = self.x
        c, s = math.cos(yaw), math.sin(yaw)
        ax = c * a_level[0] - s * a_level[1]
        ay = s * a_level[0] + c * a_level[1]
        self.x = np.array([px + vx * dt, py + vy * dt, vx + ax * dt, vy + ay * dt, wrap(yaw + (gz - bg) * dt), bg])
        F = np.eye(6)
        F[0, 2] = F[1, 3] = dt
        F[2, 4] = -ay * dt
        F[3, 4] = ax * dt
        F[4, 5] = -dt
        Q = np.diag([1e-6, 1e-6, (0.6**2) * dt, (0.6**2) * dt, (0.01**2) * dt, (1e-4**2) * dt])
        self.P = F @ self.P @ F.T + Q

    def update_pose(self, z, sigma_xy=0.03, sigma_yaw=0.01):
        H = np.zeros((3, 6))
        H[0, 0] = H[1, 1] = H[2, 4] = 1.0
        R = np.diag([sigma_xy**2, sigma_xy**2, sigma_yaw**2])
        y = np.array([z[0] - self.x[0], z[1] - self.x[1], wrap(z[2] - self.x[4])])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[4] = wrap(self.x[4])
        self.P = (np.eye(6) - K @ H) @ self.P


def nearest(pts, ref):
    d2 = ((pts[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2)
    idx = d2.argmin(axis=1)
    return idx, np.sqrt(d2[np.arange(len(pts)), idx])


def icp(scan_body, map_pts, pose, iters=12):
    """Align level-frame scan points to the map. Returns (x, y, yaw, inlier ratio, rms)."""
    x, y, yaw = pose
    gate = 0.6
    inliers = np.zeros(len(scan_body), dtype=bool)
    dist = np.zeros(len(scan_body))
    for i in range(iters):
        world = scan_body @ rot2(yaw).T + np.array([x, y])
        idx, dist = nearest(world, map_pts)
        inliers = dist < gate
        if inliers.sum() < 20:
            return x, y, yaw, 0.0, 1.0
        a, b = world[inliers], map_pts[idx[inliers]]
        ca, cb = a.mean(axis=0), b.mean(axis=0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        Rd = Vt.T @ U.T
        if np.linalg.det(Rd) < 0:
            Vt[1] *= -1
            Rd = Vt.T @ U.T
        dyaw = math.atan2(Rd[1, 0], Rd[0, 0])
        t = cb - Rd @ ca
        new = Rd @ np.array([x, y]) + t
        x, y, yaw = new[0], new[1], wrap(yaw + dyaw)
        gate = max(0.15, gate * 0.8)
        if abs(dyaw) < 1e-4 and np.hypot(t[0] + (Rd @ ca - ca)[0], t[1] + (Rd @ ca - ca)[1]) < 1e-3 and i > 3:
            break
    return x, y, yaw, float(inliers.mean()), float(np.sqrt((dist[inliers] ** 2).mean()))


class Nav(Node):
    def __init__(self, out_dir):
        super().__init__('gps_denied_nav')
        self.out_dir = out_dir
        self.ekf = Ekf()
        self.t = None
        self.t0 = None
        self.calib_g, self.calib_a = [], []
        self.bias_g = None
        self.roll = self.pitch = 0.0
        self.dr = np.zeros(5)  # IMU-only baseline: x, y, vx, vy, yaw
        self.alt = 0.0
        self.got_range = False
        self.map = None
        self.last_kf = None
        self.truth = None
        self.state = 'CALIB'
        self.wp = 0
        self.next_ctrl = 0.0
        self.next_log = 0.0
        self.next_print = 0.0
        self.land_t = None
        self.scans_used = 0
        self.scans_rejected = 0
        self.rows = []
        self.done = False
        self.cmd_pub = self.create_publisher(Twist, '/X3/gazebo/command/twist', 10)
        self.create_subscription(Imu, '/imu', self.on_imu, 500)
        self.create_subscription(LaserScan, '/scan', self.on_scan, 5)
        self.create_subscription(LaserScan, '/range', self.on_range, 10)
        self.create_subscription(Odometry, '/model/x3/odometry', self.on_truth, 50)
        self.get_logger().info('waiting for sensors')

    def on_truth(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.truth = (p.x, p.y, p.z, yaw)

    def on_range(self, m):
        r = m.ranges[0] if len(m.ranges) else float('inf')
        if not math.isfinite(r):
            r = 0.0 if r < 0 or not self.got_range else self.alt
        self.alt = float(r) * math.cos(self.roll) * math.cos(self.pitch)
        self.got_range = True

    def on_imu(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if t <= 0:
            return
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
            if t - self.t0 >= CALIB_TIME:
                self.bias_g = np.mean(self.calib_g, axis=0)
                am = np.mean(self.calib_a, axis=0)
                self.roll = math.atan2(am[1], am[2])
                self.pitch = math.atan2(-am[0], math.hypot(am[1], am[2]))
                self.state = 'TAKEOFF'
                self.get_logger().info(f'calibrated at t={t:.1f}s; gyro bias {self.bias_g.round(5)}; taking off')
            self.publish_cmd(0, 0, 0, 0)
            return

        w = g - self.bias_g
        self.roll += (w[0] + math.sin(self.roll) * math.tan(self.pitch) * w[1] + math.cos(self.roll) * math.tan(self.pitch) * w[2]) * dt
        self.pitch += (math.cos(self.roll) * w[1] - math.sin(self.roll) * w[2]) * dt
        # A very slow pull toward the accelerometer, which on a multirotor only weakly observes tilt.
        k = dt / 15.0
        self.roll += k * wrap(math.atan2(a[1], a[2]) - self.roll)
        self.pitch += k * wrap(math.atan2(-a[0], math.hypot(a[1], a[2])) - self.pitch)

        a_w = tilt_matrix(self.roll, self.pitch) @ a
        a_level = a_w[:2]
        self.ekf.predict(a_level, g[2] - self.bias_g[2], dt)

        # IMU-only baseline.
        self.dr[4] = wrap(self.dr[4] + w[2] * dt)
        c, s = math.cos(self.dr[4]), math.sin(self.dr[4])
        self.dr[0] += self.dr[2] * dt
        self.dr[1] += self.dr[3] * dt
        self.dr[2] += (c * a_level[0] - s * a_level[1]) * dt
        self.dr[3] += (s * a_level[0] + c * a_level[1]) * dt

        if t >= self.next_ctrl:
            self.next_ctrl = t + 0.05
            self.control(t)
        if t >= self.next_log and self.truth is not None:
            self.next_log = t + 0.1
            e = self.ekf.x
            self.rows.append([t - self.t0, *self.truth, e[0], e[1], e[4], self.alt, self.dr[0], self.dr[1], self.state, self.wp])
        if t >= self.next_print:
            self.next_print = t + 10.0
            err = math.hypot(self.truth[0] - self.ekf.x[0], self.truth[1] - self.ekf.x[1]) if self.truth else float('nan')
            self.get_logger().info(
                f't={t - self.t0:5.1f}s {self.state} wp {self.wp}/{len(WAYPOINTS)} est=({self.ekf.x[0]:.2f},{self.ekf.x[1]:.2f}) '
                f'alt={self.alt:.2f} err={err:.3f}m map={0 if self.map is None else len(self.map)}pts used={self.scans_used} rej={self.scans_rejected}')
        if t - self.t0 > MAX_SIM_TIME and not self.done:
            self.get_logger().warning('time limit reached')
            self.state = 'TIMEOUT'
            self.finish()

    def on_scan(self, m):
        if self.state == 'CALIB' or self.done:
            return
        r = np.array(m.ranges, dtype=float)
        ang = m.angle_min + np.arange(len(r)) * m.angle_increment
        keep = np.isfinite(r) & (r > m.range_min + 0.02) & (r < m.range_max - 0.1)
        keep[1::2] = False
        if keep.sum() < 30:
            return
        lx, ly = r[keep] * np.cos(ang[keep]), r[keep] * np.sin(ang[keep])
        p = tilt_matrix(self.roll, self.pitch) @ np.vstack([lx, ly, np.zeros_like(lx)])
        pts = p[:2].T
        x, y, yaw = self.ekf.x[0], self.ekf.x[1], self.ekf.x[4]

        if self.map is None:
            if self.alt < 1.0:  # Start the map once clear of the floor.
                return
            self.map = pts @ rot2(yaw).T + np.array([x, y])
            self.last_kf = (x, y)
            self.get_logger().info(f'map started with {len(self.map)} points')
            return

        near = ((self.map - np.array([x, y])) ** 2).sum(axis=1) < 14.0**2
        ref = self.map[near]
        if len(ref) < 30:
            return
        nx, ny, nyaw, ratio, rms = icp(pts, ref, (x, y, yaw))
        jump = math.hypot(nx - x, ny - y)
        if ratio < 0.55 or rms > 0.15 or jump > 0.8 or abs(wrap(nyaw - yaw)) > 0.25:
            self.scans_rejected += 1
            return
        self.ekf.update_pose((nx, ny, nyaw))
        self.scans_used += 1

        ex, ey = self.ekf.x[0], self.ekf.x[1]
        if math.hypot(ex - self.last_kf[0], ey - self.last_kf[1]) > 0.5:
            world = pts @ rot2(self.ekf.x[4]).T + np.array([ex, ey])
            _, d = nearest(world, ref)
            new = world[d > 0.1]
            if len(new):
                self.map = np.vstack([self.map, new])
            self.last_kf = (ex, ey)

    def publish_cmd(self, vx_w, vy_w, vz, yaw_rate):
        yaw = self.ekf.x[4]
        c, s = math.cos(yaw), math.sin(yaw)
        m = Twist()
        m.linear.x = c * vx_w + s * vy_w
        m.linear.y = -s * vx_w + c * vy_w
        m.linear.z = float(vz)
        m.angular.z = float(yaw_rate)
        self.cmd_pub.publish(m)

    def control(self, t):
        e = self.ekf.x
        vz_hold = float(np.clip(1.0 * (CRUISE_ALT - self.alt), -0.5, 0.5))
        yaw_rate = float(np.clip(-0.8 * e[4], -0.5, 0.5))
        if self.state == 'TAKEOFF':
            self.publish_cmd(0, 0, vz_hold, yaw_rate)
            if abs(self.alt - CRUISE_ALT) < 0.1 and self.map is not None:
                self.state = 'MISSION'
                self.get_logger().info('at altitude; starting mission')
        elif self.state == 'MISSION':
            wx, wy = WAYPOINTS[self.wp]
            dx, dy = wx - e[0], wy - e[1]
            d = math.hypot(dx, dy)
            if d < WP_RADIUS:
                self.get_logger().info(f'reached waypoint {self.wp + 1}/{len(WAYPOINTS)} at t={t - self.t0:.1f}s')
                self.wp += 1
                if self.wp >= len(WAYPOINTS):
                    self.state = 'LAND'
                    self.get_logger().info('mission complete; landing')
                return
            speed = min(CRUISE_SPEED, 0.8 * d)
            self.publish_cmd(speed * dx / d, speed * dy / d, vz_hold, yaw_rate)
        elif self.state == 'LAND':
            kp = 0.8
            self.publish_cmd(kp * (0 - e[0]) * 0.5, kp * (0 - e[1]) * 0.5, -0.3, yaw_rate)
            if self.alt < 0.1:
                if self.land_t is None:
                    self.land_t = t
                if t - self.land_t > 1.0:
                    self.state = 'LANDED'
                    self.get_logger().info('landed')
                    self.finish()

    def finish(self):
        if self.done:
            return
        self.done = True
        self.publish_cmd(0, 0, 0, 0)
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, 'run.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['t', 'true_x', 'true_y', 'true_z', 'true_yaw', 'est_x', 'est_y', 'est_yaw', 'alt', 'imu_x', 'imu_y', 'state', 'wp'])
            w.writerows(self.rows)
        if self.map is not None:
            np.save(os.path.join(self.out_dir, 'map.npy'), self.map)
        with open(os.path.join(self.out_dir, 'status.txt'), 'w') as f:
            f.write(f'{self.state} waypoints={self.wp}/{len(WAYPOINTS)} scans_used={self.scans_used} scans_rejected={self.scans_rejected}\n')
        self.get_logger().info('log saved')


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else 'results'
    rclpy.init()
    node = Nav(out_dir)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.5)
    except KeyboardInterrupt:
        node.finish()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
