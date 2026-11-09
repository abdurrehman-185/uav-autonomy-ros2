"""Shared navigation pieces: planar EKF, scan-to-map ICP, attitude helpers and GNSS conversion."""
import math

import numpy as np

LAT0, LON0 = 0.35, 32.58      # Must match <spherical_coordinates> in make_world.world(). Near the equator so
                              # a degree of latitude and of longitude are the same length.
A_WGS, E2 = 6378137.0, 6.69437999014e-3


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def tilt_matrix(roll, pitch):
    cr, sr, cp, sp = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch)
    return np.array([[cp, sp * sr, sp * cr], [0.0, cr, -sr], [-sp, cp * sr, cp * cr]])


def rot2(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]])


def lla_to_enu(lat, lon):
    phi = math.radians(LAT0)
    s = math.sin(phi) ** 2
    rn = A_WGS * (1 - E2) / (1 - E2 * s) ** 1.5
    re = A_WGS / math.sqrt(1 - E2 * s)
    return math.radians(lon - LON0) * re * math.cos(phi), math.radians(lat - LAT0) * rn


def enu_to_lla(e, n):
    phi = math.radians(LAT0)
    s = math.sin(phi) ** 2
    rn = A_WGS * (1 - E2) / (1 - E2 * s) ** 1.5
    re = A_WGS / math.sqrt(1 - E2 * s)
    return LAT0 + math.degrees(n / rn), LON0 + math.degrees(e / (re * math.cos(phi)))


class AttitudeFilter:
    """Gyro-integrated roll/pitch with a slow pull toward the accelerometer."""

    def __init__(self, tau=15.0):
        self.roll = self.pitch = 0.0
        self.tau = tau

    def init(self, accel_mean):
        a = accel_mean
        self.roll = math.atan2(a[1], a[2])
        self.pitch = math.atan2(-a[0], math.hypot(a[1], a[2]))

    def update(self, w, a, dt):
        r, p = self.roll, self.pitch
        self.roll += (w[0] + math.sin(r) * math.tan(p) * w[1] + math.cos(r) * math.tan(p) * w[2]) * dt
        self.pitch += (math.cos(r) * w[1] - math.sin(r) * w[2]) * dt
        k = dt / self.tau
        self.roll += k * wrap(math.atan2(a[1], a[2]) - self.roll)
        self.pitch += k * wrap(math.atan2(-a[0], math.hypot(a[1], a[2])) - self.pitch)


class PlanarEkf:
    """State: x, y, vx, vy, yaw, gyro-z bias. IMU-driven prediction, position/pose updates."""

    def __init__(self, x0=0.0, y0=0.0, pos_std=1.0):
        self.x = np.array([x0, y0, 0.0, 0.0, 0.0, 0.0])
        self.P = np.diag([pos_std**2, pos_std**2, 0.01, 0.01, 1e-4, 1e-6])

    def predict(self, a_level, gz, dt, accel_psd=0.6):
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
        Q = np.diag([1e-6, 1e-6, accel_psd**2 * dt, accel_psd**2 * dt, (0.01**2) * dt, (1e-4**2) * dt])
        self.P = F @ self.P @ F.T + Q

    def innovation(self, z_xy, sigma):
        """Innovation and normalized innovation squared (NIS) for a position fix."""
        y = np.asarray(z_xy) - self.x[:2]
        S = self.P[:2, :2] + np.eye(2) * sigma**2
        return y, S, float(y @ np.linalg.solve(S, y))

    def update_position(self, z_xy, sigma):
        H = np.zeros((2, 6))
        H[0, 0] = H[1, 1] = 1.0
        y, S, _ = self.innovation(z_xy, sigma)
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[4] = wrap(self.x[4])
        self.P = (np.eye(6) - K @ H) @ self.P

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
    """Point-to-point ICP of level-frame scan points against map points.

    Returns (x, y, yaw, inlier ratio, rms)."""
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
        step = math.hypot(new[0] - x, new[1] - y)
        x, y, yaw = new[0], new[1], wrap(yaw + dyaw)
        gate = max(0.15, gate * 0.8)
        if abs(dyaw) < 1e-4 and step < 1e-3 and i > 3:
            break
    return x, y, yaw, float(inliers.mean()), float(np.sqrt((dist[inliers] ** 2).mean()))


class ScanMapper:
    """Scan-to-map localization with incremental map growth (2D lidar SLAM without loop closure)."""

    def __init__(self, keyframe_dist=0.5, map_radius=25.0, max_points=80000):
        self.map = None
        self.last_kf = None
        self.keyframe_dist = keyframe_dist
        self.map_radius = map_radius
        self.max_points = max_points

    @staticmethod
    def scan_points(msg, roll, pitch, stride=2):
        r = np.array(msg.ranges, dtype=float)
        ang = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        keep = np.isfinite(r) & (r > msg.range_min + 0.02) & (r < msg.range_max - 0.1)
        sel = np.zeros_like(keep)
        sel[::stride] = True
        keep &= sel
        if keep.sum() < 30:
            return None
        lx, ly = r[keep] * np.cos(ang[keep]), r[keep] * np.sin(ang[keep])
        p = tilt_matrix(roll, pitch) @ np.vstack([lx, ly, np.zeros_like(lx)])
        return p[:2].T

    def start(self, pts, pose):
        self.map = pts @ rot2(pose[2]).T + np.array(pose[:2])
        self.last_kf = pose[:2]

    def localize(self, pts, pose):
        near = ((self.map - np.array(pose[:2])) ** 2).sum(axis=1) < self.map_radius**2
        ref = self.map[near]
        if len(ref) < 30:
            return None
        return icp(pts, ref, pose) + (ref,)

    def grow(self, pts, pose, ref):
        if math.hypot(pose[0] - self.last_kf[0], pose[1] - self.last_kf[1]) < self.keyframe_dist:
            return
        world = pts @ rot2(pose[2]).T + np.array(pose[:2])
        _, d = nearest(world, ref)
        new = world[d > 0.12]
        if len(new):
            self.map = np.vstack([self.map, new])[-self.max_points:]
        self.last_kf = pose[:2]
