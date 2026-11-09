#!/usr/bin/env python3
"""Scores a run against ground truth and saves a plot."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results'
with open(os.path.join(out, 'run.csv')) as f:
    rows = list(csv.DictReader(f))
num = lambda k: np.array([float(r[k]) for r in rows])
t, tx, ty, tz = num('t'), num('true_x'), num('true_y'), num('true_z')
ex, ey, ix, iy = num('est_x'), num('est_y'), num('imu_x'), num('imu_y')
tyaw, eyaw = num('true_yaw'), num('est_yaw')
flying = np.array([r['state'] in ('MISSION', 'LAND') for r in rows])
if not flying.any():
    flying = np.ones_like(t, dtype=bool)

err = np.hypot(tx - ex, ty - ey)
ierr = np.hypot(tx - ix, ty - iy)
yerr = np.degrees(np.abs((tyaw - eyaw + np.pi) % (2 * np.pi) - np.pi))
dist = float(np.hypot(np.diff(tx[flying]), np.diff(ty[flying])).sum())
status = open(os.path.join(out, 'status.txt')).read().strip()
metrics = {
    'status': status,
    'mission_seconds': round(float(t[flying][-1] - t[flying][0]), 1),
    'distance_flown_m': round(dist, 1),
    'lidar_ekf_rmse_m': round(float(np.sqrt((err[flying] ** 2).mean())), 3),
    'lidar_ekf_max_error_m': round(float(err[flying].max()), 3),
    'lidar_ekf_final_error_m': round(float(err[-1]), 3),
    'lidar_ekf_yaw_rmse_deg': round(float(np.sqrt((yerr[flying] ** 2).mean())), 2),
    'imu_only_rmse_m': round(float(np.sqrt((ierr[flying] ** 2).mean())), 2),
    'imu_only_final_error_m': round(float(ierr[-1]), 2),
    'true_end_position_m': [round(float(tx[-1]), 2), round(float(ty[-1]), 2), round(float(tz[-1]), 2)],
    'peak_true_altitude_m': round(float(tz.max()), 2),
}
with open(os.path.join(out, 'metrics.json'), 'w') as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))

fig, (a, b) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.5, 1]})
for x0, y0, w, h in [(-2.1, -7.1, 20.2, 0.2), (-2.1, 6.9, 20.2, 0.2), (-2.1, -7.1, 0.2, 14.2), (17.9, -7.1, 0.2, 14.2), (5.9, -7, 0.2, 9), (11.9, -2, 0.2, 9)]:
    a.add_patch(patches.Rectangle((x0, y0), w, h, color='0.35'))
if os.path.exists(os.path.join(out, 'map.npy')):
    m = np.load(os.path.join(out, 'map.npy'))
    a.scatter(m[:, 0], m[:, 1], s=1, c='tab:orange', alpha=0.5, label='Map built by the drone')
a.plot(ix, iy, color='tab:red', lw=1, ls=':', label='IMU only (no lidar)')
a.plot(tx, ty, color='k', lw=2.5, label='True path (Gazebo)')
a.plot(ex, ey, color='tab:cyan', lw=1.2, label='Lidar + IMU estimate')
a.set_xlim(-3.5, 19.5)
a.set_ylim(-8.5, 8.5)
a.set_aspect('equal')
a.grid(alpha=0.3)
a.set_xlabel('x (m)')
a.set_ylabel('y (m)')
a.set_title('GPS-denied flight: top view')
a.legend(loc='upper left', fontsize=8)

b.plot(t, err, color='tab:cyan', label='Lidar + IMU')
b.plot(t, ierr, color='tab:red', ls=':', label='IMU only')
b.set_yscale('log')
b.set_xlabel('time (s)')
b.set_ylabel('position error (m, log scale)')
b.set_title(f"RMSE: lidar+IMU {metrics['lidar_ekf_rmse_m']} m, IMU only {metrics['imu_only_rmse_m']} m")
b.grid(alpha=0.3, which='both')
b.legend()
plt.tight_layout()
plt.savefig(os.path.join(out, 'result.png'), dpi=130)
