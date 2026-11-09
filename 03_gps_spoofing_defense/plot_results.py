#!/usr/bin/env python3
"""Compares the three spoofing runs, writes metrics, a summary table, plots and video configs."""
import csv
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from make_world import POLES, buildings

root = sys.argv[1] if len(sys.argv) > 1 else 'results'
MODES = [m for m in ('naive', 'gated', 'lidar') if os.path.exists(os.path.join(root, m, 'run.csv'))]
TITLES = {'naive': 'GNSS + IMU (no checks)', 'gated': 'GNSS + IMU + chi-square gate',
          'lidar': 'GNSS + IMU + lidar integrity monitor'}
RECTS = buildings()


def clearance(x, y):
    """Distance from a point to the nearest building footprint or pole surface."""
    best = 1e9
    for x0, y0, w, h in RECTS:
        dx = max(x0 - x, 0, x - (x0 + w))
        dy = max(y0 - y, 0, y - (y0 + h))
        best = min(best, math.hypot(dx, dy))
    for px, py in POLES:
        best = min(best, math.hypot(x - px, y - py) - 0.18)
    return best


data, table = {}, {}
for m in MODES:
    with open(os.path.join(root, m, 'run.csv')) as f:
        rows = list(csv.DictReader(f))
    col = lambda k: np.array([float(r[k]) for r in rows])
    d = {k: col(k) for k in ('t', 'true_x', 'true_y', 'true_z', 'est_x', 'est_y', 'gps_x', 'gps_y', 'spoof_y', 'nis')}
    d['state'] = [r['state'] for r in rows]
    s = json.load(open(os.path.join(root, m, 'summary.json')))
    flying = np.array([st in ('MISSION', 'LAND', 'CRASHED') for st in d['state']]) & (d['true_z'] > 0.5)
    clear = np.array([clearance(x, y) for x, y in zip(d['true_x'], d['true_y'])])
    err = np.hypot(d['est_x'] - d['true_x'], d['est_y'] - d['true_y'])
    if s['outcome'] != 'CRASHED' and flying.any() and clear[flying].min() < 0.35:
        s['outcome'] = 'HIJACKED INTO BUILDING'
    s.update({
        'max_true_cross_track_m': round(float(np.abs(d['true_y'][flying]).max()), 2) if flying.any() else None,
        'min_clearance_to_obstacles_m': round(float(clear[flying].min()), 2) if flying.any() else None,
        'max_navigation_error_m': round(float(err[flying].max()), 2) if flying.any() else None,
        'max_spoof_offset_m': round(float(np.abs(d['spoof_y']).max()), 2),
    })
    json.dump(s, open(os.path.join(root, m, 'summary.json'), 'w'), indent=2)
    data[m], table[m] = d, s

lines = ['| Navigation | Outcome | Spoof detected after | Max true cross-track (m) | Min clearance to obstacles (m) | Max navigation error (m) |',
         '|---|---|---|---|---|---|']
for m in MODES:
    s = table[m]
    det = f"{s['detection_delay_s']} s" if s['detection_delay_s'] is not None else 'never'
    lines.append(f"| {TITLES[m]} | {s['outcome']} | {det} | {s['max_true_cross_track_m']} | {s['min_clearance_to_obstacles_m']} | {s['max_navigation_error_m']} |")
open(os.path.join(root, 'summary.md'), 'w').write('\n'.join(lines) + '\n')
json.dump(table, open(os.path.join(root, 'metrics.json'), 'w'), indent=2)
print('\n'.join(lines))

fig, axes = plt.subplots(len(MODES) + 1, 1, figsize=(13, 3.6 * (len(MODES) + 1)))
for ax, m in zip(axes, MODES):
    d, s = data[m], table[m]
    for x0, y0, w, h in RECTS:
        ax.add_patch(patches.Rectangle((x0, y0), w, h, color='0.55'))
    for px, py in POLES:
        ax.add_patch(patches.Circle((px, py), 0.4, color='0.3'))
    ax.plot(d['gps_x'], d['gps_y'], '.', ms=2, color='tab:red', alpha=0.5, label='GNSS fixes received (spoofed)')
    ax.plot(d['true_x'], d['true_y'], color='k', lw=2.5, label='True path')
    ax.plot(d['est_x'], d['est_y'], color='tab:cyan', lw=1.3, label='What the drone believed')
    ax.axhline(0, color='tab:green', ls='--', lw=1, label='Planned route')
    ax.set_xlim(-8, 72)
    ax.set_ylim(-14, 14)
    ax.set_aspect('equal')
    det = f", spoof detected after {s['detection_delay_s']} s" if s['detection_delay_s'] is not None else ''
    ax.set_title(f"{TITLES[m]}: {s['outcome']}{det}")
    ax.legend(fontsize=7, loc='lower right', ncol=4)
ax = axes[-1]
for m, c in zip(MODES, ('tab:red', 'tab:orange', 'tab:green')):
    ax.plot(data[m]['t'], data[m]['true_y'], color=c, lw=2, label=f'{TITLES[m]}: true cross-track')
ax.plot(data[MODES[0]]['t'], data[MODES[0]]['spoof_y'], color='k', ls=':', label='Spoofer offset (north is +)')
ax.axhline(7.0, color='0.4', ls='--', lw=1, label='Building line')
ax.set_xlabel('time (s)')
ax.set_ylabel('y (m)')
ax.grid(alpha=0.3)
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(root, 'result.png'), dpi=120)

walls = [list(r) for r in RECTS]
for m in MODES:
    s = table[m]
    events = []
    if s['attack_start_s'] is not None:
        events.append({'t': s['attack_start_s'], 'until': s['alarm_s'] or 1e9, 'text': 'GNSS SPOOFING ATTACK'})
    if s['alarm_s'] is not None:
        events.append({'t': s['alarm_s'], 'until': 1e9, 'text': 'SPOOF DETECTED: GNSS ISOLATED'})
    if s['crash_s'] is not None:
        events.append({'t': s['crash_s'], 'until': 1e9, 'text': 'CRASH'})
    cfg = {
        'title': f'GNSS spoofing attack: {TITLES[m]}', 'title_scale': 0.52,
        'run_csv': f'{m}/run.csv',
        'frames': {'chase': f'{m}/frames/chase_cam', 'overhead': f'{m}/frames/overhead_cam'},
        'truth': ['true_x', 'true_y'],
        'series': [
            {'x': 'gps_x', 'y': 'gps_y', 'label': 'GNSS received', 'color': [60, 60, 230], 'width': 1, 'error': False},
            {'x': 'est_x', 'y': 'est_y', 'label': 'Drone belief', 'color': [230, 200, 40], 'width': 2},
        ],
        'bounds': [-8, 72, -16, 16], 'walls': walls, 'circles': [[px, py, 0.4] for px, py in POLES],
        'waypoints': [[60, 0], [0, 0]], 'error_ylim': 12, 'events': events, 'speed': 2,
        'out': f'../media/spoofing_{m}.mp4', 'gif': f'../media/spoofing_{m}.gif', 'gif_start': 10, 'gif_seconds': 16,
    }
    json.dump(cfg, open(os.path.join(root, f'video_{m}.json'), 'w'), indent=2)
print('wrote summary.md, metrics.json, result.png, video configs')
