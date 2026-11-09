#!/usr/bin/env python3
"""Plots trajectories and horizontal error for all estimators across the GNSS outage."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else 'results'
with open(os.path.join(out, 'timeseries.csv')) as f:
    rows = list(csv.DictReader(f))
metrics = json.load(open(os.path.join(out, 'metrics.json')))
col = lambda k: np.array([float(r[k]) for r in rows])
t = col('t')
tx, ty = col('true_x'), col('true_y')
o0, o1 = metrics['outage_s']
style = {'dr': ('Dead reckoning', 'tab:red', ':'), 'ekf': ('Error-state EKF', 'tab:blue', '-'),
         'pf': ('Bootstrap PF', 'tab:orange', '-'), 'rpf': ('Roughened PF', 'tab:purple', '--'),
         'mpf': ('Marginalized PF', 'tab:green', '-')}

fig, (a, b) = plt.subplots(1, 2, figsize=(16, 6.2), gridspec_kw={'width_ratios': [1.25, 1]})
a.plot(tx, ty, color='k', lw=3, label='Truth (Gazebo)', zorder=1)
for k, (name, c, ls) in style.items():
    a.plot(col(f'{k}_x'), col(f'{k}_y'), color=c, ls=ls, lw=1.4, label=name)
out_m = (t >= o0) & (t < o1)
a.plot(tx[out_m], ty[out_m], color='gold', lw=6, alpha=0.45, zorder=0, label='Flown during outage')
a.set_xlim(-30, 30)
a.set_ylim(-18, 18)
a.set_aspect('equal')
a.grid(alpha=0.3)
a.legend(fontsize=8, loc='upper left')
a.set_xlabel('East (m)')
a.set_ylabel('North (m)')
a.set_title('Figure-eight flight in Gazebo, GNSS jammed for 60 s (axes clipped)')

for k, (name, c, ls) in style.items():
    e = np.hypot(col(f'{k}_x') - tx, col(f'{k}_y') - ty)
    b.plot(t, np.maximum(e, 1e-3), color=c, ls=ls, lw=1.4, label=name)
b.axvspan(o0, o1, color='gold', alpha=0.25, label='GNSS jammed')
b.set_yscale('log')
b.set_xlabel('time since estimation start (s)')
b.set_ylabel('horizontal error (m, log)')
b.grid(alpha=0.3, which='both')
b.legend(fontsize=8)
b.set_title('Horizontal position error')
plt.tight_layout()
plt.savefig(os.path.join(out, 'result.png'), dpi=130)
print('wrote', os.path.join(out, 'result.png'))
