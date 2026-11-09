#!/usr/bin/env python3
"""Replays a recorded Gazebo flight through five estimators with a 60 s GNSS outage.

Estimators (from estimators/gps_denied_navigation): inertial dead reckoning, 15-state
error-state EKF, bootstrap PF, roughened PF and marginalized (Rao-Blackwellized) PF.
They receive only IMU and GNSS packets through the package's step() interface. Ground
truth is used for the initial state (as the package's initialize_from_truth option does)
and for scoring.
"""
import csv
import json
import math
import os
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'estimators'))
from gps_denied_navigation.config import FilterConfig, GPSConfig, IMUConfig, ParticleFilterConfig  # noqa: E402
from gps_denied_navigation.filters.base_filter import NavigationState  # noqa: E402
from gps_denied_navigation.filters.dead_reckoning import DeadReckoning  # noqa: E402
from gps_denied_navigation.filters.ekf import ExtendedKalmanFilter  # noqa: E402
from gps_denied_navigation.filters.interface import EstimatorInput, step  # noqa: E402
from gps_denied_navigation.filters.particle_filter import ParticleFilter  # noqa: E402
from gps_denied_navigation.filters.pf_variants import MarginalizedParticleFilter, RoughenedParticleFilter  # noqa: E402
from gps_denied_navigation.filters.quaternion import from_rpy  # noqa: E402
from gps_denied_navigation.sensors.sensor_models import GPSMeasurement, IMUMeasurement  # noqa: E402

import make_world as sensor_spec  # noqa: E402  Same noise numbers the simulator used.

START_AFTER = 8.0          # s after the first truth sample: drone hovering at altitude
OUTAGE = (50.0, 110.0)     # s after estimation start: GNSS jammed
FILTER_RATE = 100          # IMU pairs averaged from 200 Hz to 100 Hz
PF_SEEDS = 5
PARTICLES = 1000
LOG_DT = 0.1

IMU_CFG = IMUConfig(frequency=FILTER_RATE, accel_noise_std=sensor_spec.ACCEL_STD / math.sqrt(2),
                    gyro_noise_std=sensor_spec.GYRO_STD / math.sqrt(2),
                    accel_bias_random_walk=5e-4, gyro_bias_random_walk=2e-6)
GPS_CFG = GPSConfig(frequency=sensor_spec.GPS_RATE, position_noise_std=sensor_spec.GPS_STD_H)
FILTER_CFG = FilterConfig(initialize_from_truth=True, initial_position_std=0.5, initial_velocity_std=0.2,
                          initial_attitude_std=0.02, initial_accel_bias_std=0.08, initial_gyro_bias_std=0.002)
PF_CFG = ParticleFilterConfig(particles=PARTICLES)


def read_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        return np.array([[float(v) for v in row[:8]] for row in r if row])


def quat_to_rpy(qx, qy, qz, qw):
    roll = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    return roll, pitch, yaw


def load(results):
    imu, gps, truth = (read_csv(os.path.join(results, f'{k}.csv')) for k in ('imu', 'gps', 'truth'))
    t_start = truth[0, 0] + START_AFTER
    imu = imu[imu[:, 0] >= t_start - 0.02]
    n = (len(imu) // 2) * 2
    pairs = imu[:n].reshape(-1, 2, 7)
    imu100 = np.column_stack([pairs[:, 1, 0], pairs[:, :, 1:].mean(axis=1)])
    gps = gps[gps[:, 0] > t_start]
    truth = truth[truth[:, 0] >= t_start - 0.1]
    return t_start, imu100, gps, truth


def packets(t_start, imu, gps):
    out = []
    gi = 0
    prev = imu[0, 0]
    for k in range(1, len(imu)):
        t = imu[k, 0]
        dt = t - prev
        if dt <= 0:
            continue
        m = IMUMeasurement(time=0.5 * (t + prev), dt=dt, acceleration=imu[k, 1:4], angular_velocity=imu[k, 4:7])
        fix = None
        while gi < len(gps) and gps[gi, 0] <= t:
            if gps[gi, 0] > prev:
                fix = gps[gi]
            gi += 1
        rel = t - t_start
        jammed = OUTAGE[0] <= rel < OUTAGE[1]
        if fix is not None and not jammed:
            g = GPSMeasurement(time=t, available=True, position=fix[1:4])
        else:
            g = GPSMeasurement(time=t, available=False, reason='jammed' if jammed else 'no fix this epoch')
        out.append(EstimatorInput(k, t, m, g))
        prev = t
    return out


def initial_state(truth, t_start):
    i = int(np.searchsorted(truth[:, 0], t_start))
    j = int(np.searchsorted(truth[:, 0], t_start + 0.2))
    p = truth[i, 1:4]
    v = (truth[j, 1:4] - truth[i, 1:4]) / (truth[j, 0] - truth[i, 0])
    rpy = quat_to_rpy(*truth[i, 4:8])
    return NavigationState(np.array(p), np.array(v), from_rpy(np.array(rpy)), np.zeros(3), np.zeros(3))


def make(name, init, seed):
    rng = np.random.default_rng(seed)
    if name == 'DR':
        return DeadReckoning(init, IMU_CFG.gravity)
    if name == 'EKF':
        return ExtendedKalmanFilter(init, IMU_CFG, GPS_CFG, FILTER_CFG)
    cls = {'PF': ParticleFilter, 'RPF': RoughenedParticleFilter, 'MPF': MarginalizedParticleFilter}[name]
    return cls(init, IMU_CFG, GPS_CFG, FILTER_CFG, PF_CFG, rng)


def run_one(job):
    name, seed, results = job
    t_start, imu, gps, truth = load(results)
    pk = packets(t_start, imu, gps)
    est = make(name, initial_state(truth, t_start), seed)
    t0 = time.perf_counter()
    rows, next_log = [], pk[0].time
    nees = []
    for p in pk:
        out = step(est, p)
        if p.time >= next_log:
            next_log += LOG_DT
            rows.append((p.time, *out.position))
            if out.position_covariance is not None:
                rows[-1] = rows[-1] + (out.position_covariance,)
    return name, seed, rows, time.perf_counter() - t0, len(pk)


def interp_truth(truth, times):
    return np.column_stack([np.interp(times, truth[:, 0], truth[:, c]) for c in (1, 2, 3)])


def main():
    results = sys.argv[1] if len(sys.argv) > 1 else 'results'
    t_start, imu, gps, truth = load(results)
    print(f'IMU epochs: {len(imu)}, GNSS fixes: {len(gps)}, duration {imu[-1, 0] - imu[0, 0]:.1f} s')

    # Sanity check: GNSS (converted from lat/lon) against truth before the outage.
    gt = interp_truth(truth, gps[:, 0])
    d = gps[:, 1:4] - gt
    pre = (gps[:, 0] - t_start) < OUTAGE[0]
    gnss_check = {'mean_offset_m': d[pre].mean(axis=0).round(3).tolist(), 'std_m': d[pre].std(axis=0).round(3).tolist()}
    print('GNSS minus truth:', gnss_check)

    jobs = [('DR', 0, results), ('EKF', 0, results)]
    jobs += [(n, s, results) for n in ('PF', 'RPF', 'MPF') for s in range(PF_SEEDS)]
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 4), mp_context=mp.get_context("fork")) as ex:
        outs = list(ex.map(run_one, jobs))

    names = {'DR': 'Dead reckoning', 'EKF': 'Error-state EKF', 'PF': 'Bootstrap PF',
             'RPF': 'Roughened PF', 'MPF': 'Marginalized PF'}
    metrics = {'gnss_check': gnss_check, 'outage_s': OUTAGE, 'particles': PARTICLES, 'pf_seeds': PF_SEEDS, 'estimators': {}}
    traces = {}
    for name, seed, rows, runtime, n_epochs in outs:
        arr = np.array([r[:4] for r in rows])
        tt = arr[:, 0]
        rel = tt - t_start
        tr = interp_truth(truth, tt)
        err_h = np.hypot(arr[:, 1] - tr[:, 0], arr[:, 2] - tr[:, 1])
        err_3d = np.linalg.norm(arr[:, 1:4] - tr, axis=1)
        gnss = rel < OUTAGE[0]
        out_m = (rel >= OUTAGE[0]) & (rel < OUTAGE[1])
        after = rel >= OUTAGE[1]
        end_i = int(np.searchsorted(rel, OUTAGE[1])) - 1
        rec = None
        if after.any():
            ok = np.where(after & (err_h < 3.0))[0]
            rec = round(float(rel[ok[0]] - OUTAGE[1]), 1) if len(ok) else None
        m = {
            'rmse_with_gnss_m': float(np.sqrt((err_h[gnss] ** 2).mean())),
            'rmse_during_outage_m': float(np.sqrt((err_h[out_m] ** 2).mean())),
            'error_at_outage_end_m': float(err_h[end_i]),
            'max_3d_error_during_outage_m': float(err_3d[out_m].max()),
            'recovery_s_after_gnss_returns': rec,
            'runtime_s': round(runtime, 1),
            'us_per_epoch': round(runtime / n_epochs * 1e6, 1),
        }
        if len(rows[0]) > 4:
            nees = []
            for r, e in zip(rows, arr[:, 1:4] - tr):
                try:
                    nees.append(float(e @ np.linalg.solve(np.asarray(r[4])[:3, :3], e)))
                except np.linalg.LinAlgError:
                    pass
            nees = np.array(nees)
            m['median_position_nees_outage'] = float(np.median(nees[out_m[:len(nees)]])) if len(nees) else None
        metrics['estimators'].setdefault(name, []).append(m)
        traces.setdefault(name, []).append((seed, arr, err_h))

    summary = {}
    for name, runs in metrics['estimators'].items():
        agg = {}
        for k in runs[0]:
            vals = [r[k] for r in runs if r[k] is not None]
            if not vals or isinstance(vals[0], str):
                continue
            agg[k] = round(float(np.median(vals)), 3)
            if len(runs) > 1:
                agg[k + '_min'] = round(float(np.min(vals)), 3)
                agg[k + '_max'] = round(float(np.max(vals)), 3)
        summary[name] = agg
    metrics['summary_median_over_seeds'] = summary
    with open(os.path.join(results, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    order = ['DR', 'EKF', 'PF', 'RPF', 'MPF']
    lines = ['| Estimator | RMSE with GNSS (m) | RMSE in 60 s outage (m) | Error at outage end (m) | Recovery after GNSS returns (s) | Time per epoch (us) |',
             '|---|---|---|---|---|---|']
    for n in order:
        s = summary[n]
        span = lambda k: f"{s[k]:.2f}" + (f" ({s[k + '_min']:.2f}-{s[k + '_max']:.2f})" if k + '_min' in s else '')
        rec = s.get('recovery_s_after_gnss_returns')
        lines.append(f"| {names[n]} | {span('rmse_with_gnss_m')} | {span('rmse_during_outage_m')} | "
                     f"{span('error_at_outage_end_m')} | {rec if rec is not None else 'none'} | {s['us_per_epoch']:.0f} |")
    table = '\n'.join(lines)
    with open(os.path.join(results, 'summary.md'), 'w') as f:
        f.write(table + '\n')
    print(table)

    # Timeseries for plots and video: the median-RMSE seed of each PF type.
    pick = {}
    for n, runs in traces.items():
        scores = [np.sqrt(np.mean(e ** 2)) for _, _, e in runs]
        pick[n] = runs[int(np.argsort(scores)[len(scores) // 2])]
    tt = pick['EKF'][1][:, 0]
    tr = interp_truth(truth, tt)
    with open(os.path.join(results, 'timeseries.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['t', 'true_x', 'true_y', 'true_z'] + [f'{n.lower()}_{c}' for n in order for c in 'xyz'] + ['state'])
        for k, t in enumerate(tt):
            rel = t - t_start
            state = 'GNSS JAMMED' if OUTAGE[0] <= rel < OUTAGE[1] else 'GNSS OK'
            row = [round(rel, 2), *tr[k].round(3)]
            for n in order:
                a = pick[n][1]
                j = min(k, len(a) - 1)
                row += list(a[j, 1:4].round(3))
            w.writerow(row + [state])
    print('wrote metrics.json, summary.md, timeseries.csv')


if __name__ == '__main__':
    main()
