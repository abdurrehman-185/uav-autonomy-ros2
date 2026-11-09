#!/usr/bin/env python3
"""Writes worlds/outdoor_flight.sdf: open field, MEMS-grade IMU and a GNSS receiver."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
import make_world as mw  # noqa: E402

# Sensor noise. The replay uses these same numbers as the estimator noise model.
IMU_RATE = 200
ACCEL_STD, ACCEL_BIAS = 0.05, 0.04
GYRO_STD, GYRO_BIAS = 0.002, 0.0008
GPS_RATE, GPS_STD_H, GPS_STD_V = 5, 1.5, 1.5



def main():
    sensors = (mw.imu_sensor(IMU_RATE, ACCEL_STD, ACCEL_BIAS, GYRO_STD, GYRO_BIAS)
               + mw.gps_sensor(GPS_RATE, GPS_STD_H, GPS_STD_V)
               + mw.camera('chase_cam', '-4 0 2.2 0 0.4 0', 1.3, 300))

    scenery = ''
    houses = [(-26, 16, 8, 6, 7), (-8, 18, 10, 7, 9), (12, 17, 7, 7, 6), (28, 15, 9, 6, 11),
              (-27, -16, 9, 7, 8), (-6, -18, 8, 6, 6), (14, -17, 10, 8, 10), (29, -15, 7, 7, 7)]
    for i, (x, y, sx, sy, sz) in enumerate(houses):
        shade = 0.55 + 0.05 * (i % 4)
        scenery += mw.box(f'house{i}', x, y, sx, sy, sz, 0.1 * (i % 3), f'{shade} {shade - 0.1} {shade - 0.15} 1')
    for i, (x, y) in enumerate([(-33, 0), (33, 2), (-20, 11), (20, -11), (0, 13), (0, -13), (-12, 0.5), (12, -0.5)]):
        scenery += mw.tree(f'tree{i}', x, y, 6 + i % 3)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds', 'outdoor_flight.sdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='\n') as f:
        f.write(mw.world('outdoor_flight', mw.drone(sensors), scenery, '0 0 42 0 1.5708 1.5708', 1.2))
    print('wrote', out)


if __name__ == '__main__':
    main()
