#!/usr/bin/env python3
"""Writes worlds/street.sdf: a street lined with buildings and poles, for a delivery flight."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
import make_world as mw  # noqa: E402

# (x_center, length along x, depth, height) for buildings; north side faces y = +NORTH_FACE.
NORTH_FACE, SOUTH_FACE = 7.0, -7.0
NORTH = [(-4, 10, 10, 9), (10, 12, 12, 12), (25, 9, 9, 8), (39, 14, 11, 14), (55, 11, 10, 10), (68, 8, 9, 9)]
SOUTH = [(0, 12, 10, 10), (15, 8, 9, 8), (28, 13, 12, 11), (43, 9, 9, 9), (57, 12, 11, 13)]
POLES = [(x, 4.6 if i % 2 else -4.6) for i, x in enumerate(range(3, 66, 9))]


def buildings():
    """Footprints as (x0, y0, width, height) rectangles, for plots and collision checks."""
    rects = [(x - L / 2, NORTH_FACE, L, d) for x, L, d, _ in NORTH]
    rects += [(x - L / 2, SOUTH_FACE - d, L, d) for x, L, d, _ in SOUTH]
    return rects


def main():
    sensors = (mw.imu_sensor(200, 0.05, 0.04, 0.002, 0.0008)
               + mw.gps_sensor(5, 1.2, 2.0)
               + mw.lidar_sensor(30.0)
               + mw.range_sensor()
               + mw.camera('chase_cam', '-4.5 0 2.2 0 0.32 0', 1.35, 300))
    scenery = ''
    for i, (x, L, d, h) in enumerate(NORTH):
        g = 0.5 + 0.06 * (i % 4)
        scenery += mw.box(f'n{i}', x, NORTH_FACE + d / 2, L, d, h, 0, f'{g + 0.1} {g} {g - 0.08} 1')
    for i, (x, L, d, h) in enumerate(SOUTH):
        g = 0.5 + 0.06 * (i % 4)
        scenery += mw.box(f's{i}', x, SOUTH_FACE - d / 2, L, d, h, 0, f'{g - 0.05} {g} {g + 0.08} 1')
    for i, (x, y) in enumerate(POLES):
        scenery += (f'<collision name="pole{i}_c"><pose>{x} {y} 3 0 0 0</pose><geometry><cylinder><radius>0.18</radius><length>6</length></cylinder></geometry></collision>'
                    f'<visual name="pole{i}_v"><pose>{x} {y} 3 0 0 0</pose><geometry><cylinder><radius>0.18</radius><length>6</length></cylinder></geometry>'
                    '<material><ambient>0.25 0.25 0.28 1</ambient><diffuse>0.25 0.25 0.28 1</diffuse></material></visual>')
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds', 'street.sdf')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w', newline='\n') as f:
        f.write(mw.world('street', mw.drone(sensors), scenery, '30 0 62 0 1.5708 1.5708', 1.25, '0.45 0.47 0.45 1'))
    print('wrote', out)


if __name__ == '__main__':
    main()
