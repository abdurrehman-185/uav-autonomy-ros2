#!/usr/bin/env python3
"""Adds a chase camera to the x3 drone and a fixed overhead camera to a world file.

Usage: python3 add_cameras.py WORLD.sdf OVERHEAD_X OVERHEAD_Y OVERHEAD_HEIGHT
"""
import sys

path, ox, oy, oz = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = open(path, encoding='utf-8').read()
if 'chase_cam' in s:
    print('cameras already present:', path)
    sys.exit(0)

chase = '''        <sensor name="chase_cam" type="camera">
          <pose>-1.3 0 1.1 0 0.55 0</pose>
          <always_on>1</always_on>
          <update_rate>15</update_rate>
          <topic>chase_cam</topic>
          <camera>
            <horizontal_fov>1.3</horizontal_fov>
            <image><width>640</width><height>360</height><format>R8G8B8</format></image>
            <clip><near>0.1</near><far>120</far></clip>
          </camera>
        </sensor>
      </link>

      <link name="rotor_0">'''
anchor = '</link>\n\n      <link name="rotor_0">'
assert s.count(anchor) == 1, 'drone base_link anchor not found'
s = s.replace(anchor, chase.lstrip(' ').replace('<sensor', '<sensor', 1), 1)

overhead = f'''
    <model name="overhead_cam">
      <static>true</static>
      <pose>{ox} {oy} {oz} 0 1.5708 1.5708</pose>
      <link name="link">
        <sensor name="overhead_cam" type="camera">
          <always_on>1</always_on>
          <update_rate>15</update_rate>
          <topic>overhead_cam</topic>
          <camera>
            <horizontal_fov>1.15</horizontal_fov>
            <image><width>640</width><height>360</height><format>R8G8B8</format></image>
            <clip><near>0.5</near><far>300</far></clip>
          </camera>
        </sensor>
      </link>
    </model>

    <model name="x3">'''
anchor = '\n    <model name="x3">'
assert s.count(anchor) == 1, 'x3 model anchor not found'
s = s.replace(anchor, overhead, 1)
open(path, 'w', encoding='utf-8', newline='\n').write(s)
print('cameras added:', path)
