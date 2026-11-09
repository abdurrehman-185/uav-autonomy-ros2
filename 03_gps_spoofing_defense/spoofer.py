#!/usr/bin/env python3
"""GNSS spoofer model: a slow 'carry-off' attack on the drone's receiver.

Republishes /navsat as /navsat/received. From ATTACK_START (seconds after the first fix)
it adds a position offset that grows at RATE m/s toward DIRECTION, capped at MAX_OFFSET.
A slow ramp keeps each fix close to the previous one, which is what lets the attack slip
past simple innovation gates.
"""
import math
import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float64MultiArray

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'common'))
from nav_core import enu_to_lla, lla_to_enu  # noqa: E402

ATTACK_START = 25.0
RATE = 0.35
DIRECTION = (0.0, -1.0)   # Fake position pushed south, so the autopilot "corrects" north.
MAX_OFFSET = 25.0


class Spoofer(Node):
    def __init__(self, enabled):
        super().__init__('gnss_spoofer')
        self.enabled = enabled
        self.t0 = None
        self.pub = self.create_publisher(NavSatFix, '/navsat/received', 20)
        self.offset_pub = self.create_publisher(Float64MultiArray, '/spoofer/offset', 20)
        self.create_subscription(NavSatFix, '/navsat', self.on_fix, 20)

    def on_fix(self, m):
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if self.t0 is None:
            self.t0 = t
        mag = 0.0
        if self.enabled and t - self.t0 >= ATTACK_START:
            mag = min(MAX_OFFSET, RATE * (t - self.t0 - ATTACK_START))
        if mag > 0:
            e, n = lla_to_enu(m.latitude, m.longitude)
            m.latitude, m.longitude = enu_to_lla(e + mag * DIRECTION[0], n + mag * DIRECTION[1])
        self.pub.publish(m)
        off = Float64MultiArray()
        off.data = [t, mag * DIRECTION[0], mag * DIRECTION[1]]
        self.offset_pub.publish(off)


def main():
    rclpy.init()
    node = Spoofer(enabled=(len(sys.argv) < 2 or sys.argv[1] != 'off'))
    try:
        rclpy.spin(node)
    except Exception:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
