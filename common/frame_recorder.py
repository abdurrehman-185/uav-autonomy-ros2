#!/usr/bin/env python3
"""Saves camera frames from ROS image topics as JPEG files named by simulation time.

Usage: python3 frame_recorder.py OUT_DIR TOPIC [TOPIC ...]
Each topic is saved under OUT_DIR/<topic name without slashes>/<sim seconds>.jpg
"""
import os
import sys

import cv2
import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image


class FrameRecorder(Node):
    def __init__(self, out_dir, topics):
        super().__init__('frame_recorder')
        self.counts = {}
        for topic in topics:
            name = topic.strip('/').replace('/', '_')
            folder = os.path.join(out_dir, name)
            os.makedirs(folder, exist_ok=True)
            self.counts[name] = 0
            self.create_subscription(Image, topic, lambda m, f=folder, n=name: self.save(m, f, n), 5)

    def save(self, msg, folder, name):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(folder, f'{t:010.3f}.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        self.counts[name] += 1


def main():
    rclpy.init()
    node = FrameRecorder(sys.argv[1], sys.argv[2:])
    print(f'recording {sys.argv[2:]}', flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    print(f'frames saved: {node.counts}', flush=True)
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
