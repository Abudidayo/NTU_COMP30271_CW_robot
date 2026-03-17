#!/usr/bin/env python3
"""
Landmark CSV Logger
-------------------
Records YOLO-detected landmarks to a CSV file with position and time context.

Subscriptions:
  /detections              (yolo_msgs/DetectionArray)
  /atlas/odom_ground_truth (nav_msgs/Odometry)

CSV columns:
  index, item_scanned, time_elapsed_s, pos_x, pos_y, pos_z, yaw_deg

Run directly:
  python3 landmark_csv_logger.py

Or add to a launch file via ExecuteProcess.
"""

import csv
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry
from yolo_msgs.msg import DetectionArray

# ── CONFIG ────────────────────────────────────────────────────────────────────
LOG_DIR  = "/home/ntu-user/ros2_ws/src/NTU_COMP30271_CW_RobotSim/landmark_logs"
MIN_CONF = 0.5      # minimum YOLO confidence to log
MIN_AREA = 1500     # minimum bounding-box area (px²) — filters tiny/far detections
# ─────────────────────────────────────────────────────────────────────────────


def quat_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert a quaternion to yaw angle in degrees (rotation about Z-axis)."""
    yaw_rad = math.atan2(2.0 * (qw * qz + qx * qy),
                         1.0 - 2.0 * (qy * qy + qz * qz))
    return math.degrees(yaw_rad)


class LandmarkCSVLogger(Node):

    def __init__(self):
        super().__init__('landmark_csv_logger')

        os.makedirs(LOG_DIR, exist_ok=True)
        ts            = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.csv_path = os.path.join(LOG_DIR, f"landmarks_{ts}.csv")

        # Write CSV header
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'index', 'item_scanned', 'time_elapsed_s',
                'pos_x', 'pos_y', 'pos_z', 'yaw_deg'
            ])

        # Robot pose (relative to start, updated from odometry)
        self.rel_x   = 0.0
        self.rel_y   = 0.0
        self.rel_z   = 0.0
        self.yaw_deg = 0.0

        # Starting position — captured from the first odometry message
        self.start_x: float | None = None
        self.start_y: float | None = None
        self.start_z: float | None = None

        # Timing
        self.start_time = time.monotonic()

        # Detection bookkeeping
        self.index   = 1
        self.seen    = set()    # class names already logged — each class logged once only
        self.rows    = []       # in-memory copy for end-of-run summary

        # ── Subscriptions ────────────────────────────────────────────────────
        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(
            Odometry,
            '/atlas/odom_ground_truth',
            self._odom_cb,
            best_effort_qos,
        )

        self.create_subscription(
            DetectionArray,
            '/detections',
            self._detection_cb,
            10,
        )

        self.get_logger().info('Landmark CSV Logger started')
        self.get_logger().info(f'Saving to: {self.csv_path}')
        print(f'\n[LandmarkCSVLogger] Output → {self.csv_path}')
        print(f'  Min confidence : {MIN_CONF}')
        print(f'  Min bbox area  : {MIN_AREA} px²')
        print(f'  Mode           : unique landmarks only (each class logged once)')
        print('  Waiting for detections...  (Ctrl+C to stop)\n')

    # ── ODOMETRY ─────────────────────────────────────────────────────────────

    def _odom_cb(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        # Latch the first reading as the starting position
        if self.start_x is None:
            self.start_x = pos.x
            self.start_y = pos.y
            self.start_z = pos.z
            self.get_logger().info(
                f'Start position latched: '
                f'({self.start_x:.3f}, {self.start_y:.3f}, {self.start_z:.3f})'
            )

        # Coordinates relative to starting position
        self.rel_x   = pos.x - self.start_x
        self.rel_y   = pos.y - self.start_y
        self.rel_z   = pos.z - self.start_z
        self.yaw_deg = quat_to_yaw_deg(ori.x, ori.y, ori.z, ori.w)

    # ── YOLO DETECTIONS ──────────────────────────────────────────────────────

    def _detection_cb(self, msg: DetectionArray) -> None:
        for d in msg.detections:
            # Filter by confidence
            if d.score < MIN_CONF:
                continue

            # Filter by bounding-box area (removes tiny/distant detections)
            area = d.bbox.size.x * d.bbox.size.y
            if area < MIN_AREA:
                continue

            name = d.class_name

            # Skip if this landmark class has already been logged
            if name in self.seen:
                continue

            self.seen.add(name)

            elapsed = round(time.monotonic() - self.start_time, 2)
            row = [
                self.index,
                name,
                elapsed,
                round(self.rel_x, 3),
                round(self.rel_y, 3),
                round(self.rel_z, 3),
                round(self.yaw_deg, 2),
            ]
            self.rows.append(row)

            # Append to CSV immediately so data is not lost on crash
            with open(self.csv_path, 'a', newline='') as f:
                csv.writer(f).writerow(row)

            self.get_logger().info(
                f'[{self.index}] {name:<20}  conf={d.score:.2f}  '
                f't={elapsed:.1f}s  '
                f'pos=({self.rel_x:.2f},{self.rel_y:.2f},{self.rel_z:.2f})  '
                f'yaw={self.yaw_deg:.1f}°'
            )
            print(
                f'  [{self.index:>3}]  {name:<22}  '
                f't={elapsed:>7.1f}s  '
                f'pos=({self.rel_x:>6.2f}, {self.rel_y:>6.2f}, {self.rel_z:>6.2f})  '
                f'yaw={self.yaw_deg:>7.1f}°'
            )

            self.index += 1

    # ── SUMMARY ──────────────────────────────────────────────────────────────

    def print_summary(self) -> None:
        total = time.monotonic() - self.start_time
        print(f'\n{"=" * 66}')
        print(f'  LANDMARK SESSION SUMMARY')
        print(f'  Total elapsed   : {total:.1f} s')
        print(f'  Landmarks logged: {len(self.rows)}')
        if self.rows:
            hdr = f'  {"#":<5}  {"Item":<22}  {"t(s)":<9}  {"X":>7}  {"Y":>7}  {"Z":>7}  {"Yaw°":>8}'
            print(f'\n{hdr}')
            print(f'  {"-" * 63}')
            for r in self.rows:
                print(
                    f'  {r[0]:<5}  {r[1]:<22}  {r[2]:<9}  '
                    f'{r[3]:>7}  {r[4]:>7}  {r[5]:>7}  {r[6]:>8}'
                )
        print(f'\n  CSV saved to: {self.csv_path}')
        print(f'{"=" * 66}\n')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LandmarkCSVLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.print_summary()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
