#!/usr/bin/env python3
"""
Right-hand wall following node with YOLO sign detection for maze navigation.

Behavior:
  - Follows the right wall to systematically traverse the entire maze
  - Reacts to YOLO-detected signs:
      Stop_Sign  -> stop for a configurable duration, then resume
      Slow_Sign  -> reduce speed
      Fast_Sign  -> increase speed
  - Counts sightings of Orange, Tree, Car objects and saves to file
"""

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import Twist
import numpy as np

try:
    from yolo_msgs.msg import DetectionArray
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False

COUNTS_FILE = os.path.expanduser('~/ros2_ws/object_counts.txt')


def pointcloud2_to_xy_distances(msg):
    """Extract 2D (angle, distance) arrays from a PointCloud2 message."""
    from sensor_msgs_py import point_cloud2 as pc2
    xs, ys = [], []
    for p in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
        x, y = float(p[0]), float(p[1])
        d = np.hypot(x, y)
        if 0.05 < d < 8.0:
            xs.append(x)
            ys.append(y)
    if not xs:
        return None, None
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    return np.arctan2(ys, xs), np.hypot(xs, ys)


class WallFollower(Node):
    def __init__(self):
        super().__init__('wall_follower')

        # Tuneable parameters
        self.declare_parameter('linear_speed', 0.20)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('wall_follow_dist', 0.6)
        self.declare_parameter('front_stop_dist', 0.5)
        self.declare_parameter('stop_duration', 3.0)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.wall_follow_dist = self.get_parameter('wall_follow_dist').value
        self.front_stop_dist = self.get_parameter('front_stop_dist').value
        self.stop_duration = self.get_parameter('stop_duration').value

        # --- subscriptions --------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST, depth=5)

        self.create_subscription(
            PointCloud2, '/atlas/velodyne_points',
            self.lidar_cb, sensor_qos)

        if HAS_YOLO:
            self.create_subscription(
                DetectionArray, '/detections',
                self.detection_cb, 10)
            self.get_logger().info('YOLO detection subscription active')
        else:
            self.get_logger().warn('yolo_msgs not found – sign detection disabled')

        # --- publisher -------------------------------------------------------
        self.cmd_pub = self.create_publisher(Twist, '/atlas/cmd_vel', 10)

        # --- state -----------------------------------------------------------
        self.front_dist = float('inf')
        self.right_dist = float('inf')
        self.left_dist = float('inf')
        self.lidar_ok = False

        self.stopped = False
        self.stop_start_ns = None
        self.speed_factor = 1.0
        self.sign_cooldown = {}

        # --- object counting -------------------------------------------------
        self.object_counts = {}
        self.object_cooldown = {}
        self.COUNTABLE = {'orange', 'tree', 'trees', 'car'}

        # --- control loop at 10 Hz ------------------------------------------
        self.create_timer(0.1, self.control_loop)
        # --- save object counts every 10 s ----------------------------------
        self.create_timer(10.0, self.save_counts)
        self.get_logger().info(
            f'Wall follower started  speed={self.linear_speed}  '
            f'wall_d={self.wall_follow_dist}  front_d={self.front_stop_dist}')

    # ── LIDAR ───────────────────────────────────────────────────────────────
    def lidar_cb(self, msg):
        angles, dists = pointcloud2_to_xy_distances(msg)
        if angles is None:
            self.get_logger().warn(
                'LIDAR cb: no valid points parsed', throttle_duration_sec=3.0)
            return
        self.lidar_ok = True

        def sector_min(lo_deg, hi_deg, max_r=3.0):
            lo, hi = np.radians(lo_deg), np.radians(hi_deg)
            m = (angles >= lo) & (angles <= hi) & (dists < max_r)
            return float(np.min(dists[m])) if np.any(m) else float('inf')

        self.front_dist = sector_min(-30, 30)
        self.right_dist = sector_min(-120, -60)
        self.left_dist  = sector_min(60, 120)
        self.get_logger().info(
            f'LIDAR pts={len(dists)}  '
            f'front={self.front_dist:.2f}  '
            f'right={self.right_dist:.2f}  '
            f'left={self.left_dist:.2f}',
            throttle_duration_sec=2.0)

    # ── YOLO DETECTIONS ─────────────────────────────────────────────────────
    def detection_cb(self, msg):
        now_ns = self.get_clock().now().nanoseconds
        for d in msg.detections:
            if d.score < 0.5:
                continue
            name = d.class_name
            area = d.bbox.size.x * d.bbox.size.y

            # --- count objects (orange, tree, car) ---
            if name.lower() in self.COUNTABLE and area > 2000:
                last_count = self.object_cooldown.get(name, 0)
                if (now_ns - last_count) > 15_000_000_000:
                    self.object_counts[name] = self.object_counts.get(name, 0) + 1
                    self.object_cooldown[name] = now_ns
                    self.get_logger().info(
                        f'COUNTED {name}  (total: {self.object_counts[name]})  '
                        f'conf={d.score:.2f}  area={area:.0f}')
                    self.save_counts()
                continue

            # --- sign reactions: only when close enough (area >= 6000) ---
            if area < 4000:
                continue

            # cooldown: ignore same sign class within 30s (sim-time)
            last = self.sign_cooldown.get(name, 0)
            if (now_ns - last) < 30_000_000_000:
                continue

            if 'stop' in name.lower():
                self.get_logger().info(
                    f'STOP sign  conf={d.score:.2f}  area={area:.0f} – '
                    f'stopping {self.stop_duration}s')
                self.stopped = True
                self.stop_start_ns = now_ns
                self.sign_cooldown[name] = now_ns

            elif 'slow' in name.lower():
                self.get_logger().info(
                    f'SLOW sign  conf={d.score:.2f}  area={area:.0f} – '
                    f'reducing speed')
                self.speed_factor = 0.5
                self.sign_cooldown[name] = now_ns

            elif 'fast' in name.lower():
                self.get_logger().info(
                    f'FAST sign  conf={d.score:.2f}  area={area:.0f} – '
                    f'increasing speed')
                self.speed_factor = 1.5
                self.sign_cooldown[name] = now_ns

    # ── SAVE OBJECT COUNTS TO FILE ─────────────────────────────────────────
    def save_counts(self):
        if self.object_counts:
            try:
                with open(COUNTS_FILE, 'w') as f:
                    f.write('=== YOLO Object Detection Counts ===\n\n')
                    for k in sorted(self.object_counts.keys()):
                        f.write(f'{k}: {self.object_counts[k]}\n')
                    total = sum(self.object_counts.values())
                    f.write(f'\nTotal objects detected: {total}\n')
            except Exception:
                pass
            summary = '  '.join(
                f'{k}: {v}' for k, v in sorted(self.object_counts.items()))
            self.get_logger().info(
                f'=== OBJECT COUNTS: {summary} ===', throttle_duration_sec=10.0)

    # ── CONTROL LOOP ────────────────────────────────────────────────────────
    def control_loop(self):
        twist = Twist()

        # ---- handle stop state ---------------------------------------------
        if self.stopped:
            if self.stop_start_ns is not None:
                elapsed = (self.get_clock().now().nanoseconds
                           - self.stop_start_ns) / 1e9
                if elapsed > self.stop_duration:
                    self.stopped = False
                    self.stop_start_ns = None
                    self.speed_factor = 1.0
                    self.get_logger().info('Resuming after stop')
            self.cmd_pub.publish(twist)
            return

        # ---- wait for first LIDAR scan -------------------------------------
        if not self.lidar_ok:
            self.cmd_pub.publish(twist)
            return

        speed = self.linear_speed * self.speed_factor
        turn = self.angular_speed
        wd = self.wall_follow_dist

        # ---- right-hand wall following -------------------------------------
        if self.front_dist < self.front_stop_dist:
            # blocked ahead → turn left in place
            twist.linear.x = 0.0
            twist.angular.z = turn

        elif self.front_dist < self.front_stop_dist * 2.0:
            # approaching front wall → slow down and start turning left
            twist.linear.x = speed * 0.3
            twist.angular.z = turn * 0.5

        elif self.right_dist > wd * 2.0:
            # no wall on right → turn right to find it
            twist.linear.x = speed * 0.5
            twist.angular.z = -turn * 0.7

        elif self.right_dist < wd * 0.4:
            # too close to right wall → veer left
            twist.linear.x = speed * 0.5
            twist.angular.z = turn * 0.5

        else:
            # cruising along right wall
            twist.linear.x = speed
            # proportional correction to maintain wall distance
            error = self.right_dist - wd
            twist.angular.z = -error * 0.8

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save and print final counts
        node.save_counts()
        if node.object_counts:
            summary = '  '.join(
                f'{k}: {v}' for k, v in sorted(node.object_counts.items()))
            node.get_logger().info(f'=== FINAL COUNTS: {summary} ===')
            print(f'\n=== FINAL OBJECT COUNTS ===')
            for k in sorted(node.object_counts.keys()):
                print(f'  {k}: {node.object_counts[k]}')
            print(f'Saved to {COUNTS_FILE}')
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
