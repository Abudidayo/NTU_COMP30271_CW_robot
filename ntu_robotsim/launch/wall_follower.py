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
from std_msgs.msg import Bool
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
        self.create_subscription(
            Bool, '/atlas/wall_follower_enabled',
            self.wall_follower_enabled_cb, 10)

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
        self.permanent_stop = False           # stop sign = permanent halt
        self.wall_follower_enabled = True

        # --- speed change timer (for both slow and fast signs) -----------------
        self.speed_change_start_ns = None
        self.speed_change_duration = 10.0     # seconds before reverting to normal

        # --- stop sign confirmation (must see for N seconds before stopping) --
        self.stop_confirm_secs = 5.0          # seconds of continuous sighting
        self.stop_first_seen_ns = None        # when we first started seeing it
        self.stop_last_seen_ns = None         # last time stop sign was detected

        # --- object counting -------------------------------------------------
        self.object_counts = {}
        self.object_cooldown = {}
        self.COUNTABLE = {'orange', 'tree', 'trees', 'car'}

        # --- object centering & pause ----------------------------------------
        self.obj_pause = False              # True = pausing in front of object
        self.obj_pause_start_ns = None
        self.obj_pause_duration = 5.0       # seconds to stay in front
        self.obj_centering = False          # True = steering to center the object
        self.obj_bbox_center_x = None       # bbox center x in pixels
        self.obj_target_name = None         # name of object being centered on
        self.obj_target_score = 0.0         # confidence score
        self.obj_avg_area = 0.0             # average bbox area of group
        self.obj_min_area = 8000.0          # min avg area before pausing (close enough)
        self.image_width = 640.0            # camera image width in pixels
        self.obj_visited = set()            # objects already centered & counted

        # --- stuck detection & recovery --------------------------------------
        self.stuck_start_ns = None        # when we first detected being stuck
        self.stuck_threshold = 1.5        # seconds before triggering recovery
        self.recovering = False           # True = currently backing up
        self.recover_start_ns = None
        self.recover_duration = 2.0       # seconds to back up and turn
        self.last_positions = []          # track recent front_dist to detect no movement

        # --- control loop at 10 Hz ------------------------------------------
        self.create_timer(0.1, self.control_loop)
        # --- save object counts every 10 s ----------------------------------
        self.create_timer(10.0, self.save_counts)
        self.get_logger().info(
            f'Wall follower started  speed={self.linear_speed}  '
            f'wall_d={self.wall_follow_dist}  front_d={self.front_stop_dist}')

    def wall_follower_enabled_cb(self, msg):
        was_enabled = self.wall_follower_enabled
        self.wall_follower_enabled = bool(msg.data)

        if was_enabled == self.wall_follower_enabled:
            return

        state = 'enabled' if self.wall_follower_enabled else 'paused'
        self.get_logger().info(f'Wall follower override: {state}')

        if not self.wall_follower_enabled:
            self.cmd_pub.publish(Twist())

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

        # --- first pass: collect positions & areas per countable type ----------
        frame_info = {}   # {name_lower: {'count': N, 'xs': [...], 'areas': [...]}}
        for d in msg.detections:
            if d.score < 0.5:
                continue
            nl = d.class_name.lower()
            area = d.bbox.size.x * d.bbox.size.y
            if nl in self.COUNTABLE and area > 2000:
                if nl not in frame_info:
                    frame_info[nl] = {'count': 0, 'xs': [], 'areas': []}
                frame_info[nl]['count'] += 1
                frame_info[nl]['xs'].append(d.bbox.center.position.x)
                frame_info[nl]['areas'].append(area)

        for d in msg.detections:
            if d.score < 0.5:
                continue
            name = d.class_name
            area = d.bbox.size.x * d.bbox.size.y

            # --- count objects (orange, tree, car) ---
            if name.lower() in self.COUNTABLE and area > 2000:
                if name.lower() in self.obj_visited:
                    continue
                nl = name.lower()
                info = frame_info.get(nl, {'count': 0})
                last_count = self.object_cooldown.get(name, 0)
                need_pause = info['count'] >= 2

                # Multiple detected → always trigger centering (no cooldown)
                if need_pause and not self.obj_centering and not self.obj_pause:
                    avg_x = sum(info['xs']) / len(info['xs'])
                    avg_area = sum(info['areas']) / len(info['areas'])
                    self.obj_centering = True
                    self.obj_target_name = name
                    self.obj_target_score = d.score
                    self.obj_bbox_center_x = avg_x
                    self.obj_avg_area = avg_area
                    self.get_logger().info(
                        f'OBJECT spotted: {name} x{info["count"]}  '
                        f'avg_area={avg_area:.0f}  avg_x={avg_x:.0f} – '
                        f'centering on group')
                # Single detection → count with cooldown, don't mark visited
                elif not need_pause and (now_ns - last_count) > 15_000_000_000:
                    self.object_counts[name] = self.object_counts.get(name, 0) + 1
                    self.object_cooldown[name] = now_ns
                    self.get_logger().info(
                        f'COUNTED {name} (single)  (total: {self.object_counts[name]})  '
                        f'conf={d.score:.2f}  area={area:.0f}')
                    self.save_counts()
                if self.obj_centering and not self.obj_pause:
                    # Update to average position of all same-type detections
                    info = frame_info.get(nl, None)
                    if info and len(info['xs']) > 0:
                        self.obj_bbox_center_x = sum(info['xs']) / len(info['xs'])
                        self.obj_avg_area = sum(info['areas']) / len(info['areas'])
                # Count during the pause (YOLO keeps detecting while stopped)
                if self.obj_pause:
                    if (now_ns - last_count) > 15_000_000_000:
                        self.object_counts[name] = self.object_counts.get(name, 0) + 1
                        self.object_cooldown[name] = now_ns
                        self.get_logger().info(
                            f'COUNTED {name}  (total: {self.object_counts[name]})  '
                            f'conf={d.score:.2f}  area={area:.0f}')
                        self.save_counts()
                continue

            # --- sign reactions: only when close enough (area >= 6000) ---
            if area < 6000:
                continue

            # cooldown: ignore same sign class within 30s (sim-time)
            last = self.sign_cooldown.get(name, 0)
            if (now_ns - last) < 30_000_000_000:
                continue

            if 'stop' in name.lower():
                # First time seeing stop sign → start the confirmation timer
                self.stop_last_seen_ns = now_ns
                if self.stop_first_seen_ns is None:
                    self.stop_first_seen_ns = now_ns
                    self.get_logger().info(
                        f'STOP sign spotted  conf={d.score:.2f}  area={area:.0f} – '
                        f'waiting {self.stop_confirm_secs}s to confirm')
                # Check if we've been seeing it long enough
                elapsed = (now_ns - self.stop_first_seen_ns) / 1e9
                if elapsed >= self.stop_confirm_secs:
                    self.get_logger().info(
                        f'STOP sign CONFIRMED after {elapsed:.1f}s – '
                        f'PERMANENTLY STOPPED')
                    self.permanent_stop = True
                    self.stop_first_seen_ns = None
                    self.sign_cooldown[name] = now_ns
                else:
                    self.get_logger().info(
                        f'STOP sign seen for {elapsed:.1f}/{self.stop_confirm_secs}s',
                        throttle_duration_sec=1.0)

            elif 'slow' in name.lower():
                self.get_logger().info(
                    f'SLOW sign  conf={d.score:.2f}  area={area:.0f} – '
                    f'reducing speed for {self.speed_change_duration}s')
                self.speed_factor = 0.25
                self.speed_change_start_ns = now_ns
                self.sign_cooldown[name] = now_ns

            elif 'fast' in name.lower():
                self.get_logger().info(
                    f'FAST sign  conf={d.score:.2f}  area={area:.0f} – '
                    f'increasing speed for {self.speed_change_duration}s')
                self.speed_factor = 2.5
                self.speed_change_start_ns = now_ns
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

        if not self.wall_follower_enabled:
            self.cmd_pub.publish(twist)
            return

        # ---- reset stop confirmation if sign disappeared for >2s ------------
        if self.stop_first_seen_ns is not None and self.stop_last_seen_ns is not None:
            gap = (self.get_clock().now().nanoseconds - self.stop_last_seen_ns) / 1e9
            if gap > 2.0:
                self.get_logger().info('Stop sign lost – resetting confirmation timer')
                self.stop_first_seen_ns = None
                self.stop_last_seen_ns = None

        # ---- permanent stop (stop sign confirmed) ----------------------------
        if self.permanent_stop:
            self.cmd_pub.publish(twist)
            return

        # ---- speed change expires after a few seconds ------------------------
        if self.speed_change_start_ns is not None:
            elapsed = (self.get_clock().now().nanoseconds
                       - self.speed_change_start_ns) / 1e9
            if elapsed > self.speed_change_duration:
                self.get_logger().info(
                    f'Speed change expired (was {self.speed_factor}x) – back to normal')
                self.speed_factor = 1.0
                self.speed_change_start_ns = None

        # ---- object pause: stay in front of object for 5s --------------------
        if self.obj_pause:
            elapsed = (self.get_clock().now().nanoseconds
                       - self.obj_pause_start_ns) / 1e9
            if elapsed >= self.obj_pause_duration:
                # Done pausing – mark as visited and resume
                self.get_logger().info(
                    f'Object pause done ({self.obj_target_name}) – resuming')
                if self.obj_target_name:
                    self.obj_visited.add(self.obj_target_name.lower())
                self.obj_pause = False
                self.obj_pause_start_ns = None
                self.obj_centering = False
                self.obj_bbox_center_x = None
                self.obj_target_name = None
            else:
                # Stay still
                self.cmd_pub.publish(twist)
                return

        # ---- object centering: steer to center & approach if too far ---------
        if self.obj_centering and self.obj_bbox_center_x is not None:
            img_center = self.image_width / 2.0
            offset = self.obj_bbox_center_x - img_center  # +ve = object is right
            tolerance = self.image_width * 0.1  # 10% of image width
            centered = abs(offset) < tolerance
            close_enough = self.obj_avg_area >= self.obj_min_area

            if centered and close_enough:
                # Centered and close enough – start the 5s pause
                self.get_logger().info(
                    f'Object centered & close (offset={offset:.0f}px, '
                    f'avg_area={self.obj_avg_area:.0f}) – '
                    f'pausing {self.obj_pause_duration}s for YOLO')
                self.obj_pause = True
                self.obj_pause_start_ns = self.get_clock().now().nanoseconds
                self.cmd_pub.publish(twist)
                return
            else:
                # Turn toward the object group
                if not centered:
                    turn_speed = 0.3 if abs(offset) > tolerance * 2 else 0.15
                    twist.angular.z = -turn_speed if offset > 0 else turn_speed
                # Move forward if too far away
                if not close_enough:
                    twist.linear.x = 0.12
                else:
                    twist.linear.x = 0.03
                self.get_logger().info(
                    f'Centering: offset={offset:.0f}px  avg_area={self.obj_avg_area:.0f}'
                    f'/{self.obj_min_area:.0f}',
                    throttle_duration_sec=1.0)
                self.cmd_pub.publish(twist)
                return

        # ---- wait for first LIDAR scan -------------------------------------
        if not self.lidar_ok:
            self.cmd_pub.publish(twist)
            return

        now_ns = self.get_clock().now().nanoseconds

        # ---- stuck recovery: back up and turn left -------------------------
        if self.recovering:
            elapsed = (now_ns - self.recover_start_ns) / 1e9
            if elapsed < self.recover_duration:
                twist.linear.x = -0.20          # reverse
                twist.angular.z = 0.6           # turn left while reversing
                self.cmd_pub.publish(twist)
                self.get_logger().info(
                    f'RECOVERING: backing up {elapsed:.1f}/{self.recover_duration}s',
                    throttle_duration_sec=0.5)
                return
            else:
                self.get_logger().info('Recovery done – resuming wall following')
                self.recovering = False
                self.recover_start_ns = None
                self.stuck_start_ns = None

        # ---- stuck detection: front & right both blocked -------------------
        is_stuck = (self.front_dist < self.front_stop_dist * 0.8
                    and self.right_dist < self.wall_follow_dist * 0.5)
        # Also stuck if front is very close and left is also close (boxed in)
        is_boxed = (self.front_dist < self.front_stop_dist * 0.6
                    and self.left_dist < self.wall_follow_dist * 0.8)

        if is_stuck or is_boxed:
            if self.stuck_start_ns is None:
                self.stuck_start_ns = now_ns
            elif (now_ns - self.stuck_start_ns) / 1e9 > self.stuck_threshold:
                self.get_logger().warn(
                    f'STUCK detected (front={self.front_dist:.2f} '
                    f'right={self.right_dist:.2f} left={self.left_dist:.2f}) '
                    f'– starting recovery')
                self.recovering = True
                self.recover_start_ns = now_ns
                self.cmd_pub.publish(twist)
                return
        else:
            self.stuck_start_ns = None

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
