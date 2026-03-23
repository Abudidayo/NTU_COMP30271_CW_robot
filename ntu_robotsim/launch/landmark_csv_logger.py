#!/usr/bin/env python3
"""
Landmark database and interactive navigator.

This node:
  - records unique YOLO landmarks to both CSV and TXT files
  - prints indexed landmark entries to the terminal as they are discovered
  - lets the user enter a landmark index to send a Nav2 goal to that pose
  - starts explore_lite after the selected landmark has been reached
  - pauses explore_lite again if the user selects another landmark later
"""

import csv
import math
import os
import signal
import subprocess
import threading
import time
from datetime import datetime

import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Bool
from yolo_msgs.msg import DetectionArray


LOG_DIR = "/home/ntu-user/ros2_ws/src/NTU_COMP30271_CW_RobotSim/landmark_logs"
MIN_CONF = 0.5
MIN_AREA = 1500


def quat_to_yaw_deg(qx: float, qy: float, qz: float, qw: float) -> float:
    """Convert quaternion to yaw angle in degrees."""
    yaw_rad = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    return math.degrees(yaw_rad)


def yaw_deg_to_quat(yaw_deg: float) -> tuple[float, float, float, float]:
    """Return a Z-only quaternion from yaw in degrees."""
    yaw_rad = math.radians(yaw_deg)
    half = yaw_rad * 0.5
    return 0.0, 0.0, math.sin(half), math.cos(half)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def wrap_angle(angle_rad: float) -> float:
    while angle_rad > math.pi:
        angle_rad -= 2.0 * math.pi
    while angle_rad < -math.pi:
        angle_rad += 2.0 * math.pi
    return angle_rad


def pointcloud2_to_xy_distances(msg: PointCloud2) -> tuple[list[float], list[float]]:
    from sensor_msgs_py import point_cloud2 as pc2

    angles = []
    dists = []
    for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
        x, y = float(point[0]), float(point[1])
        distance = math.hypot(x, y)
        if 0.05 < distance < 8.0:
            angles.append(math.atan2(y, x))
            dists.append(distance)
    return angles, dists


class LandmarkCSVLogger(Node):
    def __init__(self):
        super().__init__("landmark_csv_logger")

        os.makedirs(LOG_DIR, exist_ok=True)
        ts = datetime.now().strftime("%Y-%m-%d-%H-%M")
        self.csv_path = os.path.join(LOG_DIR, f"landmarks_{ts}.csv")

        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "index",
                "item_scanned",
                "time_elapsed_s",
                "pos_x",
                "pos_y",
                "pos_z",
                "yaw_deg",
            ])

        self.abs_x = 0.0
        self.abs_y = 0.0
        self.abs_z = 0.0
        self.rel_x = 0.0
        self.rel_y = 0.0
        self.rel_z = 0.0
        self.yaw_deg = 0.0
        self.pose_source = "none"
        self.last_vo_time = 0.0
        self.front_dist = float("inf")
        self.left_dist = float("inf")
        self.right_dist = float("inf")
        self.lidar_ok = False

        self.start_x: float | None = None
        self.start_y: float | None = None
        self.start_z: float | None = None

        self.start_time = time.monotonic()
        self.index = 1
        self.seen = set()
        self.rows = []
        self.landmarks = {}
        self.lock = threading.Lock()

        self.explore_process = None
        self.navigation_in_progress = False
        self.selected_landmark = None
        self.input_thread_started = False

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.create_subscription(
            Odometry,
            "/atlas/odom_ground_truth",
            self._odom_ground_truth_cb,
            best_effort_qos,
        )
        self.create_subscription(
            Odometry,
            "/atlas/vo",
            self._odom_visual_cb,
            best_effort_qos,
        )
        self.create_subscription(
            DetectionArray,
            "/detections",
            self._detection_cb,
            10,
        )
        self.create_subscription(
            PointCloud2,
            "/atlas/velodyne_points",
            self._lidar_cb,
            best_effort_qos,
        )

        self.cancel_nav_client = self.create_client(
            CancelGoal,
            "/navigate_to_pose/_action/cancel_goal",
        )
        self.wall_follower_enabled_pub = self.create_publisher(
            Bool,
            "/atlas/wall_follower_enabled",
            10,
        )
        self.cmd_pub = self.create_publisher(
            Twist,
            "/atlas/cmd_vel",
            10,
        )
        self.create_timer(1.0, self._maybe_start_input_thread)

        self.get_logger().info("Landmark database started")
        self.get_logger().info(f"CSV file: {self.csv_path}")
        print(f"\n[LandmarkDatabase] CSV  -> {self.csv_path}")
        print(f"  Min confidence : {MIN_CONF}")
        print(f"  Min bbox area  : {MIN_AREA} px^2")
        print("  Pose source    : /atlas/vo when available, else /atlas/odom_ground_truth")
        print("  Unique landmark classes are indexed once each.")
        print("  Type a landmark number to navigate there.")
        print("  Type 'l' to list, 'e' to start explore manually, 'q' to quit.\n")

    def _maybe_start_input_thread(self) -> None:
        if self.input_thread_started:
            return
        self.input_thread_started = True
        threading.Thread(target=self._input_loop, daemon=True).start()

    def _update_pose(self, msg: Odometry, source: str) -> None:
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation

        if self.start_x is None:
            self.start_x = pos.x
            self.start_y = pos.y
            self.start_z = pos.z
            self.get_logger().info(
                f"Start position latched: ({self.start_x:.3f}, "
                f"{self.start_y:.3f}, {self.start_z:.3f})"
            )

        self.abs_x = pos.x
        self.abs_y = pos.y
        self.abs_z = pos.z
        self.rel_x = pos.x - self.start_x
        self.rel_y = pos.y - self.start_y
        self.rel_z = pos.z - self.start_z
        self.yaw_deg = quat_to_yaw_deg(ori.x, ori.y, ori.z, ori.w)
        self.pose_source = source

    def _odom_ground_truth_cb(self, msg: Odometry) -> None:
        if (time.monotonic() - self.last_vo_time) < 1.0:
            return
        self._update_pose(msg, "ground_truth")

    def _odom_visual_cb(self, msg: Odometry) -> None:
        self.last_vo_time = time.monotonic()
        self._update_pose(msg, "visual_odometry")

    def _detection_cb(self, msg: DetectionArray) -> None:
        for d in msg.detections:
            if d.score < MIN_CONF:
                continue

            area = d.bbox.size.x * d.bbox.size.y
            if area < MIN_AREA:
                continue

            name = d.class_name
            if name in self.seen:
                continue

            self.seen.add(name)
            elapsed = round(time.monotonic() - self.start_time, 2)
            row = [
                self.index,
                name,
                elapsed,
                round(self.abs_x, 3),
                round(self.abs_y, 3),
                round(self.abs_z, 3),
                round(self.yaw_deg, 2),
            ]

            with self.lock:
                self.rows.append(row)
                self.landmarks[self.index] = {
                    "name": name,
                    "elapsed": elapsed,
                    "x": self.abs_x,
                    "y": self.abs_y,
                    "z": self.abs_z,
                    "yaw_deg": self.yaw_deg,
                    "rel_x": self.rel_x,
                    "rel_y": self.rel_y,
                    "rel_z": self.rel_z,
                }

            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)

            self.get_logger().info(
                f"[{self.index}] {name:<20} conf={d.score:.2f} "
                f"t={elapsed:.1f}s pos=({self.abs_x:.2f},{self.abs_y:.2f},{self.abs_z:.2f}) "
                f"yaw={self.yaw_deg:.1f}deg"
            )
            print(
                f"  [{self.index}] {name:<22} "
                f"map=({self.abs_x:>6.2f}, {self.abs_y:>6.2f}, {self.abs_z:>6.2f}) "
                f"yaw={self.yaw_deg:>7.1f}deg"
            )

            self.index += 1

    def _lidar_cb(self, msg: PointCloud2) -> None:
        angles, dists = pointcloud2_to_xy_distances(msg)
        if not angles:
            return

        self.lidar_ok = True
        front_candidates = [
            dist for angle, dist in zip(angles, dists)
            if -math.radians(25.0) <= angle <= math.radians(25.0)
        ]
        left_candidates = [
            dist for angle, dist in zip(angles, dists)
            if math.radians(35.0) <= angle <= math.radians(100.0)
        ]
        right_candidates = [
            dist for angle, dist in zip(angles, dists)
            if -math.radians(100.0) <= angle <= -math.radians(35.0)
        ]
        self.front_dist = min(front_candidates) if front_candidates else float("inf")
        self.left_dist = min(left_candidates) if left_candidates else float("inf")
        self.right_dist = min(right_candidates) if right_candidates else float("inf")

    def _print_landmarks(self) -> None:
        with self.lock:
            if not self.landmarks:
                print("  No landmarks stored yet.")
                return

            print("\n  Stored landmarks:")
            for idx in sorted(self.landmarks):
                entry = self.landmarks[idx]
                print(
                    f"    {idx}. {entry['name']}  "
                    f"map=({entry['x']:.2f}, {entry['y']:.2f}, {entry['z']:.2f})  "
                    f"yaw={entry['yaw_deg']:.1f}deg"
                )
            print("")

    def _input_loop(self) -> None:
        while rclpy.ok():
            try:
                raw = input("Select landmark index (`l`, `e`, `q`): ").strip().lower()
            except EOFError:
                return
            except KeyboardInterrupt:
                return

            if not raw:
                continue

            if raw == "q":
                print("  Closing landmark selector.")
                rclpy.shutdown()
                return

            if raw == "l":
                self._print_landmarks()
                continue

            if raw == "e":
                self._start_explore()
                continue

            if not raw.isdigit():
                print("  Enter a number, `l`, `e`, or `q`.")
                continue

            self._navigate_to_landmark(int(raw))

    def _navigate_to_landmark(self, index: int) -> None:
        with self.lock:
            landmark = self.landmarks.get(index)

        if landmark is None:
            print(f"  Landmark {index} is not available yet.")
            return

        if self.pose_source == "none":
            print("  Waiting for odometry before driving to a landmark.")
            return

        if self.navigation_in_progress:
            print("  Navigation is already in progress. Wait for it to finish first.")
            return

        self._set_wall_follower_enabled(False)
        self._pause_explore_navigation()
        self._stop_explore()

        self.selected_landmark = index
        self.navigation_in_progress = True

        print(
            f"  Driving to landmark {index}: {landmark['name']} "
            f"at ({landmark['x']:.2f}, {landmark['y']:.2f}, {landmark['z']:.2f}), "
            f"yaw {landmark['yaw_deg']:.1f}deg using {self.pose_source}"
        )

        success = self._drive_to_landmark_pose(
            landmark["x"],
            landmark["y"],
            landmark["yaw_deg"],
        )
        self.navigation_in_progress = False

        self._set_wall_follower_enabled(True)
        if success:
            print(f"  Reached landmark {self.selected_landmark}. Starting explore.")
            self._start_explore()
        else:
            print("  Landmark drive did not complete cleanly. Explore not started.")

    def _pause_explore_navigation(self) -> None:
        print("  Cancelling any active Nav2 goal before manual landmark navigation.")

        if not self.cancel_nav_client.wait_for_service(timeout_sec=2.0):
            print("  Nav2 cancel service is not ready yet, continuing anyway.")
            return

        request = CancelGoal.Request()
        future = self.cancel_nav_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

        if not future.done():
            print("  Timed out while cancelling the active Nav2 goal.")
            return

        try:
            response = future.result()
        except Exception as exc:
            print(f"  Failed to cancel active Nav2 goal: {exc}")
            return

        if response is None:
            print("  Cancel request returned no response.")
            return

        if response.return_code == CancelGoal.Response.ERROR_NONE:
            print("  Active Nav2 goal cancelled.")
        elif response.return_code == CancelGoal.Response.ERROR_GOAL_TERMINATED:
            print("  No active Nav2 goal needed cancelling.")
        else:
            print(
                f"  Cancel request returned code {response.return_code}; "
                "trying manual goal anyway."
            )

        time.sleep(0.5)

    def _set_wall_follower_enabled(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = enabled

        for _ in range(3):
            self.wall_follower_enabled_pub.publish(msg)
            time.sleep(0.1)

        state = "enabled" if enabled else "paused"
        print(f"  Wall follower {state}.")

    def _drive_to_landmark_pose(
        self,
        target_x: float,
        target_y: float,
        target_yaw_deg: float,
        timeout_sec: float = 60.0,
    ) -> bool:
        print("  Manual landmark controller active.")
        start_time = time.monotonic()
        last_status = 0.0
        twist = Twist()

        while rclpy.ok():
            dx = target_x - self.abs_x
            dy = target_y - self.abs_y
            distance = math.hypot(dx, dy)
            current_yaw = math.radians(self.yaw_deg)
            heading_to_goal = math.atan2(dy, dx)
            heading_error = wrap_angle(heading_to_goal - current_yaw)
            final_yaw_error = wrap_angle(math.radians(target_yaw_deg) - current_yaw)

            if distance < 0.18 and abs(final_yaw_error) < math.radians(6.0):
                self.cmd_pub.publish(Twist())
                return True

            if time.monotonic() - start_time > timeout_sec:
                self.cmd_pub.publish(Twist())
                print("  Timed out while driving to the selected landmark.")
                return False

            twist = Twist()
            obstacle_close = self.lidar_ok and self.front_dist < 0.45
            obstacle_near = self.lidar_ok and self.front_dist < 0.75
            turn_left_preferred = self.left_dist >= self.right_dist
            avoidance_turn = 0.65 if turn_left_preferred else -0.65

            if obstacle_close:
                twist.linear.x = 0.05
                twist.angular.z = avoidance_turn
            elif distance >= 0.18:
                if abs(heading_error) > math.radians(20.0):
                    twist.angular.z = clamp(1.8 * heading_error, -0.8, 0.8)
                else:
                    base_speed = clamp(0.5 * distance, 0.06, 0.22)
                    if obstacle_near:
                        base_speed = min(base_speed, 0.08)
                        twist.angular.z = avoidance_turn * 0.8
                    twist.linear.x = base_speed
                    if not obstacle_near:
                        twist.angular.z = clamp(1.4 * heading_error, -0.6, 0.6)
            else:
                twist.angular.z = clamp(1.6 * final_yaw_error, -0.6, 0.6)

            now = time.monotonic()
            if now - last_status > 1.0:
                print(
                    f"  Driving... dist={distance:.2f}m "
                    f"heading_err={math.degrees(heading_error):.1f}deg "
                    f"final_yaw_err={math.degrees(final_yaw_error):.1f}deg "
                    f"front={self.front_dist:.2f}m "
                    f"left={self.left_dist:.2f}m "
                    f"right={self.right_dist:.2f}m "
                    f"pose_source={self.pose_source}"
                )
                last_status = now

            self.cmd_pub.publish(twist)
            time.sleep(0.1)

    def _explore_command(self) -> list[str]:
        config_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "config", "explore_params.yaml")
        )
        return [
            "ros2",
            "run",
            "explore_lite",
            "explore",
            "--ros-args",
            "--params-file",
            config_path,
            "-p",
            "use_sim_time:=true",
            "-r",
            "__node:=explore_node",
        ]

    def _start_explore(self) -> None:
        if self.explore_process is not None and self.explore_process.poll() is None:
            print("  Explore is already running.")
            return

        try:
            self.explore_process = subprocess.Popen(
                self._explore_command(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            print("  Explore started.")
        except Exception as exc:
            print(f"  Failed to start explore: {exc}")

    def _stop_explore(self) -> None:
        if self.explore_process is None or self.explore_process.poll() is not None:
            self.explore_process = None
            return

        print("  Stopping explore so the selected landmark can override it.")
        try:
            os.killpg(os.getpgid(self.explore_process.pid), signal.SIGTERM)
            self.explore_process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.explore_process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            self.explore_process = None

    def print_summary(self) -> None:
        total = time.monotonic() - self.start_time
        print(f"\n{'=' * 66}")
        print("  LANDMARK SESSION SUMMARY")
        print(f"  Total elapsed   : {total:.1f} s")
        print(f"  Landmarks logged: {len(self.rows)}")
        if self.rows:
            print("\n  #      Item                    t(s)      X       Y       Z     Yaw")
            print(f"  {'-' * 63}")
            for row in self.rows:
                print(
                    f"  {row[0]:<6} {row[1]:<22} {row[2]:<8} "
                    f"{row[3]:>7} {row[4]:>7} {row[5]:>7} {row[6]:>7}"
                )
        print(f"\n  CSV saved to: {self.csv_path}")
        print(f"{'=' * 66}\n")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LandmarkCSVLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._set_wall_follower_enabled(True)
        node._stop_explore()
        node.print_summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
