#!/usr/bin/env python3
"""
JetBot Motor Driver Node
------------------------
Subscribes to /cmd_vel (geometry_msgs/Twist) and drives the Waveshare JetBot
motors using the NVIDIA jetbot Python library (Adafruit MotorHAT-based).

The jetbot library uses motor values in the range [-1.0, 1.0]:
  +1.0  = full forward
   0.0  = stop
  -1.0  = full reverse

Differential drive conversion:
  left_speed  = linear.x - angular.z * track_width_factor
  right_speed = linear.x + angular.z * track_width_factor

Run standalone:
  python3 jetbot_motor_driver.py

Or via launch:
  ros2 run ntu_robotsim jetbot_motor_driver
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    from jetbot import Robot
    HAS_JETBOT = True
except ImportError:
    HAS_JETBOT = False


class JetBotMotorDriver(Node):

    def __init__(self):
        super().__init__('jetbot_motor_driver')

        # ── Parameters ──────────────────────────────────────────────────────
        # max_speed: maps the maximum ROS velocity (m/s) to motor value 1.0
        # track_factor: how much angular.z affects differential (tune on robot)
        self.declare_parameter('max_speed', 0.25)       # m/s → maps to motor 1.0
        self.declare_parameter('track_factor', 0.5)     # angular scale factor

        self.max_speed    = self.get_parameter('max_speed').value
        self.track_factor = self.get_parameter('track_factor').value

        # ── Motor initialisation ─────────────────────────────────────────────
        if HAS_JETBOT:
            self.robot = Robot()
            self.get_logger().info('jetbot.Robot initialised — motor driver ready')
        else:
            self.robot = None
            self.get_logger().warn(
                'jetbot library not found. Running in DRY-RUN mode '
                '(motor commands will be printed but NOT sent to hardware).'
            )

        # ── Subscription ─────────────────────────────────────────────────────
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)
        self.get_logger().info('Subscribed to /cmd_vel')

    # ── Callback ─────────────────────────────────────────────────────────────

    def _cmd_vel_cb(self, msg: Twist) -> None:
        linear  = msg.linear.x
        angular = msg.angular.z

        # Normalise linear to [-1, 1] based on max_speed
        lin_norm = linear / self.max_speed

        # Differential drive
        left  = lin_norm - angular * self.track_factor
        right = lin_norm + angular * self.track_factor

        # Clamp to [-1, 1]
        left  = max(-1.0, min(1.0, left))
        right = max(-1.0, min(1.0, right))

        self._drive(left, right)

    def _drive(self, left: float, right: float) -> None:
        if self.robot is not None:
            self.robot.set_motors(left, right)
        else:
            self.get_logger().info(
                f'DRY-RUN  left={left:+.3f}  right={right:+.3f}',
                throttle_duration_sec=1.0,
            )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        if self.robot is not None:
            self.robot.stop()
            self.get_logger().info('Motors stopped')
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = JetBotMotorDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
