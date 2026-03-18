#!/usr/bin/env python3
"""
JetBot Ultrasonic Distance Driver
-----------------------------------
Reads one or two HC-SR04 ultrasonic sensors via Jetson Nano GPIO and
publishes distances as std_msgs/Float32 topics.

  /distance/front  — distance to obstacle directly ahead (metres)
  /distance/right  — distance to right wall (metres)   [optional]

Wiring (HC-SR04 → Jetson Nano J41 header):
  HC-SR04 VCC  → Pin 2  (5 V)
  HC-SR04 GND  → Pin 6  (GND)
  HC-SR04 TRIG → any free GPIO output pin
  HC-SR04 ECHO → GPIO input  **via voltage divider**
                 (5 V → 1 kΩ → ECHO_PIN → 2 kΩ → GND)
                 because Jetson GPIO is 3.3 V tolerant only.

Default GPIO board pins (configurable via ROS2 parameters):
  front_trig_pin  : 15
  front_echo_pin  : 13
  right_trig_pin  : 11   (set enable_right:=false if sensor not fitted)
  right_echo_pin  : 7

Run standalone:
  python3 jetbot_ultrasonic_driver.py

Or configure pins:
  ros2 run ntu_robotsim jetbot_ultrasonic_driver \
    --ros-args -p front_trig_pin:=15 -p front_echo_pin:=13 \
               -p enable_right:=false
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

try:
    import Jetson.GPIO as GPIO
    HAS_GPIO = True
except ImportError:
    HAS_GPIO = False

# Speed of sound in m/s (at ~20 °C)
SOUND_SPEED = 343.0
# Maximum measurable distance (metres) — avoids infinite waits
MAX_DIST_M  = 2.0
MAX_WAIT_S  = (MAX_DIST_M * 2) / SOUND_SPEED + 0.005  # travel time + margin

# Minimum valid distance (filters noise/spurious echoes)
MIN_DIST_M  = 0.02


def _measure_once(trig_pin: int, echo_pin: int) -> float:
    """Return one distance reading in metres, or inf on timeout/error."""
    # Send 10 µs trigger pulse
    GPIO.output(trig_pin, GPIO.HIGH)
    time.sleep(10e-6)
    GPIO.output(trig_pin, GPIO.LOW)

    t_start = time.monotonic()

    # Wait for echo rising edge
    deadline = t_start + MAX_WAIT_S
    while GPIO.input(echo_pin) == GPIO.LOW:
        if time.monotonic() > deadline:
            return float('inf')
    pulse_start = time.monotonic()

    # Wait for echo falling edge
    deadline = pulse_start + MAX_WAIT_S
    while GPIO.input(echo_pin) == GPIO.HIGH:
        if time.monotonic() > deadline:
            return float('inf')
    pulse_end = time.monotonic()

    dist = (pulse_end - pulse_start) * SOUND_SPEED / 2.0
    if dist < MIN_DIST_M or dist > MAX_DIST_M:
        return float('inf')
    return dist


class UltrasonicDriver(Node):

    def __init__(self):
        super().__init__('jetbot_ultrasonic_driver')

        # ── Parameters ───────────────────────────────────────────────────────
        self.declare_parameter('front_trig_pin', 15)
        self.declare_parameter('front_echo_pin', 13)
        self.declare_parameter('right_trig_pin', 11)
        self.declare_parameter('right_echo_pin', 7)
        self.declare_parameter('enable_right', True)
        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('num_samples', 3)        # median filter size

        self.front_trig = self.get_parameter('front_trig_pin').value
        self.front_echo = self.get_parameter('front_echo_pin').value
        self.right_trig = self.get_parameter('right_trig_pin').value
        self.right_echo = self.get_parameter('right_echo_pin').value
        self.enable_right = self.get_parameter('enable_right').value
        rate_hz = self.get_parameter('publish_rate_hz').value
        self.num_samples = self.get_parameter('num_samples').value

        # ── GPIO setup ───────────────────────────────────────────────────────
        if HAS_GPIO:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setup(self.front_trig, GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.front_echo, GPIO.IN)
            if self.enable_right:
                GPIO.setup(self.right_trig, GPIO.OUT, initial=GPIO.LOW)
                GPIO.setup(self.right_echo, GPIO.IN)
            self.get_logger().info(
                f'GPIO ready  front TRIG={self.front_trig} ECHO={self.front_echo}'
                + (f'  right TRIG={self.right_trig} ECHO={self.right_echo}'
                   if self.enable_right else '  (right sensor disabled)')
            )
        else:
            self.get_logger().warn(
                'Jetson.GPIO not found — running in SIMULATION mode '
                '(publishing fixed test distances of 1.0 m).'
            )

        # ── Publishers ───────────────────────────────────────────────────────
        self.front_pub = self.create_publisher(Float32, '/distance/front', 10)
        if self.enable_right:
            self.right_pub = self.create_publisher(Float32, '/distance/right', 10)
        else:
            self.right_pub = None

        # ── Timer ────────────────────────────────────────────────────────────
        self.create_timer(1.0 / rate_hz, self._poll)
        self.get_logger().info(f'Ultrasonic driver started at {rate_hz:.0f} Hz')

    # ── Polling ──────────────────────────────────────────────────────────────

    def _read_median(self, trig: int, echo: int) -> float:
        """Take num_samples readings and return the median."""
        samples = []
        for _ in range(self.num_samples):
            d = _measure_once(trig, echo)
            if d != float('inf'):
                samples.append(d)
            time.sleep(0.005)   # brief gap between pulses
        if not samples:
            return float('inf')
        samples.sort()
        return samples[len(samples) // 2]

    def _poll(self) -> None:
        if HAS_GPIO:
            front_dist = self._read_median(self.front_trig, self.front_echo)
            right_dist = (
                self._read_median(self.right_trig, self.right_echo)
                if self.enable_right else float('inf')
            )
        else:
            # Simulation / no-GPIO fallback: publish fixed 1 m
            front_dist = 1.0
            right_dist = 1.0

        msg_front = Float32()
        msg_front.data = float(front_dist)
        self.front_pub.publish(msg_front)

        if self.right_pub is not None:
            msg_right = Float32()
            msg_right.data = float(right_dist)
            self.right_pub.publish(msg_right)

        self.get_logger().debug(
            f'front={front_dist:.3f} m  right={right_dist:.3f} m',
        )

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def destroy_node(self) -> None:
        if HAS_GPIO:
            GPIO.cleanup()
            self.get_logger().info('GPIO cleaned up')
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = UltrasonicDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
