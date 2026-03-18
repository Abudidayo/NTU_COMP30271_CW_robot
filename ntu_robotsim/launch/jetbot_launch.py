"""
JetBot Real-Hardware Launch File
----------------------------------
Replaces yolo_unified.launch.py for the physical Waveshare JetBot.
No Gazebo, no Nav2, no OctoMap — just the nodes needed on hardware.

Startup sequence:
  t=0s   Camera driver     (v4l2_camera publishes /image_raw)
  t=2s   YOLO detection    (subscribes /image_raw → publishes /detections)
  t=3s   Motor driver      (subscribes /cmd_vel  → drives motors)
  t=3s   Ultrasonic driver (reads GPIO → publishes /distance/front, /distance/right)
  t=5s   Wall follower     (subscribes /distance/* + /detections → /cmd_vel)

Usage:
  ros2 launch ntu_robotsim jetbot_launch.py

Optional overrides:
  ros2 launch ntu_robotsim jetbot_launch.py \
      camera_device:=/dev/video1 \
      yolo_model:=/path/to/best_nano.pt \
      enable_right_sensor:=false \
      enable_landmark_logger:=true
"""

import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    TimerAction,
    ExecuteProcess,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

# ── Paths ─────────────────────────────────────────────────────────────────────
LAUNCH_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(
    os.path.dirname(LAUNCH_DIR), 'models', 'custom_models'
)
DEFAULT_MODEL = os.path.join(MODELS_DIR, 'best_nano.pt')


def generate_launch_description() -> LaunchDescription:

    # ── Declare arguments ─────────────────────────────────────────────────────
    args = [
        DeclareLaunchArgument(
            'camera_device', default_value='/dev/video0',
            description='V4L2 camera device for the JetBot CSI camera'),
        DeclareLaunchArgument(
            'camera_topic', default_value='/image_raw',
            description='ROS2 topic where the camera publishes images'),
        DeclareLaunchArgument(
            'yolo_model', default_value=DEFAULT_MODEL,
            description='Path to YOLO .pt weights file'),
        DeclareLaunchArgument(
            'yolo_threshold', default_value='0.5',
            description='YOLO confidence threshold'),
        DeclareLaunchArgument(
            'yolo_device', default_value='cuda:0',
            description='Torch device for YOLO (cuda:0 or cpu)'),
        DeclareLaunchArgument(
            'linear_speed', default_value='0.20',
            description='Base forward speed (m/s equivalent, mapped to motor 0-1)'),
        DeclareLaunchArgument(
            'wall_follow_dist', default_value='0.50',
            description='Target distance to right wall (metres)'),
        DeclareLaunchArgument(
            'front_stop_dist', default_value='0.40',
            description='Front distance at which robot turns (metres)'),
        DeclareLaunchArgument(
            'enable_right_sensor', default_value='true',
            description='Set false if only front ultrasonic sensor is fitted'),
        DeclareLaunchArgument(
            'front_trig_pin', default_value='15',
            description='Jetson Nano BOARD pin number for front TRIG'),
        DeclareLaunchArgument(
            'front_echo_pin', default_value='13',
            description='Jetson Nano BOARD pin number for front ECHO'),
        DeclareLaunchArgument(
            'right_trig_pin', default_value='11',
            description='Jetson Nano BOARD pin number for right TRIG'),
        DeclareLaunchArgument(
            'right_echo_pin', default_value='7',
            description='Jetson Nano BOARD pin number for right ECHO'),
        DeclareLaunchArgument(
            'enable_landmark_logger', default_value='false',
            description='Set true to also start the landmark CSV logger'),
    ]

    # ── Nodes ─────────────────────────────────────────────────────────────────

    # 1. Camera driver (v4l2_camera)
    #    Publishes /image_raw from the JetBot CSI camera
    camera_node = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        name='jetbot_camera',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('camera_device'),
            'image_size':   [640, 480],
            'pixel_format': 'YUYV',
        }],
    )

    # 2. YOLO detection node
    #    Input:  /image_raw (camera topic)
    #    Output: /detections (yolo_msgs/DetectionArray)
    yolo_node = TimerAction(
        period=2.0,
        actions=[
            Node(
                package='yolo_ros',
                executable='yolo_node',
                name='yolo_detector',
                output='screen',
                parameters=[{
                    'model':       LaunchConfiguration('yolo_model'),
                    'device':      LaunchConfiguration('yolo_device'),
                    'threshold':   LaunchConfiguration('yolo_threshold'),
                    'input_image_topic': LaunchConfiguration('camera_topic'),
                }],
            )
        ],
    )

    # 3. Motor driver
    #    Subscribes: /cmd_vel  →  drives JetBot motors via jetbot library
    motor_driver = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_motor_driver.py'),
                ],
                output='screen',
            )
        ],
    )

    # 4. Ultrasonic distance driver
    #    Reads HC-SR04 via Jetson GPIO → /distance/front, /distance/right
    ultrasonic_driver = TimerAction(
        period=3.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_ultrasonic_driver.py'),
                    '--ros-args',
                    '-p', ['front_trig_pin:=', LaunchConfiguration('front_trig_pin')],
                    '-p', ['front_echo_pin:=', LaunchConfiguration('front_echo_pin')],
                    '-p', ['right_trig_pin:=', LaunchConfiguration('right_trig_pin')],
                    '-p', ['right_echo_pin:=', LaunchConfiguration('right_echo_pin')],
                    '-p', ['enable_right:=',   LaunchConfiguration('enable_right_sensor')],
                ],
                output='screen',
            )
        ],
    )

    # 5. Wall follower (main navigation node)
    #    Subscribes: /distance/front, /distance/right, /detections
    #    Publishes:  /cmd_vel
    wall_follower = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_wall_follower.py'),
                    '--ros-args',
                    '-p', ['linear_speed:=',    LaunchConfiguration('linear_speed')],
                    '-p', ['wall_follow_dist:=', LaunchConfiguration('wall_follow_dist')],
                    '-p', ['front_stop_dist:=',  LaunchConfiguration('front_stop_dist')],
                ],
                output='screen',
            )
        ],
    )

    # 6. Landmark CSV logger (optional)
    landmark_logger = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    'python3',
                    os.path.join(LAUNCH_DIR, 'jetbot_landmark_csv_logger.py'),
                ],
                output='screen',
            )
        ],
    )

    # ── Assemble ──────────────────────────────────────────────────────────────
    ld = LaunchDescription(args + [
        camera_node,
        yolo_node,
        motor_driver,
        ultrasonic_driver,
        wall_follower,
    ])

    # Conditionally add landmark logger
    # (LaunchCondition on string args requires IfCondition)
    from launch.conditions import IfCondition
    from launch.actions import GroupAction
    ld.add_action(
        GroupAction(
            condition=IfCondition(LaunchConfiguration('enable_landmark_logger')),
            actions=[landmark_logger],
        )
    )

    return ld
