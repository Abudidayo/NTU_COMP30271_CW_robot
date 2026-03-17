"""
jetbot_real_bringup.launch.py
==============================
Bringup launch file for the PHYSICAL JetBot (ROS2 Eloquent, Ubuntu 18.04).
No Gazebo, no topic bridges — direct hardware drivers only.

Stack started (in order):
  T=0s   Intel RealSense D435 camera driver
  T=2s   pointcloud_to_laserscan  (depth cloud → /scan for wall follower)
  T=2s   rf2o_laser_odometry      (LaserScan → /odom visual odometry)
  T=3s   Static TF publishers     (base_link → camera_link, base_link → laser)
  T=5s   Nav2 navigation stack    (using jetbot_nav2_params.yaml)
  T=8s   YOLO object detection    (best_nano.pt on /camera/color/image_raw)
  T=10s  Wall follower            (subscribes /scan + /detections, publishes /cmd_vel)
  T=10s  Landmark CSV logger      (subscribes /odom + /detections)

Prerequisites on JetBot:
  sudo apt install ros-eloquent-realsense2-camera
  sudo apt install ros-eloquent-pointcloud-to-laserscan
  pip3 install rf2o_laser_odometry  (or build from source)
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('ntu_robotsim')
    config_dir = os.path.join(pkg, 'config')

    nav2_params   = os.path.join(config_dir, 'jetbot_nav2_params.yaml')
    pc2scan_cfg   = os.path.join(config_dir, 'pointcloud_to_laserscan.yaml')
    yolo_model    = os.path.join(pkg, 'models', 'custom_models', 'best_nano.pt')

    # ── T=0s  RealSense D435 camera driver ───────────────────────────────────
    realsense_node = Node(
        package='realsense2_camera',
        executable='realsense2_camera_node',
        name='realsense2_camera',
        output='screen',
        parameters=[{
            'enable_color': True,
            'enable_depth': True,
            'enable_pointcloud': True,
            'pointcloud_texture_stream': 'RS2_STREAM_COLOR',
            'color_width': 640,
            'color_height': 480,
            'depth_width': 640,
            'depth_height': 480,
            'color_fps': 30.0,
            'depth_fps': 30.0,
        }],
    )

    # ── T=2s  pointcloud_to_laserscan ────────────────────────────────────────
    # Converts /camera/depth/color/points → /scan (2D LaserScan)
    pc_to_scan_node = TimerAction(
        period=2.0,
        actions=[Node(
            package='pointcloud_to_laserscan',
            executable='pointcloud_to_laserscan_node',
            name='pointcloud_to_laserscan',
            output='screen',
            remappings=[
                ('cloud_in', '/camera/depth/color/points'),
                ('scan',     '/scan'),
            ],
            parameters=[pc2scan_cfg],
        )],
    )

    # ── T=2s  rf2o_laser_odometry (visual/laser odometry from /scan) ─────────
    # Publishes /odom based on consecutive LaserScan frame matching.
    # Alternative: replace with any odometry source that publishes nav_msgs/Odometry on /odom
    rf2o_node = TimerAction(
        period=2.0,
        actions=[Node(
            package='rf2o_laser_odometry',
            executable='rf2o_laser_odometry_node',
            name='rf2o_laser_odometry',
            output='screen',
            parameters=[{
                'laser_scan_topic': '/scan',
                'odom_topic':       '/odom',
                'publish_tf':       True,
                'base_frame_id':    'base_link',
                'odom_frame_id':    'odom',
                'init_pose_from_topic': '',
                'freq':             10.0,
            }],
        )],
    )

    # ── T=3s  Static TF publishers ───────────────────────────────────────────
    # Positions match the simulation SDF:
    #   camera_link: (x=0.1, y=0, z=0.19) relative to base_link
    #   laser frame: (x=0,   y=0, z=0.27) relative to base_link (depth scan origin)
    tf_camera = TimerAction(
        period=3.0,
        actions=[Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_base_to_camera',
            arguments=['0.1', '0', '0.19', '0', '0', '0', 'base_link', 'camera_link'],
        )],
    )

    tf_laser = TimerAction(
        period=3.0,
        actions=[Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_base_to_laser',
            arguments=['0', '0', '0.27', '0', '0', '0', 'base_link', 'laser'],
        )],
    )

    tf_map_odom = TimerAction(
        period=3.0,
        actions=[Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='tf_map_to_odom',
            arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
        )],
    )

    # ── T=5s  Nav2 navigation stack ──────────────────────────────────────────
    nav2 = TimerAction(
        period=5.0,
        actions=[Node(
            package='nav2_bringup',
            executable='bringup_launch.py',
            name='nav2_bringup',
            output='screen',
            parameters=[nav2_params],
        )],
    )

    # ── T=8s  YOLO detection ─────────────────────────────────────────────────
    # Uses best_nano.pt (YOLOv8 Nano). If Eloquent's yolo_bringup differs,
    # you may need to run this node directly instead of via yolo.launch.py.
    yolo = TimerAction(
        period=8.0,
        actions=[Node(
            package='yolo_bringup',
            executable='yolo_node',
            name='yolo_node',
            output='screen',
            parameters=[{
                'model':             yolo_model,
                'device':            'cuda:0',
                'threshold':         0.5,
                'input_image_topic': '/camera/color/image_raw',
                'image_reliability': 'best_effort',
            }],
        )],
    )

    # ── T=10s  Wall follower ─────────────────────────────────────────────────
    wall_follower = TimerAction(
        period=10.0,
        actions=[Node(
            package='ntu_robotsim',
            executable='wall_follower',
            name='wall_follower',
            output='screen',
        )],
    )

    # ── T=10s  Landmark CSV logger ───────────────────────────────────────────
    landmark_logger = TimerAction(
        period=10.0,
        actions=[ExecuteProcess(
            cmd=['python3', os.path.join(pkg, 'launch', 'landmark_csv_logger.py')],
            output='screen',
        )],
    )

    return LaunchDescription([
        realsense_node,
        pc_to_scan_node,
        rf2o_node,
        tf_camera,
        tf_laser,
        tf_map_odom,
        nav2,
        yolo,
        wall_follower,
        landmark_logger,
    ])
