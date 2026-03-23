import os
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Launch RTAB-Map Visual Odometry (rgbd_odometry) for the Atlas robot.

    Subscribes to the robot's RGB-D camera and publishes visual odometry
    on /atlas/vo, replacing the need for ground-truth odometry.
    """

    # --- RTAB-Map Visual Odometry node ------------------------------------
    rgbd_odometry = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'frame_id': 'atlas/base_link',
            'odom_frame_id': 'atlas/odom_vo',
            'publish_tf': True,
            'wait_for_transform': 0.2,
            'approx_sync': True,
            'queue_size': 10,
            'wait_imu_to_init': False,
            'Odom/ResetCountdown': '1',
            'Odom/Strategy': '0',           # 0=Frame-to-Map (more robust)
            'Vis/FeatureType': '6',         # GFTT/BRIEF features
            'Vis/MaxFeatures': '1000',      # more features for better tracking
            'Vis/MinInliers': '10',         # lower threshold to accept matches
        }],
        remappings=[
            ('rgb/image', '/atlas/rgbd_camera/image'),
            ('rgb/camera_info', '/atlas/rgbd_camera/camera_info'),
            ('depth/image', '/atlas/rgbd_camera/depth_image'),
            ('odom', '/atlas/vo'),
        ],
    )

    # --- RTAB-Map SLAM node (uses VO for mapping) -------------------------
    rtabmap_slam = Node(
        package='rtabmap_slam',
        executable='rtabmap',
        name='rtabmap',
        output='screen',
        parameters=[{
            'use_sim_time': True,
            'frame_id': 'atlas/base_link',
            'odom_frame_id': 'atlas/odom_vo',
            'map_frame_id': 'map',
            'subscribe_depth': True,
            'approx_sync': True,
            'queue_size': 10,
            'publish_tf': False,            # we publish map→odom_vo statically
            # RTAB-Map database (reset each run)
            'database_path': '/tmp/rtabmap.db',
            'Mem/IncrementalMemory': 'true',
            'Mem/InitWMWithAllNodes': 'true',
        }],
        remappings=[
            ('rgb/image', '/atlas/rgbd_camera/image'),
            ('rgb/camera_info', '/atlas/rgbd_camera/camera_info'),
            ('depth/image', '/atlas/rgbd_camera/depth_image'),
            ('odom', '/atlas/vo'),
        ],
    )

    return LaunchDescription([
        rgbd_odometry,
        rtabmap_slam,
    ])
