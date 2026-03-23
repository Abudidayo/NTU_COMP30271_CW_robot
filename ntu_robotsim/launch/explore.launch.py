import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.actions import IncludeLaunchDescription
from launch.actions import TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_ntu_robotsim = get_package_share_directory("ntu_robotsim")

    launch_unified = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ntu_robotsim, "launch", "unified_launch.launch.py")
        )
    )

    launch_yolo = TimerAction(
        period=10.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    FindPackageShare("yolo_bringup"),
                    "/launch/yolo.launch.py",
                ]),
                launch_arguments={
                    "model": (
                        "/home/ntu-user/ros2_ws/src/NTU_COMP30271_CW_RobotSim/"
                        "ntu_robotsim/models/custom_models/best_n.pt"
                    ),
                    "device": "cuda:0",
                    "threshold": "0.5",
                    "input_image_topic": "/atlas/rgbd_camera/image",
                }.items(),
            )
        ],
    )

    selector_script = os.path.join(pkg_ntu_robotsim, "launch", "landmark_csv_logger.py")
    launch_selector = TimerAction(
        period=18.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "gnome-terminal",
                    "--",
                    "bash",
                    "-c",
                    (
                        f"python3 {selector_script} --ros-args "
                        "-p use_sim_time:=true; exec bash"
                    ),
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        launch_unified,
        launch_yolo,
        launch_selector,
    ])
