# JetBot Real-Robot Instructions
**NTU COMP30271 — Sim-to-Real Migration**

> **Hardware:** NVIDIA Jetson Nano · Intel RealSense D435 · ROS2 Eloquent · Ubuntu 18.04 · Python 3.6

---

## 0. First-Time Setup

### 0.1 Connect to the JetBot

Check the IP address on the piOLED display, then connect from your laptop:

```bash
ssh -X jetbot@<jetbot_ip_address>
# Password: jetbot
```

Or open JupyterLab in a browser: `http://<jetbot_ip_address>:8888` (password: `jetbot`)

---

### 0.2 Install Dependencies

```bash
sudo apt-get update
sudo apt install -y \
  ros-eloquent-navigation2 \
  ros-eloquent-nav2-bringup \
  ros-eloquent-realsense2-camera \
  ros-eloquent-pointcloud-to-laserscan \
  ros-eloquent-tf2-ros \
  ros-eloquent-teleop-twist-keyboard
```

**rf2o_laser_odometry** — provides odometry from the laser scan (no wheel encoders needed):

```bash
cd ~/ros2_ws/src
git clone https://github.com/MAPIRlab/rf2o_laser_odometry.git
cd ~/ros2_ws
colcon build --symlink-install --packages-select rf2o_laser_odometry
```

---

### 0.3 Copy the Project to the JetBot

From your laptop (inside `ros2_ws/src/`):

```bash
scp -r NTU_COMP30271_CW_RobotSim jetbot@<jetbot_ip_address>:~/ros2_ws/src/
```

---

### 0.4 Build the Project

```bash
cd ~/ros2_ws
source /opt/ros/eloquent/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## 1. Verify Hardware Before Running

### 1.1 Check the RealSense camera is detected

```bash
rs-enumerate-devices
# Should list: Intel RealSense D435
```

### 1.2 Launch camera driver and confirm topics

```bash
# Terminal 1
source /opt/ros/eloquent/setup.bash
ros2 launch realsense2_camera rs_camera.launch.py

# Terminal 2
ros2 topic list | grep camera
```

Expected topics:
- `/camera/color/image_raw`
- `/camera/depth/color/points`
- `/camera/depth/image_rect_raw`
- `/camera/color/camera_info`

### 1.3 Confirm camera rate

```bash
ros2 topic hz /camera/color/image_raw
# Expected: ~30 Hz
```

### 1.4 Test basic motion before enabling autonomy

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

Use arrow keys to drive. Confirm the wheels spin in the correct direction — if left/right are swapped, check the motor wiring.

---

## 2. Run Full Autonomous Stack

```bash
cd ~/ros2_ws
source /opt/ros/eloquent/setup.bash
source install/setup.bash
ros2 launch ntu_robotsim jetbot_real_bringup.launch.py
```

**Startup sequence:**

| Time | Component |
|------|-----------|
| 0s   | RealSense D435 camera driver |
| 2s   | `pointcloud_to_laserscan` — depth cloud → `/scan` |
| 2s   | `rf2o_laser_odometry` — `/scan` → `/odom` |
| 3s   | Static TF publishers (`base_link` → `camera_link`, `laser`) |
| 5s   | Nav2 navigation stack |
| 8s   | YOLO object detection (`best_nano.pt`, CUDA) |
| 10s  | Right-hand wall follower + sign reactions |
| 10s  | Landmark CSV logger |

**The robot will autonomously:**
- Follow the right wall to traverse the maze
- React to detected signs:
  - `Stop_Sign` → stop for 3 seconds, then resume
  - `Slow_Sign` → reduce speed to 50%
  - `Fast_Sign` → increase speed to 150%
- Count detected objects (Orange, Tree, Car)
- Log unique landmarks to a timestamped CSV

---

## 3. Outputs

### Landmark CSV
```
~/ros2_ws/src/NTU_COMP30271_CW_RobotSim/landmark_logs/landmarks_YYYY-MM-DD-HH-MM.csv
```
Columns: `index`, `item_scanned`, `time_elapsed_s`, `pos_x`, `pos_y`, `pos_z`, `yaw_deg`

### Object counts
```
~/ros2_ws/object_counts.txt
```
Updated every 10 seconds and on shutdown.

### Live detection display *(optional, separate terminal)*

```bash
cd ~/ros2_ws/src/NTU_COMP30271_CW_RobotSim/ntu_robotsim/launch
python3 detection_printer.py
```

### Full detection log *(optional, separate terminal)*

```bash
cd ~/ros2_ws/src/NTU_COMP30271_CW_RobotSim/ntu_robotsim/launch
python3 detection_logger.py
```

---

## 4. Individual Components (debugging)

#### A — RealSense camera
```bash
ros2 launch realsense2_camera rs_camera.launch.py
```

#### B — Depth cloud → LaserScan
```bash
ros2 run pointcloud_to_laserscan pointcloud_to_laserscan_node \
  --ros-args \
  -r cloud_in:=/camera/depth/color/points \
  -r scan:=/scan \
  --params-file $(ros2 pkg prefix ntu_robotsim)/share/ntu_robotsim/config/pointcloud_to_laserscan.yaml
```

#### C — Laser odometry
```bash
ros2 run rf2o_laser_odometry rf2o_laser_odometry_node \
  --ros-args \
  -p laser_scan_topic:=/scan \
  -p odom_topic:=/odom \
  -p base_frame_id:=base_link \
  -p publish_tf:=true
```

#### D — Static TF publishers
```bash
ros2 run tf2_ros static_transform_publisher 0.1 0 0.19 0 0 0 base_link camera_link
ros2 run tf2_ros static_transform_publisher 0 0 0.27 0 0 0 base_link laser
ros2 run tf2_ros static_transform_publisher 0 0 0 0 0 0 map odom
```

#### E — Nav2
```bash
ros2 launch nav2_bringup navigation_launch.py \
  params_file:=$(ros2 pkg prefix ntu_robotsim)/share/ntu_robotsim/config/jetbot_nav2_params.yaml \
  use_sim_time:=false
```

#### F — YOLO detection
```bash
ros2 launch yolo_bringup yolo.launch.py \
  model:=$(ros2 pkg prefix ntu_robotsim)/share/ntu_robotsim/models/custom_models/best_nano.pt \
  device:=cuda:0 \
  threshold:=0.5 \
  input_image_topic:=/camera/color/image_raw
```

#### G — Wall follower (standalone)
```bash
cd ~/ros2_ws && source install/setup.bash
python3 src/NTU_COMP30271_CW_RobotSim/ntu_robotsim/launch/wall_follower.py
```

#### H — Landmark logger (standalone)
```bash
python3 src/NTU_COMP30271_CW_RobotSim/ntu_robotsim/launch/landmark_csv_logger.py
```

#### I — Keyboard teleop
```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

---

## 5. Debugging Tips

#### Verify the TF tree
```bash
ros2 run tf2_tools view_frames
# Open frames.pdf — should show: map → odom → base_link → camera_link / laser
```

#### Check topic rates
```bash
ros2 topic list
ros2 topic hz /scan        # expect ~30 Hz
ros2 topic hz /odom        # expect ~10 Hz
ros2 topic hz /detections  # expect ~5–10 Hz when YOLO is running
```

#### Monitor Jetson Nano resources
```bash
tegrastats
# If RAM > 3.5 GB: reduce camera resolution in the launch file or close JupyterLab
```

#### Record a bag for offline analysis
```bash
ros2 bag record /camera/color/image_raw /scan /odom /detections /cmd_vel
```

#### Replay a bag on your laptop
```bash
ros2 bag play <bag_folder>
```

#### View live camera feed
```bash
ros2 run rqt_image_view rqt_image_view /camera/color/image_raw
```

---

## 6. Simulation vs Physical JetBot

| | Simulation | Physical JetBot |
|---|---|---|
| ROS2 version | Humble | Eloquent |
| Python | 3.10+ | 3.6 |
| Odometry | Ground truth (perfect) | rf2o laser odometry (drifts slowly) |
| LIDAR | 360° Velodyne (simulated) | 85° FOV from depth camera only |
| `cmd_vel` topic | `/atlas/cmd_vel` | `/cmd_vel` |
| Camera topic | `/atlas/rgbd_camera/image` | `/camera/color/image_raw` |
| Depth topic | `/atlas/rgbd_camera/points` | `/camera/depth/color/points` |
| Nav2 simple_commander | Available | **Not available** — use action clients |
| 3D mapping | OctoMap | `slam_toolbox` (2D) recommended |
| Sim time | `use_sim_time: true` | `use_sim_time: false` |
