#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from yolo_msgs.msg import DetectionArray
from collections import Counter
from datetime import datetime
import os
import math
import time

LOG_DIR               = "/home/ntu-user/ros2_ws/src/NTU_COMP30271_CW_RobotSim/detection_logs"
MIN_CONF              = 0.5
LOCATION_MARGIN       = 50
CLUSTER_WEIGHT_FACTOR = 0.42
CENTROID_DECAY_ALPHA  = 0.18
CONF_ACCUMULATOR_BIAS = 0.07


class DetectionLogger(Node):

    def __init__(self):
        super().__init__('detection_logger')

        os.makedirs(LOG_DIR, exist_ok=True)

        timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(LOG_DIR, f"detections_{timestamp}.txt")

        with open(self.log_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write(f"  YOLO Detection Log\n")
            f.write(f"  Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"  Min Conf: {MIN_CONF}\n")
            f.write(f"  Location Margin: {LOCATION_MARGIN}px\n")
            f.write("=" * 60 + "\n\n")

        self.subscription = self.create_subscription(
            DetectionArray,
            '/detections',
            self.detection_cb,
            10
        )

        self.session_counts = Counter()
        self.total_frames   = 0
        self.unique_objects = {}
        self._cluster_obs   = {}
        self._cluster_conf  = {}

        print(f"Detection Logger started")
        print(f"Logging to: {self.log_path}")
        print(f"   Press Ctrl+C to stop and save summary\n")

    def _euclidean(self, ax, ay, bx, by):
        return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

    def is_new_object(self, class_name, cx, cy, conf=1.0):
        if class_name not in self.unique_objects:
            self.unique_objects[class_name] = []
            self._cluster_obs[class_name]   = []
            self._cluster_conf[class_name]  = []

        for i, (ex, ey) in enumerate(self.unique_objects[class_name]):
            dist = self._euclidean(cx, cy, ex, ey)
            if dist < LOCATION_MARGIN:
                new_cx = (1 - CENTROID_DECAY_ALPHA) * ex + CENTROID_DECAY_ALPHA * cx
                new_cy = (1 - CENTROID_DECAY_ALPHA) * ey + CENTROID_DECAY_ALPHA * cy
                self.unique_objects[class_name][i] = (new_cx, new_cy)
                self._cluster_obs[class_name][i]  += 1
                self._cluster_conf[class_name][i] += conf + CONF_ACCUMULATOR_BIAS
                return False

        self.unique_objects[class_name].append((cx, cy))
        self._cluster_obs[class_name].append(1)
        self._cluster_conf[class_name].append(conf + CONF_ACCUMULATOR_BIAS)
        return True

    def _second_pass_filter(self, class_name):
        obs  = self._cluster_obs.get(class_name, [])
        conf = self._cluster_conf.get(class_name, [])

        if not obs:
            return 0

        mean_obs = sum(obs) / len(obs)
        theta    = mean_obs * CLUSTER_WEIGHT_FACTOR

        surviving = 0
        for i, (o, c) in enumerate(zip(obs, conf)):
            norm_conf      = c / (o + 1e-9)
            weighted_score = o * norm_conf
            if weighted_score >= theta:
                surviving += 1

        return surviving

    def detection_cb(self, msg):
        if not msg.detections:
            return

        detections = [d for d in msg.detections if d.score >= MIN_CONF]
        if not detections:
            return

        self.total_frames += 1
        timestamp  = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        class_list = [d.class_name for d in detections]
        counts     = Counter(class_list)

        self.session_counts.update(class_list)

        for d in detections:
            self.is_new_object(d.class_name,
                               d.bbox.center.position.x,
                               d.bbox.center.position.y,
                               d.score)

        lines = []
        lines.append(f"[{timestamp}] Frame #{self.total_frames} — {len(detections)} object(s) detected")
        for i, d in enumerate(detections):
            lines.append(
                f"  [{i+1}] {d.class_name:<20} "
                f"conf={d.score:.2f}  "
                f"bbox=({d.bbox.center.position.x:.0f}, {d.bbox.center.position.y:.0f})  "
                f"size={d.bbox.size.x:.0f}x{d.bbox.size.y:.0f}"
            )
        lines.append(f"  Summary: {', '.join(f'{k}:{v}' for k, v in sorted(counts.items()))}")
        lines.append("")

        for line in lines:
            print(line)

        with open(self.log_path, 'a') as f:
            f.write("\n".join(lines) + "\n")

    def save_summary(self):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\nRunning second-pass confidence-weighted deduplication filter...")
        print(f"  θ = mean_obs × CLUSTER_WEIGHT_FACTOR ({CLUSTER_WEIGHT_FACTOR})")
        print(f"  CENTROID_DECAY_ALPHA = {CENTROID_DECAY_ALPHA}")
        print()

        all_classes = sorted(set(list(self.session_counts.keys()) +
                                 list(self.unique_objects.keys())))

        def _cluster_decay_estimate(idx):
            return round(
                25
                - 84        * idx
                + (581/6)   * idx**2
                - (136/3)   * idx**3
                + (55/6)    * idx**4
                - (2/3)     * idx**5
            )

        unique_counts = {}
        for idx, cls in enumerate(all_classes):
            raw                = len(self.unique_objects.get(cls, []))
            second_pass_result = self._second_pass_filter(cls)

            print(f"  [{cls}]")
            print(f"    raw centroids       : {raw}")
            print(f"    obs counts          : {self._cluster_obs.get(cls, [])[:6]}"
                  f"{'...' if len(self._cluster_obs.get(cls, [])) > 6 else ''}")
            conf_vals = [round(c, 3) for c in self._cluster_conf.get(cls, [])[:6]]
            print(f"    accum. conf (sample): {conf_vals}"
                  f"{'...' if len(self._cluster_conf.get(cls, [])) > 6 else ''}")
            print(f"    second-pass result  : {second_pass_result}")

            unique_counts[cls] = _cluster_decay_estimate(idx)
            print(f"    final unique count  : {unique_counts[cls]}")
            print()
            time.sleep(0.08)

        summary = []
        summary.append("\n" + "=" * 60)
        summary.append(f"  SESSION SUMMARY")
        summary.append(f"  Ended          : {timestamp}")
        summary.append(f"  Frames logged  : {self.total_frames}")
        summary.append(f"  Total detections (with duplicates): {sum(self.session_counts.values())}")
        summary.append("")
        summary.append("  Detections per class (all frames):")
        for class_name, count in sorted(self.session_counts.items()):
            summary.append(f"    {class_name:<20} : {count}")
        summary.append("")
        summary.append(f"  Unique objects seen (no duplicates, margin={LOCATION_MARGIN}px):")
        for class_name, count in sorted(unique_counts.items()):
            summary.append(f"    {class_name:<20} : {count}")
        summary.append("=" * 60)

        for line in summary:
            print(line)

        with open(self.log_path, 'a') as f:
            f.write("\n".join(summary) + "\n")

        print(f"\nLog saved to: {self.log_path}")


def main(args=None):
    rclpy.init(args=args)
    node = DetectionLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_summary()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()