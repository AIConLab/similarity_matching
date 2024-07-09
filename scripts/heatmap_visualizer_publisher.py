#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from similarity_mapping.msg import Keypoints
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


# Input:
#    target image (ROS Sensor_msgs/Image), scene image (ROS Sensor_msgs/Image), keypoints_target (Keypoints.msg (Float32[])), keypoints_scene (Keypoints.msg (Float32[])), similarity_score (Float32)
# Output:
#    heatmap visualization (ROS Sensor_msgs/Image)

class HeatmapVisualizerPublisher:
    def __init__(self):
        rospy.init_node('heatmap_visualizer', anonymous=True)
        self.bridge = CvBridge()
        
        # Subscribers
        rospy.Subscriber("current_target_image", Image, self.__current_target_image_callback)
        rospy.Subscriber("current_scene_image", Image, self.__current_scene_callback)
        rospy.Subscriber("current_target_keypoints", Keypoints, self.__current_target_keypoints_callback)
        rospy.Subscriber("current_scene_keypoints", Keypoints, self.__current_scene_keypoints_callback)
        rospy.Subscriber("current_similarity_score", Float32, self.__current_similarity_score_callback)
        
        # Publisher
        self.heatmap_pub = rospy.Publisher('heatmap_visualization', Image, queue_size=1)
        
        # Initialize data storage
        self.target_image = None
        self.scene_image = None
        self.target_keypoints = None
        self.scene_keypoints = None
        self.similarity_score = None
        
        # Set up timer for periodic publishing
        rospy.Timer(rospy.Duration(0.1), self.publish_heatmap)

    def __current_target_image_callback(self, msg):
        self.target_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def __current_scene_callback(self, msg):
        self.scene_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def __current_target_keypoints_callback(self, msg):
        self.target_keypoints = [(kp.x, kp.y) for kp in msg.keypoints]

    def __current_scene_keypoints_callback(self, msg):
        self.scene_keypoints = [(kp.x, kp.y) for kp in msg.keypoints]

    def __current_similarity_score_callback(self, msg):
        self.similarity_score = msg.data

    def __create_heatmap(self, shape, keypoints):
        heatmap = np.zeros(shape[:2], dtype=np.float32)
        for x, y in keypoints:
            x, y = int(x), int(y)
            if 0 <= x < shape[1] and 0 <= y < shape[0]:
                heatmap[y, x] += 1
        heatmap = gaussian_filter(heatmap, sigma=10)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        return heatmap

    def __overlay_heatmap(self, image, heatmap):
        heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        return cv2.addWeighted(image, 0.7, heatmap_color, 0.3, 0)

    def publish_heatmap(self, event):
        if all([self.target_image is not None,
                self.scene_image is not None,
                self.target_keypoints is not None,
                self.scene_keypoints is not None,
                self.similarity_score is not None]):
            
            # Create heatmaps
            target_heatmap = self.__create_heatmap(self.target_image.shape, self.target_keypoints)
            scene_heatmap = self.__create_heatmap(self.scene_image.shape, self.scene_keypoints)
            
            # Apply heatmaps
            target_with_heatmap = self.__overlay_heatmap(self.target_image, target_heatmap)
            scene_with_heatmap = self.__overlay_heatmap(self.scene_image, scene_heatmap)
            
            # Resize target image for overlay
            h, w = self.scene_image.shape[:2]
            target_small = cv2.resize(target_with_heatmap, (w // 4, h // 4))
            
            # Create final frame
            final_frame = scene_with_heatmap.copy()
            final_frame[10:10+target_small.shape[0], 10:10+target_small.shape[1]] = target_small
            
            # Add similarity score
            cv2.putText(final_frame, f"Similarity: {self.similarity_score:.2f}", (w - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Publish the final frame
            heatmap_msg = self.bridge.cv2_to_imgmsg(final_frame, "bgr8")
            self.heatmap_pub.publish(heatmap_msg)

if __name__ == '__main__':
    try:
        heatmap_visualizer = HeatmapVisualizerPublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass