#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Float32.h>
#include <ros/package.h>

#include <similarity_mapping/Keypoints.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/types.hpp>

#include <mutex>
#include <thread>
#include <atomic>

class SimilarityMatcher
{
    public:
        SimilarityMatcher(ros::NodeHandle &nh);
        ~SimilarityMatcher();

    private:
        ros::NodeHandle _nh;
        ros::Subscriber _camera_stream_sub;
        ros::Subscriber _target_image_sub;
        
        ros::Publisher _current_target_image_pub;
        ros::Publisher _current_scene_image_pub;
        ros::Publisher _current_similarity_score_pub;
        ros::Publisher _current_target_keypoints_pub;
        ros::Publisher _current_scene_keypoints_pub;
        
        cv::Mat _target_image;
        cv::Mat _latest_scene_image;
        float _latest_similarity_score;
        std::vector<cv::KeyPoint> _target_keypoints;
        std::vector<cv::KeyPoint> _scene_keypoints;

        std::mutex _data_mutex;
        std::thread _processing_thread;
        std::atomic<bool> _stop_thread;
        std::atomic<bool> _all_pub_data_ready;

        // Pipes for communication with Python process
        int _pipe_to_python[2];
        int _pipe_from_python[2];

        void _camera_stream_callback(const sensor_msgs::Image::ConstPtr &msg);
        void _target_image_callback(const sensor_msgs::Image::ConstPtr &msg);

        void _publish_data();
        
        bool _write_image_to_python(const cv::Mat& image);
        bool _read_python_output();
        void _process_images();
        bool _are_images_ready();
};