#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Image.h>

// Abstract class for similarity matching
class Similarity_Matcher 
{
    public:
        Similarity_Matcher(ros::NodeHandle &nh);
        ~Similarity_Matcher();

    private:

        // Data
        sensor_msgs::Image _latest_target_image;

        // ROS nodes
        ros::NodeHandle &_nh;
        ros::Subscriber _target_image_sub;
        ros::Subscriber _camera_image_stream_sub;
        ros::Publisher _similarity_score_pub;

        // Callback for the target image subscriber, updates the latest target image
        void _target_image_callback(const sensor_msgs::Image::ConstPtr &msg);

        // Callback for the similarity score publisher, publishes the similarity score when an image is published to it
        // Uses the latest target image and the image published to it to calculate the similarity score
        void _similarity_score_callback(const sensor_msgs::Image::ConstPtr &msg);
};