#include "similarity_matcher.hh"
#include <std_msgs/UInt8.h>

SimilarityMatcher::SimilarityMatcher(ros::NodeHandle &nh) : _nh(nh), _image_buffer(10)
{
    ros::NodeHandle private_nh("~");  // Create a private node handle

    _target_image_sub = _nh.subscribe("target_image", 1, &SimilarityMatcher::_target_image_callback, this);
    _similarity_score_sub = _nh.subscribe("camera_image_stream", 1, &SimilarityMatcher::_similarity_score_callback, this);
    _similarity_score_pub = private_nh.advertise<std_msgs::UInt8>("similarity_score", 1);
}


SimilarityMatcher::~SimilarityMatcher() {}

// Recieve the target image and store it in the _target_image member variable as a cv::Mat
void SimilarityMatcher::_target_image_callback(const sensor_msgs::Image::ConstPtr &msg)
{
    try
    {
        // Convert ROS image message to OpenCV image
        _target_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("Failed to update target image: %s", e.what());
    }
}

void SimilarityMatcher::_similarity_score_callback(const sensor_msgs::Image::ConstPtr &msg)
{
    try
    {
        // Convert ROS image message to OpenCV image
        cv::Mat scene_image = cv_bridge::toCvCopy(msg, "bgr8")->image;

        // Add the scene image to the buffer
        _image_buffer.push_back(scene_image);

        // Only compare images if a valid target image has been received
        if (!_target_image.empty())
        {
            uint8_t similarity_score = _image_matcher.compare_images(_target_image, _image_buffer.back());
            std_msgs::UInt8 score_msg;
            score_msg.data = similarity_score;
            _similarity_score_pub.publish(score_msg);
        }
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
    catch (std::runtime_error &e)
    {
        ROS_ERROR("Runtime error: %s", e.what());
    }
}