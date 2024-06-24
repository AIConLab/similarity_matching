#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/UInt8.h>
#include <cv_bridge/cv_bridge.h>
#include "image_matcher.hh"
#include <boost/circular_buffer.hpp>

class SimilarityMatcher
{
public:
    SimilarityMatcher(ros::NodeHandle &nh);
    ~SimilarityMatcher();
private:
    ros::NodeHandle &_nh;
    ros::Subscriber _target_image_sub;
    ros::Subscriber _similarity_score_sub;
    ros::Publisher _similarity_score_pub;
    
    cv::Mat _target_image;
    ImageMatcher _image_matcher;
    
    boost::circular_buffer<cv::Mat> _image_buffer;
    
    void _target_image_callback(const sensor_msgs::Image::ConstPtr &msg);
    void _similarity_score_callback(const sensor_msgs::Image::ConstPtr &msg);
};
