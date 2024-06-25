#pragma once
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

#include <boost/circular_buffer.hpp>

#include "image_matcher.hh"

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
    
    // default to empty image cv
    cv::Mat _target_image = cv::Mat();

    ImageMatcher _image_matcher;
    
    boost::circular_buffer<cv::Mat> _image_buffer;
    
    void _target_image_callback(const sensor_msgs::Image::ConstPtr &msg);
    void _similarity_score_callback(const sensor_msgs::Image::ConstPtr &msg);
};
