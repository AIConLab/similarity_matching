#include "similarity_matcher.hh"
#include <std_msgs/UInt8.h>

Similarity_Matcher::Similarity_Matcher(ros::NodeHandle &nh) : _nh(nh)
{

    _target_image_sub = _nh.subscribe<sensor_msgs::Image>("/target_image", 1, &Similarity_Matcher::_target_image_callback, this);
    _similarity_score_sub = _nh.subscribe<sensor_msgs::Image>("/camera_image_stream", 1, &Similarity_Matcher::_similarity_score_callback, this);
    _similarity_score_pub = _nh.advertise<std_msgs::Uint8>("/similarity_score", 1);
}

Similarity_Matcher::~Similarity_Matcher()
{
}

void Similarity_Matcher::_target_image_callback(const sensor_msgs::Image::ConstPtr &msg)
{
    _latest_target_image = *msg;
}