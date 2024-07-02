#include <ros/ros.h>
#include "similarity_matcher.hpp"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "similarity_matching_node");
    ros::NodeHandle nh;

    SimilarityMatcher simp(nh);
    ros::spin();

    return 0;
}