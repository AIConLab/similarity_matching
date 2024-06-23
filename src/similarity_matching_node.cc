#include <ros/ros.h>
#include "similarity_matcher.hh"

int main(int argc, char **argv)
{
    ros::init(argc, argv, "similarity_matching_node");
    ros::NodeHandle nh;

    Similarity_Matcher matcher(nh);
    ros::spin();

    return 0;
}