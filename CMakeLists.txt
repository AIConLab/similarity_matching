cmake_minimum_required(VERSION 3.0.2)
project(similarity_mapping)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O3 -Wall -Wextra -Wpedantic)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  actionlib
  cv_bridge
  geographic_msgs
  image_transport
  move_base_msgs
  roscpp
  std_msgs
  sensor_msgs
  roslib
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES similarity_mapping
  CATKIN_DEPENDS actionlib cv_bridge geographic_msgs image_transport move_base_msgs roscpp std_msgs sensor_msgs
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/image_matcher.cc
  src/similarity_matcher.cc
)

## Declare a C++ executable
add_executable(similarity_matching_node src/similarity_matching_node.cc)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
)

target_link_libraries(similarity_matching_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
)

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/target_image_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)