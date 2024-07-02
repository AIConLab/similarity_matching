cmake_minimum_required(VERSION 3.0.2)
project(similarity_mapping)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11 -O3 -Wall -Wextra -Wpedantic -Werror)

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
  message_generation
)

## System dependencies are found with CMake's conventions
find_package(OpenCV REQUIRED)


add_message_files(
  FILES
  Keypoints.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES similarity_mapping
  CATKIN_DEPENDS actionlib cv_bridge geographic_msgs image_transport move_base_msgs roscpp std_msgs sensor_msgs message_runtime
  DEPENDS OpenCV
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  src/similarity_matcher.cpp
)

## Declare a C++ executable
add_executable(similarity_matching_node src/similarity_matching_node.cc)

## Specify libraries to link a library or executable target against
target_link_libraries(similarity_matching_node
  ${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  rt
  pthread
)

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/target_image_publisher.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Mark cpp header files for installation
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h *.hpp *.hh"
  PATTERN ".svn" EXCLUDE
)