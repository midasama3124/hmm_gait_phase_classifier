cmake_minimum_required(VERSION 2.8.3)
project(hmm_gait_phase_classifier)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  roslib
)

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES hmm_gait_phase_classifier
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs roslib
#  DEPENDS system_lib
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

# A collection of utilities for the gait phase classifier
add_library(feature_extractor
  src/util/feature_extractor.cpp
)
target_link_libraries(feature_extractor
  ${catkin_LIBRARIES}
)

add_library(gait_cycle_classifier
  src/util/gait_cycle_classifier.cpp
)
target_link_libraries(gait_cycle_classifier
  ${catkin_LIBRARIES}
  feature_extractor
)

# Build main ROS node for threshold-based gait phase detection
add_executable(threshold_detection_node src/nodes/threshold_detection_node.cpp)
target_link_libraries(threshold_detection_node
  ${catkin_LIBRARIES}
  feature_extractor
  gait_cycle_classifier
)

# Build main ROS node for offline threshold-based gait phase detection
add_executable(offline_threshold_detection_node src/nodes/offline_threshold_detection_node.cpp)
target_link_libraries(offline_threshold_detection_node
  ${catkin_LIBRARIES}
  feature_extractor
  gait_cycle_classifier
)

#############
## Install ##
#############

install(TARGETS feature_extractor gait_cycle_classifier threshold_detection_node offline_threshold_detection_node
#install(TARGETS feature_extractor gait_cycle_classifier threshold_detection_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_hmm_gait_phase_classifier.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
