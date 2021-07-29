################################################################################
# Set minimum required version of cmake, project name and compile options
################################################################################
cmake_minimum_required(VERSION 2.8.3)
project(thesis_detect)

################################################################################
# Find catkin packages and libraries for catkin and system dependencies
################################################################################
find_package(catkin REQUIRED COMPONENTS
  dynamic_reconfigure
  rospy
  std_msgs
  geometry_msgs
  message_generation
)

################################################################################
# Setup for python modules and scripts
################################################################################
catkin_python_setup()

################################################################################
# Declare ROS messages, services and actions
################################################################################

################################################################################
# Declare ROS dynamic reconfigure parameters
################################################################################
generate_dynamic_reconfigure_options(
  cfg/DetectLaneParams.cfg
)

add_message_files(
  FILES
  CrossingData.msg
  CrossingDataArray.msg
  DetectTrafficPosition.msg
  DetectCollision.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)
################################################################################
# Declare catkin specific configuration to be passed to dependent projects
################################################################################
catkin_package(
  CATKIN_DEPENDS dynamic_reconfigure
  message_runtime
)

################################################################################
# Build
################################################################################

################################################################################
# Install
################################################################################
catkin_install_python(PROGRAMS
  nodes/detect_lane
  nodes/detect_traffic_light
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY cfg file launch param
	DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

################################################################################
# Test
################################################################################

