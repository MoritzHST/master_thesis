# How to use

1. Download sourcecode and put in in the catkin-workspace directory. All of those folders need to be wrapped by another folder
2. Build the workspace with `catkin build`
3. Source the setup.bash `source devel/setup.bash`
4. Run `roslaunch thesis_sim master_thesis_simulation.launch`
5. Run `export AUTO_IN_CALIB=action` and `export GAZEBO_MODE=true`
6. Run `roslaunch thesis_camera intrinsic_camera_calibration.launch`
7. Run `roslaunch export AUTO_DT_CALIB=action` and `export AUTO_EX_CALIB=action`
7. Run `roslaunch thesis_core core.launch`
8. Run `rostopic pub -1 rostopic pub -1 /$robot_name$/core/decided_mode std_msgs/UInt8 "data: 2"` where $robot_name$ ist either `test_robot` or `other_robot`
