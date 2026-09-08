cmake_minimum_required(VERSION 3.0.2)
project(physical_nav)
find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs nav_msgs geometry_msgs tf2_ros visualization_msgs)
catkin_package(CATKIN_DEPENDS rospy std_msgs sensor_msgs nav_msgs geometry_msgs tf2_ros visualization_msgs)
catkin_install_python(PROGRAMS
  scripts/physical_consistency_node.py
  scripts/physical_ros_gateway.py
  scripts/physical_sensor_ros_bridge.py
  scripts/physical_velocity_safety.py
  scripts/velocity_command_mux.py
  scripts/physical_interaction_policy_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(FILES scripts/physical_consistency.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
