cmake_minimum_required(VERSION 3.0.2)
project(semantic_mapping_py_pkg)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  map_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2_ros
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/object_detection_visual_test.py
  scripts/object_detection_node.py
  scripts/room_attribute_node.py
  scripts/semantic_mapping_gt_replay.py
  scripts/semantic_mapping_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY scripts/semantic_mapping_py_pkg
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY config launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
