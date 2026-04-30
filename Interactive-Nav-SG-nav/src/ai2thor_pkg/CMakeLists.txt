cmake_minimum_required(VERSION 3.0.2)
project(ai2thor_pkg)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  cv_bridge
)

catkin_package()

#catkin_install_python(PROGRAMS
#  script/ai2thor_ros.py
#  script/thor_sensor.py
#  script/thor_controller.py
#  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
#)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)


