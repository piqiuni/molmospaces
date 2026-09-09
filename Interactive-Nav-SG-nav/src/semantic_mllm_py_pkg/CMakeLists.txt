cmake_minimum_required(VERSION 3.0.2)
project(semantic_mllm_py_pkg)

find_package(catkin REQUIRED COMPONENTS rospy std_msgs)

catkin_python_setup()
catkin_package()

install(DIRECTORY scripts/semantic_mllm_py_pkg
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
