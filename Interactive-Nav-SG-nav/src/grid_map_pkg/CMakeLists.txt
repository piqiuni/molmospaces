cmake_minimum_required(VERSION 3.10)
project(grid_map_pkg)

## C++17标准（required by log4cxx headers in current toolchain）
add_compile_options(-std=c++17)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  nav_msgs
  geometry_msgs
  tf
)

## System dependencies
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES grid_map_pkg
  CATKIN_DEPENDS roscpp rospy std_msgs sensor_msgs nav_msgs geometry_msgs tf
  DEPENDS EIGEN3 OpenCV
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

## 地图库
add_library(grid_map_lib
  src/grid_map.cpp
  src/grid_mapper.cpp
)
target_link_libraries(grid_map_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

## 地图节点
add_executable(grid_mapper_node src/grid_map_node.cpp)
target_link_libraries(grid_mapper_node
  grid_map_lib
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
)

#############
## Install ##
#############

install(TARGETS grid_map_lib grid_mapper_node
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
