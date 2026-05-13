cmake_minimum_required(VERSION 3.0.2)
project(semantic_mapping_pkg)

# Set C++ standard (required for shared_mutex in log4cxx headers)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  roslib
  sensor_msgs
  pcl_conversions
  pcl_ros
  tf
  tf2
  tf2_ros
  visualization_msgs
)

find_package(OpenCV 3.0 QUIET)
find_package(Eigen3 REQUIRED)
# RapidJSON is header-only; some systems do not ship RapidJSONConfig.cmake.
find_package(RapidJSON QUIET)
if(NOT RapidJSON_FOUND)
  find_path(RAPIDJSON_INCLUDE_DIR rapidjson/document.h
    PATHS
      /usr/include
      /usr/local/include
      /opt/local/include
  )
  if(RAPIDJSON_INCLUDE_DIR)
    set(RapidJSON_FOUND TRUE)
    set(RapidJSON_INCLUDE_DIRS ${RAPIDJSON_INCLUDE_DIR})
  endif()
endif()
if(NOT RapidJSON_FOUND)
  message(FATAL_ERROR "RapidJSON headers not found. Please install rapidjson-dev.")
endif()
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED)

catkin_package(
  CATKIN_DEPENDS
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    visualization_msgs
    tf
    tf2
    tf2_ros
    pcl_conversions
    pcl_ros
  INCLUDE_DIRS
    include
  LIBRARIES
    object_semantic_mapping_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/include
  ${EIGEN3_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
  ${RapidJSON_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  /usr/local/include
  /usr/include
  include/semantic_mapping_pkg
)

add_definitions(${PCL_DEFINITIONS})

# ============================================
# Libraries
# ============================================
# Common utilities library
add_library(common_utils_lib
  src/common/utils.cpp
)
target_link_libraries(common_utils_lib
  ${catkin_LIBRARIES}
)

# Object semantic mapping library
add_library(object_semantic_mapping_lib
  src/object_semantic_map.cpp
  src/object_semantic_mapper.cpp
  src/scene_semantic_map.cpp
  src/scene_semantic_mapper.cpp
)
target_link_libraries(object_semantic_mapping_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  common_utils_lib
)

# ============================================
# Executables (Nodes)
# ============================================
add_executable(semantic_mapping_node
  src/semantic_mapping_node.cpp
)
target_link_libraries(semantic_mapping_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  object_semantic_mapping_lib
  common_utils_lib
)
add_dependencies(semantic_mapping_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# ============================================
# Python scripts
# ============================================
catkin_install_python(PROGRAMS
  script/blip_ros_node_mock.py
  script/blip_ros_node_client.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ============================================
# Shell scripts
# ============================================
install(PROGRAMS
  script/run_blip_ros_node.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# ============================================
# Install rules
# ============================================
install(TARGETS semantic_mapping_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h"
  PATTERN ".svn" EXCLUDE
)

install(DIRECTORY launch config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

