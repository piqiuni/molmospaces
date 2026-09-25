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
install(FILES scripts/physical_nav_runtime.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
# catkin's generated devel relay executes with lib/physical_nav on sys.path,
# not the source script directory. Provide the same helper in both layouts.
configure_file(scripts/physical_nav_runtime.py
  ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/physical_nav_runtime.py COPYONLY)
# One source of truth. Never install the source-tree compatibility symlinks:
# their targets would escape a relocated install prefix.
get_filename_component(PHYSICAL_NAV_RUNTIME "${CMAKE_CURRENT_SOURCE_DIR}/.runtime" REALPATH)
if(NOT EXISTS "${PHYSICAL_NAV_RUNTIME}/physical_protocol.py")
  message(FATAL_ERROR "physical_nav requires its sibling deployment addon")
endif()
install(DIRECTORY "${PHYSICAL_NAV_RUNTIME}/"
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/.runtime
  USE_SOURCE_PERMISSIONS
  FILES_MATCHING PATTERN "*.py" PATTERN "*.sh" PATTERN "*.yaml"
  PATTERN "*.launch" PATTERN "*.rviz"
  PATTERN "tests" EXCLUDE PATTERN "__pycache__" EXCLUDE
  PATTERN ".pytest_cache" EXCLUDE PATTERN "*.bak*" EXCLUDE)
install(DIRECTORY "${PHYSICAL_NAV_RUNTIME}/launch" "${PHYSICAL_NAV_RUNTIME}/config"
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
