#include <ros/ros.h>
#include "explore_pkg/explore_manager.h"
#include <locale.h>

int main(int argc, char** argv)
{
  // 设置中文本地化支持
  setlocale(LC_CTYPE, "zh_CN.utf8");
  
  ros::init(argc, argv, "explore_manager_node");
  
  explore_pkg::ExploreManager manager;
  
  if (!manager.initialize()) {
    ROS_ERROR("Failed to initialize ExploreManager");
    return -1;
  }
  
  manager.start();
  
  ROS_INFO("ExploreManager node is running...");
  ros::spin();
  
  manager.stop();
  return 0;
}

