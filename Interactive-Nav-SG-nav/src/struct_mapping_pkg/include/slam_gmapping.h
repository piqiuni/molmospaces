/*
 * slam_gmapping
 * Copyright (c) 2008, Willow Garage, Inc.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the names of Stanford University or Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived from
 *     this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/* Author: Brian Gerkey */

#include "ros/ros.h"
#include "sensor_msgs/LaserScan.h"
#include "sensor_msgs/PointCloud2.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Empty.h"
#include "nav_msgs/GetMap.h"
#include "tf/transform_listener.h"
#include "tf/transform_broadcaster.h"
#include "message_filters/subscriber.h"
#include "tf/message_filter.h"
#include "nav_msgs/Odometry.h"

#include "gmapping/gridfastslam/gridslamprocessor.h"
#include "gmapping/sensor/sensor_base/sensor.h"

#include <boost/thread.hpp>
#include <deque>
#include <vector>

class SlamGMapping
{
  public:
    SlamGMapping();
    SlamGMapping(ros::NodeHandle& nh, ros::NodeHandle& pnh);
    SlamGMapping(unsigned long int seed, unsigned long int max_duration_buffer);
    ~SlamGMapping();

    void init();
    void startLiveSlam();
    void startReplay(const std::string & bag_fname, std::string scan_topic);
    void publishTransform();
  
    void laserCallback(const sensor_msgs::LaserScan::ConstPtr& scan);
    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud);
    void odomCallback(const nav_msgs::Odometry::ConstPtr& odom);
    void resetCallback(const std_msgs::Empty::ConstPtr& msg);
    bool mapCallback(nav_msgs::GetMap::Request  &req,
                     nav_msgs::GetMap::Response &res);
    void publishLoop(double transform_publish_period);

  private:
    ros::NodeHandle node_;
    ros::Publisher entropy_publisher_;
    ros::Publisher sst_;
    ros::Publisher sstm_;
    ros::Publisher filtered_cloud_pub_;  // 发布高度滤波后的点云
    ros::ServiceServer ss_;
    ros::Subscriber reset_sub_;
    ros::Subscriber odom_sub_;
    tf::TransformListener tf_;
    message_filters::Subscriber<sensor_msgs::PointCloud2>* scan_filter_sub_;
    tf::MessageFilter<sensor_msgs::PointCloud2>* scan_filter_;
    tf::TransformBroadcaster* tfB_;

    GMapping::GridSlamProcessor* gsp_;
    GMapping::RangeSensor* gsp_laser_;
    // The angles in the laser, going from -x to x (adjustment is made to get the laser between
    // symmetrical bounds as that's what gmapping expects)
    std::vector<double> laser_angles_;
    // The pose, in the original laser frame, of the corresponding centered laser with z facing up
    tf::Stamped<tf::Pose> centered_laser_pose_;
    // Depending on the order of the elements in the scan and the orientation of the scan frame,
    // We might need to change the order of the scan
    bool do_reverse_range_;
    unsigned int gsp_laser_beam_count_;
    GMapping::OdometrySensor* gsp_odom_;

    bool got_first_scan_;

    bool got_map_;
    nav_msgs::GetMap::Response map_;

    ros::Duration map_update_interval_;
    tf::Transform map_to_odom_;
    boost::mutex map_to_odom_mutex_;
    boost::mutex map_mutex_;
    boost::mutex odom_time_mutex_;

    int laser_count_;
    int throttle_scans_;

    boost::thread* transform_thread_;

    std::string base_frame_;
    std::string laser_frame_;
    std::string map_frame_;
    std::string odom_frame_;
    std::string reset_topic_;
    std::string odom_topic_;

    void updateMap(const sensor_msgs::LaserScan& scan);
    bool getOdomPose(GMapping::OrientedPoint& gmap_pose, const ros::Time& t);
    bool initMapper(const sensor_msgs::LaserScan& scan);
    bool addScan(const sensor_msgs::LaserScan& scan, GMapping::OrientedPoint& gmap_pose);
    bool convertPointCloudToLaserScan(const sensor_msgs::PointCloud2::ConstPtr& cloud, sensor_msgs::LaserScan& scan);
    bool transformPointCloudToFrame(sensor_msgs::PointCloud2& cloud, const std::string& target_frame);
    double computePoseEntropy();
    void filterPointCloudByHeight(sensor_msgs::PointCloud2& cloud);
    void inflateObstacles(nav_msgs::OccupancyGrid& map);
    void applyLocalOverwrite(nav_msgs::OccupancyGrid& map,
                             const sensor_msgs::LaserScan& scan,
                             const GMapping::OrientedPoint& sensor_pose);
    void syncOverwriteLayerToMap(const nav_msgs::MapMetaData& info);
    void updateOverwriteLayer(const nav_msgs::OccupancyGrid& map,
                              const sensor_msgs::LaserScan& scan,
                              const GMapping::OrientedPoint& sensor_pose);
    void applyOverwriteLayer(nav_msgs::OccupancyGrid& map);
    bool mapInfoMatchesOverwriteLayer(const nav_msgs::MapMetaData& info) const;
    bool worldToMapInfo(const nav_msgs::MapMetaData& info, double wx, double wy, int& mx, int& my) const;
    void clearOverwriteCell(size_t idx);
    void clearOverwriteLayer();
    void resetOverwriteLayerIfPoseCorrectionJumped(const GMapping::OrientedPoint& map_pose,
                                                   const GMapping::OrientedPoint& odom_pose);
    double angleDiff(double a, double b) const;
    bool worldToMap(const nav_msgs::OccupancyGrid& map, double wx, double wy, int& mx, int& my) const;
    bool hasMatchedOdomStamp(const ros::Time& stamp, double* dt_sec = NULL);
    
    // Parameters used by GMapping
    double maxRange_;
    double maxUrange_;
    double maxrange_;
    double minimum_score_;
    double sigma_;
    int kernelSize_;
    double lstep_;
    double astep_;
    int iterations_;
    double lsigma_;
    double ogain_;
    int lskip_;
    double srr_;
    double srt_;
    double str_;
    double stt_;
    double linearUpdate_;
    double angularUpdate_;
    double temporalUpdate_;
    double resampleThreshold_;
    int particles_;
    double xmin_;
    double ymin_;
    double xmax_;
    double ymax_;
    double delta_;
    double occ_thresh_;
    double llsamplerange_;
    double llsamplestep_;
    double lasamplerange_;
    double lasamplestep_;
    bool use_odom_pose_for_mapping_;
    bool scan_matching_lock_yaw_to_odom_;
    double scan_matching_max_translation_correction_;
    
    // 点云高度滤波参数
    bool enable_height_filter_;
    double filter_height_center_;
    double filter_height_tolerance_;
    std::string filter_height_frame_;
    
    // 障碍物膨胀参数
    bool enable_obstacle_inflation_;
    double inflation_radius_;
    double inscribed_radius_;

    // 局部覆写参数（用最新扫描覆盖机器人附近区域）
    bool enable_local_overwrite_;
    double local_overwrite_radius_;
    bool local_overwrite_mark_occupied_;
    int local_overwrite_clear_confirm_count_;
    int local_overwrite_occupy_confirm_count_;
    double local_overwrite_ttl_sec_;
    bool local_overwrite_clear_static_occupied_;
    double local_overwrite_reset_on_correction_trans_delta_;
    double local_overwrite_reset_on_correction_rot_delta_;
    bool overwrite_layer_initialized_;
    bool overwrite_correction_anchor_initialized_;
    GMapping::OrientedPoint overwrite_correction_anchor_;
    nav_msgs::MapMetaData overwrite_layer_info_;
    std::vector<int8_t> overwrite_values_;
    std::vector<unsigned short> overwrite_free_counts_;
    std::vector<unsigned short> overwrite_occupied_counts_;
    std::vector<ros::Time> overwrite_last_seen_;

    // 时间同步保护参数
    bool enable_time_sync_guard_;
    double max_odom_cloud_time_diff_;
    double max_cloud_age_sec_;
    bool enforce_cloud_age_drop_;
    size_t odom_stamp_buffer_size_;
    std::deque<ros::Time> odom_stamp_buffer_;
    int scan_filter_queue_size_;
    
    ros::NodeHandle private_nh_;
    
    unsigned long int seed_;
    
    double transform_publish_period_;
    double tf_delay_;
};
