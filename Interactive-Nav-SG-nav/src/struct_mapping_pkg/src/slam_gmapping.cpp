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
/* Modified by: Charles DuHadway */


/**

@mainpage slam_gmapping

@htmlinclude manifest.html

@b slam_gmapping is a wrapper around the GMapping SLAM library. It reads laser
scans and odometry and computes a map. This map can be
written to a file using e.g.

  "rosrun map_server map_saver static_map:=dynamic_map"

<hr>

@section topic ROS topics

Subscribes to (name/type):
- @b "scan"/<a href="../../sensor_msgs/html/classstd__msgs_1_1LaserScan.html">sensor_msgs/LaserScan</a> : data from a laser range scanner 
- @b "/tf": odometry from the robot


Publishes to (name/type):
- @b "/tf"/tf/tfMessage: position relative to the map


@section services
 - @b "~dynamic_map" : returns the map


@section parameters ROS parameters

Reads the following parameters from the parameter server

Parameters used by our GMapping wrapper:

- @b "~throttle_scans": @b [int] throw away every nth laser scan
- @b "~base_frame": @b [string] the tf frame_id to use for the robot base pose
- @b "~map_frame": @b [string] the tf frame_id where the robot pose on the map is published
- @b "~odom_frame": @b [string] the tf frame_id from which odometry is read
- @b "~map_update_interval": @b [double] time in seconds between two recalculations of the map


Parameters used by GMapping itself:

Laser Parameters:
- @b "~/maxRange" @b [double] maximum range of the laser scans. Rays beyond this range get discarded completely. (default: maximum laser range minus 1 cm, as received in the the first LaserScan message)
- @b "~/maxUrange" @b [double] maximum range of the laser scanner that is used for map building (default: same as maxRange)
- @b "~/sigma" @b [double] standard deviation for the scan matching process (cell)
- @b "~/kernelSize" @b [int] search window for the scan matching process
- @b "~/lstep" @b [double] initial search step for scan matching (linear)
- @b "~/astep" @b [double] initial search step for scan matching (angular)
- @b "~/iterations" @b [int] number of refinement steps in the scan matching. The final "precision" for the match is lstep*2^(-iterations) or astep*2^(-iterations), respectively.
- @b "~/lsigma" @b [double] standard deviation for the scan matching process (single laser beam)
- @b "~/ogain" @b [double] gain for smoothing the likelihood
- @b "~/lskip" @b [int] take only every (n+1)th laser ray for computing a match (0 = take all rays)
- @b "~/minimumScore" @b [double] minimum score for considering the outcome of the scanmatching good. Can avoid 'jumping' pose estimates in large open spaces when using laser scanners with limited range (e.g. 5m). (0 = default. Scores go up to 600+, try 50 for example when experiencing 'jumping' estimate issues)

Motion Model Parameters (all standard deviations of a gaussian noise model)
- @b "~/srr" @b [double] linear noise component (x and y)
- @b "~/stt" @b [double] angular noise component (theta)
- @b "~/srt" @b [double] linear -> angular noise component
- @b "~/str" @b [double] angular -> linear noise component

Others:
- @b "~/linearUpdate" @b [double] the robot only processes new measurements if the robot has moved at least this many meters
- @b "~/angularUpdate" @b [double] the robot only processes new measurements if the robot has turned at least this many rads

- @b "~/resampleThreshold" @b [double] threshold at which the particles get resampled. Higher means more frequent resampling.
- @b "~/particles" @b [int] (fixed) number of particles. Each particle represents a possible trajectory that the robot has traveled

Likelihood sampling (used in scan matching)
- @b "~/llsamplerange" @b [double] linear range
- @b "~/lasamplerange" @b [double] linear step size
- @b "~/llsamplestep" @b [double] linear range
- @b "~/lasamplestep" @b [double] angular step size

Initial map dimensions and resolution:
- @b "~/xmin" @b [double] minimum x position in the map [m]
- @b "~/ymin" @b [double] minimum y position in the map [m]
- @b "~/xmax" @b [double] maximum x position in the map [m]
- @b "~/ymax" @b [double] maximum y position in the map [m]
- @b "~/delta" @b [double] size of one pixel [m]

*/



#include "slam_gmapping.h"

#include <iostream>
#include <cmath>
#include <limits>

#include <time.h>

#include "ros/ros.h"
#include "ros/console.h"
#include "nav_msgs/MapMetaData.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"

#include "gmapping/sensor/sensor_range/rangesensor.h"
#include "gmapping/sensor/sensor_odometry/odometrysensor.h"

#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>
#define foreach BOOST_FOREACH

// compute linear index for given map coords
#define MAP_IDX(sx, i, j) ((sx) * (j) + (i))

SlamGMapping::SlamGMapping():
  map_to_odom_(tf::Transform(tf::createQuaternionFromRPY( 0, 0, 0 ), tf::Point(0, 0, 0 ))),
  laser_count_(0), private_nh_("~"), scan_filter_sub_(NULL), scan_filter_(NULL), transform_thread_(NULL)
{
  seed_ = time(NULL);
  init();
}

SlamGMapping::SlamGMapping(ros::NodeHandle& nh, ros::NodeHandle& pnh):
  map_to_odom_(tf::Transform(tf::createQuaternionFromRPY( 0, 0, 0 ), tf::Point(0, 0, 0 ))),
  laser_count_(0),node_(nh), private_nh_(pnh), scan_filter_sub_(NULL), scan_filter_(NULL), transform_thread_(NULL)
{
  seed_ = time(NULL);
  init();
}

SlamGMapping::SlamGMapping(long unsigned int seed, long unsigned int max_duration_buffer):
  map_to_odom_(tf::Transform(tf::createQuaternionFromRPY( 0, 0, 0 ), tf::Point(0, 0, 0 ))),
  laser_count_(0), private_nh_("~"), scan_filter_sub_(NULL), scan_filter_(NULL), transform_thread_(NULL),
  seed_(seed), tf_(ros::Duration(1.0)) // 减少TF缓存时间到1秒，避免占用过多内存
{
  init();
}


void SlamGMapping::init()
{
  // log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME)->setLevel(ros::console::g_level_lookup[ros::console::levels::Debug]);

  // The library is pretty chatty
  //gsp_ = new GMapping::GridSlamProcessor(std::cerr);
  gsp_ = new GMapping::GridSlamProcessor();
  ROS_ASSERT(gsp_);

  tfB_ = new tf::TransformBroadcaster();
  ROS_ASSERT(tfB_);

  gsp_laser_ = NULL;
  gsp_odom_ = NULL;

  got_first_scan_ = false;
  got_map_ = false;
  

  
  // Parameters used by our GMapping wrapper
  if(!private_nh_.getParam("throttle_scans", throttle_scans_))
    throttle_scans_ = 1;
  if(!private_nh_.getParam("base_frame", base_frame_))
    base_frame_ = "tf_frame_base_link";
  if(!private_nh_.getParam("map_frame", map_frame_))
    map_frame_ = "tf_frame_map";
  if(!private_nh_.getParam("odom_frame", odom_frame_))
    odom_frame_ = "tf_frame_odom";
  if(!private_nh_.getParam("reset_topic", reset_topic_))
    reset_topic_ = "/nav_system/reset";
  if(!private_nh_.getParam("odom_topic", odom_topic_))
    odom_topic_ = "/odom";
  if(!private_nh_.getParam("scan_filter_queue_size", scan_filter_queue_size_))
    scan_filter_queue_size_ = 1;
  if (scan_filter_queue_size_ < 1)
    scan_filter_queue_size_ = 1;

  private_nh_.param("transform_publish_period", transform_publish_period_, 0.05);

  double tmp;
  if(!private_nh_.getParam("map_update_interval", tmp))
    tmp = 5.0;
  map_update_interval_.fromSec(tmp);
  
  // Parameters used by GMapping itself
  maxUrange_ = 0.0;  maxRange_ = 0.0; // preliminary default, will be set in initMapper()
  if(!private_nh_.getParam("minimumScore", minimum_score_))
    minimum_score_ = 0;
  if(!private_nh_.getParam("sigma", sigma_))
    sigma_ = 0.05;
  if(!private_nh_.getParam("kernelSize", kernelSize_))
    kernelSize_ = 1;
  if(!private_nh_.getParam("lstep", lstep_))
    lstep_ = 0.05;
  if(!private_nh_.getParam("astep", astep_))
    astep_ = 0.05;
  if(!private_nh_.getParam("iterations", iterations_))
    iterations_ = 5;
  if(!private_nh_.getParam("lsigma", lsigma_))
    lsigma_ = 0.075;
  if(!private_nh_.getParam("ogain", ogain_))
    ogain_ = 3.0;
  if(!private_nh_.getParam("lskip", lskip_))
    lskip_ = 0;
  if(!private_nh_.getParam("srr", srr_))
    srr_ = 0.1;
  if(!private_nh_.getParam("srt", srt_))
    srt_ = 0.2;
  if(!private_nh_.getParam("str", str_))
    str_ = 0.1;
  if(!private_nh_.getParam("stt", stt_))
    stt_ = 0.2;
  if(!private_nh_.getParam("linearUpdate", linearUpdate_))
    linearUpdate_ = 1.0;
  if(!private_nh_.getParam("angularUpdate", angularUpdate_))
    angularUpdate_ = 0.5;
  if(!private_nh_.getParam("temporalUpdate", temporalUpdate_))
    temporalUpdate_ = -1.0;
  if(!private_nh_.getParam("resampleThreshold", resampleThreshold_))
    resampleThreshold_ = 0.5;
  if(!private_nh_.getParam("particles", particles_))
    particles_ = 30;
  if(!private_nh_.getParam("xmin", xmin_))
    xmin_ = -100.0;
  if(!private_nh_.getParam("ymin", ymin_))
    ymin_ = -100.0;
  if(!private_nh_.getParam("xmax", xmax_))
    xmax_ = 100.0;
  if(!private_nh_.getParam("ymax", ymax_))
    ymax_ = 100.0;
  if(!private_nh_.getParam("delta", delta_))
    delta_ = 0.05;
  if(!private_nh_.getParam("occ_thresh", occ_thresh_))
    occ_thresh_ = 0.25;
  if(!private_nh_.getParam("llsamplerange", llsamplerange_))
    llsamplerange_ = 0.01;
  if(!private_nh_.getParam("llsamplestep", llsamplestep_))
    llsamplestep_ = 0.01;
  if(!private_nh_.getParam("lasamplerange", lasamplerange_))
    lasamplerange_ = 0.005;
  if(!private_nh_.getParam("lasamplestep", lasamplestep_))
    lasamplestep_ = 0.005;
    
  // 点云高度滤波参数
  if(!private_nh_.getParam("enable_height_filter", enable_height_filter_))
    enable_height_filter_ = false;
  if(!private_nh_.getParam("filter_height_center", filter_height_center_))
    filter_height_center_ = 0.6;  // 默认0米高度
  if(!private_nh_.getParam("filter_height_tolerance", filter_height_tolerance_))
    filter_height_tolerance_ = 0.5;  // 默认保留±0.5米范围
  if(!private_nh_.getParam("filter_height_frame", filter_height_frame_))
    filter_height_frame_ = base_frame_;
  
  // 障碍物膨胀参数
  if(!private_nh_.getParam("enable_obstacle_inflation", enable_obstacle_inflation_))
    enable_obstacle_inflation_ = false;
  if(!private_nh_.getParam("inflation_radius", inflation_radius_))
    inflation_radius_ = 0.3;  // 默认0.3米膨胀半径
  if(!private_nh_.getParam("inscribed_radius", inscribed_radius_))
    inscribed_radius_ = 0.2;  // 默认0.2米内切半径

  // 局部覆写参数
  if(!private_nh_.getParam("enable_local_overwrite", enable_local_overwrite_))
    enable_local_overwrite_ = false;
  if(!private_nh_.getParam("local_overwrite_radius", local_overwrite_radius_))
    local_overwrite_radius_ = 8.0;  // 默认仅覆写机器人附近8米范围
  if(!private_nh_.getParam("local_overwrite_mark_occupied", local_overwrite_mark_occupied_))
    local_overwrite_mark_occupied_ = true;

  // 时间同步保护参数
  if(!private_nh_.getParam("enable_time_sync_guard", enable_time_sync_guard_))
    enable_time_sync_guard_ = true;
  if(!private_nh_.getParam("max_odom_cloud_time_diff", max_odom_cloud_time_diff_))
    max_odom_cloud_time_diff_ = 0.03;  // 默认30ms
  if(!private_nh_.getParam("max_cloud_age_sec", max_cloud_age_sec_))
    max_cloud_age_sec_ = 0.20;  // 默认丢弃超过200ms的延迟点云
  if(!private_nh_.getParam("enforce_cloud_age_drop", enforce_cloud_age_drop_))
    enforce_cloud_age_drop_ = false;  // 默认仅告警，不直接丢弃
  int odom_stamp_buffer_size = 200;
  if(!private_nh_.getParam("odom_stamp_buffer_size", odom_stamp_buffer_size))
    odom_stamp_buffer_size = 200;
  odom_stamp_buffer_size_ = static_cast<size_t>(std::max(10, odom_stamp_buffer_size));
    
  if(!private_nh_.getParam("tf_delay", tf_delay_))
    tf_delay_ = transform_publish_period_;

}


void SlamGMapping::startLiveSlam()
{
  ROS_INFO("Starting SLAM GMapping...");
  entropy_publisher_ = private_nh_.advertise<std_msgs::Float64>("entropy", 1, true);
  sst_ = node_.advertise<nav_msgs::OccupancyGrid>("struct_mapping/occ_map", 1, true);
  sstm_ = node_.advertise<nav_msgs::MapMetaData>("map_metadata", 1, true);
  filtered_cloud_pub_ = node_.advertise<sensor_msgs::PointCloud2>("filtered_pointcloud", 1, true);
  ss_ = node_.advertiseService("dynamic_map", &SlamGMapping::mapCallback, this);
  reset_sub_ = node_.subscribe(reset_topic_, 1, &SlamGMapping::resetCallback, this);
  odom_sub_ = node_.subscribe(odom_topic_, 200, &SlamGMapping::odomCallback, this);
  scan_filter_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(
      node_, "registered_scan", scan_filter_queue_size_);
  scan_filter_ = new tf::MessageFilter<sensor_msgs::PointCloud2>(
      *scan_filter_sub_, tf_, odom_frame_, scan_filter_queue_size_);
  scan_filter_->setTolerance(ros::Duration(0.03));
  scan_filter_->registerCallback([this](auto msg){ pointCloudCallback(msg); });

  transform_thread_ = new boost::thread(boost::bind(&SlamGMapping::publishLoop, this, transform_publish_period_));
  ROS_INFO("SLAM GMapping started. Waiting for TF and scan data...");
  ROS_INFO("Subscribed to topic: /registered_scan (PointCloud2)");
  ROS_INFO("Target odom frame: %s", odom_frame_.c_str());
  ROS_INFO("Subscribed to odom topic: %s", odom_topic_.c_str());
  ROS_INFO("Scan filter queue size: %d", scan_filter_queue_size_);
  ROS_INFO("Publishing map to topic: /struct_mapping/occ_map");
  ROS_INFO("Subscribed to reset topic: %s", reset_topic_.c_str());
}

void SlamGMapping::startReplay(const std::string & bag_fname, std::string scan_topic)
{
  double transform_publish_period;
  ros::NodeHandle private_nh_("~");
  entropy_publisher_ = private_nh_.advertise<std_msgs::Float64>("entropy", 1, true);
  sst_ = node_.advertise<nav_msgs::OccupancyGrid>("map", 1, true);
  sstm_ = node_.advertise<nav_msgs::MapMetaData>("map_metadata", 1, true);
  ss_ = node_.advertiseService("dynamic_map", &SlamGMapping::mapCallback, this);
  reset_sub_ = node_.subscribe(reset_topic_, 1, &SlamGMapping::resetCallback, this);
  
  rosbag::Bag bag;
  bag.open(bag_fname, rosbag::bagmode::Read);
  
  std::vector<std::string> topics;
  topics.push_back(std::string("/tf"));
  topics.push_back(scan_topic);
  rosbag::View viewall(bag, rosbag::TopicQuery(topics));

  // Store up to 5 messages and there error message (if they cannot be processed right away)
  std::queue<std::pair<sensor_msgs::LaserScan::ConstPtr, std::string> > s_queue;
  foreach(rosbag::MessageInstance const m, viewall)
  {
    tf::tfMessage::ConstPtr cur_tf = m.instantiate<tf::tfMessage>();
    if (cur_tf != NULL) {
      for (size_t i = 0; i < cur_tf->transforms.size(); ++i)
      {
        geometry_msgs::TransformStamped transformStamped;
        tf::StampedTransform stampedTf;
        transformStamped = cur_tf->transforms[i];
        tf::transformStampedMsgToTF(transformStamped, stampedTf);
        tf_.setTransform(stampedTf);
      }
    }

    sensor_msgs::LaserScan::ConstPtr s = m.instantiate<sensor_msgs::LaserScan>();
    if (s != NULL) {
      if (!(ros::Time(s->header.stamp)).is_zero())
      {
        s_queue.push(std::make_pair(s, ""));
      }
      // Just like in live processing, only process the latest 5 scans
      if (s_queue.size() > 5) {
        ROS_WARN_STREAM("Dropping old scan: " << s_queue.front().second);
        s_queue.pop();
      }
      // ignoring un-timestamped tf data 
    }

    // Only process a scan if it has tf data
    while (!s_queue.empty())
    {
      try
      {
        tf::StampedTransform t;
        tf_.lookupTransform(s_queue.front().first->header.frame_id, odom_frame_, s_queue.front().first->header.stamp, t);
        this->laserCallback(s_queue.front().first);
        s_queue.pop();
      }
      // If tf does not have the data yet
      catch(tf2::TransformException& e)
      {
        // Store the error to display it if we cannot process the data after some time
        s_queue.front().second = std::string(e.what());
        break;
      }
    }
  }

  bag.close();
}

void SlamGMapping::resetCallback(const std_msgs::Empty::ConstPtr& msg)
{
  (void)msg;

  boost::mutex::scoped_lock map_lock(map_mutex_);
  boost::mutex::scoped_lock tf_lock(map_to_odom_mutex_);

  delete gsp_;
  gsp_ = new GMapping::GridSlamProcessor();
  ROS_ASSERT(gsp_);

  if (gsp_laser_) {
    delete gsp_laser_;
    gsp_laser_ = NULL;
  }
  if (gsp_odom_) {
    delete gsp_odom_;
    gsp_odom_ = NULL;
  }

  got_first_scan_ = false;
  got_map_ = false;
  laser_count_ = 0;
  map_ = nav_msgs::GetMap::Response();
  map_to_odom_ = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Point(0, 0, 0));

  nav_msgs::OccupancyGrid cleared_map;
  cleared_map.header.stamp = ros::Time::now();
  cleared_map.header.frame_id = tf_.resolve(map_frame_);
  sst_.publish(cleared_map);

  ROS_WARN("Received reset signal on %s, gmapping map/state cleared", reset_topic_.c_str());
}

void SlamGMapping::publishLoop(double transform_publish_period){
  if(transform_publish_period == 0)
    return;

  ros::Rate r(1.0 / transform_publish_period);
  while(ros::ok()){
    publishTransform();
    r.sleep();
  }
}

SlamGMapping::~SlamGMapping()
{
  if(transform_thread_){
    transform_thread_->join();
    delete transform_thread_;
  }

  delete gsp_;
  if(gsp_laser_)
    delete gsp_laser_;
  if(gsp_odom_)
    delete gsp_odom_;
  if (scan_filter_)
    delete scan_filter_;
  if (scan_filter_sub_)
    delete scan_filter_sub_;
}

bool
SlamGMapping::getOdomPose(GMapping::OrientedPoint& gmap_pose, const ros::Time& t)
{
  // Get the pose of the centered laser at the right time
  centered_laser_pose_.stamp_ = t;
  // Get the laser's pose that is centered
  tf::Stamped<tf::Transform> odom_pose;
  try
  {
    tf_.transformPose(odom_frame_, centered_laser_pose_, odom_pose);
  }
  catch(tf::TransformException e)
  {
    ROS_WARN("Failed to compute odom pose, skipping scan (%s)", e.what());
    return false;
  }
  double yaw = tf::getYaw(odom_pose.getRotation());

  gmap_pose = GMapping::OrientedPoint(odom_pose.getOrigin().x(),
                                      odom_pose.getOrigin().y(),
                                      yaw);
  return true;
}

bool
SlamGMapping::initMapper(const sensor_msgs::LaserScan& scan)
{
  laser_frame_ = scan.header.frame_id;
  // Get the laser's pose, relative to base.
  tf::Stamped<tf::Pose> ident;
  tf::Stamped<tf::Transform> laser_pose;
  ident.setIdentity();
  ident.frame_id_ = laser_frame_;
  ident.stamp_ = scan.header.stamp;
  try
  {
    tf_.transformPose(base_frame_, ident, laser_pose);
  }
  catch(tf::TransformException e)
  {
    ROS_WARN("Failed to compute laser pose, aborting initialization (%s)",
             e.what());
    ROS_WARN("Make sure TF transforms are being published for frames: %s -> %s", 
             laser_frame_.c_str(), base_frame_.c_str());
    return false;
  }

  // create a point 1m above the laser position and transform it into the laser-frame
  tf::Vector3 v;
  v.setValue(0, 0, 1 + laser_pose.getOrigin().z());
  tf::Stamped<tf::Vector3> up(v, scan.header.stamp,
                                      base_frame_);
  try
  {
    tf_.transformPoint(laser_frame_, up, up);
    ROS_DEBUG("Z-Axis in sensor frame: %.3f", up.z());
  }
  catch(tf::TransformException& e)
  {
    ROS_WARN("Unable to determine orientation of laser: %s",
             e.what());
    return false;
  }
  
  // gmapping doesnt take roll or pitch into account. So check for correct sensor alignment.
  if (fabs(fabs(up.z()) - 1) > 0.001)
  {
    ROS_WARN("Laser has to be mounted planar! Z-coordinate has to be 1 or -1, but gave: %.5f",
                 up.z());
    return false;
  }

  gsp_laser_beam_count_ = scan.ranges.size();

  double angle_center = (scan.angle_min + scan.angle_max)/2;

  if (up.z() > 0)
  {
    do_reverse_range_ = scan.angle_min > scan.angle_max;
    centered_laser_pose_ = tf::Stamped<tf::Pose>(tf::Transform(tf::createQuaternionFromRPY(0,0,angle_center),
                                                               tf::Vector3(0,0,0)), ros::Time::now(), laser_frame_);
    ROS_INFO("Laser is mounted upwards.");
  }
  else
  {
    do_reverse_range_ = scan.angle_min < scan.angle_max;
    centered_laser_pose_ = tf::Stamped<tf::Pose>(tf::Transform(tf::createQuaternionFromRPY(M_PI,0,-angle_center),
                                                               tf::Vector3(0,0,0)), ros::Time::now(), laser_frame_);
    ROS_INFO("Laser is mounted upside down.");
  }

  // Compute the angles of the laser from -x to x, basically symmetric and in increasing order
  laser_angles_.resize(scan.ranges.size());
  // Make sure angles are started so that they are centered
  double theta = - std::fabs(scan.angle_min - scan.angle_max)/2;
  for(unsigned int i=0; i<scan.ranges.size(); ++i)
  {
    laser_angles_[i]=theta;
    theta += std::fabs(scan.angle_increment);
  }

  ROS_DEBUG("Laser angles in laser-frame: min: %.3f max: %.3f inc: %.3f", scan.angle_min, scan.angle_max,
            scan.angle_increment);
  ROS_DEBUG("Laser angles in top-down centered laser-frame: min: %.3f max: %.3f inc: %.3f", laser_angles_.front(),
            laser_angles_.back(), std::fabs(scan.angle_increment));

  GMapping::OrientedPoint gmap_pose(0, 0, 0);

  // setting maxRange and maxUrange here so we can set a reasonable default
  ros::NodeHandle private_nh_("~");
  if(!private_nh_.getParam("maxRange", maxRange_))
    maxRange_ = scan.range_max - 0.01;
  if(!private_nh_.getParam("maxUrange", maxUrange_))
    maxUrange_ = maxRange_;

  // The laser must be called "FLASER".
  // We pass in the absolute value of the computed angle increment, on the
  // assumption that GMapping requires a positive angle increment.  If the
  // actual increment is negative, we'll swap the order of ranges before
  // feeding each scan to GMapping.
  gsp_laser_ = new GMapping::RangeSensor("FLASER",
                                         gsp_laser_beam_count_,
                                         fabs(scan.angle_increment),
                                         gmap_pose,
                                         0.0,
                                         maxRange_);
  ROS_ASSERT(gsp_laser_);

  GMapping::SensorMap smap;
  smap.insert(make_pair(gsp_laser_->getName(), gsp_laser_));
  gsp_->setSensorMap(smap);

  gsp_odom_ = new GMapping::OdometrySensor(odom_frame_);
  ROS_ASSERT(gsp_odom_);


  /// @todo Expose setting an initial pose
  GMapping::OrientedPoint initialPose;
  if(!getOdomPose(initialPose, scan.header.stamp))
  {
    ROS_WARN("Unable to determine inital pose of laser! Starting point will be set to zero.");
    initialPose = GMapping::OrientedPoint(0.0, 0.0, 0.0);
  }

  gsp_->setMatchingParameters(maxUrange_, maxRange_, sigma_,
                              kernelSize_, lstep_, astep_, iterations_,
                              lsigma_, ogain_, lskip_);

  gsp_->setMotionModelParameters(srr_, srt_, str_, stt_);
  gsp_->setUpdateDistances(linearUpdate_, angularUpdate_, resampleThreshold_);
  gsp_->setUpdatePeriod(temporalUpdate_);
  gsp_->setgenerateMap(false);
  gsp_->GridSlamProcessor::init(particles_, xmin_, ymin_, xmax_, ymax_,
                                delta_, initialPose);
  gsp_->setllsamplerange(llsamplerange_);
  gsp_->setllsamplestep(llsamplestep_);
  /// @todo Check these calls; in the gmapping gui, they use
  /// llsamplestep and llsamplerange intead of lasamplestep and
  /// lasamplerange.  It was probably a typo, but who knows.
  gsp_->setlasamplerange(lasamplerange_);
  gsp_->setlasamplestep(lasamplestep_);
  gsp_->setminimumScore(minimum_score_);

  // Call the sampling function once to set the seed.
  GMapping::sampleGaussian(1,seed_);

  ROS_INFO("Initialization complete");

  return true;
}

bool
SlamGMapping::addScan(const sensor_msgs::LaserScan& scan, GMapping::OrientedPoint& gmap_pose)
{
  if(!getOdomPose(gmap_pose, scan.header.stamp))
     return false;
  
  if(scan.ranges.size() != gsp_laser_beam_count_)
    return false;

  // GMapping wants an array of doubles...
  double* ranges_double = new double[scan.ranges.size()];
  // If the angle increment is negative, we have to invert the order of the readings.
  if (do_reverse_range_)
  {
    ROS_DEBUG("Inverting scan");
    int num_ranges = scan.ranges.size();
    for(int i=0; i < num_ranges; i++)
    {
      const int src_idx = num_ranges - i - 1;
      const bool observed = (src_idx >= 0 &&
                             src_idx < static_cast<int>(scan.intensities.size()) &&
                             scan.intensities[src_idx] > 0.0f);
      const float r = scan.ranges[src_idx];

      // 对于前向深度相机生成的伪360scan：
      // 未观测波束必须忽略（设为 > maxRange），避免被错误当成“远距离自由空间”。
      if (!observed || !std::isfinite(r) || r < scan.range_min)
        ranges_double[i] = maxRange_ + 1.0;
      else
        ranges_double[i] = static_cast<double>(r);
    }
  } else 
  {
    for(unsigned int i=0; i < scan.ranges.size(); i++)
    {
      const bool observed = (i < scan.intensities.size() &&
                             scan.intensities[i] > 0.0f);
      const float r = scan.ranges[i];

      if (!observed || !std::isfinite(r) || r < scan.range_min)
        ranges_double[i] = maxRange_ + 1.0;
      else
        ranges_double[i] = static_cast<double>(r);
    }
  }

  GMapping::RangeReading reading(scan.ranges.size(),
                                 ranges_double,
                                 gsp_laser_,
                                 scan.header.stamp.toSec());

  // ...but it deep copies them in RangeReading constructor, so we don't
  // need to keep our array around.
  delete[] ranges_double;

  reading.setPose(gmap_pose);

  /*
  ROS_DEBUG("scanpose (%.3f): %.3f %.3f %.3f\n",
            scan.header.stamp.toSec(),
            gmap_pose.x,
            gmap_pose.y,
            gmap_pose.theta);
            */
  ROS_DEBUG("processing scan");

  return gsp_->processScan(reading);
}

void
SlamGMapping::pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& cloud)
{
  if (enable_time_sync_guard_)
  {
    const ros::Time now = ros::Time::now();
    if (!cloud->header.stamp.isZero())
    {
      const double cloud_age_sec = (now - cloud->header.stamp).toSec();
      if (cloud_age_sec > max_cloud_age_sec_)
      {
        if (enforce_cloud_age_drop_)
        {
          ROS_WARN_THROTTLE(
              2.0,
              "Drop stale pointcloud: age=%.3fs exceeds max_cloud_age_sec=%.3fs",
              cloud_age_sec,
              max_cloud_age_sec_);
          return;
        }
        ROS_WARN_THROTTLE(
            2.0,
            "Stale pointcloud accepted (no drop): age=%.3fs exceeds max_cloud_age_sec=%.3fs",
            cloud_age_sec,
            max_cloud_age_sec_);
      }
    }

    double odom_dt_sec = 0.0;
    if (!hasMatchedOdomStamp(cloud->header.stamp, &odom_dt_sec))
    {
      ROS_WARN_THROTTLE(
          2.0,
          "Drop unsynced pointcloud: nearest odom dt=%.4fs exceeds threshold=%.4fs",
          odom_dt_sec,
          max_odom_cloud_time_diff_);
      return;
    }
  }

  laser_count_++;
  if ((laser_count_ % throttle_scans_) != 0)
    return;

  static ros::Time last_map_update(0,0);
  
  // 输出调试信息：接收到点云数据
  if(laser_count_ % 50 == 0)  // 每50帧输出一次
    ROS_INFO("Received point cloud data, processed %d scans so far", laser_count_ / throttle_scans_);

  // 创建点云副本用于滤波（避免修改原始消息）
  sensor_msgs::PointCloud2 filtered_cloud = *cloud;
  
  // 应用高度滤波
  filterPointCloudByHeight(filtered_cloud);
  
  // 发布滤波后的点云
  if (filtered_cloud_pub_.getNumSubscribers() > 0)
  {
    filtered_cloud_pub_.publish(filtered_cloud);
  }

  // Convert PointCloud2 to LaserScan
  sensor_msgs::LaserScan scan;
  sensor_msgs::PointCloud2::ConstPtr filtered_cloud_ptr = boost::make_shared<sensor_msgs::PointCloud2>(filtered_cloud);
  if(!convertPointCloudToLaserScan(filtered_cloud_ptr, scan))
  {
    ROS_WARN("Failed to convert point cloud to laser scan");
    return;
  }

  // We can't initialize the mapper until we've got the first scan
  if(!got_first_scan_)
  {
    ROS_INFO("Received first scan, initializing mapper...");
    if(!initMapper(scan))
    {
      ROS_WARN("Failed to initialize mapper, waiting for next scan...");
      return;
    }
    got_first_scan_ = true;
    ROS_INFO("Mapper initialized successfully!");
  }

  GMapping::OrientedPoint odom_pose;

  if(addScan(scan, odom_pose))
  {
    ROS_DEBUG("scan processed");

    GMapping::OrientedPoint mpose = gsp_->getParticles()[gsp_->getBestParticleIndex()].pose;
    ROS_DEBUG("new best pose: %.3f %.3f %.3f", mpose.x, mpose.y, mpose.theta);
    ROS_DEBUG("odom pose: %.3f %.3f %.3f", odom_pose.x, odom_pose.y, odom_pose.theta);
    ROS_DEBUG("correction: %.3f %.3f %.3f", mpose.x - odom_pose.x, mpose.y - odom_pose.y, mpose.theta - odom_pose.theta);

    // 输出位姿信息
    if(laser_count_ % 50 == 0)
      ROS_INFO("Current pose: x=%.2f y=%.2f theta=%.2f", mpose.x, mpose.y, mpose.theta);

    tf::Transform laser_to_map = tf::Transform(tf::createQuaternionFromRPY(0, 0, mpose.theta), tf::Vector3(mpose.x, mpose.y, 0.0));
    tf::Transform odom_to_laser = tf::Transform(tf::createQuaternionFromRPY(0, 0, odom_pose.theta), tf::Vector3(odom_pose.x, odom_pose.y, 0.0));

    // map_to_odom_mutex_.lock();
    // map_to_odom_ = (odom_to_laser * laser_to_map).inverse();
    // map_to_odom_mutex_.unlock();

    if(!got_map_ || (cloud->header.stamp - last_map_update) > map_update_interval_)
    {
      updateMap(scan);
      last_map_update = cloud->header.stamp;
      ROS_INFO("Map updated at time %.2f", cloud->header.stamp.toSec());
    }
  } else {
    ROS_DEBUG("cannot process scan");
    ROS_WARN_THROTTLE(5.0, "Cannot process scan - check TF and odometry data");
  }
}

void
SlamGMapping::laserCallback(const sensor_msgs::LaserScan::ConstPtr& scan)
{
  laser_count_++;
  if ((laser_count_ % throttle_scans_) != 0)
    return;

  static ros::Time last_map_update(0,0);

  // We can't initialize the mapper until we've got the first scan
  if(!got_first_scan_)
  {
    if(!initMapper(*scan))
      return;
    got_first_scan_ = true;
  }

  GMapping::OrientedPoint odom_pose;

  if(addScan(*scan, odom_pose))
  {
    ROS_DEBUG("scan processed");

    GMapping::OrientedPoint mpose = gsp_->getParticles()[gsp_->getBestParticleIndex()].pose;
    ROS_DEBUG("new best pose: %.3f %.3f %.3f", mpose.x, mpose.y, mpose.theta);
    ROS_DEBUG("odom pose: %.3f %.3f %.3f", odom_pose.x, odom_pose.y, odom_pose.theta);
    ROS_DEBUG("correction: %.3f %.3f %.3f", mpose.x - odom_pose.x, mpose.y - odom_pose.y, mpose.theta - odom_pose.theta);

    tf::Transform laser_to_map = tf::Transform(tf::createQuaternionFromRPY(0, 0, mpose.theta), tf::Vector3(mpose.x, mpose.y, 0.0));
    tf::Transform odom_to_laser = tf::Transform(tf::createQuaternionFromRPY(0, 0, odom_pose.theta), tf::Vector3(odom_pose.x, odom_pose.y, 0.0));

    // map_to_odom_mutex_.lock();
    // map_to_odom_ = (odom_to_laser * laser_to_map).inverse();
    // map_to_odom_mutex_.unlock();

    if(!got_map_ || (scan->header.stamp - last_map_update) > map_update_interval_)
    {
      updateMap(*scan);
      last_map_update = scan->header.stamp;
      ROS_DEBUG("Updated the map");
    }
  } else
    ROS_DEBUG("cannot process scan");
}

void SlamGMapping::odomCallback(const nav_msgs::Odometry::ConstPtr& odom)
{
  boost::mutex::scoped_lock lock(odom_time_mutex_);
  odom_stamp_buffer_.push_back(odom->header.stamp);
  while (odom_stamp_buffer_.size() > odom_stamp_buffer_size_)
    odom_stamp_buffer_.pop_front();
}

double
SlamGMapping::computePoseEntropy()
{
  double weight_total=0.0;
  for(std::vector<GMapping::GridSlamProcessor::Particle>::const_iterator it = gsp_->getParticles().begin();
      it != gsp_->getParticles().end();
      ++it)
  {
    weight_total += it->weight;
  }
  double entropy = 0.0;
  for(std::vector<GMapping::GridSlamProcessor::Particle>::const_iterator it = gsp_->getParticles().begin();
      it != gsp_->getParticles().end();
      ++it)
  {
    if(it->weight/weight_total > 0.0)
      entropy += it->weight/weight_total * log(it->weight/weight_total);
  }
  return -entropy;
}

void
SlamGMapping::updateMap(const sensor_msgs::LaserScan& scan)
{
  ROS_DEBUG("Update map");
  boost::mutex::scoped_lock map_lock (map_mutex_);
  GMapping::ScanMatcher matcher;

  matcher.setLaserParameters(scan.ranges.size(), &(laser_angles_[0]),
                             gsp_laser_->getPose());

  matcher.setlaserMaxRange(maxRange_);
  matcher.setusableRange(maxUrange_);
  matcher.setgenerateMap(true);

  GMapping::GridSlamProcessor::Particle best =
          gsp_->getParticles()[gsp_->getBestParticleIndex()];
  std_msgs::Float64 entropy;
  entropy.data = computePoseEntropy();
  if(entropy.data > 0.0)
    entropy_publisher_.publish(entropy);

  if(!got_map_) {
    map_.map.info.resolution = delta_;
    map_.map.info.origin.position.x = 0.0;
    map_.map.info.origin.position.y = 0.0;
    map_.map.info.origin.position.z = 0.0;
    map_.map.info.origin.orientation.x = 0.0;
    map_.map.info.origin.orientation.y = 0.0;
    map_.map.info.origin.orientation.z = 0.0;
    map_.map.info.origin.orientation.w = 1.0;
  } 

  GMapping::Point center;
  center.x=(xmin_ + xmax_) / 2.0;
  center.y=(ymin_ + ymax_) / 2.0;

  GMapping::ScanMatcherMap smap(center, xmin_, ymin_, xmax_, ymax_, 
                                delta_);

  ROS_DEBUG("Trajectory tree:");
  for(GMapping::GridSlamProcessor::TNode* n = best.node;
      n;
      n = n->parent)
  {
    ROS_DEBUG("  %.3f %.3f %.3f",
              n->pose.x,
              n->pose.y,
              n->pose.theta);
    if(!n->reading)
    {
      ROS_DEBUG("Reading is NULL");
      continue;
    }
    matcher.invalidateActiveArea();
    matcher.computeActiveArea(smap, n->pose, &((*n->reading)[0]));
    matcher.registerScan(smap, n->pose, &((*n->reading)[0]));
  }

  // the map may have expanded, so resize ros message as well
  if(map_.map.info.width != (unsigned int) smap.getMapSizeX() || map_.map.info.height != (unsigned int) smap.getMapSizeY()) {

    // NOTE: The results of ScanMatcherMap::getSize() are different from the parameters given to the constructor
    //       so we must obtain the bounding box in a different way
    GMapping::Point wmin = smap.map2world(GMapping::IntPoint(0, 0));
    GMapping::Point wmax = smap.map2world(GMapping::IntPoint(smap.getMapSizeX(), smap.getMapSizeY()));
    xmin_ = wmin.x; ymin_ = wmin.y;
    xmax_ = wmax.x; ymax_ = wmax.y;
    
    ROS_DEBUG("map size is now %dx%d pixels (%f,%f)-(%f, %f)", smap.getMapSizeX(), smap.getMapSizeY(),
              xmin_, ymin_, xmax_, ymax_);

    map_.map.info.width = smap.getMapSizeX();
    map_.map.info.height = smap.getMapSizeY();
    map_.map.info.origin.position.x = xmin_;
    map_.map.info.origin.position.y = ymin_;
    map_.map.data.resize(map_.map.info.width * map_.map.info.height);

    ROS_DEBUG("map origin: (%f, %f)", map_.map.info.origin.position.x, map_.map.info.origin.position.y);
  }

  for(int x=0; x < smap.getMapSizeX(); x++)
  {
    for(int y=0; y < smap.getMapSizeY(); y++)
    {
      /// @todo Sort out the unknown vs. free vs. obstacle thresholding
      GMapping::IntPoint p(x, y);
      double occ=smap.cell(p);
      assert(occ <= 1.0);
      if(occ < 0)
        map_.map.data[MAP_IDX(map_.map.info.width, x, y)] = -1;
      else if(occ > occ_thresh_)
      {
        //map_.map.data[MAP_IDX(map_.map.info.width, x, y)] = (int)round(occ*100.0);
        map_.map.data[MAP_IDX(map_.map.info.width, x, y)] = 100;
      }
      else
        map_.map.data[MAP_IDX(map_.map.info.width, x, y)] = 0;
    }
  }
  got_map_ = true;

  if (enable_local_overwrite_)
  {
    applyLocalOverwrite(map_.map, scan, best.pose);
  }
  
  // 应用障碍物膨胀
  if (enable_obstacle_inflation_)
  {
    inflateObstacles(map_.map);
  }

  //make sure to set the header information on the map
  map_.map.header.stamp = ros::Time::now();
  map_.map.header.frame_id = tf_.resolve( map_frame_ );

  sst_.publish(map_.map);
  sstm_.publish(map_.map.info);
}

bool 
SlamGMapping::mapCallback(nav_msgs::GetMap::Request  &req,
                          nav_msgs::GetMap::Response &res)
{
  boost::mutex::scoped_lock map_lock (map_mutex_);
  if(got_map_ && map_.map.info.width && map_.map.info.height)
  {
    res = map_;
    return true;
  }
  else
    return false;
}

void SlamGMapping::publishTransform()
{
  map_to_odom_mutex_.lock();
  ros::Time tf_expiration = ros::Time::now() + ros::Duration(tf_delay_);
  //ros::Time tf_expiration = ros::Time::now();
  tfB_->sendTransform( tf::StampedTransform (map_to_odom_, tf_expiration, map_frame_, odom_frame_));
  map_to_odom_mutex_.unlock();
}

bool SlamGMapping::convertPointCloudToLaserScan(const sensor_msgs::PointCloud2::ConstPtr& cloud, sensor_msgs::LaserScan& scan)
{
  // Set basic scan parameters
  scan.header = cloud->header;
  scan.header.frame_id = cloud->header.frame_id;
  
  // Set scan parameters (these should be configurable)
  scan.angle_min = -M_PI;
  scan.angle_max = M_PI;
  scan.angle_increment = M_PI / 180.0; // 1 degree resolution
  scan.time_increment = 0.0;
  scan.scan_time = 0.1; // 10 Hz
  scan.range_min = 0.1;
  scan.range_max = 30.0;
  
  // Calculate number of beams
  int num_beams = (scan.angle_max - scan.angle_min) / scan.angle_increment + 1;
  scan.ranges.resize(num_beams);
  scan.intensities.resize(num_beams);
  
  // Initialize all ranges to max range (unknown)
  std::fill(scan.ranges.begin(), scan.ranges.end(), scan.range_max);
  std::fill(scan.intensities.begin(), scan.intensities.end(), 0.0);
  
  // Convert point cloud to laser scan
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*cloud, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*cloud, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*cloud, "z");
  
  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
  {
    // Skip invalid points
    if (std::isnan(*iter_x) || std::isnan(*iter_y) || std::isnan(*iter_z))
      continue;
      
    // Calculate range and angle
    double range = sqrt(*iter_x * *iter_x + *iter_y * *iter_y);
    double angle = atan2(*iter_y, *iter_x);
    
    // Skip points outside range
    if (range < scan.range_min || range > scan.range_max)
      continue;
    
    // Find corresponding beam index
    int beam_index = (angle - scan.angle_min) / scan.angle_increment;
    
    // Clamp beam index to valid range
    if (beam_index < 0) beam_index = 0;
    if (beam_index >= num_beams) beam_index = num_beams - 1;
    
    // Update range if this is closer
    if (range < scan.ranges[beam_index])
    {
      scan.ranges[beam_index] = range;
      scan.intensities[beam_index] = 1.0; // Simple intensity
    }
  }
  
  return true;
}

void SlamGMapping::filterPointCloudByHeight(sensor_msgs::PointCloud2& cloud)
{
  if (!enable_height_filter_)
    return;
  
  // 计算高度范围
  double height_min = filter_height_center_ - filter_height_tolerance_;
  double height_max = filter_height_center_ + filter_height_tolerance_;
  const std::string source_frame = cloud.header.frame_id;

  // Height threshold should be interpreted in a stable robot frame (e.g. base_link).
  // We only use transformed z for filtering, while keeping original point coords for scan conversion.
  bool use_transformed_height = false;
  tf::StampedTransform source_to_height_tf;
  if (!filter_height_frame_.empty() && source_frame != filter_height_frame_)
  {
    try
    {
      tf_.lookupTransform(
          filter_height_frame_,
          source_frame,
          cloud.header.stamp,
          source_to_height_tf);
      use_transformed_height = true;
    }
    catch (tf::TransformException& e)
    {
      ROS_WARN_THROTTLE(
          5.0,
          "Height filter TF lookup failed (%s -> %s): %s. Falling back to source-frame z.",
          source_frame.c_str(),
          filter_height_frame_.c_str(),
          e.what());
    }
  }
  
  // 原地过滤：使用双指针法，只遍历一次
  sensor_msgs::PointCloud2Iterator<float> iter_x_read(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y_read(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z_read(cloud, "z");
  
  sensor_msgs::PointCloud2Iterator<float> iter_x_write(cloud, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y_write(cloud, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z_write(cloud, "z");
  
  size_t original_points = cloud.width * cloud.height;
  size_t filtered_points = 0;
  
  // 单次遍历，原地移动有效点到前面
  for (size_t i = 0; i < original_points; ++i, ++iter_x_read, ++iter_y_read, ++iter_z_read)
  {
    float x = *iter_x_read;
    float y = *iter_y_read;
    float z = *iter_z_read;
    
    // 跳过无效点
    if (std::isnan(x) || std::isnan(y) || std::isnan(z))
      continue;
    
    double z_for_filter = z;
    if (use_transformed_height)
    {
      tf::Vector3 p_src(x, y, z);
      tf::Vector3 p_height = source_to_height_tf * p_src;
      z_for_filter = p_height.z();
    }

    // 检查高度是否在指定范围内（可在目标高度坐标系中判定）
    if (z_for_filter >= height_min && z_for_filter <= height_max)
    {
      // 只有当写位置不同于读位置时才需要复制
      if (filtered_points != i)
      {
        *iter_x_write = x;
        *iter_y_write = y;
        *iter_z_write = z;
      }
      
      ++iter_x_write;
      ++iter_y_write;
      ++iter_z_write;
      ++filtered_points;
    }
  }
  
  // 如果没有点通过滤波，记录警告并返回
  if (filtered_points == 0)
  {
    ROS_WARN_THROTTLE(5.0, "Height filter removed all points! Check filter parameters.");
    cloud.width = 0;
    cloud.data.clear();
    return;
  }
  
  // 更新点云元数据（不需要重新分配内存，只是调整大小）
  cloud.width = filtered_points;
  cloud.height = 1;
  cloud.is_dense = false;
  cloud.row_step = cloud.point_step * cloud.width;
  cloud.data.resize(cloud.row_step);
  
  if (laser_count_ % 100 == 0)  // 降低日志频率
  {
    ROS_DEBUG("Height filter: kept %zu/%zu points (%.1f%%)", 
              filtered_points, original_points, 
              100.0 * filtered_points / original_points);
  }
}

void SlamGMapping::inflateObstacles(nav_msgs::OccupancyGrid& map)
{
  if (!enable_obstacle_inflation_)
    return;
  
  // 计算膨胀半径对应的栅格数
  int inflation_cells = static_cast<int>(std::ceil(inflation_radius_ / map.info.resolution));
  int inscribed_cells = static_cast<int>(std::ceil(inscribed_radius_ / map.info.resolution));
  
  if (inflation_cells <= 0)
  {
    ROS_WARN_ONCE("Inflation radius too small, no inflation applied");
    return;
  }
  
  // 创建地图副本以避免在迭代中修改原始数据
  std::vector<int8_t> original_map = map.data;
  int width = map.info.width;
  int height = map.info.height;
  
  ROS_DEBUG("Inflating obstacles with radius: %.2f m (%d cells)", inflation_radius_, inflation_cells);
  
  // 遍历原始地图，找到所有障碍物
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      int index = MAP_IDX(width, x, y);
      
      // 如果是障碍物（占据值 >= 100）
      if (original_map[index] >= 100)
      {
        // 在膨胀半径内的所有栅格都标记为障碍物
        for (int dy = -inflation_cells; dy <= inflation_cells; ++dy)
        {
          for (int dx = -inflation_cells; dx <= inflation_cells; ++dx)
          {
            int nx = x + dx;
            int ny = y + dy;
            
            // 检查边界
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
              float dist = std::sqrt(dx * dx + dy * dy) * map.info.resolution;
              int nindex = MAP_IDX(width, nx, ny);
              
              // 如果当前格子是未知区域（-1），不进行膨胀
              if (original_map[nindex] == -1)
                continue;
              
              // 在内切半径内，设置为完全占据（100）
              if (dist <= inscribed_radius_)
              {
                map.data[nindex] = 100;
              }
              // 在膨胀半径内但不在内切半径内，设置为部分占据
              else if (dist <= inflation_radius_)
              {
                // 如果该格子还未被标记为障碍物，则标记
                // 使用渐变值：距离越远，占据值越小
                int8_t current_value = map.data[nindex];
                
                // 计算基于距离的占据值（线性衰减）
                // 从内切半径的100衰减到膨胀半径的50
                float ratio = (inflation_radius_ - dist) / (inflation_radius_ - inscribed_radius_);
                int8_t inflation_value = static_cast<int8_t>(50 + ratio * 50);
                
                // 保留较高的占据值（避免被较低的值覆盖）
                if (inflation_value > current_value)
                {
                  map.data[nindex] = inflation_value;
                }
              }
            }
          }
        }
      }
    }
  }
  
  ROS_DEBUG_THROTTLE(5.0, "Obstacle inflation completed");
}

bool SlamGMapping::worldToMap(const nav_msgs::OccupancyGrid& map, double wx, double wy, int& mx, int& my) const
{
  const double origin_x = map.info.origin.position.x;
  const double origin_y = map.info.origin.position.y;
  const double resolution = map.info.resolution;

  if (wx < origin_x || wy < origin_y)
    return false;

  mx = static_cast<int>((wx - origin_x) / resolution);
  my = static_cast<int>((wy - origin_y) / resolution);

  if (mx < 0 || my < 0 ||
      mx >= static_cast<int>(map.info.width) ||
      my >= static_cast<int>(map.info.height))
    return false;

  return true;
}

bool SlamGMapping::hasMatchedOdomStamp(const ros::Time& stamp, double* dt_sec)
{
  boost::mutex::scoped_lock lock(odom_time_mutex_);
  if (odom_stamp_buffer_.empty())
  {
    if (dt_sec)
      *dt_sec = std::numeric_limits<double>::infinity();
    return false;
  }

  double best_dt = std::numeric_limits<double>::infinity();
  for (const ros::Time& t : odom_stamp_buffer_)
  {
    const double dt = std::fabs((t - stamp).toSec());
    if (dt < best_dt)
      best_dt = dt;
  }

  if (dt_sec)
    *dt_sec = best_dt;
  return best_dt <= max_odom_cloud_time_diff_;
}

void SlamGMapping::applyLocalOverwrite(nav_msgs::OccupancyGrid& map,
                                       const sensor_msgs::LaserScan& scan,
                                       const GMapping::OrientedPoint& sensor_pose)
{
  if (map.info.width == 0 || map.info.height == 0 || scan.ranges.empty())
    return;

  const double resolution = map.info.resolution;
  const double usable_radius = std::max(0.0, std::min(local_overwrite_radius_, maxUrange_));
  if (usable_radius <= 0.0)
    return;

  int sensor_mx = 0;
  int sensor_my = 0;
  if (!worldToMap(map, sensor_pose.x, sensor_pose.y, sensor_mx, sensor_my))
  {
    ROS_WARN_THROTTLE(5.0, "Local overwrite skipped: sensor pose is outside map bounds");
    return;
  }

  size_t updated_free_cells = 0;
  size_t updated_occupied_cells = 0;

  for (size_t i = 0; i < scan.ranges.size(); ++i)
  {
    // 对于由前向深度相机生成的“伪360 LaserScan”，仅处理真实观测到的波束。
    // 未观测方向（如后方）会保持 intensity=0，必须跳过，避免被错误清空。
    if (i >= scan.intensities.size() || scan.intensities[i] <= 0.0)
      continue;

    const double beam_angle = sensor_pose.theta + scan.angle_min + static_cast<double>(i) * scan.angle_increment;
    const double measured_range = scan.ranges[i];

    if (!std::isfinite(measured_range) || measured_range < scan.range_min)
      continue;

    const bool valid_hit = measured_range < scan.range_max;
    const double trace_range = std::min(measured_range, usable_radius);
    if (trace_range <= 0.0)
      continue;

    const int step_count = std::max(1, static_cast<int>(trace_range / resolution));
    const double cos_theta = std::cos(beam_angle);
    const double sin_theta = std::sin(beam_angle);

    for (int step = 1; step <= step_count; ++step)
    {
      const double dist = std::min(trace_range, static_cast<double>(step) * resolution);
      const double wx = sensor_pose.x + dist * cos_theta;
      const double wy = sensor_pose.y + dist * sin_theta;

      int mx = 0;
      int my = 0;
      if (!worldToMap(map, wx, wy, mx, my))
        break;

      const int idx = MAP_IDX(map.info.width, mx, my);
      if (map.data[idx] != 0)
      {
        map.data[idx] = 0;
        ++updated_free_cells;
      }
    }

    if (valid_hit && measured_range <= usable_radius && local_overwrite_mark_occupied_)
    {
      const double hit_wx = sensor_pose.x + measured_range * cos_theta;
      const double hit_wy = sensor_pose.y + measured_range * sin_theta;
      int hit_mx = 0;
      int hit_my = 0;
      if (worldToMap(map, hit_wx, hit_wy, hit_mx, hit_my))
      {
        const int hit_idx = MAP_IDX(map.info.width, hit_mx, hit_my);
        if (map.data[hit_idx] != 100)
        {
          map.data[hit_idx] = 100;
          ++updated_occupied_cells;
        }
      }
    }
  }

  ROS_DEBUG_THROTTLE(2.0,
                     "Local overwrite updated map: free_cells=%zu, occupied_cells=%zu, radius=%.2f",
                     updated_free_cells,
                     updated_occupied_cells,
                     usable_radius);
}
