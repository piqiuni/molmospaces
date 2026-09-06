#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>

ros::Publisher pub;

void cb(const geometry_msgs::Twist::ConstPtr& msg)
{
    geometry_msgs::TwistStamped out;
    out.header.stamp = ros::Time::now();
    out.twist = *msg;
    pub.publish(out);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "relay_node");
    ros::NodeHandle nh("~");

    std::string input_topic;
    std::string output_topic;
    nh.param<std::string>("input_topic", input_topic, "/cmd_vel");
    nh.param<std::string>("output_topic", output_topic, "/cmd_vel_stamped");

    ros::Subscriber sub = nh.subscribe(input_topic, 10, cb);
    pub = nh.advertise<geometry_msgs::TwistStamped>(output_topic, 10);
    ROS_INFO("velocity relay: %s -> %s", input_topic.c_str(), output_topic.c_str());

    ros::spin();
    return 0;
}
