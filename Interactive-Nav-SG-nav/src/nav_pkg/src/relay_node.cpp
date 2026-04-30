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


    ros::Subscriber sub = nh.subscribe("/cmd_vel", 10, cb);
    pub = nh.advertise<geometry_msgs::TwistStamped>("/cmd_vel_stamped", 10);

    ros::spin();
    return 0;
}
