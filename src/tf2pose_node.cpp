#include <ros/ros.h>

#include <geometry_msgs/PoseStamped.h>
#include <tf2_msgs/TFMessage.h>

class Subscribe_And_Publish
{
private:
    ros::Publisher pos_pub;
    ros::Subscriber tf_sub;
    ros::NodeHandle nh;

public:
    Subscribe_And_Publish() : nh("~")
    {
        tf_sub = nh.subscribe<tf2_msgs::TFMessage> ("/tf", 100, &Subscribe_And_Publish::callback, this);
        pos_pub = nh.advertise<geometry_msgs::PoseStamped>("/pose", 10);
    }

    void callback(const tf2_msgs::TFMessage::ConstPtr& msg)
    {
        for (const geometry_msgs::TransformStamped &tf_msg : msg->transforms)
        {
            if (tf_msg.header.frame_id == "world" && tf_msg.child_frame_id == "base_link")
            {
                geometry_msgs::PoseStamped pos_msg;

                pos_msg.header =tf_msg.header;
                pos_msg.pose.position.x = tf_msg.transform.translation.x;
                pos_msg.pose.position.y = tf_msg.transform.translation.y;
                pos_msg.pose.position.z = tf_msg.transform.translation.z;
                pos_msg.pose.orientation.x = tf_msg.transform.rotation.x;
                pos_msg.pose.orientation.y = tf_msg.transform.rotation.y;
                pos_msg.pose.orientation.z = tf_msg.transform.rotation.z;
                pos_msg.pose.orientation.w = tf_msg.transform.rotation.w;

                pos_pub.publish(pos_msg);
            }
        }
    }

};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "tf2pose_node");
    Subscribe_And_Publish SAPObject;

    ros::spin();

    return 0;
}