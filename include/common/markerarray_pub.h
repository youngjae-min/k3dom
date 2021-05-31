#pragma once

#include <geometry_msgs/Point.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <std_msgs/ColorRGBA.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseStamped.h>

#include <ros/node_handle.h>
#include <ros/publisher.h>

#include <cmath>
#include <string>

namespace dom {
    
    std_msgs::ColorRGBA heightMapColor(double h) {

        std_msgs::ColorRGBA color;
        color.a = 1.0;
        // blend over HSV-values (more colors)

        double s = 1.0;
        double v = 1.0;

        h -= floor(h);
        h *= 6;
        int i;
        double m, n, f;

        i = floor(h);
        f = h - i;
        if (!(i & 1))
            f = 1 - f; // if i is even
        m = v * (1 - s);
        n = v * (1 - s * f);

        switch (i) {
            case 6:
            case 0:
                color.r = v;
                color.g = n;
                color.b = m;
                break;
            case 1:
                color.r = n;
                color.g = v;
                color.b = m;
                break;
            case 2:
                color.r = m;
                color.g = v;
                color.b = n;
                break;
            case 3:
                color.r = m;
                color.g = n;
                color.b = v;
                break;
            case 4:
                color.r = n;
                color.g = m;
                color.b = v;
                break;
            case 5:
                color.r = v;
                color.g = m;
                color.b = n;
                break;
            default:
                color.r = 1;
                color.g = 0.5;
                color.b = 0.5;
                break;
        }
        return color;
    }

    class MarkerArrayPub {
    public:
        MarkerArrayPub(ros::NodeHandle nh, std::string map_topic, std::string traj_topic, float resolution,
                        std::string map_frame_id, std::string traj_frame_id) : nh(nh),
                                                                            map_msg(new visualization_msgs::MarkerArray),
                                                                            traj_msg(new nav_msgs::Path),
                                                                            map_topic(map_topic), traj_topic(traj_topic),
                                                                            cloud_updated_time(ros::Time::now()),
                                                                            resolution(resolution),
                                                                            markerarray_frame_id(map_frame_id),
                                                                            traj_frame_id(traj_frame_id) {
            pub_map = nh.advertise<visualization_msgs::MarkerArray>(map_topic, 1, true);
            pub_traj = nh.advertise<nav_msgs::Path>(traj_topic, 1, true);

            map_msg->markers.resize(1);
            int i = 0;
            map_msg->markers[i].header.frame_id = markerarray_frame_id;
            map_msg->markers[i].ns = "map";
            map_msg->markers[i].id = i;
            map_msg->markers[i].type = visualization_msgs::Marker::CUBE_LIST;
            map_msg->markers[i].scale.x = resolution * pow(2, i);
            map_msg->markers[i].scale.y = resolution * pow(2, i);
            map_msg->markers[i].scale.z = resolution * pow(2, i);
            std_msgs::ColorRGBA color;
            color.r = 0.0;
            color.g = 0.0;
            color.b = 1.0;
            color.a = 1.0;
            map_msg->markers[i].color = color;

            traj_msg->header.frame_id = traj_frame_id;
        }

        void insert_trajectory(const geometry_msgs::PoseStamped::ConstPtr& msg_p) {
            traj_msg->poses.push_back(*msg_p);
        }

        // for k3dom
        void insert_point3d_class(float x, float y, float z, float f_mass, float s_mass, float d_mass) {
            float total = s_mass + f_mass + d_mass;
            if (total <= 0.0f) {return;}

            geometry_msgs::Point center;
            
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;

            map_msg->markers[depth].points.push_back(center);

            std_msgs::ColorRGBA color;
            
            color.r = 0;
            color.g = s_mass / total;
            color.b = d_mass / total;
            color.a = 1;
            
            map_msg->markers[depth].colors.push_back(color);
        }
        // for ds-phd/mib
        void insert_point3d_class(float x, float y, float z, float speed, float speed_thresh) {
            geometry_msgs::Point center;
            
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;

            map_msg->markers[depth].points.push_back(center);

            std_msgs::ColorRGBA color;
            
            color.r = 0;
            color.b = std::min(speed / speed_thresh, 1.0f);
            color.g = 1.0 - color.b;
            color.a = 1.0;
            
            map_msg->markers[depth].colors.push_back(color);
        }

        void insert_point3d_height(float x, float y, float z, float min_val, float max_val, float val) {
            if (min_val >= max_val) {return;}
            
            geometry_msgs::Point center;
            
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;

            map_msg->markers[depth].points.push_back(center);

            
            double h = (1.0 - std::min(std::max((val - min_val) / (max_val - min_val), 0.0f), 1.0f)) * 0.8;
            map_msg->markers[depth].colors.push_back(heightMapColor(h));
            
        }

        void insert_point3d_var(float x, float y, float z, float min_val, float max_val, float val) {
            if (min_val >= max_val) {return;}
            
            geometry_msgs::Point center;
            
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;

            map_msg->markers[depth].points.push_back(center);

            std_msgs::ColorRGBA color;
            
            double h = (1.0 - std::min(std::max((val - min_val) / (max_val - min_val), 0.0f), 1.0f));

            color.r = h;
            color.g = h;
            color.b = h;
            color.a = 1;
            
            map_msg->markers[depth].colors.push_back(color);
            
        }

        void clear() {
            for (int i = 0; i < 1; ++i) {
                map_msg->markers[i].points.clear();
                map_msg->markers[i].colors.clear();
            }
        }

        void update_time(ros::Time t) {
            cloud_updated_time = t;
        }

        void publish() const {
            map_msg->markers[0].header.stamp = cloud_updated_time;
            pub_map.publish(*map_msg);
            //ros::spinOnce();
            traj_msg->header.stamp = cloud_updated_time;
            pub_traj.publish(*traj_msg);
        }

    private:
        ros::NodeHandle nh;
        ros::Publisher pub_map;
        ros::Publisher pub_traj;
        visualization_msgs::MarkerArray::Ptr map_msg;
        nav_msgs::Path::Ptr traj_msg;
        ros::Time cloud_updated_time;
        std::string markerarray_frame_id;
        std::string traj_frame_id;
        std::string map_topic;
        std::string traj_topic;
        float resolution;
    };

}
