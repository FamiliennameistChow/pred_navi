 /**************************************************************************
 * record_trj.cpp
 * 
 * Author： Born Chow
 * Date: 2022.1.04
 * 
 * 【说明】:
 *  该程序记录小车在gazeo中的轨迹点
 * 【订阅】
 * 

 ***************************************************************************/
#include <ros/ros.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>

#include <sensor_msgs/PointCloud2.h>
#include <gazebo_msgs/GetModelState.h>
#include <geometry_msgs/PoseStamped.h>

typedef pcl::PointXYZI  PointType;
using namespace std;

void PublishPC(ros::Publisher pub, pcl::PointCloud<PointType>::Ptr pc){
    sensor_msgs::PointCloud2 tempMsgCloud;
    pcl::toROSMsg(*pc, tempMsgCloud);
    tempMsgCloud.header.stamp = ros::Time();
    tempMsgCloud.header.frame_id = "map";
    pub.publish(tempMsgCloud);
}

int main(int argc, char **argv){

    
    ros::init(argc, argv, "predict");
    ros::NodeHandle nh;

    ros::Rate rate(10);

    pcl::PointCloud<PointType>::Ptr car_pose_for_vis_ (new pcl::PointCloud<PointType>());
    PointType cloud_pose_3d_for_vis_last;

    // 订阅服务
    ros::ServiceClient get_model_state_client_ = nh.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    gazebo_msgs::GetModelState model_state_srv;
    model_state_srv.request.model_name = "scout/";

    ros::Publisher car_key_pose_pub_ = nh.advertise<sensor_msgs::PointCloud2>("/car_key_pose", 10); //发布小车行驶轨迹


    Eigen::Isometry3d T_map_to_world = Eigen::Isometry3d::Identity(); //world表示的是gazebo的坐标系，　map表示的是slam的坐标系
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();

    // moon mounatin sense
    // rotation_matrix << -0.924035,  -0.382308, -4.50299e-05,
    //                    0.382308,    -0.924035, -1.09356e-06,
    //                    -4.11911e-05, -1.82257e-05,       1;
    // Eigen::Vector3d t1;
    // t1 << 3.89596, -1.95579, 1.56811;

    // outside tree sense
    // rotation_matrix << 0.785091,    -0.61938, -6.87165e-05,
    //         0.61938,     0.785091,  7.26688e-05,
    //         8.93915e-06, -9.96133e-05,            1;

    // Eigen::Vector3d t1;

    // t1 << -35.1572, 10.5496, 2.83862;

    // outside tree sense
    rotation_matrix << 0.999585,    -0.0288199, -1.27727e-05,
                       0.0288199,     0.999585,  9.59108e-05,
                       1.00033e-05,  -9.6239e-05,          1;
    Eigen::Vector3d t1;
    t1 << 2.80667, -5.60424, 2.83862;

    T_map_to_world.rotate(rotation_matrix);
    T_map_to_world.pretranslate(t1);


    geometry_msgs::PoseStamped car_pose_;

    std::cout << "=== [PredDemo thread] ==> odom thread start " << std::endl;

    while (ros::ok())
    {
        PointType cloud_pose_3d;

        if (get_model_state_client_.call(model_state_srv))
        {
            // std::cout <<"gazebo pose: " << std::endl << model_state_srv.response.pose.position <<std::endl;
            Eigen::Quaterniond q(model_state_srv.response.pose.orientation.w, 
                                model_state_srv.response.pose.orientation.x, 
                                model_state_srv.response.pose.orientation.y, 
                                model_state_srv.response.pose.orientation.z);
            Eigen::Vector3d t = Eigen::Vector3d(model_state_srv.response.pose.position.x,
                                                model_state_srv.response.pose.position.y, 
                                                model_state_srv.response.pose.position.z);

            Eigen::Isometry3d P_in_world = Eigen::Isometry3d::Identity(); 
            P_in_world.rotate(q.toRotationMatrix());
            P_in_world.pretranslate(t);
            Eigen::Isometry3d P_in_map = T_map_to_world.inverse() * P_in_world;
            Eigen::Quaterniond Q(P_in_map.rotation());
            
            
            car_pose_.header.stamp = model_state_srv.response.header.stamp;
            car_pose_.pose.position.x = P_in_map.translation()[0];
            car_pose_.pose.position.y = P_in_map.translation()[1];
            car_pose_.pose.position.z = P_in_map.translation()[2];
            car_pose_.pose.orientation.w = Q.w();
            car_pose_.pose.orientation.x = Q.x();
            car_pose_.pose.orientation.y = Q.y();
            car_pose_.pose.orientation.z = Q.z();

            cloud_pose_3d.x = car_pose_.pose.position.x;
            cloud_pose_3d.y = car_pose_.pose.position.y;
            cloud_pose_3d.z = car_pose_.pose.position.z;

            // cloud_pose_3d.x = t(0);
            // cloud_pose_3d.y = t(1);
            // cloud_pose_3d.z = t(2);
        }

        


        //记录小车行驶位姿

        if (car_pose_for_vis_->points.empty())
        {
            car_pose_for_vis_->points.push_back(cloud_pose_3d);
            cloud_pose_3d_for_vis_last = cloud_pose_3d;
        }else
        {
            //每隔0.3米记录一个位置点
            if (sqrt(pow(cloud_pose_3d.x - cloud_pose_3d_for_vis_last.x,2) + pow(cloud_pose_3d.y - cloud_pose_3d_for_vis_last.y, 2) > 0.3 ))
            {
                car_pose_for_vis_->points.push_back(cloud_pose_3d);
                cloud_pose_3d_for_vis_last = cloud_pose_3d;
            }
        }
        PublishPC(car_key_pose_pub_, car_pose_for_vis_);

        rate.sleep();
        ros::spinOnce();
    }

}