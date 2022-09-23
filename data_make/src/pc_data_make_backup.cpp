 /**************************************************************************
 * pc_data_make.cpp
 * 
 * Author： Born Chow
 * Date: 2021.07.15
 * 
 * 
 * 说明:  1.获取LIO-SAM中的注册点云，拼接为全局点云(LIO-SAM原作中的全局点云经过体素降采样而变得稀疏)
 * 　　　 2.以机器人为中心裁剪SLAM发布的全局地图，制作局部点云数据
 *       3. https://blog.csdn.net/qinqinxiansheng/article/details/105492925
 *         　https://pointclouds.org/documentation/classpcl_1_1_crop_box.html
 * 【订阅】
 ***************************************************************************/

#include <ros/ros.h>

#include <tf/transform_listener.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/geometry.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl_conversions/pcl_conversions.h>

#include <queue>
#include <mutex>
#include <iostream>
#include <fstream>

#include <octomap/octomap.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

# define PI 3.14
// # define TEST
// # define DEBUG_ELE
typedef pcl::PointXYZI  PointType;
typedef octomap::OcTree OcTreeT;
using namespace std;
using namespace cv;
class DataMake
{
private:
    ros::NodeHandle nh;

    ros::Subscriber laser_reg_pc_sub_;
    ros::Subscriber laser_odom_sub_;
    ros::Subscriber imu_data_sub_;

    ros::Publisher laser_local_map_pub_;
    ros::Publisher laser_gloabl_map_pub_;

    PointType cloud_pose_3d_;
    PointType cloud_pose_3d_last_;
    PointType local_map_area_;
    string pc_frame_id_;
    string data_save_dir_;
    double theta_;
    double half_diagonal_;
    double imu_time_window_;
    bool save_data_;

    shared_ptr<OcTreeT> map_tree_;
    double map_res_;

    pcl::PointCloud<PointType>::Ptr gloabl_map_ptr_;

    pcl::CropBox<PointType> box_filter_;

    pcl::PassThrough<PointType> pass_x_;
    pcl::PassThrough<PointType> pass_y_;
    pcl::PassThrough<PointType> pass_z_;

    std::queue<sensor_msgs::Imu::ConstPtr> imu_data_buffer_;
    std::mutex m_buf_;
    

private:
    void laser_reg_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& msg);

    void laser_odom_cb(const nav_msgs::Odometry::ConstPtr &laser_odom_msg);

    void imu_cb(const sensor_msgs::Imu::ConstPtr &imu_msg);

    void pc_to_eleva(pcl::PointCloud<PointType>::Ptr pc, shared_ptr<octomap::OcTree> &map, double res, double time);


public:
    DataMake(ros::NodeHandle &nh, ros::NodeHandle& nh_);
    ~DataMake();
};

DataMake::DataMake(ros::NodeHandle &nh, ros::NodeHandle& nh_) :
nh(nh),
gloabl_map_ptr_(new pcl::PointCloud<PointType>())
{

    nh_.param<float>("local_map_length", local_map_area_.x, 2.0);
    nh_.param<float>("local_map_width", local_map_area_.y, 1.0);
    nh_.param<float>("local_map_height", local_map_area_.z, 4.0);
    nh_.param<string>("data_save_dir", data_save_dir_, "/home/bornchow/ROS_WS/slam_ws/src/data_make/data/");
    nh_.param<bool>("save_data", save_data_, true);
    nh_.param<double>("map_res", map_res_, 0.1);


    laser_reg_pc_sub_ = nh.subscribe<sensor_msgs::PointCloud2>("/lio_sam/mapping/cloud_registered", 10, &DataMake::laser_reg_cloud_cb, this);
    laser_odom_sub_ = nh.subscribe<nav_msgs::Odometry>("/lio_sam/mapping/odometry", 10, &DataMake::laser_odom_cb, this);
    imu_data_sub_ = nh.subscribe<sensor_msgs::Imu>("imu/data", 100, &DataMake::imu_cb, this);

    laser_local_map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("laser_cloud_map_local", 10);

    laser_gloabl_map_pub_ = nh.advertise<sensor_msgs::PointCloud2>("laser_cloud_map_gloabl", 10);

    // init param
    pass_x_.setFilterFieldName("x");
    pass_y_.setFilterFieldName("y");
    pass_z_.setFilterFieldName("z");

    cloud_pose_3d_last_.x = 0.0;
    cloud_pose_3d_last_.y = 0.0;
    cloud_pose_3d_last_.z = 0.0;

    imu_time_window_ = 1.0;

    //在原点设置点云初始ROI区域
    // Max Min并非立方体中随意两个对角定点，一定要严格找出x,y,z轴数值最大与最小两个定点
    box_filter_.setMax(Eigen::Vector4f(local_map_area_.x/2 , local_map_area_.y/2, local_map_area_.z/2, 1.0));
    box_filter_.setMin(Eigen::Vector4f(-local_map_area_.x/2 , -local_map_area_.y/2, -local_map_area_.z/2, 1.0));
    // theta_ = atan2(local_map_area_.x , local_map_area_.y); //计算矩形夹角
    // half_diagonal_ = sqrt(pow(local_map_area_.x, 2) + pow(local_map_area_.y, 2)) / 2;
    cout<< "11111111" << endl;

    map_tree_.reset(new OcTreeT(map_res_));

}

DataMake::~DataMake()
{
}

void DataMake::imu_cb(const sensor_msgs::Imu::ConstPtr &imu_msg){
    m_buf_.lock();
    imu_data_buffer_.push(imu_msg);
    m_buf_.unlock();
}

void DataMake::laser_odom_cb(const nav_msgs::Odometry::ConstPtr &laser_odom_msg){
    cloud_pose_3d_.x = laser_odom_msg->pose.pose.position.x;
    cloud_pose_3d_.y = laser_odom_msg->pose.pose.position.y;
    cloud_pose_3d_.z = laser_odom_msg->pose.pose.position.z;
    double pose_time = laser_odom_msg->header.stamp.toSec(); //cloud_pose_3d_.intensity时间不会变？？

    double laserRoll, laserPitch, laserYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(laser_odom_msg->pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(laserRoll, laserPitch, laserYaw);

    cout << "======================" << endl;
    cout << "Laser roll pitch yaw: " << endl;
    cout << "roll: " << laserRoll *180 / PI << ", pitch: " << laserPitch * 180 / PI << ", yaw: " << laserYaw * 180 / PI << endl;
    cout << "pose: " << cloud_pose_3d_ << endl;
    cout << "dis: " << pcl::geometry::squaredDistance(cloud_pose_3d_, cloud_pose_3d_last_) << endl;
    cout << "time: " << std::to_string(pose_time) << endl;
    // cout << "time2: " << std::to_string(cloud_pose_3d_.intensity) << endl;


    if (pcl::geometry::squaredDistance(cloud_pose_3d_, cloud_pose_3d_last_) > 2.0) {
       
        // 记录pointCloud
        pcl::PointCloud<PointType>::Ptr this_cloud_map(new pcl::PointCloud<PointType>());

        // pcl::copyPointCloud(*gloabl_map_ptr_,  *this_cloud_map);

        //  double delta;
        // PointType p1, p2;  //p1和p2是指立方体的两个对角点
        // if (laserYaw < 0)
        // {
        //     laserYaw = PI + laserYaw;
        // }

        // if (laserYaw < PI / 2)
        // {
        //     delta = PI / 2 - laserYaw + theta_;
        // }

        // if (laserYaw > PI / 2 )
        // {
        //     delta = theta_ - (laserYaw - PI  / 2);
        // }

        // p1.x = cloud_pose_3d_.x + sin(delta)*half_diagonal_;
        // p1.y = cloud_pose_3d_.y + cos(delta)*half_diagonal_;
        // p1.z = cloud_pose_3d_.z + local_map_area_.z/2;

        // p2.x = cloud_pose_3d_.x - sin(delta)*half_diagonal_;
        // p2.y = cloud_pose_3d_.y - cos(delta)*half_diagonal_;
        // p2.z = cloud_pose_3d_.z - local_map_area_.z/2;

        // box_filter_.setMax(Eigen::Vector4f(p1.x , p1.y, p1.z, 1.0));
        // box_filter_.setMin(Eigen::Vector4f(p2.x , p2.y, p2.z, 1.0));


        //获取小车周围点云
        box_filter_.setTranslation(Eigen::Vector3f(cloud_pose_3d_.x, cloud_pose_3d_.y, cloud_pose_3d_.z)); // box平移量
        box_filter_.setRotation(Eigen::Vector3f(0.0, 0.0, laserYaw)); // box旋转量
        box_filter_.setNegative(false); //false保留立方体内的点而去除其他点，true是将盒子内的点去除，默认为false
        box_filter_.setInputCloud(gloabl_map_ptr_);
        box_filter_.filter(*this_cloud_map);

        cout << "p_min: " << box_filter_.getMin() << " p_max: " << box_filter_.getMax() << endl;

        //去离群点
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(this_cloud_map);
        sor.setMeanK(50);
        sor.setStddevMulThresh(2.0);
        sor.filter(*this_cloud_map);


        //将点转换到车体坐标系
        Eigen::Affine3d transform = Eigen::Affine3d::Identity();  // transform_body_to_map
        transform.translation() << cloud_pose_3d_.x, cloud_pose_3d_.y, cloud_pose_3d_.z;
        transform.rotate(Eigen::AngleAxisd(laserRoll,  Eigen::Vector3d::UnitX()));
        transform.rotate(Eigen::AngleAxisd(laserPitch, Eigen::Vector3d::UnitY()));
        transform.rotate(Eigen::AngleAxisd(laserYaw,   Eigen::Vector3d::UnitZ()));
        Eigen::Affine3d transform_1 = transform.inverse();

        pcl::PointCloud<PointType>::Ptr pc_in_body(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud (*this_cloud_map, *pc_in_body, transform_1);

        //发布点云以显示
        sensor_msgs::PointCloud2 tempMsgCloud;
        pcl::toROSMsg(*this_cloud_map, tempMsgCloud);
        tempMsgCloud.header.stamp = ros::Time();
        tempMsgCloud.header.frame_id = pc_frame_id_;
        laser_local_map_pub_.publish(tempMsgCloud);

        cloud_pose_3d_last_.x = cloud_pose_3d_.x;
        cloud_pose_3d_last_.y = cloud_pose_3d_.y;
        cloud_pose_3d_last_.z = cloud_pose_3d_.z;

        
        if(save_data_){
            //如果save_data_设置为true, 则会保存三个文件 1.点云pcd  2.imu数据txt  3.高程图
            //保存点云
            pcl::io::savePCDFile(data_save_dir_ + std::to_string(pose_time) + ".pcd", *pc_in_body);


            // 记录imu的值
            std::ofstream imu_data_write(data_save_dir_ + std::to_string(pose_time) + ".txt", ios::out);
            m_buf_.lock();
            while ((pose_time - imu_data_buffer_.front()->header.stamp.toSec()) > imu_time_window_)
            {
                imu_data_buffer_.pop();
            }
            while (!imu_data_buffer_.empty())
            {
                imu_data_write << std::to_string(imu_data_buffer_.front()->header.stamp.toSec()) << "\t" 
                            << std::to_string(imu_data_buffer_.front()->linear_acceleration.x) << "\t"
                            << std::to_string(imu_data_buffer_.front()->linear_acceleration.y) << "\t"
                            << std::to_string(imu_data_buffer_.front()->linear_acceleration.z) << "\t"
                            << std::to_string(imu_data_buffer_.front()->angular_velocity.x) << "\t"
                            << std::to_string(imu_data_buffer_.front()->angular_velocity.y) << "\t"
                            << std::to_string(imu_data_buffer_.front()->angular_velocity.z) << "\t"
                            << std::to_string(imu_data_buffer_.front()->orientation.w) << "\t"
                            << std::to_string(imu_data_buffer_.front()->orientation.x) << "\t"
                            << std::to_string(imu_data_buffer_.front()->orientation.y) << "\t"
                            << std::to_string(imu_data_buffer_.front()->orientation.z) << "\t" << endl;
                imu_data_buffer_.pop();
            } 
            m_buf_.unlock();
            imu_data_write.close();

            //将点云做成高程图
            pc_to_eleva(pc_in_body, map_tree_, map_res_, pose_time);
        }
    }
    
#ifndef TEST
    if (pcl::geometry::squaredDistance(cloud_pose_3d_, cloud_pose_3d_last_) > 2.0) 
    {
        // 获取点云数据
        // 设置裁剪区域
        pcl::PointCloud<PointType>::Ptr this_cloud_map(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*gloabl_map_ptr_,  *this_cloud_map);
        pass_x_.setFilterLimits(cloud_pose_3d_.x - local_map_area_.x/2, cloud_pose_3d_.x + local_map_area_.x/2);
        pass_y_.setFilterLimits(cloud_pose_3d_.y - local_map_area_.y/2, cloud_pose_3d_.y + local_map_area_.y/2);
        pass_z_.setFilterLimits(cloud_pose_3d_.z - local_map_area_.z/2, cloud_pose_3d_.z + local_map_area_.z/2);

        pass_x_.setInputCloud((*this_cloud_map).makeShared());
        pass_x_.filter(*this_cloud_map);
        pass_y_.setInputCloud((*this_cloud_map).makeShared());
        pass_y_.filter(*this_cloud_map);
        pass_z_.setInputCloud((*this_cloud_map).makeShared());
        pass_z_.filter(*this_cloud_map);
        
        sensor_msgs::PointCloud2 tempMsgCloud;
        pcl::toROSMsg(*this_cloud_map, tempMsgCloud);
        tempMsgCloud.header.stamp = ros::Time();
        tempMsgCloud.header.frame_id = pc_frame_id_;
        laser_local_map_pub_.publish(tempMsgCloud);

        cloud_pose_3d_last_.x = cloud_pose_3d_.x;
        cloud_pose_3d_last_.y = cloud_pose_3d_.y;
        cloud_pose_3d_last_.z = cloud_pose_3d_.z;
    }
#endif    

}


void DataMake::laser_reg_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& msg){

        pcl::PointCloud<PointType>::Ptr reg_pc_ptr(new pcl::PointCloud<PointType>());
        pcl::fromROSMsg(*msg, *reg_pc_ptr);
        *gloabl_map_ptr_ += *reg_pc_ptr;
        pc_frame_id_ = msg->header.frame_id;
        
        sensor_msgs::PointCloud2 tempMsgCloud;
        pcl::toROSMsg(*gloabl_map_ptr_, tempMsgCloud);
        tempMsgCloud.header.stamp = ros::Time();
        tempMsgCloud.header.frame_id = msg->header.frame_id;
        laser_gloabl_map_pub_.publish(tempMsgCloud);  
}


void DataMake::pc_to_eleva(pcl::PointCloud<PointType>::Ptr pc, shared_ptr<octomap::OcTree> &map, double res, double time){
    map->clear();
    //加载pc
    for(auto p:pc->points){
        map->updateNode(octomap::point3d(p.x, p.y, p.z), true);
    }

    unsigned max_TreeDepth = map->getTreeDepth();

    double size_x, size_y, size_z;
    map -> getMetricSize(size_x, size_y, size_z); //获得 octomap 的 size
    cv::Mat img(ceil(local_map_area_.x/res), ceil(local_map_area_.y/res), CV_16SC1, cv::Scalar(32767));  //根据小车的roi区域每次创建的高程图大小都是一致的

    cout << "img size: " << img.rows << " " << img.cols << endl;
    cout << "map size: " << size_x << " " << size_y << " " << size_z  << " "<<  ceil(size_x/res) <<" " << ceil(size_y/res) << " " << size_z/res << " " << endl;
    
    int i = 0;
    double img_x = 0, img_y = 0;  //图像坐标系
    int pixel_x = 0; int pixel_y = 0; //像素坐标系
    //   图像坐标系定义
    //      -------->  y
    //      |
    //      | 
    //      |
    //   x  V  
    //遍历octomap 
    for (OcTreeT::iterator it = map->begin(max_TreeDepth), end = map->end(); it != end; ++it)
    {
        if (it.getDepth() == max_TreeDepth){ //说明是最小的节点
            i++;
            img_x = it.getX()/res;
            img_y = it.getY()/res;
            pixel_x = ceil(img_x + local_map_area_.x/(2*res)) -1;   //　-1是因为像素坐标是从０开始
            pixel_y = ceil(img_y + local_map_area_.y/(2*res)) -1;

            if(pixel_x < img.rows && pixel_y < img.cols && pixel_y >= 0 && pixel_x >= 0){
                cout << i  << " or: " << it.getX() << "," << it.getY() << 
                              " img_p:  " << img_x << " , " << img_y << " , " << it.getZ()*1000 << 
                              " --> " << pixel_x << " " << pixel_y << endl;
                img.at<short>(pixel_x, pixel_y) = it.getZ()*1000; //高程的单位换为毫米
            }
        }
    }

    // map->write( "map.ot");
    cout <<"---"<< endl;
    // cout << img << endl;
    Mat img_save;
    img.convertTo(img_save, CV_16UC1, 1, 32768);

    cv::imwrite(data_save_dir_ + std::to_string(time)+".png", img_save);

    // cout << img_save << endl;
    cout << "----finish----" <<endl;
}

// ==========================class end =========================

int main(int argc, char **argv)
{
    ros::init(argc, argv, "crop_global_map");
    ros::NodeHandle nh;
    ros::NodeHandle nh_("~");

    DataMake datamake(nh, nh_);

    ros::spin();
    return 0;

}