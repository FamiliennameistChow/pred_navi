 /**************************************************************************
 * pc_data_make_offline.cpp
 * 
 * Author： Born Chow
 * Date: 2021.09.13
 * 
 * 【说明】:
 *  使用离线地图, 通过gazebo获取小车位姿，进行随机游走数据采集
 * 【订阅】:
 *  1  laser_reg_pc_sub_ \\ sensor_msgs::PointCloud2  \\全局点云
 *  2  imu_data_sub_  \\  sensor_msgs::Imu \\ imu数据
 ***************************************************************************/

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <gazebo_msgs/GetModelState.h>
#include <gazebo_msgs/SetModelState.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>

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
#include <math.h>
#include <random>
#include <ctime>

#include <octomap/octomap.h>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tictoc.h>

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
    ros::Subscriber imu_data_sub_;

    ros::Publisher laser_local_map_pub_, laser_local_map_car_pub_;
    ros::Publisher car_cmd_pub_;
    ros::Publisher car_key_pose_pub_;

    ros::ServiceClient get_model_state_client_;
    ros::ServiceClient set_model_state_client_;

    tf::TransformBroadcaster br_;
    tf::Transform transform_;


    PointType cloud_pose_3d_;
    PointType cloud_pose_3d_last_;
    PointType local_map_area_;
    string pc_frame_id_;
    string data_save_dir_;

    string slam_type_;
    string laser_odom_topic_;
    string pointcloud_topic_;
    string imu_topic_;

    double theta_;
    double half_diagonal_;
    double imu_time_window_;
    double sample_dis_;
    bool save_data_;

    shared_ptr<OcTreeT> map_tree_;
    double map_res_;

    pcl::PointCloud<PointType>::Ptr gloabl_map_ptr_;
    pcl::PointCloud<PointType>::Ptr car_pose_for_vis_;

    pcl::CropBox<PointType> box_filter_;

    std::vector<sensor_msgs::Imu::ConstPtr> imu_data_buffer_;
    std::queue<geometry_msgs::PoseStamped> car_pose_buffer_; //待处理位姿点队列， 一旦队列中位姿数据大于3, 则处理最老的位姿数据
    std::mutex m_buf_;
    

private:
    void map_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& msg);

    // void laser_odom_cb(const nav_msgs::Odometry::ConstPtr &laser_odom_msg);
    void record_data(geometry_msgs::PoseStamped car_pose);

    void imu_cb(const sensor_msgs::Imu::ConstPtr &imu_msg);

    void pc_to_eleva(pcl::PointCloud<PointType>::Ptr pc, shared_ptr<octomap::OcTree> &map, double res, double time);

    void reset_car();

    double cal_dis(geometry_msgs::PoseStamped p1, geometry_msgs::PoseStamped p2);


public:
    DataMake(ros::NodeHandle &nh, ros::NodeHandle& nh_);
    ~DataMake();
};

DataMake::DataMake(ros::NodeHandle &nh, ros::NodeHandle& nh_) :
nh(nh),
gloabl_map_ptr_(new pcl::PointCloud<PointType>()),
car_pose_for_vis_(new pcl::PointCloud<PointType>)
{

    //私有参数
    nh_.param<float>("local_map_length", local_map_area_.x, 2.0);
    nh_.param<float>("local_map_width", local_map_area_.y, 2.0);
    nh_.param<float>("local_map_height", local_map_area_.z, 4.0);
    nh_.param<string>("data_save_dir", data_save_dir_, "/home/bornchow/ROS_WS/slam_ws/src/data_make/data/");
    nh_.param<bool>("save_data", save_data_, true);
    nh_.param<double>("map_res", map_res_, 0.1);
    nh_.param<double>("sample_dis", sample_dis_, 0.5);
    nh_.param<double>("imu_time_window", imu_time_window_, 1.5); //采样imu数据的时间窗口

    nh_.param<string>("pc_topic", pointcloud_topic_, "/cloud_pcd"); //来源于 pcd to pointcloud
    nh_.param<string>("imu_potic", imu_topic_, "/imu"); // 仿真中的话题是　/imu   真实小车的话题是　/imu/data


    
    laser_reg_pc_sub_ = nh.subscribe<sensor_msgs::PointCloud2>(pointcloud_topic_, 10, &DataMake::map_cloud_cb, this);
    imu_data_sub_ = nh.subscribe<sensor_msgs::Imu>(imu_topic_, 100, &DataMake::imu_cb, this);
    

    laser_local_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("laser_cloud_map_local", 10);
    laser_local_map_car_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("laser_cloud_map_local_in_car", 10); // car坐标系下的点云
    car_cmd_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    car_key_pose_pub_ = nh.advertise<sensor_msgs::PointCloud2>("car_key_pose", 10);

    get_model_state_client_ = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    set_model_state_client_ = nh_.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");

    // init param
    cloud_pose_3d_last_.x = 0.0;
    cloud_pose_3d_last_.y = 0.0;
    cloud_pose_3d_last_.z = 0.0;


    //在原点设置点云初始ROI区域
    // Max Min并非立方体中随意两个对角定点，一定要严格找出x,y,z轴数值最大与最小两个定点
    box_filter_.setMax(Eigen::Vector4f(local_map_area_.x/2 , local_map_area_.y/2, local_map_area_.z/2, 1.0));
    box_filter_.setMin(Eigen::Vector4f(-local_map_area_.x/2 , -local_map_area_.y/2, -local_map_area_.z/2, 1.0));
    // theta_ = atan2(local_map_area_.x , local_map_area_.y); //计算矩形夹角
    // half_diagonal_ = sqrt(pow(local_map_area_.x, 2) + pow(local_map_area_.y, 2)) / 2;
    cout<< "pc data make init" << endl;

    map_tree_.reset(new OcTreeT(map_res_));

    ros::Rate rate(10);

    


    gazebo_msgs::GetModelState model_state_srv;
    model_state_srv.request.model_name = "scout/";

    Eigen::Isometry3d T_map_to_world = Eigen::Isometry3d::Identity(); //world表示的是gazebo的坐标系，　map表示的是slam的坐标系
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();
    rotation_matrix << -0.924035,  -0.382308, -4.50299e-05,
                       0.382308,    -0.924035, -1.09356e-06,
                      -4.11911e-05, -1.82257e-05,       1;
    Eigen::Vector3d t1;
    t1 << 3.89596, -1.95579, 1.56811;
    T_map_to_world.rotate(rotation_matrix);
    T_map_to_world.pretranslate(t1);
    
    int count = -1;

    geometry_msgs::PoseStamped car_pose_last;
    car_pose_last.pose.position.x = 0;
    car_pose_last.pose.position.y = 0;
    car_pose_last.pose.position.z = 0;

    while (ros::ok())
    {
        TicToc total_time;
        ros::spinOnce();
        // total_time.toc(">> spin time: ");

        // TicToc wait_for_imu_time;
        if (imu_data_buffer_.size() == 0){ //因为主函数运行很快，所以不一定有imu数据出来
            continue;
        }
        // wait_for_imu_time.toc(">> wait for imu time: ");

        count++;
        std::cout << "===================================== " <<  count <<std::endl;
        
        if(get_model_state_client_.call(model_state_srv)){
            std::cout <<"gazebo pose: " << std::endl << model_state_srv.response.pose.position <<std::endl;
            
            // // // 测试初始变换矩阵  
            // Eigen::Quaterniond q1(model_state_srv.response.pose.orientation.w, 
            //                     model_state_srv.response.pose.orientation.x, 
            //                     model_state_srv.response.pose.orientation.y, 
            //                     model_state_srv.response.pose.orientation.z);
            // Eigen::Vector3d t1 = Eigen::Vector3d(model_state_srv.response.pose.position.x,
            //                                 model_state_srv.response.pose.position.y, 
            //                                 model_state_srv.response.pose.position.z);
            // T_map_to_world = Eigen::Isometry3d::Identity(); //world表示的是gazebo的坐标系，　map表示的是slam的坐标系
            // T_map_to_world.rotate(q1.toRotationMatrix());
            // T_map_to_world.pretranslate(t1);
            // std::cout << "T_map_to_world: " <<std::endl << T_map_to_world.matrix() << std::endl;
            // continue;
            
        
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
            geometry_msgs::PoseStamped car_pose;
            car_pose.header.stamp = model_state_srv.response.header.stamp;
            car_pose.pose.position.x = P_in_map.translation()[0];
            car_pose.pose.position.y = P_in_map.translation()[1];
            car_pose.pose.position.z = P_in_map.translation()[2];
            car_pose.pose.orientation.w = Q.w();
            car_pose.pose.orientation.x = Q.x();
            car_pose.pose.orientation.y = Q.y();
            car_pose.pose.orientation.z = Q.z();

            //发布tf  base_link -->> map
            transform_.setOrigin(tf::Vector3(car_pose.pose.position.x, car_pose.pose.position.y, car_pose.pose.position.z));
            transform_.setRotation(tf::Quaternion(Q.x(), Q.y(), Q.z(), Q.w()));
            br_.sendTransform(tf::StampedTransform(transform_, model_state_srv.response.header.stamp, "map", "base_link"));

            
            std::cout << "P_t_in_map: " <<std::endl << P_in_map.translation().transpose() << std::endl;
            // std::cout << "P_Q_in_map: " <<std::endl << Q.coeffs().transpose() << std::endl;

            //判断是否重启模型
            TicToc reset_model_time;
            double laserRoll, laserPitch, laserYaw;
            tf::Quaternion orientation;
            tf::quaternionMsgToTF(car_pose.pose.orientation, orientation);
            tf::Matrix3x3(orientation).getRPY(laserRoll, laserPitch, laserYaw);
            std::cout << "gazbeo call--> roll: " << laserRoll *180 / M_PI << ", pitch: " 
                                                << laserPitch * 180 / M_PI << ", yaw: " 
                                                << laserYaw * 180 / M_PI << std::endl;
            // 重设小车位置条件
            if(abs(laserPitch * 180 / M_PI) > 40 || abs(laserRoll * 180 / M_PI) > 40 || count % 500 == 0 ||
                abs(model_state_srv.response.pose.position.x) > 40 ||
                abs(model_state_srv.response.pose.position.y) > 40 ){
                
                // 停止小车运动
                geometry_msgs::Twist cmd;
                cmd.linear.x = 0.0;
                car_cmd_pub_.publish(cmd);

                //路径队列中最后的数据，并清空位置队列
                while (!car_pose_buffer_.empty())
                {
                    record_data(car_pose_buffer_.front());
                    car_pose_buffer_.pop();
                }
                // 重启小车位置
                reset_car();
                continue;
            }
            reset_model_time.toc(">>reset model time: ");

            // 判断是否记录数据
            TicToc record_data_time;
            double dis = cal_dis(car_pose, car_pose_last);
            cout << "dis: " << dis << endl;
            std::cout << "car pose Num: " << car_pose_buffer_.size() << std::endl;
            if(dis > sample_dis_){
                car_pose_buffer_.push(car_pose);

                if (car_pose_buffer_.size() > 3){
                    //记录数据
                    record_data(car_pose_buffer_.front());
                    car_pose_buffer_.pop();
                }

                car_pose_last = car_pose;
            }
            record_data_time.toc(">>record data time: ");

        }else
        {
            ROS_ERROR("Failed to call service /gazebo/get_model_state ");
        }


        // 发布小车控制指令
        geometry_msgs::Twist cmd;
        cmd.linear.x = 0.5;
        car_cmd_pub_.publish(cmd);

        rate.sleep();
        total_time.toc(">>total time: ");
    }

}

DataMake::~DataMake()
{
}

double DataMake::cal_dis(geometry_msgs::PoseStamped p1, geometry_msgs::PoseStamped p2){
    return sqrt(pow((p1.pose.position.x - p2.pose.position.x),2) + 
                pow((p1.pose.position.y - p2.pose.position.y),2) + 
                pow((p1.pose.position.z - p2.pose.position.z),2));
}


void DataMake::reset_car(){
    gazebo_msgs::SetModelState set_model_srv;
    set_model_srv.request.model_state.model_name = "scout/";
    set_model_srv.request.model_state.reference_frame = "world";

    //生产随机数
    default_random_engine e(time(0));
    uniform_real_distribution<double> x(-30.0, 30.0);
    uniform_real_distribution<double> y(-30.0, 30.0);
    uniform_real_distribution<double> yaw(-1, 1);

    Eigen::AngleAxisd Vz( M_PI * yaw(e), Eigen::Vector3d(0,0,1)); //
    Eigen::Quaterniond Q(Vz);


    set_model_srv.request.model_state.pose.position.x = x(e);
    set_model_srv.request.model_state.pose.position.y = y(e);
    set_model_srv.request.model_state.pose.position.z = 3;
    set_model_srv.request.model_state.pose.orientation.w = Q.w();
    set_model_srv.request.model_state.pose.orientation.x = Q.x();
    set_model_srv.request.model_state.pose.orientation.y = Q.y();
    set_model_srv.request.model_state.pose.orientation.z = Q.z();

    if (set_model_state_client_.call(set_model_srv)){
        std::cout << " >>>> ====reset model pose --> " << x(e) << " " << y(e) << " " << std::endl;
        sleep(5); //等待小车平稳  单位s
        ros::spinOnce();
        //清除小车位置突变时的imu数据
        m_buf_.lock();
        imu_data_buffer_.clear();
        m_buf_.unlock();

    }else
    {
        ROS_ERROR("Failed to call service /gazebo/set_model_state ");
    }


}


void DataMake::imu_cb(const sensor_msgs::Imu::ConstPtr &imu_msg){
    m_buf_.lock();
    imu_data_buffer_.push_back(imu_msg);
    m_buf_.unlock();
}

void DataMake::record_data(geometry_msgs::PoseStamped car_pose){
    cloud_pose_3d_.x = car_pose.pose.position.x;
    cloud_pose_3d_.y = car_pose.pose.position.y;
    cloud_pose_3d_.z = car_pose.pose.position.z;
    double pose_time = car_pose.header.stamp.toSec();

    // 记录小车采样位置, 发布小车位置轨迹
    car_pose_for_vis_->points.push_back(cloud_pose_3d_);
    sensor_msgs::PointCloud2 tempMsgCloud0;
    pcl::toROSMsg(*car_pose_for_vis_, tempMsgCloud0);
    tempMsgCloud0.header.stamp = ros::Time();
    tempMsgCloud0.header.frame_id = pc_frame_id_;
    car_key_pose_pub_.publish(tempMsgCloud0);


    double laserRoll, laserPitch, laserYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(car_pose.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(laserRoll, laserPitch, laserYaw);

    std::cout << "===================================== " << std::endl;
    std::cout << "==============record data============ " << std::to_string(pose_time) << std::endl;
    std::cout << "===================================== " << std::endl;
    cout << "Laser roll pitch yaw: " << endl;
    cout << "roll: " << laserRoll *180 / M_PI << ", pitch: " << laserPitch * 180 / M_PI << ", yaw: " << laserYaw * 180 / M_PI << endl;
    cout << "pose: " << cloud_pose_3d_ << endl;
    cout << "dis record: " << pcl::geometry::squaredDistance(cloud_pose_3d_, cloud_pose_3d_last_) << endl;
    cout << "time: " << std::to_string(pose_time) << endl;
    std::cout <<"imu data size:  " << imu_data_buffer_.size() << std::endl;
    std::cout <<"imu data time: " << pose_time - imu_data_buffer_.front()->header.stamp.toSec() << std::endl;
 
    // 记录pointCloud
    pcl::PointCloud<PointType>::Ptr this_cloud_map(new pcl::PointCloud<PointType>());

    //获取小车周围点云
    TicToc filter_time;
    box_filter_.setTranslation(Eigen::Vector3f(cloud_pose_3d_.x, cloud_pose_3d_.y, cloud_pose_3d_.z)); // box平移量
    box_filter_.setRotation(Eigen::Vector3f(0.0, 0.0, laserYaw)); // box旋转量
    box_filter_.setNegative(false); //false保留立方体内的点而去除其他点，true是将盒子内的点去除，默认为false
    box_filter_.setInputCloud(gloabl_map_ptr_);
    box_filter_.filter(*this_cloud_map);

    cout << "p_min: " << box_filter_.getMin().transpose() << " p_max: " << box_filter_.getMax().transpose() << endl;

    if(this_cloud_map->size() <= 0){
        std::cout << " no data!!!! " << std::endl;
        return;
    }
    //去离群点
    
    pcl::StatisticalOutlierRemoval<PointType> sor;
    sor.setInputCloud(this_cloud_map);
    sor.setMeanK(50);
    sor.setStddevMulThresh(2.0);
    sor.filter(*this_cloud_map);
    filter_time.toc(">> filter time: ");


    //将点转换到车体坐标系
    // @ 测试点云不转换到车体坐标系， 而是在后期进行点云归一化操作 20210925
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();  // transform_body_to_map
    transform.translation() << cloud_pose_3d_.x, cloud_pose_3d_.y, cloud_pose_3d_.z;
    // 为何只使用yaw, 这样可以保证点云数据 在xy平面是以小车为中心，坐标系x轴与小车x轴对其, 同时地形形状还是以世界坐标系为依据
    // transform.rotate(Eigen::AngleAxisd(laserRoll,  Eigen::Vector3d::UnitX()));
    // transform.rotate(Eigen::AngleAxisd(laserPitch, Eigen::Vector3d::UnitY()));
    transform.rotate(Eigen::AngleAxisd(laserYaw,   Eigen::Vector3d::UnitZ()));

    
    Eigen::Affine3d transform_1 = transform.inverse();
    pcl::PointCloud<PointType>::Ptr pc_in_body(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud (*this_cloud_map, *pc_in_body, transform_1);

    //发布全局点云以显示
    sensor_msgs::PointCloud2 tempMsgCloud;
    pcl::toROSMsg(*this_cloud_map, tempMsgCloud);
    tempMsgCloud.header.stamp = ros::Time();
    tempMsgCloud.header.frame_id = pc_frame_id_;
    laser_local_map_pub_.publish(tempMsgCloud);
    //发布机体坐标系下的点云
    sensor_msgs::PointCloud2 tempMsgCloud2;
    pcl::toROSMsg(*pc_in_body, tempMsgCloud2);
    tempMsgCloud2.header.stamp = ros::Time();
    tempMsgCloud2.header.frame_id = pc_frame_id_;
    laser_local_map_car_pub_.publish(tempMsgCloud2);


    cloud_pose_3d_last_.x = cloud_pose_3d_.x;
    cloud_pose_3d_last_.y = cloud_pose_3d_.y;
    cloud_pose_3d_last_.z = cloud_pose_3d_.z;

    
    if(save_data_){
        //如果save_data_设置为true, 则会保存三个文件 1.点云pcd  2.imu数据txt  3.高程图(@@如果不使用pc_in_body,高程图可能要离线制作)

        //保存点云

        // @@ 测试点云不转换到车体坐标系， 而是在后期进行点云归一化操作 20210925 这里直接保存全局坐标系下的点云 this_cloud_map --> 无效
        // @@ 之前保存的是车体坐标系下的点云 pc_in_body
        pcl::io::savePCDFile(data_save_dir_ + std::to_string(pose_time) + ".pcd", *pc_in_body);


        // 记录imu的值
        std::ofstream imu_data_write(data_save_dir_ + std::to_string(pose_time) + ".txt", ios::out);


        m_buf_.lock();
        while ((pose_time - imu_data_buffer_.front()->header.stamp.toSec()) > imu_time_window_)
        {
            imu_data_buffer_.erase(imu_data_buffer_.begin());
        }
        std::cout << "imu size after: " << imu_data_buffer_.size() << std::endl;
        for (auto iter: imu_data_buffer_)
        {
            // std::cout<< "imu time: " << pose_time - iter->header.stamp.toSec() << std::endl;
            if(abs(pose_time - iter->header.stamp.toSec()) <= imu_time_window_){
                // std::cout << "imu data record!!!!" << std::endl;
                imu_data_write << std::to_string(iter->header.stamp.toSec()) << "\t" 
                    << std::to_string(iter->linear_acceleration.x) << "\t"
                    << std::to_string(iter->linear_acceleration.y) << "\t"
                    << std::to_string(iter->linear_acceleration.z) << "\t"
                    << std::to_string(iter->angular_velocity.x) << "\t"
                    << std::to_string(iter->angular_velocity.y) << "\t"
                    << std::to_string(iter->angular_velocity.z) << "\t"
                    << std::to_string(iter->orientation.w) << "\t"
                    << std::to_string(iter->orientation.x) << "\t"
                    << std::to_string(iter->orientation.y) << "\t"
                    << std::to_string(iter->orientation.z) << "\t" << endl;
            }else
            {
                std::cout << "imu data is enough!!!!" << std::endl;
                break;
            }
            

        }
        m_buf_.unlock();
        imu_data_write.close();

        //将点云做成高程图
        pc_to_eleva(pc_in_body, map_tree_, map_res_, pose_time);
    }
     

}


void DataMake::map_cloud_cb(const sensor_msgs::PointCloud2ConstPtr& msg){

    pcl::fromROSMsg(*msg, *gloabl_map_ptr_);

    pc_frame_id_ = msg->header.frame_id;

}


//采用遍历的方式
void DataMake::pc_to_eleva(pcl::PointCloud<PointType>::Ptr pc, shared_ptr<octomap::OcTree> &map, double res, double time){
    map->clear();
    //加载pc
    for(auto p:pc->points){
        map->updateNode(octomap::point3d(p.x, p.y, p.z), true);
    }

    unsigned max_TreeDepth = map->getTreeDepth();

    double size_x, size_y, size_z;
    map -> getMetricSize(size_x, size_y, size_z); //获得 octomap 的 size

    double minX, minY, minZ;
    double maxX, maxY, maxZ;
    map->getMetricMin(minX, minY, minZ);
    map->getMetricMax(maxX, maxY, maxZ);
    cout << "Map min: " << minX << " " << minY << "  " << minZ << " Max : " << maxX << " " << maxY << " " << maxZ << std::endl;
    cv::Mat img(ceil(local_map_area_.x/res), ceil(local_map_area_.y/res), CV_16SC1, cv::Scalar(32767));  //根据小车的roi区域每次创建的高程图大小都是一致的

    if (ceil(size_x/res) < img.rows || ceil(size_y/res) < img.cols){
        minY = minY < -size_y /2 ? minY : -size_y /2;
        minX = minX < -size_x /2 ? minX : -size_x /2;
        cout << "min adjust: " << minX << " " << minY << "  " << minZ << " Max : " << maxX << " " << maxY << " " << maxZ << std::endl;
    }

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
            img_x = it.getX();
            img_y = it.getY();
            pixel_x = floor((img_x - minX)/res);
            pixel_y = floor((img_y - minY)/res);

            if(pixel_x < img.rows && pixel_y < img.cols && pixel_y >= 0 && pixel_x >= 0) {
                std::cout << i << " img_P " << it.getX() << "," << it.getY() << " " << it.getZ() * 1000 <<
                          " min:  " << minX << " , " << minY << " , " <<
                          " --> " << pixel_x << " " << pixel_y << std::endl;
                img.at<short>(pixel_x, pixel_y) = it.getZ() * 1000; //高程的单位换为毫米
            }
        }
    }

    // map->write( "map.ot");
    cout <<"---"<< endl;
    // cout << img << endl;
    Mat img_save;
    img.convertTo(img_save, CV_16UC1, 1, 32768);

//    cv::imshow("img", img);
//    cv::waitKey(0);
    cv::imwrite(data_save_dir_ + std::to_string(time)+".png", img_save);

    // cout << img_save << endl;
    cout << "=============finish============= " << data_save_dir_+"/"+std::to_string(time)+".png" << endl;
}

// ==========================class end =========================

int main(int argc, char **argv)
{
    ros::init(argc, argv, "data_make_offline");
    ros::NodeHandle nh;
    ros::NodeHandle nh_("~");

    DataMake datamake(nh, nh_);

    ros::spin();
    return 0;

}