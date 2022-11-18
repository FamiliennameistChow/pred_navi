 /**************************************************************************
 * navi.cpp
 * 
 * Author： Born Chow
 * Date: 2021.12.01
 * 
 * 【说明】:
 *  该程序演示三种不同模型的预测结果
 * 【订阅】
 * 

 ***************************************************************************/

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <gazebo_msgs/GetModelState.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_listener.h>
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>

#include <pcl/filters/crop_box.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/distances.h>


#include <navi_pred/NetPred.hpp>
#include <navi_pred/tictoc.h>
#include <navi_pred/point_type_define.h>

#include <thread>
#include <mutex>
#include <fstream>
#include <map>



class PredDemo
{
private:
    ros::NodeHandle nh;

    // 订阅与发布
    ros::Subscriber local_pc_sub_;   // 订阅局部点云
    ros::Subscriber goal_sub_;
    ros::Publisher point_pred_in_map_pub_, point_pred_in_car_pub_;
    ros::Publisher cnn_pred_in_car_pub_, cnn_pred_in_map_pub_;
    ros::Publisher cnn_resnet_pred_in_car_pub_, cnn_resnet_pred_in_map_pub_;
    ros::Publisher local_map_pub_;
    ros::Publisher local_map_in_car_pub_;

    ros::ServiceClient get_model_state_client_;

    // ros参数
    string point_model_dir_;
    string pointcloud_topic_;
    int sample_point_num_;
    float box_stride_;
    float box_kernel_;
    PointType local_map_area_;
    PointType robot_size_;
    int img_size_;
    float map_res_;
    string cnn_model_dir_;
    string cnn_resnet_model_dir_;


    string pc_frame_id_;
    bool get_goal_;

    // 网络预测模块
    shared_ptr<NetPred> point_net_pred_ = make_shared<NetPred>();
    // cnn 模型 (self)
    shared_ptr<NetPred> cnn_net_pred_ = make_shared<NetPred>();
    // cnn 模型 (resnet)
    shared_ptr<NetPred> cnn_resnet_net_pred_ = make_shared<NetPred>();

    pcl::PointCloud<PointType>::Ptr map_ptr_; //订阅到的全局地图
    pcl::PointCloud<PointType>::Ptr process_pc_in_map_;
    pcl::CropBox<PointType> box_filter_;


    geometry_msgs::PoseStamped car_pose_;
    geometry_msgs::PoseStamped this_car_pose_; //当前正在预测地形的小车位置
    geometry_msgs::Point start_point_, goal_point_, pre_goal_point_;

    std::mutex pose_mtx_;

    PointType local_target_;
   

private:
    void LocalMapCallBack(const sensor_msgs::PointCloud2ConstPtr& msg);
    void GoalCallBack(const geometry_msgs::PointStamped::ConstPtr &msg);

    void PublishPC(ros::Publisher pub, pcl::PointCloud<PointType>::Ptr pc);
    bool point_equal(geometry_msgs::Point p1, geometry_msgs::Point p2){
		if(abs(p1.x - p2.x)< 0.00001 && abs(p1.y - p2.y)< 0.00001 && abs(p1.z - p2.z)< 0.00001){
			return true;
		}else
		{
			return false;
		}	
	}

public:
    
    PredDemo(ros::NodeHandle &nh, ros::NodeHandle& nh_);
    void plan();
    void replan();
    void PublishOdomThread();
    ~PredDemo();
};

PredDemo::PredDemo(ros::NodeHandle &nh, ros::NodeHandle& nh_):
nh(nh),
map_ptr_(new pcl::PointCloud<PointType>()),
process_pc_in_map_(new pcl::PointCloud<PointType>()),
get_goal_(false),
pc_frame_id_("map")
{

    pose_mtx_.lock();
    this_car_pose_ = car_pose_;
    pose_mtx_.unlock();

    //私有参数
    nh_.param<string>("net_model_dir", point_model_dir_, "/home/bornchow/ROS_WS/slam_ws/src/navi_pred/model/model_3210_0.9759_0.9509.pt");
    nh_.param<float>("box_stride", box_stride_, 0.5);
    nh_.param<float>("box_kernel", box_kernel_, 2.0);
    nh_.param<int>("sample_point_num", sample_point_num_, 500);
    nh_.param<string>("pc_topic", pointcloud_topic_, "/cloud_pcd"); // 
    // cnn模型参数
    nh_.param<string>("cnn_model_dir", cnn_model_dir_, "/home/bornchow/workFile/navi_net/model/cnnself_model_926_33.497849.pt");
    nh_.param<int>("img_size", img_size_, 20);
    nh_.param<float>("map_res", map_res_, 0.1);

    nh_.param<string>("cnn_model_dir", cnn_resnet_model_dir_, "/home/bornchow/workFile/navi_net/model/cnnresnet_model_119_141.968216.pt");
    // 车体周围local 区域
    nh_.param<float>("local_map_length", local_map_area_.x, 10.0);
    nh_.param<float>("local_map_width", local_map_area_.y, 10.0);
    nh_.param<float>("local_map_height", local_map_area_.z, 2.0);  //6.0
    // 车体大小 这个应该与网络训练时, data_make 的大小一致 --> 网络的输入区域
    nh_.param<float>("robot_size_length", robot_size_.x, 2.0);
    nh_.param<float>("robot_size_width", robot_size_.y, 2.0);
    nh_.param<float>("robot_szie_height", robot_size_.z, 4.0);


    // 订阅话题
    goal_sub_ =nh.subscribe<geometry_msgs::PointStamped>("clicked_point", 1, &PredDemo::GoalCallBack, this); //订阅目标点
    local_pc_sub_ = nh.subscribe<sensor_msgs::PointCloud2>(pointcloud_topic_, 1, &PredDemo::LocalMapCallBack, this); // 订阅局部点云
    

    // 发布话题
    local_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_map", 1); //发布全局坐标下的局部地图
    local_map_in_car_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_map_in_car", 1); // car坐标系下的地图
    // 发布点云模型的预测结果
    point_pred_in_car_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("point_pred_pc_in_car", 1); //发布处理的局部点云
    point_pred_in_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("point_pred_pc_in_map", 1); //发布处理的局部点云
    // 发布CNN模型的预测结果(slef)
    cnn_pred_in_car_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cnn_pred_pc_in_car", 1); //发布处理的局部点云
    cnn_pred_in_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cnn_pred_pc_in_map", 1); //发布处理的局部点云
    // 发布CNN模型的预测结果(resnet)
    cnn_resnet_pred_in_car_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cnn_resnet_pred_pc_in_car", 1); //发布处理的局部点云
    cnn_resnet_pred_in_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("cnn_resnet_pred_pc_in_map", 1); //发布处理的局部点云


    // 订阅服务
    get_model_state_client_ = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");

    // 加载预测模块
    point_net_pred_->InitParam(sample_point_num_, box_stride_, box_kernel_, point_model_dir_);
   
    cnn_net_pred_->InitParam(img_size_, map_res_, box_stride_, box_kernel_, cnn_model_dir_);

    cnn_resnet_net_pred_->InitParam(img_size_, map_res_, box_stride_, box_kernel_, cnn_resnet_model_dir_);


    //在原点设置点云初始ROI区域
    // Max Min并非立方体中随意两个对角定点，一定要严格找出x,y,z轴数值最大与最小两个定点

    box_filter_.setMax(Eigen::Vector4f(local_map_area_.x/2 , local_map_area_.y/2, local_map_area_.z/2, 1.0));
    box_filter_.setMin(Eigen::Vector4f(-local_map_area_.x/2 , -local_map_area_.y/2, -local_map_area_.z/2, 1.0));

    // 初始化历史目标点
    pre_goal_point_.x = pre_goal_point_.y = pre_goal_point_.z = 0.0;

 
}

PredDemo::~PredDemo()
{
}

/**
 * @brief plan 规划函数
 * 主要有四个步骤：
 * 1. 获取小车周围的点云
 */
void PredDemo::plan(){

    std::cout << "=== [Predict] ==> plan process ========================" <<std::endl; 
    // ---------- 1. 获取小车周围点云 --------------------
    PointType cloud_pose_3d;
    pose_mtx_.lock();
    cloud_pose_3d.x = car_pose_.pose.position.x;
    cloud_pose_3d.y = car_pose_.pose.position.y;
    cloud_pose_3d.z = car_pose_.pose.position.z;
    double pose_time = car_pose_.header.stamp.toSec();
    double laserRoll, laserPitch, laserYaw;
    tf::Quaternion orientation;
    tf::quaternionMsgToTF(car_pose_.pose.orientation, orientation);
    tf::Matrix3x3(orientation).getRPY(laserRoll, laserPitch, laserYaw);
    this_car_pose_ = car_pose_;
    pose_mtx_.unlock();

    cout << "roll: " << laserRoll *180 / M_PI << ", pitch: " << laserPitch * 180 / M_PI << ", yaw: " << laserYaw * 180 / M_PI << endl;
    cout << "pose: " << cloud_pose_3d << endl;


    // 记录pointCloud
    pcl::PointCloud<PointType>::Ptr this_cloud_map(new pcl::PointCloud<PointType>());

    //获取小车周围点云
    TicToc filter_time;
    double x = cloud_pose_3d.x + 3 * cos(laserYaw);
    double y = cloud_pose_3d.y + 3 * sin(laserYaw);
    box_filter_.setTranslation(Eigen::Vector3f(x, y, cloud_pose_3d.z)); // box平移量
    box_filter_.setRotation(Eigen::Vector3f(0.0, 0.0, laserYaw)); // box旋转量
    box_filter_.setNegative(false); //false保留立方体内的点而去除其他点，true是将盒子内的点去除，默认为false
    box_filter_.setInputCloud(map_ptr_);
    box_filter_.filter(*this_cloud_map);

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
    filter_time.toc("=== [Predict] ==> filter time: ");

    //将点转换到车体坐标系
    // @ 测试点云不转换到车体坐标系， 而是在后期进行点云归一化操作 20210925
    Eigen::Affine3d transform = Eigen::Affine3d::Identity();  // transform_body_to_map
    transform.translation() << cloud_pose_3d.x, cloud_pose_3d.y, cloud_pose_3d.z;
    // 为何只使用yaw, 这样可以保证点云数据 在xy平面是以小车为中心，坐标系x轴与小车x轴对其, 同时地形形状还是以世界坐标系为依据
    // transform.rotate(Eigen::AngleAxisd(laserRoll,  Eigen::Vector3d::UnitX()));
    // transform.rotate(Eigen::AngleAxisd(laserPitch, Eigen::Vector3d::UnitY()));
    transform.rotate(Eigen::AngleAxisd(laserYaw,   Eigen::Vector3d::UnitZ()));

    
    Eigen::Affine3d transform_1 = transform.inverse();
    pcl::PointCloud<PointType>::Ptr pc_in_body(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud (*this_cloud_map, *pc_in_body, transform_1);

    // 发布全局坐标系下的局部地图
    PublishPC(local_map_pub_, this_cloud_map);

    
    // // 发布小车坐标系下的局部地图
    PublishPC(local_map_in_car_pub_, pc_in_body);

    // 接下来要用的是 pc_in_body
    // --------------------- 2.点云预测 -----------------------------------
    std::cout << "=== [Predict] ==> point model prediction ---------------" <<std::endl;
    // 处理局部地图 10*10 local map
    TicToc point_pre_time;
    pcl::PointCloud<PointType>::Ptr point_pred_pc_in_body (new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr point_pred_pc_pass (new pcl::PointCloud<PointType>());

    point_net_pred_->PointPred(pc_in_body, point_pred_pc_in_body, point_pred_pc_pass); //只预测local map
    // point_net_pred_->PointPred(pc_in_body, sample_point_pcs, process_pc_in_body, process_pc_pass, trj_sample_pred);
    point_pre_time.toc("=== [Predict] ==> point model predict time");

    // 发布小车坐标系下的预测结果点云
    PublishPC(point_pred_in_car_pub_, point_pred_pc_in_body);

    // 将预测后的点云转换到map 坐标系下
    pcl::transformPointCloud (*point_pred_pc_in_body, *process_pc_in_map_, transform);

    // 发布map坐标系下的预测结果点云
    PublishPC(point_pred_in_map_pub_, process_pc_in_map_);


    // --------------------- 3.CNN模型预测(self ) -----------------------------------
    std::cout << "=== [Predict] ==> CNN model self prediction ---------------" <<std::endl;
    TicToc cnn_pre_time;

    pcl::PointCloud<PointType>::Ptr cnn_pred_pc_in_body (new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cnn_pred_pc_pass (new pcl::PointCloud<PointType>());

    cnn_net_pred_->CNNPred(pc_in_body, cnn_pred_pc_in_body, cnn_pred_pc_pass);
    cnn_pre_time.toc("=== [Predict] ==> cnn model predict time");

    // 发布小车坐标系下的预测结果
    PublishPC(cnn_pred_in_car_pub_, cnn_pred_pc_in_body);

    // 将预测后的点云转换到map 坐标系下
    pcl::PointCloud<PointType>::Ptr cnn_pred_pc_in_map (new pcl::PointCloud<PointType>());
    pcl::transformPointCloud (*cnn_pred_pc_in_body, *cnn_pred_pc_in_map, transform);

    // 发布map坐标系下的预测结果点云
    PublishPC(cnn_pred_in_map_pub_, cnn_pred_pc_in_map);

    // --------------------- 4.CNN模型预测(resnet ) -----------------------------------
    std::cout << "=== [Predict] ==> CNN resnet model self prediction ---------------" <<std::endl;

    TicToc cnn_resnet_pre_time;
    pcl::PointCloud<PointType>::Ptr cnn_resnet_pred_pc_in_body (new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr cnn_resnet_pred_pc_pass (new pcl::PointCloud<PointType>());

    cnn_resnet_net_pred_->CNNPred(pc_in_body, cnn_resnet_pred_pc_in_body, cnn_resnet_pred_pc_pass);
    cnn_resnet_pre_time.toc("=== [Predict] ==> cnn resnet model predict time");

    // 发布小车坐标系下的预测结果
    PublishPC(cnn_resnet_pred_in_car_pub_, cnn_resnet_pred_pc_in_body);

    // 将预测后的点云转换到map 坐标系下
    pcl::PointCloud<PointType>::Ptr cnn_resnet_pred_pc_in_map (new pcl::PointCloud<PointType>());
    pcl::transformPointCloud (*cnn_resnet_pred_pc_in_body, *cnn_resnet_pred_pc_in_map, transform);

    // 发布map坐标系下的预测结果点云
    PublishPC(cnn_resnet_pred_in_map_pub_, cnn_resnet_pred_pc_in_map);

}


/**
 * @brief replan 重规划函数
 * 检查重规划标志符, 该函数目前由map回调函数启动
 */
void PredDemo::replan(){

    //没有目标点, 不需要重规划
    if (!get_goal_)
    {
        return;
    }
    
    //重规划标志符1　是否走出当前预测区域, 如果是 --> 重规划
    pose_mtx_.lock();
    double dis = sqrt(pow(car_pose_.pose.position.x - this_car_pose_.pose.position.x, 2) + pow(car_pose_.pose.position.y - this_car_pose_.pose.position.y, 2));
    pose_mtx_.unlock();
    if (dis > 3.0)
    {
        std::cout << "=== [PredDemo replan] ==> replan by outsiding the area: " << std::endl;
        plan();
        return;
    }
    
}


/**
 * @brief PublishOdomThread 从gazbeo中获取小车真值,　并发布 [bask_link -> map] 的tf转换
 */
void PredDemo::PublishOdomThread(){
    ros::Rate rate(10);

    tf::TransformBroadcaster br_;
    tf::Transform transform_;

    gazebo_msgs::GetModelState model_state_srv;
    model_state_srv.request.model_name = "scout/";


    Eigen::Isometry3d T_map_to_world = Eigen::Isometry3d::Identity(); //world表示的是gazebo的坐标系，　map表示的是slam的坐标系
    Eigen::Matrix3d rotation_matrix = Eigen::Matrix3d::Identity();

    // moon mounatin sense
    // rotation_matrix << -0.924035,  -0.382308, -4.50299e-05,
    //                    0.382308,    -0.924035, -1.09356e-06,
    //                    -4.11911e-05, -1.82257e-05,       1;
    // Eigen::Vector3d t1;
    // t1 << 3.89596, -1.95579, 1.56811;

    // outside tree sense
    rotation_matrix << 0.999585,    -0.0288199, -1.27727e-05,
                       0.0288199,     0.999585,  9.59108e-05,
                       1.00033e-05,  -9.6239e-05,          1;
    Eigen::Vector3d t1;
    t1 << 2.80667, -5.60424, 2.83862;

    T_map_to_world.rotate(rotation_matrix);
    T_map_to_world.pretranslate(t1);

    std::cout << "=== [PredDemo thread] ==> odom thread start " << std::endl;

    while (ros::ok())
    {
        pose_mtx_.lock();
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

            //发布tf  base_link -->> map
            transform_.setOrigin(tf::Vector3(car_pose_.pose.position.x, car_pose_.pose.position.y, car_pose_.pose.position.z));
            transform_.setRotation(tf::Quaternion(Q.x(), Q.y(), Q.z(), Q.w()));
            br_.sendTransform(tf::StampedTransform(transform_, model_state_srv.response.header.stamp, "map", "base_link"));
            
        }
        pose_mtx_.unlock();

        rate.sleep();
        ros::spinOnce();
    }

}


/**
 * @brief PublishPC 发布点云
 */
void PredDemo::PublishPC(ros::Publisher pub, pcl::PointCloud<PointType>::Ptr pc){
    sensor_msgs::PointCloud2 tempMsgCloud;
    pcl::toROSMsg(*pc, tempMsgCloud);
    tempMsgCloud.header.stamp = ros::Time();
    tempMsgCloud.header.frame_id = pc_frame_id_;
    pub.publish(tempMsgCloud);
}


void PredDemo::LocalMapCallBack(const sensor_msgs::PointCloud2ConstPtr& msg){

    pcl::fromROSMsg(*msg, *map_ptr_);
    pc_frame_id_ = msg->header.frame_id;

    replan();

}


void PredDemo::GoalCallBack(const geometry_msgs::PointStamped::ConstPtr &msg){
    geometry_msgs::Point p = msg->point;
    if (!point_equal(p, pre_goal_point_))
    {
        goal_point_ = p;
        std::cout << "=== [PredDemo] ==>  Goal point set to: " << p.x << " " << p.y << " " << p.z << std::endl;
        pre_goal_point_ = p;
        get_goal_ = true;

        plan();
        
    }
    
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "predict");
    ros::NodeHandle nh;
    ros::NodeHandle nh_("~");

    PredDemo pred_demo(nh, nh_);

    std::thread odom_thread(&PredDemo::PublishOdomThread, &pred_demo);


    ros::spin();
    return 0;

}