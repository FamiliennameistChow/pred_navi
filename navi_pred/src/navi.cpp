 /**************************************************************************
 * navi.cpp
 * 
 * Author： Born Chow
 * Date: 2021.09.30
 * 
 * 【说明】:
 *  使用网络预测结果导航, 定位采用gazebo中的真值, 地图是预先建立好的
 *  这里的导航分为两个线程：
 *  1. 线程1: plan() 用于处理一个10M*10M的局部区域，包括: 1). 获取小车周围的点云  2). 点云预测 3). 基于预测的导航规划既是生成局部导航引导线
 *           这个函数不是一直执行的，只有接收到新目标 GoalCallBack() 或者满足 重规划机制时 replan() 才会调用,；
 *           这里的重规划replan()函数， 在每次地图更新的时候执行一次
 *  2. 线程2： PublishCmdThread() 实时控制线程， 这是一个20Hz的控制循环, 主要根据plan()中的局部引导线和轨迹库信息,评估最优控制轨迹，然后给机器人发布控制指令
 * 【订阅】
 * 
 * modify data: 20211021
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
#include <pcl/segmentation/extract_clusters.h>
// #include <pcl/segmentation/impl/extract_clusters.hpp>
// #include <pcl/search/impl/search.hpp>

#include <navi_pred/NetPred.hpp>
#include <navi_pred/tictoc.h>
#include <navi_pred/point_type_define.h>

#include <thread>
#include <mutex>
#include <fstream>
#include <map>

#define CONTROL


class Navi
{
private:
    ros::NodeHandle nh;

    // 订阅与发布
    ros::Subscriber local_pc_sub_;   // 订阅局部点云
    ros::Subscriber goal_sub_;
    ros::Publisher local_process_in_map_pub_, local_process_in_car_pub_, local_process_pass_in_map_pub_;
    ros::Publisher local_map_pub_;
    ros::Publisher local_map_in_car_pub_;
    ros::Publisher trj_sample_point_pub_;
    ros::Publisher pc_temp_pub_;
    ros::Publisher trj_pub_;
    ros::Publisher guide_line_pub_;
    ros::Publisher car_cmd_pub_;
    ros::Publisher car_key_pose_pub_;

    ros::Publisher vis_goal_pub_, vis_local_goal_pub_;
    ros::ServiceClient get_model_state_client_;

    // ros参数
    string net_model_dir_;
    string config_dir_;
    string pointcloud_topic_;
    int sample_point_num_;
    float box_stride_;
    float box_kernel_;
    PointType local_map_area_;
    PointType robot_size_;
    float guide_line_step_;
    float cost_factor_;


    string pc_frame_id_;
    bool set_goal_;
    bool is_planning_;
    bool trj_outside_map_flag_; //重规划标志， 判断局部轨迹是否在局部地图外，如果是，重规划
    std::vector<TRJ_INFO> trj_lab_info_;
    std::unordered_map<PointType, std::vector<int> > sample_point_info_;

    // 网络预测模块
    shared_ptr<NetPred> net_pred_ = make_shared<NetPred>();

    pcl::PointCloud<PointType>::Ptr map_ptr_; //订阅到的全局地图
    pcl::PointCloud<PointType>::Ptr this_local_map_ptr_; //当前的局部地图
    pcl::PointCloud<PointType>::Ptr guide_line_ptr_; //局部引导线
    pcl::PointCloud<PointType>::Ptr process_pc_in_map_;
    pcl::CropBox<PointType> box_filter_, box_filter_robot_size_;
    pcl::PointCloud<perception::PointXYZIYaw>::Ptr trj_sample_point_;
    pcl::PointCloud<PointType>::Ptr trj_point_;
    pcl::PointCloud<PointType>::Ptr car_pose_for_vis_; //用于显示小车轨迹

    pcl::EuclideanClusterExtraction<PointType> ece_;

    geometry_msgs::PoseStamped car_pose_;
    geometry_msgs::PoseStamped this_car_pose_; //当前正在预测地形的小车位置
    geometry_msgs::Point start_point_, goal_point_, pre_goal_point_;

    std::mutex pose_mtx_;
    std::mutex cmd_mtx_;

    double cmd_v_;
    double cmd_w_;
    PointType local_target_;
   

    visualization_msgs::Marker mk_;

private:
    void LocalMapCallBack(const sensor_msgs::PointCloud2ConstPtr& msg);
    void GoalCallBack(const geometry_msgs::PointStamped::ConstPtr &msg);

    void PublishPC(ros::Publisher pub, pcl::PointCloud<PointType>::Ptr pc);
    void PublishVisLocalGoal(float x, float y, float z);
    bool point_equal(geometry_msgs::Point p1, geometry_msgs::Point p2){
		if(abs(p1.x - p2.x)< 0.00001 && abs(p1.y - p2.y)< 0.00001 && abs(p1.z - p2.z)< 0.00001){
			return true;
		}else
		{
			return false;
		}	
	}

public:
    
    Navi(ros::NodeHandle &nh, ros::NodeHandle& nh_);
    void plan();
    void replan();
    void PublishOdomThread();
    void PublishCmdThread();
    ~Navi();
};

Navi::Navi(ros::NodeHandle &nh, ros::NodeHandle& nh_):
nh(nh),
map_ptr_(new pcl::PointCloud<PointType>()),
process_pc_in_map_(new pcl::PointCloud<PointType>()),
this_local_map_ptr_(new pcl::PointCloud<PointType>()),
guide_line_ptr_(new pcl::PointCloud<PointType>()),
trj_sample_point_(new pcl::PointCloud<perception::PointXYZIYaw>()),
set_goal_(false),
trj_outside_map_flag_(false),
pc_frame_id_("map"),
trj_point_(new pcl::PointCloud<PointType>()),
car_pose_for_vis_(new pcl::PointCloud<PointType>())
{

    cmd_mtx_.lock();
    cmd_v_ = 0.0;
    cmd_w_ = 0.0;
    is_planning_ = true;
    cmd_mtx_.unlock();

    pose_mtx_.lock();
    this_car_pose_ = car_pose_;
    pose_mtx_.unlock();

    //私有参数
    nh_.param<string>("net_model_dir", net_model_dir_, "/home/bornchow/ROS_WS/slam_ws/src/navi_pred/model/model_3210_0.9759_0.9509.pt");
    nh_.param<string>("config_dir", config_dir_, "/home/bornchow/ROS_WS/slam_ws/src/navi_pred/config");
    nh_.param<float>("box_stride", box_stride_, 0.5);
    nh_.param<float>("box_kernel", box_kernel_, 2.0);
    nh_.param<int>("sample_point_num", sample_point_num_, 500);
    nh_.param<string>("pc_topic", pointcloud_topic_, "/cloud_pcd"); // 
    // 车体周围local 区域
    nh_.param<float>("local_map_length", local_map_area_.x, 10.0);
    nh_.param<float>("local_map_width", local_map_area_.y, 10.0);
    nh_.param<float>("local_map_height", local_map_area_.z, 6.0);
    // 车体大小 这个应该与网络训练时, data_make 的大小一致 --> 网络的输入区域
    nh_.param<float>("robot_size_length", robot_size_.x, 2.0);
    nh_.param<float>("robot_size_width", robot_size_.y, 2.0);
    nh_.param<float>("robot_szie_height", robot_size_.z, 4.0);
    // 引导线步长
    nh_.param<float>("guide_line_step", guide_line_step_, 0.1);
    // 用来表征最优轨迹是趋于壁障 还是趋于目标点 的权重因子  越大表示趋于壁障
    nh_.param<float>("cost_factor", cost_factor_, 0.0); 

    // 订阅话题
    goal_sub_ =nh.subscribe<geometry_msgs::PointStamped>("clicked_point", 1, &Navi::GoalCallBack, this); //订阅目标点
    local_pc_sub_ = nh.subscribe<sensor_msgs::PointCloud2>(pointcloud_topic_, 1, &Navi::LocalMapCallBack, this); // 订阅局部点云
    

    // 发布话题
    local_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_map", 1); //发布全局坐标下的局部地图
    local_map_in_car_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_map_in_car", 1); // car坐标系下的地图
    local_process_in_car_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_process_pc_in_car", 1); //发布处理的局部点云
    local_process_in_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_process_pc_in_map", 1); //发布处理的局部点云
    local_process_pass_in_map_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_process_pass_pc_in_map", 1); //发布处理的局部点云(可穿越点云)
    pc_temp_pub_ =  nh_.advertise<sensor_msgs::PointCloud2>("pc_temp", 1); //发布临时点云 测试用
    car_key_pose_pub_ = nh.advertise<sensor_msgs::PointCloud2>("car_key_pose", 10); //发布小车行驶轨迹
    // 发布轨迹
    trj_sample_point_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("trj_sample_point", 1); //发布轨迹采样点(map系下)
    trj_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("opti_trj", 1); //发布当前轨迹
    guide_line_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("guide_line", 1); //发布当前引导线

    car_cmd_pub_ = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10); //控制量
    vis_goal_pub_ = nh_.advertise<visualization_msgs::Marker>( "goal_node", 10 ); // 发布采样点
    vis_local_goal_pub_ = nh_.advertise<visualization_msgs::Marker>( "local_goal_node", 10 ); // 发布采样点

    // 订阅服务
    get_model_state_client_ = nh_.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");

    // 加载预测模块
    net_pred_->InitParam(sample_point_num_, box_stride_, box_kernel_, net_model_dir_);


    // 加载轨迹库文件  sample_point_info.txt \ trj_lab_info.txt \ sample_trj_point_remove_overlap_yaw.pcd
    // 1. trj_lab_info.txt  -> std::vector<TRJ_INFO> trj_lab_info_;
      
    string trj_lab_info_file_name = config_dir_ + "/trj_lab_info.txt";
    ifstream trj_lab_info_file(trj_lab_info_file_name);
    if (trj_lab_info_file)
    {
        std::cout << "=== [init] ==> load config " << trj_lab_info_file_name <<std::endl;
        string line;
        while (getline(trj_lab_info_file, line))
        {
            std::vector<string> line_str;
            line_str = boostsplit(line, " ");
            TRJ_INFO trj_info;
            trj_info.trj_num = atoi(line_str[0].c_str());
            trj_info.v = atof(line_str[1].c_str());
            trj_info.w = atof(line_str[2].c_str());
            trj_info.sample_point_num = atoi(line_str[3].c_str());
            //std::cout << " trj info: " << trj_info.trj_num << " " << trj_info.v << " " << trj_info.w << " " << trj_info.sample_point_num << std::endl;
            trj_lab_info_.push_back(trj_info);
        }
        
    }else{
        std::cout << "=== [init] ==> no such file: " << trj_lab_info_file_name <<std::endl;
        return;
    }

    // 2. sample_point_info.txt -> std::unordered_map<PointType, std::vector<int>> sample_point_info_;
    string sample_point_info_name = config_dir_ + "/sample_point_info.txt";
    ifstream sample_point_info_file(sample_point_info_name);
    
    if (sample_point_info_file)
    {
        std::cout << "=== [init] ==> load config " << sample_point_info_name <<std::endl;
        string line;
        while (getline(sample_point_info_file, line))
        {
            std::vector<string> line_str;
            line_str = boostsplit(line, " "); //多返回一个乱码字符??
            //std::cout << line <<std::endl;
            //std::cout << line_str.size() << std::endl;
            PointType p;
            p.x = atof(line_str[0].c_str());
            p.y = atof(line_str[1].c_str());
            p.z = atof(line_str[2].c_str());
            p.intensity = atof(line_str[3].c_str());
         
            std::vector<int> trj_nums;
            for (size_t i = 4; i < line_str.size(); i++)
            {
                trj_nums.push_back(atoi(line_str[i].c_str()));
            }

            if(trj_nums.back() == 0){
                trj_nums.pop_back();
            }

            sample_point_info_.insert(std::make_pair(p, trj_nums));     
        }
        
    }else
    {
        std::cout << "=== [init] ==> no such file: " << sample_point_info_name <<std::endl;
        return;
    }
    
    
    // ********************打印轨迹库结果
    // for(auto trj_info : trj_lab_info_){
    //     std::cout << " trj info: " << trj_info.trj_num << " " << trj_info.v << " " << trj_info.w << " " << trj_info.sample_point_num << std::endl;
    // }

    // for(auto &sample_point : sample_point_info_){
    //     std::cout << "sample points: " << sample_point.first << " ";

    //     for(auto i : sample_point.second){
    //         std::cout << i << " ";
    //     }

    //     std::cout << std::endl;
    // }

    // 3. 加载pcd文件 sample_trj_point_remove_overlap_yaw.pcd
    string pcd_file_name = config_dir_ + "/sample_trj_point_remove_overlap_yaw.pcd";
    // pcl::PointCloud<perception::PointXYZIYaw>::Ptr trj_sample_point_ (new pcl::PointCloud<perception::PointXYZIYaw>);
    if (pcl::io::loadPCDFile<perception::PointXYZIYaw>(pcd_file_name, *trj_sample_point_) == -1)
    {
        PCL_ERROR ("Couldn't read file sample_trj_point_remove_overlap_yaw.pcd \n");
        return;
    }

    // 4. 记载pcd文件
    string pcd_trj_file_name = config_dir_ + "/trj_dataset_remove_overlap_useful.pcd";
    if (pcl::io::loadPCDFile<PointType>(pcd_trj_file_name, *trj_point_) == -1)
    {
        PCL_ERROR ("Couldn't read file trj_dataset_remove_overlap_useful.pcd \n");
        return;
    }

    // // ***********************显示采样轨迹点
    // pcl::PointCloud<PointType>::Ptr trj_sample_point_1 (new pcl::PointCloud<PointType>);
    // pcl::copyPointCloud(*trj_sample_point_, *trj_sample_point_1);

    // boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer("3D Viewer"));
    // viewer->setBackgroundColor(1, 1, 1);
    // pcl::visualization::PointCloudColorHandlerGenericField<PointType> fildColor(trj_sample_point_1, "intensity"); // 按照 intensity 强度字段进行渲染
    // viewer->addPointCloud<PointType>(trj_sample_point_1, fildColor, "cloud");
    // viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "cloud");

    // while (!viewer->wasStopped()){
    //     viewer->spinOnce();
    // }



    //在原点设置点云初始ROI区域
    // Max Min并非立方体中随意两个对角定点，一定要严格找出x,y,z轴数值最大与最小两个定点
    // box_filter_.setMax(Eigen::Vector4f(local_map_area_.x/2 + box_stride_*3 , local_map_area_.y/2 + box_stride_*3, local_map_area_.z/2, 1.0));
    // box_filter_.setMin(Eigen::Vector4f(-local_map_area_.x/2 - box_stride_*3 , -local_map_area_.y/2 - box_stride_*3 , -local_map_area_.z/2, 1.0));

    // box_filter_.setMax(Eigen::Vector4f(local_map_area_.x/2 , local_map_area_.y/2, local_map_area_.z/2, 1.0));
    box_filter_.setMax(Eigen::Vector4f(local_map_area_.x/2 , local_map_area_.y/2, 1, 1.0));
    box_filter_.setMin(Eigen::Vector4f(-local_map_area_.x/2 , -local_map_area_.y/2, -local_map_area_.z/2, 1.0));

    box_filter_robot_size_.setMax(Eigen::Vector4f(robot_size_.x/2 , robot_size_.y/2, robot_size_.z/2, 1.0));
    box_filter_robot_size_.setMin(Eigen::Vector4f(-robot_size_.x/2 , -robot_size_.y/2, -robot_size_.z/2, 1.0));

    // box_filter_body_.setMax(Eigen::Vector4f(local_map_area_.x/2 , local_map_area_.y/2, local_map_area_.z/2, 1.0));
    // box_filter_body_.setMin(Eigen::Vector4f(-local_map_area_.x/2 , -local_map_area_.y/2, -local_map_area_.z/2, 1.0));

    // 聚类参数
    ece_.setClusterTolerance (0.3); //设置近邻搜索的搜索半径为10cm
    ece_.setMinClusterSize (10);//设置一个聚类需要的最少点数目为100
    ece_.setMaxClusterSize (25000); //设置一个聚类需要的最大点数目为25000


    // 初始化历史目标点
    pre_goal_point_.x = pre_goal_point_.y = pre_goal_point_.z = 0.0;

    //目标点显示 vis
    mk_.header.frame_id = pc_frame_id_;
    mk_.header.stamp = ros::Time::now();
    mk_.ns = "goal";
    mk_.type = visualization_msgs::Marker::CUBE;
    mk_.action = visualization_msgs::Marker::ADD;

    mk_.pose.orientation.x = 0.0;
    mk_.pose.orientation.y = 0.0;
    mk_.pose.orientation.z = 0.0;
    mk_.pose.orientation.w = 1.0;

    mk_.color.a = 1.0;
    mk_.color.r = 1.0;
    mk_.color.g = 0.0;
    mk_.color.b = 1.0;

    mk_.scale.x = 0.5;
    mk_.scale.y = 0.5;
    mk_.scale.z = 0.5;

 
}

Navi::~Navi()
{
}

/**
 * @brief plan 规划函数
 * 主要有三个步骤：
 * 1. 获取小车周围的点云
 * 2. 点云预测
 * 3. 基于预测的导航规划
 */
void Navi::plan(){
    cmd_mtx_.lock();
    is_planning_ = true;
    cmd_mtx_.unlock();
    std::cout << "=== [Navi] ==> plan process ---------------" <<std::endl;
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
    filter_time.toc("=== [Navi] ==> filter time: ");

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

    cmd_mtx_.lock();
    pcl::copyPointCloud(*this_cloud_map, *this_local_map_ptr_);
    cmd_mtx_.unlock();

    // 发布全局坐标系下的局部地图
    PublishPC(local_map_pub_, this_cloud_map);

    
    // // 发布小车坐标系下的局部地图
    PublishPC(local_map_in_car_pub_, pc_in_body);


    // --------------------- 2.点云预测 -----------------------------------

    // 处理局部地图 10*10 local map
    TicToc pre_time;
    pcl::PointCloud<PointType>::Ptr process_pc_in_body (new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr process_pc_pass (new pcl::PointCloud<PointType>());

    net_pred_->PointPred(pc_in_body, process_pc_in_body, process_pc_pass); //只预测local map
    // net_pred_->PointPred(pc_in_body, sample_point_pcs, process_pc_in_body, process_pc_pass, trj_sample_pred);
    pre_time.toc("=== [Navi] ==> predict time: ");

    // box_filter_body_.setTranslation(Eigen::Vector3f(0.0, 0.0, 0.0)); // box平移量
    // box_filter_body_.setRotation(Eigen::Vector3f(0.0, 0.0, 0.0)); // box旋转量
    // box_filter_body_.setNegative(false); //false保留立方体内的点而去除其他点，true是将盒子内的点去除，默认为false
    // box_filter_body_.setInputCloud(process_pc_in_body);
    // box_filter_body_.filter(*process_pc_in_body);

    // 发布小车坐标系下的预测结果点云
    PublishPC(local_process_in_car_pub_, process_pc_in_body);

    // 将预测后的点云转换到map 坐标系下
    pcl::transformPointCloud (*process_pc_in_body, *process_pc_in_map_, transform);

    pcl::transformPointCloud (*process_pc_pass, *process_pc_pass, transform);

    // 发布map坐标系下的预测结果点云
    PublishPC(local_process_in_map_pub_, process_pc_in_map_);

    // 发布可穿越点云
    PublishPC(local_process_pass_in_map_pub_, process_pc_pass);

    // ---------------------- 3.基于预测的导航规划 -------------------------
    // 基于 process_pc_pass 生成局部目标点与导航引导线

    TicToc guide_line_time;
    if (process_pc_pass->points.empty())
    {
        return;
    }
    

    if (sqrt(pow(goal_point_.x - cloud_pose_3d.x, 2) + pow(goal_point_.y - cloud_pose_3d.y, 2)) < 4) // 判断如果机器人离目标点很近
    {
        local_target_.x = goal_point_.x;
        local_target_.y = goal_point_.y;
        local_target_.z = goal_point_.z;
    }else
    {

        //欧式聚类
        std::vector<pcl::PointIndices> ece_inlier;
        pcl::search::KdTree<PointType>::Ptr tree (new pcl::search::KdTree<PointType>);
        tree->setInputCloud(process_pc_pass);

        ece_.setSearchMethod(tree);
        ece_.setInputCloud (process_pc_pass);
        ece_.extract (ece_inlier);//从点云中提取聚类，并将点云索引保存在ece_inlier中

        std::cout <<"seg num: " << ece_inlier.size() << std::endl;

        // k 近邻 查找距离机器人最近的那片点云
        PointType thisPosePoint;
        thisPosePoint = cloud_pose_3d;

        pcl::KdTreeFLANN<PointType> segKdtree;
        int segK = 1;
        std::vector<int> pointSegIdxNKNSearch(segK);
        std::vector<float> pointSegNKNSquaredDistance(segK);

        float min_dis = 1000;
        int min_index = 1000;
        float this_dis = 10;

        for (int i = 0; i < ece_inlier.size(); i++)
        {
            // 获取分割后的点云
            pcl::PointCloud<PointType>::Ptr cloud_seg (new pcl::PointCloud<PointType>());
            std::vector<int> this_inlier = ece_inlier[i].indices;
            pcl::copyPointCloud(*process_pc_pass, this_inlier, *cloud_seg);//按照索引提取点云数据

            // 计算该片点云与机器人的最近距离 this_dis
            segKdtree.setInputCloud(cloud_seg);
            if ( segKdtree.nearestKSearch (thisPosePoint, segK, pointSegIdxNKNSearch, pointSegNKNSquaredDistance) > 0 ){
                this_dis = pointSegNKNSquaredDistance[0];
                if(this_dis < min_dis){
                    std::cout << cloud_seg->points[pointSegIdxNKNSearch[0]] << std::endl;
                    std::cout <<"dis: " << this_dis << std::endl;
                    min_dis = this_dis;
                    min_index = i;
                }
            }
        }

        // 获取需要处理的点云 该点云是机器人所在的可通行点云， 后续的局部target是在这片点云上进行搜索的
        std::cout << "min_index: " << min_index << std::endl;
        std::vector<int> this_inlier = ece_inlier[min_index].indices;
        pcl::PointCloud<PointType>::Ptr choosed_pc (new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*process_pc_pass, this_inlier, *choosed_pc);//按照索引提取点云数据

        
        // 搜索局部导航点：
        pcl::KdTreeFLANN<PointType> kdtree;
        kdtree.setInputCloud(choosed_pc);
        int K = 100;
        PointType searchPoint;
        searchPoint.x = goal_point_.x;
        searchPoint.y = goal_point_.y;
        searchPoint.z = cloud_pose_3d.z;

        std::vector<int> pointIdxNKNSearch(K);
        std::vector<float> pointNKNSquaredDistance(K);
        pcl::PointCloud<PointType>::Ptr local_target_pc (new pcl::PointCloud<PointType>());
        if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ){
            for (size_t i = 0; i < pointIdxNKNSearch.size(); i++)
            {
                local_target_pc->points.push_back(choosed_pc->points[pointIdxNKNSearch[i] ]);
            }
        }
        PublishPC(pc_temp_pub_, local_target_pc);


        Eigen::Vector4f centroid;					// 质心
        pcl::compute3DCentroid(*local_target_pc, centroid);	// 齐次坐标，（c0,c1,c2,1）
        std::cout << "=== [Navi] ==> local map target: " << centroid.transpose() << std::endl;

        //发布局部目标点
        local_target_.x = centroid(0);
        local_target_.y = centroid(1);
        local_target_.z = 0;
        PublishVisLocalGoal(centroid(0), centroid(1), centroid(2));
    }
    
    
    //生成引导线
    pcl::PointCloud<PointType>::Ptr pc_guide_line (new pcl::PointCloud<PointType>());
    double theta = atan2(abs(local_target_.x - cloud_pose_3d.x), abs(local_target_.y - cloud_pose_3d.y));
    double dis = sqrt(pow(local_target_.x - cloud_pose_3d.x, 2) + pow(local_target_.y - cloud_pose_3d.y, 2));
    float delta_x = guide_line_step_*sin(theta);
    float delta_y = guide_line_step_*cos(theta);
    for(int i = 0; i*guide_line_step_ < dis + 2; i++){ //让引导线变长一点
        PointType p;
        p.x = (local_target_.x - cloud_pose_3d.x > 0) ?  cloud_pose_3d.x + delta_x*i : cloud_pose_3d.x - delta_x*i;
        p.y = (local_target_.y - cloud_pose_3d.y > 0) ?  cloud_pose_3d.y + delta_y*i : cloud_pose_3d.y - delta_y*i;
        p.z = 0;   
        pc_guide_line->points.push_back(p);
    }

    PublishPC(guide_line_pub_, pc_guide_line);

    cmd_mtx_.lock();
    pcl::copyPointCloud(*pc_guide_line, *guide_line_ptr_);
    cmd_mtx_.unlock();
    guide_line_time.toc("=== [Navi] ==> guide line time: ");



    // 将轨迹采样点转到map坐标系下
    pcl::PointCloud<perception::PointXYZIYaw>::Ptr trj_sample_point_in_map (new pcl::PointCloud<perception::PointXYZIYaw>);
    pcl::transformPointCloud (*trj_sample_point_, *trj_sample_point_in_map, transform);

    pcl::PointCloud<PointType>::Ptr trj_sample_point_in_map_1 (new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*trj_sample_point_in_map, *trj_sample_point_in_map_1);

    // 发布采样轨迹点
    PublishPC(trj_sample_point_pub_, trj_sample_point_in_map_1);

    cmd_mtx_.lock();
    is_planning_ = false;
    cmd_mtx_.unlock();


}


/**
 * @brief replan 重规划函数
 * 检查重规划标志符, 该函数目前由map回调函数启动
 */
void Navi::replan(){

    //没有目标点, 不需要重规划
    if (!set_goal_)
    {
        return;
    }
    
    //重规划标志符1　是否走了一段距离, 如果是 --> 重规划
    pose_mtx_.lock();
    double dis = sqrt(pow(car_pose_.pose.position.x - this_car_pose_.pose.position.x, 2) + pow(car_pose_.pose.position.y - this_car_pose_.pose.position.y, 2));
    pose_mtx_.unlock();
    if (dis > 4.0)
    {
        std::cout << "=== [Navi replan] ==> replan by outsiding the area: " << std::endl;
        plan();
        return;
    }

    //重规划标识符2 局部轨迹是否超出当前区域 如果是 --> 重规划
    if (trj_outside_map_flag_)
    {
        std::cout << "=== [Navi replan] ==> replan by trj outsiding the area: " << std::endl;
        plan();
        cmd_mtx_.lock();
        trj_outside_map_flag_ = false;
        cmd_mtx_.unlock();
        return;
    }

    //重规划标志符 3 到达局部目标点 这里主要解决的是局部目标点离小车太近的问题
    //局部目标点到目标点的距离
    float dis_to_goal = sqrt(pow(goal_point_.x - local_target_.x, 2) + pow(goal_point_.y - local_target_.y, 2));
    if(dis_to_goal < 0.1){ //说明此刻的局部路标点是目标点，不需要重规划
        return;
    }else //局部目标点不是目标点时
    {
        //小车距离局部目标点的距离
        double local_dis = sqrt(pow(car_pose_.pose.position.x - local_target_.x, 2) + pow(car_pose_.pose.position.y - local_target_.y, 2));
        if (local_dis < 0.5)
        {
            std::cout << "=== [Navi replan] ==> replan by near local point: " << std::endl;
            plan();
            return;
        }
        
    }
    


    

    
}

void Navi::PublishVisLocalGoal(float x, float y, float z){

    visualization_msgs::Marker mk;

    mk.header.frame_id = pc_frame_id_;
    mk.header.stamp = ros::Time::now();
    mk.ns = "local_goal";
    mk.type = visualization_msgs::Marker::CUBE;
    mk.action = visualization_msgs::Marker::ADD;

    mk.pose.orientation.x = 0.0;
    mk.pose.orientation.y = 0.0;
    mk.pose.orientation.z = 0.0;
    mk.pose.orientation.w = 1.0;

    mk.color.a = 1.0;
    mk.color.r = 0.0;
    mk.color.g = 0.0;
    mk.color.b = 1.0;

    mk.scale.x = 0.5;
    mk.scale.y = 0.5;
    mk.scale.z = 0.5;

    mk.pose.position.x = x;
    mk.pose.position.y = y;
    mk.pose.position.z = z;

    vis_local_goal_pub_.publish(mk);
}

/**
 * @brief PublishCmdThread 获取计算出的cmd_v_　与　cmd_w_ 发布小车控制指令【/cmd_vel】
 */
void Navi::PublishCmdThread(){

    ros::Rate rate(20);
    bool trj_outsid_local_map = false;

    //显示轨迹用
    PointType cloud_pose_3d_for_vis_last;

    while (ros::ok())
    {   

        if (!set_goal_) //如果没有目标点, 不运动
        {
            // cmd_mtx_.lock();
            // cmd_v_ = 0.0;
            // cmd_w_ = 0.0;
            // cmd_mtx_.unlock();
            geometry_msgs::Twist cmd;
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;

            #ifdef CONTROL
            car_cmd_pub_.publish(cmd);
            #endif

            continue;
        }
        

        // -------------------------------------------------------------
        //               实时预测采样轨迹点的可通过性
        
        
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
        pose_mtx_.unlock();

        //记录小车行驶位姿
        cloud_pose_3d.intensity = laserYaw;

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


        cmd_mtx_.lock();
        bool planing = is_planning_;
        cmd_mtx_.unlock();

        if(planing){
            geometry_msgs::Twist cmd;
            cmd_mtx_.lock();
            cmd.linear.x = cmd_v_;
            cmd.angular.z = cmd_w_;
            cmd_mtx_.unlock();

            #ifdef CONTROL
            car_cmd_pub_.publish(cmd);
            #endif
            continue;
        }
        
        
        // 判断是否到底目标点
        if (sqrt(pow(goal_point_.x - cloud_pose_3d.x, 2) + pow(goal_point_.y - cloud_pose_3d.y, 2)) < 0.5)
        {
            geometry_msgs::Twist cmd;
            cmd.linear.x = 0.0;
            cmd.angular.z = 0.0;
            #ifdef CONTROL
            car_cmd_pub_.publish(cmd);
            #endif
            std::cout << "=== [cmd thread] ==> get goal ---------------" <<std::endl;
            set_goal_ = false;
            continue;
        }
        
        std::cout << "=== [cmd thread] ==> real time process ---------------" <<std::endl;

        TicToc real_time_pred;

        //将点转换到车体坐标系
        // @ 测试点云不转换到车体坐标系， 而是在后期进行点云归一化操作 20210925
        Eigen::Affine3d transform = Eigen::Affine3d::Identity();  // transform_body_to_map
        transform.translation() << cloud_pose_3d.x, cloud_pose_3d.y, cloud_pose_3d.z;
        // 为何只使用yaw, 这样可以保证点云数据 在xy平面是以小车为中心，坐标系x轴与小车x轴对其, 同时地形形状还是以世界坐标系为依据
        // transform.rotate(Eigen::AngleAxisd(laserRoll,  Eigen::Vector3d::UnitX()));
        // transform.rotate(Eigen::AngleAxisd(laserPitch, Eigen::Vector3d::UnitY()));
        transform.rotate(Eigen::AngleAxisd(laserYaw,   Eigen::Vector3d::UnitZ()));
        //将轨迹采样点转到map坐标系下
        pcl::PointCloud<perception::PointXYZIYaw>::Ptr trj_sample_point_in_map (new pcl::PointCloud<perception::PointXYZIYaw>);
        pcl::transformPointCloud (*trj_sample_point_, *trj_sample_point_in_map, transform);
        real_time_pred.toc("=== [cmd thread] ==> tranform: ");
        
        TicToc get_local_map_time;
        std::vector<pcl::PointCloud<PointType>::Ptr > sample_point_pcs;

        pcl::PointCloud<PointType>::Ptr this_cloud_map (new pcl::PointCloud<PointType>);
        cmd_mtx_.lock();
        pcl::copyPointCloud(*this_local_map_ptr_, *this_cloud_map);
        cmd_mtx_.unlock();

        box_filter_robot_size_.setNegative(false); //false保留立方体内的点而去除其他点，true是将盒子内的点去除，默认为false
        box_filter_robot_size_.setInputCloud(this_cloud_map);
        trj_outsid_local_map = false;
        for (size_t i = 0; i < trj_sample_point_in_map->points.size(); i++)
        {
            pcl::PointCloud<PointType>::Ptr temp (new pcl::PointCloud<PointType>);
            box_filter_robot_size_.setTranslation(Eigen::Vector3f(trj_sample_point_in_map->points[i].x, 
                                                        trj_sample_point_in_map->points[i].y, 
                                                        cloud_pose_3d.z)); // box平移量
            box_filter_robot_size_.setRotation(Eigen::Vector3f(0.0, 0.0, laserYaw + trj_sample_point_in_map->points[i].yaw)); // box旋转量
            box_filter_robot_size_.filter(*temp);
            sample_point_pcs.push_back(temp);

            if (temp->points.empty()) //说明有轨迹已经超出局部地图
            {
                trj_outsid_local_map = true;
                continue;
            }
            
        }

        get_local_map_time.toc("=== [cmd thread] ==> get local map ");

        if (trj_outsid_local_map)
        {
            cmd_mtx_.lock();
            trj_outside_map_flag_ = true;
            cmd_mtx_.unlock();
            continue;
        }
        

        TicToc pred_time;
        std::vector<int> trj_sample_pred; 
        net_pred_->PointPred(sample_point_pcs, trj_sample_pred);
        std::cout << "trj sample pred size: " << trj_sample_pred.size() << std::endl;
        pred_time.toc("=== [cmd thread] ==> pred time ");


        TicToc opti_trj_time;
        //判断每条轨迹的可通过性
        std::vector<int> trj_pred_cout(trj_lab_info_.size());
        for (size_t i = 0; i < trj_sample_pred.size(); i++)
        {
            if (trj_sample_pred[i] == 0) //表示可通过
            {
                continue;
            }
            
            // 第i个预测结果对应的是点 trj_sample_point_->points[i]； 接下来在sample_point_info_中找出该点对应的轨迹编号
            for(auto &sample_point : sample_point_info_){
                float dis = sqrt(pow(trj_sample_point_->points[i].x - sample_point.first.x, 2) + 
                                pow(trj_sample_point_->points[i].y - sample_point.first.y, 2) +
                                pow(trj_sample_point_->points[i].z - sample_point.first.z, 2) );

                // std::cout << sample_point.first << " " << dis <<std::endl;
                if (dis < 0.00001 && trj_sample_point_->points[i].intensity == sample_point.first.intensity)
                {
                    for (auto i : sample_point.second)
                    {
                        trj_pred_cout[i] += 1;
                    }
                }
            }
        }

        std::vector<double> trj_pred_cost(trj_lab_info_.size()); //表征的是不通过的概率, 所以cost越小,越好 0-1
        //double min_cost = 1;
        //int min_index;
        for (size_t i = 0; i < trj_pred_cout.size(); i++)
        {
            if(trj_lab_info_[i].sample_point_num == 0){ //没有采样到的轨迹cost最大
                trj_pred_cost[i] = 1;
            }else
            {
                trj_pred_cost[i] = 1.0 * trj_pred_cout[i] / trj_lab_info_[i].sample_point_num;
            } 
            // std::cout << i << "  -- > " << trj_pred_cout[i] << "/" << trj_lab_info_[i].sample_point_num << " -- " << trj_pred_cost[i] << std::endl;
            //if (trj_pred_cost[i] <= min_cost)
            //{
            //    min_cost = trj_pred_cost[i];
            //    min_index = i;
            //}
        }

       

        // 发布当前选中的轨迹
        // 将轨迹库transform 到map下
        pcl::PointCloud<PointType>::Ptr trj_point_in_map (new pcl::PointCloud<PointType>());
        pcl::transformPointCloud (*trj_point_, *trj_point_in_map, transform);
        
        pcl::PointCloud<PointType>::Ptr pc_guide_line (new pcl::PointCloud<PointType>());
        cmd_mtx_.lock();
        pcl::copyPointCloud(*guide_line_ptr_, *pc_guide_line); 
        cmd_mtx_.unlock();

        // PublishPC(pc_temp_pub_, trj_point_in_map);


        //计算轨迹库每个点与引导线的距离
        pcl::KdTreeFLANN<PointType> kdtree;
        kdtree.setInputCloud(pc_guide_line);
        int K = 1;
        std::vector< std::vector<float> > trj_dis_cout(trj_lab_info_.size()); // 记录的是 轨迹编号与距离序列 ，例如： 轨迹编号1: --> {该轨迹每一个点与 引导线的夹角}
        for (size_t i = 0; i < trj_point_in_map->points.size(); i++)
        {
            PointType searchPoint;
            searchPoint.x = trj_point_in_map->points[i].x;
            searchPoint.y = trj_point_in_map->points[i].y;
            searchPoint.z = trj_point_in_map->points[i].z;
            searchPoint.intensity = trj_point_in_map->points[i].intensity;

            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            pcl::PointCloud<PointType>::Ptr local_target_pc (new pcl::PointCloud<PointType>());
            if ( kdtree.nearestKSearch (searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0 ){
                double dis_near_to_start = sqrt(pow(pc_guide_line->points[pointIdxNKNSearch[0] ].x -  pc_guide_line->points.front().x, 2) + 
                                                pow(pc_guide_line->points[pointIdxNKNSearch[0] ].y -  pc_guide_line->points.front().y, 2)
                                                );
                double dis_search_to_near = sqrt(pow(pc_guide_line->points[pointIdxNKNSearch[0] ].x -  searchPoint.x, 2) + 
                                                    pow(pc_guide_line->points[pointIdxNKNSearch[0] ].y -  searchPoint.y, 2)
                                                );
                float theta = atan2(dis_search_to_near, dis_near_to_start);
                trj_dis_cout[int(searchPoint.intensity)].push_back(theta);
                // trj_dis_point[int(searchPoint.intensity)].push_back(searchPoint);
            }
        }
        

        //计算原始距离代价
        std::vector<float> trj_dis_cost_ori(trj_lab_info_.size());
        float maxcost = 0.0;
        for (size_t i = 0; i < trj_dis_cost_ori.size(); i++)
        {
            if(trj_lab_info_[i].sample_point_num == 0){ //没有采样到的轨迹cost最大
                trj_dis_cost_ori[i] = 1;
            }else
            {
                std::vector<float>::iterator biggest = std::max_element(std::begin(trj_dis_cout[i]), std::end(trj_dis_cout[i]));
                trj_dis_cost_ori[i] = *biggest;

                if (trj_dis_cost_ori[i] > maxcost)
                {
                    maxcost = trj_dis_cost_ori[i];
                }
            }
        }

        //归一化距离代价
        std::vector<float> trj_dis_cost(trj_lab_info_.size());
        for (size_t i = 0; i < trj_dis_cout.size(); i++)
        {
            if(trj_lab_info_[i].sample_point_num == 0){ //没有采样到的轨迹cost最大
                trj_dis_cost[i] = 1;
            }else{
                trj_dis_cost[i] = 1.0 * trj_dis_cost_ori[i] / maxcost;
            }
            
        }
        
        // 打印 每条轨迹的距离代价
        std::cout << "--------- trj dis cost---------- " << std::endl;
        for(size_t i=0;  i< trj_dis_cost.size(); i++){
           std::cout << " " << i <<" " <<  trj_dis_cost[i] << "\t  ";
        }
        std::cout << std::endl;

        //std::cout << "-- maxcost " << maxcost <<std::endl;



        //计算总体代价 与 最优值 trj_dis_cost 与 trj_pred_cost  
        // trj_total_cost = cost_factor_ * trj_pred_cost + (1- cost_factor_) * trj_dis_cost
        float min_cost = 1;
        int min_index;

        std::vector<float> trj_total_cost(trj_lab_info_.size());
        for (size_t i = 0; i < trj_total_cost.size(); i++)
        {
            if(trj_lab_info_[i].sample_point_num == 0){
                trj_total_cost[i] = 1;
            }else
            {
                trj_total_cost[i] = cost_factor_ * trj_pred_cost[i] + (1- cost_factor_) * trj_dis_cost[i];
            }

            if (trj_total_cost[i] <= min_cost)
            {
               min_cost = trj_total_cost[i];
               min_index = i;
            }
        }

         std::cout << "min_index: " << min_index << " min_cost: " << min_cost  
                   << " min pred cost: " << trj_pred_cost[min_index] 
                   << " min trj cost: "  << trj_dis_cost[min_index] << std::endl;

         
        
        // 发布最优轨迹
        pcl::PointCloud<PointType>::Ptr this_trj_point (new pcl::PointCloud<PointType>());
        for (size_t i = 0; i < trj_point_in_map->points.size(); i++)
        {
            if (trj_point_in_map->points[i].intensity == min_index)
                {
                    this_trj_point->points.push_back(trj_point_in_map->points[i]);
                }
        }
        PublishPC(trj_pub_, this_trj_point);


        opti_trj_time.toc("=== [cmd thread] ==> opti trj time ");

        // 发布小车控制指令
        cmd_mtx_.lock();
        cmd_v_ = trj_lab_info_[min_index].v;
        cmd_w_ = trj_lab_info_[min_index].w;
        cmd_mtx_.unlock();
        real_time_pred.toc("=== [cmd thread] ==> real time navi ");

        // return;

        geometry_msgs::Twist cmd;
        cmd_mtx_.lock();
        cmd.linear.x = cmd_v_;
        cmd.angular.z = cmd_w_;
        cmd_mtx_.unlock();

        #ifdef CONTROL
        car_cmd_pub_.publish(cmd);
        #endif

        rate.sleep();
        ros::spinOnce();

        
    }
    
}

/**
 * @brief PublishOdomThread 从gazbeo中获取小车真值,　并发布 [bask_link -> map] 的tf转换
 */
void Navi::PublishOdomThread(){
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

    std::cout << "=== [Navi thread] ==> odom thread start " << std::endl;

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
void Navi::PublishPC(ros::Publisher pub, pcl::PointCloud<PointType>::Ptr pc){
    sensor_msgs::PointCloud2 tempMsgCloud;
    pcl::toROSMsg(*pc, tempMsgCloud);
    tempMsgCloud.header.stamp = ros::Time();
    tempMsgCloud.header.frame_id = pc_frame_id_;
    pub.publish(tempMsgCloud);
}


void Navi::LocalMapCallBack(const sensor_msgs::PointCloud2ConstPtr& msg){

    pcl::fromROSMsg(*msg, *map_ptr_);
    pc_frame_id_ = msg->header.frame_id;

    replan();
    // 预测
    // TicToc pre_time;
    // pcl::PointCloud<PointType>::Ptr process_pc (new pcl::PointCloud<PointType>());
    // process_pc = net_pred_->PointPred(gloabl_map_ptr_);
    // pre_time.toc("ped time :");

    // sensor_msgs::PointCloud2 tempMsgCloud;
    // pcl::toROSMsg(*process_pc, tempMsgCloud);
    // tempMsgCloud.header.stamp = ros::Time();
    // tempMsgCloud.header.frame_id = "map";
    // local_process_pub_.publish(tempMsgCloud);
}


void Navi::GoalCallBack(const geometry_msgs::PointStamped::ConstPtr &msg){
    geometry_msgs::Point p = msg->point;
    if (!point_equal(p, pre_goal_point_))
    {
        goal_point_ = p;
        std::cout << "=== [Navi] ==>  Goal point set to: " << p.x << " " << p.y << " " << p.z << std::endl;
        pre_goal_point_ = p;
        set_goal_ = true;

        mk_.pose.position.x = goal_point_.x;
        mk_.pose.position.y = goal_point_.y;
        mk_.pose.position.z = goal_point_.z;
        vis_goal_pub_.publish(mk_);

        //清空小车轨迹 每设定一次目标点，就重新计数轨迹
        car_pose_for_vis_->clear();

        plan();
        
    }
    
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "navi");
    ros::NodeHandle nh;
    ros::NodeHandle nh_("~");

    Navi navi(nh, nh_);

    std::thread odom_thread(&Navi::PublishOdomThread, &navi);

    std::thread cmd_thread(&Navi::PublishCmdThread, &navi);

    
    ros::spin();
    return 0;

}