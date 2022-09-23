#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/geometry.h>

// #include <iostream>
// #include <fstream>

#include <octomap/octomap.h>
typedef pcl::PointXYZI  PointType;
typedef octomap::OcTree OcTreeT;
using namespace std;

void pc_to_eleva(pcl::PointCloud<PointType>::Ptr pc, shared_ptr<octomap::OcTree> &map, double res, double time){
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

int main(int argc, char **argv)
{
    ros::init(argc, argv, "test_octomap");
    ros::NodeHandle nh;
    ros::NodeHandle nh_("~");
    shared_ptr<octomap::OcTree> map;
    pcl::PointCloud<PointType>::Ptr cloud (new pcl::PointCloud<PointType>);

    map.reset(new OcTreeT(0.1));
    
    if (pcl::io::loadPCDFile<PointType> ("/home/bornchow/ROS_WS/slam_ws/src/data_make/data/1517155865.561851.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file scene.pcd \n");
        return (-1);
    }
    cout << "11111" << endl;
    pc_to_eleva(cloud, map);


    ros::spin();
    return 0;

}