#ifndef _COMMON_H_
#define _COMMON_H_


#include <iostream>
#include <vector>
#include <math.h>

//pcl
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/random_sample.h>
#include <pcl/common/transforms.h>
#include <pcl/common/common.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>

//opencv
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//octomap
#include <octomap/octomap.h>

using namespace std;
typedef pcl::PointXYZI  PointType;
typedef octomap::OcTree OcTreeT;

struct Bbox_AABB{
    PointType min_point;
    PointType max_point;
};

struct TRJ_INFO{
    int trj_num; //轨迹编号
    float v; //轨迹对应的线速度
    float w; //轨迹对应的角速度
    int sample_point_num; //该轨迹上的采样点数
};


std::vector<std::string> boostsplit(const std::string& input, string str)
{
    std::vector <std::string> fields;
    boost::split( fields, input, boost::is_any_of( str ) );
    return fields;
}



// 为 std::unordered_map<PointType, std::vector<int> >  
namespace std{
    template<>
    struct hash<PointType>{//哈希的模板定制
    public:
        size_t operator()(const PointType &pt) const 
        {
            return hash<double>()(pt.x) ^ hash<double>()(pt.y) ^ hash<double>()(pt.z) ^ hash<double>()(pt.intensity);
        }
        
    };
    
    template<>
    struct equal_to<PointType>{//等比的模板定制
    public:
        bool operator()(const PointType &p1, const PointType &p2) const
        {
            return (p1.x - p2.x) < 0.0000001 && (p1.y - p2.y) < 0.0000001 && (p1.z - p2.z) < 0.0000001 && (p1.intensity -p2.intensity) < 0.1;
        }
        
    };
}

#endif
