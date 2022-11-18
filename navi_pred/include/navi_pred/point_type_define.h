/**************************************************************************
 * point_type_define.h
 * 
 * @Author： bornchow
 * @Date: 2021.10.08
 * 
 * @Description:
 *  https://blog.csdn.net/zhanghm1995/article/details/108241524
 * 
 ***************************************************************************/
//

#ifndef PCD_READ_POINT_TYPE_DEFINE_H
#define PCD_READ_POINT_TYPE_DEFINE_H

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>

#define PCL_NO_PRECOMPILE

namespace perception {

    struct EIGEN_ALIGN16 _PointXYZIYaw
    {
        PCL_ADD_POINT4D;                  // preferred way of adding a XYZ+padding
        float intensity;
        float yaw;
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
    };

/**
 * @brief A point structure representing Euclidean xyz coordinates, and the ring value.
 */
    struct EIGEN_ALIGN16 PointXYZIYaw : public _PointXYZIYaw
    {
        inline PointXYZIYaw (const _PointXYZIYaw &p)
        {
            x = p.x; y = p.y; z = p.z; data[3] = 1.0f;
            intensity = p.intensity;
            yaw = p.yaw;
        }

        inline PointXYZIYaw ()
        {
            x = y = z = 0.0f;
            data[3] = 1.0f;
            intensity = 0;
            yaw = 0;
        }

        inline PointXYZIYaw (float _x, float _y, float _z, float _intensity, float _yaw)
        {
            x = _x; y = _y; z = _z;
            data[3] = 1.0f;
            intensity = _intensity;
            yaw = _yaw;
        }

        friend std::ostream& operator << (std::ostream& os, const PointXYZIYaw& p)
        {
            os << "(" << p.x << "," << p.y << "," << p.z << " - " <<p.intensity << " " << p.yaw << ")";
            return (os);
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

} // namespace perception

POINT_CLOUD_REGISTER_POINT_STRUCT(perception::PointXYZIYaw,
                                  (float, x, x)
                                          (float, y, y)
                                          (float, z, z)
                                          (float, intensity, intensity)
                                          (float, yaw, yaw)
)


#endif //PCD_READ_POINT_TYPE_DEFINE_H
