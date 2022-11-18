#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_pred  --- > cul_traj_len.py
# Author: bornchow
# Date:20211217
# 计算小车轨迹长度
# 订阅小车发布的关键位置点 点云
# ------------------------------------
import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import math
import copy

p_last = list()


def on_new_point_cloud(data):
    p_last_x = p_last_y = p_last_z = 0.0
    trj_leng = 0.0 
    pc = pc2.read_points(data, skip_nans=True, field_names=("x", "y", "z"))
    for p in pc:
        if trj_leng == 0.0:
            p_last_x = p[0]
            p_last_y = p[1]
            p_last_z = p[2]
            trj_leng += 0.1
        else:
            trj_leng += math.pow(math.pow(p[0] - p_last_x ,2) + math.pow(p[1] - p_last_y ,2) , 0.5)
            p_last_x = p[0]
            p_last_y = p[1]
            p_last_z = p[2]

    print("leng: " , trj_leng)

    # p = pcl.PointCloud()
    # p.from_list(pc_list)
    # seg = p.make_segmenter()
    # seg.set_model_type(pcl.SACMODEL_PLANE)
    # seg.set_method_type(pcl.SAC_RANSAC)
    # indices, model = seg.segment()

rospy.init_node('listener', anonymous=True)
rospy.Subscriber("/car_key_pose", PointCloud2, on_new_point_cloud)
rospy.spin()