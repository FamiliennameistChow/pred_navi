#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > show_point_dataset.py
# Author: bornchow
# Date:20210924
# 显示点云数据集中的点云
# ------------------------------------
import numpy as np
import open3d as o3d
from Dataset import PointData
from torch.utils.data import DataLoader
import os

BATCH_SIZE = 1
SAMPLE_POINTS = 600



'''
show dataset point in open3d
'''
# train_data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22_modify/dataSet"
# label_file = "std.md"
#
# train_dataset = PointData(train_data_dir, label_file, SAMPLE_POINTS)
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
#
# pts, label = train_dataset[529]
#
# print(pts.shape)
# pts_np = pts.numpy()
# print(pts_np.shape)
#
# pointCloud = o3d.geometry.PointCloud()
# pointCloud.points = o3d.utility.Vector3dVector(pts_np)
# FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
#
# o3d.visualization.draw_geometries([pointCloud, FOR1])


'''
show point 
'''

pts_file = "309.112000.pcd"

SAMPLE_POINTS = 700
train_data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22_20210925/dataTest/"

pcd = o3d.io.read_point_cloud(os.path.join(train_data_dir, pts_file), format="pcd")
print(pcd)

pts_np = np.asarray(pcd.points)


choice = np.random.choice(len(pcd.points), SAMPLE_POINTS, replace=True)
# resample
point_set = pts_np[choice, :]
# normalize
# point_set = point_set - np.expand_dims(np.array([0, 0, np.mean(point_set, axis=0)[2]]), 0)  # 将点云放在z轴0位置

point_set2 = point_set - np.expand_dims(np.array([np.mean(point_set, axis=0)[0], np.mean(point_set, axis=0)[1], -0.75]), 0)

pointCloud = o3d.geometry.PointCloud()
pointCloud.points = o3d.utility.Vector3dVector(point_set)

pointCloud2 = o3d.geometry.PointCloud()
pointCloud2.points = o3d.utility.Vector3dVector(point_set2)
pointCloud2.paint_uniform_color([0.4, 1, 0.18431])

FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

# o3d.visualization.draw_geometries([pointCloud, pointCloud2, FOR1])

o3d.visualization.draw_geometries([pointCloud2, FOR1])
