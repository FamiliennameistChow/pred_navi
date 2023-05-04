#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > predict_point_mutil.py
# Author: bornchow
# Date:20210923
# 推理局部点云的代码
# ------------------------------------
import numpy as np
import open3d as o3d
import torch
from model import PointModel
import copy
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SHOW_SIGLE = False
CONVERT_MODEL = False

def create_bboxs(aabb, stride, kernel):
    bboxs = []
    centers = []
    center = [0]*3
    range_j = round((aabb.max_bound[1] - aabb.min_bound[1] - kernel)/stride)
    range_i = round((aabb.max_bound[0] - aabb.min_bound[0] - kernel)/stride)
    print(range_i, range_j)
    for j in range(range_j+1):
        for i in range(range_i+1):
            bbox = o3d.geometry.AxisAlignedBoundingBox()
            center[0] = aabb.min_bound[0] + kernel / 2 + stride * i
            center[1] = aabb.min_bound[1] + kernel / 2 + stride * j
            center[2] = (aabb.min_bound[2] + aabb.max_bound[2]) / 2
            bbox.min_bound = [center[0] - kernel / 2, center[1] - kernel / 2, center[2] - 2]
            bbox.max_bound = [center[0] + kernel / 2, center[1] + kernel / 2, center[2] + 2]
            centers.append(center)
            bboxs.append(bbox)

    return bboxs

# 处理局部点云:
#https://blog.csdn.net/qq_39311949/article/details/119802507?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EsearchFromBaidu%7Edefault-1.searchformbaiduhighlight&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EsearchFromBaidu%7Edefault-1.searchformbaiduhighlight
# pcd = o3d.io.read_point_cloud("/home/bornchow/ROS_WS/slam_ws/src/data_make/data/48.368000.pcd")

pcd = o3d.io.read_point_cloud("/home/bornchow/workFile/test/pcl_test/pcd/0.pcd")
print(pcd)

# print(np.asarray(pcd.points))
aabb = pcd.get_axis_aligned_bounding_box()
# aabb.color = (1, 0, 0)
# print(aabb)
# print(aabb.min_bound)

# # crop data
# # https://github.com/isl-org/Open3D/issues/1410
# bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-5.04305, -4.99705, -1.65826), max_bound=(0, 0, 0))
# bbox.color = (0, 1, 0)
# crop = pcd.crop(bbox)
# point_set_np = np.asarray(crop.points)
# print(point_set_np.shape)
#
# # o3d.visualization.draw_geometries([crop, pcd, aabb, bbox])


bboxs = create_bboxs(aabb, 0.5, 2.0)
print("bboxs size:", len(bboxs))
FOR1 = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

if SHOW_SIGLE:
    index = 15
    crop = pcd.crop(bboxs[index])

    # bbox = o3d.geometry.AxisAlignedBoundingBox([-1, -1, -2], [1, 1, 2])
    # crop = pcd.crop(bbox)

    pts_set = np.asarray(crop.points)

    # cents = [0]*3
    # cents[0] = (bboxs[index].min_bound[0] + bboxs[index].max_bound[0])/2.0
    # cents[1] = (bboxs[index].min_bound[1] + bboxs[index].max_bound[1])/2.0
    # cents[2] = np.mean(pts_set, axis=0)[2]
    # print(cents)

    # pts = pts_set - np.expand_dims(np.array(cents), 0)

    pts = pts_set - np.expand_dims(np.mean(pts_set, axis=0), 0)  # center

    crop_normal = o3d.geometry.PointCloud()
    crop_normal.points = o3d.utility.Vector3dVector(pts)
    # crop_normal.paint_uniform_color([0, 0, 1.0])

    choice = np.random.choice(pts.shape[0], 500, replace=True)
    pts = pts[choice, :]

    pts = np.expand_dims(pts, 0)
    print(pts.shape)

    point_set_tensor = torch.from_numpy(pts)
    pts_tensor = point_set_tensor.transpose(2, 1).to(device=device).float()  # (bash_size * channel * points_num)

    # model_dir = "/home/bornchow/workFile/navi_net/model/model_1576_25.226139.pth"
    model_dir = "/home/bornchow/workFile/navi_net/model/model_3210_0.9759_0.9509.pth"

    net = PointModel().to(device=device)
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    out = net(pts_tensor)

    print("predict: ", out.data.max(1)[1])

    o3d.visualization.draw_geometries([crop_normal, pcd, FOR1, bboxs[index]])
else:
    time_start = time.time()
    com = np.ones((len(bboxs), 500, 3))
    # print(com[1, :, :].shape)
    for i in range(len(bboxs)):
        crop = pcd.crop(bboxs[i])
        if len(crop.points) == 0:
            continue
        point_set_np = np.asarray(crop.points)
        # print("bbox point: ", point_set_np.shape)

        # 将点云转换到小车坐标系中
        # cents = [0] * 3
        # cents[0] = (bboxs[i].min_bound[0] + bboxs[i].max_bound[0]) / 2.0
        # cents[1] = (bboxs[i].min_bound[1] + bboxs[i].max_bound[1]) / 2.0
        # cents[2] = np.mean(point_set_np, axis=0)[2] + 0.8

        # point_set_np = point_set_np - np.expand_dims(np.array(cents), 0)
        point_set_np = point_set_np - np.expand_dims(np.mean(point_set_np, axis=0), 0)  # center

        choice = np.random.choice(point_set_np.shape[0], 500, replace=True)
        point_set = point_set_np[choice, :]

        com[i, :, :] = point_set

        print(point_set)

    point_set_tensor = torch.from_numpy(com)
    pts_tensor = point_set_tensor.transpose(2, 1).to(device=device).float()  # (bash_size * channels * points_num)
    print("input size: ", pts_tensor.size())

    # model_dir = "/home/bornchow/workFile/navi_net/model/model_1576_25.226139.pth"
    model_dir = "/home/bornchow/workFile/navi_net/model/model_3210_0.9759_0.9509.pth"

    net = PointModel().to(device=device)
    net.load_state_dict(torch.load(model_dir))
    net.eval()

    out = net(pts_tensor)

    if CONVERT_MODEL:
        print(" ======== convert model to libtorch ====================")
        traced_script_module = torch.jit.trace(net, pts_tensor)
        output = traced_script_module(pts_tensor)
        traced_script_module.save("./model/model_3210_0.9759_0.9509.pt")
        print("PredCov: ",  output.data.max(1)[1].tolist())

    pred = out.data.max(1)[1].tolist()
    print("predict: ", pred)
    time_end = time.time()
    print("predict time: ", time_end - time_start)

    # 给点云上色
    color = np.array([0.4, 1, 0.18431], dtype=float)
    colors = np.expand_dims(color, axis=0).repeat(len(pcd.points), axis=0)

    for i in range(len(bboxs)):
        if pred[i] == 1:
            idx = bboxs[i].get_point_indices_within_bounding_box((pcd.points))
            bboxs[i].color = (1, 0, 0)
            colors[idx] += [0.1, -0.1, 0.0]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd, FOR1])


# crop2 = pcd.crop(bboxs[0])
# point_set_np2 = np.asarray(crop2.points)
# choice = np.random.choice(point_set_np2.shape[0], 500, replace=True)
# point_set2 = point_set_np2[choice, :]
# print("point_set_np2: ", point_set2.shape)
#
# crop3 = pcd.crop(bboxs[3])
# point_set_np3 = np.asarray(crop3.points)
# choice = np.random.choice(point_set_np3.shape[0], 500, replace=True)
# point_set3 = point_set_np3[choice, :]
# print("point_set_np3: ", point_set3.shape)








