#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: main.py --- > ElevaDataset.py.py
# Author: bornchow
# Time:20210817
# Modify:
#  1. 获取高程图的dataset 数据存在data文件夹中， label存在data文件夹下的std.md中
#  2. 修正高程图，将一通道高程图转为二通道, 详细见<1>处理思路: 20210826
#  3. 添加制作点云数据集代码 20210905
#  4. 修订高程图dataset为分类数据集 20210913
#  5. 取消高程图的归一化操作,直接将高程图输入 20211119
# ------------------------------------

import torch

from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
from torchvision import transforms


class ElevaData(Dataset):
    '''
    制作高程数据集代码
    '''
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.label_file = os.path.join(data_dir, label_file)
        self.img_files_list = []
        self.label_dict = {}
        self.transform = transform

        files = os.listdir(self.data_dir)
        for file in files:
            if file.split(".")[-1] == "png":
                self.img_files_list.append(file)

        with open(self.label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                self.label_dict[line.split(" ")[0]] = line.split(" ")[-1]

        # print(self.label_file)
        # print(self.img_files_list)
        # print(self.label_dict)

    def __getitem__(self, item):
        img_name = self.img_files_list[item]
        # print("img name: ", img_name)
        img_dir = os.path.join(self.data_dir, img_name)
        img = cv2.imread(img_dir, -1)
        # print("before process img: ", img)

        # <2>处理思路:
        # 由于采集的高程图使用uint16格式编码, 65535表示高程无效值
        # 构建一个三维数组 例如(20*10*1)，　将高程值转换为真实高程单位为mm(减去32768), 并将高程无效值设置为0， 单位转换为cm
        mask = np.ones(img.shape, dtype=np.int16) * 32768
        img_out = img - mask
        img_out[img_out == 32767] = 0  # 将高程无效值设置为0

        img_out = img_out * 0.1  # 单位换算为cm

        # print(img_out)

        label_index = str(img_name.split(".")[0]) + "." + str(img_name.split(".")[1]) + ".txt"
        # label = int(float(self.label_dict[label_index])*10) # 分为10类
        # label = float(self.label_dict[label_index])  # 回归预测
        cls = 1 if (float(self.label_dict[label_index]) > 0.8) else 0
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        if self.transform:
            img_out = self.transform(img_out)
        return img_out, cls

    def __len__(self):
        return len(self.img_files_list)


class PointData(Dataset):
    def __init__(self, data_dir, label_file, sample_num):
        self.data_dir = data_dir
        self.label_file = os.path.join(data_dir, label_file)
        self.pts_file_list = []
        self.label_dict = {}
        self.npoints = sample_num

        files = os.listdir(self.data_dir)

        for file in files:
            if file.split(".")[-1] == "pcd":
                self.pts_file_list.append(file)

        # print(self.pts_file_list)

        with open(self.label_file, "r") as f:
            lines = f.readlines()
            for line in lines:
                id = str(line.split(" ")[0].split(".")[0] + "." + line.split(" ")[0].split(".")[1] + "." + "pcd")
                self.label_dict[id] = line.split(" ")[-1]

        # print(self.label_dict)

    def __getitem__(self, item):
        pts_file = self.pts_file_list[item]
        # print("pcd file name: ", pts_file)
        with open(os.path.join(self.data_dir, pts_file), "r") as f:
            lines = f.readlines()
            pts_num = len(lines) - 11
            pts = np.zeros((pts_num, 3), dtype=np.float32)
            i = 0
            for line in lines[11:]:
                pts[i] = [line.split(" ")[0], line.split(" ")[1], line.split(" ")[2]]
                i += 1

        choice = np.random.choice(pts_num, self.npoints, replace=True)
        # resample
        point_set = pts[choice, :]

        # normalize no test
        # point_set = point_set - np.expand_dims(np.array([0, 0, np.mean(point_set, axis=0)[2]]), 0)  # 将点云放在z轴0位置
        # dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        # point_set = point_set / dist  # scale

        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)

        point_set = torch.from_numpy(point_set)
        cls = 1 if (float(self.label_dict[pts_file]) > 0.8) else 0
        cls = torch.from_numpy(np.array([cls]).astype(np.int64))

        return point_set, cls

    def __len__(self):
        return len(self.pts_file_list)


if __name__ == '__main__':

    # 测试高程数据集
    data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22_20210925/dataSet"
    label_file = "std.md"

    my_dataset = ElevaData(data_dir, label_file, transform=transforms.ToTensor())
    # print(my_dataset[0])
    print(len(my_dataset))
    img, label = my_dataset[0]

    print("=======111111======================")

    print(label.shape)
    print(img)

    print("=============================")
    # show eleva img
    import matplotlib.pyplot as plt
    show_data = ElevaData(data_dir, label_file)
    img, label = show_data[0]
    print("=======111111======================")
    print(label.shape)
    print(img)
    plt.imshow(img)
    plt.show()

    # train_dataloader = DataLoader(my_dataset, batch_size=3, shuffle=True)

    # for data in train_dataloader:
    #     img, label = data
    #     print("--", label)
    #     label = label.squeeze()
    #     print(label.type())


    # 测试点云数据集
    # data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22/dataSet"
    # label_file = "std.md"
    # my_dataset = PointData(data_dir, label_file, 300)
    # points, target = my_dataset[0]
    # print(points.size(), points.type())
    # print(target.size(), target.type())
    #
    # # 分析数据集中正负样本
    # count, count_neg = 0, 0
    # for i in range(len(my_dataset)):
    #     points, label = my_dataset[i]
    #     if label.data == 1:
    #         count += 1
    #     else:
    #         count_neg += 1
    #
    # print("neg data num: ", count)
    # print("positive data num: ", count_neg)

    # target = target[:, 0]
    # print(target)





