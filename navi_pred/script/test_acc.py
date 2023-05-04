#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > test_acc.py.py
# Author: bornchow
# Date:20210920
#
# ------------------------------------

from model import ElevaModel
import torch
from torch.utils.data import DataLoader
from Dataset import ElevaData
from torchvision import transforms

BATCH_SIZE = 10

train_data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22_20210925/dataSet"
label_file = "std.md"


# dataloader
train_dataset = ElevaData(train_data_dir, label_file, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

my_net = ElevaModel()

for data in train_dataloader:
    img, target = data
    target = target.squeeze()
    print("input data size: ", img.shape)
    print("target after view: ", target.shape)
    my_net.train()
    out = my_net(img)
    print("out shape: ", out.shape)

    # cal acc
    print("out ", out)
    print("target ", target)

    # pred = out.argmax(dim=1, keepdim=True)
    # print("pred ", pred)
    # print("target ", target.view_as(pred))
    # acc = pred.eq(target.view_as(pred)).sum().item()
    # print(acc)

    print("pred: ", out.data.max(1)[1])

    acc = (out.data.max(1)[1] == target).sum().item() / BATCH_SIZE
    print(acc)

    break




