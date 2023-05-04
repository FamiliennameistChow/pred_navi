#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > train_pts.py
# Author: bornchow
# Date:20210915
#   训练pointNet网络
# ------------------------------------


import os.path
import math
import time
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from Dataset import PointData
from model import PointModel
from common import Logger

save_name = time.strftime("%m%d%H%M", time.localtime())
save_dir = os.path.join(os.getcwd(), "train_logs_"+save_name)

# parameters
BATCH_SIZE = 64
LR = 0.00001
EPOCHS = 20000
MODEL = "pointNet_self"
SAMPLE_POINTS = 500

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22_modify/dataSet"
label_file = "std.md"

test_data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22_modify/dataTest"


# dataloader
train_dataset = PointData(train_data_dir, label_file, SAMPLE_POINTS)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = PointData(test_data_dir, label_file, SAMPLE_POINTS)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)


# 训练信息:
os.mkdir(save_dir)
log_file = os.path.join(save_dir, "log.txt")
sys.stdout = Logger(log_file)

print("================= info ============================")
print("BATCH_SIZE: {} \nLR        : {} \nEPOCHS    : {} \nmodel     : {}\nsample_points       :{}".
      format(BATCH_SIZE, LR, EPOCHS, MODEL, SAMPLE_POINTS))
print("train_dataset      : ", train_data_dir)
print("train_dataset_size : ", train_dataset_size)
print("test_dataset       : ", test_data_dir)
print("test_dataset_size  : ", test_dataset_size)

# 定义网络
net = PointModel(k=2).to(device=device)

print("========= model =========\n", net)
print("========= model summary =========")
summary(net, input_size=(3, SAMPLE_POINTS), device=device.type)
# sys.stdout.close()

# 优化器：adam-Adaptive Moment Estimation(自适应矩估计)，利用梯度的一阶矩和二阶矩动态调整每个参数的学习率
# betas：用于计算梯度一阶矩和二阶矩的系数
optimizer = torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.999))
# 学习率调整：每个step_size次epoch后，学习率x0.5
# scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
scheduler = CosineAnnealingLR(optimizer, T_max=32)


# #set param for train data
total_train_step = 0
total_test_step = 0
loss_min = 3000

# 添加tensorboard
writer = SummaryWriter(save_dir)
total_train_step = 0

for epoch in range(EPOCHS):
    total_train_loss = 0.0
    total_train_acc = 0
    for data in train_dataloader:
        total_train_step += 1
        points, label = data
        points = points.transpose(2, 1).to(device=device)
        label = label[:, 0].to(device=device)
        # print(points.size(), label.size())
        # 计算loss并进行反向传播
        optimizer.zero_grad()
        net = net.train()
        pred = net(points)
        loss = F.nll_loss(pred, label)  # 损失函数：负log似然损失，在分类网络中使用了log_softmax，二者结合其实就是交叉熵损失函数
        loss.backward()
        optimizer.step()
        # 计算acc
        pred_choice = pred.data.max(1)[1] # max(1)返回每一行中的最大值及索引,[1]取出索引（代表着类别）
        correct = (pred_choice == label).sum().item()  # 判断和target是否匹配，并计算匹配的数量
        total_train_loss += loss
        total_train_acc += correct
        scheduler.step()

    acc_train = 1.0 * total_train_acc / train_dataset_size

    writer.add_scalar("train loss", total_train_loss, epoch)

    with torch.no_grad():
        total_test_loss = 0.0
        total_test_acc = 0
        for data in test_dataloader:
            points, label = data

            points = points.transpose(2, 1).to(device)
            label = label[:, 0].to(device)
            # print(points.size(), label.size())

            net = net.eval()
            pred = net(points)
            loss_test = F.nll_loss(pred, label)
            pred_choice = pred.data.max(1)[1]
            correct_test = pred_choice.eq(label.data).cpu().sum().item()
            total_test_loss += loss_test
            total_test_acc += correct_test

    acc_test = 1.0 * total_test_acc / test_dataset_size
    writer.add_scalar("val loss", total_test_loss, epoch)

    print('[%d / %d] train loss: %f accuracy: %f == test loss: %f accuracy: %f ' %
          (epoch, EPOCHS, total_train_loss, acc_train, total_test_loss, acc_test))

    if (epoch > 100) and (epoch != 0):
        if total_train_loss < loss_min:
            loss_min = total_train_loss
            torch.save(net.state_dict(), os.path.join(save_dir,
                                                      "model_{}_{:.6f}.pth".format(epoch, total_train_loss.item())))
            print("===========save: model_{}_{:.6f}.pth".format(epoch, total_train_loss.item()))


