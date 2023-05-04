#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > train.py.py
# Author: bornchow
# Date:20210818
#
# ------------------------------------
import copy
import os.path

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from Dataset import ElevaData
import os
from model import ElevaModel, ElevaModelSelf
import math
import time
import sys
from common import Logger
from torchsummary import summary

TRAIN_REMOTE = False

if TRAIN_REMOTE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

save_name = time.strftime("%m%d%H%M", time.localtime())

# parameters
BATCH_SIZE = 128
LR = 0.000005  # 0.000005
EPOCHS = 2000
# MODEL = "resnet18"
MODEL = "self"

save_dir = os.path.join(os.getcwd(), "train_logs_" + MODEL + "_"+save_name)

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if TRAIN_REMOTE:
    train_data_dir = "./data_22_20210925/dataSet"
    label_file = "std.md"
    test_data_dir = "./data_22_20210925/dataTest"
else:
    train_data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22_20210925/dataSet"
    label_file = "std.md"
    test_data_dir = "/home/bornchow/ROS_WS/slam_ws/src/data_make/data_22_20210925/dataTest"

# dataloader
train_dataset = ElevaData(train_data_dir, label_file, transform=transforms.ToTensor())
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = ElevaData(test_data_dir, label_file, transform=transforms.ToTensor())
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

train_dataset_size = len(train_dataset)
test_dataset_size = len(test_dataset)

# 训练信息:
os.mkdir(save_dir)
log_file = os.path.join(save_dir, "log.txt")
sys.stdout = Logger(log_file)
print("================= info ============================")
print("BATCH_SIZE: {} \nLR        : {} \nEPOCHS    : {} \nmodel     : {}\n".
      format(BATCH_SIZE, LR, EPOCHS, MODEL))
print("Learning rate method:  StepLR")
print("train_dataset      : ", train_data_dir)
print("train_dataset_size : ", train_dataset_size)
print("test_dataset       : ", test_data_dir)
print("test_dataset_size  : ", test_dataset_size)

# 定义model
if MODEL == "self":
    net = ElevaModelSelf().to(device=device)
elif MODEL == "resnet18":
    net = ElevaModel().to(device=device)

print("========= model =========\n", net)
print("========= model summary =========")
summary(net, input_size=(1, 20, 20), device=device.type)

# loss_fn = nn.CrossEntropyLoss().to(device=device)
# loss_fn = nn.SmoothL1Loss().to(device=device)
loss_fn = nn.NLLLoss(reduction="sum")

# optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=100, gamma=0.95)


# #set param for train data
total_train_step = 0
total_test_step = 0
loss_min = 3000

# 添加tensorboard
writer = SummaryWriter(save_dir)

for epoch in range(EPOCHS):
    total_train_loss = 0.0
    total_train_acc = 0

    for data in train_dataloader:
        img, target = data
        img = img.type(torch.FloatTensor)  # np.array，然后默认保存成float64，但是pytorch中默认是float32
        img = img.to(device)
        # print("img: ", img.shape, " ", img.type(), img)
        # label = label.to(device).float().view([-1, 1])
        target = target.to(device).squeeze().long()
        # print("label: ", label.shape, " ", label.type())
        optimizer.zero_grad()
        net.train()
        output = net(img)
        # print("output: ", output.shape)
        loss = loss_fn(output, target)
        # print("-----loss-----:", loss.item())
        # 优化模型
        loss.backward()
        optimizer.step()

        total_train_step += 1
        total_train_loss += loss
        # 计算准确率，详看test_acc.py
        acc = (output.data.max(1)[1] == target).sum().item()
        total_train_acc += acc
        scheduler.step()

    per_train_loss = total_train_loss / (math.ceil(len(train_dataset) / BATCH_SIZE))
    writer.add_scalar("train loss",  per_train_loss, epoch)
    # writer.add_scalar("train loss", total_train_loss, epoch)

    # test
    # 每训练一轮进行测试
    total_test_loss = 0
    total_test_acc = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.type(torch.FloatTensor)  # np.array，然后默认保存成float64，但是pytorch中默认是float32
            imgs = imgs.to(device)
            # targets = targets.to(device).float().view([-1, 1])
            targets = targets.to(device).squeeze().long()
            net.eval()  # if they are affected, e.g. Dropout, BatchNorm,
            outputs = net(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss

            # 计算准确率，详看test_acc.py
            acc = (outputs.data.max(1)[1] == targets).sum().item()
            total_test_acc += acc

    # scheduler.step()
    per_test_loss = total_test_loss / (math.ceil(len(test_dataset) / BATCH_SIZE))
    writer.add_scalar("val loss", per_test_loss, epoch)
    # writer.add_scalar("val loss", total_test_loss, epoch)

    print("epoch {}/{}  train loss : {:.6f} train acc: {:.6f} || val loss: {:.6f} test acc: {:.6f}".format(epoch,
                                                                          EPOCHS,
                                                                  per_train_loss,
                                            total_train_acc / train_dataset_size,
                                                                   per_test_loss,
                                            total_test_acc / test_dataset_size))
    # print("epoch {}/{}  train loss : {} val loss: {} test acc: {}".format(epoch, EPOCHS,
    #                                                                       total_train_loss, total_test_loss,
    #                                                                       total_acc / test_dataset_size))

    if (epoch > 100) and (epoch != 0):
        if per_train_loss < loss_min:
            loss_min = per_train_loss
            torch.save(net.state_dict(), os.path.join(save_dir,
                                                      "model_{}_{:.6f}.pth".format(epoch, per_train_loss)))
            print("===========save: model_{}_{:.6f}.pth".format(epoch, per_train_loss))
