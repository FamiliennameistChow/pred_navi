#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: main.py --- > model.py.py
# Author: bornchow
# Time:20210817
# Modify: 主体网络修改为resnet18,(修订网络输入层为2通道，输出层接一个线性层和Sigmod()层) 20210901
#         修订主体网络由回归预测改为分类预测 20210913
# ------------------------------------

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models


class ElevaModel(nn.Module):
    def __init__(self):
        super(ElevaModel, self).__init__()

        self.model = models.resnet18(pretrained=True)
        # self.model.fc = nn.Linear(512, 2)
        self.model.conv1 = nn.Conv2d(1, 64, 7, stride=3, padding=3, bias=False)
        # 回归预测
        # self.linear = nn.Linear(1000, 1)
        # self.sig = nn.Sigmoid()
        # 分类预测
        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.model(x)
        x = self.dropout(self.fc1(x))
        x = self.dropout(self.fc2(x))
        # return x
        return F.log_softmax(x, dim=1)
        # out = self.sig(x)
        # return out


class ElevaModelSelf(nn.Module):
    def __init__(self):
        super(ElevaModelSelf, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 5, padding=2),
            nn.BatchNorm2d(16),
            nn.Flatten(),
            nn.Linear(6400, 512),
            nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 128),
            nn.Dropout(0.5),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)


class PointModel(nn.Module):
    def __init__(self, k=2):
        super(PointModel, self).__init__()
        # mlp  当kernel_size=1，stride=1，padding=0时，每个卷积核计算后输出数据和输入数据的长度相同
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        # mlp
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        # fc
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k)
        #
        self.relu = nn.ReLU()
        # bn
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.dropout(self.fc1(x))))
        x = F.relu(self.bn5(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


if __name__ == '__main__':
    from torchsummary import summary
    device = torch.device("cpu")

    TEST_MODEL = "cnn_resnet"
    # TEST_MODEL = "cnn_self"
    # TEST_MODEL = "point"

    if TEST_MODEL == "cnn_resnet":

        # // ------ 检测resnet CNN网络的大小 -----
        print("cnn resnet model test ......")
        input = torch.rand((64, 1, 20, 20))
        print(input.size())
        # my_net = models.resnet18(pretrained=False).conv1
        my_net = ElevaModel()
        my_net = my_net.eval()
        print(my_net)
        out = my_net(input)
        #
        print("size ", out.shape)
        print("out ", out)

        summary(my_net, input_size=(1, 20, 20), device=device.type)

    elif TEST_MODEL == "cnn_self":

        print("cnn self model test ......")
        # //// ------ 检测自定义CNN网络的大小 -----
        my_net = ElevaModelSelf()
        my_net = my_net.eval()
        out = my_net(input)
        print("size ", out.shape)
        summary(my_net, input_size=(1, 20, 20), device=device.type)

    elif TEST_MODEL == "point":

        print("cpoint model test ......")
        # ////-----检测点云网络的大小-------
        point_net = PointModel()
        print(point_net)
        summary(point_net, input_size=(3, 500), device=device.type)

    # net = models.squeezenet1_1(pretrained=True)
    # net.classifier[1] = nn.Conv2d(in_channels=512, out_channels=2, kernel_size=(1, 1), stride=(1, 1))
    # net.num_classes = 2
    # print(net)

    # vgg16_model = models.vgg16(pretrained=False)
    # print(vgg16_model)
    # summary(vgg16_model, input_size=(3, 512, 512), device=device.type)


