#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > predict.py.py
# Author: bornchow
# Date:20210902
# 利用训练网络进行推理CNN模型
# ------------------------------------

import torch
import torchvision
from torch import nn
from torchvision import transforms

from model import ElevaModel
import cv2
import numpy as np
# 定义网络


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# resnet18模型
net = ElevaModel().to(device=device)
net.load_state_dict(torch.load("train_logs_09011720/model_7600_0.0017503430135548115.pth"))
net.eval()

#img = cv2.imread("/home/bornchow/ROS_WS/slam_ws/src/data_make/data_real/1517155912.560805.png", -1)
img = cv2.imread("/home/bornchow/ROS_WS/slam_ws/src/data_make/dataTest/320.360000.png", -1)
# print("img", img)

valid_img = np.zeros(img.shape, dtype=np.uint8)
valid_img[img == 65535] = 255  # 将无效值赋值为第二层的255
# print("valid_img", valid_img)

min = np.min(img)
# print("min: ", min)
img[img == 65535] = 0
max = np.max(img)
# print("max: ", max)
diff = int(max) - int(min)
if diff == 0:
    img_out = np.zeros(img.shape, dtype=np.uint8)
else:
    mask = np.ones(img.shape, dtype=np.int16) * min
    # print(img)
    img_ = img - mask
    img_[img_ < 0] = 0
    # print("img - min: ", img_)

    img_out = (img_/diff *255).astype(np.uint8)

# print("img 2 processed: ", img_out)
enhanced_img = np.concatenate([img_out.reshape(img.shape[0], img.shape[1], 1),
                               valid_img.reshape(valid_img.shape[0], valid_img.shape[1], 1)], 2)

transform = transforms.ToTensor()
tensor_img = transform(enhanced_img).to(device=device).unsqueeze(0)
print(tensor_img.size())

out = net(tensor_img)

print("out: ", out.item())



