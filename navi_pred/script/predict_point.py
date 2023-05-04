#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > predict_point.py.py
# Author: bornchow
# Date:20210922
#
# ------------------------------------

import torch
import numpy as np
import os
from model import PointModel
import open3d as o3d

# 处理单帧点云
#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > predict_point.py.py
# Author: bornchow
# Date:20210922
#
# ------------------------------------

import torch
import numpy as np
import os
from model import PointModel

npoints = 450

data_dir = "/home/bornchow/workFile/test/pcl_test/pcd/"

pts_file = "112.016000.pcd"

model_dir = "/home/bornchow/workFile/navi_net/model/model_1576_25.226139.pth"

with open(os.path.join(data_dir, pts_file), "r") as f:
    lines = f.readlines()
    pts_num = len(lines) - 11
    pts = np.zeros((pts_num, 3), dtype=np.float32)
    i = 0
    for line in lines[11:]:
        pts[i] = [line.split(" ")[0], line.split(" ")[1], line.split(" ")[2]]
        i += 1

choice = np.random.choice(pts_num, npoints, replace=True)
# resample
point_set = pts[choice, :]

point_set = torch.from_numpy(point_set).unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

points = point_set.transpose(2, 1).to(device=device)  # (bash_size * channel * points_num)

net = PointModel().to(device=device)
net.load_state_dict(torch.load(model_dir))
net.eval()

out = net(points)

print("predict: ", out.data.max(1)[1].item())





