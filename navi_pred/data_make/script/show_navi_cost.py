# #!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: matplt  --- > show_navi_cost.py
# Author: bornchow
# Date:20210816
# 功能：
# 可视化navi_cost
# ------------------------------------

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

fig = plt.figure()
ax = Axes3D(fig)

DATA_DIR = "../data/"

std_file = os.path.join(DATA_DIR, "std.md")
X = []
Y = []
Z = []
with open(std_file, "r") as f:
    lines = f.readlines()

    print(float(lines[0].split(" ")[-1]))

    for line in lines:
        X.append(float(line.split(" ")[1])) #acc_z
        Y.append(abs(float(line.split(" ")[2])))
        Z.append(abs(float(line.split(" ")[-1])))

X, Y = np.meshgrid(X, Y)


Z = np.sin(X)+np.cos(Y)

print(Z)

ax.plot_surface(X, Y, Z, rstride = 1, cstride = 1, cmap = plt.get_cmap('rainbow'))


ax.contour(X, Y, Z, zdir = 'z', offset = 0, cmap = plt.get_cmap('rainbow'))

ax.set_zlim(0, 1)

plt.show()