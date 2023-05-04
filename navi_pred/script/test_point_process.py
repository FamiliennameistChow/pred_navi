#!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: navi_net  --- > test_point_process.py
# Author: bornchow
# Date:20210923
#
# ------------------------------------

import numpy as np

# 归一化操作
point_set = np.random.rand(10, 3)
print(point_set)

print("-----")
print(np.mean(point_set, axis=0))
print("-----")
print(np.expand_dims(np.mean(point_set, axis=0), 0))


point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # center
print("-----")
print(point_set)

dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
point_set = point_set / dist  # scale