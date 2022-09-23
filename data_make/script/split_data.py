# #!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: matplt  --- > split_data.py
# Author: bornchow
# Date: 20210913
# 功能：
# 分离正负样本
# ------------------------------------
import os
from shutil import copyfile

DIR = "../data_22_modify/dataTest"
Neg_DIR = "../data_22_modify/data_neg"
ACG_DIR = "../data_22_modify/data_acg"

files = os.listdir(DIR)
data = []
label_dict = {}

# for file in files:
#     if file.split(".")[-1] == "pcd":
#         data.apend(file)

with open(os.path.join(DIR, "std.md"), "r") as f:
    lines = f.readlines()
    for line in lines:
        id = str(line.split(" ")[0].split(".")[0] + "." + line.split(" ")[0].split(".")[1])
        cls = line.split(" ")[-1]
        if float(cls) > 0.8: # neg data
            print("copy: ", os.path.join(DIR, id+".pcd"), "-->>", os.path.join(Neg_DIR, id+".pcd"))
            copyfile(os.path.join(DIR, id+".pcd"), os.path.join(Neg_DIR, id+".pcd"))
            copyfile(os.path.join(DIR, id+".txt"), os.path.join(Neg_DIR, id+".txt"))
            copyfile(os.path.join(DIR, id+".png"), os.path.join(Neg_DIR, id+".png"))
        else:
            print("copy: ", os.path.join(DIR, id+".pcd"), "-->", os.path.join(ACG_DIR, id+".pcd"))
            copyfile(os.path.join(DIR, id+".pcd"), os.path.join(ACG_DIR, id+".pcd"))
            copyfile(os.path.join(DIR, id+".txt"), os.path.join(ACG_DIR, id+".txt"))
            copyfile(os.path.join(DIR, id+".png"), os.path.join(ACG_DIR, id+".png"))
            

