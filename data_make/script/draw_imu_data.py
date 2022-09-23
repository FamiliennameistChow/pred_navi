# #!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: matplt  --- > draw_imu_data.py
# Author: bornchow
# Date:20210719
# 功能：
# 1. 读取data/下的imu数据并画图
# https://blog.csdn.net/itnerd/article/details/109062516
# 2. 计算acc_z的 标准差 20210811
# ------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pylab import *
# from scipy.fftpack import fft,ifft
import os
import math
import numpy.fft as fft
import sys


try:
    DATA_DIR = sys.argv[1]
except IndexError:
    DATA_DIR = "../dataTest/"

try:
    DRAW_PIC = sys.argv[2]
except IndexError:
    DRAW_PIC = False


imu_file_dir_list = []
time = []
acc_z = [] 
ang_velo_x = []
ang_velo_y = []
roll = []
pitch = []
acc_z_std = []

std_file = os.path.join(DATA_DIR, "std.md")

if os.path.exists(std_file):
    os.remove(std_file)


def quat_to_rpy(w, x, y, z): # 单位为角度
    r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
    p = math.asin(2*(w*y-z*x))
    y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

    angleR = r*180/math.pi
    angleP = p*180/math.pi
    angleY = y*180/math.pi
    return angleR, angleP, angleY


# 获取data下的目录
print(DATA_DIR)
for path, folder_name_list, file_name_list in os.walk(DATA_DIR):
    print(path)
    # print(folder_name_list)
    # print(file_name_list)

# 获取imu数据的txt
for i in range(len(file_name_list)):
    if(file_name_list[i].split(".")[-1] == "txt"):
        imu_file_dir_list.append(file_name_list[i])


# print("imu data: ", imu_file_dir_list)

# txt书写格式查看readme.md文件
# 时间戳　linear_acceleration.x　linear_acceleration.y linear_acceleration.z \
# angular_velocity.x angular_velocity.y angular_velocity.z \
# orientation.w orientation.x orientation.y orientation.z
#

for i in range(len(imu_file_dir_list)):
    print("process %s" % imu_file_dir_list[i])
    # 清空列表
    time.clear()
    acc_z.clear()
    ang_velo_x.clear()
    ang_velo_y.clear()
    roll.clear()
    pitch.clear()
    with open(os.path.join(DATA_DIR, imu_file_dir_list[i]), "r") as f:
        for line in f.readlines():
            line_data = line.split("\t")
            
            time.append(float(line_data[0]))
            # print(time)
            acc_z.append(float(line_data[3]))
            # print(acc_z)
            ang_velo_x.append(float(line_data[4]))
            ang_velo_y.append(float(line_data[5]))
            # print(float(line_data[8]))
            r, p ,y = quat_to_rpy(float(line_data[7]), float(line_data[8]), float(line_data[9]),float(line_data[10]))
            roll.append(r)
            pitch.append(p)
            

    print("there have %d msgs" % len(time))

    # 计算acc_Z的 标准差
    acc_z_std = np.std(acc_z, ddof=1)

    roll_mean = np.mean(roll)
    pitch_mean = np.mean(pitch)

    roll_std = np.std(roll)
    pitch_std = np.std(pitch)

    roll_max = np.max(roll)
    pitch_max = np.max(pitch)
    roll_min = np.min(roll)
    pitch_min = np.min(pitch)

    roll_max_min = roll_max if abs(roll_max) > abs(roll_min) else roll_min
    pitch_max_min = pitch_max if abs(pitch_max) > abs(pitch_min) else pitch_min

    if(abs(roll_max_min) > 20  or abs(pitch_max_min) > 20):
        navi_cost = 1
    else:
        # navi_cost = (acc_z_std / 10) * 0.8 + 0.1 * (abs(roll_max_min)/ 20 + abs(pitch_max_min)/20)
        navi_cost =  abs(roll_max_min)/ 20 + abs(pitch_max_min)/20  # test in 2021.08
    
    if navi_cost > 1:
        navi_cost = 1

    print(imu_file_dir_list[i] + " " + str(acc_z_std) + " " + 
            # str(roll_mean) + " " + str(pitch_mean) + " " +
            str(roll_max_min) + " " + str(pitch_max_min) + " " +
            str(roll_std) + " " + str(pitch_std) + " " + str(navi_cost) ) 

    with open(std_file, "a") as f:
        f.write(imu_file_dir_list[i] + " " + str(acc_z_std) + " " + 
                # str(roll_mean) + " " + str(pitch_mean) + " " + 
                str(roll_max_min) + " " + str(pitch_max_min) + " "  +
                str(roll_std) + " " + str(pitch_std) +  " " + str(navi_cost) + "\n")


    if DRAW_PIC:
        ## 快速傅里叶变换 https://www.cnblogs.com/LXP-Never/p/11558302.html
        Fs = 300
        T = 1/Fs
        L = len(time)
        t = [i*T for i in range(L)]
        t = np.array(t)
        # 得到分解波的频率序列
        freqs = fft.fftfreq(t.size, t[1] - t[0])

        # acc_z_f = abs(fft(acc_z) / (len(time)/2)) # 取摸 归一化
        # ang_velo_x_f = abs(fft(ang_velo_x) / (len(time)/2))
        # ang_velo_y_f = abs(fft(ang_velo_y) / (len(time)/2))
        # xf = np.arange(len(acc_z))  

        acc_z_complex_array = fft.fft(acc_z)
        # 复数的模为信号的振幅（能量大小）
        acc_z_f = np.abs(acc_z_complex_array)

        ang_velo_x_complex_array = fft.fft(ang_velo_x)
        ang_velo_x_f = np.abs(ang_velo_x_complex_array)

        ang_velo_y_complex_array = fft.fft(ang_velo_y)
        ang_velo_y_f = np.abs(ang_velo_y_complex_array)

        # 画图
        fig = plt.figure()
        fig.set_size_inches(16, 10)
        fig.subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=0.2, hspace=0.3)

        ax1 = fig.add_subplot(4,2,1)
        ax2 = fig.add_subplot(4,2,2)
        ax3 = fig.add_subplot(4,2,3)
        ax4 = fig.add_subplot(4,2,4)
        ax5 = fig.add_subplot(4,2,5)
        ax6 = fig.add_subplot(4,2,6)
        ax7 = fig.add_subplot(4,2,7)


        # acc_z
        ax1.plot(time, acc_z, label="acceleration_z")
        ax1.legend(loc="best")

        ax2.plot(freqs[freqs > 0], acc_z_f[freqs > 0], label="acc_z_f")
        ax2.legend(loc="best")

        # angular_velocity_x
        ax3.plot(time, ang_velo_x, label="angular_velocity_x")
        ax3.legend(loc="best")

        ax4.plot(freqs[freqs > 0], ang_velo_x_f[freqs > 0], label="ang_velo_x_f")
        ax4.legend(loc="best")

        # angular_velocity_y
        ax5.plot(time, ang_velo_y, label="angular_velocity_y")
        ax5.legend(loc="best")

        ax6.plot(freqs[freqs > 0], ang_velo_y_f[freqs > 0], label="ang_velo_y_f")
        ax6.legend(loc="best")

        ax7.plot(time, roll, c="blue", label = "roll")
        ax7.plot(time, pitch, c="red", label = "pitch")
        ax7.legend(loc="best")

        filename = str(imu_file_dir_list[i].split(".")[0]) + "." + str(imu_file_dir_list[i].split(".")[1]) + ".jpg"


        plt.savefig(os.path.join(DATA_DIR, filename))
        plt.show()
        plt.close()








