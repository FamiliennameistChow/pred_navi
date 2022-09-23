# #!/usr/bin/python
# -*- coding: UTF-8 -*-
#
# File: matplt  --- > draw_imu_data.py
# Author: bornchow
# Date:20210712
# 订阅imu数据并画图
# https://blog.csdn.net/itnerd/article/details/109062516
# ------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from pylab import *
from matplotlib.animation import FuncAnimation  # 导入负责绘制动画的接口
import rospy
from sensor_msgs.msg import Imu


class DrawImuData(object):
    def __init__(self):
        self.x = []  # 用于接受后更新的数据
        self.y1, self.y2, self.y3, self.y4 = [], [], [], []
        self.ax1 = plt.subplot(221)  # 生成轴和fig,  可迭代的对象
        self.ax2 = plt.subplot(222)  # 生成轴和fig,  可迭代的对象
        self.ax3 = plt.subplot(223)  # 生成轴和fig,  可迭代的对象
        self.ax4 = plt.subplot(224)  # 生成轴和fig,  可迭代的对象
        # self.line, = plt.plot([], [], '.-')  # 绘制线对象，plot返回值类型，要加逗号
        # self.line1, = plt.plot([], [], '.-')  # 绘制线对象，plot返回值类型，要加逗号
        self.imu_data = Imu()
        self.line1 = None

        subplots_adjust(left=None, bottom=None, right=None, top=None,wspace=1, hspace=1)

        rospy.init_node("draw_imu_data", anonymous=True)
        imu_topic = rospy.get_param("imu_topic", "/imu")

        rospy.Subscriber(imu_topic, Imu, self.imu_cb)
        rate = rospy.Rate(100.0)

        # plt config
        plt.ion()
        
        # x_major_locator = MultipleLocator(250)
        # self.ax3.xaxis.set_major_locator(x_major_locator)

        while not rospy.is_shutdown():

            seq = self.imu_data.header.seq
            r, p, y = self.quat_to_rpy(self.imu_data.orientation.w, self.imu_data.orientation.x, self.imu_data.orientation.y, self.imu_data.orientation.z)
            print("r ", r, "p ", p, "y ", y)
            self.x.append(seq)
            self.y1.append(self.imu_data.linear_acceleration.z)
            self.y2.append(self.imu_data.angular_velocity.x) # self.imu_data.angular_velocity.y
            self.y3.append(r)
            self.y4.append(p)

            if self.line1 is None:
                self.line1 = self.ax1.plot(self.x, self.y1, '-r', label="acceleration z")[0]
                self.line2 = self.ax2.plot(self.x, self.y2, '-g', label="angular_velocity x")[0]
                self.line3 = self.ax3.plot(self.x, self.y3, '-g', label="r")[0]
                self.line4 = self.ax4.plot(self.x, self.y4, '-g', label="p")[0]

                self.ax1.set_title("acceleration z")
                self.ax2.set_title("angular_velocity x")
                self.ax3.set_title("r")
                self.ax4.set_title("p")
                self.ax1.grid(True)
                self.ax2.grid(True)
                self.ax3.grid(True)
                self.ax4.grid(True)

                self.ax1.legend(loc='best')
                self.ax2.legend(loc='best')
                self.ax3.legend(loc='best')
                self.ax4.legend(loc='best')


            self.line1.set_xdata(self.x)
            self.line1.set_ydata(self.y1)
            self.ax1.set_xlim([seq - 3000, seq + 1])
            self.ax1.set_ylim([4, 16])

            self.line2.set_xdata(self.x)
            self.line2.set_ydata(self.y2)
            self.ax2.set_xlim([seq - 3000, seq + 1])
            self.ax2.set_ylim([-3, 3])

            self.line3.set_xdata(self.x)
            self.line3.set_ydata(self.y3)
            self.ax3.set_xlim([seq - 3000, seq + 1])
            self.ax3.set_ylim([-20, 20])


            self.line4.set_xdata(self.x)
            self.line4.set_ydata(self.y4)
            self.ax4.set_xlim([seq - 3000, seq + 1])
            self.ax4.set_ylim([-20, 20])


            plt.pause(0.01)

        rospy.spin()

    # def init(self):
    #     # 初始化函数用于绘制一块干净的画布，为后续绘图做准备
    #     self.ax.set_xlim(0, 50)  # 初始函数，设置绘图范围
    #     self.ax.set_ylim(-5, 15)
    #     return self.line

    def imu_cb(self, msg):
        # rospy.loginfo(rospy.get_caller_id() + "I heard %s", )
        self.imu_data = msg
        # print(msg)

    def quat_to_rpy(self, w, x, y, z):
        r = math.atan2(2*(w*x+y*z),1-2*(x*x+y*y))
        p = math.asin(2*(w*y-z*x))
        y = math.atan2(2*(w*z+x*y),1-2*(z*z+y*y))

        angleR = r*180/math.pi
        angleP = p*180/math.pi
        angleY = y*180/math.pi
        return angleR, angleP, angleY


    # def update(self, step):  # 通过帧数来不断更新新的数值
    #     self.x.append(step)
    #     self.y.append(self.imu_data.linear_acceleration.z)  # 计算y
    #     self.line.set_data(self.x, self.y)
    #     return self.line


if __name__ == "__main__":

    draw = DrawImuData()


