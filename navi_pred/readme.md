# navi_pred

# 基于地形点云可穿越性预测的导航 -- 3.实时导航部分

项目基于ros开发，采用**gazbeo**仿真器进行仿真演示,采用libtorch推理训练好的 **点云可穿越性预测模型**

## 项目内容包括:

1. 采用滑动窗口方法对局部地图(10*10m范围)进行可穿越性性推理
2. 基于推理结果生成局部地图导航引导
3. 结合引导线与轨迹库(trajector library)的采样点进行实时导航推理

## 安装与使用：

### 预测地形
地形预测目前是在具有全局点云地图的基础上进行的，（该全局地图是事先采用LIO-SAM激光slam构建的）

1. 第一步: 启动仿真器


仿真月面场景

```sh
roslaunch navi_pred scout_moon_sim.launch
```

仿真室外场景

```
roslaunch navi_pred scout_outside_sim.launch
```

2. 加载预先准备的点云地图

```sh
roslaunch navi_pred pc_to_octomap.launch
```

注意修改`pc_to_octomap.launch`: 

`mountain_moon_10.pcd` 是 仿真月面场景  

`outside_flat_tree.pcd` 是 仿真室外场景

```xml
  <!-- <node pkg="pcl_ros" type="pcd_to_pointcloud" name="pcd_to_pointcloud" args="$(find navi_pred)/maps/outside_flat_tree.pcd 1 _frame_id:=$(arg mapFrame)" /> -->
  <node pkg="pcl_ros" type="pcd_to_pointcloud" name="pcd_to_pointcloud" args="$(find navi_pred)/maps/mountain_moon_10.pcd 1 _frame_id:=$(arg mapFrame)" />

```

3. 启动地形预测

```sh
rosrun navi_pred predict
```

### 导航

前两步与`预测地形`一致

3. 导航

```
roslaunch navi_pred navi.launch 2>&1 | tee test.txt
```

4. 计算轨迹长度

```py
python3 cul_traj_len.py
```

5. 记录轨迹点云

```sh
rosrun pcl_ros pointcloud_to_pcd input:=/car_key_pose
```