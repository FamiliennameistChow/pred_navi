# Content

[Chinese](./readme.md)

---

This is the code for dataset collection.

---

## data Folder

Store the IMU data(txt file) and the point cloud data(pcd file). They are named according to the ros timestamp, which indicates the time of vehicle pose when the local point cloud is sampled. 

imu data (txt file) represents as follow:
```
timestamp linear_acceleration.x　linear_acceleration.y linear_acceleration.z angular_velocity.x angular_velocity.y angular_velocity.z orientation.w orientation.x orientation.y orientation.z
```


## How to make the dataset

### Step One: Collect the raw data (there are two ways: using the SLAM; offline)
1. Using SLAM to make dataset (Using the ROS bag)

In this approach, a point cloud map is constructed using a real-time slam system, and then sample data form it; This can be used to collect data from real scenes for training.

- Play the ros bag
```
rosbag play xxx.bag
```

- Show the octomap 
```
roslaunch octomap_navi octomap_convert.launch slamType:=aloam
```

- Collect the data
```
roslaunch data_make pc_data_make.launch
```


2. Offline Collection
First, a pre-produced point cloud map is loaded, Then the unmanned vehicle uses a random strategy to collect data. (localization is done by reading the ground truth of the vehicle in gazebo, not by SLAM)

Pre-collection notes:

- Modify `pc_data_make_offline.launch`:

```xml

  <node pkg="data_make" type="pc_data_make_offline" name="pc_data_make" output="screen">
    <param name="local_map_length" value= "2.0" />  <!-- size of 3D box when record data, 2*2*4 in paper 10*10*4 in prediction-->
    <param name="local_map_width" value="2.0" />
    <param name="local_map_height" value="4.0" />
    <param name="data_save_dir" value="/home/bornchow/ROS_WS/slam_ws/src/data_make/data/" />  <!-- where to save the data -->
    <param name="save_data" value="false" />  <!-- if record data, make sure to be set as true -->
    <param name="map_res" value="0.1" />
    <param name="sample_dis" value="0.5" />
    <param name="imu_time_window" value="1.5" />

    <param name="pc_topic" value="/cloud_pcd" />  <!-- global point cloud map topic -->
    <param name="imu_topic" value="/imu" /> <!-- imu topic -->
  </node>

```

- Modify `pc_to_octomap.launch`:

line 9 Modify the pcd file to the global point cloud map
```xml
  <node pkg="pcl_ros" type="pcd_to_pointcloud" name="pcd_to_pointcloud" args="$(find data_make)/maps/mountain_moon_10.pcd 1 _frame_id:=$(arg mapFrame)" />

```

- Collect the data
```
data_make_offline.sh

```


After recording: The data will be saved in the data folder(`data_save_dir` in `pc_data_make_offline.launch`), which needs to be modified manually to `dataSet` / `dataTest`.

---

### Step Two: Self-labeling of raw data

```shell
python3 draw_imu_data.py  dir to dataset(str)  show the pic?(bool)
```

The annotation file `std.md` will be generated under the dataset path after the annotation is completed
