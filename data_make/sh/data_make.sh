## 3D rrt path planning in octomap
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 2; rosbag play /home/bornchow/ROS_WS/slam_ws/outdoor_small_loop.bag; exec bash"' \
--tab -e 'bash -c "sleep 2; roslaunch lio_sam run.launch; exec bash"' \
--tab -e 'bash -c "sleep 2; roslaunch data_make pc_data_make.launch; exec bash"' 


