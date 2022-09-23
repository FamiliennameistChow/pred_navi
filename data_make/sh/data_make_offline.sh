## 3D rrt path planning in octomap
gnome-terminal --window -e 'bash -c "roscore; exec bash"' \
--tab -e 'bash -c "sleep 3; roslaunch data_make scout_moon_sim.launch; exec bash"' \
--tab -e 'bash -c "sleep 3; roslaunch data_make pc_to_octomap.launch; exec bash"' \
--tab -e 'bash -c "sleep 40; roslaunch data_make pc_data_make_offline.launch; exec bash"'


