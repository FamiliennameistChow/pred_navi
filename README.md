# pred_navi
This is a ROS package for autonomous navigation in off-road environments with self-supervised traversal cost prediction

this project is well test in:

- [ ] ubuntu 20.04 ros noetic

# Struct 

```
-- scout_description # URDF model for scout
-- scout_gazebo_sim  # GAZEBO simulator set
-- data_make         # code for dataset make
-- gazebo_model      # gazebo model 
```

----

# Install

## ros

- gazbeo

```
sudo apt-get install ros-$ROS_DISTRO--gazebo-ros-control
sudo apt install ros-$ROS_DISTRO-teleop-twist-keyboard
sudo apt-get install ros-$ROS_DISTRO--joint-state-publisher-gui
sudo apt install ros-$ROS_DISTRO--ros-controllers
sudo apt install ros-$ROS_DISTRO-navigation

```
- velodyne

```
sudo apt-get install ros-$ROS_DISTRO-velodyne*
```

- octomap

```
sudo apt-get install ros-$ROS_DISTRO-octomap-ros 
sudo apt-get install ros-$ROS_DISTRO-octomap-msgs
sudo apt-get install ros-$ROS_DISTRO-octomap-server
sudo apt-get install ros-$ROS_DISTRO-octomap-rviz-plugins
```

- navigation

```
sudo apt install ros-$ROS_DISTRO-navigation
```

## Write the gazebo environment to the system environment

```
cd gazebo_model
```

bash
```
echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)" >> ~/.bashrc
```

zsh
```
echo "export GAZEBO_MODEL_PATH=$GAZEBO_MODEL_PATH:$(pwd)" >> ~/.zshrc
```

----

# Dataset make

We use GAZEBO to collect and make dataset. GO TO [data_make](./data_make)


# Network training and test




# real-time navigation

The simulation test in GAZEBO. GO TO [navi_pred](./navi_pred)