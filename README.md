# k3dom
Implementation of Kernel-Based 3-D Dynamic Occupancy Mapping with Particle Tracking

## How to build
Tested for Ubuntu 18.04 LTS with ROS melodic (1.14.9)

```console
cd ~/catkin_ws/src
git clone https://github.com/youngjae-min/k3dom.git
cd ~/catkin_ws && catkin_make
source devel/setup.bash
roslaunch k3dom k3dom_demo.launch
```

<p align="center">
  <img src="./docs/complex.gif">
</p>

## References
Youngjae Min, Do-Un Kim, and Han-Lim Choi, "Kernel-Based 3-D Dynamic Occupancy Mapping with Particle Tracking," 2021 IEEE International Conference on Robotics and Automation (ICRA)

[TheCodez/dynamic-occupancy-grid-map](https://github.com/TheCodez/dynamic-occupancy-grid-map) has been a great reference
