# Semantic Mapping Package

This package is a refactored version of `semantic_map_pkg`, designed with improved architecture and modularity.

## Package Structure

```
semantic_mapping_pkg/
├── CMakeLists.txt          # Build configuration
├── package.xml             # Package metadata and dependencies
├── README.md               # This file
├── config/                 # Configuration files (YAML, etc.)
├── include/                # Header files
│   └── semantic_mapping_pkg/
├── launch/                 # Launch files
└── src/                    # Source files (C++ nodes, Python scripts)
```

## TODO

- [ ] Define package architecture
- [ ] Implement core semantic mapping functionality
- [ ] Create node interfaces
- [ ] Add configuration files
- [ ] Create launch files
- [ ] Add documentation

## Dependencies

- ROS (roscpp, rospy)
- PCL (Point Cloud Library)
- Eigen3
- RapidJSON
- yaml-cpp
- OpenCV (optional)

## Usage

```bash
# Build the package
cd ~/robot_ws/semantic_ws
catkin_make

# Source the workspace
source devel/setup.bash

# Run nodes (after implementation)
roslaunch semantic_mapping_pkg semantic_mapping.launch
```

## Notes

This is a work-in-progress package. Implementation details will be discussed and added incrementally.

