# Similarity Matching Package

A ROS package for image-based navigation using similarity matching between a reference image and camera stream. The package computes similarity scores and keypoint matching to enable navigation based on visual features. This implementation uses the XFeat model for feature detection and matching.

## Features

- Real-time similarity scoring between target and scene images
- Keypoint detection and matching using XFeat
- ROS integration with image streams
- Debug visualization tools
- Python-C++ interop for efficient processing

## Prerequisites

- ROS (tested on ROS Noetic)
- OpenCV
- Python 3
- PyTorch
- Additional Python packages:
  - numpy
  - torch

## Installation

1. Clone the repository with submodules:
```bash
git clone --recursive https://github.com/your-repo/similarity_matching.git
```

2. Install dependencies:
```bash
sudo apt-get install python3-dev python3-numpy python3-torch libopencv-dev
```

3. Build the package:
```bash
cd your_workspace
catkin build similarity_matching
source devel/setup.bash
```

## Usage

1. Launch the main node:
```bash
roslaunch similarity_matching similarity_matching.launch
```

2. Publish a target image:
```bash
rosrun similarity_matching target_image_publisher.py _image_path:=/path/to/target/image.jpg
```

3. The node will begin processing the camera stream and publishing:
   - Similarity scores
   - Detected keypoints
   - Debug visualizations (if enabled)

## Nodes

### similarity_matching_node

Main node handling the similarity computation and ROS integration.

## Debug Tools

The package includes debugging tools to visualize the matching process:

```bash
python3 scripts/xfeat_interface.py \
    --target_img_path=/path/to/target/image.jpg \
    --scene_img_path=/path/to/scene/image.jpg \
    --debug_print_results_to_console \
    --save_matching_results_to_image
```

## Acknowledgments

This package uses the XFeat model for feature detection and matching. Check out the project [here](https://github.com/verlab/accelerated_features?tab=readme-ov-file).
