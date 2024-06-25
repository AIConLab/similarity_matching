# similarity_mapping

Package for mapping and navigation based on similarity values to a reference image.

## Installation

TODO

## Usage

See launch file.


## Debug

From the package directory:
```python3
python3 scripts/xfeat_interface.py \
    --target_img_path=/home/jc/Workspaces/nav_sim_ws/src/drone_waypoint_sim/missions/mission_2024_06_21_10_06_41/waypoint_2024_06_21_10_08_04/target_image.jpg \
    --scene_img_path=/home/jc/Workspaces/nav_sim_ws/src/drone_waypoint_sim/missions/mission_2024_06_21_10_06_41/waypoint_2024_06_21_10_08_04/target_image.jpg \
    --debug_print_results_to_console \
    --save_matching_results_to_image
```