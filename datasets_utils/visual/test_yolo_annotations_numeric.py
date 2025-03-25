
import numpy as np


# Convert the four corner points to cx, cy, w, h, and radians
def corners_to_center_height_width_rotation(corners):
    # Extract the coordinates
    x1, y1, x2, y2, x3, y3, x4, y4 = corners

    # Calculate center (cx, cy)
    cx = (x1 + x2 + x3 + x4) / 4
    cy = (y1 + y2 + y3 + y4) / 4

    # Calculate width (w) and height (h)
    w = np.linalg.norm([x1 - x2, y1 - y2])
    h = np.linalg.norm([x2 - x3, y2 - y3])

    # Calculate the rotation angle (radians) using the angle of the edge (x1, y1) to (x2, y2)
    angle = np.arctan2(y2 - y1, x2 - x1) % (np.pi / 2)

    return cx, cy, h, w, angle


if __name__ == "__main__":
    # Test data (example from your input)
    corners = [
        [0.6382, 0.2757, 0.7417, 0.2469, 0.7636, 0.3256, 0.6601, 0.3544],
        [0.5143, 0.1441, 0.6150, 0.1437, 0.6152, 0.1987, 0.5145, 0.1991],
        [0.7828, 0.3011, 0.8799, 0.2896, 0.8874, 0.3535, 0.7903, 0.3649]
    ]

    # Convert and print results
    for corners_set in corners:
        cx, cy, w, h, angle = corners_to_center_height_width_rotation(corners_set)
        print(f"cx: {cx}, cy: {cy}, w: {w}, h: {h}, radians: {angle}")

    """
    output:
    cx: 0.7009, cy: 0.30065, w: 0.08169026869829721, h: 0.10743225772550816, radians: 1.2994010459932608
    cx: 0.56475, cy: 0.1714, w: 0.05500036363516154, h: 0.10070079443579381, radians: 1.5668241530486944
    cx: 0.8351, cy: 0.327275, w: 0.06433863536010065, h: 0.09777862752156011, radians: 1.4529108601520506
    """

"""
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.plotting import plot_images
import os
import yaml
import numpy as np
from PIL import Image

# Load the dataset YAML configuration file
dataset_yaml_path = "./yolo_obb_UNKNOWN_colors_20241018_1324/yolo_obb_UNKNOWN_colors_20241018_1324.yaml"
with open(dataset_yaml_path, 'r') as file:
    dataset_yaml_dict = yaml.safe_load(file)

# Image directory
img_path = './yolo_obb_UNKNOWN_colors_20241018_1324/images/'

# Load the dataset
dataset = YOLODataset(img_path, augment=False, data=dataset_yaml_dict, task='obb')
"""
