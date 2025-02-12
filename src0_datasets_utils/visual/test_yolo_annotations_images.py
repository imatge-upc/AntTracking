
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.utils.plotting import plot_images


def draw_obboxes(image_tensor, obboxes_tensor):
    """
    Draws OBBs on an image tensor.

    Parameters:
    - image_tensor: Tensor of shape (C, H, W) representing the image.
    - obboxes_tensor: Tensor of shape (N, 5) representing OBBs, each with (cx, cy, h, w, rotation).
    """
    # Convert the image tensor to numpy for plotting
    _, height, width = image_tensor.shape
    image = image_tensor.permute(1, 2, 0).numpy()

    # Create a plot
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Iterate through the OBBs and draw them
    for obbox in obboxes_tensor:
        cx, cy, h, w, rad = obbox.numpy()
        cx = cx * width
        cy = cy * height
        h = h * height
        w = w * width

        # Create a rotated rectangle (center, height, width, angle)
        rect = patches.Rectangle(
            (cx - w / 2, cy - h / 2),  # Upper-left corner of the rectangle
            w, h,  # Width and Height
            angle=np.degrees(rad) + 90,  # Convert radian to degree
            rotation_point='center',
            linewidth=2, edgecolor='r', facecolor='none'
        )

        ax.add_patch(rect)
        center = patches.Circle((cx, cy), 2)
        ax.add_patch(center)

    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Define paths
    dataset_yaml_path = "./yolo_obb_UNKNOWN_colors_20241018_1324/yolo_obb_UNKNOWN_colors_20241018_1324.yaml"
    img_path = './yolo_obb_UNKNOWN_colors_20241018_1324/images/'

    # Load the dataset YAML configuration file
    with open(dataset_yaml_path, 'r') as file:
        dataset_yaml_dict = yaml.safe_load(file)

    # Load the dataset
    dataset = YOLODataset(img_path, augment=False, data=dataset_yaml_dict, task='obb')

    # Example usage
    i = 0
    image_tensor = dataset[i]['img']
    obboxes_tensor = dataset[i]['bboxes']

    draw_obboxes(image_tensor, obboxes_tensor)
