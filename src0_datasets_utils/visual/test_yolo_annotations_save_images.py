
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import yaml
from ultralytics.data.dataset import YOLODataset


def draw_and_save_obboxes(image_tensor, obboxes_tensor, save_path):

    _, height, width = image_tensor.shape
    image = image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for obbox in obboxes_tensor:
        cx, cy, h, w, rad = obbox.numpy()
        cx = cx * width
        cy = cy * height
        h = h * height
        w = w * width

        rect = patches.Rectangle(
            (cx - w / 2, cy - h / 2),  # Upper-left corner of the rectangle
            w, h,  # Width and Height
            angle=np.degrees(rad) + 90,  # Convert radian to degree
            rotation_point='center',
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)

        center = patches.Circle((cx, cy), 2, color='blue')
        ax.add_patch(center)

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

if __name__ == "__main__":
    dataset_yaml_path = "./yolo_obb_UNKNOWN_colors_20241018_1324/yolo_obb_UNKNOWN_colors_20241018_1324.yaml"
    img_path = './yolo_obb_UNKNOWN_colors_20241018_1324/images/'
    save_folder = "./what_yolo_see"

    os.makedirs(save_folder, exist_ok=True)

    with open(dataset_yaml_path, 'r') as file:
        dataset_yaml_dict = yaml.safe_load(file)

    dataset = YOLODataset(img_path, augment=False, data=dataset_yaml_dict, task='obb')

    for i in range(len(dataset)):
        image_tensor = dataset[i]['img']
        obboxes_tensor = dataset[i]['bboxes']
        save_path = os.path.join(save_folder, f"image_{i}.png")

        draw_and_save_obboxes(image_tensor, obboxes_tensor, save_path)
        print(f"Saved: {save_path}")
