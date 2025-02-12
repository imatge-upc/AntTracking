
import cv2
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
from tqdm import tqdm
from ultralytics.data.dataset import YOLODataset
import yaml


def mask_to_polygon(mask, shape):
    mask_np = cv2.resize(mask.squeeze(0).numpy().astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)

    unique_ids = np.unique(mask_np)
    polygons_per_id = {}

    for obj_id in unique_ids:
        if obj_id == 0:  # Skip background
            continue

        class_mask = (mask_np == obj_id).astype(np.uint8)
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        polygons = []
        for contour in contours:
            contour = contour.squeeze(1)
            if len(contour) > 2:
                polygons.append(contour)
        
        polygons_per_id[obj_id] = polygons
    
    return polygons_per_id

def draw_and_save_masks(image_tensor, masks, bboxes_tensor, save_path, colors):
    _, height, width = image_tensor.shape
    image = image_tensor.permute(1, 2, 0).numpy()

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    for obj_id, mask_polygons in masks.items():

        color = [c / 255. for c in colors[obj_id % len(colors)]]

        for mask in mask_polygons:
            mask_points = np.array(mask)# * np.array([width, height])
            if len(mask_points) > 2:
                polygon = patches.Polygon(mask_points, edgecolor=color, facecolor=None, linewidth=2, alpha=0.4)
                ax.add_patch(polygon)

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

def draw_and_save_labels(image, masks, save_path, colors):
    height, width, _ = image.shape

    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    for obj_id, mask in enumerate(masks):
        color = [c / 255. for c in colors[obj_id % len(colors)]]

        mask_points = np.array(mask) * np.array([width, height])
        if len(mask_points) > 2:
            polygon = patches.Polygon(mask_points, edgecolor=color, facecolor=None, linewidth=2, alpha=0.4)
            ax.add_patch(polygon)

    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    dataset_yaml_path = "./yolo_seg_UNKNOWN_colors_20241018_1324/yolo_seg_UNKNOWN_colors_20241018_1324.yaml"
    img_path = './yolo_seg_UNKNOWN_colors_20241018_1324/images/'
    save_folder = "./what_yolo_see"
    save_folder2 = "./what_yolo_have"
    colors_json_path = "./ANTS/src8_tracking/wp2_visualization/100_colors.json"

    # Load colors from JSON file
    with open(colors_json_path, 'r') as f:
        colors = json.load(f)

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(save_folder2, exist_ok=True)

    with open(dataset_yaml_path, 'r') as file:
        dataset_yaml_dict = yaml.safe_load(file)

    dataset = YOLODataset(img_path, augment=False, data=dataset_yaml_dict, task='segment')

    for i in tqdm(range(len(dataset))):
        image_tensor = dataset[i]['img']
        bboxes_tensor = dataset[i]['bboxes']

        _, height, width = image_tensor.shape
        masks = mask_to_polygon(dataset[i]['masks'], (height, width))

        img_file_1 = dataset[i]['im_file']
        img_file_2 = dataset.labels[i]['im_file']

        assert img_file_1 == img_file_2

        save_path = os.path.join(save_folder, f"image_{i}.png")
        save_path2 = os.path.join(save_folder2, f"image_{i}.png")

        draw_and_save_masks(image_tensor, masks, bboxes_tensor, save_path, colors)

        im_file = dataset.labels[i]['im_file']
        image = cv2.imread(im_file)
        masks = dataset.labels[i]['segments']
        draw_and_save_labels(image, masks, save_path2, colors)
        # print(f"Saved: {save_path}")
