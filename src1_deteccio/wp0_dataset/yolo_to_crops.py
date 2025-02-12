
import cv2
from docopt import docopt
import numpy as np
import os
from shapely import Polygon
import shutil
import sys
from tqdm import tqdm
from ultralytics.data.dataset import YOLODataset

from utils.gt_crops import random_crop, filter_annotations


def get_bbox_from_mask(mask):
    polygon = Polygon(np.array(mask.xyn[0]) * [mask.img_size[1], mask.img_size[0]])
    min_x, min_y, max_x, max_y = polygon.bounds
    return [min_x, min_y, max_x - min_x, max_y - min_y] # x1y1wh (It's ok)


def adjust_segment_annotations(segments, initial, crop_width, crop_height):
    adjusted_segments = []
    for segment in segments:
        polygon = Polygon(segment)
        cropped_polygon = polygon.translate(-initial[0], -initial[1])
        clipped_polygon = cropped_polygon.intersection(Polygon([(0, 0), (crop_width, 0), (crop_width, crop_height), (0, crop_height)]))
        if clipped_polygon.is_empty:
            continue
        normalized_polygon = np.array(clipped_polygon.exterior.coords) / [crop_width, crop_height]
        adjusted_segments.append(normalized_polygon.tolist())
    return adjusted_segments

def process_video(dataset_path, train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir, crop_width, crop_height):

    dataset = YOLODataset(dataset_path, task='segment')
    class_id_map = {}
    for i in tqdm(range(len(dataset)), desc="Processing dataset"):

        img = dataset[i]['img']
        img_id = os.path.splitext(os.path.basename(dataset[i]['im_file']))[0]
        masks = dataset[i]['masks']
        split = dataset[i]['split']
        class_ids = dataset[i]['cls']
        class_names = dataset.names

        bboxes = [get_bbox_from_mask(mask) for mask in masks]

        width, height = img.shape[2], img.shape[1]
        seen = np.full(len(bboxes), False)
        seen[bboxes[:, 2] > crop_width] = True
        seen[bboxes[:, 3] > crop_height] = True
        idx = 1

        while not np.all(seen):
            initial, final = random_crop(bboxes, seen, crop_width, crop_height, width, height)
            cropped_img = img[:, initial[1] : final[1], initial[0] : final[0]]
            
            within, seen = filter_annotations(bboxes, seen, initial, final)
            segments_save = adjust_segment_annotations([mask.xyn[0] * [width, height] for mask in masks], initial, crop_width, crop_height)
            
            base_filename = f"{img_id}_{idx}_{len(segments_save)}"
            filename = f"{base_filename}.png"
            labels_filename = f"{base_filename}.txt"
            idx += 1
            
            if split == 'train':
                crop_dir, label_dir = train_img_dir, train_label_dir
            elif split == 'val':
                crop_dir, label_dir = val_img_dir, val_label_dir
            else:
                crop_dir, label_dir = test_img_dir, test_label_dir

            os.makedirs(crop_dir, exist_ok=True)
            os.makedirs(label_dir, exist_ok=True)
            
            cv2.imwrite(os.path.join(crop_dir, filename), cropped_img)
            with open(os.path.join(label_dir, labels_filename), 'w') as f:
                labels = '\n'.join([f"{class_ids[j]} " + ' '.join(map(str, seg)) for j, seg in enumerate(segments_save)])
                f.write(labels)
            
            for j in class_ids:
                class_id_map[j] = class_names[j]
        
    return class_id_map

            
DOCTEXT = f"""
Usage:
  yolo_to_crops.py <yolo_seg_dataset> <output_file> [--width=<w>] [--height=<h>]

Options:
  --test_frac=<tf>          The fraction of frames used for testing. [default: 0.3]
  --sampling_rate=<sr>      Number of frames skipped between saved images. [default: 2]
  --width=<w>               Width of the crop. [default: 640]
  --height=<h>              Height of the crop. [default: 640]

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    yolo_seg_dataset = args['<yolo_seg_dataset>']
    output_file = args['<output_file>']

    crop_width = int(args['--width'])
    crop_height = int(args['--height'])

    basename = os.path.basename(output_file)
    yolo_config_dir = os.path.join(output_file, basename)

    train_img_dir = os.path.join(output_file, basename, 'images', 'train')
    val_img_dir = os.path.join(output_file, basename, 'images', 'val')
    test_img_dir = os.path.join(output_file, basename, 'images', 'test')
    train_label_dir = os.path.join(output_file, basename, 'labels', 'train')
    val_label_dir = os.path.join(output_file, basename, 'labels', 'val')
    test_label_dir = os.path.join(output_file, basename, 'labels', 'test')

    os.makedirs(yolo_config_dir, exist_ok=False)
    os.makedirs(train_img_dir, exist_ok=False)
    os.makedirs(val_img_dir, exist_ok=False)
    os.makedirs(test_img_dir, exist_ok=False)
    os.makedirs(train_label_dir, exist_ok=False)
    os.makedirs(val_label_dir, exist_ok=False)
    os.makedirs(test_label_dir, exist_ok=False)

    class_id_map = process_video(yolo_seg_dataset, train_img_dir, train_label_dir, val_img_dir, val_label_dir, test_img_dir, test_label_dir, crop_width, crop_height)

    class_names_text = '\n    '.join([f"{k}: {v}" for k, v in class_id_map.items()])

    config_text = f"""
    # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    path: ./{yolo_config_dir}  # dataset root dir
    train: images/train  # train images (relative to 'path')
    val: images/val  # val images (relative to 'path')
    test: images/test # test images (optional)

    # Classes
    names:
    {class_names_text}

    """

    with open(os.path.join(yolo_config_dir, f'{basename}.yaml'), 'w') as f:
        f.write(config_text)

    shutil.make_archive(output_file, 'zip', output_file)
    shutil.rmtree(output_file)
