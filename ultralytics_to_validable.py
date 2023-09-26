
from docopt import docopt
import cv2 as cv
import numpy as np
import os
import re
import sys
from ultralytics.yolo.data.dataset import YOLODataset
import yaml


colours = [
    (240,163,255),
    (0,117,220),
    (153,63,0),
    (76,0,92),
    (25,25,25),
    (0,92,49),
    (43,206,72),
    (255,204,153),
    (128,128,128),
    (148,255,181),
    (143,124,0),
    (157,204,0),
    (194,0,136),
    (0,51,128),
    (255,164,5),
    (255,168,187),
    (66,102,0),
    (255,0,16),
    (94,241,242),
    (0,153,143),
    (224,255,102),
    (116,10,255),
    (153,0,0),
    (255,255,128),
    (255,255,0),
    (255,80,5)
]

def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        (dict): YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        s = f.read()  # string

        # Remove special characters
        if not s.isprintable():
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)

        # Add YAML filename to dict and return
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)

DOCTEXT = f"""
Usage:
  ultralytics_to_validable.py <data_path> [--imgsz=<is>]

Options:
  --imgsz=<is>              Image size [default: 640]
"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    data_path = args['<data_path>']
    imgsz = int(args['--imgsz'])

    dirname = os.path.dirname(data_path)

    data = os.path.join(data_path, f'{dirname}.yaml')
    data = yaml_load(data, append_filename=True)
    
    dataset = YOLODataset(img_path=data_path, imgsz=imgsz, batch_size=1, augment=False, pad=0.0, data=data, cache=False)
    num_elems = len(dataset)

    crop_path = os.path.join(data_path, 'crops/')
    os.makedirs(crop_path, exist_ok=False)

    for i, batch_dict in enumerate(dataset):

        if i % 500 == 0:
            print(f'Processing {i} / {num_elems}')

        img_path = batch_dict['im_file'] # Extract basename with .png
        img = batch_dict['img']
        out_img = np.ascontiguousarray(np.moveaxis(img.numpy(), [0, 1, 2], [2, 0, 1]), dtype=np.uint8)
        bboxes = batch_dict['bboxes'] # x-center, y-center, width, height normalized by image shape

        sub_dirname = os.path.basename(os.path.normpath(os.path.dirname(img_path)))
        basename = os.path.basename(img_path)
        output_file = os.path.join(crop_path, sub_dirname, basename)
        os.makedirs(os.path.join(crop_path, sub_dirname), exist_ok=True)

        #if len(bboxes) > 1: # NOTE: bboxes should be (1, 4) for my current usage
        #    continue

        for i, bbox in enumerate(bboxes):

            x1 = int((bbox[1] - bbox[3] / 2) * img.shape[1])
            x2 = int((bbox[1] + bbox[3] / 2) * img.shape[1])
            y1 = int((bbox[0] - bbox[2] / 2) * img.shape[2])
            y2 = int((bbox[0] + bbox[2] / 2) * img.shape[2])

            if len(bboxes) == 1:
                crop = out_img[x1 : x2, y1 : y2, :]
                cv.imwrite(output_file, crop)

            else:
                colour = colours[i % len(colours)][::-1]
                thinkness = 2
                cv.rectangle(out_img, (y1, x1), (y2, x2), colour, thinkness)
                if i == (len(bboxes) - 1):
                    cv.imwrite(output_file, out_img)
