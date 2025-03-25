
from docopt import docopt
import json
import os
import pandas as pd
import shutil
import sys


_NAMES = ['fr', 'id', 'x', 'y', 'w', 'h', 'conf', 'None1', 'None2', 'None3', 'a']
def parse_custom_annotations(filename):

    data = pd.read_csv(filename, delimiter=',', header=None, names=_NAMES)

    frames = {}

    for _, row in data.iterrows():
        frame_num = int(row['fr'])
        track_id = int(row['id'])
        x_center, y_center, width, height = row['x'], row['y'], row['w'], row['h']
        #confidence = row['conf']
        rotation = row['a']

        x_tl = x_center - width / 2
        y_tl = y_center - height / 2
        bbox = [x_tl, y_tl, width, height]

        if frame_num not in frames:
            frames[frame_num] = []

        frames[frame_num].append({
            "id": 0,
            "type": "bbox",
            "attributes": {
                "occluded": False,
                "rotation": rotation,
                "track_id": track_id,
                "keyframe": True
            },
            "group": 0,
            "label_id": 0,
            "z_order": 0,
            "bbox": bbox
        })

    return frames


def generate_cvat_json(input_file):

    frames = parse_custom_annotations(input_file)

    cvat_json = { # Datumaro format
        "info": {},
        "categories": {
            "label": {
                "labels": [{"name": "ant", "parent": "", "attributes": []}],
                "attributes": ["occluded"]
            },
            "points": {
                "items": []
            }
        },
        "items": []
    }

    for frame_num_from_1, annotations in frames.items():
        frame_num = frame_num_from_1 - 1
        frame_entry = {
            "id": f"frame_{frame_num:06d}",
            "annotations": annotations,
            "attr": {"frame": frame_num},
            "point_cloud": {"path": ""}
        }
        cvat_json["items"].append(frame_entry)

    return cvat_json


DOCTEXT = f"""
    Usage:
      obb_to_cvat_json.py <input> <output>
"""

if __name__ == "__main__":
    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    input_filename = args['<input>']
    output_filename = args['<output>']

    os.makedirs(os.path.dirname(output_filename) or '.', exist_ok=True)

    cvat_json = generate_cvat_json(input_filename)

    path = os.path.dirname(output_filename)
    base_name = os.path.splitext(os.path.basename(output_filename))[0]
    dir_name = os.path.join(path, base_name)
    sec_dir_name = os.path.join(dir_name, 'annotations')

    os.makedirs(sec_dir_name, exist_ok=False)

    with open(os.path.join(sec_dir_name, f'{base_name}.json'), 'w') as f:
        json.dump(cvat_json, f, indent=4)

    shutil.make_archive(dir_name, 'zip', dir_name)
    shutil.rmtree(dir_name)
