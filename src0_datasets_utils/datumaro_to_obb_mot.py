# NOTE: Datumaro library should do it better, but some bug make it not work

from docopt import docopt
import json
import os
import sys


FRAMES = 'items'
FRAME_INFO = 'attr'
FRAME_NUM = 'frame'
DETECTIONS = 'annotations'
CLASS_ID = 'label_id'
X1Y1WH_BBOX = 'bbox'
DETECTION_INFO = 'attributes'
ROTATION = 'rotation'
TRACK_ID = 'track_id'
KEYFRAME = 'keyframe'


def obb_mot_line_generator(filename, upsample_factor=2):

    with open(filename, 'r') as f:
        dataset = json.load(f)

    for frame in dataset[FRAMES]:
        frame_num = int(frame[FRAME_INFO][FRAME_NUM] + 1)

        for detection in frame[DETECTIONS]:
            #class_id = detection[CLASS_ID]
            bbox = [elem * upsample_factor for elem in detection[X1Y1WH_BBOX]]
            rotation = detection[DETECTION_INFO][ROTATION]
            track_id = int(detection[DETECTION_INFO][TRACK_ID])
            confidence = 1.0 if detection[DETECTION_INFO][KEYFRAME] else 0.9

            x = bbox[0] + bbox[2] / 2
            y = bbox[1] + bbox[3] / 2

            line = f'{frame_num}, {track_id}, {x:.02f}, {y:.02f}, {bbox[2]:.02f}, {bbox[3]:.02f}, {confidence:.01f}, -1, -1, -1, {rotation:.02f}'
            yield line


DOCTEXT = f"""
    Usage:
      datumaro_to_obb_mot.py <input> <output>
"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    filename = args['<input>']
    output = args['<output>']

    try:
        os.makedirs(os.path.dirname(output), exist_ok=True)
    except (FileNotFoundError, FileExistsError): # output will be in the current directory ('' and '.')
        pass

    with open(output, 'w') as f:
        for line in obb_mot_line_generator(filename):
            print(line, end='\n', file=f)
