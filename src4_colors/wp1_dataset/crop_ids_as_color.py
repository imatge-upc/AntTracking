
# input: video, mot file, output_base_folder, set of ids per color

# Read video and mot file
# If id in color id set : save it in color subfolder
# Else : save it on black
 
# Later review color folders and separate crops that I think color can't be seen

import cv2
from docopt import docopt
import os
import sys

from ceab_ants.io.mot_loader import PrecomputedOMOTDetector
from ceab_ants.io.video_contextmanager import VideoCapture


NO_COLOR = "black"
COLORS = [
    "red",
    "green",
    "blue",
    "pink",
    "azure",
    "yellow"
]

DOCTEXT = f"""
    Usage:
      crop_ids_as_color.py <video> <mot_file> <output_folder> [--downsampling=<d>] {' '.join([f"[--{COLOR}=<{COLOR[:2]}>...]" for COLOR in COLORS])}

    Options:
      --downsampling=<d>    For each abs(<d>) frames, 1 will be used [default: 1]
"""


if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video = args['<video>']
    mot_file = args['<mot_file>']
    output_folder = args['<output_folder>']

    downsampling = abs(int(args['--downsampling']))
    color_sets = { COLOR : {int(id_) for id_ in args[f"--{COLOR}"]} for COLOR in COLORS}

    try:
        os.makedirs(output_folder, exist_ok=True)
    except (FileNotFoundError, FileExistsError): # output will be in the current directory ('' and '.')
        pass
    
    os.makedirs(f'{output_folder}/{NO_COLOR}/', exist_ok=True)
    for color, set_ in color_sets.items():
        if len(set_):
            os.makedirs(f'{output_folder}/{color}/', exist_ok=True)

    detector = PrecomputedOMOTDetector(mot_file, verbose=True)

    with VideoCapture(video) as capture:
        current_frame = 0
        for i, fr in enumerate(range(1, detector.last_frame, downsampling)):
            detections = []
            offset = -1
            while (len(detections) == 0) and (offset < downsampling):
                offset += 1
                detector.current_frame = fr + offset
                detections = detector(i)
            
            if len(detections) == 0 : continue

            # MKV doesn't work with cv2.CAP_PROP_POS_FRAMES
            for _ in range(fr + offset - current_frame):
                _, frame = capture.read()
                current_frame += 1

            for det in detections:

                color = NO_COLOR
                for COLOR in COLORS:
                    if int(det[5]) in color_sets[COLOR]:
                        color = COLOR
                        break

                x = int(det[0])
                y = int(det[1])
                w = int(det[2])
                h = int(det[3])
                angle = det[4]

                x1 = x - w // 2
                x2 = x + w // 2
                y1 = y - h // 2
                y2 = y + h // 2

                M = cv2.getRotationMatrix2D((x, y), angle, 1.0)
                rotated = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]), flags=cv2.INTER_LANCZOS4)
                cv2.imwrite(f'{output_folder}/{color}/{fr + offset:09}_{int(det[5])}.png', rotated[ y1 : y2, x1 : x2])
