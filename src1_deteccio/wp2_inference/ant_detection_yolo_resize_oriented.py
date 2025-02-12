
# NOTE: Easy version to begin and understand how it works, other version may be faster but I/O is more complex (I'm working towards loading from N and writting to N, 'cause my boss asked me)

from docopt import docopt
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
from ultralytics import YOLO

from ceab_ants.io.video_contextmanager import VideoCapture


def extract_obboxes(yolo_results):

    result = yolo_results[0]
    
    xywhr = result.obb.xywhr.cpu().numpy().reshape(-1, 5) # Input dimensions
    confidences = result.obb.conf.cpu().numpy().reshape(-1, 1)
    xywhr[:, -1] = np.rad2deg(xywhr[:, -1])

    obboxes = np.concatenate((xywhr, confidences), axis=1)

    return obboxes

def main(video_source, model_path, output):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = YOLO(model_path)
    model.to(device)

    with VideoCapture(video_source) as cap:
        with open(output, 'w') as out_file:
            fr = 0
            
            def generator():
                while cap.isOpened():
                    yield

            for _ in tqdm(generator(), mininterval=10, maxinterval=10):
                fr += 1

                # TODO: Modify to batch more frames at once (now @ UPC: 18min 49s for 9014 frames -> x12 -> 03h 45min 48s + tracking per 20 min 20 formigues (Total Tracking took: 71.159 seconds for 9013 frames or 126.7 FPS))

                if (fr == 1) or (fr == 5) or (fr == 10) or (fr == 25) or (fr == 50) or (fr % 100 == 0):
                    print(f'Processing frame {fr}', file=sys.stderr)

                _, frame = cap.read()
                if frame is None:
                    break

                results = model.predict(frame, verbose=False)
                obboxes = extract_obboxes(results)

                if len(obboxes) > 0:
                    # bbox = (x, y, w, h, r, s) -> (fr, -1, x, y, w, h, s, -1, -1, -1, r)
                    MOTDet_line = lambda fr, obbox : f'{fr:.0f},-1,{obbox[0]:.5f},{obbox[1]:.5f},{obbox[2]:.5f},{obbox[3]:.5f},{obbox[5]:.5f},-1,-1,-1,{obbox[4]:.1f}'
                    detection_text = '\n'.join([MOTDet_line(fr, obbox) for obbox in obboxes])
                    print(detection_text, end='\n', file=out_file)

DOCTEXT = """
Usage:
  ant_detection_yolo_resize.py <video_source> <output> <model_path>
  ant_detection_yolo_resize.py -h | --help

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video_source = args['<video_source>']
    model_path = args['<model_path>'] # Training 22
    output = args["<output>"]

    os.makedirs(os.path.dirname(output), exist_ok=True)

    main(video_source, model_path, output)
