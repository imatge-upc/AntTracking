
from docopt import docopt
import numpy as np
import os
import sys
import torch
from ultralytics import YOLO

from ceab_ants.detection.process_video import process_video


QUEUE_GET_TIMEOUT = 2 # seconds
TQDM_INTERVAL = 10 # iterations


def extract_obboxes(yolo_results, initial_frame):

    processed_results = []
    frame_index = initial_frame
    
    for result in yolo_results:
        xywhr = result.obb.xywhr.cpu().numpy().reshape(-1, 5)
        confidences = result.obb.conf.cpu().numpy().reshape(-1, 1)
        xywhr[:, -1] = np.rad2deg(xywhr[:, -1])

        obboxes = np.concatenate((xywhr, confidences), axis=1)
        processed_results.append((frame_index, obboxes))

        frame_index += 1

    return processed_results, frame_index


def main(video_source, model_path, output, queue_size=8, batch_size=4, min_batch_size=1, initial_frame=0, num_frames=-1):

    def build_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = YOLO(model_path)
        model.to(device)

        print(f"model built on {device}")

        return model

    def apply_model(model, batch):
        results = model.predict(batch, verbose=False)
        return results

    process_video(video_source, output, build_model, apply_model, None, extract_obboxes, queue_size, batch_size, min_batch_size, initial_frame=initial_frame, num_frames=num_frames, timeout_get=QUEUE_GET_TIMEOUT, tqdm_interval=TQDM_INTERVAL)    


DOCTEXT = """
Usage:
  ant_detection_yolo_resize_oriented_gpu.py <video_source> <output> <model_path> [--queue_size=<int>] [--batch_size=<int>]
  ant_detection_yolo_resize_oriented_gpu.py -h | --help

Options:
  --queue_size=<int>       Max size of the frame queue [default: 40].
  --batch_size=<int>       Number of frames per batch for YOLO inference [default: 8].

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video_source = args['<video_source>']
    model_path = args['<model_path>']
    output = args["<output>"]

    queue_size = int(args["--queue_size"])
    batch_size = int(args["--batch_size"])

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    main(video_source, model_path, output, queue_size=queue_size, batch_size=batch_size)
