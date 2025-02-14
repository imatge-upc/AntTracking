
from docopt import docopt
import numpy as np
import os
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import sys
import torch

from ceab_ants.detection.process_video import process_video


QUEUE_GET_TIMEOUT = 2 # seconds
TQDM_INTERVAL = 10 # iterations


def extract_obboxes(sliced_results, initial_frame):
    processed_results = []
    frame_index = initial_frame
    
    for result in sliced_results:
        xywhr = np.array([[det.bbox.x, det.bbox.y, det.bbox.width, det.bbox.height, det.bbox.angle] 
                          for det in result.object_prediction_list])
        confidences = np.array([[det.score.value] for det in result.object_prediction_list])
        xywhr[:, -1] = np.rad2deg(xywhr[:, -1])  # Convert radians to degrees
        
        obboxes = np.concatenate((xywhr, confidences), axis=1)
        processed_results.append((frame_index, obboxes))
        frame_index += 1
    
    return processed_results, frame_index

def main(video_source, model_path, output, queue_size=8, batch_size=4, min_batch_size=1, initial_frame=0, num_frames=-1, slice_size=640, overlap=0.2, confidence_threshold=0.3):

    def build_model():
        detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"SAHI model loaded on {detection_model.device}")
        return detection_model

    def apply_model(model, batch):
        results = [
            get_sliced_prediction(
                image,
                model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
            ) for image in batch
        ]
        return results

    process_video(video_source, output, build_model, apply_model, None, extract_obboxes, queue_size, batch_size, min_batch_size, initial_frame=initial_frame, num_frames=num_frames, timeout_get=QUEUE_GET_TIMEOUT, tqdm_interval=TQDM_INTERVAL)    


DOCTEXT = """
Usage:
  ant_detection_yolo_sahi_oriented_gpu.py <video_source> <output> <model_path> [--queue_size=<int>] [--batch_size=<int>] [--slice_size=<int>] [--overlap=<float>] [--confidence_threshold=<float>]
  ant_detection_yolo_sahi_oriented_gpu.py -h | --help

Options:
  --queue_size=<int>             Max size of the frame queue [default: 40].
  --batch_size=<int>             Number of frames per batch for YOLO inference [default: 8].
  --slice_size=<int>             Size of slices for SAHI inference [default: 640].
  --overlap=<float>              Overlap ratio for SAHI slicing [default: 0.2].
  --confidence_threshold=<float> Confidence threshold for YOLO detection [default: 0.3].
"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video_source = args['<video_source>']
    model_path = args['<model_path>']
    output = args["<output>"]
    
    queue_size = int(args["--queue_size"])
    batch_size = int(args["--batch_size"])
    slice_size = int(args["--slice_size"])
    overlap = float(args["--overlap"])
    confidence_threshold = float(args["--confidence_threshold"])
    
    os.makedirs(os.path.dirname(output), exist_ok=True)
    
    main(video_source, model_path, output, queue_size=queue_size, batch_size=batch_size, slice_size=slice_size, overlap=overlap, confidence_threshold=confidence_threshold)
