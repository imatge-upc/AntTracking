
from docopt import docopt
import os
import sys

from ceab_ants.detection.single_video_processor import SingleVideoProcessor

from input_utils.resize_loader import preprocess
from models.yolo_sahi import YOLO_SAHI


DOCTEXT = """
Usage:
  ant_detection_yolo_sahi_oriented_gpu.py <video_source> <output> <model_path> [--queue_size=<int>] [--batch_size=<int>] [--imgsz=<int>] [--overlap=<float>] [--confidence_threshold=<float>]
  ant_detection_yolo_sahi_oriented_gpu.py -h | --help

Options:
  --queue_size=<int>              Max size of the frame queue [default: 40].
  --batch_size=<int>              Number of frames per batch for YOLO inference [default: 8].
  --imgsz=<int>                   Size of slices for SAHI inference [default: 640].
  --overlap=<float>               Overlap ratio for SAHI slicing [default: 0.2].
  --confidence_threshold=<float>  Confidence threshold for YOLO detection [default: 0.3].
"""

if __name__ == "__main__":

    initial_frame = 0
    tqdm_interval = 1

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video_source = args['<video_source>']
    model_path = args['<model_path>']
    output = args["<output>"]
    
    queue_size = int(args["--queue_size"])
    batch_size = int(args["--batch_size"])
    min_batch_size = batch_size
    slice_size = int(args["--imgsz"])
    overlap = float(args["--overlap"])
    confidence_threshold = float(args["--confidence_threshold"])
    
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    model = YOLO_SAHI(model_path, imgsz=slice_size, overlap=overlap, conf=confidence_threshold, verbose=True)

    worker = SingleVideoProcessor(model.build_model, model.apply_model, preprocess, model.postprocess, queue_size, batch_size, min_batch_size, tqdm_interval=tqdm_interval)

    worker.process_video(video_source, output, -1, initial_frame)
