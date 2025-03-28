
from docopt import docopt
import os
import sys

from ceab_ants.detection.single_video_processor import SingleVideoProcessor

from input_utils.resize_loader import preprocess
from models.yolo_resize import YOLO_Resize



DOCTEXT = """
Usage:
  ant_detection_yolo_resize_oriented_gpu.py <video_source> <output> <model_path> [--queue_size=<int>] [--batch_size=<int>]
  ant_detection_yolo_resize_oriented_gpu.py -h | --help

Options:
  --queue_size=<int>       Max size of the frame queue [default: 40].
  --batch_size=<int>       Number of frames per batch for YOLO inference [default: 8].

"""

if __name__ == "__main__":

    initial_frame = 0
    conf = 0.3
    tqdm_interval = 1

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video_source = args['<video_source>']
    model_path = args['<model_path>']
    output = args["<output>"]

    queue_size = int(args["--queue_size"])
    batch_size = int(args["--batch_size"])
    min_batch_size = batch_size

    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)

    model = YOLO_Resize(model_path, conf=conf, verbose=True)

    worker = SingleVideoProcessor(model.build_model, model.apply_model, preprocess, model.postprocess, queue_size, batch_size, min_batch_size, tqdm_interval=tqdm_interval)

    worker.process_video(video_source, output, -1, initial_frame)
