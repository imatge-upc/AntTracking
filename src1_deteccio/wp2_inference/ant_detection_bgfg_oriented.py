
import os
import sys

from ceab_ants.detection.single_video_processor import SingleVideoProcessor

from docopts.help_ant_detection_bgfg_oriented import parse_args
from input_utils.resize_loader import preprocess
from models.bgfg import BGFG


QUEUE_GET_TIMEOUT = 2 # seconds
TQDM_INTERVAL = 10 # iterations
                

if __name__ == '__main__':

    batch_size = 4
    min_batch_size = 2
    tqdm_interval = 1

    (input_video, detection_file, var_thresh, 
     min_size, start_write, num_frames, lr_train, lr,
     frames_train, queue_size) = parse_args(sys.argv)

    os.makedirs(os.path.dirname(detection_file) or '.', exist_ok=True)

    model = BGFG(input_video, var_thresh, min_size, close_kernel=5, open_kernel=2, frames_train=frames_train, lr_train=lr_train, lr_infer=lr, queue_size=1, verbose=True)

    worker = SingleVideoProcessor(model.build_model, model.apply_model, preprocess, model.postprocess, queue_size, batch_size, min_batch_size, tqdm_interval=tqdm_interval)

    worker.process_video(input_video, detection_file, num_frames, start_write)
