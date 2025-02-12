
import os
import queue
import sys
from threading import Event
from tqdm import tqdm

from ceab_ants.detection.models.bgfg.foreground_mask_object_detector import build_detector as build_bgfg_detector
from ceab_ants.detection.process_video import process_video
from ceab_ants.io.queued_frame_loading import QueuedVideoLoaderContext

from docopts.help_ant_detection_bgfg_oriented import parse_args


QUEUE_GET_TIMEOUT = 2 # seconds
TQDM_INTERVAL = 10 # iterations


def main(input_video, detection_file, var_thresh, min_size, start_write, num_frames, lr_train, lr, frames_train, queue_size=4):

    # Build the model
    def build_model():
        detector_model = build_bgfg_detector(var_thresh, min_size, close_kernel=5, open_kernel=2)

        frame_queue = queue.Queue(maxsize=queue_size)
        stop_event = Event()
        fr = 0
        print (f'Training', file=sys.stderr)
        with QueuedVideoLoaderContext(input_video, video_id=0, frame_queue=frame_queue, stop_event=stop_event, preprocess_func=None):

            def generator(fr):
                while not stop_event.is_set() and ((frames_train < 0) or (fr < frames_train)):
                    try:
                        _, frame = frame_queue.get(timeout=QUEUE_GET_TIMEOUT)
                        yield frame
                    except queue.Empty:
                        if stop_event.is_set():
                            break

            for frame in tqdm(generator(fr), mininterval=TQDM_INTERVAL, maxinterval=TQDM_INTERVAL):
                    fr = fr + 1
                    detector_model.train(frame, lr=lr_train)
        
        return detector_model
    
    apply_model = lambda model, batch: model(batch[0], lr=lr)

    process_video(input_video, detection_file, build_model, apply_model, None, None, queue_size, 1, 1, start_write, num_frames, QUEUE_GET_TIMEOUT, TQDM_INTERVAL)
                

if __name__ == '__main__':
    # read arguments
    (input_video, detection_file, var_thresh, 
     min_size, start_write, num_frames, lr_train, lr,
     frames_train, queue_size) = parse_args(sys.argv)

    os.makedirs(os.path.dirname(detection_file), exist_ok=True)

    main(input_video, detection_file, var_thresh, min_size, start_write, num_frames, lr_train, lr, frames_train, queue_size)
