
import numpy as np
import queue
import sys
from threading import Event, Lock
from tqdm import tqdm

from ceab_ants.detection.models.bgfg.foreground_mask_object_detector import build_detector as build_bgfg_detector
from ceab_ants.io.queued_video_loader import QueuedVideoLoaderContext


QUEUE_GET_TIMEOUT = 2 # seconds


class BGFG():

    def __init__(self, train_video, var_thresh, min_size, close_kernel=5, open_kernel=2, frames_train=100, lr_train=-1, lr_infer=-1, queue_size=1, verbose=True, tqdm_interval=1):
        self.train_video = train_video
        self.var_thresh = var_thresh
        self.min_size = min_size
        self.close_kernel = close_kernel
        self.open_kernel = open_kernel
        self.frames_train = frames_train
        self.lr_train = lr_train
        self.lr_infer = lr_infer
        self.queue_size = queue_size

        self.verbose = verbose
        self.tqdm_interval = tqdm_interval

    def build_model(self):
        detector_model = build_bgfg_detector(self.var_thresh, self.min_size, close_kernel=5, open_kernel=2)

        frame_queue = queue.Queue(maxsize=self.queue_size)
        metadata_queue = queue.Queue()
        queue_lock = Lock()
        stop_event = Event()
        if self.verbose : print (f'Training', file=sys.stderr)

        with QueuedVideoLoaderContext(self.train_video, frame_queue=frame_queue, metadata_queue=metadata_queue, queue_lock=queue_lock, stop_event=stop_event, preprocess_func=None):

            def generator(initial_fr):
                fr = initial_fr
                while not stop_event.is_set() and ((self.frames_train < 0) or (fr < self.frames_train)):
                    try:
                        frame = frame_queue.get(timeout=QUEUE_GET_TIMEOUT)
                        yield np.array(frame[0])
                        fr = fr + 1
                    except queue.Empty:
                        if stop_event.is_set():
                            break

            for frame in tqdm(generator(0), mininterval=self.tqdm_interval, maxinterval=self.tqdm_interval):
                prev = frame.copy()
                detector_model.train(frame, lr=self.lr_train)
        
        if self.verbose : print (f'Trained', file=sys.stderr)
        return detector_model

    def apply_model(self, model, batch):
        result = [model(np.array(image), lr=self.lr_infer) for image in batch]
        return result
    
    def postprocess(self, x, metadata):
        if x:
            x = np.hstack((x[0], np.full((len(x[0]), 1), 0.3)))
        return (metadata[1], x)
