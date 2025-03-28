
import queue
import signal
import sys
from threading import Event, Lock
import time

from ceab_ants.io.queued_video_loader import QueuedVideoLoaderContext
from ceab_ants.io.obb_writer import OBBPostProcessWriterContext


QUEUE_GET_TIMEOUT = 2 # seconds


class SingleVideoProcessor():

    def __init__(self, build_model, apply_model, preprocess=None, postprocess=None, queue_size=64, max_batch_size=16, min_batch_size=None, tqdm_interval=1):
        self.build_model = build_model
        self.apply_model = apply_model
        self.preprocess = preprocess
        self.postprocess = postprocess
        
        self.queue_size = queue_size
        self.batch_size = max_batch_size
        self.min_batch_size = min_batch_size or max_batch_size

        self.stop_event_load = None
        self.stop_event_write = None
        self.output_queue = None

        self.tqdm_interval=tqdm_interval

    def set_stop_signal(self):
        def handler(sig, frame):
            if self.stop_event_load is not None:
                self.stop_event_load.set()
            if self.stop_event_write is not None:
                self.stop_event_write.set()
                elems = self.output_queue.qsize()
                prev = elems
                print(f'\nStopping - Flushing {prev} / {elems}', flush=True)
                while self.output_queue.qsize():
                    if self.output_queue.qsize() != prev:
                        prev = self.output_queue.qsize()
                        print(f'Stopping - Flushing {prev} / {elems}', flush=True)
                    time.sleep(1)
                print(f'Stopping', flush=True)
            sys.exit(0)
        signal.signal(signal.SIGINT, handler)

    def generator(self, input_video, frame_queue, metadata_queue, num_frames=-1, initial_frame=0):
    
        queue_lock = Lock()
        stop_event_load = Event()
        self.stop_event_load = stop_event_load

        self.set_stop_signal()

        with QueuedVideoLoaderContext(
            input_video,
            frame_queue, 
            metadata_queue,
            queue_lock, 
            stop_event_load, 
            initial_frame=0, # Either this initial_frame=initial_frame or the generator has it fr counter and initial_frame=0
            preprocess_func=self.preprocess
        ):

            fr = initial_frame
            batch = []
            while (num_frames <= 0 or fr <= (initial_frame + num_frames)) and not (stop_event_load.is_set() and frame_queue.empty()):
                try:
                    frame = frame_queue.get(timeout=QUEUE_GET_TIMEOUT)
                    batch.extend(frame)

                    if (len(batch) < self.batch_size) and (not frame_queue.empty() or (len(batch) < self.min_batch_size)):
                        continue
                    
                    yield batch
                    batch = []
                    fr += 1
                except queue.Empty:
                    if stop_event_load.is_set():
                        break

        if not stop_event_load.is_set():
            if len(batch) > 0:
                yield batch

    def process_video(self, input_video, output_file, num_frames, initial_frame):

        frame_queue = queue.Queue(maxsize=self.queue_size)
        metadata_queue = queue.Queue()

        model = self.build_model()

        output_queue = queue.Queue()
        stop_event_write = Event()
        self.stop_event_write = stop_event_write
        self.output_queue = output_queue

        with OBBPostProcessWriterContext(output_queue, metadata_queue, stop_event_write, output_file, postprocess_func=self.postprocess, tqdm_interval=self.tqdm_interval):
            for batch in self.generator(input_video, frame_queue, metadata_queue, num_frames, initial_frame):
                results = self.apply_model(model, batch)
                for result in results:
                    output_queue.put(result)

    