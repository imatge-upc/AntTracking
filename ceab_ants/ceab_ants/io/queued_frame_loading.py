
from contextlib import contextmanager, ExitStack
from multiprocessing.pool import ThreadPool
import numpy as np
import queue
from threading import Thread, Event, Lock

from ceab_ants.io.video_contextmanager import VideoCapture


QUEUE_PUT_TIMEOUT = 0.5


@contextmanager
def ContextEvent():
    event = Event()
    try:
        yield event
    finally:
        event.set()

class QueuedVideoLoader():

    @classmethod
    def init_start(cls, video_path, video_id, frame_queue, queue_lock, stop_event, initial_frame=0, preprocess_func=None):
        try:
            loader = cls(video_path, video_id, frame_queue, queue_lock, stop_event, initial_frame, preprocess_func)
            loader.start()
        except Exception as e:
            print(f"Error in VideoLoader for video {video_path}: {e}")
            stop_event.set()

    def __init__(self, video_path, video_id, frame_queue, queue_lock, stop_event, initial_frame=0, preprocess_func=None):
        self.video_path = video_path
        self.video_id = video_id
        self.frame_queue = frame_queue
        self.queue_lock = queue_lock
        self.stop_event = stop_event
        self.initial_frame = initial_frame
        if preprocess_func is None:
            self.preprocess_func = lambda x : [x] # np.transpose(x / 255.0, (2, 0, 1)).astype(np.float32)[np.newaxis, ...]
        else:
            self.preprocess_func = preprocess_func
    
    def start(self):
        with VideoCapture(self.video_path) as cap:

            fr = 0
            while not self.stop_event.is_set():

                ret, frame = cap.read()
                if not ret:
                    self.stop_event.set()
                    break

                fr += 1
                if fr < self.initial_frame:
                    continue

                processed_data = self.preprocess_func(frame)
                processed_data = processed_data if isinstance(processed_data, list) else [processed_data]
                batch_size = len(processed_data)

                while batch_size and not self.stop_event.is_set():
                    with self.queue_lock:
                        free_space = self.frame_queue.maxsize - self.frame_queue.qsize()
                        if free_space >= batch_size:
                            for data in processed_data:
                                self.frame_queue.put((self.video_id, data), timeout=QUEUE_PUT_TIMEOUT)
                                batch_size -= 1
                        else:
                            self.stop_event.wait(QUEUE_PUT_TIMEOUT)

    __call__ = start

@contextmanager
def QueuedVideoLoaderContext(video_path, video_id, frame_queue, queue_lock, stop_event, initial_frame=0, preprocess_func=None):
    video_loader = QueuedVideoLoader(video_path, video_id, frame_queue, queue_lock, stop_event, initial_frame=initial_frame, preprocess_func=preprocess_func)
    video_loader_thread = Thread(target=video_loader.start, daemon=True)
    try:
        video_loader_thread.start()
        yield
    finally:
        stop_event.set()
        video_loader_thread.join()

class QueuedFramesBatcher():

    def __init__(self, frame_queue, stop_events, output_queue, min_batch, max_batch, batch_lock):
        self.frame_queue = frame_queue
        self.stop_events = stop_events
        self.output_queue = output_queue
        self.min_batch = min_batch
        self.max_batch = max_batch
        self.batch_lock = batch_lock
    
    def start(self):

        buffer = []
        while not all( event.is_set() for event in self.stop_events ) or not self.frame_queue.empty():
            try:

                while len(buffer) < self.max_batch and not self.frame_queue.empty():
                    frame = self.frame_queue.get(block=True, timeout=0.5)
                    buffer.append(frame)

                # TODO: Maybe await for space if queue doesn't have enough space

                if len(buffer) >= self.min_batch:
                    with self.batch_lock:
                        self.output_queue.put(buffer)
                        buffer = []

            except queue.Empty:
                continue

        if buffer:
            with self.batch_lock:
                self.output_queue.put(buffer)

    __call__ = start

def frame_batch_generator(video_paths, min_batch, max_batch, preprocess_func=None):

    frame_queue = queue.Queue(maxsize=max_batch)
    queue_lock = Lock()
    stop_events = [Event() for _ in video_paths]
    output_queue = queue.Queue()
    batch_lock = Lock()

    with ThreadPool(len(video_paths)) as pool:
        with ExitStack() as stack:
            stop_events = [ stack.enter_context(ContextEvent()) for _ in range(len(video_paths)) ]

            pool.starmap_async(
                QueuedVideoLoader.init_start,
                [(video_path, video_id, frame_queue, queue_lock, stop_events[video_id], preprocess_func) for video_id, video_path in enumerate(video_paths)],
            )

            batcher = QueuedFramesBatcher(frame_queue, stop_events, output_queue, min_batch, max_batch, batch_lock)
            batcher_thread = Thread(target=batcher.start, daemon=True)
            batcher_thread.start()

            while not all(event.is_set() for event in stop_events) or not output_queue.empty():
                try:
                    batch = output_queue.get(timeout=2)
                    yield batch

                except queue.Empty:
                    pass
