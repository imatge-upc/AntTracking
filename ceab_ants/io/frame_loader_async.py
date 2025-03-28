
from contextlib import contextmanager, ExitStack
from multiprocessing.pool import ThreadPool
import numpy as np
import queue
from threading import Thread, Event, Lock

from ceab_ants.io.queued_video_loader import QueuedVideoLoader


QUEUE_PUT_TIMEOUT = 0.5

# TODO: New QueuedVideoLoader so modify to make it work and use it


@contextmanager
def ContextEvent():
    event = Event()
    try:
        yield event
    finally:
        event.set()

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
