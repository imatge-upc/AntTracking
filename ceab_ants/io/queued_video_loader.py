
from contextlib import contextmanager
from threading import Thread

from ceab_ants.io.video_contextmanager import VideoCapture


QUEUE_PUT_TIMEOUT = 0.5

class QueuedVideoLoader():

    @classmethod
    def init_start(cls, video_path, frame_queue, metadata_queue, queue_lock, stop_event, initial_frame=0, preprocess_func=None):
        try:
            loader = cls(video_path, frame_queue, metadata_queue, queue_lock, stop_event, initial_frame, preprocess_func)
            loader.start()
        except Exception as e:
            print(f"Error in VideoLoader for video {video_path}: {e}")
            stop_event.set()

    def __init__(self, video_path, frame_queue, metadata_queue, queue_lock, stop_event, initial_frame=0, preprocess_func=None):
        
        self.video_path = video_path
        
        self.frame_queue = frame_queue
        self.metadata_queue = metadata_queue
        self.queue_lock = queue_lock

        self.stop_event = stop_event

        self.initial_frame = initial_frame
        self.fr = 0

        if preprocess_func is None:
            # preprocess prepares model input and metadata. Metadata is a tuple with at least (video id, frame number, number of model outputs)
            self.preprocess_func = lambda x, fr=-1 : ([[x]], (0, fr, 1)) # np.transpose(x / 255.0, (2, 0, 1)).astype(np.float32)[np.newaxis, ...]
        else:
            self.preprocess_func = preprocess_func

    def frame_generator(self):
        with VideoCapture(self.video_path) as cap:
            self.fr = 0
            while not self.stop_event.is_set():

                ret, frame = cap.read()
                if not ret:
                    self.stop_event.set()
                    break

                self.fr += 1
                if self.fr < self.initial_frame:
                    continue

                yield frame, self.fr

    def start(self):
        for frame, fr in self.frame_generator():

            processed_data, metadata = self.preprocess_func(frame, fr)
            processed_data = processed_data if isinstance(processed_data, list) else [processed_data]
            batch_size = len(processed_data)

            while metadata and not self.stop_event.is_set():
                with self.queue_lock:
                    free_space = self.frame_queue.maxsize - self.frame_queue.qsize()
                    if free_space >= batch_size:
                        for data in processed_data:
                            self.frame_queue.put(data)
                        self.metadata_queue.put(metadata)
                        break
                    
                    elif self.frame_queue.maxsize < batch_size:
                        self.stop_event.set()
                        raise Exception(f'After preprocessig the frame, the program tries to put {batch_size=} on a queue with maxsize {self.frame_queue.maxsize}')
                    else:
                        self.stop_event.wait(QUEUE_PUT_TIMEOUT)

    __call__ = start

@contextmanager
def QueuedVideoLoaderContext(video_path, frame_queue, metadata_queue, queue_lock, stop_event, initial_frame=0, preprocess_func=None):
    video_loader = QueuedVideoLoader(video_path, frame_queue, metadata_queue, queue_lock, stop_event, initial_frame=initial_frame, preprocess_func=preprocess_func)
    video_loader_thread = Thread(target=video_loader.start, daemon=True)
    try:
        video_loader_thread.start()
        yield
    finally:
        stop_event.set()
        video_loader_thread.join()
