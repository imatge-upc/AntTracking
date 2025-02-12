
from contextlib import contextmanager
import queue
from threading import Barrier, Event, Thread, Lock, BrokenBarrierError

from ceab_ants.io.queued_frame_loading import QueuedVideoLoaderContext


QUEUE_GET_TIMEOUT = 0.5
ALL_FRAMES_TIMEOUT = 6


class SyncFrameLoader():

    def __init__(self, video_list, max_buffer=5, initial_frame=0, preprocess_func=None):
        self.video_list = video_list
        self.queues = [queue.Queue(maxsize=max_buffer) for _ in video_list]
        self.stop_event = Event()
        self.barrier = Barrier(len(video_list) + 1)
        self.threads = []

        self.initial_frame = initial_frame
        self.preprocess_func = preprocess_func # Maybe we can apply homography beforehand (input: x, video_id)

    def _video_loader_thread(self, video_path, video_id, initial_frame=0, preprocess_func=None):
        preprocess_func_one = None
        if preprocess_func is not None:
            preprocess_func_one = lambda x : preprocess_func(x, video_id)
            
        with QueuedVideoLoaderContext(video_path, video_id, self.queues[video_id], Lock(), self.stop_event, initial_frame=initial_frame, preprocess_func=preprocess_func_one):
            while not self.stop_event.is_set():
                if not self.queues[video_id].empty():
                    try:
                        self.barrier.wait(timeout=QUEUE_GET_TIMEOUT) # Maybe some video stop early
                    except BrokenBarrierError:
                        self.barrier.reset() # When a barrier timesout, it break and all threads raise the error until it is reset.
                        continue # Whenever one thread check if someone stopped, check it too
                else:
                    self.stop_event.wait(QUEUE_GET_TIMEOUT)

    def start(self):
        for video_id, video_path in enumerate(self.video_list):
            thread = Thread(
                target=self._video_loader_thread,
                args=(video_path, video_id, self.initial_frame, self.preprocess_func),
                daemon=True
            )
            thread.start()
            self.threads.append(thread)
    
    def frames_generator(self):
        while not self.stop_event.is_set():
            try:
                self.barrier.wait(timeout=ALL_FRAMES_TIMEOUT)
            except BrokenBarrierError:
                self.barrier.reset()
                continue # Whenever a lot of time passed or one thread check if someone stopped early, check it too

            frames = [queue.get(timeout=1) for queue in self.queues]  # Get one (video_id, frame) per queue
            yield frames  # Synchronized fram from all videos
    
    def stop(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()

@contextmanager
def SyncFrameLoaderContext(video_list, max_buffer=5, initial_frame=0, preprocess_func=None):
    sync_frame_loader = SyncFrameLoader(video_list, max_buffer, initial_frame, preprocess_func)

    try:
        sync_frame_loader.start()
        yield sync_frame_loader
    finally:
        sync_frame_loader.stop()
