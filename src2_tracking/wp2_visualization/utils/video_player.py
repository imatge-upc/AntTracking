
# TODO: Object that gets a stream of frames and plot them at a fixed framerate. 
# TODO: With memory to go back a certain number of frames
# TODO: With buffer to go forward (It's meant to be a thread)

# TODO: TODO_WAY_LATER: Go back, pause, speed up/down, start recording, stop recording, frame to frame back and fordward
# TODO: TODO_WAY_LATER: graphical interfice (apart from the basic plotting windows)

from contextlib import contextmanager
import cv2
import numpy as np
from queue import Queue, Empty
from threading import Thread
import time


class VideoPlayer:
    def __init__(self, frame_queue, stop_event, framerate=7.5, memory_size=50, buffer_size=10, window_name=None, output_resolution=None):
        self.frame_queue = frame_queue
        self.stop_event = stop_event

        self.window_name = window_name or "VideoPlayer"
        self.output_resolution = output_resolution or (4000, 3000)

        self.framerate = framerate
        self.time_interval = 1.0 / framerate
        
        self.past_frames = [] 
        self.memory_size = memory_size
        self.future_frames = Queue(maxsize=buffer_size)
        self.buffer_size = buffer_size
    
    def play(self):

        fr = 0
        while not self.stop_event.is_set():
            start_time = time.time()
            
            # TODO: block the frame_queue when reading so no skip and future_frames not get an async frame 
            if self.future_frames.qsize():
                # TODO: a thread that fill self.future_frames, and set a signal to take from futur_frames
                frame = self.future_frames.get()
            else:
                try:
                    frame = self.frame_queue.get(timeout=1)
                except Empty:
                    continue
            
            self.past_frames.append(frame)
            if len(self.past_frames) > self.memory_size:
                self.past_frames.pop(0)
            
            fr += 1
            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            elapsed_time = time.time() - start_time
            time.sleep(max(0, self.time_interval - elapsed_time))
    
    def stop(self):
        self.stop_event.set()
        cv2.destroyWindow(self.window_name)

@contextmanager
def VideoPlayerContext(frame_queue, stop_event, framerate=7.5, memory_size=50, buffer_size=10, window_name=None, output_resolution=None):
    video_player = VideoPlayer(frame_queue, stop_event, framerate=framerate, memory_size=memory_size, buffer_size=buffer_size, window_name=window_name, output_resolution=output_resolution)
    video_player_thread = Thread(target=video_player.play, daemon=True)
    try:
        video_player_thread.start()
        yield
    finally:
        video_player.stop()
        video_player_thread.join()
