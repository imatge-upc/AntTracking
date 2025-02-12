
from contextlib import contextmanager
import cv2
import queue
import threading


class VideoWriter:
    def __init__(self, output_path, frame_queue, stop_event, fps=7.5, resolution=(4000, 4000), fourcc='mp4v', color=True):
        self.output_path = output_path
        self.fps = fps
        self.resolution = tuple(int(dim) for dim in resolution)
        self.color = color
        
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.video_writer = cv2.VideoWriter(output_path, self.fourcc, fps, self.resolution, color)
        
        self.frame_queue = frame_queue
        self.stop_event = stop_event
    
    def write(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                self.video_writer.write(frame)
            except queue.Empty:
                continue
    
    def stop(self):
        self.stop_event.set()
        self.video_writer.release()
        print("VideoWriter released.")
        
@contextmanager
def VideoWriterContext(output_path, frame_queue, stop_event, fps=7.5, resolution=(4000, 4000), fourcc='mp4v', color=True):
    video_writer = VideoWriter(output_path, frame_queue, stop_event, fps, resolution, fourcc, color)
    writer_thread = threading.Thread(target=video_writer.write, daemon=True)
    try:
        writer_thread.start()
        yield video_writer
    finally:
        video_writer.stop()
        writer_thread.join()
