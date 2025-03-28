
from contextlib import contextmanager
import cv2
import ffmpeg
import numpy as np
import queue
import threading
import torch


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

class VideoWriterGPU:
    def __init__(self, output_path, frame_queue, stop_event, fps=7.5, resolution=(4000, 4000), crf=32):

        w, h = tuple(int(dim) for dim in resolution)
        self.process = (
            ffmpeg
            .input('pipe:', format='rawvideo', s=f'{w}x{h}', pixel_format="bgr24", framerate=fps, codec='rawvideo', hwaccel='auto')
            .output(output_path, vcodec='h264_nvenc', rc="constqp", qp=crf, loglevel="quiet") # gpu=gpu_id
            .overwrite_output().run_async(pipe_stdin=True, pipe_stdout=False, pipe_stderr=False)
        )
        
        self.frame_queue = frame_queue
        self.stop_event = stop_event
    
    def write(self):
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                self.process.stdin.write(frame.astype(np.uint8).tobytes())
            except queue.Empty:
                continue

    def stop(self):
        self.stop_event.set()
        self.process.communicate()
        self.process.wait()
        self.process.terminate()


@contextmanager
def VideoWriterContext(output_path, frame_queue, stop_event, fps=7.5, resolution=(4000, 4000), fourcc='mp4v', color=True, gpu=None):
    gpu = gpu or torch.cuda.is_available()
    if gpu:
        video_writer = VideoWriterGPU(output_path, frame_queue, stop_event, fps, resolution)
    else:
        video_writer = VideoWriter(output_path, frame_queue, stop_event, fps, resolution, fourcc, color)
    writer_thread = threading.Thread(target=video_writer.write, daemon=True)
    try:
        writer_thread.start()
        yield video_writer
    finally:
        video_writer.stop()
        writer_thread.join()
