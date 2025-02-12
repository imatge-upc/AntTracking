
from contextlib import contextmanager
import cv2
import sys


@contextmanager
def VideoCapture(input_video, verbose=True):

    # findFileOrKeep allows more searching paths
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(input_video))
    if not capture.isOpened():
        if verbose : print('Unable to open: ' + input_video, file=sys.stderr)
        exit(0)

    try:
        yield capture
    finally:
        # Release the video capture object at the end
        capture.release()

class CV2VideoWritter():

    def __init__(self, videoName, fps, vidSize, fourcc='mp4v', color=False):

        self.outVid = cv2.VideoWriter(videoName, cv2.VideoWriter_fourcc(*fourcc), fps, vidSize, color)

    def release(self):
        self.outVid.release()
    
    def write(self, frame):
        self.outVid.write(frame)

@contextmanager
def VideoWritter(out_video, fps, vidSize, fourcc='mp4v', color=False):
    out = CV2VideoWritter(out_video, fps, vidSize, fourcc, color)

    try: 
        yield out
    finally:
        out.release()

