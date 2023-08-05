
from docopt import docopt
from contextlib import contextmanager
import cv2
import numpy as np
import sys


@contextmanager
def VideoCapture(input_video):

    # findFileOrKeep allows more searching paths
    capture = cv2.VideoCapture(cv2.samples.findFileOrKeep(input_video))
    if not capture.isOpened():
        print('Unable to open: ' + input_video, file=sys.stderr)
        exit(0)

    try:
        yield capture
    finally:
        # Release the video capture object at the end
        capture.release()


DOCTEXT = f"""
Usage:
  video_color_mean_std.py <video_path>

"""


if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    video_path = args['<video_path>']
    
    with VideoCapture(video_path) as cap:

        psum    = np.array([0.0, 0.0, 0.0]) # BGR
        psum_sq = np.array([0.0, 0.0, 0.0]) # BGR
        num_px  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) * int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret: # frame is None
                break

            i += 1
            if i % 100 == 0:
                print(f'{i} / nframes')

            psum = frame.sum(axis=(0, 1))
            psum_sq = (frame ** 2).sum(axis=(0, 1))
    
    total_mean = psum / num_px
    total_var  = (psum_sq / num_px) - (total_mean ** 2)
    total_std  = np.sqrt(total_var)

    print('\n\n\n')
    print(f'mean (BGR): {total_mean}')
    print(f'std (BGR): {total_std}')
