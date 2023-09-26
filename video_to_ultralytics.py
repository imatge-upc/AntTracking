# Be careful of using CVAT annotations with the original video as their height and width is half the original one.

from docopt import docopt
from contextlib import contextmanager
import cv2 as cv
import numpy as np
import os
import shutil
import sys


@contextmanager
def VideoCapture(input_video):

    # findFileOrKeep allows more searching paths
    capture = cv.VideoCapture(cv.samples.findFileOrKeep(input_video))
    if not capture.isOpened():
        print('Unable to open: ' + input_video, file=sys.stderr)
        exit(0)

    try:
        yield capture
    finally:
        # Release the video capture object at the end
        capture.release()


class PrecomputedMOTTracker():

    def __init__(self, seq_path=None, verbose=False):
        self.seq_dets = np.loadtxt(seq_path, delimiter=',')
        
        self.last_frame = int(self.seq_dets[:, 0].max())
        
        self.verbose = verbose

        self.frames_seen = 0
    
    def __call__(self, frame):

        self.frames_seen = (self.frames_seen + 1) % 500
        if self.verbose and self.frames_seen == 0:
            print (f'Processing frame {frame}', file=sys.stderr)

        tracks = self.seq_dets[self.seq_dets[:, 0] == frame, :]
        return tracks


DOCTEXT = f"""
Usage:
  video_to_ultralytics.py <video_path> <seq_path> <output_file> [--test_frac=<tf>] [--sampling_rate=<sr>]

Options:
  --test_frac=<tf>          The fraction of frames used for testing. [default: 0.3]
  --sampling_rate=<sr>      Number of frames skipped between saved images. [default: 2]

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    video_path = args['<video_path>']
    seq_path = args['<seq_path>']
    output_file = args['<output_file>']

    test_frac = float(args['--test_frac'])
    sampling_rate = int(args['--sampling_rate'])

    basename = os.path.basename(output_file)
    yolo_config_dir = os.path.join(output_file, basename)
    train_img_dir = os.path.join(output_file, basename, 'images', 'train')
    val_img_dir = os.path.join(output_file, basename, 'images', 'val')
    train_label_dir = os.path.join(output_file, basename, 'labels', 'train')
    val_label_dir = os.path.join(output_file, basename, 'labels', 'val')

    tracker = PrecomputedMOTTracker(seq_path, verbose=True)
    save_frames = np.arange(1, tracker.last_frame, sampling_rate, dtype=int)

    valid_frames = save_frames.copy()
    np.random.shuffle(valid_frames)
    valid_frames = valid_frames[: int(len(valid_frames) * test_frac)]

    os.makedirs(yolo_config_dir, exist_ok=False)
    os.makedirs(train_img_dir, exist_ok=False)
    os.makedirs(val_img_dir, exist_ok=False)
    os.makedirs(train_label_dir, exist_ok=False)
    os.makedirs(val_label_dir, exist_ok=False)

    with VideoCapture(video_path) as capture:

        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

        for fr in save_frames:

            tracks = tracker(fr)

            capture.set(cv.CAP_PROP_POS_FRAMES, fr - 1)
            _, frame = capture.read()
            if frame is None:
                print (f'Frame {fr} is None')
                break
            
            filename = f'{basename}_{fr:06}_{len(tracks)}.png'
            labels_filename = f'{basename}_{fr:06}_{len(tracks)}.txt'

            mot2yolo = lambda trk : ['0', f'{(trk[2] + (trk[4] / 2)) / width}', f'{(trk[3] + (trk[5] / 2)) / height}', f'{trk[4] / width}', f'{trk[5] / height}']
            labels = '\n'.join([' '.join(mot2yolo(trk)) for trk in tracks])

            if fr in valid_frames:
                cv.imwrite(os.path.join(val_img_dir, filename), frame)
                with open(os.path.join(val_label_dir, labels_filename), 'w') as f:
                    f.write(labels)
            else:
                cv.imwrite(os.path.join(train_img_dir, filename), frame)
                with open(os.path.join(train_label_dir, labels_filename), 'w') as f:
                    f.write(labels)

    config_text = f"""
    # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    path: ./{output_file}  # dataset root dir
    train: images/train  # train images (relative to 'path') 128 images
    val: images/val  # val images (relative to 'path') 128 images
    test:  # test images (optional)

    # Classes
    names:
        0: ant

    """

    with open(os.path.join(yolo_config_dir, f'{basename}.yaml'), 'w') as f:
        f.write(config_text)

    shutil.make_archive(output_file, 'zip', output_file)
    shutil.rmtree(output_file)
