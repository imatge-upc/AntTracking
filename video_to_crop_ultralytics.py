# Be careful of using CVAT annotations with the original video as their height and width is half the original one.

from docopt import docopt
from contextlib import contextmanager
import cv2 as cv
import numpy as np
import os
import shutil
import sys

from utils.random_centered_crop import propose_random_crop, find_intersections, improve_crop, desired_movments


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

def random_crop(tracks, seen, crop_width, crop_height, width, height):
    current = np.argwhere(~seen).flatten()[0]
    trk = tracks[current]
    
    initial, final, low, high = propose_random_crop(trk, crop_width, crop_height, width, height)

    intersect = find_intersections(tracks, initial, final)

    unseen_intersect = intersect & (~seen)
    unseen_tracks = tracks[unseen_intersect, :]

    initial, final = improve_crop(unseen_tracks, initial, final, low, high, width, height)

    return initial, final

def adjust_annotations(tracks, seen, initial, final, crop_width, crop_height):
    delta_w, delta_h = desired_movments(tracks, initial, final)
    seen = seen | ((delta_w == 0) & (delta_h == 0))
    within = ((np.abs(delta_w) < tracks[:, 4]) & (np.abs(delta_h) < tracks[:, 5]))

    tracks_save = tracks[within, :].copy()
    new_left = tracks_save[:, 2] - initial[0]
    tracks_save[:, 2] = np.clip(new_left, 0, None)
    tracks_save[:, 4] = np.minimum(tracks_save[:, 4], tracks_save[:, 4] + new_left)
    tracks_save[:, 4] = np.minimum(tracks_save[:, 4], crop_width - tracks_save[:, 2])

    new_up = tracks_save[:, 3] - initial[1]
    tracks_save[:, 3] = np.clip(new_up, 0, None)
    tracks_save[:, 5] = np.minimum(tracks_save[:, 5], tracks_save[:, 5] + new_up)
    tracks_save[:, 5] = np.minimum(tracks_save[:, 5], crop_height - tracks_save[:, 3])

    return tracks_save, seen


DOCTEXT = f"""
Usage:
  video_to_crop_ultralytics.py <video_path> <seq_path> <output_file> [--test_frac=<tf>] [--sampling_rate=<sr>] [--width=<w>] [--height=<h>]

Options:
  --test_frac=<tf>          The fraction of frames used for testing. [default: 0.3]
  --sampling_rate=<sr>      Number of frames skipped between saved images. [default: 2]
  --width=<w>               Width of the crop. [default: 640]
  --height=<h>              Height of the crop. [default: 640]

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    video_path = args['<video_path>']
    seq_path = args['<seq_path>']
    output_file = args['<output_file>']

    test_frac = float(args['--test_frac'])
    sampling_rate = int(args['--sampling_rate'])

    crop_width = int(args['--width'])
    crop_height = int(args['--height'])

    basename = os.path.basename(output_file)
    yolo_config_dir = os.path.join(output_file, basename)
    train_img_dir = os.path.join(output_file, basename, 'images', 'train')
    val_img_dir = os.path.join(output_file, basename, 'images', 'val')
    train_label_dir = os.path.join(output_file, basename, 'labels', 'train')
    val_label_dir = os.path.join(output_file, basename, 'labels', 'val')

    tracker = PrecomputedMOTTracker(seq_path, verbose=True)
    #save_frames = np.arange(1, tracker.last_frame, sampling_rate, dtype=int)
    save_frames = np.unique(tracker.seq_dets[:, 0])[::sampling_rate].astype(int)

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

        if crop_width > width or crop_height > height:
            print(f'crop_width = {crop_width} >? width = {width}\ncrop_height = {crop_height} >? height = {height}')
            raise Exception(f'crop_width = {crop_width} >? width = {width}\ncrop_height = {crop_height} >? height = {height}')

        for fr in save_frames:

            tracks = tracker(fr)

            tracks[:, 2] = np.maximum(tracks[:, 2], 0)
            tracks[:, 3] = np.maximum(tracks[:, 3], 0)
            tracks[:, 4] = np.minimum(tracks[:, 4], width - tracks[:, 2])
            tracks[:, 5] = np.minimum(tracks[:, 5], height - tracks[:, 3])

            capture.set(cv.CAP_PROP_POS_FRAMES, fr - 1)
            _, frame = capture.read()
            if frame is None:
                print (f'Frame {fr} is None')
                break
            
            # Crop and each uncentered and uncut ant group is in one and only one output!
            seen = np.full(len(tracks), False)
            seen[tracks[:, 4] > crop_width] = True
            seen[tracks[:, 5] > crop_height] = True
            idx = 1
            while not np.all(seen):

                initial, final = random_crop(tracks, seen, crop_width, crop_height, width, height)
                img = frame[initial[1] : final[1], initial[0] : final[0]]

                tracks_save, seen = adjust_annotations(tracks, seen, initial, final, crop_width, crop_height)

                filename = f'{basename}_{fr:06}_{idx}_{len(tracks_save)}.jpg'
                labels_filename = f'{basename}_{fr:06}_{idx}_{len(tracks_save)}.txt'
                idx += 1

                mot2yolo = lambda trk : ['0', f'{(trk[2] + (trk[4] / 2)) / crop_width}', f'{(trk[3] + (trk[5] / 2)) / crop_height}', f'{trk[4] / crop_width}', f'{trk[5] / crop_height}']
                labels = '\n'.join([' '.join(mot2yolo(trk)) for trk in tracks_save])

                if fr in valid_frames:
                    cv.imwrite(os.path.join(val_img_dir, filename), img)
                    with open(os.path.join(val_label_dir, labels_filename), 'w') as f:
                        f.write(labels)
                else:
                    cv.imwrite(os.path.join(train_img_dir, filename), img)
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
