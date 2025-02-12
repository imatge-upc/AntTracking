# Be careful of using CVAT annotations with the original video as their height and width is half the original one.

from docopt import docopt
import cv2 as cv
import numpy as np
import os
from shapely import Polygon, affinity
import shutil
import sys
from tqdm import tqdm

from ceab_ants.io.mot_loader import PrecomputedMOTDetector, PrecomputedOMOTDetector
from ceab_ants.io.video_contextmanager import VideoCapture

from utils.gt_crops import adjust_bbox_annotations, adjust_obbox_annotations, random_crop, filter_annotations


def obb_bounds(x, y, w, h, a, _):
    rectangle = Polygon([(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)])
    obb = affinity.translate(affinity.rotate(rectangle, a, use_radians=False), x, y)
    return obb.bounds

def x1y1x2y2_to_bboxes(bboxes):
    bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
    return bboxes

def obboxes_to_bboxes(obboxes):
    obboxes = obboxes.copy()
    bounds = np.asarray([obb_bounds(*obb) for obb in obboxes]).reshape(-1, 4)
    obboxes[:, 0] = bounds[:, 0]
    obboxes[:, 1] = bounds[:, 1]
    obboxes[:, 2] = bounds[:, 2] - bounds[:, 0]
    obboxes[:, 3] = bounds[:, 3] - bounds[:, 1]
    return obboxes[:, [0, 1, 2, 3, 5]]

def process_video(video_path, seq_path, sampling_rate, test_frac, crop_width, crop_height, basename, val_img_dir, val_label_dir, train_img_dir, train_label_dir, video_id, obbox=False, obb_min_area=120, verbose=True):

    if obbox:
        mot_loader = PrecomputedOMOTDetector
        base_to_tracks = obboxes_to_bboxes
        adjust_annotations = lambda tracks, within, initial, crop_width, crop_height : adjust_obbox_annotations(tracks, within, initial, crop_width, crop_height, min_area=obb_min_area)
        mot2yolo = lambda trk : ['0', f'{trk[0]}', f'{trk[1]}', f'{trk[2]}', f'{trk[3]}', f'{trk[4]}', f'{trk[5]}', f'{trk[6]}', f'{trk[7]}']
    else:
        mot_loader = PrecomputedMOTDetector
        base_to_tracks = x1y1x2y2_to_bboxes
        adjust_annotations = lambda tracks, within, initial, crop_width, crop_height : adjust_bbox_annotations(x1y1x2y2_to_bboxes(tracks), within, initial, crop_width, crop_height)
        mot2yolo = lambda trk : ['0', f'{trk[0]}', f'{trk[1]}', f'{trk[2]}', f'{trk[3]}']

    tracker = mot_loader(seq_path, verbose=verbose)
    tracker = mot_loader(seq_path, max_rows=tracker.last_row + 1, verbose=verbose)

    #save_frames = np.arange(1, tracker.last_frame, sampling_rate, dtype=int)
    save_frames = np.unique(tracker.seq_dets[:, 0])[::sampling_rate].astype(int)

    valid_frames = save_frames.copy()
    #np.random.shuffle(valid_frames)
    valid_frames = valid_frames[: int(len(valid_frames) * test_frac)]
    
    with VideoCapture(video_path) as capture:

        width  = capture.get(cv.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv.CAP_PROP_FRAME_HEIGHT)

        if verbose:
            print(f"INPUT (WIDTH, HEIGHT) = ({width}, {height})")

        if crop_width > width or crop_height > height:
            print(f'crop_width = {crop_width} >? width = {width}\ncrop_height = {crop_height} >? height = {height}')
            raise Exception(f'crop_width = {crop_width} >? width = {width}\ncrop_height = {crop_height} >? height = {height}')

        for fr in tqdm(save_frames):

            tracks_base = tracker(fr, override=True)
            tracks = base_to_tracks(tracks_base)

            tracks[:, 0] = np.maximum(tracks[:, 0], 0)
            tracks[:, 1] = np.maximum(tracks[:, 1], 0)
            tracks[:, 2] = np.minimum(tracks[:, 2], width - tracks[:, 0])
            tracks[:, 3] = np.minimum(tracks[:, 3], height - tracks[:, 1])

            capture.set(cv.CAP_PROP_POS_FRAMES, fr - 1)
            _, frame = capture.read()
            if frame is None:
                print (f'Frame {fr} is None')
                break
            
            # Crop and each uncentered and uncut ant group is in one and only one output!
            seen = np.full(len(tracks), False)
            seen[tracks[:, 2] > crop_width] = True
            seen[tracks[:, 3] > crop_height] = True
            idx = 1
            while not np.all(seen):

                initial, final = random_crop(tracks, seen, crop_width, crop_height, width, height)
                img = frame[initial[1] : final[1], initial[0] : final[0]] # OpenCV images are (h, w), the rest of the code is (w, h)

                within, seen = filter_annotations(tracks, seen, initial, final)
                tracks_save = adjust_annotations(tracks_base, within, initial, crop_width, crop_height)

                base_filename = f'{basename}_{video_id}_{fr:06}_{idx}_{len(tracks_save)}'
                filename = f'{base_filename}.png'
                labels_filename = f'{base_filename}.txt'
                idx += 1

                labels = '\n'.join([' '.join(mot2yolo(trk)) for trk in tracks_save])

                if fr in valid_frames:
                    cv.imwrite(os.path.join(val_img_dir, filename), img)
                    with open(os.path.join(val_label_dir, labels_filename), 'w') as f:
                        f.write(labels)
                else:
                    cv.imwrite(os.path.join(train_img_dir, filename), img)
                    with open(os.path.join(train_label_dir, labels_filename), 'w') as f:
                        f.write(labels)

DOCTEXT = f"""
Usage:
  video_with_gt_to_crops.py [obbox] <output_file> (<video_path> <seq_path>)... [--test_frac=<tf>] [--sampling_rate=<sr>] [--width=<w>] [--height=<h>] [--obb_min_area=<oma>]

Options:
  --test_frac=<tf>          The fraction of frames used for testing. [default: 0.3]
  --sampling_rate=<sr>      Number of frames skipped between saved images. [default: 2]
  --width=<w>               Width of the crop. [default: 640]
  --height=<h>              Height of the crop. [default: 640]
  --obb_min_area=<oma>      Minimum area of a cropped obb [default: 120]

"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    obbox = args['obbox']

    video_pathes = args['<video_path>']
    seq_pathes = args['<seq_path>']
    
    output_file = args['<output_file>']

    test_frac = float(args['--test_frac'])
    sampling_rate = int(args['--sampling_rate'])

    crop_width = int(args['--width'])
    crop_height = int(args['--height'])

    obb_min_area = int(args['--obb_min_area'])

    basename = os.path.basename(output_file)
    yolo_config_dir = os.path.join(output_file, basename)
    train_img_dir = os.path.join(output_file, basename, 'images', 'train')
    val_img_dir = os.path.join(output_file, basename, 'images', 'val')
    train_label_dir = os.path.join(output_file, basename, 'labels', 'train')
    val_label_dir = os.path.join(output_file, basename, 'labels', 'val')

    os.makedirs(yolo_config_dir, exist_ok=False)
    os.makedirs(train_img_dir, exist_ok=False)
    os.makedirs(val_img_dir, exist_ok=False)
    os.makedirs(train_label_dir, exist_ok=False)
    os.makedirs(val_label_dir, exist_ok=False)

    for i, (video_path, seq_path) in enumerate(zip(video_pathes, seq_pathes)):
        print(f'VIDEO {i + 1} OF {len(video_pathes)}')
        process_video(video_path, seq_path, sampling_rate, test_frac, crop_width, crop_height, basename, val_img_dir, val_label_dir, train_img_dir, train_label_dir, i, obbox=obbox, obb_min_area=obb_min_area, verbose=True)

    config_text = f"""
    # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    path: ./{output_file}  # dataset root dir
    train: images/train  # train images (relative to 'path')
    val: images/val  # val images (relative to 'path')
    test:  # test images (optional)

    # Classes
    names:
        0: ant

    """

    with open(os.path.join(yolo_config_dir, f'{basename}.yaml'), 'w') as f:
        f.write(config_text)

    shutil.make_archive(output_file, 'zip', output_file)
    shutil.rmtree(output_file)
