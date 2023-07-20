
from contextlib import contextmanager
import cv2 as cv
from docopt import docopt
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.spatial import distance
import sys
import torch

from models.deepsort_utils.fastreid_adaptor import FastReID
from models.apparence_bbox_detector import ApparenceBBoxDetector


np.random.seed(0)


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


def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    return rotated, M

def crop_pad_rotations(frame, bbox, background_color, height, width, axis):

    output = []
    for rot in axis:

        img = np.full((height, width, len(background_color)), background_color).reshape(len(background_color), height, width)
        
        h = int(min(bbox[3], height))
        w = int(min(bbox[2], width))

        rot_frame, M = rotate(frame, rot, (bbox[0] + w/2, bbox[1] + h/2))

        center = np.array((bbox[0] + w/2, bbox[1] + h/2))
        origin = np.dot(bbox[:2] - center, M).astype(int)[:2] + center

        deltas = bbox[2:4] * np.abs(np.cos(np.deg2rad(rot))) + bbox[4:2:-1] * np.abs(np.sin(np.deg2rad(rot)))
        deltas = deltas.astype(int)
        h = int(min(deltas[1], height))
        w = int(min(deltas[0], width))

        crop = rot_frame[int(origin[1]) : int(origin[1]) + h, int(origin[0]) : int(origin[0]) + w, :]
        h = crop.shape[0]
        w = crop.shape[1]

        start_h = int(height // 2 - h // 2)
        start_w = int(width // 2 - w // 2)
        img[:, start_h : start_h + h, start_w : start_w + w] = np.moveaxis(crop, [0, 1, 2], [1, 2, 0])

        output.append(img)
    
    return output


CONFIG_FILE = 'runs/apparence/train01_colonia_256_128/config.yaml'
WEIGHTS_PATH = 'runs/apparence/train01_colonia_256_128/model_best.pth'

DOCTEXT = f""" 
Usage:
  reid_rotation_test.py <input_video> <trckFile> <output_file> [--config=<cf>] [--weights=<wp>] [--num_imgs=<ni>] [--num_steps=<ns>]
  reid_rotation_test.py -h | --help

Options:
  --config=<cf>       Config file from the fastreid model. [default: {CONFIG_FILE}]
  --weights=<wp>      Weights path from the fastreid model. [default: {WEIGHTS_PATH}]
  --num_imgs=<ni>     Number of ants detections used for the experiment [default: 5]
  --num_steps=<ns>    How many divisions of 360º are used when rotating the images [default: 18]

"""

if __name__ == '__main__':

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    input_video = args['<input_video>']
    trckFile = args['<trckFile>']
    output_file = args['<output_file>']

    config_file = args['--config']
    weights_path = args['--weights']

    num_imgs = int(args['--num_imgs'])
    num_steps = int(args['--num_steps'])

    apparence_model = FastReID(config_file, weights_path)
    def apparence_model_applier(x):
        with torch.no_grad():
            out = apparence_model(x)
            out = torch.nn.functional.normalize(out, dim=-1).numpy(force=True)
        
        return out
    
    seq_dets = np.loadtxt(trckFile, delimiter=',')
    subset = np.arange(len(seq_dets))
    np.random.shuffle(subset)
    subset = subset[:num_imgs]
    seq_dets = seq_dets[subset]
    frame_ids = seq_dets[:, 0].copy()

    axis = np.linspace(0, 360, num_steps, endpoint=False)

    seq_dets_iter = iter(seq_dets)
    
    distances = []
    with VideoCapture(input_video) as capture:
        results = []
        # We shold be able to skip loading empty frames
        for frame_id in frame_ids:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_id - 1)

            _, frame = capture.read()
            if frame is None:
                print (f'Frame {frame_id} is None', file=sys.stderr)
                break

            bbox = next(seq_dets_iter)
            background_color = np.mean(frame, (0, 1))
            inputs = torch.Tensor(np.stack(crop_pad_rotations(frame, bbox[2:], background_color, 128, 64, axis), axis=0))

            feats = apparence_model_applier(inputs)

            base = feats[0]
            distances.append([distance.cosine(feat, base) for feat in feats])

    distances = np.array(distances)

    min_dist = distances.min(axis=0)
    max_dist = distances.max(axis=0)
    mean_dist = distances.mean(axis=0)

    rad_axis = np.deg2rad(axis)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.plot(rad_axis, mean_dist)
    ax.plot(rad_axis, min_dist)
    ax.plot(rad_axis, max_dist)
    ax.grid(True)
    ax.legend(['mean', 'min', 'max'])

    fig.savefig(output_file, dpi=300)
