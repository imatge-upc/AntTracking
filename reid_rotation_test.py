
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

WIDTH = 64
HEIGHT = 64 # 128

YLIM_MAX = 1


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

def crop_pad(frame, bbox, pad_color, crop_h, crop_w):

    h = int(bbox[3])
    w = int(bbox[2])

    crop = frame[int(bbox[1]) : int(bbox[1]) + h, int(bbox[0]) : int(bbox[0]) + w, :]
    
    if h > crop_h:
        excess = h - crop_h
        crop = crop[excess // 2 : excess // 2 + crop_h, :, :]
        h = crop_h
    
    if w > crop_w:
        excess = w - crop_w
        crop = crop[:, excess // 2 : excess // 2 + crop_w, :]
        w = crop_w
    
    if h < crop_h or w < crop_w:
        pad_h = (crop_h - h) // 2
        pad_w = (crop_w - w) // 2
        pad = ((pad_h, crop_h - h - pad_h), (pad_w, crop_w - w - pad_w))
        
        crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    
    crop = np.moveaxis(crop, [0, 1, 2], [1, 2, 0])
    return crop

def pad_reshape(frame, bbox, pad_color, crop_h, crop_w):

    h = int(bbox[3])
    w = int(bbox[2])

    crop = frame[int(bbox[1]) : int(bbox[1]) + h, int(bbox[0]) : int(bbox[0]) + w, :]

    ar = crop_h / crop_w
    
    pad_h = int((w * ar - h) // 2)
    pad_w = int((h / ar - w) // 2)
    pad = ((pad_h, int(w * ar - h - pad_h)), (0, 0)) if h < w * ar else ((0, 0), (pad_w, int(h / ar - w - pad_w)))
    
    crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)

    crop = cv.resize(crop, (crop_w, crop_h), interpolation=cv.INTER_AREA)

    crop = np.moveaxis(crop, [0, 1, 2], [1, 2, 0])
    return crop

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
        
        h = int(min(bbox[3], height))
        w = int(min(bbox[2], width))

        center = np.array((bbox[0] + w/2, bbox[1] + h/2))

        rot_frame, M = rotate(frame, rot, center.tolist())
        #origin = np.dot(bbox[:2] - center, M).astype(int)[:2] + center

        deltas = bbox[2:4] * np.abs(np.cos(np.deg2rad(rot))) + bbox[4:2:-1] * np.abs(np.sin(np.deg2rad(rot)))
        deltas = deltas.astype(int)
        h = int(min(deltas[1], height))
        w = int(min(deltas[0], width))

        bbox = np.array([int(center[0] - w / 2), int(center[1] - h / 2), w, h])
        crop = crop_pad(rot_frame, bbox, background_color, height, width)

        output.append(crop)
    
    return output

def pad_reshape_rotations(frame, bbox, background_color, height, width, axis):

    output = []
    for rot in axis:
        
        h = int(min(bbox[3], height))
        w = int(min(bbox[2], width))

        center = np.array((bbox[0] + w/2, bbox[1] + h/2))

        rot_frame, M = rotate(frame, rot, center.tolist())
        #origin = np.dot(bbox[:2] - center, M).astype(int)[:2] + center

        deltas = bbox[2:4] * np.abs(np.cos(np.deg2rad(rot))) + bbox[4:2:-1] * np.abs(np.sin(np.deg2rad(rot)))
        deltas = deltas.astype(int)
        h = int(min(deltas[1], height))
        w = int(min(deltas[0], width))

        bbox = np.array([int(center[0] - w / 2), int(center[1] - h / 2), w, h])
        crop = pad_reshape(rot_frame, bbox, background_color, height, width)

        output.append(crop)
    
    return output

def ant_with_itself(input_video, seq_dets, apparence_model_applier, frame_ids, axis):
    seq_dets_iter = iter(seq_dets)
    
    distances = []
    with VideoCapture(input_video) as capture:
        # We shold be able to skip loading empty frames
        for frame_id in frame_ids:
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_id - 1)

            _, frame = capture.read()
            if frame is None:
                print (f'Frame {frame_id} is None', file=sys.stderr)
                break

            bbox = next(seq_dets_iter)
            background_color = np.mean(frame, (0, 1))
            # inputs = torch.Tensor(np.stack(crop_pad_rotations(frame, bbox[2:], background_color, HEIGHT, WIDTH, axis), axis=0))
            inputs = torch.Tensor(np.stack(pad_reshape_rotations(frame, bbox[2:], background_color, HEIGHT, WIDTH, axis), axis=0))

            feats = apparence_model_applier(inputs)

            base = feats[0]
            distances.append([distance.cosine(feat, base) for feat in feats])

    distances = np.array(distances)

    min_dist = distances.min(axis=0)
    max_dist = distances.max(axis=0)
    mean_dist = distances.mean(axis=0)

    return min_dist, max_dist, mean_dist

def ant_with_another(input_video, seq_dets1, seq_dets2, apparence_model_applier, frame_ids1, frame_ids2, axis):
    seq_dets_iter1 = iter(seq_dets1)
    seq_dets_iter2 = iter(seq_dets2)
    
    distances = []
    with VideoCapture(input_video) as capture:
        # We shold be able to skip loading empty frames
        for frame_id1, frame_id2 in zip(frame_ids1, frame_ids2):

            capture.set(cv.CAP_PROP_POS_FRAMES, frame_id1 - 1)
            _, frame1 = capture.read()
            if frame1 is None:
                print (f'Frame {frame_id1} is None', file=sys.stderr)
                break

            capture.set(cv.CAP_PROP_POS_FRAMES, frame_id2 - 1)
            _, frame2 = capture.read()
            if frame2 is None:
                print (f'Frame {frame_id2} is None', file=sys.stderr)
                break

            bbox1 = next(seq_dets_iter1)
            bbox2 = next(seq_dets_iter2)

            background_color1 = np.mean(frame1, (0, 1))
            background_color2 = np.mean(frame2, (0, 1))

            # base_input = torch.Tensor(crop_pad(frame2, bbox2[2:], background_color2, HEIGHT, WIDTH)[np.newaxis, :])
            # rotation_inputs = torch.Tensor(np.stack(crop_pad_rotations(frame1, bbox1[2:], background_color1, HEIGHT, WIDTH, axis), axis=0))
            base_input = torch.Tensor(pad_reshape(frame2, bbox2[2:], background_color2, HEIGHT, WIDTH)[np.newaxis, :])
            rotation_inputs = torch.Tensor(np.stack(pad_reshape_rotations(frame1, bbox1[2:], background_color1, HEIGHT, WIDTH, axis), axis=0))

            base = apparence_model_applier(base_input)[0]
            rotation_feats = apparence_model_applier(rotation_inputs)

            distances.append([distance.cosine(feat, base) for feat in rotation_feats])

    distances = np.array(distances)

    min_dist = distances.min(axis=0)
    max_dist = distances.max(axis=0)
    mean_dist = distances.mean(axis=0)

    return min_dist, max_dist, mean_dist


CONFIG_FILE = 'runs/apparence/ants_fastreid_vocal-sweep-6_cee5t8ta/config.yaml'
WEIGHTS_PATH = 'runs/apparence/ants_fastreid_vocal-sweep-6_cee5t8ta/model_best.pth'

DOCTEXT = f""" 
Usage:
  reid_rotation_test.py <input_video> <trckFile> <output_file> [--config=<cf>] [--weights=<wp>] [--num_imgs=<ni>] [--num_steps=<ns>] [--max_ids=<mi>]
  reid_rotation_test.py -h | --help

Options:
  --config=<cf>                 Config file from the fastreid model. [default: {CONFIG_FILE}]
  --weights=<wp>                Weights path from the fastreid model. [default: {WEIGHTS_PATH}]
  --num_imgs=<ni>               Number of ants detections used for the rotation experiments [default: 5]
  --num_steps=<ns>              How many divisions of 360º are used when rotating the images [default: 18]
  --max_ids=<mi>                Number of ids used for rotation, it cannot be greater or equal to the total number of ids (there must be available ids for query) [default: 5]

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
    max_ids = int(args['--max_ids'])

    apparence_model = FastReID(config_file, weights_path)
    def apparence_model_applier(x):
        with torch.no_grad():
            out = apparence_model(x)
            out = torch.nn.functional.normalize(out, dim=-1).numpy(force=True)
        
        return out
    
    seq_dets = np.loadtxt(trckFile, delimiter=',')

    all_ids = np.unique(seq_dets[:, 1])
    rot_ids = all_ids[:max_ids]
    query_ids = all_ids[~np.isin(all_ids, rot_ids)]

    seq_dets1 = seq_dets[np.isin(seq_dets[:, 1], rot_ids)]

    subset1 = np.arange(len(seq_dets1))
    np.random.shuffle(subset1)

    subset1 = subset1[:num_imgs]
    seq_dets1 = seq_dets1[subset1]
    frame_ids1 = seq_dets1[:, 0].copy()

    seq_dets2 = seq_dets[np.isin(seq_dets[:, 1], query_ids)]

    subset2 = np.arange(len(seq_dets2))
    np.random.shuffle(subset2)

    subset2 = subset2[:num_imgs]
    seq_dets2 = seq_dets2[subset2]
    frame_ids2 = seq_dets2[:, 0].copy()

    axis = np.linspace(0, 360, num_steps, endpoint=False)

    min_dist, max_dist, mean_dist = ant_with_itself(input_video, seq_dets1, apparence_model_applier, frame_ids1, axis)

    rad_axis = np.deg2rad(axis)
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(6.4 * 2.5, 4.8 * 1.25* 1.2), constrained_layout=True)
    fig.suptitle('Apparence Features Rotation Test\n(The radial axis contains the cosine distance, the angular axis contains the rotation in degrees)')

    ax1.plot(rad_axis, mean_dist)
    ax1.plot(rad_axis, min_dist)
    ax1.plot(rad_axis, max_dist)
    ax1.set_ylim([0, YLIM_MAX])
    ax1.grid(True)
    ax1.legend(['mean', 'min', 'max'])
    ax1.set_title(f'Ant crop compared with itself rotated\n(experiment done for {num_imgs} detections)')

    min_dist, max_dist, mean_dist = ant_with_another(input_video, seq_dets1, seq_dets2, apparence_model_applier, frame_ids1, frame_ids2, axis)

    ax2.plot(rad_axis, mean_dist)
    ax2.plot(rad_axis, min_dist)
    ax2.plot(rad_axis, max_dist)
    ax2.set_ylim([0, YLIM_MAX])
    ax2.grid(True)
    ax2.legend(['mean', 'min', 'max'])
    ax2.set_title(f'Ant crop compared with another ant rotated\n(experiment done for {num_imgs} pairs of different identities)')

    fig.savefig(output_file, dpi=300)
