
from docopt import docopt
from contextlib import contextmanager
import cv2 as cv
import numpy as np
import os
import shutil
import sys

from models.ocsort_utils.metric_utils import iou_batch


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

    def __init__(self, seq_path=None, first_frame=1, verbose=False, min_frames=6, iou_th=0.7, min_area=1500, sampling_rate=5):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')

        self.first_frame = first_frame
        self.current_frame = first_frame

        self.last_frame = int(self.seq_dets[:, 0].max())

        # Considering we use ground truth tracks, so all the identities are well annotated. We discard bboxes that may have more than 1 identity.
        for fr in range(1, self.last_frame):

            if verbose and (fr % 500 == 0):
                print (f'\tPreprocessing frame {fr}', file=sys.stderr)

            tcks = self.seq_dets[self.seq_dets[:, 0] == fr, :]

            tcks2 = tcks[:, 2:7].copy()
            area = tcks2[:, 2] * tcks2[:, 3]
            tcks2[:, 2:4] += tcks2[:, 0:2]
            
            iou_matrix = iou_batch(tcks2, tcks2)
            
            valid_mask = iou_matrix < iou_th # When the dataset was made from the original video where, I skiped the meeting of ants
            valid_mask[np.eye(*valid_mask.shape, dtype=bool)] = True
            valid_mask = np.all(valid_mask, axis=0) & (area > min_area)
            
            self.seq_dets = np.vstack([self.seq_dets[self.seq_dets[:, 0] != fr, :], tcks[valid_mask, :]])

        ids = np.unique(self.seq_dets[:, 1])
        for id_ in ids:
            if sum(self.seq_dets[:, 1] == id_) < min_frames:
                self.seq_dets = self.seq_dets[self.seq_dets[:, 1] != id_, :]

        self.verbose = verbose
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def __call__(self, frame, aux=False):

        if self.verbose and not aux and (frame % 500 == 0):
            print (f'Processing frame {frame}', file=sys.stderr)

        tcks = self.seq_dets[self.seq_dets[:, 0] == frame, :]

        self.current_frame += 1

        return tcks

def crop_pad(crop, bbox, crop_w, crop_h):

    h = bbox[3]
    w = bbox[2]
    
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
        pad_color = np.median(crop, axis=(0, 1))
        
        crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    
    return crop

def pad_reshape(crop, bbox, crop_w, crop_h):

    h = bbox[3]
    w = bbox[2]

    ar = crop_h / crop_w
    
    pad_h = (w * ar - h) // 2
    pad_w = (h / ar - w) // 2
    pad = ((pad_h, w * ar - h - pad_h), (0, 0)) if h < w * ar else ((0, 0), (pad_w, h / ar - w - pad_w))
    pad_color = np.median(crop, axis=(0, 1))
    
    crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    
    crop = cv.resize(crop, (crop_w, crop_h), interpolation=cv.INTER_AREA)
    
    return crop

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale)
    rotated = cv.warpAffine(image, M, (w, h))

    return rotated, M

def crop_pca_rotate_crop(gray_frame, frame, bbox, post_bbox, background_th, min_size=20):
    gray_crop = gray_frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2]]
    pts = np.argwhere(gray_crop < background_th).reshape(-1, 2).astype(np.float32)

    if len(pts) < min_size : return 0

    mean = np.empty((0))
    _, eigenvectors, _ = cv.PCACompute2(pts, mean)

    pca_angle_ori = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

    cntr = bbox[:2] + bbox[2:4] / 2
    post_cntr = post_bbox[:2] + post_bbox[2:4] / 2
    module = np.linalg.norm(post_cntr - cntr)
    angle = np.pi / 2 - np.arccos((post_cntr - cntr)[0] / module) * (1. if (post_cntr - cntr)[1] >= 0 else -1.)

    angle_dist = min(np.abs(angle - pca_angle_ori), 2 * np.pi - np.abs(angle - pca_angle_ori))
    pca_angle = pca_angle_ori if (2 * angle_dist) < np.pi else pca_angle_ori - np.pi

    rot_frame, M = rotate(frame, np.rad2deg(pca_angle), (int(cntr[1]), int(cntr[0])))

    origin = np.dot(bbox[:2] - cntr, M).astype(int)[:2] + cntr
    deltas = bbox[2:4] * np.abs(np.cos(pca_angle)) + bbox[4:2:-1] * np.abs(np.sin(pca_angle))
    w, h = deltas.astype(int)

    crop = rot_frame[int(origin[1]) : int(origin[1]) + h, int(origin[0]) : int(origin[0]) + w, :]

    return crop

def process_video(seen_ids, video_path, seq_path, sampling_rate, test_frac, query_frac, query_prob, reshape, do_pad_reshape, crop_w, crop_h, train_dir, query_dir, test_dir, verbose=True):
    min_frames = 3
    tracker = PrecomputedMOTTracker(seq_path, verbose=verbose, min_frames=min_frames * 2, sampling_rate=sampling_rate)
    ids = np.unique(tracker.seq_dets[:, 1].astype(int))
    new_id = {id_ : id_ if id_ not in seen_ids else max(seen_ids) + 1 for id_ in ids}
    seen_ids.update(set(new_id.values()))

    if verbose:
        print("\tNEW IDs")
        print(new_id)
        print("\tSEEN IDS")
        print(seen_ids)
    
    np.random.shuffle(ids)
    train_ids = ids[int(len(ids) * test_frac):]
    test_ids = ids[:int(len(ids) * test_frac)]

    np.random.shuffle(test_ids)
    query_ids = test_ids[:int(len(test_ids) * query_frac)]

    query_id_frames = []
    for id_ in query_ids:
        frames = tracker.seq_dets[tracker.seq_dets[:, 1] == id_, 0].copy()
        idxs = np.random.random(frames.size) < query_prob

        if sum(idxs) >= min_frames:
            queries = frames[idxs]

            if len(queries) > len(frames) // 2:
                #np.random.shuffle(frames) # Do not shuffle because frames too near in time are more or less equal
                queries = frames[ : len(frames) // 2]
            
            query_id_frames.append(queries)

        else:
            #np.random.shuffle(frames) # Do not shuffle because frames too near in time are more or less equal
            query_id_frames.append(frames[:min_frames])

    wrong = 0
    skipped = 0
    with VideoCapture(video_path) as capture:
        for fr in range(1, tracker.last_frame - 1):

            tracks = tracker(fr)
            post_tracks = tracker(fr + 1, aux=True)
            if len(tracks) == 0:
                continue

            capture.set(cv.CAP_PROP_POS_FRAMES, fr - 1)
            _, frame = capture.read()
            if frame is None:
                print (f'Frame {fr} is None')
                break

            gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            background_th = np.mean(gray_frame) * 0.5

            for tck in tracks:
                bbox = tck[2:6].astype(int)
                id_ = tck[1].astype(int)

                if tck[1] not in post_tracks[:, 1]:
                    skipped += 1
                    if skipped % 10 == 0:
                        print(f'{skipped} skipped')
                    continue

                post_bbox = post_tracks[post_tracks[:, 1] == tck[1], 2:6].squeeze()
                crop = crop_pca_rotate_crop(gray_frame, frame, bbox, post_bbox, background_th, min_size=20)

                if reshape:
                    crop = cv.resize(crop, (crop_w, crop_h), interpolation=cv.INTER_AREA)
                elif do_pad_reshape:
                    crop = pad_reshape(crop, bbox, crop_w, crop_h)
                else:
                    crop = crop_pad(crop, bbox, crop_w, crop_h)
                
                if crop.shape[0] != crop_h or crop.shape[1] != crop_w:
                    continue

                if id_ in train_ids: # train set
                    cid = np.random.randint(1, 3) # camara id 1 or 2
                    filename = f'{new_id[id_]:04}_c{cid}s1_{fr:06}_01.png'
                    try:
                        cv.imwrite(os.path.join(train_dir, filename), crop)
                        wrong = 0
                    except Exception as e:
                        wrong += 1
                        if wrong > 10:
                            print("10 consecutive wrong")
                            raise e
                    
                else: # test set or query set
                    if (id_ in query_ids) and (fr in query_id_frames[int(np.where(query_ids == id_)[0])]): # query set
                        cid = np.random.randint(3, 5) # camara id 3 or 4
                        filename = f'{new_id[id_]:04}_c{cid}s1_{fr:06}_01.png'
                        try:
                            cv.imwrite(os.path.join(query_dir, filename), crop)
                            wrong = 0
                        except Exception as e:
                            wrong += 1
                            if wrong > 10:
                                print("10 consecutive wrong")
                                raise e

                    else: # test set; TODO: add background crops (frame id: 0)
                        cid = np.random.randint(5, 7) # camara id 5 or 6
                        filename = f'{new_id[id_]:04}_c{cid}s1_{fr:06}_01.png'
                        try:
                            cv.imwrite(os.path.join(test_dir, filename), crop)
                            wrong = 0
                        except Exception as e:
                            wrong += 1
                            if wrong > 10:
                                print("10 consecutive wrong")
                                raise e
                            
    return seen_ids

    
DOCTEXT = f"""
Usage:
  video_to_oriented_apparences.py (<video_path> <seq_path>)... [--test_frac=<tf>] [--query_frac=<qf>] [--query_prob=<qp>] [--crop_h=<ch>] [--crop_w=<cw>] [--sampling_rate=<sr>] [--reshape | --pad_reshape]

Options:
  --test_frac=<tf>          The fraction of identities used for testing. [default: 0.5]
  --query_frac=<qf>         The fraction of test identities used for the query set. [default: 0.8]
  --query_prob=<qp>         The probability of putting images into the query set instead of the test set after both sets have 3 images. [default: 0.1]
  --crop_h=<ch>             Identity crop height size [default: 64]
  --crop_w=<cw>             Identity crop width size [default: 32]
  --sampling_rate=<sr>      Sampling rate [default: 5]
  --reshape                 Reshape into size instead of crop and pad.
  --pad_reshape             Pad small size until he big size and then reshape into size in stead of crop and pad.
"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    video_pathes = args['<video_path>']
    seq_pathes = args['<seq_path>']

    test_frac = float(args['--test_frac'])
    query_frac = float(args['--query_frac'])
    query_prob = float(args['--query_prob'])
    crop_h = int(args['--crop_h'])
    crop_w = int(args['--crop_w'])
    sampling_rate = int(args['--sampling_rate'])
    reshape = args['--reshape']
    do_pad_reshape = args['--pad_reshape']

    output_file = "Market-1501-v15.09.15"
    train_dir = os.path.join(output_file, output_file, 'bounding_box_train')
    test_dir = os.path.join(output_file, output_file, 'bounding_box_test')
    query_dir = os.path.join(output_file, output_file, 'query')

    os.makedirs(output_file, exist_ok=False)
    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(test_dir, exist_ok=False)
    os.makedirs(query_dir, exist_ok=False)

    seen_ids = set()
    for i, (video_path, seq_path) in enumerate(zip(video_pathes, seq_pathes)):
        print(f'VIDEO {i + 1} OF {len(video_pathes)}')
        seen_ids = process_video(seen_ids, video_path, seq_path, sampling_rate, test_frac, query_frac, query_prob, reshape, do_pad_reshape, crop_w, crop_h, train_dir, query_dir, test_dir, verbose=True)

    shutil.make_archive(output_file, 'zip', output_file)
    shutil.rmtree(output_file)
