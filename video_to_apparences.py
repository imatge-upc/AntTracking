
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

    def __init__(self, seq_path=None, first_frame=1, verbose=False, min_frames=6, iou_th=0.3, min_area=2500):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')

        # Considering we use ground truth tracks, so all the identities are well annotated. We discard bboxes that may have more than 1 identity.
        for fr in range(1, int(self.seq_dets[:, 0].max())):

            if verbose and (fr % 500 == 0):
                print (f'\tPreprocessing frame {fr}', file=sys.stderr)

            tcks = self.seq_dets[self.seq_dets[:, 0] == fr, :]

            tcks2 = tcks[:, 2:7].copy()
            area = tcks2[:, 2] * tcks2[:, 3]
            tcks2[:, 2:4] += tcks2[:, 0:2]
            
            iou_matrix = iou_batch(tcks2, tcks2)
            
            valid_mask = iou_matrix < iou_th
            valid_mask[np.eye(*valid_mask.shape, dtype=bool)] = True
            valid_mask = np.all(valid_mask, axis=0) & (area > min_area)
            
            self.seq_dets = np.vstack([self.seq_dets[self.seq_dets[:, 0] != fr, :], tcks[valid_mask, :]])

        ids = np.unique(self.seq_dets[:, 1])
        for id_ in ids:
            if sum(self.seq_dets[:, 1] == id_) < min_frames:
                self.seq_dets = self.seq_dets[self.seq_dets[:, 1] != id_, :]

        self.last_frame = int(self.seq_dets[:, 0].max())

        self.first_frame = first_frame
        self.current_frame = first_frame


        self.verbose = verbose
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def __call__(self, frame):

        if self.verbose and (self.current_frame % 500 == 0):
            print (f'Processing frame {self.current_frame}', file=sys.stderr)

        tcks = self.seq_dets[self.seq_dets[:, 0] == self.current_frame, :]

        self.current_frame += 1

        return tcks
    
    
DOCTEXT = f"""
Usage:
  video_to_apparences.py <video_path> <seq_path> [--test_frac=<tf>] [--query_frac=<qf>] [--query_prob=<qp>]

Options:
  --test_frac=<tf>      The fraction of identities used for testing. [default: 0.5]
  --query_frac=<qf>     The fraction of test identities used for the query set. [default: 0.8]
  --query_prob=<qp>     The probability of putting images into the query set instead of the test set after both sets have 3 images. [default: 0.1]
"""

if __name__ == "__main__":

    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)
    video_path = args['<video_path>']
    seq_path = args['<seq_path>']
    test_frac = float(args['--test_frac'])
    query_frac = float(args['--query_frac'])
    query_prob = float(args['--query_prob'])

    output_file = "Market-1501-v15.09.15"
    train_dir = os.path.join(output_file, output_file, 'bounding_box_train')
    test_dir = os.path.join(output_file, output_file, 'bounding_box_test')
    query_dir = os.path.join(output_file, output_file, 'query')

    min_frames = 3
    tracker = PrecomputedMOTTracker(seq_path, verbose=True, min_frames=min_frames * 2)
    ids = np.unique(tracker.seq_dets[:, 1])
    
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

            if len(queries) > len(frames) // 2: #- min_frames:
                np.random.shuffle(frames)
                queries = frames[ : len(frames) // 2]
            
            query_id_frames.append(queries)

        else:
            np.random.shuffle(frames)
            query_id_frames.append(frames[:min_frames])

    os.makedirs(output_file, exist_ok=False)
    os.makedirs(train_dir, exist_ok=False)
    os.makedirs(test_dir, exist_ok=False)
    os.makedirs(query_dir, exist_ok=False)

    with VideoCapture(video_path) as capture:
        for fr in range(1, tracker.last_frame):

            tracks = tracker(fr)
            if len(tracks) == 0:
                continue

            capture.set(cv.CAP_PROP_POS_FRAMES, fr - 1)
            _, frame = capture.read()
            if frame is None:
                print (f'Frame {fr} is None')
                break

            for tck in tracks:
                bbox = tck[2:6].astype(int)
                id_ = tck[1].astype(int)

                crop = frame[bbox[1] : bbox[1] + bbox[3], bbox[0] : bbox[0] + bbox[2], :].copy()

                h = bbox[3]
                w = bbox[2]
                if h / w < 1:
                    crop = np.swapaxes(crop, 0, 1)
                    h = bbox[2]
                    w = bbox[3]
                
                if h > 128:
                    excess = h - 128
                    crop = crop[excess // 2 : excess // 2 + 128, :, :]
                    h = 128
                
                if w > 64:
                    excess = w - 64
                    crop = crop[:, excess // 2 : excess // 2 + 64, :]
                    w = 64
                
                if h < 128 or w < 64:
                    pad_h = (128 - h) // 2
                    pad_w = (64 - w) // 2
                    pad = ((pad_h, 128 - h - pad_h), (pad_w, 64 - w - pad_w))
                    pad_color = np.median(crop, axis=(0, 1))
                    
                    crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)

                if id_ in train_ids: # train set
                    cid = np.random.randint(1, 3) # camara id 1 or 2
                    filename = f'{id_:04}_c{cid}s1_{fr:06}_01.jpg'
                    cv.imwrite(os.path.join(train_dir, filename), crop)
                    
                else: # test set or query set
                    if (id_ in query_ids) and (fr in query_id_frames[int(np.where(query_ids == id_)[0])]): # query set
                        cid = np.random.randint(3, 5) # camara id 3 or 4
                        filename = f'{id_:04}_c{cid}s1_{fr:06}_01.jpg'
                        cv.imwrite(os.path.join(query_dir, filename), crop)

                    else: # test set; TODO: add background crops (frame id: 0)
                        cid = np.random.randint(5, 7) # camara id 5 or 6
                        filename = f'{id_:04}_c{cid}s1_{fr:06}_01.jpg'
                        cv.imwrite(os.path.join(test_dir, filename), crop)

    shutil.make_archive(output_file, 'zip', output_file)
    shutil.rmtree(output_file)
