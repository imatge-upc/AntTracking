
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

    def __init__(self, seq_path=None, first_frame=1, verbose=False, min_frames=6, iou_th=0.3, min_area=2500, sampling_rate=5):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')

        self.first_frame = first_frame
        self.current_frame = first_frame

        self.last_frame = int(self.seq_dets[:, 0].max())

        sampling_mask = np.in1d(self.seq_dets[:, 0], np.arange(1, self.last_frame, sampling_rate, dtype=int))
        self.seq_dets = self.seq_dets[sampling_mask, :]

        # Considering we use ground truth tracks, so all the identities are well annotated. We discard bboxes that may have more than 1 identity.
        for fr in range(1, int(self.seq_dets[:, 0].max())):

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
    
    def __call__(self, frame):

        if self.verbose and (self.current_frame % 500 == 0):
            print (f'Processing frame {self.current_frame}', file=sys.stderr)

        tcks = self.seq_dets[self.seq_dets[:, 0] == self.current_frame, :]

        self.current_frame += 1

        return tcks

def crop_pad(bbox, imgsz):

    h = bbox[3]
    w = bbox[2]
    
    if h > imgsz:
        excess = h - imgsz
        crop = crop[excess // 2 : excess // 2 + imgsz, :, :]
        h = imgsz
    
    if w > imgsz:
        excess = w - imgsz
        crop = crop[:, excess // 2 : excess // 2 + imgsz, :]
        w = imgsz
    
    if h < imgsz or w < imgsz:
        pad_h = (imgsz - h) // 2
        pad_w = (imgsz - w) // 2
        pad = ((pad_h, imgsz - h - pad_h), (pad_w, imgsz - w - pad_w))
        pad_color = np.median(crop, axis=(0, 1))
        
        crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    
    return crop

def pad_reshape(bbox, imgsz):

    h = bbox[3]
    w = bbox[2]
    
    pad = (max(h, w) - min(h, w)) // 2
    pad = ((pad, max(h, w) - h - pad), (0, 0)) if h < w else ((0, 0), (pad, max(h, w) - w - pad))
    pad_color = np.median(crop, axis=(0, 1))
    
    crop = np.stack([np.pad(crop[:, :, c], pad, mode='constant', constant_values=pad_color[c]) for c in range(3)], axis=2)
    
    crop = cv.resize(crop, (imgsz, imgsz), interpolation=cv.INTER_AREA)
    
    return crop

def process_video(seen_ids, video_path, seq_path, sampling_rate, test_frac, query_frac, query_prob, reshape, do_pad_reshape, imgsz, train_dir, query_dir, test_dir, verbose=True):
    min_frames = 3
    tracker = PrecomputedMOTTracker(seq_path, verbose=verbose, min_frames=min_frames * 2, sampling_rate=sampling_rate)
    ids = np.unique(tracker.seq_dets[:, 1].astype(int))
    new_id = {id_ : id_ if id_ not in seen_ids else max(seen_ids) + 1 for id_ in ids}
    seen_ids.update(set(new_id.values()))
    
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

                if reshape:
                    crop = cv.resize(crop, (imgsz, imgsz), interpolation=cv.INTER_AREA)
                elif do_pad_reshape:
                    crop = pad_reshape(bbox, imgsz)
                else:
                    crop = crop_pad(bbox, imgsz)
                
                if crop.shape[0] != imgsz or crop.shape[1] != imgsz:
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
  video_to_apparences.py (<video_path> <seq_path>)... [--test_frac=<tf>] [--query_frac=<qf>] [--query_prob=<qp>] [--imgsz=<is>] [--sampling_rate=<sr>] [--reshape | --pad_reshape]

Options:
  --test_frac=<tf>          The fraction of identities used for testing. [default: 0.5]
  --query_frac=<qf>         The fraction of test identities used for the query set. [default: 0.8]
  --query_prob=<qp>         The probability of putting images into the query set instead of the test set after both sets have 3 images. [default: 0.1]
  --imgsz=<is>              Image size [default: 64]
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
    imgsz = int(args['--imgsz'])
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
        print(f'VIDEO {i} OF {len(video_pathes)}')
        process_video(seen_ids, video_path, seq_path, sampling_rate, test_frac, query_frac, query_prob, reshape, do_pad_reshape, imgsz, train_dir, query_dir, test_dir, verbose=True)

    shutil.make_archive(output_file, 'zip', output_file)
    shutil.rmtree(output_file)
