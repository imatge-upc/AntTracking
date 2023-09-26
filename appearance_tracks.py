
from contextlib import contextmanager
import cv2 as cv
import numpy as np
import os
import sys
import time
import torch
from torchvision.models import vit_b_32 as VisionTransformer

from docopts.help_appearance_tracks import parse_args
from models.deepsort_utils.fastreid_adaptor import FastReID
from models.deepsort_utils.feature_extractor import FeatureExtractor
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


class PrecomputedMOTTracker():

    def __init__(self, seq_path=None, first_frame=1, verbose=False):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')
        self.last_frame = int(self.seq_dets[:, 0].max())

        self.first_frame = first_frame
        self.current_frame = first_frame
        self.seen_frames = 0

        self.verbose = verbose
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def set_current_frame(self, current_frame):
        self.current_frame = current_frame
    
    def __call__(self, frame):

        self.seen_frames += 1
        if self.verbose and (self.seen_frames % 250 == 0):
            print (f'Processing frame {self.current_frame}', file=sys.stderr)

        tcks = self.seq_dets[self.seq_dets[:, 0] == self.current_frame, :]
        self.current_frame += 1

        return tcks


if __name__ == '__main__':

    input_video, detection_file, tracking_file, config_file, weights_path = parse_args(sys.argv)

    total_time = 0.0
    total_frames = 0

    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    
    detector_model = PrecomputedMOTTracker(detection_file, verbose=True)

    #apparence_model = FeatureExtractor(VisionTransformer(weights='DEFAULT'), ['encoder']) # Output list of 1 Tensor [#bboxes, 50, 768]
    #apparence_model.eval()
    #apparence_model_applier = lambda x : apparence_model(x)[0].mean(2).numpy(force=True) # Output [#bboxes, 50]
    apparence_model = FastReID(config_file, weights_path)
    def apparence_model_applier(x):
        with torch.no_grad():
            out = apparence_model(x)
            out = torch.nn.functional.normalize(out, dim=-1).numpy(force=True)
        
        return out

    model = ApparenceBBoxDetector(detector_model, apparence_model_applier, skip=2, height=128, width=64)

    print(f'Processing {detection_file}')

    # Apply the model
    with VideoCapture(input_video) as capture:
        results = []
        # We shold be able to skip loading empty frames
        for frame_id in np.sort(np.unique(detector_model.seq_dets[:, 0])):
            capture.set(cv.CAP_PROP_POS_FRAMES, frame_id - 1)
            detector_model.set_current_frame(frame_id)
            total_frames += 1

            _, frame = capture.read()
            if frame is None:
                print (f'Frame {frame_id} is None', file=sys.stderr)
                break

            start_time = time.time()
            online_targets = model(frame)
            cycle_time = time.time() - start_time

            total_time += cycle_time

            for t in online_targets:
                results.append(f"{t[0]}, {t[1]}, {', '.join([str(b) for b in t[2:]])}\n")

    with open(os.path.join('OUTPUT', tracking_file), 'w') as f:
        f.writelines(results)

    print(f"Total Tracking took: {total_time:.3f} seconds for {total_frames:d} frames or {total_frames / total_time:.1f} FPS")
