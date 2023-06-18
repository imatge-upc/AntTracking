
from contextlib import contextmanager
import cv2 as cv
import numpy as np
import os
import sys
import time
from torchvision.models import vit_b_32 as VisionTransformer

from docopts.help_apparence_tracks import parse_args
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

        self.verbose = verbose
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def __call__(self, frame):

        if self.verbose and (self.current_frame % 500 == 0):
            print (f'Processing frame {self.current_frame}', file=sys.stderr)

        tcks = self.seq_dets[self.seq_dets[:, 0] == self.current_frame, :]
        self.current_frame += 1

        return tcks


if __name__ == '__main__':

    input_video, detection_file, tracking_file = parse_args(sys.argv)

    total_time = 0.0
    total_frames = 0

    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    
    detector_model = PrecomputedMOTTracker(detection_file, verbose=True)

    apparence_model = FeatureExtractor(VisionTransformer(weights='DEFAULT'), ['encoder']) # Output list of 1 Tensor [#bboxes, 50, 768]
    apparence_model.eval()
    apparence_model_applier = lambda x : apparence_model(x)[0].mean(2).numpy(force=True) # Output [#bboxes, 50]

    model = ApparenceBBoxDetector(detector_model, apparence_model_applier, skip=2)

    print(f'Processing {detection_file}')

    # Apply the model
    with VideoCapture(input_video) as capture:
        # As we do not want to load the video, this trick is enough
        results = []
        for frame_id in range(1, detector_model.last_frame + 1):
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
                results.append(f"{t[0]}, {t[1]}, {', '.join([str(b) for b in t[2:]])[:-2]}\n")

    with open(os.path.join('OUTPUT', tracking_file), 'w') as f:
        f.writelines(results)

    print(f"Total Tracking took: {total_time:.3f} seconds for {total_frames:d} frames or {total_frames / total_time:.1f} FPS")
