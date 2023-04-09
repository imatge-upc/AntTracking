
import numpy as np
import os
import sys
import time

from docopts.help_deepsort_track import parse_args
from models.sort import Sort
from models.deepsort_utils.associator import DeepSortAssociator
from models.deepsort_utils.deepsort_kalman_estimator import DeepSORTBBoxKalmanEstimator
from models.deepsort_utils.nn_apparence_scorer import NNApparenceScorer
from models.deepsort_utils.track_manager import TrackManager


np.random.seed(0)


class PrecomputedMOTDetector():

    def __init__(self, seq_path=None, first_frame=1, verbose=False):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')

        self.first_frame = first_frame
        self.last_frame = int(self.seq_dets[:, 0].max())
        self.current_frame = first_frame

        self.mask = np.full((self.seq_dets.shape[1]), True)
        self.mask[:2] = False
        self.mask[7:10] = False

        self.verbose = verbose
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def __call__(self, frame):

        if self.verbose and ((frame - 1) % 500 == 0):
            print (f'Processing frame {frame - 1}', file=sys.stderr)

        dets = self.seq_dets[self.seq_dets[:, 0] == self.current_frame, :]
        dets = dets[:, self.mask]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        self.current_frame += 1

        return dets


if __name__ == '__main__':

    detection_file, tracking_file, max_age, min_hits, iou_threshold = parse_args(sys.argv)

    total_time = 0.0
    total_frames = 0

    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    
    detector = PrecomputedMOTDetector(detection_file, verbose=True)
    apparence_scorer_cls = lambda det : NNApparenceScorer(det, 100)
    track_manager = TrackManager(DeepSORTBBoxKalmanEstimator, apparence_scorer_cls, None, max_age, min_hits)
    associator = DeepSortAssociator(num_feats=50, average_factor=0, gate_value=1e+5, th_maha=None, th_appa=1, iou_threshold=iou_threshold)

    model = Sort(detector, associator, track_manager)
    print(f'Processing {detection_file}')

    # As we do not want to load the video, this trick is enough
    results = []
    for frame in range(1, detector.last_frame + 1):
        total_frames += 1

        start_time = time.time()
        online_targets = model(frame)
        cycle_time = time.time() - start_time

        total_time += cycle_time

        for t in online_targets:
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            tid = int(t[4])
            results.append(f"{frame},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},1.0,-1,-1,-1\n")

    with open(os.path.join('OUTPUT', tracking_file), 'w') as f:
        f.writelines(results)

    print(f"Total Tracking took: {total_time:.3f} seconds for {total_frames:d} frames or {total_frames / total_time:.1f} FPS")
