
import numpy as np
import os
import sys
import time

from docopts.help_sort_inference import parse_args
from models.sort import Sort
from models.sort_utils.iou_associator import IoUAssociator
from models.sort_utils.reduced_bbox_kalman_estimator import ReducedBBoxKalmanEstimator
from models.sort_utils.track_manager import TrackManager


np.random.seed(0)


class PrecomputedMOTDetector():

    def __init__(self, seq_path=None, first_frame=1):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')
        self.first_frame = first_frame
        
        self.last_frame = int(self.seq_dets[:, 0].max())

        self.current_frame = first_frame
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def __call__(self, frame):

        dets = self.seq_dets[self.seq_dets[:, 0] == self.current_frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] #convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        self.current_frame += 1

        return dets.reshape(-1, 5) # np.array([[x1, y1, x2, y2, score], ...]).reshape(N, 5)


if __name__ == '__main__':

    seq_path, max_age, min_hits, iou_threshold = parse_args(sys.argv)

    total_time = 0.0
    total_frames = 0

    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    
    detector = PrecomputedMOTDetector(seq_path)
    track_manager = TrackManager(ReducedBBoxKalmanEstimator, max_age, min_hits)
    associator = IoUAssociator(iou_threshold)

    sort_model = Sort(detector, associator, track_manager)

    seq = os.path.basename(seq_path)
    with open(os.path.join('OUTPUT', f'{seq}'), 'w') as out_file:
        print(f'Processing {seq_path}')

        # As we do not want to load the video, this trick is enough
        for frame in range(1, detector.last_frame + 1):
            total_frames += 1

            start_time = time.time()
            trackers = sort_model(frame)
            cycle_time = time.time() - start_time

            total_time += cycle_time

            for d in trackers:
                print(f'{frame:d},{int(d[4]):d},{d[0]:.2f},{d[1]:.2f},{d[2] - d[0]:.2f},{d[3] - d[1]:.2f},1,-1,-1,-1', file=out_file)
    
    print(f"Total Tracking took: {total_time:.3f} seconds for {total_frames:d} frames or {total_frames / total_time:.1f} FPS")
