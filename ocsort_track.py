
import numpy as np
import os
import sys
import time

from docopts.help_ocsort_track import parse_args
from models.ocsort_utils.associator import OCSortAssociator
from models.ocsort_utils.metric_utils import iou_batch, giou_batch, diou_batch, ciou_batch, ct_dist_batch
from models.ocsort_utils.ocsort_kalman_estimator import OCSortKalmanEstimator
from models.ocsort_utils.track_manager import TrackManager
from models.sort import Sort

np.random.seed(0)


ASSO_FUNCS = {  "iou": iou_batch,
                "giou": giou_batch,
                "ciou": ciou_batch,
                "diou": diou_batch,
                "ct_dist": ct_dist_batch}


class PrecomputedMOTDetector():

    def __init__(self, seq_path=None, min_score=0.1, first_frame=1, verbose=False):

        self.seq_dets = np.loadtxt(seq_path, delimiter=',')
        self.min_score = min_score
        self.first_frame = first_frame
        
        self.last_frame = int(self.seq_dets[:, 0].max())

        self.current_frame = first_frame

        self.verbose = verbose
    
    def reset(self):
        self.current_frame = self.first_frame
    
    def __call__(self, frame):

        if self.verbose and ((frame - 1) % 500 == 0):
            print (f'Processing frame {frame - 1}', file=sys.stderr)

        dets = self.seq_dets[self.seq_dets[:, 0] == self.current_frame, 2:7]
        dets[:, 2:4] += dets[:, 0:2] # convert to [x1,y1,w,h] to [x1,y1,x2,y2]

        dets = dets[dets[:, -1] >= self.min_score] # filter out low score detections

        self.current_frame += 1

        return dets.reshape(-1, 5) # np.array([[x1, y1, x2, y2, score], ...]).reshape(N, 5)


if __name__ == '__main__':

    detection_file, tracking_file,\
    _, _,\
    track_thresh, iou_thresh,\
    min_hits, use_byte,\
    assoc_func = parse_args(sys.argv)

    total_time = 0.0
    total_frames = 0

    if not os.path.exists('OUTPUT'):
        os.makedirs('OUTPUT')
    
    detector = PrecomputedMOTDetector(detection_file, verbose=True)
    associator = OCSortAssociator(
        det_threshold=track_thresh,
        iou_threshold=iou_thresh,
        second_iou_threshold=iou_thresh,
        second_asso_func=ASSO_FUNCS[assoc_func],
        inertia_weight=0.2,
        use_byte=use_byte
    )
    estiamtor_cls = lambda bbox : OCSortKalmanEstimator(bbox, delta_t=3)
    track_manager = TrackManager(estiamtor_cls, None, None, max_age=30, min_hits=min_hits, max_last_update=1, det_threshold=track_thresh)

    ocsort_model = Sort(detector, associator, track_manager)

    print(f'Processing {detection_file}')

    # As we do not want to load the video, this trick is enough
    results = []
    for frame in range(1, detector.last_frame + 1):
        total_frames += 1

        start_time = time.time()
        online_targets = ocsort_model(frame)
        cycle_time = time.time() - start_time

        total_time += cycle_time

        for t in online_targets:
            tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
            tid = int(t[4])
            conf = t[5]
            results.append(f"{frame},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{conf:.2f},-1,-1,-1\n")

    with open(os.path.join('OUTPUT', tracking_file), 'w') as f:
        f.writelines(results)

    print(f"Total Tracking took: {total_time:.3f} seconds for {total_frames:d} frames or {total_frames / total_time:.1f} FPS")
