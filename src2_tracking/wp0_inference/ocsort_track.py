
import numpy as np
import os
import sys
import time

from ceab_ants.bbox_metrics.bbox_metrics import iou_bbox_batch, giou_bbox_batch, diou_bbox_batch, ciou_bbox_batch, ct_dist_bbox_batch
from ceab_ants.io.mot_loader import PrecomputedMOTDetector
from ceab_ants.tracking.box_trackers.bbox_ocsort import build_bbox_ocsort

from docopts.help_ocsort_track import parse_args


REDUCED = True
np.random.seed(0)

ASSO_FUNCS = {
    "iou": iou_bbox_batch,
    "giou": giou_bbox_batch,
    "ciou": ciou_bbox_batch,
    "diou": diou_bbox_batch,
    "ct_dist": ct_dist_bbox_batch
}


def main(
        detection_file, 
        tracking_file, 
        assoc_func="ciou", 
        th_conf=0.5, 
        th_first_score=0.3, 
        th_second_score=0.3, 
        inertia_weight=0.2, 
        use_byte=False, 
        delta_t=3, 
        max_age=30, 
        min_hits=3, 
        max_last_update=1, 
        reduced=True, 
        verbose=True
):

    total_time = 0.0
    total_frames = 0

    try:
        os.makedirs(os.path.dirname(tracking_file), exist_ok=True)
    except FileNotFoundError:
        pass
    
    detector = PrecomputedMOTDetector(detection_file, verbose=verbose)
    
    ocsort_model = build_bbox_ocsort(
        detector=detector,
        second_score_function=ASSO_FUNCS[assoc_func],
        th_conf=th_conf,
        th_first_score=th_first_score,
        th_second_score=th_second_score,
        inertia_weight=inertia_weight,
        use_byte=use_byte,
        delta_t=delta_t,
        max_age=max_age,
        min_hits=min_hits,
        max_last_update=max_last_update,
        reduced=reduced
    )

    print(f'Processing {detection_file}')

    # For large files it is better to keep it open
    with open(tracking_file, 'w') as out_file: # open can use buffering= to set the buffer size

        # As we do not want to load the video, this trick is enough
        for frame in range(1, detector.last_frame + 1):
            total_frames += 1

            start_time = time.time()
            online_targets = ocsort_model(frame)
            cycle_time = time.time() - start_time

            total_time += cycle_time
            
            if online_targets is not None:
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = int(t[5])
                    conf = t[4]
                    tracks = f"{frame},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{conf:.2f},-1,-1,-1"
                    print(tracks, end='\n', file=out_file)

    print(f"Total Tracking took: {total_time:.3f} seconds for {total_frames:d} frames or {total_frames / total_time:.1f} FPS")

if __name__ == '__main__':

    detection_file, tracking_file,\
    _, _,\
    track_thresh, iou_thresh,\
    min_hits, use_byte,\
    assoc_func = parse_args(sys.argv)

    main(
        detection_file, 
        tracking_file, 
        assoc_func, 
        th_conf=track_thresh, 
        th_first_score=iou_thresh, 
        th_second_score=iou_thresh, 
        inertia_weight=0.2, 
        use_byte=use_byte, 
        delta_t=3, 
        max_age=30, 
        min_hits=min_hits, 
        max_last_update=1, 
        reduced=REDUCED, 
        verbose=True
    )
