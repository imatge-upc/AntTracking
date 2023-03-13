
import lap
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )
        
def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )


def iou_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    bb_gt = np.expand_dims(bb_gt, 0) # 1, M, 5
    bb_test = np.expand_dims(bb_test, 1) # N, 1, 5

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0]) # N, M
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1]) # N, M
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2]) # N, M
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3]) # N, M

    w = np.maximum(0., xx2 - xx1) # N, M
    h = np.maximum(0., yy2 - yy1) # N, M

    intersection_area_matrix = w * h # N, M

    w_test = (bb_test[..., 2] - bb_test[..., 0]) # N, 1
    h_test = (bb_test[..., 3] - bb_test[..., 1]) # N, 1

    area_test = w_test * h_test # N, 1

    w_gt = (bb_gt[..., 2] - bb_gt[..., 0]) # 1, M
    h_gt = (bb_gt[..., 3] - bb_gt[..., 1]) # 1, M

    area_gt = w_gt * h_gt # 1, M

    # (area_test + area_gt).shape == (N, M)
    union_area_matrix = (area_test + area_gt - intersection_area_matrix) # N, M

    iou_matrix = intersection_area_matrix / union_area_matrix # N, M

    return iou_matrix

class IoUAssociator():

    def __init__(self, iou_threshold=0.3):
        self.iou_threshold = iou_threshold

    def associate(self, frame, detections, estimations):
        # frame is not used but other associators may need it
        # estimations is a list of M predictions np.array([x1, y1, x2, y2, score]).reshape((1, 5))
        # detections is a np.array([[x1, y1, x2, y2, score], ...]).reshape(N, 5)

        if len(estimations) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int)

        trackers = np.vstack(estimations) # np.array([[x1, y1, x2, y2, score], ...]).reshape(M, 5)
        estimation_index = np.arange(len(trackers)).reshape(-1, 1)
        base_trackers = np.hstack((trackers, estimation_index))
        trackers = np.ma.compress_rows(np.ma.masked_invalid(base_trackers)) # invalid (NaN and Inf) are deleted so the index are displaced!!!

        estimation_index = trackers[:, -1].reshape(-1)

        iou_matrix = iou_batch(detections, trackers[:, :5])

        pos_associations_matrix = (iou_matrix >= self.iou_threshold).astype(np.int32)

        if pos_associations_matrix.sum(1).max() == 1 and pos_associations_matrix.sum(0).max() == 1:
            # If only 1 candidate of detection-track pair at most (and at least 1 pair)
            matched_indices = np.stack(np.where(pos_associations_matrix), axis=1)

        else:
            # Independently of the thereshold, minimum cost criterion, filtered later
            matched_indices = linear_assignment(-iou_matrix)

        #filter out matched with low IOU
        matches = [m.reshape(1, 2) for m in matched_indices if iou_matrix[m[0], m[1]] >= self.iou_threshold]

        if(len(matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            base_matches = np.concatenate(matches, axis=0)
            matches = np.stack((base_matches[:, 0], estimation_index[base_matches[:, 1]]), axis=1)

        return matches # detection_idx, estimation_idx if a pair exists
        
    __call__ = associate
