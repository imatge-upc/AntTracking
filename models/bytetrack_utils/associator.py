
import lap
import numpy as np
from scipy.optimize import linear_sum_assignment

from ..ocsort_utils.metric_utils import iou_batch


def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )
        
def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )

class ByteAssociator():

    def get_unmatched(self, full_matrix, matched_idxs):
        # bool mask is faster than new array using delete when the number of indices are small (<1000)
        unmatched_mask = np.full(full_matrix.shape[0], True)
        unmatched_mask[matched_idxs] = False
        unmatched_idxs = np.argwhere(unmatched_mask).reshape(-1)
        unmatched_matrix = full_matrix[unmatched_mask, ...]
        return unmatched_matrix, unmatched_idxs

    def __init__(self, det_threshold=0.6, iou_threshold=0.3, second_iou_threshold=0.3, second_asso_func=None):
        self.det_threshold = det_threshold
        self.iou_threshold = iou_threshold
        self.second_iou_threshold = second_iou_threshold

        self.second_asso_func = second_asso_func or iou_batch

    def associate(self, frame, detections, estimations):
        # frame is not used but other associators may need it
        # estimations is a list of M predictions np.array([x1, y1, x2, y2, score, v_x, v_y, *bbox_last[:5], *bbox_kth[:5]]).reshape((1, 17))
        # detections is a np.array([[x1, y1, x2, y2, score], ...]).reshape(N, 5)

        # START OF PREPARATION
        if len(estimations) == 0 or len(detections) == 0:
            #print(f'0, N, N\t0', end='')
            return np.empty((0, 2), dtype=int)

        base_trackers = np.vstack(estimations) # np.array([[x1, y1, x2, y2, score, v_x, v_y, *bbox_last[:5], *bbox_kth[:5]], ...]).reshape(M, 17)
        estimation_index = np.arange(len(base_trackers)).reshape(-1, 1)
        base_trackers = np.hstack((base_trackers, estimation_index))
        base_trackers = np.ma.compress_rows(np.ma.masked_invalid(base_trackers)) # invalid (NaN and Inf) are deleted so the index are displaced!!!

        trackers = base_trackers[:, :5]
        estimation_index = base_trackers[:, -1].reshape(-1)

        low_mask = detections[:, -1] < self.det_threshold
        low_detections = detections[low_mask]
        low_index = np.argwhere(low_mask).reshape(-1)
        high_detections = detections[~low_mask]
        high_index = np.argwhere(~low_mask).reshape(-1)

        # START OF MATCHING
        high_matches = self.associate_high(high_detections, trackers)

        low_matches = np.empty((0, 2), dtype=int)
        if len(low_detections) > 0 and len(high_matches) < len(trackers):
            unmatched_trackers, unmatched_trackers_idx = self.get_unmatched(trackers, high_matches[:, 1])
            low_matches = self.associate_low(low_detections, unmatched_trackers)
            low_matches[:, 1] = unmatched_trackers_idx[low_matches[:, 1]]

        # START OF JOINING
        rect_high_matches = np.stack((high_index[high_matches[:, 0]], estimation_index[high_matches[:, 1]]), axis=1)
        rect_low_matches = np.stack((low_index[low_matches[:, 0]], estimation_index[low_matches[:, 1]]), axis=1)
        matches = np.concatenate((rect_high_matches, rect_low_matches), axis=0)

        return matches

    __call__ = associate

    def associate_high(self, high_detections, trackers):
        high_iou_matrix = iou_batch(high_detections, trackers)

        pos_associations_matrix = (high_iou_matrix >= self.iou_threshold).astype(np.int32)
        if pos_associations_matrix.sum(1).max() == 1 and pos_associations_matrix.sum(0).max() == 1:
            # If only 1 candidate of detection-track pair at most (and at least 1 pair)
            matched_indices = np.stack(np.where(pos_associations_matrix), axis=1)
        else:
            # Independently of the thereshold, minimum cost criterion, filtered later
            cost = -(high_iou_matrix)
            matched_indices = linear_assignment(cost)
        
        #filter out linear assignments matched with low IOU
        high_matches = [m.reshape(1, 2) for m in matched_indices if high_iou_matrix[m[0], m[1]] >= self.iou_threshold]

        if(len(high_matches) == 0):
            high_matches = np.empty((0, 2), dtype=int)
        else:
            high_matches = np.concatenate(high_matches, axis=0)

        return high_matches # high_detection_idx, estimation_idx if a pair exists

    def associate_low(self, low_detections, unmatched_trackers):
        low_score_matrix = self.second_asso_func(low_detections, unmatched_trackers)
        low_score_matrix = np.array(low_score_matrix)

        matched_indices = []
        if low_score_matrix.max() > self.second_iou_threshold:
            matched_indices = linear_assignment(-low_score_matrix)
        
        #filter out linear assignments matched with low IOU
        low_matches = [m.reshape(1, 2) for m in matched_indices if low_score_matrix[m[0], m[1]] >= self.second_iou_threshold]

        if(len(low_matches) == 0):
            low_matches = np.empty((0, 2), dtype=int)
        else:
            low_matches = np.concatenate(low_matches, axis=0)
        
        return low_matches
    
