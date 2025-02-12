
import lap
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )

def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )

class ByteAssociator():

    def get_unmatched(self, full_matrix, matched_idxs):
        unmatched_mask = np.full(full_matrix.shape[0], True)
        unmatched_mask[matched_idxs] = False
        unmatched_idxs = np.argwhere(unmatched_mask).reshape(-1)
        unmatched_matrix = full_matrix[unmatched_mask, ...]
        return unmatched_matrix, unmatched_idxs
    
    def __init__(self, first_score_function, second_score_function=None, estimations_decoder=None, detections_decoder=None, th_conf=0.6, th_first_score=0.3, th_second_score=0.3):

        self.first_score_function = first_score_function
        self.second_score_function = second_score_function or self.first_score_function

        if estimations_decoder is None:
            estimations_decoder = lambda x : np.vstack(x)
        self.estimations_decoder = estimations_decoder

        if detections_decoder is None:
            detections_decoder = lambda x : x
        self.detections_decoder = detections_decoder

        self.th_conf = th_conf
        self.th_first_score = th_first_score
        self.th_second_score = th_second_score

    def associate(self, input_, detections, tracks):

        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int)
        
        estimations = [trk[-1] for trk in tracks]

        base_trackers = self.estimations_decoder(estimations)
        estimation_index = np.arange(len(base_trackers)).reshape(-1, 1)
        base_trackers = np.hstack((base_trackers, estimation_index))
        base_trackers = np.ma.compress_rows(np.ma.masked_invalid(base_trackers))

        trackers = base_trackers[:, :-1]
        estimation_index = base_trackers[:, -1].reshape(-1)

        detections = self.detections_decoder(detections)

        low_mask = detections[:, -1] < self.th_conf
        low_detections = detections[low_mask]
        low_index = np.argwhere(low_mask).reshape(-1)
        high_detections = detections[~low_mask]
        high_index = np.argwhere(~low_mask).reshape(-1)

        high_matches = self.associate_high(high_detections, trackers)

        low_matches = np.empty((0, 2), dtype=int)
        if (len(low_detections) > 0) and (len(high_matches) < len(trackers)):
            unmatched_trackers, unmatched_trackers_idx = self.get_unmatched(trackers, high_matches[:, 1])
            low_matches = self.associate_low(low_detections, unmatched_trackers)
            low_matches[:, 1] = unmatched_trackers_idx[low_matches[:, 1]]
        
        rect_high_matches = np.stack((high_index[high_matches[:, 0]], estimation_index[high_matches[:, 1]]), axis=1)
        rect_low_matches = np.stack((low_index[low_matches[:, 0]], estimation_index[low_matches[:, 1]]), axis=1)
        matches = np.concatenate((rect_high_matches, rect_low_matches), axis=0)

        return matches

    __call__ = associate

    def associate_high(self, high_detections, trackers):
        score_matrix = self.first_score_function(high_detections, trackers)

        pos_associations_matrix = (score_matrix >= self.th_first_score).astype(np.int32)
        if (pos_associations_matrix.sum(1).max() == 1) and (pos_associations_matrix.sum(0).max() == 1):
            matched_indices = np.stack(np.where(pos_associations_matrix), axis=1)
        else:
            matched_indices = linear_assignment(-score_matrix)
        
        matches = [m.reshape(1, 2) for m in matched_indices if score_matrix[m[0], m[1]] >= self.th_first_score]

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)
        
        return matches
    
    def associate_low(self, low_detections, unmatched_trackers):
        score_matrix = self.second_score_function(low_detections, unmatched_trackers)
        score_matrix = np.array(score_matrix)

        matched_indices = []
        if score_matrix.max() > self.th_second_score:
            matched_indices = linear_assignment(-score_matrix)
        
        low_matches = [m.reshape(1, 2) for m in matched_indices if score_matrix[m[0], m[1]] >= self.th_second_score]

        if len(low_matches) == 0:
            low_matches = np.empty((0, 2), dtype=int)
        else:
            low_matches = np.concatenate(low_matches, axis=0)

        return low_matches
