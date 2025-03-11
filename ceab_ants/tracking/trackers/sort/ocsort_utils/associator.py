
import lap
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )

def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )

class OCSORTAssociator():

    def get_unmatched(self, full_matrix, matched_idxs):
        unmatched_mask = np.full(full_matrix.shape[0], True)
        unmatched_mask[matched_idxs] = False
        unmatched_idxs = np.argwhere(unmatched_mask).reshape(-1)
        unmatched_matrix = full_matrix[unmatched_mask, ...]
        return unmatched_matrix, unmatched_idxs
    
    def __init__(self, first_score_function, momentum_score_function, estimations_decoder, second_score_function=None, detections_decoder=None, th_conf=0.6, th_first_score=0.3, th_second_score=0.3, inertia_weight=0.2, use_byte=False):
        self.first_score_function = first_score_function
        self.momentum_score_function = momentum_score_function
        self.second_score_function = second_score_function or self.first_score_function

        self.estimations_decoder = estimations_decoder

        if detections_decoder is None:
            detections_decoder = lambda x : x
        self.detections_decoder = detections_decoder

        self.th_conf = th_conf
        self.th_first_score = th_first_score
        self.th_second_score = th_second_score

        self.inertia_weight = inertia_weight
        self.use_byte = use_byte

    def associate(self, input_, detections, tracks):

        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int)
        
        estimations = [trk[-1] for trk in tracks]

        trackers, velocities, last_obs, prev_obs = self.estimations_decoder(estimations)
        detections = self.detections_decoder(detections)

        estimation_index = np.arange(len(trackers)).reshape(-1, 1)
        base_trackers = np.hstack((trackers, velocities, last_obs, prev_obs, estimation_index))
        base_trackers = np.ma.compress_rows(np.ma.masked_invalid(base_trackers))

        estimation_index = base_trackers[:, -1].reshape(-1).astype(int)
        trackers = trackers[estimation_index, ...]
        velocities = velocities[estimation_index, ...]
        last_obs = last_obs[estimation_index, ...]
        prev_obs = prev_obs[estimation_index, ...]

        low_mask = detections[:, -1] < self.th_conf
        low_detections = detections[low_mask]
        low_index = np.argwhere(low_mask).reshape(-1)
        high_detections = detections[~low_mask]
        high_index = np.argwhere(~low_mask).reshape(-1)

        high_matches = self.associate_high(high_detections, trackers, velocities, prev_obs)

        low_matches = np.empty((0, 2), dtype=int)
        if self.use_byte and (len(low_detections) > 0) and (len(high_matches) < len(trackers)):
            unmatched_trackers, unmatched_trackers_idx = self.get_unmatched(trackers, high_matches[:, 1])
            low_matches = self.associate_low(low_detections, unmatched_trackers)
            low_matches[:, 1] = unmatched_trackers_idx[low_matches[:, 1]]
                
        if (len(high_matches) < len(high_detections)) and ((len(high_matches) + len(low_matches)) < len(trackers)):
            matched_trackers = np.concatenate((high_matches, low_matches), axis=0)[:, 1]
            unmatched_trackers, unmatched_trackers_idx = self.get_unmatched(last_obs, matched_trackers)

            unmatched_detections, unmatched_detections_idx = self.get_unmatched(high_detections, high_matches[:, 0])

            second_matches = self.associate_low(unmatched_detections, unmatched_trackers)
            second_matches[:, 0] = unmatched_detections_idx[second_matches[:, 0]]
            second_matches[:, 1] = unmatched_trackers_idx[second_matches[:, 1]]

            high_matches = np.concatenate((high_matches, second_matches), axis=0)
        
        rect_high_matches = np.stack((high_index[high_matches[:, 0]], estimation_index[high_matches[:, 1]]), axis=1)
        rect_low_matches = np.stack((low_index[low_matches[:, 0]], estimation_index[low_matches[:, 1]]), axis=1)
        matches = np.concatenate((rect_high_matches, rect_low_matches), axis=0)

        return matches

    __call__ = associate

    def associate_high(self, high_detections, trackers, velocities, prev_obs):

        if len(high_detections) == 0 or len(trackers) == 0:
            return np.empty((0, 2), dtype=int)

        score_matrix = self.first_score_function(high_detections, trackers)

        high_matches = []
        pos_associations_matrix = (score_matrix >= self.th_first_score).astype(np.int32)
        if (pos_associations_matrix.sum(1).max() == 1) and (pos_associations_matrix.sum(0).max() == 1):
            matched_indices = np.stack(np.where(pos_associations_matrix), axis=1)

        else:
            momentum_score = self.momentum_score_function(high_detections, velocities, prev_obs, self.inertia_weight)

            cost = -(score_matrix + momentum_score)

            matched_indices = linear_assignment(cost)
        
        high_matches = [m.reshape(1, 2) for m in matched_indices if score_matrix[m[0], m[1]] >= self.th_first_score]

        if len(high_matches) == 0:
            high_matches = np.empty((0, 2), dtype=int)
        else:
            high_matches = np.concatenate(high_matches, axis=0)
        
        return high_matches
    
    def associate_low(self, low_detections, unmatched_trackers):

        if len(low_detections) == 0 or len(unmatched_trackers) == 0:
            return np.empty((0, 2), dtype=int)

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
