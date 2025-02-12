
import lap
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )

def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )

class SORTAssociator():

    def __init__(self, score_function, threshold=0.3, estimations_decoder=None, detections_decoder=None):
        self.score_function = score_function
        self.threshold = threshold

        if estimations_decoder is None:
            estimations_decoder = lambda x : np.vstack(x) # A list of row vectors into an array where each row is a vector.
        self.estimations_decoder = estimations_decoder

        if detections_decoder is None:
            detections_decoder = lambda x : x # a one row vector
        self.detections_decoder = detections_decoder

    def associate(self, input_, detections, tracks):
        # input_ is the frame, it won't be used here but may be needed by associators like online versions of deepSORT
        # detections has __len__ and is compatible with the choosed cost_function
        # tracks has __len__, is iterable, its elements are iterable with the last element vstack-able and the result is compatible with the choosed cost_function

        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int)
        
        estimations = [trk[-1] for trk in tracks]
        trackers = self.estimations_decoder(estimations) # np.array([[x1, y1, x2, y2, score], ...]).reshape(M, 5)
        
        estimation_index = np.arange(len(trackers)).reshape(-1, 1)
        base_trackers = np.hstack((trackers, estimation_index))
        trackers = np.ma.compress_rows(np.ma.masked_invalid(base_trackers)) # invalid (NaN and Inf) are deleted so the index are displaced!!!

        estimation_index = trackers[:, -1].reshape(-1)

        base_matches = self.sort_associate(self.detections_decoder(detections), trackers[:, :-1])

        if(len(base_matches) == 0):
            matches = np.empty((0, 2), dtype=int)
        else:
            base_matches = np.concatenate(base_matches, axis=0)
            matches = np.stack((base_matches[:, 0], estimation_index[base_matches[:, 1]]), axis=1)

        return matches # detection_idx, estimation_idx if a pair exists
        
    __call__ = associate

    def sort_associate(self, detections, trackers):
        score_matrix = self.score_function(detections, trackers)

        pos_associations_matrix = (score_matrix >= self.threshold).astype(np.int32)

        if pos_associations_matrix.sum(1).max() == 1 and pos_associations_matrix.sum(0).max() == 1:
            # If only 1 candidate of detection-track pair at most (and at least 1 pair)
            matched_indices = np.stack(np.where(pos_associations_matrix), axis=1)

        else:
            # Independently of the thereshold, minimum cost criterion, filtered later
            matched_indices = linear_assignment(-score_matrix)

        #filter out matched with low IOU
        matches = [m.reshape(1, 2) for m in matched_indices if score_matrix[m[0], m[1]] >= self.threshold]
        
        return matches
