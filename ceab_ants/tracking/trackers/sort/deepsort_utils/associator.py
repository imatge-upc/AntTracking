
import lap
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


chi2inv95 = {
    1 : 3.8415,
    2 : 5.9915,
    3 : 7.8147,
    4 : 9.4877,
    5 : 11.070,
    6 : 12.592,
    7 : 14.067,
    8 : 15.507,
    9 : 16.919
}

def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )

def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )

class DeepSORTAssociator():
    def __init__(self, feature_extractor, score_function, covariance_dist_function, estimations_decoder, detections_decoder=None, average_factor=0, gate_value=1e5, th_score=0.3, th_feats=1, th_cov=None):
        # NOTE: find th_feats empirically

        self.feature_extractor = feature_extractor
        self.score_function = score_function
        self.covariance_dist_function = covariance_dist_function

        self.estimations_decoder = estimations_decoder
        if detections_decoder is None:
            detections_decoder = lambda x : x
        self.detections_decoder = detections_decoder

        self.average_factor = average_factor
        self.gate_value = gate_value
        
        self.th_score = th_score
        self.th_feats = th_feats
        self.th_cov = th_cov or chi2inv95[4]
    
    def associate(self, input_, detections, tracks):
        # input_ is the frame
        # detections has __len__ and is compatible with the choosed cost_function, feature_extractor and covariance_cost_function
        # tracks has __len__, is iterable and callable, its elements are iterable with the last element vstack-able, the result can be decoded by estimations_decoder and the decoded data is compatible with the choosed cost_function, covariance_cost_function and contain track ages

        if len(tracks) == 0 or len(detections) == 0:
            return np.empty((0, 2), dtype=int)
        
        estimations = np.vstack( [trk[-1] for trk in tracks] )
        predictions, covariances, track_ages = self.estimations_decoder(estimations)

        det_features = self.feature_extractor(input_, detections)
        detections = self.detections_decoder(detections)

        feats_dist = np.stack([trk(det_features) for trk in tracks], axis=0)
        matches, unmatched_estimations = self.deepsort_association(detections, feats_dist, predictions, covariances, track_ages)

        unmatched_mask = np.full(det_features.shape[0], True)
        unmatched_mask[matches[:, 0]] = False
        unmatched_dets = np.argwhere(unmatched_mask).reshape(-1)

        if (len(unmatched_estimations) > 0) and (len(unmatched_dets) > 0):
            # TODO: unmatched_dets and unmatched_estimations outside
            new_matches = self.sort_association(detections, unmatched_dets, predictions, unmatched_estimations)
            matches = np.concatenate((matches, new_matches), axis=0)
        
        return matches
    
    __call__ = associate

    def deepsort_association(self, detections, feats_dist, predictions, covariances, track_ages):
        cov_dist = np.stack([self.covariance_dist_function(detections, pred, cov) for pred, cov in zip(predictions, covariances)], axis=0)

        # each row is a prediction, each column is a detection
        gate = (cov_dist > self.th_cov) & (feats_dist > self.th_feats)
        cost = self.average_factor * cov_dist + (1 - self.average_factor) * feats_dist
        cost[gate] = self.gate_value

        young_indexs = [np.argwhere(track_ages == i) for i in np.sort(np.unique(track_ages))]
        young_indexs = np.argwhere(track_ages <= 1)

        matches = np.empty((0, 2), dtype=int)
        for idxs in young_indexs:
            unmatched_mask = np.full(feats_dist.shape[0], True)
            unmatched_mask[matches[:, 1]] = False
            unmatched_idxs = np.argwhere(unmatched_mask).reshape(-1)

            if len(unmatched_idxs) > 0:
                
                aged_cost = cost[np.ix_(idxs[:, 0], unmatched_idxs)].reshape((len(idxs[:, 0]), len(unmatched_idxs)))
                new_matches = linear_assignment(aged_cost) # array of pred, det indexs

                new_matches = [m.reshape(1, 2) for m in new_matches if not gate[m[0], m[1]]]
                if len(new_matches) > 0:
                    new_matches = np.vstack(new_matches[:, ::-1]) # array of det, pred indexs
                else:
                    new_matches = np.empty((0, 2), dtype=int)
                new_matches = np.stack((unmatched_idxs[new_matches[:, 0]], new_matches[:, 1]), axis=1)

                matches = np.concatenate((matches, new_matches), axis=0)

        unmatched_estimations = np.setdiff1d(young_indexs, matches[:, 1], assume_unique=True)

        return matches, unmatched_estimations

    def sort_association(self, detections, unmatched_dets, predictions, unmatched_estimations):
        score_matrix = self.score_function(detections[unmatched_dets, :], predictions[unmatched_estimations, :])#.reshape(len(unmatched_dets), len(unmatched_estimations))

        pos_associations_matrix = (score_matrix >= self.th_cost).astype(np.int32)
        if (pos_associations_matrix.sum(1).max() == 1) and (pos_associations_matrix.sum(0).max() == 1):
            new_matches = np.stack(np.where(pos_associations_matrix), axis=1)
        else:
            new_matches = linear_assignment(-score_matrix)
        
        new_matches = [m.reshape(1, 2) for m in new_matches if score_matrix[m[0], m[1]] >= self.th_score]
        if len(new_matches) > 0:
            new_matches = np.vstack(new_matches)
            new_matches = np.stack((unmatched_dets[new_matches[:, 0]], unmatched_estimations[new_matches[:, 1]]), axis=1)
        else:
            new_matches = np.empty((0, 2), dtype=int)

        return new_matches

