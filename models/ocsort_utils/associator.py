
import lap
import numpy as np
from scipy.optimize import linear_sum_assignment

from .metric_utils import iou_batch


def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )
        
def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )


def speed_direction_batch(dets, tracks):
    tracks = tracks[..., np.newaxis] # M, 5 -> M, 5, 1

    cx1 = (dets[:, 0] + dets[:, 2]) / 2.0 # N
    cy1 = (dets[:, 1] + dets[:, 3]) / 2.0 # N

    cx2 = (tracks[:, 0] + tracks[:, 2]) / 2.0 # M, 1
    cy2 = (tracks[:, 1] + tracks[:, 3]) / 2.0 # M, 1
    
    dx = cx1 - cx2 # M, N
    dy = cy1 - cy2 # M, N
    
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6 # M, N
    
    dx = dx / norm # M, N
    dy = dy / norm # M, N

    return dx, dy

def angle_score_batch(detections, velocities, previous_obs, inertia_weight=0.2):
    dx, dy = speed_direction_batch(detections, previous_obs) # M, N & M, N

    inertia_x = np.repeat(velocities[:, [1]], dx.shape[1], axis=1) # M, 2 -> M, 1 -> M, N
    inertia_y = np.repeat(velocities[:, [0]], dy.shape[1], axis=1) # M, 2 -> M, 1 -> M, N

    # diff_angle_cos is a score (better to worse) : 1, 0, -1
    diff_angle_cos = inertia_x * dx + inertia_y * dy # dot-product  : M, N
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)     # M, N

    # the angle is a distance (better to worse) : 0, pi/2 & pi
    diff_angle_dist = np.arccos(diff_angle_cos)                     # M, N

    # diff_angle_score is a score (better to worse) : 0.5, 0, -0.5
    diff_angle_score = 0.5 - (np.abs(diff_angle_dist) / np.pi)      # M, N

    valid_mask = np.ones(previous_obs.shape[0])         # M
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0    # M
    valid_mask = np.repeat(valid_mask[:, np.newaxis], dx.shape[1], axis=1) # M -> M, 1 -> M, N

    scores = np.repeat(detections[:, [-1]], previous_obs.shape[0], axis=1) # N, 5 -> N, 1 -> N, M

    angle_diff_score = (valid_mask * diff_angle_score) * inertia_weight # M, N
    angle_diff_score = angle_diff_score.T                               # N, M
    angle_diff_score = angle_diff_score * scores                        # N, M

    return angle_diff_score

class OCSortAssociator():

    def get_unmatched(self, full_matrix, matched_idxs):
        # bool mask is faster than new array using delete when the number of indices are small (<1000)
        unmatched_mask = np.full(full_matrix.shape[0], True)
        unmatched_mask[matched_idxs] = False
        unmatched_idxs = np.argwhere(unmatched_mask).reshape(-1)
        unmatched_matrix = full_matrix[unmatched_mask, ...]
        return unmatched_matrix, unmatched_idxs

    def __init__(self, det_threshold=0.6, iou_threshold=0.3, second_iou_threshold=0.3, second_asso_func=None, inertia_weight=0.2, use_byte=False):
        self.det_threshold = det_threshold
        self.iou_threshold = iou_threshold
        self.second_iou_threshold = second_iou_threshold
        self.inertia_weight = inertia_weight

        self.second_asso_func = second_asso_func or iou_batch

        self.use_byte = use_byte

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
        velocities = base_trackers[:, 5:7]
        last_boxes = base_trackers[:, 7:12]
        previous_obs = base_trackers[:, 12:17]
        estimation_index = base_trackers[:, -1].reshape(-1)

        low_mask = detections[:, -1] < self.det_threshold
        low_detections = detections[low_mask]
        low_index = np.argwhere(low_mask).reshape(-1)
        high_detections = detections[~low_mask]
        high_index = np.argwhere(~low_mask).reshape(-1)

        # START OF MATCHING
        high_matches = self.associate_high(high_detections, trackers, velocities, previous_obs)
        #print(f'{len(high_matches)}', end='')

        low_matches = np.empty((0, 2), dtype=int)
        if self.use_byte and len(low_detections) > 0 and len(high_matches) < len(trackers):
            #print('kk')
            unmatched_trackers, unmatched_trackers_idx = self.get_unmatched(trackers, high_matches[:, 1])
            low_matches = self.associate_low(low_detections, unmatched_trackers)
            low_matches[:, 1] = unmatched_trackers_idx[low_matches[:, 1]]
        
        if len(high_matches) < len(high_detections) and len(high_matches) + len(low_matches) < len(trackers):
            matched_trackers = np.concatenate((high_matches, low_matches), axis=0)[:, 1]
            unmatched_trackers, unmatched_trackers_idx = self.get_unmatched(last_boxes, matched_trackers)

            unmatched_detections, unmatched_detections_idx = self.get_unmatched(high_detections, high_matches[:, 0])
            #print(f', {len(unmatched_trackers)}, {len(unmatched_detections)}', end='\t')

            second_matches = self.associate_low(unmatched_detections, unmatched_trackers)
            second_matches[:, 0] = unmatched_detections_idx[second_matches[:, 0]]
            second_matches[:, 1] = unmatched_trackers_idx[second_matches[:, 1]]
            high_matches = np.concatenate((high_matches, second_matches), axis=0)
        
        #    print(f'{len(second_matches)}', end='')
        #else:
        #    print(f', N, N\t0', end='')

        # START OF JOINING
        rect_high_matches = np.stack((high_index[high_matches[:, 0]], estimation_index[high_matches[:, 1]]), axis=1)
        rect_low_matches = np.stack((low_index[low_matches[:, 0]], estimation_index[low_matches[:, 1]]), axis=1)
        matches = np.concatenate((rect_high_matches, rect_low_matches), axis=0)

        return matches

    __call__ = associate

    def associate_high(self, high_detections, trackers, velocities, previous_obs):
        high_iou_matrix = iou_batch(high_detections, trackers)

        pos_associations_matrix = (high_iou_matrix >= self.iou_threshold).astype(np.int32)
        if pos_associations_matrix.sum(1).max() == 1 and pos_associations_matrix.sum(0).max() == 1:
            # If only 1 candidate of detection-track pair at most (and at least 1 pair)
            matched_indices = np.stack(np.where(pos_associations_matrix), axis=1)
            
        else:
            angle_diff_score = angle_score_batch(high_detections, velocities, previous_obs, self.inertia_weight)

            # Independently of the thereshold, minimum cost criterion, filtered later
            cost = -(high_iou_matrix + angle_diff_score)
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
    
