
import lap
import numpy as np
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix):
    # TODO: posible point where a cache could be useful
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )
        
def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )

class MOTGroundTruthAssigner():

    def __init__(self, distance_func, assignment_th=0.5, divergence_th=None):
        self.distance_func = distance_func

        self.assignment_th = assignment_th
        self.divergence_th = divergence_th or assignment_th

        # In my case, I made the tracks continuous in the ground truth, so memory 1 is enough
        # NOTE: In future applications, use a list of "dict" (with max length) so a longer memory is allowed (ideal: maximum ground truth fragmentation)
        self.previous_assigments = np.empty((0, 2), dtype=int)

    def reset(self):
        self.previous_assigments = np.empty((0, 2), dtype=int)

    def step(self, ground_truth, tracks):
        # Inputs are of size N x 10 and M x 10

        if len(ground_truth) == 0 or len(tracks) == 0:
            assigments = np.empty((0, 2), dtype=int)
            self.previous_assigments = assigments
            return assigments
        
        # Compute distances
        bb_trks = tracks[:, 2:6] # M
        bb_gts = ground_truth[:, 2:6] # N
        distance_matrix = self.distance_func(bb_gts, bb_trks) # N, M

        # Find index of previous Ids
        _, prev_gt_idx, _ = np.intersect1d(self.previous_assigments[:, 0], ground_truth[:, 1], assume_unique=True, return_indices=True) # Merge sort: O(n*log(n))
        if len(prev_gt_idx) > 0 and prev_gt_idx.max() >= len(self.previous_assigments):
            print(ground_truth[:, :2])
            print(prev_gt_idx)
            print(self.previous_assigments)
        prev_trk_ids = np.intersect1d(self.previous_assigments[prev_gt_idx, 1], tracks[:, 1], assume_unique=True)
        
        prev_gt_ids = []
        reassign_gt_idx = []
        reassign_trk_idx = []
        for trk in prev_trk_ids:
            prev_gt_ids.append(self.previous_assigments[self.previous_assigments[:, 1] == trk, 0])
            reassign_gt_idx.append(np.argwhere(ground_truth[:, 1] == prev_gt_ids[-1])[0, 0])
            reassign_trk_idx.append(np.argwhere(tracks[:, 1] == trk)[0, 0])
        prev_gt_ids = np.asarray(prev_gt_ids).reshape(prev_trk_ids.shape)

        # Keep index of new IDs
        new_gt_idx = np.setdiff1d(np.arange(len(ground_truth)), reassign_gt_idx, assume_unique=True) # Merge sort: O(n*log(n))
        new_trk_idx = np.setdiff1d(np.arange(len(tracks)), reassign_trk_idx, assume_unique=True)

        # Reassign the previous ones that are inside the divergence_th
        # NOTE: Some Metrics reassign the ones outside the divergence_th but do not keep them in the previous_assigments (freeing the ground truth)
        reassigned_mask = distance_matrix[reassign_gt_idx, reassign_trk_idx] < self.divergence_th
        prev_assigments_ids = np.stack((prev_gt_ids[reassigned_mask], prev_trk_ids[reassigned_mask]), axis=1)
        # assigments_idxs = np.concatenate((reassign_gt_idx[reassigned_mask], reassign_trk_idx[reassigned_mask]), axis=1)

        # Remove assigned elements from the distance matrix
        distance_matrix = distance_matrix[np.ix_(new_gt_idx, new_trk_idx)]

        # Assign the remaining ones with the Hungarian algorithm (keep only the ones that are inside the assigment_th)
        if distance_matrix.size > 0:
            matched_idxs = linear_assignment(-distance_matrix) # Hungarian algorithm: O(n^3)
            matches = [m.reshape(1, 2) for m in matched_idxs if distance_matrix[m[0], m[1]] <= self.assignment_th]
            if len(matches) > 0:
                matches = np.concatenate(matches, axis=0)
                new_gt_ids = ground_truth[new_gt_idx[matches[:, 0]], 1]
                new_trk_ids = tracks[new_trk_idx[matches[:, 1]], 1]
                new_assigments_ids = np.stack((new_gt_ids, new_trk_ids), axis=1)
            else:
                new_assigments_ids = np.empty((0, 2), dtype=int)
        else:
            new_assigments_ids = np.empty((0, 2), dtype=int)

        # Join previous with new
        assigments = np.concatenate((prev_assigments_ids, new_assigments_ids), axis=0)

        # Update the previous_assigments
        self.previous_assigments = assigments
        return assigments

    __call__ = step
