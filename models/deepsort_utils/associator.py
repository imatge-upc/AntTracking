
import lap
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919
}

def linear_assignment(cost_matrix):
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array( [[y[i], i] for i in x if i >= 0] )
        
def linear_assignment2(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array( list(zip(x, y)) )

def bboxes_to_pred(bboxes):
        
    w = bboxes[..., 2] - bboxes[..., 0]
    h = bboxes[..., 3] - bboxes[..., 1]
    
    x = bboxes[..., 0] + w/2.
    y = bboxes[..., 1] + h/2.
    s = w * h    #scale is just area
    r = w / np.float32(h + 1e-6)

    return np.stack([x, y, s, r], axis=-1).reshape((-1, 4))

def squared_mahalanobis_dist(dets, pred, covariance):
    mean = bboxes_to_pred(pred) # 1, 4
    cholesky_factor = np.linalg.cholesky(covariance) # 4, 4
    measurements = bboxes_to_pred(dets) # N, 4

    d = measurements - mean
    z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)

    squared_maha = np.sum(z * z, axis=0)
    return squared_maha.reshape(-1) # N

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


class DeepSortAssociator():
    
    def __init__(self, num_feats=50, average_factor=0, gate_value=1e+5, th_maha=None, th_appa=1, iou_threshold=0.3):
        # NOTE: find th_appa empirically

        self.num_feats = num_feats
        self.average_factor = average_factor
        self.gate_value = gate_value

        self.th_maha = th_maha or chi2inv95[4]
        self.th_appa = th_appa
        self.iou_threshold = iou_threshold
    
    def associate(self, frame, detections, tracks):
        # detections: M
        # estimations: N

        if len(tracks) == 0 or len(detections) == 0:
            #print(f'0, N, N\t0', end='')
            return np.empty((0, 2), dtype=int)
        
        estimations = np.vstack([trk[-1] for trk in tracks])

        det_bboxes = detections[:, :5]
        det_apparences = detections[:, 5:] # M, num_feats

        pred_bboxes = estimations[:, :5]
        covariances = estimations[:, 5:21].reshape(-1, 4, 4)
        pred_ages = estimations[:, -1]

        mahalanobis = np.stack([squared_mahalanobis_dist(det_bboxes, pred, covariance) for pred, covariance in zip(pred_bboxes, covariances)], axis=0) # N, M
        apparence_score = np.stack([trk(det_apparences) for trk in tracks], axis=0) # N, M
        
        gate = (mahalanobis < self.th_maha) & (apparence_score < self.th_appa) # N, M
        cost = self.average_factor * mahalanobis + (1 - self.average_factor) * apparence_score # N, M
        cost[gate] = self.gate_value

        young_indexs = [np.argwhere(pred_ages == i) for i in np.sort(np.unique(pred_ages))]
        youngest_indexs = np.argwhere(pred_ages <= 1)

        matches = np.empty((0, 2), dtype=int)
        for idxs in young_indexs:
            unmatched_mask = np.full(det_apparences.shape[0], True)
            unmatched_mask[matches[:, 1]] = False
            unmatched_idxs = np.argwhere(unmatched_mask).reshape(-1)

            if len(unmatched_idxs) > 0:
                
                aged_cost = cost[idxs[0], unmatched_idxs].reshape((len(idxs[0]), len(unmatched_idxs)))
                new_matches = linear_assignment(aged_cost)

                new_matches = [m.reshape(1, 2) for m in new_matches if not gate[m[0], m[1]]]
                if len(new_matches) > 0:
                    new_matches = np.vstack(new_matches)
                else:
                    new_matches = np.empty((0, 2), dtype=int) # continue
                new_matches = np.stack((new_matches[:, 0], unmatched_idxs[new_matches[:, 1]]), axis=1)

                matches = np.concatenate((matches, new_matches), axis=0)
        
        unmatched_estimations = np.setdiff1d(youngest_indexs, matches[:, 0], assume_unique=True)

        unmatched_mask = np.full(det_apparences.shape[0], True)
        unmatched_mask[matches[:, 1]] = False
        unmatched_dets = np.argwhere(unmatched_mask).reshape(-1)
        
        if len(unmatched_estimations) > 0 and len(unmatched_dets) > 0:
            iou_matrix = iou_batch(pred_bboxes[unmatched_estimations, :5], det_bboxes[unmatched_dets, :5]).reshape(len(unmatched_estimations), len(unmatched_dets))

            pos_associations_matrix = (iou_matrix >= self.iou_threshold).astype(np.int32)
            if pos_associations_matrix.sum(1).max() == 1 and pos_associations_matrix.sum(0).max() == 1:
                # If only 1 candidate of detection-track pair at most (and at least 1 pair)
                new_matches = np.stack(np.where(pos_associations_matrix), axis=1)
            else:
                # Independently of the thereshold, minimum cost criterion, filtered later
                new_matches = linear_assignment(-iou_matrix)
            
            new_matches = [m.reshape(1, 2) for m in new_matches if iou_matrix[m[0], m[1]] >= self.iou_threshold]
            if len(new_matches) > 0:
                new_matches = np.vstack(new_matches)
            else:
                new_matches = np.empty((0, 2), dtype=int) # continue
            new_matches = np.stack((unmatched_estimations[new_matches[:, 0]], unmatched_dets[new_matches[:, 1]]), axis=1)

            matches = np.concatenate((matches, new_matches), axis=0)
        
        matches = np.stack((new_matches[:, 1], new_matches[:, 0]), axis=1)
        return matches

    __call__ = associate
