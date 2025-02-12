
import numpy as np


def bbox_speed_direction(obs1, obs2):

    center = lambda bbox, shape=-1 : (
        ((bbox[..., 0] + bbox[..., 2]) / 2.).reshape(shape), 
        ((bbox[..., 1] + bbox[..., 3]) / 2.).reshape(shape)
    )

    cx1, cy1 = center(obs1, shape=-1)
    cx2, cy2 = center(obs2, shape=(-1, 1))

    speed = np.array([cy2 - cy1, cx2 - cx1])
    norm = np.linalg.norm(speed)

    nspeed = speed / (norm + np.finfo(speed.dtype).eps)
    
    return nspeed[0, ...], nspeed[1, ...]

def angle_score_bbox_batch(detections, velocities, previous_obs, inertia_weight=0.2):
    # OCSort momentum score

    dx, dy = bbox_speed_direction(detections, previous_obs) # M, N & M, N

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

######### JUST ONE SET #########

def area_batch(bb):
    w = (bb[..., 2] - bb[..., 0]) 
    h = (bb[..., 3] - bb[..., 1])

    area = w * h
    return area

######### PAIR OF SETS #########

def intersection_bbox_batch(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0) # 1, M, 5
    bb_test = np.expand_dims(bb_test, 1) # N, 1, 5

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0]) # N, M
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1]) # N, M
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2]) # N, M
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3]) # N, M

    w = np.maximum(0., xx2 - xx1) # N, M
    h = np.maximum(0., yy2 - yy1) # N, M

    intersection_area_matrix = w * h # N, M
    return intersection_area_matrix

def enclosing_bbox_batch(bb_test, bb_gt):
    # See enclosing box (Minimum bounding box) or Minimum bounding rectangle (MBR)
    bb_gt = np.expand_dims(bb_gt, 0) # 1, M, 5
    bb_test = np.expand_dims(bb_test, 1) # N, 1, 5

    xx1 = np.minimum(bb_test[..., 0], bb_gt[..., 0]) # N, M
    yy1 = np.minimum(bb_test[..., 1], bb_gt[..., 1]) # N, M
    xx2 = np.maximum(bb_test[..., 2], bb_gt[..., 2]) # N, M
    yy2 = np.maximum(bb_test[..., 3], bb_gt[..., 3]) # N, M

    w = xx2 - xx1 # N, M
    h = yy2 - yy1 # N, M

    enclose_area_matrix = w * h # N, M
    return enclose_area_matrix

def center_distance_bbox_batch(bb_test, bb_gt, distance=None):
    if distance is None:
        distance = lambda cx1, cy1, cx2, cy2 : (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    bb_gt = np.expand_dims(bb_gt, 0) # 1, M, 5
    bb_test = np.expand_dims(bb_test, 1) # N, 1, 5

    cx1 = (bb_test[..., 0] + bb_test[..., 2]) / 2.0 # 1, M
    cy1 = (bb_test[..., 1] + bb_test[..., 3]) / 2.0 # 1, M

    cx2 = (bb_gt[..., 0] + bb_gt[..., 2]) / 2.0 # N, 1
    cy2 = (bb_gt[..., 1] + bb_gt[..., 3]) / 2.0 # N, 1

    inner_diag_matrix = distance(cx1, cy1, cx2, cy2) # N, M
    return inner_diag_matrix

def enclosing_diagonal_bbox_batch(bb_test, bb_gt, distance=None):
    if distance is None:
        distance = lambda cx1, cy1, cx2, cy2 : (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    
    bb_gt = np.expand_dims(bb_gt, 0) # 1, M, 5
    bb_test = np.expand_dims(bb_test, 1) # N, 1, 5

    xx1 = np.minimum(bb_test[..., 0], bb_gt[..., 0]) # N, M
    yy1 = np.minimum(bb_test[..., 1], bb_gt[..., 1]) # N, M
    xx2 = np.maximum(bb_test[..., 2], bb_gt[..., 2]) # N, M
    yy2 = np.maximum(bb_test[..., 3], bb_gt[..., 3]) # N, M

    outer_diag_matrix = distance(xx1, yy1, xx2, yy2) # N, M
    return outer_diag_matrix

def aspect_ratio_distance_bbox_batch(bb_test, bb_gt, distance=None):
    if distance is None:
        max_dist = np.pi ** 2. / 4.0 # Normalization term
        distance = lambda w1, h1, w2, h2 : ((np.arctan(w2 / (h2 + 1)) - np.arctan(w1 / (h1 + 1))) ** 2) / max_dist # cuadratic angular distance

    bb_gt = np.expand_dims(bb_gt, 0) # 1, M, 5
    bb_test = np.expand_dims(bb_test, 1) # N, 1, 5

    w1 = bb_test[..., 2] - bb_test[..., 0]
    h1 = bb_test[..., 3] - bb_test[..., 1]
    w2 = bb_gt[..., 2] - bb_gt[..., 0]
    h2 = bb_gt[..., 3] - bb_gt[..., 1]

    ar_dist_matrix = distance(w1, h1, w2, h2)
    return ar_dist_matrix


def iou_bbox_batch(bb_test, bb_gt):
    """
    From SORT: Computes IOU between two bboxes in the form [x1,y1,x2,y2]
    """
    intersection_area_matrix = intersection_bbox_batch(bb_test, bb_gt) # N, M
    area_test = area_batch(np.expand_dims(bb_test, 1)) # N, 1
    area_gt = area_batch(np.expand_dims(bb_gt, 0)) # 1, M

    # (area_test + area_gt).shape == (N, M)
    union_area_matrix = (area_test + area_gt - intersection_area_matrix) # N, M

    iou_matrix = intersection_area_matrix / union_area_matrix # N, M
    return iou_matrix

def giou_bbox_batch(bb_test, bb_gt):
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf

    intersection_area_matrix = intersection_bbox_batch(bb_test, bb_gt) # N, M
    area_test = area_batch(np.expand_dims(bb_test, 1)) # N, 1
    area_gt = area_batch(np.expand_dims(bb_gt, 0)) # 1, M
    enclose_area_matrix = enclosing_bbox_batch(bb_test, bb_gt) # N, M

    union_area_matrix = (area_test + area_gt - intersection_area_matrix) # N, M
    complement_area_matrix = (enclose_area_matrix - union_area_matrix) # N, M

    iou_matrix = intersection_area_matrix / union_area_matrix # N, M
    coe_matrix = complement_area_matrix / enclose_area_matrix # N, M
    
    giou_matrix = iou_matrix - coe_matrix # N, M with values from -1 to 1
    giou_matrix = (giou_matrix + 1.) / 2.0 # N, M with values from 0 to 1
    return giou_matrix


def diou_bbox_batch(bb_test, bb_gt):
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf

    iou_matrix = iou_bbox_batch(bb_test, bb_gt) # N, M

    inner_dist_matrix = center_distance_bbox_batch(bb_test, bb_gt) # N, M
    outer_dist_matrix = enclosing_diagonal_bbox_batch(bb_test, bb_gt) # N, M

    ioo_dist_matrix = inner_dist_matrix / outer_dist_matrix # N, M

    diou_matrix = iou_matrix - ioo_dist_matrix # N, M with values from -1 to 1
    diou_matrix = (diou_matrix + 1.) / 2.0 # N, M with values from 0 to 1
    return diou_matrix

def ciou_bbox_batch(bb_test, bb_gt):
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf

    iou_matrix = iou_bbox_batch(bb_test, bb_gt) # N, M

    inner_dist_matrix = center_distance_bbox_batch(bb_test, bb_gt) # N, M
    outer_dist_matrix = enclosing_diagonal_bbox_batch(bb_test, bb_gt) # N, M

    ioo_dist_matrix = inner_dist_matrix / outer_dist_matrix # N, M

    ar_dist_matrix = aspect_ratio_distance_bbox_batch(bb_test, bb_gt) #N, M [0, 1]
    
    # if S_matrix == 0 : ar_dist_matrix == 0 so ar_cost_matrix == 0/0, it should be 0. In case of 0/0 : 0/eps == 0
    S_matrix = 1 - iou_matrix
    S_matrix[S_matrix == 0] = np.finfo(S_matrix.dtype).eps
    ar_cost_matrix = ar_dist_matrix / (ar_dist_matrix + S_matrix) # N, M

    ciou_matrix = iou_matrix - ioo_dist_matrix - ar_cost_matrix * ar_dist_matrix # N, M
    ciou_matrix = (ciou_matrix + 1.) / 2.0 # N, M with values from 0 to 1
    return ciou_matrix

def ct_dist_bbox_batch(bb_test, bb_gt):
    distance = lambda cx1, cy1, cx2, cy2 : np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    
    ct_dist_matrix = center_distance_bbox_batch(bb_test, bb_gt, distance=distance)
    ct_dist_matrix = ct_dist_matrix / ct_dist_matrix.max()
    ct_score_matrix = ct_dist_matrix.max() - ct_dist_matrix
    return ct_score_matrix

