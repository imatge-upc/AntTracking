
import numpy as np


######### JUST ONE SET #########

def area_batch(bb):
    w = (bb[..., 2] - bb[..., 0]) 
    h = (bb[..., 3] - bb[..., 1])

    area = w * h
    return area

def area_batch2(bb_test, bb_gt):
    bb_gt = np.expand_dims(bb_gt, 0) # 1, M, 5
    bb_test = np.expand_dims(bb_test, 1) # N, 1, 5

    area_test = area_batch(bb_test) # N, 1
    area_gt = area_batch(bb_gt) # 1, M

    return area_test, area_gt


######### PAIR OF SETS #########

def intersection_batch(bb_test, bb_gt):
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

def enclosing_batch(bb_test, bb_gt):
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

def center_distance_batch(bb_test, bb_gt, distance=None):
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

def enclosing_diagonal_batch(bb_test, bb_gt, distance=None):
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

def aspect_ratio_distance_batch(bb_test, bb_gt, distance=None):
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


def iou_batch(bb_test, bb_gt):
    intersection_area_matrix = intersection_batch(bb_test, bb_gt) # N, M
    area_test, area_gt = area_batch2(bb_test, bb_gt) # N, 1 & 1, M

    # (area_test + area_gt).shape == (N, M)
    union_area_matrix = (area_test + area_gt - intersection_area_matrix) # N, M

    iou_matrix = intersection_area_matrix / union_area_matrix # N, M
    return iou_matrix

def giou_batch(bb_test, bb_gt):
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf

    intersection_area_matrix = intersection_batch(bb_test, bb_gt) # N, M
    area_test, area_gt = area_batch2(bb_test, bb_gt) # N, 1 & 1, M
    enclose_area_matrix = enclosing_batch(bb_test, bb_gt) # N, M

    union_area_matrix = (area_test + area_gt - intersection_area_matrix) # N, M
    complement_area_matrix = (enclose_area_matrix - union_area_matrix) # N, M

    iou_matrix = intersection_area_matrix / union_area_matrix # N, M
    coe_matrix = complement_area_matrix / enclose_area_matrix # N, M
    
    giou_matrix = iou_matrix - coe_matrix # N, M with values from -1 to 1
    giou_matrix = (giou_matrix + 1.) / 2.0 # N, M with values from 0 to 1
    return giou_matrix


def diou_batch(bb_test, bb_gt):
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf

    iou_matrix = iou_batch(bb_test, bb_gt) # N, M

    inner_dist_matrix = center_distance_batch(bb_test, bb_gt) # N, M
    outer_dist_matrix = enclosing_diagonal_batch(bb_test, bb_gt) # N, M

    ioo_dist_matrix = inner_dist_matrix / outer_dist_matrix # N, M

    diou_matrix = iou_matrix - ioo_dist_matrix # N, M with values from -1 to 1
    diou_matrix = (diou_matrix + 1.) / 2.0 # N, M with values from 0 to 1
    return diou_matrix

def ciou_batch(bb_test, bb_gt):
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf

    iou_matrix = iou_batch(bb_test, bb_gt) # N, M

    inner_dist_matrix = center_distance_batch(bb_test, bb_gt) # N, M
    outer_dist_matrix = enclosing_diagonal_batch(bb_test, bb_gt) # N, M

    ioo_dist_matrix = inner_dist_matrix / outer_dist_matrix # N, M

    ar_dist_matrix = aspect_ratio_distance_batch(bb_test, bb_gt) #N, M [0, 1]
    
    # if S_matrix == 0 : ar_dist_matrix == 0 so ar_cost_matrix == 0/0, it should be 0. In case of 0/0 : 0/eps == 0
    S_matrix = 1 - iou_matrix
    S_matrix[S_matrix == 0] = np.finfo(S_matrix.dtype).eps
    ar_cost_matrix = ar_dist_matrix / (ar_dist_matrix + S_matrix) # N, M

    ciou_matrix = iou_matrix - ioo_dist_matrix - ar_cost_matrix * ar_dist_matrix # N, M
    ciou_matrix = (ciou_matrix + 1.) / 2.0 # N, M with values from 0 to 1
    return ciou_matrix

def ct_dist_batch(bb_test, bb_gt):
    distance = lambda cx1, cy1, cx2, cy2 : np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
    
    ct_dist_matrix = center_distance_batch(bb_test, bb_gt, distance=distance)
    ct_dist_matrix = ct_dist_matrix / ct_dist_matrix.max()
    ct_score_matrix = ct_dist_matrix.max() - ct_dist_matrix
    return ct_score_matrix

