
import numpy as np
import scipy
import scipy.linalg


def bbox_to_pred(bbox):
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / (h + np.finfo(float).eps)

    return np.array([x, y, s, r]).reshape((4, 1))

def obbox_to_pred(obbox):
    x = obbox[0]
    y = obbox[1]
    w = obbox[2]
    h = obbox[3]
    angle = obbox[4]

    s = w * h
    r = w / (h + np.finfo(float).eps)

    return np.array([x, y, s, r, angle]).reshape((5, 1))

def bboxes_squared_mahalanobis_dist(dets, pred, covariance, format_function=None):
    format_function = format_function or bbox_to_pred # or obbox_to_pred

    mean = format_function(pred) # Needed because the cavariance uses xysr format instead of ltrb format
    cholesky_factor = np.linalg.cholesky(covariance)
    measurments = format_function(dets)

    d = measurments - mean
    z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)

    squared_maha = np.sum(z * z, axis=0)
    return squared_maha.reshape(-1)

def obboxes_squared_mahalanobis_dist(dets, pred, covariance):
    return bboxes_squared_mahalanobis_dist(dets, pred, covariance, format_function=obbox_to_pred)
