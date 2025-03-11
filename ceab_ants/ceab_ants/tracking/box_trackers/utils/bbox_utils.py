
import numpy as np


def bbox_to_observation(bbox=None):
    if bbox is None : return None

    w = bbox[..., 2] - bbox[..., 0]
    h = bbox[..., 3] - bbox[..., 1]

    x = bbox[..., 0] + w / 2.
    y = bbox[..., 1] + h / 2.
    s = w * h
    r = w / (h + np.finfo(float).eps)

    return np.array([x, y, s, r]).reshape((4, -1))

def features_to_bbox(x=None, score=None):
    if x is None : return np.array([-1, -1, -1, -1, -1]).reshape((1, 5))
    score = np.array([score or 0.0])

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    left = x[0] - w / 2.
    top = x[1] - h / 2.
    right = x[0] + w / 2.
    bottom = x[1] + h / 2.

    return np.array([left, top, right, bottom, score]).reshape((1, 5))

def features_to_ccwh(x):
        
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    return np.array([x[0], x[1], w, h]).reshape((1, 4))

def bbox_to_ccwh(bbox):

    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.

    return np.array([x, y, w, h]).reshape((1, 4))

def ccwh_to_features(bbox):

    x = bbox[0]
    y = bbox[1]
    s = bbox[2] * bbox[3]
    r = bbox[2] / (bbox[3] + np.finfo(bbox.dtype).eps)

    return np.array([x, y, s, r]).reshape((4, -1))
