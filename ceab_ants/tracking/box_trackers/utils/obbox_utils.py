# OBBOX : [x, y, w, h, a, score]
# KALMAN OBSERVATION : [x, y, s, AR, a]

import numpy as np


def obbox_to_observation(bbox=None):
    if bbox is None : return None

    x = bbox[..., 0]
    y = bbox[..., 1]
    w = bbox[..., 2]
    h = bbox[..., 3]
    angle = bbox[..., 4]

    s = w * h
    r = w / (h + np.finfo(float).eps)

    return np.array([x, y, s, r, angle]).reshape((5, -1))

def features_to_obbox(x=None, score=None):
    if x is None: return np.array([-1, -1, -1, -1, -1, -1]).reshape((1, 6))
    score = np.array([score or 0.0])

    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    return np.array([x[0], x[1], w, h, x[4], score]).reshape((1, 6))
