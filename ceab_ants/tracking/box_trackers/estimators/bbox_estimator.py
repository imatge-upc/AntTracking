
import numpy as np

from ceab_ants.tracking.trackers.sort.sort import SORTEstimator
from ceab_ants.tracking.box_trackers.utils.bbox_utils import features_to_bbox, bbox_to_observation


class BBoxEstimator():

    num_features = 4

    transition_matrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0], # x2 = x1 + dx
            [0, 1, 0, 0, 0, 1, 0, 0], # y2 = y1 + dy
            [0, 0, 1, 0, 0, 0, 1, 0], # s2 = s1 + ds; s = w * h
            [0, 0, 0, 1, 0, 0, 0, 1], # AR2 = AR1 + dAR; AR = w / h
            [0, 0, 0, 0, 1, 0, 0, 0], # dx
            [0, 0, 0, 0, 0, 1, 0, 0], # dy
            [0, 0, 0, 0, 0, 0, 1, 0], # ds
            [0, 0, 0, 0, 0, 0, 0, 1]  # dAR
        ])
    
    state_covariance = np.array([
            [1e1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1e1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1e1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1e1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e4]
        ])
    
    state_uncertainty = np.array([
            [1e-0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 1e-0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1e-0, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 1e-0, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1e-2, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 1e-2, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1e-4, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1e-4]
        ])
    
    feature_uncertainty = np.array([
            [1e1, 0.0, 0.0, 0.0],
            [0.0, 1e1, 0.0, 0.0],
            [0.0, 0.0, 1e1, 0.0],
            [0.0, 0.0, 0.0, 1e1],
        ])

    def __init__(self, estimator_cls=None, postprocess_funct=None, preprocess_funct=None):
        
        estimator_cls = estimator_cls or SORTEstimator
        postprocess_funct = postprocess_funct or features_to_bbox
        preprocess_funct = preprocess_funct or bbox_to_observation

        self.estimator = estimator_cls(
            self.num_features,
            self.transition_matrix,
            self.state_covariance,
            self.state_uncertainty,
            self.feature_uncertainty,
            process_obs=preprocess_funct,
            process_pred=postprocess_funct
        )

    def predict(self):

        if (self.estimator.state[6] + self.estimator.state[2]) <= 0:
            self.estimator.state[6] = 0.0
        
        if (self.estimator.state[7] + self.estimator.state[3]) <= 0:
            self.estimator.state[7] = 0.0

        estimation = self.estimator()
        
        return estimation
        
    def update(self, bbox):
        return self.estimator.update(bbox)
    
    __call__ = predict
