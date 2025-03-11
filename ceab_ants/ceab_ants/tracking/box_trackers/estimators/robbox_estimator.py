
import numpy as np

from ceab_ants.tracking.trackers.sort.sort import SORTEstimator
from ceab_ants.tracking.box_trackers.utils.obbox_utils import features_to_obbox, obbox_to_observation


class ROBBoxEstimator():

    num_features = 5

    transition_matrix = np.array([
            [1, 0, 0, 0, 0, 1, 0, 0], # x2 = x1 + dx
            [0, 1, 0, 0, 0, 0, 1, 0], # y2 = y1 + dy
            [0, 0, 1, 0, 0, 0, 0, 0], # s2 = s1; s = w * h (when vertical)
            [0, 0, 0, 1, 0, 0, 0, 0], # AR2 = AR1; AR = w / h (when vertical)
            [0, 0, 0, 0, 1, 0, 0, 1], # angle2 = angle1 + dangle
            [0, 0, 0, 0, 0, 1, 0, 0], # dx
            [0, 0, 0, 0, 0, 0, 1, 0], # dy
            [0, 0, 0, 0, 0, 0, 0, 1]  # dangle # NOTE: If angle doesn't work 360º, just make all inputs the 180º equivalent
        ])
    
    state_covariance = np.array([
            [1e1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1e1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1e1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1e1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1e1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1e4, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e4, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1e4]
        ])
    
    state_uncertainty = np.array([
            [1e-0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 1e-0, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 1e-0, 0.00, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 1e-0, 0.00, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 1e-0, 0.00, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 1e-2, 0.00, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1e-2, 0.00],
            [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1e-2]
        ])
    
    feature_uncertainty = np.array([
            [1e1, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1e1, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1e1, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1e1, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1e1]
        ])

    def __init__(self, estimator_cls=None, postprocess_funct=None, preprocess_funct=None):
        
        estimator_cls = estimator_cls or SORTEstimator
        postprocess_funct = postprocess_funct or features_to_obbox
        preprocess_funct = preprocess_funct or obbox_to_observation

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

        estimation = self.estimator()
        
        return estimation
        
    def update(self, bbox):
        return self.estimator.update(bbox)
    
    __call__ = predict
