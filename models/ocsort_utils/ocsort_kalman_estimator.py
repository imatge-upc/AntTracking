
from copy import deepcopy
from filterpy.kalman import KalmanFilter
import numpy as np


def speed_direction(bbox1, bbox2):

    cx1 = (bbox1[..., 0] + bbox1[..., 2]) / 2.0 # 1
    cy1 = (bbox1[..., 1] + bbox1[..., 3]) / 2.0 # 1

    cx2 = (bbox2[..., 0] + bbox2[..., 2]) / 2.0 # 1
    cy2 = (bbox2[..., 1] + bbox2[..., 3]) / 2.0 # 1

    speed = np.array([cy2 - cy1, cx2 - cx1]).reshape((1, 2))
    norm = np.sqrt((cy2 - cy1) ** 2 + (cx2 - cx1) ** 2) + 1e-6

    return speed / norm


class OCSortKalmanEstimator():
    # bbox : (center_x, center_y, scale, aspect_ratio); aspect_ratio is considered constant (velocity = 0)

    @classmethod
    def convert_bbox_to_z(cls, bbox): # CCSR
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h + 1e-6)

        return np.array([x, y, s, r]).reshape((4, 1))

    @classmethod
    def convert_x_to_bbox(cls, x, score=None): # LTRB
        score = score or np.array([0.0])

        w = np.sqrt(x[2] * x[3])
        h = x[2] / w

        return np.array([ x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score ], dtype=float).reshape((1, 5))
    
    @classmethod
    def convert_x_to_ccwh(cls, x):

        w = np.sqrt(x[2] * x[3])
        h = np.sqrt(x[2] / x[3])

        return np.array([x[0], x[1], w, h]).reshape((1, 4))

    @classmethod
    def k_previous_obs(cls, history_obs, history_obs_indices, delta_t, score=None):
        if len(history_obs_indices) > 0:
            # Search from delta_t frames ago, "oldest best" criteria
            delta_t_idxs = [idx for idx in history_obs_indices[(-delta_t):] if idx > (len(history_obs) - delta_t)]
            # If no updates for delta_t frames, "newest best" criteria
            prev_idx = delta_t_idxs[0] if len(delta_t_idxs) > 0 else history_obs_indices[-1]
            previous_box = cls.convert_x_to_bbox(history_obs[prev_idx], score=None)

            return previous_box
        else:
            return np.array([-1, -1, -1, -1, -1]).reshape((1, 5))


    @classmethod
    def configure_reducedbbox_kalman(cls, kf):
        # State transistion matrix [(dim_x, dim_x)]:
        #   (cx, cy, s, ar, d_cx, d_cy, d_s) rows
        kf.F = np.array([
            [1,0,0,0,1,0,0],
            [0,1,0,0,0,1,0],
            [0,0,1,0,0,0,1],
            [0,0,0,1,0,0,0],
            [0,0,0,0,1,0,0],
            [0,0,0,0,0,1,0],
            [0,0,0,0,0,0,1]
        ])
        
        # Measurement function [(dim_z, dim_x)]
        kf.H = np.array([
            [1,0,0,0,0,0,0],
            [0,1,0,0,0,0,0],
            [0,0,1,0,0,0,0],
            [0,0,0,1,0,0,0]
        ])

        # Measurement uncertainty/noise [(dim_z, dim_z), default: eye(dim_x)]:
        #   scale and aspect ratio are less certain
        kf.R[2:,2:] *= 10.

        # Covariance matrix [(dim_x, dim_x), default: eye(dim_x)]:
        #   give higher uncertainty to the unobservable initial velocities
        kf.P[4:,4:] *= 1000.
        kf.P *= 10.

        # Process uncertainty/noise [(dim_x, dim_x), default: eye(dim_x)]
        kf.Q[-1,-1] *= 0.01
        kf.Q[4:,4:] *= 0.01

    def __init__(self, bbox, delta_t=3):
        # bbox is an observation; additional estimator options are optional and the constructor may be warped in a function to set them
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.configure_reducedbbox_kalman(self.kf)
        self.kf.update_sequential = None # deactivate a not adapted method

        # initialize kalman with the first observation
        z_bbox = self.convert_bbox_to_z(bbox)
        self.kf.x[:4] = z_bbox

        # OCSort changes
        # keep all observations after creation (only updates)
        self.history_obs = []
        self.history_obs_indices = []
        self.unfreeze_indices = [] # Previous unfreezing observations are not used when unfreezing
        self.delta_t = delta_t

        self.attr_saved = None
        self.observed = False

        self.velocity = np.array((0, 0)).reshape((1, 2))

    def predict(self):
        # returns the prediction as a np.array([x1, y1, x2, y2, score, v_x, v_y, *bbox_last[:5], *bbox_kth[:5]]).reshape((1, 17))

        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()

        estimation = self.convert_x_to_bbox(self.kf.x) # 1, 5
        velocity = self.velocity.reshape(1, 2) # 1, 2
        if len(self.history_obs_indices) > 0:
            bbox_last = self.convert_x_to_bbox(self.history_obs[self.history_obs_indices[-1]]) # 1, 5
        else:
            bbox_last = np.array([-1, -1, -1, -1, -1]).reshape((1, 5))
        bbox_kth = self.k_previous_obs(self.history_obs, self.history_obs_indices, self.delta_t) # 1, 5

        return np.hstack((estimation, velocity, bbox_last, bbox_kth)).reshape((1, 17))

    def update(self, bbox):
        if bbox is None:
            self.history_obs.append(None)

            self.kf_none_update()
            return None

        else:
            if len(self.history_obs_indices) > 0:
                previous_box = self.k_previous_obs(self.history_obs, self.history_obs_indices, self.delta_t)
                self.velocity = speed_direction(previous_box, bbox.reshape(-1))

            z_bbox = self.convert_bbox_to_z(bbox)
            self.history_obs.append(z_bbox)
            self.history_obs_indices.append(len(self.history_obs) - 1)

            if not self.observed:
                # Make a motion model prediction using the previous and current observations
                # Apply predict-update the model as many times as needed to "have the filter at the current time (just after predict)"
                self.unfreeze()
            self.observed = True

            self.kf.update(z_bbox)
            return self.convert_x_to_bbox(self.kf.x)
    
    __call__ = predict

    def kf_none_update(self):
        self.kf._log_likelihood = None
        self.kf._likelihood = None
        self.kf._mahalanobis = None

        if self.observed:
            self.freeze()
        self.observed = False

        self.kf.z = np.array([[None] * self.kf.dim_z]).T
        self.kf.x_post = self.kf.x.copy()
        self.kf.P_post = self.kf.P.copy()
        self.kf.y = np.zeros((self.kf.dim_z, 1))

    def freeze(self):
        self.attr_saved = deepcopy(self.kf.__dict__)

    def unfreeze(self):
        if self.attr_saved is not None:
            
            self.kf.__dict__ = self.attr_saved

            occur = [int((d is None) or (i in self.unfreeze_indices)) for i, d in enumerate(self.history_obs)]
            self.unfreeze_indices.append(len(self.history_obs) - 1) # Do not use anymore the current observation for unfreezing

            indices = np.where(np.array(occur) == 0)[0] # Not None indices
            index1 = indices[-2] if len(indices) > 1 else 0 # last observation idx, but I cannot assure it exists because first observation is not kept!
            index2 = indices[-1] # current observation idx, at least 1
            time_gap = index2 - index1

            box1 = self.history_obs[index1]
            x1, y1, w1, h1 = self.convert_x_to_ccwh(box1).reshape(-1)
            box2 = self.history_obs[index2]
            x2, y2, w2, h2 = self.convert_x_to_ccwh(box2).reshape(-1)

            dx = (x2 - x1) / time_gap
            dy = (y2 - y1) / time_gap 
            dw = (w2 - w1) / time_gap 
            dh = (h2 - h1) / time_gap

            for i in range(1, time_gap + 1):
                x = x1 + i * dx 
                y = y1 + i * dy 
                w = w1 + i * dw 
                h = h1 + i * dh

                s = w * h 
                r = w / float(h)
                
                new_box = np.array([x, y, s, r]).reshape((4, 1))
                self.kf.update(new_box)
                if not i == time_gap:
                    self.kf.predict()
