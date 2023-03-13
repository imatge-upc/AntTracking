
from filterpy.kalman import KalmanFilter
import numpy as np


class ReducedBBoxKalmanEstimator():
    # bbox : (center_x, center_y, scale, aspect_ratio); aspect_ratio is considered constant (velocity = 0)

    @classmethod
    def convert_bbox_to_z(cls, bbox):
        
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        x = bbox[0] + w/2.
        y = bbox[1] + h/2.
        s = w * h    #scale is just area
        r = w / float(h + 1e-6)

        return np.array([x, y, s, r]).reshape((4, 1))

    @classmethod
    def convert_x_to_bbox(cls, x, score=None):
        score = score or 0.0

        w = np.sqrt(x[2] * x[3])
        h = x[2] / w

        return np.array([x[0] - w/2., x[1] - h/2., x[0] + w/2., x[1] + h/2., score], dtype=float).reshape((1, 5))

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

    def __init__(self, bbox):
        # bbox is an observation; additional estimator options are optional and the constructor may be warped in a function to set them
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.configure_reducedbbox_kalman(self.kf)

        # initialize kalman with the first observation
        self.kf.x[:4] = self.convert_bbox_to_z(bbox)

    def predict(self):

        if((self.kf.x[6] + self.kf.x[2]) <= 0):
            self.kf.x[6] *= 0.0

        self.kf.predict()

        return self.convert_x_to_bbox(self.kf.x)
    
    def update(self, bbox):
        self.kf.update(self.convert_bbox_to_z(bbox))
        return self.convert_x_to_bbox(self.kf.x)
    
    __call__ = predict
