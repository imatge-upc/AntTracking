
import numpy as np

from ceab_ants.vector_metrics.cosine_distance import cosine_disance


class DeepOCSORTPrototypeDistance():

    def __init__(self, th_conf=0.6, alpha=0.95, distance=None):
        # first element of vector is the observation confidence
        
        self.prototype = None
        self.distance = distance or cosine_disance

        self.th_conf = th_conf
        self.alpha = alpha

    def update(self, vector):

        trust = (vector[0] - self.th_conf) / (1 - self.th_conf)
        vector_alpha = self.alpha + (1 - self.alpha) * (1 - trust)

        self.prototype = self.prototype or vector # if self.prototype == none : self.prototype = vector 
        self.prototype = vector_alpha * self.prototype + (1 - vector_alpha) * vector[1:]
        self.prototype /= np.linalg.norm(self.prototype)

    def forward(self, querys):
        return self.distance(self.prototype, querys)
    
    __call__ = forward
