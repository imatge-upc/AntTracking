
import numpy as np


def cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)

class PrototypeApparenceScorer():

    def __init__(self, det, det_threshold=0.6, alpha_fixed_emb=0.95):
        
        self.prototype = det[5:].reshape(1, -1)

        self.det_threshold = det_threshold
        self.alpha_fixed_emb = alpha_fixed_emb

    def update(self, det):

        trust = (det[4] - self.det_threshold) / (1 - self.det_threshold)
        af = self.alpha_fixed_emb
        dets_alpha = af + (1 - af) * (1 - trust)

        self.prototype = dets_alpha * dets_alpha + (1 - dets_alpha) * det[5:].reshape(1, -1)
        self.prototype /= np.linalg.norm(self.prototype)

    def forward(self, querys):
        return cosine_distance(self.prototype, querys)

    __call__ = forward

# apparence_scorer_cls = lambda det : PrototypeApparenceScorer(det, det_threshold, alpha_fixed_emb)
