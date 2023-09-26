
import numpy as np


def cosine_distance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)

class NNApparenceScorer():

    def __init__(self, det, memory_size=100):
        
        self.memory = [det[5:].reshape(1, -1)]
        self.memory_size = memory_size

    def update(self, det):
        self.memory.append(det[5:].reshape(1, -1))
        self.memory = self.memory[-self.memory_size:]

    def forward(self, querys):
        apparence_memory = np.vstack(self.memory)
        return cosine_distance(apparence_memory, querys).min(axis=0)

    __call__ = forward

# apparence_scorer_cls = lambda det : NNApparenceScorer(det, my_memory_size)
