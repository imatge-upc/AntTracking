# For DeepSORT feature distance

import numpy as np


def cosine_disance(a, b, data_is_normalized=False):
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    return 1.0 - np.dot(a, b.T)


class NNCosineDistance():
    # Nearest Neighbor Cosine Distance

    def __init__(self, memory_size=100):
        
        self.memory = list()
        self.memory_size = memory_size
    
    def update(self, vector):
        self.memory.append(vector)
        self.memory = self.memory[-self.memory_size:]
    
    def forward(self, querys):
        vector_memory = np.vstack(self.memory)
        return cosine_disance(vector_memory, querys).min(axis=0)
    
    __call__ = forward

    # scorer_cls = lambda vector : NNCosineScorer(vector, memory_size=my_memory_size)

