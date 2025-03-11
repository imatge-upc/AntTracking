# Following the citation on github.com/noahcao/OC_SORT

import numpy as np

from .kalman_filter import KalmanFilter


def linear_update_generator(initial_obs, final_obs, time_gap):
    increment = (final_obs - initial_obs) / time_gap
    for i in range(1, time_gap + 1):
        new_obs = initial_obs + i * increment
        yield new_obs

class ORUKalmanFilter(KalmanFilter):

    def __init__(self, num_features, transition_matrix, state_covariance=None, state_uncertainty=None, feature_uncertainty=None, unfreeze_update_generator=None):
        super().__init__(num_features, transition_matrix, state_covariance, state_uncertainty, feature_uncertainty)

        self.previous = None
        self.time_gap = 0

        self.freezed = False

        self.unfreeze_update_generator = unfreeze_update_generator or linear_update_generator

    def freeze(self):
        #if not self.freezed: save attr to unfreeze
        self.freezed = True
    
    def unfreeze(self, observation):

        generator = self.unfreeze_update_generator(self.previous, observation, self.time_gap)
        for i, new_obs in enumerate(generator):

            self._update(new_obs)
            
            generator.send(self)

            if not i == self.time_gap:
                self.predict()

        self.freezed = False
        self.time_gap = 0

    def update(self, observation):

        if observation is None:

            self.time_gap += 1
            self.freeze()

        else:

            observation = np.asarray(observation).reshape(-1, 1)

            if self.freezed:
                self.unfreeze(observation)
            else:
                self.previous = observation

            self._update(observation)
        
        return self.state
    
    def predict(self):
        return super().predict()
    
    __call__ = predict
