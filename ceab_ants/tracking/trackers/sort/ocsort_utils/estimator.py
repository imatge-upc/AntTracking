# Kalman uses column vectors

import numpy as np

from ceab_ants.tracking.kalman.oru_kalman_filter import ORUKalmanFilter


class OCRecoveryMemory():
        
    def __init__(self, delta_t=3, skip_first=True):
        
        self.history_obs = []
        self.history_obs_indices = []

        self.delta_t = delta_t
        self.skip_first = skip_first

    def __len__(self):
        return len(self.history_obs_indices)

    def __getitem__(self, key):
        return self.history_obs[self.history_obs_indices[key]]
    
    def k_previous_obs(self, delta_t):
        if len(self.history_obs_indices) > 0:
            # Search from delta_t frames ago, "oldest best" criteria
            delta_t_idxs = [idx for idx in self.history_obs_indices[(-delta_t):] if idx > (len(self.history_obs) - delta_t)]
            # If no updates for delta_t frames, "newest best" criteria
            prev_idx = delta_t_idxs[0] if len(delta_t_idxs) > 0 else self.history_obs_indices[-1]

            return self.history_obs[prev_idx]
        else:
            return None
        
    def reduce(self, size=75):
        self.history_obs = self.history_obs[:-size]
        self.history_obs_indices = [i for i, obs in self.history_obs if obs is not None]
    
    def predict(self):
        return self.k_previous_obs(self.delta_t)

    def update(self, observation):
        if self.skip_first:
            self.skip_first = False
            return
        
        self.history_obs.append(observation)

        if observation is not None:
            self.history_obs_indices.append(len(self.history_obs) - 1)
    
    __call__ = predict

def default_momentum(obs1, obs2):
    obs2 = obs2.reshape(obs1.shape)
    speed = np.asarray(obs2) - np.asarray(obs1)
    norm = np.linalg.norm(speed) + 1e-6
    return speed / norm

class OCMomentum():

    def __init__(self, memory=None, delta_t=3, momentum_function=None):
        
        self.internal_memory = memory == None
        self.memory = memory if not self.internal_memory else OCRecoveryMemory()
        self.delta_t = delta_t
        self.momentum_function = momentum_function or default_momentum

        self.velocity = None

    def predict(self):
        return self.velocity if self.velocity is not None else None

    def update(self, observation):
        previous = self.memory.k_previous_obs(self.delta_t)

        if previous is None: return
        self.velocity = self.momentum_function(previous, observation)
        if self.internal_memory:
            self.memory.update(observation)

    __call__ = predict

class OCSORTEstimator(ORUKalmanFilter):
    
    def __init__(
            self, 
            num_features, 
            transition_matrix, 
            state_covariance=None, 
            state_uncertainty=None, 
            feature_uncertainty=None,
            process_obs=None, 
            process_pred=None, 
            unfreeze_update_generator=None, 
            momentum_function=None, 
            delta_t=3
    ):

        super().__init__(
            num_features=num_features, 
            transition_matrix=transition_matrix, 
            state_covariance=state_covariance, 
            state_uncertainty=state_uncertainty,
            feature_uncertainty=feature_uncertainty, 
            unfreeze_update_generator=unfreeze_update_generator
        )

        if process_obs == None:
            process_obs = lambda *x : x
        self.process_obs = process_obs

        if process_pred == None:
            process_pred = lambda *x : x
        self.process_pred = process_pred
        
        self.delta_t = delta_t

        self.ocr = OCRecoveryMemory(delta_t=self.delta_t)
        self.ocm = OCMomentum(memory=self.ocr, delta_t=self.delta_t, momentum_function=momentum_function)

    def predict(self):
        kalman_pred = super().predict()
        momentum = self.ocm.predict()
        last_obs = self.ocr[-1] if len(self.ocr) > 0 else None
        kth_obs = self.ocr.predict()

        pred = self.process_pred(kalman_pred, momentum, last_obs, kth_obs)

        return pred

    def update(self, observation):        
        if observation is not None:
            # These parts of OCSORT work with raw observations
            self.ocm.update(observation)
            self.ocr.update(observation)

        state = super().update(self.process_obs(observation))

        return state

    __call__ = predict

