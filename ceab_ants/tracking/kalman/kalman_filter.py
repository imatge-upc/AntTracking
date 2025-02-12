
import numpy as np


class KalmanFilter(object):
    # On projects make a class where building parameters are fixed, IO addapted to the problem

    def __init__(self, num_features, transition_matrix, state_covariance=None, state_uncertainty=None, feature_uncertainty=None, init=False):
        
        self.num_features = num_features
        self.transition_matrix = np.asarray(transition_matrix) # from state i to i+1

        assert len(self.transition_matrix.shape) == 2
        assert self.transition_matrix.shape[0] == self.transition_matrix.shape[1]
        assert self.transition_matrix.shape[0] >= self.num_features
        assert (state_covariance is None) or (np.asarray(state_covariance).shape == self.transition_matrix.shape)
        assert (state_uncertainty is None) or (np.asarray(state_uncertainty).shape == self.transition_matrix.shape)
        assert (feature_uncertainty is None) or (np.asarray(feature_uncertainty).shape == (self.num_features, self.num_features))

        self.num_states = self.transition_matrix.shape[0]
        self.num_hidden = self.num_states - self.num_features

        self.state = np.zeros((self.num_states, 1))
        self.init = init
        
        self.observation_matrix = np.concatenate((np.eye(self.num_features), np.zeros((self.num_features, self.num_hidden))), axis=1)

        self.state_uncertainty = np.asarray(state_uncertainty) if state_uncertainty is not None else np.eye(self.num_states)
        self.state_covariance = np.asarray(state_covariance) if state_covariance is not None else np.eye(self.num_states) # self.transition_matrix @ self.state_covariance @ self.transition_matrix.T + self.state_uncertainty
        self.feature_uncertainty = np.asarray(feature_uncertainty) if feature_uncertainty is not None else np.eye(self.num_features)

    def predict(self):

        self.state = self.transition_matrix @ self.state    
        self.state_covariance = self.transition_matrix @ self.state_covariance @ self.transition_matrix.T + self.state_uncertainty

        return self.observation_matrix @ self.state
    
    def _update(self, observation):

        observation = np.asarray(observation).reshape(-1, 1)

        if not self.init:
            self.state[:self.num_features] = observation
            self.init = True
            return

        error = observation - self.observation_matrix @ self.state

        covariance_projection = self.state_covariance @ self.observation_matrix.T
        uncertainty = self.observation_matrix @ covariance_projection + self.feature_uncertainty
        inv_uncertainty = np.linalg.inv(uncertainty)
        gain = covariance_projection @ inv_uncertainty

        self.state = self.state + gain @ error

        state_covariance_gain = np.eye(self.num_states) - gain @ self.observation_matrix
        self.state_covariance = state_covariance_gain @ self.state_covariance @ state_covariance_gain.T + gain @ self.feature_uncertainty @ gain.T

    def update(self, observation):

        if observation is not None: self._update(observation)
        return self.state

    __call__ = predict
