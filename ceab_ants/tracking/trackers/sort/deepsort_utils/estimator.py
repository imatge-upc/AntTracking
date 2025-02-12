
from ceab_ants.tracking.kalman.kalman_filter import KalmanFilter


class DeepSORTEstimator(KalmanFilter):

    def __init__(
            self, 
            num_features, 
            transition_matrix, 
            state_covariance=None, 
            state_uncertainty=None, 
            feature_uncertainty=None,
            process_obs=None, 
            process_pred=None
    ):
        super().__init__(
            num_features=num_features, 
            transition_matrix=transition_matrix, 
            state_covariance=state_covariance, 
            state_uncertainty=state_uncertainty,
            feature_uncertainty=feature_uncertainty
        )

        if process_obs == None:
            process_obs = lambda *x : x
        self.process_obs = process_obs

        if process_pred == None:
            process_pred = lambda *x : x
        self.process_pred = process_pred

        self.age = 0
 
    def predict(self):
        kalman_pred = super().predict()
        covariance = super().state_covariance.copy()

        self.age += 1

        pred = self.process_pred(kalman_pred, covariance, self.age)

        return pred
    
    def update(self, observation):
        kalman_observation = self.process_obs(observation)
        state = super().update(kalman_observation)

        return state

    __call__ = predict
