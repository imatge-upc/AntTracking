
from ceab_ants.tracking.kalman.kalman_filter import KalmanFilter


class SORTEstimator(KalmanFilter):
    
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

        #NOTE: Be careful about the estimator and detector outputs, by instance, kalamn doesn't predict detection confidence so its output is smaller.
        if process_pred == None:
            process_pred = lambda *x : x
        self.process_pred = process_pred
        
    def predict(self):
        kalman_pred = super().predict()
        pred = self.process_pred(kalman_pred)
        return pred
    
    def update(self, observation):
        kalman_observation = self.process_obs(observation)
        return super().update(kalman_observation)

    __call__ = predict
