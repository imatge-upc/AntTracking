
from ceab_ants.tracking.trackers.sort.track import Track


class OCSORTTrack(Track):

    def __init__(self, estimator, distance):
        Track.__init__(self, estimator, distance)

    def update(self, observation, features=None):
        output = self.estimator.update(observation)
        
        if observation is not None:
            if self.distance is not None:
                self.distance.update(features or observation) # For DeepOCSORT, first feature is confidence

            self.history = [observation]
            self.time_since_update = 0

            self.hits += 1
            self.hit_streak += 1

            return output
        return self.history[-1]
