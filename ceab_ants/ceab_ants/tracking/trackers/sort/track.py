
class Track():

    instance_count = 0

    @classmethod
    def reset(cls):
        cls.instance_count = 0
    
    def __init__(self, estimator, distance):
        self.estimator = estimator
        self.distance = distance # None or a callable object to compute the distance between the track and a query that has an update(self, x) method

        Track.instance_count += 1
        self.id = Track.instance_count

        self.history = []
        self.time_since_update = 0

        self.age = 0

        self.hits = 0
        self.hit_streak = 0
    
    def predict(self):

        prediction = self.estimator()
        
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
        
        self.time_since_update += 1
        self.history.append(prediction)

        return prediction
    
    def update(self, observation, features=None):
        if observation is None: # SORT TrackManager directly does nothing if it is None
            return None
        
        if self.distance is not None:
            self.distance.update(features or observation)
        
        self.history = [observation]
        self.time_since_update = 0

        self.hits += 1
        self.hit_streak += 1

        return self.estimator.update(observation)
    
    def __getitem__(self, key):
        return self.history[key]
    
    def __len__(self):
        return len(self.history)
    
    def __call__(self, query):
        return self.distance(query)
    