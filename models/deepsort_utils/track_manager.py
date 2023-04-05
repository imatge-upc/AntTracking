
import numpy as np


class Track():

    instance_count = 0
    
    memory_size = 100

    @classmethod
    def reset(cls):
        cls.instance_count = 0

    def __init__(self, estimator, det):
        self.estimator = estimator
        self.apparence_memory = [det[5:].reshape(1, -1)]

        # identify the instance with a unique ID for dead and alive tracks (starts with 1 for MOT)
        Track.instance_count += 1
        self.id = Track.instance_count

        # Track statistics
        self.history = []
        self.time_since_update = 0 # equal to len(self.history)
        
        self.age = 0

        self.hits = 0
        self.hit_streak = 0

    def predict(self):

        prediction = self.estimator() # 1, 21 (5 of bbox & 16 of 4x4 covariance)

        self.age += 1
        
        if(self.time_since_update > 0):
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(prediction)

        return np.concatenate((prediction, *self.apparence_memory, np.full((1, 1), self.age)), axis=1) # 1, 5+4x4+Nx50+1
    
    def update(self, observation):
        if observation is not None:
            self.history = []
            self.time_since_update = 0

            self.hits += 1
            self.hit_streak += 1

            self.apparence_memory.append(observation[5:].reshape(1, -1))
            self.apparence_memory = self.apparence_memory[-self.memory_size:]

            return self.estimator.update(observation)
        return self.history[-1]
    
    __call__ = predict

class TrackManager():

    @classmethod
    def set_track_memory(cls, memory_size):
        Track.memory_size = memory_size

    def __init__(self, estiamtor_cls, max_age=1, min_hits=3, max_last_update=1, memory_size=100):

        self.estiamtor_cls = estiamtor_cls
        self.set_track_memory(memory_size)

        self.max_age = max_age
        self.min_hits = min_hits
        self.max_last_update = max_last_update

        self.trackers = []
        self.frame_count = 0
    
    def reset(self):
        self.trackers = []
        self.frame_count = 0

    def get_track(self, key):
        return self.trackers[key]

    def manage_tracks(self, detections, estimations, matches):
        # estimations is a list of predictions np.array([x1, y1, x2, y2, score]).reshape((1, 5))
        # detections is a np.array([[x1, y1, x2, y2, score], ...]).reshape(N, 5)
        self.frame_count += 1

        unmatched_detections = np.setdiff1d(np.arange(len(detections)), matches[:, 0], assume_unique=True)
        unmatched_estimations = np.setdiff1d(np.arange(len(estimations)), matches[:, 1], assume_unique=True)

        output = []
        for i_det, i_trck in matches:
            trk = self.trackers[int(i_trck)]
            det = detections[int(i_det)]

            pred_out = trk.update(det)

            # Prepare output from active tracks
            if (trk.time_since_update < self.max_last_update) and ((trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)):
                output.append(np.concatenate((pred_out[..., :4].reshape(-1), [trk.id])).reshape(1, -1))
        
        # Make a mirror VIEW of unmatched_estimations so the sorting is descending in the actual unmatched_estimations, so pop won't mess the indexes
        unmatched_estimations[::-1].sort()
        for i_trck in unmatched_estimations:
            pred = estimations[int(i_trck)]
            trk = self.trackers[int(i_trck)]

            if np.any(~(np.isfinite(pred))) or trk.time_since_update > self.max_age:
                # remove invalid trackers
                self.trackers.pop(i_trck)

            else:
                pred_out = pred

                if (trk.time_since_update < self.max_last_update) and ((trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)):
                    output.append(np.concatenate((pred_out[..., :4].reshape(-1), [trk.id])).reshape(1, -1))

        # Create new Tracks and Prepare output if needed
        for i_det in unmatched_detections:
            det = detections[int(i_det)]

            trk = Track(self.estiamtor_cls(det), det)
            self.trackers.append(trk)

            if (trk.time_since_update < self.max_last_update) and ((trk.hit_streak >= self.min_hits) or (self.frame_count <= self.min_hits)):
                output.append(np.concatenate((det[:4], [trk.id])).reshape(1, -1))

        if len(output) > 0:
            return np.concatenate(output)
        return np.empty((0, 5))

    __getitem__ = get_track
    __call__ = manage_tracks
