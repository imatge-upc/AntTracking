
import numpy as np


class Track():

    instance_count = 0

    @classmethod
    def reset(cls):
        cls.instance_count = 0

    def __init__(self, estimator, apparence_scorer):
        self.estimator = estimator
        self.apparence_scorer = apparence_scorer

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

        prediction = self.estimator()

        self.age += 1
        
        if(self.time_since_update > 0):
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(prediction)

        return prediction
    
    def update(self, observation):
        output = self.estimator.update(observation)
        if observation is not None:
            self.history = []
            self.time_since_update = 0

            self.hits += 1
            self.hit_streak += 1

            return output
        return self.history[-1]
    
    def __getitem__(self, key):
        return self.history[key]
    
    def __call__(self, query):
        raise NotImplementedError()


class TrackManager():

    def __init__(self, estiamtor_cls, apparence_scorer_cls=None, track_cls=None, max_age=1, min_hits=3, max_last_update=1, det_threshold=0.6, return_pred=True):

        self.estiamtor_cls = estiamtor_cls

        self.apparence_scorer_cls = apparence_scorer_cls
        if apparence_scorer_cls is None:
            self.apparence_scorer_cls = lambda det : None

        self.track_cls = track_cls
        if track_cls is None:
            self.track_cls = Track

        self.max_age = max_age
        self.min_hits = min_hits
        self.max_last_update = max_last_update
        self.det_threshold = det_threshold

        self.trackers = []
        self.frame_count = 0

        self.return_pred = return_pred
    
    def reset(self):
        self.trackers = []
        self.frame_count = 0

    def get_track(self, key, predict=True):
        track = self.trackers[key]
        if predict:
            track.predict()
        return track
    
    def chk_output(self, trk):
        last_update_criteria = trk.time_since_update < self.max_last_update
        hit_streak_criteria = trk.hit_streak >= self.min_hits
        initial_frames_criteria = self.frame_count <= self.min_hits

        return last_update_criteria and (hit_streak_criteria or initial_frames_criteria)

    def manage_tracks(self, detections, matches):
        # tracks are on self.trackers, the estiamtion is self.trackers[int(i_trck)][-1]
        # detections is a np.array([[x1, y1, x2, y2, score], ...]).reshape(N, 5)

        # NOTE: Old version: estimations is a list of M predictions np.array([x1, y1, x2, y2, score, v_x, v_y, *bbox_last[:5], *bbox_kth[:5]]).reshape((1, 17))

        self.frame_count += 1

        unmatched_detections = np.setdiff1d(np.arange(len(detections)), matches[:, 0], assume_unique=True)
        unmatched_estimations = np.setdiff1d(np.arange(len(self.trackers)), matches[:, 1], assume_unique=True)
        #print(f', {len(unmatched_estimations)}, {len(unmatched_detections)}', end='\t')

        output = []
        for i_det, i_trck in matches:
            trk = self.trackers[int(i_trck)]
            det = detections[int(i_det)]

            pred_out = trk[-1] if self.return_pred else det

            trk.update(det) # Unfreeze if Freezed, output = next estimation

            # Prepare output from active tracks
            if self.chk_output(trk):
                output.append(np.concatenate((pred_out[..., :4].reshape(-1), [trk.id, pred_out[..., 4]])).reshape(1, -1))
        
        # Make a mirror VIEW of unmatched_estimations so the sorting is descending in the actual unmatched_estimations, so pop won't mess the indexes
        unmatched_estimations[::-1].sort()
        for i_trck in unmatched_estimations:
            trk = self.trackers[int(i_trck)]
            pred = trk[-1]

            if np.any( ~(np.isfinite(pred)) ) or (trk.time_since_update > self.max_age):
                # remove invalid trackers, if I had a list of to remove index, np.delete(arr, index_list_to_del) could be used outside a for
                self.trackers.pop(i_trck)

            else:
                # Update must be done too but pred_out can be prediction or previous
                pred_out = trk.update(None) # Freeze, pred_out = previous estimation (== pred)

                if self.chk_output(trk):
                    output.append(np.concatenate((pred_out[..., :4].reshape(-1), [trk.id, pred_out[..., 4]])).reshape(1, -1))

        # Create new Tracks and Prepare output if needed
        for i_det in unmatched_detections:
            det = detections[int(i_det)]

            if det[4] >= self.det_threshold:
                trk = self.track_cls(self.estiamtor_cls(det), self.apparence_scorer_cls(det))
                self.trackers.append(trk)

                if self.chk_output(trk):
                    output.append(np.concatenate((det[:4], [trk.id, det[..., 4]])).reshape(1, -1))

        #print(f'{len(self.trackers)}, {len(output)}')

        if len(output) > 0:
            return np.concatenate(output)
        return np.empty((0, 6))

    __getitem__ = get_track
    __call__ = manage_tracks
