
import numpy as np

from ceab_ants.tracking.trackers.sort.track import Track


class TrackManager():

    def _none(self, *args, **kargs):
        # self._none as features_distance_cls will create the track but None will raise an error on call (it is ok for not appearance tracking)
        return None
    
    def concatenate_and_reshape(self, out, trk, return_pred=False):
        return np.concatenate((out, [trk.id])).reshape(1, -1)
    
    def _no_process_features(self, input_, det):
        return det

    def __init__(
            self,
            estimator_cls,
            features_distance_cls=None,
            feature_extractor=None,
            track_cls=None,
            max_age=1,
            min_hits=3,
            max_last_update=1,
            return_pred=False,
            process_output=None
    ):
        
        self.estimator_cls = estimator_cls
        self.features_distance_cls = features_distance_cls or self._none
        self.feature_extractor = feature_extractor or self._no_process_features
        self.track_cls = track_cls or Track
        
        self.return_pred = return_pred
        self.process_output = process_output or self.concatenate_and_reshape

        self.max_age = max_age
        self.min_hits = min_hits
        self.max_last_update = max_last_update

        self.trackers = []
        self.input_count = 0

    def reset(self):
        self.trackers = []
        self.input_count = 0

    def get_track(self, key, predict=True):
        track = self.trackers[key]
        if predict:
            track.predict()
        return track
    
    def chk_output(self, trk):
        last_update_criteria = trk.time_since_update < self.max_last_update
        hit_streak_criteria = trk.hit_streak >= self.min_hits
        initial_frames_criteria = self.input_count <= self.min_hits

        return last_update_criteria and (hit_streak_criteria or initial_frames_criteria)
    
    def manage_tracks(self, input_, detections, matches):
        self.input_count += 1

        unmatched_detections = np.setdiff1d(np.arange(len(detections)), matches[:, 0], assume_unique=True)
        unmatched_estimations = np.setdiff1d(np.arange(len(self.trackers)), matches[:, 1], assume_unique=True)

        matched = self.update_matched_tracks(input_, detections, matches)
        lost = self.manage_lost_tracks(unmatched_estimations)
        new = self.create_new_tracks(input_, detections, unmatched_detections)

        #print(len(matches), len(unmatched_estimations), len(new))
        
        if len(matched) + len(lost) + len(new) > 0:
            return np.concatenate(matched + lost + new)
        return None

    __getitem__ = get_track
    __call__ = manage_tracks

    def update_matched_tracks(self, input_, detections, matches):
        output = []
        for i_det, i_trck in matches:
            trk = self.trackers[int(i_trck)]
            det = detections[int(i_det)]

            out = trk[-1] if self.return_pred else det

            features = self.feature_extractor(input_, det)
            trk.update(det, features)

            if self.chk_output(trk):
                out = self.process_output(out, trk, return_pred=self.return_pred)
                output.append(out)
        return output
    
    def manage_lost_tracks(self, unmatched_estimations):
        output = []
        unmatched_estimations[::-1].sort()
        for i_trck in unmatched_estimations:
            trk = self.trackers[int(i_trck)]
            pred = trk[-1]

            if np.any(~(np.isfinite(pred))) or (trk.time_since_update > self.max_age):
                self.trackers.pop(i_trck)
            
            else:
                if self.chk_output(trk):
                    out = self.process_output(pred, trk, return_pred=True)
                    output.append(out)
        return output
    
    def create_new_tracks(self, input_, detections, unmatched_detections):
        output = []
        for i_det in unmatched_detections:
            det = detections[int(i_det)]
            
            features = self.feature_extractor(input_, det)
            trk = self.track_cls(self.estimator_cls(), self.features_distance_cls())
            trk.update(det, features)
            self.trackers.append(trk)

            if self.chk_output(trk):
                out = self.process_output(det, trk, return_pred=False)
                output.append(out)
        return output

