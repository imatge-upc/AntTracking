
from ceab_ants.tracking.trackers.sort.ocsort_utils.track import OCSORTTrack
from ceab_ants.tracking.trackers.sort.track_manager import TrackManager


class OCSORTTrackManager(TrackManager):

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
            process_output=None,
            confidence_funct=None,
            th_conf=0.6
    ):
        track_cls = track_cls or OCSORTTrack
        super().__init__(estimator_cls, features_distance_cls, feature_extractor, track_cls, max_age, min_hits, max_last_update, return_pred, process_output)

        self.confidence_funct = confidence_funct
        if self.confidence_funct is None:
            self.confidence_funct = lambda x : 1
        self.th_conf = th_conf

    def manage_lost_tracks(self, unmatched_estimations):
        output = []
        unmatched_estimations[::-1].sort()
        for i_trck in unmatched_estimations:
            trk = self.trackers[int(i_trck)]
            pred = trk[-1]

            if trk.time_since_update > self.max_age:
                self.trackers.pop(i_trck)
            
            else:
                pred = trk.update(None)

                if self.chk_output(trk):
                    out = self.process_output(pred, trk, return_pred=True)
                    output.append(out)
        return output
    
    def create_new_tracks(self, input_, detections, unmatched_detections):
        output = []
        for i_det in unmatched_detections:
            det = detections[int(i_det)]
            
            if self.confidence_funct(det) > self.th_conf:
                features = self.feature_extractor(input_, det)
                trk = self.track_cls(self.estimator_cls(), self.features_distance_cls())
                trk.update(det, features)
                self.trackers.append(trk)

                if self.chk_output(trk):
                    out = self.process_output(det, trk, return_pred=False)
                    output.append(out)
        return output

