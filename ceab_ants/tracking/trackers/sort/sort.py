
from ceab_ants.tracking.trackers.sort.associator import SORTAssociator # NOTE: Tell it how the detections and estimations are read (estimations_decoder and detection_decoder)
from ceab_ants.tracking.trackers.sort.estimator import SORTEstimator # NOTE: Tell it how is your estimation (process_pred); the track manager will use them and detections indistinctly, either make it similar or manage it on the manager
from ceab_ants.tracking.trackers.sort.track import Track as SORTTrack
from ceab_ants.tracking.trackers.sort.track_manager import TrackManager as SORTTrackManager # NOTE: Tell it how do you want the output (process_output); if the predicions and detections are incompatible, solve it


class SORT():

    def __init__(self, detector, associator, track_manager):
        #NOTE: Be careful about the estimator and detector outputs, by instance, kalamn doesn't predict detection confidence so its output is smaller.

        self.detector = detector
        self.associator = associator
        self.track_manager = track_manager # The estimator is included here

    def update(self, input_):

        # Get detections
        detections = self.detector(input_)
        
        # Apply a prediction step on each track when iterating self.track_manager
        tracks = [track for track in self.track_manager]

        # Associate each detection to a active track or None (so a new track should be created)
        matches = self.associator(input_, detections, tracks)

        # Creates the current output while managing ("updating active", "creating new" and "deleting dead") tracks and managing appearence
        current_outputs = self.track_manager(input_, detections, matches)
        return current_outputs
    
    __call__ = update
    