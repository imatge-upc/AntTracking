
class Sort():

    def __init__(self, detector, associator, track_manager):
        self.detector = detector
        self.associator = associator
        self.track_manager = track_manager # The estimator is included here

    def update(self, frame):

        # get detections
        detections = self.detector(frame)
        
        # TrackManagers implements a Iterable interface for active tracks (implements __getitem__(self, key) or __iter__(self))
        # Each track is a callable ESTIMATOR for prediction with a update(self, detection) method for managing
        estimations = [track() for track in self.track_manager]

        # Associate each detection to a active track or None (so a new track should be created)
        matches = self.associator(frame, detections, estimations)

        # Creates the current output while managing ("updating active", "creating new" and "deleting dead") tracks
        current_tracks = self.track_manager(detections, estimations, matches)
        return current_tracks
    
    __call__ = update
