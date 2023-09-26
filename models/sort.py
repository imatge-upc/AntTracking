
# TrackManagers implements a Iterable interface for active tracks (implements __getitem__(self, key, predict=True) or __iter__(self))
# Each track is a mix of a ESTIMATOR and an APPARENCE SCORING MODEL
# 
# When obtaining a track from the manager, it will have a prediction step applied on the ESTIMATOR (prediction deactivable with predict=False)
# Each track is a buffer of previous predictions before an associated observation:
#   Accessible through __getitem__, the current prediction is the -1 index.
# The track's ESTIMATOR can be updated through  a update(self, detection) method
# The track's APPARENCE SCORING MODEL is applied through a Call (Track implements __call__(self, query))

class Sort():

    def __init__(self, detector, associator, track_manager):
        self.detector = detector
        self.associator = associator
        self.track_manager = track_manager # The estimator is included here

    def update(self, frame):

        # Get detections
        detections = self.detector(frame)
        
        # Apply a prediction step on each track when iterating self.track_manager
        tracks = [track for track in self.track_manager]

        # Associate each detection to a active track or None (so a new track should be created)
        matches = self.associator(frame, detections, tracks)

        # Creates the current output while managing ("updating active", "creating new" and "deleting dead") tracks
        current_outputs = self.track_manager(detections, matches)
        return current_outputs
    
    __call__ = update
