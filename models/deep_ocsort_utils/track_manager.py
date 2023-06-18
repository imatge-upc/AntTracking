
import numpy as np

from ..deepsort_utils.track_manager import Track as DeepSortTrack
from ..ocsort_utils.track_manager import TrackManager as OCSortTrackManager


def TrackManager(estiamtor_cls, apparence_scorer_cls, track_cls=None, max_age=1, min_hits=3, max_last_update=1, det_threshold=0.6, return_pred=True):
    
    if track_cls is None:
        track_cls = DeepSortTrack

    return OCSortTrackManager(estiamtor_cls, apparence_scorer_cls, track_cls, max_age, min_hits, max_last_update, det_threshold, return_pred)
