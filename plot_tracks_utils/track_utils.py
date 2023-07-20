
from itertools import groupby
import numpy as np
from operator import itemgetter
import pandas as pd
from scipy.optimize import linear_sum_assignment
from typing import Dict, List


def make_equiv_dict(raw_data, data):
    """
    Parameters:
       raw_data: data structure, 1-based & missing-values, read using trackeval.datasets.MotChallenge2DBox().get_raw_seq_data()
       data: data structure, 0-based, read using trackeval.datasets.MotChallenge2DBox().get_preprocessed_seq_data()
    Return:
       ids_equiv : Equivalences between the track ids on the original files (1-based & missing-values, trcks & gt, array values) and the 0-based ones (array index) 
    """

    ids_equiv = dict()
    ids_equiv["tracker"] = np.zeros((data['num_tracker_ids']), dtype=int)
    ids_equiv["gt"] = np.zeros((data['num_gt_ids']), dtype=int)

    for ori_ids, ids in zip(raw_data['tracker_ids'], data['tracker_ids']):
        for ori_id, id in zip(ori_ids, ids):
            ids_equiv["tracker"][id] = ori_id

    for ori_ids, ids in zip(raw_data['gt_ids'], data['gt_ids']):
        for ori_id, id in zip(ori_ids, ids):
            ids_equiv["gt"][id] = ori_id
    
    return ids_equiv

def global_alignment(data):
    potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
    gt_id_count      = np.zeros((data['num_gt_ids'], 1))
    tracker_id_count = np.zeros((1, data['num_tracker_ids']))

    # First loop through each timestep and accumulate global track information.
    for gt_ids_t, tracker_ids_t, similarity in zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores']):
        # Count the potential matches between ids in each timestep
        # These are normalised, weighted by the match similarity.

        sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps

        sim_iou = np.zeros_like(similarity)        
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]

        potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

        # Calculate the total number of dets for each gt_id and tracker_id.
        gt_id_count[gt_ids_t, 0] += 1
        tracker_id_count[0, tracker_ids_t] += 1

    # Calculate overall jaccard alignment score (before unique matching) between IDs
    global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
    return global_alignment_score

def match_gt_pred_tracks (data):
    '''
    Match the predicted and GT tracks using the same method as in TrackEval's HOTA metric. 
    The dataset (GT and tracking files) structure should mimic the one used in MOT20 mot_challenge_2d_box
    
    Parameters:
       data: data structure, read using trackeval.datasets.MotChallenge2DBox().get_preprocessed_seq_data()

    Return:
       final_associations : Associations between gt and tracker ids
    '''
    
    # Get a score from the full sequence
    global_alignment_score = global_alignment(data)

    # Initialize a [time, #trackers] array, It seems that len(data['gt_ids']) are all de frames (included empty ones)
    final_associations = np.ones((len(data['gt_ids']), data['num_tracker_ids']), dtype=int) * (-1)
    
    # Calculate scores for each timestep
    for t, (gt_ids_t, tracker_ids_t, similarity) in enumerate(zip(data['gt_ids'], data['tracker_ids'], data['similarity_scores'])):

        # Get matching scores between pairs of dets for optimizing HOTA
        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

        # Hungarian algorithm to find best matches
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        match_gt_ids    = gt_ids_t[match_rows]
        match_track_ids = tracker_ids_t[match_cols]

        # Assign at each (timestamp, track_id) pairs a ground truth ID
        for trk_id, gt_id in zip(match_track_ids, match_gt_ids):
            final_associations[t, trk_id] = gt_id

    return final_associations


def segmentate_track(df, track_id):
    # Create the list of SORTED frames for this track id
    frames = sorted( list( df.loc[ df.loc[:, 'trackId'] == track_id ]['frameId'] ) )
    # Group by consecutive equidistant frames (when the distance between the index on a SORTED list and its value is constant)
    # g contains (index, frameId) tuples, so we get the frameId item and make a list with them
    segments = [ list(map(itemgetter(1), g)) for _, g in groupby(enumerate(frames), lambda x: x[0] - x[1]) ]

    return segments

def track_segments(df:pd.DataFrame) -> Dict[int, List[List[int]]]:
    '''
    Create a dictionary (key: track id) that for each track has the list of the segments for this track.
    A segment is the list of consecutive frames for a tracklet.
    '''
    
    # list of unique GT track ids
    tids = sorted(list(set(df['trackId'])))

    # For each GT track ...
    seg_dict = dict()
    for track in tids:
        
        segments = segmentate_track(df, track)
        
        # Code de segments by keeping only the first and last frameIds
        red_segs = [(seg[0], seg[-1]) for seg in segments]
        seg_dict[track] = red_segs

    return seg_dict


