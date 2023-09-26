
from itertools import chain
import numpy as np

from .track_utils import track_segments


def print_segments(df, df_track):

    # track_segments: Dictionaries: trackId -> List of tuples; each tuple is a segment: (firstFrameId, lastFrameId)
    gt_segments      = track_segments(df)
    tracker_segments = track_segments(df_track)

    print ('Segments for the GT ids')
    for key, val in gt_segments.items():
        print (f'GT id {key} segments: {val}')
    print ('-----------------------------------------------------------------\n')
        
    print ('Segments for the tracker ids')
    for key, val in tracker_segments.items():
        print (f'Tracker id {key} segments: {val}')
    print ('-----------------------------------------------------------------\n')

def print_tracker_info(final_associations, ids_equiv):
            
    # Tracker IDs that do not have association with any GT id:
    unassociated_tracker_ids = [ii for ii in range(final_associations.shape[1]) if np.all(final_associations[:, ii] == final_associations[0, ii])]
    unassociated_tracker_ids = np.array(unassociated_tracker_ids, dtype=int)
    print ('Tracker ids not associated with any GT ids:')
    print (ids_equiv['tracker'][unassociated_tracker_ids].tolist())
    print ('-----------------------------------------------------------------\n')
    
            
    # Associations between tracker id and GT ids
    print ('GT ids associated with each tracker id: ')
    assoc = dict()
    for ii in range(final_associations.shape[1]):
        ll = list(set(final_associations[:, ii][final_associations[:, ii] != -1].tolist()))
        if ll != []:
            assoc[ii] = np.array(ll, dtype=int)
            print (f'Tracker id: {ids_equiv["tracker"][ii]}, Associated GT ids: {ids_equiv["gt"][assoc[ii]].tolist()}')
    print ('-----------------------------------------------------------------\n')

    # Associations between GT id and tracker ids:
    gt_assoc = {ids_equiv["gt"][key] : [] for key in np.unique(final_associations) if key != -1}
    for ii, jj in np.argwhere(final_associations != -1): # track ids
            gtid = final_associations[ii, jj]

            gtid = ids_equiv["gt"][gtid]    # original gt id
            trid = ids_equiv["tracker"][jj] # original tracker id

            gt_assoc[gtid].append(trid)

    print ('Tracker Ids associated with each GT ids:')
    for key, val in gt_assoc.items():
        print (f'GT id: {key}, Associated tracker ids: {list(set(val))}')
    print ('-----------------------------------------------------------------\n')

    # GT ids not associated with any tracker id:
    tot_assoc = list()
    for ii in range(final_associations.shape[1]):
        if ii in assoc.keys():
            tot_assoc.append(assoc[ii].tolist())
    tot_assoc = list(chain.from_iterable(tot_assoc)) # Collapse list levels
            
    associated_gt_ids = list(set(tot_assoc))
    unassociated_gt_ids = [x for x in range(ids_equiv['gt'].shape[0]) if x not in associated_gt_ids]
    print ('GT ids not associated with any tracker ids:')
    if unassociated_gt_ids == []:
        print ([])
    else:
        unassociated_gt_ids = np.array(unassociated_gt_ids, dtype=int)
        print (ids_equiv["gt"][unassociated_gt_ids].tolist())
    print ('-----------------------------------------------------------------\n')

def print_config(gt_folder, tracks_folder, tracker_list, one_file, dataset_config):
    print (f'GT folder: {gt_folder}')
    print (f'Tracks folder: {tracks_folder}')
    print (f'Tracker list: {tracker_list}')
    print (f'One file: {one_file}')
    print (dataset_config)

def print_trackfile(tfn, pred_name, df, df_track, final_associations, ids_equiv):
    print (tfn)
    print (pred_name)
    print_segments(df, df_track)
    print_tracker_info(final_associations, ids_equiv)