"""  
Plot the GT and predictes tracks for a given sequence. GT and track files should be in MOT20 format:
Tracking and annotation files are simple comma-separated value (CSV) files. Each line represents one 
object instance and contains 9 values for GT and 10 for results files. The first number indicates in 
which frame the object appears, while the second number identifies that object as belonging to a 
trajectory by assigning a unique ID. Each object can be assigned to only one trajectory. The next 
four numbers indicate the position of the bounding box in 2D image coordinates. The position is 
indicated by the top-left corner as well as width and height of the bounding box. 

For the ground truth and results files, the 7th value acts as a flag whether the entry is to be 
considered. A value of 0 means that this particular instance is ignored in the evaluation, while a 
value of 1 is used to mark it as active. In the GT files, the 8th number indicates the type of object 
annotated. A value of 1 should be used (scoring and visualization are class-agnostic). The last number
shows the visibility ratio of each bounding box. 

In the results files, the 7th, 8th, 9th and 10th numbers are -1.

GT file example: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <valid>, <1>, <ignored>, 
Results file example: <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <ignored>, <ignored>, <ignored>, <ignored>

Source: MOT20: A benchmark for multi object tracking in crowded scenes. https://arxiv.org/pdf/2003.09003.pdf

Usage:
  plot_tracks.py <gtFolder> <tracksFolder> [--trackerList=<tl>] [--oneFile=<of>]
  plot_tracks.py -h | --help

  <gtFolder>              Folder with the gt annotations (MOT20 format). For instance: data/gt/mot_challenge/
  <trackFolder>           Folder with the tracking results (MOT20 format). For instance: data/trackers/mot_challenge/
  -------------
  USAGE EXAMPLE: cd /imatge/morros/workspace/mediapro/post_tracking_reid;  python plot_tracks.py data/gt/mot_challenge/ data/trackers/mot_challenge/ --trackerList=OTrack
Options:
  --trackerList=<tl>      Name of the trackers to evaluate. String separated by commas [default: '']
  --oneFile=<of>          Plot just <gtFolder> [default: False]
"""

import pandas as pd
import numpy as np
from operator import itemgetter
from itertools import groupby
from itertools import chain
import cv2
import trackeval
from scipy.optimize import linear_sum_assignment
from color_palettes import list_16_colors, list_32_colors, list_64_colors, list_96_colors
import sys
import os
import glob
from docopt import docopt
from distutils.util import strtobool
#from PIL import Image

from track_reid import track_segments


def match_gt_pred_tracks (raw_data, cls, dataset):
    '''
    Match the predicted and GT tracks using the same method as in TrackEval's HOTA metric. 
    The dataset (GT and tracking files) structure should mimic the one used in MOT20 mot_challenge_2d_box
    Parameters:
       raw_data: data structure, read using trackeval.datasets.MotChallenge2DBox().get_raw_seq_data()
       cls: The class of the objects. 
       dataset: a dataset object 
    Return:
       final_associations : Associations between gt and tracker ids
       ids_equiv : Equivalences between the track ids on the original files and the 0-based ones 
    '''
    # Code adapted from https://github.com/JonathonLuiten/TrackEval (retrieved April, 24th)
    
    # In raw_data, tracks and timestamps are 1-based. In data, they are zero-based
    data = dataset.get_preprocessed_seq_data(raw_data, cls)


    # Equivelence between track ids in raw_data (original ids in text files)
    # and in data (consecutive ids, starting at zero)
    ids_equiv = dict()
    ids_equiv["tracker"] = np.zeros((data['num_tracker_ids']), dtype=int)

    for ii, (ori_ids, ids) in enumerate(zip(raw_data['tracker_ids'], data['tracker_ids'])):
        for jj in range(len(ori_ids)):
            ids_equiv["tracker"][ids[jj]] = ori_ids[jj]

    ids_equiv["gt"] = np.zeros((data['num_gt_ids']), dtype=int)
    for ii, (ori_ids,ids) in enumerate(zip(raw_data['gt_ids'], data['gt_ids'])):
        for jj in range(len(ori_ids)):
            ids_equiv["gt"][ids[jj]] = ori_ids[jj]
            
    array_labels = np.arange(0.05, 0.99, 0.05)

    potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
    gt_id_count      = np.zeros((data['num_gt_ids'], 1))
    tracker_id_count = np.zeros((1, data['num_tracker_ids']))


    # First loop through each timestep and accumulate global track information.
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
        # Count the potential matches between ids in each timestep
        # These are normalised, weighted by the match similarity.
        similarity = data['similarity_scores'][t]
        sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
        sim_iou = np.zeros_like(similarity)
        sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
        sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
        potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

        # Calculate the total number of dets for each gt_id and tracker_id.
        gt_id_count[gt_ids_t] += 1
        tracker_id_count[0, tracker_ids_t] += 1

    # Calculate overall jaccard alignment score (before unique matching) between IDs
    global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
    matches_counts = [np.zeros_like(potential_matches_count) for _ in array_labels]

    final_associations = np.ones((len(data['gt_ids']), data['num_tracker_ids']),dtype=int) * (-1)

    
    # Calculate scores for each timestep
    for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):

        # Get matching scores between pairs of dets for optimizing HOTA
        similarity = data['similarity_scores'][t]
        score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

        # Hungarian algorithm to find best matches
        match_rows, match_cols = linear_sum_assignment(-score_mat)

        match_gt_ids    = gt_ids_t[match_rows]
        match_track_ids = tracker_ids_t[match_cols]

        for ii in range(0,len(match_track_ids)):
            final_associations[t,match_track_ids[ii]] = match_gt_ids[ii]

    return final_associations, ids_equiv




from PIL import Image,ImageDraw,ImageFont
def vertical_axis (ima:np.ndarray, num_tracks:int, ids_equiv:np.ndarray, vertical_spacing:int, top_margin:int):

    imaP = Image.fromarray(ima)
    imaD = ImageDraw.Draw(imaP)

    font1 = ImageFont.truetype('Ubuntu-R.ttf', 35)
    font2 = ImageFont.truetype('Ubuntu-R.ttf', 20)
    
    shape = [(60,0), (60,ima.shape[0]-1)]
    
    imaD.line(shape, fill ="black", width = 2)
    

    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.line(ima, (60,0), (60,ima.shape[0]-1), (0,0,0), thickness=2)

    for ii in range(num_tracks + 1):
        pos = top_margin + ii*vertical_spacing
        if ii % 5 == 0:
            
            imaD.line([(50,pos),(70,pos)], fill ="black", width = 3)
            imaD.text((5,pos-20),f'{ii}', fill='black', font=font1)
            #cv2.line(ima,(50,pos),(70,pos), (0,0,0), thickness=3) # Ticks
            #cv2.putText(ima, f'{ii}', (5,pos+10), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            imaD.line([(55,pos),(65,pos)], fill ="black", width = 2)
            #cv2.line(ima,(55,pos),(65,pos), (0,0,0), thickness=2) # Ticks

        # Write real ids
        if ii < ids_equiv.shape[0]:
            rid = ids_equiv[ii]
            imaD.text((70,pos-10),f'{rid}', fill='black', font=font2)
            
    
    return np.asarray(imaP)


def horizontal_axis(ima, num_frames, pixels_per_frame, left_margin):
    imaP = Image.fromarray(ima)
    imaD = ImageDraw.Draw(imaP)

    font1 = ImageFont.truetype('Ubuntu-R.ttf', 25)
    imaD.line([(0,50), (ima.shape[1]-1,50)], fill ="black", width = 2)
    
    #font = cv2.FONT_HERSHEY_SIMPLEX

    last_tick = ((num_frames//10)+1)*10
    
    #cv2.line(ima, (0,50), (ima.shape[1]-1,50), (0,0,0), thickness=2)
    for ii in range(0,last_tick,10):
        pos = left_margin + int(np.round(ii*pixels_per_frame))

        if ii % 100 == 0:
            imaD.line([(pos,40),(pos,60)], fill ="black", width = 3)
            if ii == 0:
                pos = pos + 15
            imaD.text((pos-20, 20),f'{ii}', fill='black', font=font1)
            #cv2.line(ima,(pos,40),(pos,60), (0,0,0), thickness=3) # Ticks
            #cv2.putText(ima, f'{ii}', (pos-10,35), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            imaD.line([(pos,45),(pos,55)], fill ="black", width = 2)
            #cv2.line(ima,(pos,45),(pos,55), (0,0,0), thickness=2) # Ticks

    return np.asarray(imaP)


def plot_tracks(df_gt:pd.DataFrame, df_track:pd.DataFrame, final_associations:np.ndarray, ima_size=(3440,1440), one_file=False):

    # list of unique GT track ids
    tids = sorted(list(set(df_gt['trackId'])))

    largest_gt_track_id   = len(tids) # max(tids)    
    
    if one_file:
        ptids = list()
        largest_pred_track_id = 0
    else:
        # list of unique pred track ids
        ptids = sorted(list(set(df_track['trackId'])))
        largest_pred_track_id = len(ptids) # max(ptids)

    max_tracks = largest_pred_track_id # max(largest_pred_track_id,largest_gt_track_id)
    if max_tracks > 64:
        colours = list_96_colors
    elif max_tracks > 32:
        colours = list_64_colors
    elif max_tracks > 16:
        colours = list_32_colors
    else:
        colours = list_16_colors

    # list of unique frames
    tot_frames = sorted(list(set(df_gt['frameId'])))
    num_frames = tot_frames[-1] - tot_frames[0] + 1

    left_margin   = 100 # Left margin width in pixels. Can contain white space, vertical axis and vertical label
    right_margin  = 20 # Right margin width in pixels. Contains white space
    top_margin    = 80 # Top margin in pixels
    bottom_margin = 0 # Bottom margin in pixels. Contains horizontal axis, horizontal label and white space
    
    plot_width = ima_size[0] - left_margin - right_margin
    pixels_per_frame = plot_width / num_frames
    print (f'Pixels/frame = {pixels_per_frame}')

    # Each trackId needs 'vertical_spacing' pixels: 2 for the GT line, 2 for the pred line, 4 empty pixels between GT & pred and 8 empty pixels
    # to separate from the next track
    line_thickness   = 2 # Pixels
    gt_pred_sep      = 4
    tracks_sep       = 20
    vertical_spacing = 2*line_thickness + gt_pred_sep + tracks_sep

    out_ima = np.ones((ima_size[1],ima_size[0],3), dtype=np.uint8)* 255

    # Plot vertical axis (track ids)
    out_ima = vertical_axis (out_ima, max(largest_gt_track_id, largest_pred_track_id), ids_equiv['gt'], vertical_spacing, top_margin)
    # Plot horizontal axis (frame #)
    out_ima = horizontal_axis(out_ima, num_frames, pixels_per_frame, left_margin)

    # For each GT track ...
    for ii in range(len(tids)):
        # Create the list of frames for this track id
        frames = sorted(list(df_gt.loc[df_gt.loc[:,'trackId'] == ids_equiv['gt'][ii]]['frameId']))

        # split the list of frames into smaller lists based on the frames missing in the sequence
        # https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
        segments = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(frames), lambda x: x[0]-x[1])]

        color = (0,0,0)
        
        segs_coord = list()
        for seg in segments:
            seg_coord = list()
            for jj, fr in enumerate(seg):  # Here, fr is 1-based
                # Convert the consecutive trackid to coordinates
                y_coord = top_margin + ii*vertical_spacing
                # Convert the frame_numbers to coordinates
                x_coord = left_margin + int(np.round((fr-tot_frames[0]) * pixels_per_frame))

                seg_coord.append((x_coord,y_coord))

                if jj != 0:
                    x1,y1 = seg_coord[-2]
                    x2,y2 = seg_coord[-1] 
                    cv2.line(out_ima, (x1, y1), (x2, y2), color, thickness=line_thickness)
                    
            segs_coord.append(seg_coord)

    if one_file:
        return out_ima
    
    associated_tracks = list() # List of predicted tracks associated with any GT

    # Plot the predicted tracks that are associated with a GT track. Mark the unassociated ones for later.
    for ii in range(len(ptids)):
        
        # Create the list of frames for this track id
        frames = sorted(list(df_track.loc[df_track.loc[:,'trackId'] == ids_equiv['tracker'][ii]]['frameId']))

        # Split the list of frames into smaller lists based on the frames missing in the sequence
        # https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
        segments = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(frames), lambda x: x[0]-x[1])]

        # Select a color from the list ands convert from #xxxxxx hexadecimal representation to RGB tuple
        color = tuple(int(colours[ii%len(colours)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) 

        segs_coord = list()
        for kk, seg in enumerate(segments):
            seg_coord = list()

            assoc_id = -1
            for jj, fr in enumerate(seg):
                if final_associations[fr-tot_frames[0],ii] == -1:
                    continue

                # Keep count of the tracks that have been associated
                associated_tracks.append(ii)
                
                y_coord = top_margin + final_associations[fr-tot_frames[0],ii]*vertical_spacing + gt_pred_sep + line_thickness
                x_coord = left_margin + int(np.round((fr-tot_frames[0]) * pixels_per_frame))

                seg_coord.append((x_coord,y_coord))

                if jj != 0:
                    x1,y1 = seg_coord[-2]
                    x2,y2 = seg_coord[-1] 
                    cv2.line(out_ima, (x1, y1), (x2, y2), color, thickness=line_thickness)
                    
            segs_coord.append(seg_coord)

    associated_tracks   = list(set(associated_tracks))
    unassociated_tracks = [x for x in range(len(ptids)) if x not in associated_tracks] 
    
    # Plot the unassociated tracks.
    for ii,tr in enumerate(unassociated_tracks):
        # Create the list of frames for this track id
        frames = sorted(list(df_track.loc[df_track.loc[:,'trackId']==ids_equiv['tracker'][tr]]['frameId']))

        # Split the list of frames into smaller lists based on the frames missing in the sequence
        # https://stackoverflow.com/questions/3149440/splitting-list-based-on-missing-numbers-in-a-sequence
        segments = [list(map(itemgetter(1), g)) for k, g in groupby(enumerate(frames), lambda x: x[0]-x[1])]
        
        # Select a color from the list ands convert from #xxxxxx hexadecimal representation to RGB tuple
        color = tuple(int(colours[ii%len(colours)].lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) 

        segs_coord = list()
        for seg in segments:
            seg_coord = list()
            for jj, fr in enumerate(seg):
                y_coord = top_margin + (largest_gt_track_id+1+ii)*vertical_spacing + gt_pred_sep + line_thickness
                x_coord = left_margin + int(np.round((fr-tot_frames[0]) * pixels_per_frame))
                seg_coord.append((x_coord,y_coord))

                if jj != 0:
                    x1,y1 = seg_coord[-2]
                    x2,y2 = seg_coord[-1] 
                    cv2.line(out_ima, (x1, y1), (x2, y2), color, thickness=line_thickness)
                    
            segs_coord.append(seg_coord)
            
    return out_ima



def print_tracker_info(df, df_track, final_associations, ids_equiv):

    gt_segments      = track_segments(df)
    tracker_segments = track_segments(df_track)

    print ('Segments for the GT ids')
    for key,val in gt_segments.items():
        print (f'GT id {key} segments: {val}')
    print ('-----------------------------------------------------------------\n')
        
    print ('Segments for the tracker ids')
    for key,val in tracker_segments.items():
        print (f'Tracker id {key} segments: {val}')
    print ('-----------------------------------------------------------------\n')
            
    # Tracker IDs that do not have association with any GT id:
    unassociated_tracker_ids = list()
    for ii in range(final_associations.shape[1]):
        if np.all(final_associations[:,ii] == final_associations[0,ii]):
            unassociated_tracker_ids.append(ii)
    unassociated_tracker_ids = np.array(unassociated_tracker_ids, dtype=int)
    print ('Tracker ids not associated with any GT ids:')
    print (ids_equiv['tracker'][unassociated_tracker_ids].tolist())
    print ('-----------------------------------------------------------------\n')
    
            
    # List of associations between tracker and GT
    print ('GT ids associated with each tracker id: ')
    assoc = dict()
    for ii in range(final_associations.shape[1]):
        ll = list(set(final_associations[:,ii][final_associations[:,ii] != -1].tolist()))
        if ll != []:
            assoc[ii] = np.array(ll, dtype=int)
            print (f'Tracker id: {ids_equiv["tracker"][ii]}, Associated GT ids: {ids_equiv["gt"][assoc[ii]].tolist()}')
    print ('-----------------------------------------------------------------\n')

    # List of associations between GT and tracker ids:
    gt_assoc = dict()
    for ii in range(final_associations.shape[0]): # Frame numbers
        for jj in range(final_associations.shape[1]): # track ids
            gtid = final_associations[ii,jj]
            if gtid == -1:
                continue
            gtid = ids_equiv["gt"][gtid]    # original gt id
            trid = ids_equiv["tracker"][jj] # original tracker id
            if gtid not in gt_assoc:
                gt_assoc[gtid] = [trid]
            else:
                gt_assoc[gtid].append(trid)

    print ('Tracker Ids associated with each GT ids:')
    for key,val in gt_assoc.items():
        print (f'GT id: {key}, Associated tracker ids: {list(set(val))}')
    print ('-----------------------------------------------------------------\n')

    # GT ids not associated with any tracker id:
    tot_assoc = list()
    for ii in range(final_associations.shape[1]):
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

    

if __name__ == '__main__':
    # read arguments    
    args = docopt(__doc__)
    
    gt_folder     = args["<gtFolder>"]          # /mnt/gpid08/datasets/sports_analytics/SoccerNet/tracking/train/SNMOT-170/img1
    tracks_folder = args["<tracksFolder>"]     # /mnt/gpid08/datasets/sports_analytics/SoccerNet/tracking/train/SNMOT-170/gt/gt.txt
    tracker_list  = args['--trackerList'].split(',')
    one_file      = bool(strtobool(args['--oneFile']))

    print (f'GT folder: {gt_folder}')
    print (f'Tracks folder: {tracks_folder}')
    print (f'Tracker list: {tracker_list}')
    print (f'One file: {one_file}')

    dataset_config = {'TRACKERS_TO_EVAL': tracker_list,'BENCHMARK': 'MOT20', 'GT_FOLDER':gt_folder, 'TRACKERS_FOLDER': tracks_folder}
    cls = 'pedestrian' # Can only check class 'pedestrian' (value 1). Do not use class info in the GT file

    print (dataset_config)
    
    dataset  = trackeval.datasets.MotChallenge2DBox(dataset_config)

    for tracker_name in tracker_list:
        print (tracker_name)
        
        trfiles_dir = os.path.join(tracks_folder, f'{dataset_config["BENCHMARK"]}-train', tracker_name, 'data')
        gtfiles_dir = os.path.join(gt_folder, f'{dataset_config["BENCHMARK"]}-train')
        
        track_files = glob.glob(f'{trfiles_dir}/*.txt')
        
        for tfn in track_files:
            print (tfn)
            
            pred_name = os.path.splitext(os.path.basename(tfn))[0]
            print (pred_name)
            
            gt_name   = os.path.join(gtfiles_dir, pred_name, 'gt/gt.txt')

            # Read GT and tracker files. Note: check whether it is possible to skip this and use the information directly from final_associations and ids_equiv.
            df       = pd.read_csv(gt_name, header=0, names=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'a','b','c'])
            df_track = pd.read_csv(tfn, header=0, names=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf', 'a','b','c'])

            raw_data  = dataset.get_raw_seq_data(tracker_name, pred_name)
            final_associations, ids_equiv = match_gt_pred_tracks(raw_data, cls, dataset)


            gt_segments      = track_segments(df)
            tracker_segments = track_segments(df_track)


            print_tracker_info(df, df_track, final_associations, ids_equiv)

            
            out_ima  = plot_tracks(df,df_track, final_associations, one_file=one_file)

            out_ima = cv2.cvtColor(out_ima, cv2.COLOR_RGB2BGR)
            
            out_name = f'{tracker_name}-{pred_name}.png'
            print (f'Saving image to {out_name}')
            cv2.imwrite(out_name,out_ima)

