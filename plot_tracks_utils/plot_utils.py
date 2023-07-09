
import cv2
from itertools import groupby
import numpy as np
from operator import itemgetter
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .color_palettes import list_16_colors, list_32_colors, list_64_colors, list_96_colors
from .track_utils import segmentate_track


def vertical_axis (ima:np.ndarray, num_tracks:int, ids_equiv:np.ndarray, vertical_spacing:int, top_margin:int):

    imaP = Image.fromarray(ima)
    imaD = ImageDraw.Draw(imaP)

    font1 = ImageFont.truetype('Ubuntu-R.ttf', 35)
    font2 = ImageFont.truetype('Ubuntu-R.ttf', 20)
    
    shape = [(60, 0), (60, ima.shape[0] - 1)]
    imaD.line(shape, fill="black", width=2)

    # Draw ticks
    for ii in range(num_tracks + 1):
        pos = top_margin + ii * vertical_spacing

        if ii % 5 == 0:
            imaD.line([(50, pos), (70, pos)], fill="black", width=3)
            imaD.text((5, pos - 20), f'{ii}', fill='black', font=font1)
        else:
            imaD.line([(55, pos), (65, pos)], fill="black", width=2)

        # Write real ids
        if ii < ids_equiv.shape[0]:
            rid = ids_equiv[ii]
            imaD.text((70, pos - 10), f'{rid}', fill='black', font=font2)
    
    return np.asarray(imaP)

def horizontal_axis(ima, num_frames, pixels_per_frame, left_margin):

    imaP = Image.fromarray(ima)
    imaD = ImageDraw.Draw(imaP)

    font1 = ImageFont.truetype('Ubuntu-R.ttf', 25)

    shape = [(0, 50), (ima.shape[1] - 1, 50)]
    imaD.line(shape, fill="black", width=2)
    
    # Draw ticks
    last_tick = ((num_frames // 10) + 1) * 10
    for ii in range(0, last_tick, 10):
        pos = left_margin + int(np.round(ii * pixels_per_frame))

        if ii % 100 == 0:

            imaD.line([(pos, 40), (pos, 60)], fill ="black", width = 3)
            
            if ii == 0:
                pos = pos + 15

            imaD.text((pos - 20, 20), f'{ii}', fill='black', font=font1)
        else:
            imaD.line([(pos, 45), (pos, 55)], fill="black", width=2)

    return np.asarray(imaP)

def set_plot_colors(max_tracks):
    if max_tracks > 64:
        colours = list_96_colors
    elif max_tracks > 32:
        colours = list_64_colors
    elif max_tracks > 16:
        colours = list_32_colors
    else:
        colours = list_16_colors
    
    return colours

def draw_ground_truth(out_ima, tids, df_gt, ids_equiv, frame_offset, top_margin, vertical_spacing, left_margin, pixels_per_frame, line_thickness, color=(0, 0, 0)):
    # For each GT track ...
    for ii in range(len(tids)): # ID 0-based
        # Create the list of frames for this track id
        segments = segmentate_track(df_gt, ids_equiv['gt'][ii])

        for seg in segments:
            seg_coord = list()
            for jj, fr in enumerate(seg):  # fr is 1-based
                # Convert the consecutive trackid to coordinates
                y_coord = top_margin + ii * vertical_spacing
                # Convert the frame_numbers to coordinates
                x_coord = left_margin + int(np.round((fr - frame_offset) * pixels_per_frame))

                seg_coord.append((x_coord, y_coord))

                if jj != 0:
                    x1,y1 = seg_coord[-2]
                    x2,y2 = seg_coord[-1] 
                    cv2.line(out_ima, (x1, y1), (x2, y2), color, thickness=line_thickness)
    
    return tids

def draw_associated(out_ima, ptids, df_track, ids_equiv, frame_offset, final_associations, top_margin, vertical_spacing, left_margin, pixels_per_frame, line_thickness, gt_pred_sep, colours):
    
    # Plot the predicted tracks that are associated with a GT track. Mark the unassociated ones for later.
    associated_tracks = list() # List of predicted tracks associated with any GT
    for ii in range(len(ptids)):

        # Create the list of frames for this track id
        segments = segmentate_track(df_track, ids_equiv['tracker'][ii])

        # Select a color from the list ands convert from #xxxxxx hexadecimal representation to RGB tuple
        color = tuple(int(colours[ii % len(colours)].lstrip('#')[i : i + 2], 16) for i in (0, 2, 4)) 

        for seg in segments:
            seg_coord = list()

            for jj, fr in enumerate(seg):

                fr_idx = fr - 1 # - frame_offset # NOTE: IGNASI 28/06/2023 The original version that works on other data uses the "- frame_offset" instead of "- 1"
                fr_plot_point = fr_idx - frame_offset # NOTE: IGNASI 28/06/2023 When ploting, I do need to correct the offset like the original version
                
                if final_associations[fr_idx, ii] == -1:
                    continue

                # Keep count of the tracks that have been associated
                associated_tracks.append(ii)
                
                y_coord = top_margin + final_associations[fr_idx, ii] * vertical_spacing + gt_pred_sep + line_thickness
                x_coord = left_margin + int(np.round(fr_plot_point * pixels_per_frame))

                seg_coord.append((x_coord, y_coord))

                # if jj != 0: # JRMR 31/05/2023 Fails in some situations because there is only one element in seg_coord
                if len(seg_coord) > 1:
                    x1,y1 = seg_coord[-2]
                    x2,y2 = seg_coord[-1] 
                    cv2.line(out_ima, (x1, y1), (x2, y2), color, thickness=line_thickness)
                    
    associated_tracks = list(set(associated_tracks))
    return associated_tracks

def draw_unassociated(out_ima, unassociated_tracks, df_track, ids_equiv, frame_offset, largest_gt_track_id, top_margin, vertical_spacing, left_margin, pixels_per_frame, line_thickness, gt_pred_sep, colours):

    # Plot the unassociated tracks.
    for ii,tr in enumerate(unassociated_tracks):
        # Create the list of frames for this track id
        segments = segmentate_track(df_track, ids_equiv['tracker'][tr])
        
        # Select a color from the list ands convert from #xxxxxx hexadecimal representation to RGB tuple
        color = tuple(int(colours[ii % len(colours)].lstrip('#')[i : i + 2], 16) for i in (0, 2, 4)) 

        for seg in segments:
            seg_coord = list()
            for jj, fr in enumerate(seg):

                fr_idx = fr - 1 # - frame_offset # NOTE: IGNASI 28/06/2023 The original version that works on other data uses the "- frame_offset" instead of "- 1", in my case it doesn't work with it
                fr_plot_point = fr_idx - frame_offset # NOTE: IGNASI 28/06/2023 When ploting, I do need to correct the offset like the original version

                y_coord = top_margin + (largest_gt_track_id + 1 + ii) * vertical_spacing + gt_pred_sep + line_thickness
                x_coord = left_margin + int(np.round(fr_plot_point * pixels_per_frame))
                
                seg_coord.append((x_coord, y_coord))

                if jj != 0:
                    x1,y1 = seg_coord[-2]
                    x2,y2 = seg_coord[-1] 
                    cv2.line(out_ima, (x1, y1), (x2, y2), color, thickness=line_thickness)
                        
    return unassociated_tracks

def plot_tracks(df_gt:pd.DataFrame, df_track:pd.DataFrame, final_associations:np.ndarray, ids_equiv, ima_size=(3440, 1440), one_file=False, left_margin=100, right_margin=20, top_margin=80):
    """
    Left margin: width in pixels. Can contain white space, vertical axis and vertical label
    Right margin: width in pixels. Contains white space
    Top margin: width in pixels
    """

    # list of unique GT track ids
    tids = sorted(list(set(df_gt['trackId'])))
    # list of unique pred track ids
    ptids = list() if one_file else sorted(list(set(df_track['trackId'])))

    largest_gt_track_id = len(tids) # max(tids)
    largest_pred_track_id = len(ptids) # max(ptids)

    colours = set_plot_colors(largest_pred_track_id)
    
    # list of unique frames
    tot_frames = sorted(list(set(df_gt['frameId'])))
    frame_offset = tot_frames[0]
    num_frames = tot_frames[-1] - frame_offset + 1

    # Drawing configuration
    plot_width = ima_size[0] - left_margin - right_margin
    pixels_per_frame = plot_width / num_frames
    print (f'Pixels/frame = {pixels_per_frame}')

    # Each trackId needs 'vertical_spacing' pixels: 2 for the GT line, 2 for the pred line, 4 empty pixels between GT & pred and 20 empty pixels
    # to separate from the next track
    line_thickness   = 2 # Pixels
    gt_pred_sep      = 4
    tracks_sep       = 20
    vertical_spacing = 2 * line_thickness + gt_pred_sep + tracks_sep

    # Empty canvas
    out_ima = np.ones((ima_size[1], ima_size[0], 3), dtype=np.uint8) * 255

    # Plot vertical axis (track ids)
    out_ima = vertical_axis (out_ima, max(largest_gt_track_id, largest_pred_track_id), ids_equiv['gt'], vertical_spacing, top_margin)
    # Plot horizontal axis (frame #)
    out_ima = horizontal_axis(out_ima, num_frames, pixels_per_frame, left_margin)

    # Draw ground truth black lines. Return what was drawn.
    draw_ground_truth(out_ima, tids, df_gt, ids_equiv, frame_offset, top_margin, vertical_spacing, left_margin, pixels_per_frame, line_thickness)

    if one_file:
        return out_ima
    
    # Draw tracks (that are associated with a GT track) coloured lines. Return List of predicted tracks associated with any GT (what was drawn).
    associated_tracks = draw_associated(out_ima, ptids, df_track, ids_equiv, frame_offset, final_associations, top_margin, vertical_spacing, left_margin, pixels_per_frame, line_thickness, gt_pred_sep, colours)
    
    unassociated_tracks = [x for x in range(len(ptids)) if x not in associated_tracks] 
    
    # Draw remaining tracks coloured lines. Return what was drawn.
    draw_unassociated(out_ima, unassociated_tracks, df_track, ids_equiv, frame_offset, largest_gt_track_id, top_margin, vertical_spacing, left_margin, pixels_per_frame, line_thickness, gt_pred_sep, colours)
    
    return out_ima
