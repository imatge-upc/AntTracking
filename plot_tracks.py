
import cv2
import glob
import os
import pandas as pd
import sys

import trackeval

from docopts.help_plot_tracks import parse_args
from plot_tracks_utils.plot_utils import plot_tracks
from plot_tracks_utils.print_utils import print_config, print_trackfile
from plot_tracks_utils.track_utils import make_equiv_dict, match_gt_pred_tracks


if __name__ == '__main__':

    # Input arguments and options
    gt_folder, tracks_folder, tracker_list, one_file = parse_args(sys.argv)

    # Hardwired values that make it work
    dataset_config = {'TRACKERS_TO_EVAL': tracker_list,'BENCHMARK': 'MOT20', 'GT_FOLDER': gt_folder, 'TRACKERS_FOLDER': tracks_folder}
    cls = 'pedestrian' # Can only check class 'pedestrian' (value 1). Do not use class info in the GT file

    print_config(gt_folder, tracks_folder, tracker_list, one_file, dataset_config)

    # Dataset metadata
    dataset  = trackeval.datasets.MotChallenge2DBox(dataset_config)

    # For each tracker to be analysed
    for tracker_name in tracker_list:
        print(tracker_name)
        
        trfiles_dir = os.path.join(tracks_folder, f'{dataset_config["BENCHMARK"]}-train', tracker_name, 'data')
        gtfiles_dir = os.path.join(gt_folder, f'{dataset_config["BENCHMARK"]}-train')
        
        # For a given tracker, each track filename (with a associated ground truth within a folder with the same name)
        track_files = glob.glob(f'{trfiles_dir}/*.txt')

        for tfn in track_files:

            pred_name = os.path.splitext(os.path.basename(tfn))[0]
            gt_name = os.path.join(gtfiles_dir, pred_name, 'gt/gt.txt')

            # Read GT and tracker files
            df       = pd.read_csv(gt_name, header=0, names=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'a','b','c'])
            df_track = pd.read_csv(tfn, header=0, names=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf', 'a','b','c'])

            # Read tracking and ground truth data
            raw_data  = dataset.get_raw_seq_data(tracker_name, pred_name)
            data = dataset.get_preprocessed_seq_data(raw_data, cls)

            # Associate tracks and ground truth
            final_associations = match_gt_pred_tracks(data)

            ids_equiv = make_equiv_dict(raw_data, data)
            print_trackfile(tfn, pred_name, df, df_track, final_associations, ids_equiv)
            
            # Generate the image
            out_ima  = plot_tracks(df, df_track, final_associations, ids_equiv, ima_size=(12000, 7600), one_file=one_file)
            out_ima = cv2.cvtColor(out_ima, cv2.COLOR_RGB2BGR)
            
            # Save the image
            out_name = f'{tracker_name}-{pred_name}.png'
            print (f'Saving image to {out_name}')
            cv2.imwrite(out_name, out_ima)
