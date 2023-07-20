
from docopt import docopt
import numpy as np
import os
import pandas as pd
import sys

from reid_clustering_test_utils.analysis_utils import (
    compute_ground_truth,
    mean_descriptor,
    compute_matrix,
    compute_merging,
    compare_first_with_last
)
from reid_clustering_test_utils.plot_utils import (
    plot_global_feats,
    plot_distance_info,
    plot_first_last
)


def read_tracks_with_apparence(tracking_file):
    seq_dets = np.loadtxt(tracking_file, delimiter=',', dtype=np.float64)
    tracklets_df = pd.DataFrame(seq_dets[:, :10], columns=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf','a','b', 'c'])
    feats = seq_dets[:, 10:]
    tracklets_df['feats'] = feats.tolist()

    return tracklets_df

DOCTEXT = f"""
Usage:
  reid_clustering_test.py <tracking_file> <gt_tracking_file> <output_dir> [--thr=<th>]

Options:
  --thr=<th>      Max normalized distance between tracklets features [default: 0.00025]
"""


if __name__ == '__main__':

    # INPUT CONFIG
    args = docopt(DOCTEXT, argv=sys.argv[1:], help=True, version=None, options_first=False)

    tracking_file = args['<tracking_file>']
    gt_tracking_file = args['<gt_tracking_file>']
    output_dir = args['<output_dir>']
    thr = float(args['--thr'])

    #  READ DATA
    gt_tracking_df = read_tracks_with_apparence(gt_tracking_file)
    tracklets_df = read_tracks_with_apparence(tracking_file)

    # MAKE VALID PYTHON MATRIX INDEX
    trackIds = np.sort(np.unique(tracklets_df['trackId']))
    index = {tck_id : mat_id for mat_id, tck_id in enumerate(trackIds)}
    tracklets_df['trackIdx'] = tracklets_df['trackId'].apply(lambda x : index[x])

    # COMPUTE TRACKLETS trackIdx GROUND TRUTH trackId
    split_to_gt, gt_to_split = compute_ground_truth(trackIds, tracklets_df, gt_tracking_df, index)

    # COMPUTE TRACKLETS MEAN FEATURES
    descriptors = {mat_id : mean_descriptor(tracklets_df[tracklets_df['trackId'] == tck_id]) for mat_id, tck_id in enumerate(trackIds)}
    descriptors_df = pd.DataFrame(descriptors)

    # COMPUTE THE DISTANCE BETWEEN TRACKLET FEATURES
    dist_matrix = compute_matrix(tracklets_df, descriptors_df, causality=True)

    # FILTER OUT BIG DISTANCES
    merging_matrix = dist_matrix.copy()
    merging_matrix[merging_matrix > thr] = np.inf

    # GET THE RELATION BETWEEN TRACKLETS
    merging_dict = compute_merging(merging_matrix)
    
    # EDIT THE TRACK ID SO IT IS VALID MOT
    merged_tracklets_df = tracklets_df[['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf','a','b', 'c']].copy()
    merged_tracklets_df['trackId'] = tracklets_df['trackIdx'].apply(lambda x : merging_dict.get(x, x) + 1)

    # RANK 1 ACCURACY
    rank1 = sum([1 for k, v in split_to_gt.items() if merging_dict.get(k, k) in gt_to_split[v]]) / len(split_to_gt)
    # CORRECT MERGINGS (precision)
    precision = sum([1 for k, v in merging_dict.items() if split_to_gt[k] == split_to_gt[v]]) / len(merging_dict)

    # PX, TIME and APPARENCE DISTANCE BETWEEN SPLITS FROM THE SAME TRACK, APPARENCE DISTANCE WITH OTHER TRACKS IN THE TIME FRAME
    fr_dist, px_dist, app_dist, err_app_dist = compare_first_with_last(gt_to_split, tracklets_df, gt_tracking_df)

    print(f'rank1 = {rank1}')
    print(f'precision = {precision}')

    global_feats_hist, global_range_bar = plot_global_feats(descriptors)
    tracklet_dist_hist = plot_distance_info(dist_matrix, gt_to_split)
    split_scope_hist, split_dist_hist = plot_first_last(fr_dist, px_dist, app_dist, err_app_dist)
    # PLOT histogram vs histogram distances correct/incorrect

    os.makedirs(output_dir, exist_ok=False)
    global_feats_hist.savefig(f'{output_dir}/01_global_feats_hist.png', dpi=300)
    global_range_bar.savefig(f'{output_dir}/02_global_range_bar.png', dpi=300)
    tracklet_dist_hist.savefig(f'{output_dir}/03_tracklet_dist_hist.png', dpi=300)
    split_scope_hist.savefig(f'{output_dir}/04_split_scope_hist.png', dpi=300)
    split_dist_hist.savefig(f'{output_dir}/05_split_dist_hist.png', dpi=300)
    merged_tracklets_df.astype(int).to_csv(f'{output_dir}/results.txt', index=False, header=False)
