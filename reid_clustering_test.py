
from docopt import docopt
from itertools import combinations
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.spatial import distance
import sys


def plot_global_feats(descriptors):

    desc = np.vstack(list(descriptors.values()))
    scale = np.sum(np.abs(desc), axis=0) / desc.shape[0]
    range_ = np.sort(np.max(desc, axis=0) - np.min(desc, axis=0))[::-1]

    fig1 = plt.figure()
    fig1.suptitle('Histogram of Global Features')
    ax = fig1.add_subplot()
    ax.set_title('hist( np.sum(np.abs(desc), axis=0) / desc.shape[0], log=True)')
    ax.hist(scale, log=True)

    fig2 = plt.figure()
    fig2.suptitle('Dynamic Range of Features (sorted and normalized)')
    ax = fig2.add_subplot()
    ax.set_title('np.max(desc, axis=0) - np.min(desc, axis=0)')
    ax.bar(np.arange(desc.shape[1]), range_ / np.max(range_))

    return fig1, fig2

def plot_distance_info(dist_matrix):
    valid = np.isfinite(dist_matrix)
    values = dist_matrix.flatten()[valid.flatten()]

    fig1 = plt.figure()
    fig1.suptitle('Histogram of Tracklet Distances')
    ax = fig1.add_subplot()
    ax.hist(values, bins=np.logspace(-5, 0, 100), log=True)
    ax.set_xscale('log')
    ax.set_xlabel('Distance')

    fig2 = plt.figure()
    fig2.suptitle('Cumulative Normalized Histogram of Tracklet Distances')
    ax = fig2.add_subplot()
    ax.hist(values, bins=np.logspace(-5, 0, 100), log=True, cumulative=True, density=True)
    ax.set_xscale('log')
    ax.set_xlabel('Distance')


    return fig1, fig2

def plot_first_last(first_vs_last, first_vs_mean, last_vs_mean, first_vs_worse, last_vs_worse, max_examples=50):

    x_pos = np.arange(min(len(first_vs_last), max_examples))
    worse_ok = np.max(np.vstack((first_vs_last, first_vs_mean, last_vs_mean)), axis=0)
    worse_nok = np.min(np.vstack((first_vs_worse, last_vs_worse)), axis=0)
    mask_ok = worse_ok > worse_nok

    fig1 = plt.figure(figsize=(6.4 * 2.5, 4.8 * 2))
    fig1.suptitle(f'Furthest Frames Distance Analysis ({len(first_vs_last)} tracks used)')

    ax = fig1.add_subplot(2, 2, 1)
    ax.set_title(f'Comparison of furthest frames within its tracklet ({len(x_pos)} examples)')
    ax.bar(x_pos, first_vs_last[:max_examples], width=0.3, label='d(first, last)')
    ax.bar(x_pos + 0.33, first_vs_mean[:max_examples], width=0.3, label='d(first, mean)')
    ax.bar(x_pos + 0.66, last_vs_mean[:max_examples], width=0.3, label='d(last, mean)')
    ax.set_xlabel('Track ID')
    ax.set_ylim(0, 1)
    ax.legend()
    
    ax = fig1.add_subplot(2, 2, 2)
    ax.set_title(f'Comparison of furthest frames with other tracklets ({len(x_pos)} examples)')
    ax.bar(x_pos, first_vs_last[:max_examples], width=0.3, label='d(first, last)')
    ax.bar(x_pos + 0.33, first_vs_worse[:max_examples], width=0.3, label='d(first, error)')
    ax.bar(x_pos + 0.66, last_vs_worse[:max_examples], width=0.3, label='d(last, error)')
    ax.set_xlabel('Track ID')
    ax.set_ylim(0, 1)
    ax.legend()

    ax = fig1.add_subplot(2, 2, 3)
    ax.set_title('Histogram of distances when there is error')
    ax.hist([worse_ok[mask_ok], worse_nok[mask_ok]], 100, label=['maximum intra distance', 'minimum inter distance'])
    ax.legend()
    ax.set_xlabel('Distance')
    ax.set_ylim(0, 20)

    ax = fig1.add_subplot(2, 2, 4)
    ax.set_title('Histogram of distances when there is no error')
    ax.hist([worse_ok[~mask_ok], worse_nok[~mask_ok]], 100, label=['maximum intra distance', 'minimum inter distance'])
    ax.legend()
    ax.set_xlabel('Distance')
    ax.set_ylim(0, 20)

    return fig1

def mean_descriptor(track_df):
    feats = np.array(track_df['feats'].values.tolist())
    conf = track_df['conf'].to_numpy()

    descriptor = (feats.T @ conf) / np.sum(conf)
    return descriptor

def compute_simultaneous(tracklets_df):

    # Filter alone tracks
    simultaneous = pd.DataFrame(tracklets_df.groupby('frameId')['trackIdx'].apply(list), columns=['trackIdx'])
    simultaneous = simultaneous[simultaneous['trackIdx'].apply(len) > 1]

    # Remove duplicated rows
    simultaneous['key'] = simultaneous['trackIdx'].apply(lambda x : str(sorted(x)))
    simultaneous.drop_duplicates(subset='key', inplace=True)

    # Generate pairs from simultaneous
    simultaneous['trackIdx'] = simultaneous['trackIdx'].apply(lambda x : list(combinations(x, 2)))
    simultaneous = simultaneous.explode('trackIdx')

    # Remove duplicated pairs
    simultaneous['key'] = simultaneous['trackIdx'].apply(lambda x : str(sorted(x)))
    simultaneous.drop_duplicates(subset='key', inplace=True)

    # Generate output
    simultaneous = pd.DataFrame(simultaneous['trackIdx'].tolist(), columns=['Idx1', 'Idx2'])
    idx1 = simultaneous['Idx1'].to_numpy()
    idx2 = simultaneous['Idx2'].to_numpy()

    return idx1, idx2

def compute_matrix(tracklets_df, descriptors_df, causality=True):
    descriptors_mat = descriptors_df.to_numpy().T

    dist_matrix = distance.cdist(descriptors_mat, descriptors_mat, metric='cosine') # 1 - cos(angle)

    if causality:
        # Apply anti-self-association and causality (a track can only be compared with a previous track)
        invalid_mask = np.full_like(dist_matrix, True, dtype=bool)
        invalid_mask = np.tril(invalid_mask)
    else:
        # We do not want to compare one track with itself (anti-self-association)
        invalid_mask = np.eye(dist_matrix.shape[0], dtype=bool)

    # 2 tracks that appers at the same time cannot be the same track
    idx1, idx2 = compute_simultaneous(tracklets_df)
    invalid_mask[idx1, idx2] = True
    invalid_mask[idx2, idx1] = True

    dist_matrix[invalid_mask] = np.inf

    return dist_matrix

def compute_merging(merging_matrix):
    valid = np.isfinite(merging_matrix.min(axis=0))
    paired = np.arange(len(valid))[valid]
    pair = merging_matrix.argmin(axis=0)[valid]

    merging_dict = dict()
    for ini, fin in zip(paired, pair):
        merging_dict[ini] = merging_dict.get(fin, fin)
    
    return merging_dict

def compare_first_with_last(tracklets_df, descriptors, min_group=4):
    # vs near tracklets

    desc = pd.DataFrame(descriptors).to_numpy().T

    # Filter groups of less than min_group
    simultaneous = pd.DataFrame(tracklets_df.groupby('frameId')['trackIdx'].apply(list), columns=['trackIdx'])
    simultaneous = simultaneous[simultaneous['trackIdx'].apply(len) > min_group]

    # Remove duplicated rows
    simultaneous['key'] = simultaneous['trackIdx'].apply(lambda x : str(sorted(x)))
    simultaneous.drop_duplicates(subset='key', inplace=True)

    first_vs_last = []
    first_vs_mean = []
    last_vs_mean = []

    first_vs_worse = []

    last_vs_worse = []

    seen = set()
    for trackIds in simultaneous['trackIdx']:
        for tck in trackIds:
            if tck in seen:
                continue
            seen.add(tck)

            first = np.array(tracklets_df[tracklets_df['trackIdx'] == tck].iloc[0]['feats'])
            last = np.array(tracklets_df[tracklets_df['trackIdx'] == tck].iloc[-1]['feats'])
            mean = descriptors[tck]

            first_vs_last.append(distance.cosine(first, last))
            first_vs_mean.append(distance.cosine(first, mean))
            last_vs_mean.append(distance.cosine(last, mean))

            first_vs_worse.append(distance.cdist(desc, first.reshape(1, -1)).flatten().min())
            last_vs_worse.append(distance.cdist(desc, last.reshape(1, -1)).flatten().min())
    
    return first_vs_last, first_vs_mean, last_vs_mean, first_vs_worse, last_vs_worse


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

    # READ SPLIT TRACKS
    seq_dets = np.loadtxt(tracking_file, delimiter=',', dtype=np.float64)
    tracklets_df = pd.DataFrame(seq_dets[:, :10], columns=['frameId', 'trackId', 'tlx', 'tly', 'width', 'height', 'conf','a','b', 'c'])
    feats = seq_dets[:, 10:]
    tracklets_df['feats'] = feats.tolist()

    # TODO: READ GROUND TRUTH

    # MAKE VALID PYTHON MATRIX INDEX
    trackIds = np.sort(np.unique(tracklets_df['trackId']))
    index = {tck_id : mat_id for mat_id, tck_id in enumerate(trackIds)}
    tracklets_df['trackIdx'] = tracklets_df['trackId'].apply(lambda x : index[x])

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

    # MAKE THE GROUND TRUTH IDs COINCIDE WITH THE FIRST PART OF THEIR SPLIT NEW ID: Problem joining errors!
    #gt_tracking_df['trackId'] = gt_tracking_file['trackId'].apply(lambda x : merging_dict.get(index[x], index[x]) + 1)

    # TODO: RANK 1 (for each merging_matrix, see if gt_tracking_df[tracklets_df['trackIdx'] == key]['trackId'].iloc[0])

    first_vs_last, first_vs_mean, last_vs_mean, first_vs_worse, last_vs_worse = compare_first_with_last(tracklets_df, descriptors, min_group=4)
    first_last_bar = plot_first_last(first_vs_last, first_vs_mean, last_vs_mean, first_vs_worse, last_vs_worse)

    global_feats_hist, global_range_bar = plot_global_feats(descriptors)
    tracklet_dist_hist, tracklet_dist_cum_hist = plot_distance_info(dist_matrix)

    os.makedirs(output_dir, exist_ok=False)
    global_feats_hist.savefig(f'{output_dir}/01_global_feats_hist.png', dpi=300)
    global_range_bar.savefig(f'{output_dir}/02_global_range_bar.png', dpi=300)
    tracklet_dist_hist.savefig(f'{output_dir}/03_tracklet_dist_hist.png', dpi=300)
    tracklet_dist_cum_hist.savefig(f'{output_dir}/04_tracklet_dist_cum_hist.png', dpi=300)
    first_last_bar.savefig(f'{output_dir}/05_first_last_bar.png', dpi=300)
    merged_tracklets_df.astype(int).to_csv(f'{output_dir}/results.txt', index=False, header=False)
